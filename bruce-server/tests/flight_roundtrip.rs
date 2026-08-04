//! In-process round-trip tests for the Arrow Flight surface
//! (FLIGHT-001): serve `BruceFlightService` on an ephemeral port in a
//! tokio task, connect with the arrow-flight client, and check the
//! answers against direct `Database::run` on an identical toy table.

use std::collections::HashMap;

use arrow::array::{Float64Array, StringArray};
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::utils::flight_data_to_batches;
use arrow_flight::{
    Action, Criteria, Empty, FlightData, FlightDescriptor, HandshakeRequest, Ticket,
};
use bruce_query::{Column, Database, Table};
use bruce_server::flight::{serve_with_shutdown, BruceFlightService};
use futures::StreamExt;
use ndarray::{Array1, Array2};
use tonic::transport::Channel;
use tonic::Code;

const SQL: &str = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.5) \
                   FROM movies WHERE year >= 2000 GROUP BY genre";
const Q: [f64; 4] = [1.0, 0.2, -0.3, 0.5];

/// Six-row toy movies table: 3 genres, one pre-2000 row so the
/// filter actually filters, d=4 keys.
fn toy_table() -> Table {
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: vec![0, 1, 0, 2, 1, 0],
            dict: vec!["drama".into(), "comedy".into(), "scifi".into()],
        },
    );
    t.columns.insert(
        "rating".into(),
        Column::ScalarF64(vec![7.5, 6.0, 8.2, 9.1, 5.5, 4.0]),
    );
    t.columns.insert(
        "year".into(),
        Column::ScalarF64(vec![2001.0, 2005.0, 1999.0, 2010.0, 2020.0, 2003.0]),
    );
    let emb = Array2::from_shape_vec(
        (6, 4),
        vec![
            0.9, 0.1, 0.0, 0.3, //
            0.2, 0.8, 0.1, 0.0, //
            0.7, 0.3, 0.2, 0.1, //
            0.0, 0.1, 0.9, 0.2, //
            0.4, 0.4, 0.4, 0.4, //
            0.6, 0.0, 0.1, 0.8, //
        ],
    )
    .unwrap();
    t.columns.insert("emb".into(), Column::KeyF64(emb));
    t
}

fn toy_db() -> Database {
    let mut db = Database::new();
    db.register("movies", toy_table());
    db
}

/// Serve a fresh toy database on 127.0.0.1:0; returns the client and
/// a shutdown sender (dropping it also stops the server).
async fn spawn_and_connect() -> (
    FlightServiceClient<Channel>,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let svc = BruceFlightService::new(toy_db());
    let server = tokio::spawn(async move {
        serve_with_shutdown(svc, listener, async {
            let _ = rx.await;
        })
        .await
        .expect("flight server crashed");
    });
    let client = FlightServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to in-process flight server");
    (client, tx, server)
}

fn ticket_json(sql: &str) -> Ticket {
    Ticket::new(serde_json::json!({ "sql": sql, "params": { "q": Q.to_vec() } }).to_string())
}

#[tokio::test]
async fn do_get_matches_direct_database_run() {
    // The reference answer: same table, same SQL, in-process.
    let mut direct = toy_db();
    let mut params = HashMap::new();
    params.insert("q".to_string(), Array1::from(Q.to_vec()));
    let (expect, planned) = direct.run(SQL, &params).expect("direct run");
    let expect_explain = planned.explain();
    assert!(!expect.labels.is_empty(), "toy query must cover groups");

    let (mut client, stop, server) = spawn_and_connect().await;
    let resp = client.do_get(ticket_json(SQL)).await.expect("do_get");
    let msgs: Vec<FlightData> = resp
        .into_inner()
        .map(|r| r.expect("stream item"))
        .collect()
        .await;
    assert_eq!(msgs.len(), 2, "schema message + one batch message");

    // EXPLAIN text rides the batch message's app_metadata.
    let meta: String = msgs
        .iter()
        .map(|m| String::from_utf8(m.app_metadata.to_vec()).unwrap())
        .collect();
    assert!(meta.contains("== chosen plan =="), "metadata: {meta}");
    assert_eq!(meta, expect_explain);

    let batches = flight_data_to_batches(&msgs).expect("decode batches");
    assert_eq!(batches.len(), 1);
    let b = &batches[0];
    assert_eq!(b.num_rows(), expect.labels.len());
    let labels: Vec<String> = b
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("label column is Utf8")
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();
    let values: Vec<f64> = b
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("value column is Float64")
        .values()
        .to_vec();
    assert_eq!(labels, expect.labels);
    for (i, (got, want)) in values.iter().zip(&expect.values).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "value[{i}]: got {got}, want {want}"
        );
    }

    stop.send(()).unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn malformed_tickets_are_invalid_argument_not_panic() {
    let (mut client, _stop, _server) = spawn_and_connect().await;

    // Not JSON at all.
    let err = client
        .do_get(Ticket::new("definitely not json"))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);

    // Not UTF-8.
    let err = client
        .do_get(Ticket::new(vec![0xff, 0xfe, 0x00, 0x9f]))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);

    // Valid JSON, wrong shape.
    let err = client
        .do_get(Ticket::new(r#"{"query": "SELECT 1"}"#))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);

    // Well-formed ticket, unknown table (bind error).
    let err = client
        .do_get(ticket_json(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.5) FROM nope GROUP BY genre",
        ))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);

    // Well-formed ticket, unbound parameter (bind error).
    let err = client
        .do_get(Ticket::new(
            serde_json::json!({ "sql": SQL, "params": {} }).to_string(),
        ))
        .await
        .expect_err("must reject");
    assert_eq!(err.code(), Code::InvalidArgument);

    // The server survived all of it and still answers correctly.
    let ok = client.do_get(ticket_json(SQL)).await.expect("still alive");
    let msgs: Vec<FlightData> = ok.into_inner().map(|r| r.unwrap()).collect().await;
    assert!(flight_data_to_batches(&msgs).unwrap()[0].num_rows() > 0);
}

#[tokio::test]
async fn list_flights_and_get_flight_info_name_tables() {
    let (mut client, _stop, _server) = spawn_and_connect().await;

    let infos: Vec<_> = client
        .list_flights(Criteria::default())
        .await
        .expect("list_flights")
        .into_inner()
        .map(|r| r.expect("info item"))
        .collect()
        .await;
    assert_eq!(infos.len(), 1);
    let desc = infos[0].flight_descriptor.as_ref().expect("descriptor");
    assert_eq!(desc.path, vec!["movies".to_string()]);
    assert_eq!(infos[0].total_records, 6);

    let info = client
        .get_flight_info(FlightDescriptor::new_path(vec!["movies".into()]))
        .await
        .expect("get_flight_info")
        .into_inner();
    assert_eq!(info.total_records, 6);

    let err = client
        .get_flight_info(FlightDescriptor::new_path(vec!["nope".into()]))
        .await
        .expect_err("unknown table");
    assert_eq!(err.code(), Code::NotFound);

    let err = client
        .get_flight_info(FlightDescriptor::new_path(vec![]))
        .await
        .expect_err("empty path");
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[tokio::test]
async fn unused_endpoints_return_unimplemented() {
    let (mut client, _stop, _server) = spawn_and_connect().await;

    let err = client
        .handshake(futures::stream::iter(vec![HandshakeRequest::default()]))
        .await
        .expect_err("handshake");
    assert_eq!(err.code(), Code::Unimplemented);

    let err = client
        .poll_flight_info(FlightDescriptor::new_path(vec!["movies".into()]))
        .await
        .expect_err("poll_flight_info");
    assert_eq!(err.code(), Code::Unimplemented);

    let err = client
        .get_schema(FlightDescriptor::new_path(vec!["movies".into()]))
        .await
        .expect_err("get_schema");
    assert_eq!(err.code(), Code::Unimplemented);

    let err = client
        .do_put(futures::stream::iter(vec![FlightData::default()]))
        .await
        .expect_err("do_put");
    assert_eq!(err.code(), Code::Unimplemented);

    let err = client
        .do_exchange(futures::stream::iter(vec![FlightData::default()]))
        .await
        .expect_err("do_exchange");
    assert_eq!(err.code(), Code::Unimplemented);

    let err = client
        .do_action(Action::new("whatever", ""))
        .await
        .expect_err("do_action");
    assert_eq!(err.code(), Code::Unimplemented);

    let err = client
        .list_actions(Empty::default())
        .await
        .expect_err("list_actions");
    assert_eq!(err.code(), Code::Unimplemented);
}

#[tokio::test]
async fn concurrent_do_gets_serialize_cleanly() {
    // The Database sits behind one Mutex; concurrent tickets must all
    // answer (serialized), none dropped or corrupted.
    let (client, _stop, _server) = spawn_and_connect().await;
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let mut c = client.clone();
        tasks.push(tokio::spawn(async move {
            let resp = c.do_get(ticket_json(SQL)).await.expect("do_get");
            let msgs: Vec<FlightData> = resp.into_inner().map(|r| r.unwrap()).collect().await;
            flight_data_to_batches(&msgs).unwrap()[0].num_rows()
        }));
    }
    for t in tasks {
        assert!(t.await.unwrap() > 0);
    }
}

//! Arrow Flight surface over the bruce-query `Database` (backlog
//! #16; ROUTES verdict: arrow-flight crate, auth later, FlightSQL
//! dialect later).
//!
//! v1 protocol — SQL-in-ticket:
//!
//! ```json
//! { "sql": "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.1) ...",
//!   "params": { "q": [0.1, 0.2, ...] } }
//! ```
//!
//! `do_get` executes the ticket's SQL against the server's one
//! `Database` and streams back a single RecordBatch with columns
//! `(label: Utf8, value: Float64)`; the batch message's
//! `app_metadata` carries the planner's EXPLAIN text, so every answer
//! travels with the plan that produced it. `list_flights` /
//! `get_flight_info` are minimal-viable: they name the registered
//! tables. Every other endpoint returns `Status::unimplemented` —
//! cleanly, never a panic.
//!
//! Concurrency: the `Database` sits behind one async Mutex.
//! `Database::run` needs `&mut self` (lazy stats refresh), so
//! queries serialize; at the current 8-14 ms/query that is the v1
//! contract, not a bottleneck. Multi-reader planning arrives with the
//! storage milestone, not here.

// tonic::Status IS the error type of the generated FlightService
// trait; boxing it would just move the size into every trait impl
// signature mismatch. tonic's own generated code carries the same
// allow.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::utils::batches_to_flight_data;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use bruce_query::{Database, QueryError};
use futures::stream::BoxStream;
use ndarray::Array1;
use serde::Deserialize;
use tonic::{Request, Response, Status, Streaming};

/// The v1 ticket payload: SQL plus named `:param` query vectors.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TicketRequest {
    /// The SQL text (bruce-query dialect: SOFTAVG/SIM).
    sql: String,
    /// Parameter name -> query vector, binding `:name` in the SQL.
    #[serde(default)]
    params: HashMap<String, Vec<f64>>,
}

/// Arrow Flight service backed by one `bruce_query::Database`.
#[derive(Clone)]
pub struct BruceFlightService {
    db: Arc<tokio::sync::Mutex<Database>>,
}

impl BruceFlightService {
    /// Wrap a loaded database. The service owns it for the process
    /// lifetime.
    pub fn new(db: Database) -> Self {
        BruceFlightService {
            db: Arc::new(tokio::sync::Mutex::new(db)),
        }
    }

    /// The tonic server wrapper, ready for `Server::add_service`.
    pub fn into_server(self) -> FlightServiceServer<Self> {
        FlightServiceServer::new(self)
    }
}

/// The one result schema every do_get answer uses.
fn result_schema() -> Schema {
    Schema::new(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// PG-aligned status mapping (C4): parse/bind failures are the
/// client's fault (invalid_argument ~ syntax_error/undefined_table);
/// execution failures are ours (internal).
fn query_error_status(e: QueryError) -> Status {
    match &e {
        QueryError::Parse(_) | QueryError::Bind(_) => Status::invalid_argument(e.to_string()),
        QueryError::Exec(_) => Status::internal(e.to_string()),
    }
}

/// Minimal FlightInfo for one registered table.
fn table_flight_info(name: &str, rows: usize) -> Result<FlightInfo, Status> {
    // Advertised schema is the do_get RESULT shape — every ticket
    // against this server yields (label, value).
    FlightInfo::new()
        .try_with_schema(&result_schema())
        .map_err(|e| Status::internal(format!("encode schema: {e}")))
        .map(|info| {
            info.with_descriptor(FlightDescriptor::new_path(vec![name.to_string()]))
                .with_endpoint(FlightEndpoint::new())
                .with_total_records(rows as i64)
                .with_total_bytes(-1)
                .with_ordered(false)
        })
}

#[tonic::async_trait]
impl FlightService for BruceFlightService {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let text = std::str::from_utf8(&ticket.ticket)
            .map_err(|e| Status::invalid_argument(format!("ticket is not UTF-8: {e}")))?;
        let req: TicketRequest = serde_json::from_str(text).map_err(|e| {
            Status::invalid_argument(format!(
                "ticket is not the expected JSON {{\"sql\": ..., \"params\": {{...}}}}: {e}"
            ))
        })?;
        let params: HashMap<String, Array1<f64>> = req
            .params
            .into_iter()
            .map(|(k, v)| (k, Array1::from(v)))
            .collect();

        let (result, planned) = {
            let mut db = self.db.lock().await;
            db.run(&req.sql, &params).map_err(query_error_status)?
        };
        let explain = planned.explain();

        let schema = result_schema();
        let labels: ArrayRef = Arc::new(StringArray::from(result.labels));
        let values: ArrayRef = Arc::new(Float64Array::from(result.values));
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![labels, values])
            .map_err(|e| Status::internal(format!("build result batch: {e}")))?;
        let mut msgs = batches_to_flight_data(&schema, vec![batch])
            .map_err(|e| Status::internal(format!("encode flight data: {e}")))?;
        // msgs = [schema message, batch message]; the EXPLAIN text
        // rides the batch message's app_metadata.
        if let Some(last) = msgs.last_mut() {
            last.app_metadata = explain.into_bytes().into();
        }
        Ok(Response::new(Box::pin(futures::stream::iter(
            msgs.into_iter().map(Ok),
        ))))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let db = self.db.lock().await;
        let mut names: Vec<&String> = db.catalog.tables.keys().collect();
        names.sort();
        let infos: Vec<Result<FlightInfo, Status>> = names
            .into_iter()
            .map(|n| {
                let rows = db.catalog.tables[n]
                    .columns
                    .values()
                    .next()
                    .map(|c| c.len())
                    .unwrap_or(0);
                table_flight_info(n, rows)
            })
            .collect();
        Ok(Response::new(Box::pin(futures::stream::iter(infos))))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let desc = request.into_inner();
        let name = match desc.path.as_slice() {
            [one] => one.clone(),
            _ => {
                return Err(Status::invalid_argument(
                    "descriptor path must be exactly one registered table name",
                ))
            }
        };
        let db = self.db.lock().await;
        let table = db
            .catalog
            .tables
            .get(&name)
            .ok_or_else(|| Status::not_found(format!("no table {name:?}")))?;
        let rows = table.columns.values().next().map(|c| c.len()).unwrap_or(0);
        Ok(Response::new(table_flight_info(&name, rows)?))
    }

    // ---- everything below: clean unimplemented, never a panic ----

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented(
            "handshake: auth arrives later (ROUTES)",
        ))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented(
            "do_put: the write path is QuerySession/CDC for now",
        ))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions"))
    }
}

/// Serve the Flight service on an already-bound listener until
/// `shutdown` resolves, then drain gracefully.
///
/// Binding is the caller's job so that `127.0.0.1:0` (ephemeral
/// port) works: bind, read `local_addr()`, then serve. Used by the
/// `bruce-flight-server` binary and, in-process, by the integration
/// tests.
pub async fn serve_with_shutdown(
    svc: BruceFlightService,
    listener: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()>,
) -> Result<(), tonic::transport::Error> {
    use futures::StreamExt as _;
    // TCP_NODELAY must be set here: tonic applies its own
    // `tcp_nodelay(true)` only on the serve(addr) path, NOT on
    // serve_with_incoming. Without it, the schema message and the
    // batch message land in separate small writes and Nagle +
    // delayed-ACK turns every do_get into a ~40 ms stall (measured;
    // see paper_sigmod_bruce/experiments/m16_flight/).
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener).map(|res| {
        res.inspect(|stream| {
            let _ = stream.set_nodelay(true);
        })
    });
    tonic::transport::Server::builder()
        .add_service(svc.into_server())
        .serve_with_incoming_shutdown(incoming, shutdown)
        .await
}

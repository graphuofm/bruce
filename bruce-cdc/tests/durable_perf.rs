//! Durable-mirror measurements (workstream 12 v2), medians of >= 5:
//!   - recovery time: Mirror::load from disk + walsender reconnect +
//!     catch-up of 1000 pending single-row txs written while the
//!     subscriber was down;
//!   - snapshot write throughput: Mirror::save of a 100k-row mirror.
//!
//! Numbers land in the JSON file named by $BRUCE_CDC_DURABLE_RESULTS
//! (nothing is written when unset); the m12_cdc results.json is
//! updated from that file — never hardcoded downstream.
//!
//! Needs the live PG on /tmp:54329; owns cdc_movies_perf +
//! bruce_perf_pub + bruce_cdc_perf.

use std::time::{Duration, Instant};

use postgres::{Client, NoTls};

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::source::{ChangeSource, PgOutputSource, SlotSetup, SourceConfig};

const TABLE: &str = "cdc_movies_perf";
const PUB: &str = "bruce_perf_pub";
const SLOT: &str = "bruce_cdc_perf";
const SEED_ROWS: i32 = 1000;
const PENDING_ROWS: i32 = 1000;
const ROUNDS: usize = 5;
const SNAPSHOT_ROWS: usize = 100_000;

fn control() -> Client {
    let user = std::env::var("USER").unwrap_or_else(|_| "jding".into());
    Client::connect(
        &format!("host=/tmp port=54329 user={user} dbname=postgres"),
        NoTls,
    )
    .expect("PG must be running on /tmp:54329")
}

fn drop_slot(pg: &mut Client, slot: &str) {
    for _ in 0..100 {
        let _ = pg.execute(
            "SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots \
             WHERE slot_name = $1 AND active = false",
            &[&slot],
        );
        let gone = pg
            .query_one(
                "SELECT count(*) FROM pg_replication_slots WHERE slot_name = $1",
                &[&slot],
            )
            .map(|row| row.get::<_, i64>(0) == 0)
            .unwrap_or(false);
        if gone {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("slot {slot} still active after 10s");
}

fn table_map() -> TableMap {
    TableMap {
        table: TABLE.into(),
        pk: "movie_id".into(),
        label_cols: vec!["genre".into()],
        scalar_cols: vec![
            "movie_id".into(),
            "rating".into(),
            "year".into(),
            "e0".into(),
            "e1".into(),
        ],
        key_col: "emb".into(),
        key_parts: vec!["e0".into(), "e1".into()],
    }
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn durable_perf_medians() {
    // ---------------- snapshot write throughput (no PG) ----------------
    let names: Vec<String> = ["movie_id", "genre", "rating", "year", "e0", "e1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<Vec<Option<String>>> = (0..SNAPSHOT_ROWS)
        .map(|i| {
            let theta = (i % 6283) as f64 / 1000.0;
            vec![
                Some(i.to_string()),
                Some(format!("g{}", i % 32)),
                Some(format!("{:.2}", 1.0 + (i % 900) as f64 / 100.0)),
                Some("2000".into()),
                Some(theta.cos().to_string()),
                Some(theta.sin().to_string()),
            ]
        })
        .collect();
    let big = Mirror::from_snapshot(table_map(), &names, &rows).unwrap();
    let snap_path = std::env::temp_dir().join(format!(
        "bruce_cdc_perf_snapshot_{}.mirror",
        std::process::id()
    ));
    big.save(&snap_path).unwrap(); // warm-up (not measured)
    let file_bytes = std::fs::metadata(&snap_path).unwrap().len();
    let mut save_ms = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        big.save(&snap_path).unwrap();
        save_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    // and verify the artifact still loads to the same row count
    assert_eq!(Mirror::load(&snap_path).unwrap().n_rows(), SNAPSHOT_ROWS);
    std::fs::remove_file(&snap_path).unwrap();
    let save_med = median_ms(save_ms.clone());
    let mb_per_s = (file_bytes as f64 / (1024.0 * 1024.0)) / (save_med / 1e3);

    // ---------------- recovery time (live PG) ----------------
    let mut pg = control();
    let state = std::env::temp_dir().join(format!(
        "bruce_cdc_perf_recovery_{}.mirror",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&state);
    drop_slot(&mut pg, SLOT);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB};
         DROP TABLE IF EXISTS {TABLE};
         CREATE TABLE {TABLE}(
           movie_id int primary key, genre text, rating float8,
           year float8, e0 float8, e1 float8);
         ALTER TABLE {TABLE} REPLICA IDENTITY FULL;
         CREATE PUBLICATION {PUB} FOR TABLE {TABLE};
         INSERT INTO {TABLE}
           SELECT i, 'g' || (i % 8), 1.0 + (i % 90)::float8 / 10, 2000,
                  cos(i::float8), sin(i::float8)
           FROM generate_series(1, {SEED_ROWS}) i;"
    ))
    .unwrap();

    let cfg = || {
        let mut c = SourceConfig::local_default(SLOT);
        c.publication = PUB.into();
        c
    };
    {
        let mut src = PgOutputSource::connect(cfg()).unwrap();
        let mirror = match src.create_slot_with_snapshot().unwrap() {
            SlotSetup::CreatedSnapshotOpen { .. } => {
                let (cols, rows) = src
                    .snapshot_query(&format!(
                        "SELECT movie_id, genre, rating, year, e0, e1 FROM {TABLE}"
                    ))
                    .unwrap();
                src.commit_snapshot().unwrap();
                Mirror::from_snapshot(table_map(), &cols, &rows).unwrap()
            }
            SlotSetup::AlreadyExists => panic!("slot {SLOT} must be fresh"),
        };
        assert_eq!(mirror.n_rows(), SEED_ROWS as usize);
        mirror.save(&state).unwrap();
        drop(src);
    }

    let mut recovery_ms = Vec::with_capacity(ROUNDS);
    let mut next_id = SEED_ROWS;
    for round in 0..ROUNDS {
        // subscriber is DOWN; 1000 single-row txs pile up in the slot
        for _ in 0..PENDING_ROWS {
            next_id += 1;
            pg.execute(
                &format!(
                    "INSERT INTO {TABLE} \
                     SELECT $1, 'g' || ($1 % 8), 1.0 + ($1 % 90)::float8 / 10, 2000, \
                            cos($1::float8), sin($1::float8)"
                ),
                &[&next_id],
            )
            .unwrap();
        }
        // recovery = open durable state + reconnect + catch up
        let t0 = Instant::now();
        let mut mirror = Mirror::load(&state).unwrap();
        let before = mirror.rows_applied;
        let mut src = PgOutputSource::connect(cfg()).unwrap();
        match src.create_slot_with_snapshot().unwrap() {
            SlotSetup::AlreadyExists => {}
            SlotSetup::CreatedSnapshotOpen { .. } => panic!("slot vanished"),
        }
        src.start().unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut last_lsn = 0;
        while mirror.rows_applied - before < PENDING_ROWS as usize {
            assert!(Instant::now() < deadline, "round {round}: catch-up stalled");
            if let Some(tx) = src.next_tx(Duration::from_millis(200)).unwrap() {
                mirror.apply_tx(&tx).unwrap();
                last_lsn = tx.end_lsn;
            }
        }
        recovery_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        // persist + ack OUTSIDE the measured window, then go down again
        mirror.save(&state).unwrap();
        src.ack(last_lsn).unwrap();
        assert_eq!(
            mirror.n_rows(),
            (SEED_ROWS + (round as i32 + 1) * PENDING_ROWS) as usize
        );
        drop(src);
    }
    let recov_med = median_ms(recovery_ms.clone());

    // ---------------- report ----------------
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let loadavg = loadavg
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "durable perf: recovery(load+reconnect+catchup {PENDING_ROWS} rows) median \
         {recov_med:.1} ms (all {recovery_ms:.1?}); snapshot save of {SNAPSHOT_ROWS} rows \
         ({file_bytes} bytes) median {save_med:.1} ms = {mb_per_s:.0} MB/s \
         (all {save_ms:.1?}); loadavg {loadavg}"
    );
    if let Ok(path) = std::env::var("BRUCE_CDC_DURABLE_RESULTS") {
        let date = String::from_utf8(
            std::process::Command::new("date")
                .arg("+%Y-%m-%d")
                .output()
                .map(|o| o.stdout)
                .unwrap_or_default(),
        )
        .unwrap_or_default()
        .trim()
        .to_string();
        let r_all: Vec<String> = recovery_ms.iter().map(|v| format!("{v:.1}")).collect();
        let s_all: Vec<String> = save_ms.iter().map(|v| format!("{v:.2}")).collect();
        let json = format!(
            "{{\n  \"date\": \"{date}\",\n  \"loadavg\": \"{loadavg}\",\n  \"recovery\": {{\n    \"pending_rows\": {PENDING_ROWS},\n    \"seed_rows\": {SEED_ROWS},\n    \"median_ms\": {recov_med:.1},\n    \"all_ms\": [{}]\n  }},\n  \"snapshot_write\": {{\n    \"rows\": {SNAPSHOT_ROWS},\n    \"file_bytes\": {file_bytes},\n    \"median_ms\": {save_med:.2},\n    \"mb_per_s\": {mb_per_s:.1},\n    \"all_ms\": [{}]\n  }}\n}}\n",
            r_all.join(", "),
            s_all.join(", ")
        );
        std::fs::write(&path, json).unwrap();
        eprintln!("durable perf results written to {path}");
    }

    // ---- leave PG healthy ----
    drop_slot(&mut pg, SLOT);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB}; DROP TABLE IF EXISTS {TABLE};"
    ))
    .unwrap();
    let _ = std::fs::remove_file(&state);
}

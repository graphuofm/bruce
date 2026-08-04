//! End-to-end proof of the sidecar form against the live local PG
//! instance (unix socket /tmp, port 54329): seed -> consistent
//! snapshot -> real START_REPLICATION stream -> insert_row /
//! delete_where -> maintained SOFTAVG answers equal (a) a
//! from-scratch bruce recomputation on the final PG state and (b) the
//! same soft average computed by PG itself in SQL. Then a
//! disconnect/reconnect proves slot-based resume.
//!
//! Requires the PG instance to be up with wal_level=logical and the
//! bruce_pub publication's table creatable — the test owns
//! cdc_movies and the bruce_cdc_e2e slot, and leaves PG healthy.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::Array1;
use postgres::{Client, NoTls};

use bruce_cdc::apply::{result_map, Mirror, TableMap};
use bruce_cdc::protocol::now_pg_us;
use bruce_cdc::source::{ChangeSource, PgOutputSource, SlotSetup, SourceConfig};

const SLOT: &str = "bruce_cdc_e2e";
const EPS: f64 = 0.1;
const QX: f64 = 0.6;
const QY: f64 = 0.8;
const N_SEED: i32 = 1000;
const N_INS: i32 = 100;
const N_DEL: i32 = 50;
const REL_TOL: f64 = 1e-9;

const GENRES: [&str; 8] = [
    "action",
    "drama",
    "comedy",
    "horror",
    "scifi",
    "romance",
    "thriller",
    "documentary",
];

fn control() -> Client {
    let user = std::env::var("USER").unwrap_or_else(|_| "jding".into());
    Client::connect(
        &format!("host=/tmp port=54329 user={user} dbname=postgres"),
        NoTls,
    )
    .expect("PG must be running on /tmp:54329 (see bruce-cdc/README.md)")
}

/// Deterministic row content for id `i` (shared by seed and stream
/// phases so the test is reproducible).
fn row(i: i32) -> (&'static str, f64, f64, f64, f64) {
    let mut h = (i as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h ^= h >> 33;
    let genre = GENRES[(h % 8) as usize];
    let rating = 1.0 + ((h >> 8) % 900) as f64 / 100.0;
    let year = 1980.0 + ((h >> 24) % 45) as f64;
    let theta = ((h >> 16) % 6283) as f64 / 1000.0;
    (genre, rating, year, theta.cos(), theta.sin())
}

fn insert_one(pg: &mut Client, id: i32) {
    let (g, r, y, e0, e1) = row(id);
    pg.execute(
        "INSERT INTO cdc_movies VALUES ($1,$2,$3,$4,$5,$6)",
        &[&id, &g, &r, &y, &e0, &e1],
    )
    .unwrap();
}

fn softavg_sql() -> String {
    format!("SELECT genre, SOFTAVG(rating, SIM(emb, :q), {EPS}) FROM cdc_movies GROUP BY genre")
}

fn params() -> HashMap<String, Array1<f64>> {
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from_vec(vec![QX, QY]));
    p
}

/// From-scratch ground truth 1: fresh snapshot of the CURRENT PG
/// state into a new Database (no views), same SOFTAVG SQL.
fn fresh_bruce_answer(pg: &mut Client) -> HashMap<String, f64> {
    let rows = pg
        .query(
            "SELECT movie_id, genre, rating, year, e0, e1 FROM cdc_movies",
            &[],
        )
        .unwrap();
    let col_names: Vec<String> = ["movie_id", "genre", "rating", "year", "e0", "e1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let text_rows: Vec<Vec<Option<String>>> = rows
        .iter()
        .map(|r| {
            vec![
                Some(r.get::<_, i32>(0).to_string()),
                Some(r.get::<_, String>(1)),
                Some(r.get::<_, f64>(2).to_string()),
                Some(r.get::<_, f64>(3).to_string()),
                Some(r.get::<_, f64>(4).to_string()),
                Some(r.get::<_, f64>(5).to_string()),
            ]
        })
        .collect();
    let mut fresh = Mirror::from_snapshot(TableMap::cdc_movies(), &col_names, &text_rows).unwrap();
    let (res, _) = fresh.db.run(&softavg_sql(), &params()).unwrap();
    result_map(&res.labels, &res.values)
}

/// From-scratch ground truth 2: PG computes the soft average itself.
fn pg_sql_answer(pg: &mut Client) -> HashMap<String, f64> {
    let rows = pg
        .query(
            "SELECT genre, SUM(rating * EXP((e0*$1 + e1*$2)/$3)) / SUM(EXP((e0*$1 + e1*$2)/$3)) \
             FROM cdc_movies GROUP BY genre",
            &[&QX, &QY, &EPS],
        )
        .unwrap();
    rows.iter()
        .map(|r| (r.get::<_, String>(0), r.get::<_, f64>(1)))
        .collect()
}

fn assert_close(got: &HashMap<String, f64>, want: &HashMap<String, f64>, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: group count");
    let mut max_rel = 0.0f64;
    for (label, w) in want {
        let g = got
            .get(label)
            .unwrap_or_else(|| panic!("{what}: missing group {label}"));
        let rel = (g - w).abs() / w.abs().max(1e-300);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= REL_TOL,
            "{what}: group {label}: got {g}, want {w} (rel {rel:.3e})"
        );
    }
    max_rel
}

fn drop_slot(pg: &mut Client) {
    // the walsender holds the slot until its exit is noticed; retry
    for _ in 0..50 {
        let r = pg.execute(
            "SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots \
             WHERE slot_name = $1 AND active = false",
            &[&SLOT],
        );
        let gone = pg
            .query_one(
                "SELECT count(*) FROM pg_replication_slots WHERE slot_name = $1",
                &[&SLOT],
            )
            .map(|row| row.get::<_, i64>(0) == 0)
            .unwrap_or(false);
        if r.is_ok() && gone {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("slot {SLOT} still active after 5s");
}

#[test]
fn cdc_end_to_end() {
    let mut pg = control();

    // ---- reset the artefacts this test owns ----
    drop_slot(&mut pg);
    pg.batch_execute(
        "DROP TABLE IF EXISTS cdc_movies;
         CREATE TABLE cdc_movies(
           movie_id int primary key, genre text, rating float8,
           year float8, e0 float8, e1 float8);
         ALTER TABLE cdc_movies REPLICA IDENTITY FULL;
         DROP PUBLICATION IF EXISTS bruce_pub;
         CREATE PUBLICATION bruce_pub FOR TABLE cdc_movies;",
    )
    .unwrap();

    // ---- seed (before the slot: arrives via snapshot, not stream) ----
    {
        let mut tx = pg.transaction().unwrap();
        for i in 1..=N_SEED {
            let (g, r, y, e0, e1) = row(i);
            tx.execute(
                "INSERT INTO cdc_movies VALUES ($1,$2,$3,$4,$5,$6)",
                &[&i, &g, &r, &y, &e0, &e1],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }

    // ---- slot + consistent snapshot on the SAME connection ----
    let mut src = PgOutputSource::connect(SourceConfig::local_default(SLOT)).unwrap();
    let mut mirror = match src.create_slot_with_snapshot().unwrap() {
        SlotSetup::CreatedSnapshotOpen { .. } => {
            let (cols, rows) = src
                .snapshot_query("SELECT movie_id, genre, rating, year, e0, e1 FROM cdc_movies")
                .unwrap();
            src.commit_snapshot().unwrap();
            Mirror::from_snapshot(TableMap::cdc_movies(), &cols, &rows).unwrap()
        }
        SlotSetup::AlreadyExists => panic!("stale slot {SLOT} survived drop_slot"),
    };
    assert_eq!(mirror.n_rows(), N_SEED as usize);

    let x = Array1::from_vec(vec![QX, QY]);
    mirror
        .db
        .create_view("v_genre", "cdc_movies", "genre", "rating", "emb", &x, EPS)
        .unwrap();

    src.start().unwrap();

    // ---- the workload: 100 single-row INSERT txs + 50 DELETE txs,
    // issued by a writer thread WHILE the main thread applies the
    // stream, so commit->applied lag measures the live pipeline and
    // not the test's own write phase ----
    let writer = std::thread::spawn(|| {
        let mut pg = control();
        let start = Instant::now();
        for i in (N_SEED + 1)..=(N_SEED + N_INS) {
            insert_one(&mut pg, i);
        }
        for k in 1..=N_DEL {
            pg.execute("DELETE FROM cdc_movies WHERE movie_id = $1", &[&(k * 20)])
                .unwrap();
        }
        start.elapsed()
    });

    let want_rows = (N_INS + N_DEL) as usize;
    let mut lags_us: Vec<i64> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    let apply_start = Instant::now();
    while mirror.rows_applied < want_rows {
        assert!(
            Instant::now() < deadline,
            "stream fell behind: {} of {want_rows} row changes",
            mirror.rows_applied
        );
        if let Some(tx) = src.next_tx(Duration::from_millis(500)).unwrap() {
            let n = mirror.apply_tx(&tx).unwrap();
            src.ack(tx.end_lsn).unwrap();
            if n > 0 {
                lags_us.push(now_pg_us() - tx.commit_ts_us);
            }
        }
    }
    let catchup_wall = apply_start.elapsed();
    let write_wall = writer.join().expect("writer thread");
    assert_eq!(mirror.n_rows(), (N_SEED + N_INS - N_DEL) as usize);

    // ---- freshness: mirror == both from-scratch ground truths ----
    let (res, planned) = mirror.db.run(&softavg_sql(), &params()).unwrap();
    let mirror_ans = result_map(&res.labels, &res.values);
    let view_scan = matches!(
        planned.chosen,
        bruce_query::PhysicalPlan::MaintainedViewScan { .. }
    );

    let fresh = fresh_bruce_answer(&mut pg);
    let pg_ans = pg_sql_answer(&mut pg);
    assert_eq!(mirror_ans.len(), GENRES.len());
    let rel_fresh = assert_close(&mirror_ans, &fresh, "mirror vs fresh bruce snapshot");
    let rel_pg = assert_close(&mirror_ans, &pg_ans, "mirror vs PG SQL softavg");

    // the maintained view must agree with the table it mirrors, too
    let view = &mirror.db.views[0];
    let dict = match mirror.db.catalog.tables["cdc_movies"].columns.get("genre") {
        Some(bruce_query::Column::DictU32 { dict, .. }) => dict.clone(),
        _ => panic!("genre must be dictionary-encoded"),
    };
    let view_ans: HashMap<String, f64> = view
        .read()
        .into_iter()
        .map(|(g, v)| (dict[g].clone(), v))
        .collect();
    let rel_view = assert_close(&view_ans, &pg_ans, "maintained view vs PG SQL softavg");

    // ---- resume: drop the connection, write more, reconnect ----
    drop(src);
    for i in 2001..=2010 {
        insert_one(&mut pg, i);
    }
    let mut src2 = PgOutputSource::connect(SourceConfig::local_default(SLOT)).unwrap();
    match src2.create_slot_with_snapshot().unwrap() {
        SlotSetup::AlreadyExists => {}
        SlotSetup::CreatedSnapshotOpen { .. } => panic!("slot vanished across reconnect"),
    }
    src2.start().unwrap(); // 0/0 = resume from confirmed_flush_lsn
    let before = mirror.rows_applied;
    let deadline = Instant::now() + Duration::from_secs(60);
    while mirror.rows_applied < before + 10 {
        assert!(Instant::now() < deadline, "resume stream fell behind");
        if let Some(tx) = src2.next_tx(Duration::from_millis(500)).unwrap() {
            mirror.apply_tx(&tx).unwrap();
            src2.ack(tx.end_lsn).unwrap();
        }
    }
    assert_eq!(mirror.n_rows(), (N_SEED + N_INS - N_DEL + 10) as usize);
    let (res2, _) = mirror.db.run(&softavg_sql(), &params()).unwrap();
    let mirror_ans2 = result_map(&res2.labels, &res2.values);
    let pg_ans2 = pg_sql_answer(&mut pg);
    let rel_resume = assert_close(&mirror_ans2, &pg_ans2, "post-resume mirror vs PG SQL");

    // ---- lag stats + optional results JSON ----
    lags_us.sort_unstable();
    let n = lags_us.len();
    let mean_us = lags_us.iter().sum::<i64>() as f64 / n as f64;
    let p50_us = lags_us[n / 2];
    let max_us = *lags_us.last().unwrap();
    eprintln!(
        "cdc e2e: {} txs / {} row changes applied; commit->applied lag mean {:.0} us, p50 {} us, max {} us; \
         writes {} ms, catch-up {} ms; plan used maintained view: {view_scan}; \
         max rel err: fresh {rel_fresh:.2e}, pg-sql {rel_pg:.2e}, view {rel_view:.2e}, resume {rel_resume:.2e}",
        mirror.txs_applied, mirror.rows_applied, mean_us, p50_us, max_us,
        write_wall.as_millis(), catchup_wall.as_millis()
    );

    if let Ok(path) = std::env::var("BRUCE_CDC_RESULTS") {
        let date = chrono_date();
        let txs = mirror.txs_applied;
        let rows_applied = mirror.rows_applied;
        let final_rows = mirror.n_rows();
        let write_ms = write_wall.as_millis();
        let catchup_ms = catchup_wall.as_millis();
        let groups = mirror_ans.len();
        let json = format!(
            "{{\n  \"date\": \"{date}\",\n  \"rows_seeded\": {N_SEED},\n  \"inserts_streamed\": {N_INS},\n  \"deletes_streamed\": {N_DEL},\n  \"resume_inserts\": 10,\n  \"txs_applied\": {txs},\n  \"row_changes_applied\": {rows_applied},\n  \"final_rows\": {final_rows},\n  \"lag_us\": {{\"mean\": {mean_us:.1}, \"p50\": {p50_us}, \"max\": {max_us}, \"n\": {n}}},\n  \"write_wall_ms\": {write_ms},\n  \"catchup_wall_ms\": {catchup_ms},\n  \"plan_used_maintained_view\": {view_scan},\n  \"groups_verified\": {groups},\n  \"max_rel_err\": {{\"vs_fresh_bruce\": {rel_fresh:.3e}, \"vs_pg_sql\": {rel_pg:.3e}, \"view_vs_pg_sql\": {rel_view:.3e}, \"post_resume_vs_pg_sql\": {rel_resume:.3e}}},\n  \"tolerance_rel\": {REL_TOL:.0e},\n  \"eps\": {EPS},\n  \"query\": [{QX}, {QY}]\n}}\n"
        );
        std::fs::write(&path, json).unwrap();
        eprintln!("results written to {path}");
    }

    // ---- leave PG healthy: close the stream, drop the test slot ----
    drop(src2);
    drop_slot(&mut pg);
}

/// Today's date without a chrono dependency.
fn chrono_date() -> String {
    String::from_utf8(
        std::process::Command::new("date")
            .arg("+%Y-%m-%d")
            .output()
            .map(|o| o.stdout)
            .unwrap_or_default(),
    )
    .unwrap_or_default()
    .trim()
    .to_string()
}

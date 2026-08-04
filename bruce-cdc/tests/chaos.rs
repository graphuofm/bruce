//! Workstream 12 — chaos: kill-resume x5, PG restart mid-stream,
//! large-transaction atomicity. Needs the live PG instance
//! (unix socket /tmp, port 54329, wal_level=logical). Each test owns
//! its table + publication + slot, serializes on a process-wide lock
//! (the restart test bounces the shared server), and cleans up —
//! `cdc_movies`, `movies` and `bruce_pub` are never touched.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use postgres::{Client, NoTls};

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::source::{
    is_transient, ChangeSource, CommittedTx, PgOutputSource, RetryingSource, RowChange, SlotSetup,
    SourceConfig,
};
use bruce_cdc::CdcError;

/// The three tests share one PG instance and one of them restarts it:
/// strictly one at a time.
static PG_LOCK: Mutex<()> = Mutex::new(());

fn pg_serial() -> MutexGuard<'static, ()> {
    PG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn pgv_bin(tool: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/jding".into());
    format!("{home}/miniforge3/envs/pgv/bin/{tool}")
}

fn pgdata() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/jding".into());
    format!("{home}/bruce/experiments/cidr_one_query/pgdata")
}

/// The instance's canonical startup options (postmaster.opts may lack
/// them if the server was last started by hand) and a real log file —
/// the restarted postmaster must NOT inherit this test's stdio pipes,
/// which close when the harness exits.
const PG_OPTS: &str = "-p 54329 -k /tmp -c shared_buffers=4GB -c max_parallel_workers_per_gather=8";

fn pg_restart_cmd() -> std::process::Command {
    let mut c = std::process::Command::new(pgv_bin("pg_ctl"));
    c.args([
        "-D",
        &pgdata(),
        "-m",
        "fast",
        "-w",
        "-l",
        &format!("{}/server.log", pgdata()),
        "-o",
        PG_OPTS,
        "restart",
    ]);
    c
}

fn control() -> Client {
    control_retry(Duration::from_secs(30))
}

/// Control-plane connection with retries (PG may be mid-restart).
fn control_retry(budget: Duration) -> Client {
    let user = std::env::var("USER").unwrap_or_else(|_| "jding".into());
    let conninfo = format!("host=/tmp port=54329 user={user} dbname=postgres");
    let deadline = Instant::now() + budget;
    loop {
        match Client::connect(&conninfo, NoTls) {
            Ok(c) => return c,
            Err(e) => {
                assert!(
                    Instant::now() < deadline,
                    "PG not reachable on /tmp:54329 within {budget:?}: {e}"
                );
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

/// Deterministic splitmix64 — the tests' only randomness source.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn table_map(table: &str) -> TableMap {
    TableMap {
        table: table.into(),
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

fn cfg(slot: &str, publication: &str) -> SourceConfig {
    let mut c = SourceConfig::local_default(slot);
    c.publication = publication.into();
    c
}

/// Reset table + publication + slot owned by one test.
fn reset_artefacts(pg: &mut Client, table: &str, publication: &str, slot: &str) {
    drop_slot(pg, slot);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {publication};
         DROP TABLE IF EXISTS {table};
         CREATE TABLE {table}(
           movie_id int primary key, genre text, rating float8,
           year float8, e0 float8, e1 float8);
         ALTER TABLE {table} REPLICA IDENTITY FULL;
         CREATE PUBLICATION {publication} FOR TABLE {table};",
    ))
    .unwrap();
}

fn cleanup(pg: &mut Client, table: &str, publication: &str, slot: &str) {
    drop_slot(pg, slot);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {publication}; DROP TABLE IF EXISTS {table};"
    ))
    .unwrap();
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

fn insert_sql(table: &str) -> String {
    format!("INSERT INTO {table} VALUES ($1,$2,$3,$4,$5,$6)")
}

fn insert_one(pg: &mut Client, table: &str, id: i32, rng: &mut u64) {
    let h = mix(rng);
    let genre = ["action", "drama", "comedy", "horror"][(h % 4) as usize];
    let rating = 1.0 + ((h >> 8) % 900) as f64 / 100.0;
    let theta = ((h >> 16) % 6283) as f64 / 1000.0;
    pg.execute(
        &insert_sql(table),
        &[&id, &genre, &rating, &2000.0f64, &theta.cos(), &theta.sin()],
    )
    .unwrap();
}

/// PG-side ground truth: (row count, FNV-1a checksum of (id, rating
/// bits) sorted by id).
fn pg_truth(pg: &mut Client, table: &str) -> (usize, u64) {
    let rows = pg
        .query(
            &format!("SELECT movie_id, rating FROM {table} ORDER BY movie_id"),
            &[],
        )
        .unwrap();
    let pairs: Vec<(i32, f64)> = rows
        .iter()
        .map(|r| (r.get::<_, i32>(0), r.get::<_, f64>(1)))
        .collect();
    (pairs.len(), checksum(&pairs))
}

/// Mirror-side: same pairs out of the columnar table, same checksum.
fn mirror_truth(m: &Mirror, table: &str) -> (usize, u64) {
    let t = &m.db.catalog.tables[table];
    let ids = match &t.columns["movie_id"] {
        bruce_query::Column::ScalarF64(v) => v,
        _ => panic!("movie_id must be ScalarF64"),
    };
    let ratings = match &t.columns["rating"] {
        bruce_query::Column::ScalarF64(v) => v,
        _ => panic!("rating must be ScalarF64"),
    };
    let mut pairs: Vec<(i32, f64)> = ids
        .iter()
        .zip(ratings.iter())
        .map(|(&i, &r)| (i as i32, r))
        .collect();
    pairs.sort_by_key(|&(id, _)| id);
    (pairs.len(), checksum(&pairs))
}

/// FNV-1a over (id, rating) with rating compared by exact f64 bits —
/// PG's float8 text output round-trips shortest-exact, so the mirror
/// (which parsed the text) must match bit-for-bit.
fn checksum(pairs: &[(i32, f64)]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    };
    for &(id, rating) in pairs {
        for b in id.to_le_bytes() {
            eat(b);
        }
        for b in rating.to_bits().to_le_bytes() {
            eat(b);
        }
    }
    h
}

/// Snapshot-create a mirror + streaming source for `table`.
fn subscribe(table: &str, slot: &str, publication: &str) -> (PgOutputSource, Mirror) {
    let mut src = PgOutputSource::connect(cfg(slot, publication)).unwrap();
    let mirror = match src.create_slot_with_snapshot().unwrap() {
        SlotSetup::CreatedSnapshotOpen { .. } => {
            let (cols, rows) = src
                .snapshot_query(&format!(
                    "SELECT movie_id, genre, rating, year, e0, e1 FROM {table}"
                ))
                .unwrap();
            src.commit_snapshot().unwrap();
            Mirror::from_snapshot(table_map(table), &cols, &rows).unwrap()
        }
        SlotSetup::AlreadyExists => panic!("slot {slot} must be fresh"),
    };
    src.start().unwrap();
    (src, mirror)
}

/// Reconnect to an EXISTING slot and resume streaming; bounded
/// retries because the killed walsender may still hold the slot
/// (55006) for a moment. Returns the source and the measured
/// reconnect duration.
fn resume(slot: &str, publication: &str) -> (PgOutputSource, Duration) {
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs(30);
    loop {
        let attempt =
            PgOutputSource::connect(cfg(slot, publication)).and_then(|mut s| s.start().map(|()| s));
        match attempt {
            Ok(s) => return (s, t0.elapsed()),
            Err(e) => {
                assert!(Instant::now() < deadline, "resume of {slot} timed out: {e}");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

// ---------------------------------------------------------------- (a)

const CHAOS_TXS: i32 = 300;
const CHAOS_SEED_ROWS: i32 = 100;
const N_KILLS: usize = 5;

#[test]
fn kill_resume_x5_exactly_once() {
    let _guard = pg_serial();
    let (table, publication, slot) = ("cdc_movies_chaos", "bruce_chaos_pub", "bruce_cdc_chaos");
    let mut pg = control();
    reset_artefacts(&mut pg, table, publication, slot);

    // seed BEFORE the slot: arrives via snapshot, not the stream
    let mut rng: u64 = 0xB01D_FACE;
    for i in 1..=CHAOS_SEED_ROWS {
        insert_one(&mut pg, table, i, &mut rng);
    }
    let (mut src, mut mirror) = subscribe(table, slot, publication);
    assert_eq!(mirror.n_rows(), CHAOS_SEED_ROWS as usize);

    // writer: 300 single-row txs, interleaved INSERT/DELETE, ids
    // tracked so deletes always hit a live row
    let done = Arc::new(AtomicI32::new(0));
    let done_w = done.clone();
    let writer = std::thread::spawn(move || {
        let mut pg = control();
        let mut rng: u64 = 0xDEAD_BEEF;
        let mut live: Vec<i32> = (1..=CHAOS_SEED_ROWS).collect();
        let mut next_id = CHAOS_SEED_ROWS;
        for _ in 0..CHAOS_TXS {
            let h = mix(&mut rng);
            if live.len() < 20 || !h.is_multiple_of(3) {
                next_id += 1;
                insert_one(&mut pg, table, next_id, &mut rng);
                live.push(next_id);
            } else {
                let victim = live.swap_remove((h >> 8) as usize % live.len());
                let n = pg
                    .execute(
                        &format!("DELETE FROM {table} WHERE movie_id = $1"),
                        &[&victim],
                    )
                    .unwrap();
                assert_eq!(n, 1);
            }
        }
        done_w.store(1, Ordering::SeqCst);
    });

    // apply loop with 5 kills; kills land BETWEEN apply and ack — the
    // exactly-once window — so the next resume redelivers an
    // already-applied transaction and the watermark must filter it.
    let kill_after_rows = [37usize, 91, 148, 202, 261];
    let mut kills = 0usize;
    let mut replays_filtered = 0usize;
    let mut recovery: Vec<Duration> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        assert!(
            Instant::now() < deadline,
            "chaos stream did not converge in 120s"
        );
        match src.next_tx(Duration::from_millis(300)).unwrap() {
            Some(tx) => {
                let n = mirror.apply_tx(&tx).unwrap();
                if n == 0 && !tx.changes.is_empty() {
                    replays_filtered += 1; // exactly-once watermark hit
                }
                if kills < N_KILLS && mirror.rows_applied >= kill_after_rows[kills] {
                    kills += 1;
                    drop(src); // CRASH: applied but never acked
                    let (s, dt) = resume(slot, publication);
                    src = s;
                    recovery.push(dt);
                    continue;
                }
                src.ack(tx.end_lsn).unwrap();
            }
            None => {
                if done.load(Ordering::SeqCst) == 1 {
                    let (want_n, want_sum) = pg_truth(&mut pg, table);
                    let (got_n, got_sum) = mirror_truth(&mirror, table);
                    if want_n == got_n && want_sum == got_sum {
                        break;
                    }
                }
            }
        }
    }
    writer.join().unwrap();
    assert_eq!(kills, N_KILLS, "all 5 kills must have fired");
    assert!(
        replays_filtered >= N_KILLS,
        "each kill-before-ack must redeliver >=1 applied tx (got {replays_filtered})"
    );

    // final exactly-once assertion, explicit
    let (want_n, want_sum) = pg_truth(&mut pg, table);
    let (got_n, got_sum) = mirror_truth(&mirror, table);
    assert_eq!(got_n, want_n, "row count: no dups, no losses");
    assert_eq!(got_sum, want_sum, "checksum of sorted (id, rating) pairs");
    assert!(
        mirror.txs_applied >= CHAOS_TXS as usize,
        "all writer txs applied once"
    );

    let recov: Vec<String> = recovery
        .iter()
        .map(|d| format!("{}ms", d.as_millis()))
        .collect();
    eprintln!(
        "chaos kill-resume: {} txs applied, {} rows final, {replays_filtered} replayed txs \
         filtered by the exactly-once watermark; reconnect times {:?}",
        mirror.txs_applied, got_n, recov
    );

    drop(src);
    cleanup(&mut pg, table, publication, slot);
}

// --------------------------------------------------------------- (a2)

/// PG-side FULL ground truth over every mapped column: (row count,
/// FNV-1a of (id, genre, rating/year/e0/e1 bits) sorted by id) —
/// updates mutate genre/rating/embedding, so the insert-only checksum
/// of `pg_truth` is not discriminating enough here.
fn pg_truth_full(pg: &mut Client, table: &str) -> (usize, u64) {
    let rows = pg
        .query(
            &format!("SELECT movie_id, genre, rating, year, e0, e1 FROM {table} ORDER BY movie_id"),
            &[],
        )
        .unwrap();
    let tuples: Vec<(i32, String, f64, f64, f64, f64)> = rows
        .iter()
        .map(|r| {
            (
                r.get::<_, i32>(0),
                r.get::<_, String>(1),
                r.get::<_, f64>(2),
                r.get::<_, f64>(3),
                r.get::<_, f64>(4),
                r.get::<_, f64>(5),
            )
        })
        .collect();
    (tuples.len(), checksum_full(&tuples))
}

fn mirror_truth_full(m: &Mirror, table: &str) -> (usize, u64) {
    let t = &m.db.catalog.tables[table];
    let scalar = |name: &str| -> &Vec<f64> {
        match &t.columns[name] {
            bruce_query::Column::ScalarF64(v) => v,
            _ => panic!("{name} must be ScalarF64"),
        }
    };
    let (ids, ratings, years, e0s, e1s) = (
        scalar("movie_id"),
        scalar("rating"),
        scalar("year"),
        scalar("e0"),
        scalar("e1"),
    );
    let genres: Vec<String> = match &t.columns["genre"] {
        bruce_query::Column::DictU32 { codes, dict } => {
            codes.iter().map(|&c| dict[c as usize].clone()).collect()
        }
        _ => panic!("genre must be DictU32"),
    };
    let mut tuples: Vec<(i32, String, f64, f64, f64, f64)> = (0..ids.len())
        .map(|i| {
            (
                ids[i] as i32,
                genres[i].clone(),
                ratings[i],
                years[i],
                e0s[i],
                e1s[i],
            )
        })
        .collect();
    tuples.sort_by_key(|t| t.0);
    (tuples.len(), checksum_full(&tuples))
}

fn checksum_full(tuples: &[(i32, String, f64, f64, f64, f64)]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    };
    for (id, genre, rating, year, e0, e1) in tuples {
        for b in id.to_le_bytes() {
            eat(b);
        }
        for &b in genre.as_bytes() {
            eat(b);
        }
        eat(0); // genre terminator, so ("ab","c") != ("a","bc")
        for v in [rating, year, e0, e1] {
            for b in v.to_bits().to_le_bytes() {
                eat(b);
            }
        }
    }
    h
}

/// Workstream 12 v2 (BACKLOG #12): the kill-resume chaos extended to
/// a mixed insert/update/delete workload with updates >= 40% of
/// transactions, 5 kills in the apply-then-ack window, exactly-once
/// asserted against the FULL PG ground truth (every mapped column).
#[test]
fn kill_resume_update_heavy_exactly_once() {
    let _guard = pg_serial();
    let (table, publication, slot) = (
        "cdc_movies_updchaos",
        "bruce_updchaos_pub",
        "bruce_cdc_updchaos",
    );
    let mut pg = control();
    reset_artefacts(&mut pg, table, publication, slot);

    let mut rng: u64 = 0xFACE_0FF5;
    for i in 1..=CHAOS_SEED_ROWS {
        insert_one(&mut pg, table, i, &mut rng);
    }
    let (mut src, mut mirror) = subscribe(table, slot, publication);
    assert_eq!(mirror.n_rows(), CHAOS_SEED_ROWS as usize);

    // writer: 300 single-row txs — >=40% UPDATE (including occasional
    // pk-moving updates), the rest insert/delete; ids tracked so
    // updates and deletes always hit a live row
    let done = Arc::new(AtomicI32::new(0));
    let done_w = done.clone();
    let n_updates = Arc::new(AtomicI32::new(0));
    let n_updates_w = n_updates.clone();
    let writer = std::thread::spawn(move || {
        let mut pg = control();
        let mut rng: u64 = 0xC0FF_EE00;
        let mut live: Vec<i32> = (1..=CHAOS_SEED_ROWS).collect();
        let mut next_id = CHAOS_SEED_ROWS;
        let mut upd = 0i32;
        for t in 0..CHAOS_TXS {
            let h = mix(&mut rng);
            let kind = h % 20;
            if kind < 9 && !live.is_empty() {
                // 45%: UPDATE
                upd += 1;
                let vi = (h >> 8) as usize % live.len();
                let victim = live[vi];
                if t % 7 == 3 {
                    // pk-moving update: old 'O' tuple carries old pk
                    next_id += 1;
                    let n = pg
                        .execute(
                            &format!("UPDATE {table} SET movie_id = $2 WHERE movie_id = $1"),
                            &[&victim, &next_id],
                        )
                        .unwrap();
                    assert_eq!(n, 1);
                    live[vi] = next_id;
                } else {
                    let g = ["action", "drama", "comedy", "horror"][(h >> 16) as usize % 4];
                    let rating = 1.0 + ((h >> 24) % 900) as f64 / 100.0;
                    let theta = ((h >> 40) % 6283) as f64 / 1000.0;
                    let n = pg
                        .execute(
                            &format!(
                                "UPDATE {table} SET genre=$2, rating=$3, e0=$4, e1=$5 \
                                 WHERE movie_id = $1"
                            ),
                            &[&victim, &g, &rating, &theta.cos(), &theta.sin()],
                        )
                        .unwrap();
                    assert_eq!(n, 1);
                }
            } else if kind < 15 || live.len() < 20 {
                // 30%: INSERT
                next_id += 1;
                insert_one(&mut pg, table, next_id, &mut rng);
                live.push(next_id);
            } else {
                // 25%: DELETE
                let victim = live.swap_remove((h >> 8) as usize % live.len());
                let n = pg
                    .execute(
                        &format!("DELETE FROM {table} WHERE movie_id = $1"),
                        &[&victim],
                    )
                    .unwrap();
                assert_eq!(n, 1);
            }
        }
        n_updates_w.store(upd, Ordering::SeqCst);
        done_w.store(1, Ordering::SeqCst);
    });

    let kill_after_rows = [37usize, 91, 148, 202, 261];
    let mut kills = 0usize;
    let mut replays_filtered = 0usize;
    let mut updates_applied = 0usize;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        assert!(
            Instant::now() < deadline,
            "update-heavy chaos did not converge in 120s"
        );
        match src.next_tx(Duration::from_millis(300)).unwrap() {
            Some(tx) => {
                let has_update = tx
                    .changes
                    .iter()
                    .any(|c| matches!(c, RowChange::Update { .. }));
                let n = mirror.apply_tx(&tx).unwrap();
                if n == 0 && !tx.changes.is_empty() {
                    replays_filtered += 1;
                } else if has_update {
                    updates_applied += 1;
                }
                if kills < N_KILLS && mirror.rows_applied >= kill_after_rows[kills] {
                    kills += 1;
                    drop(src); // CRASH: applied but never acked
                    let (s, _dt) = resume(slot, publication);
                    src = s;
                    continue;
                }
                src.ack(tx.end_lsn).unwrap();
            }
            None => {
                if done.load(Ordering::SeqCst) == 1 {
                    let (want_n, want_sum) = pg_truth_full(&mut pg, table);
                    let (got_n, got_sum) = mirror_truth_full(&mirror, table);
                    if want_n == got_n && want_sum == got_sum {
                        break;
                    }
                }
            }
        }
    }
    writer.join().unwrap();
    assert_eq!(kills, N_KILLS, "all 5 kills must have fired");
    assert!(
        replays_filtered >= N_KILLS,
        "each kill-before-ack must redeliver >=1 applied tx (got {replays_filtered})"
    );
    let upd = n_updates.load(Ordering::SeqCst);
    assert!(
        upd as f64 >= 0.40 * CHAOS_TXS as f64,
        "workload must be >=40% updates (got {upd}/{CHAOS_TXS})"
    );
    assert!(
        updates_applied > 0,
        "the stream must actually have delivered updates"
    );

    // final exactly-once assertion over EVERY mapped column
    let (want_n, want_sum) = pg_truth_full(&mut pg, table);
    let (got_n, got_sum) = mirror_truth_full(&mirror, table);
    assert_eq!(got_n, want_n, "row count: no dups, no losses");
    assert_eq!(got_sum, want_sum, "full checksum (genre+rating+year+emb)");

    eprintln!(
        "chaos update-heavy: {} txs ({upd} updates = {:.0}%), {} rows final, \
         {replays_filtered} replays filtered, {updates_applied} update txs applied",
        mirror.txs_applied,
        100.0 * upd as f64 / CHAOS_TXS as f64,
        got_n
    );

    drop(src);
    cleanup(&mut pg, table, publication, slot);
}

// ---------------------------------------------------------------- (b)

#[test]
fn pg_restart_mid_stream_reconnects_and_converges() {
    let _guard = pg_serial();
    let (table, publication, slot) = (
        "cdc_movies_restart",
        "bruce_restart_pub",
        "bruce_cdc_restart",
    );
    let mut pg = control();
    reset_artefacts(&mut pg, table, publication, slot);

    let mut rng: u64 = 0x5EED_0001;
    for i in 1..=50 {
        insert_one(&mut pg, table, i, &mut rng);
    }
    let (src, mut mirror) = subscribe(table, slot, publication);
    assert_eq!(mirror.n_rows(), 50);
    let mut src = RetryingSource::new(src, 60, Duration::from_millis(500));

    // phase 1: 30 streamed txs applied normally
    for i in 51..=80 {
        insert_one(&mut pg, table, i, &mut rng);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while mirror.n_rows() < 80 {
        assert!(Instant::now() < deadline, "phase 1 fell behind");
        if let Some(tx) = src.next_tx(Duration::from_millis(300)).unwrap() {
            mirror.apply_tx(&tx).unwrap();
            src.ack(tx.end_lsn).unwrap();
        }
    }

    // restart PG while the subscriber is connected AND streaming: the
    // fast shutdown waits for the logical walsender to flush its
    // remaining WAL to a *reading* client, so the restart runs in a
    // thread while this thread keeps polling the stream (a subscriber
    // that stops reading would hold the shutdown until pg_ctl's
    // timeout).
    drop(pg); // our own control conn must not block the fast shutdown
    let t_restart = Instant::now();
    let restarter = std::thread::spawn(move || pg_restart_cmd().output().expect("pg_ctl must run"));
    let deadline = Instant::now() + Duration::from_secs(90);
    while !restarter.is_finished() {
        assert!(Instant::now() < deadline, "pg_ctl restart wedged");
        // transient stream errors during the bounce are absorbed by
        // RetryingSource; committed stragglers still apply
        if let Some(tx) = src.next_tx(Duration::from_millis(200)).unwrap() {
            mirror.apply_tx(&tx).unwrap();
            src.ack(tx.end_lsn).unwrap();
        }
    }
    let out = restarter.join().unwrap();
    assert!(
        out.status.success(),
        "pg_ctl restart failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // phase 2: new writes after the restart; the subscriber must
    // reconnect through RetryingSource and converge
    let mut pg = control_retry(Duration::from_secs(30));
    for i in 81..=110 {
        insert_one(&mut pg, table, i, &mut rng);
    }
    let mut first_after: Option<Duration> = None;
    let deadline = Instant::now() + Duration::from_secs(60);
    while mirror.n_rows() < 110 {
        assert!(Instant::now() < deadline, "post-restart stream fell behind");
        if let Some(tx) = src.next_tx(Duration::from_millis(300)).unwrap() {
            if mirror.apply_tx(&tx).unwrap() > 0 && first_after.is_none() {
                first_after = Some(t_restart.elapsed());
            }
            src.ack(tx.end_lsn).unwrap();
        }
    }
    assert!(
        src.reconnects >= 1,
        "the wrapper must actually have reconnected"
    );

    let (want_n, want_sum) = pg_truth(&mut pg, table);
    let (got_n, got_sum) = mirror_truth(&mirror, table);
    assert_eq!(
        (got_n, got_sum),
        (want_n, want_sum),
        "post-restart convergence"
    );

    eprintln!(
        "chaos pg-restart: {} reconnects; restart -> first applied tx {:?}; {} rows converged",
        src.reconnects,
        first_after.unwrap(),
        got_n
    );

    drop(src);
    cleanup(&mut pg, table, publication, slot);
}

// ---------------------------------------------------------------- (c)

const BIG_TX_ROWS: usize = 5000;

#[test]
fn large_tx_5000_inserts_is_commit_buffered_and_atomic() {
    let _guard = pg_serial();
    let (table, publication, slot) = ("cdc_movies_bigtx", "bruce_bigtx_pub", "bruce_cdc_bigtx");
    let mut pg = control();
    reset_artefacts(&mut pg, table, publication, slot);

    let (mut src, mut mirror) = subscribe(table, slot, publication);
    assert_eq!(mirror.n_rows(), 0);

    // open transaction: 5000 rows inserted but NOT committed
    let mut tx = pg.transaction().unwrap();
    tx.execute(
        &format!(
            "INSERT INTO {table} \
             SELECT i, 'g' || (i % 8), 1.0 + (i % 90)::float8 / 10, 2000, \
                    cos(i::float8), sin(i::float8) \
             FROM generate_series(1, $1) i"
        ),
        &[&(BIG_TX_ROWS as i32)],
    )
    .unwrap();

    // CONSISTENCY CONTRACT, part 1 — commit-buffered delivery: while
    // the transaction is open, the source yields NOTHING (pgoutput
    // itself only decodes at commit; the source additionally buffers
    // Begin..Commit). The mirror cannot observe a partial state that
    // was never delivered.
    for _ in 0..3 {
        assert!(
            src.next_tx(Duration::from_millis(200)).unwrap().is_none(),
            "no transaction may be delivered before COMMIT"
        );
        assert_eq!(mirror.n_rows(), 0, "mirror must not move pre-commit");
    }

    tx.commit().unwrap();

    // CONSISTENCY CONTRACT, part 2 — the whole transaction arrives as
    // ONE CommittedTx and lands in ONE apply_tx call: readers
    // sequenced between apply_tx calls (the only readers possible —
    // apply_tx takes &mut self) see 0 or 5000 rows, never a slice.
    let deadline = Instant::now() + Duration::from_secs(60);
    let committed = loop {
        assert!(Instant::now() < deadline, "big tx never arrived");
        if let Some(t) = src.next_tx(Duration::from_millis(500)).unwrap() {
            break t;
        }
    };
    assert_eq!(
        committed.changes.len(),
        BIG_TX_ROWS,
        "one tx, all 5000 changes"
    );
    assert!(committed
        .changes
        .iter()
        .all(|c| matches!(c, RowChange::Insert { rel, .. } if rel == table)));

    let t_apply = Instant::now();
    assert_eq!(mirror.apply_tx(&committed).unwrap(), BIG_TX_ROWS);
    let apply_wall = t_apply.elapsed();
    src.ack(committed.end_lsn).unwrap();
    assert_eq!(mirror.n_rows(), BIG_TX_ROWS, "0 -> 5000 in one step");

    let (want_n, want_sum) = pg_truth(&mut pg, table);
    let (got_n, got_sum) = mirror_truth(&mirror, table);
    assert_eq!((got_n, got_sum), (want_n, want_sum), "big tx content exact");

    eprintln!(
        "chaos big-tx: {BIG_TX_ROWS} inserts delivered as one CommittedTx, applied in {:?}",
        apply_wall
    );

    drop(src);
    cleanup(&mut pg, table, publication, slot);
}

// ------------------------------------------------- pure-logic pins

/// The exactly-once watermark, without PG: a replayed transaction
/// (same end_lsn) must be a no-op — this is the defect the
/// kill-resume test exposes at the system level.
#[test]
fn replayed_tx_is_filtered_by_watermark() {
    let names: Vec<String> = ["movie_id", "genre", "rating", "year", "e0", "e1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &names, &[]).unwrap();

    let ins = |id: i32, lsn: u64| CommittedTx {
        changes: vec![RowChange::Insert {
            rel: "cdc_movies".into(),
            cols: names
                .iter()
                .cloned()
                .zip(
                    [
                        id.to_string(),
                        "action".into(),
                        "5".into(),
                        "2000".into(),
                        "1".into(),
                        "0".into(),
                    ]
                    .map(bruce_cdc::pgoutput::TupleDatum::Text),
                )
                .collect(),
        }],
        commit_ts_us: 0,
        end_lsn: lsn,
    };

    assert_eq!(m.apply_tx(&ins(1, 100)).unwrap(), 1);
    assert_eq!(m.last_lsn, 100);
    // crash-after-apply: the same tx is redelivered
    assert_eq!(
        m.apply_tx(&ins(1, 100)).unwrap(),
        0,
        "replay must be filtered"
    );
    assert_eq!(m.n_rows(), 1, "no duplicate row");
    assert_eq!(m.txs_applied, 1, "replay must not count");
    // and an older LSN likewise
    assert_eq!(m.apply_tx(&ins(2, 90)).unwrap(), 0);
    // a genuinely new tx still applies
    assert_eq!(m.apply_tx(&ins(2, 101)).unwrap(), 1);
    assert_eq!(m.n_rows(), 2);
}

/// A delete replay would otherwise fail loudly ("removed 0 rows");
/// the watermark must catch it BEFORE the delete runs.
#[test]
fn replayed_delete_does_not_error() {
    let names: Vec<String> = ["movie_id", "genre", "rating", "year", "e0", "e1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let seed = vec![vec![
        Some("1".to_string()),
        Some("action".to_string()),
        Some("5".to_string()),
        Some("2000".to_string()),
        Some("1".to_string()),
        Some("0".to_string()),
    ]];
    let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &names, &seed).unwrap();
    let del = CommittedTx {
        changes: vec![RowChange::Delete {
            rel: "cdc_movies".into(),
            old: names
                .iter()
                .cloned()
                .zip(
                    seed[0]
                        .iter()
                        .map(|v| bruce_cdc::pgoutput::TupleDatum::Text(v.clone().unwrap())),
                )
                .collect(),
        }],
        commit_ts_us: 0,
        end_lsn: 50,
    };
    assert_eq!(m.apply_tx(&del).unwrap(), 1);
    assert_eq!(m.n_rows(), 0);
    // redelivered after a crash-before-ack: must be a filtered no-op,
    // not an "out of sync" apply error
    match m.apply_tx(&del) {
        Ok(0) => {}
        Ok(n) => panic!("replayed delete applied {n} rows"),
        Err(e) => panic!("replayed delete must not error: {e}"),
    }
}

/// RetryingSource's transient/semantic split, without PG: the exact
/// classification the reconnect loop keys on. Changing the retryable
/// set must be a conscious act that updates this pin.
#[test]
fn transient_error_classification() {
    let transient = [
        CdcError::Io("broken pipe".into()),
        CdcError::Protocol("connection closed".into()),
        CdcError::Backend {
            code: "57P01".into(),
            message: "terminating".into(),
        },
        CdcError::Backend {
            code: "57P03".into(),
            message: "starting up".into(),
        },
        CdcError::Backend {
            code: "55006".into(),
            message: "slot active".into(),
        },
    ];
    for e in &transient {
        assert!(is_transient(e), "must be retryable: {e}");
    }
    let semantic = [
        CdcError::Decode("bad tuple".into()),
        CdcError::Apply("out of sync".into()),
        CdcError::Backend {
            code: "42710".into(),
            message: "duplicate".into(),
        },
        CdcError::Backend {
            code: "42601".into(),
            message: "syntax".into(),
        },
    ];
    for e in &semantic {
        assert!(!is_transient(e), "must NOT be retryable: {e}");
    }
}

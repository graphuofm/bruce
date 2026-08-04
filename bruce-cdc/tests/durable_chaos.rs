//! Durable-mirror chaos (workstream 12 v2): the subscriber is a REAL
//! child process (this test binary re-execed with BRUCE_CDC_CHILD=1)
//! that loads the mirror FROM DISK, streams, saves after every apply,
//! then acks. The parent SIGKILLs it at 5 deterministic
//! rows-applied thresholds (read from the durable file, so the kill
//! lands at whatever instant the pipeline is in — mid-apply, mid-save
//! or pre-ack) and restarts it from disk. Exactly-once must survive
//! actual process death: the final durable state equals the live PG
//! ground truth over every mapped column, bit for bit.
//!
//! Workload: mixed insert/update/delete with updates >= 40%.
//! Needs the live PG on /tmp:54329; owns cdc_movies_dur +
//! bruce_dur_pub + bruce_cdc_dur.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use postgres::{Client, NoTls};

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::source::{ChangeSource, PgOutputSource, SlotSetup, SourceConfig};

const TABLE: &str = "cdc_movies_dur";
const PUB: &str = "bruce_dur_pub";
const SLOT: &str = "bruce_cdc_dur";
const SEED_ROWS: i32 = 100;
const WORKLOAD_TXS: i32 = 400;
const KILL_AFTER_ROWS: [usize; 5] = [37, 91, 148, 202, 261];

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

fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

fn insert_one(pg: &mut Client, id: i32, rng: &mut u64) {
    let h = mix(rng);
    let genre = ["action", "drama", "comedy", "horror"][(h % 4) as usize];
    let rating = 1.0 + ((h >> 8) % 900) as f64 / 100.0;
    let theta = ((h >> 16) % 6283) as f64 / 1000.0;
    pg.execute(
        &format!("INSERT INTO {TABLE} VALUES ($1,$2,$3,$4,$5,$6)"),
        &[&id, &genre, &rating, &2000.0f64, &theta.cos(), &theta.sin()],
    )
    .unwrap();
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
        eat(0);
        for v in [rating, year, e0, e1] {
            for b in v.to_bits().to_le_bytes() {
                eat(b);
            }
        }
    }
    h
}

fn pg_truth_full(pg: &mut Client) -> (usize, u64) {
    let rows = pg
        .query(
            &format!("SELECT movie_id, genre, rating, year, e0, e1 FROM {TABLE} ORDER BY movie_id"),
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

fn mirror_truth_full(m: &Mirror) -> (usize, u64) {
    let t = &m.db.catalog.tables[TABLE];
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

fn state_path() -> PathBuf {
    std::env::temp_dir().join("bruce_cdc_durable_chaos.mirror")
}

// ------------------------------------------------------------ child

/// The subscriber child. A no-op under normal `cargo test`; the real
/// loop runs only when the parent re-execs this binary with
/// BRUCE_CDC_CHILD=1. It NEVER snapshots: state comes from disk only
/// — that is the property under test. Save-THEN-ack ordering makes
/// every kill window safe:
///   kill after apply, before save -> tx not acked, redelivered,
///     reapplied onto the on-disk state that never saw it;
///   kill after save, before ack  -> tx redelivered, filtered by the
///     DURABLE last_lsn watermark;
///   kill mid-save                -> atomic rename leaves the previous
///     complete file.
#[test]
fn child_subscriber_loop() {
    if std::env::var("BRUCE_CDC_CHILD").is_err() {
        return; // normal test run: nothing to do
    }
    let path = state_path();
    let lifetime = Instant::now() + Duration::from_secs(120); // orphan safety
    let mut mirror = Mirror::load(&path).expect("child must start from the durable state");

    // resume the EXISTING slot; the previous incarnation's walsender
    // may hold it for a moment (55006), so retry within a deadline
    let mut src = loop {
        assert!(Instant::now() < lifetime, "child could not resume in time");
        let mut cfg = SourceConfig::local_default(SLOT);
        cfg.publication = PUB.into();
        let attempt = PgOutputSource::connect(cfg).and_then(|mut s| {
            match s.create_slot_with_snapshot()? {
                SlotSetup::AlreadyExists => {}
                SlotSetup::CreatedSnapshotOpen { .. } => {
                    panic!("slot vanished: the child must resume, never re-snapshot")
                }
            }
            s.start().map(|()| s)
        });
        match attempt {
            Ok(s) => break s,
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    };

    loop {
        if Instant::now() > lifetime {
            std::process::exit(0); // orphaned: parent died first
        }
        match src.next_tx(Duration::from_millis(200)) {
            Ok(Some(tx)) => {
                let before = mirror.last_lsn;
                mirror.apply_tx(&tx).expect("apply must not fail");
                if mirror.last_lsn != before {
                    mirror.save(&path).expect("save must not fail");
                }
                // ack ONLY after the state is durable
                let _ = src.ack(tx.end_lsn);
            }
            Ok(None) => {}
            Err(_) => std::process::exit(3), // transport died; parent respawns
        }
    }
}

fn spawn_child() -> Child {
    Command::new(std::env::current_exe().expect("current_exe"))
        .args(["child_subscriber_loop", "--exact", "--nocapture"])
        .env("BRUCE_CDC_CHILD", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn child subscriber")
}

// ----------------------------------------------------------- parent

#[test]
fn kill9_x5_restart_from_disk_exactly_once() {
    let mut pg = control();
    let path = state_path();
    let _ = std::fs::remove_file(&path);
    drop_slot(&mut pg, SLOT);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB};
         DROP TABLE IF EXISTS {TABLE};
         CREATE TABLE {TABLE}(
           movie_id int primary key, genre text, rating float8,
           year float8, e0 float8, e1 float8);
         ALTER TABLE {TABLE} REPLICA IDENTITY FULL;
         CREATE PUBLICATION {PUB} FOR TABLE {TABLE};"
    ))
    .unwrap();

    // seed + slot + consistent snapshot + FIRST durable save, all in
    // the parent; from here on only child processes touch the stream
    let mut rng: u64 = 0xD0D0_CAFE;
    for i in 1..=SEED_ROWS {
        insert_one(&mut pg, i, &mut rng);
    }
    {
        let mut cfg = SourceConfig::local_default(SLOT);
        cfg.publication = PUB.into();
        let mut src = PgOutputSource::connect(cfg).unwrap();
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
        mirror.save(&path).unwrap();
        drop(src); // walsender closes; the slot keeps the WAL
    }

    // writer: 400 mixed txs, updates >= 40%
    let done = Arc::new(AtomicI32::new(0));
    let done_w = done.clone();
    let n_updates = Arc::new(AtomicI32::new(0));
    let n_updates_w = n_updates.clone();
    let writer = std::thread::spawn(move || {
        let mut pg = control();
        let mut rng: u64 = 0xBADD_CAFE;
        let mut live: Vec<i32> = (1..=SEED_ROWS).collect();
        let mut next_id = SEED_ROWS;
        let mut upd = 0i32;
        for t in 0..WORKLOAD_TXS {
            let h = mix(&mut rng);
            let kind = h % 20;
            if kind < 9 && !live.is_empty() {
                upd += 1;
                let vi = (h >> 8) as usize % live.len();
                let victim = live[vi];
                if t % 11 == 5 {
                    next_id += 1;
                    let n = pg
                        .execute(
                            &format!("UPDATE {TABLE} SET movie_id = $2 WHERE movie_id = $1"),
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
                                "UPDATE {TABLE} SET genre=$2, rating=$3, e0=$4, e1=$5 \
                                 WHERE movie_id = $1"
                            ),
                            &[&victim, &g, &rating, &theta.cos(), &theta.sin()],
                        )
                        .unwrap();
                    assert_eq!(n, 1);
                }
            } else if kind < 15 || live.len() < 20 {
                next_id += 1;
                insert_one(&mut pg, next_id, &mut rng);
                live.push(next_id);
            } else {
                let victim = live.swap_remove((h >> 8) as usize % live.len());
                let n = pg
                    .execute(
                        &format!("DELETE FROM {TABLE} WHERE movie_id = $1"),
                        &[&victim],
                    )
                    .unwrap();
                assert_eq!(n, 1);
            }
        }
        n_updates_w.store(upd, Ordering::SeqCst);
        done_w.store(1, Ordering::SeqCst);
    });

    // supervise: kill -9 at each rows-applied threshold READ FROM THE
    // DURABLE FILE, restart from disk
    let mut child = spawn_child();
    let mut kills = 0usize;
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        assert!(Instant::now() < deadline, "durable chaos did not converge");
        std::thread::sleep(Duration::from_millis(25));
        if let Some(status) = child.try_wait().unwrap() {
            panic!("child died on its own (status {status}) — apply/save/resume defect");
        }
        let snap = match Mirror::load(&path) {
            Ok(m) => m,
            Err(_) => continue, // raced the very first child save; retry
        };
        if kills < KILL_AFTER_ROWS.len() && snap.rows_applied >= KILL_AFTER_ROWS[kills] {
            kills += 1;
            child.kill().unwrap(); // SIGKILL — no cleanup, no ack
            child.wait().unwrap();
            child = spawn_child();
            continue;
        }
        if done.load(Ordering::SeqCst) == 1 && kills == KILL_AFTER_ROWS.len() {
            let (want_n, want_sum) = pg_truth_full(&mut pg);
            let (got_n, got_sum) = (snap.n_rows(), mirror_truth_full(&snap).1);
            if got_n == want_n && got_sum == want_sum {
                break;
            }
        }
    }
    writer.join().unwrap();
    child.kill().unwrap();
    child.wait().unwrap();

    assert_eq!(kills, 5, "all 5 SIGKILLs must have fired");
    let upd = n_updates.load(Ordering::SeqCst);
    assert!(
        upd as f64 >= 0.40 * WORKLOAD_TXS as f64,
        "workload must be >=40% updates (got {upd}/{WORKLOAD_TXS})"
    );

    // final: the DURABLE state alone (no in-memory survivor exists —
    // every incarnation was SIGKILLed) equals PG, bit for bit
    let final_state = Mirror::load(&path).unwrap();
    let (want_n, want_sum) = pg_truth_full(&mut pg);
    let (got_n, got_sum) = (final_state.n_rows(), mirror_truth_full(&final_state).1);
    assert_eq!(
        got_n, want_n,
        "row count: no dups, no losses across 5 kill -9s"
    );
    assert_eq!(got_sum, want_sum, "full checksum across 5 kill -9s");

    eprintln!(
        "durable chaos: {WORKLOAD_TXS} txs ({upd} updates = {:.0}%), 5 SIGKILLs, \
         final durable state = PG ground truth ({got_n} rows), \
         rows_applied {} txs_applied {}",
        100.0 * upd as f64 / WORKLOAD_TXS as f64,
        final_state.rows_applied,
        final_state.txs_applied,
    );

    // ---- leave PG healthy ----
    drop_slot(&mut pg, SLOT);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB}; DROP TABLE IF EXISTS {TABLE};"
    ))
    .unwrap();
    let _ = std::fs::remove_file(&path);
}

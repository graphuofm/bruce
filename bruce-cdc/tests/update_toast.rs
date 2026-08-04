//! Live-PG round trip for the TOAST datum split (workstream 11 v2):
//! a >8KB text column with STORAGE EXTERNAL (forced out-of-line), an
//! UPDATE that does not touch it (arrives `TupleDatum::Unchanged`
//! and resolves from the mirror), an UPDATE that sets it NULL
//! (arrives `TupleDatum::Null` — the corpus must distinguish them),
//! a REPLICA IDENTITY DEFAULT pk change ('K' old tuple), and the
//! live pin that REPLICA IDENTITY NOTHING makes PG itself refuse the
//! UPDATE with SQLSTATE 55000 (which is why the assembler's typed
//! error is only reachable from untrusted streams).
//!
//! Needs the live PG on /tmp:54329; owns cdc_movies_toast +
//! bruce_toast_pub + bruce_cdc_toast (and the _nothing_ variants);
//! never touches cdc_movies / movies / bruce_pub.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use postgres::{Client, NoTls};

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::pgoutput::TupleDatum;
use bruce_cdc::source::{ChangeSource, PgOutputSource, RowChange, SlotSetup, SourceConfig};

static PG_LOCK: Mutex<()> = Mutex::new(());

fn pg_serial() -> MutexGuard<'static, ()> {
    PG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const TABLE: &str = "cdc_movies_toast";
const PUB: &str = "bruce_toast_pub";
const SLOT: &str = "bruce_cdc_toast";
/// >8KB, larger than the TOAST threshold with STORAGE EXTERNAL.
const BIG_LEN: usize = 10_000;

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

/// A deterministic, poorly-compressible-irrelevant big text (STORAGE
/// EXTERNAL disables compression, so repetition is fine) — unique per
/// id, byte length BIG_LEN.
fn big_text(id: i32) -> String {
    let seed = format!("movie-{id}-payload:");
    let mut s = String::with_capacity(BIG_LEN + seed.len());
    while s.len() < BIG_LEN {
        s.push_str(&seed);
        s.push((b'a' + (s.len() % 26) as u8) as char);
    }
    s.truncate(BIG_LEN);
    s
}

fn table_map() -> TableMap {
    TableMap {
        table: TABLE.into(),
        pk: "movie_id".into(),
        label_cols: vec!["genre".into(), "big".into()],
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

/// The mirror's current `big` value for a pk, out of the dict column.
fn mirror_big(m: &Mirror, id: f64) -> String {
    let t = &m.db.catalog.tables[TABLE];
    let ids = match &t.columns["movie_id"] {
        bruce_query::Column::ScalarF64(v) => v,
        _ => panic!("movie_id must be ScalarF64"),
    };
    let i = ids.iter().position(|&v| v == id).expect("row present");
    match &t.columns["big"] {
        bruce_query::Column::DictU32 { codes, dict } => dict[codes[i] as usize].clone(),
        _ => panic!("big must be DictU32"),
    }
}

fn mirror_rating(m: &Mirror, id: f64) -> f64 {
    let t = &m.db.catalog.tables[TABLE];
    let ids = match &t.columns["movie_id"] {
        bruce_query::Column::ScalarF64(v) => v,
        _ => panic!(),
    };
    let i = ids.iter().position(|&v| v == id).expect("row present");
    match &t.columns["rating"] {
        bruce_query::Column::ScalarF64(v) => v[i],
        _ => panic!(),
    }
}

fn next_tx_deadline(src: &mut PgOutputSource) -> bruce_cdc::source::CommittedTx {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(Instant::now() < deadline, "tx never arrived");
        if let Some(tx) = src.next_tx(Duration::from_millis(300)).unwrap() {
            return tx;
        }
    }
}

#[test]
fn toast_unchanged_vs_null_round_trip() {
    let _guard = pg_serial();
    let mut pg = control();
    drop_slot(&mut pg, SLOT);
    // REPLICA IDENTITY DEFAULT on purpose (the pk identifies rows);
    // STORAGE EXTERNAL forces >2KB values out of line, uncompressed,
    // so an untouched `big` arrives as the 'u' marker, never inline.
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB};
         DROP TABLE IF EXISTS {TABLE};
         CREATE TABLE {TABLE}(
           movie_id int primary key, genre text, rating float8,
           year float8, e0 float8, e1 float8, big text);
         ALTER TABLE {TABLE} ALTER COLUMN big SET STORAGE EXTERNAL;
         CREATE PUBLICATION {PUB} FOR TABLE {TABLE};"
    ))
    .unwrap();
    for id in 1..=3i32 {
        pg.execute(
            &format!("INSERT INTO {TABLE} VALUES ($1,$2,$3,$4,$5,$6,$7)"),
            &[
                &id,
                &"action",
                &(id as f64),
                &2000.0f64,
                &1.0f64,
                &0.0f64,
                &big_text(id),
            ],
        )
        .unwrap();
    }
    // prove the seed values really are TOASTed out of line: the TOAST
    // relation itself must hold (at least) the three 10KB payloads
    let toasted: i64 = pg
        .query_one(
            &format!(
                "SELECT pg_total_relation_size(reltoastrelid) FROM pg_class \
                 WHERE relname = '{TABLE}'"
            ),
            &[],
        )
        .unwrap()
        .get(0);
    assert!(
        toasted >= 3 * (BIG_LEN as i64),
        "big column must live in the TOAST relation (got {toasted} bytes)"
    );
    const { assert!(BIG_LEN > 8192, ">8KB per the workstream spec") };

    let mut src = PgOutputSource::connect({
        let mut c = SourceConfig::local_default(SLOT);
        c.publication = PUB.into();
        c
    })
    .unwrap();
    let mut mirror = match src.create_slot_with_snapshot().unwrap() {
        SlotSetup::CreatedSnapshotOpen { .. } => {
            let (cols, rows) = src
                .snapshot_query(&format!(
                    "SELECT movie_id, genre, rating, year, e0, e1, big FROM {TABLE}"
                ))
                .unwrap();
            src.commit_snapshot().unwrap();
            Mirror::from_snapshot(table_map(), &cols, &rows).unwrap()
        }
        SlotSetup::AlreadyExists => panic!("slot {SLOT} must be fresh"),
    };
    src.start().unwrap();
    assert_eq!(mirror.n_rows(), 3);
    assert_eq!(
        mirror_big(&mirror, 2.0),
        big_text(2),
        "snapshot carries big"
    );

    // ---- 1) UPDATE not touching big: arrives Unchanged, resolves
    //         from the mirror ----
    pg.execute(
        &format!("UPDATE {TABLE} SET rating = 9.5 WHERE movie_id = 2"),
        &[],
    )
    .unwrap();
    let tx = next_tx_deadline(&mut src);
    assert_eq!(tx.changes.len(), 1);
    match &tx.changes[0] {
        RowChange::Update { rel, old, new } => {
            assert_eq!(rel, TABLE);
            assert!(
                old.is_none(),
                "REPLICA IDENTITY DEFAULT + key untouched => no old tuple"
            );
            let big = new.iter().find(|(n, _)| n == "big").unwrap();
            assert_eq!(
                big.1,
                TupleDatum::Unchanged,
                "untouched TOAST must arrive as the 'u' marker"
            );
        }
        _ => panic!("want Update"),
    }
    assert_eq!(mirror.apply_tx(&tx).unwrap(), 1);
    src.ack(tx.end_lsn).unwrap();
    assert_eq!(mirror_rating(&mirror, 2.0), 9.5, "rating updated");
    assert_eq!(
        mirror_big(&mirror, 2.0),
        big_text(2),
        "unchanged TOAST resolved byte-exactly from the mirror"
    );

    // ---- 2) UPDATE setting big NULL: arrives Null, DISTINCT from
    //         Unchanged; the demo schema forbids NULL => typed error,
    //         mirror untouched ----
    pg.execute(
        &format!("UPDATE {TABLE} SET big = NULL WHERE movie_id = 2"),
        &[],
    )
    .unwrap();
    let tx = next_tx_deadline(&mut src);
    match &tx.changes[0] {
        RowChange::Update { new, .. } => {
            let big = new.iter().find(|(n, _)| n == "big").unwrap();
            assert_eq!(big.1, TupleDatum::Null, "SET NULL must arrive as 'n'");
            assert_ne!(
                big.1,
                TupleDatum::Unchanged,
                "the corpus must distinguish NULL from unchanged TOAST"
            );
        }
        _ => panic!("want Update"),
    }
    let err = mirror.apply_tx(&tx).unwrap_err().to_string();
    assert!(err.contains("NULL in column big"), "got: {err}");
    assert_eq!(
        mirror_big(&mirror, 2.0),
        big_text(2),
        "failed apply must not have touched the mirror"
    );
    src.ack(tx.end_lsn).unwrap(); // schema-forbidden NULL: skip + ack

    // resync row 2 (PG has big=NULL, the mirror rejected it): set a
    // fresh materialized value — arrives as 't' and applies
    let fresh = big_text(99);
    pg.execute(
        &format!("UPDATE {TABLE} SET big = $1 WHERE movie_id = 2"),
        &[&fresh],
    )
    .unwrap();
    let tx = next_tx_deadline(&mut src);
    // the mirror's watermark is BEHIND (the rejected tx advanced
    // nothing), so this tx is new and must apply
    assert_eq!(mirror.apply_tx(&tx).unwrap(), 1);
    src.ack(tx.end_lsn).unwrap();
    assert_eq!(mirror_big(&mirror, 2.0), fresh, "materialized 't' applies");

    // ---- 3) pk change under REPLICA IDENTITY DEFAULT: 'K' old tuple
    //         carries the old key; big untouched arrives Unchanged ----
    pg.execute(
        &format!("UPDATE {TABLE} SET movie_id = 42 WHERE movie_id = 3"),
        &[],
    )
    .unwrap();
    let tx = next_tx_deadline(&mut src);
    match &tx.changes[0] {
        RowChange::Update { old, new, .. } => {
            let old = old.as_ref().expect("key change => 'K' old tuple");
            let old_pk = old.iter().find(|(n, _)| n == "movie_id").unwrap();
            assert_eq!(old_pk.1, TupleDatum::text("3"));
            let old_big = old.iter().find(|(n, _)| n == "big").unwrap();
            assert_eq!(old_big.1, TupleDatum::Null, "'K' tuple: non-key NULL");
            let new_pk = new.iter().find(|(n, _)| n == "movie_id").unwrap();
            assert_eq!(new_pk.1, TupleDatum::text("42"));
        }
        _ => panic!("want Update"),
    }
    assert_eq!(mirror.apply_tx(&tx).unwrap(), 1);
    src.ack(tx.end_lsn).unwrap();
    assert_eq!(mirror.n_rows(), 3);
    assert_eq!(
        mirror_big(&mirror, 42.0),
        big_text(3),
        "row moved to pk 42 with its TOAST payload resolved"
    );

    // ---- leave PG healthy ----
    drop(src);
    drop_slot(&mut pg, SLOT);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB}; DROP TABLE IF EXISTS {TABLE};"
    ))
    .unwrap();
}

/// Live pin: PG itself refuses UPDATE on a published table with
/// REPLICA IDENTITY NOTHING (SQLSTATE 55000, message naming replica
/// identity) — so the assembler's typed guard for identity 'n' is
/// only reachable from untrusted/synthetic streams, and the RIGHT
/// user-facing fix (ALTER TABLE ... REPLICA IDENTITY) is the same one
/// our typed error names (pinned in tests/conformance.rs).
#[test]
fn replica_identity_nothing_update_is_refused_by_pg_itself() {
    let _guard = pg_serial();
    let mut pg = control();
    pg.batch_execute(
        "DROP PUBLICATION IF EXISTS bruce_toast_nothing_pub;
         DROP TABLE IF EXISTS cdc_toast_nothing;
         CREATE TABLE cdc_toast_nothing(movie_id int primary key, rating float8);
         ALTER TABLE cdc_toast_nothing REPLICA IDENTITY NOTHING;
         CREATE PUBLICATION bruce_toast_nothing_pub FOR TABLE cdc_toast_nothing;
         INSERT INTO cdc_toast_nothing VALUES (1, 5.0);",
    )
    .unwrap();
    let err = pg
        .execute(
            "UPDATE cdc_toast_nothing SET rating = 6.0 WHERE movie_id = 1",
            &[],
        )
        .expect_err("PG must refuse the UPDATE");
    let db_err = err.as_db_error().expect("server-side error");
    assert_eq!(
        db_err.code().code(),
        "55000",
        "object_not_in_prerequisite_state"
    );
    assert!(
        db_err.message().contains("replica identity"),
        "got: {}",
        db_err.message()
    );
    pg.batch_execute(
        "DROP PUBLICATION IF EXISTS bruce_toast_nothing_pub;
         DROP TABLE IF EXISTS cdc_toast_nothing;",
    )
    .unwrap();
}

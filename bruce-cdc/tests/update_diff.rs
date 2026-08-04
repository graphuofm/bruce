//! Differential test for the Update apply path (workstream 12 v2):
//! a random update-heavy workload on a REPLICA IDENTITY DEFAULT
//! table, streamed into the mirror, then compared against
//!   (a) the live PG state, bitwise (float8 text round-trips
//!       shortest-exact, so every mapped column must match bit for
//!       bit), and
//!   (b) a from-scratch re-snapshot Mirror — table bitwise, and the
//!       INCREMENTALLY maintained view (delete+insert through the
//!       (m,num,den) group inverse) against the re-snapshot's freshly
//!       recomputed view within 1e-12 relative.
//!
//! Needs the live PG on /tmp:54329; owns cdc_movies_diff +
//! bruce_diff_pub + bruce_cdc_diff.

use std::time::{Duration, Instant};

use ndarray::Array1;
use postgres::{Client, NoTls};

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::source::{ChangeSource, PgOutputSource, SlotSetup, SourceConfig};

const TABLE: &str = "cdc_movies_diff";
const PUB: &str = "bruce_diff_pub";
const SLOT: &str = "bruce_cdc_diff";
const SEED_ROWS: i32 = 150;
const WORKLOAD_TXS: i32 = 250;
const EPS: f64 = 0.1;

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

/// Deterministic splitmix64.
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

/// Bit-exact fingerprint of every mapped column, sorted by pk.
fn table_bits(m: &Mirror) -> Vec<(u64, String, u64, u64, u64, u64)> {
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
    // the key column must stay consistent with the e0/e1 scalars
    let emb = match &t.columns["emb"] {
        bruce_query::Column::KeyF64(a) => a,
        _ => panic!("emb must be KeyF64"),
    };
    for i in 0..ids.len() {
        assert_eq!(emb[(i, 0)].to_bits(), e0s[i].to_bits(), "emb[,0] == e0");
        assert_eq!(emb[(i, 1)].to_bits(), e1s[i].to_bits(), "emb[,1] == e1");
    }
    let mut rows: Vec<_> = (0..ids.len())
        .map(|i| {
            (
                ids[i].to_bits(),
                genres[i].clone(),
                ratings[i].to_bits(),
                years[i].to_bits(),
                e0s[i].to_bits(),
                e1s[i].to_bits(),
            )
        })
        .collect();
    rows.sort();
    rows
}

fn resnapshot(pg: &mut Client) -> Mirror {
    let rows = pg
        .query(
            &format!("SELECT movie_id, genre, rating, year, e0, e1 FROM {TABLE}"),
            &[],
        )
        .unwrap();
    let cols: Vec<String> = ["movie_id", "genre", "rating", "year", "e0", "e1"]
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
    Mirror::from_snapshot(table_map(), &cols, &text_rows).unwrap()
}

#[test]
fn random_update_workload_mirror_equals_pg_and_resnapshot() {
    let mut pg = control();
    drop_slot(&mut pg, SLOT);
    // REPLICA IDENTITY DEFAULT (the CREATE TABLE default): non-key
    // updates arrive with NO old tuple, pk changes with a 'K' tuple —
    // both Update shapes are exercised against a real walsender.
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB};
         DROP TABLE IF EXISTS {TABLE};
         CREATE TABLE {TABLE}(
           movie_id int primary key, genre text, rating float8,
           year float8, e0 float8, e1 float8);
         CREATE PUBLICATION {PUB} FOR TABLE {TABLE};"
    ))
    .unwrap();

    let mut rng: u64 = 0xD1FF_5EED;
    let genres = ["action", "drama", "comedy", "horror"];
    for id in 1..=SEED_ROWS {
        let h = mix(&mut rng);
        let theta = (h % 6283) as f64 / 1000.0;
        pg.execute(
            &format!("INSERT INTO {TABLE} VALUES ($1,$2,$3,$4,$5,$6)"),
            &[
                &id,
                &genres[(h >> 8) as usize % 4],
                &(1.0 + ((h >> 16) % 900) as f64 / 100.0),
                &2000.0f64,
                &theta.cos(),
                &theta.sin(),
            ],
        )
        .unwrap();
    }

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
                    "SELECT movie_id, genre, rating, year, e0, e1 FROM {TABLE}"
                ))
                .unwrap();
            src.commit_snapshot().unwrap();
            Mirror::from_snapshot(table_map(), &cols, &rows).unwrap()
        }
        SlotSetup::AlreadyExists => panic!("slot {SLOT} must be fresh"),
    };
    src.start().unwrap();
    assert_eq!(mirror.n_rows(), SEED_ROWS as usize);

    // the view rides the whole workload INCREMENTALLY (group inverse
    // on every update's delete half)
    let x = Array1::from_vec(vec![0.6, 0.8]);
    mirror
        .db
        .create_view("v_diff", TABLE, "genre", "rating", "emb", &x, EPS)
        .unwrap();

    // random update-only workload (plus a few pk moves): every tx is
    // an UPDATE — this is the pure differential for the new path
    let mut live: Vec<i32> = (1..=SEED_ROWS).collect();
    let mut next_id = SEED_ROWS;
    let mut pk_moves = 0;
    for t in 0..WORKLOAD_TXS {
        let h = mix(&mut rng);
        let vi = (h >> 8) as usize % live.len();
        let victim = live[vi];
        if t % 10 == 7 {
            next_id += 1;
            pg.execute(
                &format!("UPDATE {TABLE} SET movie_id = $2 WHERE movie_id = $1"),
                &[&victim, &next_id],
            )
            .unwrap();
            live[vi] = next_id;
            pk_moves += 1;
        } else {
            let theta = ((h >> 16) % 6283) as f64 / 1000.0;
            pg.execute(
                &format!("UPDATE {TABLE} SET genre=$2, rating=$3, e0=$4, e1=$5 WHERE movie_id=$1"),
                &[
                    &victim,
                    &genres[(h >> 4) as usize % 4],
                    &(1.0 + ((h >> 24) % 900) as f64 / 100.0),
                    &theta.cos(),
                    &theta.sin(),
                ],
            )
            .unwrap();
        }
    }
    assert!(pk_moves >= 20, "workload must include pk-moving updates");

    // apply until all workload txs have landed
    let deadline = Instant::now() + Duration::from_secs(60);
    while mirror.txs_applied < WORKLOAD_TXS as usize {
        assert!(Instant::now() < deadline, "stream fell behind");
        if let Some(tx) = src.next_tx(Duration::from_millis(300)).unwrap() {
            mirror.apply_tx(&tx).unwrap();
            src.ack(tx.end_lsn).unwrap();
        }
    }
    assert_eq!(
        mirror.n_rows(),
        SEED_ROWS as usize,
        "updates keep the count"
    );

    // ---- (b) from-scratch re-snapshot: table bitwise ----
    let fresh = resnapshot(&mut pg);
    assert_eq!(
        table_bits(&mirror),
        table_bits(&fresh),
        "mirror table must equal a from-scratch re-snapshot bit for bit"
    );

    // ---- view: incrementally maintained vs freshly recomputed ----
    let mut fresh = fresh;
    fresh
        .db
        .create_view("v_fresh", TABLE, "genre", "rating", "emb", &x, EPS)
        .unwrap();
    // group codes may differ between the two dicts; compare by label
    let label_of = |m: &Mirror, code: usize| -> String {
        match &m.db.catalog.tables[TABLE].columns["genre"] {
            bruce_query::Column::DictU32 { dict, .. } => dict[code].clone(),
            _ => panic!(),
        }
    };
    let got: std::collections::HashMap<String, f64> = mirror.db.views[0]
        .read()
        .into_iter()
        .map(|(g, v)| (label_of(&mirror, g), v))
        .collect();
    let want: std::collections::HashMap<String, f64> = fresh.db.views[0]
        .read()
        .into_iter()
        .map(|(g, v)| (label_of(&fresh, g), v))
        .collect();
    assert_eq!(got.len(), want.len(), "group count");
    let mut max_rel = 0.0f64;
    for (label, w) in &want {
        let g = got.get(label).unwrap_or_else(|| panic!("missing {label}"));
        let rel = (g - w).abs() / w.abs().max(1e-300);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= 1e-12,
            "group {label}: maintained {g} vs recomputed {w} (rel {rel:.3e})"
        );
    }
    eprintln!(
        "update diff: {WORKLOAD_TXS} update txs ({pk_moves} pk moves) over {SEED_ROWS} rows; \
         table bitwise-equal to re-snapshot; view max rel err {max_rel:.3e} (tol 1e-12)"
    );

    // ---- leave PG healthy ----
    drop(src);
    drop_slot(&mut pg, SLOT);
    pg.batch_execute(&format!(
        "DROP PUBLICATION IF EXISTS {PUB}; DROP TABLE IF EXISTS {TABLE};"
    ))
    .unwrap();
}

//! Durable-mirror format tests: bitwise round trip, integrity
//! (corruption/truncation/magic are typed errors, never panics),
//! atomic overwrite. No PostgreSQL needed.

use std::path::PathBuf;

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::pgoutput::TupleDatum;
use bruce_cdc::source::{CommittedTx, RowChange};
use bruce_cdc::CdcError;
use bruce_query::Column;

fn tmp_path(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "bruce_cdc_durable_test_{}_{name}.mirror",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&p);
    p
}

fn snap_cols() -> Vec<String> {
    ["movie_id", "genre", "rating", "year", "e0", "e1"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn row(id: i64, genre: &str, rating: f64, e0: f64, e1: f64) -> Vec<Option<String>> {
    vec![
        Some(id.to_string()),
        Some(genre.into()),
        Some(rating.to_string()),
        Some("2000".into()),
        Some(e0.to_string()),
        Some(e1.to_string()),
    ]
}

fn seeded_mirror() -> Mirror {
    let rows = vec![
        row(1, "action", 5.0, 1.0, 0.0),
        row(2, "drama", 7.25, 0.0, 1.0),
        row(3, "comedy", -0.5, 0.6, 0.8),
    ];
    let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &snap_cols(), &rows).unwrap();
    // push the counters + watermark off their defaults
    let names = snap_cols();
    let tx = CommittedTx {
        changes: vec![RowChange::Insert {
            rel: "cdc_movies".into(),
            cols: names
                .iter()
                .cloned()
                .zip(
                    row(4, "horror", 9.75, -1.0, 0.0)
                        .into_iter()
                        .map(|v| TupleDatum::Text(v.unwrap())),
                )
                .collect(),
        }],
        commit_ts_us: 7,
        end_lsn: 0xAB_CDEF,
    };
    m.apply_tx(&tx).unwrap();
    m
}

/// Every mapped column compared bitwise between two mirrors.
fn assert_tables_bitwise_equal(a: &Mirror, b: &Mirror) {
    let ta = &a.db.catalog.tables["cdc_movies"];
    let tb = &b.db.catalog.tables["cdc_movies"];
    assert_eq!(ta.columns.len(), tb.columns.len(), "column count");
    for (name, ca) in &ta.columns {
        let cb = tb.columns.get(name).expect("column present after load");
        match (ca, cb) {
            (Column::ScalarF64(va), Column::ScalarF64(vb)) => {
                let ba: Vec<u64> = va.iter().map(|v| v.to_bits()).collect();
                let bb: Vec<u64> = vb.iter().map(|v| v.to_bits()).collect();
                assert_eq!(ba, bb, "scalar column {name} bitwise");
            }
            (
                Column::DictU32 {
                    codes: ka,
                    dict: da,
                },
                Column::DictU32 {
                    codes: kb,
                    dict: db,
                },
            ) => {
                assert_eq!(ka, kb, "codes of {name}");
                assert_eq!(da, db, "dict of {name}");
            }
            (Column::KeyF64(ka), Column::KeyF64(kb)) => {
                assert_eq!(ka.dim(), kb.dim(), "key shape of {name}");
                let ba: Vec<u64> = ka.iter().map(|v| v.to_bits()).collect();
                let bb: Vec<u64> = kb.iter().map(|v| v.to_bits()).collect();
                assert_eq!(ba, bb, "key column {name} bitwise");
            }
            _ => panic!("column {name} changed kind across save/load"),
        }
    }
}

#[test]
fn save_load_round_trip_is_bitwise() {
    let path = tmp_path("round_trip");
    let m = seeded_mirror();
    m.save(&path).unwrap();
    let l = Mirror::load(&path).unwrap();
    assert_eq!(l.last_lsn, 0xAB_CDEF, "watermark survives");
    assert_eq!(l.rows_applied, 1);
    assert_eq!(l.txs_applied, 1);
    assert_eq!(l.n_rows(), 4);
    assert_tables_bitwise_equal(&m, &l);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn loaded_mirror_keeps_applying_and_filtering() {
    let path = tmp_path("keeps_applying");
    let m = seeded_mirror();
    m.save(&path).unwrap();
    let mut l = Mirror::load(&path).unwrap();

    let names = snap_cols();
    let ins = |id: i64, lsn: u64| CommittedTx {
        changes: vec![RowChange::Insert {
            rel: "cdc_movies".into(),
            cols: names
                .iter()
                .cloned()
                .zip(
                    row(id, "action", 1.0, 1.0, 0.0)
                        .into_iter()
                        .map(|v| TupleDatum::Text(v.unwrap())),
                )
                .collect(),
        }],
        commit_ts_us: 0,
        end_lsn: lsn,
    };
    // the DURABLE watermark filters a replayed tx after reload
    assert_eq!(
        l.apply_tx(&ins(4, 0xAB_CDEF)).unwrap(),
        0,
        "replay <= durable watermark must be filtered"
    );
    // and a genuinely new one applies
    assert_eq!(l.apply_tx(&ins(5, 0xAB_CDF0)).unwrap(), 1);
    assert_eq!(l.n_rows(), 5);
    // views can be recreated on the loaded db
    let x = ndarray::Array1::from_vec(vec![0.6, 0.8]);
    l.db.create_view("v", "cdc_movies", "genre", "rating", "emb", &x, 0.1)
        .unwrap();
    assert!(!l.db.views[0].read().is_empty());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn save_overwrites_atomically_and_leaves_no_temp_files() {
    let path = tmp_path("overwrite");
    let mut m = seeded_mirror();
    m.save(&path).unwrap();
    let first = std::fs::read(&path).unwrap();

    // mutate and save again over the same path
    let names = snap_cols();
    let tx = CommittedTx {
        changes: vec![RowChange::Delete {
            rel: "cdc_movies".into(),
            old: names
                .iter()
                .cloned()
                .zip(
                    row(1, "action", 5.0, 1.0, 0.0)
                        .into_iter()
                        .map(|v| TupleDatum::Text(v.unwrap())),
                )
                .collect(),
        }],
        commit_ts_us: 0,
        end_lsn: 0xAB_CDF5,
    };
    m.apply_tx(&tx).unwrap();
    m.save(&path).unwrap();
    let second = std::fs::read(&path).unwrap();
    assert_ne!(first, second, "state change must change the bytes");
    let l = Mirror::load(&path).unwrap();
    assert_eq!(l.n_rows(), 3);
    assert_eq!(l.last_lsn, 0xAB_CDF5);

    // no .tmp.* litter next to the target
    let dir = path.parent().unwrap();
    let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !(name.starts_with(&stem) && name.contains(".tmp.")),
            "temp file left behind: {name}"
        );
    }
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn corrupted_byte_is_a_typed_error() {
    let path = tmp_path("corrupt");
    seeded_mirror().save(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();
    match Mirror::load(&path) {
        Err(CdcError::Decode(msg)) => assert!(msg.contains("checksum"), "got: {msg}"),
        Err(e) => panic!("corrupt file must be Decode error, got {e}"),
        Ok(_) => panic!("corrupt file must not load"),
    }
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn truncated_file_is_a_typed_error() {
    let path = tmp_path("truncated");
    seeded_mirror().save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    for cut in [0, 4, 8, 20, bytes.len() / 2, bytes.len() - 1] {
        std::fs::write(&path, &bytes[..cut]).unwrap();
        match Mirror::load(&path) {
            Err(CdcError::Decode(_)) => {}
            Err(e) => panic!("truncation at {cut} must be Decode error, got {e}"),
            Ok(_) => panic!("truncation at {cut} must not load"),
        }
    }
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn wrong_magic_and_missing_file_are_typed_errors() {
    let path = tmp_path("magic");
    match Mirror::load(&path) {
        Err(CdcError::Io(_)) => {}
        Err(e) => panic!("missing file must be Io error, got {e}"),
        Ok(_) => panic!("missing file must not load"),
    }
    std::fs::write(&path, b"NOTBRCDC0123456789moredata").unwrap();
    match Mirror::load(&path) {
        Err(CdcError::Decode(msg)) => assert!(msg.contains("magic"), "got: {msg}"),
        Err(e) => panic!("wrong magic must be Decode error, got {e}"),
        Ok(_) => panic!("wrong magic must not load"),
    }
    std::fs::remove_file(&path).unwrap();
}

//! Workstream 20 (memory-safety audit): registration must be total.
//!
//! `Table.columns` and `Column` are public, so a hand-built table can
//! reach `Database::register` (and `TableStats::collect`) carrying
//! DictU32 codes outside `[0, dict.len())` — a dangling code is the
//! columnar analogue of a broken foreign key from `codes` to `dict`.
//! PG's ANALYZE never fails a table it merely samples; stats
//! collection here must not panic either. Corruption is surfaced by
//! the query-time layers as typed errors, never by an index panic in
//! the stats collector.

use std::collections::HashMap;

use bruce_query::{Column, Database, Table, TableStats};

/// A 4-row table whose "genre" codes contain 7 and u32::MAX, both
/// outside [0, 2).
fn corrupt_table() -> Table {
    let mut columns = HashMap::new();
    columns.insert(
        "genre".to_string(),
        Column::DictU32 {
            codes: vec![0, 1, 7, u32::MAX],
            dict: vec!["a".into(), "b".into()],
        },
    );
    columns.insert(
        "rating".to_string(),
        Column::ScalarF64(vec![1.0, 2.0, 3.0, 4.0]),
    );
    Table { columns }
}

#[test]
fn register_with_out_of_range_dict_codes_does_not_panic() {
    let mut db = Database::new();
    db.register("t", corrupt_table()); // panicked before the guard
                                       // The table is registered and stats exist for it.
    assert!(db.catalog.tables.contains_key("t"));
    assert!(db.stats.contains_key("t"));
}

#[test]
fn out_of_range_codes_are_not_attributed_to_any_group() {
    let stats = TableStats::collect(&corrupt_table(), 16);
    let d = &stats.dicts["genre"];
    // n_groups stays the dictionary's size; the two dangling rows are
    // counted nowhere (their label is unknowable), so the per-group
    // counts under-cover n_rows — conservative for selectivity.
    assert_eq!(d.n_groups, 2);
    assert_eq!(d.group_counts, vec![1, 1]);
    assert_eq!(stats.n_rows, 4);
    let counted: u64 = d.group_counts.iter().sum();
    assert!(counted as usize <= stats.n_rows);
}

/// NaN keys are the engine's NULL encoding (bruce-core mask.rs skips
/// NaN rows). The sketch's `sims()` used `partial_cmp().unwrap()`,
/// which panics on a NaN similarity — reachable by registering a
/// table whose key column contains a NaN row and asking for a
/// contract estimate (any budgeted SOFTAVG plans through this).
#[test]
fn nan_keys_do_not_panic_contract_estimation() {
    let mut columns = HashMap::new();
    columns.insert(
        "emb".to_string(),
        Column::KeyF64(ndarray::arr2(&[[1.0, 0.0], [f64::NAN, 0.5], [0.0, 1.0]])),
    );
    columns.insert("rating".to_string(), Column::ScalarF64(vec![1.0, 2.0, 3.0]));
    let stats = TableStats::collect(&Table { columns }, 16);
    let x = ndarray::arr1(&[1.0, 0.0]);
    // panicked before the NaN skip in sims()
    let est = stats.keys["emb"].estimate_contract(&x.view(), 0.3, 0.5, 1.0, 3);
    assert!(est.kstar <= 3);
}

/// All-NaN keys: every sampled similarity is skipped, the sketch has
/// nothing to certify with, and the estimate must say so
/// (`resolution_limited`) so the planner falls back to the exact
/// plan — conservative by construction, never a panic.
#[test]
fn all_nan_keys_yield_resolution_limited_estimate() {
    let mut columns = HashMap::new();
    columns.insert(
        "emb".to_string(),
        Column::KeyF64(ndarray::arr2(&[[f64::NAN, f64::NAN], [f64::NAN, f64::NAN]])),
    );
    columns.insert("rating".to_string(), Column::ScalarF64(vec![1.0, 2.0]));
    let stats = TableStats::collect(&Table { columns }, 16);
    let x = ndarray::arr1(&[1.0, 0.0]);
    let est = stats.keys["emb"].estimate_contract(&x.view(), 0.3, 0.5, 1.0, 2);
    assert!(est.resolution_limited);
    assert_eq!(est.kstar, 2); // the whole input: no contract admitted
}

/// End to end: a budgeted SOFTAVG over a table with a NaN key row
/// must return an answer (or a typed error), not panic while the
/// planner consults the sketch.
#[test]
fn budgeted_query_over_nan_keys_is_total() {
    let mut columns = HashMap::new();
    columns.insert(
        "genre".to_string(),
        Column::DictU32 {
            codes: vec![0, 0, 1],
            dict: vec!["a".into(), "b".into()],
        },
    );
    columns.insert("rating".to_string(), Column::ScalarF64(vec![1.0, 2.0, 3.0]));
    columns.insert(
        "emb".to_string(),
        Column::KeyF64(ndarray::arr2(&[[1.0, 0.0], [f64::NAN, 0.5], [0.0, 1.0]])),
    );
    let mut db = Database::new();
    db.register("t", Table { columns });
    let mut params = HashMap::new();
    params.insert("q".to_string(), ndarray::arr1(&[1.0, 0.0]));
    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3, 0.5) FROM t GROUP BY genre";
    // A typed error is acceptable; a panic is not.
    let _ = db.run(sql, &params);
}

//! Workstream 10 — error-path totality across the public bruce-query
//! surface: `parse_query`, `optimize`, the planner (driven through
//! `Database::run`; there is no separate `physical::lower` — physical
//! plans are enumerated inside `planner::plan`), `execute`,
//! `Database::{register, run, insert_row, delete_where, create_view}`,
//! `Table::{from_parquet, attach_key_f64, attach_key_f32}`.
//!
//! Every adversarial case must return `Err` — never panic. Each call
//! is wrapped in `catch_unwind` so a panic fails the test with the
//! case's name rather than aborting the suite.
//!
//! Semantics pinned/defined by this suite (see db.rs / views.rs
//! comments):
//!   - maintained views serve KeyF64 AND KeyF32 columns, through the
//!     WHOLE write path — create/read/insert/delete (flipped
//!     2026-08-03: the reads by the f32-tail track, delete_where by
//!     the hnsw-finish track; see create_view_f32_semantics);
//!   - maintained views require finite eps > 0 (the eps=0 tropical
//!     endpoint and invalid eps are rejected at create_view);
//!   - view names are unique per database (duplicate create errors);
//!   - `Database::register` over an existing name REPLACES the table
//!     and DROPS maintained views built on it (CREATE OR REPLACE +
//!     cascade, PG DROP..CASCADE vocabulary);
//!   - `insert_row` rejects rows naming unknown or ill-kinded columns
//!     (PG: INSERT naming a nonexistent column is an error);
//!   - `Database::run` returns Err (not panic) on bound query vectors
//!     whose dimension mismatches the key column, and on tables whose
//!     dict codes exceed the dictionary (catalog-invariant violation
//!     via the pub catalog fields).
//!
//! Formerly-listed residual gap, now CLOSED (2026-08-03, hnsw track):
//! `execute` on a HAND-BUILT `TopKContractScan` whose bound param
//! dimension mismatches the key column used to panic inside its sims
//! loop (ndarray dot assert). exec.rs's `check_param_dim` makes it a
//! typed Bind error; it is driven in tests/topk_access_path.rs
//! (`hand_built_topk_contract_scan_dim_mismatch_is_typed_error`),
//! which owns that operator's totality suite.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use ndarray::{Array1, Array2};

use bruce_query::db::RowValues;
use bruce_query::logical::{ScoreExpr, SimKind};
use bruce_query::{
    execute, optimize, parse_query, Column, Database, PhysicalPlan, Pred, QueryError, Table,
};

// ---------------------------------------------------------- helpers

/// Assert the closure returns Err and does not panic.
fn must_err<T, F>(ctx: &str, f: F)
where
    F: FnOnce() -> Result<T, QueryError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(_)) => panic!("{ctx}: expected Err, got Ok"),
        Ok(Err(_)) => {}
        Err(_) => panic!("{ctx}: PANICKED (totality violation)"),
    }
}

/// Assert the closure returns Err whose message contains `needle`.
fn must_err_containing<T, F>(ctx: &str, needle: &str, f: F)
where
    F: FnOnce() -> Result<T, QueryError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(_)) => panic!("{ctx}: expected Err, got Ok"),
        Ok(Err(e)) => assert!(
            e.to_string().contains(needle),
            "{ctx}: error {e:?} does not mention {needle:?}"
        ),
        Err(_) => panic!("{ctx}: PANICKED (totality violation)"),
    }
}

fn toy_table() -> Table {
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: vec![0, 0, 1, 1],
            dict: vec!["A".into(), "B".into()],
        },
    );
    t.columns
        .insert("rating".into(), Column::ScalarF64(vec![1.0, 3.0, 0.5, 6.0]));
    t.columns
        .insert("id".into(), Column::ScalarF64(vec![0.0, 1.0, 2.0, 3.0]));
    t.columns.insert(
        "emb".into(),
        Column::KeyF64(
            Array2::from_shape_vec((4, 2), vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9]).unwrap(),
        ),
    );
    t
}

fn toy_db() -> Database {
    let mut db = Database::new();
    db.register("movies", toy_table());
    db
}

fn q2() -> HashMap<String, Array1<f64>> {
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0]));
    p
}

const OK_SQL: &str = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
                      FROM movies GROUP BY genre";

// ------------------------------------------------------ parse_query

#[test]
fn parse_rejects_malformed_sql() {
    let cases: &[(&str, &str)] = &[
        ("garbage", "((( not sql"),
        ("empty", ""),
        ("two statements", "SELECT 1; SELECT 2"),
        ("not a query", "INSERT INTO t VALUES (1)"),
        ("no SOFTAVG", "SELECT genre FROM movies GROUP BY genre"),
        (
            "SOFTAVG arity 1",
            "SELECT genre, SOFTAVG(rating) FROM movies GROUP BY genre",
        ),
        (
            "SOFTAVG arity 5",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3, 0.1, 9) \
             FROM movies GROUP BY genre",
        ),
        (
            "no GROUP BY",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) FROM movies",
        ),
        (
            "two group cols",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
             FROM movies GROUP BY genre, id",
        ),
        (
            "unknown score fn",
            "SELECT genre, SOFTAVG(rating, COSINE(emb, :q), 0.3) \
             FROM movies GROUP BY genre",
        ),
        (
            "SIM arity",
            "SELECT genre, SOFTAVG(rating, SIM(emb), 0.3) FROM movies GROUP BY genre",
        ),
        (
            "param not a placeholder",
            "SELECT genre, SOFTAVG(rating, SIM(emb, 42), 0.3) \
             FROM movies GROUP BY genre",
        ),
        (
            "unsupported WHERE op",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
             FROM movies WHERE id < 3 GROUP BY genre",
        ),
        (
            "negative eps literal",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), -0.5) \
             FROM movies GROUP BY genre",
        ),
        (
            "eps is a random identifier",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), banana) \
             FROM movies GROUP BY genre",
        ),
        (
            "two tables in FROM",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
             FROM movies, others GROUP BY genre",
        ),
        (
            "subquery in FROM",
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
             FROM (SELECT * FROM movies) m GROUP BY genre",
        ),
    ];
    for (name, sql) in cases {
        must_err(&format!("parse_query[{name}]"), || parse_query(sql));
    }
}

#[test]
fn optimize_is_total_on_parsed_plans() {
    // optimize is infallible by signature; prove it does not panic on
    // the shapes the parser can produce, including both endpoints.
    for sql in [
        OK_SQL,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), INF) FROM movies GROUP BY genre",
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0) FROM movies GROUP BY genre",
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3, 0.01) \
         FROM movies WHERE id >= 1 GROUP BY genre",
    ] {
        let plan = parse_query(sql).unwrap();
        let r = catch_unwind(AssertUnwindSafe(|| optimize(plan)));
        assert!(r.is_ok(), "optimize panicked on {sql:?}");
    }
}

// --------------------------------------- Database::run (plan+exec)

#[test]
fn run_missing_table_and_columns_error() {
    must_err("run: missing table", || {
        toy_db().run(
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.3) FROM nosuch GROUP BY g",
            &q2(),
        )
    });
    must_err("run: missing group col", || {
        toy_db().run(
            "SELECT nocol, SOFTAVG(rating, SIM(emb, :q), 0.3) FROM movies GROUP BY nocol",
            &q2(),
        )
    });
    must_err("run: group col is scalar", || {
        toy_db().run(
            "SELECT rating, SOFTAVG(rating, SIM(emb, :q), 0.3) FROM movies GROUP BY rating",
            &q2(),
        )
    });
    must_err("run: val col is dict", || {
        toy_db().run(
            "SELECT genre, SOFTAVG(genre, SIM(emb, :q), 0.3) FROM movies GROUP BY genre",
            &q2(),
        )
    });
    must_err("run: key col missing", || {
        toy_db().run(
            "SELECT genre, SOFTAVG(rating, SIM(noemb, :q), 0.3) FROM movies GROUP BY genre",
            &q2(),
        )
    });
    must_err("run: key col is scalar", || {
        toy_db().run(
            "SELECT genre, SOFTAVG(rating, SIM(rating, :q), 0.3) FROM movies GROUP BY genre",
            &q2(),
        )
    });
    must_err("run: filter col missing", || {
        toy_db().run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
             FROM movies WHERE nocol >= 1 GROUP BY genre",
            &q2(),
        )
    });
    must_err("run: filter col is dict", || {
        toy_db().run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
             FROM movies WHERE genre >= 1 GROUP BY genre",
            &q2(),
        )
    });
}

#[test]
fn run_param_errors() {
    must_err_containing("run: unbound param", "unbound parameter", || {
        toy_db().run(OK_SQL, &HashMap::new())
    });
    // wrong-dimension query vector, exact plan
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0, 5.0]));
    must_err("run: param dim mismatch (exact)", || {
        toy_db().run(OK_SQL, &p)
    });
    // wrong-dimension query vector with a declared budget: this path
    // reaches the key sketch's estimator, which dots the sample rows
    // against the bound vector — must be Err, not a panic
    must_err("run: param dim mismatch (budget)", || {
        toy_db().run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3, 0.01) \
             FROM movies GROUP BY genre",
            &p,
        )
    });
}

#[test]
fn run_eps_inf_needs_no_key_or_param() {
    // R3 pins this: eps = INF degenerates to PlainGroupAvg, the key
    // column and the parameter leave the plan entirely.
    let (res, _) = toy_db()
        .run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :nope), INF) FROM movies GROUP BY genre",
            &HashMap::new(),
        )
        .expect("eps=INF must run without the param");
    let got: HashMap<_, _> = res
        .labels
        .iter()
        .cloned()
        .zip(res.values.iter().cloned())
        .collect();
    assert!((got["A"] - 2.0).abs() < 1e-12);
    assert!((got["B"] - 3.25).abs() < 1e-12);
}

#[test]
fn run_gid_overflow_is_an_error_not_a_panic() {
    // Corrupt the catalog through its pub fields: a dict code beyond
    // the dictionary. Both the finite-eps path (fused kernel) and the
    // eps=INF path (ExactGroupAvg) must answer with Err.
    let mut db = toy_db();
    if let Some(Column::DictU32 { codes, .. }) = db
        .catalog
        .tables
        .get_mut("movies")
        .unwrap()
        .columns
        .get_mut("genre")
    {
        codes[0] = 99;
    }
    must_err("run: gid overflow, finite eps", || db.run(OK_SQL, &q2()));
    must_err("run: gid overflow, eps=INF", || {
        db.run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), INF) FROM movies GROUP BY genre",
            &q2(),
        )
    });
}

#[test]
fn delete_on_corrupt_dict_codes_does_not_panic() {
    // A view exists; then the group column is corrupted via the pub
    // catalog fields. Deleting a row whose code exceeds the view's
    // group table must not panic (the phantom group stays hidden).
    let mut db = toy_db();
    db.create_view(
        "v",
        "movies",
        "genre",
        "rating",
        "emb",
        &Array1::from(vec![1.0, 0.0]),
        0.3,
    )
    .unwrap();
    if let Some(Column::DictU32 { codes, .. }) = db
        .catalog
        .tables
        .get_mut("movies")
        .unwrap()
        .columns
        .get_mut("genre")
    {
        codes[0] = 99;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        db.delete_where("movies", &Pred::Eq("id".into(), 0.0))
    }));
    match r {
        Ok(Ok(n)) => assert_eq!(n, 1, "one row matched the delete"),
        Ok(Err(_)) => {} // an error is acceptable; a panic is not
        Err(_) => panic!("delete on corrupt dict codes PANICKED"),
    }
}

// ------------------------------------------- Database write surface

#[test]
fn insert_row_errors() {
    let full_row = || RowValues {
        scalars: [("rating".to_string(), 5.0), ("id".to_string(), 9.0)]
            .into_iter()
            .collect(),
        labels: [("genre".to_string(), "A".to_string())]
            .into_iter()
            .collect(),
        keys: [("emb".to_string(), vec![0.5, 0.5])].into_iter().collect(),
    };
    must_err("insert: missing table", || {
        toy_db().insert_row("nosuch", &full_row())
    });
    must_err("insert: missing scalar col", || {
        let mut r = full_row();
        r.scalars.remove("rating");
        toy_db().insert_row("movies", &r)
    });
    must_err("insert: missing label col", || {
        let mut r = full_row();
        r.labels.remove("genre");
        toy_db().insert_row("movies", &r)
    });
    must_err("insert: missing key col", || {
        let mut r = full_row();
        r.keys.remove("emb");
        toy_db().insert_row("movies", &r)
    });
    must_err("insert: wrong key dim", || {
        let mut r = full_row();
        r.keys.insert("emb".to_string(), vec![0.5, 0.5, 0.5]);
        toy_db().insert_row("movies", &r)
    });
    // Defined semantics: a row naming a column the table does not have
    // (or of another kind) is an error, like PG's INSERT.
    must_err_containing("insert: unknown scalar col", "unknown", || {
        let mut r = full_row();
        r.scalars.insert("typo_col".to_string(), 1.0);
        toy_db().insert_row("movies", &r)
    });
    must_err_containing("insert: label for a scalar col", "unknown", || {
        let mut r = full_row();
        r.labels.insert("rating".to_string(), "oops".to_string());
        toy_db().insert_row("movies", &r)
    });
    must_err_containing("insert: key for a dict col", "unknown", || {
        let mut r = full_row();
        r.keys.insert("genre".to_string(), vec![0.1, 0.2]);
        toy_db().insert_row("movies", &r)
    });
}

#[test]
fn delete_where_errors() {
    must_err("delete: missing table", || {
        toy_db().delete_where("nosuch", &Pred::Eq("id".into(), 1.0))
    });
    must_err("delete: missing column", || {
        toy_db().delete_where("movies", &Pred::Eq("nocol".into(), 1.0))
    });
    must_err_containing("delete: dict column", "ScalarF64", || {
        toy_db().delete_where("movies", &Pred::Eq("genre".into(), 0.0))
    });
    must_err_containing("delete: key column", "ScalarF64", || {
        toy_db().delete_where("movies", &Pred::GtEq("emb".into(), 0.0))
    });
}

#[test]
fn create_view_errors() {
    let x = Array1::from(vec![1.0, 0.0]);
    must_err("view: missing table", || {
        toy_db().create_view("v", "nosuch", "genre", "rating", "emb", &x, 0.3)
    });
    must_err("view: missing group col", || {
        toy_db().create_view("v", "movies", "nocol", "rating", "emb", &x, 0.3)
    });
    must_err_containing("view: group col not dict", "DictU32", || {
        toy_db().create_view("v", "movies", "rating", "rating", "emb", &x, 0.3)
    });
    must_err_containing("view: val col not scalar", "ScalarF64", || {
        toy_db().create_view("v", "movies", "genre", "genre", "emb", &x, 0.3)
    });
    must_err("view: key col missing", || {
        toy_db().create_view("v", "movies", "genre", "rating", "noemb", &x, 0.3)
    });
    // FLIPPED 2026-08-03 (f32-tail): maintained views now serve KeyF32
    // columns too (views.rs: f32 scoring, f64 state); the former typed
    // refusal is gone. See create_view_f32_semantics below for the
    // positive pin over the whole write path (delete_where included
    // since 2026-08-03, hnsw-finish track).
    {
        let mut db = toy_db();
        let n = 4;
        db.catalog
            .tables
            .get_mut("movies")
            .unwrap()
            .attach_key_f32("emb32", Array2::<f32>::zeros((n, 2)))
            .unwrap();
        db.create_view("v", "movies", "genre", "rating", "emb32", &x, 0.3)
            .expect("create_view over a KeyF32 column must succeed");
    }
    // Defined semantics: query-vector dimension must match the key
    // column (previously an ndarray panic inside the build fold).
    must_err("view: x dim mismatch", || {
        let x3 = Array1::from(vec![1.0, 0.0, 0.0]);
        toy_db().create_view("v", "movies", "genre", "rating", "emb", &x3, 0.3)
    });
    // Defined semantics: eps must be a valid temperature and > 0 —
    // the incremental (m, num, den) maintenance is the anchored
    // softmax; the eps=0 tropical endpoint has no incremental form
    // here and previously silently produced NaN state.
    must_err("view: eps negative", || {
        toy_db().create_view("v", "movies", "genre", "rating", "emb", &x, -1.0)
    });
    must_err("view: eps NaN", || {
        toy_db().create_view("v", "movies", "genre", "rating", "emb", &x, f64::NAN)
    });
    must_err_containing("view: eps zero", "eps", || {
        toy_db().create_view("v", "movies", "genre", "rating", "emb", &x, 0.0)
    });
    // Defined semantics: view names are unique.
    must_err_containing("view: duplicate name", "exists", || {
        let mut db = toy_db();
        db.create_view("v", "movies", "genre", "rating", "emb", &x, 0.3)
            .unwrap();
        db.create_view("v", "movies", "genre", "rating", "emb", &x, 0.7)
    });
}

/// f32-view facade semantics (2026-08-03, f32-tail track; delete arm
/// FLIPPED POSITIVE the same night by the hnsw-finish track). The view
/// layer itself is dtype-complete (views.rs mod tests): build, insert
/// deltas, group-inverse deletes with re-anchor all work over KeyF32.
/// Through the Database facade:
///   * create_view + insert_row on an f32 key column work end-to-end;
///   * delete_where on a table with an f32-keyed VIEW now SUCCEEDS and
///     leaves the view exactly equal to a from-scratch rebuild over
///     the post-delete table — db.rs's per-view survivor capture reads
///     keys through the dtype-polymorphic `key_rows_f64` (was the
///     KeyF64-only `views::key_col_of`, which made this a typed
///     error). The f32 -> f64 -> f32 round trip through the wire
///     format is bit-exact, so "equal" here is EXACT, not approximate.
#[test]
fn create_view_f32_semantics() {
    let mut db = toy_db();
    let n = db.catalog.tables["movies"].columns["rating"].len();
    let mut emb32 = Array2::<f32>::zeros((n, 2));
    for r in 0..n {
        emb32[(r, 0)] = 1.0 - 0.1 * r as f32;
        emb32[(r, 1)] = 0.1 * r as f32;
    }
    db.catalog
        .tables
        .get_mut("movies")
        .unwrap()
        .attach_key_f32("emb32", emb32)
        .unwrap();
    let x = Array1::from(vec![1.0, 0.0]);
    db.create_view("v32", "movies", "genre", "rating", "emb32", &x, 0.3)
        .expect("create_view over KeyF32");
    // insert delta flows through the f32 scoring path
    let mut row = RowValues::default();
    for (name, col) in &db.catalog.tables["movies"].columns {
        match col {
            Column::ScalarF64(_) => {
                row.scalars.insert(name.clone(), 1.0);
            }
            Column::DictU32 { .. } => {
                row.labels.insert(name.clone(), "X".into());
            }
            Column::KeyF64(a) => {
                row.keys.insert(name.clone(), vec![0.5; a.ncols()]);
            }
            Column::KeyF32(a) => {
                row.keys.insert(name.clone(), vec![0.5; a.ncols()]);
            }
        }
    }
    db.insert_row("movies", &row).expect("insert with f32 view");

    // FLIPPED 2026-08-03 (hnsw-finish track; was a pinned typed
    // "KeyF64" error). First delete removes id=0, the ANCHOR scorer of
    // group 0 under x=[1,0] — the bounded re-anchor pass must read the
    // surviving KeyF32 rows. Second delete removes a NON-anchor row
    // (the O(1) group-inverse arm). After each, the maintained state
    // must equal a from-scratch rebuild over the post-delete table
    // EXACTLY: the f32 -> f64 -> f32 wire round trip is bit-preserving.
    for pred in [Pred::Eq("id".into(), 0.0), Pred::GtEq("rating".into(), 6.0)] {
        let removed = db
            .delete_where("movies", &pred)
            .expect("delete_where on an f32-viewed table must succeed");
        assert_eq!(removed, 1, "{pred:?} should delete exactly one row");
        let rebuilt = bruce_query::SoftAggView::build(
            "rebuild",
            "movies",
            &db.catalog.tables["movies"],
            "genre",
            "rating",
            "emb32",
            &x.view(),
            0.3,
        )
        .expect("rebuild over the post-delete table");
        let live = db.views[0].read();
        let want = rebuilt.read();
        assert_eq!(live.len(), want.len(), "covered groups after {pred:?}");
        for ((g, got), (wg, exp)) in live.iter().zip(&want) {
            assert_eq!(g, wg, "group order after {pred:?}");
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "group {g} after {pred:?}: maintained {got} vs rebuild {exp}"
            );
        }
    }
}

#[test]
fn register_replaces_table_and_drops_its_views() {
    // Defined semantics (this suite + db.rs): registering over an
    // existing name is CREATE OR REPLACE; maintained views built on
    // the old contents are dropped (stale view state must never serve
    // answers for the new table).
    let mut db = toy_db();
    let x = Array1::from(vec![1.0, 0.0]);
    db.create_view("v", "movies", "genre", "rating", "emb", &x, 0.3)
        .unwrap();
    assert_eq!(db.views.len(), 1);

    // replacement table: same schema, one row, different content
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: vec![0],
            dict: vec!["Z".into()],
        },
    );
    t.columns
        .insert("rating".into(), Column::ScalarF64(vec![7.0]));
    t.columns.insert("id".into(), Column::ScalarF64(vec![0.0]));
    t.columns.insert(
        "emb".into(),
        Column::KeyF64(Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap()),
    );
    db.register("movies", t);

    assert!(
        db.views.is_empty(),
        "views on the replaced table must be dropped"
    );
    let (res, _) = db
        .run(OK_SQL, &q2())
        .expect("replaced table must serve queries");
    assert_eq!(res.labels, vec!["Z".to_string()]);
    assert!((res.values[0] - 7.0).abs() < 1e-12);

    // registering an unrelated table leaves other tables' views alone
    let mut db2 = toy_db();
    db2.create_view("v", "movies", "genre", "rating", "emb", &x, 0.3)
        .unwrap();
    db2.register("other", toy_table());
    assert_eq!(db2.views.len(), 1, "views on other tables must survive");
}

// -------------------------------------------------- execute (direct)

#[test]
fn execute_direct_adversarial_plans_error() {
    let db = toy_db();
    let score = ScoreExpr {
        key_col: "emb".into(),
        param: "q".into(),
        kind: SimKind::Dot,
    };

    let fused = |table: &str, group: &str, val: &str, score: ScoreExpr, eps: f64| {
        PhysicalPlan::FusedGroupScan {
            table: table.into(),
            group_col: group.into(),
            val_col: val.into(),
            score,
            eps,
            sel: None,
        }
    };

    must_err("execute: missing table", || {
        execute(
            &fused("nosuch", "genre", "rating", score.clone(), 0.3),
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err("execute: group col wrong kind", || {
        execute(
            &fused("movies", "rating", "rating", score.clone(), 0.3),
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err("execute: val col wrong kind", || {
        execute(
            &fused("movies", "genre", "genre", score.clone(), 0.3),
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err("execute: key col missing", || {
        let s = ScoreExpr {
            key_col: "noemb".into(),
            ..score.clone()
        };
        execute(
            &fused("movies", "genre", "rating", s, 0.3),
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err_containing("execute: NegSq not yet executable", "Dot", || {
        let s = ScoreExpr {
            kind: SimKind::NegSq,
            ..score.clone()
        };
        execute(
            &fused("movies", "genre", "rating", s, 0.3),
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err("execute: unbound param", || {
        execute(
            &fused("movies", "genre", "rating", score.clone(), 0.3),
            &db.catalog,
            &HashMap::new(),
            &[],
        )
    });
    must_err("execute: param dim mismatch", || {
        let mut p = HashMap::new();
        p.insert("q".to_string(), Array1::from(vec![1.0, 0.0, 0.0]));
        execute(
            &fused("movies", "genre", "rating", score.clone(), 0.3),
            &db.catalog,
            &p,
            &[],
        )
    });
    must_err("execute: invalid eps in a hand-built plan", || {
        execute(
            &fused("movies", "genre", "rating", score.clone(), -1.0),
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err("execute: sel on a dict column", || {
        let plan = PhysicalPlan::FusedGroupScan {
            table: "movies".into(),
            group_col: "genre".into(),
            val_col: "rating".into(),
            score: score.clone(),
            eps: 0.3,
            sel: Some(Pred::Eq("genre".into(), 0.0)),
        };
        execute(&plan, &db.catalog, &q2(), &[])
    });
    must_err("execute: ExactGroupAvg missing val col", || {
        let plan = PhysicalPlan::ExactGroupAvg {
            table: "movies".into(),
            group_col: "genre".into(),
            val_col: "nocol".into(),
            sel: None,
        };
        execute(&plan, &db.catalog, &q2(), &[])
    });
    must_err_containing("execute: unknown view", "no view", || {
        execute(
            &PhysicalPlan::MaintainedViewScan {
                view: "ghost".into(),
            },
            &db.catalog,
            &q2(),
            &[],
        )
    });
    must_err("execute: TopK missing table", || {
        let plan = PhysicalPlan::TopKContractScan {
            table: "nosuch".into(),
            group_col: "genre".into(),
            val_col: "rating".into(),
            score: score.clone(),
            eps: 0.3,
            sel: None,
            budget: 0.01,
            est_kstar: 2,
            est_delta: 0.1,
        };
        execute(&plan, &db.catalog, &q2(), &[])
    });
    must_err("execute: TopK unbound param", || {
        let plan = PhysicalPlan::TopKContractScan {
            table: "movies".into(),
            group_col: "genre".into(),
            val_col: "rating".into(),
            score: score.clone(),
            eps: 0.3,
            sel: None,
            budget: 0.01,
            est_kstar: 2,
            est_delta: 0.1,
        };
        execute(&plan, &db.catalog, &HashMap::new(), &[])
    });
}

#[test]
fn empty_table_registers_and_runs_without_panic() {
    // Stats collection over a 0-row table used to underflow inside the
    // key-sketch sampler; db.rs now skips sketch collection for empty
    // tables. Defined semantics: a query over an empty table succeeds
    // and covers no groups.
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: vec![],
            dict: vec![],
        },
    );
    t.columns.insert("rating".into(), Column::ScalarF64(vec![]));
    t.columns
        .insert("emb".into(), Column::KeyF64(Array2::<f64>::zeros((0, 2))));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut db = Database::new();
        db.register("movies", t);
        db.run(OK_SQL, &q2())
    }));
    match r {
        Ok(Ok((res, _))) => assert!(res.labels.is_empty(), "empty table covers no groups"),
        Ok(Err(e)) => panic!("empty table: expected empty Ok result, got Err {e}"),
        Err(_) => panic!("empty table register/run PANICKED"),
    }
}

// ------------------------------------------------ Table ingest edge

#[test]
fn from_parquet_errors() {
    must_err("parquet: missing file", || {
        Table::from_parquet("/nonexistent/path.parquet")
    });
    // garbage bytes are not a parquet footer
    let p = std::env::temp_dir().join(format!("bruce_err_totality_{}.parquet", std::process::id()));
    std::fs::write(&p, b"this is not a parquet file at all").unwrap();
    must_err("parquet: garbage file", || Table::from_parquet(&p));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn attach_key_errors() {
    must_err_containing("attach f64: row mismatch", "rows", || {
        toy_table().attach_key_f64("k", Array2::<f64>::zeros((3, 2)))
    });
    must_err_containing("attach f32: row mismatch", "rows", || {
        toy_table().attach_key_f32("k", Array2::<f32>::zeros((5, 2)))
    });
}

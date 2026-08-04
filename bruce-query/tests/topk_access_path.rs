//! Top-k access paths: executor totality for hand-built contracted
//! plans, the HNSW index lifecycle on `Database`, planner admission of
//! `HnswTopKScan`, and differential correctness of the index-served
//! fold against the exact fused scan.
//!
//! Semantics pinned here (see also planner.rs / physical.rs docs):
//! `HnswTopKScan` is admitted WITHOUT a declared error budget because
//! it is exact within the engine precision contract — the planner
//! admits it only when the predicted omitted softmax mass is <=
//! `HNSW_TAIL_TOL`, and the executor re-checks the achieved bound at
//! runtime, falling back to the exact fold when the probe misses it.

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_query::cost::HNSW_TAIL_TOL;
use bruce_query::exec::execute_with_indexes;
use bruce_query::logical::SimKind;
use bruce_query::{
    execute, Column, Database, PhysicalPlan, Pred, QueryError, RowValues, ScoreExpr, Table, Verdict,
};

// ------------------------------------------------------------ helpers

/// Single-group synthetic table: sims are designed directly
/// (key = [sim, 0, 0, 0], query = e1), sims strictly decreasing in
/// row order, values LCG pseudo-random in [0, 10).
fn single_group_table(n: usize) -> (Table, Vec<f64>, Vec<f64>) {
    let sims: Vec<f64> = (0..n)
        .map(|i| 1.0 - 2.0 * (i as f64) / (n as f64))
        .collect();
    let mut state = 88172645463325252u64;
    let vals: Vec<f64> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 1000.0
        })
        .collect();
    let mut keys = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        keys[(i, 0)] = sims[i];
    }
    let mut t = Table::default();
    t.columns.insert(
        "all".into(),
        Column::DictU32 {
            codes: vec![0; n],
            dict: vec!["all".into()],
        },
    );
    t.columns
        .insert("rating".into(), Column::ScalarF64(vals.clone()));
    t.columns.insert(
        "id".into(),
        Column::ScalarF64((0..n).map(|i| i as f64).collect()),
    );
    t.columns.insert("emb".into(), Column::KeyF64(keys));
    (t, sims, vals)
}

fn score4() -> ScoreExpr {
    ScoreExpr {
        key_col: "emb".into(),
        param: "q".into(),
        kind: SimKind::Dot,
    }
}

/// Exact max-anchored softavg over (sims, vals) restricted by `keep`.
fn softavg_ref(sims: &[f64], vals: &[f64], keep: impl Fn(usize) -> bool, eps: f64) -> f64 {
    let m = sims
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let (mut num, mut den) = (0.0, 0.0);
    for (i, (s, v)) in sims.iter().zip(vals).enumerate() {
        if keep(i) {
            let w = ((s - m) / eps).exp();
            num += w * v;
            den += w;
        }
    }
    num / den
}

// ------------------------------------- job 1: executor totality guard

/// Residual gap (TESTING_MATRIX 2026-08-03): a HAND-BUILT
/// TopKContractScan with a dim-mismatched bound param must return a
/// typed Bind error, not panic in the sims loop. (`Database::run`'s
/// validate_run rejects it earlier; `execute` is pub and must be total
/// on its own.)
#[test]
fn hand_built_topk_contract_scan_dim_mismatch_is_typed_error() {
    let (t, _, _) = single_group_table(64);
    let mut db = Database::new();
    db.register("movies", t);
    let plan = PhysicalPlan::TopKContractScan {
        table: "movies".into(),
        group_col: "all".into(),
        val_col: "rating".into(),
        score: score4(),
        eps: 0.05,
        sel: None,
        budget: 0.05,
        est_kstar: 8,
        est_delta: 1e-3,
    };
    // bound param has dim 3; the key column has dim 4
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0, 0.0]));
    let err = execute(&plan, &db.catalog, &p, &[]).unwrap_err();
    assert!(
        matches!(err, QueryError::Bind(_)),
        "expected Bind, got {err:?}"
    );
    assert!(
        err.to_string().contains("dim"),
        "message should name the dimension mismatch: {err}"
    );
}

/// Same guard for KeyF32 storage: the f32 sims loop previously
/// zip-truncated a short param silently (worse than a panic — a wrong
/// answer). Both storage dtypes take the typed error.
#[test]
fn hand_built_topk_contract_scan_dim_mismatch_f32_is_typed_error() {
    let n = 16;
    let mut t = Table::default();
    t.columns.insert(
        "all".into(),
        Column::DictU32 {
            codes: vec![0; n],
            dict: vec!["all".into()],
        },
    );
    t.columns
        .insert("rating".into(), Column::ScalarF64(vec![1.0; n]));
    t.columns.insert(
        "emb".into(),
        Column::KeyF32(ndarray::Array2::<f32>::zeros((n, 4))),
    );
    let mut db = Database::new();
    db.register("movies", t);
    let plan = PhysicalPlan::TopKContractScan {
        table: "movies".into(),
        group_col: "all".into(),
        val_col: "rating".into(),
        score: score4(),
        eps: 0.05,
        sel: None,
        budget: 0.05,
        est_kstar: 8,
        est_delta: 1e-3,
    };
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0]));
    let err = execute(&plan, &db.catalog, &p, &[]).unwrap_err();
    assert!(
        matches!(err, QueryError::Bind(_)),
        "expected Bind, got {err:?}"
    );
}

// ---------------------------------------------- hot-ring test tables

/// Single-group table with NAVIGABLE geometry (unit-norm keys on a
/// circle in the first two of `d` dims — the L2-normalized setting the
/// hnsw.rs module docs assume; the raw `[sim, 0, 0, 0]` design above
/// is unnormalized MIPS, which those docs explicitly disclaim for
/// graph search) and a SHARP top: rows `i < hot` score
/// `0.99 - 0.01 i` (concentrated softmax head), rows `i >= hot` are
/// background from 0.3 down to -0.5. Sims strictly decrease in row
/// order; `sim(row i) == id i`'s key dot e1 exactly.
fn hot_ring_table(n: usize, hot: usize, d: usize) -> (Table, Vec<f64>, Vec<f64>) {
    assert!(hot < n && d >= 2);
    let sims: Vec<f64> = (0..n)
        .map(|i| {
            if i < hot {
                0.99 - 0.01 * i as f64
            } else {
                0.3 - 0.8 * (i - hot) as f64 / (n - hot) as f64
            }
        })
        .collect();
    let mut state = 88172645463325252u64;
    let vals: Vec<f64> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 1000.0
        })
        .collect();
    let mut keys = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        keys[(i, 0)] = sims[i];
        keys[(i, 1)] = (1.0 - sims[i] * sims[i]).sqrt();
    }
    let mut t = Table::default();
    t.columns.insert(
        "bucket".into(),
        Column::DictU32 {
            codes: vec![0; n],
            dict: vec!["all".into()],
        },
    );
    t.columns
        .insert("rating".into(), Column::ScalarF64(vals.clone()));
    t.columns.insert(
        "id".into(),
        Column::ScalarF64((0..n).map(|i| i as f64).collect()),
    );
    t.columns.insert("emb".into(), Column::KeyF64(keys));
    (t, sims, vals)
}

fn qd(d: usize) -> HashMap<String, Array1<f64>> {
    let mut x = vec![0.0; d];
    x[0] = 1.0;
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(x));
    p
}

fn hnsw_plan(eps: f64, sel: Option<Pred>, k: usize, ef: usize) -> PhysicalPlan {
    PhysicalPlan::HnswTopKScan {
        table: "movies".into(),
        group_col: "bucket".into(),
        val_col: "rating".into(),
        score: score4(),
        eps,
        sel,
        k,
        ef,
        predicted_tail: 0.0,
    }
}

// ------------------------------------ job 2: index lifecycle on `Database`

#[test]
fn create_index_typed_errors() {
    let (t, _, _) = hot_ring_table(256, 64, 4);
    let mut db = Database::new();
    db.register("movies", t);

    // missing table / missing column / non-key column
    assert!(matches!(
        db.create_index("nope", "emb"),
        Err(QueryError::Bind(_))
    ));
    assert!(matches!(
        db.create_index("movies", "nope"),
        Err(QueryError::Bind(_))
    ));
    assert!(matches!(
        db.create_index("movies", "rating"),
        Err(QueryError::Bind(_))
    ));

    // KeyF32 storage: typed refusal for now (views convention)
    let mut t32 = Table::default();
    t32.columns.insert(
        "all".into(),
        Column::DictU32 {
            codes: vec![0; 4],
            dict: vec!["all".into()],
        },
    );
    t32.columns.insert(
        "emb".into(),
        Column::KeyF32(ndarray::Array2::<f32>::zeros((4, 4))),
    );
    db.register("m32", t32);
    let err = db.create_index("m32", "emb").unwrap_err();
    assert!(
        err.to_string().contains("KeyF64"),
        "f32 refusal should name the required dtype: {err}"
    );

    // duplicate CREATE INDEX
    db.create_index("movies", "emb").unwrap();
    let err = db.create_index("movies", "emb").unwrap_err();
    assert!(
        err.to_string().contains("already exists"),
        "duplicate should say so: {err}"
    );
}

#[test]
fn register_replace_drops_index_like_views() {
    let (t, _, _) = hot_ring_table(256, 64, 4);
    let (t2, _, _) = hot_ring_table(128, 32, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    assert_eq!(db.indexes.len(), 1);
    db.register("movies", t2); // CREATE OR REPLACE: cascade drops
    assert!(
        db.indexes.is_empty(),
        "replace must drop dependent indexes (stale candidates would serve the new table)"
    );
}

#[test]
fn delete_where_tombstones_index_and_results_stay_exact() {
    let n = 512;
    let (t, sims, vals) = hot_ring_table(n, 64, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    assert_eq!(db.indexes[0].tombstone_fraction(), 0.0);
    assert_eq!(db.indexes[0].len(), n);

    // delete the top TWO scorers plus a background slab
    db.delete_where("movies", &Pred::GtEq("id".into(), 384.0))
        .unwrap();
    db.delete_where("movies", &Pred::Eq("id".into(), 0.0))
        .unwrap();
    db.delete_where("movies", &Pred::Eq("id".into(), 1.0))
        .unwrap();
    let doomed = |i: usize| !(2..384).contains(&i);
    let frac = db.indexes[0].tombstone_fraction();
    assert!(
        (frac - 130.0 / 512.0).abs() < 1e-12,
        "tombstone_fraction surfaces the doomed share, got {frac}"
    );
    assert_eq!(db.indexes[0].len(), n - 130);

    // the index-served fold sees only survivors (no tombstone leaks),
    // anchored at the NEW max (old rank-2 scorer)
    let eps = 1e-3;
    let got = execute_with_indexes(
        &hnsw_plan(eps, None, 16, 64),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    let want = softavg_ref(&sims, &vals, |i| !doomed(i), eps);
    assert!(
        (got.values[0] - want).abs() <= 2.0 * HNSW_TAIL_TOL * 10.0 + 1e-12,
        "post-delete: {} vs {}",
        got.values[0],
        want
    );
}

#[test]
fn insert_row_extends_index_incrementally() {
    let n = 512;
    let (t, mut sims, mut vals) = hot_ring_table(n, 64, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();

    // new GLOBAL max scorer (sim 1.0 > designed max 0.99): if the
    // index missed it, the sharp fold would anchor on the wrong row
    let mut row = RowValues::default();
    row.labels.insert("bucket".into(), "all".into());
    row.scalars.insert("rating".into(), 9.5);
    row.scalars.insert("id".into(), n as f64);
    row.keys.insert("emb".into(), vec![1.0, 0.0, 0.0, 0.0]);
    db.insert_row("movies", &row).unwrap();
    sims.push(1.0);
    vals.push(9.5);
    assert_eq!(db.indexes[0].len(), n + 1);

    let eps = 1e-3;
    let got = execute_with_indexes(
        &hnsw_plan(eps, None, 16, 64),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    let want = softavg_ref(&sims, &vals, |_| true, eps);
    assert!(
        (got.values[0] - want).abs() <= 2.0 * HNSW_TAIL_TOL * 10.0 + 1e-12,
        "post-insert: {} vs {}",
        got.values[0],
        want
    );
}

// --------------------------- job 3: planner integration + EXPLAIN why

/// The planner's whole decision surface on one indexed 20k-row table:
/// sharp eps chooses the index probe (and EXPLAIN says why), diffuse
/// eps refuses it on the predicted tail (and EXPLAIN says why), and
/// every served answer stays within the no-budget precision contract
/// (2 * HNSW_TAIL_TOL * vmax) of the exact fold. One test fn because
/// the 20k index build is the expensive part.
#[test]
fn planner_admission_choice_and_differentials_across_eps_and_selectivity() {
    let n = 20_000;
    let (t, sims, vals) = hot_ring_table(n, 64, 8);
    let mut db = Database::new();
    db.stats_sample = 4096; // scale ~4.9: rank-16 quantiles resolvable
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    let vmax = 10.0;
    let tol_abs = 2.0 * HNSW_TAIL_TOL * vmax + 1e-12;

    // (1) sharp eps, unfiltered: HnswTopKScan chosen, k from the
    // ladder, EXPLAIN carries the admission proof
    for eps in [1e-4, 1e-3] {
        let sql = format!(
            "SELECT bucket, SOFTAVG(rating, SIM(emb, :q), {eps}) FROM movies GROUP BY bucket"
        );
        let (out, planned) = db.run(&sql, &qd(8)).unwrap();
        assert!(
            matches!(planned.chosen, PhysicalPlan::HnswTopKScan { .. }),
            "eps={eps} should choose the index probe:\n{}",
            planned.explain()
        );
        let ex = planned.explain();
        assert!(
            ex.contains("predicted tail") && ex.contains("HnswTopKScan"),
            "EXPLAIN must print the admission reason:\n{ex}"
        );
        let want = softavg_ref(&sims, &vals, |_| true, eps);
        assert!(
            (out.values[0] - want).abs() <= tol_abs,
            "eps={eps}: {} vs {} (contract 2*tol*vmax)",
            out.values[0],
            want
        );
    }

    // (2) diffuse eps: refused on predicted tail; exact plan chosen
    let (out, planned) = db
        .run(
            "SELECT bucket, SOFTAVG(rating, SIM(emb, :q), 0.1) FROM movies GROUP BY bucket",
            &qd(8),
        )
        .unwrap();
    assert!(
        matches!(planned.chosen, PhysicalPlan::FusedGroupScan { .. }),
        "diffuse eps stays exact:\n{}",
        planned.explain()
    );
    assert!(
        planned.candidates.iter().any(|c| matches!(
            (&c.plan, &c.verdict),
            (PhysicalPlan::HnswTopKScan { .. }, Verdict::Inadmissible(r)) if r.contains("tail")
        )),
        "EXPLAIN must carry the tail refusal:\n{}",
        planned.explain()
    );
    let want = softavg_ref(&sims, &vals, |_| true, 0.1);
    assert!((out.values[0] - want).abs() < 1e-9);

    // (3) mid eps 1e-2: admissible only at a larger k — whatever the
    // cost model picks, the answer must stay inside the contract
    let (out, planned) = db
        .run(
            "SELECT bucket, SOFTAVG(rating, SIM(emb, :q), 0.01) FROM movies GROUP BY bucket",
            &qd(8),
        )
        .unwrap();
    let want = softavg_ref(&sims, &vals, |_| true, 0.01);
    assert!(
        (out.values[0] - want).abs() <= tol_abs,
        "eps=1e-2 ({:?}): {} vs {}",
        planned.chosen,
        out.values[0],
        want
    );

    // (4) filtered x eps grid: the filter-aware probe serves the
    // admitted set when the achieved bound holds, the runtime
    // re-check falls back when it does not (e.g. the deep 10% filter
    // at k=16), and diffuse eps stays on the exact plan — the answer
    // contract holds at every point regardless of which path served
    for (thresh, eps) in [
        (8.0, 1e-4),
        (8.0, 1e-3),
        (10_000.0, 1e-4),
        (10_000.0, 1e-3),
        (18_000.0, 1e-4),
        (18_000.0, 1e-2),
        (10_000.0, 0.1),
    ] {
        let sql = format!(
            "SELECT bucket, SOFTAVG(rating, SIM(emb, :q), {eps}) \
             FROM movies WHERE id >= {thresh} GROUP BY bucket"
        );
        let (out, planned) = db.run(&sql, &qd(8)).unwrap();
        let want = softavg_ref(&sims, &vals, |i| (i as f64) >= thresh, eps);
        assert!(
            (out.values[0] - want).abs() <= tol_abs,
            "filtered id>={thresh} eps={eps} ({:?}): {} vs {}",
            planned.chosen,
            out.values[0],
            want
        );
    }
}

/// GROUP BY over more than one group: the index is a global top-k, so
/// v1 refuses with a typed EXPLAIN reason and stays on the fused scan.
#[test]
fn planner_refuses_index_for_group_by_shapes() {
    let n = 2000;
    let (mut t, _, _) = hot_ring_table(n, 64, 4);
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: (0..n).map(|i| (i % 2) as u32).collect(),
            dict: vec!["A".into(), "B".into()],
        },
    );
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    let (_, planned) = db
        .run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.001) FROM movies GROUP BY genre",
            &qd(4),
        )
        .unwrap();
    assert!(matches!(
        planned.chosen,
        PhysicalPlan::FusedGroupScan { .. }
    ));
    assert!(
        planned.candidates.iter().any(|c| matches!(
            (&c.plan, &c.verdict),
            (PhysicalPlan::HnswTopKScan { .. }, Verdict::Inadmissible(r))
                if r.contains("GROUP BY")
        )),
        "EXPLAIN must say WHY the index was refused:\n{}",
        planned.explain()
    );
}

// --------------------- job 3/4: executor totality + runtime fallback

#[test]
fn hand_built_hnsw_plan_without_index_is_typed_error() {
    let (t, _, _) = hot_ring_table(64, 16, 4);
    let mut db = Database::new();
    db.register("movies", t);
    // `execute` has no index registry: typed Bind, never a panic
    let err = execute(&hnsw_plan(1e-3, None, 8, 32), &db.catalog, &qd(4), &[]).unwrap_err();
    assert!(
        matches!(&err, QueryError::Bind(m) if m.contains("no hnsw index")),
        "got {err:?}"
    );
}

#[test]
fn hand_built_hnsw_plan_multi_group_is_typed_error() {
    let n = 64;
    let (mut t, _, _) = hot_ring_table(n, 16, 4);
    t.columns.insert(
        "bucket".into(), // overwrite the single group with two groups
        Column::DictU32 {
            codes: (0..n).map(|i| (i % 2) as u32).collect(),
            dict: vec!["A".into(), "B".into()],
        },
    );
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    let err = execute_with_indexes(
        &hnsw_plan(1e-3, None, 8, 32),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap_err();
    assert!(
        matches!(&err, QueryError::Exec(m) if m.contains("single-group")),
        "got {err:?}"
    );
}

#[test]
fn hand_built_hnsw_plan_eps_zero_and_nan_are_typed_errors() {
    let (t, _, _) = hot_ring_table(64, 16, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    for eps in [0.0, f64::NAN, -1.0] {
        let err = execute_with_indexes(
            &hnsw_plan(eps, None, 8, 32),
            &db.catalog,
            &qd(4),
            &[],
            &db.indexes,
        )
        .unwrap_err();
        assert!(matches!(err, QueryError::Exec(_)), "eps={eps}: got {err:?}");
    }
}

/// Runtime re-check: a hand-built probe at diffuse eps (which the
/// planner would never admit) must fall back to the exact fused fold —
/// identical answer, bit for bit, because the fallback IS that plan.
#[test]
fn runtime_tail_recheck_falls_back_to_exact() {
    let (t, _, _) = hot_ring_table(512, 64, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    let eps = 0.5; // achieved tail bound >> tol at any k << n
    let got = execute_with_indexes(
        &hnsw_plan(eps, None, 8, 32),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    let exact = execute(
        &PhysicalPlan::FusedGroupScan {
            table: "movies".into(),
            group_col: "bucket".into(),
            val_col: "rating".into(),
            score: score4(),
            eps,
            sel: None,
        },
        &db.catalog,
        &qd(4),
        &[],
    )
    .unwrap();
    assert_eq!(
        got.values, exact.values,
        "fallback must BE the exact plan (same code path, same bits)"
    );
}

/// k = 0 in a hand-built plan: the probe returns nothing; the executor
/// falls back to exact rather than dividing by an empty fold.
#[test]
fn hand_built_hnsw_plan_k_zero_falls_back() {
    let (t, sims, vals) = hot_ring_table(128, 32, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    let got = execute_with_indexes(
        &hnsw_plan(1e-3, None, 0, 0),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    let want = softavg_ref(&sims, &vals, |_| true, 1e-3);
    assert!((got.values[0] - want).abs() < 1e-9);
}

/// Empty admitted set (filter matches nothing): empty result, no
/// panic, no division by zero.
#[test]
fn hnsw_plan_with_empty_selection_returns_empty() {
    let (t, _, _) = hot_ring_table(128, 32, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();
    let got = execute_with_indexes(
        &hnsw_plan(1e-3, Some(Pred::GtEq("id".into(), 1e9)), 8, 32),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    assert!(got.labels.is_empty() && got.values.is_empty());
}

// ---------------- job 5: runtime probe diagnostics (EXPLAIN ANALYZE)

/// `execute_hnsw_with_stats` reports what the runtime re-check
/// actually saw — the number the m6_hnsw regret grid records as
/// "achieved" against the plan's `predicted_tail`. Three regimes on
/// one table: sharp eps (admitted, no fallback), diffuse eps
/// (fallback), and a probe that returns the whole admitted set (tail
/// exactly 0, nothing omitted).
#[test]
fn hnsw_probe_stats_report_the_runtime_recheck() {
    let n = 512;
    let (t, sims, vals) = hot_ring_table(n, 64, 4);
    let mut db = Database::new();
    db.register("movies", t);
    db.create_index("movies", "emb").unwrap();

    // (a) sharp eps: the bound clears the tolerance, no fallback
    let (got, st) = bruce_query::exec::execute_hnsw_with_stats(
        &hnsw_plan(1e-3, None, 16, 64),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    assert!(
        !st.fell_back,
        "sharp eps should serve from the probe: {st:?}"
    );
    assert_eq!(st.admitted_rows, n);
    assert_eq!(st.hits, 16);
    assert!(
        st.achieved_tail <= HNSW_TAIL_TOL,
        "no fallback implies the achieved bound cleared tol: {st:?}"
    );
    // the reported extremes are the probe's own rescored sims
    assert!(st.s_max >= st.s_k);
    assert!(
        st.s_max <= sims[0] + 1e-12,
        "s_max cannot beat the true max"
    );
    // and the answer is the k-row fold, close to exact at this eps
    let want = softavg_ref(&sims, &vals, |_| true, 1e-3);
    assert!(
        (got.values[0] - want).abs() <= 2.0 * st.achieved_tail.max(1e-12) * 10.0 + 1e-9,
        "answer {} vs exact {want}, achieved tail {}",
        got.values[0],
        st.achieved_tail
    );

    // (b) diffuse eps: fallback, and the answer IS the exact plan
    let (got, st) = bruce_query::exec::execute_hnsw_with_stats(
        &hnsw_plan(0.5, None, 16, 64),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    assert!(st.fell_back, "diffuse eps must miss the bound: {st:?}");
    assert!(st.achieved_tail > HNSW_TAIL_TOL);
    let exact = execute(
        &PhysicalPlan::FusedGroupScan {
            table: "movies".into(),
            group_col: "bucket".into(),
            val_col: "rating".into(),
            score: score4(),
            eps: 0.5,
            sel: None,
        },
        &db.catalog,
        &qd(4),
        &[],
    )
    .unwrap();
    assert_eq!(got.values, exact.values);

    // (c) probe covers the whole admitted set: no omitted mass at all
    let (_, st) = bruce_query::exec::execute_hnsw_with_stats(
        &hnsw_plan(0.5, None, n, 4 * n),
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap();
    assert_eq!(st.hits, n, "ef >> n should return every row");
    assert_eq!(
        st.achieved_tail, 0.0,
        "omitting nothing is a tail of exactly 0: {st:?}"
    );
    assert!(!st.fell_back);
}

/// The diagnostics entry point is `HnswTopKScan`-only: any other plan
/// variant is a typed Bind error, not a silent wrong answer.
#[test]
fn hnsw_probe_stats_reject_other_plan_variants() {
    let (t, _, _) = hot_ring_table(64, 16, 4);
    let mut db = Database::new();
    db.register("movies", t);
    let err = bruce_query::exec::execute_hnsw_with_stats(
        &PhysicalPlan::FusedGroupScan {
            table: "movies".into(),
            group_col: "bucket".into(),
            val_col: "rating".into(),
            score: score4(),
            eps: 1e-3,
            sel: None,
        },
        &db.catalog,
        &qd(4),
        &[],
        &db.indexes,
    )
    .unwrap_err();
    assert!(matches!(err, QueryError::Bind(_)), "got {err:?}");
}

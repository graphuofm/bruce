//! Workstream 8 — EXPLAIN golden strings + design-doc invariants.
//!
//! Golden tests pin the exact EXPLAIN rendering of the physical plans
//! (fused with pushdown, no-filter, the R3 endpoint, the contracted
//! scan, the view scan) and the SHAPE of `PlannedQuery::explain()`'s
//! candidate table (stable section headers and per-line fields — not
//! exact cost floats, which are model-calibration dependent).
//!
//! Design invariants encoded as NEGATIVE tests:
//! R1 legality — a predicate naming the score's key column must NOT be
//! pushed below the aggregate, and the query layer must reject (with a
//! typed error, never a panic or a silent wrong answer) any plan or
//! query that filters on a non-scalar column. Current behavior pinned
//! here and documented in optimizer.rs.

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_query::logical::SimKind;
use bruce_query::{
    optimize, plan, Column, Database, LogicalPlan, PhysicalPlan, Pred, QueryError, ScoreExpr, Table,
};

fn toy_db() -> Database {
    let mut t = Table::default();
    t.columns.insert(
        "g".into(),
        Column::DictU32 {
            codes: vec![0, 0, 1, 1],
            dict: vec!["A".into(), "B".into()],
        },
    );
    t.columns
        .insert("v".into(), Column::ScalarF64(vec![1.0, 2.0, 3.0, 4.0]));
    t.columns
        .insert("y".into(), Column::ScalarF64(vec![1.0, 2.0, 3.0, 4.0]));
    t.columns.insert(
        "k".into(),
        Column::KeyF64(
            Array2::from_shape_vec((4, 2), vec![0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8]).unwrap(),
        ),
    );
    let mut db = Database::new();
    db.register("t", t);
    db
}

fn params() -> HashMap<String, Array1<f64>> {
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0]));
    p
}

// ------------------------------------------------------------ goldens

#[test]
fn golden_fused_plan_with_pushdown() {
    let mut db = toy_db();
    let (_, planned) = db
        .run(
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.25) FROM t WHERE y >= 2.5 GROUP BY g",
            &params(),
        )
        .unwrap();
    let expect = "FusedGroupScan[eps=0.25] kernel=grouped_softavg\n  \
                  group=g val=v score=Dot(k,:q)\n  \
                  fused Filter[eps=0]: GtEq(\"y\", 2.5)  (rows never scored)\n  \
                  Scan t";
    assert_eq!(planned.chosen.explain(), expect);
}

#[test]
fn golden_fused_plan_no_filter() {
    let mut db = toy_db();
    let (_, planned) = db
        .run(
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.25) FROM t GROUP BY g",
            &params(),
        )
        .unwrap();
    let expect = "FusedGroupScan[eps=0.25] kernel=grouped_softavg\n  \
                  group=g val=v score=Dot(k,:q)\n  \
                  Scan t";
    assert_eq!(planned.chosen.explain(), expect);
}

#[test]
fn golden_r3_endpoint_plan() {
    let mut db = toy_db();
    let (_, planned) = db
        .run(
            "SELECT g, SOFTAVG(v, SIM(k, :q), INF) FROM t WHERE y >= 2.5 GROUP BY g",
            &params(),
        )
        .unwrap();
    let expect = "ExactGroupAvg[eps=inf endpoint] group=g val=v\n  \
                  (R3: scoring dropped; key column not read)\n  \
                  fused Filter[eps=0]: GtEq(\"y\", 2.5)\n  \
                  Scan t";
    assert_eq!(planned.chosen.explain(), expect);
}

#[test]
fn golden_topk_contract_scan_rendering() {
    // constructed directly: est_* fields are planner estimates, fixed
    // here so the rendering (incl. the {:.3e} delta) is deterministic
    let p = PhysicalPlan::TopKContractScan {
        table: "t".into(),
        group_col: "g".into(),
        val_col: "v".into(),
        score: ScoreExpr {
            key_col: "k".into(),
            param: "q".into(),
            kind: SimKind::Dot,
        },
        eps: 0.02,
        sel: None,
        budget: 0.05,
        est_kstar: 37,
        est_delta: 0.001953,
    };
    let expect = "TopKContractScan[eps=0.02] budget=0.05 est k*=37 est delta=1.953e-3\n  \
                  group=g val=v score=Dot(k,:q)\n  \
                  Scan t (sims stream all keys)";
    assert_eq!(p.explain(), expect);
}

#[test]
fn golden_maintained_view_scan_rendering() {
    let p = PhysicalPlan::MaintainedViewScan { view: "v1".into() };
    assert_eq!(p.explain(), "MaintainedViewScan view=v1  (O(groups) read)");
}

// ---------------------------------------------- candidate-table shape

/// `PlannedQuery::explain()` shape contract: header lines are stable,
/// exactly one candidate is marked chosen, and every candidate line
/// carries the est/ms/MB fields. Floats themselves are NOT pinned.
#[test]
fn planned_query_explain_candidate_table_shape() {
    let mut db = toy_db();
    let (_, planned) = db
        .run(
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.25) FROM t GROUP BY g",
            &params(),
        )
        .unwrap();
    let text = planned.explain();
    let lines: Vec<&str> = text.lines().collect();

    assert_eq!(lines[0], "== chosen plan ==");
    let ci = lines
        .iter()
        .position(|l| *l == "== candidates ==")
        .expect("candidates header present");
    let cand_lines = &lines[ci + 1..];
    assert_eq!(cand_lines.len(), planned.candidates.len());
    assert_eq!(
        cand_lines
            .iter()
            .filter(|l| l.starts_with("-> chosen   "))
            .count(),
        1,
        "exactly one chosen candidate:\n{text}"
    );
    for l in cand_lines {
        assert!(
            l.contains(" est ") && l.contains(" ms  (") && l.contains(" MB, "),
            "candidate line missing cost fields: {l:?}"
        );
    }
    // no budget declared -> the contracted plan must not be enumerated
    assert!(
        !text.contains("TopKContractScan"),
        "unbudgeted query enumerated a contract:\n{text}"
    );
}

/// With a declared budget the contracted candidate appears in the
/// table (whatever its verdict), under the same line shape.
#[test]
fn planned_query_explain_lists_contract_candidate_under_budget() {
    let mut db = toy_db();
    let (_, planned) = db
        .run(
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.02, 0.05) FROM t GROUP BY g",
            &params(),
        )
        .unwrap();
    let text = planned.explain();
    assert!(
        text.contains("TopKContractScan"),
        "budgeted query must enumerate the contracted candidate:\n{text}"
    );
}

// ------------------------------------------------- R1 negative cases

/// R1 legality NEGATIVE case, optimizer level: a Filter whose
/// predicate names the score's key column stays ABOVE the aggregate
/// (optimize refuses the push), and the planner then refuses the whole
/// plan rather than planning around an illegal shape. Pinned behavior,
/// documented in optimizer.rs.
#[test]
fn r1_negative_key_col_filter_is_not_pushed_and_fails_to_plan() {
    let illegal = LogicalPlan::Filter {
        pred: Pred::GtEq("k".into(), 0.5),
        input: Box::new(LogicalPlan::SoftAgg {
            group_col: "g".into(),
            val_col: "v".into(),
            score: ScoreExpr {
                key_col: "k".into(),
                param: "q".into(),
                kind: SimKind::Dot,
            },
            eps: 0.25,
            budget: None,
            input: Box::new(LogicalPlan::Scan { table: "t".into() }),
        }),
    };
    let kept = optimize(illegal);
    assert!(
        matches!(kept, LogicalPlan::Filter { .. }),
        "filter on the score's key column must NOT push below the aggregate, got {kept:?}"
    );

    let db = toy_db();
    let err = plan(&kept, &db.stats["t"], &[], &params(), &db.model).unwrap_err();
    assert!(
        err.to_string().contains("plans must end in an aggregate"),
        "unexpected planner error: {err}"
    );
}

/// R1 legality NEGATIVE case, pipeline level: `WHERE` on the key
/// column parses (the grammar is type-agnostic) but execution rejects
/// it with a typed Bind error — eps = 0 selection is defined over
/// ScalarF64 columns only; key and dictionary columns are not legal
/// filter inputs. Never a panic, never a silent wrong answer.
#[test]
fn r1_negative_pipeline_rejects_filter_on_key_and_dict_columns() {
    for (col, q) in [
        (
            "k",
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.25) FROM t WHERE k >= 0.5 GROUP BY g",
        ),
        (
            "g",
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.25) FROM t WHERE g >= 1 GROUP BY g",
        ),
        // absent columns take the same typed rejection (message pinned
        // as-is: the filter surface is "ScalarF64 or nothing")
        (
            "nosuch",
            "SELECT g, SOFTAVG(v, SIM(k, :q), 0.25) FROM t WHERE nosuch >= 1 GROUP BY g",
        ),
    ] {
        let mut db = toy_db();
        let err = db.run(q, &params()).unwrap_err();
        assert!(
            matches!(err, QueryError::Bind(_)),
            "column {col}: expected a Bind error, got {err:?}"
        );
        assert!(
            err.to_string().contains("must be ScalarF64"),
            "column {col}: unexpected message: {err}"
        );
    }
}

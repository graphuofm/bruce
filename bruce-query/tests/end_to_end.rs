//! End-to-end and rule-equivalence tests for the query layer (v2 API:
//! Database facade, planner enumeration).

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_query::{optimize, parse_query, Column, Database, LogicalPlan, Table};

fn toy_table() -> Table {
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: vec![0, 0, 0, 1, 1, 1],
            dict: vec!["A".into(), "B".into()],
        },
    );
    t.columns.insert(
        "rating".into(),
        Column::ScalarF64(vec![1.0, 3.0, 8.0, 0.5, 6.0, 2.0]),
    );
    t.columns.insert(
        "year".into(),
        Column::ScalarF64(vec![2001.0, 2005.0, 1999.0, 2010.0, 2003.0, 1995.0]),
    );
    t.columns.insert(
        "emb".into(),
        Column::KeyF64(
            Array2::from_shape_vec(
                (6, 2),
                vec![
                    0.99, 0.10, 0.95, 0.30, 0.05, 0.99, 0.90, 0.40, 0.10, 0.99, 0.70, 0.70,
                ],
            )
            .unwrap(),
        ),
    );
    t
}

fn toy_db() -> Database {
    let mut db = Database::new();
    db.register("movies", toy_table());
    db
}

fn params() -> HashMap<String, Array1<f64>> {
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0]));
    p
}

fn softmax_ref(sims: &[f64], vals: &[f64], eps: f64) -> f64 {
    let m = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut num = 0.0;
    let mut den = 0.0;
    for (s, v) in sims.iter().zip(vals) {
        let w = ((s - m) / eps).exp();
        num += w * v;
        den += w;
    }
    num / den
}

#[test]
fn pipeline_parses_optimizes_plans_executes() {
    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
               FROM movies WHERE year >= 2000 GROUP BY genre";
    let (out, planned) = toy_db().run(sql, &params()).unwrap();

    let a = softmax_ref(&[0.99, 0.95], &[1.0, 3.0], 0.3);
    let b = softmax_ref(&[0.90, 0.10], &[0.5, 6.0], 0.3);
    let got: HashMap<_, _> = out
        .labels
        .iter()
        .cloned()
        .zip(out.values.iter().cloned())
        .collect();
    assert!((got["A"] - a).abs() < 1e-12);
    assert!((got["B"] - b).abs() < 1e-12);
    assert!(planned.explain().contains("chosen"));
}

#[test]
fn explain_shows_fused_pushdown_and_candidates() {
    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
               FROM movies WHERE year >= 2000 GROUP BY genre";
    let (_, planned) = toy_db().run(sql, &params()).unwrap();
    let text = planned.explain();
    assert!(text.contains("grouped_softavg"));
    assert!(text.contains("fused Filter[eps=0]"));
    assert!(text.contains("rows never scored"));
    assert!(text.contains("== candidates =="));
}

#[test]
fn rule_r1_pushes_only_when_legal() {
    // filter on a scalar column pushes below the aggregate
    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
               FROM movies WHERE year >= 2000 GROUP BY genre";
    let plan = optimize(parse_query(sql).unwrap());
    let LogicalPlan::SoftAgg { input, .. } = &plan else {
        panic!("expected SoftAgg at root, got {plan:?}");
    };
    assert!(matches!(input.as_ref(), LogicalPlan::Filter { .. }));

    // a filter naming the score's key column must NOT push below
    use bruce_query::{Pred, ScoreExpr};
    let illegal = LogicalPlan::Filter {
        pred: Pred::GtEq("emb".into(), 0.0),
        input: Box::new(LogicalPlan::SoftAgg {
            group_col: "genre".into(),
            val_col: "rating".into(),
            score: ScoreExpr {
                key_col: "emb".into(),
                param: "q".into(),
                kind: bruce_query::logical::SimKind::Dot,
            },
            eps: 0.3,
            budget: None,
            input: Box::new(LogicalPlan::Scan {
                table: "movies".into(),
            }),
        }),
    };
    let kept = optimize(illegal);
    assert!(matches!(kept, LogicalPlan::Filter { .. }));
}

#[test]
fn unbound_parameter_is_a_bind_error() {
    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :missing), 0.3) \
               FROM movies GROUP BY genre";
    let err = toy_db().run(sql, &params()).unwrap_err();
    assert!(err.to_string().contains("unbound parameter"));
}

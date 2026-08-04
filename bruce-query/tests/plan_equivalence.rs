//! Workstream 6 — planner equivalence fuzz.
//!
//! For 200 seeded random valid queries over a random in-memory catalog
//! (200-2000 rows, 1-16 groups, f32 OR f64 key column): execute the
//! NAIVE lowering of the parsed logical plan (no optimizer, no
//! planner: SoftAgg maps 1:1 to FusedGroupScan, keeping eps verbatim —
//! including eps = inf on the kernel's uniform path) against the full
//! `Database::run` pipeline (optimize -> cost-plan -> execute).
//! Labels must be identical and values equal to 1e-10 relative.
//!
//! No error budgets are generated: without a declared budget only
//! exact plans are admissible (planner contract), so the two paths
//! must agree to rounding, not to a tolerance contract.

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_query::exec::GroupResult;
use bruce_query::{execute, parse_query, Column, Database, LogicalPlan, PhysicalPlan, Pred, Table};

// ---------------------------------------------------- deterministic RNG

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn coin(&mut self) -> bool {
        self.next() & 1 == 0
    }
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next() as f64 / u64::MAX as f64)
    }
}

// ---------------------------------------------------- naive lowering

/// The 1:1 physical translation of the PARSED plan: no rule has run,
/// so eps = inf still routes through the scoring kernel's uniform
/// path and the filter sits wherever the parser put it.
fn lower_naive(plan: &LogicalPlan) -> PhysicalPlan {
    fn base_of(input: &LogicalPlan) -> (String, Option<Pred>) {
        match input {
            LogicalPlan::Scan { table } => (table.clone(), None),
            LogicalPlan::Filter { pred, input } => match input.as_ref() {
                LogicalPlan::Scan { table } => (table.clone(), Some(pred.clone())),
                other => panic!("unexpected filter input {other:?}"),
            },
            other => panic!("unexpected aggregate input {other:?}"),
        }
    }
    match plan {
        LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            budget: _,
            input,
        } => {
            let (table, sel) = base_of(input);
            PhysicalPlan::FusedGroupScan {
                table,
                group_col: group_col.clone(),
                val_col: val_col.clone(),
                score: score.clone(),
                eps: *eps,
                sel,
            }
        }
        LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input,
        } => {
            let (table, sel) = base_of(input);
            PhysicalPlan::ExactGroupAvg {
                table,
                group_col: group_col.clone(),
                val_col: val_col.clone(),
                sel,
            }
        }
        other => panic!("parser emitted unexpected root {other:?}"),
    }
}

// ---------------------------------------------------- random catalog

struct Case {
    table: Table,
    sql: String,
    eps_is_inf: bool,
    d: usize,
}

const EPS_LITS: [&str; 9] = [
    "0", "0.001", "0.1", "0.5", "1", "10", "1000000", "1e-9", "INF",
];

fn gen_case(rng: &mut Rng) -> Case {
    let n = 200 + rng.below(1801);
    let n_groups = 1 + rng.below(16);
    let d = 2 + rng.below(5);
    let use_f32 = rng.coin();

    let codes: Vec<u32> = (0..n).map(|_| rng.below(n_groups) as u32).collect();
    let dict: Vec<String> = (0..n_groups).map(|j| format!("grp{j}")).collect();
    let vals: Vec<f64> = (0..n).map(|_| rng.f64_in(-5.0, 5.0)).collect();
    // y quantised to multiples of 5 in [0, 95]: `=` predicates can
    // actually select rows and `>=` boundaries are exact in SQL text
    let ys: Vec<f64> = (0..n).map(|_| (rng.below(20) as f64) * 5.0).collect();
    let scale = 1.0 / (d as f64).sqrt();
    let mut kf = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for c in 0..d {
            kf[(i, c)] = rng.f64_in(-1.0, 1.0) * scale;
        }
    }

    let mut t = Table::default();
    t.columns
        .insert("g".into(), Column::DictU32 { codes, dict });
    t.columns.insert("v".into(), Column::ScalarF64(vals));
    t.columns.insert("y".into(), Column::ScalarF64(ys));
    if use_f32 {
        t.columns
            .insert("k".into(), Column::KeyF32(kf.mapv(|x| x as f32)));
    } else {
        t.columns.insert("k".into(), Column::KeyF64(kf));
    }

    let eps_lit = EPS_LITS[rng.below(EPS_LITS.len())];
    let simfn = if rng.coin() { "SIM" } else { "DOT" };
    // c up to 120 > max(y): some predicates empty the whole table
    let where_sql = match rng.below(4) {
        0 => String::new(),
        1 | 2 => format!(" WHERE y >= {:?}", (rng.below(25) as f64) * 5.0),
        _ => format!(" WHERE y = {:?}", (rng.below(25) as f64) * 5.0),
    };
    let sql =
        format!("SELECT g, SOFTAVG(v, {simfn}(k, :q), {eps_lit}) FROM t{where_sql} GROUP BY g");
    Case {
        table: t,
        sql,
        eps_is_inf: eps_lit == "INF",
        d,
    }
}

fn q_param(rng: &mut Rng, d: usize) -> HashMap<String, Array1<f64>> {
    let scale = 1.0 / (d as f64).sqrt();
    let q: Array1<f64> = (0..d).map(|_| rng.f64_in(-1.0, 1.0) * scale).collect();
    let mut p = HashMap::new();
    p.insert("q".to_string(), q);
    p
}

fn assert_equivalent(seed: u64, sql: &str, naive: &GroupResult, opt: &GroupResult) {
    assert_eq!(
        naive.labels, opt.labels,
        "seed {seed}: label divergence\nsql: {sql}"
    );
    for (i, (a, b)) in naive.values.iter().zip(&opt.values).enumerate() {
        let tol = 1e-10 * a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "seed {seed}: value divergence at group {} ({a} vs {b})\nsql: {sql}",
            naive.labels[i]
        );
    }
}

// ---------------------------------------------------- the fuzz loop

#[test]
fn fuzz_200_random_queries_optimized_equals_naive() {
    for seed in 0..200u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x0DDB_1A5E_5BD0_2E67).wrapping_add(3));
        let case = gen_case(&mut rng);
        let params = q_param(&mut rng, case.d);

        let mut db = Database::new();
        db.register("t", case.table);

        let (out_opt, planned) = db
            .run(&case.sql, &params)
            .unwrap_or_else(|e| panic!("seed {seed}: pipeline failed ({e}): {}", case.sql));

        // R3 shape invariant: inf degenerates, finite eps stays fused
        if case.eps_is_inf {
            assert!(
                matches!(planned.chosen, PhysicalPlan::ExactGroupAvg { .. }),
                "seed {seed}: inf eps must choose ExactGroupAvg:\n{}",
                planned.explain()
            );
        } else {
            assert!(
                matches!(planned.chosen, PhysicalPlan::FusedGroupScan { .. }),
                "seed {seed}: finite eps without budget must choose FusedGroupScan:\n{}",
                planned.explain()
            );
        }

        let logical = parse_query(&case.sql).unwrap();
        let naive_plan = lower_naive(&logical);
        let out_naive = execute(&naive_plan, &db.catalog, &params, &[])
            .unwrap_or_else(|e| panic!("seed {seed}: naive execution failed ({e}): {}", case.sql));

        assert_equivalent(seed, &case.sql, &out_naive, &out_opt);
    }
}

/// Boundary: predicates that admit no row at all. Both lowerings must
/// return the empty result (no covered groups), not an error.
#[test]
fn empty_selection_is_equivalent_and_empty() {
    for (seed, where_sql) in [
        (9001u64, " WHERE y >= 1000"),
        (9002, " WHERE y = 2.5"),
        (9003, " WHERE y >= 100.5"),
    ] {
        let mut rng = Rng::new(seed);
        let mut case = gen_case(&mut rng);
        let params = q_param(&mut rng, case.d);
        case.sql = format!("SELECT g, SOFTAVG(v, SIM(k, :q), 0.3) FROM t{where_sql} GROUP BY g");

        let mut db = Database::new();
        db.register("t", case.table);
        let (out_opt, _) = db.run(&case.sql, &params).unwrap();
        let naive_plan = lower_naive(&parse_query(&case.sql).unwrap());
        let out_naive = execute(&naive_plan, &db.catalog, &params, &[]).unwrap();

        assert!(
            out_opt.labels.is_empty(),
            "seed {seed}: expected empty result"
        );
        assert_equivalent(seed, &case.sql, &out_naive, &out_opt);
    }
}

/// Focused R3 check with an independent oracle: at eps = inf the naive
/// lowering (scoring kernel, uniform path) and the optimized plan
/// (ExactGroupAvg, no key read) must both equal the filtered plain
/// mean computed directly from the columns.
#[test]
fn inf_endpoint_naive_kernel_equals_exact_group_avg_and_oracle() {
    let mut rng = Rng::new(0xB0BA_F877);
    let mut case = gen_case(&mut rng);
    let params = q_param(&mut rng, case.d);
    case.sql = "SELECT g, SOFTAVG(v, SIM(k, :q), INF) FROM t WHERE y >= 50 GROUP BY g".into();

    // manual oracle straight off the generated columns
    let (codes, dict) = match (&case.table.columns["g"], &case.table.columns["v"]) {
        (Column::DictU32 { codes, dict }, _) => (codes.clone(), dict.clone()),
        _ => unreachable!(),
    };
    let vals = match &case.table.columns["v"] {
        Column::ScalarF64(v) => v.clone(),
        _ => unreachable!(),
    };
    let ys = match &case.table.columns["y"] {
        Column::ScalarF64(v) => v.clone(),
        _ => unreachable!(),
    };
    let mut sum = vec![0.0; dict.len()];
    let mut cnt = vec![0usize; dict.len()];
    for r in 0..codes.len() {
        if ys[r] >= 50.0 {
            sum[codes[r] as usize] += vals[r];
            cnt[codes[r] as usize] += 1;
        }
    }

    let mut db = Database::new();
    db.register("t", case.table);
    let (out_opt, planned) = db.run(&case.sql, &params).unwrap();
    assert!(matches!(planned.chosen, PhysicalPlan::ExactGroupAvg { .. }));
    let naive_plan = lower_naive(&parse_query(&case.sql).unwrap());
    let out_naive = execute(&naive_plan, &db.catalog, &params, &[]).unwrap();

    assert_equivalent(0xB0BA_F877, &case.sql, &out_naive, &out_opt);
    for (g, label) in dict.iter().enumerate() {
        let pos = out_opt.labels.iter().position(|l| l == label);
        if cnt[g] == 0 {
            assert!(pos.is_none(), "uncovered group {label} must be absent");
        } else {
            let want = sum[g] / cnt[g] as f64;
            let got = out_opt.values[pos.unwrap()];
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                "group {label}: {got} vs oracle {want}"
            );
        }
    }
}

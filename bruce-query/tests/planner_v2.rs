//! The temperature-aware optimizer, tested end to end: R3 endpoint
//! degeneration, maintained views under the write path (including
//! deletion of an anchor scorer), the error-contract admissibility
//! flip across temperatures, estimator accuracy against the oracle,
//! and stats freshness after writes.

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_query::{Column, Database, PhysicalPlan, Pred, Table, Verdict};

const GROUPS: [&str; 4] = ["A", "B", "C", "D"];

/// Synthetic table: sims are designed directly (key = [sim, 0, 0, 0],
/// query = e1), groups cycle, values are LCG pseudo-random in [0, 10).
fn synth_table(n: usize) -> (Table, Vec<f64>, Vec<u32>, Vec<f64>) {
    let sims: Vec<f64> = (0..n)
        .map(|i| 1.0 - 2.0 * (i as f64) / (n as f64))
        .collect();
    let codes: Vec<u32> = (0..n).map(|i| (i % GROUPS.len()) as u32).collect();
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
        "genre".into(),
        Column::DictU32 {
            codes: codes.clone(),
            dict: GROUPS.iter().map(|s| s.to_string()).collect(),
        },
    );
    t.columns
        .insert("rating".into(), Column::ScalarF64(vals.clone()));
    t.columns.insert(
        "id".into(),
        Column::ScalarF64((0..n).map(|i| i as f64).collect()),
    );
    t.columns.insert("emb".into(), Column::KeyF64(keys));
    (t, sims, codes, vals)
}

fn q4() -> HashMap<String, Array1<f64>> {
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0, 0.0, 0.0]));
    p
}

fn softmax_ref(sims: &[f64], vals: &[f64], eps: f64) -> f64 {
    let m = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (mut num, mut den) = (0.0, 0.0);
    for (s, v) in sims.iter().zip(vals) {
        let w = ((s - m) / eps).exp();
        num += w * v;
        den += w;
    }
    num / den
}

fn group_ref(sims: &[f64], codes: &[u32], vals: &[f64], g: u32, eps: f64) -> f64 {
    let (s, v): (Vec<f64>, Vec<f64>) = sims
        .iter()
        .zip(codes)
        .zip(vals)
        .filter(|((_, &c), _)| c == g)
        .map(|((s, _), v)| (*s, *v))
        .unzip();
    softmax_ref(&s, &v, eps)
}

// ---------------------------------------------------------------- R3

#[test]
fn r3_inf_degenerates_to_exact_group_avg() {
    let (t, _, codes, vals) = synth_table(4000);
    let mut db = Database::new();
    db.register("movies", t);
    let (out, planned) = db
        .run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), INF) FROM movies GROUP BY genre",
            &q4(),
        )
        .unwrap();
    assert!(matches!(planned.chosen, PhysicalPlan::ExactGroupAvg { .. }));
    // no key bytes in the chosen plan's cost drivers
    let chosen = planned
        .candidates
        .iter()
        .find(|c| matches!(c.verdict, Verdict::Chosen))
        .unwrap();
    assert!(chosen.cost.note.contains("no key read"));
    // equals the plain mean per group
    for (gi, gname) in GROUPS.iter().enumerate() {
        let (mut s, mut n) = (0.0, 0.0);
        for (c, v) in codes.iter().zip(&vals) {
            if *c as usize == gi {
                s += v;
                n += 1.0;
            }
        }
        let want = s / n;
        let got = out.values[out.labels.iter().position(|l| l == gname).unwrap()];
        assert!((got - want).abs() < 1e-12, "group {gname}: {got} vs {want}");
    }
}

#[test]
fn eps_zero_tropical_matches_argmax_average() {
    // rows 0..4 tie the per-group max sim by construction below
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: vec![0, 0, 0, 1, 1],
            dict: vec!["A".into(), "B".into()],
        },
    );
    t.columns.insert(
        "rating".into(),
        Column::ScalarF64(vec![2.0, 4.0, 9.0, 1.0, 5.0]),
    );
    let mut keys = Array2::<f64>::zeros((5, 2));
    // group A: sims 0.9, 0.9 (tie), 0.1 -> argmax avg = (2+4)/2 = 3
    // group B: sims 0.5, 0.8         -> argmax avg = 5
    for (i, s) in [0.9, 0.9, 0.1, 0.5, 0.8].iter().enumerate() {
        keys[(i, 0)] = *s;
    }
    t.columns.insert("emb".into(), Column::KeyF64(keys));
    let mut db = Database::new();
    db.register("movies", t);
    let mut p = HashMap::new();
    p.insert("q".to_string(), Array1::from(vec![1.0, 0.0]));
    let (out, _) = db
        .run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.0) FROM movies GROUP BY genre",
            &p,
        )
        .unwrap();
    let got: HashMap<_, _> = out
        .labels
        .iter()
        .cloned()
        .zip(out.values.iter().cloned())
        .collect();
    assert!((got["A"] - 3.0).abs() < 1e-12);
    assert!((got["B"] - 5.0).abs() < 1e-12);
}

// ------------------------------------------------- maintained views

#[test]
fn view_serves_query_and_survives_writes_including_anchor_delete() {
    let n = 2000;
    let (t, mut sims, mut codes, mut vals) = synth_table(n);
    let mut db = Database::new();
    db.register("movies", t);
    let x = Array1::from(vec![1.0, 0.0, 0.0, 0.0]);
    db.create_view("v_genre", "movies", "genre", "rating", "emb", &x, 0.1)
        .unwrap();

    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.1) FROM movies GROUP BY genre";

    // (1) planner serves the view; result matches the reference
    let (out, planned) = db.run(sql, &q4()).unwrap();
    assert!(matches!(
        planned.chosen,
        PhysicalPlan::MaintainedViewScan { .. }
    ));
    for (gi, gname) in GROUPS.iter().enumerate() {
        let want = group_ref(&sims, &codes, &vals, gi as u32, 0.1);
        let got = out.values[out.labels.iter().position(|l| l == gname).unwrap()];
        assert!((got - want).abs() < 1e-10, "{gname}: {got} vs {want}");
    }

    // (2) a filtered query cannot use the view
    let (_, planned) = db
        .run(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.1) \
             FROM movies WHERE id >= 0 GROUP BY genre",
            &q4(),
        )
        .unwrap();
    assert!(matches!(
        planned.chosen,
        PhysicalPlan::FusedGroupScan { .. }
    ));
    assert!(planned
        .candidates
        .iter()
        .any(|c| matches!(&c.verdict, Verdict::Inadmissible(r) if r.contains("filter"))));

    // (3) insert keeps the view exact
    let mut row = bruce_query::RowValues::default();
    row.scalars.insert("rating".into(), 7.5);
    row.scalars.insert("id".into(), n as f64);
    row.labels.insert("genre".into(), "B".into());
    row.keys.insert("emb".into(), vec![0.42, 0.0, 0.0, 0.0]);
    db.insert_row("movies", &row).unwrap();
    sims.push(0.42);
    codes.push(1);
    vals.push(7.5);

    let (out, _) = db.run(sql, &q4()).unwrap();
    let want = group_ref(&sims, &codes, &vals, 1, 0.1);
    let got = out.values[out.labels.iter().position(|l| l == "B").unwrap()];
    assert!((got - want).abs() < 1e-10, "after insert: {got} vs {want}");

    // (4) delete a NON-anchor row: no re-anchor
    let re0 = db.views[0].n_reanchors;
    let victim = 1001usize; // mid-pack sim, group B (1001 % 4 == 1)
    assert_eq!(codes[victim], 1);
    db.delete_where("movies", &Pred::Eq("id".into(), victim as f64))
        .unwrap();
    sims.remove(victim);
    codes.remove(victim);
    vals.remove(victim);
    assert_eq!(
        db.views[0].n_reanchors, re0,
        "non-anchor delete must not re-anchor"
    );

    // (5) delete THE anchor scorer of group A (id 0 has the global
    // max sim and 0 % 4 == 0): exactly one bounded re-anchor
    db.delete_where("movies", &Pred::Eq("id".into(), 0.0))
        .unwrap();
    sims.remove(0);
    codes.remove(0);
    vals.remove(0);
    assert_eq!(
        db.views[0].n_reanchors,
        re0 + 1,
        "anchor delete re-anchors once"
    );

    let (out, _) = db.run(sql, &q4()).unwrap();
    for (gi, gname) in GROUPS.iter().enumerate() {
        let want = group_ref(&sims, &codes, &vals, gi as u32, 0.1);
        let got = out.values[out.labels.iter().position(|l| l == gname).unwrap()];
        assert!(
            (got - want).abs() < 1e-10,
            "after deletes {gname}: {got} vs {want}"
        );
    }
}

// -------------------------------------------- the error contract

#[test]
fn contract_admissibility_flips_with_temperature() {
    let n = 8000;
    let (t, sims, codes, vals) = synth_table(n);
    let mut db = Database::new();
    db.register("movies", t);

    // sharp read: the sketch certifies a small k*; the contracted
    // plan is cheaper and gets chosen
    let sharp = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.02, 0.05) \
                 FROM movies GROUP BY genre";
    let (out, planned) = db.run(sharp, &q4()).unwrap();
    assert!(
        matches!(planned.chosen, PhysicalPlan::TopKContractScan { .. }),
        "sharp eps should choose the contracted plan:\n{}",
        planned.explain()
    );
    if let PhysicalPlan::TopKContractScan { est_kstar, .. } = &planned.chosen {
        assert!(
            *est_kstar < n / 4,
            "k* should be a small fraction, got {est_kstar}"
        );
    }
    // the runtime guard keeps every group inside the budget
    for (gi, gname) in GROUPS.iter().enumerate() {
        let want = group_ref(&sims, &codes, &vals, gi as u32, 0.02);
        let got = out.values[out.labels.iter().position(|l| l == gname).unwrap()];
        assert!(
            (got - want).abs() <= 0.05 + 1e-12,
            "{gname}: |{got} - {want}| > budget"
        );
    }

    // diffuse read: near-uniform weights; the contract buys nothing
    // and the exact fused scan wins
    let diffuse = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 1.0, 0.05) \
                   FROM movies GROUP BY genre";
    let (_, planned) = db.run(diffuse, &q4()).unwrap();
    assert!(
        matches!(planned.chosen, PhysicalPlan::FusedGroupScan { .. }),
        "diffuse eps should fall back to the exact plan:\n{}",
        planned.explain()
    );

    // super-sharp read: extreme-value regime; the sketch declares
    // itself resolution-limited and the planner refuses to trust it
    let supersharp = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.0001, 0.05) \
                      FROM movies GROUP BY genre";
    let (_, planned) = db.run(supersharp, &q4()).unwrap();
    assert!(matches!(
        planned.chosen,
        PhysicalPlan::FusedGroupScan { .. }
    ));
    assert!(
        planned
            .candidates
            .iter()
            .any(|c| matches!(&c.verdict, Verdict::Inadmissible(r) if r.contains("resolution"))),
        "expected a resolution-limited veto:\n{}",
        planned.explain()
    );

    // no budget declared: the contracted plan is never enumerated
    let exact = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.02) \
                 FROM movies GROUP BY genre";
    let (_, planned) = db.run(exact, &q4()).unwrap();
    assert!(!planned
        .candidates
        .iter()
        .any(|c| matches!(c.plan, PhysicalPlan::TopKContractScan { .. })));
}

#[test]
fn sketch_kstar_is_within_3x_of_oracle() {
    let n = 8000;
    let (t, sims, _, vals) = synth_table(n);
    let mut db = Database::new();
    db.register("movies", t);
    let stats = &db.stats["movies"];
    let sk = &stats.keys["emb"];
    let x = Array1::from(vec![1.0, 0.0, 0.0, 0.0]);

    let eps = 0.05;
    let budget = 0.05;
    let vmax = vals.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let est = sk.estimate_contract(&x.view(), eps, budget, vmax, n);
    assert!(!est.resolution_limited);

    // oracle k* over the full sims
    let mut s = sims.clone();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let m = s[0];
    let w: Vec<f64> = s.iter().map(|si| ((si - m) / eps).exp()).collect();
    let z: f64 = w.iter().sum();
    let mut acc = 0.0;
    let mut oracle = n;
    for (i, wi) in w.iter().enumerate() {
        acc += wi;
        let delta = 1.0 - acc / z;
        if delta * (1.0 + 1.0 / (1.0 - delta)) * vmax <= budget {
            oracle = i + 1;
            break;
        }
    }
    let ratio = est.kstar as f64 / oracle as f64;
    assert!(
        (0.33..=3.0).contains(&ratio),
        "estimate {} vs oracle {} (ratio {ratio:.2})",
        est.kstar,
        oracle
    );
}

// ------------------------------------------------- stats freshness

#[test]
fn selectivity_estimate_is_close_and_stats_refresh_after_writes() {
    let n = 4000;
    let (t, _, _, _) = synth_table(n);
    let mut db = Database::new();
    db.register("movies", t);

    // uniform id in [0, n): id >= n/2 selects half
    let sql = format!(
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) \
         FROM movies WHERE id >= {} GROUP BY genre",
        n / 2
    );
    let (_, planned) = db.run(&sql, &q4()).unwrap();
    let fused = planned
        .candidates
        .iter()
        .find(|c| matches!(c.plan, PhysicalPlan::FusedGroupScan { .. }))
        .unwrap();
    let sel = fused.cost.rows / n as f64;
    assert!((0.45..=0.55).contains(&sel), "selectivity {sel}");

    // delete 3/4 of the table; stats must refresh before the next plan
    db.delete_where("movies", &Pred::GtEq("id".into(), (n / 4) as f64))
        .unwrap();
    let sql = "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) FROM movies GROUP BY genre";
    let (_, planned) = db.run(sql, &q4()).unwrap();
    let fused = planned
        .candidates
        .iter()
        .find(|c| matches!(c.plan, PhysicalPlan::FusedGroupScan { .. }))
        .unwrap();
    assert!(
        (fused.cost.rows - (n / 4) as f64).abs() < 1.0,
        "stats stale: cost.rows = {}",
        fused.cost.rows
    );
}

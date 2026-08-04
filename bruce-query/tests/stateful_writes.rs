//! Workstream 9 — stateful property test of the write path.
//!
//! From a seeded random table, create 1-3 maintained views (different
//! eps), then apply 300 random operations drawn from {insert_row,
//! delete_where(Eq on the scalar id column), delete_where(GtEq)}.
//! After EVERY operation assert:
//!   (a) each maintained view's answers equal a from-scratch
//!       recomputation over the current table state (1e-9 rel), and
//!   (b) Database::run answers equal the recomputation too — once per
//!       view eps (exercises MaintainedViewScan) and once at an eps no
//!       view serves (exercises FusedGroupScan as a differential
//!       oracle against the independent reference fold).
//!
//! Shrink-friendly: every assertion message carries the seed and the
//! op index, so a failure replays with a one-line loop bound.
//!
//! 5 seeds run in the default suite; 50 under `--ignored`.

use std::collections::{BTreeMap, HashMap};

use ndarray::{Array1, Array2};

use bruce_query::db::RowValues;
use bruce_query::{Column, Database, Pred, Table};

const N_INIT: usize = 500;
const N_OPS: usize = 300;
const D_KEY: usize = 4;
const N_GROUPS: usize = 6;
const VIEW_EPS: [f64; 3] = [0.35, 0.9, 2.2];
/// An eps no view was built at: forces the fused kernel path.
const FRESH_EPS: f64 = 0.55;

// ------------------------------------------------------------ PRNG

/// SplitMix64: tiny, seedable, deterministic; no dev-dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// --------------------------------------------- mirror (test oracle)

/// One logical row, mirrored outside the engine.
#[derive(Clone)]
struct MirrorRow {
    id: f64,
    group: usize,
    rating: f64,
    key: Vec<f64>,
}

/// Reference soft average — an independent implementation of the
/// kernel's semantics (max-anchored softmax weights).
fn softavg_ref(rows: &[&MirrorRow], q: &[f64], eps: f64) -> f64 {
    let sims: Vec<f64> = rows
        .iter()
        .map(|r| r.key.iter().zip(q).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let m = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (mut num, mut den) = (0.0, 0.0);
    for (s, r) in sims.iter().zip(rows) {
        let w = ((s - m) / eps).exp();
        num += w * r.rating;
        den += w;
    }
    num / den
}

/// Expected per-group answers over the current mirror state; groups
/// with no rows are absent (matches view coverage / kernel coverage).
fn expected(mirror: &[MirrorRow], q: &[f64], eps: f64) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for g in 0..N_GROUPS {
        let rows: Vec<&MirrorRow> = mirror.iter().filter(|r| r.group == g).collect();
        if !rows.is_empty() {
            out.insert(format!("g{g}"), softavg_ref(&rows, q, eps));
        }
    }
    out
}

fn assert_close(got: &BTreeMap<String, f64>, want: &BTreeMap<String, f64>, ctx: &str) {
    let gk: Vec<&String> = got.keys().collect();
    let wk: Vec<&String> = want.keys().collect();
    assert_eq!(
        gk, wk,
        "{ctx}: covered groups differ (got {gk:?}, want {wk:?})"
    );
    for (label, w) in want {
        let g = got[label];
        let tol = 1e-9 * g.abs().max(w.abs()).max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: group {label}: got {g}, want {w} (|diff|={})",
            (g - w).abs()
        );
    }
}

// ----------------------------------------------------------- driver

fn build_table(rng: &mut Rng) -> (Table, Vec<MirrorRow>) {
    let mut mirror = Vec::with_capacity(N_INIT);
    for i in 0..N_INIT {
        mirror.push(MirrorRow {
            id: i as f64,
            group: rng.below(N_GROUPS),
            rating: 10.0 * rng.f64(),
            key: (0..D_KEY).map(|_| 2.0 * rng.f64() - 1.0).collect(),
        });
    }
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: mirror.iter().map(|r| r.group as u32).collect(),
            dict: (0..N_GROUPS).map(|g| format!("g{g}")).collect(),
        },
    );
    t.columns.insert(
        "rating".into(),
        Column::ScalarF64(mirror.iter().map(|r| r.rating).collect()),
    );
    t.columns.insert(
        "id".into(),
        Column::ScalarF64(mirror.iter().map(|r| r.id).collect()),
    );
    let mut keys = Array2::<f64>::zeros((N_INIT, D_KEY));
    for (i, r) in mirror.iter().enumerate() {
        for (j, &k) in r.key.iter().enumerate() {
            keys[(i, j)] = k;
        }
    }
    t.columns.insert("emb".into(), Column::KeyF64(keys));
    (t, mirror)
}

/// Check views and Database::run against the mirror after one op.
fn verify(db: &mut Database, mirror: &[MirrorRow], q: &Array1<f64>, view_eps: &[f64], ctx: &str) {
    let qs: Vec<f64> = q.to_vec();
    let mut params = HashMap::new();
    params.insert("q".to_string(), q.clone());

    // (a) every maintained view equals a from-scratch recomputation
    for (vi, &eps) in view_eps.iter().enumerate() {
        let want = expected(mirror, &qs, eps);
        let view = &db.views[vi];
        let got: BTreeMap<String, f64> = view
            .read()
            .into_iter()
            .map(|(g, val)| (format!("g{g}"), val))
            .collect();
        assert_close(&got, &want, &format!("{ctx} view[{vi}] eps={eps}"));
    }

    // (b) Database::run equals recomputation: once per view eps (the
    // planner serves these from MaintainedViewScan) and once at an
    // eps no view covers (forced FusedGroupScan).
    for &eps in view_eps.iter().chain(std::iter::once(&FRESH_EPS)) {
        let sql =
            format!("SELECT genre, SOFTAVG(rating, SIM(emb, :q), {eps}) FROM t GROUP BY genre");
        let want = expected(mirror, &qs, eps);
        if want.is_empty() {
            // table (or every group) is empty: the run must still
            // succeed and cover nothing
            let (res, _) = db
                .run(&sql, &params)
                .unwrap_or_else(|e| panic!("{ctx} run eps={eps} on empty state errored: {e}"));
            assert!(
                res.labels.is_empty(),
                "{ctx} run eps={eps}: expected no groups"
            );
            continue;
        }
        let (res, _) = db
            .run(&sql, &params)
            .unwrap_or_else(|e| panic!("{ctx} run eps={eps} errored: {e}"));
        let got: BTreeMap<String, f64> = res
            .labels
            .iter()
            .cloned()
            .zip(res.values.iter().cloned())
            .collect();
        assert_close(&got, &want, &format!("{ctx} run eps={eps}"));
    }
}

fn run_seed(seed: u64) {
    let mut rng = Rng::new(seed);
    let (table, mut mirror) = build_table(&mut rng);
    let mut db = Database::new();
    db.register("t", table);

    // 1-3 views at distinct eps, all bound to the same query vector
    let q = Array1::from(
        (0..D_KEY)
            .map(|_| 2.0 * rng.f64() - 1.0)
            .collect::<Vec<f64>>(),
    );
    let n_views = 1 + rng.below(3);
    let view_eps = &VIEW_EPS[..n_views];
    for (i, &eps) in view_eps.iter().enumerate() {
        db.create_view(&format!("v{i}"), "t", "genre", "rating", "emb", &q, eps)
            .unwrap_or_else(|e| panic!("seed={seed}: create_view v{i} failed: {e}"));
    }

    let mut next_id = N_INIT as f64;
    verify(
        &mut db,
        &mirror,
        &q,
        view_eps,
        &format!("seed={seed} op=init"),
    );

    for op in 0..N_OPS {
        let ctx = format!("seed={seed} op={op}");
        let roll = rng.below(100);
        if roll < 60 || mirror.is_empty() {
            // ---- insert_row (random row; may revive an empty group)
            let row = MirrorRow {
                id: next_id,
                group: rng.below(N_GROUPS),
                rating: 10.0 * rng.f64(),
                key: (0..D_KEY).map(|_| 2.0 * rng.f64() - 1.0).collect(),
            };
            next_id += 1.0;
            let rv = RowValues {
                scalars: [
                    ("rating".to_string(), row.rating),
                    ("id".to_string(), row.id),
                ]
                .into_iter()
                .collect(),
                labels: [("genre".to_string(), format!("g{}", row.group))]
                    .into_iter()
                    .collect(),
                keys: [("emb".to_string(), row.key.clone())].into_iter().collect(),
            };
            db.insert_row("t", &rv)
                .unwrap_or_else(|e| panic!("{ctx}: insert failed: {e}"));
            mirror.push(row);
        } else if roll < 85 {
            // ---- delete_where(Eq on id): one random live row
            let id = mirror[rng.below(mirror.len())].id;
            let n = db
                .delete_where("t", &Pred::Eq("id".into(), id))
                .unwrap_or_else(|e| panic!("{ctx}: delete Eq failed: {e}"));
            let before = mirror.len();
            mirror.retain(|r| r.id != id);
            assert_eq!(n, before - mirror.len(), "{ctx}: delete Eq count mismatch");
        } else {
            // ---- delete_where(GtEq on id): a tail of recent rows
            // (occasionally a bigger sweep to stress multi-row,
            // multi-group deletions incl. group anchors)
            let mut ids: Vec<f64> = mirror.iter().map(|r| r.id).collect();
            ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let reach = if roll % 5 == 0 { 20 } else { 6 };
            let k = 1 + rng.below(reach.min(ids.len()));
            let thr = ids[ids.len() - k];
            let n = db
                .delete_where("t", &Pred::GtEq("id".into(), thr))
                .unwrap_or_else(|e| panic!("{ctx}: delete GtEq failed: {e}"));
            let before = mirror.len();
            mirror.retain(|r| r.id < thr);
            assert_eq!(
                n,
                before - mirror.len(),
                "{ctx}: delete GtEq count mismatch"
            );
        }
        verify(&mut db, &mirror, &q, view_eps, &ctx);
    }
}

// ------------------------------------------------------------ tests

#[test]
fn stateful_writes_five_seeds() {
    for seed in [11, 22, 33, 44, 55] {
        run_seed(seed);
    }
}

#[test]
#[ignore = "50-seed soak; run with cargo test -- --ignored"]
fn stateful_writes_fifty_seeds() {
    for seed in 1..=50 {
        run_seed(seed);
    }
}

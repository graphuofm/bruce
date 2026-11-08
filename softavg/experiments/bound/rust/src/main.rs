//! Driver. Subcommands:
//!   run   <datadir> <out.json> <n_queries> <threads> <sweep_queries>
//!   bench <datadir> <out.json>
//!   argpatch <datadir> <out.json> <n_queries> <threads>
//!   viol <datadir> <query> <eps> <width>
//! <datadir> holds v.f64 (n), facet.u32 (n), scores.f64 (n_queries x n,
//! query-major), written by ../prep.py: admitted rows in item order,
//! s = float64(E) @ q.
use bound_exp::dd::DD;
use bound_exp::ddacc::DdInv;
use bound_exp::ess::{EssAcc, SoftEss};
use bound_exp::exact::SegTree;
use bound_exp::json::J;
use bound_exp::local::{self, Local};
use bound_exp::stats::LogHist;
use softagg_core::{InvAcc, ScalarAcc, ABS_TOL, REFOLD_GAIN, U};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const EPS: [f64; 5] = [1e-4, 1e-3, 0.02, 0.1, 1.0];
const WIDTHS: [usize; 3] = [50, 400, 5000];
const CHURN_OPS: usize = 10_000;
const CHURN_MIN_POOL: usize = 100;
const ADV_FACETS: usize = 10;
const ADV_POOL: usize = 2000;
const FOLD_BUDGET: f64 = 2e7; // rows of f64 two-pass folds per group
const LOOSE: f64 = 1e-2; // a state is "loose" when err / bound < LOOSE

// ------------------------------------------------------------------ data
fn read_f64(p: &str) -> Vec<f64> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_u32(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

struct Corpus {
    n: usize,
    nq: usize,
    v: Vec<f64>,
    facet: Vec<u32>,
    scores: Vec<f64>,
    vmax: f64,
}

fn load(dir: &str) -> Corpus {
    let v = read_f64(&format!("{dir}/v.f64"));
    let facet = read_u32(&format!("{dir}/facet.u32"));
    let scores = read_f64(&format!("{dir}/scores.f64"));
    let n = v.len();
    assert_eq!(facet.len(), n);
    assert_eq!(scores.len() % n, 0);
    let vmax = v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    Corpus { n, nq: scores.len() / n, v, facet, scores, vmax }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// --------------------------------------------------------------- scenarios
/// One independent state: a pool of corpus rows (storage order), positions
/// absorbed up front, then steps of (pos << 1 | absorb) ops; the value is
/// read after every step.
#[derive(Clone)]
struct Group {
    pool: Vec<u32>,
    init: Vec<u32>,
    ops: Vec<u32>,
    step_end: Vec<u32>,
    avg_live: f64,
}

fn sliding(n: usize, w: usize) -> Group {
    let mut ops = Vec::with_capacity(2 * n);
    let mut step_end = Vec::with_capacity(n);
    for t in 0..n {
        ops.push((t as u32) << 1 | 1);
        if t >= w {
            ops.push(((t - w) as u32) << 1);
        }
        step_end.push(ops.len() as u32);
    }
    Group { pool: (0..n as u32).collect(), init: vec![], ops, step_end, avg_live: w as f64 }
}

fn facet_pools(c: &Corpus) -> Vec<(u32, Vec<u32>)> {
    let mut m: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (i, &f) in c.facet.iter().enumerate() {
        m.entry(f).or_default().push(i as u32);
    }
    m.into_iter().collect()
}

fn churn(pools: &[(u32, Vec<u32>)]) -> Vec<Group> {
    let mut out = vec![];
    for (f, pool) in pools {
        if pool.len() < CHURN_MIN_POOL {
            continue;
        }
        let mut r = Lcg(0x9e3779b97f4a7c15 ^ (*f as u64 + 1));
        let mut live = vec![false; pool.len()];
        let mut init = vec![];
        for p in 0..pool.len() {
            if r.next() < 0.5 {
                init.push(p as u32);
                live[p] = true;
            }
        }
        let mut ops = vec![];
        let mut step_end = vec![0u32];
        for _ in 0..CHURN_OPS {
            let p = ((r.next() * pool.len() as f64) as usize).min(pool.len() - 1);
            ops.push((p as u32) << 1 | (!live[p]) as u32);
            live[p] = !live[p];
            step_end.push(ops.len() as u32);
        }
        out.push(Group { pool: pool.clone(), init, ops, step_end, avg_live: pool.len() as f64 / 2.0 });
    }
    out
}

/// the ADV_FACETS largest facets, first ADV_POOL rows each; ops filled per query
fn adversarial_pools(pools: &[(u32, Vec<u32>)]) -> Vec<Vec<u32>> {
    let mut p: Vec<&Vec<u32>> = pools.iter().map(|x| &x.1).collect();
    p.sort_by_key(|x| std::cmp::Reverse(x.len()));
    p.into_iter().take(ADV_FACETS).map(|x| x[..x.len().min(ADV_POOL)].to_vec()).collect()
}

fn adversarial(pool: &[u32], sv: &[(f64, f64)]) -> Group {
    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_by(|&a, &b| sv[b].0.partial_cmp(&sv[a].0).unwrap().then(a.cmp(&b)));
    let mut ops = vec![];
    let mut step_end = vec![0u32];
    for &p in &order {
        ops.push((p as u32) << 1);
        step_end.push(ops.len() as u32);
    }
    Group {
        pool: pool.to_vec(),
        init: (0..pool.len() as u32).collect(),
        ops,
        step_end,
        avg_live: pool.len() as f64 / 2.0,
    }
}

// ------------------------------------------------------------ accumulators
trait Acc: Copy {
    fn absorb(&mut self, s: f64, v: f64, eps: f64);
    fn remove(&mut self, s: f64, v: f64, eps: f64);
    fn exact(&self) -> bool;
    fn value(&self) -> Option<f64>;
    fn bound(&self) -> f64;
    fn tol(&self) -> f64;
    fn reset(&mut self);
}

impl<const I: bool> Acc for Local<I> {
    fn absorb(&mut self, s: f64, v: f64, e: f64) {
        Local::absorb(self, s, v, e)
    }
    fn remove(&mut self, s: f64, v: f64, e: f64) {
        Local::remove(self, s, v, e)
    }
    fn exact(&self) -> bool {
        self.is_exact()
    }
    fn value(&self) -> Option<f64> {
        self.finalize()
    }
    fn bound(&self) -> f64 {
        self.value_error_bound()
    }
    fn tol(&self) -> f64 {
        Local::tol(self)
    }
    fn reset(&mut self) {
        Local::reset(self)
    }
}

impl Acc for DdInv {
    fn absorb(&mut self, s: f64, v: f64, e: f64) {
        DdInv::absorb(self, s, v, e)
    }
    fn remove(&mut self, s: f64, v: f64, e: f64) {
        DdInv::remove(self, s, v, e)
    }
    fn exact(&self) -> bool {
        self.is_exact()
    }
    fn value(&self) -> Option<f64> {
        self.finalize()
    }
    fn bound(&self) -> f64 {
        self.value_error_bound()
    }
    fn tol(&self) -> f64 {
        DdInv::tol(self)
    }
    fn reset(&mut self) {
        DdInv::reset(self)
    }
}

/// The shipped core InvAcc run in lockstep with the instrumented local copy;
/// every field the core has must agree bit-for-bit after every operation.
#[derive(Clone, Copy)]
struct Pair {
    core: InvAcc,
    loc: Local<true>,
    mismatch: bool,
}
impl Pair {
    fn new() -> Self {
        Pair { core: InvAcc::new(), loc: Local::new(ABS_TOL, REFOLD_GAIN), mismatch: false }
    }
    #[inline]
    fn cmp(&mut self) {
        let (c, l) = (&self.core, &self.loc);
        let same = c.acc == l.acc
            && c.n == l.n
            && c.err_z.to_bits() == l.err_z.to_bits()
            && c.err_u.to_bits() == l.err_u.to_bits()
            && c.v_scale.to_bits() == l.v_scale.to_bits()
            && c.inexact == l.inexact;
        self.mismatch |= !same;
    }
}
impl Acc for Pair {
    fn absorb(&mut self, s: f64, v: f64, e: f64) {
        self.core.absorb(s, v, e);
        self.loc.absorb(s, v, e);
        self.cmp();
    }
    fn remove(&mut self, s: f64, v: f64, e: f64) {
        self.core.remove(s, v, e);
        self.loc.remove(s, v, e);
        self.cmp();
    }
    fn exact(&self) -> bool {
        self.core.is_exact()
    }
    fn value(&self) -> Option<f64> {
        self.core.finalize()
    }
    fn bound(&self) -> f64 {
        self.core.value_error_bound()
    }
    fn tol(&self) -> f64 {
        self.loc.tol()
    }
    fn reset(&mut self) {
        let m = self.mismatch;
        *self = Pair::new();
        self.mismatch = m;
    }
}

// ------------------------------------------------------------------ replay
enum Policy<'a> {
    Bound,
    Oracle(&'a [DD]),
}

#[inline]
fn err_vs(x: f64, ex: DD) -> f64 {
    ((x - ex.hi) - ex.lo).abs()
}

struct Live {
    flag: Vec<bool>,
    lo: usize,
    hi: usize,
    count: usize,
}
impl Live {
    fn new(n: usize) -> Self {
        Live { flag: vec![false; n], lo: n, hi: 0, count: 0 }
    }
    #[inline]
    fn on(&mut self, p: usize) {
        debug_assert!(!self.flag[p]);
        self.flag[p] = true;
        self.count += 1;
        self.lo = self.lo.min(p);
        self.hi = self.hi.max(p + 1);
    }
    #[inline]
    fn off(&mut self, p: usize) {
        debug_assert!(self.flag[p]);
        self.flag[p] = false;
        self.count -= 1;
        while self.lo < self.hi && !self.flag[self.lo] {
            self.lo += 1;
        }
    }
}

/// Replays a group; refolds (reset + absorb the live rows in storage order)
/// when the policy says so. Calls `f(step, &acc, live_count)` after each
/// non-empty step. Returns (refolds, refolded rows, refolds whose bound was
/// infinite, i.e. the computed z no longer exceeded its own error bound).
fn replay<A: Acc, F: FnMut(usize, &A, usize)>(
    g: &Group,
    sv: &[(f64, f64)],
    eps: f64,
    mut acc: A,
    policy: Policy,
    mut f: F,
) -> (u64, u64, u64) {
    let mut live = Live::new(g.pool.len());
    for &p in &g.init {
        let p = p as usize;
        acc.absorb(sv[p].0, sv[p].1, eps);
        live.on(p);
    }
    let (mut refolds, mut rows, mut inf) = (0u64, 0u64, 0u64);
    let mut a = 0usize;
    for (k, &end) in g.step_end.iter().enumerate() {
        for &op in &g.ops[a..end as usize] {
            let p = (op >> 1) as usize;
            if op & 1 == 1 {
                acc.absorb(sv[p].0, sv[p].1, eps);
                live.on(p);
            } else {
                acc.remove(sv[p].0, sv[p].1, eps);
                live.off(p);
            }
        }
        a = end as usize;
        if live.count == 0 {
            continue;
        }
        let refold = match policy {
            Policy::Bound => !acc.exact(),
            Policy::Oracle(ex) => match acc.value() {
                Some(x) if x.is_finite() => !(err_vs(x, ex[k]) <= acc.tol()),
                _ => true,
            },
        };
        if refold {
            inf += acc.bound().is_infinite() as u64;
            acc.reset();
            for p in live.lo..live.hi {
                if live.flag[p] {
                    acc.absorb(sv[p].0, sv[p].1, eps);
                }
            }
            refolds += 1;
            rows += live.count as u64;
        }
        f(k, &acc, live.count);
    }
    (refolds, rows, inf)
}

/// Exact value after every step (NaN hi for an empty step) and, on a
/// sample of steps, the fresh two-pass f64 fold (NaN where unsampled).
fn exact_pass(g: &Group, sv: &[(f64, f64)], eps: f64) -> (Vec<DD>, Vec<f64>) {
    let mut leaves: Vec<Option<(f64, f64)>> = vec![None; g.pool.len()];
    let mut live = Live::new(g.pool.len());
    for &p in &g.init {
        leaves[p as usize] = Some(sv[p as usize]);
        live.on(p as usize);
    }
    let mut tree = SegTree::build(&leaves, eps);
    let steps = g.step_end.len();
    let stride = ((steps as f64 * g.avg_live / FOLD_BUDGET).ceil() as usize).max(1);
    let mut ex = Vec::with_capacity(steps);
    let mut fold = vec![f64::NAN; steps];
    let mut a = 0usize;
    for (k, &end) in g.step_end.iter().enumerate() {
        for &op in &g.ops[a..end as usize] {
            let p = (op >> 1) as usize;
            if op & 1 == 1 {
                tree.set(p, Some(sv[p]));
                live.on(p);
            } else {
                tree.set(p, None);
                live.off(p);
            }
        }
        a = end as usize;
        match tree.all().value() {
            Some(x) => ex.push(x),
            None => ex.push(DD { hi: f64::NAN, lo: 0.0 }),
        }
        if live.count > 0 && k % stride == 0 {
            let mut m = f64::NEG_INFINITY;
            for p in live.lo..live.hi {
                if live.flag[p] {
                    m = m.max(sv[p].0);
                }
            }
            let (mut num, mut den) = (0.0, 0.0);
            for p in live.lo..live.hi {
                if live.flag[p] {
                    let w = ((sv[p].0 - m) / eps).exp();
                    num += w * sv[p].1;
                    den += w;
                }
            }
            fold[k] = num / den;
        }
    }
    (ex, fold)
}

// ------------------------------------------------------------------- stats
const OPS_BINS: usize = 48;

#[derive(Clone)]
struct MainStats {
    ratio_exact: LogHist,
    ratio_exact_incr: LogHist,
    ratio_fold: LogHist,
    fold_err: LogHist,
    err_vscale: LogHist,
    bound_vscale: LogHist,
    viol_exact: u64,
    viol_fold: u64,
    viol_tol: u64,
    max_err_vmax: f64,
    share_all: [f64; 4],
    n_all: u64,
    share_loose: [f64; 4],
    n_loose: u64,
    term_all: [f64; 3],
    term_loose: [f64; 3],
    ops_cnt: [u64; OPS_BINS],
    ops_zero: [u64; OPS_BINS],
    ops_sumlog: [f64; OPS_BINS],
    refolds: u64,
    refolds_inf: u64,
    refold_rows: u64,
    steps: u64,
    mismatch: u64,
    groups: u64,
}

impl MainStats {
    fn new() -> Self {
        MainStats {
            ratio_exact: LogHist::new(),
            ratio_exact_incr: LogHist::new(),
            ratio_fold: LogHist::new(),
            fold_err: LogHist::new(),
            err_vscale: LogHist::new(),
            bound_vscale: LogHist::new(),
            viol_exact: 0,
            viol_fold: 0,
            viol_tol: 0,
            max_err_vmax: 0.0,
            share_all: [0.0; 4],
            n_all: 0,
            share_loose: [0.0; 4],
            n_loose: 0,
            term_all: [0.0; 3],
            term_loose: [0.0; 3],
            ops_cnt: [0; OPS_BINS],
            ops_zero: [0; OPS_BINS],
            ops_sumlog: [0.0; OPS_BINS],
            refolds: 0,
            refolds_inf: 0,
            refold_rows: 0,
            steps: 0,
            mismatch: 0,
            groups: 0,
        }
    }
    fn merge(&mut self, o: &MainStats) {
        self.ratio_exact.merge(&o.ratio_exact);
        self.ratio_exact_incr.merge(&o.ratio_exact_incr);
        self.ratio_fold.merge(&o.ratio_fold);
        self.fold_err.merge(&o.fold_err);
        self.err_vscale.merge(&o.err_vscale);
        self.bound_vscale.merge(&o.bound_vscale);
        self.viol_exact += o.viol_exact;
        self.viol_fold += o.viol_fold;
        self.viol_tol += o.viol_tol;
        self.max_err_vmax = self.max_err_vmax.max(o.max_err_vmax);
        for i in 0..4 {
            self.share_all[i] += o.share_all[i];
            self.share_loose[i] += o.share_loose[i];
        }
        for i in 0..3 {
            self.term_all[i] += o.term_all[i];
            self.term_loose[i] += o.term_loose[i];
        }
        self.n_all += o.n_all;
        self.n_loose += o.n_loose;
        for i in 0..OPS_BINS {
            self.ops_cnt[i] += o.ops_cnt[i];
            self.ops_zero[i] += o.ops_zero[i];
            self.ops_sumlog[i] += o.ops_sumlog[i];
        }
        self.refolds += o.refolds;
        self.refolds_inf += o.refolds_inf;
        self.refold_rows += o.refold_rows;
        self.steps += o.steps;
        self.mismatch += o.mismatch;
        self.groups += o.groups;
    }
    fn to_json(&self, vmax: f64) -> J {
        let avg = |s: &[f64], n: u64| J::nums(&s.iter().map(|x| x / n.max(1) as f64).collect::<Vec<_>>());
        let mut ops = vec![];
        for i in 0..OPS_BINS {
            if self.ops_cnt[i] > 0 {
                let nz = self.ops_cnt[i] - self.ops_zero[i];
                ops.push(J::obj(vec![
                    ("log2_ops_lo", J::I(i as i64)),
                    ("states", J::I(self.ops_cnt[i] as i64)),
                    ("zero_err", J::I(self.ops_zero[i] as i64)),
                    ("geomean_ratio_nonzero", J::N(if nz > 0 { 10f64.powf(self.ops_sumlog[i] / nz as f64) } else { f64::NAN })),
                ]));
            }
        }
        J::obj(vec![
            ("steps", J::I(self.steps as i64)),
            ("groups", J::I(self.groups as i64)),
            ("refolds", J::I(self.refolds as i64)),
            ("refolds_bound_infinite", J::I(self.refolds_inf as i64)),
            ("refold_rows", J::I(self.refold_rows as i64)),
            ("ratio_vs_exact", self.ratio_exact.to_json()),
            ("ratio_vs_exact_incremental", self.ratio_exact_incr.to_json()),
            ("ratio_vs_f64_fold", self.ratio_fold.to_json()),
            ("ratio_vs_exact_hist", self.ratio_exact.bins_json()),
            ("f64_fold_err_over_vscale", self.fold_err.to_json()),
            ("err_over_vscale", self.err_vscale.to_json()),
            ("bound_over_vscale", self.bound_vscale.to_json()),
            ("violations_vs_exact", J::I(self.viol_exact as i64)),
            ("violations_vs_f64_fold", J::I(self.viol_fold as i64)),
            ("err_above_tol", J::I(self.viol_tol as i64)),
            ("max_err_over_vmax", J::N(self.max_err_vmax)),
            ("vmax", J::N(vmax)),
            ("bound_share_all", avg(&self.share_all, self.n_all)),
            ("bound_share_loose", avg(&self.share_loose, self.n_loose)),
            ("bound_terms_all", avg(&self.term_all, self.n_all)),
            ("bound_terms_loose", avg(&self.term_loose, self.n_loose)),
            ("states_all", J::I(self.n_all as i64)),
            ("states_loose", J::I(self.n_loose as i64)),
            ("slack_by_ops_since_fresh", J::A(ops)),
            ("core_vs_local_copy_mismatches", J::I(self.mismatch as i64)),
        ])
    }
}

#[derive(Clone, Default)]
struct SweepStats {
    refolds: u64,
    refolds_inf: u64,
    refold_rows: u64,
    steps: u64,
    max_err_vmax: f64,
    max_err_over_tol: f64,
    err_above_tol: u64,
    viol_bound: u64,
    viol_max_err_vscale: f64,
    vscale_zero: u64,
    nonfinite: u64,
    ratio: Option<LogHist>,
}
impl SweepStats {
    fn merge(&mut self, o: &SweepStats) {
        self.refolds += o.refolds;
        self.refolds_inf += o.refolds_inf;
        self.refold_rows += o.refold_rows;
        self.steps += o.steps;
        self.max_err_vmax = self.max_err_vmax.max(o.max_err_vmax);
        self.max_err_over_tol = self.max_err_over_tol.max(o.max_err_over_tol);
        self.err_above_tol += o.err_above_tol;
        self.viol_bound += o.viol_bound;
        self.viol_max_err_vscale = self.viol_max_err_vscale.max(o.viol_max_err_vscale);
        self.vscale_zero += o.vscale_zero;
        self.nonfinite += o.nonfinite;
        match (&mut self.ratio, &o.ratio) {
            (Some(a), Some(b)) => a.merge(b),
            (None, Some(b)) => self.ratio = Some(b.clone()),
            _ => {}
        }
    }
    fn to_json(&self) -> Vec<(&'static str, J)> {
        let mut v = vec![
            ("refolds", J::I(self.refolds as i64)),
            ("refolds_bound_infinite", J::I(self.refolds_inf as i64)),
            ("refold_rows", J::I(self.refold_rows as i64)),
            ("steps", J::I(self.steps as i64)),
            ("max_err_over_vmax", J::N(self.max_err_vmax)),
            ("max_err_over_tol", J::N(self.max_err_over_tol)),
            ("err_above_tol", J::I(self.err_above_tol as i64)),
            ("bound_violations_vs_exact", J::I(self.viol_bound as i64)),
            ("violations_max_abs_err_over_vmax", J::N(self.viol_max_err_vscale)),
            ("states_with_all_zero_values", J::I(self.vscale_zero as i64)),
            ("nonfinite_values", J::I(self.nonfinite as i64)),
        ];
        if let Some(r) = &self.ratio {
            v.push(("ratio_vs_exact", r.to_json()));
        }
        v
    }
}

fn sweep_run<A: Acc>(g: &Group, sv: &[(f64, f64)], eps: f64, acc: A, policy: Policy, ex: &[DD], vmax: f64, with_ratio: bool) -> SweepStats {
    let mut st = SweepStats { ratio: if with_ratio { Some(LogHist::new()) } else { None }, ..Default::default() };
    let (r, rows, inf) = replay(g, sv, eps, acc, policy, |k, a, _| {
        st.steps += 1;
        let x = a.value().unwrap();
        let e = err_vs(x, ex[k]);
        let tol = a.tol();
        st.max_err_vmax = st.max_err_vmax.max(e / vmax);
        st.max_err_over_tol = st.max_err_over_tol.max(e / tol);
        if !(e <= tol) {
            st.err_above_tol += 1;
        }
        let b = a.bound();
        if !(e <= b) {
            st.viol_bound += 1;
            st.viol_max_err_vscale = st.viol_max_err_vscale.max(e / vmax);
        }
        if !x.is_finite() {
            st.nonfinite += 1;
        }
        if tol <= f64::MIN_POSITIVE {
            st.vscale_zero += 1;
        }
        if let Some(h) = st.ratio.as_mut() {
            h.add(if b > 0.0 { e / b } else if e == 0.0 { 0.0 } else { f64::INFINITY });
        }
    });
    st.refolds = r;
    st.refold_rows = rows;
    st.refolds_inf = inf;
    st
}

/// (label, abs_tol, gain) for the local-copy sweep
fn sweep_configs() -> Vec<(String, f64, f64)> {
    let mut v = vec![];
    for &a in &[1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6] {
        v.push((format!("abs={a:e},gain=100"), a, 100.0));
    }
    for &g in &[0.0, 1.0, 10.0, 1e3, 1e4, 1e5] {
        v.push((format!("abs=1e-10,gain={g:e}"), 1e-10, g));
    }
    for &g in &[1.0, 10.0, 100.0, 1e3, 1e4] {
        v.push((format!("abs=0,gain={g:e}"), 0.0, g));
    }
    v
}
fn dd_configs() -> Vec<(String, f64, f64)> {
    vec![
        ("abs=1e-10,gain=100".into(), 1e-10, 100.0),
        ("abs=1e-12,gain=100".into(), 1e-12, 100.0),
        ("abs=1e-14,gain=100".into(), 1e-14, 100.0),
        ("abs=1e-14,gain=1".into(), 1e-14, 1.0),
    ]
}

#[derive(Clone)]
struct Cell {
    main: MainStats,
    oracle: SweepStats,
    sweep: Vec<SweepStats>,
    dd: Vec<SweepStats>,
}

fn run_group(g: &Group, sv: &[(f64, f64)], eps: f64, vmax: f64, do_sweep: bool, cell: &mut Cell) {
    let (ex, fold) = exact_pass(g, sv, eps);
    // main: shipped core, instrumented copy in lockstep
    let m = &mut cell.main;
    m.groups += 1;
    let (r, rows, inf) = replay(g, sv, eps, Pair::new(), Policy::Bound, |k, a, _| {
        m.steps += 1;
        let x = a.value().unwrap();
        let e = err_vs(x, ex[k]);
        let b = a.bound();
        let vs = a.loc.v_scale;
        let ratio = if b > 0.0 { e / b } else if e == 0.0 { 0.0 } else { f64::INFINITY };
        m.ratio_exact.add(ratio);
        if a.loc.removes > 0 {
            m.ratio_exact_incr.add(ratio);
        }
        m.err_vscale.add(e / vs);
        m.bound_vscale.add(b / vs);
        if !(e <= b) {
            m.viol_exact += 1;
        }
        if !(e <= a.tol()) {
            m.viol_tol += 1;
        }
        m.max_err_vmax = m.max_err_vmax.max(e / vmax);
        if !fold[k].is_nan() {
            let ef = (x - fold[k]).abs();
            m.ratio_fold.add(if b > 0.0 { ef / b } else if ef == 0.0 { 0.0 } else { f64::INFINITY });
            if !(ef <= b) {
                m.viol_fold += 1;
            }
            m.fold_err.add(err_vs(fold[k], ex[k]) / vs);
        }
        // decomposition of the bound
        if b > 0.0 && b.is_finite() {
            let l = &a.loc;
            let z = l.acc.z;
            let f = (l.acc.u / z).abs();
            let d = z - l.err_z;
            let mut sh = [0.0; 4];
            for j in 0..3 {
                sh[j] = (l.cu[j] + f * l.cz[j]) / d / b;
            }
            sh[3] = U * f / b;
            let t = l.bound_terms();
            let loose = ratio < LOOSE;
            for j in 0..4 {
                m.share_all[j] += sh[j];
                if loose {
                    m.share_loose[j] += sh[j];
                }
            }
            for j in 0..3 {
                m.term_all[j] += t[j] / b;
                if loose {
                    m.term_loose[j] += t[j] / b;
                }
            }
            m.n_all += 1;
            if loose {
                m.n_loose += 1;
            }
            let bin = ((l.ops as f64 + 1.0).log2().floor() as usize).min(OPS_BINS - 1);
            m.ops_cnt[bin] += 1;
            if ratio == 0.0 {
                m.ops_zero[bin] += 1;
            } else {
                m.ops_sumlog[bin] += ratio.log10();
            }
        }
        m.mismatch += a.mismatch as u64;
    });
    m.refolds += r;
    m.refold_rows += rows;
    m.refolds_inf += inf;
    let _ = (local::W, local::A, local::R);
    // oracle: same arithmetic, refold only when the true error exceeds tol
    let o = sweep_run(g, sv, eps, Local::<false>::new(ABS_TOL, REFOLD_GAIN), Policy::Oracle(&ex), &ex, vmax, false);
    cell.oracle.merge(&o);
    if do_sweep {
        for (i, (_, a, gn)) in sweep_configs().iter().enumerate() {
            let s = sweep_run(g, sv, eps, Local::<false>::new(*a, *gn), Policy::Bound, &ex, vmax, false);
            cell.sweep[i].merge(&s);
        }
    }
    for (i, (_, a, gn)) in dd_configs().iter().enumerate() {
        if i > 0 && !do_sweep {
            continue;
        }
        let s = sweep_run(g, sv, eps, DdInv::new(*a, *gn), Policy::Bound, &ex, vmax, i == 0);
        cell.dd[i].merge(&s);
    }
}

fn new_cell() -> Cell {
    Cell {
        main: MainStats::new(),
        oracle: SweepStats::default(),
        sweep: vec![SweepStats::default(); sweep_configs().len()],
        dd: vec![SweepStats::default(); dd_configs().len()],
    }
}

fn merge_cell(a: &mut Cell, b: &Cell) {
    a.main.merge(&b.main);
    a.oracle.merge(&b.oracle);
    for (x, y) in a.sweep.iter_mut().zip(&b.sweep) {
        x.merge(y);
    }
    for (x, y) in a.dd.iter_mut().zip(&b.dd) {
        x.merge(y);
    }
}

// --------------------------------------------------------------------- run
fn run(dir: &str, out: &str, nq: usize, threads: usize, sweep_q: usize) {
    let c = load(dir);
    let nq = nq.min(c.nq);
    let pools = facet_pools(&c);
    let mut scen: Vec<(String, Vec<Group>)> = WIDTHS.iter().map(|&w| (format!("sliding_w{w}"), vec![sliding(c.n, w)])).collect();
    scen.push(("churn".into(), churn(&pools)));
    let adv_pools = adversarial_pools(&pools);
    let names: Vec<String> = scen.iter().map(|s| s.0.clone()).chain(std::iter::once("adversarial".to_string())).collect();
    let units: Vec<(usize, usize)> = (0..nq).flat_map(|q| (0..EPS.len()).map(move |e| (q, e))).collect();
    let next = AtomicUsize::new(0);
    let cells: Mutex<BTreeMap<(usize, usize), Cell>> = Mutex::new(BTreeMap::new());
    // ESS: per eps, (ess, n) of every facet group and query; max |merge - seq| / seq
    let ess: Mutex<Vec<(Vec<(f64, f64)>, f64)>> = Mutex::new(vec![(vec![], 0.0); EPS.len()]);
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= units.len() {
                    break;
                }
                let (q, ei) = units[i];
                let eps = EPS[ei];
                let s = &c.scores[q * c.n..(q + 1) * c.n];
                let do_sweep = q < sweep_q;
                let mut local: BTreeMap<(usize, usize), Cell> = BTreeMap::new();
                for (si, (_, groups)) in scen.iter().enumerate() {
                    let cell = local.entry((si, ei)).or_insert_with(new_cell);
                    for g in groups {
                        let sv: Vec<(f64, f64)> = g.pool.iter().map(|&r| (s[r as usize], c.v[r as usize])).collect();
                        run_group(g, &sv, eps, c.vmax, do_sweep, cell);
                    }
                }
                let si = scen.len();
                let cell = local.entry((si, ei)).or_insert_with(new_cell);
                for p in &adv_pools {
                    let sv: Vec<(f64, f64)> = p.iter().map(|&r| (s[r as usize], c.v[r as usize])).collect();
                    let g = adversarial(p, &sv);
                    run_group(&g, &sv, eps, c.vmax, do_sweep, cell);
                }
                // ESS over facet groups
                let mut ev = vec![];
                let mut md = 0.0f64;
                for (_, pool) in &pools {
                    let mut seq = EssAcc::new();
                    for &r in pool {
                        seq.absorb(s[r as usize], eps);
                    }
                    let mut tot = EssAcc::new();
                    for ch in pool.chunks(pool.len().div_ceil(7)) {
                        let mut p = EssAcc::new();
                        for &r in ch {
                            p.absorb(s[r as usize], eps);
                        }
                        tot.merge(&p, eps);
                    }
                    let (a, b) = (seq.finalize().unwrap(), tot.finalize().unwrap());
                    md = md.max((a - b).abs() / a);
                    ev.push((a, pool.len() as f64));
                }
                {
                    let mut e = ess.lock().unwrap();
                    e[ei].0.extend(ev);
                    e[ei].1 = e[ei].1.max(md);
                }
                let mut all = cells.lock().unwrap();
                for (k, v) in local {
                    match all.get_mut(&k) {
                        Some(x) => merge_cell(x, &v),
                        None => {
                            all.insert(k, v);
                        }
                    }
                }
                eprintln!("[{:.0}s] unit {}/{} (q={q}, eps={eps:e}) done", t0.elapsed().as_secs_f64(), i + 1, units.len());
            });
        }
    });
    let cells = cells.into_inner().unwrap();
    let mut jc = vec![];
    for ((si, ei), cell) in &cells {
        let sweep: Vec<J> = sweep_configs()
            .iter()
            .zip(&cell.sweep)
            .map(|((lab, a, g), s)| {
                let mut kv = vec![("config", J::S(lab.clone())), ("abs_tol", J::N(*a)), ("gain", J::N(*g))];
                kv.extend(s.to_json());
                J::obj(kv)
            })
            .collect();
        let dd: Vec<J> = dd_configs()
            .iter()
            .zip(&cell.dd)
            .map(|((lab, a, g), s)| {
                let mut kv = vec![("config", J::S(lab.clone())), ("abs_tol", J::N(*a)), ("gain", J::N(*g))];
                kv.extend(s.to_json());
                J::obj(kv)
            })
            .collect();
        let mut okv = vec![("policy", J::S("refold iff |value - exact| > tol (default tol)".into()))];
        okv.extend(cell.oracle.to_json());
        jc.push(J::obj(vec![
            ("scenario", J::S(names[*si].clone())),
            ("eps", J::N(EPS[*ei])),
            ("shipped", cell.main.to_json(c.vmax)),
            ("oracle", J::obj(okv)),
            ("sweep_local_copy", J::A(sweep)),
            ("ddinv", J::A(dd)),
        ]));
    }
    let ess = ess.into_inner().unwrap();
    let mut je = vec![];
    for (ei, (vals, md)) in ess.iter().enumerate() {
        let mut e: Vec<f64> = vals.iter().map(|x| x.0).collect();
        let mut fr: Vec<f64> = vals.iter().map(|x| x.0 / x.1).collect();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        fr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let qt = |v: &Vec<f64>, p: f64| v[((p * (v.len() - 1) as f64).round()) as usize];
        je.push(J::obj(vec![
            ("eps", J::N(EPS[ei])),
            ("groups", J::I(e.len() as i64)),
            ("ess_p10", J::N(qt(&e, 0.1))),
            ("ess_median", J::N(qt(&e, 0.5))),
            ("ess_p90", J::N(qt(&e, 0.9))),
            ("frac_ess_below_2", J::N(e.iter().filter(|&&x| x < 2.0).count() as f64 / e.len() as f64)),
            ("frac_ess_below_10", J::N(e.iter().filter(|&&x| x < 10.0).count() as f64 / e.len() as f64)),
            ("ess_over_n_median", J::N(qt(&fr, 0.5))),
            ("max_rel_diff_merge7_vs_sequential", J::N(*md)),
        ]));
    }
    let res = J::obj(vec![
        ("n_rows", J::I(c.n as i64)),
        ("queries", J::I(nq as i64)),
        ("sweep_queries", J::I(sweep_q.min(nq) as i64)),
        ("vmax", J::N(c.vmax)),
        ("eps", J::nums(&EPS)),
        ("widths", J::A(WIDTHS.iter().map(|&w| J::I(w as i64)).collect())),
        ("churn", J::obj(vec![("ops_per_group", J::I(CHURN_OPS as i64)), ("min_pool", J::I(CHURN_MIN_POOL as i64)), ("groups", J::I(scen[3].1.len() as i64))])),
        ("adversarial", J::obj(vec![("facets", J::I(adv_pools.len() as i64)), ("pool_cap", J::I(ADV_POOL as i64))])),
        ("fold_budget_rows_per_group", J::N(FOLD_BUDGET)),
        ("loose_threshold", J::N(LOOSE)),
        ("sweep_configs", J::A(sweep_configs().iter().map(|x| J::S(x.0.clone())).collect())),
        ("dd_configs", J::A(dd_configs().iter().map(|x| J::S(x.0.clone())).collect())),
        ("wall_seconds", J::N(t0.elapsed().as_secs_f64())),
        ("threads", J::I(threads as i64)),
        ("cells", J::A(jc)),
        ("ess", J::A(je)),
    ]);
    std::fs::write(out, res.render()).unwrap();
}

// ------------------------------------------------------------------- bench
fn time_it<F: FnMut() -> f64>(rows: usize, reps: usize, mut f: F) -> f64 {
    let mut t = vec![];
    for _ in 0..reps {
        let t0 = Instant::now();
        std::hint::black_box(f());
        t.push(t0.elapsed().as_secs_f64() * 1e9 / rows as f64);
    }
    // minimum over repetitions: on a shared machine the minimum is the
    // least-disturbed estimate of the intrinsic cost
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[0]
}

fn bench(dir: &str, out: &str) {
    let c = load(dir);
    let sv: Vec<(f64, f64)> = (0..c.n).map(|i| (c.scores[i], c.v[i])).collect();
    let rep = (20_000_000 / c.n).max(1);
    let rows = rep * c.n;
    let reps = 9;
    let w = 400usize;
    let mut cells = vec![];
    for &eps in &EPS {
        let eps_bb = std::hint::black_box(eps);
        let mut r = vec![];
        r.push(("scalar_absorb", time_it(rows, reps, || {
            let mut a = ScalarAcc::new();
            for _ in 0..rep {
                for &(s, v) in &sv {
                    a.absorb(s, v, eps_bb);
                }
            }
            a.u / a.z
        })));
        r.push(("ess_absorb", time_it(rows, reps, || {
            let mut a = EssAcc::new();
            for _ in 0..rep {
                for &(s, _) in &sv {
                    a.absorb(s, eps_bb);
                }
            }
            a.q
        })));
        r.push(("softavg_plus_ess_absorb", time_it(rows, reps, || {
            let mut a = SoftEss::new();
            for _ in 0..rep {
                for &(s, v) in &sv {
                    a.absorb(s, v, eps_bb);
                }
            }
            a.u / a.z + a.q
        })));
        r.push(("invacc_absorb", time_it(rows, reps, || {
            let mut a = InvAcc::new();
            for _ in 0..rep {
                for &(s, v) in &sv {
                    a.absorb(s, v, eps_bb);
                }
            }
            a.acc.u + a.err_u
        })));
        r.push(("local_copy_absorb", time_it(rows, reps, || {
            let mut a = Local::<false>::new(ABS_TOL, REFOLD_GAIN);
            for _ in 0..rep {
                for &(s, v) in &sv {
                    a.absorb(s, v, eps_bb);
                }
            }
            a.acc.u + a.err_u
        })));
        r.push(("ddinv_absorb", time_it(rows, reps, || {
            let mut a = DdInv::new(ABS_TOL, REFOLD_GAIN);
            for _ in 0..rep {
                for &(s, v) in &sv {
                    a.absorb(s, v, eps_bb);
                }
            }
            a.uh + a.ru
        })));
        // sliding window as a host runs it: one absorb + one remove per row,
        // and a refold of the 400-row frame whenever the state goes inexact
        let mut refolds = [0u64; 2];
        r.push(("invacc_slide400_with_refolds_per_row", time_it(rows, reps, || {
            let mut acc = 0.0;
            let mut nref = 0u64;
            for _ in 0..rep {
                let mut a = InvAcc::new();
                for i in 0..sv.len() {
                    a.absorb(sv[i].0, sv[i].1, eps_bb);
                    if i >= w {
                        a.remove(sv[i - w].0, sv[i - w].1, eps_bb);
                    }
                    if !a.is_exact() {
                        a.reset();
                        for &(s, v) in &sv[(i + 1).saturating_sub(w)..=i] {
                            a.absorb(s, v, eps_bb);
                        }
                        nref += 1;
                    }
                }
                acc += a.acc.u;
            }
            refolds[0] = nref / rep as u64;
            acc
        })));
        r.push(("ddinv_slide400_with_refolds_per_row", time_it(rows, reps, || {
            let mut acc = 0.0;
            let mut nref = 0u64;
            for _ in 0..rep {
                let mut a = DdInv::new(ABS_TOL, REFOLD_GAIN);
                for i in 0..sv.len() {
                    a.absorb(sv[i].0, sv[i].1, eps_bb);
                    if i >= w {
                        a.remove(sv[i - w].0, sv[i - w].1, eps_bb);
                    }
                    if !a.is_exact() {
                        a.reset();
                        for &(s, v) in &sv[(i + 1).saturating_sub(w)..=i] {
                            a.absorb(s, v, eps_bb);
                        }
                        nref += 1;
                    }
                }
                acc += a.uh;
            }
            refolds[1] = nref / rep as u64;
            acc
        })));
        let base = r[0].1;
        let mut kv = vec![("eps", J::N(eps))];
        let mut m = vec![];
        for (k, ns) in &r {
            m.push(J::obj(vec![("op", J::S(k.to_string())), ("ns_per_row", J::N(*ns)), ("vs_scalar_absorb", J::N(ns / base))]));
        }
        kv.push(("ops", J::A(m)));
        kv.push(("slide400_refolds_per_pass", J::obj(vec![("invacc", J::I(refolds[0] as i64)), ("ddinv", J::I(refolds[1] as i64)), ("rows_per_pass", J::I(sv.len() as i64))])));
        cells.push(J::obj(kv));
        eprintln!("bench eps={eps:e} done");
    }
    let res = J::obj(vec![
        ("rows_per_rep", J::I(rows as i64)),
        ("reps_min_of", J::I(reps as i64)),
        ("data", J::S("IMDb admitted rows in item order, query 0 scores, one group".into())),
        ("state_bytes", J::obj(vec![
            ("ScalarAcc", J::I(std::mem::size_of::<ScalarAcc>() as i64)),
            ("InvAcc", J::I(std::mem::size_of::<InvAcc>() as i64)),
            ("DdInv_prototype_incl_2_tol_fields", J::I(std::mem::size_of::<DdInv>() as i64)),
            ("EssAcc", J::I(std::mem::size_of::<EssAcc>() as i64)),
        ])),
        ("cells", J::A(cells)),
    ]);
    std::fs::write(out, res.render()).unwrap();
}

// --------------------------------------------------- argument-term patch
/// shipped vs the proposed argument-rounding patch (local copy, arg_term),
/// same tolerance: refolds, violations against the exact value.
fn argpatch(dir: &str, out: &str, nq: usize, threads: usize) {
    let c = load(dir);
    let nq = nq.min(c.nq);
    let pools = facet_pools(&c);
    let mut scen: Vec<(String, Vec<Group>)> = WIDTHS.iter().map(|&w| (format!("sliding_w{w}"), vec![sliding(c.n, w)])).collect();
    scen.push(("churn".into(), churn(&pools)));
    let adv_pools = adversarial_pools(&pools);
    let units: Vec<(usize, usize)> = (0..nq).flat_map(|q| (0..EPS.len()).map(move |e| (q, e))).collect();
    let next = AtomicUsize::new(0);
    let acc: Mutex<BTreeMap<(String, usize), [SweepStats; 2]>> = Mutex::new(BTreeMap::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= units.len() {
                    break;
                }
                let (q, ei) = units[i];
                let eps = EPS[ei];
                let s = &c.scores[q * c.n..(q + 1) * c.n];
                let mut jobs: Vec<(String, Group, Vec<(f64, f64)>)> = vec![];
                for (name, groups) in &scen {
                    for g in groups {
                        let sv: Vec<(f64, f64)> = g.pool.iter().map(|&r| (s[r as usize], c.v[r as usize])).collect();
                        jobs.push((name.clone(), g.clone(), sv));
                    }
                }
                for p in &adv_pools {
                    let sv: Vec<(f64, f64)> = p.iter().map(|&r| (s[r as usize], c.v[r as usize])).collect();
                    let g = adversarial(p, &sv);
                    jobs.push(("adversarial".into(), g, sv));
                }
                for (name, g, sv) in jobs {
                    let (ex, _) = exact_pass(&g, &sv, eps);
                    let a = sweep_run(&g, &sv, eps, Local::<false>::new(ABS_TOL, REFOLD_GAIN), Policy::Bound, &ex, c.vmax, false);
                    let mut l = Local::<false>::new(ABS_TOL, REFOLD_GAIN);
                    l.arg_term = true;
                    let b = sweep_run(&g, &sv, eps, l, Policy::Bound, &ex, c.vmax, false);
                    let mut m = acc.lock().unwrap();
                    let e = m.entry((name, ei)).or_insert_with(|| [SweepStats::default(), SweepStats::default()]);
                    e[0].merge(&a);
                    e[1].merge(&b);
                }
                eprintln!("argpatch unit {}/{}", i + 1, units.len());
            });
        }
    });
    let m = acc.into_inner().unwrap();
    let mut cells = vec![];
    for ((name, ei), st) in &m {
        let mut a = vec![("scenario", J::S(name.clone())), ("eps", J::N(EPS[*ei]))];
        a.push(("shipped", J::obj(st[0].to_json())));
        a.push(("patched", J::obj(st[1].to_json())));
        cells.push(J::obj(a));
    }
    let res = J::obj(vec![("queries", J::I(nq as i64)), ("n_rows", J::I(c.n as i64)), ("cells", J::A(cells))]);
    std::fs::write(out, res.render()).unwrap();
}

// ------------------------------------------------------------- violations
/// Print the violating states of one sliding-window run (diagnostics).
fn viol(dir: &str, q: usize, eps: f64, w: usize) {
    let c = load(dir);
    let s = &c.scores[q * c.n..(q + 1) * c.n];
    let g = sliding(c.n, w);
    let sv: Vec<(f64, f64)> = g.pool.iter().map(|&r| (s[r as usize], c.v[r as usize])).collect();
    let (ex, _) = exact_pass(&g, &sv, eps);
    let (mut nv, mut max_abs, mut zero_val, mut shown) = (0u64, 0.0f64, 0u64, 0);
    let mut max_ratio = 0.0f64;
    replay(&g, &sv, eps, Pair::new(), Policy::Bound, |k, a, _| {
        let x = a.value().unwrap();
        let e = err_vs(x, ex[k]);
        let b = a.bound();
        if e > b {
            nv += 1;
            max_abs = max_abs.max(e / a.loc.v_scale);
            max_ratio = max_ratio.max(if b > 0.0 { e / b } else { f64::INFINITY });
            if x == 0.0 {
                zero_val += 1;
            }
            if shown < 8 {
                shown += 1;
                let lo = (k + 1).saturating_sub(w);
                let m = sv[lo..=k].iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
                let nsub = sv[lo..=k].iter().filter(|r| ((r.0 - a.loc.acc.mu) / eps).exp() < f64::MIN_POSITIVE).count();
                println!("step {k}: got {x:e} exact {:e} err {e:e} bound {b:e} v_scale {:e} n {} z {:e} mu {} frame_max {} rows_with_subnormal_or_zero_weight {nsub} err_z {:e} err_u {:e}",
                    ex[k].hi, a.loc.v_scale, a.loc.n, a.loc.acc.z, a.loc.acc.mu, m, a.loc.err_z, a.loc.err_u);
            }
        }
    });
    println!("violations {nv}, of which delivered value == 0: {zero_val}; max err / v_scale {max_abs:e}; max ratio {max_ratio:e}");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("run") => run(&a[2], &a[3], a[4].parse().unwrap(), a[5].parse().unwrap(), a[6].parse().unwrap()),
        Some("bench") => bench(&a[2], &a[3]),
        Some("viol") => viol(&a[2], a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap()),
        Some("argpatch") => argpatch(&a[2], &a[3], a[4].parse().unwrap(), a[5].parse().unwrap()),
        _ => eprintln!("usage: bound-exp run <dir> <out> <nq> <threads> <sweep_q> | bench <dir> <out>"),
    }
}

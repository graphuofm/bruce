//! TRACK 1 / workstream 3 — delete-drift soak for `IncrementalMemory`
//! (Lemma A: the `(m, num, den)` group accumulator with true DELETE).
//!
//! The exact-unlearning claim is that after ANY interleaving of
//! inserts and deletes the maintained output equals the from-scratch
//! computation over the surviving records. Floating-point subtraction
//! is not exact, so we bound the accumulated drift instead:
//!
//! * fast default: 50_000 random insert/delete cycles (deletes only
//!   ever target live keys), checkpoint every N/10 cycles, assert
//!   max relative drift < 1e-9;
//! * `#[ignore]`-gated soak: 1_000_000 cycles, same bound (run with
//!   `cargo test -p bruce-core --test drift_soak -- --ignored`).
//!
//! Each checkpoint compares the incremental output against TWO
//! independent oracles:
//!   1. a from-scratch rebuild (fresh `IncrementalMemory`, insert-only,
//!      same insertion order) — isolates delete-induced drift;
//!   2. a grouped mirror — `grouped_softavg` over the live rows as a
//!      single group with the same query vector — a physically
//!      different code path (RowAcc fold in mask.rs) computing the
//!      same F_eps.
//!
//! The observed max drift is printed so soak runs leave a number in
//! the log. If the bound is ever exceeded that is a REAL finding (the
//! re-anchor path is `IncrementalMemory::rescale`, triggered on
//! delete-of-max); the bound must not be loosened silently.

use bruce_core::mask::grouped_softavg;
use bruce_core::types::{Eps, Sim};
use bruce_core::IncrementalMemory;
use ndarray::{Array1, Array2};

const D_K: usize = 4;
const D_V: usize = 2;
const TARGET_LIVE: usize = 256;
const DRIFT_BOUND: f64 = 1e-9;

/// xorshift64* — deterministic, no rand dev-dependency.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f64(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct LiveRow {
    id: String,
    k: Array1<f64>,
    v: Array1<f64>,
}

fn rel_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-30))
        .fold(0.0, f64::max)
}

/// Run `cycles` insert/delete operations; checkpoint 10 times; return
/// (max observed relative drift, number of rescales triggered).
fn soak(cycles: usize, seed: u64) -> (f64, u64) {
    let mut rng = Rng::new(seed);
    let x = Array1::from_iter((0..D_K).map(|_| rng.f64() * 2.0 - 1.0));
    let eps = Eps::ONE;
    let mut mem = IncrementalMemory::new(x.view(), eps, D_V, Sim::Dot);
    let mut live: Vec<LiveRow> = Vec::new();
    let mut next_id = 0u64;
    let checkpoint_every = (cycles / 10).max(1);
    let mut max_drift = 0.0f64;

    for cycle in 1..=cycles {
        let insert =
            live.len() < TARGET_LIVE / 4 || (rng.next() & 1 == 0 && live.len() < 4 * TARGET_LIVE);
        if insert {
            let id = format!("k{next_id}");
            next_id += 1;
            // scores ~ dot(x, k) with k in [-1, 1]^4: O(1) spread, so
            // the max is regularly deleted and the rescale (re-anchor)
            // path is genuinely exercised.
            let k = Array1::from_iter((0..D_K).map(|_| rng.f64() * 2.0 - 1.0));
            // values in [0.5, 5]: relative drift is well-conditioned
            let v = Array1::from_iter((0..D_V).map(|_| rng.f64() * 4.5 + 0.5));
            mem.insert(&id, k.view(), v.view()).unwrap();
            live.push(LiveRow { id, k, v });
        } else {
            // delete only live keys, uniformly at random
            let idx = rng.below(live.len());
            let row = live.swap_remove(idx);
            mem.delete(&row.id).unwrap();
        }

        if cycle % checkpoint_every == 0 && !live.is_empty() {
            let got = mem.output();

            // oracle 1: from-scratch rebuild over the survivors
            let mut fresh = IncrementalMemory::new(x.view(), eps, D_V, Sim::Dot);
            for r in &live {
                fresh.insert(&r.id, r.k.view(), r.v.view()).unwrap();
            }
            let want_rebuild = fresh.output();

            // oracle 2: grouped mirror (single group through the
            // RowAcc fold in mask.rs)
            let n = live.len();
            let mut k_mat = Array2::<f64>::zeros((n, D_K));
            let mut v_mat = Array2::<f64>::zeros((n, D_V));
            for (i, r) in live.iter().enumerate() {
                k_mat.row_mut(i).assign(&r.k);
                v_mat.row_mut(i).assign(&r.v);
            }
            let gid = vec![0u32; n];
            let (mirror, covered) =
                grouped_softavg(&x.view(), &k_mat.view(), &v_mat.view(), &gid, 1, None, eps)
                    .unwrap();
            assert!(
                covered[0],
                "cycle {cycle}: mirror uncovered with {n} live rows"
            );
            let mirror_row = mirror.row(0).to_owned();

            let d1 = rel_diff(&got, &want_rebuild);
            let d2 = rel_diff(&got, &mirror_row);
            max_drift = max_drift.max(d1).max(d2);
            assert!(
                d1 < DRIFT_BOUND && d2 < DRIFT_BOUND,
                "cycle {cycle}: drift rebuild={d1:e} mirror={d2:e} exceeds {DRIFT_BOUND:e} \
                 (n_live={n}, n_rescales={}) — REAL FINDING, check crud.rs re-anchor path",
                mem.n_rescales()
            );
        }
    }
    (max_drift, mem.n_rescales())
}

/// Fast default: 50k cycles.
#[test]
fn drift_after_50k_insert_delete_cycles_below_1e9() {
    let (max_drift, rescales) = soak(50_000, 0x5EED);
    println!("drift soak 50k cycles: max rel drift = {max_drift:.3e}, rescales = {rescales}");
    assert!(max_drift < DRIFT_BOUND);
    // the delete-of-max re-anchor path must actually have run,
    // otherwise the soak proved nothing about it
    assert!(
        rescales > 0,
        "no rescale ever triggered — workload too tame"
    );
}

/// Long soak: 1M cycles. `cargo test -p bruce-core --test drift_soak -- --ignored`
#[test]
#[ignore = "long soak: 1M cycles; run with -- --ignored"]
fn drift_after_1m_insert_delete_cycles_below_1e9() {
    let (max_drift, rescales) = soak(1_000_000, 0xD1CE);
    println!("drift soak 1M cycles: max rel drift = {max_drift:.3e}, rescales = {rescales}");
    assert!(max_drift < DRIFT_BOUND);
    assert!(rescales > 0);
}

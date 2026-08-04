//! Generic masked attention over an intensionally-given mask.
//!
//! The mask is consumed as a stream of `(i, j)` pairs in **arbitrary
//! order** (no duplicates): row `i` of `Q` attends to row `j` of
//! `K`/`V` iff the pair `(i, j)` appears in the stream. This is the
//! "enumerate-then-fold" evaluator of the PODS paper's free-connex
//! transfer theorem: any duplicate-free enumeration of a mask —
//! causal, sliding window, ancestor tree, join-query output — feeds
//! the same per-row fold, and the result is independent of the
//! enumeration order because the fold is a commutative-monoid
//! homomorphism (the structure lemma).
//!
//! ```text
//!     out[i]  =  A_ε(q_i, {(k_j, v_j) : (i,j) ∈ pairs})
//! ```
//!
//! Complexity: O(|pairs| · d) time after O(|pairs|) validation;
//! O(N_q · (d_v + 2)) accumulator space. The parallel path splits the
//! pair stream into chunks and merges per-chunk accumulators — this
//! merge is exactly the partition-reduce identity (Lemma B), so the
//! parallel result equals the sequential one up to floating-point
//! associativity.
//!
//! Temperature semantics (one code path per regime, same fold shape):
//! - `ε > 0` finite: max-shifted `(μ, u, z)` accumulator,
//!   `out[i] = u/z` (online-softmax).
//! - `ε = 0`: tropical accumulator `(μ, Σv over argmax, count)`,
//!   `out[i]` = mean of values over the argmax set (uniform tie
//!   handling, matching `semiring::softmax_eps`).
//! - `ε = ∞`: all weights 1, `out[i]` = plain mean over the mask row.
//!
//! Rows `i` with no pair in the stream are *uncovered*: the output
//! row is zero and the returned `covered[i]` flag is `false`. NaN
//! (the engine's SQL-NULL encoding) skips a pair in every regime —
//! see [`masked_attention`]'s NULL-semantics section.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

use crate::error::BruceError;
use crate::types::Eps;

/// Below this pair count the sequential fold wins (rayon overhead +
/// per-chunk accumulator allocation dominate).
const PAIR_PARALLEL_THRESHOLD: usize = 1 << 15;

/// Per-row accumulator in the max-shifted representation.
///
/// Invariant for finite `ε > 0` after absorbing a set S of pairs:
/// `u = e^{-μ/ε} Σ_{j∈S} e^{s_j/ε} v_j`, `z = e^{-μ/ε} Σ_{j∈S} e^{s_j/ε}`,
/// `μ = max_{j∈S} s_j`. For `ε = 0`: `z` is the argmax multiplicity
/// and `u` the value-sum over the argmax set. For `ε = ∞`: `z` is the
/// count and `u` the plain value-sum.
#[derive(Clone, Debug)]
struct RowAcc {
    mu: f64,
    z: f64,
    u: Vec<f64>,
}

impl RowAcc {
    fn new(d_v: usize) -> Self {
        Self {
            mu: f64::NEG_INFINITY,
            z: 0.0,
            u: vec![0.0; d_v],
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.z == 0.0
    }

    /// Absorb one record with score `s` and value row `v_row`.
    ///
    /// Score-infinity policy (previously UNDEFINED — `exp(inf - inf)`
    /// poisoned the accumulator with NaN; pinned by
    /// `tests/numerical_edges.rs`, PG-aligned per C4):
    /// * `s = -inf` carries weight 0 at `ε = 0` and finite `ε` and is
    ///   skipped outright. This is the `Sim::Indicator` "no match"
    ///   encoding, so a group whose every row scores `-inf` stays
    ///   *uncovered* — SQL's empty equi-join match set aggregating to
    ///   NULL — matching `IncrementalMemory`'s tropical path in
    ///   crud.rs (which also drops non-finite scores). Note this
    ///   deliberately diverges from `semiring::softmax_eps`'s
    ///   pathological all-`-inf` uniform fallback.
    /// * `s = +inf` at finite `ε` dominates every finite score: the
    ///   accumulator collapses to argmax semantics over the `+inf`
    ///   rows, with uniform tie handling exactly as at `ε = 0`.
    /// * At `ε = ∞` the score is not consulted (plain mean), so `±inf`
    ///   rows count like any other row.
    ///
    /// NaN scores/values never reach `absorb` from the grouped
    /// kernels — the SQL-NULL skip happens at the call sites (see
    /// [`grouped_softavg`]).
    #[inline]
    fn absorb(&mut self, s: f64, v_row: &ArrayView1<'_, f64>, eps: Eps) {
        if eps.is_zero() {
            if s == f64::NEG_INFINITY {
                // Indicator "no match": weight 0 (policy above)
                return;
            }
            // tropical: keep only the argmax set
            if s > self.mu {
                self.mu = s;
                self.z = 1.0;
                for (c, uc) in self.u.iter_mut().enumerate() {
                    *uc = v_row[c];
                }
            } else if s == self.mu {
                self.z += 1.0;
                for (c, uc) in self.u.iter_mut().enumerate() {
                    *uc += v_row[c];
                }
            }
            return;
        }
        if eps.is_inf() {
            // uniform: plain count + sum
            self.z += 1.0;
            for (c, uc) in self.u.iter_mut().enumerate() {
                *uc += v_row[c];
            }
            return;
        }
        // finite ε > 0
        if s == f64::NEG_INFINITY {
            // exp(-inf / ε) = 0: contributes nothing (policy above)
            return;
        }
        if s == f64::INFINITY || self.mu == f64::INFINITY {
            // argmax collapse: only +inf-scored rows retain weight
            if s == f64::INFINITY && self.mu == f64::INFINITY {
                // tie among +inf rows: uniform, as at ε = 0
                self.z += 1.0;
                for (c, uc) in self.u.iter_mut().enumerate() {
                    *uc += v_row[c];
                }
            } else if s == f64::INFINITY {
                // first +inf row dominates the finite prefix
                self.mu = f64::INFINITY;
                self.z = 1.0;
                for (c, uc) in self.u.iter_mut().enumerate() {
                    *uc = v_row[c];
                }
            }
            // else: finite s under a +inf anchor: weight exp(-inf) = 0
            return;
        }
        if self.is_empty() {
            self.mu = s;
            self.z = 1.0;
            for (c, uc) in self.u.iter_mut().enumerate() {
                *uc = v_row[c];
            }
            return;
        }
        let mu2 = self.mu.max(s);
        let scale = ((self.mu - mu2) / eps.0).exp();
        let w = ((s - mu2) / eps.0).exp();
        for (c, uc) in self.u.iter_mut().enumerate() {
            *uc = *uc * scale + w * v_row[c];
        }
        self.z = self.z * scale + w;
        self.mu = mu2;
    }

    /// Merge another accumulator into this one — the partition-reduce
    /// identity (Lemma B): disjoint pair sets combine by re-basing both
    /// sides to the common maximum and adding.
    fn merge(&mut self, other: &RowAcc, eps: Eps) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other.clone();
            return;
        }
        if eps.is_zero() {
            if other.mu > self.mu {
                *self = other.clone();
            } else if other.mu == self.mu {
                self.z += other.z;
                for (c, uc) in self.u.iter_mut().enumerate() {
                    *uc += other.u[c];
                }
            }
            return;
        }
        if eps.is_inf() {
            self.z += other.z;
            for (c, uc) in self.u.iter_mut().enumerate() {
                *uc += other.u[c];
            }
            return;
        }
        // finite ε with a +inf anchor on either side (see `absorb`'s
        // policy comment): the +inf side(s) dominate; exp(inf - inf)
        // must never be evaluated. Pinned by tests/numerical_edges.rs.
        if self.mu == f64::INFINITY || other.mu == f64::INFINITY {
            if self.mu == f64::INFINITY && other.mu == f64::INFINITY {
                self.z += other.z;
                for (c, uc) in self.u.iter_mut().enumerate() {
                    *uc += other.u[c];
                }
            } else if other.mu == f64::INFINITY {
                *self = other.clone();
            }
            // else: self is the +inf side; other's finite rows weigh 0
            return;
        }
        let mu2 = self.mu.max(other.mu);
        let s1 = ((self.mu - mu2) / eps.0).exp();
        let s2 = ((other.mu - mu2) / eps.0).exp();
        for (c, uc) in self.u.iter_mut().enumerate() {
            *uc = *uc * s1 + other.u[c] * s2;
        }
        self.z = self.z * s1 + other.z * s2;
        self.mu = mu2;
    }

    /// Final per-row output: `u / z` in every regime (for `ε > 0` this
    /// is the softmax-normalised value; for `ε = 0` the argmax mean;
    /// for `ε = ∞` the plain mean). `None` if no pair was absorbed.
    fn finalize(&self) -> Option<Vec<f64>> {
        if self.is_empty() {
            return None;
        }
        Some(self.u.iter().map(|uc| uc / self.z).collect())
    }
}

fn fold_sequential(
    q: &ArrayView2<'_, f64>,
    k: &ArrayView2<'_, f64>,
    v: &ArrayView2<'_, f64>,
    pairs: &[(usize, usize)],
    v_ok: &[bool],
    eps: Eps,
) -> Vec<RowAcc> {
    let mut accs = vec![RowAcc::new(v.ncols()); q.nrows()];
    for &(i, j) in pairs {
        let s = q.row(i).dot(&k.row(j));
        // SQL NULL discipline (C4): NaN encodes NULL. A pair whose
        // score is NaN (NaN anywhere in q_i or k_j) or whose value row
        // holds a NaN component is skipped in every eps regime — the
        // same rule as the grouped kernels. Exposed by
        // tests/numerical_edges.rs (mod masked_attention_nan_policy):
        // NaN scores used to poison the accumulator on this surface.
        if s.is_nan() || !v_ok[j] {
            continue;
        }
        accs[i].absorb(s, &v.row(j), eps);
    }
    accs
}

fn fold_parallel(
    q: &ArrayView2<'_, f64>,
    k: &ArrayView2<'_, f64>,
    v: &ArrayView2<'_, f64>,
    pairs: &[(usize, usize)],
    v_ok: &[bool],
    eps: Eps,
) -> Vec<RowAcc> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = pairs.len().div_ceil(n_threads);
    pairs
        .par_chunks(chunk)
        .map(|c| fold_sequential(q, k, v, c, v_ok, eps))
        .reduce(
            || vec![RowAcc::new(v.ncols()); q.nrows()],
            |mut a, b| {
                for (ra, rb) in a.iter_mut().zip(b.iter()) {
                    ra.merge(rb, eps);
                }
                a
            },
        )
}

/// Per-value-row NULL flags, computed once per call: `v_ok[j]` is
/// false iff any component of value row `j` is NaN. Hoisting the scan
/// out of the pair loop keeps the per-pair cost at one bool load
/// (pairs typically outnumber value rows by the mask's fan-out).
fn value_row_ok(v: &ArrayView2<'_, f64>) -> Vec<bool> {
    (0..v.nrows())
        .map(|j| !v.row(j).iter().any(|c| c.is_nan()))
        .collect()
}

/// Masked attention over a pair stream (see module docs).
///
/// * `q`: `(N_q, d_k)` queries, indexed by the `i` of each pair.
/// * `k`, `v`: `(N_k, d_k)` keys and `(N_k, d_v)` values, indexed by `j`.
/// * `pairs`: the mask, as `(i, j)` index pairs in any order,
///   duplicate-free (duplicates are *not* detected and would be
///   double-counted, exactly as a bag-semantics mask would be).
/// * `eps`: temperature; `Eps::ZERO`, finite positive, and `Eps::INF`
///   are all supported.
///
/// ### NULL / non-finite semantics (pinned by `tests/numerical_edges.rs`,
/// mod `masked_attention_nan_policy`)
///
/// NaN is the engine's encoding of SQL NULL. A pair `(i, j)` is
/// SKIPPED, in every eps regime, when its score `q_i . k_j` is NaN
/// (NaN anywhere in `q_i` or `k_j` makes the dot NaN) or when any
/// component of value row `v_j` is NaN — the identical NULL discipline
/// of [`grouped_softavg`] (PG two-argument aggregates, cf.
/// `corr`/`covar_samp`). Infinite scores are real values, not NULLs:
/// see [`RowAcc::absorb`]'s policy. bruce-pg's `ScalarAcc` mirrors
/// this policy (C2).
///
/// Returns `(out, covered)` where `out` is `(N_q, d_v)` and
/// `covered[i]` is `false` (with a zero output row) iff no pair
/// mentioned row `i` or every pair mentioning it was skipped as NULL.
pub fn masked_attention(
    q: &ArrayView2<'_, f64>,
    k: &ArrayView2<'_, f64>,
    v: &ArrayView2<'_, f64>,
    pairs: &[(usize, usize)],
    eps: Eps,
) -> Result<(Array2<f64>, Vec<bool>), BruceError> {
    let n_q = q.nrows();
    let n_k = k.nrows();
    if v.nrows() != n_k {
        return Err(BruceError::DimensionMismatch {
            expected: n_k,
            got: v.nrows(),
        });
    }
    if q.ncols() != k.ncols() {
        return Err(BruceError::DimensionMismatch {
            expected: q.ncols(),
            got: k.ncols(),
        });
    }
    for &(i, j) in pairs {
        if i >= n_q || j >= n_k {
            return Err(BruceError::InvalidArgument(format!(
                "mask pair ({i}, {j}) out of range for N_q = {n_q}, N_k = {n_k}",
            )));
        }
    }

    let v_ok = value_row_ok(v);
    let accs = if pairs.len() < PAIR_PARALLEL_THRESHOLD {
        fold_sequential(q, k, v, pairs, &v_ok, eps)
    } else {
        fold_parallel(q, k, v, pairs, &v_ok, eps)
    };

    let d_v = v.ncols();
    let mut out = Array2::<f64>::zeros((n_q, d_v));
    let mut covered = vec![false; n_q];
    for (i, acc) in accs.iter().enumerate() {
        if let Some(row) = acc.finalize() {
            covered[i] = true;
            for (c, val) in row.into_iter().enumerate() {
                out[(i, c)] = val;
            }
        }
    }
    Ok((out, covered))
}

/// Convenience generator: the causal mask `{(i, j) : j ≤ i}` on `n`
/// rows, in row-major order. `n(n+1)/2` pairs.
pub fn causal_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut p = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in 0..=i {
            p.push((i, j));
        }
    }
    p
}

/// Convenience generator: the sliding-window mask
/// `{(i, j) : 0 ≤ i − j ≤ w}` on `n` rows. At most `n(w+1)` pairs.
pub fn window_pairs(n: usize, w: usize) -> Vec<(usize, usize)> {
    let mut p = Vec::with_capacity(n * (w + 1));
    for i in 0..n {
        let lo = i.saturating_sub(w);
        for j in lo..=i {
            p.push((i, j));
        }
    }
    p
}

/// Grouped soft-average: the fused physical operator for
/// `SELECT g, SOFTAVG(v WEIGHT sim(k, x) TEMP eps) ... GROUP BY g`.
///
/// Compared with routing the same computation through
/// [`masked_attention`], this operator (a) takes the grouping column as
/// a dictionary-encoded id array instead of a materialised `(i, j)`
/// pair stream, and (b) fuses an optional `eps = 0` selection mask into
/// the same pass, so filtered rows are never scored. One scan; per
/// group only the `(mu, z, u)` accumulator of the max-shifted
/// representation.
///
/// * `x`: `(d_k,)` query vector (shared by every group).
/// * `k`: `(N, d_k)` keys; `v`: `(N, d_v)` values.
/// * `gid`: `(N,)` group ids in `[0, n_groups)`.
/// * `sel`: optional `(N,)` selection; `false` rows are skipped
///   before scoring (the pushed-down exact filter).
/// * `eps`: temperature; all three regimes supported.
///
/// ### NULL / non-finite semantics (pinned by `tests/numerical_edges.rs`)
///
/// NaN is this engine's encoding of SQL NULL (bruce-query's ingest
/// maps Parquet NULL → NaN). SOFTAVG is a two-argument aggregate
/// (value, weight-score), and per PG's two-argument-aggregate NULL
/// discipline (C4; cf. `corr`, `covar_samp`: "rows with either input
/// null are ignored") a row is SKIPPED in **every** ε regime when its
/// score is NaN or **any** component of its value row is NaN (the
/// value is one vector datum). A group left with no surviving rows is
/// uncovered — SQL NULL. Infinite scores are real values, not NULLs:
/// `+inf` at finite ε takes argmax semantics, `-inf` weighs 0 (an
/// all-`-inf` group is uncovered), and ε = ∞ stays score-blind; see
/// `RowAcc::absorb` for the full policy.
///
/// Returns `(out, covered)`: `out` is `(n_groups, d_v)`; `covered[g]`
/// is `false` (zero row) iff no selected row carried group `g`.
pub fn grouped_softavg(
    x: &ArrayView1<'_, f64>,
    k: &ArrayView2<'_, f64>,
    v: &ArrayView2<'_, f64>,
    gid: &[u32],
    n_groups: usize,
    sel: Option<&[bool]>,
    eps: Eps,
) -> Result<(Array2<f64>, Vec<bool>), BruceError> {
    let n = k.nrows();
    if v.nrows() != n {
        return Err(BruceError::DimensionMismatch {
            expected: n,
            got: v.nrows(),
        });
    }
    if k.ncols() != x.len() {
        return Err(BruceError::DimensionMismatch {
            expected: x.len(),
            got: k.ncols(),
        });
    }
    if gid.len() != n {
        return Err(BruceError::DimensionMismatch {
            expected: n,
            got: gid.len(),
        });
    }
    if let Some(s) = sel {
        if s.len() != n {
            return Err(BruceError::DimensionMismatch {
                expected: n,
                got: s.len(),
            });
        }
    }
    if let Some(&g) = gid.iter().max() {
        if (g as usize) >= n_groups {
            return Err(BruceError::DimensionMismatch {
                expected: n_groups,
                got: g as usize + 1,
            });
        }
    }
    let d_v = v.ncols();

    let fold_range = |lo: usize, hi: usize| -> Vec<RowAcc> {
        let mut accs = vec![RowAcc::new(d_v); n_groups];
        for r in lo..hi {
            if let Some(s) = sel {
                if !s[r] {
                    continue;
                }
            }
            let score = x.dot(&k.row(r));
            let v_row = v.row(r);
            // SQL NULL discipline (C4): NaN encodes NULL; skip the row
            // if either aggregate argument is NULL, in every ε regime
            // (see the doc comment; pinned by tests/numerical_edges.rs).
            if score.is_nan() || v_row.iter().any(|c| c.is_nan()) {
                continue;
            }
            accs[gid[r] as usize].absorb(score, &v_row, eps);
        }
        accs
    };

    let accs = if n < PAIR_PARALLEL_THRESHOLD {
        fold_range(0, n)
    } else {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(n_threads);
        let bounds: Vec<(usize, usize)> = (0..n)
            .step_by(chunk)
            .map(|lo| (lo, (lo + chunk).min(n)))
            .collect();
        bounds
            .par_iter()
            .map(|&(lo, hi)| fold_range(lo, hi))
            .reduce(
                || vec![RowAcc::new(d_v); n_groups],
                |mut a, b| {
                    for (ra, rb) in a.iter_mut().zip(b.iter()) {
                        ra.merge(rb, eps);
                    }
                    a
                },
            )
    };

    let mut out = Array2::<f64>::zeros((n_groups, d_v));
    let mut covered = vec![false; n_groups];
    for (g, acc) in accs.iter().enumerate() {
        if let Some(row) = acc.finalize() {
            covered[g] = true;
            for (c, val) in row.into_iter().enumerate() {
                out[(g, c)] = val;
            }
        }
    }
    Ok((out, covered))
}

/// f32 dot product with 4-way unrolled partial sums. The unroll widens
/// the autovectorizer's window AND cuts each partial sum's sequential
/// rounding chain to length d/4 (pairwise-style error growth instead
/// of a single length-d chain).
///
/// SIMD note (2026-08-03, measured, change reverted per the >=5% keep
/// gate): an explicit AVX2+FMA kernel (4x 256-bit fmadd accumulators,
/// runtime `is_x86_feature_detected!` dispatch) was implemented and
/// benchmarked. Single-thread dot microbench at d=384: 1.87x faster
/// cache-resident (73.7 vs 39.6 GB/s), 1.27x in DRAM. But the full
/// `grouped_softavg_f32` operator is rayon-wide and its gated
/// 1M x d384 config is DRAM-bandwidth-bound (~60 GB/s aggregate), so
/// the criterion median moved only 25.60 -> 25.03 ms (-2.2%), below
/// the 5% keep threshold — the unsafe was not worth carrying. A safe
/// 8-way-unrolled variant was also measured and is SLOWER than this
/// 4-way form under the baseline x86-64 target (0.298 vs 0.159 ms
/// cache-resident; the wider accumulator array defeats the SSE2
/// autovectorizer). Numbers in
/// paper_sigmod_bruce/experiments/m2_mixed_precision/results_m2.json
/// (key "avx2_experiment_2026-08-03").
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n4 = a.len() - a.len() % 4;
    let mut acc = [0.0f32; 4];
    for (ca, cb) in a[..n4].chunks_exact(4).zip(b[..n4].chunks_exact(4)) {
        acc[0] += ca[0] * cb[0];
        acc[1] += ca[1] * cb[1];
        acc[2] += ca[2] * cb[2];
        acc[3] += ca[3] * cb[3];
    }
    for (x, y) in a[n4..].iter().zip(&b[n4..]) {
        acc[0] += x * y;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// f32-storage variant of [`grouped_softavg`]: identical contract, but
/// `k` and `x` are `f32` and each row's score is computed in f32.
///
/// Precision contract: **f32 storage and scoring, f64 accumulation.**
/// The dot product runs in f32 (4-way unrolled partial sums), the
/// score is widened to f64 exactly once per row, and everything
/// downstream — the max-shifted `(mu, z, u)` monoid, `exp`, the merge,
/// the finalize — is the same f64 [`RowAcc`] the f64 kernel uses.
/// Rationale: at the scan's scale the wall is memory bandwidth, so
/// halving the key bytes is the win; the anchoring that keeps sharp-eps
/// answers finite lives in the f64 fold and is unaffected by the
/// storage dtype. `v` stays f64 (one column; not the bandwidth term).
///
/// Same rayon chunk-reduce structure and threshold as the f64 kernel.
/// Non-contiguous key rows fall back to ndarray's f32 dot (still f32
/// scoring, sequential summation order).
pub fn grouped_softavg_f32(
    x: &ArrayView1<'_, f32>,
    k: &ArrayView2<'_, f32>,
    v: &ArrayView2<'_, f64>,
    gid: &[u32],
    n_groups: usize,
    sel: Option<&[bool]>,
    eps: Eps,
) -> Result<(Array2<f64>, Vec<bool>), BruceError> {
    let n = k.nrows();
    if v.nrows() != n {
        return Err(BruceError::DimensionMismatch {
            expected: n,
            got: v.nrows(),
        });
    }
    if k.ncols() != x.len() {
        return Err(BruceError::DimensionMismatch {
            expected: x.len(),
            got: k.ncols(),
        });
    }
    if gid.len() != n {
        return Err(BruceError::DimensionMismatch {
            expected: n,
            got: gid.len(),
        });
    }
    if let Some(s) = sel {
        if s.len() != n {
            return Err(BruceError::DimensionMismatch {
                expected: n,
                got: s.len(),
            });
        }
    }
    if let Some(&g) = gid.iter().max() {
        if (g as usize) >= n_groups {
            return Err(BruceError::DimensionMismatch {
                expected: n_groups,
                got: g as usize + 1,
            });
        }
    }
    let d_v = v.ncols();

    // one contiguous copy of the query vector (d floats, per call)
    let xs: Vec<f32> = x.iter().copied().collect();

    let fold_range = |lo: usize, hi: usize| -> Vec<RowAcc> {
        let mut accs = vec![RowAcc::new(d_v); n_groups];
        for r in lo..hi {
            if let Some(s) = sel {
                if !s[r] {
                    continue;
                }
            }
            let row = k.row(r);
            let s32 = match row.as_slice() {
                Some(rs) => dot_f32(&xs, rs),
                None => row.dot(x),
            };
            // widen once per row; the fold below is all-f64
            let score = s32 as f64;
            let v_row = v.row(r);
            // SQL NULL discipline (C4): NaN encodes NULL; skip the row
            // if either aggregate argument is NULL, in every ε regime
            // (same contract as grouped_softavg; pinned by
            // tests/numerical_edges.rs).
            if score.is_nan() || v_row.iter().any(|c| c.is_nan()) {
                continue;
            }
            accs[gid[r] as usize].absorb(score, &v_row, eps);
        }
        accs
    };

    let accs = if n < PAIR_PARALLEL_THRESHOLD {
        fold_range(0, n)
    } else {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(n_threads);
        let bounds: Vec<(usize, usize)> = (0..n)
            .step_by(chunk)
            .map(|lo| (lo, (lo + chunk).min(n)))
            .collect();
        bounds
            .par_iter()
            .map(|&(lo, hi)| fold_range(lo, hi))
            .reduce(
                || vec![RowAcc::new(d_v); n_groups],
                |mut a, b| {
                    for (ra, rb) in a.iter_mut().zip(b.iter()) {
                        ra.merge(rb, eps);
                    }
                    a
                },
            )
    };

    let mut out = Array2::<f64>::zeros((n_groups, d_v));
    let mut covered = vec![false; n_groups];
    for (g, acc) in accs.iter().enumerate() {
        if let Some(row) = acc.finalize() {
            covered[g] = true;
            for (c, val) in row.into_iter().enumerate() {
                out[(g, c)] = val;
            }
        }
    }
    Ok((out, covered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::softmax_eps;
    use crate::tree::{chain_tree, tree_causal_attention};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    /// Deterministic pseudo-random matrix (no rand dependency).
    fn pseudo(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut state = seed;
        Array2::from_shape_fn((n, d), |_| {
            // xorshift64*
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
    }

    /// Brute-force reference: per row, gather the masked records and
    /// apply `softmax_eps` directly.
    fn brute(
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        pairs: &[(usize, usize)],
        eps: Eps,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((q.nrows(), v.ncols()));
        for i in 0..q.nrows() {
            let js: Vec<usize> = pairs.iter().filter(|p| p.0 == i).map(|p| p.1).collect();
            if js.is_empty() {
                continue;
            }
            let scores: Vec<f64> = js.iter().map(|&j| q.row(i).dot(&k.row(j))).collect();
            let weights = if eps.is_inf() {
                vec![1.0 / js.len() as f64; js.len()]
            } else {
                softmax_eps(&scores, eps)
            };
            let mut row = Array1::<f64>::zeros(v.ncols());
            for (idx, &j) in js.iter().enumerate() {
                row.scaled_add(weights[idx], &v.row(j));
            }
            out.row_mut(i).assign(&row);
        }
        out
    }

    /// A fixed permutation with no rand dependency: stride through the
    /// indices by a step coprime to the length.
    fn shuffled<T: Clone>(xs: &[T]) -> Vec<T> {
        let n = xs.len();
        let mut step = (n / 2) | 1;
        while n.is_multiple_of(step) && step < n {
            step += 2;
        }
        (0..n).map(|t| xs[(t * step + 3) % n].clone()).collect()
    }

    #[test]
    fn causal_pairs_match_chain_tree_attention() {
        // The chain tree's ancestor sets are exactly the causal mask.
        let n = 24;
        let q = pseudo(n, 6, 1);
        let k = pseudo(n, 6, 2);
        let v = pseudo(n, 3, 3);
        for eps in [Eps::ONE, Eps(0.37)] {
            let (out, covered) =
                masked_attention(&q.view(), &k.view(), &v.view(), &causal_pairs(n), eps).unwrap();
            let reference =
                tree_causal_attention(&q.view(), &k.view(), &v.view(), &chain_tree(n), eps)
                    .unwrap();
            assert!(covered.iter().all(|&c| c));
            for (a, b) in out.iter().zip(reference.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn order_invariance_under_shuffle() {
        // The fold is a commutative-monoid homomorphism: any
        // enumeration order gives the same output (structure lemma).
        let n = 32;
        let q = pseudo(n, 5, 7);
        let k = pseudo(n, 5, 8);
        let v = pseudo(n, 4, 9);
        let pairs = window_pairs(n, 6);
        let perm = shuffled(&pairs);
        assert_ne!(pairs, perm);
        for eps in [Eps::ZERO, Eps(0.5), Eps::ONE, Eps::INF] {
            let (a, _) = masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();
            let (b, _) = masked_attention(&q.view(), &k.view(), &v.view(), &perm, eps).unwrap();
            for (x, y) in a.iter().zip(b.iter()) {
                assert_abs_diff_eq!(x, y, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn window_mask_matches_brute_force() {
        let n = 20;
        let q = pseudo(n, 4, 11);
        let k = pseudo(n, 4, 12);
        let v = pseudo(n, 2, 13);
        let pairs = window_pairs(n, 3);
        for eps in [Eps::ZERO, Eps(0.8), Eps::INF] {
            let (out, _) = masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();
            let reference = brute(&q, &k, &v, &pairs, eps);
            for (a, b) in out.iter().zip(reference.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn tropical_ties_take_uniform_argmax_mean() {
        // Two keys tie on the max score: ε = 0 must average their values.
        let q = ndarray::array![[1.0, 0.0]];
        let k = ndarray::array![[2.0, 0.0], [2.0, 0.0], [0.0, 5.0]];
        let v = ndarray::array![[10.0], [30.0], [999.0]];
        let pairs = vec![(0, 0), (0, 1), (0, 2)];
        let (out, covered) =
            masked_attention(&q.view(), &k.view(), &v.view(), &pairs, Eps::ZERO).unwrap();
        assert!(covered[0]);
        assert_abs_diff_eq!(out[(0, 0)], 20.0, epsilon = 1e-12);
    }

    #[test]
    fn eps_inf_is_plain_mean() {
        let q = ndarray::array![[1.0], [1.0]];
        let k = ndarray::array![[100.0], [-3.0], [5.0]];
        let v = ndarray::array![[3.0], [6.0], [9.0]];
        let pairs = vec![(0, 0), (0, 1), (0, 2), (1, 2)];
        let (out, _) = masked_attention(&q.view(), &k.view(), &v.view(), &pairs, Eps::INF).unwrap();
        assert_abs_diff_eq!(out[(0, 0)], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[(1, 0)], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn uncovered_rows_are_flagged() {
        let q = pseudo(3, 2, 21);
        let k = pseudo(2, 2, 22);
        let v = pseudo(2, 2, 23);
        let pairs = vec![(0, 0), (2, 1)];
        let (out, covered) =
            masked_attention(&q.view(), &k.view(), &v.view(), &pairs, Eps::ONE).unwrap();
        assert_eq!(covered, vec![true, false, true]);
        assert_eq!(out[(1, 0)], 0.0);
        assert_eq!(out[(1, 1)], 0.0);
    }

    #[test]
    fn parallel_fold_equals_sequential_fold() {
        // Partition-reduce (Lemma B) in code: the chunked-parallel fold
        // must agree with the sequential one for every regime.
        let n = 48;
        let q = pseudo(n, 4, 31);
        let k = pseudo(n, 4, 32);
        let v = pseudo(n, 3, 33);
        let pairs = causal_pairs(n);
        let v_ok = value_row_ok(&v.view());
        for eps in [Eps::ZERO, Eps(0.9), Eps::INF] {
            let seq = fold_sequential(&q.view(), &k.view(), &v.view(), &pairs, &v_ok, eps);
            let par = fold_parallel(&q.view(), &k.view(), &v.view(), &pairs, &v_ok, eps);
            for (a, b) in seq.iter().zip(par.iter()) {
                match (a.finalize(), b.finalize()) {
                    (Some(x), Some(y)) => {
                        for (xc, yc) in x.iter().zip(y.iter()) {
                            assert_abs_diff_eq!(xc, yc, epsilon = 1e-12);
                        }
                    }
                    (None, None) => {}
                    _ => panic!("coverage mismatch between folds"),
                }
            }
        }
    }

    #[test]
    fn rejects_out_of_range_pairs() {
        let q = pseudo(2, 2, 41);
        let k = pseudo(2, 2, 42);
        let v = pseudo(2, 2, 43);
        let r = masked_attention(&q.view(), &k.view(), &v.view(), &[(0, 5)], Eps::ONE);
        assert!(r.is_err());
    }

    /// PARALLEL-003: grouped_softavg must agree with masked_attention
    /// routed through an explicit (group, row) pair stream, in every
    /// temperature regime, with and without a fused selection.
    #[test]
    fn grouped_softavg_matches_masked_attention() {
        let n = 500;
        let d_k = 8;
        let d_v = 3;
        let n_groups = 7;
        let k = pseudo(n, d_k, 11);
        let v = pseudo(n, d_v, 12);
        let xq = pseudo(1, d_k, 13);
        let x = xq.row(0).to_owned();
        let gid: Vec<u32> = (0..n).map(|r| ((r * 31 + 7) % n_groups) as u32).collect();
        let sel: Vec<bool> = (0..n).map(|r| r % 3 != 0).collect();

        for eps in [Eps::ZERO, Eps(0.7), Eps::INF] {
            for use_sel in [false, true] {
                // reference: pair stream through masked_attention
                let mut q = Array2::<f64>::zeros((n_groups, d_k));
                for g in 0..n_groups {
                    for c in 0..d_k {
                        q[(g, c)] = x[c];
                    }
                }
                let pairs: Vec<(usize, usize)> = (0..n)
                    .filter(|&r| !use_sel || sel[r])
                    .map(|r| (gid[r] as usize, r))
                    .collect();
                let (want, want_cov) =
                    masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();

                let (got, got_cov) = grouped_softavg(
                    &x.view(),
                    &k.view(),
                    &v.view(),
                    &gid,
                    n_groups,
                    if use_sel { Some(&sel) } else { None },
                    eps,
                )
                .unwrap();

                assert_eq!(want_cov, got_cov);
                for g in 0..n_groups {
                    for c in 0..d_v {
                        assert_abs_diff_eq!(want[(g, c)], got[(g, c)], epsilon = 1e-12);
                    }
                }
            }
        }
    }

    /// MIXED-001: the f32 kernel must agree with the f64 kernel run on
    /// the SAME stored numbers (f32 values upcast) to within the
    /// scoring-precision budget: the only difference between the two
    /// paths is the f32 dot product, so the output error is bounded by
    /// the score rounding amplified by 1/eps. Covered flags must agree
    /// exactly. Values are offset away from zero so relative error is
    /// well-defined.
    #[test]
    fn grouped_softavg_f32_matches_f64_kernel() {
        let n = 4000;
        let d_k = 16;
        let d_v = 2;
        let n_groups = 11;
        let k32 = pseudo(n, d_k, 51).mapv(|x| x as f32);
        let k64 = k32.mapv(|x| x as f64); // identical stored numbers
        let v = pseudo(n, d_v, 52).mapv(|x| x + 2.0);
        let x32 = pseudo(1, d_k, 53).row(0).mapv(|x| x as f32);
        let x64 = x32.mapv(|x| x as f64);
        let gid: Vec<u32> = (0..n).map(|r| ((r * 29 + 5) % n_groups) as u32).collect();
        let sel: Vec<bool> = (0..n).map(|r| r % 7 != 0).collect();

        for eps in [Eps(0.1), Eps::ONE] {
            for use_sel in [false, true] {
                let s = if use_sel { Some(sel.as_slice()) } else { None };
                let (want, want_cov) =
                    grouped_softavg(&x64.view(), &k64.view(), &v.view(), &gid, n_groups, s, eps)
                        .unwrap();
                let (got, got_cov) = grouped_softavg_f32(
                    &x32.view(),
                    &k32.view(),
                    &v.view(),
                    &gid,
                    n_groups,
                    s,
                    eps,
                )
                .unwrap();
                assert_eq!(want_cov, got_cov);
                for g in 0..n_groups {
                    for c in 0..d_v {
                        let rel = (got[(g, c)] - want[(g, c)]).abs() / want[(g, c)].abs();
                        assert!(
                            rel < 1e-5,
                            "eps={:?} sel={use_sel} group {g} col {c}: rel err {rel:e}",
                            eps
                        );
                    }
                }
            }
        }
    }

    /// MIXED-002: chunking must not change the f32 kernel's answer
    /// beyond f64 accumulation-order noise — the scores are identical
    /// f32 values in every chunking, only the f64 merge order differs.
    #[test]
    fn grouped_softavg_f32_parallel_matches_single_thread() {
        let n = (1 << 15) + 999;
        let d_k = 8;
        let k = pseudo(n, d_k, 61).mapv(|x| x as f32);
        let v = pseudo(n, 2, 62).mapv(|x| x + 2.0);
        let x = pseudo(1, d_k, 63).row(0).mapv(|x| x as f32);
        let gid: Vec<u32> = (0..n).map(|r| (r % 13) as u32).collect();
        let (par, cov_par) =
            grouped_softavg_f32(&x.view(), &k.view(), &v.view(), &gid, 13, None, Eps(0.5)).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (seq, cov_seq) = pool
            .install(|| {
                grouped_softavg_f32(&x.view(), &k.view(), &v.view(), &gid, 13, None, Eps(0.5))
            })
            .unwrap();
        assert_eq!(cov_par, cov_seq);
        for g in 0..13 {
            for c in 0..2 {
                assert_abs_diff_eq!(par[(g, c)], seq[(g, c)], epsilon = 1e-12);
            }
        }
    }

    /// PARALLEL-004: result must not depend on the chunking. We force
    /// the parallel path by exceeding the threshold and compare against
    /// a hand-rolled sequential fold.
    #[test]
    fn grouped_softavg_parallel_matches_sequential() {
        let n = (1 << 15) + 1234;
        let d_k = 4;
        let k = pseudo(n, d_k, 21);
        let v = pseudo(n, 2, 22);
        let xq = pseudo(1, d_k, 23);
        let x = xq.row(0).to_owned();
        let gid: Vec<u32> = (0..n).map(|r| (r % 13) as u32).collect();
        let (par, _) =
            grouped_softavg(&x.view(), &k.view(), &v.view(), &gid, 13, None, Eps(0.5)).unwrap();

        // sequential reference via the pair path below threshold is too
        // slow to build here; instead run the same operator on a single
        // rayon thread pool of one.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (seq, _) = pool
            .install(|| grouped_softavg(&x.view(), &k.view(), &v.view(), &gid, 13, None, Eps(0.5)))
            .unwrap();
        for g in 0..13 {
            for c in 0..2 {
                assert_abs_diff_eq!(par[(g, c)], seq[(g, c)], epsilon = 1e-12);
            }
        }
    }
}

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
//! row is zero and the returned `covered[i]` flag is `false`.

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
    #[inline]
    fn absorb(&mut self, s: f64, v_row: &ArrayView1<'_, f64>, eps: Eps) {
        if eps.is_zero() {
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
    eps: Eps,
) -> Vec<RowAcc> {
    let mut accs = vec![RowAcc::new(v.ncols()); q.nrows()];
    for &(i, j) in pairs {
        let s = q.row(i).dot(&k.row(j));
        accs[i].absorb(s, &v.row(j), eps);
    }
    accs
}

fn fold_parallel(
    q: &ArrayView2<'_, f64>,
    k: &ArrayView2<'_, f64>,
    v: &ArrayView2<'_, f64>,
    pairs: &[(usize, usize)],
    eps: Eps,
) -> Vec<RowAcc> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = pairs.len().div_ceil(n_threads);
    pairs
        .par_chunks(chunk)
        .map(|c| fold_sequential(q, k, v, c, eps))
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
/// Returns `(out, covered)` where `out` is `(N_q, d_v)` and
/// `covered[i]` is `false` (with a zero output row) iff no pair
/// mentioned row `i`.
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

    let accs = if pairs.len() < PAIR_PARALLEL_THRESHOLD {
        fold_sequential(q, k, v, pairs, eps)
    } else {
        fold_parallel(q, k, v, pairs, eps)
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
            let js: Vec<usize> =
                pairs.iter().filter(|p| p.0 == i).map(|p| p.1).collect();
            if js.is_empty() {
                continue;
            }
            let scores: Vec<f64> =
                js.iter().map(|&j| q.row(i).dot(&k.row(j))).collect();
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
        while n % step == 0 && step < n {
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
            let (out, covered) = masked_attention(
                &q.view(), &k.view(), &v.view(), &causal_pairs(n), eps,
            )
            .unwrap();
            let reference = tree_causal_attention(
                &q.view(), &k.view(), &v.view(), &chain_tree(n), eps,
            )
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
            let (a, _) =
                masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps)
                    .unwrap();
            let (b, _) =
                masked_attention(&q.view(), &k.view(), &v.view(), &perm, eps)
                    .unwrap();
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
            let (out, _) =
                masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps)
                    .unwrap();
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
        let (out, covered) = masked_attention(
            &q.view(), &k.view(), &v.view(), &pairs, Eps::ZERO,
        )
        .unwrap();
        assert!(covered[0]);
        assert_abs_diff_eq!(out[(0, 0)], 20.0, epsilon = 1e-12);
    }

    #[test]
    fn eps_inf_is_plain_mean() {
        let q = ndarray::array![[1.0], [1.0]];
        let k = ndarray::array![[100.0], [-3.0], [5.0]];
        let v = ndarray::array![[3.0], [6.0], [9.0]];
        let pairs = vec![(0, 0), (0, 1), (0, 2), (1, 2)];
        let (out, _) = masked_attention(
            &q.view(), &k.view(), &v.view(), &pairs, Eps::INF,
        )
        .unwrap();
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
            masked_attention(&q.view(), &k.view(), &v.view(), &pairs, Eps::ONE)
                .unwrap();
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
        for eps in [Eps::ZERO, Eps(0.9), Eps::INF] {
            let seq = fold_sequential(&q.view(), &k.view(), &v.view(), &pairs, eps);
            let par = fold_parallel(&q.view(), &k.view(), &v.view(), &pairs, eps);
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
        let r = masked_attention(
            &q.view(), &k.view(), &v.view(), &[(0, 5)], Eps::ONE,
        );
        assert!(r.is_err());
    }
}

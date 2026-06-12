//! The F_ε operator: the central object of Bruce.
//!
//! Given a query `x`, a key matrix `K`, and a value matrix `V`:
//!
//! ```text
//!     wⱼ(ε)  =  exp(sim(x, kⱼ) / ε)
//!     A_ε    =  (Σⱼ wⱼ · vⱼ)  /  (Σⱼ wⱼ)        — Bruce attention
//!     Q_ε    =   Σⱼ wⱼ · vⱼ                      — Bruce sum (SQL-style)
//! ```
//!
//! At ε = 1 with `sim = Dot`, `A_ε` is the standard softmax attention.
//! At ε = 0 with `sim = Indicator`, `Q_ε` is SQL equi-join + GROUP BY.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

use crate::semiring::softmax_eps;
use crate::types::{Aggregator, Eps, Sim};

/// Below this row count, sequential is faster (rayon overhead dominates).
const PARALLEL_THRESHOLD: usize = 1024;

/// The F_ε operator.
///
/// A configuration object — cheap to clone, no mutable state inside.
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct F_eps {
    /// Temperature `ε`. Determines the semiring.
    pub eps: Eps,
    /// Similarity function applied between query and keys.
    pub sim: Sim,
    /// Which aggregation flavour to return.
    pub agg: Aggregator,
}

impl F_eps {
    /// Construct a new F_ε at the given temperature and similarity.
    /// The aggregator defaults to `Softmax` (attention).
    pub fn new(eps: Eps, sim: Sim) -> Self {
        Self {
            eps,
            sim,
            agg: Aggregator::Softmax,
        }
    }

    /// Build with a non-default aggregator (e.g. `Sum` for SQL semantics).
    pub fn with_agg(mut self, agg: Aggregator) -> Self {
        self.agg = agg;
        self
    }

    /// Compute scores `sim(x, kⱼ)` for all rows `j` of `K`.
    ///
    /// For `Sim::Dot`, uses ndarray's native `K.dot(x)` matrix-vector
    /// product (contiguous, SIMD-friendly, cache-blocked); this is
    /// significantly faster than a manual per-row loop. For
    /// `Sim::NegSquared` / `Sim::Indicator` (rare paths), falls back
    /// to per-row rayon-parallel.
    #[inline]
    pub fn scores(&self, x: &ArrayView1<'_, f64>, k: &ArrayView2<'_, f64>) -> Vec<f64> {
        let n = k.nrows();
        match self.sim {
            Sim::Dot => {
                // `K.dot(x)` returns `Array1<f64>` of length N; this is
                // the standard matrix-vector product. ndarray handles
                // SIMD + cache blocking internally; with the BLAS
                // feature it would dispatch to dgemv.
                k.dot(x).to_vec()
            }
            Sim::NegSquared => {
                if n < PARALLEL_THRESHOLD {
                    (0..n)
                        .map(|j| {
                            let diff = x - &k.row(j);
                            -0.5 * diff.dot(&diff)
                        })
                        .collect()
                } else {
                    (0..n)
                        .into_par_iter()
                        .map(|j| {
                            let diff = x - &k.row(j);
                            -0.5 * diff.dot(&diff)
                        })
                        .collect()
                }
            }
            Sim::Indicator => {
                if n < PARALLEL_THRESHOLD {
                    (0..n)
                        .map(|j| {
                            let diff = x - &k.row(j);
                            let l2 = diff.dot(&diff);
                            if l2 == 0.0 { 0.0 } else { f64::NEG_INFINITY }
                        })
                        .collect()
                } else {
                    (0..n)
                        .into_par_iter()
                        .map(|j| {
                            let diff = x - &k.row(j);
                            let l2 = diff.dot(&diff);
                            if l2 == 0.0 { 0.0 } else { f64::NEG_INFINITY }
                        })
                        .collect()
                }
            }
        }
    }

    /// Batched attention: `[A_ε(Q[i], K, V)]_i` for B queries at once.
    ///
    /// Computes the full B×N similarity matrix via `Q @ K^T` (one
    /// large matmul), then softmax-per-row, then `weights @ V` — the
    /// total cost is dominated by two large matrix multiplies, which
    /// is dramatically faster than B separate per-query calls when
    /// B is large (typical throughput workloads).
    ///
    /// Only supports `Sim::Dot` (the softmax-attention path) for now;
    /// indicator / neg-squared keep the per-query API.
    pub fn attention_batch(
        &self,
        q: &ArrayView2<'_, f64>,  // (B, d_k)
        k: &ArrayView2<'_, f64>,  // (N, d_k)
        v: &ArrayView2<'_, f64>,  // (N, d_v)
    ) -> ndarray::Array2<f64> {
        assert!(matches!(self.sim, Sim::Dot),
                "attention_batch only supports Sim::Dot");
        let b_count = q.nrows();
        let n = k.nrows();
        let d_v = v.ncols();
        // scores: (B, N)
        let scores = q.dot(&k.t());
        // softmax per row + multiply V
        let mut out = ndarray::Array2::<f64>::zeros((b_count, d_v));
        let eps_val = self.eps.0;
        // rayon-parallel across queries
        let rows: Vec<ndarray::Array1<f64>> = (0..b_count)
            .into_par_iter()
            .map(|i| {
                let row = scores.row(i);
                let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if !m.is_finite() {
                    return ndarray::Array1::<f64>::zeros(d_v);
                }
                let mut weights = ndarray::Array1::<f64>::zeros(n);
                let mut z = 0.0_f64;
                for (j, &s) in row.iter().enumerate() {
                    let w = ((s - m) / eps_val).exp();
                    weights[j] = w;
                    z += w;
                }
                if z == 0.0 {
                    return ndarray::Array1::<f64>::zeros(d_v);
                }
                let w_norm = &weights / z;
                w_norm.dot(v)
            })
            .collect();
        for (i, r) in rows.into_iter().enumerate() {
            out.row_mut(i).assign(&r);
        }
        out
    }

    /// Standard attention: `A_ε(x, K, V) = softmax(scores / ε) · V`.
    ///
    /// At ε = 0 with `Sim::Indicator` this reduces to the SQL semantics
    /// where the result equals the average value over rows whose key
    /// exactly matches the query.
    pub fn attention(
        &self,
        x: &ArrayView1<'_, f64>,
        k: &ArrayView2<'_, f64>,
        v: &ArrayView2<'_, f64>,
    ) -> Array1<f64> {
        let scores = self.scores(x, k);
        let weights = softmax_eps(&scores, self.eps);
        let mut out = Array1::<f64>::zeros(v.ncols());
        for (j, w) in weights.iter().enumerate() {
            if *w == 0.0 {
                continue;
            }
            out.scaled_add(*w, &v.row(j));
        }
        out
    }

    /// SQL-style un-normalised sum: `Σⱼ wⱼ · vⱼ` (no normalisation).
    ///
    /// At ε = 0 with `Sim::Indicator` this is `SELECT SUM(v) WHERE k = x`.
    pub fn sum(
        &self,
        x: &ArrayView1<'_, f64>,
        k: &ArrayView2<'_, f64>,
        v: &ArrayView2<'_, f64>,
    ) -> Array1<f64> {
        let scores = self.scores(x, k);
        let n = scores.len();
        if n == 0 {
            return Array1::<f64>::zeros(v.ncols());
        }
        // For ε = 0, this is just sum of v rows whose score equals max
        if self.eps.is_zero() {
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut out = Array1::<f64>::zeros(v.ncols());
            for (j, &s) in scores.iter().enumerate() {
                if s == m && s.is_finite() {
                    out.scaled_add(1.0, &v.row(j));
                }
            }
            return out;
        }
        // Otherwise, use a numerically-stable shift
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return Array1::<f64>::zeros(v.ncols());
        }
        let mut out = Array1::<f64>::zeros(v.ncols());
        let mut z = 0.0;
        for (j, &s) in scores.iter().enumerate() {
            let w = ((s - m) / self.eps.0).exp();
            out.scaled_add(w, &v.row(j));
            z += w;
        }
        let _ = z;
        // we return the m-shifted sum; full sum requires the m factor
        out * (m / self.eps.0).exp()
    }

    /// Count of records with maximum (positive) weight.
    /// At ε = 0 this is `SELECT COUNT(*) WHERE k = x`.
    pub fn count(&self, x: &ArrayView1<'_, f64>, k: &ArrayView2<'_, f64>) -> f64 {
        let scores = self.scores(x, k);
        if self.eps.is_zero() {
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            return scores
                .iter()
                .filter(|&&s| s == m && s.is_finite())
                .count() as f64;
        }
        let weights = softmax_eps(&scores, self.eps);
        weights.iter().filter(|&&w| w > 0.0).count() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn attention_eps_one_matches_known_value() {
        // tiny example we can compute by hand
        // x = [1, 0], K = [[1, 0], [0, 1]], V = [[10, 0], [0, 10]]
        // scores = [1, 0]; weights = softmax([1, 0]) = [e/(e+1), 1/(e+1)]
        // out = [10·e/(e+1), 10/(e+1)] ≈ [7.31, 2.69]
        let x = array![1.0, 0.0];
        let k = array![[1.0, 0.0], [0.0, 1.0]];
        let v = array![[10.0, 0.0], [0.0, 10.0]];
        let op = F_eps::new(Eps::ONE, Sim::Dot);
        let out = op.attention(&x.view(), &k.view(), &v.view());
        let e = std::f64::consts::E;
        assert_abs_diff_eq!(out[0], 10.0 * e / (e + 1.0), epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], 10.0 / (e + 1.0), epsilon = 1e-12);
    }

    #[test]
    fn at_eps_zero_indicator_recovers_sql_groupby() {
        // SQL: SELECT SUM(v) WHERE k = x
        // x = [1, 0]; K has two rows matching x and one not.
        let x = array![1.0, 0.0];
        let k = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let v = array![[5.0], [7.0], [99.0]];
        let op = F_eps::new(Eps::ZERO, Sim::Indicator).with_agg(Aggregator::Sum);
        let out = op.sum(&x.view(), &k.view(), &v.view());
        // exact two rows match; sum should be 12.0
        assert_eq!(out[0], 12.0);
    }

    #[test]
    fn count_at_eps_zero_indicator_is_sql_count() {
        let x = array![1.0, 0.0];
        let k = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let op = F_eps::new(Eps::ZERO, Sim::Indicator);
        let c = op.count(&x.view(), &k.view());
        assert_eq!(c, 3.0);
    }

    #[test]
    fn attention_with_zero_values_returns_zero() {
        let x = array![1.0, 0.0];
        let k = array![[1.0, 0.0]];
        let v = array![[0.0, 0.0]];
        let op = F_eps::new(Eps::ONE, Sim::Dot);
        let out = op.attention(&x.view(), &k.view(), &v.view());
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }
}

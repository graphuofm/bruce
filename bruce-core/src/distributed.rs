//! Distributed F_ε via partition-reduce (Lemma B).
//!
//! Lemma B states that the F_ε operator is exact under arbitrary
//! partitioning of the K/V memory:
//!
//! ```text
//!     for partition P = {P_1, ..., P_p} of {1..N}:
//!         A_ε(x, K, V)  =  Σᵢ A_ε(x, K|_Pᵢ, V|_Pᵢ) · wᵢ
//! ```
//!
//! where the weights `wᵢ` come from the partition function. In
//! practice we don't combine via weights — we combine via the
//! (m_local, num_local, den_local) triples that each partition emits.
//!
//! This module gives the **algebra** of partition-reduce. The actual
//! cross-node transport is the caller's job (Bruce ships with a
//! reference MPI/Tonic implementation in `bruce-server` / examples).

use ndarray::Array1;

use crate::types::Eps;

/// One partition's contribution: running max + (m-shifted) numerator
/// and denominator.
#[derive(Debug, Clone)]
pub struct PartialTriple {
    /// Local max of this partition's scores.
    pub m_local: f64,
    /// Local m-shifted numerator: Σⱼ exp((sⱼ − m_local) / ε) · vⱼ.
    pub num_local: Array1<f64>,
    /// Local m-shifted denominator: Σⱼ exp((sⱼ − m_local) / ε).
    pub den_local: f64,
}

impl PartialTriple {
    /// An empty (identity) partial.
    pub fn empty(d_v: usize) -> Self {
        Self {
            m_local: f64::NEG_INFINITY,
            num_local: Array1::<f64>::zeros(d_v),
            den_local: 0.0,
        }
    }

    /// Build from a slice of (score, value) pairs.
    pub fn from_pairs(scores: &[f64], values: &[Array1<f64>], eps: Eps) -> Self {
        debug_assert_eq!(scores.len(), values.len());
        let d_v = values.first().map(|v| v.len()).unwrap_or(0);
        if scores.is_empty() || d_v == 0 {
            return Self::empty(d_v);
        }
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return Self::empty(d_v);
        }
        let mut num = Array1::<f64>::zeros(d_v);
        let mut den = 0.0;
        if eps.is_zero() {
            // tropical: only argmax contributes weight = 1
            for (s, v) in scores.iter().zip(values) {
                if *s == m {
                    num.scaled_add(1.0, v);
                    den += 1.0;
                }
            }
        } else {
            for (s, v) in scores.iter().zip(values) {
                let w = ((s - m) / eps.0).exp();
                num.scaled_add(w, v);
                den += w;
            }
        }
        Self {
            m_local: m,
            num_local: num,
            den_local: den,
        }
    }
}

/// Merge a sequence of partial triples into a single (M, num, den) at
/// the global max M. This is the **reduce** step of partition-reduce.
///
/// `eps = 0` triggers the tropical merge: only partitions that hit
/// the global max contribute.
pub fn combine(partials: &[PartialTriple], eps: Eps) -> PartialTriple {
    if partials.is_empty() {
        return PartialTriple::empty(0);
    }
    let d_v = partials[0].num_local.len();
    let big_m = partials
        .iter()
        .map(|p| p.m_local)
        .fold(f64::NEG_INFINITY, f64::max);
    if !big_m.is_finite() {
        return PartialTriple::empty(d_v);
    }
    let mut num = Array1::<f64>::zeros(d_v);
    let mut den = 0.0;
    if eps.is_zero() {
        for p in partials {
            if p.m_local == big_m {
                num.scaled_add(1.0, &p.num_local);
                den += p.den_local;
            }
        }
    } else {
        for p in partials {
            let factor = ((p.m_local - big_m) / eps.0).exp();
            num.scaled_add(factor, &p.num_local);
            den += p.den_local * factor;
        }
    }
    PartialTriple {
        m_local: big_m,
        num_local: num,
        den_local: den,
    }
}

/// Final attention output from a combined triple.
pub fn finalize(combined: &PartialTriple) -> Array1<f64> {
    if combined.den_local <= 0.0 {
        Array1::<f64>::zeros(combined.num_local.len())
    } else {
        &combined.num_local / combined.den_local
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn partition_reduce_matches_single_machine() {
        let n = 1000;
        let d_v = 4;
        let eps = Eps::ONE;
        // synth N rows
        let scores: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
        let values: Vec<Array1<f64>> = (0..n)
            .map(|i| Array1::<f64>::from_elem(d_v, i as f64))
            .collect();

        // single-machine reference
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut num = Array1::<f64>::zeros(d_v);
        let mut den = 0.0;
        for (s, v) in scores.iter().zip(&values) {
            let w = ((s - m) / eps.0).exp();
            num.scaled_add(w, v);
            den += w;
        }
        let ref_out = num / den;

        // partition into P partitions, partition-reduce
        for p in [2, 4, 8, 16, 64, 128] {
            let chunk = n / p;
            let partials: Vec<PartialTriple> = (0..p)
                .map(|i| {
                    let lo = i * chunk;
                    let hi = if i == p - 1 { n } else { (i + 1) * chunk };
                    PartialTriple::from_pairs(&scores[lo..hi], &values[lo..hi], eps)
                })
                .collect();
            let combined = combine(&partials, eps);
            let out = finalize(&combined);
            // partition-reduce should match single-machine to machine ε,
            // which is ~|x| · 2.22e-16 in absolute terms
            let scale = ref_out[0].abs().max(1.0);
            for j in 0..d_v {
                assert_abs_diff_eq!(out[j], ref_out[j], epsilon = scale * 1e-12);
            }
        }
    }

    #[test]
    fn tropical_partition_reduce_correct() {
        // ε = 0: only argmax contributes
        let scores = vec![1.0, 5.0, 2.0, 5.0, 3.0];
        let values: Vec<Array1<f64>> = (0..5).map(|i| array![i as f64]).collect();
        let p1 = PartialTriple::from_pairs(&scores[..2], &values[..2], Eps::ZERO);
        let p2 = PartialTriple::from_pairs(&scores[2..], &values[2..], Eps::ZERO);
        let combined = combine(&[p1, p2], Eps::ZERO);
        let out = finalize(&combined);
        // argmax at indices 1, 3 → sum = 1 + 3 = 4; count = 2; mean = 2
        assert_eq!(out[0], 2.0);
    }
}

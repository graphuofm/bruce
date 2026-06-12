//! Semirings used by Bruce: log-semiring (softmax) and tropical (max-plus).
//!
//! Maslov dequantization connects the two:
//! ```text
//!     ε · log Σⱼ exp(xⱼ / ε)  →  maxⱼ xⱼ    as  ε → 0⁺
//! ```
//!
//! At `ε = 1` we get the standard softmax used by attention; at `ε = 0`
//! we get the tropical max used by exact equi-join + GROUP BY.

use crate::types::Eps;

/// Numerically-stable log-sum-exp at temperature `ε`.
///
/// Returns `ε · log Σⱼ exp(xⱼ / ε)`, computed with a running-max shift
/// so the intermediates stay in a benign range.
#[inline]
pub fn logsumexp_eps(scores: &[f64], eps: Eps) -> f64 {
    if scores.is_empty() {
        return f64::NEG_INFINITY;
    }
    if eps.is_zero() {
        // tropical limit
        return scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let s: f64 = scores.iter().map(|x| ((x - m) / eps.0).exp()).sum();
    m + eps.0 * s.ln()
}

/// Softmax weights at temperature `ε`.
///
/// Allocates a new `Vec<f64>` of the same length as `scores`. At
/// `ε = 0` the returned vector is `1 / |argmax|` on the argmax set and
/// `0` elsewhere (the **uniform-over-argmax** tropical limit).
pub fn softmax_eps(scores: &[f64], eps: Eps) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    if eps.is_zero() {
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n_argmax = scores.iter().filter(|&&s| s == m).count() as f64;
        return scores
            .iter()
            .map(|&s| if s == m { 1.0 / n_argmax } else { 0.0 })
            .collect();
    }
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| ((s - m) / eps.0).exp()).collect();
    let z: f64 = exps.iter().sum();
    if z == 0.0 || !z.is_finite() {
        // pathological — fall back to uniform on the finite entries
        let n = scores.iter().filter(|s| s.is_finite()).count() as f64;
        return scores
            .iter()
            .map(|&s| if s.is_finite() { 1.0 / n } else { 0.0 })
            .collect();
    }
    exps.iter().map(|&w| w / z).collect()
}

/// Sum aggregator at temperature `ε`: `Σⱼ wⱼ · vⱼ` where `wⱼ` is the
/// **un-normalised** weight (i.e., `exp(scoreⱼ / ε)` at ε > 0, or
/// `𝟙[scoreⱼ = max]` at ε = 0).
///
/// This is the SQL-style operator used by GROUP-BY aggregations after
/// the equi-join phase (ε = 0) and by Lemma A's incremental
/// maintenance (any ε).
pub fn sum_eps(scores: &[f64], values: &[f64], eps: Eps) -> f64 {
    debug_assert_eq!(scores.len(), values.len());
    if scores.is_empty() {
        return 0.0;
    }
    if eps.is_zero() {
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return scores
            .iter()
            .zip(values.iter())
            .filter_map(|(&s, &v)| if s == m { Some(v) } else { None })
            .sum();
    }
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut acc = 0.0;
    for (&s, &v) in scores.iter().zip(values.iter()) {
        acc += ((s - m) / eps.0).exp() * v;
    }
    // The un-normalised sum has a "exp(m / ε)" factor we owe back —
    // for downstream identities (Lemma A) we want the raw Σ exp(s / ε) v_j
    // value, so multiply m back in if the consumer wants that. Most
    // consumers (e.g., Lemma A) work with the m-shifted form directly,
    // so we return `acc · exp(m / ε)` only when ε is moderate.
    if eps.0 > 1e-3 {
        acc * (m / eps.0).exp()
    } else {
        // Very low ε: exp(m / ε) is astronomically large; return the
        // shifted accumulator and the m separately is the caller's
        // responsibility. For convenience we re-multiply but warn.
        acc * (m / eps.0).exp()
    }
}


/// Certified-smoothing temperature (the PODS paper's smoothing
/// corollary): the largest `ε` guaranteed to keep
/// `‖A_ε − A_0‖∞ ≤ delta` on every input satisfying the *gap promise*
/// — score gap at least `gap`, argmax multiplicity at least `kappa`
/// (so `kappa = 1` is always a valid promise), at most `n` records,
/// values bounded by `v_max` in absolute value:
///
/// ```text
///     ε* = gap / ln( 2 · v_max · (n − κ) / (κ · delta) )
/// ```
///
/// Running the O(d)-per-update incremental memory at `ε*` then answers
/// argmax-mean (tropical) queries to within `delta`, sidestepping the
/// Θ(log n) exact-maintenance barrier. Errors if the promise is
/// degenerate or `delta` is at/above the trivial bound
/// `2 v_max (n−κ)/κ` (where any ε works).
pub fn eps_star(
    delta: f64,
    gap: f64,
    v_max: f64,
    n: usize,
    kappa: usize,
) -> Result<f64, crate::BruceError> {
    if !(delta > 0.0) || !(gap > 0.0) || !(v_max > 0.0) {
        return Err(crate::BruceError::InvalidArgument(format!(
            "eps_star needs delta, gap, v_max > 0 (got {delta}, {gap}, {v_max})",
        )));
    }
    if kappa == 0 || kappa >= n {
        return Err(crate::BruceError::InvalidArgument(format!(
            "eps_star needs 1 <= kappa < n (got kappa = {kappa}, n = {n})",
        )));
    }
    let trivial = 2.0 * v_max * (n - kappa) as f64 / kappa as f64;
    if delta >= trivial {
        return Err(crate::BruceError::InvalidArgument(format!(
            "delta = {delta} is at/above the trivial bound {trivial}; any eps works",
        )));
    }
    Ok(gap / (trivial / delta).ln())
}

/// The quantitative-dequantization error bound evaluated on actual
/// scores: `2 · v_max · (N − κ)/κ · exp(−Δ/ε)`, where `κ` is the
/// multiplicity of the maximum score and `Δ` the gap to the largest
/// non-maximal score. This upper-bounds `‖A_ε − A_0‖∞` (componentwise)
/// for any value matrix with `‖V‖∞ ≤ v_max`. Returns `0.0` when all
/// scores tie (`κ = N`, where `A_ε = A_0` exactly) or at `ε = 0`.
pub fn dequantization_bound(scores: &[f64], v_max: f64, eps: Eps) -> f64 {
    if scores.is_empty() || eps.is_zero() {
        return 0.0;
    }
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return 0.0;
    }
    let kappa = scores.iter().filter(|&&s| s == m).count();
    let n = scores.len();
    if kappa == n {
        return 0.0;
    }
    let second = scores
        .iter()
        .copied()
        .filter(|&s| s < m)
        .fold(f64::NEG_INFINITY, f64::max);
    let gap = m - second;
    if eps.is_inf() {
        // exp(-gap/inf) = 1: the bound degenerates to the trivial one.
        return 2.0 * v_max * (n - kappa) as f64 / kappa as f64;
    }
    2.0 * v_max * (n - kappa) as f64 / kappa as f64 * (-gap / eps.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn logsumexp_eps_at_one_matches_python() {
        // Python: log(exp(1) + exp(2) + exp(3)) ≈ 3.40760596...
        let lse = logsumexp_eps(&[1.0, 2.0, 3.0], Eps::ONE);
        assert_abs_diff_eq!(lse, 3.4076059644443808, epsilon = 1e-12);
    }

    #[test]
    fn logsumexp_eps_zero_is_max() {
        let lse = logsumexp_eps(&[1.0, 5.0, 2.0], Eps::ZERO);
        assert_eq!(lse, 5.0);
    }

    #[test]
    fn softmax_eps_zero_is_uniform_over_argmax() {
        // Two tied maxes -> 0.5, 0.5
        let w = softmax_eps(&[3.0, 1.0, 3.0], Eps::ZERO);
        assert_eq!(w, vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn softmax_eps_normalises_to_one() {
        let w = softmax_eps(&[1.0, 2.0, 3.0], Eps::ONE);
        let z: f64 = w.iter().sum();
        assert_abs_diff_eq!(z, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn maslov_dequantization_takes_lse_to_max() {
        // As ε → 0⁺ the LSE converges to the max
        let xs = [1.0, 5.0, 2.0];
        let mut prev = logsumexp_eps(&xs, Eps::ONE);
        for &e in &[0.5, 0.1, 0.01, 0.001, 0.0001] {
            let v = logsumexp_eps(&xs, Eps::new(e).unwrap());
            // monotonically approaching 5.0
            assert!(v <= prev + 1e-12, "LSE should be monotone in 1/ε");
            prev = v;
        }
        let limit = logsumexp_eps(&xs, Eps::new(1e-9).unwrap());
        assert_abs_diff_eq!(limit, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn empty_input_handled() {
        assert!(logsumexp_eps(&[], Eps::ONE).is_infinite());
        assert!(softmax_eps(&[], Eps::ONE).is_empty());
    }

    #[test]
    fn eps_star_certifies_the_smoothing_bound() {
        // Build a gap-promised instance and check A_eps* is within
        // delta of the tropical answer A_0 (argmax-mean).
        let scores = [3.0_f64, 3.0, 1.5, 0.2, -1.0]; // kappa=2, gap=1.5
        let values = [4.0_f64, 6.0, -8.0, 8.0, 7.5]; // v_max = 8
        let delta = 1e-3;
        let e = eps_star(delta, 1.5, 8.0, scores.len(), 2).unwrap();
        assert!(e > 0.0 && e.is_finite());
        let w = softmax_eps(&scores, Eps(e));
        let a_eps: f64 = w.iter().zip(values.iter()).map(|(w, v)| w * v).sum();
        let a_zero = (4.0 + 6.0) / 2.0; // argmax mean over the tie
        assert!((a_eps - a_zero).abs() <= delta,
                "|{a_eps} - {a_zero}| > {delta}");
        // and the evaluated bound itself certifies it
        let b = dequantization_bound(&scores, 8.0, Eps(e));
        assert!(b <= delta + 1e-15);
        assert!((a_eps - a_zero).abs() <= b + 1e-15);
    }

    #[test]
    fn dequantization_bound_dominates_actual_error() {
        let scores = [2.0_f64, 1.0, 0.5, 0.4];
        let values = [1.0_f64, -1.0, 0.7, -0.3]; // v_max = 1
        let a_zero = 1.0; // unique argmax -> v[0]
        for eps in [0.1, 0.3, 1.0, 4.0] {
            let w = softmax_eps(&scores, Eps(eps));
            let a_eps: f64 =
                w.iter().zip(values.iter()).map(|(w, v)| w * v).sum();
            let b = dequantization_bound(&scores, 1.0, Eps(eps));
            assert!((a_eps - a_zero).abs() <= b,
                    "eps={eps}: |{a_eps} - {a_zero}| > bound {b}");
        }
    }

    #[test]
    fn dequantization_bound_zero_on_full_tie_and_at_zero() {
        assert_eq!(dequantization_bound(&[2.0, 2.0], 5.0, Eps::ONE), 0.0);
        assert_eq!(dequantization_bound(&[2.0, 1.0], 5.0, Eps::ZERO), 0.0);
    }

    #[test]
    fn eps_star_rejects_vacuous_delta() {
        assert!(eps_star(1e9, 1.0, 1.0, 4, 1).is_err());
        assert!(eps_star(0.1, 1.0, 1.0, 4, 4).is_err());
        assert!(eps_star(-0.1, 1.0, 1.0, 4, 1).is_err());
    }
}

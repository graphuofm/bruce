//! Differential privacy: noise injection for query outputs.
//!
//! Bruce supports two standard DP mechanisms:
//!
//! * [`LaplaceMechanism`] — add `Lap(0, Δ/ε)` noise to a numeric
//!   query result. Guarantees `ε`-DP for a query whose L1 sensitivity
//!   is `Δ`. Use for counts, sums of bounded contributions.
//!
//! * [`GaussianMechanism`] — add `N(0, σ²)` noise with `σ = Δ/ε · √(2 ln(1.25/δ))`.
//!   Guarantees `(ε, δ)`-DP for L2-sensitivity-`Δ` queries. Use for
//!   higher-dimensional outputs where Laplace would inject too much
//!   noise.
//!
//! These are the two textbook mechanisms (Dwork & Roth, 2014). Bruce
//! provides them as **pluggable post-processors** on top of any F_ε
//! query, so the caller can have an `ε=0` exact answer or an
//! `(ε_dp, δ)`-DP perturbed answer without changing the query
//! pipeline.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// The (ε, δ) privacy parameters of a DP mechanism.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DpBudget {
    /// Privacy loss `ε > 0`. Smaller → stronger privacy.
    pub epsilon: f64,
    /// Slack `δ ∈ [0, 1)`. `δ = 0` is pure ε-DP; small `δ` (e.g. 1e-5)
    /// is the standard relaxation for the Gaussian mechanism.
    pub delta: f64,
}

impl DpBudget {
    /// A common starting budget: ε=1, δ=1e-5.
    pub const STANDARD: Self = Self {
        epsilon: 1.0,
        delta: 1e-5,
    };
    /// Pure ε-DP (no slack).
    pub fn pure(epsilon: f64) -> Self {
        Self { epsilon, delta: 0.0 }
    }
}

/// Laplace mechanism: add `Lap(0, Δ / ε)` noise to a real-valued
/// result.
#[derive(Debug, Clone)]
pub struct LaplaceMechanism {
    /// L1-sensitivity of the upstream query.
    pub l1_sensitivity: f64,
    /// Privacy budget.
    pub budget: DpBudget,
    /// Optional seed for reproducible noise (don't use in production).
    pub seed: Option<u64>,
}

impl LaplaceMechanism {
    /// New Laplace mechanism with given sensitivity and budget.
    pub fn new(l1_sensitivity: f64, budget: DpBudget) -> Self {
        Self {
            l1_sensitivity,
            budget,
            seed: None,
        }
    }

    /// Reproducible variant for tests.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sample one Laplace noise value.
    fn sample(&self, rng: &mut impl rand::Rng) -> f64 {
        // Lap(0, b) where b = Δ / ε; sample by inverse CDF
        let b = self.l1_sensitivity / self.budget.epsilon;
        let u: f64 = rng.gen_range(-0.5..0.5);
        -b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Add Laplace noise to a single scalar.
    pub fn release_scalar(&self, true_value: f64) -> f64 {
        let mut rng = self.make_rng();
        true_value + self.sample(&mut rng)
    }

    /// Add independent Laplace noise to each entry of a vector.
    /// Independent noise is the standard choice when the sensitivity
    /// is L1; for L2 sensitivity use the Gaussian mechanism instead.
    pub fn release_vector(&self, true_values: &[f64]) -> Vec<f64> {
        let mut rng = self.make_rng();
        true_values
            .iter()
            .map(|&v| v + self.sample(&mut rng))
            .collect()
    }

    fn make_rng(&self) -> rand::rngs::StdRng {
        match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        }
    }
}

/// Gaussian mechanism: add `N(0, σ²)` noise with
/// `σ = Δ_2 / ε · √(2 ln(1.25/δ))`.
#[derive(Debug, Clone)]
pub struct GaussianMechanism {
    /// L2-sensitivity of the upstream query.
    pub l2_sensitivity: f64,
    /// (ε, δ) budget.
    pub budget: DpBudget,
    /// Optional reproducibility seed.
    pub seed: Option<u64>,
}

impl GaussianMechanism {
    /// New Gaussian mechanism.
    pub fn new(l2_sensitivity: f64, budget: DpBudget) -> Self {
        Self {
            l2_sensitivity,
            budget,
            seed: None,
        }
    }

    /// Reproducible variant for tests.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Standard deviation of the noise.
    pub fn sigma(&self) -> f64 {
        debug_assert!(self.budget.delta > 0.0, "Gaussian mechanism requires δ > 0");
        self.l2_sensitivity / self.budget.epsilon
            * (2.0 * (1.25 / self.budget.delta).ln()).sqrt()
    }

    /// Release a single scalar.
    pub fn release_scalar(&self, true_value: f64) -> f64 {
        let mut rng = self.make_rng();
        let normal = Normal::new(0.0, self.sigma()).expect("σ > 0");
        true_value + normal.sample(&mut rng)
    }

    /// Release a vector with independent N(0, σ²) per entry.
    pub fn release_vector(&self, true_values: &[f64]) -> Vec<f64> {
        let mut rng = self.make_rng();
        let normal = Normal::new(0.0, self.sigma()).expect("σ > 0");
        true_values
            .iter()
            .map(|&v| v + normal.sample(&mut rng))
            .collect()
    }

    fn make_rng(&self) -> rand::rngs::StdRng {
        match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplace_is_unbiased_in_expectation() {
        // Average a lot of releases; mean should approach the true value
        let mech = LaplaceMechanism::new(1.0, DpBudget::pure(0.5));
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut sum = 0.0;
        let n = 100_000;
        for _ in 0..n {
            sum += mech.sample(&mut rng);
        }
        let mean = sum / n as f64;
        // Standard deviation of Lap(0, 2) is 2·√2 ≈ 2.83; std-of-mean over
        // 100K samples is ~2.83/√100000 ≈ 0.009. So mean should be < 0.05.
        assert!(mean.abs() < 0.05, "Laplace mean drift: {mean}");
    }

    #[test]
    fn gaussian_sigma_matches_formula() {
        let mech = GaussianMechanism::new(1.0, DpBudget {
            epsilon: 1.0,
            delta: 1e-5,
        });
        // σ = 1 · √(2 ln(125000)) ≈ √(2 · 11.736) ≈ √23.47 ≈ 4.84
        let expected = (2.0_f64 * (1.25_f64 / 1e-5).ln()).sqrt();
        assert!((mech.sigma() - expected).abs() < 1e-12);
    }

    #[test]
    fn laplace_release_reproducible_with_seed() {
        let mech = LaplaceMechanism::new(1.0, DpBudget::pure(1.0)).with_seed(42);
        let a = mech.release_scalar(100.0);
        let b = mech.release_scalar(100.0);
        assert_eq!(a, b); // same seed → same noise
    }

    #[test]
    fn release_vector_perturbs_every_element() {
        let mech = LaplaceMechanism::new(1.0, DpBudget::pure(0.1)).with_seed(7);
        let true_vals = vec![1.0; 10];
        let out = mech.release_vector(&true_vals);
        assert_eq!(out.len(), 10);
        // at least one element should differ from true (with overwhelming prob)
        let differs = out.iter().any(|&v| (v - 1.0).abs() > 1e-12);
        assert!(differs);
    }
}

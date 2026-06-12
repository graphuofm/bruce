//! Public types: temperature, similarity function, aggregator, score.

use serde::{Deserialize, Serialize};

/// The temperature ε of the F_ε operator.
///
/// `ε = 0` gives tropical / max-plus semantics (exact equi-join with
/// aggregation). `ε = 1` gives the standard softmax / log-sum-exp.
/// Intermediate values give a continuous interpolation that SQL cannot
/// express but that Bruce can. `ε → ∞` is the uniform mean.
///
/// We carry a positive-real internally and use a sentinel for the
/// limits, so `Eps::ZERO` and `Eps::INF` short-circuit to specialised
/// fast paths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Eps(pub f64);

impl Eps {
    /// The tropical limit, ε → 0⁺. Triggers indicator-based SQL semantics.
    pub const ZERO: Self = Eps(0.0);
    /// The standard softmax temperature.
    pub const ONE: Self = Eps(1.0);
    /// A common cool temperature.
    pub const QUARTER: Self = Eps(0.25);
    /// The uniform-mean limit, ε → ∞: every weight equals 1.
    pub const INF: Self = Eps(f64::INFINITY);

    /// Build a checked temperature: nonnegative, not NaN. `0.0` is the
    /// tropical limit and `f64::INFINITY` the uniform-mean limit; both
    /// are legal sentinels with specialised fast paths.
    pub fn new(value: f64) -> Result<Self, crate::BruceError> {
        if value < 0.0 || value.is_nan() {
            return Err(crate::BruceError::InvalidEpsilon(value));
        }
        Ok(Self(value))
    }

    /// Is this exactly the tropical limit?
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// Is this the uniform-mean limit, ε = ∞?
    #[inline]
    pub fn is_inf(self) -> bool {
        self.0.is_infinite()
    }
}

/// Similarity function between a query and a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sim {
    /// Standard inner product `⟨x, k⟩` — used by softmax attention.
    Dot,
    /// Negative squared distance `−‖x − k‖² / 2` — used by RBF attention.
    NegSquared,
    /// Indicator of exact equality `[x = k]` — used by ε = 0 SQL equi-join.
    Indicator,
}

/// Aggregation flavour returned by the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregator {
    /// Normalised attention output: `Σ wᵢ vᵢ / Σ wᵢ`.
    Softmax,
    /// Unnormalised SQL-style sum: `Σ wᵢ vᵢ`.
    Sum,
    /// Count of records with positive weight: `Σ 𝟙[wᵢ > 0]`.
    Count,
    /// Mean over surviving records (post-filter average).
    Mean,
}

/// A raw score `⟨x, kⱼ⟩ / ε`, before the exp / max.
pub type Score = f64;

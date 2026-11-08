//! The max-shifted soft-average monoid (absorb / merge / finalize).

/// Scalar-value (`d_v = 1`) accumulator in the max-shifted
/// representation. Copied verbatim from bruce_tool/bruce-pg 0.1.2
/// (`ScalarAcc`), which mirrors bruce-core's `RowAcc` and is pinned
/// against it there; the conformance vectors in `tests/` pin this copy.
///
/// Invariant for finite `eps > 0` after absorbing a set S:
/// `u = e^{-mu/eps} sum_{r in S} e^{s_r/eps} v_r`,
/// `z = e^{-mu/eps} sum_{r in S} e^{s_r/eps}`, `mu = max_{r in S} s_r`.
/// For `eps = 0`: `z` = argmax multiplicity, `u` = value-sum over the
/// argmax set. For `eps = inf`: `z` = count, `u` = plain value-sum.
/// Constraint: one accumulator sees exactly one `eps`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarAcc {
    pub mu: f64,
    pub z: f64,
    pub u: f64,
}

impl ScalarAcc {
    pub fn new() -> Self {
        Self {
            mu: f64::NEG_INFINITY,
            z: 0.0,
            u: 0.0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.z == 0.0
    }

    /// Absorb one `(score, value)` record. Mirrors `RowAcc::absorb`,
    /// including its score-infinity policy:
    /// * `-inf` weighs 0 at `eps = 0` and finite `eps` — skipped
    ///   outright (the Indicator "no match" encoding), so an
    ///   all-`-inf` group stays empty and finalizes to SQL NULL.
    /// * `+inf` at finite `eps` dominates every finite score: the
    ///   accumulator collapses to argmax semantics over the `+inf`
    ///   rows, with uniform tie handling exactly as at `eps = 0`.
    ///   `exp(inf - inf)` is never evaluated.
    /// * At `eps = inf` the score is not consulted (plain mean), so
    ///   `±inf` rows count like any other row.
    ///
    /// NaN never reaches `absorb`: `bruce_softavg_sfunc` diverts a NaN
    /// score or value into the state's sticky NaN bit instead (see the
    /// module docs and README.md), which is precisely why this factor
    /// of the state stays bit-identical to bruce-core's.
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if eps == 0.0 {
            if s == f64::NEG_INFINITY {
                // Indicator "no match": weight 0
                return;
            }
            // tropical: keep only the argmax set
            if s > self.mu {
                self.mu = s;
                self.z = 1.0;
                self.u = v;
            } else if s == self.mu {
                self.z += 1.0;
                self.u += v;
            }
            return;
        }
        if eps.is_infinite() {
            // uniform: plain count + sum
            self.z += 1.0;
            self.u += v;
            return;
        }
        // finite eps > 0
        if s == f64::NEG_INFINITY {
            // exp(-inf / eps) = 0: contributes nothing
            return;
        }
        if s == f64::INFINITY || self.mu == f64::INFINITY {
            // argmax collapse: only +inf-scored rows retain weight
            if s == f64::INFINITY && self.mu == f64::INFINITY {
                // tie among +inf rows: uniform, as at eps = 0
                self.z += 1.0;
                self.u += v;
            } else if s == f64::INFINITY {
                // first +inf row dominates the finite prefix
                self.mu = f64::INFINITY;
                self.z = 1.0;
                self.u = v;
            }
            // else: finite s under a +inf anchor: weight exp(-inf) = 0
            return;
        }
        if self.is_empty() {
            self.mu = s;
            self.z = 1.0;
            self.u = v;
            return;
        }
        // One exp per row. Bit-identical to the two-exp form
        // (`scale`, `w` with mu2 = max(mu, s)) because exp(0) == 1 exactly
        // and x * 1.0 == x, 1.0 * v == v in IEEE 754.
        if s <= self.mu {
            let w = ((s - self.mu) / eps).exp();
            self.u += w * v;
            self.z += w;
        } else {
            let scale = ((self.mu - s) / eps).exp();
            self.u = self.u * scale + v;
            self.z = self.z * scale + 1.0;
            self.mu = s;
        }
    }

    /// Merge another accumulator — the partition-reduce identity:
    /// disjoint row sets combine by re-basing both sides to the common
    /// maximum and adding. Mirrors `RowAcc::merge`.
    pub fn merge(&mut self, other: &ScalarAcc, eps: f64) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = *other;
            return;
        }
        if eps == 0.0 {
            if other.mu > self.mu {
                *self = *other;
            } else if other.mu == self.mu {
                self.z += other.z;
                self.u += other.u;
            }
            return;
        }
        if eps.is_infinite() {
            self.z += other.z;
            self.u += other.u;
            return;
        }
        // finite eps with a +inf anchor on either side (see `absorb`'s
        // policy): the +inf side(s) dominate; `exp(inf - inf)` must
        // never be evaluated. Mirrors `RowAcc::merge`.
        if self.mu == f64::INFINITY || other.mu == f64::INFINITY {
            if self.mu == f64::INFINITY && other.mu == f64::INFINITY {
                self.z += other.z;
                self.u += other.u;
            } else if other.mu == f64::INFINITY {
                *self = *other;
            }
            // else: self is the +inf side; other's finite rows weigh 0
            return;
        }
        let mu2 = self.mu.max(other.mu);
        let s1 = ((self.mu - mu2) / eps).exp();
        let s2 = ((other.mu - mu2) / eps).exp();
        self.u = self.u * s1 + other.u * s2;
        self.z = self.z * s1 + other.z * s2;
        self.mu = mu2;
    }

    /// `u / z` in every regime; `None` if nothing was absorbed.
    /// Mirrors `RowAcc::finalize`.
    pub fn finalize(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.u / self.z)
    }
}

impl Default for ScalarAcc {
    fn default() -> Self {
        Self::new()
    }
}


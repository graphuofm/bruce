//! softagg-core — the one implementation of the soft-average state that
//! every host-engine extension in paper_icde links (PostgreSQL via pgrx,
//! DuckDB and SQLite via their C APIs from Rust). One implementation, so
//! "the same operator in three engines" is literally the same arithmetic.
//!
//! `SOFTAVG(value, score, eps)` over a row set S is
//! `sum_r e^{s_r/eps} v_r / sum_r e^{s_r/eps}`, kept max-anchored so that
//! `e^{s/eps}` is never materialised.
//!
//! Two accumulators:
//! - [`ScalarAcc`]: absorb / merge / finalize, the aggregate monoid
//!   (copied verbatim from bruce-pg 0.1.2, which is pinned to bruce-core).
//! - [`InvAcc`]: adds `remove`, the inverse needed by window functions
//!   (SQLite `xInverse`, PostgreSQL `MINVFUNC`) and incremental
//!   maintenance. Removal never re-anchors: the anchor `mu` only has to
//!   stay an upper bound of the live scores. What can go wrong is
//!   rounding: cancellation in `z` or in `u` (removing rows that held most
//!   of the weight), error accumulated over many removals, and live
//!   weights underflowing under a stale anchor at very sharp `eps`.
//!   [`InvAcc`] therefore carries a running first-order error bound on `z`
//!   and `u` (Higham, *Accuracy and Stability of Numerical Algorithms*,
//!   §3.3), and reports [`InvAcc::is_exact`] = false as soon as the bound
//!   on the delivered value `u / z` exceeds
//!   `max(ABS_TOL, REFOLD_GAIN * n * U) * v_scale`, i.e. once the value
//!   could be off by more than 1e-10 of the largest |value| seen, or by
//!   more than 100x what recomputing the rows would guarantee. The host
//!   must then recompute the frame/group (or raise an error); it never
//!   silently returns a value outside that bound.
//!
//! NaN is NOT handled here: each host decides at its call site
//! (PostgreSQL propagates NaN like AVG; DuckDB/SQLite map NULL to skip).

mod acc;
pub use acc::ScalarAcc;

/// Unit roundoff of f64.
pub const U: f64 = f64::EPSILON / 2.0;
/// Absolute tolerance on the delivered value, in units of the largest
/// |value| absorbed (`v_scale`).
pub const ABS_TOL: f64 = 1e-10;
/// A state is kept while its error bound is within this factor of what a
/// fresh fold of its `n` rows would guarantee (`n * U * v_scale`).
pub const REFOLD_GAIN: f64 = 100.0;
/// Absolute error of a weight or product that underflows (2^-1074).
const TINY: f64 = 4.9406564584124654e-324;

/// Underflow terms for one row's weight `w` and product `w v` (B1).
#[inline]
fn tiny_terms(w: f64, v: f64) -> (f64, f64) {
    let tz = if w < f64::MIN_POSITIVE { TINY } else { 0.0 };
    let wv = (w * v).abs();
    let tu = tz * v.abs() + if wv < f64::MIN_POSITIVE { TINY } else { 0.0 };
    (tz, tu)
}

/// Inverse-capable accumulator: [`ScalarAcc`] plus the live-row count, a
/// running first-order bound on the absolute error of `z` and of `u` (in
/// the current anchor's scale), the largest |value| absorbed, and a sticky
/// inexact flag. 64 bytes, `#[repr(C)]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvAcc {
    pub acc: ScalarAcc,
    /// Rows absorbed and not removed that carry weight (`-inf` scores at
    /// finite or zero eps carry none and are not counted).
    pub n: u64,
    /// Bound on |computed z - exact z|.
    pub err_z: f64,
    /// Bound on |computed u - exact u|.
    pub err_u: f64,
    /// Largest |value| absorbed since the last fresh state.
    pub v_scale: f64,
    /// Sticky: the value can no longer be trusted within the bound.
    /// Cleared only by [`InvAcc::reset`] or by removing every row.
    pub inexact: bool,
}

impl Default for InvAcc {
    fn default() -> Self {
        Self::new()
    }
}

impl InvAcc {
    pub fn new() -> Self {
        Self {
            acc: ScalarAcc::new(),
            n: 0,
            err_z: 0.0,
            err_u: 0.0,
            v_scale: 0.0,
            inexact: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    #[inline]
    fn carries_weight(s: f64, eps: f64) -> bool {
        eps.is_infinite() || s != f64::NEG_INFINITY
    }

    /// Bound on |delivered u/z - exact u/z| for the current state.
    pub fn value_error_bound(&self) -> f64 {
        let z = self.acc.z;
        if self.n == 0 {
            return 0.0;
        }
        if z.partial_cmp(&self.err_z) != Some(std::cmp::Ordering::Greater) {
            return f64::INFINITY;
        }
        let f = (self.acc.u / z).abs();
        (self.err_u + f * self.err_z) / (z - self.err_z) + U * f
    }

    #[inline]
    fn check(&mut self) {
        if self.inexact || self.n == 0 {
            return;
        }
        let tol = ABS_TOL.max(REFOLD_GAIN * self.n as f64 * U) * self.v_scale.max(f64::MIN_POSITIVE);
        // a NaN bound counts as exceeding the tolerance
        let within = matches!(
            self.value_error_bound().partial_cmp(&tol),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        );
        if !within {
            self.inexact = true;
        }
    }

    /// Absorb one row (NaN-free; see the module docs).
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if !Self::carries_weight(s, eps) {
            return;
        }
        self.v_scale = self.v_scale.max(v.abs());
        let (mu0, z0, u0) = (self.acc.mu, self.acc.z, self.acc.u);
        self.acc.absorb(s, v, eps);
        self.n += 1;
        let (z1, u1) = (self.acc.z, self.acc.u);
        if eps > 0.0 && eps.is_finite() && mu0.is_finite() && s.is_finite() && z0 != 0.0 {
            if s <= mu0 {
                // u += w v, z += w with w = exp(x): exp 1 ulp, mul, add;
                // rounding of the argument x = (s - mu)/eps perturbs w by
                // up to 2U|x| relatively; underflowed w or w v by 2^-1074
                let x = (s - mu0) / eps;
                let w = x.exp();
                let (tz, tu) = tiny_terms(w, v);
                self.err_z += 2.0 * U * w + U * z1.abs() + 2.0 * U * x.abs() * w + tz;
                self.err_u += 3.0 * U * (w * v).abs() + U * u1.abs()
                    + 2.0 * U * x.abs() * (w * v).abs() + tu;
            } else {
                // anchor rose: z = z0*scale + 1, u = u0*scale + v
                let xs = (mu0 - s) / eps;
                let scale = xs.exp();
                let t = if scale < f64::MIN_POSITIVE { TINY } else { 0.0 };
                self.err_z = self.err_z * scale + 3.0 * U * z0.abs() * scale + U * z1.abs()
                    + 2.0 * U * xs.abs() * z0.abs() * scale + t * z0.abs();
                self.err_u = self.err_u * scale + 3.0 * U * u0.abs() * scale + U * u1.abs()
                    + 2.0 * U * xs.abs() * u0.abs() * scale + t * u0.abs();
            }
        } else if eps.is_infinite() || eps == 0.0 {
            // counts are exact integers; the value sum rounds once
            self.err_u += U * u1.abs();
        }
        // first row, or a +-inf anchor regime: the state is exact as built
        self.check();
    }

    /// Remove one row previously absorbed with the same `(s, v)`.
    /// Never rescans; may mark the state inexact (see the module docs).
    #[inline]
    pub fn remove(&mut self, s: f64, v: f64, eps: f64) {
        if !Self::carries_weight(s, eps) {
            return;
        }
        if self.n == 0 {
            // removing from an empty state: the caller's bookkeeping is
            // wrong; refuse to invent a value
            self.inexact = true;
            return;
        }
        self.n -= 1;
        if self.n == 0 {
            // exact by construction: nothing is left
            *self = Self::new();
            return;
        }
        let a = &mut self.acc;
        if eps.is_infinite() {
            a.z -= 1.0;
            a.u -= v;
            self.err_u += U * a.u.abs();
            self.check();
            return;
        }
        if eps == 0.0 || a.mu == f64::INFINITY || s == f64::INFINITY {
            // argmax regimes: z counts the argmax set
            if s == a.mu {
                a.z -= 1.0;
                a.u -= v;
                self.err_u += U * a.u.abs();
                if a.z <= 0.0 {
                    // the argmax set emptied but rows remain: the next
                    // anchor is unknown without the data
                    self.inexact = true;
                }
            } else if s > a.mu {
                self.inexact = true;
            }
            // s < mu: the row never carried weight in this regime
            self.check();
            return;
        }
        // finite eps, finite anchor
        if s > a.mu {
            // cannot have been absorbed under this anchor
            self.inexact = true;
            return;
        }
        let x = (s - a.mu) / eps;
        let w = x.exp();
        a.u -= w * v;
        a.z -= w;
        let (tz, tu) = tiny_terms(w, v);
        self.err_z += 2.0 * U * w + U * a.z.abs() + 2.0 * U * x.abs() * w + tz;
        self.err_u += 3.0 * U * (w * v).abs() + U * a.u.abs()
            + 2.0 * U * x.abs() * (w * v).abs() + tu;
        self.check();
    }

    pub fn merge(&mut self, other: &InvAcc, eps: f64) {
        if other.n == 0 {
            self.inexact |= other.inexact;
            return;
        }
        if self.n == 0 {
            let flag = self.inexact;
            *self = *other;
            self.inexact |= flag;
            return;
        }
        let (mu_a, z_a, u_a) = (self.acc.mu, self.acc.z, self.acc.u);
        let (mu_b, z_b, u_b) = (other.acc.mu, other.acc.z, other.acc.u);
        self.acc.merge(&other.acc, eps);
        let (z1, u1) = (self.acc.z, self.acc.u);
        if eps > 0.0 && eps.is_finite() && mu_a.is_finite() && mu_b.is_finite() {
            let m = mu_a.max(mu_b);
            let (x1, x2) = ((mu_a - m) / eps, (mu_b - m) / eps);
            let (s1, s2) = (x1.exp(), x2.exp());
            let t1 = if s1 < f64::MIN_POSITIVE { TINY } else { 0.0 };
            let t2 = if s2 < f64::MIN_POSITIVE { TINY } else { 0.0 };
            self.err_z = self.err_z * s1 + other.err_z * s2
                + 3.0 * U * (z_a * s1 + z_b * s2).abs() + U * z1.abs()
                + 2.0 * U * (x1.abs() * z_a.abs() * s1 + x2.abs() * z_b.abs() * s2)
                + t1 * z_a.abs() + t2 * z_b.abs();
            self.err_u = self.err_u * s1 + other.err_u * s2
                + 3.0 * U * (u_a.abs() * s1 + u_b.abs() * s2) + U * u1.abs()
                + 2.0 * U * (x1.abs() * u_a.abs() * s1 + x2.abs() * u_b.abs() * s2)
                + t1 * u_a.abs() + t2 * u_b.abs();
        } else {
            // argmax / uniform regimes: sums of the two sides
            self.err_z += other.err_z;
            self.err_u += other.err_u + U * u1.abs();
        }
        self.n += other.n;
        self.v_scale = self.v_scale.max(other.v_scale);
        self.inexact |= other.inexact;
        self.check();
    }

    /// `true` while the delivered value is within the documented bound.
    #[inline]
    pub fn is_exact(&self) -> bool {
        !self.inexact
    }

    /// `u / z`, or `None` when empty. Check [`InvAcc::is_exact`] first.
    pub fn finalize(&self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        self.acc.finalize()
    }
}

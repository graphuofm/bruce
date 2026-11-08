//! The DuckDB aggregate state: softagg-core's `ScalarAcc` plus the two
//! host-level facts DuckDB needs, and nothing else.
//!
//! All soft-average arithmetic (absorb / merge / finalize, the +/-Inf
//! policy, eps = 0 and eps = inf) is `softagg_core::ScalarAcc`, linked from
//! `../core`; this file holds only what the core leaves to the host:
//!
//! * **eps tag.** `combine` receives two states and no arguments, so the
//!   temperature must ride inside the state. `eps` is NaN while the state is
//!   unbound (no row absorbed yet), then the group's eps. A second, different
//!   eps for the same group is an error (`EpsMismatch`).
//! * **NaN policy** (the core does not handle NaN, by design). DuckDB's AVG
//!   returns NaN if any input is NaN, so a NaN value or score sets a sticky
//!   bit and stays out of the core accumulator (a NaN anchor would poison the
//!   max comparisons); finalize then returns NaN.
//! * **NULL policy** lives in `lib.rs::update`: a row whose value or score is
//!   NULL is skipped, like AVG; a live row with NULL eps is an error.
//!
//! Layout: `#[repr(C)]`, 40 bytes, plain data, no destructor. DuckDB
//! allocates `state_size()` bytes per group inside its hash table (or per
//! segment-tree node for window frames) and calls `init` on them.

use softagg_core::ScalarAcc;
use std::fmt;

/// Errors surfaced to DuckDB through `duckdb_aggregate_function_set_error`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoftAvgError {
    InvalidEps(f64),
    EpsMismatch(f64, f64),
    NullEps,
}

impl fmt::Display for SoftAvgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SoftAvgError::InvalidEps(e) => write!(
                f,
                "softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got {e}"
            ),
            SoftAvgError::EpsMismatch(a, b) => write!(
                f,
                "softavg: eps must be constant within an aggregate group ({a} vs {b})"
            ),
            SoftAvgError::NullEps => write!(f, "softavg: eps must not be NULL"),
        }
    }
}

/// `eps >= 0` (NaN fails both comparisons).
#[inline]
pub fn check_eps(eps: f64) -> Result<(), SoftAvgError> {
    if eps >= 0.0 {
        Ok(())
    } else {
        Err(SoftAvgError::InvalidEps(eps))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct SoftAvgState {
    /// softagg-core accumulator `(mu, z, u)`.
    pub acc: ScalarAcc,
    /// The group's eps; NaN while unbound.
    pub eps: f64,
    /// A qualifying row carried a NaN value or score (sticky).
    pub nan_seen: bool,
}

impl SoftAvgState {
    #[inline]
    pub fn new() -> Self {
        Self {
            acc: ScalarAcc::new(),
            eps: f64::NAN,
            nan_seen: false,
        }
    }

    /// Bind (or check) the group's eps. `eps` must already be validated
    /// with [`check_eps`]. One bit comparison on the hot path.
    #[inline(always)]
    pub fn bind_eps(&mut self, eps: f64) -> Result<(), SoftAvgError> {
        if self.eps.to_bits() == eps.to_bits() {
            return Ok(());
        }
        self.bind_eps_slow(eps)
    }

    #[cold]
    #[inline(never)]
    fn bind_eps_slow(&mut self, eps: f64) -> Result<(), SoftAvgError> {
        if self.eps.is_nan() {
            self.eps = eps;
            Ok(())
        } else if self.eps == eps {
            // +0.0 vs -0.0: same temperature
            Ok(())
        } else {
            Err(SoftAvgError::EpsMismatch(self.eps, eps))
        }
    }

    /// Absorb one non-NULL row whose eps has been validated and bound.
    #[inline(always)]
    pub fn absorb_bound(&mut self, s: f64, v: f64, eps: f64) {
        if s.is_nan() || v.is_nan() {
            self.nan_seen = true;
            return;
        }
        self.acc.absorb(s, v, eps);
    }

    /// Absorb one non-NULL row (validates and binds eps).
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) -> Result<(), SoftAvgError> {
        check_eps(eps)?;
        self.bind_eps(eps)?;
        self.absorb_bound(s, v, eps);
        Ok(())
    }

    /// Componentwise merge: core merge, OR on the sticky bit; eps tags must
    /// agree (an unbound side is an empty partial and adopts the other tag).
    #[inline]
    pub fn merge(&mut self, other: &SoftAvgState) -> Result<(), SoftAvgError> {
        if other.eps.is_nan() {
            // unbound partial: empty accumulator, no NaN seen
            return Ok(());
        }
        self.bind_eps(other.eps)?;
        self.nan_seen |= other.nan_seen;
        self.acc.merge(&other.acc, self.eps);
        Ok(())
    }

    /// NaN if the sticky bit is set; otherwise `u / z`; `None` (SQL NULL)
    /// when the accumulator is empty (no qualifying rows, or only -Inf
    /// scores at finite eps / eps = 0).
    #[inline]
    pub fn finalize(&self) -> Option<f64> {
        if self.nan_seen {
            return Some(f64::NAN);
        }
        self.acc.finalize()
    }
}

impl Default for SoftAvgState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_is_plain_40_bytes() {
        assert_eq!(std::mem::size_of::<ScalarAcc>(), 24);
        assert_eq!(std::mem::size_of::<SoftAvgState>(), 40);
        assert_eq!(std::mem::align_of::<SoftAvgState>(), 8);
    }

    #[test]
    fn nan_is_sticky_and_eps_is_checked() {
        let mut st = SoftAvgState::new();
        st.absorb(0.1, 1.0, 0.5).unwrap();
        st.absorb(f64::NAN, 1.0, 0.5).unwrap();
        assert!(st.finalize().unwrap().is_nan());
        let mut st = SoftAvgState::new();
        assert_eq!(st.absorb(0.0, 1.0, -1.0), Err(SoftAvgError::InvalidEps(-1.0)));
        assert!(matches!(st.absorb(0.0, 1.0, f64::NAN), Err(SoftAvgError::InvalidEps(_))));
        st.absorb(0.0, 1.0, 0.5).unwrap();
        assert_eq!(st.absorb(0.0, 1.0, 0.25), Err(SoftAvgError::EpsMismatch(0.5, 0.25)));
        let mut other = SoftAvgState::new();
        other.absorb(0.0, 1.0, 0.25).unwrap();
        assert_eq!(st.merge(&other), Err(SoftAvgError::EpsMismatch(0.5, 0.25)));
        // an unbound partial merges into anything, and adopts a bound one
        let mut st = SoftAvgState::new();
        st.absorb(0.0, 4.0, 0.5).unwrap();
        st.merge(&SoftAvgState::new()).unwrap();
        assert_eq!(st.finalize(), Some(4.0));
        let mut empty = SoftAvgState::new();
        empty.merge(&st).unwrap();
        assert_eq!(empty.finalize(), Some(4.0));
        // +0.0 and -0.0 are the same temperature
        let mut z = SoftAvgState::new();
        z.absorb(0.0, 1.0, 0.0).unwrap();
        z.absorb(0.0, 3.0, -0.0).unwrap();
        assert_eq!(z.finalize(), Some(2.0));
    }

    #[test]
    fn nan_seen_survives_merge_of_nan_only_partial() {
        let mut a = SoftAvgState::new();
        a.absorb(0.0, 1.0, 0.5).unwrap();
        let mut b = SoftAvgState::new();
        b.absorb(0.0, f64::NAN, 0.5).unwrap();
        a.merge(&b).unwrap();
        assert!(a.finalize().unwrap().is_nan());
    }

    #[test]
    fn wrapper_equals_core_fold_bitwise() {
        // the state must not change the core's arithmetic
        let mut seed = 7u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for eps in [1e-4, 1e-3, 0.02, 0.1, 1.0, 0.0, f64::INFINITY] {
            let mut st = SoftAvgState::new();
            let mut core = ScalarAcc::new();
            for _ in 0..3000 {
                let (s, v) = (2.0 * next() - 1.0, 10.0 * next());
                st.absorb(s, v, eps).unwrap();
                core.absorb(s, v, eps);
            }
            assert_eq!(st.finalize().unwrap().to_bits(), core.finalize().unwrap().to_bits());
        }
    }
}

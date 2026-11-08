//! The per-group / per-partition state behind `softavg` in SQLite, kept
//! free of any SQLite call so it can be unit-tested directly.
//!
//! The arithmetic is softagg-core's [`InvAcc`] (absorb / remove /
//! finalize). What this file adds is what SQLite needs on top of it:
//!
//! * `eps` validation (constant per group, `>= 0`, `+inf` allowed);
//! * a **shadow** of the current window frame's `(score, value)` rows, so
//!   that when the core's running error bound marks the state inexact
//!   (the delivered value could be off by more than its tolerance, which
//!   is absolute in units of max|value|) the frame can be refolded. SQLite cannot recompute a frame for
//!   an aggregate window function, and returning a guarded value would be
//!   silently wrong.
//!
//! # Why the shadow is lazy
//!
//! SQLite uses one registration (`sqlite3_create_window_function`) for
//! both `GROUP BY` aggregation and `OVER (...)`, and gives no signal in
//! `xStep` which of the two it is running. The first `xValue` call is
//! the first moment window mode is known (plain aggregation never calls
//! `xValue`). Recording every row from the start would make the
//! `GROUP BY` state O(rows) instead of O(1), so:
//!
//! * before the first `xValue` (mode unknown) the state records at most
//!   [`PREFIX_CAP`] rows; past that it stops recording and only counts
//!   (`unknown`), so `GROUP BY` pays O(1) memory per group beyond the
//!   first `PREFIX_CAP` rows;
//! * from the first `xValue` on, every stepped row is recorded, and every
//!   `xInverse` pops the oldest one (SQLite removes frame rows in the
//!   order it added them; the pop is verified bit-for-bit and a mismatch
//!   is an error, never a guess);
//! * rows not recorded (`unknown`) are always the *oldest* rows of the
//!   frame, so they leave the frame first;
//! * when the state is inexact at `xValue`/`xFinal` time and
//!   `unknown == 0`, the frame is refolded from the shadow (O(frame)
//!   once, then O(1) per row again); if `unknown > 0` the refold is
//!   impossible and the call raises an error instead of returning a
//!   number. A state that has seen no removal since it was built is a
//!   fresh fold and is never refolded. With `ROWS BETWEEN k PRECEDING AND CURRENT ROW` the first
//!   frame holds one row, so this error cannot occur; it needs a first
//!   frame longer than [`PREFIX_CAP`] rows (`m FOLLOWING` with
//!   `m >= PREFIX_CAP`) *and* a guard trip while one of those first
//!   rows is still in the frame, or a frame longer than [`SHADOW_CAP`].

use softagg_core::{InvAcc, ScalarAcc};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Rows recorded per group before the mode is known.
pub const PREFIX_CAP: usize = 64;
/// Rows recorded per window frame (16 bytes each, 16 MiB at the cap).
pub const SHADOW_CAP: usize = 1 << 20;

/// Process-wide counters for the rare paths (never touched per row).
pub static REFOLDS: AtomicU64 = AtomicU64::new(0);
pub static REFOLD_ROWS: AtomicU64 = AtomicU64::new(0);
pub static SHADOW_SPILLS: AtomicU64 = AtomicU64::new(0);
pub static REFOLD_REFUSED: AtomicU64 = AtomicU64::new(0);
/// Values served from an absorb-only state the core had flagged (never
/// expected: a fresh fold is within the core's bound by construction).
pub static FRESH_FLAGGED: AtomicU64 = AtomicU64::new(0);

pub fn stats_json() -> String {
    format!(
        "{{\"refolds\":{},\"refold_rows\":{},\"shadow_spills\":{},\"refold_refused\":{},\"fresh_flagged\":{}}}",
        REFOLDS.load(Relaxed),
        REFOLD_ROWS.load(Relaxed),
        SHADOW_SPILLS.load(Relaxed),
        REFOLD_REFUSED.load(Relaxed),
        FRESH_FLAGGED.load(Relaxed)
    )
}

pub fn stats_reset() {
    for c in [&REFOLDS, &REFOLD_ROWS, &SHADOW_SPILLS, &REFOLD_REFUSED, &FRESH_FLAGGED] {
        c.store(0, Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    EpsNull,
    EpsInvalid,
    EpsNotConstant,
    /// `xInverse` with nothing recorded and nothing unknown.
    InverseOnEmpty,
    /// `xInverse` row is not the oldest recorded row.
    InverseNotFifo,
    /// Guard tripped while unrecorded rows are still in the frame.
    CannotRefold,
}

impl Error {
    pub fn message(self) -> &'static [u8] {
        match self {
            Error::EpsNull => b"softavg: eps must not be NULL\0",
            Error::EpsInvalid => b"softavg: eps must be >= 0 (use 9e999 for +infinity)\0",
            Error::EpsNotConstant => b"softavg: eps must be constant within a group / partition\0",
            Error::InverseOnEmpty => {
                b"softavg: internal: xInverse on an empty frame (refusing to invent a value)\0"
            }
            Error::InverseNotFifo => {
                b"softavg: internal: xInverse row is not the oldest row of the frame (refusing to guess)\0"
            }
            Error::CannotRefold => {
                b"softavg: window frame lost precision (error bound exceeded) and its first rows \
were not recorded, so it cannot be refolded exactly; use a frame whose first frame has \
at most 64 rows, or a larger eps\0"
            }
        }
    }
}

pub struct WinState {
    inv: InvAcc,
    eps: f64,
    eps_set: bool,
    /// `xValue` has been seen: this is a window, record every row.
    window: bool,
    /// `false` once the prefix overflowed in unknown mode.
    recording: bool,
    /// Oldest frame rows that are not in `shadow`.
    unknown: u64,
    /// A row was removed since the state was last built by absorbs only.
    /// Without removals the state *is* a fresh fold (bit-identical to
    /// `ScalarAcc`), so it is never refolded, whatever the core's bound says.
    removed: bool,
    shadow: VecDeque<(f64, f64)>,
}

impl Default for WinState {
    fn default() -> Self {
        Self::new()
    }
}

impl WinState {
    pub fn new() -> Self {
        Self {
            inv: InvAcc::new(),
            eps: 0.0,
            eps_set: false,
            window: false,
            recording: true,
            unknown: 0,
            removed: false,
            shadow: VecDeque::new(),
        }
    }

    /// Validate `eps` for a live row (`None` = SQL NULL).
    #[inline]
    pub fn check_eps(&mut self, eps: Option<f64>) -> Result<f64, Error> {
        let e = eps.ok_or(Error::EpsNull)?;
        if self.eps_set {
            if e != self.eps {
                return Err(Error::EpsNotConstant);
            }
            return Ok(e);
        }
        if !(e >= 0.0) {
            return Err(Error::EpsInvalid);
        }
        self.eps = e;
        self.eps_set = true;
        Ok(e)
    }

    /// One live row (`s`, `v` non-NULL, non-NaN), after [`check_eps`]
    /// accepted this row's eps (the state keeps the one eps it uses).
    #[inline]
    pub fn step(&mut self, s: f64, v: f64) {
        debug_assert!(self.eps_set);
        self.inv.absorb(s, v, self.eps);
        if !self.recording {
            self.unknown += 1;
            return;
        }
        let cap = if self.window { SHADOW_CAP } else { PREFIX_CAP };
        if self.shadow.len() >= cap {
            // all recorded rows become unknown (they are the oldest)
            self.unknown += self.shadow.len() as u64;
            self.shadow.clear();
            if self.window {
                SHADOW_SPILLS.fetch_add(1, Relaxed);
            } else {
                // mode unknown and the group is large: almost surely a
                // GROUP BY; stop paying for the record
                self.shadow.shrink_to_fit();
                self.recording = false;
                self.unknown += 1;
                return;
            }
        }
        self.shadow.push_back((s, v));
    }

    /// Remove the oldest live row of the frame.
    #[inline]
    pub fn inverse(&mut self, s: f64, v: f64) -> Result<(), Error> {
        if self.unknown > 0 {
            self.unknown -= 1;
        } else {
            match self.shadow.pop_front() {
                Some((s0, v0)) => {
                    if s0.to_bits() != s.to_bits() || v0.to_bits() != v.to_bits() {
                        return Err(Error::InverseNotFifo);
                    }
                }
                None => return Err(Error::InverseOnEmpty),
            }
        }
        self.inv.remove(s, v, self.eps);
        self.removed = true;
        Ok(())
    }

    /// Current value (`xValue`); marks the state as a window.
    #[inline]
    pub fn value(&mut self) -> Result<Option<f64>, Error> {
        if !self.window {
            self.window = true;
            self.recording = true;
        }
        self.current()
    }

    /// Final value (`xFinal`): exact in aggregate mode by construction.
    pub fn finalize(&mut self) -> Result<Option<f64>, Error> {
        self.current()
    }

    #[inline]
    fn current(&mut self) -> Result<Option<f64>, Error> {
        if !self.inv.is_exact() {
            if self.removed {
                self.refold()?;
            } else {
                // absorbs only: this is a fresh fold already
                FRESH_FLAGGED.fetch_add(1, Relaxed);
            }
        }
        Ok(self.inv.finalize())
    }

    #[cold]
    fn refold(&mut self) -> Result<(), Error> {
        if self.unknown > 0 {
            REFOLD_REFUSED.fetch_add(1, Relaxed);
            return Err(Error::CannotRefold);
        }
        let mut a = InvAcc::new();
        for &(s, v) in &self.shadow {
            a.absorb(s, v, self.eps);
        }
        self.inv = a;
        self.removed = false;
        REFOLDS.fetch_add(1, Relaxed);
        REFOLD_ROWS.fetch_add(self.shadow.len() as u64, Relaxed);
        Ok(())
    }

    pub fn is_exact(&self) -> bool {
        self.inv.is_exact()
    }
    pub fn shadow_len(&self) -> usize {
        self.shadow.len()
    }
    pub fn unknown(&self) -> u64 {
        self.unknown
    }
}

/// State of `softavg_agg`: aggregate-only (no `xInverse`), so it is the
/// core's plain monoid `ScalarAcc` with no row count, error bound or
/// prefix record. Plain data; valid once `init` is set.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AggState {
    pub acc: ScalarAcc,
    pub eps: f64,
    pub eps_set: bool,
}

impl Default for AggState {
    fn default() -> Self {
        Self::new()
    }
}

impl AggState {
    pub fn new() -> Self {
        Self { acc: ScalarAcc::new(), eps: 0.0, eps_set: false }
    }

    #[inline]
    pub fn step(&mut self, s: f64, v: f64, eps: Option<f64>) -> Result<(), Error> {
        let e = eps.ok_or(Error::EpsNull)?;
        if self.eps_set {
            if e != self.eps {
                return Err(Error::EpsNotConstant);
            }
        } else {
            if !(e >= 0.0) {
                return Err(Error::EpsInvalid);
            }
            self.eps = e;
            self.eps_set = true;
        }
        self.acc.absorb(s, v, e);
        Ok(())
    }

    pub fn finalize(&self) -> Option<f64> {
        self.acc.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn fold(rows: &[(f64, f64)], eps: f64) -> Option<f64> {
        let mut a = ScalarAcc::new();
        for &(s, v) in rows {
            a.absorb(s, v, eps);
        }
        a.finalize()
    }

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * a.abs().max(b.abs()).max(1e-300)
    }

    /// Drive the state exactly like SQLite does for
    /// `ROWS BETWEEN p PRECEDING AND f FOLLOWING`: step rows up to the
    /// first frame end, then per output row: value, inverse, step.
    fn sliding(rows: &[(f64, f64)], eps: f64, p: usize, f: usize) -> Vec<Result<Option<f64>, Error>> {
        let mut st = WinState::new();
        let n = rows.len();
        let mut stepped = 0;
        let mut removed = 0;
        let mut out = vec![];
        for i in 0..n {
            let hi = (i + f).min(n - 1);
            while stepped <= hi {
                st.check_eps(Some(eps)).unwrap();
                st.step(rows[stepped].0, rows[stepped].1);
                stepped += 1;
            }
            let lo = i.saturating_sub(p);
            let mut bad = None;
            while removed < lo {
                if let Err(e) = st.check_eps(Some(eps)).and_then(|_| st.inverse(rows[removed].0, rows[removed].1)) {
                    bad = Some(e);
                }
                removed += 1;
            }
            out.push(match bad {
                Some(e) => Err(e),
                None => st.value(),
            });
        }
        out
    }

    #[test]
    fn eps_validation() {
        let mut st = WinState::new();
        assert_eq!(st.check_eps(None), Err(Error::EpsNull));
        assert_eq!(st.check_eps(Some(-1.0)), Err(Error::EpsInvalid));
        assert_eq!(st.check_eps(Some(f64::NAN)), Err(Error::EpsInvalid));
        assert_eq!(st.check_eps(Some(0.1)), Ok(0.1));
        assert_eq!(st.check_eps(Some(0.2)), Err(Error::EpsNotConstant));
        let mut st = WinState::new();
        assert_eq!(st.check_eps(Some(f64::INFINITY)), Ok(f64::INFINITY));
        assert_eq!(st.check_eps(Some(f64::INFINITY)), Ok(f64::INFINITY));
    }

    #[test]
    fn aggregate_mode_is_o1_beyond_prefix() {
        let mut st = WinState::new();
        let mut r = Lcg(1);
        let rows: Vec<(f64, f64)> = (0..10_000).map(|_| (2.0 * r.next() - 1.0, 10.0 * r.next())).collect();
        for &(s, v) in &rows {
            st.check_eps(Some(0.02)).unwrap();
            st.step(s, v);
        }
        assert_eq!(st.shadow_len(), 0);
        assert_eq!(st.unknown(), 10_000);
        let got = st.finalize().unwrap().unwrap();
        assert!(close(got, fold(&rows, 0.02).unwrap(), 1e-12));
    }

    #[test]
    fn sliding_matches_fresh_fold_at_sharp_eps() {
        // sharp eps: the guard trips often; every value must still match
        stats_reset();
        let mut r = Lcg(7);
        let rows: Vec<(f64, f64)> = (0..3000).map(|_| (2.0 * r.next() - 1.0, 10.0 * r.next())).collect();
        for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0, 0.0, f64::INFINITY] {
            for &(p, f) in &[(49usize, 0usize), (0, 0), (5, 5), (10, 63), (199, 0)] {
                let out = sliding(&rows, eps, p, f);
                for (i, got) in out.iter().enumerate() {
                    let lo = i.saturating_sub(p);
                    let hi = (i + f).min(rows.len() - 1);
                    let want = fold(&rows[lo..=hi], eps);
                    match got {
                        // PROTOCOL window tolerance: absolute, 1e-9 * max|v|
                        // (values here are in [0, 10))
                        Ok(Some(x)) => assert!(
                            (x - want.unwrap()).abs() <= 1e-9 * 10.0,
                            "eps={eps} p={p} f={f} i={i}: {x} vs {want:?}"
                        ),
                        Ok(None) => assert!(want.is_none()),
                        Err(e) => panic!("eps={eps} p={p} f={f} i={i}: {e:?}"),
                    }
                }
            }
        }
        assert!(REFOLDS.load(Relaxed) > 0, "sharp eps should have tripped the guard");
    }

    #[test]
    fn long_first_frame_refuses_rather_than_guessing() {
        // first frame of 1 + 200 rows (> PREFIX_CAP); the max of the
        // first rows leaves the frame while unrecorded rows remain
        let mut rows: Vec<(f64, f64)> = (0..400).map(|i| (-(i as f64) * 1e-3 - 0.5, i as f64)).collect();
        rows[0] = (0.9, 1000.0);
        let out = sliding(&rows, 1e-3, 0, 200);
        let mut refused = 0;
        for (i, got) in out.iter().enumerate() {
            let want = fold(&rows[i..=(i + 200).min(399)], 1e-3).unwrap();
            match got {
                Ok(Some(x)) => assert!((x - want).abs() <= 1e-9 * 1000.0, "i={i}"),
                Ok(None) => panic!(),
                Err(e) => {
                    assert_eq!(*e, Error::CannotRefold);
                    refused += 1;
                }
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn agg_state_is_the_bare_monoid_and_matches_winstate() {
        let mut r = Lcg(3);
        let rows: Vec<(f64, f64)> = (0..5000).map(|_| (2.0 * r.next() - 1.0, 10.0 * r.next())).collect();
        for &eps in &[1e-4, 1e-3, 0.02, 1.0, 0.0, f64::INFINITY] {
            let mut a = AggState::new();
            let mut w = WinState::new();
            for &(s, v) in &rows {
                a.step(s, v, Some(eps)).unwrap();
                w.check_eps(Some(eps)).unwrap();
                w.step(s, v);
            }
            let x = a.finalize().unwrap();
            assert_eq!(x.to_bits(), w.finalize().unwrap().unwrap().to_bits(), "eps={eps}");
            assert_eq!(x.to_bits(), fold(&rows, eps).unwrap().to_bits());
        }
        let mut a = AggState::new();
        assert_eq!(a.step(0.1, 1.0, None), Err(Error::EpsNull));
        assert_eq!(a.step(0.1, 1.0, Some(-1.0)), Err(Error::EpsInvalid));
        a.step(0.1, 1.0, Some(0.1)).unwrap();
        assert_eq!(a.step(0.1, 1.0, Some(0.2)), Err(Error::EpsNotConstant));
        assert!(std::mem::size_of::<AggState>() <= 40);
    }

    #[test]
    fn inverse_must_be_fifo() {
        let mut st = WinState::new();
        st.check_eps(Some(0.1)).unwrap();
        st.step(0.1, 1.0);
        st.value().unwrap();
        st.step(0.2, 2.0);
        assert_eq!(st.inverse(0.2, 2.0), Err(Error::InverseNotFifo));
        let mut st = WinState::new();
        assert_eq!(st.inverse(0.2, 2.0), Err(Error::InverseOnEmpty));
    }

    #[test]
    fn empty_frame_is_null_and_minus_inf_rows_are_skipped() {
        let mut st = WinState::new();
        assert_eq!(st.value(), Ok(None));
        st.check_eps(Some(0.1)).unwrap();
        st.step(f64::NEG_INFINITY, 3.0);
        assert_eq!(st.value(), Ok(None));
        st.step(0.5, 2.0);
        assert_eq!(st.value(), Ok(Some(2.0)));
        st.inverse(f64::NEG_INFINITY, 3.0).unwrap();
        st.inverse(0.5, 2.0).unwrap();
        assert_eq!(st.value(), Ok(None));
    }
}

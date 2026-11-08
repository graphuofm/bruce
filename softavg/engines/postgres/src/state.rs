//! Pure-Rust aggregate states (no pgrx calls, so the plain `#[test]`s can
//! drive them: a `#[test]` that reaches a pgrx `error!` fails to link,
//! see README.md "Tests").
//!
//! The arithmetic is `softagg_core` (`ScalarAcc` / `InvAcc`), shared with
//! the DuckDB, SQLite and ClickHouse extensions. What lives here is only
//! the PostgreSQL call-site policy, unchanged from 0.1.2:
//!
//! * NULL `value` or `score` rows are skipped (AVG's NULL discipline,
//!   applied to the `(value, score)` pair) -- done by the caller;
//! * a NaN `value` or `score` makes the group NaN (PG value semantics:
//!   `AVG`/`SUM` propagate NaN); the NaN row never enters the core state;
//! * `eps` must be `>= 0`, not NaN, not NULL, and constant within a group.

use softagg_core::{InvAcc, ScalarAcc};

/// Why a row or a merge was rejected. The pgrx layer turns these into
/// the exact 0.1.2 error messages ([`Reject::message`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reject {
    NullEps,
    BadEps(f64),
    EpsChanged(f64, f64),
    CombineEps(f64, f64),
}

impl Reject {
    pub fn message(&self) -> String {
        match *self {
            Reject::NullEps => "softavg: eps must not be NULL".to_string(),
            Reject::BadEps(e) => format!(
                "softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got {e}"
            ),
            Reject::EpsChanged(a, b) => {
                format!("softavg: eps must be constant within an aggregate group ({a} vs {b})")
            }
            Reject::CombineEps(a, b) => {
                format!("softavg: cannot combine states with different eps ({a} vs {b})")
            }
        }
    }
}

/// Validate a live row's `eps` (same domain as bruce-core's `Eps::new`).
#[inline(always)]
pub fn check_eps(eps: Option<f64>) -> Result<f64, Reject> {
    match eps {
        None => Err(Reject::NullEps),
        Some(e) if e < 0.0 || e.is_nan() => Err(Reject::BadEps(e)),
        Some(e) => Ok(e),
    }
}

// ---------------------------------------------------------------------
// Plain aggregate state (SFUNC / COMBINEFUNC / SERIALFUNC / FINALFUNC)
// ---------------------------------------------------------------------

/// The `internal` state of `softavg` / `softavg_state`: allocated once
/// per group (in the aggregate memory context) at the group's first
/// live row, updated in place afterwards. 0.1.2 kept the same four
/// numbers plus the NaN bit in a `float8[5]` varlena that was decoded,
/// re-allocated and re-encoded on every row.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AggState {
    pub acc: ScalarAcc,
    pub eps: f64,
    /// Some qualifying row carried a NaN `value` or `score`.
    pub nan_seen: bool,
}

/// Bytes in the serialized form: mu, z, u, eps (f64 LE) + nan flag.
pub const SERIAL_LEN: usize = 33;

impl AggState {
    #[inline(always)]
    pub fn new(eps: f64) -> Self {
        Self {
            acc: ScalarAcc::new(),
            eps,
            nan_seen: false,
        }
    }

    /// Absorb one live row whose `eps` was already validated.
    #[inline(always)]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) -> Result<(), Reject> {
        // bit-compare like 0.1.2 (`eps0 != eps` on floats; eps is never NaN here)
        if self.eps != eps {
            return Err(Reject::EpsChanged(self.eps, eps));
        }
        if s.is_nan() || v.is_nan() {
            self.nan_seen = true;
        } else {
            self.acc.absorb(s, v, eps);
        }
        Ok(())
    }

    /// COMBINEFUNC: merge `other` (a disjoint row set) into `self`.
    pub fn merge(&mut self, other: &AggState) -> Result<(), Reject> {
        if self.eps != other.eps {
            return Err(Reject::CombineEps(self.eps, other.eps));
        }
        self.acc.merge(&other.acc, self.eps);
        self.nan_seen |= other.nan_seen;
        Ok(())
    }

    /// NaN if a NaN row qualified; `u / z`; `None` (SQL NULL) when the
    /// core state is empty (e.g. every row scored `-inf`).
    #[inline]
    pub fn finalize(&self) -> Option<f64> {
        if self.nan_seen {
            return Some(f64::NAN);
        }
        self.acc.finalize()
    }

    /// The 0.1.2 state array `[mu, z, u, eps, nan_seen]` (softavg_state).
    pub fn to_array(&self) -> [f64; 5] {
        [
            self.acc.mu,
            self.acc.z,
            self.acc.u,
            self.eps,
            if self.nan_seen { 1.0 } else { 0.0 },
        ]
    }

    pub fn serialize(&self) -> [u8; SERIAL_LEN] {
        let mut out = [0u8; SERIAL_LEN];
        out[0..8].copy_from_slice(&self.acc.mu.to_le_bytes());
        out[8..16].copy_from_slice(&self.acc.z.to_le_bytes());
        out[16..24].copy_from_slice(&self.acc.u.to_le_bytes());
        out[24..32].copy_from_slice(&self.eps.to_le_bytes());
        out[32] = self.nan_seen as u8;
        out
    }

    pub fn deserialize(b: &[u8]) -> Option<Self> {
        if b.len() != SERIAL_LEN || b[32] > 1 {
            return None;
        }
        let f = |i: usize| f64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Some(Self {
            acc: ScalarAcc {
                mu: f(0),
                z: f(8),
                u: f(16),
            },
            eps: f(24),
            nan_seen: b[32] == 1,
        })
    }
}

// ---------------------------------------------------------------------
// Moving-aggregate state (MSFUNC / MINVFUNC / MFINALFUNC)
// ---------------------------------------------------------------------

/// Outcome of an inverse transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inverse {
    /// The state now describes the frame without the row.
    Done,
    /// The core reported the state inexact (cancellation guard, or an
    /// emptied argmax set): the host must recompute the frame.
    Punt,
}

/// The `internal` moving-aggregate state. PostgreSQL forbids a moving
/// transition function from returning NULL (NULL from MINVFUNC is the
/// "cannot invert, recompute" signal), so this state exists from the
/// first row of a frame on, even if that row is NULL; `eps` stays NaN
/// until the first live row fixes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovState {
    pub inv: InvAcc,
    /// NaN = not yet fixed by a live row.
    pub eps: f64,
    /// Live NaN rows currently in the frame (a count, not a sticky bit,
    /// so a NaN row that leaves the frame stops poisoning it).
    pub nan_rows: u64,
}

impl Default for MovState {
    fn default() -> Self {
        Self::new()
    }
}

impl MovState {
    pub fn new() -> Self {
        Self {
            inv: InvAcc::new(),
            eps: f64::NAN,
            nan_rows: 0,
        }
    }

    #[inline(always)]
    fn fix_eps(&mut self, eps: f64) -> Result<(), Reject> {
        if self.eps.is_nan() {
            self.eps = eps;
        } else if self.eps != eps {
            return Err(Reject::EpsChanged(self.eps, eps));
        }
        Ok(())
    }

    #[inline(always)]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) -> Result<(), Reject> {
        self.fix_eps(eps)?;
        if s.is_nan() || v.is_nan() {
            self.nan_rows += 1;
        } else {
            self.inv.absorb(s, v, eps);
        }
        Ok(())
    }

    /// Remove a row previously absorbed with the same arguments.
    #[inline(always)]
    pub fn remove(&mut self, s: f64, v: f64, eps: f64) -> Result<Inverse, Reject> {
        self.fix_eps(eps)?;
        if s.is_nan() || v.is_nan() {
            if self.nan_rows == 0 {
                return Ok(Inverse::Punt);
            }
            self.nan_rows -= 1;
            return Ok(Inverse::Done);
        }
        self.inv.remove(s, v, eps);
        Ok(if self.inv.is_exact() {
            Inverse::Done
        } else {
            Inverse::Punt
        })
    }

    /// Same answer as [`AggState::finalize`] over the rows in the frame.
    /// `None` also when inexact; the PG layer never gets there because
    /// MINVFUNC punts first.
    #[inline]
    pub fn finalize(&self) -> Option<f64> {
        if self.nan_rows > 0 {
            return Some(f64::NAN);
        }
        if !self.inv.is_exact() {
            return None;
        }
        self.inv.finalize()
    }
}

/// Bytes of a serialized [`MovState`] (the incremental-view state).
pub const MOV_SERIAL_LEN: usize = 74;
const MOV_SERIAL_VERSION: u8 = 1;

impl MovState {
    /// Merge a disjoint row set (an insert delta) into `self`.
    pub fn merge(&mut self, other: &MovState) -> Result<(), Reject> {
        if other.eps.is_nan() {
            self.nan_rows += other.nan_rows;
            return Ok(());
        }
        self.fix_eps(other.eps)?;
        self.inv.merge(&other.inv, self.eps);
        self.nan_rows += other.nan_rows;
        Ok(())
    }

    /// Live rows that carry weight or are NaN (rows with NULL value/score
    /// and `-inf`-scored rows at finite eps are not counted).
    pub fn rows(&self) -> u64 {
        self.inv.n + self.nan_rows
    }

    /// Little-endian, versioned: [ver, mu, z, u, n, err_z, err_u, v_scale,
    /// eps, nan_rows, inexact].
    pub fn serialize(&self) -> [u8; MOV_SERIAL_LEN] {
        let mut o = [0u8; MOV_SERIAL_LEN];
        o[0] = MOV_SERIAL_VERSION;
        let a = &self.inv;
        let words: [u64; 9] = [
            a.acc.mu.to_bits(),
            a.acc.z.to_bits(),
            a.acc.u.to_bits(),
            a.n,
            a.err_z.to_bits(),
            a.err_u.to_bits(),
            a.v_scale.to_bits(),
            self.eps.to_bits(),
            self.nan_rows,
        ];
        for (i, w) in words.iter().enumerate() {
            o[1 + 8 * i..9 + 8 * i].copy_from_slice(&w.to_le_bytes());
        }
        o[73] = a.inexact as u8;
        o
    }

    pub fn deserialize(b: &[u8]) -> Option<Self> {
        if b.len() != MOV_SERIAL_LEN || b[0] != MOV_SERIAL_VERSION || b[73] > 1 {
            return None;
        }
        let w = |i: usize| u64::from_le_bytes(b[1 + 8 * i..9 + 8 * i].try_into().unwrap());
        let f = |i: usize| f64::from_bits(w(i));
        Some(Self {
            inv: InvAcc {
                acc: ScalarAcc { mu: f(0), z: f(1), u: f(2) },
                n: w(3),
                err_z: f(4),
                err_u: f(5),
                v_scale: f(6),
                inexact: b[73] == 1,
            },
            eps: f(7),
            nan_rows: w(8),
        })
    }
}

// ---------------------------------------------------------------------
// 0.1.2 float8[5] encoding, kept for the SQL-callable helpers
// bruce_softavg_sfunc / _combine / _final (explicit re-association of
// softavg_state outputs in SQL).
// ---------------------------------------------------------------------

pub const STATE_LEN: usize = 5;

/// `None` when the array is not a well-formed state.
pub fn decode_checked(st: &[f64]) -> Option<AggState> {
    if st.len() != STATE_LEN {
        return None;
    }
    Some(AggState {
        acc: ScalarAcc {
            mu: st[0],
            z: st[1],
            u: st[2],
        },
        eps: st[3],
        nan_seen: st[4] != 0.0,
    })
}

#[cfg(test)]
mod pure_tests {
    use super::*;

    fn fold(rows: &[(f64, f64)], eps: f64) -> AggState {
        let mut st = AggState::new(eps);
        for &(s, v) in rows {
            st.absorb(s, v, eps).unwrap();
        }
        st
    }

    #[test]
    fn nan_is_sticky_and_kept_out_of_the_core_state() {
        let rows = [(1.0, 10.0), (f64::NAN, 777.0), (2.0, 20.0), (0.5, f64::NAN)];
        let clean = [(1.0, 10.0), (2.0, 20.0)];
        for eps in [0.0, 1e-4, 0.37, 1.0, f64::INFINITY] {
            let st = fold(&rows, eps);
            assert!(st.nan_seen);
            assert_eq!(st.acc, fold(&clean, eps).acc, "eps={eps}");
            assert!(st.finalize().unwrap().is_nan());
        }
    }

    #[test]
    fn nan_bit_survives_merge_from_either_side() {
        for eps in [0.0, 0.37, f64::INFINITY] {
            let clean = fold(&[(0.5, 1.0)], eps);
            let dirty = fold(&[(0.5, f64::NAN)], eps);
            for (a, b) in [(clean, dirty), (dirty, clean)] {
                let mut m = a;
                m.merge(&b).unwrap();
                assert!(m.finalize().unwrap().is_nan());
            }
            let mut m = clean;
            m.merge(&clean).unwrap();
            assert_eq!(m.finalize(), Some(1.0));
        }
    }

    #[test]
    fn eps_rules() {
        assert_eq!(check_eps(None), Err(Reject::NullEps));
        assert_eq!(check_eps(Some(-0.5)), Err(Reject::BadEps(-0.5)));
        assert!(matches!(check_eps(Some(f64::NAN)), Err(Reject::BadEps(_))));
        assert_eq!(check_eps(Some(0.0)), Ok(0.0));
        let mut st = AggState::new(0.6);
        assert_eq!(st.absorb(0.0, 1.0, 0.7), Err(Reject::EpsChanged(0.6, 0.7)));
        assert_eq!(
            Reject::EpsChanged(0.6, 0.7).message(),
            "softavg: eps must be constant within an aggregate group (0.6 vs 0.7)"
        );
        let mut a = AggState::new(0.1);
        assert_eq!(a.merge(&AggState::new(0.2)), Err(Reject::CombineEps(0.1, 0.2)));
    }

    #[test]
    fn serialization_round_trips() {
        for eps in [0.0, 0.37, f64::INFINITY] {
            for rows in [
                &[(1.0, 10.0), (f64::INFINITY, 5.0)][..],
                &[(f64::NEG_INFINITY, 3.0)][..],
                &[(0.5, f64::NAN), (1.5, 2.0)][..],
                &[][..],
            ] {
                let st = fold(rows, eps);
                let back = AggState::deserialize(&st.serialize()).unwrap();
                assert_eq!(back.serialize(), st.serialize());
                assert_eq!(back.nan_seen, st.nan_seen);
                assert_eq!(back.eps.to_bits(), st.eps.to_bits());
                let arr = st.to_array();
                let dec = decode_checked(&arr).unwrap();
                assert_eq!(dec.serialize(), st.serialize());
            }
        }
        assert!(AggState::deserialize(&[0u8; 32]).is_none());
        assert!(decode_checked(&[0.0; 4]).is_none());
    }

    #[test]
    fn moving_state_nan_rows_leave_the_frame() {
        let mut m = MovState::new();
        m.absorb(0.1, 1.0, 0.5).unwrap();
        m.absorb(f64::NAN, 2.0, 0.5).unwrap();
        assert!(m.finalize().unwrap().is_nan());
        assert_eq!(m.remove(f64::NAN, 2.0, 0.5), Ok(Inverse::Done));
        assert_eq!(m.finalize(), Some(1.0));
    }

    #[test]
    fn moving_state_punts_when_the_argmax_leaves_at_sharp_eps() {
        let mut m = MovState::new();
        m.absorb(1.0, 10.0, 1e-3).unwrap();
        m.absorb(-1.0, 20.0, 1e-3).unwrap();
        // removing the max row leaves e^{-2000} of the peak weight
        assert_eq!(m.remove(1.0, 10.0, 1e-3), Ok(Inverse::Punt));
    }

    #[test]
    fn moving_state_serialization_and_merge() {
        let mut a = MovState::new();
        let mut b = MovState::new();
        let mut whole = MovState::new();
        for i in 0..200 {
            let (s, v) = ((i as f64 * 0.37).cos(), (i % 10) as f64);
            if i % 3 == 0 { a.absorb(s, v, 0.05).unwrap() } else { b.absorb(s, v, 0.05).unwrap() }
            whole.absorb(s, v, 0.05).unwrap();
        }
        b.absorb(f64::NAN, 1.0, 0.05).unwrap();
        let back = MovState::deserialize(&b.serialize()).unwrap();
        assert_eq!(back.serialize(), b.serialize());
        assert_eq!(back.nan_rows, 1);
        a.merge(&back).unwrap();
        assert_eq!(a.rows(), 201);
        assert!(a.finalize().unwrap().is_nan());
        assert_eq!(a.remove(f64::NAN, 1.0, 0.05), Ok(Inverse::Done));
        let (x, y) = (a.finalize().unwrap(), whole.finalize().unwrap());
        assert!((x - y).abs() <= 1e-12 * y.abs());
        // an empty, eps-less state merges as the identity
        let mut e = MovState::new();
        e.merge(&whole).unwrap();
        assert_eq!(e.serialize(), whole.serialize());
        assert!(MovState::deserialize(&[0u8; MOV_SERIAL_LEN]).is_none());
        assert_eq!(std::mem::size_of::<MovState>(), 80);
        assert_eq!(std::mem::size_of::<AggState>(), 40);
    }

    /// The window half of the conformance vectors, driven through the
    /// moving state exactly as PostgreSQL drives MSFUNC/MINVFUNC: on a
    /// punt, recompute the frame from scratch.
    #[test]
    fn conformance_window_through_the_moving_state() {
        let txt = include_str!("../../core/conformance/vectors.json");
        let j: serde_json::Value = serde_json::from_str(txt).unwrap();
        let num = |x: &serde_json::Value| -> f64 {
            match x {
                serde_json::Value::String(s) if s == "inf" => f64::INFINITY,
                serde_json::Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
                _ => x.as_f64().unwrap(),
            }
        };
        let rows: Vec<(f64, f64)> = j["table"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r[0].as_u64().unwrap() < 2000)
            .map(|r| (num(&r[2]), num(&r[3])))
            .collect();
        let k = j["window_rows_preceding"].as_u64().unwrap() as usize;
        let vmax = rows.iter().fold(0.0f64, |a, r| a.max(r.1.abs()));
        for case in j["window"].as_array().unwrap() {
            let eps = num(&case["eps"]);
            let exp: Vec<f64> = case["expected"].as_array().unwrap().iter().map(num).collect();
            let mut m = MovState::new();
            let mut punts = 0;
            for i in 0..rows.len() {
                if i > k {
                    let (s, v) = rows[i - k - 1];
                    if m.remove(s, v, eps).unwrap() == Inverse::Punt {
                        punts += 1;
                        m = MovState::new();
                        for &(s, v) in &rows[i - k..i] {
                            m.absorb(s, v, eps).unwrap();
                        }
                    }
                }
                m.absorb(rows[i].0, rows[i].1, eps).unwrap();
                let got = m.finalize().unwrap();
                let want = exp[i];
                // PROTOCOL: windows are compared in absolute value units
                assert!(
                    (got - want).abs() <= 1e-9 * vmax,
                    "eps={eps} row {i}: {got} vs {want} (punts so far {punts})"
                );
            }
        }
    }
}

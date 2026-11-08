//! A local copy of softagg-core's `InvAcc` (engines/core/src/lib.rs, as of
//! the 2026-11-05 "Decisions" entry) with two changes that do not touch the
//! arithmetic:
//! * the refold tolerance `max(abs_tol, gain * n * U) * v_scale` takes
//!   `abs_tol` / `gain` from fields instead of the crate constants, so the
//!   sweep in main.rs does not change shipped behaviour;
//! * when `I = true`, the bound is additionally split into the three kinds
//!   of rounding it charges (per-row weight terms, running-sum terms,
//!   anchor-rescale terms), rescaled with the anchor exactly like
//!   `err_z` / `err_u`. The split is bookkeeping only; `err_z` / `err_u`
//!   are computed with the core's expressions in the core's order, and
//!   `tests/local_copy.rs` checks bit-identity with the core.

use softagg_core::{ScalarAcc, U};

/// absolute error of an underflowed (subnormal or zero) exp or product
const TINY: f64 = 4.9406564584124654e-324;

#[inline]
fn tiny_terms(w: f64, v: f64) -> (f64, f64) {
    let tz = if w < f64::MIN_POSITIVE { TINY } else { 0.0 };
    let wv = (w * v).abs();
    let tu = tz * v.abs() + if wv < f64::MIN_POSITIVE { TINY } else { 0.0 };
    (tz, tu)
}

pub const W: usize = 0; // per-row weight terms: 2U w (z), 3U |w v| (u)
pub const A: usize = 1; // running-sum terms: U |z1|, U |u1|
pub const R: usize = 2; // anchor-rescale terms: 3U |z0| scale, 3U |u0| scale

#[derive(Clone, Copy, Debug)]
pub struct Local<const I: bool> {
    pub acc: ScalarAcc,
    pub n: u64,
    pub err_z: f64,
    pub err_u: f64,
    pub v_scale: f64,
    pub inexact: bool,
    pub abs_tol: f64,
    pub gain: f64,
    // instrumentation (I = true only)
    pub cz: [f64; 3],
    pub cu: [f64; 3],
    pub ops: u64,
    pub removes: u64,
    /// proposed soundness patch (CORE_CHANGES.md): also charge the rounding
    /// of the exponent argument, 2U|x| relative per weight, and the absolute
    /// error of a weight or product that underflows (2^-1074 each) (off = core)
    pub arg_term: bool,
}

impl<const I: bool> Local<I> {
    pub fn new(abs_tol: f64, gain: f64) -> Self {
        Self {
            acc: ScalarAcc::new(),
            n: 0,
            err_z: 0.0,
            err_u: 0.0,
            v_scale: 0.0,
            inexact: false,
            abs_tol,
            gain,
            cz: [0.0; 3],
            cu: [0.0; 3],
            ops: 0,
            removes: 0,
            arg_term: false,
        }
    }

    pub fn reset(&mut self) {
        let p = self.arg_term;
        *self = Self::new(self.abs_tol, self.gain);
        self.arg_term = p;
    }

    #[inline]
    fn carries_weight(s: f64, eps: f64) -> bool {
        eps.is_infinite() || s != f64::NEG_INFINITY
    }

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

    /// The three summands of `value_error_bound`:
    /// (err_u / (z - err_z), |F| err_z / (z - err_z), U |F|).
    pub fn bound_terms(&self) -> [f64; 3] {
        let z = self.acc.z;
        let f = (self.acc.u / z).abs();
        let d = z - self.err_z;
        [self.err_u / d, f * self.err_z / d, U * f]
    }

    pub fn tol(&self) -> f64 {
        self.abs_tol.max(self.gain * self.n as f64 * U) * self.v_scale.max(f64::MIN_POSITIVE)
    }

    #[inline]
    fn check(&mut self) {
        if self.inexact || self.n == 0 {
            return;
        }
        let tol = self.tol();
        let within = matches!(
            self.value_error_bound().partial_cmp(&tol),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        );
        if !within {
            self.inexact = true;
        }
    }

    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if !Self::carries_weight(s, eps) {
            return;
        }
        if I {
            self.ops += 1;
        }
        self.v_scale = self.v_scale.max(v.abs());
        let (mu0, z0, u0) = (self.acc.mu, self.acc.z, self.acc.u);
        self.acc.absorb(s, v, eps);
        self.n += 1;
        let (z1, u1) = (self.acc.z, self.acc.u);
        if eps > 0.0 && eps.is_finite() && mu0.is_finite() && s.is_finite() && z0 != 0.0 {
            if s <= mu0 {
                let w = ((s - mu0) / eps).exp();
                self.err_z += 2.0 * U * w + U * z1.abs();
                self.err_u += 3.0 * U * (w * v).abs() + U * u1.abs();
                if self.arg_term {
                    let x = ((s - mu0) / eps).abs();
                    let (tz, tu) = tiny_terms(w, v);
                    self.err_z += 2.0 * U * x * w + tz;
                    self.err_u += 2.0 * U * x * (w * v).abs() + tu;
                }
                if I {
                    self.cz[W] += 2.0 * U * w;
                    self.cz[A] += U * z1.abs();
                    self.cu[W] += 3.0 * U * (w * v).abs();
                    self.cu[A] += U * u1.abs();
                }
            } else {
                let scale = ((mu0 - s) / eps).exp();
                self.err_z = self.err_z * scale + 3.0 * U * z0.abs() * scale + U * z1.abs();
                self.err_u = self.err_u * scale + 3.0 * U * u0.abs() * scale + U * u1.abs();
                if self.arg_term {
                    let x = ((mu0 - s) / eps).abs();
                    let t = if scale < f64::MIN_POSITIVE { TINY } else { 0.0 };
                    self.err_z += 2.0 * U * x * z0.abs() * scale + t * z0.abs();
                    self.err_u += 2.0 * U * x * u0.abs() * scale + t * u0.abs();
                }
                if I {
                    for c in self.cz.iter_mut().chain(self.cu.iter_mut()) {
                        *c *= scale;
                    }
                    self.cz[R] += 3.0 * U * z0.abs() * scale;
                    self.cz[A] += U * z1.abs();
                    self.cu[R] += 3.0 * U * u0.abs() * scale;
                    self.cu[A] += U * u1.abs();
                }
            }
        } else if eps.is_infinite() || eps == 0.0 {
            self.err_u += U * u1.abs();
            if I {
                self.cu[A] += U * u1.abs();
            }
        }
        self.check();
    }

    #[inline]
    pub fn remove(&mut self, s: f64, v: f64, eps: f64) {
        if !Self::carries_weight(s, eps) {
            return;
        }
        if I {
            self.ops += 1;
            self.removes += 1;
        }
        if self.n == 0 {
            self.inexact = true;
            return;
        }
        self.n -= 1;
        if self.n == 0 {
            let (t, g) = (self.abs_tol, self.gain);
            *self = Self::new(t, g);
            return;
        }
        let a = &mut self.acc;
        if eps.is_infinite() {
            a.z -= 1.0;
            a.u -= v;
            self.err_u += U * a.u.abs();
            if I {
                self.cu[A] += U * a.u.abs();
            }
            self.check();
            return;
        }
        if eps == 0.0 || a.mu == f64::INFINITY || s == f64::INFINITY {
            if s == a.mu {
                a.z -= 1.0;
                a.u -= v;
                self.err_u += U * a.u.abs();
                if a.z <= 0.0 {
                    self.inexact = true;
                }
            } else if s > a.mu {
                self.inexact = true;
            }
            self.check();
            return;
        }
        if s > a.mu {
            self.inexact = true;
            return;
        }
        let w = ((s - a.mu) / eps).exp();
        a.u -= w * v;
        a.z -= w;
        self.err_z += 2.0 * U * w + U * a.z.abs();
        self.err_u += 3.0 * U * (w * v).abs() + U * a.u.abs();
        if self.arg_term {
            let x = ((s - a.mu) / eps).abs();
            let (tz, tu) = tiny_terms(w, v);
            self.err_z += 2.0 * U * x * w + tz;
            self.err_u += 2.0 * U * x * (w * v).abs() + tu;
        }
        if I {
            self.cz[W] += 2.0 * U * w;
            self.cz[A] += U * a.z.abs();
            self.cu[W] += 3.0 * U * (w * v).abs();
            self.cu[A] += U * a.u.abs();
        }
        self.check();
    }

    #[inline]
    pub fn is_exact(&self) -> bool {
        !self.inexact
    }

    pub fn finalize(&self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        self.acc.finalize()
    }
}

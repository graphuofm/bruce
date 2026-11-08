//! Prototype of a tighter inverse-capable state ("DdInv"): same max anchor
//! and the same refold tolerance as softagg-core's `InvAcc`, but
//! * `z` and `u` are double-double sums (TwoSum; `w * v` via FMA TwoProd),
//!   so additions and subtractions no longer leave an O(U |z_running|)
//!   residual behind;
//! * the exponent argument `(s - mu) / eps` is formed to ~U^2 (TwoSum for
//!   the difference, an FMA remainder for the division, then
//!   `exp(x + xl) ~ e + e*xl`), so a weight is within 3U of
//!   `e^{(s - mu)/eps}` regardless of how far below the anchor it is;
//! * with an unchanged anchor, the weight subtracted by `remove` is bit-for-
//!   bit the weight that `absorb` added, so removal cancels exactly. Only a
//!   row absorbed before an anchor rise can mismatch, by at most
//!   `(K + 2) * 3U` of its weight (K = rises since the last fresh state);
//!   rows with `s > mu_prev` (the anchor before the latest rise) are known
//!   to have been absorbed after it and are exempt.
//!
//! Bound on |delivered - exact|, first order:
//!   (ru + |F| rz) / (z - rz)          residuals: mismatches, dd rounding,
//!                                      subnormal weights
//! + (K + 1) * 3U * (v_scale + |F|)    every live weight is within
//!                                      (K + 1) * 3U of exact
//! + U |F|                             the final division.
//! Weights below the smallest normal f64 (exponent argument < ln 2^-1022)
//! are skipped instead of computed: no subnormal `exp` or subnormal dd
//! arithmetic (both are slow on x86), and each skipped absorb/remove charges
//! 2 * 2^-1022 * (1, 1 + |v|) to (rz, ru). An anchor rise by more than that
//! drops the old contributions the same way. Absorb and remove under the
//! same anchor make the same skip decision.
//! Finite eps > 0 and finite scores only (prototype).

use crate::dd::two_sum;
use softagg_core::U;

/// relative error of one computed weight: exp (1 ulp = 2U) + the FMA (U),
/// with headroom for the O(U^2) argument and correction terms
const CW: f64 = 3.01 * U;
/// absolute charge for a skipped (sub-normal) weight
const SKIP: f64 = 2.0 * f64::MIN_POSITIVE;
/// ln(2^-1022): below this the weight is not a normal f64
const LN_MIN_NORMAL: f64 = -708.3964185322641;
/// dd rounding per operation, relative to the operands (generous)
const DDR: f64 = 4.0 * U * U;

#[derive(Clone, Copy, Debug)]
pub struct DdInv {
    pub mu: f64,
    pub zh: f64,
    pub zl: f64,
    pub uh: f64,
    pub ul: f64,
    pub n: u64,
    pub mu_prev: f64,
    pub k: u64,
    pub rz: f64,
    pub ru: f64,
    pub v_scale: f64,
    pub inexact: bool,
    pub abs_tol: f64,
    pub gain: f64,
}

#[inline]
fn weight(s: f64, mu: f64, eps: f64) -> f64 {
    let (d, dl) = two_sum(s, -mu); // s - mu = d + dl exactly
    let x = d / eps;
    let rem = (-x).mul_add(eps, d); // d - x*eps, exact
    if x < LN_MIN_NORMAL {
        return 0.0; // skipped
    }
    let xl = (rem + dl) / eps;
    let e = x.exp();
    e.mul_add(xl, e)
}

#[inline]
fn dd_add(h: &mut f64, l: &mut f64, a: f64) {
    let (s, e) = two_sum(*h, a);
    let e = e + *l;
    let hs = s + e;
    *l = e - (hs - s);
    *h = hs;
}

#[inline]
fn dd_scale(h: &mut f64, l: &mut f64, c: f64) {
    let p = *h * c;
    let pe = h.mul_add(c, -p) + *l * c;
    let hs = p + pe;
    *l = pe - (hs - p);
    *h = hs;
}

impl DdInv {
    pub fn new(abs_tol: f64, gain: f64) -> Self {
        Self {
            mu: f64::NEG_INFINITY,
            zh: 0.0,
            zl: 0.0,
            uh: 0.0,
            ul: 0.0,
            n: 0,
            mu_prev: f64::NEG_INFINITY,
            k: 0,
            rz: 0.0,
            ru: 0.0,
            v_scale: 0.0,
            inexact: false,
            abs_tol,
            gain,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.abs_tol, self.gain);
    }

    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        debug_assert!(eps > 0.0 && eps.is_finite() && s.is_finite());
        self.v_scale = self.v_scale.max(v.abs());
        self.n += 1;
        if self.n == 1 {
            self.mu = s;
            self.zh = 1.0;
            self.zl = 0.0;
            self.uh = v;
            self.ul = 0.0;
            self.check();
            return;
        }
        if s <= self.mu {
            let w = weight(s, self.mu, eps);
            if w == 0.0 {
                self.rz += SKIP;
                self.ru += SKIP * (1.0 + v.abs());
            } else {
                self.add_row(w, v, 1.0);
            }
        } else {
            let scale = weight(self.mu, s, eps);
            if scale == 0.0 {
                // every old contribution falls below 2^-1022 of the new row
                self.rz = SKIP * (self.zh.abs() + self.rz + 1.0);
                self.ru = SKIP * (self.uh.abs() + self.ru + 1.0);
                self.zh = 0.0;
                self.zl = 0.0;
                self.uh = 0.0;
                self.ul = 0.0;
            } else {
                dd_scale(&mut self.zh, &mut self.zl, scale);
                dd_scale(&mut self.uh, &mut self.ul, scale);
                self.rz = self.rz * scale + DDR * self.zh.abs();
                self.ru = self.ru * scale + DDR * self.uh.abs();
            }
            dd_add(&mut self.zh, &mut self.zl, 1.0);
            dd_add(&mut self.uh, &mut self.ul, v);
            self.rz += DDR * self.zh.abs();
            self.ru += DDR * (self.uh.abs() + v.abs());
            self.mu_prev = self.mu;
            self.mu = s;
            self.k += 1;
        }
        self.check();
    }

    #[inline]
    fn add_row(&mut self, w: f64, v: f64, sign: f64) {
        let (p, pe) = (w * v, w.mul_add(v, -(w * v)));
        dd_add(&mut self.zh, &mut self.zl, sign * w);
        dd_add(&mut self.uh, &mut self.ul, sign * p);
        self.ul += sign * pe;
        self.rz += DDR * (self.zh.abs() + w);
        self.ru += DDR * (self.uh.abs() + p.abs());
    }

    #[inline]
    pub fn remove(&mut self, s: f64, v: f64, eps: f64) {
        if self.n == 0 || s > self.mu {
            self.inexact = true;
            return;
        }
        self.n -= 1;
        if self.n == 0 {
            self.reset();
            return;
        }
        let w = weight(s, self.mu, eps);
        if w == 0.0 {
            // skipped now; if it was added before an anchor rise, what is
            // left of it is below 2^-1022 (1 + (K+2) CW)
            self.rz += SKIP;
            self.ru += SKIP * (1.0 + v.abs());
        } else {
            self.add_row(w, v, -1.0);
            if self.k > 0 && s <= self.mu_prev {
                // possibly absorbed before an anchor rise: stored and removed
                // weights may differ
                let m = (self.k as f64 + 2.0) * CW * w;
                self.rz += m;
                self.ru += m * v.abs();
            }
        }
        self.check();
    }

    #[inline]
    pub fn z(&self) -> f64 {
        self.zh + self.zl
    }

    pub fn finalize(&self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let q = self.uh / self.zh;
        // one Newton correction with the low parts
        let r = (-q).mul_add(self.zh, self.uh) + self.ul - q * self.zl;
        Some(q + r / self.zh)
    }

    pub fn value_error_bound(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let z = self.z();
        if !(z > self.rz) {
            return f64::INFINITY;
        }
        let f = self.finalize().unwrap().abs();
        (self.ru + f * self.rz) / (z - self.rz) + (self.k as f64 + 1.0) * CW * (self.v_scale + f) + 1.01 * U * f
    }

    pub fn tol(&self) -> f64 {
        self.abs_tol.max(self.gain * self.n as f64 * U) * self.v_scale.max(f64::MIN_POSITIVE)
    }

    #[inline]
    fn check(&mut self) {
        if self.inexact || self.n == 0 {
            return;
        }
        if !(self.value_error_bound() <= self.tol()) {
            self.inexact = true;
        }
    }

    pub fn is_exact(&self) -> bool {
        !self.inexact
    }
}

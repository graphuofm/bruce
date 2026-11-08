//! Mergeable effective-sample-size accumulator for the softmax weights of
//! SOFTAVG: ESS = (sum w)^2 / sum w^2 with w = e^{s/eps}, kept under the
//! same max anchor as `ScalarAcc` (z = sum w, q = sum w^2, both scaled by
//! e^{-mu/eps} and e^{-2 mu/eps}; a re-anchor multiplies z by `scale` and q
//! by `scale^2`). Regimes and the +-inf score policy mirror
//! `ScalarAcc::absorb`: eps = 0 gives the argmax multiplicity, eps = inf
//! gives the row count, -inf scores carry no weight.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EssAcc {
    pub mu: f64,
    pub z: f64,
    pub q: f64,
}

impl Default for EssAcc {
    fn default() -> Self {
        Self::new()
    }
}

impl EssAcc {
    pub fn new() -> Self {
        Self { mu: f64::NEG_INFINITY, z: 0.0, q: 0.0 }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.z == 0.0
    }

    #[inline]
    pub fn absorb(&mut self, s: f64, eps: f64) {
        if eps == 0.0 {
            if s == f64::NEG_INFINITY {
                return;
            }
            if s > self.mu {
                *self = Self { mu: s, z: 1.0, q: 1.0 };
            } else if s == self.mu {
                self.z += 1.0;
                self.q += 1.0;
            }
            return;
        }
        if eps.is_infinite() {
            self.z += 1.0;
            self.q += 1.0;
            return;
        }
        if s == f64::NEG_INFINITY {
            return;
        }
        if s == f64::INFINITY || self.mu == f64::INFINITY {
            if s == f64::INFINITY && self.mu == f64::INFINITY {
                self.z += 1.0;
                self.q += 1.0;
            } else if s == f64::INFINITY {
                *self = Self { mu: s, z: 1.0, q: 1.0 };
            }
            return;
        }
        if self.is_empty() {
            *self = Self { mu: s, z: 1.0, q: 1.0 };
            return;
        }
        if s <= self.mu {
            let w = ((s - self.mu) / eps).exp();
            self.z += w;
            self.q += w * w;
        } else {
            let scale = ((self.mu - s) / eps).exp();
            self.z = self.z * scale + 1.0;
            self.q = self.q * (scale * scale) + 1.0;
            self.mu = s;
        }
    }

    pub fn merge(&mut self, o: &EssAcc, eps: f64) {
        if o.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = *o;
            return;
        }
        if eps == 0.0 {
            if o.mu > self.mu {
                *self = *o;
            } else if o.mu == self.mu {
                self.z += o.z;
                self.q += o.q;
            }
            return;
        }
        if eps.is_infinite() {
            self.z += o.z;
            self.q += o.q;
            return;
        }
        if self.mu == f64::INFINITY || o.mu == f64::INFINITY {
            if self.mu == f64::INFINITY && o.mu == f64::INFINITY {
                self.z += o.z;
                self.q += o.q;
            } else if o.mu == f64::INFINITY {
                *self = *o;
            }
            return;
        }
        let m = self.mu.max(o.mu);
        let s1 = ((self.mu - m) / eps).exp();
        let s2 = ((o.mu - m) / eps).exp();
        self.z = self.z * s1 + o.z * s2;
        self.q = self.q * (s1 * s1) + o.q * (s2 * s2);
        self.mu = m;
    }

    /// (sum w)^2 / sum w^2, in [1, n]; None when empty.
    pub fn finalize(&self) -> Option<f64> {
        if self.is_empty() {
            None
        } else {
            Some(self.z * self.z / self.q)
        }
    }
}

/// softavg and its ESS in one state (the "fold it into softavg" option):
/// ScalarAcc's (mu, z, u) plus q. Used only to time the extra per-row cost.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SoftEss {
    pub mu: f64,
    pub z: f64,
    pub u: f64,
    pub q: f64,
}

impl SoftEss {
    pub fn new() -> Self {
        Self { mu: f64::NEG_INFINITY, z: 0.0, u: 0.0, q: 0.0 }
    }
    /// finite eps > 0, finite s (timing only)
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if self.z == 0.0 {
            *self = Self { mu: s, z: 1.0, u: v, q: 1.0 };
            return;
        }
        if s <= self.mu {
            let w = ((s - self.mu) / eps).exp();
            self.u += w * v;
            self.z += w;
            self.q += w * w;
        } else {
            let scale = ((self.mu - s) / eps).exp();
            self.u = self.u * scale + v;
            self.z = self.z * scale + 1.0;
            self.q = self.q * (scale * scale) + 1.0;
            self.mu = s;
        }
    }
}

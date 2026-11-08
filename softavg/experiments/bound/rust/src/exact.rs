//! Exact reference for SOFTAVG over the f64 inputs (finite eps > 0):
//! max-anchored states in double-double, with the exponent argument
//! `(mu_a - mu)/eps` itself formed in double-double. Kept in a segment tree
//! so that a sliding frame, a point insert/delete, or a suffix costs
//! O(log n) double-double merges instead of an O(frame) refold.
//! Accuracy: ~1e-28 relative per merge, far below anything measured here.

use crate::dd::{exp_dd, DD};

#[derive(Clone, Copy, Debug)]
pub struct Ex {
    pub mu: f64,
    pub z: DD,
    pub u: DD,
}

impl Ex {
    pub const EMPTY: Ex = Ex { mu: f64::NEG_INFINITY, z: DD::ZERO, u: DD::ZERO };

    #[inline]
    pub fn leaf(s: f64, v: f64) -> Ex {
        Ex { mu: s, z: DD::ONE, u: DD::from(v) }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.z.hi == 0.0
    }

    #[inline]
    fn rescaled(&self, m: f64, eps: f64) -> (DD, DD) {
        if self.mu == m {
            return (self.z, self.u);
        }
        let f = exp_dd(DD::diff(self.mu, m).div_f(eps));
        (self.z.mul(f), self.u.mul(f))
    }

    #[inline]
    pub fn merge(a: &Ex, b: &Ex, eps: f64) -> Ex {
        if a.is_empty() {
            return *b;
        }
        if b.is_empty() {
            return *a;
        }
        let m = a.mu.max(b.mu);
        let (za, ua) = a.rescaled(m, eps);
        let (zb, ub) = b.rescaled(m, eps);
        Ex { mu: m, z: za.add(zb), u: ua.add(ub) }
    }

    /// u / z in double-double (None if empty)
    pub fn value(&self) -> Option<DD> {
        if self.is_empty() {
            None
        } else {
            Some(self.u.div(self.z))
        }
    }
}

/// Iterative segment tree (commutative monoid, arbitrary n).
pub struct SegTree {
    n: usize,
    t: Vec<Ex>,
    eps: f64,
}

impl SegTree {
    /// leaves[i] = Some((s, v)) for an active row, None for an empty slot
    pub fn build(leaves: &[Option<(f64, f64)>], eps: f64) -> SegTree {
        let n = leaves.len().max(1);
        let mut t = vec![Ex::EMPTY; 2 * n];
        for (i, l) in leaves.iter().enumerate() {
            if let Some((s, v)) = l {
                t[n + i] = Ex::leaf(*s, *v);
            }
        }
        for i in (1..n).rev() {
            t[i] = Ex::merge(&t[2 * i], &t[2 * i + 1], eps);
        }
        SegTree { n, t, eps }
    }

    pub fn set(&mut self, i: usize, x: Option<(f64, f64)>) {
        let mut p = i + self.n;
        self.t[p] = match x {
            Some((s, v)) => Ex::leaf(s, v),
            None => Ex::EMPTY,
        };
        while p > 1 {
            p >>= 1;
            self.t[p] = Ex::merge(&self.t[2 * p], &self.t[2 * p + 1], self.eps);
        }
    }

    /// state of the half-open range [l, r)
    pub fn query(&self, l: usize, r: usize) -> Ex {
        let (mut l, mut r) = (l + self.n, r + self.n);
        let mut acc = Ex::EMPTY;
        while l < r {
            if l & 1 == 1 {
                acc = Ex::merge(&acc, &self.t[l], self.eps);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                acc = Ex::merge(&acc, &self.t[r], self.eps);
            }
            l >>= 1;
            r >>= 1;
        }
        acc
    }

    pub fn all(&self) -> Ex {
        self.query(0, self.n)
    }
}

/// Exact value of a row set by a direct double-double fold (tests).
pub fn exact_fold(rows: &[(f64, f64)], eps: f64) -> Option<DD> {
    let m = rows.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
    if rows.is_empty() {
        return None;
    }
    let (mut z, mut u) = (DD::ZERO, DD::ZERO);
    for &(s, v) in rows {
        let w = exp_dd(DD::diff(s, m).div_f(eps));
        z = z.add(w);
        u = u.add(w.mul_f(v));
    }
    Some(u.div(z))
}

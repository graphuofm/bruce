//! Double-double arithmetic (about 106 significant bits) and a
//! double-double `exp`, used ONLY to build the exact reference value of a
//! soft average (`exact.rs`). Error-free transformations: Knuth TwoSum,
//! FMA TwoProd (Dekker/Bailey QD style).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DD {
    pub hi: f64,
    pub lo: f64,
}

#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl DD {
    pub const ZERO: DD = DD { hi: 0.0, lo: 0.0 };
    pub const ONE: DD = DD { hi: 1.0, lo: 0.0 };

    #[inline]
    pub fn from(x: f64) -> DD {
        DD { hi: x, lo: 0.0 }
    }

    /// exact difference of two f64
    #[inline]
    pub fn diff(a: f64, b: f64) -> DD {
        let (s, e) = two_sum(a, -b);
        DD { hi: s, lo: e }
    }

    #[inline]
    pub fn add(self, b: DD) -> DD {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        DD { hi, lo }
    }

    #[inline]
    pub fn neg(self) -> DD {
        DD { hi: -self.hi, lo: -self.lo }
    }

    #[inline]
    pub fn sub(self, b: DD) -> DD {
        self.add(b.neg())
    }

    #[inline]
    pub fn mul(self, b: DD) -> DD {
        let (p1, p2) = two_prod(self.hi, b.hi);
        let p2 = p2 + (self.hi * b.lo + self.lo * b.hi);
        let (hi, lo) = quick_two_sum(p1, p2);
        DD { hi, lo }
    }

    #[inline]
    pub fn mul_f(self, b: f64) -> DD {
        let (p1, p2) = two_prod(self.hi, b);
        let p2 = p2 + self.lo * b;
        let (hi, lo) = quick_two_sum(p1, p2);
        DD { hi, lo }
    }

    #[inline]
    pub fn div(self, b: DD) -> DD {
        let q1 = self.hi / b.hi;
        let r = self.sub(b.mul_f(q1));
        let q2 = r.hi / b.hi;
        let r = r.sub(b.mul_f(q2));
        let q3 = r.hi / b.hi;
        let (q1, q2) = quick_two_sum(q1, q2);
        DD { hi: q1, lo: q2 }.add(DD::from(q3))
    }

    #[inline]
    pub fn div_f(self, b: f64) -> DD {
        self.div(DD::from(b))
    }

    /// multiply by 2^k exactly (unless the result is subnormal)
    #[inline]
    pub fn ldexp(self, k: i32) -> DD {
        let h = k / 2;
        let f1 = pow2(h);
        let f2 = pow2(k - h);
        DD { hi: self.hi * f1 * f2, lo: self.lo * f1 * f2 }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

#[inline]
fn pow2(k: i32) -> f64 {
    // 2^k for -1022 <= k <= 1023 as a normal f64
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

const LN2: DD = DD { hi: 6.931471805599452862e-01, lo: 2.319046813846299558e-17 };

/// exp of a double-double argument, relative error ~1e-29 on the normal
/// range; in the subnormal range (x < -708) only the high part survives
/// (absolute error < 2^-1074); returns 0 below -800.
pub fn exp_dd(x: DD) -> DD {
    if x.hi == 0.0 && x.lo == 0.0 {
        return DD::ONE;
    }
    if x.hi < -800.0 {
        return DD::ZERO;
    }
    assert!(x.hi < 709.0, "exp_dd overflow: {x:?}");
    let k = (x.hi / LN2.hi).round();
    let r = x.sub(LN2.mul_f(k));
    // r in about [-ln2/2, ln2/2]; scale down by 2^-10 then square 10 times
    let r = r.mul_f(1.0 / 1024.0);
    let mut term = r;
    let mut sum = DD::ONE.add(r);
    for i in 2..=13 {
        term = term.mul(r).div_f(i as f64);
        sum = sum.add(term);
    }
    for _ in 0..10 {
        sum = sum.mul(sum);
    }
    sum.ldexp(k as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_dd_matches_f64_exp() {
        for &x in &[-700.0, -300.5, -20.25, -1.0, -1e-3, 0.0, 0.5, 3.0, 100.0] {
            let e = exp_dd(DD::from(x)).to_f64();
            let r = f64::exp(x);
            assert!(((e - r) / r).abs() < 3e-16, "x={x}: {e} vs {r}");
        }
        // exp(a)exp(b) = exp(a+b) to dd accuracy
        let a = DD::from(-3.25);
        let b = DD::from(-1.0 / 3.0);
        let lhs = exp_dd(a).mul(exp_dd(b));
        let rhs = exp_dd(a.add(b));
        let rel = lhs.sub(rhs).div(rhs).to_f64().abs();
        assert!(rel < 1e-28, "rel {rel:e}");
    }
}

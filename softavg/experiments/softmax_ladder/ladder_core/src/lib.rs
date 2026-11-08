//! Softmax-ladder ablation, rung S3: online softmax WITH plain removal and
//! NO certificate.
//!
//! [`PlainInv`] is softagg-core's [`InvAcc`](softagg_core::InvAcc) with every
//! line that maintains or consults the running error certificate deleted
//! (`err_z`, `err_u`, `v_scale`, `inexact`, `check`). What is left:
//!
//! * absorb = `ScalarAcc::absorb` (bit-identical, it IS that function) plus the
//!   live-row count `n`;
//! * merge  = `ScalarAcc::merge` plus `n`;
//! * remove = `InvAcc::remove`'s subtraction at the CURRENT anchor
//!   (`w = exp((s - mu)/eps); u -= w v; z -= w`), never re-anchors, never
//!   refuses. The only bookkeeping kept from `InvAcc::remove` is the row count:
//!   removing the last live row returns the state to empty (exact by
//!   construction, no certificate involved).
//!
//! So S3 and S4 (the shipped `InvAcc` + host refold) differ ONLY in the missing
//! certificate / refold. Consequences S3 inherits from `ScalarAcc` unchanged:
//! `finalize` returns `None` (SQL NULL) when `z == 0.0`, and `absorb` treats a
//! state whose `z` cancelled to exactly 0.0 as empty (the next row becomes the
//! anchor and the remaining live rows' weight is dropped).

use softagg_core::ScalarAcc;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlainInv {
    pub acc: ScalarAcc,
    /// live rows that carry weight (as `InvAcc::n`)
    pub n: u64,
}

impl Default for PlainInv {
    fn default() -> Self {
        Self::new()
    }
}

impl PlainInv {
    pub fn new() -> Self {
        Self { acc: ScalarAcc::new(), n: 0 }
    }

    #[inline]
    fn carries_weight(s: f64, eps: f64) -> bool {
        eps.is_infinite() || s != f64::NEG_INFINITY
    }

    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if !Self::carries_weight(s, eps) {
            return;
        }
        self.acc.absorb(s, v, eps);
        self.n += 1;
    }

    /// Plain subtraction at the current anchor. No certificate, no refusal.
    #[inline]
    pub fn remove(&mut self, s: f64, v: f64, eps: f64) {
        if !Self::carries_weight(s, eps) {
            return;
        }
        if self.n == 0 {
            return;
        }
        self.n -= 1;
        if self.n == 0 {
            *self = Self::new();
            return;
        }
        let a = &mut self.acc;
        if eps.is_infinite() {
            a.z -= 1.0;
            a.u -= v;
            return;
        }
        if eps == 0.0 || a.mu == f64::INFINITY || s == f64::INFINITY {
            if s == a.mu {
                a.z -= 1.0;
                a.u -= v;
            }
            return;
        }
        let w = ((s - a.mu) / eps).exp();
        a.u -= w * v;
        a.z -= w;
    }

    pub fn merge(&mut self, other: &PlainInv, eps: f64) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        self.acc.merge(&other.acc, eps);
        self.n += other.n;
    }

    /// `u / z`; `None` when no live row or when `z == 0.0` (ScalarAcc's rule).
    pub fn finalize(&self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        self.acc.finalize()
    }

    pub const SERIAL_LEN: usize = 32;

    /// little-endian [mu, z, u, n]
    pub fn serialize(&self) -> [u8; 32] {
        let mut o = [0u8; 32];
        for (i, w) in [self.acc.mu.to_bits(), self.acc.z.to_bits(), self.acc.u.to_bits(), self.n].iter().enumerate() {
            o[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
        }
        o
    }

    pub fn deserialize(b: &[u8]) -> Option<Self> {
        if b.len() != 32 {
            return None;
        }
        let w = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
        Some(Self {
            acc: ScalarAcc { mu: f64::from_bits(w(0)), z: f64::from_bits(w(1)), u: f64::from_bits(w(2)) },
            n: w(3),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use softagg_core::InvAcc;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// While the certificate holds, S3's (mu, z, u) is bit-identical to
    /// InvAcc's: the only difference is the certificate.
    #[test]
    fn same_bits_as_invacc_until_the_certificate_fires() {
        let mut r = Lcg(7);
        for &eps in &[1e-3, 0.02, 0.1, 1.0, f64::INFINITY] {
            let rows: Vec<(f64, f64)> = (0..5000).map(|_| (r.next(), 1.0 + 9.0 * r.next())).collect();
            let (mut a, mut b) = (PlainInv::new(), InvAcc::new());
            let k = 50;
            for i in 0..rows.len() {
                a.absorb(rows[i].0, rows[i].1, eps);
                b.absorb(rows[i].0, rows[i].1, eps);
                if i >= k {
                    let (s, v) = rows[i - k];
                    a.remove(s, v, eps);
                    b.remove(s, v, eps);
                }
                if !b.is_exact() {
                    break;
                }
                assert_eq!(a.acc, b.acc, "eps {eps} row {i}");
                assert_eq!(a.n, b.n);
                assert_eq!(a.finalize(), b.finalize());
            }
        }
    }

    #[test]
    fn absorb_and_merge_are_scalaracc() {
        let mut r = Lcg(3);
        let rows: Vec<(f64, f64)> = (0..1000).map(|_| (r.next(), r.next())).collect();
        let (mut a, mut s) = (PlainInv::new(), ScalarAcc::new());
        for &(x, v) in &rows {
            a.absorb(x, v, 0.02);
            s.absorb(x, v, 0.02);
        }
        assert_eq!(a.acc, s);
        let (mut p, mut q) = (PlainInv::new(), PlainInv::new());
        for &(x, v) in &rows[..400] {
            p.absorb(x, v, 0.02);
        }
        for &(x, v) in &rows[400..] {
            q.absorb(x, v, 0.02);
        }
        let mut s2 = p.acc;
        s2.merge(&q.acc, 0.02);
        p.merge(&q, 0.02);
        assert_eq!(p.acc, s2);
        assert_eq!(p.n, 1000);
        assert_eq!(PlainInv::deserialize(&p.serialize()), Some(p));
    }

    /// The failure S3 exists to show: the argmax leaves at sharp eps, the
    /// anchor stays, the survivors' weights cancel / underflow.
    #[test]
    fn stale_anchor_goes_wrong_without_refusing() {
        let eps = 1e-3;
        let mut a = PlainInv::new();
        a.absorb(1.0, 5.0, eps);
        a.absorb(0.95, 1.0, eps); // weight e^-50 under the anchor
        a.absorb(0.95, 3.0, eps);
        a.remove(1.0, 5.0, eps);
        // exact answer is 2.0; S3 returns whatever the cancelled sums say
        let got = a.finalize();
        assert!(got.map_or(true, |x| (x - 2.0).abs() > 1e-6), "{got:?}");
    }
}

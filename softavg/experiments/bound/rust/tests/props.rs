use bound_exp::ddacc::DdInv;
use bound_exp::ess::EssAcc;
use bound_exp::exact::exact_fold;
use bound_exp::local::Local;
use softagg_core::{InvAcc, ABS_TOL, REFOLD_GAIN};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn cosine_like(seed: u64, n: usize) -> Vec<(f64, f64)> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|i| {
            let c = 0.5 + 0.3 * ((i as f64) / 700.0).sin();
            (c + 0.1 * (r.next() - 0.5), (1.0 + 90.0 * r.next()).round() / 10.0)
        })
        .collect()
}

fn same(c: &InvAcc, l: &Local<false>) -> bool {
    c.acc == l.acc
        && c.n == l.n
        && c.err_z.to_bits() == l.err_z.to_bits()
        && c.err_u.to_bits() == l.err_u.to_bits()
        && c.v_scale.to_bits() == l.v_scale.to_bits()
        && c.inexact == l.inexact
}

/// The local copy with the default tolerance is the core, bit for bit,
/// across absorb, remove, inexact flags and refolds, in every eps regime.
#[test]
fn local_copy_is_bit_identical_to_core() {
    let rows = cosine_like(1, 20_000);
    for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0, 0.0, f64::INFINITY] {
        let mut c = InvAcc::new();
        let mut l = Local::<false>::new(ABS_TOL, REFOLD_GAIN);
        let w = 300;
        for i in 0..rows.len() {
            c.absorb(rows[i].0, rows[i].1, eps);
            l.absorb(rows[i].0, rows[i].1, eps);
            assert!(same(&c, &l), "absorb eps={eps} i={i}");
            if i >= w {
                c.remove(rows[i - w].0, rows[i - w].1, eps);
                l.remove(rows[i - w].0, rows[i - w].1, eps);
                assert!(same(&c, &l), "remove eps={eps} i={i}");
            }
            assert_eq!(c.value_error_bound().to_bits(), l.value_error_bound().to_bits());
            if !c.is_exact() {
                c.reset();
                l.reset();
                for &(s, v) in &rows[(i + 1).saturating_sub(w)..=i] {
                    c.absorb(s, v, eps);
                    l.absorb(s, v, eps);
                }
                assert!(same(&c, &l));
            }
        }
    }
}

/// DdInv: every value reported exact is within its bound of the exact
/// (double-double) value; sliding windows and descending deletes.
#[test]
fn ddinv_bound_holds_against_exact() {
    let rows = cosine_like(7, 6000);
    for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0] {
        for &w in &[50usize, 400] {
            let mut a = DdInv::new(ABS_TOL, REFOLD_GAIN);
            let mut refolds = 0;
            for i in 0..rows.len() {
                a.absorb(rows[i].0, rows[i].1, eps);
                if i >= w {
                    a.remove(rows[i - w].0, rows[i - w].1, eps);
                }
                let frame = &rows[(i + 1).saturating_sub(w)..=i];
                if !a.is_exact() {
                    refolds += 1;
                    a.reset();
                    for &(s, v) in frame {
                        a.absorb(s, v, eps);
                    }
                }
                let ex = exact_fold(frame, eps).unwrap();
                let e = ((a.finalize().unwrap() - ex.hi) - ex.lo).abs();
                assert!(e <= a.value_error_bound(), "eps={eps} w={w} i={i}: {e:e} > {:e}", a.value_error_bound());
            }
            eprintln!("ddinv eps={eps} w={w} refolds={refolds}");
        }
        // descending deletes
        let mut r = rows[..1500].to_vec();
        r.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        let mut a = DdInv::new(ABS_TOL, REFOLD_GAIN);
        for &(s, v) in rows[..1500].iter() {
            a.absorb(s, v, eps);
        }
        let live: Vec<(f64, f64)> = r.clone();
        for i in 0..live.len() - 1 {
            a.remove(live[i].0, live[i].1, eps);
            let rest = &live[i + 1..];
            if !a.is_exact() {
                a.reset();
                for &(s, v) in rest {
                    a.absorb(s, v, eps);
                }
            }
            let ex = exact_fold(rest, eps).unwrap();
            let e = ((a.finalize().unwrap() - ex.hi) - ex.lo).abs();
            assert!(e <= a.value_error_bound(), "desc eps={eps} i={i}");
        }
    }
}

fn ess_ref(s: &[f64], eps: f64) -> f64 {
    if eps.is_infinite() {
        return s.len() as f64;
    }
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if eps == 0.0 {
        return s.iter().filter(|&&x| x == m).count() as f64;
    }
    let (mut z, mut q) = (0.0, 0.0);
    for &x in s {
        let w = ((x - m) / eps).exp();
        z += w;
        q += w * w;
    }
    z * z / q
}

#[test]
fn ess_matches_two_pass_and_merge_equals_sequential() {
    let rows = cosine_like(3, 50_000);
    let s: Vec<f64> = rows.iter().map(|r| r.0).collect();
    for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0, 0.0, f64::INFINITY] {
        let mut seq = EssAcc::new();
        for &x in &s {
            seq.absorb(x, eps);
        }
        let want = ess_ref(&s, eps);
        let got = seq.finalize().unwrap();
        assert!((got - want).abs() <= 1e-12 * want, "eps={eps}: {got} vs {want}");
        assert!(got >= 1.0 - 1e-12 && got <= s.len() as f64 * (1.0 + 1e-12));
        for parts in [2usize, 7, 64, 1000] {
            let mut tot = EssAcc::new();
            for ch in s.chunks(s.len().div_ceil(parts)) {
                let mut p = EssAcc::new();
                for &x in ch {
                    p.absorb(x, eps);
                }
                tot.merge(&p, eps);
            }
            let m = tot.finalize().unwrap();
            assert!((m - got).abs() <= 1e-12 * got, "eps={eps} parts={parts}: {m} vs {got}");
        }
        // reversed order (every row raises the anchor at first)
        let mut rev = EssAcc::new();
        for &x in s.iter().rev() {
            rev.absorb(x, eps);
        }
        assert!((rev.finalize().unwrap() - got).abs() <= 1e-12 * got);
    }
}

#[test]
fn ess_infinities_and_empty() {
    let mut a = EssAcc::new();
    assert_eq!(a.finalize(), None);
    a.absorb(f64::NEG_INFINITY, 0.1);
    assert_eq!(a.finalize(), None);
    a.absorb(0.3, 0.1);
    a.absorb(0.3, 0.1);
    assert!((a.finalize().unwrap() - 2.0).abs() < 1e-15);
    a.absorb(f64::INFINITY, 0.1);
    assert_eq!(a.finalize(), Some(1.0));
    a.absorb(f64::INFINITY, 0.1);
    assert_eq!(a.finalize(), Some(2.0));
}

#[test]
fn segment_tree_matches_direct_exact_fold() {
    use bound_exp::exact::SegTree;
    let rows = cosine_like(11, 3000);
    for &eps in &[1e-4, 0.02, 1.0] {
        let leaves: Vec<Option<(f64, f64)>> = rows.iter().map(|&r| Some(r)).collect();
        let mut t = SegTree::build(&leaves, eps);
        for &(l, r) in &[(0usize, 3000usize), (17, 67), (1000, 2999), (5, 6)] {
            let a = t.query(l, r).value().unwrap();
            let b = exact_fold(&rows[l..r], eps).unwrap();
            assert!(a.sub(b).to_f64().abs() <= 1e-26 * b.hi.abs(), "eps={eps} [{l},{r})");
        }
        for i in (0..3000).step_by(3) {
            t.set(i, None);
        }
        let live: Vec<(f64, f64)> = rows.iter().enumerate().filter(|(i, _)| i % 3 != 0).map(|x| *x.1).collect();
        let a = t.all().value().unwrap();
        let b = exact_fold(&live, eps).unwrap();
        assert!(a.sub(b).to_f64().abs() <= 1e-26 * b.hi.abs());
    }
}

/// Audit: the shipped bound charges exp as 1 ulp but not the rounding of
/// its argument fl(fl(s - mu) / eps), which perturbs a weight by up to
/// ~|x| U relatively (|x| up to ~745). When the delivered value is itself
/// tiny (dominant row has v = 0), that exceeds the bound. The excess is
/// ~1e-300 in absolute terms, far below the refold tolerance, so it never
/// matters for `is_exact()`; it matters only for the claim "error <=
/// value_error_bound()". DdInv forms the argument in double-double.
#[test]
fn shipped_bound_omits_exponent_argument_rounding() {
    let eps = 1e-3;
    let mut worst: f64 = 0.0;
    let mut found = 0;
    for k in 0..200 {
        let s2 = 0.7 - 0.69 - 1e-6 * k as f64; // x ~ -690
        let rows = [(0.7, 0.0), (s2 + 0.0, 10.0)];
        let mut a = InvAcc::new();
        let mut d = DdInv::new(ABS_TOL, REFOLD_GAIN);
        for &(s, v) in &rows {
            a.absorb(s, v, eps);
            d.absorb(s, v, eps);
        }
        let ex = exact_fold(&rows, eps).unwrap();
        let e = ((a.finalize().unwrap() - ex.hi) - ex.lo).abs();
        if e > a.value_error_bound() {
            found += 1;
            worst = worst.max(e / a.value_error_bound());
        }
        let ed = ((d.finalize().unwrap() - ex.hi) - ex.lo).abs();
        assert!(ed <= d.value_error_bound(), "DdInv k={k}");
        assert!(a.is_exact(), "the gap never reaches the tolerance");
    }
    eprintln!("argument-rounding gap: {found}/200 two-row states exceed the shipped bound, worst x{worst:.1}");
    assert!(found > 0);
}

/// The proposed patch (charge 2U|x| w per weight, 2U|x_s| scale |z0| per
/// rescale) closes the gap above and stays sound on sliding windows.
#[test]
fn argument_term_patch_closes_the_gap() {
    let eps = 1e-3;
    for k in 0..200 {
        let s2 = 0.7 - 0.69 - 1e-6 * k as f64;
        let rows = [(0.7, 0.0), (s2, 10.0)];
        let mut a = Local::<false>::new(ABS_TOL, REFOLD_GAIN);
        a.arg_term = true;
        for &(s, v) in &rows {
            a.absorb(s, v, eps);
        }
        let ex = exact_fold(&rows, eps).unwrap();
        let e = ((a.finalize().unwrap() - ex.hi) - ex.lo).abs();
        assert!(e <= a.value_error_bound(), "k={k}: {e:e} > {:e}", a.value_error_bound());
    }
    let rows = cosine_like(5, 4000);
    for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0] {
        let w = 200;
        let mut a = Local::<false>::new(ABS_TOL, REFOLD_GAIN);
        a.arg_term = true;
        for i in 0..rows.len() {
            a.absorb(rows[i].0, rows[i].1, eps);
            if i >= w {
                a.remove(rows[i - w].0, rows[i - w].1, eps);
            }
            let frame = &rows[(i + 1).saturating_sub(w)..=i];
            if !a.is_exact() {
                a.reset();
                for &(s, v) in frame {
                    a.absorb(s, v, eps);
                }
            }
            let ex = exact_fold(frame, eps).unwrap();
            let e = ((a.finalize().unwrap() - ex.hi) - ex.lo).abs();
            assert!(e <= a.value_error_bound(), "eps={eps} i={i}");
        }
    }
}

/// Underflow gap: dominant row v = 0 and a second row whose weight
/// e^{-746} underflows to exactly 0, so the shipped state returns 0 with
/// bound 0, while the exact value 8000 e^{x} / (1 + e^{x}) ~ 1.6e-320 is a
/// representable positive number (computed here as exp(x + ln 8000) in
/// double-double, which never forms the unrepresentable weight). The
/// patch charges 2^-1074 per underflowed weight / product.
#[test]
fn underflow_gap_and_patch() {
    use bound_exp::dd::{exp_dd, DD};
    let eps = 1e-4;
    let (s1, s2, v2) = (0.5, 0.5 - 0.0746, 8000.0);
    let rows = [(s1, 0.0), (s2, v2)];
    let x = DD::diff(s2, s1).div_f(eps);
    let exact = exp_dd(x.add(DD::from(f64::ln(v2)))).to_f64(); // denominator 1 + e^x = 1 in f64
    assert!(exact > 0.0);
    let mut a = InvAcc::new();
    let mut p = Local::<false>::new(ABS_TOL, REFOLD_GAIN);
    p.arg_term = true;
    for &(s, v) in &rows {
        a.absorb(s, v, eps);
        p.absorb(s, v, eps);
    }
    eprintln!("underflow: shipped {:e} (bound {:e}), exact {exact:e}, patched bound {:e}",
              a.finalize().unwrap(), a.value_error_bound(), p.value_error_bound());
    assert_eq!(a.finalize(), Some(0.0));
    assert!((a.finalize().unwrap() - exact).abs() > a.value_error_bound());
    assert!((p.finalize().unwrap() - exact).abs() <= p.value_error_bound());
}

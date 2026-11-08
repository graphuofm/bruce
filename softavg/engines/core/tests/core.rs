use softagg_core::{InvAcc, ScalarAcc};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn rows(seed: u64, n: usize) -> Vec<(f64, f64)> {
    let mut r = Lcg(seed);
    (0..n).map(|_| (2.0 * r.next() - 1.0, 10.0 * r.next())).collect()
}

/// Two-pass max-shifted reference.
fn reference(rows: &[(f64, f64)], eps: f64) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    if eps.is_infinite() {
        return Some(rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64);
    }
    let m = rows.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
    if eps == 0.0 {
        let set: Vec<f64> = rows.iter().filter(|r| r.0 == m).map(|r| r.1).collect();
        return Some(set.iter().sum::<f64>() / set.len() as f64);
    }
    let (mut num, mut den) = (0.0, 0.0);
    for &(s, v) in rows {
        let w = ((s - m) / eps).exp();
        num += w * v;
        den += w;
    }
    Some(num / den)
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    (a - b).abs() <= rtol * a.abs().max(b.abs()).max(1e-300)
}

const EPS: [f64; 7] = [1e-4, 1e-3, 0.02, 0.1, 1.0, 0.0, f64::INFINITY];

#[test]
fn absorb_matches_two_pass_reference() {
    for (k, &eps) in EPS.iter().enumerate() {
        let r = rows(k as u64 + 1, 5000);
        let mut a = ScalarAcc::new();
        for &(s, v) in &r {
            a.absorb(s, v, eps);
        }
        let want = reference(&r, eps).unwrap();
        assert!(close(a.finalize().unwrap(), want, 1e-12), "eps={eps}");
    }
}

#[test]
fn merge_over_partitions_equals_sequential() {
    for (k, &eps) in EPS.iter().enumerate() {
        let r = rows(100 + k as u64, 4000);
        let mut seq = ScalarAcc::new();
        for &(s, v) in &r {
            seq.absorb(s, v, eps);
        }
        for parts in [2usize, 7, 64] {
            let mut tot = ScalarAcc::new();
            for chunk in r.chunks(r.len().div_ceil(parts)) {
                let mut p = ScalarAcc::new();
                for &(s, v) in chunk {
                    p.absorb(s, v, eps);
                }
                tot.merge(&p, eps);
            }
            assert!(close(tot.finalize().unwrap(), seq.finalize().unwrap(), 1e-12),
                    "eps={eps} parts={parts}");
        }
    }
}

#[test]
fn infinities_and_empty() {
    let mut a = ScalarAcc::new();
    assert_eq!(a.finalize(), None);
    a.absorb(f64::NEG_INFINITY, 3.0, 0.1);
    assert_eq!(a.finalize(), None, "all -inf is empty (SQL NULL)");
    a.absorb(0.5, 1.0, 0.1);
    a.absorb(f64::INFINITY, 7.0, 0.1);
    a.absorb(f64::INFINITY, 9.0, 0.1);
    assert_eq!(a.finalize(), Some(8.0), "+inf rows dominate, uniform among themselves");
}

#[test]
fn naive_form_overflows_where_the_state_does_not() {
    // the motivating fact: exp(s/eps) overflows at sharp eps
    let eps = 1e-3;
    assert!((0.8f64 / eps).exp().is_infinite());
    let mut a = ScalarAcc::new();
    a.absorb(0.8, 1.0, eps);
    a.absorb(0.79, 3.0, eps);
    let want = (1.0 + 3.0 * ((0.79f64 - 0.8) / eps).exp()) / (1.0 + ((0.79f64 - 0.8) / eps).exp());
    assert!(close(a.finalize().unwrap(), want, 1e-12));
}

#[test]
fn random_insert_remove_streams_stay_exact_or_say_so() {
    for (k, &eps) in [0.02, 0.1, 1.0, f64::INFINITY].iter().enumerate() {
        let pool = rows(500 + k as u64, 400);
        let mut live = vec![false; pool.len()];
        let mut acc = InvAcc::new();
        let mut rng = Lcg(9 + k as u64);
        let mut flagged = 0;
        for step in 0..20_000 {
            let i = (rng.next() * pool.len() as f64) as usize;
            let (s, v) = pool[i];
            if live[i] {
                acc.remove(s, v, eps);
            } else {
                acc.absorb(s, v, eps);
            }
            live[i] = !live[i];
            let cur: Vec<(f64, f64)> = pool.iter().zip(&live).filter(|x| *x.1).map(|x| *x.0).collect();
            if acc.is_exact() {
                match (acc.finalize(), reference(&cur, eps)) {
                    (None, None) => {}
                    (Some(g), Some(w)) => assert!(close(g, w, 1e-9), "eps={eps} step={step}: {g} vs {w}"),
                    other => panic!("eps={eps} step={step}: {other:?}"),
                }
            } else {
                flagged += 1;
                // the host would refold; emulate it
                acc.reset();
                for &(s, v) in &cur {
                    acc.absorb(s, v, eps);
                }
            }
        }
        // random churn rarely cancels: the guard should be quiet here
        assert!(flagged < 50, "eps={eps}: {flagged} refolds on random churn");
    }
}

#[test]
fn adversarial_descending_deletes_are_caught_not_wrong() {
    // delete rows in descending score order at a sharp eps: the anchor
    // goes stale and the weight cancels; every value reported as exact
    // must still be right
    let eps = 1e-3;
    let mut r = rows(77, 3000);
    r.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut acc = InvAcc::new();
    for &(s, v) in &r {
        acc.absorb(s, v, eps);
    }
    let mut caught = false;
    for i in 0..r.len() - 1 {
        acc.remove(r[i].0, r[i].1, eps);
        let rest = &r[i + 1..];
        if acc.is_exact() {
            let g = acc.finalize().unwrap();
            let w = reference(rest, eps).unwrap();
            assert!(close(g, w, 1e-8), "after {} deletes: {g} vs {w}", i + 1);
        } else {
            caught = true;
            break;
        }
    }
    assert!(caught, "descending deletes at eps=1e-3 must trip the guard");
}

#[test]
fn removing_everything_returns_to_empty_and_exact() {
    let r = rows(5, 50);
    let mut acc = InvAcc::new();
    for &(s, v) in &r {
        acc.absorb(s, v, 0.1);
    }
    for &(s, v) in &r {
        acc.remove(s, v, 0.1);
    }
    assert!(acc.is_exact());
    assert_eq!(acc.finalize(), None);
    assert_eq!(acc.n, 0);
}

#[test]
fn inv_merge_matches_sequential() {
    let r = rows(11, 3000);
    for &eps in &[0.02, 0.1, f64::INFINITY] {
        let mut seq = InvAcc::new();
        for &(s, v) in &r {
            seq.absorb(s, v, eps);
        }
        let mut tot = InvAcc::new();
        for chunk in r.chunks(500) {
            let mut p = InvAcc::new();
            for &(s, v) in chunk {
                p.absorb(s, v, eps);
            }
            tot.merge(&p, eps);
        }
        assert_eq!(tot.n, seq.n);
        assert!(close(tot.finalize().unwrap(), seq.finalize().unwrap(), 1e-12));
    }
}

#[test]
fn c_layout_is_what_the_header_says() {
    assert_eq!(std::mem::size_of::<ScalarAcc>(), 24);
    assert_eq!(std::mem::size_of::<InvAcc>(), 64);
}

/// Exact reference for one frame in f64 via a fresh two-pass fold.
fn frame_ref(rows: &[(f64, f64)], eps: f64) -> f64 {
    reference(rows, eps).unwrap()
}

/// Sliding windows over clustered cosine-like scores, the shape that made
/// the old z-only guard drift to ~4e-9 on IMDb at eps = 0.02: every value
/// reported as exact must be within its own error bound, and within
/// ABS_TOL * v_scale of a fresh fold.
#[test]
fn sliding_windows_respect_the_error_bound() {
    use softagg_core::{ABS_TOL, U};
    let mut r = Lcg(4242);
    let n = 60_000;
    // scores: slowly drifting centre plus noise, so the frame max moves
    let rows: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let c = 0.5 + 0.3 * ((i as f64) / 900.0).sin();
            (c + 0.08 * (r.next() - 0.5), (1.0 + 9.0 * r.next() * 10.0).round() / 10.0)
        })
        .collect();
    for &eps in &[1e-3, 0.02, 0.1] {
        for &width in &[50usize, 400] {
            let mut acc = InvAcc::new();
            let (mut refolds, mut max_err, mut max_rel) = (0u64, 0.0f64, 0.0f64);
            for i in 0..n {
                acc.absorb(rows[i].0, rows[i].1, eps);
                if i >= width {
                    let (s, v) = rows[i - width];
                    acc.remove(s, v, eps);
                }
                let lo = i.saturating_sub(width - 1).max(if i >= width { i - width + 1 } else { 0 });
                let frame = &rows[lo..=i];
                if !acc.is_exact() {
                    refolds += 1;
                    acc.reset();
                    for &(s, v) in frame {
                        acc.absorb(s, v, eps);
                    }
                }
                let got = acc.finalize().unwrap();
                let want = frame_ref(frame, eps);
                let err = (got - want).abs();
                // soundness of the bound (first order, so allow the
                // reference's own rounding: a fresh fold of the frame)
                let slack = 4.0 * frame.len() as f64 * U * acc.v_scale;
                assert!(err <= acc.value_error_bound() + slack,
                        "eps={eps} w={width} i={i}: err {err:e} > bound {:e}", acc.value_error_bound());
                max_err = max_err.max(err / acc.v_scale);
                max_rel = max_rel.max(err / want.abs());
            }
            assert!(max_err <= ABS_TOL + 4.0 * width as f64 * U * 100.0,
                    "eps={eps} w={width}: max err {max_err:e} of v_scale");
            eprintln!("eps={eps} width={width}: refolds={refolds} max_err/v_scale={max_err:.2e} max_rel={max_rel:.2e}");
        }
    }
}

#[test]
fn large_groups_are_not_refolded_on_every_delete() {
    // IVM-style: one big group, random deletes and inserts; the bound must
    // not force a refold just because n is large
    let r = rows(31, 200_000);
    let mut acc = InvAcc::new();
    for &(s, v) in &r {
        acc.absorb(s, v, 0.1);
    }
    assert!(acc.is_exact(), "a fresh fold of 200k rows must count as exact");
    let mut g = Lcg(8);
    for _ in 0..20_000 {
        let i = (g.next() * r.len() as f64) as usize;
        acc.remove(r[i].0, r[i].1, 0.1);
        acc.absorb(r[i].0, r[i].1, 0.1);
    }
    assert!(acc.is_exact(), "random churn on a large group should not trip the bound");
}

/// B1 (2026-11-06): the bound must cover the rounding of the exp argument
/// x = (s - mu)/eps (relative ~2U|x| on the weight). Two rows, the dominant
/// one with v = 0, so the delivered value is the tiny second weight itself
/// and the argument rounding is the whole error. Exact weight:
/// exp(fl(x)) * (1 + r), r = (true x - fl(x)) recovered with an FMA.
#[test]
fn bound_covers_exp_argument_rounding() {
    let (mu, eps) = (0.875f64, 1e-3f64);
    let mut g = Lcg(2026);
    for _ in 0..2000 {
        // dyadic grid so that s - mu is exact and the only argument
        // rounding is the division by eps
        let t = ((0.4 + 0.3 * g.next()) * (1u64 << 30) as f64).round() / (1u64 << 30) as f64;
        let s = mu - t;
        let mut acc = InvAcc::new();
        acc.absorb(mu, 0.0, eps);
        acc.absorb(s, 1.0, eps);
        let d = s - mu;
        assert_eq!(d + mu, s, "test assumes s - mu is exact");
        let x = d / eps;
        let r = (-x).mul_add(eps, d) / eps; // true x - fl(x), to first order
        let w_true = x.exp() * (1.0 + r); // exp(fl x) itself is within 1 ulp: covered by the bound's exp term
        let f_true = w_true / (1.0 + w_true);
        let got = acc.finalize().unwrap();
        let err = (got - f_true).abs();
        // allow the 1-ulp exp error of the reference itself
        assert!(err <= acc.value_error_bound() + 2.0 * softagg_core::U * f_true,
                "s={s}: err {err:e} > bound {:e}", acc.value_error_bound());
    }
}

/// B1: a weight that underflows is off by up to 2^-1074; the state must not
/// report a zero bound for a value that is not exactly representable.
#[test]
fn bound_covers_underflow() {
    let eps = 1e-3;
    let mut acc = InvAcc::new();
    acc.absorb(0.0, 0.0, eps);
    acc.absorb(-0.744, 1.0, eps); // w = e^-744 is subnormal
    assert!(acc.acc.u > 0.0 && acc.acc.u < f64::MIN_POSITIVE);
    assert!(acc.value_error_bound() >= 4.9e-324, "bound {:e}", acc.value_error_bound());
    // and removal of a subnormal-weight row is charged the same way
    acc.remove(-0.744, 1.0, eps);
    assert!(acc.value_error_bound() >= 4.9e-324);
}

//! Property-based tests for `bruce_core::mask::masked_attention`.
//!
//! These encode the two theorems the evaluator implements:
//! order-invariance of the fold (the structure lemma: the per-row
//! accumulator is a commutative-monoid homomorphism) and agreement
//! with a direct per-row softmax reference, across all three
//! temperature regimes (tropical, finite, uniform).

use bruce_core::mask::masked_attention;
use bruce_core::types::Eps;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

/// Strategy: a small random instance — dimensions, data, a
/// duplicate-free pair set, and a temperature regime selector.
fn instance() -> impl Strategy<
    Value = (
        usize,               // n_q
        usize,               // n_k
        usize,               // d_k
        usize,               // d_v
        Vec<f64>,            // q data
        Vec<f64>,            // k data
        Vec<f64>,            // v data
        Vec<(usize, usize)>, // pairs (deduplicated)
        u8,                  // regime: 0 -> eps=0, 1 -> finite, 2 -> inf
        f64,                 // finite eps in (0.05, 4)
    ),
> {
    (2usize..8, 2usize..8, 1usize..4, 1usize..4).prop_flat_map(|(n_q, n_k, d_k, d_v)| {
        let qlen = n_q * d_k;
        let klen = n_k * d_k;
        let vlen = n_k * d_v;
        (
            Just(n_q),
            Just(n_k),
            Just(d_k),
            Just(d_v),
            proptest::collection::vec(-3.0f64..3.0, qlen..=qlen),
            proptest::collection::vec(-3.0f64..3.0, klen..=klen),
            proptest::collection::vec(-5.0f64..5.0, vlen..=vlen),
            proptest::collection::btree_set((0..n_q, 0..n_k), 1..=n_q * n_k)
                .prop_map(|s| s.into_iter().collect::<Vec<_>>()),
            0u8..3,
            0.05f64..4.0,
        )
    })
}

fn build(
    n_q: usize,
    n_k: usize,
    d_k: usize,
    d_v: usize,
    q: &[f64],
    k: &[f64],
    v: &[f64],
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    (
        Array2::from_shape_vec((n_q, d_k), q.to_vec()).unwrap(),
        Array2::from_shape_vec((n_k, d_k), k.to_vec()).unwrap(),
        Array2::from_shape_vec((n_k, d_v), v.to_vec()).unwrap(),
    )
}

fn pick_eps(regime: u8, fin: f64) -> Eps {
    match regime {
        0 => Eps::ZERO,
        1 => Eps(fin),
        _ => Eps::INF,
    }
}

/// Direct per-row reference: gather masked records, softmax, combine.
fn reference(
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    pairs: &[(usize, usize)],
    eps: Eps,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((q.nrows(), v.ncols()));
    for i in 0..q.nrows() {
        let js: Vec<usize> = pairs.iter().filter(|p| p.0 == i).map(|p| p.1).collect();
        if js.is_empty() {
            continue;
        }
        let scores: Vec<f64> = js.iter().map(|&j| q.row(i).dot(&k.row(j))).collect();
        let weights: Vec<f64> = if eps.is_inf() {
            vec![1.0 / js.len() as f64; js.len()]
        } else {
            bruce_core::semiring::softmax_eps(&scores, eps)
        };
        let mut row = Array1::<f64>::zeros(v.ncols());
        for (idx, &j) in js.iter().enumerate() {
            row.scaled_add(weights[idx], &v.row(j));
        }
        out.row_mut(i).assign(&row);
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Any permutation of the pair stream yields the same output.
    #[test]
    fn order_invariance((n_q, n_k, d_k, d_v, qd, kd, vd, pairs, regime, fin) in instance()) {
        let (q, k, v) = build(n_q, n_k, d_k, d_v, &qd, &kd, &vd);
        let eps = pick_eps(regime, fin);
        let (a, cov_a) = masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();
        let mut rev = pairs.clone();
        rev.reverse();
        // a second, interleaved order
        let mut inter: Vec<_> = pairs.iter().copied().step_by(2).collect();
        inter.extend(pairs.iter().copied().skip(1).step_by(2));
        for other in [rev, inter] {
            let (b, cov_b) =
                masked_attention(&q.view(), &k.view(), &v.view(), &other, eps).unwrap();
            prop_assert_eq!(&cov_a, &cov_b);
            for (x, y) in a.iter().zip(b.iter()) {
                prop_assert!((x - y).abs() <= 1e-9, "order changed result: {} vs {}", x, y);
            }
        }
    }

    /// The fold agrees with the direct per-row reference in all regimes.
    #[test]
    fn matches_reference((n_q, n_k, d_k, d_v, qd, kd, vd, pairs, regime, fin) in instance()) {
        let (q, k, v) = build(n_q, n_k, d_k, d_v, &qd, &kd, &vd);
        let eps = pick_eps(regime, fin);
        let (out, covered) =
            masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();
        let want = reference(&q, &k, &v, &pairs, eps);
        for i in 0..n_q {
            let has = pairs.iter().any(|p| p.0 == i);
            prop_assert_eq!(covered[i], has);
            for c in 0..d_v {
                prop_assert!(
                    (out[(i, c)] - want[(i, c)]).abs() <= 1e-9,
                    "row {} col {}: {} vs {}", i, c, out[(i, c)], want[(i, c)]
                );
            }
        }
    }
}

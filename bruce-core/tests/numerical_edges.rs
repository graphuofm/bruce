//! TRACK 1 / workstreams 2 + 4 — numerical edge semantics and the
//! f32-vs-f64 precision contract for the grouped soft-average kernels.
//!
//! This suite PINS the following semantics (PG-aligned per C4; they
//! were previously undefined — NaN scores poisoned the finite-eps
//! accumulator, +/-Inf scores produced NaN via `exp(inf - inf)`):
//!
//! | input                       | semantics                                    |
//! |-----------------------------|----------------------------------------------|
//! | NaN score or NaN value      | row SKIPPED in every regime (SQL NULL:       |
//! |                             | ingest.rs encodes NULL as NaN; PG's          |
//! |                             | two-argument aggregates skip the row if      |
//! |                             | either argument is NULL, cf. corr/covar)     |
//! | +Inf score, finite eps      | argmax semantics: +inf rows dominate, ties   |
//! |                             | among them average uniformly (as at eps = 0) |
//! | -Inf score, eps = 0/finite  | weight 0, skipped (Indicator "no match");    |
//! |                             | an all--inf group is uncovered = SQL empty   |
//! |                             | equi-join result -> NULL                     |
//! | any score, eps = inf        | score not consulted (plain mean = AVG); only |
//! |                             | the NaN (NULL) skip applies                  |
//! | eps = 1e-300                | behaves as the tropical limit (tie-free      |
//! |                             | inputs: exactly the eps = 0 answer)          |
//! | eps = 1e300                 | behaves as the uniform limit (bit-equal to   |
//! |                             | eps = inf: every weight rounds to 1.0)       |
//! | empty selection / no rows   | group uncovered, zero row (SQL NULL)         |
//! | single row                  | output == its value row, bit-exact, any eps  |
//! | all-equal scores            | == plain mean at ANY finite eps (weights all |
//! |                             | exp(0) = 1: bit-equal to the eps = inf run)  |
//! | subnormal / underflowing    | == max-anchored truth, finite, no NaN        |
//! |   tail weights              |                                              |
//!
//! Workstream 4 (precision contract) lives at the bottom: table-driven
//! per-eps ceilings on the f32 kernel's relative error vs the f64
//! kernel on identical stored numbers.

use bruce_core::mask::{grouped_softavg, grouped_softavg_f32};
use bruce_core::types::Eps;
use ndarray::{Array1, Array2};

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

/// Run one of the two kernels with scores injected via a d_k = 1 key
/// column and x = [1] (score(r) == scores[r]; NaN/Inf pass through the
/// 1-element dot product unchanged in both dtypes).
#[allow(clippy::too_many_arguments)] // mirrors the kernel signature + dtype switch
fn run(
    f32_kernel: bool,
    scores: &[f64],
    values: &[f64],
    d_v: usize,
    gid: &[u32],
    n_groups: usize,
    sel: Option<&[bool]>,
    eps: Eps,
) -> (Array2<f64>, Vec<bool>) {
    let n = scores.len();
    let v = Array2::from_shape_vec((n, d_v), values.to_vec()).unwrap();
    if f32_kernel {
        let x = Array1::from_vec(vec![1.0f32]);
        let k = Array2::from_shape_vec((n, 1), scores.iter().map(|&s| s as f32).collect()).unwrap();
        grouped_softavg_f32(&x.view(), &k.view(), &v.view(), gid, n_groups, sel, eps).unwrap()
    } else {
        let x = Array1::from_vec(vec![1.0f64]);
        let k = Array2::from_shape_vec((n, 1), scores.to_vec()).unwrap();
        grouped_softavg(&x.view(), &k.view(), &v.view(), gid, n_groups, sel, eps).unwrap()
    }
}

fn single_group(n: usize) -> Vec<u32> {
    vec![0; n]
}

/// The regimes every semantics test must hold in.
const REGIMES: [Eps; 3] = [Eps::ZERO, Eps(0.37), Eps::INF];

// ---------------------------------------------------------------- (a)

/// A NaN score is the engine's encoding of a SQL NULL weight argument
/// (bruce-query/src/ingest.rs maps Parquet NULL -> NaN): the row is
/// skipped, in every regime, in both kernels.
#[test]
fn nan_score_rows_are_skipped_in_every_regime() {
    let with_nan = [1.0, NAN, 2.0, 0.5];
    let without = [1.0, 2.0, 0.5];
    let v_with = [10.0, 777.0, 20.0, 30.0];
    let v_without = [10.0, 20.0, 30.0];
    for f32k in [false, true] {
        for eps in REGIMES {
            let (a, ca) = run(f32k, &with_nan, &v_with, 1, &single_group(4), 1, None, eps);
            let (b, cb) = run(
                f32k,
                &without,
                &v_without,
                1,
                &single_group(3),
                1,
                None,
                eps,
            );
            assert_eq!(ca, cb, "f32={f32k} eps={eps:?}");
            assert!(a[(0, 0)].is_finite(), "f32={f32k} eps={eps:?}: NaN leaked");
            assert!(
                (a[(0, 0)] - b[(0, 0)]).abs() <= 1e-15,
                "f32={f32k} eps={eps:?}: {} vs {}",
                a[(0, 0)],
                b[(0, 0)]
            );
        }
    }
}

/// A group whose every row has a NaN score is uncovered (SQL: the
/// aggregate saw only NULLs -> NULL), and does not disturb siblings.
#[test]
fn all_nan_score_group_is_uncovered() {
    let scores = [NAN, NAN, 3.0];
    let values = [1.0, 2.0, 42.0];
    let gid = [0u32, 0, 1];
    for f32k in [false, true] {
        for eps in REGIMES {
            let (out, cov) = run(f32k, &scores, &values, 1, &gid, 2, None, eps);
            assert_eq!(cov, vec![false, true], "f32={f32k} eps={eps:?}");
            assert_eq!(out[(0, 0)], 0.0);
            assert_eq!(out[(1, 0)], 42.0);
        }
    }
}

// ---------------------------------------------------------------- (c)

/// A NaN in ANY component of the value row skips the whole row (the
/// value is one vector datum; a NULL datum skips the row — same rule
/// as AVG over a NULL input).
#[test]
fn nan_value_rows_are_skipped_whole_row() {
    let scores = [1.0, 1.5, 2.0];
    let d_v = 2;
    let values = [10.0, 11.0, 5.0, NAN, 20.0, 21.0]; // row 1 tainted
    let scores_ref = [1.0, 2.0];
    let values_ref = [10.0, 11.0, 20.0, 21.0];
    for f32k in [false, true] {
        for eps in REGIMES {
            let (a, ca) = run(f32k, &scores, &values, d_v, &single_group(3), 1, None, eps);
            let (b, cb) = run(
                f32k,
                &scores_ref,
                &values_ref,
                d_v,
                &single_group(2),
                1,
                None,
                eps,
            );
            assert_eq!(ca, cb);
            for c in 0..d_v {
                assert!(a[(0, c)].is_finite());
                assert!(
                    (a[(0, c)] - b[(0, c)]).abs() <= 1e-15,
                    "f32={f32k} eps={eps:?} col {c}: {} vs {}",
                    a[(0, c)],
                    b[(0, c)]
                );
            }
        }
    }
}

/// All value rows NaN -> uncovered group.
#[test]
fn all_nan_value_group_is_uncovered() {
    let scores = [1.0, 2.0];
    let values = [NAN, NAN];
    for f32k in [false, true] {
        for eps in REGIMES {
            let (out, cov) = run(f32k, &scores, &values, 1, &single_group(2), 1, None, eps);
            assert_eq!(cov, vec![false], "f32={f32k} eps={eps:?}");
            assert_eq!(out[(0, 0)], 0.0);
        }
    }
}

// ---------------------------------------------------------------- (b)

/// +Inf score at finite eps: that row's weight dominates (argmax
/// semantics); ties among +inf rows average uniformly, mirroring the
/// eps = 0 tie rule. At eps = 0 the +inf row simply IS the argmax.
#[test]
fn pos_inf_score_dominates_at_finite_and_zero_eps() {
    for f32k in [false, true] {
        for eps in [Eps::ZERO, Eps(0.37), Eps::ONE] {
            let (out, cov) = run(
                f32k,
                &[1.0, INF, 2.0],
                &[10.0, 55.0, 20.0],
                1,
                &single_group(3),
                1,
                None,
                eps,
            );
            assert_eq!(cov, vec![true]);
            assert_eq!(out[(0, 0)], 55.0, "f32={f32k} eps={eps:?}");
            // two +inf rows tie -> uniform mean of their values
            let (out2, _) = run(
                f32k,
                &[INF, 3.0, INF],
                &[50.0, 999.0, 60.0],
                1,
                &single_group(3),
                1,
                None,
                eps,
            );
            assert_eq!(out2[(0, 0)], 55.0, "f32={f32k} eps={eps:?} tie");
        }
    }
}

/// At eps = inf the score is not consulted (plain mean = AVG): +/-inf
/// scored rows count like any other row.
#[test]
fn eps_inf_is_score_blind_even_for_infinite_scores() {
    for f32k in [false, true] {
        let (out, cov) = run(
            f32k,
            &[INF, -INF, 1.0],
            &[3.0, 6.0, 9.0],
            1,
            &single_group(3),
            1,
            None,
            Eps::INF,
        );
        assert_eq!(cov, vec![true]);
        assert_eq!(out[(0, 0)], 6.0, "f32={f32k}");
    }
}

/// +Inf rows landing in DIFFERENT rayon chunks must survive the
/// accumulator merge: the merged answer is the uniform mean over all
/// +inf rows (exercises the finite-eps +inf branch of RowAcc::merge).
#[test]
fn pos_inf_rows_in_separate_parallel_chunks_merge_to_argmax_mean() {
    let chunk = 8192usize;
    let n = 5 * chunk; // above the 2^15 parallel threshold
    let mut scores: Vec<f64> = (0..n)
        .map(|r| ((r * 37 + 11) % 100) as f64 / 25.0)
        .collect();
    let mut values: Vec<f64> = (0..n).map(|r| (r % 50) as f64 + 1.0).collect();
    scores[chunk + 5] = INF; // inside chunk #2
    values[chunk + 5] = 70.0;
    scores[4 * chunk + 9] = INF; // inside chunk #5
    values[4 * chunk + 9] = 30.0;
    let gid = single_group(n);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build()
        .unwrap();
    for eps in [Eps(0.37), Eps::ONE] {
        let (out, cov) = pool.install(|| run(false, &scores, &values, 1, &gid, 1, None, eps));
        assert_eq!(cov, vec![true]);
        assert_eq!(out[(0, 0)], 50.0, "eps={eps:?}");
    }
}

/// -Inf score carries weight 0 at eps = 0 and finite eps: skipped.
/// A group of only -inf rows is uncovered — the Indicator similarity
/// encodes "no equi-join match" as -inf, and SQL returns NULL for an
/// aggregate over an empty match set. (This matches the tropical path
/// of IncrementalMemory in crud.rs, which also drops non-finite
/// scores.)
#[test]
fn neg_inf_score_is_weight_zero_and_all_neg_inf_group_uncovered() {
    for f32k in [false, true] {
        for eps in [Eps::ZERO, Eps(0.37), Eps::ONE] {
            let (out, cov) = run(
                f32k,
                &[-INF, 1.0],
                &[888.0, 10.0],
                1,
                &single_group(2),
                1,
                None,
                eps,
            );
            assert_eq!(cov, vec![true]);
            assert_eq!(out[(0, 0)], 10.0, "f32={f32k} eps={eps:?}");

            let (out2, cov2) = run(
                f32k,
                &[-INF, -INF],
                &[888.0, 999.0],
                1,
                &single_group(2),
                1,
                None,
                eps,
            );
            assert_eq!(cov2, vec![false], "f32={f32k} eps={eps:?}");
            assert_eq!(out2[(0, 0)], 0.0);
        }
    }
}

// ---------------------------------------------------------------- (d)

/// eps = 1e-300 must not overflow/NaN: with a tie-free score set it
/// gives exactly the tropical (eps = 0) answer — every non-max weight
/// underflows to exp(-inf) = 0 and the max row keeps weight 1.
#[test]
fn eps_1e_minus_300_equals_tropical_answer() {
    let scores = [0.5, 3.0, -2.0, 1.5];
    let values = [10.0, 20.0, 30.0, 40.0];
    let gid = [0u32, 0, 1, 1];
    for f32k in [false, true] {
        let (tiny, cov_t) = run(f32k, &scores, &values, 1, &gid, 2, None, Eps(1e-300));
        let (trop, cov_z) = run(f32k, &scores, &values, 1, &gid, 2, None, Eps::ZERO);
        assert_eq!(cov_t, cov_z);
        for g in 0..2 {
            assert!(tiny[(g, 0)].is_finite());
            assert_eq!(tiny[(g, 0)], trop[(g, 0)], "f32={f32k} group {g}");
        }
    }
}

/// eps = 1e300 must not collapse to garbage: (s - mu)/1e300 rounds to
/// a value whose exp() is exactly 1.0 for O(1) scores, so the run is
/// bit-equal to the eps = inf (plain mean) run.
#[test]
fn eps_1e300_equals_uniform_mean_bit_exact() {
    let scores = [0.5, 3.0, -2.0, 1.5, 2.5];
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    let gid = [0u32, 0, 1, 1, 0];
    for f32k in [false, true] {
        let (huge, cov_h) = run(f32k, &scores, &values, 1, &gid, 2, None, Eps(1e300));
        let (unif, cov_i) = run(f32k, &scores, &values, 1, &gid, 2, None, Eps::INF);
        assert_eq!(cov_h, cov_i);
        for g in 0..2 {
            assert!(huge[(g, 0)].is_finite());
            assert_eq!(huge[(g, 0)], unif[(g, 0)], "f32={f32k} group {g}");
        }
    }
}

// ---------------------------------------------------------------- (e)

/// All-false selection and zero-row input both yield uncovered groups
/// with zero rows (SQL: aggregate over the empty set -> NULL).
#[test]
fn empty_selection_and_empty_input_are_uncovered() {
    for f32k in [false, true] {
        for eps in REGIMES {
            let sel = vec![false; 3];
            let (out, cov) = run(
                f32k,
                &[1.0, 2.0, 3.0],
                &[10.0, 20.0, 30.0],
                1,
                &[0u32, 1, 0],
                2,
                Some(&sel),
                eps,
            );
            assert_eq!(cov, vec![false, false], "f32={f32k} eps={eps:?}");
            assert_eq!(out[(0, 0)], 0.0);
            assert_eq!(out[(1, 0)], 0.0);

            let (out0, cov0) = run(f32k, &[], &[], 1, &[], 2, None, eps);
            assert_eq!(cov0, vec![false, false]);
            assert_eq!(out0[(0, 0)], 0.0);
            assert_eq!(out0[(1, 0)], 0.0);
        }
    }
}

// ---------------------------------------------------------------- (f)

/// A single absorbed row finalizes to u/z = (1*v)/1: the value row,
/// bit-exact, in every regime including the eps extremes.
#[test]
fn single_row_output_is_its_value_bit_exact() {
    let v = [std::f64::consts::PI, -std::f64::consts::E];
    for f32k in [false, true] {
        for eps in [
            Eps::ZERO,
            Eps(1e-300),
            Eps(0.37),
            Eps::ONE,
            Eps(1e300),
            Eps::INF,
        ] {
            let (out, cov) = run(f32k, &[0.75], &v, 2, &[0u32], 1, None, eps);
            assert_eq!(cov, vec![true], "f32={f32k} eps={eps:?}");
            assert_eq!(out[(0, 0)], v[0]);
            assert_eq!(out[(0, 1)], v[1]);
        }
    }
}

// ---------------------------------------------------------------- (g)

/// All-equal scores: every weight is exp(0) = 1.0 exactly, so ANY
/// finite eps computes the identical arithmetic to the eps = inf
/// plain-mean path — asserted bit-exact (sequential path, n below the
/// parallel threshold).
#[test]
fn all_equal_scores_equal_plain_mean_at_any_finite_eps() {
    let n = 100;
    let scores = vec![1.25f64; n];
    let values: Vec<f64> = (0..n).map(|r| (r as f64).sin() + 2.0).collect();
    let gid: Vec<u32> = (0..n).map(|r| (r % 4) as u32).collect();
    for f32k in [false, true] {
        let (unif, _) = run(f32k, &scores, &values, 1, &gid, 4, None, Eps::INF);
        for eps in [Eps(1e-300), Eps(0.37), Eps::ONE, Eps(1e300)] {
            let (out, cov) = run(f32k, &scores, &values, 1, &gid, 4, None, eps);
            assert_eq!(cov, vec![true; 4]);
            for g in 0..4 {
                assert_eq!(
                    out[(g, 0)],
                    unif[(g, 0)],
                    "f32={f32k} eps={eps:?} group {g}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------- (h)

/// Subnormal-weight tail: one anchor row at score 0 plus tail rows
/// whose shifted weights exp(-700)..exp(-745) underflow to subnormals
/// (and one at exp(-800) = 0 exactly). The answer must equal the
/// max-anchored truth — the anchor's value, since the tail's total
/// contribution is ~1e-298 * 1e6, below one ulp — and contain no NaN.
#[test]
fn subnormal_weight_tail_matches_anchored_truth() {
    let anchor = 3.375; // exactly representable; not a math constant
    let scores = [0.0, -700.0, -720.0, -745.0, -800.0];
    let values = [anchor, 1e6, 1e6, 1e6, 1e9];
    for f32k in [false, true] {
        // f32 note: -745 and -800 are exactly representable; the f32
        // kernel widens the score to f64 before the exp, so the same
        // anchoring applies.
        let (out, cov) = run(
            f32k,
            &scores,
            &values,
            1,
            &single_group(5),
            1,
            None,
            Eps::ONE,
        );
        assert_eq!(cov, vec![true]);
        assert!(out[(0, 0)].is_finite(), "f32={f32k}: NaN/Inf leaked");
        let rel = (out[(0, 0)] - anchor).abs() / anchor;
        assert!(
            rel <= 1e-12,
            "f32={f32k}: {} vs anchored truth {anchor}",
            out[(0, 0)]
        );
    }
}

// ------------------------------------------------- workstream 4 ----

/// Deterministic pseudo-random matrix (xorshift64*, no rand dep).
fn pseudo(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut state = seed | 1;
    Array2::from_shape_fn((n, d), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    })
}

/// Precision contract (workstream 4): per-eps ceilings on the f32
/// kernel's max relative error vs the f64 kernel on the SAME stored
/// numbers (keys generated in f32, upcast for the f64 run). 10k rows,
/// 16 groups, real d_k = 16 dot products, values offset away from 0.
/// (The masked_attention NaN-policy mod lives at the bottom of this
/// file, separate from the grouped-kernel pins above.)
///
/// Ceilings are generous — the only difference between the kernels is
/// the f32 dot product (score error ~1e-7 abs), amplified by exp(d/eps)
/// on the weights — but tight enough that a precision regression
/// (e.g. accumulating in f32, or losing the max-anchor) fails loudly.
#[test]
fn precision_contract_f32_vs_f64_per_eps_ceilings() {
    let n = 10_000;
    let d_k = 16;
    let n_groups = 16;
    let k32 = pseudo(n, d_k, 0xC0FFEE).mapv(|x| x as f32);
    let k64 = k32.mapv(|x| x as f64); // identical stored numbers
    let v = pseudo(n, 1, 0xBEEF).mapv(|x| x + 2.0); // values in [1.5, 2.5]
    let x32 = pseudo(1, d_k, 0xF00D).row(0).mapv(|x| x as f32);
    let x64 = x32.mapv(|x| x as f64);
    let gid: Vec<u32> = (0..n).map(|r| ((r * 31 + 7) % n_groups) as u32).collect();

    let table: [(f64, f64); 4] = [(1e-4, 1e-3), (1e-2, 1e-4), (0.1, 1e-5), (1.0, 1e-5)];
    for (eps_val, ceiling) in table {
        let eps = Eps(eps_val);
        let (want, cov64) = grouped_softavg(
            &x64.view(),
            &k64.view(),
            &v.view(),
            &gid,
            n_groups,
            None,
            eps,
        )
        .unwrap();
        let (got, cov32) = grouped_softavg_f32(
            &x32.view(),
            &k32.view(),
            &v.view(),
            &gid,
            n_groups,
            None,
            eps,
        )
        .unwrap();
        assert_eq!(cov64, cov32, "coverage differs at eps={eps_val}");
        let mut max_rel = 0.0f64;
        for g in 0..n_groups {
            let rel = (got[(g, 0)] - want[(g, 0)]).abs() / want[(g, 0)].abs();
            max_rel = max_rel.max(rel);
        }
        println!(
            "precision contract: eps={eps_val:>6}  max_rel={max_rel:.3e}  ceiling={ceiling:e}"
        );
        assert!(
            max_rel < ceiling,
            "eps={eps_val}: f32-vs-f64 max rel err {max_rel:e} exceeds ceiling {ceiling:e}"
        );
    }
}

// =================================================================
// masked_attention NaN policy (2026-08-03, f32-tail track).
//
// Residual gap closed: the raw pair-stream surface used to NaN-poison
// its accumulator on NaN scores while the grouped kernels skipped the
// row. Policy now PINNED (same NULL discipline; NaN is the engine's
// encoding of SQL NULL):
//
//   * a pair (i, j) is SKIPPED, in every eps regime, when its score
//     q_i . k_j is NaN (NaN anywhere in q_i or k_j makes the dot NaN)
//     OR any component of value row v_j is NaN;
//   * a query row whose every pair is skipped is UNCOVERED
//     (covered[i] = false, zero output row) — SQL NULL;
//   * +/-Inf scores are NOT NULLs and keep the argmax / weight-0
//     semantics pinned above.
//
// bruce-pg's ScalarAcc cross-checks mirror this policy (C2).
// =================================================================
mod masked_attention_nan_policy {
    use bruce_core::mask::masked_attention;
    use bruce_core::types::Eps;
    use ndarray::Array2;

    const NAN: f64 = f64::NAN;
    const REGIMES: [Eps; 4] = [Eps::ZERO, Eps(0.37), Eps::ONE, Eps::INF];

    fn m(rows: usize, cols: usize, data: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap()
    }

    /// A NaN key row NaN-poisons its scores: pairs referencing it are
    /// skipped, and the row's answer equals the run without the pair.
    #[test]
    fn nan_scored_pairs_are_skipped_in_every_regime() {
        let q = m(1, 2, &[1.0, 0.5]);
        let k = m(3, 2, &[1.0, 0.0, NAN, 1.0, 0.0, 2.0]); // key row 1 tainted
        let v = m(3, 1, &[10.0, 777.0, 30.0]);
        let k_clean = m(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let v_clean = m(2, 1, &[10.0, 30.0]);
        for eps in REGIMES {
            let (a, ca) = masked_attention(
                &q.view(),
                &k.view(),
                &v.view(),
                &[(0, 0), (0, 1), (0, 2)],
                eps,
            )
            .unwrap();
            let (b, cb) = masked_attention(
                &q.view(),
                &k_clean.view(),
                &v_clean.view(),
                &[(0, 0), (0, 1)],
                eps,
            )
            .unwrap();
            assert_eq!(ca, cb, "eps={eps:?}");
            assert!(a[(0, 0)].is_finite(), "eps={eps:?}: NaN leaked");
            assert!(
                (a[(0, 0)] - b[(0, 0)]).abs() <= 1e-15,
                "eps={eps:?}: {} vs {}",
                a[(0, 0)],
                b[(0, 0)]
            );
        }
    }

    /// A NaN in ANY component of a value row skips every pair that
    /// references it (the value is one vector datum — same rule as the
    /// grouped kernels and PG's AVG over a NULL input).
    #[test]
    fn nan_value_row_skips_its_pairs() {
        let q = m(1, 1, &[1.0]);
        let k = m(3, 1, &[1.0, 5.0, 2.0]);
        let v = m(3, 2, &[10.0, 11.0, 5.0, NAN, 30.0, 31.0]); // value row 1 tainted
        let k_clean = m(2, 1, &[1.0, 2.0]);
        let v_clean = m(2, 2, &[10.0, 11.0, 30.0, 31.0]);
        for eps in REGIMES {
            let (a, ca) = masked_attention(
                &q.view(),
                &k.view(),
                &v.view(),
                &[(0, 0), (0, 1), (0, 2)],
                eps,
            )
            .unwrap();
            let (b, cb) = masked_attention(
                &q.view(),
                &k_clean.view(),
                &v_clean.view(),
                &[(0, 0), (0, 1)],
                eps,
            )
            .unwrap();
            assert_eq!(ca, cb, "eps={eps:?}");
            for c in 0..2 {
                assert!(a[(0, c)].is_finite(), "eps={eps:?} col {c}: NaN leaked");
                assert!(
                    (a[(0, c)] - b[(0, c)]).abs() <= 1e-15,
                    "eps={eps:?} col {c}: {} vs {}",
                    a[(0, c)],
                    b[(0, c)]
                );
            }
        }
    }

    /// A query row whose every pair is skipped (here: the q row itself
    /// is NaN, so every score is NaN) is uncovered — SQL NULL — and
    /// does not disturb sibling rows.
    #[test]
    fn all_pairs_skipped_row_is_uncovered() {
        let q = m(2, 1, &[NAN, 1.0]);
        let k = m(2, 1, &[1.0, 2.0]);
        let v = m(2, 1, &[10.0, 42.0]);
        for eps in REGIMES {
            let (out, cov) = masked_attention(
                &q.view(),
                &k.view(),
                &v.view(),
                &[(0, 0), (0, 1), (1, 1)],
                eps,
            )
            .unwrap();
            assert_eq!(cov, vec![false, true], "eps={eps:?}");
            assert_eq!(out[(0, 0)], 0.0, "eps={eps:?}: uncovered row must be zero");
            assert_eq!(out[(1, 0)], 42.0, "eps={eps:?}");
        }
    }

    /// The strongest pin: on NaN-laced data, masked_attention routed
    /// through an explicit (group, row) pair stream must agree with
    /// grouped_softavg exactly — one NULL discipline across both
    /// surfaces (and, via bruce-pg's cross-checks, the PG aggregate).
    #[test]
    fn agrees_with_grouped_softavg_on_nan_laced_data() {
        use bruce_core::mask::grouped_softavg;
        use ndarray::Array1;
        let n = 200;
        let n_groups = 5;
        let d_k = 3;
        let mut state = 0xD1CEu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut k = Array2::from_shape_fn((n, d_k), |_| next());
        let mut v = Array2::from_shape_fn((n, 1), |_| next() + 2.0);
        for r in (0..n).step_by(17) {
            k[(r, r % d_k)] = NAN; // NaN score rows
        }
        for r in (0..n).step_by(23) {
            v[(r, 0)] = NAN; // NaN value rows
        }
        let x = Array1::from_shape_fn(d_k, |_| next());
        let gid: Vec<u32> = (0..n).map(|r| ((r * 31 + 7) % n_groups) as u32).collect();
        let mut q = Array2::<f64>::zeros((n_groups, d_k));
        for g in 0..n_groups {
            for c in 0..d_k {
                q[(g, c)] = x[c];
            }
        }
        let pairs: Vec<(usize, usize)> = (0..n).map(|r| (gid[r] as usize, r)).collect();
        for eps in REGIMES {
            let (want, want_cov) =
                grouped_softavg(&x.view(), &k.view(), &v.view(), &gid, n_groups, None, eps)
                    .unwrap();
            let (got, got_cov) =
                masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();
            assert_eq!(want_cov, got_cov, "eps={eps:?}");
            for g in 0..n_groups {
                assert!(
                    (want[(g, 0)] - got[(g, 0)]).abs() <= 1e-12,
                    "eps={eps:?} group {g}: {} vs {}",
                    want[(g, 0)],
                    got[(g, 0)]
                );
            }
        }
    }
}

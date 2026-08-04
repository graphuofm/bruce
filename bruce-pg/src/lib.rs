//! bruce_pg — the `(mu, z, u)` soft-average monoid as a native
//! PostgreSQL aggregate (constitution C2: the monoid maps 1:1 onto
//! PG's aggregate contract).
//!
//! ```text
//!     RowAcc::absorb   ->  SFUNC        bruce_softavg_sfunc
//!     RowAcc::merge    ->  COMBINEFUNC  bruce_softavg_combine
//!     RowAcc::finalize ->  FINALFUNC    bruce_softavg_final
//! ```
//!
//! `softavg(value, score, eps)` over a row set S computes
//! `sum_{r in S} w_r * value_r / sum w_r` with `w_r = e^{score_r/eps}`
//! in the max-anchored representation, so it never materialises
//! `e^{score/eps}` — the naive SQL spelling overflows float8 at sharp
//! eps (see `test_c_*`), the anchored monoid does not.
//!
//! Temperature semantics (same three regimes as bruce-core):
//! - `eps > 0` finite: max-shifted softmax average, `u/z`.
//! - `eps = 0`: tropical limit — mean of values over the argmax set.
//! - `eps = 'infinity'`: plain mean (every weight 1), i.e. `AVG`.
//!
//! SQL state = `float8[5] = [mu, z, u, eps, nan_seen]`; the empty
//! accumulator is SQL NULL. A varlena array (not `internal`) so workers'
//! partial states cross process boundaries with no serialfunc /
//! deserialfunc, and `softavg_state` can hand the raw monoid element
//! back to SQL. `eps` rides inside the state because COMBINEFUNC only
//! receives two states, and merge needs the temperature.
//!
//! Special float values (full rationale in README.md, "Special float
//! values"): `±Inf` mirrors bruce-core's `RowAcc` verbatim, because
//! there the policy lives *inside* the monoid. NaN deliberately does
//! **not**: bruce-core skips NaN at its *call sites* (NaN is that
//! engine's encoding of SQL NULL — there is no null bitmap in an
//! `ndarray<f64>`), whereas PG has a real NULL and `'NaN'::float8` is
//! a real value that `AVG`/`SUM` propagate. So PG's call site — this
//! SFUNC — makes the PG-native choice and propagates NaN. The monoid
//! itself is untouched: the state is the product monoid
//! `(mu, z, u) × ({false, true}, ∨)`, whose first factor still equals
//! bruce-core's skip answer exactly.

use pgrx::prelude::*;

::pgrx::pg_module_magic!(name, version);

/// Scalar-value (`d_v = 1`) accumulator in the max-shifted
/// representation. Mirror of the file-private `RowAcc` in
/// bruce-core/src/mask.rs — same absorb/merge/finalize recurrences,
/// pinned against `bruce_core::masked_attention` by the
/// `bruce_core_cross_check` tests below. Pure Rust, no pgrx calls:
/// callable from non-Postgres tests.
///
/// Invariant for finite `eps > 0` after absorbing a set S:
/// `u = e^{-mu/eps} sum_{r in S} e^{s_r/eps} v_r`,
/// `z = e^{-mu/eps} sum_{r in S} e^{s_r/eps}`, `mu = max_{r in S} s_r`.
/// For `eps = 0`: `z` = argmax multiplicity, `u` = value-sum over the
/// argmax set. For `eps = inf`: `z` = count, `u` = plain value-sum.
/// Constraint: one accumulator sees exactly one `eps`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarAcc {
    pub mu: f64,
    pub z: f64,
    pub u: f64,
}

impl ScalarAcc {
    pub fn new() -> Self {
        Self {
            mu: f64::NEG_INFINITY,
            z: 0.0,
            u: 0.0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.z == 0.0
    }

    /// Absorb one `(score, value)` record. Mirrors `RowAcc::absorb`,
    /// including its score-infinity policy:
    /// * `-inf` weighs 0 at `eps = 0` and finite `eps` — skipped
    ///   outright (the Indicator "no match" encoding), so an
    ///   all-`-inf` group stays empty and finalizes to SQL NULL.
    /// * `+inf` at finite `eps` dominates every finite score: the
    ///   accumulator collapses to argmax semantics over the `+inf`
    ///   rows, with uniform tie handling exactly as at `eps = 0`.
    ///   `exp(inf - inf)` is never evaluated.
    /// * At `eps = inf` the score is not consulted (plain mean), so
    ///   `±inf` rows count like any other row.
    ///
    /// NaN never reaches `absorb`: `bruce_softavg_sfunc` diverts a NaN
    /// score or value into the state's sticky NaN bit instead (see the
    /// module docs and README.md), which is precisely why this factor
    /// of the state stays bit-identical to bruce-core's.
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if eps == 0.0 {
            if s == f64::NEG_INFINITY {
                // Indicator "no match": weight 0
                return;
            }
            // tropical: keep only the argmax set
            if s > self.mu {
                self.mu = s;
                self.z = 1.0;
                self.u = v;
            } else if s == self.mu {
                self.z += 1.0;
                self.u += v;
            }
            return;
        }
        if eps.is_infinite() {
            // uniform: plain count + sum
            self.z += 1.0;
            self.u += v;
            return;
        }
        // finite eps > 0
        if s == f64::NEG_INFINITY {
            // exp(-inf / eps) = 0: contributes nothing
            return;
        }
        if s == f64::INFINITY || self.mu == f64::INFINITY {
            // argmax collapse: only +inf-scored rows retain weight
            if s == f64::INFINITY && self.mu == f64::INFINITY {
                // tie among +inf rows: uniform, as at eps = 0
                self.z += 1.0;
                self.u += v;
            } else if s == f64::INFINITY {
                // first +inf row dominates the finite prefix
                self.mu = f64::INFINITY;
                self.z = 1.0;
                self.u = v;
            }
            // else: finite s under a +inf anchor: weight exp(-inf) = 0
            return;
        }
        if self.is_empty() {
            self.mu = s;
            self.z = 1.0;
            self.u = v;
            return;
        }
        let mu2 = self.mu.max(s);
        let scale = ((self.mu - mu2) / eps).exp();
        let w = ((s - mu2) / eps).exp();
        self.u = self.u * scale + w * v;
        self.z = self.z * scale + w;
        self.mu = mu2;
    }

    /// Merge another accumulator — the partition-reduce identity:
    /// disjoint row sets combine by re-basing both sides to the common
    /// maximum and adding. Mirrors `RowAcc::merge`.
    pub fn merge(&mut self, other: &ScalarAcc, eps: f64) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = *other;
            return;
        }
        if eps == 0.0 {
            if other.mu > self.mu {
                *self = *other;
            } else if other.mu == self.mu {
                self.z += other.z;
                self.u += other.u;
            }
            return;
        }
        if eps.is_infinite() {
            self.z += other.z;
            self.u += other.u;
            return;
        }
        // finite eps with a +inf anchor on either side (see `absorb`'s
        // policy): the +inf side(s) dominate; `exp(inf - inf)` must
        // never be evaluated. Mirrors `RowAcc::merge`.
        if self.mu == f64::INFINITY || other.mu == f64::INFINITY {
            if self.mu == f64::INFINITY && other.mu == f64::INFINITY {
                self.z += other.z;
                self.u += other.u;
            } else if other.mu == f64::INFINITY {
                *self = *other;
            }
            // else: self is the +inf side; other's finite rows weigh 0
            return;
        }
        let mu2 = self.mu.max(other.mu);
        let s1 = ((self.mu - mu2) / eps).exp();
        let s2 = ((other.mu - mu2) / eps).exp();
        self.u = self.u * s1 + other.u * s2;
        self.z = self.z * s1 + other.z * s2;
        self.mu = mu2;
    }

    /// `u / z` in every regime; `None` if nothing was absorbed.
    /// Mirrors `RowAcc::finalize`.
    pub fn finalize(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.u / self.z)
    }
}

impl Default for ScalarAcc {
    fn default() -> Self {
        Self::new()
    }
}

/// SQL state layout: `float8[5] = [mu, z, u, eps, nan_seen]`.
///
/// Slots 1..3 are the bruce-core monoid element; slot 5 is the second
/// factor of the product monoid `(mu, z, u) × ({false, true}, ∨)` — a
/// sticky bit (0.0 / 1.0) recording that some qualifying row carried a
/// NaN `value` or `score`. Only `FINALFUNC` consults it. Grew from
/// `float8[4]` in 0.1.1; see README.md.
const STATE_LEN: usize = 5;

/// The PostgreSQL-side accumulator: bruce-core's monoid in a **product**
/// with the sticky-NaN monoid, `(mu, z, u) × ({false, true}, ∨)`.
///
/// This type, not `ScalarAcc`, is what the three support functions
/// implement — and the second factor is the whole of bruce-pg's
/// deliberate NaN divergence from bruce-core (README.md, "Special float
/// values"). Keeping it here rather than inside `ScalarAcc` is the
/// point: the bruce-core factor is left *exactly* as bruce-core wrote
/// it, so `self.acc` always holds precisely the state bruce-core's
/// call-site skip would have produced.
///
/// Pure Rust, no pgrx calls, so the cross-check tests can drive the real
/// transition logic without a Postgres backend. (They must: `cargo pgrx
/// test` relies on `--gc-sections` pruning pgrx-pg-sys's unreachable
/// `extern "C"` declarations out of the test binary, so a plain
/// `#[test]` that reaches a pgrx `error!()` fails to link.)
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PgAcc {
    pub acc: ScalarAcc,
    /// Some qualifying row carried a NaN `value` or `score`.
    pub nan_seen: bool,
}

impl PgAcc {
    pub fn new() -> Self {
        Self {
            acc: ScalarAcc::new(),
            nan_seen: false,
        }
    }

    /// Absorb one qualifying `(score, value)` row. A NaN in either
    /// component sets the sticky bit and is kept *out* of the monoid
    /// factor — letting NaN into `mu` would corrupt the `±Inf` branches
    /// and silently skew the max comparisons (`f64::max` ignores NaN).
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if s.is_nan() || v.is_nan() {
            self.nan_seen = true;
            return;
        }
        self.acc.absorb(s, v, eps);
    }

    /// Componentwise merge: `ScalarAcc::merge` on the monoid factor, OR
    /// on the sticky bit — so a NaN seen by only one parallel worker
    /// still reaches the final result.
    pub fn merge(&mut self, other: &PgAcc, eps: f64) {
        self.acc.merge(&other.acc, eps);
        self.nan_seen |= other.nan_seen;
    }

    /// NaN if the sticky bit is set (PG value semantics, and a group of
    /// nothing *but* NaN rows is therefore NaN and not NULL); otherwise
    /// `u / z`, or `None` — SQL NULL — when the monoid factor is empty
    /// (no qualifying rows, or every row scored `-Inf`).
    pub fn finalize(&self) -> Option<f64> {
        if self.nan_seen {
            return Some(f64::NAN);
        }
        self.acc.finalize()
    }
}

/// Pure half of [`decode`]: `None` when the array is not a well-formed
/// state. Split out so the plain-Rust tests can exercise the encoding
/// without making a pgrx `error!` reachable from the test binary (see
/// [`PgAcc`] for why that matters).
fn decode_checked(st: &[f64]) -> Option<(PgAcc, f64)> {
    if st.len() != STATE_LEN {
        return None;
    }
    Some((
        PgAcc {
            acc: ScalarAcc {
                mu: st[0],
                z: st[1],
                u: st[2],
            },
            nan_seen: st[4] != 0.0,
        },
        st[3],
    ))
}

fn decode(st: &[f64]) -> (PgAcc, f64) {
    decode_checked(st).unwrap_or_else(|| {
        error!(
            "softavg: malformed state — expected float8[{}], got {} elements",
            STATE_LEN,
            st.len()
        )
    })
}

fn encode(st: &PgAcc, eps: f64) -> Vec<f64> {
    vec![
        st.acc.mu,
        st.acc.z,
        st.acc.u,
        eps,
        if st.nan_seen { 1.0 } else { 0.0 },
    ]
}

fn check_eps(eps: f64) -> f64 {
    // same domain as bruce-core's Eps::new: nonnegative, not NaN;
    // 0 and infinity are legal sentinels
    if eps < 0.0 || eps.is_nan() {
        error!("softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got {eps}");
    }
    eps
}

/// Aggregate transition (SFUNC). Non-strict: NULL state is the empty
/// accumulator; rows with NULL value or NULL score are ignored,
/// matching AVG's treatment of NULL inputs.
///
/// A row whose `value` or `score` is `'NaN'::float8` is a *qualifying*
/// row (NaN is a value in PG, not a NULL): it does not enter the
/// monoid — letting NaN into `mu` would corrupt the `±Inf` branches
/// and the max comparisons — but it sets the state's sticky NaN bit,
/// so the group finalizes to NaN the way `AVG`/`SUM` would. This is
/// the one deliberate divergence from bruce-core, which skips NaN
/// because there NaN *is* its NULL encoding. See README.md.
#[pg_extern(immutable, parallel_safe)]
fn bruce_softavg_sfunc(
    state: Option<Vec<f64>>,
    value: Option<f64>,
    score: Option<f64>,
    eps: Option<f64>,
) -> Option<Vec<f64>> {
    let (Some(v), Some(s)) = (value, score) else {
        return state;
    };
    // NaN rows are live rows, so the eps contract applies to them too
    // (unlike the NULL rows skipped above).
    let Some(eps) = eps else {
        error!("softavg: eps must not be NULL");
    };
    let eps = check_eps(eps);
    let mut acc = match state {
        None => PgAcc::new(),
        Some(st) => {
            let (acc, eps0) = decode(&st);
            // eps is carried per-row by the aggregate signature but is a
            // property of the whole group: reject mid-group changes
            // instead of silently mixing temperatures
            if eps0 != eps {
                error!("softavg: eps must be constant within an aggregate group ({eps0} vs {eps})");
            }
            acc
        }
    };
    acc.absorb(s, v, eps);
    Some(encode(&acc, eps))
}

/// Partition combine (COMBINEFUNC) — the monoid merge. PG feeds it the
/// partial states of parallel workers; the state is a plain float8[]
/// varlena, so no serialfunc/deserialfunc is needed.
#[pg_extern(immutable, parallel_safe)]
fn bruce_softavg_combine(a: Option<Vec<f64>>, b: Option<Vec<f64>>) -> Option<Vec<f64>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(x), Some(y)) => {
            let (mut acc_a, eps_a) = decode(&x);
            let (acc_b, eps_b) = decode(&y);
            if eps_a != eps_b {
                error!("softavg: cannot combine states with different eps ({eps_a} vs {eps_b})");
            }
            acc_a.merge(&acc_b, eps_a);
            Some(encode(&acc_a, eps_a))
        }
    }
}

/// FINALFUNC: `u / z`. NULL state (no qualifying rows) -> NULL,
/// matching AVG over an empty group.
///
/// Two non-NULL states do not finalize to a number:
/// * the sticky NaN bit is set -> `'NaN'::float8` (PG value semantics,
///   see README.md), whatever the monoid factor holds;
/// * every qualifying row scored `-Inf` (weight 0) -> the monoid
///   factor is empty -> SQL NULL, matching bruce-core's uncovered
///   all-`-inf` group.
#[pg_extern(immutable, parallel_safe)]
fn bruce_softavg_final(state: Option<Vec<f64>>) -> Option<f64> {
    decode(&state?).0.finalize()
}

extension_sql!(
    r#"
-- softavg(value, score, eps): max-anchored softmax average.
-- The three support functions are exactly the (mu, z, u) monoid:
-- SFUNC = absorb, COMBINEFUNC = merge, FINALFUNC = u/z.
CREATE AGGREGATE softavg(float8, float8, float8) (
    SFUNC       = bruce_softavg_sfunc,
    STYPE       = float8[],
    COMBINEFUNC = bruce_softavg_combine,
    FINALFUNC   = bruce_softavg_final,
    PARALLEL    = SAFE
);

-- softavg_state: same transition, no finalfunc — returns the raw
-- monoid element [mu, z, u, eps, nan_seen] so SQL can re-associate
-- explicitly (see test_e_partition_combine_identity).
CREATE AGGREGATE softavg_state(float8, float8, float8) (
    SFUNC       = bruce_softavg_sfunc,
    STYPE       = float8[],
    COMBINEFUNC = bruce_softavg_combine,
    PARALLEL    = SAFE
);
"#,
    name = "softavg_aggregates",
    requires = [
        bruce_softavg_sfunc,
        bruce_softavg_combine,
        bruce_softavg_final
    ],
);

// ---------------------------------------------------------------------
// Cross-checks against bruce-core (plain Rust tests, no Postgres):
// pin ScalarAcc to the reference RowAcc semantics through the public
// masked_attention API (q = [[1.0]], k = scores, so score_j = s_j).
// ---------------------------------------------------------------------
#[cfg(test)]
mod bruce_core_cross_check {
    use super::{PgAcc, ScalarAcc};
    use bruce_core::{masked_attention, Eps};
    use ndarray::Array2;

    /// Deterministic fixture with duplicate scores (exercises the
    /// eps = 0 tie path) and both signs.
    fn fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
        let scores: Vec<f64> = (0..n)
            .map(|i| ((i * 7919 % 97) as f64) / 48.5 - 1.0)
            .collect();
        let values: Vec<f64> = (0..n)
            .map(|i| ((i * 104729 % 1009) as f64) / 100.9 - 5.0)
            .collect();
        (scores, values)
    }

    /// bruce-core's answer, or `None` when it reports the row
    /// uncovered (SQL NULL — e.g. an all-`-inf` group).
    fn reference_opt(scores: &[f64], values: &[f64], eps: Eps) -> Option<f64> {
        let n = scores.len();
        let q = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let k = Array2::from_shape_vec((n, 1), scores.to_vec()).unwrap();
        let v = Array2::from_shape_vec((n, 1), values.to_vec()).unwrap();
        let pairs: Vec<(usize, usize)> = (0..n).map(|j| (0, j)).collect();
        let (out, covered) =
            masked_attention(&q.view(), &k.view(), &v.view(), &pairs, eps).unwrap();
        covered[0].then(|| out[(0, 0)])
    }

    fn reference(scores: &[f64], values: &[f64], eps: Eps) -> f64 {
        reference_opt(scores, values, eps).expect("reference row uncovered")
    }

    /// Fold a fixture through `ScalarAcc` alone (no PG call site).
    fn fold(scores: &[f64], values: &[f64], eps: Eps) -> ScalarAcc {
        let mut acc = ScalarAcc::new();
        for (&s, &v) in scores.iter().zip(values.iter()) {
            acc.absorb(s, v, eps.0);
        }
        acc
    }

    #[test]
    fn sequential_fold_matches_bruce_core() {
        let (scores, values) = fixture(257);
        for eps in [Eps::ZERO, Eps(1e-4), Eps(0.3), Eps::ONE, Eps::INF] {
            let mut acc = ScalarAcc::new();
            for (&s, &v) in scores.iter().zip(values.iter()) {
                acc.absorb(s, v, eps.0);
            }
            let got = acc.finalize().unwrap();
            let want = reference(&scores, &values, eps);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "eps={:?}: got {got}, want {want}",
                eps
            );
        }
    }

    #[test]
    fn partitioned_merge_matches_bruce_core() {
        let (scores, values) = fixture(257);
        for eps in [Eps::ZERO, Eps(1e-4), Eps(0.3), Eps::ONE, Eps::INF] {
            for n_parts in [2, 4, 7] {
                let chunk = scores.len().div_ceil(n_parts);
                let mut total = ScalarAcc::new();
                for part in scores.chunks(chunk).zip(values.chunks(chunk)) {
                    let mut acc = ScalarAcc::new();
                    for (&s, &v) in part.0.iter().zip(part.1.iter()) {
                        acc.absorb(s, v, eps.0);
                    }
                    total.merge(&acc, eps.0);
                }
                let got = total.finalize().unwrap();
                let want = reference(&scores, &values, eps);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "eps={:?} parts={n_parts}: got {got}, want {want}",
                    eps
                );
            }
        }
    }

    #[test]
    fn empty_and_identity_element() {
        // empty acc is the monoid identity on both sides of merge
        for eps in [0.0, 0.5, f64::INFINITY] {
            let mut acc = ScalarAcc::new();
            assert_eq!(acc.finalize(), None);
            let mut filled = ScalarAcc::new();
            filled.absorb(0.3, 7.0, eps);
            acc.merge(&filled, eps);
            assert_eq!(acc.finalize(), Some(7.0));
            let mut left = filled;
            left.merge(&ScalarAcc::new(), eps);
            assert_eq!(left.finalize(), Some(7.0));
        }
    }

    /// ±Inf is part of the MONOID in bruce-core (`RowAcc` branches on
    /// it internally), so `ScalarAcc` mirrors it verbatim and the
    /// cross-check covers it directly — sequentially and under every
    /// partition, since the `+inf` collapse must survive `merge`.
    /// Before 0.1.2 every case here produced NaN via `exp(inf - inf)`.
    #[test]
    fn inf_scores_match_bruce_core_sequentially_and_under_partitions() {
        const INF: f64 = f64::INFINITY;
        let cases: [(&[f64], &[f64]); 5] = [
            // +inf dominates a finite prefix and a finite suffix
            (&[1.0, INF, 2.0], &[10.0, 55.0, 20.0]),
            // ties among +inf rows average uniformly, as at eps = 0
            (&[INF, 3.0, INF], &[50.0, 999.0, 60.0]),
            // -inf weighs 0
            (&[-INF, 1.0], &[888.0, 10.0]),
            // both infinities plus finite rows
            (&[-INF, 1.0, INF, 0.5, INF], &[888.0, 10.0, 4.0, 20.0, 6.0]),
            // +inf first, so the anchor is set before any finite row
            (&[INF, -INF, 0.25], &[7.0, 888.0, 20.0]),
        ];
        for eps in [Eps::ZERO, Eps(1e-4), Eps(0.37), Eps::ONE, Eps::INF] {
            for (scores, values) in cases {
                let want = reference(scores, values, eps);
                let got = fold(scores, values, eps).finalize().expect("empty acc");
                assert!(
                    got == want || (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "eps={eps:?} scores={scores:?}: got {got}, want {want}"
                );
                for n_parts in [2, 3, 5] {
                    let chunk = scores.len().div_ceil(n_parts);
                    let mut total = ScalarAcc::new();
                    for part in scores.chunks(chunk).zip(values.chunks(chunk)) {
                        total.merge(&fold(part.0, part.1, eps), eps.0);
                    }
                    let merged = total.finalize().expect("empty acc after merge");
                    assert!(
                        merged == want || (merged - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "eps={eps:?} parts={n_parts} scores={scores:?}: {merged} vs {want}"
                    );
                }
            }
        }
    }

    /// An all-`-inf` group is UNCOVERED in bruce-core (SQL NULL); the
    /// mirror must leave `ScalarAcc` empty so FINALFUNC returns NULL,
    /// not a number and not NaN.
    #[test]
    fn all_neg_inf_group_is_empty_like_bruce_cores_uncovered() {
        const NINF: f64 = f64::NEG_INFINITY;
        for eps in [Eps::ZERO, Eps(0.37), Eps::ONE] {
            assert_eq!(reference_opt(&[NINF, NINF], &[888.0, 999.0], eps), None);
            let acc = fold(&[NINF, NINF], &[888.0, 999.0], eps);
            assert!(acc.is_empty(), "eps={eps:?}: -inf rows must carry weight 0");
            assert_eq!(acc.finalize(), None, "eps={eps:?}");
        }
        // eps = inf is score-blind: the same rows are a plain mean.
        let acc = fold(&[NINF, NINF], &[888.0, 999.0], Eps::INF);
        assert_eq!(acc.finalize(), Some(943.5));
        assert_eq!(
            reference_opt(&[NINF, NINF], &[888.0, 999.0], Eps::INF),
            Some(943.5)
        );
    }

    /// THE DELIBERATE DIVERGENCE (decided in README.md, "Special float
    /// values"): bruce-core SKIPS NaN because NaN is its encoding of
    /// SQL NULL; bruce-pg PROPAGATES NaN because PostgreSQL has a real
    /// NULL and `'NaN'::float8` is a real value that AVG/SUM propagate.
    ///
    /// This test pins BOTH halves of that claim at once, which is
    /// exactly why the divergence is safe under C2:
    ///  * the MONOID factor of the PG state still equals bruce-core's
    ///    skip answer, bit for bit — the skip was never in the monoid,
    ///    it is a call-site rule on both sides;
    ///  * the PG call site (`PgAcc`, which the SFUNC/COMBINEFUNC/
    ///    FINALFUNC are thin encode/decode wrappers over) turns that
    ///    same state into NaN.
    #[test]
    fn nan_input_diverges_from_bruce_core_by_design() {
        const NAN: f64 = f64::NAN;
        let scores = [1.0, NAN, 2.0, 0.5, 3.0];
        let values = [10.0, 777.0, 20.0, NAN, 30.0];
        // what bruce-core sees after ITS call-site skip
        let clean_scores = [1.0, 2.0, 3.0];
        let clean_values = [10.0, 20.0, 30.0];
        for eps in [Eps::ZERO, Eps(1e-4), Eps(0.37), Eps::ONE, Eps::INF] {
            let want = reference(&clean_scores, &clean_values, eps);

            let mut pg = PgAcc::new();
            for (&s, &v) in scores.iter().zip(values.iter()) {
                pg.absorb(s, v, eps.0);
            }
            assert!(pg.nan_seen, "eps={eps:?}: sticky NaN bit must be set");

            // (1) the monoid factor is bruce-core's, untouched: the NaN
            // rows never entered it, so it equals the skip answer.
            let bruce = fold(&clean_scores, &clean_values, eps);
            assert_eq!(pg.acc, bruce, "eps={eps:?}: monoid factor diverged");
            let monoid = pg.acc.finalize().unwrap();
            assert!(
                (monoid - want).abs() <= 1e-12 * want.abs().max(1.0),
                "eps={eps:?}: monoid factor {monoid} != bruce-core {want}"
            );

            // (2) the PG call site turns that same state into NaN.
            let out = pg.finalize().expect("must not be NULL");
            assert!(
                out.is_nan(),
                "eps={eps:?}: PG call site must propagate NaN, got {out}"
            );
        }
    }

    /// The sticky bit is a monoid on its own (OR), so it survives
    /// COMBINEFUNC even when only ONE partial state ever saw a NaN —
    /// the parallel-plan hazard.
    #[test]
    fn nan_bit_survives_combine_from_either_side() {
        const NAN: f64 = f64::NAN;
        for eps in [0.0, 0.37, f64::INFINITY] {
            let mut clean = PgAcc::new();
            clean.absorb(0.5, 1.0, eps);
            let mut dirty = PgAcc::new();
            dirty.absorb(0.5, NAN, eps);
            for (a, b) in [(clean, dirty), (dirty, clean)] {
                let mut merged = a;
                merged.merge(&b, eps);
                assert!(merged.nan_seen, "eps={eps}: NaN bit lost across merge");
                assert!(merged.finalize().unwrap().is_nan(), "eps={eps}");
            }
            // clean + clean stays clean
            let mut merged = clean;
            merged.merge(&clean, eps);
            assert!(!merged.nan_seen, "eps={eps}: NaN bit invented from nothing");
            assert_eq!(merged.finalize(), Some(1.0), "eps={eps}");
        }
    }

    /// An all-NaN group is NaN, not NULL: unlike an all-NULL group,
    /// every row here QUALIFIED — matching `AVG` over an all-NaN
    /// column, which is NaN and not NULL. Contrast the all-`-Inf`
    /// group above, which really is NULL.
    #[test]
    fn all_nan_group_is_nan_not_null() {
        const NAN: f64 = f64::NAN;
        for eps in [0.0, 0.37, f64::INFINITY] {
            let mut pg = PgAcc::new();
            for _ in 0..3 {
                pg.absorb(NAN, NAN, eps);
            }
            assert!(
                pg.acc.is_empty(),
                "eps={eps}: monoid factor must stay empty (nothing absorbed)"
            );
            let out = pg.finalize();
            assert!(
                out.is_some_and(|x| x.is_nan()),
                "eps={eps}: want NaN, got {out:?}"
            );
        }
    }

    /// Round-trip through the SQL state encoding: both factors survive
    /// `encode`/`decode`, including `mu = ±inf` and the sticky bit.
    #[test]
    fn state_encoding_round_trips_both_factors() {
        const NAN: f64 = f64::NAN;
        for eps in [0.0, 0.37, f64::INFINITY] {
            for rows in [
                &[(1.0, 10.0), (f64::INFINITY, 5.0)][..],
                &[(f64::NEG_INFINITY, 3.0)][..],
                &[(0.5, NAN), (1.5, 2.0)][..],
            ] {
                let mut pg = PgAcc::new();
                for &(s, v) in rows {
                    pg.absorb(s, v, eps);
                }
                let enc = super::encode(&pg, eps);
                assert_eq!(enc.len(), super::STATE_LEN);
                let (back, eps_back) = super::decode_checked(&enc).expect("well-formed state");
                // a wrong-length array is rejected, not misread
                assert!(super::decode_checked(&enc[..4]).is_none());
                assert_eq!(back, pg, "eps={eps}: state did not round-trip");
                assert_eq!(eps_back, eps, "eps={eps}: temperature did not round-trip");
            }
        }
    }
}

// ---------------------------------------------------------------------
// SQL tests through the extension (cargo pgrx test pg17).
// ---------------------------------------------------------------------
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    fn get_f64(query: &str) -> f64 {
        Spi::get_one::<f64>(query)
            .expect("SPI failed")
            .expect("NULL result")
    }

    /// (a) all scores equal -> softavg == AVG at every temperature
    /// (eps = 0: the argmax set is the whole group).
    #[pg_test]
    fn test_a_equal_scores_match_avg() {
        Spi::run(
            "CREATE TABLE ta AS SELECT (i::float8) * 0.37 - 5.0 AS v \
             FROM generate_series(1, 100) g(i)",
        )
        .unwrap();
        let want = get_f64("SELECT AVG(v) FROM ta");
        for eps in ["0.0", "0.5", "1.0", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, 0.7, {eps}) FROM ta"));
            assert!(
                (got - want).abs() < 1e-12,
                "eps={eps}: softavg {got} != avg {want}"
            );
        }
    }

    /// (b) finite eps matches the naive SQL softmax average on a small
    /// fixture whose scores are tame enough not to overflow.
    #[pg_test]
    fn test_b_matches_sql_softmax_average() {
        Spi::run(
            "CREATE TABLE tb(v float8, s float8); \
             INSERT INTO tb VALUES (3.0, -2.0), (-1.5, -0.7), (0.25, 0.0), \
                                   (7.0, 0.4), (-4.0, 1.1), (2.5, 1.9), (9.0, -1.3)",
        )
        .unwrap();
        for (eps, ref_expr) in [
            ("1.0", "SUM(EXP(s) * v) / SUM(EXP(s))"),
            ("0.5", "SUM(EXP(s / 0.5) * v) / SUM(EXP(s / 0.5))"),
        ] {
            let want = get_f64(&format!("SELECT {ref_expr} FROM tb"));
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tb"));
            assert!(
                (got - want).abs() < 1e-9,
                "eps={eps}: softavg {got} != sql softmax {want}"
            );
        }
    }

    fn make_sharp_fixture() {
        // the +1.0-score row goes first so the naive spelling hits
        // EXP(10000) -> overflow on the very first row it evaluates
        Spi::run(
            "CREATE TABLE tc(v float8, s float8); \
             INSERT INTO tc VALUES (10.0, 1.0), (20.0, -1.0), (30.0, 0.5)",
        )
        .unwrap();
    }

    /// (c) naive half: SUM(EXP(s/eps)*v)/SUM(EXP(s/eps)) at eps = 1e-4
    /// with scores in [-1, 1] overflows float8 — EXP(1/1e-4) = e^10000.
    #[pg_test(error = "value out of range: overflow")]
    fn test_c_naive_softmax_overflows() {
        make_sharp_fixture();
        Spi::get_one::<f64>("SELECT SUM(EXP(s / 0.0001) * v) / SUM(EXP(s / 0.0001)) FROM tc")
            .unwrap();
    }

    /// (c) anchored half: same query through softavg is finite and
    /// correct — at eps = 1e-4 the group collapses onto its argmax row.
    #[pg_test]
    fn test_c_softavg_survives_sharp_eps() {
        make_sharp_fixture();
        let got = get_f64("SELECT softavg(v, s, 0.0001) FROM tc");
        assert!(got.is_finite(), "softavg overflowed: {got}");
        assert!((got - 10.0).abs() < 1e-12, "sharp-eps result {got} != 10.0");
    }

    /// (d) parallel-safety: force a parallel plan over 200k rows,
    /// verify the plan really gathers partial aggregates, and compare
    /// against the single-worker result.
    #[pg_test]
    fn test_d_parallel_matches_serial() {
        Spi::run(
            "CREATE TABLE td AS SELECT sin(i::float8) * 10.0 AS v, cos(i::float8) AS s \
             FROM generate_series(1, 200000) g(i)",
        )
        .unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
        let serial = get_f64("SELECT softavg(v, s, 0.25) FROM td");

        Spi::run("SET parallel_setup_cost = 0").unwrap();
        Spi::run("SET parallel_tuple_cost = 0").unwrap();
        Spi::run("SET min_parallel_table_scan_size = 0").unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 4").unwrap();
        let plan = explain("SELECT softavg(v, s, 0.25) FROM td");
        assert!(
            plan.contains("Gather") && plan.contains("Partial Aggregate"),
            "plan did not parallelize:\n{plan}"
        );
        let parallel = get_f64("SELECT softavg(v, s, 0.25) FROM td");
        assert!(
            (parallel - serial).abs() <= 1e-9 * serial.abs().max(1.0),
            "parallel {parallel} != serial {serial}"
        );
    }

    /// (e) the monoid identity in SQL: combine(state(A), state(B))
    /// finalized == softavg(A ∪ B), for an arbitrary partition.
    #[pg_test]
    fn test_e_partition_combine_identity() {
        Spi::run(
            "CREATE TABLE te AS SELECT i, sin(i::float8) * 3.0 AS v, cos(i::float8) AS s \
             FROM generate_series(1, 1000) g(i)",
        )
        .unwrap();
        let whole = get_f64("SELECT softavg(v, s, 0.25) FROM te");
        let recombined = get_f64(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
                 (SELECT softavg_state(v, s, 0.25) FROM te WHERE i % 2 = 0), \
                 (SELECT softavg_state(v, s, 0.25) FROM te WHERE i % 2 = 1)))",
        );
        assert!(
            (whole - recombined).abs() < 1e-12,
            "partition-combine {recombined} != whole {whole}"
        );
    }

    /// NULL discipline: NULL value/score rows are ignored like AVG's;
    /// an all-NULL (or empty) group yields NULL.
    #[pg_test]
    fn test_null_handling_matches_avg() {
        Spi::run(
            "CREATE TABLE tn(v float8, s float8); \
             INSERT INTO tn VALUES (1.0, 0.2), (NULL, 0.9), (3.0, NULL), (5.0, -0.1)",
        )
        .unwrap();
        let with_nulls = get_f64("SELECT softavg(v, s, 0.5) FROM tn");
        let without =
            get_f64("SELECT softavg(v, s, 0.5) FROM tn WHERE v IS NOT NULL AND s IS NOT NULL");
        assert!((with_nulls - without).abs() < 1e-15);
        let empty = Spi::get_one::<f64>("SELECT softavg(v, s, 0.5) FROM tn WHERE false").unwrap();
        assert!(empty.is_none(), "empty group must be NULL");
    }

    /// eps = 0 endpoint: mean over the argmax set (uniform ties).
    #[pg_test]
    fn test_eps_zero_argmax_mean() {
        Spi::run(
            "CREATE TABLE tz(v float8, s float8); \
             INSERT INTO tz VALUES (2.0, 1.0), (4.0, 1.0), (100.0, 0.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 0.0) FROM tz");
        assert!((got - 3.0).abs() < 1e-15, "tropical mean {got} != 3.0");
    }

    /// eps = infinity endpoint: plain mean, scores ignored.
    #[pg_test]
    fn test_eps_inf_plain_mean() {
        Spi::run(
            "CREATE TABLE ti(v float8, s float8); \
             INSERT INTO ti VALUES (1.0, 5.0), (2.0, -3.0), (6.0, 0.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 'infinity'::float8) FROM ti");
        assert!((got - 3.0).abs() < 1e-15, "uniform mean {got} != 3.0");
    }

    /// eps must be constant within a group (fixture rows carry
    /// e = 0.6, 0.7, 0.8 — the transition trips on the second row).
    #[pg_test(error = "softavg: eps must be constant within an aggregate group (0.6 vs 0.7)")]
    fn test_eps_change_rejected() {
        Spi::run(
            "CREATE TABLE tv AS SELECT i::float8 AS v, 0.0::float8 AS s, \
             (0.5 + i * 0.1)::float8 AS e FROM generate_series(1, 3) g(i)",
        )
        .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, e) FROM tv").unwrap();
    }

    // -----------------------------------------------------------------
    // Semantics conformance (TESTING_MATRIX workstream 14): softavg's
    // NULL / empty / strictness / grouping / eps-domain behavior pinned
    // against what PG's own AVG promises (constitution C4).
    // -----------------------------------------------------------------

    fn get_i64(query: &str) -> i64 {
        Spi::get_one::<i64>(query)
            .expect("SPI failed")
            .expect("NULL result")
    }

    /// (14a) NULL discipline == AVG's skip, proven against AVG itself:
    /// with equal scores softavg degenerates to a mean at every eps, so
    /// softavg over the NULL-y table must equal AVG over the subset
    /// where BOTH aggregate inputs are non-NULL. (AVG(v) alone is NOT
    /// the right-hand side: it skips NULL v but keeps rows whose score
    /// is NULL, which softavg must also drop.)
    #[pg_test]
    fn test_f_null_skip_equals_avg_on_nonnull_subset() {
        Spi::run(
            "CREATE TABLE tf(v float8, s float8); \
             INSERT INTO tf VALUES \
                 (1.0, 0.7), (NULL, 0.7), (3.0, NULL), (5.0, 0.7), \
                 (NULL, NULL), (9.0, 0.7)",
        )
        .unwrap();
        let want = get_f64("SELECT AVG(v) FROM tf WHERE v IS NOT NULL AND s IS NOT NULL");
        assert!((want - 5.0).abs() < 1e-15, "fixture: avg of {{1,5,9}} is 5");
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tf"));
            assert!(
                (got - want).abs() < 1e-12,
                "eps={eps}: softavg {got} != AVG-on-non-null-subset {want}"
            );
        }
    }

    /// (14b) empty input -> NULL (not 0, not an error), exactly AVG's
    /// contract — asserted for a WHERE-false scan, a truly empty table,
    /// and an all-NULL-input table, side by side with AVG. One
    /// deliberate asymmetry is pinned at the end: AVG's qualifying
    /// input is the single column `v`, softavg's is the complete
    /// `(v, s)` pair — a table holding a value but no score is "empty"
    /// to softavg even though AVG(v) still averages it.
    #[pg_test]
    fn test_f_empty_input_is_null_like_avg() {
        Spi::run(
            "CREATE TABLE tfe(v float8, s float8); \
             CREATE TABLE tfn(v float8, s float8); \
             INSERT INTO tfn VALUES (NULL, 0.5), (NULL, NULL)",
        )
        .unwrap();
        for from in ["tfe", "tfn", "tfe WHERE false"] {
            let avg = Spi::get_one::<f64>(&format!("SELECT AVG(v) FROM {from}")).unwrap();
            let soft =
                Spi::get_one::<f64>(&format!("SELECT softavg(v, s, 0.5) FROM {from}")).unwrap();
            assert!(avg.is_none(), "AVG over {from} must be NULL");
            assert!(
                soft.is_none(),
                "softavg over {from} must be NULL like AVG, got {soft:?}"
            );
        }
        // ...and it is one NULL row, not zero rows: aggregates over an
        // empty input still produce a single row.
        assert_eq!(
            get_i64("SELECT COUNT(*) FROM (SELECT softavg(v, s, 0.5) FROM tfe) q"),
            1
        );
        // the pair rule: value present, score NULL on every row ->
        // no qualifying (v, s) pair -> NULL, even though AVG(v) = 1.0.
        Spi::run("CREATE TABLE tfp(v float8, s float8); INSERT INTO tfp VALUES (1.0, NULL)")
            .unwrap();
        assert_eq!(get_f64("SELECT AVG(v) FROM tfp"), 1.0);
        let soft = Spi::get_one::<f64>("SELECT softavg(v, s, 0.5) FROM tfp").unwrap();
        assert!(
            soft.is_none(),
            "no complete (v, s) pair -> NULL, got {soft:?}"
        );
    }

    /// (14c) strictness declaration audit. The transition function MUST
    /// be declared non-strict: PG would reject a strict multi-argument
    /// SFUNC with a NULL initcond (no way to bootstrap the state), and a
    /// strict SFUNC would also skip rows where only `eps` is NULL
    /// instead of raising. So the declaration promises "callee handles
    /// NULLs" — the companion tests (test_f_*, test_null_eps_errors)
    /// prove the callee keeps that promise (skip NULL inputs, error on
    /// NULL eps). Combine/final are equally non-strict and NULL-total.
    #[pg_test]
    fn test_f_strictness_declaration_audit() {
        for agg in ["softavg", "softavg_state"] {
            let strict_sfunc = Spi::get_one::<bool>(&format!(
                "SELECT p.proisstrict FROM pg_aggregate a \
                 JOIN pg_proc p ON p.oid = a.aggtransfn \
                 WHERE a.aggfnoid = '{agg}(float8, float8, float8)'::regprocedure"
            ))
            .unwrap()
            .expect("aggregate not found in pg_aggregate");
            assert!(
                !strict_sfunc,
                "{agg}: SFUNC must be non-strict (handles NULLs itself)"
            );

            let strict_combine = Spi::get_one::<bool>(&format!(
                "SELECT p.proisstrict FROM pg_aggregate a \
                 JOIN pg_proc p ON p.oid = a.aggcombinefn \
                 WHERE a.aggfnoid = '{agg}(float8, float8, float8)'::regprocedure"
            ))
            .unwrap()
            .expect("combinefn not declared");
            assert!(!strict_combine, "{agg}: COMBINEFUNC must be non-strict");
        }
        let strict_final = Spi::get_one::<bool>(
            "SELECT p.proisstrict FROM pg_aggregate a \
             JOIN pg_proc p ON p.oid = a.aggfinalfn \
             WHERE a.aggfnoid = 'softavg(float8, float8, float8)'::regprocedure",
        )
        .unwrap()
        .expect("finalfn not declared");
        assert!(
            !strict_final,
            "softavg: FINALFUNC is non-strict (maps NULL state -> NULL)"
        );
        // sanity: AVG(float8) makes the same declaration choice PG-side
        // reference — its transfn float8_accum IS strict (single-arg,
        // state bootstrap works); the audit above documents WHY softavg
        // legitimately differs (3 args + NULL initcond).
    }

    /// (14c) behavior half of the audit: a NULL `eps` reaching a live
    /// row is an error — the non-strict SFUNC must not silently invent
    /// a temperature. (NULL value/score rows are skipped before the eps
    /// check, matching the AVG-skip discipline.)
    #[pg_test(error = "softavg: eps must not be NULL")]
    fn test_f_null_eps_on_live_row_errors() {
        Spi::run("CREATE TABLE tne(v float8, s float8); INSERT INTO tne VALUES (1.0, 0.5)")
            .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, NULL::float8) FROM tne").unwrap();
    }

    /// (14d) grouping-text parity: softavg does not group — PG does.
    /// GROUP BY over unicode text must behave exactly as it does for
    /// AVG (constitution C4): equal strings collapse ('猫' twice),
    /// canonically-equivalent-but-byte-different strings stay separate
    /// (NFC 'café' vs NFD 'cafe' + U+0301), and per-group results match
    /// AVG under equal scores.
    #[pg_test]
    fn test_g_text_group_by_unicode_parity() {
        Spi::run(
            "CREATE TABLE tg(k text, v float8, s float8); \
             INSERT INTO tg VALUES \
                 (E'caf\\u00e9', 1.0, 0.3), (E'caf\\u00e9', 3.0, 0.3), \
                 (E'cafe\\u0301', 100.0, 0.3), \
                 (E'\\u732b', 10.0, 0.3), (E'\\u732b', 30.0, 0.3), \
                 ('Stra\u{00df}e', 7.0, 0.3)",
        )
        .unwrap();
        // PG's equality decides the groups: 4 distinct keys.
        assert_eq!(
            get_i64("SELECT COUNT(*) FROM (SELECT k FROM tg GROUP BY k) q"),
            4,
            "NFC/NFD must be distinct groups; equal strings must collapse"
        );
        // per-group softavg == per-group AVG (equal scores) — grouping
        // applied to softavg is the same grouping applied to AVG.
        assert_eq!(
            get_i64(
                "SELECT COUNT(*) FROM ( \
                     SELECT k, softavg(v, s, 0.7) AS sa, AVG(v) AS av \
                     FROM tg GROUP BY k) q \
                 WHERE q.sa IS DISTINCT FROM q.av OR abs(q.sa - q.av) > 1e-12"
            ),
            0,
            "some group's softavg diverged from AVG"
        );
        // the byte-different lookalikes really landed in different groups
        let nfc = get_f64("SELECT softavg(v, s, 0.7) FROM tg WHERE k = E'caf\\u00e9'");
        let nfd = get_f64("SELECT softavg(v, s, 0.7) FROM tg WHERE k = E'cafe\\u0301'");
        assert!(
            (nfc - 2.0).abs() < 1e-12,
            "NFC group {{1,3}} -> 2.0, got {nfc}"
        );
        assert!(
            (nfd - 100.0).abs() < 1e-12,
            "NFD group {{100}} -> 100.0, got {nfd}"
        );
    }

    /// (14e) negative eps -> clean SQL ERROR (ereport, not a crash or a
    /// NaN result); same domain as bruce-core's Eps::new.
    #[pg_test(
        error = "softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got -0.5"
    )]
    fn test_h_negative_eps_is_sql_error() {
        Spi::run("CREATE TABLE tneg(v float8, s float8); INSERT INTO tneg VALUES (1.0, 0.5)")
            .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, -0.5) FROM tneg").unwrap();
    }

    /// (14e) eps = 0 with a two-way tie at the max score -> the mean of
    /// exactly the tied rows; the raw state confirms the argmax set has
    /// multiplicity 2 (z counts ties in the tropical regime).
    #[pg_test]
    fn test_h_eps_zero_two_way_tie_is_tied_mean() {
        Spi::run(
            "CREATE TABLE tt(v float8, s float8); \
             INSERT INTO tt VALUES (10.0, 5.0), (999.0, 4.9), (20.0, 5.0), (-7.0, -5.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 0.0) FROM tt");
        assert!((got - 15.0).abs() < 1e-15, "tie mean (10+20)/2, got {got}");
        // state [mu, z, u, eps, nan]: mu = max score, z = tie count, u = tied value sum
        let (mu, z, u) = Spi::get_three::<f64, f64, f64>(
            "SELECT st[1], st[2], st[3] FROM (SELECT softavg_state(v, s, 0.0) AS st FROM tt) q",
        )
        .expect("SPI failed");
        assert_eq!(mu, Some(5.0));
        assert_eq!(
            z,
            Some(2.0),
            "argmax multiplicity must count both tied rows"
        );
        assert_eq!(u, Some(30.0));
    }

    // -----------------------------------------------------------------
    // Special float values (pg-parity track). ±Inf mirrors bruce-core
    // verbatim; NaN deliberately diverges (PROPAGATE, not skip) — the
    // decision and its cost are argued in README.md, "Special float
    // values: NaN, ±Inf, and one deliberate divergence".
    // -----------------------------------------------------------------

    /// Assert a scalar query returns SQL NULL.
    fn assert_sql_null(query: &str, why: &str) {
        let got = Spi::get_one::<f64>(query).expect("SPI failed");
        assert!(got.is_none(), "{why}: expected NULL, got {got:?}");
    }

    /// (j) NaN `value` propagates in every regime — `softavg` joins the
    /// AVG/SUM family rather than silently dropping a real float8.
    #[pg_test]
    fn test_j_nan_value_propagates() {
        Spi::run(
            "CREATE TABLE tjv(v float8, s float8); \
             INSERT INTO tjv VALUES (1.0, 0.2), ('NaN', 0.9), (5.0, -0.1)",
        )
        .unwrap();
        // the PG reference: AVG over the same column is NaN, not 3.0
        assert!(
            get_f64("SELECT AVG(v) FROM tjv").is_nan(),
            "PG's own AVG must be NaN here"
        );
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjv"));
            assert!(
                got.is_nan(),
                "eps={eps}: NaN value must propagate, got {got}"
            );
        }
    }

    /// (j) NaN `score` propagates too — including at
    /// `eps = 'infinity'`, where the score is arithmetically unused.
    /// That follows this crate's established PAIR rule (a NULL score
    /// already drops its row at every eps; see
    /// `test_f_empty_input_is_null_like_avg`), deliberately in
    /// preference to PG's `regr_avgy`, which ignores a NaN in the
    /// argument it never reads. One rule, all three regimes.
    #[pg_test]
    fn test_j_nan_score_propagates_in_every_regime() {
        Spi::run(
            "CREATE TABLE tjs(v float8, s float8); \
             INSERT INTO tjs VALUES (1.0, 0.2), (2.0, 'NaN'), (5.0, -0.1)",
        )
        .unwrap();
        // AVG(v) alone is finite: the divergence is real and intended.
        assert!((get_f64("SELECT AVG(v) FROM tjs") - 8.0 / 3.0).abs() < 1e-12);
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjs"));
            assert!(
                got.is_nan(),
                "eps={eps}: NaN score must propagate, got {got}"
            );
        }
    }

    /// (j) The divergence from bruce-core, pinned explicitly and named:
    /// bruce-core SKIPS NaN (there NaN encodes SQL NULL — see
    /// README.md); bruce_pg PROPAGATES. The skip answer is exactly what
    /// the user gets back by filtering NaN out themselves, which is the
    /// argument for propagating by default: `WHERE x <> 'NaN'::float8`
    /// (PG treats NaN = NaN as true, so this really does filter NaN)
    /// recovers bruce-core's semantics, while nothing recovers a value
    /// a skipping aggregate already discarded.
    #[pg_test]
    fn test_j_nan_propagates_where_bruce_core_skips() {
        Spi::run(
            "CREATE TABLE tjd(v float8, s float8); \
             INSERT INTO tjd VALUES (10.0, 1.0), (777.0, 'NaN'), (20.0, 2.0), \
                                    ('NaN', 0.5), (30.0, 3.0)",
        )
        .unwrap();
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let propagated = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjd"));
            assert!(propagated.is_nan(), "eps={eps}: bruce_pg must propagate");
            // bruce-core's semantics, reconstructed by the user in SQL
            let skipped = get_f64(&format!(
                "SELECT softavg(v, s, {eps}) FROM tjd \
                 WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8"
            ));
            assert!(
                skipped.is_finite(),
                "eps={eps}: skip answer must be finite, got {skipped}"
            );
        }
        // the skip answer at eps = 0 is the argmax row's value...
        let skipped0 = get_f64(
            "SELECT softavg(v, s, 0.0) FROM tjd WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8",
        );
        assert!(
            (skipped0 - 30.0).abs() < 1e-15,
            "argmax over {{1,2,3}} -> 30, got {skipped0}"
        );
        // ...and at eps = inf the mean of the three surviving values.
        let skipped_inf = get_f64(
            "SELECT softavg(v, s, 'infinity'::float8) FROM tjd \
             WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8",
        );
        assert!(
            (skipped_inf - 20.0).abs() < 1e-12,
            "mean{{10,20,30}} = 20, got {skipped_inf}"
        );
    }

    /// (j) The monoid factor of the state is UNTOUCHED by NaN rows —
    /// this is the C2 argument made observable: `[mu, z, u]` equals what
    /// bruce-core's skip produces, and only the fifth slot (the sticky
    /// NaN bit of the product monoid) records the divergence.
    #[pg_test]
    fn test_j_nan_state_component_matches_bruce_core_skip() {
        Spi::run(
            "CREATE TABLE tjst(v float8, s float8); \
             INSERT INTO tjst VALUES (10.0, 5.0), (777.0, 'NaN'), (20.0, 5.0), (-7.0, -5.0)",
        )
        .unwrap();
        let st = |col: usize, pred: &str| {
            Spi::get_one::<f64>(&format!(
                "SELECT st[{col}] FROM (SELECT softavg_state(v, s, 0.0) AS st \
                 FROM tjst {pred}) q"
            ))
            .unwrap()
            .expect("state slot NULL")
        };
        // state is float8[5] now
        let len = Spi::get_one::<i32>(
            "SELECT array_length(st, 1) FROM (SELECT softavg_state(v, s, 0.0) AS st FROM tjst) q",
        )
        .unwrap()
        .expect("no state");
        assert_eq!(len, 5, "state layout is [mu, z, u, eps, nan_seen]");
        // slots 1..3 identical with and without the NaN row present
        for col in 1..=3 {
            assert_eq!(
                st(col, ""),
                st(col, "WHERE s <> 'NaN'::float8"),
                "slot {col}: NaN row must not enter the monoid factor"
            );
        }
        assert_eq!(st(1, ""), 5.0, "mu = max score over absorbed rows");
        assert_eq!(st(2, ""), 2.0, "z = argmax multiplicity");
        assert_eq!(st(3, ""), 30.0, "u = tied value sum");
        // ...and only the sticky bit differs
        assert_eq!(st(5, ""), 1.0, "NaN bit must be set");
        assert_eq!(
            st(5, "WHERE s <> 'NaN'::float8"),
            0.0,
            "NaN bit must be clear"
        );
        // finalize disagrees even though the monoid agrees
        assert!(get_f64("SELECT softavg(v, s, 0.0) FROM tjst").is_nan());
        assert_eq!(
            get_f64("SELECT softavg(v, s, 0.0) FROM tjst WHERE s <> 'NaN'::float8"),
            15.0
        );
    }

    /// (j) An all-NaN group is NaN, NOT NULL — the rows qualified (NaN
    /// is a value); contrast `test_f_empty_input_is_null_like_avg`,
    /// where all-NULL input is NULL. AVG makes the same distinction.
    #[pg_test]
    fn test_j_all_nan_group_is_nan_not_null() {
        Spi::run(
            "CREATE TABLE tjan(v float8, s float8); \
             INSERT INTO tjan VALUES ('NaN', 0.5), ('NaN', 'NaN')",
        )
        .unwrap();
        assert!(
            get_f64("SELECT AVG(v) FROM tjan").is_nan(),
            "PG's AVG: NaN, not NULL"
        );
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjan"));
            assert!(
                got.is_nan(),
                "eps={eps}: all-NaN group must be NaN, got {got}"
            );
        }
    }

    /// (j) NaN must survive the PARALLEL path: a NaN seen by exactly one
    /// worker still reaches the final answer, because the sticky bit is
    /// OR-ed in COMBINEFUNC. Forced parallel plan, as in test_d.
    #[pg_test]
    fn test_j_nan_survives_parallel_combine() {
        Spi::run(
            "CREATE TABLE tjp AS SELECT i, sin(i::float8) * 10.0 AS v, \
             cos(i::float8) AS s FROM generate_series(1, 200000) g(i)",
        )
        .unwrap();
        // exactly one NaN, deep in the scan
        Spi::run("UPDATE tjp SET s = 'NaN' WHERE i = 137021").unwrap();

        Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
        assert!(
            get_f64("SELECT softavg(v, s, 0.25) FROM tjp").is_nan(),
            "serial"
        );

        Spi::run("SET parallel_setup_cost = 0").unwrap();
        Spi::run("SET parallel_tuple_cost = 0").unwrap();
        Spi::run("SET min_parallel_table_scan_size = 0").unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 4").unwrap();
        let plan = explain("SELECT softavg(v, s, 0.25) FROM tjp");
        assert!(
            plan.contains("Gather") && plan.contains("Partial Aggregate"),
            "plan did not parallelize:\n{plan}"
        );
        assert!(
            get_f64("SELECT softavg(v, s, 0.25) FROM tjp").is_nan(),
            "one worker's NaN must survive COMBINEFUNC"
        );
        // and it is genuinely the NaN row doing it
        assert!(
            get_f64("SELECT softavg(v, s, 0.25) FROM tjp WHERE s <> 'NaN'::float8").is_finite(),
            "without the NaN row the same parallel plan is finite"
        );
    }

    /// (j) +Inf score at finite eps: argmax collapse, uniform ties —
    /// mirrored verbatim from bruce-core (there it lives INSIDE the
    /// monoid, so there is no PG-convention conflict). Before 0.1.2
    /// these returned NaN via `exp(inf - inf)`.
    #[pg_test]
    fn test_j_pos_inf_score_collapses_to_argmax() {
        Spi::run(
            "CREATE TABLE tji(v float8, s float8); \
             INSERT INTO tji VALUES (10.0, 1.0), (55.0, 'infinity'), (20.0, 2.0)",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "1.0"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tji"));
            assert_eq!(got, 55.0, "eps={eps}: +inf row must dominate");
        }
        // ties among +inf rows average uniformly, exactly as at eps = 0
        Spi::run(
            "CREATE TABLE tjt(v float8, s float8); \
             INSERT INTO tjt VALUES (50.0, 'infinity'), (999.0, 3.0), (60.0, 'infinity')",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "1.0"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjt"));
            assert_eq!(got, 55.0, "eps={eps}: +inf tie must average uniformly");
        }
    }

    /// (j) the `+Inf` collapse must survive COMBINEFUNC. Two proofs:
    /// an explicit SQL re-association of two partial states that each
    /// hold a `+inf` row, and a forced parallel plan over 200k rows
    /// (test_d's recipe) whose `+inf` rows are spread across the scan.
    #[pg_test]
    fn test_j_pos_inf_parallel_combine_matches_serial() {
        // (1) explicit combine of two +inf-carrying partial states
        Spi::run(
            "CREATE TABLE tjc(i int, v float8, s float8); \
             INSERT INTO tjc VALUES (0, 10.0, 'infinity'), (1, 5.0, 0.5), \
                                    (2, 30.0, 'infinity'), (3, 7.0, 0.9)",
        )
        .unwrap();
        // partition so that EACH side holds exactly one +inf row: this
        // is the mu = +inf on BOTH sides branch of merge, where a naive
        // re-basing would evaluate exp(inf - inf).
        let recombined = get_f64(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i IN (0, 1)), \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i IN (2, 3))))",
        );
        assert_eq!(
            recombined, 20.0,
            "combine must keep both +inf rows (mean 10,30)"
        );
        // asymmetric case: only ONE side holds the +inf row
        let one_sided = get_f64(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i = 0), \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i IN (1, 3))))",
        );
        assert_eq!(one_sided, 10.0, "+inf side must dominate the finite side");

        // (2) real parallel plan
        Spi::run(
            "CREATE TABLE tjpi AS SELECT i, sin(i::float8) * 10.0 AS v, \
             cos(i::float8) AS s FROM generate_series(1, 200000) g(i)",
        )
        .unwrap();
        Spi::run(
            "UPDATE tjpi SET s = 'infinity', v = i / 1000.0 \
             WHERE i IN (7, 60013, 120017, 190019)",
        )
        .unwrap();
        let want = (7.0 + 60013.0 + 120017.0 + 190019.0) / 4000.0;

        Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
        let serial = get_f64("SELECT softavg(v, s, 0.25) FROM tjpi");
        assert!((serial - want).abs() < 1e-12, "serial {serial} != {want}");

        Spi::run("SET parallel_setup_cost = 0").unwrap();
        Spi::run("SET parallel_tuple_cost = 0").unwrap();
        Spi::run("SET min_parallel_table_scan_size = 0").unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 4").unwrap();
        let plan = explain("SELECT softavg(v, s, 0.25) FROM tjpi");
        assert!(
            plan.contains("Gather") && plan.contains("Partial Aggregate"),
            "plan did not parallelize:\n{plan}"
        );
        let parallel = get_f64("SELECT softavg(v, s, 0.25) FROM tjpi");
        assert!(
            (parallel - serial).abs() < 1e-12,
            "parallel {parallel} != serial {serial} (uniform mean over the +inf rows)"
        );
    }

    /// (j) -Inf carries weight 0 at eps = 0 and finite eps (the
    /// Indicator "no match" encoding), and an all--Inf group is SQL
    /// NULL — bruce-core's "uncovered" row, spelled the PG way.
    #[pg_test]
    fn test_j_neg_inf_is_weight_zero_and_all_neg_inf_is_null() {
        Spi::run(
            "CREATE TABLE tjn(v float8, s float8); \
             INSERT INTO tjn VALUES (888.0, '-infinity'), (10.0, 1.0)",
        )
        .unwrap();
        Spi::run(
            "CREATE TABLE tjnn(v float8, s float8); \
             INSERT INTO tjnn VALUES (888.0, '-infinity'), (999.0, '-infinity')",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "1.0"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjn"));
            assert_eq!(got, 10.0, "eps={eps}: -inf row must weigh 0");
            assert_sql_null(
                &format!("SELECT softavg(v, s, {eps}) FROM tjnn"),
                &format!("eps={eps}: all--inf group"),
            );
        }
        // and the empty monoid factor is a NULL, not a NaN and not a 0
        assert_eq!(
            get_i64(
                "SELECT COUNT(*) FROM (SELECT softavg(v, s, 0.5) AS x FROM tjnn) q \
                 WHERE q.x IS NULL"
            ),
            1
        );
    }

    /// (j) eps = 'infinity' is score-blind: ±Inf scores count like any
    /// other row (plain mean), so the all--Inf group is NOT null here.
    #[pg_test]
    fn test_j_eps_inf_is_score_blind_for_infinite_scores() {
        Spi::run(
            "CREATE TABLE tjb(v float8, s float8); \
             INSERT INTO tjb VALUES (3.0, 'infinity'), (6.0, '-infinity'), (9.0, 1.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 'infinity'::float8) FROM tjb");
        assert_eq!(got, 6.0, "plain mean of 3, 6, 9");
        assert!(
            (got - get_f64("SELECT AVG(v) FROM tjb")).abs() < 1e-15,
            "eps = inf must still equal AVG"
        );
        // all--inf scores at eps = inf: a plain mean, not NULL
        let all_neg =
            get_f64("SELECT softavg(v, s, 'infinity'::float8) FROM tjb WHERE s = '-infinity'");
        assert_eq!(all_neg, 6.0);
    }

    /// (j) mixed NaN + ±Inf: NaN wins over everything, in every regime.
    /// (The sticky bit is consulted by FINALFUNC before the monoid
    /// factor, so a +Inf collapse or an empty all--Inf factor cannot
    /// mask a NaN that qualified.)
    #[pg_test]
    fn test_j_nan_dominates_mixed_infinities() {
        Spi::run(
            "CREATE TABLE tjm(v float8, s float8); \
             INSERT INTO tjm VALUES (5.0, 'infinity'), (1.0, '-infinity'), \
                                    (2.0, 'NaN'), (3.0, 0.5)",
        )
        .unwrap();
        // all--inf rows plus a NaN: the monoid factor is EMPTY, yet the
        // answer must be NaN rather than NULL.
        Spi::run(
            "CREATE TABLE tjm2(v float8, s float8); \
             INSERT INTO tjm2 VALUES (1.0, '-infinity'), ('NaN', '-infinity')",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjm"));
            assert!(got.is_nan(), "eps={eps}: NaN must dominate +inf, got {got}");
        }
        for eps in ["0.0", "0.37"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjm2"));
            assert!(
                got.is_nan(),
                "eps={eps}: NaN over an empty factor must be NaN, got {got}"
            );
        }
    }

    /// (j) a NaN row is a LIVE row, so the eps contract still applies to
    /// it — the NULL-input skip happens before the eps checks, the NaN
    /// poison after. A NaN row with a negative eps must still raise.
    #[pg_test(
        error = "softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got -0.5"
    )]
    fn test_j_nan_row_still_honours_eps_domain() {
        Spi::run("CREATE TABLE tje(v float8, s float8); INSERT INTO tje VALUES ('NaN', 'NaN')")
            .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, -0.5) FROM tje").unwrap();
    }

    // -----------------------------------------------------------------
    // Upgrade path (workstream 13): ALTER EXTENSION bruce_pg UPDATE
    // through a real versioned upgrade script, exercised in-database.
    // The harness installs the extension at the current default version
    // (0.1.2) before tests run, so the test stages a synthetic
    // 0.1.2 -> 0.1.3 script (same trivial shape as the shipped
    // sql/bruce_pg--0.1.1--0.1.2.sql) into <sharedir>/extension/ and
    // drives PG's own update machinery over it.
    // -----------------------------------------------------------------
    #[pg_test]
    fn test_i_upgrade_path_alter_extension() {
        let installed = Spi::get_one::<String>(
            "SELECT extversion FROM pg_extension WHERE extname = 'bruce_pg'",
        )
        .unwrap()
        .expect("extension not installed");
        assert_eq!(
            installed, "0.1.2",
            "control default_version tracks Cargo.toml"
        );

        let sharedir =
            Spi::get_one::<String>("SELECT setting FROM pg_config WHERE name = 'SHAREDIR'")
                .unwrap()
                .expect("pg_config view gave no SHAREDIR");
        let script = std::path::Path::new(&sharedir)
            .join("extension")
            .join("bruce_pg--0.1.2--0.1.3.sql");
        std::fs::write(
            &script,
            "COMMENT ON AGGREGATE softavg(float8, float8, float8) IS \
             'bruce_pg 0.1.3 upgrade probe';\n",
        )
        .expect("cannot stage upgrade script into sharedir");

        // the actual machinery under test
        let outcome = Spi::run("ALTER EXTENSION bruce_pg UPDATE TO '0.1.3'");
        // stage file cleanup must happen even if the ALTER failed
        // (the catalog change itself rolls back with the test txn)
        let _ = std::fs::remove_file(&script);
        outcome.expect("ALTER EXTENSION bruce_pg UPDATE TO '0.1.3' failed");

        let now = Spi::get_one::<String>(
            "SELECT extversion FROM pg_extension WHERE extname = 'bruce_pg'",
        )
        .unwrap()
        .unwrap();
        assert_eq!(now, "0.1.3", "extversion did not advance");
        let comment = Spi::get_one::<String>(
            "SELECT obj_description('softavg(float8, float8, float8)'::regprocedure, 'pg_proc')",
        )
        .unwrap()
        .expect("upgrade script's COMMENT did not apply");
        assert_eq!(comment, "bruce_pg 0.1.3 upgrade probe");
    }

    fn explain(query: &str) -> String {
        Spi::connect(|client| {
            let table = client.select(&format!("EXPLAIN (COSTS OFF) {query}"), None, &[])?;
            let mut out = String::new();
            for row in table {
                if let Some(line) = row.get::<String>(1)? {
                    out.push_str(&line);
                    out.push('\n');
                }
            }
            Ok::<String, pgrx::spi::Error>(out)
        })
        .expect("EXPLAIN failed")
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}

//! TRACK 1 / workstream 1 — monoid property tests for the grouped
//! soft-average kernels (`grouped_softavg`, `grouped_softavg_f32`).
//!
//! The kernel's correctness claim is algebraic: the per-group
//! `(mu, z, u)` accumulator is a commutative monoid, and the fold is a
//! monoid homomorphism from the free commutative monoid on
//! `(score, value)` records. Concretely that means:
//!
//! 1. **Partition/order invariance** — split the record multiset into
//!    1..8 shards, shuffle each shard, concatenate the shards in any
//!    order: the fold over the resulting permutation equals the fold
//!    over the original order (absorb commutativity/associativity).
//! 2. **Merge == sequential** — the rayon parallel path folds each
//!    chunk independently and combines with `RowAcc::merge`; the
//!    merged result must equal the single-thread fold (Lemma B).
//! 3. **Empty accumulator is the identity** — a shard that absorbs
//!    nothing must merge as a no-op.
//!
//! ### Why idempotence is NOT tested
//!
//! `merge(a, a) != a`: the monoid is on **multisets under bag-union**
//! (PG combinefunc contract, C2) — merging a partial state with itself
//! doubles `z` and `u`, i.e. it represents the double-counted bag.
//! (The *normalised* output `u/z` happens to coincide, but pinning
//! that would mislead: the accumulator-level contract is what PG's
//! parallel aggregation relies on, and combinefunc is only ever called
//! on disjoint partitions.)
//!
//! ### Tolerance derivation (documented per the testing matrix)
//!
//! At finite eps each absorb performs <= 3 rounded f64 ops per
//! accumulator component (exp, mul, add); all weights are >= 0 and the
//! generated values are >= 0.5, so the sums have no catastrophic
//! cancellation and a fold of n records carries relative error
//! <= ~3n*u, u = 2^-53 ~ 1.1e-16. For the proptest sizes (n <= 64)
//! that is <= 2.2e-14 per run; comparing two orders doubles it to
//! <= 4.4e-14, and the merge test's n ~ 3.7e4 gives <= 1.2e-11 worst
//! case but in practice per-group n ~ n/16 so ~ 8e-13. Contracted
//! ceilings asserted here:
//!   - f64 kernel: 1e-12 relative;
//!   - f32 kernel vs its own single-thread run: 1e-5 relative. (Both
//!     runs compute identical per-row f32 scores — only the f64
//!     accumulation order differs, so the observed diff is again
//!     ~1e-13; 1e-5 is the kernel's external precision promise and is
//!     deliberately the looser, contract-level bound.)

use bruce_core::mask::{grouped_softavg, grouped_softavg_f32};
use bruce_core::types::Eps;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

/// The four contracted temperatures: tropical, two finite, uniform.
const EPS_GRID: [Eps; 4] = [Eps::ZERO, Eps(0.37), Eps::ONE, Eps::INF];

const REL_F64: f64 = 1e-12;
const REL_F32: f64 = 1e-5;

/// xorshift64* — deterministic, no rand dev-dependency needed.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f64(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// One generated instance: per-record (score, d_v values), a group id
/// per record, a shard id per record, and a shuffle seed.
#[derive(Debug, Clone)]
struct Instance {
    scores: Vec<f64>,
    values: Vec<f64>, // n * d_v, row-major
    d_v: usize,
    gid: Vec<u32>,
    n_groups: usize,
    shard: Vec<u8>,
    n_shards: usize,
    shuffle_seed: u64,
}

fn instance(single_group: bool) -> impl Strategy<Value = Instance> {
    (2usize..64, 1usize..4, 1usize..6, 1usize..9).prop_flat_map(
        move |(n, d_v, n_groups, n_shards)| {
            let n_groups = if single_group { 1 } else { n_groups };
            (
                proptest::collection::vec(-8.0f64..8.0, n..=n),
                // values >= 0.5 keep relative error well-conditioned
                // (see tolerance derivation in the module docs)
                proptest::collection::vec(0.5f64..5.0, n * d_v..=n * d_v),
                Just(d_v),
                proptest::collection::vec(0..n_groups as u32, n..=n),
                Just(n_groups),
                proptest::collection::vec(0..n_shards as u8, n..=n),
                Just(n_shards),
                any::<u64>(),
            )
                .prop_map(
                    |(scores, values, d_v, gid, n_groups, shard, n_shards, shuffle_seed)| {
                        Instance {
                            scores,
                            values,
                            d_v,
                            gid,
                            n_groups,
                            shard,
                            n_shards,
                            shuffle_seed,
                        }
                    },
                )
        },
    )
}

/// Realise the shard structure as a permutation of record indices:
/// Fisher-Yates-shuffle each shard's records, then concatenate the
/// shards in a shuffled shard order. Any such permutation is exactly
/// "random partition into shards + random per-shard order".
fn shard_permutation(inst: &Instance) -> Vec<usize> {
    let mut rng = Rng::new(inst.shuffle_seed);
    let mut shards: Vec<Vec<usize>> = vec![Vec::new(); inst.n_shards];
    for (r, &s) in inst.shard.iter().enumerate() {
        shards[s as usize].push(r);
    }
    for sh in shards.iter_mut() {
        // Fisher–Yates
        for i in (1..sh.len()).rev() {
            let j = rng.below(i + 1);
            sh.swap(i, j);
        }
    }
    // shuffle the shard order itself
    for i in (1..shards.len()).rev() {
        let j = rng.below(i + 1);
        shards.swap(i, j);
    }
    shards.into_iter().flatten().collect()
}

/// Run the f64 kernel with scores injected via a d_k = 1 key column
/// and x = [1.0] (so score(r) == scores[r] exactly).
fn run_f64(
    scores: &[f64],
    values: &[f64],
    d_v: usize,
    gid: &[u32],
    n_groups: usize,
    eps: Eps,
) -> (Array2<f64>, Vec<bool>) {
    let n = scores.len();
    let x = Array1::from_vec(vec![1.0f64]);
    let k = Array2::from_shape_vec((n, 1), scores.to_vec()).unwrap();
    let v = Array2::from_shape_vec((n, d_v), values.to_vec()).unwrap();
    grouped_softavg(&x.view(), &k.view(), &v.view(), gid, n_groups, None, eps).unwrap()
}

/// Same for the f32 kernel: scores are pre-rounded to f32 so every
/// permutation sees bit-identical per-row f32 scores.
fn run_f32(
    scores: &[f64],
    values: &[f64],
    d_v: usize,
    gid: &[u32],
    n_groups: usize,
    eps: Eps,
) -> (Array2<f64>, Vec<bool>) {
    let n = scores.len();
    let x = Array1::from_vec(vec![1.0f32]);
    let k = Array2::from_shape_vec((n, 1), scores.iter().map(|&s| s as f32).collect()).unwrap();
    let v = Array2::from_shape_vec((n, d_v), values.to_vec()).unwrap();
    grouped_softavg_f32(&x.view(), &k.view(), &v.view(), gid, n_groups, None, eps).unwrap()
}

fn assert_close(a: &Array2<f64>, b: &Array2<f64>, rel: f64, ctx: &str) {
    for (x, y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(y.abs()).max(1.0);
        let d = (x - y).abs() / denom;
        assert!(d <= rel, "{ctx}: {x} vs {y} (rel {d:e} > {rel:e})");
    }
}

fn permute<T: Copy>(xs: &[T], perm: &[usize], width: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(xs.len());
    for &r in perm {
        out.extend_from_slice(&xs[r * width..(r + 1) * width]);
    }
    out
}

fn check_partition_invariance(inst: &Instance, f32_kernel: bool) {
    let perm = shard_permutation(inst);
    let p_scores = permute(&inst.scores, &perm, 1);
    let p_values = permute(&inst.values, &perm, inst.d_v);
    let p_gid: Vec<u32> = perm.iter().map(|&r| inst.gid[r]).collect();
    for eps in EPS_GRID {
        let (a, cov_a, b, cov_b, rel) = if f32_kernel {
            let (a, ca) = run_f32(
                &inst.scores,
                &inst.values,
                inst.d_v,
                &inst.gid,
                inst.n_groups,
                eps,
            );
            let (b, cb) = run_f32(&p_scores, &p_values, inst.d_v, &p_gid, inst.n_groups, eps);
            (a, ca, b, cb, REL_F32)
        } else {
            let (a, ca) = run_f64(
                &inst.scores,
                &inst.values,
                inst.d_v,
                &inst.gid,
                inst.n_groups,
                eps,
            );
            let (b, cb) = run_f64(&p_scores, &p_values, inst.d_v, &p_gid, inst.n_groups, eps);
            (a, ca, b, cb, REL_F64)
        };
        assert_eq!(
            cov_a, cov_b,
            "coverage changed under partition, eps={eps:?}"
        );
        assert_close(&a, &b, rel, &format!("partition invariance eps={eps:?}"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    /// Multi-group, f64 kernel: any shard partition + per-shard order
    /// folds to the sequential result.
    #[test]
    fn f64_multi_group_partition_invariance(inst in instance(false)) {
        check_partition_invariance(&inst, false);
    }

    /// Single group, f64 kernel (the "one big aggregate" case).
    #[test]
    fn f64_single_group_partition_invariance(inst in instance(true)) {
        check_partition_invariance(&inst, false);
    }

    /// Multi-group, f32 kernel: same property against its own
    /// baseline-order run (identical f32 scores; only f64
    /// accumulation order differs).
    #[test]
    fn f32_multi_group_partition_invariance(inst in instance(false)) {
        check_partition_invariance(&inst, true);
    }

    /// Single group, f32 kernel.
    #[test]
    fn f32_single_group_partition_invariance(inst in instance(true)) {
        check_partition_invariance(&inst, true);
    }
}

/// Synthesize a large deterministic workload (above the kernel's
/// parallel threshold of 2^15 pairs) so the rayon chunk-merge path
/// (`RowAcc::merge`) actually runs, with 1..8 chunks.
fn big_workload(n: usize, d_v: usize, n_groups: usize, seed: u64) -> Instance {
    let mut rng = Rng::new(seed);
    Instance {
        scores: (0..n).map(|_| rng.f64() * 16.0 - 8.0).collect(),
        values: (0..n * d_v).map(|_| rng.f64() * 4.5 + 0.5).collect(),
        d_v,
        gid: (0..n).map(|_| rng.below(n_groups) as u32).collect(),
        n_groups,
        shard: vec![0; n],
        n_shards: 1,
        shuffle_seed: seed,
    }
}

/// Workstream 1 merge check: for every shard count 1..8, run the
/// kernel inside a rayon pool of exactly that many threads (the
/// kernel chunks by `current_num_threads`, so this pins the number of
/// merged partial folds) and compare to the 1-thread run.
#[test]
fn merge_of_1_to_8_shards_equals_sequential_fold() {
    let inst = big_workload((1 << 15) + 4097, 2, 16, 0xB0B5);
    for f32_kernel in [false, true] {
        let rel = if f32_kernel { REL_F32 } else { REL_F64 };
        for eps in EPS_GRID {
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let (base, cov_base) = one.install(|| {
                if f32_kernel {
                    run_f32(
                        &inst.scores,
                        &inst.values,
                        inst.d_v,
                        &inst.gid,
                        inst.n_groups,
                        eps,
                    )
                } else {
                    run_f64(
                        &inst.scores,
                        &inst.values,
                        inst.d_v,
                        &inst.gid,
                        inst.n_groups,
                        eps,
                    )
                }
            });
            for threads in 2..=8usize {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let (out, cov) = pool.install(|| {
                    if f32_kernel {
                        run_f32(
                            &inst.scores,
                            &inst.values,
                            inst.d_v,
                            &inst.gid,
                            inst.n_groups,
                            eps,
                        )
                    } else {
                        run_f64(
                            &inst.scores,
                            &inst.values,
                            inst.d_v,
                            &inst.gid,
                            inst.n_groups,
                            eps,
                        )
                    }
                });
                assert_eq!(cov, cov_base, "coverage, {threads} shards, eps={eps:?}");
                assert_close(
                    &out,
                    &base,
                    rel,
                    &format!("merge f32={f32_kernel} shards={threads} eps={eps:?}"),
                );
            }
        }
    }
}

/// Workstream 1 identity check: a shard whose every record is
/// deselected produces the empty accumulator for every group; merging
/// it must be the identity. We deselect exactly the third of five
/// rayon chunks and compare against (a) the same input on one thread
/// and (b) the physically-shrunk input.
#[test]
fn merge_with_empty_shard_is_identity() {
    let chunk = 8192usize;
    let n = 5 * chunk; // 40960 >= 2^15 -> parallel path
    let inst = big_workload(n, 2, 8, 0x1DE47);
    let mut sel = vec![true; n];
    for s in sel.iter_mut().take(3 * chunk).skip(2 * chunk) {
        *s = false; // chunk #3 of 5 is entirely empty
    }
    // physically remove the deselected rows for reference (b)
    let keep: Vec<usize> = (0..n).filter(|&r| sel[r]).collect();
    let scores_b = permute(&inst.scores, &keep, 1);
    let values_b = permute(&inst.values, &keep, inst.d_v);
    let gid_b: Vec<u32> = keep.iter().map(|&r| inst.gid[r]).collect();

    let x = Array1::from_vec(vec![1.0f64]);
    let k = Array2::from_shape_vec((n, 1), inst.scores.clone()).unwrap();
    let v = Array2::from_shape_vec((n, inst.d_v), inst.values.clone()).unwrap();
    for eps in EPS_GRID {
        let five = rayon::ThreadPoolBuilder::new()
            .num_threads(5)
            .build()
            .unwrap();
        let (out, cov) = five
            .install(|| {
                grouped_softavg(
                    &x.view(),
                    &k.view(),
                    &v.view(),
                    &inst.gid,
                    inst.n_groups,
                    Some(&sel),
                    eps,
                )
            })
            .unwrap();
        // (a) same masked input, single thread (no merge at all)
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (seq, cov_seq) = one
            .install(|| {
                grouped_softavg(
                    &x.view(),
                    &k.view(),
                    &v.view(),
                    &inst.gid,
                    inst.n_groups,
                    Some(&sel),
                    eps,
                )
            })
            .unwrap();
        assert_eq!(cov, cov_seq);
        assert_close(&out, &seq, REL_F64, &format!("empty-shard (a) eps={eps:?}"));
        // (b) shrunk input, sequential
        let (shrunk, cov_shrunk) =
            run_f64(&scores_b, &values_b, inst.d_v, &gid_b, inst.n_groups, eps);
        assert_eq!(cov, cov_shrunk);
        assert_close(
            &out,
            &shrunk,
            REL_F64,
            &format!("empty-shard (b) eps={eps:?}"),
        );
    }
}

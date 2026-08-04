//! Workstream 17 — criterion suite for the fold kernels + memory CRUD.
//!
//! Coverage (ids are stable — bench_compare.py keys on them):
//!   grouped_softavg/{f64,f32}_{100k,1M}_d{64,384}   (d_v = 1, the SQL
//!       SOFTAVG shape; 32 groups, eps = 0.1 like the movie workload)
//!   masked_attention/window_100k_pairs_d64          (pair-stream form)
//!   kv_memory/{insert_1k,delete_1k}                 (CRUD throughput,
//!       d_k = d_v = 64)
//!
//! Anti-noise protocol (documented here AND in the saved baseline):
//!   * criterion's MEDIAN point estimate is the number that is saved
//!     and compared — not the mean, so a stray descheduling spike
//!     cannot move the gate;
//!   * idle-box assumption: run with the machine otherwise quiet (the
//!     32-core box; no taskset pinning — the kernels are rayon-wide by
//!     design, pinning would benchmark a different engine);
//!   * fixed deterministic inputs (arithmetic fill, no RNG state);
//!   * regression gate at +15%, comfortably above the run-to-run
//!     jitter observed on this box (single-digit percent on medians);
//!   * compare with scripts/bench_compare.py against
//!     paper_sigmod_bruce/experiments/perf_baselines/.
//!
//! Run:  cargo bench -p bruce-core --bench fold

use std::time::Duration;

use bruce_core::mask::{grouped_softavg, grouped_softavg_f32, masked_attention, window_pairs};
use bruce_core::memory::KvMemory;
use bruce_core::types::Eps;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use ndarray::{Array1, Array2};

/// Deterministic fill in [-0.5, 0.5) — same convention as mask_bench.
fn fill(i: usize, j: usize) -> f64 {
    ((i * 31 + j * 17) % 97) as f64 / 97.0 - 0.5
}

/// Pseudo-random-ish group ids in [0, n_groups) (Knuth multiplicative
/// hash so groups are not a perfect stride pattern).
fn gids(n: usize, n_groups: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) % n_groups) as u32)
        .collect()
}

fn bench_grouped(c: &mut Criterion) {
    let eps = Eps::new(0.1).unwrap();
    let n_groups = 32usize;
    let mut g = c.benchmark_group("grouped_softavg");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4));

    for &n in &[100_000usize, 1_000_000] {
        for &d in &[64usize, 384] {
            // build per-config and drop before the next (1M x 384 f64
            // keys are ~3 GB; holding all configs at once would not
            // fit the soak's memory discipline)
            let x64 = Array1::from_shape_fn(d, |j| fill(0, j));
            let k64 = Array2::from_shape_fn((n, d), |(i, j)| fill(i, j));
            let v = Array2::from_shape_fn((n, 1), |(i, _)| fill(i, 7));
            let gid = gids(n, n_groups);
            let tag = if n == 1_000_000 { "1M" } else { "100k" };

            g.throughput(Throughput::Elements(n as u64));
            g.bench_function(format!("f64_{tag}_d{d}"), |b| {
                b.iter(|| {
                    grouped_softavg(
                        black_box(&x64.view()),
                        black_box(&k64.view()),
                        black_box(&v.view()),
                        black_box(&gid),
                        n_groups,
                        None,
                        eps,
                    )
                    .unwrap()
                })
            });

            let x32 = x64.mapv(|t| t as f32);
            let k32 = k64.mapv(|t| t as f32);
            drop(k64);
            g.bench_function(format!("f32_{tag}_d{d}"), |b| {
                b.iter(|| {
                    grouped_softavg_f32(
                        black_box(&x32.view()),
                        black_box(&k32.view()),
                        black_box(&v.view()),
                        black_box(&gid),
                        n_groups,
                        None,
                        eps,
                    )
                    .unwrap()
                })
            });
        }
    }
    g.finish();
}

fn bench_masked(c: &mut Criterion) {
    // ~100k pairs: sliding window w = 63 over 1600 rows
    // (1600 * 64 - 63*64/2 = 100_384 pairs), d_k = d_v = 64.
    let n = 1600usize;
    let d = 64usize;
    let q = Array2::from_shape_fn((n, d), |(i, j)| fill(i, j));
    let k = Array2::from_shape_fn((n, d), |(i, j)| fill(i + 1, j));
    let v = Array2::from_shape_fn((n, d), |(i, j)| fill(i + 2, j));
    let pairs = window_pairs(n, 63);
    assert_eq!(pairs.len(), 100_384);

    let mut g = c.benchmark_group("masked_attention");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
        .throughput(Throughput::Elements(pairs.len() as u64));
    g.bench_function("window_100k_pairs_d64", |b| {
        b.iter(|| {
            masked_attention(
                black_box(&q.view()),
                black_box(&k.view()),
                black_box(&v.view()),
                black_box(&pairs),
                Eps::ONE,
            )
            .unwrap()
        })
    });
    g.finish();
}

fn bench_kv_memory(c: &mut Criterion) {
    let (d_k, d_v, n) = (64usize, 64usize, 1000usize);
    let keys: Vec<Array1<f64>> = (0..n)
        .map(|i| Array1::from_shape_fn(d_k, |j| fill(i, j)))
        .collect();
    let vals: Vec<Array1<f64>> = (0..n)
        .map(|i| Array1::from_shape_fn(d_v, |j| fill(i + 1, j)))
        .collect();
    let ids: Vec<String> = (0..n).map(|i| format!("fact_{i:05}")).collect();

    let mut g = c.benchmark_group("kv_memory");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
        .throughput(Throughput::Elements(n as u64));

    g.bench_function("insert_1k", |b| {
        b.iter_batched(
            || KvMemory::new(d_k, d_v),
            |mut m| {
                for i in 0..n {
                    m.write(&ids[i], keys[i].view(), vals[i].view(), "bench")
                        .unwrap();
                }
                m
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("delete_1k", |b| {
        b.iter_batched(
            || {
                let mut m = KvMemory::new(d_k, d_v);
                for i in 0..n {
                    m.write(&ids[i], keys[i].view(), vals[i].view(), "bench")
                        .unwrap();
                }
                m
            },
            |mut m| {
                for id in &ids {
                    m.delete(id, "bench").unwrap();
                }
                m
            },
            BatchSize::LargeInput,
        )
    });
    g.finish();
}

criterion_group!(benches, bench_grouped, bench_masked, bench_kv_memory);
criterion_main!(benches);

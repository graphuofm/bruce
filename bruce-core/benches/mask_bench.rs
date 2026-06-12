//! Microbenchmarks for the generic masked-attention evaluator.
//!
//! Tracks the cost model O(|pairs| * d): causal (quadratic pairs),
//! sliding window (linear pairs), and the tropical regime.

use bruce_core::mask::{causal_pairs, masked_attention, window_pairs};
use bruce_core::types::Eps;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;

fn data(n: usize, d_k: usize, d_v: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let f = |(i, j): (usize, usize)| ((i * 31 + j * 17) % 97) as f64 / 97.0 - 0.5;
    (
        Array2::from_shape_fn((n, d_k), f),
        Array2::from_shape_fn((n, d_k), f),
        Array2::from_shape_fn((n, d_v), f),
    )
}

fn bench_masked(c: &mut Criterion) {
    let (q, k, v) = data(512, 64, 64);
    let causal = causal_pairs(512);
    c.bench_function("masked_causal_N512_d64_eps1", |b| {
        b.iter(|| {
            masked_attention(
                black_box(&q.view()),
                black_box(&k.view()),
                black_box(&v.view()),
                black_box(&causal),
                Eps::ONE,
            )
            .unwrap()
        })
    });

    let (q8, k8, v8) = data(8192, 64, 64);
    let win = window_pairs(8192, 64);
    c.bench_function("masked_window_N8192_w64_d64_eps1", |b| {
        b.iter(|| {
            masked_attention(
                black_box(&q8.view()),
                black_box(&k8.view()),
                black_box(&v8.view()),
                black_box(&win),
                Eps::ONE,
            )
            .unwrap()
        })
    });
    c.bench_function("masked_window_N8192_w64_d64_tropical", |b| {
        b.iter(|| {
            masked_attention(
                black_box(&q8.view()),
                black_box(&k8.view()),
                black_box(&v8.view()),
                black_box(&win),
                Eps::ZERO,
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_masked);
criterion_main!(benches);

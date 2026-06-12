//! Microbenchmark for the F_ε operator and incremental memory.

use bruce_core::{Eps, F_eps, IncrementalMemory, Sim};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};

fn bench_attention_dense(c: &mut Criterion) {
    let n = 10_000;
    let d = 64;
    let x = Array1::<f64>::from_elem(d, 1.0);
    let k = Array2::<f64>::from_elem((n, d), 0.5);
    let v = Array2::<f64>::from_elem((n, d), 0.1);
    let op = F_eps::new(Eps::ONE, Sim::Dot);

    c.bench_function("attention_dense_N10K_d64", |b| {
        b.iter(|| {
            let out = op.attention(&black_box(x.view()), &k.view(), &v.view());
            black_box(out);
        });
    });
}

fn bench_incremental_delete(c: &mut Criterion) {
    // build a memory of 100K records, then time DELETE
    let d_v = 8;
    let x = Array1::<f64>::from_elem(8, 1.0);
    let mut mem = IncrementalMemory::new(x.view(), Eps::ONE, d_v, Sim::Dot);
    for i in 0..100_000 {
        let k = Array1::<f64>::from_elem(8, (i % 1000) as f64 * 0.001);
        let v = Array1::<f64>::from_elem(d_v, (i % 100) as f64);
        mem.insert(&format!("k{i}"), k.view(), v.view()).unwrap();
    }
    let mut to_delete = 0u64;
    c.bench_function("incremental_delete_at_N100K", |b| {
        b.iter(|| {
            let k = Array1::<f64>::from_elem(8, ((to_delete % 1000) as f64) * 0.001);
            let v = Array1::<f64>::from_elem(d_v, (to_delete % 100) as f64);
            let id = format!("kd{to_delete}");
            mem.insert(&id, k.view(), v.view()).unwrap();
            mem.delete(&id).unwrap();
            to_delete += 1;
            black_box(mem.output());
        });
    });
}

criterion_group!(benches, bench_attention_dense, bench_incremental_delete);
criterion_main!(benches);

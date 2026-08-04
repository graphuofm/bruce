//! End-to-end pure-Rust run of the movie query over the real data:
//! Parquet in, SQL in, grouped answer out. Usage:
//!   cargo run --release -p bruce-query --example one_query -- \
//!     <movies.parquet> <emb.npy> [eps]

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array1, Array2};

use bruce_query::db::Database;
use bruce_query::Table;

/// Minimal .npy reader for C-order float32/float64 2-D arrays.
fn read_npy_2d(path: &str) -> Array2<f64> {
    let bytes = std::fs::read(path).expect("read npy");
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    let f32_kind = header.contains("<f4");
    assert!(
        f32_kind || header.contains("<f8"),
        "unsupported dtype: {header}"
    );
    let shape_part = header.split("'shape':").nth(1).unwrap();
    let dims: Vec<usize> = shape_part
        .split(')')
        .next()
        .unwrap()
        .trim_start_matches([' ', '('])
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let (n, d) = (dims[0], dims[1]);
    let data = &bytes[10 + header_len..];
    let v: Vec<f64> = if f32_kind {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect()
    } else {
        data.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    Array2::from_shape_vec((n, d), v).expect("npy shape")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (pq, npy) = (&args[1], &args[2]);
    let eps: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.1);

    let t0 = Instant::now();
    let mut table = Table::from_parquet(pq).expect("parquet");
    let emb = read_npy_2d(npy);
    let d = emb.ncols();
    table.attach_key_f64("emb", emb).expect("attach");
    let mut db = Database::new();
    db.register("movies", table);
    println!("load: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let sql = format!(
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), {eps}) \
         FROM movies WHERE year >= 2000 GROUP BY genre"
    );

    // query vector: read from a sibling npy if given, else first row basis
    let qv: Array1<f64> = if let Some(qp) = args.get(4) {
        read_npy_2d(qp).row(0).to_owned()
    } else {
        Array1::zeros(d)
    };
    let mut params = HashMap::new();
    params.insert("q".to_string(), qv);

    // warm + median of 7 (Database::run plans with the cost model,
    // consults maintained views, then executes)
    let (_, planned) = db.run(&sql, &params).unwrap();
    println!("{}", planned.chosen.explain());
    let mut times = Vec::new();
    let mut out = None;
    for _ in 0..7 {
        let t = Instant::now();
        out = Some(db.run(&sql, &params).unwrap().0);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let out = out.unwrap();
    println!("query: {:.1} ms (median of 7)", times[3]);
    let mut rows: Vec<_> = out.labels.iter().zip(out.values.iter()).collect();
    rows.sort_by(|a, b| b.1.total_cmp(a.1));
    for (l, v) in rows.iter().take(5) {
        println!("  {l:20} {v:.4}");
    }
}

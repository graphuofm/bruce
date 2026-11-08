//! ns per `f64::exp` call (Rust std -> glibc libm) by argument range, the
//! ranges a max-anchored weight `exp((s - mu)/eps)` lands in as eps shrinks.
//! Prints one JSON object to stdout. Best of 7 passes over 2M arguments.
use std::time::Instant;

fn main() {
    let n = 2_000_000usize;
    let mut seed = 1u64;
    let mut r = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let u: Vec<f64> = (0..n).map(|_| r()).collect();
    let ranges: [(&str, f64, f64); 7] = [
        ("[-1,0] normal", -1.0, 0.0),
        ("[-500,0] normal", -500.0, 0.0),
        ("[-700,-512] normal, slow path", -700.0, -512.0),
        ("[-745,-708.4] subnormal result", -745.0, -708.4),
        ("[-900,-746] result 0 (underflow)", -900.0, -746.0),
        ("[-5000,-1100] result 0 (early out)", -5000.0, -1100.0),
        ("[700,1000] overflow side", 700.0, 1000.0),
    ];
    let mut out = Vec::new();
    for (name, lo, hi) in ranges {
        let x: Vec<f64> = u.iter().map(|t| lo + (hi - lo) * t).collect();
        let mut best = f64::MAX;
        let mut chk = 0.0;
        for _ in 0..7 {
            let t0 = Instant::now();
            let mut s = 0.0;
            for &v in &x {
                s += std::hint::black_box(v).exp();
            }
            chk += s;
            best = best.min(t0.elapsed().as_secs_f64());
        }
        out.push(format!(
            "{{\"range\": \"{name}\", \"lo\": {lo}, \"hi\": {hi}, \"ns_per_exp\": {:.3}, \"checksum_finite\": {}}}",
            best * 1e9 / n as f64,
            chk.is_finite()
        ));
    }
    println!("{{\"n\": {n}, \"passes\": 7, \"stat\": \"best\", \"cells\": [{}]}}", out.join(", "));
}

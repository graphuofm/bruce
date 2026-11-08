//! Diagnostic: slide a ROWS frame over (score, value) pairs read from a
//! little-endian f64 file [s0, v0, s1, v1, ...] with softagg-core's InvAcc,
//! refolding exactly when the core marks the state inexact (its running
//! error bound), and compare every frame with a fresh ScalarAcc fold.
//! Also checks the core's claim: |error| <= value_error_bound() whenever
//! the state is reported exact. Prints JSON lines.
//! Usage: guard_bound <file> <frame>
use softagg_core::{InvAcc, ScalarAcc};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("input file");
    let f: usize = args[2].parse().expect("frame");
    let x: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let rows: Vec<(f64, f64)> = x.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    let vmax = rows.iter().map(|r| r.1.abs()).fold(0.0, f64::max);
    for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0] {
        let mut a = InvAcc::new();
        let (mut refolds, mut maxrel, mut maxabs, mut violations, mut max_ratio) = (0u64, 0f64, 0f64, 0u64, 0f64);
        for i in 0..rows.len() {
            a.absorb(rows[i].0, rows[i].1, eps);
            if i >= f {
                a.remove(rows[i - f].0, rows[i - f].1, eps);
            }
            let lo = (i + 1).saturating_sub(f);
            if !a.is_exact() {
                a = InvAcc::new();
                for &(s, v) in &rows[lo..=i] {
                    a.absorb(s, v, eps);
                }
                refolds += 1;
            }
            let got = a.finalize().unwrap();
            let mut r = ScalarAcc::new();
            for &(s, v) in &rows[lo..=i] {
                r.absorb(s, v, eps);
            }
            let want = r.finalize().unwrap();
            let e = (got - want).abs();
            let bound = a.value_error_bound();
            if e > bound {
                violations += 1;
            }
            if bound > 0.0 {
                max_ratio = max_ratio.max(e / bound);
            }
            maxrel = maxrel.max(e / want.abs());
            maxabs = maxabs.max(e);
        }
        println!(
            "{{\"eps\":{eps:e},\"frame\":{f},\"rows\":{},\"vmax\":{vmax:e},\"refolds\":{refolds},\"max_rel_err\":{maxrel:e},\"max_abs_err\":{maxabs:e},\"max_abs_err_over_vmax\":{:e},\"bound_violations\":{violations},\"max_err_over_bound\":{max_ratio:e}}}",
            rows.len(),
            maxabs / vmax
        );
    }
}

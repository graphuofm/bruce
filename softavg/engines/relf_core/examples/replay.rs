//! State-size replay for the relF states (experiments/anchored/state_sizes.py).
//!
//! input: little-endian f64 triples (group, score, value) in stream order (a file)
//! usage:
//!   replay window <file> <k> <d>    per group, FIFO frames of k+1 rows (WinBand):
//!                                   prints frames, mean/p99/max held rows, mean frame rows,
//!                                   cutoff moves per row, max |value - exact| / max|v|
//!   replay groupby <file> <d>       per group, insert-only Band: final band sum, peak band max
use relf_core::{relf_ref, Band, WinBand};
use std::collections::HashMap;

fn read(path: &str) -> Vec<(u64, f64, f64)> {
    let b = std::fs::read(path).expect("read input");
    let f = |i: usize| f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
    (0..b.len() / 24).map(|r| (f(3 * r) as u64, f(3 * r + 1), f(3 * r + 2))).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rows = read(&a[2]);
    let mut by: HashMap<u64, Vec<(f64, f64)>> = HashMap::new();
    for (g, s, v) in &rows {
        by.entry(*g).or_default().push((*s, *v));
    }
    match a[1].as_str() {
        "window" => {
            let k: usize = a[3].parse().unwrap();
            let d: f64 = a[4].parse().unwrap();
            let vmax = rows.iter().map(|r| r.2.abs()).fold(0.0, f64::max);
            let (mut frames, mut held_sum, mut frame_sum, mut moves) = (0u64, 0u64, 0u64, 0u64);
            let mut hist: Vec<u64> = Vec::new();
            let mut maxerr = 0.0f64;
            let check_every = 97usize;
            let mut held_max = 0usize;
            for (_, xs) in by.iter() {
                let mut w = WinBand::new(d);
                for i in 0..xs.len() {
                    w.push(xs[i].0, xs[i].1);
                    if i > k {
                        let o = xs[i - k - 1];
                        w.pop(o.0, o.1).unwrap();
                    }
                    frames += 1;
                    held_sum += w.len() as u64;
                    held_max = held_max.max(w.len());
                    frame_sum += w.frame_rows();
                    hist.push(w.len() as u64);
                    if i % check_every == 0 {
                        let lo = i.saturating_sub(k);
                        let e = (w.value().unwrap() - relf_ref(&xs[lo..=i], d).unwrap()).abs() / vmax;
                        maxerr = maxerr.max(e);
                    }
                }
                moves += w.cut_moves;
            }
            hist.sort_unstable();
            let p99 = hist[((hist.len() as f64) * 0.99) as usize - 1];
            println!(
                "{{\"k\":{k},\"d\":{d},\"frames\":{frames},\"held_mean\":{},\"held_p99\":{p99},\"held_max\":{held_max},\"frame_rows_mean\":{},\"cutoff_moves_per_row\":{},\"max_abs_err_over_vmax_sampled\":{maxerr:e}}}",
                held_sum as f64 / frames as f64,
                frame_sum as f64 / frames as f64,
                moves as f64 / frames as f64
            );
        }
        "groupby" => {
            let d: f64 = a[3].parse().unwrap();
            let (mut fin, mut peak_max, mut peak_sum) = (0usize, 0usize, 0usize);
            for (_, xs) in by.iter() {
                let mut b = Band::new(d);
                for (s, v) in xs {
                    b.absorb(*s, *v);
                }
                fin += b.len();
                peak_max = peak_max.max(b.peak);
                peak_sum += b.peak;
            }
            println!("{{\"d\":{d},\"final_band_sum\":{fin},\"peak_band_max\":{peak_max},\"peak_band_sum\":{peak_sum},\"rows\":{}}}", rows.len());
        }
        _ => panic!("mode"),
    }
}

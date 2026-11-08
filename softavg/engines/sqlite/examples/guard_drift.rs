//! Measures how far a sliding softavg (InvAcc + core guard + shadow refold)
//! drifts from a fresh fold, per eps and frame length. Diagnostic for
//! notes/CORE_CHANGES.md; prints JSON lines.
use softagg_core::ScalarAcc;
use softagg_sqlite::state::{self, WinState};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn main() {
    let mut r = Lcg(7);
    let n = 20_000;
    let rows: Vec<(f64, f64)> = (0..n).map(|_| (2.0 * r.next() - 1.0, 10.0 * r.next())).collect();
    for &eps in &[1e-4, 1e-3, 0.02, 0.1, 1.0, f64::INFINITY] {
        for &p in &[49usize, 199, 999] {
            state::stats_reset();
            let mut st = WinState::new();
            let (mut max_rel, mut max_abs_range) = (0f64, 0f64);
            for i in 0..n {
                st.check_eps(Some(eps)).unwrap();
                st.step(rows[i].0, rows[i].1);
                if i > p {
                    st.inverse(rows[i - p - 1].0, rows[i - p - 1].1).unwrap();
                }
                let got = st.value().unwrap().unwrap();
                let lo = i.saturating_sub(p);
                let mut a = ScalarAcc::new();
                for &(s, v) in &rows[lo..=i] { a.absorb(s, v, eps); }
                let want = a.finalize().unwrap();
                max_rel = max_rel.max((got - want).abs() / want.abs());
                max_abs_range = max_abs_range.max((got - want).abs() / 10.0);
            }
            println!("{{\"eps\":{eps:e},\"frame\":{},\"rows\":{n},\"max_rel_err\":{max_rel:e},\"max_abs_err_over_value_range\":{max_abs_range:e},\"stats\":{}}}",
                     p + 1, state::stats_json());
        }
    }
}

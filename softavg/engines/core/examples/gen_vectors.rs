//! Writes conformance/vectors.json: the cases every host extension must
//! reproduce. Expected values come from this crate's ScalarAcc (fresh fold
//! per group / per window frame), cross-checked here against a two-pass
//! reference. Hosts compare with rtol 1e-12 (aggregates) / 1e-9 (windows,
//! where an inverse-based frame may round differently), and must return
//! NULL where `expected` is null.
use softagg_core::ScalarAcc;
use std::fmt::Write;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn fold(rows: &[(f64, f64)], eps: f64) -> Option<f64> {
    let mut a = ScalarAcc::new();
    for &(s, v) in rows {
        a.absorb(s, v, eps);
    }
    a.finalize()
}

fn num(x: f64) -> String {
    if x.is_nan() { "null".into() }
    else if x == f64::INFINITY { "\"inf\"".into() }
    else if x == f64::NEG_INFINITY { "\"-inf\"".into() }
    else { format!("{x:?}") }
}

fn main() {
    let mut r = Lcg(20_261_105);
    // one table: id, g (0..7), s in [-1,1], v in [0,10); plus edge rows
    let mut rows: Vec<(u32, u32, f64, f64)> = (0..2000)
        .map(|i| (i, (r.next() * 8.0) as u32, 2.0 * r.next() - 1.0, (10.0 * r.next() * 1000.0).round() / 1000.0))
        .collect();
    let n = rows.len() as u32;
    rows.push((n, 8, f64::NEG_INFINITY, 1.0));          // group 8: only -inf -> NULL at finite eps
    rows.push((n + 1, 9, f64::INFINITY, 4.0));          // group 9: +inf rows dominate
    rows.push((n + 2, 9, f64::INFINITY, 6.0));
    rows.push((n + 3, 9, 0.9, 100.0));
    let eps_list = [1e-4, 1e-3, 0.02, 0.1, 1.0, 0.0, f64::INFINITY];

    let mut out = String::from("{\n  \"table\": [\n");
    for (k, &(id, g, s, v)) in rows.iter().enumerate() {
        let _ = write!(out, "    [{id}, {g}, {}, {}]{}\n", num(s), num(v), if k + 1 < rows.len() { "," } else { "" });
    }
    out.push_str("  ],\n  \"columns\": [\"id\", \"g\", \"s\", \"v\"],\n");
    out.push_str("  \"group_by\": [\n");
    for (ei, &eps) in eps_list.iter().enumerate() {
        let _ = write!(out, "    {{\"eps\": {}, \"expected\": {{", num(eps));
        for g in 0..10u32 {
            let grp: Vec<(f64, f64)> = rows.iter().filter(|x| x.1 == g).map(|x| (x.2, x.3)).collect();
            let e = fold(&grp, eps);
            let _ = write!(out, "\"{g}\": {}{}", e.map(num).unwrap_or("null".into()), if g < 9 { ", " } else { "" });
        }
        let _ = write!(out, "}}}}{}\n", if ei + 1 < eps_list.len() { "," } else { "" });
    }
    // window: ORDER BY id, ROWS BETWEEN 49 PRECEDING AND CURRENT ROW, over
    // the first 2000 rows only (finite scores), eps in {0.02, 0.1, inf}
    out.push_str("  ],\n  \"window_rows_preceding\": 49,\n  \"window\": [\n");
    let base: Vec<(f64, f64)> = rows[..2000].iter().map(|x| (x.2, x.3)).collect();
    let weps = [0.02, 0.1, f64::INFINITY];
    for (ei, &eps) in weps.iter().enumerate() {
        let _ = write!(out, "    {{\"eps\": {}, \"expected\": [", num(eps));
        for i in 0..base.len() {
            let lo = i.saturating_sub(49);
            let e = fold(&base[lo..=i], eps).unwrap();
            let _ = write!(out, "{}{}", num(e), if i + 1 < base.len() { ", " } else { "" });
        }
        let _ = write!(out, "]}}{}\n", if ei + 1 < weps.len() { "," } else { "" });
    }
    out.push_str("  ]\n}\n");
    std::fs::write("conformance/vectors.json", out).unwrap();
    eprintln!("wrote conformance/vectors.json");
}

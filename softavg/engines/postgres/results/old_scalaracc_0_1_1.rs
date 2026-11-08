// Verbatim 0.1.1 ScalarAcc (no +/-Inf branches, no NaN bit) — what the
// shipped extension returned before the pg-parity fix.
#[derive(Clone, Copy, Debug)]
struct Old { mu: f64, z: f64, u: f64 }
impl Old {
    fn new() -> Self { Old { mu: f64::NEG_INFINITY, z: 0.0, u: 0.0 } }
    fn is_empty(&self) -> bool { self.z == 0.0 }
    fn absorb(&mut self, s: f64, v: f64, eps: f64) {
        if eps == 0.0 {
            if s > self.mu { self.mu = s; self.z = 1.0; self.u = v; }
            else if s == self.mu { self.z += 1.0; self.u += v; }
            return;
        }
        if eps.is_infinite() { self.z += 1.0; self.u += v; return; }
        if self.is_empty() { self.mu = s; self.z = 1.0; self.u = v; return; }
        let mu2 = self.mu.max(s);
        let scale = ((self.mu - mu2) / eps).exp();
        let w = ((s - mu2) / eps).exp();
        self.u = self.u * scale + w * v;
        self.z = self.z * scale + w;
        self.mu = mu2;
    }
    fn finalize(&self) -> Option<f64> { if self.is_empty() { None } else { Some(self.u / self.z) } }
}
fn run(name: &str, rows: &[(f64, f64)], eps: f64, want: &str) {
    let mut a = Old::new();
    for &(s, v) in rows { a.absorb(s, v, eps); }
    println!("{name:<46} eps={eps:<6} 0.1.1 -> {:?}   0.1.2 -> {want}", a.finalize());
}
fn main() {
    const I: f64 = f64::INFINITY;
    const N: f64 = f64::NEG_INFINITY;
    const NAN: f64 = f64::NAN;
    for eps in [0.37f64, 1.0] {
        run("+Inf row next to finite rows", &[(1.0,10.0),(I,55.0),(2.0,20.0)], eps, "55.0");
        run("two +Inf rows (tie)", &[(I,50.0),(3.0,999.0),(I,60.0)], eps, "55.0");
        run("-Inf row next to a finite row", &[(N,888.0),(1.0,10.0)], eps, "10.0");
        run("all -Inf rows", &[(N,888.0),(N,999.0)], eps, "None (SQL NULL)");
    }
    run("all -Inf rows", &[(N,888.0),(N,999.0)], 0.0, "None (SQL NULL)");
    run("-Inf row next to a finite row", &[(N,888.0),(1.0,10.0)], 0.0, "10.0");
    run("NaN score + finite rows", &[(1.0,10.0),(NAN,777.0),(2.0,20.0)], 0.0, "NaN");
    run("NaN score + finite rows", &[(1.0,10.0),(NAN,777.0),(2.0,20.0)], 0.37, "NaN");
    run("NaN value + finite rows", &[(1.0,10.0),(0.9,NAN),(2.0,20.0)], 0.0, "NaN");
    run("NaN value + finite rows", &[(1.0,10.0),(0.9,NAN),(2.0,20.0)], f64::INFINITY, "NaN");
}

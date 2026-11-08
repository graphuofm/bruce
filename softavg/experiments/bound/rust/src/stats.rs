//! Mergeable statistics: a log10 histogram for ratio/error distributions.
use crate::json::J;

const LO: f64 = -26.0;
const HI: f64 = 4.0;
const PER_DECADE: f64 = 50.0;

#[derive(Clone)]
pub struct LogHist {
    pub bins: Vec<u64>,
    pub zeros: u64,
    pub count: u64,
    pub max: f64,
    pub above: u64, // values above 10^HI (or inf)
}

impl Default for LogHist {
    fn default() -> Self {
        Self::new()
    }
}

impl LogHist {
    pub fn new() -> Self {
        let nb = ((HI - LO) * PER_DECADE) as usize;
        Self { bins: vec![0; nb], zeros: 0, count: 0, max: 0.0, above: 0 }
    }
    #[inline]
    pub fn add(&mut self, x: f64) {
        self.count += 1;
        if x > self.max || x.is_nan() {
            self.max = if x.is_nan() { f64::INFINITY } else { x };
        }
        if x <= 0.0 {
            self.zeros += 1;
            return;
        }
        let l = x.log10();
        if !(l < HI) {
            self.above += 1;
            return;
        }
        let i = (((l - LO) * PER_DECADE).floor().max(0.0)) as usize;
        self.bins[i] += 1;
    }
    pub fn merge(&mut self, o: &LogHist) {
        for (a, b) in self.bins.iter_mut().zip(&o.bins) {
            *a += b;
        }
        self.zeros += o.zeros;
        self.count += o.count;
        self.above += o.above;
        self.max = self.max.max(o.max);
    }
    /// upper edge of the bin holding the p-quantile (0 if it falls on zeros)
    pub fn quantile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let target = (p * self.count as f64).ceil().max(1.0) as u64;
        let mut c = self.zeros;
        if c >= target {
            return 0.0;
        }
        for (i, &b) in self.bins.iter().enumerate() {
            c += b;
            if c >= target {
                let edge = 10f64.powf(LO + (i as f64 + 1.0) / PER_DECADE);
                return edge.min(self.max);
            }
        }
        self.max
    }
    pub fn to_json(&self) -> J {
        J::obj(vec![
            ("count", J::I(self.count as i64)),
            ("zeros", J::I(self.zeros as i64)),
            ("median", J::N(self.quantile(0.5))),
            ("p90", J::N(self.quantile(0.9))),
            ("p99", J::N(self.quantile(0.99))),
            ("p999", J::N(self.quantile(0.999))),
            ("max", J::N(self.max)),
        ])
    }
    /// compact histogram for plotting: (log10 lower edge, count) of non-empty bins
    pub fn bins_json(&self) -> J {
        let mut v = Vec::new();
        for (i, &b) in self.bins.iter().enumerate() {
            if b > 0 {
                v.push(J::A(vec![J::N(LO + i as f64 / PER_DECADE), J::I(b as i64)]));
            }
        }
        J::obj(vec![
            ("per_decade", J::N(PER_DECADE)),
            ("zeros", J::I(self.zeros as i64)),
            ("above", J::I(self.above as i64)),
            ("bins", J::A(v)),
        ])
    }
}

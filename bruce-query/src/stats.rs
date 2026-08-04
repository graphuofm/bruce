//! Statistics layer: what the temperature-aware optimizer estimates.
//!
//! Classical stats (row counts, min/max, histograms, group counts)
//! answer the classical question: how many rows survive a predicate.
//! The new object is the *weight-concentration sketch*: a deterministic
//! row sample of the key column from which, for ANY query vector and
//! ANY temperature, the planner estimates how concentrated the softmax
//! weights are — the quantity that decides whether a truncated
//! (top-k) plan is admissible under a declared error budget, and what
//! k it needs (`k*`).
//!
//! The estimator is honest about its own failure mode: at very sharp
//! temperatures the top of the weight distribution is an
//! extreme-value statistic that a uniform sample under-resolves, so
//! the estimate carries a `resolution_limited` flag and the planner
//! falls back to the exact plan rather than trust it (conservative by
//! construction).

use ndarray::ArrayView1;

use crate::catalog::{Column, Table};
use crate::logical::Pred;

/// Number of equi-width histogram buckets for scalar columns.
const HIST_BUCKETS: usize = 64;

/// Statistics for one scalar (f64) column.
#[derive(Debug, Clone)]
pub struct ScalarStats {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Equi-width histogram counts over `[min, max]`.
    pub hist: Vec<u64>,
    /// Row count.
    pub n: usize,
}

impl ScalarStats {
    fn collect(v: &[f64]) -> Self {
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &x in v {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
        let mut hist = vec![0u64; HIST_BUCKETS];
        if v.is_empty() || min >= max {
            return Self {
                min,
                max,
                hist,
                n: v.len(),
            };
        }
        let w = (max - min) / HIST_BUCKETS as f64;
        for &x in v {
            let b = (((x - min) / w) as usize).min(HIST_BUCKETS - 1);
            hist[b] += 1;
        }
        Self {
            min,
            max,
            hist,
            n: v.len(),
        }
    }

    /// Estimated selectivity of a predicate over this column.
    pub fn selectivity(&self, p: &Pred) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        match p {
            Pred::GtEq(_, k) => {
                if *k <= self.min {
                    1.0
                } else if *k > self.max {
                    0.0
                } else {
                    let w = (self.max - self.min) / HIST_BUCKETS as f64;
                    let b = (((k - self.min) / w) as usize).min(HIST_BUCKETS - 1);
                    // full buckets above b, plus a linear fraction of bucket b
                    let above: u64 = self.hist[b + 1..].iter().sum();
                    let frac_in_b = 1.0 - ((k - self.min) - b as f64 * w) / w;
                    (above as f64 + self.hist[b] as f64 * frac_in_b.clamp(0.0, 1.0)) / self.n as f64
                }
            }
            Pred::Eq(_, _) => {
                // point predicate on a continuous column: one bucket's
                // uniform share (dictionary columns answer this better)
                let nonzero = self.hist.iter().filter(|&&c| c > 0).count().max(1);
                1.0 / (nonzero as f64 * HIST_BUCKETS as f64 / nonzero as f64).max(1.0)
            }
        }
    }
}

/// Statistics for a dictionary-encoded group column.
#[derive(Debug, Clone)]
pub struct DictStats {
    /// Number of distinct groups.
    pub n_groups: usize,
    /// Rows per group code.
    pub group_counts: Vec<u64>,
}

/// The weight-concentration sketch for a key (embedding) column: a
/// deterministic uniform row sample. Query- and temperature-agnostic
/// at collection time; scored on demand at estimation time.
#[derive(Debug, Clone)]
pub struct KeySketch {
    /// Sampled row indices (deterministic stride, no RNG).
    pub rows: Vec<usize>,
    /// The sampled key rows, flattened `(len(rows), d)`.
    pub keys: ndarray::Array2<f64>,
    /// Max |entry| over the whole column (bound for value ranges when
    /// keys double as values, e.g. attention reads).
    pub abs_max: f64,
    /// Total rows in the column at collection time.
    pub n_total: usize,
}

/// What the sketch says about a (query, eps, budget) triple.
#[derive(Debug, Clone)]
pub struct ContractEstimate {
    /// Estimated smallest k meeting the budget (scaled to the table).
    pub kstar: usize,
    /// Estimated omitted weight mass at that k.
    pub delta: f64,
    /// Certified-bound value `delta*(1+1/(1-delta))*vmax` at that k.
    pub bound: f64,
    /// Whether the sketch could resolve the answer at this eps: false
    /// when the whole budget fits inside fewer than
    /// `RESOLUTION_MIN_SAMPLES` sample points, i.e. the extreme-value
    /// regime a uniform sample under-resolves.
    pub resolution_limited: bool,
}

/// Below this many sample points inside the admitted mass, the
/// estimate is extreme-value-limited and must not be trusted.
const RESOLUTION_MIN_SAMPLES: usize = 8;

impl KeySketch {
    fn collect(keys: &ndarray::Array2<f64>, sample: usize) -> Self {
        let n = keys.nrows();
        let take = sample.min(n).max(1);
        let stride = (n as f64 / take as f64).max(1.0);
        let rows: Vec<usize> = (0..take)
            .map(|i| ((i as f64 + 0.5) * stride) as usize)
            .map(|r| r.min(n - 1))
            .collect();
        let d = keys.ncols();
        let mut sk = ndarray::Array2::<f64>::zeros((rows.len(), d));
        for (i, &r) in rows.iter().enumerate() {
            sk.row_mut(i).assign(&keys.row(r));
        }
        let abs_max = keys.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        Self {
            rows,
            keys: sk,
            abs_max,
            n_total: n,
        }
    }

    /// f32 storage variant: sketch statistics stay f64 — only the
    /// bounded sample (`sample * d` values) is upcast, never the
    /// column.
    fn collect_f32(keys: &ndarray::Array2<f32>, sample: usize) -> Self {
        let n = keys.nrows();
        let take = sample.min(n).max(1);
        let stride = (n as f64 / take as f64).max(1.0);
        let rows: Vec<usize> = (0..take)
            .map(|i| ((i as f64 + 0.5) * stride) as usize)
            .map(|r| r.min(n - 1))
            .collect();
        let d = keys.ncols();
        let mut sk = ndarray::Array2::<f64>::zeros((rows.len(), d));
        for (i, &r) in rows.iter().enumerate() {
            for c in 0..d {
                sk[(i, c)] = keys[(r, c)] as f64;
            }
        }
        let abs_max = keys.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()));
        Self {
            rows,
            keys: sk,
            abs_max,
            n_total: n,
        }
    }

    /// Score the sketch against a query vector: sampled sims, sorted
    /// descending. This is the only per-query work the sketch does.
    ///
    /// NaN similarities (a NaN key row or query entry — NaN is the
    /// engine's NULL encoding, and bruce-core mask.rs skips NaN rows
    /// the same way) are dropped from the sample rather than sorted:
    /// the previous `partial_cmp().unwrap()` panicked on them
    /// (tests/register_safety.rs). May return fewer than
    /// `rows.len()` entries; can be empty when every sim is NaN.
    pub fn sims(&self, x: &ArrayView1<'_, f64>) -> Vec<f64> {
        let mut s: Vec<f64> = self
            .keys
            .rows()
            .into_iter()
            .map(|r| r.dot(x))
            .filter(|v| !v.is_nan())
            .collect();
        s.sort_by(|a, b| b.total_cmp(a));
        s
    }

    /// Estimate the contract for `(x, eps, budget, vmax)` over a
    /// population of `n_rows` rows (post-selection). `budget` is the
    /// absolute error tolerance on the read; `vmax` bounds |values|.
    pub fn estimate_contract(
        &self,
        x: &ArrayView1<'_, f64>,
        eps: f64,
        budget: f64,
        vmax: f64,
        n_rows: usize,
    ) -> ContractEstimate {
        let s = self.sims(x);
        // Every sampled sim was NaN (NULL-encoded keys): the sketch
        // has nothing to certify with. Say so via
        // `resolution_limited` — the planner then refuses the
        // contract and falls back to the exact plan, conservative by
        // construction (tests/register_safety.rs).
        if s.is_empty() {
            return ContractEstimate {
                kstar: n_rows,
                delta: 1.0,
                bound: f64::INFINITY,
                resolution_limited: true,
            };
        }
        let m = s[0];
        let w: Vec<f64> = s.iter().map(|&si| ((si - m) / eps).exp()).collect();
        let z: f64 = w.iter().sum();
        // walk the sample's mass curve; find the first sample index
        // whose omitted mass certifies the budget
        let mut acc = 0.0;
        let mut hit: Option<(usize, f64)> = None;
        for (i, &wi) in w.iter().enumerate() {
            acc += wi;
            let delta = 1.0 - acc / z;
            let bound = delta * (1.0 + 1.0 / (1.0 - delta).max(1e-12)) * vmax;
            if bound <= budget {
                hit = Some((i + 1, delta));
                break;
            }
        }
        let scale = n_rows as f64 / self.rows.len().max(1) as f64;
        match hit {
            Some((k_samples, delta)) => {
                let bound = delta * (1.0 + 1.0 / (1.0 - delta).max(1e-12)) * vmax;
                ContractEstimate {
                    kstar: ((k_samples as f64) * scale).ceil() as usize,
                    delta,
                    bound,
                    resolution_limited: k_samples < RESOLUTION_MIN_SAMPLES,
                }
            }
            None => ContractEstimate {
                kstar: n_rows,
                delta: 0.0,
                bound: 0.0,
                resolution_limited: false,
            },
        }
    }
}

/// Statistics for one table.
#[derive(Debug, Clone, Default)]
pub struct TableStats {
    /// Scalar column stats by name.
    pub scalars: std::collections::HashMap<String, ScalarStats>,
    /// Dict column stats by name.
    pub dicts: std::collections::HashMap<String, DictStats>,
    /// Key sketches by column name.
    pub keys: std::collections::HashMap<String, KeySketch>,
    /// Row count.
    pub n_rows: usize,
}

impl TableStats {
    /// Collect stats over a table. `sample` bounds the key sketch size.
    pub fn collect(t: &Table, sample: usize) -> Self {
        let mut s = TableStats::default();
        for (name, col) in &t.columns {
            s.n_rows = col.len();
            match col {
                Column::ScalarF64(v) => {
                    s.scalars.insert(name.clone(), ScalarStats::collect(v));
                }
                Column::DictU32 { codes, dict } => {
                    let mut counts = vec![0u64; dict.len()];
                    for &c in codes {
                        // Guard (tests/register_safety.rs): codes are
                        // declared in [0, dict.len()), but Table is a
                        // public struct — a hand-built column can
                        // arrive with dangling codes (the columnar
                        // analogue of a broken FK from codes to dict;
                        // PG ANALYZE never errors on sampled rows).
                        // A dangling row's label is unknowable, so it
                        // is attributed to no group: counts under-
                        // cover n_rows, which is conservative for
                        // selectivity. Query-time layers report the
                        // corruption as typed errors; stats stay
                        // total.
                        if let Some(slot) = counts.get_mut(c as usize) {
                            *slot += 1;
                        }
                    }
                    s.dicts.insert(
                        name.clone(),
                        DictStats {
                            n_groups: dict.len(),
                            group_counts: counts,
                        },
                    );
                }
                Column::KeyF64(a) => {
                    s.keys.insert(name.clone(), KeySketch::collect(a, sample));
                }
                Column::KeyF32(a) => {
                    s.keys
                        .insert(name.clone(), KeySketch::collect_f32(a, sample));
                }
            }
        }
        s
    }

    /// Estimated selectivity of a predicate (1.0 when unknown).
    pub fn selectivity(&self, p: &Pred) -> f64 {
        self.scalars
            .get(p.column())
            .map(|cs| cs.selectivity(p))
            .unwrap_or(1.0)
    }
}

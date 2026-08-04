//! The cost model: a bandwidth model with kernel constants, calibrated
//! from the M1 measurements (grouped_softavg over 459,865 x 384 f64
//! rows: 13.0 ms fused single-socket run; the fold is
//! memory-bandwidth-bound at ~55 GB/s on the reference node —
//! paper_sigmod_bruce/experiments/m1_grouped_kernel/results_m1.json).
//!
//! cost_seconds = bytes_touched / bandwidth
//!              + rows * row_overhead
//!              + groups * group_overhead
//!
//! The model's job is ORDERING plans, not clock prediction; the
//! calibration test asserts ranking fidelity and coarse magnitude
//! (within ~3x), which is what plan choice needs.

/// Planner admission tolerance for the HNSW-served top-k path
/// (tests/topk_access_path.rs): `HnswTopKScan` carries no declared
/// error budget, so it is admitted only when the predicted omitted
/// softmax MASS is <= this bound, and the executor re-checks the
/// achieved bound at runtime (falling back to the exact fused fold
/// when the probe misses it).
///
/// Derivation (mass -> answer error). Let delta be the omitted
/// softmax mass, Z = z_top + z_tail the full partition sum,
/// delta = z_tail / Z, and vmax = max |value|. Then
///
///   |num/Z - num_top/z_top|
///     = |z_top*num_tail - num_top*z_tail| / (z_top * Z)
///    <= (z_top*vmax*z_tail + vmax*z_top*z_tail) / (z_top * Z)
///     = 2 * vmax * delta,
///
/// i.e. omitting mass delta perturbs the answer by at most
/// 2*delta*vmax (the same algebra behind TopKContractScan's
/// `delta*(1 + 1/(1-delta))*vmax` certified bound, of which
/// 2*delta*vmax is the small-delta limit and a lower bound).
///
/// "Within the engine precision contract": the tightest per-eps
/// ceiling the f32/f64 precision contract pins is 1e-5 relative
/// error (bruce-core tests/numerical_edges.rs, workstream 4; the
/// sharp-eps rows this path serves are pinned at 1e-3..1e-4). With
/// delta <= 1e-6 the induced error is <= 2e-6 * vmax — under the
/// 1e-5 contract ceiling for any answer of magnitude >= 0.2*vmax,
/// with >= 100x headroom at the sharp temperatures this path is
/// actually admitted for. Conservative on the mass side because the
/// bound `n * exp((s_k - s_max)/eps)` (see [`hnsw_tail_bound`])
/// multiplies by the full row count.
pub const HNSW_TAIL_TOL: f64 = 1e-6;

/// Upper bound on the softmax mass of every row scoring at most
/// `s_k`, out of `n` rows, when some row scores `s_max`.
///
/// Derivation (max-anchored weight form, mask.rs semantics): anchor
/// weights at the maximum, w_i = exp((s_i - s_max)/eps), so the
/// anchor row has w = 1 and Z = sum_i w_i >= 1. If every omitted row
/// has s_i <= s_k, each omitted weight is <= exp((s_k - s_max)/eps)
/// and there are at most n of them, hence
///
///   omitted mass = (sum_omitted w_i) / Z
///                <= n * exp((s_k - s_max)/eps) / 1.
///
/// Deliberately conservative: uses n (not n - k) and Z >= 1 (not
/// Z >= z_top). CAVEAT the caller must own: the bound is exact only
/// under the containment assumption that no omitted row scores above
/// s_k. An approximate index can miss a high scorer (a recall miss);
/// the planner mitigates with an ef >> k probe margin and the
/// differential suite (tests/topk_access_path.rs) measures the
/// achieved error against this bound.
///
/// Total on all inputs: any NaN input or eps <= 0 returns +inf (the
/// caller then refuses / falls back — conservative), matching the
/// no-panic doctrine.
pub fn hnsw_tail_bound(n: f64, s_k: f64, s_max: f64, eps: f64) -> f64 {
    // NaN eps also lands in the refusing arm (NaN fails the >).
    if eps <= 0.0 || eps.is_nan() || n.is_nan() || s_k.is_nan() || s_max.is_nan() {
        return f64::INFINITY;
    }
    n * ((s_k - s_max) / eps).exp()
}

/// Sketch-based prediction of the softmax mass omitted by a top-`k`
/// probe, for planner admission of `HnswTopKScan`. Lives here (not
/// stats.rs) by tonight's ownership split; reads only `KeySketch`'s
/// public surface.
#[derive(Debug, Clone)]
pub struct HnswTailPrediction {
    /// Predicted omitted-mass BOUND after the top-k rows, in the same
    /// `n * exp((s_k - s_max)/eps)` form the executor re-checks —
    /// planner and runtime must agree on the form, or the planner
    /// admits probes the runtime then rejects (systematic fallback ==
    /// pure regret).
    pub tail: f64,
    /// The sample under-resolves rank k (extreme-value regime): too
    /// few samples land inside the top-k for a quantile estimate.
    /// The planner must not trust `tail` and refuses the index path
    /// (same convention as `ContractEstimate`).
    pub resolution_limited: bool,
}

/// Predict [`hnsw_tail_bound`] for a top-`k` probe from the key
/// sketch: with sampled sims s_0 >= s_1 >= ... (each sample
/// representing `scale = n_total / m` population rows, sample j
/// estimating population rank ~ (j + 0.5) * scale), take
///
///   s_hat_max = s_0            (sample max UNDER-estimates the true
///                               max -> widens the gap denominator's
///                               exponent -> conservative)
///   s_hat_k   = s_{ks - 1},    ks = floor(k / scale)
///                              (rank ~ (ks - 0.5) * scale <= k, an
///                               OVER-estimate of the true k-th best
///                               -> conservative)
///
/// and predict `hnsw_tail_bound(n_eff, s_hat_k, s_hat_max, eps)`.
/// Both quantile choices push the predicted bound UP, so admission
/// errs toward refusal.
///
/// Resolution honesty: `ks < 2` means rank k sits inside the sample's
/// first stride — an extreme-value statistic a uniform sample cannot
/// resolve — so the prediction is flagged and the planner refuses.
/// Under a filter the quantiles come from the unfiltered sample
/// (filter/sim independence assumed at plan time; `n_eff` is already
/// selectivity-scaled); the executor's runtime re-check enforces the
/// actual achieved bound either way.
pub fn predict_hnsw_tail(
    sketch: &crate::stats::KeySketch,
    x: &ndarray::ArrayView1<'_, f64>,
    eps: f64,
    k: usize,
    n_eff: f64,
) -> HnswTailPrediction {
    predict_hnsw_tail_from_sims(&sketch.sims(x), sketch.n_total, eps, k, n_eff)
}

/// [`predict_hnsw_tail`] from pre-scored sketch sims (sorted
/// descending, as `KeySketch::sims` returns them) — the planner walks
/// a k ladder and must not re-dot the whole sketch per rung.
pub fn predict_hnsw_tail_from_sims(
    s: &[f64],
    n_total: usize,
    eps: f64,
    k: usize,
    n_eff: f64,
) -> HnswTailPrediction {
    if s.len() < 2 || eps <= 0.0 || eps.is_nan() {
        return HnswTailPrediction {
            tail: f64::INFINITY,
            resolution_limited: true,
        };
    }
    let scale = n_total as f64 / s.len() as f64;
    let ks = (k as f64 / scale).floor() as usize;
    if ks < 2 {
        return HnswTailPrediction {
            tail: f64::INFINITY,
            resolution_limited: true,
        };
    }
    let s_hat_k = s[(ks - 1).min(s.len() - 1)];
    HnswTailPrediction {
        tail: hnsw_tail_bound(n_eff, s_hat_k, s[0], eps),
        resolution_limited: false,
    }
}

/// Calibrated model constants.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Effective memory bandwidth, bytes/second.
    pub bandwidth: f64,
    /// Per-row kernel overhead, seconds (branching, exp, accumulate).
    pub row_overhead: f64,
    /// Per-group overhead, seconds (state init + output).
    pub group_overhead: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        // M1: 460k rows x 384 d x 8 B = 1.41 GB keys (+ ~7 MB
        // vals/codes) in 13.0 ms single-pass => ~2.4 ms of the budget
        // is non-stream overhead at 55 GB/s. Attribute it per row.
        CostModel {
            bandwidth: 55.0e9,
            row_overhead: 5.0e-9,
            group_overhead: 2.0e-6,
        }
    }
}

/// A cost estimate with its drivers, for EXPLAIN.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Predicted seconds.
    pub seconds: f64,
    /// Bytes the plan streams.
    pub bytes: f64,
    /// Rows the plan touches after selection.
    pub rows: f64,
    /// One-line driver note for EXPLAIN.
    pub note: String,
}

impl CostModel {
    /// Cost of the fused grouped soft-average scan: streams the key
    /// column (d * 8 B/row), the value column, the code column, and
    /// the selection column when present. Selection is fused, so
    /// filtered rows still stream their key bytes but skip the exp.
    pub fn fused_group_scan(
        &self,
        n_rows: f64,
        selectivity: f64,
        d: usize,
        n_groups: f64,
    ) -> CostEstimate {
        let bytes = n_rows * (d as f64 * 8.0 + 8.0 + 4.0 + 1.0);
        let rows = n_rows * selectivity;
        let seconds =
            bytes / self.bandwidth + rows * self.row_overhead + n_groups * self.group_overhead;
        CostEstimate {
            seconds,
            bytes,
            rows,
            note: format!("streams keys ({d} f64/row); sel fused"),
        }
    }

    /// Cost of the exact endpoint scan (eps = inf uniform average, or
    /// eps = 0 indicator equality): never touches the key column for
    /// eps = inf; touches it once for the equality compare at eps = 0.
    pub fn exact_group_scan(
        &self,
        n_rows: f64,
        selectivity: f64,
        d_key_read: usize,
        n_groups: f64,
    ) -> CostEstimate {
        let bytes = n_rows * (d_key_read as f64 * 8.0 + 8.0 + 4.0 + 1.0);
        let rows = n_rows * selectivity;
        let seconds = bytes / self.bandwidth
            + rows * self.row_overhead * 0.5
            + n_groups * self.group_overhead;
        CostEstimate {
            seconds,
            bytes,
            rows,
            note: if d_key_read == 0 {
                "no key read (uniform endpoint)".into()
            } else {
                "key read for equality compare only".into()
            },
        }
    }

    /// Cost of the contracted top-k plan WITHOUT an index: the sims
    /// must still stream every key row; only the fold shrinks. The
    /// model makes the honest point that under pure scan cost this
    /// candidate cannot beat the fused scan by much — the win arrives
    /// with an index or a maintained view.
    pub fn topk_contract_scan(
        &self,
        n_rows: f64,
        selectivity: f64,
        d: usize,
        kstar: f64,
        n_groups: f64,
    ) -> CostEstimate {
        let bytes = n_rows * (d as f64 * 8.0 + 4.0 + 1.0) + kstar * 8.0;
        let rows = n_rows * selectivity;
        let seconds = bytes / self.bandwidth
            + rows * self.row_overhead * 0.6
            + kstar * self.row_overhead
            + n_groups * self.group_overhead;
        CostEstimate {
            seconds,
            bytes,
            rows,
            note: format!("sims still stream all keys; fold shrinks to k*={kstar:.0}"),
        }
    }

    /// Cost of the HNSW-served top-k fold: the probe touches
    /// ~`ef * m0` candidate nodes (m0 = 32, the layer-0 degree of the
    /// default build), each a d x f32 dot on a RANDOM-access key —
    /// charged at 4x row overhead for the cache misses + heap ops —
    /// then k exact f64 rescores and a k-row fold. A fused filter
    /// still evaluates its scalar column over all rows (8 B/row).
    /// The point the model must get right for ORDERING: no term
    /// scales with n_rows * d — the index path's whole advantage.
    pub fn hnsw_topk_scan(
        &self,
        n_rows: f64,
        d: usize,
        k: usize,
        ef: usize,
        filtered: bool,
    ) -> CostEstimate {
        let visited = ((ef * 32) as f64).min(n_rows.max(1.0));
        let mut bytes = visited * (d as f64) * 4.0 + (k as f64) * (d as f64) * 8.0;
        if filtered {
            bytes += n_rows * 8.0; // sel column eval for the probe filter
        }
        let seconds = bytes / self.bandwidth
            + visited * self.row_overhead * 4.0
            + (k as f64) * self.row_overhead
            + self.group_overhead;
        CostEstimate {
            seconds,
            bytes,
            rows: k as f64,
            note: format!("index probe ~{visited:.0} nodes; fold k={k}"),
        }
    }

    /// Cost of serving from a maintained view: O(groups) read.
    pub fn maintained_view_scan(&self, n_groups: f64) -> CostEstimate {
        let bytes = n_groups * 24.0;
        let seconds = bytes / self.bandwidth + n_groups * self.group_overhead;
        CostEstimate {
            seconds,
            bytes,
            rows: 0.0,
            note: "O(groups) read of maintained (m,num,den)".into(),
        }
    }
}

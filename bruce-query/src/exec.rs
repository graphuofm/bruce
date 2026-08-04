//! Plan execution against the catalog (in-memory columns; the storage
//! milestone swaps the backing store, not this interface).

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_core::mask::{grouped_softavg, grouped_softavg_f32};
use bruce_core::Eps;

use crate::catalog::{Catalog, Column};
use crate::db::HnswIndexEntry;
use crate::logical::{Pred, SimKind};
use crate::physical::PhysicalPlan;
use crate::views::SoftAggView;
use crate::QueryError;

/// A query result: group label -> value (v1: single aggregate).
#[derive(Debug, Clone)]
pub struct GroupResult {
    /// Group label per covered group.
    pub labels: Vec<String>,
    /// Aggregate value per covered group (same order as `labels`).
    pub values: Vec<f64>,
}

/// Execute a physical plan. `params` binds `:name` query vectors;
/// `views` backs MaintainedViewScan. Kept for API stability: no
/// index registry, so a `HnswTopKScan` takes a typed Bind error —
/// use [`execute_with_indexes`] (what `Database::run` calls) to
/// serve index plans.
pub fn execute(
    plan: &PhysicalPlan,
    catalog: &Catalog,
    params: &HashMap<String, Array1<f64>>,
    views: &[SoftAggView],
) -> Result<GroupResult, QueryError> {
    execute_with_indexes(plan, catalog, params, views, &[])
}

/// Runtime diagnostics of one `HnswTopKScan` execution — the
/// EXPLAIN-ANALYZE side of the planner's admission bet, and what the
/// m6_hnsw regret grid records as "achieved" against the plan's
/// `predicted_tail`.
///
/// `achieved_tail` is the same `cost::hnsw_tail_bound` the executor
/// gates on, evaluated on the ACTUAL probe result (0.0 exactly when
/// the probe returned every admitted row, so no mass is omitted).
/// When `fell_back` is true the returned answer is the exact fused
/// fold — the plan's saving is gone, its semantics are not.
#[derive(Debug, Clone)]
pub struct HnswProbeStats {
    /// Rows the filter admitted (the fold's population).
    pub admitted_rows: usize,
    /// Probe hits that survived the filter re-check and the NaN skip.
    pub hits: usize,
    /// Best rescored similarity among the hits (NaN if none).
    pub s_max: f64,
    /// Worst rescored similarity among the hits (NaN if none).
    pub s_k: f64,
    /// Achieved omitted-mass bound (see above).
    pub achieved_tail: f64,
    /// The runtime re-check rejected the probe; the answer is the
    /// exact fused fold.
    pub fell_back: bool,
}

/// [`execute_with_indexes`] restricted to `HnswTopKScan`, returning
/// the probe's runtime diagnostics alongside the answer. Any other
/// plan variant is a typed Bind error (the caller asked the wrong
/// question). Used by the m6_hnsw regret grid and pinned in
/// tests/topk_access_path.rs.
pub fn execute_hnsw_with_stats(
    plan: &PhysicalPlan,
    catalog: &Catalog,
    params: &HashMap<String, Array1<f64>>,
    views: &[SoftAggView],
    indexes: &[HnswIndexEntry],
) -> Result<(GroupResult, HnswProbeStats), QueryError> {
    match plan {
        PhysicalPlan::HnswTopKScan { .. } => exec_hnsw(plan, catalog, params, views, indexes),
        _ => Err(QueryError::Bind(
            "execute_hnsw_with_stats takes an HnswTopKScan plan".into(),
        )),
    }
}

/// [`execute`] plus the HNSW index registry backing `HnswTopKScan`.
pub fn execute_with_indexes(
    plan: &PhysicalPlan,
    catalog: &Catalog,
    params: &HashMap<String, Array1<f64>>,
    views: &[SoftAggView],
    indexes: &[HnswIndexEntry],
) -> Result<GroupResult, QueryError> {
    match plan {
        PhysicalPlan::FusedGroupScan {
            table,
            group_col,
            val_col,
            score,
            eps,
            sel,
        } => {
            let t = get_table(catalog, table)?;
            let (codes, dict) = dict_col(t, group_col)?;
            let vals = scalar_col(t, val_col)?;
            let keys = keys_col(t, &score.key_col)?;
            if score.kind != SimKind::Dot {
                return Err(QueryError::Bind(
                    "v2 executes Dot scores; NegSq/Indicator arrive with the operator matrix"
                        .into(),
                ));
            }
            let x = param(params, &score.param)?;
            let sel_mask: Option<Vec<bool>> = sel.as_ref().map(|p| eval_pred(p, t)).transpose()?;
            let v2 = Array2::from_shape_fn((vals.len(), 1), |(r, _)| vals[r]);
            let eps = Eps::new(*eps).map_err(|e| QueryError::Exec(e.to_string()))?;
            let (out, covered) = match keys {
                Keys::F64(keys) => grouped_softavg(
                    &x.view(),
                    &keys.view(),
                    &v2.view(),
                    codes,
                    dict.len(),
                    sel_mask.as_deref(),
                    eps,
                ),
                Keys::F32(keys) => {
                    // the bound :param arrives as f64; the f32 kernel
                    // scores in storage precision, so the vector is
                    // cast down once (d values) — the same rounding
                    // the encoder applied to the stored keys
                    let x32 = x.mapv(|v| v as f32);
                    grouped_softavg_f32(
                        &x32.view(),
                        &keys.view(),
                        &v2.view(),
                        codes,
                        dict.len(),
                        sel_mask.as_deref(),
                        eps,
                    )
                }
            }
            .map_err(|e| QueryError::Exec(e.to_string()))?;
            collect(dict, &covered, |g| out[(g, 0)])
        }

        PhysicalPlan::ExactGroupAvg {
            table,
            group_col,
            val_col,
            sel,
        } => {
            let t = get_table(catalog, table)?;
            let (codes, dict) = dict_col(t, group_col)?;
            let vals = scalar_col(t, val_col)?;
            let sel_mask: Option<Vec<bool>> = sel.as_ref().map(|p| eval_pred(p, t)).transpose()?;
            let mut sum = vec![0.0f64; dict.len()];
            let mut cnt = vec![0u64; dict.len()];
            for r in 0..codes.len() {
                if sel_mask.as_ref().map(|m| m[r]).unwrap_or(true) {
                    sum[codes[r] as usize] += vals[r];
                    cnt[codes[r] as usize] += 1;
                }
            }
            let covered: Vec<bool> = cnt.iter().map(|&c| c > 0).collect();
            collect(dict, &covered, |g| sum[g] / cnt[g] as f64)
        }

        PhysicalPlan::TopKContractScan {
            table,
            group_col,
            val_col,
            score,
            eps,
            sel,
            budget,
            est_kstar,
            est_delta: _,
        } => {
            // Per-group contracted fold with a RUNTIME guard. The
            // estimate chose this plan; execution must never trade
            // semantics for it: every group survives (k_g >= 1), the
            // true omitted mass per group is computed from the sims
            // we streamed anyway, and any group whose certified bound
            // misses the budget is re-folded exactly. The plan's
            // saving is the value-column bytes and fold work on the
            // rows it skips; its guard costs one comparison per group.
            let t = get_table(catalog, table)?;
            let (codes, dict) = dict_col(t, group_col)?;
            let vals = scalar_col(t, val_col)?;
            let keys = keys_col(t, &score.key_col)?;
            let x = param(params, &score.param)?;
            // Executor totality guard (tests/topk_access_path.rs): a
            // HAND-BUILT plan with a dim-mismatched bound param used to
            // panic inside ndarray's dot in the sims loop (f64 keys) or
            // zip-truncate silently (f32 keys). `Database::run` rejects
            // the mismatch earlier; `execute` is pub and must be total
            // on its own.
            check_param_dim(&keys, x, &score.param, &score.key_col)?;
            let sel_mask: Option<Vec<bool>> = sel.as_ref().map(|p| eval_pred(p, t)).transpose()?;

            // one sims pass (streams all keys, as the cost model says);
            // f32 keys score in storage precision, widened per row —
            // the same contract as the fused kernel
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let sim_of = |r: usize| -> f64 {
                match keys {
                    Keys::F64(a) => a.row(r).dot(&x.view()),
                    Keys::F32(a) => {
                        a.row(r).iter().zip(&x32).map(|(k, q)| k * q).sum::<f32>() as f64
                    }
                }
            };
            let n_groups = dict.len();
            let mut by_group: Vec<Vec<(f64, usize)>> = vec![Vec::new(); n_groups];
            let mut n_sel = 0usize;
            for r in 0..codes.len() {
                if sel_mask.as_ref().map(|m| m[r]).unwrap_or(true) {
                    by_group[codes[r] as usize].push((sim_of(r), r));
                    n_sel += 1;
                }
            }
            let vmax = vals.iter().fold(0.0f64, |a, &v| a.max(v.abs()));

            let mut labels = Vec::new();
            let mut values = Vec::new();
            for (g, rows) in by_group.iter_mut().enumerate() {
                if rows.is_empty() {
                    continue;
                }
                // per-group k_g scaled from the global estimate
                let k_g = ((*est_kstar as f64) * rows.len() as f64 / n_sel.max(1) as f64)
                    .ceil()
                    .max(1.0) as usize;
                rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let m = rows[0].0;
                let z_full: f64 = rows.iter().map(|(s, _)| ((s - m) / eps).exp()).sum();
                let take = k_g.min(rows.len());
                let mut num = 0.0;
                let mut z_top = 0.0;
                for &(s, r) in rows.iter().take(take) {
                    let w = ((s - m) / eps).exp();
                    num += w * vals[r];
                    z_top += w;
                }
                let delta = 1.0 - z_top / z_full;
                let bound = delta * (1.0 + 1.0 / (1.0 - delta).max(1e-12)) * vmax;
                let answer = if bound <= *budget {
                    num / z_top
                } else {
                    // runtime guard: contract missed for this group;
                    // fold it exactly
                    let mut num_f = 0.0;
                    for &(s, r) in rows.iter() {
                        num_f += ((s - m) / eps).exp() * vals[r];
                    }
                    num_f / z_full
                };
                labels.push(dict[g].clone());
                values.push(answer);
            }
            Ok(GroupResult { labels, values })
        }

        PhysicalPlan::MaintainedViewScan { view } => {
            let v = views
                .iter()
                .find(|v| v.name == *view)
                .ok_or_else(|| QueryError::Bind(format!("no view {view}")))?;
            let t = get_table(catalog, &v.table)?;
            let (_, dict) = dict_col(t, &v.group_col)?;
            let mut labels = Vec::new();
            let mut values = Vec::new();
            for (g, val) in v.read() {
                labels.push(dict[g].clone());
                values.push(val);
            }
            Ok(GroupResult { labels, values })
        }

        PhysicalPlan::HnswTopKScan { .. } => {
            exec_hnsw(plan, catalog, params, views, indexes).map(|(r, _)| r)
        }
    }
}

/// The `HnswTopKScan` mechanism, split out so the answer and its
/// runtime diagnostics ([`HnswProbeStats`]) can be returned together;
/// [`execute_with_indexes`] drops the stats, `execute_hnsw_with_stats`
/// keeps them.
fn exec_hnsw(
    plan: &PhysicalPlan,
    catalog: &Catalog,
    params: &HashMap<String, Array1<f64>>,
    views: &[SoftAggView],
    indexes: &[HnswIndexEntry],
) -> Result<(GroupResult, HnswProbeStats), QueryError> {
    match plan {
        PhysicalPlan::HnswTopKScan {
            table,
            group_col,
            val_col,
            score,
            eps,
            sel,
            k,
            ef,
            predicted_tail: _,
        } => {
            // Mechanism, kept total on hand-built plans
            // (tests/topk_access_path.rs): every unsound shape the
            // planner would never emit takes a typed error, and the
            // achieved tail bound is re-checked at runtime with an
            // exact-fold fallback — the planner's admission is a
            // prediction, never a semantic contract.
            let t = get_table(catalog, table)?;
            let (codes, dict) = dict_col(t, group_col)?;
            let vals = scalar_col(t, val_col)?;
            let keys = keys_col(t, &score.key_col)?;
            if score.kind != SimKind::Dot {
                return Err(QueryError::Bind(
                    "v2 executes Dot scores; NegSq/Indicator arrive with the operator matrix"
                        .into(),
                ));
            }
            let x = param(params, &score.param)?;
            check_param_dim(&keys, x, &score.param, &score.key_col)?;
            let keys = match keys {
                Keys::F64(a) => a,
                Keys::F32(_) => {
                    return Err(QueryError::Bind(format!(
                        "column {} must be KeyF64 (index plans rescore in f64; \
                         KeyF32 indexing is the views convention's typed refusal)",
                        score.key_col
                    )))
                }
            };
            if !eps.is_finite() || *eps <= 0.0 {
                return Err(QueryError::Exec(format!(
                    "HnswTopKScan requires finite eps > 0 (got {eps}): the eps=0 \
                     tropical endpoint needs exact argmax ties and eps=inf reads \
                     no scores; both are other plans' shapes"
                )));
            }
            let entry = indexes
                .iter()
                .find(|i| i.table == *table && i.key_col == score.key_col)
                .ok_or_else(|| {
                    QueryError::Bind(format!("no hnsw index on {table}.{}", score.key_col))
                })?;
            if entry.row_len() != codes.len() {
                return Err(QueryError::Exec(format!(
                    "index on {table}.{} is out of sync: {} indexed rows vs {} table \
                     rows (catalog mutated behind the index)",
                    score.key_col,
                    entry.row_len(),
                    codes.len()
                )));
            }
            let sel_mask: Option<Vec<bool>> = sel.as_ref().map(|p| eval_pred(p, t)).transpose()?;
            let admitted = |r: usize| sel_mask.as_ref().map(|m| m[r]).unwrap_or(true);
            let n_eff = match &sel_mask {
                Some(m) => m.iter().filter(|&&a| a).count(),
                None => codes.len(),
            };
            if n_eff == 0 {
                return Ok((
                    GroupResult {
                        labels: Vec::new(),
                        values: Vec::new(),
                    },
                    HnswProbeStats {
                        admitted_rows: 0,
                        hits: 0,
                        s_max: f64::NAN,
                        s_k: f64::NAN,
                        achieved_tail: 0.0,
                        fell_back: false,
                    },
                ));
            }
            // Soundness gate: the fold is a GLOBAL top-k; serving a
            // GROUP BY shape from it would starve minority groups.
            // The admitted set must cover exactly one group.
            let mut the_group: Option<u32> = None;
            for (r, &c) in codes.iter().enumerate() {
                if admitted(r) {
                    match the_group {
                        None => the_group = Some(c),
                        Some(g) if g != c => {
                            return Err(QueryError::Exec(format!(
                                "HnswTopKScan serves single-group shapes only; \
                                 admitted rows of {table} cover groups {:?} and {:?} \
                                 (GROUP BY stays on FusedGroupScan in v1)",
                                dict[g as usize], dict[c as usize]
                            )));
                        }
                        Some(_) => {}
                    }
                }
            }
            let g = the_group.expect("n_eff > 0") as usize;

            // Probe: f32 query (the graph's storage precision), filter
            // admits by row through the id map. Exact f64 rescoring
            // follows for every hit.
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let filter = |id: u32| entry.row_of(id).map(&admitted).unwrap_or(false);
            let hits = entry
                .index
                .search(
                    &x32,
                    *k,
                    *ef,
                    sel_mask.as_ref().map(|_| &filter as &dyn Fn(u32) -> bool),
                )
                .map_err(|e| QueryError::Exec(format!("index probe: {e}")))?;
            let mut top: Vec<(f64, usize)> = Vec::with_capacity(hits.len());
            for (id, _s32) in hits {
                let Some(r) = entry.row_of(id) else { continue };
                if !admitted(r) {
                    continue; // unfiltered probe cannot check; re-check here
                }
                let s = keys.row(r).dot(&x.view());
                if s.is_nan() {
                    continue; // NaN = NULL encoding; skipped like the fused kernel
                }
                top.push((s, r));
            }

            // Runtime admission re-check (cost.rs hnsw_tail_bound):
            // fall back to the exact fused fold when the achieved
            // bound misses the tolerance — the plan's saving is gone
            // but its semantics never degrade.
            let exact_fallback = || {
                execute_with_indexes(
                    &PhysicalPlan::FusedGroupScan {
                        table: table.clone(),
                        group_col: group_col.clone(),
                        val_col: val_col.clone(),
                        score: score.clone(),
                        eps: *eps,
                        sel: sel.clone(),
                    },
                    catalog,
                    params,
                    views,
                    indexes,
                )
            };
            if top.is_empty() {
                // probe starved (or k = 0)
                return exact_fallback().map(|r| {
                    (
                        r,
                        HnswProbeStats {
                            admitted_rows: n_eff,
                            hits: 0,
                            s_max: f64::NAN,
                            s_k: f64::NAN,
                            achieved_tail: f64::INFINITY,
                            fell_back: true,
                        },
                    )
                });
            }
            let s_max = top
                .iter()
                .map(|&(s, _)| s)
                .fold(f64::NEG_INFINITY, f64::max);
            let s_k = top.iter().map(|&(s, _)| s).fold(f64::INFINITY, f64::min);
            let tail = if top.len() >= n_eff {
                0.0 // the probe returned every admitted row
            } else {
                crate::cost::hnsw_tail_bound(n_eff as f64, s_k, s_max, *eps)
            };
            let stats = HnswProbeStats {
                admitted_rows: n_eff,
                hits: top.len(),
                s_max,
                s_k,
                achieved_tail: tail,
                fell_back: tail > crate::cost::HNSW_TAIL_TOL,
            };
            if stats.fell_back {
                return exact_fallback().map(|r| (r, stats));
            }

            // Exact max-anchored fold over the k hits (f64, the same
            // (m, num, den) form as the kernel).
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for &(s, r) in &top {
                let w = ((s - s_max) / eps).exp();
                num += w * vals[r];
                den += w;
            }
            Ok((
                GroupResult {
                    labels: vec![dict[g].clone()],
                    values: vec![num / den],
                },
                stats,
            ))
        }
        _ => Err(QueryError::Bind(
            "exec_hnsw takes an HnswTopKScan plan".into(),
        )),
    }
}

fn collect(
    dict: &[String],
    covered: &[bool],
    val: impl Fn(usize) -> f64,
) -> Result<GroupResult, QueryError> {
    let mut labels = Vec::new();
    let mut values = Vec::new();
    for (g, cov) in covered.iter().enumerate() {
        if *cov {
            labels.push(dict[g].clone());
            values.push(val(g));
        }
    }
    Ok(GroupResult { labels, values })
}

fn get_table<'a>(c: &'a Catalog, name: &str) -> Result<&'a crate::catalog::Table, QueryError> {
    c.tables
        .get(name)
        .ok_or_else(|| QueryError::Bind(format!("no table {name}")))
}

fn dict_col<'a>(
    t: &'a crate::catalog::Table,
    name: &str,
) -> Result<(&'a Vec<u32>, &'a Vec<String>), QueryError> {
    crate::views::dict_col(t, name)
}

fn scalar_col<'a>(t: &'a crate::catalog::Table, name: &str) -> Result<&'a Vec<f64>, QueryError> {
    crate::views::scalar_col(t, name)
}

/// A resolved key column: the executor dispatches the scan kernel on
/// the storage dtype (KeyF64 -> `grouped_softavg`, KeyF32 ->
/// `grouped_softavg_f32`).
enum Keys<'a> {
    F64(&'a Array2<f64>),
    F32(&'a ndarray::Array2<f32>),
}

fn keys_col<'a>(t: &'a crate::catalog::Table, name: &str) -> Result<Keys<'a>, QueryError> {
    match t.columns.get(name) {
        Some(Column::KeyF64(a)) => Ok(Keys::F64(a)),
        Some(Column::KeyF32(a)) => Ok(Keys::F32(a)),
        _ => Err(QueryError::Bind(format!(
            "column {name} must be a key column"
        ))),
    }
}

/// Executor totality guard (tests/topk_access_path.rs): reject a
/// bound param whose dimension does not match the key column before
/// any per-row dot product runs. `Database::run` performs the same
/// check up front; `execute` is pub and must be total on its own.
fn check_param_dim(
    keys: &Keys<'_>,
    x: &Array1<f64>,
    param_name: &str,
    key_col: &str,
) -> Result<(), QueryError> {
    let d = match keys {
        Keys::F64(a) => a.ncols(),
        Keys::F32(a) => a.ncols(),
    };
    if x.len() != d {
        return Err(QueryError::Bind(format!(
            "parameter :{param_name} has dim {} but key column {key_col} has dim {d}",
            x.len()
        )));
    }
    Ok(())
}

fn param<'a>(
    params: &'a HashMap<String, Array1<f64>>,
    name: &str,
) -> Result<&'a Array1<f64>, QueryError> {
    params
        .get(name)
        .ok_or_else(|| QueryError::Bind(format!("unbound parameter :{name}")))
}

pub(crate) fn eval_pred(p: &Pred, t: &crate::catalog::Table) -> Result<Vec<bool>, QueryError> {
    let col = p.column();
    let Some(Column::ScalarF64(v)) = t.columns.get(col) else {
        return Err(QueryError::Bind(format!(
            "filter column {col} must be ScalarF64"
        )));
    };
    Ok(match p {
        Pred::GtEq(_, k) => v.iter().map(|x| x >= k).collect(),
        Pred::Eq(_, k) => v.iter().map(|x| x == k).collect(),
    })
}

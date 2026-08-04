//! The temperature-aware planner: enumerate legal physical
//! alternatives for an optimized logical plan, cost each with the
//! calibrated model, rule on admissibility (approximation ONLY under
//! a declared error budget whose estimated bound the sketch can
//! certify), and choose the cheapest admissible candidate.
//!
//! EXPLAIN shows every candidate with its estimated cost and, for the
//! rejected ones, WHY: costlier, inadmissible, or illegal.

use std::collections::HashMap;

use ndarray::Array1;

use crate::cost::{CostEstimate, CostModel};
use crate::db::HnswIndexEntry;
use crate::logical::{LogicalPlan, Pred, ScoreExpr};
use crate::physical::PhysicalPlan;
use crate::stats::TableStats;
use crate::views::{param_fingerprint, SoftAggView};
use crate::QueryError;

/// The k ladder the planner tries for `HnswTopKScan`, smallest
/// first — the smallest admissible k wins (cheapest probe). The probe
/// beam is `ef = 4k.max(64)`: the recall margin that keeps the
/// runtime tail re-check from tripping (hnsw.rs measured recalls).
const HNSW_K_LADDER: [usize; 3] = [16, 64, 256];

/// Verdict on one enumerated candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Chosen: cheapest admissible plan.
    Chosen,
    /// Legal and admissible, but a cheaper candidate exists.
    Costlier,
    /// Only legal under a contract the query did not declare, or the
    /// estimate could not certify the declared budget.
    Inadmissible(String),
}

/// One enumerated candidate with its cost and verdict.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The physical plan.
    pub plan: PhysicalPlan,
    /// Estimated cost.
    pub cost: CostEstimate,
    /// The planner's ruling.
    pub verdict: Verdict,
}

/// A planned query: the chosen plan plus the full candidate set.
#[derive(Debug, Clone)]
pub struct PlannedQuery {
    /// The chosen physical plan.
    pub chosen: PhysicalPlan,
    /// All candidates, in enumeration order.
    pub candidates: Vec<Candidate>,
}

impl PlannedQuery {
    /// Full EXPLAIN: chosen plan, then the candidate table.
    pub fn explain(&self) -> String {
        let mut s = String::from("== chosen plan ==\n");
        s.push_str(&self.chosen.explain());
        s.push_str("\n== candidates ==\n");
        for c in &self.candidates {
            let head = match &c.verdict {
                Verdict::Chosen => "-> chosen   ".to_string(),
                Verdict::Costlier => "   costlier ".to_string(),
                Verdict::Inadmissible(r) => format!("   inadmissible ({r}) "),
            };
            let name = match &c.plan {
                PhysicalPlan::FusedGroupScan { .. } => "FusedGroupScan",
                PhysicalPlan::ExactGroupAvg { .. } => "ExactGroupAvg",
                PhysicalPlan::TopKContractScan { .. } => "TopKContractScan",
                PhysicalPlan::MaintainedViewScan { .. } => "MaintainedViewScan",
                PhysicalPlan::HnswTopKScan { .. } => "HnswTopKScan",
            };
            s.push_str(&format!(
                "{head}{name:<20} est {:.3} ms  ({:.1} MB, {})\n",
                c.cost.seconds * 1e3,
                c.cost.bytes / 1e6,
                c.cost.note
            ));
        }
        s
    }
}

/// Decompose the optimized logical plan into (agg shape, sel, table).
struct Shape<'a> {
    table: String,
    sel: Option<Pred>,
    kind: ShapeKind<'a>,
}

enum ShapeKind<'a> {
    Soft {
        group_col: &'a str,
        val_col: &'a str,
        score: &'a ScoreExpr,
        eps: f64,
        budget: Option<f64>,
    },
    Plain {
        group_col: &'a str,
        val_col: &'a str,
    },
}

fn shape<'a>(plan: &'a LogicalPlan) -> Result<Shape<'a>, QueryError> {
    let (kind, input) = match plan {
        LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            budget,
            input,
        } => (
            ShapeKind::Soft {
                group_col,
                val_col,
                score,
                eps: *eps,
                budget: *budget,
            },
            input,
        ),
        LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input,
        } => (ShapeKind::Plain { group_col, val_col }, input),
        _ => return Err(QueryError::Bind("plans must end in an aggregate".into())),
    };
    let (table, sel) = match input.as_ref() {
        LogicalPlan::Scan { table } => (table.clone(), None),
        LogicalPlan::Filter { pred, input } => match input.as_ref() {
            LogicalPlan::Scan { table } => (table.clone(), Some(pred.clone())),
            _ => return Err(QueryError::Bind("one filter directly over the scan".into())),
        },
        _ => return Err(QueryError::Bind("unsupported aggregate input".into())),
    };
    Ok(Shape { table, sel, kind })
}

/// Plan an optimized logical plan against stats, views, and bound
/// parameters. Kept for API stability: no index registry, so
/// `HnswTopKScan` is never enumerated — use [`plan_with_indexes`]
/// (what `Database::run` calls) to consider index plans.
pub fn plan(
    logical: &LogicalPlan,
    stats: &TableStats,
    views: &[SoftAggView],
    params: &HashMap<String, Array1<f64>>,
    model: &CostModel,
) -> Result<PlannedQuery, QueryError> {
    plan_with_indexes(logical, stats, views, &[], params, model)
}

/// [`plan`] plus the HNSW index registry: enumerates `HnswTopKScan`
/// for the single-group sharp-eps shape when an index covers the
/// score's key column and the sketch-predicted omitted mass clears
/// `cost::HNSW_TAIL_TOL` (see cost.rs for the derivation). Every
/// refusal is a `Verdict::Inadmissible` with the reason, so EXPLAIN
/// always shows WHY the index was or was not used.
pub fn plan_with_indexes(
    logical: &LogicalPlan,
    stats: &TableStats,
    views: &[SoftAggView],
    indexes: &[HnswIndexEntry],
    params: &HashMap<String, Array1<f64>>,
    model: &CostModel,
) -> Result<PlannedQuery, QueryError> {
    let sh = shape(logical)?;
    let n_rows = stats.n_rows as f64;
    let selv = sh.sel.as_ref().map(|p| stats.selectivity(p)).unwrap_or(1.0);

    let mut cands: Vec<Candidate> = Vec::new();

    match sh.kind {
        ShapeKind::Plain { group_col, val_col } => {
            let n_groups = stats
                .dicts
                .get(group_col)
                .map(|d| d.n_groups as f64)
                .unwrap_or(1.0);
            let c = model.exact_group_scan(n_rows, selv, 0, n_groups);
            cands.push(Candidate {
                plan: PhysicalPlan::ExactGroupAvg {
                    table: sh.table.clone(),
                    group_col: group_col.into(),
                    val_col: val_col.into(),
                    sel: sh.sel.clone(),
                },
                cost: c,
                verdict: Verdict::Chosen,
            });
        }
        ShapeKind::Soft {
            group_col,
            val_col,
            score,
            eps,
            budget,
        } => {
            let n_groups = stats
                .dicts
                .get(group_col)
                .map(|d| d.n_groups as f64)
                .unwrap_or(1.0);
            let d = stats
                .keys
                .get(&score.key_col)
                .map(|k| k.keys.ncols())
                .unwrap_or(0);

            // (1) exact fused scan: always legal, always admissible
            let c = model.fused_group_scan(n_rows, selv, d, n_groups);
            cands.push(Candidate {
                plan: PhysicalPlan::FusedGroupScan {
                    table: sh.table.clone(),
                    group_col: group_col.into(),
                    val_col: val_col.into(),
                    score: score.clone(),
                    eps,
                    sel: sh.sel.clone(),
                },
                cost: c,
                verdict: Verdict::Costlier, // provisional
            });

            // (2) maintained view, if one matches this exact shape
            if let Some(x) = params.get(&score.param) {
                let fp = param_fingerprint(&x.view());
                if let Some(v) = views
                    .iter()
                    .find(|v| v.matches(&sh.table, group_col, val_col, &score.key_col, fp, eps))
                {
                    if sh.sel.is_some() {
                        cands.push(Candidate {
                            plan: PhysicalPlan::MaintainedViewScan {
                                view: v.name.clone(),
                            },
                            cost: model.maintained_view_scan(n_groups),
                            verdict: Verdict::Inadmissible(
                                "view has no filter; query declares one".into(),
                            ),
                        });
                    } else {
                        cands.push(Candidate {
                            plan: PhysicalPlan::MaintainedViewScan {
                                view: v.name.clone(),
                            },
                            cost: model.maintained_view_scan(n_groups),
                            verdict: Verdict::Costlier,
                        });
                    }
                }
            }

            // (3) contracted top-k: only under a declared budget, only
            // when the sketch certifies it at this temperature
            match (
                budget,
                params.get(&score.param),
                stats.keys.get(&score.key_col),
            ) {
                (Some(b), Some(x), Some(sk)) => {
                    let vmax = stats
                        .scalars
                        .get(val_col)
                        .map(|s| s.min.abs().max(s.max.abs()))
                        .unwrap_or(sk.abs_max);
                    let est =
                        sk.estimate_contract(&x.view(), eps, b, vmax, (n_rows * selv) as usize);
                    let cost =
                        model.topk_contract_scan(n_rows, selv, d, est.kstar as f64, n_groups);
                    let verdict = if est.resolution_limited {
                        Verdict::Inadmissible(format!(
                            "sketch resolution-limited at eps={eps} (extreme-value regime); \
                             falling back to exact"
                        ))
                    } else if est.kstar >= (n_rows * selv) as usize {
                        Verdict::Inadmissible(format!(
                            "estimated k*={} is the whole input; contract buys nothing",
                            est.kstar
                        ))
                    } else {
                        Verdict::Costlier
                    };
                    cands.push(Candidate {
                        plan: PhysicalPlan::TopKContractScan {
                            table: sh.table.clone(),
                            group_col: group_col.into(),
                            val_col: val_col.into(),
                            score: score.clone(),
                            eps,
                            sel: sh.sel.clone(),
                            budget: b,
                            est_kstar: est.kstar,
                            est_delta: est.delta,
                        },
                        cost,
                        verdict,
                    });
                }
                (Some(_), _, _) => cands.push(Candidate {
                    plan: PhysicalPlan::FusedGroupScan {
                        table: sh.table.clone(),
                        group_col: group_col.into(),
                        val_col: val_col.into(),
                        score: score.clone(),
                        eps,
                        sel: sh.sel.clone(),
                    },
                    cost: model.fused_group_scan(n_rows, selv, d, n_groups),
                    verdict: Verdict::Inadmissible(
                        "budget declared but no sketch/param to certify it".into(),
                    ),
                }),
                (None, _, _) => {}
            }

            // (4) HNSW-served top-k: needs no declared budget — it is
            // admitted only when the sketch predicts the omitted
            // softmax mass under cost::HNSW_TAIL_TOL (answer then
            // indistinguishable from exact within the engine precision
            // contract; derivation in cost.rs), and the executor
            // re-checks the achieved bound at runtime. Every refusal
            // carries its reason into EXPLAIN.
            if let Some(ix) = indexes
                .iter()
                .find(|i| i.table == sh.table && i.key_col == score.key_col)
            {
                cands.push(hnsw_candidate(
                    &sh, group_col, val_col, score, eps, ix, stats, params, model, n_rows, selv, d,
                    n_groups,
                ));
            }
        }
    }

    // choose: cheapest candidate whose verdict is not Inadmissible
    let mut best: Option<usize> = None;
    for (i, c) in cands.iter().enumerate() {
        if matches!(c.verdict, Verdict::Inadmissible(_)) {
            continue;
        }
        if best
            .map(|b| c.cost.seconds < cands[b].cost.seconds)
            .unwrap_or(true)
        {
            best = Some(i);
        }
    }
    let best = best.ok_or_else(|| QueryError::Bind("no admissible physical plan".into()))?;
    for (i, c) in cands.iter_mut().enumerate() {
        if matches!(c.verdict, Verdict::Inadmissible(_)) {
            continue;
        }
        c.verdict = if i == best {
            Verdict::Chosen
        } else {
            Verdict::Costlier
        };
    }
    Ok(PlannedQuery {
        chosen: cands[best].plan.clone(),
        candidates: cands,
    })
}

/// Enumerate the `HnswTopKScan` candidate for one (index, query)
/// pair: walk the k ladder smallest-first, admit the first k whose
/// sketch-predicted omitted mass clears the tolerance, otherwise
/// return the candidate refused with the concrete reason. (The
/// argument list is the planner's whole decision context; a struct
/// would rename the same facts.)
#[allow(clippy::too_many_arguments)]
fn hnsw_candidate(
    sh: &Shape<'_>,
    group_col: &str,
    val_col: &str,
    score: &ScoreExpr,
    eps: f64,
    ix: &HnswIndexEntry,
    stats: &TableStats,
    params: &HashMap<String, Array1<f64>>,
    model: &CostModel,
    n_rows: f64,
    selv: f64,
    d: usize,
    n_groups: f64,
) -> Candidate {
    let mk_plan = |k: usize, ef: usize, predicted_tail: f64| PhysicalPlan::HnswTopKScan {
        table: sh.table.clone(),
        group_col: group_col.into(),
        val_col: val_col.into(),
        score: score.clone(),
        eps,
        sel: sh.sel.clone(),
        k,
        ef,
        predicted_tail,
    };
    let ef_of = |k: usize| (4 * k).max(64);
    let refuse = |reason: String| {
        let k = *HNSW_K_LADDER.last().expect("non-empty ladder");
        Candidate {
            plan: mk_plan(k, ef_of(k), 1.0),
            cost: model.hnsw_topk_scan(n_rows, d, k, ef_of(k), sh.sel.is_some()),
            verdict: Verdict::Inadmissible(reason),
        }
    };

    if n_groups > 1.0 {
        return refuse(format!(
            "GROUP BY over {n_groups:.0} groups: the index serves a global top-k; \
             grouped shapes stay on FusedGroupScan in v1"
        ));
    }
    if eps == 0.0 {
        return refuse(
            "eps=0 tropical endpoint needs exact argmax ties; an approximate \
             probe cannot certify ties"
                .into(),
        );
    }
    let Some(x) = params.get(&score.param) else {
        return refuse(format!("no bound :{} to probe with", score.param));
    };
    let Some(sk) = stats.keys.get(&score.key_col) else {
        return refuse(format!("no key sketch for {}", score.key_col));
    };
    if ix.is_empty() {
        return refuse("index has no live vectors".into());
    }

    let n_eff = (n_rows * selv).max(1.0);
    let sims = sk.sims(&x.view()); // scored once for the whole ladder
    let mut best_tail = f64::INFINITY;
    let mut all_limited = true;
    for k in HNSW_K_LADDER {
        if (k as f64) >= n_eff {
            break; // a probe of the whole input: the fused scan's job
        }
        let pred = crate::cost::predict_hnsw_tail_from_sims(&sims, sk.n_total, eps, k, n_eff);
        if pred.resolution_limited {
            continue;
        }
        all_limited = false;
        best_tail = best_tail.min(pred.tail);
        if pred.tail <= crate::cost::HNSW_TAIL_TOL {
            return Candidate {
                plan: mk_plan(k, ef_of(k), pred.tail),
                cost: model.hnsw_topk_scan(n_rows, d, k, ef_of(k), sh.sel.is_some()),
                verdict: Verdict::Costlier, // provisional; selection loop rules
            };
        }
    }
    if all_limited {
        refuse(format!(
            "sketch cannot resolve rank k <= {} at sample size {} over {} rows \
             (extreme-value regime); falling back to exact",
            HNSW_K_LADDER.last().expect("non-empty ladder"),
            sk.rows.len(),
            sk.n_total
        ))
    } else {
        refuse(format!(
            "predicted tail {best_tail:.3e} > tol {:.0e} at every k <= {} \
             (eps too diffuse for a top-k read)",
            crate::cost::HNSW_TAIL_TOL,
            HNSW_K_LADDER.last().expect("non-empty ladder")
        ))
    }
}

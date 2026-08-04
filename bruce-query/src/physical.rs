//! Physical plans: the executable alternatives the planner enumerates.

use crate::logical::{Pred, ScoreExpr};

/// Physical plans (v2).
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalPlan {
    /// One fused pass of `grouped_softavg`: dictionary-encoded group
    /// column, selection evaluated before scoring, per-group
    /// `(m, num, den)` state. The M1 kernel; the exact plan for any
    /// finite eps > 0 (and the tropical path at eps = 0).
    FusedGroupScan {
        /// Table to scan.
        table: String,
        /// Group column (DictU32 in storage).
        group_col: String,
        /// Value column (ScalarF64).
        val_col: String,
        /// Score expression.
        score: ScoreExpr,
        /// Temperature.
        eps: f64,
        /// Fused exact selection, if any.
        sel: Option<Pred>,
    },
    /// Exact uniform group average (the eps = inf endpoint after R3):
    /// no key column read, no exp.
    ExactGroupAvg {
        /// Table to scan.
        table: String,
        /// Group column.
        group_col: String,
        /// Value column.
        val_col: String,
        /// Fused exact selection, if any.
        sel: Option<Pred>,
    },
    /// Contracted top-k fold: stream sims for every admitted row, but
    /// fold only the k* best rows per the declared error budget.
    /// Admissible ONLY when the query declares a budget and the
    /// sketch estimate is not resolution-limited.
    TopKContractScan {
        /// Table to scan.
        table: String,
        /// Group column.
        group_col: String,
        /// Value column.
        val_col: String,
        /// Score expression.
        score: ScoreExpr,
        /// Temperature.
        eps: f64,
        /// Fused exact selection, if any.
        sel: Option<Pred>,
        /// Declared absolute error budget.
        budget: f64,
        /// Planner-estimated k* (global across groups, scaled).
        est_kstar: usize,
        /// Planner-estimated omitted mass at k*.
        est_delta: f64,
    },
    /// Serve from a registered maintained view: O(groups).
    MaintainedViewScan {
        /// View name.
        view: String,
    },
    /// Serve a SINGLE-GROUP sharp-eps fold from the HNSW index:
    /// probe top-k (filter-aware), rescore the k rows exactly in f64,
    /// fold max-anchored. Admitted WITHOUT a declared error budget —
    /// the planner admits it only when the predicted omitted softmax
    /// mass is <= `cost::HNSW_TAIL_TOL`, and the executor re-checks
    /// the achieved `hnsw_tail_bound` at runtime, falling back to the
    /// exact fused fold when the probe misses it. GROUP BY shapes
    /// (more than one group) are never admitted in v1: the index is a
    /// global top-k, and a per-group read served from a global probe
    /// would silently starve minority groups.
    HnswTopKScan {
        /// Table to scan.
        table: String,
        /// Group column (must cover exactly one group).
        group_col: String,
        /// Value column.
        val_col: String,
        /// Score expression (its key_col names the indexed column).
        score: ScoreExpr,
        /// Temperature (finite, > 0).
        eps: f64,
        /// Exact selection, served by the index's filter-aware probe.
        sel: Option<Pred>,
        /// Rows the fold keeps.
        k: usize,
        /// Probe beam width (recall margin; ef >> k).
        ef: usize,
        /// Planner-predicted omitted softmax mass (admission proof).
        predicted_tail: f64,
    },
}

impl PhysicalPlan {
    /// EXPLAIN-style rendering of this plan alone.
    pub fn explain(&self) -> String {
        match self {
            PhysicalPlan::FusedGroupScan {
                table,
                group_col,
                val_col,
                score,
                eps,
                sel,
            } => {
                let mut s = format!(
                    "FusedGroupScan[eps={eps}] kernel=grouped_softavg\n  \
                     group={group_col} val={val_col} score={:?}({},:{})\n",
                    score.kind, score.key_col, score.param
                );
                if let Some(p) = sel {
                    s.push_str(&format!(
                        "  fused Filter[eps=0]: {p:?}  (rows never scored)\n"
                    ));
                }
                s.push_str(&format!("  Scan {table}"));
                s
            }
            PhysicalPlan::ExactGroupAvg {
                table,
                group_col,
                val_col,
                sel,
            } => {
                let mut s = format!(
                    "ExactGroupAvg[eps=inf endpoint] group={group_col} val={val_col}\n  \
                     (R3: scoring dropped; key column not read)\n"
                );
                if let Some(p) = sel {
                    s.push_str(&format!("  fused Filter[eps=0]: {p:?}\n"));
                }
                s.push_str(&format!("  Scan {table}"));
                s
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
                est_delta,
            } => {
                let mut s = format!(
                    "TopKContractScan[eps={eps}] budget={budget} est k*={est_kstar} \
                     est delta={est_delta:.3e}\n  group={group_col} val={val_col} \
                     score={:?}({},:{})\n",
                    score.kind, score.key_col, score.param
                );
                if let Some(p) = sel {
                    s.push_str(&format!("  fused Filter[eps=0]: {p:?}\n"));
                }
                s.push_str(&format!("  Scan {table} (sims stream all keys)"));
                s
            }
            PhysicalPlan::MaintainedViewScan { view } => {
                format!("MaintainedViewScan view={view}  (O(groups) read)")
            }
            PhysicalPlan::HnswTopKScan {
                table,
                group_col,
                val_col,
                score,
                eps,
                sel,
                k,
                ef,
                predicted_tail,
            } => {
                let mut s = format!(
                    "HnswTopKScan[eps={eps}] k={k} ef={ef} \
                     admitted: predicted tail <= {predicted_tail:.3e} \
                     (tol {:.0e}, runtime re-checked)\n  \
                     group={group_col} val={val_col} score={:?}({},:{})\n",
                    crate::cost::HNSW_TAIL_TOL,
                    score.kind,
                    score.key_col,
                    score.param
                );
                if let Some(p) = sel {
                    s.push_str(&format!("  filter-aware probe: {p:?}\n"));
                }
                s.push_str(&format!(
                    "  Probe hnsw({table}.{})  (no full key stream)",
                    score.key_col
                ));
                s
            }
        }
    }
}

//! Rule-based optimizer. Rules are correctness-preserving and each
//! ships with an equivalence property test in tests/.
//!
//! R1  predicate pushdown: Filter[eps=0] commutes below SoftAgg
//!     because selection and scoring touch disjoint columns;
//!     legality: the predicate references no score input.
//! R3  endpoint degeneration:
//!     SoftAgg[eps = inf]           -> PlainGroupAvg (drop scoring;
//!                                     the key column leaves the plan)
//!     SoftAgg[eps = 0, Indicator]  -> stays SoftAgg; the kernel's
//!                                     tropical path IS the exact
//!                                     endpoint (argmax/equality
//!                                     semantics are the mask's job).
//!
//! eps is a SEMANTIC parameter: no rule changes it. Approximation is
//! the planner's business and only under a declared error budget.
//!
//! R1-NEGATIVE, pinned behavior (tests/explain_golden.rs): a Filter
//! whose predicate names the score's key column does NOT commute
//! below the aggregate — `pushdown` leaves it above, and the planner
//! then refuses the whole plan ("plans must end in an aggregate"), so
//! an illegal push can never happen silently. Through the SQL
//! pipeline the same predicate parses BELOW the aggregate (the
//! grammar is type-agnostic) and is rejected at execution with a
//! typed Bind error ("filter column ... must be ScalarF64"): eps = 0
//! selection is defined over ScalarF64 columns only; key and
//! dictionary columns are not legal filter inputs.

use crate::logical::LogicalPlan;

/// Optimize a logical plan (R1 + R3, to fixpoint on this v2 shape).
pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
    let plan = pushdown(plan);
    degenerate(plan)
}

/// R1: sink filters below aggregates when legal.
fn pushdown(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Filter { pred, input } => match *input {
            LogicalPlan::SoftAgg {
                group_col,
                val_col,
                score,
                eps,
                budget,
                input: agg_in,
            } if pred.column() != score.key_col => LogicalPlan::SoftAgg {
                group_col,
                val_col,
                score,
                eps,
                budget,
                input: Box::new(pushdown(LogicalPlan::Filter {
                    pred,
                    input: agg_in,
                })),
            },
            LogicalPlan::PlainGroupAvg {
                group_col,
                val_col,
                input: agg_in,
            } => LogicalPlan::PlainGroupAvg {
                group_col,
                val_col,
                input: Box::new(pushdown(LogicalPlan::Filter {
                    pred,
                    input: agg_in,
                })),
            },
            other => LogicalPlan::Filter {
                pred,
                input: Box::new(pushdown(other)),
            },
        },
        LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            budget,
            input,
        } => LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            budget,
            input: Box::new(pushdown(*input)),
        },
        LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input,
        } => LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input: Box::new(pushdown(*input)),
        },
        leaf @ LogicalPlan::Scan { .. } => leaf,
    }
}

/// R3: endpoint degeneration. eps = inf drops the score expression —
/// the uniform limit weighs every admitted row equally, so the plan
/// no longer needs the key column at all.
fn degenerate(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score: _,
            eps,
            budget: _,
            input,
        } if eps.is_infinite() => LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input: Box::new(degenerate(*input)),
        },
        LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            budget,
            input,
        } => LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            budget,
            input: Box::new(degenerate(*input)),
        },
        LogicalPlan::Filter { pred, input } => LogicalPlan::Filter {
            pred,
            input: Box::new(degenerate(*input)),
        },
        LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input,
        } => LogicalPlan::PlainGroupAvg {
            group_col,
            val_col,
            input: Box::new(degenerate(*input)),
        },
        leaf @ LogicalPlan::Scan { .. } => leaf,
    }
}

//! bruce-query — the eps-algebra query layer.
//!
//! SQL text in, executed result out, with the temperature `eps` as a
//! first-class quantity at every stage. See docs/QUERY_LAYER_DESIGN.md.
//!
//! v1 scope: single-table SOFTAVG queries with an optional exact
//! filter — the pipeline (parse -> bind -> logical -> optimize ->
//! physical -> execute) is complete end to end, and every later
//! operator slots into the same stages.

pub mod catalog;
pub mod cost;
pub mod db;
pub mod exec;
pub mod ingest;
pub mod logical;
pub mod optimizer;
pub mod parse;
pub mod physical;
pub mod planner;
pub mod stats;
pub mod views;

pub use catalog::{Catalog, Column, Table};
pub use cost::{CostEstimate, CostModel};
pub use db::{Database, RowValues};
pub use exec::execute;
pub use logical::{LogicalPlan, Pred, ScoreExpr};
pub use optimizer::optimize;
pub use parse::parse_query;
pub use physical::PhysicalPlan;
pub use planner::{plan, Candidate, PlannedQuery, Verdict};
pub use stats::{ContractEstimate, KeySketch, TableStats};
pub use views::SoftAggView;

use thiserror::Error;

/// Errors across the query layer.
#[derive(Debug, Error)]
pub enum QueryError {
    /// The SQL text did not parse or used unsupported syntax.
    #[error("parse error: {0}")]
    Parse(String),
    /// A referenced table or column is not in the catalog.
    #[error("binding error: {0}")]
    Bind(String),
    /// Plan execution failed in the kernel layer.
    #[error("execution error: {0}")]
    Exec(String),
}

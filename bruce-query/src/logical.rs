//! The eps-algebra logical IR. Every operator carries its
//! temperature; exact and soft nodes live in one tree.

/// Similarity kind for a score expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimKind {
    /// Inner product (softmax attention's score).
    Dot,
    /// Negative half squared distance (RBF).
    NegSq,
    /// Exact-equality indicator (the eps = 0 endpoint).
    Indicator,
}

/// `sim(key_col, :param)` — scoring a key column against a bound
/// query vector parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreExpr {
    /// Key (embedding) column name.
    pub key_col: String,
    /// Placeholder name of the query vector (e.g. `q`).
    pub param: String,
    /// Similarity kind.
    pub kind: SimKind,
}

/// Exact scalar predicate (v1: a single comparison).
#[derive(Debug, Clone, PartialEq)]
pub enum Pred {
    /// `col >= value`
    GtEq(String, f64),
    /// `col = value`
    Eq(String, f64),
}

impl Pred {
    /// Column the predicate touches (legality input for pushdown).
    pub fn column(&self) -> &str {
        match self {
            Pred::GtEq(c, _) | Pred::Eq(c, _) => c,
        }
    }
}

/// The logical plan tree (v1 subset).
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Base table scan.
    Scan {
        /// Table name in the catalog.
        table: String,
    },
    /// Exact (eps = 0) selection over the input.
    Filter {
        /// Predicate.
        pred: Pred,
        /// Input plan.
        input: Box<LogicalPlan>,
    },
    /// Grouped soft-average: `SELECT g, SOFTAVG(val WEIGHT score
    /// TEMP eps) ... GROUP BY g`.
    SoftAgg {
        /// Grouping column (dictionary-encoded in storage).
        group_col: String,
        /// Value column being averaged.
        val_col: String,
        /// Scoring expression producing the weights.
        score: ScoreExpr,
        /// Temperature.
        eps: f64,
        /// Declared absolute error budget, if any. The planner may
        /// substitute an approximate plan ONLY under this contract;
        /// absent a budget, only exact plans are admissible.
        budget: Option<f64>,
        /// Input plan.
        input: Box<LogicalPlan>,
    },
    /// Exact uniform group average (the eps -> inf endpoint after R3:
    /// scoring dropped entirely, no key column in the plan).
    PlainGroupAvg {
        /// Grouping column.
        group_col: String,
        /// Value column.
        val_col: String,
        /// Input plan.
        input: Box<LogicalPlan>,
    },
}

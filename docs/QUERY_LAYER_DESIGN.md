# bruce-query: the eps-algebra query layer (design, 2026-08-02)

The core of a database is what queries compile into. This document
fixes the design of bruce's query layer: SQL text in, executed plan
out, with the temperature eps as a first-class quantity at every
stage. Architecture reference: DuckDB (vectorized, push-based,
embeddable); the algebra is our own.

## 1. Pipeline

  SQL text
    -> parse        (sqlparser-rs dialect; SOFTAVG/sim/TEMP as
                     ordinary function syntax so the grammar stays
                     standard SQL)
    -> bind         (resolve tables/columns against the catalog;
                     attach types, group-dictionary handles, stats)
    -> logical plan (eps-algebra IR, Section 2)
    -> optimize     (rule pass + cost pass, Section 4)
    -> physical plan(Section 3)
    -> execute      (chunked scans over columnar storage, rayon)

## 2. Logical IR (eps-algebra)

Every operator carries its temperature. eps = 0 nodes are classical
relational algebra; eps > 0 nodes are the soft family; the IR does not
segregate them.

  LogicalPlan
    = Scan      { table }
    | Filter    { pred, eps = 0 }              -- exact selection
    | SoftAgg   { group_col, val_expr,
                  score_expr, eps }            -- SOFTAVG/SUM/COUNT
    | TopK      { score_expr, k, eps }         -- ranked read
    | AttnRead  { query_expr, eps = 1 }        -- attention over rows
    | Project   { exprs }

  ScoreExpr   = Sim { key_col, query_param, kind: Dot|NegSq|Indicator }
  Pred        = cmp over scalar cols (v1: col >= const, col = const,
                conjunctions)

## 3. Physical operators (v1)

  FusedGroupScan   -- M1 kernel grouped_softavg: one pass,
                      dictionary-encoded groups, selection fused,
                      (mu, z, u) monoid state per group.
  MaskStreamFold   -- generic masked_attention over (i, j) pairs;
                      fallback for arbitrary masks (trees, windows).
  ExactHashAgg     -- eps = 0 GROUP BY via hash (no scoring).
  TopKScan         -- heap scan; later HNSWProbe when the index
                      lands (M-storage).

Physical selection is by mask shape + eps + stats: GROUP BY mask ->
FusedGroupScan; arbitrary mask -> MaskStreamFold; eps = 0 with
equality score -> ExactHashAgg.

## 4. Optimizer

Rule pass (v1, correctness-preserving, always on):
  R1 predicate pushdown: Filter[eps=0] commutes below SoftAgg/TopK
     because selection and scoring touch disjoint columns; legality =
     pred references no score input. Physically realised as the fused
     `sel` mask (rows are never scored).
  R2 fusion: Filter + SoftAgg + (same-scan Project) -> one
     FusedGroupScan.
  R3 endpoint degeneration: SoftAgg[eps=0, Indicator] ->
     ExactHashAgg; SoftAgg[eps=inf] -> plain AVG (drop scoring).

Cost pass (M-optimizer milestone):
  bandwidth model: cost ~ bytes_scanned / B_mem + kernel constants,
  calibrated from measured 55 GB/s wall (results_m1.json). Chooses
  precision (f32/f64 scan) and, once HNSW lands, full-scan vs top-k
  with the entropy bound deciding admissibility. Decision quality is
  evaluated against an oracle (regret), per the fairness protocol in
  paper_sigmod_bruce/ROADMAP.md.

## 5. Storage interface (consumed by, not owned by, this layer)

  Catalog: table -> segments; column kinds:
    ScalarF64, KeyF64 / KeyF32 (precision chosen at load or by ALTER),
    DictU32 { dictionary: Vec<String> } for group columns,
    delete bitmap per segment.
  Stats per column: n, n_distinct, min/max; per key column: norm
  range (feeds the overflow-safety and entropy estimates).

## 6. Testing discipline

Every rule ships with an equivalence property test (optimized plan ==
naive plan output to 1e-12 across eps in {0, finite, inf}); every
physical operator ships with a cross-check against the reference
masked_attention path. The one-query harness in
bruce/experiments/cidr_one_query is the end-to-end regression.

## 7. Non-goals (v1)

Joins between tables (the relational side exists in join.rs and gets
planned in a later milestone), transactions (server WAL already
exists; grouped incremental views are the M-storage milestone),
distributed execution.

## 8. v2: the temperature-aware optimizer (2026-08-02, SHIPPED)

The cost pass of Section 4 is now implemented, plus the pieces it
needs. Modules: stats.rs, cost.rs, planner.rs, views.rs, db.rs.

### 8.1 Statistics: what this optimizer estimates

Classical stats answer "how many rows survive the predicate"
(equi-width 64-bucket histograms; validated ~ +/-5% on uniform data).
The NEW estimated quantity is weight concentration: a deterministic
uniform row sample of each key column (KeySketch, default 1024 rows,
strided, no RNG) is scored against the query vector at plan time, and
from the sampled sim distribution the planner estimates k*(eps,
budget): the smallest k whose certified bound
delta*(1+1/(1-delta))*max|v| meets the declared absolute error budget.

Honest failure mode, by construction: at very sharp temperatures the
top of the weight distribution is an extreme-value statistic a
uniform sample cannot resolve (fewer than 8 sample points inside the
admitted mass). The estimate then carries resolution_limited=true and
the planner REFUSES the contract plan rather than trust it.
Validated: estimated k* within 3x of the oracle at moderate eps
(test sketch_kstar_is_within_3x_of_oracle).

### 8.2 Cost model

cost = bytes/bandwidth + rows*row_overhead + groups*group_overhead,
calibrated from M1 (55 GB/s, 13.0 ms for 460k x 384 f64). Its job is
ORDERING plans, not clock prediction. Per-operator formulas encode
the physical truths: ExactGroupAvg reads no key bytes (the R3 prize);
TopKContractScan still streams every key for sims -- without an index
the contract saves only value bytes + fold work, and the model says
so; MaintainedViewScan is O(groups).

### 8.3 Rules

R1 (pushdown) unchanged. R3 endpoint degeneration implemented:
SoftAgg[eps=inf] -> PlainGroupAvg -- the score expression leaves the
plan, so the key column is never read (measured in the demo: 52.5 MB
-> 1.3 MB). eps stays a SEMANTIC parameter: no rule changes it.

### 8.4 Enumeration and admissibility

planner::plan enumerates: FusedGroupScan (always legal),
MaintainedViewScan (when a registered view matches table/group/val/
key/param-fingerprint/eps and the query has no extra filter),
TopKContractScan (ONLY when the query declares a budget -- SOFTAVG's
optional 4th argument -- and the sketch certifies it). Verdicts:
Chosen / Costlier / Inadmissible(reason); EXPLAIN prints all
candidates with estimated ms, MB, and the reason for every rejection.

TopKContractScan execution carries a RUNTIME GUARD: per-group k_g,
true omitted mass computed from the streamed sims, and any group
whose realized bound misses the budget is re-folded exactly. The plan
can be slower than estimated; it can never be wronger than the
budget, and no group is ever silently dropped (the incumbent top-k
failure the CIDR paper documents).

### 8.5 Write path and maintained views

Database::insert_row / delete_where mutate the columns, mark stats
stale (lazily recollected before the next plan), and apply deltas to
every matching SoftAggView: insert O(1) amortized; non-anchor delete
O(1) subtraction; deleting an anchor scorer triggers ONE bounded
re-anchor pass over that group's survivors (n_reanchors counts them
-- observability for the SIGMOD experiments). Freshness is tested
through the planner: after writes the served view answers match a
from-scratch reference to 1e-10.

### 8.6 What v2 still does not do

No plan enumeration beyond the four shapes (no join order yet); no
index-backed candidate (HNSWProbe lands with M-storage, and only then
does the contract plan's cost drop below the scan by more than
constants); no per-group sketches (group k_g is scaled from the
global estimate; the runtime guard covers the error); no cross-query
view advisor (views are user-created); stats staleness is
whole-table recollection, not incremental.

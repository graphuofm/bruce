# DuckDB: the same engine with and without `softavg`

Extension: `../../engines/duckdb` (C extension API, pure Rust, links `../../engines/core`).
DuckDB 1.5.6 (python package in `/home/jding/bruce/sigmod2027/.venv`), vss from the DuckDB
extension repository. **All timings are smoke numbers** (`"timing_status": "smoke, machine
shared"`): other agents compile and benchmark on this machine; the observed load is recorded
before/after each run (`load_before` / `load_after`, from `experiments/quiet.py`,
which reported the machine *not* quiet at some of these points). Formal timings are to be
re-run serially on an idle machine by the main thread.

## What is compared (PROTOCOL.md)

Same DuckDB, same table, same similarity (`array_inner_product(emb, q::FLOAT[384])::DOUBLE`
over embeddings stored as `FLOAT[384]`), only the aggregate spelling changes:

| plan | SQL |
|---|---|
| naive | `SUM(EXP(s/eps)*v)/SUM(EXP(s/eps))` |
| maxshift_cte | `WITH b AS (...), mx AS (SELECT k, MAX(s) m ... GROUP BY k)` joined back, `EXP((s-m)/eps)` (DuckDB materialises the CTE: `grid.explain`) |
| maxshift_window | `MAX(s) OVER (PARTITION BY k)`, then `EXP((s-m)/eps)` |
| clamp | maxshift_window with `CASE WHEN (s-m)/eps < -700 THEN 0 ELSE EXP(...) END` |
| softavg | `softavg(v, s, eps)`, the extension (in the round-1 grid: over the pre-P1 core) |
| softavg_p2 | round-1 grid only: the extension over core + P1 + P2 (P2 not adopted) |

Data: IMDb (`/data/softavg/corpora/imdb`, read-only) -> `data/imdb.duckdb`
(`build_db.py`; embeddings round-trip bit-exactly, checked on 9 rows). Predicate
`filter_num >= 2000`; 200 queries chosen in experiments/PROTOCOL.md (`RandomState(0)`, first sentence 4-20
words, de-duplicated; rows in `results.json -> grid.setup.query_rows`). eps grid
1e-4, 1e-3, 0.02, 0.1, 1, inf. Two references per cell: numpy float64 (`E64 @ q64`, the
PROTOCOL reference; its error is dominated by DuckDB's float32 dot product) and a longdouble
max-shifted reference over DuckDB's *own* scores (isolates the aggregation error).

## Where the earlier 13 ms vs 15 ms gap came from (run_gap.py)

The starting point measured, at 1 thread on the 1.31M-row reviews table, naive SQL
13 ms vs softavg 15 ms (eps = 0.1). Round 1 (2026-11-05, raw data in
`runs/archive/gap_pre_core_p1.json`) attributed the per-row cost with a `diag` build whose
`softavg_diag_*` aggregates replace only `update`, and with two proposed core patches:

1. **DuckDB's hash aggregate + C-API wrapper** (`capi_noop_update` vs `count_star`): the
   wrapper flattens all three inputs and builds a `DataChunk` per vector. Not reachable from
   an extension; a bind callback (to fold the constant eps) does not exist in the C API.
2. **Touching the states** (`capi_touch_update`, = softavg at eps = inf).
3. **The per-row checks in `update`.** The starting point validated eps, checked the eps tag
   and NaN, and branched on the eps regime per row; the shipped `update` hoists all of that to
   once per vector and specialises the loop per regime.
4. **Two `exp` calls per row in core's absorb** (`scale = exp((mu-mu2)/eps)` is `exp(0)`
   whenever the anchor does not move), where naive SQL pays one (DuckDB eliminates the
   common subexpression `EXP(s/eps)`). Patch P1 (`core_proposal/p1_one_exp.patch`) calls
   `exp` once and is bit-identical (`exp(0) == 1`, `x*1 == x`). **The main thread applied P1
   to core on 2026-11-05**; the shipped build now includes it.
5. **Sharp eps.** glibc `exp` is far slower when its result is subnormal
   (x in [-745, -708.4]) and somewhat slower on its other special-case paths (`exp_bench`
   table below). Patch P2 (skip weights below 2^-1022 relative to the anchor) removed that in
   round 1 but is not bit-identical in corner cases and **was not adopted**.

The current `run_gap.py` (round 2) re-measures naive SQL, the starting-point binary
(`before`), the round-1 shipped build (`pre_p1`: today's extension source over core with
P1 reversed, `build_variants.sh`) and the current shipped build, interleaved round by round,
plus the `diag` probes. Bitwise checks: the shipped build is compared with `before` and
`pre_p1` (both compute the two-exp recurrence); equality means P1 changed no bit, which is
what made the round-1 "+ core P1" variant bitwise equal to the round-1 shipped build.
Source-level check: `engines/core/src/acc.rs` differs from the `pre_p1` copy by exactly the
hunks of `core_proposal/p1_one_exp.patch`.

## Window functions (run_window.py)

DuckDB's C API has no window callback and no inverse; DuckDB still runs
`softavg(...) OVER (...)` through its own window operators on top of `combine`
(WindowSegmentTree for ROWS frames; `debug_window_mode = 'separate'` forces the naive
aggregator: update + finalize per frame). Conformance window cases pass in both modes
(`../../engines/duckdb/README.md`). `run_window.py` times
`ROWS BETWEEN 49 PRECEDING AND CURRENT ROW` per facet on IMDb (precomputed scores) against
the spellings available without the extension: naive (`SUM(EXP(s/eps)*v) OVER w / ...`),
the textbook shift by the *partition* max (the frame max cannot be used inside a window
SUM), and an exact SQL spelling (frame max via `MAX(s) OVER w`, then the 50 frame rows joined
back through `range(50)` as an equi-join). Results are checked row by row against a
longdouble per-frame reference; tolerance is absolute, 1e-9 * max|v| (PROTOCOL.md).

## Findings (read with the generated tables below; no numbers are typed here)

- **Silent wrong answers, no error.** DuckDB never raised an error in any cell of the grid
  (`grid.accuracy[].queries_error`). Naive SQL returned NaN for every facet of every query at
  eps = 1e-4 and NaN/Inf for some facets of most queries at eps = 1e-3 (`queries_with_nan`,
  `queries_with_inf`, `*_groups_total`), without a warning. All max-shift spellings and both
  softavg builds returned finite values everywhere.
- **Accuracy is decided by the float32 dot product, not by the aggregate.** Against the numpy
  float64 reference every finite plan has the same error (it comes from DuckDB's float32
  `array_inner_product`, and grows as eps shrinks); against a longdouble reference on DuckDB's
  own scores every plan, softavg included, is at rounding level.
- **Spelling matters for latency when the score is computed in the query.** DuckDB rewrites
  `MAX(s) OVER (PARTITION BY k)` into a hash aggregate joined back to a *second scan*
  (`grid.explain.maxshift_window`: two `SEQ_SCAN`s over `emb`), so the window and clamp
  spellings compute every dot product twice; the CTE spelling is materialised and computes it
  once but adds a join and a second hash aggregate. softavg is one scan, one hash aggregate,
  the same plan shape as naive SQL.
- **Aggregation only (run_gap.py):** the vectorised update brought softavg level with naive
  SQL at moderate eps; with P1 in core the shipped build is ahead of it (gap tables). At sharp
  eps the remaining cost is glibc's subnormal `exp` path (P2, not adopted). What remains above
  `count(*)` is DuckDB's hash aggregate plus the C-API wrapper (`capi_noop_update`), which an
  extension cannot remove.
- **Sliding windows (run_window.py).** Without the extension the only correct spelling is the
  self-join (`exact_selfjoin`), several times slower than `softavg ... OVER w` on DuckDB's
  segment tree. The cheap spellings fail silently at sharp eps: naive returns NaN for almost
  every row at eps = 1e-4 and for some rows at 1e-3; the partition-max shift returns NaN (0/0)
  for most rows at 1e-4 *and* finite but wrong values for others (`window.summary[].wrong_rows`,
  `max_abs_err_over_vmax`), again without an error. softavg is within tolerance in every mode.
  softavg on the segment tree costs more than the naive window SUMs (which do no max-shift
  at all); the naive window aggregator (`debug_window_mode = 'separate'`) costs more again.
- **Bitwise equality at 8 threads** is not expected even between runs of one build: the order
  in which DuckDB merges thread-local partials varies, and floating-point merge is not
  associative to the bit. At 1 thread the shipped build (core with P1) is bitwise equal to
  both pre-P1 builds in every gap cell.
- **vss top-k + AVG** answers a different question with partial facet coverage at small k
  (coverage and recall@k columns). The HNSW index is still chosen when the predicate is in the
  query on the full table, but it filters *after* the k nearest neighbours are fetched, so
  fewer than k rows come back at k = 100 / 1000 (last column); building the index on the
  pre-filtered table avoids this but needs one index per predicate.

## Results

<!-- BEGIN GENERATED by collect.py from results.json; do not edit by hand -->

Generated 2026-11-05T17:54:05; timing_status: **smoke, machine shared**.
Tests: 48 passed in 4.07s (pytest), test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s (cargo).

### Before/after on the aggregate itself (gap analysis, run_gap.py)

**reviews** (1,308,416 rows, GROUP BY genre, stored DOUBLE score), min of 21 interleaved runs, ms:

| threads | eps | naive SQL | before (paper_softavg build) | round 1 shipped (core pre-P1) | shipped (core with P1) |
|---|---|---|---|---|---|
| 1 | 0.0001 | 20.2 (NaN x29, Inf x0) | 18.0 | 15.4 | 12.2 |
| 1 | 0.001 | 13.4 (NaN x0, Inf x1) | 17.5 | 16.1 | 13.2 |
| 1 | 0.02 | 13.4 | 14.6 | 12.8 | 10.5 |
| 1 | 0.1 | 12.9 | 14.6 | 12.9 | 10.5 |
| 1 | 1.0 | 12.7 | 14.6 | 12.8 | 10.4 |
| 1 | inf | 12.2 | 7.30 | 7.01 | 6.92 |
| 8 | 0.0001 | 4.29 (NaN x29, Inf x0) | 3.74 | 3.23 | 2.68 |
| 8 | 0.001 | 3.02 (NaN x0, Inf x1) | 3.40 | 3.27 | 2.63 |
| 8 | 0.02 | 2.99 | 3.22 | 2.91 | 2.40 |
| 8 | 0.1 | 3.05 | 3.26 | 2.97 | 2.45 |
| 8 | 1.0 | 3.03 | 3.22 | 2.92 | 2.40 |
| 8 | inf | 2.88 | 1.76 | 1.73 | 1.74 |

threads 1: softavg results bitwise equal to the shipped build in 18/18 cells.

threads 8: softavg results bitwise equal to the shipped build in 12/18 cells (differing: softavg@before, softavg@pre_p1).

Cost attribution (reviews, min ms; `softavg_diag_*` come from the diag build):

| threads | count_star | avg_v | sum_s_sum_v | sum_exp | capi_noop_update | capi_touch_update | capi_softavg_2arg_eps_fixed | capi_softavg_eps_0.1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.60 | 4.71 | 8.50 | 10.2 | 5.36 | 7.02 | 10.3 | 11.0 |
| 8 | 0.96 | 1.16 | 1.93 | 2.25 | 1.36 | 1.66 | 2.21 | 2.37 |

**imdb_q0** (232,828 rows, GROUP BY facet, stored DOUBLE score), min of 21 interleaved runs, ms:

| threads | eps | naive SQL | before (paper_softavg build) | round 1 shipped (core pre-P1) | shipped (core with P1) |
|---|---|---|---|---|---|
| 1 | 0.0001 | 3.42 (NaN x28, Inf x0) | 3.59 | 3.07 | 2.55 |
| 1 | 0.001 | 2.55 | 3.32 | 3.00 | 2.53 |
| 1 | 0.02 | 2.60 | 3.04 | 2.72 | 2.23 |
| 1 | 0.1 | 2.60 | 3.14 | 2.72 | 2.24 |
| 1 | 1.0 | 2.56 | 2.98 | 2.61 | 2.20 |
| 1 | inf | 2.31 | 1.61 | 1.49 | 1.55 |
| 8 | 0.0001 | 2.24 (NaN x28, Inf x0) | 2.23 | 1.91 | 1.61 |
| 8 | 0.001 | 1.72 | 1.99 | 1.82 | 1.55 |
| 8 | 0.02 | 1.69 | 1.85 | 1.69 | 1.48 |
| 8 | 0.1 | 1.72 | 1.85 | 1.69 | 1.43 |
| 8 | 1.0 | 1.65 | 1.88 | 1.65 | 1.43 |
| 8 | inf | 1.59 | 1.07 | 1.03 | 1.04 |

threads 1: softavg results bitwise equal to the shipped build in 18/18 cells.

threads 8: softavg results bitwise equal to the shipped build in 18/18 cells.

Cost attribution (imdb_q0, min ms; `softavg_diag_*` come from the diag build):

| threads | count_star | avg_v | sum_s_sum_v | sum_exp | capi_noop_update | capi_touch_update | capi_softavg_2arg_eps_fixed | capi_softavg_eps_0.1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.73 | 1.05 | 1.65 | 1.94 | 1.29 | 1.46 | 2.11 | 2.30 |
| 8 | 0.56 | 0.76 | 1.12 | 1.28 | 0.85 | 1.03 | 1.26 | 1.36 |

glibc `exp` cost by argument range (exp_bench, best of 7 passes over 2,000,000 arguments, pinned to one P-core):

| argument range | ns per exp |
|---|---|
| [-1,0] normal | 2.58 |
| [-500,0] normal | 2.66 |
| [-700,-512] normal, slow path | 3.64 |
| [-745,-708.4] subnormal result | 34.89 |
| [-900,-746] result 0 (underflow) | 4.61 |
| [-5000,-1100] result 0 (early out) | 3.89 |
| [700,1000] overflow side | 4.22 |

### IMDb grid (run_grid.py)

Measured in round 1 (before core adopted P1): column `softavg` = the extension over the pre-P1 core, `softavg_p2` = over core + P1 + P2 (P2 not adopted). P1 is bit-identical, so the accuracy columns hold for the current build; the current build's aggregate latency is in the gap tables above.

**Accuracy**, 200 queries x eps, threads 8. Cell = `errors / queries with silent NaN or Inf (groups) / max rel err vs numpy f64 / max rel err vs longdouble on DuckDB's own scores`:

| eps | naive | maxshift_cte | maxshift_window | clamp | softavg | softavg_p2 |
|---|---|---|---|---|---|---|
| 0.0001 | 0 / 200 (5600) / - / - | 0 / 0 (0) / 7.1e-04 / 3.1e-16 | 0 / 0 (0) / 7.1e-04 / 3.1e-16 | 0 / 0 (0) / 7.1e-04 / 3.1e-16 | 0 / 0 (0) / 7.1e-04 / 3.1e-16 | 0 / 0 (0) / 7.1e-04 / 3.1e-16 |
| 0.001 | 0 / 161 (353) / 8.6e-05 / 5.9e-16 | 0 / 0 (0) / 8.6e-05 / 5.8e-16 | 0 / 0 (0) / 8.6e-05 / 6.9e-16 | 0 / 0 (0) / 8.6e-05 / 6.9e-16 | 0 / 0 (0) / 8.6e-05 / 6.9e-16 | 0 / 0 (0) / 8.6e-05 / 6.9e-16 |
| 0.02 | 0 / 0 (0) / 1.2e-05 / 7.8e-14 | 0 / 0 (0) / 1.2e-05 / 9.4e-14 | 0 / 0 (0) / 1.2e-05 / 1.4e-13 | 0 / 0 (0) / 1.2e-05 / 1.4e-13 | 0 / 0 (0) / 1.2e-05 / 2.5e-13 | 0 / 0 (0) / 1.2e-05 / 2.1e-13 |
| 0.1 | 0 / 0 (0) / 7.4e-07 / 3.0e-14 | 0 / 0 (0) / 7.4e-07 / 2.9e-14 | 0 / 0 (0) / 7.4e-07 / 4.2e-14 | 0 / 0 (0) / 7.4e-07 / 2.7e-14 | 0 / 0 (0) / 7.4e-07 / 3.2e-14 | 0 / 0 (0) / 7.4e-07 / 3.6e-14 |
| 1.0 | 0 / 0 (0) / 6.2e-09 / 4.3e-14 | 0 / 0 (0) / 6.2e-09 / 2.2e-14 | 0 / 0 (0) / 6.2e-09 / 3.7e-14 | 0 / 0 (0) / 6.2e-09 / 2.5e-14 | 0 / 0 (0) / 6.2e-09 / 3.7e-14 | 0 / 0 (0) / 6.2e-09 / 4.3e-14 |
| inf | 0 / 0 (0) / 1.9e-14 / 4.9e-15 | 0 / 0 (0) / 2.3e-14 / 1.8e-14 | 0 / 0 (0) / 1.9e-14 / 4.9e-15 | 0 / 0 (0) / 1.9e-14 / 4.9e-15 | 0 / 0 (0) / 1.9e-14 / 4.9e-15 | 0 / 0 (0) / 1.9e-14 / 4.9e-15 |

**Latency, score computed in the query (array_inner_product over FLOAT[384])**: median over 10 queries of the per-query median of 3 runs, ms (smoke, machine shared):

| threads | eps | naive | maxshift_cte | maxshift_window | clamp | softavg | softavg_p2 |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 296.5 (non-finite) | 307.2 | 592.6 | 589.0 | 302.4 | 293.4 |
| 1 | 0.001 | 298.4 (non-finite) | 306.5 | 595.2 | 600.1 | 297.6 | 298.2 |
| 1 | 0.02 | 299.5 | 307.9 | 592.2 | 591.4 | 295.3 | 298.3 |
| 1 | 0.1 | 295.4 | 303.5 | 592.5 | 595.3 | 298.8 | 293.8 |
| 1 | 1.0 | 296.1 | 306.9 | 591.2 | 587.5 | 295.4 | 293.7 |
| 1 | inf | 297.9 | 305.7 | 601.7 | 589.4 | 293.3 | 293.2 |
| 8 | 0.0001 | 99.3 (non-finite) | 101.6 | 197.8 | 197.1 | 99.0 | 97.6 |
| 8 | 0.001 | 99.3 (non-finite) | 101.7 | 197.4 | 198.4 | 98.7 | 98.6 |
| 8 | 0.02 | 98.6 | 101.2 | 196.3 | 198.4 | 98.7 | 97.9 |
| 8 | 0.1 | 99.1 | 101.8 | 197.5 | 198.0 | 99.3 | 99.0 |
| 8 | 1.0 | 99.6 | 101.8 | 199.1 | 199.0 | 99.2 | 99.0 |
| 8 | inf | 98.9 | 101.7 | 197.5 | 198.3 | 97.7 | 98.1 |

**Latency, precomputed scores (aggregation only)**: median over 10 queries of the per-query median of 3 runs, ms (smoke, machine shared):

| threads | eps | naive | maxshift_cte | maxshift_window | clamp | softavg | softavg_p2 |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 6.41 (non-finite) | 12.8 | 12.1 | 12.0 | 5.84 | 4.28 |
| 1 | 0.001 | 5.35 (non-finite) | 14.1 | 13.4 | 13.5 | 7.60 | 5.78 |
| 1 | 0.02 | 5.37 | 12.9 | 12.2 | 12.8 | 5.66 | 4.93 |
| 1 | 0.1 | 4.99 | 12.1 | 11.7 | 12.3 | 5.30 | 4.67 |
| 1 | 1.0 | 5.06 | 12.2 | 11.7 | 12.4 | 5.38 | 4.74 |
| 1 | inf | 4.84 | 12.0 | 11.7 | 12.3 | 4.07 | 3.96 |
| 8 | 0.0001 | 5.16 (non-finite) | 5.06 | 10.1 | 10.3 | 5.00 | 4.17 |
| 8 | 0.001 | 4.60 (non-finite) | 5.22 | 10.5 | 10.8 | 5.22 | 4.84 |
| 8 | 0.02 | 4.60 | 5.21 | 10.1 | 10.5 | 4.77 | 4.45 |
| 8 | 0.1 | 4.58 | 4.96 | 9.92 | 10.6 | 4.78 | 4.29 |
| 8 | 1.0 | 4.56 | 5.02 | 10.0 | 10.6 | 4.77 | 4.30 |
| 8 | inf | 4.47 | 4.82 | 9.99 | 9.97 | 3.81 | 3.86 |

### Sliding window, DuckDB segment tree over `combine` (run_window.py)

Frame `(PARTITION BY k ORDER BY rid ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)` over 232,828 rows per query, 5 queries, precomputed scores. Accuracy cell = `NaN / Inf / NULL / finite-but-wrong rows (all queries) ; max abs err / max|v|`, threads 8 (wrong = abs err > 1e-9 max|v|):

| eps | naive_window | maxshift_partition | exact_selfjoin | softavg_segtree | softavg_naive_mode |
|---|---|---|---|---|---|
| 0.0001 | 1164114 / 0 / 0 / 0 ; 8.9e-17 | 1117803 / 0 / 0 / 1313 ; 4.0e-02 | 0 / 0 / 0 / 0 ; 3.6e-16 | 0 / 0 / 0 / 0 ; 3.6e-16 | 0 / 0 / 0 / 0 ; 3.6e-16 |
| 0.001 | 823 / 0 / 0 / 0 ; 4.4e-16 | 3 / 0 / 0 / 6 ; 2.1e-08 | 0 / 0 / 0 / 0 ; 5.3e-16 | 0 / 0 / 0 / 0 ; 4.4e-16 | 0 / 0 / 0 / 0 ; 5.3e-16 |
| 0.02 | 0 / 0 / 0 / 0 ; 1.2e-15 | 0 / 0 / 0 / 0 ; 1.2e-15 | 0 / 0 / 0 / 0 ; 2.0e-15 | 0 / 0 / 0 / 0 ; 1.4e-15 | 0 / 0 / 0 / 0 ; 1.8e-15 |
| 0.1 | 0 / 0 / 0 / 0 ; 7.1e-16 | 0 / 0 / 0 / 0 ; 8.0e-16 | 0 / 0 / 0 / 0 ; 1.1e-15 | 0 / 0 / 0 / 0 ; 7.1e-16 | 0 / 0 / 0 / 0 ; 9.8e-16 |
| 1.0 | 0 / 0 / 0 / 0 ; 7.1e-16 | 0 / 0 / 0 / 0 ; 7.1e-16 | 0 / 0 / 0 / 0 ; 8.9e-16 | 0 / 0 / 0 / 0 ; 8.0e-16 | 0 / 0 / 0 / 0 ; 1.1e-15 |
| inf | 0 / 0 / 0 / 0 ; 5.3e-16 | 0 / 0 / 0 / 0 ; 5.3e-16 | 0 / 0 / 0 / 0 ; 5.3e-16 | 0 / 0 / 0 / 0 ; 5.3e-16 | 0 / 0 / 0 / 0 ; 5.3e-16 |

Latency (median over queries of the median of 3 runs, ms; query wrapped in `SELECT count(x), sum(x)`; smoke, machine shared):

| threads | eps | naive_window | maxshift_partition | exact_selfjoin | softavg_segtree | softavg_naive_mode |
|---|---|---|---|---|---|---|
| 1 | 0.0001 | 49.5 | 55.8 | 424.3 | 88.3 | 158.4 |
| 1 | 0.001 | 49.7 | 60.5 | 397.3 | 69.8 | 132.7 |
| 1 | 0.02 | 46.0 | 54.8 | 398.8 | 68.6 | 129.3 |
| 1 | 0.1 | 47.6 | 56.1 | 398.3 | 68.5 | 130.1 |
| 1 | 1.0 | 48.3 | 58.3 | 399.4 | 69.6 | 129.0 |
| 1 | inf | 45.3 | 53.7 | 378.9 | 48.9 | 84.6 |
| 8 | 0.0001 | 19.0 | 31.6 | 96.6 | 23.9 | 41.9 |
| 8 | 0.001 | 15.4 | 21.8 | 93.0 | 18.9 | 33.5 |
| 8 | 0.02 | 16.6 | 21.5 | 91.5 | 18.6 | 33.6 |
| 8 | 0.1 | 17.3 | 22.0 | 99.6 | 20.0 | 34.8 |
| 8 | 1.0 | 15.4 | 36.2 | 123.0 | 25.2 | 27.4 |
| 8 | inf | 23.4 | 34.7 | 105.9 | 25.0 | 40.2 |

### vss HNSW top-k + AVG (run_topk.py)

vss b833341 (REPOSITORY), HNSW metric 'ip', index build over the 232,828 admitted rows: 22.0 s at 8 threads; 28 facets in the admitted set. Index used: {'adm': True, 'full_t_where': True}.

| k | coverage mean (min-max) | recall@k vs exact mean (min) | rows returned min | latency 1 thr ms | latency 8 thr ms | WHERE on full table: rows returned mean, groups mean |
|---|---|---|---|---|---|---|
| 100 | 48.7% (3.6%-71.4%) | 0.907 (0.580) | 100 | 1.47 | 1.71 | 48, 10.1 |
| 1000 | 86.3% (64.3%-96.4%) | 0.954 (0.726) | 1000 | 7.69 | 8.68 | 476, 21.5 |
| 10000 | 99.4% (92.9%-100.0%) | 1.000 (0.994) | 10000 | 62.5 | 37.9 | 10000, 27.8 |

<!-- END GENERATED -->

## Files

```
build_db.py         corpus.parquet + emb.npy -> data/imdb.duckdb (FLOAT[384]), data/queries.json
build_variants.sh   build/diag (diag probes) and build/pre_p1 (core copy with P1 reversed)
core_proposal/      the two round-1 core patches against the pre-P1 acc.rs (P1 adopted, P2 not)
common.py           provenance (versions, sha256 of binaries and sources), load gate, pinning, error metrics
run_gap.py          -> runs/gap.json   before/after + cost attribution, interleaved
run_grid.py         -> runs/grid.json  IMDb accuracy (200 queries) + timing (subset), threads 1 and 8
run_window.py       -> runs/window.json  sliding-window softavg vs naive / partition-shift / exact SQL
run_topk.py         -> runs/topk.json  vss HNSW top-k + AVG: coverage, recall, latency, WHERE behaviour
exp_bench/          -> runs/exp_bench.json  ns per glibc exp by argument range (Rust)
collect.py          runs/*.json + cargo test + pytest -> results.json and the tables above
```

Pinning: threads = 1 runs on logical CPU 4 (a P-core), threads = 8 on CPUs 0,2,...,14
(one SMT thread per P-core); E-cores (16-31) are excluded.

## Rerun

```bash
cd /home/jding/bruce/paper_icde/engines/duckdb && nice -n 10 ./build.sh
cd /home/jding/bruce/paper_icde/experiments/duckdb
PY=python
$PY build_db.py && ./build_variants.sh
$PY run_gap.py && $PY run_grid.py && $PY run_window.py && $PY run_topk.py
(cd exp_bench && CARGO_TARGET_DIR=../build/target nice -n 10 cargo build --release --offline) && \
  taskset -c 4 build/target/release/exp_bench > runs/exp_bench.json
$PY collect.py
rm -rf data build          # cleanup: derived DB (~1 GB) and variant builds
```

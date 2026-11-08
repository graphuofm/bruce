# experiments/window_inverse — floating-point sliding-window SUM maintained by subtraction

Question (follow-up to `notes/INTEGRATION_ISSUES.md` A3 and `../llm_sql` T4): a host that slides
a window `SUM`/`AVG` by adding the entering row and **subtracting the leaving one** (an inverse
transition) can return finite but wrong values once the frame holds terms of very different size.
`exp(s/eps)` weights make that routine. Which engines do this, under which defaults and settings,
and what does it cost on the SQL an LLM actually wrote?

**Status (2026-11-06):** survey done for all 14 engines (source code or docs, each claim marked).
Measured on 9 engine builds (SQLite 3.37.2 and 3.54.0, DuckDB, PostgreSQL 17.10, ClickHouse via chdb,
DataFusion, Spark, Polars, pandas). MySQL and MariaDB are **not measured**: the binary tarballs
downloaded, but the permission system refused to run the downloaded server binaries (commands under
"Not run" below). Timing was not measured.

## Findings

1. **Among SQL engines, most are safe by default; the dataframe libraries are not.** Measured safe
   (correct to about 1e-14 wherever the formula is defined): PostgreSQL built-in `sum(float8)`, DuckDB,
   ClickHouse and Spark. None of them slides a float sum by subtraction. MySQL with its default setting is safe by source and docs (not run).
   Measured exposed: **SQLite (every version) and Apache DataFusion**, and both dataframe libraries,
   **Polars** (including its SQL front end, which compiles to the same kernel) and **pandas**. MariaDB and MySQL with
   `windowing_use_high_precision=OFF` are exposed by source (not run). PostgreSQL `sum(numeric)` has an
   inverse but exact arithmetic, and it is correct.
2. **Plain subtraction (SQLite <= 3.42, DataFusion) is wrong at the T4 setting itself.** At eps = 0.02
   with a 50-row frame, 119,309 of the 1,379,595 result rows (3 queries) are more than 1e-6 off.
   The worst row is a rolling average rating of **35,471.6** where the answer is 7.73 (SQLite 3.37.2,
   max relative error 4.6e3); DataFusion's worst is **-6,195** where the answer is 7.72 (8.0e2). About 500 rows fall outside
   the 1-10 rating scale. With a 1000-row frame the error shrinks (max 7e-3 relative, 0.04 rating points),
   because a wide frame usually holds a heavy term itself. At eps = 1e-3 almost every row is wrong (up to
   5e107) and 1.4k-21k rows are NULL/NaN where the formula is finite. In `../llm_sql` (sqlite-vec float32 scores,
   10 queries) the same SQL reached 3.7e4 / 234,918. Here (float64 scores, 3 queries) it reaches 4.6e3 / 35,472:
   same order, different inputs.
3. **It is the inverse, bit for bit.** The same SQL in PostgreSQL, with nothing changed except that the
   aggregate is the manual's own `unsafe_sum` (`float8pl` + `MINVFUNC float8mi`), returns arrays
   **bitwise identical** to SQLite 3.37.2 in the cells checked (eps 0.02 and 0.1, both frames); same 119,309
   wrong rows, same 35,471.6. DataFusion is bitwise identical to a numpy loop that adds the entering row and then
   subtracts the leaving one. PostgreSQL's built-in `sum(float8)` restarts the frame instead and is exact.
4. **Compensated summation postpones the failure; it does not remove it.** SQLite 3.43.0 switched
   `sum`/`total`/`avg` to Kahan-Babuska-Neumaier for both step and inverse. SQLite 3.54.0 is correct at
   eps = 0.02 and 0.1, but at eps = 1e-3 it is wrong-finite in 656k (frame 50) / 473k (frame 1000) rows
   and returns **NULL** in 289k / 164k rows where the formula is finite: when a 6.8e-266 weight leaves, a
   frame whose true weight sum is 3.7e-302 comes back as exactly 0.0, and `x/0` is NULL in SQLite. Polars and pandas
   (Kahan with separate add and subtract compensations) are wrong at eps = 0.02, frame 50 in 2 of 3 queries:
   worst row **1.58 where the answer is 6.30**, a wrong value that looks plausible. At eps = 1e-3 they are
   wrong by up to 4e62. One compensation term covers a dynamic range of about 1/u^2 = 8e31 inside a
   state's history; `exp((s-1)/eps)` with cosine `s` in [-1, 1] spans up to e^{2/eps}, i.e. e^100 = 2.7e43
   at eps = 0.02 and past the double range at eps = 1e-3.
5. **exp only makes the corner case routine.** The non-exponential minimal case `[1, 1, 1e20, 1, 1, 1, 1]`,
   3-row frame, returns 1 instead of 3 in SQLite 3.37.2 and in PG `unsafe_sum`, and 0 in DataFusion.
   `[1, 1e40, 1e20, 1, 1, 1, 1]` also defeats one compensation term: SQLite 3.54.0 returns 0, Polars 1.
   pandas is right on both: it folds the add-compensation into the next add. PG built-in `sum`, DuckDB,
   ClickHouse and Spark are right. The PostgreSQL manual uses this very example (`1e20` then `1`) to explain why
   `sum(float8)` has no inverse.
6. **The formula has a separate failure that is not the host's.** At eps = 1e-3 the LLM's constant
   shift by 1/T (equal to the global max-shift here) underflows every weight in some frames. 114,732
   rows (frame 50) and 1,511 (frame 1000) are 0/0 in the formula itself. With PG's clamp at -700
   the counts are 356,148 / 4,286. These rows are counted separately ("F 0/0") and never charged to the host.
   PostgreSQL is loud here: the verbatim SQL raises `value out of range: underflow`, and with Opus's own
   clamp it raises `division by zero`. Only with an added `NULLIF(den, 0)` does it run, and then it is correct on every
   row where the formula is defined. DuckDB/ClickHouse/Spark return NaN/NULL for exactly the F 0/0 rows and
   are correct elsewhere. The inverse engines are worse here too: at frame 50 SQLite 3.37.2 returns a **finite number** for
   92,149 of the 114,732 F 0/0 rows (SQLite 3.54.0: 74,594; DataFusion: 92,413; Polars: 19,268), i.e. an "average" of rows that all have zero
   weight, made of residue left by rows that already left the frame.

## Survey: how each engine computes a floating-point `SUM`/`AVG` over a sliding frame

Evidence levels: **S** = verified from source (file:line at the version stated), **D** = verified from
official docs, **M** = measured here, **U** = unverified. "Sliding" = a frame whose start moves
(`ROWS BETWEEN n PRECEDING AND CURRENT ROW`).

| engine (version read) | sliding float SUM/AVG | default / setting | exposed? | evidence |
|---|---|---|---|---|
| SQLite <= 3.42 (read 3.37.2) | inverse: `sumInverse` does `p->rSum -= x` (`src/func.c` L1550-1571 @version-3.37.2); `sum`, `total`, `avg` all registered with it (L2271-2273) | always: `window.c` uses `xInverse` for every sliding frame; only an `EXCLUDE` clause switches to a full frame scan (`windowFullScan`, `src/window.c` L1438-1439, L1944-1947 @master) | **yes** | S, M |
| SQLite >= 3.43.0 (read master = 3.55.0; ran 3.54.0) | inverse with Kahan-Babuska-Neumaier compensation: `kahanBabuskaNeumaierStep` (`src/func.c` L1925-1940), `sumInverse` steps `-x` (L2017-2050), registration L3507-3508; KBN absent in 3.42.0, present in 3.43.0 | always, no setting | **yes, mitigated** (fails at sharper eps / wider dynamic range) | S, M |
| PostgreSQL 17.10 | `sum(float4/float8)` and `avg(float4/float8)` have **no** `aggminvtransfn` (`src/include/catalog/pg_aggregate.dat` L40-45, L70-73); window code restarts and re-aggregates the frame when the head moves and no inverse exists (`src/backend/executor/nodeWindowAgg.c` L703-712, L785-792). `numeric`, int, money and interval sums do have inverses (exact arithmetic). The manual documents the hazard with `unsafe_sum` (`doc/src/sgml/xaggr.sgml` L234-271; https://www.postgresql.org/docs/17/xaggr.html#XAGGR-MOVING-AGGREGATES). An inverse may return NULL to make the host recompute (nodeWindowAgg.c L705-712) | built-ins safe; any user/extension float aggregate that declares a `MINVFUNC` is exposed | no (built-in); yes for a float `MINVFUNC` (measured with the manual's `unsafe_sum`) | S, D, M |
| DuckDB 1.5.6 | no inverse in the aggregate API at all. Dispatch in `src/function/window/window_aggregate_function.cpp` L55-75: naive / distinct / constant / custom (only aggregates with a `window` callback, `window_custom_aggregator.cpp` L10-22) / segment tree. `sum(DOUBLE)` has no window callback, so it goes to `WindowSegmentTree` (combines O(log n) partial states, `window_segment_tree.cpp` L13-23, L490-510); `sum(DOUBLE)` = plain add (`extension/core_functions/aggregate/distributive/sum.cpp` L62, L283), `fsum`/`kahan_sum` separate (L63, L302) | always | no | S, M |
| MySQL 8.0 / 8.4 | `Item_sum_sum::check_wf_semantics1` (`sql/item_sum.cc` L1936-1955 @8.0): for `REAL_RESULT` with a moving frame start, inversion is allowed only if `windowing_use_high_precision` is OFF; AVG derives from SUM. Docs: "The default is to avoid inverse aggregation, which is slower but preserves precision" (https://dev.mysql.com/doc/refman/8.0/en/window-function-optimization.html) | `windowing_use_high_precision = ON` by default; OFF enables the inverse | no by default; **yes with `windowing_use_high_precision=OFF`** | S, D (not run) |
| MariaDB 11.4 | `Item_sum_sum::add_helper(perform_removal)` does `sum -= arg_val_real()` for REAL (`sql/item_sum.cc` L1641-1720, `remove()` L1772-1777 @11.4 branch); `Frame_n_rows_preceding` calls `remove_value_from_items()` when the top bound moves (`sql/sql_window.cc` L1179-1190, L1975-1995) | no setting found | **yes** | S (not run) |
| ClickHouse 26.9 (chdb 4.4.0 = 26.9.2.1) | recompute: `WindowTransform::updateAggregationState` resets the aggregate state (`destroy`/`create`) and re-adds the whole frame whenever the frame start moves: "for now we always have to reset when the frame start changes" (`src/Processors/Transforms/WindowTransform.cpp` L1202-1252, comment L1228-1232 @v26.9.2.8-stable; same at L945-990 @v25.8.1.5101-lts). Current master moved this to `src/WindowFunctions/Aggregate/AggregateWindowFunctionRecalculate.cpp` (same reset-and-re-add; prefix accumulation only for `UNBOUNDED PRECEDING`, `IAggregateWindowFunction.cpp` L10-15) | always | no | S, M |
| Spark SQL 3.5.9 | recompute: `SlidingWindowFunctionFrame` re-initialises the processor and re-aggregates the whole buffer whenever the buffer changes (`sql/core/.../window/WindowFunctionFrame.scala` L402-490, L479-486 @branch-3.5) | always | no | S, M |
| Apache DataFusion 55.0.0 | inverse: bounded frames use `SlidingAggregateWindowExpr` (`datafusion/physical-plan/src/windows/mod.rs` L150-160), which calls `Accumulator::retract_batch` (`physical-expr/src/window/sliding_aggregate.rs` L228, L262); `SlidingSumAccumulator::retract_batch` = `self.sum.sub_wrapping(sum(batch))` (`functions-aggregate/src/sum.rs` L576-580); float AVG retract `sum - x` (`average.rs` L629-635) | always, no setting | **yes** | S, M |
| Polars 2.0.0 (`rolling_sum`, and SQL window frames) | inverse: `SumWindow::update` subtracts leaving values (`crates/polars-compute/src/rolling/sum.rs` L132-150 @py-2.0.0) with Kahan add and a separate Kahan subtract compensation (L52-66) | always | **yes, mitigated** | S, M |
| pandas 2.3.3 (`rolling().sum()`, also `mean`) | inverse: `roll_sum` removes leaving values (`pandas/_libs/window/aggregations.pyx` L131-182, deletes L166-167) with separate Kahan compensations for add/remove (`add_sum` L90-110, `remove_sum` L115-127). Docs warn that "large values may have an impact on windows, which do not include these values" (https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/window.html, Overview warning) | always | **yes, mitigated** | S, D, M |
| Oracle | no public description of the window-aggregate algorithm found | - | unknown | U |
| SQL Server | no official description of how a `ROWS n PRECEDING` float `SUM` is maintained found (searched; nothing citable) | - | unknown | U |
| Snowflake | docs only say float aggregates accumulate rounding error and may be nondeterministic, and suggest casting to fixed point (https://docs.snowflake.com/en/sql-reference/data-types-numeric); sliding-frame algorithm not documented | - | unknown | U |
| BigQuery | window-call reference says only that "the window function result is computed using the selected window of rows as input" (https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/window-function-calls); no algorithm, no precision note | - | unknown | U |

Source files were read from upstream GitHub at the tag/branch named (SQLite `version-3.37.2` and
`master`, DuckDB `v1.5.6`, MySQL `8.0`, MariaDB `11.4`, ClickHouse `master` and `v25.8.1.5101-lts`,
Spark `branch-3.5`, DataFusion `55.0.0`, Polars `py-2.0.0`, pandas `v2.3.3`); PostgreSQL from the
local 17.10 source tree (`~/.pgrx/17.10/src`). Line numbers refer to those revisions.

## Why: three ways to evaluate a sliding frame

- **Inverse (subtract the leaving row):** O(1) per row, but the state is a *running* float sum. When
  a heavy term enters, the small terms are absorbed below its rounding unit; when it leaves, what
  is left is rounding noise of the heavy term's size, not the small terms. The error depends on the
  largest term the state has seen since its last fresh start. That term is not in the current frame,
  so the result is wrong for a frame whose own terms are harmless. Compensation (Kahan, KBN) widens
  the representable range from about 1/u to about 1/u^2 but has the same failure beyond it. Engines:
  SQLite, DataFusion, Polars, pandas, MariaDB, MySQL with `windowing_use_high_precision=OFF`, any PG
  aggregate with a float `MINVFUNC`.
- **Segment tree (DuckDB):** each frame is combined from O(log n) precomputed partial sums of
  disjoint ranges; nothing is ever subtracted, so every term enters exactly the frames that contain it.
  Accuracy is that of a fresh summation of the frame.
- **Recompute (PostgreSQL float, ClickHouse, Spark, MySQL default):** re-aggregate the frame from
  scratch when its start moves. O(frame) per row, exact up to ordinary summation error.

## What a correct extension must do

For an aggregate that wants O(1) window maintenance (and incremental view maintenance, which has
the same delete problem), the inverse must come with an **error certificate**, and the host must
have a way to act on it:

1. Keep the exponentials anchored (max-shift, never materialise `e^{s/eps}`) so the state does not
   overflow. That is necessary but not sufficient: removal never re-anchors, so cancellation is still possible.
2. Carry a running bound on the absolute error of the state and refuse to deliver a value whose
   bound exceeds the tolerance. Ours is `InvAcc` in `engines/core/src/lib.rs`: module contract
   L1-28 (removal never rescans; running first-order bound on `z` and `u` after Higham §3.3; the host
   must recompute once `is_exact()` is false), `value_error_bound` L106-116, the tolerance check
   `max(1e-10, 100 n u) * v_scale` L119-131, `remove` L177, `is_exact` L278.
3. Recompute through whatever the host offers. The three host contracts (A5) differ:
   - **PostgreSQL:** `MINVFUNC` may return NULL. The host then restarts and re-aggregates the frame
     (nodeWindowAgg.c L705-712), so the certificate maps directly onto the contract.
   - **SQLite:** `xInverse` cannot refuse. The extension keeps its own lazy frame shadow and refolds
     itself when the certificate fails (`../sqlite`: `win_softavg`, 1.6e-11 max rel err at eps = 0.02,
     1,124 refolds over 698k frames, in the window table of `../sqlite/README.md`).
   - **DuckDB:** no inverse in the C API. The window path is the segment tree (combine only), so
     `remove` is never called and nothing can go wrong this way.
   Compensated summation alone (SQLite >= 3.43, Polars, pandas) is not a fix: it is still wrong at
   eps = 1e-3 here, and it gives no signal when it is wrong.

## Setup

- **Data.** IMDb (`/data/softavg/corpora/imdb`, read-only), all 459,865 rows, 29 categories,
  `rating` 1-10, `ts = 2000-01-01 + id s` (as in `../llm_sql`). Queries: the first 3 of the PROTOCOL
  sampler (RandomState(0); rows 256557, 107447, 258157, the same rows as `../postgres/queries.json`).
  Score `s<q>` = float64 cosine to the query row, precomputed once into the table (`common.py`). The
  queries are table rows, so max `s` = 1.
- **SQL.** claude-opus's `T4_rolling_window` answers (`../llm_sql/outputs/claude-opus.jsonl`), which use
  a constant shift: `w = exp((cos - 1)/T)`. Adapted only in dialect, plus one substitution for every
  engine: the engine's similarity function is replaced by the precomputed `s<q>` column, so every engine
  slides bitwise-identical weights. Temperature `T in {1e-3, 0.02, 0.1}`, frame `ROWS BETWEEN k-1
  PRECEDING AND CURRENT ROW` with `k in {50, 1000}`. PostgreSQL also runs: Opus's own clamp
  (`CASE WHEN z < -700 THEN 0`) at T = 1e-3, where the verbatim SQL raises; the manual's `unsafe_sum`; and
  `sum(w::numeric)`. Polars runs the SQL through `pl.SQLContext` and as `rolling_sum().over()`; pandas as
  `groupby().rolling().sum()`. The exact SQL per engine, variant and (eps, frame) is in `results.json` (`engines.<e>.sql`, shown for q0; q1/q2 differ only in the `s<q>` column).
- **References** (numpy float64, per frame from scratch, no subtraction): **F** = the SQL's own formula
  `exp((s-1)/T)` summed fresh per frame. Deviation from F is the host's. **T** = the per-frame max-shifted
  softmax the prompt asked for. A row is wrong if finite and more than 1e-6 relative off F.
- **Minimal cases:** `SUM(x) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` on 7 rows,
  against correctly rounded exact sums (`fractions`).

## Generated tables

<!-- BEGIN GENERATED (assemble.py) -->

### Versions

| engine | version | how it was run |
|---|---|---|
| SQLite | 3.37.2 | binding: Python 3.10.12 sqlite3 |
| SQLite | 3.54.0 | binding: apsw 3.54.0.0 (bundled SQLite) |
| Apache DataFusion (Python bindings) | 55.0.0 |  |
| Polars | 2.0.0 |  |
| pandas | 2.3.3 |  |
| PostgreSQL | 17.10 | server: PostgreSQL 17.10 on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, 64-bit; build: release (~/.pgrx/17.10/pg-release-install); aggminvtransfn: {'avg(double precision)': '-', 'sum(double precision)': '-', 'sum(numeric)': 'numeric_accum_inv', 'unsafe_sum(double precision)': 'float8mi'} |
| DuckDB | 1.5.6 | threads: 32 |
| ClickHouse (chdb, embedded) | 26.9.2.1 | chdb: 4.4.0 |
| Apache Spark SQL (local[8]) | 3.5.9 | java: openjdk version "1.8.0_452"; ansi: false |

### Rolling softmax-weighted average (opus T4 SQL), 3 queries x 459,865 rows per cell

Class = worst over the 3 queries, judged against the formula reference F (host attribution); `q` = queries with that class. wrong rows = finite but > 1e-6 relative off F; host NULL/NaN = engine returns NULL/NaN/Inf where F is finite; abs err in rating points (ratings are 1-10); out of [1,10] = finite outputs outside the rating scale; F 0/0 = rows where every weight in the frame underflows (the formula's own failure, identical for every engine; not counted against the host); finite where F 0/0 = the engine returns a number although every weight in the frame is 0 (built from residue of rows that already left).

| engine | spelling | eps | frame | class (q) | wrong rows | host NULL/NaN | max rel err | max abs err | out of [1,10] | F 0/0 rows | finite where F 0/0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sqlite_stdlib | opus SQL | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 872664 | 19487 | 5.0e+107 | 6.5e+107 | 254842 | 114732 | 92149 |
| sqlite_stdlib | opus SQL | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 799078 | 1350 | 6.5e+89 | 4.4e+90 | 176100 | 1511 | 0 |
| sqlite_stdlib | opus SQL | 0.02 | 50 | **wrong-finite** (3) | 119309 | 0 | 4.59e+03 | 3.5e+04 | 494 | 0 | 0 |
| sqlite_stdlib | opus SQL | 0.02 | 1000 | **wrong-finite** (3) | 12333 | 0 | 7.2e-03 | 0.0421 | 0 | 0 | 0 |
| sqlite_stdlib | opus SQL | 0.1 | 50 | correct (3) | 0 | 0 | 1.2e-13 | 6.1e-13 | 0 | 0 | 0 |
| sqlite_stdlib | opus SQL | 0.1 | 1000 | correct (3) | 0 | 0 | 6.4e-14 | 4.4e-13 | 0 | 0 | 0 |
| sqlite_apsw | opus SQL | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 655948 | 288647 | 2.5e+53 | 1.6e+54 | 224103 | 114732 | 74594 |
| sqlite_apsw | opus SQL | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 473426 | 164234 | 342 | 2.43e+03 | 161986 | 1511 | 0 |
| sqlite_apsw | opus SQL | 0.02 | 50 | correct (3) | 0 | 0 | 3.2e-15 | 1.7e-14 | 0 | 0 | 0 |
| sqlite_apsw | opus SQL | 0.02 | 1000 | correct (3) | 0 | 0 | 1.3e-15 | 9.8e-15 | 0 | 0 | 0 |
| sqlite_apsw | opus SQL | 0.1 | 50 | correct (3) | 0 | 0 | 8.2e-16 | 5.3e-15 | 0 | 0 | 0 |
| sqlite_apsw | opus SQL | 0.1 | 1000 | correct (3) | 0 | 0 | 9.7e-16 | 5.3e-15 | 0 | 0 | 0 |
| datafusion | opus SQL | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 875198 | 20738 | 5.0e+107 | 6.5e+107 | 258471 | 114732 | 92413 |
| datafusion | opus SQL | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 799015 | 1413 | 6.5e+89 | 4.4e+90 | 176101 | 1511 | 0 |
| datafusion | opus SQL | 0.02 | 50 | **wrong-finite** (3) | 119291 | 0 | 803 | 6.2e+03 | 500 | 0 | 0 |
| datafusion | opus SQL | 0.02 | 1000 | **wrong-finite** (3) | 12343 | 0 | 5.9e-03 | 0.0343 | 0 | 0 | 0 |
| datafusion | opus SQL | 0.1 | 50 | correct (3) | 0 | 0 | 1.3e-13 | 6.9e-13 | 0 | 0 | 0 |
| datafusion | opus SQL | 0.1 | 1000 | correct (3) | 0 | 0 | 5.8e-14 | 3.8e-13 | 0 | 0 | 0 |
| polars | rolling_sum().over() | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 159442 | 3611 | 3.9e+62 | 2.6e+63 | 58110 | 114732 | 19268 |
| polars | rolling_sum().over() | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 133298 | 15 | 7.3e+48 | 5.5e+49 | 44601 | 1511 | 0 |
| polars | rolling_sum().over() | 0.02 | 50 | **wrong-finite** (2) | 8352 | 0 | 0.749 | 5.89 | 0 | 0 | 0 |
| polars | rolling_sum().over() | 0.02 | 1000 | correct (3) | 0 | 0 | 4.7e-12 | 2.0e-11 | 0 | 0 | 0 |
| polars | rolling_sum().over() | 0.1 | 50 | correct (3) | 0 | 0 | 1.5e-14 | 9.4e-14 | 0 | 0 | 0 |
| polars | rolling_sum().over() | 0.1 | 1000 | correct (3) | 0 | 0 | 1.1e-15 | 6.2e-15 | 0 | 0 | 0 |
| polars | opus SQL via pl.SQLContext | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 159442 | 3611 | 3.9e+62 | 2.6e+63 | 58110 | 114732 | 19268 |
| polars | opus SQL via pl.SQLContext | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 133298 | 15 | 7.3e+48 | 5.5e+49 | 44601 | 1511 | 0 |
| polars | opus SQL via pl.SQLContext | 0.02 | 50 | **wrong-finite** (2) | 8352 | 0 | 0.749 | 5.89 | 0 | 0 | 0 |
| polars | opus SQL via pl.SQLContext | 0.02 | 1000 | correct (3) | 0 | 0 | 4.7e-12 | 2.0e-11 | 0 | 0 | 0 |
| polars | opus SQL via pl.SQLContext | 0.1 | 50 | correct (3) | 0 | 0 | 1.5e-14 | 9.4e-14 | 0 | 0 | 0 |
| polars | opus SQL via pl.SQLContext | 0.1 | 1000 | correct (3) | 0 | 0 | 1.1e-15 | 6.2e-15 | 0 | 0 | 0 |
| pandas | groupby().rolling().sum() | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 159691 | 3633 | 3.9e+62 | 2.6e+63 | 50170 | 114732 | 0 |
| pandas | groupby().rolling().sum() | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 133078 | 15 | 7.3e+48 | 5.5e+49 | 43900 | 1511 | 0 |
| pandas | groupby().rolling().sum() | 0.02 | 50 | **wrong-finite** (2) | 5793 | 0 | 0.749 | 5.89 | 0 | 0 | 0 |
| pandas | groupby().rolling().sum() | 0.02 | 1000 | correct (3) | 0 | 0 | 5.2e-09 | 3.3e-08 | 0 | 0 | 0 |
| pandas | groupby().rolling().sum() | 0.1 | 50 | correct (3) | 0 | 0 | 1.5e-14 | 9.4e-14 | 0 | 0 | 0 |
| pandas | groupby().rolling().sum() | 0.1 | 1000 | correct (3) | 0 | 0 | 9.8e-16 | 5.3e-15 | 0 | 0 | 0 |
| postgresql | opus SQL verbatim | 0.001 | 50 | error (3) | `NumericValueOutOfRange: value out of range: underflow` | | | | | | |
| postgresql | opus SQL verbatim | 0.001 | 1000 | error (3) | `NumericValueOutOfRange: value out of range: underflow` | | | | | | |
| postgresql | opus SQL verbatim | 0.02 | 50 | correct (3) | 0 | 0 | 2.0e-15 | 1.4e-14 | 0 | 0 | 0 |
| postgresql | opus SQL verbatim | 0.02 | 1000 | correct (3) | 0 | 0 | 8.0e-15 | 5.8e-14 | 0 | 0 | 0 |
| postgresql | opus SQL verbatim | 0.1 | 50 | correct (3) | 0 | 0 | 1.5e-15 | 1.1e-14 | 0 | 0 | 0 |
| postgresql | opus SQL verbatim | 0.1 | 1000 | correct (3) | 0 | 0 | 5.7e-15 | 3.7e-14 | 0 | 0 | 0 |
| postgresql | + opus's PG clamp (z < -700 -> 0) | 0.001 | 50 | error (3) | `DivisionByZero: division by zero` | | | | | | |
| postgresql | + opus's PG clamp (z < -700 -> 0) | 0.001 | 1000 | error (3) | `DivisionByZero: division by zero` | | | | | | |
| postgresql | clamp_nullif | 0.001 | 50 | correct (3) | 0 | 0 | 6.8e-16 | 4.4e-15 | 0 | 356148 | 0 |
| postgresql | clamp_nullif | 0.001 | 1000 | correct (3) | 0 | 0 | 6.1e-16 | 4.4e-15 | 0 | 4286 | 0 |
| postgresql | sum(w::numeric) (exact inverse), clamped | 0.001 | 50 | correct (3) | 0 | 0 | 8.8e-15 | 7.3e-14 | 0 | 356148 | 0 |
| postgresql | sum(w::numeric) (exact inverse), clamped | 0.001 | 1000 | correct (3) | 0 | 0 | 7.6e-15 | 6.6e-14 | 0 | 4286 | 0 |
| postgresql | sum(w::numeric) (exact inverse), clamped | 0.02 | 50 | correct (3) | 0 | 0 | 7.9e-15 | 6.6e-14 | 0 | 0 | 0 |
| postgresql | sum(w::numeric) (exact inverse), clamped | 0.02 | 1000 | correct (3) | 0 | 0 | 7.6e-15 | 6.0e-14 | 0 | 0 | 0 |
| postgresql | sum(w::numeric) (exact inverse), clamped | 0.1 | 50 | correct (3) | 0 | 0 | 5.4e-15 | 3.1e-14 | 0 | 0 | 0 |
| postgresql | sum(w::numeric) (exact inverse), clamped | 0.1 | 1000 | correct (3) | 0 | 0 | 5.4e-15 | 3.1e-14 | 0 | 0 | 0 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | 0.001 | 50 | **wrong-finite+NaN/NULL** (3) | 658133 | 19011 | 2.9e+88 | 2.0e+89 | 250438 | 356148 | 243510 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | 0.001 | 1000 | **wrong-finite+NaN/NULL** (3) | 796336 | 4097 | 6.5e+89 | 4.4e+90 | 175815 | 4286 | 0 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | 0.02 | 50 | **wrong-finite** (3) | 119309 | 0 | 4.59e+03 | 3.5e+04 | 494 | 0 | 0 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | 0.02 | 1000 | **wrong-finite** (3) | 12333 | 0 | 7.2e-03 | 0.0421 | 0 | 0 | 0 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | 0.1 | 50 | correct (3) | 0 | 0 | 1.2e-13 | 6.1e-13 | 0 | 0 | 0 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | 0.1 | 1000 | correct (3) | 0 | 0 | 6.4e-14 | 4.4e-13 | 0 | 0 | 0 |
| duckdb | opus SQL | 0.001 | 50 | correct (3) | 0 | 0 | 8.4e-16 | 5.3e-15 | 0 | 114732 | 0 |
| duckdb | opus SQL | 0.001 | 1000 | correct (3) | 0 | 0 | 6.3e-16 | 4.4e-15 | 0 | 1511 | 0 |
| duckdb | opus SQL | 0.02 | 50 | correct (3) | 0 | 0 | 1.7e-15 | 1.2e-14 | 0 | 0 | 0 |
| duckdb | opus SQL | 0.02 | 1000 | correct (3) | 0 | 0 | 2.7e-15 | 2.1e-14 | 0 | 0 | 0 |
| duckdb | opus SQL | 0.1 | 50 | correct (3) | 0 | 0 | 1.4e-15 | 8.9e-15 | 0 | 0 | 0 |
| duckdb | opus SQL | 0.1 | 1000 | correct (3) | 0 | 0 | 1.7e-15 | 1.1e-14 | 0 | 0 | 0 |
| clickhouse | opus SQL | 0.001 | 50 | correct (3) | 0 | 0 | 2.9e-15 | 1.6e-14 | 0 | 114732 | 0 |
| clickhouse | opus SQL | 0.001 | 1000 | correct (3) | 0 | 0 | 8.6e-16 | 5.3e-15 | 0 | 1511 | 0 |
| clickhouse | opus SQL | 0.02 | 50 | correct (3) | 0 | 0 | 1.5e-15 | 1.1e-14 | 0 | 0 | 0 |
| clickhouse | opus SQL | 0.02 | 1000 | correct (3) | 0 | 0 | 5.1e-15 | 3.6e-14 | 0 | 0 | 0 |
| clickhouse | opus SQL | 0.1 | 50 | correct (3) | 0 | 0 | 1.3e-15 | 8.9e-15 | 0 | 0 | 0 |
| clickhouse | opus SQL | 0.1 | 1000 | correct (3) | 0 | 0 | 4.7e-15 | 2.8e-14 | 0 | 0 | 0 |
| spark | opus SQL | 0.001 | 50 | correct (3) | 0 | 0 | 2.9e-15 | 1.6e-14 | 0 | 114732 | 0 |
| spark | opus SQL | 0.001 | 1000 | correct (3) | 0 | 0 | 6.1e-16 | 4.4e-15 | 0 | 1511 | 0 |
| spark | opus SQL | 0.02 | 50 | correct (3) | 0 | 0 | 2.0e-15 | 1.4e-14 | 0 | 0 | 0 |
| spark | opus SQL | 0.02 | 1000 | correct (3) | 0 | 0 | 8.0e-15 | 5.8e-14 | 0 | 0 | 0 |
| spark | opus SQL | 0.1 | 50 | correct (3) | 0 | 0 | 1.5e-15 | 1.1e-14 | 0 | 0 | 0 |
| spark | opus SQL | 0.1 | 1000 | correct (3) | 0 | 0 | 5.7e-15 | 3.7e-14 | 0 | 0 | 0 |
| numpy emulation | plain add/subtract, never recompute | 0.001 | 50 | wrong-finite+NaN/NULL (3) | 875198 | 20738 | 5.0e+107 | 6.5e+107 | 258471 | 114732 | 92413 |
| numpy emulation | plain add/subtract, never recompute | 0.001 | 1000 | wrong-finite+NaN/NULL (3) | 799015 | 1413 | 6.5e+89 | 4.4e+90 | 176101 | 1511 | 0 |
| numpy emulation | plain add/subtract, never recompute | 0.02 | 50 | wrong-finite (3) | 119291 | 0 | 803 | 6.2e+03 | 500 | 0 | 0 |
| numpy emulation | plain add/subtract, never recompute | 0.02 | 1000 | wrong-finite (3) | 12343 | 0 | 5.9e-03 | 0.0343 | 0 | 0 | 0 |
| numpy emulation | plain add/subtract, never recompute | 0.1 | 50 | correct (3) | 0 | 0 | 1.3e-13 | 6.9e-13 | 0 | 0 | 0 |
| numpy emulation | plain add/subtract, never recompute | 0.1 | 1000 | correct (3) | 0 | 0 | 5.8e-14 | 3.8e-13 | 0 | 0 | 0 |

Worst row per engine at eps = 0.02, frame 50 (the T4 setting):

| engine | spelling | query | id | engine value | formula ref F | true softmax ref T |
|---|---|---|---|---|---|---|
| sqlite_stdlib | opus SQL | q2 | 269623 | 35471.6 | 7.7274 | 7.7274 |
| datafusion | opus SQL | q2 | 269582 | -6195.03 | 7.72097 | 7.72097 |
| polars | rolling_sum().over() | q2 | 283868 | 1.58208 | 6.29591 | 6.29591 |
| polars | opus SQL via pl.SQLContext | q2 | 283868 | 1.58208 | 6.29591 | 6.29591 |
| pandas | groupby().rolling().sum() | q2 | 283868 | 1.58208 | 6.29591 | 6.29591 |
| postgresql | manual's `unsafe_sum` (float8 MINVFUNC), clamped | q2 | 269623 | 35471.6 | 7.7274 | 7.7274 |

### Minimal non-exponential cases: `SUM(x) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`

`one_large`: x = [1.0, 1.0, 1e+20, 1.0, 1.0, 1.0, 1.0]; correctly rounded frame sums = [1.0, 2.0, 1e+20, 1e+20, 1e+20, 3.0, 3.0]

`two_large`: x = [1.0, 1e+40, 1e+20, 1.0, 1.0, 1.0, 1.0]; correctly rounded frame sums = [1.0, 1e+40, 1e+40, 1e+40, 1e+20, 3.0, 3.0]

| engine | spelling | one_large | two_large |
|---|---|---|---|
| sqlite_stdlib | default | **wrong** ['1', '2', '1e+20', '1e+20', '1e+20', '1', '1'] | **wrong** ['1', '1e+40', '1e+40', '1e+40', '1', '-1e+20', '-1e+20'] |
| sqlite_apsw | default | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | **wrong** ['1', '1e+40', '1e+40', '1e+40', '1e+20', '0', '0'] |
| datafusion | default | **wrong** ['1', '2', '1e+20', '1e+20', '1e+20', '0', '0'] | **wrong** ['1', '1e+40', '1e+40', '1e+40', '0', '-1e+20', '-1e+20'] |
| polars | default | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | **wrong** ['1', '1e+40', '1e+40', '1e+40', '1e+20', '1', '1'] |
| pandas | default | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | ok ['1', '1e+40', '1e+40', '1e+40', '1e+20', '3', '3'] |
| postgresql | SUM | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | ok ['1', '1e+40', '1e+40', '1e+40', '1e+20', '3', '3'] |
| postgresql | unsafe_sum | **wrong** ['1', '2', '1e+20', '1e+20', '1e+20', '1', '1'] | **wrong** ['1', '1e+40', '1e+40', '1e+40', '1', '-1e+20', '-1e+20'] |
| duckdb | default | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | ok ['1', '1e+40', '1e+40', '1e+40', '1e+20', '3', '3'] |
| clickhouse | default | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | ok ['1', '1e+40', '1e+40', '1e+40', '1e+20', '3', '3'] |
| spark | default | ok ['1', '2', '1e+20', '1e+20', '1e+20', '3', '3'] | ok ['1', '1e+40', '1e+40', '1e+40', '1e+20', '3', '3'] |

<!-- END GENERATED -->

## Not run (permission refused): MySQL 8.4.3 and MariaDB 11.4.4

The tarballs downloaded (MariaDB sha256 matched the published `sha256sums.txt`; MySQL md5
`465fa52c59b1cee6e15849e97fa51607`), but starting the downloaded `mysqld`/`mariadbd` was refused, so
they were deleted. `run_engine.py` has both engines (MySQL runs every cell with
`windowing_use_high_precision` ON and OFF). To run them yourself:

```bash
cd paper_icde/experiments/window_inverse
mkdir -p ~/opt && cd ~/opt
curl -LO https://cdn.mysql.com/archives/mysql-8.4/mysql-8.4.3-linux-glibc2.17-x86_64-minimal.tar.xz && tar xJf mysql-8.4.3-*.tar.xz
curl -LO https://archive.mariadb.org/mariadb-11.4.4/bintar-linux-systemd-x86_64/mariadb-11.4.4-linux-systemd-x86_64.tar.gz && tar xzf mariadb-11.4.4-*.tar.gz
cd -; python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -c "import common; common.prepare()"
MYSQL_BASE=~/opt/mysql-8.4.3-linux-glibc2.17-x86_64-minimal .venv/bin/python run_engine.py mysql
MARIADB_BASE=~/opt/mariadb-11.4.4-linux-systemd-x86_64 .venv/bin/python run_engine.py mariadb
.venv/bin/python assemble.py      # adds them to results.json and the tables above
```
Both servers run as the current user on port 28872 with a socket and data dir under `work/`, and are
shut down by the script. Expected from source: MySQL default correct, MySQL OFF and MariaDB like
SQLite 3.37.2 / DataFusion.

## Files and rerun

- `common.py` — data prep (`work/reviews.parquet`), grid, SQL text, minimal cases.
- `run_engine.py <engine>` — one engine per process; writes `work/out/<engine>.{npz,json}`.
  PostgreSQL starts its own scratch cluster from `~/.pgrx/17.10/pg-release-install` on port 28871
  and stops and deletes it at the end. Spark runs `local[8]` with Java 8.
- `assemble.py` — references (`work/refs.npz`), grading, `results.json`, the generated tables.
- `results.json` — versions, per-cell grades (both references), worst rows, minimal-case outputs,
  the SQL used, the numpy add/subtract emulation.
- `requirements.txt` — pinned Python packages (the venv itself is deleted after the run).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import common; common.prepare()"
for e in sqlite_stdlib sqlite_apsw duckdb postgresql clickhouse datafusion spark polars pandas; do .venv/bin/python run_engine.py $e; done
.venv/bin/python assemble.py
rm -rf work .venv
```

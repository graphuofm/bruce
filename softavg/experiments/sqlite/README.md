# SQLite: before / after `softavg`

Same SQLite (the system library behind Python's `sqlite3`), same table, same
score function, only the aggregate spelling changes (PROTOCOL.md). The
extension is `../../engines/sqlite` (softagg-sqlite, linking softagg-core).
**All timings are smoke numbers on a shared machine**; the main thread reruns
them on an idle box.

## What is measured

* **Table.** `work/imdb.db`: `imdb(id, facet, value, filter_num, emb BLOB)`,
  the 384-d float32 embedding stored as a BLOB in sqlite-vec's format, plus
  a sqlite-vec `vec0` table (`distance_metric=cosine`, metadata column
  `filter_num`) for KNN. Query: `GROUP BY facet` over `filter_num >= 2000`.
* **Score.** `softagg_dot(emb, :q)` (this extension; float32 BLOB inner
  product summed in float64). SQLite has no native vector type. sqlite-vec's
  `1 - vec_distance_cosine` gives the same ranking on these unit vectors; its
  cost is reported next to ours.
* **Spellings.** `naive`, `maxshift_cte` (per-group MAX in a CTE, joined
  back), `maxshift_window` (`MAX(s) OVER (PARTITION BY facet)`), `clamp`
  (max-shift + `exp(max(x, -700))`), `softavg` (window-capable) and
  `softavg_agg` (aggregate-only twin: bare `ScalarAcc` state, no `InvAcc`
  error bound or 64-row prefix record; bit-identical result). Each runs over the same
  scores in two modes: **agg** (scores precomputed into a TEMP table,
  untimed: isolates the aggregate) and **e2e** (`WITH sc AS MATERIALIZED
  (score from the BLOB)` + the same aggregate, timed). `softavg_direct` is
  the one-statement spelling without the CTE.
* **Why MATERIALIZED in e2e.** SQLite flattens a plain CTE and evaluates
  the score once per reference, and its `GROUP BY` sorter carries the
  referenced columns (the 1.5 KB BLOB) rather than the computed score. Both
  favour whoever references `s` least, so every e2e spelling gets the same
  materialized score CTE.
* **top-k.** sqlite-vec `vec0` KNN (`emb MATCH :q AND k = :k AND filter_num
  >= 2000`), then `AVG(value)` per facet: coverage of facets and the gap to
  the full softavg. sqlite-vec 0.1.9 caps `k` at 4096 (the `k = 10000` cell
  records its error verbatim). vec0 in 0.1.x is an exact brute-force scan,
  so the top-k gap is truncation, not approximation.
* **Sliding window.** `ORDER BY id ROWS BETWEEN 49 PRECEDING AND CURRENT
  ROW` over the admitted rows (scores precomputed): `win_softavg`
  (`xInverse`, O(1)/row + exact refolds), `win_naive` (SQLite's own `SUM`
  window, which also subtracts on frame exit), `win_global_shift` (shift by
  the table-wide max: the only shift one window pass can express, since
  window calls cannot nest), `win_exact_sql` (per-frame `MAX(s) OVER w`, then
  a correlated subquery over the frame: exact, O(rows x frame)).
* **Reference.** numpy float64: `float64(E) @ q`, two-pass max-shift per
  group / per frame. `silent` = returned without an SQL error.
* **SQLite math.** `exp()` exists only when SQLite is compiled with
  `SQLITE_ENABLE_MATH_FUNCTIONS` (3.35+); this build has it (recorded in
  results.json as `sqlite_math_functions`). SQLite's `exp` overflows to
  `Inf` and underflows to 0 silently, and any NaN becomes NULL: so the
  failure mode here is a **silent NULL**, not an error (PG) or a NaN (DuckDB).

## Run

```bash
cd ../../engines/sqlite && nice -n 10 cargo build --release && nice -n 10 cargo build --release --example guard_bound
cd ../../experiments/sqlite
nice -n 10 python3 run.py --n-queries 200 --n-time 5 --reps 3 --n-window-queries 3   # -> results.json
nice -n 10 python3 run_guard_bound.py 3                                              # -> guard_bound.json
python3 report.py                                                                    # tables below
rm -rf work/                                                                         # 1.6 GB derived DB
```

`deps/sqlite_vec-0.1.9/vec0.so` was extracted from the PyPI wheel
`sqlite_vec-0.1.9-py3-none-manylinux_2_17_x86_64` (`pip download
sqlite-vec`); its sha256 is in results.json.

## Results

Generated from results.json / guard_bound.json / guard_tau_old_core.json by
`report.py`; do not edit by hand. `guard_tau_old_core.json` is the record of the
previous core (z-only guard, TAU = 1e-6, with a host-side tau sweep); its
generator (`examples/guard_tau.rs`) was removed when the core dropped TAU.

<!-- BEGIN GENERATED (report.py) -->
Host: SQLite 3.37.2 (Python 3.10.12 stdlib sqlite3, system libsqlite3); math functions compiled in: True; softagg-sqlite softagg-sqlite 0.1.0 (softagg-core path dep; prefix_cap=64, shadow_cap=1048576); build release; softagg-core 0.1.0; sqlite-vec v0.1.9. Threads: 1. Timing status: **smoke, machine shared** (load at start: cpu busy 0.071, 1-min load avg 3.88).

Data: 459,865 rows, 232,828 admitted by `filter_num >= 2000`, 28 facets. Accuracy over 200 queries; latency over the first 5 queries, 3 reps each (per query: min over reps; reported: median over timed queries. Aggregate cells fetch the 28 result rows; window cells are timed as SELECT count(r), total(r) FROM (window query) so the 232,828 output rows are not shipped to Python).

### GROUP BY facet: before / after

Silent = a result returned without any SQL error. `NULL grp` counts facet results that came back NULL although the reference is finite (summed over queries; in parentheses: queries affected). Max rel err is over the non-NULL finite results. *agg ms*: scores precomputed (aggregate only); *e2e ms*: score from the BLOB in the same statement (`WITH sc AS MATERIALIZED`). Error columns come from the agg-mode run over all 200 queries, except `softavg_direct` (e2e only), which covers the 5 timed queries. The shared max rel err of the exact spellings at sharp eps is the score path (softagg_dot vs numpy summation order, amplified by 1/eps), not the aggregate.

| eps | spelling | SQL errors | silent NULL grp | silent Inf grp | max rel err | agg ms | e2e ms |
|---|---|---|---|---|---|---|---|
| 0.0001 | naive | 0 | 5600 (200) | 0 (0) | 0 | 65 | 331 |
| 0.0001 | maxshift_cte | 0 | 0 (0) | 0 (0) | 7.8e-13 | 147 | 648 |
| 0.0001 | maxshift_window | 0 | 0 (0) | 0 (0) | 7.8e-13 | 229 | 357 |
| 0.0001 | clamp | 0 | 0 (0) | 0 (0) | 7.8e-13 | 156 | 639 |
| 0.0001 | softavg | 0 | 0 (0) | 0 (0) | 7.8e-13 | 61 | 306 |
| 0.0001 | softavg_agg | 0 | 0 (0) | 0 (0) | 7.8e-13 | 57 | 296 |
| 0.0001 | softavg_direct | 0 | 0 (0) | 0 (0) | 3.8e-14 | - | 302 |
| 0.001 | naive | 0 | 344 (152) | 9 (9) | 5.6e-14 | 72 | 336 |
| 0.001 | maxshift_cte | 0 | 0 (0) | 0 (0) | 7.8e-14 | 148 | 668 |
| 0.001 | maxshift_window | 0 | 0 (0) | 0 (0) | 7.8e-14 | 223 | 359 |
| 0.001 | clamp | 0 | 0 (0) | 0 (0) | 7.8e-14 | 151 | 651 |
| 0.001 | softavg | 0 | 0 (0) | 0 (0) | 7.8e-14 | 57 | 304 |
| 0.001 | softavg_agg | 0 | 0 (0) | 0 (0) | 7.8e-14 | 56 | 291 |
| 0.001 | softavg_direct | 0 | 0 (0) | 0 (0) | 8.3e-15 | - | 300 |
| 0.02 | naive | 0 | 0 (0) | 0 (0) | 6.2e-13 | 69 | 337 |
| 0.02 | maxshift_cte | 0 | 0 (0) | 0 (0) | 4.0e-15 | 148 | 649 |
| 0.02 | maxshift_window | 0 | 0 (0) | 0 (0) | 4.0e-15 | 224 | 365 |
| 0.02 | clamp | 0 | 0 (0) | 0 (0) | 4.0e-15 | 151 | 647 |
| 0.02 | softavg | 0 | 0 (0) | 0 (0) | 7.2e-14 | 57 | 292 |
| 0.02 | softavg_agg | 0 | 0 (0) | 0 (0) | 7.2e-14 | 55 | 283 |
| 0.02 | softavg_direct | 0 | 0 (0) | 0 (0) | 1.9e-14 | - | 292 |
| 0.1 | naive | 0 | 0 (0) | 0 (0) | 7.5e-14 | 66 | 327 |
| 0.1 | maxshift_cte | 0 | 0 (0) | 0 (0) | 5.1e-15 | 144 | 629 |
| 0.1 | maxshift_window | 0 | 0 (0) | 0 (0) | 5.1e-15 | 226 | 361 |
| 0.1 | clamp | 0 | 0 (0) | 0 (0) | 5.1e-15 | 153 | 618 |
| 0.1 | softavg | 0 | 0 (0) | 0 (0) | 6.0e-14 | 57 | 300 |
| 0.1 | softavg_agg | 0 | 0 (0) | 0 (0) | 6.0e-14 | 57 | 295 |
| 0.1 | softavg_direct | 0 | 0 (0) | 0 (0) | 2.3e-14 | - | 305 |
| 1.0 | naive | 0 | 0 (0) | 0 (0) | 8.5e-14 | 64 | 335 |
| 1.0 | maxshift_cte | 0 | 0 (0) | 0 (0) | 8.0e-16 | 148 | 648 |
| 1.0 | maxshift_window | 0 | 0 (0) | 0 (0) | 8.0e-16 | 224 | 370 |
| 1.0 | clamp | 0 | 0 (0) | 0 (0) | 8.0e-16 | 158 | 650 |
| 1.0 | softavg | 0 | 0 (0) | 0 (0) | 5.1e-14 | 58 | 301 |
| 1.0 | softavg_agg | 0 | 0 (0) | 0 (0) | 5.1e-14 | 59 | 297 |
| 1.0 | softavg_direct | 0 | 0 (0) | 0 (0) | 1.3e-14 | - | 302 |
| inf | naive | 0 | 0 (0) | 0 (0) | 0 | 65 | 335 |
| inf | maxshift_cte | 0 | 0 (0) | 0 (0) | 0 | 152 | 646 |
| inf | maxshift_window | 0 | 0 (0) | 0 (0) | 0 | 227 | 360 |
| inf | clamp | 0 | 0 (0) | 0 (0) | 0 | 158 | 616 |
| inf | softavg | 0 | 0 (0) | 0 (0) | 0 | 56 | 296 |
| inf | softavg_agg | 0 | 0 (0) | 0 (0) | 0 | 59 | 295 |
| inf | softavg_direct | 0 | 0 (0) | 0 (0) | 0 | - | 311 |

### sqlite-vec vec0 KNN top-k + AVG (the index-side alternative)

Coverage = facets present in the top-k / admitted facets. The last columns compare the top-k AVG with the full softavg reference (it is a different quantity; shown to size the gap).

| k | errors | coverage mean | coverage min | median ms | max rel err vs softavg eps=0.02 | vs eps=inf |
|---|---|---|---|---|---|---|
| 100 | 0/200 | 48.9% | 3.6% | 186 | 4.44 | 0.81 |
| 1000 | 0/200 | 86.5% | 60.7% | 601 | 4.54 | 0.81 |
| 4096 | 0/200 | 96.8% | 82.1% | 606 | 4.49 | 0.80 |
| 10000 | 200/200 | - | - | - | - | - |

Verbatim errors: `OperationalError: k value in knn query too large, provided 10000 and the limit is 4096`

Score function alone (SUM over admitted rows, median ms): softagg_dot 106, sqlite-vec `1 - vec_distance_cosine` 137.

### Sliding window: `ORDER BY id ROWS BETWEEN 49 PRECEDING AND CURRENT ROW` over 232,828 rows, 3 queries

| eps | spelling | SQL errors | silent NULL rows | silent Inf rows | max rel err | median ms | refolds |
|---|---|---|---|---|---|---|---|
| 0.0001 | win_softavg | 0 | 0 | 0 | 2.7e-12 | 106 | 14478 |
| 0.0001 | win_naive | 0 | 698484 | 0 | 0 | 100 | - |
| 0.0001 | win_global_shift | 0 | 612540 | 0 | 5.1e+00 | 114 | - |
| 0.0001 | win_exact_sql | 0 | 0 | 0 | 3.4e-13 | 2624 | - |
| 0.001 | win_softavg | 0 | 0 | 0 | 6.1e-12 | 101 | 12222 |
| 0.001 | win_naive | 0 | 365335 | 0 | 1.4e+148 | 98 | - |
| 0.001 | win_global_shift | 0 | 3631 | 0 | 7.9e+132 | 118 | - |
| 0.001 | win_exact_sql | 0 | 0 | 0 | 5.1e-14 | 2493 | - |
| 0.02 | win_softavg | 0 | 0 | 0 | 1.6e-11 | 98 | 1124 |
| 0.02 | win_naive | 0 | 0 | 0 | 9.8e+04 | 98 | - |
| 0.02 | win_global_shift | 0 | 0 | 0 | 7.9e-01 | 112 | - |
| 0.02 | win_exact_sql | 0 | 0 | 0 | 4.6e-15 | 2493 | - |
| 0.1 | win_softavg | 0 | 0 | 0 | 1.2e-13 | 101 | 3 |
| 0.1 | win_naive | 0 | 0 | 0 | 1.5e-13 | 98 | - |
| 0.1 | win_global_shift | 0 | 0 | 0 | 2.1e-13 | 112 | - |
| 0.1 | win_exact_sql | 0 | 0 | 0 | 1.7e-15 | 2487 | - |
| 1.0 | win_softavg | 0 | 0 | 0 | 3.9e-14 | 97 | 0 |
| 1.0 | win_naive | 0 | 0 | 0 | 2.5e-14 | 98 | - |
| 1.0 | win_global_shift | 0 | 0 | 0 | 3.1e-14 | 113 | - |
| 1.0 | win_exact_sql | 0 | 0 | 0 | 1.4e-15 | 2503 | - |
| inf | win_softavg | 0 | 0 | 0 | 2.0e-14 | 95 | 0 |
| inf | win_naive | 0 | 0 | 0 | 2.0e-14 | 98 | - |
| inf | win_global_shift | 0 | 0 | 0 | 2.0e-14 | 113 | - |
| inf | win_exact_sql | 0 | 0 | 0 | 9.8e-16 | 2477 | - |

Same `win_softavg` cells under the previous core (z-only guard): win_softavg under the previous softagg-core (z-only guard, TAU=1e-6), same run.py settings (3 queries). The previous results.json was overwritten by this run; these values are transcribed from report.py's rendering of it (2 significant digits) in the first SQLite report of 2026-11-05. The InvAcc-level old-vs-new comparison with full precision is guard_tau_old_core.json vs guard_bound.json.

| eps | refolds old | max rel err old | median ms old |
|---|---|---|---|
| 0.0001 | 14333 | 1.2e-10 | 108 |
| 0.001 | 11199 | 2.9e-10 | 104 |
| 0.02 | 184 | 3.8e-09 | 102 |
| 0.1 | 0 | 1.6e-13 | 113 |
| 1.0 | 0 | 3.9e-14 | 102 |
| inf | 0 | 2.0e-14 | 100 |

### Window drift: new core guard (running error bound) vs old (z-only, TAU = 1e-6)

guard_bound.json vs guard_tau_old_core.json: 3 queries x 232,828 rows, frame 50, InvAcc driven directly, error vs a fresh fold per frame. Abs err is in units of max|v|; the PROTOCOL window bound is 1e-9. `err/bound` = worst ratio of the true error to the core's own `value_error_bound()`.

| eps | refolds new | refolds old | max abs err / max\|v\| new | old | max rel err new | old | bound violations new | max err/bound new |
|---|---|---|---|---|---|---|---|---|
| 0.0001 | 14177 | 14034 | 4.8e-13 | 8.7e-11 | 2.7e-12 | 1.2e-10 | 0 | 0.39 |
| 0.001 | 11958 | 10954 | 3.0e-12 | 2.3e-10 | 6.1e-12 | 2.9e-10 | 0 | 0.92 |
| 0.02 | 1103 | 183 | 8.4e-12 | 1.8e-09 | 1.6e-11 | 3.8e-09 | 0 | 0.80 |
| 0.1 | 3 | 0 | 6.0e-14 | 6.0e-14 | 9.6e-14 | 9.6e-14 | 0 | 0.08 |
| 1 | 0 | 0 | 7.3e-15 | 7.3e-15 | 1.2e-14 | 1.2e-14 | 0 | 0.11 |
<!-- END GENERATED (report.py) -->

## Findings (numbers: the generated tables above)

1. **Naive SQL fails silently in SQLite, as NULL.** At eps = 1e-4 every
   facet of every query comes back NULL (`exp` overflows to Inf, Inf/Inf is
   NaN, SQLite turns NaN into NULL) with no error; at eps = 1e-3 some
   facets come back NULL and some Inf. Nothing in the result says
   "overflow". PG raises, DuckDB returns NaN; SQLite returns a NULL that
   looks like "no data".
2. **Max-shift fixes the value and costs a second pass.** Both max-shift
   spellings and the clamp are exact at every eps (SQLite's `exp`
   underflows to 0 silently, so the clamp is not needed here), but they are
   the slowest aggregate spellings: the CTE scans the scores twice plus a
   join, the window form sorts for `PARTITION BY`.
3. **`softavg` is one pass, exact, and the cheapest aggregate spelling.**
   Aggregate-only it is at or below the naive spelling's time, i.e. the
   numerically safe form costs nothing over the unsafe one; end to end the
   score computation (one 384-d dot per admitted row) dominates every
   spelling. `softavg_direct` (no CTE) costs about the same as the
   materialized form here. The aggregate-only twin `softavg_agg` is within
   run-to-run noise of `softavg` (the `softavg` vs `softavg_agg` rows): at
   SQLite's per-row interpretation cost, the `InvAcc` error-bound
   bookkeeping and the 64-row prefix record are not visible.
4. **Index-side top-k is not a substitute.** sqlite-vec's `vec0` KNN caps k
   at 4096 (k = 10000 is a hard error), misses facets even at the cap (see
   coverage min), and its AVG is a different quantity from softavg at any
   eps.
5. **Sliding windows: SQLite's own `SUM` window is silently wrong, not just
   overflow-prone.** SQLite implements window `SUM` with an inverse
   (subtract on frame exit), so `exp(s/eps)` terms leaving the frame cancel
   catastrophically: at eps <= 0.02 `win_naive` returns finite numbers
   that are wrong by orders of magnitude, besides the NULLs. Shifting by
   the global max (the only shift one window pass can express) still
   underflows/cancels. The exact SQL spelling (per-frame max + correlated
   subquery) is correct but O(rows x frame), more than an order of magnitude slower.
   `win_softavg` runs at the speed of the naive window and is exact up to
   the core guard's tolerance, refolding when the guard trips (refold
   counts in the table).
6. **Core guard precision (old vs new core).** Under the previous core
   (z-only guard, TAU = 1e-6) `win_softavg` at eps = 0.02 exceeded the
   window tolerance (see the previous-core table). The current core keeps a
   running error bound; in the drift table its worst error at every eps is
   far inside the PROTOCOL's `1e-9 * max|v|`, the true error never exceeded
   the core's own `value_error_bound()`, and the price is more refolds at
   eps = 0.02 (and slightly more at sharp eps), with no visible change in
   `win_softavg` latency. The SQLite window table reports relative error;
   the absolute error in units of max|v| is the drift table's (same rows,
   same `InvAcc` path). The conformance window cases pass.

Not measured here: threads (SQLite has no intra-query parallelism),
maintenance (no IVM in SQLite; triggers would be the analogue), Amazon /
StackExchange scale.

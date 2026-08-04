# bruce_tool — Changelog

## 2026-08-03 (orchestrator: verifier follow-ups)

Closing the items the night-six independent audit reported but left for
the owning tracks.

- **bruce-core/src/mask.rs** — `dot_f32`'s doc comment pointed at
  `m2_mixed_precision/results.json`, which does not exist; the AVX2
  numbers live in `results_m2.json`. A dangling pointer to measured
  numbers is a doctrine violation even when the numbers are right.
- **docs/UNSAFE_INVENTORY.md** — bruce-cdc's file count said 6; the
  cdc-v2 track added `durable.rs` after the audit ran, so it is 7. The
  unsafe count (0) and the audit's conclusion are unchanged.
- **m6_hnsw/results.json** — the Track-H HNSW correctness numbers
  (recall@10 0.912 unfiltered, 0.996 filtered, 0.928 survivor, the
  100k uniform-random ef ladder 0.45/0.55/0.83/0.95) existed only as a
  Markdown table and as test assertions, in no results.json. Landed as
  keys `correctness_track_h` and `scale_100k_uniform_random`, captured
  from re-running the suites and parsing their printed output rather
  than transcribed from the README.
- **CHANGELOG.md** — three night-six sections (lifecycle-safety,
  f32-tail, cdc-v2) had been appended to the end of the file, below
  2026-08-01 entries; moved up beside the other night-six sections so
  the file reads reverse-chronologically. The audit section's own
  header was dated 2026-08-04; corrected to 2026-08-03.

Not closed here, and it needs a decision rather than a patch: **the
entire night-one-through-six campaign is uncommitted.** `git log` ends
at 2026-06-12; six crates, ~530 tests and every measurement above live
only in the working tree, with no stash and no reflog entry. The audit
raised it because it could not verify the f32 track's "kernel ends
byte-identical" claim against any artifact — but the bigger point is
that there is nothing to fall back to.


## 2026-08-03 (night-six INDEPENDENT AUDIT — verification only, two doc
## corrections + one fmt sweep; no engine behaviour changed)

Full re-run of every suite on the final tree by a verifier who wrote
none of the six night-six tracks. Suites: cargo test --workspace 222/0
(+3 `#[ignore]` soaks re-run in release, all pass), pytest 298/0 on a
freshly rebuilt wheel, bruce-cdc 53/0, bruce-pg 39/0 on pg17 AND pg16,
scripts/regress.sh OVERALL PASS (golden one_query 14.5 ms in band
[10,20]), clippy 0 warnings workspace + bruce-cdc + bruce-pg (workspace
clippy re-run from a cold target dir, 233 crates checked).

- **docs/TESTING_MATRIX.md** — the gap-ledger section header still read
  "Residual gaps (confirmed open after the 2026-08-03 campaign)" while
  6 of its 9 entries had been closed during the campaign and were
  individually annotated `[CLOSED]`/`FIXED`. Header rewritten so the
  section states plainly which entries are closed audit trail and which
  are actually open. Every `[CLOSED]`/`FIXED` entry's named pin was
  checked to exist and pass.
- **docs/TESTING_MATRIX.md** — the M4 planner gap claimed "planning
  costs ~12x the execution of the plan it chooses". Recomputed from
  m6_hnsw/results.json (`planner_v1_regret.grid[].plan_time_ms`): the
  12x is against the ORACLE plan (max plan/oracle = 12.82); against the
  plan actually chosen the ratio never exceeds 3.93x. Corrected; the
  m6_hnsw/README.md wording ("the work it is choosing between") was
  already right.
- **cargo fmt --all** — bruce-server/src/flight.rs AND
  bruce-server/src/npy.rs were unformatted (pre-existing, from an
  earlier track; the hnsw-finish caveat named only flight.rs).
  Formatted; bruce-server tests re-run green, clippy still 0.


## 2026-08-03 (night six, hnsw-finish track: the HNSW access path, end to end)

Closes backlog #6 and seeds M4. Covers the operator/planner/lifecycle
code landed earlier this night (never changelogged), tonight's
measurement campaign, the f32 delete-gap fix, and a neighbor-selection
result that comes out opposite on synthetic and real data.

**The access path (bruce-query, landed earlier this night — recorded
here for the first time).**

- **bruce-query/src/cost.rs** — `HNSW_TAIL_TOL = 1e-6` with its full
  derivation (omitting softmax mass `delta` perturbs the answer by at
  most `2*delta*vmax`, so 1e-6 sits under the engine's tightest 1e-5
  precision-contract ceiling with >= 100x headroom at the sharp
  temperatures this path serves). `hnsw_tail_bound(n, s_k, s_max,
  eps) = n*exp((s_k-s_max)/eps)` — max-anchored, deliberately
  conservative (uses `n` not `n-k`, `Z >= 1` not `Z >= z_top`), total
  on all inputs (NaN or eps <= 0 -> +inf, i.e. refuse).
  `HnswTailPrediction` + `predict_hnsw_tail` /
  `predict_hnsw_tail_from_sims`: sketch-quantile prediction with
  conservative choices on both ends (the sample max UNDER-estimates
  the true max; a sample rank below k OVER-estimates the k-th best),
  so admission errs toward refusal, plus a `resolution_limited` flag
  for the extreme-value regime a uniform sample cannot resolve.
- **bruce-query/src/physical.rs** — `HnswTopKScan` variant + EXPLAIN
  rendering (k, ef, predicted tail, tolerance, "runtime re-checked").
- **bruce-query/src/planner.rs** — `plan_with_indexes`, the k ladder
  {16, 64, 256} with `ef = max(4k, 64)`, smallest admissible k wins;
  every refusal carries its concrete reason into EXPLAIN (grouped
  shape / eps=0 / no bound param / no sketch / empty index /
  resolution-limited / predicted tail over tolerance).
- **bruce-query/src/exec.rs** — execution with exact f64 rescoring of
  the probe hits, a RUNTIME re-check of the achieved tail bound, and
  an exact-fold fallback when the probe misses it (the planner's
  admission is a prediction, never a semantic contract); typed
  refusals for eps <= 0, KeyF32 storage, an out-of-sync index, and
  multi-group admitted sets.
- **bruce-query/src/db.rs** — `create_index(table, key_col)` and the
  index lifecycle: `register`-replace DROPS indexes exactly as it
  drops views, `delete_where` tombstones ids and re-maps survivors in
  the O(n) pass the delete already pays, `insert_row` extends the
  graph incrementally.
- **bruce-query/tests/topk_access_path.rs** — 14 tests (totality
  guards on hand-built plans, `create_index` typed errors, both
  lifecycle cascades, differential correctness against the exact
  fused fold across eps and selectivity).

**New this leg.**

- **bruce-query/src/exec.rs** — `HnswProbeStats` +
  `execute_hnsw_with_stats`: EXPLAIN-ANALYZE for the index path
  (admitted rows, probe hits, s_max/s_k, ACHIEVED tail bound,
  `fell_back`). The `HnswTopKScan` arm is factored into `exec_hnsw` so
  the answer and its diagnostics come from one code path —
  `execute_with_indexes` drops the stats, the new entry point keeps
  them. +2 tests (topk_access_path 14 -> 16): the three runtime
  regimes on one table (admitted / fallback / probe-covers-everything,
  where the tail is exactly 0), and the typed refusal of any other
  plan variant.
- **bruce-query/src/db.rs — DEFECT FIXED (a gap the f32-tail track
  opened the same night):** `delete_where` on a table carrying an
  f32-KEYED maintained view returned a typed error, because the
  per-view survivor capture read keys through the KeyF64-only
  `views::key_col_of`. New `key_rows_f64` reads dtype-polymorphically
  (KeyF64 borrows zero-copy; KeyF32 widens once into an owned f64
  matrix). The round trip is EXACT — `views::score_slice` casts every
  component back down to f32 before dotting for an f32 view — so the
  maintained state after a delete equals a from-scratch rebuild BIT
  FOR BIT, which is what the flipped tests assert. Cost owned in the
  doc comment: the KeyF32 arm allocates n*d*8 B per f32 view per call,
  paid only by tables that actually carry one, and it keeps the
  survivor iterator lazy so the bounded re-anchor still walks a single
  group.
- **bruce-query/src/db.rs** — `create_index` now refuses KeyF32 by an
  EXPLICIT dtype check instead of inheriting an accessor's narrowness
  (the v1 restriction is about the f64 rescore contract and must not
  silently disappear when an accessor widens). The now-unused
  `views::key_col_of` is removed and the three doc comments that
  pinned on it are corrected.
- **Pins flipped positive:** `bruce-query/tests/error_totality.rs`
  `create_view_f32_semantics` (delete the group's ANCHOR scorer, then
  a non-anchor row; after each, maintained state == rebuild, compared
  on raw bits) and `bruce-py/tests/test_f32_views.py`
  `test_delete_where_maintains_f32_view` (was
  `test_delete_where_with_f32_view_is_typed_error`).
- **bruce-core/src/hnsw.rs** — `NeighborSelection::{KeepBest,
  Diversity}` build-time policy. `Diversity` is Malkov & Yashunin
  Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) without `extendCandidates`
  / `keepPrunedConnections`, applied at BOTH call sites (a new node's
  own links and the backlink prune), tie-admitting so an exact
  duplicate never loses its only link. `set_selection` refuses after
  the first insert (typed `InvalidArgument`) rather than leaving a
  half-and-half graph. **Default unchanged**, so every previously
  measured recall number still describes the shipped graph. +5 unit
  tests (bruce-core 90 -> 95).
- **docs/TESTING_MATRIX.md** — residual gaps: struck the exec.rs
  hand-built-`TopKContractScan` dim-mismatch panic (closed by
  `check_param_dim`, driven in topk_access_path.rs) and the f32-view
  delete gap (closed above).

**Measured (paper_sigmod_bruce/experiments/m6_hnsw/results.json).**

- `planner_v1_regret` — the M4 seed. Real 459,865 x 384 movie
  embeddings, real query vector, real `year >= T` predicates for the
  selectivity axis, constant group column (the v1 operator serves
  SINGLE-GROUP shapes only; the real `GROUP BY genre` shape is run as
  `grouped_control` and shows the typed refusal verbatim). 20 grid
  points (eps in {1e-4, 1e-3, 1e-2, 0.1, 1.0} x selectivity in {1.000,
  0.183, 0.050, 0.011}); at each one EVERY alternative is executed —
  the fused scan and all three k-ladder rungs, admitted or refused —
  so the oracle is measured, not argued. Medians of 11, loadavg at
  both ends. Index build 294.6 s, 915 MB.
  **Max regret 16.80x, median 1.00x, 12/20 points perfect.**
  Decision quality decomposes cleanly:
  * direction is never wrong — 0 points where the planner refused the
    index and the index would have won;
  * admission is sound, never optimistic — 0 chosen plans fell back at
    runtime;
  * accuracy is free — the chosen plan's relative error against the
    exact fold is 0 at 19/20 points and 2.8e-16 at the twentieth;
  * **every non-unit regret is one defect: k-overshoot.** At all 8
    admitted points the planner took k=256 while k=16 was exact and
    5-16x faster. Cause: with a 16,384-row sketch (stride 28.1) rank
    16 is unresolvable, and rank 64's quantile comes from population
    rank ~42, making the tail bound up to 129 ORDERS OF MAGNITUDE too
    pessimistic (5.5e4 predicted vs 1.3e-67 achieved at eps=1e-3).
    Only k=256 underflows past the tolerance. The included
    `sketch_resolution_sweep` shows the fix: at 65,536 rows (stride
    7.0, 0.1 s to collect) k=16 becomes both resolvable and
    admissible. It also shows that at the shipped DEFAULT
    `stats_sample = 1024` NOTHING is ever admitted — the access path
    is unreachable on a table this size.
  * two smaller defects named with numbers: the cost model charges a
    FILTERED probe only one extra scalar pass while the measured swing
    is 1.41 -> 17.45 ms (it ignores the `ef*8` beam-widening budget),
    and planning (2.62 / 2.79 / 5.75 ms min/median/max, dominated by
    scoring the sketch) costs ~12x the 0.24 ms execution of the plan
    it is choosing between.
- `hnsw_neighbor_selection` / `hnsw_neighbor_selection_real` — the
  Malkov-Yashunin heuristic, measured on both workload classes, with
  OPPOSITE answers, both reported:
  * 100k uniform-random unit vectors d=64 (the case
    m6_hnsw/README.md called "the v2 lever"): recall@10 0.5040 ->
    0.5290 at ef=64, 0.6840 -> 0.7010 at 128, 0.8600 -> 0.8640 at 256,
    and 0.9670 -> **0.9650** at ef=512 — nothing, for a 27% slower
    build. Algorithm 4 is a near-duplicate remedy and uniform-random
    d=64 vectors have no near-duplicates.
  * REAL 459,865 x 384 MiniLM embeddings (the corpus this access path
    actually indexes): recall@10 0.8930 -> 0.9390 at ef=64, 0.9110 ->
    0.9510 at 128, 0.9320 -> **0.9700** at 256, 0.9620 -> 0.9740 at
    512. Per-query latency is 29% HIGHER at fixed ef, so the honest
    reading is at ISO-RECALL: for recall@10 >= 0.96, KeepBest needs
    ef=512 at 1.438 ms/query while Diversity needs only ef=256 at
    0.955 ms/query — better recall at **1.5x lower latency**. Build
    281.1 s -> 392.5 s (+40%).
  * NOT flipped to default in this leg: it costs 40% build time, and
    the regret grid says M4's bottleneck is sketch resolution, not
    recall (the chosen plans already achieve 0 relative error).
- Harnesses: `experiments/m6_hnsw/regret_grid` and
  `experiments/m6_hnsw/diversity_bench` (standalone crates, outside
  the workspace on purpose; both merge additively into results.json
  and never clobber other keys).

- cargo fmt clean; clippy 0 warnings on bruce-core + bruce-query (all
  targets); bruce-core 90 -> 95 tests; bruce-query topk_access_path
  14 -> 16.

## 2026-08-03 (orchestrator: Python package surface parity)

- **bruce-py/python/bruce/__init__.py** — `KvSnapshot` was registered by
  PyO3 but never re-exported, so `bruce.KvSnapshot` did not exist while
  `bruce._bruce.KvSnapshot` did. Added to the import block and `__all__`.
- **bruce-py/python/bruce/_bruce.pyi** — added stubs for `KvSnapshot`,
  `KvMemory.{bulk_insert,snapshot,restore}`, `grouped_softavg` (the f32
  twin was typed, the f64 one was not), and `QuerySession` — the entry
  point to the whole query layer had no type stubs at all.
- **bruce-py/tests/test_api_surface.py (NEW)** — pins the three surfaces
  against each other: everything PyO3 registers must be in `__all__`,
  must be importable, and must be declared in the `.pyi`; and the `.pyi`
  must not declare anything the extension does not provide. This class of
  gap was invisible to every other test because they import what they
  need directly. Red on the pre-fix wheel (3 failures = the 3 gaps
  above), green after.


## 2026-08-03 (night six, pg-parity track: NaN / ±Inf semantics in bruce-pg, 0.1.1 → 0.1.2)

Closes the last residual gap of the campaign — "bruce-pg ScalarAcc does
not yet mirror bruce-core's new NaN/±Inf policy". bruce-pg is a
standalone crate; nothing in the workspace is touched.

**The decision (made and written into README before any code).** The
mirror is not symmetric, and the asymmetry is the point:

- **±Inf: mirrored verbatim.** In bruce-core the ±Inf policy lives
  *inside* `RowAcc::absorb`/`merge`, i.e. inside the monoid. C2 says
  the monoid must be the same monoid on both sides, so `ScalarAcc`
  now carries the same branches: `+Inf` at finite `eps` collapses to
  argmax with uniform ties (and the collapse survives `merge`, so
  parallel plans agree with serial ones), `-Inf` weighs 0, an
  all-`-Inf` group is SQL NULL, `eps = 'infinity'` stays score-blind.
  `exp(inf - inf)` is never evaluated.
- **NaN: deliberately diverges — bruce-pg PROPAGATES, bruce-core
  SKIPS.** In bruce-core NaN *is* the engine's encoding of SQL NULL
  (`bruce-query/src/ingest.rs` maps Parquet NULL → NaN; an
  `ndarray<f64>` has no null bitmap), which is why skipping is right
  *there*. PostgreSQL already has NULL, and `'NaN'::float8` is a real
  value that `AVG`/`SUM` propagate. Importing another engine's NULL
  encoding would make this extension silently delete a value the
  database beside it faithfully carries — and disagree with the `AVG`
  in the same `SELECT` list.

  C2 survives intact, and bruce-core's own source is the authority:
  `RowAcc::absorb`'s doc comment says *"NaN scores/values never reach
  `absorb` … the SQL-NULL skip happens at the call sites."* The skip
  was never part of the monoid, so PG's call site (the transition
  function) is free to make the PG-native choice. Formally the PG
  state is now the **product monoid**
  `(mu, z, u) × ({false, true}, ∨)`: `ScalarAcc` is left exactly as
  bruce-core wrote it, and a sticky NaN bit rides alongside it.
  `test_j_nan_state_component_matches_bruce_core_skip` makes that
  observable in-database — `softavg_state` slots 1..3 are
  **bit-identical** with and without the NaN row present; only slot 5
  differs. Supporting arguments and the losing option's cost are in
  bruce-pg/README.md, "Special float values: NaN, ±Inf, and one
  deliberate divergence"; the decisive one is recoverability: a user
  can write `WHERE v <> 'NaN'::float8` to *get* bruce-core's skip
  semantics, but nobody can recover a value a skipping aggregate has
  already thrown away.

**It replaced an undefined behaviour, not a working one.** 0.1.1 had no
±Inf branches and no NaN policy at all. Replaying its recurrences
verbatim (reproducer kept at `bruce-pg/results/old_scalaracc_0_1_1.rs`,
`rustc -O`; all 13 cases in
`bruce-pg/results/pg_parity_semantics.json`):

| rows | eps | 0.1.1 | 0.1.2 |
|---|---|---|---|
| `+Inf` beside finite rows | 0.37, 1 | NaN | 55.0 |
| two `+Inf` rows (tie) | 0.37, 1 | NaN | 55.0 |
| all rows `-Inf` | 0.37, 1 | NaN | NULL |
| all rows `-Inf` | 0 | 943.5 | NULL |
| NaN score beside finite rows | 0 | 20.0 (silently dropped) | NaN |
| NaN score beside finite rows | 0.37 | NaN | NaN |
| NaN value beside finite rows | 0 | 20.0 (silently dropped) | NaN |
| NaN value beside finite rows | ∞ | NaN | NaN |

The eps = 0 rows are the tell: a NaN row *vanished* there (both
`s > mu` and `s == mu` are false for NaN) while the identical row
poisoned the accumulator at finite `eps`. Old behaviour was neither
skip nor propagate — it was whatever the float comparisons produced.

**Changes**

- **bruce-pg/src/lib.rs** — `ScalarAcc::absorb`/`merge` gain
  bruce-core's `±Inf` branches verbatim. New pgrx-free `PgAcc` (the
  product monoid) holds the transition/merge/finalize logic; the three
  `#[pg_extern]`s are thin encode/decode wrappers over it. `decode`
  split into a pure `decode_checked` + the `error!`-raising wrapper.
  Two behaviour fixes fall out of the policy: `FINALFUNC` now returns
  SQL NULL for an empty monoid factor (reachable for the first time —
  an all-`-Inf` group — where it previously would have raised
  "empty state should be NULL"), and NaN over an empty factor returns
  NaN rather than NULL.
- **State layout** `float8[4] = [mu, z, u, eps]` →
  `float8[5] = [mu, z, u, eps, nan_seen]`. `STYPE` is `float8[]` either
  way so no catalog change is needed, but `softavg_state` output is
  observably one element longer.
- **Version 0.1.1 → 0.1.2** + `sql/bruce_pg--0.1.1--0.1.2.sql`.
  Required: observable aggregate behaviour changed (table above).
  Unlike the inert 0.1.0→0.1.1 script this one carries the DBA-facing
  explanation and refreshed `COMMENT`s, including the one operational
  consequence — `REFRESH` any materialized view aggregating data with
  NaN or `±Inf` scores. `test_i_upgrade_path_alter_extension` retargeted
  at a synthetic `0.1.2--0.1.3`.
- **Tests: 21 → 39**, green on **pg16 (16.14) AND pg17 (17.10)**
  (30 `#[pg_test]` + 9 pure-Rust cross-check; suite wall time
  median-of-5 pg17 4.03 s / pg16 2.97 s, loadavg 9.61 7.12 5.26 —
  machine shared with the other night-six tracks; this track makes no
  performance claim). 12 new `test_j_*` in-database tests cover
  NaN value / NaN score / all-NaN / `+Inf` argmax + ties / `-Inf`
  weight-0 / all-`-Inf` NULL / `eps=∞` score-blindness / mixed
  NaN-vs-`±Inf` / NaN-row-still-honours-eps-domain, **including the
  parallel path both ways** (test_d's forced-`Gather` recipe): one NaN
  among 200k rows still poisons the answer through `COMBINEFUNC`, and
  four `+Inf` rows spread across the scan give the same uniform mean
  serially and in parallel — plus an explicit SQL `combine` of two
  partial states that each hold a `+Inf` row (the `mu = +Inf` on *both*
  sides branch). 6 new cross-checks add `±Inf` fixtures under every
  partition and pin the divergence itself
  (`nan_input_diverges_from_bruce_core_by_design`).
- **Cross-check re-pinned per option (b)**: `bruce_core_cross_check`
  feeds bruce-core NaN-free inputs; the divergence gets its own named
  test asserting *both* halves — monoid factor equal to bruce-core's
  skip answer, PG finalize returning NaN.
- **Harness note discovered the hard way** (now in README): `cargo pgrx
  test` relies on the linker's `--gc-sections` pass to prune
  pgrx-pg-sys's unreachable `extern "C"` declarations out of the test
  binary. A plain `#[test]` that calls a `#[pg_extern]` (or anything
  reaching pgrx's `error!`) makes them live and the binary fails to
  link with `undefined symbol: errstart`. That constraint is why the
  logic was extracted into `PgAcc`/`decode_checked` — better design
  reached by way of a linker error.
- **PG reference behaviour confirmed in-database** (not from memory):
  `AVG` over a column containing NaN is NaN; `NaN = NaN` is true so
  `<> 'NaN'::float8` really does filter NaN; core PG has no `isnan()`.
- clippy `--tests --features pg_test`: 0 warnings. `cargo fmt --check`:
  clean (the crate was not previously rustfmt-formatted; now is).
- Docs updated in this task: bruce-pg/README.md (new "Special float
  values" section with the decision, the rejected option's cost, the
  measured 0.1.1 table, and every new test), docs/TESTING_MATRIX.md
  (residual gap closed), bruce-pg/results/pg_parity_semantics.json.

## 2026-08-03 (night six, kv-snapshot track: KvMemory bulk/snapshot API)

Closes the residual gap "KvMemory: no bulk/Arrow snapshot API" found by
the m17 sidecar track. Raw contiguous buffers in bruce-core (C1); the
Arrow wrapping stays at the py/server layer (C3).

- **bruce-core/src/memory.rs** — new `KvSnapshot` struct (row-major
  `Vec<f64>` key/value buffers + ids/owners/written_at, live rows only,
  insertion order) and three `KvMemory` APIs: `bulk_insert` (contiguous
  slices; audit/owner/last-write-wins parity with a `write` loop, one
  shared timestamp per batch, all-or-nothing on shape or ownership
  errors), `snapshot()` (skips tombstones), `restore()` (bitwise
  round-trip: same decode bits out of `attention_query` /
  `snapshot_alive` / `read_exact`; owners + written_at preserved so
  owner-enforced delete keeps working; audit log deliberately empty —
  audit history travels via parquet, not hot-path snapshots; typed
  errors for malformed buffers, never panic). 10 new unit tests
  (bruce-core 80 -> 90; the first attempt of this track landed most of
  this, audited + kept).
- **bruce-py/src/lib.rs** — `KvMemory.bulk_insert(ids, keys, values,
  owner)` (2-D numpy in; explicit shape check so a mis-shaped buffer
  with matching total length cannot be silently re-chunked),
  `KvMemory.snapshot()` -> new `KvSnapshot` pyclass (ids/owners/
  written_at/keys/values accessors, one copy out per accessor —
  documented), `KvMemory.restore(snap)` staticmethod,
  `KvSnapshot.from_arrays` (reassemble from buffers persisted at this
  layer, e.g. Arrow). NOT yet re-exported from `bruce/__init__.py`
  (file not owned by this track tonight; reachable via
  `KvMemory.snapshot()` / `bruce._bruce.KvSnapshot`) — follow-up line
  for whoever owns the py package surface next.
- **DEFECT found + fixed (tests first): Fortran-order numpy input to
  the new `bulk_insert` binding was silently ingested column-major**
  (rows scrambled) because rust-numpy `as_slice()` returns the raw
  buffer of ANY contiguous array, F-contiguous included. Red test
  `test_bulk_insert_fortran_order_input` (m17 suite) first, then the
  fix: guard the zero-copy path on ndarray standard row-major layout,
  fall back to a logical-order flattening copy otherwise.
- **experiments/m17_kv_sidecar** (paper workspace) — sidecar's numpy
  mirror DROPPED: ingest is one `bulk_insert` per document, decode
  serves from cached engine snapshot buffers, `rebuild_from_store`
  stays per-row `read_exact` as the snapshot-free ground truth. All 8
  original proof obligations green unchanged (bitwise rebuild residual
  0.0); +7 snapshot-specific tests (round-trip bitwise incl. decode
  bits, tombstone exclusion, restore-then-delete with owner
  enforcement, bulk-vs-write-loop bitwise, F-order defect pin, shape
  guard, from_arrays round-trip). results.json extended additively
  with a `snapshot_api` block (snapshot/restore medians of 7 at ~99k
  live rows x d64, throughput, bitwise residual, loadavg at bench).
- clippy 0 warnings on owned files (also removed a stale `PyDict`
  import in bruce-py/src/lib.rs that predated this track).

## 2026-08-03 (night six, lifecycle-safety track — workstreams 16+20)
- Workstream 16 (session lifecycle) landed:
  bruce-py/tests/test_lifecycle.py (6 tests, suite now 285/285 on
  the 04:46 wheel). Pins at the Python level: register over an
  existing name replaces the table and cascade-DROPS its maintained
  views (content change visible, view absent from candidates, name
  freed for re-create); stale key references after replace are typed
  ValueError for both read (`column emb must be a key column`) and
  write (`insert names unknown key column`) with the session
  surviving; duplicate CREATE VIEW name is a typed error; views on
  other tables survive unrelated re-registers and stay
  incrementally correct under inserts. Long-lived-session soak:
  10,000 queries + 100 full lifecycle cycles in one session, RSS
  sampled per cycle — 0 kB post-warmup growth (bound 32 MiB,
  justified against leak signatures: one leaked 1 MiB table
  generation/cycle => >=90 MiB, 4 KiB/query => 36 MiB; measured
  clean noise 256 kB/290 cycles). Samples in
  docs/qa/lifecycle_rss.json.
- abi3 import matrix (scripts/abi3_matrix.py): the one cp39-abi3 .so
  loaded from the same wheel file into every interpreter present on
  this box — CPython 3.10.12 (/usr/bin) and 3.13.13 (miniforge base,
  numpy 2.5.0): import + one SOFTAVG query, oracle-checked, both
  PASS; conda env `pgv` recorded as python-free (PG runtime only).
  Absent interpreters not faked. docs/qa/abi3_matrix.json.
- Workstream 20 (memory-safety audit) landed:
  docs/UNSAFE_INVENTORY.md — rg sweep over all 7 crates at campaign
  start AND end: ZERO hand-written unsafe blocks anywhere (the one
  'unsafe' token in mask.rs:543 is the f32 track's comment recording
  its AVX2 kernel was measured below the 5% gate and reverted).
  Macro boundaries documented (pyo3 0.22/abi3 exercised by 285
  pytest + matrix; pgrx =0.18.1 by 21/21 on pg16+pg17), forbid/deny
  (unsafe_code) recommendations recorded for the lib.rs owners.
  ASAN deferred: no nightly on the box (stable 1.95 only, pgrx pin),
  exact future command in the inventory. Leak sampling
  (scripts/leak_soak.py; valgrind not installed — /proc VmRSS,
  5-cycle block medians, loadavg ~10 recorded): 200 whole-session
  build/drop cycles -28 kB post-warmup growth; 200 in-session
  replace/cascade/view-rebuild cycles 0 kB; median cycle 0.5 ms.
  Index build/drop not in the soak (no Python index API yet; hnsw
  track owns that surface). docs/qa/leak_soak.json.
- stats.rs registration-totality guard (residual gap from last
  night): found already landed by the pre-restart attempt — audited,
  kept, verified. Out-of-range DictU32 codes are attributed to no
  group (counts under-cover n_rows, conservative for selectivity;
  PG ANALYZE never fails a sampled table); NaN sims are dropped in
  KeySketch::sims (was partial_cmp().unwrap() panic); all-NaN
  sketches return resolution_limited so the planner falls back to
  the exact plan. Pinned by bruce-query/tests/register_safety.rs
  (5 tests, green; clippy clean).
- TESTING_MATRIX: 16 ticked; 20 ticked with parenthetical (ASAN
  deferred: needs nightly); residual-gap entries for stats.rs and
  16/20 updated in place.

## 2026-08-03 (night six, f32-tail track — mixed-precision tail, backlog #1)
- f32 kernel SIMD experiment — HONEST NEGATIVE, reverted per the >=5%
  keep gate: explicit AVX2+FMA dot (4x 256-bit fmadd accumulators,
  is_x86_feature_detected! dispatch) implemented, tested green, and
  benchmarked: grouped_softavg/f32_1M_d384 median 25.03 ms (median of
  5 criterion runs) vs 25.60 ms saved baseline = -2.2%, under the 5%
  gate. Root cause: that config is DRAM-bandwidth-bound (~60 GB/s
  aggregate over 32 rayon threads); the dot itself IS 1.87x faster
  cache-resident (single-thread microbench, 73.7 vs 39.6 GB/s) and
  showed end-to-end at 100k_d64 (-11.5%), but the gated shape cannot
  benefit. A safe 8-way unroll measured SLOWER than the existing
  4-way (defeats the SSE2 autovectorizer). All numbers under
  "avx2_experiment_2026-08-03" in m2_mixed_precision/results_m2.json;
  decision + pointers recorded on dot_f32's doc comment. No unsafe
  carried; mask.rs kernel code is byte-identical to last night's.
- masked_attention NaN pinning (residual gap): AUDITED the pre-restart
  attempt's landing, kept and verified — NaN-scored pairs (NaN
  anywhere in q_i/k_j) and NaN value rows are SKIPPED in every eps
  regime, matching the grouped kernels' SQL-NULL discipline; rows
  whose every pair is skipped are uncovered (covered=false, zero row).
  +/-Inf scores stay real values (argmax collapse / weight 0). Pinned
  by bruce-core/tests/numerical_edges.rs mod masked_attention_nan_policy
  (4 tests, incl. exact agreement with grouped_softavg on NaN-laced
  data). POLICY FOR THE PG MIRROR (C2, pg-parity runs next): skip on
  NaN score OR any-NaN value row, every regime; uncovered group = SQL
  NULL; infinities per RowAcc::absorb. TESTING_MATRIX residual gap
  marked CLOSED.
- SoftAggView over KeyF32 (views.rs): maintained views now serve f32
  key columns — f32 storage/scoring (wire f64 rows cast down before
  the dot, bit-identical to a rebuild over the stored column; query
  cached as f32), unchanged f64 (m,num,den) state, same group-inverse
  delete + bounded re-anchor. key_col_of stays KeyF64-only (db.rs
  create_index and delete_where survivor capture pin on its refusal).
  Former typed refusal FLIPPED in bruce-query/tests/error_totality.rs
  (+ new create_view_f32_semantics: insert delta works end-to-end;
  delete_where on an f32-viewed table pinned as the remaining typed
  error — db.rs is another track's file tonight) and in bruce-py
  tests/test_error_paths.py. New views.rs mod tests (3): random
  insert/delete stream vs from-scratch rebuild at every prefix
  (rel <= 1e-4, 3 seeds x 80 ops), view-vs-grouped_softavg_f32 kernel
  agreement, clean group emptying. New bruce-py/tests/test_f32_views.py
  (4): view-served answers == viewless f32 scan (rel 1e-4),
  maintained under inserts, delete gap typed + state intact, wrong
  param not served. Suites: bruce-core 125 green, bruce-query 78
  green, pytest 289 green on a fresh wheel; clippy clean on owned
  files.

## 2026-08-03 night six (Track cdc-v2: UPDATE support + durable mirror, BACKLOG #12 v2)

bruce-cdc 28 -> 53 tests green; clippy --all-targets 0 warnings; fmt
clean. PG 54329 left healthy (wal_level=logical, 0 slots, bruce_pub
only, cdc_movies 1060 + movies 459865).

- **Datum split (pgoutput.rs)** — `TupleDatum::{Null, Unchanged,
  Text}` wired through the decoder (the enum existed from the aborted
  first attempt; `tuple()` and the message variants still collapsed
  'u'/'n' to `None` — now 3-way distinct). Relation carries the
  replica-identity byte. Conformance: corpus regenerated (8 .bin, +
  update_variants.bin; unchanged_toast.bin now drives 't'/'u'/'n' in
  one Update new-tuple), and a LIVE round trip
  (tests/update_toast.rs) with a 10KB STORAGE EXTERNAL column:
  untouched -> `Unchanged`, SET NULL -> `Null`, byte-exact resolution
  from the mirror.
- **Update apply (apply.rs)** — delete(old pk) + insert(resolved new)
  through the standard write path (views absorb via the (m,num,den)
  group inverse). Identity FULL 'O', DEFAULT key-only 'K' and
  DEFAULT no-old-tuple all handled; `Unchanged` resolves from the
  mirror's current row BEFORE the delete; NULL in a mapped column is
  a typed error with the mirror untouched. REPLICA IDENTITY NOTHING
  -> typed error naming ALTER TABLE ... REPLICA IDENTITY (and pinned
  live: PG itself refuses such UPDATEs, SQLSTATE 55000). New
  `TxAssembler` (source.rs): the pgoutput -> CommittedTx state
  machine, connection-free so conformance tests drive it.
- **Differential (tests/update_diff.rs)** — 250 random update txs
  (10% pk moves) on an identity-DEFAULT table: mirror table ==
  from-scratch re-snapshot BIT FOR BIT; incrementally maintained view
  == recomputed view within 1e-12 (observed ~1e-16).
- **Durable mirror (durable.rs, NEW)** — `Mirror::save/load`: one
  BRCDCM01 file (table + counters + last_lsn watermark), FNV-1a
  trailer, atomic tmp+fsync+rename+dir-fsync; corrupt/truncated/
  wrong-magic files are typed errors. Views not serialized (recreate
  after load). main.rs `--state PATH`: save-after-apply-before-ack,
  slot resume FROM DISK instead of the v0 "drop the slot" error.
- **Kill -9 chaos (tests/durable_chaos.rs)** — the subscriber is a
  REAL child process (test binary re-execed), SIGKILLed 5x at
  deterministic rows-applied thresholds read from the durable file,
  restarted FROM DISK each time, under a 400-tx workload with >=40%
  updates: final on-disk state == PG over every mapped column, bit
  for bit. Save-then-ack makes all three kill windows safe (before
  save / after save / mid-save).
- **Update-heavy kill-resume (tests/chaos.rs)** — in-process chaos
  re-run with >=45% updates incl. pk moves, 5 kills, full 6-column
  checksum exact.
- **Measured (m12_cdc/results.json `durable_v2`, release, medians of
  5, loadavg ~6)** — recovery (open durable state + reconnect + catch
  up 1000 pending txs): median 7.7 ms; snapshot save of a 100k-row
  mirror (6.0 MB): median 27.3 ms = 210 MB/s. m12_cdc README gained
  the v2 section + merge-not-overwrite note for results.json.

## 2026-08-03 (night-five consolidation)

- **bruce-server/src/flight.rs** — `res.map(|s| {...; s})` ->
  `res.inspect(...)` (clippy manual_inspect, the single workspace
  warning the testing campaign's verifier found); workspace clippy is
  back to 0 warnings, bruce-server 12/12 green.
- **docs/TESTING_MATRIX.md** — added the residual-gaps section
  (exec.rs TopK panic, stats.rs corrupt-dict panic, masked_attention
  NaN, ScalarAcc C2 parity, KvMemory snapshot API, HNSW diversity v2,
  workstreams 16/20 open).
- Campaign totals on the final tree, independently verified: workspace
  cargo test 178/0 (+3 ignored soaks re-run green), pytest 279/0 on a
  fresh wheel, bruce-cdc 28/0, bruce-pg 21/21 on pg16 AND pg17,
  regress.sh OVERALL PASS (golden 14.0 ms in [10,20] band).

## 2026-08-03 (Track 3: write path + error totality, workstreams 9, 10)

Tests first: two new bruce-query suites + one bruce-py suite; every
production edit below exists only because one of these tests exposed a
defect or forced a semantics decision (each carries a code comment
naming the suite).

- **bruce-query/tests/stateful_writes.rs (NEW, ws 9)** — seeded
  stateful property test: random 500-row table (dict + 2 scalars +
  KeyF64), 1-3 maintained views at distinct eps, 300 random ops from
  {insert_row, delete_where(Eq on id), delete_where(GtEq)}; after
  EVERY op, each view and Database::run (once per view eps via
  MaintainedViewScan, once at a fresh eps via FusedGroupScan) must
  equal an independent from-scratch softmax recomputation (1e-9
  rel/abs hybrid). Failures print seed + op index. 5 seeds default,
  50 under `--ignored` (both green).
- **bruce-query/tests/error_totality.rs (NEW, ws 10)** — 15 tests,
  ~70 adversarial cases over parse_query / optimize / plan-via-run /
  execute / Database::{register,run,insert_row,delete_where,
  create_view} / Table::{from_parquet,attach_key_f64,attach_key_f32},
  each behind a catch_unwind wrapper: Err always, panic never. (No
  `physical::lower` exists; physical enumeration lives in
  planner::plan and is driven through Database::run.)
- **bruce-py/tests/test_error_paths.py (NEW)** — the same philosophy
  through QuerySession (18 tests): bad input => ValueError (TypeError
  only at PyO3 argument extraction), never PanicException/abort; plus
  a lifecycle smoke: 200 queries + 50 writes in one session, peak-RSS
  growth < 50 MB (measured ~0).

Defects the tests exposed, fixed (db.rs / views.rs only):
- **delete_where double subtraction** (stateful_writes, seed=11 op=3):
  when one call deleted a group's anchor scorer plus another row of
  the same group, rows processed after the bounded re-anchor were
  subtracted from state that already excluded them. Fix: per-group
  settlement — after a re-anchor (detected via n_reanchors) the
  remaining doomed rows of that group are skipped.
- **SoftAggView::build panicked** on a query vector whose dimension
  mismatches the key column (ndarray dot assert) -> typed Bind error.
- **Database::run panicked** on (a) bound-param dim mismatch with a
  declared budget (the stats sketch dots sample rows; stats.rs is
  another track's file, so the guard sits in db.rs::validate_run) and
  (b) dict codes beyond the dictionary (ExactGroupAvg indexing; same
  guard). views.rs on_delete also grows its group table instead of
  indexing out of bounds when fed a corrupt code.
- **Stats collection underflowed on 0-row tables** with a key column
  (stats.rs `r.min(n - 1)` with n = 0). Guarded at db.rs's two call
  sites (collect_stats): empty tables get default stats; queries over
  them return the empty covered set. stats.rs itself left to its
  owner.

Previously-undefined semantics, now pinned (PG vocabulary, C4):
- `Database::register` over an existing name = CREATE OR REPLACE and
  DROPS maintained views on the old table (stale view state must
  never answer for the new contents). bruce-cdc audited: it registers
  before creating views — unaffected (28 tests green).
- `create_view`: names are unique (duplicate -> Err); eps must satisfy
  Eps::new AND be > 0 (the eps=0 tropical endpoint has no incremental
  (m,num,den) form and previously built NaN state silently); KeyF64
  only (f32 keys -> Err, pinned as current behavior).
- `insert_row` rejects rows naming unknown or ill-kinded columns
  (PG INSERT: unknown target column is an error).
- eps=INF queries need no key column or bound param (R3 endpoint,
  pinned in run_eps_inf_needs_no_key_or_param).

Suites: bruce-query 55 green (incl. the 50-seed soak run once),
workspace `--exclude bruce-server` 160/0 (bruce-server was mid-edit
by its own track at run time: declared flight_server bin not yet on
disk), bruce-cdc 28/0, pytest 279/0, wheel rebuilt via `make python`.
clippy: zero warnings in bruce-query/bruce-py (two warnings in
another track's new bruce-core/hnsw.rs left to its owner).

## 2026-08-03 (Track 2: query-layer fuzz + differential, workstreams 5-8)

Tests-first hardening of the SQL front half. No production defect
found (the frontend is total on 1000 fuzz cases; the planner is
equivalence-clean on 200); two previously-undefined semantics pinned
and documented; production edits are doc comments only.

- **bruce-query/tests/frontend_fuzz.rs (NEW, ws 5)** — deterministic
  seeded generator (xorshift64*, zero new deps): 500 structurally
  random VALID queries over `{g dict, v,y scalar, k key}` (random
  keyword case, whitespace incl. \t \n \r, eps literals 0 / tiny /
  huge / E-notation / INF / INFINITY, all four sim names, optional
  budget, optional predicate) must parse, match the generated
  structure exactly (group/val/key/param/kind/eps to the bit), and
  re-lower through optimize() without panic; 500 MUTATED strings
  (required-token deletion, keyword swap/split, junk injection incl.
  non-alphabetic unicode + NUL + RTL-override, unterminated
  literals/comments/idents, balanced+unbalanced paren towers to 4000
  deep, truncation) must return Err and NEVER panic — every call
  wrapped in catch_unwind. Plus error-determinism (same input -> same
  message, 50 cases). PINNED: eps literal beyond f64 range (`1e999`)
  saturates to +inf and degenerates via R3 to the uniform mean — not
  a parse error (doc comment added at parse.rs::eps_of).
- **bruce-query/tests/plan_equivalence.rs (NEW, ws 6)** — 200 seeded
  random queries over random in-memory catalogs (200-2000 rows, 1-16
  groups, f32 OR f64 keys, quantised filter column so `=` predicates
  hit): naive lowering (parsed plan mapped 1:1 to FusedGroupScan, eps
  inf on the kernel's uniform path) vs the full Database::run
  pipeline — labels identical, values to 1e-10 rel. R3 shape asserted
  each case (inf -> ExactGroupAvg, finite -> FusedGroupScan). Plus
  empty-selection boundary (3 cases) and an inf-endpoint test against
  a manual filtered-mean oracle. No divergence found.
- **bruce-py/tests/test_differential.py (NEW, ws 7)** — the oracle
  harness: 100 seeded pandas datasets+queries (both key dtypes forced
  by seed parity) through bruce.QuerySession vs (a) numpy
  anchored-softmax and (b) DuckDB executing the max-anchored SQL
  (`WITH s AS (...), m AS (MAX per group) SUM(EXP((sim-mx)/eps)*v)/
  SUM(EXP((sim-mx)/eps))`); agreement |a-b| <= rtol*max(1,|a|,|b|)
  at rtol 1e-9 (f64 keys) / 1e-4 (f32 keys; f32 eps pool >= 0.1 so
  the oracle's own f32 rounding stays inside the contract). Edge
  suite: single row, single group, empty-after-filter (both engines
  return zero rows), filter-empties-one-group (covered-set
  agreement), eps=0 == DuckDB `AVG(v) FILTER (WHERE sim = mx)` with
  constructed exact ties, eps=INF == plain `AVG(v) GROUP BY`.
- **bruce-query/tests/explain_golden.rs (NEW, ws 8)** — exact golden
  strings for FusedGroupScan explain (with + without fused filter),
  the R3 ExactGroupAvg endpoint, TopKContractScan (fixed est fields
  so the {:.3e} delta is deterministic), MaintainedViewScan; shape
  contract for PlannedQuery::explain() (stable headers, exactly one
  `-> chosen`, est/ms/MB fields per candidate line, floats NOT
  pinned; no TopK candidate without a budget, TopK enumerated with
  one). PINNED R1-NEGATIVE (documented in optimizer.rs): a filter on
  the score's key column is not pushed and the planner refuses the
  plan ("plans must end in an aggregate"); via SQL the same predicate
  is rejected at execution with a typed Bind error ("filter column
  ... must be ScalarF64") — key/dict columns are not legal filter
  inputs, never a panic or silent wrong answer.
- Totals this track: +16 Rust tests (2 x 500 fuzz cases + 200
  equivalence cases inside), +106 pytest items. cargo test -p
  bruce-query all green; workspace (minus bruce-server, mid-landing
  by another track) all green; full pytest 279 green; my files
  clippy-clean (the 3 db.rs warnings belong to the write-path track).

## 2026-08-03 (Track 4: CDC conformance corpus + chaos, workstreams 11+12)

Tests-first hardening of the maintenance plane; two defects exposed
and fixed, three previously-undefined semantics pinned.

- **bruce-cdc/tests/conformance.rs + tests/corpus/ (NEW, ws 11)** —
  golden pgoutput v1 byte corpus, 7 `.bin` files (framing `u32 BE
  length || CopyData payload`; deterministic generator
  `tests/corpus/gen_corpus.py` + inventory README): multi-row tx with
  every header field asserted exactly, NULL columns + key-only ('K')
  delete, text edges (quotes / newlines / 2-3-byte UTF-8 / 4-byte
  emoji / empty string), unchanged-TOAST 'u' marker, Relation re-sent
  mid-stream with an ADDED column, Origin/Type/Truncate as
  `Other(tag)` (ignored, never an error), keepalive with
  reply-requested set and clear. 9 tests incl. a catch-all
  every-file-decodes pass and the apply-plane pin that surplus
  columns are ignored until re-snapshot while a missing mapped column
  errs loudly.
- **bruce-cdc/tests/chaos.rs (NEW, ws 12)** — against the live PG
  54329, serialized on a process lock: (a) kill-resume x5 — writer
  thread issues 300 interleaved single-row INSERT/DELETE txs while
  the subscriber is killed 5x deliberately in the apply-then-ack
  window; final mirror == PG ground truth exactly (row count + FNV-1a
  checksum of sorted (id, rating-bits) pairs), each kill's replayed
  tx observed and filtered; (b) `pg_ctl -m fast restart` mid-stream —
  subscriber reconnects through the new bounded-retry path and
  converges; (c) one 5000-insert transaction: nothing delivered
  before COMMIT, arrives as ONE CommittedTx, applied 0 -> 5000 in one
  `apply_tx`. Plus 3 pure-logic pins (watermark insert/delete replay,
  transient-error classification).
- **DEFECT FIXED (exposed by chaos kill-resume): exactly-once
  watermark** — `Mirror.last_lsn` (apply.rs): a kill between apply
  and ack redelivers already-applied txs from the slot; previously an
  insert replay silently duplicated rows and a delete replay errored
  "out of sync". `apply_tx` now filters `end_lsn <= last_lsn` as a
  counted-nowhere no-op; consistency contract documented on `Mirror`.
- **DEFECT EXPOSED (chaos empty-snapshot mirror): bruce-query stats
  `n - 1` underflow on 0-row tables** — `Mirror::from_snapshot` of an
  empty relation (legitimate CDC start state) panicked in
  `KeySketch::collect`. Track 3 landed the fix concurrently (db.rs
  collect_stats guard: empty tables get default stats); this track's
  interim stats.rs guard was reverted in favor of it — stats.rs stays
  with its owner. Pinned here by chaos.rs
  replayed_tx_is_filtered_by_watermark (empty snapshot) and the
  large-tx test (empty table subscribe).
- **DEFECT FIXED (exposed by the full-suite chaos run): reconnect
  deadlock against PG's fast shutdown** — protocol.rs `PgConn::
  shutdown` + source.rs `redial`: PG's fast shutdown waits for THIS
  walsender's client while refusing new connections (57P03); the old
  reconnect kept the dead socket open until a new dial succeeded, so
  subscriber and server waited on each other and `pg_ctl -w` timed
  out (observed wedged full-suite run; standalone run passed on
  timing luck). `redial` now tears the old socket down at the OS
  level before dialing.
- **Production additions justified by chaos (b)** — source.rs:
  `PgOutputSource::redial` (fresh walsender, pending + relation cache
  cleared, resume from slot) and `RetryingSource` (bounded-retry
  `ChangeSource` wrapper; transient = Io / Protocol / SQLSTATE 57P01,
  57P03, 55006 via new pub `is_transient`; semantic errors propagate
  immediately; lost acks are safe by watermark).
- **Semantics DEFINED + documented**: (1) commit-buffered delivery /
  no-partial-reads / no crash atomicity contract on `Mirror`; (2)
  mid-stream Relation with added column -> unmapped columns ignored
  until re-snapshot, dropped mapped column fails loudly; (3)
  unchanged-TOAST 'u' decodes as None, NULL-indistinguishable — safe
  only while Update is rejected (pgoutput.rs).
- Measured (debug build, same-host PG over the unix socket; two
  runs): kill-resume reconnect (redial + START_REPLICATION on the
  live slot) ~1 ms per kill, 5/5 kills; `pg_ctl -m fast restart` ->
  first re-applied tx 0.88-0.99 s end-to-end (server bounce + one
  500 ms-backoff reconnect + post-restart writes); 5000-row tx
  delivered as one CommittedTx, applied in 38-76 ms. Chaos test
  learning encoded in the test itself: PG's fast shutdown waits for
  the logical walsender to flush to a READING client, so the restart
  runs in a thread while the subscriber keeps polling — and pg_ctl
  gets explicit `-l`/`-o` so the postmaster never inherits the test
  harness's stdio pipes. bruce-cdc README: run section + consistency
  contract updated.

## 2026-08-03 (Track 1: kernel math — workstreams 1, 2, 3, 4)

Tests first; NaN/±Inf score semantics were previously UNDEFINED in the
grouped kernels (NaN poisoned the finite-eps accumulator; ±Inf scores
produced NaN via `exp(inf - inf)`) — now pinned, PG-aligned per C4.

- **bruce-core/tests/prop_monoid.rs** (NEW, workstream 1) — proptest
  (4×192 cases) partition/order invariance of `grouped_softavg` /
  `grouped_softavg_f32`: random record multisets split into 1..8
  shards with random per-shard orders fold to the sequential result,
  single- and multi-group, eps in {0, 0.37, 1, inf}; plus a
  deterministic merge test pinning 1..8 rayon chunks == 1 thread
  (37k rows, forces `RowAcc::merge`) and an empty-shard-is-identity
  test (a fully-deselected chunk merges as a no-op). Tolerances
  derived and documented in the module docs (~3n·2⁻⁵³; ceilings:
  f64 1e-12 rel, f32-vs-its-own-single-thread 1e-5 rel). Idempotence
  deliberately NOT tested: the monoid is on multisets under bag-union
  (PG combinefunc, C2), `merge(a,a)` doubles `(z, u)` by design.
- **bruce-core/tests/numerical_edges.rs** (NEW, workstreams 2+4) — 15
  tests pinning: NaN score/value rows SKIPPED in every eps regime
  (SQL NULL discipline — ingest.rs encodes NULL as NaN; PG
  two-argument aggregates skip the row if either argument is NULL);
  +Inf score at finite eps = argmax semantics with uniform ties (also
  across parallel-chunk merges); -Inf = weight 0, all-`-inf` group
  uncovered (Indicator "no match" == empty equi-join -> NULL,
  matching crud.rs); eps = inf stays score-blind; eps = 1e-300 ==
  tropical answer, eps = 1e300 bit-equal to the eps = inf mean; empty
  selection/input uncovered; single row bit-exact; all-equal scores
  bit-equal to plain mean at ANY finite eps; subnormal-weight tails
  == max-anchored truth, no NaN. Workstream 4's precision contract:
  f32-vs-f64 max rel error on 10k rows / 16 groups per eps —
  measured {1e-4: 0.0, 1e-2: 3.2e-7, 0.1: 4.5e-8, 1.0: 1.6e-10}
  against ceilings {1e-3, 1e-4, 1e-5, 1e-5}.
- **bruce-core/tests/drift_soak.rs** (NEW, workstream 3) — 50k
  random insert/delete cycles on `IncrementalMemory` (deletes target
  only live keys), checkpoint every N/10 against BOTH a from-scratch
  rebuild and a `grouped_softavg` single-group mirror; drift bound
  1e-9 rel. Measured: max drift 2.6e-15 @ 50k (127 re-anchor
  rescales), 4.2e-15 @ 1M cycles (905 rescales; `#[ignore]`-gated,
  run once via `-- --include-ignored`). The crud.rs re-anchor path
  holds; no defect found.
- **bruce-core/src/mask.rs** (production, minimal, each change
  demanded by a red test above): (1) NaN skip at the two grouped
  fold call sites (SQL NULL discipline, comment cites ingest.rs +
  the PG corr/covar rule); (2) `RowAcc::absorb` — `-inf` skipped at
  eps = 0 and finite eps, `+inf` argmax-collapse branch at finite
  eps; (3) `RowAcc::merge` — finite-eps `+inf`-anchor branch so
  `exp(inf - inf)` is never evaluated; (4) policy doc comments on
  `absorb` and `grouped_softavg` (notes the deliberate divergence
  from `semiring::softmax_eps`'s all-`-inf` uniform fallback and the
  consistency with `IncrementalMemory`'s tropical path).
- `masked_attention` inherits the ±Inf policy through `RowAcc`; its
  NaN behaviour is NOT pinned (no call-site skip there — out of this
  track's scope).
- Suite: bruce-core 80 unit + 23 new integration tests, all green;
  clippy zero warnings. Workspace `--no-fail-fast`: only failures
  are bruce-query `error_totality` (7) + `stateful_writes` (1) —
  Track "write path & views" (workstreams 9/10) red tests in flight,
  created today by that track; verified unrelated (finite data;
  view-vs-rebuild 0.08 discrepancy flows through identical kernel
  code on both sides).

## 2026-08-03 (Track 6: ingest robustness + perf/golden harness — workstreams 15, 17, 18, 19)

Tests first; two ingest defects exposed and fixed; the perf/regression
harness is now a one-command nightly gate.

- **bruce-query/tests/ingest_robust.rs (new, 13 tests)** — dirty-file
  corpus built at test time with the same arrow/parquet crates the
  engine links: zero-row file, all-NULL string column, numeric NULLs
  (NaN), unsupported types (timestamp/boolean) skipped alongside
  supported ones, 3-row-group / multi-batch boundary consistency
  (dictionary codes stable across batches), 120k-distinct-string
  dictionary growth, file-not-found / truncated-at-5-points /
  non-parquet bytes all typed `Err`, `attach_key_{f64,f32}`
  dim-mismatch + empty-table semantics. The example's minimal .npy
  reader lives only in `examples/one_query.rs` (not library code), so
  npy edges are out of scope; the library key-ingest path is
  `attach_key_*`, covered.
- **DEFECTS FOUND AND FIXED (bruce-query/src/ingest.rs)** — duplicate
  column names in a parquet schema: same-type duplicates silently
  appended both columns into one Vec (length 2n, corrupt table);
  mixed-type duplicates hit the `unreachable!()` in the column-variant
  match (panic on user input). DEFINED SEMANTICS: duplicate field
  names are rejected up front with typed `QueryError::Bind`
  ("duplicate column name ..."), aligned with PG 42701
  `duplicate_column` (C4). With unique names each name maps to one
  Arrow dtype per file, making the remaining `unreachable!` arms
  sound. PINNED (previously undefined, now documented in tests):
  zero-row parquet loads as an empty column map (columns materialise
  per batch; schema not preserved), and `attach_key_*` on a table with
  no columns accepts any row count — the first key column then defines
  the table's row count.
- **bruce-core/benches/fold.rs (new criterion suite, workstream 17)**
  — `grouped_softavg/{f64,f32}_{100k,1M}_d{64,384}` (d_v = 1, 32
  groups, eps = 0.1 — the SQL SOFTAVG shape), `masked_attention/
  window_100k_pairs_d64`, `kv_memory/{insert_1k,delete_1k}`; registered
  as `[[bench]] fold` in bruce-core/Cargo.toml. Baseline medians
  (32 threads, this box): f64 1M×d384 48.4 ms, f32 1M×d384 25.6 ms
  (the ~1.9x bandwidth story), f64 1M×d64 9.3 ms, f32 1M×d64 5.0 ms,
  masked 100k pairs 1.08 ms, insert_1k 169 µs, delete_1k 61 µs.
- **scripts/bench_compare.py (new)** — snapshots criterion's own
  `estimates.json` medians (+ machine metadata) to
  `paper_sigmod_bruce/experiments/perf_baselines/fold_baseline.json`
  (`--save`), and diffs a fresh run against it, exit 1 on any median
  >15% slower. Anti-noise protocol documented in the script and the
  baseline: medians only, idle-box assumption, no taskset (the
  kernels are rayon-wide by design), deterministic inputs,
  cross-machine compare refused without `--force`.
- **scripts/soak.py (new, workstream 18)** — 10M rows × d=64 f32
  (chunked generation, f32 straight from the RNG), eps ∈ {0, 0.1, ∞}:
  all finite + all covered; peak RSS 4.85 GiB < 6 GiB; 1M→10M latency
  factor 9.25x within [8, 13]; group-cardinality extremes at 1M rows
  (1 group and 1,000,000 groups) numpy-spot-checked at eps 0.1/∞
  (worst rel err 2.6e-6; eps=0 exercised for finiteness/coverage only
  — near-tie argmax under f32 scoring is the documented precision
  contract), coverage flags == group occupancy exactly (632,246
  occupied), t(1M groups)/t(100k groups) = 6.8 (no quadratic blowup).
  Python chosen over an #[ignore]d Rust test because the oracle is
  numpy, RSS comes from getrusage, and it soaks the shipped wheel.
- **scripts/regress.sh (new, workstream 19)** — the nightly gate:
  workspace cargo test + pytest (bruce-py/tests) + the one_query
  example on the real 460k-row movie workload with a golden band
  assertion on the printed query median (10–20 ms f64; frozen number
  13.5 ms). Opt-in expensive stages: `REGRESS_BENCH=1` (criterion +
  baseline compare), `REGRESS_SOAK=1` (scale soak). run_m2 is not in
  the gate: it has no quick mode (thread-sweep subprocesses, minutes)
  and stage 3 already pins the same engine path and golden number.
  PASS/FAIL summary per stage, exit 0 iff all executed stages pass.

## 2026-08-03 (Track 5: PG version matrix + semantics conformance — workstreams 13, 14)

bruce-pg grows from 13 to **21 tests**, now green on TWO PostgreSQL
majors: `cargo pgrx test pg16` AND `pg17` both pass 21/21 (18
`#[pg_test]`s + 3 bruce-core cross-checks).

- **Version matrix (WS 13)** — PostgreSQL 16.14 built via
  `cargo pgrx init --pg16 download --pg17 <existing pg_config>
  --configure-flag=--without-icu --configure-flag=--without-readline`
  (passing BOTH versions is load-bearing: init rewrites
  `~/.pgrx/config.toml` with exactly the versions given). PG 16
  tarballs still ship pregenerated parser files, so the missing
  `bison` on this host is a non-issue (it only bites at PG >= 17,
  which is already built); `flex` is at `~/miniforge3/bin/flex`.
- **Upgrade path (WS 13)** — extension bumped 0.1.0 -> 0.1.1; new
  hand-written `bruce-pg/sql/bruce_pg--0.1.0--0.1.1.sql` (trivial
  COMMENT). Verified that `cargo pgrx install`/`test` copies
  `sql/*.sql` into `<sharedir>/extension/`, so packaged installs carry
  the upgrade chain. New `test_i_upgrade_path_alter_extension` drives
  PG's real update machinery: stages a synthetic `0.1.1--0.1.2` script
  into the sharedir, runs `ALTER EXTENSION bruce_pg UPDATE TO
  '0.1.2'`, asserts `extversion` and the script's effect
  (obj_description), then removes the staged file (catalog change
  rolls back with the test transaction).
- **Semantics conformance (WS 14)** — 7 new `#[pg_test]`s pinning
  softavg to AVG's promises (C4): NULL-skip == AVG over the
  both-non-NULL subset at eps {0, 0.5, inf}; empty/WHERE-false/
  all-NULL input -> SQL NULL (one row, never 0, never an error) side
  by side with AVG, incl. the pair rule (qualifying input is the
  complete (value, score) pair — value-with-NULL-score tables are
  "empty" to softavg while AVG(v) is not); catalog audit that
  SFUNC/COMBINEFUNC/FINALFUNC are declared non-strict to match their
  NULL-handling behavior (PG rejects a strict 3-arg SFUNC with NULL
  initcond, and only a non-strict SFUNC can raise on NULL eps); NULL
  eps on a live row -> clean ERROR; unicode GROUP BY parity ('猫'
  rows collapse, NFC 'café' vs NFD stay distinct — PG's equality
  unchanged through softavg, per-group == AVG at equal scores);
  negative eps -> clean SQL ERROR; eps = 0 two-way max-score tie ->
  mean of exactly the tied rows with `softavg_state` showing
  [mu, z, u] = [max, 2, tied-sum].
- Production-code deltas: version bump + upgrade SQL only — no defect
  in the aggregate itself; one test-authoring iteration sharpened the
  documented semantics (AVG's qualifying input is a column, softavg's
  is a pair). clippy still zero warnings.

## 2026-08-03 (Track B / M14: PostgreSQL extension via pgrx, bruce-pg)

Constitution C2's shortest proof, shipped: the `(mu, z, u)`
soft-average monoid IS PostgreSQL's aggregate contract. New standalone
crate `bruce-pg/` (empty `[workspace]` table — NOT a member of this
workspace; root Cargo.toml untouched), pgrx 0.18.1 pinned (0.19.x
needs rustc >= 1.96), PostgreSQL 17.10 via
`cargo pgrx init --pg17 download`.

- **bruce-pg/src/lib.rs** — `ScalarAcc`, the scalar (d_v = 1) mirror
  of bruce-core's file-private `RowAcc` (same absorb/merge/finalize
  recurrences; bruce-core is a dev-dependency so the .so stays lean),
  wired as `bruce_softavg_sfunc` / `bruce_softavg_combine` /
  `bruce_softavg_final` into `CREATE AGGREGATE
  softavg(float8, float8, float8) ... PARALLEL SAFE`. State =
  `float8[4] = [mu, z, u, eps]` — a varlena, so parallel workers need
  no serialfunc/deserialfunc. A finalfunc-less twin `softavg_state`
  hands the raw monoid element back to SQL. All three eps regimes:
  0 = argmax-mean (uniform ties), finite = max-anchored softmax,
  'infinity' = plain mean. NULL discipline matches AVG; mid-group eps
  change is an error.
- Tests: `cargo pgrx test pg17` = **13 green** — 10 SQL `#[pg_test]`s
  (softavg == AVG at equal scores for all eps; == naive SQL softmax at
  eps 1/0.5 to 1e-9; THE KILLER: at eps = 1e-4 naive
  `SUM(EXP(s/eps)*v)/SUM(EXP(s/eps))` raises `value out of range:
  overflow` while softavg returns the exact limit; forced-parallel
  200k-row run with `Gather` + `Partial Aggregate` asserted in-plan
  and equal to serial; partition-combine identity written literally in
  SQL) + 3 cross-checks pinning `ScalarAcc` to
  `bruce_core::masked_attention` (1e-12, incl. partitioned merges).
- Measured (1M rows, pgrx-managed PG 17.10, port 28817; conda PG on
  54329 untouched): softavg serial 322.4 ms -> 4 workers 71.4 ms
  (4.51x, leader participates); parallel == serial to 1e-18; naive
  softmax 53.2 ms at eps = 0.25 but dead below eps ~ 1.41e-3, softavg
  eps-uniform (323.4 ms at eps = 1e-4). Numbers + caveats (float8[]
  detoast per row ~320 ns vs AVG ~30 ns) in
  paper_sigmod_bruce/experiments/m14_pg_extension/results.json.

## 2026-08-03 (Track C: CDC maintenance plane — bruce-cdc, backlog #12)

The sidecar form is now end-to-end real: PG stays the system of
record, bruce-cdc subscribes via logical replication and keeps the
maintained SOFTAVG state fresh through the ordinary write path.

- **bruce-cdc (new standalone crate, not a workspace member)** —
  `protocol.rs`: raw v3 protocol over the unix socket (released
  rust-postgres has no replication support): `replication=database`
  startup, replication commands, CopyBoth stream, standby status
  updates. `pgoutput.rs`: proto_version 1 decode (Begin/Commit/
  Relation/Insert/Delete; Update recognized, rejected in v0).
  `source.rs`: `ChangeSource` trait (transport seam) +
  `PgOutputSource` — real START_REPLICATION; slot created with
  USE_SNAPSHOT in a read-only repeatable-read walsender txn so the
  initial copy meets the stream exactly at the consistent point;
  `ack(end_lsn)` advances `confirmed_flush_lsn` (PG itself is the
  resume checkpoint; restart streams from `0/0`). `apply.rs`:
  `Mirror` — snapshot rows -> Database table; committed txs ->
  `insert_row`/`delete_where` (REPLICA IDENTITY FULL old tuple ->
  `Pred::Eq` on the pk; a delete that doesn't remove exactly 1 row
  fails loudly). Demo binary prints the maintained answer per commit.
- **PG instance (port 54329)** — `wal_level=logical` set and
  restarted (kept); table `cdc_movies` + publication `bruce_pub`.
- Tests: 12 pure unit (decode/apply/view-freshness) + 1 end-to-end
  against the live stream: seed 1000, then 100 INSERT txs + 50
  DELETE txs applied while committing, freshness asserted 4 ways
  (vs from-scratch bruce snapshot of final PG state, vs PG's own SQL
  soft average, view read vs PG SQL, post-resume vs PG SQL; max rel
  err 1.1e-15, plan = MaintainedViewScan), then disconnect + 10 more
  rows + reconnect proves slot-based resume. All green; clippy clean.
- Measured (release, same-host PG): commit->applied lag mean 1.90 ms,
  p50 1.89 ms, max 2.32 ms over 150 interleaved single-row txs
  (PG committed them in 282 ms). Results:
  bruce/paper_sigmod_bruce/experiments/m12_cdc/results.json.

## 2026-08-03 (Track A: mixed-precision scan — f32 storage, f64 fold)

The iso-cost anchor: the incumbents store float32, so the fused scan
now meets them at their own storage cost point.

- **bruce-core mask.rs** — `grouped_softavg_f32`: same contract as
  `grouped_softavg` but f32 keys/query. Precision contract: f32
  storage and scoring (4-way unrolled partial-sum dot — vectorizes
  AND shortens each rounding chain to d/4), score widened to f64 once
  per row, then the existing f64 `(mu, z, u)` RowAcc monoid (reused,
  not duplicated — the C2 PG-aggregate mapping is unchanged). Same
  rayon chunk-reduce structure and 2^15 threshold. 2 new tests
  (kernel-vs-kernel rel err < 1e-5 at eps 0.1/1.0 with exact covered
  agreement; chunking determinism).
- **bruce-query** — `Column::KeyF32` variant; `Table::attach_key_f32`;
  executor dispatches KeyF64 -> `grouped_softavg`, KeyF32 ->
  `grouped_softavg_f32` (bound :param cast down once, the same
  rounding the encoder applied to the keys); TopKContractScan's sims
  pass scores f32 keys in storage precision; stats collect a KeySketch
  from KeyF32 by upcasting only the bounded sample; write path
  (insert/delete) handles KeyF32 (RowValues stays f64, cast on
  append). Views remain KeyF64-only.
- **bruce-py** — `QuerySession.attach_key` dispatches on dtype
  (float64 -> KeyF64 as before, float32 -> KeyF32 with NO upcast);
  raw `bruce.grouped_softavg_f32` exposed + stub. 10 new tests in
  tests/test_grouped_f32.py (kernel parity, end-to-end f32-vs-f64
  session rtol 1e-4, eps=1e-4 stays finite and anchored, f32 write
  path). Suites: Rust 93, Python 155, all green.
- Measured (32 cores, 460k x 384 movie workload, eps=0.1, median of
  7): f32 fused query end-to-end 8.5 ms vs f64 14.0 ms (1.65x;
  kernel-only 7.7 vs 12.9 ms, 1.67x — f64 sits at the ~55 GB/s DRAM
  ceiling, the f32 dot reaches 46 GB/s, partly compute-limited at
  4-way unroll); max rel err f32 vs f64 kernel on the same stored
  numbers 9.8e-9; key bytes 357.6 MB vs 715.2 MB. Results:
  bruce/paper_sigmod_bruce/experiments/m2_mixed_precision/results_m2.json.

## 2026-08-02 (M-Q v2: the temperature-aware optimizer, bruce-query)

The cost pass promised in docs/QUERY_LAYER_DESIGN.md §4 is
implemented; the query layer now estimates, prices, enumerates,
chooses, and survives writes. New modules in `bruce-query`:

- **stats.rs** — histogram selectivity (64-bucket) + `KeySketch`:
  a deterministic strided row sample of each key column, scored
  against the bound query at plan time to estimate `k*(eps, budget)`
  (smallest k whose certified bound meets the declared error budget).
  Declares itself `resolution_limited` in the extreme-value regime
  (sharp eps) instead of guessing; validated within 3x of the oracle
  at moderate eps.
- **cost.rs** — bandwidth cost model calibrated from the M1
  measurements (55 GB/s, 13 ms / 460k x 384 f64). Ranks plans; the
  formulas encode that ExactGroupAvg reads no key bytes and that
  top-k without an index still streams every key (only value bytes +
  fold shrink).
- **optimizer.rs** — R3 endpoint degeneration: `SOFTAVG(.., INF)` ->
  `PlainGroupAvg`, the score expression (and the key column) leave
  the plan. eps remains a semantic parameter: no rule changes it.
- **planner.rs** — enumeration with verdicts (Chosen / Costlier /
  Inadmissible(reason)); the contracted top-k plan is enumerated ONLY
  when the query declares a budget (SOFTAVG's optional 4th argument)
  and the sketch certifies it. EXPLAIN prints every candidate with
  est ms / MB and each rejection's reason.
- **exec.rs** — `TopKContractScan` executes with a RUNTIME GUARD:
  per-group k_g, true omitted mass from the streamed sims, exact
  re-fold of any group whose realized bound misses the budget. No
  group is ever dropped (the incumbent top-k semantics failure).
  `ExactGroupAvg` and `MaintainedViewScan` execute natively.
- **views.rs / db.rs** — maintained soft-agg views + the write path:
  `Database::{register, create_view, run, insert_row, delete_where}`.
  Writes apply view deltas (insert O(1); anchor-scorer delete = one
  bounded re-anchor pass, counted in `n_reanchors`) and mark stats
  stale for lazy recollection before the next plan.
- Tests: 10 new/updated (R3 equivalence incl. the eps=0 tropical
  argmax endpoint; view freshness under insert + non-anchor delete +
  anchor delete; the contract admissibility FLIP across sharp /
  diffuse / super-sharp temperatures; estimator-vs-oracle; stats
  freshness after deletes). Workspace: 91 Rust tests green, clippy
  clean. `examples/explain_demo.rs` shows all six plan flips.

## 2026-08-02 (TemperaturePlan: the whole running example as one plan)

- **`bruce.frontend.TemperaturePlan`**: three temperature stages
  compiled together — (1) eps=0 exact filter, (2) eps>0 soft
  group-aggregate (SoftAggQuery), (3) attention read over the winning
  group's rows. One shared similarity pass feeds stage-2 weights,
  stage-3 read, AND the truncation contract: the omitted weight mass
  delta certifies |err| <= delta*(1+1/(1-delta))*max|v| for a top-k
  stage-3 read, so an optimizer can substitute the approximation only
  under an explicit error contract (never by silently changing
  semantics). `explain()` prints the three-stage plan.
- **`bruce.frontend.MaintainedPlan`**: maintained IncrementalMemory
  state for stage 2 (per group, d_v=1) and stage 3 (winning group,
  d_v=dim); a single `delete(key_id)` refreshes both the aggregate and
  the model-facing read (measured: 1.2 us for both, errs 1e-14/1e-16
  vs rebuild, on the 4,159-row winning group of the CIDR movie query).
- Frontend tests: 3-stage explain, contract bound holds, dual-stage
  delete matches rebuild; plus empty-group semantics regressions
  (fully-filtered group is absent, not NaN). 142 tests green.
- Makefile smoke-test tolerance fixed (truncated e constant made the
  1e-10 assert mathematically unsatisfiable; now 1e-8).

## 2026-06-12 (industrialization pass — toward publication)

### Foundation
- **Version control**: repository initialized; baseline commit captures
  the pre-industrialization state, every layer below is a separate
  reviewable commit.
- **Panic policy**: release profile switched from `panic = "abort"` to
  unwind — with abort, any Rust panic killed the host Python
  interpreter; with unwind PyO3 raises PanicException. Library paths
  audited: no panics on user input remain (attention_batch returns
  Result; DP mechanisms and k_ary_balanced_tree validate at the
  Python boundary with ValueError).
- **MSRV 1.81** declared and CI-checked.
- **Static analysis**: cargo clippy --workspace --all-targets at ZERO
  warnings (was ~50); cargo fmt enforced.

### CI / packaging
- ci.yml rewritten: full-workspace clippy -D warnings, stable+beta x
  linux+macos test matrix, MSRV job, maturin-action wheel builds with
  strict pytest (the old pipeline used venv-dependent maturin develop
  and `pytest || echo`, which silently swallowed failures), cargo-deny
  audit job + deny.toml license/advisory policy.
- release.yml: tag-driven wheels (manylinux x86_64/aarch64, macOS
  x86_64/aarch64, sdist) -> PyPI via OIDC trusted publishing.
- pyproject.toml: requires-python >=3.9 (matches abi3-py39), full
  classifiers, keywords, URLs; FIXED a latent TOML bug where a
  misplaced table would have swallowed `dependencies`.
- PEP 561: py.typed + complete _bruce.pyi stub (33/33 names) now ship
  inside the wheel; __pycache__ excluded from the wheel.

### bruce-server hardening (smoke-tested 9/9)
- WAL append failures now surface as HTTP 500 + bruce_wal_fail_total
  metric (previously silently swallowed after acking the client);
  mutex poisoning recovered instead of crashing.
- JWT cross-tenant enforcement: token sub must equal owner on
  POST /facts and DELETE /facts/:id (was documented, never enforced).
- GET /ready readiness probe; tower-http request tracing; graceful
  SIGINT/SIGTERM drain for both plaintext and TLS paths.
- Dockerfile: non-root runtime user (uid 10001) + HEALTHCHECK.

### Robustness & docs
- proptest property suite for masked_attention (order invariance +
  reference agreement, 256 cases x 2 properties, all eps regimes).
- criterion benchmarks for the mask evaluator (causal N=512: ~0.49 ms;
  window N=8192 w=64: ~14 ms on the dev box).
- README production-deployment + versioning sections; CONTRIBUTING.md
  (quality gates, engineering rules); SECURITY.md (reporting policy +
  server security model).

### Known pre-publication items
- PyPI/crates.io name availability for "bruce" unverified
  (PUBLISH-NAME-001 in bruce/TODO.txt).
- GitHub remote not yet created; CI is ready to run on first push.

## 2026-06-12 (PODS-paper theory back-ported into the kernel)

### bruce-core
- **mask.rs (NEW MODULE)**: `masked_attention(Q, K, V, pairs, eps)` —
  the PODS paper's free-connex "enumerate-then-fold" evaluator as
  code. Consumes any duplicate-free `(i, j)` mask stream in ARBITRARY
  order (causal, window, tree, join-query output) and folds per-row
  max-shifted accumulators; one code path covers eps = 0 (tropical
  argmax-mean, uniform ties), finite eps > 0 (online softmax), and
  eps = inf (plain mean). Parallel path chunks the stream and merges
  accumulators via the partition-reduce identity (Lemma B). Returns
  `(out, covered)` with per-row coverage flags. Plus `causal_pairs(n)`
  and `window_pairs(n, w)` generators. 8 new Rust tests incl.
  order-invariance under shuffle, parallel == sequential fold,
  tropical ties, and equivalence with `tree_causal_attention` on the
  chain (KERNEL-MASKSTREAM-001).
- **semiring.rs**: NEW `eps_star(delta, gap, v_max, n, kappa)` — the
  certified-smoothing temperature of the paper's smoothing corollary
  (multiplicity promise is a LOWER bound; sign as fixed in the
  internal review round 2), and `dequantization_bound(scores, v_max,
  eps)` — the quantitative Maslov bound 2*v_max*(N-k)/k*exp(-gap/eps)
  evaluated on actual scores. 4 new tests incl. "A_eps* within delta
  of the tropical answer" and "bound dominates actual error"
  (KERNEL-EPSSTAR-001).
- **types.rs**: `Eps::INF` const + `Eps::is_inf()`; `Eps::new` now
  accepts +inf (the uniform-mean sentinel the doc comment always
  promised). NaN and negatives still rejected.

### bruce-py
- NEW bindings + top-level exports: `bruce.masked_attention`,
  `bruce.causal_pairs`, `bruce.window_pairs`, `bruce.eps_star`,
  `bruce.dequantization_bound`. `pairs` is an int64 (P, 2) array.
- NEW `tests/test_mask.py` — 22 tests: bit-level vs numpy dense
  reference (atol 1e-12) across all temperature regimes, shuffle
  order-invariance, parallel-path spot checks at 33,930 pairs,
  tree-attention equivalence, certified-smoothing bound incl. the
  kappa=1-is-always-safe direction.

### Test totals after this change
- cargo: 76 passed (was 62) + 1 doctest.
- pytest: 117 passed, 1 skipped (was 95/1).


## 2026-05-26 (overnight VLDB attack)

### bruce-core
- **operator.rs**: `F_eps::scores()` now uses `K.dot(x)` (ndarray native
  matmul, contiguous + SIMD-friendly) for `Sim::Dot`. Falls back to
  rayon-parallel per-row loop for `Sim::NegSquared` and `Sim::Indicator`
  when `N >= 1024` to avoid thread-pool overhead at small N.
- **operator.rs**: NEW `F_eps::attention_batch(Q, K, V)` — process B
  queries against the same (K, V) in one call. Two matmuls (Q @ K^T,
  weights @ V) dominate; per-query latency drops as B grows.
  Measured: 1.64× speedup at B=1024 vs single-query (270 → 443 q/s).
  Gap to numpy reference (2826 q/s, BLAS dgemm) is 6.4× — closing
  this gap is TODO `WHEEL-BLAS-001`.
- **join.rs**: `hash_join()` probe phase now uses rayon
  `par_iter().flat_map_iter()` when `|L| >= 4096`. The Python binding
  `bruce.hash_join_indices` may dispatch to a different path — see
  `WHEEL-PARALLEL-002b` in `bruce/TODO.txt`.

### bruce-py
- **lib.rs**: NEW PyO3 binding for `Operator.attention_batch(Q, K, V)`
  returning `numpy.ndarray` of shape `(B, d_v)`.
- abi3-py39 manylinux_2_34 wheel rebuilt; shipped to
  `/project/jding2/hkenv` on iTiger (md5 verified).

### Measured impact (iTiger H100 / hkenv)
- Single attention thread scaling: s_serial 0.94 → 0.57; speedup at 32
  cores 1.07× → 1.42× (still memory-bound per query; batch is the
  better win).
- attention_batch B=1024 on iTiger: 1.64× over single-query; output
  matches numpy reference to 2.71e-13 (machine ε).
- HW v3 GPU: tree-attention now routes through `bruce.torch.tree_attention`
  on CUDA, giving 4.89× speedup over CPU at N=100K (4.81ms vs 23.5ms).

## 2026-05-26 (afternoon) — observability, typed client, GPU batch

### bruce-server
- **main.rs**: NEW `Metrics` struct (atomic counters) + `/metrics`
  endpoint exposing 12 Prometheus counters/gauges:
  `bruce_requests_total`, `bruce_writes_total`,
  `bruce_writes_fail_total`, `bruce_reads_total`,
  `bruce_reads_404_total`, `bruce_deletes_total`,
  `bruce_deletes_fail_total`, `bruce_queries_total`,
  `bruce_alive_facts`, `bruce_total_facts`,
  `bruce_audit_length`, `bruce_uptime_seconds`. Counters are
  bumped from each handler without holding the main RwLock.
  Verified locally: 3 ops → `bruce_requests_total{} = 3`,
  `bruce_writes_total{} = 1`, etc.

### bruce-py
- **python/bruce/client.py**: NEW `BruceClient(base_url)` —
  typed sync client. Methods: `write`, `read`, `delete`,
  `attention`, `info`, `health`, `audit_root`, `audit_length`,
  `metrics` (parses Prometheus text into a dict). Replaces raw
  `urllib.request` calls; exposes `BruceClient`,
  `BruceClientError`, `ServerInfo` from `bruce.client`.
- **python/bruce/torch.py**: NEW `attention_batch(Q, K, V, eps, sim)`
  — batched GPU attention via two cuBLAS matmuls (Q@K^T,
  softmax_ε @ V). Closes WHEEL-GPU-002 for the batched
  Operator.attention path; H100 ms/q expected to drop from CPU
  rayon's 1.8 ms/q at B=1024 to ~0.05 ms/q.

## 2026-05-26 (continued) — bruce-server WAL + auto-replay

### bruce-server
- **main.rs**: NEW `--wal-path <PATH>` CLI flag. Writes (`/facts`
  POST + DELETE) append JSONL records to the WAL; on startup, if
  the WAL file exists, each entry is replayed before serving
  traffic.
- **main.rs**: NEW `WalRecord` enum (`Write`, `Delete`) with serde
  serialize / deserialize.
- **main.rs**: `Inner` now holds `Option<Mutex<File>>` for the
  WAL handle; absent if flag is empty.
- Recovery semantics: after SIGKILL with 5000 writes pending,
  restart with same `--wal-path` reloads all state. Verified
  200/200 keys recovered bit-level on iTiger (run 76835).

## Pending (see `bruce/TODO.txt` for IDs)

- WHEEL-BLAS-001: link ndarray against system BLAS (openblas / MKL) to
  close the 6× gap to numpy.
- WHEEL-PARALLEL-002b: investigate `bruce.hash_join_indices` binding;
  if it bypasses `bruce-core::join::hash_join`, retarget.
- WHEEL-GPU-002: native CUDA Operator.attention via torch C++ extension
  or cuBLAS.
- WHEEL-PERSIST-001: Parquet-backed `KvMemory` variant.
- WHEEL-FAILOVER-001: auto-replay audit log on `bruce-server` startup
  (currently the audit log is durable but state must be replayed by
  client).
- WHEEL-OBSERVE-001: Prometheus `/metrics` endpoint for bruce-server.
- WHEEL-CLIENT-001: typed Python client for bruce-server (currently
  using urllib.request).
- WHEEL-SECURITY-001: TLS + JWT auth for bruce-server.
- WHEEL-INDEX-001: HNSW-style precompute for ε > 0 (sketch is ε → 0
  only today).
- WHEEL-API-001: surface `top_k`, `fuzzy_join`, and the partition-
  reduce reducer as first-class Python APIs.

## Unreleased (2026-08-01)
- Add `bruce.frontend`: minimal eps-algebra query frontend
  (`SoftAggQuery`: parse -> logical/physical plan EXPLAIN -> execution via
  `masked_attention`). Used by the CIDR one-query-three-engines experiment.
  9 new tests in bruce-py/tests/test_frontend.py.

## Unreleased (2026-08-02)
- Add `grouped_softavg` (bruce-core mask.rs + pyo3 + `bruce.grouped_softavg`):
  fused physical operator for grouped soft-averages — dictionary-encoded
  group ids (no pair materialisation) and an optional selection mask
  fused into the scan. Rayon chunk-reduce above 2^15 rows.
  13 new tests; frontend gains `execute_grouped` and its EXPLAIN names
  the new kernel. Measured 13.0 ms vs 71 ms pair path on the 460k-row
  movie workload (see bruce/paper_sigmod_bruce/experiments/m1_grouped_kernel/).

## Unreleased (2026-08-02, night two)
- New crate `bruce-query`: the eps-algebra query layer — sqlparser-based
  SQL frontend (SOFTAVG(val, SIM(key,:param), eps) as standard function
  syntax), logical IR with per-operator eps, rule optimizer (R1
  predicate pushdown with legality check), physical lowering to the
  fused `grouped_softavg` kernel, executor over an in-memory columnar
  catalog (DictU32 / ScalarF64 / KeyF64). EXPLAIN renders the fused
  plan. 4 end-to-end tests incl. rule-equivalence across eps regimes.
  Design: docs/QUERY_LAYER_DESIGN.md. RDB positioning decided:
  architecture reference DuckDB, baselines DuckDB + PostgreSQL,
  the algebra stays ours.

## Unreleased (2026-08-03, night three — interfaces)
- bruce-query: Parquet/Arrow ingestion (`Table::from_parquet`, strings
  dictionary-encoded at load; `attach_key_f64` for external encoders);
  `examples/one_query.rs` runs the real 460k-row movie query pure-Rust
  end to end (load ~0.7 s once, query 13.6 ms via the Database facade).
- bruce-py: new `QuerySession` class — the client plane. register_parquet,
  attach_key, create_view, run(sql, params) -> (labels, values, EXPLAIN),
  insert_row, delete_where (view-maintaining). 3 new tests; suites now
  Rust 91 + Python 145, all green.
- Integration strategy fixed in paper_sigmod_bruce/ROADMAP.md (M-I):
  host ladder L1-L5, three planes (Arrow/Parquet data, QuerySession
  client, CDC maintenance next).

## Unreleased (2026-08-03, track S — m17 KV sidecar)
- New experiment bruce/experiments/m17_kv_sidecar/ (backlog #17, ROUTES
  verdict: sidecar mirror, NO vLLM fork): toy serving loop where the
  attention state lives in bruce (`KvMemory` store + `masked_attention`
  decode reads + `cascade_delete` document deletes). Wheel used as-is,
  no engine changes. Honest simulation: no real LLM, deterministic
  pseudo-embeddings; the attention arithmetic and the store are real
  bruce kernels. 8 pytest proofs, all green: delete -> next decode ==
  from-scratch store rebuild bit-exactly (residual 0.0, verified at
  100k tokens); forgetting is targeted (about-query delta 1.11 vs
  unrelated 6e-16); ingest 383k tok/s, 1k-token doc delete 0.46 ms
  (~607x faster than full CPU re-ingest rebuild — lower bound vs real
  re-prefill); wrong-owner delete refused with receipt, state bitwise
  unchanged. Wheel finding documented in README: `cascade_delete`
  skips-and-receipts owner mismatches (only per-row `KvMemory.delete`
  raises); the sidecar turns receipt shortfall into ValueError.
  Bench numbers in results.json; hook points for real vLLM integration
  listed honestly in README.md.

## 2026-08-03 (Track H: HNSW access path — backlog #6)

New `bruce-core/src/hnsw.rs` (+ `pub mod hnsw;` in lib.rs): our own
multi-layer navigable-small-world index over f32 keys, dot-product
similarity (normalized embeddings: dot == cosine; MIPS caveat for
unnormalized futures documented). Built in-crate, not bound, because
the two thesis-critical hooks live INSIDE the neighbor-expansion
loop: (1) ACORN-flavored predicate-aware traversal — non-matching
nodes route but never enter results, with adaptive beam widening
(budget ef*8 pops, never enforced below k accepted) so selective
predicates keep recall; (2) delete-bitmap cooperation — tombstones
excluded from results immediately, routable-until-compact,
`tombstone_fraction()` as the rebuild signal (no graph repair in v1).
Role: access path for sharp-eps (top-k-shaped) F_eps reads; the fused
scan stays the dense path; the optimizer chooses (HNSWProbe per
docs/QUERY_LAYER_DESIGN.md lands with M-storage).

- Deterministic by construction: no RNG — levels via splitmix64(id)
  mapped to a geometric law (continuation 1/e); all heap/prune ties
  break on node index; identical insert sequences => bit-identical
  graphs AND search results (verified debug vs release too).
- API is Result-typed where dims can mismatch: `insert`, `search`
  (BruceError::DimensionMismatch), duplicate id / missing delete are
  clean errors. C1 intact: pure algorithm, no IO, no new deps.
- New `bruce-core/tests/hnsw_search.rs` — 7 tests (6 + 1 ignored
  scale). Measured on 5000 uniform-random unit vectors d=64, M=16,
  ef_c=128, ef=64: recall@10 0.9120; filtered (20% selectivity)
  0.9960; extreme filter (1%) returns full k=10 on all 100 queries;
  after 20% deletes survivors-recall 0.9280 with zero tombstone
  leaks; determinism bit-exact across two builds; edge + incremental
  (mid-build top-1 exact vs brute force at every checkpoint).
- Ignored scale test (release): 100k vectors d=64 build 17-18 s
  (~5.6-5.9k inserts/s, ~64 MiB), search k=10 ef=64 at 0.098
  ms/query (bound: 2 ms). Recall/ef curve on the uniform-random
  worst case: 0.45 @ ef=64, 0.55 @ 128, 0.83 @ 256, 0.95 @ 512
  (0.66 ms/query) — the v1 keep-M-best selection prices recall in
  ef at scale; the Malkov-Yashunin diversity heuristic is the v2
  lever. Numbers + reproduce line in
  bruce/paper_sigmod_bruce/experiments/m6_hnsw/README.md.
- Suites: hnsw_search 6/6 green (debug and release, identical
  printed recalls); full `cargo test -p bruce-core` green (80 lib +
  31 integration incl. concurrent tracks' suites); clippy zero
  warnings; MSRV respected (no is_none_or/is_multiple_of, both
  post-1.81).

## Unreleased (2026-08-03, Track F — Arrow Flight surface)
- bruce-server: Arrow Flight service (backlog #16; ROUTES verdict:
  arrow-flight crate, auth later, FlightSQL dialect later). New lib
  target with `flight` module: `do_get` executes SQL-in-ticket
  (UTF-8 JSON {"sql", "params"}) against one `bruce_query::Database`
  and streams a single RecordBatch (label Utf8, value Float64) with
  the planner's EXPLAIN text in `app_metadata`; `list_flights` /
  `get_flight_info` name registered tables; all other endpoints are
  clean `Status::unimplemented`. Parse/bind errors map to
  `invalid_argument`, exec errors to `internal` (C4). New bin
  `bruce-flight-server` (`--flight-addr 127.0.0.1:0` prints
  `FLIGHT_PORT <port>` on stdout, logs to stderr; `--parquet
  name=path`, `--key table:col=npy_path`; graceful shutdown on
  ctrl-c/SIGTERM). New `npy` module: minimal NPY v1/v2/v3 reader
  (`<f4`/`<f8`, C-order, 2-D) so key attach stays out of bruce-core
  (C1) and bruce-query keeps Arrow/Parquet as the only interchange
  formats (C3); f32 files attach as KeyF32 without upcast (M2
  precision contract). Deps added to bruce-server only:
  arrow-flight 53 (locks 53.4.1 next to workspace arrow 53),
  tonic 0.12, futures 0.3, tokio-stream 0.1, bytes 1.
- Gotcha fixed and documented: tonic applies `tcp_nodelay` only on
  the `serve(addr)` path, not `serve_with_incoming` (needed for
  ephemeral ports) — without `set_nodelay(true)` on accepted
  streams, Nagle + delayed-ACK stalls every do_get ~40 ms on
  loopback (measured 52 ms -> 11.6 ms after the fix).
- Tests: bruce-server/tests/flight_roundtrip.rs (5) — in-process
  tonic serve on an ephemeral port; do_get answer bit-equal to
  direct `Database::run` labels/values and EXPLAIN; malformed /
  non-UTF-8 / wrong-shape / unknown-table / unbound-param tickets
  all `InvalidArgument` with the server surviving; list_flights +
  get_flight_info (NotFound/InvalidArgument paths); 7 unimplemented
  endpoints checked; 8 concurrent do_gets serialize cleanly. Plus 7
  npy unit tests (dtypes, fortran-order/1-D/truncation/bad-magic
  rejects). `cargo test -p bruce-server` 12/12 green, clippy clean.
- Measured (bruce/paper_sigmod_bruce/experiments/m16_flight/,
  client_demo.py + results.json): real movie query over
  pyarrow.flight on loopback, median of 7 after 2 warmups —
  11.57 ms round-trip vs 9.87 ms in-process f32 same run: network
  tax 1.70 ms (17%); vs M2 anchors 8.51/14.04 ms. Answers
  bit-identical to in-process over 28 groups; SIGTERM exit 0.
  Demo isolates bruce-wheel and pyarrow imports in separate
  subprocesses (coexistence inflates pyarrow reads ~0.7 -> ~5 ms;
  client-side artifact, documented in the README).

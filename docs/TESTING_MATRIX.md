# TESTING MATRIX — the 20 fine-grained test workstreams (2026-08-03)
Mature engines are mostly tests (SQLite ~1000:1 test:code). This file
tracks the 20 workstreams; [ ] open, [x] landed v1.

Kernel math
 [x] 1 monoid property tests (proptest: assoc/comm/identity, random
      shard partitions == sequential; f64 and f32 ulp bounds)
 [x] 2 numerical edge suite (NaN/Inf scores, subnormals, eps extremes,
      empty groups, all-equal scores; DEFINED semantics, PG-aligned)
 [x] 3 delete-drift soak (1M insert/delete cycles, drift bound vs
      rebuild, re-anchor policy trigger)
 [x] 4 precision contract tests (per-eps f32-vs-f64 error ceilings)
Query layer
 [x] 5 SQL frontend fuzz (no panics on malformed input, stable errors)
 [x] 6 planner equivalence fuzz (random queries: optimized == naive)
 [x] 7 differential testing vs DuckDB/numpy oracle (random queries)
 [x] 8 EXPLAIN golden tests + design-doc invariants encoded
Write path & views
 [x] 9 stateful property tests (random op sequences: view == rebuild
      at every step)
 [x]10 error-path totality (typed errors everywhere, zero panics
      across the public surface)
CDC
 [x]11 pgoutput conformance corpus (multi-msg txs, mid-stream ALTER,
      NULL/unicode/quoting, keepalives, replica-identity variants)
 [x]12 chaos resume (random kills of subscriber/PG; exactly-once
      asserted vs ground truth)
PG extension
 [x]13 PG version matrix (pg16+pg17 full suite; upgrade path)
 [x]14 PG semantics conformance (NULL discipline == AVG, empty input
      NULL, text grouping == PG equality)
Ingest & interfaces
 [x]15 Parquet/npy robustness corpus (dirty real-world files)
 [x]16 session lifecycle (memory stability, re-register semantics)
Performance & scale
 [x]17 criterion bench suite + saved baselines + regression compare
 [x]18 scale soak (10M+ rows, group-cardinality extremes, bounded mem)
 [x]19 golden end-to-end regression (one-query numbers frozen, nightly
      diff)
Safety & portability
 [x]20 memory-safety audit (unsafe inventory, ASAN, leak sampling) +
      abi3 import matrix (ASAN deferred: needs nightly; valgrind
      absent, /proc RSS medians used — docs/UNSAFE_INVENTORY.md)

## Gap ledger after the 2026-08-03 campaign
## Entries marked [CLOSED]/FIXED are CLOSED — they are kept as the
## audit trail of what was found and where its pin lives, NOT as open
## work. The only items still OPEN are the ones marked "NEW GAP",
## the annotated HNSW-recall follow-up (100k scale soak stays
## #[ignore], release-only) and the ws16/ws20 sub-items at the end.
## Verified 2026-08-04 by the independent night-six audit: every
## [CLOSED]/FIXED entry's named pin exists and passes.
- [CLOSED 2026-08-03 hnsw track] exec.rs: a HAND-BUILT
  TopKContractScan with a dim-mismatched bound param panicked in the
  sims loop. `exec.rs::check_param_dim` makes it a typed Bind error on
  both KeyF64 and KeyF32 storage; pinned by bruce-query
  tests/topk_access_path.rs (`hand_built_topk_contract_scan_dim_
  mismatch_is_typed_error` + the `_f32` twin).
- FIXED 2026-08-03 (tests/register_safety.rs, 5 tests):
  stats.rs: registering a table whose DictU32 codes already violate
  [0, dict.len()) panics during stats collection (guards cover the
  post-register corruption paths only).
- [CLOSED 2026-08-03 f32-tail track] bruce-core masked_attention NaN
  poisoning: NaN-scored pairs and NaN value rows are now SKIPPED in
  every eps regime (same SQL-NULL discipline as the grouped kernels);
  pinned by tests/numerical_edges.rs mod masked_attention_nan_policy
  (4 tests incl. exact agreement with grouped_softavg on NaN-laced
  data). The bruce-pg mirror landed the same night (pg-parity track
  below): +-Inf verbatim, NaN intentionally PG-native instead.
- [CLOSED 2026-08-03 hnsw-finish track, same night it was opened by
  f32-tail] delete_where on a table with an f32-KEYED VIEW took a
  typed error. db.rs's per-view survivor capture now reads keys via
  the dtype-polymorphic `key_rows_f64` (KeyF64 borrows, KeyF32 widens
  once; the f32 -> f64 -> f32 wire round trip is bit-exact), so the
  view write path is dtype-complete end to end. `create_index` keeps
  its KeyF64-only v1 restriction by an EXPLICIT dtype check rather
  than by the old accessor's narrowness. Both pins flipped positive:
  bruce-query tests/error_totality.rs `create_view_f32_semantics`
  (maintained state == from-scratch rebuild, bit for bit, across an
  anchor delete and a non-anchor delete) and bruce-py
  tests/test_f32_views.py `test_delete_where_maintains_f32_view`.
- [CLOSED 2026-08-03 pg-parity track] bruce-pg ScalarAcc did not
  mirror bruce-core's NaN/+-Inf policy. +-Inf is now mirrored VERBATIM
  (it lives inside bruce-core's monoid): before this, a +Inf row beside
  a finite one, two +Inf rows, or two -Inf rows all evaluated
  exp(+-inf - +-inf) and returned NaN; an all--Inf group at eps=0
  returned the mean of the -Inf-scored values instead of SQL NULL.
  NaN DELIBERATELY DIVERGES: bruce-pg PROPAGATES (AVG/SUM family)
  where bruce-core SKIPS, because there NaN is the engine's encoding
  of SQL NULL while PostgreSQL has a real NULL and 'NaN'::float8 is a
  real value. C2 is intact — bruce-core's own RowAcc doc says the NaN
  skip happens at the CALL SITES, not in the monoid, so the PG state
  is the product monoid (mu,z,u) x ({false,true}, OR) with ScalarAcc
  behaviourally untouched (asserted by
  test_j_nan_state_component_matches_bruce_core_skip: state slots 1..3
  are bit-identical with and without the NaN row; only slot 5 differs).
  Decision, the rejected option, and its cost: bruce-pg/README.md
  "Special float values". Measured before/after table:
  bruce-pg/results/pg_parity_semantics.json. Version 0.1.1 -> 0.1.2
  with sql/bruce_pg--0.1.1--0.1.2.sql (observable behaviour change;
  state grew float8[4] -> float8[5]). Suite 21 -> 39 tests, green on
  pg16 AND pg17.
- [CLOSED 2026-08-03 kv-snapshot track] KvMemory bulk/snapshot API
  landed (bulk_insert/snapshot/restore, bitwise round-trip; m17
  sidecar's numpy mirror dropped; see CHANGELOG night six).
- HNSW recall, MEASURED 2026-08-03 (hnsw-finish track), claim
  partly refuted: uniform-random 100k still needs ef=512 for recall
  ~.96, and Malkov-Yashunin diversity selection (now implemented as
  `NeighborSelection::Diversity`, DEFAULT OFF) does NOT help there
  (.967 -> .965 at ef=512, 27% slower build) — it is a near-duplicate
  remedy and uniform-random d=64 has no near-duplicates. On the REAL
  460k MiniLM corpus it DOES help: recall .932 -> .970 at ef=256, and
  at iso-recall >= .96 it is 1.438 -> 0.955 ms/query (1.5x). Not
  flipped to default: +40% build time, and the regret grid says M4's
  bottleneck is sketch resolution, not recall. Numbers:
  paper_sigmod_bruce/experiments/m6_hnsw/results.json keys
  hnsw_neighbor_selection{,_real}. 100k scale soak is still
  #[ignore], release-only.
- NEW GAP, quantified 2026-08-03 (hnsw-finish track), for M4:
  `cost::predict_hnsw_tail` cannot resolve the bottom of the planner's
  k ladder at realistic sketch sizes, so the planner overshoots to
  k=256 where k=16 is exact and 5-16x faster (max regret 16.80x over
  20 real-workload grid points; median 1.00x). At the shipped DEFAULT
  `Database::stats_sample = 1024` the HNSW path is never admitted at
  all on a 460k-row table. Two smaller ones in the same key: the cost
  model ignores the filtered probe's `ef*8` beam-widening budget
  (1.41 -> 17.45 ms measured swing), and planning costs up to 12.8x
  the execution of the FASTEST alternative it is choosing between
  (max plan_time/oracle_ms = 12.82; against the plan actually chosen
  the ratio never exceeds 3.93x — both computed from results.json
  key planner_v1_regret, grid[].plan_time_ms). Full diagnosis + the
  sketch-size sweep that fixes it: m6_hnsw/README.md and results.json key
  planner_v1_regret.
- Workstreams 16 (session lifecycle) and 20 (memory-safety audit)
  LANDED 2026-08-03 (lifecycle-safety track); sub-items still open:
  ASAN (needs nightly rustc) and index build/drop in the leak soak
  (no Python index API yet).

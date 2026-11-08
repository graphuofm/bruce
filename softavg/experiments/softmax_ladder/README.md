# experiments/softmax_ladder — what the certificate buys: the softmax ladder, rung by rung

**Timings are smoke numbers on a shared machine** (`results.json` -> `timing_status`).
Every number in the generated block comes from `results.json` via `summarize.py`;
nothing in that block is typed by hand.

**PostgreSQL is an engine run on PostgreSQL 18.6** (2026-11-07; `results.json` ->
`postgres_engine`): the conda env `ggdb-pg`'s release server, used read-only, with
bruce_pg 0.2.0 and the `pg_ablation/` build staged in `../../pg18env/ext`
(`../../pg18env/README.md`). Before that server existed (the release PostgreSQL 17.10
was deleted on 2026-11-06) the PG cells were a **replay of PostgreSQL's call order
outside the server** over the same Rust arithmetic; those rows are kept, labelled
"PostgreSQL (replay)", and table 3b / 5b compare the engine with them cell by cell.
The SQLite rows are engine runs. DuckDB is cited.

## Question

`softavg` is the softmax branch: `sum e^{s/eps} v / sum e^{s/eps}`. Which softmax
implementations could one build in the same engine, and what does each get wrong?

| rung | what | where it lives |
|---|---|---|
| S0 | naive softmax in stock SQL, `exp(s/eps)` | SQL |
| S1 | safe softmax, max-shift in stock SQL. In a window one pass can only shift by the table-wide max (`S1`); the exact spelling (frame max, then a correlated subquery over the frame) is `S1x`, O(rows x k) | SQL |
| S2 | online softmax (mergeable max-anchored state (mu, z, u), Milakov / FlashAttention), **no removal**: a window recomputes every frame, maintenance recomputes a group on every delete | PG: `softavg_fwd` (bruce_pg's own SFUNC without a moving form); SQLite: `softavg_s2` (ablation build, see below); DuckDB: the segment tree |
| S3 | online softmax **with plain removal and no certificate**: remove subtracts `w = exp((s - mu)/eps)` and `w v` at the current anchor, never re-anchors, never refuses | ablation builds only: `ladder_core::PlainInv` |
| S4 | ours, the shipped `softavg`: `InvAcc` removal + running first-order error certificate; refold when the certificate fails | engines/{postgres,sqlite} |

S3 is `InvAcc` with every certificate line deleted: absorb and merge are
`ScalarAcc`'s (bit-identical, it calls them), remove is `InvAcc::remove`'s
subtraction. The unit test `same_bits_as_invacc_until_the_certificate_fires`
checks that S3's `(mu, z, u)` equals `InvAcc`'s bit for bit until the certificate
first fires; after that point S4 refolds and S3 keeps going. So the only
difference between S3 and S4 is the missing certificate and refold.

## Setup

- **Rows.** IMDb, all 459,865 rows, `ORDER BY id`; the PROTOCOL predicate subset
  `filter_num >= 2000` (232,828 rows), `ORDER BY id`; Amazon, the 1,524,113
  reviews with verified review timestamps (`../timeorder/amazon_ts.parquet`),
  `ORDER BY ts, id`. Scores: precomputed float64 columns (float32 inner product of
  the corpus embedding with the first 3 PROTOCOL queries, numpy) so that the error
  measured is the aggregation's. `rn` = position in the window order (INTEGER
  PRIMARY KEY in SQLite).
- **Windows.** `ROWS BETWEEN k PRECEDING AND CURRENT ROW`, k in {10, 100, 1000},
  eps in {1e-4, 1e-3, 0.02, 0.1, 1}, 3 queries per corpus. Reference: numpy
  float64 two-pass per frame. Wrong row: abs err > 1e-9 x max|v| of the frame, or
  NULL / Inf where the reference is finite (SQLite has no NaN; NaN comes back as
  NULL). Latency: query 0, `SELECT count(r), total(r) FROM (window query)`,
  median of 3 after the correctness fetch.
- **SQLite S2 had to be built in the extension.** SQLite 3.37.2 will not run an
  aggregate without `xInverse` as a window function: `sqlite3_create_window_function`
  refuses `xValue` without `xInverse` (SQLITE_MISUSE, observed on 3.37.2; the rule
  "both or neither of xValue, xInverse" is in `sqlite3CreateFunc`, read in the 3.51.3
  amalgamation bundled with libsqlite3-sys, not in 3.37.2's source), and a legacy aggregate is
  rejected at prepare time ("may not be used as a window function"; also the shipped
  `softavg_agg`). Both probed on 3.37.2 (`results.json` -> `sqlite_probe`). So
  `softavg_s2` registers an `xInverse` that only drops the oldest row of a frame
  shadow; the next `xValue` refolds the whole frame with `ScalarAcc` (what PG does
  for an aggregate without MINVFUNC).
- **PostgreSQL replay** (`replay/src/main.rs`, `replay.py`). Window: PG's
  `eval_windowaggregates` order: MINVFUNC for the row leaving, restart (fresh state,
  MSFUNC over the frame) if it returned NULL, MSFUNC for the new row, MFINALFUNC.
  S2 = restart every frame (ScalarAcc fold), S3 = `PlainInv`, S4 = `InvAcc` with
  NULL from MINVFUNC when `!is_exact()` (bruce_pg 0.2.0 `softavg_minvfunc`).
  Maintenance: the statement-level trigger of `softavg_ivm_create`: deletes removed
  from the group's state, inserts folded into a delta and merged, update = both;
  S4 recomputes groups whose state is inexact, S3 never recomputes, S2 recomputes
  every group a DELETE/UPDATE touched. **Validation:** the replay's S4 recompute
  counts equal the counts PostgreSQL itself reported when it ran the same streams
  (column `PG measured S4` in table 5) on the maint_ivm stream and on both
  time-ordered streams at every eps measured there; on the two random controls
  two cells differ by 1-2 groups (the random draw walks a list of live ids whose
  order differs slightly between the two harnesses). The maint_ivm scores differ
  in the last bits (numpy here, pgvector there); the argmax structure does not.
- **PostgreSQL engine run** (`pg_window.py`, `pg_maint.py`; PostgreSQL 18.6, cluster
  `../../pg18env`, port 28860, settings of `../postgres/README.md` plus `jit = off`,
  `max_parallel_workers_per_gather = 0`). Window: same grid, tables `lad_<row set>`
  with `rn` as PRIMARY KEY (the plans are index scans), S0 / S1 as written in table 1's
  header (PG raises on `exp` overflow and underflow, so S1 carries the clamp
  `GREATEST(..., -700)`), S2 = `softavg_fwd` (bruce_pg's SFUNC without a moving form),
  S3 = `softavg_s3` (`pg_ablation/`), S4 = shipped `softavg`. Maintenance: every arm on a
  fresh base table with the replay's statement list; an INSERT re-inserts the row's current
  values; initial rows loaded in id order (the replay's fold order); after every statement
  the touched groups are compared in the server with a fresh `GROUP BY` fold, every
  checkpoint all groups. Latency of the maintenance statements is per statement with the
  trigger inside it.
- **Maintenance streams.** (a) the 10,000-op INSERT/DELETE/UPDATE stream of
  `../postgres/maint_ivm` (IMDb, all rows, first 29 ops delete each facet's argmax);
  (b) the time-ordered retention streams of `../timeorder` on Amazon (T0 =
  2020-01-01, 50,000 inserts, retention R = 30 and 2 days) and their random
  controls. eps in {1e-4, 1e-3, 0.02, 0.1, 1}. v_scale 10 (IMDb), 5 (Amazon).
- **Trace for a figure.** `trace.json` (1.1 MB): Amazon query 0, `ORDER BY ts, id`,
  k in {50, 1000}, eps in {0.02, 1e-3}: per-row values of SQLite S0, S1, S3, S4 and
  PG-replay S3, S4 plus the reference and max|v|, over a 20,000-row slice that holds
  the worst S3 error (and the worst S1 error when both fit), every 10th row plus the
  50 worst rows of every series; the worst row of every series over the whole run is
  recorded separately (`worst_overall`).

## Results

<!-- BEGIN GENERATED (summarize.py) -->
_Timings: smoke, machine shared; part of the SQLite grid ran concurrently with this experiment's CPU-only replay job and with other users' jobs (1-min load up to ~28 on 32 cores, GPU busy); table 4b re-times S2/S3/S4 interleaved._

- SQLite 3.37.2 (system libsqlite3 3.37.2-2ubuntu0.4 behind Python 3.10.12's sqlite3); ablation .so sha256 `7fbee1386178d868...`, shipped S4 .so `88fe88e700941007...`; build: cargo build --release (opt-level=3, lto=fat, codegen-units=1), --offline, CARGO_TARGET_DIR=softmax_ladder/target
- machine at start of the SQLite window run: quiet=False, load average [1.61, 1.47, 1.35]; at assemble: load average [1.98, 2.54, 4.86]
- rows: IMDb 459,865 (predicate subset 232,828), Amazon 1,524,113 reviews with verified timestamps (1997-09-10 .. 2023-09-06); 3 queries per corpus; max score per query IMDb [1.0, 1.0, 1.0], Amazon [1.0, 1.0, 1.0]
- SQLite S2 probe: `sqlite3_create_window_function` with xValue but no xInverse returns 21 (SQLITE_MISUSE = 21); a legacy aggregate registers (rc 0) but `OVER (...)` fails at prepare: `OperationalError: softavg_s2_legacy() may not be used as a window function`; the shipped aggregate-only twin: `OperationalError: softavg_agg() may not be used as a window function`
- **PostgreSQL engine run**: PostgreSQL 18.6 (conda env ggdb-pg, release build, read-only; cluster `pg18env`, extensions via `dynamic_library_path` / `extension_control_path`); bruce_pg 0.2.0, softavg_ablation 0.1.0; build: bruce_pg, softavg_ablation: cargo pgrx package (release profile: opt-level 3, lto fat, codegen-units 1), --features pg18 --no-default-features, pgrx 0.18.1; vector 0.8.7, pg_ivm 1.16: PGXS -O2; settings: shared_buffers 4GB, work_mem 256MB, jit off, max_parallel_workers_per_gather 0
- PG timing conditions: smoke, machine shared: the 4 window processes (amazon k=1000 | amazon k=10,100 | imdb_all | imdb_ge) and the 5 maintenance processes (one per stream) ran concurrently, each single-threaded on its own backend (32 cores); another user's job kept the GPU at 100%; load averages are recorded per header / run; load average at the start of each window process [[2.38, 1.99, 2.03], [2.55, 4.95, 5.88], [2.38, 1.99, 2.03], [2.38, 1.99, 2.03], [2.38, 1.99, 2.03], [2.55, 4.95, 5.88], [2.55, 4.95, 5.88], [2.55, 4.95, 5.88]], at assemble [3.57, 4.28, 5.03]
- bruce_pg test suite on PG 18.6 (`experiments/postgres/tests_pg18.json`): 55 passed, 1 failed (tests::pg_test_i_upgrade_path_alter_extension: the test stages its probe script in the compiled-in sharedir, which PG 18 does not read when the control file comes from `extension_control_path`; the same probe by hand passes)

### 1. The ladder, sliding windows (all row sets x k in {10, 100, 1000} x 3 queries pooled)

Cell: wrong rows / rows (NULL, Inf among them); max abs err / max|v| over the finite rows. Wrong = abs err > 1e-9 max|v| of the frame, or NULL / Inf. SQLite = engine run; PostgreSQL 18.6 = engine run (`postgres_engine`); PostgreSQL (replay) = PG's call order replayed outside the server over the same Rust arithmetic (done while no PG server existed; kept for the comparison in table 3b). PG S0 cells that raise an ERROR (exp overflow) count as SQL errors, with no rows. S1x (exact SQL) ran for query 0 only, at k = 10, 100 (all row sets) and k = 1000 (IMDb subset).

| engine | rung | eps=0.0001 | eps=0.001 | eps=0.02 | eps=0.1 | eps=1 |
|---|---|---|---|---|---|---|
| SQLite | S0 naive | 19,951,236 / 19,951,254 (19,951,236 NULL, 0 Inf); 1.8e-16 | 17,874,462 / 19,951,254 (13,958,046 NULL, 0 Inf); 6.9e+283 | 13,716,901 / 19,951,254; 2.7e+05 | 0 / 19,951,254; 1.9e-12 | 0 / 19,951,254; 1.2e-13 |
| SQLite | S1 global-max shift | 19,930,658 / 19,951,254 (19,679,040 NULL, 0 Inf); 0.82 | 12,676,631 / 19,951,254 (1,441,341 NULL, 0 Inf); 6.4e+220 | 13,561,422 / 19,951,254; 1.3e+06 | 0 / 19,951,254; 1.9e-12 | 0 / 19,951,254; 2.5e-13 |
| SQLite | S1x exact SQL (frame max + correlated subquery) | 0 / 4,666,440; 7.1e-16 | 0 / 4,666,440; 2.1e-15 | 0 / 4,666,440; 5.6e-15 | 0 / 4,666,440; 4.0e-15 | 0 / 4,666,440; 3.8e-15 |
| SQLite | S2 online, no removal (frame refold) | 0 / 19,951,254; 1.1e-15 | 0 / 19,951,254; 2.1e-15 | 0 / 19,951,254; 8.5e-15 | 0 / 19,951,254; 6.0e-15 | 0 / 19,951,254; 5.5e-15 |
| SQLite | S3 online + plain removal, no certificate | 19,927,367 / 19,951,254 (19,923,204 NULL, 0 Inf); 0.82 | 12,408,954 / 19,951,254 (12 NULL, 0 Inf); 1.5e+79 | 13,818,685 / 19,951,254; 1.3e+06 | 0 / 19,951,254; 2.5e-12 | 0 / 19,951,254; 1.3e-13 |
| SQLite | S4 softavg (certificate + refold) | 0 / 19,951,254; 5.6e-12 | 0 / 19,951,254; 2.3e-11 | 0 / 19,951,254; 2.6e-11 | 0 / 19,951,254; 1.5e-12 | 0 / 19,951,254; 9.2e-14 |
| PostgreSQL 18.6 | S0 naive | 0 / 0; -; 27 SQL errors | 0 / 698,484; 7.5e-16; 24 SQL errors | 0 / 19,951,254; 8.7e-15 | 0 / 19,951,254; 5.9e-15 | 0 / 19,951,254; 5.5e-15 |
| PostgreSQL 18.6 | S1 global-max shift | 19,806,078 / 19,951,254; 0.75 | 5,229,660 / 19,951,254; 0.75 | 0 / 19,951,254; 1.0e-14 | 0 / 19,951,254; 5.7e-15 | 0 / 19,951,254; 5.7e-15 |
| PostgreSQL 18.6 | S2 online, no removal (frame refold) | 0 / 19,951,254; 1.1e-15 | 0 / 19,951,254; 2.1e-15 | 0 / 19,951,254; 8.5e-15 | 0 / 19,951,254; 6.0e-15 | 0 / 19,951,254; 5.5e-15 |
| PostgreSQL 18.6 | S3 online + plain removal, no certificate | 19,927,367 / 19,951,254 (19,923,204 NULL, 0 Inf); 0.82 | 12,408,954 / 19,951,254 (12 NULL, 0 Inf); 1.5e+79 | 13,818,685 / 19,951,254; 1.3e+06 | 0 / 19,951,254; 2.5e-12 | 0 / 19,951,254; 1.3e-13 |
| PostgreSQL 18.6 | S4 softavg (certificate + refold) | 0 / 19,951,254; 5.6e-12 | 0 / 18,427,141; 2.3e-11; 1 SQL errors | 0 / 19,951,254; 2.6e-11 | 0 / 19,951,254; 1.5e-12 | 0 / 19,951,254; 9.2e-14 |
| PostgreSQL (replay) | S2 online, no removal (frame refold) | 0 / 19,951,254; 1.1e-15 | 0 / 19,951,254; 2.1e-15 | 0 / 19,951,254; 8.5e-15 | 0 / 19,951,254; 6.0e-15 | 0 / 19,951,254; 5.5e-15 |
| PostgreSQL (replay) | S3 online + plain removal, no certificate | 19,927,367 / 19,951,254 (19,923,204 NULL, 0 Inf); 0.82 | 12,408,954 / 19,951,254 (12 NULL, 0 Inf); 1.5e+79 | 13,818,685 / 19,951,254; 1.3e+06 | 0 / 19,951,254; 2.5e-12 | 0 / 19,951,254; 1.3e-13 |
| PostgreSQL (replay) | S4 softavg (certificate + refold) | 0 / 19,951,254; 5.6e-12 | 0 / 19,951,254; 2.3e-11 | 0 / 19,951,254; 2.6e-11 | 0 / 19,951,254; 1.5e-12 | 0 / 19,951,254; 9.2e-14 |

### 2. S3 vs S4 per row set and frame (SQLite engine run; 3 queries pooled)

Cell: S3 wrong rows (NULL/Inf) ; S3 max abs err / max|v| || S4 max abs err / max|v| ; S4 refolds. PG replay S3 / S4 in the next table.

| row set | k | eps=0.0001 | eps=0.001 | eps=0.02 | eps=0.1 | eps=1 |
|---|---|---|---|---|---|---|
| IMDb all | 10 | 1,379,427 (1,379,385) ; 0.82 \|\| 5.2e-12 ; 128,084 | 1,158,856 ; 1.0e+12 \|\| 1.1e-11 ; 108,559 | 1,234,466 ; 1.3e+06 \|\| 2.6e-11 ; 8,900 | 0 ; 6.4e-13 \|\| 4.1e-13 ; 27 | 0 ; 7.1e-14 \|\| 5.8e-14 ; 5 |
| IMDb all | 100 | 1,379,021 (1,379,014) ; 0.49 \|\| 6.5e-13 ; 14,557 | 1,130,249 ; 4.1e+14 \|\| 1.8e-12 ; 12,332 | 1,137,982 ; 1967 \|\| 8.1e-12 ; 1,274 | 0 ; 2.0e-13 \|\| 1.6e-13 ; 7 | 0 ; 2.1e-14 \|\| 3.2e-14 ; 3 |
| IMDb all | 1000 | 1,372,307 (1,369,342) ; 0.53 \|\| 1.2e-14 ; 1,473 | 1,054,085 ; 357 \|\| 8.6e-14 ; 1,289 | 974,293 ; 0.07 \|\| 1.2e-12 ; 203 | 0 ; 1.0e-13 \|\| 5.1e-14 ; 4 | 0 ; 3.4e-14 \|\| 3.3e-14 ; 3 |
| IMDb filter_num>=2000 | 10 | 698,325 (698,305) ; 0.6 \|\| 5.0e-12 ; 64,675 | 577,017 ; 4291 \|\| 1.2e-11 ; 54,994 | 534,707 ; 6.4e+05 \|\| 2.5e-11 ; 4,473 | 0 ; 2.7e-13 \|\| 1.7e-13 ; 10 | 0 ; 4.8e-14 \|\| 4.8e-14 ; 3 |
| IMDb filter_num>=2000 | 100 | 697,950 (697,944) ; 0.54 \|\| 4.5e-13 ; 7,253 | 575,893 ; 26 \|\| 1.2e-12 ; 6,151 | 402,893 ; 508 \|\| 5.7e-12 ; 643 | 0 ; 6.5e-14 \|\| 6.5e-14 ; 2 | 0 ; 2.1e-14 \|\| 2.1e-14 ; 0 |
| IMDb filter_num>=2000 | 1000 | 693,458 (693,455) ; 0.58 \|\| 6.2e-15 ; 710 | 531,562 ; 94 \|\| 7.6e-14 ; 626 | 287,685 ; 0.13 \|\| 1.7e-12 ; 86 | 0 ; 3.0e-14 \|\| 3.0e-14 ; 1 | 0 ; 1.4e-14 \|\| 1.4e-14 ; 0 |
| Amazon (ts order) | 10 | 4,571,992 (4,571,793) ; 0.6 \|\| 5.6e-12 ; 411,111 | 2,306,920 (12) ; 1.5e+79 \|\| 2.3e-11 ; 364,938 | 4,217,925 ; 8.3e+05 \|\| 2.1e-11 ; 53,996 | 0 ; 2.5e-12 \|\| 1.5e-12 ; 156 | 0 ; 1.3e-13 \|\| 9.2e-14 ; 27 |
| Amazon (ts order) | 100 | 4,571,303 (4,571,157) ; 0.4 \|\| 6.3e-13 ; 44,875 | 2,587,268 ; 3603 \|\| 1.4e-12 ; 39,280 | 3,268,997 ; 9293 \|\| 9.7e-12 ; 5,302 | 0 ; 5.3e-13 \|\| 2.0e-13 ; 34 | 0 ; 1.1e-13 \|\| 3.3e-14 ; 18 |
| Amazon (ts order) | 1000 | 4,563,584 (4,562,809) ; 0.8 \|\| 5.2e-14 ; 4,530 | 2,487,104 ; 54 \|\| 1.0e-13 ; 4,032 | 1,759,737 ; 0.014 \|\| 2.0e-12 ; 779 | 0 ; 1.3e-13 \|\| 5.1e-14 ; 19 | 0 ; 1.8e-14 \|\| 1.8e-14 ; 15 |

### 3. PostgreSQL replay: S3 vs S4 per row set and frame (3 queries pooled)

Cell: S3 wrong rows (NULL/Inf) ; S3 max abs err / max|v| || S4 max abs err / max|v| ; S4 punts (MINVFUNC returned NULL, frame restarted) / rows re-absorbed by the restarts.

| row set | k | eps=0.0001 | eps=0.001 | eps=0.02 | eps=0.1 | eps=1 |
|---|---|---|---|---|---|---|
| IMDb all | 10 | 1,379,427 (1,379,385) ; 0.82 \|\| 5.2e-12 ; 128,084 / 1,408,924 | 1,158,856 ; 1.0e+12 \|\| 1.1e-11 ; 108,559 / 1,194,149 | 1,234,466 ; 1.3e+06 \|\| 2.6e-11 ; 8,900 / 97,900 | 0 ; 6.4e-13 \|\| 4.1e-13 ; 27 / 297 | 0 ; 7.1e-14 \|\| 5.8e-14 ; 5 / 55 |
| IMDb all | 100 | 1,379,021 (1,379,014) ; 0.49 \|\| 6.5e-13 ; 14,557 / 1,470,257 | 1,130,249 ; 4.1e+14 \|\| 1.8e-12 ; 12,332 / 1,245,532 | 1,137,982 ; 1967 \|\| 8.1e-12 ; 1,274 / 128,674 | 0 ; 2.0e-13 \|\| 1.6e-13 ; 7 / 707 | 0 ; 2.1e-14 \|\| 3.2e-14 ; 3 / 303 |
| IMDb all | 1000 | 1,372,307 (1,369,342) ; 0.53 \|\| 1.2e-14 ; 1,473 / 1,474,473 | 1,054,085 ; 357 \|\| 8.6e-14 ; 1,289 / 1,290,289 | 974,293 ; 0.07 \|\| 1.2e-12 ; 203 / 203,203 | 0 ; 1.0e-13 \|\| 5.1e-14 ; 4 / 4,004 | 0 ; 3.4e-14 \|\| 3.3e-14 ; 3 / 3,003 |
| IMDb filter_num>=2000 | 10 | 698,325 (698,305) ; 0.6 \|\| 5.0e-12 ; 64,675 / 711,425 | 577,017 ; 4291 \|\| 1.2e-11 ; 54,994 / 604,934 | 534,707 ; 6.4e+05 \|\| 2.5e-11 ; 4,473 / 49,203 | 0 ; 2.7e-13 \|\| 1.7e-13 ; 10 / 110 | 0 ; 4.8e-14 \|\| 4.8e-14 ; 3 / 33 |
| IMDb filter_num>=2000 | 100 | 697,950 (697,944) ; 0.54 \|\| 4.5e-13 ; 7,253 / 732,553 | 575,893 ; 26 \|\| 1.2e-12 ; 6,151 / 621,251 | 402,893 ; 508 \|\| 5.7e-12 ; 643 / 64,943 | 0 ; 6.5e-14 \|\| 6.5e-14 ; 2 / 202 | 0 ; 2.1e-14 \|\| 2.1e-14 ; 0 / 0 |
| IMDb filter_num>=2000 | 1000 | 693,458 (693,455) ; 0.58 \|\| 6.2e-15 ; 710 / 710,710 | 531,562 ; 94 \|\| 7.6e-14 ; 626 / 626,626 | 287,685 ; 0.13 \|\| 1.7e-12 ; 86 / 86,086 | 0 ; 3.0e-14 \|\| 3.0e-14 ; 1 / 1,001 | 0 ; 1.4e-14 \|\| 1.4e-14 ; 0 / 0 |
| Amazon (ts order) | 10 | 4,571,992 (4,571,793) ; 0.6 \|\| 5.6e-12 ; 411,111 / 4,522,221 | 2,306,920 (12) ; 1.5e+79 \|\| 2.3e-11 ; 364,938 / 4,014,318 | 4,217,925 ; 8.3e+05 \|\| 2.1e-11 ; 53,996 / 593,956 | 0 ; 2.5e-12 \|\| 1.5e-12 ; 156 / 1,716 | 0 ; 1.3e-13 \|\| 9.2e-14 ; 27 / 297 |
| Amazon (ts order) | 100 | 4,571,303 (4,571,157) ; 0.4 \|\| 6.3e-13 ; 44,875 / 4,532,375 | 2,587,268 ; 3603 \|\| 1.4e-12 ; 39,280 / 3,967,280 | 3,268,997 ; 9293 \|\| 9.7e-12 ; 5,302 / 535,502 | 0 ; 5.3e-13 \|\| 2.0e-13 ; 34 / 3,434 | 0 ; 1.1e-13 \|\| 3.3e-14 ; 18 / 1,818 |
| Amazon (ts order) | 1000 | 4,563,584 (4,562,809) ; 0.8 \|\| 5.2e-14 ; 4,530 / 4,534,530 | 2,487,104 ; 54 \|\| 1.0e-13 ; 4,032 / 4,036,032 | 1,759,737 ; 0.014 \|\| 2.0e-12 ; 779 / 779,779 | 0 ; 1.3e-13 \|\| 5.1e-14 ; 19 / 19,019 | 0 ; 1.8e-14 \|\| 1.8e-14 ; 15 / 15,015 |

### 3b. PostgreSQL 18.6 engine vs the call-order replay (window cells, S2 / S3 / S4)

Per cell (row set x k x eps x query) the engine's error summary is compared with the replay's: n_rows, NULL, Inf, wrong (finite / all), max abs err / max|v|, max rel err and the worst row, each compared exactly (float equality). S4 punts: `softavg_window_stats()` in the server vs the replay's MINVFUNC-NULL count.

| rung | cells compared | engine ERROR | all fields equal | wrong / NULL / Inf counts equal | max rel. diff of max err | S4 punts equal |
|---|---|---|---|---|---|---|
| S2 | 135 | 0 | 135 | 135 | 0 | - |
| S3 | 135 | 0 | 135 | 135 | 0 | - |
| S4 | 134 | 1 | 134 | 134 | 0 | 134 / 134 |

**S4 ['amazon', 100, 0.001, 2] raised in the server**: `ERROR:  softavg: moving-aggregate state is inexact at finalize` (SQLSTATE XX000); the replay returned a value there (0 wrong rows) and logged `S4 inexact after absorb at row 808229`.

### 3c. PostgreSQL 18.6 engine: S3 vs S4 per row set and frame (3 queries pooled)

Cell: S3 wrong rows (NULL/Inf) ; S3 max abs err / max|v| || S4 max abs err / max|v| ; S4 punts (`softavg_window_stats()`: MINVFUNC returned NULL, PG restarted the frame).

| row set | k | eps=0.0001 | eps=0.001 | eps=0.02 | eps=0.1 | eps=1 |
|---|---|---|---|---|---|---|
| IMDb all | 10 | 1,379,427 (1,379,385) ; 0.82 \|\| 5.2e-12 ; 128,084 | 1,158,856 ; 1.0e+12 \|\| 1.1e-11 ; 108,559 | 1,234,466 ; 1.3e+06 \|\| 2.6e-11 ; 8,900 | 0 ; 6.4e-13 \|\| 4.1e-13 ; 27 | 0 ; 7.1e-14 \|\| 5.8e-14 ; 5 |
| IMDb all | 100 | 1,379,021 (1,379,014) ; 0.49 \|\| 6.5e-13 ; 14,557 | 1,130,249 ; 4.1e+14 \|\| 1.8e-12 ; 12,332 | 1,137,982 ; 1967 \|\| 8.1e-12 ; 1,274 | 0 ; 2.0e-13 \|\| 1.6e-13 ; 7 | 0 ; 2.1e-14 \|\| 3.2e-14 ; 3 |
| IMDb all | 1000 | 1,372,307 (1,369,342) ; 0.53 \|\| 1.2e-14 ; 1,473 | 1,054,085 ; 357 \|\| 8.6e-14 ; 1,289 | 974,293 ; 0.07 \|\| 1.2e-12 ; 203 | 0 ; 1.0e-13 \|\| 5.1e-14 ; 4 | 0 ; 3.4e-14 \|\| 3.3e-14 ; 3 |
| IMDb filter_num>=2000 | 10 | 698,325 (698,305) ; 0.6 \|\| 5.0e-12 ; 64,675 | 577,017 ; 4291 \|\| 1.2e-11 ; 54,994 | 534,707 ; 6.4e+05 \|\| 2.5e-11 ; 4,473 | 0 ; 2.7e-13 \|\| 1.7e-13 ; 10 | 0 ; 4.8e-14 \|\| 4.8e-14 ; 3 |
| IMDb filter_num>=2000 | 100 | 697,950 (697,944) ; 0.54 \|\| 4.5e-13 ; 7,253 | 575,893 ; 26 \|\| 1.2e-12 ; 6,151 | 402,893 ; 508 \|\| 5.7e-12 ; 643 | 0 ; 6.5e-14 \|\| 6.5e-14 ; 2 | 0 ; 2.1e-14 \|\| 2.1e-14 ; 0 |
| IMDb filter_num>=2000 | 1000 | 693,458 (693,455) ; 0.58 \|\| 6.2e-15 ; 710 | 531,562 ; 94 \|\| 7.6e-14 ; 626 | 287,685 ; 0.13 \|\| 1.7e-12 ; 86 | 0 ; 3.0e-14 \|\| 3.0e-14 ; 1 | 0 ; 1.4e-14 \|\| 1.4e-14 ; 0 |
| Amazon (ts order) | 10 | 4,571,992 (4,571,793) ; 0.6 \|\| 5.6e-12 ; 411,111 | 2,306,920 (12) ; 1.5e+79 \|\| 2.3e-11 ; 364,938 | 4,217,925 ; 8.3e+05 \|\| 2.1e-11 ; 53,996 | 0 ; 2.5e-12 \|\| 1.5e-12 ; 156 | 0 ; 1.3e-13 \|\| 9.2e-14 ; 27 |
| Amazon (ts order) | 100 | 4,571,303 (4,571,157) ; 0.4 \|\| 6.3e-13 ; 44,875 | 2,587,268 ; 3603 \|\| 1.2e-12 ; 26,530 | 3,268,997 ; 9293 \|\| 9.7e-12 ; 5,302 | 0 ; 5.3e-13 \|\| 2.0e-13 ; 34 | 0 ; 1.1e-13 \|\| 3.3e-14 ; 18 |
| Amazon (ts order) | 1000 | 4,563,584 (4,562,809) ; 0.8 \|\| 5.2e-14 ; 4,530 | 2,487,104 ; 54 \|\| 1.0e-13 ; 4,032 | 1,759,737 ; 0.014 \|\| 2.0e-12 ; 779 | 0 ; 1.3e-13 \|\| 5.1e-14 ; 19 | 0 ; 1.8e-14 \|\| 1.8e-14 ; 15 |

### 4. SQLite latency (query 0, median of 3, ms) and what S4 costs over S3 / S2

eps = 0.001

| row set | k | S0 | S1 | S1x | S2 | S3 | S4 | S4 / S3 | S2 / S4 |
|---|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | 204 | 247 | 3148 | 220 | 197 | 212 | 1.08 | 1.04 |
| IMDb all | 100 | 206 | 248 | 9587 | 411 | 240 | 260 | 1.08 | 1.58 |
| IMDb all | 1000 | 287 | 335 | - | 2013 | 289 | 302 | 1.04 | 6.66 |
| IMDb filter_num>=2000 | 10 | 104 | 127 | 748 | 116 | 100 | 123 | 1.23 | 0.94 |
| IMDb filter_num>=2000 | 100 | 109 | 128 | 4728 | 216 | 126 | 133 | 1.06 | 1.62 |
| IMDb filter_num>=2000 | 1000 | 146 | 164 | 45309 | 1013 | 145 | 154 | 1.06 | 6.59 |
| Amazon (ts order) | 10 | 711 | 866 | 5187 | 771 | 683 | 740 | 1.08 | 1.04 |
| Amazon (ts order) | 100 | 1477 | 1787 | 31698 | 2637 | 847 | 892 | 1.05 | 2.96 |
| Amazon (ts order) | 1000 | 1147 | 1298 | - | 10496 | 1738 | 1380 | 0.79 | 7.61 |

eps = 0.02

| row set | k | S0 | S1 | S1x | S2 | S3 | S4 | S4 / S3 | S2 / S4 |
|---|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | 206 | 239 | 1958 | 221 | 195 | 206 | 1.06 | 1.07 |
| IMDb all | 100 | 208 | 239 | 9320 | 410 | 239 | 248 | 1.04 | 1.65 |
| IMDb all | 1000 | 289 | 326 | - | 1946 | 286 | 296 | 1.03 | 6.58 |
| IMDb filter_num>=2000 | 10 | 109 | 127 | 747 | 117 | 103 | 109 | 1.06 | 1.07 |
| IMDb filter_num>=2000 | 100 | 109 | 125 | 4727 | 215 | 125 | 130 | 1.05 | 1.65 |
| IMDb filter_num>=2000 | 1000 | 152 | 173 | 43995 | 986 | 145 | 150 | 1.03 | 6.57 |
| Amazon (ts order) | 10 | 715 | 837 | 5193 | 773 | 684 | 720 | 1.05 | 1.07 |
| Amazon (ts order) | 100 | 721 | 842 | 31466 | 1436 | 839 | 869 | 1.04 | 1.65 |
| Amazon (ts order) | 1000 | 1167 | 1321 | - | 9825 | 1691 | 1812 | 1.07 | 5.42 |

eps = 1

| row set | k | S0 | S1 | S1x | S2 | S3 | S4 | S4 / S3 | S2 / S4 |
|---|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | 207 | 240 | 2722 | 221 | 195 | 209 | 1.07 | 1.05 |
| IMDb all | 100 | 216 | 252 | 9355 | 408 | 238 | 248 | 1.04 | 1.64 |
| IMDb all | 1000 | 290 | 326 | - | 1931 | 287 | 295 | 1.03 | 6.54 |
| IMDb filter_num>=2000 | 10 | 109 | 126 | 745 | 117 | 103 | 108 | 1.05 | 1.08 |
| IMDb filter_num>=2000 | 100 | 109 | 129 | 4723 | 216 | 125 | 129 | 1.04 | 1.67 |
| IMDb filter_num>=2000 | 1000 | 157 | 176 | 44145 | 978 | 146 | 149 | 1.03 | 6.54 |
| Amazon (ts order) | 10 | 750 | 863 | 4967 | 765 | 1263 | 1172 | 0.93 | 0.65 |
| Amazon (ts order) | 100 | 684 | 805 | 33057 | 1363 | 803 | 827 | 1.03 | 1.65 |
| Amazon (ts order) | 1000 | 2190 | 1274 | - | 13685 | 1757 | 1853 | 1.05 | 7.38 |

### 4b. SQLite, interleaved timing of S2 / S3 / S4 (query 0; 7 repetitions, rungs rotated within each; median ms; ratio = median of per-repetition ratios)

| row set | k | eps | S2 | S3 | S4 | S4 / S3 | S2 / S4 | load avg |
|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | 0.001 | 220 | 196 | 210 | 1.077 | 1.04 | 2.41 |
| IMDb all | 10 | 0.02 | 220 | 195 | 207 | 1.053 | 1.08 | 2.45 |
| IMDb all | 10 | 1 | 220 | 195 | 203 | 1.049 | 1.08 | 2.5 |
| IMDb all | 100 | 0.001 | 411 | 239 | 252 | 1.057 | 1.63 | 2.42 |
| IMDb all | 100 | 0.02 | 408 | 238 | 247 | 1.037 | 1.65 | 2.47 |
| IMDb all | 100 | 1 | 410 | 237 | 246 | 1.036 | 1.65 | 2.48 |
| IMDb all | 1000 | 0.001 | 1919 | 279 | 285 | 1.027 | 6.73 | 2.49 |
| IMDb all | 1000 | 0.02 | 1855 | 273 | 283 | 1.037 | 6.54 | 2.35 |
| IMDb all | 1000 | 1 | 1841 | 269 | 279 | 1.035 | 6.60 | 2.25 |
| IMDb filter_num>=2000 | 10 | 0.001 | 112 | 99.7 | 107 | 1.070 | 1.05 | 2.31 |
| IMDb filter_num>=2000 | 10 | 0.02 | 111 | 97.4 | 103 | 1.054 | 1.07 | 2.28 |
| IMDb filter_num>=2000 | 10 | 1 | 111 | 97.8 | 102 | 1.046 | 1.08 | 2.28 |
| IMDb filter_num>=2000 | 100 | 0.001 | 205 | 119 | 125 | 1.054 | 1.64 | 2.26 |
| IMDb filter_num>=2000 | 100 | 0.02 | 206 | 122 | 123 | 1.036 | 1.66 | 2.24 |
| IMDb filter_num>=2000 | 100 | 1 | 205 | 118 | 122 | 1.034 | 1.67 | 2.22 |
| IMDb filter_num>=2000 | 1000 | 0.001 | 967 | 137 | 144 | 1.050 | 6.69 | 2.19 |
| IMDb filter_num>=2000 | 1000 | 0.02 | 937 | 136 | 142 | 1.037 | 6.62 | 2.16 |
| IMDb filter_num>=2000 | 1000 | 1 | 929 | 136 | 141 | 1.035 | 6.60 | 2.05 |
| Amazon (ts order) | 10 | 0.001 | 738 | 651 | 703 | 1.081 | 1.05 | 2.12 |
| Amazon (ts order) | 10 | 0.02 | 730 | 648 | 699 | 1.079 | 1.04 | 2.28 |
| Amazon (ts order) | 10 | 1 | 735 | 648 | 678 | 1.047 | 1.08 | 2.45 |
| Amazon (ts order) | 100 | 0.001 | 1391 | 803 | 847 | 1.055 | 1.64 | 2.29 |
| Amazon (ts order) | 100 | 0.02 | 1362 | 800 | 825 | 1.033 | 1.65 | 2.25 |
| Amazon (ts order) | 100 | 1 | 1354 | 799 | 827 | 1.035 | 1.64 | 2.43 |
| Amazon (ts order) | 1000 | 0.001 | 7174 | 920 | 972 | 1.057 | 7.38 | 2.29 |
| Amazon (ts order) | 1000 | 0.02 | 6170 | 900 | 938 | 1.042 | 6.57 | 2.25 |
| Amazon (ts order) | 1000 | 1 | 6128 | 899 | 933 | 1.039 | 6.57 | 2.16 |

S4 / S3 over the 27 cells: min 1.027, median 1.046, max 1.081; S2 / S4 at k = 1000: 6.54 .. 7.38.

### 4c. PostgreSQL 18.6 latency (query 0, median of 3 or 1 run when the fetch took > 20 s, ms)

`SELECT count(x), sum(x) FROM (window query)`. `err` = the query raised an ERROR (S0: exp overflow), no latency. S0, S1 use PG's `sum(float8)`, which has no inverse, so PG recomputes every frame (O(k) per row), like S2.

eps = 0.001

| row set | k | S0 | S1 | S2 | S3 | S4 | S4 / S3 | S2 / S4 |
|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | err | 577 | 298 | 121 | 146 | 1.20 | 2.05 |
| IMDb all | 100 | err | 4962 | 2046 | 134 | 161 | 1.20 | 12.72 |
| IMDb all | 1000 | err | 48807 | 19158 | 135 | 162 | 1.19 | 118.62 |
| IMDb filter_num>=2000 | 10 | 220 | 295 | 159 | 64.3 | 74.0 | 1.15 | 2.15 |
| IMDb filter_num>=2000 | 100 | 1668 | 2245 | 1562 | 62.2 | 111 | 1.78 | 14.12 |
| IMDb filter_num>=2000 | 1000 | 18295 | 24601 | 9821 | 71.0 | 82.9 | 1.17 | 118.42 |
| Amazon (ts order) | 10 | err | 1956 | 1005 | 407 | 490 | 1.20 | 2.05 |
| Amazon (ts order) | 100 | err | 16922 | 7159 | 463 | 547 | 1.18 | 13.09 |
| Amazon (ts order) | 1000 | err | 147612 | 57501 | 391 | 458 | 1.17 | 125.46 |

eps = 0.02

| row set | k | S0 | S1 | S2 | S3 | S4 | S4 / S3 | S2 / S4 |
|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | 433 | 559 | 299 | 120 | 186 | 1.55 | 1.61 |
| IMDb all | 100 | 3728 | 4813 | 2087 | 138 | 144 | 1.04 | 14.49 |
| IMDb all | 1000 | 32838 | 42807 | 17724 | 124 | 128 | 1.03 | 138.44 |
| IMDb filter_num>=2000 | 10 | 220 | 283 | 151 | 61.3 | 94.8 | 1.55 | 1.60 |
| IMDb filter_num>=2000 | 100 | 1883 | 2459 | 1055 | 68.8 | 73.5 | 1.07 | 14.35 |
| IMDb filter_num>=2000 | 1000 | 18292 | 24026 | 9695 | 70.4 | 94.8 | 1.35 | 102.28 |
| Amazon (ts order) | 10 | 1454 | 1900 | 1173 | 464 | 492 | 1.06 | 2.38 |
| Amazon (ts order) | 100 | 12389 | 16295 | 7023 | 459 | 490 | 1.07 | 14.33 |
| Amazon (ts order) | 1000 | 108924 | 139337 | 56407 | 406 | 425 | 1.05 | 132.73 |

eps = 1

| row set | k | S0 | S1 | S2 | S3 | S4 | S4 / S3 | S2 / S4 |
|---|---|---|---|---|---|---|---|---|
| IMDb all | 10 | 442 | 568 | 302 | 120 | 127 | 1.06 | 2.37 |
| IMDb all | 100 | 3587 | 4650 | 1953 | 127 | 131 | 1.03 | 14.89 |
| IMDb all | 1000 | 32909 | 42241 | 17024 | 123 | 126 | 1.02 | 134.90 |
| IMDb filter_num>=2000 | 10 | 226 | 283 | 239 | 60.8 | 62.6 | 1.03 | 3.82 |
| IMDb filter_num>=2000 | 100 | 1852 | 2429 | 1064 | 68.8 | 71.3 | 1.04 | 14.93 |
| IMDb filter_num>=2000 | 1000 | 16607 | 20837 | 8598 | 62.3 | 64.4 | 1.03 | 133.58 |
| Amazon (ts order) | 10 | 1654 | 2146 | 1156 | 462 | 476 | 1.03 | 2.43 |
| Amazon (ts order) | 100 | 11392 | 14797 | 6421 | 424 | 435 | 1.03 | 14.74 |
| Amazon (ts order) | 1000 | 108966 | 139375 | 56482 | 404 | 413 | 1.02 | 136.67 |

### 5. Maintenance (PostgreSQL softavg_ivm trigger semantics, replayed outside the server; no latency)

After every statement, each group the statement touched is compared with a fresh fold of its live rows (the view's `fresh query`); at every checkpoint all groups are. Cell: groups recomputed ; max |view - fresh| / v_scale over all per-statement checks ; wrong checks (> 1e-9 v_scale, NULL or non-finite) ; first wrong statement. `PG measured S4` = groups the shipped softavg_ivm recomputed when PostgreSQL ran the same stream (experiments/postgres, experiments/timeorder; eps 1e-3, 0.02, 0.1 only).

| stream | eps | S2 | S3 | S4 | PG measured S4 |
|---|---|---|---|---|---|
| maint_ivm_10k | 0.0001 | 5,521 ; 0 ; 0 | 0 ; 0.3 ; 223 (NULL 36, non-finite 178) ; #0 | 29 ; 0 ; 0 | - |
| maint_ivm_10k | 0.001 | 5,521 ; 1.8e-16 ; 0 | 0 ; 0.3 ; 21 (NULL 10, non-finite 0) ; #1 | 24 ; 4.4e-14 ; 0 | 24 |
| maint_ivm_10k | 0.02 | 5,521 ; 5.9e-15 ; 0 | 0 ; 1.1e-07 ; 2 ; #4 | 2 ; 2.4e-13 ; 0 | 2 |
| maint_ivm_10k | 0.1 | 5,521 ; 1.8e-14 ; 0 | 0 ; 1.8e-14 ; 0 | 0 ; 1.8e-14 ; 0 | 0 |
| maint_ivm_10k | 1 | 5,521 ; 9.9e-15 ; 0 | 0 ; 9.9e-15 ; 0 | 0 ; 9.9e-15 ; 0 | - |
| amazon_time_R30 | 0.0001 | 52,162 ; 0 ; 0 | 0 ; 1.2 ; 48,922 (NULL 70, non-finite 46194) ; #3442 | 86 ; 2.0e-15 ; 0 | - |
| amazon_time_R30 | 0.001 | 52,162 ; 3.6e-16 ; 0 | 0 ; 2.9 ; 23,819 (NULL 18, non-finite 0) ; #5621 | 68 ; 1.1e-13 ; 0 | 68 |
| amazon_time_R30 | 0.02 | 52,162 ; 3.9e-15 ; 0 | 0 ; 6.5e-12 ; 0 | 6 ; 4.4e-13 ; 0 | 6 |
| amazon_time_R30 | 0.1 | 52,162 ; 5.0e-15 ; 0 | 0 ; 1.5e-14 ; 0 | 0 ; 1.5e-14 ; 0 | 0 |
| amazon_time_R30 | 1 | 52,162 ; 2.3e-15 ; 0 | 0 ; 1.1e-14 ; 0 | 0 ; 1.1e-14 ; 0 | - |
| amazon_random_R30 | 0.0001 | 52,173 ; 0 ; 0 | 0 ; 1.7 ; 41,618 (NULL 90, non-finite 33663) ; #1514 | 81 ; 7.1e-16 ; 0 | - |
| amazon_random_R30 | 0.001 | 52,173 ; 3.6e-16 ; 0 | 0 ; 9.5e+139 ; 19,227 (NULL 19, non-finite 0) ; #1514 | 65 ; 1.9e-13 ; 0 | 66 |
| amazon_random_R30 | 0.02 | 52,173 ; 3.4e-15 ; 0 | 0 ; 1.0e-12 ; 0 | 2 ; 5.4e-13 ; 0 | 2 |
| amazon_random_R30 | 0.1 | 52,173 ; 3.4e-15 ; 0 | 0 ; 7.6e-15 ; 0 | 0 ; 7.6e-15 ; 0 | 0 |
| amazon_random_R30 | 1 | 52,173 ; 3.2e-15 ; 0 | 0 ; 1.2e-14 ; 0 | 0 ; 1.2e-14 ; 0 | - |
| amazon_time_R2 | 0.0001 | 49,027 ; 1.8e-16 ; 0 | 0 ; 1.3 ; 93,224 (NULL 104, non-finite 92540) ; #263 | 1,209 ; 1.3e-13 ; 0 | - |
| amazon_time_R2 | 0.001 | 49,027 ; 5.3e-16 ; 0 | 0 ; 2.2e+118 ; 40,467 (NULL 228, non-finite 0) ; #263 | 1,079 ; 2.7e-12 ; 0 | 1079 |
| amazon_time_R2 | 0.02 | 49,027 ; 1.2e-15 ; 0 | 0 ; 4.7e-06 ; 465 ; #5636 | 196 ; 2.9e-12 ; 0 | 196 |
| amazon_time_R2 | 0.1 | 49,027 ; 1.1e-15 ; 0 | 0 ; 8.3e-14 ; 0 | 0 ; 8.3e-14 ; 0 | 0 |
| amazon_time_R2 | 1 | 49,027 ; 1.2e-15 ; 0 | 0 ; 1.1e-14 ; 0 | 0 ; 1.1e-14 ; 0 | - |
| amazon_random_R2 | 0.0001 | 49,028 ; 0 ; 0 | 0 ; 2.3 ; 91,499 (NULL 151, non-finite 90077) ; #27 | 1,211 ; 6.1e-13 ; 0 | - |
| amazon_random_R2 | 0.001 | 49,028 ; 5.3e-16 ; 0 | 0 ; 2.1e+210 ; 40,521 (NULL 308, non-finite 0) ; #103 | 1,080 ; 3.4e-12 ; 0 | 1078 |
| amazon_random_R2 | 0.02 | 49,028 ; 1.4e-15 ; 0 | 0 ; 1.1e-06 ; 703 ; #10488 | 163 ; 3.2e-12 ; 0 | 163 |
| amazon_random_R2 | 0.1 | 49,028 ; 1.4e-15 ; 0 | 0 ; 3.9e-14 ; 0 | 0 ; 3.9e-14 ; 0 | 0 |
| amazon_random_R2 | 1 | 49,028 ; 1.2e-15 ; 0 | 0 ; 2.0e-14 ; 0 | 0 ; 2.0e-14 ; 0 | - |

- `maint_ivm_10k` setup: corpus=imdb, rows=459865, n_ops=10000, mix={'delete': 4505, 'insert': 4479, 'update': 1016}, first_ops_delete_argmax_rows=29, seed=7, checkpoint_every=1000, v_scale=10.0
- `amazon_time_R2` setup: corpus=amazon, t0=2020-01-01, retention_days=2, initial_rows=1896, n_inserts=50000, statements=74726, delete_statements=24726, rows_deleted=50887, v_scale=5.0, checkpoint_every_statements=2000

### 5b. Maintenance, PostgreSQL 18.6 engine run (same streams, statement triggers in the server)

Same checks as table 5, done in the server after every statement (touched groups vs a fresh `GROUP BY` fold) and at every checkpoint. Cell: groups recomputed ; max |view - fresh| / v_scale ; wrong checks (NULL, non-finite) ; first wrong statement ; median ms per statement (trigger included). `= replay`: groups recomputed, wrong / NULL / non-finite / checked counts and first wrong statement all equal to table 5, per arm (S2 S3 S4).

| stream | eps | S2 | S3 | S4 | = replay |
|---|---|---|---|---|---|
| maint_ivm_10k | 0.0001 | 5,521 ; 0 ; 0 ; 1.44 ms | 0 ; 0.3 ; 223 (NULL 36, non-finite 178) ; #0 ; 0.17 ms | 29 ; 0 ; 0 ; 0.18 ms | y y y |
| maint_ivm_10k | 0.001 | 5,521 ; 0 ; 0 ; 1.42 ms | 0 ; 0.3 ; 21 (NULL 10, non-finite 0) ; #1 ; 0.18 ms | 24 ; 4.4e-14 ; 0 ; 0.18 ms | y y y |
| maint_ivm_10k | 0.02 | 5,521 ; 2.7e-16 ; 0 ; 1.54 ms | 0 ; 1.1e-07 ; 2 ; #4 ; 0.17 ms | 2 ; 2.4e-13 ; 0 ; 0.17 ms | y y y |
| maint_ivm_10k | 0.1 | 5,521 ; 5.3e-16 ; 0 ; 1.44 ms | 0 ; 2.2e-14 ; 0 ; 0.18 ms | 0 ; 2.2e-14 ; 0 ; 0.19 ms | y y y |
| maint_ivm_10k | 1 | 5,521 ; 5.3e-16 ; 0 ; 1.32 ms | 0 ; 1.1e-14 ; 0 ; 0.18 ms | 0 ; 1.1e-14 ; 0 ; 0.18 ms | y y y |
| amazon_time_R30 | 0.0001 | 52,162 ; 0 ; 0 ; 0.19 ms | 0 ; 1.2 ; 48,922 (NULL 70, non-finite 46194) ; #3442 ; 0.15 ms | 86 ; 2.0e-15 ; 0 ; 0.14 ms | y y y |
| amazon_time_R30 | 0.001 | 52,162 ; 0 ; 0 ; 0.19 ms | 0 ; 2.9 ; 23,819 (NULL 18, non-finite 0) ; #5621 ; 0.14 ms | 69 ; 1.1e-13 ; 0 ; 0.14 ms | y y **n** |
| amazon_time_R30 | 0.02 | 52,162 ; 0 ; 0 ; 0.19 ms | 0 ; 6.5e-12 ; 0 ; 0.16 ms | 6 ; 4.4e-13 ; 0 ; 0.15 ms | y y y |
| amazon_time_R30 | 0.1 | 52,162 ; 0 ; 0 ; 0.18 ms | 0 ; 1.5e-14 ; 0 ; 0.13 ms | 0 ; 1.5e-14 ; 0 ; 0.12 ms | y y y |
| amazon_time_R30 | 1 | 52,162 ; 3.2e-15 ; 0 ; 0.15 ms | 0 ; 1.1e-14 ; 0 ; 0.13 ms | 0 ; 1.1e-14 ; 0 ; 0.12 ms | y y y |
| amazon_random_R30 | 0.0001 | 52,173 ; 0 ; 0 ; 0.19 ms | 0 ; 1.7 ; 41,618 (NULL 90, non-finite 33663) ; #1514 ; 0.14 ms | 81 ; 8.9e-16 ; 0 ; 0.14 ms | y y y |
| amazon_random_R30 | 0.001 | 52,173 ; 0 ; 0 ; 0.19 ms | 0 ; 9.5e+139 ; 19,231 (NULL 19, non-finite 0) ; #1514 ; 0.15 ms | 66 ; 1.9e-13 ; 0 ; 0.14 ms | y **n** **n** |
| amazon_random_R30 | 0.02 | 52,173 ; 8.9e-16 ; 0 ; 0.19 ms | 0 ; 1.0e-12 ; 0 ; 0.14 ms | 2 ; 5.4e-13 ; 0 ; 0.14 ms | y y y |
| amazon_random_R30 | 0.1 | 52,173 ; 1.8e-15 ; 0 ; 0.15 ms | 0 ; 8.0e-15 ; 0 ; 0.13 ms | 0 ; 7.8e-15 ; 0 ; 0.13 ms | y y y |
| amazon_random_R30 | 1 | 52,173 ; 1.8e-15 ; 0 ; 0.15 ms | 0 ; 1.1e-14 ; 0 ; 0.12 ms | 0 ; 1.1e-14 ; 0 ; 0.13 ms | y y y |
| amazon_time_R2 | 0.0001 | 49,027 ; 0 ; 0 ; 0.18 ms | 0 ; 1.3 ; 93,224 (NULL 104, non-finite 92540) ; #263 ; 0.14 ms | 1,209 ; 1.3e-13 ; 0 ; 0.14 ms | y y y |
| amazon_time_R2 | 0.001 | 49,027 ; 1.8e-16 ; 0 ; 0.19 ms | 0 ; 2.2e+118 ; 40,471 (NULL 230, non-finite 0) ; #263 ; 0.14 ms | 1,079 ; 2.7e-12 ; 0 ; 0.15 ms | y **n** y |
| amazon_time_R2 | 0.02 | 49,027 ; 0 ; 0 ; 0.17 ms | 0 ; 4.7e-06 ; 465 ; #5636 ; 0.14 ms | 196 ; 2.9e-12 ; 0 ; 0.15 ms | y y y |
| amazon_time_R2 | 0.1 | 49,027 ; 0 ; 0 ; 0.17 ms | 0 ; 8.3e-14 ; 0 ; 0.14 ms | 0 ; 8.3e-14 ; 0 ; 0.14 ms | y y y |
| amazon_time_R2 | 1 | 49,027 ; 7.1e-16 ; 0 ; 0.18 ms | 0 ; 1.1e-14 ; 0 ; 0.14 ms | 0 ; 1.1e-14 ; 0 ; 0.14 ms | y y y |
| amazon_random_R2 | 0.0001 | 49,028 ; 0 ; 0 ; 0.19 ms | 0 ; 2.3 ; 91,499 (NULL 151, non-finite 90077) ; #27 ; 0.14 ms | 1,211 ; 6.1e-13 ; 0 ; 0.16 ms | y y y |
| amazon_random_R2 | 0.001 | 49,028 ; 0 ; 0 ; 0.18 ms | 0 ; 2.1e+210 ; 40,649 (NULL 301, non-finite 0) ; #103 ; 0.14 ms | 1,078 ; 3.4e-12 ; 0 ; 0.15 ms | y **n** **n** |
| amazon_random_R2 | 0.02 | 49,028 ; 0 ; 0 ; 0.18 ms | 0 ; 1.1e-06 ; 686 ; #10488 ; 0.14 ms | 163 ; 3.2e-12 ; 0 ; 0.14 ms | y **n** y |
| amazon_random_R2 | 0.1 | 49,028 ; 8.9e-16 ; 0 ; 0.19 ms | 0 ; 4.5e-14 ; 0 ; 0.14 ms | 0 ; 4.5e-14 ; 0 ; 0.15 ms | y y y |
| amazon_random_R2 | 1 | 49,028 ; 0 ; 0 ; 0.18 ms | 0 ; 2.0e-14 ; 0 ; 0.13 ms | 0 ; 1.9e-14 ; 0 ; 0.15 ms | y y y |

Counts equal to the replay in 68 of 75 (stream, eps, arm) runs.

### 6. Cited, not re-measured

DuckDB 1.5.6 (experiments/duckdb/results.json, IMDb predicate subset, 5 queries, `PARTITION BY facet ORDER BY rid ROWS 49 PRECEDING`, 1 thread). DuckDB's C API has no inverse, so S3 cannot be built there; the shipped softavg runs on the segment tree (S2 = S4 path). Cell: NaN / Inf / NULL / finite-wrong rows ; max abs err / max|v| ; ms.

| plan (rung) | eps=0.0001 | eps=0.001 | eps=0.02 | eps=0.1 | eps=1 |
|---|---|---|---|---|---|
| naive_window (S0) | 1,164,114 / 0 / 0 / 0 ; 8.9e-17 ; 49.5 | 823 / 0 / 0 / 0 ; 4.4e-16 ; 49.7 | 0 / 0 / 0 / 0 ; 1.2e-15 ; 46.0 | 0 / 0 / 0 / 0 ; 7.1e-16 ; 47.6 | 0 / 0 / 0 / 0 ; 7.1e-16 ; 48.3 |
| maxshift_partition (S1) | 1,117,803 / 0 / 0 / 1,313 ; 0.04 ; 55.8 | 3 / 0 / 0 / 6 ; 2.1e-08 ; 60.5 | 0 / 0 / 0 / 0 ; 1.2e-15 ; 54.8 | 0 / 0 / 0 / 0 ; 8.0e-16 ; 56.1 | 0 / 0 / 0 / 0 ; 7.1e-16 ; 58.3 |
| exact_selfjoin (S1x) | 0 / 0 / 0 / 0 ; 3.6e-16 ; 424 | 0 / 0 / 0 / 0 ; 5.3e-16 ; 397 | 0 / 0 / 0 / 0 ; 2.0e-15 ; 399 | 0 / 0 / 0 / 0 ; 1.1e-15 ; 398 | 0 / 0 / 0 / 0 ; 8.9e-16 ; 399 |
| softavg_segtree (S2 = S4) | 0 / 0 / 0 / 0 ; 3.6e-16 ; 88.3 | 0 / 0 / 0 / 0 ; 4.4e-16 ; 69.8 | 0 / 0 / 0 / 0 ; 1.4e-15 ; 68.6 | 0 / 0 / 0 / 0 ; 7.1e-16 ; 68.5 | 0 / 0 / 0 / 0 ; 8.0e-16 ; 69.6 |
| softavg_naive_mode (S2) | 0 / 0 / 0 / 0 ; 3.6e-16 ; 158 | 0 / 0 / 0 / 0 ; 5.3e-16 ; 133 | 0 / 0 / 0 / 0 ; 1.8e-15 ; 129 | 0 / 0 / 0 / 0 ; 9.8e-16 ; 130 | 0 / 0 / 0 / 0 ; 1.1e-15 ; 129 |

SQLite GROUP BY facet, aggregate only (experiments/sqlite/results.json, 200 queries, timing 5 queries): median ms S2 (`softavg_agg`, bare ScalarAcc) vs S4 (`softavg`, InvAcc bookkeeping + 64-row prefix record):

| eps | S0 naive | S1 maxshift_cte | S2 softavg_agg | S4 softavg | S4 / S2 |
|---|---|---|---|---|---|
| 0.0001 | 64.5 | 147 | 57.2 | 60.5 | 1.06 |
| 0.001 | 72.2 | 148 | 56.5 | 57.1 | 1.01 |
| 0.02 | 68.6 | 148 | 55.3 | 56.6 | 1.02 |
| 0.1 | 66.4 | 144 | 57.4 | 56.7 | 0.99 |
| 1 | 64.0 | 148 | 58.8 | 57.8 | 0.98 |

<!-- END GENERATED -->

## Findings (qualitative; every number is in the generated tables)

1. **S3 goes wrong wherever the shipped certificate refolds, and nowhere else
   it measurably matters.** At eps <= 0.02 removal without a certificate is wrong in
   every window cell (every row set, every k, every query) and in the maintenance
   streams: at eps = 1e-4 almost every window row comes back NULL (the stale anchor
   makes every live weight underflow, z = 0, `ScalarAcc::finalize` returns no
   value); at eps = 1e-3 and 0.02 most rows are finite and wrong, by up to many
   orders of magnitude of max|v| (table 1, 2). Nothing in the result says so: S3
   never errors and never returns NULL except when z is exactly 0.
2. **Where the S3 failure did NOT materialise.** At eps = 0.1 and 1 S3 is within
   the PROTOCOL tolerance in every window cell and every maintenance stream, at
   errors of the same order as S4 (tables 1, 2, 5); there the certificate refolds
   only a handful of frames and buys nothing measurable. In maintenance at
   eps = 0.02 S3 stayed within tolerance on the Amazon R = 30 streams (time order and
   random), though with a larger drift than S4; it failed on the maint_ivm stream
   and on the R = 2 streams (table 5). In windows the size of the S3 error shrinks
   with k at eps = 0.02 (orders of magnitude smaller at k = 1000 than at k = 10) but
   stays far above the tolerance (table 2).
3. **S3 degenerates into S1.** S3's anchor is the running maximum since the first
   row and is never lowered, so once the corpus' top-scoring row has entered (the
   score maximum is ~1.0 for every query, reached early) S3 subtracts at a fixed
   anchor: the same arithmetic as S1's global-max shift. The two rungs fail in the
   same cells with the same worst errors (table 1; `trace.json` -> `worst_overall`:
   same worst row and same error for `sqlite_S1` and `sqlite_S3` at eps = 0.02).
   Online softmax with removal is only safe with something that notices when the
   anchor has gone stale; the certificate is that something.
4. **The SQLite engine and the PG call-order replay deliver bit-identical S3 and S4
   values** (`crosscheck.json`: every row of 6 cells, IMDb subset and Amazon, eps from
   1e-4 to 0.1), and S4's refold count in SQLite equals the replay's punt count cell
   by cell (tables 2, 3). This is what licenses reading the PG replay rows as the
   values PostgreSQL would deliver; it says nothing about PG latency.
5. **S4 is correct everywhere measured** (0 wrong rows in every window cell, drift
   at rounding level in every maintenance stream), and **what it costs over S3 is
   small**: in SQLite the interleaved S4 / S3 latency ratio is a few percent (table
   4b; the back-to-back timings of table 4 deviate both ways in single cells under the
   machine's load). The ratio is about the same at eps = 1, where almost nothing
   refolds, so what S4 pays is the per-row bookkeeping (certificate terms and the
   frame shadow SQLite needs), not the refolds. Its refolds
   re-absorb about one frame per k+1 rows at sharp eps (table 3, restart rows), so
   the refold work stays O(rows), not O(rows x k).
6. **S2 is correct and pays O(k) per row.** S2 is exact in every cell, but its
   window cost grows with k (S2 / S4 in tables 4, 4b: about equal at k = 10, a few times
   S4 at k = 1000). In maintenance S2 recomputes every group a delete touches:
   orders of magnitude more groups and refolded rows than S4 (table 5;
   `refold_rows` in results.json). In SQLite S2 cannot even be expressed as an
   aggregate: the extension has to keep the frame and refold it itself.
7. **DuckDB has no S3.** Its C API has no inverse, so its windows always run the
   segment tree (S2 = S4 path, cited table 6). PostgreSQL GROUP BY also has S2 = S4
   (the plain SFUNC is ScalarAcc). In SQLite GROUP BY the S4 bookkeeping over S2
   (`softavg` vs `softavg_agg`) is within noise (cited table 6).

8. **PostgreSQL 18.6 delivers what the replay predicted, with one exception that is a
   bruce_pg defect.** Every S2 / S3 / S4 window cell the server completed has the same
   error summary as the replay, compared exactly (wrong / NULL / Inf counts, max error,
   worst row), and S4's punt count in the server equals the replay's in every cell
   (table 3b). The exception: in one S4 cell the whole query raised
   `softavg: moving-aggregate state is inexact at finalize` (table 3b). PG calls MINVFUNC
   for the leaving row (the certificate still holds, so no punt), then MSFUNC for the new
   row, whose absorb can push the certificate over the tolerance; MFINALFUNC then refuses
   the inexact state. The replay had logged exactly this row on stderr ("S4 inexact after
   absorb") but had kept the value. It is loud, not silent, and SQLite is not affected
   (its xValue refolds the frame shadow); the fix is a bruce_pg TODO
   (`../../notes/INTEGRATION_ISSUES.md` B14), not done here. In maintenance most (stream,
   eps, arm) runs give exactly the replay's counts (table 5b); the others differ in a few
   S3 checks or by 1-2 S4 recomputes because PG folds initial states and recomputes in
   heap order and removes a multi-row delete in `array_agg` order, while the replay folds
   in id order: S3's output after a cancellation is rounding noise of that order (loading
   the initial rows in id order made the first differing run match exactly), and S4 sits
   on its threshold in a few groups. S4 has 0 wrong checks in every run, as in the replay.
9. **PG latency** (table 4c, smoke): S0, S1 and S2 all pay O(k) per row in PostgreSQL
   (`sum(float8)` has no inverse), so at k = 1000 they are two orders of magnitude slower
   than S3 / S4; S0 cannot even finish at sharp eps (PG raises on `exp` overflow). S4 / S3
   in PG is a few percent in most cells but single cells are off by large factors in both
   directions: the 9 harness processes ran concurrently on the shared machine, back to
   back and not interleaved, so table 4c says nothing about the certificate's cost in PG
   (SQLite's interleaved table 4b is the measurement; a PG interleaved re-timing is not done).

## Not run

Full list in `not_run.json` (copied into `results.json` -> `not_run`).

- **PostgreSQL 17.10.** Not rebuilt; the PG engine cells ran on PostgreSQL 18.6 instead
  (2026-11-07, `../../pg18env`), so there is no PG 17 vs PG 18 run of this ladder. Also not
  run in PG: S1x (exact SQL), and GROUP BY (S2 = S4 there). History of the 2026-11-06 state
  (kept as it was written then):
- ~~**Every PostgreSQL engine cell.**~~ (superseded 2026-11-07) `~/.pgrx` (13 GB, including the release
  PostgreSQL 17.10 install `~/.pgrx/17.10/pg-release-install`) was deleted by the
  2026-11-06 19:10 disk cleanup (`~/cleanup_1106.log`). Rebuilding it from the
  official tarball (sha256 verified) was refused by the permission system; not
  worked around. Refused command:

  ```bash
  cd postgresql-17.10 && ./configure '--prefix=/home/jding/.pgrx/17.10/pg-release-install' \
      '--with-pgport=28817' '--without-icu' '--without-readline' && nice -n 10 make -j16 && make install
  ```

  Then, never run: `cargo pgrx init --pg17 <that>/bin/pg_config`; `cargo pgrx install --release`
  of `engines/postgres` (S4) and of `pg_ablation/` (S3 + IVM arms; **source never compiled**);
  a scratch cluster on port 28850 with database `ladder`; `python pg_window.py 28850`;
  `python pg_maint.py 28850` (both harnesses **never executed**). Consequently: no PG latency
  for any rung, no PG S0 / S1 window cells, no per-statement PG maintenance latency; the PG
  rows above are values from the call-order replay only.
- **SQLite S1x** at k = 1000 on IMDb-all and Amazon (O(rows x k), minutes per fetch), and S1x
  for queries 1-2.
- **DuckDB**: no new build (no inverse in its C API, so S3 cannot exist there); cited numbers are
  from a different setup (IMDb subset, 5 queries, k = 49, per facet).
- **GROUP BY**: not re-measured (S3 = S2 in GROUP BY, no removal happens; cited SQLite
  `softavg_agg` vs `softavg`; in PG GROUP BY S2 = S4 by construction).

## Files

| file | role |
|---|---|
| `ladder_core/` | `PlainInv` (S3) over softagg-core's `ScalarAcc`; 3 unit tests |
| `sqlite_ablation/` | SQLite loadable extension: `softavg_s3`, `softavg_s2`, registration probes, counters |
| `replay/` | `ladder_replay`: PG window / softavg_ivm call sequences for S2, S3, S4 |
| `pg_ablation/` | pgrx extension `softavg_ablation` (S3 moving aggregate, S2/S3 IVM arms); compiled for PG 18.6 on 2026-11-07 (added the `pg18` feature and a `Cargo.lock` copied from `engines/postgres`; no source change) |
| `data.py` | row sets + scores -> scratch npz + SQLite db (deleted after the run); `data_meta.json` |
| `common.py` | grid, numpy reference, error metrics, machine state, hashes |
| `sqlite_window.py` -> `sqlite_window.jsonl` | SQLite window cells |
| `sqlite_timing.py` -> `sqlite_timing.jsonl` | interleaved S2/S3/S4 timing (certificate cost) |
| `replay.py` -> `replay_window.jsonl`, `replay_ivm.json` | PG replays |
| `trace.py` -> `trace.json` | per-row trace for the figure |
| `crosscheck.py` -> `crosscheck.json` | SQLite S3/S4 output vs the PG replay, bitwise |
| `pg_window.py` -> `pg_window.jsonl` | PG 18.6 window cells (S0-S4); fixed on first run: VACUUM in its own call; added resume, `--datasets/--ks/--eps`, S3 counters |
| `pg_maint.py` -> `pg_maint.json` | PG 18.6 maintenance arms; rewritten on first run: replay row values on re-insert, id-ordered initial load, per-statement checks of touched groups (the replay's rule), COPY loading, resume, per-stream databases |
| `assemble_pg.py` | adds `postgres_engine` (+ comparison with the replay) to `results.json` without touching the other keys |
| `not_run.json` | what was refused / not run |
| `assemble.py` -> `results.json`; `summarize.py` -> this README's tables | |

## Rerun

```bash
PY=python
export LADDER_WORK=$(mktemp -d)
(cd ladder_core && CARGO_TARGET_DIR=../target cargo test --release)
(cd sqlite_ablation && CARGO_TARGET_DIR=../target cargo build --release)
(cd replay && CARGO_TARGET_DIR=../target cargo build --release)
(cd ../../engines/sqlite && CARGO_TARGET_DIR=$PWD/../../experiments/softmax_ladder/target/shipped_sqlite cargo build --release)
$PY data.py
$PY sqlite_window.py
$PY sqlite_window.py --rungs S1x --queries 1 --ks 10 100
$PY sqlite_window.py --rungs S1x --queries 1 --ks 1000 --datasets imdb_ge
$PY sqlite_timing.py
$PY replay.py window && $PY replay.py ivm && $PY trace.py && $PY crosscheck.py
$PY assemble.py && $PY summarize.py
rm -rf $LADDER_WORK target
```

PostgreSQL 18.6 part (2026-11-07; extensions staged as in `../../pg18env/README.md`):

```bash
PY=python
export LADDER_WORK=$(mktemp -d) && $PY data.py            # same npz as above (data_meta.json unchanged)
../../pg18env/start.sh && createdb -h 127.0.0.1 -p 28860 -U jding ladder
$PY pg_window.py --load-only
# window cells, split into processes (each resumes from its own file):
$PY pg_window.py --datasets imdb_ge --out pg_window_p4.jsonl &   # ... imdb_all, amazon --ks 10 100, amazon --ks 1000 --eps <e>
# maintenance, one process / database per stream:
for s in maint_ivm_10k amazon_time_R30 amazon_random_R30 amazon_time_R2 amazon_random_R2; do
  $PY pg_maint.py --streams $s --db lm_${s//_/} --out pg_maint_$s.json & done; wait
# assemble_pg.py concatenates pg_window_p*.jsonl -> pg_window.jsonl and merges pg_maint_*.json -> pg_maint.json
$PY assemble_pg.py && $PY summarize.py                   # server must be up
../../pg18env/stop.sh; rm -rf $LADDER_WORK
```

# bound: how loose is InvAcc's error bound, what does tightening cost, and an ESS diagnostic

Answers the questions ("analyze the factors behind the slack, can it be
tightened") and ("report worst-case guarantee and empirical conservatism
separately; quantify the cost of statistics needed to tighten"; "how do users pick
eps") for **our own** guarantee: softagg-core's `InvAcc` (engines/core, unmodified),
which removes rows without re-anchoring, carries a running first-order bound
`value_error_bound()` and goes inexact when it exceeds
`max(ABS_TOL = 1e-10, REFOLD_GAIN = 100 * n * U) * v_scale`.

Everything here drives softagg-core directly from Rust (no host engine: this is the
arithmetic all three extensions link). **Timings are smoke numbers, machine shared**
(other agents were compiling/running; see `machine_*` in results.json).

## Files

| file | what |
|---|---|
| `prep.py` | admitted rows (PROTOCOL predicate) in item order, value, facet code, `s = float64(E) @ q` for the first 20 queries of the PROTOCOL draw -> `work/<corpus>/` (deleted after the run) |
| `rust/src/exact.rs`, `dd.rs` | exact reference: max-anchored states in double-double (~106 bits; exponent argument `(mu_a - mu)/eps` also in double-double) in a segment tree, so every step of every scenario gets an exact value in O(log n) |
| `rust/src/local.rs` | local copy of `InvAcc` with runtime `abs_tol`/`gain` and the bound split into its rounding components; `tests/props.rs` + a lockstep check in every run prove it bit-identical to the core (0 mismatches) |
| `rust/src/ddacc.rs` | `DdInv`: compensated (double-double) prototype with its own bound (see the proposal in `../../notes/CORE_CHANGES.md`) |
| `rust/src/ess.rs` | `EssAcc`: mergeable ESS accumulator `(mu, z = sum w, q = sum w^2)`; `SoftEss` = ScalarAcc + q, for timing |
| `rust/src/main.rs` | `run` (scenarios, all variants), `bench` (ns/row), `argpatch` (B1 patch vs shipped), `viol` (prints violating states of one sliding run) |
| `run.py` | build (release) -> prep -> run -> bench -> results.json, deletes work/; `run.py bench` re-times only |
| `argpatch.py` | shipped bound vs the B1 soundness patch on all three corpora -> results.json `argpatch` |
| `report.py` | figures/ and the tables below, from results.json only |

Reproduce: `python3 run.py 20 10 5 && python3 argpatch.py && python3 run.py bench && python3 report.py`
(about 1.5 h on 10 threads of a shared box); tests:
`cargo test --release --manifest-path rust/Cargo.toml`.

## Setup

* **Corpora**: IMDb (`filter_num >= 2000`, 232,828 rows), Amazon (`>= 2015`,
  1,349,709), StackExchange (`>= 3`, 499,708; values are answer scores in
  [-515, 8387], so `v_scale` is far above typical |v|). 20 queries each; eps in
  {1e-4, 1e-3, 0.02, 0.1, 1}.
* **Scenarios** (the host refolds = reset + absorb the live rows, exactly when
  `!is_exact()`):
  (a) sliding windows of 50 / 400 / 5000 rows over all admitted rows in item order;
  (b) IVM churn: every facet with >= 100 admitted rows is one group, half its rows
  absorbed, then 10,000 random insert/delete toggles;
  (c) adversarial: the 10 largest facets (first 2,000 rows each) absorbed, then
  deleted in descending score order to empty.
* **Reference**: the exact value (double-double) at every step; additionally a
  fresh two-pass f64 fold (what a host refold would compute) on a sample of steps
  (<= 2e7 folded rows per group).
* **Ratio** = |delivered - exact| / `value_error_bound()` over every state the
  core reports exact (after the host's refolds). A ratio > 1 is a violation.
* **Oracle**: same arithmetic, refold only when the *true* error exceeds the
  tolerance: the fewest refolds any bound on this arithmetic could achieve.

## Results

All numbers below are in the generated tables (T0-T7) and in `results.json`; the proposal
built on them is the "[Bound agent, 2026-11-06]" entry of `../../notes/CORE_CHANGES.md`.

1. **Worst case vs typical (R4 D1).** Over 833.7M reported states (T0): the worst states reach
   0.13-0.90 of the bound (the first-order bound is nearly attained), while the median state is
   ~100-25,000x conservative (median ratio 4e-5 to 1.3e-2 at eps >= 1e-3; p99 2e-3 to 1.2e-1).
   At eps = 1e-4 one row carries the weight and the value is exact (median ratio <= 1e-26).
2. **Soundness gap (fixed by a 1-FMA patch).** 45,825 states (all StackExchange, eps <= 1e-3)
   exceed the shipped bound, by up to 25x or with a bound of 0, but by at most 7.4e-26 * max|v|
   absolute: the bound does not charge the rounding of the exp argument (|x| U relative per
   weight) nor underflowed weights. With both terms (T0b) there are 0 violations at +0.02%
   refolds. Reproduced in `tests/props.rs` (two-row states, up to 166x).
3. **What the bound is made of (R1 W3/D2).** 95-100% is the running-sum rounding `U|z1|`,
   `U|u1|` charged on every absorb/remove, split evenly between `err_u` and `|F| err_z` (T3).
   It grows linearly with the operations since the last fresh fold, the true error only like
   a random walk: the ratio falls as about ops^-1/2 (`figures/slack_vs_ops.pdf`), and at mild
   eps this drift alone causes 100-270 refolds per 20-query sliding scenario that no accuracy
   requires (oracle: 0).
4. **Refolds at sharp eps are forced, not caused by slack** (T0, T4). At eps = 1e-4, 93-95% of
   window/churn refolds have an infinite bound (z cancelled below its own error: stale anchor),
   and the oracle still needs 95-97% of them; 75-86% at eps = 1e-3. Headroom exists only near
   eps = 0.02 (oracle 8-25% of shipped).
5. **Tolerance (T5, `figures/tradeoff.pdf`).** ABS_TOL = 1e-10 is a sensible middle point:
   1e-11 costs 1.6-2.8x refold work for 4-9x less error, 1e-9 halves refolds at eps = 0.02 for
   10x more error and saves ~10% where refolds are forced. REFOLD_GAIN only matters above
   9,000-row groups. Max error at the default: 2.5e-11 * max|v|.
6. **Tighter arithmetic (DdInv, T4-T6).** A double-double state whose remove cancels exactly
   cuts refolds 4.2x at eps = 0.02 (below the oracle for the shipped arithmetic), removes the
   drift refolds, and lowers the error to ~1e-16 * max|v|, with 0 violations; it costs ~1.5x
   per absorb, ~1.7x per sliding row and 96 vs 64 bytes. The refold work it saves is
   <= 0.14 rows/step outside descending deletes, so it does not pay for itself in throughput;
   it is a precision option, not a default. (At eps = 1e-4 it is faster only because it skips
   sub-normal weights, which InvAcc could do alone: DuckDB agent's P2.)
7. **f64 refolds are not a reference** for large groups: against a fresh two-pass f64 fold the
   state "violates" its bound up to 14x on 10^4-10^5-row IVM groups (T2), against the exact
   value it does not (outside item 2); the fold's own error reaches 3.1e-14 * v_scale.
8. **ESS (R4 D2, T6/T7, `figures/ess.pdf`).** `EssAcc` (24 bytes) merges exactly like
   ScalarAcc (merge == sequential to <= 1.1e-12 on the corpora, 1e-12 in tests) and adds no
   measurable per-row cost (0.95-1.13x ScalarAcc alone, 0.85-1.06x folded into softavg's
   state). ESS per facet group: ~1 for eps <= 1e-3 (>= 97% of groups), median 2.6-5.0 at
   0.02, 584-16,400 at 0.1, ESS/n = 0.99 at 1. Proposed as a separate `softavg_ess(s, eps)`.

Caveats: timings are smoke numbers on a heavily shared machine (cpu ~90% busy, see T6) and
use the minimum of 9 repetitions; sweeps and non-default DdInv configs use the first 5
queries; the double-double reference is exact to < 2^-1074 * |v| per row in the sub-normal
range; the DdInv numbers in T0/T4/T5 come from v1 (sub-normal weights computed), the bench
and the proposal use v2 (skipped), which differs only at eps = 1e-4 by <= 1% in refolds
(`results.json -> ddinv_v2_check_imdb`). `arg_term` (the B1 patch) was added to
`local.rs` after the main run; it defaults to off and the local copy stays bit-identical to
the core (`tests/props.rs`). Source hashes: `experiment_src_sha256_16` (top level) is the
code of the main run; `bench`, `argpatch` and `ddinv_v2_check_imdb` carry the hash of the code
that produced them (later edits: arg_term, DdInv v2, bench min-of-9, the `argpatch`/`viol`
subcommands; none changes the shipped-InvAcc path).

<!-- TABLES -->
### T0. Headline, pooled over IMDb + Amazon + StackExchange and all queries

ratio = true error / bound (exact reference). Refolds: shipped bound / oracle (refold only when the true error exceeds the tolerance) / DdInv prototype. `rows/step` = refolded rows per step (host refold work).

| scenario | eps | states | violations | ratio median | p90 | p99 | max | refolds shipped | (bound infinite) | oracle | DdInv | rows/step shipped | rows/step DdInv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sliding w=50 | 0.0001 | 41,644,900 | 31836 | <=1e-26 | 8.7e-11 | 3.3e-2 | inf | 816,865 | 772,020 | 794,197 | 753,785 | 0.981 | 0.905 |
| sliding w=50 | 0.001 | 41,644,900 | 1690 | 1.2e-3 | 2.5e-2 | 1.2e-1 | 2.31e+1 | 676,129 | 382,044 | 581,031 | 409,668 | 0.812 | 0.492 |
| sliding w=50 | 0.02 | 41,644,900 | 0 | 1.3e-2 | 4.2e-2 | 9.1e-2 | 8.00e-1 | 45,599 | 9 | 11,268 | 10,893 | 0.0547 | 0.0131 |
| sliding w=50 | 0.1 | 41,644,900 | 0 | 5.2e-4 | 1.7e-3 | 6.0e-3 | 3.65e-1 | 269 | 0 | 0 | 127 | 0.000323 | 0.000152 |
| sliding w=50 | 1 | 41,644,900 | 0 | 2.1e-4 | 6.6e-4 | 2.1e-3 | 3.15e-1 | 108 | 0 | 0 | 41 | 0.00013 | 4.92e-05 |
| sliding w=400 | 0.0001 | 41,644,900 | 1369 | <=1e-26 | 2.6e-7 | 5.2e-3 | 1.32e+1 | 104,947 | 98,202 | 100,953 | 95,463 | 1.01 | 0.917 |
| sliding w=400 | 0.001 | 41,644,900 | 42 | 3.6e-4 | 3.8e-3 | 2.3e-2 | 8.70e+0 | 87,448 | 45,385 | 70,985 | 49,675 | 0.84 | 0.477 |
| sliding w=400 | 0.02 | 41,644,900 | 0 | 4.8e-3 | 1.5e-2 | 3.3e-2 | 6.31e-1 | 7,773 | 3 | 1,374 | 1,075 | 0.0747 | 0.0103 |
| sliding w=400 | 0.1 | 41,644,900 | 0 | 3.3e-4 | 1.1e-3 | 3.6e-3 | 3.65e-1 | 157 | 0 | 0 | 13 | 0.00151 | 0.000125 |
| sliding w=400 | 1 | 41,644,900 | 0 | 1.6e-4 | 5.2e-4 | 2.6e-3 | 3.15e-1 | 100 | 0 | 0 | 0 | 0.000961 | 0 |
| sliding w=5000 | 0.0001 | 41,644,900 | 67 | <=1e-26 | 4.0e-6 | 5.2e-4 | 3.31e+0 | 9,121 | 8,516 | 8,708 | 8,248 | 1.1 | 0.99 |
| sliding w=5000 | 0.001 | 41,644,900 | 2 | 4.0e-5 | 3.5e-4 | 2.3e-3 | 1.08e+0 | 7,931 | 3,919 | 5,988 | 4,225 | 0.952 | 0.507 |
| sliding w=5000 | 0.02 | 41,644,900 | 0 | 1.7e-3 | 5.0e-3 | 1.1e-2 | 6.31e-1 | 1,316 | 2 | 234 | 147 | 0.158 | 0.0176 |
| sliding w=5000 | 0.1 | 41,644,900 | 0 | 3.2e-4 | 1.0e-3 | 4.2e-3 | 3.65e-1 | 128 | 0 | 0 | 0 | 0.0154 | 0 |
| sliding w=5000 | 1 | 41,644,900 | 0 | 1.8e-4 | 7.6e-4 | 3.6e-3 | 3.15e-1 | 100 | 0 | 0 | 0 | 0.012 | 0 |
| IVM churn | 0.0001 | 40,604,060 | 10699 | <=1e-26 | 1.0e-4 | 1.2e-2 | 2.54e+1 | 48,483 | 44,928 | 46,646 | 43,068 | 0.495 | 0.44 |
| IVM churn | 0.001 | 40,604,060 | 114 | 1.4e-4 | 3.8e-3 | 3.0e-2 | 3.93e+0 | 36,399 | 16,491 | 28,774 | 15,455 | 0.386 | 0.146 |
| IVM churn | 0.02 | 40,604,060 | 0 | 2.2e-3 | 6.9e-3 | 2.0e-2 | 7.18e-1 | 386 | 0 | 32 | 1 | 0.00857 | 2.07e-06 |
| IVM churn | 0.1 | 40,604,060 | 0 | 1.4e-3 | 4.6e-3 | 1.3e-2 | 1.77e-1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IVM churn | 1 | 40,604,060 | 0 | 1.4e-3 | 4.8e-3 | 1.3e-2 | 1.28e-1 | 0 | 0 | 0 | 0 | 0 | 0 |
| descending deletes | 0.0001 | 1,200,000 | 6 | 1.8e-3 | 9.5e-3 | 5.0e-2 | 5.93e+0 | 214,875 | 12,513 | 137,173 | 114,519 | 199 | 102 |
| descending deletes | 0.001 | 1,200,000 | 0 | 4.4e-3 | 1.7e-2 | 5.8e-2 | 7.23e-1 | 33,713 | 635 | 20,572 | 20,691 | 33.5 | 18.7 |
| descending deletes | 0.02 | 1,200,000 | 0 | 6.3e-3 | 1.7e-2 | 3.3e-2 | 3.88e-1 | 1,940 | 0 | 1,073 | 1,162 | 1.3 | 0.509 |
| descending deletes | 0.1 | 1,200,000 | 0 | 4.2e-3 | 1.1e-2 | 2.2e-2 | 3.46e-1 | 524 | 0 | 50 | 279 | 0.0151 | 0.000411 |
| descending deletes | 1 | 1,200,000 | 0 | 4.2e-3 | 1.0e-2 | 1.7e-2 | 2.52e-1 | 406 | 0 | 0 | 0 | 0.00242 | 0 |

Total reported states: 833,693,800; violations of the shipped bound against the exact value: 45825; core-vs-local-copy mismatches: 0; DdInv bound violations (all configs): 0.

### T0b. Shipped bound vs the proposed soundness patch (same tolerance, exact reference)

Patch = also charge 2U|x| per weight (rounding of the exp argument x = (s - mu)/eps) and 2^-1074 per underflowed weight/product. Only cells where either variant has a violation or the refold counts differ are listed; `max viol err` is the largest |error| / max|v| among violating states.

| corpus (queries) | scenario | eps | refolds shipped | refolds patched | violations shipped | max viol err shipped | violations patched |
|---|---|---|---|---|---|---|---|
| IMDb (20) | sliding w=50 | 0.001 | 81,209 | 81,217 | 0 | 0 | 0 |
| IMDb (20) | sliding w=50 | 0.02 | 8,479 | 8,520 | 0 | 0 | 0 |
| IMDb (20) | sliding w=50 | 0.1 | 22 | 23 | 0 | 0 | 0 |
| IMDb (20) | sliding w=400 | 0.02 | 1,584 | 1,587 | 0 | 0 | 0 |
| IMDb (20) | descending deletes | 0.0001 | 77,414 | 77,453 | 0 | 0 | 0 |
| IMDb (20) | descending deletes | 0.001 | 12,452 | 12,470 | 0 | 0 | 0 |
| Amazon (5) | sliding w=50 | 0.001 | 113,958 | 113,962 | 0 | 0 | 0 |
| Amazon (5) | sliding w=50 | 0.02 | 11,577 | 11,641 | 0 | 0 | 0 |
| Amazon (5) | sliding w=50 | 0.1 | 70 | 82 | 0 | 0 | 0 |
| Amazon (5) | sliding w=50 | 1 | 29 | 30 | 0 | 0 | 0 |
| Amazon (5) | sliding w=400 | 0.02 | 1,655 | 1,659 | 0 | 0 | 0 |
| Amazon (5) | sliding w=400 | 0.1 | 40 | 41 | 0 | 0 | 0 |
| Amazon (5) | descending deletes | 0.0001 | 23,342 | 23,353 | 0 | 0 | 0 |
| Amazon (5) | descending deletes | 0.001 | 3,801 | 3,805 | 0 | 0 | 0 |
| StackExchange (20) | sliding w=50 | 0.0001 | 191,280 | 191,280 | 31,836 | 7.4e-26 | 0 |
| StackExchange (20) | sliding w=50 | 0.001 | 147,640 | 147,643 | 1,690 | 7.0e-26 | 0 |
| StackExchange (20) | sliding w=50 | 0.02 | 2,340 | 2,366 | 0 | 0 | 0 |
| StackExchange (20) | sliding w=400 | 0.0001 | 25,131 | 25,131 | 1,369 | <=1e-26 | 0 |
| StackExchange (20) | sliding w=400 | 0.001 | 19,199 | 19,199 | 42 | 1.8e-26 | 0 |
| StackExchange (20) | sliding w=5000 | 0.0001 | 2,493 | 2,493 | 67 | <=1e-26 | 0 |
| StackExchange (20) | sliding w=5000 | 0.001 | 2,012 | 2,012 | 2 | <=1e-26 | 0 |
| StackExchange (20) | IVM churn | 0.0001 | 44,492 | 44,492 | 10,699 | <=1e-26 | 0 |
| StackExchange (20) | IVM churn | 0.001 | 33,042 | 33,043 | 114 | 4.1e-26 | 0 |
| StackExchange (20) | descending deletes | 0.0001 | 53,675 | 53,708 | 6 | <=1e-26 | 0 |
| StackExchange (20) | descending deletes | 0.001 | 7,897 | 7,906 | 0 | 0 | 0 |

Total refolds, all listed corpora/cells: shipped 1,162,725, patched 1,163,008.

Queries per corpus: 20 (sweeps and DdInv variants other than the default: first 5). Machine at start: `{"quiet": false, "require_quiet_message": "[quiet] REFUSING to record timings: cpu 34.5% busy (limit 15%), gpu 0% (limit 20%).", "cpu_busy_fraction": 0.688, "load_average_1min": 19.45, "timestamp": "2026-11-06T14:40:17"}`.

### T1. true error / bound, shipped InvAcc, exact (double-double) reference, every reported state

**IMDb** (232,828 admitted rows, max|v| = 10)

| scenario | eps | states | median | p90 | p99 | max | violations | bound/v_scale median | err/v_scale max |
|---|---|---|---|---|---|---|---|---|---|
| sliding w=50 | 0.0001 | 4,656,560 | <=1e-26 | 1.3e-5 | 2.9e-2 | 7.79e-1 | 0 | 6.6e-15 | 1.5e-12 |
| sliding w=50 | 0.001 | 4,656,560 | 2.6e-3 | 2.4e-2 | 1.1e-1 | 8.97e-1 | 0 | 6.9e-15 | 4.5e-12 |
| sliding w=50 | 0.02 | 4,656,560 | 1.6e-2 | 4.8e-2 | 1.0e-1 | 6.90e-1 | 0 | 2.1e-13 | 8.9e-12 |
| sliding w=50 | 0.1 | 4,656,560 | 5.5e-4 | 2.1e-3 | 1.0e-2 | 3.65e-1 | 0 | 2.0e-11 | 3.6e-13 |
| sliding w=50 | 1 | 4,656,560 | 1.7e-4 | 5.0e-4 | 2.2e-3 | 2.59e-1 | 0 | 3.2e-11 | 3.4e-14 |
| sliding w=400 | 0.0001 | 4,656,560 | <=1e-26 | 3.6e-5 | 4.2e-3 | 4.41e-1 | 0 | 4.8e-14 | 1.3e-13 |
| sliding w=400 | 0.001 | 4,656,560 | 4.6e-4 | 3.5e-3 | 2.3e-2 | 5.87e-1 | 0 | 5.0e-14 | 3.3e-13 |
| sliding w=400 | 0.02 | 4,656,560 | 6.0e-3 | 1.8e-2 | 4.4e-2 | 5.96e-1 | 0 | 4.8e-13 | 3.0e-12 |
| sliding w=400 | 0.1 | 4,656,560 | 4.0e-4 | 1.1e-3 | 5.8e-3 | 3.65e-1 | 0 | 2.8e-11 | 1.0e-13 |
| sliding w=400 | 1 | 4,656,560 | 1.2e-4 | 4.6e-4 | 3.3e-3 | 2.59e-1 | 0 | 2.9e-11 | 2.3e-14 |
| sliding w=5000 | 0.0001 | 4,656,560 | <=1e-26 | 1.8e-5 | 4.2e-4 | 4.21e-1 | 0 | 5.5e-13 | 1.4e-15 |
| sliding w=5000 | 0.001 | 4,656,560 | 4.0e-5 | 3.2e-4 | 2.4e-3 | 5.87e-1 | 0 | 5.5e-13 | 3.2e-14 |
| sliding w=5000 | 0.02 | 4,656,560 | 1.8e-3 | 5.8e-3 | 1.5e-2 | 5.96e-1 | 0 | 1.8e-12 | 6.1e-13 |
| sliding w=5000 | 0.1 | 4,656,560 | 3.0e-4 | 1.1e-3 | 6.9e-3 | 3.65e-1 | 0 | 2.9e-11 | 9.7e-14 |
| sliding w=5000 | 1 | 4,656,560 | 2.5e-4 | 1.1e-3 | 6.9e-3 | 2.59e-1 | 0 | 2.9e-11 | 4.1e-14 |
| IVM churn | 0.0001 | 5,200,520 | <=1e-26 | 3.2e-5 | 2.4e-3 | 8.15e-1 | 0 | 1.4e-13 | 3.2e-13 |
| IVM churn | 0.001 | 5,200,520 | 7.9e-5 | 1.9e-3 | 1.4e-2 | 7.93e-1 | 0 | 1.7e-13 | 4.6e-12 |
| IVM churn | 0.02 | 5,200,520 | 2.5e-3 | 8.3e-3 | 2.6e-2 | 5.20e-1 | 0 | 7.6e-13 | 2.5e-12 |
| IVM churn | 0.1 | 5,200,520 | 1.6e-3 | 5.0e-3 | 1.4e-2 | 5.75e-2 | 0 | 8.7e-13 | 1.8e-14 |
| IVM churn | 1 | 5,200,520 | 1.5e-3 | 4.8e-3 | 1.3e-2 | 6.75e-2 | 0 | 8.7e-13 | 1.7e-14 |
| descending deletes | 0.0001 | 400,000 | 1.8e-3 | 9.1e-3 | 4.6e-2 | 6.72e-1 | 0 | 1.0e-12 | 1.6e-11 |
| descending deletes | 0.001 | 400,000 | 4.4e-3 | 1.7e-2 | 5.5e-2 | 6.77e-1 | 0 | 2.3e-12 | 1.7e-11 |
| descending deletes | 0.02 | 400,000 | 5.8e-3 | 1.5e-2 | 2.8e-2 | 3.44e-1 | 0 | 3.3e-12 | 6.0e-12 |
| descending deletes | 0.1 | 400,000 | 4.4e-3 | 1.0e-2 | 2.2e-2 | 1.51e-1 | 0 | 1.0e-12 | 1.7e-12 |
| descending deletes | 1 | 400,000 | 4.2e-3 | 1.1e-2 | 1.6e-2 | 2.23e-1 | 0 | 5.2e-13 | 1.3e-12 |

**Amazon** (1,349,709 admitted rows, max|v| = 5)

| scenario | eps | states | median | p90 | p99 | max | violations | bound/v_scale median | err/v_scale max |
|---|---|---|---|---|---|---|---|---|---|
| sliding w=50 | 0.0001 | 26,994,180 | <=1e-26 | 5.2e-19 | 2.5e-2 | 7.76e-1 | 0 | 8.7e-15 | 4.1e-12 |
| sliding w=50 | 0.001 | 26,994,180 | 3.0e-7 | 2.3e-2 | 1.0e-1 | 8.86e-1 | 0 | 9.1e-15 | 2.3e-11 |
| sliding w=50 | 0.02 | 26,994,180 | 1.4e-2 | 4.2e-2 | 9.5e-2 | 8.00e-1 | 0 | 5.8e-13 | 1.1e-11 |
| sliding w=50 | 0.1 | 26,994,180 | 4.8e-4 | 1.7e-3 | 5.5e-3 | 3.43e-1 | 0 | 1.9e-11 | 5.9e-13 |
| sliding w=50 | 1 | 26,994,180 | 2.0e-4 | 6.3e-4 | 2.1e-3 | 3.15e-1 | 0 | 4.6e-11 | 7.5e-14 |
| sliding w=400 | 0.0001 | 26,994,180 | <=1e-26 | 5.8e-18 | 3.6e-3 | 5.92e-1 | 0 | 6.6e-14 | 1.7e-13 |
| sliding w=400 | 0.001 | 26,994,180 | 8.3e-5 | 3.5e-3 | 2.0e-2 | 6.70e-1 | 0 | 6.9e-14 | 6.8e-13 |
| sliding w=400 | 0.02 | 26,994,180 | 4.8e-3 | 1.5e-2 | 3.3e-2 | 6.31e-1 | 0 | 1.7e-12 | 5.1e-12 |
| sliding w=400 | 0.1 | 26,994,180 | 2.8e-4 | 9.5e-4 | 3.6e-3 | 3.43e-1 | 0 | 2.6e-11 | 1.4e-13 |
| sliding w=400 | 1 | 26,994,180 | 1.5e-4 | 5.2e-4 | 2.8e-3 | 3.15e-1 | 0 | 4.6e-11 | 5.2e-14 |
| sliding w=5000 | 0.0001 | 26,994,180 | <=1e-26 | 5.2e-14 | 3.5e-4 | 4.63e-1 | 0 | 7.9e-13 | 8.2e-15 |
| sliding w=5000 | 0.001 | 26,994,180 | 2.2e-5 | 3.2e-4 | 1.9e-3 | 6.70e-1 | 0 | 8.7e-13 | 3.8e-14 |
| sliding w=5000 | 0.02 | 26,994,180 | 1.5e-3 | 4.6e-3 | 1.1e-2 | 6.31e-1 | 0 | 4.6e-12 | 1.1e-12 |
| sliding w=5000 | 0.1 | 26,994,180 | 2.9e-4 | 1.1e-3 | 3.8e-3 | 3.43e-1 | 0 | 3.2e-11 | 1.7e-13 |
| sliding w=5000 | 1 | 26,994,180 | 1.7e-4 | 6.6e-4 | 3.2e-3 | 3.15e-1 | 0 | 5.0e-11 | 5.5e-14 |
| IVM churn | 0.0001 | 6,600,660 | <=1e-26 | 4.8e-12 | 1.0e-4 | 5.33e-1 | 0 | 2.4e-12 | 2.4e-16 |
| IVM churn | 0.001 | 6,600,660 | 5.5e-6 | 9.5e-5 | 6.6e-4 | 7.16e-1 | 0 | 2.5e-12 | 1.8e-14 |
| IVM churn | 0.02 | 6,600,660 | 1.6e-3 | 4.2e-3 | 7.9e-3 | 1.07e-1 | 0 | 3.0e-12 | 3.7e-13 |
| IVM churn | 0.1 | 6,600,660 | 1.4e-3 | 3.5e-3 | 5.8e-3 | 2.11e-2 | 0 | 3.2e-12 | 2.2e-14 |
| IVM churn | 1 | 6,600,660 | 1.4e-3 | 3.5e-3 | 6.3e-3 | 2.08e-2 | 0 | 3.0e-12 | 3.0e-14 |
| descending deletes | 0.0001 | 400,000 | 1.7e-3 | 9.1e-3 | 5.0e-2 | 6.46e-1 | 0 | 1.2e-12 | 2.5e-11 |
| descending deletes | 0.001 | 400,000 | 4.2e-3 | 1.7e-2 | 6.3e-2 | 6.97e-1 | 0 | 2.8e-12 | 2.3e-11 |
| descending deletes | 0.02 | 400,000 | 6.6e-3 | 1.9e-2 | 4.4e-2 | 3.40e-1 | 0 | 3.8e-12 | 5.7e-12 |
| descending deletes | 0.1 | 400,000 | 4.2e-3 | 1.2e-2 | 2.5e-2 | 1.27e-1 | 0 | 1.4e-12 | 1.9e-12 |
| descending deletes | 1 | 400,000 | 4.4e-3 | 1.1e-2 | 1.9e-2 | 2.52e-1 | 0 | 7.2e-13 | 1.2e-12 |

**StackExchange** (499,708 admitted rows, max|v| = 8387)

| scenario | eps | states | median | p90 | p99 | max | violations | bound/v_scale median | err/v_scale max |
|---|---|---|---|---|---|---|---|---|---|
| sliding w=50 | 0.0001 | 9,994,160 | <=1e-26 | 3.2e-3 | 1.1e-1 | inf | 31836 | 7.9e-16 | 2.0e-12 |
| sliding w=50 | 0.001 | 9,994,160 | 5.2e-3 | 3.0e-2 | 1.4e-1 | 2.31e+1 | 1690 | 9.1e-16 | 5.2e-12 |
| sliding w=50 | 0.02 | 9,994,160 | 9.5e-3 | 3.3e-2 | 7.2e-2 | 7.06e-1 | 0 | 3.0e-13 | 1.2e-11 |
| sliding w=50 | 0.1 | 9,994,160 | 5.8e-4 | 1.9e-3 | 5.8e-3 | 2.91e-1 | 0 | 2.8e-13 | 3.2e-14 |
| sliding w=50 | 1 | 9,994,160 | 2.4e-4 | 7.6e-4 | 1.8e-3 | 2.22e-1 | 0 | 2.4e-13 | 2.4e-15 |
| sliding w=400 | 0.0001 | 9,994,160 | <=1e-26 | 6.6e-4 | 1.8e-2 | 1.32e+1 | 1369 | 2.5e-15 | 1.5e-13 |
| sliding w=400 | 0.001 | 9,994,160 | 8.7e-4 | 4.6e-3 | 2.9e-2 | 8.70e+0 | 42 | 3.0e-15 | 7.4e-13 |
| sliding w=400 | 0.02 | 9,994,160 | 4.0e-3 | 1.3e-2 | 2.8e-2 | 5.56e-1 | 0 | 3.5e-13 | 3.2e-12 |
| sliding w=400 | 0.1 | 9,994,160 | 4.4e-4 | 1.3e-3 | 3.2e-3 | 2.91e-1 | 0 | 2.6e-13 | 9.3e-15 |
| sliding w=400 | 1 | 9,994,160 | 1.8e-4 | 5.8e-4 | 2.2e-3 | 2.22e-1 | 0 | 2.3e-13 | 1.3e-15 |
| sliding w=5000 | 0.0001 | 9,994,160 | <=1e-26 | 6.3e-5 | 2.3e-3 | 3.31e+0 | 67 | 9.5e-15 | 9.6e-15 |
| sliding w=5000 | 0.001 | 9,994,160 | 7.6e-5 | 4.4e-4 | 3.5e-3 | 1.08e+0 | 2 | 1.1e-14 | 7.5e-14 |
| sliding w=5000 | 0.02 | 9,994,160 | 1.8e-3 | 5.2e-3 | 1.0e-2 | 5.56e-1 | 0 | 2.4e-13 | 9.7e-13 |
| sliding w=5000 | 0.1 | 9,994,160 | 3.6e-4 | 1.0e-3 | 4.2e-3 | 2.91e-1 | 0 | 2.5e-13 | 1.5e-15 |
| sliding w=5000 | 1 | 9,994,160 | 2.1e-4 | 9.1e-4 | 3.6e-3 | 2.22e-1 | 0 | 2.3e-13 | 9.9e-16 |
| IVM churn | 0.0001 | 28,802,880 | <=1e-26 | 2.6e-4 | 2.1e-2 | 2.54e+1 | 10699 | 7.6e-15 | 1.1e-11 |
| IVM churn | 0.001 | 28,802,880 | 3.2e-4 | 5.0e-3 | 3.8e-2 | 3.93e+0 | 114 | 1.0e-14 | 9.9e-12 |
| IVM churn | 0.02 | 28,802,880 | 2.3e-3 | 7.2e-3 | 2.0e-2 | 7.18e-1 | 0 | 8.7e-14 | 4.0e-12 |
| IVM churn | 0.1 | 28,802,880 | 1.4e-3 | 5.0e-3 | 1.4e-2 | 1.77e-1 | 0 | 9.1e-14 | 3.7e-15 |
| IVM churn | 1 | 28,802,880 | 1.4e-3 | 5.0e-3 | 1.4e-2 | 1.28e-1 | 0 | 9.1e-14 | 3.5e-15 |
| descending deletes | 0.0001 | 400,000 | 2.0e-3 | 1.0e-2 | 5.2e-2 | 5.93e+0 | 6 | 2.3e-13 | inf |
| descending deletes | 0.001 | 400,000 | 4.6e-3 | 1.8e-2 | 5.8e-2 | 7.23e-1 | 0 | 5.2e-13 | inf |
| descending deletes | 0.02 | 400,000 | 6.3e-3 | 1.5e-2 | 3.0e-2 | 3.88e-1 | 0 | 9.5e-13 | inf |
| descending deletes | 0.1 | 400,000 | 4.2e-3 | 1.1e-2 | 1.5e-2 | 3.46e-1 | 0 | 3.5e-14 | 2.2e-12 |
| descending deletes | 1 | 400,000 | 4.0e-3 | 1.0e-2 | 1.7e-2 | 2.48e-1 | 0 | 2.0e-14 | 8.4e-13 |

### T2. Against a fresh two-pass f64 fold instead (sampled states)

A violation here means the state and the f64 fold differ by more than the state's bound; the last column is the f64 fold's own error against the exact value.

| corpus | scenario | eps | sampled | ratio p99 | ratio max | violations vs f64 fold | f64 fold err / v_scale max |
|---|---|---|---|---|---|---|---|
| IMDb | sliding w=50 | 0.001 | 4,656,560 | 4.2e-2 | 1.08e+0 | 1 | 1.1e-15 |
| IMDb | IVM churn | 0.0001 | 3,859,580 | 2.4e-4 | 1.02e+0 | 1 | 2.7e-16 |
| IMDb | IVM churn | 0.001 | 3,859,580 | 1.3e-2 | 1.93e+0 | 46 | 5.3e-16 |
| IMDb | IVM churn | 0.02 | 3,859,580 | 4.2e-2 | 1.44e+1 | 352 | 2.8e-14 |
| IMDb | IVM churn | 0.1 | 3,859,580 | 1.4e-2 | 7.74e-2 | 0 | 1.7e-14 |
| IMDb | IVM churn | 1 | 3,859,580 | 1.5e-2 | 8.63e-2 | 0 | 1.3e-14 |
| Amazon | IVM churn | 0.0001 | 838,280 | 0 | 1.90e-1 | 0 | 2.4e-16 |
| Amazon | IVM churn | 0.001 | 838,280 | 1.4e-3 | 6.66e-1 | 0 | 5.5e-16 |
| Amazon | IVM churn | 0.02 | 838,280 | 1.4e-2 | 7.42e+0 | 16 | 3.1e-14 |
| Amazon | IVM churn | 0.1 | 838,280 | 9.1e-3 | 3.45e-2 | 0 | 1.9e-14 |
| Amazon | IVM churn | 1 | 838,280 | 9.1e-3 | 2.46e-2 | 0 | 2.1e-14 |
| StackExchange | sliding w=50 | 0.0001 | 4,997,080 | 8.7e-3 | 2.24e+1 | 238 | 3.1e-16 |
| StackExchange | sliding w=50 | 0.001 | 4,997,080 | 5.2e-2 | 3.55e+0 | 51 | 6.9e-16 |
| StackExchange | sliding w=400 | 0.0001 | 999,420 | 1.3e-3 | 1.16e+0 | 1 | 2.3e-16 |
| StackExchange | IVM churn | 0.0001 | 24,550,680 | 1.4e-3 | 2.67e+0 | 100 | 2.5e-16 |
| StackExchange | IVM churn | 0.001 | 24,550,680 | 3.5e-2 | 2.13e+0 | 753 | 8.0e-16 |
| StackExchange | IVM churn | 0.02 | 24,550,680 | 2.2e-2 | 1.30e+1 | 376 | 2.9e-15 |
| StackExchange | IVM churn | 0.1 | 24,550,680 | 1.5e-2 | 1.48e-1 | 0 | 8.9e-16 |
| StackExchange | IVM churn | 1 | 24,550,680 | 1.5e-2 | 1.17e-1 | 0 | 9.4e-16 |

(rows shown: every cell with a violation, plus all churn cells; all other cells have 0 violations.)

### T3. What the bound is made of (mean share of the bound; all states / loose states with ratio < 1e-2)

Terms: `err_u/(z-err_z)`, `|F| err_z/(z-err_z)`, `U|F|`. Components: per-row weight terms (2U w, 3U|wv|), running-sum terms (U|z1|, U|u1|), anchor-rescale terms (3U|z0|scale, 3U|u0|scale), and U|F|.

| corpus | scenario | eps | loose / all | terms (u, z, div), loose | components (weight, running-sum, rescale, div), loose |
|---|---|---|---|---|---|
| IMDb | sliding w=50 | 0.0001 | 4,489,822 / 4,656,560 | 0.49, 0.49, 0.02 | 0.000, 0.977, 0.000, 0.022 |
| IMDb | sliding w=50 | 0.001 | 3,413,097 / 4,656,560 | 0.49, 0.49, 0.02 | 0.003, 0.979, 0.001, 0.018 |
| IMDb | sliding w=50 | 0.02 | 1,559,011 / 4,656,560 | 0.50, 0.49, 0.00 | 0.030, 0.963, 0.005, 0.002 |
| IMDb | sliding w=50 | 0.1 | 4,612,416 / 4,656,560 | 0.50, 0.50, 0.00 | 0.047, 0.953, 0.001, 0.000 |
| IMDb | sliding w=50 | 1 | 4,651,276 / 4,656,560 | 0.49, 0.51, 0.00 | 0.047, 0.953, 0.000, 0.000 |
| IMDb | sliding w=400 | 0.0001 | 4,639,727 / 4,656,560 | 0.50, 0.50, 0.00 | 0.000, 0.996, 0.000, 0.004 |
| IMDb | sliding w=400 | 0.001 | 4,536,770 / 4,656,560 | 0.50, 0.50, 0.00 | 0.000, 0.996, 0.000, 0.003 |
| IMDb | sliding w=400 | 0.02 | 3,320,849 / 4,656,560 | 0.50, 0.50, 0.00 | 0.004, 0.995, 0.001, 0.000 |
| IMDb | sliding w=400 | 0.1 | 4,640,854 / 4,656,560 | 0.49, 0.51, 0.00 | 0.006, 0.994, 0.000, 0.000 |
| IMDb | sliding w=400 | 1 | 4,646,543 / 4,656,560 | 0.49, 0.51, 0.00 | 0.006, 0.994, 0.000, 0.000 |
| IMDb | sliding w=5000 | 0.0001 | 4,654,771 / 4,656,560 | 0.50, 0.50, 0.00 | 0.000, 0.999, 0.000, 0.001 |
| IMDb | sliding w=5000 | 0.001 | 4,646,454 / 4,656,560 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| IMDb | sliding w=5000 | 0.02 | 4,529,000 / 4,656,560 | 0.50, 0.50, 0.00 | 0.000, 0.999, 0.000, 0.000 |
| IMDb | sliding w=5000 | 0.1 | 4,630,974 / 4,656,560 | 0.49, 0.51, 0.00 | 0.001, 0.999, 0.000, 0.000 |
| IMDb | sliding w=5000 | 1 | 4,633,143 / 4,656,560 | 0.49, 0.51, 0.00 | 0.001, 0.999, 0.000, 0.000 |
| IMDb | IVM churn | 0.0001 | 5,190,422 / 5,200,520 | 0.50, 0.50, 0.00 | 0.000, 0.998, 0.000, 0.002 |
| IMDb | IVM churn | 0.001 | 5,119,319 / 5,200,520 | 0.50, 0.50, 0.00 | 0.001, 0.998, 0.000, 0.002 |
| IMDb | IVM churn | 0.02 | 4,858,567 / 5,200,520 | 0.50, 0.50, 0.00 | 0.003, 0.996, 0.000, 0.000 |
| IMDb | IVM churn | 0.1 | 5,090,077 / 5,200,520 | 0.50, 0.50, 0.00 | 0.004, 0.995, 0.001, 0.000 |
| IMDb | IVM churn | 1 | 5,096,085 / 5,200,520 | 0.50, 0.50, 0.00 | 0.004, 0.995, 0.001, 0.000 |
| IMDb | descending deletes | 0.0001 | 365,096 / 400,000 | 0.50, 0.50, 0.00 | 0.009, 0.987, 0.003, 0.001 |
| IMDb | descending deletes | 0.001 | 315,265 / 400,000 | 0.50, 0.50, 0.00 | 0.010, 0.984, 0.005, 0.001 |
| IMDb | descending deletes | 0.02 | 307,186 / 400,000 | 0.50, 0.50, 0.00 | 0.006, 0.989, 0.004, 0.000 |
| IMDb | descending deletes | 0.1 | 361,884 / 400,000 | 0.50, 0.50, 0.00 | 0.003, 0.994, 0.003, 0.000 |
| IMDb | descending deletes | 1 | 349,371 / 400,000 | 0.50, 0.50, 0.00 | 0.002, 0.995, 0.002, 0.000 |
| Amazon | sliding w=50 | 0.0001 | 26,233,107 / 26,994,180 | 0.49, 0.49, 0.02 | 0.001, 0.977, 0.000, 0.022 |
| Amazon | sliding w=50 | 0.001 | 20,490,202 / 26,994,180 | 0.49, 0.49, 0.02 | 0.003, 0.977, 0.001, 0.019 |
| Amazon | sliding w=50 | 0.02 | 10,509,623 / 26,994,180 | 0.50, 0.50, 0.00 | 0.032, 0.962, 0.004, 0.001 |
| Amazon | sliding w=50 | 0.1 | 26,917,318 / 26,994,180 | 0.51, 0.49, 0.00 | 0.047, 0.952, 0.001, 0.000 |
| Amazon | sliding w=50 | 1 | 26,975,681 / 26,994,180 | 0.51, 0.49, 0.00 | 0.047, 0.952, 0.000, 0.000 |
| Amazon | sliding w=400 | 0.0001 | 26,915,638 / 26,994,180 | 0.50, 0.50, 0.00 | 0.000, 0.996, 0.000, 0.004 |
| Amazon | sliding w=400 | 0.001 | 26,372,993 / 26,994,180 | 0.50, 0.50, 0.00 | 0.000, 0.996, 0.000, 0.003 |
| Amazon | sliding w=400 | 0.02 | 21,279,574 / 26,994,180 | 0.50, 0.50, 0.00 | 0.004, 0.995, 0.001, 0.000 |
| Amazon | sliding w=400 | 0.1 | 26,949,060 / 26,994,180 | 0.50, 0.50, 0.00 | 0.006, 0.994, 0.000, 0.000 |
| Amazon | sliding w=400 | 1 | 26,957,217 / 26,994,180 | 0.50, 0.50, 0.00 | 0.006, 0.994, 0.000, 0.000 |
| Amazon | sliding w=5000 | 0.0001 | 26,988,050 / 26,994,180 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| Amazon | sliding w=5000 | 0.001 | 26,952,005 / 26,994,180 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| Amazon | sliding w=5000 | 0.02 | 26,659,443 / 26,994,180 | 0.50, 0.50, 0.00 | 0.000, 0.999, 0.000, 0.000 |
| Amazon | sliding w=5000 | 0.1 | 26,965,352 / 26,994,180 | 0.50, 0.50, 0.00 | 0.001, 0.999, 0.000, 0.000 |
| Amazon | sliding w=5000 | 1 | 26,975,265 / 26,994,180 | 0.50, 0.50, 0.00 | 0.001, 0.999, 0.000, 0.000 |
| Amazon | IVM churn | 0.0001 | 6,600,136 / 6,600,660 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| Amazon | IVM churn | 0.001 | 6,597,739 / 6,600,660 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| Amazon | IVM churn | 0.02 | 6,575,848 / 6,600,660 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| Amazon | IVM churn | 0.1 | 6,593,912 / 6,600,660 | 0.50, 0.50, 0.00 | 0.000, 0.999, 0.000, 0.000 |
| Amazon | IVM churn | 1 | 6,594,157 / 6,600,660 | 0.50, 0.50, 0.00 | 0.000, 0.999, 0.000, 0.000 |
| Amazon | descending deletes | 0.0001 | 365,355 / 400,000 | 0.50, 0.50, 0.00 | 0.009, 0.986, 0.002, 0.002 |
| Amazon | descending deletes | 0.001 | 316,398 / 400,000 | 0.50, 0.50, 0.00 | 0.010, 0.984, 0.005, 0.001 |
| Amazon | descending deletes | 0.02 | 277,947 / 400,000 | 0.50, 0.50, 0.00 | 0.007, 0.988, 0.005, 0.000 |
| Amazon | descending deletes | 0.1 | 337,450 / 400,000 | 0.51, 0.49, 0.00 | 0.004, 0.994, 0.003, 0.000 |
| Amazon | descending deletes | 1 | 354,443 / 400,000 | 0.51, 0.49, 0.00 | 0.002, 0.995, 0.002, 0.000 |
| StackExchange | sliding w=50 | 0.0001 | 9,422,077 / 9,980,649 | 0.49, 0.49, 0.02 | 0.001, 0.978, 0.000, 0.022 |
| StackExchange | sliding w=50 | 0.001 | 6,591,424 / 9,994,160 | 0.50, 0.49, 0.02 | 0.004, 0.979, 0.001, 0.016 |
| StackExchange | sliding w=50 | 0.02 | 5,223,108 / 9,994,160 | 0.54, 0.45, 0.00 | 0.038, 0.960, 0.001, 0.000 |
| StackExchange | sliding w=50 | 0.1 | 9,977,989 / 9,994,160 | 0.56, 0.44, 0.00 | 0.048, 0.952, 0.000, 0.000 |
| StackExchange | sliding w=50 | 1 | 9,987,197 / 9,994,160 | 0.56, 0.44, 0.00 | 0.048, 0.952, 0.000, 0.000 |
| StackExchange | sliding w=400 | 0.0001 | 9,849,342 / 9,982,083 | 0.50, 0.50, 0.00 | 0.000, 0.996, 0.000, 0.004 |
| StackExchange | sliding w=400 | 0.001 | 9,639,029 / 9,994,160 | 0.50, 0.50, 0.00 | 0.001, 0.996, 0.000, 0.003 |
| StackExchange | sliding w=400 | 0.02 | 8,453,834 / 9,994,160 | 0.53, 0.47, 0.00 | 0.005, 0.995, 0.000, 0.000 |
| StackExchange | sliding w=400 | 0.1 | 9,984,218 / 9,994,160 | 0.55, 0.45, 0.00 | 0.006, 0.994, 0.000, 0.000 |
| StackExchange | sliding w=400 | 1 | 9,982,897 / 9,994,160 | 0.55, 0.45, 0.00 | 0.006, 0.994, 0.000, 0.000 |
| StackExchange | sliding w=5000 | 0.0001 | 9,939,361 / 9,976,044 | 0.50, 0.50, 0.00 | 0.000, 0.999, 0.000, 0.001 |
| StackExchange | sliding w=5000 | 0.001 | 9,963,773 / 9,994,160 | 0.50, 0.50, 0.00 | 0.000, 1.000, 0.000, 0.000 |
| StackExchange | sliding w=5000 | 0.02 | 9,887,634 / 9,994,160 | 0.49, 0.51, 0.00 | 0.000, 0.999, 0.000, 0.000 |
| StackExchange | sliding w=5000 | 0.1 | 9,977,076 / 9,994,160 | 0.54, 0.46, 0.00 | 0.001, 0.999, 0.000, 0.000 |
| StackExchange | sliding w=5000 | 1 | 9,961,967 / 9,994,160 | 0.54, 0.46, 0.00 | 0.001, 0.999, 0.000, 0.000 |
| StackExchange | IVM churn | 0.0001 | 28,342,290 / 28,785,241 | 0.50, 0.50, 0.00 | 0.000, 0.995, 0.000, 0.004 |
| StackExchange | IVM churn | 0.001 | 27,383,739 / 28,802,880 | 0.50, 0.50, 0.00 | 0.002, 0.996, 0.000, 0.002 |
| StackExchange | IVM churn | 0.02 | 27,435,167 / 28,802,880 | 0.50, 0.50, 0.00 | 0.008, 0.992, 0.001, 0.000 |
| StackExchange | IVM churn | 0.1 | 28,149,728 / 28,802,880 | 0.50, 0.50, 0.00 | 0.008, 0.990, 0.001, 0.000 |
| StackExchange | IVM churn | 1 | 28,133,717 / 28,802,880 | 0.50, 0.50, 0.00 | 0.008, 0.990, 0.001, 0.000 |
| StackExchange | descending deletes | 0.0001 | 360,921 / 399,998 | 0.52, 0.48, 0.00 | 0.010, 0.986, 0.003, 0.001 |
| StackExchange | descending deletes | 0.001 | 308,313 / 399,999 | 0.52, 0.48, 0.00 | 0.010, 0.984, 0.005, 0.001 |
| StackExchange | descending deletes | 0.02 | 296,518 / 399,999 | 0.49, 0.51, 0.00 | 0.006, 0.991, 0.003, 0.000 |
| StackExchange | descending deletes | 0.1 | 353,114 / 400,000 | 0.53, 0.47, 0.00 | 0.003, 0.995, 0.002, 0.000 |
| StackExchange | descending deletes | 1 | 355,363 / 400,000 | 0.53, 0.47, 0.00 | 0.002, 0.996, 0.002, 0.000 |

### T4. Refolds: shipped bound vs oracle vs DdInv prototype (same tolerance, all queries)

oracle = shipped arithmetic, refold only when the true error exceeds the tolerance (the floor for ANY bound on this arithmetic). `inf` = refolds where z no longer exceeded its own error bound (cancellation/underflow).

| corpus | scenario | eps | steps | shipped refolds (inf) | refolded rows / step | oracle refolds | DdInv refolds (inf) | max err/max|v| shipped | max err/max|v| DdInv | DdInv violations |
|---|---|---|---|---|---|---|---|---|---|---|
| IMDb | sliding w=50 | 0.0001 | 4,656,560 | 95,396 (90,734) | 1.02 | 93,987 | 88,882 (84,574) | 1.2e-12 | 8.0e-14 | 0 |
| IMDb | sliding w=50 | 0.001 | 4,656,560 | 81,209 (49,324) | 0.872 | 73,007 | 51,527 (29,884) | 3.9e-12 | 1.2e-12 | 0 |
| IMDb | sliding w=50 | 0.02 | 4,656,560 | 8,479 (2) | 0.091 | 2,698 | 2,215 (0) | 8.4e-12 | 1.0e-12 | 0 |
| IMDb | sliding w=50 | 0.1 | 4,656,560 | 22 (0) | 0.000236 | 0 | 2 (0) | 3.6e-13 | 3.7e-16 | 0 |
| IMDb | sliding w=50 | 1 | 4,656,560 | 0 (0) | 0 | 0 | 0 (0) | 3.4e-14 | 8.8e-17 | 0 |
| IMDb | sliding w=400 | 0.0001 | 4,656,560 | 12,413 (11,721) | 1.07 | 12,138 | 11,458 (10,867) | 1.1e-13 | 4.2e-15 | 0 |
| IMDb | sliding w=400 | 0.001 | 4,656,560 | 10,790 (6,220) | 0.927 | 9,260 | 6,569 (3,639) | 3.1e-13 | 9.0e-13 | 0 |
| IMDb | sliding w=400 | 0.02 | 4,656,560 | 1,584 (2) | 0.136 | 409 | 254 (0) | 2.8e-12 | 4.0e-13 | 0 |
| IMDb | sliding w=400 | 0.1 | 4,656,560 | 6 (0) | 0.000515 | 0 | 0 (0) | 1.0e-13 | 1.2e-16 | 0 |
| IMDb | sliding w=400 | 1 | 4,656,560 | 0 (0) | 0 | 0 | 0 (0) | 2.3e-14 | 7.3e-17 | 0 |
| IMDb | sliding w=5000 | 0.0001 | 4,656,560 | 911 (867) | 0.978 | 890 | 850 (799) | 1.4e-15 | 9.2e-17 | 0 |
| IMDb | sliding w=5000 | 0.001 | 4,656,560 | 839 (479) | 0.901 | 664 | 482 (273) | 3.0e-14 | 1.3e-14 | 0 |
| IMDb | sliding w=5000 | 0.02 | 4,656,560 | 214 (1) | 0.23 | 33 | 11 (0) | 6.1e-13 | 1.3e-13 | 0 |
| IMDb | sliding w=5000 | 0.1 | 4,656,560 | 2 (0) | 0.00215 | 0 | 0 (0) | 9.7e-14 | 7.4e-17 | 0 |
| IMDb | sliding w=5000 | 1 | 4,656,560 | 0 (0) | 0 | 0 | 0 (0) | 4.1e-14 | 5.9e-17 | 0 |
| IMDb | IVM churn | 0.0001 | 5,200,520 | 3,777 (3,575) | 0.513 | 3,717 | 3,453 (3,243) | 2.8e-13 | 1.7e-15 | 0 |
| IMDb | IVM churn | 0.001 | 5,200,520 | 3,173 (1,681) | 0.445 | 2,651 | 1,591 (865) | 4.0e-12 | 4.9e-14 | 0 |
| IMDb | IVM churn | 0.02 | 5,200,520 | 187 (0) | 0.0324 | 15 | 0 (0) | 2.0e-12 | 2.1e-15 | 0 |
| IMDb | IVM churn | 0.1 | 5,200,520 | 0 (0) | 0 | 0 | 0 (0) | 1.7e-14 | 5.3e-17 | 0 |
| IMDb | IVM churn | 1 | 5,200,520 | 0 (0) | 0 | 0 | 0 (0) | 1.7e-14 | 4.8e-17 | 0 |
| IMDb | descending deletes | 0.0001 | 400,000 | 77,414 (4,500) | 219 | 48,668 | 40,313 (2,839) | 1.2e-11 | 4.3e-12 | 0 |
| IMDb | descending deletes | 0.001 | 400,000 | 12,452 (257) | 38.2 | 7,429 | 7,416 (124) | 1.2e-11 | 1.7e-12 | 0 |
| IMDb | descending deletes | 0.02 | 400,000 | 754 (0) | 1.7 | 420 | 447 (0) | 4.7e-12 | 4.6e-13 | 0 |
| IMDb | descending deletes | 0.1 | 400,000 | 200 (0) | 0.0178 | 9 | 125 (0) | 1.7e-12 | 5.1e-13 | 0 |
| IMDb | descending deletes | 1 | 400,000 | 200 (0) | 0.00313 | 0 | 0 (0) | 1.3e-12 | 4.1e-14 | 0 |
| Amazon | sliding w=50 | 0.0001 | 26,994,180 | 530,189 (501,788) | 0.982 | 514,477 | 491,044 (464,803) | 4.1e-12 | 1.8e-13 | 0 |
| Amazon | sliding w=50 | 0.001 | 26,994,180 | 447,280 (255,344) | 0.828 | 380,878 | 273,371 (145,725) | 2.3e-11 | 9.0e-12 | 0 |
| Amazon | sliding w=50 | 0.02 | 26,994,180 | 34,780 (5) | 0.0644 | 8,054 | 8,207 (0) | 1.1e-11 | 1.8e-12 | 0 |
| Amazon | sliding w=50 | 0.1 | 26,994,180 | 247 (0) | 0.000458 | 0 | 125 (0) | 5.9e-13 | 1.0e-15 | 0 |
| Amazon | sliding w=50 | 1 | 26,994,180 | 108 (0) | 0.0002 | 0 | 41 (0) | 7.5e-14 | 1.8e-16 | 0 |
| Amazon | sliding w=400 | 0.0001 | 26,994,180 | 67,403 (63,131) | 0.999 | 64,695 | 61,503 (57,548) | 1.7e-13 | 1.1e-14 | 0 |
| Amazon | sliding w=400 | 0.001 | 26,994,180 | 57,459 (29,698) | 0.851 | 46,154 | 32,694 (15,341) | 6.8e-13 | 1.2e-12 | 0 |
| Amazon | sliding w=400 | 0.02 | 26,994,180 | 5,727 (0) | 0.0849 | 860 | 730 (0) | 5.1e-12 | 3.3e-13 | 0 |
| Amazon | sliding w=400 | 0.1 | 26,994,180 | 151 (0) | 0.00224 | 0 | 13 (0) | 1.4e-13 | 5.6e-16 | 0 |
| Amazon | sliding w=400 | 1 | 26,994,180 | 100 (0) | 0.00148 | 0 | 0 (0) | 5.2e-14 | 1.4e-16 | 0 |
| Amazon | sliding w=5000 | 0.0001 | 26,994,180 | 5,717 (5,280) | 1.06 | 5,426 | 5,132 (4,788) | 8.2e-15 | 1.8e-16 | 0 |
| Amazon | sliding w=5000 | 0.001 | 26,994,180 | 5,080 (2,379) | 0.941 | 3,768 | 2,647 (1,137) | 3.8e-14 | 3.2e-12 | 0 |
| Amazon | sliding w=5000 | 0.02 | 26,994,180 | 959 (0) | 0.178 | 165 | 113 (0) | 1.1e-12 | 1.3e-13 | 0 |
| Amazon | sliding w=5000 | 0.1 | 26,994,180 | 126 (0) | 0.0233 | 0 | 0 (0) | 1.7e-13 | 2.9e-16 | 0 |
| Amazon | sliding w=5000 | 1 | 26,994,180 | 100 (0) | 0.0185 | 0 | 0 (0) | 5.5e-14 | 1.1e-16 | 0 |
| Amazon | IVM churn | 0.0001 | 6,600,660 | 214 (204) | 0.508 | 205 | 191 (178) | 2.4e-16 | 8.6e-17 | 0 |
| Amazon | IVM churn | 0.001 | 6,600,660 | 184 (78) | 0.441 | 118 | 54 (28) | 1.8e-14 | 1.8e-16 | 0 |
| Amazon | IVM churn | 0.02 | 6,600,660 | 9 (0) | 0.0189 | 4 | 0 (0) | 3.7e-13 | 1.0e-16 | 0 |
| Amazon | IVM churn | 0.1 | 6,600,660 | 0 (0) | 0 | 0 | 0 (0) | 2.2e-14 | 9.1e-17 | 0 |
| Amazon | IVM churn | 1 | 6,600,660 | 0 (0) | 0 | 0 | 0 (0) | 3.0e-14 | 9.1e-17 | 0 |
| Amazon | descending deletes | 0.0001 | 400,000 | 83,786 (4,342) | 232 | 51,258 | 42,528 (2,705) | 2.5e-11 | 2.7e-12 | 0 |
| Amazon | descending deletes | 0.001 | 400,000 | 13,364 (210) | 40 | 7,781 | 7,949 (90) | 2.3e-11 | 3.1e-12 | 0 |
| Amazon | descending deletes | 0.02 | 400,000 | 776 (0) | 1.65 | 422 | 473 (0) | 5.7e-12 | 9.9e-13 | 0 |
| Amazon | descending deletes | 0.1 | 400,000 | 200 (0) | 0.0267 | 41 | 154 (0) | 1.9e-12 | 9.9e-13 | 0 |
| Amazon | descending deletes | 1 | 400,000 | 200 (0) | 0.00412 | 0 | 0 (0) | 1.2e-12 | 8.1e-14 | 0 |
| StackExchange | sliding w=50 | 0.0001 | 9,994,160 | 191,280 (179,498) | 0.957 | 185,733 | 173,859 (163,207) | 2.9e-13 | 3.3e-14 | 0 |
| StackExchange | sliding w=50 | 0.001 | 9,994,160 | 147,640 (77,376) | 0.739 | 127,146 | 84,770 (40,136) | 1.6e-12 | 1.5e-12 | 0 |
| StackExchange | sliding w=50 | 0.02 | 9,994,160 | 2,340 (2) | 0.0117 | 516 | 471 (0) | 4.6e-12 | 3.2e-13 | 0 |
| StackExchange | sliding w=50 | 0.1 | 9,994,160 | 0 (0) | 0 | 0 | 0 (0) | 1.7e-14 | 3.6e-16 | 0 |
| StackExchange | sliding w=50 | 1 | 9,994,160 | 0 (0) | 0 | 0 | 0 (0) | 1.4e-15 | 1.2e-17 | 0 |
| StackExchange | sliding w=400 | 0.0001 | 9,994,160 | 25,131 (23,350) | 1.01 | 24,120 | 22,502 (20,949) | 1.1e-14 | 1.7e-15 | 0 |
| StackExchange | sliding w=400 | 0.001 | 9,994,160 | 19,199 (9,467) | 0.768 | 15,571 | 10,412 (4,608) | 1.1e-13 | 2.0e-13 | 0 |
| StackExchange | sliding w=400 | 0.02 | 9,994,160 | 462 (1) | 0.0185 | 105 | 91 (0) | 1.2e-12 | 1.3e-13 | 0 |
| StackExchange | sliding w=400 | 0.1 | 9,994,160 | 0 (0) | 0 | 0 | 0 (0) | 4.3e-15 | 7.4e-17 | 0 |
| StackExchange | sliding w=400 | 1 | 9,994,160 | 0 (0) | 0 | 0 | 0 (0) | 2.3e-16 | 2.9e-18 | 0 |
| StackExchange | sliding w=5000 | 0.0001 | 9,994,160 | 2,493 (2,369) | 1.25 | 2,392 | 2,266 (2,122) | 9.6e-16 | 6.3e-18 | 0 |
| StackExchange | sliding w=5000 | 0.001 | 9,994,160 | 2,012 (1,061) | 1.01 | 1,556 | 1,096 (552) | 6.9e-15 | 2.4e-13 | 0 |
| StackExchange | sliding w=5000 | 0.02 | 9,994,160 | 143 (1) | 0.0715 | 36 | 23 (0) | 4.4e-13 | 2.6e-13 | 0 |
| StackExchange | sliding w=5000 | 0.1 | 9,994,160 | 0 (0) | 0 | 0 | 0 (0) | 1.2e-15 | 4.8e-18 | 0 |
| StackExchange | sliding w=5000 | 1 | 9,994,160 | 0 (0) | 0 | 0 | 0 (0) | 1.8e-16 | 1.9e-18 | 0 |
| StackExchange | IVM churn | 0.0001 | 28,802,880 | 44,492 (41,149) | 0.489 | 42,724 | 39,424 (36,355) | 3.4e-14 | 1.6e-16 | 0 |
| StackExchange | IVM churn | 0.001 | 28,802,880 | 33,042 (14,732) | 0.363 | 26,005 | 13,810 (5,296) | 1.3e-13 | 1.7e-14 | 0 |
| StackExchange | IVM churn | 0.02 | 28,802,880 | 190 (0) | 0.00188 | 13 | 1 (0) | 3.2e-14 | 6.4e-18 | 0 |
| StackExchange | IVM churn | 0.1 | 28,802,880 | 0 (0) | 0 | 0 | 0 (0) | 7.9e-17 | 1.1e-18 | 0 |
| StackExchange | IVM churn | 1 | 28,802,880 | 0 (0) | 0 | 0 | 0 (0) | 9.0e-17 | 5.3e-19 | 0 |
| StackExchange | descending deletes | 0.0001 | 400,000 | 53,675 (3,671) | 146 | 37,247 | 31,678 (2,265) | 2.6e-12 | 3.0e-12 | 0 |
| StackExchange | descending deletes | 0.001 | 400,000 | 7,897 (168) | 22.4 | 5,362 | 5,326 (73) | 5.4e-12 | 3.7e-12 | 0 |
| StackExchange | descending deletes | 0.02 | 400,000 | 410 (0) | 0.549 | 231 | 242 (0) | 2.2e-12 | 1.4e-12 | 0 |
| StackExchange | descending deletes | 0.1 | 400,000 | 124 (0) | 0.00065 | 0 | 0 (0) | 5.3e-13 | 1.9e-14 | 0 |
| StackExchange | descending deletes | 1 | 400,000 | 6 (0) | 1.75e-05 | 0 | 0 (0) | 7.7e-14 | 7.7e-16 | 0 |

### T5. Tolerance sweep (local copy of InvAcc, first sweep queries, pooled over corpora and scenarios)

Cell: refolded rows per step / max |error| / max|v|. `tol = max(abs, gain * n * U) * v_scale`.

| config | eps=0.0001 | eps=0.001 | eps=0.02 | eps=0.1 | eps=1 |
|---|---|---|---|---|---|
| abs=1e-14,gain=100 | 2.74 / 1.7e-12 | 1.13 / 6.9e-13 | 0.2 / 9.2e-13 | 0.0254 / 1.3e-13 | 0.02 / 3.9e-14 |
| abs=1e-13,gain=100 | 2.74 / 1.7e-12 | 1.13 / 6.9e-13 | 0.2 / 9.2e-13 | 0.0254 / 1.3e-13 | 0.02 / 3.9e-14 |
| abs=1e-12,gain=100 | 2.73 / 1.7e-12 | 1.13 / 6.9e-13 | 0.19 / 9.2e-13 | 0.0214 / 1.3e-13 | 0.0168 / 3.9e-14 |
| abs=1e-11,gain=100 | 2.71 / 1.7e-12 | 1.11 / 1.7e-12 | 0.147 / 1.2e-12 | 0.0118 / 4.1e-13 | 0.00937 / 1.3e-13 |
| abs=1e-10,gain=100 | 2.4 / 2.0e-11 | 1.01 / 2.3e-11 | 0.0922 / 1.1e-11 | 0.00446 / 1.8e-12 | 0.00329 / 1.1e-12 |
| abs=1e-9,gain=100 | 2.14 / 1.5e-10 | 0.913 / 2.0e-10 | 0.046 / 1.0e-10 | 0.000461 / 1.7e-11 | 6.24e-07 / 1.2e-11 |
| abs=1e-8,gain=100 | 1.96 / 1.5e-9 | 0.842 / 2.2e-9 | 0.0252 / 1.0e-9 | 1.92e-06 / 1.6e-10 | 0 / 1.3e-11 |
| abs=1e-6,gain=100 | 1.72 / 1.4e-7 | 0.735 / 1.0e-7 | 0.00971 / 1.3e-7 | 0 / 5.4e-10 | 0 / 1.3e-11 |
| abs=1e-10,gain=0e0 | 2.4 / 2.0e-11 | 1.01 / 2.3e-11 | 0.0922 / 1.1e-11 | 0.00446 / 1.8e-12 | 0.00329 / 1.1e-12 |
| abs=1e-10,gain=1e0 | 2.4 / 2.0e-11 | 1.01 / 2.3e-11 | 0.0922 / 1.1e-11 | 0.00446 / 1.8e-12 | 0.00329 / 1.1e-12 |
| abs=1e-10,gain=1e1 | 2.4 / 2.0e-11 | 1.01 / 2.3e-11 | 0.0922 / 1.1e-11 | 0.00446 / 1.8e-12 | 0.00329 / 1.1e-12 |
| abs=1e-10,gain=1e3 | 2.34 / 2.0e-11 | 0.98 / 2.3e-11 | 0.0721 / 1.1e-11 | 0.00158 / 1.8e-12 | 0.000293 / 1.1e-12 |
| abs=1e-10,gain=1e4 | 2.11 / 2.0e-10 | 0.909 / 1.0e-10 | 0.0506 / 1.0e-10 | 0.000325 / 2.0e-12 | 0.000101 / 1.1e-12 |
| abs=1e-10,gain=1e5 | 1.94 / 1.7e-9 | 0.845 / 4.7e-10 | 0.0299 / 9.9e-10 | 9.13e-05 / 7.6e-12 | 2.39e-05 / 1.1e-12 |
| abs=0,gain=1e0 | 1.96e+03 / 7.9e-15 | 1.96e+03 / 8.1e-15 | 1.93e+03 / 2.5e-14 | 2.11e+03 / 1.7e-14 | 2.3e+03 / 1.7e-14 |
| abs=0,gain=1e1 | 3.48 / 1.3e-13 | 1.39 / 5.1e-14 | 0.456 / 1.0e-13 | 0.231 / 2.8e-14 | 0.22 / 2.4e-14 |
| abs=0,gain=1e2 | 2.74 / 1.7e-12 | 1.13 / 6.9e-13 | 0.2 / 9.2e-13 | 0.0254 / 1.3e-13 | 0.02 / 3.9e-14 |
| abs=0,gain=1e3 | 2.35 / 1.3e-11 | 1 / 6.9e-12 | 0.1 / 9.9e-12 | 0.00352 / 5.2e-13 | 0.00151 / 1.1e-13 |
| abs=0,gain=1e4 | 2.11 / 2.0e-10 | 0.912 / 1.0e-10 | 0.0537 / 1.0e-10 | 0.000432 / 2.0e-12 | 0.000181 / 3.6e-13 |
| DdInv abs=1e-10,gain=100 | 1.54 / 4.3e-12 | 0.539 / 9.0e-12 | 0.0139 / 1.8e-12 | 7.22e-05 / 9.9e-13 | 1.23e-05 / 8.1e-14 |
| DdInv abs=1e-12,gain=100 | 1.66 / 1.1e-12 | 0.605 / 3.4e-13 | 0.0368 / 8.0e-14 | 0.00237 / 1.3e-14 | 0.0013 / 2.7e-15 |
| DdInv abs=1e-14,gain=100 | 1.66 / 1.1e-12 | 0.607 / 3.4e-13 | 0.0402 / 8.0e-14 | 0.00343 / 2.2e-15 | 0.00205 / 7.9e-16 |
| DdInv abs=1e-14,gain=1 | 1.95 / 3.6e-15 | 0.767 / 4.7e-15 | 0.164 / 9.3e-16 | 0.103 / 2.2e-16 | 0.0972 / 1.2e-16 |

GAIN only matters for groups above ABS_TOL/(GAIN U) = 9,000 rows (churn groups); per-scenario numbers are in results.json -> corpora.*.cells[].sweep_local_copy.

### T6. Per-row cost (ns/row, minimum of 9 repetitions, 19,790,380 rows; timing_status: smoke, machine shared; machine: `{"quiet": false, "require_quiet_message": "[quiet] REFUSING to record timings: cpu 88.6% busy (limit 15%), gpu 82% (limit 20%).", "cpu_busy_fraction": 0.9, "load_average_1min": 32.42, "timestamp": "2026-11-06T16:06:06"}`)

| eps | scalar_absorb | ess_absorb | softavg_plus_ess_absorb | invacc_absorb | local_copy_absorb | ddinv_absorb | invacc_slide400_with_refolds_per_row | ddinv_slide400_with_refolds_per_row |
|---|---|---|---|---|---|---|---|---|
| 0.0001 | 6.13 (1.00x) | 5.83 (0.95x) | 5.19 (0.85x) | 14.41 (2.35x) | 14.31 (2.33x) | 9.04 (1.47x) | 31.11 (5.07x) | 19.32 (3.15x) |
| 0.001 | 8.38 (1.00x) | 9.46 (1.13x) | 8.87 (1.06x) | 17.97 (2.15x) | 17.89 (2.14x) | 18.46 (2.20x) | 29.29 (3.50x) | 42.51 (5.07x) |
| 0.02 | 3.98 (1.00x) | 3.99 (1.00x) | 3.75 (0.94x) | 11.26 (2.83x) | 11.18 (2.81x) | 16.51 (4.15x) | 20.82 (5.23x) | 34.26 (8.61x) |
| 0.1 | 4.11 (1.00x) | 4.13 (1.00x) | 3.80 (0.93x) | 10.49 (2.55x) | 11.83 (2.88x) | 17.27 (4.20x) | 21.26 (5.17x) | 36.38 (8.85x) |
| 1 | 4.30 (1.00x) | 4.36 (1.01x) | 4.06 (0.94x) | 12.02 (2.79x) | 11.97 (2.78x) | 17.45 (4.06x) | 20.46 (4.75x) | 35.78 (8.31x) |

State sizes (bytes): `{"ScalarAcc": 24, "InvAcc": 64, "DdInv_prototype_incl_2_tol_fields": 112, "EssAcc": 24}`.

### T7. ESS = (sum w)^2 / sum w^2 per facet group (all admitted rows of a facet), all queries

| corpus | eps | groups | p10 | median | p90 | share ESS < 2 | share ESS < 10 | median ESS/n | max rel diff merge(7) vs sequential |
|---|---|---|---|---|---|---|---|---|---|
| IMDb | 0.0001 | 560 | 1 | 1 | 1 | 0.99 | 1.00 | 0.000549 | 5.0e-16 |
| IMDb | 0.001 | 560 | 1 | 1 | 1.09 | 0.97 | 1.00 | 0.000549 | 1.1e-15 |
| IMDb | 0.02 | 560 | 1.04 | 2.58 | 8.16 | 0.39 | 0.93 | 0.0015 | 1.1e-12 |
| IMDb | 0.1 | 560 | 65.4 | 631 | 6.04e+03 | 0.00 | 0.01 | 0.362 | 8.4e-14 |
| IMDb | 1 | 560 | 313 | 2.11e+03 | 3.08e+04 | 0.00 | 0.00 | 0.994 | 9.7e-14 |
| Amazon | 0.0001 | 660 | 1 | 1 | 1 | 0.99 | 1.00 | 2.19e-05 | 0 |
| Amazon | 0.001 | 660 | 1 | 1 | 1.07 | 0.98 | 1.00 | 2.21e-05 | 1.0e-15 |
| Amazon | 0.02 | 660 | 1.19 | 3.98 | 17.2 | 0.25 | 0.79 | 0.000103 | 3.2e-13 |
| Amazon | 0.1 | 660 | 5.22e+03 | 1.64e+04 | 2.46e+04 | 0.00 | 0.00 | 0.394 | 4.6e-14 |
| Amazon | 1 | 660 | 2.88e+04 | 4.54e+04 | 4.71e+04 | 0.00 | 0.00 | 0.993 | 4.5e-14 |
| StackExchange | 0.0001 | 3000 | 1 | 1 | 1 | 1.00 | 1.00 | 0.001 | 4.4e-16 |
| StackExchange | 0.001 | 3000 | 1 | 1 | 1.15 | 0.99 | 1.00 | 0.00103 | 1.6e-15 |
| StackExchange | 0.02 | 3000 | 1.3 | 4.96 | 22.6 | 0.21 | 0.73 | 0.00526 | 9.3e-14 |
| StackExchange | 0.1 | 3000 | 110 | 584 | 4.73e+03 | 0.00 | 0.00 | 0.646 | 2.5e-14 |
| StackExchange | 1 | 3000 | 185 | 1.03e+03 | 8.25e+03 | 0.00 | 0.00 | 0.996 | 2.6e-14 |
<!-- /TABLES -->

## Figures (from results.json by report.py)

* `figures/slack_ratio.pdf`: true error / bound (median, p99, max) vs eps, per scenario and corpus.
* `figures/slack_vs_ops.pdf`: geometric-mean ratio vs operations since the last fresh fold (slope -1/2 reference).
* `figures/tradeoff.pdf`: refolded rows per step vs max error, ABS_TOL sweep (shipped arithmetic) and DdInv.
* `figures/ess.pdf`: ESS per facet group vs eps.

# softavg for DuckDB (C extension API, pure Rust, over softagg-core)

`softavg(value DOUBLE, score DOUBLE, eps DOUBLE) -> DOUBLE`, the max-anchored soft average
`sum exp(s/eps) v / sum exp(s/eps)`, as a first-class DuckDB **aggregate**: it runs inside
DuckDB's parallel hash aggregate (`combine` merges thread-local partials) and, as a window
function, inside DuckDB's window segment tree (also through `combine`).

Started 2026-11-05 as a copy of `../../../paper_softavg/experiments/duckdb_ext/` (that
directory is untouched). What changed:

1. **One implementation.** The private accumulator (`src/acc.rs`) is gone. All arithmetic is
   `softagg_core::ScalarAcc` from `../core` (path dependency), the crate the PostgreSQL,
   SQLite and ClickHouse extensions link. `src/state.rs` adds only what DuckDB needs and the
   core deliberately leaves to the host: the eps tag (`combine` receives no arguments, so eps
   rides in the state) and the NaN policy.
2. **Vectorised update.** One `update` call per DuckDB vector (<= 2048 rows): eps is read once
   per vector (a vectorisable bit-equality scan over the flattened eps column), validated once,
   then one tight loop per eps regime (`eps = 0`, `eps = inf`, finite; the regime branches in
   `ScalarAcc::absorb` fold away because eps is a literal in the first two) with no per-row
   validation. Vectors with NULL value/score use the same loop plus a validity test; only a
   vector whose eps differs between rows (legal: eps must be constant per *group*) or has NULL
   eps takes the per-row fallback.
3. **State layout**: `#[repr(C)] { acc: ScalarAcc {mu, z, u}, eps: f64, nan_seen: bool }`,
   40 bytes, 8-byte aligned, plain data, no destructor; eps = NaN means "unbound", so the
   per-row eps check is one 64-bit compare.
4. **Window path and conformance test** (below).
5. **Diagnostic build** `--features diag` registers `softavg_diag_noop/_touch/_2arg` (same
   state, different `update`) for cost attribution; never in the shipped build.

## Semantics (unchanged from the starting point)

| input | result |
|---|---|
| finite `eps > 0` | max-shifted softmax average (core `ScalarAcc`) |
| `eps = 0` | mean of `value` over the argmax-score set; ties = exact equality |
| `eps = 'infinity'` | plain mean = `AVG(value)`; the score is not consulted |
| NULL `value` or NULL `score` | row ignored, like `AVG`; a group with no live rows -> NULL |
| NULL `eps` on a live row | error `softavg: eps must not be NULL` (NULL eps on an ignored row is fine) |
| `eps < 0` or NaN | error `softavg: eps must be >= 0 ...` |
| eps differs within a group (incl. between partials in `combine`) | error `softavg: eps must be constant within an aggregate group (a vs b)` |
| score `-inf` (finite eps, eps = 0) | weight 0; an all-`-inf` group -> NULL |
| score `+inf` (finite eps) | dominates: argmax over the `+inf` rows, uniform ties |
| NaN `value` or `score` | result NaN (sticky bit in the state, kept out of the core accumulator), like `AVG` |
| `+0.0` vs `-0.0` eps | same temperature |

The core does not handle NaN by design (`engines/core/src/lib.rs`): DuckDB maps NULL to skip
and NaN to a sticky NaN result, matching DuckDB's own `AVG`.

## What DuckDB's C API allows (v1.5.6, `tools/reference/`)

- Aggregates: `state_size / init / update / combine / finalize` (+ optional `destroy`),
  `set_special_handling` (we see NULLs), `set_extra_info`. **No** `simple_update`, **no**
  `window` callback, **no** inverse/`MINVFUNC`, **no** bind callback (so eps cannot be
  constant-folded at bind time) and no statistics/ordering hints.
- The C-API wrapper (`aggregate_function-c.cpp`) flattens every input vector (a constant eps
  literal becomes a dense 2048-value column) and builds a `DataChunk` per `update` call; this
  per-vector cost is measured by `softavg_diag_noop` (see `../../experiments/duckdb/`).
- **Windows work without a window callback.** DuckDB picks the window aggregator itself
  (`src/function/window/window_aggregate_function.cpp`, v1.5.6): C-API aggregates have no
  `window` callback, so `WindowCustomAggregator` is skipped; ROWS/RANGE frames go to
  `WindowSegmentTree` (Leis et al. VLDB'15), whose inner nodes and frame results are
  `combine`s of our states; whole-partition frames go to `WindowConstantAggregator`
  (update + combine); `DISTINCT` goes to the merge-sort-tree `WindowDistinctAggregator`;
  `SET debug_window_mode = 'separate'` forces `WindowNaiveAggregator` (update + finalize per
  frame). Hence the core's `InvAcc::remove` has no hook in DuckDB: a sliding frame costs
  O(log n) combines instead of one add + one remove. That is DuckDB's design for every
  aggregate, built-ins included.

## Layout

```
Cargo.toml        softagg-core (path ../core) + libduckdb-sys =1.10506.0 (DuckDB 1.5.6, loadable-extension)
src/lib.rs        C-API glue: vectorised update, combine, finalize, registration, entrypoint
src/state.rs      SoftAvgState = ScalarAcc + eps tag + NaN bit; 4 cargo unit tests
build.sh          cargo build --release --offline + metadata footer -> build/softavg.duckdb_extension
tools/            append_extension_metadata.py (DuckDB extension-ci-tools) + v1.5.6 reference sources
tests/test_softavg.py      the 18 tests of the starting point + 3 update-path tests
tests/test_conformance.py  ../core/conformance/vectors.json in DuckDB (GROUP BY + window)
```

## Build and test

```bash
cd /home/jding/bruce/paper_icde/engines/duckdb
nice -n 10 ./build.sh                       # -> build/softavg.duckdb_extension (release, unsigned)
nice -n 10 cargo test --release --offline   # unit tests of src/state.rs
python -m pytest -q tests/   # duckdb==1.5.6
```

```python
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/home/jding/bruce/paper_icde/engines/duckdb/build/softavg.duckdb_extension'")
con.sql("SELECT facet, softavg(value, array_inner_product(emb, $q)::DOUBLE, 0.02) FROM movies GROUP BY facet")
con.sql("SELECT softavg(v, s, 0.1) OVER (ORDER BY id ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) FROM t")
```

Test status (2026-11-05, rebuilt against core with P1): `cargo test` 4 passed; `pytest tests/` 48 passed (18 carried over,
3 new update-path tests, 27 conformance: table load, GROUP BY at all 7 eps x threads {1, 8}
at rtol 1e-12 with null -> NULL, window at eps {0.02, 0.1, inf} x threads {1, 8} x
{segment tree, naive 'separate'} at absolute error <= 1e-9 * max|v|, the PROTOCOL's window
tolerance since 2026-11-05). Same counts after the rebuild against the updated core.

## Notes

- Load only into DuckDB 1.5.6 (libduckdb-sys copies the 1.5.6-shaped `duckdb_ext_api_v1`
  struct); rebuild against the matching libduckdb-sys for another version.
- The `.so` exports only `softavg_init_c_api` (softagg-core is an rlib without a C ABI since
  2026-11-05).
- softagg-core adopted the one-`exp`-per-row absorb (P1, proposed from the cost attribution
  here; bit-identical) on 2026-11-05; the shipped build is linked against it.
- Measurements, the before/after comparison and the cost attribution:
  `../../experiments/duckdb/README.md`. Proposed core changes that came out of the
  attribution: `../../notes/CORE_CHANGES.md` (DuckDB section).

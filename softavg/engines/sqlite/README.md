# softagg-sqlite: `softavg` as a SQLite aggregate and window function

A SQLite loadable extension in Rust (cdylib). It links **softagg-core**
(`../core`) as a path dependency: the arithmetic is core's `InvAcc`
(absorb / remove / finalize). This crate adds only the SQLite glue
(`src/lib.rs`) and the state the host needs around the core (`src/state.rs`:
eps validation and the frame shadow for inexact windows).

Built on `libsqlite3-sys 0.37` with the `loadable_extension` feature (every
SQLite call goes through the `sqlite3_api_routines` table the host passes in;
no SQLite is linked into the .so). The crates came from crates.io (network was
available; `rusqlite`/`sqlite-loadable` were not in `~/.cargo/registry`,
and the higher-level wrappers were not needed).

## Build and load

```bash
nice -n 10 cargo build --release        # target/release/libsoftagg_sqlite.so
nice -n 10 cargo test --release         # Rust unit tests (state machine, dot)
python3 tests/test_sqlite.py            # loads the .so into Python's sqlite3
```

```sql
.load ./target/release/libsoftagg_sqlite       -- entry point sqlite3_extension_init
SELECT facet, softavg(value, softagg_dot(emb, :q), 0.02) FROM imdb GROUP BY facet;
SELECT id, softavg(v, s, 0.02) OVER (ORDER BY id ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) FROM t;
```

Runtime requirement: SQLite >= 3.34.1 (the libsqlite3-sys bindings refuse
older libraries at load time; window functions themselves need 3.25). The
system SQLite here is 3.37.2 (Python's `sqlite3`), compiled with
`ENABLE_MATH_FUNCTIONS` and `ENABLE_LOAD_EXTENSION`.

## SQL surface

| function | kind | notes |
|---|---|---|
| `softavg(value, score, eps)` | aggregate + aggregate window (`xStep`, `xFinal`, `xValue`, `xInverse`) | DETERMINISTIC, INNOCUOUS |
| `softavg_agg(value, score, eps)` | aggregate only (`xStep`, `xFinal`) | same semantics and bit-identical GROUP BY result; state is the bare `ScalarAcc` monoid + eps (no `InvAcc` bound, no prefix record); `OVER (...)` is refused by SQLite |
| `softagg_dot(a, b)` | scalar | inner product of two float32 BLOBs in sqlite-vec's format (`dim*4` bytes, little-endian, no header); f32 products are exact in f64, summed in f64 over 8 fixed lanes; NULL in -> NULL; non-BLOB, length not a multiple of 4, or different lengths -> error |
| `softavg_stats()` / `softavg_stats_reset()` | scalar | process-wide JSON counters of the rare window paths: `refolds`, `refold_rows`, `shadow_spills`, `refold_refused`, `fresh_flagged` |
| `softagg_sqlite_version()` | scalar | version string |

## Semantics (NULL, NaN, +-Inf, eps)

| case | behaviour | same as PG / DuckDB? |
|---|---|---|
| NULL `value` or NULL `score` | row skipped, like `AVG` (also at eps = inf: the pair is one datum) | yes |
| no live rows in the group / frame | NULL | yes |
| `eps` NULL on a live row | error `softavg: eps must not be NULL` | yes (DuckDB) |
| `eps < 0` | error | yes |
| `eps` not constant within a group / partition | error | yes |
| `eps = 0` | argmax mean (uniform over ties) | yes |
| `eps = 9e999` (+infinity) | plain mean, `== AVG(value)`; score not consulted, so +-Inf scores count | yes |
| score `-inf` (finite eps or 0) | weight 0; all `-inf` -> NULL | yes (core) |
| score `+inf` (finite eps) | the `+inf` rows dominate, uniform among themselves | yes (core) |
| NaN `value` / `score` | **cannot occur**: SQLite has no NaN. It stores a NaN as NULL, and its arithmetic returns NULL where IEEE gives NaN (`0.0/0.0`, `1e999-1e999`, `sqrt(-1)`). A NaN therefore arrives as NULL and is skipped exactly as `AVG` skips it. The extension also skips a NaN defensively should one ever reach it. | PG/DuckDB propagate NaN (sticky) because their NaN is a value; in SQLite it is NULL, so "like AVG" means skip |
| TEXT / BLOB arguments | converted with `sqlite3_value_double`, as `AVG` does | SQLite-specific |

## Window functions and the inexact case (design decision)

`xInverse` calls `InvAcc::remove`, which never rescans. When removing rows
leaves a state the core's cancellation guard no longer trusts
(`is_exact() == false`: the core's running error bound on the delivered
value exceeds `max(1e-10, 100 n U) * max|v|`), the frame has to be refolded. SQLite cannot
recompute a frame on behalf of an aggregate window function, so the
extension keeps what it needs itself:

* **Shadow of the frame.** In window mode every live row stepped is
  appended to a FIFO (16 bytes per row); every `xInverse` pops the oldest
  one. SQLite removes frame rows in the order it added them; the pop is
  checked bit-for-bit against the `xInverse` arguments and a mismatch is an
  error, never a guess.
* **Refold lazily, exactly.** When the state is inexact at `xValue` /
  `xFinal` time, a fresh `InvAcc` is built from the shadow (O(frame) once;
  several removals between two outputs cost one refold), then the window
  is back to O(1) per row. A state that has seen no removal since it was
  built is a fresh fold (bit-identical to `ScalarAcc`) and is returned
  as is even if the core's bound were to flag it, so `GROUP BY` can never
  hit the refusal below (counter `fresh_flagged`, expected 0).
* **Why "lazy" shadow.** SQLite uses one registration for `GROUP BY` and
  `OVER (...)` and gives no signal in `xStep` which it is running; the
  first `xValue` is the first proof of window mode. To keep the `GROUP BY`
  state O(1), only the first `PREFIX_CAP = 64` rows of a group are
  recorded before that; past 64 the record is dropped and rows are only
  counted (`unknown`). Unrecorded rows are always the oldest rows of the
  frame, so they leave it first.
* **Refuse instead of guessing.** If a refold is needed while unrecorded
  rows are still in the frame, `softavg` raises
  `softavg: window frame lost precision ... cannot be refolded exactly`.
  This needs a first frame longer than 64 rows (e.g. `ROWS BETWEEN ... AND
  64 FOLLOWING` or wider) *and* a guard trip while one of those first rows
  is still in the frame, or a single frame longer than `SHADOW_CAP = 2^20`
  rows. `ROWS BETWEEN k PRECEDING AND CURRENT ROW` (first frame = one row)
  can never hit it. Tested in `tests/test_sqlite.py::test_refuse_instead_of_guess`.

Cost: window mode stores 16 bytes per row in the frame (frames with
`UNBOUNDED PRECEDING` never call `xInverse` but still record, i.e. 16 bytes
per partition row, capped at `SHADOW_CAP`); each refold re-absorbs one
frame. `GROUP BY` stores at most 64 rows (1 KiB) per group, freed as soon as
the group passes 64 rows, plus the core state. Measured refold counts are in
`experiments/sqlite/results.json` (`window.cells[].stats`).

Precision: the extension uses the core's guard as shipped, so all hosts
share one arithmetic. Since the 2026-11-05 core change the guard is a
running first-order error bound on `z` and `u` and the guarantee is
absolute in units of max|v|, which is also the PROTOCOL's window
tolerance (`|err| <= 1e-9 * max|v|`). The measured drift on IMDb, new
guard vs the old z-only guard, is in `../../experiments/sqlite/README.md`
(`guard_bound.json`, `guard_tau_old_core.json`).

Other frame shapes (`RANGE`, `GROUPS`, `EXCLUDE CURRENT ROW`, `PARTITION
BY`, `FOLLOWING`, `UNBOUNDED`) are exercised in
`tests/test_sqlite.py::test_window_shapes` at eps in {1e-4, 1e-3, 0.02, 1,
0, inf} against a fresh numpy fold per frame.

## Tests

* `cargo test --release`: unit tests of `state.rs` (eps validation,
  O(1) aggregate mode, sliding frames at sharp eps against a fresh fold,
  refusal on a long first frame, FIFO check, empty frames, `-inf` rows) and
  `dot.rs`; `softavg_agg`'s state is bit-identical to `softavg` in
  aggregate mode.
* `python3 tests/test_sqlite.py`: loads the release .so into Python's
  sqlite3 (3.37.2 has `load_extension`), loads
  `../core/conformance/vectors.json` into a table and checks GROUP BY at
  every eps for both `softavg` and `softavg_agg` (rtol 1e-12, null ->
  NULL) and the window cases (ROWS 49 PRECEDING, `|err| <= 1e-9 *
  max|v|`); `softavg_agg` semantics (bitwise equal to `softavg`, errors,
  refused under `OVER`); then NULL / error / NaN semantics, every frame
  shape, the refusal path, `softagg_dot` (including over sqlite-vec
  `vec_f32` blobs and against `1 - vec_distance_cosine`). It prints the
  number of checks passed and failed and exits non-zero on any failure.
* `examples/guard_drift.rs` (synthetic data through `WinState`) and
  `examples/guard_bound.rs` (IMDb scores through `InvAcc`, checks the
  error against `value_error_bound()`): drift diagnostics.

## Files

```
Cargo.toml            cdylib + rlib; softagg-core = { path = "../core" }; libsqlite3-sys loadable_extension
src/lib.rs            SQLite C-API glue, registration, entry points
src/state.rs          WinState: InvAcc + eps + lazy frame shadow + refold/refuse
src/dot.rs            float32 BLOB inner product
tests/test_sqlite.py  conformance + semantics tests through Python's sqlite3
examples/             guard_drift.rs, guard_bound.rs (diagnostics)
```

## Open issues

* SQLite runs one statement on one thread, so there is no partial-state
  combine to exercise: `InvAcc::merge` is unused here.
* `softavg` under `GROUP BY` pays the `InvAcc` error-bound bookkeeping and
  the first-64-row record because the mode is unknown at `xStep`;
  `softavg_agg` avoids both at the cost of a second name. The measured gap
  is in the experiment README (`softavg` vs `softavg_agg` rows). The
  planner cannot pick between them for the user.

# bruce_pg 0.2.0 — `softavg` as a native PostgreSQL aggregate (paper_icde)

`softavg(value, score, eps)` = `sum_r e^{score_r/eps} value_r / sum_r e^{score_r/eps}`
per group, computed in the max-anchored form so `e^{score/eps}` is never
materialised. This is the PostgreSQL host of `../core` (softagg-core): the
absorb / merge / finalize / remove arithmetic is that crate's, shared with the
DuckDB, SQLite and ClickHouse extensions; this crate is only the PostgreSQL
call site.

Origin: copied from `bruce_tool/bruce-pg` 0.1.2 (commit 51b62f1, clean) on
2026-11-05. `bruce_tool` itself was not modified.

## What changed from 0.1.2 (results are identical)

| | 0.1.2 | 0.2.0 |
|---|---|---|
| state | `float8[5]` varlena, decoded, re-allocated and re-encoded on **every row** | `internal`: a 40-byte struct (`state::AggState`) palloc'd **once per group** in the aggregate memory context, updated in place |
| parallel | partial states are arrays, no serial/deserial | `SERIALFUNC`/`DESERIALFUNC` (33 bytes), `COMBINEFUNC` copies into the aggregate context when it adopts a deserialized state; still `PARALLEL SAFE` |
| arithmetic | vendored `ScalarAcc` | `softagg_core::ScalarAcc` (path dependency `../core`) |
| windows | frame re-aggregated per row | moving-aggregate form: `MSFUNC`/`MINVFUNC`/`MFINALFUNC` over `softagg_core::InvAcc` (80-byte state), O(1) per row |
| maintenance | REFRESH only | `softavg_ivm_create`: per-group `InvAcc` state table maintained by statement triggers (below) |
| support functions | pgrx `#[pg_extern]` wrappers | raw V1 C-ABI functions (`#[pg_guard]`), SQL declared by hand; per row: read 4 `NullableDatum`s, compare `eps`, `ScalarAcc::absorb` |

Unchanged: NULL discipline, the NaN / ±Inf / `eps = 0` / `eps = 'infinity'`
policy (below), every error message, `softavg_state`'s `float8[5]` result,
and the SQL-callable array helpers `bruce_softavg_sfunc` /
`bruce_softavg_combine` / `bruce_softavg_final`.

## What it ships

```sql
CREATE AGGREGATE softavg(float8, float8, float8) (
    SFUNC = softavg_sfunc, STYPE = internal, SSPACE = 40,
    COMBINEFUNC = softavg_combine,
    SERIALFUNC = softavg_serialize, DESERIALFUNC = softavg_deserialize,
    FINALFUNC = softavg_finalfn,
    MSFUNC = softavg_msfunc, MINVFUNC = softavg_minvfunc,
    MSTYPE = internal, MSSPACE = 80, MFINALFUNC = softavg_mfinalfn,
    PARALLEL = SAFE);

-- softavg_state: same transition, FINALFUNC returns [mu, z, u, eps, nan_seen]
CREATE AGGREGATE softavg_state(float8, float8, float8) (...);

softavg_window_stats() -> (fresh_states, absorbs, inverses, punts)  -- per backend
softavg_window_stats_reset()
softavg_core_version() -> text

-- incremental maintenance (next section)
softavg_ivm_create(view, base, group_col, value_expr, score_expr, eps), softavg_ivm_drop(view)
softavg_ivm_create_join(view, base, base_key, bridge, bridge_key, group_col, value_expr, score_expr, eps)
softavg_inv_build(value, score, eps) -> bytea          -- aggregate: serialized InvAcc of a row set
softavg_inv_merge(bytea, bytea), softavg_inv_remove_many(bytea, float8[], float8[], eps)
softavg_inv_value / _exact / _error_bound / _rows(bytea)
softavg_ivm_stats() -> (trigger_statements, groups_recomputed), softavg_ivm_stats_reset()
```

The full definitions are in `src/lib.rs` (`extension_sql!`) and, identically,
in `sql/bruce_pg--0.1.2--0.2.0.sql`.

Temperature semantics (unchanged): `eps > 0` finite — max-anchored softmax
average; `eps = 0` — mean of `value` over the argmax-score set;
`eps = 'infinity'` — plain mean (`== AVG(value)` over complete pairs).
NULL `value` or `score` rows are skipped like `AVG` skips NULLs; an empty
group is NULL; `eps` must be `>= 0`, non-NULL, and constant within a group.

## Moving aggregate (sliding windows)

PostgreSQL uses `MSFUNC`/`MINVFUNC` whenever a window frame's head moves
(`ROWS BETWEEN k PRECEDING AND CURRENT ROW`, `RANGE ... PRECEDING`, ...).
`MINVFUNC` removes the departing row with `InvAcc::remove`, which never
re-anchors. When the core reports the state inexact — the cancellation guard
`z <= 1e-6 z_peak` fired (the departing rows held most of the weight, typical
at sharp `eps`), or the argmax set of an `eps = 0` / `+inf`-score frame
emptied — `MINVFUNC` returns SQL NULL. That is PostgreSQL's documented
"cannot invert" signal: the executor discards the state and re-aggregates the
current frame through `MSFUNC`. `softavg_window_stats()` counts these
"punts".

Details that differ from the plain aggregate, all forced by PostgreSQL's
moving-aggregate contract:

- `MSFUNC` never returns NULL (PostgreSQL rejects that), so the moving state
  is allocated at the first row of a frame even when that row is NULL; `eps`
  is fixed by the first live row.
- NaN rows are **counted** (`nan_rows`), not a sticky bit, so a NaN row that
  leaves the frame stops poisoning it.
- Precision: a frame computed through removals is not bit-identical to a
  fresh fold. softagg-core's `InvAcc` keeps a running first-order error bound
  on `z` and `u` and reports inexact once the bound on the delivered value
  exceeds `max(1e-10, 100 n u) * max|v|` -- an ABSOLUTE guarantee in units of
  the largest |value| (PROTOCOL window rule: `|err| <= 1e-9 * max|v|`). A
  frame whose exact answer is ~0 may therefore come back as 0; that is within
  the bound. (Before the core change the guard was `z <= 1e-6 z_peak`;
  `experiments/postgres/results.json` keeps both window runs.)

Window frames whose head never moves (`ROWS UNBOUNDED PRECEDING`,
`PARTITION BY` without a frame) use the plain `SFUNC` and call the read-only
`FINALFUNC` repeatedly.

## Incremental maintenance (`softavg_ivm_create`)

```sql
SELECT softavg_ivm_create('mv', 'base_table', 'group_col', 'value_expr', 'score_expr', eps);
SELECT * FROM mv;                 -- (group_col, softavg, n)
SELECT softavg_ivm_drop('mv');
```

Creates `mv_state(g PRIMARY KEY, cnt bigint, state bytea)` (a serialized
`MovState` = `InvAcc` + eps + live NaN rows, 74 bytes), the trigger function
`mv_maint()`, four statement-level triggers on the base table (`AFTER
INSERT` / `DELETE` / `UPDATE` with transition tables, `AFTER TRUNCATE`) and the
view `mv`. Per statement:

1. deleted rows (OLD TABLE) are removed from their groups' states with
   `softavg_inv_remove_many` (`InvAcc::remove`), inserted rows (NEW TABLE)
   are folded with the `softavg_inv_build` aggregate and merged
   (`InvAcc::merge`, `INSERT ... ON CONFLICT DO UPDATE`); `UPDATE` is both;
   empty groups are deleted;
2. touched groups whose state is now inexact (softagg-core's error bound) are
   recomputed from the base table in a separate statement of the same
   trigger, so the view never serves a value outside the bound;
3. `softavg_ivm_stats()` counts trigger statements and recomputed groups
   (per backend).

Why this is correct under MVCC: the state is an ordinary table, so every
change commits or rolls back with the statement that caused it (savepoints
included). Concurrent writers to the same group serialize on the group's
state row: the second blocks until the first commits and then applies its
delta to the committed state (READ COMMITTED re-evaluates the UPDATE on the
newest row version); the recompute runs in a later statement than the state
update, so its snapshot (READ COMMITTED) already includes every committed
writer, and writers that commit later apply their deltas on top of it. Under
REPEATABLE READ / SERIALIZABLE the conflicting writer gets a serialization
failure instead. `value_expr` / `score_expr` are arbitrary SQL expressions
over the base table's columns (e.g. `-(emb <#> '[...]')`). Limitations: NULL
group keys are not maintained; `eps` is fixed per view; the expressions are
trusted text (the extension is superuser-only).

## Special float values: NaN, ±Inf, and one deliberate divergence

bruce-core pins a NaN/±Inf policy
(`bruce-core/tests/numerical_edges.rs`, `mod masked_attention_nan_policy`).
bruce-pg mirrors **±Inf exactly** and deliberately **diverges on NaN**.
The reasoning, because the divergence is the interesting part:

### The tension

In bruce-core, NaN *is* the engine's encoding of SQL NULL —
`bruce-query/src/ingest.rs` maps a Parquet NULL to NaN, because an
`ndarray<f64>` has no null bitmap. Skipping NaN there is therefore not
a numerics choice at all; it is NULL handling spelled in the only
alphabet that dtype has.

PostgreSQL has no such shortage. NULL is spelled `NULL`, and
`'NaN'::float8` is a **legitimate value** with defined behaviour:
`AVG` over a column containing NaN returns NaN, `SUM` returns NaN, and
NaN sorts *greatest* under the float8 ordering. Importing bruce-core's
encoding would mean this extension silently deletes a real value that
the database next to it faithfully carries.

### The decision: **PROPAGATE** (option b)

`softavg` returns `'NaN'::float8` for any group in which at least one
qualifying row has a NaN `value` **or** a NaN `score` — in every eps
regime (0, finite, `'infinity'`). Not skipped, not NULL.

Four reasons, in decreasing order of force:

1. **The skip is not part of the monoid, so C2 is untouched.**
   bruce-core says so itself, in `RowAcc::absorb`'s own doc comment:
   *"NaN scores/values never reach `absorb` from the grouped kernels —
   the SQL-NULL skip happens at the call sites."* The skip lives in
   `fold_sequential` / `grouped_softavg`, not in
   absorb/merge/finalize. ±Inf, by contrast, **is** branched on
   *inside* `RowAcc` — which is exactly why ±Inf is mirrored verbatim
   and NaN is not. C2 requires the same monoid on both sides; it does
   not require the same call site, and PG's call site is the aggregate
   transition function.
   Formally, bruce-pg's state is a **product monoid**
   `(mu, z, u) × ({false, true}, ∨)`: the first factor is bruce-core's
   monoid, behaviourally unchanged (`ScalarAcc` still holds precisely
   the value bruce-core's skip would have produced — this is asserted,
   see `test_j_nan_state_component_matches_bruce_core_skip`); the
   second factor is a sticky NaN bit that only the PG-side
   `FINALFUNC` consults. Nothing was added to, or removed from, the
   monoid under test.

2. **Skip destroys information; propagate does not.** A user who wants
   bruce-core's semantics writes
   `WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8` and gets them
   exactly (PG treats `NaN = NaN` as true, so `<>` really does filter
   NaN; core PG has no `isnan()`). A user who wants propagation out of
   a *skipping* aggregate cannot recover it — the aggregate already
   threw the evidence away. The more expressive of two semantics is the
   one that should be the default.

3. **C4 — PG-aligned vocabulary.** `softavg` is an *arithmetic*
   aggregate: its output is a weighted mean of `value`. Its family is
   `AVG`/`SUM`/`corr`/`covar_samp`, all of which propagate NaN, not
   `MAX`/`ORDER BY`, whose ordering vocabulary ("NaN is greatest") is a
   different contract. A user reading
   `SELECT softavg(v, s, 'infinity'), AVG(v) FROM t` should not see the
   two columns disagree about the same NaN.

4. **The pair rule already exists here.** bruce-pg already treats the
   `(value, score)` pair as one datum even at `eps = 'infinity'`, where
   the score is arithmetically unused: a NULL `score` drops the row and
   an all-NULL-score table aggregates to NULL, though `AVG(value)`
   would happily return a number
   (pinned by `test_f_empty_input_is_null_like_avg`). So a NaN `score`
   poisons in *every* regime, including `'infinity'`. One rule, stated
   once, no eps-dependent NULL semantics. (PG's own `regr_avgy(y, x)`
   would *not* propagate a NaN `x`, since `x` never enters its
   arithmetic — we follow this crate's established pair discipline over
   that precedent, deliberately, and pin the difference.)

### What the losing option would have bought, and what this one costs

Option (a) **SKIP** — mirror bruce-core exactly — buys a trivially
1:1 cross-check (every test could feed NaN straight through both
implementations) and one policy sentence for the whole project. Its
cost is what killed it: inside a database that already has NULL, it
takes a real `float8` value the user stored on purpose, deletes it, and
returns a confident number computed from a *different* row set than the
`AVG` in the same `SELECT` list — silently, with no way for the user to
detect or undo it.

The chosen option is not free, and the price is paid here:

- **Bit-parity with bruce-core is gone on NaN inputs.** (0.1.2 pinned
  this with `bruce_core_cross_check` tests against bruce-core itself;
  0.2.0 links softagg-core instead, whose own tests and the shared
  conformance vectors pin the arithmetic, and keeps the in-database pin
  `test_j_nan_propagates_where_bruce_core_skips` plus the pure-Rust
  `nan_is_sticky_and_kept_out_of_the_core_state`.)
- **The state grew** from `float8[4]` to `float8[5]` in 0.1.2:
  `[mu, z, u, eps, nan_seen]`, `nan_seen ∈ {0, 1}`. `softavg_state`
  still returns exactly this array in 0.2.0 (now built by its FINALFUNC
  from the internal state).
- **Round-tripping bruce-query output into PG is now a NULL-encoding
  hazard**: a pipeline that lets `ingest.rs`'s NULL→NaN mapping reach a
  `float8` column will get NaN out of `softavg` where bruce-core gives
  a skip-answer. The fix is on the writer's side — materialise SQL NULL
  into PG, not NaN — and this direction does not exist in the tree
  today (bruce-cdc reads *from* PG).

### ±Inf: mirrored verbatim, no PG conflict

Identical to `RowAcc`, and part of the monoid rather than the call site:

| input | `eps = 0` | finite `eps` | `eps = 'infinity'` |
|---|---|---|---|
| `+Inf` score | is the argmax | argmax collapse; ties among `+Inf` rows average uniformly | counted like any row (score-blind) |
| `-Inf` score | weight 0, skipped | weight 0, skipped | counted like any row (score-blind) |
| all rows `-Inf` | **NULL** | **NULL** | plain mean of values |

`+Inf` collapse survives `COMBINEFUNC`, so a parallel plan whose
workers each hold some `+Inf` rows returns the same uniform mean over
all of them as the serial plan
(`test_j_pos_inf_parallel_combine_matches_serial`). `exp(inf - inf)` is
never evaluated on any path.

### What 0.1.1 actually did (measured, not assumed)

`ScalarAcc` had no `±Inf` branches at all before 0.1.2, and no NaN
policy, so both behaviours were emergent rather than chosen. Replaying
the 0.1.1 recurrences verbatim gives (`results/pg_parity_semantics.json`
holds these numbers; the reproducer is in the CHANGELOG entry):

| rows | eps | 0.1.1 returned | 0.1.2 returns |
|---|---|---|---|
| `+Inf` row beside finite rows | 0.37, 1 | **NaN** | 55.0 |
| two `+Inf` rows (tie) | 0.37, 1 | **NaN** | 55.0 |
| all rows `-Inf` | 0.37, 1 | **NaN** | NULL |
| all rows `-Inf` | 0 | **943.5** | NULL |
| NaN score beside finite rows | 0 | **20.0** (dropped) | NaN |
| NaN score beside finite rows | 0.37 | **NaN** | NaN |
| NaN value beside finite rows | 0 | **20.0** (dropped) | NaN |
| NaN value beside finite rows | ∞ | **NaN** | NaN |

Note the last four rows: 0.1.1's NaN handling was not "skip" and was
not "propagate", it was *whichever the float comparisons happened to
produce*. At `eps = 0` a NaN row silently vanished, because both
`s > mu` and `s == mu` are false for NaN; at finite `eps` the same row
poisoned the accumulator. So this track did not choose between two
working behaviours — it replaced an undefined one, and the choice of
which defined behaviour to install is the argument above.


### Over a many-to-many join: `softavg_ivm_create_join` (overlapping groups)

```sql
-- items(id PK, value, s, ...) and a bridge item_facet(item_id, facet): an item belongs to 0..n facets
SELECT softavg_ivm_create_join('mv', 'items', 'id', 'item_facet', 'item_id', 'facet', 'value', 's', eps);
SELECT * FROM mv;                 -- (facet, softavg, n) == SELECT br.facet, softavg(i.value, i.s, eps), count(*)
                                  --    FROM items i JOIN item_facet br ON br.item_id = i.id GROUP BY 1
SELECT softavg_ivm_drop('mv');
```

`softavg_ivm_create` puts its triggers on one table and reads the group key
from the changed row, so it cannot maintain a GROUP BY over a join (a
trigger on a view does not fire for base-table changes; the old/new rows of
the base table do not carry their groups). `softavg_ivm_create_join` keeps
the same per-group `InvAcc` state table and adds statement triggers on
**both** tables:

- a base-row edit is joined with the bridge: **one state delta per
  membership** (a movie in 5 genres touches 5 states; `remove` for the old
  row, `merge` for the new one, `UPDATE` both; a key change moves the
  memberships);
- a bridge-row edit is joined with the base: one delta for that membership;
- the touched state rows are locked in key order first (multi-group writers),
  empty groups are deleted, and groups whose state is inexact after the
  removals are recomputed from the join for that group only;
- the value/score expressions are evaluated over the base table's own
  columns (old/new transition rows on the base side, the base table on the
  bridge side), so column names shared with the bridge are not ambiguous.

Join-specific concurrency: a base row and a bridge row inserted by two
concurrent transactions would each join with the other table's committed
state only, and the membership would be lost from the view. Base-side
triggers therefore take a shared, bridge-side triggers an exclusive
transaction-level advisory lock (key derived from the state table's oid)
before computing their delta; under READ COMMITTED each delta query runs
after the lock is granted and sees the other side's committed rows. Base
writers run concurrently with each other. Bridge writers, and base writers
under REPEATABLE READ / SERIALIZABLE, also update a one-row `<view>_sync`
table, so a REPEATABLE READ base writer whose snapshot predates a committed
bridge write gets a serialization failure (40001) instead of silently
missing the membership; bridge writes under REPEATABLE READ / SERIALIZABLE
are refused (0A000) because their snapshot cannot see base rows committed
after it. Cascading foreign keys from the bridge to the base are refused at
creation: the cascaded bridge delete fires before the base statement trigger
and neither trigger can see the other side's old rows. Bag semantics: a
duplicated bridge row counts twice, exactly as the join does. Measured in
`../../experiments/multilabel/` (per-edit latency and recomputes vs fan-out,
two-session races).

## Install

```bash
export LIBCLANG_PATH=$HOME/.local/lib/python3.10/site-packages/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/12/include"
export PATH="$HOME/miniforge3/bin:$PATH"
cd paper_icde/engines/postgres
# release build (what every experiment uses), into a given server:
nice -n 10 cargo pgrx install --release --pg-config ~/.pgrx/17.10/pg-release-install/bin/pg_config
# the hand-written 0.1.2 install script is only needed to test the upgrade path:
cp sql/bruce_pg--0.1.2.sql "$(~/.pgrx/17.10/pg-release-install/bin/pg_config --sharedir)/extension/"
```

`cargo pgrx install` copies the `sql/*--*--*.sql` upgrade scripts but not
`sql/bruce_pg--0.1.2.sql` (a full install script of an old version; it is the
script pgrx generated for 0.1.2, kept so `CREATE EXTENSION bruce_pg VERSION
'0.1.2'` works for the upgrade check).

Note: `cargo pgrx test pg17` installs a debug `pg_test` build of the
extension into the pgrx-managed PostgreSQL (`~/.pgrx/17.10/pgrx-install`),
replacing whatever `bruce_pg.so` is there.

## Demo: anchored retrieval scoring in plain SQL

```sql
CREATE TABLE reviews(movie_id int, rating float8, plot_score float8);
-- plot_score = similarity of the review's embedding to a query anchor.

SELECT movie_id,
       softavg(rating, plot_score, 0.05)  AS anchored_rating,
       AVG(rating)                        AS flat_rating
FROM   reviews
GROUP  BY movie_id;
```

With pgvector installed the score argument is just the similarity
expression: `emb <#> :q` is the *negative* inner product, so pass
`-(emb <#> :q)` (rescaled as needed) as `score`:

```sql
SELECT softavg(rating, -(emb <#> :query_emb), 0.05) FROM reviews;
```

No index, no staging table: any expression yielding a `float8` score
works, and the sharper `eps` gets, the more the aggregate behaves like
`ORDER BY score DESC LIMIT 1` — while staying a single scan that PG
parallelizes with the monoid's own `merge`.


## Versioned upgrades

- `sql/bruce_pg--0.1.0--0.1.1.sql`, `sql/bruce_pg--0.1.1--0.1.2.sql` — from
  bruce-pg, unchanged.
- `sql/bruce_pg--0.1.2--0.2.0.sql` — creates the internal-state, moving and
  incremental-maintenance support functions and the helper functions, then
  `CREATE OR REPLACE AGGREGATE` for `softavg` and `softavg_state` (keeps
  their OIDs, so views and materialized views over `softavg` survive). No
  result changes. `experiments/postgres/upgrade_check.py` runs it in a real
  database: views and materialized views keep their bits across the upgrade
  and `REFRESH`, and the catalog of the upgraded database equals a fresh
  0.2.0 install object by object.
- `test_i_upgrade_path_alter_extension` stages a synthetic `0.2.0--0.2.1`
  script and drives `ALTER EXTENSION ... UPDATE` in-database.

## Tests

```bash
nice -n 10 cargo pgrx test pg17
```

Counts and the log digest of the last run are in
`../../experiments/postgres/results.json` (`tests`).

Pure-Rust (`src/state.rs`, no PostgreSQL): MovState serialization and merge, NaN stickiness and its survival
across merges, eps rules and error texts, serialization round trip,
moving-state NaN counting, the guard punting at sharp eps, and the
conformance window vectors driven through `MovState` exactly as PostgreSQL
drives `MSFUNC`/`MINVFUNC` (recompute on punt).

In-database (`#[pg_test]`), 0.2.0 additions:

- `test_conformance_vectors_group_by` — loads
  `../core/conformance/vectors.json` into a table (JSON parsed by
  PostgreSQL's `jsonb`), runs `GROUP BY g` at every eps of the file, serial
  and through a forced parallel plan (`Gather` + `Partial` asserted), and
  compares with `expected` at rtol 1e-12; `null` must be SQL NULL.
- `test_conformance_vectors_window` — the window cases
  (`ROWS BETWEEN 49 PRECEDING AND CURRENT ROW` over ids < 2000) at rtol
  1e-9, and asserts the moving path was used (`inverses > 0`).
- `test_k_catalog_declares_internal_state_and_moving_form` — `pg_aggregate`
  declares internal state, serial/deserial, combine, moving functions,
  read-only finalizers; `PARALLEL SAFE`; strictness of each support function.
- `test_k_grouped_parallel_matches_serial` — 300k rows, ~1000 groups (some
  seen by only one worker), `+inf` scores, four eps regimes: forced parallel
  plan equals the serial plan group by group.
- `test_k_hashagg_spill_matches_sort_agg` — hash aggregation spilling at
  `work_mem = 64kB` vs sort aggregation.
- `test_k_state_array_helpers_still_compose` — `softavg_state` +
  `bruce_softavg_combine` / `_final` re-associate; an aggregate built from
  the kept 0.1.2 float8[] functions agrees with 0.2.0 bit for bit.
- `test_w_window_matches_frame_recompute` — 3000 rows with NULL, NaN, `+inf`,
  `-inf` scores and ties; six eps regimes x frame widths 0/1/7/49; every
  window value equals a fresh per-frame `softavg` (rtol 1e-9, NULL = NULL,
  NaN = NaN).
- `test_w_moving_path_counters` — exactly one fresh state per partition and
  one inverse per departing row at eps = 1, zero punts; at eps = 1e-3 the
  guard fires (punts > 0, fewer than inverses) and results stay correct.
- `test_w_eps_change_rejected_in_window`, `test_w_growing_frame_and_partition_window`.
- `test_ivm_tracks_inserts_deletes_updates_at_every_regime` -- six eps regimes;
  argmax deletes, multi-row statements, updates moving rows between groups,
  NULL value/score, `-inf`, NaN entering and leaving, a group appearing and
  disappearing, a NULL group key: view == fresh `GROUP BY` (|err| <= 1e-9 x
  value scale, NULL/NaN agree, same groups and counts) after every step.
- `test_ivm_recomputes_inexact_groups`, `test_ivm_rollback_restores_state`
  (state bytes identical after a rolled-back subtransaction),
  `test_ivm_truncate_and_drop`.
- `test_ivm_join_tracks_base_and_bridge_edits_at_every_regime` -- base
  `(id, v, s)` x bridge `(item, g)` with 0..4 overlapping memberships per
  item, six eps regimes: argmax base deletes (fan-out), multi-row base
  statements, value/score updates, a key change, inserts that pick up
  dangling bridge rows, NULL value/score, `-inf`, bridge inserts / deletes /
  duplicates / group moves / item moves, a NULL group, NaN entering through
  the base and leaving through the bridge: view == fresh `GROUP BY` over the
  join (values, groups and membership counts; no inexact state visible).
- `test_ivm_join_fanout_recomputes_each_dominated_group` -- one delete of an
  item dominating 9 groups at eps = 1e-3: one trigger statement, exactly 9
  groups recomputed; the re-insert picks its 9 memberships up again.
- `test_ivm_join_rollback_truncate_and_drop`, `test_ivm_join_cascading_fk_rejected`.

The 0.1.2 in-database tests (`test_a_*` .. `test_j_*`, NULL / empty / strictness /
unicode grouping / eps domain / NaN / ±Inf / parallel combine) are kept
unchanged except the version strings in `test_i_*`.

Why the per-row logic lives in the pgrx-free `src/state.rs`: `cargo pgrx
test` relies on `--gc-sections` to prune pgrx-pg-sys's unreachable `extern
"C"` declarations from the test binary; a plain `#[test]` that reaches pgrx's
`error!` fails to link (`undefined symbol: errstart`).

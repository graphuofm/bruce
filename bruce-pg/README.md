# bruce-pg — `softavg` as a native PostgreSQL aggregate

The shortest possible proof of constitution **C2**: the `(mu, z, u)`
soft-average monoid of `bruce-core/src/mask.rs` maps **1:1** onto
PostgreSQL's aggregate contract.

| bruce-core (`RowAcc`) | PostgreSQL aggregate | bruce_pg function      |
|-----------------------|----------------------|------------------------|
| `absorb`              | `SFUNC`              | `bruce_softavg_sfunc`  |
| `merge`               | `COMBINEFUNC`        | `bruce_softavg_combine`|
| `finalize`            | `FINALFUNC`          | `bruce_softavg_final`  |

The state is a plain `float8[5] = [mu, z, u, eps, nan_seen]` (empty =
`NULL`), so partial states cross worker boundaries as ordinary varlena
values — no `serialfunc`/`deserialfunc`, and the aggregate is
`PARALLEL SAFE` with PG driving the partition-reduce itself.

`RowAcc` is file-private in bruce-core, so `ScalarAcc` in `src/lib.rs`
restates the same absorb/merge/finalize recurrences for scalar values
(`d_v = 1`); the `bruce_core_cross_check` tests pin it against
`bruce_core::masked_attention` to 1e-12.

The three support functions are thin encode/decode wrappers over
`PgAcc`, which is `ScalarAcc` in a **product** with one extra monoid —
a sticky NaN bit, `(mu, z, u) × ({false, true}, ∨)`. That second
factor is the whole of bruce-pg's deliberate NaN divergence from
bruce-core; `ScalarAcc` itself is left exactly as bruce-core wrote it,
which is what keeps C2 honest. See “Special float values” below.

## What it ships

```sql
-- softavg(value float8, score float8, eps float8) -> float8
CREATE AGGREGATE softavg(float8, float8, float8) (
    SFUNC       = bruce_softavg_sfunc,
    STYPE       = float8[],
    COMBINEFUNC = bruce_softavg_combine,
    FINALFUNC   = bruce_softavg_final,
    PARALLEL    = SAFE
);

-- softavg_state: same transition without the finalfunc — returns the
-- raw monoid element [mu, z, u, eps, nan_seen] for re-association.
CREATE AGGREGATE softavg_state(float8, float8, float8) (
    SFUNC       = bruce_softavg_sfunc,
    STYPE       = float8[],
    COMBINEFUNC = bruce_softavg_combine,
    PARALLEL    = SAFE
);
```

Temperature semantics (identical to bruce-core):

- `eps > 0` finite — max-anchored softmax average `sum w_r v_r / sum w_r`,
  `w_r = e^{score_r/eps}`, computed without ever materialising
  `e^{score/eps}`. The naive SQL spelling
  `SUM(EXP(s/eps)*v)/SUM(EXP(s/eps))` raises
  `value out of range: overflow` already at `eps = 1e-4` with scores in
  `[-1, 1]`; `softavg` returns the exact limit value.
- `eps = 0` — tropical endpoint: mean of `value` over the argmax-score
  set (uniform tie handling).
- `eps = 'infinity'` — plain mean; `softavg(v, s, 'infinity') == AVG(v)`.

NULL discipline matches `AVG`: rows with NULL `value` or `score` are
ignored; an empty group returns NULL. `eps` must be constant within a
group (enforced with an error, not silently mixed).

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

- **Bit-parity with bruce-core is gone on NaN inputs.** The
  `bruce_core_cross_check` tests therefore feed NaN-free data (they
  cover eps regimes, partitions, and ±Inf), and the divergence gets its
  own explicit pin naming this decision:
  `nan_input_diverges_from_bruce_core_by_design`, plus the in-database
  `test_j_nan_propagates_where_bruce_core_skips`.
- **The state grew** from `float8[4]` to `float8[5]`:
  `[mu, z, u, eps, nan_seen]`, `nan_seen ∈ {0, 1}`. `softavg_state`
  exposes it, so this is an observable change → version `0.1.2`.
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

## Install

```bash
# toolchain (pinned: cargo-pgrx 0.19.x needs rustc >= 1.96)
cargo install cargo-pgrx --version 0.18.1 --locked
cargo pgrx init --pg17 download   # compiles PostgreSQL 17.x under ~/.pgrx
# version matrix: add PG 16 next to an existing pg17 by passing BOTH
# (init rewrites config.toml with exactly the versions given):
#   cargo pgrx init --pg16 download \
#       --pg17 ~/.pgrx/17.10/pgrx-install/bin/pg_config \
#       --configure-flag=--without-icu --configure-flag=--without-readline

cd bruce_tool/bruce-pg
cargo pgrx install --release      # into the pgrx-managed pg17
# or against an existing server:  cargo pgrx install --release -c /path/to/pg_config
```

```sql
CREATE EXTENSION bruce_pg;
```

Build notes for minimal hosts (what this machine needed): PG 17
tarballs require `bison`/`flex` (distprep removed — PG 16 tarballs
still ship the pregenerated parser files, so 16 builds without either;
`flex` lives at `~/miniforge3/bin/flex` here, put that dir on PATH);
`cargo pgrx init`
flags `--configure-flag=--without-icu --configure-flag=--without-readline`
skip missing system libs; bindgen needs a `libclang.so`
(`LIBCLANG_PATH`) — on this machine a durable copy lives at
`/home/jding/.local/lib/python3.10/site-packages/clang/native`
(installed via `pip install --user libclang`), so use
`export LIBCLANG_PATH=$HOME/.local/lib/python3.10/site-packages/clang/native`
— plus `BINDGEN_EXTRA_CLANG_ARGS="-isystem
/usr/lib/gcc/x86_64-linux-gnu/12/include"` if clang's builtin headers
are absent.

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

The extension is versioned (`Cargo.toml` `version` -> control file
`default_version`). pgrx regenerates the full `bruce_pg--<ver>.sql`
schema every build but does **not** author cross-version scripts;
hand-written upgrade scripts live in `sql/` and
`cargo pgrx install`/`test` copies `sql/*.sql` into
`<sharedir>/extension/` next to the generated schema — verified on
this machine (the `cargo pgrx test` log copies both scripts), so a
packaged install carries the upgrade chain and
`ALTER EXTENSION bruce_pg UPDATE` just works.

- `sql/bruce_pg--0.1.0--0.1.1.sql` — inert; 0.1.1 is the release that
  introduced the upgrade machinery. A trivial `COMMENT`.
- `sql/bruce_pg--0.1.1--0.1.2.sql` — **observable behaviour change**
  (this track): `±Inf` policy, NaN propagation, and the
  `float8[4] -> float8[5]` state. No `CREATE`/`ALTER` is needed —
  the aggregates, their support functions and `STYPE = float8[]` are
  unchanged, and the new behaviour arrives with the shared library —
  so the script's body is documentation plus refreshed `COMMENT`s. It
  is the place the upgrade is *explained* to a DBA, including the one
  operational consequence: `REFRESH` any materialized view aggregating
  data with NaN or `±Inf` scores, because its values may change.

`test_i_upgrade_path_alter_extension` drives the real machinery
in-database: it stages a synthetic `0.1.2--0.1.3` script into the
sharedir, runs `ALTER EXTENSION bruce_pg UPDATE TO '0.1.3'`, and
asserts both `pg_extension.extversion` and the script's observable
effect.

## Tests

`cargo pgrx test pg17` and `cargo pgrx test pg16` — **39 tests, green
on both** PostgreSQL 16.14 and 17.10 (TESTING_MATRIX workstream 13):
30 in-database `#[pg_test]`s + 9 pure-Rust cross-checks.

Both invocations need the host build environment noted under
“Install”:

```bash
export LIBCLANG_PATH=$HOME/.local/lib/python3.10/site-packages/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-linux-gnu/12/include"
export PATH="$HOME/miniforge3/bin:$PATH"   # flex, for the PG 17 build
cargo pgrx test pg17 && cargo pgrx test pg16
```

SQL tests through the extension:

- `test_a_equal_scores_match_avg` — equal scores ⇒ `softavg == AVG` at
  `eps ∈ {0, 0.5, 1, ∞}`.
- `test_b_matches_sql_softmax_average` — finite `eps` matches the naive
  SQL softmax average to 1e-9 on a tame fixture.
- `test_c_naive_softmax_overflows` / `test_c_softavg_survives_sharp_eps`
  — the killer: at `eps = 1e-4`, naive SQL raises
  `value out of range: overflow`, `softavg` returns the exact answer.
- `test_d_parallel_matches_serial` — 200k rows, parallel workers forced
  on, plan asserted to contain `Gather` + `Partial Aggregate`, result
  equal to the single-worker run.
- `test_e_partition_combine_identity` — `final(combine(state(A),
  state(B))) == softavg(A ∪ B)` written literally in SQL.
- NULL discipline, both endpoints (`eps = 0`, `eps = ∞`), and the
  mid-group `eps`-change error.

Semantics-conformance tests (TESTING_MATRIX workstream 14 — softavg
pinned against what `AVG` promises, constitution C4):

- `test_f_null_skip_equals_avg_on_nonnull_subset` — NULL `value` or
  `score` rows are skipped exactly like AVG skips NULL inputs: with
  equal scores, softavg over the NULL-y table `== AVG` over the
  both-non-NULL subset at `eps ∈ {0, 0.5, ∞}`.
- `test_f_empty_input_is_null_like_avg` — empty table, `WHERE false`,
  and all-NULL inputs each yield SQL NULL (one row, not zero; never 0,
  never an error), side by side with AVG. Also pins the pair rule:
  softavg's qualifying input is the complete `(value, score)` pair, so
  a table with values but only NULL scores is "empty" to softavg even
  though `AVG(value)` is not.
- `test_f_strictness_declaration_audit` — catalog-level audit that
  SFUNC/COMBINEFUNC/FINALFUNC are declared non-strict, which is what
  their behavior implements (PG would reject a strict 3-arg SFUNC with
  a NULL initcond, and a strict SFUNC could not raise on NULL `eps`).
- `test_f_null_eps_on_live_row_errors` — NULL `eps` reaching a live
  row is `ERROR: softavg: eps must not be NULL`, never a guessed
  temperature.
- `test_g_text_group_by_unicode_parity` — softavg does not group; PG
  does, unchanged: unicode `GROUP BY` keys collapse/separate exactly
  as for AVG (equal '猫' rows collapse; NFC `café` vs NFD
  `cafe`+U+0301 stay distinct groups), and every group's softavg
  equals its AVG under equal scores.
- `test_h_negative_eps_is_sql_error` — `eps < 0` is a clean
  `ERROR` (same domain as bruce-core's `Eps::new`), not a crash.
- `test_h_eps_zero_two_way_tie_is_tied_mean` — `eps = 0` with two rows
  tied at the max score returns the mean of exactly the tied rows, and
  the raw `softavg_state` shows `[mu, z, u] = [max, 2, tied-sum]`.
- `test_i_upgrade_path_alter_extension` — see “Versioned upgrades”.

Special-float tests (the `test_j_*` family — see “Special float
values” for the policy these pin, and why it is what it is):

- `test_j_nan_value_propagates` / `test_j_nan_score_propagates_in_every_regime`
  — a NaN in either argument makes the group NaN at
  `eps ∈ {0, 0.5, ∞}`. Each asserts PG's own comparand in the same
  test: `AVG(v)` is NaN in the first (agreement), and finite in the
  second (the intended, argued divergence — the score is part of the
  datum).
- `test_j_nan_propagates_where_bruce_core_skips` — the divergence
  named explicitly: bruce_pg returns NaN, and
  `WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8` reproduces
  bruce-core's skip answer exactly (30.0 at `eps = 0`, 20.0 at
  `eps = ∞`). This is the recoverability argument, executable.
- `test_j_nan_state_component_matches_bruce_core_skip` — the C2
  argument made observable: `softavg_state` is `float8[5]`, slots
  1..3 are **bit-identical** with and without the NaN row present, and
  only slot 5 (the sticky bit) differs. The monoid was not modified;
  a second monoid was multiplied alongside it.
- `test_j_all_nan_group_is_nan_not_null` — all-NaN input is NaN, not
  NULL (the rows qualified), unlike all-NULL input. AVG agrees.
- `test_j_nan_survives_parallel_combine` — 200k rows, exactly **one**
  NaN, forced parallel plan (`Gather` + `Partial Aggregate` asserted):
  the one worker that saw it still poisons the final answer, because
  the sticky bit is OR-ed in `COMBINEFUNC`. The same plan without that
  row is finite.
- `test_j_pos_inf_score_collapses_to_argmax` — `+Inf` dominates finite
  scores, and `+Inf` ties average uniformly, at `eps ∈ {0, 0.37, 1}`.
- `test_j_pos_inf_parallel_combine_matches_serial` — the collapse
  survives re-association two ways: an explicit SQL `combine` of two
  partial states that **each** hold a `+Inf` row (the `mu = +Inf` on
  both sides branch), the asymmetric one-sided case, and a forced
  parallel plan over 200k rows agreeing with the serial run.
- `test_j_neg_inf_is_weight_zero_and_all_neg_inf_is_null` — `-Inf`
  weighs 0; an all-`-Inf` group is SQL NULL (`IS NULL`, one row).
- `test_j_eps_inf_is_score_blind_for_infinite_scores` — at
  `eps = 'infinity'` the `±Inf` scores are not consulted: plain mean,
  equal to `AVG`, and the all-`-Inf` group is a number, not NULL.
- `test_j_nan_dominates_mixed_infinities` — NaN beats `+Inf`, and a
  NaN over an *empty* monoid factor (every other row `-Inf`) is NaN
  rather than NULL.
- `test_j_nan_row_still_honours_eps_domain` — a NaN row is a *live*
  row, so `eps = -0.5` still raises; contrast NULL rows, which are
  skipped before the `eps` checks.

Pure-Rust cross-checks (`bruce_core_cross_check`, 9 tests) pin
`ScalarAcc` to `bruce_core::masked_attention` sequentially and under
arbitrary partitions, now including `±Inf` fixtures
(`inf_scores_match_bruce_core_sequentially_and_under_partitions`,
`all_neg_inf_group_is_empty_like_bruce_cores_uncovered`) and the
divergence itself (`nan_input_diverges_from_bruce_core_by_design`,
which asserts the monoid factor **equals** bruce-core's while the PG
finalize returns NaN, plus `nan_bit_survives_combine_from_either_side`,
`all_nan_group_is_nan_not_null`, `state_encoding_round_trips_both_factors`).

Note for anyone adding pure-Rust tests here: `cargo pgrx test` relies
on the linker's `--gc-sections` pass to prune pgrx-pg-sys's unreachable
`extern "C"` declarations out of the test binary. A plain `#[test]`
that calls a `#[pg_extern]` function (or anything reaching pgrx's
`error!`) makes those references live and the test binary fails to
link with a wall of `undefined symbol: errstart`. That is why the
transition logic lives in the pgrx-free `PgAcc` / `decode_checked`,
with the `#[pg_extern]`s as thin encode/decode wrappers over it.

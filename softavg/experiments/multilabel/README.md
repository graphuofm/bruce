# experiments/multilabel — overlapping facets (complete IMDb genre memberships) in PostgreSQL

**Timings are smoke numbers on a shared machine** (`results.json` ->
`timing_status = "smoke, machine shared"`). Every number in the generated
block below comes from `results.json` via `summarize.py`.

Closes INTEGRATION_ISSUES C (multi-label part) and the question of
overlapping facet memberships need explicit accounting of
membership counts, update fan-out and per-edit work; use IMDb's complete
genre memberships).

## Data: where the complete memberships come from

The IMDb corpus (`/data/softavg/corpora/imdb`, read-only) has
one genre per movie: `archive/experiments/cidr_one_query/embed_titles.py`
took `MIN(movie_info.info)` over info_type `genres`, i.e. the alphabetically
first genre (the path `/home/jding/bruce/experiments/` in `build_imdb.py` was
moved to `/home/jding/bruce/archive/experiments/` on 2026-10-05). The complete
memberships are in the JOB/IMDb dump that is still on disk
(`archive/datasets/job_imdb/imdb.tgz`, `movie_info.csv`, `info_type_id = 3`).
`build_bridge.py` streams `movie_info.csv` out of the tarball (nothing is
extracted or downloaded) and writes `bridge.parquet` (`item_id, facet`, one row
per movie x genre). Movies with no genre row keep the corpus' `(none)` facet,
so every corpus row has at least one membership, and for every movie the
corpus' single label is the minimum of its multi-label set (checked: 0
mismatches). `bridge.parquet` is 2 MB and kept (it is the derived dataset).

## What is measured

1. **GROUP BY over the join** (`bench_join.py`): `imdb JOIN imdb_genre USING
   (id) GROUP BY imdb_genre.facet` vs the single-label `GROUP BY imdb.facet`,
   with the PROTOCOL spellings naive / shift / clamp / softavg, precomputed
   scores of the first 3 PROTOCOL queries, eps grid, filter none and
   `filter_num >= 2000`, 0 and 8 workers; numpy float64 reference over the
   multiset the join produces.
2. **Maintenance over the join** (`maint_join.py`): `softavg_ivm_create`
   cannot maintain a join view, so bruce_pg gained `softavg_ivm_create_join`
   (`../../engines/postgres/README.md`): statement triggers on both the base
   table and the bridge; one base-row edit fans out to one state delta per
   membership. Arms: `softavg_ivm_join`, `softavg_ivm_single` (the same
   base-row stream through the single-label view: fan-out 1), pg_ivm IMMV of
   the linear `SUM(EXP())` form over the same join, `REFRESH MATERIALIZED
   VIEW` at every checkpoint, recompute-on-read. Per op: latency, fan-out
   (the movie's live memberships), groups recomputed; per checkpoint: drift
   vs a fresh query. A fan-out sweep deletes and re-inserts up to 100 movies
   for every exact fan-out 1..13 (per-edit latency and recomputes vs
   fan-out). Transactional checks: rollback, base-vs-bridge two-session races
   in both orders, a REPEATABLE READ race, a bridge write under REPEATABLE
   READ.

## Results

<!-- BEGIN GENERATED (summarize.py) -->
_All timings: smoke, machine shared._

- server: PostgreSQL 17.10 on x86_64-pc-linux-gnu, build **release**, extensions {'bruce_pg': '0.2.0', 'pg_ivm': '1.16', 'plpgsql': '1.0', 'vector': '0.8.7'}, bruce_pg.so md5 `a4a650d92dff8ced981633b2a30cafa6`; extension build **release**, source sha256 `afc0bbe5a5728a5f...`
- `cargo pgrx test pg17`: 56 passed, 0 failed (8 pure-Rust + 48 in-database), at 2026-11-06T14:32:44; join-view tests: ivm_join_cascading_fk_rejected=ok, ivm_join_tracks_base_and_bridge_edits_at_every_regime=ok, ivm_join_rollback_truncate_and_drop=ok, ivm_join_fanout_recomputes_each_dominated_group=ok
- 0.1.2 -> 0.2.0 upgrade re-checked with `softavg_ivm_create_join` in both the install and the upgrade script (`../postgres/upgrade_check.py`): pass=True, catalog objects 34, diff vs fresh install 0, views bit-identical True
- machine at start of the GROUP BY matrix: quiet=False, load1=16.94

### Bridge: complete IMDb genre memberships

459,865 movies, 706,206 memberships, 30 groups; 162,218 movies have no genre row (kept as the corpus' `(none)` facet); memberships per movie mean 1.54, median 1, p90 3, p99 5, max 13; the corpus' single label equals the alphabetically first genre for every movie (mismatches: 0).

| memberships per movie | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| movies | 305,129 | 94,120 | 37,559 | 16,879 | 4,813 | 1,071 | 221 | 60 | 9 | 1 | 2 | 1 |

| group | multi-label rows | single-label rows | ratio |
|---|---|---|---|
| (none) | 162,218 | 162,218 | 1.0 |
| Drama | 100,017 | 64,955 | 1.5 |
| Comedy | 77,578 | 64,410 | 1.2 |
| Short | 64,006 | 13,923 | 4.6 |
| Documentary | 39,180 | 34,574 | 1.1 |
| Romance | 25,266 | 2,462 | 10.3 |
| Action | 25,012 | 25,012 | 1.0 |
| Thriller | 22,515 | 3,401 | 6.6 |
| Crime | 20,408 | 12,756 | 1.6 |
| Animation | 19,382 | 15,905 | 1.2 |
| Family | 18,620 | 4,111 | 4.5 |
| Horror | 16,407 | 8,726 | 1.9 |
| Adventure | 15,783 | 9,538 | 1.7 |
| Adult | 11,515 | 11,458 | 1.0 |
| Sci-Fi | 11,276 | 2,041 | 5.5 |
| Fantasy | 11,153 | 1,991 | 5.6 |
| Music | 11,149 | 3,677 | 3.0 |
| Mystery | 10,277 | 1,609 | 6.4 |
| Musical | 6,766 | 1,501 | 4.5 |
| War | 6,546 | 422 | 15.5 |
| History | 6,058 | 706 | 8.6 |
| Biography | 5,880 | 5,469 | 1.1 |
| Western | 5,580 | 2,279 | 2.4 |
| Sport | 5,019 | 1,454 | 3.5 |
| Reality-TV | 3,777 | 2,680 | 1.4 |
| Game-Show | 1,586 | 1,055 | 1.5 |
| Talk-Show | 1,562 | 828 | 1.9 |
| News | 1,207 | 664 | 1.8 |
| Film-Noir | 462 | 40 | 11.6 |
| Lifestyle | 1 | 0 | - |

### GROUP BY, single-label (459,865 aggregated rows; precomputed s0, q0, no filter)

| plan \ eps (ms, 0 / 8 workers) | 0.0001 | 0.001 | 0.02 | 0.1 | 1.0 | inf |
|---|---|---|---|---|---|---|
| naive | ERR overflow / ERR overflow | ERR overflow / ERR overflow | 115 / 20.4 | 114 / 19.2 | 83.0 / 21.7 | 118 / 20.8 |
| shift | ERR underflow / ERR underflow | ERR underflow / ERR underflow | 240 / 186 | 253 / 126 | 172 / 155 | 260 / 131 |
| clamp | 175 / 123 | 182 / 125 | 175 / 197 | 281 / 192 | 279 / 179 | 240 / 160 |
| softavg | 106 / 18.3 | 90.0 / 19.7 | 113 / 30.3 | 115 / 32.0 | 106 / 17.7 | 74.4 / 17.4 |

### GROUP BY, multi-label (706,206 aggregated rows; precomputed s0, q0, no filter)

| plan \ eps (ms, 0 / 8 workers) | 0.0001 | 0.001 | 0.02 | 0.1 | 1.0 | inf |
|---|---|---|---|---|---|---|
| naive | ERR overflow / ERR overflow | ERR overflow / ERR overflow | 194 / 164 | 309 / 186 | 205 / 165 | 196 / 147 |
| shift | ERR underflow / ERR underflow | ERR underflow / ERR underflow | 667 / 429 | 646 / 422 | 546 / 525 | 410 / 508 |
| clamp | 769 / 485 | 757 / 534 | 567 / 395 | 506 / 462 | 700 / 446 | 730 / 450 |
| softavg | 228 / 132 | 205 / 140 | 202 / 141 | 276 / 187 | 343 / 177 | 201 / 165 |

### Correctness over all GROUP BY cells (12 query x filter x workers cells per entry)

| labels | plan | 0.0001 | 0.001 | 0.02 | 0.1 | 1.0 | inf |
|---|---|---|---|---|---|---|---|
| single | naive | ERR 12/12 (overflow) | 5.1e-16 ERR 10/12 (overflow) | 5.3e-14 | 2.7e-14 | 3.0e-14 | 1.8e-14 |
| single | shift | ERR 12/12 (underflow) | ERR 12/12 (underflow) | 4.5e-14 | 3.1e-14 | 2.3e-14 | 1.8e-14 |
| single | clamp | 0 | 4.5e-16 | 4.5e-14 | 3.1e-14 | 2.3e-14 | 1.8e-14 |
| single | softavg | 0 | 4.3e-16 | 4.5e-14 | 3.9e-14 | 2.2e-14 | 1.8e-14 |
| multi | naive | ERR 12/12 (overflow) | 3.4e-16 ERR 10/12 (overflow) | 4.7e-14 | 3.6e-14 | 3.7e-14 | 3.3e-14 |
| multi | shift | ERR 12/12 (underflow) | ERR 12/12 (underflow) | 5.1e-14 | 2.4e-14 | 2.9e-14 | 3.3e-14 |
| multi | clamp | 0 | 6.8e-16 | 5.1e-14 | 2.4e-14 | 2.9e-14 | 3.3e-14 |
| multi | softavg | 0 | 6.8e-16 | 5.1e-14 | 2.6e-14 | 4.3e-14 | 3.3e-14 |

Plan shapes (q0, no filter, 8 workers, eps 0.1):

- single / naive: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Seq Scan`
- single / shift: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Hash Join > Seq Scan > Hash > Subquery Scan > HashedAggregate > Seq Scan`
- single / clamp: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Hash Join > Seq Scan > Hash > Subquery Scan > HashedAggregate > Seq Scan`
- single / softavg: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Seq Scan`
- multi / naive: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Nested Loop > Seq Scan > Index Scan<imdb_pkey>`
- multi / shift: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Hash Join > Nested Loop > Seq Scan > Index Scan<imdb_pkey> > Hash > Subquery Scan > HashedAggregate > Hash Join > Seq Scan > Hash > Seq Scan`
- multi / clamp: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Hash Join > Nested Loop > Seq Scan > Index Scan<imdb_pkey> > Hash > Subquery Scan > HashedAggregate > Hash Join > Seq Scan > Hash > Seq Scan`
- multi / softavg: `Finalize SortedAggregate > Gather Merge(w=8) > Sort > Partial HashedAggregate > Nested Loop > Seq Scan > Index Scan<imdb_pkey>`

### Maintenance over the join

stream: 10000 single-statement ops {'delete': 4042, 'insert': 4004, 'update': 1021, 'bridge_insert': 454, 'bridge_delete': 479}; first 24 ops delete each multi-label group's argmax item; autocommit, one statement per op, synchronous_commit=off in the stream session; checkpoints every 1000 ops; fan-out = the edited movie's live memberships (bridge ops: 1); membership deltas implied by the stream: 14,863 (1.49 per op)

softavg_ivm_create_join transactional checks: rollback leaves state bytes identical = True; base-vs-bridge race (READ COMMITTED), base first: second blocked = True, view == fresh = True (Drama n 100018 vs 100018); bridge first: blocked = True, view == fresh = True; REPEATABLE READ base writer vs committed bridge write: error 40001 (view == fresh = True); bridge write under REPEATABLE READ: error 0A000

Race without the join-side serialization (`race_demo.py`, view over 20k-row copies, eps 0.1; T1 inserts a dominant base row, T2 its Drama membership): as_shipped/base_first: second blocked True, Drama n view 5782 vs fresh 5782, softavg view 9.7989 vs fresh 9.7989; as_shipped/bridge_first: second blocked True, Drama n view 5783 vs fresh 5783, softavg view 9.8488 vs fresh 9.8488; locks_stripped/base_first: second blocked False, Drama n view 5781 vs fresh 5782, softavg view 5.7908 vs fresh 9.7989; locks_stripped/bridge_first: second blocked False, Drama n view 5781 vs fresh 5783, softavg view 5.7908 vs fresh 9.8488

`softavg_ivm_create` on a join view: error 42809 `ERROR:  "jv" is a view`

| eps | arm | base op ms median del/ins/upd ; p99 | bridge op ms median ins/del | groups recomputed | max drift / v_scale | read ms | notes |
|---|---|---|---|---|---|---|---|
| 0.001 | softavg_ivm_join | 0.18 / 0.23 / 0.33 ; p99 0.43 / 0.47 / 0.62 | 0.28 / 0.30 | 28 | 2.7e-16 | 0.25 |  |
| 0.001 | softavg_ivm_single | 0.11 / 0.13 / 0.17 ; p99 0.28 / 0.29 / 0.39 | 0.05 / 0.07 | 13 | 2.7e-16 | 0.20 |  |
| 0.001 | pg_ivm_linear_join | ERR overflow | - | - | - | - | - | - |
| 0.001 | refresh_mv | 0.04 / 0.06 / 0.06 ; p99 0.17 / 0.18 / 0.17 | 0.05 / 0.06 | 0 | 0 | 0.43 | refresh 388 ms; stale err 0 |
| 0.001 | recompute_on_read | 0.06 / 0.07 / 0.08 ; p99 0.16 / 0.17 / 0.20 | 0.06 / 0.08 | 0 | 0 | 404 |  |
| 0.02 | softavg_ivm_join | 0.16 / 0.22 / 0.29 ; p99 0.38 / 0.42 / 0.62 | 0.24 / 0.26 | 2 | 5.2e-15 | 0.20 |  |
| 0.02 | softavg_ivm_single | 0.10 / 0.11 / 0.15 ; p99 0.27 / 0.30 / 0.39 | 0.05 / 0.06 | 1 | 5.7e-15 | 0.22 |  |
| 0.02 | pg_ivm_linear_join | 0.56 / 0.54 / 0.97 ; p99 0.99 / 0.94 / 1.66 | 0.52 / 0.57 | 0 | 2.8e-14 | 0.23 |  |
| 0.02 | refresh_mv | 0.04 / 0.05 / 0.05 ; p99 0.14 / 0.15 / 0.16 | 0.04 / 0.06 | 0 | 0 | 0.48 | refresh 347 ms; stale err 2.0e-04 |
| 0.02 | recompute_on_read | 0.04 / 0.06 / 0.06 ; p99 0.14 / 0.15 / 0.19 | 0.05 / 0.07 | 0 | 0 | 415 |  |
| 0.1 | softavg_ivm_join | 0.19 / 0.24 / 0.31 ; p99 0.41 / 0.54 / 0.65 | 0.25 / 0.27 | 0 | 2.8e-15 | 0.19 |  |
| 0.1 | softavg_ivm_single | 0.12 / 0.14 / 0.18 ; p99 0.31 / 0.35 / 0.46 | 0.06 / 0.07 | 0 | 1.1e-14 | 0.20 |  |
| 0.1 | pg_ivm_linear_join | 0.57 / 0.55 / 0.98 ; p99 0.95 / 0.97 / 1.87 | 0.52 / 0.58 | 0 | 1.3e-14 | 0.28 |  |
| 0.1 | refresh_mv | 0.07 / 0.07 / 0.09 ; p99 0.12 / 0.13 / 0.16 | 0.06 / 0.08 | 0 | 0 | 0.65 | refresh 400 ms; stale err 4.3e-04 |
| 0.1 | recompute_on_read | 0.07 / 0.07 / 0.09 ; p99 0.21 / 0.24 / 0.26 | 0.06 / 0.09 | 0 | 0 | 382 |  |

Fan-out sweep, `softavg_ivm_join` (fan-out = the movie's multi-label memberships; items per fan-out: {0: 19, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 60, 9: 9, 10: 1, 11: 2, 13: 1}; all (fan-out, item) pairs visited in one random order):

| eps | metric | f=0 | f=1 | f=2 | f=3 | f=4 | f=5 | f=6 | f=7 | f=8 | f=9 | f=10 | f=11 | f=13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | delete ms (median) | 0.07 | 0.13 | 0.14 | 0.14 | 0.15 | 0.15 | 0.17 | 0.17 | 0.18 | 0.17 | 0.18 | 0.19 | 0.18 |
| 0.001 | insert ms (median) | 0.08 | 0.16 | 0.19 | 0.19 | 0.22 | 0.20 | 0.21 | 0.22 | 0.24 | 0.21 | 0.21 | 0.22 | 0.22 |
| 0.001 | groups recomputed / deletes | 0/19 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/60 | 0/9 | 0/1 | 0/2 | 0/1 |
| 0.001 | linear fit ms (f >= 1) | delete 0.177 + 1.5 us x f; insert 0.211 + 4.4 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.001 | after sweep drift / mismatched groups | 2.7e-16 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.02 | delete ms (median) | 0.09 | 0.14 | 0.15 | 0.16 | 0.16 | 0.17 | 0.17 | 0.18 | 0.17 | 0.22 | 0.41 | 0.16 | 0.20 |
| 0.02 | insert ms (median) | 0.10 | 0.19 | 0.21 | 0.22 | 0.21 | 0.23 | 0.24 | 0.24 | 0.24 | 0.27 | 0.31 | 0.23 | 0.24 |
| 0.02 | groups recomputed / deletes | 0/19 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/60 | 0/9 | 0/1 | 0/2 | 0/1 |
| 0.02 | linear fit ms (f >= 1) | delete 0.159 + 6.1 us x f; insert 0.221 + 7.2 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.02 | after sweep drift / mismatched groups | 5.1e-15 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.1 | delete ms (median) | 0.11 | 0.17 | 0.19 | 0.20 | 0.21 | 0.21 | 0.22 | 0.23 | 0.23 | 0.27 | 0.25 | 0.30 | 0.26 |
| 0.1 | insert ms (median) | 0.11 | 0.22 | 0.24 | 0.25 | 0.26 | 0.26 | 0.27 | 0.28 | 0.28 | 0.30 | 0.31 | 0.36 | 0.31 |
| 0.1 | groups recomputed / deletes | 0/19 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/60 | 0/9 | 0/1 | 0/2 | 0/1 |
| 0.1 | linear fit ms (f >= 1) | delete 0.176 + 8.5 us x f; insert 0.227 + 9.0 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.1 | after sweep drift / mismatched groups | 2.8e-15 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |

Fan-out sweep, `softavg_ivm_single` (fan-out = the movie's multi-label memberships; items per fan-out: {0: 19, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 60, 9: 9, 10: 1, 11: 2, 13: 1}; all (fan-out, item) pairs visited in one random order):

| eps | metric | f=0 | f=1 | f=2 | f=3 | f=4 | f=5 | f=6 | f=7 | f=8 | f=9 | f=10 | f=11 | f=13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | delete ms (median) | 0.11 | 0.12 | 0.12 | 0.12 | 0.12 | 0.11 | 0.12 | 0.13 | 0.12 | 0.12 | 0.12 | 0.10 | 0.11 |
| 0.001 | insert ms (median) | 0.12 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.15 | 0.14 | 0.13 | 0.15 | 0.12 | 0.13 |
| 0.001 | groups recomputed / deletes | 0/19 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/60 | 0/9 | 0/1 | 0/2 | 0/1 |
| 0.001 | linear fit ms (f >= 1) | delete 0.137 + -0.0 us x f; insert 0.145 + 1.0 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.001 | after sweep drift / mismatched groups | 2.7e-16 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.02 | delete ms (median) | 0.15 | 0.12 | 0.12 | 0.12 | 0.12 | 0.13 | 0.13 | 0.11 | 0.12 | 0.11 | 0.18 | 0.13 | 0.12 |
| 0.02 | insert ms (median) | 0.17 | 0.14 | 0.15 | 0.14 | 0.14 | 0.15 | 0.15 | 0.14 | 0.14 | 0.13 | 0.18 | 0.16 | 0.16 |
| 0.02 | groups recomputed / deletes | 0/19 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/60 | 0/9 | 0/1 | 0/2 | 0/1 |
| 0.02 | linear fit ms (f >= 1) | delete 0.126 + 1.7 us x f; insert 0.149 + -0.1 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.02 | after sweep drift / mismatched groups | 6.5e-15 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.1 | delete ms (median) | 0.08 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.11 | 0.08 | 0.10 | 0.09 |
| 0.1 | insert ms (median) | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.11 | 0.10 | 0.10 | 0.10 | 0.13 | 0.09 | 0.11 | 0.10 |
| 0.1 | groups recomputed / deletes | 0/19 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/60 | 0/9 | 0/1 | 0/2 | 0/1 |
| 0.1 | linear fit ms (f >= 1) | delete 0.104 + -0.3 us x f; insert 0.164 + -5.7 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.1 | after sweep drift / mismatched groups | 9.5e-15 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |

Fan-out sweep, `pg_ivm_linear_join` (fan-out = the movie's multi-label memberships; items per fan-out: {0: 19, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 60, 9: 9, 10: 1, 11: 2, 13: 1}; all (fan-out, item) pairs visited in one random order):

| eps | metric | f=0 | f=1 | f=2 | f=3 | f=4 | f=5 | f=6 | f=7 | f=8 | f=9 | f=10 | f=11 | f=13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | delete ms (median) | 0.28 | 0.48 | 0.49 | 0.48 | 0.50 | 0.51 | 0.51 | 0.51 | 0.52 | 0.56 | 0.49 | 0.57 | 0.46 |
| 0.02 | insert ms (median) | 0.28 | 0.45 | 0.47 | 0.47 | 0.48 | 0.49 | 0.49 | 0.50 | 0.51 | 0.52 | 0.58 | 0.62 | 0.48 |
| 0.02 | linear fit ms (f >= 1) | delete 0.455 + 8.8 us x f; insert 0.463 + 13.0 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.02 | after sweep drift / mismatched groups | 2.8e-14 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.1 | delete ms (median) | 0.31 | 0.55 | 0.57 | 0.56 | 0.59 | 0.58 | 0.57 | 0.60 | 0.60 | 0.60 | 0.53 | 0.84 | 0.81 |
| 0.1 | insert ms (median) | 0.32 | 0.56 | 0.57 | 0.56 | 0.59 | 0.59 | 0.59 | 0.61 | 0.61 | 0.62 | 0.53 | 0.68 | 0.68 |
| 0.1 | linear fit ms (f >= 1) | delete 0.555 + 7.1 us x f; insert 0.590 + 3.8 us x f |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.1 | after sweep drift / mismatched groups | 1.2e-14 / 0 |  |  |  |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED -->

## What it shows (qualitative; numbers in the generated tables)

- **Membership accounting.** Complete memberships add 54% rows to the
  aggregation (706,206 vs 459,865); 52% of the 297,647 movies with any genre
  have two or more (up to 13). Several groups grow by 5-15x (War, Romance,
  Film-Noir, History, Thriller), i.e. the single-label corpus answered a
  materially different question for them. `(none)` (movies with no genre
  row) is the largest group in both and has fan-out 1.
- **GROUP BY over the join, before/after.** The failure modes are the same
  as single-label: naive overflows and max-shift underflows (PostgreSQL
  raises) at eps <= 1e-3; clamp and softavg are correct everywhere (max rel
  err <= 5.1e-14 over 12 cells per entry). The max-shift spellings now run
  the join **twice** (the per-group MAX needs its own join; plan shape above)
  and are slower than softavg (medians over all cells, serial: shift 2.7x,
  clamp 3.3x; per cell 1.8-3.7x), which stays a one-pass
  aggregate on top of the join. Absolute cost is dominated by the join: the
  multi-label softavg is ~2-3x the single-label one serially, and with 8
  workers PostgreSQL picks a parallel nested loop over `imdb_pkey` that
  scales poorly (multi-label 132-187 ms at 8 workers vs 200-340 ms serial;
  single-label 17-32 ms vs 74-115 ms) -- a host planner effect that is the
  same for naive and softavg. The matrix ran with the machine loaded by
  other agents (load1 ~17).
- **Maintenance over the join.** `softavg_ivm_create` refuses a view (42809),
  and even a trigger that accepted it would never fire for base-table
  writes, so bruce_pg gained `softavg_ivm_create_join`. Per edit it does one
  state delta per membership: the fan-out sweep (interleaved over fan-outs,
  so load drift is not confounded with f) shows a fixed ~0.13-0.23 ms
  trigger cost plus a few microseconds per membership (fits: 1.5-9 us per
  membership); the single-label view on the same edits is flat in f.
  Drift vs a fresh query stays at rounding level (<= 5.2e-15 x v_scale)
  at every checkpoint, also after the sweep. Groups are recomputed only at
  sharp eps when an argmax membership leaves (28 at eps 1e-3, 2 at 0.02, 0
  at 0.1 over 10,000 ops), and one base delete can recompute several groups
  (the pgrx test: an item dominating 9 groups -> exactly 9 recomputes in one
  statement).
- **Before (same engine):** pg_ivm can maintain the linear `SUM(EXP())`
  form over the join only where `EXP` does not overflow (`create_immv`
  fails at eps = 1e-3, as single-label) and costs ~2-3x softavg_ivm_join
  per op where it works; REFRESH costs 350-400 ms per refresh (the join) and
  is stale in between (up to 4.3e-4 x v_scale); recompute-on-read pays the
  join on every read (380-415 ms).
- **MVCC: maintaining a join view through triggers needs serialization
  across the two tables.** With the advisory lock stripped from the
  generated trigger functions, a base insert and its bridge insert in two
  concurrent READ COMMITTED transactions each miss the other: the view
  silently loses the membership (Drama softavg 5.79 vs fresh 9.80). As
  shipped, the second writer blocks and the view equals the fresh query in
  both orders; a REPEATABLE READ base writer that raced a committed bridge
  write gets 40001; bridge writes under REPEATABLE READ are refused (0A000).
  Cost of the lock: bridge writers serialize with base writers (base
  writers among themselves do not).
- Timings are smoke (machine shared; this rerun of `maint_join.py` ran under
  heavier load than the first, whose latencies were ~1.5x lower).

## Files

| file | role |
|---|---|
| `build_bridge.py` -> `bridge.parquet`, `bridge_stats.json` | complete genre memberships from the JOB dump |
| `load_bridge.py` -> `load_bridge.json` | `imdb_genre(id, facet)` into the release server |
| `bench_join.py` -> `cells_join.jsonl` | GROUP BY over the join, single vs multi label |
| `maint_join.py` -> `results_maint_join.json` | maintenance over the join |
| `race_demo.py` -> `race_demo.json`, `ivm_create_on_join_view.json` | the two-session race with the join-side lock stripped vs as shipped; `softavg_ivm_create` given a join view |
| `upgrade_check_after_join.json` | `../postgres/upgrade_check.py rel` output after the join-IVM SQL was added (copied here; `../postgres/upgrade_rel.json` keeps the 2026-11-05 run) |
| `summarize.py` -> `results.json`, this README's tables | |

## Rerun

```bash
PY=python
# release server + bruce_pg release build + ../postgres/load.py rel 3, as in ../postgres/README.md "Rerun"
$PY build_bridge.py          # once; streams movie_info.csv out of imdb.tgz (~2 min)
$PY load_bridge.py
$PY bench_join.py 3
$PY maint_join.py 10000 1000 100
$PY race_demo.py
$PY summarize.py             # server must still be up
```

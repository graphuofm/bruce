# experiments/timeorder — real time order for windows and maintenance (PostgreSQL)

**Timings are smoke numbers on a shared machine** (`results.json` ->
`timing_status = "smoke, machine shared"`). Every number in the generated
block comes from `results.json` via `summarize.py`.

Closes INTEGRATION_ISSUES C2 for PostgreSQL: windows used to slide over
`ORDER BY id` and the maintenance stream was random.

## Which corpus has real timestamps

| corpus | raw source | timestamp in the source? | recovered here? |
|---|---|---|---|
| amazon | McAuley-Lab/Amazon-Reviews-2023, `raw/review_categories/<cat>.jsonl`, streamed by `build_amazon.py` | **yes**: `timestamp` (ms) per review; `build_amazon.py` read it but kept only the year (`filter_num`) | **yes** (2026-11-06, download approved by the author): `build_amazon_ts.py` re-streamed each category file up to the corpus' last line number (`item_id = "<cat>:<line>"`), took `timestamp` at those lines and checked rating == `value` and year == `filter_num`: **1,524,113 of 1,524,113 rows recovered and verified** -> `amazon_ts.parquet` (item_id, ts, ok; 12.8 MB). The HTTP stream was parsed in memory: no raw file and no HF cache entry was written, so there was nothing to delete afterwards (checked: no new file anywhere under `~` or `/tmp` except the parquet and a log in the session scratchpad, since removed) |
| stackexchange | donfu/oa-stackexchange (`build_stackexchange.py`) | **no**: METADATA = `answer_score`, `question_score`, `tags`; no CreationDate, no post id (dataset card in the local HF cache). `item_id = "<site>:<stream index>"` | no; only possible by re-matching question text against the archive.org dumps (very large download) |
| imdb | JOB/IMDb dump (`archive/datasets/job_imdb/imdb.tgz`), the corpus' own source | **yes**: `movie_info` rows of info_type 16 `release dates` ("USA:22 January 2006") | **yes**, locally: `build_imdb_release_ts.py` -> `imdb_ts.parquet` (item_id, corpus row, earliest release date, precision) |

The runs below use both: Amazon review timestamps (a user-event stream, ms
precision; nearly every review has its own timestamp) and IMDb release dates
(an arrival stream of catalogue items, day precision: only rows with day
precision are used, since year- or month-only dates would pile up on the 1st
of January / of the month). Caveat for Amazon: `build_amazon.py` sampled every
5th line of a bounded prefix of each category file, so how the 33 categories
mix inside a time slice depends on each file's line order; within a category
the reviews are real and their timestamps exact.

## What is measured

1. **Windows** (`window_ts.py`): Amazon: the 99k reviews with timestamp in
   [2020-01-01, 2020-07-01), RANGE frames of 1 hour, 1 day, 7 days. IMDb: the
   same rows (day-precision release date in [2005-01-01, 2011-01-01)) under `ORDER BY id ROWS k` (today's setup),
   `ORDER BY ts, id ROWS k`, and time-based `ORDER BY ts RANGE BETWEEN
   INTERVAL 'T days' PRECEDING AND CURRENT ROW` (T = 1, 7, 30; peers
   included); softavg's moving aggregate vs frame recompute (`softavg_fwd`)
   vs the naive `SUM(EXP())` window; punts = cancellation-guard firings
   (PostgreSQL recomputes the frame).
2. **Maintenance** (`maint_ts.py`): `softavg_ivm_create` on `GROUP BY facet`.
   The time stream replays inserts in release-date order and, after each
   insert, deletes the rows older than the retention R (one statement, only
   when there are any). Amazon: 50,000 reviews from 2020-01-01, R = 30 and 2
   days (steady state ~16k and ~1k live rows, 33 categories). IMDb: 50,000
   movies from 2006-01-01, R = 365 and 30 days (~16k and ~1k rows, 29 facets). The random stream inserts the same
   rows in random order and, at the same steps, deletes the same number of
   uniformly chosen live rows. Same initial table, same statement and row
   counts; reported: groups recomputed (the refolds), how many deleted rows
   were their facet's current argmax, latency, drift vs a fresh query.

## Results

<!-- BEGIN GENERATED (summarize.py) -->
_All timings: smoke, machine shared._

- Amazon runs: PostgreSQL 17.10 on x86_64-pc-linux-gnu, build **release**, bruce_pg.so md5 `0ada3e171035b654b299ff9e816dcb82`, extension build **release**, source sha256 `fbd2de171ef2e005...` (softagg-core with the B1 bound change)
- IMDb runs: bruce_pg.so md5 `a4a650d92dff8ced981633b2a30cafa6`, build **release** / **release**, source sha256 `afc0bbe5a5728a5f...` (before B1)
- Amazon review timestamps: 1,524,113 rows recovered, 1,524,113 pass the rating + year check; range 1997-09-10 .. 2023-09-06; 1,522,583 distinct timestamps (ms); score s0 = query row 1072155 ("It’s a great deal for a great price")
- IMDb release dates: 402,589 of 459,865 movies (87.5%) have a parsable release date; precision {'day': 355725, 'year': 25040, 'month': 21824}; range 1888-10-14 .. 2014-12-01, median 2001-10-21
- machine at start of the window runs: IMDb quiet=False, Amazon quiet=False

### Amazon: sliding windows, ORDER BY id vs ORDER BY ts (same 98,626 reviews with timestamp in [2020-01-01, 2020-07-01), score s0)

softavg_moving: ms, abs err / max|v|, punts / inverses. Baselines: ms (max abs err / max|v|) or the error.

| frame | plan | 0.0001 | 0.001 | 0.02 | 0.1 | 1.0 | inf |
|---|---|---|---|---|---|---|---|
| id rows 10 (width mean 11, max 11) | softavg_moving | 42.0, 3.1e-12, 8965/98615 | 34.3, 8.2e-12, 8084/98615 | 30.2, 2.0e-11, 1524/98615 | 40.7, 1.3e-13, 6/98615 | 41.0, 2.9e-14, 0/98615 | 27.7, 0, 0/98615 |
| id rows 10 (width mean 11, max 11) | softavg_fwd | 96.2, 3.6e-16 | 70.8, 5.3e-16 | 84.8, 8.9e-16 | 85.1, 8.9e-16 | 72.1, 7.1e-16 | 80.2, 0 |
| id rows 10 (width mean 11, max 11) | naive_window | ERR overflow | ERR overflow | 99.5, 1.2e-15 | 91.7, 8.9e-16 | 128, 7.1e-16 | 94.1, 0 |
| id rows 100 (width mean 101, max 101) | softavg_moving | 41.4, 3.9e-13, 982/98525 | 34.4, 4.9e-13, 883/98525 | 29.8, 4.4e-12, 108/98525 | 30.1, 5.0e-14, 2/98525 | 30.4, 1.4e-14, 0/98525 | 29.6, 0, 0/98525 |
| id rows 100 (width mean 101, max 101) | softavg_fwd | 572, 1.8e-16 | 468, 7.1e-16 | 464, 3.6e-15 | 461, 1.6e-15 | 452, 1.6e-15 | 418, 0 |
| id rows 100 (width mean 101, max 101) | naive_window | ERR overflow | ERR overflow | 673, 2.5e-15 | 695, 2.0e-15 | 649, 2.0e-15 | 707, 0 |
| id rows 1000 (width mean 996, max 1001) | softavg_moving | 42.5, 3.6e-16, 109/97625 | 51.6, 2.0e-14, 99/97625 | 28.3, 8.9e-13, 11/97625 | 31.7, 1.1e-13, 1/97625 | 39.7, 1.9e-14, 0/97625 | 28.9, 0, 0/97625 |
| id rows 1000 (width mean 996, max 1001) | softavg_fwd | 5208, 3.6e-16 | 4740, 5.3e-16 | 3762, 5.5e-15 | 3781, 4.3e-15 | 4131, 4.1e-15 | 3869, 0 |
| id rows 1000 (width mean 996, max 1001) | naive_window | ERR overflow | ERR overflow | 5882, 6.0e-15 | 5895, 5.2e-15 | 6426, 4.6e-15 | 6566, 0 |
| ts rows 10 (width mean 11, max 11) | softavg_moving | 62.5, 1.3e-12, 8950/98615 | 78.7, 6.1e-12, 8094/98615 | 47.4, 1.2e-11, 1645/98615 | 62.3, 1.5e-13, 4/98615 | 45.7, 1.6e-14, 0/98615 | 31.2, 0, 0/98615 |
| ts rows 10 (width mean 11, max 11) | softavg_fwd | 222, 3.6e-16 | 117, 5.3e-16 | 151, 1.1e-15 | 139, 7.1e-16 | 147, 7.1e-16 | 70.8, 0 |
| ts rows 10 (width mean 11, max 11) | naive_window | ERR overflow | ERR overflow | 121, 1.1e-15 | 132, 1.1e-15 | 134, 8.9e-16 | 100, 0 |
| ts rows 100 (width mean 101, max 101) | softavg_moving | 42.2, 1.1e-13, 959/98525 | 47.2, 4.8e-13, 831/98525 | 44.7, 4.8e-12, 97/98525 | 41.2, 4.6e-14, 0/98525 | 41.1, 2.0e-14, 0/98525 | 57.7, 0, 0/98525 |
| ts rows 100 (width mean 101, max 101) | softavg_fwd | 470, 1.8e-16 | 459, 5.3e-16 | 448, 2.5e-15 | 445, 2.0e-15 | 451, 1.8e-15 | 666, 0 |
| ts rows 100 (width mean 101, max 101) | naive_window | ERR overflow | ERR overflow | 668, 3.4e-15 | 748, 2.3e-15 | 754, 1.8e-15 | 707, 0 |
| ts rows 1000 (width mean 996, max 1001) | softavg_moving | 55.2, 5.0e-15, 86/97625 | 67.3, 2.2e-14, 78/97625 | 48.2, 8.5e-13, 9/97625 | 45.8, 1.4e-14, 0/97625 | 48.5, 1.2e-14, 0/97625 | 49.1, 0, 0/97625 |
| ts rows 1000 (width mean 996, max 1001) | softavg_fwd | 4148, 3.6e-16 | 4424, 7.1e-16 | 4305, 6.8e-15 | 4314, 4.8e-15 | 5249, 4.1e-15 | 4173, 0 |
| ts rows 1000 (width mean 996, max 1001) | naive_window | ERR overflow | ERR overflow | 6859, 5.3e-15 | 6987, 5.0e-15 | 6679, 4.1e-15 | 6767, 0 |
| ts range 1 hour (width mean 30, max 85) | softavg_moving | 216, 8.6e-13, 4045/94132 | 75.5, 4.3e-12, 3666/94552 | 72.0, 1.3e-11, 767/97608 | 90.3, 4.7e-13, 27/98341 | 77.9, 2.2e-13, 5/98352 | 68.7, 0, 2/98353 |
| ts range 1 hour (width mean 30, max 85) | softavg_fwd | 311, 3.6e-16 | 202, 5.3e-16 | 154, 1.6e-15 | 217, 1.6e-15 | 128, 1.1e-15 | 154, 0 |
| ts range 1 hour (width mean 30, max 85) | naive_window | ERR overflow | ERR overflow | 197, 1.6e-15 | 249, 1.2e-15 | 169, 1.2e-15 | 257, 0 |
| ts range 1 day (width mean 577, max 990) | softavg_moving | 84.3, 5.5e-15, 180/97902 | 76.3, 6.6e-14, 154/97929 | 69.9, 1.7e-12, 18/98075 | 69.0, 3.3e-14, 0/98089 | 109, 2.3e-14, 0/98089 | 72.3, 0, 0/98089 |
| ts range 1 day (width mean 577, max 990) | softavg_fwd | 1677, 3.6e-16 | 1327, 5.3e-16 | 1246, 5.2e-15 | 1514, 4.4e-15 | 2188, 3.4e-15 | 1141, 0 |
| ts range 1 day (width mean 577, max 990) | naive_window | ERR overflow | ERR overflow | 1984, 5.0e-15 | 2036, 3.7e-15 | 1946, 3.6e-15 | 2035, 0 |
| ts range 7 days (width mean 3895, max 6143) | softavg_moving | 77.5, 1.8e-16, 20/95013 | 118, 4.5e-15, 19/95013 | 70.8, 3.2e-13, 2/95033 | 118, 4.0e-14, 0/95033 | 108, 2.1e-14, 0/95033 | 86.7, 0, 0/95033 |
| ts range 7 days (width mean 3895, max 6143) | softavg_fwd | 8486, 0 | 8590, 3.6e-16 | 7862, 1.2e-14 | 9285, 8.7e-15 | 7952, 7.1e-15 | 7234, 0 |
| ts range 7 days (width mean 3895, max 6143) | naive_window | ERR overflow | ERR overflow | 12217, 8.5e-15 | 13064, 8.0e-15 | 12492, 9.6e-15 | 12527, 0 |

### Amazon: softavg_ivm, time-ordered stream with sliding retention R = 30 days vs random stream

initial table: 22,255 rows (timestamp in [2020-01-01 - 30 d, 2020-01-01)); 50,000 inserts (2020-01-01 .. 2020-03-14); time stream: 25,129 retention DELETE statements removing 54,373 rows; the random stream issues the same statements with the same row counts (deleted rows uniform over live rows); autocommit, synchronous_commit=off; v_scale 5; machine at start quiet=False

| eps | stream | rows deleted | argmax rows deleted | groups recomputed (delete stmts) | (insert stmts) | ms median ins / del stmt | ms p99 ins / del stmt | max drift / v_scale | group mismatches |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | time | 54373 | 89 | 68 | 0 | 0.16 / 0.18 | 0.25 / 0.31 | 3.5e-14 | 0 |
| 0.001 | random | 54373 | 84 | 66 | 0 | 0.14 / 0.15 | 0.22 / 0.26 | 2.7e-13 | 0 |
| 0.02 | time | 54373 | 89 | 6 | 0 | 0.14 / 0.15 | 0.22 / 0.25 | 8.0e-13 | 0 |
| 0.02 | random | 54373 | 84 | 2 | 0 | 0.14 / 0.14 | 0.22 / 0.25 | 3.0e-13 | 0 |
| 0.1 | time | 54373 | 89 | 0 | 0 | 0.14 / 0.14 | 0.21 / 0.24 | 1.1e-14 | 0 |
| 0.1 | random | 54373 | 84 | 0 | 0 | 0.14 / 0.15 | 0.22 / 0.25 | 1.1e-14 | 0 |

### Amazon: softavg_ivm, time-ordered stream with sliding retention R = 2 days vs random stream

initial table: 1,896 rows (timestamp in [2020-01-01 - 2 d, 2020-01-01)); 50,000 inserts (2020-01-01 .. 2020-03-14); time stream: 24,726 retention DELETE statements removing 50,887 rows; the random stream issues the same statements with the same row counts (deleted rows uniform over live rows); autocommit, synchronous_commit=off; v_scale 5; machine at start quiet=False

| eps | stream | rows deleted | argmax rows deleted | groups recomputed (delete stmts) | (insert stmts) | ms median ins / del stmt | ms p99 ins / del stmt | max drift / v_scale | group mismatches |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | time | 50887 | 1237 | 1079 | 0 | 0.14 / 0.13 | 0.23 / 0.31 | 2.3e-14 | 0 |
| 0.001 | random | 50887 | 1241 | 1078 | 0 | 0.14 / 0.13 | 0.23 / 0.37 | 3.8e-13 | 0 |
| 0.02 | time | 50887 | 1237 | 196 | 0 | 0.15 / 0.14 | 0.23 / 0.27 | 2.2e-12 | 0 |
| 0.02 | random | 50887 | 1241 | 163 | 0 | 0.16 / 0.14 | 0.25 / 0.27 | 2.6e-12 | 0 |
| 0.1 | time | 50887 | 1237 | 0 | 0 | 0.15 / 0.14 | 0.24 / 0.23 | 2.1e-14 | 0 |
| 0.1 | random | 50887 | 1241 | 0 | 0 | 0.15 / 0.13 | 0.23 / 0.23 | 1.4e-14 | 0 |

### IMDb: sliding windows, ORDER BY id vs ORDER BY ts (same 102,296 rows: day-precision release dates in [2005-01-01, 2011-01-01), score s0)

softavg_moving: ms, abs err / max|v|, punts / inverses. Baselines: ms (max abs err / max|v|) or the error.

| frame | plan | 0.0001 | 0.001 | 0.02 | 0.1 | 1.0 | inf |
|---|---|---|---|---|---|---|---|
| id rows 10 (width mean 11, max 11) | softavg_moving | 96.1, 1.9e-12, 9428/102285 | 58.1, 5.6e-12, 8276/102285 | 52.6, 1.9e-11, 932/102285 | 48.3, 7.6e-14, 1/102285 | 59.0, 3.2e-14, 0/102285 | 46.2, 2.6e-14, 0/102285 |
| id rows 10 (width mean 11, max 11) | softavg_fwd | 192, 2.1e-16 | 131, 4.3e-16 | 153, 8.3e-16 | 160, 6.3e-16 | 132, 6.0e-16 | 97.7, 4.0e-16 |
| id rows 10 (width mean 11, max 11) | naive_window | ERR overflow | 108, 6.8e-16 | 162, 1.1e-15 | 111, 7.8e-16 | 165, 6.7e-16 | 114, 4.0e-16 |
| id rows 100 (width mean 101, max 101) | softavg_moving | 106, 3.8e-14, 1126/102195 | 58.1, 7.2e-13, 958/102195 | 48.2, 4.6e-12, 100/102195 | 50.3, 3.1e-14, 0/102195 | 48.4, 9.1e-15, 0/102195 | 51.3, 2.8e-15, 0/102195 |
| id rows 100 (width mean 101, max 101) | softavg_fwd | 1702, 2.3e-16 | 855, 6.3e-16 | 817, 2.1e-15 | 811, 1.5e-15 | 839, 1.3e-15 | 558, 8.5e-16 |
| id rows 100 (width mean 101, max 101) | naive_window | ERR overflow | 1197, 6.5e-16 | 1177, 2.4e-15 | 1179, 1.4e-15 | 1040, 1.4e-15 | 1165, 8.5e-16 |
| id rows 1000 (width mean 996, max 1001) | softavg_moving | 103, 7.4e-15, 109/101295 | 66.0, 9.6e-15, 95/101295 | 48.2, 1.0e-12, 11/101295 | 48.8, 1.9e-14, 0/101295 | 44.5, 8.3e-15, 0/101295 | 47.4, 3.5e-15, 0/101295 |
| id rows 1000 (width mean 996, max 1001) | softavg_fwd | 7002, 1.9e-16 | 7759, 4.8e-16 | 6268, 4.3e-15 | 7788, 3.3e-15 | 6823, 4.4e-15 | 6951, 2.2e-15 |
| id rows 1000 (width mean 996, max 1001) | naive_window | ERR overflow | 11343, 4.7e-16 | 8734, 4.1e-15 | 11142, 3.5e-15 | 11017, 3.2e-15 | 11364, 2.2e-15 |
| ts rows 10 (width mean 11, max 11) | softavg_moving | 84.8, 3.4e-12, 9259/102285 | 74.7, 8.4e-12, 8174/102285 | 45.7, 1.9e-11, 1028/102285 | 51.3, 9.3e-14, 1/102285 | 67.9, 1.5e-14, 0/102285 | 80.5, 1.8e-14, 0/102285 |
| ts rows 10 (width mean 11, max 11) | softavg_fwd | 153, 2.4e-16 | 128, 4.4e-16 | 178, 8.2e-16 | 106, 7.1e-16 | 141, 7.7e-16 | 119, 3.8e-16 |
| ts rows 10 (width mean 11, max 11) | naive_window | ERR overflow | 159, 5.4e-16 | 145, 1.1e-15 | 135, 7.4e-16 | 137, 6.7e-16 | 186, 3.8e-16 |
| ts rows 100 (width mean 101, max 101) | softavg_moving | 121, 7.1e-14, 1023/102195 | 83.3, 5.4e-13, 879/102195 | 84.4, 3.2e-12, 64/102195 | 107, 2.9e-14, 0/102195 | 70.0, 5.9e-15, 0/102195 | 57.2, 1.7e-15, 0/102195 |
| ts rows 100 (width mean 101, max 101) | softavg_fwd | 1751, 2.2e-16 | 930, 4.2e-16 | 955, 2.7e-15 | 931, 1.2e-15 | 963, 1.1e-15 | 759, 8.3e-16 |
| ts rows 100 (width mean 101, max 101) | naive_window | ERR overflow | 1345, 6.1e-16 | 1300, 1.9e-15 | 1374, 1.3e-15 | 1258, 1.2e-15 | 1248, 8.3e-16 |
| ts rows 1000 (width mean 996, max 1001) | softavg_moving | 173, 1.6e-14, 92/101295 | 88.6, 2.6e-14, 84/101295 | 86.1, 8.0e-13, 6/101295 | 92.9, 8.9e-15, 0/101295 | 74.1, 6.7e-15, 0/101295 | 69.1, 4.2e-15, 0/101295 |
| ts rows 1000 (width mean 996, max 1001) | softavg_fwd | 11645, 1.9e-16 | 8104, 4.1e-16 | 8267, 4.1e-15 | 7463, 3.4e-15 | 8055, 3.3e-15 | 6835, 2.0e-15 |
| ts rows 1000 (width mean 996, max 1001) | naive_window | ERR overflow | 11756, 4.1e-16 | 11606, 5.1e-15 | 11917, 2.9e-15 | 11390, 3.0e-15 | 11280, 2.0e-15 |
| ts range 1 days (width mean 100, max 183) | softavg_moving | 120, 7.1e-14, 1079/71939 | 90.1, 5.1e-13, 996/74692 | 82.9, 4.6e-12, 132/99725 | 83.5, 9.6e-14, 2/102244 | 86.3, 3.2e-14, 1/102251 | 83.8, 5.9e-14, 0/102251 |
| ts range 1 days (width mean 100, max 183) | softavg_fwd | 116, 1.0e-16 | 90.0, 4.3e-16 | 113, 2.5e-15 | 85.2, 1.0e-15 | 90.7, 9.6e-16 | 89.1, 7.1e-16 |
| ts range 1 days (width mean 100, max 183) | naive_window | ERR overflow | 126, 5.1e-16 | 92.7, 1.4e-15 | 132, 1.0e-15 | 100, 1.0e-15 | 121, 7.1e-16 |
| ts range 7 days (width mean 393, max 593) | softavg_moving | 152, 1.6e-14, 253/94602 | 87.0, 1.9e-13, 223/95481 | 91.8, 1.1e-12, 16/101652 | 93.9, 2.2e-14, 0/102119 | 86.9, 3.6e-14, 0/102119 | 88.6, 2.4e-14, 0/102119 |
| ts range 7 days (width mean 393, max 593) | softavg_fwd | 197, 0 | 127, 3.1e-16 | 149, 3.1e-15 | 129, 1.9e-15 | 144, 2.1e-15 | 140, 1.9e-15 |
| ts range 7 days (width mean 393, max 593) | naive_window | ERR overflow | 169, 4.1e-16 | 165, 3.3e-15 | 192, 2.3e-15 | 167, 2.2e-15 | 171, 1.9e-15 |
| ts range 30 days (width mean 1491, max 2114) | softavg_moving | 143, 5.8e-16, 61/99331 | 108, 3.0e-15, 58/99388 | 130, 4.3e-13, 6/100972 | 107, 3.7e-14, 0/101140 | 121, 1.9e-14, 0/101140 | 119, 1.8e-14, 0/101140 |
| ts range 30 days (width mean 1491, max 2114) | softavg_fwd | 633, 0 | 349, 3.1e-16 | 340, 4.5e-15 | 323, 3.4e-15 | 352, 3.5e-15 | 350, 2.3e-15 |
| ts range 30 days (width mean 1491, max 2114) | naive_window | ERR overflow | 371, 4.7e-16 | 583, 4.6e-15 | 470, 3.5e-15 | 476, 3.7e-15 | 498, 2.3e-15 |

### IMDb: softavg_ivm, time-ordered stream with sliding retention R = 30 days vs random stream

initial table: 1,003 rows (timestamp in [2006-01-01 - 30 d, 2006-01-01)); 50,000 inserts (2006-01-01 .. 2008-11-14); time stream: 1,048 retention DELETE statements removing 49,024 rows; the random stream issues the same statements with the same row counts (deleted rows uniform over live rows); autocommit, synchronous_commit=off; v_scale 10; machine at start quiet=False

| eps | stream | rows deleted | argmax rows deleted | groups recomputed (delete stmts) | (insert stmts) | ms median ins / del stmt | ms p99 ins / del stmt | max drift / v_scale | group mismatches |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | time | 49024 | 924 | 759 | 0 | 0.16 / 0.70 | 0.38 / 1.83 | 2.5e-13 | 0 |
| 0.001 | random | 49024 | 926 | 769 | 0 | 0.12 / 0.60 | 0.34 / 2.45 | 6.1e-13 | 0 |
| 0.02 | time | 49024 | 924 | 90 | 0 | 0.15 / 0.52 | 0.35 / 1.38 | 1.4e-12 | 0 |
| 0.02 | random | 49024 | 926 | 85 | 0 | 0.14 / 0.51 | 0.33 / 1.23 | 1.3e-12 | 0 |
| 0.1 | time | 49024 | 924 | 0 | 0 | 0.13 / 0.46 | 0.34 / 0.93 | 1.1e-14 | 0 |
| 0.1 | random | 49024 | 926 | 0 | 0 | 0.15 / 0.59 | 0.35 / 1.61 | 9.8e-15 | 0 |

### IMDb: softavg_ivm, time-ordered stream with sliding retention R = 365 days vs random stream

initial table: 16,497 rows (timestamp in [2006-01-01 - 365 d, 2006-01-01)); 50,000 inserts (2006-01-01 .. 2008-11-14); time stream: 1,048 retention DELETE statements removing 49,719 rows; the random stream issues the same statements with the same row counts (deleted rows uniform over live rows); autocommit, synchronous_commit=off; v_scale 10; machine at start quiet=False

| eps | stream | rows deleted | argmax rows deleted | groups recomputed (delete stmts) | (insert stmts) | ms median ins / del stmt | ms p99 ins / del stmt | max drift / v_scale | group mismatches |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | time | 49719 | 85 | 68 | 0 | 0.15 / 0.58 | 0.36 / 1.68 | 1.1e-13 | 0 |
| 0.001 | random | 49719 | 75 | 63 | 0 | 0.13 / 0.51 | 0.35 / 1.79 | 2.8e-14 | 0 |
| 0.02 | time | 49719 | 85 | 3 | 0 | 0.15 / 0.60 | 0.37 / 1.16 | 1.2e-13 | 0 |
| 0.02 | random | 49719 | 75 | 2 | 0 | 0.15 / 0.57 | 0.36 / 1.50 | 9.6e-14 | 0 |
| 0.1 | time | 49719 | 85 | 0 | 0 | 0.16 / 0.53 | 0.32 / 1.09 | 7.4e-15 | 0 |
| 0.1 | random | 49719 | 75 | 0 | 0 | 0.15 / 0.61 | 0.36 / 1.20 | 1.0e-14 | 0 |
<!-- END GENERATED -->

## What it shows (qualitative; numbers in the generated tables)

- **Sources.** Amazon's raw reviews carry ms timestamps; all 1,524,113
  corpus rows were recovered and verified (rating and year match). IMDb
  release dates cover 87.5% of the movies, 77% with day precision.
  StackExchange's source has no timestamps at all.
- **Windows: time order does not change the guard's behaviour.** On both
  corpora, `ORDER BY ts, id` and `ORDER BY id` on the same rows give about
  the same number of punts at every k and eps, close to n/(k+1) at sharp eps
  (the departing row is the frame's argmax about once per frame length, as
  for an exchangeable sequence). Neither a title's nor a review's query score
  is visibly autocorrelated in time. Every moving-aggregate cell stays within
  the PROTOCOL bound (max 2.0e-11 x max|v| on Amazon, 1.9e-11 on IMDb). The
  naive window overflows at eps <= 1e-3 on Amazon (its s0 reaches 1.0) and at
  1e-4 on IMDb.
- **RANGE frames: the moving aggregate's advantage depends on timestamp
  granularity.** Amazon timestamps are essentially unique (1.52M distinct in
  1.52M rows), so every row has its own RANGE frame and frame recompute
  grows with the frame: 7-day frames (mean width 3,895) cost 7.2-9.3 s with
  `softavg_fwd` and 12-13 s with the naive window vs 71-118 ms for the moving
  aggregate (73-111x), the same order as `ROWS 1000`. On IMDb's day-granular
  dates about 47 movies share each frame and PostgreSQL evidently evaluates a
  non-moving aggregate once per distinct frame (inferred from the timings,
  not checked in the executor source), so the gap shrinks to 2.6-4.4x for
  30-day frames, and vanishes for 1-day frames.
- **Maintenance: replaying in timestamp order with sliding retention gives
  essentially the same refold counts as the random stream.** Amazon (50,000
  inserts, ~25k retention DELETE statements of ~2 rows each): 68 vs 66
  (R = 30 d) and 1,079 vs 1,078 (R = 2 d) at eps 1e-3. At eps 0.02 the time
  stream refolds somewhat more: 6 vs 2 and 196 vs 163 (+20%), with the same
  number of argmax rows deleted (89 vs 84, 1,237 vs 1,241); not investigated
  further. No refolds at 0.1. IMDb (50,000 inserts, 1,048 batched
  deletes): 68 vs 63 and 759 vs 769 at 1e-3, 3 vs 2 and 90 vs 85 at 0.02.
  The refold rate is set by eps and by group size (the short retentions keep
  ~1-2k rows, so argmax rows leave 11-14x as often), not by order. Drift
  <= 2.6e-12 x v_scale everywhere (inside softagg-core's 1e-10 absolute
  bound), no group mismatch at any checkpoint. So the random maintenance
  stream of `../postgres/maint_ivm` did not materially bias recompute counts
  on either corpus.
- The Amazon runs used the build with softagg-core's B1 bound change, the
  IMDb runs the build before it (both recorded in `results.json`); the
  time-vs-random comparison is always within one build.

## Files

| file | role |
|---|---|
| `build_imdb_release_ts.py` -> `imdb_ts.parquet`, `imdb_ts_stats.json` | earliest release date per movie from the JOB dump |
| `build_amazon_ts.py` -> `amazon_ts.parquet` | recovers Amazon review timestamps by re-streaming the raw category files (run 2026-11-06, ~7 min; nothing but the parquet is written) |
| `load_amazon_ts.py` -> `load_amazon_ts.json` | `amz_ts(id, facet, value, s0, ts)` into the release server; s0 = query 0 of the PROTOCOL sampler applied to the amazon corpus |
| `window_ts.py` -> `window_ts.jsonl`, `window_ts_amazon.jsonl` (`CORPUS=amazon`) | windows, id vs ts order, ROWS and RANGE frames |
| `maint_ts.py` -> `results_maint_ts_R<R>.json`, `results_maint_ts_amazon_R<R>.json` (`CORPUS=amazon`) | softavg_ivm, time-ordered retention vs random stream |
| `summarize.py` -> `results.json`, this README's tables | |

## Rerun

```bash
PY=python
# release server + bruce_pg release build + ../postgres/load.py rel 3, as in ../postgres/README.md "Rerun"
$PY build_imdb_release_ts.py     # once (~2 min, streams movie_info.csv out of imdb.tgz)
$PY window_ts.py 2005-01-01 2011-01-01
$PY maint_ts.py 50000 5000 2006-01-01 365; $PY maint_ts.py 50000 5000 2006-01-01 30
$PY build_amazon_ts.py           # once (~7 min HTTP stream of the raw review files)
$PY load_amazon_ts.py
CORPUS=amazon $PY window_ts.py 2020-01-01 2020-07-01
CORPUS=amazon $PY maint_ts.py 50000 5000 2020-01-01 30; CORPUS=amazon $PY maint_ts.py 50000 5000 2020-01-01 2
$PY summarize.py                 # server must still be up
```

# Softmax Weighted Aggregation over Vector Search in Relational Databases: artifact

Code, harnesses and measured results for the paper

> Jiacheng Ding, Xiaofei Zhang (University of Memphis).
> *Softmax Weighted Aggregation over Vector Search in Relational Databases.*

This directory continues [Bruce](../README.md) as a database operator; the PostgreSQL extension
keeps Bruce's crate name, `bruce_pg`. All paths below are relative to this `softavg/` directory.

The paper adds one aggregate, `softavg(value, score, eps)`, a softmax-weighted average
(`sum exp(s/eps) * v / sum exp(s/eps)`, computed over a max-anchored state), to PostgreSQL,
DuckDB and SQLite. The aggregate can be used grouped, as a window function and as a maintained
view. One Rust crate holds the arithmetic and a running error bound, the *certificate*, that
makes removal safe. Three thin host extensions link that crate. Every number in the paper comes
from a `results.json` (or a raw file next to it) under `experiments/`.

## Layout

```
engines/
  core/        softagg-core: max-anchored state, removable InvAcc with running error bound,
               conformance vectors (conformance/vectors.json) shared by every host
  postgres/    bruce_pg (pgrx 0.18.1): softavg aggregate (internal state, parallel combine,
               moving-aggregate form), softavg_ivm_create / softavg_ivm_create_join
               (trigger-based incremental view maintenance)
  sqlite/      softagg-sqlite: loadable extension, aggregate + window function (xInverse),
               softagg_dot similarity for float32 BLOB embeddings
  duckdb/      softavg: DuckDB C-API extension (pure Rust); tools/reference holds the
               DuckDB 1.5.6 C API headers used as reference (MIT, from DuckDB)
  relf_core/   build dependency only: the three host crates above depend on it by path
               (Cargo.toml). It is not part of the softavg code path and is not
               evaluated in the paper.
experiments/
  PROTOCOL.md      shared experiment conventions: corpora, spellings, eps grid, queries
  quiet.py         machine-load probe recorded next to every timing
  softmax_ladder/  the softmax ladder S0-S4 in each engine: grouped, windowed, maintained
                   (PostgreSQL 18.6 engine runs, SQLite engine runs, S2/S3 ablation builds)
  postgres/        PostgreSQL before/after, windows, pg_ivm comparison, upgrade check, tests
  sqlite/          SQLite before/after, sqlite-vec top-k, window guard/bound checks
  duckdb/          DuckDB before/after, windows (segment tree), VSS top-k, float32 scores
  window_inverse/  sliding-window SUM maintained by subtraction (SQLite 3.37 / 3.54, PG
                   unsafe_sum, DuckDB, and other engines as context)
  llm_sql/         SQL that LLMs write for a softmax-weighted group average (incl. the
                   constant-shift window SQL used for Case 2)
  multilabel/      Case 3: maintenance over an item-genre bridge table (fan-out, race demo)
  timeorder/       Case 2: real time order for windows and maintenance
  bound/           tightness and cost of the certificate, double-double replay, ESS
  crossover/       when AVG / top-k / threshold / sampling are acceptable vs eps
  esci_task/       Case 1: findings on Amazon ESCI labels vs threshold / top-k / top-b
paper/figures/     figure PDFs of the paper, the three figure scripts that live here,
                   and SOURCES.md (which script produced which PDF)
pg18env/           how the PostgreSQL 18.6 server was set up (README, start/stop, build env)
```

Each experiment directory has its own `README.md` with the exact commands, the generated
result tables and the caveats. Timings are labelled `"timing_status": "smoke, machine shared"`
in the result files: they were taken on a shared machine and the paper reports them as ratios
within one session. Errors and counts do not depend on load.

The experiment READMEs and `results.json` files were written during the runs and contain
absolute paths of the authors' machine (`/home/jding/...`). They are kept unchanged because the
result files record what was run; adjust the paths in the harness scripts (see "Data" below).

## Requirements

| component | version used |
|---|---|
| Rust | rustc 1.95 (stable) |
| cargo-pgrx / pgrx | 0.18.1 (pinned in `engines/postgres/Cargo.toml`) |
| PostgreSQL | 17.10 (release build) and 18.6 (conda-forge build), with pgvector 0.8.7 and pg_ivm 1.16 |
| DuckDB | 1.5.6 (Python package), VSS extension from the DuckDB extension repository |
| SQLite | 3.37.2 (Python's `sqlite3`), sqlite-vec 0.1.9; SQLite 3.54.0 via `apsw==3.54.0.0` |
| Python | 3.10 with numpy, pandas, pyarrow, matplotlib, duckdb==1.5.6, psycopg2; sentence-transformers for embedding (esci_task) |

`experiments/window_inverse/requirements.txt` pins the Python packages of that experiment.

## Building the extensions

Every host crate links `engines/core` by path; build in release mode (all measurements use
`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`).

```bash
# shared core: unit tests + regression tests of the bound
cd engines/core && cargo test --release

# SQLite: target/release/libsoftagg_sqlite.so
cd engines/sqlite && cargo build --release && cargo test --release
python3 tests/test_sqlite.py        # conformance vectors + window checks inside sqlite3
#   needs sqlite-vec's vec0.so (pip download sqlite-vec==0.1.9, extract vec0.so from the
#   manylinux wheel to experiments/sqlite/deps/sqlite_vec-0.1.9/vec0.so, or set VEC0_SO)

# DuckDB: build/softavg.duckdb_extension (unsigned; LOAD with allow_unsigned_extensions=true)
cd engines/duckdb && ./build.sh && cargo test --release
python3 -m pytest -q tests/          # duckdb==1.5.6

# PostgreSQL: pgrx extension bruce_pg 0.2.0
cd engines/postgres
cargo pgrx install --release --pg-config /path/to/pg_config                      # PG 17 (default feature)
cargo pgrx package --features pg18 --no-default-features --pg-config /path/to/pg_config   # PG 18
cargo pgrx test pg17                                                              # 8 Rust + 48 in-database tests
```

`engines/postgres/sql/bruce_pg--0.1.2.sql` is only needed for the upgrade check
(`CREATE EXTENSION bruce_pg VERSION '0.1.2'; ALTER EXTENSION bruce_pg UPDATE`); copy it into
the server's `share/extension` by hand. See `engines/*/README.md` for the SQL surface,
semantics (NULL, NaN, +-Inf, eps rules) and the design of each host's window path.

The ladder ablation builds (rung S2, online without removal, for SQLite; rung S3, removal
without a certificate, for SQLite and PostgreSQL) are separate crates under
`experiments/softmax_ladder/{ladder_core,sqlite_ablation,pg_ablation,replay}`; they link the
same core.

## PostgreSQL 18.6 setup

The window and maintenance cells of the ladder table ran on PostgreSQL 18.6.
`pg18env/README.md` documents the setup in full; in short:

1. Use a PostgreSQL 18.6 release build (the authors used the conda-forge build) and keep
   its installation read-only.
2. Build pgvector 0.8.7 and pg_ivm 1.16 with PGXS, and `bruce_pg` and
   `experiments/softmax_ladder/pg_ablation` with `cargo pgrx package --features pg18
   --no-default-features`. Stage the resulting `.so` files in `ext/lib` and the control/SQL
   files in `ext/share/extension`.
3. `initdb -D data -U <user> --no-locale -E UTF8`, then append the settings block from
   `pg18env/README.md` (port 28860, `shared_preload_libraries = 'pg_ivm'`, `jit = off`, and
   PostgreSQL 18's `dynamic_library_path` / `extension_control_path` pointing at `ext/`).
4. `pg18env/start.sh` / `pg18env/stop.sh` start and stop that cluster (edit the binary
   directory `B=` at the top).

`pg18env/build/env.sh` is the build environment used with the conda headers, and
`pg18env/build/compat/xlocale.h` is the one-line shim it needs on glibc >= 2.26.

The PostgreSQL 17.10 runs (`experiments/postgres`, `multilabel`, `timeorder`) used a scratch
cluster with the same settings; the commands are in `experiments/postgres/README.md`
("Rerun").

## Data (not redistributed)

No dataset, derived table, embedding matrix or database file is included. IMDb's licence
forbids redistribution, and the other sources have their own terms. Obtain them from:

| corpus | source |
|---|---|
| IMDb (459,865 movies, genres, release dates) | the IMDb snapshot distributed with the Join Order Benchmark (JOB, `imdb.tgz`; files `info_type.csv`, `movie_info.csv`); IMDb non-commercial terms apply |
| Amazon Reviews 2023 (1,524,113 reviews, 33 categories, timestamps; product metadata) | McAuley Lab, <https://amazon-reviews-2023.github.io/> (Hugging Face `McAuley-Lab/Amazon-Reviews-2023`) |
| StackExchange (1.2M answers) | Hugging Face `donfu/oa-stackexchange` |
| ESCI shopping queries | <https://github.com/amazon-science/esci-data> |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalised) |

Each harness reads a corpus directory `<root>/<corpus>/` with `corpus.parquet` (columns
`item_id, facet, value, filter_num, filter_bool, text`) and `emb.npy` (float32, 384-d,
L2-normalised, row-aligned with the parquet). The root is a constant (`CORP` or `CORPORA`)
near the top of each harness (e.g. `experiments/postgres/common.py`,
`experiments/softmax_ladder/data.py`); point it at your copy. The 200 query vectors per corpus
are corpus rows drawn with `numpy.random.RandomState(0)` (first sentence of 4-20 words,
de-duplicated); `experiments/postgres/make_queries.py` regenerates `queries.json`.

Derived files that the original runs kept next to the harness are rebuilt by:

| derived file | rebuilt by |
|---|---|
| `experiments/multilabel/bridge.parquet` (movie x genre) | `experiments/multilabel/build_bridge.py` (reads JOB `imdb.tgz`) |
| `experiments/timeorder/imdb_ts.parquet` (release dates) | `experiments/timeorder/build_imdb_release_ts.py` (reads JOB `imdb.tgz`) |
| `experiments/timeorder/amazon_ts.parquet` (review timestamps) | `experiments/timeorder/build_amazon_ts.py` (streams the raw review files) |
| `experiments/esci_task/data/*.parquet` | `experiments/esci_task/scripts/s1_fetch_esci.py`, `s2_join_meta.py`, `s3_build_table.py` |
| SQLite / DuckDB / PostgreSQL databases | the `load*.py` / `build_db.py` / `data.py` step of each experiment |

The scripts that built `corpus.parquet` / `emb.npy` from the raw sources are not part of this
repository; `experiments/PROTOCOL.md` gives the expected layout under `/data/softavg/corpora`.
Some experiment READMEs cite internal design notes (`notes/INTEGRATION_ISSUES.md`,
`notes/CORE_CHANGES.md`) that are not released; the README of each experiment states the
finding itself.

## Proof of the error bound

`PROOF.md` holds the full derivation of the lemma, theorem and corollary on the error of certified values.

## Rerunning the experiments

Run each experiment from its own directory. The commands, expected runtimes and cleanup are in
the "Rerun" / "Run" / "Reproduce" section of its README:

| paper element | directory | entry point (see README) |
|---|---|---|
| Table "What breaks today": PostgreSQL | `experiments/postgres` | `load.py`, `bench.py`, `window.py`, `maint_ivm/run.py`, `summarize.py` |
| Table "What breaks today": SQLite | `experiments/sqlite` | `run.py`, `run_guard_bound.py`, `report.py` |
| Table "What breaks today": DuckDB | `experiments/duckdb` | `build_db.py`, `run_gap.py`, `run_grid.py`, `run_window.py`, `run_topk.py`, `collect.py` |
| Windowed hazard (SQLite 3.54 KBN, PG `unsafe_sum`) | `experiments/window_inverse` | `common.prepare()`, `run_engine.py <engine>`, `assemble.py` |
| Constant-shift window SQL written by an LLM (Case 2) | `experiments/llm_sql` | `evaluate.py` (`--score` re-scores from `exec_cache.jsonl` without engines) |
| Softmax ladder table, rolling trace (Case 2) | `experiments/softmax_ladder` | `data.py`, `sqlite_window.py`, `sqlite_timing.py`, `pg_window.py`, `pg_maint.py`, `replay.py`, `trace.py`, `assemble*.py`, `summarize.py` |
| Case 2 on real timestamps | `experiments/timeorder` | `window_ts.py`, `maint_ts.py`, `summarize.py` |
| Case 3 multi-label maintenance, fan-out figure | `experiments/multilabel` | `load_bridge.py`, `bench_join.py`, `maint_join.py`, `race_demo.py`, `summarize.py` |
| Certificate tightness and cost, ESS | `experiments/bound` | `run.py 20 10 5`, `argpatch.py`, `run.py bench`, `report.py` |
| eps crossover | `experiments/crossover` | `run_crossover.py`, `make_figure.py`, `make_tables.py` |
| Case 1 findings (ESCI) | `experiments/esci_task` | `scripts/s1` ... `s6_summarise.py`, `scripts/make_figure.py` |

Most PostgreSQL harnesses expect a running server with `bruce_pg` (release build), pgvector
and pg_ivm installed, and the IMDb table loaded by `experiments/postgres/load.py rel 3`.
Every `summarize.py` / `report.py` / `assemble.py` regenerates its tables from the raw files
already in the directory, so the tables can be checked without rerunning the measurements.

## Regenerating the figures

The figure PDFs in `paper/figures/` are copies of figures drawn from the result files (no
numbers are typed by hand). `paper/figures/SOURCES.md` maps each PDF to its script:

```bash
python3 paper/figures/ladder_rolling_trace.py      # reads experiments/softmax_ladder/trace.json
python3 paper/figures/case3_fanout_latency.py      # reads experiments/multilabel/results_maint_join.json
(cd paper/figures && python3 host_contracts_execution.py)   # schematic, no data
(cd experiments/bound && python3 report.py)        # figures/{slack_ratio,slack_vs_ops,tradeoff,ess}.pdf
(cd experiments/crossover && python3 make_figure.py)        # figures/crossover.pdf
(cd experiments/esci_task && python3 scripts/make_figure.py)  # figures/findings_tradeoff.pdf
```

The first two scripts resolve their inputs relative to the repository root and write the PDF
next to the script. The experiment scripts write into their own `figures/` directory; the
paper copies are renamed as listed in `SOURCES.md` (for example `bound/figures/slack_ratio.pdf`
-> `paper/figures/bound_slack_ratio_by_eps.pdf`).

## Licence

The engine crates declare `Apache-2.0` in their `Cargo.toml`; see `LICENSE`. The DuckDB headers
in `engines/duckdb/tools/reference/` are DuckDB's (MIT licence). Datasets are not included and
remain under their providers' terms.

# Experiment conventions (shared by all engines)

## Operator

`softavg(value, score, eps)`: the per-group max-anchored softmax-weighted average. Semantics and
arithmetic come from `../engines/core` (crate `softagg-core`); every host extension links this
crate, none re-implements it.

Conformance: every extension has a test that reads `../engines/core/conformance/vectors.json`,
runs GROUP BY (and window queries where the host supports them) inside the host, and compares
with `expected` (aggregates: rtol 1e-12; windows: absolute error <= 1e-9 * max|v|, because the
sliding-window guarantee is absolute). `null` must map to SQL NULL.

## Data

Corpora live under `/data/softavg/corpora/{imdb,amazon,stackexchange}` (symlink or edit the
`CORP` constants). Each has `corpus.parquet` (item_id, facet, value, filter_num, filter_bool,
text) and `emb.npy` (float32, 384-d all-MiniLM-L6-v2, L2-normalised). IMDb has 459,865 rows,
Amazon 1.52M, StackExchange 1.2M. Treat them as read-only; derived data goes into each
experiment's own directory and is deleted after use.

Queries: `rng = np.random.RandomState(0)`; take each item's first sentence, keep 4-20 words,
deduplicate, sample 200; the query vectors are those items' embeddings. Predicates: IMDb
`filter_num >= 2000`, Amazon `filter_num >= 2015`, StackExchange `filter_num >= 3`.

## Before / after

Same engine, same table, same similarity function (the host's own inner product); only the
spelling of the aggregate changes:

- **naive** (S0): `SUM(exp(s/eps)*v) / SUM(exp(s/eps))`
- **max-shift** (S1): per-group `MAX(s)` first, then `exp((s-m)/eps)` (CTE or window, as idiomatic)
- **clamp**: max-shift plus a clamp against underflow (PostgreSQL's float8 `exp` raises on underflow)
- **ours** (S4): `softavg(v, s, eps)`
- hosts with a vector index (pgvector HNSW, DuckDB VSS, sqlite-vec) also run **top-k + AVG**
  and record group coverage.

eps grid: 1e-4, 1e-3, 0.02, 0.1, 1, inf. Recorded: error / NaN / Inf, maximum relative error
against a numpy float64 reference, latency, thread count.

## Results

Each experiment directory has a `README.md` and a `results.json` that records the host version,
the extension's source hash, the build type (release), the softagg-core version, and machine
load (`experiments/quiet.py`). The machine was shared during these runs, so results carry
`"timing_status": "smoke, machine shared"`; compare spellings within one run and use ratios.

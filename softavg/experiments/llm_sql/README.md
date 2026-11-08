# experiments/llm_sql — what SQL do LLMs write for a softmax-weighted group average, and does it run?

Question (PAPER_PLAN §3, "text-to-SQL / AI agent"): when someone asks an LLM for a
relevance-weighted (softmax) per-group average in SQL, what SQL comes back, and how
often does it run and return the right answer in stock PostgreSQL, DuckDB and SQLite —
in particular, how often does it fail **silently** (NaN / NULL / Inf / finite-but-wrong /
answers another question) rather than with an error?

**Status (2026-11-06):** complete for prompts v1: 315 graded samples (3 local models x 15 prompts x 6
samples + 3 Claude models x 15 prompts x 1 sample), each run against 10 query vectors on the full
IMDb table in all three stock engines. Every number below is in `results.json` (the tables at the
bottom are generated from it by `evaluate.py`). Timing was not measured.

## Findings

1. **Most outputs never run, and those failures are loud.** 243 of the 270 local-model samples
   (qwen2.5:7b, qwen2.5:14b, qwen3:14b) are rejected by the engine before running, mostly because
   they call a cosine-similarity function that does not exist (`cosine_similarity`, `COSINE_SIM`,
   `vec_cosine_similarity`, ...: 182 of the 243). Not one local sample is correct, at default
   sampling or at temperature 0. None of the 45 Claude-arm samples is rejected, and 28 are correct
   (Opus 14/15, Sonnet 10/15, Haiku 4/15).
2. **When the SQL does return a result, it is often silently wrong.** Claude arm: 39 samples
   return a result and 11 of them are silent wrong (9 if two float32-rounding cases are not
   counted). Local arm: 15 return a result and all 15 are silent wrong. So in total 26 of the 54 samples that
   returned anything are wrong without an error. All 15 local ones compute a different quantity:
   13 are graded (f), and the 2 NaN results (c) also use L2 distance. The common errors are cosine
   *distance* `<=>` or negative inner product `<#>` used as the similarity (so the least similar
   reviews get the most weight), L2 distance, sigmoid weights, a dropped or multiplied
   temperature, and dividing by COUNT(*).
3. **The same formula fails differently in each engine** (the A1 matrix in
   `notes/INTEGRATION_ISSUES.md`, now reached by LLM-written SQL):
   - Naive `exp(s/T)` or a *global* max-shift at T <= 1e-3: PostgreSQL raises
     (`value out of range: overflow` / `underflow`); DuckDB returns NaN and SQLite returns NULL for
     the affected categories, with no error. Haiku used one table-wide `MAX(sim)` in all 15 answers:
     3 PG errors and 6 silent NaN/NULL results in DuckDB/SQLite (at T = 1e-4, 28 of 29 categories
     come back NaN/NULL).
   - A per-category max-shift with no clamp is correct in DuckDB and SQLite but **raises in
     PostgreSQL** at T <= 1e-3 because PG's float8 `exp` errors on underflow (Sonnet, 3 samples).
     Opus added `CASE WHEN z < -700 THEN 0` in its PG answers and is correct there.
   - **Rolling window, T = 0.02** (no overflow possible, max exponent 50): the same window SQL is
     correct in PostgreSQL (float8 `sum` has no inverse, so PG recomputes each frame) and in DuckDB
     (segment tree; float32 similarity costs about 2e-6 unless cast to DOUBLE, as Opus did). It is
     **finite and wrong in SQLite**, which slides `SUM` by adding and subtracting. All three Claude
     answers (global max-shift / constant shift by 1) reach a max relative error of 3.7e4: a
     "rolling average rating" of 234,918 on a 1–10 scale. The naive form (qwen3, repaired arm)
     reaches 7.1e3 (−41,184). A numpy emulation of an add/subtract sliding sum gives errors of the
     same size (`add_subtract_emulation_err_vs_ref`, up to 4.5e4). This is the one failure no model
     avoided: every T4 SQL that ran on SQLite is wrong.
4. **What the SQL looks like.** The local models write the naive `exp(s/T)` form almost always
   (214 of 270; tagger). The Claude models almost never do: 3 of 45 naive, all three Opus at
   T = 0.1 with a comment that no stabilisation is needed there, which is correct. They use a
   per-category max-shift (21: Sonnet 12, Opus 9, three of them with a clamp), a global max-shift
   (Haiku 15) or a constant shift by the cosine bound for T4 (Opus 3, Sonnet 3). No model used
   log-sum-exp, and none used a vector-index top-k + AVG. Only one T3 answer ("which categories
   have the best ratings") added a `LIMIT 10` (qwen2.5:7b), and it did not bind.
5. **Control, T = 0.1 (T5):** as expected, the naive form is safe here. All 9 Claude T5 answers are
   correct (Opus's 3 are naive), and in the repaired arm 8 naive local T5 answers become correct. The
   failures in points 2 and 3 come from the sharp temperatures and from the window, not from
   the harness.
6. **Secondary, counterfactual (not in any count above):** if only the hallucinated similarity
   function is replaced by the engine's real one (125 local samples), 68 return a result. 11 of
   these are correct and 57 are silent wrong (31 NaN/NULL/Inf, 20 different question, 6 finite
   wrong). So the binding error is what keeps most local-model SQL from failing silently. Two
   repaired qwen3 SQLite answers have the right formula `SUM(rating * exp/total)` but are silently
   wrong (rel err 1.0): Inf/Inf becomes NULL in SQLite and `SUM` skips NULLs, so the category
   silently loses its most relevant reviews.

**What this supports for the paper:** in stock engines, the SQL an LLM writes for this question
either does not run (the local 7-14B models) or runs and is mostly right (Claude: 28/45; Opus
14/15). Honest reading: strong models already know the max-shift trick. The silent failures fall in
the cells the failure matrix predicts: a global shift at sharp T in DuckDB/SQLite, PG's
underflow error for the textbook max-shift, and above all SQLite's sliding SUM, which no model's
care avoided (even Opus). Treat this as an illustration of the failure matrix with real
LLM-written SQL, not as a frequency estimate (see caveats).

## Caveats

- **Prompt wording and setting.** One fixed English prompt per task/engine, a single turn, "SQL
  only", no schema introspection, no tool use, no error feedback. An agent that runs its SQL and
  reads the error would probably fix most (a) cases (unknown function names), and might then land in
  the silent classes, as the repaired arm suggests. Loud vs silent rates are properties of this
  setting.
- **Small samples.** Claude: 1 sample per prompt (45 total), so per-cell counts are 0/1. Local:
  6 per prompt. 10 query vectors per SQL.
- **Claude arm ran inside an agent harness** (other agents, their own system prompt and tools,
  temperature not controlled). I did not see their instructions or what they had read: they
  worked in this repository, where the notes discuss exp overflow, so their outputs may be better
  than a bare API call would give. Their comments ("log-sum-exp trick", "exponents stay within
  [-10, 10]") show awareness of overflow.
- **Local models** are Q4_K_M ollama builds on a shared GPU (partial CPU offload for the 14B
  qwen2.5); the seed is fixed but outputs may not be bit-reproducible across GPU/CPU splits.
- **Data scale.** IMDb ratings are 1–10 and cosine scores lie in [-1, 1]. At T = 0.1 the naive form
  is fine (shown by T5), and at T = 0.02 it cannot overflow. Overflow needs T below ~1/709.
  The query vectors are rows of the table, so each query's best similarity is 1.0 and a global
  max-shift equals the constant shift by 1. With queries that are not in the table the
  global-shift failures (NULL/NaN for categories > 745*T below the best) would hit at least as
  many categories.
- **Tolerance.** 1e-6 relative against the best of several valid score definitions. Two DuckDB T4
  answers miss it by float32 rounding (2.1e-6, below the 8e-6 spread between the float64 and the
  engine's float32 references); they are counted as (d) and flagged "within rounding spread".
- **Pattern tags** for SQL that never ran come from a static tagger (heuristic); every SQL that ran
  was read and tagged by hand (`annotations.json`), which is also where "different question" for
  finite results comes from.
- **One main-arm timeout** (qwen3:14b, T5, PostgreSQL) is not graded; 5 more in the repaired arm.

## Files

- `prompts_v1.json` — the fixed prompt set (not edited).
- `gen_ollama.py` — local arm generation (ollama HTTP API). `outputs/<model>.jsonl` — raw replies
  verbatim + extracted SQL (local: `raw_text`, `thinking`, `extracted_sql`, sampling options, model
  digest; Claude arm: `raw`, `sql`).
- `evaluate.py` — engines, references, grading, tables. `exec_cache.jsonl` — one record per
  distinct (engine, SQL): every run's class, engine message verbatim, error per score variant and per
  alternative question. `engines_env.json` — engine versions/builds and the reference spread.
  `annotations.json` — manual semantic tags and pattern overrides (keyed by `sql_sha`).
- `results.json` — everything below plus per-sample rows (`samples`, `repaired.samples`).

## Rerun

```bash
cd paper_icde/experiments/llm_sql
python3 gen_ollama.py qwen2.5:7b      # etc.; skips samples already in outputs/
python3 evaluate.py                   # executes only (engine, SQL) pairs not in exec_cache.jsonl, re-scores all
python3 evaluate.py --score           # re-score only (no engine started)
```
`evaluate.py` re-executes itself under `python` (duckdb 1.5.6,
psycopg2), starts and stops its own scratch PostgreSQL cluster (`work/pg`, port 28857) and deletes
`work/` at the end. New files in `outputs/` (any of the field spellings above) are picked up by
the same command. A SQL that finishes as (d) should be read and tagged in `annotations.json`.
Delete `exec_cache.jsonl` (or bump `HARNESS_VERSION`) to re-execute everything.



## Setup

**Prompts** — `prompts_v1.json`, fixed before any output was collected: 5 tasks x 3 engines.
Each prompt gives the engine and version, the table
`reviews(id, category, rating, ts, emb)` with the engine's own types, says the embeddings are
L2-normalised and that the query vector is the parameter `$1` (PG, DuckDB) / `?1` (SQLite),
and asks for "a single SQL query ... SQL only, in one sql code block".

| task | wording | temperature |
|---|---|---|
| T1_softmax_named | "average rating ... weighted by the softmax of its cosine similarity ..., with temperature 0.001" | 1e-3 |
| T2_exp_weight | "weight of a review is exp(cosine_similarity / 0.001)" (same quantity, softmax not named) | 1e-3 |
| T3_plain_language | "which categories have the best ratings among reviews relevant to my query ... sharp softmax (temperature 0.0001)" | 1e-4 |
| T4_rolling_window | per review, rolling weighted average over it and the 49 previous reviews of its category (by ts) | 0.02 |
| T5_moderate_temp | as T1 at temperature 0.1 (control: the naive form is safe here) | 0.1 |

**Models.**
- Local arm (`gen_ollama.py`, ollama 0.17.5 at 127.0.0.1:11434, Q4_K_M weights): qwen2.5:7b,
  qwen2.5:14b, qwen3:14b. Per prompt 5 samples at the model's default sampling (no temperature
  passed; qwen2.5's modelfile sets none, so ollama's default 0.8 applies; qwen3:14b's modelfile
  sets 0.6, top_k 20, top_p 0.95; seed = sample index) + 1 sample at temperature 0
  (sample_idx 5). qwen3 runs with its default thinking mode; the thinking trace is stored in
  `thinking`, the SQL is taken from the answer. The prompt is sent as a single user message
  (qwen2.5's built-in system prompt applies). `num_ctx` 16384. One model at a time; the GPU
  was shared with another user's jobs, so part of the 14B models ran on CPU (no effect on outputs
  other than speed).
- Claude arm: `outputs/claude-{haiku,sonnet,opus}.jsonl`, written by three other agents (one
  sample per prompt, temperature not controlled: they ran inside an agent harness, with its
  system prompt and tools, not through the bare API).
- Raw replies are stored verbatim (`raw_text` / `raw`); the graded SQL is the last ```sql block.

**Data.** IMDb corpus (read-only), all 459,865 rows: `facet -> category` (29 categories),
`value -> rating` (1-10), `emb -> emb` (float32, 384-d), `id` = corpus row, and a synthetic
`ts = 2000-01-01 + id seconds` (monotone in corpus order, unique, so the T4 frame is well defined).
Loaded identically into PostgreSQL 17.10 (release build `~/.pgrx/17.10/pg-release-install`,
pgvector 0.8.7, scratch cluster on port 28857, `vector(384)`, `timestamptz`, UTC), DuckDB 1.5.6
(in-memory, `FLOAT[384]`, `TIMESTAMP`, 8 threads) and SQLite 3.37.2 (Python stdlib, math
functions on) + sqlite-vec 0.1.9 (`vec0.so`; `emb` a float32 blob, `ts` ISO-8601 text).
**No softavg extension is loaded anywhere**: these are stock engines.

**Queries.** The PROTOCOL sampler (RandomState(0), first sentence 4-20 words, deduplicated,
200 drawn); the first 10 (same rows as `../postgres/queries.json`). Each SQL is run once per
query vector. Note that every query vector is itself a row of the table, so each query's
best global similarity is 1.0 (a global max-shift then equals the constant shift by 1).

**Execution** (`evaluate.py`). `$1`, `?1` and bare `?` outside string literals are replaced by a
typed literal of the query vector (`'[...]'::vector(384)`, `CAST([...] AS FLOAT[384])`, `X'..'`
float32 blob); other placeholder spellings are left as written. Every run is inside a
transaction that is rolled back. Multi-statement replies run all statements, the last result is
graded. Timeouts: 120 s per aggregate run, 300 s per window run (a timed-out SQL is not run for the
remaining queries and is not graded). Results larger than 2 x 459,865 rows stop the fetch and count as (f)
(row explosion). Classification of
one run:

- (a) the engine rejects the SQL before running it (PG SQLSTATE class 42/0A; DuckDB
  Parser/Binder/Catalog errors; SQLite errors at prepare, checked with `EXPLAIN`);
- (b) error while running (verbatim message recorded, e.g. `value out of range: underflow`);
- (c) runs, but some group/row value is NaN, NULL or Inf;
- (f) runs and finite, but answers another question: groups missing / duplicated, or the values
  match an alternative quantity (plain AVG, distance used as similarity, temperature multiplied,
  L2 score, linear similarity weights, global softmax without per-group renormalisation; T4:
  frame 49/51, no PARTITION BY, cumulative frame), or tagged by hand in `annotations.json`
  after reading the SQL;
- (d) runs, finite, right shape, but max relative error > 1e-6 against every reference;
- (e) within 1e-6 of a reference.

The reference is numpy float64, max-shifted two-pass softavg per group (T4: per frame of the
review and its 49 predecessors in the same category by ts), computed for several **valid score
definitions**: float64 cosine, float64 dot, and the engine's own float32 similarity functions
(pgvector `1-(emb<=>q)` and `-(emb<#>q)`; DuckDB `array_cosine_similarity`,
`array_inner_product`; sqlite-vec `1-vec_distance_cosine`). This matters: at temperature 1e-4 the
references from float64 cosine and from the engine's float32 cosine differ by up to 2e-4
relative (`engines_env.json -> ref_spread_vs_f64_cos`), so a fixed float64 reference would mark
correct SQL wrong. A (d) sample whose error is below that spread is reported separately as
"within rounding spread". The SQL-level class is the worst over its 10 runs
(a > timeout > b > c > f > d > e). Silent wrong = c + d + f.

Pattern tags come from a static tagger (`tag_pattern`: which kind of shift is applied inside
`exp`, clamp, LIMIT), overridden in `annotations.json` for every SQL that runs and was read by
hand. For SQL that never ran the tag is the tagger's.



## Generated tables

<!-- BEGIN GENERATED (evaluate.py) -->

Samples graded: 315 (not yet executed: 0). Class letters: a = parse/bind error, t = timeout, b = runtime error, c = NaN/NULL/Inf, f = different question, d = finite but wrong (> 1e-06 rel), e = correct. Silent wrong = c + d + f.

### Headline per arm

| arm | n | correct | rejected before running (a) | timeout | runtime error (b) | returned a result | of which silent wrong (c+d+f) | silent beyond float32 rounding | ran and answers another question |
|---|---|---|---|---|---|---|---|---|---|
| claude (agent harness, 1 sample/prompt) | 45 | 28 | 0 | 0 | 6 | 39 | 11 | 9 | 0 |
| local ollama (6 samples/prompt) | 270 | 0 | 243 | 1 | 11 | 15 | 15 | 15 | 20 |

T4 (rolling window, T = 0.02) SQL that ran on SQLite, main and repaired arm: claude-haiku s0 d [max-shift(global)] max rel err 3.74e+04; claude-opus s0 d [constant-shift] max rel err 3.74e+04; claude-sonnet s0 d [constant-shift] max rel err 3.74e+04; qwen2.5:14b (repaired) s4 f [naive] max rel err 8.34; qwen2.5:7b (repaired) s3 f [subtraction-without-max] max rel err 4.35; qwen3:14b (repaired) s0 d [naive] max rel err 7.1e+03; qwen3:14b (repaired) s1 d [naive] max rel err 7.1e+03; qwen3:14b (repaired) s2 d [naive] max rel err 7.1e+03; qwen3:14b (repaired) s4 d [naive] max rel err 7.1e+03; qwen3:14b (repaired) s5 f [naive] max rel err 6.09

### Totals per model (SQL-level class, worst over the query runs)

| model | n | a | t | b | c | f | d | (d within rounding spread) | e | silent (c+d+f) | silent beyond rounding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku | 15 | 0 | 0 | 3 | 6 | 0 | 2 | (1) | 4 | 8 | 7 |
| claude-opus | 15 | 0 | 0 | 0 | 0 | 0 | 1 | (0) | 14 | 1 | 1 |
| claude-sonnet | 15 | 0 | 0 | 3 | 0 | 0 | 2 | (1) | 10 | 2 | 1 |
| qwen2.5:14b | 90 | 87 | 0 | 1 | 0 | 2 | 0 | (0) | 0 | 2 | 2 |
| qwen2.5:7b | 90 | 78 | 0 | 4 | 2 | 6 | 0 | (0) | 0 | 8 | 8 |
| qwen3:14b | 90 | 78 | 1 | 6 | 0 | 5 | 0 | (0) | 0 | 5 | 5 |
| **all** | 315 | 243 | 1 | 17 | 8 | 13 | 5 | (2) | 28 | 26 | 24 |

### By model and temperature

| model, temperature | classes |
|---|---|
| claude-haiku / not controlled (agent harness) | b3 c6 d2 e4 |
| claude-opus / not controlled (agent harness) | d1 e14 |
| claude-sonnet / not controlled (agent harness) | b3 d2 e10 |
| qwen2.5:14b / default sampling | a72 b1 f2 |
| qwen2.5:14b / temperature 0 | a15 |
| qwen2.5:7b / default sampling | a65 b3 c2 f5 |
| qwen2.5:7b / temperature 0 | a13 b1 f1 |
| qwen3:14b / default sampling | a64 t1 b5 f5 |
| qwen3:14b / temperature 0 | a14 b1 |

### By task (all models) and by engine (all models)

| task | eps | classes | silent |
|---|---|---|---|
| T1_softmax_named | 0.001 | a49 b7 c2 e5 | 2 |
| T2_exp_weight | 0.001 | a50 b4 c4 e5 | 4 |
| T3_plain_language | 0.0001 | a48 b6 c2 f2 e5 | 4 |
| T4_rolling_window | 0.02 | a51 f3 d5 e4 | 8 |
| T5_moderate_temp | 0.1 | a45 t1 f8 e9 | 8 |

| engine | classes | silent |
|---|---|---|
| duckdb | a88 c5 d2 e10 | 7 |
| postgresql | a65 t1 b17 f13 e9 | 13 |
| sqlite | a90 c3 d3 e9 | 6 |

### model x engine x task (class counts per cell)

| model | engine | T1 (eps 0.001) | T2 (eps 0.001) | T3 (eps 0.0001) | T4 (eps 0.02) | T5 (eps 0.1) |
|---|---|---|---|---|---|---|
| claude-haiku | duckdb | c1 | c1 | c1 | d1 | e1 |
| claude-haiku | postgresql | b1 | b1 | b1 | e1 | e1 |
| claude-haiku | sqlite | c1 | c1 | c1 | d1 | e1 |
| claude-opus | duckdb | e1 | e1 | e1 | e1 | e1 |
| claude-opus | postgresql | e1 | e1 | e1 | e1 | e1 |
| claude-opus | sqlite | e1 | e1 | e1 | d1 | e1 |
| claude-sonnet | duckdb | e1 | e1 | e1 | d1 | e1 |
| claude-sonnet | postgresql | b1 | b1 | b1 | e1 | e1 |
| claude-sonnet | sqlite | e1 | e1 | e1 | d1 | e1 |
| qwen2.5:14b | duckdb | a6 | a6 | a6 | a6 | a6 |
| qwen2.5:14b | postgresql | a6 | a5 b1 | a6 | a5 f1 | a5 f1 |
| qwen2.5:14b | sqlite | a6 | a6 | a6 | a6 | a6 |
| qwen2.5:7b | duckdb | a6 | a4 c2 | a6 | a6 | a6 |
| qwen2.5:7b | postgresql | a3 b3 | a5 b1 | a4 f2 | a6 | a2 f4 |
| qwen2.5:7b | sqlite | a6 | a6 | a6 | a6 | a6 |
| qwen3:14b | duckdb | a6 | a6 | a6 | a6 | a6 |
| qwen3:14b | postgresql | a4 b2 | a6 | a2 b4 | a4 f2 | a2 t1 f3 |
| qwen3:14b | sqlite | a6 | a6 | a6 | a6 | a6 |

### Query-level view (each SQL x query vector run; a SQL that fails to bind counts once)

| model / task | runs by class |
|---|---|
| claude-haiku / T1_softmax_named | b10 c10 d3 e7 |
| claude-haiku / T2_exp_weight | b10 c10 d3 e7 |
| claude-haiku / T3_plain_language | b10 c20 |
| claude-haiku / T4_rolling_window | d20 e10 |
| claude-haiku / T5_moderate_temp | e30 |
| claude-opus / T1_softmax_named | e30 |
| claude-opus / T2_exp_weight | e30 |
| claude-opus / T3_plain_language | e30 |
| claude-opus / T4_rolling_window | d10 e20 |
| claude-opus / T5_moderate_temp | e30 |
| claude-sonnet / T1_softmax_named | b10 e20 |
| claude-sonnet / T2_exp_weight | b10 e20 |
| claude-sonnet / T3_plain_language | b10 e20 |
| claude-sonnet / T4_rolling_window | d20 e10 |
| claude-sonnet / T5_moderate_temp | e30 |
| qwen2.5:14b / T1_softmax_named | a18 |
| qwen2.5:14b / T2_exp_weight | a17 b10 |
| qwen2.5:14b / T3_plain_language | a18 |
| qwen2.5:14b / T4_rolling_window | a17 f1 |
| qwen2.5:14b / T5_moderate_temp | a17 f10 |
| qwen2.5:7b / T1_softmax_named | a15 b30 |
| qwen2.5:7b / T2_exp_weight | a15 b10 c20 |
| qwen2.5:7b / T3_plain_language | a16 f20 |
| qwen2.5:7b / T4_rolling_window | a18 |
| qwen2.5:7b / T5_moderate_temp | a14 f40 |
| qwen3:14b / T1_softmax_named | a16 b20 |
| qwen3:14b / T2_exp_weight | a18 |
| qwen3:14b / T3_plain_language | a14 b40 |
| qwen3:14b / T4_rolling_window | a16 f20 |
| qwen3:14b / T5_moderate_temp | a14 t1 f30 |

### Why (a): engine's rejection message, bucketed

| model | causes |
|---|---|
| qwen2.5:14b | unknown function (mostly a guessed similarity function): 71; operator / type mismatch: 8; unknown / ambiguous column or table (incl. parameter used as a column): 4; syntax error: 3; other: 1 |
| qwen2.5:7b | unknown function (mostly a guessed similarity function): 47; unknown / ambiguous column or table (incl. parameter used as a column): 13; operator / type mismatch: 9; syntax error: 7; other: 2 |
| qwen3:14b | unknown function (mostly a guessed similarity function): 64; operator / type mismatch: 10; syntax error: 2; aggregate / GROUP BY misuse: 1; other: 1 |

### SQL pattern used (static tagger + manual overrides)

| pattern | claude-haiku | claude-opus | claude-sonnet | qwen2.5:14b | qwen2.5:7b | qwen3:14b | classes (all models) |
|---|---|---|---|---|---|---|---|
| naive | 0 | 3 | 0 | 83 | 42 | 89 | a192 t1 b11 c2 f8 e3 |
| max-shift(global) | 15 | 0 | 0 | 5 | 0 | 0 | a4 b3 c6 f1 d2 e4 |
| max-shift(window) | 0 | 6 | 12 | 0 | 0 | 1 | a1 b3 e15 |
| no-exp | 0 | 0 | 0 | 0 | 19 | 0 | a17 f2 |
| subtraction-without-max | 0 | 0 | 0 | 1 | 17 | 0 | a18 |
| constant-shift | 0 | 3 | 3 | 0 | 0 | 0 | d3 e3 |
| max-shift(window) + clamp | 0 | 3 | 0 | 0 | 0 | 0 | e3 |
| subtraction-without-max + clamp | 0 | 0 | 0 | 0 | 3 | 0 | a3 |
| naive + clamp | 0 | 0 | 0 | 0 | 3 | 0 | a3 |
| naive + has-LIMIT | 0 | 0 | 0 | 1 | 1 | 0 | a2 |
| subtraction-without-max + has-LIMIT | 0 | 0 | 0 | 0 | 2 | 0 | a2 |
| other (sigmoid) | 0 | 0 | 0 | 0 | 2 | 0 | f2 |
| no-exp + has-LIMIT | 0 | 0 | 0 | 0 | 1 | 0 | a1 |

(d) finite-but-wrong: 5 samples; 2 of them have max rel err <= 0.001 (float32-rounding scale); max rel errs: 2.1e-06, 2.1e-06, 3.7e+04, 3.7e+04, 3.7e+04

### Secondary: SQL with a hallucinated similarity function, re-run after replacing only that call

SECONDARY analysis, not part of the main counts: every sample whose SQL calls a non-existent cosine-similarity function (vec_cosine_similarity / cosine_similarity / cosine_sim / cos_sim / cosine) is re-run with only that call replaced by the engine's real one (PG 1-(a<=>b), DuckDB array_cosine_similarity, SQLite 1-vec_distance_cosine). Shows what the rest of the SQL would have done. Not included in any count above.

| model | n repaired | classes after repair |
|---|---|---|
| qwen2.5:14b | 57 | a15 t5 b10 c14 f8 e5 |
| qwen2.5:7b | 39 | a17 b3 c8 f10 e1 |
| qwen3:14b | 29 | a1 b6 c9 f2 d6 e5 |

| task | classes after repair |
|---|---|
| T1_softmax_named | a8 t2 b4 c14 f2 d2 |
| T2_exp_weight | a3 b10 c13 f2 |
| T3_plain_language | b5 c3 f1 |
| T4_rolling_window | a13 t2 c1 f10 d4 e3 |
| T5_moderate_temp | a9 t1 f5 e8 |

| pattern (repaired SQL) | classes |
|---|---|
| naive | a19 t4 b19 c30 f16 d6 e11 |
| subtraction-without-max | a4 f2 |
| no-exp | a4 |
| max-shift(global) | a2 t1 |
| subtraction-without-max + clamp | a2 |
| other (sigmoid) | f2 |
| naive + has-LIMIT | a1 |
| constant-shift | c1 |
| naive + clamp | a1 |

### Representative failures (verbatim SQL and engine message; first sample of each class x engine x task, for (a) one per engine)

**a — qwen2.5:14b T1_softmax_named__duckdb sample 0**

```sql
WITH similarities AS (
    SELECT 
        id, 
        category, 
        rating, 
        ts, 
        emb,
        COSINE_SIMILARITY(emb, $1) AS cos_sim
    FROM reviews
),
softmax_weights AS (
    SELECT 
        id, 
        category, 
        rating, 
        EXP(cos_sim / 0.001) AS exp_sim,
        SUM(EXP(cos_sim / 0.001)) OVER () AS total_exp_sim
    FROM similarities
),
weighted_reviews AS (
    SELECT 
        id, 
        category, 
        rating, 
        (exp_sim / total_exp_sim) * rating AS weighted_rating
    FROM softmax_weights
)
SELECT 
    category, 
    AVG(weighted_rating) AS weighted_avg_rating
FROM weighted_reviews
GROUP BY category;
```
Engine: `{"error_type": "CatalogException", "message": "Catalog Error: Scalar Function with name cosine_similarity does not exist!\nDid you mean \"cos\"?\n\nLINE 8:         COSINE_SIMILARITY(emb, (CAST([-0.05493432655930519,0.032122...\n                ^"}`

**a — qwen2.5:14b T1_softmax_named__postgresql sample 0**

```sql
WITH similarities AS (
    SELECT
        id,
        category,
        rating,
        ts,
        emb,
        EXP(COSINE_SIMILARITY(emb, $1) / 0.001) AS exp_sim
    FROM
        reviews
),
softmax_total AS (
    SELECT
        category,
        SUM(exp_sim) AS softmax_denominator
    FROM
        similarities
    GROUP BY
        category
)
SELECT
    s.category,
    SUM(s.rating * s.exp_sim / sm.softmax_denominator) AS weighted_average_rating
FROM
    similarities s
JOIN
    softmax_total sm
ON
    s.category = sm.category
GROUP BY
    s.category;
```
Engine: `{"sqlstate": "42883", "message": "ERROR:  function cosine_similarity(vector, vector) does not exist\nLINE 8:         EXP(COSINE_SIMILARITY(emb, ('[-0.05493432655930519,0...\n                    ^\nHINT:  No function matches the given name and argument types. You might need to add explicit type casts."}`

**a — qwen2.5:14b T1_softmax_named__sqlite sample 0**

```sql
WITH CategoryEmbeddings AS (
    SELECT category, AVG(emb) AS avg_emb
    FROM reviews
    GROUP BY category
),
Similarities AS (
    SELECT 
        c.category,
        r.id,
        r.rating,
        r.emb,
        COSINE_SIM(?1, r.emb) AS cos_sim
    FROM reviews r
    JOIN CategoryEmbeddings c ON c.category = r.category
),
SoftmaxWeights AS (
    SELECT 
        category,
        id,
        rating,
        emb,
        cos_sim,
        EXP(cos_sim / 0.001) / (
            SELECT SUM(EXP(cos_sim / 0.001)) 
            FROM Similarities s2 
            WHERE s2.category = Similarities.category
        ) AS weight
    FROM Similarities
)
SELECT 
    category,
    SUM(weight * rating) / SUM(weight) AS weighted_avg_rating
FROM SoftmaxWeights
GROUP BY category;
```
Engine: `{"error_type": "OperationalError", "message": "no such function: COSINE_SIM", "stage": "prepare"}`

**t — qwen3:14b T5_moderate_temp__postgresql sample 3**

```sql
WITH similarities AS (
    SELECT 
        category,
        rating,
        (1 - (emb <=> $1)) AS similarity,
        EXP((1 - (emb <=> $1)) / 0.1) AS exp_sim
    FROM reviews
),
sum_exps AS (
    SELECT 
        category,
        SUM(exp_sim) OVER (PARTITION BY category) AS sum_exp
    FROM similarities
)
SELECT 
    category,
    SUM(rating * (exp_sim / sum_exp)) AS weighted_avg_rating
FROM similarities
JOIN sum_exps USING (category)
GROUP BY category;
```
Engine: `{"sqlstate": "57014", "message": "ERROR:  canceling statement due to statement timeout"}`

**b — claude-haiku T1_softmax_named__postgresql sample 0**

```sql
WITH scored AS (
  SELECT category, rating, 1 - (emb <=> $1) AS sim
  FROM reviews
),
mx AS (SELECT max(sim) AS m FROM scored),
w AS (
  SELECT s.category, s.rating, exp((s.sim - mx.m) / 0.001) AS w
  FROM scored s CROSS JOIN mx
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category;
```
Engine: `{"sqlstate": "22003", "message": "ERROR:  value out of range: underflow"}`

**b — claude-haiku T2_exp_weight__postgresql sample 0**

```sql
WITH scored AS (
  SELECT category, rating, 1 - (emb <=> $1) AS sim
  FROM reviews
),
mx AS (SELECT max(sim) AS m FROM scored),
w AS (
  SELECT s.category, s.rating, exp((s.sim - mx.m) / 0.001) AS w
  FROM scored s CROSS JOIN mx
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category;
```
Engine: `{"sqlstate": "22003", "message": "ERROR:  value out of range: underflow"}`

**b — claude-haiku T3_plain_language__postgresql sample 0**

```sql
WITH scored AS (
  SELECT category, rating, 1 - (emb <=> $1) AS sim
  FROM reviews
),
mx AS (SELECT max(sim) AS m FROM scored),
w AS (
  SELECT s.category, s.rating, exp((s.sim - mx.m) / 0.0001) AS w
  FROM scored s CROSS JOIN mx
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category
ORDER BY weighted_avg_rating DESC;
```
Engine: `{"sqlstate": "22003", "message": "ERROR:  value out of range: underflow"}`

**c — claude-haiku T1_softmax_named__duckdb sample 0**

```sql
WITH scored AS (
  SELECT category, rating, array_cosine_similarity(emb, $1::FLOAT[384]) AS sim
  FROM reviews
),
mx AS (SELECT max(sim) AS m FROM scored),
w AS (
  SELECT s.category, s.rating, exp((s.sim - mx.m) / 0.001) AS w
  FROM scored s CROSS JOIN mx
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category;
```
Result (query 1): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 0, "n_nan": 1, "n_inf": 0, "missing_groups": 0}`

**c — claude-haiku T1_softmax_named__sqlite sample 0**

```sql
WITH scored AS (
  SELECT category, rating, 1.0 - vec_distance_cosine(emb, ?1) AS sim
  FROM reviews
),
w AS (
  SELECT category, rating,
         exp((sim - (SELECT max(sim) FROM scored)) / 0.001) AS w
  FROM scored
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category;
```
Result (query 1): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 1, "n_nan": 0, "n_inf": 0, "missing_groups": 0}`

**c — claude-haiku T2_exp_weight__duckdb sample 0**

```sql
WITH scored AS (
  SELECT category, rating, array_cosine_similarity(emb, $1::FLOAT[384]) AS sim
  FROM reviews
),
mx AS (SELECT max(sim) AS m FROM scored),
w AS (
  SELECT s.category, s.rating, exp((s.sim - mx.m) / 0.001) AS w
  FROM scored s CROSS JOIN mx
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category;
```
Result (query 1): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 0, "n_nan": 1, "n_inf": 0, "missing_groups": 0}`

**c — claude-haiku T2_exp_weight__sqlite sample 0**

```sql
WITH scored AS (
  SELECT category, rating, 1.0 - vec_distance_cosine(emb, ?1) AS sim
  FROM reviews
),
w AS (
  SELECT category, rating,
         exp((sim - (SELECT max(sim) FROM scored)) / 0.001) AS w
  FROM scored
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category;
```
Result (query 1): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 1, "n_nan": 0, "n_inf": 0, "missing_groups": 0}`

**c — claude-haiku T3_plain_language__duckdb sample 0**

```sql
WITH scored AS (
  SELECT category, rating, array_cosine_similarity(emb, $1::FLOAT[384]) AS sim
  FROM reviews
),
mx AS (SELECT max(sim) AS m FROM scored),
w AS (
  SELECT s.category, s.rating, exp((s.sim - mx.m) / 0.0001) AS w
  FROM scored s CROSS JOIN mx
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category
ORDER BY weighted_avg_rating DESC;
```
Result (query 0): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 0, "n_nan": 28, "n_inf": 0, "missing_groups": 0}`

**c — claude-haiku T3_plain_language__sqlite sample 0**

```sql
WITH scored AS (
  SELECT category, rating, 1.0 - vec_distance_cosine(emb, ?1) AS sim
  FROM reviews
),
w AS (
  SELECT category, rating,
         exp((sim - (SELECT max(sim) FROM scored)) / 0.0001) AS w
  FROM scored
)
SELECT category, sum(w * rating) / sum(w) AS weighted_avg_rating
FROM w
GROUP BY category
ORDER BY weighted_avg_rating DESC;
```
Result (query 0): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 28, "n_nan": 0, "n_inf": 0, "missing_groups": 0}`

**f — qwen2.5:14b T4_rolling_window__postgresql sample 2**

```sql
WITH RankedReviews AS (
    SELECT id, category, rating, ts, emb,
           ROW_NUMBER() OVER(PARTITION BY category ORDER BY ts) AS row_num
    FROM reviews
),
Similarities AS (
    SELECT id, category, rating, ts, emb,
           row_num,
           (emb <=> $1) AS cosine_similarity,
           EXP((emb <=> $1) / 0.02) AS exp_sim,
           EXP((emb <=> $1) / 0.02) / SUM(EXP((emb <=> $1) / 0.02)) OVER (PARTITION BY category ORDER BY row_num ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS softmax_weight
    FROM RankedReviews
),
WeightedAverages AS (
    SELECT category, row_num,
           SUM(softmax_weight * rating) OVER (PARTITION BY category ORDER BY row_num ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) /
           SUM(softmax_weight) OVER (PARTITION BY category ORDER BY row_num ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS relevance_weighted_avg_rating
    FROM Similarities
)
SELECT r.id, r.category, r.rating, r.ts, r.emb, wa.relevance_weighted_avg_rating
FROM RankedReviews r
JOIN WeightedAverages wa ON r.row_num = wa.row_num
ORDER BY r.ts;
```
Result (query 0): `{"why": "row explosion: more than 920730 rows (fetch stopped)"}`

**f — qwen2.5:14b T5_moderate_temp__postgresql sample 1**

```sql
WITH cosine_sim AS (
    SELECT 
        id, 
        category, 
        rating, 
        ts, 
        emb, 
        (emb <=> $1) AS cosine_similarity
    FROM 
        reviews
),
softmax_weights AS (
    SELECT 
        id, 
        category, 
        rating, 
        cosine_similarity, 
        EXP((cosine_similarity - (SELECT MAX(cosine_similarity) FROM cosine_sim)) / 0.1) AS weight
    FROM 
        cosine_sim
),
normalized_weights AS (
    SELECT 
        id, 
        category, 
        rating, 
        weight, 
        SUM(weight) OVER () AS total_weight
    FROM 
        softmax_weights
),
weighted_ratings AS (
    SELECT 
        category, 
        (rating * weight / total_weight) AS weighted_rating
    FROM 
        normalized_weights
)
SELECT 
    category, 
    AVG(weighted_rating) AS weighted_avg_rating
FROM 
    weighted_ratings
GROUP BY 
    category;
```
Result (query 0): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 0, "n_nan": 0, "n_inf": 0, "missing_groups": 0, "err_best": 0.9999986835099368, "best_variant": "f64_dot"}`

**f — qwen2.5:7b T3_plain_language__postgresql sample 1**

```sql
WITH similarity AS (
    SELECT
        id,
        category,
        rating,
        ts,
        emb,
        (1 - CAST((emb <=> $1) AS REAL)) * 100 AS similarity_score
    FROM
        reviews
),
weighted_ratings AS (
    SELECT
        category,
        SUM(rating * similarity_score / (SELECT SUM(similarity_score) FROM similarity)) AS weighted_avg_rating
    FROM
        similarity
    GROUP BY
        category
)
SELECT
    category,
    weighted_avg_rating
FROM
    weighted_ratings
ORDER BY
    weighted_avg_rating DESC;
```
Result (query 0): `{"n_rows": 29, "value_col": "weighted_avg_rating", "n_null": 0, "n_nan": 0, "n_inf": 0, "missing_groups": 0, "err_best": 0.9999179260642318, "best_variant": "f64_cos"}`

**d — claude-haiku T4_rolling_window__duckdb sample 0**

```sql
WITH scored AS (
  SELECT id, category, ts, rating, array_cosine_similarity(emb, $1::FLOAT[384]) AS sim
  FROM reviews
),
w AS (
  SELECT id, category, ts, rating,
         exp((sim - max(sim) OVER ()) / 0.02) AS w
  FROM scored
)
SELECT id, category, ts,
       sum(w * rating) OVER win / sum(w) OVER win AS rolling_weighted_avg_rating
FROM w
WINDOW win AS (PARTITION BY category ORDER BY ts, id ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
ORDER BY category, ts, id;
```
Result (query 0): `{"n_rows": 459865, "value_col": "rolling_weighted_avg_rating", "n_null": 0, "n_nan": 0, "n_inf": 0, "missing_rows": 0, "err_best": 1.3610937654298933e-06, "best_variant": "duckdb_array_cosine_similarity"}`

**d — claude-haiku T4_rolling_window__sqlite sample 0**

```sql
WITH scored AS (
  SELECT id, category, ts, rating, 1.0 - vec_distance_cosine(emb, ?1) AS sim
  FROM reviews
),
w AS (
  SELECT id, category, ts, rating,
         exp((sim - max(sim) OVER ()) / 0.02) AS w
  FROM scored
)
SELECT id, category, ts,
       sum(w * rating) OVER win / sum(w) OVER win AS rolling_weighted_avg_rating
FROM w
WINDOW win AS (PARTITION BY category ORDER BY ts, id ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
ORDER BY category, ts, id;
```
Result (query 0): `{"n_rows": 459865, "value_col": "rolling_weighted_avg_rating", "n_null": 0, "n_nan": 0, "n_inf": 0, "missing_rows": 0, "err_best": 3.478976584562054, "best_variant": "f64_dot"}`

<!-- END GENERATED -->

# ESCI task: does softavg mine more, and more correct, findings than AVG / top-k / threshold?

Built 2026-11-06. Ground truth = Amazon's human E/S/C/I relevance labels (ESCI, KDD Cup 2022);
no humans in the loop here. Every number below is in `results.json` (written by
`scripts/s6_summarise.py`); the figure is drawn from `results.json` by `scripts/make_figure.py`.

## Bottom line

1. **No distractors in the pool (the setting the brief asked for): softavg beats the baselines
   people use today, but it does not beat a simple relative cutoff.** With eps picked on dev
   queries (0.05 to 0.2 every time), softavg's findings-F1 on test is above the absolute
   threshold in 7 of 8 settings. The paired-bootstrap 95% CI excludes 0 in 3 of them
   (+0.039 to +0.042), touches 0 in 4 (+0.006 to +0.023), and goes the other way in 1
   (-0.014, n.s.). Against global top-k + AVG it is +0.024 to +0.106, with the CI excluding 0 in
   7 of 8; against per-facet top-b, +0.037 to +0.139, 8 of 8. Unlike threshold and top-k, it
   keeps 100% facet coverage and keeps the facet ranking: pairwise ordering accuracy on
   informative pairs is 0.86 to 0.93, against 0.40 to 0.71 for the threshold, which drops facets.
   **But the gain is not specific to the exponential kernel.** A per-query relative threshold
   (`s >= max_q(s) - d`, "relQ") ties softavg in all 8 settings (CIs include 0). A per-facet hard
   box (`AVG(v) FILTER (WHERE s >= MAX(s) OVER (PARTITION BY facet) - d)`, "relF") is higher in 7
   of 8: significantly in 2 (-0.048 and -0.017 for softavg), n.s. in the other 5. What helps is
   anchoring on the facet's or query's own best match. Absolute cosine levels vary too much
   across queries for a global tau or k.
2. **With realistic distractors (200 in-category products per query) and brand facets, softavg
   alone is much worse than a threshold** (F1 0.22 to 0.26 vs 0.43 to 0.47). softavg never
   drops a facet, so it answers for about 160 distractor-only brands per query. Facet precision
   is 0.03 to 0.04 and top-1 hit is 0.03 to 0.04. Adding `HAVING MAX(s) >= tau` ("sa+gate") or
   `WHERE s >= tau` before softavg ("thr+sa") recovers it, but only to a tie with the threshold
   alone (0.40 to 0.47). With category facets and distractors, softavg ties the threshold for
   relevant = E (+0.002, +0.000, CIs straddle 0) and loses for E+S (-0.043, -0.015). relF is best,
   or within 0.004 of the best, in all 4 category settings.
3. **eps band.** softavg is only competitive for eps in about [0.03, 0.2]. The within-query
   standard deviation of the cosine over labelled products has median 0.095 (IQR 0.07 to 0.13),
   so the useful eps is about one score standard deviation. For eps <= 0.01 it is an argmax per
   facet and is worse than the threshold in all 8 labelled settings. As eps grows past 0.3 it
   moves toward AVG and findings-F1 falls: 0.08 to 0.23 at eps = 0.3, <= 0.03 at eps = 1.
4. **The absolute level is modest for every estimator** (F1 0.16 to 0.37 without distractors).
   The relevance signal itself is the bottleneck: the within-query AUC of title-query cosine
   for E vs non-E is 0.74 mean, 0.79 median.

So the honest claim this experiment supports: "relevance-anchored aggregation (softavg, or a
relative cutoff) detects relevant-vs-overall differences with a findings-F1 about 12 to 15%
higher than a fixed cosine threshold when relevant = E (significant in 3 of 4 settings), and
about 1.1 to 1.55 times that of global top-k (typically 1.4), while keeping every facet". The claim it does **not** support: "the softmax weighting itself is what
finds them", or "softavg is better than a threshold when the pool contains many irrelevant
groups". softavg does no group selection: it needs a gate. The argument left to softavg is
operational, and this experiment does not measure it. It computes the anchored estimate in one
mergeable, invertible pass, while relF needs a window or a second pass over each facet. That
argument belongs to the engine sections, not to "more findings".

## Data and join coverage (step 1)

- **ESCI**: official release `github.com/amazon-science/esci-data`
  (`shopping_queries_dataset_{examples,products}.parquet`), read with DuckDB column projection.
  (The HF mirror `tasksource/esci` rate-limited us with HTTP 429.) US locale: 1,818,825
  judgments, 97,345 queries, 1,215,851 products. Labels: E 1,247,558; S 369,313; I 161,905;
  C 40,049.
- **Amazon Reviews 2023 item metadata**: all 34 `meta_*.jsonl.gz` files, 26.3 GB gzip /
  101.4 GB uncompressed, 35.4M items. Streamed from the McAuley-lab mirror through
  `curl | gunzip`; only matching rows were kept, nothing else was stored. The HF copy is
  rate-limited for anonymous clients.
- **Join rule**: ESCI `product_id` (ASIN) == AR2023 `parent_asin`. Item metadata is keyed by
  parent ASIN only. Child ASINs could only be mapped through the per-review files (hundreds of
  GB), which was not done.

| coverage (`results.json: coverage`) | |
|---|---|
| ESCI US products matched | **447,879 / 1,215,851 = 36.8%** (all have `average_rating`) |
| ... with a price | 240,156 = 19.8% |
| judgments with a rating | 35.1% (E 34.3%, S 35.5%, I 37.7%, C 42.9%) |
| queries with >= 8 / 12 / 16 rated labelled products | 34,724 / 13,950 / 4,698 |

This is thin but enough, so no other numeric attribute was needed. ESCI's own catalogue has no
numeric column; price and rating come only from the join. Bias: the join keeps products whose
ASIN is its own parent, so single-variant listings are likely over-represented, and I/C products match
slightly more often than E.

## Task (step 2)

- Queries: every ESCI US query with >= 8 labelled products that have a rating and >= 1 E among
  them: **33,692 queries**, 401,246 labelled rows. Pool size per query is 8 / 11 / 17
  (p10 / median / p90). They are split 50/50 into dev/test by `md5(query_id)`.
- Pools: **labelled** = the query's labelled products. **labelled+distractors** adds 200
  products per query, drawn at random from rated ESCI products of the same `main_category` as
  the query's dominant one and never labelled for that query (6.73M rows).
- Score: `s = cos(query, product_title)`, all-MiniLM-L6-v2, normalized embeddings, on the GPU
  (batch 64; the GPU was shared). Mean s: E 0.546, S 0.455, C 0.455, I 0.329, distractor 0.150.
- Facets: `brand` (ESCI `product_brand`) and `main_category` (AR2023). Values: `rating` (AR2023
  `average_rating`) and `log_price` (log of AR2023 `price`; rows without a price leave the
  table in the price settings).
- Ground truth per (query, facet): mean value over the facet's products labelled **E**, or **E+S**.
  Undefined when the facet has none.

## Estimators and metrics (step 3, `scripts/s4_evaluate.py`)

Per (query, facet), over the pool: `avg` (plain AVG = softavg at eps = inf); `softavg@eps`
(eps in 1e-3 .. 1); `thresh@tau` (AVG over s >= tau); `topk@k` (global top-k of the query, then
AVG per facet; k in 5..200, including the 10/50/100 asked for); `facet_topb@b` (AVG of the
facet's b best); and three additions I made so the comparison is fair: `thresh_relq@d`
(s >= max_query(s) - d), `thresh_relf@d` (s >= max_facet(s) - d, the hard box version of softavg's
anchoring), `softavg_gate` (softavg with HAVING max(s) >= tau) and `thresh_softavg` (WHERE s >= tau,
then softavg). **All hyperparameters are chosen on the dev queries by findings-F1** and reported
on the test queries. 95% CIs come from 1,000 bootstrap resamples of the test queries, paired
across estimators.

- **Finding**: a (query, facet) where |GT - base| > delta, base being the facet's plain AVG over the
  pool (a real relevant-vs-overall difference). delta = 0.2 stars for rating and 0.2 for
  log price (about 22%). Sensitivity runs use 0.1 / 0.3 stars and 0.1 / 0.4 log.
- **Detected**: the estimator answers and est - base has the same sign as GT - base with
  |est - base| > delta.
- **False finding**: |est - base| > delta where GT is undefined (no relevant product in the
  facet), within delta of base, or of the opposite sign. Plain AVG detects nothing by construction.
- Also reported: MAE vs GT on answered GT facets and on *mixed* facets (both relevant and
  non-relevant products; the only facets where estimators can differ), coverage of GT facets,
  facet precision (answered facets that have >= 1 relevant product), pairwise ordering accuracy
  of GT facets (Kendall tau-a = 2*acc - 1; "informative" pairs have >= 1 mixed facet; a missing
  answer counts as wrong), and top-1 hit.

## Results: findings-F1 on test (dev-chosen parameter in parentheses)

`results.json: headline`, `settings.<key>.paired_diff_softavg_minus`. Rows marked +D use the
labelled+distractors pool.

| setting (facet / value / relevant) | GT findings | softavg | thr | top-k | top-b | relQ | relF | sa - thr (95% CI) | sa - relF (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| brand / rating / E | 730 | 0.225 (0.05) | 0.201 (0.55) | 0.200 (5) | 0.188 (1) | 0.211 (0.075) | 0.273 (0.1) | +0.023 [-0.002, +0.050] | -0.048 [-0.068, -0.029] |
| brand / rating / E+S | 311 | 0.157 (0.15) | 0.151 (0.5) | 0.118 (5) | 0.068 (2) | 0.180 (0.2) | 0.165 (0.2) | +0.006 [-0.030, +0.041] | -0.009 [-0.041, +0.025] |
| brand / log_price / E | 1723 | 0.320 (0.1) | 0.278 (0.55) | 0.232 (5) | 0.247 (1) | 0.314 (0.15) | 0.333 (0.15) | +0.042 [+0.022, +0.062] | -0.013 [-0.028, +0.002] |
| brand / log_price / E+S | 847 | 0.215 (0.15) | 0.194 (0.475) | 0.148 (5) | 0.133 (2) | 0.203 (0.2) | 0.206 (0.15) | +0.022 [-0.000, +0.043] | +0.009 [-0.005, +0.024] |
| main_category / rating / E | 1937 | 0.335 (0.07) | 0.296 (0.475) | 0.242 (5) | 0.227 (1) | 0.335 (0.15) | 0.352 (0.15) | +0.039 [+0.021, +0.058] | -0.017 [-0.031, -0.004] |
| main_category / rating / E+S | 745 | 0.234 (0.15) | 0.247 (0.35) | 0.151 (5) | 0.095 (2) | 0.249 (0.3) | 0.249 (0.2) | -0.014 [-0.038, +0.014] | -0.016 [-0.035, +0.002] |
| main_category / log_price / E | 3840 | 0.370 (0.1) | 0.330 (0.5) | 0.264 (5) | 0.304 (2) | 0.366 (0.15) | 0.376 (0.15) | +0.040 [+0.028, +0.053] | -0.006 [-0.016, +0.004] |
| main_category / log_price / E+S | 1868 | 0.254 (0.2) | 0.241 (0.425) | 0.189 (5) | 0.174 (2) | 0.254 (0.2) | 0.264 (0.2) | +0.013 [-0.004, +0.029] | -0.010 [-0.024, +0.003] |
| +D brand / rating / E | 2170 | 0.223 (0.2) | 0.427 (0.5) | 0.385 (10) | 0.092 (2) | 0.390 (0.15) | 0.364 (0.3) | -0.204 [-0.224, -0.185] | -0.141 [-0.159, -0.125] |
| +D brand / rating / E+S | 2296 | 0.259 (0.2) | 0.460 (0.5) | 0.460 (10) | 0.094 (2) | 0.394 (0.15) | 0.420 (0.3) | -0.202 [-0.220, -0.183] | -0.162 [-0.180, -0.144] |
| +D brand / log_price / E | 4114 | 0.230 (0.3) | 0.453 (0.525) | 0.393 (5) | 0.151 (3) | 0.396 (0.1) | 0.383 (0.3) | -0.223 [-0.237, -0.209] | -0.153 [-0.166, -0.142] |
| +D brand / log_price / E+S | 4236 | 0.259 (0.3) | 0.470 (0.475) | 0.406 (5) | 0.150 (3) | 0.383 (0.15) | 0.430 (0.3) | -0.211 [-0.224, -0.198] | -0.171 [-0.184, -0.159] |
| +D main_category / rating / E | 7173 | 0.647 (0.05) | 0.645 (0.5) | 0.641 (5) | 0.661 (5) | 0.669 (0.15) | 0.684 (0.15) | +0.002 [-0.007, +0.011] | -0.037 [-0.045, -0.031] |
| +D main_category / rating / E+S | 6380 | 0.629 (0.05) | 0.672 (0.475) | 0.681 (10) | 0.695 (5) | 0.679 (0.2) | 0.691 (0.2) | -0.043 [-0.052, -0.034] | -0.061 [-0.070, -0.053] |
| +D main_category / log_price / E | 12208 | 0.752 (0.07) | 0.751 (0.45) | 0.725 (5) | 0.734 (3) | 0.756 (0.15) | 0.764 (0.15) | +0.000 [-0.005, +0.005] | -0.012 [-0.016, -0.008] |
| +D main_category / log_price / E+S | 12607 | 0.765 (0.07) | 0.780 (0.425) | 0.756 (5) | 0.766 (3) | 0.779 (0.2) | 0.787 (0.2) | -0.015 [-0.020, -0.009] | -0.022 [-0.027, -0.017] |

sa+gate and thr+sa: within +/-0.01 of softavg in the labelled pools; 0.40 to 0.47 (about equal
to thr) with +D brand. Counts for one representative setting, labelled / main_category /
log_price / E (test, 16,146 queries, 9,067 mixed facets):

| | param | detected | false | MAE mixed | coverage | facet prec. | pair acc. (informative) | top-1 |
|---|---|---|---|---|---|---|---|---|
| AVG | - | 0 | 0 | 0.266 | 1.00 | 0.80 | 0.852 | 0.867 |
| softavg | eps 0.1 | 1594 | 3181 | 0.236 | 1.00 | 0.80 | 0.856 | 0.827 |
| AVG, s >= tau | 0.5 | 1255 | 2516 | 0.240 | 0.76 | 0.90 | 0.515 | 0.650 |
| global top-k | 5 | 1030 | 2936 | 0.270 | 0.92 | 0.84 | 0.590 | 0.770 |
| per-facet top-b | 2 | 1657 | 5406 | 0.311 | 1.00 | 0.80 | 0.818 | 0.798 |
| relQ | 0.15 | 1573 | 3180 | 0.243 | 0.88 | 0.87 | 0.553 | 0.728 |
| relF | 0.15 | 1552 | 2862 | 0.241 | 1.00 | 0.80 | 0.856 | 0.840 |

Findings precision is low for every estimator (0.23 to 0.35 here). Roughly two of every
three claimed differences are wrong, because the cosine is a weak relevance signal (AUC 0.74).
Top-1 is best for plain AVG in the labelled pools, because most facets are singletons or all-E;
the facet ranking is not where the estimators differ.

**Figure** `figures/findings_tradeoff.{pdf,png}`: true vs false findings on test across each
family's whole parameter sweep, with the dev-chosen point ringed. In the two labelled panels shown,
the softavg and relF curves lie on or above the threshold and top-k curves at every
false-finding budget, with relF slightly above softavg. With +D brand, the threshold and top-k reach about 1,000
detections at about 1,500 to 2,300 false ones; softavg needs about 10,000 false ones for the same.

**eps band** (`results.json: eps_band`): within the labelled pools, softavg beats the dev-chosen
threshold for eps in [0.02, 0.1], [0.1, 0.15], [0.03, 0.2], [0.1, 0.2], [0.05, 0.15], (none),
[0.03, 0.15], [0.1, 0.2] (same row order as the table). It beats top-k over a wider band, and
relF at no eps in 7 of 8.

**delta sensitivity** (`results.json: delta_sensitivity`, labelled pools, 200 bootstrap
resamples): the same picture at delta = 0.1 and 0.3 stars / 0.1 and 0.4 log. softavg - thr is
+0.036 to +0.057 (CI excludes 0) for relevant = E in all 4 settings at both deltas, and
-0.011 to +0.028 for E+S. softavg - relF is -0.028 to +0.016, never significantly positive.

**Examples** (labelled pool, brand, log price; GT = geometric mean over E):
- 'rigid wet saw', brand Ridgid: 12 products, all with s in [0.24, 0.46]. Global threshold
  tau = 0.55 returns NULL for the facet. GT $222, plain AVG $98, softavg(0.1) $129: right
  direction, about a quarter of the way to GT.
- 'jojo siwa doll', brand Just Play: E product $26 (s 0.77), S product $6 (s 0.64). AVG $13,
  softavg $19; tau = 0.55 keeps both products and returns $13.

## Engine check

`data/engine_check.json`: the softavg values used above equal what the shipped DuckDB extension
returns (`engines/duckdb/build/softavg.duckdb_extension`, sha256/16 recorded) for
`GROUP BY query_id, facet` over the full 7.1M-row table. 5,758,251 brand groups and 76,888
category groups, every eps in the grid and inf: max relative difference 3.2e-15. Run with
DuckDB 1.5.5 python. The per-eps seconds there are smoke numbers on a shared machine.

## Reproduce

```
python3 scripts/s1_fetch_esci.py          # 20 s: data/esci_us.parquet, data/catalog_us.parquet
python3 scripts/s2_join_meta.py 10        # ~45 min, streams 26 GB gz; data/meta_join.parquet (kept)
python3 scripts/s2b_coverage.py           # data/coverage.json
python scripts/s3_build_table.py --min-labelled 8 --n-distract 200 --batch 64
python3 scripts/s4_evaluate.py --boot 1000                                  # ~20 min
python3 scripts/s4_evaluate.py --boot 200 --pools labelled --delta-rating 0.1 --delta-price 0.1 --out data/results_delta_lo.json
python3 scripts/s4_evaluate.py --boot 200 --pools labelled --delta-rating 0.3 --delta-price 0.4 --out data/results_delta_hi.json
python3 scripts/s5_engine_check.py
python3 scripts/s6_summarise.py && python3 scripts/make_figure.py
```

s3 must run in the sigmod2027 venv: the default `~/.local` env has torch 2.4.0 with
transformers 5.9, and `import sentence_transformers` fails there. Kept: `data/meta_join.parquet`
(11 MB, the expensive step), `data/meta_join_report.json`, `data/coverage.json`,
`data/queries.parquet`, `data/engine_check.json`. Deleted after the run: the ESCI tables
(refetch with s1), `task.parquet` (194 MB, rebuild with s3; the distractor draw is seeded, the
embeddings are recomputed), and the raw s4 outputs (folded into results.json).

## Caveats that matter for the paper

- Values are product-level (`average_rating`, `price`), not review-level. A "finding" is about
  which products are relevant, not about review text.
- The labelled pool is ESCI's own candidate set (already retrieved by Amazon's search engine,
  so mostly relevant: 67% E). The +D pool is our own construction. Its random in-category
  distractors are easy (mean s 0.15), and still the facet-selection problem dominates the brand
  results.
- Brand facets are mostly singletons (240k groups for 33k queries). Only 6,366 (rating, E) test
  facets are mixed, and that is the only place any estimator can differ from AVG.

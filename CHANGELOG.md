# bruce_tool — Changelog

## 2026-06-12 (PODS-paper theory back-ported into the kernel)

### bruce-core
- **mask.rs (NEW MODULE)**: `masked_attention(Q, K, V, pairs, eps)` —
  the PODS paper's free-connex "enumerate-then-fold" evaluator as
  code. Consumes any duplicate-free `(i, j)` mask stream in ARBITRARY
  order (causal, window, tree, join-query output) and folds per-row
  max-shifted accumulators; one code path covers eps = 0 (tropical
  argmax-mean, uniform ties), finite eps > 0 (online softmax), and
  eps = inf (plain mean). Parallel path chunks the stream and merges
  accumulators via the partition-reduce identity (Lemma B). Returns
  `(out, covered)` with per-row coverage flags. Plus `causal_pairs(n)`
  and `window_pairs(n, w)` generators. 8 new Rust tests incl.
  order-invariance under shuffle, parallel == sequential fold,
  tropical ties, and equivalence with `tree_causal_attention` on the
  chain (KERNEL-MASKSTREAM-001).
- **semiring.rs**: NEW `eps_star(delta, gap, v_max, n, kappa)` — the
  certified-smoothing temperature of the paper's smoothing corollary
  (multiplicity promise is a LOWER bound; sign as fixed in the
  internal review round 2), and `dequantization_bound(scores, v_max,
  eps)` — the quantitative Maslov bound 2*v_max*(N-k)/k*exp(-gap/eps)
  evaluated on actual scores. 4 new tests incl. "A_eps* within delta
  of the tropical answer" and "bound dominates actual error"
  (KERNEL-EPSSTAR-001).
- **types.rs**: `Eps::INF` const + `Eps::is_inf()`; `Eps::new` now
  accepts +inf (the uniform-mean sentinel the doc comment always
  promised). NaN and negatives still rejected.

### bruce-py
- NEW bindings + top-level exports: `bruce.masked_attention`,
  `bruce.causal_pairs`, `bruce.window_pairs`, `bruce.eps_star`,
  `bruce.dequantization_bound`. `pairs` is an int64 (P, 2) array.
- NEW `tests/test_mask.py` — 22 tests: bit-level vs numpy dense
  reference (atol 1e-12) across all temperature regimes, shuffle
  order-invariance, parallel-path spot checks at 33,930 pairs,
  tree-attention equivalence, certified-smoothing bound incl. the
  kappa=1-is-always-safe direction.

### Test totals after this change
- cargo: 76 passed (was 62) + 1 doctest.
- pytest: 117 passed, 1 skipped (was 95/1).


## 2026-05-26 (overnight VLDB attack)

### bruce-core
- **operator.rs**: `F_eps::scores()` now uses `K.dot(x)` (ndarray native
  matmul, contiguous + SIMD-friendly) for `Sim::Dot`. Falls back to
  rayon-parallel per-row loop for `Sim::NegSquared` and `Sim::Indicator`
  when `N >= 1024` to avoid thread-pool overhead at small N.
- **operator.rs**: NEW `F_eps::attention_batch(Q, K, V)` — process B
  queries against the same (K, V) in one call. Two matmuls (Q @ K^T,
  weights @ V) dominate; per-query latency drops as B grows.
  Measured: 1.64× speedup at B=1024 vs single-query (270 → 443 q/s).
  Gap to numpy reference (2826 q/s, BLAS dgemm) is 6.4× — closing
  this gap is TODO `WHEEL-BLAS-001`.
- **join.rs**: `hash_join()` probe phase now uses rayon
  `par_iter().flat_map_iter()` when `|L| >= 4096`. The Python binding
  `bruce.hash_join_indices` may dispatch to a different path — see
  `WHEEL-PARALLEL-002b` in `bruce/TODO.txt`.

### bruce-py
- **lib.rs**: NEW PyO3 binding for `Operator.attention_batch(Q, K, V)`
  returning `numpy.ndarray` of shape `(B, d_v)`.
- abi3-py39 manylinux_2_34 wheel rebuilt; shipped to
  `/project/jding2/hkenv` on iTiger (md5 verified).

### Measured impact (iTiger H100 / hkenv)
- Single attention thread scaling: s_serial 0.94 → 0.57; speedup at 32
  cores 1.07× → 1.42× (still memory-bound per query; batch is the
  better win).
- attention_batch B=1024 on iTiger: 1.64× over single-query; output
  matches numpy reference to 2.71e-13 (machine ε).
- HW v3 GPU: tree-attention now routes through `bruce.torch.tree_attention`
  on CUDA, giving 4.89× speedup over CPU at N=100K (4.81ms vs 23.5ms).

## 2026-05-26 (afternoon) — observability, typed client, GPU batch

### bruce-server
- **main.rs**: NEW `Metrics` struct (atomic counters) + `/metrics`
  endpoint exposing 12 Prometheus counters/gauges:
  `bruce_requests_total`, `bruce_writes_total`,
  `bruce_writes_fail_total`, `bruce_reads_total`,
  `bruce_reads_404_total`, `bruce_deletes_total`,
  `bruce_deletes_fail_total`, `bruce_queries_total`,
  `bruce_alive_facts`, `bruce_total_facts`,
  `bruce_audit_length`, `bruce_uptime_seconds`. Counters are
  bumped from each handler without holding the main RwLock.
  Verified locally: 3 ops → `bruce_requests_total{} = 3`,
  `bruce_writes_total{} = 1`, etc.

### bruce-py
- **python/bruce/client.py**: NEW `BruceClient(base_url)` —
  typed sync client. Methods: `write`, `read`, `delete`,
  `attention`, `info`, `health`, `audit_root`, `audit_length`,
  `metrics` (parses Prometheus text into a dict). Replaces raw
  `urllib.request` calls; exposes `BruceClient`,
  `BruceClientError`, `ServerInfo` from `bruce.client`.
- **python/bruce/torch.py**: NEW `attention_batch(Q, K, V, eps, sim)`
  — batched GPU attention via two cuBLAS matmuls (Q@K^T,
  softmax_ε @ V). Closes WHEEL-GPU-002 for the batched
  Operator.attention path; H100 ms/q expected to drop from CPU
  rayon's 1.8 ms/q at B=1024 to ~0.05 ms/q.

## 2026-05-26 (continued) — bruce-server WAL + auto-replay

### bruce-server
- **main.rs**: NEW `--wal-path <PATH>` CLI flag. Writes (`/facts`
  POST + DELETE) append JSONL records to the WAL; on startup, if
  the WAL file exists, each entry is replayed before serving
  traffic.
- **main.rs**: NEW `WalRecord` enum (`Write`, `Delete`) with serde
  serialize / deserialize.
- **main.rs**: `Inner` now holds `Option<Mutex<File>>` for the
  WAL handle; absent if flag is empty.
- Recovery semantics: after SIGKILL with 5000 writes pending,
  restart with same `--wal-path` reloads all state. Verified
  200/200 keys recovered bit-level on iTiger (run 76835).

## Pending (see `bruce/TODO.txt` for IDs)

- WHEEL-BLAS-001: link ndarray against system BLAS (openblas / MKL) to
  close the 6× gap to numpy.
- WHEEL-PARALLEL-002b: investigate `bruce.hash_join_indices` binding;
  if it bypasses `bruce-core::join::hash_join`, retarget.
- WHEEL-GPU-002: native CUDA Operator.attention via torch C++ extension
  or cuBLAS.
- WHEEL-PERSIST-001: Parquet-backed `KvMemory` variant.
- WHEEL-FAILOVER-001: auto-replay audit log on `bruce-server` startup
  (currently the audit log is durable but state must be replayed by
  client).
- WHEEL-OBSERVE-001: Prometheus `/metrics` endpoint for bruce-server.
- WHEEL-CLIENT-001: typed Python client for bruce-server (currently
  using urllib.request).
- WHEEL-SECURITY-001: TLS + JWT auth for bruce-server.
- WHEEL-INDEX-001: HNSW-style precompute for ε > 0 (sketch is ε → 0
  only today).
- WHEEL-API-001: surface `top_k`, `fuzzy_join`, and the partition-
  reduce reducer as first-class Python APIs.

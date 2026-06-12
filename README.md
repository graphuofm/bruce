# Bruce

> A privacy-aware retrieval-memory primitive, built on the F_ε operator
> that unifies relational-database equi-join and Transformer attention.

```text
            wⱼ(ε)  =  exp(sim(x, kⱼ) / ε)
            A_ε    =  (Σⱼ wⱼ · vⱼ)  /  (Σⱼ wⱼ)        — attention
            Q_ε    =   Σⱼ wⱼ · vⱼ                      — SQL-style sum
```

* **ε = 0** with `sim = indicator` is SQL equi-join + GROUP BY.
* **ε = 1** with `sim = dot product` is standard softmax attention.
* The operator interpolates continuously between the two via Maslov
  dequantization.

Bruce ships the algebra in Rust (the `bruce-core` crate), a Python
package with NumPy / PyTorch bindings (`pip install bruce`), a
`bruce` CLI binary, and a `bruce-server` HTTP service.

## Why

Relational databases and Transformer attention have evolved as two
disjoint computational traditions for half a century. Bruce identifies
them with a single operator and gives you a **toolkit**: exact CRUD,
exact unlearning, fuzzy joins, sub-quadratic tree-causal attention,
signed provenance, encrypted-at-rest storage, differential-privacy
releases, GDPR cascade-deletes, and a tamper-evident audit log — all
as composable primitives.

## Install

```bash
pip install bruce
```

That gives you the Python package backed by the Rust extension module.
For a from-source build:

```bash
git clone <repo>
cd bruce
cargo build --release          # Rust crates + CLI + HTTP server binaries
cd bruce-py && maturin develop --release   # Python wheel into your env
```

## Quick start

```python
import bruce, numpy as np

# F_ε attention at ε = 1.0 (standard softmax-attention)
op = bruce.Operator(eps=1.0, sim="dot")
out = op.attention(
    x=np.array([1.0, 0.0]),
    K=np.array([[1.0, 0.0], [0.0, 1.0]]),
    V=np.array([[10.0, 0.0], [0.0, 10.0]]),
)
# out ≈ [7.31, 2.69] — softmax weighted average

# F_ε at ε = 0 with indicator similarity = SQL equi-join + GROUP BY:
sql = bruce.Operator(eps=0.0, sim="indicator")
total = sql.sum(
    x=np.array([1.0, 0.0]),
    K=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    V=np.array([[5.0], [7.0], [99.0]]),
)
# total == [12.0]      (5 + 7, the two rows whose key matched)

# Sub-quadratic causal attention via a tree mask:
N, d = 1024, 64
Q = K = V = np.random.randn(N, d)
out = bruce.tree_attention(Q, K, V, bruce.balanced_binary_tree(N), eps=1.0)
# O(N log N · d) — vs O(N²·d) for full causal. Bit-exact on chain trees.
```

## Examples

The `examples/` directory contains 14 runnable scripts covering every
primitive:

```bash
python examples/01_quickstart.py         # ε sweep across SQL ↔ softmax
python examples/02_exact_unlearning.py   # bit-level GDPR erasure (O(d))
python examples/03_dp_release.py         # Laplace + Gaussian noise releases
python examples/04_audit_log.py          # tamper-evident Merkle log
python examples/05_signed_facts.py       # Ed25519 fact provenance
python examples/06_encrypted_at_rest.py  # AES-256-GCM envelope
python examples/07_fuzzy_sketch.py       # O(d_φ · d_v) state, any N
python examples/08_privacy_pipeline.py   # all five primitives composed
python examples/09_distributed_partition_reduce.py  # Lemma B across shards
python examples/10_streaming_chain_join.py          # online 3-way chain join
python examples/11_triangle_lftj.py                 # AGM-optimal triangle count
python examples/12_gdpr_cascade.py                  # multi-table cascade-delete
python examples/13_http_server.py                   # talk to a deployed bruce-server
python examples/14_tree_attention_paper_a1.py       # sub-quadratic tree-causal
```

Sample output from `02_exact_unlearning.py`:

```
Output with poison (should be ~999):     [998.82, 998.82, 998.82, 998.82]
Output after delete:                     [-0.061, 0.192, 0.172, -0.130]
Output if poison was never inserted:     [-0.061, 0.192, 0.172, -0.130]

Max abs error (delete vs never-inserted): 3.33e-16
→ This is the bit-level exact-unlearning guarantee.
```

## The primitives

### Core (Rust-backed, CPU)

| API | Behaviour |
|-----|-----------|
| `bruce.Operator` | F_ε attention and SUM, parameterised by ε and similarity |
| `bruce.IncrementalMemory` | O(d) insert / delete / update — Lemma A |
| `bruce.KvMemory` | durable K/V memory with audit log |
| `bruce.FuzzyJoinSketch` | linear-attention kernel sketch, O(d_φ · d_v) state |
| `bruce.tree_attention` | sub-quadratic causal attention on a tree mask |
| `bruce.{chain,balanced_binary,k_ary_balanced,star}_tree` | tree builders |
| `bruce.hash_join`, `bruce.sort_merge_join`, `bruce.lftj_three` | three equi-join algorithms |
| `bruce.StreamingChainJoin` | online 3-way chain join |
| `bruce.PartialTriple`, `bruce.combine`, `bruce.finalize` | partition-reduce (Lemma B) |

### Privacy

| API | Behaviour |
|-----|-----------|
| `bruce.MerkleAuditLog` | append-only tamper-evident log with inclusion proofs |
| `bruce.Identity` + `bruce.SignedFact` | Ed25519 keypair + signed facts |
| `bruce.LaplaceMechanism` | ε-DP via Laplace noise |
| `bruce.GaussianMechanism` | (ε, δ)-DP via Gaussian noise |
| `bruce.AnonymityGuard` | k-anonymity / l-diversity query guard |
| `bruce.EncryptedBlob` | AES-256-GCM encrypted-at-rest envelope |
| `bruce.cascade_delete` | multi-table erasure with a single signed receipt |

### GPU (optional, requires PyTorch)

| API | Behaviour |
|-----|-----------|
| `bruce.torch.attention` | F_ε attention on torch tensors (CPU or CUDA) |
| `bruce.torch.hybrid_attention` | structural + semantic in one CUDA kernel |

## CLI

```bash
$ bruce demo
Bruce CLI demo — F_ε attention on a 3-record memory
  ε = 0  (tropical / SQL): out = [10.0, 0.0]
  ε = 0.25:                out = [7.43, 2.66]
  ε = 1.0  (softmax):      out = [6.33, 5.22]
  ε = 4.0:                 out = [5.40, 7.40]
```

## HTTP server

```bash
# build + run
make server
./target/release/bruce-server --addr 0.0.0.0:8080 --d-k 128 --d-v 16

# in another shell
curl -s http://127.0.0.1:8080/health        # → ok
curl -s http://127.0.0.1:8080/info | jq
curl -s -X POST http://127.0.0.1:8080/facts \
  -H 'Content-Type: application/json' \
  -d '{"fact_id":"f1","k":[...],"v":[...],"owner":"alice"}'
```

Endpoints: `/health`, `/info`, `/facts` (POST/GET/DELETE),
`/query/attention` (POST), `/audit/{root,length,append}`. Owner-mismatch
deletes return HTTP 403. See `examples/13_http_server.py` for a Python
client. `make test-server` runs the smoke-test suite.

## Tests

```bash
cargo test -p bruce-core   --release   # 62 Rust unit tests
cargo test -p bruce-cli    --release
cargo test -p bruce-server --release
pytest bruce-py/tests/                  # 77 Python integration tests
```

All 139 tests must pass before any release. `make test` runs them all.

## Layout

```
bruce/
├── bruce-core/      Rust crate: F_ε + privacy primitives + tree attention
├── bruce-py/        Python wheel (PyO3 bindings + torch backend + tests)
├── bruce-cli/       `bruce` command-line binary
├── bruce-server/    `bruce-server` HTTP service (axum)
├── examples/        14 runnable demos covering every primitive
├── scripts/         test_bruce_server.sh and other dev helpers
├── docs/
├── Dockerfile       Reproducible build environment, ships CLI + server + wheel
└── Makefile         `make test`, `make demo`, `make python`, `make server`,
                      `make test-server`, `make docker`
```

## License

Apache-2.0.

## Status

Alpha. Useful as a research toolkit and as a reference implementation
of the F_ε operator. APIs may evolve before 1.0.

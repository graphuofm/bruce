"""Bruce — the F_ε operator.

A unified algebra of relational databases and Transformer attention,
with a privacy-aware retrieval-memory toolkit on top.

Core operators (Rust-backed, CPU):
    bruce.Operator              — F_ε attention: A_ε(x, K, V) at any ε
    bruce.IncrementalMemory     — O(d) insert/delete/update via Lemma A
    bruce.KvMemory              — durable K/V memory with audit log
    bruce.FuzzyJoinSketch       — linear-attention kernel trick
    bruce.tree_attention        — sub-quadratic tree-causal F_ε (paper A1)
    bruce.{chain,balanced_binary,k_ary_balanced,star}_tree — tree builders

Distributed (Lemma B partition-reduce):
    bruce.PartialTriple         — per-shard (m, num, den) state
    bruce.combine, bruce.finalize  — reduce shards → A_ε exactly

Streaming:
    bruce.StreamingChainJoin    — online 3-way chain join

Joins:
    bruce.hash_join             — O(N+M) inner equi-join
    bruce.sort_merge_join       — O(N+M) over sorted keys
    bruce.lftj_three            — Leapfrog Triejoin (AGM-optimal)

Privacy primitives:
    bruce.MerkleAuditLog        — tamper-evident append-only audit log
    bruce.Identity              — Ed25519 signing keypair
    bruce.SignedFact            — signed (fact_id, owner, payload)
    bruce.LaplaceMechanism      — ε-DP via Laplace noise
    bruce.GaussianMechanism     — (ε, δ)-DP via Gaussian noise
    bruce.AnonymityGuard        — k-anonymity / l-diversity guard
    bruce.EncryptedBlob         — AES-256-GCM encrypted-at-rest envelope

Optional GPU backend (requires PyTorch):
    bruce.torch.attention(...)
    bruce.torch.hybrid_attention(...)
    bruce.torch.tree_attention(Q, K, V, parents, eps=1.0)
        — GPU-vectorised tree-causal F_ε for H100-scale N; mirrors
          the Rust `bruce.tree_attention` modulo float precision.

Quickstart:
    >>> import bruce, numpy as np
    >>> op = bruce.Operator(eps=1.0, sim="dot")
    >>> out = op.attention(np.array([1., 0.]),
    ...                    np.array([[1., 0.], [0., 1.]]),
    ...                    np.array([[10., 0.], [0., 10.]]))
"""

from bruce._bruce import (   # noqa: F401
    AnonymityGuard,
    EncryptedBlob,
    FuzzyJoinSketch,
    GaussianMechanism,
    Identity,
    IncrementalMemory,
    KvMemory,
    LaplaceMechanism,
    MerkleAuditLog,
    Operator,
    PartialTriple,
    SignedFact,
    StreamingChainJoin,
    __version__,
    balanced_binary_tree,
    cascade_delete,
    chain_tree,
    combine,
    finalize,
    hash_join,
    hash_join_count,
    hash_join_indices,
    hash_join_reduce,
    k_ary_balanced_tree,
    lftj_three,
    sort_merge_join,
    star_tree,
    tree_attention,
    masked_attention,
    causal_pairs,
    window_pairs,
    eps_star,
    dequantization_bound,
)

try:
    from bruce import torch  # noqa: F401
except ImportError:
    pass

__all__ = [
    "AnonymityGuard",
    "EncryptedBlob",
    "FuzzyJoinSketch",
    "GaussianMechanism",
    "Identity",
    "IncrementalMemory",
    "KvMemory",
    "LaplaceMechanism",
    "MerkleAuditLog",
    "Operator",
    "PartialTriple",
    "SignedFact",
    "StreamingChainJoin",
    "__version__",
    "balanced_binary_tree",
    "cascade_delete",
    "chain_tree",
    "combine",
    "finalize",
    "hash_join",
    "hash_join_count",
    "hash_join_indices",
    "hash_join_reduce",
    "k_ary_balanced_tree",
    "lftj_three",
    "sort_merge_join",
    "star_tree",
    "masked_attention",
    "causal_pairs",
    "window_pairs",
    "eps_star",
    "dequantization_bound",
    "torch",
    "tree_attention",
]

# typed HTTP client for bruce-server (pure-Python module)
from bruce.client import BruceClient, BruceClientError, ServerInfo  # noqa: F401,E402

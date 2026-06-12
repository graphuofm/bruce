"""Paper A1 — tree-structured causal attention via the Bruce identity.

The identity:
    Causal-mask attention with mask M (where M[i,j] = j is an ancestor
    of i in some tree) is equivalent to running F_ε on each row's
    ancestor path. For a balanced binary tree, that's O(N log N · d)
    total work — vs O(N² · d) for full causal attention.

Run:
    python examples/14_tree_attention_paper_a1.py

This example:
  1. shows a chain tree recovers full causal attention bit-exact
  2. compares wall-clock chain (O(N²·d)) vs balanced binary (O(N log N · d))
  3. confirms the balanced-tree output and the chain output disagree
     on which positions attend to which — they're answering *different
     questions* (one is full causal, one is tree-causal)
"""

from __future__ import annotations

import time

import numpy as np

import bruce


def causal_softmax_reference(Q, K, V, eps=1.0):
    """Standard dense O(N²·d) causal softmax attention."""
    N = Q.shape[0]
    S = Q @ K.T
    S = np.where(np.triu(np.ones((N, N), bool), k=1), -np.inf, S)
    W = np.exp((S - S.max(axis=1, keepdims=True)) / eps)
    W /= W.sum(axis=1, keepdims=True)
    return W @ V


def main() -> None:
    print(f"Bruce {bruce.__version__}  —  Paper A1 demo")
    rng = np.random.default_rng(0)

    # 1. small-N bit-exactness: chain tree == full causal attention
    for N in [4, 8, 32]:
        d = 8
        Q = rng.normal(size=(N, d))
        K = rng.normal(size=(N, d))
        V = rng.normal(size=(N, d))
        parents = bruce.chain_tree(N)
        out = bruce.tree_attention(Q, K, V, parents, eps=1.0)
        ref = causal_softmax_reference(Q, K, V, eps=1.0)
        diff = float(np.max(np.abs(out - ref)))
        print(f"  N={N:<5d}  max |tree(chain) − causal|  =  {diff:.2e}")

    # 2. wall-clock: chain (O(N²·d)) vs balanced binary (O(N log N · d))
    print(f"\n{'N':<8s}  {'shape':<8s}  {'time':>8s}  {'expected':<20s}")
    print("-" * 60)
    d = 64
    for N in [1_024, 4_096, 16_384]:
        Q = rng.normal(size=(N, d)).astype(np.float64)
        K = rng.normal(size=(N, d)).astype(np.float64)
        V = rng.normal(size=(N, d)).astype(np.float64)

        for shape, parents, label in [
            ("chain",     bruce.chain_tree(N),            f"O(N²·d) = {N*N*d:_}"),
            ("balanced",  bruce.balanced_binary_tree(N),  f"O(N log N · d) = {int(N*np.log2(N)*d):_}"),
        ]:
            t0 = time.perf_counter()
            _ = bruce.tree_attention(Q, K, V, parents, eps=1.0)
            dt = time.perf_counter() - t0
            print(f"  N={N:<5d}  {shape:<8s}  {dt*1000:>6.1f}ms  {label}")

    # 3. demonstrate that the two trees answer different questions
    N, d = 16, 4
    Q = rng.normal(size=(N, d)); K = rng.normal(size=(N, d)); V = rng.normal(size=(N, d))
    out_chain = bruce.tree_attention(Q, K, V, bruce.chain_tree(N), eps=1.0)
    out_balbi = bruce.tree_attention(Q, K, V, bruce.balanced_binary_tree(N), eps=1.0)
    diff = float(np.linalg.norm(out_chain - out_balbi))
    print(f"\n‖chain − balanced‖_F = {diff:.3e}")
    print("(non-zero because chain attends over ALL prior tokens, "
          "balanced binary attends only over O(log N) ancestors)")


if __name__ == "__main__":
    main()

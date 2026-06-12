"""Tests for `bruce.tree_attention` — the wheel-native sub-quadratic
causal attention used by paper A1."""
from __future__ import annotations

import numpy as np
import pytest

import bruce


def _causal_softmax_reference(Q, K, V, eps=1.0):
    """Standard dense causal softmax attention, used as ground truth."""
    N = Q.shape[0]
    S = Q @ K.T
    mask = np.triu(np.ones((N, N), bool), k=1)
    S = np.where(mask, -np.inf, S)
    M = S.max(axis=1, keepdims=True)
    W = np.exp((S - M) / eps)
    W /= W.sum(axis=1, keepdims=True)
    return W @ V


def test_chain_tree_recovers_full_causal_attention():
    rng = np.random.default_rng(0)
    N, d = 8, 4
    Q = rng.normal(size=(N, d))
    K = rng.normal(size=(N, d))
    V = rng.normal(size=(N, d))
    parents = bruce.chain_tree(N)
    out = bruce.tree_attention(Q, K, V, parents, eps=1.0)
    ref = _causal_softmax_reference(Q, K, V, eps=1.0)
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_root_row_equals_value_row():
    """The root of the tree attends only to itself, so its output is V[root]."""
    rng = np.random.default_rng(1)
    N, d = 5, 3
    Q = rng.normal(size=(N, d))
    K = rng.normal(size=(N, d))
    V = rng.normal(size=(N, d))
    parents = bruce.balanced_binary_tree(N)
    assert parents[0] == -1
    out = bruce.tree_attention(Q, K, V, parents, eps=1.0)
    np.testing.assert_allclose(out[0], V[0], atol=1e-12)


def test_star_topology_two_row_softmax_at_each_leaf():
    """Star: every leaf attends to {leaf, root}, so output is a 2-row softmax."""
    rng = np.random.default_rng(2)
    N, d = 4, 2
    Q = rng.normal(size=(N, d))
    K = rng.normal(size=(N, d))
    V = rng.normal(size=(N, d))
    parents = bruce.star_tree(N)
    out = bruce.tree_attention(Q, K, V, parents, eps=1.0)
    # leaf i ∈ {1,2,3}: scores = [Q[i]·K[i], Q[i]·K[0]], softmax over those two
    for i in range(1, N):
        s = np.array([Q[i] @ K[i], Q[i] @ K[0]])
        w = np.exp(s - s.max()); w /= w.sum()
        expected = w[0] * V[i] + w[1] * V[0]
        np.testing.assert_allclose(out[i], expected, atol=1e-12)


def test_eps_zero_picks_argmax_ancestor():
    """At ε=0 each row collapses to picking the ancestor with the max dot product."""
    # 4-row chain. We pick K,Q so the argmax along each path is predictable.
    Q = np.array([[1.0], [1.0], [1.0], [1.0]])
    K = np.array([[1.0], [5.0], [3.0], [2.0]])     # row 1 always wins
    V = np.array([[10.], [99.], [30.], [40.]])
    parents = bruce.chain_tree(4)
    out = bruce.tree_attention(Q, K, V, parents, eps=0.0)
    # row 0 sees only [0]: out = V[0] = 10
    np.testing.assert_allclose(out[0], [10.0])
    # row 1 sees [1,0]: scores=[5,1], argmax=row 1, V[1]=99
    np.testing.assert_allclose(out[1], [99.0])
    # row 2 sees [2,1,0]: scores=[3,5,1], argmax=row 1, V[1]=99
    np.testing.assert_allclose(out[2], [99.0])
    # row 3 sees [3,2,1,0]: scores=[2,3,5,1], argmax=row 1, V[1]=99
    np.testing.assert_allclose(out[3], [99.0])


def test_balanced_binary_tree_topology():
    parents = bruce.balanced_binary_tree(7)
    assert parents == [-1, 0, 0, 1, 1, 2, 2]


def test_k_ary_balanced_tree():
    parents = bruce.k_ary_balanced_tree(7, 3)
    # parents[i] = (i-1)/3
    assert parents == [-1, 0, 0, 0, 1, 1, 1]


def test_rejects_forward_parent():
    Q = np.zeros((3, 1)); K = Q.copy(); V = Q.copy()
    with pytest.raises(ValueError):
        bruce.tree_attention(Q, K, V, [-1, 2, -1], eps=1.0)


def test_mismatched_dims_raise():
    Q = np.zeros((3, 2)); K = np.zeros((3, 3)); V = np.zeros((3, 2))
    with pytest.raises(ValueError):
        bruce.tree_attention(Q, K, V, bruce.chain_tree(3), eps=1.0)

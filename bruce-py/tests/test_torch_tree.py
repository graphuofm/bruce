"""Tests for `bruce.torch.tree_attention` — the GPU-vectorised
sub-quadratic causal attention from paper A1.

The CPU code path is exercised here because not every CI machine has
a CUDA device; the algorithm is device-agnostic and we lift it to
'cuda' inside a single test guarded by ``torch.cuda.is_available()``.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

import bruce
import bruce.torch as bt


# Clear the path cache between tests so we don't carry padding from
# a different N across tests.
@pytest.fixture(autouse=True)
def _clear_cache():
    bt._tree_path_cache_clear()
    yield
    bt._tree_path_cache_clear()


def test_chain_tree_matches_causal_sdpa_float32():
    """At eps=1, chain_tree tree_attention is causal full attention.

    Compare against PyTorch's ``F.scaled_dot_product_attention`` with
    ``is_causal=True`` and ``scale=1.0`` (so the dot scores are not
    rescaled by 1/sqrt(d) — our op has no such rescale).
    """
    torch.manual_seed(0)
    N, d = 64, 16
    Q = torch.randn(N, d, dtype=torch.float32)
    K = torch.randn(N, d, dtype=torch.float32)
    V = torch.randn(N, d, dtype=torch.float32)

    parents = bruce.chain_tree(N)
    out = bt.tree_attention(Q, K, V, parents, eps=1.0)

    ref = F.scaled_dot_product_attention(
        Q.unsqueeze(0).unsqueeze(0),
        K.unsqueeze(0).unsqueeze(0),
        V.unsqueeze(0).unsqueeze(0),
        is_causal=True,
        scale=1.0,
    ).squeeze(0).squeeze(0)

    assert out.shape == ref.shape == (N, d)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_balanced_binary_consistent_with_rust():
    """The torch path must agree with the Rust CPU reference to float32
    tolerance on a balanced binary tree."""
    rng = np.random.default_rng(7)
    N, d_k, d_v = 31, 8, 4
    Q_np = rng.normal(size=(N, d_k))
    K_np = rng.normal(size=(N, d_k))
    V_np = rng.normal(size=(N, d_v))
    parents = bruce.balanced_binary_tree(N)

    ref = bruce.tree_attention(Q_np, K_np, V_np, parents, eps=1.0)

    Q = torch.from_numpy(Q_np).to(torch.float32)
    K = torch.from_numpy(K_np).to(torch.float32)
    V = torch.from_numpy(V_np).to(torch.float32)
    out = bt.tree_attention(Q, K, V, parents, eps=1.0)

    assert out.shape == (N, d_v)
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-5, rtol=1e-5)


def test_balanced_binary_consistent_with_rust_float64():
    """In float64 the torch path is bit-equivalent (within 1e-12) to
    the Rust reference — useful as a stricter sanity check."""
    rng = np.random.default_rng(11)
    N, d_k, d_v = 64, 12, 5
    Q_np = rng.normal(size=(N, d_k))
    K_np = rng.normal(size=(N, d_k))
    V_np = rng.normal(size=(N, d_v))
    parents = bruce.balanced_binary_tree(N)

    ref = bruce.tree_attention(Q_np, K_np, V_np, parents, eps=1.0)
    out = bt.tree_attention(
        torch.from_numpy(Q_np),
        torch.from_numpy(K_np),
        torch.from_numpy(V_np),
        parents,
        eps=1.0,
    )
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-12, rtol=1e-12)


def test_eps_zero_picks_argmax_ancestor():
    """At eps=0 each row collapses to the argmax-scoring ancestor.

    Hand-crafted chain of N=4 where row 1's K dominates the dot
    product along every path, so every row returns V[1]=[99].
    """
    Q = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float64)
    K = torch.tensor([[1.0], [5.0], [3.0], [2.0]], dtype=torch.float64)
    V = torch.tensor([[10.0], [99.0], [30.0], [40.0]], dtype=torch.float64)
    parents = bruce.chain_tree(4)

    out = bt.tree_attention(Q, K, V, parents, eps=0.0)

    # Row 0 sees only itself: V[0] = 10.
    torch.testing.assert_close(out[0], torch.tensor([10.0], dtype=torch.float64))
    # Rows 1,2,3: K[1]=5 wins along every path -> V[1] = 99.
    for i in (1, 2, 3):
        torch.testing.assert_close(
            out[i], torch.tensor([99.0], dtype=torch.float64),
            msg=f"row {i} did not pick the argmax ancestor",
        )


def test_root_row_is_value_row():
    rng = np.random.default_rng(3)
    N, d = 16, 4
    Q = torch.from_numpy(rng.normal(size=(N, d)))
    K = torch.from_numpy(rng.normal(size=(N, d)))
    V = torch.from_numpy(rng.normal(size=(N, d)))
    parents = bruce.balanced_binary_tree(N)
    out = bt.tree_attention(Q, K, V, parents, eps=1.0)
    torch.testing.assert_close(out[0], V[0], atol=1e-12, rtol=1e-12)


def test_accepts_numpy_and_tensor_parents():
    """parents may be a list, numpy array, or torch tensor."""
    rng = np.random.default_rng(5)
    N, d = 16, 4
    Q = torch.from_numpy(rng.normal(size=(N, d)))
    K = torch.from_numpy(rng.normal(size=(N, d)))
    V = torch.from_numpy(rng.normal(size=(N, d)))
    p_list = bruce.balanced_binary_tree(N)
    p_np = np.asarray(p_list, dtype=np.int64)
    p_t = torch.as_tensor(p_list, dtype=torch.int64)

    out_list = bt.tree_attention(Q, K, V, p_list, eps=1.0)
    out_np = bt.tree_attention(Q, K, V, p_np, eps=1.0)
    out_t = bt.tree_attention(Q, K, V, p_t, eps=1.0)
    torch.testing.assert_close(out_list, out_np)
    torch.testing.assert_close(out_list, out_t)


def test_rejects_forward_parent():
    Q = torch.zeros(3, 1)
    K = torch.zeros(3, 1)
    V = torch.zeros(3, 1)
    with pytest.raises(ValueError):
        bt.tree_attention(Q, K, V, [-1, 2, -1], eps=1.0)


def test_batched_4d_shape():
    """Batched (B, H, N, d) input goes through with shared parents."""
    torch.manual_seed(13)
    B, H, N, d = 2, 3, 16, 4
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    parents = bruce.balanced_binary_tree(N)
    out = bt.tree_attention(Q, K, V, parents, eps=1.0)
    assert out.shape == (B, H, N, d)
    # Each (b, h) slice must match the per-slice computation.
    slice_00 = bt.tree_attention(Q[0, 0], K[0, 0], V[0, 0], parents, eps=1.0)
    torch.testing.assert_close(out[0, 0], slice_00, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA device available")
def test_cuda_path_matches_cpu():
    """When a GPU is present, CUDA results match CPU within float32 noise."""
    torch.manual_seed(17)
    N, d = 128, 16
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    parents = bruce.balanced_binary_tree(N)

    out_cpu = bt.tree_attention(Q, K, V, parents, eps=1.0)
    out_gpu = bt.tree_attention(
        Q.cuda(), K.cuda(), V.cuda(), parents, eps=1.0
    ).cpu()
    torch.testing.assert_close(out_cpu, out_gpu, atol=1e-5, rtol=1e-5)

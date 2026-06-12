"""GPU-accelerated F_ε implementation, backed by PyTorch.

This module provides the same algebra as the Rust `bruce_core::operator::F_eps`,
but evaluated on torch tensors so it runs on CUDA. Use this when you
have a GPU and your inputs are torch tensors; use the Rust wheel
(via `import bruce`) when you want portable CPU.

Both paths produce bit-identical results within float64 precision.

Quickstart:
    >>> import torch, bruce.torch as bt
    >>> x = torch.randn(64, device='cuda', dtype=torch.float64)
    >>> K = torch.randn(1_000_000, 64, device='cuda', dtype=torch.float64)
    >>> V = torch.randn(1_000_000, 32, device='cuda', dtype=torch.float64)
    >>> out = bt.attention(x, K, V, eps=1.0, sim='dot')
    >>> out.shape
    torch.Size([32])

For sub-quadratic tree-causal attention on H100-scale N, see
`tree_attention(Q, K, V, parents, eps)`.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch


def attention(x: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              eps: float = 1.0, sim: str = "dot") -> torch.Tensor:
    """F_ε attention: A_ε(x, K, V) = softmax(score / ε) · V.

    Args:
        x: query, shape (d_k,)
        K: keys,  shape (N, d_k)
        V: values, shape (N, d_v)
        eps: temperature ε ≥ 0. ε=0 triggers tropical / SQL semantics.
        sim: 'dot' | 'negsq' | 'indicator'

    Returns: (d_v,) attention output on the same device as the inputs.
    """
    scores = _score(x, K, sim)
    if eps == 0.0:
        # tropical: uniform over argmax (matches bruce-core semantic)
        m = scores.max()
        is_max = (scores == m)
        n_argmax = is_max.sum().to(scores.dtype)
        weights = is_max.to(scores.dtype) / n_argmax.clamp(min=1)
        return weights @ V
    # softmax shifted by m for numerical stability
    m = scores.max()
    e = torch.exp((scores - m) / eps)
    return (e @ V) / e.sum()


def attention_batch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     eps: float = 1.0, sim: str = "dot") -> torch.Tensor:
    """Batched F_ε attention on GPU. WHEEL-GPU-002.

    Q: (B, d_k), K: (N, d_k), V: (N, d_v). Returns (B, d_v).

    Two large matmuls dominate:
      scores = Q @ K^T        # (B, N)
      out    = softmax_ε(scores) @ V   # (B, d_v)
    Both go through torch's BLAS/cuBLAS path, so on H100 this is
    massively faster than the bruce-wheel CPU rayon attention_batch.

    Only `sim='dot'` is supported in batch mode.
    """
    if sim != "dot":
        raise NotImplementedError(
            f"attention_batch only supports sim='dot' (got {sim!r})")
    scores = Q @ K.t()                                   # (B, N)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores / eps)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights @ V                                    # (B, d_v)


def sum_op(x: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
           eps: float = 1.0, sim: str = "dot") -> torch.Tensor:
    """F_ε un-normalised sum: Q_ε(x, K, V) = Σⱼ wⱼ · vⱼ.

    At ε=0 with sim='indicator' this is `SELECT SUM(v) WHERE k = x`.
    """
    scores = _score(x, K, sim)
    if eps == 0.0:
        m = scores.max()
        # mask -inf scores so they don't contribute
        live = torch.isfinite(scores) & (scores == m)
        return live.to(V.dtype) @ V
    m = scores.max()
    e = torch.exp((scores - m) / eps)
    # return m-shifted SUM (caller multiplies exp(m/ε) back if needed)
    return e @ V * torch.exp(m / eps)


def _score(x: torch.Tensor, K: torch.Tensor, sim: str) -> torch.Tensor:
    if sim == "dot":
        return K @ x
    if sim == "negsq":
        diff = K - x
        return -0.5 * (diff * diff).sum(dim=-1)
    if sim == "indicator":
        diff = K - x
        l2 = (diff * diff).sum(dim=-1)
        out = torch.where(l2 == 0,
                          torch.zeros_like(l2),
                          torch.full_like(l2, float("-inf")))
        return out
    raise ValueError(f"unknown sim {sim!r}; expected dot, negsq, indicator")


def hybrid_attention(x: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     structural_mask: torch.Tensor,
                     eps: float = 1.0, sim: str = "dot",
                     top_k: int | None = None) -> torch.Tensor:
    """Hybrid query in one pass on GPU.

    Computes F_ε attention but only over rows where `structural_mask[j] = True`.
    Equivalent to two-pass `SQL filter then ANN`, but in one CUDA kernel.

    Returns:
      If top_k is None: (d_v,) attention output (zero if no survivors).
      Else: (top_k,) indices of top-K rows by softmax weight.
    """
    scores = _score(x, K, sim)
    masked = torch.where(structural_mask,
                          scores,
                          torch.full_like(scores, float("-inf")))
    if not torch.any(structural_mask):
        if top_k is not None:
            return torch.empty(0, dtype=torch.int64, device=x.device)
        return torch.zeros(V.shape[-1], device=x.device, dtype=V.dtype)

    m = masked[torch.isfinite(masked)].max()
    if eps == 0.0:
        is_max = masked == m
        if top_k is not None:
            return torch.nonzero(is_max, as_tuple=False).squeeze(-1)[:top_k]
        n_argmax = is_max.sum().to(V.dtype)
        weights = is_max.to(V.dtype) / n_argmax.clamp(min=1)
        return weights @ V
    weights = torch.exp((masked - m) / eps)
    weights = torch.where(structural_mask, weights, torch.zeros_like(weights))

    if top_k is not None:
        n_alive = int(structural_mask.sum())
        k = min(top_k, n_alive)
        _, idx = torch.topk(weights, k=k)
        return idx
    return (weights @ V) / weights.sum()


# ----------------------------------------------------------------------------
# Tree-structured causal attention (paper A1) — GPU vectorised over rows.
# Mirrors `bruce_core::tree::tree_causal_attention` and the Rust binding
# `bruce.tree_attention`.
# ----------------------------------------------------------------------------

# Cache of (parents_signature, device) -> (paths, valid_mask, max_depth).
# `paths` is (N, max_depth) int64 on `device`, with row i listing
# [i, parents[i], parents[parents[i]], ...] padded with 0 once the path
# terminates. `valid_mask` is (N, max_depth) bool, True where the entry
# is a real ancestor and False on the pad.
_TREE_PATH_CACHE: "dict[tuple, tuple[torch.Tensor, torch.Tensor, int]]" = {}


def _parents_to_tuple(parents) -> Tuple[int, ...]:
    """Canonicalise parents to a hashable tuple of int."""
    if isinstance(parents, tuple):
        return parents
    if isinstance(parents, torch.Tensor):
        return tuple(parents.detach().to("cpu", torch.int64).tolist())
    if hasattr(parents, "tolist"):
        # numpy array or similar
        return tuple(int(x) for x in parents.tolist())
    return tuple(int(x) for x in parents)


def _build_paths(parents_t: Tuple[int, ...], device: torch.device,
                 ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build (paths, valid_mask, max_depth) for a parents vector.

    Walks each row's ancestor chain on CPU using plain Python (parents
    has only N entries and the work is O(Σ depth) = O(N log N) for a
    balanced tree, O(N²) for a chain — but the result is cached). Then
    pads with 0 and ships the int64 index tensor + bool mask to `device`.
    """
    N = len(parents_t)
    # Validate (mirror the Rust check).
    for i, p in enumerate(parents_t):
        if p < -1 or p >= i:
            raise ValueError(
                f"parents[{i}] = {p}; must be in [-1, {i})"
            )

    # First pass: compute each row's depth via memoisation on parent.
    depth = [0] * N
    for i in range(N):
        p = parents_t[i]
        depth[i] = 1 + (depth[p] if p >= 0 else 0)
    max_depth = max(depth) if N > 0 else 0

    # Second pass: write the paths.
    paths_cpu = torch.zeros((N, max_depth), dtype=torch.int64)
    mask_cpu = torch.zeros((N, max_depth), dtype=torch.bool)
    for i in range(N):
        j = i
        s = 0
        while j != -1:
            paths_cpu[i, s] = j
            mask_cpu[i, s] = True
            j = parents_t[j]
            s += 1

    paths = paths_cpu.to(device)
    mask = mask_cpu.to(device)
    return paths, mask, max_depth


def _get_cached_paths(parents, device: torch.device,
                      ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    parents_t = _parents_to_tuple(parents)
    key = (parents_t, str(device))
    hit = _TREE_PATH_CACHE.get(key)
    if hit is None:
        hit = _build_paths(parents_t, device)
        _TREE_PATH_CACHE[key] = hit
    return hit


def tree_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                   parents, eps: float = 1.0) -> torch.Tensor:
    """Tree-structured causal attention, GPU-vectorised.

    For each row i, computes softmax-weighted attention over i and its
    ancestors under `parents` (with -1 marking a root). This recovers
    full causal attention when `parents = chain_tree(N)`, and runs in
    O(N · max_depth · d) total work — O(N log N · d) for balanced trees.

    Args:
        Q: queries, shape ``(N, d_k)`` or batched ``(B, H, N, d_k)``.
        K: keys, same leading dims as Q, last dim must match Q.
        V: values, same leading dims as Q, last dim ``d_v`` is free.
        parents: length-N int sequence with ``parents[i] ∈ {-1, 0..i-1}``;
            accepted as ``list[int]``, 1-D ``numpy.ndarray``, or a
            1-D ``torch.Tensor``. Built once per unique parents vector
            and cached (keyed by value + device).
        eps: temperature ε ≥ 0.
            * ``eps > 0`` -> softmax over ``(scores - row_max) / eps``.
            * ``eps == 0`` -> uniform over the argmax ancestor(s)
              (tropical / max-plus limit, matches the Rust semantic).

    Returns:
        Tensor of shape ``(N, d_v)`` or ``(B, H, N, d_v)`` on the same
        device and dtype as the inputs.

    Numerics: equivalent to ``bruce.tree_attention`` (the Rust CPU
    implementation) modulo float32 vs float64 precision. The path
    index tensor itself is int64 and lives on the input device, so
    only the padded ``(N, max_depth)`` index matrix is moved between
    host and GPU per unique tree.
    """
    if Q.shape != K.shape:
        raise ValueError(
            f"Q and K must have the same shape; got {tuple(Q.shape)} and {tuple(K.shape)}"
        )
    if Q.shape[:-1] != V.shape[:-1]:
        raise ValueError(
            f"Q and V must agree on all but last dim; got {tuple(Q.shape)} and {tuple(V.shape)}"
        )
    if eps < 0:
        raise ValueError(f"eps must be non-negative; got {eps}")

    # Collapse leading dims to a single batch axis B'. The path tensor
    # is shared across all batch elements (the tree is per-sequence).
    if Q.dim() == 2:
        squeeze_back = True
        Q2 = Q.unsqueeze(0)            # (1, N, d_k)
        K2 = K.unsqueeze(0)
        V2 = V.unsqueeze(0)
    elif Q.dim() == 4:
        squeeze_back = False
        B, H, N, d_k = Q.shape
        Q2 = Q.reshape(B * H, N, d_k)
        K2 = K.reshape(B * H, N, d_k)
        V2 = V.reshape(B * H, N, V.shape[-1])
    elif Q.dim() == 3:
        squeeze_back = False
        Q2, K2, V2 = Q, K, V
    else:
        raise ValueError(
            f"Q must be 2-D (N, d) or 3-D (B, N, d) or 4-D (B, H, N, d); got {Q.dim()}-D"
        )

    Bp, N, d_k = Q2.shape
    d_v = V2.shape[-1]

    paths, mask, max_depth = _get_cached_paths(parents, Q.device)
    if paths.shape[0] != N:
        raise ValueError(
            f"parents length ({paths.shape[0]}) must match sequence length N ({N})"
        )

    # Gather K and V along the sequence axis using `paths` (N, P).
    # Use plain fancy indexing — `K2[:, paths_flat, :]` selects N*P
    # rows in one shot, then reshape to (Bp, N, P, d_k).
    P = max_depth
    paths_flat = paths.reshape(-1)                          # (N*P,)
    K_idx = K2[:, paths_flat, :].reshape(Bp, N, P, d_k)     # (Bp, N, P, d_k)
    V_idx = V2[:, paths_flat, :].reshape(Bp, N, P, d_v)     # (Bp, N, P, d_v)

    # Scores: (Bp, N, P) = sum over d_k of Q[Bp,N,:,d_k] * K_idx[Bp,N,P,d_k].
    scores = torch.einsum("bnd,bnpd->bnp", Q2, K_idx)

    # Mask invalid (pad) entries with -inf so they cannot win softmax.
    neg_inf = torch.full_like(scores, float("-inf"))
    scores = torch.where(mask, scores, neg_inf)

    if eps == 0.0:
        # Tropical: uniform over the row-wise argmax set (within the
        # valid mask). Matches `softmax_eps(_, Eps::ZERO)` in bruce-core.
        row_max = scores.max(dim=-1, keepdim=True).values        # (Bp, N, 1)
        is_max = (scores == row_max) & mask
        n_argmax = is_max.sum(dim=-1, keepdim=True).to(V2.dtype)
        weights = is_max.to(V2.dtype) / n_argmax.clamp(min=1)
    else:
        row_max = scores.max(dim=-1, keepdim=True).values        # (Bp, N, 1)
        # Wherever a whole row is masked-out (shouldn't happen because
        # row i always includes i itself), row_max would be -inf. Guard
        # by replacing -inf -> 0 before subtracting; resulting weights
        # are zero anyway because of the mask.
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        shifted = (scores - row_max) / eps
        # exp on -inf -> 0, which is what we want for the pad entries.
        w = torch.exp(shifted)
        w = torch.where(mask, w, torch.zeros_like(w))
        z = w.sum(dim=-1, keepdim=True)
        weights = w / z.clamp(min=torch.finfo(w.dtype).tiny)

    # out[b, i, :] = Σ_p weights[b, i, p] · V_idx[b, i, p, :]
    out = torch.einsum("bnp,bnpd->bnd", weights, V_idx)

    if squeeze_back:
        return out.squeeze(0)
    if Q.dim() == 4:
        B, H, N, _ = Q.shape
        return out.reshape(B, H, N, d_v)
    return out


def _tree_path_cache_clear() -> None:
    """Drop the cached (paths, mask) tensors. Intended for tests."""
    _TREE_PATH_CACHE.clear()

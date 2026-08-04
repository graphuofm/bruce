"""Tests for the fused grouped_softavg physical operator."""
import numpy as np
import pytest

import bruce

RNG = np.random.default_rng(7)
N, DK, DV, G = 800, 6, 2, 9
K = RNG.normal(size=(N, DK))
V = RNG.normal(size=(N, DV))
X = RNG.normal(size=DK)
GID = (np.arange(N) * 13 % G).astype(np.uint32)
SEL = (np.arange(N) % 4 != 0)


def pair_reference(eps, sel=None):
    rows = np.arange(N) if sel is None else np.flatnonzero(sel)
    pairs = np.column_stack([GID[rows].astype(np.int64), rows.astype(np.int64)])
    Q = np.repeat(X.reshape(1, -1), G, axis=0)
    out, cov = bruce.masked_attention(Q, K, V, pairs, eps=eps)
    return out, cov


@pytest.mark.parametrize("eps", [0.0, 0.3, 1.0, float("inf")])
@pytest.mark.parametrize("use_sel", [False, True])
def test_matches_pair_path(eps, use_sel):
    sel = SEL if use_sel else None
    want, want_cov = pair_reference(eps, sel)
    got, got_cov = bruce.grouped_softavg(X, K, V, GID, G, eps=eps, sel=sel)
    assert list(got_cov) == list(want_cov)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_fused_filter_never_scores_filtered_rows():
    # a filtered row with a huge score must not leak into the answer
    K2 = K.copy()
    K2[0] = X * 1e6          # would dominate its group if scored
    sel = np.ones(N, bool)
    sel[0] = False
    got, _ = bruce.grouped_softavg(X, K2, V, GID, G, eps=0.5, sel=sel)
    ref, _ = bruce.grouped_softavg(X, K, V, GID, G, eps=0.5, sel=sel)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_empty_group_reported_uncovered():
    gid = GID.copy()
    gid[gid == 3] = 4        # group 3 becomes empty
    _, cov = bruce.grouped_softavg(X, K, V, gid, G, eps=1.0)
    assert not cov[3] and cov[4]


def test_bad_group_id_rejected():
    gid = GID.copy()
    gid[0] = G               # out of range
    with pytest.raises(ValueError):
        bruce.grouped_softavg(X, K, V, gid, G, eps=1.0)

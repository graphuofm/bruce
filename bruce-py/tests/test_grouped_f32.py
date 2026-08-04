"""Tests for the f32-storage scan path (grouped_softavg_f32 + KeyF32).

Precision contract under test: f32 storage/scoring, f64 accumulation.
The f64 reference is always run on the SAME stored numbers (the f32
values upcast), so the comparison isolates scoring arithmetic.
"""
import numpy as np
import pytest

import bruce

RNG = np.random.default_rng(11)
N, DK, DV, G = 1500, 16, 2, 7
K32 = RNG.normal(size=(N, DK)).astype(np.float32)
K64 = K32.astype(np.float64)                  # identical stored numbers
V = RNG.normal(size=(N, DV)) + 2.0            # offset: rel err well-defined
X32 = RNG.normal(size=DK).astype(np.float32)
X64 = X32.astype(np.float64)
GID = (np.arange(N) * 17 % G).astype(np.uint32)
SEL = (np.arange(N) % 5 != 0)


@pytest.mark.parametrize("eps", [0.1, 1.0])
@pytest.mark.parametrize("use_sel", [False, True])
def test_f32_kernel_matches_f64_kernel(eps, use_sel):
    sel = SEL if use_sel else None
    want, want_cov = bruce.grouped_softavg(X64, K64, V, GID, G, eps=eps, sel=sel)
    got, got_cov = bruce.grouped_softavg_f32(X32, K32, V, GID, G, eps=eps, sel=sel)
    assert list(got_cov) == list(want_cov)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=0)


def test_f32_kernel_rejects_bad_group_id():
    gid = GID.copy()
    gid[0] = G
    with pytest.raises(ValueError):
        bruce.grouped_softavg_f32(X32, K32, V, gid, G, eps=1.0)


def test_sharp_eps_stays_finite_and_anchored():
    # eps = 1e-4 with O(1) score spread: unanchored exp() would
    # overflow; the max-shifted f64 fold must return finite answers on
    # the f32 path, converging to the per-group argmax value.
    out, cov = bruce.grouped_softavg_f32(X32, K32, V, GID, G, eps=1e-4)
    assert all(cov)
    assert np.isfinite(out).all()
    sims = (K64 @ X64)
    for g in range(G):
        rows = np.flatnonzero(GID == g)
        top = rows[np.argmax(sims[rows])]
        np.testing.assert_allclose(out[g], V[top], rtol=1e-3)


@pytest.fixture()
def toy(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "genre": ["A", "A", "A", "B", "B", "B"],
        "rating": [1.0, 3.0, 8.0, 0.5, 6.0, 2.0],
        "year": [2001.0, 2005.0, 1999.0, 2010.0, 2003.0, 1995.0],
    })
    pq = tmp_path / "toy.parquet"
    df.to_parquet(pq)
    emb = np.array([[0.99, 0.10], [0.95, 0.30], [0.05, 0.99],
                    [0.90, 0.40], [0.10, 0.99], [0.70, 0.70]])
    return pq, emb


TOY_Q = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) "
         "FROM movies WHERE year >= 2000 GROUP BY genre")
TOY_X = np.array([1.0, 0.0])


def test_session_f32_matches_f64(toy):
    pq, emb = toy
    s64 = bruce.QuerySession()
    s64.register_parquet("movies", str(pq))
    s64.attach_key("movies", "emb", emb)                       # float64
    s32 = bruce.QuerySession()
    s32.register_parquet("movies", str(pq))
    s32.attach_key("movies", "emb", emb.astype(np.float32))    # float32
    l64, v64, _ = s64.run(TOY_Q, {"q": TOY_X})
    l32, v32, explain = s32.run(TOY_Q, {"q": TOY_X})
    assert l32 == l64
    np.testing.assert_allclose(v32, v64, rtol=1e-4)
    assert "grouped_softavg" in explain


def test_session_f32_sharp_eps_finite(toy):
    pq, emb = toy
    s = bruce.QuerySession()
    s.register_parquet("movies", str(pq))
    s.attach_key("movies", "emb", emb.astype(np.float32))
    q = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.0001) "
         "FROM movies GROUP BY genre")
    labels, values, _ = s.run(q, {"q": TOY_X})
    assert set(labels) == {"A", "B"}
    assert np.isfinite(values).all()
    got = dict(zip(labels, values))
    # eps -> 0: the argmax row's rating per group
    assert got["A"] == pytest.approx(1.0, rel=1e-6)   # sim 0.99 row
    assert got["B"] == pytest.approx(0.5, rel=1e-6)   # sim 0.90 row


def test_session_f32_write_path(toy):
    pq, emb = toy
    s = bruce.QuerySession()
    s.register_parquet("movies", str(pq))
    s.attach_key("movies", "emb", emb.astype(np.float32))
    n = s.delete_where("movies", "year", "=", 2005.0)
    assert n == 1
    labels, values, _ = s.run(TOY_Q, {"q": TOY_X})
    assert dict(zip(labels, values))["A"] == pytest.approx(1.0, rel=1e-6)
    s.insert_row(
        "movies",
        {"rating": 9.0, "year": 2020.0},
        {"genre": "C"},
        {"emb": np.array([1.0, 0.0])},
    )
    labels, values, _ = s.run(TOY_Q, {"q": TOY_X})
    assert dict(zip(labels, values))["C"] == pytest.approx(9.0)


def test_attach_key_rejects_other_dtypes(toy):
    pq, emb = toy
    s = bruce.QuerySession()
    s.register_parquet("movies", str(pq))
    with pytest.raises(ValueError):
        s.attach_key("movies", "emb", emb.astype(np.int32))

"""End-to-end f32 maintained views (f32-tail track, 2026-08-03).

SoftAggView now serves KeyF32 columns: f32 storage/scoring, the same
f64 (m, num, den) state and group-inverse delete path as f64 views
(bruce-query/src/views.rs; rebuild-equivalence property tests live in
its mod tests). This suite pins the Python facade: create_view + run
serve the same answers as the viewless f32 scan path, and stay
maintained under insert_row AND delete_where.

The delete arm was a pinned typed error until 2026-08-03 (hnsw-finish
track): db.rs's per-view survivor capture read keys through the
KeyF64-only `views::key_col_of`; it now uses the dtype-polymorphic
`key_rows_f64`, so the whole write path is dtype-complete.
"""
import numpy as np
import pytest

import bruce

RNG = np.random.default_rng(23)
N, DK, G = 400, 8, 5
GENRES = [f"g{i}" for i in range(G)]
EPS = 0.1
SQL = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.1) "
       "FROM movies GROUP BY genre")
X = RNG.normal(size=DK)
EMB32 = RNG.normal(size=(N, DK)).astype(np.float32)


@pytest.fixture()
def pq(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "genre": [GENRES[i % G] for i in range(N)],
        "rating": RNG.normal(size=N) + 2.0,
        "year": np.full(N, 2000.0),
    })
    p = tmp_path / "movies.parquet"
    df.to_parquet(p)
    return p


def session(pq, with_view):
    s = bruce.QuerySession()
    s.register_parquet("movies", str(pq))
    s.attach_key("movies", "emb", EMB32)          # float32 -> KeyF32
    if with_view:
        s.create_view("v32", "movies", "genre", "rating", "emb", X, eps=EPS)
    return s


def chosen_plan(explain):
    return explain.split("== candidates ==")[0]


def test_create_view_and_run_on_f32_key(pq):
    # was a typed error before 2026-08-03 (pinned in bruce-query's
    # tests/error_totality.rs, now flipped)
    viewed = session(pq, with_view=True)
    plain = session(pq, with_view=False)
    lv, vv, explain = viewed.run(SQL, {"q": X})
    lp, vp, _ = plain.run(SQL, {"q": X})
    assert lv == lp
    assert np.isfinite(vv).all()
    # view scoring is sequential f32, the scan kernel's dot is 4-way
    # unrolled f32: same precision class, rel <= 1e-4 budget
    np.testing.assert_allclose(vv, vp, rtol=1e-4)
    assert "MaintainedViewScan" in chosen_plan(explain)


def test_view_maintained_under_inserts(pq):
    viewed = session(pq, with_view=True)
    plain = session(pq, with_view=False)
    for i in range(4):
        # f32-exact wire values: both sessions store identical bits
        key = RNG.normal(size=DK).astype(np.float32).astype(np.float64)
        scalars = {"rating": 5.0 + i, "year": 2001.0}
        labels = {"genre": GENRES[i % G]}
        viewed.insert_row("movies", scalars, labels, {"emb": key})
        plain.insert_row("movies", scalars, labels, {"emb": key})
    lv, vv, explain = viewed.run(SQL, {"q": X})
    lp, vp, _ = plain.run(SQL, {"q": X})
    assert lv == lp
    np.testing.assert_allclose(vv, vp, rtol=1e-4)
    # incremental maintenance kept the view servable
    assert "MaintainedViewScan" in chosen_plan(explain)


def test_delete_where_maintains_f32_view(pq):
    # FLIPPED 2026-08-03 (hnsw-finish track; was a pinned "KeyF64"
    # typed error). delete_where on an f32-VIEWED table now maintains
    # the view: the survivor capture reads the KeyF32 column through
    # db.rs's dtype-polymorphic key_rows_f64. The maintained answer
    # must track a viewless scan of the post-delete table — the
    # f32 -> f64 -> f32 wire round trip is bit-preserving, so the only
    # difference left is the view's sequential f32 dot vs the kernel's
    # 4-way unrolled one (the same 1e-4 budget as the other cases).
    viewed = session(pq, with_view=True)
    plain = session(pq, with_view=False)
    # ratings are N(2, 1): >= 2.0 deletes roughly half the rows,
    # including several group anchors (the bounded re-anchor path)
    n_del = viewed.delete_where("movies", "rating", ">=", 2.0)
    assert 0 < n_del < N
    assert plain.delete_where("movies", "rating", ">=", 2.0) == n_del
    lv, vv, explain = viewed.run(SQL, {"q": X})
    lp, vp, _ = plain.run(SQL, {"q": X})
    assert lv == lp
    np.testing.assert_allclose(vv, vp, rtol=1e-4)
    assert "MaintainedViewScan" in chosen_plan(explain)


def test_f32_view_wrong_param_not_served(pq):
    # a different bound vector must not be served from the view
    viewed = session(pq, with_view=True)
    other = RNG.normal(size=DK)
    _, _, explain = viewed.run(SQL, {"q": other})
    assert "MaintainedViewScan" not in chosen_plan(explain)

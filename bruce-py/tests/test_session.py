"""Tests for the QuerySession client surface."""
import numpy as np
import pytest

import bruce


@pytest.fixture()
def session(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "genre": ["A", "A", "A", "B", "B", "B"],
        "rating": [1.0, 3.0, 8.0, 0.5, 6.0, 2.0],
        "year": [2001.0, 2005.0, 1999.0, 2010.0, 2003.0, 1995.0],
    })
    pq = tmp_path / "toy.parquet"
    df.to_parquet(pq)
    s = bruce.QuerySession()
    s.register_parquet("movies", str(pq))
    emb = np.array([[0.99, 0.10], [0.95, 0.30], [0.05, 0.99],
                    [0.90, 0.40], [0.10, 0.99], [0.70, 0.70]])
    s.attach_key("movies", "emb", emb)
    return s


Q = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) "
     "FROM movies WHERE year >= 2000 GROUP BY genre")
X = np.array([1.0, 0.0])


def softmax(sims, vals, eps):
    w = np.exp((np.array(sims) - max(sims)) / eps)
    return float(w @ vals / w.sum())


def test_sql_end_to_end(session):
    labels, values, explain = session.run(Q, {"q": X})
    got = dict(zip(labels, values))
    assert got["A"] == pytest.approx(softmax([0.99, 0.95], [1.0, 3.0], 0.3), rel=1e-12)
    assert got["B"] == pytest.approx(softmax([0.90, 0.10], [0.5, 6.0], 0.3), rel=1e-12)
    assert "grouped_softavg" in explain and "rows never scored" in explain


def test_write_path_updates_answers(session):
    n = session.delete_where("movies", "year", "=", 2005.0)
    assert n == 1
    labels, values, _ = session.run(Q, {"q": X})
    got = dict(zip(labels, values))
    assert got["A"] == pytest.approx(1.0, rel=1e-9)  # only c1 remains in A

    session.insert_row(
        "movies",
        {"rating": 9.0, "year": 2020.0},
        {"genre": "C"},
        {"emb": np.array([1.0, 0.0])},
    )
    labels, values, _ = session.run(Q, {"q": X})
    assert dict(zip(labels, values))["C"] == pytest.approx(9.0)


def test_unbound_param_errors(session):
    with pytest.raises(ValueError):
        session.run(Q, {})

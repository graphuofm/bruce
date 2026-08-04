"""Workstream 10 (Python side) — error-path totality through
QuerySession: every method fed bad input raises a Python exception
(ValueError for value-shaped errors; TypeError only where PyO3's
argument extraction rejects the object before our code runs) and NEVER
aborts, segfaults, or leaks a Rust panic (pyo3_runtime.PanicException
derives from BaseException, so a panic would fail these tests).

Plus a session-lifecycle smoke: 200 queries + 50 writes in one
session with bounded RSS growth (< 50 MB) — catches leaks of the
wholesale kind (per-call table copies that never free).
"""
import resource

import numpy as np
import pytest

import bruce

Q = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) "
     "FROM movies WHERE year >= 2000 GROUP BY genre")
X = np.array([1.0, 0.0])
EMB = np.array([[0.99, 0.10], [0.95, 0.30], [0.05, 0.99],
                [0.90, 0.40], [0.10, 0.99], [0.70, 0.70]])


def make_parquet(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "genre": ["A", "A", "A", "B", "B", "B"],
        "rating": [1.0, 3.0, 8.0, 0.5, 6.0, 2.0],
        "year": [2001.0, 2005.0, 1999.0, 2010.0, 2003.0, 1995.0],
        "id": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })
    pq = tmp_path / "toy.parquet"
    df.to_parquet(pq)
    return pq


@pytest.fixture()
def session(tmp_path):
    s = bruce.QuerySession()
    s.register_parquet("movies", str(make_parquet(tmp_path)))
    s.attach_key("movies", "emb", EMB)
    return s


# ------------------------------------------------- register_parquet

def test_register_parquet_missing_file():
    s = bruce.QuerySession()
    with pytest.raises(ValueError):
        s.register_parquet("t", "/nonexistent/nope.parquet")


def test_register_parquet_garbage_file(tmp_path):
    junk = tmp_path / "junk.parquet"
    junk.write_bytes(b"definitely not a parquet footer")
    s = bruce.QuerySession()
    with pytest.raises(ValueError):
        s.register_parquet("t", str(junk))


def test_register_replace_drops_stale_views(tmp_path):
    # Defined semantics: re-registering a name replaces the table and
    # drops maintained views built on the old contents; queries then
    # answer from the new table (never from stale view state).
    s = bruce.QuerySession()
    pq = make_parquet(tmp_path)
    s.register_parquet("movies", str(pq))
    s.attach_key("movies", "emb", EMB)
    s.create_view("v", "movies", "genre", "rating", "emb", X, eps=0.3)
    s.register_parquet("movies", str(pq))  # replace: view dropped
    s.attach_key("movies", "emb", EMB)
    labels, values, explain = s.run(
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) "
        "FROM movies GROUP BY genre", {"q": X})
    assert set(labels) == {"A", "B"}
    assert "MaintainedViewScan" not in explain.split("== chosen plan ==")[1].split("==")[0]


# ------------------------------------------------------- attach_key

def test_attach_key_missing_table(session):
    with pytest.raises(ValueError):
        session.attach_key("nosuch", "emb", EMB)


def test_attach_key_row_mismatch(session):
    with pytest.raises(ValueError):
        session.attach_key("movies", "emb2", np.zeros((3, 2)))


def test_attach_key_wrong_ndim(session):
    with pytest.raises(ValueError):
        session.attach_key("movies", "emb2", np.zeros(6))


def test_attach_key_wrong_dtype(session):
    with pytest.raises(ValueError):
        session.attach_key("movies", "emb2", np.zeros((6, 2), dtype=np.int64))


# ------------------------------------------------------ create_view

def test_create_view_errors(session):
    with pytest.raises(ValueError):   # missing table
        session.create_view("v", "nosuch", "genre", "rating", "emb", X, eps=0.3)
    with pytest.raises(ValueError):   # group col not a dict column
        session.create_view("v", "movies", "rating", "rating", "emb", X, eps=0.3)
    with pytest.raises(ValueError):   # val col not scalar
        session.create_view("v", "movies", "genre", "genre", "emb", X, eps=0.3)
    with pytest.raises(ValueError):   # key col missing
        session.create_view("v", "movies", "genre", "rating", "noemb", X, eps=0.3)
    with pytest.raises(ValueError):   # query vector dim mismatch
        session.create_view("v", "movies", "genre", "rating", "emb",
                            np.array([1.0, 0.0, 0.0]), eps=0.3)
    with pytest.raises(ValueError):   # eps = 0: no incremental form
        session.create_view("v", "movies", "genre", "rating", "emb", X, eps=0.0)
    with pytest.raises(ValueError):   # eps invalid
        session.create_view("v", "movies", "genre", "rating", "emb", X, eps=-1.0)
    with pytest.raises(ValueError):   # eps NaN
        session.create_view("v", "movies", "genre", "rating", "emb", X,
                            eps=float("nan"))


def test_create_view_f32_keys_now_supported(session):
    # FLIPPED 2026-08-03 (f32-tail): maintained views serve KeyF32 too
    # (f32 scoring, f64 state — bruce-query/src/views.rs; end-to-end
    # coverage in test_f32_views.py). Formerly pinned as a KeyF64-only
    # typed error.
    session.attach_key("movies", "emb32", EMB.astype(np.float32))
    session.create_view("v", "movies", "genre", "rating", "emb32", X, eps=0.3)


def test_create_view_duplicate_name(session):
    session.create_view("v", "movies", "genre", "rating", "emb", X, eps=0.3)
    with pytest.raises(ValueError, match="exists"):
        session.create_view("v", "movies", "genre", "rating", "emb", X, eps=0.7)


# -------------------------------------------------------------- run

def test_run_bad_sql(session):
    for sql in [
        "((( not sql",
        "",
        "SELECT 1; SELECT 2",
        "INSERT INTO movies VALUES (1)",
        "SELECT genre FROM movies GROUP BY genre",                    # no SOFTAVG
        "SELECT genre, SOFTAVG(rating) FROM movies GROUP BY genre",   # arity
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) FROM movies",  # no GROUP BY
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), -0.5) "
        "FROM movies GROUP BY genre",                                 # negative eps
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3) "
        "FROM movies WHERE year < 2000 GROUP BY genre",               # unsupported op
    ]:
        with pytest.raises(ValueError):
            session.run(sql, {"q": X})


def test_run_missing_table_and_columns(session):
    with pytest.raises(ValueError):
        session.run("SELECT g, SOFTAVG(v, SIM(k, :q), 0.3) FROM nosuch GROUP BY g",
                    {"q": X})
    with pytest.raises(ValueError):
        session.run("SELECT genre, SOFTAVG(rating, SIM(noemb, :q), 0.3) "
                    "FROM movies GROUP BY genre", {"q": X})
    with pytest.raises(ValueError):
        session.run("SELECT rating, SOFTAVG(rating, SIM(emb, :q), 0.3) "
                    "FROM movies GROUP BY rating", {"q": X})


def test_run_unbound_param(session):
    with pytest.raises(ValueError):
        session.run(Q, {})


def test_run_param_dim_mismatch(session):
    bad = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        session.run(Q, {"q": bad})
    # With a declared error budget the planner consults the key sketch;
    # this path used to panic across the FFI (PanicException) before
    # the Database::run dimension guard.
    with pytest.raises(ValueError):
        session.run("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.3, 0.01) "
                    "FROM movies GROUP BY genre", {"q": bad})


def test_run_param_wrong_type(session):
    # PyO3 argument extraction rejects non-array params before our
    # code runs; that is a TypeError by Python convention.
    with pytest.raises((TypeError, ValueError)):
        session.run(Q, {"q": "not an array"})


# ------------------------------------------------------- insert_row

def test_insert_row_errors(session):
    ok_scalars = {"rating": 5.0, "year": 2021.0, "id": 9.0}
    ok_labels = {"genre": "A"}
    ok_keys = {"emb": np.array([0.5, 0.5])}
    with pytest.raises(ValueError):   # missing table
        session.insert_row("nosuch", ok_scalars, ok_labels, ok_keys)
    with pytest.raises(ValueError):   # missing scalar column
        session.insert_row("movies", {"rating": 5.0}, ok_labels, ok_keys)
    with pytest.raises(ValueError):   # missing label column
        session.insert_row("movies", ok_scalars, {}, ok_keys)
    with pytest.raises(ValueError):   # missing key column
        session.insert_row("movies", ok_scalars, ok_labels, {})
    with pytest.raises(ValueError):   # wrong key dimension
        session.insert_row("movies", ok_scalars, ok_labels,
                           {"emb": np.array([0.5, 0.5, 0.5])})
    with pytest.raises(ValueError):   # unknown column named (typo guard)
        session.insert_row("movies", dict(ok_scalars, typo_col=1.0),
                           ok_labels, ok_keys)


# ----------------------------------------------------- delete_where

def test_delete_where_errors(session):
    with pytest.raises(ValueError):   # missing table
        session.delete_where("nosuch", "id", "=", 1.0)
    with pytest.raises(ValueError):   # missing column
        session.delete_where("movies", "nocol", "=", 1.0)
    with pytest.raises(ValueError):   # dict column is not filterable
        session.delete_where("movies", "genre", "=", 0.0)
    with pytest.raises(ValueError):   # unsupported operator
        session.delete_where("movies", "id", "<", 1.0)


# ---------------------------------------------- session lifecycle

def test_session_lifecycle_memory_stable(tmp_path):
    """200 queries + 50 writes in one session; RSS growth < 50 MB.

    A generous bound: it will not flag allocator noise, but it does
    catch wholesale leaks (a fresh table or view copy per call).
    ru_maxrss on Linux is the peak RSS in KB.
    """
    import pandas as pd
    n = 2000
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "genre": [f"g{i % 8}" for i in range(n)],
        "rating": rng.uniform(0, 10, n),
        "year": rng.uniform(1990, 2025, n),
        "id": np.arange(n, dtype=np.float64),
    })
    pq = tmp_path / "life.parquet"
    df.to_parquet(pq)

    s = bruce.QuerySession()
    s.register_parquet("movies", str(pq))
    emb = rng.standard_normal((n, 16))
    s.attach_key("movies", "emb", emb)
    s.create_view("v", "movies", "genre", "rating", "emb",
                  np.ascontiguousarray(emb[0]), eps=0.5)

    q = np.ascontiguousarray(emb[0])
    sql = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.5) "
           "FROM movies GROUP BY genre")

    # warm-up establishes the peak baseline (first-call allocations)
    for _ in range(5):
        s.run(sql, {"q": q})
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    next_id = float(n)
    for i in range(200):
        labels, values, _ = s.run(sql, {"q": q})
        assert labels and all(np.isfinite(v) for v in values)
        if i % 4 == 0 and i // 4 < 50:
            if i % 8 == 0:
                s.insert_row(
                    "movies",
                    {"rating": 5.0, "year": 2026.0, "id": next_id},
                    {"genre": "g0"},
                    {"emb": np.ascontiguousarray(rng.standard_normal(16))},
                )
                next_id += 1.0
            else:
                s.delete_where("movies", "id", "=", next_id - 1.0)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (after_kb - before_kb) / 1024.0
    assert growth_mb < 50.0, f"RSS grew {growth_mb:.1f} MB over 250 ops"

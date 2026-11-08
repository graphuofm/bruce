"""pytest suite for the softavg DuckDB extension.

Run:  python -m pytest -q tests/
Needs build/softavg.duckdb_extension (./build.sh) and a duckdb python package whose
version matches the DuckDB the extension was built against (1.5.6).
Copied from paper_softavg/experiments/duckdb_ext (18 tests); the update-path
tests at the end were added with the vectorised update.

Tests 1-5 are the required ones; the rest pin the documented semantics
(eps regimes, +/-Inf policy, NaN, eps validation).
"""
import math
import os

import duckdb
import numpy as np
import pyarrow as pa
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.abspath(os.path.join(HERE, "..", "build", "softavg.duckdb_extension"))
INF = float("inf")


def connect(threads=None):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXT}'")
    if threads is not None:
        con.execute(f"SET threads={int(threads)}")
    return con


def eps_sql(eps):
    return "'infinity'::DOUBLE" if math.isinf(eps) else repr(float(eps))


def ref_softavg(s, v, eps):
    """float128 (numpy longdouble) max-shifted reference over one group."""
    s = np.asarray(s, dtype=np.longdouble)
    v = np.asarray(v, dtype=np.longdouble)
    if s.size == 0:
        return None
    if math.isinf(eps):
        return float(v.sum() / np.longdouble(v.size))
    m = s.max()
    if eps == 0:
        sel = s == m
        return float(v[sel].sum() / np.longdouble(sel.sum()))
    w = np.exp((s - m) / np.longdouble(eps))
    return float((w * v).sum() / w.sum())


def ref_by_group(g, s, v, eps):
    out = {}
    order = np.argsort(g, kind="stable")
    gs = g[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for idx in np.split(order, bounds):
        out[int(g[idx[0]])] = ref_softavg(s[idx], v[idx], eps)
    return out


def register(con, name, **cols):
    con.register(name, pa.table(cols))


def rel_err(x, r):
    if r is None or x is None:
        return 0.0 if (r is None and x is None) else INF
    if not math.isfinite(x):
        return INF
    return abs(x - r) / max(abs(r), 1e-300)


# --------------------------------------------------------------------------- 1
@pytest.mark.parametrize("eps", [0.0, 0.5, 1.0, INF])
def test_1_equal_scores_equals_avg(eps):
    rng = np.random.default_rng(1)
    n, ng = 100_000, 10
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 100, n)
    s = np.full(n, 0.3)
    con = connect()
    register(con, "t", g=g, v=v, s=s)
    rows = con.execute(
        f"SELECT g, softavg(v, s, {eps_sql(eps)}), avg(v) FROM t GROUP BY g ORDER BY g"
    ).fetchall()
    assert len(rows) == ng
    for _, sa, av in rows:
        assert rel_err(sa, av) <= 1e-12, (eps, sa, av)


# --------------------------------------------------------------------------- 2
@pytest.mark.parametrize("eps", [1.0, 0.1, 0.01, 0.001, 1e-4])
def test_2_random_matches_float128_reference(eps):
    rng = np.random.default_rng(2)
    n, ng = 100_000, 50
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 100, n)
    s = rng.normal(0.0, 1.0, n)
    con = connect()
    register(con, "t", g=g, v=v, s=s)
    got = dict(con.execute(f"SELECT g, softavg(v, s, {eps_sql(eps)}) FROM t GROUP BY g").fetchall())
    ref = ref_by_group(g, s, v, eps)
    assert got.keys() == ref.keys()
    worst = max(rel_err(got[k], ref[k]) for k in ref)
    assert worst <= 1e-9, (eps, worst)


# --------------------------------------------------------------------------- 3
@pytest.mark.parametrize("eps", [0.01, 1e-4])
def test_3_parallel_combine_matches_single_thread(eps):
    rng = np.random.default_rng(3)
    n, ng = 1_000_000, 20
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 100, n)
    s = rng.normal(0.0, 1.0, n)
    tbl = pa.table({"g": g, "v": v, "s": s})
    res = {}
    for threads in (1, 32):
        con = connect(threads)
        con.register("t", tbl)
        # materialise so the aggregate reads a real (parallel-scannable) table
        con.execute("CREATE TABLE tt AS SELECT * FROM t")
        res[threads] = dict(
            con.execute(f"SELECT g, softavg(v, s, {eps_sql(eps)}) FROM tt GROUP BY g").fetchall()
        )
    assert res[1].keys() == res[32].keys() and len(res[1]) == ng
    worst = max(rel_err(res[32][k], res[1][k]) for k in res[1])
    assert worst <= 1e-12, (eps, worst)
    ref = ref_by_group(g, s, v, eps)
    assert max(rel_err(res[32][k], ref[k]) for k in ref) <= 1e-9


# --------------------------------------------------------------------------- 4
def test_4_sharp_eps_naive_breaks_softavg_does_not(capsys):
    rng = np.random.default_rng(4)
    n, ng, eps = 100_000, 5, 1e-4
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 100, n)
    s = rng.uniform(-1, 1, n)
    con = connect()
    register(con, "t", g=g, v=v, s=s)
    naive = (
        f"SELECT g, SUM(EXP(s/{eps_sql(eps)})*v)/SUM(EXP(s/{eps_sql(eps)})) FROM t GROUP BY g ORDER BY g"
    )
    try:
        naive_rows = con.execute(naive).fetchall()
        naive_outcome = f"returned {naive_rows}"
        naive_ok = all(r is not None and math.isfinite(r) for _, r in naive_rows)
    except Exception as e:  # noqa: BLE001 - we want the verbatim DuckDB error
        naive_outcome = f"raised {type(e).__name__}: {e}"
        naive_ok = False
    print("NAIVE SQL at eps=1e-4:", naive_outcome)
    assert not naive_ok, "naive SQL unexpectedly produced finite values"
    got = dict(con.execute(f"SELECT g, softavg(v, s, {eps_sql(eps)}) FROM t GROUP BY g").fetchall())
    ref = ref_by_group(g, s, v, eps)
    assert all(math.isfinite(x) for x in got.values())
    assert max(rel_err(got[k], ref[k]) for k in ref) <= 1e-9


# --------------------------------------------------------------------------- 5
@pytest.mark.parametrize("eps", [0.5, INF])
def test_5_null_handling_matches_avg(eps):
    rng = np.random.default_rng(5)
    n, ng = 50_000, 8
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 100, n).astype(object)
    s = np.full(n, 0.3).astype(object)
    v[rng.random(n) < 0.2] = None  # NULL values
    s[rng.random(n) < 0.2] = None  # NULL scores (must also drop the row)
    v[g == 7] = None  # one group entirely NULL -> NULL result
    con = connect()
    con.register("t", pa.table({"g": g, "v": pa.array(v, pa.float64()), "s": pa.array(s, pa.float64())}))
    rows = con.execute(
        f"""SELECT g, softavg(v, s, {eps_sql(eps)}),
                   avg(CASE WHEN s IS NULL THEN NULL ELSE v END),
                   count(v), count(CASE WHEN s IS NULL THEN NULL ELSE v END)
            FROM t GROUP BY g ORDER BY g"""
    ).fetchall()
    assert len(rows) == ng
    for gk, sa, av, _, cnt in rows:
        if cnt == 0:
            assert sa is None and av is None, gk
        else:
            assert rel_err(sa, av) <= 1e-12, (gk, sa, av)
    # empty input -> NULL, like AVG; NULL eps on a NULL row is ignored like the row
    assert con.execute("SELECT softavg(v, s, 0.5), avg(v) FROM t WHERE false").fetchall() == [(None, None)]
    assert con.execute(
        "SELECT softavg(v, s, e) FROM (VALUES (NULL::DOUBLE, 0.0, NULL::DOUBLE), (2.0, 0.0, 0.5)) t(v, s, e)"
    ).fetchall() == [(2.0,)]


# ------------------------------------------------------------------ semantics
def test_eps_validation_errors():
    con = connect()
    with pytest.raises(duckdb.InvalidInputException, match="eps must be >= 0"):
        con.execute("SELECT softavg(v, s, -1.0) FROM (VALUES (1.0, 0.0)) t(v, s)").fetchall()
    with pytest.raises(duckdb.InvalidInputException, match="eps must be constant"):
        con.execute(
            "SELECT softavg(v, s, e) FROM (VALUES (1.0, 0.0, 0.5), (2.0, 0.0, 0.25)) t(v, s, e)"
        ).fetchall()
    with pytest.raises(duckdb.InvalidInputException, match="eps must not be NULL"):
        con.execute("SELECT softavg(v, s, NULL::DOUBLE) FROM (VALUES (1.0, 0.0)) t(v, s)").fetchall()
    with pytest.raises(duckdb.InvalidInputException, match="eps must be >= 0"):
        con.execute("SELECT softavg(v, s, 'nan'::DOUBLE) FROM (VALUES (1.0, 0.0)) t(v, s)").fetchall()


def test_eps_zero_is_argmax_mean_with_exact_ties():
    con = connect()
    r = con.execute(
        "SELECT softavg(v, s, 0.0) FROM (VALUES (1.0, 0.5), (10.0, 0.9), (20.0, 0.9), (100.0, 0.1)) t(v, s)"
    ).fetchall()
    assert r == [(15.0,)]


def test_infinite_scores_follow_reference_policy():
    con = connect()
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    # -inf weighs 0; an all--inf group is NULL
    assert q("SELECT softavg(v, s, 0.5) FROM (VALUES (1.0, '-infinity'::DOUBLE)) t(v, s)") == [(None,)]
    assert q("SELECT softavg(v, s, 0.5) FROM (VALUES (1.0, '-infinity'::DOUBLE), (3.0, 0.0)) t(v, s)") == [(3.0,)]
    # +inf dominates finite rows, ties among +inf rows are uniform
    assert q(
        "SELECT softavg(v, s, 0.5) FROM (VALUES (1.0, 0.0), (10.0, 'infinity'::DOUBLE), (2.0, 0.5), (20.0, 'infinity'::DOUBLE)) t(v, s)"
    ) == [(15.0,)]
    # at eps = inf the score is not consulted
    assert q(
        "SELECT softavg(v, s, 'infinity'::DOUBLE) FROM (VALUES (1.0, '-infinity'::DOUBLE), (3.0, 'infinity'::DOUBLE)) t(v, s)"
    ) == [(2.0,)]


def test_nan_input_propagates_like_avg():
    con = connect()
    sa, av = con.execute(
        "SELECT softavg(v, s, 0.5), avg(v) FROM (VALUES (1.0, 0.0), ('nan'::DOUBLE, 0.0)) t(v, s)"
    ).fetchall()[0]
    assert math.isnan(sa) and math.isnan(av)


# ------------------------------------------------------- vectorised update paths
def test_per_group_eps_takes_general_path():
    """eps varies inside one vector but is constant per group: legal."""
    rng = np.random.default_rng(6)
    n, ng = 20_000, 4
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 10, n)
    s = rng.uniform(-1, 1, n)
    eps_of = {0: 0.01, 1: 0.1, 2: 0.0, 3: INF}
    con = connect()
    register(con, "t", g=g, v=v, s=s)
    got = dict(con.execute(
        "SELECT g, softavg(v, s, CASE g WHEN 0 THEN 0.01 WHEN 1 THEN 0.1 WHEN 2 THEN 0.0 "
        "ELSE 'infinity'::DOUBLE END) FROM t GROUP BY g").fetchall())
    for k, e in eps_of.items():
        m = g == k
        assert rel_err(got[k], ref_softavg(s[m], v[m], e)) <= 1e-12, (k, got[k])


def test_masked_uniform_path_all_null_vector_ignores_bad_eps():
    """A vector whose rows are all NULL never validates eps (like AVG ignores them)."""
    con = connect()
    con.execute("CREATE TABLE t AS SELECT i::DOUBLE AS v, NULL::DOUBLE AS s FROM range(5000) r(i)")
    assert con.execute("SELECT softavg(v, s, -1.0) FROM t").fetchall() == [(None,)]
    con.execute("INSERT INTO t VALUES (1.0, 0.0)")
    with pytest.raises(duckdb.InvalidInputException, match="eps must be >= 0"):
        con.execute("SELECT softavg(v, s, -1.0) FROM t").fetchall()


def test_masked_uniform_path_matches_reference():
    rng = np.random.default_rng(7)
    n, ng = 50_000, 6
    g = rng.integers(0, ng, n)
    v = rng.uniform(0, 100, n)
    s = rng.normal(0, 0.3, n)
    vo, so = v.astype(object), s.astype(object)
    drop = rng.random(n) < 0.3
    so[drop] = None
    con = connect()
    con.register("t", pa.table({"g": g, "v": pa.array(vo, pa.float64()), "s": pa.array(so, pa.float64())}))
    for eps in (1e-3, 0.1, 0.0, INF):
        got = dict(con.execute(f"SELECT g, softavg(v, s, {eps_sql(eps)}) FROM t GROUP BY g").fetchall())
        ref = ref_by_group(g[~drop], s[~drop], v[~drop], eps)
        assert max(rel_err(got[k], ref[k]) for k in ref) <= 1e-12, eps

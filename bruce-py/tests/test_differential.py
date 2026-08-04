"""Workstream 7 — differential oracle harness for the SQL pipeline.

100 seeded random datasets + queries run through bruce.QuerySession and
compared against
  (a) a numpy anchored-softmax oracle, and
  (b) DuckDB executing the equivalent max-anchored SQL
      (WITH s AS (...), m AS (per-group MAX) ... SUM(EXP((sim-mx)/eps)*v)
       / SUM(EXP((sim-mx)/eps))).

Tolerances (task contract): rtol 1e-9 for f64 key columns, 1e-4 for
f32 key columns.  f32 scoring is storage-precision by design
(bruce-core mask.rs precision contract) and numpy's float32 matmul
rounds in a different summation order than the kernel's 4-way unrolled
dot, so bit equality is undefined — only the contract tolerance is.
Comparisons use |a-b| <= rtol * max(1, |a|, |b|): the unit floor keeps
near-zero group averages from degenerating the relative test.

Edge datasets: single row, single group, empty-after-filter, and a
filter that empties one group only.  bruce returns NO row for an
uncovered group; the DuckDB anchored query must agree on the covered
set (GROUP BY over zero surviving rows emits no row either).
The eps endpoints are pinned against their SQL equivalents: eps = 0 is
the argmax-mean (ties averaged, AVG(v) FILTER (WHERE sim = mx)) and
eps = INF is plain AVG(v) GROUP BY.
"""
import numpy as np
import pandas as pd
import duckdb
import pytest

import bruce

RTOL_F64 = 1e-9
RTOL_F32 = 1e-4

# f32 pools keep eps >= 0.1: at sharper temperatures the oracle's own
# f32 rounding (different summation order) approaches the 1e-4 contract
# and the comparison would test numpy, not bruce.
EPS_POOL_F64 = [0.05, 0.1, 0.3, 1.0, 3.0]
EPS_POOL_F32 = [0.1, 0.3, 1.0, 3.0]


# ------------------------------------------------------------------ oracles

def np_softavg(g, v, sims, eps, mask=None):
    """Anchored softmax average per group -> {label: value}."""
    g = np.asarray(g)
    v = np.asarray(v, dtype=np.float64)
    sims = np.asarray(sims, dtype=np.float64)
    if mask is None:
        mask = np.ones(len(g), dtype=bool)
    out = {}
    for label in pd.unique(g[mask]):
        sel = mask & (g == label)
        s, vals = sims[sel], v[sel]
        w = np.exp((s - s.max()) / eps)
        out[str(label)] = float(w @ vals / w.sum())
    return out


def np_argmax_avg(g, v, sims, mask=None):
    """eps = 0 oracle: mean of v over the per-group argmax set (ties)."""
    g = np.asarray(g)
    v = np.asarray(v, dtype=np.float64)
    sims = np.asarray(sims, dtype=np.float64)
    if mask is None:
        mask = np.ones(len(g), dtype=bool)
    out = {}
    for label in pd.unique(g[mask]):
        sel = mask & (g == label)
        s, vals = sims[sel], v[sel]
        out[str(label)] = float(vals[s == s.max()].mean())
    return out


def duck_softavg(df_scored, eps, where_sql=""):
    """DuckDB executing the max-anchored equivalent SQL."""
    con = duckdb.connect()
    try:
        con.register("t", df_scored)
        rows = con.execute(
            f"""
            WITH s AS (SELECT g, v, sim FROM t{where_sql}),
                 m AS (SELECT g, MAX(sim) AS mx FROM s GROUP BY g)
            SELECT s.g,
                   SUM(EXP((s.sim - m.mx) / {eps}) * s.v)
                   / SUM(EXP((s.sim - m.mx) / {eps}))
            FROM s JOIN m ON s.g = m.g
            GROUP BY s.g
            """
        ).fetchall()
    finally:
        con.close()
    return {str(g): float(val) for g, val in rows}


def agree(got, want, rtol, ctx):
    assert set(got) == set(want), (
        f"{ctx}: covered-group mismatch: {sorted(got)} vs {sorted(want)}"
    )
    for label in got:
        a, b = got[label], want[label]
        tol = rtol * max(1.0, abs(a), abs(b))
        assert abs(a - b) <= tol, (
            f"{ctx}: group {label}: {a!r} vs {b!r} (tol {tol:g})"
        )


# ------------------------------------------------------------------ harness

def make_dataset(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    n_groups = int(rng.integers(1, 8))
    d = int(rng.integers(2, 12))
    use_f32 = seed % 2 == 1  # guaranteed coverage of both key dtypes

    g = np.array([f"g{j}" for j in rng.integers(0, n_groups, n)])
    v = rng.uniform(-10.0, 10.0, n)
    # y quantised to multiples of 10 in [0, 90]: '=' predicates can
    # match rows and '>=' boundaries are exact in SQL text
    y = rng.integers(0, 10, n).astype(np.float64) * 10.0
    keys = rng.uniform(-1.0, 1.0, (n, d)) / np.sqrt(d)
    if use_f32:
        keys = keys.astype(np.float32)
    q = rng.uniform(-1.0, 1.0, d) / np.sqrt(d)

    eps = (EPS_POOL_F32 if use_f32 else EPS_POOL_F64)[
        int(rng.integers(0, len(EPS_POOL_F32 if use_f32 else EPS_POOL_F64)))
    ]
    kind = int(rng.integers(0, 3))
    if kind == 0:
        where_sql = ""
    elif kind == 1:
        where_sql = f" WHERE y >= {float(rng.integers(0, 13) * 10)}"
    else:
        where_sql = f" WHERE y = {float(rng.integers(0, 13) * 10)}"
    return dict(g=g, v=v, y=y, keys=keys, q=q, eps=eps,
                where_sql=where_sql, use_f32=use_f32)


def build_session(tmp_path, g, v, y, keys):
    df = pd.DataFrame({"g": g, "v": v, "y": y})
    pq = tmp_path / "t.parquet"
    df.to_parquet(pq)
    s = bruce.QuerySession()
    s.register_parquet("t", str(pq))
    s.attach_key("t", "k", keys)
    return s


def sims_of(keys, q):
    """Score exactly as the engine's storage contract: f32 keys score
    in f32 (query cast down once), then widen to f64."""
    if keys.dtype == np.float32:
        return (keys @ q.astype(np.float32)).astype(np.float64)
    return keys @ q


def filter_mask(y, where_sql):
    if not where_sql:
        return np.ones(len(y), dtype=bool)
    if ">=" in where_sql:
        return y >= float(where_sql.split(">=")[1])
    return y == float(where_sql.split("=")[1])


def run_bruce(session, sql, q):
    labels, values, explain = session.run(sql, {"q": np.asarray(q, dtype=np.float64)})
    return dict(zip(labels, values)), explain


# ------------------------------------------------------------- 100 randoms

@pytest.mark.parametrize("seed", range(100))
def test_random_differential(seed, tmp_path):
    ds = make_dataset(seed)
    session = build_session(tmp_path, ds["g"], ds["v"], ds["y"], ds["keys"])
    sql = (f"SELECT g, SOFTAVG(v, SIM(k, :q), {ds['eps']}) "
           f"FROM t{ds['where_sql']} GROUP BY g")
    got, explain = run_bruce(session, sql, ds["q"])
    assert "FusedGroupScan" in explain  # sanity: exact fused path ran

    sims = sims_of(ds["keys"], ds["q"])
    mask = filter_mask(ds["y"], ds["where_sql"])
    rtol = RTOL_F32 if ds["use_f32"] else RTOL_F64
    ctx = f"seed {seed} ({'f32' if ds['use_f32'] else 'f64'} keys): {sql}"

    oracle = np_softavg(ds["g"], ds["v"], sims, ds["eps"], mask)
    agree(got, oracle, rtol, f"{ctx} [numpy]")

    df_scored = pd.DataFrame(
        {"g": ds["g"], "v": ds["v"], "y": ds["y"], "sim": sims})
    duck = duck_softavg(df_scored, ds["eps"], ds["where_sql"])
    agree(got, duck, rtol, f"{ctx} [duckdb]")


# ------------------------------------------------------------- edge cases

def test_single_row(tmp_path):
    g = np.array(["only"])
    v = np.array([4.25])
    y = np.array([10.0])
    keys = np.array([[0.3, -0.7]])
    q = np.array([1.0, 0.5])
    session = build_session(tmp_path, g, v, y, keys)
    for eps in (0.05, 1.0):
        got, _ = run_bruce(
            session, f"SELECT g, SOFTAVG(v, SIM(k, :q), {eps}) FROM t GROUP BY g", q)
        agree(got, {"only": 4.25}, RTOL_F64, f"single row eps={eps}")
        duck = duck_softavg(pd.DataFrame(
            {"g": g, "v": v, "y": y, "sim": sims_of(keys, q)}), eps)
        agree(got, duck, RTOL_F64, f"single row eps={eps} [duckdb]")


def test_single_group(tmp_path):
    rng = np.random.default_rng(4242)
    n, d, eps = 60, 4, 0.2
    g = np.array(["g0"] * n)
    v = rng.uniform(-5, 5, n)
    y = rng.integers(0, 10, n).astype(np.float64) * 10.0
    keys = rng.uniform(-1, 1, (n, d)) / 2.0
    q = rng.uniform(-1, 1, d) / 2.0
    session = build_session(tmp_path, g, v, y, keys)
    got, _ = run_bruce(
        session, f"SELECT g, SOFTAVG(v, SIM(k, :q), {eps}) FROM t GROUP BY g", q)
    sims = sims_of(keys, q)
    agree(got, np_softavg(g, v, sims, eps), RTOL_F64, "single group [numpy]")
    agree(got, duck_softavg(pd.DataFrame({"g": g, "v": v, "y": y, "sim": sims}), eps),
          RTOL_F64, "single group [duckdb]")


def test_empty_after_filter(tmp_path):
    """Predicate admits no row: bruce covers no group and returns zero
    rows; the DuckDB anchored query returns zero rows too."""
    rng = np.random.default_rng(7)
    n, d = 40, 3
    g = np.array([f"g{j}" for j in rng.integers(0, 3, n)])
    v = rng.uniform(-5, 5, n)
    y = rng.integers(0, 10, n).astype(np.float64) * 10.0
    keys = rng.uniform(-1, 1, (n, d))
    q = rng.uniform(-1, 1, d)
    session = build_session(tmp_path, g, v, y, keys)
    where = " WHERE y >= 1000"
    got, _ = run_bruce(
        session, f"SELECT g, SOFTAVG(v, SIM(k, :q), 0.3) FROM t{where} GROUP BY g", q)
    assert got == {}
    duck = duck_softavg(
        pd.DataFrame({"g": g, "v": v, "y": y, "sim": sims_of(keys, q)}), 0.3, where)
    assert duck == {}


def test_filter_empties_one_group_only(tmp_path):
    """Uncovered-group semantics on the covered set: one group loses
    every row to the filter; bruce, numpy, and DuckDB must all report
    exactly the surviving groups."""
    g = np.array(["a", "a", "b", "b", "c"])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([50.0, 60.0, 10.0, 20.0, 70.0])  # b dies at y >= 50
    keys = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])
    q = np.array([1.0, 0.0])
    eps = 0.2
    session = build_session(tmp_path, g, v, y, keys)
    where = " WHERE y >= 50"
    got, _ = run_bruce(
        session, f"SELECT g, SOFTAVG(v, SIM(k, :q), {eps}) FROM t{where} GROUP BY g", q)
    assert set(got) == {"a", "c"}
    sims = sims_of(keys, q)
    mask = filter_mask(y, where)
    agree(got, np_softavg(g, v, sims, eps, mask), RTOL_F64, "one group dies [numpy]")
    agree(got, duck_softavg(pd.DataFrame({"g": g, "v": v, "y": y, "sim": sims}),
                            eps, where),
          RTOL_F64, "one group dies [duckdb]")


def test_eps_zero_is_sql_argmax_mean_with_ties(tmp_path):
    """The tropical endpoint: eps = 0 equals AVG(v) over the per-group
    argmax rows, ties included — pinned against DuckDB's
    AVG(v) FILTER (WHERE sim = mx).  Exact ties are constructed from
    duplicated key rows (identical f64 dots on both sides)."""
    rng = np.random.default_rng(99)
    base = rng.uniform(-1, 1, (3, 4))
    idx = rng.integers(0, 3, 30)
    keys = base[idx]  # duplicates -> exact sim ties within groups
    n = len(idx)
    g = np.array([f"g{j}" for j in rng.integers(0, 2, n)])
    v = rng.uniform(-5, 5, n)
    y = np.zeros(n)
    q = rng.uniform(-1, 1, 4)
    session = build_session(tmp_path, g, v, y, keys)
    got, _ = run_bruce(
        session, "SELECT g, SOFTAVG(v, SIM(k, :q), 0.0) FROM t GROUP BY g", q)
    sims = sims_of(keys, q)
    agree(got, np_argmax_avg(g, v, sims), RTOL_F64, "eps=0 [numpy argmax-mean]")

    con = duckdb.connect()
    try:
        con.register("t", pd.DataFrame({"g": g, "v": v, "sim": sims}))
        rows = con.execute(
            """
            WITH m AS (SELECT g, MAX(sim) AS mx FROM t GROUP BY g)
            SELECT t.g, AVG(t.v) FILTER (WHERE t.sim = m.mx)
            FROM t JOIN m ON t.g = m.g
            GROUP BY t.g
            """
        ).fetchall()
    finally:
        con.close()
    agree(got, {str(gl): float(val) for gl, val in rows},
          RTOL_F64, "eps=0 [duckdb argmax-mean]")


def test_eps_inf_is_plain_group_avg(tmp_path):
    """The uniform endpoint: eps = INF equals AVG(v) GROUP BY g."""
    rng = np.random.default_rng(123)
    n, d = 80, 3
    g = np.array([f"g{j}" for j in rng.integers(0, 4, n)])
    v = rng.uniform(-5, 5, n)
    y = rng.integers(0, 10, n).astype(np.float64) * 10.0
    keys = rng.uniform(-1, 1, (n, d))
    q = rng.uniform(-1, 1, d)
    session = build_session(tmp_path, g, v, y, keys)
    got, explain = run_bruce(
        session, "SELECT g, SOFTAVG(v, SIM(k, :q), INF) FROM t GROUP BY g", q)
    assert "ExactGroupAvg" in explain  # R3 fired

    con = duckdb.connect()
    try:
        con.register("t", pd.DataFrame({"g": g, "v": v}))
        rows = con.execute("SELECT g, AVG(v) FROM t GROUP BY g").fetchall()
    finally:
        con.close()
    agree(got, {str(gl): float(val) for gl, val in rows}, RTOL_F64, "eps=INF [duckdb AVG]")
    agree(got, {str(gl): float(v[g == gl].mean()) for gl in pd.unique(g)},
          RTOL_F64, "eps=INF [numpy mean]")

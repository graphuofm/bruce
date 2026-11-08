"""Conformance: engines/core/conformance/vectors.json executed inside DuckDB.

PROTOCOL.md: load the table, run GROUP BY at every eps (rtol 1e-12, `null`
must be SQL NULL) and the window cases (ROWS BETWEEN 49 PRECEDING AND
CURRENT ROW over ids < 2000; absolute error <= 1e-9 * max|v| over those rows,
the PROTOCOL's window tolerance since 2026-11-05).

DuckDB has no window callback in its C API, so `softavg(...) OVER (...)` runs
through DuckDB's own window machinery on top of our `combine`:
WindowSegmentTree for ROWS frames (default mode), and WindowNaiveAggregator
when `debug_window_mode = 'separate'` (update + finalize per frame). Both are
checked. Run: python -m pytest -q tests/test_conformance.py
"""
import json
import math
import os

import duckdb
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.abspath(os.path.join(HERE, "..", "build", "softavg.duckdb_extension"))
VECTORS = os.path.abspath(os.path.join(HERE, "..", "..", "core", "conformance", "vectors.json"))

with open(VECTORS) as f:
    V = json.load(f)


def num(x):
    """vectors.json encodes +/-inf as strings, NaN never appears (null = SQL NULL)."""
    if x is None:
        return None
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return float(x)


def eps_sql(e):
    e = num(e)
    return "'infinity'::DOUBLE" if math.isinf(e) else repr(e)


def close(got, want, rtol):
    if want is None:
        return got is None
    if got is None or not math.isfinite(got):
        return False
    return abs(got - want) <= rtol * max(abs(got), abs(want), 1e-300)


def connect(threads):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXT}'")
    con.execute(f"SET threads={threads}")
    con.execute("CREATE TABLE vt(id INTEGER, g INTEGER, s DOUBLE, v DOUBLE)")
    con.executemany("INSERT INTO vt VALUES (?, ?, ?, ?)",
                    [(int(i), int(g), num(s), num(v)) for i, g, s, v in V["table"]])
    return con


def test_table_loaded_exactly():
    con = connect(1)
    assert con.execute("SELECT count(*) FROM vt").fetchone()[0] == len(V["table"])
    inf = con.execute("SELECT count(*) FILTER (WHERE s = 'infinity'::DOUBLE), "
                      "count(*) FILTER (WHERE s = '-infinity'::DOUBLE) FROM vt").fetchone()
    want = (sum(1 for r in V["table"] if r[2] == "inf"), sum(1 for r in V["table"] if r[2] == "-inf"))
    assert inf == want


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("case", V["group_by"], ids=lambda c: f"eps={c['eps']}")
def test_group_by(case, threads):
    con = connect(threads)
    rows = con.execute(f"SELECT g, softavg(v, s, {eps_sql(case['eps'])}) FROM vt GROUP BY g").fetchall()
    got = {str(g): x for g, x in rows}
    assert set(got) == set(case["expected"])
    bad = {g: (got[g], w) for g, w in case["expected"].items() if not close(got[g], num(w), 1e-12)}
    assert not bad, bad


@pytest.mark.parametrize("mode", ["default", "separate"])
@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("case", V["window"], ids=lambda c: f"eps={c['eps']}")
def test_window(case, threads, mode):
    con = connect(threads)
    if mode != "default":
        con.execute(f"SET debug_window_mode = '{mode}'")
    k = int(V["window_rows_preceding"])
    n = len(case["expected"])
    rows = con.execute(
        f"""SELECT id, softavg(v, s, {eps_sql(case['eps'])})
                   OVER (ORDER BY id ROWS BETWEEN {k} PRECEDING AND CURRENT ROW)
            FROM vt WHERE id < {n} ORDER BY id""").fetchall()
    assert [r[0] for r in rows] == list(range(n))
    vmax = max(abs(num(r[3])) for r in V["table"] if r[0] < n)
    tol = 1e-9 * vmax
    bad = [(i, rows[i][1], w) for i, w in enumerate(case["expected"])
           if rows[i][1] is None or not math.isfinite(rows[i][1]) or abs(rows[i][1] - num(w)) > tol]
    assert not bad, bad[:5]

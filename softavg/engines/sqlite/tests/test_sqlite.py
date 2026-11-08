"""SQLite-side tests for softagg-sqlite: load the release .so into Python's
sqlite3, replay engines/core/conformance/vectors.json (GROUP BY at every eps,
rtol 1e-12, null -> NULL; window ROWS 49 PRECEDING, rtol 1e-9), then NULL /
error / NaN / window-frame-shape / softagg_dot / sqlite-vec checks.

Run:  python3 tests/test_sqlite.py        (exit 0 = all pass; prints counts)
      SOFTAGG_SQLITE_SO=... VEC0_SO=...   to override the library paths.
numpy is used only as a reference (two-pass max-shift in float64).
"""
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENG = HERE.parent
SO = os.environ.get("SOFTAGG_SQLITE_SO", str(ENG / "target/release/libsoftagg_sqlite"))
VEC0 = os.environ.get(
    "VEC0_SO", str(ENG.parents[1] / "experiments/sqlite/deps/sqlite_vec-0.1.9/vec0"))
VECTORS = ENG.parent / "core/conformance/vectors.json"

CHECKS = {"pass": 0, "fail": 0}
FAILS = []


def check(ok, what):
    CHECKS["pass" if ok else "fail"] += 1
    if not ok:
        FAILS.append(what)


def num(x):
    if x == "inf":
        return math.inf
    if x == "-inf":
        return -math.inf
    return x


def close(got, want, rtol):
    if want is None:
        return got is None
    if got is None:
        return False
    return abs(got - want) <= rtol * max(abs(got), abs(want), 1e-300)


def connect(vec=False):
    c = sqlite3.connect(":memory:")
    c.enable_load_extension(True)
    c.load_extension(SO)
    if vec:
        c.load_extension(VEC0)
    c.enable_load_extension(False)
    return c


def ref_softavg(s, v, eps):
    """float64 two-pass reference (same regimes as softagg-core)."""
    s = np.asarray(s, float)
    v = np.asarray(v, float)
    if eps == math.inf:
        return float(v.mean()) if len(v) else None
    keep = s != -math.inf
    s, v = s[keep], v[keep]
    if len(s) == 0:
        return None
    m = s.max()
    if eps == 0 or m == math.inf:
        sel = s == m
        return float(v[sel].mean())
    w = np.exp((s - m) / eps)
    return float((w * v).sum() / w.sum())


# ------------------------------------------------------------ conformance
def test_conformance():
    d = json.load(open(VECTORS))
    c = connect()
    c.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, g INTEGER, s REAL, v REAL)")
    c.executemany("INSERT INTO t VALUES (?,?,?,?)",
                  [(r[0], r[1], num(r[2]), num(r[3])) for r in d["table"]])
    n_groups = 0
    for fn in ("softavg", "softavg_agg"):
        for case in d["group_by"]:
            eps = num(case["eps"])
            got = dict(c.execute(f"SELECT g, {fn}(v, s, ?) FROM t GROUP BY g", (eps,)))
            for g, want in case["expected"].items():
                n_groups += 1
                check(int(g) in got, f"{fn} group_by eps={eps} g={g} missing")
                check(close(got.get(int(g)), want, 1e-12),
                      f"{fn} group_by eps={eps} g={g}: got {got.get(int(g))!r} want {want!r}")
    p = d["window_rows_preceding"]
    n_win = 0
    # PROTOCOL: window tolerance is absolute, |err| <= 1e-9 * max|v|
    vmax = max(abs(num(r[3])) for r in d["table"][:len(d["window"][0]["expected"])])
    for case in d["window"]:
        eps = num(case["eps"])
        rows = c.execute(
            f"SELECT softavg(v, s, ?) OVER (ORDER BY id ROWS BETWEEN {p} PRECEDING AND CURRENT ROW) "
            "FROM t WHERE id < ? ORDER BY id", (eps, len(case["expected"]))).fetchall()
        check(len(rows) == len(case["expected"]), f"window eps={eps}: row count")
        for i, ((got,), want) in enumerate(zip(rows, case["expected"])):
            n_win += 1
            check(got is not None and abs(got - want) <= 1e-9 * vmax,
                  f"window eps={eps} row {i}: got {got!r} want {want!r}")
    return n_groups, n_win


# ------------------------------------------------------------ NULL / errors
def test_null_and_errors():
    c = connect()
    c.execute("CREATE TABLE t(g INTEGER, s REAL, v REAL)")
    c.executemany("INSERT INTO t VALUES (?,?,?)", [
        (1, 0.1, 1.0), (1, 0.3, 3.0), (1, None, 100.0), (1, 0.2, None),
        (2, None, 5.0), (2, 0.5, None),            # no live rows -> NULL
        (3, 0.1, 2.0), (3, 0.2, 4.0)])
    got = dict(c.execute("SELECT g, softavg(v, s, 9e999) FROM t GROUP BY g"))
    # NULL score drops the row even at eps=inf (pair semantics, as in PG/DuckDB)
    check(close(got[1], 2.0, 1e-15), f"NULL skip at inf: {got[1]}")
    check(got[2] is None, "all-NULL group -> NULL")
    check(close(got[3], 3.0, 1e-15), "eps=inf == AVG")
    got = dict(c.execute("SELECT g, softavg(v, s, 0.1) FROM t GROUP BY g"))
    check(close(got[1], ref_softavg([0.1, 0.3], [1.0, 3.0], 0.1), 1e-12), "NULL skip at 0.1")
    check(c.execute("SELECT softavg(v, s, 0.1) FROM t WHERE g = 99").fetchone()[0] is None,
          "empty input -> NULL")
    check(c.execute("SELECT softavg(v, s, 0) FROM t WHERE g = 3").fetchone()[0] == 4.0,
          "eps=0 is the argmax mean")
    # equal scores -> AVG at every eps
    for eps in (0.0, 1e-4, 0.5, 9e999):
        a, b = c.execute("SELECT softavg(v, 0.25, ?), AVG(v) FROM t WHERE v IS NOT NULL", (eps,)).fetchone()
        check(close(a, b, 1e-12), f"equal scores eps={eps}: {a} vs {b}")
    # NaN: SQLite has none (0.0/0.0 is NULL), so it is skipped like AVG skips it
    a, b = c.execute("SELECT softavg(CASE WHEN g=1 THEN 0.0/0.0 ELSE v END, s, 0.1), "
                     "softavg(v, s, 0.1) FROM t WHERE g IN (1,3)").fetchone()
    check(close(a, c.execute("SELECT softavg(v, s, 0.1) FROM t WHERE g=3").fetchone()[0], 1e-15),
          "NaN (=NULL in SQLite) skipped")
    # integer arguments are fine
    check(close(c.execute("SELECT softavg(2, 1, 1)").fetchone()[0], 2.0, 0), "integer args")
    # errors
    errs = {
        "NULL eps": "SELECT softavg(v, s, NULL) FROM t",
        "negative eps": "SELECT softavg(v, s, -1) FROM t",
        "eps varies": "SELECT softavg(v, s, CASE WHEN v > 2 THEN 0.1 ELSE 0.2 END) FROM t",
        "eps varies in window": "SELECT softavg(v, s, 0.1 + rowid) OVER (ROWS 2 PRECEDING) FROM t",
    }
    for what, q in errs.items():
        try:
            c.execute(q).fetchall()
            check(False, f"{what}: no error")
        except sqlite3.OperationalError as e:
            check("softavg: eps" in str(e), f"{what}: {e}")
    # NULL eps on a non-live row is not an error (row is skipped first)
    check(c.execute("SELECT softavg(v, s, NULL) FROM t WHERE g = 2").fetchone()[0] is None,
          "NULL eps on skipped rows")


def test_softavg_agg():
    """Aggregate-only twin: same NULL/eps semantics, bit-identical to softavg
    under GROUP BY, and refused as a window function."""
    c = connect()
    rng = np.random.RandomState(5)
    c.execute("CREATE TABLE a(g INTEGER, s REAL, v REAL)")
    rows = [(int(rng.randint(0, 40)), float(rng.uniform(-1, 1)), float(rng.uniform(0, 10))) for _ in range(20000)]
    rows += [(1, None, 5.0), (2, 0.3, None), (99, None, None)]
    c.executemany("INSERT INTO a VALUES (?,?,?)", rows)
    for eps in (1e-4, 1e-3, 0.02, 1.0, 0.0, math.inf):
        r = c.execute("SELECT g, softavg(v, s, ?), softavg_agg(v, s, ?) FROM a GROUP BY g", (eps, eps)).fetchall()
        check(all((x is None and y is None) or (x is not None and y is not None and x == y) for _, x, y in r),
              f"softavg_agg == softavg bitwise at eps={eps}")
        check(dict((g, y) for g, _, y in r)[99] is None, "softavg_agg all-NULL group -> NULL")
    check(c.execute("SELECT softavg_agg(v, s, 0.1) FROM a WHERE g = 1000").fetchone()[0] is None, "agg empty")
    for q in ("SELECT softavg_agg(v, s, NULL) FROM a", "SELECT softavg_agg(v, s, -1) FROM a",
              "SELECT softavg_agg(v, s, CASE WHEN v > 2 THEN 0.1 ELSE 0.2 END) FROM a"):
        try:
            c.execute(q).fetchall()
            check(False, f"no error: {q}")
        except sqlite3.OperationalError as e:
            check("softavg: eps" in str(e), f"{q}: {e}")
    try:
        c.execute("SELECT softavg_agg(v, s, 0.1) OVER (ROWS 2 PRECEDING) FROM a").fetchall()
        check(False, "softavg_agg accepted as a window function")
    except sqlite3.OperationalError as e:
        check("window" in str(e), f"softavg_agg OVER: {e}")


# ------------------------------------------------------------ window shapes
def test_window_shapes():
    """Sharp eps and every frame shape SQLite supports, vs a fresh numpy fold
    per frame. Bound (PROTOCOL): |err| <= 1e-9 * max|v|."""
    rng = np.random.RandomState(0)
    n = 1500
    s = rng.uniform(-1, 1, n)
    v = np.round(rng.uniform(0, 10, n), 3)
    part = rng.randint(0, 3, n)
    c = connect()
    c.execute("CREATE TABLE w(id INTEGER PRIMARY KEY, p INTEGER, s REAL, v REAL)")
    rows = [(i, int(part[i]), float(s[i]), float(v[i])) for i in range(n)]
    rows[33] = (33, rows[33][1], None, rows[33][3])       # NULL score
    rows[34] = (34, rows[34][1], rows[34][2], None)       # NULL value
    c.executemany("INSERT INTO w VALUES (?,?,?,?)", rows)
    live = {i: (r[2], r[3]) for i, r in enumerate(rows) if r[2] is not None and r[3] is not None}
    vmax = max(abs(x[1]) for x in live.values())
    # frame(k, ids) -> ids in the frame of the k-th row of the partition
    frames = {
        "ROWS 49 PRECEDING": lambda k, ids: ids[max(0, k - 49):k + 1],
        "ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING": lambda k, ids: ids[max(0, k - 5):k + 6],
        "ROWS BETWEEN 10 PRECEDING AND 60 FOLLOWING": lambda k, ids: ids[max(0, k - 10):k + 61],
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW": lambda k, ids: ids[:k + 1],
        "ROWS BETWEEN 20 PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW":
            lambda k, ids: ids[max(0, k - 20):k],
        "RANGE BETWEEN 30 PRECEDING AND CURRENT ROW":
            lambda k, ids: [j for j in ids[max(0, k - 31):k + 1] if ids[k] - 30 <= j],
        "GROUPS BETWEEN 3 PRECEDING AND 1 FOLLOWING": lambda k, ids: ids[max(0, k - 3):k + 2],
    }
    n_checked = 0
    worst = 0.0
    c.execute("SELECT softavg_stats_reset()")
    for eps in (1e-4, 1e-3, 0.02, 1.0, 0.0, math.inf):
        for fname, fr in frames.items():
            q = (f"SELECT id, softavg(v, s, ?) OVER (PARTITION BY p ORDER BY id {fname}) "
                 "FROM w ORDER BY id")
            got = dict(c.execute(q, (eps,)).fetchall())
            for pp in range(3):
                ids = [i for i in range(n) if part[i] == pp]
                for k, i in enumerate(ids):
                    js = [j for j in fr(k, ids) if j in live]
                    want = ref_softavg([live[j][0] for j in js], [live[j][1] for j in js], eps)
                    g = got[i]
                    n_checked += 1
                    if want is None or g is None:
                        check(want is None and g is None, f"{fname} eps={eps} id={i}: {g} vs {want}")
                        continue
                    err = abs(g - want) / vmax
                    worst = max(worst, err)
                    check(err <= 1e-9, f"{fname} eps={eps} id={i}: {g} vs {want}")
    stats = json.loads(c.execute("SELECT softavg_stats()").fetchone()[0])
    check(stats["refolds"] > 0, f"sharp eps should refold: {stats}")
    check(stats["refold_refused"] == 0, f"no refused refold expected: {stats}")
    return n_checked, worst, stats


def test_refuse_instead_of_guess():
    """First frame of 201 rows (> prefix cap 64) and a guard trip while those
    unrecorded rows are still in the frame: must raise, not return."""
    c = connect()
    c.execute("CREATE TABLE r(id INTEGER PRIMARY KEY, s REAL, v REAL)")
    c.executemany("INSERT INTO r VALUES (?,?,?)",
                  [(i, -0.5 - i * 1e-3, float(i)) for i in range(400)])
    c.execute("UPDATE r SET s = 0.9, v = 1000 WHERE id = 0")
    try:
        c.execute("SELECT softavg(v, s, 1e-3) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 200 FOLLOWING) "
                  "FROM r").fetchall()
        check(False, "expected refusal")
    except sqlite3.OperationalError as e:
        check("cannot be refolded" in str(e), f"refusal message: {e}")
    # same data, frame whose first frame is one row: refolds and is exact
    got = [x for (x,) in c.execute(
        "SELECT softavg(v, s, 1e-3) OVER (ORDER BY id ROWS BETWEEN 200 PRECEDING AND CURRENT ROW) FROM r")]
    data = c.execute("SELECT s, v FROM r ORDER BY id").fetchall()
    ok = all(abs(got[i] - ref_softavg([x[0] for x in data[max(0, i - 200):i + 1]],
                                      [x[1] for x in data[max(0, i - 200):i + 1]], 1e-3))
             <= 1e-9 * 1000 for i in range(400))
    check(ok, "preceding frame over the same data")


# ------------------------------------------------------------ dot / sqlite-vec
def test_dot():
    c = connect(vec=os.path.exists(VEC0 + ".so"))
    rng = np.random.RandomState(1)
    for dim in (1, 7, 384):
        a = rng.randn(dim).astype(np.float32)
        b = rng.randn(dim).astype(np.float32)
        want = float(a.astype(np.float64) @ b.astype(np.float64))
        got = c.execute("SELECT softagg_dot(?, ?)", (a.tobytes(), b.tobytes())).fetchone()[0]
        check(abs(got - want) <= 1e-12 * max(1.0, abs(want)), f"dot dim={dim}: {got} vs {want}")
    check(c.execute("SELECT softagg_dot(NULL, x'00000000')").fetchone()[0] is None, "dot NULL")
    for q in ("SELECT softagg_dot('[1,2]', '[1,2]')", "SELECT softagg_dot(x'0000803f', x'0000803f00000000')"):
        try:
            c.execute(q).fetchone()
            check(False, f"no error: {q}")
        except sqlite3.OperationalError as e:
            check("softagg_dot" in str(e), str(e))
    if os.path.exists(VEC0 + ".so"):
        got = c.execute("SELECT softagg_dot(vec_f32('[1,2,3]'), vec_f32('[4,5,6]'))").fetchone()[0]
        check(got == 32.0, "dot over sqlite-vec vec_f32 blobs")
        a = rng.randn(384).astype(np.float32)
        a /= np.linalg.norm(a)
        b = rng.randn(384).astype(np.float32)
        b /= np.linalg.norm(b)
        d, ip = c.execute("SELECT vec_distance_cosine(?, ?), softagg_dot(?, ?)",
                          (a.tobytes(), b.tobytes(), a.tobytes(), b.tobytes())).fetchone()
        check(abs((1 - d) - ip) < 1e-6, f"1 - cosine distance == dot on unit vectors: {1 - d} vs {ip}")


if __name__ == "__main__":
    print("sqlite", sqlite3.sqlite_version, "| lib", SO)
    ng, nw = test_conformance()
    print(f"conformance: {ng} group cells, {nw} window rows")
    test_null_and_errors()
    test_softavg_agg()
    nc, worst, stats = test_window_shapes()
    print(f"window shapes: {nc} frame values, worst |err|/max|v| = {worst:.3e}, stats {stats}")
    test_refuse_instead_of_guess()
    test_dot()
    print(f"checks: {CHECKS['pass']} passed, {CHECKS['fail']} failed")
    for f in FAILS[:30]:
        print("FAIL", f)
    sys.exit(1 if CHECKS["fail"] else 0)

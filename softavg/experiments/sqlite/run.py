"""SQLite before/after matrix (experiments/PROTOCOL.md) on IMDb.

Same SQLite, same table, same score function (softagg_dot over the float32
BLOB embedding; numpy float64 is the reference), only the aggregate spelling
changes:

  naive            SUM(exp(s/eps)*v) / SUM(exp(s/eps))
  maxshift_cte     per-group MAX(s) in a CTE, joined back, exp((s-mx)/eps)
  maxshift_window  MAX(s) OVER (PARTITION BY facet), then the shifted sums
  clamp            maxshift_cte with exp(max((s-mx)/eps, -700))
  softavg          softavg(v, s, eps)                    (this extension; window-capable)
  softavg_agg      softavg_agg(v, s, eps): aggregate-only twin (bare ScalarAcc
                   state, no InvAcc bound / prefix record), bit-identical result
  topk_avg_k       sqlite-vec vec0 KNN (cosine == inner product on unit
                   vectors), k rows passing the predicate, AVG(v) per facet

Two timing modes for the aggregate spellings:
  agg   scores precomputed into TEMP TABLE sc (untimed): times the aggregate only
  e2e   WITH sc AS MATERIALIZED (score from BLOB) + the same aggregate (timed);
        plus softavg_direct, the one-statement spelling without the CTE.

Sliding window (ORDER BY id ROWS BETWEEN 49 PRECEDING AND CURRENT ROW, over the
admitted rows, scores precomputed):
  win_softavg      softavg(...) OVER w                   (xInverse, O(1)/row)
  win_naive        SUM(exp(s/eps)*v) OVER w / SUM(exp(s/eps)) OVER w
  win_global_shift the same shifted by the table-wide MAX(s) (the only shift a
                   single window pass can express; nested windows are illegal)
  win_exact_sql    per-frame MAX(s) OVER w, then a correlated subquery over the
                   frame (exact, O(rows x frame))

Output: results.json (+ README tables via report.py). Timings are smoke.
"""
import argparse
import json
import math
import time

import numpy as np

import common

ap = argparse.ArgumentParser()
ap.add_argument("--n-queries", type=int, default=200, help="queries for the accuracy matrix")
ap.add_argument("--n-time", type=int, default=5, help="queries (first n) that are timed")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--topk", type=int, nargs="+", default=[100, 1000, 4096, 10000])
ap.add_argument("--n-window-queries", type=int, default=3)
ap.add_argument("--window-frame", type=int, default=50)
ap.add_argument("--out", default=str(common.HERE / "results.json"))
a = ap.parse_args()

INF = float("inf")
df, emb = common.load_corpus()
c = common.build_db(df, emb)
prov = common.provenance(c)
print(json.dumps(prov["machine"]["load"]), flush=True)

adm = df.filter_num.values >= common.MIN_NUM
ids_adm = np.flatnonzero(adm)
facets_all = df.facet.values
import pandas as pd  # noqa: E402
codes, facets = pd.factorize(facets_all[adm])
G = len(facets)
vals = df.value.values[adm].astype(np.float64)
E64 = emb[adm].astype(np.float64)
qidx, Q = common.queries(df, emb, a.n_queries)

SCORE_CTE = ("sc AS MATERIALIZED (SELECT facet, value, softagg_dot(emb, :q) AS s "
             "FROM imdb WHERE filter_num >= 2000)")
MCTE = "m AS (SELECT facet, MAX(s) AS mx FROM sc GROUP BY facet)"
SPELLINGS = {
    "naive": ([], "SELECT facet, SUM(exp(s/:eps)*value) / SUM(exp(s/:eps)) FROM sc GROUP BY facet"),
    "maxshift_cte": ([MCTE], "SELECT sc.facet, SUM(exp((s-mx)/:eps)*value) / SUM(exp((s-mx)/:eps)) "
                             "FROM sc JOIN m USING (facet) GROUP BY sc.facet"),
    "maxshift_window": ([], "SELECT facet, SUM(exp((s-mx)/:eps)*value) / SUM(exp((s-mx)/:eps)) FROM "
                            "(SELECT facet, value, s, MAX(s) OVER (PARTITION BY facet) AS mx FROM sc) "
                            "GROUP BY facet"),
    "clamp": ([MCTE], "SELECT sc.facet, SUM(exp(max((s-mx)/:eps, -700))*value) / "
                      "SUM(exp(max((s-mx)/:eps, -700))) FROM sc JOIN m USING (facet) GROUP BY sc.facet"),
    "softavg": ([], "SELECT facet, softavg(value, s, :eps) FROM sc GROUP BY facet"),
    "softavg_agg": ([], "SELECT facet, softavg_agg(value, s, :eps) FROM sc GROUP BY facet"),
}


def sql_for(mode, name):
    ctes, body = SPELLINGS[name]
    if mode == "e2e":
        ctes = [SCORE_CTE] + ctes
    return ("WITH " + ", ".join(ctes) + " " if ctes else "") + body


DIRECT = ("SELECT facet, softavg(value, softagg_dot(emb, :q), :eps) FROM imdb "
          "WHERE filter_num >= 2000 GROUP BY facet")
TOPK = ("SELECT i.facet, AVG(i.value), COUNT(*) FROM (SELECT rowid, distance FROM vec_imdb "
        "WHERE emb MATCH :q AND k = :k AND filter_num >= 2000) t JOIN imdb i ON i.id = t.rowid "
        "GROUP BY i.facet")


def run(sql, params, reps):
    """-> (rows or None, error string or None, list of seconds)"""
    times, rows, err = [], None, None
    for _ in range(reps):
        t0 = time.perf_counter()
        try:
            rows = c.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001  -- record verbatim
            err = f"{type(e).__name__}: {e}"
            times.append(time.perf_counter() - t0)
            break
        times.append(time.perf_counter() - t0)
    return rows, err, times


def classify(rows, ref):
    """Per-facet comparison against the float64 reference."""
    got = {f: x for f, x, *_ in rows} if rows is not None else {}
    out = dict(n_groups=len(got), silent_null=0, silent_inf=0, silent_nan=0, missing=0,
               max_rel_err=0.0, max_abs_err=0.0, worst=None)
    for g, f in enumerate(facets):
        r = ref[g]
        if r is None:
            continue
        if f not in got:
            out["missing"] += 1
            continue
        x = got[f]
        if x is None:
            out["silent_null"] += 1
        elif math.isinf(x):
            out["silent_inf"] += 1
        elif math.isnan(x):
            out["silent_nan"] += 1
        else:
            rel = abs(x - r) / abs(r)
            if rel > out["max_rel_err"]:
                out["worst"] = {"facet": f, "got": x, "ref": r}
            out["max_rel_err"] = max(out["max_rel_err"], rel)
            out["max_abs_err"] = max(out["max_abs_err"], abs(x - r))
    return out


def merge(acc, cl, err, times, timed):
    acc["queries"] += 1
    if err:
        acc["errors"] += 1
        acc["error_samples"] = sorted(set(acc["error_samples"]) | {err})[:3]
    else:
        for k in ("silent_null", "silent_inf", "silent_nan", "missing"):
            acc[k] += cl[k]
            acc["queries_with_" + k] += int(cl[k] > 0)
        if cl["max_rel_err"] >= acc["max_rel_err"]:
            acc["worst"] = cl["worst"]
        acc["max_rel_err"] = max(acc["max_rel_err"], cl["max_rel_err"])
        acc["max_abs_err"] = max(acc["max_abs_err"], cl["max_abs_err"])
    if timed:
        acc["times_s"].append(min(times))


def new_acc():
    d = dict(queries=0, errors=0, error_samples=[], max_rel_err=0.0, max_abs_err=0.0, worst=None, times_s=[])
    for k in ("silent_null", "silent_inf", "silent_nan", "missing"):
        d[k] = 0
        d["queries_with_" + k] = 0
    return d


cells = {}   # (mode, spelling, eps) -> acc
topk = {}    # k -> acc per eps + coverage
score_fn = {"softagg_dot": [], "sqlite_vec_1_minus_cosine": []}
t_start = time.time()
for qi, qv in enumerate(Q):
    timed = qi < a.n_time
    reps = a.reps if timed else 1
    qb = qv.tobytes()
    s_ref = E64 @ qv.astype(np.float64)
    refs = {e: common.softavg_ref(codes, G, s_ref, vals, e) for e in common.EPS_GRID}
    c.execute("DROP TABLE IF EXISTS temp.sc")
    c.execute("CREATE TEMP TABLE sc AS SELECT facet, value, softagg_dot(emb, :q) AS s "
              "FROM imdb WHERE filter_num >= 2000", {"q": qb})
    for eps in common.EPS_GRID:
        P = {"q": qb, "eps": eps}
        for mode in ("agg", "e2e"):
            if mode == "e2e" and not timed:
                continue   # identical SQL results; e2e only for timing
            names = list(SPELLINGS) + (["softavg_direct"] if mode == "e2e" else [])
            for name in names:
                sql = DIRECT if name == "softavg_direct" else sql_for(mode, name)
                rows, err, times = run(sql, P, reps)
                acc = cells.setdefault((mode, name, eps), new_acc())
                merge(acc, classify(rows, refs[eps]) if not err else None, err, times, timed)
    # top-k (one KNN per k, compared at every eps)
    for k in a.topk:
        rows, err, times = run(TOPK, {"q": qb, "k": k}, reps)
        t = topk.setdefault(k, {"queries": 0, "errors": 0, "error_samples": [], "coverage": [],
                                "rows_returned": [], "times_s": [], "by_eps": {}})
        t["queries"] += 1
        if err:
            t["errors"] += 1
            t["error_samples"] = sorted(set(t["error_samples"]) | {err})[:3]
            continue
        t["coverage"].append(len(rows) / G)
        t["rows_returned"].append(sum(r[2] for r in rows))
        if timed:
            t["times_s"].append(min(times))
        for eps in common.EPS_GRID:
            cl = classify(rows, refs[eps])
            b = t["by_eps"].setdefault(common.eps_name(eps), {"max_rel_err": 0.0, "missing_groups": 0})
            b["max_rel_err"] = max(b["max_rel_err"], cl["max_rel_err"])
            b["missing_groups"] += cl["missing"]
    if timed:
        for name, sql in (("softagg_dot", "SELECT sum(softagg_dot(emb, :q)) FROM imdb WHERE filter_num >= 2000"),
                          ("sqlite_vec_1_minus_cosine",
                           "SELECT sum(1 - vec_distance_cosine(emb, :q)) FROM imdb WHERE filter_num >= 2000")):
            _, _, times = run(sql, {"q": qb}, a.reps)
            score_fn[name].append(min(times))
    if qi % 10 == 0:
        print(f"query {qi + 1}/{len(Q)}  {time.time() - t_start:.0f}s", flush=True)


# ------------------------------------------------------------ sliding window
F = a.window_frame
W = f"WINDOW w AS (ORDER BY rn ROWS BETWEEN {F - 1} PRECEDING AND CURRENT ROW)"
WIN = {
    "win_softavg": f"SELECT rn, softavg(value, s, :eps) OVER w AS r FROM sc {W}",
    "win_naive": f"SELECT rn, SUM(exp(s/:eps)*value) OVER w / SUM(exp(s/:eps)) OVER w AS r FROM sc {W}",
    "win_global_shift": ("SELECT rn, SUM(exp((s-g)/:eps)*value) OVER w / SUM(exp((s-g)/:eps)) OVER w AS r "
                         f"FROM (SELECT rn, value, s, (SELECT MAX(s) FROM sc) AS g FROM sc) {W}"),
    "win_exact_sql": ("SELECT a.rn, (SELECT SUM(exp((b.s-a.m)/:eps)*b.value) / SUM(exp((b.s-a.m)/:eps)) "
                      f"FROM sc b WHERE b.rn BETWEEN a.rn - {F - 1} AND a.rn) AS r "
                      f"FROM (SELECT rn, MAX(s) OVER w AS m FROM sc {W}) a"),
}
# timing wraps the window query so that 232k rows are not shipped to Python
TIMED = "SELECT count(r), total(r) FROM ({})"
win = {}
for qi in range(a.n_window_queries):
    qv = Q[qi]
    qb = qv.tobytes()
    c.execute("DROP TABLE IF EXISTS temp.sc")
    # rn: dense 1..N in id order, so the correlated spelling can address a ROWS frame
    c.execute("CREATE TEMP TABLE sc(rn INTEGER PRIMARY KEY, id INTEGER, value REAL, s REAL)")
    c.execute("INSERT INTO sc SELECT row_number() OVER (ORDER BY id), id, value, softagg_dot(emb, :q) "
              "FROM imdb WHERE filter_num >= 2000 ORDER BY id", {"q": qb})
    idcol = np.array([r[0] for r in c.execute("SELECT id FROM sc ORDER BY rn")])
    assert np.array_equal(idcol, ids_adm)
    s_ref = E64 @ qv.astype(np.float64)
    sw = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(F - 1, -np.inf), s_ref]), F)
    vw = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.zeros(F - 1), vals]), F)
    m = sw.max(axis=1, keepdims=True)
    for eps in common.EPS_GRID:
        if eps == INF:
            cnt = np.minimum(np.arange(1, len(vals) + 1), F)
            ref = vw.sum(axis=1) / cnt
        else:
            w = np.exp((sw - m) / eps)
            ref = (w * vw).sum(axis=1) / w.sum(axis=1)
        for name, sql in WIN.items():
            c.execute("SELECT softavg_stats_reset()")
            rows, err, _ = run(sql + " ORDER BY 1", {"eps": eps}, 1)
            acc = win.setdefault((name, eps), new_acc())
            acc["queries"] += 1
            if name == "win_softavg":
                acc.setdefault("stats", []).append(json.loads(c.execute("SELECT softavg_stats()").fetchone()[0]))
            if err:
                acc["errors"] += 1
                acc["error_samples"] = sorted(set(acc["error_samples"]) | {err})[:3]
                continue
            _, _, times = run(TIMED.format(sql), {"eps": eps}, a.reps)
            acc["times_s"].append(min(times))
            got = np.array([np.nan if r[1] is None else r[1] for r in rows])
            nulls = int(np.isnan(got).sum())
            infs = int(np.isinf(got).sum())
            ok = np.isfinite(got)
            rel = np.abs(got[ok] - ref[ok]) / np.abs(ref[ok])
            acc["silent_null"] += nulls
            acc["silent_inf"] += infs
            acc["queries_with_silent_null"] += int(nulls > 0)
            acc["queries_with_silent_inf"] += int(infs > 0)
            acc["max_rel_err"] = max(acc["max_rel_err"], float(rel.max()) if len(rel) else 0.0)
            acc["rows"] = len(rows)
    print(f"window query {qi + 1}/{a.n_window_queries}", flush=True)
c.execute("DROP TABLE IF EXISTS temp.sc")


def summarise(acc):
    t = acc.pop("times_s")
    acc["median_ms"] = float(np.median(t)) * 1e3 if t else None
    acc["timed_queries"] = len(t)
    return acc


out = {
    "experiment": "sqlite before/after: GROUP BY facet softavg on IMDb (filter_num >= 2000), sliding window",
    "provenance": prov,
    "timing_status": "smoke, machine shared",
    "data": {"corpus": str(common.CORP), "rows_total": int(len(df)), "rows_admitted": int(adm.sum()),
             "predicate": "filter_num >= 2000", "groups_admitted": int(G),
             "build": json.loads((common.WORK / "build.json").read_text())},
    "queries": {"draw": "PROTOCOL: RandomState(0), first sentence 4-20 words, dedup, 200 drawn",
                "n_accuracy": len(Q), "n_timed": min(a.n_time, len(Q)), "reps": a.reps,
                "latency": "per query: min over reps; reported: median over timed queries. Aggregate cells "
                           "fetch the 28 result rows; window cells are timed as SELECT count(r), total(r) "
                           "FROM (window query) so the 232,828 output rows are not shipped to Python"},
    "reference": "numpy float64: E(float32->float64) @ q, two-pass max-shift per group",
    "eps_grid": [common.eps_name(e) for e in common.EPS_GRID],
    "aggregate": [dict(mode=m, spelling=n, eps=common.eps_name(e), **summarise(v))
                  for (m, n, e), v in cells.items()],
    "topk": [],
    "score_function_ms": {k: float(np.median(v)) * 1e3 if v else None for k, v in score_fn.items()},
    "window": {"frame": f"ORDER BY id ROWS BETWEEN {F - 1} PRECEDING AND CURRENT ROW",
               "rows": int(adm.sum()), "n_queries": a.n_window_queries,
               "cells": [dict(spelling=n, eps=common.eps_name(e), **summarise(v)) for (n, e), v in win.items()]},
}
for k, t in topk.items():
    out["topk"].append({"k": k, "queries": t["queries"], "errors": t["errors"], "error_samples": t["error_samples"],
                        "coverage_mean": float(np.mean(t["coverage"])) if t["coverage"] else None,
                        "coverage_min": float(np.min(t["coverage"])) if t["coverage"] else None,
                        "rows_returned_mean": float(np.mean(t["rows_returned"])) if t["rows_returned"] else None,
                        "median_ms": float(np.median(t["times_s"])) * 1e3 if t["times_s"] else None,
                        "avg_vs_softavg_ref_by_eps": t["by_eps"]})
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out, f"{time.time() - t_start:.0f}s")

"""Sliding-window softavg on IMDb: DuckDB's segment tree over our `combine`
versus the SQL spellings a user has without the extension.

Frame W = (PARTITION BY facet ORDER BY rid ROWS BETWEEN 49 PRECEDING AND
CURRENT ROW) over the admitted rows (filter_num >= 2000), score
array_inner_product(emb, q)::DOUBLE materialised once per query in a temp
table `sc` (the dot product is identical for every spelling and would
otherwise dominate; run_grid.py covers the in-query case for GROUP BY).

  naive_window          SUM(EXP(s/eps)*v) OVER W / SUM(EXP(s/eps)) OVER W
  maxshift_partition    the textbook shift, by the PARTITION max:
                        EXP((s-M)/eps) with M = MAX(s) OVER (PARTITION BY facet);
                        a frame whose rows all sit > ~745*eps below M becomes 0/0
  exact_selfjoin        exact SQL: frame max m = MAX(s) OVER W, then the 50 frame
                        rows joined back through range(50) and EXP((s_j - m_i)/eps)
                        summed per row (an equi-join, 50 x rows)
  softavg_segtree       softavg(v, s, eps) OVER W  (default mode: WindowSegmentTree)
  softavg_naive_mode    the same with debug_window_mode = 'separate'
                        (WindowNaiveAggregator: update + finalize per frame)

Reference: per-frame two-pass max-shifted longdouble over DuckDB's own scores.
Error is absolute in units of max|v| (PROTOCOL.md window tolerance 1e-9*max|v|);
NaN / Inf / NULL results and finite results outside the tolerance are counted.
Timings run the query wrapped in SELECT count(x), sum(x) so client transfer of
232,828 rows is not timed; the accuracy run fetches every row.
"""
import argparse
import json
import math
import os
import statistics
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import common as C

CORP = "/data/softavg/corpora/imdb"
DB = os.path.join(C.HERE, "data", "imdb.duckdb")
K = 49
WIN = f"(PARTITION BY k ORDER BY rid ROWS BETWEEN {K} PRECEDING AND CURRENT ROW)"
SC = ("CREATE OR REPLACE TEMP TABLE sc AS SELECT rid, facet AS k, value AS v, "
      "array_inner_product(emb, ?::FLOAT[384])::DOUBLE AS s FROM d.movies WHERE filter_num >= 2000")


def sqls(e):
    E = C.eps_sql(e)
    return {
        "naive_window": ("builtin", f"SELECT rid, SUM(EXP(s/{E})*v) OVER w / SUM(EXP(s/{E})) OVER w AS x "
                                    f"FROM sc WINDOW w AS {WIN}"),
        "maxshift_partition": ("builtin",
                               f"SELECT rid, SUM(EXP((s-M)/{E})*v) OVER w / SUM(EXP((s-M)/{E})) OVER w AS x "
                               f"FROM (SELECT *, MAX(s) OVER (PARTITION BY k) AS M FROM sc) WINDOW w AS {WIN}"),
        "exact_selfjoin": ("builtin",
                           f"WITH b AS (SELECT k, rid, v, s, row_number() OVER (PARTITION BY k ORDER BY rid) AS rn, "
                           f"MAX(s) OVER w AS m FROM sc WINDOW w AS {WIN}), "
                           f"x AS (SELECT a.rid, a.m, a.k, a.rn - o.j AS rn2 FROM b a CROSS JOIN range({K + 1}) o(j)), "
                           f"f AS (SELECT x.rid, x.m, c.s, c.v FROM x JOIN b c ON c.k = x.k AND c.rn = x.rn2) "
                           f"SELECT rid, SUM(EXP((s-m)/{E})*v)/SUM(EXP((s-m)/{E})) AS x FROM f GROUP BY rid"),
        "softavg_segtree": ("shipped", f"SELECT rid, softavg(v, s, {E}) OVER w AS x FROM sc WINDOW w AS {WIN}"),
        "softavg_naive_mode": ("shipped_sep", f"SELECT rid, softavg(v, s, {E}) OVER w AS x FROM sc WINDOW w AS {WIN}"),
    }


def reference(k, rid, v, s, e):
    """rid-indexed per-frame reference (longdouble, two-pass with the frame max)."""
    out = {}
    order = np.lexsort((rid, k))
    k, rid, v, s = k[order], rid[order], v[order], s[order]
    bounds = np.flatnonzero(k[1:] != k[:-1]) + 1
    res = np.empty(len(rid), dtype=np.float64)
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(k)]):
        sg = np.concatenate([np.full(K, -np.inf), s[lo:hi]]).astype(np.longdouble)
        vg = np.concatenate([np.zeros(K), v[lo:hi]]).astype(np.longdouble)
        S = sliding_window_view(sg, K + 1)
        Vv = sliding_window_view(vg, K + 1)
        live = np.isfinite(S)
        if math.isinf(e):
            res[lo:hi] = (Vv.sum(1) / live.sum(1)).astype(np.float64)
        else:
            m = S.max(1, keepdims=True)
            w = np.exp((S - m) / np.longdouble(e))
            res[lo:hi] = ((w * Vv).sum(1) / w.sum(1)).astype(np.float64)
    return rid, res


def score(con, sql, ref_rid, ref_val, vmax):
    try:
        r = con.execute(sql).fetchnumpy()
    except Exception as ex:  # noqa: BLE001 - verbatim error
        return {"error": f"{type(ex).__name__}: {ex}"}
    rid = np.asarray(r["rid"])
    xm = r["x"]
    null = np.ma.getmaskarray(xm) if np.ma.isMaskedArray(xm) else np.zeros(len(rid), bool)
    x = np.ma.filled(xm, np.nan).astype(np.float64) if np.ma.isMaskedArray(xm) else np.asarray(xm, np.float64)
    pos = np.searchsorted(ref_rid, rid)
    ref = ref_val[pos]
    fin = np.isfinite(x) & ~null
    err = np.abs(x[fin] - ref[fin]) / vmax
    tol = 1e-9
    return {"error": None, "rows": int(len(rid)), "null_rows": int(null.sum()),
            "nan_rows": int((np.isnan(x) & ~null).sum()), "inf_rows": int(np.isinf(x).sum()),
            "wrong_rows_abs_err_gt_1e-9_vmax": int((err > tol).sum()),
            "max_abs_err_over_vmax": float(err.max()) if err.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(C.HERE, "runs", "window.json"))
    a = ap.parse_args()
    emb = np.load(os.path.join(CORP, "emb.npy"), mmap_mode="r")
    qrows = json.load(open(os.path.join(C.HERE, "data", "queries.json")))["rows"][: a.queries]
    out = {"meta": C.meta(), "load_before": C.load_conditions(),
           "setup": {"frame": WIN, "rows_preceding": K, "queries": len(qrows), "query_rows": qrows,
                     "rounds": a.rounds, "tolerance": "abs err <= 1e-9 * max|v| (PROTOCOL.md)",
                     "sql": {k: v[1] for k, v in sqls(0.1).items()}},
           "cells": [], "explain": {}}
    exts = {"builtin": None, "shipped": C.SHIPPED, "shipped_sep": C.SHIPPED}
    for th in (1, 8):
        C.pin(th)
        cons = {n: C.instance(p, attach=[("d", DB)], threads=th) for n, p in exts.items()}
        cons["shipped_sep"].execute("SET debug_window_mode = 'separate'")
        for qi, r in enumerate(qrows):
            q = emb[r].tolist()
            for c in cons.values():
                c.execute(SC, [q])
            d = cons["builtin"].execute("SELECT k, rid, v, s FROM sc").fetchnumpy()
            kk, rid, v, s = (np.asarray(d["k"]).astype(str), np.asarray(d["rid"]),
                             np.asarray(d["v"], np.float64), np.asarray(d["s"], np.float64))
            vmax = float(np.abs(v).max())
            if th == 1 and qi == 0:
                for n, (inst, sql) in sqls(0.1).items():
                    out["explain"][n] = cons[inst].execute("EXPLAIN " + sql).fetchall()[0][1]
            for e in C.EPS_GRID:
                ref_rid, ref_val = reference(kk, rid, v, s, e)
                o = np.argsort(ref_rid)
                ref_rid, ref_val = ref_rid[o], ref_val[o]
                qs = sqls(e)
                acc = {n: score(cons[inst], sql, ref_rid, ref_val, vmax) for n, (inst, sql) in qs.items()}
                ts = {n: [] for n in qs}
                for _ in range(a.rounds):
                    for n, (inst, sql) in qs.items():
                        if acc[n]["error"]:
                            continue
                        t0 = time.perf_counter()
                        cons[inst].execute(f"SELECT count(x), sum(x) FROM ({sql})").fetchall()
                        ts[n].append((time.perf_counter() - t0) * 1e3)
                for n in qs:
                    out["cells"].append({"query": qi, "eps": C.eps_key(e), "plan": n, "threads": th,
                                         "median_ms": statistics.median(ts[n]) if ts[n] else None,
                                         "runs_ms": ts[n], "vmax": vmax, **acc[n]})
                print(th, qi, C.eps_key(e), " ".join(
                    f"{n}={statistics.median(t):.1f}" if t else f"{n}=ERR" for n, t in ts.items()), flush=True)
            C.dump(a.out, out)
        for c in cons.values():
            c.close()
    os.sched_setaffinity(0, range(os.cpu_count()))
    out["load_after"] = C.load_conditions()
    C.dump(a.out, out)


if __name__ == "__main__":
    main()

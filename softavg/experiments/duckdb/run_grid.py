"""Before/after grid on IMDb (PROTOCOL.md): the same DuckDB, the same table,
the same similarity function (DuckDB's array_inner_product over FLOAT[384]),
only the aggregate spelling changes.

  naive            SUM(exp(s/eps)*v) / SUM(exp(s/eps))
  maxshift_cte     per-group MAX(s) in a CTE, joined back, exp((s-m)/eps)
  maxshift_window  MAX(s) OVER (PARTITION BY facet), exp((s-m)/eps)
  clamp            maxshift_window with CASE WHEN (s-m)/eps < -700 THEN 0
  softavg          softavg(v, s, eps)  (engines/duckdb, shipped build)
  softavg_p2       the same extension over core + core_proposal/p2 (NOT shipped; P2 was
                   not adopted). Only run when SOFTAVG_P2_EXT points at such a build.
runs/grid.json was produced on 2026-11-05 with the pre-P1 core (softavg column) and the
P1+P2 variant (softavg_p2 column); P1 is bit-identical, so the accuracy columns are
unchanged by the core update, only the softavg latency column predates it.

Two passes:
  accuracy  every query (200) x eps grid x plan, threads 8, one run each:
            errors (verbatim), NaN/Inf/NULL returned silently, max relative
            error vs numpy float64 (scores E64 @ q64) and vs a longdouble
            reference on DuckDB's own float32-dot scores.
  timing    the first --timing-queries queries x eps x plan x threads {1, 8},
            interleaved rounds, both with the score computed in the query and
            over precomputed scores (aggregation only). Smoke numbers.
"""
import argparse
import json
import math
import os
import statistics
import time

import numpy as np
import pandas as pd

import common as C

CORP = "/data/softavg/corpora/imdb"
DB = os.path.join(C.HERE, "data", "imdb.duckdb")
SIM = "array_inner_product(emb, ?::FLOAT[384])::DOUBLE"
BASE = f"SELECT facet AS k, value AS v, {SIM} AS s FROM d.movies WHERE filter_num >= 2000"


def sqls(e, src):
    """src: 'in_query' (score computed from emb) or 'precomputed' (temp table sc)."""
    E = C.eps_sql(e)
    b = f"({BASE})" if src == "in_query" else "(SELECT k, v, s FROM sc)"
    w = f"(SELECT k, v, s, MAX(s) OVER (PARTITION BY k) AS m FROM {b} b)"
    clampw = f"CASE WHEN (s-m)/{E} < -700 THEN 0 ELSE EXP((s-m)/{E}) END"
    return {
        "naive": ("builtin", f"SELECT k, SUM(EXP(s/{E})*v)/SUM(EXP(s/{E})) FROM {b} b GROUP BY k"),
        "maxshift_cte": ("builtin",
                         f"WITH b AS {b}, mx AS (SELECT k, MAX(s) AS m FROM b GROUP BY k) "
                         f"SELECT b.k, SUM(EXP((b.s-mx.m)/{E})*b.v)/SUM(EXP((b.s-mx.m)/{E})) "
                         f"FROM b JOIN mx ON b.k = mx.k GROUP BY b.k"),
        "maxshift_window": ("builtin", f"SELECT k, SUM(EXP((s-m)/{E})*v)/SUM(EXP((s-m)/{E})) FROM {w} w GROUP BY k"),
        "clamp": ("builtin", f"SELECT k, SUM({clampw}*v)/SUM({clampw}) FROM {w} w GROUP BY k"),
        "softavg": ("shipped", f"SELECT k, softavg(v, s, {E}) FROM {b} b GROUP BY k"),
        "softavg_p2": ("p2", f"SELECT k, softavg(v, s, {E}) FROM {b} b GROUP BY k"),
    }


_sqls_all = sqls


def sqls(e, src):  # noqa: F811
    qs = _sqls_all(e, src)
    if not os.environ.get("SOFTAVG_P2_EXT"):
        qs.pop("softavg_p2")
    return qs


def ref_groups(codes, facets, s, v, e, dtype):
    s = s.astype(dtype)
    v = v.astype(dtype)
    G = len(facets)
    if math.isinf(e):
        num = np.zeros(G, dtype); den = np.zeros(G, dtype)
        np.add.at(num, codes, v); np.add.at(den, codes, 1)
        return {facets[g]: float(num[g] / den[g]) for g in range(G) if den[g] > 0}
    m = np.full(G, -np.inf, dtype); np.maximum.at(m, codes, s)
    w = np.exp((s - m[codes]) / dtype(e))
    num = np.zeros(G, dtype); den = np.zeros(G, dtype)
    np.add.at(num, codes, w * v); np.add.at(den, codes, w)
    return {facets[g]: float(num[g] / den[g]) for g in range(G) if den[g] > 0}


def run(con, sql, q):
    try:
        return con.execute(sql, [q] * sql.count("?")).fetchall(), None
    except Exception as ex:  # noqa: BLE001 - verbatim DuckDB error
        return None, f"{type(ex).__name__}: {ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--timing-queries", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(C.HERE, "runs", "grid.json"))
    a = ap.parse_args()

    df = pd.read_parquet(os.path.join(CORP, "corpus.parquet"), columns=["facet", "value", "filter_num"])
    emb = np.load(os.path.join(CORP, "emb.npy"), mmap_mode="r")
    qrows = json.load(open(os.path.join(C.HERE, "data", "queries.json")))["rows"][: a.queries]
    adm = np.flatnonzero(df.filter_num.values >= 2000)
    codes, facets = pd.factorize(df.facet.values[adm])
    vals = df.value.values[adm].astype(np.float64)
    E64 = np.asarray(emb[adm], dtype=np.float64)

    exts = {"builtin": None, "shipped": C.SHIPPED, "p2": os.environ.get("SOFTAVG_P2_EXT")}
    out = {"meta": C.meta(), "load_before": C.load_conditions(),
           "setup": {"db": DB, "rows_admitted": int(len(adm)), "facets_admitted": int(len(facets)),
                     "similarity": SIM, "predicate": "filter_num >= 2000", "queries": len(qrows),
                     "query_rows": qrows, "timing_queries": a.timing_queries, "rounds": a.rounds,
                     "sql_in_query": {k: v[1] for k, v in sqls(0.1, "in_query").items()},
                     "sql_precomputed": {k: v[1] for k, v in sqls(0.1, "precomputed").items()}},
           "accuracy": [], "timing": [], "explain": {}}

    # ---------------------------------------------------------------- accuracy
    C.pin(8)
    cons = {n: C.instance(p, attach=[("d", DB)], threads=8) for n, p in exts.items()}
    q0 = emb[qrows[0]].tolist()
    for name, (inst, sql) in sqls(0.1, "in_query").items():
        out["explain"][name] = cons[inst].execute("EXPLAIN " + sql, [q0] * sql.count("?")).fetchall()[0][1]
    t_acc = time.time()
    for qi, r in enumerate(qrows):
        q = emb[r].tolist()
        s64 = E64 @ np.asarray(emb[r], dtype=np.float64)
        # DuckDB's own scores, in rid order (the admitted rows in corpus order)
        s_dd = cons["builtin"].execute(
            f"SELECT {SIM} AS s FROM d.movies WHERE filter_num >= 2000 ORDER BY rid", [q]).fetchnumpy()["s"]
        s_dd = np.asarray(s_dd, dtype=np.float64)
        for e in C.EPS_GRID:
            ref = ref_groups(codes, facets, s64, vals, e, np.float64)
            ref_same = ref_groups(codes, facets, s_dd, vals, e, np.longdouble)
            for name, (inst, sql) in sqls(e, "in_query").items():
                rows, err = run(cons[inst], sql, q)
                cell = {"query": qi, "eps": C.eps_key(e), "plan": name, "threads": 8, "error": err}
                if rows is not None:
                    cell.update(C.score_result(rows, ref, ref_same))
                out["accuracy"].append(cell)
        if qi % 20 == 0:
            print(f"accuracy {qi + 1}/{len(qrows)} {time.time() - t_acc:.0f}s", flush=True)
            C.dump(a.out, out)
    for c in cons.values():
        c.close()

    # ------------------------------------------------------------------ timing
    for th in (1, 8):
        C.pin(th)
        cons = {n: C.instance(p, attach=[("d", DB)], threads=th) for n, p in exts.items()}
        for qi, r in enumerate(qrows[: a.timing_queries]):
            q = emb[r].tolist()
            for c in cons.values():
                c.execute("CREATE OR REPLACE TEMP TABLE sc AS " + BASE, [q])
            s64 = E64 @ np.asarray(emb[r], dtype=np.float64)
            for e in C.EPS_GRID:
                ref = ref_groups(codes, facets, s64, vals, e, np.float64)
                for src in ("in_query", "precomputed"):
                    qs = sqls(e, src)
                    ts = {n: [] for n in qs}
                    res = {}
                    for n, (inst, sql) in qs.items():  # warm-up
                        res[n] = run(cons[inst], sql, q)
                    for _ in range(a.rounds):
                        for n, (inst, sql) in qs.items():
                            if res[n][1] is not None:
                                continue
                            t0 = time.perf_counter()
                            cons[inst].execute(sql, [q] * sql.count("?")).fetchall()
                            ts[n].append((time.perf_counter() - t0) * 1e3)
                    for n in qs:
                        rows, err = res[n]
                        cell = {"query": qi, "eps": C.eps_key(e), "plan": n, "threads": th, "score_source": src,
                                "error": err, "median_ms": statistics.median(ts[n]) if ts[n] else None,
                                "runs_ms": ts[n]}
                        if rows is not None:
                            cell.update(C.score_result(rows, ref))
                        out["timing"].append(cell)
            print(f"timing threads={th} query {qi + 1}/{a.timing_queries}", flush=True)
            C.dump(a.out, out)
        for c in cons.values():
            c.close()
    os.sched_setaffinity(0, range(os.cpu_count()))
    out["load_after"] = C.load_conditions()
    C.dump(a.out, out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Before/after matrix (PROTOCOL "之前 / 之后") on IMDb, one server at a time.

Plans, same table / same score expression / only the aggregate spelling changes:
  naive         SUM(EXP(s/eps)*v) / SUM(EXP(s/eps))
  shift         per-facet MAX(s) in a CTE, EXP((s-m)/eps)
  clamp         shift + GREATEST((s-m)/eps, -700) against exp underflow errors
  softavg       bruce_pg 0.2.0 softavg(v, s, eps)            (internal state)
  softavg_v012  bruce_pg 0.1.2's float8[5] state path, same .so (state-representation A/B)
  topk_avg      [pgvector] pgvector HNSW ORDER BY emb <#> q LIMIT k, then AVG(value)  (coverage)
  topk_softavg  [pgvector] same top-k rows, then softavg(value, s, eps)

Score modes: live = -(emb <#> q) evaluated in the query (pgvector);
precomp = float8 column s<j> (isolates the aggregation).
Every result is compared with a numpy float64 max-shifted reference computed
from the scores the server itself holds/computes.

usage: bench.py rel|dbg [n_live_queries] [n_precomp_queries]
"""
import json, sys, time
import numpy as np
from common import (EPS_GRID, FILTERS, HERE, TIMING_STATUS, TOPK, WORKERS, compare, connect, eps_key, eps_sql,
                    grouped_reference, load_queries, machine_state, query_embeddings, run_timed, vec_literal)

SERVER = sys.argv[1]
N_LIVE = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N_PRE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUT = HERE / f"cells_{SERVER}.jsonl"

PLANS = {
    "naive": "SELECT facet, SUM(EXP(s/{E})*value) / SUM(EXP(s/{E})) FROM (SELECT facet, value, {S} AS s FROM imdb {F}) t GROUP BY facet",
    "shift": ("WITH m AS (SELECT facet, MAX({S}) AS m FROM imdb {F} GROUP BY facet) "
              "SELECT t.facet, SUM(EXP((t.s - m.m)/{E})*t.value) / SUM(EXP((t.s - m.m)/{E})) "
              "FROM (SELECT facet, value, {S} AS s FROM imdb {F}) t JOIN m USING (facet) GROUP BY t.facet"),
    "clamp": ("WITH m AS (SELECT facet, MAX({S}) AS m FROM imdb {F} GROUP BY facet) "
              "SELECT t.facet, SUM(EXP(GREATEST((t.s - m.m)/{E}, -700))*t.value) / SUM(EXP(GREATEST((t.s - m.m)/{E}, -700))) "
              "FROM (SELECT facet, value, {S} AS s FROM imdb {F}) t JOIN m USING (facet) GROUP BY t.facet"),
    "softavg": "SELECT facet, softavg(value, {S}, {E}) FROM imdb {F} GROUP BY facet",
    "softavg_v012": "SELECT facet, softavg_v012(value, {S}, {E}) FROM imdb {F} GROUP BY facet",
}
TOPK_SQL = ("SELECT facet, {AGG}, count(*) FROM (SELECT facet, value, -(emb <#> {Q}) AS s FROM imdb {F} "
            "ORDER BY emb <#> {Q} LIMIT {K}) t GROUP BY facet")
TOPK_SETTINGS = ["SET hnsw.ef_search = {EF}", "SET hnsw.iterative_scan = 'relaxed_order'",
                 "SET hnsw.max_scan_tuples = 200000", "SET max_parallel_workers_per_gather = 0"]


def render(t, **kw):
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def fetch_base(cur):
    cur.execute("SELECT id, facet, value, filter_num FROM imdb ORDER BY id")
    r = cur.fetchall()
    return (np.array([x[0] for x in r]), np.array([x[1] for x in r], dtype=object),
            np.array([x[2] for x in r], dtype=np.float64), np.array([x[3] for x in r], dtype=np.float64))


def fetch_scores(cur, expr):
    cur.execute(f"SELECT {expr} FROM imdb ORDER BY id")
    return np.array([x[0] for x in cur.fetchall()], dtype=np.float64)


def emit(f, cell):
    f.write(json.dumps(cell) + "\n"); f.flush()
    print(json.dumps({k: cell.get(k) for k in ("mode", "query", "plan", "eps", "filter", "workers", "k", "median_ms",
                                                "max_rel_err", "coverage", "error_message")}), flush=True)


def main():
    queries = load_queries()["queries"]
    conn = connect(SERVER); cur = conn.cursor()
    ids, facet, value, fnum = fetch_base(cur)
    masks = {"none": np.ones(len(ids), bool), "ge2000": fnum >= 2000}
    jobs = []  # (mode, qi, score_expr)
    cur.execute("SELECT count(*) > 0 FROM pg_extension WHERE extname = 'vector'")
    if cur.fetchone()[0]:
        Q = query_embeddings([q["row"] for q in queries[:max(N_LIVE, 1)]])
        jobs += [("live", j, f"-(emb <#> {vec_literal(Q[j])})") for j in range(N_LIVE)]
    jobs += [("precomp", j, f"s{j}") for j in range(N_PRE)]
    with open(OUT, "w") as f:
        f.write(json.dumps(dict(header=True, server=SERVER, machine_at_start=machine_state(),
                                timing_status=TIMING_STATUS)) + "\n")
        for mode, j, S in jobs:
            s = fetch_scores(cur, S)
            refs = {(fn, eps_key(e)): grouped_reference(facet[m], value[m], s[m], e)
                    for fn, m in masks.items() for e in EPS_GRID}
            if mode == "live" and N_PRE > j:  # the precomputed column must hold the same scores
                sc = fetch_scores(cur, f"s{j}")
                diff = float(np.abs(sc - s).max())
                f.write(json.dumps(dict(check="live_vs_precomp_scores", query=j, max_abs_diff=diff)) + "\n")
            for W in WORKERS:
                for fn, F in FILTERS.items():
                    for plan, tmpl in PLANS.items():
                        for e in EPS_GRID:
                            sql = render(tmpl, S=S, F=F, E=eps_sql(e))
                            r = run_timed(cur, sql, settings=[f"SET max_parallel_workers_per_gather = {W}"])
                            cell = dict(server=SERVER, mode=mode, query=j, query_text=queries[j]["text"], plan=plan,
                                        eps=eps_key(e), filter=fn, workers=W, k=None)
                            if r["sql_ok"]:
                                cell.update(compare(r.pop("rows"), refs[(fn, eps_key(e))]))
                            cell.update(r); emit(f, cell)
            if mode == "live":  # top-k arms
                for fn, F in FILTERS.items():
                    for K in TOPK:
                        settings = [render(x, EF=min(K, 1000)) for x in TOPK_SETTINGS]
                        sql = render(TOPK_SQL, AGG="AVG(value)", Q=vec_literal(Q[j]), F=F, K=K)
                        r = run_timed(cur, sql, settings=settings)
                        base = dict(server=SERVER, mode=mode, query=j, query_text=queries[j]["text"], filter=fn,
                                    workers=0, k=K, hnsw=dict(ef_search=min(K, 1000), iterative_scan="relaxed_order",
                                                              max_scan_tuples=200000))
                        if r["sql_ok"]:
                            rows = r.pop("rows")
                            r["topk_rows_returned"] = int(sum(x[2] for x in rows))
                            for e in EPS_GRID:  # AVG answers one question; scored against each eps
                                emit(f, dict(base, plan="topk_avg", eps=eps_key(e),
                                             **compare(rows, refs[(fn, eps_key(e))]), **r))
                        else:
                            emit(f, dict(base, plan="topk_avg", eps=None, **r))
                        for e in EPS_GRID:
                            sql = render(TOPK_SQL, AGG=f"softavg(value, s, {eps_sql(e)})", Q=vec_literal(Q[j]), F=F, K=K)
                            r = run_timed(cur, sql, settings=settings)
                            cell = dict(base, plan="topk_softavg", eps=eps_key(e))
                            if r["sql_ok"]:
                                rows = r.pop("rows")
                                r["topk_rows_returned"] = int(sum(x[2] for x in rows))
                                cell.update(compare(rows, refs[(fn, eps_key(e))]))
                            cell.update(r); emit(f, cell)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")
    print("done", OUT)


if __name__ == "__main__":
    main()

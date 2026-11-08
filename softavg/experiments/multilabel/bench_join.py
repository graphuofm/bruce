#!/usr/bin/env python
"""GROUP BY over the many-to-many join (overlapping facets): before / after.

single-label: GROUP BY imdb.facet (the corpus' one genre per movie; as ../postgres/bench.py)
multi-label:  imdb JOIN imdb_genre USING (id) GROUP BY imdb_genre.facet (complete genre memberships)

Spellings (PROTOCOL): naive, shift (per-facet MAX in a CTE, joined back), clamp
(shift + GREATEST(.., -700)), softavg. Score = precomputed s<j> (isolates the
aggregation; s<j> = -(emb <#> q_j) computed by pgvector at load). eps grid,
filter none / filter_num >= 2000, workers 0 / 8. Reference: numpy float64
max-shifted over the same (facet, value, score) multiset the join produces.

usage: bench_join.py [n_queries]   -> cells_join.jsonl
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import (EPS_GRID, TIMING_STATUS, WORKERS, compare, connect, eps_key, eps_sql,  # noqa: E402
                    grouped_reference, machine_state, run_timed)

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = HERE / "cells_join.jsonl"
FILTERS = {"none": "", "ge2000": "WHERE i.filter_num >= 2000"}
SRC = {"single": "SELECT i.facet AS facet, i.value, i.{S} AS s FROM imdb i {F}",
       "multi": "SELECT g.facet AS facet, i.value, i.{S} AS s FROM imdb i JOIN imdb_genre g ON g.id = i.id {F}"}
PLANS = {
    "naive": "SELECT facet, SUM(EXP(s/{E})*value) / SUM(EXP(s/{E})) FROM ({T}) t GROUP BY facet",
    "shift": ("WITH m AS (SELECT facet, MAX(s) AS m FROM ({T}) t0 GROUP BY facet) "
              "SELECT t.facet, SUM(EXP((t.s - m.m)/{E})*t.value) / SUM(EXP((t.s - m.m)/{E})) "
              "FROM ({T}) t JOIN m USING (facet) GROUP BY t.facet"),
    "clamp": ("WITH m AS (SELECT facet, MAX(s) AS m FROM ({T}) t0 GROUP BY facet) "
              "SELECT t.facet, SUM(EXP(GREATEST((t.s - m.m)/{E}, -700))*t.value) / SUM(EXP(GREATEST((t.s - m.m)/{E}, -700))) "
              "FROM ({T}) t JOIN m USING (facet) GROUP BY t.facet"),
    "softavg": "SELECT facet, softavg(value, s, {E}) FROM ({T}) t GROUP BY facet",
}


def main():
    conn = connect("rel"); cur = conn.cursor()
    cur.execute(f"SELECT id, facet, value, filter_num, {', '.join(f's{j}' for j in range(NQ))} FROM imdb ORDER BY id")
    r = cur.fetchall()
    ids = np.array([x[0] for x in r]); assert (ids == np.arange(len(ids))).all()
    facet = np.array([x[1] for x in r], dtype=object); value = np.array([x[2] for x in r])
    fnum = np.array([x[3] for x in r]); S = np.array([x[4:] for x in r], dtype=np.float64)
    cur.execute("SELECT id, facet FROM imdb_genre")
    br = cur.fetchall(); bid = np.array([x[0] for x in br]); bfac = np.array([x[1] for x in br], dtype=object)
    sel = {"single": (np.arange(len(ids)), facet), "multi": (bid, bfac)}
    with open(OUT, "w") as f:
        f.write(json.dumps(dict(header=True, machine_at_start=machine_state(), timing_status=TIMING_STATUS,
                                rows_single=len(ids), rows_multi=len(bid))) + "\n")
        for j in range(NQ):
            for lab, (rows_idx, keys) in sel.items():
                for fn, F in FILTERS.items():
                    m = np.ones(len(rows_idx), bool) if fn == "none" else fnum[rows_idx] >= 2000
                    ri, k = rows_idx[m], keys[m]
                    refs = {eps_key(e): grouped_reference(k, value[ri], S[ri, j], e) for e in EPS_GRID}
                    T = SRC[lab].replace("{S}", f"s{j}").replace("{F}", F)
                    for W in WORKERS:
                        for plan, tmpl in PLANS.items():
                            for e in EPS_GRID:
                                sql = tmpl.replace("{T}", T).replace("{E}", eps_sql(e))
                                res = run_timed(cur, sql, settings=[f"SET max_parallel_workers_per_gather = {W}"])
                                cell = dict(labels=lab, query=j, plan=plan, eps=eps_key(e), filter=fn, workers=W,
                                            input_rows=int(len(ri)))
                                if res["sql_ok"]:
                                    cell.update(compare(res.pop("rows"), refs[eps_key(e)]))
                                cell.update(res)
                                f.write(json.dumps(cell) + "\n"); f.flush()
                                print(json.dumps({x: cell.get(x) for x in ("labels", "query", "plan", "eps", "filter", "workers",
                                                                           "median_ms", "max_rel_err", "n_groups", "plan_shape",
                                                                           "error_message")}), flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")


if __name__ == "__main__":
    main()

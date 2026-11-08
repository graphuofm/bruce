"""Where does the time go? Before/after and cost attribution for the
GROUP BY softavg aggregate, interleaved in one process.

Instances (each a fresh in-memory DuckDB that ATTACHes the data read-only):
  builtin  no extension: naive SQL and built-in reference aggregates
  before   the starting point (paper_softavg/experiments/duckdb_ext build)
  shipped  engines/duckdb (vectorised update over softagg-core, which has P1 since
           2026-11-05: one exp per absorbed row)
  pre_p1   engines/duckdb against core with P1 reversed (= the shipped build of the
           first round; build_variants.sh). The first round's numbers, with the
           P1/P2 variants, are kept in runs/archive/gap_pre_core_p1.json
  diag     engines/duckdb --features diag (softavg_diag_* cost probes)

Datasets:
  reviews  /home/jding/bruce/paper_softavg/experiments/duckdb_ext/bench/reviews.duckdb
           (1,308,416 rows, GROUP BY genre, stored DOUBLE score; the table the
           earlier "naive 13 ms vs softavg 15 ms" number was measured on)
  imdb_q0  IMDb admitted rows (filter_num >= 2000) with the query-0 score
           array_inner_product(emb, q0) precomputed as DOUBLE (data/imdb_q0.duckdb)

Rounds are interleaved (every variant once per round) so drift on the
shared machine hits all variants alike; min and median are reported.
"""
import argparse
import json
import math
import os
import statistics
import time

import numpy as np

import common as C

REVIEWS = "/home/jding/bruce/paper_softavg/experiments/duckdb_ext/bench/reviews.duckdb"
Q0DB = os.path.join(C.HERE, "data", "imdb_q0.duckdb")


def build_q0():
    if os.path.exists(Q0DB):
        os.remove(Q0DB)
    q = json.load(open(os.path.join(C.HERE, "data", "queries.json")))["rows"][0]
    con = C.instance(attach=[("d", os.path.join(C.HERE, "data", "imdb.duckdb"))])
    con.execute(f"ATTACH '{Q0DB}' AS o")
    con.execute(f"""CREATE TABLE o.sc AS SELECT facet AS k, value AS v,
                   array_inner_product(emb, (SELECT emb FROM d.movies WHERE rid = {int(q)}))::DOUBLE AS s
                   FROM d.movies WHERE filter_num >= 2000 ORDER BY rid""")
    con.execute("DETACH o")


DATASETS = {
    "reviews": {"path": REVIEWS, "table": "x.reviews", "k": "genre", "v": "rating", "s": "s"},
    "imdb_q0": {"path": Q0DB, "table": "x.sc", "k": "k", "v": "v", "s": "s"},
}


def queries(ds, e):
    t, k, v, s = ds["table"], ds["k"], ds["v"], ds["s"]
    E = C.eps_sql(e)
    return {
        "naive": ("builtin", f"SELECT {k}, SUM(EXP({s}/{E})*{v})/SUM(EXP({s}/{E})) FROM {t} GROUP BY {k}"),
        "softavg@before": ("before", f"SELECT {k}, softavg({v}, {s}, {E}) FROM {t} GROUP BY {k}"),
        "softavg@pre_p1": ("pre_p1", f"SELECT {k}, softavg({v}, {s}, {E}) FROM {t} GROUP BY {k}"),
        "softavg@shipped": ("shipped", f"SELECT {k}, softavg({v}, {s}, {E}) FROM {t} GROUP BY {k}"),
    }


def diag_queries(ds):
    t, k, v, s = ds["table"], ds["k"], ds["v"], ds["s"]
    return {
        "count_star": ("builtin", f"SELECT {k}, count(*) FROM {t} GROUP BY {k}"),
        "avg_v": ("builtin", f"SELECT {k}, AVG({v}) FROM {t} GROUP BY {k}"),
        "sum_s_sum_v": ("builtin", f"SELECT {k}, SUM({s}), SUM({v}) FROM {t} GROUP BY {k}"),
        "sum_exp": ("builtin", f"SELECT {k}, SUM(EXP({s}/0.1)) FROM {t} GROUP BY {k}"),
        "capi_noop_update": ("diag", f"SELECT {k}, softavg_diag_noop({v}, {s}, 0.1) FROM {t} GROUP BY {k}"),
        "capi_touch_update": ("diag", f"SELECT {k}, softavg_diag_touch({v}, {s}, 0.1) FROM {t} GROUP BY {k}"),
        "capi_softavg_2arg_eps_fixed": ("diag", f"SELECT {k}, softavg_diag_2arg({v}, {s}) FROM {t} GROUP BY {k}"),
        "capi_softavg_eps_0.1": ("diag", f"SELECT {k}, softavg({v}, {s}, 0.1) FROM {t} GROUP BY {k}"),
    }


def run_set(cons, qs, rounds, warm=2):
    for name, (inst, sql) in qs.items():
        for _ in range(warm):
            cons[inst].execute(sql).fetchall()
    ts = {n: [] for n in qs}
    res = {}
    for _ in range(rounds):
        for name, (inst, sql) in qs.items():
            t0 = time.perf_counter()
            r = cons[inst].execute(sql).fetchall()
            ts[name].append((time.perf_counter() - t0) * 1e3)
            res[name] = r
    return ts, res


def reference(path, table, k, v, s, e):
    con = C.instance(attach=[("x", path)])
    rows = con.execute(f"SELECT {k}, {v}, {s} FROM {table}").fetchall()
    keys = np.array([r[0] for r in rows])
    vv = np.array([r[1] for r in rows], dtype=np.longdouble)
    ss = np.array([r[2] for r in rows], dtype=np.longdouble)
    out = {}
    for g in np.unique(keys):
        m = keys == g
        if math.isinf(e):
            out[g] = float(vv[m].mean())
        else:
            w = np.exp((ss[m] - ss[m].max()) / np.longdouble(e))
            out[g] = float((w * vv[m]).sum() / w.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(C.HERE, "runs", "gap.json"))
    a = ap.parse_args()
    if not os.path.exists(Q0DB):
        build_q0()
    out = {"meta": C.meta(), "load_before": C.load_conditions(), "rounds": a.rounds, "cells": [], "diag": []}
    exts = {"builtin": None, "before": C.BEFORE, "shipped": C.SHIPPED, "pre_p1": C.VARIANTS["pre_p1"],
            "diag": C.DIAG}
    for dname, ds in DATASETS.items():
        refs = {e: reference(ds["path"], ds["table"], ds["k"], ds["v"], ds["s"], e) for e in C.EPS_GRID}
        for th in (1, 8):
            C.pin(th)
            cons = {n: C.instance(p, attach=[("x", ds["path"])], threads=th) for n, p in exts.items()}
            n_rows = cons["builtin"].execute(f"SELECT count(*) FROM {ds['table']}").fetchone()[0]
            for e in C.EPS_GRID:
                ts, res = run_set(cons, queries(ds, e), a.rounds)
                ship = dict(res["softavg@shipped"])
                for name in ts:
                    got = dict(res[name])
                    sc = C.score_result(res[name], refs[e])
                    cell = {"dataset": dname, "rows": n_rows, "threads": th, "eps": C.eps_key(e), "plan": name,
                            "min_ms": min(ts[name]), "median_ms": statistics.median(ts[name]), "runs_ms": ts[name],
                            **{k2: v2 for k2, v2 in sc.items() if k2 != "max_rel_err_vs_f64"},
                            "max_rel_err_vs_longdouble_same_scores": sc["max_rel_err_vs_f64"]}
                    if name.startswith("softavg@"):
                        cell["bitwise_equal_to_shipped"] = all(
                            (got[g] == ship[g]) or (got[g] != got[g] and ship[g] != ship[g]) for g in ship)
                    out["cells"].append(cell)
                print(dname, th, C.eps_key(e), " ".join(
                    f"{n}={min(v):.2f}/{statistics.median(v):.2f}" for n, v in ts.items()), flush=True)
            ts, _ = run_set(cons, diag_queries(ds), a.rounds)
            for name, v in ts.items():
                out["diag"].append({"dataset": dname, "rows": n_rows, "threads": th, "probe": name,
                                    "min_ms": min(v), "median_ms": statistics.median(v), "runs_ms": v})
            print(dname, th, "diag", " ".join(f"{n}={min(v):.2f}" for n, v in ts.items()), flush=True)
            for c in cons.values():
                c.close()
    os.sched_setaffinity(0, range(os.cpu_count()))
    out["load_after"] = C.load_conditions()
    C.dump(a.out, out)


if __name__ == "__main__":
    main()

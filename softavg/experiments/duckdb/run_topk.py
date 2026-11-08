"""The vector-index alternative inside the same DuckDB: vss (HNSW) top-k + AVG.

  SELECT facet, AVG(value) FROM (SELECT facet, value FROM adm
      ORDER BY array_negative_inner_product(emb, q) LIMIT k) GROUP BY facet

`adm` = the admitted rows (filter_num >= 2000) copied into an in-memory table
with an HNSW index (metric 'ip'); k in {100, 1000, 10000}. Recorded per query:
facet coverage (groups returned / facets in the admitted set), rows returned,
recall@k against the exact top-k (numpy float64 scores), latency. A second
index over the full table checks what happens when the predicate is in the
query instead (does DuckDB still use the index, and how many rows come back).
"""
import argparse
import json
import os
import statistics
import time

import duckdb
import numpy as np
import pandas as pd

import common as C

CORP = "/data/softavg/corpora/imdb"
DB = os.path.join(C.HERE, "data", "imdb.duckdb")
KS = [100, 1000, 10000]
TOPK = ("SELECT facet, AVG(value), count(*) FROM (SELECT facet, value FROM {t} {w} "
        "ORDER BY array_negative_inner_product(emb, ?::FLOAT[384]) LIMIT {k}) GROUP BY facet")
RIDS = "SELECT rid FROM {t} {w} ORDER BY array_negative_inner_product(emb, ?::FLOAT[384]) LIMIT {k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--timing-queries", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(C.HERE, "runs", "topk.json"))
    a = ap.parse_args()
    out = {"meta": C.meta(), "load_before": C.load_conditions()}
    # vss: installed from the DuckDB extension repository
    try:
        con = C.instance(attach=[("d", DB)], threads=8, extra_load=("vss",))
    except duckdb.Error:
        boot = duckdb.connect()
        boot.execute("INSTALL vss")
        con = C.instance(attach=[("d", DB)], threads=8, extra_load=("vss",))
    out["vss"] = dict(zip(("name", "version", "install_path", "install_mode"), con.execute(
        "SELECT extension_name, extension_version, install_path, install_mode FROM duckdb_extensions() "
        "WHERE extension_name = 'vss'").fetchone()))
    out["vss"]["settings"] = con.execute(
        "SELECT name, value FROM duckdb_settings() WHERE name LIKE 'hnsw%'").fetchall()
    out["vss"]["index_options"] = "defaults (metric='ip'; vss defaults ef_construction=128, ef_search=64, M=16)"

    C.pin(8)
    t0 = time.time()
    con.execute("CREATE TABLE adm AS SELECT rid, facet, value, emb FROM d.movies WHERE filter_num >= 2000 ORDER BY rid")
    con.execute("CREATE TABLE full_t AS SELECT rid, facet, value, filter_num, emb FROM d.movies ORDER BY rid")
    out["copy_seconds"] = time.time() - t0
    t0 = time.time()
    con.execute("CREATE INDEX hnsw_adm ON adm USING HNSW (emb) WITH (metric = 'ip')")
    out["index_build_seconds_adm_8threads"] = time.time() - t0
    t0 = time.time()
    con.execute("CREATE INDEX hnsw_full ON full_t USING HNSW (emb) WITH (metric = 'ip')")
    out["index_build_seconds_full_8threads"] = time.time() - t0
    out["memory_after_index"] = con.execute(
        "SELECT tag, memory_usage_bytes FROM duckdb_memory() WHERE memory_usage_bytes > 0").fetchall()

    df = pd.read_parquet(os.path.join(CORP, "corpus.parquet"), columns=["facet", "filter_num"])
    emb = np.load(os.path.join(CORP, "emb.npy"), mmap_mode="r")
    adm = np.flatnonzero(df.filter_num.values >= 2000)
    G = len(set(df.facet.values[adm]))
    E64 = np.asarray(emb[adm], dtype=np.float64)
    qrows = json.load(open(os.path.join(C.HERE, "data", "queries.json")))["rows"][: a.queries]
    out["setup"] = {"rows_admitted": int(len(adm)), "facets_admitted": G, "ks": KS, "queries": len(qrows),
                    "timing_queries": a.timing_queries, "rounds": a.rounds,
                    "sql_index": TOPK.format(t="adm", w="", k="{k}"),
                    "sql_filtered_full_table": TOPK.format(t="full_t", w="WHERE filter_num >= 2000", k="{k}")}
    q0 = emb[qrows[0]].tolist()
    out["explain"] = {
        "adm": con.execute("EXPLAIN " + TOPK.format(t="adm", w="", k=1000), [q0]).fetchall()[0][1],
        "full_t_where": con.execute("EXPLAIN " + TOPK.format(t="full_t", w="WHERE filter_num >= 2000", k=1000),
                                    [q0]).fetchall()[0][1]}
    out["uses_hnsw"] = {k: ("HNSW_INDEX_SCAN" in v) for k, v in out["explain"].items()}

    cells = []
    for qi, r in enumerate(qrows):
        q = emb[r].tolist()
        s64 = E64 @ np.asarray(emb[r], dtype=np.float64)
        order = np.argsort(-s64, kind="stable")
        for k in KS:
            rows = con.execute(TOPK.format(t="adm", w="", k=k), [q]).fetchall()
            rids = {x[0] for x in con.execute(RIDS.format(t="adm", w="", k=k), [q]).fetchall()}
            exact = set(int(x) for x in adm[order[:k]])
            frows = con.execute(TOPK.format(t="full_t", w="WHERE filter_num >= 2000", k=k), [q]).fetchall()
            cells.append({"query": qi, "k": k, "groups": len(rows), "coverage": len(rows) / G,
                          "rows_returned": int(sum(x[2] for x in rows)),
                          "recall_vs_exact_topk": len(rids & exact) / k,
                          "filtered_full_table_groups": len(frows),
                          "filtered_full_table_rows_returned": int(sum(x[2] for x in frows))})
        if qi % 50 == 0:
            print(f"topk coverage {qi + 1}/{len(qrows)}", flush=True)
    out["coverage_cells"] = cells

    timing = []
    for th in (1, 8):
        C.pin(th)
        con.execute(f"SET threads={th}")
        for qi, r in enumerate(qrows[: a.timing_queries]):
            q = emb[r].tolist()
            for k in KS:
                for t, w, name in (("adm", "", "hnsw_topk_avg"),
                                   ("full_t", "WHERE filter_num >= 2000", "hnsw_topk_avg_filtered_full_table")):
                    sql = TOPK.format(t=t, w=w, k=k)
                    con.execute(sql, [q]).fetchall()
                    ts = []
                    for _ in range(a.rounds):
                        t0 = time.perf_counter()
                        con.execute(sql, [q]).fetchall()
                        ts.append((time.perf_counter() - t0) * 1e3)
                    timing.append({"query": qi, "k": k, "plan": name, "threads": th,
                                   "median_ms": statistics.median(ts), "runs_ms": ts})
        print(f"topk timing threads={th}", flush=True)
    out["timing"] = timing
    os.sched_setaffinity(0, range(os.cpu_count()))
    out["load_after"] = C.load_conditions()
    con.close()
    C.dump(a.out, out)


if __name__ == "__main__":
    main()

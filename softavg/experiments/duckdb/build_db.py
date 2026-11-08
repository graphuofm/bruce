"""Build data/imdb.duckdb: the IMDb corpus with embeddings as FLOAT[384].

Table movies(rid INTEGER, item_id VARCHAR, facet VARCHAR, value DOUBLE,
filter_num DOUBLE, filter_bool BOOLEAN, emb FLOAT[384]); rid = row number in
corpus.parquet / emb.npy. Also writes data/queries.json: the 200 query rows
chosen exactly in experiments/PROTOCOL.md (RandomState(0); first sentence of `text`, 4-20
words, de-duplicated) -- PROTOCOL.md.

The corpus files are read-only inputs; data/ is derived and deleted after the
runs (see README "Cleanup").
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = "/data/softavg/corpora/imdb"
DATA = os.path.join(HERE, "data")
DB = os.path.join(DATA, "imdb.duckdb")
N_QUERIES = 200


def choose_queries(df):
    rng = np.random.RandomState(0)
    titles = df.text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    return rng.choice(np.flatnonzero(ok.values), size=N_QUERIES, replace=False)


def main():
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(os.path.join(CORP, "corpus.parquet"))
    emb = np.load(os.path.join(CORP, "emb.npy"), mmap_mode="r")
    assert emb.dtype == np.float32 and emb.shape == (len(df), 384), (emb.dtype, emb.shape)
    qidx = choose_queries(df)
    with open(os.path.join(DATA, "queries.json"), "w") as f:
        json.dump({"rule": "RandomState(0); first sentence 4-20 words, de-duplicated; 200 rows; "
                           "query vector = emb.npy[row]",
                   "rows": [int(i) for i in qidx]}, f)

    if os.path.exists(DB):
        os.remove(DB)
    con = duckdb.connect(DB)
    con.execute("""CREATE TABLE movies(rid INTEGER, item_id VARCHAR, facet VARCHAR, value DOUBLE,
                   filter_num DOUBLE, filter_bool BOOLEAN, emb FLOAT[384])""")
    step = 50_000
    for lo in range(0, len(df), step):
        hi = min(lo + step, len(df))
        part = df.iloc[lo:hi]
        flat = pa.array(np.ascontiguousarray(emb[lo:hi]).reshape(-1), type=pa.float32())
        tbl = pa.table({
            "rid": pa.array(np.arange(lo, hi, dtype=np.int32)),
            "item_id": pa.array(part.item_id.values),
            "facet": pa.array(part.facet.values),
            "value": pa.array(part.value.values, type=pa.float64()),
            "filter_num": pa.array(part.filter_num.values, type=pa.float64()),
            "filter_bool": pa.array(part.filter_bool.values),
            "emb": pa.FixedSizeListArray.from_arrays(flat, 384),
        })
        con.register("chunk", tbl)
        con.execute("INSERT INTO movies SELECT rid, item_id, facet, value, filter_num, filter_bool, "
                    "emb::FLOAT[384] FROM chunk")
        con.unregister("chunk")
    con.execute("CHECKPOINT")
    n, nf, adm = con.execute("SELECT count(*), count(DISTINCT facet), count(*) FILTER (WHERE filter_num >= 2000) "
                             "FROM movies").fetchone()
    # round-trip check: DuckDB's stored floats equal emb.npy bit for bit
    probe = [0, 1, len(df) // 2, len(df) - 1] + [int(i) for i in qidx[:5]]
    for r in probe:
        got = np.array(con.execute("SELECT emb FROM movies WHERE rid = ?", [r]).fetchone()[0], dtype=np.float32)
        assert np.array_equal(got, emb[r]), r
    con.close()
    stats = {"rows": n, "facets": nf, "admitted_filter_num_ge_2000": adm,
             "db_bytes": os.path.getsize(DB), "build_seconds": round(time.time() - t0, 1),
             "duckdb": duckdb.__version__, "emb_roundtrip_rows_checked": len(probe)}
    with open(os.path.join(DATA, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    sys.exit(main())

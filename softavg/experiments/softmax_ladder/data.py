#!/usr/bin/env python
"""Build the ladder's row sets once (scratch; deleted after the run).

  imdb_all : IMDb corpus, all 459,865 rows, ORDER BY id (= corpus row index)
  imdb_ge  : the PROTOCOL predicate subset filter_num >= 2000 (232,828 rows), ORDER BY id
  amazon   : Amazon reviews with a verified review timestamp
             (../timeorder/amazon_ts.parquet, 1,524,113 rows), ORDER BY ts, id

Scores s0, s1, s2: float32 inner product (numpy) of the row embedding with the
first three PROTOCOL queries, stored as float64 (precomputed, so the error
measured is the aggregation's). IMDb queries: ../postgres/queries.json (rows
0..2). Amazon queries: the sampler of ../timeorder/load_amazon_ts.py
(RandomState(0), first sentence 4-20 words, deduplicated, 200 sampled; first 3).

Writes $LADDER_WORK/<name>.npz (id, facet code, value, s0..s2[, ts_ms]) and
$LADDER_WORK/ladder.sqlite (tables imdb_all, imdb_ge, amazon with
rn INTEGER PRIMARY KEY = position in the window order) and data_meta.json here.

usage: data.py
"""
import json, os, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ["LADDER_WORK"])
WORK.mkdir(parents=True, exist_ok=True)
CORP = Path("/data/softavg/corpora")
NQ = 3


def scores(emb_path, rows_q, n):
    emb = np.load(emb_path, mmap_mode="r")
    Q = np.asarray(emb[np.asarray(rows_q)], dtype=np.float32)
    S = np.empty((n, len(rows_q)), dtype=np.float32)
    for a in range(0, n, 200_000):
        S[a:a + 200_000] = np.asarray(emb[a:a + 200_000], dtype=np.float32) @ Q.T
    return S.astype(np.float64)


def main():
    meta = {}
    # ---------------- IMDb
    df = pd.read_parquet(CORP / "imdb/corpus.parquet", columns=["facet", "value", "filter_num"])
    qs = json.loads((HERE.parent / "postgres/queries.json").read_text())["queries"][:NQ]
    S = scores(CORP / "imdb/emb.npy", [q["row"] for q in qs], len(df))
    fac, fcodes = pd.factorize(df.facet)
    base = dict(id=np.arange(len(df)), facet=fac.astype(np.int32), value=df.value.values.astype(np.float64),
                **{f"s{j}": S[:, j] for j in range(NQ)})
    np.savez(WORK / "imdb_all.npz", **base, facet_names=np.array(fcodes, dtype=object))
    ge = df.filter_num.values >= 2000
    np.savez(WORK / "imdb_ge.npz", **{k: v[ge] for k, v in base.items()}, facet_names=np.array(fcodes, dtype=object))
    meta["imdb"] = dict(rows=int(len(df)), rows_ge2000=int(ge.sum()), facets=int(len(fcodes)),
                        queries=[dict(row=q["row"], text=q["text"]) for q in qs], order="id",
                        score="float32 inner product (numpy) of corpus emb.npy rows, stored as float64",
                        s_max=[float(S[:, j].max()) for j in range(NQ)])
    # ---------------- Amazon (sampler copied from ../timeorder/load_amazon_ts.py)
    df = pd.read_parquet(CORP / "amazon/corpus.parquet", columns=["item_id", "facet", "value", "text"])
    rng = np.random.RandomState(0)
    first = df.text.str.split(".").str[0].str.strip()
    ok = first.str.split().str.len().between(4, 20) & ~first.duplicated()
    qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)[:NQ]
    S = scores(CORP / "amazon/emb.npy", qidx, len(df))
    ts = pd.read_parquet(HERE.parent / "timeorder/amazon_ts.parquet")
    n_ts = len(ts); ts = ts[ts.ok]
    row = pd.Series(np.arange(len(df)), index=df.item_id)
    ids = row.reindex(ts.item_id).values.astype(np.int64)
    tms = (ts.ts.values.astype("datetime64[ms]").astype(np.int64))
    order = np.lexsort((ids, tms))  # ORDER BY ts, id
    ids, tms = ids[order], tms[order]
    fac, fcodes = pd.factorize(df.facet)
    np.savez(WORK / "amazon.npz", id=ids, facet=fac[ids].astype(np.int32), value=df.value.values[ids].astype(np.float64),
             ts_ms=tms, **{f"s{j}": S[ids, j] for j in range(NQ)}, facet_names=np.array(fcodes, dtype=object))
    meta["amazon"] = dict(rows=int(len(ids)), rows_in_ts_file=int(n_ts), facets=int(len(fcodes)),
                          queries=[dict(row=int(q), text=str(first.values[q])) for q in qidx], order="ts, id",
                          ts_min=str(pd.Timestamp(tms.min(), unit="ms")), ts_max=str(pd.Timestamp(tms.max(), unit="ms")),
                          distinct_ts=int(len(np.unique(tms))),
                          score="float32 inner product (numpy) of corpus emb.npy rows, stored as float64",
                          s_max=[float(S[ids, j].max()) for j in range(NQ)])
    # ---------------- SQLite tables
    db = WORK / "ladder.sqlite"
    if db.exists():
        db.unlink()
    c = sqlite3.connect(db)
    c.execute("PRAGMA journal_mode=OFF"); c.execute("PRAGMA synchronous=OFF")
    for name in ("imdb_all", "imdb_ge", "amazon"):
        z = np.load(WORK / f"{name}.npz", allow_pickle=True)
        tcol = ", ts INTEGER" if name == "amazon" else ""
        c.execute(f"CREATE TABLE {name}(rn INTEGER PRIMARY KEY, id INTEGER, value REAL, s0 REAL, s1 REAL, s2 REAL{tcol})")
        cols = [np.arange(len(z["id"])), z["id"], z["value"], z["s0"], z["s1"], z["s2"]] + ([z["ts_ms"]] if name == "amazon" else [])
        ph = ",".join("?" * len(cols))
        c.executemany(f"INSERT INTO {name} VALUES ({ph})",
                      zip(*[a.tolist() for a in cols]))
        c.commit()
    c.execute("VACUUM"); c.close()
    (HERE / "data_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Load Amazon rows that have a verified review timestamp into the release server.

  amz_ts(id int PRIMARY KEY -- row index in amazon corpus.parquet,
         facet text, value float8, s0 float8, ts timestamp(3))

s0 = float32 inner product of the row's embedding with query 0 of the PROTOCOL
sampler applied to the amazon corpus (RandomState(0); first sentence 4-20 words,
deduplicated; 200 sampled; query = that item's embedding), stored as float8.
Only rows whose recovered timestamp passed the rating/year check (amazon_ts.ok).

usage: load_amazon_ts.py   -> load_amazon_ts.json
"""
import io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import connect  # noqa: E402

CORP = Path("/data/softavg/corpora/amazon")
df = pd.read_parquet(CORP / "corpus.parquet", columns=["item_id", "facet", "value", "text"])
rng = np.random.RandomState(0)
first = df.text.str.split(".").str[0].str.strip()
ok = first.str.split().str.len().between(4, 20) & ~first.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)
emb = np.load(CORP / "emb.npy", mmap_mode="r")
q = np.asarray(emb[qidx[0]], dtype=np.float32)
s0 = np.empty(len(df), dtype=np.float32)
for a in range(0, len(df), 200_000):
    s0[a:a + 200_000] = np.asarray(emb[a:a + 200_000], dtype=np.float32) @ q
ts = pd.read_parquet(HERE / "amazon_ts.parquet")
ts = ts[ts.ok]
row = pd.Series(np.arange(len(df)), index=df.item_id)
ts["id"] = row.reindex(ts.item_id).values
t = pd.DataFrame(dict(id=ts.id.astype(int), facet=df.facet.values[ts.id], value=df.value.values[ts.id],
                      s0=s0[ts.id].astype(np.float64), ts=ts.ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:23]))
cur = connect("rel").cursor()
cur.execute("DROP TABLE IF EXISTS amz_ts")
cur.execute("CREATE TABLE amz_ts(id int PRIMARY KEY, facet text, value float8, s0 float8, ts timestamp(3) NOT NULL)")
buf = io.StringIO(); t.to_csv(buf, sep="\t", header=False, index=False, float_format="%.17g"); buf.seek(0)
cur.copy_expert("COPY amz_ts FROM STDIN", buf)
cur.execute("CREATE INDEX ON amz_ts(ts, id)")
cur.execute("VACUUM ANALYZE amz_ts")
cur.execute("SELECT count(*), count(DISTINCT facet), count(DISTINCT ts), min(ts)::text, max(ts)::text FROM amz_ts")
st = dict(zip(["rows", "facets", "distinct_ts", "ts_min", "ts_max"], cur.fetchone()))
st.update(query_row=int(qidx[0]), query_text=str(first.values[qidx[0]]), s0_max=float(t.s0.max()), s0_min=float(t.s0.min()),
          score="float32 inner product (numpy), stored as float8")
(HERE / "load_amazon_ts.json").write_text(json.dumps(st, indent=1))
print(json.dumps(st, indent=1))

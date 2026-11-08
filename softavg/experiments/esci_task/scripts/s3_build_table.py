"""Step 2: build the task table and embed it.

Inputs: data/esci_us.parquet, data/meta_join.parquet
Output: data/task.parquet  one row per (query_id, product_id) in a pool:
          query_id, product_id, label (E/S/C/I, or 'D' for a distractor),
          brand, main_category, rating, rating_number, price, s (cosine)
        data/queries.parquet  query_id, query, split ('dev'/'test')

Pools
  For every query with >= MIN_LABELLED labelled products that have a value
  (rating) and at least one E product, the pool is all its labelled
  products.  Distractors (label 'D'): N_DISTRACT products drawn at random
  from the products labelled for OTHER queries whose dominant
  main_category equals this query's dominant main_category (in-category,
  so they are plausible but were judged for a different query).  A
  distractor that is also labelled for this query is skipped.

Scores: s = cosine(query, product_title) with all-MiniLM-L6-v2,
normalised embeddings, computed on the GPU in float32.
"""
import argparse, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"
ap = argparse.ArgumentParser()
ap.add_argument("--min-labelled", type=int, default=12)
ap.add_argument("--n-distract", type=int, default=200)
ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=128)
a = ap.parse_args()

e = pd.read_parquet(D / "esci_us.parquet")
m = pd.read_parquet(D / "meta_join.parquet")
# a parent_asin can occur in two category files; keep the record with most ratings
m = (m.sort_values("rating_number", ascending=False)
      .drop_duplicates("product_id"))
df = e.merge(m, on="product_id", how="inner")
df = df[df.average_rating.notna()]
df["brand"] = df.product_brand.fillna("").str.strip()
df.loc[df.brand == "", "brand"] = None
g = df.groupby("query_id")
ok = (g.size() >= a.min_labelled) & g.esci_label.apply(lambda x: (x == "E").any())
qids = ok[ok].index.values
rng = np.random.RandomState(a.seed)
if a.max_queries and len(qids) > a.max_queries:
    qids = np.sort(rng.choice(qids, a.max_queries, replace=False))
df = df[df.query_id.isin(qids)].copy()
print(f"queries kept {len(qids):,}; labelled rows {len(df):,}")

# distractors: in-category products labelled for other queries
prod = (pd.read_parquet(D / "esci_us.parquet", columns=["product_id", "product_title",
                                                       "product_brand"])
          .drop_duplicates("product_id").merge(m, on="product_id"))
prod = prod[prod.average_rating.notna()].reset_index(drop=True)
by_cat = {c: grp.index.values for c, grp in prod.groupby("main_category")}
dom = df.groupby("query_id").main_category.agg(
    lambda x: x.value_counts().index[0] if x.notna().any() else None)
# all products labelled for q (with or without a value) are excluded
labelled_for = e.groupby("query_id").product_id.agg(set)
pid = prod.product_id.values
rows = []
for qid in qids:
    pool = by_cat.get(dom[qid])
    if pool is None or a.n_distract == 0:
        continue
    have = labelled_for[qid]
    cand = rng.choice(pool, min(len(pool), a.n_distract + len(have)), replace=False)
    cand = [i for i in cand if pid[i] not in have][:a.n_distract]
    rows.append(pd.DataFrame({"row": cand, "query_id": qid}))
if rows:
    r = pd.concat(rows, ignore_index=True)
    dis = prod.loc[r.row.values].reset_index(drop=True)
    dis["query_id"] = r.query_id.values
    dis["esci_label"] = "D"
else:
    dis = pd.DataFrame()
if len(dis):
    dis["brand"] = dis.product_brand.fillna("").str.strip()
    dis.loc[dis.brand == "", "brand"] = None
cols = ["query_id", "product_id", "esci_label", "product_title", "brand",
        "main_category", "average_rating", "rating_number", "price"]
t = pd.concat([df[cols], dis[cols]] if len(dis) else [df[cols]], ignore_index=True)
t = t.rename(columns={"esci_label": "label", "average_rating": "rating"})
qs = df.drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)
# fixed dev/test split by a hash of query_id (no peeking at labels)
qs["split"] = ["dev" if int(hashlib.md5(str(q).encode()).hexdigest(), 16) % 2 == 0
               else "test" for q in qs.query_id]

import torch
from sentence_transformers import SentenceTransformer
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"GPU free {free:.1f} GiB")
dev = "cuda" if free > 1.2 else "cpu"
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
t0 = time.time()
qemb = enc.encode(qs["query"].tolist(), batch_size=a.batch, convert_to_numpy=True,
                  normalize_embeddings=True, show_progress_bar=False)
titles = t.drop_duplicates("product_id")[["product_id", "product_title"]].reset_index(drop=True)
pemb = enc.encode(titles.product_title.fillna("").tolist(), batch_size=a.batch,
                  convert_to_numpy=True, normalize_embeddings=True,
                  show_progress_bar=False)
print(f"embedded {len(qs):,} queries + {len(titles):,} titles in {time.time()-t0:.0f}s on {dev}")
qi = pd.Series(np.arange(len(qs)), index=qs.query_id)
pi = pd.Series(np.arange(len(titles)), index=titles.product_id)
a_q = qemb[qi[t.query_id].values]
a_p = pemb[pi[t.product_id].values]
t["s"] = np.einsum("ij,ij->i", a_q, a_p).astype(np.float64)
t = t.drop(columns=["product_title"])
t.to_parquet(D / "task.parquet", index=False)
qs.to_parquet(D / "queries.parquet", index=False)
print(t.groupby("label").s.describe().to_string())
print(f"wrote task.parquet {len(t):,} rows, {(D/'task.parquet').stat().st_size/1e6:.1f} MB")

"""Derive the per-corpus inputs of the bound experiment (PROTOCOL.md).

For IMDb, Amazon and StackExchange: the admitted rows (predicate of
PROTOCOL.md) in item (storage) order, their value, a facet code, and the
scores s = float64(E) @ q for the first N queries of the PROTOCOL draw
(RandomState(0), first sentence of 4-20 words, de-duplicated, 200 drawn).
Writes work/<corpus>/{v.f64, facet.u32, scores.f64, meta.json};
work/ is derived data and is deleted after the run (run.py).
"""
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # shared machine
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORPORA = Path("/data/softavg/corpora")
THR = {"imdb": 2000.0, "amazon": 2015.0, "stackexchange": 3.0}


def queries(df, emb, n):
    rng = np.random.RandomState(0)
    titles = df.text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)
    return qidx[:n], np.asarray(emb[qidx[:n]], dtype=np.float64)


def prep(corpus, n_queries):
    out = HERE / "work" / corpus
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CORPORA / corpus / "corpus.parquet", columns=["facet", "value", "filter_num", "text"])
    emb = np.load(CORPORA / corpus / "emb.npy", mmap_mode="r")
    assert emb.shape[0] == len(df)
    qidx, Q = queries(df, emb, n_queries)
    adm = np.flatnonzero(df.filter_num.values >= THR[corpus])
    codes, uniq = pd.factorize(df.facet.values[adm], sort=True)
    df.value.values[adm].astype("<f8").tofile(out / "v.f64")
    codes.astype("<u4").tofile(out / "facet.u32")
    S = np.empty((n_queries, len(adm)), dtype=np.float64)
    step = 200_000
    for a in range(0, len(adm), step):
        E = np.asarray(emb[adm[a:a + step]], dtype=np.float64)
        S[:, a:a + step] = (E @ Q.T).T
    S.astype("<f8").tofile(out / "scores.f64")
    meta = dict(corpus=corpus, predicate=f"filter_num >= {THR[corpus]:g}", n_rows_total=len(df),
                n_admitted=int(len(adm)), n_facets=int(len(uniq)), query_rows=[int(i) for i in qidx],
                score_range=[float(S.min()), float(S.max())],
                value_range=[float(df.value.values[adm].min()), float(df.value.values[adm].max())])
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(corpus, meta["n_admitted"], "rows,", meta["n_facets"], "facets, scores", meta["score_range"], flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for c in sys.argv[2:] or ["imdb", "amazon", "stackexchange"]:
        prep(c, n)

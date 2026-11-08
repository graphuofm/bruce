#!/usr/bin/env python
"""PROTOCOL query workload for IMDb: rng = RandomState(0), first sentence of
each item's text, 4-20 words, deduplicated, 200 sampled; the query vectors are
those items' rows of emb.npy. Writes queries.json (row indices + texts only;
embeddings are re-read from the read-only corpus). Same draw as experiments/PROTOCOL.md."""
import json
import numpy as np
import pandas as pd
from common import CORP, HERE

df = pd.read_parquet(CORP / "corpus.parquet")
rng = np.random.RandomState(0)
titles = df.text.str.split(".").str[0].str.strip()
ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)
out = dict(corpus="imdb", n_rows=len(df), rule="RandomState(0); first sentence 4-20 words, deduped; 200 sampled",
           queries=[dict(row=int(i), text=str(titles.values[i])) for i in qidx])
(HERE / "queries.json").write_text(json.dumps(out, indent=1))
print(len(qidx), "queries; first:", out["queries"][0])

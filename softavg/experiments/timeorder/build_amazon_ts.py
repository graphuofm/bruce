"""Run 2026-11-06 after the author approved the download (1,524,113 of 1,524,113 rows verified).

Recover the review timestamp of every amazon corpus row. build_amazon.py read
each category file of McAuley-Lab/Amazon-Reviews-2023 (raw/review_categories/
<cat>.jsonl) and kept every 5th line of a bounded prefix, with item_id =
"<cat>:<line number>"; the review's `timestamp` (ms since epoch) was read but
only its year was kept (filter_num). The line number in item_id is enough to
find the row again: stream each category's prefix up to the largest line
number the corpus holds, and take `timestamp` at those lines. Each recovered
row is checked against the corpus (rating == value, year == filter_num) so a
changed upstream file is detected rather than silently mis-joined.

Download size: the prefixes read are those of meta.json's per_category
`lines_read` (<= 250k lines per category, 33 categories, ~6.2M lines; a few
GB of JSON over HTTP). Output: amazon_ts.parquet (item_id, ts, ok).

usage: build_amazon_ts.py
"""
import json, time
from pathlib import Path
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CORP = Path("/data/softavg/corpora/amazon/corpus.parquet")
BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories"

df = pd.read_parquet(CORP, columns=["item_id", "facet", "value", "filter_num"])
df["line"] = df.item_id.str.rsplit(":", n=1).str[1].astype(int)
out = []
for cat, g in df.groupby("facet"):
    want = dict(zip(g.line, zip(g.item_id, g.value, g.filter_num)))
    last = max(want); seen = 0; t0 = time.time()
    with requests.get(f"{BASE}/{cat}.jsonl", stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1 << 20):
            if not line:
                continue
            seen += 1
            if seen in want:
                d = json.loads(line); iid, v, y = want[seen]
                ts = d.get("timestamp")
                ok = (ts is not None and float(d.get("rating")) == v
                      and time.gmtime(ts / 1000).tm_year == int(y))
                out.append((iid, pd.Timestamp(ts, unit="ms") if ts is not None else pd.NaT, ok))
            if seen >= last:
                break
    print(cat, len(want), f"{time.time() - t0:.0f}s", flush=True)
res = pd.DataFrame(out, columns=["item_id", "ts", "ok"])
res.to_parquet(HERE / "amazon_ts.parquet", index=False)
print(len(res), "rows,", int(res.ok.sum()), "verified against the corpus")

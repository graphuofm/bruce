#!/usr/bin/env python
"""Bitwise cross-check: SQLite engine output of S3 / S4 vs the PG call-order replay (ladder_replay)
on a few cells (IMDb predicate subset and Amazon, query 1). -> crosscheck.json"""
import json, struct, subprocess
import numpy as np
from common import HERE, REPLAY_BIN, WORK, load
from sqlite_window import connect, fetch, sql_for

out = []
c = connect()
for ds, cells in (("imdb_ge", [(10, 0.02), (100, 1e-3), (1000, 0.1), (10, 1e-4)]), ("amazon", [(100, 0.02), (1000, 1e-3)])):
    d = load(ds); n = len(d["value"])
    (WORK / "x.bin").write_bytes(struct.pack("<Q", n) + d["s1"].astype("<f8").tobytes() + d["value"].astype("<f8").tobytes())
    for k, e in cells:
        subprocess.run([str(REPLAY_BIN), "window", str(WORK / "x.bin"), str(k), repr(e), str(WORK / "y.bin")], check=True, capture_output=True)
        o = np.fromfile(WORK / "y.bin").reshape(3, n)
        for i, r in ((1, "S3"), (2, "S4")):
            g, _ = fetch(c, sql_for(r, ds, k, 1, e))
            same = (g.view(np.uint64) == o[i].view(np.uint64)) | (np.isnan(g) & np.isnan(o[i]))
            out.append(dict(dataset=ds, q=1, k=k, eps=e, rung=r, rows=n, bitwise_equal_rows=int(same.sum())))
            print(out[-1], flush=True)
for p in ("x.bin", "y.bin"):
    (WORK / p).unlink(missing_ok=True)
(HERE / "crosscheck.json").write_text(json.dumps(out, indent=1))

"""Window drift under the core's running error bound, on IMDb.

For the first n PROTOCOL queries: admitted rows in id order, s = float64(E) @ q,
v = rating; engines/sqlite/examples/guard_bound slides a 50-row frame with
softagg-core's InvAcc, refolding exactly when the core marks the state
inexact, and reports the worst error against a fresh fold per frame, plus
whether the error ever exceeded the core's own value_error_bound().
Writes guard_bound.json. The previous core (z-only guard, TAU = 1e-6; host-
side tau sweep) is kept in guard_tau_old_core.json for comparison.
Pure single-threaded CPU; no timings recorded.
"""
import json
import subprocess
import sys

import numpy as np

import common

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
frame = 50
exe = common.ENG / "target" / "release" / "examples" / "guard_bound"
df, emb = common.load_corpus()
adm = df.filter_num.values >= common.MIN_NUM
_, Q = common.queries(df, emb, n)
E = emb[adm].astype(np.float64)
v = df.value.values[adm].astype(np.float64)
common.WORK.mkdir(exist_ok=True)
out = {"what": "max error of a 50-row sliding InvAcc (core running error bound) vs fresh fold, by eps",
       "core_guard": "running first-order bound on z and u; inexact when bound > max(1e-10, 100 n U) * v_scale",
       "rows": int(adm.sum()), "frame": frame, "queries": n, "cells": []}
for k in range(n):
    path = common.WORK / f"sv{k}.bin"
    np.stack([E @ Q[k].astype(np.float64), v], 1).astype("<f8").tofile(path)
    res = subprocess.run([str(exe), str(path), str(frame)], capture_output=True, text=True, check=True)
    path.unlink()
    for line in res.stdout.splitlines():
        d = json.loads(line)
        d["query"] = k
        out["cells"].append(d)
    print(f"query {k + 1}/{n}", flush=True)
json.dump(out, open(common.HERE / "guard_bound.json", "w"), indent=1)
print("wrote guard_bound.json")

#!/usr/bin/env python
"""Interleaved latency of S2 / S3 / S4 in SQLite (the certificate's cost), query 0.

The main grid times each rung's 3 runs back to back; on a loaded shared machine that lets a load
spike land on one rung. Here every repetition runs S3, S4, S2 in turn (rotating the start), 7
repetitions, and reports the per-rung median and the median of the per-repetition ratios S4/S3
and S2/S4.  -> sqlite_timing.jsonl
"""
import json
import numpy as np
from common import HERE, Timer, machine_state
from sqlite_window import TIMED, connect, sql_for

REPS = 7


def main():
    c = connect()
    with open(HERE / "sqlite_timing.jsonl", "w") as f:
        f.write(json.dumps(dict(header=True, machine_at_start=machine_state(), reps=REPS)) + "\n")
        for ds in ("imdb_all", "imdb_ge", "amazon"):
            for k in (10, 100, 1000):
                for eps in (1e-3, 0.02, 1.0):
                    rungs = ["S3", "S4", "S2"]
                    sql = {r: TIMED.format(sql_for(r, ds, k, 0, eps)) for r in rungs}
                    for r in rungs:
                        c.execute(sql[r]).fetchall()  # warm
                    t = {r: [] for r in rungs}
                    for i in range(REPS):
                        for r in rungs[i % 3:] + rungs[:i % 3]:
                            with Timer() as tm:
                                c.execute(sql[r]).fetchall()
                            t[r].append(tm.ms)
                    a = {r: np.array(v) for r, v in t.items()}
                    cell = dict(dataset=ds, k=k, eps=eps, q=0, ms=t, median_ms={r: float(np.median(v)) for r, v in a.items()},
                                ratio_S4_S3=float(np.median(a["S4"] / a["S3"])), ratio_S2_S4=float(np.median(a["S2"] / a["S4"])),
                                load_average=machine_state()["load_average"])
                    f.write(json.dumps(cell) + "\n"); f.flush()
                    print(ds, k, eps, cell["median_ms"], round(cell["ratio_S4_S3"], 3), round(cell["ratio_S2_S4"], 2), flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")


if __name__ == "__main__":
    main()

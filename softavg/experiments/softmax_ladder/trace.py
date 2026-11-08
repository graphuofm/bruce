#!/usr/bin/env python
"""Per-row trace for a figure: Amazon, query 0, ORDER BY ts, id, ROWS k PRECEDING, k in {50, 1000},
eps in {0.02, 1e-3}. Every rung that returns values:
  SQLite (engine run): S0 naive, S1 global shift, S3 plain inverse, S4 shipped softavg
  PostgreSQL (CALL-ORDER REPLAY, not an engine run; README "Not run"): S3, S4
plus the numpy float64 reference and max|v| of the frame, over one contiguous slice of 20,000 rows
chosen to contain the worst SQLite S3 error (and the worst S1 error when both fit); rows sampled
every 10th, plus the 50 worst rows of every rung inside the slice; the worst row of every rung over
the whole run is recorded separately. NULL / NaN / Inf -> null + a flag list.  -> trace.json
"""
import json, struct, subprocess
import numpy as np
from common import HERE, REPLAY_BIN, WORK, load, reference, vmax_frames
from sqlite_window import connect, fetch, sql_for

SLICE = 20_000


def j(x):
    return [None if not np.isfinite(t) else float(t) for t in x]


def main():
    d = load("amazon"); n = len(d["value"]); s = d["s0"]; v = d["value"]; ts = d["ts_ms"]
    c = connect()
    inb, outb = WORK / "trace_in.bin", WORK / "trace_out.bin"
    with open(inb, "wb") as g:
        g.write(struct.pack("<Q", n)); g.write(s.astype("<f8").tobytes()); g.write(v.astype("<f8").tobytes())
    out = dict(dataset="amazon", query=0, order="ts, id", rows=n, slice_len=SLICE, sample_every=10,
               note="pg_* series are a replay of PostgreSQL's moving-aggregate call order outside the server "
                    "(no PostgreSQL server was available); sqlite_* series are SQLite 3.37.2 engine runs",
               err_unit="abs err / max|v| of the frame", traces=[])
    for k in (50, 1000):
        vmax = vmax_frames(v, k)
        for eps in (0.02, 1e-3):
            ref = reference(v, s, eps, k)
            series = {}
            for rung in ("S0", "S1", "S3", "S4"):
                got, err = fetch(c, sql_for(rung, "amazon", k, 0, eps))
                series[f"sqlite_{rung}"] = got if got is not None else np.full(n, np.nan)
            subprocess.run([str(REPLAY_BIN), "window", str(inb), str(k), repr(eps), str(outb)], check=True, capture_output=True)
            o = np.fromfile(outb, dtype="<f8").reshape(3, n)
            series["pg_replay_S3"], series["pg_replay_S4"] = o[1], o[2]
            errs = {name: np.abs(x - ref) / vmax for name, x in series.items()}
            bad = {name: ~np.isfinite(x) for name, x in series.items()}

            def worst(name):
                e = np.where(bad[name], np.inf, errs[name])
                return int(np.argmax(e))
            w3, w1 = worst("sqlite_S3"), worst("sqlite_S1")
            if abs(w3 - w1) < SLICE - 2000:
                center = (w3 + w1) // 2
            else:
                center = w3
            a = int(min(max(0, center - SLICE // 2), n - SLICE)); b = a + SLICE
            keep = set(range(a, b, 10))
            for name in series:
                e = np.where(bad[name][a:b], np.inf, np.nan_to_num(errs[name][a:b], nan=np.inf))
                keep |= set((a + np.argsort(-e)[:50]).tolist())
            idx = np.array(sorted(keep))
            tr = dict(k=k, eps=eps, slice=[a, b], center_reason=("worst S3 and worst S1 are the same row" if w1 == w3 else
                                     "between worst S3 and worst S1" if center != w3 else "worst S3 (worst S1 outside)"),
                      row=idx.tolist(), ts_ms=ts[idx].tolist(), reference=j(ref[idx]), vmax=j(vmax[idx]),
                      series={name: j(x[idx]) for name, x in series.items()},
                      nonfinite={name: np.flatnonzero(bad[name][idx]).tolist() for name in series},
                      worst_overall={name: dict(row=worst(name), ts_ms=int(ts[worst(name)]),
                                                value=(None if bad[name][worst(name)] else float(series[name][worst(name)])),
                                                reference=float(ref[worst(name)]),
                                                err_over_vmax=(None if bad[name][worst(name)] else float(errs[name][worst(name)])),
                                                in_slice=bool(a <= worst(name) < b),
                                                nonfinite_rows_total=int(bad[name].sum()))
                                     for name in series})
            out["traces"].append(tr)
            print(k, eps, tr["slice"], {x: (y["row"], y["err_over_vmax"]) for x, y in tr["worst_overall"].items()}, flush=True)
    (HERE / "trace.json").write_text(json.dumps(out, separators=(",", ":")))
    for p in (inb, outb):
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Sliding-window softavg over REAL time order (IMDb release dates) vs id order.

Row set: IMDb rows whose earliest release date has day precision and falls in
[T0, T1) (imdb_ts.parquet), score s0 (precomputed, query 0), table w_ts(id, ts, value, s0).
Frames, all over the SAME rows:
  id   ROWS k : ORDER BY id      ROWS BETWEEN k PRECEDING AND CURRENT ROW   (today's setup)
  ts   ROWS k : ORDER BY ts, id  ROWS BETWEEN k PRECEDING AND CURRENT ROW
  ts   RANGE T: ORDER BY ts      RANGE BETWEEN INTERVAL T PRECEDING AND CURRENT ROW (peers included)
Plans: softavg_moving (MSFUNC/MINVFUNC; punts = guard firings -> PG recomputes the frame),
softavg_fwd (no moving form: every frame re-aggregated), naive_window (SUM(EXP()) OVER w).
Correctness vs a numpy float64 per-frame max-shifted reference; PROTOCOL window rule
|err| <= 1e-9 * max|v| of the frame.

CORPUS=amazon (environment): the same over Amazon reviews with their recovered
review timestamps (table amz_ts from load_amazon_ts.py; ms precision); RANGE
frames are then RANGE_SECONDS (1 hour, 1 day, 7 days) -> window_ts_amazon.jsonl.

usage: [CORPUS=amazon] window_ts.py [T0] [T1]   -> window_ts.jsonl | window_ts_amazon.jsonl
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import EPS_GRID, TIMING_STATUS, connect, eps_key, eps_sql, machine_state, run_timed  # noqa: E402

T0 = sys.argv[1] if len(sys.argv) > 1 else "2005-01-01"
T1 = sys.argv[2] if len(sys.argv) > 2 else "2011-01-01"
CORPUS = os.environ.get("CORPUS", "imdb")
OUT = HERE / ("window_ts.jsonl" if CORPUS == "imdb" else f"window_ts_{CORPUS}.jsonl")
RANGE_SECONDS = [3600, 86400, 7 * 86400]
FRAMES = [("id", "rows", 10), ("id", "rows", 100), ("id", "rows", 1000),
          ("ts", "rows", 10), ("ts", "rows", 100), ("ts", "rows", 1000),
          ("ts", "range", 1), ("ts", "range", 7), ("ts", "range", 30)]
PLANS = {"softavg_moving": "softavg(value, s0, {E}) OVER w",
         "softavg_fwd": "softavg_fwd(value, s0, {E}) OVER w",
         "naive_window": "SUM(EXP(s0/{E})*value) OVER w / SUM(EXP(s0/{E})) OVER w"}


def window_clause(order, kind, k):
    if kind == "rows":
        ob = "id" if order == "id" else "ts, id"
        return f"ORDER BY {ob} ROWS BETWEEN {k} PRECEDING AND CURRENT ROW"
    if CORPUS == "imdb":
        return f"ORDER BY ts RANGE BETWEEN INTERVAL '{k} days' PRECEDING AND CURRENT ROW"
    return f"ORDER BY ts RANGE BETWEEN INTERVAL '{k} seconds' PRECEDING AND CURRENT ROW"


def frames_for(order, kind, k, ts_ns):
    n = len(ts_ns)
    end = np.arange(1, n + 1)
    if kind == "rows":
        return np.maximum(0, end - 1 - k), end
    sec = k * 86400 if CORPUS == "imdb" else k
    start = np.searchsorted(ts_ns, ts_ns - sec * 10**9, "left")
    end = np.searchsorted(ts_ns, ts_ns, "right")
    return start, end


def reference(v, s, eps, start, end, chunk=256):
    out = np.empty(len(v)); vmax = np.empty(len(v))
    W = int((end - start).max())
    for a in range(0, len(v), chunk):
        b = min(len(v), a + chunk)
        idx = start[a:b, None] + np.arange(W)[None, :]
        live = idx < end[a:b, None]
        idx = np.minimum(idx, len(v) - 1)
        S = np.where(live, s[idx], -np.inf); V = np.where(live, v[idx], 0.0)
        vmax[a:b] = np.where(live, np.abs(V), 0).max(1)
        if eps == "inf":
            out[a:b] = V.sum(1) / live.sum(1)
        else:
            m = S.max(1, keepdims=True)
            w = np.where(live, np.exp((S - m) / float(eps)), 0.0)
            out[a:b] = (w * V).sum(1) / w.sum(1)
    return out, vmax


def main():
    conn = connect("rel"); cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS w_ts_ids, w_ts")
    if CORPUS == "imdb":
        ts = pd.read_parquet(HERE / "imdb_ts.parquet")
        ts = ts[(ts.precision == "day") & (ts.ts >= T0) & (ts.ts < T1)]
        cur.execute("CREATE TABLE w_ts_ids(id int PRIMARY KEY, ts timestamp NOT NULL)")
        cur.executemany("INSERT INTO w_ts_ids VALUES (%s, %s)", list(zip(ts.row.astype(int).tolist(), ts.ts.dt.to_pydatetime().tolist())))
        cur.execute("CREATE TABLE w_ts AS SELECT i.id, t.ts, i.value, i.s0 FROM imdb i JOIN w_ts_ids t USING (id)")
    else:
        cur.execute("CREATE TABLE w_ts AS SELECT id, ts, value, s0 FROM amz_ts WHERE ts >= %s AND ts < %s", (T0, T1))
    cur.execute("CREATE INDEX ON w_ts(id); CREATE INDEX ON w_ts(ts, id)"); cur.execute("VACUUM ANALYZE w_ts")
    cur.execute("SET max_parallel_workers_per_gather = 0")
    data = {}
    for order in ("id", "ts"):
        cur.execute(f"SELECT id, value, s0, round(extract(epoch from ts) * 1000)::bigint FROM w_ts ORDER BY {'id' if order == 'id' else 'ts, id'}")
        r = cur.fetchall()
        data[order] = (np.array([x[1] for x in r]), np.array([x[2] for x in r]),
                       np.array([x[3] for x in r], dtype=np.int64) * 10**6)
    n = len(data["id"][0])
    with open(OUT, "w") as f:
        f.write(json.dumps(dict(header=True, corpus=CORPUS, rows=n, t0=T0, t1=T1,
                                ts_precision="day" if CORPUS == "imdb" else "ms", score="s0 (query 0)",
                                range_unit="days" if CORPUS == "imdb" else "seconds",
                                machine_at_start=machine_state(), timing_status=TIMING_STATUS)) + "\n")
        frames = FRAMES if CORPUS == "imdb" else [x for x in FRAMES if x[1] == "rows"] + [("ts", "range", s) for s in RANGE_SECONDS]
        for order, kind, k in frames:
            v, s, tns = data[order]
            start, end = frames_for(order, kind, k, tns)
            width = end - start
            wst = dict(mean=float(width.mean()), median=float(np.median(width)), max=int(width.max()))
            for e in EPS_GRID:
                ref, vmax = reference(v, s, e, start, end)
                for plan, expr in PLANS.items():
                    q = (f"SELECT id, ts, {expr.replace('{E}', eps_sql(e))} AS x FROM w_ts "
                         f"WINDOW w AS ({window_clause(order, kind, k)})")
                    cell = dict(order=order, frame=kind, k=k, eps=eps_key(e), plan=plan, rows=n, frame_width=wst)
                    moving = plan == "softavg_moving"
                    t = run_timed(cur, f"SELECT count(*), sum(x) FROM ({q}) t", n_warm=1 if moving else 0,
                                  n_runs=3 if moving else 1)
                    t.pop("rows", None); cell.update(t)
                    if t["sql_ok"]:
                        cur.execute("SELECT softavg_window_stats_reset()")
                        cur.execute(q + f" ORDER BY {'id' if order == 'id' else 'ts, id'}")
                        got = np.array([x[2] if x[2] is not None else np.nan for x in cur.fetchall()], dtype=np.float64)
                        cur.execute("SELECT * FROM softavg_window_stats()")
                        fr, n_abs, inv, pu = cur.fetchone()
                        ok = np.isfinite(got)
                        ab = np.abs(got[ok] - ref[ok])
                        cell.update(n_nonfinite=int((~ok).sum()),
                                    max_rel_err=float((ab / np.abs(ref[ok])).max()) if ok.any() else None,
                                    max_abs_err_over_vmax=float((ab / vmax[ok]).max()) if ok.any() else None,
                                    protocol_window_ok=bool((ab <= 1e-9 * vmax[ok]).all()))
                        if moving:
                            cell.update(fresh_states=fr, absorbs=n_abs, inverses=inv, punts=pu)
                    f.write(json.dumps(cell) + "\n"); f.flush()
                    print(json.dumps({x: cell.get(x) for x in ("order", "frame", "k", "eps", "plan", "median_ms", "max_abs_err_over_vmax",
                                                                 "punts", "inverses", "error_message")}), flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")
    cur.execute("DROP TABLE IF EXISTS w_ts_ids, w_ts")


if __name__ == "__main__":
    main()

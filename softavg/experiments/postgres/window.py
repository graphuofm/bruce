#!/usr/bin/env python
"""Sliding-window softavg (moving aggregate) vs the frame-recompute baselines.

  SELECT ... OVER (ORDER BY id ROWS BETWEEN k PRECEDING AND CURRENT ROW)
  over the first N rows of imdb (by id), score = s0 (precomputed), on one server.

Plans:
  softavg_moving  softavg(...)  -- 0.2.0 MSFUNC/MINVFUNC (InvAcc::remove; punt -> PG recomputes the frame)
  softavg_fwd     same internal state, no moving form -> PG re-aggregates every frame (O(k) per row)
  naive_window    SUM(EXP(s/eps)*v) OVER w / SUM(EXP(s/eps)) OVER w  (PG's float8 SUM has no inverse)

Timed runs wrap the window in SELECT count(*), sum(x) (no 100k-row transfer);
one extra run fetches every row for correctness against a numpy float64
sliding-window reference, after resetting softavg_window_stats() so the
number of inverse calls and punts (cancellation-guard firings) is recorded.

usage: window.py rel|dbg [N]
"""
import json, sys
import numpy as np
from common import EPS_GRID, HERE, TIMING_STATUS, connect, eps_key, eps_sql, machine_state, run_timed

SERVER = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
KS = [10, 100, 1000]
OUT = HERE / f"window_{SERVER}.jsonl"
PLANS = {
    "softavg_moving": "softavg(value, s0, {E}) OVER w",
    "softavg_fwd": "softavg_fwd(value, s0, {E}) OVER w",
    "naive_window": "SUM(EXP(s0/{E})*value) OVER w / SUM(EXP(s0/{E})) OVER w",
}


def body(expr, k):
    return (f"SELECT id, {expr} AS x FROM imdb WHERE id < {N} "
            f"WINDOW w AS (ORDER BY id ROWS BETWEEN {k} PRECEDING AND CURRENT ROW)")


def reference(v, s, eps, k, chunk=4000):
    out = np.empty(len(v))
    pad_v = np.concatenate([np.full(k, np.nan), v]); pad_s = np.concatenate([np.full(k, -np.inf), s])
    for a in range(0, len(v), chunk):
        b = min(len(v), a + chunk)
        idx = np.arange(a, b)[:, None] + np.arange(k + 1)[None, :]  # window rows in padded coords
        S = pad_s[idx]; V = pad_v[idx]; live = np.isfinite(S)
        if eps == "inf":
            out[a:b] = np.nansum(np.where(live, V, 0), 1) / live.sum(1)
        else:
            m = S.max(1, keepdims=True)
            W = np.where(live, np.exp((S - m) / float(eps)), 0.0)
            out[a:b] = (W * np.where(live, V, 0)).sum(1) / W.sum(1)
    return out


def main():
    conn = connect(SERVER); cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute(f"SELECT value, s0 FROM imdb WHERE id < {N} ORDER BY id")
    r = cur.fetchall(); v = np.array([x[0] for x in r]); s = np.array([x[1] for x in r])
    cur.execute("SELECT p.oid::regprocedure::text, a.aggmtransfn::text, a.aggminvtransfn::text FROM pg_aggregate a "
                "JOIN pg_proc p ON p.oid = a.aggfnoid WHERE p.proname IN ('sum', 'avg', 'softavg') "
                "AND p.proargtypes[0] = 'float8'::regtype ORDER BY 1")
    catalog = [dict(aggregate=a, mtransfn=b, minvtransfn=c) for a, b, c in cur.fetchall()]
    with open(OUT, "w") as f:
        f.write(json.dumps(dict(header=True, server=SERVER, rows=N, ks=KS, machine_at_start=machine_state(),
                                timing_status=TIMING_STATUS, moving_aggregate_catalog=catalog)) + "\n")
        for k in KS:
            pad = np.concatenate([np.zeros(k), np.abs(v)])
            vmax = np.array([pad[i:i + k + 1].max() for i in range(len(v))])
            for e in EPS_GRID:
                ref = reference(v, s, e, k)
                for plan, expr in PLANS.items():
                    q = body(expr.replace("{E}", eps_sql(e)), k)
                    cell = dict(server=SERVER, plan=plan, k=k, eps=eps_key(e), rows=N)
                    t = run_timed(cur, f"SELECT count(*), sum(x) FROM ({q}) t", n_warm=1, n_runs=3)
                    t.pop("rows", None)
                    cell.update(t)
                    if t["sql_ok"]:
                        cur.execute("SELECT softavg_window_stats_reset()")
                        cur.execute(q + " ORDER BY id")
                        got = np.array([x[1] if x[1] is not None else np.nan for x in cur.fetchall()], dtype=np.float64)
                        cur.execute("SELECT * FROM softavg_window_stats()")
                        fr, n_abs, inv, pu = cur.fetchone()
                        ok = np.isfinite(got)
                        ab = np.abs(got[ok] - ref[ok])
                        rel = ab / np.abs(ref[ok])
                        # PROTOCOL window rule: |err| <= 1e-9 * max|v| (per frame)
                        cell.update(n_rows_out=len(got), n_nonfinite=int((~ok).sum()),
                                    max_rel_err=float(rel.max()) if ok.any() else None,
                                    max_abs_err=float(ab.max()) if ok.any() else None,
                                    max_abs_err_over_vmax=float((ab / vmax[ok]).max()) if ok.any() else None,
                                    protocol_window_ok=bool((ab <= 1e-9 * vmax[ok]).all()))
                        if plan == "softavg_moving":
                            cell.update(fresh_states=fr, absorbs=n_abs, inverses=inv, punts=pu)
                    f.write(json.dumps(cell) + "\n"); f.flush()
                    print(json.dumps({x: cell.get(x) for x in ("plan", "k", "eps", "median_ms", "max_rel_err", "max_abs_err_over_vmax", "punts",
                                                                 "inverses", "error_message")}), flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")


if __name__ == "__main__":
    main()

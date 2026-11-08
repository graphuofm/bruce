#!/usr/bin/env python
"""Softmax ladder, sliding windows, SQLite 3.37.2 (system library behind Python's sqlite3).

  SELECT rn, <rung> OVER w FROM <t> WINDOW w AS (ORDER BY rn ROWS BETWEEN k PRECEDING AND CURRENT ROW)

rn = position in the window order (IMDb: id; Amazon: ts, id), INTEGER PRIMARY KEY.
Rungs (spellings of ../sqlite/run.py where they exist):
  S0  naive        SUM(exp(s/eps)*value) OVER w / SUM(exp(s/eps)) OVER w   (SQLite's SUM subtracts on frame exit)
  S1  global_shift the same shifted by the table-wide MAX(s) (the only shift one window pass can express;
                   SQLite's exp underflows to 0 silently, so no clamp is needed here)
  S1x exact_sql    frame max MAX(s) OVER w, then a correlated subquery over the frame rows (exact, O(rows x k))
  S2  softavg_s2   ablation build: no removal, the extension refolds the frame on every xValue after a row left
                   (SQLite will not window an aggregate without xInverse; see ladder_probe())
  S3  softavg_s3   ablation build: plain xInverse subtraction at the current anchor, no certificate
  S4  softavg      shipped engines/sqlite (InvAcc + certificate + lazy frame shadow refold)
Correctness: every query, one fetch of all rows vs the numpy float64 two-pass per-frame reference.
Latency: query 0 only, SELECT count(r), total(r) FROM (...), median of 3 (after the correctness fetch;
1 run when the fetch took > 20 s).

usage: sqlite_window.py [--datasets ...] [--ks ...] [--rungs ...] [--queries N] [--out FILE]
"""
import argparse, json, sqlite3
import numpy as np
from common import (ABL_SQLITE, EPS, KS, NQ, SHIPPED_SQLITE, TIMING_STATUS, WORK, Timer, errors, load,
                    machine_state, reference, vmax_frames, HERE)

RUNGS = {
    "S0": "SELECT rn, SUM(exp({s}/{E})*value) OVER w / SUM(exp({s}/{E})) OVER w AS r FROM {t} {W}",
    "S1": ("SELECT rn, SUM(exp(({s}-g)/{E})*value) OVER w / SUM(exp(({s}-g)/{E})) OVER w AS r "
           "FROM (SELECT rn, value, {s}, (SELECT MAX({s}) FROM {t}) AS g FROM {t}) {W}"),
    "S1x": ("SELECT a.rn, (SELECT SUM(exp((b.{s}-a.m)/{E})*b.value) / SUM(exp((b.{s}-a.m)/{E})) "
            "FROM {t} b WHERE b.rn BETWEEN a.rn - {k} AND a.rn) AS r "
            "FROM (SELECT rn, MAX({s}) OVER w AS m FROM {t} {W}) a"),
    "S2": "SELECT rn, softavg_s2(value, {s}, {E}) OVER w AS r FROM {t} {W}",
    "S3": "SELECT rn, softavg_s3(value, {s}, {E}) OVER w AS r FROM {t} {W}",
    "S4": "SELECT rn, softavg(value, {s}, {E}) OVER w AS r FROM {t} {W}",
}
TIMED = "SELECT count(r), total(r) FROM ({})"


def connect():
    c = sqlite3.connect(WORK / "ladder.sqlite")
    c.enable_load_extension(True)
    c.load_extension(str(ABL_SQLITE.with_suffix("")))
    c.load_extension(str(SHIPPED_SQLITE.with_suffix("")))
    return c


def sql_for(rung, t, k, q, eps):
    W = f"WINDOW w AS (ORDER BY rn ROWS BETWEEN {k} PRECEDING AND CURRENT ROW)"
    return RUNGS[rung].format(s=f"s{q}", E=repr(float(eps)), t=t, W=W, k=k)


def fetch(c, sql):
    try:
        rows = c.execute(sql + " ORDER BY 1").fetchall()
    except sqlite3.Error as e:
        return None, f"{type(e).__name__}: {e}"
    return np.array([np.nan if r[1] is None else r[1] for r in rows], dtype=np.float64), None


def stats(c, rung):
    if rung in ("S2", "S3"):
        return json.loads(c.execute("SELECT ladder_stats()").fetchone()[0])
    if rung == "S4":
        return json.loads(c.execute("SELECT softavg_stats()").fetchone()[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["imdb_all", "imdb_ge", "amazon"])
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--eps", nargs="+", type=float, default=EPS)
    ap.add_argument("--rungs", nargs="+", default=["S0", "S1", "S2", "S3", "S4"])
    ap.add_argument("--queries", type=int, default=NQ)
    ap.add_argument("--out", default=str(HERE / "sqlite_window.jsonl"))
    a = ap.parse_args()
    done = set()
    try:
        for line in open(a.out):
            r = json.loads(line)
            if "rung" in r:
                done.add((r["dataset"], r["k"], r["eps"], r["q"], r["rung"]))
    except FileNotFoundError:
        pass
    c = connect()
    probe = json.loads(c.execute("SELECT ladder_probe()").fetchone()[0])
    try:
        c.execute("SELECT softavg_s2_legacy(value, s0, 0.1) OVER (ORDER BY rn ROWS 2 PRECEDING) FROM imdb_ge LIMIT 1").fetchall()
        probe["legacy_aggregate_as_window"] = "accepted"
    except sqlite3.Error as e:
        probe["legacy_aggregate_as_window"] = f"{type(e).__name__}: {e}"
    try:
        c.execute("SELECT softavg_agg(value, s0, 0.1) OVER (ORDER BY rn ROWS 2 PRECEDING) FROM imdb_ge LIMIT 1").fetchall()
        probe["shipped_softavg_agg_as_window"] = "accepted"
    except sqlite3.Error as e:
        probe["shipped_softavg_agg_as_window"] = f"{type(e).__name__}: {e}"
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(header=True, sqlite_version=sqlite3.sqlite_version, probe=probe, args=vars(a),
                                machine_at_start=machine_state(), timing_status=TIMING_STATUS)) + "\n")
        for ds in a.datasets:
            d = load(ds)
            for k in a.ks:
                vmax = vmax_frames(d["value"], k)
                for eps in a.eps:
                    for q in range(a.queries):
                        todo = [r for r in a.rungs if (ds, k, eps, q, r) not in done]
                        if not todo:
                            continue
                        ref = reference(d["value"], d[f"s{q}"], eps, k)
                        for rung in todo:
                            sql = sql_for(rung, ds, k, q, eps)
                            if rung in ("S2", "S3"):
                                c.execute("SELECT ladder_stats_reset()")
                            if rung == "S4":
                                c.execute("SELECT softavg_stats_reset()")
                            with Timer() as tf:
                                got, err = fetch(c, sql)
                            cell = dict(engine="sqlite", dataset=ds, k=k, eps=eps, q=q, rung=rung, sql=sql,
                                        fetch_ms=tf.ms, error=err)
                            if got is not None:
                                cell.update(errors(got, ref, vmax))
                                cell["stats"] = stats(c, rung)
                                if q == 0:
                                    ts = []
                                    # one timed run when a single fetch already took > 20 s (S1x at large k)
                                    for _ in range(1 if tf.ms > 20000 else 3):
                                        with Timer() as t:
                                            c.execute(TIMED.format(sql)).fetchall()
                                        ts.append(t.ms)
                                    cell.update(timed_ms=ts, median_ms=float(np.median(ts)))
                            f.write(json.dumps(cell) + "\n"); f.flush()
                            print(ds, k, eps, q, rung, cell.get("median_ms"), cell.get("n_wrong"), cell.get("n_null"),
                                  cell.get("max_abs_err_over_vmax"), err, flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")


if __name__ == "__main__":
    main()

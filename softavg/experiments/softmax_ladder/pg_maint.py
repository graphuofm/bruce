#!/usr/bin/env python
"""Softmax ladder, incremental maintenance, PostgreSQL 18.6 (conda env ggdb-pg, release build; cluster
paper_icde/pg18env, port 28860; extensions staged in pg18env/ext, see pg18env/README.md).

Streams: the same statement lists replay.py feeds ladder_replay (maint_ivm 10k-op stream on IMDb;
Amazon time-ordered retention R = 30 / 2 days and their random controls).
Arms on a fresh base table each:
  S2  softavg_abl_ivm_create(..., 's2')   no removal: every group a DELETE/UPDATE touches is recomputed
  S3  softavg_abl_ivm_create(..., 's3')   plain subtraction, never recomputed
  S4  softavg_ivm_create(...)             shipped bruce_pg (certificate; inexact groups recomputed)
Row values follow the replay exactly: an INSERT re-inserts the row's CURRENT (value, s0), i.e. after an
UPDATE of that row if one happened (replay/src/main.rs keeps `cur[i]`); the initial rows are loaded in
id order, the order in which the replay folds each group's initial state.
Per statement: latency of the statement alone (the trigger runs inside it); then, untimed, every group
the statement touched is compared with a fresh GROUP BY fold of its live rows (shipped softavg plain
aggregate = ScalarAcc), with the replay's Drift rule (> 1e-9 v_scale, NULL mismatch or non-finite =
wrong); every checkpoint all groups are compared.
usage: pg_maint.py [--port 28860] [--streams ...] [--eps ...] [--arms ...] --db NAME --out FILE
       (resumes: (stream, eps, arm) runs already in FILE are skipped)
"""
import argparse, io, json, math, struct
import numpy as np
import psycopg2
from common import EPS, HERE, TIMING_STATUS, Timer, load, machine_state
from replay import maint_ivm_stream, retention_streams

STREAMS = ["maint_ivm_10k", "amazon_time_R30", "amazon_random_R30", "amazon_time_R2", "amazon_random_R2"]


class Drift:
    """replay/src/main.rs Drift, verbatim semantics."""

    def __init__(self):
        self.max_abs = 0.0; self.nonfinite = 0; self.null_mismatch = 0; self.wrong = 0; self.checked = 0

    def add(self, got, want, vs):
        self.checked += 1
        if got is not None and want is not None:
            if not math.isfinite(got):
                self.nonfinite += 1; self.wrong += 1
            else:
                dd = abs(got - want) / vs
                self.max_abs = max(self.max_abs, dd)
                if dd > 1e-9:
                    self.wrong += 1
        elif got is None and want is None:
            pass
        else:
            self.null_mismatch += 1; self.wrong += 1

    def json(self):
        return dict(max_abs_err_over_vscale=self.max_abs, nonfinite=self.nonfinite, null_mismatch=self.null_mismatch,
                    wrong=self.wrong, checked=self.checked)


def copy_rows(cur, table, ids, facet, value, s0):
    buf = io.BytesIO(); buf.write(b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0))
    for i, f, v, s in zip(ids.tolist(), facet.tolist(), value.tolist(), s0.tolist()):
        buf.write(struct.pack(">hiiiiidid", 4, 4, i, 4, f, 8, v, 8, s))
    buf.write(struct.pack(">h", -1)); buf.seek(0)
    cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT binary)", buf)


def run(cur, label, fac, val0, s00, init_ids, ops, eps, arm, vs, cp_every):
    E = repr(float(eps)) + "::float8"
    val, s0 = val0.copy(), s00.copy()  # current row values, as the replay's cur[]
    cur.execute("SELECT softavg_ivm_drop('mv'); SELECT softavg_abl_ivm_drop('mv')")  # leftovers of an interrupted run
    cur.execute("DROP TABLE IF EXISTS m_base CASCADE")
    cur.execute("CREATE TABLE m_base(id int PRIMARY KEY, facet int, value float8, s0 float8)")
    # load the initial rows in id order: the replay folds every group's initial state in id order (BTreeSet), and
    # PG's initial GROUP BY build folds in heap order; S3's output after a cancellation is rounding noise of that
    # fold order, so a different order changes which S3 checks come out NULL / non-finite / finite-wrong.
    init = np.sort(np.asarray(init_ids, dtype=np.int64))
    copy_rows(cur, "m_base", init, fac[init], val[init], s0[init])
    cur.execute("CREATE INDEX ON m_base(facet)")
    cur.execute("VACUUM ANALYZE m_base")
    if arm == "S4":
        cur.execute(f"SELECT softavg_ivm_create('mv', 'm_base', 'facet', 'value', 's0', {E})")
        cur.execute("SELECT softavg_ivm_stats_reset()")
    else:
        cur.execute(f"SELECT softavg_abl_ivm_create('mv', 'm_base', 'facet', 'value', 's0', {E}, '{arm.lower()}')")
        cur.execute("SELECT softavg_abl_ivm_reset()")
    fresh_sql = f"SELECT facet, softavg(value, s0, {E}) FROM m_base WHERE facet = ANY(%s) GROUP BY facet"
    lat = {"delete": [], "insert": [], "update": []}
    per = Drift(); first_wrong = -1; cps = []
    for st, op in enumerate(ops):
        if op[0] == "update":
            i = int(op[1]); tg = [int(fac[i])]
            val[i], s0[i] = op[2], op[3]
            sql, args = "UPDATE m_base SET value = %s, s0 = %s WHERE id = %s", (op[2], op[3], i)
        else:
            ids = np.asarray(op[1], dtype=np.int64); tg = sorted(set(fac[ids].tolist()))
            if op[0] == "delete":
                sql, args = "DELETE FROM m_base WHERE id = ANY(%s)", (ids.tolist(),)
            else:
                sql = ("INSERT INTO m_base SELECT * FROM unnest(%s::int[], %s::int[], %s::float8[], %s::float8[])")
                args = (ids.tolist(), fac[ids].tolist(), val[ids].tolist(), s0[ids].tolist())
        with Timer() as t:
            cur.execute(sql, args)
        lat[op[0]].append(t.ms)
        # untimed: touched groups vs a fresh fold
        cur.execute("SELECT facet, softavg FROM mv WHERE facet = ANY(%s)", (tg,)); got = dict(cur.fetchall())
        cur.execute(fresh_sql, (tg,)); want = dict(cur.fetchall())
        before = per.wrong
        for g in tg:
            per.add(got.get(g), want.get(g), vs)
        if per.wrong > before and first_wrong < 0:
            first_wrong = st
        if (st + 1) % cp_every == 0 or st + 1 == len(ops):
            cur.execute("SELECT facet, softavg FROM mv"); got = dict(cur.fetchall())
            cur.execute(f"SELECT facet, softavg(value, s0, {E}) FROM m_base GROUP BY facet"); want = dict(cur.fetchall())
            d = Drift()
            for g in sorted(set(got) | set(want)):
                d.add(got.get(g), want.get(g), vs)
            cps.append(dict(after_statements=st + 1, **d.json()))
    if arm == "S4":
        cur.execute("SELECT trigger_statements, groups_recomputed FROM softavg_ivm_stats()"); tstmts, rec = cur.fetchone()
        cur.execute("SELECT softavg_ivm_drop('mv')")
    else:
        cur.execute("SELECT softavg_abl_ivm_recomputed()"); rec = cur.fetchone()[0]; tstmts = None
        cur.execute("SELECT softavg_abl_ivm_drop('mv')")
    cur.execute("DROP TABLE m_base")
    allms = np.asarray(sum(lat.values(), []))
    return dict(stream=label, eps=eps, arm=arm, statements=len(ops), groups_recomputed=int(rec), trigger_statements=tstmts,
                per_statement_touched=per.json(), first_wrong_statement=first_wrong, checkpoints=cps,
                per_stmt_ms=dict(median=float(np.median(allms)), p99=float(np.percentile(allms, 99)), mean=float(allms.mean()),
                                 by_kind={k: dict(n=len(v), median=float(np.median(v)), p99=float(np.percentile(v, 99)))
                                          for k, v in lat.items() if v}),
                machine_at_end=machine_state())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=28860)
    ap.add_argument("--streams", nargs="+", default=STREAMS)
    ap.add_argument("--eps", nargs="+", type=float, default=EPS)
    ap.add_argument("--arms", nargs="+", default=["S2", "S3", "S4"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c0 = psycopg2.connect(host="127.0.0.1", port=a.port, dbname="postgres", user="jding"); c0.autocommit = True
    k0 = c0.cursor(); k0.execute("SELECT 1 FROM pg_database WHERE datname = %s", (a.db,))
    if not k0.fetchone():
        k0.execute(f"CREATE DATABASE {a.db}")
    c0.close()
    conn = psycopg2.connect(host="127.0.0.1", port=a.port, dbname=a.db, user="jding"); conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS bruce_pg; CREATE EXTENSION IF NOT EXISTS softavg_ablation")
    cur.execute("SET synchronous_commit = off; SET max_parallel_workers_per_gather = 0")
    out = HERE / a.out if "/" not in a.out else a.out
    try:
        res = json.loads(open(out).read())
    except FileNotFoundError:
        cur.execute("SELECT version()")
        res = dict(version=cur.fetchone()[0], timing_status=TIMING_STATUS, machine_at_start=machine_state(), runs=[])
    done = {(r["stream"], r["eps"], r["arm"]) for r in res["runs"]}
    cache = {}

    def corpus(name):
        if name not in cache:
            d = load(name); n = int(d["id"].max()) + 1
            fac = np.zeros(n, np.int32); v = np.zeros(n); s = np.zeros(n)
            fac[d["id"]] = d["facet"]; v[d["id"]] = d["value"]; s[d["id"]] = d["s0"]
            cache.clear(); cache[name] = (d, fac, v, s)
        return cache[name]
    T0 = int(np.datetime64("2020-01-01T00:00:00", "ms").astype(np.int64))
    for label in a.streams:
        if label == "maint_ivm_10k":
            d, fac, v, s = corpus("imdb_all"); init = d["id"]; ops, _ = maint_ivm_stream(d); vs, cp = 10.0, 1000
        else:
            d, fac, v, s = corpus("amazon"); R = int(label.split("_R")[1])
            init, _, t_ops, r_ops = retention_streams(d, T0, 50_000, R)
            ops = t_ops if "_time_" in label else r_ops; vs, cp = 5.0, 2000
        for eps in a.eps:
            for arm in a.arms:
                if (label, eps, arm) in done:
                    continue
                r = run(cur, label, fac, v, s, init, ops, eps, arm, vs, cp)
                res["runs"].append(r)
                open(out, "w").write(json.dumps(res, indent=1))
                p = r["per_statement_touched"]
                print(label, eps, arm, r["groups_recomputed"], p["max_abs_err_over_vscale"], p["wrong"],
                      round(r["per_stmt_ms"]["median"], 3), flush=True)
    res["machine_at_end"] = machine_state()
    open(out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

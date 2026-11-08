#!/usr/bin/env python
"""softavg_ivm under a TIME-ORDERED stream with sliding retention vs a random stream.

Rows: IMDb movies whose earliest release date has day precision (imdb_ts.parquet).
Base m_ts(id PK, facet, value, s0, ts); view = SELECT facet, softavg(value, s0, eps) GROUP BY facet
maintained by bruce_pg softavg_ivm_create (single label, per-facet InvAcc states).

Initial table: rows with ts in [t0 - R, t0).
time stream   : the next N rows in (ts, id) order are inserted one per statement; after
                each insert, the rows that fell out of the retention window
                (ts < ts_new - R) are deleted in one statement (only when there are any).
random stream : the SAME N rows inserted in a random order (RandomState(5)); at the same
                steps a DELETE of the same number of rows, chosen uniformly among live rows.
Both streams: same initial table, same statements count, same insert / delete row counts.
Per statement: latency, rows, groups recomputed (softavg_ivm_stats delta). Checkpoints
every C inserts: drift vs a fresh GROUP BY. Also the per-facet argmax-removal count
(how often a delete removes the row that holds the group's max score).

CORPUS=amazon (environment): the same over Amazon reviews with their recovered review
timestamps (amz_ts from load_amazon_ts.py, ms precision; v_scale 5)
-> results_maint_ts_amazon_R<retention_days>.json.

usage: [CORPUS=amazon] maint_ts.py [N] [C] [t0] [retention_days]   -> results_maint_ts_R<retention_days>.json
"""
import heapq, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import TIMING_STATUS, Timer, connect, eps_key, eps_sql, machine_state, server_env  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
C = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000
T0 = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2008-01-01")
R = float(sys.argv[4]) if len(sys.argv) > 4 else 365.0
CORPUS = os.environ.get("CORPUS", "imdb")
EPS = [1e-3, 0.02, 0.1]
V_SCALE = 10.0 if os.environ.get("CORPUS", "imdb") == "imdb" else 5.0  # IMDb ratings 1-10, Amazon stars 1-5
FRESH = "SELECT facet, softavg(value, s0, {E}) FROM m_ts GROUP BY facet"


def pct(xs):
    a = np.asarray(xs, dtype=float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()))


def drift(got_rows, ref_rows):
    got = dict(got_rows); ref = dict(ref_rows)
    ab, mism = [], 0
    for g in set(got) | set(ref):
        a, b = got.get(g), ref.get(g)
        if a is None or b is None:
            mism += (a is None) != (b is None); continue
        ab.append(abs(a - b))
    return dict(max_abs_err_over_vscale=(max(ab) / V_SCALE if ab else None), group_mismatch=mism)


def build_streams(rows):
    """rows: DataFrame(id, facet, value, s0, ts) sorted by (ts, id)."""
    lo = T0 - pd.Timedelta(days=R)
    init = rows[(rows.ts >= lo) & (rows.ts < T0)]
    fut = rows[rows.ts >= T0].iloc[:N]
    # time stream: FIFO retention
    live_ts = list(zip(init.ts, init.id))  # sorted
    head = 0
    time_ops = []
    for t, i in zip(fut.ts, fut.id):
        time_ops.append(("insert", [int(i)]))
        live_ts.append((t, i))
        cut = t - pd.Timedelta(days=R); expired = []
        while live_ts[head][0] < cut:
            expired.append(int(live_ts[head][1])); head += 1
        if expired:
            time_ops.append(("delete", expired))
    # random stream: same rows in random order, same delete counts at the same steps
    rng = np.random.RandomState(5)
    perm = list(fut.id.values[rng.permutation(len(fut))])
    live = list(init.id.astype(int)); pos = {x: k for k, x in enumerate(live)}
    rand_ops = []; it = iter(perm)

    def remove_random(cnt):
        out = []
        for _ in range(cnt):
            k = rng.randint(len(live)); x = live[k]
            last = live.pop(); 
            if k < len(live):
                live[k] = last; pos[last] = k
            del pos[x]; out.append(int(x))
        return out
    for op in time_ops:
        if op[0] == "insert":
            x = int(next(it)); rand_ops.append(("insert", [x])); pos[x] = len(live); live.append(x)
        else:
            rand_ops.append(("delete", remove_random(len(op[1]))))
    return init, fut, time_ops, rand_ops


def run(cur, name, eps, init_ids, ops, rowmap):
    E = eps_sql(eps)
    cur.execute("DROP TABLE IF EXISTS m_ts CASCADE")
    cur.execute("CREATE TABLE m_ts(id int PRIMARY KEY, facet text, value float8, s0 float8, ts timestamp)")
    cur.execute("INSERT INTO m_ts SELECT id, facet, value, s0, ts FROM ts_src WHERE id = ANY(%s)", (list(map(int, init_ids)),))
    cur.execute("CREATE INDEX ON m_ts(facet)"); cur.execute("CREATE INDEX ON m_ts(ts)"); cur.execute("VACUUM ANALYZE m_ts")
    cur.execute(f"SELECT softavg_ivm_create('mv_ts', 'm_ts', 'facet', 'value', 's0', {E})")
    cur.execute("SELECT softavg_ivm_stats_reset()")
    # argmax bookkeeping in Python: per facet, the live max score
    cur.execute("SELECT id, facet, s0 FROM m_ts"); live = {i: (f, s) for i, f, s in cur.fetchall()}
    heaps = defaultdict(list)  # per facet max-heap (-s, id), lazy deletion
    for i, (f, s) in live.items():
        heaps[f].append((-s, i))
    for h in heaps.values():
        heapq.heapify(h)

    def fmax(f):
        h = heaps[f]
        while h and h[0][1] not in live:
            heapq.heappop(h)
        return -h[0][0]
    per = defaultdict(list); rec_by = defaultdict(int); argmax_del = 0; prev = 0; cps = []; n_ins = 0
    rows_by = defaultdict(int)
    for op, ids in ops:
        if op == "delete":
            byf = {f: fmax(f) for f in {live[i][0] for i in ids}}
            argmax_del += sum(1 for i in ids if live[i][1] == byf[live[i][0]])
            with Timer() as t:
                cur.execute("DELETE FROM m_ts WHERE id = ANY(%s)", (ids,))
            for i in ids:
                live.pop(i)
        else:
            r = rowmap[ids[0]]
            with Timer() as t:
                cur.execute("INSERT INTO m_ts VALUES (%s, %s, %s, %s, %s)", r)
            live[ids[0]] = (r[1], r[3]); n_ins += 1
            heapq.heappush(heaps[r[1]], (-r[3], ids[0]))
        cur.execute("SELECT groups_recomputed FROM softavg_ivm_stats()"); st = cur.fetchone()[0]
        per[op].append(t.ms); rec_by[op] += st - prev; rows_by[op] += len(ids); prev = st
        if op == "insert" and (n_ins % C == 0 or n_ins == N):
            cur.execute(FRESH.replace("{E}", E)); ref = cur.fetchall()
            cur.execute("SELECT facet, softavg FROM mv_ts"); got = cur.fetchall()
            cps.append(dict(after_inserts=n_ins, live_rows=len(live), **drift(got, ref)))
    cur.execute("SELECT softavg_ivm_drop('mv_ts')")
    return dict(stream=name, eps=eps_key(eps), statements={k: dict(pct(v), rows=rows_by[k], recomputed=int(rec_by[k]))
                                                           for k, v in per.items()},
                groups_recomputed_total=int(sum(rec_by.values())), argmax_rows_deleted=int(argmax_del),
                max_abs_err_over_vscale=max((c["max_abs_err_over_vscale"] or 0) for c in cps),
                group_mismatch_total=sum(c["group_mismatch"] for c in cps), checkpoints=cps)


def main():
    conn = connect("rel"); cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0"); cur.execute("SET synchronous_commit = off")
    cur.execute("DROP TABLE IF EXISTS ts_ids, ts_src")
    if CORPUS == "imdb":
        ts = pd.read_parquet(HERE / "imdb_ts.parquet"); ts = ts[ts.precision == "day"]
        cur.execute("CREATE TABLE ts_ids(id int PRIMARY KEY, ts timestamp NOT NULL)")
        cur.executemany("INSERT INTO ts_ids VALUES (%s, %s)", list(zip(ts.row.astype(int).tolist(), ts.ts.dt.to_pydatetime().tolist())))
        cur.execute("CREATE TABLE ts_src AS SELECT i.id, i.facet, i.value, i.s0, t.ts FROM imdb i JOIN ts_ids t USING (id)")
    else:
        cur.execute("CREATE TABLE ts_src AS SELECT id, facet, value, s0, ts FROM amz_ts")
    cur.execute("ALTER TABLE ts_src ADD PRIMARY KEY (id)")
    cur.execute("SELECT id, facet, value, s0, ts FROM ts_src ORDER BY ts, id")
    rr = cur.fetchall()
    rows = pd.DataFrame(rr, columns=["id", "facet", "value", "s0", "ts"])
    rows["ts"] = pd.to_datetime(rows.ts)
    rowmap = {r[0]: r for r in rr}
    init, fut, time_ops, rand_ops = build_streams(rows)
    out = dict(timing_status=TIMING_STATUS, server="rel", server_build=server_env("rel")["server_build"],
               machine_at_start=machine_state(),
               setup=dict(corpus=CORPUS, t0=str(T0.date()), retention_days=R, n_inserts=len(fut), initial_rows=len(init),
                          inserted_ts_range=[str(fut.ts.min()), str(fut.ts.max())],
                          statements=dict(time=len(time_ops), random=len(rand_ops)),
                          delete_statements=sum(o[0] == "delete" for o in time_ops),
                          rows_deleted=sum(len(o[1]) for o in time_ops if o[0] == "delete"),
                          ts_precision="day" if CORPUS == "imdb" else "ms", commit="autocommit, synchronous_commit=off", v_scale=V_SCALE),
               runs=[])
    for eps in EPS:
        for name, ops in (("time", time_ops), ("random", rand_ops)):
            r = run(cur, name, eps, init.id.values, ops, rowmap)
            out["runs"].append(r)
            print(json.dumps({k: r[k] for k in ("stream", "eps", "groups_recomputed_total", "argmax_rows_deleted",
                                                "max_abs_err_over_vscale", "group_mismatch_total")}),
                  json.dumps({k: (v["median_ms"], v["recomputed"], v["rows"]) for k, v in r["statements"].items()}), flush=True)
    cur.execute("DROP TABLE IF EXISTS m_ts, ts_ids, ts_src CASCADE")
    out["machine_at_end"] = machine_state()
    tag = "" if CORPUS == "imdb" else f"_{CORPUS}"
    (HERE / f"results_maint_ts{tag}_R{R:g}.json").write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()

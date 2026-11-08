#!/usr/bin/env python
"""Maintaining SELECT facet, softavg(value, s0, eps) FROM base GROUP BY facet
under a stream of single-row INSERT / DELETE / UPDATE statements (IMDb, score
s0 = -(emb <#> q0) precomputed), inside PostgreSQL. Release server only.

Arms (each on a fresh copy `m_base(id PK, facet, value, s0)` + btree(facet)):
  softavg_ivm        bruce_pg softavg_ivm_create: per-facet InvAcc state table,
                     statement triggers over transition tables; inexact groups
                     recomputed from m_base in the same statement
  pg_ivm_linear      pg_ivm 1.16 IMMV of SUM(EXP(s0/eps)*value), SUM(EXP(s0/eps)),
                     COUNT(*); answer = num/den (pg_ivm cannot hold softavg)
  refresh_mv         MATERIALIZED VIEW of the softavg query; REFRESH at every
                     checkpoint (stale in between: staleness error recorded)
  recompute_on_read  no view; every read runs the softavg query on m_base
Stream: N ops, generated once (RandomState(7)): first the 29 per-facet argmax
rows are deleted (they carry the weight at sharp eps), then ~45% deletes of a
random live row, ~45% re-inserts of a previously deleted row, ~10% updates
(value and score taken from another row). Autocommit, one statement per op.
Checkpoints every C ops: the arm's answer vs a fresh softavg query (drift),
read latency, recomputes so far.
Also, once: rollback and two-session concurrency checks of softavg_ivm.

usage: run.py [N] [C]   -> results_maint_ivm.json (merged by ../assemble.py)
"""
import json, sys, threading, time
from pathlib import Path
import numpy as np
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TIMING_STATUS, Timer, connect, eps_key, eps_sql, machine_state, server_env  # noqa: E402

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
C = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000
EPS = [1e-3, 0.02, 0.1]
SERVER = "rel"
V_SCALE = 10.0  # IMDb ratings are in [1, 10]
FRESH = "SELECT facet, softavg(value, s0, {E}) FROM m_base GROUP BY facet"
LIN = "SELECT facet, SUM(EXP(s0/{E})*value) AS num, SUM(EXP(s0/{E})) AS den, COUNT(*) AS n FROM m_base GROUP BY facet"
READ = {
    "softavg_ivm": "SELECT facet, softavg FROM mv_soft",
    "pg_ivm_linear": "SELECT facet, num / den FROM immv_lin",
    "refresh_mv": "SELECT facet, x FROM mv_refresh",
    "recompute_on_read": FRESH,
}


def pct(xs):
    a = np.asarray(xs, dtype=float)
    return dict(n=int(len(a)), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p90_ms=float(np.percentile(a, 90)), p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()),
                total_s=float(a.sum() / 1e3))


def errmsg(e):
    return (e.pgerror or str(e)).strip().splitlines()[0]


def make_stream(cur):
    cur.execute("SELECT id, facet, value, s0 FROM imdb ORDER BY id")
    rows = {r[0]: r for r in cur.fetchall()}
    cur.execute("SELECT DISTINCT ON (facet) id FROM imdb ORDER BY facet, s0 DESC")
    argmax = [r[0] for r in cur.fetchall()]
    rng = np.random.RandomState(7)
    live = set(rows); dead = []
    ids = np.array(sorted(rows))
    ops = []
    for i in argmax:
        ops.append(("delete", i)); live.discard(i); dead.append(i)
    while len(ops) < N:
        u = rng.rand()
        if u < 0.45 or not dead:
            while True:
                i = int(ids[rng.randint(len(ids))])
                if i in live:
                    break
            ops.append(("delete", i)); live.discard(i); dead.append(i)
        elif u < 0.90:
            i = dead.pop(rng.randint(len(dead)))
            ops.append(("insert", i)); live.add(i)
        else:
            while True:
                i = int(ids[rng.randint(len(ids))])
                if i in live:
                    break
            donor = rows[int(ids[rng.randint(len(ids))])]
            ops.append(("update", i, donor[2], donor[3]))
    return ops, rows, len(argmax)


def apply(cur, op, rows):
    if op[0] == "delete":
        cur.execute("DELETE FROM m_base WHERE id = %s", (op[1],))
    elif op[0] == "insert":
        r = rows[op[1]]
        cur.execute("INSERT INTO m_base(id, facet, value, s0) VALUES (%s, %s, %s, %s)", r)
    else:
        cur.execute("UPDATE m_base SET value = %s, s0 = %s WHERE id = %s", (op[2], op[3], op[1]))


def fresh_base(cur):
    cur.execute("DROP TABLE IF EXISTS m_base CASCADE")
    cur.execute("CREATE TABLE m_base AS SELECT id, facet, value, s0 FROM imdb")
    cur.execute("ALTER TABLE m_base ADD PRIMARY KEY (id)")
    cur.execute("CREATE INDEX m_base_facet ON m_base(facet)")
    cur.execute("VACUUM ANALYZE m_base")


def drift(got_rows, ref_rows):
    got = {r[0]: r[1] for r in got_rows}; ref = {r[0]: r[1] for r in ref_rows}
    ab, rel, mism = [], [], 0
    for g in set(got) | set(ref):
        a, b = got.get(g), ref.get(g)
        if a is None or b is None:
            mism += (a is None) != (b is None)
            continue
        ab.append(abs(a - b)); rel.append(abs(a - b) / abs(b) if b else abs(a - b))
    return dict(max_abs_err_over_vscale=(max(ab) / V_SCALE if ab else None), max_rel_err=(max(rel) if rel else None),
                groups_view=len(got), groups_fresh=len(ref), group_mismatch=mism)


def run_arm(cur, arm, eps, ops, rows):
    E = eps_sql(eps)
    fresh_base(cur)
    out = dict(arm=arm, eps=eps_key(eps))
    try:
        with Timer() as t:
            if arm == "softavg_ivm":
                cur.execute(f"SELECT softavg_ivm_create('mv_soft', 'm_base', 'facet', 'value', 's0', {E})")
            elif arm == "pg_ivm_linear":
                cur.execute("SELECT pgivm.create_immv('immv_lin', $$" + LIN.replace("{E}", E) + "$$)")
            elif arm == "refresh_mv":
                cur.execute("CREATE MATERIALIZED VIEW mv_refresh AS SELECT facet, softavg(value, s0, " + E + ") AS x FROM m_base GROUP BY facet")
        out["create_ms"] = t.ms; out["created"] = True
    except psycopg2.Error as e:
        out.update(created=False, error_message=errmsg(e))
        return out
    cur.execute("SELECT softavg_ivm_stats_reset()")
    lat = {"delete": [], "insert": [], "update": []}
    cps = []
    for k, op in enumerate(ops, 1):
        with Timer() as t:
            apply(cur, op, rows)
        lat[op[0]].append(t.ms)
        if k % C == 0 or k == len(ops):
            cp = dict(after_ops=k)
            with Timer() as tf:
                cur.execute(FRESH.replace("{E}", E)); ref = cur.fetchall()
            cp["fresh_query_ms"] = tf.ms
            if arm == "refresh_mv":
                cur.execute(READ[arm]); cp["stale_before_refresh"] = drift(cur.fetchall(), ref)
                with Timer() as tr:
                    cur.execute("REFRESH MATERIALIZED VIEW mv_refresh")
                cp["refresh_ms"] = tr.ms
            with Timer() as tr:
                cur.execute(READ[arm].replace("{E}", E)); got = cur.fetchall()
            cp["read_ms"] = tr.ms
            cp.update(drift(got, ref))
            cur.execute("SELECT * FROM softavg_ivm_stats()")
            cp["trigger_statements"], cp["groups_recomputed"] = cur.fetchone()
            cps.append(cp)
    out["ops"] = {k: pct(v) for k, v in lat.items() if v}
    out["ops"]["all"] = pct(sum(lat.values(), []))
    out["checkpoints"] = cps
    out["groups_recomputed_total"] = cps[-1]["groups_recomputed"]
    out["max_abs_err_over_vscale"] = max((c["max_abs_err_over_vscale"] or 0) for c in cps)
    out["max_rel_err"] = max((c["max_rel_err"] or 0) for c in cps)
    out["group_mismatch_total"] = sum(c["group_mismatch"] for c in cps)
    out["read_ms_median"] = float(np.median([c["read_ms"] for c in cps]))
    if arm == "refresh_mv":
        out["refresh_ms_median"] = float(np.median([c["refresh_ms"] for c in cps]))
        out["stale_max_abs_err_over_vscale"] = max((c["stale_before_refresh"]["max_abs_err_over_vscale"] or 0) for c in cps)
    if arm == "softavg_ivm":
        cur.execute("SELECT softavg_ivm_drop('mv_soft')")
    return out


def transactional_checks(cur):
    """softavg_ivm under ROLLBACK and two concurrent sessions (eps = 1e-3)."""
    E = eps_sql(1e-3)
    res = {}
    fresh_base(cur)
    cur.execute(f"SELECT softavg_ivm_create('mv_soft', 'm_base', 'facet', 'value', 's0', {E})")

    def agrees(c):
        c.execute("SELECT facet, softavg FROM mv_soft"); got = c.fetchall()
        c.execute(FRESH.replace("{E}", E)); ref = c.fetchall()
        d = drift(got, ref)
        return d["group_mismatch"] == 0 and (d["max_abs_err_over_vscale"] or 0) <= 1e-9, d

    cur.execute("SELECT string_agg(encode(state, 'hex'), ',' ORDER BY g) FROM mv_soft_state"); before = cur.fetchone()[0]
    a = connect(SERVER, autocommit=False); ca = a.cursor()
    ca.execute("DELETE FROM m_base WHERE id IN (SELECT DISTINCT ON (facet) id FROM m_base ORDER BY facet, s0 DESC)")
    ca.execute("INSERT INTO m_base SELECT 9000000 + i, 'Drama', 9.9, 0.999 FROM generate_series(1, 20) i")
    a.rollback()
    cur.execute("SELECT string_agg(encode(state, 'hex'), ',' ORDER BY g) FROM mv_soft_state")
    res["rollback_state_bytes_identical"] = cur.fetchone()[0] == before
    res["rollback_view_agrees"] = agrees(cur)[0]

    # two sessions on the same facet, both orders. T1 deletes the facet's
    # argmax row (forces a recompute); T2 inserts a new dominant row.
    for order in ("t1_first", "t2_first"):
        cur.execute("SELECT facet, id FROM m_base ORDER BY s0 DESC LIMIT 1"); facet, top = cur.fetchone()
        new_id = 9_100_000 + (order == "t2_first")
        t1 = connect(SERVER, autocommit=False); c1 = t1.cursor()
        t2 = connect(SERVER, autocommit=False); c2 = t2.cursor()
        first, second = (c1, c2) if order == "t1_first" else (c2, c1)
        stmt = {id(c1): ("DELETE FROM m_base WHERE id = %s", (top,)),
                id(c2): ("INSERT INTO m_base VALUES (%s, %s, 5.5, 1.5)", (new_id, facet))}
        first.execute(*stmt[id(first)])
        th = threading.Thread(target=lambda: second.execute(*stmt[id(second)]))
        th.start(); time.sleep(0.5)
        blocked = th.is_alive()
        first.connection.commit(); th.join(); second.connection.commit()
        t1.close(); t2.close()
        ok, d = agrees(cur)
        res[order] = dict(second_session_blocked_on_state_row=blocked, view_agrees_after_both_commit=ok, drift=d)
    cur.execute("SELECT softavg_ivm_drop('mv_soft')")
    return res


def main():
    conn = connect(SERVER); cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    # single-row autocommit statements are otherwise dominated by the WAL
    # flush, which is identical for every arm and hides the maintenance cost
    cur.execute("SET synchronous_commit = off")
    ops, rows, n_argmax = make_stream(cur)
    out = dict(timing_status=TIMING_STATUS, server=SERVER, server_build=server_env(SERVER)["server_build"],
               machine_at_start=machine_state(),
               stream=dict(n_ops=len(ops), checkpoint_every=C, first_ops_delete_argmax_rows=n_argmax,
                           mix={k: sum(o[0] == k for o in ops) for k in ("delete", "insert", "update")},
                           commit="autocommit, one statement per op, synchronous_commit=off in the stream session", seed=7, v_scale=V_SCALE),
               arms=[])
    out["transactional_checks"] = transactional_checks(cur)
    print(json.dumps(out["transactional_checks"]), flush=True)
    for eps in EPS:
        for arm in ("softavg_ivm", "pg_ivm_linear", "refresh_mv", "recompute_on_read"):
            r = run_arm(cur, arm, eps, ops, rows)
            out["arms"].append(r)
            print(json.dumps({k: r.get(k) for k in ("arm", "eps", "created", "error_message", "groups_recomputed_total",
                                                     "max_abs_err_over_vscale", "read_ms_median")}),
                  (r.get("ops") or {}).get("all", {}).get("median_ms"), flush=True)
            for v in ("immv_lin",):
                cur.execute(f"DROP TABLE IF EXISTS {v}")
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_refresh")
    cur.execute("DROP TABLE IF EXISTS m_base CASCADE")
    out["machine_at_end"] = machine_state()
    (HERE / "results_maint_ivm.json").write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Maintaining the multi-label GROUP BY over a join under single-row edits.

    SELECT g.facet, softavg(b.value, b.s0, eps) FROM m_base b JOIN m_genre g ON g.id = b.id GROUP BY g.facet

m_base(id PK, facet, value, s0) = imdb; m_genre(id, facet) = imdb_genre (complete
IMDb genre memberships, 1.54 per movie, up to 13). Release server only.

Arms (fresh copies of both tables per arm):
  softavg_ivm_join    bruce_pg softavg_ivm_create_join: per-facet InvAcc states; a base-row
                      edit fans out to all its memberships, a bridge-row edit is one delta
  softavg_ivm_single  bruce_pg softavg_ivm_create on m_base.facet (single label, fan-out 1):
                      the same base-row stream, for the fan-out-1 baseline
  pg_ivm_linear_join  pg_ivm IMMV of SUM(EXP(s0/eps)*value), SUM(EXP(s0/eps)), COUNT(*) over the join
  refresh_mv          MATERIALIZED VIEW of the join query, REFRESH at every checkpoint
  recompute_on_read   every read runs the join query
Stream (RandomState(11)): first the 30 per-facet (multi-label) argmax items are deleted,
then 40% delete of a live item, 40% re-insert of a deleted item (its bridge rows stay,
so it comes back with its memberships), 10% update (value, s0 from another row),
5% bridge insert (a removed membership comes back), 5% bridge delete (one membership).
Every op is one autocommit statement. Per op: latency, fan-out (the item's live
memberships), groups recomputed (softavg_ivm_stats delta). Checkpoints every C ops:
drift vs a fresh query, read latency. Then a fan-out sweep: for each fan-out f, up to
F live items with exactly f memberships are deleted and re-inserted (per-edit latency
and recomputes vs f). Once: rollback, two-session base-vs-bridge races (both orders,
READ COMMITTED), a REPEATABLE READ base writer racing a bridge writer, and a bridge
write under REPEATABLE READ.

usage: maint_join.py [N] [C] [F]   -> results_maint_join.json
"""
import json, sys, threading, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import TIMING_STATUS, Timer, connect, eps_key, eps_sql, machine_state, server_env  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
C = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000
FS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
EPS = [1e-3, 0.02, 0.1]
V_SCALE = 10.0
FRESH = "SELECT g.facet, softavg(b.value, b.s0, {E}) FROM m_base b JOIN m_genre g ON g.id = b.id GROUP BY g.facet"
FRESH_SINGLE = "SELECT facet, softavg(value, s0, {E}) FROM m_base GROUP BY facet"
LIN = ("SELECT g.facet, SUM(EXP(b.s0/{E})*b.value) AS num, SUM(EXP(b.s0/{E})) AS den, COUNT(*) AS n "
       "FROM m_base b JOIN m_genre g ON g.id = b.id GROUP BY g.facet")
READ = {"softavg_ivm_join": "SELECT facet, softavg FROM mv_j",
        "softavg_ivm_single": "SELECT facet, softavg FROM mv_s",
        "pg_ivm_linear_join": "SELECT facet, num / den FROM immv_lin",
        "refresh_mv": "SELECT facet, x FROM mv_refresh",
        "recompute_on_read": FRESH}
ARMS = list(READ)


def pct(xs):
    a = np.asarray(xs, dtype=float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p90_ms=float(np.percentile(a, 90)), p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()))


def errmsg(e):
    return (e.pgerror or str(e)).strip().splitlines()[0]


def make_stream(cur):
    cur.execute("SELECT id, facet, value, s0 FROM imdb ORDER BY id")
    rows = {r[0]: r for r in cur.fetchall()}
    cur.execute("SELECT id, facet FROM imdb_genre")
    mem = defaultdict(set)
    for i, g in cur.fetchall():
        mem[i].add(g)
    cur.execute("SELECT DISTINCT ON (g.facet) i.id FROM imdb i JOIN imdb_genre g USING (id) ORDER BY g.facet, i.s0 DESC")
    argmax = list(dict.fromkeys(r[0] for r in cur.fetchall()))
    rng = np.random.RandomState(11)
    ids = np.array(sorted(rows)); live = set(rows); dead = []
    mem_live = {i: set(s) for i, s in mem.items()}; mem_removed = []
    ops = []

    def pick_live():
        while True:
            i = int(ids[rng.randint(len(ids))])
            if i in live:
                return i
    for i in argmax:
        ops.append(("delete", i)); live.discard(i); dead.append(i)
    while len(ops) < N:
        u = rng.rand()
        if u < 0.40 or not dead:
            i = pick_live(); ops.append(("delete", i)); live.discard(i); dead.append(i)
        elif u < 0.80:
            i = dead.pop(rng.randint(len(dead))); ops.append(("insert", i)); live.add(i)
        elif u < 0.90:
            i = pick_live(); d = rows[int(ids[rng.randint(len(ids))])]
            ops.append(("update", i, d[2], d[3]))
        elif u < 0.95 and mem_removed:
            i, g = mem_removed.pop(rng.randint(len(mem_removed)))
            ops.append(("bridge_insert", i, g)); mem_live.setdefault(i, set()).add(g)
        else:
            while True:
                i = pick_live()
                if mem_live.get(i):
                    break
            g = sorted(mem_live[i])[rng.randint(len(mem_live[i]))]
            ops.append(("bridge_delete", i, g)); mem_live[i].discard(g); mem_removed.append((i, g))
    return ops, rows, argmax


def apply(cur, op, rows):
    k = op[0]
    if k == "delete":
        cur.execute("DELETE FROM m_base WHERE id = %s", (op[1],))
    elif k == "insert":
        cur.execute("INSERT INTO m_base(id, facet, value, s0) VALUES (%s, %s, %s, %s)", rows[op[1]])
    elif k == "update":
        cur.execute("UPDATE m_base SET value = %s, s0 = %s WHERE id = %s", (op[2], op[3], op[1]))
    elif k == "bridge_insert":
        cur.execute("INSERT INTO m_genre(id, facet) VALUES (%s, %s)", (op[1], op[2]))
    else:
        cur.execute("DELETE FROM m_genre WHERE id = %s AND facet = %s", (op[1], op[2]))


def fresh_tables(cur):
    cur.execute("DROP TABLE IF EXISTS m_base, m_genre CASCADE")
    cur.execute("CREATE TABLE m_base AS SELECT id, facet, value, s0 FROM imdb")
    cur.execute("ALTER TABLE m_base ADD PRIMARY KEY (id)")
    cur.execute("CREATE INDEX m_base_facet ON m_base(facet)")
    cur.execute("CREATE TABLE m_genre AS SELECT id, facet FROM imdb_genre")
    cur.execute("CREATE INDEX m_genre_id ON m_genre(id); CREATE INDEX m_genre_facet ON m_genre(facet)")
    cur.execute("VACUUM ANALYZE m_base"); cur.execute("VACUUM ANALYZE m_genre")


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


def stats(cur):
    cur.execute("SELECT * FROM softavg_ivm_stats()")
    return cur.fetchone()


def create(cur, arm, E):
    if arm == "softavg_ivm_join":
        cur.execute(f"SELECT softavg_ivm_create_join('mv_j', 'm_base', 'id', 'm_genre', 'id', 'facet', 'value', 's0', {E})")
    elif arm == "softavg_ivm_single":
        cur.execute(f"SELECT softavg_ivm_create('mv_s', 'm_base', 'facet', 'value', 's0', {E})")
    elif arm == "pg_ivm_linear_join":
        cur.execute("SELECT pgivm.create_immv('immv_lin', $$" + LIN.replace("{E}", E) + "$$)")
    elif arm == "refresh_mv":
        cur.execute(f"CREATE MATERIALIZED VIEW mv_refresh AS SELECT g.facet, softavg(b.value, b.s0, {E}) AS x "
                    "FROM m_base b JOIN m_genre g ON g.id = b.id GROUP BY g.facet")


def drop(cur):
    cur.execute("SELECT softavg_ivm_drop('mv_j'); SELECT softavg_ivm_drop('mv_s')")
    cur.execute("DROP TABLE IF EXISTS immv_lin; DROP MATERIALIZED VIEW IF EXISTS mv_refresh")


def run_arm(cur, arm, eps, ops, rows):
    E = eps_sql(eps)
    fresh_tables(cur)
    out = dict(arm=arm, eps=eps_key(eps))
    try:
        with Timer() as t:
            create(cur, arm, E)
        out.update(create_ms=t.ms, created=True)
    except psycopg2.Error as e:
        out.update(created=False, error_message=errmsg(e))
        return out
    ivm = arm.startswith("softavg_ivm")
    fresh_q = (FRESH_SINGLE if arm == "softavg_ivm_single" else FRESH).replace("{E}", E)
    cur.execute("SELECT id, count(*) FROM m_genre GROUP BY id"); fan = defaultdict(int, dict(cur.fetchall()))
    cur.execute("SELECT softavg_ivm_stats_reset()")
    per_op = []  # (kind, fanout, ms, recomputed)
    cps = []; prev = 0
    for k, op in enumerate(ops, 1):
        f = fan[op[1]] if op[0] in ("delete", "insert", "update") else 1
        with Timer() as t:
            apply(cur, op, rows)
        rec = 0
        if ivm:
            rec = stats(cur)[1] - prev; prev += rec
        if op[0] == "bridge_insert":
            fan[op[1]] += 1
        elif op[0] == "bridge_delete":
            fan[op[1]] -= 1
        per_op.append((op[0], f, t.ms, rec))
        if k % C == 0 or k == len(ops):
            cp = dict(after_ops=k)
            with Timer() as tf:
                cur.execute(fresh_q); ref = cur.fetchall()
            cp["fresh_query_ms"] = tf.ms
            if arm == "refresh_mv":
                cur.execute(READ[arm]); cp["stale_before_refresh"] = drift(cur.fetchall(), ref)
                with Timer() as tr:
                    cur.execute("REFRESH MATERIALIZED VIEW mv_refresh")
                cp["refresh_ms"] = tr.ms
            with Timer() as tr:
                cur.execute(READ[arm].replace("{E}", E)); got = cur.fetchall()
            cp["read_ms"] = tr.ms
            cp.update(drift(got, ref)); cps.append(cp)
    kinds = sorted({p[0] for p in per_op})
    out["ops"] = {kd: dict(pct([p[2] for p in per_op if p[0] == kd]),
                           recomputed=int(sum(p[3] for p in per_op if p[0] == kd)),
                           membership_deltas=int(sum(p[1] for p in per_op if p[0] == kd))) for kd in kinds}
    out["ops"]["all"] = dict(pct([p[2] for p in per_op]), recomputed=int(sum(p[3] for p in per_op)))
    base_ops = [p for p in per_op if p[0] in ("delete", "insert", "update")]
    out["base_ops_by_fanout"] = {int(f): dict(pct([p[2] for p in base_ops if p[1] == f]),
                                              recomputed=int(sum(p[3] for p in base_ops if p[1] == f)))
                                 for f in sorted({p[1] for p in base_ops})}
    out["membership_deltas_total"] = int(sum(p[1] for p in per_op))
    out["checkpoints"] = cps
    out["groups_recomputed_total"] = int(sum(p[3] for p in per_op))
    out["max_abs_err_over_vscale"] = max((c["max_abs_err_over_vscale"] or 0) for c in cps)
    out["max_rel_err"] = max((c["max_rel_err"] or 0) for c in cps)
    out["group_mismatch_total"] = sum(c["group_mismatch"] for c in cps)
    out["read_ms_median"] = float(np.median([c["read_ms"] for c in cps]))
    out["fresh_query_ms_median"] = float(np.median([c["fresh_query_ms"] for c in cps]))
    if arm == "refresh_mv":
        out["refresh_ms_median"] = float(np.median([c["refresh_ms"] for c in cps]))
        out["stale_max_abs_err_over_vscale"] = max((c["stale_before_refresh"]["max_abs_err_over_vscale"] or 0) for c in cps)
    if arm in ("softavg_ivm_join", "softavg_ivm_single", "pg_ivm_linear_join"):
        out["fanout_sweep"] = fanout_sweep(cur, arm, E, ivm, rows)
    drop(cur)
    return out


def fanout_sweep(cur, arm, E, ivm, rows):
    """delete + re-insert up to FS live items per exact fan-out f (this arm's tables, after the stream)."""
    cur.execute("SELECT b.id, count(g.id) FROM m_base b LEFT JOIN m_genre g ON g.id = b.id GROUP BY b.id")
    by_f = defaultdict(list)
    for i, f in cur.fetchall():
        by_f[int(f)].append(int(i))
    rng = np.random.RandomState(13)
    # FS items per exact fan-out, visited in ONE global random order so that
    # machine-load drift (shared box) is not confounded with fan-out
    work = []
    for f in sorted(by_f):
        pool = sorted(by_f[f])
        work += [(f, pool[x]) for x in rng.permutation(len(pool))[:FS]]
    work = [work[x] for x in rng.permutation(len(work))]
    lat = defaultdict(lambda: {"delete": [], "insert": []}); rec = defaultdict(lambda: {"delete": 0, "insert": 0})
    prev = stats(cur)[1] if ivm else 0
    for f, i in work:
        cur.execute("SELECT value, s0, facet FROM m_base WHERE id = %s", (i,)); v, s, fc = cur.fetchone()
        for kd, sql, args in (("delete", "DELETE FROM m_base WHERE id = %s", (i,)),
                              ("insert", "INSERT INTO m_base(id, facet, value, s0) VALUES (%s, %s, %s, %s)", (i, fc, v, s))):
            with Timer() as t:
                cur.execute(sql, args)
            lat[f][kd].append(t.ms)
            if ivm:
                st = stats(cur)[1]; rec[f][kd] += st - prev; prev = st
    res = {f: dict(items=len(lat[f]["delete"]), pool=len(by_f[f]),
                   delete=dict(pct(lat[f]["delete"]), recomputed=int(rec[f]["delete"])),
                   insert=dict(pct(lat[f]["insert"]), recomputed=int(rec[f]["insert"]))) for f in sorted(lat)}
    # least-squares slope of per-edit latency vs fan-out (f >= 1), all samples
    for kd in ("delete", "insert"):
        xs = np.array([f for f in lat if f >= 1 for _ in lat[f][kd]], float)
        ys = np.array([y for f in lat if f >= 1 for y in lat[f][kd]], float)
        if len(xs) > 2:
            b, a = np.polyfit(xs, ys, 1)
            res[f"{kd}_fit_ms"] = dict(intercept=float(a), per_membership=float(b))
    # the sweep restores every row, so the view must still equal a fresh query
    fq = (FRESH_SINGLE if arm == "softavg_ivm_single" else FRESH).replace("{E}", E)
    cur.execute(fq); ref = cur.fetchall(); cur.execute(READ[arm]); got = cur.fetchall()
    fits = {k: res.pop(k) for k in list(res) if isinstance(k, str)}
    return dict(per_fanout=res, fit=fits, after_sweep=drift(got, ref))


def transactional_checks(cur):
    E = eps_sql(1e-3)
    res = {}
    fresh_tables(cur)
    create(cur, "softavg_ivm_join", E)

    def agrees(c):
        c.execute(READ["softavg_ivm_join"]); got = c.fetchall()
        c.execute(FRESH.replace("{E}", E)); ref = c.fetchall()
        d = drift(got, ref)
        return d["group_mismatch"] == 0 and (d["max_abs_err_over_vscale"] or 0) <= 1e-9, d

    snap = "SELECT string_agg(g || ':' || encode(state, 'hex'), ',' ORDER BY g) FROM mv_j_state"
    cur.execute(snap); before = cur.fetchone()[0]
    a = connect("rel", autocommit=False); ca = a.cursor()
    ca.execute("DELETE FROM m_base WHERE id IN (SELECT DISTINCT ON (g.facet) b.id FROM m_base b JOIN m_genre g "
               "USING (id) ORDER BY g.facet, b.s0 DESC)")
    ca.execute("INSERT INTO m_genre SELECT id, 'Drama' FROM m_base WHERE id % 50 = 0")
    a.rollback(); a.close()
    cur.execute(snap); res["rollback_state_bytes_identical"] = cur.fetchone()[0] == before
    res["rollback_view_agrees"] = agrees(cur)[0]

    # base writer T1 inserts a new dominant item; bridge writer T2 gives it a membership.
    # Without serialization each trigger misses the other's uncommitted row.
    for n, order in enumerate(("base_first", "bridge_first")):
        new_id = 9_200_000 + n
        t1 = connect("rel", autocommit=False); c1 = t1.cursor()
        t2 = connect("rel", autocommit=False); c2 = t2.cursor()
        s1 = ("INSERT INTO m_base VALUES (%s, 'x', 9.9, 1.5)", (new_id,))
        s2 = ("INSERT INTO m_genre VALUES (%s, 'Drama')", (new_id,))
        first, second = ((c1, s1), (c2, s2)) if order == "base_first" else ((c2, s2), (c1, s1))
        first[0].execute(*first[1])
        th = threading.Thread(target=lambda: second[0].execute(*second[1]))
        th.start(); time.sleep(0.5); blocked = th.is_alive()
        first[0].connection.commit(); th.join(); second[0].connection.commit()
        t1.close(); t2.close()
        ok, d = agrees(cur)
        cur.execute("SELECT n FROM mv_j WHERE facet = 'Drama'"); n_view = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM m_base b JOIN m_genre g USING (id) WHERE g.facet = 'Drama'"); n_fresh = cur.fetchone()[0]
        res[order] = dict(second_session_blocked=blocked, view_agrees_after_both_commit=ok, drama_n_view=n_view,
                          drama_n_fresh=n_fresh, drift=d)
    # REPEATABLE READ base writer whose snapshot predates a committed bridge write
    new_id = 9_300_000
    t1 = connect("rel", autocommit=False); c1 = t1.cursor()
    c1.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"); c1.execute("SELECT 1 FROM m_base LIMIT 1")
    cur.execute("INSERT INTO m_genre VALUES (%s, 'Comedy')", (new_id,))
    try:
        c1.execute("INSERT INTO m_base VALUES (%s, 'x', 5.0, 0.9)", (new_id,)); t1.commit()
        res["rr_base_vs_committed_bridge"] = dict(outcome="committed")
    except psycopg2.Error as e:
        t1.rollback(); res["rr_base_vs_committed_bridge"] = dict(outcome="error", sqlstate=e.pgcode, message=errmsg(e))
    t1.close()
    res["rr_base_vs_committed_bridge"]["view_agrees"] = agrees(cur)[0]
    t1 = connect("rel", autocommit=False); c1 = t1.cursor()
    c1.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    try:
        c1.execute("INSERT INTO m_genre VALUES (1, 'Western')"); t1.commit()
        res["rr_bridge_write"] = dict(outcome="committed")
    except psycopg2.Error as e:
        t1.rollback(); res["rr_bridge_write"] = dict(outcome="error", sqlstate=e.pgcode, message=errmsg(e))
    t1.close()
    res["rr_bridge_write"]["view_agrees"] = agrees(cur)[0]
    drop(cur)
    return res


def main():
    conn = connect("rel"); cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET synchronous_commit = off")
    ops, rows, argmax = make_stream(cur)
    out = dict(timing_status=TIMING_STATUS, server="rel", server_env=server_env("rel"), machine_at_start=machine_state(),
               stream=dict(n_ops=len(ops), checkpoint_every=C, fanout_sweep_items_per_f=FS,
                           first_ops_delete_argmax_items=len(argmax),
                           mix={k: sum(o[0] == k for o in ops) for k in ("delete", "insert", "update", "bridge_insert", "bridge_delete")},
                           commit="autocommit, one statement per op, synchronous_commit=off in the stream session",
                           seed=11, v_scale=V_SCALE),
               arms=[])
    out["transactional_checks"] = transactional_checks(cur)
    print(json.dumps(out["transactional_checks"]), flush=True)
    for eps in EPS:
        for arm in ARMS:
            r = run_arm(cur, arm, eps, ops, rows)
            out["arms"].append(r)
            print(json.dumps({k: r.get(k) for k in ("arm", "eps", "created", "error_message", "groups_recomputed_total",
                                                     "max_abs_err_over_vscale", "read_ms_median")}),
                  json.dumps({k: v.get("median_ms") for k, v in (r.get("ops") or {}).items()}), flush=True)
            drop(cur)
    cur.execute("DROP TABLE IF EXISTS m_base, m_genre CASCADE")
    out["machine_at_end"] = machine_state()
    (HERE / "results_maint_join.json").write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()

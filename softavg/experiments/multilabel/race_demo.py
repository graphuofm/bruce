#!/usr/bin/env python
"""Why softavg_ivm_create_join serializes base writers against bridge writers.

Creates the join view on small copies of imdb / imdb_genre, then runs the
two-session race of maint_join.py (T1 inserts a base row, T2 inserts its
bridge row, both commit) twice: with the generated trigger functions as
shipped, and with the advisory-lock / sync lines stripped from them
(CREATE OR REPLACE of the same function text minus those lines). Records,
per order, the view's membership count vs a fresh join.

Also records what softavg_ivm_create (single-table maintenance) does when
given a join view as its base (-> ivm_create_on_join_view.json).

usage: race_demo.py   -> race_demo.json, ivm_create_on_join_view.json
"""
import json, re, sys, threading, time
import psycopg2
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import connect  # noqa: E402

cur = connect("rel").cursor()


def setup():
    cur.execute("SELECT softavg_ivm_drop('mv_r'); DROP TABLE IF EXISTS r_base, r_genre CASCADE")
    cur.execute("CREATE TABLE r_base AS SELECT id, value, s0 FROM imdb WHERE id < 20000")
    cur.execute("ALTER TABLE r_base ADD PRIMARY KEY (id)")
    cur.execute("CREATE TABLE r_genre AS SELECT id, facet FROM imdb_genre WHERE id < 20000")
    cur.execute("CREATE INDEX ON r_genre(id)")
    cur.execute("SELECT softavg_ivm_create_join('mv_r', 'r_base', 'id', 'r_genre', 'id', 'facet', 'value', 's0', 0.1)")


def strip_locks():
    for fn in ("mv_r_maint_base", "mv_r_maint_bridge"):
        cur.execute("SELECT pg_get_functiondef(%s::regproc)", (fn,))
        src = cur.fetchone()[0]
        new = re.sub(r"\n\s*PERFORM pg_advisory_xact_lock(_shared)?\(\d+\);", "\n        NULL;", src)
        new = re.sub(r"\n\s*UPDATE \S+_sync SET ver = ver \+ 1 WHERE id = 1;", "\n        NULL;", new)
        assert new != src and "pg_advisory" not in new and "_sync" not in new
        cur.execute(new)


def race(order, new_id):
    t1 = connect("rel", autocommit=False); c1 = t1.cursor()
    t2 = connect("rel", autocommit=False); c2 = t2.cursor()
    s1 = ("INSERT INTO r_base VALUES (%s, 9.9, 1.5)", (new_id,))
    s2 = ("INSERT INTO r_genre VALUES (%s, 'Drama')", (new_id,))
    first, second = ((c1, s1), (c2, s2)) if order == "base_first" else ((c2, s2), (c1, s1))
    first[0].execute(*first[1])
    th = threading.Thread(target=lambda: second[0].execute(*second[1]))
    th.start(); time.sleep(0.5); blocked = th.is_alive()
    first[0].connection.commit(); th.join(); second[0].connection.commit()
    t1.close(); t2.close()
    cur.execute("SELECT n, softavg FROM mv_r WHERE facet = 'Drama'"); nv, xv = cur.fetchone()
    cur.execute("SELECT count(*), softavg(b.value, b.s0, 0.1) FROM r_base b JOIN r_genre g USING (id) WHERE g.facet = 'Drama'")
    nf, xf = cur.fetchone()
    return dict(second_blocked=blocked, drama_n_view=nv, drama_n_fresh=nf, drama_softavg_view=xv, drama_softavg_fresh=xf,
                view_agrees=(nv == nf and abs(xv - xf) <= 1e-9 * 10))


out = {}
for variant in ("as_shipped", "locks_stripped"):
    setup()
    if variant == "locks_stripped":
        strip_locks()
    out[variant] = {o: race(o, 9_500_000 + k) for k, o in enumerate(("base_first", "bridge_first"))}
cur.execute("SELECT softavg_ivm_drop('mv_r'); DROP TABLE IF EXISTS r_base, r_genre CASCADE")
(HERE / "race_demo.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

cur.execute("CREATE TEMP VIEW jv AS SELECT g.facet, i.value, i.s0 FROM imdb i JOIN imdb_genre g USING (id)")
iv = {}
try:
    cur.execute("SELECT softavg_ivm_create('mv_jv', 'jv', 'facet', 'value', 's0', 0.1)")
    iv["result"] = "created"
except psycopg2.Error as e:
    iv.update(result="error", sqlstate=e.pgcode, message=e.pgerror.strip().splitlines()[0])
(HERE / "ivm_create_on_join_view.json").write_text(json.dumps(iv, indent=1))
print(json.dumps(iv))

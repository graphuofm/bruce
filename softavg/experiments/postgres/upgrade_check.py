#!/usr/bin/env python
"""Upgrade path 0.1.2 -> 0.2.0, in-database, on one server (needs
bruce_pg--0.1.2.sql in the server's sharedir; see README "Rerun").

1. db upg_old: CREATE EXTENSION bruce_pg VERSION '0.1.2'; a plain view and a
   materialized view over softavg; record their values; ALTER EXTENSION
   bruce_pg UPDATE TO '0.2.0'; the views must survive and return the same
   bits, the aggregate must now have an internal state + moving form, and
   REFRESH MATERIALIZED VIEW must reproduce the stored values.
2. db upg_new: fresh CREATE EXTENSION bruce_pg (0.2.0).
3. Catalog diff between the two: every extension member (functions with
   language/strictness/volatility/parallel/symbol, aggregates with all
   pg_aggregate support columns, comments). Must be empty.
Writes upgrade_<server>.json; drops both databases.

usage: upgrade_check.py rel|dbg
"""
import json, sys
import psycopg2
from common import HERE, SERVERS

SERVER = sys.argv[1]


def conn(db):
    c = psycopg2.connect(host="127.0.0.1", port=SERVERS[SERVER]["port"], dbname=db, user="jding")
    c.autocommit = True
    return c


CATALOG = """
SELECT 'function ' || p.oid::regprocedure::text,
       json_build_object('rettype', p.prorettype::regtype::text, 'lang', l.lanname, 'strict', p.proisstrict,
                         'volatile', p.provolatile, 'parallel', p.proparallel, 'symbol', p.prosrc,
                         'kind', p.prokind, 'comment', obj_description(p.oid, 'pg_proc'))::text
FROM pg_depend d JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'bruce_pg'
JOIN pg_proc p ON p.oid = d.objid AND d.classid = 'pg_proc'::regclass
JOIN pg_language l ON l.oid = p.prolang
WHERE d.deptype = 'e'
UNION ALL
SELECT 'aggregate ' || a.aggfnoid::regprocedure::text,
       json_build_object('transfn', a.aggtransfn::text, 'finalfn', a.aggfinalfn::text, 'combinefn', a.aggcombinefn::text,
                         'serialfn', a.aggserialfn::text, 'deserialfn', a.aggdeserialfn::text,
                         'mtransfn', a.aggmtransfn::text, 'minvtransfn', a.aggminvtransfn::text,
                         'mfinalfn', a.aggmfinalfn::text, 'finalextra', a.aggfinalextra, 'mfinalextra', a.aggmfinalextra,
                         'finalmodify', a.aggfinalmodify, 'mfinalmodify', a.aggmfinalmodify, 'sortop', a.aggsortop,
                         'transtype', a.aggtranstype::regtype::text, 'transspace', a.aggtransspace,
                         'mtranstype', a.aggmtranstype::regtype::text, 'mtransspace', a.aggmtransspace,
                         'initval', a.agginitval, 'minitval', a.aggminitval)::text
FROM pg_aggregate a JOIN pg_depend d ON d.objid = a.aggfnoid AND d.classid = 'pg_proc'::regclass AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'bruce_pg'
ORDER BY 1
"""


def main():
    out = dict(server=SERVER)
    admin = conn("postgres")
    with admin.cursor() as c:
        for db in ("upg_old", "upg_new"):
            c.execute(f"DROP DATABASE IF EXISTS {db}"); c.execute(f"CREATE DATABASE {db}")
    old = conn("upg_old"); co = old.cursor()
    co.execute("CREATE EXTENSION bruce_pg VERSION '0.1.2'")
    co.execute("CREATE TABLE t AS SELECT i % 7 AS g, sin(i::float8) * 5 AS v, cos(i * 0.3) AS s FROM generate_series(1, 20000) i")
    co.execute("CREATE VIEW v AS SELECT g, softavg(v, s, 0.05) AS x, softavg_state(v, s, 0.05) AS st FROM t GROUP BY g")
    co.execute("CREATE MATERIALIZED VIEW mv AS SELECT g, softavg(v, s, 0.001) AS x FROM t GROUP BY g")
    co.execute("SELECT g, x, st FROM v ORDER BY g"); before_v = co.fetchall()
    co.execute("SELECT g, x FROM mv ORDER BY g"); before_mv = co.fetchall()
    co.execute("SELECT aggtranstype::regtype::text FROM pg_aggregate WHERE aggfnoid = 'softavg(float8,float8,float8)'::regprocedure")
    out["stype_before"] = co.fetchone()[0]
    co.execute("ALTER EXTENSION bruce_pg UPDATE TO '0.2.0'")
    co.execute("SELECT extversion FROM pg_extension WHERE extname = 'bruce_pg'"); out["extversion_after"] = co.fetchone()[0]
    co.execute("SELECT aggtranstype::regtype::text, aggmtransfn::text, aggminvtransfn::text, aggserialfn::text "
               "FROM pg_aggregate WHERE aggfnoid = 'softavg(float8,float8,float8)'::regprocedure")
    out["after"] = dict(zip(["stype", "mtransfn", "minvtransfn", "serialfn"], co.fetchone()))
    co.execute("SELECT g, x, st FROM v ORDER BY g"); after_v = co.fetchall()
    co.execute("SELECT g, x FROM mv ORDER BY g"); stored_mv = co.fetchall()
    co.execute("REFRESH MATERIALIZED VIEW mv")
    co.execute("SELECT g, x FROM mv ORDER BY g"); refreshed_mv = co.fetchall()
    co.execute("SELECT count(x) FROM (SELECT softavg(v, s, 0.05) OVER (ORDER BY g, v ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS x FROM t) q")
    out["window_after_upgrade_rows"] = co.fetchone()[0]
    co.execute("SELECT inverses FROM softavg_window_stats()"); out["window_after_upgrade_inverses"] = co.fetchone()[0]
    out["view_bits_identical"] = before_v == after_v
    out["mv_stored_identical"] = before_mv == stored_mv
    out["mv_refresh_bits_identical"] = before_mv == refreshed_mv
    out["n_groups"] = len(before_v)
    co.execute(CATALOG); cat_old = dict(co.fetchall())
    old.close()
    new = conn("upg_new"); cn = new.cursor()
    cn.execute("CREATE EXTENSION bruce_pg")
    cn.execute(CATALOG); cat_new = dict(cn.fetchall())
    new.close()
    diff = {k: dict(upgraded=cat_old.get(k), fresh=cat_new.get(k)) for k in sorted(set(cat_old) | set(cat_new))
            if cat_old.get(k) != cat_new.get(k)}
    out["catalog_objects"] = len(cat_new)
    out["catalog_diff"] = diff
    out["pass"] = (out["view_bits_identical"] and out["mv_stored_identical"] and out["mv_refresh_bits_identical"]
                   and out["after"]["stype"] == "internal" and out["extversion_after"] == "0.2.0" and not diff
                   and out["window_after_upgrade_inverses"] > 0)
    with admin.cursor() as c:
        for db in ("upg_old", "upg_new"):
            c.execute(f"DROP DATABASE {db}")
    (HERE / f"upgrade_{SERVER}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

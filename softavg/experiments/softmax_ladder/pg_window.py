#!/usr/bin/env python
"""Softmax ladder, sliding windows, PostgreSQL 18.6 (conda env ggdb-pg, release build; cluster
paper_icde/pg18env, port 28860; extensions staged in pg18env/ext, see pg18env/README.md).

Same grid, data, reference and metrics as sqlite_window.py. Tables lad_<dataset>(rn int PRIMARY KEY,
id int, value float8, s0, s1, s2 float8) loaded from $LADDER_WORK/<dataset>.npz by binary COPY.
Rungs:
  S0  SUM(EXP(s/eps)*value) OVER w / SUM(EXP(s/eps)) OVER w            (PG float8 sum: no inverse, frame recompute;
                                                                        PG's exp() raises on overflow AND underflow)
  S1  global-max shift with the clamp PG needs: EXP(GREATEST((s-g)/eps, -700))
  S2  softavg_fwd(...) OVER w   (bruce_pg's internal state without a moving form: PG restarts every frame)
  S3  softavg_s3(...) OVER w    (softavg_ablation: MINVFUNC subtracts, never NULL)
  S4  softavg(...) OVER w       (shipped bruce_pg 0.2.0: MINVFUNC NULL when the certificate fails)
Correctness: every query, one fetch of all rows vs the numpy float64 two-pass per-frame reference.
Latency: query 0 only, SELECT count(x), sum(x) FROM (...), median of 3 after the correctness fetch
(1 run when the fetch took > 20 s, as sqlite_window.py).
A query that raises (S0: exp overflow / underflow) is recorded with its SQLSTATE and message.

usage: pg_window.py [--port 28860] [--datasets ...] [--ks ...] [--eps ...] [--load-only] [--out FILE]   (resumes: cells already in FILE are skipped)
"""
import argparse, io, json, struct
import numpy as np
import psycopg2
from common import EPS, HERE, KS, NQ, TIMING_STATUS, Timer, errors, load, machine_state, reference, vmax_frames

RUNGS = {
    "S0": "SUM(EXP({s}/{E})*value) OVER w / SUM(EXP({s}/{E})) OVER w",
    "S1": "SUM(EXP(GREATEST(({s}-g)/{E}, -700))*value) OVER w / SUM(EXP(GREATEST(({s}-g)/{E}, -700))) OVER w",
    "S2": "softavg_fwd(value, {s}, {E}) OVER w",
    "S3": "softavg_s3(value, {s}, {E}) OVER w",
    "S4": "softavg(value, {s}, {E}) OVER w",
}
HARNESS = """
CREATE EXTENSION IF NOT EXISTS bruce_pg; CREATE EXTENSION IF NOT EXISTS softavg_ablation;
DO $$ BEGIN
  IF to_regprocedure('softavg_fwd(float8, float8, float8)') IS NULL THEN
    CREATE AGGREGATE softavg_fwd(float8, float8, float8) (
      SFUNC = softavg_sfunc, STYPE = internal, SSPACE = 40, COMBINEFUNC = softavg_combine,
      SERIALFUNC = softavg_serialize, DESERIALFUNC = softavg_deserialize, FINALFUNC = softavg_finalfn, PARALLEL = SAFE);
  END IF;
END $$;
"""


def copy_table(cur, ds, d):
    cur.execute(f"DROP TABLE IF EXISTS lad_{ds}")
    cur.execute(f"CREATE TABLE lad_{ds}(rn int, id int, value float8, s0 float8, s1 float8, s2 float8)")
    buf = io.BytesIO(); buf.write(b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0))
    for i in range(len(d["id"])):
        buf.write(struct.pack(">hiiii", 6, 4, i, 4, int(d["id"][i])))
        for c in ("value", "s0", "s1", "s2"):
            buf.write(struct.pack(">id", 8, float(d[c][i])))
    buf.write(struct.pack(">h", -1)); buf.seek(0)
    cur.copy_expert(f"COPY lad_{ds} FROM STDIN WITH (FORMAT binary)", buf)
    cur.execute(f"ALTER TABLE lad_{ds} ADD PRIMARY KEY (rn)")
    cur.execute(f"VACUUM ANALYZE lad_{ds}")  # separate call: VACUUM cannot run in the implicit transaction of a multi-statement string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=28860)
    ap.add_argument("--datasets", nargs="+", default=["imdb_all", "imdb_ge", "amazon"])
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--eps", nargs="+", type=float, default=EPS)
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--out", default=str(HERE / "pg_window.jsonl"))
    a = ap.parse_args()
    conn = psycopg2.connect(host="127.0.0.1", port=a.port, dbname="ladder", user="jding"); conn.autocommit = True
    cur = conn.cursor(); cur.execute(HARNESS); cur.execute("SET max_parallel_workers_per_gather = 0")
    done = set()
    try:
        for line in open(a.out):
            r = json.loads(line)
            if "rung" in r:
                done.add((r["dataset"], r["k"], r["eps"], r["q"], r["rung"]))
    except FileNotFoundError:
        pass
    with open(a.out, "a") as f:
        cur.execute("SELECT version()"); ver = cur.fetchone()[0]
        cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1"); exts = dict(cur.fetchall())
        f.write(json.dumps(dict(header=True, version=ver, extensions=exts, datasets=a.datasets,
                                machine_at_start=machine_state(), timing_status=TIMING_STATUS)) + "\n")
        for ds in a.datasets:
            d = load(ds)
            cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"lad_{ds}",))
            have = cur.fetchone()[0]
            if have:
                cur.execute(f"SELECT count(*) FROM lad_{ds}"); have = cur.fetchone()[0] == len(d["id"])
            if not have:
                copy_table(cur, ds, d)
            if a.load_only:
                continue
            for k in a.ks:
                vmax = vmax_frames(d["value"], k)
                W = f"WINDOW w AS (ORDER BY rn ROWS BETWEEN {k} PRECEDING AND CURRENT ROW)"
                for eps in a.eps:
                    for q in range(NQ):
                        if all((ds, k, eps, q, r) in done for r in RUNGS):
                            continue
                        ref = reference(d["value"], d[f"s{q}"], eps, k)
                        for rung, expr in RUNGS.items():
                            if (ds, k, eps, q, rung) in done:
                                continue
                            body = (f"SELECT rn, {expr.format(s=f's{q}', E=repr(float(eps)) + '::float8')} AS x "
                                    f"FROM (SELECT *, (SELECT max(s{q}) FROM lad_{ds}) AS g FROM lad_{ds}) t {W}")
                            cell = dict(engine="postgres", dataset=ds, k=k, eps=eps, q=q, rung=rung, error=None)
                            try:
                                cur.execute("SELECT softavg_window_stats_reset(); SELECT softavg_s3_window_stats_reset()")
                                with Timer() as tf:
                                    cur.execute(body + " ORDER BY rn")
                                    rows = cur.fetchall()
                                cell["fetch_ms"] = tf.ms
                                got = np.array([np.nan if r[1] is None else r[1] for r in rows], dtype=np.float64)
                                cell.update(errors(got, ref, vmax))
                                cur.execute("SELECT * FROM softavg_window_stats()")
                                cell["stats"] = dict(zip(["fresh_states", "absorbs", "inverses", "punts"], cur.fetchone()))
                                if rung == "S3":
                                    cur.execute("SELECT * FROM softavg_s3_window_stats()")
                                    cell["s3_stats"] = dict(zip(["absorbs", "inverses"], cur.fetchone()))
                                if q == 0:
                                    ts = []
                                    for _ in range(1 if tf.ms > 20_000 else 3):
                                        with Timer() as t:
                                            cur.execute(f"SELECT count(x), sum(x) FROM ({body}) z"); cur.fetchall()
                                        ts.append(t.ms)
                                    cell.update(timed_ms=ts, median_ms=float(np.median(ts)))
                            except psycopg2.Error as e:
                                cell["error"] = (e.pgerror or str(e)).strip().splitlines()[0]
                                cell["sqlstate"] = e.pgcode
                            f.write(json.dumps(cell) + "\n"); f.flush()
                            print(ds, k, eps, q, rung, cell.get("median_ms"), cell.get("n_wrong"), cell["error"], flush=True)
        f.write(json.dumps(dict(footer=True, machine_at_end=machine_state())) + "\n")


if __name__ == "__main__":
    main()

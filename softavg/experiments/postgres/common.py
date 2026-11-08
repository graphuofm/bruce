"""Shared constants and helpers for the PostgreSQL before/after harness.

Every arm runs on the RELEASE PostgreSQL 17.10 binary
(~/.pgrx/17.10/pg-release-install, server "rel") with pgvector 0.8.7 and
pg_ivm 1.16 built for it, and the RELEASE build of bruce_pg 0.2.0
(cargo pgrx install --release). The "dbg" entry (pgrx's cassert binary) is
kept only so an older run can be reproduced; no reported number uses it.

Data directories are scratch, created by the operator (README "Rerun") and
deleted afterwards.
"""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import psycopg2

HERE = Path(__file__).resolve().parent
ICDE = HERE.parents[1]
CORP = Path("/data/softavg/corpora/imdb")
ENGINE = ICDE / "engines" / "postgres"
CORE = ICDE / "engines" / "core"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402  (require_quiet / cpu_busy_fraction)

SERVERS = {
    "rel": dict(port=28840, bin=Path.home() / ".pgrx/17.10/pg-release-install/bin"),
    "dbg": dict(port=28841, bin=Path.home() / ".pgrx/17.10/pgrx-install/bin"),
}
DBNAME = "icde_pg"

EPS_GRID = [1e-4, 1e-3, 0.02, 0.1, 1.0, "inf"]
WORKERS = [0, 8]
FILTERS = {"none": "", "ge2000": "WHERE filter_num >= 2000"}
TOPK = [100, 1000, 10000]
TIMING_STATUS = "smoke, machine shared"


def connect(server, dbname=DBNAME, autocommit=True):
    c = psycopg2.connect(host="127.0.0.1", port=SERVERS[server]["port"], dbname=dbname, user="jding")
    c.autocommit = autocommit
    return c


def eps_sql(eps):
    return "'infinity'::float8" if eps == "inf" else f"{float(eps)!r}::float8"


def eps_key(eps):
    return "inf" if eps == "inf" else repr(float(eps))


def vec_literal(q):
    return "'[" + ",".join(repr(float(x)) for x in q) + "]'::vector(384)"


def load_queries():
    return json.loads((HERE / "queries.json").read_text())


def query_embeddings(idx):
    emb = np.load(CORP / "emb.npy", mmap_mode="r")
    return np.asarray(emb[np.asarray(idx)], dtype=np.float32)


def machine_state():
    """Machine load at the time of a run (PROTOCOL: quiet.require_quiet).
    The machine is shared with other agents, so a refusal is recorded, not fatal."""
    try:
        st = quiet.require_quiet(wait_seconds=0, verbose=False)
        st["quiet"] = True
    except SystemExit as e:
        st = dict(quiet=False, refusal=str(e), load_average_1min=round(os.getloadavg()[0], 2))
    st["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return st


def source_hash():
    """sha256 over the extension's and softagg-core's sources (paper_icde is not a git repo)."""
    h = hashlib.sha256()
    files = sorted(list((ENGINE / "src").glob("*.rs")) + list((ENGINE / "sql").glob("*.sql"))
                   + [ENGINE / "Cargo.toml", ENGINE / "Cargo.lock", ENGINE / "bruce_pg.control"]
                   + sorted((CORE / "src").glob("*.rs")) + [CORE / "Cargo.toml"])
    for f in files:
        h.update(str(f.relative_to(ICDE)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def server_env(server):
    b = SERVERS[server]["bin"]
    conf = subprocess.run([str(b / "pg_config"), "--configure"], capture_output=True, text=True).stdout.strip()
    so = b.parent / "lib/postgresql/bruce_pg.so"
    out = dict(binary=str(b / "postgres"), pg_config_configure=conf,
               server_build=("debug (cassert, RANDOMIZE_ALLOCATED_MEMORY)" if "enable-cassert" in conf else "release"),
               bruce_pg_so_md5=hashlib.md5(so.read_bytes()).hexdigest())
    try:
        with connect(server) as c, c.cursor() as cur:
            cur.execute("SELECT version()"); out["pg_version"] = cur.fetchone()[0]
            cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1")
            out["extensions"] = dict(cur.fetchall())
            for g in ["shared_buffers", "work_mem", "max_parallel_workers", "max_worker_processes",
                      "jit", "debug_assertions", "port", "data_directory", "shared_preload_libraries"]:
                cur.execute(f"SHOW {g}"); out[g] = cur.fetchone()[0]
            cur.execute("SELECT softavg_core_version()"); out["softagg_core"] = cur.fetchone()[0]
    except Exception as e:  # pragma: no cover
        out["error"] = str(e)
    return out


def environment():
    return dict(
        host=dict(cpu=subprocess.run("lscpu | sed -n 's/Model name: *//p'", shell=True, capture_output=True, text=True).stdout.strip(),
                  nproc=os.cpu_count(), kernel=platform.release(),
                  rustc=subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
                  python=platform.python_version()),
        extension=dict(name="bruce_pg", version="0.2.0", build_type="release",
                       build_cmd="cargo pgrx install --release --pg-config <server>/bin/pg_config (pgrx 0.18.1; profile.release opt-level=3, lto=fat, codegen-units=1)",
                       source_sha256=source_hash(), softagg_core_version="0.1.0 (path ../core)",
                       origin="bruce_tool/bruce-pg @ 51b62f1 (0.1.2), copied 2026-11-05"),
    )


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1e3


# ------------------------------------------------------------- reference
def grouped_reference(keys, v, s, eps):
    """numpy float64 max-shifted softavg per group (PROTOCOL: numpy f64)."""
    out = {}
    order = np.argsort(keys, kind="stable")
    k, vv, ss = keys[order], v[order], s[order]
    uniq, starts = np.unique(k, return_index=True)
    ends = list(starts[1:]) + [len(k)]
    for g, a, b in zip(uniq, starts, ends):
        if eps == "inf":
            out[str(g)] = float(np.mean(vv[a:b]))
        else:
            m = ss[a:b].max()
            w = np.exp((ss[a:b] - m) / float(eps))
            out[str(g)] = float(np.dot(w, vv[a:b]) / w.sum())
    return out


def compare(rows, ref):
    """rows: [(g, value, ...)], ref {g: float} -> correctness record."""
    got = {str(r[0]): r[1] for r in rows}
    n_null = sum(v is None for v in got.values())
    n_nan = sum(v is not None and v != v for v in got.values())
    n_inf = sum(v is not None and v in (float("inf"), float("-inf")) for v in got.values())
    errs = []
    for g, x in got.items():
        if g not in ref or x is None or x != x or x in (float("inf"), float("-inf")):
            continue
        r = ref[g]
        errs.append(abs(x - r) / abs(r) if r != 0 else abs(x - r))
    return dict(n_groups=len(got), n_groups_ref=len(ref), coverage=sum(g in ref for g in got) / max(1, len(ref)),
                n_null=n_null, n_nan=n_nan, n_inf=n_inf, n_compared=len(errs),
                max_rel_err=(max(errs) if errs else None))


def plan_info(cur, sql):
    """One EXPLAIN (ANALYZE, TIMING OFF): plan shape + workers launched + execution ms."""
    cur.execute("EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) " + sql)
    j = cur.fetchone()[0][0]
    shape, info = [], {"workers_launched": 0}

    def walk(n):
        t = n["Node Type"]
        if n.get("Strategy") and n["Strategy"] != "Plain":
            t = n["Strategy"] + t
        if n.get("Partial Mode") and n["Partial Mode"] != "Simple":
            t = n["Partial Mode"] + " " + t
        if "Workers Launched" in n:
            t += f"(w={n['Workers Launched']})"
            info["workers_launched"] = max(info["workers_launched"], n["Workers Launched"])
        if n.get("Index Name"):
            t += f"<{n['Index Name']}>"
        shape.append(t)
        for c in n.get("Plans", []):
            walk(c)
    walk(j["Plan"])
    return dict(plan_shape=" > ".join(shape), workers_launched=info["workers_launched"],
                explain_exec_ms=j["Execution Time"], jit=("JIT" in j))


def run_timed(cur, sql, n_warm=1, n_runs=3, settings=()):
    """Warm-ups + timed runs (client wall ms, median). Errors are recorded verbatim."""
    out = dict(sql_ok=True)
    try:
        for s in settings:
            cur.execute(s)
        for _ in range(n_warm):
            cur.execute(sql); rows = cur.fetchall()
        walls = []
        for _ in range(n_runs):
            with Timer() as t:
                cur.execute(sql); rows = cur.fetchall()
            walls.append(t.ms)
        out.update(rows=rows, wall_ms_all=walls, median_ms=float(np.median(walls)))
        out.update(plan_info(cur, sql))
    except psycopg2.Error as e:
        out.update(sql_ok=False, error_message=(e.pgerror or str(e)).strip().splitlines()[0],
                   error_detail=(e.pgerror or str(e)).strip(), sqlstate=e.pgcode)
    return out

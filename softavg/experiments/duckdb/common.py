"""Shared helpers for the DuckDB experiments: provenance metadata, machine
load, CPU pinning, instance setup, error metrics."""
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ICDE = os.path.abspath(os.path.join(HERE, "..", ".."))
ENG = os.path.join(ICDE, "engines")
SHIPPED = os.path.join(ENG, "duckdb", "build", "softavg.duckdb_extension")
BEFORE = "/home/jding/bruce/paper_softavg/experiments/duckdb_ext/build/softavg.duckdb_extension"
VARIANTS = {
    "pre_p1": os.path.join(HERE, "build", "pre_p1", "softavg.duckdb_extension"),
}
DIAG = os.path.join(HERE, "build", "diag", "softavg.duckdb_extension")
EPS_GRID = [1e-4, 1e-3, 0.02, 0.1, 1.0, math.inf]
TIMING_STATUS = "smoke, machine shared"

# i9-13900K: logical CPUs 0-15 are the 8 P-cores (2 threads each), 16-31 E-cores.
PIN = {1: [4], 8: [0, 2, 4, 6, 8, 10, 12, 14]}

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402  (read-only use of the shared load gate)


def eps_sql(e):
    return "'infinity'::DOUBLE" if math.isinf(e) else repr(float(e))


def eps_key(e):
    return "inf" if math.isinf(e) else repr(float(e))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tree(root, rel_paths):
    h = hashlib.sha256()
    for rp in sorted(rel_paths):
        h.update(rp.encode())
        h.update(open(os.path.join(root, rp), "rb").read())
    return h.hexdigest()


def src_files(root, sub=("src",), extra=("Cargo.toml",)):
    out = list(extra)
    for s in sub:
        for dp, _, fs in os.walk(os.path.join(root, s)):
            out += [os.path.relpath(os.path.join(dp, f), root) for f in fs if f.endswith(".rs")]
    return out


def cmd(c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def load_conditions():
    """quiet.require_quiet without the abort: other agents share this box, so
    timings are smoke numbers and the observed load is recorded instead."""
    try:
        c = quiet.require_quiet(wait_seconds=0, verbose=False)
        c["quiet"] = True
    except SystemExit as e:
        c = {"quiet": False, "require_quiet_message": str(e),
             "cpu_busy_fraction": round(quiet.cpu_busy_fraction(), 3),
             "load_average_1min": round(os.getloadavg()[0], 2)}
    c["at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return c


def ext_info(path):
    if not os.path.exists(path):
        return {"path": path, "missing": True}
    return {"path": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def meta():
    core = os.path.join(ENG, "core")
    ext = os.path.join(ENG, "duckdb")
    core_ver = [l.split('"')[1] for l in open(os.path.join(core, "Cargo.toml")) if l.startswith("version")][0]
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "timing_status": TIMING_STATUS,
        "host": platform.node(),
        "cpu": [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0],
        "nproc": os.cpu_count(),
        "duckdb_python": duckdb.__version__,
        "duckdb_library": duckdb.connect().execute("PRAGMA version").fetchall(),
        "build_type": "release (cargo --release: opt-level 3, lto fat, codegen-units 1, panic abort)",
        "extension": {
            "shipped": ext_info(SHIPPED),
            "source_sha256": sha256_tree(ext, src_files(ext, extra=("Cargo.toml", "Cargo.lock"))),
            "note": "no git repository; source hash = sha256 over engines/duckdb/{Cargo.toml,Cargo.lock,src/*.rs}",
        },
        "softagg_core": {"version": core_ver, "source_sha256": sha256_tree(core, src_files(core))},
        "before_extension": ext_info(BEFORE),
        "variants": {k: ext_info(v) for k, v in VARIANTS.items()},
        "rustc": cmd(["rustc", "--version"]),
        "pinning": {"threads_to_cpus": PIN, "note": "logical CPUs 0-15 = 8 P-cores x 2 SMT; 16-31 = E-cores"},
    }


def pin(threads):
    os.sched_setaffinity(0, PIN[threads])


def instance(ext=None, attach=(), threads=None, extra_load=()):
    """A fresh in-memory DuckDB instance (never shared through the instance
    cache) with `ext` loaded and data files ATTACHed read-only."""
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    if ext:
        con.execute(f"LOAD '{ext}'")
    for e in extra_load:
        con.execute(f"LOAD {e}")
    for alias, path in attach:
        con.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    return con


def rel_err(got, ref):
    if ref == 0:
        return abs(got)
    return abs(got - ref) / abs(ref)


def score_result(rows, ref, ref_same=None):
    """rows: [(key, value)]; ref/ref_same: {key: float}. Returns a dict with the
    silent-failure counts and the worst relative errors over finite results."""
    got = {k: v for k, v in rows}
    nan = sum(1 for v in got.values() if v is not None and math.isnan(v))
    inf = sum(1 for v in got.values() if v is not None and math.isinf(v))
    null = sum(1 for v in got.values() if v is None)
    missing = sorted(set(ref) - set(got))
    fin = {k: v for k, v in got.items() if v is not None and math.isfinite(v) and k in ref}
    out = {"n_groups": len(got), "nan_groups": nan, "inf_groups": inf, "null_groups": null,
           "missing_groups": len(missing),
           "max_rel_err_vs_f64": max((rel_err(v, ref[k]) for k, v in fin.items()), default=None)}
    if ref_same is not None:
        out["max_rel_err_vs_same_scores"] = max((rel_err(v, ref_same[k]) for k, v in fin.items()
                                                 if k in ref_same), default=None)
    return out


def clean(o):
    """JSON-safe: non-finite floats become the strings "nan" / "inf" / "-inf"."""
    if isinstance(o, float) and not math.isfinite(o):
        return "nan" if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, indent=1, allow_nan=False)
    os.replace(tmp, path)

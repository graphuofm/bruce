#!/usr/bin/env python
"""Add the PostgreSQL 18.6 engine run (pg_window.jsonl, pg_maint.json) to results.json under the separate key
`postgres_engine`, with a cell-by-cell comparison against the call-order replay already in results.json.
Leaves every other key of results.json untouched (assemble.py needs build artifacts that were deleted after the
SQLite run, so it is not rerun). Also refreshes results.json -> not_run from not_run.json.

Merges the per-process files (pg_window_p*.jsonl, pg_maint_<stream>.json) first when they exist.
Needs the pg18env server up (reads settings / extension versions).  usage: assemble_pg.py [--port 28860]
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import psycopg2
from common import HERE, ICDE, machine_state, sha256_file, sha256_tree

PG18ENV = ICDE / "pg18env"
SETTINGS = ["server_version", "shared_buffers", "work_mem", "maintenance_work_mem", "effective_cache_size",
            "max_worker_processes", "max_parallel_workers", "max_parallel_workers_per_gather", "jit",
            "shared_preload_libraries", "dynamic_library_path", "extension_control_path", "port"]


def jsonl(p):
    head, cells, foot = [], [], []
    for line in open(p):
        r = json.loads(line)
        (head if r.get("header") else foot if r.get("footer") else cells).append(r)
    return head, cells, foot


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def merge_parts():
    """pg_window_p*.jsonl -> pg_window.jsonl, pg_maint_<stream>.json -> pg_maint.json (when the per-process files exist)."""
    parts = sorted(HERE.glob("pg_window_p*.jsonl"))
    if parts:
        with open(HERE / "pg_window.jsonl", "w") as f:
            for p in parts:
                f.write(p.read_text())
    mparts = sorted(HERE.glob("pg_maint_*.json"))
    if mparts:
        runs, starts, ends, ver = [], [], [], None
        for p in mparts:
            d = json.loads(p.read_text()); runs += d["runs"]; starts.append(d["machine_at_start"])
            ends.append(d.get("machine_at_end")); ver = d["version"]
        order = ["maint_ivm_10k", "amazon_time_R30", "amazon_random_R30", "amazon_time_R2", "amazon_random_R2"]
        runs.sort(key=lambda r: (order.index(r["stream"]), r["eps"], r["arm"]))
        (HERE / "pg_maint.json").write_text(json.dumps(dict(
            version=ver, timing_status="smoke, machine shared", machine_at_start=starts[0], machine_at_start_per_process=starts,
            machine_at_end_per_process=ends, runs=runs), indent=1))
    return parts + mparts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=28860); a = ap.parse_args()
    merged = merge_parts()
    if merged:
        print("merged", [p.name for p in merged], "(delete them after checking)")
    R = json.loads((HERE / "results.json").read_text())
    c = psycopg2.connect(host="127.0.0.1", port=a.port, dbname="ladder", user="jding"); cur = c.cursor()
    cur.execute("SELECT version()"); version = cur.fetchone()[0]
    st = {}
    for s in SETTINGS:
        cur.execute(f"SHOW {s}"); st[s] = cur.fetchone()[0]
    cur.execute("SELECT name, default_version, installed_version FROM pg_available_extensions "
                "WHERE name IN ('bruce_pg', 'softavg_ablation', 'vector', 'pg_ivm') ORDER BY 1")
    exts = {n: dict(default_version=d, installed_in_ladder_db=i) for n, d, i in cur.fetchall()}
    c.close()

    wh, wc, wf = jsonl(HERE / "pg_window.jsonl")
    pm = json.loads((HERE / "pg_maint.json").read_text())

    # ---- window: engine vs replay, cell by cell (S2/S3/S4; the replay has no S0/S1)
    rep = {(x["dataset"], x["k"], x["eps"], x["q"], x["rung"]): x for x in R["window"]["postgres_replay"]}
    fields = ["n_rows", "n_null", "n_inf", "n_wrong_finite", "n_wrong", "max_abs_err_over_vmax", "max_rel_err", "worst_row"]
    wcmp = defaultdict(lambda: dict(cells=0, all_fields_equal=0, wrong_counts_equal=0, max_err_rel_diff_max=0.0,
                                    s4_punts_equal=0, s4_cells=0, differing=[], engine_errors=[]))
    for e in wc:
        if e["rung"] not in ("S2", "S3", "S4"):
            continue
        r = rep.get((e["dataset"], e["k"], e["eps"], e["q"], e["rung"]))
        if r is None:
            continue
        if e.get("error"):
            # the query raised in the server; the replay has a value here (and may have logged why on stderr)
            wcmp[e["rung"]]["engine_errors"].append(dict(cell=[e["dataset"], e["k"], e["eps"], e["q"]], error=e["error"],
                                                          sqlstate=e.get("sqlstate"), replay_stderr=r.get("stderr"),
                                                          replay_n_wrong=r["n_wrong"]))
            continue
        w = wcmp[e["rung"]]; w["cells"] += 1
        eq = all(same(e[f], r[f]) for f in fields)
        w["all_fields_equal"] += eq
        w["wrong_counts_equal"] += all(e[f] == r[f] for f in ("n_null", "n_inf", "n_wrong_finite", "n_wrong"))
        if e["max_abs_err_over_vmax"] and r["max_abs_err_over_vmax"]:
            w["max_err_rel_diff_max"] = max(w["max_err_rel_diff_max"],
                                            abs(e["max_abs_err_over_vmax"] - r["max_abs_err_over_vmax"]) / r["max_abs_err_over_vmax"])
        if e["rung"] == "S4":
            w["s4_cells"] += 1; w["s4_punts_equal"] += e["stats"]["punts"] == r["stats"]["punts"]
        if not eq and len(w["differing"]) < 20:
            w["differing"].append(dict(cell=[e["dataset"], e["k"], e["eps"], e["q"]],
                                       engine={f: e[f] for f in fields}, replay={f: r[f] for f in fields}))
    # ---- maintenance: engine vs replay
    mrep = {(x["stream"], x["eps"]): x for x in R["maintenance_replay"]["runs"]}
    mcmp = []
    for x in pm["runs"]:
        r = mrep[(x["stream"], x["eps"])]["arms"][x["arm"]]
        p, q = x["per_statement_touched"], r["per_statement_touched"]
        mcmp.append(dict(stream=x["stream"], eps=x["eps"], arm=x["arm"],
                         groups_recomputed=[x["groups_recomputed"], r["groups_recomputed"]],
                         wrong=[p["wrong"], q["wrong"]], null_mismatch=[p["null_mismatch"], q["null_mismatch"]],
                         nonfinite=[p["nonfinite"], q["nonfinite"]], checked=[p["checked"], q["checked"]],
                         first_wrong_statement=[x["first_wrong_statement"], r["first_wrong_statement"]],
                         max_abs_err_over_vscale=[p["max_abs_err_over_vscale"], q["max_abs_err_over_vscale"]],
                         counts_equal=(x["groups_recomputed"] == r["groups_recomputed"] and p["wrong"] == q["wrong"]
                                       and p["null_mismatch"] == q["null_mismatch"] and p["nonfinite"] == q["nonfinite"]
                                       and p["checked"] == q["checked"]
                                       and x["first_wrong_statement"] == r["first_wrong_statement"])))
    tests = json.loads((ICDE / "experiments/postgres/tests_pg18.json").read_text())
    R["postgres_engine"] = dict(
        note="ENGINE RUN on PostgreSQL 18.6 (conda env /home/jding/miniforge3/envs/ggdb-pg, conda-forge release build, used "
             "read-only); cluster paper_icde/pg18env (port 28860), extensions staged in pg18env/ext and found through "
             "dynamic_library_path / extension_control_path (pg18env/README.md). 2026-11-07. Separate from "
             "window.postgres_replay / maintenance_replay (the call-order replay outside the server, kept as is).",
        timing_status="smoke, machine shared: the 4 window processes (amazon k=1000 | amazon k=10,100 | imdb_all | imdb_ge) and "
                      "the 5 maintenance processes (one per stream) ran concurrently, each single-threaded on its own "
                      "backend (32 cores); another user's job kept the GPU at 100%; load averages are recorded per header / run",
        server=dict(version=version, settings=st, extensions=exts,
                    so_sha256={p.name: sha256_file(p) for p in sorted((PG18ENV / "ext/lib").glob("*.so"))},
                    build="bruce_pg, softavg_ablation: cargo pgrx package (release profile: opt-level 3, lto fat, codegen-units 1), "
                          "--features pg18 --no-default-features, pgrx 0.18.1; vector 0.8.7, pg_ivm 1.16: PGXS -O2",
                    source_sha256={"engines/postgres": sha256_tree(sorted((ICDE / "engines/postgres/src").glob("*.rs"))
                                                                   + [ICDE / "engines/postgres/Cargo.toml"]),
                                   "pg_ablation": sha256_tree(sorted((HERE / "pg_ablation/src").glob("*.rs"))
                                                              + [HERE / "pg_ablation/Cargo.toml",
                                                                 HERE / "pg_ablation/softavg_ablation.control"]),
                                   "softagg-core (engines/core)": sha256_tree(sorted((ICDE / "engines/core/src").glob("*.rs"))
                                                                              + [ICDE / "engines/core/Cargo.toml"])},
                    harness_sha256={p: sha256_file(HERE / p) for p in ("pg_window.py", "pg_maint.py", "common.py", "replay.py")}),
        tests=dict(source="experiments/postgres/tests_pg18.json", passed=tests["passed"], failed=tests["failed"],
                   failed_tests=[t for t, s in tests["tests"].items() if s == "FAILED"],
                   failure_analysis=tests.get("failure_analysis")),
        machine=dict(window_headers=[h["machine_at_start"] for h in wh], window_footers=[f["machine_at_end"] for f in wf],
                     maint_start=pm.get("machine_at_start"), assemble=machine_state()),
        window=dict(definition=R["window"]["definition"].replace("total(r)", "sum(x)"), cells=wc,
                    vs_replay={k: dict(v) for k, v in wcmp.items()}),
        maintenance=dict(runs=pm["runs"], vs_replay=mcmp,
                         vs_replay_summary=dict(runs=len(mcmp), counts_equal=sum(m["counts_equal"] for m in mcmp))),
    )
    R["not_run"] = json.loads((HERE / "not_run.json").read_text())
    (HERE / "results.json").write_text(json.dumps(R, indent=1))
    print("window vs replay:", {k: {kk: vv for kk, vv in v.items() if kk != "differing"} for k, v in wcmp.items()})
    print("maint vs replay: counts equal in", sum(m["counts_equal"] for m in mcmp), "of", len(mcmp))


if __name__ == "__main__":
    main()

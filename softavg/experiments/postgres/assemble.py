#!/usr/bin/env python
"""Merge every raw output of this directory into results.json.

Inputs (all written by the scripts here): load_<srv>.json, cells_<srv>.jsonl,
window_<srv>.jsonl, upgrade_<srv>.json, maint_<srv>.json, tests_pg17.json.
Servers must still be running (environment is read from them)."""
import json
from common import EPS_GRID, FILTERS, HERE, TIMING_STATUS, TOPK, WORKERS, environment, server_env


def jsonl(p):
    head, foot, rows, checks = None, None, [], []
    for line in open(p):
        c = json.loads(line)
        if c.get("header"):
            head = c
        elif c.get("footer"):
            foot = c
        elif c.get("check"):
            checks.append(c)
        else:
            rows.append(c)
    return head, foot, rows, checks


def opt(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    res = dict(
        description="PostgreSQL before/after for softavg(value, score, eps) on IMDb: native SQL (naive / max-shift / "
                    "clamp), pgvector HNSW top-k + AVG, vs bruce_pg 0.2.0 (internal state over softagg-core); "
                    "state-representation A/B against the 0.1.2 float8[] path; sliding windows through the "
                    "moving-aggregate form; pg_ivm / REFRESH maintenance; 0.1.2 -> 0.2.0 upgrade.",
        timing_status=TIMING_STATUS,
        environment=environment(),
        protocol=dict(eps_grid=[str(e) for e in EPS_GRID], workers=WORKERS, filters=FILTERS, topk=TOPK,
                      timing="1 warm-up + 3 timed runs, client wall (psycopg2 execute+fetchall), median; one "
                             "extra EXPLAIN (ANALYZE, TIMING OFF) per successful cell for plan shape / workers",
                      reference="numpy float64 max-shifted per group, from the scores the server holds or computes",
                      max_rel_err="max over groups of |pg - ref| / |ref|; NULL/NaN/Inf outputs counted separately",
                      queries="queries.json (PROTOCOL sampler, 200 queries); this smoke run uses the first 3 "
                              "(live score, top-k and precomputed columns s0..s2, all on the release server)"),
        servers={}, tests=opt(HERE / "tests_pg17.json"))
    res["server_build_per_arm"] = {
        "before_after_matrix (naive/shift/clamp/softavg/softavg_v012, live + precomp)": "release",
        "hnsw_topk": "release", "window": "release", "maintenance_ivm": "release", "upgrade": "release",
        "pgrx_tests": "pgrx-managed cassert server (correctness only, no timings)"}
    for srv in ("rel",):
        env = server_env(srv)
        s = dict(env=env, load=opt(HERE / f"load_{srv}.json"))
        p = HERE / f"cells_{srv}.jsonl"
        if p.exists():
            h, f, rows, checks = jsonl(p)
            for c in rows:
                c["server_build"] = env["server_build"]
            s["matrix"] = dict(machine_at_start=h and h["machine_at_start"], machine_at_end=f and f["machine_at_end"],
                               checks=checks, cells=rows)
        p = HERE / f"window_{srv}.jsonl"
        if p.exists():
            h, f, rows, _ = jsonl(p)
            for c in rows:
                c["server_build"] = env["server_build"]
            s["window"] = dict(rows=h["rows"], ks=h["ks"], moving_aggregate_catalog=h["moving_aggregate_catalog"],
                               softagg_core="guard = running error bound (CORE_CHANGES 'Main thread' decisions)",
                               machine_at_start=h["machine_at_start"], machine_at_end=f and f["machine_at_end"], cells=rows)
        p = HERE / f"window_{srv}_core_v0.jsonl"
        if p.exists():
            h, f, rows, _ = jsonl(p)
            s["window_core_v0"] = dict(note="same harness, previous softagg-core guard (z <= 1e-6 z_peak); "
                                            "earlier release build of bruce_pg (.so md5 3aa985a2d5225680f70b453885d715df); "
                                            "kept as the 'before' of the core change; max_abs_err not recorded then",
                                       rows=h["rows"], ks=h["ks"], cells=rows)
        s["upgrade"] = opt(HERE / f"upgrade_{srv}.json")
        s["maintenance_ivm"] = opt(HERE / "maint_ivm" / "results_maint_ivm.json")
        res["servers"][srv] = s
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print("wrote results.json")


if __name__ == "__main__":
    main()

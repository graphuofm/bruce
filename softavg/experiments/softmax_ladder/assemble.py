#!/usr/bin/env python
"""Merge every measured / replayed cell of the softmax ladder into results.json (+ provenance)."""
import json, sqlite3, subprocess
from pathlib import Path
from common import (ABL_SQLITE, EPS, HERE, ICDE, REPLAY_BIN, SHIPPED_SQLITE, TIMING_STATUS, host, machine_state,
                    sha256_file, sha256_tree)

EXP = ICDE / "experiments"


def jsonl(p):
    head, cells, foot = [], [], []
    for line in open(p):
        r = json.loads(line)
        (head if r.get("header") else foot if r.get("footer") else cells).append(r)
    return head, cells, foot


def cited():
    """Existing measurements this ladder cites instead of re-measuring (copied with their source file)."""
    out = {}
    dk = EXP / "duckdb/results.json"
    d = json.loads(dk.read_text())
    out["duckdb_window"] = dict(
        source=str(dk.relative_to(ICDE)), sha256=sha256_file(dk), setup=d["window"]["setup"],
        rung_map={"naive_window": "S0", "maxshift_partition": "S1", "exact_selfjoin": "S1x",
                  "softavg_segtree": "S2 = S4 (segment tree of combines; no inverse in the C API)",
                  "softavg_naive_mode": "S2 (debug_window_mode='separate': update + finalize per frame)"},
        cells=[c for c in d["window"]["summary"] if c["threads"] == 1 and c["eps"] in [repr(e) for e in EPS]])
    sq = EXP / "sqlite/results.json"
    s = json.loads(sq.read_text())
    out["sqlite_groupby"] = dict(
        source=str(sq.relative_to(ICDE)), sha256=sha256_file(sq),
        rung_map={"softavg_agg": "S2 (= S3 in GROUP BY: no removal happens)", "softavg": "S4", "naive": "S0",
                  "maxshift_cte": "S1"},
        cells=[{k: c[k] for k in ("spelling", "eps", "median_ms", "max_rel_err", "silent_null", "silent_inf", "queries")}
               for c in s["aggregate"] if c["mode"] == "agg" and c["spelling"] in ("naive", "maxshift_cte", "softavg", "softavg_agg")])
    pm = EXP / "postgres/maint_ivm/results_maint_ivm.json"
    p = json.loads(pm.read_text())
    out["postgres_measured_s4_recomputes"] = dict(
        note="groups recomputed by the shipped softavg_ivm in the PostgreSQL runs of the same streams (2026-11-05/06); "
             "used to validate the replay's S4 arm",
        maint_ivm=dict(source=str(pm.relative_to(ICDE)),
                       cells={a["eps"]: a["groups_recomputed_total"] for a in p["arms"] if a["arm"] == "softavg_ivm"}))
    for R in (30, 2):
        f = EXP / f"timeorder/results_maint_ts_amazon_R{R}.json"
        t = json.loads(f.read_text())
        out["postgres_measured_s4_recomputes"][f"amazon_R{R}"] = dict(
            source=str(f.relative_to(ICDE)),
            cells={f"{x['stream']}:{x['eps']}": x["groups_recomputed_total"] for x in t["runs"]})
    return out


def main():
    sh, sc, sf = jsonl(HERE / "sqlite_window.jsonl")
    rh, rc, rf = jsonl(HERE / "replay_window.jsonl")
    ivm = json.loads((HERE / "replay_ivm.json").read_text())
    th, tc, tf = jsonl(HERE / "sqlite_timing.jsonl")
    srcs = lambda d: sorted((HERE / d).rglob("*.rs")) + [HERE / d / "Cargo.toml"]
    core = sorted((ICDE / "engines/core/src").glob("*.rs")) + [ICDE / "engines/core/Cargo.toml"]
    shipped_sqlite = sorted((ICDE / "engines/sqlite/src").glob("*.rs")) + [ICDE / "engines/sqlite/Cargo.toml",
                                                                          ICDE / "engines/sqlite/Cargo.lock"]
    out = dict(
        experiment="softmax ladder: S0 naive / S1 safe two-pass / S2 online, no removal / S3 online + plain removal, "
                   "no certificate / S4 softavg (removal + running error certificate + refold)",
        timing_status=TIMING_STATUS + "; part of the SQLite grid ran concurrently with this experiment's CPU-only replay job and with other users' jobs (1-min load up to ~28 on 32 cores, GPU busy); table 4b re-times S2/S3/S4 interleaved",
        machine=dict(sqlite_window_start=[h["machine_at_start"] for h in sh], sqlite_window_end=[x["machine_at_end"] for x in sf],
                     replay_window_start=[h["machine_at_start"] for h in rh], ivm_replay=ivm.get("machine_at_start"),
                     assemble=machine_state()),
        provenance=dict(
            host=host(), sqlite_version=sqlite3.sqlite_version,
            sqlite_library="system libsqlite3 3.37.2-2ubuntu0.4 behind Python 3.10.12's sqlite3",
            build_type="cargo build --release (opt-level=3, lto=fat, codegen-units=1), --offline, CARGO_TARGET_DIR=softmax_ladder/target",
            so_sha256={"softavg_ablation_sqlite (S2, S3)": sha256_file(ABL_SQLITE),
                       "softagg_sqlite shipped (S4), rebuilt from engines/sqlite into softmax_ladder/target": sha256_file(SHIPPED_SQLITE),
                       "ladder_replay": sha256_file(REPLAY_BIN)},
            source_sha256={"ladder_core": sha256_tree(srcs("ladder_core")),
                           "sqlite_ablation": sha256_tree(srcs("sqlite_ablation")),
                           "replay": sha256_tree(srcs("replay")),
                           "pg_ablation (not compiled)": sha256_tree(srcs("pg_ablation") + [HERE / "pg_ablation/softavg_ablation.control"]),
                           "softagg-core (engines/core)": sha256_tree(core),
                           "engines/sqlite (shipped S4)": sha256_tree(shipped_sqlite)},
            harness_sha256={p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))},
            data=json.loads((HERE / "data_meta.json").read_text())),
        sqlite_probe=sh[0]["probe"],
        window=dict(
            definition="ROWS BETWEEN k PRECEDING AND CURRENT ROW over rn (IMDb: id; Amazon: ts, id); wrong = abs err > 1e-9 * max|v| "
                       "of the frame, or NULL/Inf where the reference is finite; errors vs numpy float64 two-pass per frame; "
                       "latency = query 0, median of 3 of SELECT count(r), total(r) FROM (...)",
            sqlite=sc, postgres_replay=rc,
            sqlite_interleaved_timing=dict(header=th, cells=tc, footer=tf)),
        maintenance_replay=ivm,
        crosscheck_sqlite_vs_pg_replay=json.loads((HERE / "crosscheck.json").read_text()),
        trace=dict(file="trace.json", sha256=sha256_file(HERE / "trace.json")),
        cited=cited(),
        not_run=json.loads((HERE / "not_run.json").read_text()),
    )
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print("cells: sqlite", len(sc), "replay", len(rc), "ivm runs", len(ivm["runs"]))


if __name__ == "__main__":
    main()

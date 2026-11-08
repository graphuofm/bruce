#!/usr/bin/env python
"""Merge this directory's raw outputs into results.json and render README.md's
generated block from it (no number in README.md is typed by hand).
The release server must still be up (environment is read from it)."""
import json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import EPS_GRID, TIMING_STATUS, WORKERS, environment, server_env  # noqa: E402

BEGIN, END = "<!-- BEGIN GENERATED (summarize.py) -->", "<!-- END GENERATED -->"


def jsonl(p):
    head = foot = None; rows = []
    for line in open(p):
        c = json.loads(line)
        if c.get("header"):
            head = c
        elif c.get("footer"):
            foot = c
        else:
            rows.append(c)
    return head, foot, rows


def assemble():
    h, f, cells = jsonl(HERE / "cells_join.jsonl")
    tests = json.loads((HERE.parent / "postgres" / "tests_pg17.json").read_text())
    res = dict(
        description="Multi-label (overlapping facet) softavg in PostgreSQL: complete IMDb genre memberships as a "
                    "bridge table; GROUP BY over the join (naive / max-shift / clamp / softavg, single vs multi label) "
                    "and incremental maintenance over the join (softavg_ivm_create_join, fan-out per base-row edit) "
                    "vs pg_ivm, REFRESH MATERIALIZED VIEW and recompute-on-read.",
        timing_status=TIMING_STATUS, environment=environment(), server=server_env("rel"),
        server_build_per_arm={"join_groupby": "release", "maint_join": "release",
                              "pgrx_tests": "pgrx-managed cassert server (correctness only, no timings)"},
        tests=dict(passed=tests["passed"], failed=tests["failed"], pure_rust=tests["pure_rust"], pg_test=tests["pg_test"],
                   at=tests["at"], join_tests={k: v for k, v in tests["tests"].items() if "join" in k}),
        bridge=json.loads((HERE / "bridge_stats.json").read_text()),
        load_bridge=json.loads((HERE / "load_bridge.json").read_text()),
        protocol=dict(eps_grid=[str(e) for e in EPS_GRID], workers=WORKERS, filters=["none", "filter_num >= 2000"],
                      timing="1 warm-up + 3 timed runs, client wall, median; EXPLAIN (ANALYZE, TIMING OFF) for plan shape",
                      reference="numpy float64 max-shifted per group over the multiset the join produces",
                      queries="first 3 PROTOCOL queries, precomputed score columns s0..s2"),
        join_groupby=dict(machine_at_start=h["machine_at_start"], machine_at_end=f and f["machine_at_end"], cells=cells),
        maint_join=json.loads((HERE / "results_maint_join.json").read_text()),
        race_demo=json.loads((HERE / "race_demo.json").read_text()),
        softavg_ivm_create_on_join_view=json.loads((HERE / "ivm_create_on_join_view.json").read_text()),
        upgrade_check_after_join=json.loads((HERE / "upgrade_check_after_join.json").read_text()),
    )
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    return res


def ms(x):
    return "-" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def e(x):
    return "-" if x is None else ("0" if x == 0 else f"{x:.1e}")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def short(m):
    m = (m or "").replace("ERROR:  ", "")
    return {"value out of range: overflow": "overflow", "value out of range: underflow": "underflow"}.get(m, m[:50])


def render(R):
    eps = R["protocol"]["eps_grid"]
    b = R["bridge"]; mpi = b["memberships_per_item"]
    out = ["_All timings: smoke, machine shared._", ""]
    env = R["server"]
    out.append(f"- server: {env['pg_version'].split(',')[0]}, build **{env['server_build']}**, extensions {env['extensions']}, "
               f"bruce_pg.so md5 `{env['bruce_pg_so_md5']}`; extension build **{R['environment']['extension']['build_type']}**, "
               f"source sha256 `{R['environment']['extension']['source_sha256'][:16]}...`")
    t = R["tests"]
    out.append(f"- `cargo pgrx test pg17`: {t['passed']} passed, {t['failed']} failed ({t['pure_rust']} pure-Rust + {t['pg_test']} "
               f"in-database), at {t['at']}; join-view tests: {', '.join(k.split('pg_test_')[-1] + '=' + v for k, v in t['join_tests'].items())}")
    u = R["upgrade_check_after_join"]
    out.append(f"- 0.1.2 -> 0.2.0 upgrade re-checked with `softavg_ivm_create_join` in both the install and the upgrade script "
               f"(`../postgres/upgrade_check.py`): pass={u['pass']}, catalog objects {u['catalog_objects']}, diff vs fresh install "
               f"{len(u['catalog_diff'])}, views bit-identical {u['view_bits_identical']}")
    out.append(f"- machine at start of the GROUP BY matrix: quiet={R['join_groupby']['machine_at_start'].get('quiet')}, "
               f"load1={R['join_groupby']['machine_at_start'].get('load_average_1min', R['join_groupby']['machine_at_start'].get('load1'))}")
    out += ["", "### Bridge: complete IMDb genre memberships", ""]
    out.append(f"{b['items']:,} movies, {b['memberships']:,} memberships, {b['groups']} groups; "
               f"{b['items_none']:,} movies have no genre row (kept as the corpus' `(none)` facet); "
               f"memberships per movie mean {mpi['mean']:.2f}, median {mpi['median']:.0f}, p90 {mpi['p90']:.0f}, "
               f"p99 {mpi['p99']:.0f}, max {mpi['max']}; the corpus' single label equals the alphabetically first genre "
               f"for every movie (mismatches: {b['single_label_facet_equals_min_genre_mismatches']}).")
    out += ["", table(["memberships per movie"] + list(mpi["histogram"]), [["movies"] + [f"{v:,}" for v in mpi["histogram"].values()]])]
    gm, gs = b["group_sizes_multi"], b["group_sizes_single"]
    out += ["", table(["group", "multi-label rows", "single-label rows", "ratio"],
                      [[g, f"{gm[g]:,}", f"{gs[g]:,}", (f"{gm[g] / gs[g]:.1f}" if gs[g] else "-")] for g in gm])]
    # GROUP BY latency table, query 0, no filter
    cs = R["join_groupby"]["cells"]
    for lab in ("single", "multi"):
        idx = {(c["plan"], c["eps"], c["workers"]): c for c in cs if c["labels"] == lab and c["query"] == 0 and c["filter"] == "none"}
        rows = []
        for p in ("naive", "shift", "clamp", "softavg"):
            rows.append([p] + [" / ".join(("ERR " + short(idx[(p, ep, w)]["error_message"])) if not idx[(p, ep, w)]["sql_ok"]
                                          else ms(idx[(p, ep, w)]["median_ms"]) for w in (0, 8)) for ep in eps])
        n_in = next(iter(idx.values()))["input_rows"]
        out += ["", f"### GROUP BY, {lab}-label ({n_in:,} aggregated rows; precomputed s0, q0, no filter)", "",
                table([f"plan \\ eps (ms, 0 / 8 workers)"] + eps, rows)]
    # correctness over all cells
    g = defaultdict(list)
    for c in cs:
        g[(c["labels"], c["plan"], c["eps"])].append(c)
    rows = []
    for lab in ("single", "multi"):
        for p in ("naive", "shift", "clamp", "softavg"):
            row = [lab, p]
            for ep in eps:
                xs = g[(lab, p, ep)]; bad = [c for c in xs if not c["sql_ok"]]
                ok = [c["max_rel_err"] for c in xs if c["sql_ok"] and c["max_rel_err"] is not None]
                nn = sum(c.get("n_nan", 0) + c.get("n_null", 0) + c.get("n_inf", 0) for c in xs if c["sql_ok"])
                cov = min((c["coverage"] for c in xs if c["sql_ok"]), default=None)
                s = e(max(ok)) if ok else ""
                if bad:
                    s += f" ERR {len(bad)}/{len(xs)} ({short(bad[0]['error_message'])})"
                if nn:
                    s += f" NaN/NULL/Inf {nn}"
                if cov is not None and cov < 1:
                    s += f" coverage {cov:.2f}"
                row.append(s.strip())
            rows.append(row)
    n = len({(c["query"], c["filter"], c["workers"]) for c in cs})
    out += ["", f"### Correctness over all GROUP BY cells ({n} query x filter x workers cells per entry)", "",
            table(["labels", "plan"] + eps, rows)]
    # plan shapes
    shp = {(c["labels"], c["plan"]): c.get("plan_shape") for c in cs if c["query"] == 0 and c["filter"] == "none"
           and c["workers"] == 8 and c["eps"] == "0.1" and c["sql_ok"]}
    out += ["", "Plan shapes (q0, no filter, 8 workers, eps 0.1):", ""]
    out += [f"- {k[0]} / {k[1]}: `{v}`" for k, v in shp.items()]
    # maintenance
    M = R["maint_join"]; st = M["stream"]
    out += ["", "### Maintenance over the join", "",
            f"stream: {st['n_ops']} single-statement ops {st['mix']}; first {st['first_ops_delete_argmax_items']} ops delete each "
            f"multi-label group's argmax item; {st['commit']}; checkpoints every {st['checkpoint_every']} ops; "
            f"fan-out = the edited movie's live memberships (bridge ops: 1); membership deltas implied by the stream: "
            f"{M['arms'][0]['membership_deltas_total']:,} ({M['arms'][0]['membership_deltas_total'] / st['n_ops']:.2f} per op)", ""]
    tc = M["transactional_checks"]
    out.append(f"softavg_ivm_create_join transactional checks: rollback leaves state bytes identical = {tc['rollback_state_bytes_identical']}; "
               f"base-vs-bridge race (READ COMMITTED), base first: second blocked = {tc['base_first']['second_session_blocked']}, "
               f"view == fresh = {tc['base_first']['view_agrees_after_both_commit']} (Drama n {tc['base_first']['drama_n_view']} vs "
               f"{tc['base_first']['drama_n_fresh']}); bridge first: blocked = {tc['bridge_first']['second_session_blocked']}, "
               f"view == fresh = {tc['bridge_first']['view_agrees_after_both_commit']}; REPEATABLE READ base writer vs committed bridge "
               f"write: {tc['rr_base_vs_committed_bridge']['outcome']} {tc['rr_base_vs_committed_bridge'].get('sqlstate', '')} "
               f"(view == fresh = {tc['rr_base_vs_committed_bridge']['view_agrees']}); bridge write under REPEATABLE READ: "
               f"{tc['rr_bridge_write']['outcome']} {tc['rr_bridge_write'].get('sqlstate', '')}")
    rd = R["race_demo"]
    out.append("")
    out.append("Race without the join-side serialization (`race_demo.py`, view over 20k-row copies, eps 0.1; T1 inserts a dominant "
               "base row, T2 its Drama membership): " + "; ".join(
                   f"{v}/{o}: second blocked {rd[v][o]['second_blocked']}, Drama n view {rd[v][o]['drama_n_view']} vs fresh "
                   f"{rd[v][o]['drama_n_fresh']}, softavg view {rd[v][o]['drama_softavg_view']:.4f} vs fresh {rd[v][o]['drama_softavg_fresh']:.4f}"
                   for v in rd for o in rd[v]))
    ic = R["softavg_ivm_create_on_join_view"]
    out.append("")
    out.append(f"`softavg_ivm_create` on a join view: {ic['result']} {ic.get('sqlstate', '')} `{ic.get('message', '')}`")
    rows = []
    for a in M["arms"]:
        if not a.get("created"):
            rows.append([a["eps"], a["arm"], "ERR " + short(a.get("error_message"))] + ["-"] * 6)
            continue
        o = a["ops"]
        base = [o[k] for k in ("delete", "insert", "update") if k in o]
        br = [o[k] for k in ("bridge_insert", "bridge_delete") if k in o]
        note = ""
        if a["arm"] == "refresh_mv":
            note = f"refresh {ms(a['refresh_ms_median'])} ms; stale err {e(a['stale_max_abs_err_over_vscale'])}"
        rows.append([a["eps"], a["arm"],
                     " / ".join(f"{ms(x['median_ms'])}" for x in base) + " ; p99 " + " / ".join(ms(x["p99_ms"]) for x in base),
                     " / ".join(ms(x["median_ms"]) for x in br),
                     a["groups_recomputed_total"], e(a["max_abs_err_over_vscale"]),
                     ms(a["read_ms_median"]), note])
    out += ["", table(["eps", "arm", "base op ms median del/ins/upd ; p99", "bridge op ms median ins/del",
                       "groups recomputed", "max drift / v_scale", "read ms", "notes"], rows)]
    # fan-out sweep
    for arm in ("softavg_ivm_join", "softavg_ivm_single", "pg_ivm_linear_join"):
        xs = [a for a in M["arms"] if a["arm"] == arm and a.get("fanout_sweep")]
        if not xs:
            continue
        fs = sorted({int(f) for a in xs for f in a["fanout_sweep"]["per_fanout"]})
        rows = []
        for a in xs:
            pf = a["fanout_sweep"]["per_fanout"]
            rows.append([a["eps"], "delete ms (median)"] + [ms(pf[str(f)]["delete"].get("median_ms")) if str(f) in pf else "-" for f in fs])
            rows.append([a["eps"], "insert ms (median)"] + [ms(pf[str(f)]["insert"].get("median_ms")) if str(f) in pf else "-" for f in fs])
            if arm.startswith("softavg"):
                rows.append([a["eps"], "groups recomputed / deletes"] +
                            [f"{pf[str(f)]['delete']['recomputed']}/{pf[str(f)]['items']}" if str(f) in pf else "-" for f in fs])
            fit = a["fanout_sweep"].get("fit", {})
            if fit:
                rows.append([a["eps"], "linear fit ms (f >= 1)"] + [
                    "; ".join(f"{k.split('_')[0]} {v['intercept']:.3f} + {v['per_membership'] * 1e3:.1f} us x f" for k, v in fit.items())]
                    + [""] * (len(fs) - 1))
            rows.append([a["eps"], "after sweep drift / mismatched groups"] +
                        [f"{e(a['fanout_sweep']['after_sweep']['max_abs_err_over_vscale'])} / {a['fanout_sweep']['after_sweep']['group_mismatch']}"] + [""] * (len(fs) - 1))
        items = {f: xs[0]["fanout_sweep"]["per_fanout"].get(str(f), {}).get("items") for f in fs}
        out += ["", f"Fan-out sweep, `{arm}` (fan-out = the movie's multi-label memberships; items per fan-out: {items}; "
                    "all (fan-out, item) pairs visited in one random order):", "",
                table(["eps", "metric"] + [f"f={f}" for f in fs], rows)]
    return "\n".join(out)


if __name__ == "__main__":
    R = assemble()
    txt = render(R)
    p = HERE / "README.md"
    s = p.read_text()
    a, b = s.index(BEGIN) + len(BEGIN), s.index(END)
    p.write_text(s[:a] + "\n" + txt + "\n" + s[b:])
    print(txt)

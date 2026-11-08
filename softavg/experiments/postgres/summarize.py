#!/usr/bin/env python
"""Render the tables of README.md from results.json (no number in the README
is typed by hand). Replaces the block between the GENERATED markers."""
import json, re
from collections import defaultdict
from common import HERE

R = json.loads((HERE / "results.json").read_text())
EPS = R["protocol"]["eps_grid"]
BEGIN, END = "<!-- BEGIN GENERATED (summarize.py) -->", "<!-- END GENERATED -->"


def ms(x):
    return "-" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def e(x):
    return "-" if x is None else ("0" if x == 0 else f"{x:.1e}")


def short_err(m):
    if not m:
        return ""
    m = m.replace("ERROR:  ", "")
    return {"value out of range: overflow": "overflow", "value out of range: underflow": "underflow"}.get(m, m[:40])


def cell_txt(c):
    return ("ERR " + short_err(c["error_message"])) if not c["sql_ok"] else ms(c["median_ms"])


def cells(srv):
    m = R["servers"][srv].get("matrix")
    return m["cells"] if m else []


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def latency_table(srv, mode, query=0, filt="none"):
    cs = [c for c in cells(srv) if c["mode"] == mode and c["query"] == query and c["filter"] == filt and c["k"] is None]
    if not cs:
        return "(no cells)"
    idx = {(c["plan"], c["eps"], c["workers"]): c for c in cs}
    plans = list(dict.fromkeys(c["plan"] for c in cs))
    rows = []
    for p in plans:
        rows.append([p] + [" / ".join(cell_txt(idx[(p, ep, w)]) for w in (0, 8) if (p, ep, w) in idx) for ep in EPS])
    return table(["plan \\ eps (ms, 0 / 8 workers)"] + EPS, rows)


def error_table(srv, mode):
    cs = [c for c in cells(srv) if c["mode"] == mode and c["k"] is None]
    if not cs:
        return "(no cells)"
    g = defaultdict(list)
    for c in cs:
        g[(c["plan"], c["eps"])].append(c)
    plans = list(dict.fromkeys(c["plan"] for c in cs))
    rows = []
    for p in plans:
        row = [p]
        for ep in EPS:
            xs = g[(p, ep)]
            bad = [c for c in xs if not c["sql_ok"]]
            ok = [c["max_rel_err"] for c in xs if c["sql_ok"] and c["max_rel_err"] is not None]
            nn = sum(c.get("n_nan", 0) + c.get("n_null", 0) + c.get("n_inf", 0) for c in xs if c["sql_ok"])
            s = e(max(ok)) if ok else ""
            if bad:
                s = (s + " " if s else "") + f"ERR {len(bad)}/{len(xs)} ({short_err(bad[0]['error_message'])})"
            if nn:
                s += f" NaN/NULL/Inf {nn}"
            row.append(s)
        rows.append(row)
    n = len({(c["query"], c["filter"], c["workers"]) for c in cs})
    return table([f"plan \\ eps (max rel err over {n} query x filter x workers cells)"] + EPS, rows)


def state_ab():
    rows = []
    for srv, mode in (("rel", "precomp"), ("rel", "live")):
        idx = {(c["plan"], c["eps"], c["workers"]): c for c in cells(srv)
               if c["mode"] == mode and c["query"] == 0 and c["filter"] == "none" and c["k"] is None}
        for w in (0, 8):
            a, b, n = idx.get(("softavg", "0.1", w)), idx.get(("softavg_v012", "0.1", w)), idx.get(("naive", "0.1", w))
            if not (a and b and n):
                continue
            rows.append([srv, mode, w, ms(n["median_ms"]), ms(b["median_ms"]), ms(a["median_ms"]),
                         f"{b['median_ms'] / a['median_ms']:.2f}", f"{a['median_ms'] / n['median_ms']:.2f}"])
    return table(["server", "score", "workers", "naive ms", "0.1.2 float8[] state ms", "0.2.0 internal state ms",
                  "0.1.2 / 0.2.0", "0.2.0 / naive"], rows)


def topk_table():
    cs = [c for c in cells("rel") if c["k"] is not None]
    if not cs:
        return "(no cells)"
    rows = []
    for filt in ("none", "ge2000"):
        for k in sorted({c["k"] for c in cs}):
            avg = [c for c in cs if c["k"] == k and c["filter"] == filt and c["plan"] == "topk_avg" and c["sql_ok"]]
            soft = [c for c in cs if c["k"] == k and c["filter"] == filt and c["plan"] == "topk_softavg" and c["sql_ok"]]
            if not avg:
                continue
            cov = sorted({round(c["coverage"], 3) for c in avg})
            msv = [c["median_ms"] for c in avg if c["eps"] == EPS[0]]
            err_avg = {ep: max((c["max_rel_err"] or 0) for c in avg if c["eps"] == ep) for ep in ("0.1", "0.001")}
            err_soft = {ep: max((c["max_rel_err"] or 0) for c in soft if c["eps"] == ep) for ep in ("0.1", "0.001")}
            rows.append([filt, k, f"{min(msv):.1f}-{max(msv):.1f}", f"{min(cov):.2f}-{max(cov):.2f}",
                         e(err_avg["0.1"]), e(err_avg["0.001"]), e(err_soft["0.1"]), e(err_soft["0.001"])])
    return table(["filter", "k", "top-k+AVG ms (3 queries)", "facet coverage (3 queries)",
                  "AVG err vs softavg eps=0.1", "AVG err eps=0.001", "top-k softavg err eps=0.1",
                  "top-k softavg err eps=0.001"], rows)


def window_table():
    w = R["servers"]["rel"].get("window")
    w0 = R["servers"]["rel"].get("window_core_v0")
    if not w:
        return "(no window run)"
    idx = {(c["plan"], c["k"], c["eps"]): c for c in w["cells"]}
    idx0 = {(c["plan"], c["k"], c["eps"]): c for c in (w0["cells"] if w0 else [])}
    rows = []
    for k in w["ks"]:
        for p in ("softavg_moving", "softavg_fwd", "naive_window"):
            row = [k, p]
            for ep in EPS:
                c = idx.get((p, k, ep))
                if c is None:
                    row.append("-"); continue
                if not c["sql_ok"]:
                    row.append("ERR " + short_err(c["error_message"])); continue
                s = f"{ms(c['median_ms'])} ms, abs {e(c.get('max_abs_err_over_vmax'))}, rel {e(c['max_rel_err'])}"
                if p == "softavg_moving":
                    s += f", punts {c['punts']}/{c['inverses']}"
                row.append(s)
            rows.append(row)
    out = (f"rows = {w['rows']} (first ids of imdb, ORDER BY id), score s0. abs = max |err| / max|v| of the "
           f"frame (PROTOCOL window rule: <= 1e-9), rel = max relative error, both vs a numpy f64 per-frame reference\n\n"
           + table(["k", "plan"] + EPS, rows))
    if w0:
        rows = []
        for k in w["ks"]:
            row = [k]
            for ep in EPS:
                a, b = idx0.get(("softavg_moving", k, ep)), idx.get(("softavg_moving", k, ep))
                if not (a and b and a["sql_ok"] and b["sql_ok"]):
                    row.append("-"); continue
                row.append(f"rel {e(a['max_rel_err'])} -> {e(b['max_rel_err'])}; punts {a['punts']} -> {b['punts']}; "
                           f"abs now {e(b.get('max_abs_err_over_vmax'))}")
            rows.append(row)
        out += ("\n\nMoving aggregate, old core guard (z <= 1e-6 z_peak) -> new core guard (running error bound):\n\n"
                + table(["k"] + EPS, rows))
    return out


def maint_table():
    m = R["servers"]["rel"].get("maintenance_ivm")
    if not m:
        return "(no maintenance run)"
    st = m["stream"]
    rows = []
    for a in m["arms"]:
        if not a.get("created"):
            rows.append([a["eps"], a["arm"], "ERR " + short_err(a["error_message"]), "-", "-", "-", "-", "-"])
            continue
        o = a["ops"]["all"]
        extra = ""
        if a["arm"] == "refresh_mv":
            extra = f"refresh {ms(a['refresh_ms_median'])} ms; stale err {e(a['stale_max_abs_err_over_vscale'])}"
        rows.append([a["eps"], a["arm"], f"{ms(o['median_ms'])} / {ms(o['p99_ms'])}", ms(o["mean_ms"]),
                     a.get("groups_recomputed_total", 0), e(a["max_abs_err_over_vscale"]), ms(a["read_ms_median"]), extra])
    t = m["transactional_checks"]
    head = (f"stream: {st['n_ops']} single-row ops ({st['mix']}), first {st['first_ops_delete_argmax_rows']} ops delete "
            f"each facet's argmax row; {st['commit']}; checkpoints every {st['checkpoint_every']} ops; "
            f"server build: {m['server_build']}\n\n"
            f"softavg_ivm transactional checks: rollback leaves state bytes identical = "
            f"{t['rollback_state_bytes_identical']}; two sessions on one facet, both orders: second session blocked "
            f"on the state row = {t['t1_first']['second_session_blocked_on_state_row']}/"
            f"{t['t2_first']['second_session_blocked_on_state_row']}, view == fresh query after both commit = "
            f"{t['t1_first']['view_agrees_after_both_commit']}/{t['t2_first']['view_agrees_after_both_commit']}\n\n")
    return head + table(["eps", "arm", "per-op ms median / p99", "per-op ms mean", "groups recomputed",
                         "max drift / v_scale (all checkpoints)", "read ms (median)", "notes"], rows)


def tests_txt():
    t = R.get("tests")
    if not t:
        return "(tests_pg17.json missing)"
    return (f"`cargo pgrx test pg17`: {t['passed']} passed, {t['failed']} failed "
            f"({t['pure_rust']} pure-Rust + {t['pg_test']} in-database); log digest in tests_pg17.json")


def upgrade_txt():
    u = R["servers"]["rel"].get("upgrade")
    if not u:
        return "(no upgrade run)"
    return (f"0.1.2 -> 0.2.0 on rel: pass={u['pass']}; stype {u['stype_before']} -> {u['after']['stype']}; "
            f"views bit-identical {u['view_bits_identical']}, MV refresh bit-identical {u['mv_refresh_bits_identical']}; "
            f"catalog diff vs fresh install: {len(u['catalog_diff'])} of {u['catalog_objects']} objects; "
            f"window after upgrade used {u['window_after_upgrade_inverses']} inverse calls")


def env_txt():
    out = []
    en = R["servers"]["rel"]["env"]
    out.append(f"- `rel`: {en.get('pg_version', '?').split(',')[0]}, server build **{en['server_build']}**, "
               f"extensions {en.get('extensions')}, bruce_pg.so md5 `{en['bruce_pg_so_md5']}`")
    x = R["environment"]["extension"]
    out.append(f"- extension build: **{x['build_type']}** (`{x['build_cmd']}`), source sha256 `{x['source_sha256'][:16]}...`")
    out.append("- server build per arm: " + "; ".join(f"{k}: {v}" for k, v in R["server_build_per_arm"].items()))
    m = (R["servers"]["rel"].get("matrix") or {}).get("machine_at_start")
    if m:
        out.append(f"- machine at start of the matrix: quiet={m.get('quiet')}, load1={m.get('load_average_1min')}")
    return "\n".join(out)


def main():
    blocks = [
        "### Build and environment\n\n" + env_txt(),
        "### Tests\n\n" + tests_txt(),
        "### Upgrade path\n\n" + upgrade_txt(),
        "### Before / after, aggregation only (precomputed score s0, q0, no filter)\n\n"
        + latency_table("rel", "precomp"),
        "### Correctness over all precomputed-score cells (3 queries x 2 filters x 2 worker counts)\n\n"
        + error_table("rel", "precomp"),
        "### Before / after with the live pgvector score (-(emb <#> q0), no filter)\n\n" + latency_table("rel", "live"),
        "### Correctness over all live-score cells\n\n" + error_table("rel", "live"),
        "### State representation A/B (eps = 0.1, q0, no filter)\n\n" + state_ab(),
        "### pgvector HNSW top-k + aggregate (3 queries)\n\n" + topk_table(),
        "### Sliding window: moving aggregate vs frame recompute\n\n" + window_table(),
        "### Incremental maintenance of a grouped softavg\n\n" + maint_table(),
    ]
    text = f"{BEGIN}\n_All timings: {R['timing_status']}._\n\n" + "\n\n".join(blocks) + f"\n{END}"
    readme = (HERE / "README.md").read_text()
    readme = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: text, readme, flags=re.S)
    (HERE / "README.md").write_text(readme)
    print(text)


if __name__ == "__main__":
    main()

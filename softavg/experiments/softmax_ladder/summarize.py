#!/usr/bin/env python
"""results.json -> the generated block of README.md (no number in README is typed by hand)."""
import json, re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
EPS = [1e-4, 1e-3, 0.02, 0.1, 1.0]
DS = ["imdb_all", "imdb_ge", "amazon"]
DSN = {"imdb_all": "IMDb all", "imdb_ge": "IMDb filter_num>=2000", "amazon": "Amazon (ts order)"}


def g(x, d=1):
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.{d}e}"
    return f"{x:.{d + 1}g}" if abs(x) < 100 else f"{x:.0f}"


def ms(x):
    return "-" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}")


def pool(cells):
    """sum rows / wrong / null / inf, max err over finite rows, errors."""
    p = dict(rows=0, wrong=0, wrong_fin=0, null=0, inf=0, err=None, rel=None, sql_errors=0, n=0)
    for c in cells:
        p["n"] += 1
        if c.get("error"):
            p["sql_errors"] += 1; continue
        p["rows"] += c["n_rows"]; p["wrong"] += c["n_wrong"]; p["wrong_fin"] += c["n_wrong_finite"]
        p["null"] += c["n_null"]; p["inf"] += c["n_inf"]
        for k, f in (("err", "max_abs_err_over_vmax"), ("rel", "max_rel_err")):
            if c.get(f) is not None:
                p[k] = c[f] if p[k] is None else max(p[k], c[f])
    return p


def cellstr(p):
    if p["n"] == 0:
        return "not run"
    s = f"{p['wrong']:,} / {p['rows']:,}"
    if p["null"] or p["inf"]:
        s += f" ({p['null']:,} NULL, {p['inf']:,} Inf)"
    s += f"; {g(p['err'])}"
    if p["sql_errors"]:
        s += f"; {p['sql_errors']} SQL errors"
    return s


def main():
    W = R["window"]; L = []
    sq = W["sqlite"]; pr = W["postgres_replay"]
    by = defaultdict(list)
    for c in sq:
        by[("sqlite", c["rung"], c["eps"])].append(c)
    for c in pr:
        by[("pg", c["rung"], c["eps"])].append(c)
    PE = R.get("postgres_engine")
    pe = PE["window"]["cells"] if PE else []
    for c in pe:
        by[("pge", c["rung"], c["eps"])].append(c)
    prov = R["provenance"]; d = prov["data"]
    L.append(f"_Timings: {R['timing_status']}._\n")
    L.append(f"- SQLite {prov['sqlite_version']} ({prov['sqlite_library']}); ablation .so sha256 "
             f"`{list(prov['so_sha256'].values())[0][:16]}...`, shipped S4 .so `{list(prov['so_sha256'].values())[1][:16]}...`; "
             f"build: {prov['build_type']}")
    ms0 = R["machine"]["sqlite_window_start"][0]
    L.append(f"- machine at start of the SQLite window run: quiet={ms0.get('quiet')}, load average {ms0.get('load_average')}; "
             f"at assemble: load average {R['machine']['assemble'].get('load_average')}")
    L.append(f"- rows: IMDb {d['imdb']['rows']:,} (predicate subset {d['imdb']['rows_ge2000']:,}), Amazon {d['amazon']['rows']:,} "
             f"reviews with verified timestamps ({d['amazon']['ts_min'][:10]} .. {d['amazon']['ts_max'][:10]}); 3 queries per corpus; "
             f"max score per query IMDb {[round(x, 6) for x in d['imdb']['s_max']]}, Amazon {[round(x, 6) for x in d['amazon']['s_max']]}")
    pb = R["sqlite_probe"]
    L.append(f"- SQLite S2 probe: `sqlite3_create_window_function` with xValue but no xInverse returns "
             f"{pb['create_window_function_xValue_without_xInverse_rc']} (SQLITE_MISUSE = {pb['sqlite_misuse']}); a legacy aggregate "
             f"registers (rc {pb['create_function_v2_legacy_aggregate_rc']}) but `OVER (...)` fails at prepare: "
             f"`{pb['legacy_aggregate_as_window']}`; the shipped aggregate-only twin: `{pb['shipped_softavg_agg_as_window']}`")
    if PE:
        sv = PE["server"]; ex = sv["extensions"]
        L.append(f"- **PostgreSQL engine run**: {sv['version'].split(' on ')[0]} (conda env ggdb-pg, release build, read-only; "
                 f"cluster `pg18env`, extensions via `dynamic_library_path` / `extension_control_path`); bruce_pg "
                 f"{ex['bruce_pg']['default_version']}, softavg_ablation {ex['softavg_ablation']['default_version']}; "
                 f"build: {sv['build']}; settings: shared_buffers {sv['settings']['shared_buffers']}, work_mem "
                 f"{sv['settings']['work_mem']}, jit {sv['settings']['jit']}, max_parallel_workers_per_gather "
                 f"{sv['settings']['max_parallel_workers_per_gather']}")
        la = [h.get("load_average") for h in PE["machine"]["window_headers"]] + [PE["machine"]["assemble"].get("load_average")]
        L.append(f"- PG timing conditions: {PE['timing_status']}; load average at the start of each window process "
                 f"{la[:-1]}, at assemble {la[-1]}")
        T_ = PE["tests"]
        L.append(f"- bruce_pg test suite on PG 18.6 (`{T_['source']}`): {T_['passed']} passed, {T_['failed']} failed "
                 f"({', '.join(T_['failed_tests'])}: the test stages its probe script in the compiled-in sharedir, which PG 18 "
                 "does not read when the control file comes from `extension_control_path`; the same probe by hand passes)")
    L.append("")

    L.append("### 1. The ladder, sliding windows (all row sets x k in {10, 100, 1000} x 3 queries pooled)\n")
    L.append("Cell: wrong rows / rows (NULL, Inf among them); max abs err / max|v| over the finite rows. "
             "Wrong = abs err > 1e-9 max|v| of the frame, or NULL / Inf. SQLite = engine run; "
             "PostgreSQL 18.6 = engine run (`postgres_engine`); PostgreSQL (replay) = PG's call order replayed outside "
             "the server over the same Rust arithmetic (done while no PG server existed; kept for the comparison in table 3b). "
             "PG S0 cells that raise an ERROR (exp overflow) count as SQL errors, with no rows. "
             "S1x (exact SQL) ran for query 0 only, at k = 10, 100 (all row sets) and k = 1000 (IMDb subset).\n")
    L.append("| engine | rung | " + " | ".join(f"eps={e:g}" for e in EPS) + " |")
    L.append("|---|---|" + "---|" * len(EPS))
    names = {"S0": "S0 naive", "S1": "S1 global-max shift", "S1x": "S1x exact SQL (frame max + correlated subquery)",
             "S2": "S2 online, no removal (frame refold)", "S3": "S3 online + plain removal, no certificate",
             "S4": "S4 softavg (certificate + refold)"}
    ENG = {"sqlite": "SQLite", "pge": "PostgreSQL 18.6", "pg": "PostgreSQL (replay)"}
    for eng, rungs in (("sqlite", ["S0", "S1", "S1x", "S2", "S3", "S4"]), ("pge", ["S0", "S1", "S2", "S3", "S4"] if PE else []),
                       ("pg", ["S2", "S3", "S4"])):
        for r in rungs:
            L.append(f"| {ENG[eng]} | {names[r]} | "
                     + " | ".join(cellstr(pool(by[(eng, r, e)])) for e in EPS) + " |")
    L.append("")

    L.append("### 2. S3 vs S4 per row set and frame (SQLite engine run; 3 queries pooled)\n")
    L.append("Cell: S3 wrong rows (NULL/Inf) ; S3 max abs err / max|v| || S4 max abs err / max|v| ; S4 refolds. "
             "PG replay S3 / S4 in the next table.\n")
    L.append("| row set | k | " + " | ".join(f"eps={e:g}" for e in EPS) + " |")
    L.append("|---|---|" + "---|" * len(EPS))
    idx = defaultdict(list)
    for c in sq:
        idx[(c["dataset"], c["k"], c["eps"], c["rung"])].append(c)
    for ds in DS:
        for k in (10, 100, 1000):
            row = []
            for e in EPS:
                p3 = pool(idx[(ds, k, e, "S3")]); p4 = pool(idx[(ds, k, e, "S4")])
                rf = sum((c.get("stats") or {}).get("refolds", 0) for c in idx[(ds, k, e, "S4")])
                nn = f" ({p3['null'] + p3['inf']:,})" if p3["null"] + p3["inf"] else ""
                row.append(f"{p3['wrong']:,}{nn} ; {g(p3['err'])} \\|\\| {g(p4['err'])} ; {rf:,}")
            L.append(f"| {DSN[ds]} | {k} | " + " | ".join(row) + " |")
    L.append("")

    L.append("### 3. PostgreSQL replay: S3 vs S4 per row set and frame (3 queries pooled)\n")
    L.append("Cell: S3 wrong rows (NULL/Inf) ; S3 max abs err / max|v| || S4 max abs err / max|v| ; S4 punts "
             "(MINVFUNC returned NULL, frame restarted) / rows re-absorbed by the restarts.\n")
    L.append("| row set | k | " + " | ".join(f"eps={e:g}" for e in EPS) + " |")
    L.append("|---|---|" + "---|" * len(EPS))
    ip = defaultdict(list)
    for c in pr:
        ip[(c["dataset"], c["k"], c["eps"], c["rung"])].append(c)
    for ds in DS:
        for k in (10, 100, 1000):
            row = []
            for e in EPS:
                p3 = pool(ip[(ds, k, e, "S3")]); p4 = pool(ip[(ds, k, e, "S4")])
                pu = sum(c["stats"]["punts"] for c in ip[(ds, k, e, "S4")])
                rr = sum(c["stats"]["restart_rows"] for c in ip[(ds, k, e, "S4")])
                nn = f" ({p3['null'] + p3['inf']:,})" if p3["null"] + p3["inf"] else ""
                row.append(f"{p3['wrong']:,}{nn} ; {g(p3['err'])} \\|\\| {g(p4['err'])} ; {pu:,} / {rr:,}")
            L.append(f"| {DSN[ds]} | {k} | " + " | ".join(row) + " |")
    L.append("")

    if PE:
        V = PE["window"]["vs_replay"]
        L.append("### 3b. PostgreSQL 18.6 engine vs the call-order replay (window cells, S2 / S3 / S4)\n")
        L.append("Per cell (row set x k x eps x query) the engine's error summary is compared with the replay's: "
                 "n_rows, NULL, Inf, wrong (finite / all), max abs err / max|v|, max rel err and the worst row, each compared "
                 "exactly (float equality). S4 punts: `softavg_window_stats()` in the server vs the replay's MINVFUNC-NULL count.\n")
        L.append("| rung | cells compared | engine ERROR | all fields equal | wrong / NULL / Inf counts equal | max rel. diff of max err | S4 punts equal |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ("S2", "S3", "S4"):
            v = V.get(r)
            if not v:
                continue
            pu = f"{v['s4_punts_equal']} / {v['s4_cells']}" if r == "S4" else "-"
            L.append(f"| {r} | {v['cells']} | {len(v['engine_errors'])} | {v['all_fields_equal']} | {v['wrong_counts_equal']} | {g(v['max_err_rel_diff_max'])} | {pu} |")
        ee = [(r, d) for r in ("S2", "S3", "S4") for d in (V.get(r) or {}).get("engine_errors", [])]
        for r, d in ee:
            L.append(f"\n**{r} {d['cell']} raised in the server**: `{d['error']}` (SQLSTATE {d['sqlstate']}); the replay returned a value "
                     f"there ({d['replay_n_wrong']} wrong rows) and logged `{(d['replay_stderr'] or '').strip()}`.")
        ex = [(r, d) for r in ("S2", "S3", "S4") for d in (V.get(r) or {}).get("differing", [])]
        if ex:
            L.append("\nCells whose summaries differ (first ones; cell = row set, k, eps, query):\n")
            for r, d in ex[:8]:
                e_, p_ = d["engine"], d["replay"]
                L.append(f"- {r} {d['cell']}: engine wrong {e_['n_wrong']:,} (NULL {e_['n_null']:,}), max err {g(e_['max_abs_err_over_vmax'], 3)}, "
                         f"worst row {e_['worst_row']}; replay wrong {p_['n_wrong']:,} (NULL {p_['n_null']:,}), max err "
                         f"{g(p_['max_abs_err_over_vmax'], 3)}, worst row {p_['worst_row']}")
        L.append("")
        L.append("### 3c. PostgreSQL 18.6 engine: S3 vs S4 per row set and frame (3 queries pooled)\n")
        L.append("Cell: S3 wrong rows (NULL/Inf) ; S3 max abs err / max|v| || S4 max abs err / max|v| ; S4 punts "
                 "(`softavg_window_stats()`: MINVFUNC returned NULL, PG restarted the frame).\n")
        L.append("| row set | k | " + " | ".join(f"eps={e:g}" for e in EPS) + " |")
        L.append("|---|---|" + "---|" * len(EPS))
        ie = defaultdict(list)
        for c in pe:
            ie[(c["dataset"], c["k"], c["eps"], c["rung"])].append(c)
        for ds in DS:
            for k in (10, 100, 1000):
                row = []
                for e in EPS:
                    p3 = pool(ie[(ds, k, e, "S3")]); p4 = pool(ie[(ds, k, e, "S4")])
                    pu = sum((c.get("stats") or {}).get("punts", 0) for c in ie[(ds, k, e, "S4")])
                    nn = f" ({p3['null'] + p3['inf']:,})" if p3["null"] + p3["inf"] else ""
                    row.append(f"{p3['wrong']:,}{nn} ; {g(p3['err'])} \\|\\| {g(p4['err'])} ; {pu:,}")
                L.append(f"| {DSN[ds]} | {k} | " + " | ".join(row) + " |")
        L.append("")

    L.append("### 4. SQLite latency (query 0, median of 3, ms) and what S4 costs over S3 / S2\n")
    for e in (1e-3, 0.02, 1.0):
        L.append(f"eps = {e:g}\n")
        L.append("| row set | k | S0 | S1 | S1x | S2 | S3 | S4 | S4 / S3 | S2 / S4 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for ds in DS:
            for k in (10, 100, 1000):
                t = {}
                for r in ("S0", "S1", "S1x", "S2", "S3", "S4"):
                    cs = [c for c in idx[(ds, k, e, r)] if c["q"] == 0 and c.get("median_ms") is not None]
                    t[r] = cs[0]["median_ms"] if cs else None
                rat = lambda a, b: f"{t[a] / t[b]:.2f}" if t[a] and t[b] else "-"
                L.append(f"| {DSN[ds]} | {k} | " + " | ".join(ms(t[r]) for r in ("S0", "S1", "S1x", "S2", "S3", "S4"))
                         + f" | {rat('S4', 'S3')} | {rat('S2', 'S4')} |")
        L.append("")

    T = W.get("sqlite_interleaved_timing", {}).get("cells", [])
    if T:
        L.append("### 4b. SQLite, interleaved timing of S2 / S3 / S4 (query 0; 7 repetitions, rungs rotated within each; "
                 "median ms; ratio = median of per-repetition ratios)\n")
        L.append("| row set | k | eps | S2 | S3 | S4 | S4 / S3 | S2 / S4 | load avg |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in T:
            m = c["median_ms"]
            L.append(f"| {DSN[c['dataset']]} | {c['k']} | {c['eps']:g} | {ms(m['S2'])} | {ms(m['S3'])} | {ms(m['S4'])} | "
                     f"{c['ratio_S4_S3']:.3f} | {c['ratio_S2_S4']:.2f} | {c['load_average'][0]} |")
        r43 = sorted(c["ratio_S4_S3"] for c in T); r24 = sorted(c["ratio_S2_S4"] for c in T if c["k"] == 1000)
        L.append(f"\nS4 / S3 over the {len(T)} cells: min {r43[0]:.3f}, median {r43[len(r43) // 2]:.3f}, max {r43[-1]:.3f}; "
                 f"S2 / S4 at k = 1000: {r24[0]:.2f} .. {r24[-1]:.2f}.\n")

    if PE:
        L.append("### 4c. PostgreSQL 18.6 latency (query 0, median of 3 or 1 run when the fetch took > 20 s, ms)\n")
        L.append("`SELECT count(x), sum(x) FROM (window query)`. `err` = the query raised an ERROR (S0: exp overflow), no latency. "
                 "S0, S1 use PG's `sum(float8)`, which has no inverse, so PG recomputes every frame (O(k) per row), like S2.\n")
        for e in (1e-3, 0.02, 1.0):
            L.append(f"eps = {e:g}\n")
            L.append("| row set | k | S0 | S1 | S2 | S3 | S4 | S4 / S3 | S2 / S4 |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for ds in DS:
                for k in (10, 100, 1000):
                    t = {}
                    for r in ("S0", "S1", "S2", "S3", "S4"):
                        cs = [c for c in ie[(ds, k, e, r)] if c["q"] == 0]
                        t[r] = ("err" if cs and cs[0].get("error") else cs[0].get("median_ms")) if cs else None
                    num = lambda x: x if isinstance(x, float) else None
                    rat = lambda a, b: f"{num(t[a]) / num(t[b]):.2f}" if num(t[a]) and num(t[b]) else "-"
                    L.append(f"| {DSN[ds]} | {k} | " + " | ".join(t[r] if t[r] == "err" else ms(t[r]) for r in ("S0", "S1", "S2", "S3", "S4"))
                             + f" | {rat('S4', 'S3')} | {rat('S2', 'S4')} |")
            L.append("")

    M = R["maintenance_replay"]
    L.append("### 5. Maintenance (PostgreSQL softavg_ivm trigger semantics, replayed outside the server; no latency)\n")
    L.append("After every statement, each group the statement touched is compared with a fresh fold of its live rows "
             "(the view's `fresh query`); at every checkpoint all groups are. Cell: groups recomputed ; max |view - fresh| / v_scale "
             "over all per-statement checks ; wrong checks (> 1e-9 v_scale, NULL or non-finite) ; first wrong statement. "
             "`PG measured S4` = groups the shipped softavg_ivm recomputed when PostgreSQL ran the same stream "
             "(experiments/postgres, experiments/timeorder; eps 1e-3, 0.02, 0.1 only).\n")
    meas = R["cited"]["postgres_measured_s4_recomputes"]
    L.append("| stream | eps | S2 | S3 | S4 | PG measured S4 |")
    L.append("|---|---|---|---|---|---|")
    for run in M["runs"]:
        cells = []
        for a in ("S2", "S3", "S4"):
            x = run["arms"][a]; p = x["per_statement_touched"]
            fw = x["first_wrong_statement"]
            cells.append(f"{x['groups_recomputed']:,} ; {g(p['max_abs_err_over_vscale'])} ; {p['wrong']:,}"
                         + (f" (NULL {p['null_mismatch']}, non-finite {p['nonfinite']})" if p['null_mismatch'] or p['nonfinite'] else "")
                         + (f" ; #{fw}" if fw >= 0 else ""))
        st = run["stream"]; key = None
        if st == "maint_ivm_10k":
            m = meas["maint_ivm"]["cells"].get(repr(run["eps"]))
        else:
            R_ = st.split("_R")[1]; kind = st.split("_")[1]
            m = meas[f"amazon_R{R_}"]["cells"].get(f"{kind}:{run['eps']!r}")
        L.append(f"| {st} | {run['eps']:g} | " + " | ".join(cells) + f" | {'-' if m is None else m} |")
    L.append("")
    for run in M["runs"]:
        if run["stream"] in ("maint_ivm_10k", "amazon_time_R2") and run["eps"] == 1e-3:
            s = run["setup"]
            L.append(f"- `{run['stream']}` setup: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    L.append("")

    if PE:
        L.append("### 5b. Maintenance, PostgreSQL 18.6 engine run (same streams, statement triggers in the server)\n")
        L.append("Same checks as table 5, done in the server after every statement (touched groups vs a fresh `GROUP BY` "
                 "fold) and at every checkpoint. Cell: groups recomputed ; max |view - fresh| / v_scale ; wrong checks "
                 "(NULL, non-finite) ; first wrong statement ; median ms per statement (trigger included). `= replay`: groups "
                 "recomputed, wrong / NULL / non-finite / checked counts and first wrong statement all equal to table 5, "
                 "per arm (S2 S3 S4).\n")
        cmpi = {(m["stream"], m["eps"], m["arm"]): m for m in PE["maintenance"]["vs_replay"]}
        runs = {(x["stream"], x["eps"], x["arm"]): x for x in PE["maintenance"]["runs"]}
        L.append("| stream | eps | S2 | S3 | S4 | = replay |")
        L.append("|---|---|---|---|---|---|")
        for run in M["runs"]:
            cells = []; eqs = []
            for a_ in ("S2", "S3", "S4"):
                x = runs.get((run["stream"], run["eps"], a_))
                if x is None:
                    cells.append("not run"); eqs.append("-"); continue
                p = x["per_statement_touched"]; fw = x["first_wrong_statement"]
                cells.append(f"{x['groups_recomputed']:,} ; {g(p['max_abs_err_over_vscale'])} ; {p['wrong']:,}"
                             + (f" (NULL {p['null_mismatch']}, non-finite {p['nonfinite']})" if p['null_mismatch'] or p['nonfinite'] else "")
                             + (f" ; #{fw}" if fw >= 0 else "") + f" ; {x['per_stmt_ms']['median']:.2f} ms")
                eqs.append("y" if cmpi[(run["stream"], run["eps"], a_)]["counts_equal"] else "**n**")
            L.append(f"| {run['stream']} | {run['eps']:g} | " + " | ".join(cells) + f" | {' '.join(eqs)} |")
        sm = PE["maintenance"]["vs_replay_summary"]
        L.append(f"\nCounts equal to the replay in {sm['counts_equal']} of {sm['runs']} (stream, eps, arm) runs.\n")

    L.append("### 6. Cited, not re-measured\n")
    dk = R["cited"]["duckdb_window"]
    L.append(f"DuckDB 1.5.6 ({dk['source']}, IMDb predicate subset, 5 queries, `PARTITION BY facet ORDER BY rid ROWS 49 PRECEDING`, "
             "1 thread). DuckDB's C API has no inverse, so S3 cannot be built there; the shipped softavg runs on the segment tree "
             "(S2 = S4 path). Cell: NaN / Inf / NULL / finite-wrong rows ; max abs err / max|v| ; ms.\n")
    L.append("| plan (rung) | " + " | ".join(f"eps={e:g}" for e in EPS) + " |")
    L.append("|---|" + "---|" * len(EPS))
    for plan, rung in dk["rung_map"].items():
        row = []
        for e in EPS:
            c = [x for x in dk["cells"] if x["plan"] == plan and x["eps"] == repr(e)]
            if not c:
                row.append("-"); continue
            c = c[0]
            row.append(f"{c['nan_rows']:,} / {c['inf_rows']:,} / {c['null_rows']:,} / {c['wrong_rows']:,} ; "
                       f"{g(c['max_abs_err_over_vmax'])} ; {ms(c['median_of_query_medians_ms'])}")
        L.append(f"| {plan} ({rung.split(' (')[0]}) | " + " | ".join(row) + " |")
    L.append("")
    sg = R["cited"]["sqlite_groupby"]
    L.append(f"SQLite GROUP BY facet, aggregate only ({sg['source']}, 200 queries, timing 5 queries): "
             "median ms S2 (`softavg_agg`, bare ScalarAcc) vs S4 (`softavg`, InvAcc bookkeeping + 64-row prefix record):\n")
    L.append("| eps | S0 naive | S1 maxshift_cte | S2 softavg_agg | S4 softavg | S4 / S2 |")
    L.append("|---|---|---|---|---|---|")
    for e in EPS:
        t = {c["spelling"]: c["median_ms"] for c in sg["cells"] if c["eps"] == repr(e)}
        L.append(f"| {e:g} | {ms(t.get('naive'))} | {ms(t.get('maxshift_cte'))} | {ms(t.get('softavg_agg'))} | "
                 f"{ms(t.get('softavg'))} | {t['softavg'] / t['softavg_agg']:.2f} |")
    L.append("")
    txt = (HERE / "README.md").read_text()
    new = re.sub(r"<!-- BEGIN GENERATED.*?<!-- END GENERATED -->",
                 "<!-- BEGIN GENERATED (summarize.py) -->\n" + "\n".join(L) + "\n<!-- END GENERATED -->", txt, flags=re.S)
    (HERE / "README.md").write_text(new)
    print("\n".join(L))


if __name__ == "__main__":
    main()

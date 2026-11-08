"""Render the README tables from results.json (and guard_bound.json, guard_tau_old_core.json).

Every number in README.md between the GENERATED markers comes from here;
re-run after run.py:  python3 report.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results.json"))
BEGIN, END = "<!-- BEGIN GENERATED (report.py) -->", "<!-- END GENERATED (report.py) -->"


def ms(x):
    return "-" if x is None else f"{x:.0f}"


def err(x):
    return "0" if x == 0 else f"{x:.1e}"


out = []
p = R["provenance"]
load = p["machine"]["load"]
out.append(f"Host: {p['host_engine']}; math functions compiled in: {p['sqlite_math_functions']}; "
           f"{p['extension']}; build {p['build_type'].split(' ')[0]}; softagg-core {p['softagg_core_version']}; "
           f"sqlite-vec {p['sqlite_vec']['version']}. Threads: {p['threads']}. "
           f"Timing status: **{R['timing_status']}** (load at start: cpu busy "
           f"{load.get('cpu_busy_fraction')}, 1-min load avg {load.get('load_average_1min')}).")
d = R["data"]
q = R["queries"]
out.append(f"\nData: {d['rows_total']:,} rows, {d['rows_admitted']:,} admitted by `{d['predicate']}`, "
           f"{d['groups_admitted']} facets. Accuracy over {q['n_accuracy']} queries; latency over the first "
           f"{q['n_timed']} queries, {q['reps']} reps each ({q['latency']}).")

# ---- aggregate
out.append("\n### GROUP BY facet: before / after\n")
out.append("Silent = a result returned without any SQL error. `NULL grp` counts facet results that came back "
           "NULL although the reference is finite (summed over queries; in parentheses: queries affected). "
           "Max rel err is over the non-NULL finite results. *agg ms*: scores precomputed (aggregate only); "
           "*e2e ms*: score from the BLOB in the same statement (`WITH sc AS MATERIALIZED`). Error columns come "
           f"from the agg-mode run over all {q['n_accuracy']} queries, except `softavg_direct` (e2e only), which "
           f"covers the {q['n_timed']} timed queries. The shared max rel err of the exact spellings at sharp eps "
           "is the score path (softagg_dot vs numpy summation order, amplified by 1/eps), not the aggregate.\n")
out.append("| eps | spelling | SQL errors | silent NULL grp | silent Inf grp | max rel err | agg ms | e2e ms |")
out.append("|---|---|---|---|---|---|---|---|")
agg = {(c["mode"], c["spelling"], c["eps"]): c for c in R["aggregate"]}
spellings = ["naive", "maxshift_cte", "maxshift_window", "clamp", "softavg", "softavg_agg", "softavg_direct"]
for eps in R["eps_grid"]:
    for s in spellings:
        a = agg.get(("agg", s, eps))
        e = agg.get(("e2e", s, eps))
        base = a or e
        if base is None:
            continue
        out.append(f"| {eps} | {s} | {base['errors']} | {base['silent_null']} ({base['queries_with_silent_null']}) | "
                   f"{base['silent_inf']} ({base['queries_with_silent_inf']}) | {err(base['max_rel_err'])} | "
                   f"{ms(a['median_ms']) if a else '-'} | {ms(e['median_ms']) if e else '-'} |")

# ---- top-k
out.append("\n### sqlite-vec vec0 KNN top-k + AVG (the index-side alternative)\n")
out.append("Coverage = facets present in the top-k / admitted facets. The last columns compare the top-k "
           "AVG with the full softavg reference (it is a different quantity; shown to size the gap).\n")
out.append("| k | errors | coverage mean | coverage min | median ms | max rel err vs softavg eps=0.02 | vs eps=inf |")
out.append("|---|---|---|---|---|---|---|")
for t in R["topk"]:
    by = t["avg_vs_softavg_ref_by_eps"]
    cov = "-" if t["coverage_mean"] is None else f"{t['coverage_mean']:.1%}"
    covm = "-" if t["coverage_min"] is None else f"{t['coverage_min']:.1%}"
    e002 = by.get("0.02", {}).get("max_rel_err")
    einf = by.get("inf", {}).get("max_rel_err")
    out.append(f"| {t['k']} | {t['errors']}/{t['queries']} | {cov} | {covm} | {ms(t['median_ms'])} | "
               f"{'-' if e002 is None else f'{e002:.2f}'} | {'-' if einf is None else f'{einf:.2f}'} |")
errs = sorted({e for t in R["topk"] for e in t["error_samples"]})
if errs:
    out.append("\nVerbatim errors: " + "; ".join(f"`{e}`" for e in errs))
sf = R["score_function_ms"]
out.append(f"\nScore function alone (SUM over admitted rows, median ms): softagg_dot {ms(sf['softagg_dot'])}, "
           f"sqlite-vec `1 - vec_distance_cosine` {ms(sf['sqlite_vec_1_minus_cosine'])}.")

# ---- window
w = R["window"]
out.append(f"\n### Sliding window: `{w['frame']}` over {w['rows']:,} rows, {w['n_queries']} queries\n")
out.append("| eps | spelling | SQL errors | silent NULL rows | silent Inf rows | max rel err | median ms | refolds |")
out.append("|---|---|---|---|---|---|---|---|")
for c in w["cells"]:
    rf = "-"
    if "stats" in c:
        rf = str(sum(s["refolds"] for s in c["stats"]))
    out.append(f"| {c['eps']} | {c['spelling']} | {c['errors']} | {c['silent_null']} | {c['silent_inf']} | "
               f"{err(c['max_rel_err'])} | {ms(c['median_ms'])} | {rf} |")

wp = R.get("window_previous_core")
if wp:
    out.append("\nSame `win_softavg` cells under the previous core (z-only guard): " + wp["note"] + "\n")
    out.append("| eps | refolds old | max rel err old | median ms old |")
    out.append("|---|---|---|---|")
    for c in wp["cells"]:
        out.append(f"| {c['eps']} | {c['refolds']} | {err(c['max_rel_err'])} | {ms(c['median_ms'])} |")

gb, go = HERE / "guard_bound.json", HERE / "guard_tau_old_core.json"
if gb.exists() and go.exists():
    g, o = json.load(open(gb)), json.load(open(go))
    out.append(f"\n### Window drift: new core guard (running error bound) vs old (z-only, TAU = 1e-6)\n")
    out.append(f"guard_bound.json vs guard_tau_old_core.json: {g['queries']} queries x {g['rows']:,} rows, "
               f"frame {g['frame']}, InvAcc driven directly, error vs a fresh fold per frame. "
               "Abs err is in units of max|v|; the PROTOCOL window bound is 1e-9. "
               "`err/bound` = worst ratio of the true error to the core's own `value_error_bound()`.\n")
    out.append("| eps | refolds new | refolds old | max abs err / max\\|v\\| new | old | max rel err new | old | "
               "bound violations new | max err/bound new |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for e in sorted({c["eps"] for c in g["cells"]}):
        n = [c for c in g["cells"] if c["eps"] == e]
        od = [c for c in o["cells"] if c["eps"] == e and c["tau"] == o["core_tau"]]
        vmax = max(c["vmax"] for c in n)
        out.append(f"| {e:g} | {sum(c['refolds'] for c in n)} | {sum(c['refolds'] for c in od)} | "
                   f"{err(max(c['max_abs_err_over_vmax'] for c in n))} | {err(max(c['max_abs_err'] for c in od) / vmax)} | "
                   f"{err(max(c['max_rel_err'] for c in n))} | {err(max(c['max_rel_err'] for c in od))} | "
                   f"{sum(c['bound_violations'] for c in n)} | {max(c['max_err_over_bound'] for c in n):.2f} |")

block = "\n".join(out)
readme = HERE / "README.md"
text = readme.read_text()
i, j = text.index(BEGIN), text.index(END)
readme.write_text(text[:i + len(BEGIN)] + "\n" + block + "\n" + text[j:])
print(block)

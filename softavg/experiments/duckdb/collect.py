"""runs/*.json + test runs -> results.json, and the generated tables in
README.md (between the GENERATED markers). Every number in the README comes
from here; nothing is typed by hand.

  python collect.py            # re-run the test suites, rebuild results.json + README tables
  python collect.py --no-tests # keep the test results already in results.json
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

import common as C

RUNS = os.path.join(C.HERE, "runs")
OUT = os.path.join(C.HERE, "results.json")
README = os.path.join(C.HERE, "README.md")
PY = "python"
EXT_DIR = os.path.join(C.ENG, "duckdb")
EPS_ORDER = [C.eps_key(e) for e in C.EPS_GRID]


def load(name):
    p = os.path.join(RUNS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def run_tests():
    out = {}
    r = subprocess.run(["nice", "-n", "10", "cargo", "test", "--release", "--offline"], cwd=EXT_DIR,
                       capture_output=True, text=True)
    out["cargo_test"] = {"returncode": r.returncode,
                         "result_lines": [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("test result")]}
    xml = os.path.join(C.HERE, "runs", "pytest.xml")
    r = subprocess.run([PY, "-m", "pytest", "-q", "tests/", f"--junitxml={xml}"], cwd=EXT_DIR,
                       capture_output=True, text=True)
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = []
    for tc in suite.iter("testcase"):
        status = "failed" if tc.find("failure") is not None or tc.find("error") is not None else (
            "skipped" if tc.find("skipped") is not None else "passed")
        cases.append({"file": tc.get("classname"), "name": tc.get("name"), "status": status})
    by_file = defaultdict(lambda: defaultdict(int))
    for c in cases:
        by_file[c["file"]][c["status"]] += 1
    out["pytest"] = {"returncode": r.returncode, "tests": int(suite.get("tests")),
                     "failures": int(suite.get("failures")), "errors": int(suite.get("errors")),
                     "skipped": int(suite.get("skipped")), "by_file": {k: dict(v) for k, v in by_file.items()},
                     "cases": cases, "summary_line": r.stdout.strip().splitlines()[-1]}
    return out


def mx(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def summarize_grid(g):
    acc = defaultdict(list)
    for c in g["accuracy"]:
        acc[(c["plan"], c["eps"])].append(c)
    accuracy = []
    for (plan, eps), cs in sorted(acc.items(), key=lambda kv: (kv[0][0], EPS_ORDER.index(kv[0][1]))):
        errs = [c for c in cs if c["error"]]
        ok = [c for c in cs if not c["error"]]
        accuracy.append({
            "plan": plan, "eps": eps, "threads": 8, "queries": len(cs), "queries_error": len(errs),
            "first_error": errs[0]["error"] if errs else None,
            "queries_with_nan": sum(1 for c in ok if c["nan_groups"]),
            "queries_with_inf": sum(1 for c in ok if c["inf_groups"]),
            "queries_with_null": sum(1 for c in ok if c["null_groups"]),
            "queries_missing_groups": sum(1 for c in ok if c["missing_groups"]),
            "nan_groups_total": sum(c["nan_groups"] for c in ok),
            "inf_groups_total": sum(c["inf_groups"] for c in ok),
            "groups_total": sum(c["n_groups"] for c in ok),
            "queries_all_groups_nonfinite": sum(1 for c in ok if c["nan_groups"] + c["inf_groups"] == c["n_groups"]),
            "max_rel_err_vs_f64": mx(c["max_rel_err_vs_f64"] for c in ok),
            "median_rel_err_vs_f64": statistics.median([c["max_rel_err_vs_f64"] for c in ok
                                                        if c["max_rel_err_vs_f64"] is not None] or [math.nan]),
            "max_rel_err_vs_same_scores": mx(c.get("max_rel_err_vs_same_scores") for c in ok),
        })
    tim = defaultdict(list)
    for c in g["timing"]:
        tim[(c["plan"], c["eps"], c["threads"], c["score_source"])].append(c)
    timing = []
    for (plan, eps, th, src), cs in sorted(tim.items()):
        ms = [c["median_ms"] for c in cs if c["median_ms"] is not None]
        timing.append({"plan": plan, "eps": eps, "threads": th, "score_source": src, "queries": len(cs),
                       "queries_error": sum(1 for c in cs if c["error"]),
                       "median_of_query_medians_ms": statistics.median(ms) if ms else None,
                       "min_of_query_medians_ms": min(ms) if ms else None,
                       "max_of_query_medians_ms": max(ms) if ms else None,
                       "nonfinite_groups_total": sum(c.get("nan_groups", 0) + c.get("inf_groups", 0) for c in cs)})
    return {"setup": g["setup"], "load_before": g["load_before"], "load_after": g.get("load_after"),
            "accuracy": accuracy, "timing": timing, "explain": g["explain"]}


def summarize_window(w):
    g = defaultdict(list)
    for c in w["cells"]:
        g[(c["plan"], c["eps"], c["threads"])].append(c)
    rows = []
    for (plan, eps, th), cs in sorted(g.items(), key=lambda kv: (kv[0][0], EPS_ORDER.index(kv[0][1]), kv[0][2])):
        ok = [c for c in cs if not c["error"]]
        ms = [c["median_ms"] for c in ok if c["median_ms"] is not None]
        rows.append({"plan": plan, "eps": eps, "threads": th, "queries": len(cs),
                     "queries_error": len(cs) - len(ok), "first_error": next((c["error"] for c in cs if c["error"]), None),
                     "rows_per_query": ok[0]["rows"] if ok else None,
                     "nan_rows": sum(c["nan_rows"] for c in ok), "inf_rows": sum(c["inf_rows"] for c in ok),
                     "null_rows": sum(c["null_rows"] for c in ok),
                     "wrong_rows": sum(c["wrong_rows_abs_err_gt_1e-9_vmax"] for c in ok),
                     "queries_with_bad_rows": sum(1 for c in ok if c["nan_rows"] + c["inf_rows"] + c["null_rows"]
                                                  + c["wrong_rows_abs_err_gt_1e-9_vmax"]),
                     "max_abs_err_over_vmax": mx(c["max_abs_err_over_vmax"] for c in ok),
                     "median_of_query_medians_ms": statistics.median(ms) if ms else None})
    return {"setup": w["setup"], "load_before": w["load_before"], "load_after": w.get("load_after"),
            "summary": rows, "explain": w["explain"]}


def md_window(w):
    plans = ["naive_window", "maxshift_partition", "exact_selfjoin", "softavg_segtree", "softavg_naive_mode"]
    d = {(x["plan"], x["eps"], x["threads"]): x for x in w["summary"]}
    st = w["setup"]
    any_ = next(iter(w["summary"]))
    lines = [f"\nFrame `{st['frame']}` over {any_['rows_per_query']:,} rows per query, {st['queries']} queries, "
             f"precomputed scores. Accuracy cell = `NaN / Inf / NULL / finite-but-wrong rows (all queries) ; "
             f"max abs err / max|v|`, threads 8 (wrong = abs err > 1e-9 max|v|):\n"]
    lines.append("| eps | " + " | ".join(plans) + " |")
    lines.append("|---|" + "---|" * len(plans))
    for e in EPS_ORDER:
        vals = []
        for p in plans:
            x = d.get((p, e, 8))
            vals.append("-" if not x else (f"error: `{x['first_error'][:60]}`" if x["queries_error"] else
                        f"{x['nan_rows']} / {x['inf_rows']} / {x['null_rows']} / {x['wrong_rows']} ; "
                        f"{f_err(x['max_abs_err_over_vmax'])}"))
        lines.append(f"| {e} | " + " | ".join(vals) + " |")
    lines.append(f"\nLatency (median over queries of the median of {st['rounds']} runs, ms; query wrapped in "
                 f"`SELECT count(x), sum(x)`; smoke, machine shared):\n")
    lines.append("| threads | eps | " + " | ".join(plans) + " |")
    lines.append("|---|---|" + "---|" * len(plans))
    for th in (1, 8):
        for e in EPS_ORDER:
            lines.append(f"| {th} | {e} | " + " | ".join(
                f_ms(d[(p, e, th)]["median_of_query_medians_ms"]) if (p, e, th) in d else "-" for p in plans) + " |")
    return "\n".join(lines)


def summarize_topk(t):
    by_k = defaultdict(list)
    for c in t["coverage_cells"]:
        by_k[c["k"]].append(c)
    cov = []
    for k, cs in sorted(by_k.items()):
        cov.append({"k": k, "queries": len(cs),
                    "coverage_mean": statistics.mean(c["coverage"] for c in cs),
                    "coverage_min": min(c["coverage"] for c in cs), "coverage_max": max(c["coverage"] for c in cs),
                    "groups_mean": statistics.mean(c["groups"] for c in cs),
                    "rows_returned_min": min(c["rows_returned"] for c in cs),
                    "recall_mean": statistics.mean(c["recall_vs_exact_topk"] for c in cs),
                    "recall_min": min(c["recall_vs_exact_topk"] for c in cs),
                    "filtered_full_table_rows_returned_mean":
                        statistics.mean(c["filtered_full_table_rows_returned"] for c in cs),
                    "filtered_full_table_groups_mean": statistics.mean(c["filtered_full_table_groups"] for c in cs)})
    tim = defaultdict(list)
    for c in t["timing"]:
        tim[(c["plan"], c["k"], c["threads"])].append(c["median_ms"])
    timing = [{"plan": p, "k": k, "threads": th, "queries": len(v), "median_of_query_medians_ms": statistics.median(v)}
              for (p, k, th), v in sorted(tim.items())]
    keep = {k: t[k] for k in ("vss", "setup", "copy_seconds", "index_build_seconds_adm_8threads",
                              "index_build_seconds_full_8threads", "memory_after_index", "uses_hnsw", "explain",
                              "load_before", "load_after") if k in t}
    return {**keep, "coverage": cov, "timing": timing}


# --------------------------------------------------------------- markdown
def f_ms(x):
    return "-" if x is None else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def f_err(x):
    return "-" if x is None else f"{x:.1e}"


def md_gap(gap):
    lines = []
    plans = ["naive", "softavg@before", "softavg@pre_p1", "softavg@shipped"]
    hdr = ["naive SQL", "before (paper_softavg build)", "round 1 shipped (core pre-P1)", "shipped (core with P1)"]
    for ds in ("reviews", "imdb_q0"):
        cells = [c for c in gap["cells"] if c["dataset"] == ds]
        if not cells:
            continue
        rows = cells[0]["rows"]
        lines.append(f"\n**{ds}** ({rows:,} rows, GROUP BY "
                     f"{'genre' if ds == 'reviews' else 'facet'}, stored DOUBLE score), "
                     f"min of {gap['rounds']} interleaved runs, ms:\n")
        lines.append("| threads | eps | " + " | ".join(hdr) + " |")
        lines.append("|---|---|" + "---|" * len(hdr))
        for th in (1, 8):
            for e in EPS_ORDER:
                row = {c["plan"]: c for c in cells if c["threads"] == th and c["eps"] == e}
                if not row:
                    continue
                vals = []
                for p in plans:
                    c = row.get(p)
                    s = f_ms(c["min_ms"]) if c else "-"
                    if c and (c["nan_groups"] or c["inf_groups"]):
                        s += f" (NaN x{c['nan_groups']}, Inf x{c['inf_groups']})"
                    vals.append(s)
                lines.append(f"| {th} | {e} | " + " | ".join(vals) + " |")
        for th in (1, 8):
            sv = [c for c in cells if c["threads"] == th and "bitwise_equal_to_shipped" in c]
            neq = sorted({c["plan"] for c in sv if not c["bitwise_equal_to_shipped"]})
            lines.append(f"\nthreads {th}: softavg results bitwise equal to the shipped build in "
                         f"{sum(c['bitwise_equal_to_shipped'] for c in sv)}/{len(sv)} cells"
                         + (f" (differing: {', '.join(neq)})" if neq else "") + ".")
        d = {(x["threads"], x["probe"]): x["min_ms"] for x in gap["diag"] if x["dataset"] == ds}
        if d:
            lines.append(f"\nCost attribution ({ds}, min ms; `softavg_diag_*` come from the diag build):\n")
            probes = ["count_star", "avg_v", "sum_s_sum_v", "sum_exp", "capi_noop_update", "capi_touch_update",
                      "capi_softavg_2arg_eps_fixed", "capi_softavg_eps_0.1"]
            lines.append("| threads | " + " | ".join(probes) + " |")
            lines.append("|---|" + "---|" * len(probes))
            for th in (1, 8):
                lines.append(f"| {th} | " + " | ".join(f_ms(d.get((th, p))) for p in probes) + " |")
    return "\n".join(lines)


def md_grid(g):
    lines = []
    plans = ["naive", "maxshift_cte", "maxshift_window", "clamp", "softavg", "softavg_p2"]
    a = {(x["plan"], x["eps"]): x for x in g["accuracy"]}
    nq = g["setup"]["queries"]
    lines.append("\nMeasured in round 1 (before core adopted P1): column `softavg` = the extension over the "
                 "pre-P1 core, `softavg_p2` = over core + P1 + P2 (P2 not adopted). P1 is bit-identical, so the "
                 "accuracy columns hold for the current build; the current build's aggregate latency is in the "
                 "gap tables above.")
    lines.append(f"\n**Accuracy**, {nq} queries x eps, threads 8. Cell = "
                 f"`errors / queries with silent NaN or Inf (groups) / max rel err vs numpy f64 / "
                 f"max rel err vs longdouble on DuckDB's own scores`:\n")
    lines.append("| eps | " + " | ".join(plans) + " |")
    lines.append("|---|" + "---|" * len(plans))
    for e in EPS_ORDER:
        vals = []
        for p in plans:
            x = a.get((p, e))
            if not x:
                vals.append("-")
                continue
            nf = x["queries_with_nan"] + x["queries_with_inf"]
            vals.append(f"{x['queries_error']} / {nf} ({x['nan_groups_total'] + x['inf_groups_total']}) / "
                        f"{f_err(x['max_rel_err_vs_f64'])} / {f_err(x['max_rel_err_vs_same_scores'])}")
        lines.append(f"| {e} | " + " | ".join(vals) + " |")
    errs = {f"{x['plan']} eps={x['eps']}": x["first_error"] for x in g["accuracy"] if x["first_error"]}
    if errs:
        lines.append("\nVerbatim errors: " + "; ".join(f"{k}: `{v}`" for k, v in errs.items()))
    t = {(x["plan"], x["eps"], x["threads"], x["score_source"]): x for x in g["timing"]}
    ntq = g["setup"]["timing_queries"]
    for src, title in (("in_query", "score computed in the query (array_inner_product over FLOAT[384])"),
                       ("precomputed", "precomputed scores (aggregation only)")):
        lines.append(f"\n**Latency, {title}**: median over {ntq} queries of the per-query median of "
                     f"{g['setup']['rounds']} runs, ms (smoke, machine shared):\n")
        lines.append("| threads | eps | " + " | ".join(plans) + " |")
        lines.append("|---|---|" + "---|" * len(plans))
        for th in (1, 8):
            for e in EPS_ORDER:
                vals = []
                for p in plans:
                    x = t.get((p, e, th, src))
                    s = f_ms(x["median_of_query_medians_ms"]) if x else "-"
                    if x and x["nonfinite_groups_total"]:
                        s += " (non-finite)"
                    vals.append(s)
                lines.append(f"| {th} | {e} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def md_topk(t):
    lines = [f"\nvss {t['vss'].get('version')} ({t['vss'].get('install_mode')}), HNSW metric 'ip', "
             f"index build over the {t['setup']['rows_admitted']:,} admitted rows: "
             f"{t['index_build_seconds_adm_8threads']:.1f} s at 8 threads; "
             f"{t['setup']['facets_admitted']} facets in the admitted set. "
             f"Index used: {t['uses_hnsw']}.\n"]
    lines.append("| k | coverage mean (min-max) | recall@k vs exact mean (min) | rows returned min | "
                 "latency 1 thr ms | latency 8 thr ms | WHERE on full table: rows returned mean, groups mean |")
    lines.append("|---|---|---|---|---|---|---|")
    tim = {(x["plan"], x["k"], x["threads"]): x["median_of_query_medians_ms"] for x in t["timing"]}
    for c in t["coverage"]:
        k = c["k"]
        lines.append(f"| {k} | {c['coverage_mean']:.1%} ({c['coverage_min']:.1%}-{c['coverage_max']:.1%}) | "
                     f"{c['recall_mean']:.3f} ({c['recall_min']:.3f}) | {c['rows_returned_min']} | "
                     f"{f_ms(tim.get(('hnsw_topk_avg', k, 1)))} | {f_ms(tim.get(('hnsw_topk_avg', k, 8)))} | "
                     f"{c['filtered_full_table_rows_returned_mean']:.0f}, "
                     f"{c['filtered_full_table_groups_mean']:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res = {"meta": C.meta(), "timing_status": C.TIMING_STATUS}
    res["tests"] = prev.get("tests") if a.no_tests else run_tests()
    gap, grid, topk = load("gap.json"), load("grid.json"), load("topk.json")
    expb = load("exp_bench.json")
    win = load("window.json")
    if win:
        res["window"] = summarize_window(win)
    if expb:
        res["exp_bench"] = expb
    if gap:
        res["gap"] = {k: gap[k] for k in ("rounds", "cells", "diag", "load_before", "load_after") if k in gap}
    if grid:
        res["grid"] = summarize_grid(grid)
    if topk:
        res["topk"] = summarize_topk(topk)
    res["raw_runs"] = {n: os.path.join("runs", n) for n in ("gap.json", "grid.json", "topk.json", "exp_bench.json",
                                                               "window.json", "archive/gap_pre_core_p1.json")}
    C.dump(OUT, res)

    md = ["<!-- BEGIN GENERATED by collect.py from results.json; do not edit by hand -->",
          f"\nGenerated {res['meta']['timestamp']}; timing_status: **{C.TIMING_STATUS}**.",
          f"Tests: {res['tests']['pytest']['summary_line']} (pytest), "
          f"{'; '.join(res['tests']['cargo_test']['result_lines'])} (cargo).",
          "\n### Before/after on the aggregate itself (gap analysis, run_gap.py)"]
    if gap:
        md.append(md_gap(res["gap"]))
    if expb:
        md.append(f"\nglibc `exp` cost by argument range (exp_bench, best of {expb['passes']} passes over "
                  f"{expb['n']:,} arguments, pinned to one P-core):\n")
        md.append("| argument range | ns per exp |")
        md.append("|---|---|")
        for c in expb["cells"]:
            md.append(f"| {c['range']} | {c['ns_per_exp']:.2f} |")
    md.append("\n### IMDb grid (run_grid.py)")
    if grid:
        md.append(md_grid(res["grid"]))
    if win:
        md.append("\n### Sliding window, DuckDB segment tree over `combine` (run_window.py)")
        md.append(md_window(res["window"]))
    md.append("\n### vss HNSW top-k + AVG (run_topk.py)")
    if topk:
        md.append(md_topk(res["topk"]))
    md.append("\n<!-- END GENERATED -->")
    block = "\n".join(md)
    text = open(README).read()
    text = re.sub(r"<!-- BEGIN GENERATED.*?<!-- END GENERATED -->", lambda _: block, text, flags=re.S)
    open(README, "w").write(text)
    print("wrote", OUT, "and README tables")


if __name__ == "__main__":
    main()

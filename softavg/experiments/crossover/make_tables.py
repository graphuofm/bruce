"""Regenerate the tables in README.md (between the AUTO-TABLES markers) from results.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
BEGIN, END = "<!-- AUTO-TABLES BEGIN -->", "<!-- AUTO-TABLES END -->"


def f(x, p=2):
    return "-" if x is None else f"{x:.{p}g}"


def rng(lst):
    return "never" if not lst else (f"{lst[0]} .. {lst[-1]}" if len(lst) > 1 else lst[0])


out = []
for name, C in R["corpora"].items():
    out.append(f"\n### {name} ({C['rows_admitted']:,} admitted rows, {C['facets']} facets, "
               f"{C['predicate']}, smallest facet {min(C['facet_rows'].values()):,} rows)\n")
    out.append("Acceptable eps ranges (grid points; query acceptable = all facets covered and every "
               "facet error <= 1% of range):\n")
    out.append("| method | rows in set (median) | coverage mean | queries full cov. | eps acceptable, median query "
               "| eps acceptable, 90% of queries | median query, covered facets only |")
    out.append("|---|---|---|---|---|---|---|")
    for m, x in C["methods"].items():
        out.append(f"| {m} | {x['rows_in_candidate_set']['median']:.0f} | {x['coverage_mean']:.3f} | "
                   f"{x['queries_full_coverage']} | {rng(x['acceptable_eps_median_query'])} | "
                   f"{rng(x['acceptable_eps_90pct_queries'])} | {rng(x['acceptable_eps_median_query_covered_only'])} |")
    keys = list(C["ess_by_eps"])
    sel = keys[::3] + (["inf"] if keys[-1] == "inf" and "inf" not in keys[::3] else [])
    out.append("\nWorst covered-facet error / range, median over queries (every 3rd grid point):\n")
    out.append("| method | " + " | ".join(sel) + " |")
    out.append("|---|" + "---|" * len(sel))
    out.append("| ESS median | " + " | ".join(f(C["ess_by_eps"][k]["ess_median"], 3) for k in sel) + " |")
    out.append("| ESS / n_f median | " + " | ".join(f(C["ess_by_eps"][k]["ess_frac_median"]) for k in sel) + " |")
    for m, x in C["methods"].items():
        out.append(f"| {m} | " + " | ".join(f(x["per_eps"][k]["worst_facet_err_median_q"]) for k in sel) + " |")
    out.append("\nFraction of queries acceptable (same grid points):\n")
    out.append("| method | " + " | ".join(sel) + " |")
    out.append("|---|" + "---|" * len(sel))
    for m, x in C["methods"].items():
        out.append(f"| {m} | " + " | ".join(f"{x['per_eps'][k]['frac_queries_acceptable']:.2f}" for k in sel) + " |")
    out.append("\nESS diagnostic: fraction of (query, facet, eps) cells within 1%, binned by ESS:\n")
    for kind, title in (("by_ess_fraction", "ESS / n_f"), ("by_ess_absolute", "ESS (absolute)")):
        D = C["ess_diagnostic"][kind]
        bins = [b for b in next(iter(D.values())) if b["n"] > 0]
        out.append(f"| method \\ {title} bin | " + " | ".join(f"[{b['lo']:g}, {b['hi']:.3g})" for b in bins) + " |")
        out.append("|---|" + "---|" * len(bins))
        out.append("| cells | " + " | ".join(str(b["n"]) for b in bins) + " |")
        for m, bl in D.items():
            bl = [b for b in bl if b["n"] > 0]
            out.append(f"| {m} | " + " | ".join(f(b["frac_acceptable"]) for b in bl) + " |")
        out.append("")

readme = os.path.join(HERE, "README.md")
txt = open(readme).read()
a, b = txt.index(BEGIN) + len(BEGIN), txt.index(END)
open(readme, "w").write(txt[:a] + "\n" + "\n".join(out) + "\n" + txt[b:])
print("README tables regenerated")

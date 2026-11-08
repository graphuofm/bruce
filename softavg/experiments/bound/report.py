"""Figures and README tables from results.json (no number is typed by hand).

  python3 report.py      -> figures/*.pdf|png, and the block between
                            <!-- TABLES --> and <!-- /TABLES --> in README.md
"""
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
def _num(o):
    """JSON stores +-inf as strings; turn them back into floats"""
    if isinstance(o, dict):
        return {k: _num(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_num(v) for v in o]
    if o == "inf":
        return float("inf")
    if o == "-inf":
        return float("-inf")
    return o


R = _num(json.load(open(HERE / "results.json")))
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
CORP = [c for c in ["imdb", "amazon", "stackexchange"] if c in R["corpora"]]
NAME = {"imdb": "IMDb", "amazon": "Amazon", "stackexchange": "StackExchange"}
COL = {"imdb": "#2a78d6", "amazon": "#eb6834", "stackexchange": "#1baf7a"}
MARK = {"imdb": "o", "amazon": "s", "stackexchange": "^"}
SCEN = ["sliding_w50", "sliding_w400", "sliding_w5000", "churn", "adversarial"]
SCEN_LABEL = {"sliding_w50": "sliding w=50", "sliding_w400": "sliding w=400", "sliding_w5000": "sliding w=5000",
              "churn": "IVM churn", "adversarial": "descending deletes"}
EPS = R["corpora"][CORP[0]]["eps"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def cell(c, scen, eps):
    for x in R["corpora"][c]["cells"]:
        if x["scenario"] == scen and x["eps"] == eps:
            return x
    return None


def fe(x, d=1):
    if isinstance(x, float) and 0 < x <= 1.05e-26:
        return "<=1e-26"
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    return f"{x:.{d}e}".replace("e-0", "e-").replace("e+0", "e+")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------- figure 1
def fig_slack():
    fig, axs = plt.subplots(1, len(SCEN), figsize=(10.5, 2.4), sharey=True)
    for ax, sc in zip(axs, SCEN):
        for c in CORP:
            med, p99, mx = [], [], []
            for e in EPS:
                r = cell(c, sc, e)["shipped"]["ratio_vs_exact"]
                med.append(max(r["median"], 1e-20))
                p99.append(max(r["p99"], 1e-20))
                mx.append(min(max(r["max"], 1e-20), 2.5))
            ax.plot(EPS, mx, color=COL[c], marker=MARK[c], ms=4, lw=1.5, label=f"{NAME[c]} max")
            ax.plot(EPS, p99, color=COL[c], marker=MARK[c], ms=4, lw=1, ls="--", mfc="white")
            ax.plot(EPS, med, color=COL[c], marker=MARK[c], ms=3, lw=0.8, ls=":", mfc="white")
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 3)  # an infinite max (bound 0, error > 0) is drawn at the top
        ax.set_title(SCEN_LABEL[sc], fontsize=8, color=INK)
        ax.set_xlabel("eps")
    axs[0].set_ylabel("true error / bound\n(solid max, dashed p99, dotted median)")
    axs[0].legend(loc="lower left", fontsize=7)
    save(fig, "slack_ratio")


# ------------------------------------------------------------- figure 2
def fig_ops():
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True)
    for ax, sc in zip(axs, ["sliding_w50", "sliding_w5000", "churn"]):
        for c in CORP:
            for e, ls in [(0.02, "--"), (1.0, "-")]:
                b = cell(c, sc, e)["shipped"]["slack_by_ops_since_fresh"]
                xs = [2 ** (x["log2_ops_lo"] + 0.5) for x in b if x["states"] >= 50 and x["geomean_ratio_nonzero"]]
                ys = [x["geomean_ratio_nonzero"] for x in b if x["states"] >= 50 and x["geomean_ratio_nonzero"]]
                ax.plot(xs, ys, color=COL[c], ls=ls, lw=1.3, label=f"{NAME[c]}, eps={e:g}")
        x0 = [1e2, 1e6]
        ax.plot(x0, [0.3 * (x / 1e2) ** -0.5 for x in x0], color=MUTED, lw=0.8, ls=":", label="slope -1/2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(SCEN_LABEL[sc], fontsize=8, color=INK)
        ax.set_xlabel("ops since last fresh fold")
    axs[0].set_ylabel("geo-mean true error / bound")
    axs[-1].legend(fontsize=6, loc="lower left")
    save(fig, "slack_vs_ops")


# ------------------------------------------------------------- figure 3
def pooled(key, eps, field=None, scen=None):
    """sum refolds / refold_rows / max errors over corpora and scenarios"""
    out = {}
    for c in CORP:
        for sc in scen or SCEN:
            x = cell(c, sc, eps)
            items = x[key] if isinstance(x[key], list) else [x[key]]
            for it in items:
                lab = it.get("config", key)
                o = out.setdefault(lab, {"refolds": 0, "refold_rows": 0, "steps": 0, "max_err_over_vmax": 0.0,
                                         "err_above_tol": 0, "bound_violations_vs_exact": 0, "refolds_bound_infinite": 0})
                for k in ["refolds", "refold_rows", "steps", "err_above_tol", "bound_violations_vs_exact", "refolds_bound_infinite"]:
                    o[k] += it.get(k, 0)
                o["max_err_over_vmax"] = max(o["max_err_over_vmax"], it["max_err_over_vmax"])
    return out


def fig_tradeoff():
    fig, axs = plt.subplots(1, len(EPS), figsize=(10.5, 2.5), sharey=True)
    for ax, e in zip(axs, EPS):
        sw = pooled("sweep_local_copy", e)
        pts = [(cfg, v) for cfg, v in sw.items() if cfg.endswith("gain=100") and not cfg.startswith("abs=0")]
        pts.sort(key=lambda t: float(t[0].split(",")[0][4:]))
        xs = [max(v["refold_rows"] / v["steps"], 1e-7) for _, v in pts]
        ys = [max(v["max_err_over_vmax"], 1e-18) for _, v in pts]
        ax.plot(xs, ys, color=COL["imdb"], marker="o", ms=4, lw=1.2, label="shipped arithmetic, ABS_TOL sweep")
        for (cfg, v), x, y in zip(pts, xs, ys):
            a = cfg.split(",")[0][4:]
            if a in ("1e-14", "1e-10", "1e-6"):
                ax.annotate(a, (x, y), fontsize=6, color=MUTED, xytext=(3, 2), textcoords="offset points")
        dd = pooled("ddinv", e)
        dpts = sorted(dd.items(), key=lambda t: float(t[0].split(",")[0][4:]))
        ax.plot([max(v["refold_rows"] / v["steps"], 1e-7) for _, v in dpts],
                [max(v["max_err_over_vmax"], 1e-18) for _, v in dpts], color=COL["amazon"], marker="s", ms=4,
                lw=1.2, label="DdInv prototype")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"eps={e:g}", fontsize=8, color=INK)
        ax.set_xlabel("refolded rows per step\n(0 drawn at 1e-7)")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0].set_ylabel("max |error| / max|v|")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    save(fig, "tradeoff")


# ------------------------------------------------------------- figure 4
def fig_ess():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for c in CORP:
        es = R["corpora"][c]["ess"]
        x = [r["eps"] for r in es]
        ax.plot(x, [r["ess_median"] for r in es], color=COL[c], marker=MARK[c], ms=4, lw=1.5, label=NAME[c])
        ax.fill_between(x, [r["ess_p10"] for r in es], [r["ess_p90"] for r in es], color=COL[c], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eps")
    ax.set_ylabel("ESS per facet group\n(median, p10-p90 band)")
    ax.legend(fontsize=7)
    save(fig, "ess")


# ---------------------------------------------------------------- tables
def pooled_quantiles(cells, p_list):
    """quantiles of true error / bound pooled over cells, from the stored histograms"""
    zeros, above, cnt, mx, bins = 0, 0, 0, 0.0, {}
    per = None
    for x in cells:
        m = x["shipped"]
        h = m["ratio_vs_exact_hist"]
        per = h["per_decade"]
        zeros += h["zeros"]
        above += h["above"]
        cnt += m["ratio_vs_exact"]["count"]
        mx = max(mx, m["ratio_vs_exact"]["max"])
        for lo, c in h["bins"]:
            bins[lo] = bins.get(lo, 0) + c
    out = []
    for p in p_list:
        t = max(1, math.ceil(p * cnt))
        acc = zeros
        if acc >= t:
            out.append(0.0)
            continue
        v = mx
        for lo in sorted(bins):
            acc += bins[lo]
            if acc >= t:
                v = min(10 ** (lo + 1 / per), mx)
                break
        out.append(v)
    return cnt, out + [mx]


def headline():
    L = []
    P = L.append
    P("### T0. Headline, pooled over IMDb + Amazon + StackExchange and all queries\n")
    P("ratio = true error / bound (exact reference). Refolds: shipped bound / oracle (refold only when the true error "
      "exceeds the tolerance) / DdInv prototype. `rows/step` = refolded rows per step (host refold work).\n")
    P("| scenario | eps | states | violations | ratio median | p90 | p99 | max | refolds shipped | (bound infinite) | oracle | DdInv | rows/step shipped | rows/step DdInv |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in SCEN:
        for e in EPS:
            cs = [cell(c, sc, e) for c in CORP]
            n, q = pooled_quantiles(cs, [0.5, 0.9, 0.99])
            viol = sum(x["shipped"]["violations_vs_exact"] for x in cs)
            rs = sum(x["shipped"]["refolds"] for x in cs)
            ri = sum(x["shipped"]["refolds_bound_infinite"] for x in cs)
            ro = sum(x["oracle"]["refolds"] for x in cs)
            rd = sum(x["ddinv"][0]["refolds"] for x in cs)
            st = sum(x["shipped"]["steps"] for x in cs)
            rws = sum(x["shipped"]["refold_rows"] for x in cs) / st
            rwd = sum(x["ddinv"][0]["refold_rows"] for x in cs) / sum(x["ddinv"][0]["steps"] for x in cs)
            P(f"| {SCEN_LABEL[sc]} | {e:g} | {n:,} | {viol} | {fe(q[0])} | {fe(q[1])} | {fe(q[2])} | {fe(q[3], 2)} | "
              f"{rs:,} | {ri:,} | {ro:,} | {rd:,} | {rws:.3g} | {rwd:.3g} |")
    tot = sum(x["shipped"]["ratio_vs_exact"]["count"] for c in CORP for x in R["corpora"][c]["cells"])
    vio = sum(x["shipped"]["violations_vs_exact"] for c in CORP for x in R["corpora"][c]["cells"])
    mis = sum(x["shipped"]["core_vs_local_copy_mismatches"] for c in CORP for x in R["corpora"][c]["cells"])
    dvi = sum(d["bound_violations_vs_exact"] for c in CORP for x in R["corpora"][c]["cells"] for d in x["ddinv"])
    P(f"\nTotal reported states: {tot:,}; violations of the shipped bound against the exact value: {vio}; "
      f"core-vs-local-copy mismatches: {mis}; DdInv bound violations (all configs): {dvi}.\n")
    if "argpatch" in R:
        A = R["argpatch"]["corpora"]
        P("### T0b. Shipped bound vs the proposed soundness patch (same tolerance, exact reference)\n")
        P("Patch = also charge 2U|x| per weight (rounding of the exp argument x = (s - mu)/eps) and 2^-1074 per "
          "underflowed weight/product. Only cells where either variant has a violation or the refold counts differ "
          "are listed; `max viol err` is the largest |error| / max|v| among violating states.\n")
        P("| corpus (queries) | scenario | eps | refolds shipped | refolds patched | violations shipped | max viol err shipped | violations patched |")
        P("|---|---|---|---|---|---|---|---|")
        for c in CORP:
            if c not in A:
                continue
            for sc in SCEN:
                for e in EPS:
                    for x in A[c]["cells"]:
                        if x["scenario"] == sc and x["eps"] == e:
                            a, b = x["shipped"], x["patched"]
                            if a["bound_violations_vs_exact"] or b["bound_violations_vs_exact"] or a["refolds"] != b["refolds"]:
                                P(f"| {NAME[c]} ({A[c]['queries']}) | {SCEN_LABEL[sc]} | {e:g} | {a['refolds']:,} | {b['refolds']:,} | "
                                  f"{a['bound_violations_vs_exact']:,} | {fe(a['violations_max_abs_err_over_vmax'])} | "
                                  f"{b['bound_violations_vs_exact']:,} |")
        tot = {k: sum(x[k]["refolds"] for c in A for x in A[c]["cells"]) for k in ("shipped", "patched")}
        P(f"\nTotal refolds, all listed corpora/cells: shipped {tot['shipped']:,}, patched {tot['patched']:,}.\n")
    return L


def tables():
    L = headline()
    P = L.append
    P(f"Queries per corpus: {R['queries']} (sweeps and DdInv variants other than the default: first "
      f"{R['sweep_queries']}). Machine at start: `{json.dumps(R['machine_before'])}`.\n")
    P("### T1. true error / bound, shipped InvAcc, exact (double-double) reference, every reported state\n")
    for c in CORP:
        P(f"**{NAME[c]}** ({R['corpora'][c]['n_rows']:,} admitted rows, max|v| = {R['corpora'][c]['vmax']:g})\n")
        P("| scenario | eps | states | median | p90 | p99 | max | violations | bound/v_scale median | err/v_scale max |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for sc in SCEN:
            for e in EPS:
                m = cell(c, sc, e)["shipped"]
                r = m["ratio_vs_exact"]
                P(f"| {SCEN_LABEL[sc]} | {e:g} | {r['count']:,} | {fe(r['median'])} | {fe(r['p90'])} | {fe(r['p99'])} | "
                  f"{fe(r['max'], 2)} | {m['violations_vs_exact']} | {fe(m['bound_over_vscale']['median'])} | "
                  f"{fe(m['err_over_vscale']['max'])} |")
        P("")
    P("### T2. Against a fresh two-pass f64 fold instead (sampled states)\n")
    P("A violation here means the state and the f64 fold differ by more than the state's bound; the last column is the "
      "f64 fold's own error against the exact value.\n")
    P("| corpus | scenario | eps | sampled | ratio p99 | ratio max | violations vs f64 fold | f64 fold err / v_scale max |")
    P("|---|---|---|---|---|---|---|---|")
    for c in CORP:
        for sc in SCEN:
            for e in EPS:
                m = cell(c, sc, e)["shipped"]
                r = m["ratio_vs_f64_fold"]
                if m["violations_vs_f64_fold"] or sc in ("churn",):
                    P(f"| {NAME[c]} | {SCEN_LABEL[sc]} | {e:g} | {r['count']:,} | {fe(r['p99'])} | {fe(r['max'], 2)} | "
                      f"{m['violations_vs_f64_fold']} | {fe(m['f64_fold_err_over_vscale']['max'])} |")
    P("\n(rows shown: every cell with a violation, plus all churn cells; all other cells have 0 violations.)\n")
    P("### T3. What the bound is made of (mean share of the bound; all states / loose states with ratio < 1e-2)\n")
    P("Terms: `err_u/(z-err_z)`, `|F| err_z/(z-err_z)`, `U|F|`. Components: per-row weight terms (2U w, 3U|wv|), "
      "running-sum terms (U|z1|, U|u1|), anchor-rescale terms (3U|z0|scale, 3U|u0|scale), and U|F|.\n")
    P("| corpus | scenario | eps | loose / all | terms (u, z, div), loose | components (weight, running-sum, rescale, div), loose |")
    P("|---|---|---|---|---|---|")
    for c in CORP:
        for sc in SCEN:
            for e in EPS:
                m = cell(c, sc, e)["shipped"]
                t = ", ".join(f"{x:.2f}" for x in m["bound_terms_loose"])
                s = ", ".join(f"{x:.3f}" for x in m["bound_share_loose"])
                P(f"| {NAME[c]} | {SCEN_LABEL[sc]} | {e:g} | {m['states_loose']:,} / {m['states_all']:,} | {t} | {s} |")
    P("\n### T4. Refolds: shipped bound vs oracle vs DdInv prototype (same tolerance, all queries)\n")
    P("oracle = shipped arithmetic, refold only when the true error exceeds the tolerance (the floor for ANY bound on "
      "this arithmetic). `inf` = refolds where z no longer exceeded its own error bound (cancellation/underflow).\n")
    P("| corpus | scenario | eps | steps | shipped refolds (inf) | refolded rows / step | oracle refolds | DdInv refolds (inf) | "
      "max err/max|v| shipped | max err/max|v| DdInv | DdInv violations |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CORP:
        for sc in SCEN:
            for e in EPS:
                x = cell(c, sc, e)
                m, o, d = x["shipped"], x["oracle"], x["ddinv"][0]
                P(f"| {NAME[c]} | {SCEN_LABEL[sc]} | {e:g} | {m['steps']:,} | {m['refolds']:,} ({m['refolds_bound_infinite']:,}) | "
                  f"{m['refold_rows'] / m['steps']:.3g} | {o['refolds']:,} | {d['refolds']:,} ({d['refolds_bound_infinite']:,}) | "
                  f"{fe(m['max_err_over_vmax'])} | {fe(d['max_err_over_vmax'])} | {d['bound_violations_vs_exact']} |")
    P("\n### T5. Tolerance sweep (local copy of InvAcc, first sweep queries, pooled over corpora and scenarios)\n")
    P("Cell: refolded rows per step / max |error| / max|v|. `tol = max(abs, gain * n * U) * v_scale`.\n")
    P("| config | " + " | ".join(f"eps={e:g}" for e in EPS) + " |")
    P("|---|" + "---|" * len(EPS))
    cfgs = R["corpora"][CORP[0]]["sweep_configs"]
    pe = {e: pooled("sweep_local_copy", e) for e in EPS}
    for cfg in cfgs:
        P(f"| {cfg} | " + " | ".join(
            f"{pe[e][cfg]['refold_rows'] / pe[e][cfg]['steps']:.3g} / {fe(pe[e][cfg]['max_err_over_vmax'])}" for e in EPS) + " |")
    pd_ = {e: pooled("ddinv", e) for e in EPS}
    for cfg in R["corpora"][CORP[0]]["dd_configs"]:
        P(f"| DdInv {cfg} | " + " | ".join(
            f"{pd_[e][cfg]['refold_rows'] / pd_[e][cfg]['steps']:.3g} / {fe(pd_[e][cfg]['max_err_over_vmax'])}" for e in EPS) + " |")
    P("\nGAIN only matters for groups above ABS_TOL/(GAIN U) = 9,000 rows (churn groups); per-scenario numbers are in "
      "results.json -> corpora.*.cells[].sweep_local_copy.\n")
    if "bench" in R:
        b = R["bench"]
        P(f"### T6. Per-row cost (ns/row, minimum of {b['reps_min_of']} repetitions, {b['rows_per_rep']:,} rows; "
          f"timing_status: {b['timing_status']}; machine: `{json.dumps(R.get('machine_before_bench', {}))}`)\n")
        ops = [o["op"] for o in b["cells"][0]["ops"]]
        P("| eps | " + " | ".join(ops) + " |")
        P("|---|" + "---|" * len(ops))
        for cl in b["cells"]:
            P(f"| {cl['eps']:g} | " + " | ".join(f"{o['ns_per_row']:.2f} ({o['vs_scalar_absorb']:.2f}x)" for o in cl["ops"]) + " |")
        P(f"\nState sizes (bytes): `{json.dumps(b['state_bytes'])}`.\n")
    P("### T7. ESS = (sum w)^2 / sum w^2 per facet group (all admitted rows of a facet), all queries\n")
    P("| corpus | eps | groups | p10 | median | p90 | share ESS < 2 | share ESS < 10 | median ESS/n | max rel diff merge(7) vs sequential |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for c in CORP:
        for r in R["corpora"][c]["ess"]:
            P(f"| {NAME[c]} | {r['eps']:g} | {r['groups']} | {r['ess_p10']:.3g} | {r['ess_median']:.3g} | {r['ess_p90']:.3g} | "
              f"{r['frac_ess_below_2']:.2f} | {r['frac_ess_below_10']:.2f} | {r['ess_over_n_median']:.3g} | "
              f"{fe(r['max_rel_diff_merge7_vs_sequential'])} |")
    return "\n".join(L)


fig_slack()
fig_ops()
fig_tradeoff()
fig_ess()
readme = HERE / "README.md"
txt = readme.read_text() if readme.exists() else "<!-- TABLES -->\n<!-- /TABLES -->\n"
txt = re.sub(r"<!-- TABLES -->.*<!-- /TABLES -->", lambda _: "<!-- TABLES -->\n" + tables() + "\n<!-- /TABLES -->", txt, flags=re.S)
readme.write_text(txt)
print("figures/ and README tables written")

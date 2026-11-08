"""Assemble results.json from the step outputs (all under data/):
coverage.json (s2b), task.parquet stats (s3), results_raw.json (s4, delta
defaults), results_delta_{lo,hi}.json (s4, other deltas), engine_check.json
(s5).  Also prints the headline table the README quotes."""
import hashlib, json, platform, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"
J = lambda f: json.loads((D / f).read_text())

T = pd.read_parquet(D / "task.parquet", columns=["query_id", "label", "s", "price"])
lab = T[T.label != "D"]
pool = lab.groupby("query_id").size()
task_stats = dict(
    queries=int(T.query_id.nunique()),
    labelled_rows=int(len(lab)), distractor_rows=int((T.label == "D").sum()),
    labelled_pool_size_quantiles={str(q): float(pool.quantile(q)) for q in [0.1, 0.5, 0.9, 0.99]},
    label_counts={k: int(v) for k, v in lab.label.value_counts().items()},
    labelled_rows_with_price=int(lab.price.notna().sum()),
    mean_score_by_label={k: float(v) for k, v in T.groupby("label").s.mean().items()},
)
# how well does the title-query cosine separate E from non-E within a query?
# (Mann-Whitney AUC per query, labelled products only, queries with both kinds)
L = lab.assign(e=(lab.label == "E").astype(int))
L["r"] = L.groupby("query_id").s.rank()
agg = L.groupby("query_id").agg(n1=("e", "sum"), n=("e", "size"))
agg["r1"] = L[L.e == 1].groupby("query_id").r.sum()
agg = agg[(agg.n1 > 0) & (agg.n1 < agg.n)]
n0 = agg.n - agg.n1
auc = (agg.r1 - agg.n1 * (agg.n1 + 1) / 2) / (agg.n1 * n0)
task_stats["within_query_auc_E_vs_nonE"] = dict(
    queries=int(len(auc)), mean=float(auc.mean()), median=float(auc.median()))
gs = lab.groupby("query_id").s
task_stats["within_query_score_spread_labelled"] = dict(
    std_median=float(gs.std().median()), std_iqr=[float(x) for x in gs.std().quantile([.25, .75])],
    range_median=float((gs.max() - gs.min()).median()))
src = sorted((HERE / "scripts").glob("*.py"))
h = hashlib.sha256()
for p in src:
    h.update(p.name.encode()); h.update(p.read_bytes())

raw = J("results_raw.json")
sens = {}
for tag in ["delta_lo", "delta_hi"]:
    f = D / f"results_{tag}.json"
    if f.exists():
        r = json.loads(f.read_text())
        sens[tag] = {k: dict(delta=v["delta"],
                             chosen={n: dict(param=c["param"], test=c["test"])
                                     for n, c in v["chosen"].items()},
                             paired_diff_softavg_minus=v["paired_diff_softavg_minus"])
                     for k, v in r.items()}

FAMS = ["avg", "softavg", "thresh", "topk", "facet_topb", "thresh_relq", "thresh_relf",
        "softavg_gate", "thresh_softavg"]
headline = {}
for key, v in raw.items():
    row = {}
    for n in FAMS:
        c = v["chosen"][n]
        t = c["test"]
        row[n] = dict(param=c["param"], f1=t["findings_f1"], detected=t["findings_detected"],
                      false=t["findings_false"], gt=t["findings_gt"],
                      mae_mixed=t["mae_mixed"], coverage=t["coverage"],
                      facet_precision=t["facet_precision"],
                      pair_acc_informative=t["pair_acc_informative"], top1=t["top1_hit"],
                      f1_ci95=c["ci95"]["findings_f1"])
    headline[key] = row

out = dict(
    what=("ESCI (Amazon Shopping Queries) with human E/S/C/I labels as ground truth: does "
          "softavg over title-query cosine find more, and more correct, relevant-vs-overall "
          "differences per (query, facet) than AVG, global top-k + AVG, threshold + AVG?"),
    generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    scripts_sha256_16=h.hexdigest()[:16],
    python=platform.python_version(),
    encoder="sentence-transformers/all-MiniLM-L6-v2 (sentence-transformers 4.1.0, torch 2.6.0, "
            "/home/jding/bruce/sigmod2027/.venv; the default ~/.local env has a broken "
            "torch 2.4 / transformers 5.9 pair)",
    timing_status="no timings reported except engine_check seconds_smoke (smoke, machine shared)",
    protocol=dict(
        split="queries split 50/50 by md5(query_id) into dev/test; every hyperparameter "
              "(eps, tau, k, b, d) chosen on dev by findings-F1, reported on test",
        delta_default=dict(rating=0.2, log_price=0.2),
        bootstrap="1000 resamples of test queries, paired across estimators",
        min_rated_labelled_per_query=8, distractors_per_query=200,
        distractor_rule="random products from the ESCI US catalogue with a rating, same "
                        "main_category as the query's dominant one, not labelled for the query",
    ),
    coverage=J("coverage.json"),
    task_table=task_stats,
    engine_check=J("engine_check.json") if (D / "engine_check.json").exists() else None,
    figure_settings=["labelled|main_category|rating|E", "labelled|brand|log_price|E",
                     "labelled+distractors|brand|rating|E"],
    headline=headline,
    delta_sensitivity=sens,
    settings=raw,
)
(HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote results.json")

short = dict(avg="AVG", softavg="softavg", thresh="thr", topk="topk", facet_topb="topb",
             thresh_relq="relQ", thresh_relf="relF", softavg_gate="sa+gate",
             thresh_softavg="thr+sa")
for key, row in headline.items():
    print(f"\n{key}  (GT findings on test: {row['avg']['gt']})")
    for n in FAMS:
        r = row[n]
        print(f"  {short[n]:8s} {str(r['param']):24s} F1={r['f1']:.3f} "
              f"[{r['f1_ci95'][0]:.3f},{r['f1_ci95'][1]:.3f}] det={r['detected']:5d} "
              f"false={r['false']:6d} maeMix={r['mae_mixed'] or float('nan'):.3f} "
              f"cov={r['coverage']:.2f} prec={r['facet_precision']:.2f} "
              f"pairInf={r['pair_acc_informative'] or float('nan'):.3f} top1={r['top1'] or float('nan'):.3f}")

print("\nEps band: test findings-F1 of softavg per eps, vs the dev-chosen competitors' test F1")
band = {}
for key, v in raw.items():
    ref = {n: v["chosen"][n]["test"]["findings_f1"] for n in ["thresh", "topk", "thresh_relq", "thresh_relf"]}
    g = [(p["param"], p["test"]["findings_f1"]) for p in v["grid"]["softavg"]]
    beats = {n: [e for e, f in g if f > r] for n, r in ref.items()}
    band[key] = dict(softavg_test_f1_by_eps=g, competitor_test_f1=ref, eps_beating=beats)
    print(f"{key}: " + " ".join(f"{e:g}:{f:.3f}" for e, f in g))
    print("    " + "  ".join(f"{n}={r:.3f} beaten at eps in [{min(beats[n], default='-')},{max(beats[n], default='-')}]"
                          for n, r in ref.items()))
out["eps_band"] = band
(HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))

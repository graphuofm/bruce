"""eps crossover: exact softavg vs the host-native alternatives a user writes today.

For each corpus (IMDb, Amazon), 200 queries chosen as in PROTOCOL.md, the predicate of
PROTOCOL.md, and a dense eps grid (30 log-spaced values in [1e-4, 10], plus inf):

  reference          exact per-facet softavg over all admitted rows (numpy float64, max-shifted)
  plain_avg          AVG(value) GROUP BY facet (= softavg at eps = inf)
  topk_avg_k         idealised index: exact global top-k by score (numpy argpartition), AVG per facet
  topk_softavg_k     the same top-k, softavg computed client-side over it (what a vector-DB
                     user can do with returned scores)
  thresh_avg_tau     WHERE score >= tau, AVG per facet
  sample_avg_n       uniform random sample of n admitted rows (n = the k values), AVG per facet

Metric per (query, facet): |answer - reference| / range_f (range_f = max - min of value over
the facet's admitted rows); a facet with no row in the candidate set is "not covered".
A query is "acceptable" for a method at eps iff every facet is covered and every facet's
error is <= 1% of its range. Diagnostic: effective sample size ESS_f = (sum w)^2 / sum w^2.
"""
import argparse
import datetime
import json
import math
import os
import platform
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = "/data/softavg/corpora"
PRED = {"imdb": 2000, "amazon": 2015}
EPS = [float(x) for x in np.logspace(-4, 1, 30)] + [math.inf]
KS = [100, 1000, 4096, 10000]
TAUS = [0.3, 0.4, 0.5, 0.6]
TOL = 0.01
N_Q = 200
ESS_FRAC_BINS = [0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 0.6, 0.9, 1.0 + 1e-9]
ESS_ABS_BINS = [0, 1.5, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 1e5, 1e7]

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402


def eps_key(e):
    return "inf" if math.isinf(e) else f"{e:.6g}"


def choose_queries(text):
    rng = np.random.RandomState(0)
    titles = text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    return rng.choice(np.flatnonzero(ok.values), size=N_Q, replace=False)


def group_mean(codes, v, G, idx=None):
    if idx is not None:
        codes, v = codes[idx], v[idx]
    num = np.bincount(codes, weights=v, minlength=G)
    cnt = np.bincount(codes, minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / cnt, cnt > 0


def group_softavg(codes, s, v, G, eps):
    """Max-shifted softavg per group, plus ESS = (sum w)^2 / sum w^2."""
    if math.isinf(eps):
        m, ok = group_mean(codes, v, G)
        cnt = np.bincount(codes, minlength=G).astype(np.float64)
        return m, ok, cnt
    mx = np.full(G, -np.inf)
    np.maximum.at(mx, codes, s)
    w = np.exp((s - mx[codes]) / eps)
    den = np.bincount(codes, weights=w, minlength=G)
    num = np.bincount(codes, weights=w * v, minlength=G)
    w2 = np.bincount(codes, weights=w * w, minlength=G)
    ok = den > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, ok, den * den / w2


def load_conditions():
    try:
        c = quiet.require_quiet(wait_seconds=0, verbose=False)
        c["quiet"] = True
    except SystemExit as e:
        c = {"quiet": False, "require_quiet_message": str(e)}
    c["at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return c


def run_corpus(name, nq, log):
    d = os.path.join(CORP, name)
    df = pd.read_parquet(os.path.join(d, "corpus.parquet"), columns=["facet", "value", "filter_num", "text"])
    qrows = choose_queries(df.text)[:nq]
    emb = np.load(os.path.join(d, "emb.npy"), mmap_mode="r")
    adm = np.flatnonzero(df.filter_num.values >= PRED[name])
    codes, facets = pd.factorize(df.facet.values[adm])
    G = len(facets)
    v = df.value.values[adm].astype(np.float64)
    del df
    lo = np.full(G, np.inf); hi = np.full(G, -np.inf)
    np.minimum.at(lo, codes, v); np.maximum.at(hi, codes, v)
    rng_f = hi - lo
    n_f = np.bincount(codes, minlength=G).astype(np.float64)
    Q = np.asarray(emb[qrows], dtype=np.float64)                     # nq x 384
    S = np.empty((len(qrows), len(adm)), dtype=np.float64)           # scores, float64
    step = 100_000
    for a0 in range(0, len(adm), step):
        S[:, a0:a0 + step] = Q @ np.asarray(emb[adm[a0:a0 + step]], dtype=np.float64).T
    log(f"{name}: {len(adm)} admitted rows, {G} facets, scores computed")

    methods = (["plain_avg"] + [f"topk_avg_k{k}" for k in KS] + [f"topk_softavg_k{k}" for k in KS]
               + [f"thresh_avg_tau{t}" for t in TAUS] + [f"sample_avg_n{k}" for k in KS])
    E = len(EPS)
    # per (method, eps, query): worst facet error, median facet error, covered facets, acceptable facets
    worst = {m: np.full((E, nq), np.nan) for m in methods}
    medf = {m: np.full((E, nq), np.nan) for m in methods}
    cov = {m: np.zeros(nq, dtype=int) for m in methods}
    okfac = {m: np.zeros((E, nq), dtype=int) for m in methods}
    ess_frac = np.zeros((E, nq, G)); ess_abs = np.zeros((E, nq, G))
    # per (method, eps, query, facet) acceptability, kept only for the ESS diagnostic
    acc_pf = {m: np.zeros((E, nq, G), dtype=bool) for m in methods}
    rows_in = {m: [] for m in methods}
    score_stats = []
    rs = np.random.RandomState(12345)
    for qi in range(len(qrows)):
        s = S[qi]
        score_stats.append([float(s.max()), *[float(x) for x in np.percentile(s, [50, 99, 99.9])]])
        cand = {"plain_avg": None}
        order = np.argpartition(-s, KS[-1])[: KS[-1]]
        order = order[np.argsort(-s[order], kind="stable")]
        for k in KS:
            cand[f"topk_avg_k{k}"] = order[:k]
            cand[f"topk_softavg_k{k}"] = order[:k]
            cand[f"sample_avg_n{k}"] = rs.choice(len(adm), size=k, replace=False)
        for t in TAUS:
            cand[f"thresh_avg_tau{t}"] = np.flatnonzero(s >= t)
        fixed = {}
        for m, idx in cand.items():
            rows_in[m].append(len(adm) if idx is None else len(idx))
            if not m.startswith("topk_softavg"):
                fixed[m] = group_mean(codes, v, G, idx)
        for ei, e in enumerate(EPS):
            ref, rok, ess = group_softavg(codes, s, v, G, e)
            assert rok.all()
            ess_abs[ei, qi] = ess
            ess_frac[ei, qi] = ess / n_f
            for m, idx in cand.items():
                if m.startswith("topk_softavg"):
                    ans, ok = group_softavg(codes[idx], s[idx], v[idx], G, e)[:2]
                else:
                    ans, ok = fixed[m]
                err = np.abs(ans - ref) / rng_f
                if ei == 0:
                    cov[m][qi] = int(ok.sum())
                if ok.any():
                    worst[m][ei, qi] = float(err[ok].max())
                    medf[m][ei, qi] = float(np.median(err[ok]))
                good = ok & (err <= TOL)
                okfac[m][ei, qi] = int(good.sum())
                acc_pf[m][ei, qi] = good
        if qi % 25 == 0:
            log(f"{name}: query {qi + 1}/{len(qrows)}")
    del S

    def acceptable_eps(frac, thr):
        """eps values (grid order) where the fraction of acceptable queries >= thr."""
        return [eps_key(EPS[i]) for i in range(E) if frac[i] >= thr]

    out_m = {}
    for m in methods:
        full = cov[m] == G
        acc_q = np.array([(okfac[m][ei] == G) for ei in range(E)])          # E x nq
        frac = acc_q.mean(axis=1)
        frac_cov = np.array([np.mean(okfac[m][ei] == cov[m]) for ei in range(E)])
        pf = np.array([okfac[m][ei].sum() / (nq * G) for ei in range(E)])
        out_m[m] = {
            "rows_in_candidate_set": {"median": float(np.median(rows_in[m])), "min": int(np.min(rows_in[m])),
                                      "max": int(np.max(rows_in[m]))},
            "coverage_mean": float(cov[m].mean() / G), "coverage_min": float(cov[m].min() / G),
            "queries_full_coverage": int(full.sum()),
            "per_eps": {eps_key(EPS[ei]): {
                "worst_facet_err_median_q": None if np.all(np.isnan(worst[m][ei])) else float(np.nanmedian(worst[m][ei])),
                "worst_facet_err_p90_q": None if np.all(np.isnan(worst[m][ei])) else float(np.nanpercentile(worst[m][ei], 90)),
                "worst_facet_err_max_q": None if np.all(np.isnan(worst[m][ei])) else float(np.nanmax(worst[m][ei])),
                "median_facet_err_median_q": None if np.all(np.isnan(medf[m][ei])) else float(np.nanmedian(medf[m][ei])),
                "frac_queries_acceptable": float(frac[ei]),
                "frac_queries_acceptable_on_covered_facets": float(frac_cov[ei]),
                "frac_query_facets_acceptable": float(pf[ei]),
            } for ei in range(E)},
            "acceptable_eps_median_query": acceptable_eps(frac, 0.5),
            "acceptable_eps_90pct_queries": acceptable_eps(frac, 0.9),
            "acceptable_eps_median_query_covered_only": acceptable_eps(frac_cov, 0.5),
        }

    # ESS diagnostic: per eps, median ESS (absolute and as a fraction of n_f) over (query, facet)
    ess_eps = {eps_key(EPS[ei]): {"ess_median": float(np.median(ess_abs[ei])),
                                  "ess_frac_median": float(np.median(ess_frac[ei])),
                                  "ess_frac_p10": float(np.percentile(ess_frac[ei], 10)),
                                  "ess_frac_p90": float(np.percentile(ess_frac[ei], 90)),
                                  "ess_min": float(np.min(ess_abs[ei]))} for ei in range(E)}
    # acceptability as a function of ESS: plain_avg / sample vs ESS fraction; top-k vs absolute ESS
    def binned(m, arr, bins):
        a = acc_pf[m].reshape(-1); x = arr.reshape(-1)
        res = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            sel = (x >= b0) & (x < b1)
            res.append({"lo": b0, "hi": b1, "n": int(sel.sum()),
                        "frac_acceptable": float(a[sel].mean()) if sel.any() else None})
        return res
    ess_diag = {"by_ess_fraction": {m: binned(m, ess_frac, ESS_FRAC_BINS)
                                    for m in ["plain_avg"] + [f"sample_avg_n{k}" for k in KS]
                                    + [f"thresh_avg_tau{t}" for t in TAUS]},
                "by_ess_absolute": {m: binned(m, ess_abs, ESS_ABS_BINS)
                                    for m in [f"topk_avg_k{k}" for k in KS] + [f"topk_softavg_k{k}" for k in KS]}}
    ss = np.array(score_stats)
    return {"rows_admitted": int(len(adm)), "facets": G, "predicate": f"filter_num >= {PRED[name]}",
            "query_rows": [int(r) for r in qrows],
            "facet_rows": {str(f): int(c) for f, c in zip(facets, n_f)},
            "facet_range": {str(f): float(r) for f, r in zip(facets, rng_f)},
            "score_stats_median_over_queries": {"max": float(np.median(ss[:, 0])), "p50": float(np.median(ss[:, 1])),
                                                "p99": float(np.median(ss[:, 2])), "p99.9": float(np.median(ss[:, 3]))},
            "methods": out_m, "ess_by_eps": ess_eps, "ess_diagnostic": ess_diag}


def clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return "nan" if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {str(k): clean(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=N_Q)
    ap.add_argument("--corpora", default="imdb,amazon")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    a = ap.parse_args()
    log = lambda m: print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)  # noqa: E731
    out = {"meta": {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "timing_status": "smoke, machine shared (no timings are reported here; accuracy only)",
                    "host": platform.node(), "python": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "build_type": "n/a (numpy float64 reference computation; no extension involved)",
                    "eps_grid": [eps_key(e) for e in EPS], "ks": KS, "taus": TAUS, "tolerance": TOL,
                    "queries_per_corpus": a.nq,
                    "query_rule": "RandomState(0); first sentence 4-20 words, de-duplicated; 200 rows",
                    "acceptable_definition": "query acceptable iff all facets covered and every facet error "
                                             "<= 1% of the facet's value range",
                    "sample_seed": 12345},
           "load_before": load_conditions(), "corpora": {}}
    for name in a.corpora.split(","):
        out["corpora"][name] = run_corpus(name, a.nq, log)
    out["load_after"] = load_conditions()
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(clean(out), f, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    log("done")


if __name__ == "__main__":
    sys.exit(main())

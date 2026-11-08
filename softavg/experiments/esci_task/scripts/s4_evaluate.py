"""Step 3: estimators vs ESCI ground truth.

For each setting (pool x facet x value x ground-truth label set) and each
(query, facet) group:
  GT    = mean value over the group's products labelled relevant
          (E, or E+S), undefined if the group has none;
  base  = plain AVG over every product of the group in the pool.
Estimators (NaN = no answer for that facet):
  avg               plain AVG (= softavg at eps = inf)
  softavg@eps       sum e^{s/eps} v / sum e^{s/eps} over the group
  softavg_gate      softavg@eps, facet reported only if max(s) >= tau
                    (HAVING MAX(s) >= tau; both parameters picked on dev)
  topk@k            global top-k of the query by s, then AVG per facet
  thresh@tau        AVG over s >= tau
  facet_topb@b      AVG over the b highest-scoring products of the facet
  thresh_relq@d     AVG over s >= max_query(s) - d (per-query relative cut)
  thresh_relf@d     AVG over s >= max_facet(s) - d (per-facet box: the hard
                    analogue of softavg's within-facet relative weighting)
  thresh_softavg    WHERE s >= tau, then softavg@eps ((tau, eps) on dev)
Metrics on the test half of the queries (hyperparameters chosen on the dev
half by findings-F1; the whole test grid is reported too):
  value error       MAE vs GT on GT facets the estimator answers; coverage
                    = answered GT facets / GT facets; also on "mixed"
                    facets (both relevant and non-relevant products), the
                    only facets where the estimators can differ at all
  facet precision   answered facets that have >= 1 relevant product
  ranking           pairwise ordering accuracy of GT facets (Kendall
                    tau-a = 2*acc-1; untied GT pairs only, a pair with a
                    missing estimate counts as wrong), top-1 hit (the
                    estimator's best answered facet is a GT-best facet),
                    queries with >= 2 GT facets
  findings          GT finding: |GT - base| > delta, sign(GT - base).
                    Detected: estimator answers and est - base has the
                    same sign with |est - base| > delta.  False finding:
                    |est - base| > delta where GT is undefined, within
                    delta of base, or of the opposite sign.
95% CIs: bootstrap over test queries (1000 resamples, paired across
estimators).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"

EPS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
TAUS = [round(x, 3) for x in np.arange(0.20, 0.901, 0.025)]
KS = [5, 10, 20, 30, 50, 100, 200]
BS = [1, 2, 3, 5]
DS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
DELTA = {"rating": 0.2, "log_price": 0.2}


class Setting:
    """Rows of one setting sorted by (query, facet, -s), with group indices."""

    def __init__(self, t, facet, value, rel_labels):
        t = t[t[facet].notna() & t[value].notna()]
        t = t.sort_values(["query_id", facet, "s"], ascending=[True, True, False],
                          kind="mergesort").reset_index(drop=True)
        self.q = t.query_id.values
        self.s = t.s.values.astype(np.float64)
        self.v = t[value].values.astype(np.float64)
        self.rel = t.label.isin(rel_labels).values
        newg = np.r_[True, (t.query_id.values[1:] != t.query_id.values[:-1])
                     | (t[facet].values[1:] != t[facet].values[:-1])]
        self.gstart = np.flatnonzero(newg)
        self.gid = np.cumsum(newg) - 1
        G = len(self.gstart)
        self.G = G
        self.gq = self.q[self.gstart]
        self.rank_g = np.arange(len(t)) - self.gstart[self.gid]
        # rank within query by descending s
        order = np.lexsort((-self.s, self.q))
        qs = self.q[order]
        qstart = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
        qidx = np.cumsum(np.r_[True, qs[1:] != qs[:-1]]) - 1
        r = np.empty(len(t), np.int64)
        r[order] = np.arange(len(t)) - qstart[qidx]
        self.rank_q = r
        n = self.red(np.ones_like(self.s))
        nrel = self.red(self.rel.astype(float))
        self.n = n
        self.base = self.red(self.v) / n
        self.gt = np.where(nrel > 0, self.red(self.v * self.rel) / np.maximum(nrel, 1), np.nan)
        self.has_gt = nrel > 0
        self.mixed = (nrel > 0) & (nrel < n)
        self.maxs = self.s[self.gstart]
        # query index per group (groups are sorted by query)
        qnew = np.r_[True, self.gq[1:] != self.gq[:-1]]
        self.gqi = np.cumsum(qnew) - 1
        self.qstart_g = np.flatnonzero(qnew)
        self.qids = self.gq[self.qstart_g]
        self.Q = len(self.qids)
        # untied GT pairs within a query, for ranking
        gi = np.flatnonzero(self.has_gt)
        pa, pb = [], []
        qq = self.gqi[gi]
        bounds = np.r_[0, np.flatnonzero(qq[1:] != qq[:-1]) + 1, len(gi)]
        for a0, a1 in zip(bounds[:-1], bounds[1:]):
            m = a1 - a0
            if m < 2:
                continue
            ii, jj = np.triu_indices(m, 1)
            pa.append(gi[a0 + ii]); pb.append(gi[a0 + jj])
        pa = np.concatenate(pa) if pa else np.array([], int)
        pb = np.concatenate(pb) if pb else np.array([], int)
        keep = self.gt[pa] != self.gt[pb]
        self.pa, self.pb = pa[keep], pb[keep]
        # informative pairs: at least one side mixed (only there can estimators disagree)
        self.pinf = self.mixed[self.pa] | self.mixed[self.pb]
        ngt_q = np.bincount(self.gqi, weights=self.has_gt, minlength=self.Q)
        self.rank_q_ok = ngt_q >= 2

    def red(self, x):
        return np.add.reduceat(x, self.gstart) if len(x) else x

    # ---- estimators -------------------------------------------------
    def masked_avg(self, mask):
        c = self.red(mask.astype(float))
        sm = self.red(np.where(mask, self.v, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, sm / c, np.nan)

    def softavg(self, eps):
        if eps == np.inf:
            return self.base.copy()
        w = np.exp((self.s - self.maxs[self.gid]) / eps)
        return self.red(w * self.v) / self.red(w)

    def thresh(self, tau):
        return self.masked_avg(self.s >= tau)

    def topk(self, k):
        return self.masked_avg(self.rank_q < k)

    def facet_topb(self, b):
        return self.masked_avg(self.rank_g < b)

    def thresh_relq(self, d):
        """AVG over s >= (max score of the query) - d."""
        qmax = np.maximum.reduceat(self.maxs, self.qstart_g)[self.gqi]
        return self.masked_avg(self.s >= qmax[self.gid] - d)

    def thresh_relf(self, d):
        """AVG over s >= (max score of the facet) - d: the hard (box) version of
        softavg's within-facet relative weighting; never drops a facet."""
        return self.masked_avg(self.s >= self.maxs[self.gid] - d)

    # ---- per-query metric contributions ------------------------------
    def contrib(self, est, delta):
        """Per-query counts; summing over a query sample gives the metrics."""
        Q, qi = self.Q, self.gqi
        ans = ~np.isnan(est)
        bc = lambda w: np.bincount(qi, weights=w.astype(float), minlength=Q)
        err = np.where(ans & self.has_gt, np.abs(est - np.nan_to_num(self.gt)), 0.0)
        mx = self.mixed
        d_gt = self.gt - self.base
        gt_find = self.has_gt & (np.abs(d_gt) > delta)
        d_est = est - self.base
        claim = ans & (np.abs(d_est) > delta)
        hit = claim & gt_find & (np.sign(d_est) == np.sign(d_gt))
        out = dict(
            n_gt=bc(self.has_gt), n_ans_gt=bc(ans & self.has_gt), abs_err=bc(err),
            n_mixed=bc(mx), n_ans_mixed=bc(ans & mx), abs_err_mixed=bc(np.where(mx, err, 0)),
            n_ans=bc(ans), n_ans_spurious=bc(ans & ~self.has_gt),
            n_find=bc(gt_find), n_claim=bc(claim), n_hit=bc(hit),
        )
        # ranking: pairwise accuracy over untied GT pairs
        ea, eb = est[self.pa], est[self.pb]
        sg = np.sign(self.gt[self.pa] - self.gt[self.pb])
        se = np.sign(ea - eb)
        ok = ~np.isnan(ea) & ~np.isnan(eb)
        score = np.where(ok, np.where(se == sg, 1.0, np.where(se == 0, 0.5, 0.0)), 0.0)
        pq = qi[self.pa]
        out["n_pairs"] = np.bincount(pq, minlength=Q).astype(float)
        out["pair_score"] = np.bincount(pq, weights=score, minlength=Q)
        inf = self.pinf
        out["n_pairs_inf"] = np.bincount(pq, weights=inf.astype(float), minlength=Q)
        out["pair_score_inf"] = np.bincount(pq, weights=score * inf, minlength=Q)
        out["n_pairs_inf_ans"] = np.bincount(pq, weights=(inf & ok).astype(float), minlength=Q)
        # top-1: estimator's best answered facet vs GT-best
        e2 = np.where(ans, est, -np.inf)
        emax = np.maximum.reduceat(e2, self.qstart_g)
        gmax = np.maximum.reduceat(np.where(self.has_gt, self.gt, -np.inf), self.qstart_g)
        is_top = (e2 == emax[qi]) & ans
        # first group attaining the max (deterministic tie-break by facet order)
        first = np.full(Q, -1)
        idx = np.flatnonzero(is_top)
        first_rev = idx[::-1]
        first[qi[first_rev]] = first_rev
        top1 = np.zeros(Q)
        okq = first >= 0
        top1[okq] = (self.gt[first[okq]] == gmax[okq]).astype(float)
        out["top1"] = np.where(self.rank_q_ok, top1, 0.0)
        out["top1_n"] = self.rank_q_ok.astype(float)
        return out


def summarise(c, sel):
    S = {k: v[sel].sum() for k, v in c.items()}
    f = lambda a, b: float(a / b) if b > 0 else None
    P = f(S["n_hit"], S["n_claim"])
    R = f(S["n_hit"], S["n_find"])
    F1 = (2 * P * R / (P + R)) if (P and R) else 0.0
    nq = int(sel.sum())
    return dict(
        mae=f(S["abs_err"], S["n_ans_gt"]), coverage=f(S["n_ans_gt"], S["n_gt"]),
        mae_mixed=f(S["abs_err_mixed"], S["n_ans_mixed"]),
        coverage_mixed=f(S["n_ans_mixed"], S["n_mixed"]),
        facet_precision=f(S["n_ans"] - S["n_ans_spurious"], S["n_ans"]),
        spurious_facets_per_query=f(S["n_ans_spurious"], nq),
        pair_acc=f(S["pair_score"], S["n_pairs"]),
        kendall_tau_a=(2 * f(S["pair_score"], S["n_pairs"]) - 1) if S["n_pairs"] else None,
        pair_acc_informative=f(S["pair_score_inf"], S["n_pairs_inf"]),
        pair_acc_informative_answered=f(S["pair_score_inf"], S["n_pairs_inf_ans"]),
        top1_hit=f(S["top1"], S["top1_n"]),
        findings_gt=int(S["n_find"]), findings_detected=int(S["n_hit"]),
        findings_false=int(S["n_claim"] - S["n_hit"]),
        findings_recall=R, findings_precision=P, findings_f1=F1,
    )


def boot(c, sel_idx, B, rng, keys):
    """Bootstrap the listed metrics; returns {metric: (lo, hi)} and draws."""
    draws = {k: [] for k in keys}
    n = len(sel_idx)
    cs = {k: v[sel_idx] for k, v in c.items()}
    for _ in range(B):
        r = rng.randint(0, n, n)
        S = {k: v[r].sum() for k, v in cs.items()}
        P = S["n_hit"] / S["n_claim"] if S["n_claim"] else 0
        R = S["n_hit"] / S["n_find"] if S["n_find"] else 0
        vals = dict(findings_f1=2 * P * R / (P + R) if P + R else 0,
                    findings_recall=R, findings_precision=P,
                    findings_detected=S["n_hit"], findings_false=S["n_claim"] - S["n_hit"],
                    mae_mixed=S["abs_err_mixed"] / max(S["n_ans_mixed"], 1),
                    coverage=S["n_ans_gt"] / max(S["n_gt"], 1),
                    pair_acc=S["pair_score"] / max(S["n_pairs"], 1),
                    pair_acc_informative=S["pair_score_inf"] / max(S["n_pairs_inf"], 1),
                    top1_hit=S["top1"] / max(S["top1_n"], 1))
        for k in keys:
            draws[k].append(vals[k])
    return {k: np.array(v) for k, v in draws.items()}


def families(st):
    fam = {
        "avg": [(None, lambda: st.softavg(np.inf))],
        "softavg": [(e, (lambda e=e: lambda: st.softavg(e))()) for e in EPS],
        "thresh": [(t, (lambda t=t: lambda: st.thresh(t))()) for t in TAUS],
        "topk": [(k, (lambda k=k: lambda: st.topk(k))()) for k in KS],
        "facet_topb": [(b, (lambda b=b: lambda: st.facet_topb(b))()) for b in BS],
        "thresh_relq": [(d, (lambda d=d: lambda: st.thresh_relq(d))()) for d in DS],
        "thresh_relf": [(d, (lambda d=d: lambda: st.thresh_relf(d))()) for d in DS],
    }
    return fam


def run_setting(t, qsplit, facet, value, rel, B, rng):
    st = Setting(t, facet, value, rel)
    dev = np.array([qsplit.get(q) == "dev" for q in st.qids])
    test = ~dev
    delta = DELTA[value]
    res = dict(n_queries_dev=int(dev.sum()), n_queries_test=int(test.sum()),
               n_rows=int(len(st.s)), n_groups=int(st.G),
               n_gt_groups_test=int(st.has_gt[test[st.gqi]].sum()),
               n_mixed_groups_test=int(st.mixed[test[st.gqi]].sum()),
               delta=delta, grid={}, chosen={})
    contribs = {}
    sa_cache = {}
    for name, items in families(st).items():
        res["grid"][name] = []
        best = None
        for p, fn in items:
            est = fn()
            if name == "softavg":
                sa_cache[p] = est
            c = c_dev = st.contrib(est, delta)
            m_dev = summarise(c, dev)
            m_test = summarise(c, test)
            res["grid"][name].append(dict(param=p, dev=m_dev, test=m_test))
            if best is None or m_dev["findings_f1"] > best[0]:
                best = (m_dev["findings_f1"], p, c, m_test)
        res["chosen"][name] = dict(param=best[1], test=best[3])
        contribs[name] = best[2]
    # softavg with a facet gate HAVING max(s) >= tau, (eps, tau) chosen on dev
    best = None
    for e, est in sa_cache.items():
        for tau in TAUS[::2]:
            g = np.where(st.maxs >= tau, est, np.nan)
            c = st.contrib(g, delta)
            m_dev = summarise(c, dev)
            if best is None or m_dev["findings_f1"] > best[0]:
                best = (m_dev["findings_f1"], (e, tau), c, summarise(c, test))
    res["chosen"]["softavg_gate"] = dict(param=list(best[1]), test=best[3])
    contribs["softavg_gate"] = best[2]
    # WHERE s >= tau, then softavg (threshold prefilter + softavg), (tau, eps) on dev
    best = None
    res["grid"]["thresh_softavg"] = []
    for tau in TAUS[::2]:
        keep = st.s >= tau
        for e in EPS:
            w = np.where(keep, np.exp((st.s - st.maxs[st.gid]) / e), 0.0)
            den = st.red(w)
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(den > 0, st.red(w * st.v) / den, np.nan)
            c = st.contrib(est, delta)
            m_dev, m_test = summarise(c, dev), summarise(c, test)
            res["grid"]["thresh_softavg"].append(dict(param=[tau, e], dev=m_dev, test=m_test))
            if best is None or m_dev["findings_f1"] > best[0]:
                best = (m_dev["findings_f1"], (tau, e), c, m_test)
    res["chosen"]["thresh_softavg"] = dict(param=list(best[1]), test=best[3])
    contribs["thresh_softavg"] = best[2]
    # paired bootstrap CIs on test queries
    keys = ["findings_f1", "findings_recall", "findings_precision", "findings_detected",
            "findings_false", "mae_mixed", "coverage", "pair_acc", "pair_acc_informative",
            "top1_hit"]
    idx = np.flatnonzero(test)
    draws = {}
    for name, c in contribs.items():
        r2 = np.random.RandomState(12345)  # same resamples for every estimator
        draws[name] = boot(c, idx, B, r2, keys)
        res["chosen"][name]["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                       for k, v in draws[name].items()}
    # paired differences softavg - competitor
    res["paired_diff_softavg_minus"] = {}
    for other in ["thresh", "topk", "facet_topb", "thresh_relq", "thresh_relf",
                  "softavg_gate", "thresh_softavg", "avg"]:
        res["paired_diff_softavg_minus"][other] = {
            k: dict(mean=float(np.mean(draws["softavg"][k] - draws[other][k])),
                    ci95=[float(np.percentile(draws["softavg"][k] - draws[other][k], 2.5)),
                          float(np.percentile(draws["softavg"][k] - draws[other][k], 97.5))])
            for k in ["findings_f1", "findings_detected", "findings_false", "mae_mixed",
                      "pair_acc_informative", "top1_hit"]}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(HERE / "data" / "results_raw.json"))
    ap.add_argument("--pools", default="labelled,labelled+distractors")
    ap.add_argument("--delta-rating", type=float, default=DELTA["rating"])
    ap.add_argument("--delta-price", type=float, default=DELTA["log_price"])
    a = ap.parse_args()
    DELTA.update(rating=a.delta_rating, log_price=a.delta_price)
    t0 = time.time()
    T = pd.read_parquet(D / "task.parquet")
    T["log_price"] = np.where(T.price > 0, np.log(T.price.where(T.price > 0)), np.nan)
    qs = pd.read_parquet(D / "queries.parquet")
    qsplit = dict(zip(qs.query_id, qs.split))
    rng = np.random.RandomState(0)
    out = {}
    for pool in a.pools.split(","):
        tp = T if pool != "labelled" else T[T.label != "D"]
        for facet in ["brand", "main_category"]:
            for value in ["rating", "log_price"]:
                for relname, rel in [("E", ["E"]), ("E+S", ["E", "S"])]:
                    key = f"{pool}|{facet}|{value}|{relname}"
                    out[key] = run_setting(tp, qsplit, facet, value, rel, a.boot, rng)
                    ch = out[key]["chosen"]
                    print(f"[{time.time()-t0:.0f}s] {key}: "
                          + " ".join(f"{n}={ch[n]['param']}:F1={ch[n]['test']['findings_f1']:.3f}"
                                     for n in ch), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)

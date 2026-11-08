"""numpy float64 references, per-engine grading, results.json and the generated README tables.

Two references per cell (q, eps, frame k), both numpy float64, computed per frame from scratch
(np.sum over the frame, pairwise summation of non-negative terms, no subtraction):
  * formula reference F: exactly the SQL's own formula, w = exp((s - 1)/eps) summed fresh per frame.
    Deviation from F is what the host's window implementation adds. Where F itself is 0/0 (every
    weight in the frame underflows, eps = 1e-3) the formula, not the host, is at fault.
  * true reference T: the softmax-weighted average the prompt asked for, max-shifted per frame.
"""
import json, math, sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from common import (EPS, FRAMES, HERE, MINIMAL, NQ, OUT, WORK, cell_key, cells, load_df, minimal_exact)

TOL = 1e-6
REFS = WORK / "refs.npz"


def partitions(df):
    cat = df.category.values
    order = np.lexsort((df.id.values, cat))       # ts is monotone in id
    c = cat[order]
    cut = np.flatnonzero(c[1:] != c[:-1]) + 1
    return np.split(order, cut)


def frame_refs(parts, s, v, eps, k, clamp=False, chunk=4000):
    F = np.full(len(s), np.nan); T = np.full(len(s), np.nan)
    for idx in parts:
        ss = np.concatenate([np.full(k - 1, np.nan), s[idx]])
        vv = np.concatenate([np.zeros(k - 1), v[idx]])
        S, V = swv(ss, k), swv(vv, k)
        for a in range(0, len(idx), chunk):
            Sb, Vb = S[a:a + chunk], V[a:a + chunk]
            valid = ~np.isnan(Sb)
            z = np.where(valid, (Sb - 1.0) / eps, -np.inf)
            if clamp:
                z = np.where(z < -700, -np.inf, z)
            W = np.exp(z)
            with np.errstate(invalid="ignore", divide="ignore"):
                F[idx[a:a + chunk]] = (W * Vb).sum(1) / W.sum(1)
            m = np.nanmax(Sb, axis=1, keepdims=True)
            W2 = np.where(valid, np.exp((np.where(valid, Sb, m) - m) / eps), 0.0)
            T[idx[a:a + chunk]] = (W2 * Vb).sum(1) / W2.sum(1)
    return F, T


def emulate_add_subtract(parts, s, v, eps, k):
    """plain float64 add-entering / subtract-leaving, never recomputed (what SQLite < 3.43 does)."""
    w_all = np.exp((s - 1.0) / eps)
    out = np.full(len(s), np.nan)
    for idx in parts:
        w = w_all[idx].tolist(); vv = v[idx].tolist(); num = den = 0.0
        res = [0.0] * len(w)
        for j in range(len(w)):
            num += w[j] * vv[j]; den += w[j]
            if j >= k:
                num -= w[j - k] * vv[j - k]; den -= w[j - k]
            res[j] = num / den if den != 0 else math.nan
        out[idx] = res
    return out


def build_refs(df):
    if REFS.exists():
        return dict(np.load(REFS))
    parts = partitions(df)
    v = df.rating.values.astype(np.float64)
    refs = {}
    for (q, eps, k) in cells():
        s = df[f"s{q}"].values
        key = cell_key(q, eps, k)
        refs["F_" + key], refs["T_" + key] = frame_refs(parts, s, v, eps, k)
        if eps < 0.01:
            refs["Fclamp_" + key], _ = frame_refs(parts, s, v, eps, k, clamp=True)
        refs["emu_" + key] = emulate_add_subtract(parts, s, v, eps, k)
        print("ref", key, flush=True)
    np.savez_compressed(REFS, **refs)
    return refs


def grade(x, F, T, ids):
    """x: engine output aligned by id. Host attribution is against F, the user's view against T."""
    fF, fT = np.isfinite(F), np.isfinite(T)
    fx = np.isfinite(x)
    both = fx & fF
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(x - F) / np.abs(F)
        relT = np.abs(x - T) / np.abs(T)
    rel_b = np.where(both, rel, 0.0)
    absF = np.where(both, np.abs(x - F), 0.0)
    j = int(np.argmax(rel_b)) if both.any() else None
    bT = fx & fT
    relT_b = np.where(bT, relT, 0.0)
    g = dict(
        rows=int(len(x)),
        formula_nan_rows=int((~fF).sum()),                     # F is 0/0: every weight underflowed (formula's fault)
        finite_where_formula_undefined=int((~fF & fx).sum()),  # engine invents a finite value where F is 0/0
        host_nan_rows=int((~fx & fF).sum()),                   # engine NULL/NaN/Inf where the formula is finite
        host_wrong_rows=int((rel_b > TOL).sum()),              # finite but > 1e-6 relative off the formula
        max_rel_err_vs_formula=float(rel_b.max()) if both.any() else None,
        max_abs_err_vs_formula=float(absF.max()) if both.any() else None,   # rating points (ratings are 1-10)
        out_of_range_rows=int((fx & ((x < 1.0 - 1e-9) | (x > 10.0 + 1e-9))).sum()),
        worst=dict(id=int(ids[j]), value=float(x[j]), formula_ref=float(F[j]), true_ref=float(T[j])) if j is not None and rel_b[j] > TOL else None,
        user_wrong_rows=int(((relT_b > TOL) | (~fx & fT)).sum()),    # what the user sees vs the softmax they asked for
        max_rel_err_vs_true=float(relT_b.max()) if bT.any() else None,
    )
    g["class"] = ("wrong-finite" if g["host_wrong_rows"] else "") + \
                 ("+" if g["host_wrong_rows"] and g["host_nan_rows"] else "") + \
                 ("NaN/NULL" if g["host_nan_rows"] else "")
    g["class"] = g["class"] or "correct"
    return g


def main():
    df = load_df()
    ids = np.arange(len(df))
    refs = build_refs(df)
    engines = {}
    for jf in sorted(OUT.glob("*.json")):
        meta = json.loads(jf.read_text())
        arr = dict(np.load(OUT / (jf.stem + ".npz")))
        res = {}
        for key, c in meta["cells"].items():
            base = "_".join(key.split("_")[:3])
            variant = "_".join(key.split("_")[3:])
            if c["status"] != "ok":
                res[key] = dict(cls="error", error=c["error"])
                continue
            Fk = "Fclamp_" + base if ("clamp" in variant or variant in ("unsafe_sum", "numeric")) and "Fclamp_" + base in refs else "F_" + base
            g = grade(arr[key], refs[Fk], refs["T_" + base], ids)
            res[key] = g
        meta["graded"] = res
        mins = {}
        for mname, m in MINIMAL.items():
            exact = minimal_exact(m["x"], m["frame"])
            got = meta["minimal"].get(mname)
            def judge(o):
                if isinstance(o, str):
                    return dict(outputs=o, ok=False)
                o = [None if (a is None or (isinstance(a, float) and math.isnan(a))) else float(a) for a in o]
                return dict(outputs=o, ok=all(a is not None and a == b for a, b in zip(o, exact)))
            got = got if isinstance(got, dict) else {"default": got}
            mins[mname] = dict(exact=exact, **{k2: judge(v2) for k2, v2 in got.items()})
        meta["minimal_graded"] = mins
        engines[jf.stem] = meta
    # numpy emulation of a plain inverse, for scale
    emu = {}
    for (q, eps, k) in cells():
        key = cell_key(q, eps, k)
        emu[key] = grade(refs["emu_" + key], refs["F_" + key], refs["T_" + key], ids)
    meta_data = json.loads((WORK / "meta.json").read_text())
    res = dict(
        experiment="window_inverse: floating-point sliding-window SUM maintained by subtraction",
        data=dict(corpus="IMDb (/data/softavg/corpora/imdb, read-only)", **meta_data,
                  score="float64 cosine(emb, emb[qidx]) precomputed as column s<q>; queries are table rows, so max s = 1"),
        grid=dict(eps=EPS, frames=FRAMES, queries=NQ, tol_rel=TOL),
        references="numpy float64, per frame from scratch; F = the SQL's own formula exp((s-1)/eps), T = per-frame max-shifted softmax",
        engines=engines,
        numpy_add_subtract_emulation=emu,
    )
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print_summary(res)
    write_readme_tables(res)


def summarize_engine(meta, variant=""):
    rows = {}
    for (q, eps, k) in cells():
        key = cell_key(q, eps, k, variant)
        g = meta["graded"].get(key)
        if g is None:
            continue
        rows.setdefault((eps, k), []).append(g)
    return rows


def print_summary(res):
    for name, meta in res["engines"].items():
        variants = sorted({"_".join(k.split("_")[3:]) for k in meta["graded"]})
        for var in variants:
            for (eps, k), gs in summarize_engine(meta, var).items():
                if any(g.get("cls") == "error" for g in gs):
                    print(f"{name:14s} {var:18s} eps={eps:<6g} k={k:<5d} ERROR {gs[0].get('error','')[:90]}")
                    continue
                cls = sorted({g['class'] for g in gs})
                mr = max(g["max_rel_err_vs_formula"] or 0 for g in gs)
                ma = max(g["max_abs_err_vs_formula"] or 0 for g in gs)
                nw = sum(g["host_wrong_rows"] for g in gs); nn = sum(g["host_nan_rows"] for g in gs)
                fn = sum(g["formula_nan_rows"] for g in gs); oor = sum(g["out_of_range_rows"] for g in gs)
                mt = max(g["max_rel_err_vs_true"] or 0 for g in gs)
                print(f"{name:14s} {var:18s} eps={eps:<6g} k={k:<5d} {'/'.join(cls):22s} wrong={nw:<7d} hostNaN={nn:<7d} "
                      f"formulaNaN={fn:<7d} oor={oor:<6d} maxrel={mr:.2e} maxabs={ma:.3g} maxrelT={mt:.2e}")
        print(name, "minimal:", {m: {kk: (vv["ok"] if isinstance(vv, dict) else vv) for kk, vv in d.items() if kk != "exact"}
                                 for m, d in meta["minimal_graded"].items()})


ORDER = ["sqlite_stdlib", "sqlite_apsw", "datafusion", "polars", "pandas", "postgresql", "duckdb", "clickhouse", "spark",
         "mysql", "mariadb"]
RANK = {"error": 4, "wrong-finite+NaN/NULL": 3, "wrong-finite": 2, "NaN/NULL": 1, "correct": 0}
VARLABEL = {("postgresql", ""): "opus SQL verbatim", ("postgresql", "clamp"): "+ opus's PG clamp (z < -700 -> 0)",
            ("postgresql", "unsafe_sum"): "manual's `unsafe_sum` (float8 MINVFUNC), clamped",
            ("postgresql", "numeric"): "sum(w::numeric) (exact inverse), clamped",
            ("polars", ""): "rolling_sum().over()", ("pandas", ""): "groupby().rolling().sum()", ("polars", "sql"): "opus SQL via pl.SQLContext"}


def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}e}" if (x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-2)) else f"{x:.3g}")


def write_readme_tables(res):
    L = []
    L.append("### Versions\n")
    L.append("| engine | version | how it was run |\n|---|---|---|")
    for name in ORDER:
        m = res["engines"].get(name)
        if not m:
            continue
        v = dict(m["version"])
        eng, ver = v.pop("engine"), v.pop("version")
        L.append(f"| {eng} | {ver} | " + "; ".join(f"{k}: {x}" for k, x in v.items()) + " |")
    L.append("")
    L.append("### Rolling softmax-weighted average (opus T4 SQL), 3 queries x 459,865 rows per cell\n")
    L.append("Class = worst over the 3 queries, judged against the formula reference F (host attribution); "
             "`q` = queries with that class. wrong rows = finite but > 1e-6 relative off F; host NULL/NaN = engine "
             "returns NULL/NaN/Inf where F is finite; abs err in rating points (ratings are 1-10); out of [1,10] = "
             "finite outputs outside the rating scale; F 0/0 = rows where every weight in the frame underflows "
             "(the formula's own failure, identical for every engine; not counted against the host); finite where F 0/0 = the engine returns a number although every weight in the frame is 0 (built from residue of rows that already left).\n")
    L.append("| engine | spelling | eps | frame | class (q) | wrong rows | host NULL/NaN | max rel err | max abs err | out of [1,10] | F 0/0 rows | finite where F 0/0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        m = res["engines"].get(name)
        if not m:
            continue
        variants = sorted({"_".join(k.split("_")[3:]) for k in m["graded"]}, key=lambda x: (x != "", x))
        for var in variants:
            for (eps, k), gs in summarize_engine(m, var).items():
                lab = VARLABEL.get((name, var), var or "opus SQL")
                if any(g.get("cls") == "error" for g in gs):
                    err = next(g["error"] for g in gs if g.get("cls") == "error")
                    L.append(f"| {name} | {lab} | {eps:g} | {k} | error ({sum(g.get('cls') == 'error' for g in gs)}) | "
                             f"`{err[:70]}` | | | | | | |")
                    continue
                worst = max((g["class"] for g in gs), key=RANK.get)
                nq = sum(g["class"] == worst for g in gs)
                L.append(f"| {name} | {lab} | {eps:g} | {k} | {'**' + worst + '**' if worst != 'correct' else worst} ({nq}) | "
                         f"{sum(g['host_wrong_rows'] for g in gs)} | {sum(g['host_nan_rows'] for g in gs)} | "
                         f"{fmt(max(g['max_rel_err_vs_formula'] or 0 for g in gs))} | "
                         f"{fmt(max(g['max_abs_err_vs_formula'] or 0 for g in gs))} | "
                         f"{sum(g['out_of_range_rows'] for g in gs)} | {sum(g['formula_nan_rows'] for g in gs)} | "
                         f"{sum(g['finite_where_formula_undefined'] for g in gs)} |")
    emu = res["numpy_add_subtract_emulation"]
    for eps in EPS:
        for k in FRAMES:
            gs = [emu[cell_key(q, eps, k)] for q in range(NQ)]
            worst = max((g["class"] for g in gs), key=RANK.get)
            L.append(f"| numpy emulation | plain add/subtract, never recompute | {eps:g} | {k} | {worst} ({sum(g['class'] == worst for g in gs)}) | "
                     f"{sum(g['host_wrong_rows'] for g in gs)} | {sum(g['host_nan_rows'] for g in gs)} | "
                     f"{fmt(max(g['max_rel_err_vs_formula'] or 0 for g in gs))} | {fmt(max(g['max_abs_err_vs_formula'] or 0 for g in gs))} | "
                     f"{sum(g['out_of_range_rows'] for g in gs)} | {sum(g['formula_nan_rows'] for g in gs)} | "
                     f"{sum(g['finite_where_formula_undefined'] for g in gs)} |")
    L.append("")
    L.append("Worst row per engine at eps = 0.02, frame 50 (the T4 setting):\n")
    L.append("| engine | spelling | query | id | engine value | formula ref F | true softmax ref T |\n|---|---|---|---|---|---|---|")
    for name in ORDER:
        m = res["engines"].get(name)
        if not m:
            continue
        for var in sorted({"_".join(k.split("_")[3:]) for k in m["graded"]}, key=lambda x: (x != "", x)):
            best = None
            for q in range(NQ):
                g = m["graded"].get(cell_key(q, 0.02, 50, var))
                if g and g.get("worst") and (best is None or g["max_rel_err_vs_formula"] > best[1]["max_rel_err_vs_formula"]):
                    best = (q, g)
            if best:
                w = best[1]["worst"]
                L.append(f"| {name} | {VARLABEL.get((name, var), var or 'opus SQL')} | q{best[0]} | {w['id']} | {w['value']:.6g} | {w['formula_ref']:.6g} | {w['true_ref']:.6g} |")
    L.append("")
    L.append("### Minimal non-exponential cases: `SUM(x) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`\n")
    for mname, mc in MINIMAL.items():
        L.append(f"`{mname}`: x = {mc['x']}; correctly rounded frame sums = "
                 f"{res['engines'][next(iter(res['engines']))]['minimal_graded'][mname]['exact']}\n")
    L.append("| engine | spelling | one_large | two_large |\n|---|---|---|---|")
    for name in ORDER:
        m = res["engines"].get(name)
        if not m:
            continue
        mg = m["minimal_graded"]
        for var in [k for k in mg["one_large"] if k != "exact"]:
            cells_ = []
            for mname in MINIMAL:
                d = mg[mname].get(var)
                o = d["outputs"]
                cells_.append(("ok " if d["ok"] else "**wrong** ") + (str([f"{a:g}" if a is not None else "NULL" for a in o]) if isinstance(o, list) else o))
            L.append(f"| {name} | {var} | {cells_[0]} | {cells_[1]} |")
    txt = "\n".join(L) + "\n"
    rd = HERE / "README.md"
    if rd.exists():
        r = rd.read_text()
        a, b = "<!-- BEGIN GENERATED (assemble.py) -->", "<!-- END GENERATED -->"
        if a in r and b in r:
            r = r.split(a)[0] + a + "\n\n" + txt + "\n" + b + r.split(b)[1]
            rd.write_text(r)
            return
    (WORK / "tables.md").write_text(txt)


if __name__ == "__main__":
    main()

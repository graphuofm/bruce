#!/usr/bin/env python3
"""Execution harness for the LLM-written-SQL experiment (README.md).

    python3 evaluate.py            # execute every new SQL in outputs/*.jsonl, re-score all, write results.json + README block
    python3 evaluate.py --score    # re-score from exec_cache.jsonl only (no engine is started)

What it does
  1. Reads every outputs/*.jsonl (one line per LLM sample; field names are
     normalised in load_samples, so the Claude-arm files may use their own spelling).
  2. For every distinct (engine, SQL text) that is not yet in exec_cache.jsonl,
     loads the IMDb table into that engine (stock engine, no softavg extension),
     substitutes the query vector for the parameter, runs the SQL for NQ queries,
     grades each run against numpy float64 references and appends the record
     to exec_cache.jsonl. Engines are torn down afterwards and their files deleted.
  3. Aggregates cache + annotations.json (manual semantic tags) into results.json
     and the generated block of README.md.

Stock engines: PostgreSQL 17.10 release + pgvector, DuckDB 1.5.6 (Python),
SQLite 3.37.2 (Python stdlib) + sqlite-vec 0.1.9. Needs the sigmod2027 venv
(duckdb 1.5.6, psycopg2); re-executes itself under it when started with another python.
"""
import argparse, datetime as dt, decimal, hashlib, json, math, os, re, shutil, struct, subprocess, sys, threading, time
from collections import Counter, defaultdict
from pathlib import Path

VENV_PY = "python"
try:
    import duckdb
    _ok = duckdb.__version__ == "1.5.6"
except ImportError:
    _ok = False
if not _ok and os.path.exists(VENV_PY) and not os.environ.get("LLM_SQL_REEXEC"):
    os.environ["LLM_SQL_REEXEC"] = "1"
    os.execv(VENV_PY, [VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])

import faulthandler; faulthandler.enable()
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WORK = HERE / "work"                     # bulky engine files; deleted at the end of a run
CACHE = HERE / "exec_cache.jsonl"
ANNOT = HERE / "annotations.json"
PROMPTS = HERE / "prompts_v1.json"
CORP = Path("/data/softavg/corpora/imdb")
VEC0 = HERE.parent / "sqlite" / "deps" / "sqlite_vec-0.1.9" / "vec0"
PG_BIN = Path.home() / ".pgrx/17.10/pg-release-install/bin"
PG_PORT = 28857
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

HARNESS_VERSION = 3          # bump when execution/grading changes (invalidates the cache)
NQ = 10                      # query vectors per SQL (first NQ of the PROTOCOL's 200)
TOL = 1e-6                   # relative tolerance for "correct"
LOOSE = 1e-3                 # reported only: how many (d) would pass at this tolerance
TIMEOUT_S = {"agg": 120, "window": 300}
TS_BASE = dt.datetime(2000, 1, 1)        # ts = TS_BASE + id seconds (synthetic, monotone in corpus order)
FRAME = 50                               # T4: the review and the 49 previous ones

# temperature of each task, as written in prompts_v1.json
TASKS = {
    "T1_softmax_named": dict(kind="agg", eps=1e-3),
    "T2_exp_weight": dict(kind="agg", eps=1e-3),
    "T3_plain_language": dict(kind="agg", eps=1e-4),
    "T4_rolling_window": dict(kind="window", eps=0.02),
    "T5_moderate_temp": dict(kind="agg", eps=0.1),
}
CLASSES = ["a_parse_or_bind", "timeout", "b_runtime_error", "c_nan_null_inf", "f_different_question",
           "d_finite_wrong", "e_correct"]
SILENT = {"c_nan_null_inf", "d_finite_wrong", "f_different_question"}


# =============================================================== samples
def load_samples():
    prompts = {p["id"]: p for p in json.loads(PROMPTS.read_text())["prompts"]}
    out = []
    for f in sorted(OUT.glob("*.jsonl")):
        for ln, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r.get("prompt_id") or r.get("id") or r.get("prompt")
            if pid not in prompts:
                print(f"skip {f.name}:{ln}: unknown prompt id {pid!r}", file=sys.stderr)
                continue
            raw = r.get("raw_text", r.get("raw", r.get("response", r.get("output", ""))))
            sql = r.get("extracted_sql", r.get("sql"))
            if sql is None:
                sql = extract_sql(raw or "")
            t = r.get("temperature", r.get("temp"))
            out.append(dict(model=r.get("model") or f.stem, file=f.name, prompt_id=pid,
                            task=prompts[pid]["task"], engine=prompts[pid]["engine"],
                            sample_idx=r.get("sample_idx", r.get("sample", r.get("idx", 0))),
                            temperature=t, raw_text=raw, sql=sql))
    return prompts, out


REPAIR_NAMES = r"(?<![\w.])(vec_cosine_similarity|cosine_similarity|cosine_sim|cos_sim|cosine)\s*\("


def repair_sql(sql, engine):
    """Secondary analysis only: replace calls to a non-existent cosine-similarity function
    (vec_cosine_similarity / cosine_similarity / cosine_sim / cos_sim / cosine, any case, two arguments) by the engine's real one.
    Nothing else is changed. Returns the SQL unchanged if there is no such call."""
    out, i = [], 0
    for m in re.finditer(REPAIR_NAMES, sql, flags=re.I):
        if m.start() < i:
            continue
        j, depth, args, last = m.end(), 1, [], m.end()
        while j < len(sql) and depth:
            ch = sql[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append(sql[last:j]); break
            elif ch == "," and depth == 1:
                args.append(sql[last:j]); last = j + 1
            j += 1
        if len(args) != 2:
            continue
        a, b = (x.strip() for x in args)
        rep = {"postgresql": f"(1 - (({a}) <=> ({b})))",
               "duckdb": f"array_cosine_similarity({a}, {b})",
               "sqlite": f"(1 - vec_distance_cosine({a}, {b}))"}[engine]
        out.append(sql[i:m.start()]); out.append(rep); i = j + 1
    out.append(sql[i:])
    return "".join(out)


def repaired_samples(samples):
    rep = []
    for s in samples:
        r = repair_sql(s["sql"], s["engine"])
        if r != s["sql"]:
            rep.append(dict(s, sql=r, repaired_from=sql_sha(s["sql"])))
    return rep


def extract_sql(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```sql[ \t]*\n(.*?)```", text, flags=re.S | re.I)
    if not blocks:
        blocks = re.findall(r"```[a-zA-Z]*[ \t]*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip()


def sql_key(engine, sql):
    return hashlib.sha256(f"{HARNESS_VERSION}\0{engine}\0{sql}".encode()).hexdigest()[:16]


def sql_sha(sql):
    return hashlib.sha256(sql.encode()).hexdigest()[:12]


# =============================================================== SQL text handling
def split_statements(sql):
    """Split on ';' outside quotes, dollar quotes and comments. Returns non-empty statements."""
    out, cur, i, n = [], [], 0, len(sql)
    while i < n:
        c = sql[i]
        if c in "'\"":
            j = i + 1
            while j < n:
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:
                        j += 2; continue
                    break
                j += 1
            cur.append(sql[i:j + 1]); i = j + 1; continue
        if sql.startswith("--", i):
            j = sql.find("\n", i); j = n if j < 0 else j
            cur.append(sql[i:j]); i = j; continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2); j = n if j < 0 else j + 2
            cur.append(sql[i:j]); i = j; continue
        m = re.match(r"\$[A-Za-z_]*\$", sql[i:]) if c == "$" else None
        if m:
            tag = m.group(0); j = sql.find(tag, i + len(tag)); j = n if j < 0 else j + len(tag)
            cur.append(sql[i:j]); i = j; continue
        if c == ";":
            out.append("".join(cur)); cur = []; i += 1; continue
        cur.append(c); i += 1
    out.append("".join(cur))

    def meaningful(s):
        s2 = re.sub(r"--[^\n]*", "", s); s2 = re.sub(r"/\*.*?\*/", "", s2, flags=re.S)
        return s2.strip() != ""
    return [s.strip() for s in out if meaningful(s)]


def placeholders(sql):
    s = re.sub(r"'(?:[^']|'')*'", "''", sql)
    found = Counter()
    for pat, name in [(r"\$1(?!\d)", "$1"), (r"\?1(?!\d)", "?1"), (r"\?(?![\d\w])", "?"),
                      (r"(?<![:\w]):[A-Za-z_]\w*", ":name"), (r"(?<!\w)@[A-Za-z_]\w*", "@name"),
                      (r"%s", "%s"), (r"\$[A-Za-z_]\w*", "$name"), (r"\$[2-9]", "$n>1")]:
        k = len(re.findall(pat, s))
        if k:
            found[name] = k
    return dict(found)


def substitute(sql, lit):
    """Replace $1, ?1 and bare ? (outside string literals) with the query-vector literal."""
    parts = re.split(r"('(?:[^']|'')*')", sql)
    for k in range(0, len(parts), 2):
        p = parts[k]
        p = re.sub(r"\$1(?!\d)", lambda m: lit, p)
        p = re.sub(r"\?1(?!\d)", lambda m: lit, p)
        p = re.sub(r"\?(?![\d\w])", lambda m: lit, p)
        parts[k] = p
    return "".join(parts)


def vec_literal(engine, q):
    q32 = np.asarray(q, dtype=np.float32)
    if engine == "postgresql":
        return "('[" + ",".join(repr(float(x)) for x in q32) + "]'::vector(384))"
    if engine == "duckdb":
        return "(CAST([" + ",".join(repr(float(x)) for x in q32) + "] AS FLOAT[384]))"
    return "X'" + q32.astype("<f4").tobytes().hex().upper() + "'"


# =============================================================== SQL pattern tagger (static; manual overrides in annotations.json)
def strip_sql(sql):
    s = re.sub(r"--[^\n]*", " ", sql); s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"'(?:[^']|'')*'", "''", s)
    return s


def exp_args(s):
    """Arguments of every EXP(...) call (balanced parentheses)."""
    out = []
    for m in re.finditer(r"(?<![\w.])exp\s*\(", s, flags=re.I):
        i, depth = m.end(), 1
        while i < len(s) and depth:
            depth += {"(": 1, ")": -1}.get(s[i], 0); i += 1
        out.append(s[m.end():i - 1])
    return out


def alias_defs(s):
    """{alias: expression} for every `<expr> AS alias` (expression = text back to the enclosing ',' / SELECT / '(')."""
    out = {}
    for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)", s, flags=re.I):
        i, depth = m.start() - 1, 0
        while i >= 0:
            ch = s[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and (ch == "," or re.match(r"(?i)select\b", s[i:i + 7]) and (i == 0 or not s[i - 1].isalnum())):
                break
            i -= 1
        expr = s[i + 1:m.start()].strip()
        if re.match(r"(?i)select\b", expr):
            expr = expr[6:].strip()
        if expr and not re.match(r"(?i)^(materialized|not materialized)?\s*$", expr):
            # does the SELECT block that defines this alias have a GROUP BY at its own level?
            a0, d = m.start(), 0
            while a0 > 0:
                a0 -= 1
                if s[a0] == ")":
                    d += 1
                elif s[a0] == "(":
                    if d == 0:
                        break
                    d -= 1
            b0, d = m.end(), 0
            while b0 < len(s):
                if s[b0] == "(":
                    d += 1
                elif s[b0] == ")":
                    if d == 0:
                        break
                    d -= 1
                b0 += 1
            blk, flat, d = s[a0:b0], [], 0
            for ch in blk:
                d += {"(": 1, ")": -1}.get(ch, 0)
                flat.append(ch if d <= 1 else " ")
            gb = "group by" in re.sub(r"\s+", " ", "".join(flat))
            expr = re.sub(r"(?<![\w_])max\s*\(", "maxg(" if gb else "maxn(", expr)
            out.setdefault(m.group(1).lower(), expr)
    return out


def expand(expr, defs, rounds=4):
    for _ in range(rounds):
        new = re.sub(r"\b([A-Za-z_]\w*)\b",
                     lambda m: "(" + defs[m.group(1).lower()] + ")" if m.group(1).lower() in defs and
                     not re.search(r"\b" + re.escape(m.group(1)) + r"\b", defs[m.group(1).lower()], flags=re.I) else m.group(0),
                     expr)
        if new == expr:
            break
        expr = new
    return expr


def tag_pattern(sql):
    """Static tag of the numerical pattern. Heuristic; annotations.json overrides it after manual reading."""
    s = strip_sql(sql)
    low = s.lower()
    defs = alias_defs(low)
    args = [expand(a, defs) for a in exp_args(low)]
    tags = []
    has_exp = bool(args) or bool(re.search(r"\bpow(er)?\s*\(\s*(2\.718|exp\s*\(\s*1)", low))
    if not has_exp:
        tags.append("no-exp")
    else:
        lse = re.search(r"\b(ln|log)\s*\(\s*sum\s*\(\s*exp", low)
        scopes = set()
        for a in args:
            if not re.search(r"(?<![\w_])max[gn]?\s*\(", a) or "-" not in a:
                continue
            for m in re.finditer(r"(?<![\w_])max([gn]?)\s*\(", a):
                i, depth = m.end(), 1
                while i < len(a) and depth:
                    depth += {"(": 1, ")": -1}.get(a[i], 0); i += 1
                after, before = a[i:i + 40], a[max(0, m.start() - 30):m.start()]
                if re.match(r"\s*over\s*\(\s*\)", after):
                    scopes.add("global")
                elif re.match(r"\s*over\s*\(\s*partition", after):
                    scopes.add("window")
                elif re.search(r"\(\s*select\s*$", before):
                    # scalar subquery: correlated (per group) if its WHERE mentions category / it groups
                    j, d2 = i, 0
                    while j < len(a) and d2 >= 0:
                        d2 += {"(": 1, ")": -1}.get(a[j], 0); j += 1
                    sub = a[i:j]
                    scopes.add("correlated-subquery" if re.search(r"\bwhere\b.*category", sub, flags=re.S) or "group by" in sub else "global")
                elif m.group(1) == "n":
                    scopes.add("global")
                else:
                    scopes.add("CTE/subquery")
        if lse:
            tags.append("log-sum-exp")
        for sc in sorted(scopes):
            tags.append(f"max-shift({sc})")
        if not scopes and not lse:
            const = any(re.search(r"-\s*1(\.0*)?\s*\)|^\s*-|\(\s*-\s*\(?\s*[\w.]*\s*(<=>|distance)", a) for a in args)
            tags.append("subtraction-without-max" if const else "naive")
        clamp = re.search(r"\b(greatest|least)\s*\(", low) or any(re.search(r"\bcase\b", a) for a in args) or \
            re.search(r"\bcase\b[^;]*?\bthen\b[^;]*?\bexp\s*\(", low)
        if clamp:
            tags.append("clamp")
    if re.search(r"\blimit\b", low):
        tags.append("has-LIMIT")
    return tags


# =============================================================== data and references
def load_data():
    df = pd.read_parquet(CORP / "corpus.parquet", columns=["facet", "value", "text"])
    emb = np.load(CORP / "emb.npy")
    assert emb.dtype == np.float32 and emb.shape == (len(df), 384)
    rng = np.random.RandomState(0)
    titles = df.text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)[:NQ]
    pq = HERE.parent / "postgres" / "queries.json"
    if pq.exists():
        assert [q["row"] for q in json.loads(pq.read_text())["queries"][:NQ]] == [int(i) for i in qidx]
    return dict(cat=df.facet.values.astype(object), rating=df.value.values.astype(np.float64), emb=emb,
                qidx=[int(i) for i in qidx], qtext=[str(titles.values[i]) for i in qidx],
                cats=sorted(df.facet.unique()), n=len(df))


def group_softavg(cat_codes, ncat, v, s, eps):
    """numpy float64 max-shifted two-pass softavg per group."""
    m = np.full(ncat, -np.inf); np.maximum.at(m, cat_codes, s)
    if eps == math.inf:
        w = np.ones_like(s)
    else:
        w = np.exp((s - m[cat_codes]) / eps)
    num = np.bincount(cat_codes, weights=w * v, minlength=ncat)
    den = np.bincount(cat_codes, weights=w, minlength=ncat)
    return num / den


def window_softavg(order_groups, v, s, eps, frame, cumulative=False):
    """per-row reference over `frame` rows ending at the row (rows ordered by ts inside each group).
    order_groups: list of index arrays (row ids of one partition, in ts order)."""
    out = np.empty(len(v))
    from numpy.lib.stride_tricks import sliding_window_view as swv
    for idx in order_groups:
        ss, vv = s[idx], v[idx]
        if cumulative:
            # online softmax, row by row (exact per prefix, max-anchored)
            m = np.maximum.accumulate(ss)
            res = np.empty(len(idx)); num = den = 0.0; mc = -np.inf
            for j in range(len(idx)):
                if ss[j] > mc:
                    sc = math.exp((mc - ss[j]) / eps) if mc > -np.inf else 0.0
                    num *= sc; den *= sc; mc = ss[j]
                w = math.exp((ss[j] - mc) / eps)
                num += w * vv[j]; den += w; res[j] = num / den
            out[idx] = res
            continue
        sp = np.concatenate([np.full(frame - 1, -np.inf), ss])
        vp = np.concatenate([np.zeros(frame - 1), vv])
        S, V = swv(sp, frame), swv(vp, frame)
        m = S.max(axis=1, keepdims=True)
        W = np.exp((S - m) / eps)
        out[idx] = (W * V).sum(1) / W.sum(1)
    return out


class Refs:
    """References for one engine: score variants x queries, lazily computed alternatives."""

    def __init__(self, data, native_scores):
        self.d = data
        self.cats = data["cats"]
        self.code = {c: i for i, c in enumerate(self.cats)}
        self.cat_codes = np.array([self.code[c] for c in data["cat"]])
        self.n = data["n"]
        self.v = data["rating"]
        e64 = data["emb"].astype(np.float64)
        self.norm = np.linalg.norm(e64, axis=1)
        self.e64 = e64
        self.native = native_scores            # {variant: [np.array per query]}
        self.scores = {}
        self.part_groups = [np.flatnonzero(self.cat_codes == k) for k in range(len(self.cats))]  # id order == ts order
        self._cache = {}

    def score(self, variant, qi):
        key = (variant, qi)
        if key not in self.scores:
            q = self.d["emb"][self.d["qidx"][qi]].astype(np.float64)
            if variant == "f64_cos":
                s = (self.e64 @ q) / (self.norm * np.linalg.norm(q))
            elif variant == "f64_dot":
                s = self.e64 @ q
            else:
                s = self.native[variant][qi]
            self.scores[key] = s
        return self.scores[key]

    def inverse_sum_emulation(self, eps, qi, variant=None):
        """Per-row SUM(w*v)/SUM(w) over the 50-row frame where both sums are slid by adding the
        entering row and subtracting the leaving one in float64 (no recompute), w = exp((s - 1)/eps)."""
        variant = variant or next(iter(self.native))
        key = ("emu", eps, qi, variant)
        if key not in self._cache:
            s = self.score(variant, qi); v = self.v
            w_all = np.exp((s - 1.0) / eps)
            out = np.empty(self.n)
            for idx in self.part_groups:
                w = w_all[idx].tolist(); vv = v[idx].tolist(); num = den = 0.0
                res = [0.0] * len(w)
                for j in range(len(w)):
                    num += w[j] * vv[j]; den += w[j]
                    if j >= FRAME:
                        num -= w[j - FRAME] * vv[j - FRAME]; den -= w[j - FRAME]
                    res[j] = num / den if den != 0 else math.nan
                out[idx] = res
            self._cache[key] = out
        return self._cache[key]

    def variants(self):
        return ["f64_cos", "f64_dot"] + list(self.native)

    def ref(self, kind, eps, qi, variant="f64_cos", alt=None):
        key = (kind, eps, qi, variant, alt)
        if key in self._cache:
            return self._cache[key]
        s = self.score(variant, qi); v = self.v; ncat = len(self.cats)
        if kind == "agg":
            if alt is None:
                r = group_softavg(self.cat_codes, ncat, v, s, eps)
            elif alt == "plain_avg":
                r = group_softavg(self.cat_codes, ncat, v, s, math.inf)
            elif alt == "distance_as_similarity":
                r = group_softavg(self.cat_codes, ncat, v, -s, eps)
            elif alt == "temperature_multiplied":
                r = group_softavg(self.cat_codes, ncat, v, s, 1.0 / eps)
            elif alt == "l2_distance_score":
                r = group_softavg(self.cat_codes, ncat, v, -np.sqrt(np.maximum(2 - 2 * s, 0)), eps)
            elif alt == "linear_similarity_weights":
                r = np.bincount(self.cat_codes, weights=s * v, minlength=ncat) / np.bincount(self.cat_codes, weights=s, minlength=ncat)
            elif alt == "global_softmax_no_renorm":
                w = np.exp((s - s.max()) / eps); w = w / w.sum()
                r = np.bincount(self.cat_codes, weights=w * v, minlength=ncat)
            else:
                raise KeyError(alt)
        else:
            if alt is None:
                r = window_softavg(self.part_groups, v, s, eps, FRAME)
            elif alt == "frame_51":
                r = window_softavg(self.part_groups, v, s, eps, FRAME + 1)
            elif alt == "frame_49":
                r = window_softavg(self.part_groups, v, s, eps, FRAME - 1)
            elif alt == "no_partition":
                r = window_softavg([np.arange(self.n)], v, s, eps, FRAME)
            elif alt == "cumulative":
                r = window_softavg(self.part_groups, v, s, eps, FRAME, cumulative=True)
            elif alt == "distance_as_similarity":
                r = window_softavg(self.part_groups, v, -s, eps, FRAME)
            elif alt == "plain_avg":
                r = window_softavg(self.part_groups, v, s, math.inf, FRAME) if False else \
                    window_mean(self.part_groups, v, FRAME)
            else:
                raise KeyError(alt)
        self._cache[key] = r
        return r


def window_mean(groups, v, frame):
    out = np.empty(len(v))
    for idx in groups:
        c = np.concatenate([[0.0], np.cumsum(v[idx])])
        j = np.arange(1, len(idx) + 1); lo = np.maximum(j - frame, 0)
        out[idx] = (c[j] - c[lo]) / (j - lo)
    return out


AGG_ALTS = ["plain_avg", "distance_as_similarity", "temperature_multiplied", "l2_distance_score",
            "linear_similarity_weights", "global_softmax_no_renorm"]
WIN_ALTS = ["frame_51", "frame_49", "no_partition", "cumulative", "distance_as_similarity", "plain_avg"]


# =============================================================== result grading
def to_float(x):
    if x is None:
        return None
    if isinstance(x, (bool,)):
        return float(x)
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, decimal.Decimal):
        return float(x) if x.is_finite() else (math.nan if x.is_nan() else (math.inf if x > 0 else -math.inf))
    return "nonnum"


def rel_err(x, r):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(x - r) / np.maximum(np.abs(r), 1e-300)


def ts_to_id(x):
    if isinstance(x, str):
        try:
            x = dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(x, dt.datetime):
        if x.tzinfo is not None:
            x = x.astimezone(dt.timezone.utc).replace(tzinfo=None)
        sec = (x - TS_BASE).total_seconds()
        return int(round(sec)) if abs(sec - round(sec)) < 1e-6 else None
    return None


def grade(kind, eps, qi, cols, rows, refs):
    """Grade one result set. Returns dict(cls, ...details)."""
    rec = dict(n_rows=len(rows), columns=cols)
    ncol = len(cols)
    if not rows or ncol == 0:
        rec.update(cls="f_different_question", why="empty result")
        return rec
    colvals = list(zip(*rows))
    # ---- key column(s)
    if kind == "agg":
        catset = set(refs.cats)
        keycol = None
        for j in range(ncol):
            vals = colvals[j]
            if all(isinstance(x, str) for x in vals) and sum(x in catset for x in vals) >= 0.9 * len(vals):
                keycol = j; break
        if keycol is None:
            rec.update(cls="f_different_question", why="no category column")
            return rec
        keys = [refs.code.get(x) for x in colvals[keycol]]
        idx_valid = [i for i, k in enumerate(keys) if k is not None]
        kc = Counter(k for k in keys if k is not None)
        missing = len(refs.cats) - len(kc)
        dups = sum(c - 1 for c in kc.values())
        extra = len(keys) - len(idx_valid)
        rec.update(missing_groups=missing, duplicate_rows=dups, extra_rows=extra)
        target_n = len(refs.cats)
        keyarr = np.array([keys[i] for i in idx_valid])
        skip = {keycol}
    else:
        keycol = None; how = None
        for j in range(ncol):
            nm = cols[j].lower()
            vals = colvals[j]
            if nm == "id" or (nm.endswith("id") and all(isinstance(x, int) for x in vals[:100])):
                if all(isinstance(x, int) for x in vals[:1000]):
                    keycol, how = j, "id"; break
        if keycol is None:
            for j in range(ncol):
                if all(isinstance(x, (dt.datetime, str)) for x in colvals[j][:50]) and ts_to_id(colvals[j][0]) is not None:
                    keycol, how = j, "ts"; break
        if keycol is None:
            rec.update(cls="f_different_question", why="no id/ts column to align rows")
            return rec
        if how == "id":
            keys = [int(x) if isinstance(x, int) and 0 <= x < refs.n else None for x in colvals[keycol]]
        else:
            keys = [ts_to_id(x) for x in colvals[keycol]]
            keys = [k if k is not None and 0 <= k < refs.n else None for k in keys]
        idx_valid = [i for i, k in enumerate(keys) if k is not None]
        kc = len(set(keys[i] for i in idx_valid))
        missing = refs.n - kc
        dups = len(idx_valid) - kc
        extra = len(keys) - len(idx_valid)
        rec.update(align_by=how, missing_rows=missing, duplicate_rows=dups, extra_rows=extra)
        target_n = refs.n
        keyarr = np.array([keys[i] for i in idx_valid])
        skip = {keycol}
        # columns that are a pass-through of the table (id / ts / rating / category) are not candidates
        for j in range(ncol):
            if cols[j].lower() in ("id", "ts", "category", "rating"):
                skip.add(j)
    # ---- candidate value columns
    ref0 = refs.ref(kind, eps, qi)
    target = ref0[keyarr] if len(keyarr) else np.array([])
    cands = []
    for j in range(ncol):
        if j in skip:
            continue
        xs = [to_float(colvals[j][i]) for i in idx_valid]
        if any(x == "nonnum" for x in xs):
            continue
        arr = np.array([math.nan if x is None else x for x in xs], dtype=np.float64)
        isnull = np.array([x is None for x in xs])
        isnan = np.isnan(arr) & ~isnull
        isinf = np.isinf(arr)
        fin = np.isfinite(arr)
        err = float(np.max(rel_err(arr[fin], target[fin]))) if fin.any() else math.inf
        cands.append(dict(col=j, arr=arr, n_null=int(isnull.sum()), n_nan=int(isnan.sum()), n_inf=int(isinf.sum()),
                          n_finite=int(fin.sum()), err_finite=err))
    if not cands:
        rec.update(cls="f_different_question", why="no numeric value column")
        return rec
    # value column: a fully finite column that matches the reference to 1e-2 is the answer; otherwise any
    # column holding NaN/NULL/Inf makes the run (c); otherwise the closest column is graded.
    def nbad(c):
        return c["n_null"] + c["n_nan"] + c["n_inf"]
    match = [c for c in cands if nbad(c) == 0 and c["err_finite"] <= 1e-2]
    bad = [c for c in cands if nbad(c) > 0]
    if match:
        best = min(match, key=lambda c: c["err_finite"])
    elif bad:
        best = max(bad, key=lambda c: nbad(c))
    else:
        best = min(cands, key=lambda c: c["err_finite"])
    rec.update(value_col=cols[best["col"]], n_null=best["n_null"], n_nan=best["n_nan"], n_inf=best["n_inf"],
               n_value_cols=len(cands))
    if kind == "agg":
        bad_groups = int(best["n_null"] + best["n_nan"] + best["n_inf"])
        rec["bad_groups"] = bad_groups
    shape_bad = missing > 0 or dups > 0
    if best["n_null"] + best["n_nan"] + best["n_inf"] > 0:
        rec.update(cls="c_nan_null_inf", err_finite_rows=best["err_finite"])
        if shape_bad:
            rec["also_shape"] = f"missing={missing} dups={dups}"
        return rec
    arr = best["arr"]
    # ---- errors vs every score variant
    errs = {}
    for var in refs.variants():
        r = refs.ref(kind, eps, qi, var)[keyarr]
        errs[var] = float(np.max(rel_err(arr, r)))
    rec["err_by_variant"] = errs
    bv = min(errs, key=errs.get)
    rec.update(err_best=errs[bv], best_variant=bv, err_f64_cos=errs["f64_cos"])
    if shape_bad:
        rec.update(cls="f_different_question", why=f"missing={missing} dups={dups} of {target_n}")
        return rec
    if errs[bv] <= TOL:
        rec["cls"] = "e_correct"
        return rec
    # ---- does it match a different question?
    alts = AGG_ALTS if kind == "agg" else WIN_ALTS
    alt_errs = {}
    for alt in alts:
        try:
            r = refs.ref(kind, eps, qi, "f64_cos", alt)[keyarr]
        except Exception as e:  # pragma: no cover
            continue
        alt_errs[alt] = float(np.nanmax(rel_err(arr, r)))
    rec["err_by_alt"] = alt_errs
    ba = min(alt_errs, key=alt_errs.get) if alt_errs else None
    if ba is not None and alt_errs[ba] <= 1e-4 and alt_errs[ba] < errs[bv]:
        rec.update(cls="f_different_question", why=f"matches alternative '{ba}' (rel err {alt_errs[ba]:.2e})", alt=ba)
        return rec
    if kind == "window":
        # diagnostic: emulate a host window SUM that slides by add/subtract (SQLite's sum() xInverse)
        e = refs.inverse_sum_emulation(eps, qi)[keyarr]
        rec["err_vs_add_subtract_emulation"] = float(np.nanmax(rel_err(arr, e)))
        rec["add_subtract_emulation_err_vs_ref"] = float(np.nanmax(rel_err(e, refs.ref(kind, eps, qi, next(iter(refs.native)))[keyarr])))
    rec["cls"] = "d_finite_wrong"
    return rec


# =============================================================== engines
MAX_ROWS = 2 * 459865 + 1000      # more rows than this = row explosion; fetching stops


def slim(v):
    """Drop bulky payloads (vectors as text / lists / blobs) while fetching; they are never graded."""
    if isinstance(v, (list, tuple, bytes, bytearray, memoryview)) or (isinstance(v, str) and len(v) > 200):
        return "<bulky>"
    return v


def drain(fetchmany):
    rows = []
    while True:
        b = fetchmany(20000)
        if not b:
            return rows, False
        rows.extend(tuple(slim(v) for v in r) for r in b)
        if len(rows) > MAX_ROWS:
            return rows, True


class Engine:
    name = None

    def run(self, stmts, timeout):
        """Execute setup statements then the main one inside a rolled-back transaction.
        Returns (status, payload): status in ok|a|b|timeout."""
        raise NotImplementedError


class PG(Engine):
    name = "postgresql"

    def __init__(self, data):
        import psycopg2
        self.psycopg2 = psycopg2
        self.dir = WORK / "pg"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        WORK.mkdir(exist_ok=True)
        subprocess.run([str(PG_BIN / "initdb"), "-D", str(self.dir), "-U", "jding", "--no-locale", "-E", "UTF8"],
                       check=True, capture_output=True)
        with open(self.dir / "postgresql.conf", "a") as f:
            f.write("\n".join([f"port = {PG_PORT}", "listen_addresses = '127.0.0.1'",
                               f"unix_socket_directories = '{self.dir}'", "shared_buffers = 2GB",
                               "work_mem = 256MB", "maintenance_work_mem = 2GB", "max_parallel_workers_per_gather = 2",
                               "timezone = 'UTC'", "max_wal_size = 8GB", "fsync = off", "synchronous_commit = off"]) + "\n")
        subprocess.run([str(PG_BIN / "pg_ctl"), "-D", str(self.dir), "-l", str(WORK / "pg.log"), "-w", "start"],
                       check=True, capture_output=True)
        self.conn = psycopg2.connect(host="127.0.0.1", port=PG_PORT, dbname="postgres", user="jding")
        self.conn.autocommit = True
        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION vector")
        cur.execute("CREATE TABLE reviews(id bigint PRIMARY KEY, category text, rating double precision, "
                    "ts timestamptz, emb vector(384))")
        self._load(cur, data)
        cur.execute("ANALYZE reviews")
        cur.execute("SELECT version(), (SELECT extversion FROM pg_extension WHERE extname='vector')")
        v, pv = cur.fetchone()
        conf = subprocess.run([str(PG_BIN / "pg_config"), "--configure"], capture_output=True, text=True).stdout
        self.version = dict(server=v, pgvector=pv, build=("debug (cassert)" if "enable-cassert" in conf else "release"),
                            binary=str(PG_BIN / "postgres"), extensions_loaded=["vector"])
        self.conn.autocommit = False

    def _load(self, cur, data):
        import io
        n, CH = data["n"], 20000
        emb_be = np.ascontiguousarray(data["emb"].astype(">f4"))
        vhdr = struct.pack(">hh", 384, 0)
        pg_epoch = dt.datetime(2000, 1, 1)
        base_us = int((TS_BASE - pg_epoch).total_seconds() * 1e6)
        for a in range(0, n, CH):
            buf = io.BytesIO()
            buf.write(b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0))
            for i in range(a, min(n, a + CH)):
                c = data["cat"][i].encode()
                buf.write(struct.pack(">hiq", 5, 8, i))
                buf.write(struct.pack(">i", len(c))); buf.write(c)
                buf.write(struct.pack(">id", 8, float(data["rating"][i])))
                buf.write(struct.pack(">iq", 8, base_us + i * 1_000_000))
                buf.write(struct.pack(">i", 4 + 4 * 384)); buf.write(vhdr); buf.write(emb_be[i].tobytes())
            buf.write(struct.pack(">h", -1)); buf.seek(0)
            cur.copy_expert("COPY reviews FROM STDIN WITH (FORMAT binary)", buf)

    def native_scores(self, data):
        out = {"pgvector_1-cosdist": [], "pgvector_-negip": []}
        cur = self.conn.cursor()
        for qi in range(NQ):
            lit = vec_literal("postgresql", data["emb"][data["qidx"][qi]])
            cur.execute(f"SELECT 1 - (emb <=> {lit}), -(emb <#> {lit}) FROM reviews ORDER BY id")
            a = np.array(cur.fetchall(), dtype=np.float64)
            out["pgvector_1-cosdist"].append(a[:, 0]); out["pgvector_-negip"].append(a[:, 1])
        self.conn.rollback()
        return out

    def run(self, stmts, timeout):
        cur = self.conn.cursor()
        try:
            cur.execute(f"SET LOCAL statement_timeout = '{int(timeout * 1000)}ms'")
            for st in stmts[:-1]:
                cur.execute(st)
            main = stmts[-1]
            if re.match(r"(?is)^\s*(\(\s*)*(select|with)\b", main):
                cur = self.conn.cursor(name="llm_q")          # server-side cursor: bounded client memory
                cur.itersize = 20000
                cur.execute(main)
                rows, exploded = drain(cur.fetchmany)
                cols = [d[0] for d in cur.description] if cur.description else []
            else:
                cur.execute(main)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows, exploded = drain(cur.fetchmany) if cur.description else ([], False)
            return "ok", (cols, rows, exploded)
        except self.psycopg2.Error as e:
            code = e.pgcode or ""
            msg = (e.pgerror or str(e)).strip()
            if code == "57014":
                return "timeout", dict(sqlstate=code, message=msg)
            st = "a" if code[:2] in ("42", "0A") or not code else "b"
            return st, dict(sqlstate=code, message=msg)
        finally:
            self.conn.rollback()

    def close(self):
        try:
            self.conn.close()
        finally:
            subprocess.run([str(PG_BIN / "pg_ctl"), "-D", str(self.dir), "-w", "stop", "-m", "fast"], capture_output=True)
            shutil.rmtree(self.dir, ignore_errors=True)


class Duck(Engine):
    name = "duckdb"

    def __init__(self, data):
        import pyarrow as pa
        WORK.mkdir(exist_ok=True)
        self.con = duckdb.connect(":memory:", config=dict(memory_limit="12GB", threads=8,
                                                          temp_directory=str(WORK / "duck_tmp")))
        flat = pa.array(data["emb"].reshape(-1), type=pa.float32())
        embarr = pa.FixedSizeListArray.from_arrays(flat, 384)
        tbl = pa.table(dict(id=pa.array(np.arange(data["n"], dtype=np.int64)),
                            category=pa.array(list(data["cat"]), type=pa.string()),
                            rating=pa.array(data["rating"]), emb=embarr))
        self.con.register("src", tbl)
        self.con.execute("CREATE TABLE reviews(id BIGINT PRIMARY KEY, category VARCHAR, rating DOUBLE, ts TIMESTAMP, emb FLOAT[384])")
        self.con.execute(f"INSERT INTO reviews SELECT id, category, rating, TIMESTAMP '{TS_BASE:%Y-%m-%d %H:%M:%S}' + to_seconds(id), "
                         "emb::FLOAT[384] FROM src")
        self.con.unregister("src")
        assert duckdb.__version__ == "1.5.6", duckdb.__version__
        exts = self.con.execute("SELECT extension_name FROM duckdb_extensions() WHERE loaded").fetchall()
        self.version = dict(duckdb=duckdb.__version__, extensions_loaded=[e[0] for e in exts], threads=8)

    def native_scores(self, data):
        out = {"duckdb_array_cosine_similarity": [], "duckdb_array_inner_product": []}
        for qi in range(NQ):
            lit = vec_literal("duckdb", data["emb"][data["qidx"][qi]])
            a = np.array(self.con.execute(f"SELECT array_cosine_similarity(emb, {lit})::DOUBLE, "
                                          f"array_inner_product(emb, {lit})::DOUBLE FROM reviews ORDER BY id").fetchall())
            out["duckdb_array_cosine_similarity"].append(a[:, 0]); out["duckdb_array_inner_product"].append(a[:, 1])
        return out

    def run(self, stmts, timeout):
        flag = dict(fired=False)
        def fire():
            flag["fired"] = True; self.con.interrupt()
        timer = threading.Timer(timeout, fire)
        self.con.execute("BEGIN TRANSACTION")
        timer.start()
        try:
            res = None
            for st in stmts:
                res = self.con.execute(st)
            cols = [d[0] for d in res.description] if res.description else []
            rows, exploded = drain(res.fetchmany) if res.description else ([], False)
            return "ok", (cols, rows, exploded)
        except duckdb.Error as e:
            et = type(e).__name__
            msg = str(e).strip()
            if flag["fired"] or et == "InterruptException":
                return "timeout", dict(error_type=et, message=msg)
            st = "a" if et in ("ParserException", "BinderException", "CatalogException", "NotImplementedException",
                               "SyntaxException", "InvalidTypeException") else "b"
            return st, dict(error_type=et, message=msg)
        finally:
            timer.cancel()
            try:
                self.con.execute("ROLLBACK")
            except duckdb.Error:
                pass

    def close(self):
        self.con.close()
        shutil.rmtree(WORK / "duck_tmp", ignore_errors=True)


class Lite(Engine):
    name = "sqlite"

    def __init__(self, data):
        import sqlite3
        self.sqlite3 = sqlite3
        WORK.mkdir(exist_ok=True)
        self.path = WORK / "llm_sql.db"
        if self.path.exists():
            self.path.unlink()
        c = sqlite3.connect(str(self.path), isolation_level=None)
        c.enable_load_extension(True); c.load_extension(str(VEC0)); c.enable_load_extension(False)
        c.execute("PRAGMA journal_mode = OFF"); c.execute("PRAGMA synchronous = OFF")
        c.execute("CREATE TABLE reviews(id INTEGER PRIMARY KEY, category TEXT, rating REAL, ts TEXT, emb BLOB)")
        c.execute("BEGIN")
        c.executemany("INSERT INTO reviews VALUES (?,?,?,?,?)",
                      ((i, data["cat"][i], float(data["rating"][i]),
                        (TS_BASE + dt.timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S"),
                        data["emb"][i].astype("<f4").tobytes()) for i in range(data["n"])))
        c.execute("COMMIT")
        c.execute("PRAGMA cache_size = -4000000"); c.execute("PRAGMA mmap_size = 4000000000")
        c.execute("PRAGMA temp_store = MEMORY")
        self.c = c
        self.version = dict(sqlite=sqlite3.sqlite_version, sqlite_vec=c.execute("SELECT vec_version()").fetchone()[0],
                            math_functions=c.execute("SELECT exp(1)").fetchone()[0] is not None,
                            extensions_loaded=["sqlite-vec (vec0.so)"])

    def native_scores(self, data):
        out = {"sqlitevec_1-cosdist": []}
        for qi in range(NQ):
            lit = vec_literal("sqlite", data["emb"][data["qidx"][qi]])
            a = np.array(self.c.execute(f"SELECT 1 - vec_distance_cosine(emb, {lit}) FROM reviews ORDER BY id").fetchall(),
                         dtype=np.float64)
            out["sqlitevec_1-cosdist"].append(a[:, 0])
        return out

    def run(self, stmts, timeout):
        sq = self.sqlite3
        t0 = time.time()
        flag = dict(fired=False)
        def handler():
            if time.time() - t0 > timeout:
                flag["fired"] = True; return 1
            return 0
        self.c.set_progress_handler(handler, 100000)
        self.c.execute("BEGIN")
        try:
            for k, st in enumerate(stmts):
                is_main = k == len(stmts) - 1
                self.c.text_factory = bytes                     # EXPLAIN's p4 column holds the raw blob literal
                try:
                    self.c.execute("EXPLAIN " + st).fetchall()     # compile only: bind/parse errors are class (a)
                except sq.Error as e:
                    return "a", dict(error_type=type(e).__name__, message=str(e), stage="prepare")
                finally:
                    self.c.text_factory = str
                cur = self.c.execute(st)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows, exploded = drain(cur.fetchmany) if cur.description else ([], False)
            return "ok", (cols, rows, exploded)
        except sq.Error as e:
            if flag["fired"]:
                return "timeout", dict(error_type=type(e).__name__, message=str(e))
            return "b", dict(error_type=type(e).__name__, message=str(e), stage="execute")
        finally:
            self.c.set_progress_handler(None, 0)
            try:
                self.c.execute("ROLLBACK")
            except sq.Error:
                pass

    def close(self):
        self.c.close()
        self.path.unlink(missing_ok=True)


ENGINES = {"postgresql": PG, "duckdb": Duck, "sqlite": Lite}


# =============================================================== execution
def execute_one(eng, refs, data, sql, task):
    kind, eps = TASKS[task]["kind"], TASKS[task]["eps"]
    stmts_raw = split_statements(sql)
    rec = dict(n_statements=len(stmts_raw), placeholders=placeholders(sql), runs=[])
    if not stmts_raw:
        rec["runs"].append(dict(q=0, cls="a_parse_or_bind", why="no SQL in reply"))
        return rec
    for qi in range(NQ):
        lit = vec_literal(eng.name, data["emb"][data["qidx"][qi]])
        stmts = [substitute(s, lit) for s in stmts_raw]
        t0 = time.time()
        st, payload = eng.run(stmts, TIMEOUT_S[kind])
        run = dict(q=qi, wall_s=round(time.time() - t0, 3))
        if st == "a":
            run.update(cls="a_parse_or_bind", error=payload)
            rec["runs"].append(run); break           # query-independent
        if st == "timeout":
            run.update(cls="timeout", error=payload)
            rec["runs"].append(run); break
        if st == "b":
            run.update(cls="b_runtime_error", error=payload)
            rec["runs"].append(run); continue
        cols, rows, exploded = payload
        if exploded:
            run.update(cls="f_different_question", why=f"row explosion: more than {MAX_ROWS} rows (fetch stopped)",
                       columns=cols)
            rec["runs"].append(run); break
        g = grade(kind, eps, qi, cols, rows, refs)
        run.update(g)
        rec["runs"].append(run)
    return rec


def execute_pending(samples, cache, data, only_engines=None):
    pending = defaultdict(dict)
    for s in samples:
        k = sql_key(s["engine"], s["sql"])
        if k not in cache:
            pending[s["engine"]][k] = s
    envs = {}
    for engine, todo in pending.items():
        if only_engines and engine not in only_engines:
            continue
        print(f"[{engine}] {len(todo)} SQL texts to execute; loading table ...", flush=True)
        t0 = time.time()
        eng = ENGINES[engine](data)
        print(f"[{engine}] loaded in {time.time() - t0:.0f}s; native scores ...", flush=True)
        try:
            refs = Refs(data, eng.native_scores(data))
            envs[engine] = dict(eng.version)
            # how far apart the references for different (valid) score definitions are: the inherent
            # sensitivity to float32 score rounding, per task temperature
            nf = {}
            for t, spec in TASKS.items():
                worst = {}
                for var in refs.variants():
                    if var == "f64_cos":
                        continue
                    e = 0.0
                    for qi in range(NQ):
                        a_, b_ = refs.ref(spec["kind"], spec["eps"], qi, "f64_cos"), refs.ref(spec["kind"], spec["eps"], qi, var)
                        e = max(e, float(np.max(rel_err(b_, a_))))
                    worst[var] = e
                nf[t] = worst
            envs[engine]["ref_spread_vs_f64_cos"] = nf
            for i, (k, s) in enumerate(todo.items()):
                t1 = time.time()
                rec = execute_one(eng, refs, data, s["sql"], s["task"])
                rec.update(key=k, engine=engine, task=s["task"], sql_sha=sql_sha(s["sql"]), sql=s["sql"],
                           harness_version=HARNESS_VERSION, engine_version=eng.version,
                           executed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                cache[k] = rec
                with CACHE.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                cl = Counter(r["cls"] for r in rec["runs"])
                print(f"[{engine}] {i + 1}/{len(todo)} {s['prompt_id']} {k} {dict(cl)} {time.time() - t1:.1f}s", flush=True)
        finally:
            eng.close()
    return envs


def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("harness_version") == HARNESS_VERSION:
                    cache[r["key"]] = r
    return cache


# =============================================================== scoring
PRIORITY = {c: i for i, c in enumerate(CLASSES)}


def sql_class(rec, annot):
    """SQL-level class = the worst class over its query runs (priority a > timeout > b > c > f > d > e),
    with manual semantic tags from annotations.json able to turn d (finite, wrong) into f."""
    runs = rec["runs"]
    cl = Counter(r["cls"] for r in runs)
    worst = min(cl, key=lambda c: PRIORITY[c])
    a = annot.get(rec["sql_sha"], {})
    if a.get("different_question") and worst in ("d_finite_wrong",):
        worst = "f_different_question"
    return worst, cl


def summarize(samples, cache, annot, envs):
    rows = []
    for s in samples:
        k = sql_key(s["engine"], s["sql"])
        rec = cache.get(k)
        if rec is None:
            continue
        cls, cl = sql_class(rec, annot)
        a = annot.get(rec["sql_sha"], {})
        pattern = a.get("pattern") or tag_pattern(s["sql"])
        errs = [r.get("err_best") for r in rec["runs"] if r.get("err_best") is not None]
        spread = max((envs.get(s["engine"], {}).get("ref_spread_vs_f64_cos", {}).get(s["task"], {}) or {0: 0}).values())
        rows.append(dict(model=s["model"], prompt_id=s["prompt_id"], task=s["task"], engine=s["engine"],
                         sample_idx=s["sample_idx"], temperature=s["temperature"], sql_sha=rec["sql_sha"],
                         cls=cls, run_classes=dict(cl), pattern=pattern,
                         different_question=a.get("different_question"), note=a.get("note"),
                         max_err_best=max(errs) if errs else None, ref_spread=spread,
                         d_within_ref_spread=(cls == "d_finite_wrong" and bool(errs) and max(errs) <= spread),
                         placeholders=rec["placeholders"], n_statements=rec["n_statements"]))
    return rows


def table(rows, keys):
    t = defaultdict(Counter)
    for r in rows:
        t[tuple(r[k] for k in keys)][r["cls"]] += 1
    return t


def write_outputs(prompts, samples, cache, annot, envs, rep=()):
    rows = summarize(samples, cache, annot, envs)
    rrows = summarize(list(rep), cache, annot, envs)
    for r, s in zip(rrows, [x for x in rep if sql_key(x["engine"], x["sql"]) in cache]):
        r["repaired_from"] = s["repaired_from"]
    models = sorted({r["model"] for r in rows})
    by_mte = {}
    for (m, e, t), c in sorted(table(rows, ["model", "engine", "task"]).items()):
        by_mte.setdefault(m, {}).setdefault(e, {})[t] = dict(c)
    def agg(keyf):
        out = defaultdict(Counter)
        for r in rows:
            out[keyf(r)][r["cls"]] += 1
        return {str(k): dict(v) for k, v in sorted(out.items())}
    def silent(c):
        return sum(v for k, v in c.items() if k in SILENT)
    totals = Counter(r["cls"] for r in rows)
    d_spread = Counter(r["model"] for r in rows if r["d_within_ref_spread"])
    d_spread_total = sum(d_spread.values())
    pat = defaultdict(Counter)
    for r in rows:
        pat[r["model"]][" + ".join(r["pattern"])] += 1
    pat_cls = defaultdict(Counter)
    for r in rows:
        pat_cls[" + ".join(r["pattern"])][r["cls"]] += 1
    # (d) split by error size
    d_err = [r["max_err_best"] for r in rows if r["cls"] == "d_finite_wrong" and r["max_err_best"] is not None]
    # representative failures (verbatim)
    reps = []
    seen = set()
    for r in sorted(rows, key=lambda r: (PRIORITY[r["cls"]], r["model"], r["prompt_id"], r["sample_idx"])):
        if r["cls"] == "e_correct":
            continue
        key = (r["cls"], r["engine"]) if r["cls"] == "a_parse_or_bind" else (r["cls"], r["engine"], r["task"])
        if key in seen:
            continue
        seen.add(key)
        rec = next(v for v in cache.values() if v["sql_sha"] == r["sql_sha"] and v["engine"] == r["engine"])
        first_bad = next((x for x in rec["runs"] if x["cls"] == r["cls"]), rec["runs"][0])
        reps.append(dict(cls=r["cls"], model=r["model"], prompt_id=r["prompt_id"], sample_idx=r["sample_idx"],
                         sql=rec["sql"], run=first_bad))
    # causes of (a)
    a_causes = defaultdict(Counter)
    for r in rows:
        if r["cls"] == "a_parse_or_bind":
            rec = cache[sql_key(r["engine"], next(x["sql"] for x in samples if x["model"] == r["model"] and
                                                  x["prompt_id"] == r["prompt_id"] and x["sample_idx"] == r["sample_idx"]))]
            run = rec["runs"][0]
            a_causes[r["model"]][a_cause((run.get("error") or {}).get("message") or run.get("why"))] += 1
    # per-run (query-level) view
    qlevel = defaultdict(Counter)
    for r in rows:
        for c, n in r["run_classes"].items():
            if c == "d_finite_wrong" and r["different_question"]:
                c = "f_different_question"          # manual tag applies to every run of that SQL
            qlevel[(r["model"], r["task"])][c] += n
    # headline per arm
    def arm(r):
        return "claude (agent harness, 1 sample/prompt)" if r["model"].startswith("claude") else "local ollama (6 samples/prompt)"
    headline = {}
    for name in sorted({arm(r) for r in rows}):
        rs = [r for r in rows if arm(r) == name]
        c = Counter(r["cls"] for r in rs)
        ran = [r for r in rs if r["cls"] not in ("a_parse_or_bind", "timeout")]
        finished = [r for r in rs if r["cls"] not in ("a_parse_or_bind", "timeout", "b_runtime_error")]
        sem = [r for r in ran if r["different_question"] or r["cls"] == "f_different_question"]
        dsp = sum(r["d_within_ref_spread"] for r in rs)
        headline[name] = dict(n=len(rs), classes=dict(c), correct=c.get("e_correct", 0),
                              rejected_before_running=c.get("a_parse_or_bind", 0), timeout=c.get("timeout", 0),
                              runtime_error=c.get("b_runtime_error", 0), silent_wrong=silent(c),
                              silent_wrong_beyond_rounding=silent(c) - dsp, ran=len(ran),
                              returned_a_result=len(finished),
                              silent_share_of_returned=(round(silent(c) / len(finished), 3) if finished else None),
                              ran_and_answers_another_question=len(sem))
    t4l = [dict(model=r["model"] + (" (repaired)" if r.get("repaired_from") else ""), sample_idx=r["sample_idx"],
                cls=r["cls"], pattern=r["pattern"], max_err=r["max_err_best"])
           for r in rows + rrows if r["task"] == "T4_rolling_window" and r["engine"] == "sqlite"
           and r["cls"] not in ("a_parse_or_bind", "timeout")]
    res = dict(
        headline=headline,
        t4_sqlite_runs=dict(note="every T4 SQL that ran on SQLite (main arm and repaired arm): max rel err over 10 queries x 459,865 rows",
                            runs=t4l),
        experiment="llm_sql: what SQL do LLMs write for a softmax-weighted per-group average, and how often does it fail silently",
        protocol=dict(prompts="prompts_v1.json (15 prompts = 5 tasks x 3 engines, fixed before collection)",
                      corpus="IMDb (/data/softavg/corpora/imdb), 459,865 rows; facet->category, value->rating, "
                             "row order->ts (TS_BASE + id seconds, monotone), emb->emb (float32, 384-d, L2-normalised)",
                      queries=f"PROTOCOL sampler (RandomState(0), first sentence 4-20 words, deduped, 200 drawn), first {NQ}",
                      query_rows=load_data_meta.get("qidx"), predicate="none (prompts have none)",
                      reference="numpy float64 max-shifted two-pass softavg per group (T4: per frame of 50 rows, partition by "
                                "category, ordered by ts); computed for every score variant (f64 cosine, f64 dot, and the engine's "
                                "own float32 similarity functions); a run is correct if max rel err over groups/rows <= "
                                f"{TOL} against any variant",
                      tolerance=TOL, timeout_s=TIMEOUT_S, harness_version=HARNESS_VERSION,
                      classes={"a_parse_or_bind": "does not parse / unknown column / unknown function / bad types (engine rejects before running)",
                               "timeout": "no answer within the timeout (not graded)",
                               "b_runtime_error": "error while running (e.g. exp overflow/underflow)",
                               "c_nan_null_inf": "runs, but some group/row value is NaN, NULL or Inf",
                               "f_different_question": "runs, finite, but answers another question (missing groups / LIMIT / different weighting; auto-matched or tagged in annotations.json)",
                               "d_finite_wrong": "runs, finite, all groups, but rel err > tol vs every reference",
                               "e_correct": "all runs within tol"},
                      sql_level_class="worst over the query runs: a > timeout > b > c > f > d > e",
                      silent_wrong="c + d + f",
                      parameter_substitution="$1, ?1 and bare ? replaced by a typed literal of the query vector "
                                             "(PG '[..]'::vector(384); DuckDB CAST([..] AS FLOAT[384]); SQLite X'..' float32 blob); "
                                             "other placeholder spellings are left alone",
                      transactions="every run inside a transaction that is rolled back; multi-statement replies run all statements, the last one is graded",
                      stock_engines="no softavg extension loaded in any engine",
                      build_type="release (PostgreSQL 17.10 pg-release-install); DuckDB/SQLite are the Python wheels/stdlib",
                      timing_status="timing irrelevant (not measured; wall_s in exec_cache.jsonl is for the harness only)"),
        engines=envs or {},
        machine=machine_state(),
        models=models,
        n_samples=len(rows),
        n_samples_unexecuted=len(samples) - len(rows),
        totals=dict(totals), silent_wrong_total=silent(totals),
        d_within_ref_spread_by_model=dict(d_spread), d_within_ref_spread_total=d_spread_total,
        silent_wrong_beyond_rounding_total=silent(totals) - d_spread_total,
        note_d_within_ref_spread="a (d) sample whose max error is below the spread between references computed from "
                                 "different valid score definitions (f64 cosine vs the engine's float32 similarity "
                                 "functions) for that engine and task: float32-rounding level, not a formula or "
                                 "algorithm failure",
        by_model=agg(lambda r: r["model"]),
        by_model_temperature=agg(lambda r: r["model"] + " / " + ("temperature 0" if r["temperature"] in (0, 0.0) else ("default sampling" if r["temperature"] == "default" else "not controlled (agent harness)"))),
        by_engine=agg(lambda r: r["engine"]),
        by_task=agg(lambda r: r["task"]),
        by_model_engine_task=by_mte,
        by_model_task_runs=agg_runs(qlevel),
        pattern_by_model={m: dict(c) for m, c in pat.items()},
        a_causes_by_model={m: dict(c) for m, c in a_causes.items()},
        class_by_pattern={p: dict(c) for p, c in pat_cls.items()},
        d_finite_wrong_errors=dict(n=len(d_err), le_1e3=sum(e <= LOOSE for e in d_err), sorted=sorted(d_err)),
        representative_failures=reps,
        samples=rows,
        repaired=dict(
            note="SECONDARY analysis, not part of the main counts: every sample whose SQL calls a non-existent "
                 "cosine-similarity function (vec_cosine_similarity / cosine_similarity / cosine_sim / cos_sim / cosine) is re-run with only "
                 "that call replaced by the engine's real one (PG 1-(a<=>b), DuckDB array_cosine_similarity, "
                 "SQLite 1-vec_distance_cosine). Shows what the rest of the SQL would have done.",
            n=len(rrows),
            by_model={m: dict(Counter(r["cls"] for r in rrows if r["model"] == m)) for m in sorted({r["model"] for r in rrows})},
            by_task={t: dict(Counter(r["cls"] for r in rrows if r["task"] == t)) for t in TASKS},
            by_model_engine_task={m: {e: {t: dict(Counter(r["cls"] for r in rrows if (r["model"], r["engine"], r["task"]) == (m, e, t)))
                                          for t in TASKS} for e in ("postgresql", "duckdb", "sqlite")}
                                  for m in sorted({r["model"] for r in rrows})},
            pattern={p: dict(c) for p, c in _pat_cls(rrows).items()},
            samples=rrows),
    )
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str))
    write_readme(res)
    return res


load_data_meta = {}


def _pat_cls(rows):
    out = defaultdict(Counter)
    for r in rows:
        out[" + ".join(r["pattern"])][r["cls"]] += 1
    return out


def a_cause(msg):
    m = (msg or "").lower()
    if re.search(r"(no such function|function .* does not exist|scalar function with name|table function with name)", m):
        return "unknown function (mostly a guessed similarity function)"
    if re.search(r"(operator does not exist|no function matches the given name and argument types|cannot be matched|could not convert|cannot cast|conversion)", m):
        return "operator / type mismatch"
    if re.search(r"(no such column|column .* does not exist|referenced column|no such table|relation .* does not exist|not found in from clause|missing from-clause|ambiguous)", m):
        return "unknown / ambiguous column or table (incl. parameter used as a column)"
    if re.search(r"(syntax error|parser error)", m):
        return "syntax error"
    if re.search(r"(aggregate|group by|window function|nested)", m):
        return "aggregate / GROUP BY misuse"
    return "other"


def agg_runs(q):
    return {f"{m} / {t}": dict(c) for (m, t), c in sorted(q.items())}


def machine_state():
    try:
        import quiet
        st = quiet.require_quiet(wait_seconds=0, verbose=False); st["quiet"] = True
    except SystemExit as e:
        st = dict(quiet=False, refusal=str(e), load_average_1min=round(os.getloadavg()[0], 2))
    except Exception as e:  # pragma: no cover
        st = dict(quiet=None, error=str(e))
    st["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return st


SHORT = {"a_parse_or_bind": "a", "timeout": "t", "b_runtime_error": "b", "c_nan_null_inf": "c",
         "f_different_question": "f", "d_finite_wrong": "d", "e_correct": "e"}


def fmt_counts(c):
    return " ".join(f"{SHORT[k]}{c[k]}" for k in CLASSES if c.get(k))


def write_readme(res):
    L = ["<!-- BEGIN GENERATED (evaluate.py) -->", ""]
    L.append(f"Samples graded: {res['n_samples']} (not yet executed: {res['n_samples_unexecuted']}). "
             f"Class letters: a = parse/bind error, t = timeout, b = runtime error, c = NaN/NULL/Inf, "
             f"f = different question, d = finite but wrong (> {TOL:g} rel), e = correct. "
             f"Silent wrong = c + d + f.")
    L.append("")
    L.append("### Headline per arm")
    L.append("")
    L.append("| arm | n | correct | rejected before running (a) | timeout | runtime error (b) | returned a result | of which silent wrong (c+d+f) | silent beyond float32 rounding | ran and answers another question |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, h in res["headline"].items():
        L.append(f"| {name} | {h['n']} | {h['correct']} | {h['rejected_before_running']} | {h['timeout']} | {h['runtime_error']} | "
                 f"{h['returned_a_result']} | {h['silent_wrong']} | {h['silent_wrong_beyond_rounding']} | {h['ran_and_answers_another_question']} |")
    L.append("")
    if res["t4_sqlite_runs"]["runs"]:
        L.append("T4 (rolling window, T = 0.02) SQL that ran on SQLite, main and repaired arm: " + "; ".join(
            f"{x['model']} s{x['sample_idx']} {SHORT[x['cls']]} [{' + '.join(x['pattern'])}] max rel err "
            + (f"{x['max_err']:.3g}" if x['max_err'] is not None else "-") for x in res["t4_sqlite_runs"]["runs"]))
        L.append("")
    L.append("### Totals per model (SQL-level class, worst over the query runs)")
    L.append("")
    L.append("| model | n | a | t | b | c | f | d | (d within rounding spread) | e | silent (c+d+f) | silent beyond rounding |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, c in list(res["by_model"].items()) + [("**all**", res["totals"])]:
        n = sum(c.values())
        ds = res["d_within_ref_spread_total"] if m == "**all**" else res["d_within_ref_spread_by_model"].get(m, 0)
        sil = sum(c.get(k, 0) for k in SILENT)
        L.append(f"| {m} | {n} | " + " | ".join(str(c.get(k, 0)) for k in CLASSES[:6]) +
                 f" | ({ds}) | {c.get('e_correct', 0)} | {sil} | {sil - ds} |")
    L.append("")
    L.append("### By model and temperature")
    L.append("")
    L.append("| model, temperature | classes |")
    L.append("|---|---|")
    for k, c in res["by_model_temperature"].items():
        L.append(f"| {k} | {fmt_counts(c)} |")
    L.append("")
    L.append("### By task (all models) and by engine (all models)")
    L.append("")
    L.append("| task | eps | classes | silent |")
    L.append("|---|---|---|---|")
    for t, c in res["by_task"].items():
        L.append(f"| {t} | {TASKS[t]['eps']:g} | {fmt_counts(c)} | {sum(c.get(k, 0) for k in SILENT)} |")
    L.append("")
    L.append("| engine | classes | silent |")
    L.append("|---|---|---|")
    for e, c in res["by_engine"].items():
        L.append(f"| {e} | {fmt_counts(c)} | {sum(c.get(k, 0) for k in SILENT)} |")
    L.append("")
    L.append("### model x engine x task (class counts per cell)")
    L.append("")
    tasks = list(TASKS)
    L.append("| model | engine | " + " | ".join(t.split("_")[0] + f" (eps {TASKS[t]['eps']:g})" for t in tasks) + " |")
    L.append("|---|---|" + "---|" * len(tasks))
    for m, de in res["by_model_engine_task"].items():
        for e, dt_ in de.items():
            L.append(f"| {m} | {e} | " + " | ".join(fmt_counts(dt_.get(t, {})) or "-" for t in tasks) + " |")
    L.append("")
    L.append("### Query-level view (each SQL x query vector run; a SQL that fails to bind counts once)")
    L.append("")
    L.append("| model / task | runs by class |")
    L.append("|---|---|")
    for k, c in res["by_model_task_runs"].items():
        L.append(f"| {k} | {fmt_counts(c)} |")
    L.append("")
    L.append("### Why (a): engine's rejection message, bucketed")
    L.append("")
    L.append("| model | causes |")
    L.append("|---|---|")
    for m, c in res["a_causes_by_model"].items():
        L.append(f"| {m} | " + "; ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) + " |")
    L.append("")
    L.append("### SQL pattern used (static tagger + manual overrides)")
    L.append("")
    allpat = Counter()
    for m, c in res["pattern_by_model"].items():
        allpat.update(c)
    L.append("| pattern | " + " | ".join(res["models"]) + " | classes (all models) |")
    L.append("|---|" + "---|" * len(res["models"]) + "---|")
    for p, n in allpat.most_common():
        L.append(f"| {p} | " + " | ".join(str(res["pattern_by_model"].get(m, {}).get(p, 0)) for m in res["models"]) +
                 f" | {fmt_counts(res['class_by_pattern'][p])} |")
    L.append("")
    de = res["d_finite_wrong_errors"]
    if de["n"]:
        L.append(f"(d) finite-but-wrong: {de['n']} samples; {de['le_1e3']} of them have max rel err <= {LOOSE:g} "
                 f"(float32-rounding scale); max rel errs: " + ", ".join(f"{x:.2g}" for x in de["sorted"]))
        L.append("")
    rp = res["repaired"]
    if rp["n"]:
        L.append("### Secondary: SQL with a hallucinated similarity function, re-run after replacing only that call")
        L.append("")
        L.append(rp["note"] + " Not included in any count above.")
        L.append("")
        L.append("| model | n repaired | classes after repair |")
        L.append("|---|---|---|")
        for m, c in rp["by_model"].items():
            L.append(f"| {m} | {sum(c.values())} | {fmt_counts(c)} |")
        L.append("")
        L.append("| task | classes after repair |")
        L.append("|---|---|")
        for t, c in rp["by_task"].items():
            if c:
                L.append(f"| {t} | {fmt_counts(c)} |")
        L.append("")
        L.append("| pattern (repaired SQL) | classes |")
        L.append("|---|---|")
        for p_, c in sorted(rp["pattern"].items(), key=lambda kv: -sum(kv[1].values())):
            L.append(f"| {p_} | {fmt_counts(c)} |")
        L.append("")
    L.append("### Representative failures (verbatim SQL and engine message; first sample of each class x engine x task, for (a) one per engine)")
    L.append("")
    for r in res["representative_failures"]:
        run = r["run"]
        L.append(f"**{SHORT[r['cls']]} — {r['model']} {r['prompt_id']} sample {r['sample_idx']}**")
        L.append("")
        L.append("```sql")
        L.append(r["sql"].strip())
        L.append("```")
        if "error" in run:
            L.append(f"Engine: `{json.dumps(run['error'], default=str)[:400]}`")
        else:
            det = {k: run.get(k) for k in ("n_rows", "value_col", "n_null", "n_nan", "n_inf", "missing_groups",
                                            "missing_rows", "why", "err_best", "best_variant") if run.get(k) is not None}
            L.append(f"Result (query {run['q']}): `{json.dumps(det, default=str)}`")
        L.append("")
    L.append("<!-- END GENERATED -->")
    rd = HERE / "README.md"
    txt = rd.read_text() if rd.exists() else "# llm_sql\n\n<!-- BEGIN GENERATED (evaluate.py) -->\n<!-- END GENERATED -->\n"
    new = re.sub(r"<!-- BEGIN GENERATED \(evaluate.py\) -->.*?<!-- END GENERATED -->", lambda m: "\n".join(L), txt, flags=re.S)
    rd.write_text(new)


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="re-score from the cache only")
    ap.add_argument("--engines", nargs="*", help="limit execution to these engines")
    a = ap.parse_args()
    prompts, samples = load_samples()
    rep = repaired_samples(samples)
    print(f"{len(samples)} samples from {len({s['file'] for s in samples})} files; {len(rep)} with a repairable "
          f"similarity-function name", flush=True)
    cache = load_cache()
    annot = json.loads(ANNOT.read_text()) if ANNOT.exists() else {}
    annot = {k: v for k, v in annot.items() if not k.startswith("_")}
    envs_path = HERE / "engines_env.json"
    envs = json.loads(envs_path.read_text()) if envs_path.exists() else {}
    if not a.score:
        pend = {sql_key(s["engine"], s["sql"]) for s in samples + rep} - set(cache)
        if pend:
            data = load_data()
            load_data_meta["qidx"] = data["qidx"]
            try:
                new_envs = execute_pending(samples + rep, cache, data, a.engines)
            finally:
                shutil.rmtree(WORK, ignore_errors=True)
            envs.update(new_envs)
            envs_path.write_text(json.dumps(envs, indent=1))
    if "qidx" not in load_data_meta:
        pq = HERE.parent / "postgres" / "queries.json"
        if pq.exists():
            load_data_meta["qidx"] = [q["row"] for q in json.loads(pq.read_text())["queries"][:NQ]]
    res = write_outputs(prompts, samples, cache, annot, envs, rep)
    print(json.dumps(dict(totals=res["totals"], silent=res["silent_wrong_total"], by_model=res["by_model"]), indent=1))


if __name__ == "__main__":
    main()

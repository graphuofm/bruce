"""Shared data, workload grid and SQL text for the window-inverse experiment.

The workload is claude-opus's T4_rolling_window answer (experiments/llm_sql/outputs/claude-opus.jsonl),
adapted only in dialect. One substitution for every engine: the engine-specific similarity function
(vec_distance_cosine / pgvector <=> / array_cosine_similarity) is replaced by a precomputed float64
cosine column s<q>, so that every engine slides exactly the same weights and only the window
implementation differs. -(1 - s) and (s - 1) are bitwise equal in IEEE arithmetic.
"""
import json, math
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUT = WORK / "out"
CORP = Path("/data/softavg/corpora/imdb")   # read-only
PARQUET = WORK / "reviews.parquet"

NQ = 3                          # first 3 query vectors of the PROTOCOL sampler (same rows as llm_sql)
EPS = [1e-3, 0.02, 0.1]
FRAMES = [50, 1000]
TS_BASE = np.datetime64("2000-01-01T00:00:00", "us")   # ts = TS_BASE + id seconds (as in llm_sql)


def cells():
    return [(q, eps, k) for q in range(NQ) for eps in EPS for k in FRAMES]


def cell_key(q, eps, k, variant=""):
    return f"q{q}_eps{eps:g}_k{k}" + (f"_{variant}" if variant else "")


def prepare():
    """work/reviews.parquet: id, category, rating, ts, s0..s{NQ-1} (float64 cosine to query q)."""
    import pandas as pd
    WORK.mkdir(exist_ok=True)
    df = pd.read_parquet(CORP / "corpus.parquet", columns=["facet", "value", "text"])
    emb = np.load(CORP / "emb.npy")
    rng = np.random.RandomState(0)
    titles = df.text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)[:NQ]
    pq = HERE.parent / "postgres" / "queries.json"
    if pq.exists():
        assert [q["row"] for q in json.loads(pq.read_text())["queries"][:NQ]] == [int(i) for i in qidx]
    e64 = emb.astype(np.float64)
    norm = np.linalg.norm(e64, axis=1)
    n = len(df)
    out = pd.DataFrame({"id": np.arange(n, dtype=np.int64), "category": df.facet.values.astype(str),
                        "rating": df.value.values.astype(np.float64),
                        "ts": TS_BASE + np.arange(n).astype("timedelta64[s]")})
    for qi, row in enumerate(qidx):
        q = e64[row]
        out[f"s{qi}"] = (e64 @ q) / (norm * np.linalg.norm(q))
    out.to_parquet(PARQUET, index=False)
    meta = dict(n=n, qidx=[int(i) for i in qidx], qtext=[str(titles.values[i]) for i in qidx],
                n_categories=int(out.category.nunique()),
                max_s=[float(out[f"s{qi}"].max()) for qi in range(NQ)])
    (WORK / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_df():
    import pandas as pd
    return pd.read_parquet(PARQUET)


# ----------------------------------------------------------------------------------------- SQL text
WIN = "WINDOW win AS (\n    PARTITION BY category\n    ORDER BY ts, id\n    ROWS BETWEEN {p} PRECEDING AND CURRENT ROW\n)"

TEMPLATE = """SELECT id,
       category,
       ts,
       rating,
       {SUM}(w * rating) OVER win / {SUM}(w) OVER win AS rolling_weighted_avg_rating
FROM (
    SELECT id,
           category,
           ts,
           rating,
           {W} AS w
    FROM {T}
) s
{WIN}
ORDER BY category, ts, id"""


def opus_sql(dialect, q, eps, k, table="reviews", sum_fn=None, clamp=False, nullif=False):
    """claude-opus T4 SQL in `dialect`, temperature eps, frame of k rows (k-1 PRECEDING)."""
    e = repr(float(eps))
    if dialect == "sqlite":          # opus: exp(-vec_distance_cosine(emb, ?1) / 0.02)
        w = f"exp(-(1.0 - s{q}) / {e})"
        S = sum_fn or "SUM"
    elif dialect == "postgresql":    # opus: exp(-(emb <=> $1::vector(384)) / 0.02::float8)
        z = f"-(1.0 - s{q}) / {e}::float8"
        # PG's float8 exp raises on underflow; opus's own PG answers for T1/T2 clamp with CASE ... < -700
        w = f"CASE WHEN {z} < -700 THEN 0::float8 ELSE exp({z}) END" if clamp else f"exp({z})"
        S = sum_fn or "sum"
    else:                            # duckdb / clickhouse / datafusion / spark / mysql / mariadb
        # opus (duckdb): exp((CAST(array_cosine_similarity(emb, $1) AS DOUBLE) - 1.0) / 0.02)
        w = f"exp((s{q} - 1.0) / {e})"
        S = sum_fn or ("sum" if dialect != "mysql" else "SUM")
    sql = TEMPLATE.format(SUM=S, W=w, T=table, WIN=WIN.format(p=k - 1))
    if nullif:   # not in opus's SQL: lets a frame whose weights all underflow return NULL instead of raising (PG)
        sql = sql.replace(f"/ {S}(w) OVER win", f"/ NULLIF({S}(w) OVER win, 0)")
    return sql


# ------------------------------------------------------------------- minimal non-exponential cases
MINIMAL = {
    # one large value enters and leaves a 3-row frame of small values
    "one_large": dict(x=[1.0, 1.0, 1e20, 1.0, 1.0, 1.0, 1.0], frame=3),
    # two large values of different magnitude: defeats a single compensation term (Kahan / KBN)
    "two_large": dict(x=[1.0, 1e40, 1e20, 1.0, 1.0, 1.0, 1.0], frame=3),
}


def minimal_sql(table="m", p=2):
    return f"SELECT i, SUM(x) OVER (ORDER BY i ROWS BETWEEN {p} PRECEDING AND CURRENT ROW) AS s FROM {table} ORDER BY i"


def minimal_exact(x, frame):
    """correctly rounded frame sums."""
    out = []
    for j in range(len(x)):
        out.append(float(sum(Fraction(v) for v in x[max(0, j - frame + 1):j + 1])))
    return out

"""IMDb release dates -> (item_id, ts) for the time-ordered window / maintenance runs.

Why IMDb and not Amazon / StackExchange (see README.md):
  * amazon: the raw Amazon-Reviews-2023 files carry a ms review `timestamp`,
    but build_amazon.py kept only its year (filter_num) and the raw files are
    not on this machine (re-streaming them is a ~GB download: not done,
    reported instead; build_amazon_ts.py is ready for it).
  * stackexchange: donfu/oa-stackexchange has no CreationDate / post id at all
    (METADATA = answer_score, question_score, tags).
  * imdb: the JOB/IMDb dump that the corpus came from has movie_info rows of
    info_type 16 'release dates' ("USA:22 January 2006", "(internet)" notes),
    i.e. real calendar dates per movie, locally available.

ts = the EARLIEST release date of the movie over all countries (its first
public appearance). Dates with day precision are used as is; "Month YYYY"
-> 1st of the month, "YYYY" -> 1 January (precision recorded per row).
Movies with no release-date row are absent from the table (stats report how
many). Streams movie_info.csv out of the tarball; writes ts.parquet
(item_id str, row int = corpus row index, ts timestamp[s], precision str).
"""
import csv, io, json, re, subprocess
from datetime import date
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
TGZ = Path("/home/jding/bruce/archive/datasets/job_imdb/imdb.tgz")
CORP = Path("/data/softavg/corpora/imdb/corpus.parquet")
MON = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                    "September", "October", "November", "December"], 1)}
RX = re.compile(r"^(?:(\d{1,2}) )?(?:([A-Z][a-z]+) )?(\d{4})$")
PREC = {3: "day", 2: "month", 1: "year"}

corp = pd.read_parquet(CORP, columns=["item_id", "filter_num"])
row_of = {k: i for i, k in enumerate(corp.item_id)}
best = {}  # item_id -> (date, precision rank)
n_rows = n_unparsed = 0
p = subprocess.Popen(["tar", "xzf", str(TGZ), "movie_info.csv", "-O"], stdout=subprocess.PIPE)
for r in csv.reader(io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""),
                    quotechar='"', escapechar="\\"):
    if len(r) < 4 or r[2] != "16" or r[1] not in row_of:
        continue
    n_rows += 1
    txt = r[3].split(":", 1)[-1].strip()
    m = RX.match(txt)
    if not m or (m.group(2) and m.group(2) not in MON):
        n_unparsed += 1
        continue
    d, mon, y = m.group(1), m.group(2), int(m.group(3))
    try:
        dt = date(y, MON[mon] if mon else 1, int(d) if d else 1)
    except ValueError:
        n_unparsed += 1
        continue
    prec = 3 if d else (2 if mon else 1)
    cur = best.get(r[1])
    # earliest date; at equal date prefer the more precise entry
    if cur is None or (dt, -prec) < (cur[0], -cur[1]):
        best[r[1]] = (dt, prec)
p.wait(); assert p.returncode == 0

df = pd.DataFrame([(k, row_of[k], pd.Timestamp(v[0]), PREC[v[1]]) for k, v in best.items()],
                  columns=["item_id", "row", "ts", "precision"]).sort_values(["ts", "row"]).reset_index(drop=True)
df.to_parquet(HERE / "imdb_ts.parquet", index=False)
yr = corp.set_index("item_id").filter_num.reindex(df.item_id).values
stats = dict(source=str(TGZ) + "::movie_info.csv (info_type_id=16 'release dates')", corpus=str(CORP),
             release_rows_for_corpus_movies=n_rows, unparsed_rows=n_unparsed,
             items=int(len(corp)), items_with_ts=int(len(df)), coverage=len(df) / len(corp),
             precision=df.precision.value_counts().to_dict(),
             ts_min=str(df.ts.min()), ts_max=str(df.ts.max()),
             ts_quantiles={q: str(df.ts.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
             year_minus_production_year=dict(pd.Series(df.ts.dt.year.values - yr).describe().round(3)),
             distinct_ts=int(df.ts.nunique()),
             max_items_same_ts=int(df.groupby("ts").size().max()))
(HERE / "imdb_ts_stats.json").write_text(json.dumps(stats, indent=1, default=str))
print(json.dumps(stats, indent=1, default=str))

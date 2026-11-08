"""Complete IMDb genre memberships -> bridge table (item_id, facet).

The IMDb corpus (/data/softavg/corpora/imdb, read-only) keeps
ONE genre per movie: archive/experiments/cidr_one_query/embed_titles.py took
MIN(movie_info.info) over info_type 'genres', i.e. the alphabetically first
genre. The complete memberships are in the JOB/IMDb dump
(archive/datasets/job_imdb/imdb.tgz -> movie_info.csv, info_type_id = 3
'genres'). This script streams movie_info.csv out of the tarball (nothing is
extracted to disk), keeps the genre rows of the corpus' movies, and writes

    bridge.parquet   (item_id str, facet str)   one row per (movie, genre)
    bridge_stats.json  membership statistics

Movies with no genre row get the corpus' "(none)" facet, exactly as
build_imdb.py did, so every corpus row has >= 1 membership and the
single-label facet == min(multi-label facets) for every movie (checked).
"""
import csv, io, json, subprocess, sys, tarfile
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TGZ = Path("/home/jding/bruce/archive/datasets/job_imdb/imdb.tgz")
CORP = Path("/data/softavg/corpora/imdb/corpus.parquet")

corp = pd.read_parquet(CORP, columns=["item_id", "facet"])
ids = set(corp.item_id)
assert len(ids) == len(corp), "item_id not unique"

# info_type: confirm id of 'genres'
it = subprocess.run(["tar", "xzf", str(TGZ), "info_type.csv", "-O"], capture_output=True, text=True, check=True).stdout
gid = [int(r[0]) for r in csv.reader(io.StringIO(it)) if r[1] == "genres"]
assert gid == [3], gid

p = subprocess.Popen(["tar", "xzf", str(TGZ), "movie_info.csv", "-O"], stdout=subprocess.PIPE)
rows, n_genre_rows = set(), 0
reader = csv.reader(io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""),
                    quotechar='"', escapechar="\\")
for r in reader:
    if len(r) < 4 or r[2] != "3":
        continue
    n_genre_rows += 1
    if r[1] in ids:
        rows.add((r[1], r[3]))
p.wait()
assert p.returncode == 0

br = pd.DataFrame(sorted(rows), columns=["item_id", "facet"])
have = set(br.item_id)
none = sorted(ids - have)
br = pd.concat([br, pd.DataFrame(dict(item_id=none, facet="(none)"))], ignore_index=True)
br = br.sort_values(["item_id", "facet"], kind="stable").reset_index(drop=True)

# consistency with the single-label corpus: corpus facet == min genre
mn = br.groupby("item_id").facet.min()
chk = corp.set_index("item_id").facet
mismatch = int((mn.reindex(chk.index) != chk).sum())

per_item = br.groupby("item_id").size()
dist = per_item.value_counts().sort_index()
grp = br.groupby("facet").size().sort_values(ascending=False)
single = corp.groupby("facet").size()
br.to_parquet(HERE / "bridge.parquet", index=False)
stats = dict(
    source=str(TGZ) + "::movie_info.csv (info_type_id=3 'genres')",
    corpus=str(CORP),
    genre_rows_in_dump=n_genre_rows,
    items=int(len(ids)), items_with_genre=int(len(have)), items_none=len(none),
    memberships=int(len(br)),
    memberships_per_item=dict(mean=float(per_item.mean()), median=float(per_item.median()),
                              max=int(per_item.max()), p90=float(per_item.quantile(0.9)),
                              p99=float(per_item.quantile(0.99)),
                              histogram={int(k): int(v) for k, v in dist.items()}),
    groups=int(len(grp)),
    group_sizes_multi={k: int(v) for k, v in grp.items()},
    group_sizes_single={k: int(single.get(k, 0)) for k in grp.index},
    single_label_facet_equals_min_genre_mismatches=mismatch,
)
(HERE / "bridge_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if not k.startswith("group_sizes")}, indent=1))
print(pd.DataFrame(dict(multi=grp, single=single.reindex(grp.index).fillna(0).astype(int))).to_string())

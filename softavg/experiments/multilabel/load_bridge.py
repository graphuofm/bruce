#!/usr/bin/env python
"""Load the multi-label bridge into the release server next to `imdb`.

  imdb_genre(id int, facet text)   -- id = imdb.id (corpus row index); one row per (movie, genre)
  btree(id), btree(facet); reloption parallel_workers = 8

Needs experiments/postgres/load.py rel 3 to have run (table imdb with s0..s2).
usage: load_bridge.py   -> load_bridge.json
"""
import io, json, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postgres"))
from common import CORP, connect  # noqa: E402

corp = pd.read_parquet(CORP / "corpus.parquet", columns=["item_id"])
row_of = pd.Series(range(len(corp)), index=corp.item_id)
br = pd.read_parquet(HERE / "bridge.parquet")
br["id"] = row_of.reindex(br.item_id).values
assert br.id.notna().all()
conn = connect("rel"); cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS imdb_genre CASCADE")
cur.execute("CREATE TABLE imdb_genre(id int NOT NULL, facet text NOT NULL)")
buf = io.StringIO()
br[["id", "facet"]].to_csv(buf, sep="\t", header=False, index=False); buf.seek(0)
t0 = time.perf_counter()
cur.copy_expert("COPY imdb_genre FROM STDIN", buf)
cur.execute("CREATE INDEX imdb_genre_id ON imdb_genre(id); CREATE INDEX imdb_genre_facet ON imdb_genre(facet); "
            "ALTER TABLE imdb_genre SET (parallel_workers = 8)")
cur.execute("VACUUM ANALYZE imdb_genre")
st = dict(load_s=time.perf_counter() - t0)
cur.execute("SELECT count(*), count(DISTINCT id), count(DISTINCT facet), pg_total_relation_size('imdb_genre') FROM imdb_genre")
st.update(zip(["rows", "items", "groups", "total_bytes"], cur.fetchone()))
cur.execute("SELECT count(*) FROM imdb i JOIN imdb_genre g USING (id) WHERE i.filter_num >= 2000")
st["join_rows_ge2000"] = cur.fetchone()[0]
(HERE / "load_bridge.json").write_text(json.dumps(st, indent=1))
print(json.dumps(st, indent=1))

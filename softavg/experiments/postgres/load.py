#!/usr/bin/env python
"""Load IMDb (459,865 rows) into one of the two servers.

  imdb(id int PRIMARY KEY  -- row index in corpus.parquet,
       facet text, value float8, filter_num float8,
       [with pgvector] emb vector(384),
       s0..s{P-1} float8)     -- precomputed scores of queries 0..P-1:
                                 with pgvector: -(emb <#> q) by pgvector itself (CTAS);
                                 without: float32 dot product in numpy, as float8.
                                 The reference is always computed from the
                                 scores the server holds.

With pgvector the table also gets emb vector(384) and the HNSW index
(vector_ip_ops, m=16, ef_construction=64). Always: btree on facet.
Creates the extensions and two harness-only aggregates built from objects
the 0.2.0 extension ships:
  softavg_v012 -- the 0.1.2 float8[5] state path (bruce_softavg_sfunc/combine/final)
  softavg_fwd  -- 0.2.0 internal state WITHOUT the moving-aggregate form
                  (window frames are re-aggregated per row)

usage: load.py rel|dbg [P]
"""
import io, json, struct, sys, time
import numpy as np
import pandas as pd
import psycopg2
from common import CORP, DBNAME, HERE, SERVERS, connect, load_queries, query_embeddings, vec_literal

SERVER = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CHUNK = 20000

HARNESS_AGGS = """
DROP AGGREGATE IF EXISTS softavg_v012(float8, float8, float8);
CREATE AGGREGATE softavg_v012(float8, float8, float8) (
    SFUNC = bruce_softavg_sfunc, STYPE = float8[], COMBINEFUNC = bruce_softavg_combine,
    FINALFUNC = bruce_softavg_final, PARALLEL = SAFE);
DROP AGGREGATE IF EXISTS softavg_fwd(float8, float8, float8);
CREATE AGGREGATE softavg_fwd(float8, float8, float8) (
    SFUNC = softavg_sfunc, STYPE = internal, SSPACE = 40, COMBINEFUNC = softavg_combine,
    SERIALFUNC = softavg_serialize, DESERIALFUNC = softavg_deserialize,
    FINALFUNC = softavg_finalfn, PARALLEL = SAFE);
"""


def copy_payload(ids, facets, values, fnum, emb=None, scores=None):
    buf = io.BytesIO()
    buf.write(b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0))
    nf = 4 + (emb is not None) + (0 if scores is None else scores.shape[1])
    if emb is not None:
        emb_be = np.ascontiguousarray(emb.astype(">f4"))
        vhdr = struct.pack(">ihh", 4 + 4 * emb.shape[1], emb.shape[1], 0)
    for i in range(len(ids)):
        f = facets[i].encode()
        buf.write(struct.pack(">hii", nf, 4, int(ids[i])))
        buf.write(struct.pack(">i", len(f))); buf.write(f)
        buf.write(struct.pack(">idid", 8, float(values[i]), 8, float(fnum[i])))
        if emb is not None:
            buf.write(vhdr); buf.write(emb_be[i].tobytes())
        if scores is not None:
            for x in scores[i]:
                buf.write(struct.pack(">id", 8, float(x)))
    buf.write(struct.pack(">h", -1)); buf.seek(0)
    return buf


def main():
    stats = dict(server=SERVER, n_precomputed_queries=P)
    admin = psycopg2.connect(host="127.0.0.1", port=SERVERS[SERVER]["port"], dbname="postgres", user="jding")
    admin.autocommit = True
    with admin.cursor() as c:
        c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DBNAME,))
        if not c.fetchone():
            c.execute(f"CREATE DATABASE {DBNAME}")
    admin.close()
    conn = connect(SERVER); cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS bruce_pg")
    cur.execute("SELECT count(*) = 2 FROM pg_available_extensions WHERE name IN ('vector', 'pg_ivm')")
    has_vec = cur.fetchone()[0]
    stats["pgvector_and_pg_ivm"] = has_vec
    if has_vec:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_ivm")
    cur.execute(HARNESS_AGGS)

    df = pd.read_parquet(CORP / "corpus.parquet", columns=["facet", "value", "filter_num"])
    n = len(df)
    emb = np.load(CORP / "emb.npy", mmap_mode="r")
    assert emb.shape == (n, 384)
    qs = load_queries()["queries"][:P]
    Q = query_embeddings([q["row"] for q in qs])
    ids = np.arange(n); facets = df.facet.values; values = df.value.values; fnum = df.filter_num.values

    cur.execute("DROP TABLE IF EXISTS imdb CASCADE; DROP TABLE IF EXISTS imdb_stage")
    scols = ", ".join(f"s{j} float8" for j in range(P))
    t0 = time.perf_counter()
    if has_vec:
        cur.execute("CREATE UNLOGGED TABLE imdb_stage(id int, facet text, value float8, filter_num float8, emb vector(384))")
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            cur.copy_expert("COPY imdb_stage FROM STDIN WITH (FORMAT binary)",
                            copy_payload(ids[a:b], facets[a:b], values[a:b], fnum[a:b], emb=np.asarray(emb[a:b])))
        stats["copy_s"] = time.perf_counter() - t0
        sexpr = ", ".join(f"-(emb <#> {vec_literal(Q[j])}) AS s{j}" for j in range(P))
        t0 = time.perf_counter()
        cur.execute(f"CREATE TABLE imdb AS SELECT id, facet, value, filter_num, emb, {sexpr} FROM imdb_stage")
        stats["ctas_s"] = time.perf_counter() - t0
        cur.execute("DROP TABLE imdb_stage")
    else:
        S = np.asarray(emb, dtype=np.float32) @ Q.T  # (n, P) float32 inner products
        cur.execute(f"CREATE TABLE imdb(id int, facet text, value float8, filter_num float8, {scols})")
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            cur.copy_expert("COPY imdb FROM STDIN WITH (FORMAT binary)",
                            copy_payload(ids[a:b], facets[a:b], values[a:b], fnum[a:b], scores=S[a:b]))
        stats["copy_s"] = time.perf_counter() - t0
    cur.execute("ALTER TABLE imdb ADD PRIMARY KEY (id); ALTER TABLE imdb SET (parallel_workers = 8)")
    t0 = time.perf_counter(); cur.execute("VACUUM ANALYZE imdb"); stats["vacuum_analyze_s"] = time.perf_counter() - t0
    cur.execute("CREATE INDEX imdb_facet_idx ON imdb(facet)")
    if has_vec:
        t0 = time.perf_counter()
        cur.execute("SET maintenance_work_mem = '4GB'; SET max_parallel_maintenance_workers = 8")
        cur.execute("CREATE INDEX imdb_emb_hnsw ON imdb USING hnsw (emb vector_ip_ops) WITH (m = 16, ef_construction = 64)")
        stats["hnsw_build_s"] = time.perf_counter() - t0
        stats["hnsw"] = dict(opclass="vector_ip_ops", m=16, ef_construction=64)
        cur.execute("VACUUM ANALYZE imdb")
    cur.execute("SELECT count(*), count(DISTINCT facet), count(*) FILTER (WHERE filter_num >= 2000), "
                "count(DISTINCT facet) FILTER (WHERE filter_num >= 2000), pg_total_relation_size('imdb'), "
                "pg_relation_size('imdb'), min(s0), max(s0) FROM imdb")
    r = cur.fetchone()
    stats.update(rows=r[0], n_facets=r[1], rows_ge2000=r[2], n_facets_ge2000=r[3], total_bytes=r[4], heap_bytes=r[5],
                 s0_min=r[6], s0_max=r[7])
    if has_vec:
        cur.execute("SELECT pg_relation_size('imdb_emb_hnsw')"); stats["hnsw_bytes"] = cur.fetchone()[0]
    (HERE / f"load_{SERVER}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

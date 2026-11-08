"""Run the rolling softmax-weighted average (and the minimal cases) on one engine.

usage: .venv/bin/python run_engine.py <engine>
writes work/out/<engine>.npz (per-cell result arrays, indexed by id; NaN = NULL/NaN) and
work/out/<engine>.json (version, per-cell status / error text, minimal-case outputs, SQL used).
"""
import json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

from common import (EPS, FRAMES, MINIMAL, NQ, OUT, PARQUET, WORK, cell_key, cells, load_df, minimal_sql,
                    opus_sql)

PG_BIN = Path.home() / ".pgrx/17.10/pg-release-install/bin"
PG_PORT = 28871
MY_PORT = 28872


def to_vals(ids, vals, n):
    out = np.full(n, np.nan)
    v = np.array([np.nan if x is None else float(x) for x in vals], dtype=np.float64) \
        if not isinstance(vals, np.ndarray) else vals.astype(np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    assert len(np.unique(ids)) == len(ids) == n, (len(ids), n)
    out[ids] = v
    return out


# ============================================================================== engines
class Engine:
    dialect = "duckdb"
    variants = [("", {})]           # (variant name, opus_sql kwargs)
    def minimal(self, x):           # -> list of frame sums
        raise NotImplementedError


class SQLiteStd(Engine):
    name, dialect = "sqlite_stdlib", "sqlite"
    def __init__(self, df):
        import sqlite3
        self.c = sqlite3.connect(":memory:")
        self.c.execute("CREATE TABLE reviews(id INTEGER PRIMARY KEY, category TEXT, rating REAL, ts TEXT, "
                       + ", ".join(f"s{q} REAL" for q in range(NQ)) + ")")
        ts = df.ts.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        rows = zip(df.id.tolist(), df.category.tolist(), df.rating.tolist(), ts, *[df[f"s{q}"].tolist() for q in range(NQ)])
        self.c.executemany(f"INSERT INTO reviews VALUES (?,?,?,?{',?' * NQ})", rows)
        self.version = dict(engine="SQLite", version=sqlite3.sqlite_version, binding=f"Python {sys.version.split()[0]} sqlite3")
    def run(self, sql):
        r = self.c.execute(sql).fetchall()
        return [x[0] for x in r], [x[-1] for x in r]
    def minimal(self, x, frame):
        self.c.execute("DROP TABLE IF EXISTS m"); self.c.execute("CREATE TABLE m(i INTEGER, x REAL)")
        self.c.executemany("INSERT INTO m VALUES (?,?)", list(enumerate(x)))
        return [r[1] for r in self.c.execute(minimal_sql(p=frame - 1))]


class SQLiteApsw(SQLiteStd):
    name = "sqlite_apsw"
    def __init__(self, df):
        import apsw
        self.c = apsw.Connection(":memory:")
        self.c.execute("CREATE TABLE reviews(id INTEGER PRIMARY KEY, category TEXT, rating REAL, ts TEXT, "
                       + ", ".join(f"s{q} REAL" for q in range(NQ)) + ")")
        ts = df.ts.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        rows = list(zip(df.id.tolist(), df.category.tolist(), df.rating.tolist(), ts, *[df[f"s{q}"].tolist() for q in range(NQ)]))
        with self.c:
            self.c.executemany(f"INSERT INTO reviews VALUES (?,?,?,?{',?' * NQ})", rows)
        self.version = dict(engine="SQLite", version=apsw.sqlite_lib_version(), binding=f"apsw {apsw.apsw_version()} (bundled SQLite)")
    def run(self, sql):
        r = list(self.c.execute(sql))
        return [x[0] for x in r], [x[-1] for x in r]
    def minimal(self, x, frame):
        self.c.execute("DROP TABLE IF EXISTS m"); self.c.execute("CREATE TABLE m(i INTEGER, x REAL)")
        self.c.executemany("INSERT INTO m VALUES (?,?)", list(enumerate(x)))
        return [r[1] for r in self.c.execute(minimal_sql(p=frame - 1))]


class DuckDB(Engine):
    name, dialect = "duckdb", "duckdb"
    def __init__(self, df):
        import duckdb
        self.c = duckdb.connect()
        self.c.execute(f"CREATE TABLE reviews AS SELECT * FROM read_parquet('{PARQUET}')")
        self.version = dict(engine="DuckDB", version=duckdb.__version__,
                            threads=self.c.execute("SELECT current_setting('threads')").fetchone()[0])
    def run(self, sql):
        t = self.c.execute(sql).fetchnumpy()
        return t["id"], np.ma.filled(np.ma.asarray(t["rolling_weighted_avg_rating"]).astype(float), np.nan)
    def minimal(self, x, frame):
        self.c.execute("CREATE OR REPLACE TABLE m(i INTEGER, x DOUBLE)")
        self.c.executemany("INSERT INTO m VALUES (?,?)", list(enumerate(x)))
        return [r[1] for r in self.c.execute(minimal_sql(p=frame - 1)).fetchall()]


class Postgres(Engine):
    name, dialect = "postgresql", "postgresql"
    def __init__(self, df):
        import psycopg2
        self.dir = WORK / "pg"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        subprocess.run([str(PG_BIN / "initdb"), "-D", str(self.dir), "-U", "jding", "--no-locale", "-E", "UTF8"],
                       check=True, capture_output=True)
        with open(self.dir / "postgresql.conf", "a") as f:
            f.write(f"\nport = {PG_PORT}\nlisten_addresses = '127.0.0.1'\nunix_socket_directories = '{self.dir}'\n"
                    "shared_buffers = 1GB\nwork_mem = 512MB\ntimezone = 'UTC'\nfsync = off\nsynchronous_commit = off\n")
        subprocess.run([str(PG_BIN / "pg_ctl"), "-D", str(self.dir), "-l", str(WORK / "pg.log"), "-w", "start"],
                       check=True, capture_output=True)
        self.started = True
        self.conn = psycopg2.connect(host="127.0.0.1", port=PG_PORT, dbname="postgres", user="jding")
        self.conn.autocommit = True
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE reviews(id bigint PRIMARY KEY, category text, rating double precision, ts timestamp, "
                    + ", ".join(f"s{q} double precision" for q in range(NQ)) + ")")
        import io
        buf = io.StringIO()
        cols = [df.id.values, df.category.values, df.rating.values, df.ts.dt.strftime("%Y-%m-%d %H:%M:%S").values] + \
               [df[f"s{q}"].values for q in range(NQ)]
        for row in zip(*cols):
            buf.write("\t".join([str(row[0]), row[1], repr(float(row[2])), row[3]] + [repr(float(v)) for v in row[4:]]) + "\n")
        buf.seek(0)
        cur.copy_expert("COPY reviews FROM STDIN", buf)
        cur.execute("ANALYZE reviews")
        # the PG manual's own example of an unsafe moving aggregate (xaggr.sgml, Moving-Aggregate Mode)
        cur.execute("CREATE AGGREGATE unsafe_sum (float8) (stype = float8, sfunc = float8pl, "
                    "mstype = float8, msfunc = float8pl, minvfunc = float8mi)")
        cur.execute("SELECT version(), current_setting('server_version')")
        v, sv = cur.fetchone()
        cur.execute("SELECT a.aggfnoid::regprocedure::text, a.aggminvtransfn::regproc::text FROM pg_aggregate a "
                    "WHERE a.aggfnoid::regprocedure::text IN ('sum(double precision)','avg(double precision)',"
                    "'sum(numeric)','unsafe_sum(double precision)')")
        inv = dict(cur.fetchall())
        self.version = dict(engine="PostgreSQL", version=sv, server=v, build="release (~/.pgrx/17.10/pg-release-install)",
                            aggminvtransfn=inv)
    # "" = opus verbatim (raises on exp underflow at eps=1e-3); clamp = opus's own PG clamp, only where needed
    # clamp_nullif / unsafe_sum / numeric add NULLIF(den, 0) so that eps = 1e-3 frames whose weights all
    # underflow (0/0, the formula's own failure) give NULL instead of PG's division-by-zero error
    variants = [("", {}), ("clamp", dict(clamp=True), [1e-3]), ("clamp_nullif", dict(clamp=True, nullif=True), [1e-3]),
                ("unsafe_sum", dict(clamp=True, nullif=True, sum_fn="unsafe_sum")),
                ("numeric", dict(clamp=True, nullif=True, sum_fn="numeric"))]
    def sql(self, q, eps, k, **kw):
        if kw.get("sum_fn") == "numeric":
            s = opus_sql("postgresql", q, eps, k, clamp=True)
            return s.replace("sum(w * rating) OVER win / sum(w) OVER win",
                             "(sum((w * rating)::numeric) OVER win / NULLIF(sum(w::numeric) OVER win, 0))::float8")
        return opus_sql("postgresql", q, eps, k, **kw)
    def run(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        r = cur.fetchall()
        return [x[0] for x in r], [x[-1] for x in r]
    def minimal(self, x, frame):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS m"); cur.execute("CREATE TABLE m(i int, x float8)")
        cur.executemany("INSERT INTO m VALUES (%s,%s)", list(enumerate(x)))
        out = {}
        for fn in ("SUM", "unsafe_sum"):
            cur.execute(minimal_sql(p=frame - 1).replace("SUM(", fn + "("))
            out[fn] = [r[1] for r in cur.fetchall()]
        return out
    def close(self):
        try:
            self.conn.close()
        finally:
            subprocess.run([str(PG_BIN / "pg_ctl"), "-D", str(self.dir), "-m", "fast", "-w", "stop"], capture_output=True)
            shutil.rmtree(self.dir, ignore_errors=True)


class ClickHouse(Engine):
    name, dialect = "clickhouse", "clickhouse"
    def __init__(self, df):
        import chdb
        from chdb import session
        self.s = session.Session()
        self.s.query(f"CREATE TABLE reviews ENGINE = Memory AS SELECT * FROM file('{PARQUET}', Parquet)")
        ver = self.s.query("SELECT version()", "CSV").data().strip().strip('"')
        self.version = dict(engine="ClickHouse (chdb, embedded)", version=ver, chdb=chdb.__version__)
    def run(self, sql):
        t = self.s.query(sql, "Arrow")
        import pyarrow as pa
        tab = pa.ipc.open_file(pa.BufferReader(t.bytes())).read_all() if hasattr(t, "bytes") else t
        return tab.column("id").to_numpy(), tab.column("rolling_weighted_avg_rating").to_numpy(zero_copy_only=False)
    def minimal(self, x, frame):
        self.s.query("DROP TABLE IF EXISTS m")
        self.s.query("CREATE TABLE m (i Int32, x Float64) ENGINE = Memory")
        self.s.query("INSERT INTO m VALUES " + ",".join(f"({i},{v!r})" for i, v in enumerate(x)))
        r = self.s.query(minimal_sql(p=frame - 1), "CSV").data().strip().splitlines()
        return [float(l.split(",")[1]) for l in r]


class DataFusion(Engine):
    name, dialect = "datafusion", "datafusion"
    def __init__(self, df):
        import datafusion
        self.ctx = datafusion.SessionContext()
        self.ctx.register_parquet("reviews_pq", str(PARQUET))
        self.ctx.sql("CREATE TABLE reviews AS SELECT * FROM reviews_pq").collect()
        self.version = dict(engine="Apache DataFusion (Python bindings)", version=datafusion.__version__)
    def run(self, sql):
        tab = self.ctx.sql(sql).to_arrow_table()
        return tab.column("id").to_numpy(), tab.column("rolling_weighted_avg_rating").to_numpy(zero_copy_only=False)
    def minimal(self, x, frame):
        import pyarrow as pa
        self.ctx.deregister_table("m") if self.ctx.table_exist("m") else None
        self.ctx.register_record_batches("m", [[pa.record_batch({"i": list(range(len(x))), "x": x})]])
        tab = self.ctx.sql(minimal_sql(p=frame - 1)).to_arrow_table()
        return tab.column("s").to_pylist()


class Spark(Engine):
    name, dialect = "spark", "spark"
    def __init__(self, df):
        from pyspark.sql import SparkSession
        import pyspark
        self.sp = (SparkSession.builder.master("local[8]").appName("window_inverse")
                   .config("spark.driver.memory", "12g").config("spark.sql.session.timeZone", "UTC")
                   .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                   .config("spark.ui.enabled", "false").config("spark.local.dir", str(WORK / "spark_tmp"))
                   .getOrCreate())
        self.sp.sparkContext.setLogLevel("ERROR")
        self.sp.read.parquet(str(PARQUET)).cache().createOrReplaceTempView("reviews")
        self.sp.sql("SELECT count(*) FROM reviews").collect()
        jv = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
        self.version = dict(engine="Apache Spark SQL (local[8])", version=pyspark.__version__, java=jv,
                            ansi=self.sp.conf.get("spark.sql.ansi.enabled"))
    def run(self, sql):
        pdf = self.sp.sql(sql).select("id", "rolling_weighted_avg_rating").toPandas()
        return pdf.id.values, pdf.rolling_weighted_avg_rating.astype(float).values
    def minimal(self, x, frame):
        self.sp.createDataFrame([(i, float(v)) for i, v in enumerate(x)], "i int, x double").createOrReplaceTempView("m")
        return [r["s"] for r in self.sp.sql(minimal_sql(p=frame - 1)).collect()]
    def close(self):
        self.sp.stop()
        shutil.rmtree(WORK / "spark_tmp", ignore_errors=True)


class Polars(Engine):
    """two spellings: the dataframe API rolling_sum().over() (default) and the opus SQL through pl.SQLContext."""
    name, dialect = "polars", "polars"
    variants = [("", {}), ("sql", dict(sql=True))]
    def __init__(self, df):
        import polars as pl
        self.pl = pl
        self.df = pl.read_parquet(PARQUET).sort(["category", "ts", "id"])
        self.version = dict(engine="Polars", version=pl.__version__)
    def sql(self, q, eps, k, **kw):
        if kw.get("sql"):
            return opus_sql("duckdb", q, eps, k)
        return (f"df.sort(category, ts, id).with_columns(w = ((s{q} - 1.0) / {eps!r}).exp()); "
                f"rolling_sum(w * rating, window_size={k}, min_samples=1).over('category') / "
                f"rolling_sum(w, window_size={k}, min_samples=1).over('category')")
    def run_cell(self, q, eps, k, **kw):
        pl = self.pl
        if kw.get("sql"):
            out = self.pl.SQLContext(reviews=self.df).execute(opus_sql("duckdb", q, eps, k), eager=True)
            return out["id"].to_numpy(), out["rolling_weighted_avg_rating"].to_numpy()
        w = ((pl.col(f"s{q}") - 1.0) / eps).exp()
        out = self.df.with_columns(w=w).with_columns(
            r=(pl.col("w") * pl.col("rating")).rolling_sum(window_size=k, min_samples=1).over("category")
              / pl.col("w").rolling_sum(window_size=k, min_samples=1).over("category"))
        return out["id"].to_numpy(), out["r"].to_numpy()
    def sql_probe(self):
        """does Polars SQL accept the opus window query at all?"""
        try:
            ctx = self.pl.SQLContext(reviews=self.df)
            ctx.execute(opus_sql("duckdb", 0, 0.02, 50), eager=True)
            return "accepted"
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:300]}"
    def minimal(self, x, frame):
        return self.pl.Series(x).rolling_sum(window_size=frame, min_samples=1).to_list()


class Pandas(Engine):
    name, dialect = "pandas", "pandas"
    def __init__(self, df):
        import pandas as pd
        self.pd = pd
        self.df = df.sort_values(["category", "ts", "id"]).reset_index(drop=True)
        self.version = dict(engine="pandas", version=pd.__version__)
    def sql(self, q, eps, k, **kw):
        return (f"w = exp((s{q} - 1.0) / {eps!r}); g = df.sort_values([category, ts, id]).groupby('category'); "
                f"g[w*rating].rolling({k}, min_periods=1).sum() / g[w].rolling({k}, min_periods=1).sum()")
    def run_cell(self, q, eps, k, **kw):
        d = self.df
        w = np.exp((d[f"s{q}"].values - 1.0) / eps)
        tmp = self.pd.DataFrame({"category": d.category.values, "w": w, "wv": w * d.rating.values})
        g = tmp.groupby("category", sort=False)
        num = g["wv"].rolling(k, min_periods=1).sum().reset_index(level=0, drop=True).sort_index()
        den = g["w"].rolling(k, min_periods=1).sum().reset_index(level=0, drop=True).sort_index()
        return d.id.values, (num / den).values
    def minimal(self, x, frame):
        return self.pd.Series(x).rolling(frame, min_periods=1).sum().tolist()


class MySQLish(Engine):
    """MySQL / MariaDB from a binary tarball, run as the current user in work/<name>."""
    dialect = "mysql"
    def __init__(self, df, basedir, flavor):
        import pymysql
        self.base = Path(basedir)
        self.data = WORK / f"{flavor}_data"
        self.sock = WORK / f"{flavor}.sock"
        shutil.rmtree(self.data, ignore_errors=True)
        self.flavor = flavor
        if flavor == "mysql":
            subprocess.run([str(self.base / "bin/mysqld"), "--no-defaults", "--initialize-insecure",
                            f"--basedir={self.base}", f"--datadir={self.data}"], check=True, capture_output=True)
            cmd = [str(self.base / "bin/mysqld"), "--no-defaults", f"--basedir={self.base}", f"--datadir={self.data}",
                   f"--port={MY_PORT}", f"--socket={self.sock}", "--bind-address=127.0.0.1", "--local-infile=1",
                   "--mysqlx=OFF", f"--log-error={WORK / 'mysql.err'}", f"--pid-file={WORK / 'mysql.pid'}"]
        else:
            subprocess.run([str(self.base / "scripts/mariadb-install-db"), "--no-defaults", f"--basedir={self.base}",
                            f"--datadir={self.data}", "--auth-root-authentication-method=normal"],
                           check=True, capture_output=True)
            cmd = [str(self.base / "bin/mariadbd"), "--no-defaults", f"--basedir={self.base}", f"--datadir={self.data}",
                   f"--port={MY_PORT}", f"--socket={self.sock}", "--bind-address=127.0.0.1", "--local-infile=1",
                   f"--log-error={WORK / 'mariadb.err'}", f"--pid-file={WORK / 'mariadb.pid'}"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                self.conn = pymysql.connect(unix_socket=str(self.sock), user="root", local_infile=True, autocommit=True)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        cur = self.conn.cursor()
        cur.execute("CREATE DATABASE w"); cur.execute("USE w")
        cur.execute("CREATE TABLE reviews(id BIGINT PRIMARY KEY, category VARCHAR(64), rating DOUBLE, ts DATETIME, "
                    + ", ".join(f"s{q} DOUBLE" for q in range(NQ)) + ")")
        csv = WORK / f"{flavor}_reviews.tsv"
        cols = [df.id.values, df.category.values, df.rating.values, df.ts.dt.strftime("%Y-%m-%d %H:%M:%S").values] + \
               [df[f"s{q}"].values for q in range(NQ)]
        with open(csv, "w") as f:
            for row in zip(*cols):
                f.write("\t".join([str(row[0]), row[1], repr(float(row[2])), row[3]] + [repr(float(v)) for v in row[4:]]) + "\n")
        cur.execute(f"LOAD DATA LOCAL INFILE '{csv}' INTO TABLE reviews FIELDS TERMINATED BY '\\t'")
        csv.unlink()
        cur.execute("SELECT COUNT(*), SUM(s0) FROM reviews")
        self.loaded = cur.fetchone()
        cur.execute("SELECT VERSION()")
        self.version = dict(engine=("MySQL" if flavor == "mysql" else "MariaDB"), version=cur.fetchone()[0],
                            binary=str(self.base))
        if flavor == "mysql":
            cur.execute("SELECT @@GLOBAL.windowing_use_high_precision")
            self.version["windowing_use_high_precision_default"] = cur.fetchone()[0]
    def run(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        r = cur.fetchall()
        return [x[0] for x in r], [x[-1] for x in r]
    def minimal(self, x, frame):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS m"); cur.execute("CREATE TABLE m(i INT, x DOUBLE)")
        cur.executemany("INSERT INTO m VALUES (%s,%s)", list(enumerate(x)))
        cur.execute(minimal_sql(p=frame - 1))
        return [float(r[1]) for r in cur.fetchall()]
    def close(self):
        try:
            self.conn.cursor().execute("SHUTDOWN")
        except Exception:
            pass
        try:
            self.proc.wait(60)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.data, ignore_errors=True)


class MySQL(MySQLish):
    name = "mysql"
    variants = [("", {}), ("high_precision_off", {})]
    def __init__(self, df):
        super().__init__(df, os.environ["MYSQL_BASE"], "mysql")
    def set_variant(self, v):
        self.conn.cursor().execute("SET SESSION windowing_use_high_precision = " + ("OFF" if v == "high_precision_off" else "ON"))
    def minimal(self, x, frame):
        out = {}
        for v in ("on", "off"):
            self.conn.cursor().execute(f"SET SESSION windowing_use_high_precision = {v.upper()}")
            out[f"windowing_use_high_precision={v.upper()}"] = MySQLish.minimal(self, x, frame)
        self.conn.cursor().execute("SET SESSION windowing_use_high_precision = ON")
        return out


class MariaDB(MySQLish):
    name = "mariadb"
    def __init__(self, df):
        super().__init__(df, os.environ["MARIADB_BASE"], "mariadb")


ENGINES = {c.name: c for c in [SQLiteStd, SQLiteApsw, DuckDB, Postgres, ClickHouse, DataFusion, Spark, Polars, Pandas,
                               MySQL, MariaDB]}


def main():
    name = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_df()
    n = len(df)
    cls = ENGINES[name]
    eng = cls(df)
    meta = dict(engine=name, version=eng.version, cells={}, sql={}, minimal={})
    arrays = {}
    if only:
        meta = json.loads((OUT / f"{name}.json").read_text())
        meta["version"] = eng.version
        arrays = dict(np.load(OUT / f"{name}.npz"))
        for key in [k for k in meta["cells"] if "_".join(k.split("_")[3:]) in only]:
            meta["cells"].pop(key); arrays.pop(key, None)
    try:
        if hasattr(eng, "sql_probe"):
            meta["sql_probe"] = eng.sql_probe()
        for var in cls.variants:
            vname, kw = var[0], var[1]
            eps_only = var[2] if len(var) > 2 else None
            if only is not None and vname not in only:
                continue
            if hasattr(eng, "set_variant"):
                eng.set_variant(vname)
            for (q, eps, k) in cells():
                if eps_only is not None and eps not in eps_only:
                    continue
                key = cell_key(q, eps, k, vname)
                if hasattr(eng, "run_cell"):
                    sql = eng.sql(q, eps, k, **kw)
                elif hasattr(eng, "sql"):
                    sql = eng.sql(q, eps, k, **kw)
                else:
                    sql = opus_sql(cls.dialect, q, eps, k, **kw)
                if q == 0:
                    meta["sql"][cell_key(0, eps, k, vname)] = sql
                t0 = time.time()
                try:
                    ids, vals = eng.run_cell(q, eps, k, **kw) if hasattr(eng, "run_cell") else eng.run(sql)
                    arrays[key] = to_vals(ids, vals, n)
                    meta["cells"][key] = dict(status="ok", seconds=round(time.time() - t0, 2))
                except Exception as e:
                    meta["cells"][key] = dict(status="error", error=f"{type(e).__name__}: {str(e).strip()[:400]}")
                print(name, key, meta["cells"][key].get("status"), meta["cells"][key].get("error", ""), flush=True)
        for mname, m in MINIMAL.items():
            try:
                meta["minimal"][mname] = eng.minimal(m["x"], m["frame"])
            except Exception as e:
                meta["minimal"][mname] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if hasattr(eng, "close"):
            eng.close()
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()

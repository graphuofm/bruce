"""Shared setup for the SQLite before/after experiments (PROTOCOL.md).

Builds (once) work/imdb.db with the IMDb corpus as the user would hold it in
SQLite: one row per item, the 384-d embedding as a float32 BLOB in
sqlite-vec's format, and optionally a sqlite-vec vec0 table for KNN top-k.
work/ is derived data and is deleted after the runs (see README).
"""
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ICDE = HERE.parents[1]
ENG = ICDE / "engines" / "sqlite"
CORE = ICDE / "engines" / "core"
SO = ENG / "target" / "release" / "libsoftagg_sqlite"
VEC0 = HERE / "deps" / "sqlite_vec-0.1.9" / "vec0"
CORP = Path("/data/softavg/corpora/imdb")
WORK = HERE / "work"
DB = WORK / "imdb.db"
MIN_NUM = 2000.0

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from quiet import cpu_busy_fraction, require_quiet  # noqa: E402

EPS_GRID = [1e-4, 1e-3, 0.02, 0.1, 1.0, float("inf")]


def eps_name(e):
    return "inf" if e == float("inf") else repr(e)


def load_corpus():
    df = pd.read_parquet(CORP / "corpus.parquet")
    emb = np.load(CORP / "emb.npy")
    assert emb.dtype == np.float32 and emb.shape[0] == len(df)
    return df, emb


def queries(df, emb, n):
    """The PROTOCOL query draw: RandomState(0), first sentence of 4-20 words,
    de-duplicated, 200 drawn; the first n of those 200 are used."""
    rng = np.random.RandomState(0)
    titles = df.text.str.split(".").str[0].str.strip()
    ok = titles.str.split().str.len().between(4, 20) & ~titles.duplicated()
    qidx = rng.choice(np.flatnonzero(ok.values), size=200, replace=False)
    return qidx[:n], emb[qidx[:n]]


def connect(vec=True, db=DB):
    c = sqlite3.connect(str(db))
    c.enable_load_extension(True)
    c.load_extension(str(SO))
    if vec:
        c.load_extension(str(VEC0))
    c.enable_load_extension(False)
    c.execute("PRAGMA cache_size = -4000000")      # 4 GB page cache: warm runs
    c.execute("PRAGMA mmap_size = 4000000000")
    c.execute("PRAGMA temp_store = MEMORY")
    return c


def build_db(df, emb, with_vec0=True):
    WORK.mkdir(exist_ok=True)
    if DB.exists():
        c = connect()
        have = {r[0] for r in c.execute("SELECT name FROM sqlite_master")}
        if "imdb" in have and (not with_vec0 or "vec_imdb" in have):
            return c
        c.close()
        DB.unlink()
    t0 = time.time()
    c = connect()
    c.execute("PRAGMA journal_mode = OFF")
    c.execute("PRAGMA synchronous = OFF")
    c.execute("CREATE TABLE imdb(id INTEGER PRIMARY KEY, facet TEXT NOT NULL, value REAL NOT NULL, "
              "filter_num REAL, emb BLOB NOT NULL)")
    rows = ((i, f, float(v), float(fn), emb[i].tobytes())
            for i, (f, v, fn) in enumerate(zip(df.facet.values, df.value.values, df.filter_num.values)))
    c.executemany("INSERT INTO imdb VALUES (?,?,?,?,?)", rows)
    c.commit()
    t_tab = time.time() - t0
    t_vec = None
    if with_vec0:
        t0 = time.time()
        c.execute("CREATE VIRTUAL TABLE vec_imdb USING vec0(emb float[384] distance_metric=cosine, "
                  "filter_num float)")
        c.execute("INSERT INTO vec_imdb(rowid, emb, filter_num) SELECT id, emb, filter_num FROM imdb")
        c.commit()
        t_vec = time.time() - t0
    (WORK / "build.json").write_text(json.dumps({"table_load_s": t_tab, "vec0_build_s": t_vec}))
    return c


def sha256_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.relative_to(ICDE).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def provenance(c):
    ext_src = list((ENG / "src").glob("*.rs")) + [ENG / "Cargo.toml", ENG / "Cargo.lock"]
    core_src = list((CORE / "src").glob("*.rs")) + [CORE / "Cargo.toml"]
    core_ver = [l.split('"')[1] for l in (CORE / "Cargo.toml").read_text().splitlines()
                if l.startswith("version")][0]
    opts = [r[0] for r in c.execute("PRAGMA compile_options")]
    try:
        quiet = require_quiet(wait_seconds=0, verbose=False)
        quiet["quiet"] = True
    except SystemExit as e:
        quiet = {"quiet": False, "require_quiet": str(e), "cpu_busy_fraction": round(cpu_busy_fraction(), 3),
                 "load_average_1min": round(os.getloadavg()[0], 2)}
    return {
        "host_engine": "SQLite " + sqlite3.sqlite_version + " (Python " + platform.python_version()
                       + " stdlib sqlite3, system libsqlite3)",
        "sqlite_math_functions": "ENABLE_MATH_FUNCTIONS" in opts,
        "extension": "softagg-sqlite " + c.execute("SELECT softagg_sqlite_version()").fetchone()[0],
        "extension_source_sha256": sha256_files(ext_src),
        "extension_git_commit": None,
        "extension_git_note": "paper_icde is not a git repository; source hash given instead",
        "build_type": "release (cargo build --release; opt-level=3, lto=fat, codegen-units=1, panic=abort)",
        "extension_so_sha256": hashlib.sha256(Path(str(SO) + ".so").read_bytes()).hexdigest(),
        "softagg_core_version": core_ver,
        "softagg_core_source_sha256": sha256_files(core_src),
        "sqlite_vec": {
            "version": c.execute("SELECT vec_version()").fetchone()[0],
            "source": "PyPI wheel sqlite_vec-0.1.9-py3-none-manylinux_2_17_x86_64 (pip download), vec0.so extracted",
            "vec0_so_sha256": hashlib.sha256(Path(str(VEC0) + ".so").read_bytes()).hexdigest(),
        },
        "machine": {"cpu": platform.processor() or platform.machine(), "nproc": os.cpu_count(),
                    "load": quiet},
        "timing_status": "smoke, machine shared",
        "threads": 1,
        "threads_note": "SQLite executes one statement on one thread; no parallel aggregation exists to configure",
    }


def softavg_ref(codes, G, s, v, eps):
    """numpy float64 two-pass max-shift, per group; None for empty groups."""
    out = [None] * G
    if eps == float("inf"):
        cnt = np.bincount(codes, minlength=G)
        sm = np.bincount(codes, weights=v, minlength=G)
        return [sm[g] / cnt[g] if cnt[g] else None for g in range(G)]
    m = np.full(G, -np.inf)
    np.maximum.at(m, codes, s)
    w = np.exp((s - m[codes]) / eps)
    num = np.bincount(codes, weights=w * v, minlength=G)
    den = np.bincount(codes, weights=w, minlength=G)
    for g in range(G):
        if den[g] > 0:
            out[g] = num[g] / den[g]
    return out

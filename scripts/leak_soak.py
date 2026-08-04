#!/usr/bin/env python3
"""Workstream 20 — leak sampling soak (docs/TESTING_MATRIX.md).

Loops build/drop of the allocation-heavy session objects and asserts
RSS stays flat. Two phases:

  A. SESSION churn: construct a full QuerySession (register a
     4096 x 32-key table, build two maintained views, run queries,
     write through the views), then drop the whole session. Exercises
     the Drop path of the entire database object graph.
  B. IN-SESSION churn: one long-lived session; each cycle re-registers
     the table over its own name (PG: replace = drop + create, cascade
     drops dependent views), re-attaches keys, rebuilds views, writes,
     queries. Exercises the replace/cascade drop path.

Indexes: HNSW index lifecycle is not exposed through the Python wheel
as of tonight (the hnsw-planner track owns index build/drop, Rust
side); this soak covers tables + views. Extend when the wheel gains
an index API.

Method: valgrind is NOT installed on this box (`which valgrind` is
empty), so sampling is /proc/self/status VmRSS, reported as medians
of 5-cycle blocks (box runs concurrent builds; RSS of THIS process is
load-independent, but medians are reported regardless, with loadavg
recorded). ASAN status: needs nightly rustc — see UNSAFE_INVENTORY.md.

Bounds (same justification as bruce-py/tests/test_lifecycle.py): the
table generation is ~1.2 MiB; one leaked generation per cycle is
>= 200 MiB per phase, measured clean growth is < 1 MiB — bound
32 MiB post-warmup growth per phase.

Run:  python3 scripts/leak_soak.py        (~15 s)
Exit: 0 = PASS, 1 = any phase exceeded its bound.
Output: docs/qa/leak_soak.json
"""

import json
import pathlib
import statistics
import sys
import tempfile
import time

import numpy as np

import bruce

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "qa" / "leak_soak.json"

N, D, G = 4096, 32, 8
EPS = 0.5
CYCLES = 200
WARMUP = 20          # allocator arena growth happens here
BOUND_KB = 32 * 1024
BLOCK = 5            # median block size

Q = ("SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.5) "
     "FROM movies GROUP BY genre")

failures = []


def rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found")


def block_medians(samples):
    return [int(statistics.median(samples[i:i + BLOCK]))
            for i in range(0, len(samples) - BLOCK + 1, BLOCK)]


def check(name, samples, cycle_times):
    med = block_medians(samples)
    warm_blocks = WARMUP // BLOCK
    growth = med[-1] - med[warm_blocks]
    t_ms = statistics.median(cycle_times) * 1e3
    ok = growth < BOUND_KB
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: post-warmup growth {growth} kB "
          f"over {len(samples) - WARMUP} cycles (bound {BOUND_KB} kB), "
          f"median cycle {t_ms:.1f} ms")
    if not ok:
        failures.append(f"{name}: {growth} kB")
    return {"phase": name, "cycles": len(samples), "warmup": WARMUP,
            "block_medians_kb": med, "post_warmup_growth_kb": growth,
            "bound_kb": BOUND_KB, "median_cycle_ms": round(t_ms, 2)}


def build_session(pq, emb, x):
    s = bruce.QuerySession()
    s.register_parquet("movies", pq)
    s.attach_key("movies", "emb", emb)
    s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)
    s.create_view("v2", "movies", "genre", "year", "emb", x, eps=1.0)
    return s


def churn(s, pq, emb, x):
    s.insert_row("movies", {"rating": 5.0, "year": 2020.0},
                 {"genre": "g0"}, {"emb": np.zeros(D)})
    s.delete_where("movies", "year", ">=", 2024.5)
    for _ in range(3):
        labels, values, _ = s.run(Q, {"q": x})
        assert len(labels) == G and np.isfinite(values).all()


def main() -> int:
    import pandas as pd
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((N, D))
    x = rng.standard_normal(D)

    with tempfile.TemporaryDirectory() as td:
        pq = str(pathlib.Path(td) / "soak.parquet")
        pd.DataFrame({
            "genre": [f"g{i % G}" for i in range(N)],
            "rating": rng.uniform(0, 10, N),
            "year": rng.uniform(1990, 2025, N),
        }).to_parquet(pq)

        phases = []

        # ---- A: whole-session build/drop -------------------------
        print(f"[A] session churn x{CYCLES}")
        samples, times = [], []
        for _ in range(CYCLES):
            t0 = time.perf_counter()
            s = build_session(pq, emb, x)
            churn(s, pq, emb, x)
            del s
            times.append(time.perf_counter() - t0)
            samples.append(rss_kb())
        phases.append(check("session-churn", samples, times))

        # ---- B: in-session replace/cascade churn -----------------
        print(f"[B] in-session replace churn x{CYCLES}")
        s = build_session(pq, emb, x)
        samples, times = [], []
        for _ in range(CYCLES):
            t0 = time.perf_counter()
            s.register_parquet("movies", pq)   # drops v1, v2 (cascade)
            s.attach_key("movies", "emb", emb)
            s.create_view("v1", "movies", "genre", "rating", "emb", x, eps=EPS)
            s.create_view("v2", "movies", "genre", "year", "emb", x, eps=1.0)
            churn(s, pq, emb, x)
            times.append(time.perf_counter() - t0)
            samples.append(rss_kb())
        phases.append(check("replace-churn", samples, times))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "date": "2026-08-03",
        "workstream": 20,
        "method": "/proc/self/status VmRSS, 5-cycle block medians "
                  "(valgrind not installed on this box)",
        "loadavg": open("/proc/loadavg").read().split()[:3],
        "table": {"rows": N, "key_dim": D, "groups": G},
        "phases": phases,
    }, indent=1))

    print()
    if failures:
        print(f"LEAK SOAK FAIL ({len(failures)}): {failures}")
        return 1
    print(f"LEAK SOAK PASS -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

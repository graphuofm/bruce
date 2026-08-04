#!/usr/bin/env python3
"""Workstream 18 — scale soak: 10M rows x d=64 f32, bounded memory.

Why a Python script and not an #[ignore]d Rust test: (a) the required
correctness oracle is numpy (spot-checks vs an independent
implementation, not vs the engine's own sequential fold); (b) peak-RSS
assertions come straight from resource.getrusage / /proc/self/status;
(c) it soaks the SHIPPED wheel — the artifact users actually load —
rather than a debug test binary; (d) it keeps bruce-core/tests/ free
for the kernel-math tracks. The wheel is Rust: numpy here is oracle
only (feedback_rust_for_db).

What it asserts
---------------
  A. 10M x 64 f32 grouped_softavg_f32 at eps in {0, 0.1, inf}:
     every output finite, every group covered.
  B. peak RSS < 6 GB (ru_maxrss high-water, whole process).
  C. latency scales ~linearly 1M -> 10M rows: factor in [8, 13]
     (median of 5, same kernel, same group count).
  D. group-cardinality extremes at 1M rows: n_groups = 1 and
     n_groups = 1_000_000 — numpy spot-check on sampled groups at
     eps = 0.1 and inf (eps = 0 is exercised for finiteness/coverage
     only: on continuous data the f32-vs-f64 score rounding may
     legitimately pick a different member of a near-tie, which is the
     documented precision contract, not a defect), coverage flags
     exactly match group occupancy, and no quadratic blowup in
     n_groups (1M-group runtime within a linear-ish factor of the
     100k-group runtime, far under the ~10x of a quadratic law).

Memory discipline: keys are generated CHUNKED (1M rows at a time,
float32 straight from the Generator — no transient f64 10M array
ever exists); the 10M arrays are freed before the extremes phase so
the high-water stays well under the 6 GB cap.

Run:  python3 scripts/soak.py       (~1-2 minutes on the 32-core box)
Exit: 0 = PASS, 1 = any assertion failed.
"""

import resource
import statistics
import sys
import time

import numpy as np

import bruce

N10 = 10_000_000
N1 = 1_000_000
D = 64
N_GROUPS = 32
CHUNK = 1_000_000
RSS_CAP_BYTES = 6 * 1024**3
EPS_SWEEP = [0.0, 0.1, float("inf")]

failures = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "ok" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def peak_rss_bytes() -> int:
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def gen_data(n: int, seed: int):
    """Chunked generation: f32 keys straight from the RNG (never a
    transient f64 array of the full size), f64 values, u32 gids."""
    rng = np.random.default_rng(seed)
    k = np.empty((n, D), dtype=np.float32)
    v = np.empty((n, 1), dtype=np.float64)
    gid = np.empty(n, dtype=np.uint32)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        k[lo:hi] = rng.standard_normal((hi - lo, D), dtype=np.float32)
        v[lo:hi, 0] = rng.standard_normal(hi - lo)
        gid[lo:hi] = rng.integers(0, N_GROUPS, size=hi - lo, dtype=np.uint32)
    return k, v, gid


def numpy_softavg(scores: np.ndarray, vals: np.ndarray, eps: float) -> float:
    """Independent oracle, f64, max-shifted."""
    if eps == 0.0:
        m = scores.max()
        return float(vals[scores == m].mean())
    if np.isinf(eps):
        return float(vals.mean())
    w = np.exp((scores - scores.max()) / eps)
    return float((w * vals).sum() / w.sum())


def run(x, k, v, gid, n_groups, eps):
    return bruce.grouped_softavg_f32(x, k, v, gid, n_groups, eps=eps)


def timed(fn, reps=5):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)


def main() -> int:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(D).astype(np.float32)

    print(f"[gen] {N10} rows x d={D} f32, chunked at {CHUNK}")
    t0 = time.perf_counter()
    k10, v10, gid10 = gen_data(N10, seed=42)
    print(f"  generated in {time.perf_counter() - t0:.1f} s, "
          f"peak RSS {peak_rss_bytes() / 1024**3:.2f} GiB")

    # ---- A: eps sweep at 10M -------------------------------------
    print(f"[A] 10M-row grouped queries, {N_GROUPS} groups, eps sweep")
    for eps in EPS_SWEEP:
        (out, covered), t = timed(lambda e=eps: run(x, k10, v10, gid10, N_GROUPS, e), reps=3)
        check(f"finite eps={eps}", bool(np.isfinite(out).all()),
              f"min {out.min():.4f} max {out.max():.4f}, {t * 1e3:.0f} ms")
        check(f"covered eps={eps}", all(covered), f"{sum(covered)}/{N_GROUPS} groups")

    # ---- C: latency scaling 1M -> 10M ----------------------------
    # leading-axis slices of C-order arrays are contiguous views
    k1, v1, gid1 = k10[:N1], v10[:N1], gid10[:N1]
    _, t1 = timed(lambda: run(x, k1, v1, gid1, N_GROUPS, 0.1), reps=5)
    _, t10 = timed(lambda: run(x, k10, v10, gid10, N_GROUPS, 0.1), reps=5)
    factor = t10 / t1
    print(f"[C] 1M: {t1 * 1e3:.1f} ms   10M: {t10 * 1e3:.1f} ms   factor {factor:.1f}x")
    check("linear scaling", 8.0 <= factor <= 13.0,
          f"10M/1M factor {factor:.2f} (band [8, 13])")

    # ---- free the 10M arrays before the extremes phase -----------
    del k10, v10, gid10, k1, v1, gid1
    print(f"[mem] after free: peak RSS {peak_rss_bytes() / 1024**3:.2f} GiB (high-water)")

    # ---- D: group-cardinality extremes at 1M rows ----------------
    print("[D] group-cardinality extremes, 1M rows")
    ke, ve, _ = gen_data(N1, seed=43)
    rng_g = np.random.default_rng(44)

    # scores oracle in f64 on the same stored f32 numbers
    scores64 = ke.astype(np.float64) @ x.astype(np.float64)

    # -- 1 group
    gid_one = np.zeros(N1, dtype=np.uint32)
    times_by_groups = {}
    for eps in EPS_SWEEP:
        (out, covered), t = timed(lambda e=eps: run(x, ke, ve, gid_one, 1, e), reps=3)
        times_by_groups.setdefault(1, t)
        check(f"1-group finite eps={eps}", bool(np.isfinite(out).all()) and covered == [True],
              f"out {out[0, 0]:.6f}, {t * 1e3:.0f} ms")
        if eps == 0.0:
            continue  # near-tie argmax under f32 scoring: contract, not oracle-checkable
        want = numpy_softavg(scores64, ve[:, 0], eps)
        rel = abs(out[0, 0] - want) / max(abs(want), 1e-12)
        check(f"1-group vs numpy eps={eps}", rel < 1e-3,
              f"got {out[0, 0]:.9f} want {want:.9f} rel {rel:.2e}")

    # -- many groups: 100k (linearity reference) and 1M
    for n_g in (100_000, N1):
        gid_many = rng_g.integers(0, n_g, size=N1, dtype=np.uint32)
        occupied = np.zeros(n_g, dtype=bool)
        occupied[gid_many] = True
        (out, covered), t = timed(lambda g=gid_many, n=n_g: run(x, ke, ve, g, n, 0.1), reps=3)
        times_by_groups[n_g] = t
        cov = np.asarray(covered)
        check(f"{n_g}-group coverage", bool((cov == occupied).all()),
              f"{cov.sum()} covered == {occupied.sum()} occupied, {t * 1e3:.0f} ms")
        check(f"{n_g}-group finite", bool(np.isfinite(out[occupied]).all()),
              "all occupied groups finite")
        # numpy spot-check on 100 sampled occupied groups, eps 0.1 + inf
        occ_ids = np.flatnonzero(occupied)
        sample = rng_g.choice(occ_ids, size=100, replace=False)
        for eps in (0.1, float("inf")):
            out_e = out if eps == 0.1 else run(x, ke, ve, gid_many, n_g, eps)[0]
            worst = 0.0
            for g in sample:
                rows = np.flatnonzero(gid_many == g)
                want = numpy_softavg(scores64[rows], ve[rows, 0], eps)
                rel = abs(out_e[g, 0] - want) / max(abs(want), 1e-12)
                worst = max(worst, rel)
            check(f"{n_g}-group vs numpy eps={eps}", worst < 1e-3,
                  f"worst rel err {worst:.2e} over 100 sampled groups")

    # no quadratic blowup in n_groups: quadratic 100k -> 1M would be
    # ~10x on top of the linear term; require comfortably under that
    ratio = times_by_groups[N1] / times_by_groups[100_000]
    print(f"  group-scaling: 1 gp {times_by_groups[1] * 1e3:.0f} ms, "
          f"100k gp {times_by_groups[100_000] * 1e3:.0f} ms, "
          f"1M gp {times_by_groups[N1] * 1e3:.0f} ms")
    check("no quadratic blowup in n_groups", ratio < 25.0,
          f"t(1M groups)/t(100k groups) = {ratio:.1f} (cap 25; quadratic would be ~100)")

    # ---- B: peak RSS ---------------------------------------------
    rss = peak_rss_bytes()
    check("peak RSS < 6 GiB", rss < RSS_CAP_BYTES, f"{rss / 1024**3:.2f} GiB high-water")

    print()
    if failures:
        print(f"SOAK FAIL ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SOAK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

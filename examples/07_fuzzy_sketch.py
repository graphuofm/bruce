"""Fuzzy join via linear-attention sketch — O(d_phi * d_v) state.

Build a sketch over 1,000,000 random vectors. The sketch state is
just `d_phi * d_v` doubles — independent of N. Subsequent queries are
constant-time, regardless of how many rows were ingested.

Run:
    python examples/07_fuzzy_sketch.py
"""

import time

import numpy as np

import bruce


def main() -> None:
    rng = np.random.default_rng(0)
    d_phi, d_v = 64, 16

    # Build sketches at different N — the sketch byte size should be
    # IDENTICAL because it's O(d_phi * d_v), not O(N).
    print(f"d_phi = {d_phi}, d_v = {d_v}")
    print(f"\n{'N':>10s}  {'build':>10s}  {'query':>10s}  {'state bytes':>12s}")
    print("-" * 50)
    q = rng.normal(size=d_phi)

    for N in [1_000, 10_000, 100_000, 1_000_000]:
        K = rng.normal(size=(N, d_phi))
        V = rng.normal(size=(N, d_v))

        t0 = time.perf_counter()
        s = bruce.FuzzyJoinSketch(K, V, phi="elu+1")
        t_build = (time.perf_counter() - t0) * 1000

        # warm
        _ = s.query(q)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = s.query(q)
        t_query = (time.perf_counter() - t0) / 100 * 1000

        print(f"{N:>10,d}  {t_build:>8.1f}ms  {t_query:>8.3f}ms  "
              f"{s.size_bytes:>10,d} B")

    print()
    print("Note: state bytes is IDENTICAL across N — that's the point")
    print("of the linear-attention kernel trick (Katharopoulos 2020). The")
    print("sketch answers any fuzzy similarity query in O(d_phi * d_v),")
    print("regardless of how many rows were ingested.")


if __name__ == "__main__":
    main()

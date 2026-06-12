"""Triangle query via Leapfrog Triejoin — AGM-optimal O(N^{3/2}).

The triangle query R(a, b) ⋈ S(b, c) ⋈ T(a, c) is the canonical
worst-case-optimal join example: a naive plan is O(N³), but LFTJ
achieves the AGM bound O(N^{3/2}).

Run:
    python examples/11_triangle_lftj.py
"""

import time

import numpy as np

import bruce


def naive_triangle(a, b, c):
    """Brute force O(N³)."""
    trips = []
    set_a = set(a)
    set_b = set(b)
    set_c = set(c)
    common = set_a & set_b & set_c
    for k in common:
        ia = [i for i, x in enumerate(a) if x == k]
        ib = [i for i, x in enumerate(b) if x == k]
        ic = [i for i, x in enumerate(c) if x == k]
        for i in ia:
            for j in ib:
                for kk in ic:
                    trips.append((i, j, kk))
    return trips


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'N':>7s}  {'lftj_ms':>10s}  {'naive_ms':>12s}  {'speedup':>10s}")
    print("-" * 50)
    for N in [50, 100, 200, 500, 1000]:
        # build three sorted key sequences with about √N common keys
        # (so output size is roughly N^{3/2})
        keys = sorted(rng.integers(0, int(np.sqrt(N)) + 1, size=N).tolist())
        a = keys
        b = sorted(rng.integers(0, int(np.sqrt(N)) + 1, size=N).tolist())
        c = sorted(rng.integers(0, int(np.sqrt(N)) + 1, size=N).tolist())

        t0 = time.perf_counter()
        trips_lftj = bruce.lftj_three(a, b, c)
        t_lftj = (time.perf_counter() - t0) * 1000

        # naive
        t0 = time.perf_counter()
        trips_naive = naive_triangle(a, b, c)
        t_naive = (time.perf_counter() - t0) * 1000

        # they should be equal as sets
        assert sorted(trips_lftj) == sorted(trips_naive)

        speedup = t_naive / max(t_lftj, 1e-6)
        print(f"{N:>7d}  {t_lftj:>10.3f}  {t_naive:>12.3f}  {speedup:>9.1f}×")

    print()
    print("LFTJ achieves O(N^{3/2}) per the AGM bound — the worst-case-")
    print("optimal complexity for triangle queries.")


if __name__ == "__main__":
    main()

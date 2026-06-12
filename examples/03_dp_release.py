"""Differential privacy — releasing aggregate statistics with provable
noise guarantees.

Run:
    python examples/03_dp_release.py
"""

import numpy as np

import bruce


def main() -> None:
    # Sensitive dataset: 1000 customer balances
    rng = np.random.default_rng(0)
    balances = rng.normal(loc=5000, scale=2000, size=1000)
    true_total = float(balances.sum())

    print(f"True total (sensitive!): ${true_total:,.2f}")

    # The query is "sum of balances". L1 sensitivity of a SUM where
    # each balance is bounded to [0, 10000] is 10000.
    print("\nε-DP releases via the Laplace mechanism (l1_sensitivity=10000):")
    print(f"{'ε':<10s}  {'released':>15s}  {'error':>12s}")
    print("-" * 50)
    for eps in [0.1, 0.5, 1.0, 5.0]:
        m = bruce.LaplaceMechanism(l1_sensitivity=10000.0, epsilon=eps)
        # Average 10 releases to see the noise scale
        releases = [m.release_scalar(true_total) for _ in range(10)]
        avg = float(np.mean(releases))
        err = abs(avg - true_total)
        print(f"  ε={eps:<6.2f}  ${avg:>13,.2f}   ±${err:>10,.0f}")

    print("\n(ε, δ)-DP via Gaussian (l2_sensitivity=10000, δ=1e-5):")
    print(f"{'ε':<10s}  {'σ':>10s}  {'released':>15s}")
    print("-" * 50)
    for eps in [0.1, 1.0, 5.0]:
        m = bruce.GaussianMechanism(l2_sensitivity=10000.0,
                                       epsilon=eps, delta=1e-5)
        released = m.release_scalar(true_total)
        print(f"  ε={eps:<6.2f}  σ={m.sigma:>8.0f}   ${released:>13,.2f}")

    print("\nLower ε → stronger privacy → noisier release.")


if __name__ == "__main__":
    main()

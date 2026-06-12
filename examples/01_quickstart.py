"""Bruce quickstart — the F_ε operator across ε.

Run:
    pip install bruce
    python examples/01_quickstart.py

Demonstrates the central object of Bruce: the F_ε operator at four
temperatures. Same query, same memory, different ε → different
behaviour.

This is the picture that motivates the whole framework: SQL at ε=0,
softmax attention at ε=1, smooth interpolation in between.
"""

import numpy as np

import bruce


def main() -> None:
    # tiny memory: 3 records
    x = np.array([1.0, 0.0])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]])

    print(f"Bruce {bruce.__version__}")
    print(f"\nQuery x = {x.tolist()}")
    print(f"Memory rows = {len(K)}")
    print(f"\n{'ε':<10s}  {'sim':<10s}  attention output")
    print("-" * 60)

    for eps, sim, label in [
        (0.0,  "indicator", "ε=0 (tropical / exact SQL)"),
        (0.25, "dot",       "ε=0.25"),
        (1.0,  "dot",       "ε=1.0 (standard softmax)"),
        (4.0,  "dot",       "ε=4.0 (smoothed)"),
    ]:
        op = bruce.Operator(eps=eps, sim=sim)
        out = op.attention(x, K, V)
        print(f"{label:<22s}  {sim:<10s}  {out.tolist()}")

    print()
    print("Observations:")
    print("  - ε=0 + indicator: picks only the row exactly matching x → [10, 0]")
    print("  - ε=0.25: very sharp softmax, close to ε=0 (top-1) but smooth")
    print("  - ε=1.0:  standard softmax-attention")
    print("  - ε=4.0:  approaches the uniform mean over rows")


if __name__ == "__main__":
    main()

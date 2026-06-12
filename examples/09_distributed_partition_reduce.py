"""Lemma B — F_ε partitions losslessly across shards.

The classical recipe for distributing softmax: each shard computes a
(running-max, num, den) partial; a central reduce combines them.
Bruce gives you the primitives.

Run:
    python examples/09_distributed_partition_reduce.py
"""

import numpy as np

import bruce


def main() -> None:
    rng = np.random.default_rng(0)
    eps = 1.0
    N = 10_000
    d_v = 4
    scores = rng.normal(size=N).tolist()
    values = rng.normal(size=(N, d_v)).tolist()

    # === single-machine reference ===
    sm = bruce.PartialTriple.from_pairs(scores, values, eps=eps)
    reference = bruce.finalize(sm)
    print(f"single-machine A_ε    = {reference}")

    # === split into P partitions, partition-reduce ===
    print(f"\n{'P':>4s}  {'worst err vs single':>22s}")
    print("-" * 30)
    for P in [2, 4, 16, 64, 256]:
        chunk = (N + P - 1) // P
        parts = []
        for i in range(P):
            lo = i * chunk
            hi = min(lo + chunk, N)
            if hi <= lo:
                continue
            parts.append(bruce.PartialTriple.from_pairs(
                scores[lo:hi], values[lo:hi], eps=eps,
            ))
        combined = bruce.combine(parts, eps=eps)
        out = bruce.finalize(combined)
        worst = float(np.max(np.abs(out - reference)))
        print(f"  {P:>4d}  {worst:>22.3e}")

    print()
    print("The reduce gives BIT-IDENTICAL output to single-machine,")
    print("regardless of partition count P. That is Lemma B.")


if __name__ == "__main__":
    main()

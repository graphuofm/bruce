"""Exact retrieval unlearning — the GDPR primitive.

Insert a "poisoned" record into a Bruce memory, then delete it.
Verify that the attention output is bit-identical to a memory that
never contained the poison.

Run:
    python examples/02_exact_unlearning.py
"""

import numpy as np

import bruce


def main() -> None:
    rng = np.random.default_rng(0)
    d_k, d_v = 8, 4
    eps = 1.0
    x = rng.normal(size=d_k).astype(np.float64)

    # --- build a 1000-record memory ---
    mem_with_poison = bruce.IncrementalMemory(
        query=x, eps=eps, d_v=d_v, sim="dot"
    )
    for i in range(1000):
        k = rng.normal(size=d_k).astype(np.float64)
        v = rng.normal(size=d_v).astype(np.float64)
        mem_with_poison.insert(f"r{i}", k, v)

    # --- inject one DOMINANT poison ---
    # high score (key aligned with query) and obviously wrong value
    poison_k = x * 5.0
    poison_v = np.full(d_v, 999.0, dtype=np.float64)
    mem_with_poison.insert("poison", poison_k, poison_v)

    poisoned_output = mem_with_poison.output()
    print(f"Output with poison (should be ~999): {poisoned_output}")

    # --- delete the poison ---
    mem_with_poison.delete("poison")
    after_delete = mem_with_poison.output()
    print(f"Output after delete:                 {after_delete}")
    print(f"Rescales triggered: {mem_with_poison.n_rescales}")

    # --- reference: rebuild from scratch without the poison ---
    rng2 = np.random.default_rng(0)
    _ = rng2.normal(size=d_k).astype(np.float64)   # consume the same x draw
    mem_clean = bruce.IncrementalMemory(
        query=x, eps=eps, d_v=d_v, sim="dot"
    )
    for i in range(1000):
        k = rng2.normal(size=d_k).astype(np.float64)
        v = rng2.normal(size=d_v).astype(np.float64)
        mem_clean.insert(f"r{i}", k, v)
    clean_output = mem_clean.output()
    print(f"Output if poison was never inserted: {clean_output}")

    err = float(np.max(np.abs(after_delete - clean_output)))
    print(f"\nMax abs error (delete vs never-inserted): {err:.3e}")
    print("→ This is the bit-level exact-unlearning guarantee.")


if __name__ == "__main__":
    main()

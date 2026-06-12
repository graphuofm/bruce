"""Streaming chain join — FlashAttention's online-softmax transferred
to a database 3-way chain join.

R(a, b) ⋈ S(b, c) ⋈ T(c, d): tuples arrive one at a time across the
three relations, and we maintain the count of join answers incrementally.
Each arrival triggers O(matches) work, not O(N).

Run:
    python examples/10_streaming_chain_join.py
"""

import bruce


def main() -> None:
    s = bruce.StreamingChainJoin()

    print("Stream of tuples arriving across R, S, T:")
    print(f"{'event':<30s}  {'new emitted':>12s}  {'total':>8s}")
    print("-" * 60)

    events = [
        ("R", "alice", "team-engineering"),
        ("S", "team-engineering", "project-bruce"),
        ("T", "project-bruce", "milestone-v1"),
        ("R", "bob",   "team-engineering"),
        ("R", "carol", "team-engineering"),
        ("T", "project-bruce", "milestone-v2"),
        ("S", "team-engineering", "project-other"),
        ("T", "project-other", "milestone-other"),
    ]
    for kind, x, y in events:
        if kind == "R":
            new = s.arrive_r(x, y)
        elif kind == "S":
            new = s.arrive_s(x, y)
        else:  # T
            new = s.arrive_t(x, y)
        label = f"{kind}({x!r}, {y!r})"
        print(f"  {label:<28s}  {new:>12d}  {s.n_emitted:>8d}")

    print()
    print("Each arrival's cost is O(matches), not O(|R|+|S|+|T|). For a")
    print("naive recompute on every arrival, total work would be ~N²; with")
    print("streaming, total work is O(|output|).")


if __name__ == "__main__":
    main()

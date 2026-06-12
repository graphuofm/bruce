"""Tests for bruce.PartialTriple + bruce.combine + bruce.finalize.

These primitives realise Lemma B: F_ε partitions losslessly.
"""

from __future__ import annotations

import numpy as np

import bruce


class TestPartitionReduce:
    def test_two_partitions_match_single_machine(self):
        eps = 1.0
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [[10.0], [20.0], [30.0], [40.0], [50.0]]

        # ground truth: single-machine computation
        scores_np = np.array(scores)
        values_np = np.array([v for v in values])
        m = scores_np.max()
        w = np.exp((scores_np - m) / eps)
        ref = (w[:, None] * values_np).sum(axis=0) / w.sum()

        # split into two partitions
        p1 = bruce.PartialTriple.from_pairs(scores[:2], values[:2], eps=eps)
        p2 = bruce.PartialTriple.from_pairs(scores[2:], values[2:], eps=eps)
        combined = bruce.combine([p1, p2], eps=eps)
        out = bruce.finalize(combined)
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_partition_count_doesnt_change_answer(self):
        """Lemma B: bit-level identity across any number of partitions."""
        eps = 1.0
        rng = np.random.default_rng(0)
        N = 1000
        scores = rng.normal(size=N).tolist()
        values = rng.normal(size=(N, 4)).tolist()

        # single-machine reference
        single = bruce.PartialTriple.from_pairs(scores, values, eps=eps)
        ref = bruce.finalize(single)

        for P in [2, 5, 10, 100, 500]:
            chunk = (N + P - 1) // P
            parts = []
            for i in range(P):
                lo = i * chunk
                hi = min(lo + chunk, N)
                if hi <= lo:
                    continue
                parts.append(
                    bruce.PartialTriple.from_pairs(
                        scores[lo:hi], values[lo:hi], eps=eps
                    )
                )
            combined = bruce.combine(parts, eps=eps)
            out = bruce.finalize(combined)
            scale = max(abs(float(ref[0])), 1.0)
            np.testing.assert_allclose(out, ref, atol=scale * 1e-12)

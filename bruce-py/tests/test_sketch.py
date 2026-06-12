"""Tests for bruce.FuzzyJoinSketch (linear-attention kernel sketch)."""

from __future__ import annotations

import numpy as np
import pytest

import bruce


class TestFuzzyJoinSketch:
    def test_build_returns_sketch(self):
        rng = np.random.default_rng(0)
        K = rng.normal(size=(100, 32))
        V = rng.normal(size=(100, 4))
        s = bruce.FuzzyJoinSketch(K, V, phi="elu+1")
        assert s.n_rows == 100

    def test_size_is_independent_of_N(self):
        """The whole point of the sketch: state size is O(d_phi * d_v),
        independent of N."""
        K_small = np.random.default_rng(0).normal(size=(100, 32))
        V_small = np.random.default_rng(1).normal(size=(100, 4))
        K_big = np.random.default_rng(2).normal(size=(100_000, 32))
        V_big = np.random.default_rng(3).normal(size=(100_000, 4))

        s1 = bruce.FuzzyJoinSketch(K_small, V_small, phi="elu+1")
        s2 = bruce.FuzzyJoinSketch(K_big, V_big, phi="elu+1")

        assert s1.size_bytes == s2.size_bytes, (
            f"sketch size depends on N: {s1.size_bytes} vs {s2.size_bytes}"
        )

    def test_query_constant_time(self):
        """Query latency should not grow with N."""
        import time
        d_phi, d_v = 32, 4
        rng = np.random.default_rng(0)
        q = rng.normal(size=d_phi)

        # build at N=1000 and N=100,000, time the query
        for N in [1_000, 100_000]:
            K = rng.normal(size=(N, d_phi))
            V = rng.normal(size=(N, d_v))
            s = bruce.FuzzyJoinSketch(K, V, phi="elu+1")
            # warm + measure
            _ = s.query(q)
            t0 = time.perf_counter()
            for _ in range(10):
                s.query(q)
            t = (time.perf_counter() - t0) / 10
            # both should be sub-millisecond regardless of N
            assert t < 0.01, f"query at N={N} took {t*1000:.2f}ms"

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            bruce.FuzzyJoinSketch(np.zeros((1, 2)), np.zeros((1, 2)), phi="cosine")

    def test_query_dim_mismatch_raises(self):
        s = bruce.FuzzyJoinSketch(np.zeros((1, 4)), np.zeros((1, 2)), phi="elu+1")
        with pytest.raises(ValueError, match="dim mismatch"):
            s.query(np.zeros(8))

    def test_incremental_add_consistent_with_batch(self):
        """add()-then-query equals batch-build-then-query."""
        rng = np.random.default_rng(42)
        K = rng.normal(size=(50, 16))
        V = rng.normal(size=(50, 4))
        # batch
        batch = bruce.FuzzyJoinSketch(K, V, phi="elu+1")
        # incremental
        incr = bruce.FuzzyJoinSketch(np.zeros((0, 16)), np.zeros((0, 4)), phi="elu+1")
        for i in range(50):
            incr.add(K[i], V[i])

        q = rng.normal(size=16)
        qb = batch.query(q)
        qi = incr.query(q)
        np.testing.assert_allclose(qb, qi, atol=1e-12)

    def test_numerator_denominator_accessors(self):
        s = bruce.FuzzyJoinSketch(
            np.ones((10, 8)), np.ones((10, 3)), phi="elu+1"
        )
        n = s.numerator()
        d = s.denominator()
        assert n.shape == (8, 3)
        assert d.shape == (8,)

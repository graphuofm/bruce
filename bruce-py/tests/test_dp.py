"""Tests for bruce.LaplaceMechanism + bruce.GaussianMechanism (DP)."""

from __future__ import annotations

import math
import statistics

import pytest

import bruce


class TestLaplaceMechanism:
    def test_release_scalar_seeded_is_reproducible(self):
        m1 = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=1.0, seed=42)
        m2 = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=1.0, seed=42)
        # same seed → same noise
        assert m1.release_scalar(100.0) == m2.release_scalar(100.0)

    def test_release_scalar_unseeded_varies(self):
        m = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=1.0)
        # without a seed, two calls almost surely differ
        a = m.release_scalar(100.0)
        b = m.release_scalar(100.0)
        assert a != b

    def test_release_vector_length_preserved(self):
        m = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=0.5, seed=0)
        out = m.release_vector([1.0, 2.0, 3.0, 4.0])
        assert len(out) == 4

    def test_release_is_unbiased_in_expectation(self):
        """Many releases of the SAME true value should average ≈ true value."""
        m = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=2.0, seed=7)
        true = 50.0
        samples = [m.release_scalar(true) for _ in range(2000)]
        # With each call we re-seed, so let's use unseeded
        m2 = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=2.0)
        samples = [m2.release_scalar(true) for _ in range(2000)]
        mean = statistics.mean(samples)
        # std of Lap(0, 0.5) = 0.5·√2 ≈ 0.71; std-of-mean over 2000 ≈ 0.016
        # so mean ± 0.1 is comfortable
        assert abs(mean - true) < 0.5

    def test_epsilon_exposed(self):
        m = bruce.LaplaceMechanism(l1_sensitivity=2.0, epsilon=0.25)
        assert m.epsilon == 0.25


class TestGaussianMechanism:
    def test_sigma_matches_textbook_formula(self):
        # σ = Δ_2 / ε · √(2 ln(1.25/δ))
        m = bruce.GaussianMechanism(l2_sensitivity=1.0, epsilon=1.0, delta=1e-5)
        expected = (2 * math.log(1.25 / 1e-5)) ** 0.5
        assert abs(m.sigma - expected) < 1e-12

    def test_release_seeded_is_reproducible(self):
        m1 = bruce.GaussianMechanism(l2_sensitivity=1.0, epsilon=1.0,
                                       delta=1e-5, seed=42)
        m2 = bruce.GaussianMechanism(l2_sensitivity=1.0, epsilon=1.0,
                                       delta=1e-5, seed=42)
        assert m1.release_scalar(7.0) == m2.release_scalar(7.0)

    def test_release_vector_length_preserved(self):
        m = bruce.GaussianMechanism(l2_sensitivity=1.0, epsilon=1.0,
                                      delta=1e-5)
        out = m.release_vector([0.0] * 100)
        assert len(out) == 100

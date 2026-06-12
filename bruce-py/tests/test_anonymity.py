"""Tests for bruce.AnonymityGuard."""

from __future__ import annotations

import bruce


class TestAnonymityGuard:
    def test_k_anonymity_allows_large_set(self):
        g = bruce.AnonymityGuard(k=5)
        r = g.evaluate(["a", "b", "c", "d", "e", "f"])
        assert r["status"] == "allow"

    def test_k_anonymity_denies_small_set(self):
        g = bruce.AnonymityGuard(k=5)
        r = g.evaluate(["a", "b", "c"])
        assert r["status"] == "deny_too_few"
        assert r["n"] == 3
        assert r["k"] == 5

    def test_l_diversity_denies_homogeneous(self):
        g = bruce.AnonymityGuard(k=3, l=2)
        # 10 records all = "cancer" → distinct=1, fails l=2
        r = g.evaluate(["cancer"] * 10)
        assert r["status"] == "deny_low_diversity"
        assert r["distinct"] == 1
        assert r["l"] == 2

    def test_l_diversity_allows_mixed(self):
        g = bruce.AnonymityGuard(k=3, l=2)
        r = g.evaluate(["cancer", "diabetes", "cancer", "asthma", "diabetes"])
        assert r["status"] == "allow"

    def test_l_optional_default_disabled(self):
        g = bruce.AnonymityGuard(k=3)
        # all same sensitive value but k passes
        r = g.evaluate(["X", "X", "X", "X"])
        assert r["status"] == "allow"

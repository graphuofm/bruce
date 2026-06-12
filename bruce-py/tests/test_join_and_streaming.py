"""Tests for bruce.hash_join, sort_merge_join, lftj_three, StreamingChainJoin."""

from __future__ import annotations

import bruce


class TestJoins:
    def test_hash_join_basic(self):
        pairs = bruce.hash_join([1, 2, 3, 4], [2, 4, 6, 8])
        assert sorted(pairs) == [(1, 0), (3, 1)]

    def test_sort_merge_join_basic(self):
        pairs = bruce.sort_merge_join([1, 2, 3, 4], [2, 4, 6, 8])
        assert pairs == [(1, 0), (3, 1)]

    def test_sort_merge_handles_duplicates(self):
        pairs = bruce.sort_merge_join([1, 2, 2, 3], [2, 2, 4])
        # 4 cross-product pairs at the matching key
        assert len(pairs) == 4

    def test_lftj_three_triangle(self):
        # all three sequences must have key=5 for a triple
        a = [1, 3, 5, 5, 7]
        b = [2, 5, 6]
        c = [5, 5, 9]
        trips = bruce.lftj_three(a, b, c)
        assert len(trips) == 2 * 1 * 2
        for i, j, k in trips:
            assert a[i] == 5
            assert b[j] == 5
            assert c[k] == 5

    def test_lftj_three_empty_intersection(self):
        # disjoint keys → no triples
        assert bruce.lftj_three([1, 2], [3, 4], [5, 6]) == []


class TestStreamingChainJoin:
    def test_basic_chain_emits_after_all_three(self):
        s = bruce.StreamingChainJoin()
        s.arrive_r("a", "b")
        s.arrive_s("b", "c")
        assert s.n_emitted == 0       # T-tuple missing
        s.arrive_t("c", "d")
        assert s.n_emitted == 1

    def test_emit_count_scales_cubically_on_uniform_chain(self):
        for n in [3, 6, 10]:
            s = bruce.StreamingChainJoin()
            for _ in range(n):
                s.arrive_r("a", "b")
            for _ in range(n):
                s.arrive_s("b", "c")
            for _ in range(n):
                s.arrive_t("c", "d")
            assert s.n_emitted == n * n * n

    def test_order_invariant_on_total_count(self):
        # same tuples in different arrival orders → same total
        a = bruce.StreamingChainJoin()
        a.arrive_r("a", "b")
        a.arrive_s("b", "c")
        a.arrive_t("c", "d")

        b = bruce.StreamingChainJoin()
        b.arrive_t("c", "d")
        b.arrive_r("a", "b")
        b.arrive_s("b", "c")

        assert a.n_emitted == b.n_emitted == 1

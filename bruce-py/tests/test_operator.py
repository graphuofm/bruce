"""Tests for bruce.Operator (F_ε attention)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import bruce


class TestOperatorAttention:
    def test_softmax_attention_at_eps_one_matches_handcalc(self):
        """For x=[1,0], K=[[1,0],[0,1]], V=[[10,0],[0,10]]:
           softmax([1,0]) = [e/(e+1), 1/(e+1)]
           output = [10*e/(e+1), 10/(e+1)] ≈ [7.31, 2.69]"""
        op = bruce.Operator(eps=1.0, sim="dot")
        x = np.array([1.0, 0.0])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[10.0, 0.0], [0.0, 10.0]])
        out = op.attention(x, K, V)
        e = math.e
        assert out[0] == pytest.approx(10 * e / (e + 1), abs=1e-12)
        assert out[1] == pytest.approx(10 / (e + 1), abs=1e-12)

    def test_eps_zero_with_indicator_is_sql_groupby(self):
        """At ε=0 with indicator similarity, sum() = SELECT SUM(v) WHERE k = x."""
        op = bruce.Operator(eps=0.0, sim="indicator")
        x = np.array([1.0, 0.0])
        K = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        V = np.array([[5.0], [7.0], [99.0]])
        out = op.sum(x, K, V)
        assert out[0] == pytest.approx(12.0)

    def test_invalid_eps_raises(self):
        with pytest.raises(ValueError):
            bruce.Operator(eps=-0.5, sim="dot")

    def test_invalid_sim_raises(self):
        with pytest.raises(ValueError):
            bruce.Operator(eps=1.0, sim="cosine_similarity")  # noqa


class TestIncrementalMemory:
    def test_insert_delete_recovers_never_inserted(self):
        """Insert N records, delete one; verify result == compute over N-1."""
        x = np.array([1.0])
        mem = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        for i in range(50):
            mem.insert(f"k{i}",
                       np.array([i * 0.1]),
                       np.array([float(i)]))
        # snapshot output
        before = mem.output().copy()

        # delete entry 7
        mem.delete("k7")
        after = mem.output()

        # recompute from scratch over surviving 49
        mem2 = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        for i in range(50):
            if i == 7:
                continue
            mem2.insert(f"k{i}", np.array([i * 0.1]), np.array([float(i)]))
        ref = mem2.output()

        # delete-after should match never-inserted to machine precision
        assert abs(after[0] - ref[0]) < 1e-12

    def test_len_reflects_alive(self):
        x = np.array([1.0])
        mem = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        assert len(mem) == 0
        for i in range(10):
            mem.insert(f"k{i}", np.array([0.0]), np.array([1.0]))
        assert len(mem) == 10
        mem.delete("k3")
        assert len(mem) == 9

    def test_n_rescales_only_for_max_delete(self):
        """Deleting a non-max entry should NOT trigger a rescale."""
        x = np.array([1.0])
        mem = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        # entry 0 dominates (score 10), others have score ~0
        mem.insert("dom", np.array([10.0]), np.array([1.0]))
        for i in range(10):
            mem.insert(f"k{i}", np.array([0.01]), np.array([float(i)]))
        # delete a non-max entry — no rescale
        mem.delete("k3")
        assert mem.n_rescales == 0
        # now delete the max — one rescale
        mem.delete("dom")
        assert mem.n_rescales >= 1

    def test_duplicate_insert_raises(self):
        x = np.array([1.0])
        mem = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        mem.insert("k", np.array([0.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            mem.insert("k", np.array([0.0]), np.array([1.0]))

    def test_delete_missing_raises(self):
        x = np.array([1.0])
        mem = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        with pytest.raises(ValueError):
            mem.delete("not-there")

    def test_dimension_mismatch_raises(self):
        x = np.array([1.0, 0.0])  # d_k=2
        mem = bruce.IncrementalMemory(query=x, eps=1.0, d_v=1, sim="dot")
        with pytest.raises(ValueError):
            mem.insert("k", np.array([1.0]), np.array([1.0]))  # d_k=1


class TestVersion:
    def test_version_present(self):
        assert hasattr(bruce, "__version__")
        assert isinstance(bruce.__version__, str)
        assert "." in bruce.__version__

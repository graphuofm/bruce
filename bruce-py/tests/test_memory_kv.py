"""Tests for bruce.KvMemory (durable audit-log memory)."""

from __future__ import annotations

import numpy as np
import pytest

import bruce


class TestKvMemory:
    def test_write_then_read_exact(self):
        m = bruce.KvMemory(d_k=2, d_v=2)
        k = np.array([1.0, 0.0]); v = np.array([3.14, 2.72])
        m.write("t1", k, v, owner="alice")
        kk, vv = m.read_exact("t1")
        np.testing.assert_array_equal(kk, k)
        np.testing.assert_array_equal(vv, v)

    def test_delete_owner_enforced(self):
        m = bruce.KvMemory(d_k=2, d_v=1)
        m.write("x", np.array([1.0, 0.0]), np.array([1.0]), owner="alice")
        with pytest.raises(ValueError):
            m.delete("x", owner="mallory")
        m.delete("x", owner="alice")
        assert m.read_exact("x") is None
        assert m.len_alive == 0
        assert m.len_total == 1   # logged but marked deleted

    def test_owner_required_for_overwrite(self):
        m = bruce.KvMemory(d_k=1, d_v=1)
        m.write("y", np.array([1.0]), np.array([1.0]), owner="alice")
        with pytest.raises(ValueError):
            m.write("y", np.array([2.0]), np.array([2.0]), owner="mallory")
        # alice can overwrite
        m.write("y", np.array([2.0]), np.array([2.0]), owner="alice")

    def test_audit_log_grows_with_ops(self):
        m = bruce.KvMemory(d_k=1, d_v=1)
        assert m.audit_log_len == 0
        m.write("a", np.array([0.0]), np.array([1.0]), owner="o")
        m.write("b", np.array([0.0]), np.array([2.0]), owner="o")
        m.delete("a", owner="o")
        assert m.audit_log_len == 3

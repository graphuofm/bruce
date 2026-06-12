"""Tests for the new batch + aggregate-pushdown APIs (overnight fixes
#1/#2/#4)."""
from __future__ import annotations

import numpy as np
import pytest

import bruce


# ---------------------------------------------------------------------
# IncrementalMemory.insert_many / delete_many
# ---------------------------------------------------------------------

def test_insert_many_equivalent_to_per_call():
    rng = np.random.default_rng(0)
    N, d_k, d_v = 200, 8, 3
    x = rng.normal(size=d_k)
    K = rng.normal(size=(N, d_k))
    V = rng.normal(size=(N, d_v))
    ids = [f"k{i}" for i in range(N)]

    mem_loop = bruce.IncrementalMemory(query=x, eps=1.0, d_v=d_v)
    for i in range(N):
        mem_loop.insert(ids[i], K[i], V[i])

    mem_batch = bruce.IncrementalMemory(query=x, eps=1.0, d_v=d_v)
    mem_batch.insert_many(ids, K, V)

    np.testing.assert_allclose(mem_loop.output(), mem_batch.output(),
                                atol=1e-14)


def test_insert_many_rejects_shape_mismatch():
    rng = np.random.default_rng(0)
    mem = bruce.IncrementalMemory(query=np.zeros(4), eps=1.0, d_v=2)
    K = rng.normal(size=(3, 4))
    V = rng.normal(size=(3, 2))
    with pytest.raises(ValueError):
        mem.insert_many(["a", "b"], K, V)        # 2 ids, 3 rows


def test_delete_many_matches_per_call():
    rng = np.random.default_rng(1)
    N, d_k, d_v = 100, 8, 4
    x = rng.normal(size=d_k)
    K = rng.normal(size=(N, d_k))
    V = rng.normal(size=(N, d_v))
    ids = [f"k{i}" for i in range(N)]

    mem_loop = bruce.IncrementalMemory(query=x, eps=1.0, d_v=d_v)
    mem_loop.insert_many(ids, K, V)
    for i in ids[:50]:
        mem_loop.delete(i)

    mem_batch = bruce.IncrementalMemory(query=x, eps=1.0, d_v=d_v)
    mem_batch.insert_many(ids, K, V)
    mem_batch.delete_many(ids[:50])

    np.testing.assert_allclose(mem_loop.output(), mem_batch.output(),
                                atol=1e-14)


# ---------------------------------------------------------------------
# hash_join_indices  (numpy-array variant)
# ---------------------------------------------------------------------

def test_hash_join_indices_matches_python_list():
    L = [1, 2, 3, 4, 5, 1, 2]
    R = [1, 2, 2, 6, 1]
    pairs = bruce.hash_join(L, R)
    li, ri = bruce.hash_join_indices(L, R)
    assert li.dtype == np.int64
    assert ri.dtype == np.int64
    got = sorted(zip(li.tolist(), ri.tolist()))
    expected = sorted(pairs)
    assert got == expected


# ---------------------------------------------------------------------
# hash_join_count
# ---------------------------------------------------------------------

def test_hash_join_count_matches_pair_count():
    L = [1, 2, 3, 4, 5, 1, 2]
    R = [1, 2, 2, 6, 1]
    assert bruce.hash_join_count(L, R) == len(bruce.hash_join(L, R))


def test_hash_join_count_huge_no_materialisation():
    """1 M × 1 M over 100 distinct keys = 10 B pairs.
    `hash_join_count` MUST not materialise these — if it did the test
    process would OOM."""
    rng = np.random.default_rng(0)
    L = rng.integers(0, 100, size=1_000_000).tolist()
    R = rng.integers(0, 100, size=1_000_000).tolist()
    c = bruce.hash_join_count(L, R)
    assert 9_000_000_000 < c < 11_000_000_000


# ---------------------------------------------------------------------
# hash_join_reduce
# ---------------------------------------------------------------------

def test_hash_join_reduce_count_matches():
    L = [1, 2, 3, 4, 5, 1, 2]
    R = [1, 2, 2, 6, 1]
    n = bruce.hash_join_reduce(L, R, "count")
    assert n == bruce.hash_join_count(L, R)


def test_hash_join_reduce_sum_left():
    L = [1, 2, 1]
    R = [1, 2]
    lv = np.array([10.0, 20.0, 30.0])
    # pairs: (0,0),(2,0) on key=1; (1,1) on key=2.
    # sum_left = 10 + 30 + 20 = 60
    s = bruce.hash_join_reduce(L, R, "sum_left", left_values=lv)
    assert s == 60.0


def test_hash_join_reduce_min_max():
    L = [1, 2, 1]
    R = [1, 2]
    lv = np.array([10.0, 20.0, 30.0])
    rv = np.array([100.0, 200.0])
    assert bruce.hash_join_reduce(L, R, "min_left",  left_values=lv)  == 10.0
    assert bruce.hash_join_reduce(L, R, "max_left",  left_values=lv)  == 30.0
    assert bruce.hash_join_reduce(L, R, "min_right", right_values=rv) == 100.0
    assert bruce.hash_join_reduce(L, R, "max_right", right_values=rv) == 200.0


def test_hash_join_reduce_demands_correct_values():
    L = [1, 2]; R = [1, 2]
    with pytest.raises(ValueError):
        bruce.hash_join_reduce(L, R, "sum_left")          # no left_values
    with pytest.raises(ValueError):
        bruce.hash_join_reduce(L, R, "sum_right")         # no right_values
    with pytest.raises(ValueError):
        bruce.hash_join_reduce(L, R, "bogus_agg")

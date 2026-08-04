"""Tests for the minimal eps-algebra frontend (parse -> plan -> execute)."""
import numpy as np
import pytest

from bruce.frontend import SoftAggQuery

SQL = ("SELECT genre, SOFTAVG(rating WEIGHT sim(emb, :q) TEMP {eps}) "
       "FROM movies WHERE year >= 2000 GROUP BY genre")

GK = np.array(["A", "A", "A", "B", "B", "B"])
V = np.array([1.0, 3.0, 8.0, 0.5, 6.0, 2.0])
K = np.array([[0.99, 0.10], [0.95, 0.30], [0.05, 0.99],
              [0.90, 0.40], [0.10, 0.99], [0.70, 0.70]])
X = np.array([1.0, 0.0])
ALL = np.ones(6, bool)


def softmax_ref(eps):
    out = {}
    for g in ("A", "B"):
        m = GK == g
        s = K[m] @ X
        w = np.exp((s - s.max()) / eps)
        out[g] = float((w * V[m]).sum() / w.sum())
    return out


def test_parse_roundtrip():
    q = SoftAggQuery.parse(SQL.format(eps=0.3))
    assert (q.table, q.group_col, q.val_col, q.emb_col) == ("movies", "genre", "rating", "emb")
    assert q.eps == 0.3 and q.filter_col == "year" and q.filter_val == 2000.0


def test_parse_inf_and_no_filter():
    q = SoftAggQuery.parse(
        "SELECT g, SOFTAVG(v WEIGHT sim(e, :q) TEMP inf) FROM t GROUP BY g")
    assert q.eps == float("inf") and q.filter_col is None


def test_parse_rejects_mismatched_group():
    with pytest.raises(ValueError):
        SoftAggQuery.parse(
            "SELECT a, SOFTAVG(v WEIGHT sim(e, :q) TEMP 1) FROM t GROUP BY b")


def test_explain_mentions_pushdown_and_kernel():
    text = SoftAggQuery.parse(SQL.format(eps=0.3)).explain(n_rows=6, n_groups=2)
    assert "grouped_softavg" in text
    assert "pushed-down Filter[eps=0]" in text
    assert "eps=0.3" in text


@pytest.mark.parametrize("eps", [1e-3, 0.3, 1.0])
def test_execute_matches_softmax_reference(eps):
    q = SoftAggQuery.parse(SQL.format(eps=eps))
    labels, ans, _ = q.execute(GK, V, K, X, filter_mask=ALL)
    ref = softmax_ref(eps)
    for g, a in zip(labels, ans):
        assert a == pytest.approx(ref[str(g)], rel=1e-12)


def test_execute_uniform_limit_is_plain_avg():
    q = SoftAggQuery.parse(SQL.format(eps="inf"))
    labels, ans, _ = q.execute(GK, V, K, X, filter_mask=ALL)
    got = dict(zip([str(l) for l in labels], ans))
    assert got["A"] == pytest.approx(4.0)
    assert got["B"] == pytest.approx(8.5 / 3)


def test_filter_mask_is_applied_before_scoring():
    q = SoftAggQuery.parse(SQL.format(eps=1.0))
    mask = np.array([True, True, False, True, True, True])
    labels, ans, _ = q.execute(GK, V, K, X, filter_mask=mask)
    got = dict(zip([str(l) for l in labels], ans))
    s = K[:2] @ X
    w = np.exp((s - s.max()) / 1.0)
    assert got["A"] == pytest.approx(float((w * V[:2]).sum() / w.sum()), rel=1e-12)


def test_fully_filtered_group_is_absent_not_nan():
    """SQL empty-aggregate semantics: a group whose rows are ALL excluded
    by the filter mask must not appear in the output at all (GROUP BY
    over zero rows yields no group), never a NaN/0-division row.
    The exactness comes from the mask, not from an eps -> 0 limit."""
    q = SoftAggQuery.parse(SQL.format(eps=0.1))
    mask = GK == "A"          # every B row filtered out
    labels, ans, _ = q.execute(GK, V, K, X, filter_mask=mask)
    assert list(labels) == ["A"]
    assert np.isfinite(ans).all()


def test_all_rows_filtered_yields_empty_result():
    """Filter excludes every row: the result set is empty, not an error."""
    q = SoftAggQuery.parse(SQL.format(eps=0.1))
    labels, ans, _ = q.execute(GK, V, K, X, filter_mask=np.zeros(6, bool))
    assert len(labels) == 0 and len(ans) == 0


def test_temperature_plan_three_stages_and_contract():
    """One plan: filter -> soft group-agg -> eps=1 read over the winning
    group. The truncated read's measured error must respect the bound
    certified by the omitted weight mass (the explicit error contract
    under which an optimizer may substitute an approximation)."""
    from bruce.frontend import TemperaturePlan
    rng = np.random.RandomState(0)
    n = 400
    gk = np.array(["A", "B", "C", "D"])[rng.randint(0, 4, n)]
    vals = rng.rand(n) * 10
    embs = rng.randn(n, 16)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    x = embs[0]

    plan = TemperaturePlan.parse(SQL.format(eps=0.1), read_top_k=25)
    assert "(1) Filter[eps=0]" in plan.explain()
    assert "(2) GroupSoftAvg[eps=0.1]" in plan.explain()
    assert "(3) AttentionRead[eps=1.0]" in plan.explain()

    res = plan.execute(gk, vals, embs, x, filter_mask=np.ones(n, bool))
    assert res["winning_group"] in "ABCD"
    assert res["measured_err"] <= res["certified_bound"]
    assert 0.0 <= res["omitted_mass"] < 1.0


def test_maintained_plan_delete_updates_both_stages():
    """One delete refreshes stage (2)'s aggregate AND stage (3)'s read;
    both match a from-scratch rebuild to 1e-12."""
    from bruce.frontend import TemperaturePlan, MaintainedPlan
    rng = np.random.RandomState(1)
    n = 300
    gk = np.array(["A", "B"])[rng.randint(0, 2, n)]
    vals = rng.rand(n) * 5
    embs = rng.randn(n, 8)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    x = embs[1]
    ids = [f"r{i}" for i in range(n)]

    plan = TemperaturePlan.parse(SQL.format(eps=0.2))
    res = plan.execute(gk, vals, embs, x, filter_mask=np.ones(n, bool))
    win = res["winning_group"]

    mp = MaintainedPlan(plan, ids, gk, vals, embs, x, winning_group=win)
    victim = ids[int(np.flatnonzero(gk == win)[0])]
    out = mp.delete(victim)

    alive = np.array([i != victim for i in ids]) & (gk == win)
    s = embs[alive] @ x
    w = np.exp((s - s.max()) / 0.2)
    agg_ref = float((w * vals[alive]).sum() / w.sum())
    assert abs(out["agg"] - agg_ref) / abs(agg_ref) < 1e-12

    w3 = np.exp((s - s.max()) / 1.0)
    read_ref = (w3[:, None] * embs[alive]).sum(0) / w3.sum()
    assert np.abs(out["read"] - read_ref).max() < 1e-12

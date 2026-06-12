"""Tests for bruce.masked_attention — the PODS paper's free-connex
"enumerate-then-fold" evaluator — plus the certified-smoothing
utilities eps_star / dequantization_bound.

Verification levels:
  * bit-level vs a numpy dense reference (full N x N mask with -inf),
  * order invariance under pair-stream shuffles (the structure lemma),
  * equivalence with bruce.tree_attention on ancestor masks,
  * the smoothing corollary: |A_eps* - A_0| <= delta on gap-promised
    instances (constants from the paper, sign as fixed in review
    round 2: multiplicity promise is a LOWER bound).
"""
import numpy as np
import pytest

import bruce


RNG = np.random.default_rng(20260612)


def dense_reference(q, k, v, pairs, eps):
    """Numpy reference: dense scores with -inf outside the mask."""
    n_q, n_k = q.shape[0], k.shape[0]
    scores = np.full((n_q, n_k), -np.inf)
    for i, j in pairs:
        scores[i, j] = q[i] @ k[j]
    out = np.zeros((n_q, v.shape[1]))
    covered = np.zeros(n_q, dtype=bool)
    for i in range(n_q):
        row = scores[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        covered[i] = True
        s = row[finite]
        if eps == 0.0:
            m = s.max()
            w = (s == m).astype(float)
            w /= w.sum()
        elif np.isinf(eps):
            w = np.full(s.shape, 1.0 / s.size)
        else:
            e = np.exp((s - s.max()) / eps)
            w = e / e.sum()
        out[i] = w @ v[finite]
    return out, covered


def random_instance(n_q=24, n_k=24, d_k=5, d_v=3):
    q = RNG.normal(size=(n_q, d_k))
    k = RNG.normal(size=(n_k, d_k))
    v = RNG.normal(size=(n_k, d_v))
    return q, k, v


class TestMaskedAttentionIdentities:
    @pytest.mark.parametrize("eps", [0.0, 0.31, 1.0, np.inf])
    def test_window_mask_matches_dense_reference(self, eps):
        q, k, v = random_instance()
        pairs = bruce.window_pairs(24, 5)
        out, cov = bruce.masked_attention(q, k, v, pairs, eps=eps)
        ref, ref_cov = dense_reference(q, k, v, pairs, eps)
        np.testing.assert_allclose(out, ref, atol=1e-12)
        assert list(cov) == list(ref_cov)

    @pytest.mark.parametrize("eps", [0.7, 1.0])
    def test_causal_pairs_match_tree_attention_chain(self, eps):
        # chain-tree ancestor sets ARE the causal mask: the generic
        # evaluator must agree with the specialised tree kernel.
        q, k, v = random_instance(n_q=32, n_k=32)
        out, _ = bruce.masked_attention(
            q, k, v, bruce.causal_pairs(32), eps=eps
        )
        ref = bruce.tree_attention(q, k, v, bruce.chain_tree(32), eps=eps)
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_random_sparse_mask_matches_dense_reference(self):
        q, k, v = random_instance(n_q=30, n_k=18)
        all_pairs = [(i, j) for i in range(30) for j in range(18)]
        sel = RNG.choice(len(all_pairs), size=130, replace=False)
        pairs = np.array([all_pairs[s] for s in sel], dtype=np.int64)
        for eps in (0.0, 0.5, np.inf):
            out, cov = bruce.masked_attention(q, k, v, pairs, eps=eps)
            ref, ref_cov = dense_reference(q, k, v, pairs, eps)
            np.testing.assert_allclose(out, ref, atol=1e-12)
            assert list(cov) == list(ref_cov)


class TestOrderInvariance:
    @pytest.mark.parametrize("eps", [0.0, 0.5, 1.0, np.inf])
    def test_shuffled_stream_gives_same_output(self, eps):
        # The fold is a commutative-monoid homomorphism (structure
        # lemma): enumeration order must not matter.
        q, k, v = random_instance()
        pairs = np.asarray(bruce.window_pairs(24, 7))
        perm = RNG.permutation(pairs.shape[0])
        out_a, _ = bruce.masked_attention(q, k, v, pairs, eps=eps)
        out_b, _ = bruce.masked_attention(q, k, v, pairs[perm], eps=eps)
        np.testing.assert_allclose(out_a, out_b, atol=1e-12)

    def test_parallel_path_agrees_with_small_input(self):
        # > 2^15 pairs exercises the rayon chunk-merge (partition-
        # reduce / Lemma B) path; compare against the dense reference.
        n = 260  # causal pairs: n(n+1)/2 = 33930 > 32768
        q = RNG.normal(size=(n, 4))
        k = RNG.normal(size=(n, 4))
        v = RNG.normal(size=(n, 2))
        pairs = bruce.causal_pairs(n)
        assert np.asarray(pairs).shape[0] > 2 ** 15
        out, _ = bruce.masked_attention(q, k, v, pairs, eps=1.0)
        # spot-check rows against per-row softmax (full dense would be slow)
        for i in (0, 1, 17, 128, 259):
            s = q[i] @ k[: i + 1].T
            e = np.exp(s - s.max())
            ref = (e / e.sum()) @ v[: i + 1]
            np.testing.assert_allclose(out[i], ref, atol=1e-12)


class TestTemperatureRegimes:
    def test_tropical_ties_average_uniformly(self):
        q = np.array([[1.0, 0.0]])
        k = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 5.0]])
        v = np.array([[10.0], [30.0], [999.0]])
        pairs = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
        out, cov = bruce.masked_attention(q, k, v, pairs, eps=0.0)
        assert cov[0]
        assert out[0, 0] == pytest.approx(20.0, abs=1e-12)

    def test_eps_inf_is_plain_mean(self):
        q, k, v = random_instance(n_q=8, n_k=8)
        pairs = bruce.causal_pairs(8)
        out, _ = bruce.masked_attention(q, k, v, pairs, eps=np.inf)
        for i in range(8):
            np.testing.assert_allclose(
                out[i], v[: i + 1].mean(axis=0), atol=1e-12
            )

    def test_uncovered_rows_zero_and_flagged(self):
        q, k, v = random_instance(n_q=4, n_k=4)
        pairs = np.array([[0, 0], [2, 3]], dtype=np.int64)
        out, cov = bruce.masked_attention(q, k, v, pairs, eps=1.0)
        assert list(cov) == [True, False, True, False]
        assert np.all(out[1] == 0.0) and np.all(out[3] == 0.0)


class TestErrors:
    def test_out_of_range_pair_raises(self):
        q, k, v = random_instance(n_q=3, n_k=3)
        with pytest.raises(ValueError):
            bruce.masked_attention(
                q, k, v, np.array([[0, 9]], dtype=np.int64), eps=1.0
            )

    def test_negative_index_raises(self):
        q, k, v = random_instance(n_q=3, n_k=3)
        with pytest.raises(ValueError):
            bruce.masked_attention(
                q, k, v, np.array([[-1, 0]], dtype=np.int64), eps=1.0
            )

    def test_bad_pair_shape_raises(self):
        q, k, v = random_instance(n_q=3, n_k=3)
        with pytest.raises(ValueError):
            bruce.masked_attention(
                q, k, v, np.array([[0, 0, 0]], dtype=np.int64), eps=1.0
            )


class TestCertifiedSmoothing:
    def _gap_instance(self, n=40, kappa=3, gap=2.0, v_max=5.0, d_v=2):
        """Scores with argmax multiplicity exactly kappa and gap >= gap."""
        scores = np.concatenate([
            np.full(kappa, 10.0),
            10.0 - gap - RNG.uniform(0.0, 3.0, size=n - kappa),
        ])
        values = RNG.uniform(-v_max, v_max, size=(n, d_v))
        return scores, values

    def test_eps_star_certifies_delta(self):
        n, kappa, gap, v_max, delta = 40, 3, 2.0, 5.0, 1e-4
        scores, values = self._gap_instance(n, kappa, gap, v_max)
        e = bruce.eps_star(delta, gap, v_max, n, kappa)
        w = np.exp((scores - scores.max()) / e)
        a_eps = (w / w.sum()) @ values
        a_zero = values[:kappa].mean(axis=0)
        assert np.max(np.abs(a_eps - a_zero)) <= delta

    def test_kappa_one_promise_is_always_safe(self):
        # Promise kappa=1 must certify even when the true multiplicity
        # is larger (the bound is monotone in multiplicity — the sign
        # fixed in review round 2).
        n, gap, v_max, delta = 40, 2.0, 5.0, 1e-4
        scores, values = self._gap_instance(n, kappa=4, gap=gap, v_max=v_max)
        e = bruce.eps_star(delta, gap, v_max, n, 1)
        w = np.exp((scores - scores.max()) / e)
        a_eps = (w / w.sum()) @ values
        a_zero = values[:4].mean(axis=0)
        assert np.max(np.abs(a_eps - a_zero)) <= delta

    def test_dequantization_bound_dominates_actual_error(self):
        scores, values = self._gap_instance(n=25, kappa=2, gap=1.0)
        v_max = float(np.max(np.abs(values)))
        a_zero = values[:2].mean(axis=0)
        for eps in (0.05, 0.2, 1.0, 4.0):
            w = np.exp((scores - scores.max()) / eps)
            a_eps = (w / w.sum()) @ values
            b = bruce.dequantization_bound(list(scores), v_max, eps)
            assert np.max(np.abs(a_eps - a_zero)) <= b + 1e-15

    def test_eps_star_rejects_vacuous_inputs(self):
        with pytest.raises(ValueError):
            bruce.eps_star(1e9, 1.0, 1.0, 4, 1)   # delta above trivial bound
        with pytest.raises(ValueError):
            bruce.eps_star(0.1, 1.0, 1.0, 4, 4)   # kappa == n
        with pytest.raises(ValueError):
            bruce.eps_star(-0.1, 1.0, 1.0, 4, 1)  # negative delta

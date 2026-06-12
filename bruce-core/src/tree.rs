//! Tree-structured causal attention — the Bruce identity at work.
//!
//! For each query `Qᵢ`, attention is taken only over `i`'s ancestors
//! in a tree given by `parents[i] = j` (with `-1` for roots). This
//! recovers full causal attention when the tree is a chain, and
//! achieves O(N log N · d) total work when the tree is balanced —
//! the same asymptotic gain the Yannakakis algorithm gives for
//! acyclic conjunctive queries in relational databases.
//!
//! See paper A1 (`paper_A1_yannakakis_tree_attention/`) for the
//! formal correspondence.
//!
//! Complexity for row `i`: O(depth(i) · d). Total: Σᵢ depth(i) · d.
//! Memory: never materialises the N×N mask.
//!
//! We parallelise across rows with rayon — each row's ancestor path
//! is read-only into `K`/`V`, so independent threads are safe.

use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::error::BruceError;
use crate::semiring::softmax_eps;
use crate::types::Eps;

/// Tree-structured causal attention.
///
/// `parents[i] = j`: row `i` attends to `i` itself plus all its
/// ancestors `j, parents[j], …` up to a root (`-1`).
///
/// Self-inclusion makes diagonals consistent with standard causal
/// attention; pass a sentinel `-1` only at roots.
///
/// `eps = 0` is allowed: behaviour matches `Sim::Dot` softmax at the
/// `eps → 0` (tropical) limit — weight on the maximum-score ancestor(s).
///
/// Returns `(N, d_v)`.
pub fn tree_causal_attention(
    q: &ArrayView2<'_, f64>,
    k: &ArrayView2<'_, f64>,
    v: &ArrayView2<'_, f64>,
    parents: &[i64],
    eps: Eps,
) -> Result<Array2<f64>, BruceError> {
    let n = q.nrows();
    if k.nrows() != n || v.nrows() != n || parents.len() != n {
        return Err(BruceError::DimensionMismatch {
            expected: n,
            got: k.nrows().min(v.nrows()).min(parents.len()),
        });
    }
    if q.ncols() != k.ncols() {
        return Err(BruceError::DimensionMismatch {
            expected: q.ncols(),
            got: k.ncols(),
        });
    }
    // Validate parent indices: every parents[i] in {-1} ∪ {0..i}.
    // Strict topological ordering (j < i for j = parents[i]) is what makes
    // an "ancestor walk" terminate; we check it up front rather than risking
    // an infinite loop or panic in the parallel section.
    for (i, &p) in parents.iter().enumerate() {
        if p < -1 || p >= i as i64 {
            return Err(BruceError::InvalidArgument(format!(
                "parents[{i}] = {p}; must be in [-1, {i})",
            )));
        }
    }

    let d_v = v.ncols();
    let mut out = Array2::<f64>::zeros((n, d_v));

    // Per-row work is independent → parallel rows.
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            // Walk ancestor path: i, parents[i], parents[parents[i]], ...
            // Bounded by depth(i) ≤ N, terminates because we enforced p < i.
            let mut path = Vec::<usize>::with_capacity(16);
            let mut j = i as i64;
            while j != -1 {
                path.push(j as usize);
                j = parents[j as usize];
            }

            // scores[idx] = q[i] · k[path[idx]]   (dot similarity)
            let qi = q.row(i);
            let mut scores = Vec::<f64>::with_capacity(path.len());
            for &p in &path {
                scores.push(qi.dot(&k.row(p)));
            }
            let weights = softmax_eps(&scores, eps);

            // out[i] = Σ wₛ · v[path[s]]
            for (s, &w) in weights.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let vp = v.row(path[s]);
                for c in 0..d_v {
                    row[c] += w * vp[c];
                }
            }
        });

    Ok(out)
}

// ============================================================================
// Tree builders — mirror the ones in paper_A1 src/algorithms/.
// ============================================================================

/// `parents[i] = i - 1` for i ≥ 1, `parents[0] = -1`. Recovers
/// standard causal (lower-triangular) attention; O(N²·d) total work.
pub fn chain_tree(n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| if i == 0 { -1 } else { i as i64 - 1 })
        .collect()
}

/// Heap-like balanced binary tree: `parents[i] = (i - 1) / 2`,
/// `parents[0] = -1`. Depth log₂N → O(N log N · d) total work.
pub fn balanced_binary_tree(n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| if i == 0 { -1 } else { ((i - 1) / 2) as i64 })
        .collect()
}

/// k-ary balanced tree: `parents[i] = (i - 1) / k`.
pub fn k_ary_balanced_tree(n: usize, k: usize) -> Vec<i64> {
    assert!(k >= 1, "k-ary tree needs k ≥ 1");
    (0..n)
        .map(|i| if i == 0 { -1 } else { ((i - 1) / k) as i64 })
        .collect()
}

/// Star: one root with N-1 leaves. Depth 1; O(N·d) total work but
/// most parallelism wasted because every leaf has the same parent.
pub fn star_tree(n: usize) -> Vec<i64> {
    (0..n).map(|i| if i == 0 { -1 } else { 0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn star_at_eps_one_matches_dense_softmax() {
        // 4 rows; row 0 is root; rows 1,2,3 all point to row 0.
        // For row i in {1,2,3}, path = [i, 0], scores = [qᵢ·kᵢ, qᵢ·k₀].
        let q = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]];
        let k = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]];
        let v = array![[10.0], [20.0], [30.0], [40.0]];
        let parents = star_tree(4);
        let out =
            tree_causal_attention(&q.view(), &k.view(), &v.view(), &parents, Eps::ONE).unwrap();
        // row 0 attends only to itself
        assert_abs_diff_eq!(out[(0, 0)], 10.0, epsilon = 1e-12);
        // row 1: path=[1,0], scores=[k₁·q₁, k₀·q₁]=[1.0, 0.0]
        // weights ∝ [e, 1]; out = (e·20 + 1·10)/(e+1)
        let e = std::f64::consts::E;
        assert_abs_diff_eq!(out[(1, 0)], (e * 20.0 + 10.0) / (e + 1.0), epsilon = 1e-12);
    }

    #[test]
    fn chain_recovers_causal_attention() {
        // chain_tree on N=3 is a strict causal tree:
        // row 0 sees {0}; row 1 sees {1,0}; row 2 sees {2,1,0}.
        let q = array![[1.0], [2.0], [3.0]];
        let k = array![[1.0], [1.0], [1.0]];
        let v = array![[10.0], [20.0], [30.0]];
        let parents = chain_tree(3);
        let out =
            tree_causal_attention(&q.view(), &k.view(), &v.view(), &parents, Eps::ONE).unwrap();
        // row 0 sees only itself
        assert_abs_diff_eq!(out[(0, 0)], 10.0, epsilon = 1e-12);
        // row 1: scores=[2, 2]; uniform weights → (20+10)/2 = 15
        assert_abs_diff_eq!(out[(1, 0)], 15.0, epsilon = 1e-12);
        // row 2: scores=[3, 3, 3]; uniform → 20
        assert_abs_diff_eq!(out[(2, 0)], 20.0, epsilon = 1e-12);
    }

    #[test]
    fn rejects_forward_parent() {
        // parents[1] = 2 is forbidden — would loop / can't topologically sort
        let q = array![[1.0], [1.0], [1.0]];
        let k = q.clone();
        let v = q.clone();
        let parents = vec![-1i64, 2, -1];
        let r = tree_causal_attention(&q.view(), &k.view(), &v.view(), &parents, Eps::ONE);
        assert!(r.is_err());
    }

    #[test]
    fn balanced_binary_depth_is_log() {
        let parents = balanced_binary_tree(16);
        // depth(15) walks 15→7→3→1→0 = 4 hops + self ⇒ path length 5 ≈ log₂(16)+1
        let mut j = 15i64;
        let mut depth = 0;
        while j != -1 {
            j = parents[j as usize];
            depth += 1;
        }
        assert_eq!(depth, 5);
    }
}

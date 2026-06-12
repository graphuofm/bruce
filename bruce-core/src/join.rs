//! Equi-join primitives.
//!
//! Bruce's ε = 0 path needs efficient equi-join because that is what
//! the indicator semiring reduces to. We provide:
//!
//! * `hash_join` — O(N + M) two-table inner join
//! * `sort_merge_join` — O(N log N) join when both sides are sorted
//! * `lftj` — Leapfrog Triejoin for k-way joins (Veldhuizen 2014)
//!
//! These are NOT a replacement for DuckDB / Postgres on production SQL
//! workloads. They are the **algebraic backbone** Bruce uses inside
//! the F_ε operator, callable from both the Rust crate and Python
//! bindings.

use ahash::AHashMap;
use rayon::prelude::*;
use std::hash::Hash;

const PROBE_PARALLEL_THRESHOLD: usize = 4096;

/// Inner hash join over two key columns. Returns `(left_idx, right_idx)`
/// pairs. O(|L| + |R|). The probe phase is parallelised with rayon
/// when `|L| >= PROBE_PARALLEL_THRESHOLD` (build stays serial because
/// AHashMap insertion contention costs more than the speed-up).
pub fn hash_join<K: Eq + Hash + Clone + Sync>(
    left_keys: &[K],
    right_keys: &[K],
) -> Vec<(usize, usize)> {
    let mut idx: AHashMap<K, Vec<usize>> = AHashMap::new();
    for (j, k) in right_keys.iter().enumerate() {
        idx.entry(k.clone()).or_default().push(j);
    }
    if left_keys.len() < PROBE_PARALLEL_THRESHOLD {
        let mut out = Vec::new();
        for (i, k) in left_keys.iter().enumerate() {
            if let Some(matches) = idx.get(k) {
                for &j in matches {
                    out.push((i, j));
                }
            }
        }
        out
    } else {
        // Parallel probe: each thread builds its own local Vec, then
        // we flat-concat. Cheaper than locked global Vec.
        left_keys
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, k)| {
                idx.get(k)
                    .into_iter()
                    .flat_map(move |matches| matches.iter().map(move |&j| (i, j)))
            })
            .collect()
    }
}

/// Sort-merge join over pre-sorted key columns. O(|L| + |R|).
/// Both inputs MUST be sorted ascending.
pub fn sort_merge_join<K: Ord + Clone>(left_keys: &[K], right_keys: &[K]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left_keys.len() && j < right_keys.len() {
        match left_keys[i].cmp(&right_keys[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                // emit all (i, j') with right_keys[j'] == current
                let k = &left_keys[i];
                let mut j2 = j;
                while j2 < right_keys.len() && &right_keys[j2] == k {
                    out.push((i, j2));
                    j2 += 1;
                }
                i += 1;
            }
        }
    }
    out
}

/// Three-way Leapfrog Triejoin over sorted iterators. Returns triples
/// of indices `(i, j, k)` such that `a[i] == b[j] == c[k]`.
///
/// LFTJ achieves the AGM-optimal `O(N^{ρ*})` complexity for triangle
/// queries (ρ* = 3/2 for the symmetric triangle).
pub fn lftj_three<K: Ord + Clone>(a: &[K], b: &[K], c: &[K]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < a.len() && j < b.len() && k < c.len() {
        // find the max key among the three current cursors
        let ka = &a[i];
        let kb = &b[j];
        let kc = &c[k];
        if ka == kb && kb == kc {
            // emit all triples matching this key
            let key = ka.clone();
            let i_end = (i..a.len()).take_while(|&x| a[x] == key).count();
            let j_end = (j..b.len()).take_while(|&x| b[x] == key).count();
            let k_end = (k..c.len()).take_while(|&x| c[x] == key).count();
            for ii in i..i + i_end {
                for jj in j..j + j_end {
                    for kk in k..k + k_end {
                        out.push((ii, jj, kk));
                    }
                }
            }
            i += i_end;
            j += j_end;
            k += k_end;
            continue;
        }
        // advance the smallest cursor to >= max of the three
        let max_key = ka.max(kb).max(kc).clone();
        while i < a.len() && a[i] < max_key {
            i += 1;
        }
        while j < b.len() && b[j] < max_key {
            j += 1;
        }
        while k < c.len() && c[k] < max_key {
            k += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_join_basic() {
        let l = vec![1, 2, 3, 4];
        let r = vec![2, 4, 6, 8];
        let mut pairs = hash_join(&l, &r);
        pairs.sort();
        assert_eq!(pairs, vec![(1, 0), (3, 1)]);
    }

    #[test]
    fn sort_merge_join_basic() {
        let l = vec![1, 2, 3, 4];
        let r = vec![2, 4, 6, 8];
        let pairs = sort_merge_join(&l, &r);
        assert_eq!(pairs, vec![(1, 0), (3, 1)]);
    }

    #[test]
    fn sort_merge_with_duplicates() {
        let l = vec![1, 2, 2, 3];
        let r = vec![2, 2, 4];
        let mut pairs = sort_merge_join(&l, &r);
        pairs.sort();
        // (1,0) (1,1) (2,0) (2,1)
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn hash_join_parallel_path_correctness() {
        // PARALLEL-002b: above PROBE_PARALLEL_THRESHOLD the rayon path
        // kicks in. Verify it produces the same pair set as the serial
        // path. Use 10K left × 100 right with 50% overlap, then
        // sort-and-compare against a fresh serial reference.
        let n_left = 10_000;
        let n_right = 100;
        let left: Vec<i64> = (0..n_left as i64).map(|i| i % 200).collect();
        let right: Vec<i64> = (0..n_right as i64).map(|i| i % 200).collect();
        assert!(
            left.len() >= PROBE_PARALLEL_THRESHOLD,
            "test relies on left exceeding parallel threshold"
        );
        let mut par_pairs = hash_join(&left, &right);
        par_pairs.sort();
        // serial reference: build then probe in-loop
        let mut idx: AHashMap<i64, Vec<usize>> = AHashMap::new();
        for (j, k) in right.iter().enumerate() {
            idx.entry(*k).or_default().push(j);
        }
        let mut ser_pairs = Vec::new();
        for (i, k) in left.iter().enumerate() {
            if let Some(matches) = idx.get(k) {
                for &j in matches {
                    ser_pairs.push((i, j));
                }
            }
        }
        ser_pairs.sort();
        assert_eq!(
            par_pairs.len(),
            ser_pairs.len(),
            "parallel path returns wrong pair count"
        );
        assert_eq!(
            par_pairs, ser_pairs,
            "parallel and serial paths disagree on pair set"
        );
    }

    #[test]
    fn lftj_three_triangle() {
        // simple triangle: every k = 5 in all three
        let a = vec![1, 3, 5, 5, 7];
        let b = vec![2, 5, 6];
        let c = vec![5, 5, 9];
        let trips = lftj_three(&a, &b, &c);
        // a has 5 at i=2,3; b has 5 at j=1; c has 5 at k=0,1
        // expected pairs: 2 × 1 × 2 = 4 triples
        assert_eq!(trips.len(), 4);
        for (i, j, k) in trips {
            assert_eq!(a[i], 5);
            assert_eq!(b[j], 5);
            assert_eq!(c[k], 5);
        }
    }
}

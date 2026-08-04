//! HNSW access path for the F_ε operator family (backlog #6).
//!
//! A multi-layer navigable-small-world graph over `f32` key vectors,
//! used as an **access path** for sharp-ε (top-k-shaped) reads: it
//! accelerates candidate enumeration when the optimizer decides a
//! query is selective enough, while the fused scan
//! ([`crate::mask::grouped_softavg`]) remains the dense path — the
//! optimizer chooses between them.
//!
//! ## Why our own implementation (and not hnswlib/usearch bindings)
//!
//! Two things must live INSIDE the neighbor-expansion loop, not
//! around it:
//!
//! 1. **Predicate-aware traversal** (ACORN-flavored): a relational
//!    filter must be consulted per candidate *during* beam expansion —
//!    non-matching nodes are still traversed as routing waypoints but
//!    never enter results, and the beam widens adaptively so selective
//!    predicates keep recall. Post-filtering an unmodified library's
//!    top-k cannot do this (it starves under selective predicates),
//!    and pre-filtering rebuilds the index per predicate.
//! 2. **Delete-bitmap cooperation**: tombstoned rows must be excluded
//!    from results *immediately* (CRUD Lemma A semantics) while
//!    remaining routable, which requires the tombstone check at the
//!    exact point a candidate would be accepted.
//!
//! Bindings expose neither hook; both are one-line checks here.
//!
//! ## Similarity
//!
//! Similarity is the **dot product** (higher = better). Our embeddings
//! are L2-normalized, so dot == cosine. Caveat for future unnormalized
//! use: dot-product ("MIPS") search over unnormalized vectors breaks
//! the approximate triangle-inequality assumptions navigable graphs
//! rely on; a reduction (e.g. the extra-dimension transform) or a
//! norm-aware edge selection would be needed. v1 documents, not solves,
//! this.
//!
//! ## Filter semantics (ACORN-flavored)
//!
//! `search(..., filter)` treats the filter as a *result admission*
//! predicate only: non-matching nodes are traversed and routed through,
//! but never returned. When a filter is present the beam keeps
//! expanding until `ef` ACCEPTED candidates are found (the usual
//! termination test then applies), subject to an expansion budget of
//! `ef * 8` node pops — and the budget is never enforced before at
//! least `k` accepted results exist. Consequence: any predicate with
//! at least `k` matches in the entry point's connected component
//! yields `k` results.
//!
//! ## Delete contract: routable-until-compact
//!
//! [`HnswIndex::delete`] sets a tombstone bit. The node is excluded
//! from all future RESULTS immediately, but stays in the graph as a
//! routing waypoint (removing it would degrade or disconnect the
//! graph). v1 performs **no graph repair**; construction may even link
//! new nodes to tombstones (harmless: they still route). Callers poll
//! [`HnswIndex::tombstone_fraction`] and rebuild the index when it
//! grows past their threshold (a rebuild from the live set is the
//! compaction story in v1). Re-inserting a tombstoned id is a
//! `DuplicateKey` error until such a rebuild.
//!
//! ## Determinism
//!
//! Identical insert sequences produce bit-identical graphs and search
//! results. There is no RNG: level assignment is `splitmix64(id)`
//! mapped to a geometric distribution with continuation ratio `1/e`
//! (`P(level >= l) = e^-l`), and every heap/pruning tie breaks on the
//! lower internal node index. Hash maps are used only for id lookup,
//! never iterated.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ahash::AHashMap;

use crate::error::{BruceError, Result};

/// Hard cap on layer levels (P(level >= 24) = e^-24 ≈ 4e-11; the cap
/// only guards against adversarial ids).
const LEVEL_CAP: usize = 24;

/// Beam-widening budget factor under a filter: expansion stops after
/// `ef * WIDEN_FACTOR` node pops once at least `k` accepted results
/// exist (never before).
const WIDEN_FACTOR: usize = 8;

#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic level for an external id: floor of an Exp(1) sample
/// derived from `splitmix64(id)`, i.e. geometric with continuation
/// ratio 1/e per level.
#[inline]
fn level_for(id: u32) -> usize {
    let h = splitmix64(u64::from(id));
    // 53 high bits -> u in (0, 1]; never 0, so ln is finite.
    let u = ((h >> 11) as f64 + 1.0) * (1.0 / 9_007_199_254_740_992.0);
    ((-u.ln()).floor() as usize).min(LEVEL_CAP)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// A scored candidate. `Ord` is "greater = better": higher score wins,
/// ties break toward the LOWER internal node index (deterministic).
#[derive(Clone, Copy, Debug)]
struct Cand {
    score: f32,
    node: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build-time neighbor-selection policy — the one knob that moves the
/// recall-vs-`ef` curve on hard (unstructured) data.
///
/// Applies at BUILD time only; `search` is identical under both. It is
/// therefore a property of the graph, not of a query: switching it
/// requires a rebuild.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NeighborSelection {
    /// v1 default: keep the `m` best-scoring candidates from the
    /// construction beam. Simple, and the policy every measured number
    /// in `tests/hnsw_search.rs` and m6_hnsw/README.md was taken under.
    #[default]
    KeepBest,
    /// Malkov & Yashunin's Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC),
    /// without `extendCandidates` and without `keepPrunedConnections`.
    ///
    /// Walk the candidates best-first and admit `c` only if it is
    /// closer to the base node than to every already-admitted
    /// neighbor `r`:
    ///
    ///   admit c  <=>  for all r in result: dot(c, r) <= dot(c, base)
    ///
    /// (dot-product form of the paper's distance test; our keys are
    /// L2-normalized, so this is the cosine test.) The effect is to
    /// spend the degree budget on DIRECTIONS rather than on a tight
    /// cluster of near-duplicates, which is what gives long-range
    /// links on data with no exploitable structure.
    Diversity,
}

/// Hierarchical navigable-small-world index over `f32` key vectors,
/// similarity = dot product (see module docs for the normalization
/// assumption and the filter/tombstone contracts).
#[derive(Clone, Debug)]
pub struct HnswIndex {
    d: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    selection: NeighborSelection,
    /// Flat key storage, node i occupies `[i*d, (i+1)*d)`.
    keys: Vec<f32>,
    /// External id per internal node index.
    ids: Vec<u32>,
    /// `links[node][layer]` = neighbor internal node indices.
    links: Vec<Vec<Vec<u32>>>,
    /// Tombstone bitmap (see module docs: routable-until-compact).
    deleted: Vec<bool>,
    deleted_count: usize,
    id_to_node: AHashMap<u32, u32>,
    entry: Option<u32>,
    top_level: usize,
}

impl HnswIndex {
    /// New empty index over `d`-dimensional keys. `m` is the max
    /// degree on upper layers (layer 0 allows `2*m`); `ef_construction`
    /// is the construction-time beam width. Values are clamped to
    /// `m >= 2`, `ef_construction >= 1`.
    pub fn new(d: usize, m: usize, ef_construction: usize) -> Self {
        let m = m.max(2);
        HnswIndex {
            d,
            m,
            m0: 2 * m,
            ef_construction: ef_construction.max(1),
            selection: NeighborSelection::KeepBest,
            keys: Vec::new(),
            ids: Vec::new(),
            links: Vec::new(),
            deleted: Vec::new(),
            deleted_count: 0,
            id_to_node: AHashMap::new(),
            entry: None,
            top_level: 0,
        }
    }

    /// [`HnswIndex::new`] with the paper defaults: `m = 16`,
    /// `ef_construction = 128`.
    pub fn with_defaults(d: usize) -> Self {
        Self::new(d, 16, 128)
    }

    /// Set the build-time neighbor-selection policy (default
    /// [`NeighborSelection::KeepBest`]). Must be set BEFORE the first
    /// insert — it shapes the graph as it is built, so calling it on a
    /// populated index leaves a graph that is half one policy and half
    /// the other. Returns `Err(BruceError::InvalidArgument)` in that
    /// case rather than silently producing a hybrid.
    pub fn set_selection(&mut self, s: NeighborSelection) -> Result<()> {
        if !self.ids.is_empty() {
            return Err(BruceError::InvalidArgument(
                "neighbor selection must be set before the first insert \
                 (it shapes the graph at build time; rebuild to change it)"
                    .into(),
            ));
        }
        self.selection = s;
        Ok(())
    }

    /// The build-time neighbor-selection policy in force.
    pub fn selection(&self) -> NeighborSelection {
        self.selection
    }

    /// Apply the neighbor-selection policy to `cands` (sorted
    /// best-first for `base`), returning at most `cap` node indices in
    /// the same order.
    ///
    /// [`NeighborSelection::KeepBest`] truncates. [`NeighborSelection::
    /// Diversity`] runs Malkov-Yashunin Algorithm 4: admit `c` only
    /// when no already-admitted `r` is closer to `c` than `base` is.
    /// Both are deterministic (the input order is, and the test is a
    /// total comparison on `f32`); ties (`dot(c, r) == dot(c, base)`)
    /// ADMIT, so a duplicate vector is kept rather than pruned — an
    /// exact duplicate of the base scores identically against
    /// everything, and dropping it would lose a live row's only link.
    fn select_neighbors(&self, cands: &[Cand], base: u32, cap: usize) -> Vec<u32> {
        match self.selection {
            NeighborSelection::KeepBest => cands.iter().take(cap).map(|c| c.node).collect(),
            NeighborSelection::Diversity => {
                let mut out: Vec<u32> = Vec::with_capacity(cap.min(cands.len()));
                if cands.len() <= cap {
                    // nothing to spend: the paper returns all of them
                    return cands.iter().map(|c| c.node).collect();
                }
                let base_key = self.key(base);
                for c in cands {
                    if out.len() >= cap {
                        break;
                    }
                    if c.node == base {
                        continue; // never self-link
                    }
                    let c_key = self.key(c.node);
                    let to_base = dot(c_key, base_key);
                    let dominated = out.iter().any(|&r| dot(c_key, self.key(r)) > to_base);
                    if !dominated {
                        out.push(c.node);
                    }
                }
                out
            }
        }
    }

    /// Key dimension fixed at construction.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Number of LIVE (non-tombstoned) vectors.
    pub fn len(&self) -> usize {
        self.ids.len() - self.deleted_count
    }

    /// True when no live vector remains.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fraction of ever-inserted nodes that are tombstoned, in
    /// `[0, 1]`. Callers use this to decide when to rebuild (compact)
    /// the index; v1 has no in-place graph repair.
    pub fn tombstone_fraction(&self) -> f64 {
        if self.ids.is_empty() {
            0.0
        } else {
            self.deleted_count as f64 / self.ids.len() as f64
        }
    }

    /// Rough accounting of heap memory held by the index, in bytes
    /// (key storage, adjacency lists incl. per-`Vec` headers, id maps).
    pub fn memory_bytes(&self) -> usize {
        let mut b = self.keys.capacity() * 4
            + self.ids.capacity() * 4
            + self.deleted.capacity()
            + self.id_to_node.len() * 16; // entry + bucket overhead, rough
        for layers in &self.links {
            b += 24 + layers.capacity() * 24;
            for l in layers {
                b += l.capacity() * 4;
            }
        }
        b
    }

    #[inline]
    fn key(&self, node: u32) -> &[f32] {
        let n = node as usize;
        &self.keys[n * self.d..(n + 1) * self.d]
    }

    #[inline]
    fn score(&self, q: &[f32], node: u32) -> f32 {
        dot(q, self.key(node))
    }

    /// Insert `(id, key)`. Errors: [`BruceError::DimensionMismatch`]
    /// when `key.len() != d`; [`BruceError::DuplicateKey`] when `id`
    /// was inserted before (even if since tombstoned — see module
    /// docs). Standard HNSW insert: greedy descent to the node's
    /// level, beam of `ef_construction` per layer, keep-`m`-best
    /// neighbor selection, backlink pruning to `m` (`2m` at layer 0).
    pub fn insert(&mut self, id: u32, key: &[f32]) -> Result<()> {
        if key.len() != self.d {
            return Err(BruceError::DimensionMismatch {
                expected: self.d,
                got: key.len(),
            });
        }
        if self.id_to_node.contains_key(&id) {
            return Err(BruceError::DuplicateKey(id.to_string()));
        }

        let level = level_for(id);
        let node = self.ids.len() as u32;
        self.keys.extend_from_slice(key);
        self.ids.push(id);
        self.deleted.push(false);
        self.links.push(vec![Vec::new(); level + 1]);
        self.id_to_node.insert(id, node);

        let Some(ep) = self.entry else {
            self.entry = Some(node);
            self.top_level = level;
            return Ok(());
        };

        let mut cur = Cand {
            score: self.score(key, ep),
            node: ep,
        };
        if self.top_level > level {
            cur = self.greedy_at_layers(key, cur, self.top_level, level + 1);
        }

        let start_layer = level.min(self.top_level);
        for layer in (0..=start_layer).rev() {
            // Construction accepts every node — tombstones included —
            // because the beam is about graph geometry, not results.
            let beam = self.search_layer(
                key,
                cur,
                layer,
                self.ef_construction,
                &|_| true,
                0,
                usize::MAX,
            );
            let selected: Vec<u32> = self.select_neighbors(&beam, node, self.m);
            self.links[node as usize][layer] = selected.clone();
            let cap = if layer == 0 { self.m0 } else { self.m };
            for &nb in &selected {
                self.links[nb as usize][layer].push(node);
                if self.links[nb as usize][layer].len() > cap {
                    let list = std::mem::take(&mut self.links[nb as usize][layer]);
                    let mut scored: Vec<Cand> = list
                        .into_iter()
                        .map(|x| Cand {
                            score: dot(self.key(nb), self.key(x)),
                            node: x,
                        })
                        .collect();
                    scored.sort_by(|a, b| b.cmp(a));
                    // Same policy on the BACKLINK prune (Algorithm 4 is
                    // applied at both call sites in the paper): under
                    // KeepBest this is the old truncate, under
                    // Diversity it keeps `nb`'s links spread out.
                    let kept = self.select_neighbors(&scored, nb, cap);
                    self.links[nb as usize][layer] = kept;
                }
            }
            if let Some(best) = beam.first() {
                cur = *best;
            }
        }

        if level > self.top_level {
            self.top_level = level;
            self.entry = Some(node);
        }
        Ok(())
    }

    /// Top-`k` by dot-product score, descending; `(external id, score)`
    /// pairs. `ef >= k` is the layer-0 beam width (effective width is
    /// `max(ef, k)`). `filter`, if present, admits results by EXTERNAL
    /// id under the ACORN-flavored semantics in the module docs.
    /// Errors: [`BruceError::DimensionMismatch`] when
    /// `query.len() != d`. An empty index returns `Ok(vec![])`; when
    /// fewer than `k` live nodes match, all reachable matches are
    /// returned (see module docs for the widening guarantee).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
    ) -> Result<Vec<(u32, f32)>> {
        if query.len() != self.d {
            return Err(BruceError::DimensionMismatch {
                expected: self.d,
                got: query.len(),
            });
        }
        let Some(ep) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        let ef_target = ef.max(k).max(1);
        let mut cur = Cand {
            score: self.score(query, ep),
            node: ep,
        };
        if self.top_level >= 1 {
            cur = self.greedy_at_layers(query, cur, self.top_level, 1);
        }

        let budget = if filter.is_some() {
            ef_target.saturating_mul(WIDEN_FACTOR)
        } else {
            usize::MAX
        };
        // `match` rather than `Option::is_none_or` (stable only since
        // Rust 1.82; workspace MSRV is 1.81).
        let accept = |n: u32| {
            if self.deleted[n as usize] {
                return false;
            }
            match filter {
                Some(f) => f(self.ids[n as usize]),
                None => true,
            }
        };
        let mut acc = self.search_layer(query, cur, 0, ef_target, &accept, k, budget);
        acc.truncate(k);
        Ok(acc
            .into_iter()
            .map(|c| (self.ids[c.node as usize], c.score))
            .collect())
    }

    /// Tombstone `id`: excluded from all future results immediately,
    /// still routable until the index is rebuilt (module docs).
    /// Errors: [`BruceError::KeyNotFound`] when `id` was never
    /// inserted or is already tombstoned.
    pub fn delete(&mut self, id: u32) -> Result<()> {
        match self.id_to_node.get(&id) {
            Some(&n) if !self.deleted[n as usize] => {
                self.deleted[n as usize] = true;
                self.deleted_count += 1;
                Ok(())
            }
            _ => Err(BruceError::KeyNotFound(id.to_string())),
        }
    }

    /// Greedy hill-climb from `from_layer` down to `to_layer`
    /// (inclusive), strict improvement only (deterministic given the
    /// deterministic link order).
    fn greedy_at_layers(
        &self,
        q: &[f32],
        mut cur: Cand,
        from_layer: usize,
        to_layer: usize,
    ) -> Cand {
        debug_assert!(from_layer >= to_layer);
        let mut layer = from_layer;
        loop {
            let mut improved = true;
            while improved {
                improved = false;
                let layers = &self.links[cur.node as usize];
                if layer >= layers.len() {
                    break;
                }
                for &nb in &layers[layer] {
                    let s = self.score(q, nb);
                    if s > cur.score {
                        cur = Cand { score: s, node: nb };
                        improved = true;
                    }
                }
            }
            if layer == to_layer {
                break;
            }
            layer -= 1;
        }
        cur
    }

    /// Best-first beam over one layer. `accept` (internal-node
    /// predicate) gates RESULT admission only — rejected nodes still
    /// route. Termination: frontier's best is below the worst of
    /// `ef_target` accepted results, or the `budget` (in node pops) is
    /// exhausted AND at least `k_floor` results were accepted.
    /// Returns accepted candidates sorted best-first.
    #[allow(clippy::too_many_arguments)] // private beam kernel: ef/k_floor/budget are one knob set
    fn search_layer(
        &self,
        q: &[f32],
        entry: Cand,
        layer: usize,
        ef_target: usize,
        accept: &dyn Fn(u32) -> bool,
        k_floor: usize,
        budget: usize,
    ) -> Vec<Cand> {
        let n = self.ids.len();
        let mut visited = vec![false; n];
        let mut frontier: BinaryHeap<Cand> = BinaryHeap::new(); // max-heap
        let mut accepted: BinaryHeap<Reverse<Cand>> = BinaryHeap::new(); // min-heap

        visited[entry.node as usize] = true;
        frontier.push(entry);
        if accept(entry.node) {
            accepted.push(Reverse(entry));
        }

        let mut pops: usize = 0;
        while let Some(c) = frontier.pop() {
            if accepted.len() >= ef_target {
                let worst = accepted.peek().expect("non-empty").0.score;
                if c.score < worst {
                    break;
                }
            }
            if pops >= budget && accepted.len() >= k_floor {
                break;
            }
            pops += 1;

            let layers = &self.links[c.node as usize];
            if layer >= layers.len() {
                continue;
            }
            for &nb in &layers[layer] {
                let nbi = nb as usize;
                if visited[nbi] {
                    continue;
                }
                visited[nbi] = true;
                let s = self.score(q, nb);
                let full = accepted.len() >= ef_target;
                let admit = if full {
                    s > accepted.peek().expect("non-empty").0.score
                } else {
                    true
                };
                if admit {
                    let cand = Cand { score: s, node: nb };
                    frontier.push(cand);
                    if accept(nb) {
                        accepted.push(Reverse(cand));
                        if accepted.len() > ef_target {
                            accepted.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<Cand> = accepted.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    }
}

#[cfg(test)]
mod tests {
    //! Build-time neighbor selection (2026-08-03, hnsw-finish track).
    //! The end-to-end recall effect is MEASURED, not asserted, in
    //! paper_sigmod_bruce/experiments/m6_hnsw/diversity_bench; these
    //! tests pin the policy's mechanics and its lifecycle contract.

    use super::*;

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    /// xorshift64* in [-0.5, 0.5), the numerical_edges convention.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        }
    }

    fn random_unit(rng: &mut Rng, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= n;
        }
        v
    }

    /// Algorithm 4's whole point: a candidate closer to an already
    /// admitted neighbour than to the base is pruned, so the degree
    /// budget buys DIRECTIONS instead of a cluster of near-duplicates.
    /// Same candidate set, same cap, two policies, two answers.
    #[test]
    fn diversity_prunes_near_duplicates_where_keep_best_does_not() {
        let base = unit([1.0, 0.0, 0.0]);
        let c1 = unit([0.92, 0.3919, 0.0]);
        let c2 = unit([0.91, 0.4146, 0.0]); // near-duplicate of c1
        let c3 = unit([0.90, -0.4359, 0.0]); // the opposite direction
        let build = |sel: NeighborSelection| {
            let mut ix = HnswIndex::new(3, 16, 32);
            ix.set_selection(sel).expect("empty index");
            for (i, k) in [base, c1, c2, c3].iter().enumerate() {
                ix.insert(i as u32, k).expect("insert");
            }
            ix
        };
        for sel in [NeighborSelection::KeepBest, NeighborSelection::Diversity] {
            let ix = build(sel);
            // candidates for base (node 0), already best-first
            let cands: Vec<Cand> = [1u32, 2, 3]
                .iter()
                .map(|&n| Cand {
                    score: ix.score(&base, n),
                    node: n,
                })
                .collect();
            assert!(
                cands[0].score > cands[1].score && cands[1].score > cands[2].score,
                "fixture must be strictly ordered: {cands:?}"
            );
            let got = ix.select_neighbors(&cands, 0, 2);
            match sel {
                // pure top-2 by score: the near-duplicate survives
                NeighborSelection::KeepBest => assert_eq!(got, vec![1, 2]),
                // c2 is closer to c1 than to the base -> pruned; c3 is
                // not, so the second slot goes to the other direction
                NeighborSelection::Diversity => assert_eq!(got, vec![1, 3]),
            }
        }
    }

    /// Under-full candidate sets are returned whole by both policies
    /// (the paper's early return): nothing to spend the budget on.
    #[test]
    fn selection_returns_everything_when_under_cap() {
        let mut ix = HnswIndex::new(3, 16, 32);
        ix.set_selection(NeighborSelection::Diversity).unwrap();
        for (i, k) in [unit([1.0, 0.0, 0.0]), unit([0.99, 0.14, 0.0])]
            .iter()
            .enumerate()
        {
            ix.insert(i as u32, k).unwrap();
        }
        let cands = vec![Cand {
            score: 0.99,
            node: 1,
        }];
        assert_eq!(ix.select_neighbors(&cands, 0, 8), vec![1]);
    }

    /// Lifecycle contract: the policy shapes the graph AS IT IS BUILT,
    /// so switching it mid-build would leave a hybrid graph. Typed
    /// error, never a silent hybrid.
    #[test]
    fn selection_cannot_change_after_the_first_insert() {
        let mut ix = HnswIndex::new(4, 16, 32);
        assert_eq!(ix.selection(), NeighborSelection::KeepBest);
        ix.insert(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = ix
            .set_selection(NeighborSelection::Diversity)
            .expect_err("must refuse after an insert");
        assert!(matches!(err, BruceError::InvalidArgument(_)), "{err:?}");
        assert_eq!(ix.selection(), NeighborSelection::KeepBest);
    }

    /// Determinism survives the new policy: no RNG, all ties break on
    /// the lower node index, so two builds of the same sequence give
    /// bit-identical search results.
    #[test]
    fn diversity_build_is_deterministic() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let vecs: Vec<Vec<f32>> = (0..600).map(|_| random_unit(&mut rng, 16)).collect();
        let build = || {
            let mut ix = HnswIndex::new(16, 8, 64);
            ix.set_selection(NeighborSelection::Diversity).unwrap();
            for (i, v) in vecs.iter().enumerate() {
                ix.insert(i as u32, v).unwrap();
            }
            ix
        };
        let (a, b) = (build(), build());
        for qi in 0..20 {
            let q = &vecs[qi * 7 % vecs.len()];
            let ra = a.search(q, 10, 64, None).unwrap();
            let rb = b.search(q, 10, 64, None).unwrap();
            assert_eq!(ra.len(), 10);
            assert_eq!(
                ra.iter().map(|c| c.0).collect::<Vec<_>>(),
                rb.iter().map(|c| c.0).collect::<Vec<_>>(),
                "query {qi}: ids differ between builds"
            );
            for (x, y) in ra.iter().zip(&rb) {
                assert_eq!(x.1.to_bits(), y.1.to_bits(), "query {qi}: scores differ");
            }
        }
    }

    /// The graph stays connected and exact at a wide beam under the
    /// new policy: with ef >= n every live node is reachable, so the
    /// top-1 must be the brute-force argmax.
    #[test]
    fn diversity_graph_is_exact_at_a_wide_beam() {
        let mut rng = Rng(0xDEADBEEF12345678);
        let vecs: Vec<Vec<f32>> = (0..400).map(|_| random_unit(&mut rng, 12)).collect();
        let mut ix = HnswIndex::new(12, 8, 64);
        ix.set_selection(NeighborSelection::Diversity).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            ix.insert(i as u32, v).unwrap();
        }
        for qi in 0..15 {
            let q = &vecs[qi * 13 % vecs.len()];
            let want = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (dot(q, v), i as u32))
                .fold((f32::NEG_INFINITY, u32::MAX), |acc, x| {
                    if x.0 > acc.0 || (x.0 == acc.0 && x.1 < acc.1) {
                        x
                    } else {
                        acc
                    }
                });
            let got = ix.search(q, 1, vecs.len(), None).unwrap();
            assert_eq!(got[0].0, want.1, "query {qi}: top-1 must be exact");
        }
    }
}

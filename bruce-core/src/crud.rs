//! Incremental F_ε maintenance — Lemma A turned into an algorithm.
//!
//! For a **fixed query vector x**, the F_ε attention output `A_ε(x)`
//! can be maintained in `O(d_v)` per insert / delete / update of the
//! K/V memory. This is what makes Bruce's exact-unlearning algorithm
//! `O(d)` per record at any N.
//!
//! ### Lemma A
//!
//! ```text
//!     A_ε(x, K, V)  =  Q_ε(x, K, V)  /  Q_ε(x, K, 𝟏)
//! ```
//!
//! `Q_ε` is a SUM, and SUMs form an abelian group under addition —
//! they have inverses. The "subtract the deleted entry" step makes
//! DELETE exact in `O(d_v)`, where existing online-softmax algorithms
//! (FlashAttention, StreamingLLM) are append-only.
//!
//! ### The MAX-aggregate edge case
//!
//! We track scores using a running-max shift `m = maxⱼ scoreⱼ`. When
//! the deleted entry was the unique max, `m` becomes stale and a full
//! rescale `O(n_live)` is required. This is the classical
//! "delete-from-MAX" hard case from database incremental view
//! maintenance — `Σ` has an inverse but `max` does not.

use crate::error::{BruceError, Result};
use crate::types::{Eps, Sim};
use ahash::AHashMap;
use ndarray::{Array1, ArrayView1};

/// The Lemma A accumulator: running max + (m-shifted) numerator and
/// denominator.
#[derive(Debug, Clone)]
pub struct IncrementalState {
    /// running max of the scores
    pub m: f64,
    /// Σⱼ exp((sⱼ − m) / ε) · vⱼ
    pub num: Array1<f64>,
    /// Σⱼ exp((sⱼ − m) / ε)
    pub den: f64,
    /// dimension of the value vectors
    pub d_v: usize,
}

impl IncrementalState {
    /// Build an empty accumulator for value-dim d_v.
    pub fn empty(d_v: usize) -> Self {
        Self {
            m: f64::NEG_INFINITY,
            num: Array1::<f64>::zeros(d_v),
            den: 0.0,
            d_v,
        }
    }

    /// Read the current attention output `A_ε(x)`. Returns zeros if
    /// the memory is empty.
    pub fn output(&self) -> Array1<f64> {
        if self.den <= 0.0 {
            Array1::<f64>::zeros(self.d_v)
        } else {
            &self.num / self.den
        }
    }
}

/// Per-key live entry: `(score, value)`.
#[derive(Debug, Clone)]
struct LiveEntry {
    s: f64,
    v: Array1<f64>,
}

/// Incrementally-maintained F_ε attention output for a fixed query x.
pub struct IncrementalMemory {
    x: Array1<f64>,
    eps: Eps,
    sim: Sim,
    state: IncrementalState,
    live: AHashMap<String, LiveEntry>,
    n_ops: u64,
    n_rescales: u64,
}

impl IncrementalMemory {
    /// Create a new fixed-query incremental memory.
    pub fn new(x: ArrayView1<'_, f64>, eps: Eps, d_v: usize, sim: Sim) -> Self {
        Self {
            x: x.to_owned(),
            eps,
            sim,
            state: IncrementalState::empty(d_v),
            live: AHashMap::new(),
            n_ops: 0,
            n_rescales: 0,
        }
    }

    /// Number of live records.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Is the memory empty?
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Current `A_ε(x)`.
    pub fn output(&self) -> Array1<f64> {
        self.state.output()
    }

    /// How many `O(n_live)` rescales have been triggered so far.
    pub fn n_rescales(&self) -> u64 {
        self.n_rescales
    }

    /// `score = sim(x, k)`, used internally and exposed for callers.
    pub fn score(&self, k: ArrayView1<'_, f64>) -> f64 {
        match self.sim {
            Sim::Dot => self.x.dot(&k),
            Sim::NegSquared => {
                let diff = &self.x - &k;
                -0.5 * diff.dot(&diff)
            }
            Sim::Indicator => {
                let diff = &self.x - &k;
                if diff.dot(&diff) == 0.0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Insert a new (key, value) pair, identified by `key_id` so the
    /// caller can delete it later by id. O(d_v).
    pub fn insert(
        &mut self,
        key_id: &str,
        k: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Result<()> {
        if k.len() != self.x.len() {
            return Err(BruceError::DimensionMismatch {
                expected: self.x.len(),
                got: k.len(),
            });
        }
        if v.len() != self.state.d_v {
            return Err(BruceError::DimensionMismatch {
                expected: self.state.d_v,
                got: v.len(),
            });
        }
        if self.live.contains_key(key_id) {
            return Err(BruceError::DuplicateKey(key_id.into()));
        }
        let s = self.score(k);

        if self.eps.is_zero() {
            // ε = 0: indicator semantics; only argmax contributes
            self.insert_tropical(key_id, s, v.to_owned());
        } else {
            self.insert_continuous(key_id, s, v.to_owned());
        }
        self.n_ops += 1;
        Ok(())
    }

    fn insert_continuous(&mut self, key_id: &str, s: f64, v: Array1<f64>) {
        // raise running max + rescale if needed
        if s > self.state.m {
            if self.state.m.is_finite() {
                let factor = ((self.state.m - s) / self.eps.0).exp();
                self.state.num *= factor;
                self.state.den *= factor;
            }
            self.state.m = s;
        }
        let w = ((s - self.state.m) / self.eps.0).exp();
        self.state.num.scaled_add(w, &v);
        self.state.den += w;
        self.live.insert(key_id.into(), LiveEntry { s, v });
    }

    fn insert_tropical(&mut self, key_id: &str, s: f64, v: Array1<f64>) {
        // ε = 0: only entries with s == m count
        if !s.is_finite() {
            self.live.insert(key_id.into(), LiveEntry { s, v });
            return;
        }
        if s > self.state.m {
            // new max; reset accumulator to only this entry
            self.state.num.fill(0.0);
            self.state.den = 0.0;
            self.state.m = s;
        }
        if s == self.state.m {
            self.state.num.scaled_add(1.0, &v);
            self.state.den += 1.0;
        }
        self.live.insert(key_id.into(), LiveEntry { s, v });
    }

    /// Delete a previously inserted key. O(d_v) when the deleted
    /// entry is NOT the unique max; O(n_live) otherwise.
    pub fn delete(&mut self, key_id: &str) -> Result<()> {
        let entry = self
            .live
            .remove(key_id)
            .ok_or_else(|| BruceError::KeyNotFound(key_id.into()))?;
        let s = entry.s;
        let v = entry.v;

        if self.eps.is_zero() {
            self.delete_tropical(s, &v);
        } else {
            // O(d_v) subtract
            let w = ((s - self.state.m) / self.eps.0).exp();
            self.state.num.scaled_add(-w, &v);
            self.state.den -= w;
            // if we removed (one of) the max-scorers, the running max
            // is stale — rescale once
            if s >= self.state.m - 1e-12 {
                self.rescale();
            }
        }
        self.n_ops += 1;
        Ok(())
    }

    fn delete_tropical(&mut self, s: f64, v: &Array1<f64>) {
        if s == self.state.m {
            self.state.num.scaled_add(-1.0, v);
            self.state.den -= 1.0;
            if self.state.den == 0.0 {
                // we removed the last max-scorer; rescale to find the
                // new max
                self.rescale_tropical();
            }
        }
        // otherwise (s < m) the entry contributed nothing; nothing to do
    }

    /// Replace key (k, v). Equivalent to delete + insert.
    pub fn update(
        &mut self,
        key_id: &str,
        k: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.delete(key_id)?;
        self.insert(key_id, k, v)
    }

    /// Full O(n_live) rescale: recompute (m, num, den) from the live set.
    fn rescale(&mut self) {
        self.n_rescales += 1;
        if self.live.is_empty() {
            self.state.m = f64::NEG_INFINITY;
            self.state.num.fill(0.0);
            self.state.den = 0.0;
            return;
        }
        let new_m = self
            .live
            .values()
            .map(|e| e.s)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut num = Array1::<f64>::zeros(self.state.d_v);
        let mut den = 0.0;
        for e in self.live.values() {
            let w = ((e.s - new_m) / self.eps.0).exp();
            num.scaled_add(w, &e.v);
            den += w;
        }
        self.state.m = new_m;
        self.state.num = num;
        self.state.den = den;
    }

    fn rescale_tropical(&mut self) {
        self.n_rescales += 1;
        if self.live.is_empty() {
            self.state.m = f64::NEG_INFINITY;
            self.state.num.fill(0.0);
            self.state.den = 0.0;
            return;
        }
        let new_m = self
            .live
            .values()
            .map(|e| e.s)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut num = Array1::<f64>::zeros(self.state.d_v);
        let mut den = 0.0;
        for e in self.live.values() {
            if e.s == new_m {
                num.scaled_add(1.0, &e.v);
                den += 1.0;
            }
        }
        self.state.m = new_m;
        self.state.num = num;
        self.state.den = den;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn insert_then_output_matches_recompute() {
        let x = array![1.0, 0.0];
        let mut m = IncrementalMemory::new(x.view(), Eps::ONE, 1, Sim::Dot);
        m.insert("a", array![1.0, 0.0].view(), array![10.0].view()).unwrap();
        m.insert("b", array![0.0, 1.0].view(), array![20.0].view()).unwrap();
        let out = m.output();
        // softmax([1, 0]) = [e/(e+1), 1/(e+1)] → 10·e/(e+1) + 20·1/(e+1)
        let e = std::f64::consts::E;
        let expected = 10.0 * e / (e + 1.0) + 20.0 / (e + 1.0);
        assert_abs_diff_eq!(out[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn delete_recovers_never_inserted() {
        // The exact unlearning claim: insert N records, delete one,
        // result == compute over the remaining N-1.
        let x = array![1.0];
        let mut m = IncrementalMemory::new(x.view(), Eps::ONE, 1, Sim::Dot);
        for i in 0..50 {
            m.insert(
                &format!("k{i}"),
                array![i as f64 * 0.1].view(),
                array![i as f64].view(),
            ).unwrap();
        }
        // delete record 7
        m.delete("k7").unwrap();
        let out_after = m.output();

        // recompute the F_eps over the surviving 49 records from scratch
        let mut m2 = IncrementalMemory::new(x.view(), Eps::ONE, 1, Sim::Dot);
        for i in 0..50 {
            if i == 7 {
                continue;
            }
            m2.insert(
                &format!("k{i}"),
                array![i as f64 * 0.1].view(),
                array![i as f64].view(),
            ).unwrap();
        }
        let out_clean = m2.output();
        // bit-level identity up to floating-point precision
        let err = (out_after[0] - out_clean[0]).abs();
        assert!(err < 1e-12, "expected bit-level identity, got err = {err:e}");
    }

    #[test]
    fn delete_of_dominant_record_triggers_rescale() {
        let x = array![1.0];
        let mut m = IncrementalMemory::new(x.view(), Eps::ONE, 1, Sim::Dot);
        // record 0 has score 1.0 ≫ others
        m.insert("dominant", array![10.0].view(), array![1000.0].view()).unwrap();
        for i in 1..20 {
            m.insert(
                &format!("k{i}"),
                array![0.01].view(),
                array![i as f64].view(),
            ).unwrap();
        }
        let pre = m.output();
        m.delete("dominant").unwrap();
        let post = m.output();
        assert!((pre[0] - post[0]).abs() > 1.0, "expected dominant delete to change output significantly");
        assert!(m.n_rescales() >= 1, "expected at least one rescale");
    }

    #[test]
    fn tropical_eps_zero_sum_works() {
        // ε = 0 + Sim::Indicator: only exact-matching keys contribute
        let x = array![1.0];
        let mut m = IncrementalMemory::new(x.view(), Eps::ZERO, 1, Sim::Indicator);
        m.insert("match1", array![1.0].view(), array![10.0].view()).unwrap();
        m.insert("nomatch", array![0.0].view(), array![999.0].view()).unwrap();
        m.insert("match2", array![1.0].view(), array![20.0].view()).unwrap();
        // SQL: SELECT AVG(v) WHERE k = 1 -> (10 + 20) / 2 = 15
        let out = m.output();
        assert_eq!(out[0], 15.0);
    }
}

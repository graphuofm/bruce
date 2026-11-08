//! relf-core: exact states for the per-group relative cutoff
//!
//! ```text
//! relF(v, s, d) over a row set S = AVG(v_r) over { r in S : s_r >= max_S(s) - d }
//! ```
//!
//! This is the baseline the anchored-aggregation experiment
//! (`paper_icde/experiments/anchored`) puts through the same engine paths
//! as `softavg`. It is deliberately NOT part of softagg-core. Three states,
//! one per engine path, each the smallest exact state we know for that path:
//!
//! * [`Band`] (GROUP BY, insert-only, mergeable): the rows with
//!   `s >= max - d` of the rows seen so far. Rows below the cutoff can never
//!   re-enter because the max only rises, so they are dropped for good.
//!   Size = the band, not the group; but the band is not O(1): for an
//!   exact one-pass answer any row within `d` of the running max may still
//!   count (a later row at `s' = s_r + d` would make a different prefix sum
//!   visible), so `Omega(|band|)` words are needed (fooling-set argument,
//!   README of the experiment).
//! * [`WinBand`] (FIFO sliding frames: PostgreSQL MSFUNC/MINVFUNC, SQLite
//!   xStep/xInverse): a row `j` is dropped as soon as a newer row `i` with
//!   `s_i > s_j + d` arrives (in a FIFO frame `i` outlives `j`, so the frame
//!   max stays `>= s_i` while `j` is present and `j` is never in the band).
//!   Every row that is NOT dominated this way is in the band of some future
//!   frame if the future rows are low, so this "undominated" set is the
//!   minimal exact state. Running (count, sum) over the band, adjusted when
//!   the cutoff moves; Neumaier-compensated.
//! * [`IvmBand`] (maintenance under arbitrary inserts/deletes): all group
//!   rows with `s >= floor`, `floor = max - d - h` at (re)build time, plus
//!   the group's row count. A delete of the max lowers the cutoff; as long
//!   as the new cutoff stays `>= floor` the answer is still exact. Once it
//!   drops below `floor` (or the band empties while rows remain) the state
//!   cannot know which rows re-enter and the host must rescan the group.
//!   `h = +inf` keeps the whole group (never rescans, O(group) state).
//!
//! NaN scores/values are rejected by the hosts (no SQL-equivalent ordering
//! semantics are attempted); NULL rows are skipped by the hosts.
//! `d = +inf` is plain AVG (count/sum, no band); `d` must be `>= 0`.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

#[derive(Clone, Copy, Debug)]
struct Entry {
    s: f64,
    v: f64,
}
impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    // reversed on s: BinaryHeap is a max-heap, we want the lowest score on top
    fn cmp(&self, o: &Self) -> Ordering {
        o.s.total_cmp(&self.s).then(o.v.total_cmp(&self.v))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct K(u64);
impl K {
    /// order-preserving map of a non-NaN f64 to u64 (total order)
    #[inline]
    fn of(x: f64) -> K {
        // -0.0 and 0.0 compare equal in SQL/IEEE; give them one key
        let b = (x + 0.0).to_bits();
        K(if b >> 63 == 1 { !b } else { b | (1u64 << 63) })
    }
}

/// Neumaier-compensated running sum supporting add and subtract.
#[derive(Clone, Copy, Debug, Default)]
pub struct CSum {
    s: f64,
    c: f64,
}
impl CSum {
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }
    #[inline]
    pub fn value(&self) -> f64 {
        self.s + self.c
    }
    pub fn reset(&mut self) {
        *self = CSum::default();
    }
}

pub fn check_d(d: f64) -> Result<f64, String> {
    if d.is_nan() || d < 0.0 {
        Err(format!("relf: d must be >= 0 ('infinity' = plain AVG), got {d}"))
    } else {
        Ok(d)
    }
}

// ---------------------------------------------------------------------
// Band: GROUP BY (insert-only, mergeable)
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Band {
    pub d: f64,
    pub max: f64,
    heap: BinaryHeap<Entry>,
    /// d = inf: plain AVG
    cnt: u64,
    sum: f64,
    /// largest band held so far (state-size statistic)
    pub peak: usize,
    /// rows absorbed (including those never kept)
    pub seen: u64,
}

impl Band {
    pub fn new(d: f64) -> Self {
        Band { d, max: f64::NEG_INFINITY, heap: BinaryHeap::new(), cnt: 0, sum: 0.0, peak: 0, seen: 0 }
    }
    #[inline]
    pub fn cutoff(&self) -> f64 {
        self.max - self.d
    }
    pub fn len(&self) -> usize {
        if self.d.is_infinite() {
            0
        } else {
            self.heap.len()
        }
    }
    pub fn is_empty(&self) -> bool {
        self.seen == 0
    }
    #[inline]
    fn prune(&mut self) {
        let c = self.cutoff();
        while let Some(top) = self.heap.peek() {
            if top.s < c {
                self.heap.pop();
            } else {
                break;
            }
        }
    }
    #[inline]
    pub fn absorb(&mut self, s: f64, v: f64) {
        self.seen += 1;
        if self.d.is_infinite() {
            self.cnt += 1;
            self.sum += v;
            return;
        }
        if s > self.max {
            self.max = s;
            self.heap.push(Entry { s, v });
            self.prune();
        } else if s >= self.max - self.d {
            self.heap.push(Entry { s, v });
        } else {
            return;
        }
        if self.heap.len() > self.peak {
            self.peak = self.heap.len();
        }
    }
    pub fn merge(&mut self, o: &Band) {
        self.seen += o.seen;
        if self.d.is_infinite() {
            self.cnt += o.cnt;
            self.sum += o.sum;
            return;
        }
        if o.max > self.max {
            self.max = o.max;
        }
        let c = self.cutoff();
        for e in o.heap.iter() {
            if e.s >= c {
                self.heap.push(*e);
            }
        }
        self.prune();
        self.peak = self.peak.max(o.peak).max(self.heap.len());
    }
    pub fn finalize(&self) -> Option<f64> {
        if self.seen == 0 {
            return None;
        }
        if self.d.is_infinite() {
            return Some(self.sum / self.cnt as f64);
        }
        // sum in score order (deterministic regardless of heap layout)
        let mut es: Vec<Entry> = self.heap.iter().copied().collect();
        es.sort_unstable_by(|a, b| b.cmp(a));
        let mut t = 0.0;
        for e in &es {
            t += e.v;
        }
        Some(t / es.len() as f64)
    }
    /// little-endian: d, max, seen, cnt, sum, n, then n (s, v) pairs
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(48 + 16 * self.heap.len());
        for x in [self.d, self.max, f64::from_bits(self.seen), f64::from_bits(self.cnt), self.sum] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&(self.heap.len() as u64).to_le_bytes());
        for e in self.heap.iter() {
            out.extend_from_slice(&e.s.to_le_bytes());
            out.extend_from_slice(&e.v.to_le_bytes());
        }
        out
    }
    pub fn deserialize(b: &[u8]) -> Option<Band> {
        let f = |i: usize| -> Option<f64> { Some(f64::from_le_bytes(b.get(8 * i..8 * i + 8)?.try_into().ok()?)) };
        let n = f(5)?.to_bits() as usize;
        if b.len() != 48 + 16 * n {
            return None;
        }
        let mut heap = BinaryHeap::with_capacity(n);
        for i in 0..n {
            heap.push(Entry { s: f(6 + 2 * i)?, v: f(7 + 2 * i)? });
        }
        Some(Band { d: f(0)?, max: f(1)?, seen: f(2)?.to_bits(), cnt: f(3)?.to_bits(), sum: f(4)?, heap, peak: n })
    }
}

// ---------------------------------------------------------------------
// WinBand: FIFO sliding frames
// ---------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub enum WinErr {
    /// the departing row is not the oldest row pushed (frame is not FIFO)
    NotFifo,
}

#[derive(Clone, Debug)]
pub struct WinBand {
    pub d: f64,
    by_score: BTreeMap<(K, u64), f64>,
    by_seq: BTreeMap<u64, (f64, f64)>,
    head: u64,
    next: u64,
    /// current cutoff; +inf when empty (nothing counted)
    cut: f64,
    cnt: u64,
    sum: CSum,
    /// d = inf
    pub peak: usize,
    /// sum over pushes of the state size after the push (mean = /pushes)
    pub size_acc: u128,
    pub pushes: u64,
    /// entries visited while moving the cutoff (work statistic)
    pub cut_moves: u64,
}

impl WinBand {
    pub fn new(d: f64) -> Self {
        WinBand {
            d,
            by_score: BTreeMap::new(),
            by_seq: BTreeMap::new(),
            head: 0,
            next: 0,
            cut: f64::INFINITY,
            cnt: 0,
            sum: CSum::default(),
            peak: 0,
            size_acc: 0,
            pushes: 0,
            cut_moves: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.by_seq.len()
    }
    pub fn frame_rows(&self) -> u64 {
        self.next - self.head
    }
    #[inline]
    fn counted(&self, s: f64) -> bool {
        s >= self.cut
    }
    fn set_cut(&mut self, new: f64) {
        let old = self.cut;
        if new == old {
            return;
        }
        let (lo, hi, sign) = if new > old { (old, new, -1.0) } else { (new, old, 1.0) };
        let mut moved = 0u64;
        for (_, v) in self.by_score.range((K::of(lo), 0)..(K::of(hi), 0)) {
            self.sum.add(sign * *v);
            moved += 1;
        }
        if sign > 0.0 {
            self.cnt += moved;
        } else {
            self.cnt -= moved;
        }
        self.cut_moves += moved;
        self.cut = new;
    }
    fn refresh_cut(&mut self) {
        let new = match self.by_score.keys().next_back() {
            None => f64::INFINITY,
            Some(_) => {
                let (&(_, seq), _) = self.by_score.iter().next_back().unwrap();
                self.by_seq[&seq].0 - self.d
            }
        };
        self.set_cut(new);
        if self.by_seq.is_empty() {
            // exact by construction
            self.cnt = 0;
            self.sum.reset();
        }
    }
    pub fn push(&mut self, s: f64, v: f64) {
        let seq = self.next;
        self.next += 1;
        if self.d.is_infinite() {
            self.by_seq.insert(seq, (s, v));
            self.cnt += 1;
            self.sum.add(v);
        } else {
            // drop rows dominated by the new one: s_j < s - d
            let thr = s - self.d;
            let dead: Vec<(K, u64)> = self.by_score.range(..(K::of(thr), 0)).map(|(k, _)| *k).collect();
            for k in dead {
                let v_old = self.by_score.remove(&k).unwrap();
                let (s_old, _) = self.by_seq.remove(&k.1).unwrap();
                if self.counted(s_old) {
                    self.cnt -= 1;
                    self.sum.add(-v_old);
                }
            }
            self.by_score.insert((K::of(s), seq), v);
            self.by_seq.insert(seq, (s, v));
            if self.counted(s) {
                self.cnt += 1;
                self.sum.add(v);
            }
            self.refresh_cut();
        }
        let n = self.by_seq.len();
        self.peak = self.peak.max(n);
        self.size_acc += n as u128;
        self.pushes += 1;
    }
    /// Remove the oldest row of the frame (its values are checked when it
    /// is still held).
    pub fn pop(&mut self, s: f64, v: f64) -> Result<(), WinErr> {
        if self.head >= self.next {
            return Err(WinErr::NotFifo);
        }
        let seq = self.head;
        self.head += 1;
        if let Some((s0, v0)) = self.by_seq.remove(&seq) {
            if s0.to_bits() != s.to_bits() || v0.to_bits() != v.to_bits() {
                return Err(WinErr::NotFifo);
            }
            if self.d.is_infinite() {
                self.cnt -= 1;
                self.sum.add(-v);
                if self.by_seq.is_empty() {
                    self.cnt = 0;
                    self.sum.reset();
                }
                return Ok(());
            }
            self.by_score.remove(&(K::of(s0), seq));
            if self.counted(s0) {
                self.cnt -= 1;
                self.sum.add(-v0);
            }
            self.refresh_cut();
        }
        Ok(())
    }
    pub fn value(&self) -> Option<f64> {
        if self.cnt == 0 {
            None
        } else {
            Some(self.sum.value() / self.cnt as f64)
        }
    }
    /// exact recomputation of the band average from the held rows (tests)
    pub fn value_exact(&self) -> Option<f64> {
        if self.by_seq.is_empty() {
            return None;
        }
        let m = self.by_seq.values().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
        let c = m - self.d;
        let (mut t, mut n) = (0.0, 0u64);
        for (s, v) in self.by_seq.values() {
            if self.d.is_infinite() || *s >= c {
                t += v;
                n += 1;
            }
        }
        Some(t / n as f64)
    }
}

// ---------------------------------------------------------------------
// IvmBand: maintenance under inserts and deletes
// ---------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub enum IvmErr {
    /// the cutoff fell below the floor (or the band emptied while the group
    /// still has rows): rows not held may re-enter, rescan the group
    Rescan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvmBand {
    pub d: f64,
    pub h: f64,
    /// every group row with s >= floor is held
    pub floor: f64,
    /// group rows (held or not)
    pub n: u64,
    /// sorted by (s, v)
    rows: Vec<(f64, f64)>,
}

fn cmp_row(a: &(f64, f64), b: &(f64, f64)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
}

impl IvmBand {
    pub fn empty(d: f64, h: f64) -> Self {
        IvmBand { d, h, floor: f64::NEG_INFINITY, n: 0, rows: Vec::new() }
    }
    /// Build from all rows of a group (any order).
    pub fn build(d: f64, h: f64, rows: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let mut b = Band::new(if d.is_infinite() || h.is_infinite() { f64::INFINITY } else { d + h });
        let mut all = Vec::new();
        let keep_all = d.is_infinite() || h.is_infinite();
        for (s, v) in rows {
            if keep_all {
                all.push((s, v));
                b.seen += 1;
            } else {
                b.absorb(s, v);
            }
        }
        IvmBand::from_parts(d, h, b.seen, if keep_all { all } else { b.heap.iter().map(|e| (e.s, e.v)).collect() }, if keep_all { f64::NEG_INFINITY } else { b.max - (d + h) })
    }
    fn from_parts(d: f64, h: f64, n: u64, mut rows: Vec<(f64, f64)>, floor: f64) -> Self {
        rows.sort_unstable_by(cmp_row);
        IvmBand { d, h, floor: if n == 0 { f64::NEG_INFINITY } else { floor }, n, rows }
    }
    pub fn held(&self) -> usize {
        self.rows.len()
    }
    pub fn max(&self) -> Option<f64> {
        self.rows.last().map(|r| r.0)
    }
    /// Apply one statement's deletes then inserts.
    pub fn apply(&mut self, dels: &[(f64, f64)], ins: &[(f64, f64)]) -> Result<(), IvmErr> {
        for r in dels {
            if self.n == 0 {
                return Err(IvmErr::Rescan);
            }
            self.n -= 1;
            if r.0 >= self.floor {
                match self.rows.binary_search_by(|x| cmp_row(x, r)) {
                    Ok(i) => {
                        self.rows.remove(i);
                    }
                    Err(_) => return Err(IvmErr::Rescan),
                }
            }
        }
        for r in ins {
            self.n += 1;
            if r.0 >= self.floor {
                let i = match self.rows.binary_search_by(|x| cmp_row(x, r)) {
                    Ok(i) | Err(i) => i,
                };
                self.rows.insert(i, *r);
            }
        }
        if self.n == 0 {
            *self = IvmBand::empty(self.d, self.h);
            return Ok(());
        }
        let Some(m) = self.max() else { return Err(IvmErr::Rescan) };
        if self.d.is_infinite() || self.h.is_infinite() {
            return Ok(());
        }
        if m - self.d < self.floor {
            return Err(IvmErr::Rescan);
        }
        // raise the floor with the max (keeps the state at the band + headroom)
        let f = m - self.d - self.h;
        if f > self.floor {
            self.floor = f;
            let k = self.rows.partition_point(|x| x.0 < f);
            self.rows.drain(..k);
        }
        Ok(())
    }
    pub fn value(&self) -> Option<f64> {
        let m = self.max()?;
        let (mut t, mut c) = (0.0, 0u64);
        if self.d.is_infinite() {
            for r in &self.rows {
                t += r.1;
                c += 1;
            }
        } else {
            let cut = m - self.d;
            for r in self.rows.iter().rev() {
                if r.0 < cut {
                    break;
                }
                t += r.1;
                c += 1;
            }
        }
        Some(t / c as f64)
    }
    /// little-endian: d, h, floor, n, len, then (s, v) pairs
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(40 + 16 * self.rows.len());
        for x in [self.d, self.h, self.floor, f64::from_bits(self.n), f64::from_bits(self.rows.len() as u64)] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for r in &self.rows {
            out.extend_from_slice(&r.0.to_le_bytes());
            out.extend_from_slice(&r.1.to_le_bytes());
        }
        out
    }
    pub fn deserialize(b: &[u8]) -> Option<IvmBand> {
        let f = |i: usize| -> Option<f64> { Some(f64::from_le_bytes(b.get(8 * i..8 * i + 8)?.try_into().ok()?)) };
        let len = f(4)?.to_bits() as usize;
        if b.len() != 40 + 16 * len {
            return None;
        }
        let mut rows = Vec::with_capacity(len);
        for i in 0..len {
            rows.push((f(5 + 2 * i)?, f(6 + 2 * i)?));
        }
        Some(IvmBand { d: f(0)?, h: f(1)?, floor: f(2)?, n: f(3)?.to_bits(), rows })
    }
}

/// Reference: relF over a slice (two passes, as the stock SQL does).
pub fn relf_ref(rows: &[(f64, f64)], d: f64) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let m = rows.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
    let c = m - d;
    let (mut t, mut n) = (0.0, 0u64);
    for r in rows {
        if d.is_infinite() || r.0 >= c {
            t += r.1;
            n += 1;
        }
    }
    Some(t / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }
    fn rows(n: usize, seed: u64, quant: bool) -> Vec<(f64, f64)> {
        let mut r = Rng(seed | 1);
        (0..n)
            .map(|_| {
                let s = r.next() * 0.8 - 0.1;
                let s = if quant { (s * 20.0).round() / 20.0 } else { s };
                (s, (r.next() * 10.0).round())
            })
            .collect()
    }
    fn close(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => (x - y).abs() <= tol * (1.0 + y.abs()),
            _ => false,
        }
    }

    #[test]
    fn band_matches_reference_any_order_and_merge() {
        for &d in &[0.0, 0.01, 0.05, 0.15, 0.3, 1.0, f64::INFINITY] {
            for seed in 1..20u64 {
                for quant in [false, true] {
                    let rs = rows(200 + seed as usize * 7, seed, quant);
                    let mut b = Band::new(d);
                    for r in &rs {
                        b.absorb(r.0, r.1);
                    }
                    assert!(close(b.finalize(), relf_ref(&rs, d), 1e-12), "d={d} seed={seed}");
                    // increasing order: worst case for the heap
                    let mut sorted = rs.clone();
                    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let mut b2 = Band::new(d);
                    for r in &sorted {
                        b2.absorb(r.0, r.1);
                    }
                    assert!(close(b2.finalize(), relf_ref(&rs, d), 1e-12));
                    // merge of 7 parts == sequential
                    let mut parts: Vec<Band> = (0..7).map(|_| Band::new(d)).collect();
                    for (i, r) in rs.iter().enumerate() {
                        parts[i % 7].absorb(r.0, r.1);
                    }
                    let mut m = Band::new(d);
                    for p in &parts {
                        let p2 = Band::deserialize(&p.serialize()).unwrap();
                        m.merge(&p2);
                    }
                    assert!(close(m.finalize(), relf_ref(&rs, d), 1e-12));
                    if d.is_finite() {
                        let c = relf_band_size(&rs, d);
                        assert_eq!(b.len(), c);
                    }
                }
            }
        }
        assert_eq!(Band::new(0.1).finalize(), None);
    }

    fn relf_band_size(rs: &[(f64, f64)], d: f64) -> usize {
        let m = rs.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
        rs.iter().filter(|r| r.0 >= m - d).count()
    }

    #[test]
    fn winband_matches_reference() {
        for &d in &[0.0, 0.02, 0.1, 0.15, 0.5, f64::INFINITY] {
            for &k in &[1usize, 5, 50, 300] {
                for quant in [false, true] {
                    let rs = rows(1500, 7 + k as u64, quant);
                    let mut w = WinBand::new(d);
                    for i in 0..rs.len() {
                        w.push(rs[i].0, rs[i].1);
                        if i > k {
                            let o = rs[i - k - 1];
                            w.pop(o.0, o.1).unwrap();
                        }
                        let lo = i.saturating_sub(k);
                        let want = relf_ref(&rs[lo..=i], d);
                        assert!(close(w.value(), want, 1e-12), "d={d} k={k} i={i}: {:?} vs {:?}", w.value(), want);
                        assert!(close(w.value_exact(), want, 1e-12));
                    }
                    if d < 0.5 && k >= 50 {
                        assert!(w.peak < k + 1, "pruning keeps fewer rows than the frame");
                    }
                }
            }
        }
        // popping in the wrong order is detected when the row is still held
        let mut w = WinBand::new(0.5);
        w.push(0.1, 1.0);
        w.push(0.2, 2.0);
        assert_eq!(w.pop(0.2, 2.0), Err(WinErr::NotFifo));
    }

    #[test]
    fn ivmband_matches_reference_or_asks_rescan() {
        for &d in &[0.0, 0.05, 0.15, f64::INFINITY] {
            for &h in &[0.0, 0.05, 0.2, f64::INFINITY] {
                let mut r = Rng(99);
                let mut live: Vec<(f64, f64)> = rows(300, 3, false);
                let mut st = IvmBand::build(d, h, live.iter().copied());
                let mut rescans = 0;
                for step in 0..3000 {
                    let del = r.next() < 0.5 && !live.is_empty();
                    let res = if del {
                        // every 10th delete removes the max
                        let i = if step % 10 == 0 {
                            (0..live.len()).max_by(|&a, &b| live[a].0.total_cmp(&live[b].0)).unwrap()
                        } else {
                            (r.next() * live.len() as f64) as usize % live.len()
                        };
                        let row = live.swap_remove(i);
                        st.apply(&[row], &[])
                    } else {
                        let row = (r.next() * 0.8, (r.next() * 10.0).round());
                        live.push(row);
                        st.apply(&[], &[row])
                    };
                    if res.is_err() {
                        rescans += 1;
                        st = IvmBand::build(d, h, live.iter().copied());
                    }
                    let st2 = IvmBand::deserialize(&st.serialize()).unwrap();
                    assert_eq!(st2, st);
                    assert!(close(st.value(), relf_ref(&live, d), 1e-12), "d={d} h={h} step={step}");
                    assert_eq!(st.n as usize, live.len());
                }
                if h.is_infinite() || d.is_infinite() {
                    assert_eq!(rescans, 0);
                } else if h == 0.0 {
                    assert!(rescans > 0);
                }
            }
        }
    }
}

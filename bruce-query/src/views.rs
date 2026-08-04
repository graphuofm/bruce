//! Maintained soft-aggregate views: the (m, num, den) state per group,
//! kept fresh under the write path's deltas. The planner serves a
//! matching query from the view in O(groups) instead of re-scanning.
//!
//! Maintenance contract (matches the kernel's, crud.rs):
//!   insert            O(1) per row (amortized; may rescale its group)
//!   delete, non-max   O(1) subtraction
//!   delete of a row whose score ties the group max: one bounded
//!                     re-anchor pass over THAT group's surviving rows
//!                     (reads the base table).

use ndarray::ArrayView1;

use crate::catalog::{Column, Table};
use crate::QueryError;

/// Fingerprint of a bound query vector (views are per bound query).
pub fn param_fingerprint(x: &ArrayView1<'_, f64>) -> u64 {
    // FNV-1a over the raw bits: deterministic, no deps.
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in x.iter() {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Per-group maintained state.
#[derive(Debug, Clone)]
struct GroupState {
    /// Anchor (upper bound of scores seen; re-anchored on max delete).
    m: f64,
    /// Anchored weighted numerator.
    num: f64,
    /// Anchored weight denominator.
    den: f64,
    /// Live row count.
    n: u64,
}

impl GroupState {
    fn empty() -> Self {
        GroupState {
            m: f64::NEG_INFINITY,
            num: 0.0,
            den: 0.0,
            n: 0,
        }
    }
}

/// Storage dtype of the view's key column — decides the scoring
/// arithmetic (f64 dot vs. the f32-storage contract of
/// `bruce_core::mask::grouped_softavg_f32`: f32 scoring, f64 state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyDtype {
    F64,
    F32,
}

/// Dtype-polymorphic borrow of a key column, private to views (db.rs
/// has its own wire-format reader, `key_rows_f64`, because the write
/// path needs owned f64 rows for `SoftAggView::on_delete`).
enum KeyColRef<'a> {
    F64(&'a ndarray::Array2<f64>),
    F32(&'a ndarray::Array2<f32>),
}

impl KeyColRef<'_> {
    fn ncols(&self) -> usize {
        match self {
            KeyColRef::F64(a) => a.ncols(),
            KeyColRef::F32(a) => a.ncols(),
        }
    }
}

fn any_key_col<'a>(t: &'a Table, name: &str) -> Result<KeyColRef<'a>, QueryError> {
    match t.columns.get(name) {
        Some(Column::KeyF64(a)) => Ok(KeyColRef::F64(a)),
        Some(Column::KeyF32(a)) => Ok(KeyColRef::F32(a)),
        _ => Err(QueryError::Bind(format!(
            "column {name} must be KeyF64 or KeyF32"
        ))),
    }
}

/// Score one wire-format key row (RowValues carry f64) under the
/// view's dtype. F32 views cast every component down first — exactly
/// the value the KeyF32 column stores for that row (db.rs casts on
/// append), so incremental deltas score bit-identically to a
/// from-scratch rebuild over the stored column — then dot in f32 and
/// widen once, the `grouped_softavg_f32` precision contract.
fn score_slice(dtype: KeyDtype, q64: &[f64], q32: &[f32], key_row: &[f64]) -> f64 {
    match dtype {
        KeyDtype::F64 => key_row.iter().zip(q64).map(|(a, b)| a * b).sum(),
        KeyDtype::F32 => {
            let mut acc = 0.0f32;
            for (a, b) in key_row.iter().zip(q32) {
                acc += (*a as f32) * b;
            }
            acc as f64
        }
    }
}

/// A maintained soft-aggregate view over one table.
#[derive(Debug, Clone)]
pub struct SoftAggView {
    /// View name (for EXPLAIN and registration).
    pub name: String,
    /// Base table.
    pub table: String,
    /// Group column (DictU32).
    pub group_col: String,
    /// Value column (ScalarF64).
    pub val_col: String,
    /// Key column (KeyF64 or KeyF32; the dtype fixes the scoring
    /// arithmetic — see [`score_slice`]).
    pub key_col: String,
    /// Fingerprint of the bound query vector this view serves.
    pub param_fp: u64,
    /// Temperature.
    pub eps: f64,
    /// The bound query vector (needed to score delta rows).
    query: Vec<f64>,
    /// The query cast to f32, used when `dtype` is F32 (empty for F64
    /// views).
    query32: Vec<f32>,
    /// Storage dtype of the key column at build time.
    dtype: KeyDtype,
    /// Per-group state, indexed by group code.
    groups: Vec<GroupState>,
    /// Number of bounded re-anchors performed (observability).
    pub n_reanchors: u64,
}

impl SoftAggView {
    /// Build the view from the current table contents. (The argument
    /// list is the view's full identity; a builder would just rename
    /// the same eight facts.)
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        name: &str,
        table_name: &str,
        t: &Table,
        group_col: &str,
        val_col: &str,
        key_col: &str,
        x: &ArrayView1<'_, f64>,
        eps: f64,
    ) -> Result<Self, QueryError> {
        // eps validation (tests/error_totality.rs): the incremental
        // (m, num, den) maintenance below is the max-anchored softmax,
        // which is defined only for a valid temperature 0 < eps <= inf.
        // The eps = 0 tropical endpoint has no incremental form here
        // (it needs the argmax path) and previously produced NaN state
        // silently; invalid eps (negative, NaN) is rejected by the
        // same contract as the kernel's Eps::new.
        bruce_core::Eps::new(eps).map_err(|e| QueryError::Bind(e.to_string()))?;
        if eps == 0.0 {
            return Err(QueryError::Bind(
                "maintained views require eps > 0; the eps=0 tropical endpoint \
                 is served by the exact kernel path, not incremental maintenance"
                    .into(),
            ));
        }
        let (codes, dict) = dict_col(t, group_col)?;
        let vals = scalar_col(t, val_col)?;
        // f32 views (tests in this file's mod tests + bruce-py's
        // test_f32_views.py): KeyF32 columns are first-class — f32
        // storage/scoring, same f64 (m, num, den) state and
        // group-inverse delete path as f64 views. The former typed
        // refusal is flipped in tests/error_totality.rs.
        let keys = any_key_col(t, key_col)?;
        // dimension validation (tests/error_totality.rs): scoring dots
        // the bound query against each key row; a mismatched vector
        // previously panicked inside ndarray's dot in the build fold.
        if keys.ncols() != x.len() {
            return Err(QueryError::Bind(format!(
                "view query vector has dim {}, key column {key_col} has dim {}",
                x.len(),
                keys.ncols()
            )));
        }
        let dtype = match keys {
            KeyColRef::F64(_) => KeyDtype::F64,
            KeyColRef::F32(_) => KeyDtype::F32,
        };
        let mut v = SoftAggView {
            name: name.into(),
            table: table_name.into(),
            group_col: group_col.into(),
            val_col: val_col.into(),
            key_col: key_col.into(),
            param_fp: param_fingerprint(x),
            eps,
            query: x.to_vec(),
            query32: match dtype {
                KeyDtype::F64 => Vec::new(),
                KeyDtype::F32 => x.iter().map(|&q| q as f32).collect(),
            },
            dtype,
            groups: vec![GroupState::empty(); dict.len()],
            n_reanchors: 0,
        };
        match keys {
            KeyColRef::F64(a) => {
                for r in 0..codes.len() {
                    let s = a.row(r).dot(x);
                    v.apply_insert(codes[r] as usize, s, vals[r]);
                }
            }
            KeyColRef::F32(a) => {
                // f32 dot over the stored rows, widened once per row —
                // sequential accumulation, the same order score_slice
                // uses on delta rows, so incremental maintenance and
                // rebuild score bit-identically.
                for r in 0..codes.len() {
                    let mut acc = 0.0f32;
                    for (kv, qv) in a.row(r).iter().zip(&v.query32) {
                        acc += kv * qv;
                    }
                    v.apply_insert(codes[r] as usize, acc as f64, vals[r]);
                }
            }
        }
        Ok(v)
    }

    fn score(&self, key_row: &[f64]) -> f64 {
        score_slice(self.dtype, &self.query, &self.query32, key_row)
    }

    fn apply_insert(&mut self, g: usize, s: f64, val: f64) {
        if g >= self.groups.len() {
            self.groups.resize(g + 1, GroupState::empty());
        }
        let st = &mut self.groups[g];
        if s > st.m {
            if st.m.is_finite() {
                let f = ((st.m - s) / self.eps).exp();
                st.num *= f;
                st.den *= f;
            }
            st.m = s;
        }
        let w = ((s - st.m) / self.eps).exp();
        st.num += w * val;
        st.den += w;
        st.n += 1;
    }

    /// Apply an insert delta (row already appended to the base table).
    pub fn on_insert(&mut self, g: usize, key_row: &[f64], val: f64) {
        let s = self.score(key_row);
        self.apply_insert(g, s, val);
    }

    /// Apply a delete delta. `survivors` iterates the group's
    /// surviving `(key_row, val)` pairs and is consulted only when the
    /// deleted score ties the group's anchor (bounded re-anchor).
    ///
    /// Key rows are wire-format f64 (`RowValues`); f32 views score
    /// them through the cast-down contract of [`score_slice`], both
    /// for the deleted row and for every survivor in the re-anchor
    /// pass. db.rs's `delete_where` feeds this method through
    /// `key_rows_f64`, which widens a KeyF32 column into that wire
    /// format exactly (f32 -> f64 -> f32 round trips bit-for-bit), so
    /// the whole delete path is dtype-complete as of 2026-08-03.
    pub fn on_delete<'a, I>(&mut self, g: usize, key_row: &[f64], val: f64, survivors: I)
    where
        I: IntoIterator<Item = (&'a [f64], f64)>,
    {
        let s = self.score(key_row);
        let eps = self.eps;
        // Guard (tests/error_totality.rs): a group code beyond the
        // view's state (possible only when the catalog's pub fields
        // were mutated to violate the DictU32 code invariant) must not
        // panic. Mirroring apply_insert's resize leaves the phantom
        // group with den <= 0, which read() filters out.
        if g >= self.groups.len() {
            self.groups.resize(g + 1, GroupState::empty());
        }
        let st = &mut self.groups[g];
        let w = ((s - st.m) / eps).exp();
        st.num -= w * val;
        st.den -= w;
        st.n = st.n.saturating_sub(1);
        if s >= st.m - 1e-12 {
            // deleted (one of) the anchor scorers: one bounded pass
            self.n_reanchors += 1;
            let q = self.query.clone();
            let q32 = self.query32.clone();
            let dtype = self.dtype;
            let st = &mut self.groups[g];
            *st = GroupState::empty();
            for (k, v) in survivors {
                let s = score_slice(dtype, &q, &q32, k);
                if s > st.m {
                    if st.m.is_finite() {
                        let f = ((st.m - s) / eps).exp();
                        st.num *= f;
                        st.den *= f;
                    }
                    st.m = s;
                }
                let w = ((s - st.m) / eps).exp();
                st.num += w * v;
                st.den += w;
                st.n += 1;
            }
        }
    }

    /// Read the view: (group_code, answer) for covered groups.
    pub fn read(&self) -> Vec<(usize, f64)> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, st)| st.den > 0.0 && st.n > 0)
            .map(|(g, st)| (g, st.num / st.den))
            .collect()
    }

    /// Does this view answer the given query shape?
    pub fn matches(
        &self,
        table: &str,
        group_col: &str,
        val_col: &str,
        key_col: &str,
        param_fp: u64,
        eps: f64,
    ) -> bool {
        self.table == table
            && self.group_col == group_col
            && self.val_col == val_col
            && self.key_col == key_col
            && self.param_fp == param_fp
            && (self.eps - eps).abs() < 1e-15
    }
}

pub(crate) fn dict_col<'a>(
    t: &'a Table,
    name: &str,
) -> Result<(&'a Vec<u32>, &'a Vec<String>), QueryError> {
    match t.columns.get(name) {
        Some(Column::DictU32 { codes, dict }) => Ok((codes, dict)),
        _ => Err(QueryError::Bind(format!("column {name} must be DictU32"))),
    }
}

pub(crate) fn scalar_col<'a>(t: &'a Table, name: &str) -> Result<&'a Vec<f64>, QueryError> {
    match t.columns.get(name) {
        Some(Column::ScalarF64(v)) => Ok(v),
        _ => Err(QueryError::Bind(format!("column {name} must be ScalarF64"))),
    }
}

#[cfg(test)]
mod tests {
    //! f32-view semantics (2026-08-03, f32-tail track): maintenance
    //! over KeyF32 columns — f32 storage/scoring, f64 (m, num, den)
    //! state, group-inverse deletes with bounded re-anchor — must
    //! track a from-scratch rebuild and the f32 kernel itself.

    use super::*;
    use ndarray::{Array1, Array2};

    const D_K: usize = 5;
    const N_GROUPS: usize = 4;
    const EPS: f64 = 0.1;

    /// xorshift64* in [-0.5, 0.5) — the numerical_edges convention.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        }
        fn next_u(&mut self, bound: usize) -> usize {
            (self.next_f64() + 0.5 * 3.0).to_bits() as usize % bound
        }
    }

    /// A live row in wire format: the key holds f32-exact f64 values,
    /// so casting down (score_slice) recovers the stored f32 bits.
    #[derive(Clone)]
    struct Row {
        g: usize,
        key: Vec<f64>,
        val: f64,
    }

    fn random_row(rng: &mut Rng) -> Row {
        Row {
            g: rng.next_u(N_GROUPS),
            key: (0..D_K).map(|_| (rng.next_f64() as f32) as f64).collect(),
            val: rng.next_f64() + 2.0, // offset: rel err well-defined
        }
    }

    /// Materialize the live rows as a KeyF32 table.
    fn table_of(rows: &[Row]) -> Table {
        let mut t = Table::default();
        let codes: Vec<u32> = rows.iter().map(|r| r.g as u32).collect();
        let dict: Vec<String> = (0..N_GROUPS).map(|g| format!("g{g}")).collect();
        let vals: Vec<f64> = rows.iter().map(|r| r.val).collect();
        let mut keys = Array2::<f32>::zeros((rows.len(), D_K));
        for (i, r) in rows.iter().enumerate() {
            for (c, &k) in r.key.iter().enumerate() {
                keys[(i, c)] = k as f32;
            }
        }
        t.columns
            .insert("g".into(), Column::DictU32 { codes, dict });
        t.columns.insert("v".into(), Column::ScalarF64(vals));
        t.columns.insert("k".into(), Column::KeyF32(keys));
        t
    }

    fn build_view(rows: &[Row], x: &Array1<f64>) -> SoftAggView {
        SoftAggView::build("v", "t", &table_of(rows), "g", "v", "k", &x.view(), EPS).unwrap()
    }

    fn assert_view_close(a: &SoftAggView, b: &SoftAggView, ctx: &str) {
        let ra = a.read();
        let rb = b.read();
        let ga: Vec<usize> = ra.iter().map(|&(g, _)| g).collect();
        let gb: Vec<usize> = rb.iter().map(|&(g, _)| g).collect();
        assert_eq!(ga, gb, "{ctx}: covered groups diverge");
        for (&(g, va), &(_, vb)) in ra.iter().zip(rb.iter()) {
            let rel = (va - vb).abs() / vb.abs();
            assert!(
                rel <= 1e-4,
                "{ctx}: group {g} rel err {rel:e} ({va} vs {vb})"
            );
        }
    }

    /// PROPERTY: after every prefix of a random insert/delete stream,
    /// the incrementally maintained f32 view equals a from-scratch
    /// rebuild over the surviving rows within rel 1e-4 (the f32
    /// scoring budget at eps = 0.1; scoring itself is bit-identical
    /// between the two, so the slack only covers (m, num, den)
    /// non-associativity).
    #[test]
    fn f32_view_maintenance_matches_rebuild() {
        for seed in [3u64, 0xBADD, 0xC0FFEE] {
            let mut rng = Rng(seed | 1);
            let x = Array1::from_shape_fn(D_K, |_| rng.next_f64());
            let mut live: Vec<Row> = (0..40).map(|_| random_row(&mut rng)).collect();
            let mut view = build_view(&live, &x);
            for op in 0..80 {
                if live.is_empty() || rng.next_u(2) == 0 {
                    let r = random_row(&mut rng);
                    view.on_insert(r.g, &r.key, r.val);
                    live.push(r);
                } else {
                    let victim = live.swap_remove(rng.next_u(live.len()));
                    let survivors: Vec<(&[f64], f64)> = live
                        .iter()
                        .filter(|r| r.g == victim.g)
                        .map(|r| (r.key.as_slice(), r.val))
                        .collect();
                    view.on_delete(victim.g, &victim.key, victim.val, survivors);
                }
                assert_view_close(
                    &view,
                    &build_view(&live, &x),
                    &format!("seed {seed} op {op}"),
                );
            }
        }
    }

    /// The built f32 view must agree with the f32 kernel
    /// (`grouped_softavg_f32`) on the same stored numbers. The kernel
    /// dots with 4-way-unrolled partial sums, the view sequentially —
    /// both f32 — so answers match within the same rel 1e-4 budget.
    #[test]
    fn f32_view_matches_f32_kernel() {
        let mut rng = Rng(0xD1CE);
        let x = Array1::from_shape_fn(D_K, |_| rng.next_f64());
        let rows: Vec<Row> = (0..300).map(|_| random_row(&mut rng)).collect();
        let view = build_view(&rows, &x);

        let x32 = Array1::from_shape_fn(D_K, |c| x[c] as f32);
        let k32 = Array2::from_shape_fn((rows.len(), D_K), |(r, c)| rows[r].key[c] as f32);
        let vals = Array2::from_shape_fn((rows.len(), 1), |(r, _)| rows[r].val);
        let gid: Vec<u32> = rows.iter().map(|r| r.g as u32).collect();
        let (want, covered) = bruce_core::mask::grouped_softavg_f32(
            &x32.view(),
            &k32.view(),
            &vals.view(),
            &gid,
            N_GROUPS,
            None,
            bruce_core::Eps::new(EPS).unwrap(),
        )
        .unwrap();

        let got = view.read();
        let covered_groups: Vec<usize> = (0..N_GROUPS).filter(|&g| covered[g]).collect();
        assert_eq!(
            got.iter().map(|&(g, _)| g).collect::<Vec<_>>(),
            covered_groups
        );
        for &(g, v) in &got {
            let rel = (v - want[(g, 0)]).abs() / want[(g, 0)].abs();
            assert!(rel <= 1e-4, "group {g}: rel err {rel:e}");
        }
    }

    /// Deleting a group's last row through the group-inverse path must
    /// leave the group uncovered (empty re-anchor), not NaN.
    #[test]
    fn f32_view_group_empties_cleanly() {
        let mut rng = Rng(7);
        let x = Array1::from_shape_fn(D_K, |_| rng.next_f64());
        let mut live: Vec<Row> = (0..6).map(|_| random_row(&mut rng)).collect();
        live[0].g = 3; // ensure group 3 has at least one row
        let mut view = build_view(&live, &x);
        while let Some(pos) = live.iter().position(|r| r.g == 3) {
            let victim = live.swap_remove(pos);
            let survivors: Vec<(&[f64], f64)> = live
                .iter()
                .filter(|r| r.g == 3)
                .map(|r| (r.key.as_slice(), r.val))
                .collect();
            view.on_delete(3, &victim.key, victim.val, survivors);
        }
        assert!(view.read().iter().all(|&(g, v)| g != 3 && v.is_finite()));
        assert_view_close(&view, &build_view(&live, &x), "group 3 emptied");
    }
}

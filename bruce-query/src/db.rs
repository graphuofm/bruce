//! The Database facade: catalog + statistics + maintained views +
//! the write path, behind one API. This is what the storage
//! milestone grows into an engine; the interfaces are already the
//! engine's.
//!
//! Write-path contract:
//!   insert_row  appends to every column, updates every matching
//!               view incrementally, marks stats stale.
//!   delete_where captures the doomed rows, compacts the columns,
//!               applies per-group view deltas (one bounded re-anchor
//!               per group that lost an anchor scorer), marks stats
//!               stale.
//!   Stats refresh lazily on the next plan() call.

use std::collections::HashMap;

use bruce_core::hnsw::HnswIndex;
use ndarray::{Array1, Array2, Axis};

use crate::catalog::{Catalog, Column, Table};
use crate::cost::CostModel;
use crate::exec::{execute_with_indexes, GroupResult};
use crate::logical::Pred;
use crate::planner::{plan_with_indexes, PlannedQuery};
use crate::stats::TableStats;
use crate::views::SoftAggView;
use crate::QueryError;

/// A registered HNSW index over one KeyF64 column, plus the id<->row
/// bookkeeping that keeps index results addressable after the write
/// path compacts the columns.
///
/// Identity: one index per (table, key_col). External ids are stable
/// across deletes (tombstones, hnsw.rs routable-until-compact
/// contract); row positions are NOT — `delete_where` compacts the
/// columns — so the entry maintains `row_ids` (current row position
/// -> external id) and `id_to_row` (external id -> current row
/// position, `TOMBSTONE_ROW` once deleted), refreshed in the same
/// O(n) pass the delete already pays.
#[derive(Debug, Clone)]
pub struct HnswIndexEntry {
    /// Indexed table.
    pub table: String,
    /// Indexed key column (KeyF64; the graph stores f32 casts — the
    /// index is a candidate enumerator, exact rescoring stays f64).
    pub key_col: String,
    /// The graph.
    pub index: HnswIndex,
    /// Current row position -> external id.
    row_ids: Vec<u32>,
    /// External id -> current row position (`TOMBSTONE_ROW` = deleted).
    id_to_row: Vec<u32>,
}

/// Sentinel row position for tombstoned external ids.
const TOMBSTONE_ROW: u32 = u32::MAX;

impl HnswIndexEntry {
    /// Fraction of ever-inserted nodes that are tombstoned — the
    /// caller's rebuild signal (hnsw.rs performs no graph repair; a
    /// rebuild is `create_index` after dropping, in v1 simply
    /// re-registering or re-creating).
    pub fn tombstone_fraction(&self) -> f64 {
        self.index.tombstone_fraction()
    }

    /// Live vectors in the index.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True when no live vector remains.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Number of table rows the index currently maps (the executor's
    /// out-of-sync guard compares this against the table).
    pub fn row_len(&self) -> usize {
        self.row_ids.len()
    }

    /// Current row position of an external id (None once tombstoned).
    pub fn row_of(&self, id: u32) -> Option<usize> {
        match self.id_to_row.get(id as usize) {
            Some(&r) if r != TOMBSTONE_ROW => Some(r as usize),
            _ => None,
        }
    }
}

/// One row of values for the write path, keyed by column name.
#[derive(Debug, Clone, Default)]
pub struct RowValues {
    /// Scalar column values.
    pub scalars: HashMap<String, f64>,
    /// Dictionary column labels.
    pub labels: HashMap<String, String>,
    /// Key column vectors.
    pub keys: HashMap<String, Vec<f64>>,
}

/// The database: tables, stats, views, cost model.
pub struct Database {
    /// The catalog.
    pub catalog: Catalog,
    /// Per-table statistics (lazily refreshed after writes).
    pub stats: HashMap<String, TableStats>,
    /// Registered maintained views.
    pub views: Vec<SoftAggView>,
    /// Registered HNSW indexes.
    pub indexes: Vec<HnswIndexEntry>,
    /// The calibrated cost model.
    pub model: CostModel,
    /// Key-sketch sample size for stats collection.
    pub stats_sample: usize,
    stale: std::collections::HashSet<String>,
}

impl Database {
    /// Empty database with the default calibrated cost model.
    pub fn new() -> Self {
        Database {
            catalog: Catalog::new(),
            stats: HashMap::new(),
            views: Vec::new(),
            indexes: Vec::new(),
            model: CostModel::default(),
            stats_sample: 1024,
            stale: Default::default(),
        }
    }

    /// Register a table and collect its statistics.
    ///
    /// Defined semantics (tests/error_totality.rs): registering over
    /// an existing name is CREATE OR REPLACE — the table and its
    /// statistics are replaced, and maintained views built on the old
    /// contents are DROPPED (PG vocabulary: replace = drop + create,
    /// and dropping a table cascades to its dependent views). Stale
    /// view state must never serve answers for the new table.
    /// HNSW indexes cascade identically (tests/topk_access_path.rs):
    /// an index over the old contents would serve stale candidate
    /// rows for the new table, so replace drops it — the caller
    /// re-creates with `create_index` if wanted.
    pub fn register(&mut self, name: &str, table: Table) {
        if self.catalog.tables.contains_key(name) {
            self.views.retain(|v| v.table != name);
            self.indexes.retain(|i| i.table != name);
        }
        let stats = collect_stats(&table, self.stats_sample);
        self.catalog.register(name, table);
        self.stats.insert(name.to_string(), stats);
        self.stale.remove(name);
    }

    fn ensure_stats(&mut self, table: &str) {
        if self.stale.remove(table) {
            if let Some(t) = self.catalog.tables.get(table) {
                self.stats
                    .insert(table.to_string(), collect_stats(t, self.stats_sample));
            }
        }
    }

    /// Create and register a maintained soft-aggregate view.
    ///
    /// Defined semantics (tests/error_totality.rs): view names are
    /// unique per database — creating a second view under an existing
    /// name is an error (PG: duplicate CREATE VIEW errors), because
    /// `MaintainedViewScan` resolves views by name.
    #[allow(clippy::too_many_arguments)]
    pub fn create_view(
        &mut self,
        name: &str,
        table: &str,
        group_col: &str,
        val_col: &str,
        key_col: &str,
        x: &Array1<f64>,
        eps: f64,
    ) -> Result<(), QueryError> {
        if self.views.iter().any(|v| v.name == name) {
            return Err(QueryError::Bind(format!("view {name} already exists")));
        }
        let t = self
            .catalog
            .tables
            .get(table)
            .ok_or_else(|| QueryError::Bind(format!("no table {table}")))?;
        let v = SoftAggView::build(name, table, t, group_col, val_col, key_col, &x.view(), eps)?;
        self.views.push(v);
        Ok(())
    }

    /// Create and register an HNSW index over a KeyF64 column.
    ///
    /// Defined semantics (tests/topk_access_path.rs):
    /// - one index per (table, key_col); a duplicate CREATE INDEX is
    ///   an error (PG: duplicate index name errors);
    /// - the column must be KeyF64 — KeyF32 storage takes a typed
    ///   error for now; the graph itself stores f32 casts either way,
    ///   the restriction is about the exact-rescore contract
    ///   (`HnswTopKScan` rescores its probe hits in f64, and exec.rs
    ///   refuses a KeyF32 column for the same reason). The refusal is
    ///   an EXPLICIT dtype check here, not a side effect of a narrow
    ///   accessor: `delete_where` below reads key columns
    ///   dtype-polymorphically, so nothing else pins it;
    /// - external ids are assigned by row position at build time and
    ///   stay stable across later writes (see `HnswIndexEntry`).
    pub fn create_index(&mut self, table: &str, key_col: &str) -> Result<(), QueryError> {
        if self
            .indexes
            .iter()
            .any(|i| i.table == table && i.key_col == key_col)
        {
            return Err(QueryError::Bind(format!(
                "index on {table}.{key_col} already exists"
            )));
        }
        let t = self
            .catalog
            .tables
            .get(table)
            .ok_or_else(|| QueryError::Bind(format!("no table {table}")))?;
        // Explicit v1 dtype refusal (tests/topk_access_path.rs
        // `create_index_typed_errors`): index plans rescore in f64.
        let keys = match t.columns.get(key_col) {
            Some(Column::KeyF64(a)) => a,
            Some(Column::KeyF32(_)) => {
                return Err(QueryError::Bind(format!(
                    "index on {table}.{key_col}: column must be KeyF64 \
                     (index plans rescore probe hits in f64; KeyF32 indexing \
                     arrives with the f32 rescore contract)"
                )))
            }
            _ => return Err(QueryError::Bind(format!("column {key_col} must be KeyF64"))),
        };
        let n = keys.nrows();
        if n > u32::MAX as usize {
            return Err(QueryError::Bind(format!(
                "index on {table}.{key_col}: {n} rows exceed the u32 id space"
            )));
        }
        let mut index = HnswIndex::with_defaults(keys.ncols());
        let mut buf = vec![0.0f32; keys.ncols()];
        for r in 0..n {
            for (b, &v) in buf.iter_mut().zip(keys.row(r).iter()) {
                *b = v as f32;
            }
            index
                .insert(r as u32, &buf)
                .map_err(|e| QueryError::Exec(format!("index build: {e}")))?;
        }
        self.indexes.push(HnswIndexEntry {
            table: table.into(),
            key_col: key_col.into(),
            index,
            row_ids: (0..n as u32).collect(),
            id_to_row: (0..n as u32).collect(),
        });
        Ok(())
    }

    /// Parse, optimize, plan, and execute one SQL query.
    pub fn run(
        &mut self,
        sql: &str,
        params: &HashMap<String, Array1<f64>>,
    ) -> Result<(GroupResult, PlannedQuery), QueryError> {
        let logical = crate::parse::parse_query(sql)?;
        let logical = crate::optimizer::optimize(logical);
        let table = table_of(&logical)?;
        self.validate_run(&logical, &table, params)?;
        self.ensure_stats(&table);
        let stats = self
            .stats
            .get(&table)
            .ok_or_else(|| QueryError::Bind(format!("no stats for {table}")))?;
        let planned = plan_with_indexes(
            &logical,
            stats,
            &self.views,
            &self.indexes,
            params,
            &self.model,
        )?;
        let result = execute_with_indexes(
            &planned.chosen,
            &self.catalog,
            params,
            &self.views,
            &self.indexes,
        )?;
        Ok((result, planned))
    }

    /// Pre-execution guard (tests/error_totality.rs): reject the two
    /// input classes that previously panicked deeper in the pipeline
    /// (the stats sketch and the executor live in modules owned by
    /// other tracks, so the guard sits at this facade):
    ///   1. a bound query vector whose dimension mismatches the key
    ///      column (ndarray's dot asserts shapes inside the sketch
    ///      estimator when a budget is declared);
    ///   2. dict codes beyond the dictionary — a catalog-invariant
    ///      violation reachable through the pub catalog fields — which
    ///      indexed out of bounds in the ExactGroupAvg fold.
    ///
    /// Everything else keeps flowing to the existing typed error
    /// paths in planner/executor.
    fn validate_run(
        &self,
        logical: &crate::logical::LogicalPlan,
        table: &str,
        params: &HashMap<String, Array1<f64>>,
    ) -> Result<(), QueryError> {
        use crate::logical::LogicalPlan as L;
        let Some(t) = self.catalog.tables.get(table) else {
            return Ok(()); // missing table: the stats lookup errors
        };
        fn walk(p: &L) -> (Option<&str>, Option<&crate::logical::ScoreExpr>) {
            match p {
                L::SoftAgg {
                    group_col, score, ..
                } => (Some(group_col), Some(score)),
                L::PlainGroupAvg { group_col, .. } => (Some(group_col), None),
                L::Filter { input, .. } => walk(input),
                L::Scan { .. } => (None, None),
            }
        }
        let (group_col, score) = walk(logical);
        if let Some(g) = group_col {
            if let Some(Column::DictU32 { codes, dict }) = t.columns.get(g) {
                if let Some(&mx) = codes.iter().max() {
                    if mx as usize >= dict.len() {
                        return Err(QueryError::Exec(format!(
                            "group column {g} is corrupt: code {mx} exceeds \
                             dictionary of {} entries",
                            dict.len()
                        )));
                    }
                }
            }
        }
        if let Some(s) = score {
            if let Some(x) = params.get(&s.param) {
                let d = match t.columns.get(&s.key_col) {
                    Some(Column::KeyF64(a)) => Some(a.ncols()),
                    Some(Column::KeyF32(a)) => Some(a.ncols()),
                    _ => None, // missing/ill-kinded key col: executor errors
                };
                if let Some(d) = d {
                    if d != x.len() {
                        return Err(QueryError::Bind(format!(
                            "parameter :{} has dim {}, key column {} has dim {d}",
                            s.param,
                            x.len(),
                            s.key_col
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Append one row; update views incrementally; mark stats stale.
    pub fn insert_row(&mut self, table: &str, row: &RowValues) -> Result<(), QueryError> {
        let t = self
            .catalog
            .tables
            .get_mut(table)
            .ok_or_else(|| QueryError::Bind(format!("no table {table}")))?;
        // Defined semantics (tests/error_totality.rs): a row naming a
        // column the table does not have, or one of another kind, is
        // an error — PG's INSERT rejects unknown target columns rather
        // than silently dropping them (typo protection).
        for name in row.scalars.keys() {
            if !matches!(t.columns.get(name), Some(Column::ScalarF64(_))) {
                return Err(QueryError::Bind(format!(
                    "insert names unknown scalar column {name}"
                )));
            }
        }
        for name in row.labels.keys() {
            if !matches!(t.columns.get(name), Some(Column::DictU32 { .. })) {
                return Err(QueryError::Bind(format!(
                    "insert names unknown dict column {name}"
                )));
            }
        }
        for name in row.keys.keys() {
            if !matches!(
                t.columns.get(name),
                Some(Column::KeyF64(_)) | Some(Column::KeyF32(_))
            ) {
                return Err(QueryError::Bind(format!(
                    "insert names unknown key column {name}"
                )));
            }
        }
        // validate first: every column must be covered
        for (name, col) in &t.columns {
            let ok = match col {
                Column::ScalarF64(_) => row.scalars.contains_key(name),
                Column::DictU32 { .. } => row.labels.contains_key(name),
                Column::KeyF64(a) => row
                    .keys
                    .get(name)
                    .map(|k| k.len() == a.ncols())
                    .unwrap_or(false),
                Column::KeyF32(a) => row
                    .keys
                    .get(name)
                    .map(|k| k.len() == a.ncols())
                    .unwrap_or(false),
            };
            if !ok {
                return Err(QueryError::Bind(format!(
                    "insert missing/ill-typed column {name}"
                )));
            }
        }
        // append; remember dict codes for the view deltas
        let mut codes_of: HashMap<String, usize> = HashMap::new();
        for (name, col) in t.columns.iter_mut() {
            match col {
                Column::ScalarF64(v) => v.push(row.scalars[name]),
                Column::DictU32 { codes, dict } => {
                    let label = &row.labels[name];
                    let code = match dict.iter().position(|l| l == label) {
                        Some(c) => c,
                        None => {
                            dict.push(label.clone());
                            dict.len() - 1
                        }
                    };
                    codes.push(code as u32);
                    codes_of.insert(name.clone(), code);
                }
                Column::KeyF64(a) => {
                    let r = Array1::from_vec(row.keys[name].clone());
                    a.push(Axis(0), r.view())
                        .map_err(|e| QueryError::Exec(format!("append: {e}")))?;
                }
                Column::KeyF32(a) => {
                    // RowValues carries f64 vectors; storage is f32 —
                    // cast down on append, matching the column dtype
                    let r: Array1<f32> = row.keys[name].iter().map(|&v| v as f32).collect();
                    a.push(Axis(0), r.view())
                        .map_err(|e| QueryError::Exec(format!("append: {e}")))?;
                }
            }
        }
        // view deltas
        for v in self.views.iter_mut().filter(|v| v.table == table) {
            let g = *codes_of.get(&v.group_col).ok_or_else(|| {
                QueryError::Bind(format!("view group col {} missing", v.group_col))
            })?;
            let key = &row.keys[&v.key_col];
            let val = row.scalars[&v.val_col];
            v.on_insert(g, key, val);
        }
        // index deltas: hnsw.rs supports incremental insert (mid-build
        // searches are exact per its suite), so the graph is extended
        // in place — no staleness marker needed. Fresh external id =
        // next id ever assigned; new row position = old row count.
        for ix in self.indexes.iter_mut().filter(|i| i.table == table) {
            let key = &row.keys[&ix.key_col]; // covered: validated above
            let id = ix.id_to_row.len();
            if id > u32::MAX as usize - 1 {
                return Err(QueryError::Exec(format!(
                    "index on {}.{}: u32 id space exhausted",
                    ix.table, ix.key_col
                )));
            }
            let k32: Vec<f32> = key.iter().map(|&v| v as f32).collect();
            ix.index
                .insert(id as u32, &k32)
                .map_err(|e| QueryError::Exec(format!("index insert: {e}")))?;
            ix.id_to_row.push(ix.row_ids.len() as u32);
            ix.row_ids.push(id as u32);
        }
        self.stale.insert(table.to_string());
        Ok(())
    }

    /// Delete all rows matching the predicate. Returns rows removed.
    ///
    /// Dtype-complete over key columns (2026-08-03, hnsw-finish
    /// track): the per-view survivor capture reads keys through
    /// [`key_rows_f64`], so a table carrying an f32-KEYED maintained
    /// view deletes like any other. Previously this call site used the
    /// KeyF64-only `views::key_col_of` and returned a typed error —
    /// the gap pinned in tests/error_totality.rs
    /// (`create_view_f32_semantics`) and bruce-py's
    /// tests/test_f32_views.py, both now flipped positive.
    pub fn delete_where(&mut self, table: &str, pred: &Pred) -> Result<usize, QueryError> {
        let t = self
            .catalog
            .tables
            .get_mut(table)
            .ok_or_else(|| QueryError::Bind(format!("no table {table}")))?;
        let mask = crate::exec::eval_pred(pred, t)?; // true = doomed
        let n_doomed = mask.iter().filter(|&&d| d).count();
        if n_doomed == 0 {
            return Ok(0);
        }

        // capture per-view deltas BEFORE compaction
        struct Doomed {
            group: usize,
            key: Vec<f64>,
            val: f64,
        }
        let mut per_view: Vec<Vec<Doomed>> = Vec::new();
        for v in self.views.iter().filter(|v| v.table == table) {
            let (codes, _) = crate::views::dict_col(t, &v.group_col)?;
            let vals = crate::views::scalar_col(t, &v.val_col)?;
            let keys = key_rows_f64(t, &v.key_col)?;
            per_view.push(
                mask.iter()
                    .enumerate()
                    .filter(|(_, &d)| d)
                    .map(|(r, _)| Doomed {
                        group: codes[r] as usize,
                        key: keys.row(r).to_vec(),
                        val: vals[r],
                    })
                    .collect(),
            );
        }

        // compact every column
        for col in t.columns.values_mut() {
            match col {
                Column::ScalarF64(v) => {
                    let mut i = 0;
                    v.retain(|_| {
                        let keep = !mask[i];
                        i += 1;
                        keep
                    });
                }
                Column::DictU32 { codes, .. } => {
                    let mut i = 0;
                    codes.retain(|_| {
                        let keep = !mask[i];
                        i += 1;
                        keep
                    });
                }
                Column::KeyF64(a) => {
                    let keep_rows: Vec<usize> = (0..a.nrows()).filter(|&r| !mask[r]).collect();
                    let mut b = Array2::<f64>::zeros((keep_rows.len(), a.ncols()));
                    for (i, &r) in keep_rows.iter().enumerate() {
                        b.row_mut(i).assign(&a.row(r));
                    }
                    *col = Column::KeyF64(b);
                }
                Column::KeyF32(a) => {
                    let keep_rows: Vec<usize> = (0..a.nrows()).filter(|&r| !mask[r]).collect();
                    let mut b = ndarray::Array2::<f32>::zeros((keep_rows.len(), a.ncols()));
                    for (i, &r) in keep_rows.iter().enumerate() {
                        b.row_mut(i).assign(&a.row(r));
                    }
                    *col = Column::KeyF32(b);
                }
            }
        }

        // apply view deltas against the POST-delete table (survivors)
        let t = self.catalog.tables.get(table).unwrap();
        for (vi, doomed) in self
            .views
            .iter_mut()
            .filter(|v| v.table == table)
            .zip(per_view)
        {
            let (codes, _) = crate::views::dict_col(t, &vi.group_col)?;
            let vals = crate::views::scalar_col(t, &vi.val_col)?;
            let keys = key_rows_f64(t, &vi.key_col)?;
            // Defect fix (tests/stateful_writes.rs): once a group has
            // re-anchored, its state is already the exact fold over
            // the POST-delete survivors — which exclude EVERY doomed
            // row of this delete_where call. Subtracting a later
            // doomed row of the same group would remove it twice
            // (observed as view drift when one call deletes a group's
            // anchor scorer plus another of its rows). Re-anchors are
            // detected via the view's n_reanchors counter; settled
            // groups skip the remaining deltas.
            let mut settled: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for d in doomed {
                if settled.contains(&d.group) {
                    continue;
                }
                let reanchors_before = vi.n_reanchors;
                let survivors = codes
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c as usize == d.group)
                    .map(|(r, _)| (keys.row(r).to_slice().expect("contiguous key row"), vals[r]));
                vi.on_delete(d.group, &d.key, d.val, survivors);
                if vi.n_reanchors > reanchors_before {
                    settled.insert(d.group);
                }
            }
        }

        // index deltas: tombstone doomed ids (excluded from results
        // immediately, still routable — hnsw.rs delete contract), then
        // re-map surviving ids to their post-compaction row positions
        // in the same O(n) the delete already paid. Callers watch
        // `tombstone_fraction()` for the rebuild signal.
        for ix in self.indexes.iter_mut().filter(|i| i.table == table) {
            // Totality guard: the pub catalog fields allow mutating
            // the table behind the index; a length mismatch must be a
            // typed error, not an out-of-bounds panic.
            if ix.row_ids.len() != mask.len() {
                return Err(QueryError::Exec(format!(
                    "index on {}.{} is out of sync: {} indexed rows vs {} table rows \
                     (catalog mutated behind the index; re-register or re-create)",
                    ix.table,
                    ix.key_col,
                    ix.row_ids.len(),
                    mask.len()
                )));
            }
            for (r, &doomed) in mask.iter().enumerate() {
                if doomed {
                    let id = ix.row_ids[r];
                    ix.index
                        .delete(id)
                        .map_err(|e| QueryError::Exec(format!("index delete: {e}")))?;
                    ix.id_to_row[id as usize] = TOMBSTONE_ROW;
                }
            }
            let mut r = 0;
            ix.row_ids.retain(|_| {
                let keep = !mask[r];
                r += 1;
                keep
            });
            for (pos, &id) in ix.row_ids.iter().enumerate() {
                ix.id_to_row[id as usize] = pos as u32;
            }
        }
        self.stale.insert(table.to_string());
        Ok(n_doomed)
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

/// Dtype-polymorphic read of a key column as f64 rows — the write
/// path's survivor/doomed capture, which must hand `SoftAggView` the
/// wire format (`RowValues`-style `&[f64]`) whatever the storage dtype
/// is.
///
/// KeyF64 borrows (zero copy). KeyF32 widens the column once into an
/// owned f64 matrix; the round trip is EXACT — `views::score_slice`
/// casts every component back down to f32 before dotting for an f32
/// view, recovering the stored bits — so an f32 view's delta scoring
/// is bit-identical to a from-scratch rebuild over the stored column
/// (pinned in tests/error_totality.rs `create_view_f32_semantics`).
///
/// Cost the caller owns: the KeyF32 arm allocates n*d*8 B per f32
/// view per `delete_where` call. Deliberate — it keeps the survivor
/// iterator lazy (the re-anchor pass only walks one group), and it is
/// paid only by tables that actually carry an f32-keyed view. The
/// KeyF64 path, and every table with no matching view, allocate
/// nothing.
fn key_rows_f64<'a>(
    t: &'a Table,
    name: &str,
) -> Result<std::borrow::Cow<'a, Array2<f64>>, QueryError> {
    match t.columns.get(name) {
        Some(Column::KeyF64(a)) => Ok(std::borrow::Cow::Borrowed(a)),
        Some(Column::KeyF32(a)) => Ok(std::borrow::Cow::Owned(a.mapv(f64::from))),
        _ => Err(QueryError::Bind(format!(
            "column {name} must be KeyF64 or KeyF32"
        ))),
    }
}

/// Stats collection with an empty-table guard (tests/
/// error_totality.rs): `TableStats::collect`'s key-sketch sampler
/// underflows on a 0-row key column (`r.min(n - 1)` with n = 0;
/// stats.rs is owned by another track, so the guard lives at this
/// facade's only two call sites). An empty table gets default stats
/// (n_rows = 0, no sketches) — every plan over it returns an empty
/// covered set, so estimate quality is moot.
fn collect_stats(table: &Table, sample: usize) -> TableStats {
    let n_rows = table.columns.values().next().map(|c| c.len()).unwrap_or(0);
    if n_rows == 0 {
        return TableStats::default();
    }
    TableStats::collect(table, sample)
}

fn table_of(plan: &crate::logical::LogicalPlan) -> Result<String, QueryError> {
    use crate::logical::LogicalPlan as L;
    fn walk(p: &crate::logical::LogicalPlan) -> Option<String> {
        match p {
            L::Scan { table } => Some(table.clone()),
            L::Filter { input, .. } => walk(input),
            L::SoftAgg { input, .. } => walk(input),
            L::PlainGroupAvg { input, .. } => walk(input),
        }
    }
    walk(plan).ok_or_else(|| QueryError::Bind("plan has no scan".into()))
}

//! Minimal in-memory catalog + columnar table (v1). The storage
//! milestone replaces the backing arrays with segments; the
//! interfaces here are what the planner and executor consume.

use std::collections::HashMap;

use ndarray::Array2;

/// One column of a table.
#[derive(Debug, Clone)]
pub enum Column {
    /// f64 scalar column (values being aggregated, filter columns).
    ScalarF64(Vec<f64>),
    /// f64 key (embedding) rows, shape `(n_rows, d_k)`.
    KeyF64(Array2<f64>),
    /// f32 key (embedding) rows, shape `(n_rows, d_k)` — the storage
    /// dtype encoders emit. Scored by the f32 kernel, accumulated in
    /// f64 (bruce-core mask.rs precision contract).
    KeyF32(Array2<f32>),
    /// Dictionary-encoded group column: codes + dictionary.
    DictU32 {
        /// Per-row code in `[0, dict.len())`.
        codes: Vec<u32>,
        /// Code -> label.
        dict: Vec<String>,
    },
}

impl Column {
    /// Number of rows in the column.
    pub fn len(&self) -> usize {
        match self {
            Column::ScalarF64(v) => v.len(),
            Column::KeyF64(a) => a.nrows(),
            Column::KeyF32(a) => a.nrows(),
            Column::DictU32 { codes, .. } => codes.len(),
        }
    }

    /// True iff the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A table: named columns of equal length.
#[derive(Debug, Clone, Default)]
pub struct Table {
    /// Column name -> column.
    pub columns: HashMap<String, Column>,
}

/// The catalog: table name -> table.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    /// Tables by name.
    pub tables: HashMap<String, Table>,
}

impl Catalog {
    /// Empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a table.
    pub fn register(&mut self, name: &str, table: Table) {
        self.tables.insert(name.to_string(), table);
    }
}

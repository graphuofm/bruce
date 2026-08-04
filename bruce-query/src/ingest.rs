//! The data plane: standard-format ingestion into the catalog.
//!
//! Interfaces are deliberately industry-standard (Arrow record
//! batches, Parquet files); bruce invents storage formats only where
//! the eps-algebra needs one (dictionary-encoded group columns, key
//! matrices). String columns are dictionary-encoded at load — the
//! lesson of M1: per-query factorisation is the glue stack's tax.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use ndarray::Array2;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::catalog::{Column, Table};
use crate::QueryError;

impl Table {
    /// Load a Parquet file into a table.
    ///
    /// Column mapping: Utf8 -> `DictU32` (dictionary built at load);
    /// Int32/Int64/Float32/Float64 -> `ScalarF64`. Other types are
    /// skipped (v1). Nulls in string columns map to the dictionary
    /// entry `"(null)"`; numeric nulls map to `f64::NAN`.
    pub fn from_parquet(path: impl AsRef<Path>) -> Result<Self, QueryError> {
        let file = File::open(path.as_ref())
            .map_err(|e| QueryError::Bind(format!("open parquet: {e}")))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| QueryError::Bind(format!("read parquet: {e}")))?;
        // DEFINED SEMANTICS (tests/ingest_robust.rs, C4/PG 42701
        // duplicate_column): duplicate field names in the file schema
        // are rejected up front. Before this check, a same-type
        // duplicate silently appended both columns into one Vec
        // (length 2n — corrupt table) and a mixed-type duplicate hit
        // the `unreachable!()` in the column-variant match below;
        // with unique names, each name maps to exactly one Arrow
        // dtype for the whole file, so those arms are sound.
        {
            let mut seen = std::collections::HashSet::new();
            for f in builder.schema().fields() {
                if !seen.insert(f.name().as_str()) {
                    return Err(QueryError::Bind(format!(
                        "duplicate column name {:?} in parquet schema",
                        f.name()
                    )));
                }
            }
        }
        let reader = builder
            .build()
            .map_err(|e| QueryError::Bind(format!("read parquet: {e}")))?;

        let mut table = Table::default();
        let mut dicts: HashMap<String, (HashMap<String, u32>, Vec<String>)> = HashMap::new();

        for batch in reader {
            let batch = batch.map_err(|e| QueryError::Bind(format!("parquet batch: {e}")))?;
            for (idx, field) in batch.schema().fields().iter().enumerate() {
                let name = field.name().clone();
                let col = batch.column(idx);
                match field.data_type() {
                    DataType::Utf8 => {
                        let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                        let (map, dict) = dicts.entry(name.clone()).or_default();
                        let codes: &mut Vec<u32> = match table
                            .columns
                            .entry(name.clone())
                            .or_insert_with(|| Column::DictU32 {
                                codes: Vec::new(),
                                dict: Vec::new(),
                            }) {
                            Column::DictU32 { codes, .. } => codes,
                            _ => unreachable!(),
                        };
                        for r in 0..arr.len() {
                            let s = if arr.is_null(r) {
                                "(null)"
                            } else {
                                arr.value(r)
                            };
                            let code = *map.entry(s.to_string()).or_insert_with(|| {
                                dict.push(s.to_string());
                                (dict.len() - 1) as u32
                            });
                            codes.push(code);
                        }
                    }
                    DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 => {
                        let vals: &mut Vec<f64> = match table
                            .columns
                            .entry(name.clone())
                            .or_insert_with(|| Column::ScalarF64(Vec::new()))
                        {
                            Column::ScalarF64(v) => v,
                            _ => unreachable!(),
                        };
                        push_numeric(col.as_ref(), field.data_type(), vals);
                    }
                    _ => {} // v1: skip unsupported types
                }
            }
        }

        // install dictionaries
        for (name, (_, dict)) in dicts {
            if let Some(Column::DictU32 { dict: d, .. }) = table.columns.get_mut(&name) {
                *d = dict;
            }
        }
        Ok(table)
    }

    /// Attach an externally produced key (embedding) matrix as a
    /// column — the bridge for encoders that live outside the engine.
    pub fn attach_key_f64(&mut self, name: &str, keys: Array2<f64>) -> Result<(), QueryError> {
        if let Some(c) = self.columns.values().next() {
            if c.len() != keys.nrows() {
                return Err(QueryError::Bind(format!(
                    "key column {name}: {} rows, table has {}",
                    keys.nrows(),
                    c.len()
                )));
            }
        }
        self.columns.insert(name.to_string(), Column::KeyF64(keys));
        Ok(())
    }

    /// Attach an f32 key matrix — encoder outputs at their native
    /// storage dtype (half the scan bytes of `attach_key_f64`; scored
    /// by the f32 kernel, accumulated in f64).
    pub fn attach_key_f32(&mut self, name: &str, keys: Array2<f32>) -> Result<(), QueryError> {
        if let Some(c) = self.columns.values().next() {
            if c.len() != keys.nrows() {
                return Err(QueryError::Bind(format!(
                    "key column {name}: {} rows, table has {}",
                    keys.nrows(),
                    c.len()
                )));
            }
        }
        self.columns.insert(name.to_string(), Column::KeyF32(keys));
        Ok(())
    }
}

fn push_numeric(col: &dyn Array, dt: &DataType, out: &mut Vec<f64>) {
    match dt {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            for r in 0..a.len() {
                out.push(if a.is_null(r) { f64::NAN } else { a.value(r) });
            }
        }
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
            for r in 0..a.len() {
                out.push(if a.is_null(r) {
                    f64::NAN
                } else {
                    a.value(r) as f64
                });
            }
        }
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
            for r in 0..a.len() {
                out.push(if a.is_null(r) {
                    f64::NAN
                } else {
                    a.value(r) as f64
                });
            }
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
            for r in 0..a.len() {
                out.push(if a.is_null(r) {
                    f64::NAN
                } else {
                    a.value(r) as f64
                });
            }
        }
        _ => unreachable!(),
    }
}

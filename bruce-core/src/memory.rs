//! In-memory K/V store backing Bruce CRUD operations.
//!
//! `KvMemory` is the public interface a downstream user (Python, CLI,
//! or another Rust program) interacts with. It stores key vectors,
//! value vectors, owner metadata, and an append-only audit log of
//! every operation.
//!
//! For high-throughput incremental updates (the "exact unlearning"
//! workload), use [`crate::IncrementalMemory`] which maintains the
//! Lemma A accumulator `(m, num, den)` and supports O(d) inserts and
//! deletes for a *fixed query x*.

use crate::error::{BruceError, Result};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, FixedSizeListArray, FixedSizeListBuilder,
    Float64Array, Float64Builder, RecordBatch, StringArray, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// One audit-log entry. Append-only history of every write/delete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Operation: "insert" / "update" / "delete".
    pub op: String,
    /// Fact identifier.
    pub fact_id: String,
    /// Owner that performed the operation.
    pub owner: String,
    /// Wall-clock timestamp (unix seconds).
    pub at: f64,
}

/// One row of the K/V memory.
#[derive(Debug, Clone)]
struct Row {
    k: Array1<f64>,
    v: Array1<f64>,
    owner: String,
    written_at: f64,
    deleted: bool,
}

/// Append-only K/V memory with owner-enforced delete and audit log.
///
/// Designed to be the **CRUD backend** behind the F_ε operator. It
/// supports both ε = 0 reads (exact lookup by id) and ε > 0 reads
/// (top-K similarity search).
pub struct KvMemory {
    d_k: usize,
    d_v: usize,
    rows: AHashMap<String, Row>,
    insertion_order: Vec<String>,
    log: Vec<AuditEntry>,
}

impl KvMemory {
    /// Create an empty memory parametrised by key/value dimensions.
    pub fn new(d_k: usize, d_v: usize) -> Self {
        Self {
            d_k,
            d_v,
            rows: AHashMap::new(),
            insertion_order: Vec::new(),
            log: Vec::new(),
        }
    }

    /// Number of *live* (non-deleted) rows.
    pub fn len_alive(&self) -> usize {
        self.rows.values().filter(|r| !r.deleted).count()
    }

    /// Number of rows including deleted (audit-style total).
    pub fn len_total(&self) -> usize {
        self.rows.len()
    }

    /// Insert or owner-update a record.
    pub fn write(
        &mut self,
        fact_id: &str,
        k: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
        owner: &str,
    ) -> Result<()> {
        if k.len() != self.d_k {
            return Err(BruceError::DimensionMismatch {
                expected: self.d_k,
                got: k.len(),
            });
        }
        if v.len() != self.d_v {
            return Err(BruceError::DimensionMismatch {
                expected: self.d_v,
                got: v.len(),
            });
        }
        // owner enforcement: only the original owner may overwrite an
        // existing alive row
        let op = if let Some(existing) = self.rows.get(fact_id) {
            if !existing.deleted && existing.owner != owner {
                return Err(BruceError::PermissionDenied(
                    fact_id.into(),
                    existing.owner.clone(),
                    owner.into(),
                ));
            }
            "update"
        } else {
            self.insertion_order.push(fact_id.into());
            "insert"
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.rows.insert(
            fact_id.into(),
            Row {
                k: k.to_owned(),
                v: v.to_owned(),
                owner: owner.into(),
                written_at: now,
                deleted: false,
            },
        );
        self.log.push(AuditEntry {
            op: op.into(),
            fact_id: fact_id.into(),
            owner: owner.into(),
            at: now,
        });
        Ok(())
    }

    /// Delete (mark) a record. Only the owner may delete.
    pub fn delete(&mut self, fact_id: &str, owner: &str) -> Result<()> {
        let row = self
            .rows
            .get_mut(fact_id)
            .ok_or_else(|| BruceError::KeyNotFound(fact_id.into()))?;
        if row.owner != owner {
            return Err(BruceError::PermissionDenied(
                fact_id.into(),
                row.owner.clone(),
                owner.into(),
            ));
        }
        if row.deleted {
            return Err(BruceError::KeyNotFound(fact_id.into()));
        }
        row.deleted = true;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.log.push(AuditEntry {
            op: "delete".into(),
            fact_id: fact_id.into(),
            owner: owner.into(),
            at: now,
        });
        Ok(())
    }

    /// Read (ε = 0): exact lookup by id. Returns None if not present
    /// or deleted.
    pub fn read_exact(&self, fact_id: &str) -> Option<(&Array1<f64>, &Array1<f64>)> {
        self.rows
            .get(fact_id)
            .filter(|r| !r.deleted)
            .map(|r| (&r.k, &r.v))
    }

    /// Snapshot all alive (K, V) rows as dense `Array2`s, plus the
    /// matching fact_id list. Used by downstream `F_eps` reads.
    pub fn snapshot_alive(&self) -> (Vec<String>, Array2<f64>, Array2<f64>) {
        let alive: Vec<&String> = self
            .insertion_order
            .iter()
            .filter(|id| self.rows.get(*id).map(|r| !r.deleted).unwrap_or(false))
            .collect();
        let n = alive.len();
        let mut k_mat = Array2::<f64>::zeros((n, self.d_k));
        let mut v_mat = Array2::<f64>::zeros((n, self.d_v));
        for (i, id) in alive.iter().enumerate() {
            let row = &self.rows[*id];
            k_mat.row_mut(i).assign(&row.k);
            v_mat.row_mut(i).assign(&row.v);
        }
        let ids = alive.into_iter().cloned().collect();
        (ids, k_mat, v_mat)
    }

    /// Read the full audit log (append-only).
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.log
    }

    /// Build the Arrow schema that backs the Parquet snapshot.
    ///
    /// One row per `fact_id`; key and value are `FixedSizeList<f64>` of
    /// width `d_k` and `d_v` respectively. Deleted rows are kept (with
    /// `deleted = true`) so a round-trip preserves the audit trail.
    fn arrow_schema(d_k: usize, d_v: usize) -> Arc<Schema> {
        // inner field must be `nullable = true` to match FixedSizeListBuilder default
        let k_field = Arc::new(Field::new("item", DataType::Float64, true));
        let v_field = Arc::new(Field::new("item", DataType::Float64, true));
        Arc::new(Schema::new(vec![
            Field::new("fact_id", DataType::Utf8, false),
            Field::new("k", DataType::FixedSizeList(k_field, d_k as i32), false),
            Field::new("v", DataType::FixedSizeList(v_field, d_v as i32), false),
            Field::new("owner", DataType::Utf8, false),
            Field::new("written_at", DataType::Float64, false),
            Field::new("deleted", DataType::Boolean, false),
        ]))
    }

    /// Snapshot the memory to a Parquet file at `path`.  Schema includes
    /// `fact_id, k, v, owner, written_at, deleted`.  Deleted rows are
    /// preserved so `load_parquet` produces a bit-identical `KvMemory`.
    pub fn save_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        let schema = Self::arrow_schema(self.d_k, self.d_v);

        let mut id_b = StringBuilder::new();
        let k_inner = Float64Builder::new();
        let mut k_b = FixedSizeListBuilder::new(k_inner, self.d_k as i32);
        let v_inner = Float64Builder::new();
        let mut v_b = FixedSizeListBuilder::new(v_inner, self.d_v as i32);
        let mut owner_b = StringBuilder::new();
        let mut at_b = Float64Builder::new();
        let mut del_b = BooleanBuilder::new();

        for id in self.insertion_order.iter() {
            let Some(row) = self.rows.get(id) else {
                continue;
            };
            id_b.append_value(id);
            for v in row.k.iter() {
                k_b.values().append_value(*v);
            }
            k_b.append(true);
            for v in row.v.iter() {
                v_b.values().append_value(*v);
            }
            v_b.append(true);
            owner_b.append_value(&row.owner);
            at_b.append_value(row.written_at);
            del_b.append_value(row.deleted);
        }

        let cols: Vec<ArrayRef> = vec![
            Arc::new(id_b.finish()),
            Arc::new(k_b.finish()),
            Arc::new(v_b.finish()),
            Arc::new(owner_b.finish()),
            Arc::new(at_b.finish()),
            Arc::new(del_b.finish()),
        ];
        let batch = RecordBatch::try_new(schema.clone(), cols)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("arrow: {e}")))?;

        let file = std::fs::File::create(path.as_ref())?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| BruceError::Other(anyhow::anyhow!("parquet writer: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("parquet write: {e}")))?;
        writer
            .close()
            .map_err(|e| BruceError::Other(anyhow::anyhow!("parquet close: {e}")))?;

        // also save audit log as JSONL alongside (path + ".audit.jsonl")
        let audit_path = path.as_ref().with_extension("audit.jsonl");
        let mut f = std::fs::File::create(&audit_path)?;
        use std::io::Write;
        for entry in self.log.iter() {
            let line = serde_json::to_string(entry)
                .map_err(|e| BruceError::Other(anyhow::anyhow!("json: {e}")))?;
            writeln!(f, "{line}")?;
        }
        Ok(())
    }

    /// Load a `KvMemory` snapshot from a Parquet file written by
    /// `save_parquet`.  Restores insertion order, owner metadata, the
    /// deleted-tombstone status, and (if `path + ".audit.jsonl"`
    /// exists) the audit log.
    pub fn load_parquet(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("parquet open: {e}")))?;

        // d_k, d_v from the schema's FixedSizeList width
        let schema = builder.schema().clone();
        let d_k = match schema
            .field_with_name("k")
            .map_err(|e| BruceError::Other(anyhow::anyhow!("schema: {e}")))?
            .data_type()
        {
            DataType::FixedSizeList(_, n) => *n as usize,
            other => {
                return Err(BruceError::Other(anyhow::anyhow!(
                    "expected k as FixedSizeList, got {other:?}"
                )));
            }
        };
        let d_v = match schema
            .field_with_name("v")
            .map_err(|e| BruceError::Other(anyhow::anyhow!("schema: {e}")))?
            .data_type()
        {
            DataType::FixedSizeList(_, n) => *n as usize,
            other => {
                return Err(BruceError::Other(anyhow::anyhow!(
                    "expected v as FixedSizeList, got {other:?}"
                )));
            }
        };

        let mut mem = Self::new(d_k, d_v);
        let reader = builder
            .build()
            .map_err(|e| BruceError::Other(anyhow::anyhow!("parquet build: {e}")))?;

        for batch_res in reader {
            let batch = batch_res.map_err(|e| BruceError::Other(anyhow::anyhow!("batch: {e}")))?;
            let ids = batch
                .column_by_name("fact_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| BruceError::Other(anyhow::anyhow!("fact_id column")))?;
            let ks = batch
                .column_by_name("k")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| BruceError::Other(anyhow::anyhow!("k column")))?;
            let vs = batch
                .column_by_name("v")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| BruceError::Other(anyhow::anyhow!("v column")))?;
            let owners = batch
                .column_by_name("owner")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| BruceError::Other(anyhow::anyhow!("owner column")))?;
            let at = batch
                .column_by_name("written_at")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
                .ok_or_else(|| BruceError::Other(anyhow::anyhow!("at column")))?;
            let del = batch
                .column_by_name("deleted")
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
                .ok_or_else(|| BruceError::Other(anyhow::anyhow!("deleted column")))?;

            for r in 0..batch.num_rows() {
                let id = ids.value(r);
                let k_inner = ks.value(r);
                let k_arr = k_inner
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| BruceError::Other(anyhow::anyhow!("k inner")))?;
                let v_inner = vs.value(r);
                let v_arr = v_inner
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| BruceError::Other(anyhow::anyhow!("v inner")))?;
                let k = Array1::from_iter((0..d_k).map(|i| k_arr.value(i)));
                let v = Array1::from_iter((0..d_v).map(|i| v_arr.value(i)));
                mem.insertion_order.push(id.into());
                mem.rows.insert(
                    id.into(),
                    Row {
                        k,
                        v,
                        owner: owners.value(r).into(),
                        written_at: at.value(r),
                        deleted: del.value(r),
                    },
                );
            }
        }

        // best-effort audit replay
        let audit_path = path.as_ref().with_extension("audit.jsonl");
        if audit_path.exists() {
            let bytes = std::fs::read_to_string(&audit_path)?;
            for line in bytes.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let entry: AuditEntry = serde_json::from_str(line)
                    .map_err(|e| BruceError::Other(anyhow::anyhow!("audit parse: {e}")))?;
                mem.log.push(entry);
            }
        }

        Ok(mem)
    }

    /// Run an `F_eps` attention query directly against the live rows,
    /// without materialising a dense (K, V) snapshot first. This is the
    /// hot path for `bruce-server` under concurrent agent load: it
    /// avoids the per-query allocation of two `Array2<f64>`s.
    ///
    /// Returns `Some(out)` if at least one row was alive, else `None`.
    /// `out` has length `d_v`.
    pub fn attention_query(
        &self,
        x: ArrayView1<'_, f64>,
        eps: crate::types::Eps,
        sim: crate::types::Sim,
    ) -> Option<Array1<f64>> {
        if x.len() != self.d_k {
            // caller already validates; treat dim-mismatch as "no rows"
            return None;
        }
        // Two-pass numerically-stable log-sum-exp + weighted sum.
        // The score per alive row j is sim(x, k_j); the output is
        //     A_eps = (sum_j exp(s_j / eps) v_j) / (sum_j exp(s_j / eps))
        // shifted by m = max_j s_j for stability.
        //
        // First pass: compute m. Second pass: accumulate (num, den).
        // Both passes iterate the AHashMap once and skip deleted rows.
        // The deleted check is O(1) per row; no dense allocation.
        let mut m = f64::NEG_INFINITY;
        let mut any = false;
        for id in self.insertion_order.iter() {
            let row = match self.rows.get(id) {
                Some(r) if !r.deleted => r,
                _ => continue,
            };
            any = true;
            let s = match sim {
                crate::types::Sim::Dot => x.dot(&row.k),
                crate::types::Sim::NegSquared => {
                    let mut d2 = 0.0;
                    for i in 0..x.len() {
                        let d = x[i] - row.k[i];
                        d2 += d * d;
                    }
                    -0.5 * d2
                }
                crate::types::Sim::Indicator => {
                    let mut d2 = 0.0;
                    for i in 0..x.len() {
                        let d = x[i] - row.k[i];
                        d2 += d * d;
                    }
                    if d2 == 0.0 {
                        0.0
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            };
            if s > m {
                m = s;
            }
        }
        if !any {
            return None;
        }
        // For eps = 0 with Indicator, fall back to direct equality reduce.
        if eps.is_zero() {
            // exact: pick rows whose score equals m and is finite; mean their v
            let mut sum = Array1::<f64>::zeros(self.d_v);
            let mut n = 0usize;
            for id in self.insertion_order.iter() {
                let row = match self.rows.get(id) {
                    Some(r) if !r.deleted => r,
                    _ => continue,
                };
                let s = match sim {
                    crate::types::Sim::Dot => x.dot(&row.k),
                    crate::types::Sim::NegSquared => {
                        let mut d2 = 0.0;
                        for i in 0..x.len() {
                            let d = x[i] - row.k[i];
                            d2 += d * d;
                        }
                        -0.5 * d2
                    }
                    crate::types::Sim::Indicator => {
                        let mut d2 = 0.0;
                        for i in 0..x.len() {
                            let d = x[i] - row.k[i];
                            d2 += d * d;
                        }
                        if d2 == 0.0 {
                            0.0
                        } else {
                            f64::NEG_INFINITY
                        }
                    }
                };
                if s == m && s.is_finite() {
                    sum.scaled_add(1.0, &row.v);
                    n += 1;
                }
            }
            return Some(if n == 0 { sum } else { sum / n as f64 });
        }
        // eps > 0: standard softmax over (s - m) / eps
        let mut num = Array1::<f64>::zeros(self.d_v);
        let mut den = 0.0;
        for id in self.insertion_order.iter() {
            let row = match self.rows.get(id) {
                Some(r) if !r.deleted => r,
                _ => continue,
            };
            let s = match sim {
                crate::types::Sim::Dot => x.dot(&row.k),
                crate::types::Sim::NegSquared => {
                    let mut d2 = 0.0;
                    for i in 0..x.len() {
                        let d = x[i] - row.k[i];
                        d2 += d * d;
                    }
                    -0.5 * d2
                }
                crate::types::Sim::Indicator => {
                    let mut d2 = 0.0;
                    for i in 0..x.len() {
                        let d = x[i] - row.k[i];
                        d2 += d * d;
                    }
                    if d2 == 0.0 {
                        0.0
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            };
            let w = ((s - m) / eps.0).exp();
            num.scaled_add(w, &row.v);
            den += w;
        }
        Some(num / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn write_then_read_exact() {
        let mut m = KvMemory::new(2, 2);
        let k = array![1.0, 0.0];
        let v = array![3.25, 2.75];
        m.write("t1", k.view(), v.view(), "alice").unwrap();
        let (kk, vv) = m.read_exact("t1").unwrap();
        assert_eq!(kk, &k);
        assert_eq!(vv, &v);
    }

    #[test]
    fn delete_is_owner_enforced() {
        let mut m = KvMemory::new(2, 2);
        m.write(
            "x",
            array![1.0, 0.0].view(),
            array![1.0, 0.0].view(),
            "alice",
        )
        .unwrap();
        let err = m.delete("x", "mallory").unwrap_err();
        assert!(matches!(err, BruceError::PermissionDenied(_, _, _)));
        m.delete("x", "alice").unwrap();
        assert!(m.read_exact("x").is_none());
    }

    #[test]
    fn snapshot_alive_excludes_deleted() {
        let mut m = KvMemory::new(2, 1);
        for i in 0..5 {
            m.write(
                &format!("k{i}"),
                array![i as f64, 0.0].view(),
                array![i as f64 * 10.0].view(),
                "alice",
            )
            .unwrap();
        }
        m.delete("k2", "alice").unwrap();
        let (ids, k, v) = m.snapshot_alive();
        assert_eq!(ids.len(), 4);
        assert_eq!(k.nrows(), 4);
        assert_eq!(v.nrows(), 4);
        assert!(!ids.contains(&"k2".to_string()));
    }

    #[test]
    fn audit_log_records_ops() {
        let mut m = KvMemory::new(1, 1);
        let k = array![1.0];
        let v = array![1.0];
        m.write("a", k.view(), v.view(), "alice").unwrap();
        m.write("b", k.view(), v.view(), "bob").unwrap();
        m.delete("a", "alice").unwrap();
        let log = m.audit_log();
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].op, "insert");
        assert_eq!(log[1].op, "insert");
        assert_eq!(log[2].op, "delete");
    }

    #[test]
    fn parquet_roundtrip_preserves_rows_and_audit() {
        let dir = std::env::temp_dir().join(format!("bruce_persist_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("snap.parquet");

        let mut m = KvMemory::new(3, 2);
        m.write(
            "a",
            array![1.0, 0.0, -1.0].view(),
            array![10.0, 20.0].view(),
            "alice",
        )
        .unwrap();
        m.write(
            "b",
            array![0.5, 0.5, 0.5].view(),
            array![1.5, 2.5].view(),
            "bob",
        )
        .unwrap();
        m.write(
            "c",
            array![9.0, 9.0, 9.0].view(),
            array![-1.0, -2.0].view(),
            "carol",
        )
        .unwrap();
        m.delete("b", "bob").unwrap();

        m.save_parquet(&path).unwrap();
        let m2 = KvMemory::load_parquet(&path).unwrap();

        assert_eq!(m2.len_total(), m.len_total());
        assert_eq!(m2.len_alive(), m.len_alive());
        let (a_k, a_v) = m2.read_exact("a").unwrap();
        assert_eq!(a_k, &array![1.0, 0.0, -1.0]);
        assert_eq!(a_v, &array![10.0, 20.0]);
        // b is tombstoned
        assert!(m2.read_exact("b").is_none());
        // owner enforcement preserved across reload
        let err = m2.audit_log();
        assert_eq!(err.len(), 4); // 3 writes + 1 delete

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parquet_roundtrip_preserves_attention_bitexact() {
        let dir = std::env::temp_dir().join(format!("bruce_persist_attn_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("snap.parquet");

        use crate::types::{Eps, Sim};
        let mut m = KvMemory::new(4, 3);
        for i in 0..16 {
            let k = Array1::from_iter((0..4).map(|j| ((i + j) as f64).sin()));
            let v = Array1::from_iter((0..3).map(|j| ((i * 7 + j) as f64).cos()));
            m.write(&format!("k{i}"), k.view(), v.view(), "alice")
                .unwrap();
        }
        m.delete("k5", "alice").unwrap();

        let x = Array1::from_iter((0..4).map(|j| (j as f64).sqrt()));
        let out1 = m.attention_query(x.view(), Eps::ONE, Sim::Dot).unwrap();

        m.save_parquet(&path).unwrap();
        let m2 = KvMemory::load_parquet(&path).unwrap();
        let out2 = m2.attention_query(x.view(), Eps::ONE, Sim::Dot).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-12, "bit-mismatch after parquet reload");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}

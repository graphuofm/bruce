//! Workstream 15 — Parquet/npy robustness corpus (dirty real-world
//! files) + attach_key edge semantics.
//!
//! Every fixture is built AT TEST TIME with the same arrow/parquet
//! crates the engine links (no binary blobs in the repo, no Python
//! dependency in the test path). Each test documents the semantics it
//! pins; where behavior was previously undefined the pinned choice is
//! called out explicitly and mirrored by a comment in
//! `bruce-query/src/ingest.rs`.
//!
//! Corpus:
//!   * zero-row file (schema, no row groups)
//!   * all-NULL string column
//!   * NULLs in numeric columns
//!   * unsupported column types alongside supported ones
//!   * multi-batch files (reader batch boundaries at 1024 rows)
//!   * duplicate column names (same-type and mixed-type)
//!   * >100k distinct strings (dictionary growth)
//!   * file-not-found, truncated file, non-parquet bytes
//!   * attach_key_{f64,f32} dimension mismatch + empty-table cases
//!
//! The example's minimal .npy reader lives in
//! `bruce-query/examples/one_query.rs` ONLY (not library code), so npy
//! robustness is out of scope here; the library-side ingestion of key
//! matrices is `attach_key_*`, tested below.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use bruce_query::{Column, QueryError, Table};

/// Unique temp path for a fixture (tests run concurrently).
fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "bruce_ingest_robust_{}_{}_{name}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("t")
            .replace("::", "_"),
    ));
    p
}

/// Write `batches` (all sharing `schema`) to a fresh parquet file.
fn write_parquet(name: &str, schema: Arc<Schema>, batches: &[RecordBatch]) -> PathBuf {
    write_parquet_props(name, schema, batches, None)
}

fn write_parquet_props(
    name: &str,
    schema: Arc<Schema>,
    batches: &[RecordBatch],
    props: Option<WriterProperties>,
) -> PathBuf {
    let path = tmp_path(name);
    let file = File::create(&path).expect("create fixture");
    let mut writer = ArrowWriter::try_new(file, schema, props).expect("writer");
    for b in batches {
        writer.write(b).expect("write batch");
    }
    writer.close().expect("close");
    path
}

fn scalar(t: &Table, name: &str) -> Vec<f64> {
    match t.columns.get(name) {
        Some(Column::ScalarF64(v)) => v.clone(),
        other => panic!("column {name}: expected ScalarF64, got {other:?}"),
    }
}

fn dict_col<'a>(t: &'a Table, name: &str) -> (&'a Vec<u32>, &'a Vec<String>) {
    match t.columns.get(name) {
        Some(Column::DictU32 { codes, dict }) => (codes, dict),
        other => panic!("column {name}: expected DictU32, got {other:?}"),
    }
}

// ---------------------------------------------------------------- zero rows

/// PINNED SEMANTICS: a parquet file with a schema but zero rows loads
/// as `Ok` with an EMPTY column map — columns materialise per record
/// batch, and a zero-row file yields no batches. (Consequence: the
/// schema of a zero-row file is NOT preserved; a later attach_key on
/// such a table follows the empty-table rule below.)
#[test]
fn zero_row_file_loads_as_empty_table() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("rating", DataType::Float64, true),
    ]));
    let path = write_parquet("zero_rows.parquet", schema, &[]);
    let t = Table::from_parquet(&path).expect("zero-row file must load");
    assert!(
        t.columns.is_empty(),
        "zero-row file: columns materialise per batch, so none exist"
    );
    std::fs::remove_file(path).ok();
}

// ----------------------------------------------------------- NULL handling

/// All-NULL string column: every row maps to the dictionary entry
/// "(null)" (documented in ingest.rs), dict has exactly that one entry.
#[test]
fn all_null_string_column() {
    let n = 100;
    let schema = Arc::new(Schema::new(vec![
        Field::new("genre", DataType::Utf8, true),
        Field::new("rating", DataType::Float64, false),
    ]));
    let genre: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>; n]));
    let rating: ArrayRef = Arc::new(Float64Array::from(
        (0..n).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![genre, rating]).unwrap();
    let path = write_parquet("all_null_str.parquet", schema, &[batch]);

    let t = Table::from_parquet(&path).expect("load");
    let (codes, dict) = dict_col(&t, "genre");
    assert_eq!(dict, &vec!["(null)".to_string()]);
    assert_eq!(codes.len(), n);
    assert!(codes.iter().all(|&c| c == 0));
    assert_eq!(scalar(&t, "rating").len(), n);
    std::fs::remove_file(path).ok();
}

/// NULLs in numeric columns become f64::NAN (documented in ingest.rs);
/// non-null values survive exactly. Int32/Int64/Float32 all promote to
/// ScalarF64.
#[test]
fn numeric_nulls_become_nan_and_types_promote() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("f64c", DataType::Float64, true),
        Field::new("f32c", DataType::Float32, true),
        Field::new("i64c", DataType::Int64, true),
        Field::new("i32c", DataType::Int32, true),
    ]));
    let f64c: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.5), None, Some(-3.25)]));
    let f32c: ArrayRef = Arc::new(Float32Array::from(vec![None, Some(2.5f32), Some(0.0)]));
    let i64c: ArrayRef = Arc::new(Int64Array::from(vec![Some(7), Some(-9), None]));
    let i32c: ArrayRef = Arc::new(Int32Array::from(vec![Some(42), None, Some(0)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![f64c, f32c, i64c, i32c]).unwrap();
    let path = write_parquet("numeric_nulls.parquet", schema, &[batch]);

    let t = Table::from_parquet(&path).expect("load");
    let f = scalar(&t, "f64c");
    assert_eq!(f[0], 1.5);
    assert!(f[1].is_nan());
    assert_eq!(f[2], -3.25);
    let g = scalar(&t, "f32c");
    assert!(g[0].is_nan());
    assert_eq!(g[1], 2.5);
    let i = scalar(&t, "i64c");
    assert_eq!((i[0], i[1]), (7.0, -9.0));
    assert!(i[2].is_nan());
    let j = scalar(&t, "i32c");
    assert_eq!(j[0], 42.0);
    assert!(j[1].is_nan());
    std::fs::remove_file(path).ok();
}

// ------------------------------------------------------- unsupported types

/// Unsupported column types (timestamp, boolean) alongside supported
/// ones: the file loads, unsupported columns are SKIPPED (absent from
/// the column map, per the documented v1 rule), supported columns are
/// complete and aligned.
#[test]
fn unsupported_types_skipped_not_crash() {
    let n = 10;
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("flag", DataType::Boolean, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]));
    let ts: ArrayRef = Arc::new(TimestampMillisecondArray::from(
        (0..n as i64).map(|i| i * 1000).collect::<Vec<_>>(),
    ));
    let fl: ArrayRef = Arc::new(BooleanArray::from(vec![true; n]));
    let ti: ArrayRef = Arc::new(StringArray::from(
        (0..n).map(|i| format!("t{i}")).collect::<Vec<_>>(),
    ));
    let sc: ArrayRef = Arc::new(Float64Array::from(
        (0..n).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![ts, fl, ti, sc]).unwrap();
    let path = write_parquet("unsupported.parquet", schema, &[batch]);

    let t = Table::from_parquet(&path).expect("must not crash on unsupported types");
    assert!(!t.columns.contains_key("created_at"), "timestamp skipped");
    assert!(!t.columns.contains_key("flag"), "boolean skipped");
    let (codes, dict) = dict_col(&t, "title");
    assert_eq!(codes.len(), n);
    assert_eq!(dict.len(), n);
    assert_eq!(
        scalar(&t, "score"),
        (0..n).map(|i| i as f64).collect::<Vec<_>>()
    );
    std::fs::remove_file(path).ok();
}

// -------------------------------------------------------- batch boundaries

/// 3000 rows written as 3 row groups of 1000: the reader hands them
/// back in multiple batches (default batch size 1024, so boundaries
/// fall mid-row-group too). String dictionary codes must be consistent
/// ACROSS batches (same label -> same code) and numeric columns must
/// concatenate in order with nothing lost at any boundary.
#[test]
fn multi_batch_file_boundaries() {
    let n_per = 1000usize;
    let labels = ["drama", "comedy", "horror", "sci-fi", "noir"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("genre", DataType::Utf8, false),
        Field::new("x", DataType::Float64, false),
    ]));
    let mut batches = Vec::new();
    for b in 0..3usize {
        let genre: ArrayRef = Arc::new(StringArray::from(
            (0..n_per)
                .map(|i| labels[(b * n_per + i) % labels.len()])
                .collect::<Vec<_>>(),
        ));
        let x: ArrayRef = Arc::new(Float64Array::from(
            (0..n_per)
                .map(|i| (b * n_per + i) as f64)
                .collect::<Vec<_>>(),
        ));
        batches.push(RecordBatch::try_new(schema.clone(), vec![genre, x]).unwrap());
    }
    let props = WriterProperties::builder()
        .set_max_row_group_size(n_per)
        .build();
    let path = write_parquet_props("multibatch.parquet", schema, &batches, Some(props));

    let t = Table::from_parquet(&path).expect("load");
    let x = scalar(&t, "x");
    assert_eq!(x.len(), 3 * n_per);
    assert!(
        x.iter().enumerate().all(|(i, &v)| v == i as f64),
        "row order preserved"
    );
    let (codes, dict) = dict_col(&t, "genre");
    assert_eq!(codes.len(), 3 * n_per);
    // decode every row and compare with the generator — this fails if
    // codes are not consistent across batch/row-group boundaries
    for (i, &c) in codes.iter().enumerate() {
        assert_eq!(dict[c as usize], labels[i % labels.len()], "row {i}");
    }
    assert_eq!(dict.len(), labels.len(), "no duplicate dictionary entries");
    std::fs::remove_file(path).ok();
}

// ------------------------------------------------------- duplicate columns

/// DEFINED SEMANTICS (was undefined): duplicate column names in the
/// parquet schema are rejected with a typed `QueryError::Bind`,
/// aligned with PostgreSQL's 42701 `duplicate_column`. Before this was
/// pinned, a same-type duplicate silently appended both columns
/// into one Vec of length 2n (corrupt table) and a mixed-type
/// duplicate hit an `unreachable!()` panic — both exposed by this
/// test, fixed in ingest.rs.
#[test]
fn duplicate_column_name_same_type_is_typed_err() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("dup", DataType::Float64, false),
        Field::new("dup", DataType::Float64, false),
    ]));
    let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
    let b: ArrayRef = Arc::new(Float64Array::from(vec![3.0, 4.0]));
    let batch = RecordBatch::try_new(schema.clone(), vec![a, b]).unwrap();
    let path = write_parquet("dup_same.parquet", schema, &[batch]);

    let err = Table::from_parquet(&path).expect_err("duplicate names must be rejected");
    match &err {
        QueryError::Bind(m) => assert!(m.contains("duplicate column"), "got: {m}"),
        other => panic!("expected Bind, got {other:?}"),
    }
    std::fs::remove_file(path).ok();
}

/// Mixed-type duplicate (Utf8 + Float64 under one name): before the
/// fix this panicked (`unreachable!()` on the column-variant match);
/// now the same typed error as the same-type case.
#[test]
fn duplicate_column_name_mixed_type_is_typed_err_not_panic() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("dup", DataType::Float64, false),
        Field::new("dup", DataType::Utf8, false),
    ]));
    let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
    let b: ArrayRef = Arc::new(StringArray::from(vec!["x", "y"]));
    let batch = RecordBatch::try_new(schema.clone(), vec![a, b]).unwrap();
    let path = write_parquet("dup_mixed.parquet", schema, &[batch]);

    let err = Table::from_parquet(&path).expect_err("mixed-type duplicate must be a typed error");
    match &err {
        QueryError::Bind(m) => assert!(m.contains("duplicate column"), "got: {m}"),
        other => panic!("expected Bind, got {other:?}"),
    }
    std::fs::remove_file(path).ok();
}

// ------------------------------------------------------ dictionary growth

/// 120k distinct strings (dictionary growth past 100k): the load-time
/// dictionary grows without collision or truncation; codes round-trip
/// to the exact original labels, including across the many (over 100)
/// reader batches.
#[test]
fn dictionary_growth_120k_distinct_strings() {
    let n = 120_000usize;
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let ids: ArrayRef = Arc::new(StringArray::from(
        (0..n).map(|i| format!("user_{i:06}")).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids]).unwrap();
    let path = write_parquet("dict120k.parquet", schema, &[batch]);

    let t = Table::from_parquet(&path).expect("load");
    let (codes, dict) = dict_col(&t, "id");
    assert_eq!(codes.len(), n);
    assert_eq!(dict.len(), n, "all 120k strings distinct in the dictionary");
    for i in [0usize, 1, 1023, 1024, 59_999, 100_000, n - 1] {
        assert_eq!(dict[codes[i] as usize], format!("user_{i:06}"), "row {i}");
    }
    std::fs::remove_file(path).ok();
}

// ----------------------------------------------------------- broken files

#[test]
fn file_not_found_is_typed_err() {
    let err = Table::from_parquet("/nonexistent/dir/never_there.parquet")
        .expect_err("missing file must be Err");
    match &err {
        QueryError::Bind(m) => assert!(m.contains("open parquet"), "got: {m}"),
        other => panic!("expected Bind, got {other:?}"),
    }
}

/// Truncated parquet (footer gone) and non-parquet bytes: both must
/// return a typed error, never panic. Truncation is applied at several
/// points including 0 and 3 bytes (shorter than the magic).
#[test]
fn truncated_and_garbage_files_are_typed_err() {
    // build a real file first
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
    let x: ArrayRef = Arc::new(Float64Array::from(
        (0..5000).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![x]).unwrap();
    let good = write_parquet("to_truncate.parquet", schema, &[batch]);
    let bytes = std::fs::read(&good).unwrap();

    for (i, cut) in [0usize, 3, 8, bytes.len() / 2, bytes.len() - 4]
        .into_iter()
        .enumerate()
    {
        let p = tmp_path(&format!("truncated_{i}.parquet"));
        std::fs::write(&p, &bytes[..cut]).unwrap();
        match Table::from_parquet(&p) {
            Err(QueryError::Bind(_)) => {}
            Err(other) => panic!("expected Bind for truncation at {cut}, got {other:?}"),
            Ok(_) => panic!("truncated at {cut} bytes must be Err, got Ok"),
        }
        std::fs::remove_file(p).ok();
    }

    let p = tmp_path("garbage.parquet");
    std::fs::write(&p, b"this is not a parquet file at all, sorry\n").unwrap();
    assert!(matches!(Table::from_parquet(&p), Err(QueryError::Bind(_))));
    std::fs::remove_file(p).ok();
    std::fs::remove_file(good).ok();
}

// ------------------------------------------------------------- attach_key

fn table_with_rows(n: usize) -> Table {
    let mut t = Table::default();
    t.columns.insert(
        "x".into(),
        Column::ScalarF64((0..n).map(|i| i as f64).collect()),
    );
    t
}

/// attach_key_f64 / attach_key_f32 with a row-count mismatch: typed
/// `Bind` error naming the offending column, table unchanged.
#[test]
fn attach_key_dim_mismatch_is_typed_err() {
    use ndarray::Array2;
    let mut t = table_with_rows(10);
    let err = t
        .attach_key_f64("emb", Array2::<f64>::zeros((11, 4)))
        .expect_err("11 keys vs 10 rows");
    match &err {
        QueryError::Bind(m) => {
            assert!(
                m.contains("emb") && m.contains("11") && m.contains("10"),
                "got: {m}"
            )
        }
        other => panic!("expected Bind, got {other:?}"),
    }
    assert!(
        !t.columns.contains_key("emb"),
        "failed attach must not install"
    );

    let err32 = t
        .attach_key_f32("emb32", Array2::<f32>::zeros((9, 4)))
        .expect_err("9 keys vs 10 rows");
    assert!(matches!(err32, QueryError::Bind(_)));
    assert!(!t.columns.contains_key("emb32"));

    // matching row count succeeds for both dtypes
    t.attach_key_f64("emb", Array2::<f64>::zeros((10, 4)))
        .unwrap();
    t.attach_key_f32("emb32", Array2::<f32>::zeros((10, 4)))
        .unwrap();
    assert_eq!(t.columns.get("emb").unwrap().len(), 10);
    assert_eq!(t.columns.get("emb32").unwrap().len(), 10);
}

/// PINNED SEMANTICS: attaching a key matrix to a table with NO columns
/// (e.g. a zero-row parquet load, whose column map is empty) succeeds
/// for ANY row count — there is no existing column to disagree with,
/// and the key column then defines the table's row count. A second
/// attach with a different row count is rejected against the first.
#[test]
fn attach_key_to_empty_table_defines_row_count() {
    use ndarray::Array2;
    let mut t = Table::default();
    t.attach_key_f64("emb", Array2::<f64>::zeros((7, 3)))
        .expect("empty table accepts any key row count");
    assert_eq!(t.columns.get("emb").unwrap().len(), 7);
    // now 7 rows are established; a mismatched second key is rejected
    let err = t
        .attach_key_f32("emb2", Array2::<f32>::zeros((8, 3)))
        .expect_err("second key must match the established row count");
    assert!(matches!(err, QueryError::Bind(_)));
    // and a matching one is accepted
    t.attach_key_f32("emb2", Array2::<f32>::zeros((7, 3)))
        .unwrap();
}

/// Zero-row parquet + attach_key: the end-to-end shape of the pinned
/// empty-table rule. (The zero-row load gives an empty column map, so
/// the first attach defines the row count.)
#[test]
fn zero_row_parquet_then_attach_key() {
    use ndarray::Array2;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
    let path = write_parquet("zero_then_attach.parquet", schema, &[]);
    let mut t = Table::from_parquet(&path).unwrap();
    t.attach_key_f64("emb", Array2::<f64>::zeros((3, 2)))
        .expect("attach to zero-row load follows the empty-table rule");
    assert_eq!(t.columns.get("emb").unwrap().len(), 3);
    std::fs::remove_file(path).ok();
}

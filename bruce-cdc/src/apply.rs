//! The apply path: committed transactions -> the mirror
//! [`Database`], through the same write path every caller uses
//! (`insert_row` / `delete_where`), so maintained views update
//! incrementally and stats invalidate exactly as for native writes.

use std::collections::HashMap;

use ndarray::Array2;

use bruce_query::{Column, Database, Pred, RowValues, Table};

use crate::pgoutput::TupleDatum;
use crate::source::{CommittedTx, NamedTuple, RowChange};
use crate::CdcError;

/// How one PG relation maps onto one bruce table. The key column is
/// assembled from scalar parts (PG has no vector column here; the
/// demo schema carries the embedding as e0..eN scalars).
#[derive(Debug, Clone)]
pub struct TableMap {
    /// Table name on both sides.
    pub table: String,
    /// Primary key column (scalar; drives Delete -> `Pred::Eq`).
    pub pk: String,
    /// Dictionary-encoded label columns.
    pub label_cols: Vec<String>,
    /// Scalar f64 columns (includes `pk` and the key parts).
    pub scalar_cols: Vec<String>,
    /// Name of the assembled key column on the bruce side.
    pub key_col: String,
    /// PG columns forming the key vector, in order.
    pub key_parts: Vec<String>,
}

impl TableMap {
    /// The demo schema: cdc_movies(movie_id, genre, rating, year, e0, e1).
    pub fn cdc_movies() -> Self {
        TableMap {
            table: "cdc_movies".into(),
            pk: "movie_id".into(),
            label_cols: vec!["genre".into()],
            scalar_cols: vec![
                "movie_id".into(),
                "rating".into(),
                "year".into(),
                "e0".into(),
                "e1".into(),
            ],
            key_col: "emb".into(),
            key_parts: vec!["e0".into(), "e1".into()],
        }
    }
}

/// The mirror: a [`Database`] kept fresh from a change stream.
///
/// # Consistency contract (chaos.rs pins it)
///
/// - **Commit-buffered delivery**: the source yields only whole
///   committed transactions, so nothing reaches the mirror before the
///   Commit record arrives (a 5000-row transaction lands in one
///   [`Mirror::apply_tx`] call).
/// - **No partial reads**: `apply_tx` takes `&mut self`; any reader
///   sequenced between `apply_tx` calls observes either none or all
///   of a transaction. There is NO crash atomicity — the mirror is
///   in-memory and rebuilt (snapshot or replay) after a process death.
/// - **Exactly-once across resume**: `end_lsn` strictly increases
///   over the committed transactions of one stream, and the slot
///   redelivers from `confirmed_flush_lsn` (the last ack). A crash in
///   the apply-then-ack window therefore redelivers a prefix of
///   already-applied transactions; `apply_tx` filters them with
///   [`Mirror::last_lsn`] instead of double-applying.
pub struct Mirror {
    /// The mirrored database (views may be registered on it).
    pub db: Database,
    pub(crate) map: TableMap,
    /// Row changes applied since the snapshot.
    pub rows_applied: usize,
    /// Transactions applied since the snapshot.
    pub txs_applied: usize,
    /// `end_lsn` of the last applied transaction — the exactly-once
    /// watermark (see the consistency contract above). Production
    /// change justified by tests/chaos.rs kill_resume_exactly_once:
    /// without it, a kill between apply and ack double-applies the
    /// redelivered transaction.
    pub last_lsn: u64,
}

impl Mirror {
    /// Build the mirror table from snapshot rows (text values as the
    /// wire delivers them) and register it.
    pub fn from_snapshot(
        map: TableMap,
        col_names: &[String],
        rows: &[Vec<Option<String>>],
    ) -> Result<Self, CdcError> {
        let idx_of = |c: &str| {
            col_names
                .iter()
                .position(|n| n == c)
                .ok_or_else(|| CdcError::Apply(format!("snapshot missing column {c}")))
        };
        let cell = |row: &[Option<String>], i: usize| -> Result<String, CdcError> {
            row.get(i)
                .and_then(|v| v.clone())
                .ok_or_else(|| CdcError::Apply("NULL in snapshot (schema forbids)".into()))
        };

        let mut table = Table::default();
        for c in &map.scalar_cols {
            let i = idx_of(c)?;
            let mut v = Vec::with_capacity(rows.len());
            for row in rows {
                v.push(parse_f64(&cell(row, i)?, c)?);
            }
            table.columns.insert(c.clone(), Column::ScalarF64(v));
        }
        for c in &map.label_cols {
            let i = idx_of(c)?;
            let mut dict: Vec<String> = Vec::new();
            let mut codes = Vec::with_capacity(rows.len());
            for row in rows {
                let label = cell(row, i)?;
                let code = match dict.iter().position(|l| *l == label) {
                    Some(k) => k,
                    None => {
                        dict.push(label);
                        dict.len() - 1
                    }
                };
                codes.push(code as u32);
            }
            table
                .columns
                .insert(c.clone(), Column::DictU32 { codes, dict });
        }
        let part_idx: Vec<usize> = map
            .key_parts
            .iter()
            .map(|c| idx_of(c))
            .collect::<Result<_, _>>()?;
        let mut key = Array2::<f64>::zeros((rows.len(), part_idx.len()));
        for (r, row) in rows.iter().enumerate() {
            for (j, &i) in part_idx.iter().enumerate() {
                key[(r, j)] = parse_f64(&cell(row, i)?, &map.key_parts[j])?;
            }
        }
        table
            .columns
            .insert(map.key_col.clone(), Column::KeyF64(key));

        let mut db = Database::new();
        db.register(&map.table, table);
        Ok(Mirror {
            db,
            map,
            rows_applied: 0,
            txs_applied: 0,
            last_lsn: 0,
        })
    }

    /// Apply one committed transaction. Returns rows applied (0 for
    /// transactions that only touched other relations, and 0 for a
    /// replayed transaction — `end_lsn <= last_lsn` — which is
    /// skipped without touching the mirror or the counters; the
    /// caller still acks it so the slot advances).
    pub fn apply_tx(&mut self, tx: &CommittedTx) -> Result<usize, CdcError> {
        if tx.end_lsn <= self.last_lsn {
            // Exactly-once: this transaction was already applied in a
            // previous life of the connection (crash after apply,
            // before ack). end_lsn is strictly monotone per stream.
            return Ok(0);
        }
        let mut applied = 0;
        for change in &tx.changes {
            match change {
                RowChange::Insert { rel, cols } if *rel == self.map.table => {
                    let row = self.row_values(cols)?;
                    self.db.insert_row(&self.map.table, &row)?;
                    applied += 1;
                }
                RowChange::Delete { rel, old } if *rel == self.map.table => {
                    let pk = lookup(old, &self.map.pk)?;
                    let pk = parse_f64(&pk, &self.map.pk)?;
                    let n = self
                        .db
                        .delete_where(&self.map.table, &Pred::Eq(self.map.pk.clone(), pk))?;
                    if n != 1 {
                        return Err(CdcError::Apply(format!(
                            "delete {}={pk} removed {n} rows, want 1 (mirror out of sync)",
                            self.map.pk
                        )));
                    }
                    applied += 1;
                }
                RowChange::Update { rel, old, new } if *rel == self.map.table => {
                    // Update = delete(old pk) + insert(resolved new)
                    // through the standard write path, so maintained
                    // views update via the (m,num,den) group inverse
                    // exactly as for native writes. The old row is
                    // located by:
                    // - the old tuple's pk when one was sent (REPLICA
                    //   IDENTITY FULL always; DEFAULT iff an identity
                    //   column changed — key-only 'K' tuples carry it),
                    // - else the NEW tuple's pk (DEFAULT, key
                    //   untouched: old pk == new pk by definition).
                    let pk_old_text = match old {
                        Some(o) => lookup(o, &self.map.pk)?,
                        None => match datum_of(new, &self.map.pk)? {
                            TupleDatum::Text(v) => v.clone(),
                            TupleDatum::Unchanged => {
                                return Err(CdcError::Apply(format!(
                                    "update: key column {} is unchanged-TOAST and no old \
                                     tuple was sent; cannot locate the row",
                                    self.map.pk
                                )))
                            }
                            TupleDatum::Null => {
                                return Err(CdcError::Apply(format!(
                                    "update: NULL key column {} (schema forbids)",
                                    self.map.pk
                                )))
                            }
                        },
                    };
                    let pk_old = parse_f64(&pk_old_text, &self.map.pk)?;
                    // Resolve BEFORE the delete: Unchanged datums read
                    // the mirror's current row — that is why the
                    // mirror exists.
                    let row = self.resolve_update_row(pk_old, new)?;
                    let n = self
                        .db
                        .delete_where(&self.map.table, &Pred::Eq(self.map.pk.clone(), pk_old))?;
                    if n != 1 {
                        return Err(CdcError::Apply(format!(
                            "update {}={pk_old} removed {n} rows, want 1 (mirror out of sync)",
                            self.map.pk
                        )));
                    }
                    self.db.insert_row(&self.map.table, &row)?;
                    applied += 1;
                }
                _ => {}
            }
        }
        self.rows_applied += applied;
        self.txs_applied += 1;
        self.last_lsn = tx.end_lsn;
        Ok(applied)
    }

    /// Number of rows currently in the mirror table.
    pub fn n_rows(&self) -> usize {
        self.db
            .catalog
            .tables
            .get(&self.map.table)
            .and_then(|t| t.columns.get(&self.map.pk))
            .map(|c| c.len())
            .unwrap_or(0)
    }

    fn row_values(&self, cols: &NamedTuple) -> Result<RowValues, CdcError> {
        let mut row = RowValues::default();
        for c in &self.map.scalar_cols {
            row.scalars
                .insert(c.clone(), parse_f64(&lookup(cols, c)?, c)?);
        }
        for c in &self.map.label_cols {
            row.labels.insert(c.clone(), lookup(cols, c)?);
        }
        let mut key = Vec::with_capacity(self.map.key_parts.len());
        for c in &self.map.key_parts {
            key.push(parse_f64(&lookup(cols, c)?, c)?);
        }
        row.keys.insert(self.map.key_col.clone(), key);
        Ok(row)
    }

    /// Row index of `pk` in the mirror table (linear scan of the pk
    /// column — the mirror is columnar, not indexed).
    fn find_row(&self, pk: f64) -> Result<usize, CdcError> {
        let t = self
            .db
            .catalog
            .tables
            .get(&self.map.table)
            .ok_or_else(|| CdcError::Apply(format!("no mirror table {}", self.map.table)))?;
        let ids = match t.columns.get(&self.map.pk) {
            Some(Column::ScalarF64(v)) => v,
            _ => {
                return Err(CdcError::Apply(format!(
                    "pk column {} is not ScalarF64",
                    self.map.pk
                )))
            }
        };
        ids.iter().position(|&v| v == pk).ok_or_else(|| {
            CdcError::Apply(format!(
                "update {}={pk}: row not in mirror (out of sync)",
                self.map.pk
            ))
        })
    }

    /// Build the post-update [`RowValues`] from an Update new-tuple,
    /// resolving [`TupleDatum::Unchanged`] (untouched TOAST — the
    /// walsender omits the value) from the mirror's CURRENT row for
    /// `pk_old`. Must run before the row is deleted.
    fn resolve_update_row(&self, pk_old: f64, new: &NamedTuple) -> Result<RowValues, CdcError> {
        let needs_mirror = new.iter().any(|(_, d)| matches!(d, TupleDatum::Unchanged));
        let idx = if needs_mirror {
            Some(self.find_row(pk_old)?)
        } else {
            None
        };
        let table = self.db.catalog.tables.get(&self.map.table);
        let resolve_scalar = |c: &String| -> Result<f64, CdcError> {
            match datum_of(new, c)? {
                TupleDatum::Text(v) => parse_f64(v, c),
                TupleDatum::Null => Err(CdcError::Apply(format!(
                    "NULL in column {c} (schema forbids)"
                ))),
                TupleDatum::Unchanged => {
                    let i = idx.expect("idx resolved when any datum is Unchanged");
                    match table.and_then(|t| t.columns.get(c)) {
                        Some(Column::ScalarF64(v)) => Ok(v[i]),
                        _ => Err(CdcError::Apply(format!(
                            "unchanged-TOAST column {c} not a mirror scalar"
                        ))),
                    }
                }
            }
        };
        let mut row = RowValues::default();
        for c in &self.map.scalar_cols {
            row.scalars.insert(c.clone(), resolve_scalar(c)?);
        }
        for c in &self.map.label_cols {
            let label = match datum_of(new, c)? {
                TupleDatum::Text(v) => v.clone(),
                TupleDatum::Null => {
                    return Err(CdcError::Apply(format!(
                        "NULL in column {c} (schema forbids)"
                    )))
                }
                TupleDatum::Unchanged => {
                    let i = idx.expect("idx resolved when any datum is Unchanged");
                    match table.and_then(|t| t.columns.get(c)) {
                        Some(Column::DictU32 { codes, dict }) => dict[codes[i] as usize].clone(),
                        _ => {
                            return Err(CdcError::Apply(format!(
                                "unchanged-TOAST column {c} not a mirror dict column"
                            )))
                        }
                    }
                }
            };
            row.labels.insert(c.clone(), label);
        }
        let mut key = Vec::with_capacity(self.map.key_parts.len());
        for c in &self.map.key_parts {
            key.push(resolve_scalar(c)?);
        }
        row.keys.insert(self.map.key_col.clone(), key);
        Ok(row)
    }
}

/// The datum for `name`, or a typed "tuple missing column" error (a
/// DROPPED mapped column must fail loudly, not drift).
fn datum_of<'a>(cols: &'a NamedTuple, name: &str) -> Result<&'a TupleDatum, CdcError> {
    cols.iter()
        .find(|(n, _)| n == name)
        .map(|(_, d)| d)
        .ok_or_else(|| CdcError::Apply(format!("tuple missing column {name}")))
}

/// The materialized text for `name`. `Null` is a typed error (the
/// demo schema forbids NULL); `Unchanged` is a typed error too — it
/// is only meaningful in Update new-tuples, where
/// `Mirror::resolve_update_row` resolves it against the mirror.
fn lookup(cols: &NamedTuple, name: &str) -> Result<String, CdcError> {
    match datum_of(cols, name)? {
        TupleDatum::Text(v) => Ok(v.clone()),
        TupleDatum::Null => Err(CdcError::Apply(format!(
            "NULL in column {name} (schema forbids)"
        ))),
        TupleDatum::Unchanged => Err(CdcError::Apply(format!(
            "unchanged-TOAST marker in column {name} outside an update new-tuple"
        ))),
    }
}

fn parse_f64(text: &str, col: &str) -> Result<f64, CdcError> {
    text.parse::<f64>()
        .map_err(|e| CdcError::Apply(format!("column {col}: {text:?} is not an f64: {e}")))
}

/// Result of one query, as label -> value, for comparisons.
pub fn result_map(labels: &[String], values: &[f64]) -> HashMap<String, f64> {
    labels.iter().cloned().zip(values.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::CommittedTx;

    fn snap_cols() -> Vec<String> {
        ["movie_id", "genre", "rating", "year", "e0", "e1"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn row(id: i64, genre: &str, rating: f64, e0: f64, e1: f64) -> Vec<Option<String>> {
        vec![
            Some(id.to_string()),
            Some(genre.into()),
            Some(rating.to_string()),
            Some("2000".into()),
            Some(e0.to_string()),
            Some(e1.to_string()),
        ]
    }

    /// Snapshot text row -> named tuple datums (tests only).
    fn datums(names: &[String], vals: Vec<Option<String>>) -> NamedTuple {
        names
            .iter()
            .cloned()
            .zip(vals.into_iter().map(|v| match v {
                Some(s) => TupleDatum::Text(s),
                None => TupleDatum::Null,
            }))
            .collect()
    }

    fn change_insert(id: i64, genre: &str, rating: f64, e0: f64, e1: f64) -> RowChange {
        let names = snap_cols();
        RowChange::Insert {
            rel: "cdc_movies".into(),
            cols: datums(&names, row(id, genre, rating, e0, e1)),
        }
    }

    fn change_delete(id: i64) -> RowChange {
        let names = snap_cols();
        RowChange::Delete {
            rel: "cdc_movies".into(),
            old: datums(&names, row(id, "action", 5.0, 1.0, 0.0)),
        }
    }

    #[test]
    fn snapshot_then_insert_then_delete_tracks_rows() {
        let rows = vec![
            row(1, "action", 5.0, 1.0, 0.0),
            row(2, "drama", 7.0, 0.0, 1.0),
        ];
        let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &snap_cols(), &rows).unwrap();
        assert_eq!(m.n_rows(), 2);

        let tx = CommittedTx {
            changes: vec![change_insert(3, "action", 9.0, 0.6, 0.8), change_delete(1)],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert_eq!(m.apply_tx(&tx).unwrap(), 2);
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.rows_applied, 2);
    }

    #[test]
    fn delete_of_absent_row_is_an_error() {
        let rows = vec![row(1, "action", 5.0, 1.0, 0.0)];
        let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &snap_cols(), &rows).unwrap();
        let tx = CommittedTx {
            changes: vec![change_delete(99)],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert!(m.apply_tx(&tx).is_err());
    }

    #[test]
    fn changes_for_other_relations_are_skipped() {
        let rows = vec![row(1, "action", 5.0, 1.0, 0.0)];
        let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &snap_cols(), &rows).unwrap();
        let tx = CommittedTx {
            changes: vec![RowChange::Insert {
                rel: "movies".into(),
                cols: vec![],
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert_eq!(m.apply_tx(&tx).unwrap(), 0);
        assert_eq!(m.n_rows(), 1);
    }

    #[test]
    fn maintained_view_stays_fresh_through_apply() {
        use ndarray::Array1;
        let rows = vec![
            row(1, "action", 5.0, 1.0, 0.0),
            row(2, "action", 7.0, 0.0, 1.0),
        ];
        let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &snap_cols(), &rows).unwrap();
        let x = Array1::from_vec(vec![0.6, 0.8]);
        m.db.create_view("v", "cdc_movies", "genre", "rating", "emb", &x, 0.1)
            .unwrap();

        let tx = CommittedTx {
            changes: vec![change_insert(3, "action", 9.0, 0.6, 0.8), change_delete(1)],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        m.apply_tx(&tx).unwrap();

        // view answer == from-scratch recomputation on the survivors
        let w2 = ((0.8 - 1.0) / 0.1f64).exp();
        let w3 = ((1.0 - 1.0) / 0.1f64).exp();
        let want = (7.0 * w2 + 9.0 * w3) / (w2 + w3);
        let got = m.db.views[0].read();
        let answer = got.iter().find(|(g, _)| *g == 0).unwrap().1;
        assert!((answer - want).abs() <= 1e-12 * want.abs());
    }

    // ------------------------------------------------ update apply

    fn seeded_mirror() -> (Mirror, Vec<String>) {
        let names = snap_cols();
        let rows = vec![
            row(1, "action", 5.0, 1.0, 0.0),
            row(2, "drama", 7.0, 0.0, 1.0),
        ];
        let m = Mirror::from_snapshot(TableMap::cdc_movies(), &names, &rows).unwrap();
        (m, names)
    }

    fn mirror_pair(m: &Mirror, id: f64) -> (f64, String) {
        let t = &m.db.catalog.tables["cdc_movies"];
        let ids = match &t.columns["movie_id"] {
            Column::ScalarF64(v) => v,
            _ => panic!(),
        };
        let i = ids.iter().position(|&v| v == id).expect("row present");
        let rating = match &t.columns["rating"] {
            Column::ScalarF64(v) => v[i],
            _ => panic!(),
        };
        let genre = match &t.columns["genre"] {
            Column::DictU32 { codes, dict } => dict[codes[i] as usize].clone(),
            _ => panic!(),
        };
        (rating, genre)
    }

    #[test]
    fn update_full_identity_applies_as_delete_insert() {
        let (mut m, names) = seeded_mirror();
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: Some(datums(&names, row(1, "action", 5.0, 1.0, 0.0))),
                new: datums(&names, row(1, "horror", 9.5, 1.0, 0.0)),
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert_eq!(m.apply_tx(&tx).unwrap(), 1);
        assert_eq!(m.n_rows(), 2, "update must not change the row count");
        assert_eq!(mirror_pair(&m, 1.0), (9.5, "horror".to_string()));
    }

    #[test]
    fn update_without_old_tuple_locates_by_new_pk() {
        // REPLICA IDENTITY DEFAULT, key untouched: old is None
        let (mut m, names) = seeded_mirror();
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: None,
                new: datums(&names, row(2, "drama", 3.25, 0.0, 1.0)),
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert_eq!(m.apply_tx(&tx).unwrap(), 1);
        assert_eq!(mirror_pair(&m, 2.0), (3.25, "drama".to_string()));
    }

    #[test]
    fn update_pk_change_moves_the_row() {
        // key-only 'K' old tuple: pk value + NULL non-key columns
        let (mut m, names) = seeded_mirror();
        let key_only = datums(&names, vec![Some("1".into()), None, None, None, None, None]);
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: Some(key_only),
                new: datums(&names, row(42, "action", 5.0, 1.0, 0.0)),
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert_eq!(m.apply_tx(&tx).unwrap(), 1);
        assert_eq!(m.n_rows(), 2);
        assert_eq!(mirror_pair(&m, 42.0), (5.0, "action".to_string()));
    }

    #[test]
    fn update_unchanged_datum_resolves_from_mirror() {
        // untouched TOAST arrives as Unchanged: the mirror's current
        // row supplies the value — that is WHY the mirror exists.
        let (mut m, names) = seeded_mirror();
        let mut new = datums(&names, row(1, "action", 8.0, 1.0, 0.0));
        new[1].1 = TupleDatum::Unchanged; // genre (label col)
        new[3].1 = TupleDatum::Unchanged; // year (scalar col)
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: None,
                new,
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        assert_eq!(m.apply_tx(&tx).unwrap(), 1);
        // rating updated, genre resolved from the pre-update row
        assert_eq!(mirror_pair(&m, 1.0), (8.0, "action".to_string()));
        let t = &m.db.catalog.tables["cdc_movies"];
        let years = match &t.columns["year"] {
            Column::ScalarF64(v) => v,
            _ => panic!(),
        };
        assert!(years.contains(&2000.0), "year preserved");
    }

    #[test]
    fn update_of_absent_row_is_typed_error() {
        let (mut m, names) = seeded_mirror();
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: Some(datums(&names, row(99, "action", 5.0, 1.0, 0.0))),
                new: datums(&names, row(99, "action", 6.0, 1.0, 0.0)),
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        let err = m.apply_tx(&tx).unwrap_err().to_string();
        assert!(err.contains("out of sync"), "got: {err}");
    }

    #[test]
    fn update_null_in_mapped_column_is_typed_error() {
        let (mut m, names) = seeded_mirror();
        let mut new = datums(&names, row(1, "action", 5.0, 1.0, 0.0));
        new[2].1 = TupleDatum::Null; // rating
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: None,
                new,
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        let err = m.apply_tx(&tx).unwrap_err().to_string();
        assert!(err.contains("NULL in column rating"), "got: {err}");
        // typed error BEFORE any mutation: the mirror is untouched
        assert_eq!(mirror_pair(&m, 1.0), (5.0, "action".to_string()));
    }

    #[test]
    fn update_unchanged_pk_without_old_tuple_is_typed_error() {
        let (mut m, names) = seeded_mirror();
        let mut new = datums(&names, row(1, "action", 5.0, 1.0, 0.0));
        new[0].1 = TupleDatum::Unchanged; // movie_id
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: None,
                new,
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        let err = m.apply_tx(&tx).unwrap_err().to_string();
        assert!(err.contains("cannot locate the row"), "got: {err}");
    }

    #[test]
    fn view_stays_fresh_through_update_group_inverse() {
        use ndarray::Array1;
        let (mut m, names) = seeded_mirror();
        let x = Array1::from_vec(vec![0.6, 0.8]);
        m.db.create_view("v", "cdc_movies", "genre", "rating", "emb", &x, 0.1)
            .unwrap();
        // move row 2 from drama to action with a new rating: the view
        // must absorb a delete (group inverse) + insert
        let tx = CommittedTx {
            changes: vec![RowChange::Update {
                rel: "cdc_movies".into(),
                old: Some(datums(&names, row(2, "drama", 7.0, 0.0, 1.0))),
                new: datums(&names, row(2, "action", 4.0, 0.0, 1.0)),
            }],
            commit_ts_us: 0,
            end_lsn: 1,
        };
        m.apply_tx(&tx).unwrap();
        // from-scratch recomputation: action = {row1 (1,0,5.0), row2 (0,1,4.0)}
        let w1 = ((0.6 - 1.0) / 0.1f64).exp();
        let w2 = ((0.8 - 1.0) / 0.1f64).exp();
        let want = (5.0 * w1 + 4.0 * w2) / (w1 + w2);
        let got = m.db.views[0].read();
        let action = got.iter().find(|(g, _)| *g == 0).unwrap().1;
        assert!(
            (action - want).abs() <= 1e-12 * want.abs(),
            "got {action}, want {want}"
        );
    }
}

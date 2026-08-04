//! Durable mirror: serialize the [`Mirror`]'s table, counters and —
//! crucially — the `last_lsn` exactly-once watermark to a single
//! file, so process death loses nothing (tests/durable_chaos.rs kills
//! the subscriber with SIGKILL and restarts FROM DISK).
//!
//! Format `BRCDCM01`, little-endian, hand-rolled (no serde — the
//! crate stays lean per constitution C1's spirit; storage is
//! disposable per C3, this file is a cache of PG's authoritative
//! state and may be deleted at any time, forcing a re-snapshot):
//!
//! ```text
//! magic[8] = "BRCDCM01"
//! last_lsn u64 | rows_applied u64 | txs_applied u64
//! TableMap: table, pk, key_col: str; label_cols, scalar_cols,
//!           key_parts: str-vec        (str = u32 len || utf-8)
//! n_cols u32, then per column: name str, tag u8,
//!   tag 1 ScalarF64: u64 n || n f64
//!   tag 2 DictU32:   u64 n || n u32 codes || str-vec dict
//!   tag 3 KeyF64:    u64 nrows || u64 ncols || row-major f64
//! fnv1a u64 over every preceding byte
//! ```
//!
//! Writes are atomic: temp file in the same directory, fsync, rename
//! over the target, fsync the directory. A torn or corrupted file
//! fails `load` with a typed error — never a panic, never silent
//! drift. Views are NOT serialized: recreate them after `load` with
//! `db.create_view` (recomputes from the table).

use std::fs;
use std::io::Write;
use std::path::Path;

use ndarray::Array2;

use bruce_query::{Column, Database, Table};

use crate::apply::{Mirror, TableMap};
use crate::CdcError;

const MAGIC: &[u8; 8] = b"BRCDCM01";

const TAG_SCALAR_F64: u8 = 1;
const TAG_DICT_U32: u8 = 2;
const TAG_KEY_F64: u8 = 3;

// ------------------------------------------------------------ encode

struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    fn new() -> Self {
        Enc {
            buf: MAGIC.to_vec(),
        }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn str_vec(&mut self, v: &[String]) {
        self.u32(v.len() as u32);
        for s in v {
            self.str(s);
        }
    }
}

/// FNV-1a over a byte slice (the durable file's integrity trailer;
/// same constants as the chaos suite's ground-truth checksum).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

// ------------------------------------------------------------ decode

struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CdcError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| CdcError::Decode("durable: length overflow".into()))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| CdcError::Decode("durable: truncated file".into()))?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, CdcError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, CdcError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, CdcError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, CdcError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, CdcError> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| CdcError::Decode(format!("durable: bad utf-8 string: {e}")))
    }
    fn str_vec(&mut self) -> Result<Vec<String>, CdcError> {
        let n = self.u32()? as usize;
        let mut v = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            v.push(self.str()?);
        }
        Ok(v)
    }
}

// ----------------------------------------------------------- Mirror

impl Mirror {
    /// Serialize the mirror (table + counters + `last_lsn` watermark)
    /// to `path`, atomically: temp file in the same directory, fsync,
    /// rename, fsync the directory. Concurrent readers of `path`
    /// always see a complete old or complete new file, never a torn
    /// one — SIGKILL mid-save leaves the previous state intact.
    pub fn save(&self, path: &Path) -> Result<(), CdcError> {
        let mut e = Enc::new();
        e.u64(self.last_lsn);
        e.u64(self.rows_applied as u64);
        e.u64(self.txs_applied as u64);
        e.str(&self.map.table);
        e.str(&self.map.pk);
        e.str(&self.map.key_col);
        e.str_vec(&self.map.label_cols);
        e.str_vec(&self.map.scalar_cols);
        e.str_vec(&self.map.key_parts);

        let t = self
            .db
            .catalog
            .tables
            .get(&self.map.table)
            .ok_or_else(|| CdcError::Apply(format!("save: no table {}", self.map.table)))?;
        // deterministic column order: scalar_cols, label_cols, key_col
        let order: Vec<&String> = self
            .map
            .scalar_cols
            .iter()
            .chain(self.map.label_cols.iter())
            .chain(std::iter::once(&self.map.key_col))
            .collect();
        e.u32(order.len() as u32);
        for name in order {
            let col = t.columns.get(name).ok_or_else(|| {
                CdcError::Apply(format!("save: mapped column {name} missing from table"))
            })?;
            e.str(name);
            match col {
                Column::ScalarF64(v) => {
                    e.u8(TAG_SCALAR_F64);
                    e.u64(v.len() as u64);
                    for &x in v {
                        e.f64(x);
                    }
                }
                Column::DictU32 { codes, dict } => {
                    e.u8(TAG_DICT_U32);
                    e.u64(codes.len() as u64);
                    for &c in codes {
                        e.u32(c);
                    }
                    e.str_vec(dict);
                }
                Column::KeyF64(a) => {
                    e.u8(TAG_KEY_F64);
                    e.u64(a.nrows() as u64);
                    e.u64(a.ncols() as u64);
                    for &x in a.iter() {
                        e.f64(x);
                    }
                }
                other => {
                    return Err(CdcError::Apply(format!(
                        "save: column {name} has unsupported kind {other:?}"
                    )))
                }
            }
        }
        let sum = fnv1a(&e.buf);
        e.u64(sum);

        let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        {
            let mut f = fs::File::create(&tmp)
                .map_err(|err| CdcError::Io(format!("create {}: {err}", tmp.display())))?;
            f.write_all(&e.buf)
                .map_err(|err| CdcError::Io(format!("write {}: {err}", tmp.display())))?;
            f.sync_all()
                .map_err(|err| CdcError::Io(format!("fsync {}: {err}", tmp.display())))?;
        }
        fs::rename(&tmp, path).map_err(|err| {
            CdcError::Io(format!(
                "rename {} -> {}: {err}",
                tmp.display(),
                path.display()
            ))
        })?;
        if let Some(d) = dir {
            // fsync the directory so the rename itself is durable
            if let Ok(df) = fs::File::open(d) {
                let _ = df.sync_all();
            }
        }
        Ok(())
    }

    /// Rebuild a mirror from a file written by [`Mirror::save`].
    /// Verifies magic and the FNV-1a trailer before decoding; any
    /// mismatch, truncation or malformed section is a typed error.
    /// Views are not restored — recreate them with `db.create_view`.
    pub fn load(path: &Path) -> Result<Mirror, CdcError> {
        let bytes =
            fs::read(path).map_err(|e| CdcError::Io(format!("read {}: {e}", path.display())))?;
        if bytes.len() < MAGIC.len() + 8 {
            return Err(CdcError::Decode(format!(
                "durable: {} too short ({} bytes)",
                path.display(),
                bytes.len()
            )));
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(CdcError::Decode(format!(
                "durable: {} has wrong magic (not a BRCDCM01 file)",
                path.display()
            )));
        }
        let (body, trailer) = bytes.split_at(bytes.len() - 8);
        let want = u64::from_le_bytes(trailer.try_into().unwrap());
        let got = fnv1a(body);
        if got != want {
            return Err(CdcError::Decode(format!(
                "durable: {} checksum mismatch (file corrupt)",
                path.display()
            )));
        }

        let mut d = Dec {
            buf: body,
            pos: MAGIC.len(),
        };
        let last_lsn = d.u64()?;
        let rows_applied = d.u64()? as usize;
        let txs_applied = d.u64()? as usize;
        let map = TableMap {
            table: d.str()?,
            pk: d.str()?,
            key_col: d.str()?,
            label_cols: d.str_vec()?,
            scalar_cols: d.str_vec()?,
            key_parts: d.str_vec()?,
        };
        let n_cols = d.u32()? as usize;
        let mut table = Table::default();
        for _ in 0..n_cols {
            let name = d.str()?;
            let col = match d.u8()? {
                TAG_SCALAR_F64 => {
                    let n = d.u64()? as usize;
                    let mut v = Vec::with_capacity(n.min(1 << 24));
                    for _ in 0..n {
                        v.push(d.f64()?);
                    }
                    Column::ScalarF64(v)
                }
                TAG_DICT_U32 => {
                    let n = d.u64()? as usize;
                    let mut codes = Vec::with_capacity(n.min(1 << 24));
                    for _ in 0..n {
                        codes.push(d.u32()?);
                    }
                    let dict = d.str_vec()?;
                    // codes must index into dict — a corrupt file must
                    // not smuggle a panic into later stats collection
                    if let Some(&bad) = codes.iter().find(|&&c| c as usize >= dict.len()) {
                        return Err(CdcError::Decode(format!(
                            "durable: dict column {name} code {bad} out of range \
                             (dict has {} labels)",
                            dict.len()
                        )));
                    }
                    Column::DictU32 { codes, dict }
                }
                TAG_KEY_F64 => {
                    let nrows = d.u64()? as usize;
                    let ncols = d.u64()? as usize;
                    let mut v = Vec::with_capacity((nrows * ncols).min(1 << 26));
                    for _ in 0..nrows * ncols {
                        v.push(d.f64()?);
                    }
                    let a = Array2::from_shape_vec((nrows, ncols), v).map_err(|e| {
                        CdcError::Decode(format!("durable: key column {name}: {e}"))
                    })?;
                    Column::KeyF64(a)
                }
                t => {
                    return Err(CdcError::Decode(format!(
                        "durable: column {name} has unknown tag {t}"
                    )))
                }
            };
            table.columns.insert(name, col);
        }
        if d.pos != body.len() {
            return Err(CdcError::Decode(format!(
                "durable: {} trailing garbage ({} bytes past the last column)",
                path.display(),
                body.len() - d.pos
            )));
        }

        let mut db = Database::new();
        db.register(&map.table, table);
        Ok(Mirror {
            db,
            map,
            rows_applied,
            txs_applied,
            last_lsn,
        })
    }
}

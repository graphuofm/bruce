//! pgoutput (logical replication output plugin) message decoding,
//! protocol version 1, plus the replication-stream envelope (XLogData
//! / primary keepalive). Text tuple format only — the subscription
//! never asks for `binary`.

use crate::CdcError;

/// One replication-stream CopyData payload, decoded.
pub enum WalMsg {
    /// A WAL data payload carrying one pgoutput message.
    XLogData {
        /// Start LSN of the payload.
        wal_start: u64,
        /// Current end of WAL on the server.
        wal_end: u64,
        /// The decoded pgoutput message.
        msg: PgoutputMsg,
    },
    /// Primary keepalive; `reply_requested` demands an immediate
    /// standby status update (else the walsender disconnects).
    Keepalive {
        /// Current end of WAL on the server.
        wal_end: u64,
        /// True iff the server asked for an immediate status reply.
        reply_requested: bool,
    },
}

/// One column of TupleData, with the three wire markers kept
/// distinct.
///
/// DEFINED SEMANTICS (pinned by tests/conformance.rs
/// `unchanged_toast_is_distinct_from_null` and the live-PG round trip
/// in tests/update_toast.rs): `Null` is SQL NULL (`'n'`), `Unchanged`
/// is the unchanged-TOAST marker (`'u'` — the value was not modified
/// by this UPDATE and is stored out of line, so the walsender omits
/// it), `Text` is a materialized value (`'t'`). Before v1 both `'n'`
/// and `'u'` decoded to `None`; an Update-capable apply path must
/// resolve `Unchanged` from the mirror's current row, which is only
/// possible if the two are distinguishable here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TupleDatum {
    /// SQL NULL (`'n'`).
    Null,
    /// Unchanged TOAST datum (`'u'`): resolve from the mirror.
    Unchanged,
    /// Materialized text value (`'t'`).
    Text(String),
}

impl TupleDatum {
    /// Convenience constructor for tests and builders.
    pub fn text(v: &str) -> Self {
        TupleDatum::Text(v.to_string())
    }
}

/// A decoded pgoutput message (proto_version 1 subset).
pub enum PgoutputMsg {
    /// Transaction start.
    Begin {
        /// LSN of the commit record this transaction ends at.
        final_lsn: u64,
        /// Commit timestamp, microseconds since the PG epoch.
        commit_ts_us: i64,
        /// Transaction id.
        xid: u32,
    },
    /// Transaction end.
    Commit {
        /// End LSN of the commit record + 1 (the ack point).
        end_lsn: u64,
        /// Commit timestamp, microseconds since the PG epoch.
        commit_ts_us: i64,
    },
    /// Relation metadata; sent before the first change of a relation
    /// on this connection (and again on schema change).
    Relation {
        /// Relation OID (keys Insert/Delete/Update messages).
        rel_id: u32,
        /// Schema-qualified name parts.
        namespace: String,
        /// Relation name.
        name: String,
        /// Replica identity byte: `'d'` default, `'n'` nothing, `'f'`
        /// full, `'i'` index — an Update-capable apply path must know
        /// it (REPLICA IDENTITY NOTHING makes the old row
        /// unidentifiable).
        replica_identity: u8,
        /// Column names in tuple order.
        columns: Vec<String>,
    },
    /// Row insert: the new tuple.
    Insert {
        /// Relation OID.
        rel_id: u32,
        /// New tuple, datums in relation column order.
        new: Vec<TupleDatum>,
    },
    /// Row delete: the old tuple (full with REPLICA IDENTITY FULL,
    /// else key columns only).
    Delete {
        /// Relation OID.
        rel_id: u32,
        /// Old tuple, datums in relation column order.
        old: Vec<TupleDatum>,
    },
    /// Row update: optional old tuple + new tuple.
    Update {
        /// Relation OID.
        rel_id: u32,
        /// Old tuple if the update touched identity columns (or
        /// always, with REPLICA IDENTITY FULL).
        old: Option<Vec<TupleDatum>>,
        /// New tuple (may carry [`TupleDatum::Unchanged`] for
        /// untouched TOASTed columns).
        new: Vec<TupleDatum>,
    },
    /// Any other message type (Origin, Type, Truncate, Message) —
    /// carried for observability, ignored by the apply path.
    Other(u8),
}

/// Decode one CopyData payload from the replication stream.
pub fn decode_wal(payload: &[u8]) -> Result<WalMsg, CdcError> {
    let mut c = Cursor::new(payload);
    match c.u8()? {
        b'w' => {
            let wal_start = c.u64()?;
            let wal_end = c.u64()?;
            let _send_ts = c.i64()?;
            let msg = decode_pgoutput(c.rest())?;
            Ok(WalMsg::XLogData {
                wal_start,
                wal_end,
                msg,
            })
        }
        b'k' => {
            let wal_end = c.u64()?;
            let _send_ts = c.i64()?;
            let reply_requested = c.u8()? != 0;
            Ok(WalMsg::Keepalive {
                wal_end,
                reply_requested,
            })
        }
        t => Err(CdcError::Decode(format!(
            "unknown stream message {:?}",
            t as char
        ))),
    }
}

/// Decode one pgoutput message body.
pub fn decode_pgoutput(buf: &[u8]) -> Result<PgoutputMsg, CdcError> {
    let mut c = Cursor::new(buf);
    match c.u8()? {
        b'B' => Ok(PgoutputMsg::Begin {
            final_lsn: c.u64()?,
            commit_ts_us: c.i64()?,
            xid: c.u32()?,
        }),
        b'C' => {
            let _flags = c.u8()?;
            let _commit_lsn = c.u64()?;
            let end_lsn = c.u64()?;
            let commit_ts_us = c.i64()?;
            Ok(PgoutputMsg::Commit {
                end_lsn,
                commit_ts_us,
            })
        }
        b'R' => {
            let rel_id = c.u32()?;
            let namespace = c.cstr()?;
            let name = c.cstr()?;
            let replica_identity = c.u8()?;
            let ncols = c.u16()? as usize;
            let mut columns = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                let _flags = c.u8()?;
                columns.push(c.cstr()?);
                let _typ_oid = c.u32()?;
                let _typmod = c.i32()?;
            }
            Ok(PgoutputMsg::Relation {
                rel_id,
                namespace,
                name,
                replica_identity,
                columns,
            })
        }
        b'I' => {
            let rel_id = c.u32()?;
            let kind = c.u8()?;
            if kind != b'N' {
                return Err(CdcError::Decode(format!(
                    "insert tuple kind {:?}, want 'N'",
                    kind as char
                )));
            }
            Ok(PgoutputMsg::Insert {
                rel_id,
                new: tuple(&mut c)?,
            })
        }
        b'D' => {
            let rel_id = c.u32()?;
            let kind = c.u8()?;
            if kind != b'O' && kind != b'K' {
                return Err(CdcError::Decode(format!(
                    "delete tuple kind {:?}, want 'O'/'K'",
                    kind as char
                )));
            }
            Ok(PgoutputMsg::Delete {
                rel_id,
                old: tuple(&mut c)?,
            })
        }
        b'U' => {
            let rel_id = c.u32()?;
            let mut kind = c.u8()?;
            let mut old = None;
            if kind == b'O' || kind == b'K' {
                old = Some(tuple(&mut c)?);
                kind = c.u8()?;
            }
            if kind != b'N' {
                return Err(CdcError::Decode(format!(
                    "update tuple kind {:?}, want 'N'",
                    kind as char
                )));
            }
            Ok(PgoutputMsg::Update {
                rel_id,
                old,
                new: tuple(&mut c)?,
            })
        }
        t => Ok(PgoutputMsg::Other(t)),
    }
}

/// TupleData: per column 'n' (SQL NULL), 'u' (unchanged TOAST), or
/// 't' (length-prefixed text) — the three wire markers map 1:1 onto
/// [`TupleDatum`]; see its DEFINED SEMANTICS doc.
fn tuple(c: &mut Cursor<'_>) -> Result<Vec<TupleDatum>, CdcError> {
    let n = c.u16()? as usize;
    let mut cols = Vec::with_capacity(n);
    for _ in 0..n {
        match c.u8()? {
            b'n' => cols.push(TupleDatum::Null),
            b'u' => cols.push(TupleDatum::Unchanged),
            b't' => {
                let len = c.i32()?;
                if len < 0 {
                    return Err(CdcError::Decode("negative tuple text length".into()));
                }
                let bytes = c.take(len as usize)?;
                cols.push(TupleDatum::Text(
                    String::from_utf8_lossy(bytes).into_owned(),
                ));
            }
            k => {
                return Err(CdcError::Decode(format!(
                    "tuple column kind {:?} (binary streams unsupported)",
                    k as char
                )))
            }
        }
    }
    Ok(cols)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CdcError> {
        let end = self.pos + n;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| CdcError::Decode("short pgoutput message".into()))?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, CdcError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CdcError> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, CdcError> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, CdcError> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, CdcError> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, CdcError> {
        Ok(i64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn cstr(&mut self) -> Result<String, CdcError> {
        let start = self.pos;
        let end = self.buf[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|e| start + e)
            .ok_or_else(|| CdcError::Decode("unterminated string".into()))?;
        let s = String::from_utf8_lossy(&self.buf[start..end]).into_owned();
        self.pos = end + 1;
        Ok(s)
    }

    fn rest(&mut self) -> &'a [u8] {
        let s = &self.buf[self.pos..];
        self.pos = self.buf.len();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Datum spec for the byte builder: `Some(s)` = text, `None` =
    /// SQL NULL; unchanged TOAST is pushed by `toast_marker`.
    fn tuple_bytes(cols: &[Option<&str>]) -> Vec<u8> {
        let mut b = (cols.len() as u16).to_be_bytes().to_vec();
        for c in cols {
            match c {
                None => b.push(b'n'),
                Some(s) => {
                    b.push(b't');
                    b.extend_from_slice(&(s.len() as i32).to_be_bytes());
                    b.extend_from_slice(s.as_bytes());
                }
            }
        }
        b
    }

    #[test]
    fn decodes_insert() {
        let mut m = vec![b'I'];
        m.extend_from_slice(&77u32.to_be_bytes());
        m.push(b'N');
        m.extend_from_slice(&tuple_bytes(&[Some("1"), Some("action"), None]));
        match decode_pgoutput(&m).unwrap() {
            PgoutputMsg::Insert { rel_id, new } => {
                assert_eq!(rel_id, 77);
                assert_eq!(
                    new,
                    vec![
                        TupleDatum::text("1"),
                        TupleDatum::text("action"),
                        TupleDatum::Null
                    ]
                );
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decodes_delete_old_tuple() {
        let mut m = vec![b'D'];
        m.extend_from_slice(&77u32.to_be_bytes());
        m.push(b'O');
        m.extend_from_slice(&tuple_bytes(&[Some("42")]));
        match decode_pgoutput(&m).unwrap() {
            PgoutputMsg::Delete { rel_id, old } => {
                assert_eq!(rel_id, 77);
                assert_eq!(old, vec![TupleDatum::text("42")]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decodes_update_with_key_old_tuple_and_unchanged_toast() {
        // U rel_id 'K' <old: key only> 'N' <new with 'u' marker>
        let mut m = vec![b'U'];
        m.extend_from_slice(&77u32.to_be_bytes());
        m.push(b'K');
        m.extend_from_slice(&tuple_bytes(&[Some("42"), None]));
        m.push(b'N');
        // new tuple: id text, payload UNCHANGED ('u') — build by hand
        let mut new = 2u16.to_be_bytes().to_vec();
        new.push(b't');
        new.extend_from_slice(&2i32.to_be_bytes());
        new.extend_from_slice(b"42");
        new.push(b'u');
        m.extend_from_slice(&new);
        match decode_pgoutput(&m).unwrap() {
            PgoutputMsg::Update { rel_id, old, new } => {
                assert_eq!(rel_id, 77);
                assert_eq!(
                    old,
                    Some(vec![TupleDatum::text("42"), TupleDatum::Null]),
                    "'K' old tuple: key value + NULL for non-key"
                );
                assert_eq!(new, vec![TupleDatum::text("42"), TupleDatum::Unchanged]);
                assert_ne!(
                    new[1],
                    TupleDatum::Null,
                    "unchanged TOAST must be distinguishable from NULL"
                );
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decodes_update_without_old_tuple() {
        // replica identity DEFAULT, key untouched: no old tuple at all
        let mut m = vec![b'U'];
        m.extend_from_slice(&77u32.to_be_bytes());
        m.push(b'N');
        m.extend_from_slice(&tuple_bytes(&[Some("42"), Some("v2")]));
        match decode_pgoutput(&m).unwrap() {
            PgoutputMsg::Update { old, new, .. } => {
                assert!(old.is_none());
                assert_eq!(new, vec![TupleDatum::text("42"), TupleDatum::text("v2")]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decodes_relation() {
        let mut m = vec![b'R'];
        m.extend_from_slice(&77u32.to_be_bytes());
        m.extend_from_slice(b"public\0cdc_movies\0");
        m.push(b'f'); // replica identity full
        m.extend_from_slice(&2u16.to_be_bytes());
        m.push(1);
        m.extend_from_slice(b"movie_id\0");
        m.extend_from_slice(&23u32.to_be_bytes());
        m.extend_from_slice(&(-1i32).to_be_bytes());
        m.push(0);
        m.extend_from_slice(b"genre\0");
        m.extend_from_slice(&25u32.to_be_bytes());
        m.extend_from_slice(&(-1i32).to_be_bytes());
        match decode_pgoutput(&m).unwrap() {
            PgoutputMsg::Relation {
                rel_id,
                namespace,
                name,
                replica_identity,
                columns,
            } => {
                assert_eq!(
                    (rel_id, namespace.as_str(), name.as_str()),
                    (77, "public", "cdc_movies")
                );
                assert_eq!(replica_identity, b'f', "identity byte is carried");
                assert_eq!(columns, vec!["movie_id", "genre"]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decodes_keepalive_envelope() {
        let mut m = vec![b'k'];
        m.extend_from_slice(&0x1_0000_0000u64.to_be_bytes());
        m.extend_from_slice(&0i64.to_be_bytes());
        m.push(1);
        match decode_wal(&m).unwrap() {
            WalMsg::Keepalive {
                wal_end,
                reply_requested,
            } => {
                assert_eq!(wal_end, 0x1_0000_0000);
                assert!(reply_requested);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn short_message_is_an_error_not_a_panic() {
        assert!(decode_pgoutput(&[b'I', 0, 0]).is_err());
        assert!(decode_wal(&[b'w', 1, 2]).is_err());
    }
}

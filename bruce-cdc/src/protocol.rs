//! Raw PostgreSQL frontend/backend protocol (v3) over the local unix
//! socket. Only what the maintenance plane needs and released
//! rust-postgres does not provide: the `replication=database` startup
//! parameter, replication commands through the simple-query protocol,
//! and the CopyBoth stream of START_REPLICATION (with standby status
//! updates). Trust auth only — this speaks to the local sidecar
//! instance, not the network.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::CdcError;

/// Microseconds between the Unix epoch and the PostgreSQL epoch
/// (2000-01-01T00:00:00Z).
pub const PG_EPOCH_OFFSET_US: i64 = 946_684_800_000_000;

/// A started message must complete within this bound (local socket).
const MID_MESSAGE_DEADLINE: Duration = Duration::from_secs(10);

/// Microseconds since the PostgreSQL epoch, from the system clock.
pub fn now_pg_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
        - PG_EPOCH_OFFSET_US
}

/// Format an LSN as the textual `X/Y` form.
pub fn fmt_lsn(lsn: u64) -> String {
    format!("{:X}/{:X}", lsn >> 32, lsn & 0xFFFF_FFFF)
}

/// Parse the textual `X/Y` LSN form.
pub fn parse_lsn(s: &str) -> Result<u64, CdcError> {
    let (hi, lo) = s
        .split_once('/')
        .ok_or_else(|| CdcError::Decode(format!("bad lsn {s:?}")))?;
    let hi =
        u64::from_str_radix(hi, 16).map_err(|e| CdcError::Decode(format!("bad lsn {s:?}: {e}")))?;
    let lo =
        u64::from_str_radix(lo, 16).map_err(|e| CdcError::Decode(format!("bad lsn {s:?}: {e}")))?;
    Ok((hi << 32) | lo)
}

/// One simple-query result: column names + text rows.
pub type TextResult = (Vec<String>, Vec<Vec<Option<String>>>);

/// One backend message: tag byte + payload (length header stripped).
pub struct BackendMsg {
    /// Message type byte.
    pub tag: u8,
    /// Message body.
    pub payload: Vec<u8>,
}

/// A connection in the v3 protocol; optionally a walsender
/// (`replication=database`) connection.
pub struct PgConn {
    stream: UnixStream,
}

impl PgConn {
    /// Connect over the unix socket and complete the (trust-auth)
    /// startup handshake.
    pub fn connect(
        socket_dir: &str,
        port: u16,
        db: &str,
        user: &str,
        replication: bool,
    ) -> Result<Self, CdcError> {
        let path = format!("{socket_dir}/.s.PGSQL.{port}");
        let stream =
            UnixStream::connect(&path).map_err(|e| CdcError::Io(format!("connect {path}: {e}")))?;
        let mut conn = PgConn { stream };
        conn.send_startup(db, user, replication)?;
        conn.await_ready()?;
        Ok(conn)
    }

    fn send_startup(&mut self, db: &str, user: &str, replication: bool) -> Result<(), CdcError> {
        let mut body = Vec::new();
        body.extend_from_slice(&196608i32.to_be_bytes()); // protocol 3.0
        let mut kv = vec![
            ("user", user),
            ("database", db),
            ("application_name", "bruce-cdc"),
        ];
        if replication {
            kv.push(("replication", "database"));
        }
        for (k, v) in kv {
            body.extend_from_slice(k.as_bytes());
            body.push(0);
            body.extend_from_slice(v.as_bytes());
            body.push(0);
        }
        body.push(0);
        self.stream
            .write_all(&((body.len() + 4) as i32).to_be_bytes())?;
        self.stream.write_all(&body)?;
        Ok(())
    }

    fn await_ready(&mut self) -> Result<(), CdcError> {
        loop {
            let m = self.must_read()?;
            match m.tag {
                b'R' => {
                    let code = be_i32(&m.payload, 0)?;
                    if code != 0 {
                        return Err(CdcError::Protocol(format!(
                            "auth method {code} not supported (trust only)"
                        )));
                    }
                }
                b'S' | b'K' | b'N' => {}
                b'Z' => return Ok(()),
                b'E' => return Err(backend_err(&m.payload)),
                t => {
                    return Err(CdcError::Protocol(format!(
                        "unexpected startup message {:?}",
                        t as char
                    )))
                }
            }
        }
    }

    /// Read one backend message. With `idle`, a timeout while waiting
    /// for the FIRST byte returns `Ok(None)`; a message once begun is
    /// always read to completion (bounded by a mid-message deadline).
    pub fn read_msg(&mut self, idle: Option<Duration>) -> Result<Option<BackendMsg>, CdcError> {
        self.stream
            .set_read_timeout(idle)
            .map_err(|e| CdcError::Io(format!("set_read_timeout: {e}")))?;
        let mut tag = [0u8; 1];
        loop {
            match self.stream.read(&mut tag) {
                Ok(0) => return Err(CdcError::Protocol("connection closed".into())),
                Ok(_) => break,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    if idle.is_some() {
                        return Ok(None);
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        let mut len_buf = [0u8; 4];
        self.read_full(&mut len_buf)?;
        let len = i32::from_be_bytes(len_buf);
        if !(4..=(1 << 30)).contains(&len) {
            return Err(CdcError::Protocol(format!("bad message length {len}")));
        }
        let mut payload = vec![0u8; len as usize - 4];
        self.read_full(&mut payload)?;
        Ok(Some(BackendMsg {
            tag: tag[0],
            payload,
        }))
    }

    fn must_read(&mut self) -> Result<BackendMsg, CdcError> {
        self.read_msg(None)?
            .ok_or_else(|| CdcError::Protocol("unexpected idle".into()))
    }

    fn read_full(&mut self, buf: &mut [u8]) -> Result<(), CdcError> {
        let deadline = Instant::now() + MID_MESSAGE_DEADLINE;
        let mut off = 0;
        while off < buf.len() {
            match self.stream.read(&mut buf[off..]) {
                Ok(0) => return Err(CdcError::Protocol("connection closed mid-message".into())),
                Ok(n) => off += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                    ) =>
                {
                    if Instant::now() > deadline {
                        return Err(CdcError::Protocol("timeout mid-message".into()));
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    fn send_msg(&mut self, tag: u8, payload: &[u8]) -> Result<(), CdcError> {
        let mut buf = Vec::with_capacity(5 + payload.len());
        buf.push(tag);
        buf.extend_from_slice(&((payload.len() + 4) as i32).to_be_bytes());
        buf.extend_from_slice(payload);
        self.stream.write_all(&buf)?;
        Ok(())
    }

    /// Run one SQL (or walsender replication) command through the
    /// simple-query protocol. Returns (column names, text rows).
    pub fn simple_query(&mut self, sql: &str) -> Result<TextResult, CdcError> {
        let mut payload = sql.as_bytes().to_vec();
        payload.push(0);
        self.send_msg(b'Q', &payload)?;
        let mut cols = Vec::new();
        let mut rows = Vec::new();
        let mut err: Option<CdcError> = None;
        loop {
            let m = self.must_read()?;
            match m.tag {
                b'T' => cols = row_description(&m.payload)?,
                b'D' => rows.push(data_row(&m.payload)?),
                b'C' | b'I' | b'N' | b'S' => {}
                b'E' => err = Some(backend_err(&m.payload)),
                b'Z' => break,
                t => {
                    return Err(CdcError::Protocol(format!(
                        "unexpected message {:?} in simple query",
                        t as char
                    )))
                }
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok((cols, rows)),
        }
    }

    /// Enter CopyBoth streaming from the slot at `start_lsn` (`0/0`
    /// resumes from the slot's confirmed_flush_lsn — the native
    /// checkpoint). After `Ok`, only the copy-stream methods are
    /// valid on this connection.
    pub fn start_replication(
        &mut self,
        slot: &str,
        publication: &str,
        start_lsn: u64,
    ) -> Result<(), CdcError> {
        let sql = format!(
            "START_REPLICATION SLOT {slot} LOGICAL {} (proto_version '1', publication_names '{publication}')",
            fmt_lsn(start_lsn)
        );
        let mut payload = sql.as_bytes().to_vec();
        payload.push(0);
        self.send_msg(b'Q', &payload)?;
        loop {
            let m = self.must_read()?;
            match m.tag {
                b'W' => return Ok(()),
                b'N' | b'S' => {}
                b'E' => {
                    let e = backend_err(&m.payload);
                    self.drain_to_ready()?;
                    return Err(e);
                }
                t => {
                    return Err(CdcError::Protocol(format!(
                        "unexpected message {:?} starting replication",
                        t as char
                    )))
                }
            }
        }
    }

    fn drain_to_ready(&mut self) -> Result<(), CdcError> {
        loop {
            if self.must_read()?.tag == b'Z' {
                return Ok(());
            }
        }
    }

    /// One CopyData payload from the stream; `None` on idle timeout.
    pub fn read_copy(&mut self, idle: Duration) -> Result<Option<Vec<u8>>, CdcError> {
        loop {
            let Some(m) = self.read_msg(Some(idle))? else {
                return Ok(None);
            };
            match m.tag {
                b'd' => return Ok(Some(m.payload)),
                b'c' => return Err(CdcError::Protocol("server ended the copy stream".into())),
                b'E' => return Err(backend_err(&m.payload)),
                b'N' | b'S' => {}
                t => {
                    return Err(CdcError::Protocol(format!(
                        "unexpected message {:?} in copy stream",
                        t as char
                    )))
                }
            }
        }
    }

    /// Tear the socket down at the OS level (both directions),
    /// without waiting for the peer. Idempotent; every later use of
    /// this connection fails fast with an Io error.
    ///
    /// Production change justified by tests/chaos.rs (full-suite
    /// run): during a fast shutdown PostgreSQL waits for THIS
    /// walsender's client, while refusing new connections (57P03) —
    /// a reconnect that keeps the dead connection open until the new
    /// dial succeeds therefore deadlocks the shutdown. The socket
    /// must die before the redial starts.
    pub fn shutdown(&self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }

    /// Standby status update: acknowledge received / flushed /
    /// applied LSNs (each is "last byte + 1"). Advancing `flushed`
    /// advances the slot's confirmed_flush_lsn.
    pub fn send_status(
        &mut self,
        received: u64,
        flushed: u64,
        applied: u64,
        reply: bool,
    ) -> Result<(), CdcError> {
        let mut p = Vec::with_capacity(34);
        p.push(b'r');
        p.extend_from_slice(&received.to_be_bytes());
        p.extend_from_slice(&flushed.to_be_bytes());
        p.extend_from_slice(&applied.to_be_bytes());
        p.extend_from_slice(&now_pg_us().to_be_bytes());
        p.push(u8::from(reply));
        self.send_msg(b'd', &p)
    }
}

fn backend_err(payload: &[u8]) -> CdcError {
    // ErrorResponse: (field-type byte, cstring)* then a zero byte.
    let mut code = String::new();
    let mut message = String::new();
    let mut pos = 0;
    while pos < payload.len() && payload[pos] != 0 {
        let field = payload[pos];
        pos += 1;
        let end = payload[pos..]
            .iter()
            .position(|&b| b == 0)
            .map(|e| pos + e)
            .unwrap_or(payload.len());
        let text = String::from_utf8_lossy(&payload[pos..end]).into_owned();
        pos = end + 1;
        match field {
            b'C' => code = text,
            b'M' => message = text,
            _ => {}
        }
    }
    CdcError::Backend { code, message }
}

fn be_i32(buf: &[u8], at: usize) -> Result<i32, CdcError> {
    buf.get(at..at + 4)
        .map(|b| i32::from_be_bytes(b.try_into().unwrap()))
        .ok_or_else(|| CdcError::Decode("short message (i32)".into()))
}

fn be_i16(buf: &[u8], at: usize) -> Result<i16, CdcError> {
    buf.get(at..at + 2)
        .map(|b| i16::from_be_bytes(b.try_into().unwrap()))
        .ok_or_else(|| CdcError::Decode("short message (i16)".into()))
}

fn row_description(payload: &[u8]) -> Result<Vec<String>, CdcError> {
    let n = be_i16(payload, 0)? as usize;
    let mut pos = 2;
    let mut names = Vec::with_capacity(n);
    for _ in 0..n {
        let end = payload[pos..]
            .iter()
            .position(|&b| b == 0)
            .map(|e| pos + e)
            .ok_or_else(|| CdcError::Decode("unterminated field name".into()))?;
        names.push(String::from_utf8_lossy(&payload[pos..end]).into_owned());
        // table oid(4) attnum(2) typoid(4) typlen(2) typmod(4) format(2)
        pos = end + 1 + 18;
    }
    Ok(names)
}

fn data_row(payload: &[u8]) -> Result<Vec<Option<String>>, CdcError> {
    let n = be_i16(payload, 0)? as usize;
    let mut pos = 2;
    let mut row = Vec::with_capacity(n);
    for _ in 0..n {
        let len = be_i32(payload, pos)?;
        pos += 4;
        if len < 0 {
            row.push(None);
        } else {
            let end = pos + len as usize;
            let bytes = payload
                .get(pos..end)
                .ok_or_else(|| CdcError::Decode("short data row".into()))?;
            row.push(Some(String::from_utf8_lossy(bytes).into_owned()));
            pos = end;
        }
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsn_round_trip() {
        for lsn in [0u64, 0x16B374D68, u64::from(u32::MAX), u64::MAX] {
            assert_eq!(parse_lsn(&fmt_lsn(lsn)).unwrap(), lsn);
        }
        assert_eq!(parse_lsn("1/6B374D68").unwrap(), 0x16B374D68);
        assert!(parse_lsn("nope").is_err());
    }

    #[test]
    fn data_row_decodes_nulls_and_text() {
        // 2 cols: "42", NULL
        let mut p = vec![0, 2];
        p.extend_from_slice(&2i32.to_be_bytes());
        p.extend_from_slice(b"42");
        p.extend_from_slice(&(-1i32).to_be_bytes());
        let row = data_row(&p).unwrap();
        assert_eq!(row, vec![Some("42".into()), None]);
    }

    #[test]
    fn pg_epoch_offset_is_2000_01_01() {
        // 946684800 s between 1970-01-01 and 2000-01-01.
        assert_eq!(PG_EPOCH_OFFSET_US, 946_684_800 * 1_000_000);
    }
}

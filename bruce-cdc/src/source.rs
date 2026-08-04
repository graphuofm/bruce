//! The change source: committed transactions out of PostgreSQL.
//!
//! [`ChangeSource`] is the transport seam — the apply path consumes
//! committed transactions and never sees the wire. The shipped
//! implementation, [`PgOutputSource`], speaks real START_REPLICATION
//! (pgoutput, proto_version 1) over a walsender connection; a polled
//! SQL-interface source would implement the same trait.

use std::collections::HashMap;
use std::time::Duration;

use crate::pgoutput::{decode_wal, PgoutputMsg, TupleDatum, WalMsg};
use crate::protocol::PgConn;
use crate::CdcError;

/// SQLSTATE for "object already exists" (duplicate slot).
const DUPLICATE_OBJECT: &str = "42710";

/// A tuple as (column name, datum) pairs in relation order.
pub type NamedTuple = Vec<(String, TupleDatum)>;

/// One committed row change, column datums paired with the relation's
/// column names.
#[derive(Debug)]
pub enum RowChange {
    /// A row appeared.
    Insert {
        /// Relation name (unqualified).
        rel: String,
        /// (column name, datum) in relation order.
        cols: NamedTuple,
    },
    /// A row disappeared; with REPLICA IDENTITY FULL the old tuple is
    /// complete.
    Delete {
        /// Relation name (unqualified).
        rel: String,
        /// (column name, datum) of the old tuple.
        old: NamedTuple,
    },
    /// A row changed. `old` is `None` when the update touched no
    /// identity column under REPLICA IDENTITY DEFAULT (locate the row
    /// by the key columns of `new`); with FULL it is always the
    /// complete old tuple; with DEFAULT and a key change it is the
    /// key-only tuple (non-key columns [`TupleDatum::Null`]). `new`
    /// may carry [`TupleDatum::Unchanged`] for untouched TOASTed
    /// columns — the apply path resolves those from the mirror.
    Update {
        /// Relation name (unqualified).
        rel: String,
        /// (column name, datum) of the old tuple, if sent.
        old: Option<NamedTuple>,
        /// (column name, datum) of the new tuple.
        new: NamedTuple,
    },
}

/// One committed transaction, ready to apply atomically.
#[derive(Debug)]
pub struct CommittedTx {
    /// Row changes in commit order (empty if the transaction touched
    /// only unpublished tables).
    pub changes: Vec<RowChange>,
    /// Commit timestamp, microseconds since the PG epoch.
    pub commit_ts_us: i64,
    /// The ack point: acknowledging this LSN advances the slot's
    /// confirmed_flush_lsn past the transaction.
    pub end_lsn: u64,
}

/// A stream of committed transactions plus an acknowledgement path.
pub trait ChangeSource {
    /// Next committed transaction; `None` if idle for `idle`.
    fn next_tx(&mut self, idle: Duration) -> Result<Option<CommittedTx>, CdcError>;

    /// Acknowledge everything up to and including `lsn` as applied —
    /// the durable resume checkpoint.
    fn ack(&mut self, lsn: u64) -> Result<(), CdcError>;
}

/// Where and how to subscribe.
#[derive(Debug, Clone)]
pub struct SourceConfig {
    /// Unix socket directory of the PG instance.
    pub socket_dir: String,
    /// Port (names the socket file).
    pub port: u16,
    /// Database (logical decoding requires a database-attached
    /// walsender).
    pub db: String,
    /// Role to connect as.
    pub user: String,
    /// Replication slot name.
    pub slot: String,
    /// Publication to subscribe to.
    pub publication: String,
}

impl SourceConfig {
    /// The sidecar instance this repo's experiments run against.
    pub fn local_default(slot: &str) -> Self {
        SourceConfig {
            socket_dir: "/tmp".into(),
            port: 54329,
            db: "postgres".into(),
            user: std::env::var("USER").unwrap_or_else(|_| "jding".into()),
            slot: slot.into(),
            publication: "bruce_pub".into(),
        }
    }
}

/// Outcome of [`PgOutputSource::create_slot_with_snapshot`].
pub enum SlotSetup {
    /// Slot created; the connection is inside the slot's snapshot
    /// transaction — run snapshot queries, then
    /// [`PgOutputSource::commit_snapshot`].
    CreatedSnapshotOpen {
        /// The slot's consistent point.
        consistent_point: u64,
    },
    /// Slot already existed: this is a resume — no snapshot; the
    /// stream restarts from the slot's confirmed_flush_lsn.
    AlreadyExists,
}

/// Relation metadata cached from a `Relation` message.
struct RelMeta {
    name: String,
    columns: Vec<String>,
    /// `'d'` default, `'n'` nothing, `'f'` full, `'i'` index.
    replica_identity: u8,
}

/// The pgoutput -> [`CommittedTx`] state machine, connection-free so
/// conformance tests can drive it with synthetic decoded messages.
/// Feed decoded messages in stream order; a `Commit` yields the
/// buffered transaction.
#[derive(Default)]
pub struct TxAssembler {
    /// rel_id -> cached relation metadata.
    relations: HashMap<u32, RelMeta>,
    /// Changes of the transaction currently being received.
    pending: Vec<RowChange>,
    pending_ts_us: i64,
}

impl TxAssembler {
    /// Fresh assembler (no relations cached, nothing pending).
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all buffered state (redial: the new stream redelivers the
    /// cut-off transaction from Begin, and re-sends Relation).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.relations.clear();
    }

    /// Advance the state machine by one pgoutput message. Returns the
    /// completed transaction on `Commit`, `None` otherwise.
    pub fn on_msg(&mut self, msg: PgoutputMsg) -> Result<Option<CommittedTx>, CdcError> {
        match msg {
            PgoutputMsg::Begin { commit_ts_us, .. } => {
                self.pending.clear();
                self.pending_ts_us = commit_ts_us;
            }
            PgoutputMsg::Relation {
                rel_id,
                name,
                replica_identity,
                columns,
                ..
            } => {
                // A Relation may be RE-sent mid-stream on schema
                // change (e.g. ALTER TABLE ADD COLUMN). DEFINED
                // SEMANTICS (pinned by tests/conformance.rs): the
                // cache is replaced, tuples widen, and columns the
                // mirror's TableMap does not know are ignored until a
                // re-snapshot; a DROPPED mapped column fails loudly
                // at apply.
                self.relations.insert(
                    rel_id,
                    RelMeta {
                        name,
                        columns,
                        replica_identity,
                    },
                );
            }
            PgoutputMsg::Insert { rel_id, new } => {
                let (rel, cols) = self.named(rel_id, new)?;
                self.pending.push(RowChange::Insert { rel, cols });
            }
            PgoutputMsg::Delete { rel_id, old } => {
                let (rel, old) = self.named(rel_id, old)?;
                self.pending.push(RowChange::Delete { rel, old });
            }
            PgoutputMsg::Update { rel_id, old, new } => {
                // REPLICA IDENTITY NOTHING never reaches this point on
                // a real stream (PG refuses the UPDATE statement,
                // SQLSTATE 55000) — but a decoded stream is untrusted
                // input, so the guard is total. DEFINED SEMANTICS
                // (pinned by tests/conformance.rs): typed error naming
                // the fix, never a mis-applied change.
                if let Some(meta) = self.relations.get(&rel_id) {
                    if meta.replica_identity == b'n' {
                        return Err(CdcError::Apply(format!(
                            "UPDATE on {} with REPLICA IDENTITY NOTHING: the old row is \
                             unidentifiable; fix: ALTER TABLE {} REPLICA IDENTITY DEFAULT \
                             (with a primary key) or REPLICA IDENTITY FULL",
                            meta.name, meta.name
                        )));
                    }
                }
                let old = old.map(|t| self.named(rel_id, t)).transpose()?.map(|p| p.1);
                let (rel, new) = self.named(rel_id, new)?;
                self.pending.push(RowChange::Update { rel, old, new });
            }
            PgoutputMsg::Commit {
                end_lsn,
                commit_ts_us,
            } => {
                return Ok(Some(CommittedTx {
                    changes: std::mem::take(&mut self.pending),
                    commit_ts_us,
                    end_lsn,
                }));
            }
            PgoutputMsg::Other(_) => {}
        }
        Ok(None)
    }

    fn named(
        &self,
        rel_id: u32,
        values: Vec<TupleDatum>,
    ) -> Result<(String, NamedTuple), CdcError> {
        let meta = self.relations.get(&rel_id).ok_or_else(|| {
            CdcError::Protocol(format!("change before Relation for rel {rel_id}"))
        })?;
        if meta.columns.len() != values.len() {
            return Err(CdcError::Decode(format!(
                "{}: tuple has {} columns, relation has {}",
                meta.name,
                values.len(),
                meta.columns.len()
            )));
        }
        Ok((
            meta.name.clone(),
            meta.columns.iter().cloned().zip(values).collect(),
        ))
    }
}

/// Real logical replication: walsender connection, pgoutput stream,
/// standby status updates.
pub struct PgOutputSource {
    conn: PgConn,
    cfg: SourceConfig,
    /// The pgoutput -> transaction state machine.
    asm: TxAssembler,
    /// Highest WAL end seen (reported as `received` in status).
    last_recv: u64,
    /// Highest LSN acknowledged as applied.
    last_ack: u64,
}

impl PgOutputSource {
    /// Open the walsender connection (`replication=database`).
    pub fn connect(cfg: SourceConfig) -> Result<Self, CdcError> {
        let conn = PgConn::connect(&cfg.socket_dir, cfg.port, &cfg.db, &cfg.user, true)?;
        Ok(PgOutputSource {
            conn,
            cfg,
            asm: TxAssembler::new(),
            last_recv: 0,
            last_ack: 0,
        })
    }

    /// Create the slot with USE_SNAPSHOT so the initial table copy
    /// and the change stream meet exactly at the consistent point.
    /// On [`SlotSetup::CreatedSnapshotOpen`] the caller must snapshot
    /// via [`Self::snapshot_query`] and then [`Self::commit_snapshot`].
    pub fn create_slot_with_snapshot(&mut self) -> Result<SlotSetup, CdcError> {
        // USE_SNAPSHOT (PG 18) demands a read-only repeatable-read txn
        self.conn
            .simple_query("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")?;
        let created = self.conn.simple_query(&format!(
            "CREATE_REPLICATION_SLOT {} LOGICAL pgoutput USE_SNAPSHOT",
            self.cfg.slot
        ));
        match created {
            Ok((cols, rows)) => {
                let ci = cols
                    .iter()
                    .position(|c| c == "consistent_point")
                    .ok_or_else(|| CdcError::Protocol("no consistent_point column".into()))?;
                let text = rows
                    .first()
                    .and_then(|r| r.get(ci))
                    .and_then(|v| v.as_deref())
                    .ok_or_else(|| CdcError::Protocol("no consistent_point row".into()))?;
                let consistent_point = crate::protocol::parse_lsn(text)?;
                Ok(SlotSetup::CreatedSnapshotOpen { consistent_point })
            }
            Err(CdcError::Backend { ref code, .. }) if code == DUPLICATE_OBJECT => {
                self.conn.simple_query("ROLLBACK")?;
                Ok(SlotSetup::AlreadyExists)
            }
            Err(e) => Err(e),
        }
    }

    /// Run one snapshot query inside the slot-creation transaction.
    pub fn snapshot_query(&mut self, sql: &str) -> Result<crate::protocol::TextResult, CdcError> {
        self.conn.simple_query(sql)
    }

    /// Close the snapshot transaction.
    pub fn commit_snapshot(&mut self) -> Result<(), CdcError> {
        self.conn.simple_query("COMMIT")?;
        Ok(())
    }

    /// Start streaming. `0/0` resumes from the slot's
    /// confirmed_flush_lsn — the checkpoint [`ChangeSource::ack`]
    /// advances — so restart-and-resume needs no client-side state.
    pub fn start(&mut self) -> Result<(), CdcError> {
        self.conn
            .start_replication(&self.cfg.slot, &self.cfg.publication, 0)
    }

    /// Drop the dead walsender connection, dial a fresh one, and
    /// resume streaming from the slot (`0/0` = confirmed_flush_lsn).
    ///
    /// Production change justified by tests/chaos.rs
    /// pg_restart_mid_stream: without it a server restart is fatal to
    /// the subscriber. State handling:
    /// - the assembler's `pending` is cleared — a transaction cut off
    ///   mid-flight was never acked, so the new stream redelivers it
    ///   from Begin;
    /// - the assembler's `relations` is cleared — the walsender
    ///   re-sends Relation before the first change of each relation
    ///   on a new connection (and the cache could be stale across a
    ///   schema change);
    /// - `last_ack` is kept — it is only a client-side high-water
    ///   mark; the authoritative resume point is the slot's.
    pub fn redial(&mut self) -> Result<(), CdcError> {
        // Kill the old socket FIRST — PG's fast shutdown waits for
        // this very walsender client while refusing new connections,
        // so holding the dead fd through the dial would deadlock the
        // server's restart (see PgConn::shutdown).
        self.conn.shutdown();
        self.conn = PgConn::connect(
            &self.cfg.socket_dir,
            self.cfg.port,
            &self.cfg.db,
            &self.cfg.user,
            true,
        )?;
        self.asm.reset();
        self.start()
    }
}

/// Bounded-retry resilience around [`PgOutputSource`]: transport
/// failures (socket death, server restart, a walsender that has not
/// yet released the slot) are absorbed by redialing, up to
/// `max_attempts` consecutive attempts spaced `delay` apart; then the
/// last error propagates. Semantic errors (decode, apply, other
/// backend SQLSTATEs) propagate immediately — retrying cannot fix
/// them. Exactly-once across the redial is the mirror's job (see
/// `Mirror::last_lsn`); this wrapper only restores the byte stream.
///
/// Production change justified by tests/chaos.rs
/// pg_restart_mid_stream (workstream 12).
pub struct RetryingSource {
    src: PgOutputSource,
    max_attempts: u32,
    delay: Duration,
    /// Reconnects performed over the wrapper's lifetime.
    pub reconnects: u32,
}

/// SQLSTATEs that mean "try again shortly", not "you are wrong":
/// 57P01 admin_shutdown (server going down), 57P03 cannot_connect_now
/// (server starting up), 55006 object_in_use (previous walsender
/// still holds the slot).
const RETRYABLE_SQLSTATES: [&str; 3] = ["57P01", "57P03", "55006"];

/// True iff `e` is a transport-level failure a reconnect can heal
/// (socket death, protocol cut mid-stream, or one of
/// [`RETRYABLE_SQLSTATES`]). Decode/Apply errors and every other
/// backend SQLSTATE are semantic: retrying cannot fix them. Public so
/// callers driving their own loops classify identically to
/// [`RetryingSource`]; pinned by tests/chaos.rs.
pub fn is_transient(e: &CdcError) -> bool {
    match e {
        CdcError::Io(_) | CdcError::Protocol(_) => true,
        CdcError::Backend { code, .. } => RETRYABLE_SQLSTATES.contains(&code.as_str()),
        _ => false,
    }
}

impl RetryingSource {
    /// Wrap an already-streaming source.
    pub fn new(src: PgOutputSource, max_attempts: u32, delay: Duration) -> Self {
        RetryingSource {
            src,
            max_attempts,
            delay,
            reconnects: 0,
        }
    }

    fn reconnect_bounded(&mut self) -> Result<(), CdcError> {
        let mut last = CdcError::Io("reconnect: no attempt made".into());
        for _ in 0..self.max_attempts {
            std::thread::sleep(self.delay);
            match self.src.redial() {
                Ok(()) => {
                    self.reconnects += 1;
                    return Ok(());
                }
                Err(e) if is_transient(&e) => last = e,
                Err(e) => return Err(e),
            }
        }
        Err(CdcError::Io(format!(
            "reconnect gave up after {} attempts: {last}",
            self.max_attempts
        )))
    }
}

impl ChangeSource for RetryingSource {
    fn next_tx(&mut self, idle: Duration) -> Result<Option<CommittedTx>, CdcError> {
        loop {
            match self.src.next_tx(idle) {
                Ok(v) => return Ok(v),
                Err(e) if is_transient(&e) => self.reconnect_bounded()?,
                Err(e) => return Err(e),
            }
        }
    }

    fn ack(&mut self, lsn: u64) -> Result<(), CdcError> {
        match self.src.ack(lsn) {
            Err(e) if is_transient(&e) => {
                // A lost ack is safe: the slot just stays behind and
                // the redelivered prefix is filtered by the mirror's
                // exactly-once watermark. Re-ack on the new stream.
                self.reconnect_bounded()?;
                self.src.ack(lsn)
            }
            r => r,
        }
    }
}

impl ChangeSource for PgOutputSource {
    fn next_tx(&mut self, idle: Duration) -> Result<Option<CommittedTx>, CdcError> {
        loop {
            let Some(payload) = self.conn.read_copy(idle)? else {
                return Ok(None);
            };
            match decode_wal(&payload)? {
                WalMsg::Keepalive {
                    wal_end,
                    reply_requested,
                } => {
                    self.last_recv = self.last_recv.max(wal_end);
                    if reply_requested {
                        let (r, a) = (self.last_recv, self.last_ack);
                        self.conn.send_status(r, a, a, false)?;
                    }
                }
                WalMsg::XLogData { wal_end, msg, .. } => {
                    self.last_recv = self.last_recv.max(wal_end);
                    if let Some(tx) = self.asm.on_msg(msg)? {
                        return Ok(Some(tx));
                    }
                }
            }
        }
    }

    fn ack(&mut self, lsn: u64) -> Result<(), CdcError> {
        self.last_ack = self.last_ack.max(lsn);
        let recv = self.last_recv.max(self.last_ack);
        self.conn
            .send_status(recv, self.last_ack, self.last_ack, false)
    }
}

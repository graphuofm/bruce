//! bruce-cdc — the maintenance plane.
//!
//! Three planes: bruce-core is the compute plane (kernels),
//! bruce-query is the query plane (SQL -> plan -> answer, maintained
//! views), and this crate is the maintenance plane: it subscribes to
//! PostgreSQL logical replication (pgoutput over a real
//! START_REPLICATION stream) and mirrors committed row changes into a
//! [`bruce_query::Database`] through the same write path every other
//! caller uses — `insert_row` / `delete_where` — so maintained
//! soft-aggregate views stay fresh beside PG without re-scanning.
//!
//! Transport is behind [`source::ChangeSource`], so the pgoutput
//! stream is swappable (e.g. for a polled SQL-interface source)
//! without touching the apply path.

pub mod apply;
pub mod durable;
pub mod pgoutput;
pub mod protocol;
pub mod source;

use thiserror::Error;

/// Errors across the maintenance plane.
#[derive(Debug, Error)]
pub enum CdcError {
    /// Socket-level failure.
    #[error("io: {0}")]
    Io(String),
    /// The byte stream violated the frontend/backend protocol.
    #[error("protocol: {0}")]
    Protocol(String),
    /// The server reported an error (`code` is the SQLSTATE).
    #[error("backend {code}: {message}")]
    Backend {
        /// SQLSTATE.
        code: String,
        /// Human-readable message.
        message: String,
    },
    /// A pgoutput or tuple payload did not decode.
    #[error("decode: {0}")]
    Decode(String),
    /// The change did not apply to the mirror database.
    #[error("apply: {0}")]
    Apply(String),
}

impl From<std::io::Error> for CdcError {
    fn from(e: std::io::Error) -> Self {
        CdcError::Io(e.to_string())
    }
}

impl From<bruce_query::QueryError> for CdcError {
    fn from(e: bruce_query::QueryError) -> Self {
        CdcError::Apply(e.to_string())
    }
}

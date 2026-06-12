//! Error types for Bruce-core.

use thiserror::Error;

/// Top-level error type for the Bruce operator and its CRUD layer.
#[derive(Debug, Error)]
pub enum BruceError {
    /// A key was inserted twice without an intervening delete or update.
    #[error("key {0} already present; use update() to replace it")]
    DuplicateKey(String),

    /// A delete was issued for a key that is not in the live set.
    #[error("key {0} not found")]
    KeyNotFound(String),

    /// A read or update was issued by an owner who is not the key's owner.
    #[error("permission denied: key {0} is owned by {1:?}, not {2:?}")]
    PermissionDenied(String, String, String),

    /// A query vector's dimension does not match the memory's `d_k`.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the operation required.
        expected: usize,
        /// The dimension actually supplied.
        got: usize,
    },

    /// An invalid epsilon (e.g., negative) was supplied.
    #[error("invalid temperature ε = {0}: must be non-negative")]
    InvalidEpsilon(f64),

    /// A function argument failed a validity check that isn't dimension- or
    /// owner-related (e.g., a malformed parent index in a tree topology).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Numerical overflow / underflow / NaN appeared in an accumulator.
    /// This signals a bug or an extreme operating point; the caller can
    /// retry with a higher-precision dtype.
    #[error("numerical instability in accumulator: {0}")]
    NumericalInstability(String),

    /// I/O error from the audit log, parquet sink, etc.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic catch-all wrapped via `anyhow`.
    #[error("other: {0}")]
    Other(#[from] anyhow::Error),
}

/// Bruce-core `Result` shorthand.
pub type Result<T> = std::result::Result<T, BruceError>;

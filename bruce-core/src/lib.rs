//! # Bruce — the F_ε operator
//!
//! Bruce is a unified algebra of relational-database query evaluation and
//! Transformer attention. The central object is the **F_ε operator**, a
//! single semiring-parameterised aggregator whose temperature `ε ∈ [0, ∞)`
//! interpolates between
//!
//! * `ε = 0`   ↦  the tropical (max-plus) semiring →
//!   exact equi-join with aggregation (SQL `SELECT ... GROUP BY`);
//! * `ε = 1`   ↦  the log-semiring →
//!   standard softmax attention `A = softmax(QKᵀ) V`;
//! * `ε → ∞`   ↦  the uniform-average limit.
//!
//! The interpolation is **Maslov dequantization**: as ε → 0⁺,
//! `ε · log Σᵢ exp(xᵢ/ε) → maxᵢ xᵢ`. This is the bridge a half-century
//! of database and machine-learning theory have been waiting for.
//!
//! ## Why a Rust crate?
//!
//! The database side of Bruce is meant to be a **first-class systems
//! component**, not a Python prototype. We use Rust for memory safety,
//! C-equivalent throughput, and clean FFI to Python (via `bruce-py`).
//! The attention side stays in Python where the ML ecosystem lives.
//!
//! ## Public modules
//!
//! * [`operator`] — the F_ε operator and its specialisations
//! * [`semiring`] — log-semiring + tropical semiring + their dequantisation
//! * [`memory`]   — the K/V memory backing CRUD operations
//! * [`crud`]     — INSERT / UPDATE / DELETE via the group structure of Σ
//!   (Lemma A, the O(d)-per-record exact unlearning algorithm)
//! * [`join`]     — hash-, sort-merge-, Leapfrog-Triejoin-style equi-join
//! * [`distributed`] — partition-reduce (Lemma B): F_ε is exact under
//!   partitioning of the K/V memory across nodes
//! * [`types`]    — public types: `Score`, `Eps`, `Sim`, `Aggregator`
//!
//! ## Quickstart
//!
//! ```rust
//! use bruce_core::{F_eps, Eps, Sim};
//! use ndarray::array;
//!
//! // Three records (K, V) with d_k = d_v = 2
//! let k = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let v = array![[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]];
//! let x = array![1.0, 0.0];
//!
//! let op = F_eps::new(Eps::ONE, Sim::Dot);
//! let out = op.attention(&x.view(), &k.view(), &v.view());
//! // out is the standard softmax-attention output
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// WHEEL-BLAS-001: when the `blas` feature is on, pull openblas into
// the final link graph.  Without this the cdylib silently drops
// libopenblas.a and ndarray's `cargo:rustc-link-lib=cblas` directives
// fail at load time with `undefined symbol: cblas_dgemm`.
#[cfg(feature = "blas")]
extern crate blas_src;
#[cfg(feature = "blas")]
extern crate openblas_src;

pub mod anonymity;
pub mod cascade;
pub mod crud;
pub mod distributed;
pub mod dp;
pub mod encrypted;
pub mod error;
pub mod hnsw;
pub mod join;
pub mod mask;
pub mod memory;
pub mod merkle;
pub mod operator;
pub mod provenance;
pub mod semiring;
pub mod sketch;
pub mod streaming;
pub mod tree;
pub mod types;

pub use crud::{IncrementalMemory, IncrementalState};
pub use error::{BruceError, Result};
pub use mask::masked_attention;
pub use memory::KvMemory;
pub use operator::F_eps;
pub use types::{Aggregator, Eps, Score, Sim};

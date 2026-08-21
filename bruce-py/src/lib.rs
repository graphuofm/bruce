#![allow(clippy::useless_conversion)]
// ^ pyo3 0.22's #[pymethods] expansion inserts `.map_err(Into::into)`
//   on PyResult-returning methods, which clippy >= 1.86 flags as a
//   useless conversion. The lint fires inside macro-generated code we
//   do not control; remove this allow when bumping pyo3.

//! Python bindings for Bruce.
//!
//! Exposes the F_ε operator, the K/V memory, and the incremental
//! Lemma A maintenance to Python via PyO3 + NumPy.
//!
//! ```python
//! import numpy as np
//! import bruce
//! mem = bruce.IncrementalMemory(query=np.array([1., 0.]), eps=1.0, d_v=2)
//! mem.insert("a", np.array([1., 0.]), np.array([10., 0.]))
//! mem.insert("b", np.array([0., 1.]), np.array([0., 10.]))
//! print(mem.output())   # [softmax-weighted output]
//! mem.delete("a")
//! ```

// WHEEL-BLAS-001: pin openblas into the cdylib's link graph at the
// crate that becomes _bruce.abi3.so. Without these `extern crate`s
// the openblas archive is dead-stripped and Python's `import bruce`
// dies with `undefined symbol: cblas_dgemm`.
#[cfg(feature = "blas")]
extern crate blas_src;
#[cfg(feature = "blas")]
extern crate openblas_src;

use bruce_core::anonymity::{AnonymityGuard as RsAnon, GuardOutcome as RsOutcome};
use bruce_core::cascade::CascadePlan as RsCascade;
use bruce_core::distributed::{
    combine as rs_combine, finalize as rs_finalize, PartialTriple as RsPartial,
};
use bruce_core::dp::{DpBudget, GaussianMechanism as RsGauss, LaplaceMechanism as RsLap};
use bruce_core::encrypted::{key_from_passphrase, EncryptedBlob as RsBlob};
use bruce_core::join::{
    hash_join as rs_hash_join, lftj_three as rs_lftj_three, sort_merge_join as rs_sort_merge,
};
use bruce_core::mask::{
    causal_pairs as rs_causal_pairs, grouped_softavg as rs_grouped_softavg,
    grouped_softavg_f32 as rs_grouped_softavg_f32, masked_attention as rs_masked_attention,
    window_pairs as rs_window_pairs,
};
use bruce_core::memory::{KvMemory as RsKv, KvSnapshot as RsKvSnapshot};
use bruce_core::merkle::MerkleAuditLog as RsMerkle;
use bruce_core::provenance::{Identity as RsIdentity, SignedFact as RsSigned};
use bruce_core::semiring::{
    dequantization_bound as rs_dequantization_bound, eps_star as rs_eps_star,
};
use bruce_core::sketch::{FeatureMap, FuzzyJoinSketch as RsSketch};
use bruce_core::streaming::StreamingChainJoin as RsStream;
use bruce_core::tree::{
    balanced_binary_tree as rs_balanced_binary_tree, chain_tree as rs_chain_tree,
    k_ary_balanced_tree as rs_k_ary_balanced_tree, star_tree as rs_star_tree,
    tree_causal_attention as rs_tree_attention,
};
use bruce_core::{Eps, F_eps, IncrementalMemory as RsMem, Sim};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn parse_sim(name: &str) -> PyResult<Sim> {
    match name {
        "dot" => Ok(Sim::Dot),
        "negsq" | "neg_squared" => Ok(Sim::NegSquared),
        "indicator" => Ok(Sim::Indicator),
        _ => Err(PyValueError::new_err(format!(
            "unknown sim {name:?}; expected one of dot, negsq, indicator"
        ))),
    }
}

/// `(keys, values)` pair returned by exact-read style accessors.
type OptionalPair<'py> = Option<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>;

fn parse_eps(value: f64) -> PyResult<Eps> {
    Eps::new(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Python wrapper around `bruce_core::F_eps`.
#[pyclass(name = "Operator")]
pub struct PyOperator {
    inner: F_eps,
}

#[pymethods]
impl PyOperator {
    #[new]
    #[pyo3(signature = (eps=1.0, sim="dot"))]
    fn new(eps: f64, sim: &str) -> PyResult<Self> {
        Ok(Self {
            inner: F_eps::new(parse_eps(eps)?, parse_sim(sim)?),
        })
    }

    /// Compute attention `A_ε(x, K, V)`.
    fn attention<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        k: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let out: Array1<f64> = self
            .inner
            .attention(&x.as_array(), &k.as_array(), &v.as_array());
        out.into_pyarray_bound(py)
    }

    /// Compute SQL-style sum `Q_ε(x, K, V)`.
    fn sum<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        k: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let out: Array1<f64> = self.inner.sum(&x.as_array(), &k.as_array(), &v.as_array());
        out.into_pyarray_bound(py)
    }

    /// Batched attention: B queries against the same (K, V) in one
    /// call. Returns shape (B, d_v). Two large matmuls Q @ K^T and
    /// softmax(.) @ V dominate; per-query latency drops sharply as B
    /// grows compared to B individual `attention` calls.
    fn attention_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
        k: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let out = self
            .inner
            .attention_batch(&q.as_array(), &k.as_array(), &v.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out.into_pyarray_bound(py))
    }
}

/// Python wrapper around `bruce_core::IncrementalMemory`.
#[pyclass(name = "IncrementalMemory")]
pub struct PyIncrementalMemory {
    inner: RsMem,
}

#[pymethods]
impl PyIncrementalMemory {
    #[new]
    #[pyo3(signature = (query, eps=1.0, d_v=1, sim="dot"))]
    fn new(query: PyReadonlyArray1<'_, f64>, eps: f64, d_v: usize, sim: &str) -> PyResult<Self> {
        let q = query.as_array();
        Ok(Self {
            inner: RsMem::new(q, parse_eps(eps)?, d_v, parse_sim(sim)?),
        })
    }

    /// Insert a key/value pair, identified by `key_id`.
    fn insert(
        &mut self,
        key_id: &str,
        k: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        self.inner
            .insert(key_id, k.as_array(), v.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Delete a previously-inserted pair by id.
    fn delete(&mut self, key_id: &str) -> PyResult<()> {
        self.inner
            .delete(key_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch insert: `K` is (N, d_k), `V` is (N, d_v), one `key_ids[i]`
    /// per row. Amortises the Python/Rust FFI boundary across N rows —
    /// 10-40× faster than calling `insert()` in a tight Python loop
    /// when N is large.
    fn insert_many(
        &mut self,
        key_ids: Vec<String>,
        k: PyReadonlyArray2<'_, f64>,
        v: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let kk = k.as_array();
        let vv = v.as_array();
        if kk.nrows() != key_ids.len() || vv.nrows() != key_ids.len() {
            return Err(PyValueError::new_err(format!(
                "row count mismatch: ids={}, k={}, v={}",
                key_ids.len(),
                kk.nrows(),
                vv.nrows()
            )));
        }
        for (i, id) in key_ids.iter().enumerate() {
            self.inner
                .insert(id, kk.row(i), vv.row(i))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Batch delete: like calling `delete()` for each id, but with one
    /// FFI crossing. Stops at the first missing key (does not partially
    /// roll back; missing keys are an error).
    fn delete_many(&mut self, key_ids: Vec<String>) -> PyResult<()> {
        for id in &key_ids {
            self.inner
                .delete(id)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Current attention output `A_ε(x)`.
    fn output<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.output().into_pyarray_bound(py)
    }

    /// How many live entries are in the memory.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// How many full rescales have been triggered.
    #[getter]
    fn n_rescales(&self) -> u64 {
        self.inner.n_rescales()
    }
}

fn parse_phi(name: &str) -> PyResult<FeatureMap> {
    match name {
        "elu+1" | "elu_plus_1" | "elu" => Ok(FeatureMap::EluPlus1),
        "identity" | "id" => Ok(FeatureMap::Identity),
        _ => Err(PyValueError::new_err(format!(
            "unknown feature map {name:?}; expected 'elu+1' or 'identity'"
        ))),
    }
}

/// Sketch-based fuzzy join — the **linear-attention kernel trick**
/// turned into a SQL-side index whose state size is `O(d_phi * d_v)`
/// and is independent of the number of rows N.
///
/// Build the sketch once over the (K, V) memory; subsequent queries
/// answer in `O(d_phi * d_v)` regardless of N.
///
/// Example:
///     >>> import bruce, numpy as np
///     >>> K = np.random.randn(1_000_000, 64).astype(np.float64)
///     >>> V = np.random.randn(1_000_000, 16).astype(np.float64)
///     >>> s = bruce.FuzzyJoinSketch(K, V, phi="elu+1")
///     >>> q = np.random.randn(64).astype(np.float64)
///     >>> out = s.query(q)         # shape (16,), O(d_phi * d_v)
///     >>> s.size_bytes             # constant — same at N=10K or N=10^9
#[pyclass(name = "FuzzyJoinSketch")]
pub struct PyFuzzyJoinSketch {
    inner: RsSketch,
}

#[pymethods]
impl PyFuzzyJoinSketch {
    /// Build a fuzzy-join sketch from a K/V table. O(N * d_phi * d_v).
    ///
    /// Args:
    ///   k: keys, shape (N, d_phi)
    ///   v: values, shape (N, d_v)
    ///   phi: feature map ("elu+1" or "identity")
    #[new]
    #[pyo3(signature = (k, v, phi="elu+1"))]
    fn new(
        k: PyReadonlyArray2<'_, f64>,
        v: PyReadonlyArray2<'_, f64>,
        phi: &str,
    ) -> PyResult<Self> {
        let phi = parse_phi(phi)?;
        let inner = RsSketch::build(k.as_array(), v.as_array(), phi);
        Ok(Self { inner })
    }

    /// Add one row to the sketch incrementally. O(d_phi * d_v).
    fn add(&mut self, k: PyReadonlyArray1<'_, f64>, v: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let k_arr = k.as_array();
        let v_arr = v.as_array();
        let d_phi = self.inner.numerator.nrows();
        if k_arr.len() != d_phi {
            return Err(PyValueError::new_err(format!(
                "key dim mismatch: expected {}, got {}",
                d_phi,
                k_arr.len()
            )));
        }
        self.inner.add(k_arr, v_arr);
        Ok(())
    }

    /// Answer a fuzzy-join query in O(d_phi * d_v), independent of N.
    fn query<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = x.as_array();
        let d_phi = self.inner.numerator.nrows();
        if x_arr.len() != d_phi {
            return Err(PyValueError::new_err(format!(
                "query dim mismatch: expected {}, got {}",
                d_phi,
                x_arr.len()
            )));
        }
        Ok(self.inner.query(x_arr).into_pyarray_bound(py))
    }

    /// Number of (K, V) rows aggregated into the sketch.
    #[getter]
    fn n_rows(&self) -> u64 {
        self.inner.n_rows
    }

    /// Total bytes of state (constant in N).
    #[getter]
    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    /// Expose the raw numerator (d_phi, d_v) for advanced users.
    fn numerator<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.numerator.clone().into_pyarray_bound(py)
    }

    /// Expose the raw denominator (d_phi,).
    fn denominator<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.denominator.clone().into_pyarray_bound(py)
    }
}

/// Append-only Merkle audit log for tamper-evident bookkeeping.
///
/// Each `append(bytes)` returns the entry's index. The current
/// Merkle root summarises every entry — any later modification of
/// a past entry changes the root, so publishing the root acts as a
/// commitment to the log's contents up to that point.
///
/// Example:
///     >>> import bruce
///     >>> log = bruce.MerkleAuditLog()
///     >>> i = log.append(b"INSERT customer 42")
///     >>> root = log.root()
///     >>> proof = log.proof(i)
///     >>> assert bruce.MerkleAuditLog.verify(b"INSERT customer 42",
///     ...                                    i, log.len, proof, root)
#[pyclass(name = "MerkleAuditLog")]
pub struct PyMerkleAuditLog {
    inner: RsMerkle,
}

#[pymethods]
impl PyMerkleAuditLog {
    /// Create an empty append-only log.
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMerkle::new(),
        }
    }

    /// Append one entry (arbitrary bytes). Returns the entry's index.
    fn append(&mut self, payload: &[u8]) -> usize {
        self.inner.append(payload)
    }

    /// Number of entries in the log.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Current Merkle root (32 bytes).
    fn root<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.root())
    }

    /// Inclusion proof for entry `idx`. List of sibling hashes from
    /// the leaf to the root. Returns None if `idx` is out of range.
    fn proof<'py>(&self, py: Python<'py>, idx: usize) -> Option<Vec<Bound<'py, PyBytes>>> {
        self.inner.proof(idx).map(|hashes| {
            hashes
                .into_iter()
                .map(|h| PyBytes::new_bound(py, &h))
                .collect()
        })
    }

    /// Verify that `payload` at index `idx` is included in `root`,
    /// given the `proof` (list of bytes, each 32) and `n_leaves`
    /// (the log length at the time the root was published).
    #[staticmethod]
    fn verify(
        payload: &[u8],
        idx: usize,
        n_leaves: usize,
        proof: Vec<Vec<u8>>,
        root: Vec<u8>,
    ) -> PyResult<bool> {
        if root.len() != 32 {
            return Err(PyValueError::new_err("root must be 32 bytes"));
        }
        let mut root_arr = [0u8; 32];
        root_arr.copy_from_slice(&root);
        let mut sibs: Vec<[u8; 32]> = Vec::with_capacity(proof.len());
        for s in &proof {
            if s.len() != 32 {
                return Err(PyValueError::new_err("each proof element must be 32 bytes"));
            }
            let mut h = [0u8; 32];
            h.copy_from_slice(s);
            sibs.push(h);
        }
        Ok(RsMerkle::verify(payload, idx, n_leaves, &sibs, &root_arr))
    }
}

/// An Ed25519 signing identity (keypair). Generated by
/// `bruce.Identity.generate()` or restored from a 32-byte secret via
/// `bruce.Identity.from_secret(...)`.
///
/// Sign a fact with `identity.sign_fact(fact_id, owner, payload)` and
/// verify with the returned `SignedFact.verify()`.
#[pyclass(name = "Identity")]
pub struct PyIdentity {
    inner: RsIdentity,
}

#[pymethods]
impl PyIdentity {
    /// Generate a fresh identity from the OS CSPRNG.
    #[staticmethod]
    fn generate() -> Self {
        Self {
            inner: RsIdentity::generate(),
        }
    }

    /// Reconstruct from 32-byte secret bytes (e.g. read from a KMS).
    #[staticmethod]
    fn from_secret(secret: Vec<u8>) -> PyResult<Self> {
        if secret.len() != 32 {
            return Err(PyValueError::new_err("secret must be 32 bytes"));
        }
        let mut s = [0u8; 32];
        s.copy_from_slice(&secret);
        Ok(Self {
            inner: RsIdentity::from_secret(s),
        })
    }

    /// Export the 32-byte secret. **Handle with care.**
    fn secret_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.secret_bytes())
    }

    /// Export the 32-byte public key (safe to share).
    fn public_key<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.public_key().to_bytes())
    }

    /// Sign a fact. Returns a `SignedFact` whose `.verify()` can
    /// be called by anyone holding this identity's public key.
    fn sign_fact(&self, fact_id: &str, owner: &str, payload: &[u8]) -> PySignedFact {
        let signed = self.inner.sign_fact(fact_id, owner, payload);
        PySignedFact { inner: signed }
    }
}

/// A fact + Ed25519 signature + public key. Self-contained: anyone
/// can `.verify()` it.
#[pyclass(name = "SignedFact")]
pub struct PySignedFact {
    inner: RsSigned,
}

#[pymethods]
impl PySignedFact {
    /// Verify the signature. Returns True on success, False otherwise.
    fn verify(&self) -> bool {
        self.inner.verify().is_ok()
    }

    /// Verify the signature. Raises ValueError if invalid (use this
    /// when the application wants the error rather than a bool).
    fn verify_or_raise(&self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn fact_id(&self) -> String {
        self.inner.fact_id.clone()
    }

    #[getter]
    fn owner(&self) -> String {
        self.inner.owner.clone()
    }

    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.payload)
    }

    #[getter]
    fn signature<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.signature)
    }

    #[getter]
    fn public_key<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.public_key)
    }

    #[getter]
    fn key_fingerprint(&self) -> String {
        self.inner.key_fingerprint()
    }
}

/// Laplace mechanism for ε-differential privacy on real-valued
/// queries. Add `Lap(0, Δ/ε)` noise to a query of L1-sensitivity `Δ`.
///
/// Example:
///     >>> import bruce
///     >>> mech = bruce.LaplaceMechanism(l1_sensitivity=1.0, epsilon=0.5)
///     >>> noisy = mech.release_scalar(42.0)   # 42 + Lap(0, 2)
#[pyclass(name = "LaplaceMechanism")]
pub struct PyLaplaceMechanism {
    inner: RsLap,
}

#[pymethods]
impl PyLaplaceMechanism {
    #[new]
    #[pyo3(signature = (l1_sensitivity, epsilon, seed=None))]
    fn new(l1_sensitivity: f64, epsilon: f64, seed: Option<u64>) -> PyResult<Self> {
        if !l1_sensitivity.is_finite() || l1_sensitivity <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "l1_sensitivity must be a positive finite number, got {l1_sensitivity}",
            )));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "epsilon must be a positive finite number, got {epsilon}",
            )));
        }
        let mut m = RsLap::new(l1_sensitivity, DpBudget::pure(epsilon));
        if let Some(s) = seed {
            m = m.with_seed(s);
        }
        Ok(Self { inner: m })
    }

    /// Release a single scalar with ε-DP noise added.
    fn release_scalar(&self, true_value: f64) -> f64 {
        self.inner.release_scalar(true_value)
    }

    /// Release a vector with independent Laplace noise per element.
    fn release_vector(&self, true_values: Vec<f64>) -> Vec<f64> {
        self.inner.release_vector(&true_values)
    }

    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.budget.epsilon
    }
}

/// Gaussian mechanism for (ε, δ)-differential privacy on L2-sensitivity
/// queries. Adds `N(0, σ²)` noise with `σ = Δ/ε · √(2 ln(1.25/δ))`.
///
/// Example:
///     >>> mech = bruce.GaussianMechanism(l2_sensitivity=1.0,
///     ...                                 epsilon=1.0, delta=1e-5)
///     >>> noisy = mech.release_scalar(3.14)
#[pyclass(name = "GaussianMechanism")]
pub struct PyGaussianMechanism {
    inner: RsGauss,
}

#[pymethods]
impl PyGaussianMechanism {
    #[new]
    #[pyo3(signature = (l2_sensitivity, epsilon, delta=1e-5, seed=None))]
    fn new(l2_sensitivity: f64, epsilon: f64, delta: f64, seed: Option<u64>) -> PyResult<Self> {
        if !l2_sensitivity.is_finite() || l2_sensitivity <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "l2_sensitivity must be a positive finite number, got {l2_sensitivity}",
            )));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "epsilon must be a positive finite number, got {epsilon}",
            )));
        }
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return Err(PyValueError::new_err(format!(
                "delta must lie in (0, 1) for the Gaussian mechanism, got {delta}",
            )));
        }
        let budget = DpBudget { epsilon, delta };
        let mut m = RsGauss::new(l2_sensitivity, budget);
        if let Some(s) = seed {
            m = m.with_seed(s);
        }
        Ok(Self { inner: m })
    }

    /// Release a single scalar.
    fn release_scalar(&self, true_value: f64) -> f64 {
        self.inner.release_scalar(true_value)
    }

    /// Release a vector with independent Gaussian noise per element.
    fn release_vector(&self, true_values: Vec<f64>) -> Vec<f64> {
        self.inner.release_vector(&true_values)
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }
}

/// k-anonymity / l-diversity query guard. Rejects a query whose
/// survivor set is too small or too homogeneous to release safely.
///
/// Example:
///     >>> g = bruce.AnonymityGuard(k=5)
///     >>> result = g.evaluate(survivors=[1,2,3,4,5])     # ok
///     >>> result = g.evaluate(survivors=[1])              # too small
///     >>> g_kl = bruce.AnonymityGuard(k=3, l=2)
#[pyclass(name = "AnonymityGuard")]
pub struct PyAnonymityGuard {
    inner: RsAnon,
}

#[pymethods]
impl PyAnonymityGuard {
    #[new]
    #[pyo3(signature = (k, l=None))]
    fn new(k: usize, l: Option<usize>) -> Self {
        let inner = match l {
            Some(ll) => RsAnon::k_and_l(k, ll),
            None => RsAnon::k_anonymity(k),
        };
        Self { inner }
    }

    /// Evaluate against a survivor set (Python list of hashable items).
    /// Returns a dict describing the outcome.
    fn evaluate(&self, py: Python<'_>, survivors: Vec<String>) -> PyObject {
        let outcome = self.inner.evaluate(&survivors);
        let d = pyo3::types::PyDict::new_bound(py);
        match outcome {
            RsOutcome::Allow => {
                d.set_item("status", "allow").unwrap();
            }
            RsOutcome::DenyTooFewRecords { n, k } => {
                d.set_item("status", "deny_too_few").unwrap();
                d.set_item("n", n).unwrap();
                d.set_item("k", k).unwrap();
            }
            RsOutcome::DenyTooLowDiversity { distinct, l } => {
                d.set_item("status", "deny_low_diversity").unwrap();
                d.set_item("distinct", distinct).unwrap();
                d.set_item("l", l).unwrap();
            }
        }
        d.into()
    }
}

/// AES-256-GCM encrypted blob (nonce + ciphertext + tag, in one
/// envelope). For encrypted-at-rest persistence of Bruce facts.
///
/// Example:
///     >>> import bruce
///     >>> key = bruce.EncryptedBlob.key_from_passphrase("dev-key")
///     >>> blob = bruce.EncryptedBlob.encrypt(key, b"the customer owes $1,234")
///     >>> wire = blob.to_bytes()                  # store / transmit
///     >>> blob2 = bruce.EncryptedBlob.from_bytes(wire)
///     >>> assert blob2.decrypt(key) == b"the customer owes $1,234"
#[pyclass(name = "EncryptedBlob")]
pub struct PyEncryptedBlob {
    inner: RsBlob,
}

#[pymethods]
impl PyEncryptedBlob {
    /// Derive a 32-byte key from a passphrase (SHA-256). Convenience
    /// only — for production use a real KDF.
    #[staticmethod]
    fn key_from_passphrase<'py>(py: Python<'py>, pass: &str) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &key_from_passphrase(pass))
    }

    /// Encrypt `plaintext` under `key`. Returns an EncryptedBlob.
    #[staticmethod]
    fn encrypt(key: Vec<u8>, plaintext: &[u8]) -> PyResult<Self> {
        if key.len() != 32 {
            return Err(PyValueError::new_err("key must be 32 bytes"));
        }
        let mut k = [0u8; 32];
        k.copy_from_slice(&key);
        let inner =
            RsBlob::encrypt(&k, plaintext).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Decrypt. Raises ValueError on tag mismatch.
    fn decrypt<'py>(&self, py: Python<'py>, key: Vec<u8>) -> PyResult<Bound<'py, PyBytes>> {
        if key.len() != 32 {
            return Err(PyValueError::new_err("key must be 32 bytes"));
        }
        let mut k = [0u8; 32];
        k.copy_from_slice(&key);
        let pt = self
            .inner
            .decrypt(&k)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &pt))
    }

    /// Wire-format bytes: [nonce(12) || ciphertext || tag(16)].
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.to_bytes())
    }

    /// Parse from wire format.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let inner = RsBlob::from_bytes(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn nonce<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.nonce)
    }
}

/// Durable K/V memory with audit log + owner-enforced delete.
///
/// This is the "rich" version of IncrementalMemory: it stores
/// (key, value, owner) triples and supports exact lookup by id
/// (ε=0 read) plus a snapshot of all alive rows for downstream
/// attention.
///
/// Example:
///     >>> import bruce, numpy as np
///     >>> m = bruce.KvMemory(d_k=2, d_v=1)
///     >>> m.write("t1", np.array([1., 0.]), np.array([42.]), owner="alice")
///     >>> k, v = m.read_exact("t1")
///     >>> m.delete("t1", owner="alice")
#[pyclass(name = "KvMemory")]
pub struct PyKvMemory {
    inner: RsKv,
}

#[pymethods]
impl PyKvMemory {
    #[new]
    fn new(d_k: usize, d_v: usize) -> Self {
        Self {
            inner: RsKv::new(d_k, d_v),
        }
    }

    fn write(
        &mut self,
        fact_id: &str,
        k: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
        owner: &str,
    ) -> PyResult<()> {
        self.inner
            .write(fact_id, k.as_array(), v.as_array(), owner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn delete(&mut self, fact_id: &str, owner: &str) -> PyResult<()> {
        self.inner
            .delete(fact_id, owner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Look up a fact by id (ε=0 read). Returns `(k, v)` or None if
    /// not present or deleted.
    fn read_exact<'py>(&self, py: Python<'py>, fact_id: &str) -> OptionalPair<'py> {
        self.inner.read_exact(fact_id).map(|(k, v)| {
            (
                k.clone().into_pyarray_bound(py),
                v.clone().into_pyarray_bound(py),
            )
        })
    }

    /// Number of alive (non-deleted) rows.
    #[getter]
    fn len_alive(&self) -> usize {
        self.inner.len_alive()
    }

    /// Number of rows in total (incl. deleted).
    #[getter]
    fn len_total(&self) -> usize {
        self.inner.len_total()
    }

    /// Length of the audit log.
    #[getter]
    fn audit_log_len(&self) -> usize {
        self.inner.audit_log().len()
    }

    /// Persist the memory to a Parquet file. Audit log is written to
    /// `<path>.audit.jsonl` alongside.
    fn save_parquet(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_parquet(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a memory snapshot from a Parquet file. Returns a new
    /// `KvMemory`.  Audit log is restored from `<path>.audit.jsonl`
    /// if present.
    #[staticmethod]
    fn load_parquet(path: &str) -> PyResult<Self> {
        RsKv::load_parquet(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Bulk-insert `len(ids)` rows from 2-D arrays `keys` (n, d_k)
    /// and `values` (n, d_v), all owned by `owner`.
    ///
    /// Semantically identical to `n` sequential `write` calls (same
    /// audit-log entries, same owner enforcement, last-write-wins for
    /// ids duplicated within the batch) except the whole batch shares
    /// one wall-clock timestamp. All-or-nothing: shape or ownership
    /// errors leave the memory untouched.
    ///
    /// Copies, honestly: the numpy buffers are read zero-copy when
    /// C-contiguous float64 (one engine-side copy per row into the
    /// store, same as `write`); non-contiguous input costs one extra
    /// flattening copy here.
    fn bulk_insert(
        &mut self,
        ids: Vec<String>,
        keys: PyReadonlyArray2<'_, f64>,
        values: PyReadonlyArray2<'_, f64>,
        owner: &str,
    ) -> PyResult<usize> {
        let n = ids.len();
        let (kr, kc) = keys.as_array().dim();
        let (vr, vc) = values.as_array().dim();
        // Shape check HERE, not just total length in the core: a
        // mis-shaped (n', d') buffer with n' * d' == n * d_k would
        // otherwise be silently re-chunked into wrong rows.
        if kr != n || kc != self.inner.d_k() {
            return Err(PyValueError::new_err(format!(
                "keys shape ({kr}, {kc}) != (len(ids)={n}, d_k={})",
                self.inner.d_k()
            )));
        }
        if vr != n || vc != self.inner.d_v() {
            return Err(PyValueError::new_err(format!(
                "values shape ({vr}, {vc}) != (len(ids)={n}, d_v={})",
                self.inner.d_v()
            )));
        }
        // DEFECT (exposed by m17 test_bulk_insert_fortran_order_input):
        // PyReadonlyArray::as_slice() hands back the RAW buffer of any
        // contiguous array, including F-contiguous ones, which silently
        // scrambles rows into column-major order. Guard on ndarray's
        // standard (row-major) layout instead; anything else takes the
        // logical-order flattening copy.
        let ka = keys.as_array();
        let va = values.as_array();
        match (ka.as_slice(), va.as_slice()) {
            (Some(k), Some(v)) => self.inner.bulk_insert(&ids, k, v, owner),
            _ => {
                // non-C-contiguous input: flatten in logical row-major order
                let k: Vec<f64> = ka.iter().copied().collect();
                let v: Vec<f64> = va.iter().copied().collect();
                self.inner.bulk_insert(&ids, &k, &v, owner)
            }
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Export every live row into one contiguous `KvSnapshot`
    /// (tombstoned rows skipped, insertion order preserved — the same
    /// order every decode read folds in, so a decode over the snapshot
    /// buffers reproduces the in-memory decode bit-for-bit).
    fn snapshot(&self) -> PyKvSnapshot {
        PyKvSnapshot {
            inner: self.inner.snapshot(),
        }
    }

    /// Rebuild a `KvMemory` from a snapshot. Bitwise round-trip:
    /// `KvMemory.restore(m.snapshot())` produces identical decode
    /// results. Owners and write timestamps are preserved (so
    /// owner-enforced delete keeps working); the restored memory
    /// starts with an empty audit log — audit history travels with
    /// `save_parquet`/`load_parquet`, not with hot-path snapshots.
    #[staticmethod]
    fn restore(snap: PyRef<'_, PyKvSnapshot>) -> PyResult<Self> {
        RsKv::restore(&snap.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Contiguous, row-major snapshot of the live rows of a `KvMemory`.
///
/// Produced by `KvMemory.snapshot()`, consumed by `KvMemory.restore`.
/// `keys` / `values` come back as (n_rows, d) float64 numpy arrays —
/// one copy out of the engine buffer per accessor call (cache them if
/// read repeatedly). The Arrow/Parquet wrapping of these buffers
/// belongs to this layer or above, never to bruce-core (C1/C3).
#[pyclass(name = "KvSnapshot")]
pub struct PyKvSnapshot {
    inner: RsKvSnapshot,
}

#[pymethods]
impl PyKvSnapshot {
    /// Number of live rows in the snapshot.
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// Key dimensionality.
    #[getter]
    fn d_k(&self) -> usize {
        self.inner.d_k
    }

    /// Value dimensionality.
    #[getter]
    fn d_v(&self) -> usize {
        self.inner.d_v
    }

    /// Live fact ids, in insertion order (copy).
    #[getter]
    fn ids(&self) -> Vec<String> {
        self.inner.ids.clone()
    }

    /// Row owners, parallel to `ids` (copy).
    #[getter]
    fn owners(&self) -> Vec<String> {
        self.inner.owners.clone()
    }

    /// Row write timestamps (unix seconds), parallel to `ids` (copy).
    #[getter]
    fn written_at<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.written_at.clone().into_pyarray_bound(py)
    }

    /// Keys as an (n_rows, d_k) float64 array (one copy).
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Array2::from_shape_vec(
            (self.inner.n_rows(), self.inner.d_k),
            self.inner.keys.clone(),
        )
        .map(|a| a.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Values as an (n_rows, d_v) float64 array (one copy).
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Array2::from_shape_vec(
            (self.inner.n_rows(), self.inner.d_v),
            self.inner.values.clone(),
        )
        .map(|a| a.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.n_rows()
    }

    /// Reassemble a snapshot from raw buffers (e.g. read back from an
    /// Arrow/Parquet file written at this layer): `ids`/`owners` are
    /// parallel string lists, `written_at` is (n,), `keys` is
    /// (n, d_k), `values` is (n, d_v). Validated on restore; malformed
    /// buffers raise ValueError there, never panic.
    #[staticmethod]
    fn from_arrays(
        ids: Vec<String>,
        owners: Vec<String>,
        written_at: PyReadonlyArray1<'_, f64>,
        keys: PyReadonlyArray2<'_, f64>,
        values: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let n = ids.len();
        let (kr, kc) = keys.as_array().dim();
        let (vr, vc) = values.as_array().dim();
        if kr != n || vr != n {
            return Err(PyValueError::new_err(format!(
                "keys rows {kr} / values rows {vr} != len(ids)={n}"
            )));
        }
        if written_at.as_array().len() != n {
            return Err(PyValueError::new_err(format!(
                "written_at length {} != len(ids)={n}",
                written_at.as_array().len()
            )));
        }
        Ok(Self {
            inner: RsKvSnapshot {
                d_k: kc,
                d_v: vc,
                ids,
                owners,
                written_at: written_at.as_array().iter().copied().collect(),
                keys: keys.as_array().iter().copied().collect(),
                values: values.as_array().iter().copied().collect(),
            },
        })
    }
}

/// One partition's partial F_ε state: (m, num, den) shifted by the
/// running max.
///
/// Use `bruce.PartialTriple.from_pairs(scores, values, eps)` on each
/// shard, then `bruce.combine([p1, p2, ...], eps)` at a central node
/// to reduce them to a single triple, then `bruce.finalize(triple)`
/// to obtain the final A_ε output.
///
/// This is the Lemma B distributed primitive: F_ε partitions
/// losslessly. Bit-level identical to a single-machine computation.
#[pyclass(name = "PartialTriple")]
pub struct PyPartialTriple {
    inner: RsPartial,
}

#[pymethods]
impl PyPartialTriple {
    /// Build from per-row scores and value vectors.
    #[staticmethod]
    fn from_pairs(scores: Vec<f64>, values: Vec<Vec<f64>>, eps: f64) -> PyResult<Self> {
        let eps_obj = Eps::new(eps).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let vals: Vec<Array1<f64>> = values.into_iter().map(Array1::from).collect();
        Ok(Self {
            inner: RsPartial::from_pairs(&scores, &vals, eps_obj),
        })
    }

    #[getter]
    fn m_local(&self) -> f64 {
        self.inner.m_local
    }

    #[getter]
    fn den_local(&self) -> f64 {
        self.inner.den_local
    }

    fn num_local<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.num_local.clone().into_pyarray_bound(py)
    }
}

/// Combine partial triples from many shards into a single triple.
#[pyfunction]
#[pyo3(signature = (partials, eps))]
fn combine(partials: Vec<PyRef<'_, PyPartialTriple>>, eps: f64) -> PyResult<PyPartialTriple> {
    let eps_obj = Eps::new(eps).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parts: Vec<RsPartial> = partials.iter().map(|p| p.inner.clone()).collect();
    Ok(PyPartialTriple {
        inner: rs_combine(&parts, eps_obj),
    })
}

/// Finalize a combined triple into the attention output A_ε.
#[pyfunction]
fn finalize<'py>(
    py: Python<'py>,
    combined: PyRef<'_, PyPartialTriple>,
) -> Bound<'py, PyArray1<f64>> {
    rs_finalize(&combined.inner).into_pyarray_bound(py)
}

/// Three-relation streaming chain join — tuples arrive into R, S, T
/// one at a time, and `n_emitted` is the total count of join answers
/// produced so far.
///
/// This is the FlashAttention-online-softmax recursion transferred
/// to chain join (Bruce's "online Yannakakis" catalogue entry).
#[pyclass(name = "StreamingChainJoin")]
pub struct PyStreamingChainJoin {
    inner: RsStream<String>,
}

#[pymethods]
impl PyStreamingChainJoin {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsStream::new(),
        }
    }

    /// New R-tuple (a, b) arrives. Returns count of new join answers
    /// this arrival creates.
    fn arrive_r(&mut self, a: &str, b: &str) -> u64 {
        self.inner.arrive_r(a.to_string(), b.to_string())
    }

    /// New S-tuple (b, c) arrives.
    fn arrive_s(&mut self, b: &str, c: &str) -> u64 {
        self.inner.arrive_s(b.to_string(), c.to_string())
    }

    /// New T-tuple (c, d) arrives.
    fn arrive_t(&mut self, c: &str, d: &str) -> u64 {
        self.inner.arrive_t(c.to_string(), d.to_string())
    }

    /// Total join answers emitted so far.
    #[getter]
    fn n_emitted(&self) -> u64 {
        self.inner.n_emitted()
    }
}

/// Inner hash join over two sequences of integer keys. Returns
/// `(left_idx, right_idx)` pairs. O(|L| + |R|).
///
/// Note: for very large joins (many millions of matching pairs)
/// this materialises the full pair list as a Python list of tuples,
/// which costs ≈ 64 bytes/pair of Python overhead. Use
/// `hash_join_indices` to get back two numpy `int64` arrays at
/// 16 bytes/pair, or `hash_join_count` for SELECT COUNT(*).
#[pyfunction]
fn hash_join(left: Vec<i64>, right: Vec<i64>) -> Vec<(usize, usize)> {
    rs_hash_join(&left, &right)
}

/// Memory-efficient hash join: returns two numpy `int64` arrays
/// `(left_idx, right_idx)` instead of a Python list of tuples.
///
/// Memory cost ≈ 16 bytes per matching pair (4× smaller than
/// `hash_join`). For a 100M-pair join this is the difference
/// between 1.6 GB and ~10 GB.
#[pyfunction]
fn hash_join_indices<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let pairs = rs_hash_join(&left, &right);
    let mut li = Vec::<i64>::with_capacity(pairs.len());
    let mut ri = Vec::<i64>::with_capacity(pairs.len());
    for (l, r) in pairs {
        li.push(l as i64);
        ri.push(r as i64);
    }
    (
        Array1::from_vec(li).into_pyarray_bound(py),
        Array1::from_vec(ri).into_pyarray_bound(py),
    )
}

/// Count of matching pairs for a hash join, without materialising
/// any pair list at all. Memory cost is O(distinct keys), not O(pairs).
///
/// This is the fast path for `SELECT COUNT(*) FROM L INNER JOIN R ON …`.
#[pyfunction]
fn hash_join_count(left: Vec<i64>, right: Vec<i64>) -> u64 {
    use ahash::AHashMap;
    // build on the smaller side
    let (probe, build) = if left.len() <= right.len() {
        (&right, &left)
    } else {
        (&left, &right)
    };
    let mut counts: AHashMap<i64, u64> = AHashMap::with_capacity(build.len());
    for &k in build {
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut total: u64 = 0;
    for &k in probe {
        if let Some(&c) = counts.get(&k) {
            total += c;
        }
    }
    total
}

/// Streaming hash-join + reduce: never materialises the pair list.
///
/// For each matching `(li, ri)` we apply a reducer to the corresponding
/// value(s). Supported `agg_kind`:
///   "count"    → returns the number of matches as Python int.
///   "sum_left" → SUM(left_values[li]); needs `left_values` (float64).
///   "sum_right"→ SUM(right_values[ri]); needs `right_values`.
///   "min_left" / "max_left" / "min_right" / "max_right" → element-wise.
///
/// Memory cost: O(distinct keys), independent of |pairs|. This is what
/// the JOB-113 27a+ queries needed to avoid OOM.
#[pyfunction]
#[pyo3(signature = (left, right, agg_kind, left_values=None, right_values=None))]
fn hash_join_reduce<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
    agg_kind: &str,
    left_values: Option<PyReadonlyArray1<'_, f64>>,
    right_values: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<PyObject> {
    use ahash::AHashMap;

    let want_left = matches!(agg_kind, "sum_left" | "min_left" | "max_left");
    let want_right = matches!(agg_kind, "sum_right" | "min_right" | "max_right");

    if want_left && left_values.is_none() {
        return Err(PyValueError::new_err(format!(
            "agg_kind={agg_kind} requires left_values="
        )));
    }
    if want_right && right_values.is_none() {
        return Err(PyValueError::new_err(format!(
            "agg_kind={agg_kind} requires right_values="
        )));
    }

    let lv = left_values.as_ref().map(|a| a.as_array());
    let rv = right_values.as_ref().map(|a| a.as_array());
    if let Some(lv) = lv.as_ref() {
        if lv.len() != left.len() {
            return Err(PyValueError::new_err(
                "left_values length must match left keys",
            ));
        }
    }
    if let Some(rv) = rv.as_ref() {
        if rv.len() != right.len() {
            return Err(PyValueError::new_err(
                "right_values length must match right keys",
            ));
        }
    }

    // Build on left; for sum_left/min_left/max_left we need to fold the
    // left values per matching probe — bucket left rows by key.
    // `buckets[k]` = list of row indices on the build side.
    let mut buckets: AHashMap<i64, Vec<usize>> = AHashMap::with_capacity(left.len());
    for (i, &k) in left.iter().enumerate() {
        buckets.entry(k).or_default().push(i);
    }

    let mut count: u64 = 0;
    let mut sum: f64 = 0.0;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut any = false;

    for (ri, &k) in right.iter().enumerate() {
        if let Some(bucket) = buckets.get(&k) {
            for &li in bucket {
                count += 1;
                any = true;
                let v_for_pair = match agg_kind {
                    "sum_left" | "min_left" | "max_left" => lv.unwrap()[li],
                    "sum_right" | "min_right" | "max_right" => rv.unwrap()[ri],
                    _ => 0.0,
                };
                match agg_kind {
                    "sum_left" | "sum_right" => sum += v_for_pair,
                    "min_left" | "min_right" if v_for_pair < min_v => min_v = v_for_pair,
                    "max_left" | "max_right" if v_for_pair > max_v => max_v = v_for_pair,
                    _ => {}
                }
            }
        }
    }

    let obj: PyObject = match agg_kind {
        "count" => count.into_py(py),
        "sum_left" | "sum_right" => sum.into_py(py),
        "min_left" | "min_right" => (if any { min_v } else { f64::NAN }).into_py(py),
        "max_left" | "max_right" => (if any { max_v } else { f64::NAN }).into_py(py),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown agg_kind {agg_kind:?}; use count|sum_left|sum_right|\
                 min_left|min_right|max_left|max_right"
            )))
        }
    };
    Ok(obj)
}

/// Sort-merge join over PRE-SORTED ascending integer key sequences.
/// O(|L| + |R|).
#[pyfunction]
fn sort_merge_join(left: Vec<i64>, right: Vec<i64>) -> Vec<(usize, usize)> {
    rs_sort_merge(&left, &right)
}

/// Three-way Leapfrog-Triejoin over pre-sorted integer keys. Returns
/// triples (i, j, k) such that a[i] == b[j] == c[k]. Achieves the
/// AGM-optimal O(N^{3/2}) for triangle queries.
#[pyfunction]
fn lftj_three(a: Vec<i64>, b: Vec<i64>, c: Vec<i64>) -> Vec<(usize, usize, usize)> {
    rs_lftj_three(&a, &b, &c)
}

/// GDPR-style cascade delete: erase a subject's records across many
/// (table, fact_id) references in one transactional sweep, with a
/// receipt that lists every record removed.
///
/// Example:
///     >>> import bruce, numpy as np
///     >>> mem = bruce.KvMemory(d_k=2, d_v=1)
///     >>> for i in range(5):
///     ...     mem.write(f"cust42_r{i}", np.array([1.,0.]),
///     ...               np.array([float(i)]), owner="dpo")
///     >>> receipt = bruce.cascade_delete(
///     ...     mem,
///     ...     subject_id="customer42",
///     ...     table_name="rentals",
///     ...     fact_ids=[f"cust42_r{i}" for i in range(5)],
///     ...     owner="dpo",
///     ... )
///     >>> receipt["n_total"]      # 5 records erased
#[pyfunction]
#[pyo3(signature = (mem, subject_id, table_name, fact_ids, owner))]
fn cascade_delete<'py>(
    py: Python<'py>,
    mut mem: PyRefMut<'_, PyKvMemory>,
    subject_id: &str,
    table_name: &str,
    fact_ids: Vec<String>,
    owner: &str,
) -> PyResult<PyObject> {
    let refs = vec![(table_name, fact_ids.clone())];
    let plan = RsCascade {
        subject_id: subject_id.to_string(),
        references: refs,
        owner: owner.to_string(),
    };
    let receipt = plan
        .execute(table_name, &mut mem.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("subject_id", receipt.subject_id)?;
    d.set_item("owner", receipt.owner)?;
    d.set_item("n_total", receipt.n_total)?;
    let per_table = pyo3::types::PyList::empty_bound(py);
    for (tbl, ids, requested) in &receipt.per_table {
        let row = pyo3::types::PyDict::new_bound(py);
        row.set_item("table", tbl)?;
        row.set_item("deleted_ids", ids)?;
        row.set_item("requested", requested)?;
        per_table.append(row)?;
    }
    d.set_item("per_table", per_table)?;
    Ok(d.into())
}

/// Tree-structured causal attention — Bruce's identity:
/// causal attention on a tree mask is equivalent to running F_ε
/// on each row's ancestor path. For balanced trees this gives
/// O(N log N · d) total work, vs O(N² · d) for full causal.
///
/// `parents[i] = -1` marks a root. Otherwise must be < i (the path
/// terminates because indices strictly decrease).
///
/// Returns `(N, d_v)`.
#[pyfunction]
#[pyo3(signature = (q, k, v, parents, eps=1.0))]
fn tree_attention<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    v: PyReadonlyArray2<'_, f64>,
    parents: Vec<i64>,
    eps: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let eps = parse_eps(eps)?;
    let out = rs_tree_attention(&q.as_array(), &k.as_array(), &v.as_array(), &parents, eps)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

/// Chain tree: `parents[i] = i-1`. Recovers full causal attention.
#[pyfunction]
fn chain_tree(n: usize) -> Vec<i64> {
    rs_chain_tree(n)
}

/// Heap-shaped balanced binary tree: `parents[i] = (i-1)/2`. Depth log₂N.
#[pyfunction]
fn balanced_binary_tree(n: usize) -> Vec<i64> {
    rs_balanced_binary_tree(n)
}

/// k-ary balanced tree: `parents[i] = (i-1)/k`.
#[pyfunction]
fn k_ary_balanced_tree(n: usize, k: usize) -> PyResult<Vec<i64>> {
    if k == 0 {
        return Err(PyValueError::new_err("k-ary tree needs k >= 1"));
    }
    Ok(rs_k_ary_balanced_tree(n, k))
}

/// Star tree: one root, N-1 leaves all pointing to it.
#[pyfunction]
fn star_tree(n: usize) -> Vec<i64> {
    rs_star_tree(n)
}

/// Masked attention over an intensionally-given mask: a stream of
/// `(i, j)` index pairs in ARBITRARY order (duplicate-free). Row `i`
/// of `q` attends to row `j` of `k`/`v` iff `(i, j)` appears.
///
/// This is the PODS paper's "enumerate-then-fold" evaluator: any
/// duplicate-free enumeration of any mask — causal, window, tree,
/// join-query output — feeds one per-row max-shifted fold, and the
/// result is order-invariant because the fold is a commutative-monoid
/// homomorphism. `eps=0` gives the tropical argmax-mean, finite
/// `eps>0` softmax attention, `eps=inf` the plain mean.
///
/// `pairs`: int64 array of shape (P, 2). Returns `(out, covered)`:
/// `out` is `(N_q, d_v)` float64, `covered[i]` is False (zero row)
/// iff no pair mentioned row `i`.
#[pyfunction]
#[pyo3(signature = (q, k, v, pairs, eps=1.0))]
fn masked_attention<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    v: PyReadonlyArray2<'_, f64>,
    pairs: PyReadonlyArray2<'_, i64>,
    eps: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<bool>)> {
    let eps = parse_eps(eps)?;
    let pa = pairs.as_array();
    if pa.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "pairs must have shape (P, 2); got (_, {})",
            pa.ncols()
        )));
    }
    let mut ps = Vec::with_capacity(pa.nrows());
    for r in 0..pa.nrows() {
        let (i, j) = (pa[(r, 0)], pa[(r, 1)]);
        if i < 0 || j < 0 {
            return Err(PyValueError::new_err(format!(
                "mask pair ({i}, {j}) has a negative index",
            )));
        }
        ps.push((i as usize, j as usize));
    }
    let (out, covered) = rs_masked_attention(&q.as_array(), &k.as_array(), &v.as_array(), &ps, eps)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.into_pyarray_bound(py), covered))
}

/// Grouped soft-average: fused physical operator for
/// `SELECT g, SOFTAVG(v WEIGHT sim(k, :x) TEMP eps) ... GROUP BY g`.
///
/// `gid` is the dictionary-encoded grouping column (uint32, values in
/// `[0, n_groups)`); `sel` is an optional boolean selection evaluated
/// *before* scoring (the pushed-down `eps = 0` filter). One scan, one
/// `(mu, z, u)` accumulator per group. Returns `(out, covered)` with
/// `out` of shape `(n_groups, d_v)`.
#[pyfunction]
#[pyo3(signature = (x, k, v, gid, n_groups, eps=1.0, sel=None))]
#[allow(clippy::too_many_arguments)]
fn grouped_softavg<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    v: PyReadonlyArray2<'_, f64>,
    gid: PyReadonlyArray1<'_, u32>,
    n_groups: usize,
    eps: f64,
    sel: Option<PyReadonlyArray1<'_, bool>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<bool>)> {
    let eps = parse_eps(eps)?;
    let gid_slice = gid.as_slice()?;
    let sel_vec: Option<Vec<bool>> = sel.as_ref().map(|s| s.as_array().to_vec());
    let (out, covered) = rs_grouped_softavg(
        &x.as_array(),
        &k.as_array(),
        &v.as_array(),
        gid_slice,
        n_groups,
        sel_vec.as_deref(),
        eps,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.into_pyarray_bound(py), covered))
}

/// f32-storage variant of `grouped_softavg`: same contract, but `x`
/// and `k` are float32 and each row's score is computed in f32 (4-way
/// unrolled partial sums), widened to f64 once per row, and folded by
/// the same f64 `(mu, z, u)` accumulator. `v` stays float64.
///
/// Precision contract: f32 storage/scoring, f64 accumulation —
/// bandwidth is the scan's wall (half the key bytes), while the
/// max-shift anchoring that keeps sharp-eps answers finite lives in
/// the f64 fold.
#[pyfunction]
#[pyo3(signature = (x, k, v, gid, n_groups, eps=1.0, sel=None))]
#[allow(clippy::too_many_arguments)]
fn grouped_softavg_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f32>,
    k: PyReadonlyArray2<'_, f32>,
    v: PyReadonlyArray2<'_, f64>,
    gid: PyReadonlyArray1<'_, u32>,
    n_groups: usize,
    eps: f64,
    sel: Option<PyReadonlyArray1<'_, bool>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<bool>)> {
    let eps = parse_eps(eps)?;
    let gid_slice = gid.as_slice()?;
    let sel_vec: Option<Vec<bool>> = sel.as_ref().map(|s| s.as_array().to_vec());
    let (out, covered) = rs_grouped_softavg_f32(
        &x.as_array(),
        &k.as_array(),
        &v.as_array(),
        gid_slice,
        n_groups,
        sel_vec.as_deref(),
        eps,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.into_pyarray_bound(py), covered))
}

/// The causal mask `{(i, j) : j <= i}` as an int64 array of shape
/// `(n(n+1)/2, 2)` — convenience generator for `masked_attention`.
#[pyfunction]
fn causal_pairs(py: Python<'_>, n: usize) -> Bound<'_, PyArray2<i64>> {
    let ps = rs_causal_pairs(n);
    let mut arr = ndarray::Array2::<i64>::zeros((ps.len(), 2));
    for (r, (i, j)) in ps.into_iter().enumerate() {
        arr[(r, 0)] = i as i64;
        arr[(r, 1)] = j as i64;
    }
    arr.into_pyarray_bound(py)
}

/// The sliding-window mask `{(i, j) : 0 <= i - j <= w}` as an int64
/// array of shape `(P, 2)` — convenience generator for `masked_attention`.
#[pyfunction]
fn window_pairs(py: Python<'_>, n: usize, w: usize) -> Bound<'_, PyArray2<i64>> {
    let ps = rs_window_pairs(n, w);
    let mut arr = ndarray::Array2::<i64>::zeros((ps.len(), 2));
    for (r, (i, j)) in ps.into_iter().enumerate() {
        arr[(r, 0)] = i as i64;
        arr[(r, 1)] = j as i64;
    }
    arr.into_pyarray_bound(py)
}

/// Certified-smoothing temperature eps* (PODS smoothing corollary):
/// largest eps guaranteeing ||A_eps - A_0||_inf <= delta on every
/// input with score gap >= gap, argmax multiplicity >= kappa
/// (kappa=1 always valid), size <= n, |values| <= v_max.
#[pyfunction]
#[pyo3(signature = (delta, gap, v_max, n, kappa=1))]
fn eps_star(delta: f64, gap: f64, v_max: f64, n: usize, kappa: usize) -> PyResult<f64> {
    rs_eps_star(delta, gap, v_max, n, kappa).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Evaluate the quantitative-dequantization bound
/// `2 v_max (N-kappa)/kappa exp(-gap/eps)` on actual scores: an upper
/// bound on ||A_eps - A_0||_inf for any values with |V| <= v_max.
#[pyfunction]
fn dequantization_bound(scores: Vec<f64>, v_max: f64, eps: f64) -> PyResult<f64> {
    let eps = parse_eps(eps)?;
    Ok(rs_dequantization_bound(&scores, v_max, eps))
}

/// Bruce extension module — exported as `bruce._bruce`. The pure-Python
/// `bruce/__init__.py` re-exports these names at the top level.
#[pymodule]
fn _bruce(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOperator>()?;
    m.add_class::<PyIncrementalMemory>()?;
    m.add_class::<PyFuzzyJoinSketch>()?;
    m.add_class::<PyMerkleAuditLog>()?;
    m.add_class::<PyIdentity>()?;
    m.add_class::<PySignedFact>()?;
    m.add_class::<PyLaplaceMechanism>()?;
    m.add_class::<PyGaussianMechanism>()?;
    m.add_class::<PyAnonymityGuard>()?;
    m.add_class::<PyEncryptedBlob>()?;
    m.add_class::<PyKvMemory>()?;
    m.add_class::<PyKvSnapshot>()?;
    m.add_class::<PyPartialTriple>()?;
    m.add_class::<PyStreamingChainJoin>()?;
    m.add_class::<QuerySession>()?;
    m.add_function(wrap_pyfunction!(grouped_softavg, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_softavg_f32, m)?)?;
    m.add_function(wrap_pyfunction!(combine, m)?)?;
    m.add_function(wrap_pyfunction!(finalize, m)?)?;
    m.add_function(wrap_pyfunction!(hash_join, m)?)?;
    m.add_function(wrap_pyfunction!(hash_join_indices, m)?)?;
    m.add_function(wrap_pyfunction!(hash_join_count, m)?)?;
    m.add_function(wrap_pyfunction!(hash_join_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(sort_merge_join, m)?)?;
    m.add_function(wrap_pyfunction!(lftj_three, m)?)?;
    m.add_function(wrap_pyfunction!(cascade_delete, m)?)?;
    m.add_function(wrap_pyfunction!(tree_attention, m)?)?;
    m.add_function(wrap_pyfunction!(chain_tree, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_binary_tree, m)?)?;
    m.add_function(wrap_pyfunction!(k_ary_balanced_tree, m)?)?;
    m.add_function(wrap_pyfunction!(star_tree, m)?)?;
    m.add_function(wrap_pyfunction!(masked_attention, m)?)?;
    m.add_function(wrap_pyfunction!(causal_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(window_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(eps_star, m)?)?;
    m.add_function(wrap_pyfunction!(dequantization_bound, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ---------------------------------------------------------------------
// Query layer surface: the eps-algebra database behind one session.
// ---------------------------------------------------------------------

use bruce_query::db::{Database as RsDatabase, RowValues};
use bruce_query::logical::Pred;
use bruce_query::Table as RsTable;
use std::collections::HashMap as StdHashMap;

/// One eps-algebra database session: register Parquet tables, attach
/// key (embedding) columns, create maintained views, run SQL of the
/// form `SELECT g, SOFTAVG(val, SIM(key, :param), eps) FROM t
/// [WHERE col >= c] GROUP BY g`, and write through it.
#[pyclass]
struct QuerySession {
    inner: RsDatabase,
}

#[pymethods]
impl QuerySession {
    #[new]
    fn new() -> Self {
        QuerySession {
            inner: RsDatabase::new(),
        }
    }

    /// Load a Parquet file as a table (strings dictionary-encoded at load).
    fn register_parquet(&mut self, name: &str, path: &str) -> PyResult<()> {
        let t = RsTable::from_parquet(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.register(name, t);
        Ok(())
    }

    /// Attach an externally computed key matrix as a column. Dispatch
    /// is on dtype: float64 -> KeyF64 (f64 kernel), float32 -> KeyF32
    /// stored WITHOUT upcasting (f32 kernel: f32 scoring, f64
    /// accumulation — half the scan bytes).
    fn attach_key(&mut self, table: &str, name: &str, keys: &Bound<'_, PyAny>) -> PyResult<()> {
        let t = self
            .inner
            .catalog
            .tables
            .get_mut(table)
            .ok_or_else(|| PyValueError::new_err(format!("no table {table}")))?;
        // Attaching a column changes what the statistics must describe,
        // so they are invalidated on success: without a sketch for the
        // new key column the planner can never certify an error budget
        // and the cost model prices its scan at zero bytes.
        if let Ok(k64) = keys.extract::<PyReadonlyArray2<'_, f64>>() {
            t.attach_key_f64(name, k64.as_array().to_owned())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            self.inner.invalidate_stats(table);
            return Ok(());
        }
        if let Ok(k32) = keys.extract::<PyReadonlyArray2<'_, f32>>() {
            t.attach_key_f32(name, k32.as_array().to_owned())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            self.inner.invalidate_stats(table);
            return Ok(());
        }
        Err(PyValueError::new_err(
            "attach_key expects a 2-d numpy array of dtype float64 or float32",
        ))
    }

    /// Create a maintained soft-aggregate view (incremental under writes).
    #[pyo3(signature = (name, table, group_col, val_col, key_col, x, eps=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn create_view(
        &mut self,
        name: &str,
        table: &str,
        group_col: &str,
        val_col: &str,
        key_col: &str,
        x: PyReadonlyArray1<'_, f64>,
        eps: f64,
    ) -> PyResult<()> {
        self.inner
            .create_view(
                name,
                table,
                group_col,
                val_col,
                key_col,
                &x.as_array().to_owned(),
                eps,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Parse, optimize, cost-plan, and execute one SQL query.
    /// Returns `(labels, values, explain)`.
    fn run(
        &mut self,
        sql: &str,
        params: StdHashMap<String, PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<(Vec<String>, Vec<f64>, String)> {
        let p: StdHashMap<String, Array1<f64>> = params
            .into_iter()
            .map(|(k, v)| (k, v.as_array().to_owned()))
            .collect();
        let (result, planned) = self
            .inner
            .run(sql, &p)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((result.labels, result.values, planned.explain()))
    }

    /// Append one row (scalars, labels, keys given per column name);
    /// maintained views update incrementally.
    #[pyo3(signature = (table, scalars, labels, keys))]
    fn insert_row(
        &mut self,
        table: &str,
        scalars: StdHashMap<String, f64>,
        labels: StdHashMap<String, String>,
        keys: StdHashMap<String, PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let row = RowValues {
            scalars,
            labels,
            keys: keys
                .into_iter()
                .map(|(k, v)| (k, v.as_array().to_vec()))
                .collect(),
        };
        self.inner
            .insert_row(table, &row)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Delete rows matching `col <op> value` (`op` in {">=", "="}),
    /// maintaining views; returns the number of deleted rows.
    fn delete_where(&mut self, table: &str, col: &str, op: &str, value: f64) -> PyResult<usize> {
        let pred = match op {
            ">=" => Pred::GtEq(col.to_string(), value),
            "=" => Pred::Eq(col.to_string(), value),
            _ => return Err(PyValueError::new_err(format!("unsupported op {op}"))),
        };
        self.inner
            .delete_where(table, &pred)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

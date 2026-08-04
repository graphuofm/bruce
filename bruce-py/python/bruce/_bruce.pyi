"""Type stubs for the Rust extension module ``bruce._bruce`` (PyO3 + NumPy)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str

class Operator:
    """The F_eps operator: eps-parametrised attention / SQL-sum over (K, V)."""

    def __init__(self, eps: float = 1.0, sim: str = "dot") -> None: ...
    def attention(
        self,
        x: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute attention A_eps(x, K, V); x is (d_k,), K is (N, d_k), V is (N, d_v)."""
        ...
    def sum(
        self,
        x: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute SQL-style sum Q_eps(x, K, V)."""
        ...
    def attention_batch(
        self,
        q: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Batched attention: B queries (B, d_k) against one (K, V); returns (B, d_v)."""
        ...

class IncrementalMemory:
    """Incrementally maintained F_eps memory (Lemma A): insert/delete with O(1) output reads."""

    def __init__(
        self,
        query: npt.NDArray[np.float64],
        eps: float = 1.0,
        d_v: int = 1,
        sim: str = "dot",
    ) -> None: ...
    def insert(
        self,
        key_id: str,
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> None:
        """Insert a key/value pair identified by key_id."""
        ...
    def delete(self, key_id: str) -> None:
        """Delete a previously-inserted pair by id."""
        ...
    def insert_many(
        self,
        key_ids: list[str],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> None:
        """Batch insert: K is (N, d_k), V is (N, d_v), one key_ids[i] per row (single FFI crossing)."""
        ...
    def delete_many(self, key_ids: list[str]) -> None:
        """Batch delete by ids; stops at the first missing key."""
        ...
    def output(self) -> npt.NDArray[np.float64]:
        """Current attention output A_eps(x), shape (d_v,)."""
        ...
    def __len__(self) -> int:
        """How many live entries are in the memory."""
        ...
    @property
    def n_rescales(self) -> int:
        """How many full rescales have been triggered."""
        ...

class FuzzyJoinSketch:
    """Linear-attention kernel-trick sketch: O(d_phi * d_v) state and query time, independent of N."""

    def __init__(
        self,
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        phi: str = "elu+1",
    ) -> None: ...
    def add(
        self,
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> None:
        """Add one (k, v) row to the sketch incrementally; O(d_phi * d_v)."""
        ...
    def query(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Answer a fuzzy-join query in O(d_phi * d_v), independent of N; returns (d_v,)."""
        ...
    @property
    def n_rows(self) -> int:
        """Number of (K, V) rows aggregated into the sketch."""
        ...
    @property
    def size_bytes(self) -> int:
        """Total bytes of state (constant in N)."""
        ...
    def numerator(self) -> npt.NDArray[np.float64]:
        """Raw numerator matrix, shape (d_phi, d_v)."""
        ...
    def denominator(self) -> npt.NDArray[np.float64]:
        """Raw denominator vector, shape (d_phi,)."""
        ...

class MerkleAuditLog:
    """Append-only Merkle audit log for tamper-evident bookkeeping."""

    def __init__(self) -> None: ...
    def append(self, payload: bytes) -> int:
        """Append one entry (arbitrary bytes); returns the entry's index."""
        ...
    @property
    def len(self) -> int:
        """Number of entries in the log."""
        ...
    def root(self) -> bytes:
        """Current Merkle root (32 bytes)."""
        ...
    def proof(self, idx: int) -> list[bytes] | None:
        """Inclusion proof for entry idx (sibling hashes leaf-to-root), or None if out of range."""
        ...
    @staticmethod
    def verify(
        payload: bytes,
        idx: int,
        n_leaves: int,
        proof: list[bytes],
        root: bytes,
    ) -> bool:
        """Verify payload at index idx is included in root, given proof and the log length n_leaves."""
        ...

class Identity:
    """Ed25519 signing identity (keypair); construct via generate() or from_secret()."""

    @staticmethod
    def generate() -> Identity:
        """Generate a fresh identity from the OS CSPRNG."""
        ...
    @staticmethod
    def from_secret(secret: bytes) -> Identity:
        """Reconstruct from 32-byte secret bytes (e.g. read from a KMS)."""
        ...
    def secret_bytes(self) -> bytes:
        """Export the 32-byte secret. Handle with care."""
        ...
    def public_key(self) -> bytes:
        """Export the 32-byte public key (safe to share)."""
        ...
    def sign_fact(self, fact_id: str, owner: str, payload: bytes) -> SignedFact:
        """Sign a fact; returns a SignedFact verifiable by anyone holding the public key."""
        ...

class SignedFact:
    """A fact + Ed25519 signature + public key; self-contained, anyone can verify()."""

    def verify(self) -> bool:
        """Verify the signature; True on success, False otherwise."""
        ...
    def verify_or_raise(self) -> None:
        """Verify the signature; raises ValueError if invalid."""
        ...
    @property
    def fact_id(self) -> str: ...
    @property
    def owner(self) -> str: ...
    @property
    def payload(self) -> bytes: ...
    @property
    def signature(self) -> bytes: ...
    @property
    def public_key(self) -> bytes: ...
    @property
    def key_fingerprint(self) -> str: ...

class LaplaceMechanism:
    """Laplace mechanism for eps-DP: adds Lap(0, Delta/eps) noise to L1-sensitivity queries."""

    def __init__(
        self,
        l1_sensitivity: float,
        epsilon: float,
        seed: int | None = None,
    ) -> None: ...
    def release_scalar(self, true_value: float) -> float:
        """Release a single scalar with eps-DP noise added."""
        ...
    def release_vector(self, true_values: list[float]) -> list[float]:
        """Release a vector with independent Laplace noise per element."""
        ...
    @property
    def epsilon(self) -> float: ...

class GaussianMechanism:
    """Gaussian mechanism for (eps, delta)-DP on L2-sensitivity queries."""

    def __init__(
        self,
        l2_sensitivity: float,
        epsilon: float,
        delta: float = 1e-5,
        seed: int | None = None,
    ) -> None: ...
    def release_scalar(self, true_value: float) -> float:
        """Release a single scalar with (eps, delta)-DP Gaussian noise."""
        ...
    def release_vector(self, true_values: list[float]) -> list[float]:
        """Release a vector with independent Gaussian noise per element."""
        ...
    @property
    def sigma(self) -> float: ...

class AnonymityGuard:
    """k-anonymity / l-diversity query guard: rejects survivor sets too small or too homogeneous."""

    def __init__(self, k: int, l: int | None = None) -> None: ...
    def evaluate(self, survivors: list[str]) -> dict[str, Any]:
        """Evaluate against a survivor set; returns a dict describing the outcome (key 'status')."""
        ...

class EncryptedBlob:
    """AES-256-GCM encrypted blob (nonce + ciphertext + tag) for encrypted-at-rest persistence."""

    @staticmethod
    def key_from_passphrase(passphrase: str, /) -> bytes:
        """Derive a 32-byte key from a passphrase (SHA-256); convenience only, not a real KDF."""
        ...
    @staticmethod
    def encrypt(key: bytes, plaintext: bytes) -> EncryptedBlob:
        """Encrypt plaintext under a 32-byte key; returns an EncryptedBlob."""
        ...
    def decrypt(self, key: bytes) -> bytes:
        """Decrypt under the 32-byte key; raises ValueError on tag mismatch."""
        ...
    def to_bytes(self) -> bytes:
        """Wire-format bytes: [nonce(12) || ciphertext || tag(16)]."""
        ...
    @staticmethod
    def from_bytes(bytes: bytes) -> EncryptedBlob:
        """Parse an EncryptedBlob from wire format."""
        ...
    @property
    def nonce(self) -> bytes: ...

class KvMemory:
    """Durable K/V memory with audit log + owner-enforced delete; supports exact (eps=0) reads."""

    def __init__(self, d_k: int, d_v: int) -> None: ...
    def write(
        self,
        fact_id: str,
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        owner: str,
    ) -> None:
        """Write a (key, value, owner) triple under fact_id."""
        ...
    def delete(self, fact_id: str, owner: str) -> None:
        """Delete fact_id; the owner must match the writer."""
        ...
    def read_exact(
        self, fact_id: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Look up a fact by id (eps=0 read); returns (k, v) or None if absent/deleted."""
        ...
    @property
    def len_alive(self) -> int:
        """Number of alive (non-deleted) rows."""
        ...
    @property
    def len_total(self) -> int:
        """Number of rows in total (incl. deleted)."""
        ...
    @property
    def audit_log_len(self) -> int:
        """Length of the audit log."""
        ...
    def bulk_insert(
        self,
        ids: list[str],
        keys: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        owner: str,
    ) -> int:
        """Write n rows in one call; equivalent to a `write` loop but with one
        shared timestamp per batch. All-or-nothing: a shape or ownership error
        leaves the memory untouched. Returns the number of rows written."""
        ...
    def snapshot(self) -> KvSnapshot:
        """Export the live rows (tombstones skipped, insertion order kept)."""
        ...
    @staticmethod
    def restore(snap: KvSnapshot) -> KvMemory:
        """Rebuild from a snapshot; decode results are bitwise identical.
        Owners and write timestamps survive; the audit log starts empty."""
        ...
    def save_parquet(self, path: str) -> None:
        """Persist to a Parquet file; audit log goes to <path>.audit.jsonl alongside."""
        ...
    @staticmethod
    def load_parquet(path: str) -> KvMemory:
        """Load a snapshot from a Parquet file; restores <path>.audit.jsonl if present."""
        ...

class KvSnapshot:
    """Contiguous, row-major snapshot of a KvMemory's live rows.

    Produced by `KvMemory.snapshot()`, consumed by `KvMemory.restore`.
    Every accessor copies once out of the engine buffer — cache the result
    if you read it repeatedly. Arrow/Parquet wrapping belongs at this layer
    or above, never in bruce-core.
    """

    @property
    def n_rows(self) -> int:
        """Number of live rows."""
        ...
    @property
    def d_k(self) -> int:
        """Key dimensionality."""
        ...
    @property
    def d_v(self) -> int:
        """Value dimensionality."""
        ...
    @property
    def ids(self) -> list[str]:
        """Live fact ids, in insertion order."""
        ...
    @property
    def owners(self) -> list[str]:
        """Row owners, parallel to `ids`."""
        ...
    @property
    def written_at(self) -> npt.NDArray[np.float64]:
        """Row write timestamps (unix seconds), parallel to `ids`."""
        ...
    @property
    def keys(self) -> npt.NDArray[np.float64]:
        """Keys as an (n_rows, d_k) float64 array."""
        ...
    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Values as an (n_rows, d_v) float64 array."""
        ...
    def __len__(self) -> int: ...
    @staticmethod
    def from_arrays(
        ids: list[str],
        owners: list[str],
        written_at: npt.NDArray[np.float64],
        keys: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> KvSnapshot:
        """Reassemble a snapshot from raw buffers (e.g. read back from a file
        written at this layer). Malformed buffers raise ValueError."""
        ...


class PartialTriple:
    """One partition's partial F_eps state (m, num, den) for the Lemma B distributed reduction."""

    @staticmethod
    def from_pairs(
        scores: list[float],
        values: list[list[float]],
        eps: float,
    ) -> PartialTriple:
        """Build from per-row scores and value vectors."""
        ...
    @property
    def m_local(self) -> float: ...
    @property
    def den_local(self) -> float: ...
    def num_local(self) -> npt.NDArray[np.float64]:
        """Local numerator vector, shape (d_v,)."""
        ...

class StreamingChainJoin:
    """Three-relation streaming chain join; online-softmax recursion transferred to joins."""

    def __init__(self) -> None: ...
    def arrive_r(self, a: str, b: str) -> int:
        """New R-tuple (a, b) arrives; returns count of new join answers it creates."""
        ...
    def arrive_s(self, b: str, c: str) -> int:
        """New S-tuple (b, c) arrives; returns count of new join answers it creates."""
        ...
    def arrive_t(self, c: str, d: str) -> int:
        """New T-tuple (c, d) arrives; returns count of new join answers it creates."""
        ...
    @property
    def n_emitted(self) -> int:
        """Total join answers emitted so far."""
        ...

def combine(partials: list[PartialTriple], eps: float) -> PartialTriple:
    """Combine partial triples from many shards into a single triple."""
    ...

def finalize(combined: PartialTriple) -> npt.NDArray[np.float64]:
    """Finalize a combined triple into the attention output A_eps."""
    ...

def hash_join(left: list[int], right: list[int]) -> list[tuple[int, int]]:
    """Inner hash join over integer keys; returns (left_idx, right_idx) pairs in O(|L| + |R|)."""
    ...

def hash_join_indices(
    left: list[int], right: list[int]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Memory-efficient hash join: returns two int64 index arrays instead of a pair list."""
    ...

def hash_join_count(left: list[int], right: list[int]) -> int:
    """Count of matching join pairs without materialising them; O(distinct keys) memory."""
    ...

def hash_join_reduce(
    left: list[int],
    right: list[int],
    agg_kind: str,
    left_values: npt.NDArray[np.float64] | None = None,
    right_values: npt.NDArray[np.float64] | None = None,
) -> int | float:
    """Streaming hash-join + reduce (count|sum_left|sum_right|min_*|max_*); never materialises pairs."""
    ...

def sort_merge_join(left: list[int], right: list[int]) -> list[tuple[int, int]]:
    """Sort-merge join over PRE-SORTED ascending integer keys; O(|L| + |R|)."""
    ...

def lftj_three(
    a: list[int], b: list[int], c: list[int]
) -> list[tuple[int, int, int]]:
    """Three-way Leapfrog Triejoin over pre-sorted keys; AGM-optimal O(N^(3/2)) for triangles."""
    ...

def cascade_delete(
    mem: KvMemory,
    subject_id: str,
    table_name: str,
    fact_ids: list[str],
    owner: str,
) -> dict[str, Any]:
    """GDPR-style cascade delete across (table, fact_id) references; returns a receipt dict."""
    ...

def tree_attention(
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    parents: list[int],
    eps: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Tree-structured causal attention (parents[i] = -1 marks a root, else < i); returns (N, d_v)."""
    ...

def chain_tree(n: int) -> list[int]:
    """Chain tree parents[i] = i-1; recovers full causal attention."""
    ...

def balanced_binary_tree(n: int) -> list[int]:
    """Heap-shaped balanced binary tree: parents[i] = (i-1)/2; depth log2(N)."""
    ...

def k_ary_balanced_tree(n: int, k: int) -> list[int]:
    """k-ary balanced tree: parents[i] = (i-1)/k; requires k >= 1."""
    ...

def star_tree(n: int) -> list[int]:
    """Star tree: one root, N-1 leaves all pointing to it."""
    ...

def masked_attention(
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    pairs: npt.NDArray[np.int64],
    eps: float = 1.0,
) -> tuple[npt.NDArray[np.float64], list[bool]]:
    """Masked attention over duplicate-free (i, j) pairs of shape (P, 2); returns (out (N_q, d_v), covered)."""
    ...

def causal_pairs(n: int) -> npt.NDArray[np.int64]:
    """The causal mask {(i, j) : j <= i} as an int64 array of shape (n(n+1)/2, 2)."""
    ...

def window_pairs(n: int, w: int) -> npt.NDArray[np.int64]:
    """The sliding-window mask {(i, j) : 0 <= i - j <= w} as an int64 array of shape (P, 2)."""
    ...

def grouped_softavg(
    x: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    gid: npt.NDArray[np.uint32],
    n_groups: int,
    eps: float = 1.0,
    sel: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], list[bool]]:
    """Fused grouped soft-average, the physical operator behind
    `SELECT g, SOFTAVG(v WEIGHT sim(k, :x) TEMP eps) ... GROUP BY g`.

    `gid` is the dictionary-encoded grouping column (values in [0, n_groups));
    `sel` is an optional boolean selection evaluated BEFORE scoring (the
    pushed-down filter). One scan, one (mu, z, u) accumulator per group.
    Returns (out of shape (n_groups, d_v), covered)."""
    ...

def grouped_softavg_f32(
    x: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float64],
    gid: npt.NDArray[np.uint32],
    n_groups: int,
    eps: float = 1.0,
    sel: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], list[bool]]:
    """f32-storage grouped soft-average: f32 scoring, f64 accumulation; returns (out (n_groups, d_v), covered)."""
    ...

def eps_star(delta: float, gap: float, v_max: float, n: int, kappa: int = 1) -> float:
    """Certified-smoothing temperature eps*: largest eps with ||A_eps - A_0||_inf <= delta."""
    ...

def dequantization_bound(scores: list[float], v_max: float, eps: float) -> float:
    """Evaluate the dequantization bound 2*v_max*(N-kappa)/kappa*exp(-gap/eps) on actual scores."""
    ...


class QuerySession:
    """One eps-algebra database session.

    Register Parquet tables, attach key (embedding) columns, create
    maintained views, run SQL of the form
    `SELECT g, SOFTAVG(val, SIM(key, :param), eps) FROM t [WHERE col >= c]
    GROUP BY g`, and write through it.
    """

    def __init__(self) -> None: ...
    def register_parquet(self, name: str, path: str) -> None:
        """Load a Parquet file as a table (strings dictionary-encoded at load).

        Registering over an existing name replaces the table and drops any
        maintained views built on the old one."""
        ...
    def attach_key(
        self, table: str, name: str, keys: npt.NDArray[np.float64] | npt.NDArray[np.float32]
    ) -> None:
        """Attach an externally computed key matrix as a column. Dispatch is on
        dtype: float64 -> KeyF64 (f64 kernel), float32 -> KeyF32 stored WITHOUT
        upcasting (f32 scoring, f64 accumulation — half the scan bytes)."""
        ...
    def create_view(
        self,
        name: str,
        table: str,
        group_col: str,
        val_col: str,
        key_col: str,
        x: npt.NDArray[np.float64],
        eps: float = 1.0,
    ) -> None:
        """Create a maintained soft-aggregate view, updated incrementally by
        `insert_row` / `delete_where`. View names are unique per session and
        `eps` must be > 0 (the eps=0 endpoint has no incremental form)."""
        ...
    def run(
        self, sql: str, params: dict[str, npt.NDArray[np.float64]]
    ) -> tuple[list[str], list[float], str]:
        """Parse, optimize, cost-plan, and execute one SQL query.
        Returns (labels, values, explain)."""
        ...
    def insert_row(
        self,
        table: str,
        scalars: dict[str, float],
        labels: dict[str, str],
        keys: dict[str, npt.NDArray[np.float64]],
    ) -> None:
        """Append one row (scalars, labels, keys given per column name);
        maintained views update incrementally."""
        ...
    def delete_where(self, table: str, col: str, op: str, value: float) -> int:
        """Delete rows matching `col <op> value` (`op` in {">=", "="}),
        maintaining views; returns the number of deleted rows."""
        ...

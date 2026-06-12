//! Tamper-evident audit log via Merkle tree.
//!
//! Bruce's audit log records every CRUD op. A *tamper-evident* log
//! goes one step further: each new entry is committed by hashing into
//! a running Merkle root, so any later modification (insert, edit,
//! delete) of past entries invalidates every subsequent root.
//!
//! ### Use case
//!
//! A regulator asks: "prove you haven't quietly rewritten the audit
//! log between t=10 and t=20." If you publish the root at t=20 (e.g.
//! in a chain, in a tweet, on a notary service), no later tampering
//! at t<20 can produce a matching root.
//!
//! ### Design
//!
//! We use an **append-only Merkle log** (RFC 6962-style transparency
//! log). Each entry has a leaf hash; the tree is rebuilt
//! incrementally; we expose the current root and per-entry inclusion
//! proofs.

use sha2::{Digest, Sha256};

/// SHA-256 hash output (32 bytes).
pub type Hash = [u8; 32];

#[inline]
fn h(bytes: &[u8]) -> Hash {
    let d = Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

#[inline]
fn h_pair(left: &Hash, right: &Hash) -> Hash {
    let mut buf = [0u8; 65];
    buf[0] = 0x01; // domain separator: interior node
    buf[1..33].copy_from_slice(left);
    buf[33..65].copy_from_slice(right);
    h(&buf)
}

#[inline]
fn h_leaf(bytes: &[u8]) -> Hash {
    let mut buf = Vec::with_capacity(bytes.len() + 1);
    buf.push(0x00); // domain separator: leaf
    buf.extend_from_slice(bytes);
    h(&buf)
}

/// Append-only Merkle audit log.
///
/// We keep all leaf hashes; the root is computed on-demand from the
/// current leaf vector. For a 10⁶-entry log this is ~32 MB of leaf
/// storage and a sub-second root computation — fine for an audit
/// archive.
pub struct MerkleAuditLog {
    leaves: Vec<Hash>,
}

impl MerkleAuditLog {
    /// Empty log.
    pub fn new() -> Self {
        Self { leaves: Vec::new() }
    }

    /// Append one entry (e.g., a serialised AuditEntry). Returns the
    /// **index** of the new entry.
    pub fn append(&mut self, payload: &[u8]) -> usize {
        let idx = self.leaves.len();
        self.leaves.push(h_leaf(payload));
        idx
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Is the log empty?
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Current Merkle root over all appended entries. Empty tree
    /// returns a fixed sentinel (SHA-256 of the empty string).
    pub fn root(&self) -> Hash {
        if self.leaves.is_empty() {
            return h(&[]);
        }
        let mut level: Vec<Hash> = self.leaves.clone();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i + 1 < level.len() {
                next.push(h_pair(&level[i], &level[i + 1]));
                i += 2;
            }
            if i < level.len() {
                // odd one out — promote it (RFC 6962 variant)
                next.push(level[i]);
            }
            level = next;
        }
        level[0]
    }

    /// Inclusion proof for entry `idx`: the sibling hashes along the
    /// path from the leaf to the root. Returns `None` if `idx` is
    /// out of range.
    pub fn proof(&self, idx: usize) -> Option<Vec<Hash>> {
        if idx >= self.leaves.len() {
            return None;
        }
        let mut path = Vec::new();
        let mut i = idx;
        let mut level = self.leaves.clone();
        while level.len() > 1 {
            if i.is_multiple_of(2) {
                if i + 1 < level.len() {
                    path.push(level[i + 1]); // real right sibling
                }
                // else odd-out: no proof entry, just promote up
            } else {
                path.push(level[i - 1]); // real left sibling
            }
            // build next level
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut j = 0;
            while j + 1 < level.len() {
                next.push(h_pair(&level[j], &level[j + 1]));
                j += 2;
            }
            if j < level.len() {
                next.push(level[j]);
            }
            level = next;
            i /= 2;
        }
        Some(path)
    }

    /// Verify that a leaf is included in a published root.
    pub fn verify(
        payload: &[u8],
        idx: usize,
        n_leaves: usize,
        proof: &[Hash],
        root: &Hash,
    ) -> bool {
        if n_leaves == 0 {
            return false;
        }
        let mut hh = h_leaf(payload);
        let mut i = idx;
        let mut level_size = n_leaves;
        let mut p = 0usize;
        while level_size > 1 {
            if i.is_multiple_of(2) {
                if i + 1 < level_size {
                    if p >= proof.len() {
                        return false;
                    }
                    hh = h_pair(&hh, &proof[p]);
                    p += 1;
                }
                // odd-out: no sibling, no hash
            } else {
                if p >= proof.len() {
                    return false;
                }
                hh = h_pair(&proof[p], &hh);
                p += 1;
            }
            i /= 2;
            level_size = level_size.div_ceil(2);
        }
        &hh == root
    }
}

impl Default for MerkleAuditLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_has_sentinel_root() {
        let log = MerkleAuditLog::new();
        assert_eq!(log.root(), h(&[]));
    }

    #[test]
    fn single_entry_root_is_leaf_hash() {
        let mut log = MerkleAuditLog::new();
        log.append(b"first op");
        assert_eq!(log.root(), h_leaf(b"first op"));
    }

    #[test]
    fn appending_changes_root() {
        let mut log = MerkleAuditLog::new();
        log.append(b"a");
        let r1 = log.root();
        log.append(b"b");
        let r2 = log.root();
        assert_ne!(r1, r2);
    }

    #[test]
    fn inclusion_proof_verifies() {
        let mut log = MerkleAuditLog::new();
        for i in 0..10 {
            log.append(format!("op-{i}").as_bytes());
        }
        let root = log.root();
        for i in 0..10 {
            let payload = format!("op-{i}");
            let proof = log.proof(i).unwrap();
            assert!(
                MerkleAuditLog::verify(payload.as_bytes(), i, 10, &proof, &root),
                "inclusion proof failed for idx {i}"
            );
        }
    }

    #[test]
    fn tampered_entry_fails_verification() {
        let mut log = MerkleAuditLog::new();
        for i in 0..8 {
            log.append(format!("op-{i}").as_bytes());
        }
        let root = log.root();
        let proof = log.proof(3).unwrap();
        // try to "verify" a forged payload at index 3
        let forged = b"op-3-tampered";
        assert!(!MerkleAuditLog::verify(forged, 3, 8, &proof, &root));
    }

    #[test]
    fn roots_at_various_sizes_are_unique() {
        let mut log = MerkleAuditLog::new();
        let mut roots = std::collections::HashSet::new();
        for i in 0..15 {
            log.append(format!("op-{i}").as_bytes());
            roots.insert(log.root());
        }
        assert_eq!(roots.len(), 15); // each append yields a new root
    }
}

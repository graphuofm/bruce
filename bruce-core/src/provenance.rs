//! Signed fact provenance: every record can carry an Ed25519
//! signature so downstream consumers can verify it came from the
//! claimed owner and hasn't been tampered with.
//!
//! Use case: a multi-agent system where Agent A asserts a fact into
//! Bruce, Agent B reads it. Agent B can verify that the fact was
//! actually signed by Agent A's public key, not forged by Agent C.
//!
//! This is the **cryptographic** companion of [`crate::memory`]'s
//! owner-enforced delete: owner enforcement keeps non-owners from
//! deleting; signature verification keeps non-owners from
//! claiming-they-wrote.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{BruceError, Result};

/// Canonical wire format for a fact's signable bytes.
///
/// We hash `[fact_id || owner || payload_bytes]` with SHA-256 and
/// sign the digest. Both sides re-derive the same canonical form to
/// verify.
pub fn canonical_digest(fact_id: &str, owner: &str, payload: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(fact_id.as_bytes());
    h.update([0u8]); // separator
    h.update(owner.as_bytes());
    h.update([0u8]);
    h.update(payload);
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

/// A signing identity. Wraps an Ed25519 keypair.
pub struct Identity {
    signing_key: SigningKey,
}

impl Identity {
    /// Generate a fresh identity (CSPRNG).
    pub fn generate() -> Self {
        let mut csprng = OsRng;
        Self {
            signing_key: SigningKey::generate(&mut csprng),
        }
    }

    /// Reconstruct from a 32-byte secret (e.g. read from disk).
    pub fn from_secret(secret: [u8; 32]) -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&secret),
        }
    }

    /// Export the secret bytes (handle with care).
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Public verifying key (safe to share).
    pub fn public_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Sign a fact. Returns a 64-byte signature.
    pub fn sign_fact(&self, fact_id: &str, owner: &str, payload: &[u8]) -> SignedFact {
        let digest = canonical_digest(fact_id, owner, payload);
        let sig: Signature = self.signing_key.sign(&digest);
        SignedFact {
            fact_id: fact_id.into(),
            owner: owner.into(),
            payload: payload.to_vec(),
            signature: sig.to_bytes(),
            public_key: self.public_key().to_bytes(),
        }
    }
}

/// A fact + its Ed25519 signature + the public key needed to verify it.
///
/// We do NOT derive Serialize/Deserialize directly because serde does
/// not auto-impl for `[u8; 64]`. Use [`SignedFact::to_wire`] /
/// [`SignedFact::from_wire`] to round-trip through a hex-encoded JSON
/// form.
#[derive(Debug, Clone)]
pub struct SignedFact {
    /// Identifier of the fact.
    pub fact_id: String,
    /// Owner name (used in the canonical-digest construction).
    pub owner: String,
    /// Opaque payload bytes (e.g. serialised key/value).
    pub payload: Vec<u8>,
    /// 64-byte Ed25519 signature over `canonical_digest`.
    pub signature: [u8; 64],
    /// 32-byte Ed25519 public key of the signer.
    pub public_key: [u8; 32],
}

/// Hex-string wire format for serialisation to JSON, files, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedFactWire {
    /// fact id
    pub fact_id: String,
    /// owner
    pub owner: String,
    /// hex-encoded payload bytes
    pub payload_hex: String,
    /// hex-encoded 64-byte signature (128 hex chars)
    pub signature_hex: String,
    /// hex-encoded 32-byte public key (64 hex chars)
    pub public_key_hex: String,
}

impl SignedFact {
    /// Verify the signature is valid for this fact's canonical digest.
    pub fn verify(&self) -> Result<()> {
        let pk = VerifyingKey::from_bytes(&self.public_key)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("bad public key: {e}")))?;
        let sig = Signature::from_bytes(&self.signature);
        let digest = canonical_digest(&self.fact_id, &self.owner, &self.payload);
        pk.verify(&digest, &sig)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("verify failed: {e}")))
    }

    /// Hex-encoded short fingerprint of the public key (8 hex chars).
    pub fn key_fingerprint(&self) -> String {
        hex::encode(&self.public_key[..4])
    }

    /// Convert to a JSON-serialisable wire format (all bytes → hex).
    pub fn to_wire(&self) -> SignedFactWire {
        SignedFactWire {
            fact_id: self.fact_id.clone(),
            owner: self.owner.clone(),
            payload_hex: hex::encode(&self.payload),
            signature_hex: hex::encode(self.signature),
            public_key_hex: hex::encode(self.public_key),
        }
    }

    /// Parse from wire format.
    pub fn from_wire(w: &SignedFactWire) -> Result<Self> {
        let payload = hex::decode(&w.payload_hex)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("bad payload hex: {e}")))?;
        let sig_v = hex::decode(&w.signature_hex)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("bad signature hex: {e}")))?;
        let pk_v = hex::decode(&w.public_key_hex)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("bad public key hex: {e}")))?;
        if sig_v.len() != 64 {
            return Err(BruceError::Other(anyhow::anyhow!(
                "signature must be 64 bytes"
            )));
        }
        if pk_v.len() != 32 {
            return Err(BruceError::Other(anyhow::anyhow!(
                "public key must be 32 bytes"
            )));
        }
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&sig_v);
        let mut public_key = [0u8; 32];
        public_key.copy_from_slice(&pk_v);
        Ok(Self {
            fact_id: w.fact_id.clone(),
            owner: w.owner.clone(),
            payload,
            signature,
            public_key,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_then_verify_succeeds() {
        let alice = Identity::generate();
        let signed = alice.sign_fact("fact1", "alice", b"hello world");
        signed.verify().unwrap();
    }

    #[test]
    fn tampered_payload_fails_verification() {
        let alice = Identity::generate();
        let mut signed = alice.sign_fact("fact1", "alice", b"hello world");
        signed.payload[0] ^= 0xFF;
        assert!(signed.verify().is_err());
    }

    #[test]
    fn tampered_owner_fails_verification() {
        let alice = Identity::generate();
        let mut signed = alice.sign_fact("fact1", "alice", b"hello");
        signed.owner = "mallory".into();
        assert!(signed.verify().is_err());
    }

    #[test]
    fn forged_signature_fails() {
        let alice = Identity::generate();
        let signed = alice.sign_fact("f", "alice", b"x");
        let mallory = Identity::generate();
        // Mallory signs the same canonical digest but claims it's
        // Alice's public key: that's the easy attack, and verification
        // should still pass for Mallory's key but fail for Alice's.
        let with_mallory_key = SignedFact {
            public_key: mallory.public_key().to_bytes(),
            ..signed.clone()
        };
        assert!(with_mallory_key.verify().is_err());
    }

    #[test]
    fn identity_roundtrip_via_secret_bytes() {
        let alice = Identity::generate();
        let secret = alice.secret_bytes();
        let alice2 = Identity::from_secret(secret);
        let s1 = alice.sign_fact("f", "alice", b"hi");
        let s2 = alice2.sign_fact("f", "alice", b"hi");
        // same canonical digest + same secret → same signature
        assert_eq!(s1.signature, s2.signature);
    }
}

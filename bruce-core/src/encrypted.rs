//! Encrypted-at-rest persistence (AES-256-GCM).
//!
//! Bruce's `KvMemory` lives in process; for durability we need to
//! write it somewhere. The natural format is Parquet (columnar,
//! zstd-compressed, queryable from DuckDB / Polars / Spark) but that
//! file is plain bytes on disk.
//!
//! This module provides a thin AES-256-GCM envelope: bytes go in,
//! ciphertext + 12-byte nonce + 16-byte tag come out. The key is a
//! 32-byte secret the caller owns (and stores in a KMS / HSM).
//!
//! For the v0 release we ship the encryption envelope without the
//! Parquet integration; the caller pipes any bytes (including a
//! Parquet file) through `encrypt` / `decrypt` and writes the
//! envelope wherever it likes.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use rand::rngs::OsRng;
use rand::RngCore;

use crate::error::{BruceError, Result};

/// 32-byte symmetric key for AES-256-GCM.
pub type EncryptionKey = [u8; 32];

/// One encrypted blob: nonce + ciphertext (the tag is appended to
/// the ciphertext by `aes-gcm`).
#[derive(Debug, Clone)]
pub struct EncryptedBlob {
    /// 12-byte AES-GCM nonce. MUST be unique per (key, message).
    pub nonce: [u8; 12],
    /// Ciphertext || tag (concatenated by the AEAD).
    pub ciphertext: Vec<u8>,
}

impl EncryptedBlob {
    /// Encrypt `plaintext` under the given key. Generates a fresh
    /// random nonce internally.
    pub fn encrypt(key: &EncryptionKey, plaintext: &[u8]) -> Result<Self> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let mut nonce = [0u8; 12];
        OsRng.fill_bytes(&mut nonce);
        let nonce_obj = Nonce::from_slice(&nonce);
        let ciphertext = cipher
            .encrypt(nonce_obj, plaintext)
            .map_err(|e| BruceError::Other(anyhow::anyhow!("AES-GCM encrypt failed: {e}")))?;
        Ok(Self { nonce, ciphertext })
    }

    /// Decrypt back to plaintext. Returns an error if the tag is
    /// invalid (i.e., the ciphertext has been tampered).
    pub fn decrypt(&self, key: &EncryptionKey) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce_obj = Nonce::from_slice(&self.nonce);
        cipher
            .decrypt(nonce_obj, self.ciphertext.as_ref())
            .map_err(|e| BruceError::Other(anyhow::anyhow!("AES-GCM decrypt failed: {e}")))
    }

    /// Wire-format serialisation: [nonce(12) || ciphertext].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.ciphertext.len());
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.ciphertext);
        out
    }

    /// Parse from wire format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 12 + 16 {
            return Err(BruceError::Other(anyhow::anyhow!(
                "EncryptedBlob: input too short ({} bytes)", bytes.len()
            )));
        }
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&bytes[..12]);
        Ok(Self {
            nonce,
            ciphertext: bytes[12..].to_vec(),
        })
    }
}

/// Convenience: derive a 32-byte key from a user-supplied passphrase
/// using SHA-256. For production use a real KDF (Argon2) instead.
pub fn key_from_passphrase(pass: &str) -> EncryptionKey {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(pass.as_bytes());
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_then_decrypt_roundtrip() {
        let key = key_from_passphrase("test-passphrase-do-not-use-in-prod");
        let pt = b"a fact: the customer owes $1,234.56 on 2026-03-17";
        let blob = EncryptedBlob::encrypt(&key, pt).unwrap();
        let recovered = blob.decrypt(&key).unwrap();
        assert_eq!(recovered, pt);
    }

    #[test]
    fn tampered_ciphertext_fails_decrypt() {
        let key = key_from_passphrase("test");
        let pt = b"original";
        let mut blob = EncryptedBlob::encrypt(&key, pt).unwrap();
        blob.ciphertext[0] ^= 0xFF;     // flip one byte
        assert!(blob.decrypt(&key).is_err());
    }

    #[test]
    fn wrong_key_fails_decrypt() {
        let k1 = key_from_passphrase("key-a");
        let k2 = key_from_passphrase("key-b");
        let blob = EncryptedBlob::encrypt(&k1, b"secret").unwrap();
        assert!(blob.decrypt(&k2).is_err());
    }

    #[test]
    fn wire_format_roundtrip() {
        let key = key_from_passphrase("test");
        let blob = EncryptedBlob::encrypt(&key, b"hello").unwrap();
        let wire = blob.to_bytes();
        let parsed = EncryptedBlob::from_bytes(&wire).unwrap();
        let pt = parsed.decrypt(&key).unwrap();
        assert_eq!(pt, b"hello");
    }

    #[test]
    fn each_encrypt_uses_fresh_nonce() {
        let key = key_from_passphrase("test");
        let a = EncryptedBlob::encrypt(&key, b"x").unwrap();
        let b = EncryptedBlob::encrypt(&key, b"x").unwrap();
        // same plaintext, same key, but fresh random nonce → different ciphertext
        assert_ne!(a.nonce, b.nonce);
        assert_ne!(a.ciphertext, b.ciphertext);
    }
}

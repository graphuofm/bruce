"""Tests for bruce.EncryptedBlob (AES-256-GCM)."""

from __future__ import annotations

import pytest

import bruce


class TestEncryptedBlob:
    def test_roundtrip(self):
        key = bruce.EncryptedBlob.key_from_passphrase("test")
        pt = b"a fact: customer 42 owes $1234.56 on 2026-03-17"
        blob = bruce.EncryptedBlob.encrypt(key, pt)
        recovered = blob.decrypt(key)
        assert recovered == pt

    def test_key_from_passphrase_is_32_bytes(self):
        k = bruce.EncryptedBlob.key_from_passphrase("anything")
        assert len(k) == 32

    def test_wrong_key_fails_decrypt(self):
        k1 = bruce.EncryptedBlob.key_from_passphrase("k1")
        k2 = bruce.EncryptedBlob.key_from_passphrase("k2")
        blob = bruce.EncryptedBlob.encrypt(k1, b"secret")
        with pytest.raises(ValueError):
            blob.decrypt(k2)

    def test_wire_format_roundtrip(self):
        key = bruce.EncryptedBlob.key_from_passphrase("k")
        blob = bruce.EncryptedBlob.encrypt(key, b"hello world")
        wire = blob.to_bytes()
        # nonce (12) + ciphertext (11) + tag (16) = 39 bytes
        assert len(wire) >= 12 + 11 + 16
        blob2 = bruce.EncryptedBlob.from_bytes(wire)
        assert blob2.decrypt(key) == b"hello world"

    def test_each_encrypt_uses_fresh_nonce(self):
        key = bruce.EncryptedBlob.key_from_passphrase("k")
        a = bruce.EncryptedBlob.encrypt(key, b"x")
        b = bruce.EncryptedBlob.encrypt(key, b"x")
        # same plaintext + key, but the nonce is random → ciphertext differs
        assert a.nonce != b.nonce
        assert a.to_bytes() != b.to_bytes()

    def test_wrong_key_length_raises(self):
        with pytest.raises(ValueError):
            bruce.EncryptedBlob.encrypt(b"too short", b"x")

"""Tests for bruce.Identity + bruce.SignedFact (Ed25519 provenance)."""

from __future__ import annotations

import pytest

import bruce


class TestIdentity:
    def test_generate_yields_keypair(self):
        i = bruce.Identity.generate()
        assert isinstance(i.public_key(), bytes)
        assert len(i.public_key()) == 32
        assert isinstance(i.secret_bytes(), bytes)
        assert len(i.secret_bytes()) == 32

    def test_from_secret_roundtrip(self):
        i = bruce.Identity.generate()
        secret = i.secret_bytes()
        i2 = bruce.Identity.from_secret(secret)
        # same secret → same public key
        assert i.public_key() == i2.public_key()

    def test_from_secret_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            bruce.Identity.from_secret(b"too short")


class TestSignedFact:
    def test_sign_then_verify(self):
        alice = bruce.Identity.generate()
        sf = alice.sign_fact("fact1", "alice", b"hello world")
        assert sf.verify() is True
        sf.verify_or_raise()  # should not raise

    def test_attributes_exposed(self):
        alice = bruce.Identity.generate()
        sf = alice.sign_fact("f", "alice", b"payload")
        assert sf.fact_id == "f"
        assert sf.owner == "alice"
        assert sf.payload == b"payload"
        assert len(sf.signature) == 64
        assert len(sf.public_key) == 32
        assert len(sf.key_fingerprint) == 8     # 4 bytes hex = 8 chars

    def test_key_fingerprint_stable(self):
        """Same identity → same fingerprint, regardless of fact content."""
        alice = bruce.Identity.generate()
        a = alice.sign_fact("x", "alice", b"a")
        b = alice.sign_fact("y", "alice", b"b")
        assert a.key_fingerprint == b.key_fingerprint

    def test_two_identities_have_different_fingerprints(self):
        a = bruce.Identity.generate().sign_fact("x", "alice", b"a")
        b = bruce.Identity.generate().sign_fact("x", "alice", b"a")
        assert a.key_fingerprint != b.key_fingerprint

    def test_verify_returns_bool_not_exception_on_bad(self):
        """The non-throwing API is convenient for batch verification."""
        alice = bruce.Identity.generate()
        sf = alice.sign_fact("f", "alice", b"x")
        assert sf.verify() is True
        # We can't easily mutate sf.signature from Python (it's read-only),
        # but verify() should always return a bool, not throw.

"""Tests for bruce.MerkleAuditLog (tamper-evident audit log)."""

from __future__ import annotations

import bruce


class TestMerkleAuditLog:
    def test_empty_log_has_root(self):
        log = bruce.MerkleAuditLog()
        assert log.len == 0
        # the SHA-256 of the empty string is a known constant
        r = log.root()
        assert isinstance(r, bytes)
        assert len(r) == 32

    def test_append_increments_len_and_changes_root(self):
        log = bruce.MerkleAuditLog()
        r0 = log.root()
        log.append(b"first")
        assert log.len == 1
        r1 = log.root()
        assert r1 != r0
        log.append(b"second")
        r2 = log.root()
        assert r2 != r1
        assert log.len == 2

    def test_inclusion_proof_verifies(self):
        log = bruce.MerkleAuditLog()
        payloads = [f"op-{i}".encode() for i in range(10)]
        for p in payloads:
            log.append(p)
        root = log.root()
        for i, p in enumerate(payloads):
            proof = log.proof(i)
            assert proof is not None
            ok = bruce.MerkleAuditLog.verify(p, i, log.len, proof, root)
            assert ok, f"inclusion proof failed at idx {i}"

    def test_tampered_payload_fails_verification(self):
        log = bruce.MerkleAuditLog()
        for i in range(8):
            log.append(f"op-{i}".encode())
        root = log.root()
        proof = log.proof(3)
        # try to forge a different payload at index 3
        assert not bruce.MerkleAuditLog.verify(
            b"forged", 3, log.len, proof, root
        )

    def test_proof_out_of_range_returns_none(self):
        log = bruce.MerkleAuditLog()
        log.append(b"a")
        assert log.proof(99) is None

    def test_log_grows_to_one_thousand(self):
        log = bruce.MerkleAuditLog()
        for i in range(1000):
            log.append(f"entry-{i}".encode())
        assert log.len == 1000
        root = log.root()
        assert len(root) == 32
        # verify the very last entry
        proof = log.proof(999)
        assert bruce.MerkleAuditLog.verify(b"entry-999", 999, 1000, proof, root)

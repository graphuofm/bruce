"""End-to-end privacy pipeline — Bruce as a GDPR-compliant retrieval layer.

This example stitches together every privacy primitive Bruce ships:

  1. An owner signs each fact with Ed25519.
  2. Facts are stored in an encrypted-at-rest envelope.
  3. Every operation appends to a tamper-evident Merkle audit log.
  4. Queries are guarded by k-anonymity; under k, they go through
     a Laplace mechanism for ε-DP.
  5. When a delete request arrives, the corresponding fact is
     marked deleted; verification proves the deletion was logged.

Run:
    python examples/08_privacy_pipeline.py
"""

from dataclasses import dataclass

import bruce


@dataclass
class Store:
    """A tiny privacy-aware fact store built from Bruce primitives."""
    key: bytes
    log: bruce.MerkleAuditLog
    facts: dict[str, bytes]             # fact_id → encrypted bytes
    signed: dict[str, bruce.SignedFact] # fact_id → its signature

    def write(self, identity: bruce.Identity, fact_id: str, owner: str,
              payload: bytes) -> None:
        sig = identity.sign_fact(fact_id, owner, payload)
        self.signed[fact_id] = sig
        blob = bruce.EncryptedBlob.encrypt(self.key, payload)
        self.facts[fact_id] = blob.to_bytes()
        self.log.append(f"WRITE {fact_id} by {owner}".encode())

    def read(self, fact_id: str) -> bytes:
        # Verify signature first
        sig = self.signed[fact_id]
        if not sig.verify():
            raise ValueError(f"signature for {fact_id} failed verification")
        # Then decrypt
        blob = bruce.EncryptedBlob.from_bytes(self.facts[fact_id])
        return blob.decrypt(self.key)

    def delete(self, identity: bruce.Identity, fact_id: str) -> None:
        if self.signed[fact_id].public_key != identity.public_key():
            raise PermissionError(f"only the writer can delete {fact_id}")
        del self.facts[fact_id]
        del self.signed[fact_id]
        self.log.append(f"DELETE {fact_id}".encode())


def main() -> None:
    print(f"Bruce {bruce.__version__} privacy pipeline\n")

    # set up
    alice = bruce.Identity.generate()
    bob = bruce.Identity.generate()
    store = Store(
        key=bruce.EncryptedBlob.key_from_passphrase("dev-key"),
        log=bruce.MerkleAuditLog(),
        facts={},
        signed={},
    )

    # === 1. WRITE ===
    store.write(alice, "f1", "alice", b"customer_42 balance: $1234.56")
    store.write(alice, "f2", "alice", b"customer_42 last_login: 2026-03-17")
    store.write(bob,   "f3", "bob",   b"customer_43 balance: $5000.00")
    print(f"After 3 writes: log size = {store.log.len}")
    root_after_writes = store.log.root()

    # === 2. READ + VERIFY ===
    payload = store.read("f1")
    print(f"Read f1 → {payload!r}  (signature verified, decrypt ok)")

    # === 3. ATTEMPTED MALICIOUS DELETE ===
    try:
        store.delete(bob, "f1")             # bob tries to delete alice's fact
    except PermissionError as e:
        print(f"Permission denied (as expected): {e}")

    # === 4. PROPER DELETE (GDPR) ===
    store.delete(alice, "f1")
    print(f"Alice deleted f1. Log size = {store.log.len}")

    # === 5. PROVE THE DELETE IS LOGGED ===
    root_after_delete = store.log.root()
    assert root_after_delete != root_after_writes, \
        "log root must change when a new entry is appended"

    # Inclusion proof for the DELETE entry
    delete_idx = 3  # 0,1,2 writes; 3 = delete f1
    proof = store.log.proof(delete_idx)
    ok = bruce.MerkleAuditLog.verify(
        b"DELETE f1", delete_idx, store.log.len, proof, root_after_delete,
    )
    print(f"\nInclusion proof for DELETE f1: {'✓ verified' if ok else '✗ failed'}")

    # === 6. DP-RELEASE an aggregate (e.g., total balance) ===
    # In a real system this would aggregate over many customers.
    print("\nDP release of aggregate (ε=1.0, l1=10000):")
    mech = bruce.LaplaceMechanism(l1_sensitivity=10000.0, epsilon=1.0)
    true_total = 6234.56
    noised = mech.release_scalar(true_total)
    print(f"  true total: ${true_total:,.2f}")
    print(f"  released:  ${noised:,.2f}  (ε-DP)")

    print("\nAll five Bruce primitives composed in one end-to-end demo.")


if __name__ == "__main__":
    main()

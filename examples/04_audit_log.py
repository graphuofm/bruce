"""Tamper-evident audit log via Merkle tree.

Append operations to the log, publish the root at time T, then later
prove that a specific operation was indeed in the log at time T.

Run:
    python examples/04_audit_log.py
"""

import bruce


def main() -> None:
    log = bruce.MerkleAuditLog()

    # log a sequence of CRUD operations
    ops = [
        b"INSERT customer 42",
        b"UPDATE customer 42 set email='x@y.com'",
        b"DELETE customer 42",
        b"INSERT customer 43",
        b"INSERT customer 44",
    ]
    for op in ops:
        log.append(op)

    root_at_T = log.root()
    n_leaves_at_T = log.len
    print(f"Log size at time T: {n_leaves_at_T}")
    print(f"Merkle root at T:   {root_at_T.hex()}")
    print()

    # Later: prove that "DELETE customer 42" was in the log at time T
    target = b"DELETE customer 42"
    target_idx = ops.index(target)
    proof = log.proof(target_idx)
    print(f"Proving op #{target_idx} = {target!r} was in the log")
    print(f"Proof has {len(proof)} sibling hashes")

    ok = bruce.MerkleAuditLog.verify(
        target, target_idx, n_leaves_at_T, proof, root_at_T,
    )
    print(f"Verification: {'✓ valid' if ok else '✗ invalid'}")

    # Tampering check: try to claim a forged op was at index 0
    forged = b"INSERT customer 999"
    proof_forged = log.proof(0)
    bad = bruce.MerkleAuditLog.verify(
        forged, 0, n_leaves_at_T, proof_forged, root_at_T,
    )
    print(f"\nAttempted forge {forged!r} at idx 0: "
          f"{'rejected' if not bad else 'WRONGLY ACCEPTED'}")

    # The log keeps growing — old root no longer matches
    log.append(b"INSERT customer 45")
    root_after = log.root()
    print(f"\nAfter appending one more op, root changes: "
          f"{root_after != root_at_T}")


if __name__ == "__main__":
    main()

"""GDPR cascade — erase one subject's records across many table refs
with a single signed, auditable receipt.

Use case: when a customer exercises Article 17 (right to erasure),
the DPO walks the schema, gathers every fact_id tied to the subject,
and calls `bruce.cascade_delete` to remove all of them with a single
receipt that goes into the audit log.

Run:
    python examples/12_gdpr_cascade.py
"""

import numpy as np

import bruce


def main() -> None:
    rng = np.random.default_rng(0)

    # === build a multi-row memory for several customers ===
    mem = bruce.KvMemory(d_k=4, d_v=2)
    audit = bruce.MerkleAuditLog()

    customer_to_facts = {
        "cust-007": [f"cust-007.rental.{i}" for i in range(12)],
        "cust-042": [f"cust-042.rental.{i}" for i in range(5)],
        "cust-099": [f"cust-099.rental.{i}" for i in range(8)],
    }

    for cust_id, facts in customer_to_facts.items():
        for f in facts:
            k = rng.normal(size=4)
            v = rng.normal(size=2)
            mem.write(f, k, v, owner="dpo")
            audit.append(f"WRITE {f} (owner=dpo, subject={cust_id})".encode())

    print(f"Memory state:")
    print(f"  alive rows: {mem.len_alive}  (12 + 5 + 8 = 25)")
    print(f"  audit log size: {audit.len}")

    # === GDPR erasure request arrives for cust-042 ===
    print(f"\nGDPR erasure request: customer cust-042 ----- ")
    receipt = bruce.cascade_delete(
        mem,
        subject_id="cust-042",
        table_name="rentals",
        fact_ids=customer_to_facts["cust-042"],
        owner="dpo",
    )
    audit.append(
        f"GDPR_CASCADE subject={receipt['subject_id']} "
        f"deleted={receipt['n_total']} rows".encode()
    )

    print(f"Receipt:")
    print(f"  subject:  {receipt['subject_id']}")
    print(f"  owner:    {receipt['owner']}")
    print(f"  n_total:  {receipt['n_total']}")
    print(f"  per-table:")
    for t in receipt["per_table"]:
        print(f"    table={t['table']!r}  deleted={len(t['deleted_ids'])} ids")

    print(f"\nMemory state after erasure:")
    print(f"  alive rows: {mem.len_alive}  (12 + 8 = 20)")
    print(f"  audit log size: {audit.len}")
    print(f"  root: {audit.root().hex()}")

    # === prove the cascade was logged ===
    cascade_idx = audit.len - 1
    proof = audit.proof(cascade_idx)
    target = f"GDPR_CASCADE subject=cust-042 deleted=5 rows".encode()
    ok = bruce.MerkleAuditLog.verify(
        target, cascade_idx, audit.len, proof, audit.root()
    )
    print(f"\nInclusion proof for the cascade entry: "
          f"{'✓ valid' if ok else '✗ invalid'}")
    print("→ regulator can verify the deletion happened, against a")
    print("  pre-published root.")


if __name__ == "__main__":
    main()

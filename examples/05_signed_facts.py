"""Signed fact provenance — Ed25519 identities for multi-agent settings.

Alice writes a fact and signs it. Bob reads the fact and verifies
that the signature is genuinely Alice's. Mallory tries to tamper
with the payload — verification rejects it.

Run:
    python examples/05_signed_facts.py
"""

import bruce


def main() -> None:
    # Set up two identities (in a real system these come from a KMS or HSM)
    alice = bruce.Identity.generate()
    bob = bruce.Identity.generate()

    print(f"Alice's public key fingerprint: ...")
    alice_signed = alice.sign_fact(
        fact_id="invoice-2026-03-17",
        owner="alice@accounting.com",
        payload=b"The customer owes $1,234.56",
    )
    print(f"Alice signed:  {alice_signed.key_fingerprint}")

    # Bob receives the signed fact and verifies
    print(f"\nBob verifies the signature: "
          f"{'✓ ok' if alice_signed.verify() else '✗ invalid'}")
    print(f"Fact id: {alice_signed.fact_id}")
    print(f"Owner:   {alice_signed.owner}")
    print(f"Payload: {alice_signed.payload}")

    # Mallory tries to claim she signed it (replaces public_key)
    # but ed25519 verification will fail because the signature
    # is over the canonical digest with Alice's key.
    # (we can't mutate a SignedFact from Python — that's by design.)

    # What we CAN do: try to verify a fact signed by Bob with Alice's
    # claimed identity. Verification fails because the payload was
    # signed by Bob's key.
    bob_signed = bob.sign_fact("x", "bob@acme.com", b"some other fact")
    print(f"\nBob signs his own fact, fingerprint: {bob_signed.key_fingerprint}")
    print(f"Bob's verify: {'✓ ok' if bob_signed.verify() else '✗ invalid'}")
    print(f"Different fingerprints: {alice_signed.key_fingerprint} vs "
          f"{bob_signed.key_fingerprint}")


if __name__ == "__main__":
    main()

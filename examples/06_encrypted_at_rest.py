"""Encrypted-at-rest persistence via AES-256-GCM.

Encrypt a fact payload, write the ciphertext to disk, read it back,
decrypt, verify it matches.

Run:
    python examples/06_encrypted_at_rest.py
"""

import tempfile
from pathlib import Path

import bruce


def main() -> None:
    # Key derivation — for production use a real KDF (Argon2) + a KMS
    key = bruce.EncryptedBlob.key_from_passphrase("dev-password-do-not-ship")

    payload = b"customer 42 balance = $1,234.56 on 2026-03-17"
    print(f"Plaintext: {payload!r}")

    # Encrypt
    blob = bruce.EncryptedBlob.encrypt(key, payload)
    wire = blob.to_bytes()
    print(f"Ciphertext (hex): {wire.hex()}")
    print(f"  Wire format = nonce(12) || ciphertext || tag(16) = "
          f"{len(wire)} bytes")

    # Write to disk
    with tempfile.NamedTemporaryFile(suffix=".bruce", delete=False) as f:
        f.write(wire)
        path = Path(f.name)
    print(f"\nWrote to {path}, size {path.stat().st_size} bytes")

    # Read back
    on_disk = path.read_bytes()
    blob2 = bruce.EncryptedBlob.from_bytes(on_disk)
    recovered = blob2.decrypt(key)
    print(f"Recovered: {recovered!r}")
    assert recovered == payload
    print("✓ roundtrip matches")

    # Tamper detection
    tampered = bytearray(on_disk)
    tampered[20] ^= 0xFF             # flip one byte in the ciphertext
    blob_t = bruce.EncryptedBlob.from_bytes(bytes(tampered))
    try:
        blob_t.decrypt(key)
        print("✗ tamper undetected (this should NOT happen)")
    except ValueError as e:
        print(f"✓ tamper detected: {e}")

    path.unlink()


if __name__ == "__main__":
    main()

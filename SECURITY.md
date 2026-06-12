# Security Policy

## Supported versions

Pre-1.0: only the latest released minor version receives fixes.

## Reporting a vulnerability

Email the maintainer (see `Cargo.toml` authors) with a description and
reproduction steps. Please do not open public issues for
security-sensitive reports. You should receive an acknowledgement
within 7 days.

## Security model (read before deploying bruce-server)

- **Authentication**: optional JWT (HS256) via `--jwt-secret`. When
  enabled, every endpoint except `/health`, `/ready`, `/metrics`
  requires a valid Bearer token, and the token's `sub` claim must
  match the `owner` field on writes and deletes (cross-tenant
  enforcement). When disabled, the server trusts the `owner` field
  as-is — plaintext mode is for development only.
- **Transport**: optional TLS via `--tls-cert`/`--tls-key`; otherwise
  run behind a TLS-terminating reverse proxy.
- **Durability**: with `--wal-path`, every acknowledged write is in
  the WAL; a WAL append failure is returned to the client as HTTP 500
  and counted in `bruce_wal_fail_total` — alert on that metric.
- **Metrics endpoint is unauthenticated** by design (scrapers); do not
  expose it publicly if fact counts are sensitive.
- **DP / crypto primitives** (`LaplaceMechanism`, `GaussianMechanism`,
  `EncryptedBlob`, Merkle audit log) implement standard constructions;
  parameters are validated at the Python boundary. They have not been
  independently audited — do not rely on them as the sole control for
  high-stakes data without review.

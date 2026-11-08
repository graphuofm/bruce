-- bruce_pg upgrade: 0.1.0 -> 0.1.1
--
-- No functional changes: 0.1.1 is the release that introduces the
-- versioned-upgrade machinery itself (TESTING_MATRIX workstream 13).
-- The only object change is a trivial COMMENT so that
-- `ALTER EXTENSION bruce_pg UPDATE` has an observable effect.
--
-- pgrx (0.18.x) does not generate cross-version upgrade scripts; it
-- regenerates the full bruce_pg--<version>.sql each build. Hand-written
-- upgrade scripts belong in this crate-root sql/ directory: verified
-- 2026-08-03 that `cargo pgrx install`/`test` copies sql/*.sql into
-- <sharedir>/extension/ alongside the generated schema, so a packaged
-- install carries the upgrade chain automatically.

COMMENT ON AGGREGATE softavg(float8, float8, float8) IS
    'softavg(value, score, eps): max-anchored softmax average ((mu,z,u) monoid; bruce_pg 0.1.1)';

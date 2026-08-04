-- bruce_pg upgrade: 0.1.1 -> 0.1.2
--
-- OBSERVABLE BEHAVIOUR CHANGE (unlike 0.1.0 -> 0.1.1, which was inert).
-- Nothing in the SQL-level object definitions moves: the aggregates,
-- their support functions and their STYPE (float8[]) are unchanged, so
-- there is no CREATE/ALTER to perform here. What changes lives in the
-- shared library that `module_pathname` resolves to, and it changes
-- what these same aggregates RETURN:
--
--   1. +/-Inf scores now follow bruce-core's RowAcc policy.
--      Before 0.1.1 (inclusive) ScalarAcc had no infinity branches, so
--      any group holding a +/-Inf score evaluated exp(inf - inf) and
--      returned NaN. Now:
--        * +Inf at finite eps collapses the group to argmax semantics
--          over its +Inf rows (uniform ties), and that collapse
--          survives COMBINEFUNC, so parallel and serial plans agree;
--        * -Inf carries weight 0 at eps = 0 and finite eps;
--        * a group whose every row scores -Inf is now SQL NULL (it was
--          previously NaN, or the -Inf row's own value);
--        * eps = 'infinity' stays score-blind: +/-Inf rows are counted.
--
--   2. NaN in `value` or `score` now PROPAGATES: such a group returns
--      'NaN'::float8 in every eps regime, joining the AVG/SUM family
--      rather than silently discarding a real float8 value. Previously
--      NaN leaked through the max-anchored arithmetic non-uniformly
--      (f64::max ignores NaN, so a NaN score could be dropped by the
--      argmax comparison while a NaN value poisoned the sum). This is a
--      deliberate divergence from bruce-core, which SKIPS NaN because
--      there NaN is the engine's encoding of SQL NULL; PostgreSQL has a
--      real NULL, so the PG call site makes the PG-native choice. The
--      full argument, and the cost of the option not taken, are in
--      README.md, "Special float values: NaN, +/-Inf, and one
--      deliberate divergence".
--      bruce-core's skip semantics remain available to the user as
--      `WHERE NOT isnan(value) AND NOT isnan(score)`.
--
--   3. The aggregate state grew from float8[4] = [mu, z, u, eps] to
--      float8[5] = [mu, z, u, eps, nan_seen]. STYPE is float8[] either
--      way, so no catalog change is needed, but `softavg_state` output
--      is observably one element longer and slot 5 is the sticky NaN
--      bit (0/1). NOTE: partial states are not durable objects — they
--      live only inside a running aggregate — so no stored data needs
--      rewriting by this script. A state array persisted by hand under
--      0.1.1 and fed back after upgrade raises
--      "softavg: malformed state" rather than being misread.
--
-- Existing indexes/views over softavg results are unaffected mechanically
-- but their VALUES may change for rows involving NaN or +/-Inf scores;
-- REFRESH any materialized view that aggregates such data.

COMMENT ON AGGREGATE softavg(float8, float8, float8) IS
    'softavg(value, score, eps): max-anchored softmax average ((mu,z,u) monoid; bruce_pg 0.1.2: +/-Inf per bruce-core, NaN propagates)';

COMMENT ON AGGREGATE softavg_state(float8, float8, float8) IS
    'softavg_state(value, score, eps): raw monoid element float8[5] = [mu, z, u, eps, nan_seen] (bruce_pg 0.1.2)';

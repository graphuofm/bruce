-- bruce_pg upgrade: 0.1.2 -> 0.2.0
--
-- RESULTS DO NOT CHANGE: same arithmetic (softagg-core's ScalarAcc is the
-- 0.1.2 ScalarAcc, moved into a crate shared by every paper_icde engine),
-- same NaN / +-Inf / eps = 0 / eps = 'infinity' policy, same NULL
-- discipline and error messages. What changes is the representation and
-- what the aggregate supports:
--
--   1. STYPE float8[] -> internal. 0.1.2 decoded, re-allocated and
--      re-encoded a float8[5] varlena on every input row; 0.2.0 allocates
--      a 40-byte struct once per group in the aggregate memory context
--      and updates it in place. Parallel plans move partial states with
--      SERIALFUNC/DESERIALFUNC (33 bytes); softavg stays PARALLEL SAFE.
--   2. Moving-aggregate form (MSFUNC/MINVFUNC/MFINALFUNC, MSTYPE
--      internal): softavg(...) OVER (... ROWS BETWEEN k PRECEDING AND
--      CURRENT ROW) removes the departing row in O(1) instead of
--      re-aggregating the frame. When softagg-core reports the state
--      inexact (cancellation guard), MINVFUNC returns NULL and PostgreSQL
--      recomputes that frame.
--   3. Incremental maintenance: softavg_ivm_create / softavg_ivm_drop
--      (per-group InvAcc state table + statement triggers), with the
--      softavg_inv_* state functions and the softavg_inv_build aggregate.
--   4. softavg_state keeps its float8[] result ([mu, z, u, eps, nan_seen])
--      through a FINALFUNC, so explicit re-association with
--      bruce_softavg_combine / bruce_softavg_final works as before. Those
--      float8[] helpers and bruce_softavg_sfunc are kept unchanged.
--
-- CREATE OR REPLACE AGGREGATE (PostgreSQL >= 12) keeps the aggregates'
-- OIDs, so views and materialized views that use softavg survive the
-- upgrade untouched. New support objects (below) must match the fresh
-- install script exactly; experiments/postgres/upgrade_check.py compares
-- the catalog of an upgraded database with a fresh 0.2.0 install.

-- SQL-callable helpers (pgrx #[pg_extern]; same definitions as the generated
-- bruce_pg--0.2.0.sql)
CREATE  FUNCTION "softavg_core_version"() RETURNS TEXT 
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_core_version_wrapper';

CREATE  FUNCTION "softavg_inv_error_bound"(
	"st" bytea 
) RETURNS double precision 
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_inv_error_bound_wrapper';

CREATE  FUNCTION "softavg_inv_exact"(
	"st" bytea 
) RETURNS bool 
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_inv_exact_wrapper';

CREATE  FUNCTION "softavg_inv_merge"(
	"a" bytea, 
	"b" bytea 
) RETURNS bytea 
IMMUTABLE PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_inv_merge_wrapper';

CREATE  FUNCTION "softavg_inv_remove_many"(
	"st" bytea, 
	"vs" double precision[], 
	"ss" double precision[], 
	"eps" double precision 
) RETURNS bytea 
IMMUTABLE PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_inv_remove_many_wrapper';

CREATE  FUNCTION "softavg_inv_rows"(
	"st" bytea 
) RETURNS bigint 
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_inv_rows_wrapper';

CREATE  FUNCTION "softavg_inv_value"(
	"st" bytea 
) RETURNS double precision 
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_inv_value_wrapper';

CREATE  FUNCTION "softavg_ivm_note"(
	"recomputed_groups" bigint 
) RETURNS void
STRICT VOLATILE PARALLEL RESTRICTED
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_ivm_note_wrapper';

CREATE  FUNCTION "softavg_ivm_stats"() RETURNS TABLE (
	"trigger_statements" bigint,  
	"groups_recomputed" bigint  
)
STRICT VOLATILE PARALLEL RESTRICTED
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_ivm_stats_wrapper';

CREATE  FUNCTION "softavg_ivm_stats_reset"() RETURNS void
STRICT VOLATILE PARALLEL RESTRICTED
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_ivm_stats_reset_wrapper';

CREATE  FUNCTION "softavg_window_stats"() RETURNS TABLE (
	"fresh_states" bigint,  
	"absorbs" bigint,  
	"inverses" bigint,  
	"punts" bigint  
)
STRICT VOLATILE PARALLEL RESTRICTED
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_window_stats_wrapper';

CREATE  FUNCTION "softavg_window_stats_reset"() RETURNS void
STRICT VOLATILE PARALLEL RESTRICTED
LANGUAGE c 
AS 'MODULE_PATHNAME', 'softavg_window_stats_reset_wrapper';

-- internal-state support functions (raw V1 C functions in src/lib.rs)
CREATE FUNCTION softavg_sfunc(internal, float8, float8, float8) RETURNS internal
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_sfunc';
CREATE FUNCTION softavg_combine(internal, internal) RETURNS internal
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_combine';
CREATE FUNCTION softavg_serialize(internal) RETURNS bytea
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_serialize';
CREATE FUNCTION softavg_deserialize(bytea, internal) RETURNS internal
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_deserialize';
CREATE FUNCTION softavg_finalfn(internal) RETURNS float8
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_finalfn';
CREATE FUNCTION softavg_state_finalfn(internal) RETURNS float8[]
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_state_finalfn';
CREATE FUNCTION softavg_msfunc(internal, float8, float8, float8) RETURNS internal
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_msfunc';
CREATE FUNCTION softavg_minvfunc(internal, float8, float8, float8) RETURNS internal
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_minvfunc';
CREATE FUNCTION softavg_mfinalfn(internal) RETURNS float8
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_mfinalfn';

-- softavg(value, score, eps): max-anchored softmax average.
CREATE OR REPLACE AGGREGATE softavg(float8, float8, float8) (
    SFUNC        = softavg_sfunc,
    STYPE        = internal,
    SSPACE       = 40,
    COMBINEFUNC  = softavg_combine,
    SERIALFUNC   = softavg_serialize,
    DESERIALFUNC = softavg_deserialize,
    FINALFUNC    = softavg_finalfn,
    MSFUNC       = softavg_msfunc,
    MINVFUNC     = softavg_minvfunc,
    MSTYPE       = internal,
    MSSPACE      = 80,
    MFINALFUNC   = softavg_mfinalfn,
    PARALLEL     = SAFE
);

-- softavg_state: same transition; returns the 0.1.2 state array
-- [mu, z, u, eps, nan_seen] for explicit re-association in SQL.
CREATE OR REPLACE AGGREGATE softavg_state(float8, float8, float8) (
    SFUNC        = softavg_sfunc,
    STYPE        = internal,
    SSPACE       = 40,
    COMBINEFUNC  = softavg_combine,
    SERIALFUNC   = softavg_serialize,
    DESERIALFUNC = softavg_deserialize,
    FINALFUNC    = softavg_state_finalfn,
    PARALLEL     = SAFE
);

-- ---------------------------------------------------------------------
-- Incremental maintenance (softavg_ivm_create / softavg_ivm_drop)
-- ---------------------------------------------------------------------
CREATE FUNCTION softavg_inv_build_final(internal) RETURNS bytea
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_inv_build_final';
-- softavg_inv_build(value, score, eps): the serialized InvAcc of a row set
CREATE AGGREGATE softavg_inv_build(float8, float8, float8) (
    SFUNC = softavg_msfunc, STYPE = internal, SSPACE = 80,
    FINALFUNC = softavg_inv_build_final
);

-- softavg_ivm_create(view, base, group column, value expr, score expr, eps)
-- creates  <view>_state(g PRIMARY KEY, cnt, state bytea)  -- per-group InvAcc
--          <view>_maint()                                 -- trigger function
--          4 statement triggers on <base> (INSERT/DELETE/UPDATE with
--            transition tables, TRUNCATE)
--          VIEW <view>(<group column>, softavg, n)
-- The value/score expressions are evaluated over base rows (any SQL
-- expression over the base table's columns). NULL group keys are not
-- maintained. Inexact groups (softagg-core's error bound exceeded after
-- removals) are recomputed from the base table in the same statement.
CREATE FUNCTION softavg_ivm_create(view_name name, base regclass, group_col name,
                                   value_expr text, score_expr text, eps float8)
RETURNS void LANGUAGE plpgsql AS $fn$
DECLARE
    st    text := quote_ident(view_name || '_state');
    fn    text := quote_ident(view_name || '_maint');
    g     text := quote_ident(group_col);
    e     text := quote_literal(eps::text) || '::float8';
    gtype text;
BEGIN
    IF eps IS NULL OR eps < 0 OR eps = 'NaN'::float8 THEN
        RAISE EXCEPTION 'softavg_ivm_create: eps must be >= 0, got %', eps;
    END IF;
    SELECT format_type(atttypid, atttypmod) INTO gtype FROM pg_attribute
     WHERE attrelid = base AND attname = group_col AND attnum > 0 AND NOT attisdropped;
    IF gtype IS NULL THEN
        RAISE EXCEPTION 'softavg_ivm_create: column % not found in %', group_col, base;
    END IF;
    EXECUTE format('LOCK TABLE %s IN SHARE ROW EXCLUSIVE MODE', base);
    EXECUTE format('CREATE TABLE %s (g %s PRIMARY KEY, cnt bigint NOT NULL, state bytea NOT NULL)', st, gtype);
    EXECUTE format('INSERT INTO %s SELECT %s, count(*), softavg_inv_build((%s)::float8, (%s)::float8, %s) '
                   'FROM %s WHERE %s IS NOT NULL GROUP BY 1', st, g, value_expr, score_expr, e, base, g);
    EXECUTE format($body$
CREATE FUNCTION %1$s() RETURNS trigger LANGUAGE plpgsql AS $b$
DECLARE
    touched %2$s[];
    bad %2$s[];
    n int;
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        DELETE FROM %3$s;
        RETURN NULL;
    END IF;
    IF TG_OP = 'DELETE' THEN
        touched := ARRAY(SELECT DISTINCT %4$s FROM old_rows WHERE %4$s IS NOT NULL);
    ELSIF TG_OP = 'INSERT' THEN
        touched := ARRAY(SELECT DISTINCT %4$s FROM new_rows WHERE %4$s IS NOT NULL);
    ELSE
        touched := ARRAY(SELECT %4$s FROM old_rows WHERE %4$s IS NOT NULL
                         UNION SELECT %4$s FROM new_rows WHERE %4$s IS NOT NULL);
    END IF;
    IF TG_OP IN ('DELETE', 'UPDATE') THEN
        UPDATE %3$s AS t SET cnt = t.cnt - d.c,
               state = softavg_inv_remove_many(t.state, d.vs, d.ss, %5$s)
          FROM (SELECT %4$s AS g, count(*) AS c, array_agg((%6$s)::float8) AS vs,
                       array_agg((%7$s)::float8) AS ss
                  FROM old_rows WHERE %4$s IS NOT NULL GROUP BY 1) d
         WHERE t.g = d.g;
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') THEN
        INSERT INTO %3$s AS t (g, cnt, state)
        SELECT %4$s, count(*), softavg_inv_build((%6$s)::float8, (%7$s)::float8, %5$s)
          FROM new_rows WHERE %4$s IS NOT NULL GROUP BY 1
        ON CONFLICT (g) DO UPDATE SET cnt = t.cnt + EXCLUDED.cnt,
                                      state = softavg_inv_merge(t.state, EXCLUDED.state);
    END IF;
    DELETE FROM %3$s WHERE g = ANY(touched) AND cnt = 0;
    -- Every touched state row is now locked by this transaction, and this
    -- new statement (new snapshot under READ COMMITTED) sees all committed
    -- base rows plus our own, so a recompute cannot lose a concurrent
    -- writer's rows: that writer blocks on the state row until we commit
    -- and then applies its delta on top of the recomputed state.
    bad := ARRAY(SELECT g FROM %3$s WHERE g = ANY(touched) AND NOT softavg_inv_exact(state));
    n := 0;
    IF cardinality(bad) > 0 THEN
        UPDATE %3$s AS t
           SET state = (SELECT softavg_inv_build((%6$s)::float8, (%7$s)::float8, %5$s)
                          FROM %8$s b WHERE b.%4$s = t.g)
         WHERE t.g = ANY(bad);
        GET DIAGNOSTICS n = ROW_COUNT;
    END IF;
    PERFORM softavg_ivm_note(n);
    RETURN NULL;
END
$b$
$body$, fn, gtype, st, g, e, value_expr, score_expr, base);
    EXECUTE format('CREATE TRIGGER %s AFTER INSERT ON %s REFERENCING NEW TABLE AS new_rows '
                   'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_ins'), base, fn);
    EXECUTE format('CREATE TRIGGER %s AFTER DELETE ON %s REFERENCING OLD TABLE AS old_rows '
                   'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_del'), base, fn);
    EXECUTE format('CREATE TRIGGER %s AFTER UPDATE ON %s REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows '
                   'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_upd'), base, fn);
    EXECUTE format('CREATE TRIGGER %s AFTER TRUNCATE ON %s FOR EACH STATEMENT EXECUTE FUNCTION %s()',
                   quote_ident(view_name || '_trunc'), base, fn);
    EXECUTE format('CREATE VIEW %s AS SELECT g AS %s, softavg_inv_value(state) AS softavg, cnt AS n FROM %s',
                   quote_ident(view_name), g, st);
END
$fn$;

CREATE FUNCTION softavg_ivm_drop(view_name name) RETURNS void LANGUAGE plpgsql AS $fn$
BEGIN
    EXECUTE format('DROP VIEW IF EXISTS %s', quote_ident(view_name));
    EXECUTE format('DROP FUNCTION IF EXISTS %s() CASCADE', quote_ident(view_name || '_maint'));
    -- join views (softavg_ivm_create_join)
    EXECUTE format('DROP FUNCTION IF EXISTS %s() CASCADE', quote_ident(view_name || '_maint_base'));
    EXECUTE format('DROP FUNCTION IF EXISTS %s() CASCADE', quote_ident(view_name || '_maint_bridge'));
    EXECUTE format('DROP TABLE IF EXISTS %s', quote_ident(view_name || '_sync'));
    EXECUTE format('DROP TABLE IF EXISTS %s', quote_ident(view_name || '_state'));
END
$fn$;

-- softavg_ivm_create_join(view, base, base key, bridge, bridge key, group column,
--                         value expr, score expr, eps)
-- maintains the GROUP BY over a many-to-many join (overlapping groups):
--     SELECT br.<group_col>, softavg(<value_expr>, <score_expr>, eps), count(*)
--       FROM <base> b JOIN <bridge> br ON br.<bridge_key> = b.<base_key>
--      GROUP BY 1
-- The value/score expressions are over the base table's columns only. One
-- base-row edit fans out to every membership of the row (one state delta
-- per group it belongs to); a bridge-row edit is one membership delta.
-- Creates <view>_state (as softavg_ivm_create), <view>_sync (one row),
-- <view>_maint_base() / <view>_maint_bridge() and 4 statement triggers on
-- each table, and VIEW <view>(<group_col>, softavg, n).
-- Concurrency: a base row and a bridge row written by two concurrent
-- transactions would each miss the other (each statement's delta joins with
-- the other table's committed state only). Base writers therefore take a
-- shared and bridge writers an exclusive transaction-level advisory lock
-- (the key is derived from <view>_state's oid) BEFORE computing their
-- delta; under READ COMMITTED the delta queries run after the lock is
-- granted and see the other side's committed rows. Bridge writers, and base
-- writers under REPEATABLE READ / SERIALIZABLE, also update the <view>_sync
-- row, so a REPEATABLE READ / SERIALIZABLE base writer that raced a bridge
-- writer gets a serialization failure instead of a lost membership; bridge
-- writes themselves must run under READ COMMITTED.
-- Not supported: cascading foreign keys from the bridge to the base (the
-- cascaded bridge delete runs before the base statement trigger and neither
-- side can see the other's old rows) -- rejected here.
CREATE FUNCTION softavg_ivm_create_join(view_name name, base regclass, base_key name,
                                        bridge regclass, bridge_key name, group_col name,
                                        value_expr text, score_expr text, eps float8)
RETURNS void LANGUAGE plpgsql AS $fn$
DECLARE
    st    text := quote_ident(view_name || '_state');
    sy    text := quote_ident(view_name || '_sync');
    k     text := quote_ident(base_key);
    bk    text := quote_ident(bridge_key);
    g     text := quote_ident(group_col);
    e     text := quote_literal(eps::text) || '::float8';
    gtype text;
    lk    bigint;
    xbase text;   -- base rows -> (k__, v__, s__), expressions over the base's own columns
    xold  text;
    xnew  text;
    body  text;
    side  text;
    d_old text;
    d_new text;
BEGIN
    IF eps IS NULL OR eps < 0 OR eps = 'NaN'::float8 THEN
        RAISE EXCEPTION 'softavg_ivm_create_join: eps must be >= 0, got %', eps;
    END IF;
    IF base = bridge THEN
        RAISE EXCEPTION 'softavg_ivm_create_join: base and bridge must be different tables';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = base AND attname = base_key
                   AND attnum > 0 AND NOT attisdropped) THEN
        RAISE EXCEPTION 'softavg_ivm_create_join: column % not found in %', base_key, base;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = bridge AND attname = bridge_key
                   AND attnum > 0 AND NOT attisdropped) THEN
        RAISE EXCEPTION 'softavg_ivm_create_join: column % not found in %', bridge_key, bridge;
    END IF;
    SELECT format_type(atttypid, atttypmod) INTO gtype FROM pg_attribute
     WHERE attrelid = bridge AND attname = group_col AND attnum > 0 AND NOT attisdropped;
    IF gtype IS NULL THEN
        RAISE EXCEPTION 'softavg_ivm_create_join: column % not found in %', group_col, bridge;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE contype = 'f' AND conrelid = bridge AND confrelid = base
               AND (confdeltype IN ('c', 'n', 'd') OR confupdtype IN ('c', 'n', 'd'))) THEN
        RAISE EXCEPTION 'softavg_ivm_create_join: % has a cascading foreign key to %, which this maintenance cannot follow', bridge, base
            USING HINT = 'use ON DELETE/UPDATE NO ACTION or RESTRICT and delete bridge rows explicitly';
    END IF;
    EXECUTE format('LOCK TABLE %s, %s IN SHARE ROW EXCLUSIVE MODE', base, bridge);
    EXECUTE format('CREATE TABLE %s (g %s PRIMARY KEY, cnt bigint NOT NULL, state bytea NOT NULL)', st, gtype);
    EXECUTE format('CREATE TABLE %s (id int PRIMARY KEY, ver bigint NOT NULL)', sy);
    EXECUTE format('INSERT INTO %s VALUES (1, 0)', sy);
    lk := (23063::bigint << 32) + (st::regclass)::oid::bigint;
    xbase := format('(SELECT %s AS k__, (%s)::float8 AS v__, (%s)::float8 AS s__ FROM %s)', k, value_expr, score_expr, base);
    xold  := format('(SELECT %s AS k__, (%s)::float8 AS v__, (%s)::float8 AS s__ FROM old_rows)', k, value_expr, score_expr);
    xnew  := format('(SELECT %s AS k__, (%s)::float8 AS v__, (%s)::float8 AS s__ FROM new_rows)', k, value_expr, score_expr);
    EXECUTE format('INSERT INTO %s SELECT br.%s, count(*), softavg_inv_build(x.v__, x.s__, %s) '
                   'FROM %s br JOIN %s x ON x.k__ = br.%s WHERE br.%s IS NOT NULL GROUP BY 1',
                   st, g, e, bridge, xbase, bk, g);
    FOREACH side IN ARRAY ARRAY['base', 'bridge'] LOOP
        IF side = 'base' THEN
            d_old := format('(SELECT br.%1$s AS g__, x.v__, x.s__ FROM %2$s x JOIN %3$s br ON br.%4$s = x.k__ WHERE br.%1$s IS NOT NULL)',
                            g, xold, bridge, bk);
            d_new := format('(SELECT br.%1$s AS g__, x.v__, x.s__ FROM %2$s x JOIN %3$s br ON br.%4$s = x.k__ WHERE br.%1$s IS NOT NULL)',
                            g, xnew, bridge, bk);
        ELSE
            d_old := format('(SELECT o.%1$s AS g__, x.v__, x.s__ FROM old_rows o JOIN %2$s x ON x.k__ = o.%3$s WHERE o.%1$s IS NOT NULL)',
                            g, xbase, bk);
            d_new := format('(SELECT o.%1$s AS g__, x.v__, x.s__ FROM new_rows o JOIN %2$s x ON x.k__ = o.%3$s WHERE o.%1$s IS NOT NULL)',
                            g, xbase, bk);
        END IF;
        body := format($body$
CREATE FUNCTION %1$s() RETURNS trigger LANGUAGE plpgsql AS $b$
#variable_conflict use_column
DECLARE
    touched %2$s[];
    bad %2$s[];
    n int;
BEGIN
    -- serialize against writers of the other join side before reading it
    IF %10$L = 'base' THEN
        PERFORM pg_advisory_xact_lock_shared(%9$s);
    ELSE
        IF current_setting('transaction_isolation') <> 'read committed' THEN
            RAISE EXCEPTION 'softavg_ivm join view %%: writes to the bridge table must run under READ COMMITTED', %11$L
                USING ERRCODE = 'feature_not_supported';
        END IF;
        PERFORM pg_advisory_xact_lock(%9$s);
    END IF;
    IF %10$L = 'bridge' OR current_setting('transaction_isolation') <> 'read committed' THEN
        UPDATE %4$s SET ver = ver + 1 WHERE id = 1;
    END IF;
    IF TG_OP = 'TRUNCATE' THEN
        DELETE FROM %3$s;
        RETURN NULL;
    END IF;
    IF TG_OP = 'DELETE' THEN
        touched := ARRAY(SELECT DISTINCT g__ FROM %5$s d ORDER BY 1);
    ELSIF TG_OP = 'INSERT' THEN
        touched := ARRAY(SELECT DISTINCT g__ FROM %6$s d ORDER BY 1);
    ELSE
        touched := ARRAY(SELECT g__ FROM %5$s d UNION SELECT g__ FROM %6$s d ORDER BY 1);
    END IF;
    IF cardinality(touched) = 0 THEN
        PERFORM softavg_ivm_note(0);
        RETURN NULL;
    END IF;
    -- lock the touched state rows in key order (multi-group writers)
    PERFORM 1 FROM %3$s WHERE g = ANY(touched) ORDER BY g FOR UPDATE;
    IF TG_OP IN ('DELETE', 'UPDATE') THEN
        UPDATE %3$s AS t SET cnt = t.cnt - d.c,
               state = softavg_inv_remove_many(t.state, d.vs, d.ss, %7$s)
          FROM (SELECT g__, count(*) AS c, array_agg(v__) AS vs, array_agg(s__) AS ss
                  FROM %5$s d0 GROUP BY 1) d
         WHERE t.g = d.g__;
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') THEN
        INSERT INTO %3$s AS t (g, cnt, state)
        SELECT g__, count(*), softavg_inv_build(v__, s__, %7$s) FROM %6$s d0 GROUP BY 1
        ON CONFLICT (g) DO UPDATE SET cnt = t.cnt + EXCLUDED.cnt,
                                      state = softavg_inv_merge(t.state, EXCLUDED.state);
    END IF;
    DELETE FROM %3$s WHERE g = ANY(touched) AND cnt = 0;
    bad := ARRAY(SELECT g FROM %3$s WHERE g = ANY(touched) AND NOT softavg_inv_exact(state));
    n := 0;
    IF cardinality(bad) > 0 THEN
        UPDATE %3$s AS t
           SET state = (SELECT softavg_inv_build(x.v__, x.s__, %7$s)
                          FROM %12$s br JOIN %8$s x ON x.k__ = br.%13$s WHERE br.%14$s = t.g)
         WHERE t.g = ANY(bad);
        GET DIAGNOSTICS n = ROW_COUNT;
    END IF;
    PERFORM softavg_ivm_note(n);
    RETURN NULL;
END
$b$
$body$, quote_ident(view_name || '_maint_' || side), gtype, st, sy, d_old, d_new, e, xbase, lk, side,
        view_name, bridge, bk, g);
        EXECUTE body;
    END LOOP;
    FOREACH side IN ARRAY ARRAY['base', 'bridge'] LOOP
        EXECUTE format('CREATE TRIGGER %s AFTER INSERT ON %s REFERENCING NEW TABLE AS new_rows '
                       'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_' || side || '_ins'),
                       CASE side WHEN 'base' THEN base ELSE bridge END, quote_ident(view_name || '_maint_' || side));
        EXECUTE format('CREATE TRIGGER %s AFTER DELETE ON %s REFERENCING OLD TABLE AS old_rows '
                       'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_' || side || '_del'),
                       CASE side WHEN 'base' THEN base ELSE bridge END, quote_ident(view_name || '_maint_' || side));
        EXECUTE format('CREATE TRIGGER %s AFTER UPDATE ON %s REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows '
                       'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_' || side || '_upd'),
                       CASE side WHEN 'base' THEN base ELSE bridge END, quote_ident(view_name || '_maint_' || side));
        EXECUTE format('CREATE TRIGGER %s AFTER TRUNCATE ON %s FOR EACH STATEMENT EXECUTE FUNCTION %s()',
                       quote_ident(view_name || '_' || side || '_trunc'),
                       CASE side WHEN 'base' THEN base ELSE bridge END, quote_ident(view_name || '_maint_' || side));
    END LOOP;
    EXECUTE format('CREATE VIEW %s AS SELECT g AS %s, softavg_inv_value(state) AS softavg, cnt AS n FROM %s',
                   quote_ident(view_name), g, st);
END
$fn$;

COMMENT ON AGGREGATE softavg(float8, float8, float8) IS
    'softavg(value, score, eps): max-anchored softmax average (bruce_pg 0.2.0: internal state over softagg-core, moving-aggregate form)';
COMMENT ON AGGREGATE softavg_state(float8, float8, float8) IS
    'softavg_state(value, score, eps): state array float8[5] = [mu, z, u, eps, nan_seen] (bruce_pg 0.2.0)';

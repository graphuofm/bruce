//! softavg_ablation — softmax-ladder ablation build for PostgreSQL (NOT the
//! shipped bruce_pg; experiments/softmax_ladder only).
//!
//! STATUS: written, NOT compiled or run here — the release PostgreSQL 17.10
//! install this crate builds against (~/.pgrx/17.10/pg-release-install) was
//! deleted from the machine on 2026-11-06 and rebuilding it was refused by the
//! permission system (README "Not run"). The glue mirrors bruce_pg 0.2.0's raw
//! V1 functions (engines/postgres/src/lib.rs) line for line where it can.
//!
//! * `softavg_s3(value, score, eps)` — rung S3 as a moving aggregate:
//!   SFUNC = MSFUNC = `PlainInv::absorb` (= ScalarAcc::absorb + row count),
//!   MINVFUNC = `PlainInv::remove` (subtract at the current anchor) and it
//!   NEVER returns NULL, so PostgreSQL never recomputes a frame.
//! * IVM arms, `softavg_abl_ivm_create(view, base, group_col, value_expr,
//!   score_expr, eps, mode)`: the trigger structure of bruce_pg's
//!   softavg_ivm_create, over a 40-byte bytea state [PlainInv; eps];
//!   mode 's3' = deletes subtract (softavg_s3_remove_many), never recompute;
//!   mode 's2' = no removal: every group a DELETE/UPDATE touches is recomputed
//!   from the base table.
//! * S2 for windows needs no code: `softavg_fwd`, the harness aggregate built
//!   from bruce_pg's own softavg_sfunc/combine/serialize/deserialize/finalfn
//!   without a moving form (experiments/postgres/load.py), so PostgreSQL
//!   restarts every frame.
//! * `softavg_s3_window_stats()` / `_reset()`: MSFUNC / MINVFUNC call counts.

use ladder_core::PlainInv;
use pgrx::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

::pgrx::pg_module_magic!(name, version);

macro_rules! finfo_v1 {
    ($($name:ident),* $(,)?) => {
        $(
            #[no_mangle]
            #[doc(hidden)]
            pub extern "C" fn $name() -> &'static pg_sys::Pg_finfo_record {
                const V1: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
                &V1
            }
        )*
    };
}

finfo_v1!(pg_finfo_softavg_s3_sfunc, pg_finfo_softavg_s3_minvfunc, pg_finfo_softavg_s3_finalfn, pg_finfo_softavg_s3_build_final);

#[inline(always)]
unsafe fn arg(fcinfo: pg_sys::FunctionCallInfo, i: usize) -> pg_sys::NullableDatum {
    *(*fcinfo).args.as_ptr().add(i)
}

#[inline(always)]
unsafe fn arg_f64(fcinfo: pg_sys::FunctionCallInfo, i: usize) -> Option<f64> {
    let d = arg(fcinfo, i);
    if d.isnull {
        None
    } else {
        Some(f64::from_bits(d.value.value() as u64))
    }
}

#[inline(always)]
unsafe fn arg_ptr<T>(fcinfo: pg_sys::FunctionCallInfo, i: usize) -> *mut T {
    let d = arg(fcinfo, i);
    if d.isnull {
        std::ptr::null_mut()
    } else {
        d.value.cast_mut_ptr::<T>()
    }
}

#[inline(always)]
unsafe fn ret_null(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    (*fcinfo).isnull = true;
    pg_sys::Datum::from(0usize)
}

#[cold]
unsafe fn agg_context(fcinfo: pg_sys::FunctionCallInfo, who: &str) -> pg_sys::MemoryContext {
    let mut ctx: pg_sys::MemoryContext = std::ptr::null_mut();
    if pg_sys::AggCheckCallContext(fcinfo, &mut ctx) == 0 || ctx.is_null() {
        error!("{who} called in non-aggregate context");
    }
    ctx
}

#[cold]
unsafe fn alloc_in<T>(ctx: pg_sys::MemoryContext, v: T) -> *mut T {
    let p = pg_sys::MemoryContextAlloc(ctx, std::mem::size_of::<T>()) as *mut T;
    p.write(v);
    p
}

fn check_eps(eps: Option<f64>) -> f64 {
    match eps {
        None => error!("softavg_s3: eps must not be NULL"),
        Some(e) if e < 0.0 || e.is_nan() => error!("softavg_s3: eps must be >= 0, got {e}"),
        Some(e) => e,
    }
}

/// Moving / plain state: PlainInv + the group's eps (NaN until fixed) + live NaN rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct S3State {
    pub st: PlainInv,
    pub eps: f64,
    pub nan_rows: u64,
}

impl S3State {
    fn new() -> Self {
        Self { st: PlainInv::new(), eps: f64::NAN, nan_rows: 0 }
    }
    fn fix_eps(&mut self, e: f64) {
        if self.eps.is_nan() {
            self.eps = e;
        } else if self.eps != e {
            error!("softavg_s3: eps must be constant within a group ({} vs {e})", self.eps);
        }
    }
    fn finalize(&self) -> Option<f64> {
        if self.nan_rows > 0 {
            return Some(f64::NAN);
        }
        self.st.finalize()
    }
    fn to_bytes(&self) -> Vec<u8> {
        let mut o = self.st.serialize().to_vec();
        o.extend_from_slice(&self.eps.to_le_bytes());
        o.extend_from_slice(&self.nan_rows.to_le_bytes());
        o
    }
    fn from_bytes(b: &[u8]) -> Self {
        if b.len() != 48 {
            error!("softavg_s3: malformed state ({} bytes)", b.len());
        }
        Self {
            st: PlainInv::deserialize(&b[..32]).unwrap(),
            eps: f64::from_le_bytes(b[32..40].try_into().unwrap()),
            nan_rows: u64::from_le_bytes(b[40..48].try_into().unwrap()),
        }
    }
}

static ABSORBS: AtomicU64 = AtomicU64::new(0);
static INVERSES: AtomicU64 = AtomicU64::new(0);

/// SFUNC and MSFUNC(internal, value, score, eps) -> internal. Never returns
/// NULL (PostgreSQL forbids it for a moving transition function).
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_s3_sfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let mut st = arg_ptr::<S3State>(fcinfo, 0);
    if st.is_null() {
        st = alloc_in(agg_context(fcinfo, "softavg_s3_sfunc"), S3State::new());
    }
    ABSORBS.store(ABSORBS.load(Relaxed) + 1, Relaxed);
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        let e = check_eps(arg_f64(fcinfo, 3));
        (*st).fix_eps(e);
        if s.is_nan() || v.is_nan() {
            (*st).nan_rows += 1;
        } else {
            (*st).st.absorb(s, v, e);
        }
    }
    pg_sys::Datum::from(st)
}

/// MINVFUNC(internal, value, score, eps) -> internal. Plain subtraction at
/// the current anchor; NEVER returns NULL, so the frame is never recomputed.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_s3_minvfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<S3State>(fcinfo, 0);
    if st.is_null() {
        // PostgreSQL never calls MINVFUNC on a NULL state
        return ret_null(fcinfo);
    }
    INVERSES.store(INVERSES.load(Relaxed) + 1, Relaxed);
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        let e = check_eps(arg_f64(fcinfo, 3));
        (*st).fix_eps(e);
        if s.is_nan() || v.is_nan() {
            (*st).nan_rows = (*st).nan_rows.saturating_sub(1);
        } else {
            (*st).st.remove(s, v, e);
        }
    }
    pg_sys::Datum::from(st)
}

/// FINALFUNC / MFINALFUNC(internal) -> float8, whatever u/z is.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_s3_finalfn(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<S3State>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    match (*st).finalize() {
        Some(x) => pg_sys::Datum::from(x.to_bits()),
        None => ret_null(fcinfo),
    }
}

/// softavg_s3_build's FINALFUNC(internal) -> bytea (48 bytes).
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_s3_build_final(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<S3State>(fcinfo, 0);
    let bytes = if st.is_null() { S3State::new() } else { *st }.to_bytes();
    let total = pg_sys::VARHDRSZ + bytes.len();
    let p = pg_sys::palloc(total) as *mut pg_sys::varlena;
    pgrx::set_varsize_4b(p, total as i32);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), (p as *mut u8).add(pg_sys::VARHDRSZ), bytes.len());
    pg_sys::Datum::from(p)
}

#[pg_extern(immutable, parallel_safe)]
fn softavg_s3_merge(a: Option<&[u8]>, b: Option<&[u8]>) -> Option<Vec<u8>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x.to_vec()),
        (Some(x), Some(y)) => {
            let mut m = S3State::from_bytes(x);
            let o = S3State::from_bytes(y);
            if o.eps.is_nan() {
                m.nan_rows += o.nan_rows;
            } else {
                m.fix_eps(o.eps);
                m.st.merge(&o.st, m.eps);
                m.nan_rows += o.nan_rows;
            }
            Some(m.to_bytes())
        }
    }
}

/// Remove rows by plain subtraction (no certificate; the result is never flagged).
#[pg_extern(immutable, parallel_safe)]
fn softavg_s3_remove_many(st: Option<&[u8]>, vs: Vec<Option<f64>>, ss: Vec<Option<f64>>, eps: f64) -> Vec<u8> {
    if vs.len() != ss.len() {
        error!("softavg_s3_remove_many: value and score arrays differ in length");
    }
    let e = check_eps(Some(eps));
    let mut m = st.map(S3State::from_bytes).unwrap_or_else(S3State::new);
    m.fix_eps(e);
    for (v, s) in vs.into_iter().zip(ss) {
        if let (Some(v), Some(s)) = (v, s) {
            if s.is_nan() || v.is_nan() {
                m.nan_rows = m.nan_rows.saturating_sub(1);
            } else {
                m.st.remove(s, v, e);
            }
        }
    }
    m.to_bytes()
}

#[pg_extern(immutable, parallel_safe, strict)]
fn softavg_s3_value(st: &[u8]) -> Option<f64> {
    S3State::from_bytes(st).finalize()
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_s3_window_stats() -> TableIterator<'static, (name!(absorbs, i64), name!(inverses, i64))> {
    TableIterator::once((ABSORBS.load(Relaxed) as i64, INVERSES.load(Relaxed) as i64))
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_s3_window_stats_reset() {
    ABSORBS.store(0, Relaxed);
    INVERSES.store(0, Relaxed);
}

static IVM_RECOMPUTED: AtomicU64 = AtomicU64::new(0);

#[pg_extern(volatile, parallel_restricted)]
fn softavg_abl_ivm_note(recomputed_groups: i64) {
    IVM_RECOMPUTED.store(IVM_RECOMPUTED.load(Relaxed) + recomputed_groups.max(0) as u64, Relaxed);
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_abl_ivm_recomputed() -> i64 {
    IVM_RECOMPUTED.load(Relaxed) as i64
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_abl_ivm_reset() {
    IVM_RECOMPUTED.store(0, Relaxed);
}

extension_sql!(
    r#"
CREATE FUNCTION softavg_s3_sfunc(internal, float8, float8, float8) RETURNS internal
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_s3_sfunc';
CREATE FUNCTION softavg_s3_minvfunc(internal, float8, float8, float8) RETURNS internal
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_s3_minvfunc';
CREATE FUNCTION softavg_s3_finalfn(internal) RETURNS float8
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_s3_finalfn';
CREATE FUNCTION softavg_s3_build_final(internal) RETURNS bytea
    LANGUAGE c IMMUTABLE PARALLEL SAFE AS 'MODULE_PATHNAME', 'softavg_s3_build_final';

-- rung S3: removal without certificate (MINVFUNC never returns NULL)
CREATE AGGREGATE softavg_s3(float8, float8, float8) (
    SFUNC = softavg_s3_sfunc, STYPE = internal, SSPACE = 48, FINALFUNC = softavg_s3_finalfn,
    MSFUNC = softavg_s3_sfunc, MINVFUNC = softavg_s3_minvfunc, MSTYPE = internal, MSSPACE = 48,
    MFINALFUNC = softavg_s3_finalfn
);
CREATE AGGREGATE softavg_s3_build(float8, float8, float8) (
    SFUNC = softavg_s3_sfunc, STYPE = internal, SSPACE = 48, FINALFUNC = softavg_s3_build_final
);

-- IVM arms. mode 's3': deletes subtract, nothing is ever recomputed.
--           mode 's2': no removal; every group a DELETE/UPDATE touches is recomputed.
CREATE FUNCTION softavg_abl_ivm_create(view_name name, base regclass, group_col name,
                                       value_expr text, score_expr text, eps float8, mode text)
RETURNS void LANGUAGE plpgsql AS $fn$
DECLARE
    st    text := quote_ident(view_name || '_state');
    fn    text := quote_ident(view_name || '_maint');
    g     text := quote_ident(group_col);
    e     text := quote_literal(eps::text) || '::float8';
    gtype text;
    del_sql text;
BEGIN
    IF mode NOT IN ('s2', 's3') THEN
        RAISE EXCEPTION 'softavg_abl_ivm_create: mode must be s2 or s3';
    END IF;
    SELECT format_type(atttypid, atttypmod) INTO gtype FROM pg_attribute
     WHERE attrelid = base AND attname = group_col AND attnum > 0 AND NOT attisdropped;
    EXECUTE format('CREATE TABLE %s (g %s PRIMARY KEY, cnt bigint NOT NULL, state bytea NOT NULL)', st, gtype);
    EXECUTE format('INSERT INTO %s SELECT %s, count(*), softavg_s3_build((%s)::float8, (%s)::float8, %s) '
                   'FROM %s WHERE %s IS NOT NULL GROUP BY 1', st, g, value_expr, score_expr, e, base, g);
    IF mode = 's3' THEN
        del_sql := format($d$
        UPDATE %1$s AS t SET cnt = t.cnt - d.c,
               state = softavg_s3_remove_many(t.state, d.vs, d.ss, %2$s)
          FROM (SELECT %3$s AS g, count(*) AS c, array_agg((%4$s)::float8) AS vs,
                       array_agg((%5$s)::float8) AS ss
                  FROM old_rows WHERE %3$s IS NOT NULL GROUP BY 1) d
         WHERE t.g = d.g;$d$, st, e, g, value_expr, score_expr);
    ELSE
        del_sql := format($d$
        UPDATE %1$s AS t SET cnt = t.cnt - d.c
          FROM (SELECT %2$s AS g, count(*) AS c FROM old_rows WHERE %2$s IS NOT NULL GROUP BY 1) d
         WHERE t.g = d.g;$d$, st, g);
    END IF;
    EXECUTE format($body$
CREATE FUNCTION %1$s() RETURNS trigger LANGUAGE plpgsql AS $b$
DECLARE
    dtouched %2$s[];
    touched %2$s[];
    n int := 0;
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        DELETE FROM %3$s;
        RETURN NULL;
    END IF;
    IF TG_OP IN ('DELETE', 'UPDATE') THEN
        dtouched := ARRAY(SELECT DISTINCT %4$s FROM old_rows WHERE %4$s IS NOT NULL);
        %9$s
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') THEN
        touched := ARRAY(SELECT DISTINCT %4$s FROM new_rows WHERE %4$s IS NOT NULL);
        INSERT INTO %3$s AS t (g, cnt, state)
        SELECT %4$s, count(*), softavg_s3_build((%6$s)::float8, (%7$s)::float8, %5$s)
          FROM new_rows WHERE %4$s IS NOT NULL GROUP BY 1
        ON CONFLICT (g) DO UPDATE SET cnt = t.cnt + EXCLUDED.cnt,
                                      state = softavg_s3_merge(t.state, EXCLUDED.state);
    END IF;
    DELETE FROM %3$s WHERE (g = ANY(dtouched) OR g = ANY(touched)) AND cnt = 0;
    IF %10$L = 's2' AND cardinality(dtouched) > 0 THEN
        UPDATE %3$s AS t
           SET state = (SELECT softavg_s3_build((%6$s)::float8, (%7$s)::float8, %5$s)
                          FROM %8$s b WHERE b.%4$s = t.g)
         WHERE t.g = ANY(dtouched);
        GET DIAGNOSTICS n = ROW_COUNT;
    END IF;
    PERFORM softavg_abl_ivm_note(n);
    RETURN NULL;
END
$b$
$body$, fn, gtype, st, g, e, value_expr, score_expr, base, del_sql, mode);
    EXECUTE format('CREATE TRIGGER %s AFTER INSERT ON %s REFERENCING NEW TABLE AS new_rows '
                   'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_ins'), base, fn);
    EXECUTE format('CREATE TRIGGER %s AFTER DELETE ON %s REFERENCING OLD TABLE AS old_rows '
                   'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_del'), base, fn);
    EXECUTE format('CREATE TRIGGER %s AFTER UPDATE ON %s REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows '
                   'FOR EACH STATEMENT EXECUTE FUNCTION %s()', quote_ident(view_name || '_upd'), base, fn);
    EXECUTE format('CREATE TRIGGER %s AFTER TRUNCATE ON %s FOR EACH STATEMENT EXECUTE FUNCTION %s()',
                   quote_ident(view_name || '_trunc'), base, fn);
    EXECUTE format('CREATE VIEW %s AS SELECT g AS %s, softavg_s3_value(state) AS softavg, cnt AS n FROM %s',
                   quote_ident(view_name), g, st);
END
$fn$;

CREATE FUNCTION softavg_abl_ivm_drop(view_name name) RETURNS void LANGUAGE plpgsql AS $fn$
BEGIN
    EXECUTE format('DROP VIEW IF EXISTS %s', quote_ident(view_name));
    EXECUTE format('DROP FUNCTION IF EXISTS %s() CASCADE', quote_ident(view_name || '_maint'));
    EXECUTE format('DROP TABLE IF EXISTS %s', quote_ident(view_name || '_state'));
END
$fn$;
"#,
    name = "ablation_sql",
    finalize
);

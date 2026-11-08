//! bruce_pg 0.2.0 — `softavg(value, score, eps)` as a native PostgreSQL
//! aggregate over the shared softagg-core state.
//!
//! ```text
//!   softagg_core::ScalarAcc::absorb   ->  SFUNC        softavg_sfunc
//!   softagg_core::ScalarAcc::merge    ->  COMBINEFUNC  softavg_combine
//!   (bytes)                           ->  SERIALFUNC   softavg_serialize
//!                                         DESERIALFUNC softavg_deserialize
//!   softagg_core::ScalarAcc::finalize ->  FINALFUNC    softavg_finalfn
//!   softagg_core::InvAcc::absorb      ->  MSFUNC       softavg_msfunc
//!   softagg_core::InvAcc::remove      ->  MINVFUNC     softavg_minvfunc
//!   softagg_core::InvAcc::finalize    ->  MFINALFUNC   softavg_mfinalfn
//! ```
//!
//! The state type is `internal`: a fixed-size struct ([`state::AggState`],
//! [`state::MovState`]) palloc'd once per group in the aggregate memory
//! context and updated in place. 0.1.2 used a `float8[5]` varlena that was
//! decoded, re-allocated and re-encoded on every input row. Parallel
//! plans cross process boundaries through SERIALFUNC/DESERIALFUNC
//! (33 bytes per partial state).
//!
//! The support functions are raw V1 C-ABI functions (`#[pg_guard]`,
//! declared to SQL by hand in `extension_sql!` below and in the upgrade
//! script) so that the per-row path is: read four `NullableDatum`s,
//! compare `eps`, call `ScalarAcc::absorb`. No pgrx argument unboxing, no
//! per-row allocation.
//!
//! Semantics are 0.1.2's, unchanged (README.md "Special float values"):
//! NULL `value`/`score` rows are skipped; NaN in either propagates; ±Inf
//! scores follow the core's policy; `eps` must be >= 0, non-NULL and
//! constant within a group.
//!
//! Moving aggregate: PostgreSQL uses MSFUNC/MINVFUNC when a window frame's
//! head moves (`ROWS BETWEEN k PRECEDING AND CURRENT ROW`). MINVFUNC
//! removes the row leaving the frame via `InvAcc::remove`; when the core
//! reports the state inexact (cancellation guard `z <= 1e-6 z_peak`, or
//! the argmax set of a `eps = 0` / `+inf` frame emptied) MINVFUNC returns
//! SQL NULL, which is PostgreSQL's documented "cannot invert" signal: the
//! executor discards the state and re-aggregates the current frame.

use pgrx::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub mod state;
/// relF baseline (experiments/anchored); raw V1 functions outside the extension catalog
pub mod relf;
use state::{check_eps, AggState, Inverse, MovState, Reject};

::pgrx::pg_module_magic!(name, version);

// ---------------------------------------------------------------------
// raw V1 plumbing
// ---------------------------------------------------------------------

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

finfo_v1!(
    pg_finfo_softavg_sfunc,
    pg_finfo_softavg_combine,
    pg_finfo_softavg_serialize,
    pg_finfo_softavg_deserialize,
    pg_finfo_softavg_finalfn,
    pg_finfo_softavg_state_finalfn,
    pg_finfo_softavg_msfunc,
    pg_finfo_softavg_minvfunc,
    pg_finfo_softavg_mfinalfn,
    pg_finfo_softavg_inv_build_final,
);

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

#[inline(always)]
unsafe fn ret_f64(x: f64) -> pg_sys::Datum {
    pg_sys::Datum::from(x.to_bits())
}

/// The aggregate (or window-aggregate) memory context; states live there.
#[cold]
unsafe fn agg_context(fcinfo: pg_sys::FunctionCallInfo, who: &str) -> pg_sys::MemoryContext {
    let mut ctx: pg_sys::MemoryContext = std::ptr::null_mut();
    if pg_sys::AggCheckCallContext(fcinfo, &mut ctx) == 0 || ctx.is_null() {
        error!("{who} called in non-aggregate context");
    }
    ctx
}

#[cold]
#[inline(never)]
fn raise(r: Reject) -> ! {
    error!("{}", r.message())
}

#[cold]
unsafe fn alloc_in<T>(ctx: pg_sys::MemoryContext, v: T) -> *mut T {
    let p = pg_sys::MemoryContextAlloc(ctx, std::mem::size_of::<T>()) as *mut T;
    p.write(v);
    p
}

// ---------------------------------------------------------------------
// plain aggregate: softavg / softavg_state
// ---------------------------------------------------------------------

/// SFUNC(internal, value, score, eps) -> internal. Non-strict. The state
/// stays SQL NULL until the group's first live row (so an all-NULL group
/// finalizes to NULL), then is allocated once in the aggregate context.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_sfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<AggState>(fcinfo, 0);
    let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) else {
        // NULL value or score: skip the row, keep the state as it is
        return if st.is_null() {
            ret_null(fcinfo)
        } else {
            pg_sys::Datum::from(st)
        };
    };
    let eps = match check_eps(arg_f64(fcinfo, 3)) {
        Ok(e) => e,
        Err(r) => raise(r),
    };
    let st = if st.is_null() {
        alloc_in(agg_context(fcinfo, "softavg_sfunc"), AggState::new(eps))
    } else {
        st
    };
    if let Err(r) = (*st).absorb(s, v, eps) {
        raise(r);
    }
    pg_sys::Datum::from(st)
}

/// COMBINEFUNC(internal, internal) -> internal. Non-strict: when the
/// left side is NULL the right side is copied into the aggregate context
/// (it may live in a per-tuple context, e.g. straight out of
/// DESERIALFUNC).
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_combine(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = arg_ptr::<AggState>(fcinfo, 0);
    let b = arg_ptr::<AggState>(fcinfo, 1);
    match (a.is_null(), b.is_null()) {
        (true, true) => ret_null(fcinfo),
        (false, true) => pg_sys::Datum::from(a),
        (true, false) => pg_sys::Datum::from(alloc_in(agg_context(fcinfo, "softavg_combine"), *b)),
        (false, false) => {
            if let Err(r) = (*a).merge(&*b) {
                raise(r);
            }
            pg_sys::Datum::from(a)
        }
    }
}

/// SERIALFUNC(internal) -> bytea. Strict.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_serialize(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<AggState>(fcinfo, 0);
    let bytes = (*st).serialize();
    let total = pg_sys::VARHDRSZ + bytes.len();
    let p = pg_sys::palloc(total) as *mut pg_sys::varlena;
    pgrx::set_varsize_4b(p, total as i32);
    std::ptr::copy_nonoverlapping(
        bytes.as_ptr(),
        (p as *mut u8).add(pg_sys::VARHDRSZ),
        bytes.len(),
    );
    pg_sys::Datum::from(p)
}

/// DESERIALFUNC(bytea, internal) -> internal. Strict. Allocates in the
/// current (per-tuple) context; COMBINEFUNC copies when it adopts.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_deserialize(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = arg_ptr::<pg_sys::varlena>(fcinfo, 0);
    let v = pg_sys::pg_detoast_datum_packed(raw);
    let len = pgrx::varsize_any_exhdr(v);
    let data = std::slice::from_raw_parts(pgrx::vardata_any(v) as *const u8, len);
    let Some(st) = AggState::deserialize(data) else {
        error!("softavg: malformed serialized state ({len} bytes)");
    };
    let p = pg_sys::palloc(std::mem::size_of::<AggState>()) as *mut AggState;
    p.write(st);
    pg_sys::Datum::from(p)
}

/// FINALFUNC(internal) -> float8. Read-only on the state.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_finalfn(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<AggState>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    match (*st).finalize() {
        Some(x) => ret_f64(x),
        None => ret_null(fcinfo),
    }
}

/// softavg_state's FINALFUNC(internal) -> float8[5]: the 0.1.2 state
/// array `[mu, z, u, eps, nan_seen]`, so SQL can still re-associate
/// partial results with `bruce_softavg_combine` / `bruce_softavg_final`.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_state_finalfn(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let st = arg_ptr::<AggState>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    match (*st).to_array().to_vec().into_datum() {
        Some(d) => d,
        None => ret_null(fcinfo),
    }
}

// ---------------------------------------------------------------------
// moving aggregate (window frames whose head moves)
// ---------------------------------------------------------------------

/// Per-backend counters for the moving-aggregate path, exposed by
/// `softavg_window_stats()`. Plain load/store (a backend is
/// single-threaded), so the cost is one increment per call.
static MOV_ABSORB: AtomicU64 = AtomicU64::new(0);
static MOV_INVERSE: AtomicU64 = AtomicU64::new(0);
static MOV_PUNT: AtomicU64 = AtomicU64::new(0);
static MOV_FRESH: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
fn bump(c: &AtomicU64) {
    c.store(c.load(Relaxed).wrapping_add(1), Relaxed);
}

/// MSFUNC(internal, value, score, eps) -> internal. Never returns NULL
/// (PostgreSQL forbids it for moving aggregates), so the state is
/// allocated on the first call even if that row is NULL.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_msfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let mut st = arg_ptr::<MovState>(fcinfo, 0);
    if st.is_null() {
        bump(&MOV_FRESH);
        st = alloc_in(agg_context(fcinfo, "softavg_msfunc"), MovState::new());
    }
    bump(&MOV_ABSORB);
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        let eps = match check_eps(arg_f64(fcinfo, 3)) {
            Ok(e) => e,
            Err(r) => raise(r),
        };
        if let Err(r) = (*st).absorb(s, v, eps) {
            raise(r);
        }
    }
    pg_sys::Datum::from(st)
}

/// MINVFUNC(internal, value, score, eps) -> internal, or SQL NULL when
/// the core says the remaining state can no longer be trusted (the
/// executor then recomputes the frame).
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_minvfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<MovState>(fcinfo, 0);
    if st.is_null() {
        // PostgreSQL never does this (it errors before calling); be safe
        bump(&MOV_PUNT);
        return ret_null(fcinfo);
    }
    bump(&MOV_INVERSE);
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        let eps = match check_eps(arg_f64(fcinfo, 3)) {
            Ok(e) => e,
            Err(r) => raise(r),
        };
        match (*st).remove(s, v, eps) {
            Ok(Inverse::Done) => {}
            Ok(Inverse::Punt) => {
                bump(&MOV_PUNT);
                return ret_null(fcinfo);
            }
            Err(r) => raise(r),
        }
    }
    pg_sys::Datum::from(st)
}

/// MFINALFUNC(internal) -> float8. Read-only on the state.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_mfinalfn(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<MovState>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    if !(*st).inv.is_exact() {
        // unreachable through PostgreSQL: MINVFUNC punts first
        error!("softavg: moving-aggregate state is inexact at finalize");
    }
    match (*st).finalize() {
        Some(x) => ret_f64(x),
        None => ret_null(fcinfo),
    }
}

/// Moving-aggregate counters of this backend since the last reset:
/// `fresh_states` (frame (re)starts, including after a punt),
/// `absorbs` (MSFUNC calls), `inverses` (MINVFUNC calls), `punts`
/// (MINVFUNC returned NULL: the core's guard fired, frame recomputed).
#[pg_extern(volatile, parallel_restricted)]
fn softavg_window_stats() -> TableIterator<
    'static,
    (
        name!(fresh_states, i64),
        name!(absorbs, i64),
        name!(inverses, i64),
        name!(punts, i64),
    ),
> {
    TableIterator::once((
        MOV_FRESH.load(Relaxed) as i64,
        MOV_ABSORB.load(Relaxed) as i64,
        MOV_INVERSE.load(Relaxed) as i64,
        MOV_PUNT.load(Relaxed) as i64,
    ))
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_window_stats_reset() {
    for c in [&MOV_FRESH, &MOV_ABSORB, &MOV_INVERSE, &MOV_PUNT] {
        c.store(0, Relaxed);
    }
}

/// softagg-core version this library was built against.
#[pg_extern(immutable, parallel_safe)]
fn softavg_core_version() -> &'static str {
    SOFTAGG_CORE_VERSION
}
const SOFTAGG_CORE_VERSION: &str = "softagg-core 0.1.0 (path ../core)";

// ---------------------------------------------------------------------
// 0.1.2 SQL-callable state-array helpers (float8[5]), kept: users
// re-associate softavg_state outputs explicitly in SQL with them. Same
// code paths as 0.1.2, now over softagg-core.
// ---------------------------------------------------------------------

fn decode(st: &[f64]) -> AggState {
    state::decode_checked(st).unwrap_or_else(|| {
        error!(
            "softavg: malformed state — expected float8[{}], got {} elements",
            state::STATE_LEN,
            st.len()
        )
    })
}

/// 0.1.2's float8[]-state transition, kept as a callable function (and
/// as the "before" arm of the state-representation experiment).
#[pg_extern(immutable, parallel_safe)]
fn bruce_softavg_sfunc(
    state: Option<Vec<f64>>,
    value: Option<f64>,
    score: Option<f64>,
    eps: Option<f64>,
) -> Option<Vec<f64>> {
    let (Some(v), Some(s)) = (value, score) else {
        return state;
    };
    let eps = check_eps(eps).unwrap_or_else(|r| raise(r));
    let mut acc = match state {
        None => AggState::new(eps),
        Some(st) => decode(&st),
    };
    acc.absorb(s, v, eps).unwrap_or_else(|r| raise(r));
    Some(acc.to_array().to_vec())
}

#[pg_extern(immutable, parallel_safe)]
fn bruce_softavg_combine(a: Option<Vec<f64>>, b: Option<Vec<f64>>) -> Option<Vec<f64>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(x), Some(y)) => {
            let mut acc_a = decode(&x);
            acc_a.merge(&decode(&y)).unwrap_or_else(|r| raise(r));
            Some(acc_a.to_array().to_vec())
        }
    }
}

#[pg_extern(immutable, parallel_safe)]
fn bruce_softavg_final(state: Option<Vec<f64>>) -> Option<f64> {
    decode(&state?).finalize()
}

// ---------------------------------------------------------------------
// Incremental maintenance of a grouped softavg (softavg_ivm_create):
// per-group InvAcc states (serialized MovState, bytea) in an ordinary
// table, maintained by statement-level triggers over transition tables.
// Because the state is a table row, every change is transactional
// (rolls back with the statement / transaction that caused it).
// ---------------------------------------------------------------------

static IVM_STATEMENTS: AtomicU64 = AtomicU64::new(0);
static IVM_RECOMPUTED: AtomicU64 = AtomicU64::new(0);

fn mov_or_new(b: Option<&[u8]>) -> MovState {
    match b {
        None => MovState::new(),
        Some(b) => MovState::deserialize(b)
            .unwrap_or_else(|| error!("softavg: malformed incremental state ({} bytes)", b.len())),
    }
}

/// softavg_inv_build's FINALFUNC(internal) -> bytea: the serialized
/// moving state (its SFUNC is softavg_msfunc).
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn softavg_inv_build_final(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let st = arg_ptr::<MovState>(fcinfo, 0);
    let bytes = if st.is_null() { MovState::new() } else { *st }.serialize();
    let total = pg_sys::VARHDRSZ + bytes.len();
    let p = pg_sys::palloc(total) as *mut pg_sys::varlena;
    pgrx::set_varsize_4b(p, total as i32);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), (p as *mut u8).add(pg_sys::VARHDRSZ), bytes.len());
    pg_sys::Datum::from(p)
}

/// Merge two incremental states (an insert delta into a stored state).
#[pg_extern(immutable, parallel_safe)]
fn softavg_inv_merge(a: Option<&[u8]>, b: Option<&[u8]>) -> Option<Vec<u8>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(mov_or_new(Some(x)).serialize().to_vec()),
        (Some(x), Some(y)) => {
            let mut m = mov_or_new(Some(x));
            m.merge(&mov_or_new(Some(y))).unwrap_or_else(|r| raise(r));
            Some(m.serialize().to_vec())
        }
    }
}

/// Remove rows (parallel arrays; NULL value/score entries are skipped,
/// as the aggregate skips them). The result may be inexact: check
/// softavg_inv_exact and recompute the group from the base table.
#[pg_extern(immutable, parallel_safe)]
fn softavg_inv_remove_many(
    st: Option<&[u8]>,
    vs: Vec<Option<f64>>,
    ss: Vec<Option<f64>>,
    eps: f64,
) -> Vec<u8> {
    if vs.len() != ss.len() {
        error!("softavg_inv_remove_many: value and score arrays differ in length");
    }
    let eps = check_eps(Some(eps)).unwrap_or_else(|r| raise(r));
    let mut m = mov_or_new(st);
    for (v, s) in vs.into_iter().zip(ss) {
        if let (Some(v), Some(s)) = (v, s) {
            match m.remove(s, v, eps) {
                Ok(Inverse::Done) => {}
                Ok(Inverse::Punt) => m.inv.inexact = true,
                Err(r) => raise(r),
            }
        }
    }
    m.serialize().to_vec()
}

/// The group's softavg (NaN if a NaN row is live, NULL if no weighted row).
#[pg_extern(immutable, parallel_safe, strict)]
fn softavg_inv_value(st: &[u8]) -> Option<f64> {
    let m = mov_or_new(Some(st));
    if !m.inv.is_exact() {
        error!("softavg: incremental state is inexact; it must be recomputed before it is read");
    }
    m.finalize()
}

#[pg_extern(immutable, parallel_safe, strict)]
fn softavg_inv_exact(st: &[u8]) -> bool {
    mov_or_new(Some(st)).inv.is_exact()
}

/// softagg-core's bound on |delivered - exact| (absolute, value units).
#[pg_extern(immutable, parallel_safe, strict)]
fn softavg_inv_error_bound(st: &[u8]) -> f64 {
    mov_or_new(Some(st)).inv.value_error_bound()
}

#[pg_extern(immutable, parallel_safe, strict)]
fn softavg_inv_rows(st: &[u8]) -> i64 {
    mov_or_new(Some(st)).rows() as i64
}

/// Called by the generated triggers; per-backend counters.
#[pg_extern(volatile, parallel_restricted)]
fn softavg_ivm_note(recomputed_groups: i64) {
    bump(&IVM_STATEMENTS);
    IVM_RECOMPUTED.store(
        IVM_RECOMPUTED.load(Relaxed) + recomputed_groups.max(0) as u64,
        Relaxed,
    );
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_ivm_stats(
) -> TableIterator<'static, (name!(trigger_statements, i64), name!(groups_recomputed, i64))> {
    TableIterator::once((
        IVM_STATEMENTS.load(Relaxed) as i64,
        IVM_RECOMPUTED.load(Relaxed) as i64,
    ))
}

#[pg_extern(volatile, parallel_restricted)]
fn softavg_ivm_stats_reset() {
    IVM_STATEMENTS.store(0, Relaxed);
    IVM_RECOMPUTED.store(0, Relaxed);
}

// The SQL below is duplicated verbatim in sql/bruce_pg--0.1.2--0.2.0.sql
// (the upgrade must produce the same catalog as a fresh install; the
// harness experiments/postgres/upgrade_check.py compares the two).
extension_sql!(
    r#"
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
CREATE AGGREGATE softavg(float8, float8, float8) (
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
CREATE AGGREGATE softavg_state(float8, float8, float8) (
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
"#,
    name = "softavg_aggregates",
    requires = [softavg_inv_merge, softavg_inv_remove_many, softavg_inv_exact, softavg_ivm_note],
);

// ---------------------------------------------------------------------
// SQL tests through the extension (cargo pgrx test pg17).
// ---------------------------------------------------------------------
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    fn get_f64(query: &str) -> f64 {
        Spi::get_one::<f64>(query)
            .expect("SPI failed")
            .expect("NULL result")
    }

    /// (a) all scores equal -> softavg == AVG at every temperature
    /// (eps = 0: the argmax set is the whole group).
    #[pg_test]
    fn test_a_equal_scores_match_avg() {
        Spi::run(
            "CREATE TABLE ta AS SELECT (i::float8) * 0.37 - 5.0 AS v \
             FROM generate_series(1, 100) g(i)",
        )
        .unwrap();
        let want = get_f64("SELECT AVG(v) FROM ta");
        for eps in ["0.0", "0.5", "1.0", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, 0.7, {eps}) FROM ta"));
            assert!(
                (got - want).abs() < 1e-12,
                "eps={eps}: softavg {got} != avg {want}"
            );
        }
    }

    /// (b) finite eps matches the naive SQL softmax average on a small
    /// fixture whose scores are tame enough not to overflow.
    #[pg_test]
    fn test_b_matches_sql_softmax_average() {
        Spi::run(
            "CREATE TABLE tb(v float8, s float8); \
             INSERT INTO tb VALUES (3.0, -2.0), (-1.5, -0.7), (0.25, 0.0), \
                                   (7.0, 0.4), (-4.0, 1.1), (2.5, 1.9), (9.0, -1.3)",
        )
        .unwrap();
        for (eps, ref_expr) in [
            ("1.0", "SUM(EXP(s) * v) / SUM(EXP(s))"),
            ("0.5", "SUM(EXP(s / 0.5) * v) / SUM(EXP(s / 0.5))"),
        ] {
            let want = get_f64(&format!("SELECT {ref_expr} FROM tb"));
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tb"));
            assert!(
                (got - want).abs() < 1e-9,
                "eps={eps}: softavg {got} != sql softmax {want}"
            );
        }
    }

    fn make_sharp_fixture() {
        // the +1.0-score row goes first so the naive spelling hits
        // EXP(10000) -> overflow on the very first row it evaluates
        Spi::run(
            "CREATE TABLE tc(v float8, s float8); \
             INSERT INTO tc VALUES (10.0, 1.0), (20.0, -1.0), (30.0, 0.5)",
        )
        .unwrap();
    }

    /// (c) naive half: SUM(EXP(s/eps)*v)/SUM(EXP(s/eps)) at eps = 1e-4
    /// with scores in [-1, 1] overflows float8 — EXP(1/1e-4) = e^10000.
    #[pg_test(error = "value out of range: overflow")]
    fn test_c_naive_softmax_overflows() {
        make_sharp_fixture();
        Spi::get_one::<f64>("SELECT SUM(EXP(s / 0.0001) * v) / SUM(EXP(s / 0.0001)) FROM tc")
            .unwrap();
    }

    /// (c) anchored half: same query through softavg is finite and
    /// correct — at eps = 1e-4 the group collapses onto its argmax row.
    #[pg_test]
    fn test_c_softavg_survives_sharp_eps() {
        make_sharp_fixture();
        let got = get_f64("SELECT softavg(v, s, 0.0001) FROM tc");
        assert!(got.is_finite(), "softavg overflowed: {got}");
        assert!((got - 10.0).abs() < 1e-12, "sharp-eps result {got} != 10.0");
    }

    /// (d) parallel-safety: force a parallel plan over 200k rows,
    /// verify the plan really gathers partial aggregates, and compare
    /// against the single-worker result.
    #[pg_test]
    fn test_d_parallel_matches_serial() {
        Spi::run(
            "CREATE TABLE td AS SELECT sin(i::float8) * 10.0 AS v, cos(i::float8) AS s \
             FROM generate_series(1, 200000) g(i)",
        )
        .unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
        let serial = get_f64("SELECT softavg(v, s, 0.25) FROM td");

        Spi::run("SET parallel_setup_cost = 0").unwrap();
        Spi::run("SET parallel_tuple_cost = 0").unwrap();
        Spi::run("SET min_parallel_table_scan_size = 0").unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 4").unwrap();
        let plan = explain("SELECT softavg(v, s, 0.25) FROM td");
        assert!(
            plan.contains("Gather") && plan.contains("Partial Aggregate"),
            "plan did not parallelize:\n{plan}"
        );
        let parallel = get_f64("SELECT softavg(v, s, 0.25) FROM td");
        assert!(
            (parallel - serial).abs() <= 1e-9 * serial.abs().max(1.0),
            "parallel {parallel} != serial {serial}"
        );
    }

    /// (e) the monoid identity in SQL: combine(state(A), state(B))
    /// finalized == softavg(A ∪ B), for an arbitrary partition.
    #[pg_test]
    fn test_e_partition_combine_identity() {
        Spi::run(
            "CREATE TABLE te AS SELECT i, sin(i::float8) * 3.0 AS v, cos(i::float8) AS s \
             FROM generate_series(1, 1000) g(i)",
        )
        .unwrap();
        let whole = get_f64("SELECT softavg(v, s, 0.25) FROM te");
        let recombined = get_f64(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
                 (SELECT softavg_state(v, s, 0.25) FROM te WHERE i % 2 = 0), \
                 (SELECT softavg_state(v, s, 0.25) FROM te WHERE i % 2 = 1)))",
        );
        assert!(
            (whole - recombined).abs() < 1e-12,
            "partition-combine {recombined} != whole {whole}"
        );
    }

    /// NULL discipline: NULL value/score rows are ignored like AVG's;
    /// an all-NULL (or empty) group yields NULL.
    #[pg_test]
    fn test_null_handling_matches_avg() {
        Spi::run(
            "CREATE TABLE tn(v float8, s float8); \
             INSERT INTO tn VALUES (1.0, 0.2), (NULL, 0.9), (3.0, NULL), (5.0, -0.1)",
        )
        .unwrap();
        let with_nulls = get_f64("SELECT softavg(v, s, 0.5) FROM tn");
        let without =
            get_f64("SELECT softavg(v, s, 0.5) FROM tn WHERE v IS NOT NULL AND s IS NOT NULL");
        assert!((with_nulls - without).abs() < 1e-15);
        let empty = Spi::get_one::<f64>("SELECT softavg(v, s, 0.5) FROM tn WHERE false").unwrap();
        assert!(empty.is_none(), "empty group must be NULL");
    }

    /// eps = 0 endpoint: mean over the argmax set (uniform ties).
    #[pg_test]
    fn test_eps_zero_argmax_mean() {
        Spi::run(
            "CREATE TABLE tz(v float8, s float8); \
             INSERT INTO tz VALUES (2.0, 1.0), (4.0, 1.0), (100.0, 0.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 0.0) FROM tz");
        assert!((got - 3.0).abs() < 1e-15, "tropical mean {got} != 3.0");
    }

    /// eps = infinity endpoint: plain mean, scores ignored.
    #[pg_test]
    fn test_eps_inf_plain_mean() {
        Spi::run(
            "CREATE TABLE ti(v float8, s float8); \
             INSERT INTO ti VALUES (1.0, 5.0), (2.0, -3.0), (6.0, 0.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 'infinity'::float8) FROM ti");
        assert!((got - 3.0).abs() < 1e-15, "uniform mean {got} != 3.0");
    }

    /// eps must be constant within a group (fixture rows carry
    /// e = 0.6, 0.7, 0.8 — the transition trips on the second row).
    #[pg_test(error = "softavg: eps must be constant within an aggregate group (0.6 vs 0.7)")]
    fn test_eps_change_rejected() {
        Spi::run(
            "CREATE TABLE tv AS SELECT i::float8 AS v, 0.0::float8 AS s, \
             (0.5 + i * 0.1)::float8 AS e FROM generate_series(1, 3) g(i)",
        )
        .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, e) FROM tv").unwrap();
    }

    // -----------------------------------------------------------------
    // Semantics conformance (TESTING_MATRIX workstream 14): softavg's
    // NULL / empty / strictness / grouping / eps-domain behavior pinned
    // against what PG's own AVG promises (constitution C4).
    // -----------------------------------------------------------------

    fn get_i64(query: &str) -> i64 {
        Spi::get_one::<i64>(query)
            .expect("SPI failed")
            .expect("NULL result")
    }

    /// (14a) NULL discipline == AVG's skip, proven against AVG itself:
    /// with equal scores softavg degenerates to a mean at every eps, so
    /// softavg over the NULL-y table must equal AVG over the subset
    /// where BOTH aggregate inputs are non-NULL. (AVG(v) alone is NOT
    /// the right-hand side: it skips NULL v but keeps rows whose score
    /// is NULL, which softavg must also drop.)
    #[pg_test]
    fn test_f_null_skip_equals_avg_on_nonnull_subset() {
        Spi::run(
            "CREATE TABLE tf(v float8, s float8); \
             INSERT INTO tf VALUES \
                 (1.0, 0.7), (NULL, 0.7), (3.0, NULL), (5.0, 0.7), \
                 (NULL, NULL), (9.0, 0.7)",
        )
        .unwrap();
        let want = get_f64("SELECT AVG(v) FROM tf WHERE v IS NOT NULL AND s IS NOT NULL");
        assert!((want - 5.0).abs() < 1e-15, "fixture: avg of {{1,5,9}} is 5");
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tf"));
            assert!(
                (got - want).abs() < 1e-12,
                "eps={eps}: softavg {got} != AVG-on-non-null-subset {want}"
            );
        }
    }

    /// (14b) empty input -> NULL (not 0, not an error), exactly AVG's
    /// contract — asserted for a WHERE-false scan, a truly empty table,
    /// and an all-NULL-input table, side by side with AVG. One
    /// deliberate asymmetry is pinned at the end: AVG's qualifying
    /// input is the single column `v`, softavg's is the complete
    /// `(v, s)` pair — a table holding a value but no score is "empty"
    /// to softavg even though AVG(v) still averages it.
    #[pg_test]
    fn test_f_empty_input_is_null_like_avg() {
        Spi::run(
            "CREATE TABLE tfe(v float8, s float8); \
             CREATE TABLE tfn(v float8, s float8); \
             INSERT INTO tfn VALUES (NULL, 0.5), (NULL, NULL)",
        )
        .unwrap();
        for from in ["tfe", "tfn", "tfe WHERE false"] {
            let avg = Spi::get_one::<f64>(&format!("SELECT AVG(v) FROM {from}")).unwrap();
            let soft =
                Spi::get_one::<f64>(&format!("SELECT softavg(v, s, 0.5) FROM {from}")).unwrap();
            assert!(avg.is_none(), "AVG over {from} must be NULL");
            assert!(
                soft.is_none(),
                "softavg over {from} must be NULL like AVG, got {soft:?}"
            );
        }
        // ...and it is one NULL row, not zero rows: aggregates over an
        // empty input still produce a single row.
        assert_eq!(
            get_i64("SELECT COUNT(*) FROM (SELECT softavg(v, s, 0.5) FROM tfe) q"),
            1
        );
        // the pair rule: value present, score NULL on every row ->
        // no qualifying (v, s) pair -> NULL, even though AVG(v) = 1.0.
        Spi::run("CREATE TABLE tfp(v float8, s float8); INSERT INTO tfp VALUES (1.0, NULL)")
            .unwrap();
        assert_eq!(get_f64("SELECT AVG(v) FROM tfp"), 1.0);
        let soft = Spi::get_one::<f64>("SELECT softavg(v, s, 0.5) FROM tfp").unwrap();
        assert!(
            soft.is_none(),
            "no complete (v, s) pair -> NULL, got {soft:?}"
        );
    }

    /// (14c) strictness declaration audit. The transition function MUST
    /// be declared non-strict: PG would reject a strict multi-argument
    /// SFUNC with a NULL initcond (no way to bootstrap the state), and a
    /// strict SFUNC would also skip rows where only `eps` is NULL
    /// instead of raising. So the declaration promises "callee handles
    /// NULLs" — the companion tests (test_f_*, test_null_eps_errors)
    /// prove the callee keeps that promise (skip NULL inputs, error on
    /// NULL eps). Combine/final are equally non-strict and NULL-total.
    #[pg_test]
    fn test_f_strictness_declaration_audit() {
        for agg in ["softavg", "softavg_state"] {
            let strict_sfunc = Spi::get_one::<bool>(&format!(
                "SELECT p.proisstrict FROM pg_aggregate a \
                 JOIN pg_proc p ON p.oid = a.aggtransfn \
                 WHERE a.aggfnoid = '{agg}(float8, float8, float8)'::regprocedure"
            ))
            .unwrap()
            .expect("aggregate not found in pg_aggregate");
            assert!(
                !strict_sfunc,
                "{agg}: SFUNC must be non-strict (handles NULLs itself)"
            );

            let strict_combine = Spi::get_one::<bool>(&format!(
                "SELECT p.proisstrict FROM pg_aggregate a \
                 JOIN pg_proc p ON p.oid = a.aggcombinefn \
                 WHERE a.aggfnoid = '{agg}(float8, float8, float8)'::regprocedure"
            ))
            .unwrap()
            .expect("combinefn not declared");
            assert!(!strict_combine, "{agg}: COMBINEFUNC must be non-strict");
        }
        let strict_final = Spi::get_one::<bool>(
            "SELECT p.proisstrict FROM pg_aggregate a \
             JOIN pg_proc p ON p.oid = a.aggfinalfn \
             WHERE a.aggfnoid = 'softavg(float8, float8, float8)'::regprocedure",
        )
        .unwrap()
        .expect("finalfn not declared");
        assert!(
            !strict_final,
            "softavg: FINALFUNC is non-strict (maps NULL state -> NULL)"
        );
        // sanity: AVG(float8) makes the same declaration choice PG-side
        // reference — its transfn float8_accum IS strict (single-arg,
        // state bootstrap works); the audit above documents WHY softavg
        // legitimately differs (3 args + NULL initcond).
    }

    /// (14c) behavior half of the audit: a NULL `eps` reaching a live
    /// row is an error — the non-strict SFUNC must not silently invent
    /// a temperature. (NULL value/score rows are skipped before the eps
    /// check, matching the AVG-skip discipline.)
    #[pg_test(error = "softavg: eps must not be NULL")]
    fn test_f_null_eps_on_live_row_errors() {
        Spi::run("CREATE TABLE tne(v float8, s float8); INSERT INTO tne VALUES (1.0, 0.5)")
            .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, NULL::float8) FROM tne").unwrap();
    }

    /// (14d) grouping-text parity: softavg does not group — PG does.
    /// GROUP BY over unicode text must behave exactly as it does for
    /// AVG (constitution C4): equal strings collapse ('猫' twice),
    /// canonically-equivalent-but-byte-different strings stay separate
    /// (NFC 'café' vs NFD 'cafe' + U+0301), and per-group results match
    /// AVG under equal scores.
    #[pg_test]
    fn test_g_text_group_by_unicode_parity() {
        Spi::run(
            "CREATE TABLE tg(k text, v float8, s float8); \
             INSERT INTO tg VALUES \
                 (E'caf\\u00e9', 1.0, 0.3), (E'caf\\u00e9', 3.0, 0.3), \
                 (E'cafe\\u0301', 100.0, 0.3), \
                 (E'\\u732b', 10.0, 0.3), (E'\\u732b', 30.0, 0.3), \
                 ('Stra\u{00df}e', 7.0, 0.3)",
        )
        .unwrap();
        // PG's equality decides the groups: 4 distinct keys.
        assert_eq!(
            get_i64("SELECT COUNT(*) FROM (SELECT k FROM tg GROUP BY k) q"),
            4,
            "NFC/NFD must be distinct groups; equal strings must collapse"
        );
        // per-group softavg == per-group AVG (equal scores) — grouping
        // applied to softavg is the same grouping applied to AVG.
        assert_eq!(
            get_i64(
                "SELECT COUNT(*) FROM ( \
                     SELECT k, softavg(v, s, 0.7) AS sa, AVG(v) AS av \
                     FROM tg GROUP BY k) q \
                 WHERE q.sa IS DISTINCT FROM q.av OR abs(q.sa - q.av) > 1e-12"
            ),
            0,
            "some group's softavg diverged from AVG"
        );
        // the byte-different lookalikes really landed in different groups
        let nfc = get_f64("SELECT softavg(v, s, 0.7) FROM tg WHERE k = E'caf\\u00e9'");
        let nfd = get_f64("SELECT softavg(v, s, 0.7) FROM tg WHERE k = E'cafe\\u0301'");
        assert!(
            (nfc - 2.0).abs() < 1e-12,
            "NFC group {{1,3}} -> 2.0, got {nfc}"
        );
        assert!(
            (nfd - 100.0).abs() < 1e-12,
            "NFD group {{100}} -> 100.0, got {nfd}"
        );
    }

    /// (14e) negative eps -> clean SQL ERROR (ereport, not a crash or a
    /// NaN result); same domain as bruce-core's Eps::new.
    #[pg_test(
        error = "softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got -0.5"
    )]
    fn test_h_negative_eps_is_sql_error() {
        Spi::run("CREATE TABLE tneg(v float8, s float8); INSERT INTO tneg VALUES (1.0, 0.5)")
            .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, -0.5) FROM tneg").unwrap();
    }

    /// (14e) eps = 0 with a two-way tie at the max score -> the mean of
    /// exactly the tied rows; the raw state confirms the argmax set has
    /// multiplicity 2 (z counts ties in the tropical regime).
    #[pg_test]
    fn test_h_eps_zero_two_way_tie_is_tied_mean() {
        Spi::run(
            "CREATE TABLE tt(v float8, s float8); \
             INSERT INTO tt VALUES (10.0, 5.0), (999.0, 4.9), (20.0, 5.0), (-7.0, -5.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 0.0) FROM tt");
        assert!((got - 15.0).abs() < 1e-15, "tie mean (10+20)/2, got {got}");
        // state [mu, z, u, eps, nan]: mu = max score, z = tie count, u = tied value sum
        let (mu, z, u) = Spi::get_three::<f64, f64, f64>(
            "SELECT st[1], st[2], st[3] FROM (SELECT softavg_state(v, s, 0.0) AS st FROM tt) q",
        )
        .expect("SPI failed");
        assert_eq!(mu, Some(5.0));
        assert_eq!(
            z,
            Some(2.0),
            "argmax multiplicity must count both tied rows"
        );
        assert_eq!(u, Some(30.0));
    }

    // -----------------------------------------------------------------
    // Special float values (pg-parity track). ±Inf mirrors bruce-core
    // verbatim; NaN deliberately diverges (PROPAGATE, not skip) — the
    // decision and its cost are argued in README.md, "Special float
    // values: NaN, ±Inf, and one deliberate divergence".
    // -----------------------------------------------------------------

    /// Assert a scalar query returns SQL NULL.
    fn assert_sql_null(query: &str, why: &str) {
        let got = Spi::get_one::<f64>(query).expect("SPI failed");
        assert!(got.is_none(), "{why}: expected NULL, got {got:?}");
    }

    /// (j) NaN `value` propagates in every regime — `softavg` joins the
    /// AVG/SUM family rather than silently dropping a real float8.
    #[pg_test]
    fn test_j_nan_value_propagates() {
        Spi::run(
            "CREATE TABLE tjv(v float8, s float8); \
             INSERT INTO tjv VALUES (1.0, 0.2), ('NaN', 0.9), (5.0, -0.1)",
        )
        .unwrap();
        // the PG reference: AVG over the same column is NaN, not 3.0
        assert!(
            get_f64("SELECT AVG(v) FROM tjv").is_nan(),
            "PG's own AVG must be NaN here"
        );
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjv"));
            assert!(
                got.is_nan(),
                "eps={eps}: NaN value must propagate, got {got}"
            );
        }
    }

    /// (j) NaN `score` propagates too — including at
    /// `eps = 'infinity'`, where the score is arithmetically unused.
    /// That follows this crate's established PAIR rule (a NULL score
    /// already drops its row at every eps; see
    /// `test_f_empty_input_is_null_like_avg`), deliberately in
    /// preference to PG's `regr_avgy`, which ignores a NaN in the
    /// argument it never reads. One rule, all three regimes.
    #[pg_test]
    fn test_j_nan_score_propagates_in_every_regime() {
        Spi::run(
            "CREATE TABLE tjs(v float8, s float8); \
             INSERT INTO tjs VALUES (1.0, 0.2), (2.0, 'NaN'), (5.0, -0.1)",
        )
        .unwrap();
        // AVG(v) alone is finite: the divergence is real and intended.
        assert!((get_f64("SELECT AVG(v) FROM tjs") - 8.0 / 3.0).abs() < 1e-12);
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjs"));
            assert!(
                got.is_nan(),
                "eps={eps}: NaN score must propagate, got {got}"
            );
        }
    }

    /// (j) The divergence from bruce-core, pinned explicitly and named:
    /// bruce-core SKIPS NaN (there NaN encodes SQL NULL — see
    /// README.md); bruce_pg PROPAGATES. The skip answer is exactly what
    /// the user gets back by filtering NaN out themselves, which is the
    /// argument for propagating by default: `WHERE x <> 'NaN'::float8`
    /// (PG treats NaN = NaN as true, so this really does filter NaN)
    /// recovers bruce-core's semantics, while nothing recovers a value
    /// a skipping aggregate already discarded.
    #[pg_test]
    fn test_j_nan_propagates_where_bruce_core_skips() {
        Spi::run(
            "CREATE TABLE tjd(v float8, s float8); \
             INSERT INTO tjd VALUES (10.0, 1.0), (777.0, 'NaN'), (20.0, 2.0), \
                                    ('NaN', 0.5), (30.0, 3.0)",
        )
        .unwrap();
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let propagated = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjd"));
            assert!(propagated.is_nan(), "eps={eps}: bruce_pg must propagate");
            // bruce-core's semantics, reconstructed by the user in SQL
            let skipped = get_f64(&format!(
                "SELECT softavg(v, s, {eps}) FROM tjd \
                 WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8"
            ));
            assert!(
                skipped.is_finite(),
                "eps={eps}: skip answer must be finite, got {skipped}"
            );
        }
        // the skip answer at eps = 0 is the argmax row's value...
        let skipped0 = get_f64(
            "SELECT softavg(v, s, 0.0) FROM tjd WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8",
        );
        assert!(
            (skipped0 - 30.0).abs() < 1e-15,
            "argmax over {{1,2,3}} -> 30, got {skipped0}"
        );
        // ...and at eps = inf the mean of the three surviving values.
        let skipped_inf = get_f64(
            "SELECT softavg(v, s, 'infinity'::float8) FROM tjd \
             WHERE v <> 'NaN'::float8 AND s <> 'NaN'::float8",
        );
        assert!(
            (skipped_inf - 20.0).abs() < 1e-12,
            "mean{{10,20,30}} = 20, got {skipped_inf}"
        );
    }

    /// (j) The monoid factor of the state is UNTOUCHED by NaN rows —
    /// this is the C2 argument made observable: `[mu, z, u]` equals what
    /// bruce-core's skip produces, and only the fifth slot (the sticky
    /// NaN bit of the product monoid) records the divergence.
    #[pg_test]
    fn test_j_nan_state_component_matches_bruce_core_skip() {
        Spi::run(
            "CREATE TABLE tjst(v float8, s float8); \
             INSERT INTO tjst VALUES (10.0, 5.0), (777.0, 'NaN'), (20.0, 5.0), (-7.0, -5.0)",
        )
        .unwrap();
        let st = |col: usize, pred: &str| {
            Spi::get_one::<f64>(&format!(
                "SELECT st[{col}] FROM (SELECT softavg_state(v, s, 0.0) AS st \
                 FROM tjst {pred}) q"
            ))
            .unwrap()
            .expect("state slot NULL")
        };
        // state is float8[5] now
        let len = Spi::get_one::<i32>(
            "SELECT array_length(st, 1) FROM (SELECT softavg_state(v, s, 0.0) AS st FROM tjst) q",
        )
        .unwrap()
        .expect("no state");
        assert_eq!(len, 5, "state layout is [mu, z, u, eps, nan_seen]");
        // slots 1..3 identical with and without the NaN row present
        for col in 1..=3 {
            assert_eq!(
                st(col, ""),
                st(col, "WHERE s <> 'NaN'::float8"),
                "slot {col}: NaN row must not enter the monoid factor"
            );
        }
        assert_eq!(st(1, ""), 5.0, "mu = max score over absorbed rows");
        assert_eq!(st(2, ""), 2.0, "z = argmax multiplicity");
        assert_eq!(st(3, ""), 30.0, "u = tied value sum");
        // ...and only the sticky bit differs
        assert_eq!(st(5, ""), 1.0, "NaN bit must be set");
        assert_eq!(
            st(5, "WHERE s <> 'NaN'::float8"),
            0.0,
            "NaN bit must be clear"
        );
        // finalize disagrees even though the monoid agrees
        assert!(get_f64("SELECT softavg(v, s, 0.0) FROM tjst").is_nan());
        assert_eq!(
            get_f64("SELECT softavg(v, s, 0.0) FROM tjst WHERE s <> 'NaN'::float8"),
            15.0
        );
    }

    /// (j) An all-NaN group is NaN, NOT NULL — the rows qualified (NaN
    /// is a value); contrast `test_f_empty_input_is_null_like_avg`,
    /// where all-NULL input is NULL. AVG makes the same distinction.
    #[pg_test]
    fn test_j_all_nan_group_is_nan_not_null() {
        Spi::run(
            "CREATE TABLE tjan(v float8, s float8); \
             INSERT INTO tjan VALUES ('NaN', 0.5), ('NaN', 'NaN')",
        )
        .unwrap();
        assert!(
            get_f64("SELECT AVG(v) FROM tjan").is_nan(),
            "PG's AVG: NaN, not NULL"
        );
        for eps in ["0.0", "0.5", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjan"));
            assert!(
                got.is_nan(),
                "eps={eps}: all-NaN group must be NaN, got {got}"
            );
        }
    }

    /// (j) NaN must survive the PARALLEL path: a NaN seen by exactly one
    /// worker still reaches the final answer, because the sticky bit is
    /// OR-ed in COMBINEFUNC. Forced parallel plan, as in test_d.
    #[pg_test]
    fn test_j_nan_survives_parallel_combine() {
        Spi::run(
            "CREATE TABLE tjp AS SELECT i, sin(i::float8) * 10.0 AS v, \
             cos(i::float8) AS s FROM generate_series(1, 200000) g(i)",
        )
        .unwrap();
        // exactly one NaN, deep in the scan
        Spi::run("UPDATE tjp SET s = 'NaN' WHERE i = 137021").unwrap();

        Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
        assert!(
            get_f64("SELECT softavg(v, s, 0.25) FROM tjp").is_nan(),
            "serial"
        );

        Spi::run("SET parallel_setup_cost = 0").unwrap();
        Spi::run("SET parallel_tuple_cost = 0").unwrap();
        Spi::run("SET min_parallel_table_scan_size = 0").unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 4").unwrap();
        let plan = explain("SELECT softavg(v, s, 0.25) FROM tjp");
        assert!(
            plan.contains("Gather") && plan.contains("Partial Aggregate"),
            "plan did not parallelize:\n{plan}"
        );
        assert!(
            get_f64("SELECT softavg(v, s, 0.25) FROM tjp").is_nan(),
            "one worker's NaN must survive COMBINEFUNC"
        );
        // and it is genuinely the NaN row doing it
        assert!(
            get_f64("SELECT softavg(v, s, 0.25) FROM tjp WHERE s <> 'NaN'::float8").is_finite(),
            "without the NaN row the same parallel plan is finite"
        );
    }

    /// (j) +Inf score at finite eps: argmax collapse, uniform ties —
    /// mirrored verbatim from bruce-core (there it lives INSIDE the
    /// monoid, so there is no PG-convention conflict). Before 0.1.2
    /// these returned NaN via `exp(inf - inf)`.
    #[pg_test]
    fn test_j_pos_inf_score_collapses_to_argmax() {
        Spi::run(
            "CREATE TABLE tji(v float8, s float8); \
             INSERT INTO tji VALUES (10.0, 1.0), (55.0, 'infinity'), (20.0, 2.0)",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "1.0"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tji"));
            assert_eq!(got, 55.0, "eps={eps}: +inf row must dominate");
        }
        // ties among +inf rows average uniformly, exactly as at eps = 0
        Spi::run(
            "CREATE TABLE tjt(v float8, s float8); \
             INSERT INTO tjt VALUES (50.0, 'infinity'), (999.0, 3.0), (60.0, 'infinity')",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "1.0"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjt"));
            assert_eq!(got, 55.0, "eps={eps}: +inf tie must average uniformly");
        }
    }

    /// (j) the `+Inf` collapse must survive COMBINEFUNC. Two proofs:
    /// an explicit SQL re-association of two partial states that each
    /// hold a `+inf` row, and a forced parallel plan over 200k rows
    /// (test_d's recipe) whose `+inf` rows are spread across the scan.
    #[pg_test]
    fn test_j_pos_inf_parallel_combine_matches_serial() {
        // (1) explicit combine of two +inf-carrying partial states
        Spi::run(
            "CREATE TABLE tjc(i int, v float8, s float8); \
             INSERT INTO tjc VALUES (0, 10.0, 'infinity'), (1, 5.0, 0.5), \
                                    (2, 30.0, 'infinity'), (3, 7.0, 0.9)",
        )
        .unwrap();
        // partition so that EACH side holds exactly one +inf row: this
        // is the mu = +inf on BOTH sides branch of merge, where a naive
        // re-basing would evaluate exp(inf - inf).
        let recombined = get_f64(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i IN (0, 1)), \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i IN (2, 3))))",
        );
        assert_eq!(
            recombined, 20.0,
            "combine must keep both +inf rows (mean 10,30)"
        );
        // asymmetric case: only ONE side holds the +inf row
        let one_sided = get_f64(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i = 0), \
                 (SELECT softavg_state(v, s, 0.4) FROM tjc WHERE i IN (1, 3))))",
        );
        assert_eq!(one_sided, 10.0, "+inf side must dominate the finite side");

        // (2) real parallel plan
        Spi::run(
            "CREATE TABLE tjpi AS SELECT i, sin(i::float8) * 10.0 AS v, \
             cos(i::float8) AS s FROM generate_series(1, 200000) g(i)",
        )
        .unwrap();
        Spi::run(
            "UPDATE tjpi SET s = 'infinity', v = i / 1000.0 \
             WHERE i IN (7, 60013, 120017, 190019)",
        )
        .unwrap();
        let want = (7.0 + 60013.0 + 120017.0 + 190019.0) / 4000.0;

        Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
        let serial = get_f64("SELECT softavg(v, s, 0.25) FROM tjpi");
        assert!((serial - want).abs() < 1e-12, "serial {serial} != {want}");

        Spi::run("SET parallel_setup_cost = 0").unwrap();
        Spi::run("SET parallel_tuple_cost = 0").unwrap();
        Spi::run("SET min_parallel_table_scan_size = 0").unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 4").unwrap();
        let plan = explain("SELECT softavg(v, s, 0.25) FROM tjpi");
        assert!(
            plan.contains("Gather") && plan.contains("Partial Aggregate"),
            "plan did not parallelize:\n{plan}"
        );
        let parallel = get_f64("SELECT softavg(v, s, 0.25) FROM tjpi");
        assert!(
            (parallel - serial).abs() < 1e-12,
            "parallel {parallel} != serial {serial} (uniform mean over the +inf rows)"
        );
    }

    /// (j) -Inf carries weight 0 at eps = 0 and finite eps (the
    /// Indicator "no match" encoding), and an all--Inf group is SQL
    /// NULL — bruce-core's "uncovered" row, spelled the PG way.
    #[pg_test]
    fn test_j_neg_inf_is_weight_zero_and_all_neg_inf_is_null() {
        Spi::run(
            "CREATE TABLE tjn(v float8, s float8); \
             INSERT INTO tjn VALUES (888.0, '-infinity'), (10.0, 1.0)",
        )
        .unwrap();
        Spi::run(
            "CREATE TABLE tjnn(v float8, s float8); \
             INSERT INTO tjnn VALUES (888.0, '-infinity'), (999.0, '-infinity')",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "1.0"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjn"));
            assert_eq!(got, 10.0, "eps={eps}: -inf row must weigh 0");
            assert_sql_null(
                &format!("SELECT softavg(v, s, {eps}) FROM tjnn"),
                &format!("eps={eps}: all--inf group"),
            );
        }
        // and the empty monoid factor is a NULL, not a NaN and not a 0
        assert_eq!(
            get_i64(
                "SELECT COUNT(*) FROM (SELECT softavg(v, s, 0.5) AS x FROM tjnn) q \
                 WHERE q.x IS NULL"
            ),
            1
        );
    }

    /// (j) eps = 'infinity' is score-blind: ±Inf scores count like any
    /// other row (plain mean), so the all--Inf group is NOT null here.
    #[pg_test]
    fn test_j_eps_inf_is_score_blind_for_infinite_scores() {
        Spi::run(
            "CREATE TABLE tjb(v float8, s float8); \
             INSERT INTO tjb VALUES (3.0, 'infinity'), (6.0, '-infinity'), (9.0, 1.0)",
        )
        .unwrap();
        let got = get_f64("SELECT softavg(v, s, 'infinity'::float8) FROM tjb");
        assert_eq!(got, 6.0, "plain mean of 3, 6, 9");
        assert!(
            (got - get_f64("SELECT AVG(v) FROM tjb")).abs() < 1e-15,
            "eps = inf must still equal AVG"
        );
        // all--inf scores at eps = inf: a plain mean, not NULL
        let all_neg =
            get_f64("SELECT softavg(v, s, 'infinity'::float8) FROM tjb WHERE s = '-infinity'");
        assert_eq!(all_neg, 6.0);
    }

    /// (j) mixed NaN + ±Inf: NaN wins over everything, in every regime.
    /// (The sticky bit is consulted by FINALFUNC before the monoid
    /// factor, so a +Inf collapse or an empty all--Inf factor cannot
    /// mask a NaN that qualified.)
    #[pg_test]
    fn test_j_nan_dominates_mixed_infinities() {
        Spi::run(
            "CREATE TABLE tjm(v float8, s float8); \
             INSERT INTO tjm VALUES (5.0, 'infinity'), (1.0, '-infinity'), \
                                    (2.0, 'NaN'), (3.0, 0.5)",
        )
        .unwrap();
        // all--inf rows plus a NaN: the monoid factor is EMPTY, yet the
        // answer must be NaN rather than NULL.
        Spi::run(
            "CREATE TABLE tjm2(v float8, s float8); \
             INSERT INTO tjm2 VALUES (1.0, '-infinity'), ('NaN', '-infinity')",
        )
        .unwrap();
        for eps in ["0.0", "0.37", "'infinity'::float8"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjm"));
            assert!(got.is_nan(), "eps={eps}: NaN must dominate +inf, got {got}");
        }
        for eps in ["0.0", "0.37"] {
            let got = get_f64(&format!("SELECT softavg(v, s, {eps}) FROM tjm2"));
            assert!(
                got.is_nan(),
                "eps={eps}: NaN over an empty factor must be NaN, got {got}"
            );
        }
    }

    /// (j) a NaN row is a LIVE row, so the eps contract still applies to
    /// it — the NULL-input skip happens before the eps checks, the NaN
    /// poison after. A NaN row with a negative eps must still raise.
    #[pg_test(
        error = "softavg: eps must be >= 0 (0 = argmax-mean, 'infinity' = plain mean), got -0.5"
    )]
    fn test_j_nan_row_still_honours_eps_domain() {
        Spi::run("CREATE TABLE tje(v float8, s float8); INSERT INTO tje VALUES ('NaN', 'NaN')")
            .unwrap();
        Spi::get_one::<f64>("SELECT softavg(v, s, -0.5) FROM tje").unwrap();
    }

    // -----------------------------------------------------------------
    // Upgrade path (workstream 13): ALTER EXTENSION bruce_pg UPDATE
    // through a real versioned upgrade script, exercised in-database.
    // The harness installs the extension at the current default version
    // (0.2.0) before tests run, so the test stages a synthetic
    // 0.2.0 -> 0.2.1 script (same trivial shape as the shipped
    // sql/bruce_pg--0.1.1--0.1.2.sql) into <sharedir>/extension/ and
    // drives PG's own update machinery over it.
    // -----------------------------------------------------------------
    #[pg_test]
    fn test_i_upgrade_path_alter_extension() {
        let installed = Spi::get_one::<String>(
            "SELECT extversion FROM pg_extension WHERE extname = 'bruce_pg'",
        )
        .unwrap()
        .expect("extension not installed");
        assert_eq!(
            installed, "0.2.0",
            "control default_version tracks Cargo.toml"
        );

        let sharedir =
            Spi::get_one::<String>("SELECT setting FROM pg_config WHERE name = 'SHAREDIR'")
                .unwrap()
                .expect("pg_config view gave no SHAREDIR");
        let script = std::path::Path::new(&sharedir)
            .join("extension")
            .join("bruce_pg--0.2.0--0.2.1.sql");
        std::fs::write(
            &script,
            "COMMENT ON AGGREGATE softavg(float8, float8, float8) IS \
             'bruce_pg 0.2.1 upgrade probe';\n",
        )
        .expect("cannot stage upgrade script into sharedir");

        // the actual machinery under test
        let outcome = Spi::run("ALTER EXTENSION bruce_pg UPDATE TO '0.2.1'");
        // stage file cleanup must happen even if the ALTER failed
        // (the catalog change itself rolls back with the test txn)
        let _ = std::fs::remove_file(&script);
        outcome.expect("ALTER EXTENSION bruce_pg UPDATE TO '0.2.1' failed");

        let now = Spi::get_one::<String>(
            "SELECT extversion FROM pg_extension WHERE extname = 'bruce_pg'",
        )
        .unwrap()
        .unwrap();
        assert_eq!(now, "0.2.1", "extversion did not advance");
        let comment = Spi::get_one::<String>(
            "SELECT obj_description('softavg(float8, float8, float8)'::regprocedure, 'pg_proc')",
        )
        .unwrap()
        .expect("upgrade script's COMMENT did not apply");
        assert_eq!(comment, "bruce_pg 0.2.1 upgrade probe");
    }

    fn explain(query: &str) -> String {
        Spi::connect(|client| {
            let table = client.select(&format!("EXPLAIN (COSTS OFF) {query}"), None, &[])?;
            let mut out = String::new();
            for row in table {
                if let Some(line) = row.get::<String>(1)? {
                    out.push_str(&line);
                    out.push('\n');
                }
            }
            Ok::<String, pgrx::spi::Error>(out)
        })
        .expect("EXPLAIN failed")
    }


    // ---------------------------------------------------------------------
// 0.2.0 tests: internal state, serial/deserial, moving aggregate, and the
// shared conformance vectors (../core/conformance/vectors.json).
    // ---------------------------------------------------------------------

    const VECTORS: &str = include_str!("../../core/conformance/vectors.json");


    fn force_parallel() {
        Spi::run(
            "SET parallel_setup_cost = 0; SET parallel_tuple_cost = 0; \
             SET min_parallel_table_scan_size = 0; SET max_parallel_workers_per_gather = 4",
        )
        .unwrap();
    }


    /// rows of (key, got, want) -> panics on the first mismatch.
    /// NULL must match NULL; NaN matches NaN; otherwise |got-want| <= rtol |want|.
    fn compare_rows(query: &str, rtol: f64, what: &str) -> usize {
        compare_rows_scaled(query, rtol, 0.0, what)
    }

    /// As [`compare_rows`], with the tolerance `rtol * max(|want|, scale)`:
    /// an inverse (removal) cancels the numerator `u` too, so its error is
    /// relative to the value scale, not to a result that happens to be ~0.
    fn compare_rows_scaled(query: &str, rtol: f64, scale: f64, what: &str) -> usize {
        Spi::connect(|client| {
            let rows = client.select(query, None, &[])?;
            let mut n = 0usize;
            for row in rows {
                let k = row.get::<i64>(1)?.expect("NULL key");
                let got = row.get::<f64>(2)?;
                let want = row.get::<f64>(3)?;
                match (got, want) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        let ok = (g.is_nan() && w.is_nan())
                            || g == w
                            || (g - w).abs() <= rtol * w.abs().max(scale);
                        assert!(ok, "{what}: key {k}: got {g:e}, want {w:e} (rtol {rtol:e})");
                    }
                    (g, w) => panic!("{what}: key {k}: got {g:?}, want {w:?} (NULL mismatch)"),
                }
                n += 1;
            }
            Ok::<usize, pgrx::spi::Error>(n)
        })
        .expect("SPI failed")
    }

    fn load_vectors() {
        Spi::run(&format!(
            "CREATE TABLE confj AS SELECT $vec${VECTORS}$vec$::jsonb AS j"
        ))
        .unwrap();
        Spi::run(
            "CREATE TABLE conf(id int, g int, s float8, v float8); \
             INSERT INTO conf SELECT (r->>0)::int, (r->>1)::int, (r->>2)::float8, (r->>3)::float8 \
             FROM confj, jsonb_array_elements(j->'table') r; \
             ALTER TABLE conf SET (parallel_workers = 4); ANALYZE conf",
        )
        .unwrap();
    }

    fn eps_literals(section: &str) -> Vec<String> {
        Spi::connect(|client| {
            let rows = client.select(
                &format!(
                    "SELECT e->>'eps' FROM confj, jsonb_array_elements(j->'{section}') \
                     WITH ORDINALITY AS t(e, o) ORDER BY o"
                ),
                None,
                &[],
            )?;
            let mut out = vec![];
            for r in rows {
                out.push(r.get::<String>(1)?.expect("eps"));
            }
            Ok::<Vec<String>, pgrx::spi::Error>(out)
        })
        .unwrap()
    }

    /// The shared conformance vectors, GROUP BY half: every eps, serial
    /// and through a forced parallel plan (serialize -> deserialize ->
    /// combine), rtol 1e-12, null -> SQL NULL.
    #[pg_test]
    fn test_conformance_vectors_group_by() {
        load_vectors();
        let n_rows = get_i64("SELECT count(*) FROM conf");
        assert_eq!(n_rows, 2004);
        let eps_list = eps_literals("group_by");
        assert_eq!(eps_list.len(), 7);
        for parallel in [false, true] {
            if parallel {
                force_parallel();
            } else {
                Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
            }
            for (i, eps) in eps_list.iter().enumerate() {
                let q = format!(
                    "SELECT x.k::int8, r.got, (x.v #>> '{{}}')::float8 \
                     FROM confj, jsonb_each((j->'group_by'->{i})->'expected') AS x(k, v) \
                     LEFT JOIN (SELECT g, softavg(v, s, '{eps}'::float8) AS got FROM conf GROUP BY g) r \
                       ON r.g = x.k::int ORDER BY 1"
                );
                if parallel && i == 0 {
                    let plan = explain(&format!(
                        "SELECT g, softavg(v, s, '{eps}'::float8) FROM conf GROUP BY g"
                    ));
                    assert!(
                        plan.contains("Gather") && plan.contains("Partial"),
                        "plan did not parallelize:\n{plan}"
                    );
                }
                let n = compare_rows(&q, 1e-12, &format!("group_by eps={eps} parallel={parallel}"));
                assert_eq!(n, 10, "10 groups expected");
            }
        }
    }

    /// The window half: ORDER BY id ROWS BETWEEN 49 PRECEDING AND
    /// CURRENT ROW over ids < 2000, rtol 1e-9, through the moving
    /// aggregate (asserted via softavg_window_stats()).
    #[pg_test]
    fn test_conformance_vectors_window() {
        load_vectors();
        let k = get_i64("SELECT (j->>'window_rows_preceding')::int8 FROM confj");
        let eps_list = eps_literals("window");
        assert_eq!(eps_list.len(), 3);
        for (i, eps) in eps_list.iter().enumerate() {
            Spi::run("SELECT softavg_window_stats_reset()").unwrap();
            let q = format!(
                "WITH w AS (SELECT id, softavg(v, s, '{eps}'::float8) OVER \
                   (ORDER BY id ROWS BETWEEN {k} PRECEDING AND CURRENT ROW) AS got \
                   FROM conf WHERE id < 2000), \
                 e AS (SELECT (o - 1)::int AS id, x::float8 AS want \
                   FROM confj, jsonb_array_elements_text((j->'window'->{i})->'expected') \
                   WITH ORDINALITY AS t(x, o)) \
                 SELECT e.id::int8, w.got, e.want FROM e LEFT JOIN w USING (id) ORDER BY 1"
            );
            // PROTOCOL: |err| <= 1e-9 * max|v| (absolute, value units)
            let vmax = Spi::get_one::<f64>("SELECT max(abs(v)) FROM conf WHERE id < 2000")
                .unwrap()
                .unwrap();
            let n = compare_rows_scaled(&q, 1e-9, vmax, &format!("window eps={eps}"));
            assert_eq!(n, 2000);
            let inverses = get_i64("SELECT inverses FROM softavg_window_stats()");
            assert!(inverses > 0, "eps={eps}: the moving-aggregate path was not used");
        }
    }

    /// Catalog: internal state, serial/deserial, combine, moving form.
    #[pg_test]
    fn test_k_catalog_declares_internal_state_and_moving_form() {
        let row = |col: &str| {
            Spi::get_one::<String>(&format!(
                "SELECT {col}::text FROM pg_aggregate \
                 WHERE aggfnoid = 'softavg(float8, float8, float8)'::regprocedure"
            ))
            .unwrap()
            .unwrap_or_default()
        };
        assert_eq!(row("aggtranstype::regtype"), "internal");
        assert_eq!(row("aggmtranstype::regtype"), "internal");
        assert_eq!(row("aggtransfn"), "softavg_sfunc");
        assert_eq!(row("aggcombinefn"), "softavg_combine");
        assert_eq!(row("aggserialfn"), "softavg_serialize");
        assert_eq!(row("aggdeserialfn"), "softavg_deserialize");
        assert_eq!(row("aggfinalfn"), "softavg_finalfn");
        assert_eq!(row("aggmtransfn"), "softavg_msfunc");
        assert_eq!(row("aggminvtransfn"), "softavg_minvfunc");
        assert_eq!(row("aggmfinalfn"), "softavg_mfinalfn");
        assert_eq!(row("aggfinalmodify"), "r");
        assert_eq!(row("aggmfinalmodify"), "r");
        let par = Spi::get_one::<String>(
            "SELECT proparallel::text FROM pg_proc \
             WHERE oid = 'softavg(float8, float8, float8)'::regprocedure",
        )
        .unwrap()
        .unwrap();
        assert_eq!(par, "s", "softavg must stay PARALLEL SAFE");
        // the per-row functions are non-strict (they implement the NULL
        // discipline); serial/deserial are strict
        let strict = |f: &str| {
            Spi::get_one::<bool>(&format!(
                "SELECT proisstrict FROM pg_proc WHERE proname = '{f}'"
            ))
            .unwrap()
            .unwrap()
        };
        for f in ["softavg_sfunc", "softavg_combine", "softavg_msfunc", "softavg_minvfunc"] {
            assert!(!strict(f), "{f} must be non-strict");
        }
        assert!(strict("softavg_serialize") && strict("softavg_deserialize"));
    }

    /// Grouped parallel aggregation (serialize/deserialize/combine on
    /// many groups, including groups only some workers see) equals the
    /// serial plan, group by group.
    #[pg_test]
    fn test_k_grouped_parallel_matches_serial() {
        Spi::run(
            "CREATE TABLE tk AS SELECT i, (i::int8 * 7919) % 997 AS g, sin(i::float8) * 10.0 AS v, \
             cos(i::float8 * 1.3) AS s FROM generate_series(1, 300000) g(i); \
             UPDATE tk SET g = 5000 + i WHERE i % 50000 = 0; \
             UPDATE tk SET s = 'infinity' WHERE i % 70001 = 0; \
             ALTER TABLE tk SET (parallel_workers = 4); ANALYZE tk",
        )
        .unwrap();
        for eps in ["0.0", "0.001", "0.25", "'infinity'::float8"] {
            Spi::run("SET max_parallel_workers_per_gather = 0").unwrap();
            Spi::run(&format!(
                "CREATE TEMP TABLE ser AS SELECT g, softavg(v, s, {eps}) AS x FROM tk GROUP BY g"
            ))
            .unwrap();
            force_parallel();
            let q = format!("SELECT g, softavg(v, s, {eps}) AS x FROM tk GROUP BY g");
            let plan = explain(&q);
            assert!(
                plan.contains("Gather") && plan.contains("Partial"),
                "plan did not parallelize:\n{plan}"
            );
            let bad = get_i64(&format!(
                "SELECT count(*) FROM ser FULL JOIN ({q}) p USING (g) \
                 WHERE ser.x IS DISTINCT FROM p.x AND NOT (abs(ser.x - p.x) <= 1e-12 * abs(ser.x))"
            ));
            assert_eq!(bad, 0, "eps={eps}: parallel != serial");
            assert_eq!(get_i64("SELECT count(*) FROM ser"), 997 + 6);
            Spi::run("DROP TABLE ser").unwrap();
        }
    }

    /// Hash aggregation that spills (tiny work_mem) still gives the
    /// sorted-group answer.
    #[pg_test]
    fn test_k_hashagg_spill_matches_sort_agg() {
        Spi::run(
            "CREATE TABLE th AS SELECT i % 40000 AS g, sin(i::float8) AS v, cos(i::float8) AS s \
             FROM generate_series(1, 200000) g(i); ANALYZE th",
        )
        .unwrap();
        Spi::run("SET max_parallel_workers_per_gather = 0; SET enable_hashagg = off").unwrap();
        Spi::run("CREATE TEMP TABLE srt AS SELECT g, softavg(v, s, 0.05) AS x FROM th GROUP BY g")
            .unwrap();
        Spi::run("SET enable_hashagg = on; SET enable_sort = off; SET work_mem = '64kB'").unwrap();
        let plan = explain("SELECT g, softavg(v, s, 0.05) FROM th GROUP BY g");
        assert!(plan.contains("HashAggregate"), "{plan}");
        let bad = get_i64(
            "SELECT count(*) FROM srt FULL JOIN \
             (SELECT g, softavg(v, s, 0.05) AS x FROM th GROUP BY g) h USING (g) \
             WHERE srt.x IS DISTINCT FROM h.x AND NOT (abs(srt.x - h.x) <= 1e-12 * abs(srt.x))",
        );
        // absorption order differs between the two plans: ulp-level only
        assert_eq!(bad, 0);
    }

    fn make_window_table() {
        Spi::run(
            "CREATE TABLE tw AS SELECT i, ((i * 7919) % 1000) / 100.0 AS v, \
             sin(i * 1.7) AS s FROM generate_series(1, 3000) g(i); \
             UPDATE tw SET v = NULL WHERE i % 97 = 0; \
             UPDATE tw SET s = NULL WHERE i % 89 = 0; \
             UPDATE tw SET s = 'NaN' WHERE i = 1234; \
             UPDATE tw SET s = 'infinity' WHERE i IN (500, 503, 2000); \
             UPDATE tw SET s = '-infinity' WHERE i % 61 = 0; \
             UPDATE tw SET s = 0.5 WHERE i BETWEEN 2500 AND 2520",
        )
        .unwrap();
    }

    /// Sliding windows through MSFUNC/MINVFUNC equal a fresh recompute of
    /// every frame (LATERAL subquery), at every eps regime and frame
    /// width, including NULL, NaN, +inf, -inf rows and ties.
    #[pg_test]
    fn test_w_window_matches_frame_recompute() {
        make_window_table();
        for eps in ["0.0", "0.001", "0.02", "0.1", "1.0", "'infinity'::float8"] {
            for k in [0, 1, 7, 49] {
                let q = format!(
                    "SELECT t.i::int8, \
                       softavg(t.v, t.s, {eps}) OVER (ORDER BY t.i ROWS BETWEEN {k} PRECEDING AND CURRENT ROW), \
                       (SELECT softavg(u.v, u.s, {eps}) FROM tw u WHERE u.i BETWEEN t.i - {k} AND t.i) \
                     FROM tw t ORDER BY t.i"
                );
                // values are in [0, 10): absolute tolerance 1e-9 * 10
                let n = compare_rows_scaled(&q, 1e-9, 10.0, &format!("window eps={eps} k={k}"));
                assert_eq!(n, 3000);
            }
        }
    }

    /// The moving path is really used (one fresh state per partition,
    /// one inverse per row leaving the frame) when nothing cancels, and
    /// the guard fires (punt -> PostgreSQL recomputes) when the removed
    /// row held the weight at sharp eps -- with correct results either way.
    #[pg_test]
    fn test_w_moving_path_counters() {
        Spi::run(
            "CREATE TABLE tm AS SELECT i, i % 3 AS p, (i % 10)::float8 AS v, \
             cos(i * 0.37) AS s FROM generate_series(1, 3000) g(i)",
        )
        .unwrap();
        Spi::run("SELECT softavg_window_stats_reset()").unwrap();
        let n = get_i64(
            "SELECT count(x) FROM (SELECT softavg(v, s, 1.0) OVER \
             (PARTITION BY p ORDER BY i ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS x FROM tm) q",
        );
        assert_eq!(n, 3000);
        let (fresh, inv, punts) = Spi::get_three::<i64, i64, i64>(
            "SELECT fresh_states, inverses, punts FROM softavg_window_stats()",
        )
        .unwrap();
        assert_eq!(fresh, Some(3), "one state per partition");
        assert_eq!(inv, Some(3000 - 3 * 11), "one inverse per row leaving a frame");
        assert_eq!(punts, Some(0));

        // sharp eps: removing the frame's max leaves e^{-(gap)/1e-3}
        Spi::run("SELECT softavg_window_stats_reset()").unwrap();
        let q = "SELECT i::int8, softavg(v, s, 0.001) OVER (ORDER BY i ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), \
                 (SELECT softavg(u.v, u.s, 0.001) FROM tm u WHERE u.i BETWEEN t.i - 10 AND t.i) \
                 FROM tm t ORDER BY i";
        // values are in [0, 9]; some frames' exact answer is ~1e-40
        compare_rows_scaled(q, 1e-9, 9.0, "sharp window");
        let (punts, inv) =
            Spi::get_two::<i64, i64>("SELECT punts, inverses FROM softavg_window_stats()").unwrap();
        assert!(punts.unwrap() > 0, "the cancellation guard never fired");
        assert!(punts.unwrap() < inv.unwrap(), "every inverse punted");
    }

    /// eps changing inside a frame is still an error in the moving path.
    #[pg_test(error = "softavg: eps must be constant within an aggregate group (0.6 vs 0.7)")]
    fn test_w_eps_change_rejected_in_window() {
        Spi::run(
            "CREATE TABLE twe AS SELECT i, i::float8 AS v, 0.0::float8 AS s, \
             (0.5 + i * 0.1)::float8 AS e FROM generate_series(1, 3) g(i)",
        )
        .unwrap();
        Spi::run(
            "SELECT softavg(v, s, e) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM twe",
        )
        .unwrap();
    }

    /// A window whose frame head never moves uses the plain SFUNC and
    /// calls FINALFUNC repeatedly on one state (FINALFUNC is read-only).
    #[pg_test]
    fn test_w_growing_frame_and_partition_window() {
        make_window_table();
        let q = "SELECT t.i::int8, softavg(t.v, t.s, 0.1) OVER (ORDER BY t.i ROWS UNBOUNDED PRECEDING), \
                 (SELECT softavg(u.v, u.s, 0.1) FROM tw u WHERE u.i <= t.i) FROM tw t WHERE t.i <= 1300 ORDER BY 1";
        assert_eq!(compare_rows(q, 1e-12, "growing frame"), 1300);
        let q2 = "SELECT t.i::int8, softavg(t.v, t.s, 0.1) OVER (PARTITION BY t.i % 5), \
                  (SELECT softavg(u.v, u.s, 0.1) FROM tw u WHERE u.i % 5 = t.i % 5) FROM tw t ORDER BY 1";
        assert_eq!(compare_rows(q2, 1e-12, "partition window"), 3000);
    }

    /// The 0.1.2 float8[] helpers still compose with the new state:
    /// softavg_state (internal state, array output) re-associates.
    #[pg_test]
    fn test_k_state_array_helpers_still_compose() {
        Spi::run(
            "CREATE TABLE ts AS SELECT i, sin(i::float8) AS v, cos(i::float8) AS s \
             FROM generate_series(1, 5000) g(i)",
        )
        .unwrap();
        let whole = Spi::get_one::<f64>("SELECT softavg(v, s, 0.01) FROM ts").unwrap().unwrap();
        let re = Spi::get_one::<f64>(
            "SELECT bruce_softavg_final(bruce_softavg_combine( \
               (SELECT softavg_state(v, s, 0.01) FROM ts WHERE i < 1700), \
               (SELECT softavg_state(v, s, 0.01) FROM ts WHERE i >= 1700)))",
        )
        .unwrap()
        .unwrap();
        assert!((whole - re).abs() <= 1e-12 * whole.abs());
        // and the legacy float8[] aggregate built from the kept helpers
        Spi::run(
            "CREATE AGGREGATE softavg_v012(float8, float8, float8) (SFUNC = bruce_softavg_sfunc, \
             STYPE = float8[], COMBINEFUNC = bruce_softavg_combine, FINALFUNC = bruce_softavg_final, \
             PARALLEL = SAFE)",
        )
        .unwrap();
        let old = Spi::get_one::<f64>("SELECT softavg_v012(v, s, 0.01) FROM ts").unwrap().unwrap();
        assert_eq!(old.to_bits(), whole.to_bits(), "0.1.2 path and 0.2.0 path must agree bit-for-bit");
    }

    // -----------------------------------------------------------------
    // Incremental maintenance (softavg_ivm_create)
    // -----------------------------------------------------------------

    fn ivm_fixture(eps: &str) {
        Spi::run(
            "CREATE TABLE tb_ivm(id int PRIMARY KEY, g int, v float8, s float8); \
             INSERT INTO tb_ivm SELECT i, i % 7, ((i * 7919) % 1000) / 100.0, cos(i * 0.37) \
             FROM generate_series(1, 5000) i; \
             CREATE INDEX ON tb_ivm(g)",
        )
        .unwrap();
        Spi::run(&format!(
            "SELECT softavg_ivm_create('mv_ivm', 'tb_ivm', 'g', 'v', 's', {eps})"
        ))
        .unwrap();
    }

    /// view == fresh GROUP BY (same groups; |err| <= 1e-9 * max|v|; NULL/NaN agree)
    fn ivm_check(eps: &str, what: &str) {
        let q = format!(
            "SELECT coalesce(f.g, m.g)::int8, m.softavg, f.x FROM mv_ivm m FULL JOIN \
             (SELECT g, softavg(v, s, {eps}) AS x FROM tb_ivm WHERE g IS NOT NULL GROUP BY g) f \
             ON f.g = m.g ORDER BY 1"
        );
        compare_rows_scaled(&q, 1e-9, 10.0, what);
        let bad = get_i64(
            "SELECT count(*) FROM mv_ivm m FULL JOIN (SELECT g, count(*) c FROM tb_ivm \
             WHERE g IS NOT NULL GROUP BY g) f ON f.g = m.g WHERE m.n IS DISTINCT FROM f.c",
        );
        assert_eq!(bad, 0, "{what}: group row counts differ");
    }

    #[pg_test]
    fn test_ivm_tracks_inserts_deletes_updates_at_every_regime() {
        for (i, eps) in ["0.001", "0.02", "0.1", "1.0", "0.0", "'infinity'::float8"].iter().enumerate() {
            if i > 0 {
                Spi::run("SELECT softavg_ivm_drop('mv_ivm'); DROP TABLE tb_ivm").unwrap();
            }
            ivm_fixture(eps);
            ivm_check(eps, &format!("eps={eps} initial"));
            // single-row deletes of every group's argmax row (carries the weight)
            Spi::run(
                "DO $$ DECLARE r record; BEGIN FOR r IN SELECT DISTINCT ON (g) id FROM tb_ivm \
                 ORDER BY g, s DESC LOOP DELETE FROM tb_ivm WHERE id = r.id; END LOOP; END $$",
            )
            .unwrap();
            ivm_check(eps, &format!("eps={eps} after argmax deletes"));
            // multi-row statements, updates that move rows between groups,
            // value/score changes, special values, a new group, a NULL key
            Spi::run(
                "DELETE FROM tb_ivm WHERE id % 13 = 0; \
                 UPDATE tb_ivm SET g = (g + 1) % 7 WHERE id % 17 = 0; \
                 UPDATE tb_ivm SET v = v + 1, s = s * 0.5 WHERE id % 19 = 0; \
                 INSERT INTO tb_ivm VALUES (10001, 3, NULL, 0.5), (10002, 3, 4.0, NULL), \
                     (10003, 4, 2.0, '-infinity'), (10004, 9, 5.0, 0.25), (10005, NULL, 1.0, 0.1); \
                 INSERT INTO tb_ivm SELECT 20000 + i, i % 7, 1.0 + (i % 5), 0.99 FROM generate_series(1, 300) i",
            )
            .unwrap();
            ivm_check(eps, &format!("eps={eps} after mixed statements"));
            // NaN enters and leaves
            Spi::run("UPDATE tb_ivm SET s = 'NaN' WHERE id = 10004").unwrap();
            ivm_check(eps, &format!("eps={eps} with NaN"));
            Spi::run("DELETE FROM tb_ivm WHERE id = 10004").unwrap();
            ivm_check(eps, &format!("eps={eps} NaN removed, group 9 gone"));
            assert_eq!(get_i64("SELECT count(*) FROM mv_ivm WHERE g = 9"), 0);
        }
    }

    /// Recomputes happen (and are counted) when the guard fires; the
    /// answer stays within the bound.
    #[pg_test]
    fn test_ivm_recomputes_inexact_groups() {
        ivm_fixture("0.001");
        // one dominant row per group (score 2 vs <= 1: weight ratio e^{1000})
        Spi::run("INSERT INTO tb_ivm SELECT 30000 + g, g, 5.0, 2.0 FROM generate_series(0, 6) g").unwrap();
        Spi::run("SELECT softavg_ivm_stats_reset()").unwrap();
        Spi::run(
            "DO $$ DECLARE r record; BEGIN FOR r IN SELECT id FROM tb_ivm WHERE id >= 30000 \
             LOOP DELETE FROM tb_ivm WHERE id = r.id; END LOOP; END $$",
        )
        .unwrap();
        let (stmts, rec) =
            Spi::get_two::<i64, i64>("SELECT * FROM softavg_ivm_stats()").unwrap();
        assert_eq!(stmts, Some(7));
        assert!(rec.unwrap() > 0, "removing a dominant row must recompute its group");
        ivm_check("0.001", "after recomputes");
        let inexact = get_i64("SELECT count(*) FROM mv_ivm_state WHERE NOT softavg_inv_exact(state)");
        assert_eq!(inexact, 0);
    }

    /// The state is a table: a rolled-back (sub)transaction leaves it as it was.
    #[pg_test]
    fn test_ivm_rollback_restores_state() {
        ivm_fixture("0.02");
        let before = Spi::get_one::<String>(
            "SELECT string_agg(g || ':' || encode(state, 'hex'), ',' ORDER BY g) FROM mv_ivm_state",
        )
        .unwrap()
        .unwrap();
        Spi::run(
            "DO $$ BEGIN \
               DELETE FROM tb_ivm WHERE id % 3 = 0; \
               INSERT INTO tb_ivm SELECT 90000 + i, 2, 9.0, 0.999 FROM generate_series(1, 50) i; \
               RAISE EXCEPTION 'roll back'; \
             EXCEPTION WHEN raise_exception THEN NULL; END $$",
        )
        .unwrap();
        let after = Spi::get_one::<String>(
            "SELECT string_agg(g || ':' || encode(state, 'hex'), ',' ORDER BY g) FROM mv_ivm_state",
        )
        .unwrap()
        .unwrap();
        assert_eq!(before, after, "state survived a rolled-back subtransaction");
        ivm_check("0.02", "after rollback");
    }

    #[pg_test]
    fn test_ivm_truncate_and_drop() {
        ivm_fixture("0.1");
        Spi::run("TRUNCATE tb_ivm").unwrap();
        assert_eq!(get_i64("SELECT count(*) FROM mv_ivm"), 0);
        Spi::run("INSERT INTO tb_ivm VALUES (1, 1, 2.0, 0.5), (2, 1, 4.0, 0.5)").unwrap();
        let x = Spi::get_one::<f64>("SELECT softavg FROM mv_ivm WHERE g = 1").unwrap().unwrap();
        assert_eq!(x, 3.0);
        Spi::run("SELECT softavg_ivm_drop('mv_ivm')").unwrap();
        assert_eq!(
            get_i64("SELECT count(*) FROM pg_trigger WHERE tgrelid = 'tb_ivm'::regclass AND NOT tgisinternal"),
            0
        );
    }

    // -----------------------------------------------------------------
    // Incremental maintenance over a many-to-many join (overlapping
    // groups): softavg_ivm_create_join
    // -----------------------------------------------------------------

    /// base tb_jb(id, v, s) and bridge tb_jbr(item, g): item i belongs to
    /// 0..4 groups (i % 5 memberships, groups (i + j*3) % 11), so groups
    /// overlap and some items have no membership.
    fn ivm_join_fixture(eps: &str) {
        Spi::run(
            "CREATE TABLE tb_jb(id int PRIMARY KEY, v float8, s float8); \
             INSERT INTO tb_jb SELECT i, ((i * 7919) % 1000) / 100.0, cos(i * 0.37) \
             FROM generate_series(1, 4000) i; \
             CREATE TABLE tb_jbr(item int, g int); \
             INSERT INTO tb_jbr SELECT i, (i + j * 3) % 11 FROM generate_series(1, 4000) i, \
                 generate_series(0, 3) j WHERE j < i % 5; \
             CREATE INDEX ON tb_jbr(item)",
        )
        .unwrap();
        Spi::run(&format!(
            "SELECT softavg_ivm_create_join('mv_j', 'tb_jb', 'id', 'tb_jbr', 'item', 'g', 'v', 's', {eps})"
        ))
        .unwrap();
    }

    /// view == fresh GROUP BY over the join (same groups and counts,
    /// |err| <= 1e-9 * max(|v|, 10), NULL/NaN agree)
    fn ivm_join_check(eps: &str, what: &str) {
        let fresh = format!(
            "SELECT br.g, softavg(b.v, b.s, {eps}) AS x, count(*) AS c FROM tb_jb b \
             JOIN tb_jbr br ON br.item = b.id WHERE br.g IS NOT NULL GROUP BY br.g"
        );
        compare_rows_scaled(
            &format!(
                "SELECT coalesce(f.g, m.g)::int8, m.softavg, f.x FROM mv_j m FULL JOIN ({fresh}) f \
                 ON f.g = m.g ORDER BY 1"
            ),
            1e-9,
            10.0,
            what,
        );
        let bad = get_i64(&format!(
            "SELECT count(*) FROM mv_j m FULL JOIN ({fresh}) f ON f.g = m.g WHERE m.n IS DISTINCT FROM f.c"
        ));
        assert_eq!(bad, 0, "{what}: group membership counts differ");
        assert_eq!(
            get_i64("SELECT count(*) FROM mv_j_state WHERE NOT softavg_inv_exact(state)"),
            0,
            "{what}: an inexact state is visible"
        );
    }

    #[pg_test]
    fn test_ivm_join_tracks_base_and_bridge_edits_at_every_regime() {
        for (i, eps) in ["0.001", "0.02", "0.1", "1.0", "0.0", "'infinity'::float8"].iter().enumerate() {
            if i > 0 {
                Spi::run("SELECT softavg_ivm_drop('mv_j'); DROP TABLE tb_jb, tb_jbr").unwrap();
            }
            ivm_join_fixture(eps);
            ivm_join_check(eps, &format!("eps={eps} initial"));
            // base: delete every group's argmax item one statement at a time
            // (each delete fans out to all of that item's groups)
            Spi::run(
                "DO $$ DECLARE r record; BEGIN FOR r IN SELECT DISTINCT ON (br.g) b.id FROM tb_jb b \
                 JOIN tb_jbr br ON br.item = b.id ORDER BY br.g, b.s DESC LOOP \
                 DELETE FROM tb_jb WHERE id = r.id; END LOOP; END $$",
            )
            .unwrap();
            ivm_join_check(eps, &format!("eps={eps} after argmax base deletes"));
            // base: multi-row statements; value/score updates; a key change
            // (memberships follow the key); inserts that pick up dangling
            // bridge rows (the deleted argmax items' memberships are still
            // in the bridge); a row with no membership; NULL value/score
            Spi::run(
                "DELETE FROM tb_jb WHERE id % 13 = 0; \
                 UPDATE tb_jb SET v = v + 1, s = s * 0.5 WHERE id % 19 = 0; \
                 UPDATE tb_jb SET id = id + 100000 WHERE id % 23 = 0; \
                 INSERT INTO tb_jb SELECT i, 9.5, 0.97 FROM generate_series(1, 4000) i WHERE i % 13 = 0; \
                 INSERT INTO tb_jb VALUES (50001, NULL, 0.5), (50002, 4.0, NULL), (50003, 2.0, '-infinity'), \
                     (50004, 7.0, 0.3)",
            )
            .unwrap();
            ivm_join_check(eps, &format!("eps={eps} after base statements"));
            // bridge: new memberships (also of the moved keys and of a row
            // that had none), duplicates, a new group, a NULL group, deletes,
            // memberships moving between groups, an item moving
            Spi::run(
                "INSERT INTO tb_jbr SELECT id, 11 FROM tb_jb WHERE id % 7 = 0; \
                 INSERT INTO tb_jbr VALUES (50001, 2), (50002, 2), (50003, 4), (50004, 12), (50004, 12), \
                     (50004, NULL), (100023, 5); \
                 DELETE FROM tb_jbr WHERE item % 17 = 0 AND g % 2 = 0; \
                 UPDATE tb_jbr SET g = (g + 1) % 11 WHERE item % 29 = 0; \
                 UPDATE tb_jbr SET item = item + 1 WHERE item % 31 = 0",
            )
            .unwrap();
            ivm_join_check(eps, &format!("eps={eps} after bridge statements"));
            // NaN enters through the base and leaves through the bridge
            Spi::run("UPDATE tb_jb SET s = 'NaN' WHERE id = 50004").unwrap();
            ivm_join_check(eps, &format!("eps={eps} with NaN"));
            Spi::run("DELETE FROM tb_jbr WHERE item = 50004").unwrap();
            ivm_join_check(eps, &format!("eps={eps} NaN membership removed, group 12 gone"));
            assert_eq!(get_i64("SELECT count(*) FROM mv_j WHERE g = 12"), 0);
            // a base delete of an item whose memberships are all gone is a no-op
            Spi::run("DELETE FROM tb_jb WHERE id = 50004").unwrap();
            ivm_join_check(eps, &format!("eps={eps} final"));
        }
    }

    /// One base-row delete fans out to every group of the row; when the row
    /// dominates each of them at sharp eps, each of those groups (and only
    /// those) is recomputed, in one trigger statement.
    #[pg_test]
    fn test_ivm_join_fanout_recomputes_each_dominated_group() {
        ivm_join_fixture("0.001");
        Spi::run(
            "INSERT INTO tb_jb VALUES (70000, 5.0, 2.0); \
             INSERT INTO tb_jbr SELECT 70000, g FROM generate_series(0, 8) g",
        )
        .unwrap();
        ivm_join_check("0.001", "dominant item added to 9 groups");
        Spi::run("SELECT softavg_ivm_stats_reset()").unwrap();
        Spi::run("DELETE FROM tb_jb WHERE id = 70000").unwrap();
        let (stmts, rec) = Spi::get_two::<i64, i64>("SELECT * FROM softavg_ivm_stats()").unwrap();
        assert_eq!(stmts, Some(1));
        assert_eq!(rec, Some(9), "each dominated group recomputed once");
        ivm_join_check("0.001", "after the fan-out delete");
        // its (now dangling) memberships come back with a re-insert
        Spi::run("INSERT INTO tb_jb VALUES (70000, 5.0, 2.0)").unwrap();
        ivm_join_check("0.001", "re-insert picks up 9 memberships");
        assert_eq!(get_i64("SELECT count(*) FROM tb_jbr WHERE item = 70000"), 9);
    }

    #[pg_test]
    fn test_ivm_join_rollback_truncate_and_drop() {
        ivm_join_fixture("0.02");
        let snap = "SELECT string_agg(g || ':' || encode(state, 'hex'), ',' ORDER BY g) FROM mv_j_state";
        let before = Spi::get_one::<String>(snap).unwrap().unwrap();
        Spi::run(
            "DO $$ BEGIN \
               DELETE FROM tb_jb WHERE id % 3 = 0; \
               INSERT INTO tb_jbr SELECT id, 1 FROM tb_jb WHERE id % 5 = 0; \
               RAISE EXCEPTION 'roll back'; \
             EXCEPTION WHEN raise_exception THEN NULL; END $$",
        )
        .unwrap();
        assert_eq!(before, Spi::get_one::<String>(snap).unwrap().unwrap(), "state survived a rollback");
        ivm_join_check("0.02", "after rollback");
        Spi::run("TRUNCATE tb_jbr").unwrap();
        assert_eq!(get_i64("SELECT count(*) FROM mv_j"), 0);
        Spi::run("INSERT INTO tb_jbr VALUES (1, 1), (2, 1)").unwrap();
        ivm_join_check("0.02", "after bridge truncate + insert");
        Spi::run("TRUNCATE tb_jb").unwrap();
        assert_eq!(get_i64("SELECT count(*) FROM mv_j"), 0);
        Spi::run("INSERT INTO tb_jb VALUES (1, 2.0, 0.5), (2, 4.0, 0.5)").unwrap();
        let x = Spi::get_one::<f64>("SELECT softavg FROM mv_j WHERE g = 1").unwrap().unwrap();
        assert_eq!(x, 3.0);
        Spi::run("SELECT softavg_ivm_drop('mv_j')").unwrap();
        assert_eq!(
            get_i64(
                "SELECT count(*) FROM pg_trigger WHERE tgrelid IN ('tb_jb'::regclass, 'tb_jbr'::regclass) \
                 AND NOT tgisinternal"
            ),
            0
        );
        assert_eq!(get_i64("SELECT count(*) FROM pg_class WHERE relname LIKE 'mv_j%'"), 0);
    }

    /// A cascading foreign key from the bridge to the base is refused (the
    /// cascaded bridge delete fires before the base's statement trigger and
    /// neither trigger can see the other side's old rows).
    #[pg_test(
        error = "softavg_ivm_create_join: tb_jbr2 has a cascading foreign key to tb_jb, which this maintenance cannot follow"
    )]
    fn test_ivm_join_cascading_fk_rejected() {
        Spi::run(
            "CREATE TABLE tb_jb(id int PRIMARY KEY, v float8, s float8); \
             CREATE TABLE tb_jbr2(item int REFERENCES tb_jb(id) ON DELETE CASCADE, g int)",
        )
        .unwrap();
        Spi::run(
            "SELECT softavg_ivm_create_join('mv_j2', 'tb_jb', 'id', 'tb_jbr2', 'item', 'g', 'v', 's', 0.1)",
        )
        .unwrap();
    }

}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}

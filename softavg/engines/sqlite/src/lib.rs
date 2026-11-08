//! softagg-sqlite: SQLite loadable extension for
//! `softavg(value, score, eps)` as an aggregate and an aggregate window
//! function, plus `softagg_dot(a, b)` over float32 BLOB embeddings.
//!
//! The arithmetic is softagg-core (`../core`): `InvAcc` absorb / remove /
//! finalize. `state.rs` adds eps validation and the lazy frame shadow that
//! makes the inexact-window case refold instead of guessing; this file is
//! only the SQLite C-API glue (via libsqlite3-sys `loadable_extension`,
//! which routes every call through the `sqlite3_api_routines` table).
//!
//! SQL surface:
//! * `softavg(value, score, eps)` — aggregate and window (`xStep`,
//!   `xFinal`, `xValue`, `xInverse`). NULL `value`/`score` rows are
//!   skipped like `AVG`; no live rows -> NULL; `eps` NULL on a live row,
//!   negative, or changing within a group -> error. `eps = 9e999`
//!   (+infinity) is the plain mean. SQLite has no NaN (it stores and
//!   returns NaN as NULL), so a NaN can only arrive as NULL and is
//!   skipped exactly like `AVG` skips it.
//! * `softavg_agg(value, score, eps)` — the same aggregate without window
//!   support: state is the bare `ScalarAcc` monoid (+ eps), no row count,
//!   error bound or prefix record. Same NULL / eps semantics; bit-identical
//!   result to `softavg` in GROUP BY.
//! * `softagg_dot(a, b)` — inner product of two float32 BLOBs
//!   (sqlite-vec format); NULL in -> NULL.
//! * `softavg_stats()` / `softavg_stats_reset()` — process-wide counters
//!   of the rare window paths (refolds, refused refolds, shadow spills).

pub mod dot;
pub mod state;
/// relF baseline (experiments/anchored); not part of softagg-core
pub mod relf;

use libsqlite3_sys as ffi;
use state::{AggState, Error, WinState};
use std::ffi::{c_char, c_int};
use std::mem::MaybeUninit;

/// What lives in `sqlite3_aggregate_context` (zero-filled by SQLite).
#[repr(C)]
struct Slot {
    init: u8,
    st: MaybeUninit<WinState>,
}

/// The state, allocating it if `alloc`. `None`: not allocated (and not
/// requested), or out of memory when `alloc`.
#[inline]
unsafe fn state<'a>(ctx: *mut ffi::sqlite3_context, alloc: bool) -> Option<&'a mut WinState> {
    let n = if alloc { std::mem::size_of::<Slot>() as c_int } else { 0 };
    let p = ffi::sqlite3_aggregate_context(ctx, n) as *mut Slot;
    if p.is_null() {
        return None;
    }
    let slot = &mut *p;
    if slot.init == 0 {
        if !alloc {
            return None;
        }
        slot.st.write(WinState::new());
        slot.init = 1;
    }
    Some(slot.st.assume_init_mut())
}

#[inline]
unsafe fn arg(argv: *mut *mut ffi::sqlite3_value, i: usize) -> *mut ffi::sqlite3_value {
    *argv.add(i)
}

/// `(score, value)` of a live row, or `None` if the row is skipped.
#[inline]
unsafe fn live_row(argv: *mut *mut ffi::sqlite3_value) -> Option<(f64, f64)> {
    let (av, as_) = (arg(argv, 0), arg(argv, 1));
    if ffi::sqlite3_value_type(av) == ffi::SQLITE_NULL || ffi::sqlite3_value_type(as_) == ffi::SQLITE_NULL {
        return None;
    }
    let v = ffi::sqlite3_value_double(av);
    let s = ffi::sqlite3_value_double(as_);
    if v.is_nan() || s.is_nan() {
        // unreachable in SQLite (NaN is NULL there); skip like NULL
        return None;
    }
    Some((s, v))
}

#[inline]
unsafe fn eps_arg(argv: *mut *mut ffi::sqlite3_value) -> Option<f64> {
    let a = arg(argv, 2);
    if ffi::sqlite3_value_type(a) == ffi::SQLITE_NULL {
        None
    } else {
        Some(ffi::sqlite3_value_double(a))
    }
}

#[cold]
unsafe fn raise(ctx: *mut ffi::sqlite3_context, e: Error) {
    ffi::sqlite3_result_error(ctx, e.message().as_ptr() as *const c_char, -1);
}

#[inline]
unsafe fn emit(ctx: *mut ffi::sqlite3_context, r: Result<Option<f64>, Error>) {
    match r {
        Ok(Some(x)) => ffi::sqlite3_result_double(ctx, x),
        Ok(None) => ffi::sqlite3_result_null(ctx),
        Err(e) => raise(ctx, e),
    }
}

unsafe extern "C" fn softavg_step(ctx: *mut ffi::sqlite3_context, _argc: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let Some(st) = state(ctx, true) else {
        ffi::sqlite3_result_error_nomem(ctx);
        return;
    };
    match st.check_eps(eps_arg(argv)) {
        Ok(_) => st.step(s, v),
        Err(e) => raise(ctx, e),
    }
}

unsafe extern "C" fn softavg_inverse(ctx: *mut ffi::sqlite3_context, _argc: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let Some(st) = state(ctx, false) else {
        raise(ctx, Error::InverseOnEmpty);
        return;
    };
    let r = st.check_eps(eps_arg(argv)).and_then(|_| st.inverse(s, v));
    if let Err(e) = r {
        raise(ctx, e);
    }
}

unsafe extern "C" fn softavg_value(ctx: *mut ffi::sqlite3_context) {
    match state(ctx, false) {
        Some(st) => emit(ctx, st.value()),
        None => ffi::sqlite3_result_null(ctx),
    }
}

unsafe extern "C" fn softavg_final(ctx: *mut ffi::sqlite3_context) {
    let p = ffi::sqlite3_aggregate_context(ctx, 0) as *mut Slot;
    if p.is_null() || (*p).init == 0 {
        ffi::sqlite3_result_null(ctx);
        return;
    }
    let slot = &mut *p;
    let r = slot.st.assume_init_mut().finalize();
    emit(ctx, r);
    // SQLite frees the context memory itself; drop what WinState owns
    slot.st.assume_init_drop();
    slot.init = 0;
}

/// `softavg_agg` context: plain data, zero-filled by SQLite; `init` set on first use.
#[repr(C)]
struct AggSlot {
    init: u8,
    st: AggState,
}

unsafe extern "C" fn softavg_agg_step(ctx: *mut ffi::sqlite3_context, _argc: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let p = ffi::sqlite3_aggregate_context(ctx, std::mem::size_of::<AggSlot>() as c_int) as *mut AggSlot;
    if p.is_null() {
        ffi::sqlite3_result_error_nomem(ctx);
        return;
    }
    let slot = &mut *p;
    if slot.init == 0 {
        slot.st = AggState::new();
        slot.init = 1;
    }
    if let Err(e) = slot.st.step(s, v, eps_arg(argv)) {
        raise(ctx, e);
    }
}

unsafe extern "C" fn softavg_agg_final(ctx: *mut ffi::sqlite3_context) {
    let p = ffi::sqlite3_aggregate_context(ctx, 0) as *mut AggSlot;
    if p.is_null() || (*p).init == 0 {
        ffi::sqlite3_result_null(ctx);
        return;
    }
    emit(ctx, Ok((*p).st.finalize()));
}

const DOT_TYPE_ERR: &[u8] = b"softagg_dot: expects two float32 BLOBs (sqlite-vec format)\0";
const DOT_DIM_ERR: &[u8] = b"softagg_dot: vectors have different dimensions\0";

unsafe fn blob<'a>(a: *mut ffi::sqlite3_value) -> &'a [u8] {
    let p = ffi::sqlite3_value_blob(a) as *const u8;
    let n = ffi::sqlite3_value_bytes(a);
    if p.is_null() || n <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n as usize)
    }
}

unsafe extern "C" fn softagg_dot(ctx: *mut ffi::sqlite3_context, _argc: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let (a, b) = (arg(argv, 0), arg(argv, 1));
    let (ta, tb) = (ffi::sqlite3_value_type(a), ffi::sqlite3_value_type(b));
    if ta == ffi::SQLITE_NULL || tb == ffi::SQLITE_NULL {
        ffi::sqlite3_result_null(ctx);
        return;
    }
    if ta != ffi::SQLITE_BLOB || tb != ffi::SQLITE_BLOB {
        ffi::sqlite3_result_error(ctx, DOT_TYPE_ERR.as_ptr() as *const c_char, -1);
        return;
    }
    match dot::dot_f32_blob(blob(a), blob(b)) {
        Ok(x) => ffi::sqlite3_result_double(ctx, x),
        Err(dot::DotError::NotFloat32) => ffi::sqlite3_result_error(ctx, DOT_TYPE_ERR.as_ptr() as *const c_char, -1),
        Err(dot::DotError::DimMismatch) => ffi::sqlite3_result_error(ctx, DOT_DIM_ERR.as_ptr() as *const c_char, -1),
    }
}

unsafe fn result_text(ctx: *mut ffi::sqlite3_context, s: &str) {
    // SQLITE_TRANSIENT: SQLite copies the bytes
    ffi::sqlite3_result_text(ctx, s.as_ptr() as *const c_char, s.len() as c_int, ffi::SQLITE_TRANSIENT());
}

unsafe extern "C" fn softavg_stats(ctx: *mut ffi::sqlite3_context, _argc: c_int, _argv: *mut *mut ffi::sqlite3_value) {
    result_text(ctx, &state::stats_json());
}

unsafe extern "C" fn softavg_stats_reset(ctx: *mut ffi::sqlite3_context, _argc: c_int, _argv: *mut *mut ffi::sqlite3_value) {
    state::stats_reset();
    ffi::sqlite3_result_null(ctx);
}

unsafe extern "C" fn softagg_sqlite_version(ctx: *mut ffi::sqlite3_context, _argc: c_int, _argv: *mut *mut ffi::sqlite3_value) {
    result_text(
        ctx,
        concat!(
            "softagg-sqlite ",
            env!("CARGO_PKG_VERSION"),
            " (softagg-core path dep; prefix_cap=64, shadow_cap=1048576)"
        ),
    );
}

unsafe fn register(db: *mut ffi::sqlite3) -> c_int {
    let det = ffi::SQLITE_UTF8 | ffi::SQLITE_DETERMINISTIC | ffi::SQLITE_INNOCUOUS;
    let rc = ffi::sqlite3_create_window_function(
        db,
        c"softavg".as_ptr(),
        3,
        det,
        std::ptr::null_mut(),
        Some(softavg_step),
        Some(softavg_final),
        Some(softavg_value),
        Some(softavg_inverse),
        None,
    );
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    // aggregate-only twin: no xValue/xInverse, so GROUP BY pays only the monoid
    let rc = ffi::sqlite3_create_function_v2(
        db,
        c"softavg_agg".as_ptr(),
        3,
        det,
        std::ptr::null_mut(),
        None,
        Some(softavg_agg_step),
        Some(softavg_agg_final),
        None,
    );
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    let rc = relf::register(db);
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    type Scalar = unsafe extern "C" fn(*mut ffi::sqlite3_context, c_int, *mut *mut ffi::sqlite3_value);
    let scalars: [(&std::ffi::CStr, c_int, c_int, Scalar); 4] = [
        (c"softagg_dot", 2, det, softagg_dot),
        (c"softavg_stats", 0, ffi::SQLITE_UTF8, softavg_stats),
        (c"softavg_stats_reset", 0, ffi::SQLITE_UTF8, softavg_stats_reset),
        (c"softagg_sqlite_version", 0, det, softagg_sqlite_version),
    ];
    for (name, n, flags, f) in scalars {
        let rc = ffi::sqlite3_create_function_v2(db, name.as_ptr(), n, flags, std::ptr::null_mut(), Some(f), None, None, None);
        if rc != ffi::SQLITE_OK {
            return rc;
        }
    }
    ffi::SQLITE_OK
}

unsafe fn init(db: *mut ffi::sqlite3, pz_err: *mut *mut c_char, p_api: *mut ffi::sqlite3_api_routines) -> c_int {
    if let Err(e) = ffi::rusqlite_extension_init2(p_api) {
        // report through the host's allocator if we can
        if !pz_err.is_null() {
            if let Some(m) = (*p_api).malloc {
                let msg = format!("softagg-sqlite: {e}\0");
                let p = m(msg.len() as c_int) as *mut u8;
                if !p.is_null() {
                    std::ptr::copy_nonoverlapping(msg.as_ptr(), p, msg.len());
                    *pz_err = p as *mut c_char;
                }
            }
        }
        return ffi::SQLITE_ERROR;
    }
    register(db)
}

/// Generic entry point (tried first by `load_extension(path)`).
#[no_mangle]
pub unsafe extern "C" fn sqlite3_extension_init(
    db: *mut ffi::sqlite3,
    pz_err: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    init(db, pz_err, p_api)
}

/// Entry point SQLite derives from `libsoftagg_sqlite.so`.
#[no_mangle]
pub unsafe extern "C" fn sqlite3_softaggsqlite_init(
    db: *mut ffi::sqlite3,
    pz_err: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    init(db, pz_err, p_api)
}

//! Softmax-ladder ablation build for SQLite (NOT the shipped extension; the
//! shipped `softavg` is `engines/sqlite`, rung S4).
//!
//! * `softavg_s3(value, score, eps)` — rung S3: online softmax with plain
//!   removal. `xStep` = `ScalarAcc::absorb` (+ row count), `xInverse` =
//!   subtract `w = exp((s - mu)/eps)` and `w v` at the current anchor
//!   (`ladder_core::PlainInv::remove`): never re-anchors, never refuses, keeps
//!   no frame shadow. `xValue` returns `u / z` whatever it is.
//! * `softavg_s2(value, score, eps)` — rung S2: online softmax with NO removal.
//!   SQLite 3.37.2 will not run an aggregate without `xInverse` as a window
//!   function (`sqlite3_create_window_function` refuses `xValue` without
//!   `xInverse` with SQLITE_MISUSE; a legacy aggregate is rejected at prepare
//!   time with "may not be used as a window function"; both probed below), so
//!   the extension must recompute itself: `xInverse` only drops the oldest row
//!   of a frame shadow, and the next `xValue` refolds the whole frame with
//!   `ScalarAcc::absorb` (O(frame) per output row; what PostgreSQL does for an
//!   aggregate without MINVFUNC).
//! * `ladder_probe()` — JSON: return codes of the registration probes.
//! * `ladder_stats()` / `ladder_stats_reset()` — counters (S2 refolds and rows
//!   refolded, S3 inverse calls).
//!
//! NULL `value` / `score` rows are skipped (as `AVG` and the shipped
//! `softavg`); `eps` must be non-NULL, >= 0 and constant per partition.

use ladder_core::PlainInv;
use libsqlite3_sys as ffi;
use softagg_core::ScalarAcc;
use std::collections::VecDeque;
use std::ffi::{c_char, c_int};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};

static S2_REFOLDS: AtomicU64 = AtomicU64::new(0);
static S2_REFOLD_ROWS: AtomicU64 = AtomicU64::new(0);
static S3_INVERSES: AtomicU64 = AtomicU64::new(0);
static PROBE_NOINV_RC: AtomicI64 = AtomicI64::new(-1);
static PROBE_LEGACY_RC: AtomicI64 = AtomicI64::new(-1);

const E_EPS_NULL: &[u8] = b"softavg_ablation: eps must not be NULL\0";
const E_EPS_BAD: &[u8] = b"softavg_ablation: eps must be >= 0\0";
const E_EPS_CHANGED: &[u8] = b"softavg_ablation: eps must be constant within a partition\0";
const E_FIFO: &[u8] = b"softavg_s2: xInverse row is not the oldest frame row\0";

#[inline]
unsafe fn arg(argv: *mut *mut ffi::sqlite3_value, i: usize) -> *mut ffi::sqlite3_value {
    *argv.add(i)
}

#[inline]
unsafe fn live_row(argv: *mut *mut ffi::sqlite3_value) -> Option<(f64, f64)> {
    let (av, as_) = (arg(argv, 0), arg(argv, 1));
    if ffi::sqlite3_value_type(av) == ffi::SQLITE_NULL || ffi::sqlite3_value_type(as_) == ffi::SQLITE_NULL {
        return None;
    }
    Some((ffi::sqlite3_value_double(as_), ffi::sqlite3_value_double(av)))
}

#[cold]
unsafe fn raise(ctx: *mut ffi::sqlite3_context, m: &[u8]) {
    ffi::sqlite3_result_error(ctx, m.as_ptr() as *const c_char, -1);
}

/// eps of this row checked against the partition's.
#[inline]
unsafe fn eps_ok(ctx: *mut ffi::sqlite3_context, argv: *mut *mut ffi::sqlite3_value, eps: &mut f64, set: &mut bool) -> Option<f64> {
    let a = arg(argv, 2);
    if ffi::sqlite3_value_type(a) == ffi::SQLITE_NULL {
        raise(ctx, E_EPS_NULL);
        return None;
    }
    let e = ffi::sqlite3_value_double(a);
    if *set {
        if e != *eps {
            raise(ctx, E_EPS_CHANGED);
            return None;
        }
        return Some(e);
    }
    if !(e >= 0.0) {
        raise(ctx, E_EPS_BAD);
        return None;
    }
    *eps = e;
    *set = true;
    Some(e)
}

#[inline]
unsafe fn emit(ctx: *mut ffi::sqlite3_context, x: Option<f64>) {
    match x {
        Some(x) => ffi::sqlite3_result_double(ctx, x),
        None => ffi::sqlite3_result_null(ctx),
    }
}

// --------------------------------------------------------------- S3
#[repr(C)]
struct S3 {
    init: u8,
    eps_set: bool,
    eps: f64,
    st: PlainInv,
}

#[inline]
unsafe fn s3_ctx<'a>(ctx: *mut ffi::sqlite3_context, alloc: bool) -> Option<&'a mut S3> {
    let p = ffi::sqlite3_aggregate_context(ctx, if alloc { std::mem::size_of::<S3>() as c_int } else { 0 }) as *mut S3;
    if p.is_null() {
        return None;
    }
    let s = &mut *p;
    if s.init == 0 {
        if !alloc {
            return None;
        }
        s.st = PlainInv::new();
        s.init = 1;
    }
    Some(s)
}

unsafe extern "C" fn s3_step(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let Some(st) = s3_ctx(ctx, true) else {
        ffi::sqlite3_result_error_nomem(ctx);
        return;
    };
    if let Some(e) = eps_ok(ctx, argv, &mut st.eps, &mut st.eps_set) {
        st.st.absorb(s, v, e);
    }
}

unsafe extern "C" fn s3_inverse(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let Some(st) = s3_ctx(ctx, false) else { return };
    S3_INVERSES.store(S3_INVERSES.load(Relaxed) + 1, Relaxed);
    if let Some(e) = eps_ok(ctx, argv, &mut st.eps, &mut st.eps_set) {
        st.st.remove(s, v, e);
    }
}

unsafe extern "C" fn s3_value(ctx: *mut ffi::sqlite3_context) {
    emit(ctx, s3_ctx(ctx, false).and_then(|s| s.st.finalize()));
}

unsafe extern "C" fn s3_final(ctx: *mut ffi::sqlite3_context) {
    emit(ctx, s3_ctx(ctx, false).and_then(|s| s.st.finalize()));
}

// --------------------------------------------------------------- S2
struct S2State {
    eps_set: bool,
    eps: f64,
    acc: ScalarAcc,
    /// a row left the frame since `acc` was last a fold of `shadow`
    dirty: bool,
    shadow: VecDeque<(f64, f64)>,
}

#[repr(C)]
struct S2 {
    init: u8,
    st: MaybeUninit<S2State>,
}

#[inline]
unsafe fn s2_ctx<'a>(ctx: *mut ffi::sqlite3_context, alloc: bool) -> Option<&'a mut S2State> {
    let p = ffi::sqlite3_aggregate_context(ctx, if alloc { std::mem::size_of::<S2>() as c_int } else { 0 }) as *mut S2;
    if p.is_null() {
        return None;
    }
    let s = &mut *p;
    if s.init == 0 {
        if !alloc {
            return None;
        }
        s.st.write(S2State { eps_set: false, eps: 0.0, acc: ScalarAcc::new(), dirty: false, shadow: VecDeque::new() });
        s.init = 1;
    }
    Some(s.st.assume_init_mut())
}

unsafe extern "C" fn s2_step(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let Some(st) = s2_ctx(ctx, true) else {
        ffi::sqlite3_result_error_nomem(ctx);
        return;
    };
    if let Some(e) = eps_ok(ctx, argv, &mut st.eps, &mut st.eps_set) {
        if !st.dirty {
            st.acc.absorb(s, v, e);
        }
        st.shadow.push_back((s, v));
    }
}

unsafe extern "C" fn s2_inverse(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v)) = live_row(argv) else { return };
    let Some(st) = s2_ctx(ctx, false) else { return };
    match st.shadow.pop_front() {
        Some((s0, v0)) if s0.to_bits() == s.to_bits() && v0.to_bits() == v.to_bits() => st.dirty = true,
        _ => raise(ctx, E_FIFO),
    }
}

#[inline]
fn s2_current(st: &mut S2State) -> Option<f64> {
    if st.dirty {
        let mut a = ScalarAcc::new();
        for &(s, v) in &st.shadow {
            a.absorb(s, v, st.eps);
        }
        st.acc = a;
        st.dirty = false;
        S2_REFOLDS.store(S2_REFOLDS.load(Relaxed) + 1, Relaxed);
        S2_REFOLD_ROWS.store(S2_REFOLD_ROWS.load(Relaxed) + st.shadow.len() as u64, Relaxed);
    }
    if st.shadow.is_empty() {
        return None;
    }
    st.acc.finalize()
}

unsafe extern "C" fn s2_value(ctx: *mut ffi::sqlite3_context) {
    emit(ctx, s2_ctx(ctx, false).and_then(s2_current));
}

unsafe extern "C" fn s2_final(ctx: *mut ffi::sqlite3_context) {
    let p = ffi::sqlite3_aggregate_context(ctx, 0) as *mut S2;
    if p.is_null() || (*p).init == 0 {
        ffi::sqlite3_result_null(ctx);
        return;
    }
    let x = s2_current((*p).st.assume_init_mut());
    emit(ctx, x);
    (*p).st.assume_init_drop();
    (*p).init = 0;
}

// --------------------------------------------------------------- misc
unsafe fn result_text(ctx: *mut ffi::sqlite3_context, s: &str) {
    ffi::sqlite3_result_text(ctx, s.as_ptr() as *const c_char, s.len() as c_int, ffi::SQLITE_TRANSIENT());
}

unsafe extern "C" fn ladder_stats(ctx: *mut ffi::sqlite3_context, _n: c_int, _a: *mut *mut ffi::sqlite3_value) {
    result_text(
        ctx,
        &format!(
            "{{\"s2_refolds\":{},\"s2_refold_rows\":{},\"s3_inverses\":{}}}",
            S2_REFOLDS.load(Relaxed),
            S2_REFOLD_ROWS.load(Relaxed),
            S3_INVERSES.load(Relaxed)
        ),
    );
}

unsafe extern "C" fn ladder_stats_reset(ctx: *mut ffi::sqlite3_context, _n: c_int, _a: *mut *mut ffi::sqlite3_value) {
    for c in [&S2_REFOLDS, &S2_REFOLD_ROWS, &S3_INVERSES] {
        c.store(0, Relaxed);
    }
    ffi::sqlite3_result_null(ctx);
}

unsafe extern "C" fn ladder_probe(ctx: *mut ffi::sqlite3_context, _n: c_int, _a: *mut *mut ffi::sqlite3_value) {
    result_text(
        ctx,
        &format!(
            "{{\"create_window_function_xValue_without_xInverse_rc\":{},\"create_function_v2_legacy_aggregate_rc\":{},\"sqlite_misuse\":{}}}",
            PROBE_NOINV_RC.load(Relaxed),
            PROBE_LEGACY_RC.load(Relaxed),
            ffi::SQLITE_MISUSE
        ),
    );
}

unsafe fn register(db: *mut ffi::sqlite3) -> c_int {
    let det = ffi::SQLITE_UTF8 | ffi::SQLITE_DETERMINISTIC | ffi::SQLITE_INNOCUOUS;
    let rc = ffi::sqlite3_create_window_function(
        db, c"softavg_s3".as_ptr(), 3, det, std::ptr::null_mut(),
        Some(s3_step), Some(s3_final), Some(s3_value), Some(s3_inverse), None,
    );
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    let rc = ffi::sqlite3_create_window_function(
        db, c"softavg_s2".as_ptr(), 3, det, std::ptr::null_mut(),
        Some(s2_step), Some(s2_final), Some(s2_value), Some(s2_inverse), None,
    );
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    // probe 1: a window function without xInverse (expected: SQLITE_MISUSE)
    let rc = ffi::sqlite3_create_window_function(
        db, c"softavg_s2_noinv".as_ptr(), 3, det, std::ptr::null_mut(),
        Some(s3_step), Some(s3_final), Some(s3_value), None, None,
    );
    PROBE_NOINV_RC.store(rc as i64, Relaxed);
    // probe 2: a legacy aggregate (accepted; its window use is tested in SQL)
    let rc = ffi::sqlite3_create_function_v2(
        db, c"softavg_s2_legacy".as_ptr(), 3, det, std::ptr::null_mut(),
        None, Some(s3_step), Some(s3_final), None,
    );
    PROBE_LEGACY_RC.store(rc as i64, Relaxed);
    type Scalar = unsafe extern "C" fn(*mut ffi::sqlite3_context, c_int, *mut *mut ffi::sqlite3_value);
    let scalars: [(&std::ffi::CStr, Scalar); 3] =
        [(c"ladder_stats", ladder_stats), (c"ladder_stats_reset", ladder_stats_reset), (c"ladder_probe", ladder_probe)];
    for (name, f) in scalars {
        let rc = ffi::sqlite3_create_function_v2(db, name.as_ptr(), 0, ffi::SQLITE_UTF8, std::ptr::null_mut(), Some(f), None, None, None);
        if rc != ffi::SQLITE_OK {
            return rc;
        }
    }
    ffi::SQLITE_OK
}

unsafe fn init(db: *mut ffi::sqlite3, _pz_err: *mut *mut c_char, p_api: *mut ffi::sqlite3_api_routines) -> c_int {
    if ffi::rusqlite_extension_init2(p_api).is_err() {
        return ffi::SQLITE_ERROR;
    }
    register(db)
}

#[no_mangle]
pub unsafe extern "C" fn sqlite3_extension_init(db: *mut ffi::sqlite3, pz_err: *mut *mut c_char, p_api: *mut ffi::sqlite3_api_routines) -> c_int {
    init(db, pz_err, p_api)
}

#[no_mangle]
pub unsafe extern "C" fn sqlite3_softavgablationsqlite_init(
    db: *mut ffi::sqlite3,
    pz_err: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    init(db, pz_err, p_api)
}

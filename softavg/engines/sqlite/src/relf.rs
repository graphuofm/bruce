//! relF baseline for experiments/anchored (not part of softagg-core):
//!   relf(value, score, d)      window-capable (xStep/xInverse/xValue/xFinal),
//!                              state relf_core::WinBand (FIFO frames: rows
//!                              dominated by a newer row by more than d are
//!                              dropped). SQLite's xInverse cannot refuse, so a
//!                              non-FIFO inverse is an SQL error.
//!   relf_agg(value, score, d)  GROUP BY only, state relf_core::Band
//! relf = AVG(value) over the rows with score >= MAX(score) - d.
//! NULL value/score rows are skipped; NaN is an error. States are heap
//! boxes owned by the aggregate context slot and dropped in xFinal.

use libsqlite3_sys as ffi;
use relf_core::{check_d, Band, WinBand};
use std::ffi::{c_char, c_int, CString};

#[repr(C)]
struct Slot<T> {
    p: *mut T,
}

unsafe fn slot<'a, T>(ctx: *mut ffi::sqlite3_context, alloc: bool) -> Option<&'a mut Slot<T>> {
    let n = if alloc { std::mem::size_of::<Slot<T>>() as c_int } else { 0 };
    let p = ffi::sqlite3_aggregate_context(ctx, n) as *mut Slot<T>;
    if p.is_null() {
        None
    } else {
        Some(&mut *p)
    }
}

unsafe fn err(ctx: *mut ffi::sqlite3_context, m: &str) {
    let c = CString::new(m).unwrap();
    ffi::sqlite3_result_error(ctx, c.as_ptr() as *const c_char, -1);
}

unsafe fn row(ctx: *mut ffi::sqlite3_context, argv: *mut *mut ffi::sqlite3_value) -> Option<(f64, f64, f64)> {
    let a = |i: usize| *argv.add(i);
    if ffi::sqlite3_value_type(a(0)) == ffi::SQLITE_NULL || ffi::sqlite3_value_type(a(1)) == ffi::SQLITE_NULL {
        return None;
    }
    let (v, s) = (ffi::sqlite3_value_double(a(0)), ffi::sqlite3_value_double(a(1)));
    if v.is_nan() || s.is_nan() {
        err(ctx, "relf: NaN value or score is not supported");
        return None;
    }
    if ffi::sqlite3_value_type(a(2)) == ffi::SQLITE_NULL {
        err(ctx, "relf: d must not be NULL");
        return None;
    }
    match check_d(ffi::sqlite3_value_double(a(2))) {
        Ok(d) => Some((s, v, d)),
        Err(m) => {
            err(ctx, &m);
            None
        }
    }
}

unsafe fn emit(ctx: *mut ffi::sqlite3_context, x: Option<f64>) {
    match x {
        Some(x) => ffi::sqlite3_result_double(ctx, x),
        None => ffi::sqlite3_result_null(ctx),
    }
}

unsafe extern "C" fn win_step(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v, d)) = row(ctx, argv) else { return };
    let Some(sl) = slot::<WinBand>(ctx, true) else { return ffi::sqlite3_result_error_nomem(ctx) };
    if sl.p.is_null() {
        sl.p = Box::into_raw(Box::new(WinBand::new(d)));
    } else if (*sl.p).d.to_bits() != d.to_bits() {
        return err(ctx, "relf: d must be constant within a group");
    }
    (*sl.p).push(s, v);
}

unsafe extern "C" fn win_inverse(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v, _)) = row(ctx, argv) else { return };
    match slot::<WinBand>(ctx, false) {
        Some(sl) if !sl.p.is_null() => {
            if (*sl.p).pop(s, v).is_err() {
                err(ctx, "relf: inverse of a row that is not the oldest in the frame (frame is not FIFO)");
            }
        }
        _ => err(ctx, "relf: inverse on an empty frame"),
    }
}

unsafe extern "C" fn win_value(ctx: *mut ffi::sqlite3_context) {
    match slot::<WinBand>(ctx, false) {
        Some(sl) if !sl.p.is_null() => emit(ctx, (*sl.p).value()),
        _ => ffi::sqlite3_result_null(ctx),
    }
}

unsafe extern "C" fn win_final(ctx: *mut ffi::sqlite3_context) {
    match slot::<WinBand>(ctx, false) {
        Some(sl) if !sl.p.is_null() => {
            let b = Box::from_raw(sl.p);
            sl.p = std::ptr::null_mut();
            emit(ctx, b.value());
        }
        _ => ffi::sqlite3_result_null(ctx),
    }
}

unsafe extern "C" fn agg_step(ctx: *mut ffi::sqlite3_context, _n: c_int, argv: *mut *mut ffi::sqlite3_value) {
    let Some((s, v, d)) = row(ctx, argv) else { return };
    let Some(sl) = slot::<Band>(ctx, true) else { return ffi::sqlite3_result_error_nomem(ctx) };
    if sl.p.is_null() {
        sl.p = Box::into_raw(Box::new(Band::new(d)));
    } else if (*sl.p).d.to_bits() != d.to_bits() {
        return err(ctx, "relf: d must be constant within a group");
    }
    (*sl.p).absorb(s, v);
}

unsafe extern "C" fn agg_final(ctx: *mut ffi::sqlite3_context) {
    match slot::<Band>(ctx, false) {
        Some(sl) if !sl.p.is_null() => {
            let b = Box::from_raw(sl.p);
            sl.p = std::ptr::null_mut();
            emit(ctx, b.finalize());
        }
        _ => ffi::sqlite3_result_null(ctx),
    }
}

pub unsafe fn register(db: *mut ffi::sqlite3) -> c_int {
    let det = ffi::SQLITE_UTF8 | ffi::SQLITE_DETERMINISTIC | ffi::SQLITE_INNOCUOUS;
    let rc = ffi::sqlite3_create_window_function(
        db, c"relf".as_ptr(), 3, det, std::ptr::null_mut(),
        Some(win_step), Some(win_final), Some(win_value), Some(win_inverse), None,
    );
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    ffi::sqlite3_create_function_v2(
        db, c"relf_agg".as_ptr(), 3, det, std::ptr::null_mut(), None, Some(agg_step), Some(agg_final), None,
    )
}

//! relF baseline for experiments/anchored (NOT part of the bruce_pg SQL
//! catalog: the functions below are raw V1 C functions, declared by
//! `experiments/anchored/pg/relf.sql` with `AS '$libdir/bruce_pg'`, so the
//! extension's install/upgrade scripts and catalog are unchanged).
//!
//! relf(v, s, d) = AVG(v) over the group's rows with s >= MAX(s) - d.
//! States come from `relf-core` (../relf_core): `Band` (GROUP BY, parallel
//! via serial/deserial), `WinBand` (moving aggregate; MINVFUNC returns NULL
//! -> PostgreSQL recomputes the frame -- if a departing row is not the
//! oldest held row), `IvmBand` (maintenance with headroom h; apply returns
//! NULL when the group must be rescanned).
//!
//! The states own Rust heap memory (their size is the band, not a
//! constant), so each one registers a reset callback on the memory context
//! it belongs to and is dropped when PostgreSQL resets that context.

use pgrx::prelude::*;
use pgrx::FromDatum;
use relf_core::{check_d, Band, IvmBand, WinBand};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::{agg_context, arg, arg_f64, arg_ptr, ret_f64, ret_null};

static RELF_GROUPS: AtomicU64 = AtomicU64::new(0);
static RELF_PEAK: AtomicU64 = AtomicU64::new(0);
static RELF_FINAL_SUM: AtomicU64 = AtomicU64::new(0);
static WIN_PUNTS: AtomicU64 = AtomicU64::new(0);
static WIN_PEAK: AtomicU64 = AtomicU64::new(0);
static WIN_SIZE_SUM: AtomicU64 = AtomicU64::new(0);
static WIN_PUSHES: AtomicU64 = AtomicU64::new(0);
static WIN_CUT_MOVES: AtomicU64 = AtomicU64::new(0);

macro_rules! finfo {
    ($($name:ident),* $(,)?) => {$(
        #[no_mangle]
        #[doc(hidden)]
        pub extern "C" fn $name() -> &'static pg_sys::Pg_finfo_record {
            const V1: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
            &V1
        }
    )*};
}
finfo!(
    pg_finfo_relf_sfunc,
    pg_finfo_relf_combine,
    pg_finfo_relf_serialize,
    pg_finfo_relf_deserialize,
    pg_finfo_relf_finalfn,
    pg_finfo_relf_msfunc,
    pg_finfo_relf_minvfunc,
    pg_finfo_relf_mfinalfn,
    pg_finfo_relf_ivm_sfunc,
    pg_finfo_relf_ivm_build_final,
    pg_finfo_relf_ivm_apply,
    pg_finfo_relf_ivm_value,
    pg_finfo_relf_ivm_held,
    pg_finfo_relf_stats,
    pg_finfo_relf_stats_reset,
);

unsafe extern "C-unwind" fn drop_box<T>(p: *mut ::core::ffi::c_void) {
    drop(Box::from_raw(p as *mut T));
}

/// Box `v` and drop it when `ctx` is reset or deleted.
unsafe fn boxed_in<T>(ctx: pg_sys::MemoryContext, v: T) -> *mut T {
    let p = Box::into_raw(Box::new(v));
    let cb = pg_sys::MemoryContextAlloc(ctx, std::mem::size_of::<pg_sys::MemoryContextCallback>())
        as *mut pg_sys::MemoryContextCallback;
    (*cb).func = Some(drop_box::<T>);
    (*cb).arg = p as *mut ::core::ffi::c_void;
    (*cb).next = std::ptr::null_mut();
    pg_sys::MemoryContextRegisterResetCallback(ctx, cb);
    p
}

unsafe fn bytea_of(bytes: &[u8]) -> pg_sys::Datum {
    let total = pg_sys::VARHDRSZ + bytes.len();
    let p = pg_sys::palloc(total) as *mut pg_sys::varlena;
    pgrx::set_varsize_4b(p, total as i32);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), (p as *mut u8).add(pg_sys::VARHDRSZ), bytes.len());
    pg_sys::Datum::from(p)
}

unsafe fn bytes_arg<'a>(fcinfo: pg_sys::FunctionCallInfo, i: usize) -> Option<&'a [u8]> {
    let raw = arg_ptr::<pg_sys::varlena>(fcinfo, i);
    if raw.is_null() {
        return None;
    }
    let v = pg_sys::pg_detoast_datum_packed(raw);
    let len = pgrx::varsize_any_exhdr(v);
    Some(std::slice::from_raw_parts(pgrx::vardata_any(v) as *const u8, len))
}

unsafe fn f8array(fcinfo: pg_sys::FunctionCallInfo, i: usize) -> Vec<Option<f64>> {
    let d = arg(fcinfo, i);
    <Vec<Option<f64>> as FromDatum>::from_polymorphic_datum(d.value, d.isnull, pg_sys::FLOAT8ARRAYOID)
        .unwrap_or_default()
}

#[inline]
fn live(v: f64, s: f64) -> bool {
    if v.is_nan() || s.is_nan() {
        error!("relf: NaN value or score is not supported");
    }
    true
}

#[inline]
unsafe fn d_arg(fcinfo: pg_sys::FunctionCallInfo, i: usize) -> f64 {
    match arg_f64(fcinfo, i) {
        None => error!("relf: d must not be NULL"),
        Some(d) => check_d(d).unwrap_or_else(|m| error!("{m}")),
    }
}

fn max_store(a: &AtomicU64, v: u64) {
    if v > a.load(Relaxed) {
        a.store(v, Relaxed);
    }
}

// ---------------------------------------------------------------- GROUP BY

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_sfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<Band>(fcinfo, 0);
    let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) else {
        return if st.is_null() { ret_null(fcinfo) } else { pg_sys::Datum::from(st) };
    };
    live(v, s);
    let st = if st.is_null() {
        let d = d_arg(fcinfo, 3);
        boxed_in(agg_context(fcinfo, "relf_sfunc"), Band::new(d))
    } else {
        st
    };
    (*st).absorb(s, v);
    pg_sys::Datum::from(st)
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_combine(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = arg_ptr::<Band>(fcinfo, 0);
    let b = arg_ptr::<Band>(fcinfo, 1);
    match (a.is_null(), b.is_null()) {
        (true, true) => ret_null(fcinfo),
        (false, true) => pg_sys::Datum::from(a),
        (true, false) => pg_sys::Datum::from(boxed_in(agg_context(fcinfo, "relf_combine"), (*b).clone())),
        (false, false) => {
            if (*a).d.to_bits() != (*b).d.to_bits() {
                error!("relf: cannot combine states with different d");
            }
            (*a).merge(&*b);
            pg_sys::Datum::from(a)
        }
    }
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_serialize(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<Band>(fcinfo, 0);
    bytea_of(&(*st).serialize())
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_deserialize(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let data = bytes_arg(fcinfo, 0).unwrap_or_else(|| error!("relf: NULL serialized state"));
    let Some(b) = Band::deserialize(data) else { error!("relf: malformed serialized state") };
    pg_sys::Datum::from(boxed_in(pg_sys::CurrentMemoryContext, b))
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_finalfn(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<Band>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    RELF_GROUPS.fetch_add(1, Relaxed);
    max_store(&RELF_PEAK, (*st).peak as u64);
    RELF_FINAL_SUM.fetch_add((*st).len() as u64, Relaxed);
    match (*st).finalize() {
        Some(x) => ret_f64(x),
        None => ret_null(fcinfo),
    }
}

// ---------------------------------------------------------------- windows

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_msfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let mut st = arg_ptr::<WinBand>(fcinfo, 0);
    if st.is_null() {
        let d = d_arg(fcinfo, 3);
        st = boxed_in(agg_context(fcinfo, "relf_msfunc"), WinBand::new(d));
    }
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        live(v, s);
        let before = (*st).cut_moves;
        (*st).push(s, v);
        WIN_CUT_MOVES.fetch_add((*st).cut_moves - before, Relaxed);
        let n = (*st).len() as u64;
        max_store(&WIN_PEAK, n);
        WIN_SIZE_SUM.fetch_add(n, Relaxed);
        WIN_PUSHES.fetch_add(1, Relaxed);
    }
    pg_sys::Datum::from(st)
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_minvfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<WinBand>(fcinfo, 0);
    if st.is_null() {
        WIN_PUNTS.fetch_add(1, Relaxed);
        return ret_null(fcinfo);
    }
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        let before = (*st).cut_moves;
        if (*st).pop(s, v).is_err() {
            WIN_PUNTS.fetch_add(1, Relaxed);
            return ret_null(fcinfo);
        }
        WIN_CUT_MOVES.fetch_add((*st).cut_moves - before, Relaxed);
    }
    pg_sys::Datum::from(st)
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_mfinalfn(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<WinBand>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    match (*st).value() {
        Some(x) => ret_f64(x),
        None => ret_null(fcinfo),
    }
}

// ---------------------------------------------------------------- maintenance

pub struct IvmBuild {
    d: f64,
    h: f64,
    rows: Vec<(f64, f64)>,
}

/// relf_ivm_sfunc(internal, v, s, d, h): collects the group's rows
/// (a rebuild reads the whole group anyway).
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_ivm_sfunc(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let mut st = arg_ptr::<IvmBuild>(fcinfo, 0);
    if st.is_null() {
        let d = d_arg(fcinfo, 3);
        let h = d_arg(fcinfo, 4);
        st = boxed_in(agg_context(fcinfo, "relf_ivm_sfunc"), IvmBuild { d, h, rows: Vec::new() });
    }
    if let (Some(v), Some(s)) = (arg_f64(fcinfo, 1), arg_f64(fcinfo, 2)) {
        live(v, s);
        (*st).rows.push((s, v));
    }
    pg_sys::Datum::from(st)
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_ivm_build_final(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let st = arg_ptr::<IvmBuild>(fcinfo, 0);
    if st.is_null() {
        return ret_null(fcinfo);
    }
    let b = IvmBand::build((*st).d, (*st).h, (*st).rows.iter().copied());
    bytea_of(&b.serialize())
}

fn pairs(vs: Vec<Option<f64>>, ss: Vec<Option<f64>>) -> Vec<(f64, f64)> {
    if vs.len() != ss.len() {
        error!("relf_ivm_apply: value and score arrays differ in length");
    }
    vs.into_iter()
        .zip(ss)
        .filter_map(|(v, s)| match (v, s) {
            (Some(v), Some(s)) => {
                live(v, s);
                Some((s, v))
            }
            _ => None,
        })
        .collect()
}

/// relf_ivm_apply(state bytea, del_v float8[], del_s float8[], ins_v float8[],
/// ins_s float8[]) -> bytea, or NULL when the group must be rescanned.
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_ivm_apply(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let Some(b) = bytes_arg(fcinfo, 0) else { return ret_null(fcinfo) };
    let Some(mut st) = IvmBand::deserialize(b) else { error!("relf: malformed incremental state") };
    let dels = pairs(f8array(fcinfo, 1), f8array(fcinfo, 2));
    let ins = pairs(f8array(fcinfo, 3), f8array(fcinfo, 4));
    match st.apply(&dels, &ins) {
        Ok(()) => bytea_of(&st.serialize()),
        Err(_) => ret_null(fcinfo),
    }
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_ivm_value(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let Some(b) = bytes_arg(fcinfo, 0) else { return ret_null(fcinfo) };
    let Some(st) = IvmBand::deserialize(b) else { error!("relf: malformed incremental state") };
    match st.value() {
        Some(x) => ret_f64(x),
        None => ret_null(fcinfo),
    }
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_ivm_held(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let Some(b) = bytes_arg(fcinfo, 0) else { return ret_null(fcinfo) };
    let Some(st) = IvmBand::deserialize(b) else { error!("relf: malformed incremental state") };
    pg_sys::Datum::from(st.held() as i64)
}

/// relf_stats() -> float8[]: [groups finalized, peak band (GROUP BY),
/// sum of final band sizes, window punts, window peak state, window
/// state-size sum over pushes, window pushes, window cutoff moves]
#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_stats(_fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let v: Vec<f64> = [
        &RELF_GROUPS, &RELF_PEAK, &RELF_FINAL_SUM, &WIN_PUNTS, &WIN_PEAK, &WIN_SIZE_SUM, &WIN_PUSHES,
        &WIN_CUT_MOVES,
    ]
    .iter()
    .map(|a| a.load(Relaxed) as f64)
    .collect();
    v.into_datum().unwrap()
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn relf_stats_reset(_fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    for a in [
        &RELF_GROUPS, &RELF_PEAK, &RELF_FINAL_SUM, &WIN_PUNTS, &WIN_PEAK, &WIN_SIZE_SUM, &WIN_PUSHES,
        &WIN_CUT_MOVES,
    ] {
        a.store(0, Relaxed);
    }
    pg_sys::Datum::from(0usize)
}

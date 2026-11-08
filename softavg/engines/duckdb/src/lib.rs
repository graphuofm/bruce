//! `softavg(value DOUBLE, score DOUBLE, eps DOUBLE) -> DOUBLE` as a DuckDB
//! loadable extension built on the C extension API (`duckdb_ext_api_v1`,
//! requested as "v1.2.0"; aggregate functions are part of that stable surface).
//!
//! The arithmetic is `softagg_core::ScalarAcc` (path dependency on
//! `../core`, shared with the PostgreSQL, SQLite and ClickHouse extensions);
//! `state.rs` adds only the host-level eps tag and NaN bit. The aggregate
//! contract:
//!   state_size / init  -> `SoftAvgState::new()`  (40 bytes, plain data)
//!   update             -> one call per input vector (<= 2048 rows): eps is
//!                         checked once per vector, then a tight loop over
//!                         `ScalarAcc::absorb` specialised per eps regime
//!   combine            -> `ScalarAcc::merge`  (parallel hash aggregate
//!                         partials, and the window segment tree)
//!   finalize           -> `u / z`, NULL when empty, NaN if a NaN was seen
//!
//! DuckDB's C-API wrapper flattens every input vector and the state vector
//! before calling `update`/`combine`/`finalize` (see
//! tools/reference/aggregate_function-c.cpp, `inputs[c].Flatten(count)`), so
//! a constant `eps` literal arrives as a dense column and plain pointer
//! indexing is correct. We register the function with SPECIAL null handling
//! and do the NULL bookkeeping ourselves so that a NULL `eps` on a live row
//! is an error rather than a silently dropped row.

pub mod state;
/// relF baseline (experiments/anchored); not part of softagg-core
pub mod relf;

use libduckdb_sys as ffi;
use state::{check_eps, SoftAvgError, SoftAvgState};
use std::ffi::CString;
use std::ptr;

const EXTENSION_NAME: &str = "softavg";
/// C API version requested from DuckDB at load time. The aggregate function
/// API has been stable since v1.1.0; v1.2.0 is what duckdb-rs's own
/// entrypoint macro requests by default.
const CAPI_VERSION: &str = "v1.2.0";

#[inline(always)]
unsafe fn row_valid(mask: *const u64, row: usize) -> bool {
    mask.is_null() || ((*mask.add(row >> 6) >> (row & 63)) & 1) == 1
}

unsafe fn set_error(info: ffi::duckdb_function_info, msg: &str) {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("softavg: internal error").unwrap());
    ffi::duckdb_aggregate_function_set_error(info, c.as_ptr());
}

#[inline(always)]
unsafe fn state_at<'a>(states: *mut ffi::duckdb_aggregate_state, i: usize) -> &'a mut SoftAvgState {
    &mut *((*states.add(i)) as *mut SoftAvgState)
}

unsafe extern "C" fn softavg_state_size(_info: ffi::duckdb_function_info) -> ffi::idx_t {
    std::mem::size_of::<SoftAvgState>() as ffi::idx_t
}

unsafe extern "C" fn softavg_init(_info: ffi::duckdb_function_info, state: ffi::duckdb_aggregate_state) {
    ptr::write(state as *mut SoftAvgState, SoftAvgState::new());
}

/// Hot loop: every row of the vector shares one validated `eps`. Inlined
/// into each call site below with `eps` a literal 0.0 / inf or the
/// vector's finite eps, so the regime branches inside `ScalarAcc::absorb`
/// fold away per regime. `MASKED = false` when neither value nor score has
/// a NULL in this vector (DuckDB hands us a NULL validity pointer then).
#[inline(always)]
unsafe fn absorb_uniform<const MASKED: bool>(
    states: *mut ffi::duckdb_aggregate_state,
    v: *const f64,
    s: *const f64,
    v_mask: *const u64,
    s_mask: *const u64,
    n: usize,
    eps: f64,
) -> Result<(), SoftAvgError> {
    for i in 0..n {
        if MASKED && !(row_valid(v_mask, i) && row_valid(s_mask, i)) {
            continue;
        }
        let st = state_at(states, i);
        st.bind_eps(eps)?;
        st.absorb_bound(*s.add(i), *v.add(i), eps);
    }
    Ok(())
}

#[inline(always)]
unsafe fn absorb_regime<const MASKED: bool>(
    states: *mut ffi::duckdb_aggregate_state,
    v: *const f64,
    s: *const f64,
    v_mask: *const u64,
    s_mask: *const u64,
    n: usize,
    eps: f64,
) -> Result<(), SoftAvgError> {
    if eps == 0.0 {
        absorb_uniform::<MASKED>(states, v, s, v_mask, s_mask, n, 0.0)
    } else if eps == f64::INFINITY {
        absorb_uniform::<MASKED>(states, v, s, v_mask, s_mask, n, f64::INFINITY)
    } else {
        absorb_uniform::<MASKED>(states, v, s, v_mask, s_mask, n, eps)
    }
}

/// Fallback: eps varies inside the vector (legal: it must only be constant
/// per group) or has NULLs. Per-row validation.
unsafe fn absorb_general(
    states: *mut ffi::duckdb_aggregate_state,
    v: *const f64,
    s: *const f64,
    e: *const f64,
    v_mask: *const u64,
    s_mask: *const u64,
    e_mask: *const u64,
    n: usize,
) -> Result<(), SoftAvgError> {
    for i in 0..n {
        // NULL value or NULL score: the row is ignored, exactly like AVG.
        if !(row_valid(v_mask, i) && row_valid(s_mask, i)) {
            continue;
        }
        // A live row must carry a temperature.
        if !row_valid(e_mask, i) {
            return Err(SoftAvgError::NullEps);
        }
        let eps = *e.add(i);
        check_eps(eps)?;
        let st = state_at(states, i);
        st.bind_eps(eps)?;
        st.absorb_bound(*s.add(i), *v.add(i), eps);
    }
    Ok(())
}

unsafe fn update_impl(input: ffi::duckdb_data_chunk, states: *mut ffi::duckdb_aggregate_state) -> Result<(), SoftAvgError> {
    let n = ffi::duckdb_data_chunk_get_size(input) as usize;
    if n == 0 {
        return Ok(());
    }
    let v_vec = ffi::duckdb_data_chunk_get_vector(input, 0);
    let s_vec = ffi::duckdb_data_chunk_get_vector(input, 1);
    let e_vec = ffi::duckdb_data_chunk_get_vector(input, 2);
    let v = ffi::duckdb_vector_get_data(v_vec) as *const f64;
    let s = ffi::duckdb_vector_get_data(s_vec) as *const f64;
    let e = ffi::duckdb_vector_get_data(e_vec) as *const f64;
    let v_mask = ffi::duckdb_vector_get_validity(v_vec) as *const u64;
    let s_mask = ffi::duckdb_vector_get_validity(s_vec) as *const u64;
    let e_mask = ffi::duckdb_vector_get_validity(e_vec) as *const u64;

    // One eps for the whole vector? (The usual case: a literal, flattened
    // by the C-API wrapper into n identical doubles; this scan vectorises.)
    if e_mask.is_null() {
        let e0 = *e;
        let eb = e0.to_bits();
        let es = std::slice::from_raw_parts(e, n);
        if es.iter().fold(true, |ok, x| ok & (x.to_bits() == eb)) {
            if v_mask.is_null() && s_mask.is_null() {
                // every row is live, so eps must be valid
                check_eps(e0)?;
                return absorb_regime::<false>(states, v, s, v_mask, s_mask, n, e0);
            }
            // some rows are NULL; eps is checked only if a live row exists
            let any_live = (0..n).any(|i| row_valid(v_mask, i) && row_valid(s_mask, i));
            if !any_live {
                return Ok(());
            }
            check_eps(e0)?;
            return absorb_regime::<true>(states, v, s, v_mask, s_mask, n, e0);
        }
    }
    absorb_general(states, v, s, e, v_mask, s_mask, e_mask, n)
}

unsafe extern "C" fn softavg_update(
    info: ffi::duckdb_function_info,
    input: ffi::duckdb_data_chunk,
    states: *mut ffi::duckdb_aggregate_state,
) {
    if let Err(err) = update_impl(input, states) {
        set_error(info, &err.to_string());
    }
}

/// `target[i] <- target[i] (+) source[i]`. Used by the parallel hash
/// aggregate (thread-local partials merged per group) and by the window
/// segment tree (inner nodes, and frame = combine of O(log n) nodes).
unsafe extern "C" fn softavg_combine(
    info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    target: *mut ffi::duckdb_aggregate_state,
    count: ffi::idx_t,
) {
    for i in 0..count as usize {
        let src = &*state_at(source, i);
        let dst = state_at(target, i);
        if let Err(err) = dst.merge(src) {
            set_error(info, &err.to_string());
            return;
        }
    }
}

unsafe extern "C" fn softavg_finalize(
    _info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    result: ffi::duckdb_vector,
    count: ffi::idx_t,
    offset: ffi::idx_t,
) {
    let data = ffi::duckdb_vector_get_data(result) as *mut f64;
    let mut mask: *mut u64 = ptr::null_mut();
    for i in 0..count as usize {
        let st = &*state_at(source, i);
        let row = offset as usize + i;
        match st.finalize() {
            Some(x) => *data.add(row) = x,
            None => {
                if mask.is_null() {
                    ffi::duckdb_vector_ensure_validity_writable(result);
                    mask = ffi::duckdb_vector_get_validity(result);
                }
                ffi::duckdb_validity_set_row_invalid(mask, row as ffi::idx_t);
            }
        }
    }
}

type UpdateFn = unsafe extern "C" fn(ffi::duckdb_function_info, ffi::duckdb_data_chunk, *mut ffi::duckdb_aggregate_state);

/// Register one DOUBLE^n -> DOUBLE aggregate over `SoftAvgState`.
unsafe fn register_fn(con: ffi::duckdb_connection, name: &str, n_params: usize, update: UpdateFn) -> Result<(), String> {
    let mut f = ffi::duckdb_create_aggregate_function();
    if f.is_null() {
        return Err("softavg: duckdb_create_aggregate_function returned NULL".into());
    }
    let cname = CString::new(name).unwrap();
    ffi::duckdb_aggregate_function_set_name(f, cname.as_ptr());
    let mut dbl = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE);
    for _ in 0..n_params {
        ffi::duckdb_aggregate_function_add_parameter(f, dbl); // value, score, eps
    }
    ffi::duckdb_aggregate_function_set_return_type(f, dbl);
    ffi::duckdb_destroy_logical_type(&mut dbl);
    ffi::duckdb_aggregate_function_set_functions(
        f,
        Some(softavg_state_size),
        Some(softavg_init),
        Some(update),
        Some(softavg_combine),
        Some(softavg_finalize),
    );
    // We see NULLs ourselves (skip NULL value/score, reject NULL eps).
    ffi::duckdb_aggregate_function_set_special_handling(f);
    let rc = ffi::duckdb_register_aggregate_function(con, f);
    ffi::duckdb_destroy_aggregate_function(&mut f);
    if rc != ffi::DuckDBSuccess {
        return Err(format!("softavg: duckdb_register_aggregate_function({name}) failed"));
    }
    Ok(())
}

unsafe fn register(con: ffi::duckdb_connection) -> Result<(), String> {
    register_fn(con, EXTENSION_NAME, 3, softavg_update)?;
    relf::register(con)?;
    #[cfg(feature = "diag")]
    diag::register(con)?;
    Ok(())
}

/// Diagnostic aggregates for attributing cost (built only with
/// `--features diag`; never in the shipped extension). They share the
/// state, combine and finalize of `softavg` and differ only in `update`.
#[cfg(feature = "diag")]
mod diag {
    use super::*;

    /// update does nothing: the cost of DuckDB's hash aggregate + the C-API
    /// wrapper (DataChunk construction, Flatten of all three inputs).
    unsafe extern "C" fn noop_update(_: ffi::duckdb_function_info, _: ffi::duckdb_data_chunk, _: *mut ffi::duckdb_aggregate_state) {}

    /// plain sum/count scatter, no exp: the cost of touching the states.
    unsafe extern "C" fn touch_update(_: ffi::duckdb_function_info, input: ffi::duckdb_data_chunk, states: *mut ffi::duckdb_aggregate_state) {
        let n = ffi::duckdb_data_chunk_get_size(input) as usize;
        let v = ffi::duckdb_vector_get_data(ffi::duckdb_data_chunk_get_vector(input, 0)) as *const f64;
        for i in 0..n {
            let st = state_at(states, i);
            st.acc.u += *v.add(i);
            st.acc.z += 1.0;
        }
    }

    /// softavg with eps fixed at 0.1 and only two inputs: the cost of the
    /// third (constant, flattened) input column.
    unsafe extern "C" fn two_arg_update(_: ffi::duckdb_function_info, input: ffi::duckdb_data_chunk, states: *mut ffi::duckdb_aggregate_state) {
        let n = ffi::duckdb_data_chunk_get_size(input) as usize;
        let v = ffi::duckdb_vector_get_data(ffi::duckdb_data_chunk_get_vector(input, 0)) as *const f64;
        let s = ffi::duckdb_vector_get_data(ffi::duckdb_data_chunk_get_vector(input, 1)) as *const f64;
        let _ = absorb_regime::<false>(states, v, s, ptr::null(), ptr::null(), n, 0.1);
    }

    pub unsafe fn register(con: ffi::duckdb_connection) -> Result<(), String> {
        register_fn(con, "softavg_diag_noop", 3, noop_update)?;
        register_fn(con, "softavg_diag_touch", 3, touch_update)?;
        register_fn(con, "softavg_diag_2arg", 2, two_arg_update)?;
        Ok(())
    }
}

unsafe fn init_internal(
    info: ffi::duckdb_extension_info,
    access: *const ffi::duckdb_extension_access,
) -> Result<bool, String> {
    // Like DUCKDB_EXTENSION_API_INIT: fetch the function-pointer struct and
    // populate libduckdb-sys's per-function slots.
    let have_api = ffi::duckdb_rs_extension_api_init(info, access, CAPI_VERSION).map_err(|e| e.to_string())?;
    if !have_api {
        // DuckDB has already recorded the version error on `info`.
        return Ok(false);
    }
    let get_database = (*access)
        .get_database
        .ok_or_else(|| "softavg: get_database is NULL in duckdb_extension_access".to_string())?;
    let db_ptr = get_database(info);
    if db_ptr.is_null() {
        return Ok(false);
    }
    let mut con: ffi::duckdb_connection = ptr::null_mut();
    if ffi::duckdb_connect(*db_ptr, &mut con) != ffi::DuckDBSuccess {
        return Err("softavg: failed to open a connection during extension load".into());
    }
    let res = register(con);
    ffi::duckdb_disconnect(&mut con);
    res.map(|_| true)
}

/// Entry point looked up by DuckDB as `<filebase>_init_c_api` for a
/// `softavg.duckdb_extension` file with ABI type C_STRUCT.
#[no_mangle]
pub unsafe extern "C" fn softavg_init_c_api(
    info: ffi::duckdb_extension_info,
    access: *const ffi::duckdb_extension_access,
) -> bool {
    match init_internal(info, access) {
        Ok(v) => v,
        Err(msg) => {
            if let Some(set_err) = (*access).set_error {
                let c = CString::new(msg).unwrap_or_else(|_| CString::new("softavg: load failed").unwrap());
                set_err(info, c.as_ptr());
            }
            false
        }
    }
}

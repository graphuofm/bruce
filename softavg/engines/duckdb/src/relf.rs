//! relF baseline for experiments/anchored: `relf(value, score, d)` =
//! AVG(value) over the group's rows with score >= MAX(score) - d.
//! State: a pointer to a heap `relf_core::Band` (the rows within d of the
//! running max; its size is data-dependent, not constant), freed by the
//! destructor callback. combine = band union + prune (parallel partials and
//! the window segment tree; DuckDB 1.5.6's C API has no inverse).
//! Not part of softagg-core; registered next to softavg in the same
//! extension so both run through identical host paths.

use libduckdb_sys as ffi;
use relf_core::{check_d, Band};
use std::ffi::CString;
use std::ptr;

type St = *mut Band;

#[inline(always)]
unsafe fn row_valid(mask: *const u64, row: usize) -> bool {
    mask.is_null() || ((*mask.add(row >> 6) >> (row & 63)) & 1) == 1
}

unsafe fn set_error(info: ffi::duckdb_function_info, msg: &str) {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("relf: internal error").unwrap());
    ffi::duckdb_aggregate_function_set_error(info, c.as_ptr());
}

#[inline(always)]
unsafe fn slot<'a>(states: *mut ffi::duckdb_aggregate_state, i: usize) -> &'a mut St {
    &mut *((*states.add(i)) as *mut St)
}

unsafe extern "C" fn relf_state_size(_info: ffi::duckdb_function_info) -> ffi::idx_t {
    std::mem::size_of::<St>() as ffi::idx_t
}

unsafe extern "C" fn relf_init(_info: ffi::duckdb_function_info, state: ffi::duckdb_aggregate_state) {
    ptr::write(state as *mut St, ptr::null_mut());
}

unsafe extern "C" fn relf_destroy(states: *mut ffi::duckdb_aggregate_state, count: ffi::idx_t) {
    for i in 0..count as usize {
        let s = slot(states, i);
        if !s.is_null() {
            drop(Box::from_raw(*s));
            *s = ptr::null_mut();
        }
    }
}

unsafe fn update_impl(input: ffi::duckdb_data_chunk, states: *mut ffi::duckdb_aggregate_state) -> Result<(), String> {
    let n = ffi::duckdb_data_chunk_get_size(input) as usize;
    let vv = ffi::duckdb_data_chunk_get_vector(input, 0);
    let sv = ffi::duckdb_data_chunk_get_vector(input, 1);
    let dv = ffi::duckdb_data_chunk_get_vector(input, 2);
    let v = ffi::duckdb_vector_get_data(vv) as *const f64;
    let s = ffi::duckdb_vector_get_data(sv) as *const f64;
    let d = ffi::duckdb_vector_get_data(dv) as *const f64;
    let vm = ffi::duckdb_vector_get_validity(vv) as *const u64;
    let sm = ffi::duckdb_vector_get_validity(sv) as *const u64;
    let dm = ffi::duckdb_vector_get_validity(dv) as *const u64;
    for i in 0..n {
        if !(row_valid(vm, i) && row_valid(sm, i)) {
            continue;
        }
        if !row_valid(dm, i) {
            return Err("relf: d must not be NULL".into());
        }
        let (x, y, di) = (*v.add(i), *s.add(i), *d.add(i));
        if x.is_nan() || y.is_nan() {
            return Err("relf: NaN value or score is not supported".into());
        }
        let st = slot(states, i);
        if st.is_null() {
            *st = Box::into_raw(Box::new(Band::new(check_d(di)?)));
        } else if (**st).d.to_bits() != di.to_bits() {
            return Err("relf: d must be constant within a group".into());
        }
        (**st).absorb(y, x);
    }
    Ok(())
}

unsafe extern "C" fn relf_update(
    info: ffi::duckdb_function_info,
    input: ffi::duckdb_data_chunk,
    states: *mut ffi::duckdb_aggregate_state,
) {
    if let Err(e) = update_impl(input, states) {
        set_error(info, &e);
    }
}

unsafe extern "C" fn relf_combine(
    info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    target: *mut ffi::duckdb_aggregate_state,
    count: ffi::idx_t,
) {
    for i in 0..count as usize {
        let src = *slot(source, i);
        if src.is_null() {
            continue;
        }
        let dst = slot(target, i);
        if dst.is_null() {
            *dst = Box::into_raw(Box::new((*src).clone()));
        } else {
            if (**dst).d.to_bits() != (*src).d.to_bits() {
                set_error(info, "relf: cannot combine states with different d");
                return;
            }
            (**dst).merge(&*src);
        }
    }
}

unsafe extern "C" fn relf_finalize(
    _info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    result: ffi::duckdb_vector,
    count: ffi::idx_t,
    offset: ffi::idx_t,
) {
    let data = ffi::duckdb_vector_get_data(result) as *mut f64;
    let mut mask: *mut u64 = ptr::null_mut();
    for i in 0..count as usize {
        let st = *slot(source, i);
        let row = offset as usize + i;
        match if st.is_null() { None } else { (*st).finalize() } {
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

pub unsafe fn register(con: ffi::duckdb_connection) -> Result<(), String> {
    let mut f = ffi::duckdb_create_aggregate_function();
    if f.is_null() {
        return Err("relf: duckdb_create_aggregate_function returned NULL".into());
    }
    let cname = CString::new("relf").unwrap();
    ffi::duckdb_aggregate_function_set_name(f, cname.as_ptr());
    let mut dbl = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE);
    for _ in 0..3 {
        ffi::duckdb_aggregate_function_add_parameter(f, dbl);
    }
    ffi::duckdb_aggregate_function_set_return_type(f, dbl);
    ffi::duckdb_destroy_logical_type(&mut dbl);
    ffi::duckdb_aggregate_function_set_functions(
        f,
        Some(relf_state_size),
        Some(relf_init),
        Some(relf_update),
        Some(relf_combine),
        Some(relf_finalize),
    );
    ffi::duckdb_aggregate_function_set_destructor(f, Some(relf_destroy));
    ffi::duckdb_aggregate_function_set_special_handling(f);
    let rc = ffi::duckdb_register_aggregate_function(con, f);
    ffi::duckdb_destroy_aggregate_function(&mut f);
    if rc != ffi::DuckDBSuccess {
        return Err("relf: duckdb_register_aggregate_function failed".into());
    }
    Ok(())
}

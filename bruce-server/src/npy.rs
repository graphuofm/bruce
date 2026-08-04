//! Minimal NPY (`.npy`) reader for key (embedding) matrices.
//!
//! Placement is deliberate: C1 keeps bruce-core pure-algorithm (no
//! IO), and bruce-query speaks the external interchange formats,
//! which are Arrow/Parquet only (C3). The npy file is not an
//! interchange format here — it is the encoder's native dump, read
//! once at server startup to attach embedding keys to a table
//! (`Table::attach_key_f32` / `attach_key_f64`). So the reader lives
//! in the server binary's crate.
//!
//! Supported subset (all the encoder pipeline emits): format v1.0 /
//! v2.0 / v3.0 headers, little-endian `<f4` / `<f8`, C order, 2-D
//! shape. Everything else is a descriptive error, never a panic.

use ndarray::Array2;

/// A 2-D matrix read from an `.npy` file at its stored dtype.
///
/// The dtype distinction is preserved on purpose: f32-born embeddings
/// attach as `KeyF32` (half the scan bytes; the M2 precision contract
/// keeps the fold in f64), f64 files attach as `KeyF64`.
#[derive(Debug)]
pub enum NpyMatrix {
    /// `<f4` payload.
    F32(Array2<f32>),
    /// `<f8` payload.
    F64(Array2<f64>),
}

impl NpyMatrix {
    /// (rows, cols) of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            NpyMatrix::F32(a) => (a.nrows(), a.ncols()),
            NpyMatrix::F64(a) => (a.nrows(), a.ncols()),
        }
    }
}

/// Read a 2-D `<f4`/`<f8` C-order `.npy` file.
pub fn read_npy_2d(path: impl AsRef<std::path::Path>) -> Result<NpyMatrix, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("read npy {}: {e}", path.display()))?;
    parse_npy_2d(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse an in-memory `.npy` byte image (the file-reading wrapper is
/// [`read_npy_2d`]; this split keeps the parser unit-testable).
pub fn parse_npy_2d(bytes: &[u8]) -> Result<NpyMatrix, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not an NPY file (bad magic)".into());
    }
    let major = bytes[6];
    let (header_len, header_start): (usize, usize) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated NPY v2/v3 header length".into());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => return Err(format!("unsupported NPY major version {v}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or("truncated NPY header")?;
    // The header dict is ASCII by spec (latin-1 in theory; the keys
    // we parse are ASCII either way).
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| format!("NPY header is not UTF-8: {e}"))?;

    let descr = dict_str(header, "descr").ok_or("NPY header missing 'descr'")?;
    let itemsize: usize = match descr.as_str() {
        "<f4" => 4,
        "<f8" => 8,
        d => {
            return Err(format!(
                "unsupported NPY dtype {d:?} (only little-endian '<f4'/'<f8')"
            ))
        }
    };
    if header.contains("'fortran_order': True") {
        return Err("Fortran-order NPY not supported (need C order)".into());
    }
    let shape = dict_shape(header)?;
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        s => return Err(format!("need a 2-D NPY, got shape {s:?}")),
    };

    let data = &bytes[header_end..];
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(itemsize))
        .ok_or("NPY shape overflows")?;
    if data.len() != expected {
        return Err(format!(
            "NPY payload is {n} bytes, shape ({rows}, {cols}) x {itemsize} needs {expected}",
            n = data.len()
        ));
    }

    Ok(match itemsize {
        4 => {
            let v: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            NpyMatrix::F32(
                Array2::from_shape_vec((rows, cols), v).map_err(|e| format!("shape npy: {e}"))?,
            )
        }
        _ => {
            let v: Vec<f64> = data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            NpyMatrix::F64(
                Array2::from_shape_vec((rows, cols), v).map_err(|e| format!("shape npy: {e}"))?,
            )
        }
    })
}

/// Extract a single-quoted string value for `key` from the header
/// dict literal, e.g. `'descr': '<f4'`.
fn dict_str(header: &str, key: &str) -> Option<String> {
    let pat = format!("'{key}':");
    let rest = &header[header.find(&pat)? + pat.len()..];
    let open = rest.find('\'')?;
    let rest = &rest[open + 1..];
    let close = rest.find('\'')?;
    Some(rest[..close].to_string())
}

/// Extract the `'shape': (a, b, ...)` tuple from the header dict.
fn dict_shape(header: &str) -> Result<Vec<usize>, String> {
    let pat = "'shape':";
    let rest = &header[header.find(pat).ok_or("NPY header missing 'shape'")? + pat.len()..];
    let open = rest.find('(').ok_or("NPY shape: no '('")?;
    let close = rest[open..].find(')').ok_or("NPY shape: no ')'")? + open;
    rest[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("NPY shape element {s:?}: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an NPY v1.0 byte image with the given header dict body
    /// and raw payload.
    fn npy_bytes(descr: &str, fortran: bool, shape: &str, payload: &[u8]) -> Vec<u8> {
        let dict = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
            if fortran { "True" } else { "False" }
        );
        // pad header (incl. trailing newline) so total preamble is a
        // multiple of 64, as numpy does
        let mut header = dict.into_bytes();
        let pre = 10 + header.len() + 1;
        let pad = (64 - pre % 64) % 64;
        header.resize(header.len() + pad, b' ');
        header.push(b'\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(&header);
        out.extend(payload);
        out
    }

    #[test]
    fn reads_f32_2d() {
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let payload: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let m = parse_npy_2d(&npy_bytes("<f4", false, "(2, 3)", &payload)).unwrap();
        match m {
            NpyMatrix::F32(a) => {
                assert_eq!(a.shape(), &[2, 3]);
                assert_eq!(a[[1, 2]], 6.0);
            }
            _ => panic!("expected f32"),
        }
    }

    #[test]
    fn reads_f64_2d() {
        let vals: Vec<f64> = vec![-1.5, 0.25, 7.0, 8.0];
        let payload: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let m = parse_npy_2d(&npy_bytes("<f8", false, "(2, 2)", &payload)).unwrap();
        match m {
            NpyMatrix::F64(a) => {
                assert_eq!(a.shape(), &[2, 2]);
                assert_eq!(a[[0, 1]], 0.25);
            }
            _ => panic!("expected f64"),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(parse_npy_2d(b"NOTNPY\x01\x00\x00\x00")
            .unwrap_err()
            .contains("magic"));
    }

    #[test]
    fn rejects_fortran_order() {
        let payload = [0u8; 16];
        let err = parse_npy_2d(&npy_bytes("<f4", true, "(2, 2)", &payload)).unwrap_err();
        assert!(err.contains("Fortran"), "{err}");
    }

    #[test]
    fn rejects_1d_shape() {
        let payload = [0u8; 16];
        let err = parse_npy_2d(&npy_bytes("<f4", false, "(4,)", &payload)).unwrap_err();
        assert!(err.contains("2-D"), "{err}");
    }

    #[test]
    fn rejects_wrong_payload_length() {
        let payload = [0u8; 12]; // (2,2) f32 needs 16
        let err = parse_npy_2d(&npy_bytes("<f4", false, "(2, 2)", &payload)).unwrap_err();
        assert!(err.contains("16"), "{err}");
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let payload = [0u8; 16];
        let err = parse_npy_2d(&npy_bytes("<i8", false, "(2, 1)", &payload)).unwrap_err();
        assert!(err.contains("dtype"), "{err}");
    }
}

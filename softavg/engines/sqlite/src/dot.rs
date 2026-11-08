//! Inner product over float32 vectors stored as BLOBs in sqlite-vec's
//! format: `dim * 4` bytes, little-endian IEEE-754 float32, no header.
//!
//! Products of two f32 are exact in f64 (24 + 24 < 53 mantissa bits), so
//! the only rounding is in the f64 summation; eight independent lanes
//! let the compiler vectorise it. The result is therefore at least as
//! accurate as a float32 dot product and deterministic for a given
//! build (fixed lane order).

#[derive(Debug, PartialEq, Eq)]
pub enum DotError {
    NotFloat32,
    DimMismatch,
}

#[inline]
fn f(b: &[u8]) -> f64 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64
}

pub fn dot_f32_blob(a: &[u8], b: &[u8]) -> Result<f64, DotError> {
    if a.len() % 4 != 0 || b.len() % 4 != 0 {
        return Err(DotError::NotFloat32);
    }
    if a.len() != b.len() {
        return Err(DotError::DimMismatch);
    }
    let mut acc = [0f64; 8];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for k in 0..8 {
            acc[k] += f(&x[4 * k..]) * f(&y[4 * k..]);
        }
    }
    let mut tail = 0.0;
    for (x, y) in ca.remainder().chunks_exact(4).zip(cb.remainder().chunks_exact(4)) {
        tail += f(x) * f(y);
    }
    Ok(((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn matches_f64_reference() {
        for dim in [0usize, 1, 7, 8, 9, 384, 385] {
            let a: Vec<f32> = (0..dim).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let got = dot_f32_blob(&blob(&a), &blob(&b)).unwrap();
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "dim={dim}");
        }
    }

    #[test]
    fn rejects_bad_blobs() {
        assert_eq!(dot_f32_blob(&[0; 5], &[0; 5]), Err(DotError::NotFloat32));
        assert_eq!(dot_f32_blob(&[0; 8], &[0; 4]), Err(DotError::DimMismatch));
    }
}

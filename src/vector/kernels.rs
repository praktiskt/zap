//! similarity kernels: scalar reference + AVX2/FMA fast path.
//!
//! The engine ships a portable binary (no target-cpu flags), so the vector
//! path selects the SIMD kernel at RUNTIME via `is_x86_feature_detected!`
//! and falls back to the scalar loop everywhere else (aarch64 images, old
//! x86). The scalar loop is also what LLVM auto-vectorizes when the
//! baseline ISA allows it.

/// dot product of two equal-length slices (extra tail of `a` is ignored)
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature detection above
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// squared euclidean distance of two equal-length slices
#[inline]
pub fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature detection above
            return unsafe { sq_dist_avx2(a, b) };
        }
    }
    sq_dist_scalar(a, b)
}

/// L2 norm
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    dot_scalar(a, a).max(0.0).sqrt()
}

/// scalar dot product (reference path; LLVM auto-vectorizes this loop)
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0;
    // 4-way unrolled: breaks the serial dependency chain so the FMA units
    // can pipeline (a single accumulator chains every multiply-add)
    while i + 4 <= n {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// scalar squared euclidean distance
#[inline]
pub fn sq_dist_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum256(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sq_dist_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    let mut sum = hsum256(acc);
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// horizontal sum of an AVX register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum);
    let pairs = _mm_add_ps(sum, shuf);
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_movehl_ps(pairs, pairs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32) * -0.25 + 1.0).collect();
        let want = dot_scalar(&a, &b);
        let got = dot(&a, &b);
        assert!((want - got).abs() < 1e-3, "{want} vs {got}");
    }

    #[test]
    fn sq_dist_matches_scalar() {
        let a: Vec<f32> = (0..21).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..21).map(|i| (i * i) as f32).collect();
        let want = sq_dist_scalar(&a, &b);
        let got = sq_dist(&a, &b);
        assert!((want - got).abs() < 1e-2, "{want} vs {got}");
    }

    #[test]
    fn norm_is_length() {
        let v = [3.0f32, 4.0];
        assert!((norm(&v) - 5.0).abs() < 1e-6);
    }
}

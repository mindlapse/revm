use crate::falcon::{
    error::FalconError,
    ntt_consts::{roots_for_size, SQRT_MINUS_ONE_MOD_Q},
    FALCON_N, FALCON_Q,
};

/// Forward NTT on 512 coefficients (in-place), iterative, & panic-free.
///
/// This follows the canonical Falcon forward-NTT ordering used by the reference convention
/// (conceptually: recursive split/merge), while remaining fully iterative:
/// 1) normalize input into canonical residues in `[0, q)`
/// 2) reorder into the recursion “leaf order” (bit-reversal)
/// 3) run bottom-up merges, writing interleaved outputs into a scratch buffer
#[inline]
pub(crate) fn ntt_forward_in_place(a: &mut [i16; 512]) -> Result<(), FalconError> {
    // Validate roots tables up-front so we fail before doing any input work.
    let stage_roots = forward_stage_roots()?;

    // Normalize input into [0, q) so we can safely cast to u16.
    normalize_in_place(a);

    // Iterative bottom-up merge in canonical Falcon ordering.
    ntt_forward_iterative_canonical_order(a, stage_roots)
}

/// Normalize all coefficients into canonical residues in [0, q).
/// Fixed bound loop; no allocation; no panics.
#[inline]
fn normalize_in_place(a: &mut [i16; 512]) {
    let q: i32 = FALCON_Q as i32;
    for x in a.iter_mut() {
        let mut v = (*x as i32) % q;
        if v < 0 {
            v += q;
        }
        *x = v as i16;
    }
}

const N: usize = 512;
const STAGE_MS: [usize; 8] = [4, 8, 16, 32, 64, 128, 256, 512];

#[inline]
const fn bitrev9_usize(mut x: usize) -> usize {
    // Reverse the low 9 bits of x (since n=512).
    let mut r = 0usize;
    let mut i = 0usize;
    while i < 9 {
        r = (r << 1) | (x & 1);
        x >>= 1;
        i += 1;
    }
    r
}

// Constant-time-ish and branch-free at runtime: computed entirely at compile time.
const BITREV_9: [usize; N] = {
    let mut t = [0usize; N];
    let mut i = 0usize;
    while i < N {
        t[i] = bitrev9_usize(i);
        i += 1;
    }
    t
};

#[inline]
fn forward_stage_roots() -> Result<[&'static [u16]; 8], FalconError> {
    let r4 = roots_for_size(4).ok_or(FalconError::InvalidNttConstants)?;
    let r8 = roots_for_size(8).ok_or(FalconError::InvalidNttConstants)?;
    let r16 = roots_for_size(16).ok_or(FalconError::InvalidNttConstants)?;
    let r32 = roots_for_size(32).ok_or(FalconError::InvalidNttConstants)?;
    let r64 = roots_for_size(64).ok_or(FalconError::InvalidNttConstants)?;
    let r128 = roots_for_size(128).ok_or(FalconError::InvalidNttConstants)?;
    let r256 = roots_for_size(256).ok_or(FalconError::InvalidNttConstants)?;
    let r512 = roots_for_size(512).ok_or(FalconError::InvalidNttConstants)?;

    // Ensure all indexed accesses are provably in-bounds even if constants are corrupted.
    if r4.len() != 4
        || r8.len() != 8
        || r16.len() != 16
        || r32.len() != 32
        || r64.len() != 64
        || r128.len() != 128
        || r256.len() != 256
        || r512.len() != 512
    {
        return Err(FalconError::InvalidNttConstants);
    }

    Ok([r4, r8, r16, r32, r64, r128, r256, r512])
}

#[inline]
fn ntt_forward_iterative_canonical_order(
    a: &mut [i16; 512],
    stage_roots: [&'static [u16]; 8],
) -> Result<(), FalconError> {
    // Input is already normalized into [0, q) by the caller.
    // We'll use two stack buffers and swap references each stage (still iterative).
    //
    // No-panics narrative:
    // - All loops have fixed bounds under N=512.
    // - Root slices were validated to have length m for each stage.
    // - All indices are derived from fixed ranges and validated slice lengths.
    let mut buf0 = [0u16; N];
    let mut buf1 = [0u16; N];

    // Put input into the same “leaf order” produced by recursive split(): bit-reversal.
    for i in 0..N {
        buf0[BITREV_9[i]] = a[i] as u16;
    }

    let mut src: &mut [u16; N] = &mut buf0;
    let mut dst: &mut [u16; N] = &mut buf1;

    // Base case (m = 2): [u + sqr1*v, u - sqr1*v]
    for base in (0..N).step_by(2) {
        let u = src[base];
        let v = src[base + 1];
        let t = mul_mod_q(v, SQRT_MINUS_ONE_MOD_Q);
        dst[base] = add_mod_q(u, t);
        dst[base + 1] = sub_mod_q(u, t);
    }
    std::mem::swap(&mut src, &mut dst);

    // Bottom-up merges for m = 4, 8, ..., 512.
    for (stage_idx, &m) in STAGE_MS.iter().enumerate() {
        let half = m / 2;
        let w = stage_roots[stage_idx];

        for base in (0..N).step_by(m) {
            for i in 0..half {
                let u = src[base + i];
                let v = src[base + half + i];

                // Canonical Falcon convention: for size m, twiddle index is 2*i.
                let tw = w[2 * i];
                let t = mul_mod_q(v, tw);

                // Interleaved output indices as in merge_ntt.
                let out = base + (i << 1);
                dst[out] = add_mod_q(u, t);
                dst[out + 1] = sub_mod_q(u, t);
            }
        }

        std::mem::swap(&mut src, &mut dst);
    }

    for i in 0..N {
        a[i] = src[i] as i16;
    }

    Ok(())
}

#[inline]
fn add_mod_q(a: u16, b: u16) -> u16 {
    let q = FALCON_Q as u32;
    let mut s = a as u32 + b as u32;
    if s >= q {
        s -= q;
    }
    s as u16
}

#[inline]
fn sub_mod_q(a: u16, b: u16) -> u16 {
    let q = FALCON_Q as u32;
    let aa = a as u32;
    let bb = b as u32;
    if aa >= bb {
        (aa - bb) as u16
    } else {
        (aa + q - bb) as u16
    }
}

#[inline]
fn mul_mod_q(a: u16, b: u16) -> u16 {
    // Safe: (q-1)^2 < 2^28, no u32 overflow.
    // NOTE: This is intentionally reference-simple (`% q`) for correctness review.
    // Swap to Montgomery/Barrett reduction later if profiling shows this is a bottleneck.
    let q = FALCON_Q as u32;
    ((a as u32 * b as u32) % q) as u16
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::falcon::FALCON_Q;
    use serde_json;

    #[test]
    fn test_add_mod_q_zero_identity_and_wrap_cases() {
        let q = FALCON_Q as u16;

        // identity
        assert_eq!(add_mod_q(0, 0), 0);
        assert_eq!(add_mod_q(0, 1), 1);
        assert_eq!(add_mod_q(1, 0), 1);

        // no wrap
        assert_eq!(add_mod_q(2, 3), 5);

        // exact wrap at q
        assert_eq!(add_mod_q(q - 1, 1), 0);
        assert_eq!(add_mod_q(1, q - 1), 0);

        // near wrap
        assert_eq!(add_mod_q(q - 2, 1), q - 1);
        assert_eq!(add_mod_q(q - 2, 2), 0);

        // double max (requires single subtraction only)
        // (q-1) + (q-1) = 2q - 2 -> q - 2
        assert_eq!(add_mod_q(q - 1, q - 1), q - 2);
    }

    #[test]
    fn test_add_mod_q_commutative_on_representative_values() {
        let q = FALCON_Q as u16;
        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];

        for &a in &samples {
            for &b in &samples {
                assert_eq!(add_mod_q(a, b), add_mod_q(b, a));
            }
        }
    }

    #[test]
    fn test_add_mod_q_matches_reference_for_representative_values() {
        let q = FALCON_Q as u32;
        let samples: [u16; 10] = [
            0,
            1,
            2,
            3,
            17,
            123,
            (FALCON_Q / 2) as u16,
            (FALCON_Q - 3) as u16,
            (FALCON_Q - 2) as u16,
            (FALCON_Q - 1) as u16,
        ];

        for &a in &samples {
            for &b in &samples {
                let got = add_mod_q(a, b) as u32;
                let want = ((a as u32 + b as u32) % q) as u32;
                assert_eq!(got, want, "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_add_mod_q_output_is_in_range_for_representative_values() {
        let q = FALCON_Q as u16;
        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];

        for &a in &samples {
            for &b in &samples {
                let r = add_mod_q(a, b);
                assert!(r < q, "result out of range: a={a}, b={b}, r={r}, q={q}");
            }
        }
    }


    #[test]
    fn test_sub_mod_q_zero_identity_and_wrap_cases() {
        let q = FALCON_Q as u16;

        // identity / basic
        assert_eq!(sub_mod_q(0, 0), 0);
        assert_eq!(sub_mod_q(1, 0), 1);

        // no wrap (aa >= bb)
        assert_eq!(sub_mod_q(5, 3), 2);
        assert_eq!(sub_mod_q(q - 1, 1), q - 2);

        // wrap (aa < bb)
        assert_eq!(sub_mod_q(0, 1), q - 1);
        assert_eq!(sub_mod_q(1, 2), q - 1);
        assert_eq!(sub_mod_q(2, q - 1), 3);

        // edge wrap: 0 - (q-1) = 1
        assert_eq!(sub_mod_q(0, q - 1), 1);

        // exact equal
        assert_eq!(sub_mod_q(q - 1, q - 1), 0);
    }

    #[test]
    fn test_sub_mod_q_matches_reference_for_representative_values() {
        let q = FALCON_Q as u32;

        let samples: [u16; 10] = [
            0,
            1,
            2,
            3,
            17,
            123,
            (FALCON_Q / 2) as u16,
            (FALCON_Q - 3) as u16,
            (FALCON_Q - 2) as u16,
            (FALCON_Q - 1) as u16,
        ];

        for &a in &samples {
            for &b in &samples {
                let got = sub_mod_q(a, b) as u32;
                // Reference: (a - b) mod q, computed safely in u32.
                let want = ((a as u32 + q) - b as u32) % q;
                assert_eq!(got, want, "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_sub_mod_q_is_additive_inverse_via_add_mod_q_on_representative_values() {
        let q = FALCON_Q as u16;

        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];

        for &a in &samples {
            for &b in &samples {
                // (a - b) + b == a  (mod q)
                let r = add_mod_q(sub_mod_q(a, b), b);
                assert_eq!(r, a, "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_sub_mod_q_output_is_in_range_for_representative_values() {
        let q = FALCON_Q as u16;
        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];

        for &a in &samples {
            for &b in &samples {
                let r = sub_mod_q(a, b);
                assert!(r < q, "result out of range: a={a}, b={b}, r={r}, q={q}");
            }
        }
    }


    #[test]
    fn test_mul_mod_q_zero_and_identity_cases() {
        let q = FALCON_Q as u16;

        assert_eq!(mul_mod_q(0, 0), 0);
        assert_eq!(mul_mod_q(0, 1), 0);
        assert_eq!(mul_mod_q(1, 0), 0);

        assert_eq!(mul_mod_q(1, 1), 1);
        assert_eq!(mul_mod_q(1, 2), 2);
        assert_eq!(mul_mod_q(2, 1), 2);

        // a * 0 == 0, a * 1 == a (for representative values)
        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];
        for &a in &samples {
            assert_eq!(mul_mod_q(a, 0), 0, "a={a}");
            assert_eq!(mul_mod_q(0, a), 0, "a={a}");
            assert_eq!(mul_mod_q(a, 1), a % q, "a={a}");
            assert_eq!(mul_mod_q(1, a), a % q, "a={a}");
        }
    }

    #[test]
    fn test_mul_mod_q_wrap_cases_near_modulus() {
        let q = FALCON_Q as u16;

        // (q-1) * (q-1) = 1 mod q
        assert_eq!(mul_mod_q(q - 1, q - 1), 1);

        // (q-1) * x = -x mod q = q - x (for x != 0)
        assert_eq!(mul_mod_q(q - 1, 1), q - 1);
        assert_eq!(mul_mod_q(q - 1, 2), q - 2);
        assert_eq!(mul_mod_q(q - 1, q - 2), 2);

        // (q-2) * (q-2) = 4 mod q
        assert_eq!(mul_mod_q(q - 2, q - 2), 4);

        // (q-2) * (q-1) = 2 mod q
        assert_eq!(mul_mod_q(q - 2, q - 1), 2);
        assert_eq!(mul_mod_q(q - 1, q - 2), 2);
    }

    #[test]
    fn test_mul_mod_q_commutative_on_representative_values() {
        let q = FALCON_Q as u16;
        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];

        for &a in &samples {
            for &b in &samples {
                assert_eq!(mul_mod_q(a, b), mul_mod_q(b, a), "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_mul_mod_q_distributes_over_add_mod_q_on_small_representative_values() {
        // Check: a*(b+c) == a*b + a*c (mod q)
        // Use smaller samples to keep test compact and readable.
        let q = FALCON_Q as u16;
        let a_samples: [u16; 6] = [0, 1, 2, 17, q - 2, q - 1];
        let bc_samples: [u16; 6] = [0, 1, 2, 3, 17, q - 1];

        for &a in &a_samples {
            for &b in &bc_samples {
                for &c in &bc_samples {
                    let left = mul_mod_q(a, add_mod_q(b, c));
                    let right = add_mod_q(mul_mod_q(a, b), mul_mod_q(a, c));
                    assert_eq!(left, right, "a={a}, b={b}, c={c}");
                }
            }
        }
    }

    #[test]
    fn test_mul_mod_q_matches_reference_for_representative_values() {
        let q = FALCON_Q as u32;
        let samples: [u16; 10] = [
            0,
            1,
            2,
            3,
            17,
            123,
            (FALCON_Q / 2) as u16,
            (FALCON_Q - 3) as u16,
            (FALCON_Q - 2) as u16,
            (FALCON_Q - 1) as u16,
        ];

        for &a in &samples {
            for &b in &samples {
                let got = mul_mod_q(a, b) as u32;
                let want = ((a as u32 * b as u32) % q) as u32;
                assert_eq!(got, want, "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_mul_mod_q_output_is_in_range_for_representative_values() {
        let q = FALCON_Q as u16;
        let samples: [u16; 8] = [0, 1, 2, 17, q / 2, q - 2, q - 1, 123];

        for &a in &samples {
            for &b in &samples {
                let r = mul_mod_q(a, b);
                assert!(r < q, "result out of range: a={a}, b={b}, r={r}, q={q}");
            }
        }
    }

    #[test]
    fn test_normalize_in_place_basic_and_wrap_cases() {
        let q = FALCON_Q as i16;

        let mut a = [0i16; 512];
        a[0] = 0;
        a[1] = 1;
        a[2] = -1;
        a[3] = q;
        a[4] = q + 1;
        a[5] = -q;
        a[6] = -q + 1;
        a[7] = (FALCON_Q - 1) as i16;
        a[8] = -((FALCON_Q - 1) as i16);

        normalize_in_place(&mut a);

        assert_eq!(a[0], 0);
        assert_eq!(a[1], 1);
        assert_eq!(a[2], q - 1);

        assert_eq!(a[3], 0);
        assert_eq!(a[4], 1);
        assert_eq!(a[5], 0);
        assert_eq!(a[6], 1);

        assert_eq!(a[7], (FALCON_Q - 1) as i16);
        assert_eq!(a[8], 1);
    }

    #[test]
    fn test_normalize_in_place_outputs_are_in_range() {
        let q = FALCON_Q as i16;

        let mut a = [0i16; 512];
        // Populate a variety of values, including negatives.
        a[0] = 0;
        a[1] = 1;
        a[2] = -1;
        a[3] = 17;
        a[4] = -17;
        a[5] = (FALCON_Q / 2) as i16;
        a[6] = -((FALCON_Q / 2) as i16);
        a[7] = (FALCON_Q - 1) as i16;
        a[8] = -((FALCON_Q - 1) as i16);

        // Also include extremes of i16 to ensure behavior stays sane.
        a[9] = i16::MIN;
        a[10] = i16::MAX;

        normalize_in_place(&mut a);

        for (idx, &x) in a.iter().enumerate() {
            assert!(
                x >= 0 && x < q,
                "out of range at idx={idx}: x={x}, q={q}"
            );
        }
    }

    #[test]
    fn test_normalize_in_place_matches_reference_on_representative_values() {
        let q = FALCON_Q as i32;

        let mut a = [0i16; 512];
        let samples: [i16; 12] = [
            0,
            1,
            -1,
            2,
            -2,
            17,
            -17,
            (FALCON_Q / 2) as i16,
            -((FALCON_Q / 2) as i16),
            (FALCON_Q - 1) as i16,
            -((FALCON_Q - 1) as i16),
            FALCON_Q as i16,
        ];

        for (i, &v) in samples.iter().enumerate() {
            a[i] = v;
        }

        normalize_in_place(&mut a);

        for (i, &orig) in samples.iter().enumerate() {
            let got = a[i] as i32;
            let want = {
                let mut t = (orig as i32) % q;
                if t < 0 {
                    t += q;
                }
                t
            };
            assert_eq!(got, want, "idx={i}, orig={orig}");
        }
    }

    #[test]
    fn test_normalize_in_place_is_idempotent() {
        let mut a = [0i16; 512];

        // Fill with a repeating pattern including negatives and near-q values.
        // This stays within i16 comfortably.
        for i in 0..512 {
            a[i] = match i % 6 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => 17,
                4 => -17,
                _ => (FALCON_Q - 1) as i16,
            };
        }

        normalize_in_place(&mut a);
        let once = a;

        normalize_in_place(&mut a);
        assert_eq!(a, once);
    }

    #[test]
    fn test_normalize_in_place_preserves_congruence_mod_q_on_selected_indices() {
        // For selected indices, check: x and x + k*q normalize the same (within i16 limits).
        let q = FALCON_Q as i16;

        let mut a1 = [0i16; 512];
        let mut a2 = [0i16; 512];

        // Pick a few indices and values that won't overflow i16 when adding +/-2q.
        let cases: [(usize, i16, i16); 6] = [
            (0, 0, 2),
            (1, 1, -2),
            (2, -1, 1),
            (3, 17, -1),
            (4, -17, 2),
            (5, (FALCON_Q / 2) as i16, -2),
        ];

        for &(idx, x, k) in &cases {
            a1[idx] = x;
            a2[idx] = x.wrapping_add(k.wrapping_mul(q));
        }

        normalize_in_place(&mut a1);
        normalize_in_place(&mut a2);

        for &(idx, _, _) in &cases {
            assert_eq!(a1[idx], a2[idx], "idx={idx}");
        }
    }



    fn assert_all_in_range(a: &[i16; 512]) {
        let q = FALCON_Q as i16;
        for (i, &x) in a.iter().enumerate() {
            assert!(
                x >= 0 && x < q,
                "coefficient out of range at idx={i}: x={x}, q={q}"
            );
        }
    }


    #[test]
    fn test_ntt_forward_in_place_matches_manual_composition_on_dirty_input() {
        let mut input = [0i16; 512];

        // A deliberately "dirty" mix: negatives, >q values, and i16 extremes.
        // This is exactly what normalize_in_place is meant to tame.
        for i in 0..512 {
            input[i] = match i % 9 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => FALCON_Q as i16,
                4 => (FALCON_Q as i16) + 1,
                5 => -((FALCON_Q as i16)),
                6 => -((FALCON_Q as i16)) + 1,
                7 => i16::MIN,
                _ => i16::MAX,
            };
        }

        // Wrapper result
        let mut got = input;
        ntt_forward_in_place(&mut got).expect("ntt_forward_in_place failed");
        assert_all_in_range(&got);

        // Normalization should not affect the result (the wrapper normalizes internally).
        let mut want = input;
        normalize_in_place(&mut want);
        ntt_forward_in_place(&mut want).expect("ntt_forward_in_place failed (normalized)");
        assert_all_in_range(&want);

        assert_eq!(got, want);
    }

    #[test]
    fn test_ntt_forward_in_place_equals_core_on_already_normalized_input() {
        let mut input = [0i16; 512];

        // Deterministic normalized pattern in [0, q)
        for i in 0..512 {
            input[i] = ((i * 17 + 123) % (FALCON_Q as usize)) as i16;
        }

        // Wrapper result
        let mut got = input;
        ntt_forward_in_place(&mut got).expect("ntt_forward_in_place failed");
        assert_all_in_range(&got);

        // Explicitly normalizing first should not change anything.
        let mut want = input;
        normalize_in_place(&mut want);
        ntt_forward_in_place(&mut want).expect("ntt_forward_in_place failed (explicit normalize)");
        assert_all_in_range(&want);

        assert_eq!(got, want);
    }

    #[test]
    fn test_ntt_forward_in_place_is_deterministic_on_dirty_input() {
        let mut input = [0i16; 512];

        for i in 0..512 {
            // Another dirty-but-deterministic pattern.
            // Keep arithmetic in i32 to avoid any accidental overflow assumptions.
            let v = (i as i32 * 257 - 999) as i32;
            input[i] = match i % 4 {
                0 => v as i16,
                1 => (v + FALCON_Q as i32) as i16,
                2 => (v - FALCON_Q as i32) as i16,
                _ => (v + 3 * FALCON_Q as i32) as i16,
            };
        }

        let mut a1 = input;
        let mut a2 = input;

        ntt_forward_in_place(&mut a1).expect("ntt_forward_in_place failed (a1)");
        ntt_forward_in_place(&mut a2).expect("ntt_forward_in_place failed (a2)");

        assert_eq!(a1, a2);
        assert_all_in_range(&a1);
    }

    #[test]
    fn test_ntt_forward_in_place_output_is_in_range_for_varied_input() {
        let mut input = [0i16; 512];

        // Mixed values in and out of range (but within i16)
        for i in 0..512 {
            let v = (i as i32 * 19 - 42) as i32;
            input[i] = (v % 32768) as i16;
        }

        ntt_forward_in_place(&mut input).expect("ntt_forward_in_place failed");
        assert_all_in_range(&input);
    }

    #[test]
    fn test_ntt_forward_in_place_matches_kat512_vectors() {
        // KAT format: outer array of pairs [input[512], expected_output[512]].
        // Inputs may be "dirty" (negatives / out-of-range), since the wrapper normalizes.
        const KAT_JSON: &str = include_str!("ntt_KAT512.json");

        let pairs: Vec<[Vec<i32>; 2]> =
            serde_json::from_str(KAT_JSON).expect("failed to parse ntt_KAT512.json");
        assert!(!pairs.is_empty(), "ntt_KAT512.json contained no vectors");

        let q_i32 = FALCON_Q as i32;

        for (case_idx, [input_vec, expected_vec]) in pairs.into_iter().enumerate() {
            assert_eq!(
                input_vec.len(),
                FALCON_N,
                "case {case_idx}: input length != {FALCON_N}"
            );
            assert_eq!(
                expected_vec.len(),
                FALCON_N,
                "case {case_idx}: expected length != {FALCON_N}"
            );

            let mut a = [0i16; 512];
            for (i, x) in input_vec.into_iter().enumerate() {
                assert!(
                    x >= i16::MIN as i32 && x <= i16::MAX as i32,
                    "case {case_idx}, idx {i}: input out of i16 range: {x}"
                );
                a[i] = x as i16;
            }

            if let Err(e) = ntt_forward_in_place(&mut a) {
                assert!(false, "case {case_idx}: ntt_forward_in_place returned error: {e:?}");
            }

            for i in 0..FALCON_N {
                let want = expected_vec[i];
                assert!(
                    (0..q_i32).contains(&want),
                    "case {case_idx}, idx {i}: expected out of range: want={want}, q={q_i32}"
                );

                let got = a[i] as i32;
                assert!(
                    (0..q_i32).contains(&got),
                    "case {case_idx}, idx {i}: got out of range: got={got}, q={q_i32}"
                );

                assert_eq!(got, want, "case {}, idx {}: mismatch", case_idx, i);
            }
        }
    }

}
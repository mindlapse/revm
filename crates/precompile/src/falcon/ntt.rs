use crate::falcon::{
    error::FalconError,
    ntt_consts::{roots_for_size, SQRT_MINUS_ONE_MOD_Q},
    FALCON_Q,
};

// Falcon-512 parameter set: n = 512, q = 12289.
const N: usize = 512;
const NSTAGES: usize = 8;

const Q_I32: i32 = FALCON_Q as i32;

// inv(2) mod 12289 = 6145
const INV2_MOD_Q: u16 = 6145;

/// Forward NTT on 512 coefficients (in-place; uses fixed-size scratch buffers, no heap),
/// iterative, & panic-free.
///
/// This implements the canonical Falcon forward-NTT output ordering (split/merge “leaf order”),
/// while remaining fully iterative:
/// 1) normalize input into canonical residues in `[0, q)`
/// 2) reorder into the recursion “leaf order” (bit-reversal)
/// 3) run bottom-up merges, writing interleaved outputs into a scratch buffer
///
/// Observable contract: each merge writes `(u + t, u - t)` to consecutive even/odd indices.
///
/// Convenience: constructs stage tables; prefer `_with_stages` in hot paths.
pub(crate) fn ntt(a: &mut [i16; 512]) -> Result<(), FalconError> {
    // Cheap deterministic preflight.
    // Intentionally repeated here (no globals/atomics); hot paths should hoist this via
    // `ntt_with_stages`.
    let stages = forward_stages()?;
    ntt_with_stages(a, &stages)
}

/// Same as `ntt`, but takes prevalidated stage tables so callers can avoid
/// rebuilding them per invocation.
///
/// `stages` MUST be the value returned by `forward_stages()` for this build.
#[inline]
pub(crate) fn ntt_with_stages(
    a: &mut [i16; 512],
    stages: &ForwardStages,
) -> Result<(), FalconError> {
    // Normalize input into [0, q) so we can safely cast to u16.
    // After normalization all coefficients are in [0, q) ⊂ [0, 2^15), so `as u16` is
    // value-preserving.
    normalize_in_place(a);

    ntt_with_stages_and_normalized(a, stages)
}

/// Same as `ntt_with_stages`, but assumes inputs are already normalized into
/// `[0, q)`.
///
/// Contract:
/// - All coefficients MUST be in `[0, q)`.
/// - Output is in the canonical Falcon NTT ordering produced by this module.
#[inline]
pub(crate) fn ntt_with_stages_and_normalized(
    a: &mut [i16; 512],
    stages: &ForwardStages,
) -> Result<(), FalconError> {
    debug_assert!(
        a.iter().all(|&x| (x as i32) >= 0 && (x as i32) < Q_I32),
        "ntt_with_stages_and_normalized: input not in [0,q)"
    );

    // Iterative bottom-up merge in canonical Falcon ordering.
    ntt_with_stages_and_normalized_inner(a, stages)
}

/// Normalize all coefficients into canonical residues in [0, q).
/// Fixed bound loop; no allocation; no panics.
#[inline]
pub(crate) fn normalize_in_place(a: &mut [i16; 512]) {
    let q: i32 = FALCON_Q as i32;
    for x in a.iter_mut() {
        // Normalization is outside the NTT hot loop; a per-coefficient `% q` is acceptable
        // and keeps the signed wrap handling obviously correct.
        *x = (*x as i32).rem_euclid(q) as i16;
    }
}

#[derive(Clone, Copy)]
struct Sealed;

/// Opaque stage tables for forward NTT; validated by `forward_stages()`.
#[derive(Clone, Copy)]
pub(crate) struct ForwardStages {
    roots: [&'static [u16]; NSTAGES],
    _sealed: Sealed,
}

impl ForwardStages {
    // Duplicated intentionally to keep stage types self-contained.
    const MS: [usize; NSTAGES] = [4, 8, 16, 32, 64, 128, 256, 512];
}

/// Opaque stage tables for inverse NTT; validated by `intt_stages()`.
#[derive(Clone, Copy)]
pub(crate) struct InverseStages {
    // Precomputed multiplicative inverses of the stage twiddle table entries.
    // This avoids an `inv_mod_q()` lookup per butterfly in the inverse hot loop.
    inv_roots: [[u16; N]; NSTAGES],
    // inv(sqrt(-1)) mod q, used by the inverse size-2 base case.
    inv_sqr1: u16,
    _sealed: Sealed,
}

impl InverseStages {
    // Same sizes as forward; we just traverse them in reverse.
    // Duplicated intentionally to keep stage types self-contained.
    const MS: [usize; NSTAGES] = [4, 8, 16, 32, 64, 128, 256, 512];
}

#[inline]
const fn bitrev9_u16(mut x: u16) -> u16 {
    // Reverse the low 9 bits of x (since n=512).
    let mut r = 0u16;
    let mut i: u8 = 0;
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
        t[i] = bitrev9_u16(i as u16) as usize;
        i += 1;
    }
    t
};

// Stage construction is expected to be hoisted and reused by hot paths.
// Marked cold and non-inlined to discourage code-size growth and repeated rebuilds.
// Deterministic and validated; intended to be built once per call-site, not per transform.
// This is a hint only for stage construction; transforms remain hot and should use *_with_stages.
#[cold]
#[inline(never)]
pub(crate) fn forward_stages() -> Result<ForwardStages, FalconError> {
    // Stage sizes for bottom-up merges (m = 4, 8, ..., 512). Kept adjacent to `roots`
    // so the pairing cannot drift without touching this one function.
    let ms = ForwardStages::MS;

    // Placeholder initializer: overwritten for all stages before return.
    let mut twiddles: [&'static [u16]; NSTAGES] = [&[]; NSTAGES];
    for i in 0..NSTAGES {
        twiddles[i] = roots_for_size(ms[i]).ok_or(FalconError::InvalidNttConstants)?;
    }

    // `ms` is compile-time fixed; these checks defend against table corruption/mismatched
    // build artifacts so later indexing remains provably in-bounds.
    // For each stage of size m, we run i in 0..half and read w[i<<1], i.e. indices 0, 2, ..., m-2.
    for (&m, w) in ms.iter().zip(twiddles.iter()) {
        if m > N || m < 4 || (m & (m - 1)) != 0 || w.len() != m {
            return Err(FalconError::InvalidNttConstants);
        }
    }

    Ok(ForwardStages {
        roots: twiddles,
        _sealed: Sealed,
    })
}

#[inline]
fn ntt_with_stages_and_normalized_inner(
    a: &mut [i16; 512],
    stages: &ForwardStages,
) -> Result<(), FalconError> {
    // Input is already normalized into [0, q) by the caller.
    // We'll use two stack buffers and swap references each stage (still iterative).
    //
    // No-panics:
    // - All loops have fixed bounds under N=512.
    // - Twiddle slices were validated to have length m for each stage.
    // - All indices are derived from fixed ranges and validated slice lengths.
    let mut buf0 = [0u16; N];
    let mut buf1 = [0u16; N];

    // Put input into the same “leaf order” produced by recursive split(): bit-reversal.
    // Bit-reversal is an involution, so gather == scatter; gather is chosen for sequential writes.
    for i in 0..N {
        buf0[i] = a[BITREV_9[i]] as u16;
    }

    // Base case (m = 2): [u + sqr1*v, u - sqr1*v]
    {
        let src = &buf0;
        let dst = &mut buf1;
        for base in (0..N).step_by(2) {
            let u = src[base];
            let v = src[base + 1];
            let t = mul_mod_q(v, SQRT_MINUS_ONE_MOD_Q);
            dst[base] = add_mod_q(u, t);
            dst[base + 1] = sub_mod_q(u, t);
        }
    }

    // After the base case we have written into buf1.
    let mut src_is_buf0 = false;

    // Bottom-up merges for m = 4, 8, ..., 512.
    for stage_idx in 0..NSTAGES {
        // Stage sizes are fixed by Falcon-512.
        let m = ForwardStages::MS[stage_idx];
        let half = m >> 1;
        let w = stages.roots[stage_idx];

        // Alternating ping-pong buffers. Kept as two explicit cases so reference
        // provenance is obvious and the borrow story stays simple.
        if src_is_buf0 {
            stage_merge_interleaved(&mut buf1, &buf0, half, w);
        } else {
            stage_merge_interleaved(&mut buf0, &buf1, half, w);
        }

        src_is_buf0 = !src_is_buf0;
    }

    // After each stage we flip; `src_is_buf0` indicates which buffer holds the latest results.
    let src = if src_is_buf0 { &buf0 } else { &buf1 };
    for i in 0..N {
        a[i] = src[i] as i16;
    }

    Ok(())
}

#[inline]
// Preconditions (enforced by `forward_stages()`):
// - `half == m/2` where `m` is a validated power of two in [4, 512]
// - `w.len() == 2*half` and we read indices 0, 2, ..., 2*half - 2
fn stage_merge_interleaved(dst: &mut [u16; N], src: &[u16; N], half: usize, w: &[u16]) {
    let m = half << 1;
    for base in (0..N).step_by(m) {
        let b2 = base + half;
        for i in 0..half {
            let u = src[base + i];
            let v = src[b2 + i];

            // Canonical Falcon convention: for size m, use the even-indexed twiddles.
            let j = i << 1;
            let tw = w[j];
            let t = mul_mod_q(v, tw);

            // Write outputs interleaved (even/odd) to match canonical ordering.
            let out_even = base + j;
            dst[out_even] = add_mod_q(u, t);
            dst[out_even + 1] = sub_mod_q(u, t);
        }
    }
}

/// Convenience: constructs stage tables; prefer `_with_stages` in hot paths.
pub(crate) fn intt(a: &mut [i16; 512]) -> Result<(), FalconError> {
    // NOTE: This is a convenience wrapper that constructs and validates stage tables.
    // Hot paths should prefer `intt_with_stages*` and reuse the `InverseStages`.
    let stages = intt_stages()?;
    intt_with_stages(a, &stages)
}

// Stage construction is expected to be hoisted and reused by hot paths.
// Marked cold and non-inlined to discourage code-size growth and repeated rebuilds.
// Deterministic and validated; intended to be built once per call-site, not per transform.
// This is a hint only for stage construction; transforms remain hot and should use *_with_stages.
#[cold]
#[inline(never)]
pub(crate) fn intt_stages() -> Result<InverseStages, FalconError> {
    let ms = InverseStages::MS;

    // Twiddle tables are used only to derive inverse twiddles.
    let mut twiddles: [&'static [u16]; NSTAGES] = [&[]; NSTAGES];
    for i in 0..NSTAGES {
        twiddles[i] = roots_for_size(ms[i]).ok_or(FalconError::InvalidNttConstants)?;
    }

    for (&m, r) in ms.iter().zip(twiddles.iter()) {
        if m > N || m < 4 || (m & (m - 1)) != 0 || r.len() != m {
            return Err(FalconError::InvalidNttConstants);
        }
    }

    // Build per-stage inverse twiddle tables.
    // Canonical Falcon convention uses only the even-indexed twiddles (j = i<<1).
    // Odd indices are intentionally left zero and are never read.
    let mut inv_roots = [[0u16; N]; NSTAGES];
    for (stage_idx, &m) in ms.iter().enumerate() {
        let r = twiddles[stage_idx];
        for k in (0..m).step_by(2) {
            inv_roots[stage_idx][k] = inv_mod_q(r[k]);
        }
    }

    let inv_sqr1 = inv_mod_q(SQRT_MINUS_ONE_MOD_Q);

    Ok(InverseStages {
        inv_roots,
        inv_sqr1,
        _sealed: Sealed,
    })
}

/// Inverse NTT on 512 coefficients (in-place).
///
/// Mirrors the forward transform, but in reverse:
/// 1) normalize input into canonical residues in `[0, q)` (wrapper)
/// 2) undo merges (512→4), returning to recursion “leaf order”
/// 3) undo the size-2 base case
/// 4) scatter (bit-reversal) back to natural coefficient order
#[inline]
pub(crate) fn intt_with_stages(
    a: &mut [i16; 512],
    stages: &InverseStages,
) -> Result<(), FalconError> {
    normalize_in_place(a);

    intt_with_stages_and_normalized(a, stages)
}

/// Same as `intt_with_stages`, but assumes all inputs are already normalized
/// into canonical residues in `[0, q)`.
///
/// This is a performance-oriented entrypoint for internal callers that can uphold
/// the precondition.
///
/// Contract:
/// - All coefficients MUST be in `[0, 12289)` (i.e. `[0, q)`). Values outside this range may
///   break the implicit value-preserving `as u16` conversion.
/// - Input MUST be in the canonical Falcon NTT ordering produced by `ntt_with_stages_and_normalized`
///   in this module.
#[inline]
pub(crate) fn intt_with_stages_and_normalized(
    a: &mut [i16; 512],
    stages: &InverseStages,
) -> Result<(), FalconError> {
    debug_assert!(
        a.iter().all(|&x| (x as i32) >= 0 && (x as i32) < Q_I32),
        "intt_with_stages_and_normalized: input not in [0,q)"
    );

    intt_with_stages_and_normalized_inner(a, stages)
}

#[inline]
fn intt_with_stages_and_normalized_inner(
    a: &mut [i16; 512],
    stages: &InverseStages,
) -> Result<(), FalconError> {
    // No-panics:
    // - All loops have fixed bounds under N=512.
    // - Stage sizes and table lengths are validated by `*_stages()`.
    // - All indices are derived from fixed ranges and validated slice lengths.

    let mut buf0 = [0u16; N];
    let mut buf1 = [0u16; N];

    // Load NTT-domain input directly (already in canonical NTT ordering).
    for i in 0..N {
        buf0[i] = a[i] as u16;
    }

    // After load, latest results live in buf0.
    let mut src_is_buf0 = true;

    // Undo merges from m = 512 down to 4.
    for stage_rev in (0..NSTAGES).rev() {
        let m = InverseStages::MS[stage_rev];
        let half = m >> 1;
        let inv_w = &stages.inv_roots[stage_rev][..m];

        if src_is_buf0 {
            stage_unmerge_from_interleaved(&mut buf1, &buf0, half, inv_w);
        } else {
            stage_unmerge_from_interleaved(&mut buf0, &buf1, half, inv_w);
        }

        src_is_buf0 = !src_is_buf0;
    }

    // Undo base case (m = 2) into the other buffer.
    if src_is_buf0 {
        inv_base_case_into(&mut buf1, &buf0, stages.inv_sqr1);
    } else {
        inv_base_case_into(&mut buf0, &buf1, stages.inv_sqr1);
    }
    src_is_buf0 = !src_is_buf0;

    // Final scatter out of leaf order back into coefficient order.
    // `leaf` is in recursion leaf order; scatter (bit-reversal) restores natural coefficient order.
    let leaf = if src_is_buf0 { &buf0 } else { &buf1 };
    for i in 0..N {
        a[BITREV_9[i]] = leaf[i] as i16;
    }

    Ok(())
}

#[inline]
fn inv_base_case_into(dst: &mut [u16; N], src: &[u16; N], inv_sqr1: u16) {
    for base in (0..N).step_by(2) {
        let a0 = src[base];
        let a1 = src[base + 1];

        // u = inv2*(a0 + a1)
        // v = inv2*inv(sqr1)*(a0 - a1)
        let sum = add_mod_q(a0, a1);
        let diff = sub_mod_q(a0, a1);

        dst[base] = mul_mod_q(sum, INV2_MOD_Q);
        dst[base + 1] = mul_mod_q(mul_mod_q(diff, INV2_MOD_Q), inv_sqr1);
    }
}

#[inline]
// Inverse of `stage_merge_interleaved()`.
//
// Forward stage (size m = 2*half) did for each i:
//   t  = v * tw
//   o0 = u + t
//   o1 = u - t
// written to consecutive even/odd.
//
// Inverse recovers:
//   u = inv2*(o0 + o1)
//   t = inv2*(o0 - o1)
//   v = t * inv(tw)
fn stage_unmerge_from_interleaved(dst: &mut [u16; N], src: &[u16; N], half: usize, inv_w: &[u16]) {
    let m = half << 1;
    debug_assert_eq!(inv_w.len(), m);

    for base in (0..N).step_by(m) {
        let b2 = base + half;

        for i in 0..half {
            let j = i << 1;

            let o0 = src[base + j];
            let o1 = src[base + j + 1];

            let u = mul_mod_q(add_mod_q(o0, o1), INV2_MOD_Q);
            let t = mul_mod_q(sub_mod_q(o0, o1), INV2_MOD_Q);

            // Canonical Falcon convention: twiddle is the even-indexed entry.
            // Use the precomputed inverse twiddle to avoid work in the hot loop.
            let v = mul_mod_q(t, inv_w[j]);

            dst[base + i] = u;
            dst[b2 + i] = v;
        }
    }
}

// FALCON_Q as a u32
const Q: u32 = 12289;

// MU = floor(2^32 / Q)
const MU: u32 = ((1u64 << 32) / (Q as u64)) as u32;

#[inline(always)]
/// Returns x mod Q, without needing to use the modulus operator
/// Precondition: x < Q^2 (true for (u16 mod Q) * (u16 mod Q)); fix-up uses at most two subtracts under this bound.
fn reduce_u32_mod_q(x: u32) -> u16 {
    // qhat = floor(x * MU / 2^32)
    let qhat = ((x as u64 * MU as u64) >> 32) as u32;

    // r = x - qhat*Q, computed in u32 since x < 2^28 and qhat is small
    // Since MU <= 2^32 / Q, then x * MU / 2^32 <= x / Q, so qhat <= floor(x / Q).
    // That means qhat * Q <= x, so x - qhat*Q cannot underflow
    let mut r = x - qhat * Q;

    // r may still be >= Q, but close enough to reach it with at most two subtractions
    if r >= Q {
        r -= Q;
    }
    if r >= Q {
        r -= Q;
    }
    debug_assert!(r < Q);

    r as u16
}

/// Pointwise multiply `a` by `b` modulo Q, in-place on `a`.
///
/// Intended for Falcon-512 NTT-domain Hadamard multiplication. This function keeps everything in `i16`
/// because `normalize_in_place()` ensures coefficients are canonical residues in `[0, Q)`,
/// and `Q = 12289 < i16::MAX`.
///
/// No heap allocation, no extra buffers.
#[inline(always)]
pub(crate) fn pointwise_mul_in_place(
    a: &mut [i16; 512],
    b: &[i16; 512],
) -> Result<(), FalconError> {
    // Ensure both operands are canonical residues.
    normalize_in_place(a);

    // If you want to be extra misuse-proof, normalize a local copy of `b`.
    // If you expect `b` is already normalized at call sites, you can skip this copy
    // and document that precondition instead.
    let mut b_norm = *b;
    normalize_in_place(&mut b_norm);

    for i in 0..512 {
        // After normalization, values are in [0, Q), so these casts are safe.
        let aa = a[i] as u32;
        let bb = b_norm[i] as u32;

        // aa*bb < (Q-1)^2 < Q^2, satisfying reduce_u32_mod_q precondition.
        let prod = aa * bb;
        a[i] = reduce_u32_mod_q(prod) as i16;
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
    // Correctness-first: `% q` is deterministic and easy to audit; performance upgrades
    // (Montgomery/Barrett reduction) can be reviewed separately.
    let q = FALCON_Q as u32;
    ((a as u32 * b as u32) % q) as u16
}

#[inline]
fn inv_mod_q(x: u16) -> u16 {
    // Requires your table:
    // pub(super) const INV_MOD_Q: [u16; FALCON_Q as usize] = [...]
    crate::falcon::ntt_consts::INV_MOD_Q[x as usize]
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::falcon::{FALCON_N, FALCON_Q};
    use serde_json;

    #[test]
    fn test_bitrev9_is_involution_on_0_to_n() {
        for i in 0..N {
            let j = BITREV_9[i] as usize;
            assert_eq!(BITREV_9[j] as usize, i, "bitrev not involutive at i={i}");
        }
    }

    #[test]
    fn test_stage_unmerge_is_inverse_of_stage_merge_for_all_stage_sizes() {
        let q = FALCON_Q as u32;

        let mut src = [0u16; N];
        for i in 0..N {
            // Deterministic, non-trivial residues in [0, q).
            src[i] = (((i as u32 * 73) + 19) % q) as u16;
        }

        for &m in ForwardStages::MS.iter() {
            let half = m >> 1;
            let w = roots_for_size(m).expect("missing twiddle table");

            // Build an inverse-twiddle table that matches the convention used by the inverse:
            // only even indices are used (j = i<<1), odd entries are ignored.
            let mut inv_w = vec![0u16; m];
            for k in (0..m).step_by(2) {
                inv_w[k] = inv_mod_q(w[k]);
            }
            for k in (1..m).step_by(2) {
                inv_w[k] = 7777;
            }

            let mut merged = [0u16; N];
            stage_merge_interleaved(&mut merged, &src, half, w);

            let mut unmerged = [0u16; N];
            stage_unmerge_from_interleaved(&mut unmerged, &merged, half, &inv_w);

            assert_eq!(unmerged, src, "stage inverse mismatch for m={m}");
        }
    }

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
    fn test_inv_mod_q_basic_identities() {
        // 0 has no multiplicative inverse; conventionally mapped to 0 in the table.
        assert_eq!(inv_mod_q(0), 0);

        // 1 is its own inverse.
        assert_eq!(inv_mod_q(1), 1);

        // (-1 mod q) is its own inverse.
        let qm1 = (FALCON_Q - 1) as u16;
        assert_eq!(inv_mod_q(qm1), qm1);
    }

    #[test]
    fn test_inv_mod_q_outputs_are_in_range() {
        // Table values should always be canonical residues: [0, q).
        let q = FALCON_Q as u16;
        for x in 0..q {
            let inv = inv_mod_q(x);
            assert!(inv < q, "inv_mod_q({x}) produced out-of-range value {inv}");
        }
    }

    #[test]
    fn test_inv_mod_q_is_correct_multiplicative_inverse_for_all_nonzero_x() {
        // For all x != 0, x * inv_mod_q(x) == 1 (mod q).
        // Uses the same mul_mod_q helper as the NTT code, which is deterministic and clear.
        let q = FALCON_Q as u16;

        for x in 1..q {
            let inv = inv_mod_q(x);
            let prod = mul_mod_q(x, inv);
            assert_eq!(
                prod, 1u16,
                "bad inverse: x={x}, inv={inv}, x*inv mod q = {prod}"
            );
        }
    }

    #[test]
    fn test_inv_mod_q_is_involutive_for_all_x() {
        // inv(inv(x)) == x for all x.
        // For x=0 this matches the convention 0 -> 0.
        let q = FALCON_Q as u16;

        for x in 0..q {
            let inv = inv_mod_q(x);
            let back = inv_mod_q(inv);
            assert_eq!(
                back, x,
                "involution failed: x={x}, inv={inv}, inv(inv(x))={back}"
            );
        }
    }

    #[test]
    fn test_inv_mod_q_negation_consistency() {
        // For x != 0, inv(q-x) == q - inv(x).
        // (Because inv(-x) == -inv(x) in a field.)
        let q = FALCON_Q as u16;

        for x in 1..q {
            let neg = sub_mod_q(0, x); // == q - x for x != 0
            let inv_x = inv_mod_q(x);
            let inv_neg = inv_mod_q(neg);

            let expected = sub_mod_q(0, inv_x); // == q - inv(x) for inv(x) != 0
            assert_eq!(
                inv_neg, expected,
                "negation consistency failed: x={x}, inv(x)={inv_x}, inv(-x)={inv_neg}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_inv_mod_q_matches_table_symmetry() {
        let q = FALCON_Q as u16;

        for x in 1..q {
            let inv_x = inv_mod_q(x);
            let inv_q_minus_x = inv_mod_q((q - x) % q); // (q-x) in [0,q)
            let sum = add_mod_q(inv_x, inv_q_minus_x);
            assert_eq!(
                sum, 0u16,
                "expected inv(x) + inv(q-x) == 0 (mod q): x={x}, inv={inv_x}, inv(q-x)={inv_q_minus_x}, sum={sum}"
            );
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
            assert!(x >= 0 && x < q, "out of range at idx={idx}: x={x}, q={q}");
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
                5 => -(FALCON_Q as i16),
                6 => -(FALCON_Q as i16) + 1,
                7 => i16::MIN,
                _ => i16::MAX,
            };
        }

        // Wrapper result
        let mut got = input;
        ntt(&mut got).expect("ntt failed");
        assert_all_in_range(&got);

        // Normalization should not affect the result (the wrapper normalizes internally).
        let mut want = input;
        normalize_in_place(&mut want);
        ntt(&mut want).expect("ntt failed (normalized)");
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
        ntt(&mut got).expect("ntt failed");
        assert_all_in_range(&got);

        // Explicitly normalizing first should not change anything.
        let mut want = input;
        normalize_in_place(&mut want);
        ntt(&mut want).expect("ntt failed (explicit normalize)");
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

        ntt(&mut a1).expect("ntt failed (a1)");
        ntt(&mut a2).expect("ntt failed (a2)");

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

        ntt(&mut input).expect("ntt failed");
        assert_all_in_range(&input);
    }

    #[test]
    fn test_ntt_forward_in_place_matches_kat512_vectors() {
        for (case_idx, kat) in kats().into_iter().enumerate() {
            let mut a = kat.input;
            ntt(&mut a).expect(&format!("case {}: ntt failed", case_idx));
            assert_eq!(a, kat.expected, "case {}: mismatch", case_idx);
        }
    }

    /// One NTT KAT: input coefficients and expected forward-NTT output coefficients,
    /// both in the representation your implementation returns (typically [0, q) stored in i16).
    #[derive(Clone, Copy)]
    struct NttKat {
        input: [i16; 512],
        expected: [i16; 512],
    }

    /// Apply per-coefficient: x -> x + delta (in i32), then cast back to i16.
    /// This is safe for deltas ±q since i16 range is far larger than ±12289.
    fn add_delta_mod_repr(x: &[i16; 512], delta: i32) -> [i16; 512] {
        let mut out = [0i16; 512];
        for (i, &v) in x.iter().enumerate() {
            out[i] = (v as i32 + delta) as i16;
        }
        out
    }

    /// Run the forward NTT and return the output array.
    fn run_ntt(mut a: [i16; 512]) -> [i16; 512] {
        ntt(&mut a).expect("ntt failed");
        a
    }

    fn kats() -> Vec<NttKat> {
        const KAT_JSON: &str = include_str!("ntt_KAT512.json");

        let pairs: Vec<[Vec<i32>; 2]> =
            serde_json::from_str(KAT_JSON).expect("failed to parse ntt_KAT512.json");
        assert!(!pairs.is_empty(), "ntt_KAT512.json contained no vectors");

        let mut out = Vec::with_capacity(pairs.len());
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

            let mut input = [0i16; 512];
            let mut expected = [0i16; 512];

            for (i, x) in input_vec.into_iter().enumerate() {
                assert!(
                    (i16::MIN as i32..=i16::MAX as i32).contains(&x),
                    "case {case_idx}, idx {i}: input out of i16 range: {x}"
                );
                input[i] = x as i16;
            }

            for (i, x) in expected_vec.into_iter().enumerate() {
                assert!(
                    (0..q_i32).contains(&x),
                    "case {case_idx}, idx {i}: expected out of range: {x}"
                );
                expected[i] = x as i16;
            }

            out.push(NttKat { input, expected });
        }

        out
    }

    #[test]
    fn test_forward_ntt_matches_kats() {
        for (idx, kat) in kats().into_iter().enumerate() {
            let got = run_ntt(kat.input);
            assert_eq!(got, kat.expected, "KAT mismatch at index {}", idx);
        }
    }

    #[test]
    fn test_forward_ntt_invariant_under_plus_minus_q_per_coeff() {
        let q = FALCON_Q as i32;

        for (idx, kat) in kats().into_iter().enumerate() {
            // Baseline
            let base_out = run_ntt(kat.input);

            // x + q
            let x_plus_q = add_delta_mod_repr(&kat.input, q);
            let out_plus_q = run_ntt(x_plus_q);
            assert_eq!(
                out_plus_q, base_out,
                "NTT(x+q) != NTT(x) for KAT index {}",
                idx
            );

            // x - q
            let x_minus_q = add_delta_mod_repr(&kat.input, -q);
            let out_minus_q = run_ntt(x_minus_q);
            assert_eq!(
                out_minus_q, base_out,
                "NTT(x-q) != NTT(x) for KAT index {}",
                idx
            );
        }
    }

    #[test]
    fn test_inverse_ntt_matches_kats_reverse_direction() {
        let q = FALCON_Q as i32;

        for (idx, kat) in kats().into_iter().enumerate() {
            // KAT file provides: (input_coeffs, expected_ntt)
            // We want: intt(expected_ntt) == normalize(input_coeffs) in [0,q)
            let mut got = kat.expected;
            intt(&mut got).expect("intt failed");

            let mut want = kat.input;
            // Canonicalize the same way the forward path does: into [0,q)
            for x in want.iter_mut() {
                *x = (*x as i32).rem_euclid(q) as i16;
            }

            assert_eq!(got, want, "INTT(KAT.expected) mismatch at index {}", idx);
        }
    }

    #[test]
    fn test_inverse_then_forward_roundtrips_to_kat_expected_ntt() {
        for (idx, kat) in kats().into_iter().enumerate() {
            // Start in NTT domain (kat.expected), go back to coeffs, then forward again.
            let mut coeffs = kat.expected;
            intt(&mut coeffs).expect("intt failed");

            let got_back = run_ntt(coeffs);
            assert_eq!(
                got_back, kat.expected,
                "NTT(INTT(KAT.expected)) != KAT.expected at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_inverse_ntt_invariant_under_plus_minus_q_per_coeff_in_ntt_domain() {
        let q = FALCON_Q as i32;

        for (idx, kat) in kats().into_iter().enumerate() {
            let mut base = kat.expected;
            intt(&mut base).expect("intt failed");

            // expected_ntt + q
            let mut plus_q = add_delta_mod_repr(&kat.expected, q);
            intt(&mut plus_q).expect("intt failed");
            assert_eq!(plus_q, base, "INTT(y+q) != INTT(y) for KAT index {}", idx);

            // expected_ntt - q
            let mut minus_q = add_delta_mod_repr(&kat.expected, -q);
            intt(&mut minus_q).expect("intt failed");
            assert_eq!(minus_q, base, "INTT(y-q) != INTT(y) for KAT index {}", idx);
        }
    }

    // Deterministic tiny PRNG (no external deps).
    // Produces u16 in [0, q).
    #[derive(Clone)]
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// Advance the generator once and return the raw 64-bit output.
        #[inline]
        fn next_u64(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        /// Produce a value in [0, q).
        #[inline]
        fn next_u16_mod_q(&mut self) -> u16 {
            (self.next_u64() % (FALCON_Q as u64)) as u16
        }

        /// Produce a wide signed value spanning the full i16 range.
        ///
        /// Intentionally includes negatives to exercise normalization paths.
        #[inline]
        fn next_i16_wide(&mut self) -> i16 {
            self.next_u64() as i16
        }
    }

    fn build_inv_w_for_stage(m: usize) -> [u16; N] {
        let twiddles = roots_for_size(m).expect("missing roots_for_size(m) in test");
        assert_eq!(twiddles.len(), m);

        let mut inv_w = [0u16; N];
        // Production convention: only even indices used (j = i<<1).
        for k in (0..m).step_by(2) {
            inv_w[k] = inv_mod_q(twiddles[k]);
        }
        inv_w
    }

    /// This verifies that stage_unmerge_from_interleaved is a left-inverse of
    /// stage_merge_interleaved for a single stage size m.
    fn roundtrip_one_stage(m: usize, seed: u64) {
        assert!(m.is_power_of_two());
        assert!(m >= 4 && m <= N);

        let half = m / 2;
        let twiddles = roots_for_size(m).expect("missing roots_for_size(m)");
        let inv_w_full = build_inv_w_for_stage(m);

        // src layout for the forward merge:
        // for each block of size m: [u(half), v(half)] contiguous.
        let mut src = [0u16; N];
        let mut rng = XorShift64::new(seed);

        for base in (0..N).step_by(m) {
            for i in 0..half {
                src[base + i] = rng.next_u16_mod_q(); // u
                src[base + half + i] = rng.next_u16_mod_q(); // v
            }
        }

        // Forward stage: src (contiguous u/v) -> interleaved outputs.
        let mut interleaved = [0u16; N];
        stage_merge_interleaved(&mut interleaved, &src, half, twiddles);

        // Inverse stage: interleaved -> recovered contiguous u/v
        let mut recovered = [0u16; N];
        stage_unmerge_from_interleaved(&mut recovered, &interleaved, half, &inv_w_full[..m]);

        assert_eq!(
            recovered, src,
            "stage roundtrip failed for m={m} seed={seed}"
        );
    }

    #[test]
    fn test_stage_unmerge_is_inverse_of_stage_merge_for_all_sizes() {
        // A few deterministic seeds; keeps runtime reasonable and coverage good.
        let seeds = [
            0x0123_4567_89AB_CDEF,
            0xDEAD_BEEF_F00D_F00D,
            0xC0FF_EE00_1234_5678,
            0x0,
        ];

        for &m in ForwardStages::MS.iter() {
            for (i, &seed) in seeds.iter().enumerate() {
                roundtrip_one_stage(m, seed ^ (m as u64) ^ (i as u64));
            }
        }
    }

    /// Odd indices in inv_w must be irrelevant because the implementation only
    /// reads inv_w[j] with j=i<<1 (always even).
    #[test]
    fn test_stage_unmerge_does_not_depend_on_odd_inv_w_entries() {
        let m = 64usize;
        let half = m / 2;

        let twiddles = roots_for_size(m).expect("missing roots_for_size(m)");
        let mut inv_w_a = build_inv_w_for_stage(m);
        let mut inv_w_b = inv_w_a;

        // Poison odd entries in inv_w_b; outputs should remain identical.
        let mut rng = XorShift64::new(0xBADC0FFEE_u64);
        for k in (1..m).step_by(2) {
            inv_w_b[k] = rng.next_u16_mod_q();
        }

        // Build random src for this stage.
        let mut src = [0u16; N];
        let mut rng2 = XorShift64::new(0xA5A5_A5A5_5A5A_5A5A);
        for base in (0..N).step_by(m) {
            for i in 0..half {
                src[base + i] = rng2.next_u16_mod_q();
                src[base + half + i] = rng2.next_u16_mod_q();
            }
        }

        // Forward stage -> interleaved.
        let mut interleaved = [0u16; N];
        stage_merge_interleaved(&mut interleaved, &src, half, twiddles);

        // Inverse with inv_w_a.
        let mut out_a = [0u16; N];
        stage_unmerge_from_interleaved(&mut out_a, &interleaved, half, &inv_w_a[..m]);

        // Inverse with inv_w_b (odd entries differ).
        let mut out_b = [0u16; N];
        stage_unmerge_from_interleaved(&mut out_b, &interleaved, half, &inv_w_b[..m]);

        assert_eq!(
            out_a, out_b,
            "odd inv_w entries affected output unexpectedly"
        );
        assert_eq!(
            out_a, src,
            "roundtrip failed with poisoned odd inv_w entries"
        );
    }

    /// Sanity check with the smallest non-base stage (m=4) using a structured input
    /// that makes it easier to eyeball failures.
    #[test]
    fn test_stage_unmerge_roundtrip_m4_structured() {
        let m = 4usize;
        let half = 2usize;

        let twiddles = roots_for_size(m).expect("missing roots_for_size(4)");
        let inv_w = build_inv_w_for_stage(m);

        let mut src = [0u16; N];
        for base in (0..N).step_by(m) {
            // u0,u1,v0,v1 pattern
            src[base + 0] = 1;
            src[base + 1] = 2;
            src[base + 2] = 3;
            src[base + 3] = 4;
        }

        let mut interleaved = [0u16; N];
        stage_merge_interleaved(&mut interleaved, &src, half, twiddles);

        let mut recovered = [0u16; N];
        stage_unmerge_from_interleaved(&mut recovered, &interleaved, half, &inv_w[..m]);

        assert_eq!(recovered, src);
    }

    // Forward base case used by NTT:
    // o0 = u + sqr1*v
    // o1 = u - sqr1*v
    #[inline]
    fn fwd_base_case_into(dst: &mut [u16; N], src: &[u16; N]) {
        for base in (0..N).step_by(2) {
            let u = src[base];
            let v = src[base + 1];
            let t = mul_mod_q(v, SQRT_MINUS_ONE_MOD_Q);
            dst[base] = add_mod_q(u, t);
            dst[base + 1] = sub_mod_q(u, t);
        }
    }

    #[test]
    fn test_inv_base_case_is_inverse_of_forward_base_case_roundtrip() {
        let inv_sqr1 = inv_mod_q(SQRT_MINUS_ONE_MOD_Q);

        // Build a random "leaf order" vector (pairs are u,v).
        let mut rng = XorShift64::new(0x0123_4567_89AB_CDEF);
        let mut src = [0u16; N];
        for i in 0..N {
            src[i] = rng.next_u16_mod_q();
        }

        // Forward base case (u,v) -> (o0,o1)
        let mut fwd = [0u16; N];
        fwd_base_case_into(&mut fwd, &src);

        // Inverse base case (o0,o1) -> (u,v)
        let mut recovered = [0u16; N];
        inv_base_case_into(&mut recovered, &fwd, inv_sqr1);

        assert_eq!(
            recovered, src,
            "inv_base_case_into did not invert forward base case"
        );
    }

    #[test]
    fn test_inv_base_case_pair_identities_hold() {
        let inv_sqr1 = inv_mod_q(SQRT_MINUS_ONE_MOD_Q);

        // Use a few hand-picked pairs that stress wraparound and symmetry.
        let pairs = [
            (0u16, 0u16),
            (1u16, 0u16),
            ((FALCON_Q - 1) as u16, 1u16),
            (123u16, 456u16),
            (6144u16, 6144u16),
        ];

        // Build src as repeated pairs in [0,q).
        let mut src = [0u16; N];
        for (k, &(u, v)) in pairs.iter().enumerate() {
            let base = (k * 2) % N;
            src[base] = u % (FALCON_Q as u16);
            src[base + 1] = v % (FALCON_Q as u16);
        }

        // Forward then inverse.
        let mut fwd = [0u16; N];
        fwd_base_case_into(&mut fwd, &src);

        let mut inv = [0u16; N];
        inv_base_case_into(&mut inv, &fwd, inv_sqr1);

        // Check exact recovery for the filled positions.
        for (k, &(u, v)) in pairs.iter().enumerate() {
            let base = (k * 2) % N;
            assert_eq!(inv[base], u % (FALCON_Q as u16), "u mismatch at pair {k}");
            assert_eq!(
                inv[base + 1],
                v % (FALCON_Q as u16),
                "v mismatch at pair {k}"
            );
        }

        // Also check algebraic invariants directly in the forward outputs:
        // From forward:
        //   o0 + o1 = 2u
        //   o0 - o1 = 2*sqr1*v
        // From inverse:
        //   u = inv2*(o0 + o1)
        //   v = inv2*inv(sqr1)*(o0 - o1)
        for (k, &(u, v)) in pairs.iter().enumerate() {
            let base = (k * 2) % N;
            let o0 = fwd[base];
            let o1 = fwd[base + 1];

            let two_u = add_mod_q(u % (FALCON_Q as u16), u % (FALCON_Q as u16));
            assert_eq!(add_mod_q(o0, o1), two_u, "o0+o1 != 2u at pair {k}");

            let two_sqr1_v = add_mod_q(
                mul_mod_q(v % (FALCON_Q as u16), SQRT_MINUS_ONE_MOD_Q),
                mul_mod_q(v % (FALCON_Q as u16), SQRT_MINUS_ONE_MOD_Q),
            );
            assert_eq!(
                sub_mod_q(o0, o1),
                two_sqr1_v,
                "o0-o1 != 2*sqr1*v at pair {k}"
            );
        }
    }

    #[test]
    fn test_inv_base_case_ignores_inv_sqr1_when_diff_is_zero() {
        // If a0==a1 then diff=0, so v must be 0 regardless of inv_sqr1.
        let mut src = [0u16; N];
        for base in (0..N).step_by(2) {
            src[base] = 777;
            src[base + 1] = 777; // a0 == a1
        }

        let mut out_a = [0u16; N];
        inv_base_case_into(&mut out_a, &src, 1); // wrong inv_sqr1

        let mut out_b = [0u16; N];
        inv_base_case_into(&mut out_b, &src, inv_mod_q(SQRT_MINUS_ONE_MOD_Q)); // correct inv_sqr1

        // u should be the same and v should be 0 in both.
        for base in (0..N).step_by(2) {
            assert_eq!(
                out_a[base], out_b[base],
                "u depended on inv_sqr1 unexpectedly"
            );
            assert_eq!(out_a[base + 1], 0, "expected v=0 when a0==a1");
            assert_eq!(out_b[base + 1], 0, "expected v=0 when a0==a1");
        }
    }

    #[inline]
    fn normalize_expected(mut a: [i16; 512]) -> [i16; 512] {
        // Mirror normalize_in_place semantics in a standalone helper so expectations are explicit.
        let q = FALCON_Q as i32;
        for x in a.iter_mut() {
            *x = (*x as i32).rem_euclid(q) as i16;
        }
        a
    }

    #[test]
    fn test_intt_inner_matches_public_normalized_entrypoint() {
        let fwd = forward_stages().expect("forward_stages");
        let inv = intt_stages().expect("intt_stages");

        // Build a random coeff vector, run full forward NTT (this normalizes + transforms),
        // then call both:
        // - intt_with_stages_and_normalized (wrapper)
        // - intt_with_stages_and_normalized_inner (inner)
        // They must produce identical results.
        let mut rng = XorShift64::new(0xA11CE5EED_u64);
        let mut coeffs = [0i16; 512];
        for i in 0..512 {
            coeffs[i] = rng.next_i16_wide();
        }

        let expected = normalize_expected(coeffs);

        // Forward NTT in-place.
        let mut ntt_a = coeffs;
        ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

        // Both inverse paths starting from the same normalized NTT-domain input.
        let mut via_wrapper = ntt_a;
        intt_with_stages_and_normalized(&mut via_wrapper, &inv)
            .expect("intt_with_stages_and_normalized");

        let mut via_inner = ntt_a;
        intt_with_stages_and_normalized_inner(&mut via_inner, &inv)
            .expect("intt_with_stages_and_normalized_inner");

        assert_eq!(via_inner, via_wrapper, "inner != wrapper on same input");
        assert_eq!(
            via_inner, expected,
            "inverse did not recover normalized coefficients"
        );
    }

    #[test]
    fn test_intt_inner_is_inverse_of_forward_ntt_on_random_vectors() {
        let fwd = forward_stages().expect("forward_stages");
        let inv = intt_stages().expect("intt_stages");

        // Multiple seeds = multiple vectors, deterministic and stable.
        let seeds = [
            0x0000_0000_0000_0001_u64,
            0x0123_4567_89AB_CDEF_u64,
            0xDEAD_BEEF_FEED_F00D_u64,
            0xC0FF_EE00_BAD5_EED5_u64,
        ];

        for (case_idx, &seed) in seeds.iter().enumerate() {
            let mut rng = XorShift64::new(seed);
            let mut coeffs = [0i16; 512];
            for i in 0..512 {
                coeffs[i] = rng.next_i16_wide();
            }

            let expected = normalize_expected(coeffs);

            // Forward then inverse-inner.
            let mut ntt_a = coeffs;
            ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

            intt_with_stages_and_normalized_inner(&mut ntt_a, &inv)
                .expect("intt_with_stages_and_normalized_inner");

            assert_eq!(ntt_a, expected, "roundtrip mismatch on case {case_idx}");
        }
    }

    #[test]
    fn test_intt_inner_roundtrip_on_structured_edge_vectors() {
        let fwd = forward_stages().expect("forward_stages");
        let inv = intt_stages().expect("intt_stages");
        let q = FALCON_Q as i32;

        // A few “shape” vectors that catch indexing / scatter issues:
        // - all zeros
        // - all ones
        // - ramp
        // - single impulse
        let mut cases: Vec<[i16; 512]> = Vec::new();

        cases.push([0i16; 512]);

        let mut ones = [0i16; 512];
        for x in ones.iter_mut() {
            *x = 1;
        }
        cases.push(ones);

        let mut ramp = [0i16; 512];
        for i in 0..512 {
            ramp[i] = (i as i32 % q) as i16;
        }
        cases.push(ramp);

        let mut impulse = [0i16; 512];
        impulse[0] = 1;
        cases.push(impulse);

        let mut impulse_17 = [0i16; 512];
        impulse_17[17] = -1; // exercises normalization + scatter location.
        cases.push(impulse_17);

        for (idx, coeffs) in cases.into_iter().enumerate() {
            let expected = normalize_expected(coeffs);

            let mut ntt_a = coeffs;
            ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

            intt_with_stages_and_normalized_inner(&mut ntt_a, &inv)
                .expect("intt_with_stages_and_normalized_inner");

            assert_eq!(
                ntt_a, expected,
                "structured roundtrip mismatch on case {idx}"
            );
        }
    }

    #[test]
    fn test_intt_inner_is_deterministic_on_same_input() {
        let fwd = forward_stages().expect("forward_stages");
        let inv = intt_stages().expect("intt_stages");

        let mut rng = XorShift64::new(0xBADC0FFEE_u64);
        let mut coeffs = [0i16; 512];
        for i in 0..512 {
            coeffs[i] = rng.next_i16_wide();
        }

        // Build a canonical NTT-domain vector.
        let mut ntt_a = coeffs;
        ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

        let mut run1 = ntt_a;
        let mut run2 = ntt_a;

        intt_with_stages_and_normalized_inner(&mut run1, &inv).expect("inner run1");
        intt_with_stages_and_normalized_inner(&mut run2, &inv).expect("inner run2");

        assert_eq!(run1, run2, "non-deterministic output for same input");
    }

    #[test]
    fn test_intt_with_stages_matches_normalize_then_normalized_path() {
        let fwd = forward_stages().expect("forward_stages");
        let inv = intt_stages().expect("intt_stages");

        // Build a coefficient vector with lots of out-of-range values (negatives, large positives).
        let mut rng = XorShift64::new(0xC0FFEE_u64); // or define locally
        let mut coeffs = [0i16; 512];
        for i in 0..512 {
            coeffs[i] = rng.next_i16_wide();
        }

        // Forward NTT produces a canonical NTT-domain vector.
        let mut ntt_a = coeffs;
        ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

        // Create an intentionally "dirty" NTT-domain input by pushing values outside [0,q).
        // This must be repaired by intt_with_stages() normalization.
        let mut dirty_ntt = ntt_a;
        let q = FALCON_Q as i16;
        for i in (0..512).step_by(3) {
            dirty_ntt[i] = dirty_ntt[i].wrapping_sub(q); // negative or wrap
        }
        for i in (1..512).step_by(3) {
            dirty_ntt[i] = dirty_ntt[i].wrapping_add(q); // > q
        }
        // Index 2 mod 3 unchanged.

        // Path A: use the function under test (normalizes internally).
        let mut got = dirty_ntt;
        intt_with_stages(&mut got, &inv).expect("intt_with_stages");

        // Path B: explicitly normalize then call normalized entrypoint.
        let mut expect = normalize_expected(dirty_ntt);
        intt_with_stages_and_normalized(&mut expect, &inv)
            .expect("intt_with_stages_and_normalized");

        assert_eq!(
            got, expect,
            "intt_with_stages must equal normalize + normalized"
        );
    }

    #[test]
    fn test_intt_with_stages_is_identity_on_already_normalized_ntt_domain_inputs() {
        let fwd = forward_stages().expect("forward_stages");
        let inv = intt_stages().expect("intt_stages");

        // Start from a random coefficient vector, go to NTT domain (already normalized).
        let mut rng = XorShift64::new(0x1234_5678_u64); // or define locally
        let mut coeffs = [0i16; 512];
        for i in 0..512 {
            coeffs[i] = rng.next_i16_wide();
        }
        let expected_coeffs = normalize_expected(coeffs);

        let mut ntt_a = coeffs;
        ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

        // Invert using intt_with_stages (normalization should be a no-op here).
        intt_with_stages(&mut ntt_a, &inv).expect("intt_with_stages");

        assert_eq!(
            ntt_a, expected_coeffs,
            "intt_with_stages should invert ntt_with_stages on normalized NTT inputs"
        );
    }

    #[test]
    fn test_intt_with_stages_handles_extreme_out_of_range_values_without_panicking() {
        let inv = intt_stages().expect("intt_stages");

        // This test is about the wrapper’s normalization robustness on i16 extremes.
        // We don't require the output to match any particular polynomial here, just that
        // it completes and returns Ok (normalization is rem_euclid on i32, so it should).
        let mut a = [0i16; 512];
        for i in 0..512 {
            a[i] = match i % 4 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => -1,
                _ => 0,
            };
        }

        // intt_with_stages normalizes then calls the normalized path; the normalized path assumes
        // canonical NTT ordering, which this isn't. So it may produce garbage coefficients, but
        // should remain well-defined and not error.
        let res = intt_with_stages(&mut a, &inv);
        assert!(
            res.is_ok(),
            "intt_with_stages should not fail on extreme i16 inputs"
        );
    }

    #[test]
    fn test_intt_stages_builds_and_is_deterministic() {
        let a = intt_stages().expect("intt_stages");
        let b = intt_stages().expect("intt_stages");
        assert_eq!(a.inv_sqr1, b.inv_sqr1, "inv_sqr1 must be deterministic");
        assert_eq!(a.inv_roots, b.inv_roots, "inv_roots must be deterministic");
    }

    #[test]
    fn test_intt_stages_inv_sqr1_is_inverse_of_sqrt_minus_one() {
        let s = intt_stages().expect("intt_stages");
        let one = mul_mod_q(s.inv_sqr1, SQRT_MINUS_ONE_MOD_Q);
        assert_eq!(one, 1u16, "inv_sqr1 * sqr1 mod q must be 1");
    }

    #[test]
    fn test_intt_stages_inv_roots_matches_table_inverses_on_even_indices() {
        let s = intt_stages().expect("intt_stages");

        for (stage_idx, &m) in InverseStages::MS.iter().enumerate() {
            let tw = roots_for_size(m).expect("roots_for_size missing for known size");
            // Only even indices are populated.
            for k in (0..m).step_by(2) {
                let inv = s.inv_roots[stage_idx][k];
                // inv(tw[k]) * tw[k] == 1 mod q for k != 0.
                // (twiddle tables shouldn't contain 0 for these sizes; this assertion helps catch corruption.)
                assert_ne!(
                    tw[k], 0,
                    "stage {stage_idx}, m {m}, k {k}: twiddle should not be 0"
                );
                assert_eq!(
                    mul_mod_q(inv, tw[k]),
                    1u16,
                    "stage {stage_idx}, m {m}, k {k}: inv_roots[k] must be inv(twiddle[k])"
                );
            }
        }
    }

    #[test]
    fn test_intt_stages_inv_roots_odd_indices_are_zero() {
        let s = intt_stages().expect("intt_stages");

        for (stage_idx, &m) in InverseStages::MS.iter().enumerate() {
            for k in (1..m).step_by(2) {
                assert_eq!(
                    s.inv_roots[stage_idx][k], 0u16,
                    "stage {stage_idx}, m {m}, k {k}: odd inv_roots entries must remain zero"
                );
            }
        }
    }

    #[test]
    fn test_intt_stages_inv_roots_outside_stage_size_are_zero() {
        let s = intt_stages().expect("intt_stages");

        for (stage_idx, &m) in InverseStages::MS.iter().enumerate() {
            // Everything from m..N must remain zero since intt_stages() only fills < m.
            for k in m..N {
                assert_eq!(
                    s.inv_roots[stage_idx][k], 0u16,
                    "stage {stage_idx}: inv_roots entries beyond m must remain zero"
                );
            }
        }
    }

    #[test]
    fn test_intt_stages_even_entries_are_nonzero() {
        // This is a mild sanity check to catch a “table never filled” bug.
        let s = intt_stages().expect("intt_stages");
        for (stage_idx, &m) in InverseStages::MS.iter().enumerate() {
            // Skip k=0 if you want to be ultra conservative; but inverse(1) is 1 so it should be nonzero anyway.
            for k in (0..m).step_by(2) {
                assert_ne!(
                    s.inv_roots[stage_idx][k], 0u16,
                    "stage {stage_idx}, m {m}, k {k}: expected nonzero inverse twiddle"
                );
            }
        }
    }

    #[test]
    fn test_intt_matches_intt_with_stages_on_same_input() {
        let stages = intt_stages().expect("intt_stages");

        // Start from a valid NTT-domain input (produced by forward NTT).
        let fwd = forward_stages().expect("forward_stages");
        let mut a = [0i16; 512];
        for i in 0..512 {
            a[i] = (i as i16).wrapping_mul(73).wrapping_sub(9000); // lots of negatives
        }
        ntt_with_stages(&mut a, &fwd).expect("ntt_with_stages");

        let mut got = a;
        let mut expect = a;

        intt(&mut got).expect("intt");
        intt_with_stages(&mut expect, &stages).expect("intt_with_stages");

        assert_eq!(got, expect, "intt wrapper must match explicit staged path");
    }

    #[test]
    fn test_intt_is_invariant_under_plus_minus_q_in_ntt_domain() {
        // Wrapper normalizes, so adding/subtracting q on inputs should not change result.
        let fwd = forward_stages().expect("forward_stages");

        // Any coefficient vector; we'll go to NTT domain to get a canonical NTT ordering.
        let mut coeffs = [0i16; 512];
        for i in 0..512 {
            coeffs[i] = (i as i16).wrapping_mul(-17).wrapping_add(1234);
        }

        let mut ntt_a = coeffs;
        ntt_with_stages(&mut ntt_a, &fwd).expect("ntt_with_stages");

        let mut base = ntt_a;
        intt(&mut base).expect("intt");

        let q = FALCON_Q as i16;

        let mut plus_q = ntt_a;
        for i in (0..512).step_by(5) {
            plus_q[i] = plus_q[i].wrapping_add(q);
        }
        intt(&mut plus_q).expect("intt");
        assert_eq!(plus_q, base, "intt(NTT + q) must equal intt(NTT)");

        let mut minus_q = ntt_a;
        for i in (0..512).step_by(5) {
            minus_q[i] = minus_q[i].wrapping_sub(q);
        }
        intt(&mut minus_q).expect("intt");
        assert_eq!(minus_q, base, "intt(NTT - q) must equal intt(NTT)");
    }

    #[test]
    fn test_intt_then_ntt_roundtrip_recovers_ntt_domain_canonical_values() {
        // This validates the wrapper normalization + full pipeline, without relying on KAT JSON.
        let fwd = forward_stages().expect("forward_stages");

        let mut coeffs = [0i16; 512];
        for i in 0..512 {
            coeffs[i] = (i as i16).wrapping_mul(101).wrapping_sub(20000); // wide
        }

        // Forward NTT to get canonical NTT-domain values.
        let mut ntt0 = coeffs;
        ntt_with_stages(&mut ntt0, &fwd).expect("ntt_with_stages");

        // Invert (via wrapper) back to coeffs (canonical residues).
        let mut back = ntt0;
        intt(&mut back).expect("intt");

        // Forward again; should return to the same canonical NTT representation.
        ntt_with_stages(&mut back, &fwd).expect("ntt_with_stages");
        assert_eq!(
            back, ntt0,
            "ntt(intt(NTT)) must equal original canonical NTT"
        );
    }

    #[inline]
    fn oracle_mod_q(x: u32) -> u16 {
        (x % Q) as u16
    }

    #[test]
    fn test_reduce_u32_mod_q_edge_cases_matches_modulus_oracle() {
        let qq = Q as u64;

        // Exact small / boundary values around Q
        let cases: [u32; 13] = [
            0,
            1,
            Q - 1,
            Q,
            Q + 1,
            2 * Q - 1,
            2 * Q,
            2 * Q + 1,
            3 * Q - 1,
            3 * Q,
            3 * Q + 1,
            4 * Q - 1,
            4 * Q,
        ];

        for &x in &cases {
            assert_eq!(reduce_u32_mod_q(x), oracle_mod_q(x), "mismatch at x={x}");
        }

        // Max and near-max products under the precondition x < Q^2
        let max_prod = ((Q - 1) as u64) * ((Q - 1) as u64); // (Q-1)^2
        let near_1 = ((Q - 1) as u64) * ((Q - 2) as u64);
        let near_2 = ((Q - 2) as u64) * ((Q - 2) as u64);

        for (label, x64) in [
            ("(Q-1)^2", max_prod),
            ("(Q-1)(Q-2)", near_1),
            ("(Q-2)^2", near_2),
        ] {
            assert!(x64 < qq * qq, "{label} violates x < Q^2");
            let x = x64 as u32;
            assert_eq!(
                reduce_u32_mod_q(x),
                oracle_mod_q(x),
                "mismatch at {label} (x={x})"
            );
        }
    }

    #[inline]
    fn sweep_band(start: u32, end_exclusive: u32, step: u32) {
        assert!(step != 0);
        let mut x = start;

        while x < end_exclusive {
            let got = reduce_u32_mod_q(x);
            let want = oracle_mod_q(x);
            assert_eq!(
                got, want,
                "mismatch at x={x} (start={start}, end={end_exclusive}, step={step})"
            );
            x = x.saturating_add(step);
            if x == u32::MAX {
                break;
            }
        }
    }

    #[test]
    fn test_reduce_u32_mod_q_structured_sweep_bands_match_modulus_oracle() {
        // We only need to cover the intended precondition range.
        // Here, end_max is the maximum x that can arise from (u16 mod Q)*(u16 mod Q).
        let q2 = (Q as u64) * (Q as u64);
        let end_max_inclusive = ((Q - 1) as u64) * ((Q - 1) as u64);
        assert!(end_max_inclusive < q2);

        let end_max_exclusive = (end_max_inclusive as u32).saturating_add(1);

        // Band A: small region where fix-up behavior is most visible
        // [0, 4Q)
        sweep_band(0, 4 * Q, 1);
        sweep_band(0, 4 * Q, 7);
        sweep_band(0, 4 * Q, 97);

        // Band B: around Q^2/2 (interior)
        // Choose a window of +/- 2Q around mid, then sample with a stride.
        let mid = ((q2 / 2) as u32).min(end_max_exclusive - 1);
        let lo = mid.saturating_sub(2 * Q);
        let hi = (mid.saturating_add(2 * Q)).min(end_max_exclusive);

        sweep_band(lo, hi, 13);
        sweep_band(lo, hi, 257);

        // Band C: near the top end [Q^2 - 4Q, Q^2) but capped to the true max under the precondition.
        // Since our actual max is (Q-1)^2, sweep the last ~4Q values before that.
        let top = end_max_exclusive;
        let lo_top = top.saturating_sub(4 * Q);

        sweep_band(lo_top, top, 1);
        sweep_band(lo_top, top, 19);
        sweep_band(lo_top, top, 509);
    }

    #[test]
    fn test_reduce_u32_mod_q_random_pairs_matches_modulus_oracle_512_checks() {
        let mut rng = XorShift64::new(0xBADC0FFE_EE0DDF00);

        for i in 0..512 {
            let a = rng.next_u16_mod_q() as u32;
            let b = rng.next_u16_mod_q() as u32;
            let x = a * b;

            let r = reduce_u32_mod_q(x);
            assert!(
                (r as u32) < Q,
                "reducer returned out-of-range value at iter={i}: r={r}, x={x}, a={a}, b={b}"
            );

            // And still matches the oracle (this keeps the test from being "range only")
            assert_eq!(
                r,
                oracle_mod_q(x),
                "oracle mismatch at iter={i}: r={r}, x={x}, a={a}, b={b}"
            );
        }
    }

    #[test]
    fn test_pointwise_mul_in_place_matches_reference_mul_mod_q_on_random_arrays() {
        let mut rng = XorShift64::new(0xD00D_F00D_BA5E_CAFE);

        for iter in 0..256 {
            let mut a_u16 = [0u16; 512];
            let mut b_u16 = [0u16; 512];

            for i in 0..512 {
                a_u16[i] = rng.next_u16_mod_q();
                b_u16[i] = rng.next_u16_mod_q();
            }

            // Reference (u16) output using the simple `% Q` multiply.
            let mut out_ref = [0u16; 512];
            for i in 0..512 {
                out_ref[i] = mul_mod_q(a_u16[i], b_u16[i]);
            }

            // Fast path uses i16 vectors (matches ntt/intt representation).
            let mut a_i16 = [0i16; 512];
            let mut b_i16 = [0i16; 512];
            for i in 0..512 {
                a_i16[i] = a_u16[i] as i16;
                b_i16[i] = b_u16[i] as i16;
            }

            pointwise_mul_in_place(&mut a_i16, &b_i16).unwrap();

            // Compare modulo-Q, lane by lane, and check range.
            for i in 0..512 {
                let got = a_i16[i] as i32;
                assert!(
                    (0..(Q as i32)).contains(&got),
                    "pointwise_mul_in_place produced out-of-range at iter={iter}, idx={i}, v={got}"
                );

                assert_eq!(
                    got as u16, out_ref[i],
                    "pointwise_mul_in_place mismatch at iter={iter}, idx={i}"
                );
            }
        }
    }

    #[inline]
    fn norm_q(x: u16) -> u16 {
        // If your NTT/INTT already maintain [0,Q), you can omit this,
        // but leaving it makes the test robust to minor implementation choices.
        if (x as u32) >= Q {
            ((x as u32) % Q) as u16
        } else {
            x
        }
    }

    /// Map any i16 to a canonical residue in [0, Q) as u16.
    /// This is the equality notion we care about for NTT/INTT roundtrips.
    #[inline]
    fn canon_mod_q(x: i16) -> u16 {
        let q = FALCON_Q as i32;
        let mut v = x as i32 % q;
        if v < 0 {
            v += q;
        }
        v as u16
    }

    /// Assert two i16 coefficient vectors are equal modulo Q, coefficient-wise.
    #[inline]
    fn assert_eq_mod_q(a: &[i16; 512], b: &[i16; 512], ctx: &str) {
        for i in 0..512 {
            let aa = canon_mod_q(a[i]);
            let bb = canon_mod_q(b[i]);
            assert_eq!(
                aa, bb,
                "{ctx}: mismatch at idx={i}: a[i]={} (modQ={aa}), b[i]={} (modQ={bb})",
                a[i], b[i]
            );
        }
    }

    /// Roundtrip: intt(ntt(x)) == x (mod Q), for many random vectors in [0, Q).
    #[test]
    fn ntt_intt_roundtrip_random_vectors_in_0_to_q() {
        let q = FALCON_Q as u64;
        let mut rng = XorShift64::new(0xC0FF_EE00);

        // "Many": 100 independent random vectors, each 512 coefficients.
        for iter in 0..100 {
            let mut a = [0i16; 512];
            for i in 0..512 {
                a[i] = (rng.next_u64() % q) as i16; // in [0, Q)
            }

            let orig = a;

            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();

            assert_eq_mod_q(&a, &orig, &format!("roundtrip [0,Q) iter={iter}"));
        }
    }

    /// Roundtrip: intt(ntt(x)) == x (mod Q), for many random vectors spanning full i16 range.
    /// This exercises the normalization path in `ntt()` and any sign handling.
    #[test]
    fn ntt_intt_roundtrip_random_vectors_wide_i16() {
        let mut rng = XorShift64::new(0xFEED_FACE_D00D_BEEF);

        for iter in 0..100 {
            let mut a = [0i16; 512];
            for i in 0..512 {
                a[i] = rng.next_i16_wide(); // includes negatives
            }

            let orig = a;

            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();

            assert_eq_mod_q(&a, &orig, &format!("roundtrip wide-i16 iter={iter}"));
        }
    }

    /// A couple of structured edge patterns to catch "looks random-proof but fails on structure".
    #[test]
    fn ntt_intt_roundtrip_structured_vectors() {
        let q = FALCON_Q as i16;

        // All zeros
        {
            let mut a = [0i16; 512];
            let orig = a;
            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();
            assert_eq_mod_q(&a, &orig, "structured all-zero");
        }

        // All (Q-1)
        {
            let mut a = [q - 1; 512];
            let orig = a;
            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();
            assert_eq_mod_q(&a, &orig, "structured all-(Q-1)");
        }

        // Alternating 0 / (Q-1)
        {
            let mut a = [0i16; 512];
            for i in 0..512 {
                a[i] = if (i & 1) == 0 { 0 } else { q - 1 };
            }
            let orig = a;
            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();
            assert_eq_mod_q(&a, &orig, "structured alternating");
        }

        // Ramp: i mod Q, plus a negative ramp to hit both signs deterministically
        {
            let mut a = [0i16; 512];
            for i in 0..512 {
                a[i] = (i as i16) % q;
            }
            let orig = a;
            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();
            assert_eq_mod_q(&a, &orig, "structured ramp");
        }

        {
            let mut a = [0i16; 512];
            for i in 0..512 {
                a[i] = -((i as i16) % q);
            }
            let orig = a;
            ntt(&mut a).unwrap();
            intt(&mut a).unwrap();
            assert_eq_mod_q(&a, &orig, "structured negative-ramp");
        }
    }
}

//! Falcon core verification implementation.
//!
//! This module implements the Falcon-512 core verification algorithm,
//! which consumes a signature, public key, and challenge polynomial
//! and checks the Falcon norm bound. It is independent of the
//! Hash-to-Point construction used to derive the challenge.

use crate::{
    crypto,
    falcon::{
        encoding::{self, unpack_falcon_14bit_be_polynomial},
        error::FalconError,
        ntt::{intt, normalize_in_place, ntt, pointwise_mul_in_place},
        sig_reader::SigReader,
        utils::map_falcon_result,
        UnpackedPublicKey, UnpackedSignature, ACCEPTANCE_BOUND_BETA2, FALCON_CORE_VERIFY_GAS,
        FALCON_N, FALCON_Q,
    },
    utilities::bool_to_bytes32,
    PrecompileError, PrecompileOutput, PrecompileResult,
};

/// Falcon-512 core verification precompile entrypoint.
pub fn falcon_core(input: &[u8], gas_limit: u64) -> PrecompileResult {
    if gas_limit < FALCON_CORE_VERIFY_GAS {
        return Err(PrecompileError::OutOfGas);
    }
    map_falcon_result(verify(input), FALCON_CORE_VERIFY_GAS)
}

#[inline]
fn verify(input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    let (sig_coefficients, pk, challenge) = extract_inputs_if_valid(input)?;
    let valid = crypto().falcon_core_verify(&sig_coefficients, &pk, &challenge)?;
    let out = valid.then(|| bool_to_bytes32(true)).unwrap_or_default();

    Ok(PrecompileOutput::new(FALCON_CORE_VERIFY_GAS, out))
}

pub(crate) fn falcon_core_verify(
    sig: &UnpackedSig,
    pubkey_ntt: &UnpackedPk, // NOTE: This is `h` already in NTT domain, packed as 14-bit coeffs.
    challenge: &UnpackedChallenge,
) -> Result<bool, FalconError> {
    // --- Step 0: Canonicality checks (defense in depth) ---
    //
    // unpack_falcon_14bit_be_polynomial() should already reject coefficients >= Q,
    // but keep the invariants local and obvious.
    for &c in pubkey_ntt.iter() {
        if c >= FALCON_Q {
            return Ok(false);
        }
    }
    for &c in challenge.iter() {
        if c >= FALCON_Q {
            return Ok(false);
        }
    }

    // --- Step 1: Prepare s2 (signature vector) ---
    //
    // SigReader yields signed coefficients (i32). For the NTT pipeline we use i16.
    // Reject if any coefficient cannot be represented as i16.
    let mut s2 = [0i16; FALCON_N];
    for i in 0..FALCON_N {
        let v = sig[i];
        if v < i16::MIN as i32 || v > i16::MAX as i32 {
            return Err(FalconError::InvalidSignatureEncoding);
        }
        s2[i] = v as i16;
    }

    // Preserve the original signed s2 for the norm check (spec squares s2 as-is).
    let s2_for_norm = sig;

    // --- Step 2: Convert s2 to evaluation domain (NTT) ---
    //
    // Spec: s2_ntt = ntt(s2)
    let mut s2_ntt = s2;
    ntt(&mut s2_ntt)?;

    // --- Step 3: Compute tmp_ntt = hadamard_product(s2_ntt, h) ---
    //
    // Spec: h is already in NTT domain. DO NOT call ntt(h) here.
    let mut h_ntt = [0i16; FALCON_N];
    for i in 0..FALCON_N {
        // safe because pubkey_ntt[i] < Q < i16::MAX
        h_ntt[i] = pubkey_ntt[i] as i16;
    }

    // Pointwise multiply in-place on s2_ntt: s2_ntt[i] = s2_ntt[i] * h_ntt[i] (mod Q).
    // This function should normalize internally, so it is robust even if s2_ntt contains
    // negative representatives at any point.
    pointwise_mul_in_place(&mut s2_ntt, &h_ntt)?;

    // --- Step 4: tmp = intt(tmp_ntt) ---
    //
    // Spec: tmp = intt(tmp_ntt), coefficient form modulo q.
    intt(&mut s2_ntt)?;

    // IMPORTANT: ensure tmp is in canonical residues [0, Q) before subtracting from challenge.
    // If `intt()` already guarantees this, this is redundant but harmless and clarifies intent.
    normalize_in_place(&mut s2_ntt);

    // At this point, `s2_ntt` holds tmp in canonical [0,Q) (stored as i16).

    // --- Step 5: s1 = challenge - tmp (mod Q), then center to [-Q/2, Q/2] ---
    //
    // Spec:
    //     s1 = challenge - tmp
    //     s1 = normalize_coefficients(s1, q)  (i.e. center)
    let mut s1_centered = [0i16; FALCON_N];
    for i in 0..FALCON_N {
        let c = challenge[i] as i16; // canonical [0,Q)
        let t = s2_ntt[i]; // canonical [0,Q)
        let s1_can = sub_mod_q_i16(c, t); // canonical [0,Q)
        s1_centered[i] = center_mod_q(s1_can);
    }

    // --- Step 6: Norm bound check ---
    //
    // Spec: total_norm = sum(s1^2) + sum(s2^2), return total_norm < ACCEPTANCE_BOUND.
    let mut acc: i64 = 0;
    for i in 0..FALCON_N {
        let s1 = s1_centered[i] as i64;
        let s2 = s2_for_norm[i] as i64;

        acc += s1 * s1;
        acc += s2 * s2;

        // Deterministic early exit.
        if acc >= ACCEPTANCE_BOUND_BETA2 {
            return Ok(false);
        }
    }

    Ok(acc < ACCEPTANCE_BOUND_BETA2)
}

/// Forward-compatible helper: center a canonical residue in [0, Q) into [-Q/2, Q/2].
///
/// For Falcon q=12289 (odd), Q/2 is 6144 (floor), so the result is in [-6144, 6144].
#[inline(always)]
fn center_mod_q(x: i16) -> i16 {
    debug_assert!((0..(FALCON_Q as i16)).contains(&x));
    let half_q: i16 = (FALCON_Q as i16) / 2;
    if x > half_q {
        x - (FALCON_Q as i16)
    } else {
        x
    }
}

/// Subtract in Z_q given canonical residues a,b in [0,Q):
/// returns (a-b) mod Q, still canonical in [0,Q).
#[inline(always)]
fn sub_mod_q_i16(a: i16, b: i16) -> i16 {
    debug_assert!((0..(FALCON_Q as i16)).contains(&a));
    debug_assert!((0..(FALCON_Q as i16)).contains(&b));
    let mut d = a - b;
    if d < 0 {
        d += FALCON_Q as i16;
    }
    d
}

type UnpackedSig = [i32; FALCON_N];
type UnpackedPk = [u16; FALCON_N];
type UnpackedChallenge = [u16; FALCON_N];

#[inline]
fn extract_inputs_if_valid<'a>(
    input: &'a [u8],
) -> Result<(UnpackedSig, UnpackedPk, UnpackedChallenge), FalconError> {
    let (sig, pk, challenge) = encoding::split_falcon_core_input(input)?;
    let (_salt, s2_compressed) = encoding::split_sig(sig)?;

    let pk_unpacked = unpack_falcon_14bit_be_polynomial(pk)?;
    let challenge_unpacked = unpack_falcon_14bit_be_polynomial(challenge)?;
    let sig_unpacked = SigReader::new(s2_compressed)
        .read_coefficients()
        .ok_or(FalconError::InvalidSignatureEncoding)?;

    Ok((sig_unpacked, pk_unpacked, challenge_unpacked))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::encoding::pack_falcon_14bit_be_polynomial;
    use crate::falcon::utils::test::{
        bits_to_buf, push_coeff, sample_14bit_coeff, set_coeff_14bit_packed,
    };
    use crate::falcon::{CHALLENGE_LEN, FALCON_Q, PK_LEN, S2_COMPRESSED_LEN, SIG_LEN};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_extract_inputs_if_valid_with_invalid_length() {
        // Too short input
        let input = vec![0u8; 10];
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_empty_input() {
        let input = vec![];
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_too_long_input() {
        // Input longer than expected
        let mut input = Vec::new();
        input.extend(vec![1u8; S2_COMPRESSED_LEN]);
        input.extend(vec![2u8; PK_LEN]);
        input.extend(vec![3u8; CHALLENGE_LEN]);
        input.extend(vec![4u8; 5]); // Extra bytes
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_falcon_core_gas_limit_below() {
        let input = vec![0u8; 10];
        let gas_limit = 1999;
        let result = falcon_core(&input, gas_limit);
        assert!(matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_falcon_core_gas_limit_zero() {
        let input = vec![0u8; 10];
        let gas_limit = 0;
        let result = falcon_core(&input, gas_limit);
        assert!(matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_falcon_core_gas_limit_large() {
        let input = vec![0u8; 10];
        let gas_limit = 10_000;
        let result = falcon_core(&input, gas_limit);

        assert!(result.is_ok());
        let out = result.unwrap();
        assert!(out.gas_used == FALCON_CORE_VERIFY_GAS);
        assert!(out.bytes.is_empty());
    }

    #[test]
    fn test_extract_inputs_if_valid_roundtrip_seeded_coefficients() {
        // Deterministic seeded RNG so the test is reproducible.
        let mut rng = StdRng::seed_from_u64(0xABCD1234);

        // Seeded-random signature bytes.
        let mut sig = [0u8; SIG_LEN];
        for b in sig[..40].iter_mut() {
            *b = rng.random::<u8>();
        }

        let mut sig_coefficients = Vec::new();
        for _ in 0..512 {
            sig_coefficients.push(rng.random::<i8>() as i32);
        }
        let mut s2_compressed_bits: Vec<u8> = Vec::new();
        for &coeff in sig_coefficients.iter() {
            push_coeff(&mut s2_compressed_bits, coeff);
        }

        // append sig_coefficients after the salt
        let s2_compressed = bits_to_buf::<S2_COMPRESSED_LEN>(&s2_compressed_bits);
        assert_eq!(SIG_LEN, 40 + s2_compressed.len());
        sig[40..40 + s2_compressed.len()].copy_from_slice(s2_compressed.as_slice());

        // Seeded-random valid coefficients (< q) for pk and challenge.
        let mut pk_coeffs = [0u16; FALCON_N];
        let mut challenge_coeffs = [0u16; FALCON_N];
        for i in 0..FALCON_N {
            pk_coeffs[i] = sample_14bit_coeff(&mut rng, true);
            challenge_coeffs[i] = sample_14bit_coeff(&mut rng, true);
        }

        let pk_packed = pack_falcon_14bit_be_polynomial(&pk_coeffs).unwrap();
        let challenge_packed = pack_falcon_14bit_be_polynomial(&challenge_coeffs).unwrap();

        // One contiguous input buffer: sig || pk || challenge
        let mut input = Vec::with_capacity(SIG_LEN + PK_LEN + CHALLENGE_LEN);
        input.extend_from_slice(&sig);
        input.extend_from_slice(pk_packed.as_ref());
        input.extend_from_slice(challenge_packed.as_ref());

        let (sig_out, pk_unpacked, challenge_unpacked) = extract_inputs_if_valid(&input).unwrap();

        assert_eq!(&sig_out, sig_coefficients.as_slice());
        assert_eq!(pk_unpacked, pk_coeffs);
        assert_eq!(challenge_unpacked, challenge_coeffs);
    }

    #[test]
    fn test_extract_inputs_if_valid_with_challenge_coefficient_equal_q() {
        // Deterministic seeded RNG so the test is reproducible.
        let mut rng = StdRng::seed_from_u64(0xABCD1234);

        // Seeded-random signature bytes.
        let mut sig = [0u8; SIG_LEN];
        for b in sig.iter_mut() {
            *b = rng.random::<u8>();
        }

        // Seeded-random valid coefficients (< q) for pk and challenge.
        let mut pk_coeffs = [0u16; FALCON_N];
        let mut challenge_coeffs = [0u16; FALCON_N];
        for i in 0..FALCON_N {
            pk_coeffs[i] = sample_14bit_coeff(&mut rng, true);
            challenge_coeffs[i] = sample_14bit_coeff(&mut rng, true);
        }

        let pk_packed = pack_falcon_14bit_be_polynomial(&pk_coeffs).unwrap();
        let mut challenge_packed = pack_falcon_14bit_be_polynomial(&challenge_coeffs).unwrap();

        // Corrupt one challenge coefficient after packing: set it to Q (invalid: must be < Q).
        set_coeff_14bit_packed(challenge_packed.as_mut(), 123, FALCON_Q);
        // Padding must remain canonical.
        assert_eq!(challenge_packed[CHALLENGE_LEN - 1], 0);

        // One contiguous input buffer: sig || pk || challenge
        let mut input = Vec::with_capacity(SIG_LEN + PK_LEN + CHALLENGE_LEN);
        input.extend_from_slice(&sig);
        input.extend_from_slice(pk_packed.as_ref());
        input.extend_from_slice(challenge_packed.as_ref());

        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
        assert!(matches!(result, Err(FalconError::InvalidFieldElement)));
    }

    #[test]
    fn test_falcon_core_malformed_input_returns_empty_output() {
        let input = vec![0u8; 10];
        let out = falcon_core(&input, 10_000).expect("should not error");
        assert_eq!(out.gas_used, 2000);
        assert!(out.bytes.is_empty());
    }
}

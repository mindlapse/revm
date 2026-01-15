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
        sig_reader::SigReader,
        utils::map_falcon_result,
        UnpackedPublicKey, UnpackedSignature, FALCON_CORE_VERIFY_GAS, FALCON_N,
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
    _sig: &UnpackedSignature,
    _pk: &UnpackedPublicKey,
    _challenge: &UnpackedChallenge,
) -> Result<bool, FalconError> {
    Ok(true) // TODO
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
        bits_to_buf, push_coeff, sample_14bit_coeff, set_coeff_14bit_be,
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
        set_coeff_14bit_be(challenge_packed.as_mut(), 123, FALCON_Q);
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

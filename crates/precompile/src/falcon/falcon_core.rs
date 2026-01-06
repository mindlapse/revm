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
        utils::map_falcon_result,
        FALCON_CORE_VERFIFY_GAS, FALCON_N,
    },
    utilities::bool_to_bytes32,
    PrecompileError, PrecompileOutput, PrecompileResult,
};

/// Falcon-512 core verification precompile entrypoint.
pub fn falcon_core(input: &[u8], gas_limit: u64) -> PrecompileResult {
    if gas_limit < FALCON_CORE_VERFIFY_GAS {
        return Err(PrecompileError::OutOfGas);
    }
    map_falcon_result(verify(input), FALCON_CORE_VERFIFY_GAS)
}

#[inline]
fn verify(input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    let (sig, pk, challenge) = extract_inputs_if_valid(input)?;
    let valid = crypto().falcon_core_verify(sig, &pk, &challenge)?;
    let out = valid.then(|| bool_to_bytes32(true)).unwrap_or_default();

    Ok(PrecompileOutput::new(FALCON_CORE_VERFIFY_GAS, out))
}

type SignatureBytes = [u8; 666];
type UnpackedPk = [u16; FALCON_N];
type UnpackedChallenge = [u16; FALCON_N];

#[inline]
fn extract_inputs_if_valid<'a>(
    input: &'a [u8],
) -> Result<(&'a SignatureBytes, UnpackedPk, UnpackedChallenge), FalconError> {
    let (sig, pk, challenge) = encoding::split_falcon_core_input(input)?;

    let pk_unpacked = unpack_falcon_14bit_be_polynomial(pk)?;
    let challenge_unpacked = unpack_falcon_14bit_be_polynomial(challenge)?;

    Ok((sig, pk_unpacked, challenge_unpacked))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::encoding::pack_falcon_14bit_be_polynomial;
    use crate::falcon::utils::test::sample_14bit_coeff;
    use crate::falcon::{CHALLENGE_LEN, PK_LEN, SIG_LEN};
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
        input.extend(vec![1u8; SIG_LEN]);
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
        // Should not be OutOfGas, but should fail with SpecNotFinalized
        assert!(!matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_extract_inputs_if_valid_roundtrip_seeded_coefficients() {
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
        let challenge_packed = pack_falcon_14bit_be_polynomial(&challenge_coeffs).unwrap();

        // One contiguous input buffer: sig || pk || challenge
        let mut input = Vec::with_capacity(SIG_LEN + PK_LEN + CHALLENGE_LEN);
        input.extend_from_slice(&sig);
        input.extend_from_slice(pk_packed.as_ref());
        input.extend_from_slice(challenge_packed.as_ref());

        let (sig_out, pk_unpacked, challenge_unpacked) = extract_inputs_if_valid(&input).unwrap();

        assert_eq!(sig_out, &sig);
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

        // Set one challenge coefficient to the invalid value 12289 (equal to q).
        challenge_coeffs[123] = 12289u16;

        let pk_packed = pack_falcon_14bit_be_polynomial(&pk_coeffs).unwrap();
        let challenge_packed = pack_falcon_14bit_be_polynomial(&challenge_coeffs).unwrap();

        // One contiguous input buffer: sig || pk || challenge
        let mut input = Vec::with_capacity(SIG_LEN + PK_LEN + CHALLENGE_LEN);
        input.extend_from_slice(&sig);
        input.extend_from_slice(pk_packed.as_ref());
        input.extend_from_slice(challenge_packed.as_ref());

        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
        assert!(matches!(result, Err(FalconError::InvalidFieldElement)));
    }
}

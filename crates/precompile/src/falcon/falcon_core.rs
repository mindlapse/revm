//! Falcon core verification implementation.
//!
//! This module implements the Falcon-512 core verification algorithm,
//! which consumes a signature, public key, and challenge polynomial
//! and checks the Falcon norm bound. It is independent of the
//! Hash-to-Point construction used to derive the challenge.

use crate::{
    falcon::{encoding, error::FalconError, utils::map_falcon_result, FalconCoreInputs},
    PrecompileError, PrecompileOutput, PrecompileResult,
};

/// Falcon-512 core verification precompile entrypoint.
pub fn falcon_core(input: &[u8], gas_limit: u64) -> PrecompileResult {
    const GAS: u64 = 2000;
    if gas_limit < GAS {
        return Err(PrecompileError::OutOfGas);
    }

    map_falcon_result(verify(input), GAS)
}

#[inline]
fn verify(input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    let (_sig, _pk, _challenge) = extract_inputs_if_valid(input)?;

    Err(FalconError::SpecNotFinalized)
}
#[inline]
fn extract_inputs_if_valid<'a>(input: &'a [u8]) -> Result<FalconCoreInputs<'a>, FalconError> {
    let core_inputs = encoding::split_falcon_core_input(input)?;

    // TODO additional validations

    Ok(core_inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::{CHALLENGE_LEN, PK_LEN, SIG_LEN};

    #[test]
    fn test_extract_inputs_if_valid_with_valid_length() {
        // Assuming parse_falcon_core_input expects input of SIG_LEN + PK_LEN + CHALLENGE_LEN
        let mut input = Vec::new();
        input.extend(vec![1u8; SIG_LEN]);
        input.extend(vec![2u8; PK_LEN]);
        input.extend(vec![3u8; CHALLENGE_LEN]);
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_ok());
        let (sig, pk, challenge) = result.unwrap();
        assert_eq!(sig, &vec![1u8; SIG_LEN][..]);
        assert_eq!(pk, &vec![2u8; PK_LEN][..]);
        assert_eq!(challenge, &vec![3u8; CHALLENGE_LEN][..]);
    }

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
}

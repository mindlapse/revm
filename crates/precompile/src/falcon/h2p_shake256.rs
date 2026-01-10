//! Falcon Hash-to-Point (H2P) implementation using SHAKE256.
//!
//! This module implements the NIST-compliant Hash-to-Point (H2P)
//! challenge generation step for Falcon-512, as specified in EIP-8052.
//! It takes a 32-byte message hash and a 666-byte Falcon signature
//! and deterministically derives a packed challenge polynomial.
//!
//! Once a challenge is obtained, it can then be passed to FALCON_CORE
//! with the signature and public key for signature verification,
//! provided that the signature was generated using the same H2P method.

use crate::{
    crypto,
    falcon::{
        encoding::{pack_falcon_14bit_be_polynomial, split_sig},
        error::FalconError,
        utils::map_falcon_result,
        H2PInputs, H2P_GAS,
    },
    PrecompileError, PrecompileOutput, PrecompileResult,
};
use sha3 as _;

pub fn h2p_shake256(input: &[u8], gas_limit: u64) -> PrecompileResult {
    if gas_limit < H2P_GAS {
        return Err(PrecompileError::OutOfGas);
    }
    map_falcon_result(compute_h2p(input), H2P_GAS)
}

#[inline]
fn compute_h2p(input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    let (msg, sig) = extract_inputs_if_valid(input)?;
    let (salt, _) = split_sig(sig)?;
    let challenge = crypto().falcon_h2p_shake256(msg, salt)?;
    let packed_challenge: Box<[u8]> = pack_falcon_14bit_be_polynomial(&challenge)?;
    Ok(PrecompileOutput::new(H2P_GAS, packed_challenge.into()))
}

#[inline]
fn extract_inputs_if_valid<'a>(input: &'a [u8]) -> Result<H2PInputs<'a>, FalconError> {
    Ok(crate::falcon::encoding::split_h2p_input(input)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_inputs_if_valid_with_valid_length() {
        // Assuming valid input length for H2PInputs is 698 bytes (32 + 666)
        let input = vec![0u8; 698];
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_empty_input() {
        let input = vec![];
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_too_short_length() {
        let input = vec![0u8; 10]; // Much shorter than required
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_too_long_input() {
        let input = vec![0u8; 1000]; // Longer than required
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_h2p_shake256_gas_limit_below() {
        let input = vec![0u8; 10];
        let gas_limit = 999;
        let result = h2p_shake256(&input, gas_limit);
        assert!(matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_h2p_shake256_gas_limit_zero() {
        let input = vec![0u8; 10];
        let gas_limit = 0;
        let result = h2p_shake256(&input, gas_limit);
        assert!(matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_h2p_shake256_gas_limit_large() {
        let input = vec![0u8; 10];
        let gas_limit = 10_000;
        let result = h2p_shake256(&input, gas_limit);

        assert!(result.is_ok());
        let out = result.unwrap();
        assert!(out.gas_used == H2P_GAS);
        assert!(out.bytes.is_empty());
    }

    #[test]
    fn test_h2p_shake256_malformed_input_returns_empty_output() {
        let input = vec![0u8; 10];
        let out = h2p_shake256(&input, 10_000).expect("should not error");
        assert_eq!(out.gas_used, 1000);
        assert!(out.bytes.is_empty());
    }
}

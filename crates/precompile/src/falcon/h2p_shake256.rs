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

use sha3 as _;
use crate::{PrecompileError, PrecompileOutput, PrecompileResult, falcon::{error::FalconError, utils::map_falcon_result}};

pub fn h2p_shake256(input: &[u8], gas_limit: u64) -> PrecompileResult {

    const GAS: u64 = 1000;
    if gas_limit < GAS {
        return Err(PrecompileError::OutOfGas);
    }
    map_falcon_result(compute_h2p(input), GAS)
}

fn compute_h2p(_input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    Err(FalconError::SpecNotFinalized)
}

#[cfg(test)]
mod tests {
    use super::*;


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
        // Should not be OutOfGas, but should fail with SpecNotFinalized
        assert!(!matches!(result, Err(PrecompileError::OutOfGas)));
    }
}

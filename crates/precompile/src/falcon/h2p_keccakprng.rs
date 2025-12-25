//! Falcon Hash-to-Point (H2P) implementation using a Keccak-based PRNG.
//!
//! This is an EVM-friendly alternative to the SHAKE256-based H2P,
//! intended to reduce gas costs by leveraging the existing Keccak256
//! precompile. This module is feature-gated because the Keccak-PRNG
//! construction is still subject to final specification in EIP-8052.
//!
//! Once a challenge is obtained, it can then be passed to FALCON_CORE
//! with the signature and public key for signature verification,
//! provided that the signature was generated using the same H2P method.

use crate::{PrecompileError, PrecompileOutput, PrecompileResult, falcon::{error::FalconError, utils::map_falcon_result}};

pub fn h2p_keccakprng(input: &[u8], gas_limit: u64) -> PrecompileResult {

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
    fn test_h2p_keccakprng_gas_limit_below() {
        let input = vec![0u8; 10];
        let gas_limit = 999;
        let result = h2p_keccakprng(&input, gas_limit);
        assert!(matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_h2p_keccakprng_gas_limit_zero() {
        let input = vec![0u8; 10];
        let gas_limit = 0;
        let result = h2p_keccakprng(&input, gas_limit);
        assert!(matches!(result, Err(PrecompileError::OutOfGas)));
    }

    #[test]
    fn test_h2p_keccakprng_gas_limit_large() {
        let input = vec![0u8; 10];
        let gas_limit = 10_000;
        let result = h2p_keccakprng(&input, gas_limit);
        // Should not be OutOfGas, but should fail with SpecNotFinalized
        assert!(!matches!(result, Err(PrecompileError::OutOfGas)));
    }
}

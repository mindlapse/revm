//! Falcon core verification implementation.
//!
//! This module implements the Falcon-512 core verification algorithm,
//! which consumes a signature, public key, and challenge polynomial
//! and checks the Falcon norm bound. It is independent of the
//! Hash-to-Point construction used to derive the challenge.

use crate::{PrecompileError, PrecompileOutput, PrecompileResult, falcon::{error::FalconError, utils::map_falcon_result}};


pub fn falcon_core(input: &[u8], gas_limit: u64) -> PrecompileResult {

    const GAS: u64 = 2000;
    if gas_limit < GAS {
        return Err(PrecompileError::OutOfGas);
    }
    map_falcon_result(verify(input), GAS)
}

fn verify(_input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    Err(FalconError::SpecNotFinalized)
}

#[cfg(test)]
mod tests {
    use super::*;


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

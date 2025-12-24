//! Falcon utilities for error mapping

use primitives::Bytes;

use crate::{PrecompileError, PrecompileOutput};
use crate::falcon::error::FalconError;

/// Fixed-cost precompile pattern:
/// - If OOG: return Err(OutOfGas)
/// - Otherwise: return Ok(empty bytes) on any Falcon failure.
///
/// This mirrors ECRECOVER-style behavior while preserving internal failure reasons
/// for tests / debugging (via `FalconError`), without exposing them as public errors.
#[inline]
pub(super) fn transform_falcon_result(
    res: Result<PrecompileOutput, FalconError>,
    gas_used: u64,
) -> Result<PrecompileOutput, PrecompileError> {
    match res {
        Ok(out) => Ok(out),
        Err(reason) => {
            match reason {
                FalconError::OutOfGas => Err(PrecompileError::OutOfGas),
                _ => Ok(PrecompileOutput::new(gas_used, Bytes::new())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_output(gas: u64, data: &[u8]) -> PrecompileOutput {
        PrecompileOutput::new(gas, Bytes::copy_from_slice(data))
    }

    #[test]
    fn test_transform_falcon_result_ok() {
        let output = dummy_output(100, b"abc");
        let res = Ok(output.clone());
        let result = transform_falcon_result(res, 100);
        assert_eq!(result, Ok(output));
    }

    #[test]
    fn test_transform_falcon_result_out_of_gas() {
        let res = Err(FalconError::OutOfGas);
        let result = transform_falcon_result(res, 1234);
        assert_eq!(result, Err(PrecompileError::OutOfGas));
    }

    #[test]
    fn test_transform_falcon_result_invalid_input_length() {
        let res = Err(FalconError::InvalidInputLength { expected: 32, got: 16 });
        let result = transform_falcon_result(res, 55);
        assert_eq!(result, Ok(PrecompileOutput::new(55, Bytes::new())));
    }

    #[test]
    fn test_transform_falcon_result_invalid_field_element() {
        let res = Err(FalconError::InvalidFieldElement);
        let result = transform_falcon_result(res, 77);
        assert_eq!(result, Ok(PrecompileOutput::new(77, Bytes::new())));
    }

    #[test]
    fn test_transform_falcon_result_decompression_failed() {
        let res = Err(FalconError::DecompressionFailed);
        let result = transform_falcon_result(res, 88);
        assert_eq!(result, Ok(PrecompileOutput::new(88, Bytes::new())));
    }

    #[test]
    fn test_transform_falcon_result_spec_not_finalized() {
        let res = Err(FalconError::SpecNotFinalized);
        let result = transform_falcon_result(res, 99);
        assert_eq!(result, Ok(PrecompileOutput::new(99, Bytes::new())));
    }
}

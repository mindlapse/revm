//! Falcon utilities for error mapping

use primitives::Bytes;

use crate::falcon::error::FalconError;
use crate::{PrecompileOutput, PrecompileResult};

/// Fixed-cost precompile pattern:
/// - If OOG: return Err(OutOfGas)
/// - Otherwise: return Ok(empty bytes) on any Falcon failure.
///
/// This mirrors ECRECOVER-style behavior while preserving internal failure reasons
/// for tests / debugging (via `FalconError`), without exposing them as public errors.
#[inline]
pub(super) fn map_falcon_result(
    res: Result<PrecompileOutput, FalconError>,
    gas_used: u64,
) -> PrecompileResult {
    match res {
        Ok(out) => Ok(out),
        Err(_reason) => Ok(PrecompileOutput::new(gas_used, Bytes::new())),
    }
}

#[cfg(test)]
pub(in crate::falcon) mod test {
    use crate::falcon::{CHALLENGE_LEN, COEFF_BITS, FALCON_N, FALCON_Q};

    use super::*;

    fn dummy_output(gas: u64, data: &[u8]) -> PrecompileOutput {
        PrecompileOutput::new(gas, Bytes::copy_from_slice(data))
    }

    #[test]
    fn test_transform_falcon_result_ok() {
        let output = dummy_output(100, b"abc");
        let res = Ok(output.clone());
        let result = map_falcon_result(res, 100);
        assert_eq!(result, Ok(output));
    }

    #[test]
    fn test_transform_falcon_result_invalid_input_length() {
        let res = Err(FalconError::InvalidInputLength {
            wanted: 32,
            got: 16,
        });
        let result = map_falcon_result(res, 55);
        assert_eq!(result, Ok(PrecompileOutput::new(55, Bytes::new())));
    }

    #[test]
    fn test_transform_falcon_result_invalid_field_element() {
        let res = Err(FalconError::InvalidFieldElement);
        let result = map_falcon_result(res, 77);
        assert_eq!(result, Ok(PrecompileOutput::new(77, Bytes::new())));
    }

    #[test]
    fn test_transform_falcon_result_decompression_failed() {
        let res = Err(FalconError::DecompressionFailed);
        let result = map_falcon_result(res, 88);
        assert_eq!(result, Ok(PrecompileOutput::new(88, Bytes::new())));
    }

    #[test]
    fn test_transform_falcon_result_spec_not_finalized() {
        let res = Err(FalconError::SpecNotFinalized);
        let result = map_falcon_result(res, 99);
        assert_eq!(result, Ok(PrecompileOutput::new(99, Bytes::new())));
    }

    pub(in crate::falcon) fn create_sample_falcon_coefficients() -> [u16; FALCON_N] {
        let mut rng = rand::rng();
        let mut coeffs = [0; FALCON_N];
        for c in 0..FALCON_N {
            coeffs[c] = sample_14bit_coeff(&mut rng, true);
        }
        coeffs
    }

    pub(in crate::falcon) fn create_packed_falcon_polynomial(
        coeffs: &[u16; FALCON_N],
    ) -> [u8; 897] {
        const NUM_BYTES: usize = CHALLENGE_LEN - 1;
        let mut out = [0u8; NUM_BYTES + 1];
        let mut bits_written = 0;
        for c in 0..FALCON_N {
            let val = coeffs[c];
            let mut acc = val;
            let mut acc_bits = COEFF_BITS as usize;

            while acc_bits > 0 {
                let byte_pos: usize = bits_written / 8;
                assert!(byte_pos < NUM_BYTES);

                let bit_offset: usize = bits_written % 8;
                let mut take = 8 - bit_offset;
                if take > acc_bits {
                    take = acc_bits;
                }
                let write_bits: u8 = (acc >> (acc_bits - take)) as u8;

                let left_shift = (8 - bit_offset) - take;
                out[byte_pos] |= write_bits << left_shift;

                acc_bits -= take;
                if acc_bits > 0 {
                    acc &= (1u16 << acc_bits) - 1;
                } else {
                    acc = 0;
                }
                bits_written += take;
            }
        }
        assert_eq!(bits_written, FALCON_N * 14);
        assert_eq!(out[896], 0);
        out
    }

    pub(in crate::falcon) fn sample_14bit_coeff(rng: &mut impl rand::Rng, valid: bool) -> u16 {
        loop {
            let val = rng.random::<u16>() & 0x3FFF;
            let ok = if valid {
                val < FALCON_Q
            } else {
                val >= FALCON_Q
            };
            if ok {
                return val;
            }
        }
    }
}

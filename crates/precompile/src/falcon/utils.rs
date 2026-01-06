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

    // Overwrite the 14-bit big-endian coefficient at `coeff_index` in-place.
    pub(in crate::falcon) fn set_coeff_14bit_be(
        buf: &mut [u8; CHALLENGE_LEN],
        coeff_index: usize,
        val: u16,
    ) {
        debug_assert!(coeff_index < FALCON_N);
        debug_assert!((val as u32) < (1u32 << COEFF_BITS));

        let bit_pos = coeff_index * (COEFF_BITS as usize);
        for j in 0..(COEFF_BITS as usize) {
            let bit = ((val >> ((COEFF_BITS as usize - 1) - j)) & 1) as u8;
            let global = bit_pos + j;
            let byte_index = global / 8;
            let bit_in_byte = global % 8;
            let mask = 1u8 << (7 - bit_in_byte);

            if bit == 1 {
                buf[byte_index] |= mask;
            } else {
                buf[byte_index] &= !mask;
            }
        }
    }
}

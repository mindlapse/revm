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

#[inline]
pub(super) fn read_u16_be(r: &mut impl sha3::digest::XofReader) -> u16 {
    let mut buf = [0u8; 2];
    r.read(&mut buf);
    u16::from_be_bytes(buf)
}

#[cfg(test)]
pub(in crate::falcon) mod test {
    use crate::falcon::{CHALLENGE_LEN, COEFF_BITS, FALCON_N, FALCON_Q};

    use super::*;

    fn dummy_output(gas: u64, data: &[u8]) -> PrecompileOutput {
        PrecompileOutput::new(gas, Bytes::copy_from_slice(data))
    }

    /// Push one bit into a growing MSB-first bit vector.
    pub(in crate::falcon) fn push_bit(bits: &mut Vec<u8>, b: bool) {
        bits.push(if b { 1 } else { 0 });
    }

    /// Push `nbits` from `value`, MSB-first.
    pub(in crate::falcon) fn push_bits_msb(bits: &mut Vec<u8>, value: u32, nbits: usize) {
        for i in (0..nbits).rev() {
            push_bit(bits, ((value >> i) & 1) != 0);
        }
    }

    /// Unary encode k as 0^k 1
    pub(in crate::falcon) fn push_unary(bits: &mut Vec<u8>, k: usize) {
        for _ in 0..k {
            push_bit(bits, false);
        }
        push_bit(bits, true);
    }

    /// Encode one coefficient per Falcon compression rules:
    /// sign bit, then 7 low bits of abs, then unary of (abs >> 7).
    pub(in crate::falcon) fn push_coeff(bits: &mut Vec<u8>, coeff: i32) {
        let neg = coeff < 0;
        let abs = coeff.unsigned_abs(); // u32

        // sign
        push_bit(bits, neg);

        // low 7 bits of abs
        let low7 = (abs & 0x7F) as u32;
        push_bits_msb(bits, low7, 7);

        // unary tail for high bits
        let k = (abs >> 7) as usize;
        push_unary(bits, k);
    }

    /// Convert MSB-first bit vector into a zero-padded `[u8; N]`.
    pub(in crate::falcon) fn bits_to_buf<const N: usize>(bits: &[u8]) -> [u8; N] {
        let mut buf = [0u8; N];
        let total_bits = N * 8;

        // Write as many bits as fit; any remaining are left as 0 (padding).
        let n = bits.len().min(total_bits);
        for i in 0..n {
            let byte_pos = i >> 3;
            let bit_in_byte = i & 7; // 0 = MSB
            if bits[i] != 0 {
                buf[byte_pos] |= 1u8 << (7 - bit_in_byte);
            }
        }
        buf
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

    /// Overwrite the 14-bit big-endian coefficient at `coeff_index` in-place.
    /// - buf is 897 bytes
    /// - buf[0] is the canonical left-pad byte (must remain 0)
    /// - coefficient i is stored at bit offset 14*i from the *LSB* of the integer,
    ///   and the integer is encoded as big-endian bytes in buf[1..].
    pub(in crate::falcon) fn set_coeff_14bit_packed(
        buf: &mut [u8; CHALLENGE_LEN],
        coeff_index: usize,
        val: u16,
    ) {
        debug_assert!(coeff_index < FALCON_N);
        debug_assert!((val as u32) < (1u32 << COEFF_BITS));
        debug_assert!(buf[0] == 0); // keep tests honest

        const COEFF_BITS_USIZE: usize = COEFF_BITS as usize;
        const DATA_BYTES: usize = CHALLENGE_LEN - 1; // 896

        // Bit position from the LSB of the integer.
        let bit_pos = coeff_index * COEFF_BITS_USIZE;

        // We will write the 14 bits of `val` into the data region buf[1..],
        // mapping integer bit 0 to the LSB of the last data byte.
        for j in 0..COEFF_BITS_USIZE {
            // Take bit j of val (LSB-first) because bit_pos counts from the LSB of the integer.
            let bit = ((val >> j) & 1) as u8;

            let global = bit_pos + j; // bit index from integer LSB
            let byte_from_lsb = global / 8; // which byte (counting from LSB end)
            let bit_in_byte = global % 8; // which bit within that byte (0 = LSB)

            // Map to big-endian byte array index inside buf[1..]:
            // LSB byte of the integer is the *last* byte of buf[1..].
            let byte_index_in_data = (DATA_BYTES - 1) - byte_from_lsb;

            // buf[1..] starts at offset 1 in the full buffer.
            let byte_index = 1 + byte_index_in_data;

            // Within a byte, big-endian convention means:
            // bit_in_byte=0 (LSB of integer) corresponds to bit 0 (LSB) of the byte,
            // which is mask 1<<0.
            let mask = 1u8 << bit_in_byte;

            if bit == 1 {
                buf[byte_index] |= mask;
            } else {
                buf[byte_index] &= !mask;
            }
        }
    }
}

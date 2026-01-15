use crate::falcon::{FALCON_N, S2_COMPRESSED_LEN};

pub(super) struct SigReader<'a> {
    bit_pos: usize,
    bytes: &'a [u8; S2_COMPRESSED_LEN],
    total_bits: usize,
}

impl<'a> SigReader<'a> {
    pub(super) fn new(s2_compressed: &'a [u8; S2_COMPRESSED_LEN]) -> Self {
        SigReader {
            bit_pos: 0,
            bytes: s2_compressed,
            total_bits: S2_COMPRESSED_LEN * 8,
        }
    }

    #[inline]
    fn remaining_bits(&self) -> usize {
        self.total_bits - self.bit_pos
    }

    #[inline]
    // true -> sign is negative
    // reads the bit at the current bit_pos and returns true iff the bit is 1 (the sign is negative).
    // increments self.bit_pos by 1.
    fn read_sign(&mut self) -> Option<bool> {
        if self.bit_pos >= self.total_bits {
            return None;
        }
        let byte_pos = self.bit_pos >> 3;
        let bit_in_byte = self.bit_pos & 7;
        let byte: u8 = self.bytes[byte_pos];
        let sign = (byte >> (7 - bit_in_byte)) & 1 != 0;
        self.bit_pos += 1;
        Some(sign)
    }

    #[inline]
    fn read_low_bits(&mut self) -> Option<u8> {
        let byte_pos = self.bit_pos >> 3;
        if byte_pos >= self.bytes.len() || self.remaining_bits() < 7 {
            return None;
        }

        let byte: u8 = self.bytes[byte_pos];
        let r = self.bit_pos & 7;
        let low_bits: u8;

        if r == 0 {
            low_bits = byte >> 1;
        } else if r == 1 {
            low_bits = byte & 0x7F;
        } else {
            if byte_pos + 1 >= self.bytes.len() {
                return None;
            }

            let buf: u16 = (byte as u16) << 8 | self.bytes[byte_pos + 1] as u16;
            low_bits = (buf >> (9 - r)) as u8 & 0x7F;
        }
        self.bit_pos += 7;
        Some(low_bits)
    }

    #[inline]
    fn read_unary_tail(&mut self) -> Option<u32> {
        let mut k: u32 = 0;

        loop {
            if self.bit_pos >= self.total_bits {
                return None; // ran out of bits before seeing the terminating '1'
            }

            let byte_pos = self.bit_pos >> 3;
            let bit_in_byte = self.bit_pos & 7;
            let byte = self.bytes[byte_pos];
            let bit_is_one = ((byte >> (7 - bit_in_byte)) & 1) != 0;

            self.bit_pos += 1;

            if bit_is_one {
                return Some(k); // terminator found
            } else {
                k += 1; // another leading zero
            }
        }
    }

    #[inline]
    fn read_coefficient(&mut self) -> Option<i32> {
        let neg = self.read_sign()?;
        let low = self.read_low_bits()?; // consumes 7 bits
        let k = self.read_unary_tail()?; // consumes >= 1 bit

        // Reconstruct magnitude: abs = (k << 7) | low
        let abs_u32 = (k << 7) | (low as u32);

        // Defensive: if absurdly large (should be impossible given buffer size), reject.
        if abs_u32 > i32::MAX as u32 {
            return None;
        }

        // Canonical rule: reject "negative zero"
        if neg && abs_u32 == 0 {
            return None;
        }

        let abs_i32 = abs_u32 as i32;
        Some(if neg { -abs_i32 } else { abs_i32 })
    }

    #[inline]
    fn padding_is_all_zero(&self) -> bool {
        let mut byte_pos = self.bit_pos >> 3;
        let r = self.bit_pos & 7;

        // 1) Check the tail of the current byte (if we're mid-byte)
        if r != 0 {
            if byte_pos >= self.bytes.len() {
                return false;
            }
            let b = self.bytes[byte_pos];

            // Bits remaining in this byte are the low (8 - r) bits.
            let mask: u8 = (1u8 << (8 - r)) - 1;
            if (b & mask) != 0 {
                return false;
            }

            byte_pos += 1;
        }

        // 2) All remaining full bytes must be 0
        self.bytes[byte_pos..].iter().all(|&b| b == 0)
    }

    /// Read exactly FALCON_N coefficients and then enforce that any remaining bits are 0 padding.
    pub(super) fn read_coefficients(&mut self) -> Option<[i32; FALCON_N]> {
        let mut out = [0i32; FALCON_N];

        for i in 0..FALCON_N {
            out[i] = self.read_coefficient()?;
        }

        if !self.padding_is_all_zero() {
            return None;
        }
        self.bit_pos = self.total_bits;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::falcon::utils::test::{
        bits_to_buf, push_bit, push_bits_msb, push_coeff, push_unary,
    };

    use super::*;

    /// Helper: read the sign bit (1) then read_low_bits (7) to get the low-bits field.
    fn read_sign_then_low(sr: &mut SigReader<'_>) -> Option<u8> {
        sr.read_sign()?; // advance by 1 bit
        sr.read_low_bits() // read next 7 bits
    }

    // Helper: encode a single coefficient at the start of the buffer.
    //
    // Format:
    //   first byte = (sign_bit << 7) | low7
    //   then unary tail bits begin at bit_pos = 8: k zeros then a 1
    fn buf_for_coeff(sign_neg: bool, low7: u8, k: usize) -> [u8; S2_COMPRESSED_LEN] {
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = (if sign_neg { 0x80 } else { 0x00 }) | (low7 & 0x7F);

        // Place the unary terminator at bit index (8 + k)
        // (k zeros precede it).
        let term_bit = 8 + k;
        let byte_pos = term_bit >> 3;
        let bit_in_byte = term_bit & 7; // 0 = MSB
        buf[byte_pos] |= 1u8 << (7 - bit_in_byte);

        buf
    }

    #[test]
    fn test_read_sign_reads_msb_first_within_byte() {
        // 0b1010_0001: bits read should be 1,0,1,0,0,0,0,1 (MSB->LSB)
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1010_0001;

        let mut sr = SigReader::new(&buf);

        let expected = [true, false, true, false, false, false, false, true];
        for &e in &expected {
            assert_eq!(sr.read_sign(), Some(e));
        }
    }

    #[test]
    fn test_read_sign_advances_across_byte_boundary() {
        // First byte all zeros, second byte MSB is 1.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0000_0000;
        buf[1] = 0b1000_0000;

        let mut sr = SigReader::new(&buf);

        // Consume 8 bits from first byte (all false)
        for _ in 0..8 {
            assert_eq!(sr.read_sign(), Some(false));
        }

        // Next read is MSB of second byte => true
        assert_eq!(sr.read_sign(), Some(true));
    }

    #[test]
    fn test_read_sign_correct_for_each_bit_position_in_byte() {
        // For each i, set a one-hot bit at MSB-first index i and confirm
        // the i-th read is true and the others are false.
        for i in 0..8 {
            let mut buf = [0u8; S2_COMPRESSED_LEN];
            buf[0] = 1u8 << (7 - i); // one-hot at position i (MSB-first)

            let mut sr = SigReader::new(&buf);

            for j in 0..8 {
                let got = sr.read_sign().unwrap();
                assert_eq!(got, i == j, "i={i}, j={j}, byte={:08b}", buf[0]);
            }
        }
    }

    #[test]
    fn test_read_sign_returns_none_at_end_of_stream() {
        // All-zero buffer is fine; we just want to hit EOF precisely.
        let buf = [0u8; S2_COMPRESSED_LEN];
        let mut sr = SigReader::new(&buf);

        // Consume all bits in the entire buffer.
        for _ in 0..(S2_COMPRESSED_LEN * 8) {
            assert!(sr.read_sign().is_some());
        }

        // Now it must be EOF and remain EOF.
        assert_eq!(sr.read_sign(), None);
        assert_eq!(sr.read_sign(), None);
    }

    #[test]
    fn test_read_low_bits_r_eq_1_returns_lower_7_bits_of_byte() {
        // After reading sign once, bit_pos % 8 == 1, so low bits should be byte & 0x7F.
        // Choose a byte where this is obvious.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1011_0010; // lower 7 bits = 0b011_0010

        let mut sr = SigReader::new(&buf);
        let low = read_sign_then_low(&mut sr).unwrap();
        assert_eq!(low, 0b0110_010);
    }

    #[test]
    fn test_read_low_bits_r_eq_0_returns_bits_6_to_0_when_aligned() {
        // Force r==0 by starting at bit_pos=0 and calling read_low_bits directly.
        // In that case, next 7 bits are bits 6..0 (MSB-first), which equals byte >> 1.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1101_0111; // byte >> 1 = 0b0110_1011

        let mut sr = SigReader::new(&buf);
        let low = sr.read_low_bits().unwrap();
        assert_eq!(low, 0b0110_1011);
    }

    #[test]
    fn test_read_low_bits_crosses_byte_boundary_r_eq_2() {
        // We want r==2 when calling read_low_bits, so consume 2 bits first.
        // Then the next 7 bits are: current_byte bits (5..0) + next_byte bit7
        //
        // Choose:
        //   byte0 = 0b0010_1011
        //   byte1 = 0b1000_0000
        // At r==2, the 7-bit window is: byte0[5..0] = 101011 and byte1[7] = 1
        // => 1010111
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0010_1011;
        buf[1] = 0b1000_0000;

        let mut sr = SigReader::new(&buf);
        assert_eq!(sr.read_sign(), Some(false)); // consume bit0
        assert_eq!(sr.read_sign(), Some(false)); // consume bit1 -> r==2

        let low = sr.read_low_bits().unwrap();
        assert_eq!(low, 0b1010_111);
    }

    #[test]
    fn test_read_low_bits_crosses_byte_boundary_r_eq_7() {
        // At r==7, read_low_bits consumes: current_byte bit0 (LSB) + next_byte bits 7..2.
        //
        // Make it easy:
        //   byte0 LSB = 1
        //   byte1 top 6 bits = 010101 (bits 7..2)
        // Result should be 1 010101 = 0b1010101
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0000_0001; // LSB=1
        buf[1] = 0b0101_0100; // bits 7..2 are 010101

        let mut sr = SigReader::new(&buf);

        // Consume 7 bits to make r==7
        for _ in 0..7 {
            assert!(sr.read_sign().is_some());
        }

        let low = sr.read_low_bits().unwrap();
        assert_eq!(low, 0b1010_101);
    }

    #[test]
    fn test_read_low_bits_returns_none_if_not_enough_bits_remain() {
        // Create a reader, then consume all but 6 bits. read_low_bits needs 7 bits.
        let buf = [0u8; S2_COMPRESSED_LEN];
        let mut sr = SigReader::new(&buf);

        // Leave exactly 6 bits remaining
        let consume = S2_COMPRESSED_LEN * 8 - 6;
        for _ in 0..consume {
            assert!(sr.read_sign().is_some());
        }

        assert_eq!(sr.read_low_bits(), None);
    }

    #[test]
    fn test_read_low_bits_consumes_exactly_seven_bits() {
        // After read_sign (1 bit), read_low_bits should advance cursor by 7 more bits.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0xFF;

        let mut sr = SigReader::new(&buf);
        assert_eq!(sr.bit_pos, 0);

        sr.read_sign().unwrap();
        assert_eq!(sr.bit_pos, 1);

        sr.read_low_bits().unwrap();
        assert_eq!(sr.bit_pos, 8); // 1 + 7
    }

    #[test]
    fn test_read_unary_tail_k_zero_when_next_bit_is_one() {
        // Next bit is 1 => k = 0.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1000_0000; // MSB is 1

        let mut sr = SigReader::new(&buf);
        let k = sr.read_unary_tail().unwrap();
        assert_eq!(k, 0);
        assert_eq!(sr.bit_pos, 1); // consumed exactly one bit (the terminator)
    }

    #[test]
    fn test_read_unary_tail_counts_zeros_then_one_within_same_byte() {
        // Bit pattern (MSB-first): 0 0 0 1 ...
        // => k = 3, consumes 4 bits.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0001_0000;

        let mut sr = SigReader::new(&buf);
        let k = sr.read_unary_tail().unwrap();
        assert_eq!(k, 3);
        assert_eq!(sr.bit_pos, 4);
    }

    #[test]
    fn test_read_unary_tail_crosses_byte_boundary() {
        // Arrange for a run of zeros that ends in the next byte.
        //
        // First byte: all zeros (8 zeros)
        // Second byte: MSB is 1 => terminator immediately
        // Starting at bit_pos=0, we should see 8 zeros then 1 => k = 8, consumed 9 bits.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0000_0000;
        buf[1] = 0b1000_0000;

        let mut sr = SigReader::new(&buf);
        let k = sr.read_unary_tail().unwrap();
        assert_eq!(k, 8);
        assert_eq!(sr.bit_pos, 9);
    }

    #[test]
    fn test_read_unary_tail_respects_nonzero_start_offset() {
        // Start at a non-byte-aligned bit_pos and ensure counting is correct.
        // We'll consume 3 bits, then the stream is: 0 0 1 ...
        //
        // Byte 0: 1110_0100
        // Bits (MSB-first): 1 1 1 0 0 1 0 0
        // After consuming 3 bits: position at bit 3 => next bits: 0 0 1 ...
        // So k should be 2 and it should consume 3 bits.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1110_0100;

        let mut sr = SigReader::new(&buf);
        for _ in 0..3 {
            sr.read_sign().unwrap(); // consume 3 bits
        }
        let start = sr.bit_pos;

        let k = sr.read_unary_tail().unwrap();
        assert_eq!(k, 2);
        assert_eq!(sr.bit_pos, start + 3); // 2 zeros + 1 terminator
    }

    #[test]
    fn test_read_unary_tail_returns_none_if_terminator_never_found() {
        // All bits are zero => no terminating 1 => must return None.
        let buf = [0u8; S2_COMPRESSED_LEN];
        let mut sr = SigReader::new(&buf);

        assert_eq!(sr.read_unary_tail(), None);
        // bit_pos will have advanced to total_bits during the scan
        assert_eq!(sr.bit_pos, S2_COMPRESSED_LEN * 8);
    }

    #[test]
    fn test_read_unary_tail_consumes_exactly_k_plus_one_bits() {
        // Pattern: 0 0 0 0 1 => k = 4, consumes 5 bits
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0000_1000;

        let mut sr = SigReader::new(&buf);
        let start = sr.bit_pos;
        let k = sr.read_unary_tail().unwrap();
        assert_eq!(k, 4);
        assert_eq!(sr.bit_pos, start + (k as usize) + 1);
    }

    #[test]
    fn test_read_unary_tail_for_large_k() {
        // Start at a non-byte-aligned bit_pos and ensure counting is correct.
        // We'll consume 9 bits, then 400 zeros followed by a 1.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[51] = 0b0100_0000;

        let mut sr = SigReader::new(&buf);
        for _ in 0..9 {
            sr.read_sign().unwrap(); // consume 9 bits
        }
        let start = sr.bit_pos;

        let k = sr.read_unary_tail().unwrap();
        assert_eq!(k, 400);
        assert_eq!(sr.bit_pos, start + 401);
    }

    #[test]
    fn test_read_coefficient_k0_positive_small_value() {
        // sign=0, low=5, k=0 => abs = 5
        let buf = buf_for_coeff(false, 5, 0);
        let mut sr = SigReader::new(&buf);

        let v = sr.read_coefficient().unwrap();
        assert_eq!(v, 5);
        assert_eq!(sr.bit_pos, 9); // 1 + 7 + (0 + 1)
    }

    #[test]
    fn test_read_coefficient_k0_negative_small_value() {
        // sign=1, low=5, k=0 => -5
        let buf = buf_for_coeff(true, 5, 0);
        let mut sr = SigReader::new(&buf);

        let v = sr.read_coefficient().unwrap();
        assert_eq!(v, -5);
        assert_eq!(sr.bit_pos, 9);
    }

    #[test]
    fn test_read_coefficient_rejects_negative_zero() {
        // sign=1, low=0, k=0 => negative zero is forbidden
        let buf = buf_for_coeff(true, 0, 0);
        let mut sr = SigReader::new(&buf);

        assert_eq!(sr.read_coefficient(), None);
    }

    #[test]
    fn test_read_coefficient_with_nonzero_k_and_low_bits() {
        // Choose k=2, low=0x55 => abs = 2*128 + 85 = 341
        let buf = buf_for_coeff(false, 85, 2);
        let mut sr = SigReader::new(&buf);

        let v = sr.read_coefficient().unwrap();
        assert_eq!(v, 341);
        assert_eq!(sr.bit_pos, 9 + 2); // 1 + 7 + (k + 1)
    }

    #[test]
    fn test_read_coefficient_unary_tail_crosses_byte_boundary() {
        // Make k large enough that the terminator lands in a later byte.
        // With k=10: terminator at bit 8+10=18 => byte 2, bit_in_byte=2 (MSB-first)
        // abs = 10*128 + low
        let low = 18;
        let k = 10;
        let buf = buf_for_coeff(false, low, k);
        let mut sr = SigReader::new(&buf);

        let v = sr.read_coefficient().unwrap();
        assert_eq!(v, (k as i32) * 128 + (low as i32));
        assert_eq!(sr.bit_pos, 9 + k);
    }

    #[test]
    fn test_read_coefficient_returns_none_if_missing_unary_terminator() {
        // Build a buffer that contains sign+low bits but no unary '1' terminator.
        // Easiest: all zeros, sign=0, low=0, and leave the rest all zeros.
        // read_unary_tail should hit EOF and return None, so coefficient returns None.
        let buf = [0u8; S2_COMPRESSED_LEN];
        let mut sr = SigReader::new(&buf);

        assert_eq!(sr.read_coefficient(), None);
    }

    #[test]
    fn test_read_coefficient_handles_large_k_stress() {
        // Similar spirit to your unary stress test, but full coefficient:
        // sign=0, low=0, k=400 => abs = 400*128 = 51200
        let buf = buf_for_coeff(false, 0, 400);
        let mut sr = SigReader::new(&buf);

        let v = sr.read_coefficient().unwrap();
        assert_eq!(v, 400 * 128);
        assert_eq!(sr.bit_pos, 9 + 400);
    }

    #[test]
    fn test_padding_is_all_zero_when_byte_aligned_and_rest_all_zero() {
        let buf = [0u8; S2_COMPRESSED_LEN];
        let mut sr = SigReader::new(&buf);

        sr.bit_pos = 16; // byte-aligned
        assert!(sr.padding_is_all_zero());
    }

    #[test]
    fn test_padding_is_all_zero_fails_when_byte_aligned_and_any_remaining_byte_nonzero() {
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[10] = 0x01;

        let mut sr = SigReader::new(&buf);

        // bit_pos points to start of byte 10
        sr.bit_pos = 9 * 8;
        assert!(!sr.padding_is_all_zero());

        // bit_pos points after byte 10, so the nonzero is "in the past" and shouldn't matter
        sr.bit_pos = 11 * 8;
        assert!(sr.padding_is_all_zero());
    }

    #[test]
    fn test_padding_is_all_zero_ignores_consumed_bits_in_current_byte() {
        // We'll set some high bits (already consumed) to 1, and ensure it's still considered zero padding.
        // Example: bit_pos%8 = 3 means bits 7..5 are consumed, bits 4..0 must be zero.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1110_0000; // consumed bits could be 1s; remaining bits are zeros

        let mut sr = SigReader::new(&buf);
        sr.bit_pos = 3; // mid-byte
        assert!(sr.padding_is_all_zero());
    }

    #[test]
    fn test_padding_is_all_zero_fails_if_any_remaining_bit_in_current_byte_is_one() {
        // bit_pos%8 = 3 => remaining bits are low 5 bits.
        // Set one of those low bits to 1.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b0000_0001; // LSB is in the remaining region for r=3

        let mut sr = SigReader::new(&buf);
        sr.bit_pos = 3;
        assert!(!sr.padding_is_all_zero());
    }

    #[test]
    fn test_padding_is_all_zero_fails_if_any_full_remaining_byte_is_nonzero_after_partial_byte() {
        // Put reader mid-byte, ensure partial byte is clean, but a later byte is nonzero.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        buf[0] = 0b1110_0000; // for r=3, remaining bits are zero
        buf[2] = 0x80; // later remaining byte is nonzero

        let mut sr = SigReader::new(&buf);
        sr.bit_pos = 3; // mid-byte in byte 0
        assert!(!sr.padding_is_all_zero());
    }

    #[test]
    fn test_padding_is_all_zero_true_when_at_end_of_stream() {
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        // even if last byte is nonzero, if we're at total_bits, padding check should be vacuously true
        buf[S2_COMPRESSED_LEN - 1] = 0xFF;

        let mut sr = SigReader::new(&buf);
        sr.bit_pos = S2_COMPRESSED_LEN * 8;
        assert!(sr.padding_is_all_zero());
    }

    #[test]
    fn test_padding_is_all_zero_handles_near_end_partial_byte_masking() {
        // Place sr just before the end: last byte exists, we’re mid-byte in the last byte.
        // r=6 => remaining bits are low 2 bits.
        let mut buf = [0u8; S2_COMPRESSED_LEN];
        let last = S2_COMPRESSED_LEN - 1;

        // Set consumed bits (top 6 bits) to 1, remaining low 2 bits to 0: should pass.
        buf[last] = 0b1111_1100;

        let mut sr = SigReader::new(&buf);
        sr.bit_pos = last * 8 + 6;
        assert!(sr.padding_is_all_zero());

        // Now set a remaining bit (one of low 2 bits) to 1: should fail.
        buf[last] = 0b1111_1101; // low bits = 01
        let mut sr2 = SigReader::new(&buf);
        sr2.bit_pos = last * 8 + 6;
        assert!(!sr2.padding_is_all_zero());
    }

    #[test]
    fn test_read_coefficients_all_zero_coeffs_succeeds_and_padding_zero() {
        // Coeff 0 encodes as: sign=0, low7=0, unary k=0 => "1" terminator
        // Total per coeff: 9 bits. For 512 coeffs: 4608 bits, leaving padding bits.
        let mut bits = Vec::new();
        for _ in 0..FALCON_N {
            push_coeff(&mut bits, 0);
        }

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);

        let out = sr.read_coefficients().expect("should decode");
        assert!(out.iter().all(|&x| x == 0));
        assert_eq!(sr.bit_pos, S2_COMPRESSED_LEN * 8);
    }

    #[test]
    fn test_read_coefficients_rejects_negative_zero_encoding() {
        // Build a stream where the first coefficient is "negative zero":
        // sign=1, low7=0, unary terminator immediately (k=0).
        // Then follow with valid zeros for remaining coeffs.
        let mut bits = Vec::new();

        // negative zero: sign=1, low7=0, unary=1
        push_bit(&mut bits, true);
        push_bits_msb(&mut bits, 0, 7);
        push_unary(&mut bits, 0);

        for _ in 1..FALCON_N {
            push_coeff(&mut bits, 0);
        }

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);

        assert!(sr.read_coefficients().is_none());
    }

    #[test]
    fn test_read_coefficients_fails_if_padding_contains_one_bit() {
        // Start from an all-zero-coeffs stream (which leaves padding),
        // then flip one padding bit to 1 (after all coefficients), which must be rejected.
        let mut bits = Vec::new();
        for _ in 0..FALCON_N {
            push_coeff(&mut bits, 0);
        }

        // Convert to buffer, then flip a known padding bit.
        // After 512 zeros, we've consumed 512*9 = 4608 bits, so bit 4608 is the first padding bit.
        let mut buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);

        let pad_bit = 4608;
        let byte_pos = pad_bit >> 3;
        let bit_in_byte = pad_bit & 7;
        buf[byte_pos] |= 1u8 << (7 - bit_in_byte);

        let mut sr = SigReader::new(&buf);
        assert!(sr.read_coefficients().is_none());
    }

    #[test]
    fn test_read_coefficients_mixed_values_round_trip_expected_prefix() {
        // Encode a simple repeating pattern, then decode and check exact match.
        // Keep values small so encoding stays short and leaves padding.
        let pattern: [i32; 8] = [0, 1, -1, 5, -5, 127, -127, 128];
        let mut bits = Vec::new();

        for i in 0..FALCON_N {
            push_coeff(&mut bits, pattern[i % pattern.len()]);
        }

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);
        let out = sr.read_coefficients().expect("should decode");

        for i in 0..FALCON_N {
            assert_eq!(out[i], pattern[i % pattern.len()]);
        }
    }

    #[test]
    fn test_read_coefficients_handles_large_k_coefficient_then_zeros() {
        // First coefficient has k=400, low7=0, sign=0 => abs = 51200
        // Remaining coefficients are zeros.
        let mut bits = Vec::new();
        push_coeff(&mut bits, 400 * 128);

        for _ in 1..FALCON_N {
            push_coeff(&mut bits, 0);
        }

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);

        let out = sr.read_coefficients().expect("should decode");
        assert_eq!(out[0], 400 * 128);
        assert!(out[1..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_read_coefficients_fails_if_stream_runs_out_mid_coefficient() {
        // Build a valid stream, then truncate by *clearing* the unary terminator
        // for the last coefficient so it can never find a '1' before EOF.
        let mut bits = Vec::new();
        for _ in 0..(FALCON_N - 1) {
            push_coeff(&mut bits, 0);
        }
        // Last coefficient: sign=0, low7=0, unary: 0000... but omit the final '1'
        push_bit(&mut bits, false);
        push_bits_msb(&mut bits, 0, 7);
        for _ in 0..20 {
            push_bit(&mut bits, false);
        }
        // No terminator bit.

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);
        assert!(sr.read_coefficients().is_none());
    }

    #[test]
    fn test_read_coefficients_last_terminator_lands_on_byte_boundary() {
        // Last coefficient uses k=1 (abs=128) so terminator alignment is r=0 as derived above.
        let mut bits = Vec::new();

        // First N-1 coeffs: zero
        for _ in 0..(FALCON_N - 1) {
            push_coeff(&mut bits, 0);
        }

        // Last coefficient: abs = 128 => sign=0, low7=0, k=1
        push_coeff(&mut bits, 128);

        // Sanity check alignment claim: terminator is at a byte boundary (r=0)
        let prefix = (FALCON_N - 1) * 9;
        let terminator_bit_index = prefix + 8 + 1; // +8 for sign+low, +k for zeros, then terminator
        assert_eq!(terminator_bit_index % 8, 0);

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);

        let out = sr.read_coefficients().expect("should decode");
        assert_eq!(out[FALCON_N - 1], 128);
        assert!(out[..FALCON_N - 1].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_read_coefficients_last_terminator_lands_one_bit_after_boundary() {
        // Last coefficient uses k=2 (abs=256) so terminator alignment is r=1.
        let mut bits = Vec::new();

        for _ in 0..(FALCON_N - 1) {
            push_coeff(&mut bits, 0);
        }

        // Last coefficient: abs = 256 => sign=0, low7=0, k=2
        push_coeff(&mut bits, 256);

        let prefix = (FALCON_N - 1) * 9;
        let terminator_bit_index = prefix + 8 + 2;
        assert_eq!(terminator_bit_index % 8, 1);

        let buf = bits_to_buf::<S2_COMPRESSED_LEN>(&bits);
        let mut sr = SigReader::new(&buf);

        let out = sr.read_coefficients().expect("should decode");
        assert_eq!(out[FALCON_N - 1], 256);
        assert!(out[..FALCON_N - 1].iter().all(|&x| x == 0));
    }
}

use crate::falcon::{
    error::FalconError, FalconCoreInputs, H2PInputs, PackedFalconPolynomial, CHALLENGE_LEN,
    COEFF_BITS, FALCON_N, FALCON_Q, MSG_LEN, PK_LEN, S2_COMPRESSED_LEN, SALT_LEN, SIG_LEN,
};

/// Ensures `input` is exactly `wanted` bytes long.
///
/// Length mismatches are treated as normal Falcon failures (not `OutOfGas`)
/// and are converted to empty output by the precompile entrypoint.
///
/// # Errors
/// Returns `FalconError::InvalidInputLength` if `input.len() != wanted`.
#[inline]
pub(super) fn require_len(input: &[u8], wanted: usize) -> Result<(), FalconError> {
    let len = input.len();
    if len == wanted {
        Ok(())
    } else {
        Err(FalconError::InvalidInputLength { wanted, got: len })
    }
}

/// Interprets `input` as a fixed-size byte array reference.
///
/// This helper avoids repeated slicing and `try_into()` boilerplate at call sites,
/// and provides a strongly-typed view (`&[u8; N]`) without copying.
///
/// # Errors
/// Returns `FalconError::InvalidInputLength` if `input.len() != N`.
#[inline]
pub(super) fn read_fixed<const N: usize>(input: &[u8]) -> Result<&[u8; N], FalconError> {
    input
        .try_into()
        .map_err(|_| FalconError::InvalidInputLength {
            wanted: N,
            got: input.len(),
        })
}

/// Splits an input buffer into two fixed-size segments `A` and `B`.
///
/// This is a safe, zero-copy alternative to manual `split_at` + `try_into`
/// sequences. The function requires the input length to be exactly `A + B`
/// to match the canonical precompile layout.
///
/// # Errors
/// Returns `FalconError::InvalidInputLength` if `input.len() != A + B`.
#[inline]
pub(super) fn split_fixed<const A: usize, const B: usize>(
    input: &[u8],
) -> Result<(&[u8; A], &[u8; B]), FalconError> {
    require_len(input, A + B)?;
    let (a, b) = input.split_at(A);

    let left = read_fixed::<A>(a)?;
    let right = read_fixed::<B>(b)?;

    Ok((left, right))
}

/// Splits the Hash-to-Point (H2P) precompile input into `(msg, sig)`.
///
/// Layout:
/// - `MSG_LEN` bytes: message hash
/// - `SIG_LEN` bytes: Falcon-512 compressed signature
///
/// # Errors
/// Returns `FalconError::InvalidInputLength` if the input does not match the
/// expected canonical length `MSG_LEN + SIG_LEN`.
#[inline]
pub(super) fn split_h2p_input<'a>(input: &'a [u8]) -> Result<H2PInputs<'a>, FalconError> {
    split_fixed::<MSG_LEN, SIG_LEN>(input)
}

/// Splits the Falcon core precompile input into `(sig, pk, challenge)`.
///
/// Layout:
/// - `SIG_LEN` bytes: Falcon-512 compressed signature
/// - `PK_LEN` bytes: Falcon-512 public key
/// - `CHALLENGE_LEN` bytes: packed challenge polynomial
///
/// # Errors
/// Returns `FalconError::InvalidInputLength` if the input does not match the
/// expected canonical length `SIG_LEN + PK_LEN + CHALLENGE_LEN`.
#[inline]
pub(super) fn split_falcon_core_input<'a>(
    input: &'a [u8],
) -> Result<FalconCoreInputs<'a>, FalconError> {
    require_len(input, SIG_LEN + PK_LEN + CHALLENGE_LEN)?;
    let (sig, rest) = split_fixed::<SIG_LEN, { PK_LEN + CHALLENGE_LEN }>(input)?;
    let (pk, challenge) = split_fixed::<PK_LEN, CHALLENGE_LEN>(rest)?;
    Ok((sig, pk, challenge))
}

/// Splits a Falcon-512 compressed signature into `(salt, s2_compressed)`.
///
/// The signature is canonically encoded as a 40-byte salt prefix followed by
/// the compressed `s2` tail segment.
///
/// Note: Given the typed input `&[u8; SIG_LEN]`, this split cannot fail.
/// A `Result` is returned for API uniformity with other parsing helpers.
///
/// # Errors
/// Returns `FalconError::InvalidInputLength` only if internal constants are
/// inconsistent (e.g., if `SALT_LEN + S2_COMPRESSED_LEN != SIG_LEN`).
#[inline]
pub(super) fn split_sig(
    sig: &[u8; SIG_LEN],
) -> Result<(&[u8; SALT_LEN], &[u8; S2_COMPRESSED_LEN]), FalconError> {
    let (r, s2) = sig.split_at(SALT_LEN);
    Ok((
        read_fixed::<SALT_LEN>(r)?,
        read_fixed::<S2_COMPRESSED_LEN>(s2)?,
    ))
}

/// Unpack a packed 14-bit big-endian Falcon polynomial into 512 coefficients,
/// validating each coefficient is < q. Also enforces canonical padding if present.
///
/// Layout:
/// - 512 coefficients × 14 bits = 7168 bits = 896 bytes of coefficient data.
/// - If `input_len` is 897, the final byte is padding and must be 0.
#[inline]
pub(super) fn unpack_falcon_14bit_be_polynomial<const INPUT_LEN: usize>(
    input: &[u8; INPUT_LEN],
) -> Result<[u16; FALCON_N], FalconError> {
    // You can use this helper for both public key and challenge bits without duplicating logic.

    // Determine how many bytes actually carry coefficient bits.
    // 512*14 bits = 896 bytes exactly.
    const COEFF_BYTES: usize = (FALCON_N * COEFF_BITS as usize) / 8; // 896

    // Compile-time invariants
    const _: () = assert!((FALCON_N * (COEFF_BITS as usize)) % 8 == 0);
    const _: () = assert!(COEFF_BYTES == 896);

    // Enforce 897 bits
    if INPUT_LEN != COEFF_BYTES + 1 {
        return Err(FalconError::InvalidInputLength {
            wanted: COEFF_BYTES, // or wanted: your chosen constant
            got: INPUT_LEN,
        });
    }

    // If there is a padding byte, require it to be zero for canonical encoding.
    if input[COEFF_BYTES] != 0 {
        return Err(FalconError::InvalidFieldElement);
    }

    let mut out = [0u16; FALCON_N];

    let mut acc: u32 = 0;
    let mut acc_bits: u32 = 0;
    let mut out_i: usize = 0;

    // Helper: keep only the lowest `bits` bits of `x`.
    #[inline(always)]
    fn low_mask_u32(x: u32, bits: u32) -> u32 {
        match bits {
            0 => 0,
            // In this algorithm `bits` never reaches 32, but this helps keep everyone calm.
            32 => x,
            _ => x & ((1u32 << bits) - 1),
        }
    }

    let (coeff_bytes, _pad) = input.split_at(COEFF_BYTES as usize);

    for &b in coeff_bytes {
        acc = (acc << 8) | (b as u32);
        acc_bits += 8;

        while acc_bits >= COEFF_BITS && out_i < FALCON_N {
            let shift = acc_bits - COEFF_BITS;
            let v = acc >> shift;
            acc_bits -= COEFF_BITS;

            // Keep only the remaining lower acc_bits bits.
            acc = low_mask_u32(acc, acc_bits);

            // Range check - ensure each polynomial coefficient is strictly less than Q = 12289
            if v >= (FALCON_Q as u32) {
                return Err(FalconError::InvalidFieldElement);
            }

            out[out_i] = v as u16;
            out_i += 1;
        }
    }

    // Must have produced exactly 512 coefficients.
    // If not, the input was malformed (shouldn't happen with fixed lengths, defensive only)
    if out_i != FALCON_N {
        return Err(FalconError::InternalEncoding);
    }

    // Also ensure we didn't have leftover bits that imply non-canonical encoding.
    // With 896 bytes and 14-bit chunks, acc_bits should end at 0 exactly.
    if acc_bits != 0 {
        return Err(FalconError::InternalEncoding);
    }

    Ok(out)
}

pub(super) fn pack_falcon_14bit_be_polynomial(
    coeffs: &[u16; FALCON_N],
) -> Result<Box<PackedFalconPolynomial>, FalconError> {
    const BITS_TO_WRITE: usize = FALCON_N * COEFF_BITS as usize;
    const BYTES_TO_WRITE: usize = BITS_TO_WRITE / 8;

    // Compile-time invariants (evalutated during const evaluation and not at runtime)
    const _: () = assert!(BITS_TO_WRITE % 8 == 0);
    const _: () = assert!(BYTES_TO_WRITE + 1 == CHALLENGE_LEN);

    let mut out = Box::new([0u8; BYTES_TO_WRITE + 1]);
    let out_writable = &mut out[..BYTES_TO_WRITE];
    let mut bits_written = 0;

    for c in 0..FALCON_N {
        let val = coeffs[c];
        let mut acc = val;
        let mut acc_bits = COEFF_BITS as usize;

        while acc_bits > 0 {
            let byte_pos: usize = bits_written / 8;

            let bit_offset: usize = bits_written % 8;
            let mut take = 8 - bit_offset;
            if take > acc_bits {
                take = acc_bits;
            }
            let write_bits: u8 = (acc >> (acc_bits - take)) as u8;

            let left_shift = (8 - bit_offset) - take;

            // panic-free index access
            let dst = out_writable
                .get_mut(byte_pos)
                .ok_or(FalconError::InternalEncoding)?;
            *dst |= write_bits << left_shift;

            acc_bits -= take;
            if acc_bits > 0 {
                acc &= (1u16 << acc_bits) - 1;
            } else {
                acc = 0;
            }
            bits_written += take;
        }
    }
    if bits_written != BITS_TO_WRITE || out[CHALLENGE_LEN - 1] != 0 {
        return Err(FalconError::InternalEncoding);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;
    use crate::falcon::utils::test::{create_sample_falcon_coefficients, sample_14bit_coeff};

    #[test]
    fn test_require_len_ok() {
        let input = vec![1, 2, 3, 4, 5];
        assert_eq!(require_len(&input, 5), Ok(()));
    }

    #[test]
    fn test_require_len_too_short() {
        let input = vec![1, 2, 3];
        let err = require_len(&input, 5).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, 5);
                assert_eq!(got, 3);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_require_len_too_long() {
        let input = vec![1, 2, 3, 4, 5, 6, 7];
        let err = require_len(&input, 5).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, 5);
                assert_eq!(got, 7);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_require_len_zero_length() {
        let input: Vec<u8> = vec![];
        assert_eq!(require_len(&input, 0), Ok(()));
        let err = require_len(&input, 1).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, 1);
                assert_eq!(got, 0);
            }
            _ => panic!("Unwanted error variant"),
        }
    }
    #[test]
    fn test_read_fixed_various_sizes() {
        let cases: Vec<(&[u8], usize, Result<&[u8], (usize, usize)>)> = vec![
            (&[10, 20, 30, 40], 4, Ok(&[10, 20, 30, 40])),
            (&[1, 2, 3], 4, Err((4, 3))),
            (&[1, 2, 3, 4, 5], 4, Err((4, 5))),
            (&[], 4, Err((4, 0))),
            (&[], 0, Ok(&[])),
        ];

        for (input, wanted, expected) in cases {
            let result: Result<&[u8], FalconError> = match wanted {
                0 => read_fixed::<0>(input).map(|arr| arr.as_slice()),
                4 => read_fixed::<4>(input).map(|arr| arr.as_slice()),
                _ => unreachable!(),
            };
            match (result, expected) {
                (Ok(actual), Ok(expected_slice)) => assert_eq!(actual, expected_slice),
                (
                    Err(FalconError::InvalidInputLength { wanted, got }),
                    Err((exp_wanted, exp_got)),
                ) => {
                    assert_eq!(wanted, exp_wanted);
                    assert_eq!(got, exp_got);
                }
                _ => panic!(
                    "Unexpected result for input: {:?}, wanted: {}",
                    input, wanted
                ),
            }
        }
    }

    #[test]
    fn test_split_fixed_ok() {
        let input = [1, 2, 3, 4, 5, 6];
        let (a, b) = split_fixed::<2, 4>(&input).unwrap();
        assert_eq!(a, &[1, 2]);
        assert_eq!(b, &[3, 4, 5, 6]);
    }

    #[test]
    fn test_split_fixed_too_short() {
        let input = [1, 2, 3];
        let err = split_fixed::<2, 2>(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, 4);
                assert_eq!(got, 3);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_fixed_exact_length() {
        let input = [10, 20, 30, 40];
        let (a, b) = split_fixed::<2, 2>(&input).unwrap();
        assert_eq!(a, &[10, 20]);
        assert_eq!(b, &[30, 40]);
    }

    #[test]
    fn test_split_fixed_zero_left() {
        let input = [7, 8, 9];
        let (a, b) = split_fixed::<0, 3>(&input).unwrap();
        assert_eq!(a, &[]);
        assert_eq!(b, &[7, 8, 9]);
    }

    #[test]
    fn test_split_fixed_zero_right() {
        let input = [5, 6, 7];
        let (a, b) = split_fixed::<3, 0>(&input).unwrap();
        assert_eq!(a, &[5, 6, 7]);
        assert_eq!(b, &[]);
    }

    #[test]
    fn test_split_fixed_both_zero() {
        let input: [u8; 0] = [];
        let (a, b) = split_fixed::<0, 0>(&input).unwrap();
        assert_eq!(a, &[]);
        assert_eq!(b, &[]);
    }

    #[test]
    fn test_split_h2p_input_ok() {
        // MSG_LEN + SIG_LEN bytes
        let mut input = vec![0u8; MSG_LEN + SIG_LEN];
        for i in 0..MSG_LEN {
            input[i] = 1;
        }
        for i in MSG_LEN..(MSG_LEN + SIG_LEN) {
            input[i] = 2;
        }
        let (msg, sig) = split_h2p_input(&input).unwrap();
        assert_eq!(msg, &[1u8; MSG_LEN]);
        assert_eq!(sig, &[2u8; SIG_LEN]);
    }

    #[test]
    fn test_split_h2p_input_too_short() {
        let input = vec![0u8; MSG_LEN + SIG_LEN - 1];
        let err = split_h2p_input(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, MSG_LEN + SIG_LEN);
                assert_eq!(got, MSG_LEN + SIG_LEN - 1);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_h2p_input_too_long() {
        let input = vec![0u8; MSG_LEN + SIG_LEN + 5];
        let err = split_h2p_input(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, MSG_LEN + SIG_LEN);
                assert_eq!(got, MSG_LEN + SIG_LEN + 5);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_h2p_input_zero_length() {
        let input: Vec<u8> = vec![];
        let err = split_h2p_input(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, MSG_LEN + SIG_LEN);
                assert_eq!(got, 0);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_falcon_core_input_ok() {
        // SIG_LEN + PK_LEN + CHALLENGE_LEN bytes
        let mut input = vec![0u8; SIG_LEN + PK_LEN + CHALLENGE_LEN];
        for i in 0..SIG_LEN {
            input[i] = 1;
        }
        for i in SIG_LEN..(SIG_LEN + PK_LEN) {
            input[i] = 2;
        }
        for i in (SIG_LEN + PK_LEN)..(SIG_LEN + PK_LEN + CHALLENGE_LEN) {
            input[i] = 3;
        }
        let (sig, pk, challenge) = split_falcon_core_input(&input).unwrap();
        assert!(sig.iter().all(|&b| b == 1));
        assert!(pk.iter().all(|&b| b == 2));
        assert!(challenge.iter().all(|&b| b == 3));
    }

    #[test]
    fn test_split_falcon_core_input_too_short() {
        let input = vec![0u8; SIG_LEN + PK_LEN + CHALLENGE_LEN - 1];
        let err = split_falcon_core_input(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, SIG_LEN + PK_LEN + CHALLENGE_LEN);
                assert_eq!(got, SIG_LEN + PK_LEN + CHALLENGE_LEN - 1);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_falcon_core_input_too_long() {
        let input = vec![0u8; SIG_LEN + PK_LEN + CHALLENGE_LEN + 5];
        let err = split_falcon_core_input(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, SIG_LEN + PK_LEN + CHALLENGE_LEN);
                assert_eq!(got, SIG_LEN + PK_LEN + CHALLENGE_LEN + 5);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_falcon_core_input_zero_length() {
        let input: Vec<u8> = vec![];
        let err = split_falcon_core_input(&input).unwrap_err();
        match err {
            FalconError::InvalidInputLength { wanted, got } => {
                assert_eq!(wanted, SIG_LEN + PK_LEN + CHALLENGE_LEN);
                assert_eq!(got, 0);
            }
            _ => panic!("Unwanted error variant"),
        }
    }

    #[test]
    fn test_split_sig_ok() {
        let mut sig = [0u8; SIG_LEN];
        for i in 0..SALT_LEN {
            sig[i] = 1;
        }
        for i in SALT_LEN..(SALT_LEN + S2_COMPRESSED_LEN) {
            sig[i] = 2;
        }
        let (salt, s2) = split_sig(&sig).unwrap();
        assert_eq!(salt, &[1u8; SALT_LEN]);
        assert_eq!(s2, &[2u8; S2_COMPRESSED_LEN]);
    }

    #[test]
    fn test_split_sig_salt_all_zero() {
        let mut sig = [0u8; SIG_LEN];
        for i in SALT_LEN..(SALT_LEN + S2_COMPRESSED_LEN) {
            sig[i] = 5;
        }
        let (salt, s2) = split_sig(&sig).unwrap();
        assert_eq!(salt, &[0u8; SALT_LEN]);
        assert_eq!(s2, &[5u8; S2_COMPRESSED_LEN]);
    }

    #[test]
    fn test_split_sig_s2_all_zero() {
        let mut sig = [0u8; SIG_LEN];
        for i in 0..SALT_LEN {
            sig[i] = 9;
        }
        let (salt, s2) = split_sig(&sig).unwrap();
        assert_eq!(salt, &[9u8; SALT_LEN]);
        assert_eq!(s2, &[0u8; S2_COMPRESSED_LEN]);
    }

    #[test]
    fn test_unpack_falcon_14bit_be_polynomial_all_zero_input_yields_all_zero_output() {
        // 512 coefficients × 14 bits = 896 bytes, plus 1 padding byte = 897 total.
        let input = [0u8; CHALLENGE_LEN];

        let poly = unpack_falcon_14bit_be_polynomial::<CHALLENGE_LEN>(&input).unwrap();

        assert_eq!(poly, [0u16; FALCON_N]);
    }

    #[test]
    fn test_unpack_falcon_14bit_be_polynomial_invalid_if_coeff_ge_12289() {
        let mut rng = rand::rng();
        // Repeat the test for every coefficient position, setting
        // exactly one of them to be FALCON_Q or above.
        for i in 0..FALCON_N {
            let mut coeffs = create_sample_falcon_coefficients();
            coeffs[i] = sample_14bit_coeff(&mut rng, false);

            let polynomial = pack_falcon_14bit_be_polynomial(&coeffs).unwrap();

            let result = unpack_falcon_14bit_be_polynomial::<CHALLENGE_LEN>(&polynomial);
            assert!(matches!(result, Err(FalconError::InvalidFieldElement)));
        }
    }

    #[test]
    fn test_unpack_falcon_14bit_be_polynomial_valid_if_coeff_lt_12289() {
        // try 512 random valid sets of coefficients
        for _ in 0..512 {
            let coeffs = create_sample_falcon_coefficients();
            let polynomial = pack_falcon_14bit_be_polynomial(&coeffs).unwrap();

            let result = unpack_falcon_14bit_be_polynomial::<CHALLENGE_LEN>(&polynomial);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), coeffs);
        }
    }

    #[test]
    fn test_unpack_falcon_14bit_be_polynomial_invalid_if_last_byte_nonzero() {
        let mut rng = rand::rng();
        for _ in 0..128 {
            let coeffs = create_sample_falcon_coefficients();
            let mut polynomial = pack_falcon_14bit_be_polynomial(&coeffs).unwrap();
            polynomial[896] = rng.random_range(1..255);
            let result = unpack_falcon_14bit_be_polynomial::<CHALLENGE_LEN>(&polynomial);
            assert!(matches!(result, Err(FalconError::InvalidFieldElement)));
        }
    }
}

use crate::falcon::{
    error::FalconError, FalconCoreInputs, H2PInputs, CHALLENGE_LEN, MSG_LEN, PK_LEN,
    S2_COMPRESSED_LEN, SALT_LEN, SIG_LEN,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

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

use crate::{
    PrecompileError, PrecompileOutput, PrecompileResult, crypto, falcon::{
        H2P_GAS, H2PInputs, MSG_LEN, SALT_LEN, encoding::{pack_falcon_14bit_be_polynomial, split_sig}, error::FalconError, utils::map_falcon_result
    }
};
use sha3 as _;

pub fn h2p_shake256(input: &[u8], gas_limit: u64) -> PrecompileResult {
    if gas_limit < H2P_GAS {
        return Err(PrecompileError::OutOfGas);
    }
    map_falcon_result(compute_h2p(input), H2P_GAS)
}

#[inline]
pub(crate) fn shake256_reader(
    salt: &[u8; SALT_LEN],
    msg_digest: &[u8; MSG_LEN],
) -> impl sha3::digest::XofReader {
    use sha3::{
        Shake256,
        digest::{Update, ExtendableOutput},
    };

    let mut h = Shake256::default();
    h.update(salt);
    h.update(msg_digest);
    h.finalize_xof()
}

#[inline]
fn compute_h2p(input: &[u8]) -> Result<PrecompileOutput, FalconError> {
    let (msg, sig) = extract_inputs_if_valid(input)?;
    let (salt, _) = split_sig(sig)?;
    let challenge = crypto().falcon_h2p_shake256(msg, salt)?;
    let packed_challenge: Box<[u8]> = pack_falcon_14bit_be_polynomial(&challenge)?;
    Ok(PrecompileOutput::new(H2P_GAS, packed_challenge.into()))
}

#[inline]
fn extract_inputs_if_valid<'a>(input: &'a [u8]) -> Result<H2PInputs<'a>, FalconError> {
    Ok(crate::falcon::encoding::split_h2p_input(input)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha3::digest::XofReader;
    const OUT_LEN: usize = 256;

    #[test]
    fn test_extract_inputs_if_valid_with_valid_length() {
        // Assuming valid input length for H2PInputs is 698 bytes (32 + 666)
        let input = vec![0u8; 698];
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_empty_input() {
        let input = vec![];
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_too_short_length() {
        let input = vec![0u8; 10]; // Much shorter than required
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_inputs_if_valid_with_too_long_input() {
        let input = vec![0u8; 1000]; // Longer than required
        let result = extract_inputs_if_valid(&input);
        assert!(result.is_err());
    }

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

        assert!(result.is_ok());
        let out = result.unwrap();
        assert!(out.gas_used == H2P_GAS);
        assert!(out.bytes.is_empty());
    }

    #[test]
    fn test_h2p_shake256_malformed_input_returns_empty_output() {
        let input = vec![0u8; 10];
        let out = h2p_shake256(&input, 10_000).expect("should not error");
        assert_eq!(out.gas_used, 1000);
        assert!(out.bytes.is_empty());
    }

    fn read_n(mut r: impl XofReader, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        r.read(&mut out);
        out
    }

    #[test]
    fn shake256_reader_is_deterministic() {
        let salt = [22u8; SALT_LEN];
        let msg = [33u8; MSG_LEN];

        let a = read_n(shake256_reader(&salt, &msg), OUT_LEN);
        let b = read_n(shake256_reader(&salt, &msg), OUT_LEN);

        assert_eq!(a, b);
    }

    #[test]
    fn shake256_reader_changes_if_salt_changes() {
        let salt1 = [123u8; SALT_LEN];
        let mut salt2 = [213u8; SALT_LEN];
        salt2[0] ^= 0x01;

        let msg = [2u8; MSG_LEN];

        let a = read_n(shake256_reader(&salt1, &msg), 64);
        let b = read_n(shake256_reader(&salt2, &msg), 64);

        assert_ne!(a, b);
    }

    #[test]
    fn shake256_reader_changes_if_msg_changes() {
        let salt = [3u8; SALT_LEN];

        let msg1 = [101u8; MSG_LEN];
        let mut msg2 = [202u8; MSG_LEN];
        msg2[MSG_LEN - 1] ^= 0x80;

        let a = read_n(shake256_reader(&salt, &msg1), 64);
        let b = read_n(shake256_reader(&salt, &msg2), 64);

        assert_ne!(a, b);
    }

    #[test]
    fn shake256_reader_streaming_matches_single_read() {
        let salt = [4u8; SALT_LEN];
        let msg = [5u8; MSG_LEN];

        // One-shot read.
        let one_shot = read_n(shake256_reader(&salt, &msg), OUT_LEN);

        // Chunked read.
        let mut r = shake256_reader(&salt, &msg);
        let mut chunked = Vec::with_capacity(OUT_LEN);
        for chunk_size in [1usize, 2, 3, 5, 8, 13, 21, 34, 55] {
            if chunked.len() >= OUT_LEN {
                break;
            }
            let take = core::cmp::min(chunk_size, OUT_LEN - chunked.len());
            let mut buf = vec![0u8; take];
            r.read(&mut buf);
            chunked.extend_from_slice(&buf);
        }
        // Finish in one final read if needed.
        if chunked.len() < OUT_LEN {
            let mut buf = vec![0u8; OUT_LEN - chunked.len()];
            r.read(&mut buf);
            chunked.extend_from_slice(&buf);
        }

        assert_eq!(one_shot, chunked);
    }

    #[test]
    fn shake256_reader_prefix_property() {
        let salt = [6u8; SALT_LEN];
        let msg = [7u8; MSG_LEN];

        let short = read_n(shake256_reader(&salt, &msg), 64);
        let long = read_n(shake256_reader(&salt, &msg), 128);

        assert_eq!(&long[..64], &short[..]);
    }

    #[test]
    fn shake256_reader_absorb_order_matters() {
        use sha3::{Shake256, digest::{Update, ExtendableOutput}};

        let salt = [8u8; SALT_LEN];
        let msg = [9u8; MSG_LEN];

        // Our intended order: salt || msg
        let a = read_n(shake256_reader(&salt, &msg), 64);

        // Deliberately swapped order: msg || salt (should differ)
        let mut h = Shake256::default();
        h.update(&msg);
        h.update(&salt);
        let swapped = read_n(h.finalize_xof(), 64);

        assert_ne!(a, swapped);
    }
}

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

use sha3 as _;
use crate::{PrecompileError, PrecompileResult};

pub fn h2p_shake256(_input: &[u8], _gas_limit: u64) -> PrecompileResult {
    Err(PrecompileError::Fatal("not yet implemented".into()))
}
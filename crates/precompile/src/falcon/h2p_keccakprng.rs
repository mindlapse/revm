//! Falcon Hash-to-Point (H2P) implementation using a Keccak-based PRNG.
//!
//! This is an EVM-friendly alternative to the SHAKE256-based H2P,
//! intended to reduce gas costs by leveraging the existing Keccak256
//! precompile. This module is feature-gated because the Keccak-PRNG
//! construction is still subject to final specification in EIP-8052.
//!
//! Once a challenge is obtained, it can then be passed to FALCON_CORE
//! with the signature and public key for signature verification,
//! provided that the signature was generated using the same H2P method.

use crate::{PrecompileError, PrecompileResult};

pub fn h2p_keccakprng(_input: &[u8], _gas_limit: u64) -> PrecompileResult {
    Err(PrecompileError::Fatal("not yet implemented".into()))
}
//! Falcon core verification implementation.
//!
//! This module implements the Falcon-512 core verification algorithm,
//! which consumes a signature, public key, and challenge polynomial
//! and checks the Falcon norm bound. It is independent of the
//! Hash-to-Point construction used to derive the challenge.

use crate::{PrecompileError, PrecompileResult};

pub fn falcon_core(_input: &[u8], _gas_limit: u64) -> PrecompileResult {
    Err(PrecompileError::Fatal("not yet implemented".into()))
}
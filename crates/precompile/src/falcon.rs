//! Support for (post-quantum) Falcon-512 signature verification precompiles (EIP-8052)
//!
//! ## Why Falcon is split into multiple precompiles
//!
//! Falcon verification is modeled as a two-phase pipeline:
//! 1) **Hash-to-Point (H2P)** derives a packed challenge polynomial from the `(message_hash, signature)`.
//! 2) **Core verification** checks the signature against the `(public_key, challenge)` and enforces
//!    the Falcon norm bound.
//!
//! EIP-8052 defines two H2P variants:
//! - `FALCON_HASH_TO_POINT_SHAKE256`: NIST-aligned challenge generation using SHAKE256.
//! - `FALCON_HASH_TO_POINT_KECCAKPRNG`: EVM-friendly challenge generation using a Keccak-based PRNG,
//!   feature-gated here because final byte-for-byte details may still evolve.
//!
//! The core verifier is shared across both H2P variants:
//! - `FALCON_CORE`: validates encodings, performs polynomial arithmetic (including NTT operations),
//!   and returns success or failure.
//!
//! ## Address configurability
//!
//! Precompile addresses for EIP-8052 are not assumed to be finalized here. Instead, this module
//! exposes [`FalconAddresses`] and [`precompiles_with_addresses`] so downstream users (rollups,
//! devnets, or experimental forks) can register Falcon precompiles at explicit addresses without
//! changing the default precompile sets.

use crate::{Address, Precompile, PrecompileId};

pub(crate) mod falcon_core;
pub(crate) mod falcon_precompiles;
pub(crate) use falcon_precompiles as crypto_backend;
pub(crate) mod error;

pub mod h2p_shake256;

#[cfg(feature = "falcon-keccakprng")]
pub mod h2p_keccakprng;

mod encoding;
mod utils;

/// The gas cost to invoke a Falcon hash-to-point precompile
const H2P_GAS: u64 = 1000;

/// The gas cost to invoke the Falcon signature verification precompile.
const FALCON_CORE_VERIFY_GAS: u64 = 2000;

/// Length in bytes of the message input to Falcon verification.
///
/// Falcon precompiles operate on a fixed 32-byte message hash,
/// consistent with Ethereum transaction and signing conventions.
pub const MSG_LEN: usize = 32;

/// Length in bytes of a Falcon-512 compressed signature.
///
/// This includes the salt and compressed signature vector as defined by EIP-8052.
pub const SIG_LEN: usize = 666;

/// Length in bytes of 512 14-bit coefficients packed together and a padding byte,
/// used for public keys in falcon signatures, and for falcon challenge polynomials.
pub const PACKED_POLY_LEN: usize = 897;

/// Length in bytes of a Falcon-512 public key (packed).
///
/// Public keys encode a degree-512 polynomial modulo q=12289 using
/// a packed 14-bit-per-coefficient representation.
pub const PK_LEN: usize = PACKED_POLY_LEN;

/// Length in bytes of a packed Falcon challenge polynomial (packed).
///
/// The challenge polynomial consists of 512 coefficients modulo q=12289,
/// packed as 14-bit big-endian integers.
pub const CHALLENGE_LEN: usize = PACKED_POLY_LEN;

/// Length in bytes of the salt that forms the prefix of a Falcon-512 compressed signature.
pub const SALT_LEN: usize = 40;

/// Length in bytes of the non-salt tail segment of a Falcon-512 compressed signature.
pub const S2_COMPRESSED_LEN: usize = SIG_LEN - SALT_LEN;

/// Falcon modulus q used for all polynomial arithmetic.
const FALCON_Q: u16 = 12289;

/// Falcon-512 lattice dimension (polynomial degree).
pub const FALCON_N: usize = 512;

/// Length of an (unpacked) u16 array of Falcon public key coefficients.
pub const PK_LEN_UNPACKED: usize = FALCON_N;

/// Length of an (unpacked) u16 array of Falcon challenge polynomial coefficients.
pub const CHALLENGE_LEN_UNPACKED: usize = FALCON_N;

/// Bit width used to encode Falcon coefficients (14 bits per coefficient).
const COEFF_BITS: u32 = 14;

/// Compile-time enforcement that SIG_LEN = SALT_LEN + S2_COMPRESSED_LEN
const _: [(); SIG_LEN] = [(); SALT_LEN + S2_COMPRESSED_LEN];

/// The array type for the packed representation of the coefficients of a Falcon polynomial.
pub type PackedFalconPolynomial = [u8; PACKED_POLY_LEN];

/// The unpacked form of a public key, represented as
/// `PK_LEN_UNPACKED` coefficients in the range of [0, FALCON_Q).
pub type UnpackedPublicKey = [u16; PK_LEN_UNPACKED];

/// The unpacked form of a hash-to-point challenge, represented as
/// `CHALLENGE_LEN_UNPACKED` coefficients in the range of [0, FALCON_Q).
pub type UnpackedChallenge = [u16; CHALLENGE_LEN_UNPACKED];

type FalconCoreInputs<'a> = (&'a [u8; SIG_LEN], &'a [u8; PK_LEN], &'a [u8; CHALLENGE_LEN]);
type H2PInputs<'a> = (&'a [u8; MSG_LEN], &'a [u8; SIG_LEN]);

/// Addresses for Falcon-related precompiled contracts.
///
/// This struct allows callers to explicitly configure the addresses
/// at which Falcon precompiles are registered. Addresses are not
/// hard-coded because EIP-8052 final address assignments are still TBD,
/// and downstream clients or rollups may wish to experiment with
/// custom layouts.
#[derive(Debug)]
pub struct FalconAddresses {
    /// Address of the SHAKE256-based Hash-to-Point precompile.
    pub h2p_shake256: Address,

    /// Address of the Falcon core verification precompile.
    pub falcon_core: Address,

    /// Address of the Keccak-PRNG-based Hash-to-Point precompile.
    #[cfg(feature = "falcon-keccakprng")]
    pub h2p_keccakprng: Address,
}

/// Concrete Falcon precompile instances.
///
/// This struct bundles the instantiated `Precompile` objects
/// corresponding to the configured Falcon addresses. It is primarily
/// intended for opt-in registration into a `Precompiles` set by
/// downstream users.
#[derive(Debug)]
pub struct FalconPrecompiles {
    /// SHAKE256-based Hash-to-Point precompile.
    pub h2p_shake256: Precompile,

    /// Falcon core verification precompile.
    pub falcon_core: Precompile,

    /// Keccak-PRNG-based Hash-to-Point precompile.
    #[cfg(feature = "falcon-keccakprng")]
    pub h2p_keccakprng: Precompile,
}

/// Construct Falcon precompiles with explicitly supplied addresses.
///
/// This helper wires together the Falcon precompile entry points
/// with their corresponding identifiers and runtime addresses.
/// It does not register the precompiles globally; callers are expected
/// to explicitly extend a `Precompiles` set as needed.
///
/// This design keeps Falcon support opt-in while EIP-8052 is still
/// under active discussion and address allocation is not finalized.
pub fn precompiles_with_addresses(addrs: FalconAddresses) -> FalconPrecompiles {
    FalconPrecompiles {
        h2p_shake256: Precompile::new(
            PrecompileId::FalconHashToPointShake256,
            addrs.h2p_shake256,
            h2p_shake256::h2p_shake256,
        ),
        falcon_core: Precompile::new(
            PrecompileId::FalconCore,
            addrs.falcon_core,
            falcon_core::falcon_core,
        ),

        #[cfg(feature = "falcon-keccakprng")]
        h2p_keccakprng: Precompile::new(
            PrecompileId::FalconHashToPointKeccakPrng,
            addrs.h2p_keccakprng,
            h2p_keccakprng::h2p_keccakprng,
        ),
    }
}

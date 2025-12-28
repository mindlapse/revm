//! Falcon-internal error reasons.
//! These errors are only used within the falcon precompiles and are not part of the exposed interface.
///
/// These errors are *not* intended to be part of the public precompile interface.
/// Falcon precompile entrypoints should generally convert these into "empty output"
/// (ECRECOVER-style) while still charging the fixed gas cost.
///
/// # Semantics
/// All `FalconError` values represent *soft failures*:
/// they must never cause the precompile to return a hard error.
/// Precompile entrypoints are expected to convert these into
/// empty output while still charging the fixed gas cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum FalconError {
    /// Input length does not match the expected ABI for this precompile function.
    InvalidInputLength { wanted: usize, got: usize },

    /// Unexpected encoding error occurred during bit packing/unpacking (panic-free handling)
    InternalEncoding,

    /// Encountered a non-canonical or otherwise invalid field element encoding.
    InvalidFieldElement,

    /// Decompression (e.g., point/polynomial decoding) failed.
    DecompressionFailed,

    /// Behavior depends on a spec feature gate that is disabled / not finalized.
    SpecNotFinalized,
}

#[cfg(feature = "std")]
mod std_impls {
    use crate::falcon::error::FalconError;

    impl std::error::Error for FalconError {}

    impl std::fmt::Display for FalconError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                FalconError::InternalEncoding => write!(f, "falcon: internal encoding error"),
                FalconError::InvalidInputLength { wanted, got } => {
                    write!(
                        f,
                        "falcon: invalid input length (wanted {wanted}, got {got})"
                    )
                }
                FalconError::InvalidFieldElement => write!(f, "falcon: invalid field element"),
                FalconError::DecompressionFailed => write!(f, "falcon: decompression failed"),
                FalconError::SpecNotFinalized => write!(f, "falcon: spec not finalized"),
            }
        }
    }
}

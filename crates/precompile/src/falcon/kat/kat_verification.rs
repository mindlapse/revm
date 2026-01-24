use super::kat_parser;

use crate::falcon::{
    encoding,
    falcon_core,
    sig_reader::SigReader,
    PackedFalconPolynomial,
    SALT_LEN,
    S2_COMPRESSED_LEN,
};

use super::super::ntt::{normalize_in_place, ntt};

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use rand::{rngs::StdRng, Rng, SeedableRng};

const FALCON_Q: u16 = 12289;
const FALCON_N: usize = 512;

fn read_u16_be(r: &mut impl XofReader) -> u16 {
    let mut b = [0u8; 2];
    r.read(&mut b);
    u16::from_be_bytes(b)
}

fn kat_h2p_shake256(salt: &[u8; SALT_LEN], msg: &[u8]) -> [u16; FALCON_N] {
    const ACCEPT_MAX: u16 = FALCON_Q * 5;
    const MAX_TRIES_PER_COEFF: usize = 35;

    let mut hasher = Shake256::default();
    hasher.update(salt);
    hasher.update(msg);
    let mut r = hasher.finalize_xof();

    let mut out = [0u16; FALCON_N];
    for i in 0..FALCON_N {
        let mut tries = 0usize;
        loop {
            if tries == MAX_TRIES_PER_COEFF {
                panic!("rejection sampling exceeded at coeff {i}");
            }
            tries += 1;

            let t = read_u16_be(&mut r);
            if t < ACCEPT_MAX {
                out[i] = t % FALCON_Q;
                break;
            }
        }
    }

    out
}

fn kat_pk_to_packed_poly(pk: &[u8]) -> PackedFalconPolynomial {
    assert_eq!(pk.len(), 897, "expected Falcon-512 pk length");

    // KAT pk encoding is: header(1 byte, typically 0x09) || 896 bytes of 14-bit packed coeffs.
    // Our internal packed polynomial format uses a leading 0 padding byte instead.
    let mut out = [0u8; 897];
    out[0] = 0;
    out[1..].copy_from_slice(&pk[1..]);
    out
}

fn kat_sig_to_s2_buf(sig: &[u8]) -> [u8; S2_COMPRESSED_LEN] {
    // Falcon reference signatures include a 1-byte header (type + logn), then the compressed
    // `s2` bitstream. Our EIP-8052 signature format is `salt || s2_compressed` (no header).
    assert!(!sig.is_empty(), "empty KAT sig");

    // Accept the two common Falcon signature header tags:
    // - 0x20 + logn ("ct" / compat forms in some ref outputs)
    // - 0x30 + logn ("compressed" form)
    let header = sig[0];
    let logn = header & 0x0F;
    assert_eq!(logn, 9, "unexpected logn in signature header: 0x{header:02X}");
    assert!(
        (header & 0xF0) == 0x20 || (header & 0xF0) == 0x30,
        "unexpected signature header tag: 0x{header:02X}"
    );

    let body = &sig[1..];
    assert!(
        body.len() <= S2_COMPRESSED_LEN,
        "KAT sig body too large: {} > {}",
        body.len(),
        S2_COMPRESSED_LEN
    );

    let mut out = [0u8; S2_COMPRESSED_LEN];
    out[..body.len()].copy_from_slice(body);
    out
}

#[test]
fn verify_falcon512_reference_kats() {
    // Positive test: all 100 official Falcon-512 reference vectors must verify.
    //
    // This is an end-to-end check that our implementation matches the reference KATs:
    // - Parse the `.rsp` KAT file into (pk, msg, salt, sig) tuples.
    // - Convert KAT encodings into the internal formats expected by the EIP-8052-style core verifier.
    // - Derive the challenge polynomial using the same H2P as the KAT generator.
    // - Run `falcon_core_verify` and assert success for every entry.
    let contents = include_str!("falcon512-KAT.rsp");
    let entries = kat_parser::parse_kat_file(contents).expect("parse KAT file");

    assert_eq!(100, entries.len());
    for e in &entries {
        // --- Public key ---
        // Public key: KAT encodes `h` in coefficient form. The EIP core verifier expects `h` in
        // NTT domain, so we unpack and then forward-NTT it.
        let pk_poly = kat_pk_to_packed_poly(&e.pk);
        let pk_coeffs = encoding::unpack_falcon_14bit_be_polynomial(&pk_poly)
            .unwrap_or_else(|_| panic!("pk unpack failed (count={})", e.count));

        let mut pk_ntt_i16 = [0i16; FALCON_N];
        for i in 0..FALCON_N {
            pk_ntt_i16[i] = pk_coeffs[i] as i16;
        }
        ntt(&mut pk_ntt_i16).unwrap_or_else(|err| panic!("pk ntt failed (count={}): {err:?}", e.count));
        normalize_in_place(&mut pk_ntt_i16);

        let mut pk_ntt = [0u16; FALCON_N];
        for i in 0..FALCON_N {
            pk_ntt[i] = pk_ntt_i16[i] as u16;
        }

        // --- Signature ---
        // Signature: KAT provides (salt, sig[0..sig_len]); our verifier expects fixed-size `s2`.
        // We decode the compressed `s2` from the reference signature (after stripping the 1-byte
        // Falcon header) into 512 signed coefficients.
        let s2_buf = kat_sig_to_s2_buf(&e.sig);
        let sig_coeffs = SigReader::new(&s2_buf)
            .read_coefficients()
            .unwrap_or_else(|| panic!("sig decode failed (count={})", e.count));

        // --- Challenge / H2P ---
        // Challenge: KAT uses SHAKE256(salt || msg) directly.
        let challenge = kat_h2p_shake256(&e.salt, &e.msg);

        // --- Verification ---
        let ok = falcon_core::falcon_core_verify(&sig_coeffs, &pk_ntt, &challenge)
            .unwrap_or_else(|err| panic!("core verify errored (count={}): {err:?}", e.count));

        assert!(ok, "KAT signature verification failed (count={})", e.count);
    }
}

#[test]
fn verify_falcon512_reference_kats_rejects_corrupted_s2() {
    // Negative test: if we corrupt the decoded signature coefficients (s2) *after decoding*,
    // verification must not succeed.
    //
    // This is a robustness check that the verifier is actually binding to the signature data:
    // - Decode the original signature successfully.
    // - Verify the original signature succeeds (sanity).
    // - Deterministically pick one coefficient and flip one byte in its i32 encoding.
    // - Re-run verification and assert it does not return `Ok(true)`.
    let contents = include_str!("falcon512-KAT.rsp");
    let entries = kat_parser::parse_kat_file(contents).expect("parse KAT file");

    assert_eq!(100, entries.len());

    // Deterministic: corruption locations are stable across runs.
    let mut rng = StdRng::seed_from_u64(0xFA1C_0BAD_D00D_u64);

    for e in &entries {
        // --- Prepare public key in NTT domain ---
        let pk_poly = kat_pk_to_packed_poly(&e.pk);
        let pk_coeffs = encoding::unpack_falcon_14bit_be_polynomial(&pk_poly)
            .unwrap_or_else(|_| panic!("pk unpack failed (count={})", e.count));

        let mut pk_ntt_i16 = [0i16; FALCON_N];
        for i in 0..FALCON_N {
            pk_ntt_i16[i] = pk_coeffs[i] as i16;
        }
        ntt(&mut pk_ntt_i16)
            .unwrap_or_else(|err| panic!("pk ntt failed (count={}): {err:?}", e.count));
        normalize_in_place(&mut pk_ntt_i16);

        let mut pk_ntt = [0u16; FALCON_N];
        for i in 0..FALCON_N {
            pk_ntt[i] = pk_ntt_i16[i] as u16;
        }

        // --- Decode signature coefficients from the original signature ---
        // IMPORTANT: decoding must be performed on the original signature.
        // The corruption is applied *only after* decoding has succeeded.
        let s2_buf = kat_sig_to_s2_buf(&e.sig);
        let sig_coeffs = SigReader::new(&s2_buf)
            .read_coefficients()
            .unwrap_or_else(|| panic!("sig decode failed (count={})", e.count));

        // --- Challenge: KAT uses SHAKE256(salt || msg) directly ---
        let challenge = kat_h2p_shake256(&e.salt, &e.msg);

        // Sanity check: the unmodified KAT vector must verify.
        let ok = falcon_core::falcon_core_verify(&sig_coeffs, &pk_ntt, &challenge)
            .unwrap_or_else(|err| panic!("core verify errored (count={}): {err:?}", e.count));
        assert!(ok, "baseline KAT signature verification failed (count={})", e.count);

        // --- Corrupt one byte *after decoding* ---
        // We flip the least-significant byte of one coefficient's little-endian i32 encoding.
        // This guarantees the decoded byte changes while keeping the overall value i16-safe.
        let mut corrupted = sig_coeffs;
        let coeff_index: usize = rng.random_range(0..FALCON_N);

        debug_assert!(
            (i16::MIN as i32..=i16::MAX as i32).contains(&corrupted[coeff_index]),
            "expected decoded s2 coefficient to fit i16"
        );

        let before = corrupted[coeff_index].to_le_bytes();
        let mut after = before;
        after[0] ^= 0xFF;
        assert_ne!(after[0], before[0], "byte flip did not change the byte");
        corrupted[coeff_index] = i32::from_le_bytes(after);

        // Verification must fail (either returns false, or rejects encoding).
        //
        // NOTE: We accept either outcome as a "failure" for this negative test:
        // - `Ok(false)`: the signature is well-formed but invalid.
        // - `Err(_)`: the signature is rejected by validation/canonicality checks.
        match falcon_core::falcon_core_verify(&corrupted, &pk_ntt, &challenge) {
            Ok(true) => panic!(
                "corrupted signature unexpectedly verified (count={}, coeff_index={})",
                e.count, coeff_index
            ),
            Ok(false) | Err(_) => {}
        }
    }
}

use std::{fmt, fs, path::Path};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KatEntry {
    pub count: u32,
    pub seed: Vec<u8>,  // 48 bytes
    pub mlen: usize,
    pub msg: Vec<u8>,   // mlen bytes
    pub pk: Vec<u8>,    // 897 bytes
    pub sk: Vec<u8>,    // 1281 bytes
    pub smlen: usize,
    pub sm: Vec<u8>,    // smlen bytes

    // Convenience fields derived from sm layout:
    // sm = u16_be(siglen) || salt[40] || msg || sig[siglen]
    pub sig_len: u16,
    pub salt: [u8; 40],
    pub sig: Vec<u8>,
}

#[derive(Debug)]
pub enum KatParseError {
    Io(std::io::Error),
    MissingField { count: Option<u32>, field: &'static str },
    BadLine { line_no: usize, line: String, why: String },
    BadHex { line_no: usize, field: &'static str, why: String },
    BadLen { count: Option<u32>, field: &'static str, expected: usize, got: usize },
    SmLayout { count: Option<u32>, why: String },
}

impl fmt::Display for KatParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use KatParseError::*;
        match self {
            Io(e) => write!(f, "io error: {e}"),
            MissingField { count, field } => write!(f, "missing field {field} (count={count:?})"),
            BadLine { line_no, why, .. } => write!(f, "bad line at {line_no}: {why}"),
            BadHex { line_no, field, why } => write!(f, "bad hex for {field} at line {line_no}: {why}"),
            BadLen { count, field, expected, got } => {
                write!(f, "bad length for {field} (count={count:?}): expected {expected}, got {got}")
            }
            SmLayout { count, why } => write!(f, "bad sm layout (count={count:?}): {why}"),
        }
    }
}

impl From<std::io::Error> for KatParseError {
    fn from(e: std::io::Error) -> Self {
        KatParseError::Io(e)
    }
}

fn hex_to_bytes_strict(s: &str, line_no: usize, field: &'static str) -> Result<Vec<u8>, KatParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    if s.len() % 2 != 0 {
        return Err(KatParseError::BadHex {
            line_no,
            field,
            why: format!("odd number of hex chars: {}", s.len()),
        });
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = bytes[i];
        let lo = bytes[i + 1];
        let hv = (hi as char).to_digit(16);
        let lv = (lo as char).to_digit(16);
        let (hv, lv) = match (hv, lv) {
            (Some(h), Some(l)) => (h, l),
            _ => {
                return Err(KatParseError::BadHex {
                    line_no,
                    field,
                    why: format!("non-hex at positions {}..{}: {:?}{:?}", i, i + 2, hi as char, lo as char),
                })
            }
        };
        out.push(((hv << 4) | lv) as u8);
    }
    Ok(out)
}

fn parse_u64_after_equals(line: &str, line_no: usize) -> Result<u64, KatParseError> {
    let (_, rhs) = line
        .split_once('=')
        .ok_or_else(|| KatParseError::BadLine {
            line_no,
            line: line.to_string(),
            why: "expected '='".to_string(),
        })?;
    rhs.trim().parse::<u64>().map_err(|e| KatParseError::BadLine {
        line_no,
        line: line.to_string(),
        why: format!("failed to parse integer: {e}"),
    })
}

fn parse_hex_after_equals(line: &str, line_no: usize, field: &'static str) -> Result<Vec<u8>, KatParseError> {
    let (_, rhs) = line
        .split_once('=')
        .ok_or_else(|| KatParseError::BadLine {
            line_no,
            line: line.to_string(),
            why: "expected '='".to_string(),
        })?;
    hex_to_bytes_strict(rhs, line_no, field)
}

fn derive_sm_parts(count: Option<u32>, msg: &[u8], sm: &[u8]) -> Result<(u16, [u8; 40], Vec<u8>), KatParseError> {
    if sm.len() < 2 + 40 {
        return Err(KatParseError::SmLayout {
            count,
            why: format!("sm too short: {}", sm.len()),
        });
    }
    let sig_len = u16::from_be_bytes([sm[0], sm[1]]);
    let mut salt = [0u8; 40];
    salt.copy_from_slice(&sm[2..42]);

    let after_salt = 2 + 40;
    if sm.len() < after_salt + msg.len() {
        return Err(KatParseError::SmLayout {
            count,
            why: format!(
                "sm shorter than prefix+msg: sm={}, need at least {}",
                sm.len(),
                after_salt + msg.len()
            ),
        });
    }
    let msg_in_sm = &sm[after_salt..after_salt + msg.len()];
    if msg_in_sm != msg {
        return Err(KatParseError::SmLayout {
            count,
            why: "msg embedded in sm does not match msg field".to_string(),
        });
    }

    let sig_start = after_salt + msg.len();
    let sig_end = sig_start
        .checked_add(sig_len as usize)
        .ok_or_else(|| KatParseError::SmLayout {
            count,
            why: "sig_len overflow".to_string(),
        })?;

    if sm.len() < sig_end {
        return Err(KatParseError::SmLayout {
            count,
            why: format!("sm too short for declared sig_len: sm={}, need {}", sm.len(), sig_end),
        });
    }
    // Some files may include no extra bytes after sig; if they do, we treat it as an error.
    if sm.len() != sig_end {
        return Err(KatParseError::SmLayout {
            count,
            why: format!("sm has trailing bytes: sm={}, expected {}", sm.len(), sig_end),
        });
    }

    let sig = sm[sig_start..sig_end].to_vec();
    Ok((sig_len, salt, sig))
}

/// Parse a Falcon C-reference KAT `.rsp` file into entries.
///
/// Expects entries with fields:
/// count, seed, mlen, msg, pk, sk, smlen, sm
///
/// Also validates:
/// - seed len = 48
/// - pk len = 897
/// - sk len = 1281
/// - msg len matches mlen
/// - sm len matches smlen
/// - sm layout: u16_be(siglen) || salt[40] || msg || sig
pub fn parse_kat_file(contents: &str) -> Result<Vec<KatEntry>, KatParseError> {
    let mut out: Vec<KatEntry> = Vec::new();

    // We parse line-by-line, starting a new entry at "count =".
    let mut cur: Option<KatEntry> = None;

    for (idx, raw_line) in contents.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();

        // Skip empty lines and comment header lines (some .rsp start with "# ...")
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with("count") {
            // If we had an in-progress entry, it must be complete (but spec says entries end after sm line).
            if cur.is_some() {
                return Err(KatParseError::BadLine {
                    line_no,
                    line: raw_line.to_string(),
                    why: "found 'count' before previous entry ended (expected sm line)".to_string(),
                });
            }
            let count = parse_u64_after_equals(line, line_no)? as u32;
            cur = Some(KatEntry {
                count,
                seed: Vec::new(),
                mlen: 0,
                msg: Vec::new(),
                pk: Vec::new(),
                sk: Vec::new(),
                smlen: 0,
                sm: Vec::new(),
                sig_len: 0,
                salt: [0u8; 40],
                sig: Vec::new(),
            });
            continue;
        }

        let Some(ref mut e) = cur else {
            // Ignore any preamble until first count.
            continue;
        };

        if line.starts_with("seed") {
            e.seed = parse_hex_after_equals(line, line_no, "seed")?;
        } else if line.starts_with("mlen") {
            e.mlen = parse_u64_after_equals(line, line_no)? as usize;
        } else if line.starts_with("msg") {
            e.msg = parse_hex_after_equals(line, line_no, "msg")?;
        } else if line.starts_with("pk") {
            e.pk = parse_hex_after_equals(line, line_no, "pk")?;
        } else if line.starts_with("sk") {
            e.sk = parse_hex_after_equals(line, line_no, "sk")?;
        } else if line.starts_with("smlen") {
            e.smlen = parse_u64_after_equals(line, line_no)? as usize;
        } else if line.starts_with("sm") {
            e.sm = parse_hex_after_equals(line, line_no, "sm")?;

            // Validate required fields before finalizing.
            let count_opt = Some(e.count);

            if e.seed.is_empty() {
                return Err(KatParseError::MissingField { count: count_opt, field: "seed" });
            }
            if e.mlen == 0 && !e.msg.is_empty() {
                // allow mlen=0 case, but if msg is present and mlen not set, it's suspicious
            }
            if e.msg.len() != e.mlen {
                return Err(KatParseError::BadLen {
                    count: count_opt,
                    field: "msg",
                    expected: e.mlen,
                    got: e.msg.len(),
                });
            }
            if e.pk.is_empty() {
                return Err(KatParseError::MissingField { count: count_opt, field: "pk" });
            }
            if e.sk.is_empty() {
                return Err(KatParseError::MissingField { count: count_opt, field: "sk" });
            }
            if e.smlen == 0 && e.sm.is_empty() {
                return Err(KatParseError::MissingField { count: count_opt, field: "smlen/sm" });
            }
            if e.sm.len() != e.smlen {
                return Err(KatParseError::BadLen {
                    count: count_opt,
                    field: "sm",
                    expected: e.smlen,
                    got: e.sm.len(),
                });
            }

            // Common Falcon-512 reference sizes.
            if e.seed.len() != 48 {
                return Err(KatParseError::BadLen {
                    count: count_opt,
                    field: "seed",
                    expected: 48,
                    got: e.seed.len(),
                });
            }
            if e.pk.len() != 897 {
                return Err(KatParseError::BadLen {
                    count: count_opt,
                    field: "pk",
                    expected: 897,
                    got: e.pk.len(),
                });
            }
            if e.sk.len() != 1281 {
                return Err(KatParseError::BadLen {
                    count: count_opt,
                    field: "sk",
                    expected: 1281,
                    got: e.sk.len(),
                });
            }

            // Derive and validate the embedded layout.
            let (sig_len, salt, sig) = derive_sm_parts(count_opt, &e.msg, &e.sm)?;
            e.sig_len = sig_len;
            e.salt = salt;
            e.sig = sig;

            // Finalize entry.
            out.push(cur.take().expect("cur was Some"));
        } else {
            return Err(KatParseError::BadLine {
                line_no,
                line: raw_line.to_string(),
                why: "unexpected line (unknown marker)".to_string(),
            });
        }
    }

    if cur.is_some() {
        return Err(KatParseError::BadLine {
            line_no: contents.lines().count(),
            line: "".to_string(),
            why: "file ended mid-entry (missing sm line)".to_string(),
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_entry_and_derives_sig_salt() {
        // Tiny synthetic entry where msg is 1 byte and sig is 3 bytes.
        // siglen=3 => 0x0003, salt=40 bytes of 0xAA, msg=0xBB, sig=0x010203
        let mut sm = vec![0x00, 0x03];
        sm.extend(std::iter::repeat(0xAA).take(40));
        sm.push(0xBB);
        sm.extend([0x01, 0x02, 0x03]);

        let kat = format!(
            r#"
count = 0
seed = {}
mlen = 1
msg = BB
pk = {}
sk = {}
smlen = {}
sm = {}
"#,
            "00".repeat(48),
            "09".to_string() + &"00".repeat(896),
            "11".repeat(1281),
            sm.len(),
            sm.iter().map(|b| format!("{:02X}", b)).collect::<String>(),
        );

        let v = parse_kat_file(&kat).unwrap();
        assert_eq!(v.len(), 1);
        let e = &v[0];
        assert_eq!(e.count, 0);
        assert_eq!(e.seed.len(), 48);
        assert_eq!(e.mlen, 1);
        assert_eq!(e.msg, vec![0xBB]);
        assert_eq!(e.pk.len(), 897);
        assert_eq!(e.sk.len(), 1281);
        assert_eq!(e.smlen, sm.len());
        assert_eq!(e.sm, sm);
        assert_eq!(e.sig_len, 3);
        assert_eq!(e.salt, [0xAA; 40]);
        assert_eq!(e.sig, vec![0x01, 0x02, 0x03]);
    }

    
}

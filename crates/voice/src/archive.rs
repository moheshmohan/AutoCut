//! A strict reader of the model archive (`.tar.gz`): one gzip member (RFC 1952: header fields
//! parsed, the CRC-32 and length of the inflated data checked) holding a tar archive (POSIX
//! ustar / GNU headers, each header's checksum checked). Only regular files are returned, by
//! their base name; extended headers and directories are skipped. Sizes are capped, every
//! length is checked against what is left, and anything malformed is an error, never a panic:
//! the file is untrusted input even though its SHA-256 was checked when it was downloaded.

use crate::VoiceError;

/// Largest archive read, and the largest it may inflate to.
pub const MAX_ARCHIVE: usize = 64 << 20;
pub const MAX_INFLATED: usize = 128 << 20;
/// Most tar entries.
const MAX_ENTRIES: usize = 256;
const BLOCK: usize = 512;

fn bad(what: impl Into<String>) -> VoiceError {
    VoiceError::Model(format!("model archive: {}", what.into()))
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Inflate a single-member gzip file.
pub fn gunzip(data: &[u8]) -> Result<Vec<u8>, VoiceError> {
    if data.len() > MAX_ARCHIVE {
        return Err(bad(format!("larger than {MAX_ARCHIVE} bytes")));
    }
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b {
        return Err(bad("not a gzip file"));
    }
    if data[2] != 8 {
        return Err(bad("unknown gzip compression method"));
    }
    let flags = data[3];
    if flags & 0xE0 != 0 {
        return Err(bad("reserved gzip flags set"));
    }
    let mut pos = 10usize;
    let take = |pos: &mut usize, n: usize| -> Result<(), VoiceError> {
        let end = pos.checked_add(n).filter(|e| *e <= data.len() - 8).ok_or_else(|| bad("truncated gzip header"))?;
        *pos = end;
        Ok(())
    };
    if flags & 0x04 != 0 {
        let xlen = data.get(pos..pos + 2).map(|b| usize::from(u16::from_le_bytes([b[0], b[1]]))).ok_or_else(|| bad("truncated gzip header"))?;
        take(&mut pos, 2 + xlen)?;
    }
    for flag in [0x08u8, 0x10] {
        if flags & flag != 0 {
            let rest = data.get(pos..data.len() - 8).ok_or_else(|| bad("truncated gzip header"))?;
            let z = rest.iter().position(|b| *b == 0).ok_or_else(|| bad("unterminated gzip name"))?;
            take(&mut pos, z + 1)?;
        }
    }
    if flags & 0x02 != 0 {
        take(&mut pos, 2)?;
    }
    let body = data.get(pos..data.len() - 8).ok_or_else(|| bad("truncated gzip data"))?;
    let out = miniz_oxide::inflate::decompress_to_vec_with_limit(body, MAX_INFLATED).map_err(|e| bad(format!("inflate: {e:?}")))?;
    let tail = &data[data.len() - 8..];
    let crc = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    let size = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
    if size != out.len() as u32 {
        return Err(bad("inflated size does not match the gzip trailer"));
    }
    if crc != crc32(&out) {
        return Err(bad("CRC-32 mismatch"));
    }
    Ok(out)
}

fn octal(field: &[u8]) -> Result<u64, VoiceError> {
    let mut v = 0u64;
    let mut seen = false;
    for &b in field {
        match b {
            b'0'..=b'7' => {
                v = v.checked_mul(8).and_then(|v| v.checked_add(u64::from(b - b'0'))).ok_or_else(|| bad("tar number too large"))?;
                seen = true;
            }
            b' ' | 0 if !seen => {}
            b' ' | 0 => break,
            _ => return Err(bad("tar number is not octal")),
        }
    }
    Ok(v)
}

/// A regular file from the archive.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Path as stored (`tmp/export/enc.onnx`).
    pub path: String,
    pub data: Vec<u8>,
}

impl Entry {
    /// The last path component.
    pub fn base_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

/// The regular files of a tar archive.
pub fn untar(data: &[u8]) -> Result<Vec<Entry>, VoiceError> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut entries = 0usize;
    loop {
        let Some(h) = data.get(pos..pos + BLOCK) else {
            return Err(bad("tar archive ends without its end blocks"));
        };
        if h.iter().all(|b| *b == 0) {
            return Ok(out);
        }
        entries += 1;
        if entries > MAX_ENTRIES {
            return Err(bad("too many tar entries"));
        }
        let stored = octal(&h[148..156])?;
        let sum: u64 = h.iter().enumerate().map(|(i, b)| if (148..156).contains(&i) { 32 } else { u64::from(*b) }).sum();
        if stored != sum {
            return Err(bad("tar header checksum mismatch"));
        }
        let size = usize::try_from(octal(&h[124..136])?).map_err(|_| bad("tar entry too large"))?;
        if size > MAX_INFLATED {
            return Err(bad("tar entry too large"));
        }
        let name = |r: std::ops::Range<usize>| String::from_utf8_lossy(h[r].split(|b| *b == 0).next().unwrap_or(&[])).into_owned();
        let mut path = name(0..100);
        if &h[257..262] == b"ustar" {
            let prefix = name(345..500);
            if !prefix.is_empty() {
                path = format!("{prefix}/{path}");
            }
        }
        let start = pos + BLOCK;
        let end = start.checked_add(size).filter(|e| *e <= data.len()).ok_or_else(|| bad("tar entry runs past the end"))?;
        let kind = h[156];
        if kind == b'0' || kind == 0 {
            out.push(Entry { path, data: data[start..end].to_vec() });
        }
        let padded = size.div_ceil(BLOCK) * BLOCK;
        pos = start.checked_add(padded).ok_or_else(|| bad("tar entry too large"))?;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tar archive of `files` (ustar headers).
    pub(crate) fn tar(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, data) in files {
            let mut h = [0u8; BLOCK];
            h[..name.len()].copy_from_slice(name.as_bytes());
            h[100..107].copy_from_slice(b"0000644");
            h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
            h[136..147].copy_from_slice(b"00000000000");
            h[156] = b'0';
            h[257..263].copy_from_slice(b"ustar\0");
            h[263..265].copy_from_slice(b"00");
            h[148..156].copy_from_slice(b"        ");
            let sum: u32 = h.iter().map(|b| u32::from(*b)).sum();
            h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
            out.extend_from_slice(&h);
            out.extend_from_slice(data);
            out.resize(out.len().div_ceil(BLOCK) * BLOCK, 0);
        }
        out.extend_from_slice(&[0u8; 2 * BLOCK]);
        out
    }

    /// A gzip member of `data` (stored with the given flags' fields).
    pub(crate) fn gzip(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x1f, 0x8b, 8, 0x08, 0, 0, 0, 0, 0, 3];
        out.extend_from_slice(b"model.tar\0");
        out.extend_from_slice(&miniz_oxide::deflate::compress_to_vec(data, 6));
        out.extend_from_slice(&crc32(data).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out
    }

    #[test]
    fn round_trips_and_checks() {
        let t = tar(&[("tmp/export/config.ini", b"[df]\nsr = 48000\n"), ("tmp/export/enc.onnx", &[7u8; 1000])]);
        let gz = gzip(&t);
        let back = gunzip(&gz).unwrap();
        assert_eq!(back, t);
        let files = untar(&back).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[1].base_name(), "enc.onnx");
        assert_eq!(files[1].data, vec![7u8; 1000]);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn malformed_archives_are_errors_not_panics() {
        let t = tar(&[("a/config.ini", b"[df]\n"), ("a/enc.onnx", &[1u8; 3000])]);
        let gz = gzip(&t);
        let mut rng = 0x1234_5678_9abc_def0u64;
        for i in 0..400 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mut g = gz.clone();
            let mut tt = t.clone();
            match i % 3 {
                0 => g.truncate((rng % g.len() as u64) as usize),
                1 => {
                    let at = (rng % g.len() as u64) as usize;
                    g[at] ^= 1 << (rng % 8);
                }
                _ => {
                    let at = (rng % tt.len() as u64) as usize;
                    tt[at] ^= 1 << (rng % 8);
                }
            }
            let r = std::panic::catch_unwind(|| {
                let _ = gunzip(&g);
                let _ = untar(&tt);
                let _ = untar(&tt[..tt.len() / 2]);
            });
            assert!(r.is_ok(), "case {i} panicked");
        }
        assert!(gunzip(b"not gzip at all, really").is_err());
        assert!(untar(&[0u8; 100]).is_err());
    }
}

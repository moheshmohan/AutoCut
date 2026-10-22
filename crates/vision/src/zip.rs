//! Reading one file out of a zip archive (the pose network's ONNX file comes in OpenMMLab's
//! zip): the end-of-central-directory record, the central directory, the entry's local header,
//! then its data stored or deflated (miniz_oxide, pure Rust), checked against its CRC-32 and
//! size. Every offset and length is checked against the archive; the inflated size is capped;
//! ZIP64, encryption and other methods are refused.

use crate::VisionError;

/// Largest file taken out of an archive.
pub const MAX_ENTRY: usize = 256 << 20;
/// Most central-directory entries looked at.
const MAX_ENTRIES: usize = 10_000;

fn bad(what: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("zip: {what}"))
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    let s = b.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([s[0], s[1]]))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// CRC-32 (IEEE, reflected), table-driven.
pub fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// One central-directory entry.
struct Entry {
    name: String,
    method: u16,
    flags: u16,
    crc: u32,
    packed: usize,
    size: usize,
    local: usize,
}

fn entries(zip: &[u8]) -> Result<Vec<Entry>, VisionError> {
    // the end-of-central-directory record: 22 bytes and a comment of at most 64 KiB
    let lo = zip.len().saturating_sub(22 + 0xFFFF);
    let eocd = (lo..=zip.len().saturating_sub(22)).rev().find(|&i| u32_at(zip, i) == Some(0x0605_4b50)).ok_or_else(|| bad("not a zip archive"))?;
    let count = usize::from(u16_at(zip, eocd + 10).ok_or_else(|| bad("truncated"))?);
    let cd_size = u32_at(zip, eocd + 12).ok_or_else(|| bad("truncated"))? as usize;
    let cd_at = u32_at(zip, eocd + 16).ok_or_else(|| bad("truncated"))? as usize;
    if count == 0xFFFF || cd_at == 0xFFFF_FFFF || cd_size == 0xFFFF_FFFF {
        return Err(bad("ZIP64 archives are not supported"));
    }
    if count > MAX_ENTRIES || cd_at.checked_add(cd_size).is_none_or(|e| e > eocd) {
        return Err(bad("damaged central directory"));
    }
    let mut out = Vec::with_capacity(count);
    let mut at = cd_at;
    for _ in 0..count {
        if u32_at(zip, at) != Some(0x0201_4b50) {
            return Err(bad("damaged central directory entry"));
        }
        let field = |o: usize| u16_at(zip, at + o).ok_or_else(|| bad("truncated central directory"));
        let word = |o: usize| u32_at(zip, at + o).ok_or_else(|| bad("truncated central directory"));
        let (flags, method) = (field(8)?, field(10)?);
        let (crc, packed, size) = (word(16)?, word(20)? as usize, word(24)? as usize);
        let (name_len, extra_len, comment_len) = (usize::from(field(28)?), usize::from(field(30)?), usize::from(field(32)?));
        let local = word(42)? as usize;
        let name_bytes = zip.get(at + 46..at + 46 + name_len).ok_or_else(|| bad("truncated file name"))?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        out.push(Entry { name, method, flags, crc, packed, size, local });
        at = at.checked_add(46 + name_len + extra_len + comment_len).ok_or_else(|| bad("damaged central directory"))?;
    }
    Ok(out)
}

/// The names of the files in an archive.
pub fn names(zip: &[u8]) -> Result<Vec<String>, VisionError> {
    Ok(entries(zip)?.into_iter().map(|e| e.name).collect())
}

/// The contents of the first file whose name ends with `suffix` (at most `max` bytes).
pub fn read_entry(zip: &[u8], suffix: &str, max: usize) -> Result<Vec<u8>, VisionError> {
    let e = entries(zip)?.into_iter().find(|e| e.name.ends_with(suffix)).ok_or_else(|| bad(format!("no file ending in {suffix}")))?;
    if e.flags & 1 != 0 {
        return Err(bad("encrypted entries are not supported"));
    }
    if e.size > max.min(MAX_ENTRY) || e.packed == 0xFFFF_FFFF || e.size == 0xFFFF_FFFF {
        return Err(bad(format!("{} is too large", e.name)));
    }
    if u32_at(zip, e.local) != Some(0x0403_4b50) {
        return Err(bad("damaged local header"));
    }
    let name_len = usize::from(u16_at(zip, e.local + 26).ok_or_else(|| bad("truncated local header"))?);
    let extra_len = usize::from(u16_at(zip, e.local + 28).ok_or_else(|| bad("truncated local header"))?);
    let start = e.local.checked_add(30 + name_len + extra_len).ok_or_else(|| bad("damaged local header"))?;
    let body = zip.get(start..start.checked_add(e.packed).ok_or_else(|| bad("damaged sizes"))?).ok_or_else(|| bad("truncated entry"))?;
    let data = match e.method {
        0 => body.to_vec(),
        8 => miniz_oxide::inflate::decompress_to_vec_with_limit(body, e.size).map_err(|er| bad(format!("inflate: {er:?}")))?,
        m => return Err(bad(format!("compression method {m} is not supported"))),
    };
    if data.len() != e.size || crc32(&data) != e.crc {
        return Err(bad(format!("{} does not match its size or checksum", e.name)));
    }
    Ok(data)
}

/// A zip archive of `files` (stored, or deflated with `deflate`), for tests.
#[cfg(test)]
pub(crate) fn write(files: &[(&str, &[u8])], deflate: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cd = Vec::new();
    for (name, data) in files {
        let local = out.len() as u32;
        let packed = if deflate { miniz_oxide::deflate::compress_to_vec(data, 6) } else { data.to_vec() };
        let method: u16 = if deflate { 8 } else { 0 };
        let crc = crc32(data);
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0]);
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&packed);
        cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        cd.extend_from_slice(&[20, 0, 20, 0, 0, 0]);
        cd.extend_from_slice(&method.to_le_bytes());
        cd.extend_from_slice(&[0; 4]);
        cd.extend_from_slice(&crc.to_le_bytes());
        cd.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&[0; 12]);
        cd.extend_from_slice(&local.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
    }
    let cd_at = out.len() as u32;
    out.extend_from_slice(&cd);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(cd.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_at.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn reads_stored_and_deflated_entries() {
        let model: Vec<u8> = (0..5000u32).map(|i| (i * 7 % 251) as u8).collect();
        for deflate in [false, true] {
            let z = write(&[("dir/", b""), ("dir/deploy.json", b"{}"), ("dir/end2end.onnx", &model)], deflate);
            assert_eq!(names(&z).unwrap().len(), 3);
            assert_eq!(read_entry(&z, "end2end.onnx", 1 << 20).unwrap(), model);
            assert!(read_entry(&z, "missing.onnx", 1 << 20).is_err());
            // the size cap
            assert!(read_entry(&z, "end2end.onnx", 100).is_err());
        }
    }

    #[test]
    fn damaged_archives_are_errors_not_panics() {
        let model: Vec<u8> = (0..300u32).map(|i| (i % 13) as u8).collect();
        let z = write(&[("a/deploy.json", b"{\"x\": 1}"), ("a/end2end.onnx", &model)], true);
        for cut in 0..z.len() {
            let r = std::panic::catch_unwind(|| read_entry(&z[..cut], "end2end.onnx", 1 << 20).map(|v| v.len()));
            assert!(r.is_ok(), "panicked at length {cut}");
        }
        for i in 0..z.len() {
            for bit in [0, 3, 7] {
                let mut m = z.clone();
                m[i] ^= 1 << bit;
                let r = std::panic::catch_unwind(|| read_entry(&m, "end2end.onnx", 1 << 20).map(|v| v.len()));
                assert!(r.is_ok(), "panicked flipping bit {bit} of byte {i}");
                // a flipped bit in the data is caught by the checksum, never returned as the model
                if let Ok(Ok(n)) = r {
                    assert_eq!(n, model.len());
                }
            }
        }
        assert!(read_entry(b"", "x", 10).is_err());
        assert!(read_entry(&[0u8; 100], "x", 10).is_err());
    }
}

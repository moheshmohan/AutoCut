//! A small ZIP reader and writer for template packages (PKWARE APPNOTE.TXT 6.3: local file
//! headers, central directory, end of central directory; methods 0 = stored and 8 = deflate).
//!
//! Everything in an archive is hostile: the reader caps the number of entries, every entry's size
//! and the total it inflates (so a "zip bomb" stops at the cap instead of filling memory), checks
//! every offset against the archive, verifies each entry's CRC-32 and refuses names that could
//! leave the package folder (absolute paths, `..`, drive letters, backslashes, NUL), symbolic
//! links, encryption, multi-disk archives and ZIP64. Inflating is `miniz_oxide`'s (pure Rust).
//!
//! The writer makes deterministic archives (fixed 1980-01-01 timestamps, entries in the order
//! given) so the same package always gives the same bytes.

use std::collections::BTreeMap;

/// Most entries an archive may hold.
pub const MAX_ENTRIES: usize = 1024;
/// Largest single entry, uncompressed.
pub const MAX_ENTRY_BYTES: u64 = 128 * 1024 * 1024;
/// Most bytes an archive may inflate to in all.
pub const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
/// Longest entry name, bytes.
pub const MAX_NAME_BYTES: usize = 255;
/// Deepest folder nesting of an entry name.
pub const MAX_NAME_DEPTH: usize = 16;

const LOCAL_SIG: u32 = 0x0403_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const END_SIG: u32 = 0x0605_4b50;
/// 1980-01-01 00:00 in MS-DOS date / time.
const DOS_DATE: u16 = (1 << 5) | 1;
const DOS_TIME: u16 = 0;

/// Why an archive cannot be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ZipError {
    #[error("not a ZIP archive (no end of central directory record)")]
    NotZip,
    #[error("the ZIP archive is damaged: {0}")]
    Corrupt(String),
    #[error("unsupported ZIP feature: {0}")]
    Unsupported(String),
    #[error("ZIP entry `{name}` is refused: {why}")]
    BadName { name: String, why: String },
    #[error("the ZIP archive is too large: {0}")]
    TooLarge(String),
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    let s = b.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([*s.first()?, *s.get(1)?]))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([*s.first()?, *s.get(1)?, *s.get(2)?, *s.get(3)?]))
}

fn corrupt(m: impl Into<String>) -> ZipError {
    ZipError::Corrupt(m.into())
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320), as ZIP stores it.
pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, slot) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    });
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        let i = ((c ^ u32::from(b)) & 0xFF) as usize;
        c = table.get(i).copied().unwrap_or(0) ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Check that `name` is a safe relative path inside the package: forward slashes, no empty, `.`
/// or `..` components, no drive letter, no NUL or control characters, bounded length and depth.
pub fn check_entry_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("the name is empty".into());
    }
    if name.len() > MAX_NAME_BYTES {
        return Err(format!("the name is longer than {MAX_NAME_BYTES} bytes"));
    }
    if name.contains('\\') {
        return Err("the name uses backslashes (only `/` separates folders)".into());
    }
    if name.chars().any(|c| c == '\0' || c.is_control()) {
        return Err("the name contains control characters".into());
    }
    if name.starts_with('/') {
        return Err("the name is an absolute path".into());
    }
    if name.as_bytes().get(1) == Some(&b':') {
        return Err("the name starts with a drive letter".into());
    }
    let parts: Vec<&str> = name.trim_end_matches('/').split('/').collect();
    if parts.len() > MAX_NAME_DEPTH {
        return Err(format!("the name is nested deeper than {MAX_NAME_DEPTH} folders"));
    }
    for p in parts {
        if p.is_empty() {
            return Err("the name has an empty folder component (`//`)".into());
        }
        if p == "." || p == ".." {
            return Err("the name leaves its folder (`.` or `..`)".into());
        }
    }
    Ok(())
}

/// Read every file entry of `bytes` (folders are skipped). Names are checked with
/// [`check_entry_name`]; a name given twice is refused.
pub fn read(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, ZipError> {
    // the end record: 22 bytes plus a comment of at most 65 535 bytes, at the end of the file
    let min_end = bytes.len().saturating_sub(22 + 65_535);
    let mut end = None;
    let mut at = bytes.len().checked_sub(22).ok_or(ZipError::NotZip)?;
    loop {
        if u32_at(bytes, at) == Some(END_SIG) {
            let comment = u16_at(bytes, at + 20).map(usize::from).unwrap_or(usize::MAX);
            if at.checked_add(22).and_then(|e| e.checked_add(comment)) == Some(bytes.len()) {
                end = Some(at);
                break;
            }
        }
        if at == min_end || at == 0 {
            break;
        }
        at -= 1;
    }
    let end = end.ok_or(ZipError::NotZip)?;
    let disk = u16_at(bytes, end + 4).ok_or_else(|| corrupt("short end record"))?;
    let cd_disk = u16_at(bytes, end + 6).ok_or_else(|| corrupt("short end record"))?;
    let on_disk = u16_at(bytes, end + 8).ok_or_else(|| corrupt("short end record"))?;
    let total = u16_at(bytes, end + 10).ok_or_else(|| corrupt("short end record"))?;
    let cd_size = u32_at(bytes, end + 12).ok_or_else(|| corrupt("short end record"))?;
    let cd_off = u32_at(bytes, end + 16).ok_or_else(|| corrupt("short end record"))?;
    if total == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        return Err(ZipError::Unsupported("ZIP64 archives (templates are small: use a plain ZIP)".into()));
    }
    if disk != 0 || cd_disk != 0 || on_disk != total {
        return Err(ZipError::Unsupported("multi-disk (split) archives".into()));
    }
    if usize::from(total) > MAX_ENTRIES {
        return Err(ZipError::TooLarge(format!("{total} entries (at most {MAX_ENTRIES})")));
    }
    let cd_start = cd_off as usize;
    let cd_end = cd_start.checked_add(cd_size as usize).ok_or_else(|| corrupt("central directory size overflows"))?;
    if cd_end > end {
        return Err(corrupt("the central directory runs past the end record"));
    }
    let mut out = BTreeMap::new();
    let mut inflated: u64 = 0;
    let mut p = cd_start;
    for i in 0..usize::from(total) {
        if u32_at(bytes, p) != Some(CENTRAL_SIG) {
            return Err(corrupt(format!("central directory entry {i} has no signature")));
        }
        let field = |off: usize| u16_at(bytes, p + off).ok_or_else(|| corrupt(format!("central directory entry {i} is truncated")));
        let field32 = |off: usize| u32_at(bytes, p + off).ok_or_else(|| corrupt(format!("central directory entry {i} is truncated")));
        let made_by = field(4)?;
        let flags = field(8)?;
        let method = field(10)?;
        let crc = field32(16)?;
        let csize = field32(20)?;
        let usize_ = field32(24)?;
        let name_len = usize::from(field(28)?);
        let extra_len = usize::from(field(30)?);
        let comment_len = usize::from(field(32)?);
        let external = field32(38)?;
        let local = field32(42)?;
        let name_at = p + 46;
        let raw = bytes.get(name_at..name_at + name_len).ok_or_else(|| corrupt(format!("central directory entry {i}: name runs past the directory")))?;
        let name =
            std::str::from_utf8(raw).map_err(|_| ZipError::BadName { name: String::from_utf8_lossy(raw).into(), why: "the name is not UTF-8".into() })?;
        p = name_at
            .checked_add(name_len)
            .and_then(|x| x.checked_add(extra_len))
            .and_then(|x| x.checked_add(comment_len))
            .ok_or_else(|| corrupt("central directory entry sizes overflow"))?;
        if p > cd_end {
            return Err(corrupt(format!("central directory entry `{name}` runs past the directory")));
        }
        if name.ends_with('/') {
            check_entry_name(name).map_err(|why| ZipError::BadName { name: name.into(), why })?;
            continue;
        }
        check_entry_name(name).map_err(|why| ZipError::BadName { name: name.into(), why })?;
        // Unix-made entries carry the file mode in the high half of the external attributes
        if made_by >> 8 == 3 && (external >> 16) & 0o170_000 == 0o120_000 {
            return Err(ZipError::BadName { name: name.into(), why: "symbolic links are not allowed".into() });
        }
        if flags & 1 != 0 || flags & 0x40 != 0 {
            return Err(ZipError::Unsupported(format!("encrypted entry `{name}`")));
        }
        if csize == 0xFFFF_FFFF || usize_ == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
            return Err(ZipError::Unsupported(format!("ZIP64 entry `{name}`")));
        }
        if u64::from(usize_) > MAX_ENTRY_BYTES {
            return Err(ZipError::TooLarge(format!("entry `{name}` is {usize_} bytes (at most {MAX_ENTRY_BYTES})")));
        }
        inflated = inflated.saturating_add(u64::from(usize_));
        if inflated > MAX_TOTAL_BYTES {
            return Err(ZipError::TooLarge(format!("the entries add up to more than {MAX_TOTAL_BYTES} bytes")));
        }
        let lh = local as usize;
        if lh >= cd_start || u32_at(bytes, lh) != Some(LOCAL_SIG) {
            return Err(corrupt(format!("entry `{name}`: no local header where the directory says")));
        }
        let lname = usize::from(u16_at(bytes, lh + 26).ok_or_else(|| corrupt("short local header"))?);
        let lextra = usize::from(u16_at(bytes, lh + 28).ok_or_else(|| corrupt("short local header"))?);
        let data_at = lh.checked_add(30 + lname + lextra).ok_or_else(|| corrupt("local header sizes overflow"))?;
        let data_end = data_at.checked_add(csize as usize).ok_or_else(|| corrupt("entry size overflows"))?;
        if data_end > cd_start {
            return Err(corrupt(format!("entry `{name}` runs into the central directory")));
        }
        let data = bytes.get(data_at..data_end).ok_or_else(|| corrupt(format!("entry `{name}` is truncated")))?;
        let body = match method {
            0 => {
                if csize != usize_ {
                    return Err(corrupt(format!("stored entry `{name}` has different packed and unpacked sizes")));
                }
                data.to_vec()
            }
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(data, usize_ as usize)
                .map_err(|e| corrupt(format!("entry `{name}` does not inflate: {:?}", e.status)))?,
            m => return Err(ZipError::Unsupported(format!("entry `{name}` uses compression method {m} (only stored and deflate)"))),
        };
        if body.len() != usize_ as usize {
            return Err(corrupt(format!("entry `{name}` inflates to {} bytes, the directory says {usize_}", body.len())));
        }
        if crc32(&body) != crc {
            return Err(corrupt(format!("entry `{name}` fails its CRC-32 check")));
        }
        if out.insert(name.to_string(), body).is_some() {
            return Err(ZipError::BadName { name: name.into(), why: "the archive holds two entries with this name".into() });
        }
    }
    Ok(out)
}

/// Whether `bytes` start like a ZIP archive.
pub fn is_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06")
}

/// Write `files` (name, contents) as a ZIP archive, deflated where that is smaller.
pub fn write(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>, ZipError> {
    if files.len() > MAX_ENTRIES {
        return Err(ZipError::TooLarge(format!("{} entries (at most {MAX_ENTRIES})", files.len())));
    }
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        check_entry_name(name).map_err(|why| ZipError::BadName { name: name.clone(), why })?;
        if data.len() as u64 > MAX_ENTRY_BYTES {
            return Err(ZipError::TooLarge(format!("entry `{name}` is {} bytes", data.len())));
        }
        let crc = crc32(data);
        let packed = miniz_oxide::deflate::compress_to_vec(data, 6);
        let (method, body): (u16, &[u8]) = if packed.len() < data.len() { (8, &packed) } else { (0, data) };
        let too_big = |_| ZipError::TooLarge("the archive is larger than 4 GiB".into());
        let offset = u32::try_from(out.len()).map_err(too_big)?;
        let csize = u32::try_from(body.len()).map_err(too_big)?;
        let usize_ = u32::try_from(data.len()).map_err(too_big)?;
        let nlen = u16::try_from(name.len()).map_err(|_| ZipError::BadName { name: name.clone(), why: "the name is too long".into() })?;
        // local file header (flag bit 11: the name is UTF-8)
        out.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        for v in [20u16, 1 << 11, method, DOS_TIME, DOS_DATE] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in [crc, csize, usize_] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(body);
        // central directory entry (made by: Unix, version 2.0; regular file rw-r--r--)
        central.extend_from_slice(&CENTRAL_SIG.to_le_bytes());
        for v in [(3u16 << 8) | 20, 20, 1 << 11, method, DOS_TIME, DOS_DATE] {
            central.extend_from_slice(&v.to_le_bytes());
        }
        for v in [crc, csize, usize_] {
            central.extend_from_slice(&v.to_le_bytes());
        }
        for v in [nlen, 0, 0, 0, 0] {
            central.extend_from_slice(&v.to_le_bytes());
        }
        central.extend_from_slice(&(0o100_644u32 << 16).to_le_bytes());
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let too_big = |_| ZipError::TooLarge("the archive is larger than 4 GiB".into());
    let cd_off = u32::try_from(out.len()).map_err(too_big)?;
    let cd_size = u32::try_from(central.len()).map_err(too_big)?;
    let n = u16::try_from(files.len()).map_err(|_| ZipError::TooLarge("too many entries".into()))?;
    out.extend_from_slice(&central);
    out.extend_from_slice(&END_SIG.to_le_bytes());
    for v in [0u16, 0, n, n] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<(String, Vec<u8>)> {
        vec![
            ("template.json".into(), br#"{"a": 1}"#.to_vec()),
            ("assets/loop.wav".into(), vec![7u8; 5000]),
            ("assets/noise.bin".into(), (0..300u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect()),
        ]
    }

    #[test]
    fn crc32_matches_the_check_value() {
        // the CRC-32 check value of "123456789" (IEEE)
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn write_then_read_round_trips_and_is_deterministic() {
        let a = write(&sample()).unwrap();
        assert_eq!(a, write(&sample()).unwrap(), "same input, same bytes");
        assert!(is_zip(&a));
        let back = read(&a).unwrap();
        assert_eq!(back.len(), 3);
        for (n, d) in sample() {
            assert_eq!(back.get(&n), Some(&d), "{n}");
        }
    }

    #[test]
    fn hostile_names_are_refused() {
        for bad in ["../evil", "/etc/passwd", "a/../../b", "C:/x", "a\\b", "a//b", "./a", "a\0b", ""] {
            assert!(check_entry_name(bad).is_err(), "{bad:?}");
        }
        let deep = vec!["d"; MAX_NAME_DEPTH + 1].join("/");
        assert!(check_entry_name(&deep).is_err());
        assert!(check_entry_name("assets/music/bed.wav").is_ok());
        assert!(write(&[("../x".into(), vec![1])]).is_err());
    }

    /// An archive whose entry name is patched after writing (the writer refuses such names).
    fn with_name(name: &[u8]) -> Vec<u8> {
        let placeholder = "x".repeat(name.len());
        let mut a = write(&[(placeholder.clone(), b"hello".to_vec())]).unwrap();
        let mut i = 0;
        while let Some(pos) = a[i..].windows(name.len()).position(|w| w == placeholder.as_bytes()) {
            a[i + pos..i + pos + name.len()].copy_from_slice(name);
            i += pos + name.len();
        }
        a
    }

    #[test]
    fn path_traversal_in_an_archive_is_refused() {
        for name in [&b"../../etc/x"[..], b"/abs/x", b"a\\..\\b", b"..", b"C:/win"] {
            let a = with_name(name);
            match read(&a) {
                Err(ZipError::BadName { .. }) => {}
                other => panic!("{:?}: {other:?}", String::from_utf8_lossy(name)),
            }
        }
    }

    #[test]
    fn mutations_never_panic() {
        let a = write(&sample()).unwrap();
        // truncations
        for n in 0..a.len() {
            let _ = read(&a[..n]);
        }
        // bit flips and byte overwrites everywhere
        for i in 0..a.len() {
            for v in [0x00u8, 0xFF, a[i] ^ 0x01, a[i] ^ 0x80] {
                let mut b = a.clone();
                b[i] = v;
                let r = std::panic::catch_unwind(|| read(&b));
                assert!(r.is_ok(), "panicked with byte {i} = {v:#x}");
            }
        }
    }

    #[test]
    fn a_zip_bomb_stops_at_the_cap() {
        // an entry claiming a small size but inflating to much more fails, it does not allocate
        let big = vec![0u8; 4 * 1024 * 1024];
        let mut a = write(&[("big.bin".into(), big)]).unwrap();
        // patch the declared uncompressed size (central directory) down to 1000
        let cd = a.windows(4).rposition(|w| w == CENTRAL_SIG.to_le_bytes()).unwrap();
        a[cd + 24..cd + 28].copy_from_slice(&1000u32.to_le_bytes());
        assert!(matches!(read(&a), Err(ZipError::Corrupt(_))));
        // and an entry declaring more than the cap is refused before inflating
        let cd = a.windows(4).rposition(|w| w == CENTRAL_SIG.to_le_bytes()).unwrap();
        a[cd + 24..cd + 28].copy_from_slice(&u32::MAX.wrapping_sub(1).to_le_bytes());
        assert!(matches!(read(&a), Err(ZipError::TooLarge(_))));
    }

    #[test]
    fn not_zips() {
        assert_eq!(read(b""), Err(ZipError::NotZip));
        assert_eq!(read(b"hello world, not a zip at all........"), Err(ZipError::NotZip));
        assert!(!is_zip(b"{\"format\""));
    }
}

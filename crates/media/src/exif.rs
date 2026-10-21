//! The few EXIF fields a photo carries that FilmCraft uses, read clean-room from the TIFF 6.0
//! structure EXIF is (CIPA DC-008, Exif 2.3): how to turn the picture upright (`Orientation`), when
//! it was taken (`DateTimeOriginal` and its `OffsetTimeOriginal`) and where (the GPS IFD's
//! latitude and longitude).
//!
//! The block comes from a JPEG's `APP1 "Exif\0\0"` segment ([`jpeg_exif`]), a HEIF `Exif` item
//! (`filmcraft-codecs`), or a TIFF file itself. Everything in it is hostile: offsets are checked
//! against the block, directories hold at most [`MAX_ENTRIES`] entries, and each directory is read
//! once (no loops through pointers), so a crafted block costs a bounded amount of work.

/// Most entries read from one image file directory.
pub const MAX_ENTRIES: usize = 512;
/// Most bytes of a text field kept.
const MAX_TEXT: usize = 64;
/// Most JPEG segments looked at before the image data.
const MAX_SEGMENTS: usize = 256;

/// What [`parse`] found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exif {
    /// `Orientation` (1 = upright … 8), when the block has one in range.
    pub orientation: Option<u16>,
    /// `DateTimeOriginal` (else `DateTimeDigitized`, else the main `DateTime`), as written:
    /// `"YYYY:MM:DD HH:MM:SS"` in the camera's local time.
    pub taken: Option<String>,
    /// `OffsetTimeOriginal` (else `OffsetTime`): the camera's zone, `"+04:00"`.
    pub offset: Option<String>,
    /// Where it was taken: latitude and longitude in degrees (north and east positive).
    pub gps: Option<(f64, f64)>,
}

impl Exif {
    /// When it was taken as ISO 8601 (`2022-11-02T19:39:53+04:00`, or without a zone when the
    /// camera wrote none), or `None` when the date is not one.
    pub fn taken_iso(&self) -> Option<String> {
        let t = self.taken.as_deref()?.trim();
        let b = t.as_bytes();
        let digits = |r: std::ops::Range<usize>| b.get(r).is_some_and(|s| s.iter().all(u8::is_ascii_digit));
        if b.len() < 19 || !digits(0..4) || !digits(5..7) || !digits(8..10) || !digits(11..13) || !digits(14..16) || !digits(17..19) {
            return None;
        }
        let (date, time) = (t.get(0..10)?.replace(':', "-"), t.get(11..19)?);
        if date.starts_with("0000") {
            return None;
        }
        let zone = self.offset.as_deref().map(str::trim).filter(|z| {
            let z = z.as_bytes();
            z.len() == 6 && matches!(z[0], b'+' | b'-') && z[3] == b':' && [1, 2, 4, 5].iter().all(|i| z[*i].is_ascii_digit())
        });
        Some(format!("{date}T{time}{}", zone.unwrap_or("")))
    }
}

/// The TIFF block of a JPEG's first `APP1 "Exif\0\0"` segment.
pub fn jpeg_exif(b: &[u8]) -> Option<&[u8]> {
    if !b.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut at = 2usize;
    for _ in 0..MAX_SEGMENTS {
        // markers may be padded with 0xFF bytes
        while b.get(at) == Some(&0xFF) && b.get(at + 1) == Some(&0xFF) {
            at += 1;
        }
        if b.get(at) != Some(&0xFF) {
            return None;
        }
        let marker = *b.get(at + 1)?;
        // start of scan, end of image: no metadata after them
        if marker == 0xDA || marker == 0xD9 {
            return None;
        }
        // markers without a length
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            at += 2;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([*b.get(at + 2)?, *b.get(at + 3)?]));
        if len < 2 {
            return None;
        }
        let body = b.get(at + 4..at + 2 + len)?;
        if marker == 0xE1 && body.starts_with(b"Exif\0\0") {
            return body.get(6..);
        }
        at = at.checked_add(2 + len)?;
    }
    None
}

/// The EXIF of a still image file's bytes: a JPEG's `APP1` block, or a TIFF read as one. `None`
/// for other formats (HEIF is read by `filmcraft-codecs`) or without EXIF.
pub fn of_file(b: &[u8]) -> Option<Exif> {
    if let Some(t) = jpeg_exif(b) {
        return parse(t);
    }
    if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") {
        return parse(b);
    }
    None
}

/// Byte order of the block.
#[derive(Clone, Copy)]
struct Tiff<'a> {
    b: &'a [u8],
    le: bool,
}

/// One directory entry: tag, type, count and the 4 bytes holding its value or offset.
#[derive(Clone, Copy)]
struct Entry {
    tag: u16,
    kind: u16,
    count: u32,
    value: [u8; 4],
}

impl<'a> Tiff<'a> {
    fn u16_at(&self, at: usize) -> Option<u16> {
        let s = self.b.get(at..at.checked_add(2)?)?;
        let a = [s[0], s[1]];
        Some(if self.le { u16::from_le_bytes(a) } else { u16::from_be_bytes(a) })
    }

    fn u32_at(&self, at: usize) -> Option<u32> {
        let s = self.b.get(at..at.checked_add(4)?)?;
        let a = [s[0], s[1], s[2], s[3]];
        Some(if self.le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    }

    fn u16_of(&self, a: [u8; 4]) -> u16 {
        if self.le { u16::from_le_bytes([a[0], a[1]]) } else { u16::from_be_bytes([a[0], a[1]]) }
    }

    fn u32_of(&self, a: [u8; 4]) -> u32 {
        if self.le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
    }

    /// The entries of the directory at `at` (at most [`MAX_ENTRIES`]).
    fn ifd(&self, at: usize) -> Vec<Entry> {
        let Some(n) = self.u16_at(at) else { return Vec::new() };
        (0..usize::from(n).min(MAX_ENTRIES))
            .map_while(|i| {
                let e = at.checked_add(2)?.checked_add(i.checked_mul(12)?)?;
                let s = self.b.get(e..e.checked_add(12)?)?;
                Some(Entry { tag: self.u16_at(e)?, kind: self.u16_at(e + 2)?, count: self.u32_at(e + 4)?, value: [s[8], s[9], s[10], s[11]] })
            })
            .collect()
    }

    /// The bytes of an entry's value: in the entry when they fit in 4, else at its offset.
    fn data(&self, e: &Entry) -> Option<&'a [u8]> {
        let unit: usize = match e.kind {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => return None,
        };
        let len = usize::try_from(e.count).ok()?.checked_mul(unit)?;
        if len <= 4 {
            // (borrowed from the block: find the entry's own bytes there)
            return None;
        }
        let off = usize::try_from(self.u32_of(e.value)).ok()?;
        self.b.get(off..off.checked_add(len)?)
    }

    /// An ASCII entry's text (up to its first NUL, at most [`MAX_TEXT`] bytes).
    fn text(&self, e: &Entry) -> Option<String> {
        if e.kind != 2 {
            return None;
        }
        let small;
        let bytes: &[u8] = if e.count <= 4 {
            small = e.value;
            small.get(..usize::try_from(e.count).ok()?)?
        } else {
            self.data(e)?
        };
        let end = bytes.iter().position(|c| *c == 0).unwrap_or(bytes.len()).min(MAX_TEXT);
        let s = String::from_utf8_lossy(bytes.get(..end)?).trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    /// A SHORT or LONG entry's first value.
    fn int(&self, e: &Entry) -> Option<u32> {
        match e.kind {
            3 => Some(u32::from(self.u16_of(e.value))),
            4 => Some(self.u32_of(e.value)),
            _ => None,
        }
    }

    /// Three RATIONALs (degrees, minutes, seconds) as degrees.
    fn degrees(&self, e: &Entry) -> Option<f64> {
        if e.kind != 5 || e.count != 3 {
            return None;
        }
        let d = self.data(e)?;
        let mut v = [0.0f64; 3];
        for (k, x) in v.iter_mut().enumerate() {
            let at = k * 8;
            let r = Tiff { b: d, le: self.le };
            let (n, den) = (r.u32_at(at)?, r.u32_at(at + 4)?);
            if den == 0 {
                return None;
            }
            *x = f64::from(n) / f64::from(den);
        }
        let deg = v[0] + v[1] / 60.0 + v[2] / 3600.0;
        deg.is_finite().then_some(deg)
    }
}

/// Read a TIFF-structured EXIF block (`II*\0` or `MM\0*` first). `None` when it is not one.
pub fn parse(b: &[u8]) -> Option<Exif> {
    let le = match b.get(0..4)? {
        [b'I', b'I', 42, 0] => true,
        [b'M', b'M', 0, 42] => false,
        _ => return None,
    };
    let t = Tiff { b, le };
    let ifd0 = usize::try_from(t.u32_at(4)?).ok()?;
    let mut out = Exif::default();
    let (mut exif_at, mut gps_at, mut plain_date) = (None, None, None);
    for e in t.ifd(ifd0) {
        match e.tag {
            0x0112 => out.orientation = t.int(&e).and_then(|v| u16::try_from(v).ok()).filter(|v| (1..=8).contains(v)),
            0x0132 => plain_date = t.text(&e),
            0x8769 => exif_at = t.int(&e).and_then(|v| usize::try_from(v).ok()),
            0x8825 => gps_at = t.int(&e).and_then(|v| usize::try_from(v).ok()),
            _ => {}
        }
    }
    let (mut original, mut digitized, mut offset_original, mut offset) = (None, None, None, None);
    // (each directory once: a pointer back to IFD0 or to itself reads nothing more)
    if let Some(at) = exif_at.filter(|a| *a != ifd0) {
        for e in t.ifd(at) {
            match e.tag {
                0x9003 => original = t.text(&e),
                0x9004 => digitized = t.text(&e),
                0x9011 => offset_original = t.text(&e),
                0x9010 => offset = t.text(&e),
                _ => {}
            }
        }
    }
    out.taken = original.or(digitized).or(plain_date);
    out.offset = offset_original.or(offset);
    if let Some(at) = gps_at.filter(|a| *a != ifd0 && Some(*a) != exif_at) {
        let (mut lat, mut lon, mut lat_ref, mut lon_ref) = (None, None, None, None);
        for e in t.ifd(at) {
            match e.tag {
                0x0001 => lat_ref = t.text(&e),
                0x0002 => lat = t.degrees(&e),
                0x0003 => lon_ref = t.text(&e),
                0x0004 => lon = t.degrees(&e),
                _ => {}
            }
        }
        if let (Some(la), Some(lo)) = (lat, lon) {
            let la = if lat_ref.as_deref().is_some_and(|r| r.eq_ignore_ascii_case("S")) { -la } else { la };
            let lo = if lon_ref.as_deref().is_some_and(|r| r.eq_ignore_ascii_case("W")) { -lo } else { lo };
            // (0, 0) is a phone that had no fix, not a place in the Gulf of Guinea
            if (-90.0..=90.0).contains(&la) && (-180.0..=180.0).contains(&lo) && (la != 0.0 || lo != 0.0) {
                out.gps = Some((la, lo));
            }
        }
    }
    Some(out)
}

/// Builds EXIF blocks for tests (here and in the crates that read photos).
#[doc(hidden)]
pub mod build {
    /// A little-endian TIFF block with `orientation`, `taken` (`"YYYY:MM:DD HH:MM:SS"`), `offset`
    /// (`"+04:00"`) and a GPS position (degrees; negative south and west), each optional.
    pub fn block(orientation: Option<u16>, taken: Option<&str>, offset: Option<&str>, gps: Option<(f64, f64)>) -> Vec<u8> {
        // layout: header (8) | IFD0 | Exif IFD | GPS IFD | data
        let mut ifd0: Vec<(u16, u16, u32, Vec<u8>)> = Vec::new();
        if let Some(o) = orientation {
            ifd0.push((0x0112, 3, 1, o.to_le_bytes().to_vec()));
        }
        let mut exif: Vec<(u16, u16, u32, Vec<u8>)> = Vec::new();
        let ascii = |s: &str| {
            let mut v = s.as_bytes().to_vec();
            v.push(0);
            v
        };
        if let Some(t) = taken {
            let v = ascii(t);
            exif.push((0x9003, 2, v.len() as u32, v));
        }
        if let Some(z) = offset {
            let v = ascii(z);
            exif.push((0x9011, 2, v.len() as u32, v));
        }
        let mut gpsd: Vec<(u16, u16, u32, Vec<u8>)> = Vec::new();
        if let Some((la, lo)) = gps {
            let dms = |x: f64| {
                let x = x.abs();
                let d = x.floor();
                let m = ((x - d) * 60.0).floor();
                let s = ((x - d) * 60.0 - m) * 60.0;
                let mut v = Vec::new();
                for (n, den) in [(d as u32, 1u32), (m as u32, 1), ((s * 1000.0).round() as u32, 1000)] {
                    v.extend(n.to_le_bytes());
                    v.extend(den.to_le_bytes());
                }
                v
            };
            gpsd.push((0x0001, 2, 2, ascii(if la < 0.0 { "S" } else { "N" })));
            gpsd.push((0x0002, 5, 3, dms(la)));
            gpsd.push((0x0003, 2, 2, ascii(if lo < 0.0 { "W" } else { "E" })));
            gpsd.push((0x0004, 5, 3, dms(lo)));
        }
        let ifd_len = |n: usize| 2 + 12 * n + 4;
        let extra0 = usize::from(!exif.is_empty()) + usize::from(!gpsd.is_empty());
        let at0 = 8usize;
        let at_exif = at0 + ifd_len(ifd0.len() + extra0);
        let at_gps = at_exif + if exif.is_empty() { 0 } else { ifd_len(exif.len()) };
        let mut data_at = at_gps + if gpsd.is_empty() { 0 } else { ifd_len(gpsd.len()) };
        let mut out = b"II*\0".to_vec();
        out.extend((at0 as u32).to_le_bytes());
        let mut data: Vec<u8> = Vec::new();
        let mut write_ifd = |out: &mut Vec<u8>, entries: &[(u16, u16, u32, Vec<u8>)], pointers: &[(u16, u32)]| {
            let mut all: Vec<(u16, u16, u32, Vec<u8>)> = entries.to_vec();
            for (tag, at) in pointers {
                all.push((*tag, 4, 1, at.to_le_bytes().to_vec()));
            }
            all.sort_by_key(|e| e.0);
            out.extend((all.len() as u16).to_le_bytes());
            for (tag, kind, count, bytes) in &all {
                out.extend(tag.to_le_bytes());
                out.extend(kind.to_le_bytes());
                out.extend(count.to_le_bytes());
                if bytes.len() <= 4 {
                    let mut v = bytes.clone();
                    v.resize(4, 0);
                    out.extend(v);
                } else {
                    out.extend((data_at as u32).to_le_bytes());
                    data.extend(bytes);
                    data_at += bytes.len();
                }
            }
            out.extend(0u32.to_le_bytes());
        };
        let mut pointers = Vec::new();
        if !exif.is_empty() {
            pointers.push((0x8769, at_exif as u32));
        }
        if !gpsd.is_empty() {
            pointers.push((0x8825, at_gps as u32));
        }
        write_ifd(&mut out, &ifd0, &pointers);
        if !exif.is_empty() {
            write_ifd(&mut out, &exif, &[]);
        }
        if !gpsd.is_empty() {
            write_ifd(&mut out, &gpsd, &[]);
        }
        out.extend(data);
        out
    }

    /// `jpeg` (a JPEG file) with the EXIF block `tiff` in an `APP1` segment right after its SOI.
    pub fn into_jpeg(jpeg: &[u8], tiff: &[u8]) -> Vec<u8> {
        let mut out = jpeg.get(..2).unwrap_or_default().to_vec();
        out.extend([0xFF, 0xE1]);
        out.extend(((tiff.len() + 8) as u16).to_be_bytes());
        out.extend(b"Exif\0\0");
        out.extend(tiff);
        out.extend(jpeg.get(2..).unwrap_or_default());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_orientation_dates_and_places() {
        let b = build::block(Some(6), Some("2022:11:02 19:39:53"), Some("+04:00"), Some((25.2048, -55.2708)));
        let e = parse(&b).unwrap();
        assert_eq!(e.orientation, Some(6));
        assert_eq!(e.taken.as_deref(), Some("2022:11:02 19:39:53"));
        assert_eq!(e.taken_iso().as_deref(), Some("2022-11-02T19:39:53+04:00"));
        let (la, lo) = e.gps.unwrap();
        assert!((la - 25.2048).abs() < 1e-5 && (lo + 55.2708).abs() < 1e-5, "{la} {lo}");
        // without a zone the time is the camera's own
        let e = parse(&build::block(None, Some("2020:01:05 08:00:00"), None, None)).unwrap();
        assert_eq!((e.orientation, e.taken_iso().as_deref(), e.gps), (None, Some("2020-01-05T08:00:00"), None));
        // a phone without a fix writes 0, 0
        assert_eq!(parse(&build::block(None, None, None, Some((0.0, 0.0)))).unwrap().gps, None);
        // an unset clock is no date
        assert_eq!(parse(&build::block(None, Some("0000:00:00 00:00:00"), None, None)).unwrap().taken_iso(), None);
    }

    #[test]
    fn finds_the_block_in_a_jpeg() {
        let tiff = build::block(Some(3), None, None, None);
        // a JPEG's start: SOI, an APP0, then our APP1, then SOS
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x4A, 0x46];
        jpeg.extend([0xFF, 0xDA, 0x00, 0x02]);
        let with = build::into_jpeg(&jpeg, &tiff);
        assert_eq!(of_file(&with).unwrap().orientation, Some(3));
        assert_eq!(of_file(&jpeg), None);
        assert_eq!(of_file(b"\x89PNG"), None);
    }

    #[test]
    fn hostile_blocks_never_panic() {
        let good = build::into_jpeg(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2], &build::block(Some(6), Some("2022:11:02 19:39:53"), Some("+04:00"), Some((10.0, 20.0))));
        // every truncation and many bit flips: an answer or none, never a panic
        for n in 0..good.len() {
            let _ = std::panic::catch_unwind(|| of_file(&good[..n])).expect("no panic on truncation");
        }
        for i in 0..good.len() {
            for bit in [0x01u8, 0x10, 0x80] {
                let mut b = good.clone();
                b[i] ^= bit;
                let _ = std::panic::catch_unwind(|| of_file(&b)).expect("no panic on a bit flip");
            }
        }
        // a directory claiming 65535 entries, offsets far past the end, pointers to itself
        let mut b = b"MM\0*\0\0\0\x08".to_vec();
        b.extend([0xFF, 0xFF]);
        b.extend([0x87, 0x69, 0, 4, 0, 0, 0, 1, 0, 0, 0, 8]);
        assert!(parse(&b).is_some_and(|e| e.taken.is_none()));
        let mut b = b"II*\0\xff\xff\xff\xff".to_vec();
        b.extend([0u8; 16]);
        assert_eq!(parse(&b), Some(Exif::default()));
        assert_eq!(parse(b"II*"), None);
    }
}

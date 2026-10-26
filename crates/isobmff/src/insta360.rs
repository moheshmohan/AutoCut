//! The trailer Insta360 cameras append to their files (`.insv` video, `.lrv` low-resolution
//! proxies, `.insp` photos), after the MP4 boxes (in the files we have, inside a top-level `inst`
//! box that runs to the end of the file).
//!
//! There is no public specification. The layout below was read from the files themselves
//! (Insta360 X5 footage, firmware 1.11), never from Insta360's software:
//!
//! - The file ends with 72 bytes: 32 zero bytes, the trailer's length (u32, little-endian,
//!   counted back from the end of the file), its version (u32 LE; 3) and the 32 ASCII characters
//!   `8db42d694ccc418790edff439fe026bf`.
//! - Just before them is the record index: its data, then its 6-byte header — a u16 id (0) and the
//!   data's length (u32), both little-endian. The index is a table of 10-byte entries: the record's
//!   id (u16), length (u32) and offset from the trailer's first byte (u32); all-zero entries pad it.
//! - Record `0x101` is the camera's metadata, a protocol buffer: 1 serial number, 2 camera model,
//!   3 firmware, 5 and 17 the lens calibration ("offset"), 7 the recording's start as the number
//!   yyyymmddhhmmss, 19 the recorded picture size of one lens ({1 width, 2 height}), 20 the frame
//!   rate, 22 the shooting mode, 27 the sensor's size, 54 and 56 fuller per-lens models.
//! - Record 2 is a thumbnail, record 3 a 1 kHz motion-sensor stream (a u64 microsecond
//!   timestamp and six offset-binary u16 values a sample), record 4 exposure times; the others
//!   are not described here.
//!
//! Every number is from a file, so every one is checked: lengths against the file, offsets against
//! the trailer, varints to ten bytes, strings capped. A file without a valid trailer simply has
//! none ([`read_trailer`] gives `None`).

use crate::source::ByteSource;

/// The magic at the very end of an Insta360 file.
pub const MAGIC: &[u8; 32] = b"8db42d694ccc418790edff439fe026bf";
/// Bytes of the footer: padding, length, version, magic.
const FOOTER: u64 = 72;
/// Most index entries read.
const MAX_RECORDS: usize = 256;
/// Largest metadata record read, bytes.
const MAX_INFO: u32 = 1 << 20;
/// Longest string kept from the metadata.
const MAX_STR: usize = 4096;

/// Record ids.
pub const RECORD_INFO: u16 = 0x101;
pub const RECORD_THUMBNAIL: u16 = 0x2;
pub const RECORD_MOTION: u16 = 0x3;
pub const RECORD_EXPOSURE: u16 = 0x4;

/// One record of the trailer: where its data is in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub id: u16,
    /// Absolute file offset of the data.
    pub offset: u64,
    pub len: u32,
}

/// The trailer of an Insta360 file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trailer {
    pub version: u32,
    /// Absolute file offset of the trailer's first byte.
    pub start: u64,
    pub records: Vec<Record>,
}

impl Trailer {
    pub fn record(&self, id: u16) -> Option<Record> {
        self.records.iter().copied().find(|r| r.id == id)
    }
}

/// The trailer at the end of `src`, if it has a valid one.
pub fn read_trailer(src: &dyn ByteSource) -> Option<Trailer> {
    let len = src.len();
    if len < FOOTER + 6 {
        return None;
    }
    let mut foot = [0u8; FOOTER as usize];
    src.read_at(len - FOOTER, &mut foot).ok()?;
    if foot.get(40..72)? != MAGIC.as_slice() {
        return None;
    }
    let u32le = |b: &[u8], at: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?)) };
    let total = u64::from(u32le(&foot, 32)?);
    let version = u32le(&foot, 36)?;
    if total < FOOTER + 6 || total > len {
        return None;
    }
    let start = len - total;
    let mut hdr = [0u8; 6];
    src.read_at(len - FOOTER - 6, &mut hdr).ok()?;
    let index_len = u64::from(u32le(&hdr, 2)?);
    let index_at = (len - FOOTER - 6).checked_sub(index_len).filter(|&a| a >= start)?;
    let n = usize::try_from(index_len / 10).ok()?.min(MAX_RECORDS);
    let mut index = vec![0u8; n * 10];
    src.read_at(index_at, &mut index).ok()?;
    let mut records = Vec::new();
    for e in index.as_chunks::<10>().0 {
        let id = u16::from_le_bytes([e[0], e[1]]);
        let rlen = u32le(e, 2)?;
        let off = u64::from(u32le(e, 6)?);
        if id == 0 && rlen == 0 {
            continue;
        }
        let offset = start.checked_add(off)?;
        // a record must lie inside the trailer, before the index
        if offset.checked_add(u64::from(rlen)).is_none_or(|end| end > index_at) {
            continue;
        }
        records.push(Record { id, offset, len: rlen });
    }
    Some(Trailer { version, start, records })
}

/// The bytes of one record (`None` when it is larger than `cap` or cannot be read).
pub fn read_record(src: &dyn ByteSource, r: Record, cap: u32) -> Option<Vec<u8>> {
    if r.len > cap {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(r.len).ok()?];
    src.read_at(r.offset, &mut buf).ok()?;
    Some(buf)
}

/// What the metadata record says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Info {
    pub serial: String,
    /// `Insta360 X5`, …
    pub camera: String,
    pub firmware: String,
    /// The lens calibration (field 5, else 17): `2_r_cx_cy_a_b_c_r_cx_cy_a_b_c_width_height_n`.
    pub offset: String,
    /// The fuller per-lens models (fields 54, 56, 111, 112, 53, 55), as written.
    pub lens_models: Vec<String>,
    /// The recording's start, `yyyymmddhhmmss` (local camera time).
    pub start: Option<u64>,
    /// Recorded picture size of one lens.
    pub lens_size: Option<(u32, u32)>,
    /// Frames a second, as the camera says it.
    pub frame_rate: Option<u32>,
    /// Shooting mode (`standard`, …).
    pub mode: String,
}

/// Reads the metadata record (`None`: no trailer, or no readable metadata).
pub fn read_info(src: &dyn ByteSource) -> Option<Info> {
    let t = read_trailer(src)?;
    let bytes = read_record(src, t.record(RECORD_INFO)?, MAX_INFO)?;
    Some(parse_info(&bytes))
}

/// Parses the metadata record's protocol buffer (unknown fields are skipped; a damaged buffer
/// gives what was read before the damage).
pub fn parse_info(b: &[u8]) -> Info {
    let mut info = Info::default();
    let mut offset17 = String::new();
    for (field, value) in fields(b) {
        match (field, value) {
            (1, Value::Bytes(v)) => info.serial = text(v),
            (2, Value::Bytes(v)) => info.camera = text(v),
            (3, Value::Bytes(v)) => info.firmware = text(v),
            (5, Value::Bytes(v)) => info.offset = text(v),
            (17, Value::Bytes(v)) => offset17 = text(v),
            (7, Value::Varint(v)) => info.start = Some(v),
            (19, Value::Bytes(v)) => {
                let mut w = None;
                let mut h = None;
                for (f, x) in fields(v) {
                    match (f, x) {
                        (1, Value::Varint(x)) => w = u32::try_from(x).ok(),
                        (2, Value::Varint(x)) => h = u32::try_from(x).ok(),
                        _ => {}
                    }
                }
                info.lens_size = w.zip(h);
            }
            (20, Value::Varint(v)) => info.frame_rate = u32::try_from(v).ok(),
            (22, Value::Bytes(v)) => info.mode = text(v),
            (53 | 54 | 55 | 56 | 111 | 112, Value::Bytes(v)) => {
                let s = text(v);
                if !s.is_empty() && info.lens_models.len() < 16 && !info.lens_models.contains(&s) {
                    info.lens_models.push(s);
                }
            }
            _ => {}
        }
    }
    if info.offset.is_empty() {
        info.offset = offset17;
    }
    info
}

fn text(v: &[u8]) -> String {
    let v = v.get(..v.len().min(MAX_STR)).unwrap_or_default();
    String::from_utf8_lossy(v).chars().filter(|c| !c.is_control()).collect()
}

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

/// The top-level fields of a protocol buffer, until the end or the first damaged one.
fn fields(b: &[u8]) -> impl Iterator<Item = (u64, Value<'_>)> {
    let mut i = 0usize;
    std::iter::from_fn(move || {
        let key = varint(b, &mut i)?;
        let field = key >> 3;
        let value = match key & 7 {
            0 => Value::Varint(varint(b, &mut i)?),
            1 => {
                i = i.checked_add(8).filter(|&e| e <= b.len())?;
                Value::Fixed
            }
            5 => {
                i = i.checked_add(4).filter(|&e| e <= b.len())?;
                Value::Fixed
            }
            2 => {
                let n = usize::try_from(varint(b, &mut i)?).ok()?;
                let end = i.checked_add(n).filter(|&e| e <= b.len())?;
                let v = b.get(i..end)?;
                i = end;
                Value::Bytes(v)
            }
            _ => return None,
        };
        Some((field, value))
    })
}

fn varint(b: &[u8], i: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for k in 0..10 {
        let c = *b.get(*i)?;
        *i += 1;
        v |= u64::from(c & 0x7f) << (7 * k);
        if c < 0x80 {
            return Some(v);
        }
    }
    None
}

/// One lens of the `offset` calibration, in the sensor's pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensOffset {
    /// Image-circle radius.
    pub radius: f64,
    /// Centre, in a canvas of `canvas` holding every lens side by side.
    pub cx: f64,
    pub cy: f64,
    /// The three angles that follow, degrees, as written.
    pub angles: [f64; 3],
}

/// The `offset` calibration parsed: the lenses and the canvas they are measured in.
#[derive(Clone, Debug, PartialEq)]
pub struct Offset {
    pub lenses: Vec<LensOffset>,
    /// Width and height of the canvas (every lens's square side by side).
    pub canvas: (f64, f64),
}

/// Parses `2_r_cx_cy_a_b_c_r_cx_cy_a_b_c_width_height[_n]`. `None` when it does not have that
/// shape (another camera's format) or a value is not a finite number.
pub fn parse_offset(s: &str) -> Option<Offset> {
    let v: Vec<f64> = s.split('_').map(|p| p.trim().parse::<f64>().ok().filter(|x| x.is_finite())).collect::<Option<Vec<_>>>()?;
    let n = *v.first()? as usize;
    if !(1..=4).contains(&n) || v.first().is_some_and(|f| f.fract() != 0.0) {
        return None;
    }
    let lens = |k: usize| -> Option<LensOffset> {
        let b = 1 + k * 6;
        Some(LensOffset { radius: *v.get(b)?, cx: *v.get(b + 1)?, cy: *v.get(b + 2)?, angles: [*v.get(b + 3)?, *v.get(b + 4)?, *v.get(b + 5)?] })
    };
    let lenses = (0..n).map(lens).collect::<Option<Vec<_>>>()?;
    let w = *v.get(1 + n * 6)?;
    let h = *v.get(2 + n * 6)?;
    if w <= 0.0 || h <= 0.0 || lenses.iter().any(|l| l.radius <= 0.0) {
        return None;
    }
    Some(Offset { lenses, canvas: (w, h) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pb_str(field: u64, s: &str) -> Vec<u8> {
        let mut v = vec![(field << 3 | 2) as u8];
        v.push(s.len() as u8);
        v.extend_from_slice(s.as_bytes());
        v
    }

    /// A file of `body` with a trailer holding `records` (id, data).
    fn file_with(body: &[u8], records: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut f = body.to_vec();
        let start = f.len();
        let mut index = Vec::new();
        for (id, data) in records {
            let off = (f.len() - start) as u32;
            f.extend_from_slice(data);
            // each record is followed by its own header, as the camera writes them
            f.extend_from_slice(&id.to_le_bytes());
            f.extend_from_slice(&(data.len() as u32).to_le_bytes());
            index.extend_from_slice(&id.to_le_bytes());
            index.extend_from_slice(&(data.len() as u32).to_le_bytes());
            index.extend_from_slice(&off.to_le_bytes());
        }
        index.extend_from_slice(&[0u8; 20]);
        f.extend_from_slice(&index);
        f.extend_from_slice(&0u16.to_le_bytes());
        f.extend_from_slice(&(index.len() as u32).to_le_bytes());
        f.extend_from_slice(&[0u8; 32]);
        let total = (f.len() - start + 40) as u32;
        f.extend_from_slice(&total.to_le_bytes());
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(MAGIC);
        f
    }

    const OFFSET: &str = "2_2657.856_2681.350_2673.940_0.249_-0.624_89.769_2655.921_8091.830_2669.660_-0.436_-0.465_90.327_10752_5376_1137";

    #[test]
    fn reads_the_trailer_and_metadata() {
        let mut info = pb_str(1, "SERIAL1");
        info.extend(pb_str(2, "Insta360 X5"));
        info.extend(pb_str(3, "v1.11.10_build1"));
        info.extend(pb_str(5, OFFSET));
        info.extend([7 << 3, 0x93, 0xc4, 0x94, 0xd8, 0x9f, 0xd1, 0x04]);
        // fields 19 and 20: two-byte keys
        info.extend([0x9a, 0x01, 6, 0x08, 0xc0, 0x16, 0x10, 0xc0, 0x16]);
        info.extend([0xa0, 0x01, 30]);
        let f = file_with(b"ftyp....mdat....", &[(RECORD_THUMBNAIL, vec![7; 33]), (RECORD_INFO, info)]);
        let t = read_trailer(&f.as_slice()).unwrap();
        assert_eq!(t.version, 3);
        assert_eq!(t.start, 16);
        assert_eq!(t.records.len(), 2);
        let i = read_info(&f.as_slice()).unwrap();
        assert_eq!(i.serial, "SERIAL1");
        assert_eq!(i.camera, "Insta360 X5");
        assert_eq!(i.firmware, "v1.11.10_build1");
        assert_eq!(i.offset, OFFSET);
        assert_eq!(i.lens_size, Some((2880, 2880)));
        assert_eq!(i.frame_rate, Some(30));
        assert!(i.start.is_some());
        let thumb = read_record(&f.as_slice(), t.record(RECORD_THUMBNAIL).unwrap(), 1024).unwrap();
        assert_eq!(thumb, vec![7; 33]);
    }

    #[test]
    fn parses_the_offset_string() {
        let o = parse_offset(OFFSET).unwrap();
        assert_eq!(o.lenses.len(), 2);
        assert_eq!(o.canvas, (10752.0, 5376.0));
        assert_eq!(o.lenses[1].cx, 8091.830);
        assert_eq!(o.lenses[0].angles, [0.249, -0.624, 89.769]);
        assert!(parse_offset("").is_none());
        assert!(parse_offset("2_1_2_3").is_none());
        assert!(parse_offset("2.5_1_2_3_4_5_6_1_2_3_4_5_6_7_8").is_none());
        assert!(parse_offset("2_nan_2_3_4_5_6_1_2_3_4_5_6_7_8").is_none());
        assert!(parse_offset("2_0_2_3_4_5_6_1_2_3_4_5_6_7_8").is_none());
    }

    #[test]
    fn files_without_a_trailer_have_none() {
        assert!(read_trailer(&b"".as_slice()).is_none());
        assert!(read_trailer(&[0u8; 500].as_slice()).is_none());
        let mut f = vec![0u8; 100];
        f.extend_from_slice(MAGIC);
        assert!(read_trailer(&f.as_slice()).is_none());
    }

    #[test]
    fn damaged_trailers_never_panic() {
        let info = pb_str(2, "Insta360 X5");
        let f = file_with(b"body", &[(RECORD_INFO, info)]);
        for n in 0..f.len() {
            let _ = read_trailer(&&f[n..]);
            let _ = read_info(&&f[n..]);
            let mut g = f.clone();
            g[n] ^= 0x5a;
            let _ = read_info(&g.as_slice());
            for bit in [0x80u8, 0xff] {
                let mut g = f.clone();
                g[n] = bit;
                let _ = read_info(&g.as_slice());
            }
        }
        // a varint that never ends, a length past the end, an unknown wire type
        let _ = parse_info(&[0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
        let _ = parse_info(&[0x0a, 0x7f, 1, 2]);
        let _ = parse_info(&[0x0b]);
    }
}

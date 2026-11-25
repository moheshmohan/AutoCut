//! OpenEXR stills (and the frames of EXR image sequences), read from the public OpenEXR file layout:
//! single-part and multi-part files of scan lines or tiles, with no, RLE, ZIPS (one line per
//! chunk), ZIP or PXR24 (16 lines) compression, half, float or uint channels.
//!
//! The picture shown is the file's main picture, as compositing applications show it: the
//! unprefixed R, G, B (and A) of the first part that has them, else the unprefixed Y (and A),
//! shown grey. A file of named layers only (`ViewLayer.Combined.R`) shows its richest colour
//! layer, and a part of one channel (a depth map) shows it grey ([`channels`]). Missing alpha is
//! opaque. EXR values are scene-linear with premultiplied alpha, which is exactly the compositor's
//! working format, so the frame keeps them as 32-bit floats (values above 1.0 survive into effects
//! and colour work).
//!
//! Only the chosen part is read: the headers of every part are listed (at most [`MAX_PARTS`], each
//! with at most [`MAX_ATTRIBUTES`] attributes), the offset tables of the parts before it are
//! skipped by their `chunkCount`, and other parts' chunks (deep ones too) are never touched. A
//! tiled part is read at its full resolution (level 0 of a mip-map or rip-map), each of its tiles
//! exactly once. Every chunk's position, size and coordinates are checked before the picture is
//! allocated, so a small file cannot claim a large picture.
//!
//! The display window is the frame; pixels of the data window outside it are cut, and the parts of
//! it the data window does not cover are transparent. Deep parts, chroma subsampled channels and
//! the PIZ, B44 and DWA compressions (whose bitstreams the public file layout does not specify)
//! are refused with a clear error.

mod channels;
mod inflate;

use std::borrow::Cow;
use std::sync::Arc;

use channels::ColorLayout;
use filmcraft_color::{ColorInfo, Transfer};
use filmcraft_frame::VideoFrame;
use filmcraft_time::FrameRate;
use rayon::prelude::*;

use super::{DEFAULT_STILL_DURATION, StillSource};
use crate::{MediaError, MediaInfo, MediaKind, Result, VideoStreamInfo};

/// Magic number at the start of every OpenEXR file (20000630 little-endian).
const MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];

/// Largest picture we decode (each pixel takes 16 bytes as RGBA floats).
const MAX_PIXELS: u64 = 1 << 26;
/// Maximum memory retained while decoding the data window and, when different,
/// the display window. Both buffers coexist during placement.
/// 2 GiB: an 8K RGBA f32 plate is 566 MB and the data and display windows may both be held.
const MAX_DECODED_BYTES: usize = 2 << 30;
/// Most attributes / channels we read from a header, and parts from a file.
const MAX_ATTRIBUTES: usize = 4096;
const MAX_CHANNELS: usize = 1024;
const MAX_PARTS: usize = 1024;
/// Largest uncompressed chunk (scan lines or a tile × every channel) accepted. A real chunk is a
/// few MB (16 lines of a 16K-wide, 8-channel float image is 8 MB); a hostile header naming a
/// thousand channels on a 64M-pixel-wide line would otherwise ask for hundreds of GB.
const MAX_CHUNK_BYTES: usize = 256 << 20;
/// Most bytes one byte of RLE data expands to: a two-byte run is at most 128 bytes.
const MAX_RLE_EXPANSION: usize = 64;
/// Most bytes one byte of a zlib stream inflates to: deflate's longest match (258 bytes) takes at
/// least two bits.
const MAX_ZLIB_EXPANSION: usize = 1032;

/// Version field flags: a single-part tiled file, one with long names, deep data, many parts.
const FLAG_TILED: u32 = 0x200;
const FLAG_DEEP: u32 = 0x800;
const FLAG_MULTIPART: u32 = 0x1000;

/// Whether `b` starts like an OpenEXR file.
pub fn sniff(b: &[u8]) -> bool {
    b.starts_with(&MAGIC)
}

/// The colour of EXR frames: scene-linear light, Rec.709 primaries.
const LINEAR_709: ColorInfo = ColorInfo { transfer: Transfer::Linear, ..ColorInfo::SRGB_FULL };

/// A decoded EXR picture.
pub struct Decoded {
    /// Premultiplied linear RGBA floats, display-window sized.
    pub frame: VideoFrame,
    /// Whether any pixel is not fully opaque.
    pub has_alpha: bool,
    /// For the media info, e.g. "RGBA 16-bit float".
    pub pixel_format: String,
}

/// Open an EXR file as a still.
pub fn still(name: &str, bytes: &[u8]) -> Result<StillSource> {
    let d = decode(name, bytes)?;
    let info = MediaInfo {
        name: name.to_string(),
        kind: MediaKind::Still,
        duration: DEFAULT_STILL_DURATION,
        video: Some(VideoStreamInfo {
            width: d.frame.width,
            height: d.frame.height,
            frame_rate: FrameRate::FPS_29_97,
            par: (1, 1),
            codec: "EXR".into(),
            pixel_format: d.pixel_format,
            color: LINEAR_709,
            has_alpha: d.has_alpha,
            bitrate: None,
            hdr: None,
        }),
        audio_streams: Vec::new(),
        container: "Image".into(),
        start_timecode: None,
        file_size: Some(bytes.len() as u64),
    };
    Ok(StillSource { info, frame: Arc::new(d.frame) })
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PixelType {
    Half,
    Uint,
    Float,
}

impl PixelType {
    fn size(self) -> usize {
        match self {
            PixelType::Half => 2,
            PixelType::Uint | PixelType::Float => 4,
        }
    }
}

/// Where a channel's samples go.
#[derive(Clone, Copy)]
enum Target {
    Slot(usize),
    /// Luminance: R, G and B.
    Grey,
    Skip,
}

struct Channel {
    name: String,
    ty: PixelType,
    /// Horizontal and vertical sampling: (1, 1) for a channel at full resolution.
    sampling: (u32, u32),
}

/// An inclusive pixel rectangle (`box2i`).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Window {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Window {
    fn size(self) -> Option<(usize, usize)> {
        let w = i64::from(self.x1) - i64::from(self.x0) + 1;
        let h = i64::from(self.y1) - i64::from(self.y0) + 1;
        if w <= 0 || h <= 0 || (w as u64).saturating_mul(h as u64) > MAX_PIXELS {
            return None;
        }
        Some((w as usize, h as usize))
    }
}

#[derive(Clone, Copy)]
enum Compression {
    None,
    Rle,
    Zip,
    /// Lossy 24-bit floats: deflated byte planes of per-line differences.
    Pxr24,
}

impl Compression {
    /// What compression `code` is, or the name of one this reader does not decode.
    fn of(code: u8) -> std::result::Result<Compression, &'static str> {
        match code {
            0 => Ok(Compression::None),
            1 => Ok(Compression::Rle),
            2 | 3 => Ok(Compression::Zip),
            5 => Ok(Compression::Pxr24),
            _ => {
                Err(["PIZ", "PXR24", "B44", "B44A", "DWAA", "DWAB", "HTJ2K256", "HTJ2K32"].get(usize::from(code).wrapping_sub(4)).copied().unwrap_or("unknown"))
            }
        }
    }

    fn lines_per_chunk(self, code: u8) -> usize {
        match self {
            Compression::Zip if code == 3 => 16,
            Compression::Pxr24 => 16,
            _ => 1,
        }
    }
}

/// How a part stores its pixels.
#[derive(Clone, PartialEq, Eq)]
enum Storage {
    ScanLines,
    /// Tiles of this size; only the full-resolution level is read.
    Tiles {
        width: usize,
        height: usize,
    },
    /// Deep data, or a part type this reader does not know: never shown.
    Unreadable(String),
}

/// One part's header.
struct Part {
    /// The `name` attribute (multi-part files).
    name: Option<String>,
    storage: Storage,
    channels: Vec<Channel>,
    code: u8,
    data: Window,
    display: Window,
    /// The `chunkCount` attribute (required in multi-part files).
    chunk_count: Option<usize>,
}

/// The headers of a file.
struct Headers {
    parts: Vec<Part>,
    multipart: bool,
    /// Where the offset tables start: just after the headers.
    tables: usize,
}

fn le32(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn le64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from(le32(b, at)?) | u64::from(le32(b, at.checked_add(4)?)?) << 32)
}

/// A NUL-terminated string of at most 255 bytes at `*pos`; advances past the NUL.
fn cstr<'a>(b: &'a [u8], pos: &mut usize) -> Option<&'a str> {
    let rest = b.get(*pos..)?;
    let n = rest.iter().take(256).position(|&c| c == 0)?;
    *pos += n + 1;
    std::str::from_utf8(rest.get(..n)?).ok()
}

fn window(v: &[u8]) -> Option<Window> {
    let g = |i: usize| le32(v, i * 4).map(|x| x as i32);
    Some(Window { x0: g(0)?, y0: g(1)?, x1: g(2)?, y1: g(3)? })
}

fn channels(v: &[u8]) -> Result<Vec<Channel>> {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        let name = cstr(v, &mut pos).ok_or_else(|| MediaError::Decode("EXR channel list".into()))?;
        if name.is_empty() {
            return Ok(out);
        }
        let (Some(ty), Some(xs), Some(ys)) = (le32(v, pos), le32(v, pos + 8), le32(v, pos + 12)) else {
            return Err(MediaError::Decode("EXR channel list".into()));
        };
        pos += 16;
        let ty = match ty {
            0 => PixelType::Uint,
            1 => PixelType::Half,
            2 => PixelType::Float,
            _ => return Err(MediaError::Decode(format!("EXR channel {name}: pixel type {ty}"))),
        };
        if out.len() >= MAX_CHANNELS {
            return Err(MediaError::Decode("EXR has too many channels".into()));
        }
        out.push(Channel { name: name.to_string(), ty, sampling: (xs, ys) });
    }
}

/// The headers of every part: one for a single-part file, a list ending in an empty header for
/// a multi-part one.
fn headers(name: &str, b: &[u8]) -> Result<Headers> {
    let bad = |what: &str| MediaError::Decode(format!("{name}: EXR {what}"));
    if !sniff(b) {
        return Err(bad("magic number"));
    }
    let version = le32(b, 4).ok_or_else(|| bad("version"))?;
    if version & 0xFF != 2 {
        return Err(MediaError::Unsupported(format!("{name}: OpenEXR version {}", version & 0xFF)));
    }
    let multipart = version & FLAG_MULTIPART != 0;
    let mut pos = 8;
    let mut parts = Vec::new();
    loop {
        if multipart && b.get(pos) == Some(&0) {
            pos += 1;
            break;
        }
        if parts.len() >= MAX_PARTS {
            return Err(MediaError::Unsupported(format!("{name}: EXR has more than {MAX_PARTS} parts")));
        }
        let index = parts.len();
        parts.push(part(b, &mut pos, if multipart { None } else { Some(version) }).map_err(|what| bad(&format!("part {index}: {what}")))?);
        if !multipart {
            break;
        }
    }
    if parts.is_empty() {
        return Err(bad("file has no parts"));
    }
    Ok(Headers { parts, multipart, tables: pos })
}

/// One header at `*pos`; `single` holds the version field of a single-part file (its flags say
/// whether the part is tiled or deep), `None` in a multi-part file (the `type` attribute says).
fn part(b: &[u8], pos: &mut usize, single: Option<u32>) -> std::result::Result<Part, String> {
    let (mut chans, mut code, mut data, mut display, mut tiles, mut kind, mut name, mut chunk_count) = (None, None, None, None, None, None, None, None);
    for _ in 0..MAX_ATTRIBUTES {
        let attr = cstr(b, pos).ok_or("header")?;
        if attr.is_empty() {
            let (Some(channels), Some(code), Some(data)) = (chans, code, data) else {
                return Err("header lacks channels, compression or dataWindow".into());
            };
            let tiled = || -> std::result::Result<Storage, String> {
                let (width, height) = tiles.ok_or("a tiled part without its tile description")?;
                Ok(Storage::Tiles { width, height })
            };
            let deep = matches!(kind.as_deref(), Some("deepscanline" | "deeptile"));
            let storage = match (single, kind.as_deref()) {
                (Some(version), _) if version & FLAG_DEEP != 0 || deep => Storage::Unreadable("deep".into()),
                (Some(version), _) if version & FLAG_TILED != 0 => tiled()?,
                (Some(_), _) | (None, Some("scanlineimage")) => Storage::ScanLines,
                (None, Some("tiledimage")) => tiled()?,
                (None, _) if deep => Storage::Unreadable("deep".into()),
                (None, Some(other)) => Storage::Unreadable(format!("`{other}`")),
                (None, None) if tiles.is_some() => tiled()?,
                (None, None) => Storage::ScanLines,
            };
            return Ok(Part { name, storage, channels, code, data, display: display.unwrap_or(data), chunk_count });
        }
        let ty = cstr(b, pos).ok_or("header")?;
        let size = le32(b, *pos).ok_or("header")? as usize;
        *pos += 4;
        let value = b.get(*pos..pos.saturating_add(size)).ok_or("header")?;
        *pos += size;
        let text = || std::str::from_utf8(value).ok().map(str::to_string);
        match (attr, ty) {
            ("channels", "chlist") => chans = Some(channels(value).map_err(|e| e.to_string())?),
            ("compression", "compression") => code = value.first().copied(),
            ("dataWindow", "box2i") => data = Some(window(value).ok_or("dataWindow")?),
            ("displayWindow", "box2i") => display = Some(window(value).ok_or("displayWindow")?),
            ("tiles", "tiledesc") => {
                // x size, y size, then the level and rounding modes; every level shares the size
                let (Some(w), Some(h), Some(&mode)) = (le32(value, 0), le32(value, 4), value.get(8)) else {
                    return Err("tile description".into());
                };
                if w == 0 || h == 0 || mode & 0x0F > 2 {
                    return Err(format!("tile description {w} × {h}, mode {mode}"));
                }
                tiles = Some((w as usize, h as usize));
            }
            ("type", "string") => kind = text(),
            ("name", "string") => name = text(),
            ("chunkCount", "int") => chunk_count = le32(value, 0).and_then(|n| usize::try_from(n as i32).ok()),
            _ => {}
        }
    }
    Err("header has too many attributes".into())
}
/// IEEE half → f32.
fn half(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let man = u32::from(h & 0x3FF);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalise the mantissa
            let mut e = 113u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | e << 23 | (m & 0x3FF) << 13
        }
        (31, _) => sign | 0x7F80_0000 | man << 13,
        _ => sign | (exp + 112) << 23 | man << 13,
    };
    f32::from_bits(bits)
}

/// A sample as a finite float (NaN → 0, infinities → the half range).
fn sample(ty: PixelType, s: &[u8]) -> f32 {
    let v = match (ty, s) {
        (PixelType::Half, [a, b, ..]) => half(u16::from_le_bytes([*a, *b])),
        (PixelType::Float, [a, b, c, d, ..]) => f32::from_le_bytes([*a, *b, *c, *d]),
        (PixelType::Uint, [a, b, c, d, ..]) => u32::from_le_bytes([*a, *b, *c, *d]) as f32,
        _ => 0.0,
    };
    if v.is_nan() { 0.0 } else { v.clamp(-65504.0, 65504.0) }
}

/// Undo the RLE / ZIP preprocessing: byte differences, then the split into even and odd bytes.
fn unpredict(mut t: Vec<u8>) -> Vec<u8> {
    let mut prev = 0u8;
    for (i, b) in t.iter_mut().enumerate() {
        if i > 0 {
            *b = prev.wrapping_add(*b).wrapping_sub(128);
        }
        prev = *b;
    }
    let (even, odd) = t.split_at(t.len().div_ceil(2));
    let mut out = Vec::with_capacity(t.len());
    for (i, &e) in even.iter().enumerate() {
        out.push(e);
        if let Some(&o) = odd.get(i) {
            out.push(o);
        }
    }
    out
}

/// OpenEXR run-length decoding into exactly `expected` bytes.
fn unrle(src: &[u8], expected: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(expected).ok()?;
    let mut i = 0;
    while let Some(&c) = src.get(i) {
        let c = c as i8;
        i += 1;
        if c < 0 {
            let n = usize::from(c.unsigned_abs());
            out.extend_from_slice(src.get(i..i + n)?);
            i += n;
        } else {
            let n = c as usize + 1;
            let b = *src.get(i)?;
            i += 1;
            out.resize(out.len() + n, b);
        }
        if out.len() > expected {
            return None;
        }
    }
    (out.len() == expected).then_some(out)
}

/// Bytes of one sample in PXR24's byte planes (a float keeps its top 24 bits only).
fn plane_count(ty: PixelType) -> usize {
    match ty {
        PixelType::Half => 2,
        PixelType::Float => 3,
        PixelType::Uint => 4,
    }
}

/// Undo PXR24: per scan line and channel, the samples' differences (to the previous sample of the
/// line, modulo 2^16 / 2^24 / 2^32) are stored as byte planes, most significant first. Returns the
/// lines in the plain layout (little-endian samples).
fn unpxr24(planes: &[u8], channels: &[Channel], width: usize, lines: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(planes.len().checked_mul(2)?);
    let mut at = 0usize;
    for _ in 0..lines {
        for c in channels {
            let n = plane_count(c.ty);
            let len = n.checked_mul(width)?;
            let plane = planes.get(at..at.checked_add(len)?)?;
            at += len;
            let mut acc = 0u32;
            for x in 0..width {
                let mut d = 0u32;
                for k in 0..n {
                    d = d << 8 | u32::from(*plane.get(k * width + x)?);
                }
                acc = acc.wrapping_add(d);
                match c.ty {
                    PixelType::Half => out.extend_from_slice(&(acc as u16).to_le_bytes()),
                    PixelType::Float => out.extend_from_slice(&((acc & 0xFF_FFFF) << 8).to_le_bytes()),
                    PixelType::Uint => out.extend_from_slice(&acc.to_le_bytes()),
                }
            }
        }
    }
    (at == planes.len()).then_some(out)
}

/// A part's channels laid out in a block `width` samples wide: each channel's byte offset within a
/// line, the line's size and its PXR24 byte planes.
struct Lines {
    offsets: Vec<usize>,
    line: usize,
    pxr24_line: usize,
}

impl Lines {
    fn of(channels: &[Channel], width: usize) -> Option<Lines> {
        let (mut line, mut pxr24_line) = (0usize, 0usize);
        let mut offsets = Vec::with_capacity(channels.len());
        for c in channels {
            offsets.push(line);
            line = line.checked_add(c.ty.size().checked_mul(width)?)?;
            pxr24_line = pxr24_line.checked_add(plane_count(c.ty).checked_mul(width)?)?;
        }
        Some(Lines { offsets, line, pxr24_line })
    }
}

/// Where one block's (a scan-line chunk's or a tile's) data is, and its shape.
#[derive(Clone, Copy)]
struct Block<'a> {
    data: &'a [u8],
    /// Its samples a line, and its lines.
    width: usize,
    lines: usize,
}

/// Whether a block of `size` bytes can hold `lines` lines of `l`: stored raw its size is the
/// geometry's; compressed, it must be long enough to expand to them, or a file of a few hundred
/// bytes could have the whole picture allocated (and, without alpha, written) before its first
/// chunk fails to decode.
fn plausible(compression: Compression, size: usize, l: &Lines, lines: usize) -> std::result::Result<(), &'static str> {
    let expected = l.line.checked_mul(lines).ok_or("size")?;
    let (need, most) = match compression {
        _ if size == expected => return Ok(()),
        Compression::None => return Err("has the wrong size"),
        Compression::Rle => (expected, size.saturating_mul(MAX_RLE_EXPANSION)),
        Compression::Zip => (expected, size.saturating_mul(MAX_ZLIB_EXPANSION)),
        Compression::Pxr24 => (l.pxr24_line.checked_mul(lines).ok_or("size")?, size.saturating_mul(MAX_ZLIB_EXPANSION)),
    };
    if need > most { Err("is too short for its lines") } else { Ok(()) }
}

/// A block's lines in the plain layout (every channel's samples of a line, line after line).
fn unpack<'a>(compression: Compression, block: Block<'a>, channels: &[Channel], l: &Lines) -> std::result::Result<Cow<'a, [u8]>, String> {
    let expected = l.line.checked_mul(block.lines).ok_or("size")?;
    if block.data.len() == expected {
        // stored raw when compression would not shrink it
        return Ok(block.data.into());
    }
    Ok(match compression {
        Compression::None => return Err("has the wrong size".into()),
        Compression::Rle => unpredict(unrle(block.data, expected).ok_or("bad RLE data")?).into(),
        Compression::Zip => unpredict(inflate::zlib(block.data, expected)?).into(),
        Compression::Pxr24 => {
            let planes = l.pxr24_line.checked_mul(block.lines).ok_or("size")?;
            let inflated = inflate::zlib(block.data, planes)?;
            unpxr24(&inflated, channels, block.width, block.lines).ok_or("bad PXR24 data")?.into()
        }
    })
}

/// Write a block's decoded lines (`raw`) into `rows` (`row` floats each), from pixel `x0` of each.
fn place(raw: &[u8], block: Block, l: &Lines, channels: &[Channel], targets: &[Target], alpha: bool, rows: &mut [f32], row: usize, x0: usize) {
    for (k, out) in rows.chunks_exact_mut(row).take(block.lines).enumerate() {
        let Some(out) = out.get_mut(x0 * 4..(x0 + block.width) * 4) else { continue };
        // no alpha channel: opaque, filled once the block's data is in hand
        if !alpha {
            out.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 1.0);
        }
        let src = raw.get(k * l.line..(k + 1) * l.line).unwrap_or_default();
        for ((c, &t), &o) in channels.iter().zip(targets).zip(&l.offsets) {
            if matches!(t, Target::Skip) {
                continue;
            }
            let sz = c.ty.size();
            let samples = src.get(o..o + sz * block.width).unwrap_or_default();
            for (p, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(samples.chunks_exact(sz)) {
                let v = sample(c.ty, s);
                match t {
                    Target::Slot(3) => p[3] = v.clamp(0.0, 1.0),
                    Target::Slot(k) => {
                        if let Some(q) = p.get_mut(k) {
                            *q = v;
                        }
                    }
                    Target::Grey => {
                        p[0] = v;
                        p[1] = v;
                        p[2] = v;
                    }
                    Target::Skip => {}
                }
            }
        }
    }
}

/// Decode an EXR file into premultiplied linear RGBA floats.
pub fn decode(name: &str, b: &[u8]) -> Result<Decoded> {
    let file = headers(name, b)?;
    let bad = |what: String| MediaError::Decode(format!("{name}: EXR {what}"));
    // channel → frame component: the main picture, else the richest colour layer, of the parts
    // that can be read
    let names: Vec<Vec<&str>> = file
        .parts
        .iter()
        .map(|p| if matches!(p.storage, Storage::Unreadable(_)) { Vec::new() } else { p.channels.iter().map(|c| c.name.as_str()).collect() })
        .collect();
    let Some((index, picture)) = channels::shown_picture(names.iter().map(|n| n.iter().copied())) else {
        if let Some(Storage::Unreadable(what)) = file.parts.iter().map(|p| &p.storage).find(|s| matches!(s, Storage::Unreadable(_))) {
            return Err(MediaError::Unsupported(format!("{name}: {what} OpenEXR parts are not supported yet, and no other part has a picture")));
        }
        return Err(MediaError::Unsupported(format!("{name}: EXR has no R, G, B or Y channel")));
    };
    let (part, names) = file.parts.get(index).zip(names.get(index)).ok_or_else(|| bad(format!("part {index}")))?;
    let compression = Compression::of(part.code).map_err(|what| MediaError::Unsupported(format!("{name}: {what} compressed OpenEXR is not supported yet")))?;
    if let Some(c) = part.channels.iter().find(|c| c.sampling != (1, 1)) {
        let (xs, ys) = c.sampling;
        return Err(MediaError::Unsupported(format!("{name}: EXR channel {} is subsampled ({xs}×{ys})", c.name)));
    }
    let luminance = matches!(picture.layout, ColorLayout::Y | ColorLayout::YA);
    let prefix = if picture.layer.is_empty() { String::new() } else { format!("{}.", picture.layer) };
    if luminance && names.iter().any(|n| n.strip_prefix(prefix.as_str()).is_some_and(|c| c == "RY" || c == "BY")) {
        return Err(MediaError::Unsupported(format!("{name}: luminance / chroma OpenEXR is not supported yet")));
    }
    let mut targets = vec![Target::Skip; part.channels.len()];
    let slots: &[Target] = match picture.layout {
        ColorLayout::Rgba | ColorLayout::Rgb => &[Target::Slot(0), Target::Slot(1), Target::Slot(2), Target::Slot(3)],
        ColorLayout::Y | ColorLayout::YA => &[Target::Grey, Target::Slot(3)],
    };
    for (&i, &slot) in picture.channels.iter().zip(slots) {
        if let Some(t) = targets.get_mut(i) {
            *t = slot;
        }
    }
    let used: Vec<PixelType> = part.channels.iter().zip(&targets).filter(|(_, t)| !matches!(t, Target::Skip)).map(|(c, _)| c.ty).collect();
    let alpha = picture.layout.has_alpha();

    let (dw, dh) = part.data.size().ok_or_else(|| bad("data window size".into()))?;
    let (fw, fh) = part.display.size().ok_or_else(|| bad("display window size".into()))?;
    let (Ok(fw32), Ok(fh32)) = (u32::try_from(fw), u32::try_from(fh)) else {
        return Err(bad("display window size".into()));
    };
    let px_len = dw.checked_mul(dh).and_then(|n| n.checked_mul(4)).ok_or_else(|| bad("size".into()))?;
    let frame_len = fw.checked_mul(fh).and_then(|n| n.checked_mul(4)).ok_or_else(|| bad("display window size".into()))?;
    let decoded_bytes = px_len
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|n| if part.data == part.display { Some(n) } else { frame_len.checked_mul(std::mem::size_of::<f32>()).and_then(|m| n.checked_add(m)) })
        .ok_or_else(|| bad("decoded size".into()))?;
    if decoded_bytes > MAX_DECODED_BYTES {
        return Err(MediaError::Unsupported(format!("{name}: EXR decoded buffers exceed the {MAX_DECODED_BYTES}-byte limit")));
    }

    // the part's offset table follows those of the parts before it
    let mut table = file.tables;
    if file.multipart {
        for (i, p) in file.parts.iter().enumerate().take(index) {
            let n = p.chunk_count.ok_or_else(|| bad(format!("part {i} has no chunkCount")))?;
            table = n.checked_mul(8).and_then(|n| table.checked_add(n)).filter(|t| *t <= b.len()).ok_or_else(|| bad("offset tables are truncated".into()))?;
        }
    }
    // a chunk's fields before its data: the part number (multi-part files), then the first line
    // of a scan-line chunk or the tile and level coordinates of a tile, then the data size
    let part_field = if file.multipart { 4 } else { 0 };
    let entry = |i: usize| le64(b, table + i * 8).and_then(|o| usize::try_from(o).ok());
    // the chunk at table entry `i`: its fields (as i32) and its data
    let chunk = |i: usize, fields: usize| -> Result<(Vec<i32>, &[u8])> {
        let off = entry(i).ok_or_else(|| bad(format!("offset of chunk {i}")))?;
        if off < table {
            return Err(bad(format!("chunk {i} lies inside the headers")));
        }
        let field = |k: usize| le32(b, off.checked_add(k * 4)?).map(|v| v as i32);
        if file.multipart && field(0) != i32::try_from(index).ok() {
            return Err(bad(format!("chunk {i} is not part {index}'s")));
        }
        let values = (0..fields).map(|k| field(part_field / 4 + k)).collect::<Option<Vec<i32>>>().ok_or_else(|| bad(format!("chunk {i} is truncated")))?;
        let size_at = off + part_field + fields * 4;
        let size = le32(b, size_at).ok_or_else(|| bad(format!("chunk {i} is truncated")))? as usize;
        let data = b
            .get(size_at + 4..(size_at + 4).checked_add(size).ok_or_else(|| bad(format!("chunk {i}")))?)
            .ok_or_else(|| bad(format!("chunk {i} is truncated")))?;
        Ok((values, data))
    };
    let needed = |n: usize| -> Result<()> {
        let end = n.checked_mul(8).and_then(|t| table.checked_add(t)).ok_or_else(|| bad("offset table size".into()))?;
        if end > b.len() {
            return Err(bad("offset table is truncated".into()));
        }
        if file.multipart && part.chunk_count.is_none_or(|c| c < n) {
            return Err(bad(format!("part {index}'s chunkCount does not cover its {n} chunks")));
        }
        Ok(())
    };

    // zeroed pages are committed only as blocks write them
    let mut px = vec![0f32; px_len];
    let row = dw * 4;
    match part.storage {
        Storage::Unreadable(ref what) => return Err(MediaError::Unsupported(format!("{name}: {what} OpenEXR parts are not supported yet"))),
        Storage::ScanLines => {
            let l = Lines::of(&part.channels, dw).ok_or_else(|| bad("line size".into()))?;
            let lpc = compression.lines_per_chunk(part.code);
            if l.line.checked_mul(lpc).is_none_or(|n| n > MAX_CHUNK_BYTES) {
                return Err(MediaError::Unsupported(format!("{name}: EXR scan lines too large ({} channels, {dw} wide)", part.channels.len())));
            }
            let chunks = dh.div_ceil(lpc);
            needed(chunks)?;
            // every chunk checked before the picture is written
            let mut blocks = Vec::with_capacity(chunks);
            for i in 0..chunks {
                let (fields, data) = chunk(i, 1)?;
                let want = i64::from(part.data.y0) + (i * lpc) as i64;
                if fields.first().map(|&y| i64::from(y)) != Some(want) {
                    return Err(bad(format!("chunk {i} is not at line {want}")));
                }
                let lines = lpc.min(dh - i * lpc);
                plausible(compression, data.len(), &l, lines).map_err(|e| bad(format!("chunk {i} {e}")))?;
                blocks.push(Block { data, width: dw, lines });
            }
            px.par_chunks_mut(row * lpc).zip(blocks).enumerate().try_for_each(|(i, (rows, block))| -> Result<()> {
                let raw = unpack(compression, block, &part.channels, &l).map_err(|e| bad(format!("chunk {i}: {e}")))?;
                place(&raw, block, &l, &part.channels, &targets, alpha, rows, row, 0);
                Ok(())
            })?;
        }
        Storage::Tiles { width: tw, height: th } => {
            let (ntx, nty) = (dw.div_ceil(tw), dh.div_ceil(th));
            let tiles = ntx.checked_mul(nty).ok_or_else(|| bad("tile count".into()))?;
            let l = Lines::of(&part.channels, tw.min(dw)).ok_or_else(|| bad("tile size".into()))?;
            if l.line.checked_mul(th.min(dh)).is_none_or(|n| n > MAX_CHUNK_BYTES) {
                return Err(MediaError::Unsupported(format!("{name}: EXR tiles too large ({} channels, {tw} × {th})", part.channels.len())));
            }
            // the full-resolution level's tiles come first in the table (any further levels
            // follow), each tile once
            needed(tiles)?;
            let mut blocks: Vec<Option<Block>> = vec![None; tiles];
            for i in 0..tiles {
                let (fields, data) = chunk(i, 4)?;
                let &[tx, ty, lx, ly] = fields.as_slice() else { return Err(bad(format!("chunk {i} is truncated"))) };
                if (lx, ly) != (0, 0) {
                    return Err(bad(format!("chunk {i} is a tile of level ({lx}, {ly}), not of the full-resolution level")));
                }
                let (Ok(tx), Ok(ty)) = (usize::try_from(tx), usize::try_from(ty)) else {
                    return Err(bad(format!("chunk {i} names tile ({tx}, {ty})")));
                };
                if tx >= ntx || ty >= nty {
                    return Err(bad(format!("chunk {i} names tile ({tx}, {ty}) of {ntx} × {nty}")));
                }
                let (width, lines) = (tw.min(dw - tx * tw), th.min(dh - ty * th));
                let lt = Lines::of(&part.channels, width).ok_or_else(|| bad("tile size".into()))?;
                plausible(compression, data.len(), &lt, lines).map_err(|e| bad(format!("tile ({tx}, {ty}) {e}")))?;
                let slot = blocks.get_mut(ty * ntx + tx).ok_or_else(|| bad(format!("tile ({tx}, {ty})")))?;
                if slot.is_some() {
                    return Err(bad(format!("tile ({tx}, {ty}) is stored twice")));
                }
                *slot = Some(Block { data, width, lines });
            }
            // a band of tiles a task, each tile decoded straight into place
            px.par_chunks_mut(row * th.min(dh)).enumerate().try_for_each(|(ty, rows)| -> Result<()> {
                for tx in 0..ntx {
                    let block = blocks.get(ty * ntx + tx).copied().flatten().ok_or_else(|| bad(format!("tile ({tx}, {ty}) is missing")))?;
                    let lt = Lines::of(&part.channels, block.width).ok_or_else(|| bad("tile size".into()))?;
                    let raw = unpack(compression, block, &part.channels, &lt).map_err(|e| bad(format!("tile ({tx}, {ty}): {e}")))?;
                    place(&raw, block, &lt, &part.channels, &targets, alpha, rows, row, tx * tw);
                }
                Ok(())
            })?;
        }
    }

    // place the data window into the display window
    let px = if part.data == part.display {
        px
    } else {
        let mut f = vec![0f32; fw * fh * 4];
        let dx = i64::from(part.data.x0) - i64::from(part.display.x0);
        let dy = i64::from(part.data.y0) - i64::from(part.display.y0);
        let (x0, x1) = (dx.max(0), (dx + dw as i64).min(fw as i64));
        if x0 < x1 {
            for (y, out) in f.chunks_exact_mut(fw * 4).enumerate() {
                let sy = y as i64 - dy;
                if sy < 0 || sy >= dh as i64 {
                    continue;
                }
                let src = (sy as usize * dw + (x0 - dx) as usize) * 4;
                let len = (x1 - x0) as usize * 4;
                if let (Some(d), Some(s)) = (out.get_mut(x0 as usize * 4..x0 as usize * 4 + len), px.get(src..src + len)) {
                    d.copy_from_slice(s);
                }
            }
        }
        f
    };
    let has_alpha = px.par_chunks(4).any(|p| p.get(3).is_some_and(|&a| a < 1.0));
    let depth = match used.iter().max() {
        Some(PixelType::Float) => "32-bit float",
        Some(PixelType::Uint) => "32-bit integer",
        _ => "16-bit float",
    };
    let layout = match picture.layout {
        ColorLayout::Rgba => "RGBA",
        ColorLayout::Rgb => "RGB",
        ColorLayout::YA => "YA",
        ColorLayout::Y => "Y",
    };
    let mut pixel_format = format!("{layout} {depth}");
    if file.multipart
        && let Some(p) = part.name.as_deref().filter(|p| !p.is_empty())
    {
        pixel_format.push_str(&format!(", part {p}"));
    }
    if !picture.layer.is_empty() {
        pixel_format.push_str(&format!(", layer {}", picture.layer));
    }
    let mut frame = VideoFrame::rgba_f32(fw32, fh32, px);
    frame.color = LINEAR_709;
    Ok(Decoded { frame, has_alpha, pixel_format })
}

#[cfg(test)]
mod tests;

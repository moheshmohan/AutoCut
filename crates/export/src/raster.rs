//! Still formats written directly from frames: Targa, DPX (from RGBA8) and OpenEXR (from the
//! compositor's float pictures, or RGBA8 for single stills).
//!
//! Targa follows the Truevision TGA File Format Specification 2.0 (image type 2, uncompressed
//! true-colour, bottom-left origin, the layout every reader accepts). DPX follows SMPTE ST 268
//! (version 2.0 header): one RGB image element, 10 bits per component packed into 32-bit
//! big-endian words (method A, two padding bits at the bottom), BT.709 transfer and colorimetry.
//! OpenEXR follows the public OpenEXR file layout (version 2, single-part scan-line file): half or
//! 32-bit float B, G, R (and A) channels, scene-linear values with premultiplied alpha, as EXR
//! readers expect, their primaries in the `chromaticities` attribute; uncompressed, RLE, ZIPS (zlib
//! per scan line), ZIP (zlib per 16 scan lines) or PXR24 (per 16 scan lines: each line's sample
//! differences as byte planes, zlib; a 32-bit float keeps its top 24 bits). RLE and ZIP first
//! split each chunk's bytes into even and odd halves and delta-code them (the format's
//! predictor); a chunk that would not shrink is stored raw, as the format allows.
use crate::{ExportError, ExrCompression, ExrDepth, Result};

/// An empty buffer with room for `len` bytes, or an error when the allocation fails (instead of
/// aborting the process). The caller already holds the frame, so the second, file-sized buffer
/// of a large export is the allocation most likely to fail.
fn output_buffer(len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| ExportError::Encode(format!("not enough memory for a {len}-byte frame file")))?;
    Ok(out)
}

/// `w * h * 4` when `rgba` holds exactly that many bytes.
fn checked_len(rgba: &[u8], w: u32, h: u32) -> Result<usize> {
    let n = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(4)).ok_or_else(|| ExportError::Encode("frame size".into()))?;
    if w == 0 || h == 0 || rgba.len() != n {
        return Err(ExportError::Encode("frame size".into()));
    }
    Ok(n)
}

/// The RGB bytes of an RGBA8 frame (alpha dropped).
pub(crate) fn rgb(rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let n = checked_len(rgba, w, h)?;
    let mut out = output_buffer(n / 4 * 3)?;
    out.extend(rgba.as_chunks::<4>().0.iter().flat_map(|&[r, g, b, _]| [r, g, b]));
    Ok(out)
}

/// An uncompressed Targa file: 24-bit BGR, or 32-bit BGRA with `alpha` (straight alpha, as the
/// frame carries it).
pub(crate) fn tga(rgba: &[u8], w: u32, h: u32, alpha: bool) -> Result<Vec<u8>> {
    checked_len(rgba, w, h)?;
    let (tw, th) = match (u16::try_from(w), u16::try_from(h)) {
        (Ok(tw), Ok(th)) => (tw, th),
        _ => return Err(ExportError::Unsupported(format!("Targa frames are at most 65535×65535 (got {w}×{h})"))),
    };
    let bpp: usize = if alpha { 4 } else { 3 };
    let row = w as usize * 4;
    let len = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(bpp))
        .and_then(|n| n.checked_add(18 + 26))
        .ok_or_else(|| ExportError::Encode("frame size".into()))?;
    let mut out = output_buffer(len)?;
    out.extend_from_slice(&[0, 0, 2]); // no ID, no colour map, uncompressed true-colour
    out.extend_from_slice(&[0; 5]); // colour map specification (unused)
    out.extend_from_slice(&[0, 0, 0, 0]); // x / y origin
    out.extend_from_slice(&tw.to_le_bytes());
    out.extend_from_slice(&th.to_le_bytes());
    out.push(if alpha { 32 } else { 24 });
    out.push(if alpha { 8 } else { 0 }); // attribute bits; bottom-left origin
    // bottom-left origin: the last frame row comes first
    for line in rgba.chunks_exact(row).rev() {
        for &[r, g, b, a] in line.as_chunks::<4>().0 {
            out.extend_from_slice(&[b, g, r]);
            if alpha {
                out.push(a);
            }
        }
    }
    // TGA 2.0 footer without extension or developer areas
    out.extend_from_slice(&[0; 8]);
    out.extend_from_slice(b"TRUEVISION-XFILE.\0");
    Ok(out)
}

/// Header size of the DPX files we write (file + image + orientation + film + TV headers).
const DPX_HEADER: usize = 2048;

/// A DPX file: 10-bit RGB (8-bit values widened by bit replication, so 0 → 0 and 255 → 1023).
pub(crate) fn dpx(rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let n = checked_len(rgba, w, h)?;
    // one 32-bit word per pixel; `n` is already `w * h * 4`
    let size = n.checked_add(DPX_HEADER).ok_or_else(|| ExportError::Encode("frame size".into()))?;
    let file_size = u32::try_from(size).map_err(|_| ExportError::Unsupported(format!("a {w}×{h} DPX frame exceeds 4 GB")))?;
    // SMPTE ST 268: undefined numeric fields are all ones, undefined text fields are NUL
    let mut hd = vec![0xFF_u8; DPX_HEADER];
    for (a, b) in [
        (36, 660),    // file name, creation time, creator, project, copyright
        (664, 768),   // reserved
        (1356, 1408), // reserved
        (1432, 1620), // source file name, time, input device, serial number
        (1644, 1664), // reserved
        (1664, 1712), // film manufacturer, type, offset, prefix, count, format
        (1732, 1920), // frame id, slate, reserved
        (1972, 2048), // reserved
    ] {
        hd[a..b].fill(0);
    }
    for e in 0..8 {
        let d = 780 + e * 72 + 40;
        hd[d..d + 32].fill(0); // element descriptions
    }
    let put32 = |hd: &mut Vec<u8>, at: usize, v: u32| hd[at..at + 4].copy_from_slice(&v.to_be_bytes());
    let put16 = |hd: &mut Vec<u8>, at: usize, v: u16| hd[at..at + 2].copy_from_slice(&v.to_be_bytes());
    // file information header
    hd[0..4].copy_from_slice(b"SDPX");
    put32(&mut hd, 4, DPX_HEADER as u32); // offset to the image data
    hd[8..16].copy_from_slice(b"V2.0\0\0\0\0");
    put32(&mut hd, 16, file_size);
    put32(&mut hd, 20, 1); // ditto key: new frame
    put32(&mut hd, 24, 1664); // generic header length
    put32(&mut hd, 28, 384); // industry header length
    put32(&mut hd, 32, 0); // user data length
    hd[160..169].copy_from_slice(b"FilmCraft");
    // image information header
    put16(&mut hd, 768, 0); // orientation: left to right, top to bottom
    put16(&mut hd, 770, 1); // one image element
    put32(&mut hd, 772, w);
    put32(&mut hd, 776, h);
    let el = 780;
    put32(&mut hd, el, 0); // unsigned
    put32(&mut hd, el + 4, 0); // reference low data code
    put32(&mut hd, el + 12, 1023); // reference high data code
    hd[el + 20] = 50; // descriptor: RGB
    hd[el + 21] = 6; // transfer: ITU-R BT.709
    hd[el + 22] = 6; // colorimetric: ITU-R BT.709
    hd[el + 23] = 10; // bit depth
    put16(&mut hd, el + 24, 1); // packing: filled to 32-bit words, method A
    put16(&mut hd, el + 26, 0); // no encoding (uncompressed)
    put32(&mut hd, el + 28, DPX_HEADER as u32); // offset to the element's data
    put32(&mut hd, el + 32, 0); // end-of-line padding
    put32(&mut hd, el + 36, 0); // end-of-image padding
    // image orientation header: no offset, square pixels
    put32(&mut hd, 1408, 0);
    put32(&mut hd, 1412, 0);
    put32(&mut hd, 1424, w);
    put32(&mut hd, 1428, h);
    put32(&mut hd, 1628, 1);
    put32(&mut hd, 1632, 1);
    // television header: progressive
    hd[1928] = 0;
    let mut out = output_buffer(size)?;
    out.extend_from_slice(&hd);
    let ten = |v: u8| (u32::from(v) << 2) | (u32::from(v) >> 6);
    for &[r, g, b, _] in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&((ten(r) << 22) | (ten(g) << 12) | (ten(b) << 2)).to_be_bytes());
    }
    Ok(out)
}

/// IEEE 754 binary16 bits of `v`, rounded to nearest even (overflow → infinity, NaN → quiet NaN).
fn half(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    // rounds `m >> shift` to nearest, ties to even (`shift` is 1..=24)
    let round = |m: u32, shift: u32| {
        let (q, rem, mid) = (m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1));
        if rem > mid || (rem == mid && q & 1 == 1) { q + 1 } else { q }
    };
    let e = exp - 127 + 15;
    if e >= 0x1F {
        sign | 0x7C00
    } else if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal half: the 24-bit significand scaled by 2^(e - 14); rounding up may carry
        // into the smallest normal, which is the right encoding
        sign | round(man | 0x80_0000, (14 - e) as u32) as u16
    } else {
        // a carry out of the mantissa bumps the exponent (up to infinity), as it should
        sign | round(((e as u32) << 23) | man, 13) as u16
    }
}

/// How an OpenEXR frame is written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExrLayout {
    /// Write the alpha channel (premultiplied, as EXR defines it); without it the picture is
    /// flattened over black.
    pub alpha: bool,
    pub depth: ExrDepth,
    pub compression: ExrCompression,
    /// CIE xy of red, green, blue and white: the primaries of the frame's linear values.
    pub chromaticities: [[f32; 2]; 4],
}

/// The `chromaticities` of linear light in `gamut`, with the D65 white of every gamut FilmCraft
/// works in.
pub(crate) fn chromaticities(gamut: filmcraft_color::Gamut) -> [[f32; 2]; 4] {
    let [r, g, b] = gamut.primaries().map(|[x, y]| [x as f32, y as f32]);
    [r, g, b, [0.3127, 0.3290]]
}

/// An OpenEXR file of an RGBA8 frame: half-float scene-linear Rec.709 RGB (alpha ignored), or
/// RGBA with `alpha` (premultiplied, as EXR defines alpha; the frame's straight alpha is
/// multiplied in after linearisation).
pub(crate) fn exr(rgba: &[u8], w: u32, h: u32, alpha: bool, compression: ExrCompression) -> Result<Vec<u8>> {
    let n = checked_len(rgba, w, h)?;
    let lin = filmcraft_color::srgb_u8_to_linear_table();
    let mut px = Vec::new();
    px.try_reserve_exact(n).map_err(|_| ExportError::Encode(format!("not enough memory for a {w}×{h} float frame")))?;
    px.extend(rgba.get(..n).unwrap_or_default().as_chunks::<4>().0.iter().flat_map(|&[r, g, b, a]| {
        let k = if alpha { f32::from(a) / 255.0 } else { 1.0 };
        [lin[r as usize] * k, lin[g as usize] * k, lin[b as usize] * k, k]
    }));
    let layout = ExrLayout { alpha, depth: ExrDepth::Half, compression, chromaticities: chromaticities(filmcraft_color::Gamut::Bt709) };
    exr_linear(&px, w, h, layout)
}

/// A float's top 24 bits as PXR24 stores them: rounded to nearest, a finite value never rounded
/// up to infinity, infinities and NaN kept as such.
fn float24(v: f32) -> u32 {
    let b = v.to_bits();
    let exp = (b >> 23) & 0xFF;
    if exp == 0xFF {
        // keep a NaN a NaN when its payload is in the dropped bits
        let t = b >> 8;
        return if b & 0x7F_FFFF != 0 && t & 0x7FFF == 0 { t | 1 } else { t };
    }
    let t = (b + 0x80) >> 8;
    if (t >> 15) & 0xFF == 0xFF { b >> 8 } else { t }
}

/// An OpenEXR file of premultiplied linear RGBA floats `px` (`w * h * 4`, top row first): B, G,
/// R (and A) channels of `layout.depth`, NaN written as 0 and half floats limited to ±65504.
pub(crate) fn exr_linear(px: &[f32], w: u32, h: u32, layout: ExrLayout) -> Result<Vec<u8>> {
    let n = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(4)).ok_or_else(|| ExportError::Encode("frame size".into()))?;
    if w == 0 || h == 0 || px.len() != n {
        return Err(ExportError::Encode("frame size".into()));
    }
    let ExrLayout { alpha, depth, compression, chromaticities } = layout;
    let too_large = || ExportError::Unsupported(format!("an OpenEXR frame of {w}×{h} is too large"));
    let (Ok(x_max), Ok(y_max)) = (i32::try_from(w - 1), i32::try_from(h - 1)) else {
        return Err(too_large());
    };
    // channels in the alphabetical order the file layout requires, as RGBA sample indices
    let (names, slots): (&[u8], &[usize]) = if alpha { (b"ABGR", &[3, 2, 1, 0]) } else { (b"BGR", &[2, 1, 0]) };
    let size = depth.bytes();
    let line = (w as usize).checked_mul(names.len() * size).ok_or_else(|| ExportError::Encode("frame size".into()))?;
    // a chunk's byte count is a signed 32-bit field
    let lines = compression.lines();
    if line.checked_mul(lines).and_then(|c| i32::try_from(c).ok()).is_none() {
        return Err(too_large());
    }

    let mut hd = Vec::with_capacity(512);
    hd.extend_from_slice(&[0x76, 0x2F, 0x31, 0x01]); // magic number 20000630
    hd.extend_from_slice(&[2, 0, 0, 0]); // version 2, single-part scan-line file, short names
    let attr = |hd: &mut Vec<u8>, name: &str, ty: &str, value: &[u8]| {
        for s in [name, ty] {
            hd.extend_from_slice(s.as_bytes());
            hd.push(0);
        }
        hd.extend_from_slice(&(value.len() as i32).to_le_bytes());
        hd.extend_from_slice(value);
    };
    let le = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let pixel_type: i32 = match depth {
        ExrDepth::Half => 1,
        ExrDepth::Float => 2,
    };
    let mut channels = Vec::new();
    for &c in names {
        channels.extend_from_slice(&[c, 0]);
        channels.extend_from_slice(&pixel_type.to_le_bytes());
        channels.extend_from_slice(&[0, 0, 0, 0]); // pLinear 0, reserved
        channels.extend_from_slice(&1_i32.to_le_bytes()); // x sampling
        channels.extend_from_slice(&1_i32.to_le_bytes()); // y sampling
    }
    channels.push(0);
    let window: Vec<u8> = [0, 0, x_max, y_max].iter().flat_map(|v| v.to_le_bytes()).collect();
    attr(&mut hd, "channels", "chlist", &channels);
    attr(&mut hd, "chromaticities", "chromaticities", &le(chromaticities.as_flattened()));
    attr(&mut hd, "compression", "compression", &[compression.code()]);
    attr(&mut hd, "dataWindow", "box2i", &window);
    attr(&mut hd, "displayWindow", "box2i", &window);
    attr(&mut hd, "lineOrder", "lineOrder", &[0]); // INCREASING_Y
    attr(&mut hd, "pixelAspectRatio", "float", &le(&[1.0]));
    attr(&mut hd, "screenWindowCenter", "v2f", &le(&[0.0, 0.0]));
    attr(&mut hd, "screenWindowWidth", "float", &le(&[1.0]));
    hd.push(0); // end of header

    // offset table (one entry per chunk), then the chunks: first y, data size, data
    let chunks = (h as usize).div_ceil(lines);
    let table = chunks.checked_mul(8);
    let body = (h as usize).checked_mul(line + 8);
    let total = table.zip(body).and_then(|(t, b)| hd.len().checked_add(t)?.checked_add(b)).ok_or_else(|| ExportError::Encode("frame size".into()))?;
    // the uncompressed size: compressed chunks only shrink it
    let mut out = output_buffer(total)?;
    out.extend_from_slice(&hd);
    let table_at = out.len();
    out.resize(table_at + chunks * 8, 0);
    let wn = w as usize;
    let row = wn * 4;
    // one line's samples as stored: every pixel of the first channel, then the next channel …
    let sample = |v: f32| if v.is_nan() { 0.0 } else { v };
    let mut values: Vec<u32> = vec![0; wn * names.len()];
    let mut raw = Vec::with_capacity(line * lines);
    let mut planes = Vec::new();
    // the deflate hash tables, shared by the frame's chunks instead of zeroed for each one
    let mut tables = crate::deflate::Tables::default();
    for (k, block) in px.chunks(row * lines).enumerate() {
        raw.clear();
        planes.clear();
        for src in block.chunks_exact(row) {
            for (x, p) in src.as_chunks::<4>().0.iter().enumerate() {
                for (c, &slot) in slots.iter().enumerate() {
                    let v = sample(p.get(slot).copied().unwrap_or(0.0));
                    if let Some(out) = values.get_mut(c * wn + x) {
                        *out = match depth {
                            ExrDepth::Half => u32::from(half(v.clamp(-65504.0, 65504.0))),
                            ExrDepth::Float => v.to_bits(),
                        };
                    }
                }
            }
            match depth {
                ExrDepth::Half => raw.extend(values.iter().flat_map(|&v| (v as u16).to_le_bytes())),
                ExrDepth::Float => raw.extend(values.iter().flat_map(|&v| v.to_le_bytes())),
            }
            if compression == ExrCompression::Pxr24 {
                // per channel: the differences to the previous sample of the line (modulo 2^16 or
                // 2^24), as byte planes, the most significant first
                for chan in values.chunks_exact(wn) {
                    let (bytes, coded): (usize, Vec<u32>) = match depth {
                        ExrDepth::Half => (2, chan.to_vec()),
                        ExrDepth::Float => (3, chan.iter().map(|&b| float24(f32::from_bits(b))).collect()),
                    };
                    let mut prev = 0u32;
                    let diffs: Vec<u32> = coded
                        .iter()
                        .map(|&v| {
                            let d = v.wrapping_sub(prev);
                            prev = v;
                            d
                        })
                        .collect();
                    for byte in (0..bytes).rev() {
                        planes.extend(diffs.iter().map(|d| (d >> (8 * byte)) as u8));
                    }
                }
            }
        }
        let packed = match compression {
            ExrCompression::None => None,
            ExrCompression::Rle => Some(rle(&predict(&raw))),
            ExrCompression::Zips | ExrCompression::Zip => Some(crate::deflate::zlib(&predict(&raw), &mut tables)),
            ExrCompression::Pxr24 => Some(crate::deflate::zlib(&planes, &mut tables)),
        };
        // a chunk that does not shrink is stored as it is (readers compare the sizes)
        let data = packed.as_deref().filter(|p| p.len() < raw.len()).unwrap_or(&raw);
        let offset = (out.len() as u64).to_le_bytes();
        if let Some(entry) = out.get_mut(table_at + k * 8..table_at + k * 8 + 8) {
            entry.copy_from_slice(&offset);
        }
        out.extend_from_slice(&((k * lines) as i32).to_le_bytes());
        out.extend_from_slice(&(data.len() as i32).to_le_bytes());
        out.extend_from_slice(data);
    }
    Ok(out)
}

/// The OpenEXR RLE / ZIP predictor: bytes split into even and odd halves, then each byte replaced
/// by its difference from the previous one, plus 128.
fn predict(raw: &[u8]) -> Vec<u8> {
    let half = raw.len().div_ceil(2);
    let mut t = vec![0_u8; raw.len()];
    for (i, &b) in raw.iter().enumerate() {
        t[if i % 2 == 0 { i / 2 } else { half + i / 2 }] = b;
    }
    let mut p = t.first().copied().unwrap_or(0);
    for v in t.iter_mut().skip(1) {
        let d = v.wrapping_sub(p).wrapping_add(128);
        p = *v;
        *v = d;
    }
    t
}

/// OpenEXR run-length coding: a signed count byte, then either `count + 1` copies of one byte
/// (count ≥ 0, runs of 3 to 128) or `-count` literal bytes (1 to 127).
fn rle(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() + src.len() / 127 + 1);
    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        let run = src[i..].iter().take(128).take_while(|&&x| x == b).count();
        if run >= 3 {
            out.extend_from_slice(&[(run - 1) as u8, b]);
            i += run;
            continue;
        }
        // literals up to the next run of three (or 127 bytes)
        let s = i;
        while i < src.len() && i - s < 127 {
            if src.get(i..i + 3).is_some_and(|t| t[0] == t[1] && t[1] == t[2]) {
                break;
            }
            i += 1;
        }
        out.push(((i - s) as u8).wrapping_neg());
        out.extend_from_slice(&src[s..i]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be32(b: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn dpx_header_and_packing() {
        // 2×1: pure red, then (1, 128, 255)
        let px = [255, 0, 0, 255, 1, 128, 255, 7];
        let f = dpx(&px, 2, 1).unwrap();
        assert_eq!(&f[0..4], b"SDPX");
        assert_eq!(f.len(), 2048 + 8);
        assert_eq!(be32(&f, 4), 2048);
        assert_eq!(be32(&f, 16), f.len() as u32);
        assert_eq!((be32(&f, 772), be32(&f, 776)), (2, 1));
        assert_eq!((f[800], f[803]), (50, 10));
        assert_eq!(be32(&f, 2048), 1023 << 22);
        let p = be32(&f, 2052);
        assert_eq!(((p >> 22) & 1023, (p >> 12) & 1023, (p >> 2) & 1023, p & 3), (4, 514, 1023, 0));
    }

    #[test]
    fn tga_rows_are_bottom_up_bgr() {
        // 1×2: top red, bottom blue (half transparent)
        let px = [255, 0, 0, 255, 0, 0, 255, 128];
        let f = tga(&px, 1, 2, false).unwrap();
        assert_eq!(&f[..3], &[0, 0, 2]);
        assert_eq!((f[16], f[17]), (24, 0));
        assert_eq!(&f[18..24], &[255, 0, 0, 0, 0, 255]); // blue row first, as BGR
        assert!(f.ends_with(b"TRUEVISION-XFILE.\0"));
        let f = tga(&px, 1, 2, true).unwrap();
        assert_eq!((f[16], f[17]), (32, 8));
        assert_eq!(&f[18..26], &[255, 0, 0, 128, 0, 0, 255, 255]);
    }

    #[test]
    fn half_floats_round_to_nearest_even() {
        for (v, h) in [
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3C00),
            (0.5, 0x3800),
            (-2.0, 0xC000),
            (65504.0, 0x7BFF),
            (65520.0, 0x7C00),        // halfway past the largest half: ties to even = infinity
            (1.0e9, 0x7C00),          // overflow
            (6.103_515_6e-5, 0x0400), // smallest normal
            (5.960_464_5e-8, 0x0001), // smallest subnormal, 2^-24
            (2.980_232_2e-8, 0x0000), // 2^-25: halfway to zero, ties to even
            (1.0e-10, 0x0000),
            (1.000_488_3, 0x3C00), // 1 + 2^-11: halfway, ties to even (down)
            (1.001_464_8, 0x3C02), // 1 + 3·2^-11: halfway, ties to even (up)
            (f32::INFINITY, 0x7C00),
        ] {
            assert_eq!(half(v), h, "{v}");
        }
        assert_eq!(half(f32::NAN) & 0x7E00, 0x7E00);
    }

    fn le32(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    /// Where `needle` (an attribute name, NUL-terminated) starts in the header.
    fn find(b: &[u8], needle: &[u8]) -> usize {
        b.windows(needle.len()).position(|w| w == needle).unwrap()
    }

    #[test]
    fn exr_layout() {
        // 2×1: opaque white, then (255, 0, 0) at alpha 0.5
        let px = [255, 255, 255, 255, 255, 0, 0, 128];
        for alpha in [false, true] {
            let f = exr(&px, 2, 1, alpha, ExrCompression::None).unwrap();
            let nch = if alpha { 4 } else { 3 };
            assert_eq!(&f[..8], &[0x76, 0x2F, 0x31, 0x01, 2, 0, 0, 0]);
            let ch = find(&f, b"channels\0chlist\0") + 16;
            assert_eq!(le32(&f, ch) as usize, nch * 18 + 1);
            let names: Vec<u8> = (0..nch).map(|i| f[ch + 4 + i * 18]).collect();
            assert_eq!(names, if alpha { b"ABGR".to_vec() } else { b"BGR".to_vec() });
            let dw = find(&f, b"dataWindow\0box2i\0") + 17;
            assert_eq!((le32(&f, dw), le32(&f, dw + 4), le32(&f, dw + 8), le32(&f, dw + 12)), (16, 0, 0, 1));
            let cmp = find(&f, b"compression\0compression\0") + 24;
            assert_eq!((le32(&f, cmp), f[cmp + 4]), (1, 0));
            // header ends with a NUL; one offset (scan line 0) points at the only chunk
            let end = f.len() - (8 + 2 * nch * 2) - 8;
            assert_eq!(f[end - 1], 0);
            let off = u64::from_le_bytes(f[end..end + 8].try_into().unwrap()) as usize;
            assert_eq!(off, end + 8);
            assert_eq!((le32(&f, off), le32(&f, off + 4) as usize), (0, 2 * nch * 2));
            let hv = |c: usize, x: usize| u16::from_le_bytes(f[off + 8 + (c * 2 + x) * 2..][..2].try_into().unwrap());
            let (b, g, r) = if alpha { (1, 2, 3) } else { (0, 1, 2) };
            // white is 1.0 everywhere; red is linear 1.0 · alpha (premultiplied) or 1.0 flat
            assert_eq!((hv(b, 0), hv(g, 0), hv(r, 0)), (0x3C00, 0x3C00, 0x3C00));
            assert_eq!((hv(b, 1), hv(g, 1)), (0, 0));
            if alpha {
                assert_eq!((hv(0, 0), hv(0, 1)), (0x3C00, half(128.0 / 255.0)));
                assert_eq!(hv(r, 1), half(128.0 / 255.0));
            } else {
                assert_eq!(hv(r, 1), 0x3C00);
            }
        }
    }

    /// A 40×35 test picture: gradients, a flat band and a noisy corner, with varying alpha (35
    /// rows: the last 16-line ZIP chunk is short).
    fn picture() -> (Vec<u8>, u32, u32) {
        let (w, h) = (40_u32, 35_u32);
        let px = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let noise = (x * 7919 + y * 104_729) % 251;
                let r = if y < 12 {
                    (x * 6) as u8
                } else if y < 20 {
                    200
                } else {
                    noise as u8
                };
                [r, (y * 7) as u8, ((x + y) * 3) as u8, [255, 128, 0, 64][(x / 10) as usize % 4]]
            })
            .collect();
        (px, w, h)
    }

    /// The chunks of an OpenEXR file we wrote: (first y, data).
    fn chunks(f: &[u8], count: usize) -> Vec<(u32, &[u8])> {
        let end = find(f, b"screenWindowWidth\0float\0") + 24 + 4 + 4 + 1;
        (0..count)
            .map(|k| {
                let off = u64::from_le_bytes(f[end + k * 8..end + k * 8 + 8].try_into().unwrap()) as usize;
                let n = le32(f, off + 4) as usize;
                (le32(f, off), &f[off + 8..off + 8 + n])
            })
            .collect()
    }

    /// Undo [`predict`].
    fn unpredict(t: &[u8]) -> Vec<u8> {
        let mut t = t.to_vec();
        for i in 1..t.len() {
            t[i] = t[i - 1].wrapping_add(t[i]).wrapping_sub(128);
        }
        let half = t.len().div_ceil(2);
        (0..t.len()).map(|i| t[if i % 2 == 0 { i / 2 } else { half + i / 2 }]).collect()
    }

    fn unrle(mut s: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        while let [c, rest @ ..] = s {
            let c = *c as i8;
            if c < 0 {
                out.extend_from_slice(&rest[..c.unsigned_abs() as usize]);
                s = &rest[c.unsigned_abs() as usize..];
            } else {
                out.extend(std::iter::repeat_n(rest[0], c as usize + 1));
                s = &rest[1..];
            }
        }
        out
    }

    #[test]
    fn rle_codes_runs_and_literals() {
        for src in [vec![], vec![5], vec![1, 1], vec![9; 300], (0..=255).collect::<Vec<u8>>(), [vec![1, 2], vec![3; 3], vec![4, 4, 5]].concat()] {
            assert_eq!(unrle(&rle(&src)), src);
        }
        assert_eq!(rle(&[7; 130]), vec![127, 7, 0xFE, 7, 7]);
        assert_eq!(rle(&[1, 2, 2, 2]), vec![0xFF, 1, 2, 2]);
    }

    /// Every compression stores the same pixels as the uncompressed file, in chunks of the right
    /// height, and the compressed files are smaller.
    #[test]
    fn compressed_exr_holds_the_same_pixels() {
        let (px, w, h) = picture();
        for alpha in [false, true] {
            let plain = exr(&px, w, h, alpha, ExrCompression::None).unwrap();
            let lines: Vec<&[u8]> = chunks(&plain, h as usize).into_iter().map(|(_, d)| d).collect();
            for c in [ExrCompression::Rle, ExrCompression::Zips, ExrCompression::Zip] {
                let f = exr(&px, w, h, alpha, c).unwrap();
                let cmp = find(&f, b"compression\0compression\0") + 24;
                assert_eq!(f[cmp + 4], c.code());
                assert!(f.len() < plain.len() * 3 / 4, "{c:?}: {} vs {}", f.len(), plain.len());
                let n = (h as usize).div_ceil(c.lines());
                for (k, (y, data)) in chunks(&f, n).into_iter().enumerate() {
                    assert_eq!(y as usize, k * c.lines());
                    let want: Vec<u8> = lines[k * c.lines()..((k + 1) * c.lines()).min(h as usize)].concat();
                    let got = if data.len() == want.len() {
                        data.to_vec()
                    } else if c == ExrCompression::Rle {
                        unpredict(&unrle(data))
                    } else {
                        unpredict(&crate::deflate::tests::inflate(data).unwrap())
                    };
                    assert_eq!(got, want, "{c:?} chunk {k}");
                }
            }
        }
    }

    /// ffmpeg (external test oracle) decodes our OpenEXR files to the values we meant: linear
    /// light, premultiplied alpha, rows top to bottom; in every compression.
    #[test]
    fn exr_decodes_in_ffmpeg() {
        let Some(ffmpeg) = filmcraft_testkit::ffmpeg_or_skip("OpenEXR stills") else { return };
        let (px, w, h) = picture();
        let dir = filmcraft_testkit::fixtures::workspace_root().join("target").join("export-tests").join(format!("exr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lin = filmcraft_color::srgb_u8_to_linear_table();
        for c in ExrCompression::ALL {
            for alpha in [false, true] {
                let path = dir.join(format!("f{alpha}-{c:?}.exr"));
                std::fs::write(&path, exr(&px, w, h, alpha, c).unwrap()).unwrap();
                let out = std::process::Command::new(&ffmpeg)
                    .args(["-v", "error", "-i"])
                    .arg(&path)
                    .args(["-f", "rawvideo", "-pix_fmt", "gbrapf32le", "-"])
                    .output()
                    .unwrap();
                assert!(out.status.success(), "{c:?}: {}", String::from_utf8_lossy(&out.stderr));
                let n = (w * h) as usize;
                assert_eq!(out.stdout.len(), n * 4 * 4, "{c:?}");
                let plane = |p: usize, i: usize| f32::from_le_bytes(out.stdout[(p * n + i) * 4..][..4].try_into().unwrap());
                for (i, &[r, g, b, a]) in px.as_chunks::<4>().0.iter().enumerate() {
                    let k = if alpha { f32::from(a) / 255.0 } else { 1.0 };
                    // gbrap planes: G, B, R, A
                    let want = [lin[g as usize] * k, lin[b as usize] * k, lin[r as usize] * k, k];
                    for (p, want) in want.into_iter().enumerate() {
                        let got = plane(p, i);
                        assert!((got - want).abs() <= 1e-3 + want * 1e-3, "{c:?}, alpha {alpha}, pixel {i}, plane {p}: {got} vs {want}");
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A frame-sized output buffer that cannot be allocated is an error, not an abort.
    #[test]
    fn a_failed_output_allocation_is_an_error() {
        assert!(matches!(output_buffer(usize::MAX), Err(ExportError::Encode(_))));
        assert!(output_buffer(16).is_ok_and(|b| b.is_empty() && b.capacity() >= 16));
    }

    fn layout(alpha: bool, depth: ExrDepth, compression: ExrCompression) -> ExrLayout {
        ExrLayout { alpha, depth, compression, chromaticities: chromaticities(filmcraft_color::Gamut::Bt709) }
    }

    /// A 37×21 float picture beyond what 8 bits hold: values above 1, negative values, fine
    /// gradients, alpha (premultiplied) of every level.
    fn float_picture() -> (Vec<f32>, u32, u32) {
        let (w, h) = (37_u32, 21_u32);
        let px = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x as f32, y as f32)))
            .flat_map(|(x, y)| {
                let a = ((x + y) % 5.0) / 4.0;
                [(x * 0.731 - 3.0) * a, (y * 0.0013 + 0.0001) * a, (40.0 - x * y * 0.05) * a, a]
            })
            .collect();
        (px, w, h)
    }

    /// The samples of one channel (`name`) of an uncompressed float or half file we wrote.
    fn samples(f: &[u8], name: u8, w: usize, h: usize, size: usize, nch: usize) -> Vec<f32> {
        let names: &[u8] = if nch == 4 { b"ABGR" } else { b"BGR" };
        let c = names.iter().position(|&n| n == name).unwrap();
        let lines = chunks(f, h);
        lines
            .iter()
            .flat_map(|(_, d)| {
                d[c * w * size..(c + 1) * w * size]
                    .chunks_exact(size)
                    .map(|s| if size == 4 { f32::from_le_bytes(s.try_into().unwrap()) } else { half_value(u16::from_le_bytes(s.try_into().unwrap())) })
            })
            .collect()
    }

    /// A half float's value, computed arithmetically.
    fn half_value(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let (e, m) = (i32::from((h >> 10) & 0x1F), f32::from(h & 0x3FF));
        sign * if e == 0 { m * 2f32.powi(-24) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
    }

    /// The float writer keeps the compositor's values: 32-bit floats bit for bit, half floats
    /// rounded to the nearest half, values above 1 and below 0 included. The RGBA8 writer clipped
    /// them to 0–1 and 8 bits.
    #[test]
    fn float_exr_keeps_the_compositor_values() {
        let (px, w, h) = float_picture();
        for alpha in [false, true] {
            let nch = if alpha { 4 } else { 3 };
            let f = exr_linear(&px, w, h, layout(alpha, ExrDepth::Float, ExrCompression::None)).unwrap();
            let ch = find(&f, b"channels\0chlist\0") + 16;
            assert_eq!(le32(&f, ch + 4 + 2), 2, "pixel type FLOAT");
            for (name, slot) in [(b'R', 0), (b'G', 1), (b'B', 2), (b'A', 3)].into_iter().take(nch) {
                let got = samples(&f, name, w as usize, h as usize, 4, nch);
                let want: Vec<f32> = px.as_chunks::<4>().0.iter().map(|p| p[slot]).collect();
                assert_eq!(got, want, "{}", name as char);
            }
            assert!(samples(&f, b'B', w as usize, h as usize, 4, nch).iter().any(|&v| v > 10.0));
            let g = exr_linear(&px, w, h, layout(alpha, ExrDepth::Half, ExrCompression::None)).unwrap();
            let ch = find(&g, b"channels\0chlist\0") + 16;
            assert_eq!(le32(&g, ch + 4 + 2), 1, "pixel type HALF");
            let got = samples(&g, b'R', w as usize, h as usize, 2, nch);
            let want: Vec<f32> = px.as_chunks::<4>().0.iter().map(|p| half_value(half(p[0]))).collect();
            assert_eq!(got, want);
            assert!(got.iter().any(|&v| v < -1.0));
        }
        // NaN is written as 0, and half floats stop at ±65504 instead of becoming infinite
        let odd = [f32::NAN, f32::INFINITY, -1.0e9, 1.0];
        let f = exr_linear(&odd, 1, 1, layout(true, ExrDepth::Half, ExrCompression::None)).unwrap();
        assert_eq!(b"RGBA".map(|n| samples(&f, n, 1, 1, 2, 4)[0]), [0.0, 65504.0, -65504.0, 1.0]);
        let f = exr_linear(&odd, 1, 1, layout(true, ExrDepth::Float, ExrCompression::None)).unwrap();
        assert_eq!(b"RGB".map(|n| samples(&f, n, 1, 1, 4, 4)[0]), [0.0, f32::INFINITY, -1.0e9]);
    }

    /// The chromaticities attribute names the frame's primaries.
    #[test]
    fn chromaticities_name_the_primaries() {
        let mut l = layout(false, ExrDepth::Half, ExrCompression::None);
        l.chromaticities = chromaticities(filmcraft_color::Gamut::Bt2020);
        let f = exr_linear(&[0.5; 4], 1, 1, l).unwrap();
        let at = find(&f, b"chromaticities\0chromaticities\0") + 30;
        assert_eq!(le32(&f, at), 32);
        let v: Vec<f32> = (0..8).map(|i| f32::from_bits(le32(&f, at + 4 + i * 4))).collect();
        assert_eq!(v, [0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290]);
        assert_eq!(chromaticities(filmcraft_color::Gamut::Bt709)[0], [0.64, 0.33]);
    }

    #[test]
    fn float24_rounds_and_keeps_infinities() {
        assert_eq!(float24(1.0), 1.0f32.to_bits() >> 8);
        // the dropped byte rounds to nearest
        assert_eq!(float24(f32::from_bits(0x3F80_0080)), 0x3F_8001);
        assert_eq!(float24(f32::from_bits(0x3F80_007F)), 0x3F_8000);
        // the largest finite float is not rounded up to infinity
        assert_eq!(float24(f32::MAX), f32::MAX.to_bits() >> 8);
        assert_eq!(float24(f32::INFINITY), f32::INFINITY.to_bits() >> 8);
        // a NaN whose payload is in the dropped bits stays a NaN
        let nan = float24(f32::from_bits(0x7F80_0001));
        assert!(f32::from_bits(nan << 8).is_nan());
    }

    /// Undo our PXR24 chunk: planes of differences back to samples (`size` 2 for half, 3 for the
    /// 24 bits of a float), in the plain layout.
    fn unpxr24(planes: &[u8], w: usize, lines: usize, nch: usize, size: usize) -> Vec<u8> {
        let mut out = Vec::new();
        let mut at = 0;
        for _ in 0..lines {
            for _ in 0..nch {
                let mut acc = 0u32;
                for x in 0..w {
                    let mut d = 0u32;
                    for k in 0..size {
                        d = d << 8 | u32::from(planes[at + k * w + x]);
                    }
                    acc = acc.wrapping_add(d);
                    if size == 2 {
                        out.extend_from_slice(&(acc as u16).to_le_bytes());
                    } else {
                        out.extend_from_slice(&((acc & 0xFF_FFFF) << 8).to_le_bytes());
                    }
                }
                at += size * w;
            }
        }
        assert_eq!(at, planes.len());
        out
    }

    /// PXR24 keeps half floats exactly and 32-bit floats to 24 bits, in 16-line chunks that
    /// shrink the file.
    #[test]
    fn pxr24_chunks_hold_the_picture() {
        let (px, w, h) = float_picture();
        for depth in [ExrDepth::Half, ExrDepth::Float] {
            let plain = exr_linear(&px, w, h, layout(true, depth, ExrCompression::None)).unwrap();
            let lines: Vec<&[u8]> = chunks(&plain, h as usize).into_iter().map(|(_, d)| d).collect();
            let f = exr_linear(&px, w, h, layout(true, depth, ExrCompression::Pxr24)).unwrap();
            let cmp = find(&f, b"compression\0compression\0") + 24;
            assert_eq!(f[cmp + 4], 5);
            assert!(f.len() < plain.len(), "{depth:?}: {} vs {}", f.len(), plain.len());
            for (k, (y, data)) in chunks(&f, (h as usize).div_ceil(16)).into_iter().enumerate() {
                assert_eq!(y as usize, k * 16);
                let n = 16.min(h as usize - k * 16);
                let want: Vec<u8> = lines[k * 16..k * 16 + n].concat();
                let got = if data.len() == want.len() {
                    data.to_vec()
                } else {
                    unpxr24(&crate::deflate::tests::inflate(data).unwrap(), w as usize, n, 4, if depth == ExrDepth::Half { 2 } else { 3 })
                };
                if depth == ExrDepth::Half {
                    assert_eq!(got, want, "chunk {k}");
                } else {
                    for (g, w) in got.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0) {
                        let (g, w) = (f32::from_le_bytes(*g), f32::from_le_bytes(*w));
                        assert!((g - w).abs() <= w.abs() * 2e-5, "chunk {k}: {g} vs {w}");
                    }
                }
            }
        }
    }

    /// ffmpeg (external test oracle) reads our 32-bit float files as we wrote them, PXR24's to
    /// 24 bits, and our half PXR24 files exactly.
    #[test]
    fn float_and_pxr24_exr_decode_in_ffmpeg() {
        let Some(ffmpeg) = filmcraft_testkit::ffmpeg_or_skip("float OpenEXR stills") else { return };
        let (px, w, h) = float_picture();
        let dir = filmcraft_testkit::fixtures::workspace_root().join("target").join("export-tests").join(format!("exr-float-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = (w * h) as usize;
        for depth in ExrDepth::ALL {
            for c in ExrCompression::ALL {
                let path = dir.join(format!("{depth:?}-{c:?}.exr"));
                std::fs::write(&path, exr_linear(&px, w, h, layout(true, depth, c)).unwrap()).unwrap();
                // half files are read back as halves: ffmpeg's conversion to 32-bit floats clips
                let half_out = depth == ExrDepth::Half;
                let out = std::process::Command::new(&ffmpeg)
                    .args(["-v", "error", "-i"])
                    .arg(&path)
                    .args(["-f", "rawvideo", "-pix_fmt", if half_out { "gbrapf16le" } else { "gbrapf32le" }, "-"])
                    .output()
                    .unwrap();
                assert!(out.status.success(), "{depth:?} {c:?}: {}", String::from_utf8_lossy(&out.stderr));
                assert_eq!(out.stdout.len(), n * if half_out { 8 } else { 16 }, "{depth:?} {c:?}");
                let plane = |p: usize, i: usize| {
                    if half_out {
                        half_value(u16::from_le_bytes(out.stdout[(p * n + i) * 2..][..2].try_into().unwrap()))
                    } else {
                        f32::from_le_bytes(out.stdout[(p * n + i) * 4..][..4].try_into().unwrap())
                    }
                };
                for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
                    // gbrap planes: G, B, R, A
                    for (k, slot) in [1, 2, 0, 3].into_iter().enumerate() {
                        let (got, want) = (plane(k, i), p[slot]);
                        let tol = match (depth, c) {
                            (ExrDepth::Float, ExrCompression::Pxr24) => want.abs() * 2e-5,
                            (ExrDepth::Float, _) => 0.0,
                            (ExrDepth::Half, _) => want.abs() * 1e-3 + 1e-4,
                        };
                        assert!((got - want).abs() <= tol, "{depth:?} {c:?} pixel {i} plane {k}: {got} vs {want}");
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_sizes_are_errors() {
        for depth in ExrDepth::ALL {
            let l = layout(true, depth, ExrCompression::Pxr24);
            assert!(exr_linear(&[0.0; 4], u32::MAX, u32::MAX, l).is_err());
            assert!(exr_linear(&[0.0; 7], 2, 1, l).is_err());
            assert!(exr_linear(&[], 0, 1, l).is_err());
        }
        assert!(dpx(&[0; 7], 2, 1).is_err());
        assert!(dpx(&[], 0, 0).is_err());
        assert!(tga(&[0; 4], 1, 2, false).is_err());
        assert!(tga(&[0; 4], u32::MAX, u32::MAX, false).is_err());
        assert!(rgb(&[0; 3], 1, 1).is_err());
        assert!(dpx(&[0; 4], u32::MAX, u32::MAX).is_err());
        for c in ExrCompression::ALL {
            assert!(exr(&[0; 4], u32::MAX, u32::MAX, true, c).is_err());
            assert!(exr(&[0; 7], 2, 1, false, c).is_err());
            assert!(exr(&[], 0, 1, false, c).is_err());
        }
    }
}

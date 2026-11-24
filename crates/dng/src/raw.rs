//! The raw image data of a DNG: finding the main raw IFD, and reading its strips or tiles
//! (uncompressed or lossless JPEG) into one array of 16-bit samples.

use crate::tiff::{Ifd, Tiff, tags};
use crate::{Error, Result, ljpeg};

/// Largest raw image decoded, in samples (width × height × samples per pixel).
pub const MAX_SAMPLES: usize = 1 << 27;
/// Decoded tiles are padded to tile boundaries, and a lossless JPEG may be coded a little larger
/// than its chunk ([`decode_budget`]), so the samples decoding may allocate can exceed the useful
/// image. Keep that temporary work within a quarter more than a complete decoded raw frame.
const MAX_DECODED_TILE_SAMPLES: usize = MAX_SAMPLES + MAX_SAMPLES / 4;
/// Most bits a lossless JPEG spends on one sample: a Huffman code of at most 16 bits and at most
/// 16 extra bits.
const MAX_LJPEG_BITS_PER_SAMPLE: usize = 32;

/// Photometric interpretations of DNG raw data.
pub const PHOTOMETRIC_CFA: u64 = 32803;
pub const PHOTOMETRIC_LINEAR_RAW: u64 = 34892;

/// The layout of a raw IFD's image data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawLayout {
    pub width: usize,
    pub height: usize,
    pub samples_per_pixel: usize,
    pub bits: u32,
    /// 1 uncompressed, 7 lossless JPEG.
    pub compression: u64,
    pub photometric: u64,
}

/// Raw samples: `width` × `height` pixels of `samples_per_pixel` interleaved 16-bit samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawImage {
    pub width: usize,
    pub height: usize,
    pub samples_per_pixel: usize,
    pub bits: u32,
    pub data: Vec<u16>,
}

fn uint_or(t: &Tiff, ifd: &Ifd, tag: u16, default: u64) -> Result<u64> {
    match ifd.get(tag) {
        Some(e) => t.uint(e),
        None => Ok(default),
    }
}

/// The main raw image IFD: the largest full-resolution (NewSubFileType 0) IFD whose photometric
/// interpretation is CFA or LinearRaw.
pub fn find_raw_ifd<'i>(t: &Tiff, ifds: &'i [Ifd]) -> Result<&'i Ifd> {
    let mut best: Option<(&Ifd, u64)> = None;
    for ifd in ifds {
        let Ok(kind) = uint_or(t, ifd, tags::NEW_SUBFILE_TYPE, 0) else { continue };
        let Ok(photometric) = uint_or(t, ifd, tags::PHOTOMETRIC, 0) else { continue };
        if kind != 0 || !matches!(photometric, PHOTOMETRIC_CFA | PHOTOMETRIC_LINEAR_RAW) {
            continue;
        }
        let w = uint_or(t, ifd, tags::IMAGE_WIDTH, 0).unwrap_or(0);
        let h = uint_or(t, ifd, tags::IMAGE_LENGTH, 0).unwrap_or(0);
        let area = w.saturating_mul(h);
        if best.is_none_or(|(_, a)| area > a) {
            best = Some((ifd, area));
        }
    }
    best.map(|(i, _)| i).ok_or(Error::Unsupported("no raw image in this DNG".into()))
}

/// Read and check the layout tags of a raw IFD.
pub fn layout(t: &Tiff, ifd: &Ifd) -> Result<RawLayout> {
    let width = uint_or(t, ifd, tags::IMAGE_WIDTH, 0)?;
    let height = uint_or(t, ifd, tags::IMAGE_LENGTH, 0)?;
    let spp = uint_or(t, ifd, tags::SAMPLES_PER_PIXEL, 1)?;
    let photometric = uint_or(t, ifd, tags::PHOTOMETRIC, 0)?;
    let compression = uint_or(t, ifd, tags::COMPRESSION, 1)?;
    let bits = match ifd.get(tags::BITS_PER_SAMPLE) {
        Some(e) => {
            let b = t.uints(e)?;
            let first = *b.first().ok_or(Error::Malformed("empty BitsPerSample"))?;
            if b.iter().any(|&x| x != first) {
                return Err(Error::Unsupported("different bit depths per sample".into()));
            }
            first
        }
        None => 1,
    };
    if width == 0 || height == 0 || width > 65_535 || height > 65_535 {
        return Err(Error::Malformed("raw image size"));
    }
    if !(1..=4).contains(&spp) {
        return Err(Error::Unsupported(format!("{spp} samples per pixel")));
    }
    if photometric == PHOTOMETRIC_CFA && spp != 1 {
        return Err(Error::Malformed("a CFA image with more than one sample per pixel"));
    }
    if !(1..=16).contains(&bits) {
        return Err(Error::Unsupported(format!("{bits}-bit raw samples")));
    }
    if uint_or(t, ifd, tags::SAMPLE_FORMAT, 1)? != 1 {
        return Err(Error::Unsupported("floating-point raw data".into()));
    }
    if spp > 1 && uint_or(t, ifd, tags::PLANAR_CONFIGURATION, 1)? != 1 {
        return Err(Error::Unsupported("planar raw data".into()));
    }
    match compression {
        1 => {
            if uint_or(t, ifd, tags::PREDICTOR, 1)? != 1 {
                return Err(Error::Unsupported("a TIFF predictor on uncompressed raw data".into()));
            }
        }
        7 => {}
        8 => return Err(Error::Unsupported("deflate-compressed DNG".into())),
        34892 => return Err(Error::Unsupported("lossy JPEG DNG".into())),
        52546 => return Err(Error::Unsupported("JPEG XL DNG".into())),
        c => return Err(Error::Unsupported(format!("DNG compression {c}"))),
    }
    Ok(RawLayout { width: width as usize, height: height as usize, samples_per_pixel: spp as usize, bits: bits as u32, compression, photometric })
}

/// One strip or tile: where it sits in the image and its bytes.
struct Chunk<'a> {
    x: usize,
    y: usize,
    /// Pixel size of the chunk inside the image's rows (tiles are whole even at the image edge;
    /// a last strip ends at the image).
    w: usize,
    h: usize,
    /// Rows a coder may still have written (a last strip coded at full RowsPerStrip).
    coded_h: usize,
    bytes: &'a [u8],
}

fn chunks<'a>(t: &Tiff<'a>, ifd: &Ifd, l: &RawLayout) -> Result<Vec<Chunk<'a>>> {
    let (offs, counts, tw, th) = if let (Some(o), Some(c)) = (ifd.get(tags::TILE_OFFSETS), ifd.get(tags::TILE_BYTE_COUNTS)) {
        let tw = uint_or(t, ifd, tags::TILE_WIDTH, 0)? as usize;
        let th = uint_or(t, ifd, tags::TILE_LENGTH, 0)? as usize;
        if tw == 0 || th == 0 || tw > 65_535 || th > 65_535 {
            return Err(Error::Malformed("tile size"));
        }
        (t.uints(o)?, t.uints(c)?, tw, th)
    } else if let (Some(o), Some(c)) = (ifd.get(tags::STRIP_OFFSETS), ifd.get(tags::STRIP_BYTE_COUNTS)) {
        let rows = (uint_or(t, ifd, tags::ROWS_PER_STRIP, l.height as u64)? as usize).clamp(1, l.height);
        (t.uints(o)?, t.uints(c)?, l.width, rows)
    } else {
        return Err(Error::Malformed("no strips or tiles"));
    };
    let across = l.width.div_ceil(tw);
    let down = l.height.div_ceil(th);
    let n = across * down;
    if offs.len() < n || counts.len() < n {
        return Err(Error::Malformed("fewer strips or tiles than the image needs"));
    }
    let counts = if ifd.get(tags::TILE_OFFSETS).is_none() { understated_strip(t, ifd, l, &offs, counts) } else { counts };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = ((i % across) * tw, (i / across) * th);
        let start = usize::try_from(offs[i]).map_err(|_| Error::Truncated)?;
        let len = usize::try_from(counts[i]).map_err(|_| Error::Truncated)?;
        let end = start.checked_add(len).ok_or(Error::Truncated)?;
        let bytes = t.data.get(start..end).ok_or(Error::Truncated)?;
        // strips end at the image; tiles are coded whole
        let h = if ifd.get(tags::TILE_OFFSETS).is_some() { th } else { th.min(l.height - y) };
        out.push(Chunk { x, y, w: tw, h, coded_h: th, bytes });
    }
    Ok(out)
}

/// The strip byte counts to read, trusting the file over `StripByteCounts` for one case
/// (craft-libs 60f7c5d74f and 786c5571f0, storytold/craft-libs#2040): an uncompressed image of one
/// 8- or 16-bit sample per pixel in a single strip whose byte count is exactly `ActiveArea`
/// columns (starting at column 0, ending inside the image) by `ImageLength` rows, while the file
/// holds the whole `ImageWidth` by `ImageLength` raster from the strip's offset on. DJI Mavic 3
/// DNGs count the 5280 active columns for a strip of full 5376-wide rows that ends at the end of
/// the file, so reading only the counted bytes refused a complete image. Every other short
/// strip is still refused, and the strip never reaches past the end of the file.
fn understated_strip(t: &Tiff, ifd: &Ifd, l: &RawLayout, offs: &[u64], counts: Vec<u64>) -> Vec<u64> {
    let ([offset], [count]) = (offs, &counts[..]) else { return counts };
    if l.compression != 1 || l.samples_per_pixel != 1 || !matches!(l.bits, 8 | 16) {
        return counts;
    }
    let Some(area) = ifd.get(tags::ACTIVE_AREA).and_then(|e| t.uints(e).ok()) else { return counts };
    let [top, left, bottom, right] = area[..] else { return counts };
    let (width, height) = (l.width as u64, l.height as u64);
    // an area reaching below the image is refused when developed, so it explains nothing here
    if left != 0 || bottom <= top || bottom > height || right == 0 || right >= width {
        return counts;
    }
    let bytes_of = |columns: u64| columns.checked_mul(height)?.checked_mul(u64::from(l.bits / 8));
    let (Some(counted), Some(needed)) = (bytes_of(right), bytes_of(width)) else { return counts };
    let fits = offset.checked_add(needed).is_some_and(|end| end <= t.data.len() as u64);
    if *count == counted && fits { vec![needed] } else { counts }
}

/// Unpack `w` × `h` × `spp` uncompressed samples of `bits` bits: 8 and 16-bit samples as bytes /
/// words in the file's byte order, other depths packed most significant bit first with each row
/// starting on a byte boundary (TIFF 6.0 §7, DNG BitsPerSample).
pub fn unpack(bytes: &[u8], w: usize, h: usize, spp: usize, bits: u32, big_endian: bool) -> Result<Vec<u16>> {
    // zero-bit samples need no input, so a caller could ask for any number of them; past 16 bits
    // they do not fit the samples
    if !(1..=16).contains(&bits) {
        return Err(Error::Malformed("bits per sample outside 1 to 16"));
    }
    let per_row = w.checked_mul(spp).ok_or(Error::TooLarge)?;
    let samples = per_row.checked_mul(h).filter(|&n| n <= MAX_SAMPLES).ok_or(Error::TooLarge)?;
    let row_bytes = per_row.checked_mul(bits as usize).ok_or(Error::TooLarge)?.div_ceil(8);
    let need = row_bytes.checked_mul(h).ok_or(Error::TooLarge)?;
    if bytes.len() < need {
        return Err(Error::Truncated);
    }
    let mut out = Vec::with_capacity(samples);
    for row in bytes.chunks_exact(row_bytes.max(1)).take(h) {
        match bits {
            8 => out.extend(row.iter().map(|&b| u16::from(b))),
            16 => out.extend(row.as_chunks::<2>().0.iter().map(|&b| if big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })),
            _ => {
                let mut acc: u32 = 0;
                let mut n: u32 = 0;
                let mut it = row.iter();
                for _ in 0..per_row {
                    while n < bits {
                        acc = (acc << 8) | u32::from(*it.next().ok_or(Error::Truncated)?);
                        n += 8;
                    }
                    n -= bits;
                    out.push(((acc >> n) & ((1 << bits) - 1)) as u16);
                    acc &= (1 << n) - 1;
                }
            }
        }
    }
    Ok(out)
}

/// The most samples decoding chunk `c` may allocate: its samples when uncompressed; for lossless
/// JPEG its coded rows with an eighth of slack (the frame may be coded a little larger than its
/// chunk), not a fixed allowance that lets every tiny tile decode a large frame.
fn decode_budget(c: &Chunk, l: &RawLayout) -> Option<usize> {
    let spp = l.samples_per_pixel;
    match l.compression {
        1 => c.w.checked_mul(c.h)?.checked_mul(spp),
        _ => {
            let coded = c.w.checked_mul(c.coded_h)?.checked_mul(spp)?;
            coded.checked_add(coded / 8)?.checked_add(64)
        }
    }
}

/// The decode work chunk `c` is charged against the aggregate budget: what decoding may allocate
/// (`budget`) or its own `w` × `h` × `spp` output, whichever is larger, and for lossless JPEG at
/// least its length in samples of the longest code ([`MAX_LJPEG_BITS_PER_SAMPLE`]), which the
/// budget of a real chunk (its samples and a few hundred bytes of tables) already covers. The
/// JPEG decoder reads a chunk's bytes (the segments before the scan, the entropy data, the search
/// for restart markers) whatever frame it codes, so chunks that point at the same bytes
/// (identical tiles stored once, which real files do and which still decode) each pay for those
/// bytes: a payload shared by many tiny tiles costs what it would cost stored once for each.
fn decode_charge(c: &Chunk, l: &RawLayout, budget: usize) -> Option<usize> {
    let n = c.w.checked_mul(c.h)?.checked_mul(l.samples_per_pixel)?;
    let read = if l.compression == 1 { 0 } else { c.bytes.len().saturating_mul(8).div_ceil(MAX_LJPEG_BITS_PER_SAMPLE) };
    Some(budget.max(n).max(read))
}

/// Decode one chunk into `w` × `h` × `spp` samples, allocating at most `budget` samples.
fn decode_chunk(c: &Chunk, l: &RawLayout, big_endian: bool, budget: usize) -> Result<Vec<u16>> {
    let need = c.w * c.h * l.samples_per_pixel;
    match l.compression {
        1 => unpack(c.bytes, c.w, c.h, l.samples_per_pixel, l.bits, big_endian),
        _ => {
            // the JPEG's samples fill the chunk in raster order (its own width × components may
            // differ from the chunk's, e.g. two components of half the width); the samples past
            // `need` (a last strip coded at full RowsPerStrip) are dropped
            let d = ljpeg::decode(c.bytes, budget)?;
            if d.samples.len() < need {
                return Err(Error::Malformed("lossless JPEG smaller than its tile or strip"));
            }
            let mut s = d.samples;
            s.truncate(need);
            Ok(s)
        }
    }
}

/// Read the raw image of `ifd`.
pub fn read(t: &Tiff, ifd: &Ifd) -> Result<RawImage> {
    let l = layout(t, ifd)?;
    let spp = l.samples_per_pixel;
    let total = l.width.checked_mul(l.height).and_then(|v| v.checked_mul(spp)).ok_or(Error::TooLarge)?;
    if total > MAX_SAMPLES {
        return Err(Error::TooLarge);
    }
    let chunks = chunks(t, ifd, &l)?;
    // every sample takes at least one bit of file (a Huffman code is at least one bit long)
    let coded: usize = chunks.iter().map(|c| c.bytes.len()).fold(0usize, usize::saturating_add);
    if total.div_ceil(8) > coded {
        return Err(Error::Truncated);
    }
    // what decoding may allocate and read, chunk by chunk, within one decoded frame's worth
    let mut budgets = Vec::with_capacity(chunks.len());
    let mut decoded_samples = 0usize;
    for c in &chunks {
        let n = c.w.checked_mul(c.h).and_then(|v| v.checked_mul(spp)).ok_or(Error::TooLarge)?;
        let budget = decode_budget(c, &l).filter(|&b| b <= MAX_SAMPLES).ok_or(Error::TooLarge)?;
        let charge = decode_charge(c, &l, budget).ok_or(Error::TooLarge)?;
        decoded_samples = decoded_samples.checked_add(charge).ok_or(Error::TooLarge)?;
        if n > MAX_SAMPLES || decoded_samples > MAX_DECODED_TILE_SAMPLES {
            return Err(Error::TooLarge);
        }
        if n.div_ceil(8) > c.bytes.len() {
            return Err(Error::Truncated);
        }
        budgets.push(budget);
    }
    let mut data = vec![0u16; total];
    let row_len = l.width * spp;
    for (c, &budget) in chunks.iter().zip(&budgets) {
        // Copy each padded tile immediately. Retaining every decoded tile until assembly lets
        // repeated ranges multiply memory beyond the final raw frame's bounded allocation.
        let d = decode_chunk(c, &l, t.big_endian, budget)?;
        let cw = c.w.min(l.width - c.x);
        let ch = c.h.min(l.height - c.y);
        for r in 0..ch {
            let src = d.get(r * c.w * spp..(r * c.w + cw) * spp).ok_or(Error::Truncated)?;
            let at = (c.y + r) * row_len + c.x * spp;
            data.get_mut(at..at + cw * spp).ok_or(Error::Truncated)?.copy_from_slice(src);
        }
    }
    Ok(RawImage { width: l.width, height: l.height, samples_per_pixel: spp, bits: l.bits, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_or_oversized_bit_depths_are_rejected_before_allocating() {
        // zero-bit samples consume no input: one byte used to yield a whole picture of zeros
        assert!(matches!(unpack(&[0], 1 << 20, 1 << 10, 1, 0, false), Err(Error::Malformed(_))));
        assert!(matches!(unpack(&[0; 64], 4, 1, 1, 17, false), Err(Error::Malformed(_))));
        assert!(matches!(unpack(&[0; 64], 4, 1, 1, 32, false), Err(Error::Malformed(_))));
        assert_eq!(unpack(&[0x12, 0x34, 0x50], 2, 1, 1, 12, true).unwrap(), vec![0x123, 0x450]);
        // dimensions whose products overflow, or exceed the sample limit, fail before allocating
        assert_eq!(unpack(&[], isize::MAX as usize + 1, 1, 1, 2, false), Err(Error::TooLarge));
        assert_eq!(unpack(&[], usize::MAX, 2, 1, 16, false), Err(Error::TooLarge));
        assert_eq!(unpack(&[], 1 << 20, 1 << 8, 1, 1, false), Err(Error::TooLarge));
    }
}

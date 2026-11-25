//! Uncompressed and PackBits YCbCr TIFFs (TIFF 6.0 §21), as ffmpeg writes from YUV sources. The
//! image crate reads YCbCr only inside JPEG compression, so these are decoded here: 8-bit chunky
//! data units of any subsampling (1, 2 or 4 each way), converted to RGB with the file's
//! YCbCrCoefficients and ReferenceBlackWhite (§20), chroma repeated across each data unit.
//!
//! A YCbCrCoefficients or ReferenceBlackWhite field that cannot be used (unreadable, too few
//! values, a weight outside (0, 1), a black code equal to its white) falls back to that field's
//! TIFF default, as libtiff-based readers do, instead of refusing the picture.
//!
//! # Green: a compatibility choice
//!
//! Red and blue follow the specification (TIFF 6.0 p. 91: `R = Cr·(2 − 2·LumaRed) + Y`,
//! `B = Cb·(2 − 2·LumaBlue) + Y`). Green follows libtiff, which rewrites the specification's
//! `G = (Y − LumaBlue·B − LumaRed·R) / LumaGreen` as
//! `G = Y − (LumaRed·(2 − 2·LumaRed) / LumaGreen)·Cr − (LumaBlue·(2 − 2·LumaBlue) / LumaGreen)·Cb`,
//! adding Y unscaled. The two agree whenever the weights sum to 1, as every standard set does
//! (BT.601, BT.709). For other weights the specification's form tints neutral greys; libtiff's
//! keeps them neutral and matches libtiff-based tools.
//!
//! Every number comes from the file: sizes, strip offsets and counts are checked before use, the
//! picture is capped at [`MAX_PIXELS`], a file too small to hold the picture (PackBits expands at
//! most 64 to 1) is refused before its data units are allocated, the allocations are fallible,
//! and PackBits output stops at the bytes the image needs.

use filmcraft_dng::tiff::{Tiff, tags};

use super::Raster;
use crate::{MediaError, Result};

/// Largest picture decoded (1 GiB of RGBA), as for animated images.
const MAX_PIXELS: u64 = 1 << 28;

const PHOTOMETRIC_YCBCR: u64 = 6;
const COMPRESSION_NONE: u64 = 1;
const COMPRESSION_PACKBITS: u64 = 32773;
const YCBCR_COEFFICIENTS: u16 = 529;
const YCBCR_SUBSAMPLING: u16 = 530;
const REFERENCE_BLACK_WHITE: u16 = 532;
/// Most bytes one byte of PackBits data expands to (two bytes, a run of 128).
const MAX_PACKBITS_EXPANSION: usize = 64;

/// BT.601 luma weights: the TIFF default for YCbCrCoefficients.
const BT601: [f64; 3] = [0.299, 0.587, 0.114];
/// The default ReferenceBlackWhite of 8-bit samples: full-range luma, chroma centred on 128.
const DEFAULT_REFERENCE: [f64; 6] = [0.0, 255.0, 128.0, 255.0, 128.0, 255.0];

/// How one image's 8-bit YCbCr codes map to RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
struct YCbCr {
    /// Luma weights of red, green and blue.
    luma: [f64; 3],
    /// Black and white codes of Y, Cb and Cr.
    reference: [f64; 6],
}

impl YCbCr {
    /// The conversion of these field values: weights must each lie strictly between 0 and 1 and
    /// every black / white pair must differ; otherwise (or when a field is absent) that field's
    /// default is used.
    fn from_fields(coefficients: Option<&[f64]>, reference: Option<&[f64]>) -> Self {
        let luma = coefficients
            .and_then(|v| <[f64; 3]>::try_from(v.get(..3)?).ok())
            .filter(|l| l.iter().all(|c| c.is_finite() && *c > 0.0 && *c < 1.0))
            .unwrap_or(BT601);
        let reference = reference
            .and_then(|v| <[f64; 6]>::try_from(v.get(..6)?).ok())
            .filter(|r| r.iter().all(|v| v.is_finite()) && r.as_chunks::<2>().0.iter().all(|[black, white]| (white - black).abs() > f64::EPSILON))
            .unwrap_or(DEFAULT_REFERENCE);
        YCbCr { luma, reference }
    }

    /// Per-code contributions, so a pixel costs a few additions.
    fn tables(&self) -> Tables {
        let [lr, lg, lb] = self.luma;
        let [y_black, y_white, cb_black, cb_white, cr_black, cr_white] = self.reference;
        // TIFF 6.0 §20: luma to 0–255, chroma to ±127 about its reference black.
        let code = |i: usize| i as f64;
        let y: [f64; 256] = std::array::from_fn(|i| (code(i) - y_black) * 255.0 / (y_white - y_black));
        let cb: [f64; 256] = std::array::from_fn(|i| (code(i) - cb_black) * 127.0 / (cb_white - cb_black));
        let cr: [f64; 256] = std::array::from_fn(|i| (code(i) - cr_black) * 127.0 / (cr_white - cr_black));
        let (kr, kb) = (2.0 - 2.0 * lr, 2.0 - 2.0 * lb);
        Tables { y, cr_r: cr.map(|v| kr * v), cr_g: cr.map(|v| -lr * kr * v / lg), cb_g: cb.map(|v| -lb * kb * v / lg), cb_b: cb.map(|v| kb * v) }
    }
}

/// Per-code contributions of one [`YCbCr`] conversion.
struct Tables {
    /// Scaled luma of each Y code.
    y: [f64; 256],
    /// Red and green contributions of each Cr code.
    cr_r: [f64; 256],
    cr_g: [f64; 256],
    /// Green and blue contributions of each Cb code.
    cb_g: [f64; 256],
    cb_b: [f64; 256],
}

impl Tables {
    /// R, G and B of one pixel, rounded and clamped to 0–255.
    fn rgb(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let (y, cb, cr) = (usize::from(y), usize::from(cb), usize::from(cr));
        let yv = self.y[y];
        // Green adds Y unscaled, as libtiff does (see "Green: a compatibility choice").
        let r = yv + self.cr_r[cr];
        let g = yv + self.cr_g[cr] + self.cb_g[cb];
        let b = yv + self.cb_b[cb];
        [r, g, b].map(|v| v.round().clamp(0.0, 255.0) as u8)
    }
}

/// Whether `bytes` is a TIFF whose first image is uncompressed or PackBits YCbCr (the ones this
/// module decodes instead of the image crate).
pub fn sniff(bytes: &[u8]) -> bool {
    let Ok((t, first)) = Tiff::new(bytes) else { return false };
    let Ok((ifd, _)) = t.read_ifd(first, 0) else { return false };
    let num = |tag| ifd.get(tag).and_then(|e| t.uint(e).ok());
    num(tags::PHOTOMETRIC) == Some(PHOTOMETRIC_YCBCR) && matches!(num(tags::COMPRESSION).unwrap_or(COMPRESSION_NONE), COMPRESSION_NONE | COMPRESSION_PACKBITS)
}

/// A buffer of `len` bytes of `fill`, or an error when the memory is not there.
fn filled(len: usize, fill: u8, name: &str) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| MediaError::Unsupported(format!("{name}: YCbCr TIFF: not enough memory for a {len}-byte picture")))?;
    v.resize(len, fill);
    Ok(v)
}

/// Decode the first image of a YCbCr TIFF (see [`sniff`]) to RGBA.
pub fn decode(name: &str, bytes: &[u8]) -> Result<Raster> {
    let bad = |what: &str| MediaError::Decode(format!("{name}: YCbCr TIFF: {what}"));
    let unsupported = |what: &str| MediaError::Unsupported(format!("{name}: YCbCr TIFF with {what}"));
    let (t, first) = Tiff::new(bytes).map_err(|_| bad("not a TIFF"))?;
    let (ifd, _) = t.read_ifd(first, 0).map_err(|_| bad("unreadable image directory"))?;
    let uints = |tag: u16| -> Result<Option<Vec<u64>>> { ifd.get(tag).map(|e| t.uints(e).map_err(|_| bad("unreadable field"))).transpose() };
    let one = |tag: u16| -> Result<Option<u64>> { Ok(uints(tag)?.and_then(|v| v.first().copied())) };

    let width = one(tags::IMAGE_WIDTH)?.ok_or_else(|| bad("no ImageWidth"))?;
    let height = one(tags::IMAGE_LENGTH)?.ok_or_else(|| bad("no ImageLength"))?;
    if width == 0 || height == 0 {
        return Err(bad("empty image"));
    }
    if width.saturating_mul(height) > MAX_PIXELS {
        return Err(unsupported("more than 268 million pixels"));
    }
    if one(tags::SAMPLES_PER_PIXEL)?.unwrap_or(1) != 3 {
        return Err(unsupported("other than three samples per pixel"));
    }
    if uints(tags::BITS_PER_SAMPLE)?.unwrap_or_else(|| vec![1]).iter().any(|&b| b != 8) {
        return Err(unsupported("other than 8 bits per sample"));
    }
    if one(tags::PLANAR_CONFIGURATION)?.unwrap_or(1) != 1 {
        return Err(unsupported("separate planes"));
    }
    if ifd.get(tags::TILE_OFFSETS).is_some() {
        return Err(unsupported("tiles"));
    }
    let compression = one(tags::COMPRESSION)?.unwrap_or(COMPRESSION_NONE);
    let packed = compression == COMPRESSION_PACKBITS;
    let sub = uints(YCBCR_SUBSAMPLING)?.unwrap_or_else(|| vec![2, 2]);
    let (sh, sv) = match sub.as_slice() {
        [h, v, ..] if [1, 2, 4].contains(h) && [1, 2, 4].contains(v) => (*h, *v),
        _ => return Err(bad("invalid YCbCrSubSampling")),
    };
    // §21: the data units cover the image rounded up to whole units.
    let (units_x, units_y) = (width.div_ceil(sh), height.div_ceil(sv));
    let unit_len = sh * sv + 2;
    let need = usize::try_from(units_x.saturating_mul(units_y).saturating_mul(unit_len)).map_err(|_| bad("too large"))?;
    // uncompressed units are the file's own bytes, and PackBits expands at most 64 to 1: a file
    // that cannot hold the picture is refused before the data units are allocated
    if need > bytes.len().saturating_mul(if packed { MAX_PACKBITS_EXPANSION } else { 1 }) {
        return Err(bad("image data shorter than the image"));
    }

    let offsets = uints(tags::STRIP_OFFSETS)?.ok_or_else(|| bad("no StripOffsets"))?;
    let counts = uints(tags::STRIP_BYTE_COUNTS)?.ok_or_else(|| bad("no StripByteCounts"))?;
    let mut units: Vec<u8> = Vec::new();
    units.try_reserve_exact(need).map_err(|_| unsupported("a picture too large for the memory available"))?;
    let mut strip_bytes = 0usize;
    for (&off, &len) in offsets.iter().zip(&counts) {
        if units.len() >= need {
            break;
        }
        let strip = usize::try_from(off)
            .ok()
            .zip(usize::try_from(len).ok())
            .and_then(|(o, l)| bytes.get(o..o.checked_add(l)?))
            .ok_or_else(|| bad("strip outside the file"))?;
        // the strips of one image do not overlap, so together they cannot exceed the file
        strip_bytes = strip_bytes.saturating_add(strip.len());
        if strip_bytes > bytes.len() {
            return Err(bad("strips overlap"));
        }
        let left = need - units.len();
        match compression {
            COMPRESSION_NONE => units.extend_from_slice(strip.get(..left).unwrap_or(strip)),
            _ => unpackbits(strip, &mut units, need),
        }
    }
    if units.len() < need {
        return Err(bad("image data shorter than the image"));
    }

    // an unusable field is its default (libtiff's reading), not an error
    let field = |tag: u16| ifd.get(tag).and_then(|e| t.floats(e).ok());
    let (coefficients, reference) = (field(YCBCR_COEFFICIENTS), field(REFERENCE_BLACK_WHITE));
    let tables = YCbCr::from_fields(coefficients.as_deref(), reference.as_deref()).tables();

    let (w, h) = (width as usize, height as usize);
    let (sh, sv, units_x) = (sh as usize, sv as usize, units_x as usize);
    let len = w.checked_mul(h).and_then(|n| n.checked_mul(4)).ok_or_else(|| bad("too large"))?;
    let mut rgba = filled(len, 255, name)?;
    for (i, unit) in units.chunks_exact(sh * sv + 2).enumerate() {
        let (ux, uy) = (i % units_x, i / units_x);
        let (luma, chroma) = unit.split_at(sh * sv);
        let [cb, cr] = <[u8; 2]>::try_from(chroma).map_err(|_| bad("data unit"))?;
        for (k, &y) in luma.iter().enumerate() {
            let (x, row) = (ux * sh + k % sh, uy * sv + k / sh);
            if x < w && row < h {
                let at = (row * w + x) * 4;
                if let Some(px) = rgba.get_mut(at..at + 3) {
                    px.copy_from_slice(&tables.rgb(y, cb, cr));
                }
            }
        }
    }
    Ok(Raster { width: width as u32, height: height as u32, rgba, has_alpha: false })
}

/// PackBits (TIFF 6.0 §9) from `src` appended to `out`, stopping at `limit` bytes of output or at
/// a run that the data ends inside.
fn unpackbits(src: &[u8], out: &mut Vec<u8>, limit: usize) {
    let mut p = 0;
    while out.len() < limit {
        let Some(&n) = src.get(p) else { return };
        let n = n as i8;
        p += 1;
        if n >= 0 {
            let len = n as usize + 1;
            let Some(lit) = p.checked_add(len).and_then(|end| src.get(p..end)) else { return };
            out.extend_from_slice(lit.get(..len.min(limit - out.len())).unwrap_or(lit));
            p += len;
        } else if n != -128 {
            let Some(&b) = src.get(p) else { return };
            let len = (1 - isize::from(n)) as usize;
            out.extend(std::iter::repeat_n(b, len.min(limit - out.len())));
            p += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_runs_literals_and_noops() {
        let mut out = Vec::new();
        // literal 2, a no-op, a run of 3, then a literal cut short by the end of the data
        unpackbits(&[1, 10, 11, 0x80, 0xFE, 7, 4, 1], &mut out, 100);
        assert_eq!(out, [10, 11, 7, 7, 7]);
        let mut capped = Vec::new();
        unpackbits(&[0xFE, 7, 0xFE, 8], &mut capped, 4);
        assert_eq!(capped, [7, 7, 7, 8]);
        // the longest run, 128 bytes from two, appended to what is there
        let mut out = vec![1];
        unpackbits(&[0x81, 9], &mut out, 1000);
        assert_eq!((out.len(), out[1], out[128]), (129, 9, 9));
    }

    fn convert(c: &YCbCr, [y, cb, cr]: [u8; 3]) -> [u8; 3] {
        c.tables().rgb(y, cb, cr)
    }

    #[test]
    fn jfif_codes_give_known_colours() {
        let c = YCbCr::from_fields(None, None);
        assert_eq!(convert(&c, [0, 128, 128]), [0, 0, 0]);
        assert_eq!(convert(&c, [255, 128, 128]), [255, 255, 255]);
        assert_eq!(convert(&c, [128, 128, 128]), [128, 128, 128]);
        assert_eq!(convert(&c, [76, 85, 255]), [254, 0, 0]);
        assert_eq!(convert(&c, [150, 44, 21]), [0, 255, 1]);
        assert_eq!(convert(&c, [29, 255, 107]), [0, 0, 254]);
        // studio range: Y 16–235, chroma 128 ± 112
        let studio = YCbCr::from_fields(None, Some(&[16.0, 235.0, 128.0, 240.0, 128.0, 240.0]));
        assert_eq!(convert(&studio, [16, 128, 128]), [0, 0, 0]);
        assert_eq!(convert(&studio, [235, 128, 128]), [255, 255, 255]);
    }

    #[test]
    fn weights_that_do_not_sum_to_one_keep_greys_neutral() {
        // the specification's green equation turned this grey into (128, 64, 128)
        let c = YCbCr::from_fields(Some(&[0.4, 0.4, 0.4]), None);
        for y in [0u8, 64, 128, 200, 255] {
            assert_eq!(convert(&c, [y, 128, 128]), [y, y, y]);
        }
    }

    #[test]
    fn unusable_fields_fall_back_to_their_defaults() {
        let d = YCbCr::from_fields(None, None);
        assert_eq!((d.luma, d.reference), (BT601, DEFAULT_REFERENCE));
        let c = YCbCr::from_fields(Some(&[0.2126, 0.7152, 0.0722]), Some(&[16.0, 235.0, 128.0, 240.0, 128.0, 240.0]));
        assert_eq!((c.luma, c.reference), ([0.2126, 0.7152, 0.0722], [16.0, 235.0, 128.0, 240.0, 128.0, 240.0]));
        // a weight of 0 or 1, NaN, too few values, a black equal to its white: that field's default
        for bad in [&[0.0, 0.5, 0.5][..], &[0.3, 1.0, 0.1], &[f64::NAN, 0.5, 0.5], &[0.5]] {
            assert_eq!(YCbCr::from_fields(Some(bad), None).luma, BT601, "{bad:?}");
        }
        for bad in [&[16.0, 16.0, 128.0, 240.0, 128.0, 240.0][..], &[0.0; 6], &[0.0, 255.0], &[0.0, f64::INFINITY, 128.0, 255.0, 128.0, 255.0]] {
            assert_eq!(YCbCr::from_fields(None, Some(bad)).reference, DEFAULT_REFERENCE, "{bad:?}");
        }
    }
}

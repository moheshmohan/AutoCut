//! Did the screen change? Two captures, downsampled to grayscale, compared two ways: the
//! mean absolute difference (a page moved) or the share of pixels that changed clearly (a
//! local change — a tile's check mark, the Done button lighting up: about 1% of a picker
//! window, which a whole-frame mean averages away). Cheap, deterministic, and honest enough
//! to catch "the click did nothing".

use crate::ScreenError;

const SMALL: u32 = 480;
/// Mean abs diff above this (0–255 scale) counts as "the screen changed".
const THRESHOLD: f32 = 2.0;
/// A pixel whose luma moved more than this changed clearly (blur and caret blink stay under).
const PIXEL_DELTA: u8 = 24;
/// Clearly-changed pixels above this share (1/2000 = 0.05%) count as "the screen changed".
const CHANGED_SHARE_DIV: u64 = 2000;

fn to_small_gray(png: &[u8]) -> Result<image::GrayImage, ScreenError> {
    let img = image::load_from_memory(png).map_err(|e| ScreenError::Other(format!("cannot decode a capture for comparison: {e}")))?;
    let (w, h) = (img.width().max(1), img.height().max(1));
    let long = u64::from(w.max(h));
    let f = f64::from(SMALL) / long as f64;
    let nw = ((f64::from(w) * f).round() as u32).max(1);
    let nh = ((f64::from(h) * f).round() as u32).max(1);
    Ok(img.resize_exact(nw, nh, image::imageops::FilterType::Triangle).to_luma8())
}

/// Whether two captures differ visually. Identical inputs are `false`; decode failures are
/// errors, never panics.
pub fn changed(a: &[u8], b: &[u8]) -> Result<bool, ScreenError> {
    let (ga, gb) = (to_small_gray(a)?, to_small_gray(b)?);
    let (wa, ha) = (ga.width(), ga.height());
    if wa == 0 || ha == 0 || wa != gb.width() || ha != gb.height() {
        return Ok(true); // different shapes can only mean different screens
    }
    let (mut sum, mut n, mut clear) = (0u64, 0u64, 0u64);
    for (pa, pb) in ga.pixels().zip(gb.pixels()) {
        let d = pa.0[0].abs_diff(pb.0[0]);
        sum = sum.saturating_add(u64::from(d));
        clear += u64::from(d > PIXEL_DELTA);
        n += 1;
    }
    if n == 0 {
        return Ok(true);
    }
    let mean = sum as f32 / n as f32;
    Ok(mean > THRESHOLD || clear.saturating_mul(CHANGED_SHARE_DIV) > n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_with(square_at: Option<(u32, u32)>) -> Vec<u8> {
        let mut img = image::GrayImage::from_pixel(800, 600, image::Luma([100u8]));
        if let Some((x, y)) = square_at {
            for dy in 0..160u32 {
                for dx in 0..160u32 {
                    *img.get_pixel_mut(x + dx, y + dy) = image::Luma([240u8]);
                }
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn identical_frames_are_unchanged() {
        let a = png_with(Some((100, 100)));
        assert!(!changed(&a, &a).unwrap());
    }

    #[test]
    fn a_new_tile_is_a_change() {
        let a = png_with(None);
        let b = png_with(Some((100, 100)));
        assert!(changed(&a, &b).unwrap());
    }

    #[test]
    fn subtle_noise_is_not_a_change() {
        let a = png_with(None);
        let mut img = image::load_from_memory(&a).unwrap().to_luma8();
        // +-1 luma on a few pixels: under threshold
        img.get_pixel_mut(10, 10).0[0] = img.get_pixel(10, 10).0[0].saturating_add(1);
        let mut b = std::io::Cursor::new(Vec::new());
        img.write_to(&mut b, image::ImageFormat::Png).unwrap();
        assert!(!changed(&a, &b.into_inner()).unwrap());
    }

    /// A picker-sized frame (1344x773) with an optional bright patch and a 2-px caret.
    fn picker_frame(patch: Option<(u32, u32, u32)>, caret: bool) -> Vec<u8> {
        let mut img = image::GrayImage::from_pixel(1344, 773, image::Luma([30u8]));
        if let Some((x, y, side)) = patch {
            for dy in 0..side {
                for dx in 0..side {
                    *img.get_pixel_mut(x + dx, y + dy) = image::Luma([220u8]);
                }
            }
        }
        if caret {
            for y in 85..105 {
                for x in 440..442 {
                    *img.get_pixel_mut(x, y) = image::Luma([230u8]);
                }
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn a_small_local_change_counts() {
        // a selection check mark / Done lighting up: ~30x30 of a 1344x773 window; the mean
        // alone (≈0.2) would call this "nothing happened"
        let a = picker_frame(None, false);
        let b = picker_frame(Some((1270, 80, 30)), false);
        assert!(changed(&a, &b).unwrap());
    }

    #[test]
    fn a_blinking_caret_does_not_count() {
        let a = picker_frame(None, false);
        let b = picker_frame(None, true);
        assert!(!changed(&a, &b).unwrap());
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(changed(b"xx", b"yy").is_err());
        assert!(changed(&png_with(None), b"xx").is_err());
    }
}

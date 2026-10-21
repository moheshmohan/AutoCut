//! Still images (PNG, JPEG, GIF, WebP, TIFF, BMP, Targa, DPX, OpenEXR, Radiance HDR, DNG camera raw) as media
//! sources. Animated PNG and GIF files open as movies instead ([`crate::animated`]); here they would
//! show their first frame.

use std::sync::Arc;

use filmcraft_frame::VideoFrame;
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};

use crate::{FrameRequest, MediaError, MediaInfo, MediaKind, MediaSource, Result, VideoStreamInfo};

pub mod dng;
pub mod exr;
pub mod hdr;
mod ycbcr_tiff;

/// Default still duration (Premiere's preference default is 5 seconds).
pub const DEFAULT_STILL_DURATION: Tick = Tick(5 * TICKS_PER_SECOND);

/// Longest side a decoded still keeps, pixels: a larger picture is scaled down to it as it is
/// decoded. A still is held whole in memory (an 8192 × 6144 photo is 192 MiB as RGBA8) and no
/// sequence is wider than 7680, so a 48-megapixel phone photo (8064 × 6048) stays whole while a
/// 200-megapixel one, or a file claiming more, is held at a size the app can afford.
pub const MAX_STILL_SIDE: u32 = 8192;
/// Most bytes the image decoders may allocate for one picture as decoded (before
/// [`MAX_STILL_SIDE`] applies): a header claiming more is refused before anything is allocated.
pub const MAX_DECODE_BYTES: u64 = 512 << 20;
/// Widest and tallest picture the image decoders accept at all.
pub const MAX_DECODE_SIDE: u32 = 65_535;

/// `(w, h)` scaled to fit within [`MAX_STILL_SIDE`] on its long side, the shape kept (each side at
/// least 1); unchanged when it fits.
pub fn capped_size(w: u32, h: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= MAX_STILL_SIDE || long == 0 {
        return (w, h);
    }
    let k = f64::from(MAX_STILL_SIDE) / f64::from(long);
    let side = |v: u32| ((f64::from(v) * k).round() as u32).clamp(1, MAX_STILL_SIDE);
    (side(w), side(h))
}

/// Straight-alpha RGBA8 `w` × `h` scaled down to [`capped_size`] (unchanged when it fits, or
/// when `rgba` is not `w × h × 4` bytes).
pub fn cap_rgba(w: u32, h: u32, rgba: Vec<u8>) -> (u32, u32, Vec<u8>) {
    let (cw, ch) = capped_size(w, h);
    let whole = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(4));
    if (cw, ch) == (w, h) || whole != Some(rgba.len()) {
        return (w, h, rgba);
    }
    match image::RgbaImage::from_raw(w, h, rgba) {
        Some(img) => (cw, ch, image::imageops::resize(&img, cw, ch, image::imageops::FilterType::Triangle).into_raw()),
        // (the length was checked above)
        None => (w, h, Vec::new()),
    }
}

/// Whether `b` (the contents of the file `name`) is a still image we decode.
pub fn sniff(name: &str, b: &[u8]) -> bool {
    crate::dpx::sniff(b)
        || hdr::sniff(b)
        || crate::tga::sniff(name, b)
        || b.starts_with(b"\x89PNG")
        || b.starts_with(&[0xFF, 0xD8, 0xFF])
        || b.starts_with(b"GIF8")
        || (b.len() > 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP")
        || b.starts_with(b"II*\0")
        || b.starts_with(b"MM\0*")
        || b.starts_with(b"BM")
}

/// A decoded picture: straight-alpha RGBA8, top row first.
pub struct Raster {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Whether any pixel is not fully opaque.
    pub has_alpha: bool,
}

/// Decode a still image file (`name` picks Targa, which has no magic number; the content picks
/// every other format).
pub fn decode_raster(name: &str, bytes: &[u8]) -> Result<Raster> {
    if crate::dpx::sniff(bytes) {
        return crate::dpx::decode(name, bytes).map(Raster::capped);
    }
    if crate::tga::sniff(name, bytes) {
        return crate::tga::decode(name, bytes).map(Raster::capped);
    }
    if ycbcr_tiff::sniff(bytes) {
        return ycbcr_tiff::decode(name, bytes).map(Raster::capped);
    }
    let img = decode_image(name, bytes)?;
    let has_alpha = img.color().has_alpha();
    let rgba = img.into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(Raster { width, height, rgba: rgba.into_raw(), has_alpha })
}

impl Raster {
    /// Scaled down to [`MAX_STILL_SIDE`] when larger.
    fn capped(self) -> Self {
        let (width, height, rgba) = cap_rgba(self.width, self.height, self.rgba);
        Self { width, height, rgba, ..self }
    }
}

/// Decode a PNG, JPEG, GIF, WebP, TIFF or BMP with the `image` crate: within
/// [`MAX_DECODE_BYTES`] and [`MAX_DECODE_SIDE`], turned upright by its EXIF `Orientation` (a
/// portrait phone photo is stored sideways with a tag saying so), and scaled down to
/// [`MAX_STILL_SIDE`] when larger.
fn decode_image(name: &str, bytes: &[u8]) -> Result<image::DynamicImage> {
    use image::ImageDecoder as _;
    let err = |e: image::ImageError| MediaError::Decode(format!("{name}: {e}"));
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| MediaError::Decode(format!("{name}: {e}")))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    limits.max_image_width = Some(MAX_DECODE_SIDE);
    limits.max_image_height = Some(MAX_DECODE_SIDE);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(err)?;
    // a damaged EXIF block is no reason to refuse the picture
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder).map_err(err)?;
    img.apply_orientation(orientation);
    let (w, h) = (img.width(), img.height());
    let (cw, ch) = capped_size(w, h);
    if (cw, ch) != (w, h) {
        img = img.resize_exact(cw, ch, image::imageops::FilterType::Triangle);
    }
    Ok(img)
}

pub struct StillSource {
    info: MediaInfo,
    frame: Arc<VideoFrame>,
}

impl StillSource {
    pub fn decode(name: &str, bytes: &[u8]) -> Result<Self> {
        // a DNG is a TIFF too: develop its raw image instead of reading its preview
        if dng::sniff(bytes) {
            return dng::still(name, bytes);
        }
        if hdr::sniff(bytes) {
            return hdr::still(name, bytes);
        }
        let r = decode_raster(name, bytes)?;
        Ok(Self::from_rgba(name, r.width, r.height, r.rgba, r.has_alpha, bytes.len() as u64))
    }

    pub fn from_rgba(name: &str, w: u32, h: u32, rgba: Vec<u8>, has_alpha: bool, file_size: u64) -> Self {
        let codec = name.rsplit('.').next().unwrap_or("image").to_ascii_uppercase();
        let info = MediaInfo {
            name: name.to_string(),
            kind: MediaKind::Still,
            duration: DEFAULT_STILL_DURATION,
            video: Some(VideoStreamInfo {
                width: w,
                height: h,
                frame_rate: FrameRate::FPS_29_97,
                par: (1, 1),
                codec,
                pixel_format: if has_alpha { "RGBA 8-bit".into() } else { "RGB 8-bit".into() },
                color: filmcraft_color::ColorInfo::SRGB_FULL,
                has_alpha,
                bitrate: None,
                hdr: None,
            }),
            audio_streams: Vec::new(),
            container: "Image".into(),
            start_timecode: None,
            file_size: Some(file_size),
        };
        Self { info, frame: Arc::new(VideoFrame::rgba8(w, h, rgba)) }
    }
}

impl MediaSource for StillSource {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn video_frame(&self, _req: FrameRequest) -> Result<Arc<VideoFrame>> {
        Ok(self.frame.clone())
    }
    fn audio(&self, _start: i64, _frames: usize, _sample_rate: u32) -> Result<filmcraft_frame::AudioBuffer> {
        Err(MediaError::NoStream("audio"))
    }
}

#[cfg(test)]
mod tests;

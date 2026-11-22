//! Still images (PNG, JPEG, GIF, WebP, TIFF, BMP, Targa, DPX, OpenEXR, Radiance HDR, DNG camera raw) as media
//! sources. Animated PNG and GIF files open as movies instead ([`crate::animated`]); here they would
//! show their first frame.

use std::sync::Arc;

use filmcraft_frame::VideoFrame;
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};

use crate::{FrameRequest, MediaError, MediaInfo, MediaKind, MediaSource, Result, VideoStreamInfo};

/// The EXIF Orientation a still decode turns its picture by ([`probe_geometry`]).
pub use image::metadata::Orientation;

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
    reader.limits(decode_limits());
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

/// The `image` crate's limits for one decoded still: [`MAX_DECODE_BYTES`] and [`MAX_DECODE_SIDE`].
fn decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    limits.max_image_width = Some(MAX_DECODE_SIDE);
    limits.max_image_height = Some(MAX_DECODE_SIDE);
    limits
}

/// Whether `orientation` turns the picture onto its side (the four transposing EXIF values), so
/// the decoded size is the stored size swapped.
pub fn transposes(orientation: Orientation) -> bool {
    matches!(orientation, Orientation::Rotate90 | Orientation::Rotate90FlipH | Orientation::Rotate270 | Orientation::Rotate270FlipH)
}

/// Stills whose decoder never turns the picture by an EXIF Orientation: DNG (which ignores it),
/// Radiance HDR, OpenEXR, DPX, Targa and YCbCr TIFF. Their decoded size is their stored size.
fn never_turned(name: &str, b: &[u8]) -> bool {
    dng::sniff(b) || hdr::sniff(b) || exr::sniff(b) || crate::dpx::sniff(b) || crate::tga::sniff(name, b) || ycbcr_tiff::sniff(b)
}

/// The size [`StillSource::decode`] gives the still `name` in `bytes`, and the EXIF Orientation
/// it turns the picture by, read from the file's headers alone: no picture is decoded and nothing
/// is kept, so a caller that only compares geometry (the engine's load-time still migration)
/// never pays for a decode. The size is the upright one, held at [`MAX_STILL_SIDE`] as the decode
/// holds it.
///
/// Only the formats the `image` crate decodes (JPEG, PNG, GIF, WebP, TIFF, BMP) are probed: they
/// are the ones whose decode applies an orientation. `None` for the stills whose decoder never
/// turns the picture (DNG, Radiance HDR, OpenEXR, DPX, Targa, YCbCr TIFF), for anything that is
/// not a still we open, and for a header the decode would refuse.
pub fn probe_geometry(name: &str, bytes: &[u8]) -> Option<(u32, u32, Orientation)> {
    if never_turned(name, bytes) || !sniff(name, bytes) {
        return None;
    }
    probe_image(std::io::Cursor::new(bytes))
}

/// Bytes [`probe_geometry_reader`] reads whole before it decides: a file this short is probed
/// as [`probe_geometry`] probes it, and a longer one is sniffed from this prefix.
const PROBE_PREFIX: u64 = 64 << 10;
/// The most [`probe_geometry_reader`] reads past its prefix: a damaged header cannot make it read
/// a whole file.
const PROBE_READ_BUDGET: u64 = 16 << 20;

/// Like [`probe_geometry`], reading from `r` only the bytes the headers take (within
/// [`PROBE_READ_BUDGET`]) instead of the whole file: a large TIFF whose image directory sits
/// at its end is probed without its pixels being read.
pub fn probe_geometry_reader(name: &str, r: &dyn crate::ByteReader) -> Option<(u32, u32, Orientation)> {
    let len = r.len();
    let mut prefix = vec![0u8; usize::try_from(len.min(PROBE_PREFIX)).ok()?];
    r.read_at(0, &mut prefix).ok()?;
    if len <= PROBE_PREFIX {
        return probe_geometry(name, &prefix);
    }
    if never_turned(name, &prefix) || !sniff(name, &prefix) {
        return None;
    }
    probe_image(std::io::BufReader::new(BoundedReader { inner: r, pos: 0, budget: PROBE_READ_BUDGET }))
}

/// The upright, capped size and the orientation of the picture `r` holds, from its headers.
fn probe_image<R: std::io::BufRead + std::io::Seek>(r: R) -> Option<(u32, u32, Orientation)> {
    use image::ImageDecoder as _;
    let mut reader = image::ImageReader::new(r).with_guessed_format().ok()?;
    reader.limits(decode_limits());
    let mut decoder = reader.into_decoder().ok()?;
    if decoder.total_bytes() > MAX_DECODE_BYTES {
        return None;
    }
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    // as in the decode: a damaged EXIF block is no reason to refuse the picture
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let (w, h) = if transposes(orientation) { (h, w) } else { (w, h) };
    let (w, h) = capped_size(w, h);
    Some((w, h, orientation))
}

/// `Read + Seek` over a [`crate::ByteReader`] that gives up once `budget` bytes have been read.
struct BoundedReader<'a> {
    inner: &'a dyn crate::ByteReader,
    pos: u64,
    budget: u64,
}

impl std::io::Read for BoundedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.inner.len().saturating_sub(self.pos);
        if left == 0 || buf.is_empty() {
            return Ok(0);
        }
        if self.budget == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "still header probe read past its budget"));
        }
        let n = usize::try_from((buf.len() as u64).min(left).min(self.budget)).unwrap_or(0);
        let Some(chunk) = buf.get_mut(..n) else { return Ok(0) };
        self.inner.read_at(self.pos, chunk)?;
        self.pos = self.pos.saturating_add(n as u64);
        self.budget = self.budget.saturating_sub(n as u64);
        Ok(n)
    }
}

impl std::io::Seek for BoundedReader<'_> {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek out of range");
        let next = match to {
            std::io::SeekFrom::Start(n) => Some(n),
            std::io::SeekFrom::End(d) => self.inner.len().checked_add_signed(d),
            std::io::SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        self.pos = next.ok_or_else(bad)?;
        Ok(self.pos)
    }
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

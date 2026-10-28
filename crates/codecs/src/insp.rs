//! Insta360 photos (`.insp`): a JPEG of the two lenses side by side, the camera's trailer after it
//! (`filmcraft_isobmff::insta360`: its `offset` lens calibration). Opened as a still that is 360°
//! — a dual fisheye, the image circles from the calibration ([`crate::vr::insta360_spherical`]),
//! as the camera's videos are — so it is reframed, levelled by hand and made into a tiny planet
//! like them.

use std::sync::Arc;

use filmcraft_frame::{AudioBuffer, VideoFrame};
use filmcraft_isobmff::insta360::{MAGIC, read_info};
use filmcraft_media::{FrameRequest, MediaError, MediaInfo, MediaSource, SharedSource, Spherical};

/// Largest photo read, bytes.
pub const MAX_FILE: u64 = 256 << 20;

/// A JPEG (its start) that ends with an Insta360 trailer (`tail`: the file's last bytes).
fn sniff(head: &[u8], tail: &[u8]) -> bool {
    head.starts_with(&[0xFF, 0xD8, 0xFF]) && tail.ends_with(MAGIC)
}

/// The opener for a photo held in memory.
pub fn opener(name: &str, bytes: Arc<[u8]>) -> Option<Result<SharedSource, MediaError>> {
    if !sniff(&bytes, &bytes) {
        return None;
    }
    Some(still(name, &bytes))
}

/// The reader opener: an Insta360 photo is read whole (it is one picture), up to [`MAX_FILE`].
pub fn reader_opener(name: &str, head: &[u8], reader: &filmcraft_media::SharedReader) -> Option<Result<SharedSource, MediaError>> {
    if !head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return None;
    }
    let len = reader.len();
    let tail = filmcraft_media::reader::read_range(&**reader, len.saturating_sub(MAGIC.len() as u64), MAGIC.len()).ok()?;
    if !sniff(head, &tail) {
        return None;
    }
    if len > MAX_FILE {
        return Some(Err(MediaError::Unsupported(format!("{name}: the photo is larger than {} MiB", MAX_FILE >> 20))));
    }
    let all = match filmcraft_media::reader::read_range(&**reader, 0, usize::try_from(len).unwrap_or(usize::MAX)) {
        Ok(b) => b,
        Err(e) => return Some(Err(MediaError::Io(format!("{name}: {e}")))),
    };
    Some(still(name, &all))
}

/// The photo as a 360° still: its sphere from the trailer, or a plain still when the trailer
/// says nothing usable (a picture that is not 2:1).
fn still(name: &str, bytes: &[u8]) -> Result<SharedSource, MediaError> {
    let inner = filmcraft_media::still::StillSource::decode(name, bytes)?;
    let size = inner.info().video.as_ref().map_or((0, 0), |v| (v.width, v.height));
    // (a photo scaled to the still cap can be a pixel off 2:1)
    let square = if size.0.abs_diff(size.1.saturating_mul(2)) <= 2 { (size.1.saturating_mul(2), size.1) } else { size };
    let sphere = read_info(&bytes).and_then(|info| crate::vr::insta360_spherical(&info, square));
    Ok(Arc::new(Photo360 { inner, sphere }))
}

/// A still with the sphere its camera says it shows.
struct Photo360 {
    inner: filmcraft_media::still::StillSource,
    sphere: Option<Spherical>,
}

impl MediaSource for Photo360 {
    fn info(&self) -> &MediaInfo {
        self.inner.info()
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        self.inner.video_frame(req)
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }
    fn spherical(&self) -> Option<Spherical> {
        self.sphere.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_isobmff::insta360::RECORD_INFO;

    fn pb_str(field: u64, s: &str) -> Vec<u8> {
        let mut v = vec![(field << 3 | 2) as u8];
        v.push(s.len() as u8);
        v.extend_from_slice(s.as_bytes());
        v
    }

    /// `body` with an Insta360 trailer holding `records` (id, data), as the camera writes it.
    fn with_trailer(body: &[u8], records: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut f = body.to_vec();
        let start = f.len();
        let mut index = Vec::new();
        for (id, data) in records {
            let off = (f.len() - start) as u32;
            f.extend_from_slice(data);
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

    /// A JPEG of two lens circles side by side, `w` × `h`.
    fn lenses_jpeg(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            let side = h as f32;
            let (cx, cy) = (if x < w / 2 { side / 2.0 } else { side * 1.5 }, side / 2.0);
            let r = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
            if r < side * 0.48 { image::Rgb([180, 140, 90]) } else { image::Rgb([0, 0, 0]) }
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Jpeg).unwrap();
        out
    }

    // the X5's calibration, scaled to a 512 × 256 picture's lenses (canvas 512 × 256)
    const OFFSET: &str = "2_126.6_127.7_127.3_0.249_-0.624_89.769_126.5_385.3_127.1_-0.436_-0.465_90.327_512_256_1137";

    #[test]
    fn a_photo_with_the_cameras_trailer_is_a_360_still() {
        let mut info = pb_str(2, "Insta360 X5");
        info.extend(pb_str(5, OFFSET));
        let photo = with_trailer(&lenses_jpeg(512, 256), &[(RECORD_INFO, info)]);
        let src = opener("IMG_20260912_203000_00_101.insp", photo.into()).expect("taken").unwrap();
        assert_eq!(src.info().kind, filmcraft_media::MediaKind::Still);
        assert_eq!(src.info().video.as_ref().map(|v| (v.width, v.height)), Some((512, 256)));
        let sph = src.spherical().expect("a sphere");
        let filmcraft_media::spherical::Projection::DualFisheye { lenses, .. } = &sph.projection else { panic!("{sph:?}") };
        assert_eq!(lenses.len(), 2);
        // the calibration's circles: the second lens's centre in its own half
        assert!((lenses[1].cx - (385.3 - 256.0) / 256.0).abs() < 1e-6, "{:?}", lenses[1]);
        assert!((lenses[0].radius - 0.5).abs() < 0.05, "{:?}", lenses[0]);
        // a plain JPEG is not taken (the still decoders open it)
        assert!(opener("a.jpg", lenses_jpeg(64, 32).into()).is_none());
        // a photo that is not 2:1 is a still without a sphere
        let odd = with_trailer(&lenses_jpeg(300, 200), &[(RECORD_INFO, pb_str(5, OFFSET))]);
        assert!(opener("odd.insp", odd.into()).unwrap().unwrap().spherical().is_none());
        // hostile trailers: the photo still opens
        let mut broken = lenses_jpeg(128, 64);
        broken.extend_from_slice(&[0xFF; 40]);
        broken.extend_from_slice(MAGIC);
        let s = opener("broken.insp", broken.into()).unwrap().unwrap();
        assert!(s.spherical().is_none());
        assert!(opener("empty.insp", Vec::<u8>::new().into()).is_none());
    }
}

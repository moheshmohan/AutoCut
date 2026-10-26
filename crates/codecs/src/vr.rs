//! 360° video in MP4 files: Insta360 recordings (one video track a lens, decoded side by side
//! into one dual-fisheye frame) and the spherical metadata of stitched equirectangular video
//! (`sv3d` / `st3d`, or the V1 `uuid` box; `filmcraft_isobmff::spherical`).

use filmcraft_frame::{PixelData, VideoFrame};
use filmcraft_isobmff::insta360::{Info, parse_offset};
use filmcraft_isobmff::spherical::{Projection, StereoMode};
use filmcraft_isobmff::{Mp4File, Track, TrackKind};
use filmcraft_media::Spherical;
use filmcraft_media::spherical::{FisheyeLens, StereoLayout};

use crate::{CodecError, Result};

/// The video track of an Insta360 recording's second lens, beside the first lens's track
/// `first`: another playable video track of the same codec, picture size, timescale and number of
/// pictures (give or take one).
pub(crate) fn second_lens(file: &Mp4File, first: usize) -> Option<usize> {
    let a = file.tracks.get(first)?;
    let size = |t: &Track| t.video().map(|v| (v.width, v.height));
    let codec = |t: &Track| t.codec().map(|c| c.name());
    file.tracks.iter().enumerate().position(|(j, t)| {
        j != first
            && t.kind == TrackKind::Video
            && !t.entries.is_empty()
            && !t.samples.is_empty()
            && t.timescale == a.timescale
            && size(t) == size(a)
            && codec(t) == codec(a)
            && t.samples.len().abs_diff(a.samples.len()) <= 1
    })
}

/// Two pictures of the same size and pixel format side by side: `a` on the left.
pub(crate) fn side_by_side(a: &VideoFrame, b: &VideoFrame) -> Result<VideoFrame> {
    if (a.width, a.height) != (b.width, b.height) {
        return Err(CodecError::Decode(format!("the two lenses decoded to different sizes ({}x{} and {}x{})", a.width, a.height, b.width, b.height)));
    }
    let (w, h) = (a.width as usize, a.height as usize);
    let data = match (&a.data, &b.data) {
        (PixelData::Yuv8 { planes: pa, chroma: ca, alpha: None }, PixelData::Yuv8 { planes: pb, chroma: cb, alpha: None }) if ca == cb => {
            let (sx, sy) = ca.shifts();
            if w % (1 << sx) != 0 {
                return Err(CodecError::Decode("odd lens picture width".into()));
            }
            let (cw, ch) = (w >> sx, h.div_ceil(1 << sy));
            PixelData::Yuv8 {
                planes: [
                    std::sync::Arc::new(join_rows(&pa[0], &pb[0], w, h)?),
                    std::sync::Arc::new(join_rows(&pa[1], &pb[1], cw, ch)?),
                    std::sync::Arc::new(join_rows(&pa[2], &pb[2], cw, ch)?),
                ],
                chroma: *ca,
                alpha: None,
            }
        }
        (PixelData::Yuv16 { planes: pa, chroma: ca, bits: ba, alpha: None }, PixelData::Yuv16 { planes: pb, chroma: cb, bits: bb, alpha: None })
            if ca == cb && ba == bb =>
        {
            let (sx, sy) = ca.shifts();
            if w % (1 << sx) != 0 {
                return Err(CodecError::Decode("odd lens picture width".into()));
            }
            let (cw, ch) = (w >> sx, h.div_ceil(1 << sy));
            PixelData::Yuv16 {
                planes: [
                    std::sync::Arc::new(join_rows(&pa[0], &pb[0], w, h)?),
                    std::sync::Arc::new(join_rows(&pa[1], &pb[1], cw, ch)?),
                    std::sync::Arc::new(join_rows(&pa[2], &pb[2], cw, ch)?),
                ],
                chroma: *ca,
                bits: *ba,
                alpha: None,
            }
        }
        (PixelData::Rgba8(pa), PixelData::Rgba8(pb)) => PixelData::Rgba8(std::sync::Arc::new(join_rows(pa, pb, w * 4, h)?)),
        _ => return Err(CodecError::Decode("the two lenses decoded to different pixel formats".into())),
    };
    let width = a.width.checked_mul(2).ok_or_else(|| CodecError::Decode("dual-lens picture too wide".into()))?;
    Ok(VideoFrame { width, height: a.height, data, color: a.color, par: a.par, pts: a.pts })
}

/// Rows of `a` and `b` (each `w` × `h`) joined: every output row is a row of `a` then `b`'s.
fn join_rows<T: Copy>(a: &[T], b: &[T], w: usize, h: usize) -> Result<Vec<T>> {
    let n = w.checked_mul(h).ok_or_else(|| CodecError::Decode("lens picture too large".into()))?;
    let (Some(a), Some(b)) = (a.get(..n), b.get(..n)) else {
        return Err(CodecError::Decode("short lens picture plane".into()));
    };
    let mut out = Vec::with_capacity(n * 2);
    if w == 0 {
        return Ok(out);
    }
    for (ra, rb) in a.chunks_exact(w).zip(b.chunks_exact(w)) {
        out.extend_from_slice(ra);
        out.extend_from_slice(rb);
    }
    Ok(out)
}

/// Half the field of view of an Insta360 X-series lens, degrees: where the image circle's edge
/// is from the optical axis, read as an equidistant lens (the image-circle radius from the
/// calibration ↔ this angle). Measured on X5 footage, where the two lenses' pictures agree best
/// along the seam at 97.5° (the alignment in `filmcraft_render::vr::align_lenses` refines it per
/// clip).
pub const INSTA360_HALF_FOV: f64 = 97.5;

/// The dual-fisheye description of an Insta360 recording whose decoded picture is `size`: two
/// lenses side by side, in the camera's own order — a recording's two tracks are decoded so
/// ([`crate::mp4`]), its `.lrv` proxy holds them so in one picture. The image circles come from the
/// trailer's `offset` calibration, which measures the lenses side by side in a canvas of squares
/// (the circles of the decoded pictures were measured to match it, unrotated); without a usable one
/// the circles fill their halves.
pub(crate) fn insta360_spherical(info: &Info, size: (u32, u32)) -> Option<Spherical> {
    if size.0 == 0 || size.1 == 0 || size.0 != size.1.saturating_mul(2) {
        return None;
    }
    let calibrated = parse_offset(&info.offset).filter(|o| o.lenses.len() == 2 && o.canvas.1 > 0.0 && (o.canvas.0 - 2.0 * o.canvas.1).abs() < 1.0);
    let lens = |k: usize| -> FisheyeLens {
        let rect = [k as f64 * 0.5, 0.0, 0.5, 1.0];
        let (cx, cy, radius, roll) = match calibrated.as_ref().and_then(|o| Some((o.lenses.get(k)?, o.canvas.1))) {
            Some((l, side)) => ((l.cx - k as f64 * side) / side, l.cy / side, l.radius / side, 0.0),
            None => (0.5, 0.5, 0.5, 0.0),
        };
        // the left lens looks back (at the person holding the camera), the right one forward
        let yaw = if k == 0 { 180.0 } else { 0.0 };
        FisheyeLens { rect, cx, cy, radius, half_fov: INSTA360_HALF_FOV, poly: Vec::new(), yaw, pitch: 0.0, roll }
    };
    Some(Spherical::dual_fisheye(vec![lens(0), lens(1)], &info.camera))
}

/// The spherical metadata of a stitched video track: V2 (`sv3d`, `st3d`) first, then V1. Only
/// equirectangular projections are described; a cube map or mesh is not (yet) understood, and
/// is treated as a flat picture.
pub(crate) fn track_spherical(t: &Track) -> Option<Spherical> {
    if let Some(v) = t.video()
        && let Some(sv) = &v.spherical
    {
        let Some(Projection::Equirect { top, bottom, left, right }) = sv.projection else { return None };
        let mut s = Spherical::equirect();
        s.layout = match v.stereo {
            Some(StereoMode::TopBottom) => StereoLayout::OverUnder,
            Some(StereoMode::LeftRight) => StereoLayout::SideBySide,
            _ => StereoLayout::Mono,
        };
        // bounds are 0.32 fixed-point shares of the sphere cut off each side
        let share = |v: u32| f64::from(v) / 4_294_967_296.0;
        s.captured = [360.0 * (1.0 - share(left) - share(right)).clamp(0.0, 1.0), 180.0 * (1.0 - share(top) - share(bottom)).clamp(0.0, 1.0)];
        s.pose = sv.pose;
        s.camera = sv.source.clone();
        return Some(s.sanitized());
    }
    let v1 = t.spherical_v1.as_ref().filter(|v| v.spherical)?;
    if !(v1.projection.is_empty() || v1.projection == "equirectangular") {
        return None;
    }
    let mut s = Spherical::equirect();
    s.layout = match v1.stereo.as_str() {
        "top-bottom" => StereoLayout::OverUnder,
        "left-right" => StereoLayout::SideBySide,
        _ => StereoLayout::Mono,
    };
    if let Some([fw, fh, cw, ch, _, _]) = v1.cropped
        && fw > 0
        && fh > 0
    {
        s.captured = [360.0 * cw as f64 / fw as f64, 180.0 * ch as f64 / fh as f64];
    }
    s.pose = v1.initial_view;
    s.camera = v1.stitching_software.clone();
    Some(s.sanitized())
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_frame::Chroma;
    use std::sync::Arc;

    fn yuv(w: u32, h: u32, y: u8) -> VideoFrame {
        let (cw, ch) = (w as usize / 2, (h as usize).div_ceil(2));
        VideoFrame {
            width: w,
            height: h,
            data: PixelData::Yuv8 {
                planes: [Arc::new(vec![y; (w * h) as usize]), Arc::new(vec![1; cw * ch]), Arc::new(vec![2; cw * ch])],
                chroma: Chroma::C420,
                alpha: None,
            },
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            pts: filmcraft_time::Tick::ZERO,
        }
    }

    #[test]
    fn lenses_join_side_by_side() {
        let f = side_by_side(&yuv(4, 3, 10), &yuv(4, 3, 200)).unwrap();
        assert_eq!((f.width, f.height), (8, 3));
        let PixelData::Yuv8 { planes, .. } = &f.data else { panic!() };
        assert_eq!(&planes[0][..8], &[10, 10, 10, 10, 200, 200, 200, 200]);
        assert_eq!(planes[1].len(), 4 * 2);
    }

    #[test]
    fn mismatched_lenses_are_an_error() {
        assert!(side_by_side(&yuv(4, 2, 0), &yuv(6, 2, 0)).is_err());
        assert!(side_by_side(&yuv(3, 2, 0), &yuv(3, 2, 0)).is_err());
        let mut short = yuv(4, 2, 0);
        short.data = PixelData::Yuv8 { planes: [Arc::new(vec![0; 3]), Arc::new(vec![0; 2]), Arc::new(vec![0; 2])], chroma: Chroma::C420, alpha: None };
        assert!(side_by_side(&short, &yuv(4, 2, 0)).is_err());
    }

    #[test]
    fn insta360_metadata_describes_two_lenses() {
        let info = Info {
            camera: "Insta360 X5".into(),
            offset: "2_2657.856_2681.350_2673.940_0.249_-0.624_89.769_2655.921_8091.830_2669.660_-0.436_-0.465_90.327_10752_5376_1137".into(),
            ..Default::default()
        };
        let s = insta360_spherical(&info, (5760, 2880)).unwrap();
        let filmcraft_media::spherical::Projection::DualFisheye { lenses } = &s.projection else { panic!() };
        assert_eq!(lenses.len(), 2);
        assert!((lenses[1].cx - (8091.83 - 5376.0) / 5376.0).abs() < 1e-9);
        assert_eq!(s.camera, "Insta360 X5");
        // not a 2:1 picture: not described
        assert!(insta360_spherical(&info, (2880, 2880)).is_none());
        // no calibration: circles fill their halves
        let s = insta360_spherical(&Info::default(), (1664, 832)).unwrap();
        let filmcraft_media::spherical::Projection::DualFisheye { lenses } = &s.projection else { panic!() };
        assert_eq!((lenses[0].cx, lenses[0].radius), (0.5, 0.5));
    }
}

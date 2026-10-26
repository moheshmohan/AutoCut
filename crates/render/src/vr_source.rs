//! 360° media as flat sources, for the analyses that read pictures (`filmcraft_media::MediaSource`).
//!
//! - [`PanoramaSource`]: a dual-fisheye recording seen as its stitched equirectangular sphere —
//!   what scene detection, descriptions and subject boxes want of it (a box's centre is then a
//!   direction: longitude across, latitude down).
//! - [`ReframedSource`]: a clip seen through its virtual camera (the `vr_reframe` effect, its
//!   keyframes and stabilisation at each media time) as a flat picture of the camera's frame —
//!   what tracking, masks and redaction, which work in a clip's own pixels, must read.
//!
//! Sound and timing pass through to the media.

use std::f64::consts::TAU;
use std::sync::Arc;

use filmcraft_frame::{AudioBuffer, VideoFrame};
use filmcraft_media::{FrameRequest, MediaError, MediaInfo, MediaSource, SharedSource, Spherical};
use filmcraft_project::{EffectInstance, TrackItem};

use crate::image::Image;
use crate::vr::{SourceModel, View, clip_view, reframe, source_scale, taps};

/// Widest analysis panorama, pixels.
pub const PANORAMA_MAX_WIDTH: u32 = 4096;

/// The inner source's picture at `time`, at `scale` of its size, linear (box-decimated as the
/// renderer decimates).
fn inner_image(inner: &SharedSource, time: filmcraft_time::Tick, scale: f32) -> filmcraft_media::Result<Image> {
    let full_w = inner.info().video.as_ref().map_or(1, |v| v.width.max(1));
    let f = inner.video_frame(FrameRequest { time, scale })?;
    let n = crate::decimation(f.width as f32, full_w as f32 * scale);
    let (w, h, px) = f.to_linear_f32_decimated(n);
    Ok(Image { w, h, px })
}

fn scale_of(req: &FrameRequest) -> f32 {
    if req.scale.is_finite() { req.scale.clamp(1.0 / 64.0, 1.0) } else { 1.0 }
}

/// A 360° source seen as its stitched whole-sphere equirectangular picture.
pub struct PanoramaSource {
    inner: SharedSource,
    spherical: Spherical,
    info: MediaInfo,
}

impl PanoramaSource {
    /// The panorama of `inner` (described by `spherical`): 2:1, as wide as the source's own
    /// detail (at most [`PANORAMA_MAX_WIDTH`]). `None` for a source without video.
    pub fn new(inner: SharedSource, spherical: &Spherical) -> Option<Self> {
        let spherical = spherical.sanitized();
        let v = inner.info().video.clone()?;
        let model = SourceModel::new(&spherical, (v.width as usize, v.height as usize));
        let w = ((model.pixels_per_radian() * TAU).round().clamp(64.0, f64::from(PANORAMA_MAX_WIDTH)) as u32) & !1;
        let mut info = inner.info().clone();
        if let Some(pv) = info.video.as_mut() {
            pv.width = w;
            pv.height = w / 2;
            pv.par = (1, 1);
            pv.has_alpha = false;
        }
        Some(PanoramaSource { inner, spherical, info })
    }
}

impl MediaSource for PanoramaSource {
    fn info(&self) -> &MediaInfo {
        &self.info
    }

    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let v = self.info.video.as_ref().ok_or(MediaError::NoStream("video"))?;
        let scale = scale_of(&req);
        let (w, h) = (((v.width as f32 * scale).round() as usize).max(2), ((v.height as f32 * scale).round() as usize).max(1));
        let view = View::sphere(0.0, 0.0, 0.0);
        let iv = self.inner.info().video.as_ref().ok_or(MediaError::NoStream("video"))?;
        let full = SourceModel::new(&self.spherical, (iv.width as usize, iv.height as usize));
        let src = inner_image(&self.inner, req.time, source_scale(&full, &view, w) as f32)?;
        let model = SourceModel::at(&self.spherical, (src.w, src.h), req.time);
        let pano = reframe(&src, &model, &view, w, h, 1);
        Ok(Arc::new(VideoFrame { pts: req.time, ..VideoFrame::rgba_f32(w as u32, h as u32, pano.px) }))
    }

    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }

    fn audio_stream(&self, stream: usize, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio_stream(stream, start, frames, sample_rate)
    }

    fn spherical(&self) -> Option<Spherical> {
        Some(Spherical::equirect())
    }
}

/// A clip seen through its virtual camera: frames at media time `t` are what the camera saw
/// then, at the camera's frame size.
pub struct ReframedSource {
    inner: SharedSource,
    spherical: Spherical,
    effect: EffectInstance,
    info: MediaInfo,
}

impl ReframedSource {
    /// The clip `item`'s camera over `inner` (its media, described by `spherical`; a whole
    /// equirectangular sphere without one). `None` when the clip has no enabled virtual camera or
    /// the media no video.
    pub fn new(inner: SharedSource, spherical: Option<&Spherical>, item: &TrackItem) -> Option<Self> {
        let effect = filmcraft_project::vr::reframe_of(item)?.clone();
        let (w, h) = filmcraft_project::vr::effect_size(&effect)?;
        let mut info = inner.info().clone();
        let v = info.video.as_mut()?;
        v.width = w;
        v.height = h;
        v.par = (1, 1);
        v.has_alpha = false;
        let spherical = spherical.cloned().unwrap_or_else(Spherical::equirect).sanitized();
        Some(ReframedSource { inner, spherical, effect, info })
    }
}

impl MediaSource for ReframedSource {
    fn info(&self) -> &MediaInfo {
        &self.info
    }

    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let (view, size) = clip_view(&self.effect, req.time).ok_or(MediaError::NoStream("video"))?;
        let scale = scale_of(&req);
        let (w, h) = (((size.0 as f32 * scale).round() as usize).max(1), ((size.1 as f32 * scale).round() as usize).max(1));
        let iv = self.inner.info().video.as_ref().ok_or(MediaError::NoStream("video"))?;
        let full = SourceModel::new(&self.spherical, (iv.width as usize, iv.height as usize));
        let src = inner_image(&self.inner, req.time, source_scale(&full, &view, w) as f32)?;
        let model = SourceModel::at(&self.spherical, (src.w, src.h), req.time);
        let pic = reframe(&src, &model, &view, w, h, taps(&model, &view, w));
        Ok(Arc::new(VideoFrame { pts: req.time, ..VideoFrame::rgba_f32(w as u32, h as u32, pic.px) }))
    }

    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }

    fn audio_stream(&self, stream: usize, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio_stream(stream, start, frames, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_media::spherical::FisheyeLens;
    use filmcraft_time::Tick;

    /// A still 2:1 source of `px` (RGBA f32, linear) that reports what it was asked for.
    struct Still {
        info: MediaInfo,
        w: u32,
        h: u32,
    }
    impl MediaSource for Still {
        fn info(&self) -> &MediaInfo {
            &self.info
        }
        fn video_frame(&self, _req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
            let px: Vec<f32> = (0..self.w * self.h).flat_map(|i| [((i % self.w) as f32 / self.w as f32), 0.3, 0.6, 1.0]).collect();
            Ok(Arc::new(VideoFrame::rgba_f32(self.w, self.h, px)))
        }
        fn audio(&self, _start: i64, frames: usize, rate: u32) -> filmcraft_media::Result<AudioBuffer> {
            Ok(AudioBuffer::silence(rate, 2, frames))
        }
    }

    fn still(w: u32, h: u32) -> SharedSource {
        let video = filmcraft_media::VideoStreamInfo {
            width: w,
            height: h,
            frame_rate: filmcraft_time::FrameRate::FPS_30,
            par: (1, 1),
            codec: "test".into(),
            pixel_format: "rgba".into(),
            color: filmcraft_color::ColorInfo::REC709,
            has_alpha: false,
            bitrate: None,
            hdr: None,
        };
        let info = MediaInfo {
            name: "s".into(),
            kind: filmcraft_media::MediaKind::Movie,
            duration: Tick::from_seconds_f64(2.0),
            video: Some(video),
            audio_streams: vec![],
            container: "test".into(),
            start_timecode: None,
            file_size: None,
        };
        Arc::new(Still { info, w, h })
    }

    #[test]
    fn a_dual_fisheye_is_seen_as_a_panorama() {
        let lens = |k: usize, yaw: f64| FisheyeLens {
            rect: [k as f64 * 0.5, 0.0, 0.5, 1.0],
            cx: 0.5,
            cy: 0.5,
            radius: 0.5,
            half_fov: 97.5,
            poly: vec![],
            yaw,
            pitch: 0.0,
            roll: 0.0,
        };
        let s = Spherical::dual_fisheye(vec![lens(0, 180.0), lens(1, 0.0)], "");
        let p = PanoramaSource::new(still(800, 400), &s).unwrap();
        let v = p.info().video.clone().unwrap();
        assert_eq!(v.width, 2 * v.height);
        let f = p.video_frame(FrameRequest { time: Tick::ZERO, scale: 0.25 }).unwrap();
        assert_eq!((f.width, f.height), ((v.width as f32 * 0.25).round() as u32, (v.height as f32 * 0.25).round() as u32));
        assert_eq!(p.spherical(), Some(Spherical::equirect()));
    }

    #[test]
    fn a_reframed_clip_is_its_cameras_picture() {
        let src = still(720, 360);
        let mut p = filmcraft_project::Project::new("t");
        let clip = filmcraft_project::MediaClip {
            media: filmcraft_project::MediaRef::File { path: "/x.insv".into() },
            info: src.info().clone(),
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        };
        let item = p.add_item("x", filmcraft_project::Label::Iris, filmcraft_project::ItemKind::Media(clip), None);
        let range = filmcraft_time::TimeRange::new(Tick::ZERO, Tick::from_seconds_f64(2.0));
        let mut it = p.make_track_item(item, filmcraft_project::TrackKind::Video, Tick::ZERO, range, filmcraft_time::FrameRate::FPS_30).unwrap();
        assert!(ReframedSource::new(still(720, 360), None, &it).is_none());
        let mut e = filmcraft_project::find_effect(filmcraft_project::vr::REFRAME).unwrap().instance();
        e.params.insert("width".into(), filmcraft_project::Param::new(filmcraft_project::ParamValue::Float(640.0)));
        e.params.insert("height".into(), filmcraft_project::Param::new(filmcraft_project::ParamValue::Float(360.0)));
        it.effects.push(e);
        let r = ReframedSource::new(still(720, 360), None, &it).unwrap();
        assert_eq!(r.info().video.as_ref().map(|v| (v.width, v.height)), Some((640, 360)));
        let f = r.video_frame(FrameRequest { time: Tick::ZERO, scale: 0.5 }).unwrap();
        assert_eq!((f.width, f.height), (320, 180));
        assert!(r.audio(0, 16, 48_000).is_ok());
    }
}

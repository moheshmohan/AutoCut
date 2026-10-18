//! Destinations (Export mode's Destinations column) and the pre-flight every destination shares:
//! what a file must be for YouTube, a YouTube Short or an Instagram Reel, checked before any byte
//! leaves the machine, each failure explained with its fix — and, where the export presets can
//! make one, a fixed copy.
//!
//! | command | what it does |
//! |---|---|
//! | `publish.destinations` | the rows: Media File, YouTube, YouTube Short, Instagram Reel — each with its connect state (from disk, no network) |
//! | `publish.preflight {destination, path, loudness?}` | the checks of `path` for `destination`: container, codecs, chroma, frame rate, width, aspect, bitrate, duration, size, audio rate and channels, `moov` placement, loudness (BS.1770) and true peak |
//! | `publish.fixCopy {destination, path, out?, wait?}` | job: a copy that passes — re-encoded (H.264 + AAC 48 kHz, the size, rate and bitrate the destination takes, loudness normalized to −14 LUFS, −1 dBTP) when a check needs it, else only the `moov` moved to the front (no re-encode); its answer (`publish` in `jobs.list`) is the copy and its own pre-flight |
//! | `publish.run {destination, path, …, wait?}` | the pre-flight (errors stop it, with the fixes), then the destination's publish job: `youtube.upload` (YouTube, YouTube Short) or `instagram.publish` (Instagram Reel), with progress and cancel |
//!
//! Every number read from the file is hostile: sizes and box lengths are checked, divisions are
//! guarded, the box walk and the loudness pass are bounded.

use std::collections::HashMap;
use std::io::{Read as _, Seek as _, Write as _};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_media::SharedSource;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p};
use crate::{EngineError, Result, Session};

/// The loudness social platforms play at (YouTube, Instagram), LUFS, and how far off still passes.
pub const SOCIAL_LUFS: f64 = -14.0;
pub const LUFS_TOLERANCE: f64 = 2.0;
/// The true-peak ceiling of a normalized copy, dBTP.
pub const TRUE_PEAK_DBTP: f64 = -1.0;
/// Audio measured at most (longer files: the first this many seconds).
pub const MAX_LOUDNESS_SECONDS: f64 = 10.0 * 60.0;
/// Top-level boxes walked at most.
const MAX_BOXES: usize = 512;
/// The largest `moov` read (a 15-minute Reel's is well under a megabyte).
const MAX_MOOV: u64 = 64 << 20;

/// Where an export goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Destination {
    MediaFile,
    Youtube,
    YoutubeShort,
    InstagramReel,
}

impl Destination {
    pub const ALL: [Destination; 4] = [Destination::MediaFile, Destination::Youtube, Destination::YoutubeShort, Destination::InstagramReel];

    pub fn id(self) -> &'static str {
        match self {
            Destination::MediaFile => "mediaFile",
            Destination::Youtube => "youtube",
            Destination::YoutubeShort => "youtubeShort",
            Destination::InstagramReel => "instagramReel",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Destination::MediaFile => "Media File",
            Destination::Youtube => "YouTube",
            Destination::YoutubeShort => "YouTube Short",
            Destination::InstagramReel => "Instagram Reel",
        }
    }

    /// From an id (any case) or a plain name ("instagram", "short", "file").
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let k: String = s.trim().chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        Ok(match k.as_str() {
            "mediafile" | "file" | "media" => Destination::MediaFile,
            "youtube" | "yt" => Destination::Youtube,
            "youtubeshort" | "youtubeshorts" | "short" | "shorts" => Destination::YoutubeShort,
            "instagramreel" | "instagram" | "reel" | "reels" | "ig" => Destination::InstagramReel,
            _ => {
                return Err(format!("`{}` is not a destination: mediaFile, youtube, youtubeShort or instagramReel", s.chars().take(40).collect::<String>()));
            }
        })
    }

    /// What it asks of a file.
    pub fn spec(self) -> Spec {
        match self {
            Destination::MediaFile => Spec::default(),
            Destination::InstagramReel => Spec {
                checked: true,
                mp4_only: true,
                video_codecs: &["H.264", "HEVC"],
                audio_aac: true,
                max_audio_rate: Some(48_000),
                max_audio_channels: Some(2),
                fps: Some((23.0, 60.0)),
                max_width: Some(1920),
                max_video_kbps: Some(25_000),
                seconds: (crate::instagram::MIN_REEL_SECONDS, crate::instagram::MAX_REEL_SECONDS),
                max_bytes: crate::instagram::MAX_REEL_BYTES,
                shape: Shape::VerticalRecommended,
                loudness: true,
                moov_front: true,
                chroma_420: true,
                target: (1080, 1920),
                audio_kbps: 128,
                video_kbps: 12_000,
            },
            Destination::Youtube => Spec {
                checked: true,
                mp4_only: false,
                seconds: (1.0, 12.0 * 3600.0),
                max_bytes: 256 * 1024 * 1024 * 1024,
                loudness: true,
                target: (1920, 1080),
                audio_kbps: 320,
                video_kbps: 16_000,
                ..Spec::default()
            },
            Destination::YoutubeShort => Spec {
                checked: true,
                mp4_only: false,
                seconds: (1.0, 180.0),
                max_bytes: 256 * 1024 * 1024 * 1024,
                shape: Shape::VerticalOrSquare,
                loudness: true,
                target: (1080, 1920),
                audio_kbps: 320,
                video_kbps: 12_000,
                ..Spec::default()
            },
        }
    }

    /// The engine command that publishes to it (`None`: the Media File is the export itself).
    pub fn publish_command(self) -> Option<&'static str> {
        match self {
            Destination::MediaFile => None,
            Destination::Youtube | Destination::YoutubeShort => Some("youtube.upload"),
            Destination::InstagramReel => Some("instagram.publish"),
        }
    }
}

/// The picture shape a destination wants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shape {
    #[default]
    Any,
    /// Any shape between 0.01:1 and 10:1; 9:16 recommended (a warning otherwise).
    VerticalRecommended,
    /// Vertical or square, or it is not a Short (an error).
    VerticalOrSquare,
}

/// What a destination asks of a file.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spec {
    /// Whether there is anything to check (the Media File has nothing).
    pub checked: bool,
    /// MP4 or MOV only.
    pub mp4_only: bool,
    /// Video codecs taken (prefixes of the codec name: "H.264", "HEVC"); empty: any.
    pub video_codecs: &'static [&'static str],
    /// AAC audio only.
    pub audio_aac: bool,
    pub max_audio_rate: Option<u32>,
    pub max_audio_channels: Option<u32>,
    pub fps: Option<(f64, f64)>,
    pub max_width: Option<u32>,
    pub max_video_kbps: Option<u64>,
    /// Shortest and longest, seconds.
    pub seconds: (f64, f64),
    pub max_bytes: u64,
    pub shape: Shape,
    /// Loudness checked against [`SOCIAL_LUFS`].
    pub loudness: bool,
    /// `moov` before `mdat`.
    pub moov_front: bool,
    /// 4:2:0 chroma.
    pub chroma_420: bool,
    /// The frame a fixed copy of a vertical video fits in (landscape copies swap it).
    pub target: (u32, u32),
    /// A fixed copy's audio and video bitrates, kbps.
    pub audio_kbps: u32,
    pub video_kbps: u32,
}

// -------------------------------------------------------------------------------------------
// what a file is
// -------------------------------------------------------------------------------------------

/// The top-level structure of an MP4 / MOV file.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Boxes {
    /// `(type, offset, size)` of each top-level box.
    #[serde(skip)]
    pub list: Vec<([u8; 4], u64, u64)>,
    /// The `ftyp` major brand.
    pub brand: Option<String>,
    pub has_moov: bool,
    /// `moov` comes before the first `mdat` ("fast start").
    pub moov_first: bool,
    /// A track has an edit list (`edts`).
    pub edit_lists: bool,
}

/// Walk the top-level boxes with `read(offset, len)` (`size` bytes in all): `None` when the
/// file is not an ISO base media file.
pub fn walk_boxes(read: &dyn Fn(u64, usize) -> std::io::Result<Vec<u8>>, size: u64) -> Option<Boxes> {
    let mut b = Boxes::default();
    let mut at = 0u64;
    while at.saturating_add(8) <= size && b.list.len() < MAX_BOXES {
        let head = read(at, 16).ok()?;
        let len32 = u32::from_be_bytes(head.get(0..4)?.try_into().ok()?) as u64;
        let kind: [u8; 4] = head.get(4..8)?.try_into().ok()?;
        if !kind.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
            break;
        }
        let len = match len32 {
            0 => size - at,
            1 => u64::from_be_bytes(head.get(8..16)?.try_into().ok()?),
            n => n,
        };
        if len < 8 || at.checked_add(len).is_none_or(|end| end > size) {
            break;
        }
        if b.list.is_empty() && &kind != b"ftyp" && &kind != b"wide" && &kind != b"free" && &kind != b"moov" && &kind != b"mdat" && &kind != b"skip" {
            return None;
        }
        if &kind == b"ftyp" && b.brand.is_none() {
            b.brand = head.get(8..12).map(|x| String::from_utf8_lossy(x).trim().to_string());
        }
        if &kind == b"moov" {
            b.has_moov = true;
            b.moov_first = !b.list.iter().any(|(k, _, _)| k == b"mdat");
            if len <= MAX_MOOV
                && let Ok(moov) = read(at, len as usize)
            {
                b.edit_lists = has_edit_list(moov.get(8..).unwrap_or_default());
            }
        }
        b.list.push((kind, at, len));
        at += len;
    }
    (!b.list.is_empty()).then_some(b)
}

/// Child boxes of a box body: `(type, start of the child, end)`.
fn children(body: &[u8]) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 8 <= body.len() && out.len() < 4096 {
        let Some(len) = body.get(i..i + 4).and_then(|x| x.try_into().ok()).map(u32::from_be_bytes) else { break };
        let Some(kind) = body.get(i + 4..i + 8).and_then(|x| <[u8; 4]>::try_from(x).ok()) else { break };
        let len = len as usize;
        if len < 8 || i.checked_add(len).is_none_or(|e| e > body.len()) {
            break;
        }
        out.push((kind, i, i + len));
        i += len;
    }
    out
}

fn has_edit_list(moov_body: &[u8]) -> bool {
    children(moov_body)
        .into_iter()
        .filter(|(k, _, _)| k == b"trak")
        .any(|(_, s, e)| children(moov_body.get(s + 8..e).unwrap_or_default()).iter().any(|(k, _, _)| k == b"edts"))
}

/// What the pre-flight knows about a file.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Probe {
    pub path: String,
    pub bytes: u64,
    pub extension: String,
    pub container: String,
    pub seconds: f64,
    pub video: Option<VideoFacts>,
    pub audio: Option<AudioFacts>,
    pub boxes: Option<Boxes>,
    pub loudness: Option<LoudnessFacts>,
    /// Why the loudness was not measured.
    pub loudness_note: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoFacts {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    /// Video bitrate, kbps (the track's, else the file's).
    pub kbps: Option<u64>,
    pub pixel_format: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioFacts {
    pub codec: String,
    pub rate: u32,
    pub channels: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoudnessFacts {
    pub integrated_lufs: f64,
    pub true_peak_dbtp: f64,
    /// Seconds measured (the whole file, or the first [`MAX_LOUDNESS_SECONDS`]).
    pub seconds: f64,
}

/// BS.1770 loudness of `src`'s first audio stream, bounded.
pub fn measure_loudness(src: &dyn filmcraft_media::MediaSource) -> std::result::Result<LoudnessFacts, String> {
    let info = src.info();
    let a = info.audio().ok_or("no sound")?;
    let rate = a.sample_rate;
    let channels = a.channels as usize;
    if !(8_000..=384_000).contains(&rate) || !(1..=8).contains(&channels) {
        return Err(format!("not measured ({rate} Hz, {channels} channels)"));
    }
    let total = info.duration.seconds();
    if !total.is_finite() || total <= 0.0 {
        return Err("not measured (no length)".into());
    }
    let seconds = total.min(MAX_LOUDNESS_SECONDS);
    let frames = (seconds * f64::from(rate)) as usize;
    let mut m = filmcraft_audio_dsp::loudness::LoudnessMeter::new(f64::from(rate), channels);
    let block = (rate as usize).saturating_mul(5).max(1);
    let mut pos = 0usize;
    while pos < frames {
        let n = block.min(frames - pos);
        let buf = src.audio(pos as i64, n, rate).map_err(|e| format!("the sound could not be read: {e}"))?;
        let slices: Vec<&[f32]> = buf.channels.iter().take(channels).map(|c| c.get(..n.min(c.len())).unwrap_or(&[])).collect();
        if slices.len() == channels {
            m.process(&slices);
        }
        pos += n;
    }
    let lufs = m.integrated();
    let tp = m.true_peak_dbtp();
    Ok(LoudnessFacts { integrated_lufs: if lufs.is_finite() { lufs } else { -70.0 }, true_peak_dbtp: if tp.is_finite() { tp } else { -70.0 }, seconds })
}

/// Look at `path` through the session's media pool and services.
pub fn probe(s: &Session, path: &str, loudness: bool) -> std::result::Result<(Probe, SharedSource), String> {
    let bytes = s.services.file_size(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes == 0 {
        return Err(format!("{path} is empty"));
    }
    let src = s.media.open_file(path, &*s.services).map_err(|e| format!("{path} is not a video FilmCraft can read: {e}"))?;
    if src.info().video.is_none() && !src.info().has_audio() {
        return Err(format!("{path} is not a video FilmCraft can read (it has no picture or sound)"));
    }
    let services = s.services.clone();
    let read = |off: u64, len: usize| services.read_range(path, off, len);
    let boxes = walk_boxes(&read, bytes);
    Ok((probe_source(path, bytes, &*src, boxes, loudness), src))
}

/// The facts of an opened source.
pub fn probe_source(path: &str, bytes: u64, src: &dyn filmcraft_media::MediaSource, boxes: Option<Boxes>, loudness: bool) -> Probe {
    let info = src.info();
    let seconds = info.duration.seconds();
    let seconds = if seconds.is_finite() && seconds > 0.0 { seconds } else { 0.0 };
    let video = info.video.as_ref().map(|v| {
        let fps = v.frame_rate.as_f64();
        // the file's rate when the track's is unknown
        let file_kbps = (seconds > 0.0).then(|| (bytes as f64 * 8.0 / seconds / 1000.0) as u64);
        VideoFacts {
            codec: v.codec.clone(),
            width: v.width,
            height: v.height,
            fps: if fps.is_finite() { fps } else { 0.0 },
            kbps: v.bitrate.map(|b| b / 1000).or(file_kbps),
            pixel_format: v.pixel_format.clone(),
        }
    });
    let audio = info.audio().map(|a| AudioFacts { codec: a.codec.clone(), rate: a.sample_rate, channels: a.channels });
    let (loud, note) = match (loudness, &audio) {
        (false, _) => (None, Some("not measured (asked not to)".to_string())),
        (true, None) => (None, Some("no sound".to_string())),
        (true, Some(_)) => match measure_loudness(src) {
            Ok(l) => (Some(l), None),
            Err(e) => (None, Some(e)),
        },
    };
    Probe {
        path: path.to_string(),
        bytes,
        extension: Path::new(path).extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default(),
        container: info.container.clone(),
        seconds,
        video,
        audio,
        boxes,
        loudness: loud,
        loudness_note: note,
    }
}

// -------------------------------------------------------------------------------------------
// the checks
// -------------------------------------------------------------------------------------------

/// How a check came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pass,
    /// Worth fixing; the platform takes it anyway.
    Warn,
    /// The platform refuses it (or it is not what the destination is).
    Fail,
    /// Said for the record.
    Info,
}

/// What can mend a check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fix {
    /// The `moov` moved to the front, nothing re-encoded.
    Faststart,
    /// A re-encoded copy.
    Reexport,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub id: &'static str,
    pub label: &'static str,
    pub status: Status,
    /// What the file has.
    pub value: String,
    /// What the destination takes.
    pub need: String,
    /// What to do about it (failures and warnings).
    pub fix: Option<String>,
    pub fixable: Option<Fix>,
}

fn check(id: &'static str, label: &'static str, status: Status, value: impl Into<String>, need: impl Into<String>) -> Check {
    Check { id, label, status, value: value.into(), need: need.into(), fix: None, fixable: None }
}

impl Check {
    fn fix(mut self, how: impl Into<String>, by: Option<Fix>) -> Self {
        if self.status != Status::Pass {
            self.fix = Some(how.into());
            self.fixable = by;
        }
        self
    }
}

const FIXED_COPY: &str = "export a fixed copy (Export a Fixed Copy, `publish.fixCopy`)";

/// The checks of `p` for `d`.
pub fn checks(d: Destination, p: &Probe) -> Vec<Check> {
    let spec = d.spec();
    let mut out = Vec::new();
    if !spec.checked {
        out.push(check("file", "File", Status::Pass, format!("{} MB", p.bytes / (1024 * 1024)), "any file the export wrote"));
        return out;
    }
    // container
    if spec.mp4_only {
        let ok = matches!(p.extension.as_str(), "mp4" | "mov" | "m4v") && p.boxes.as_ref().is_some_and(|b| b.has_moov);
        out.push(
            check(
                "container",
                "Container",
                if ok { Status::Pass } else { Status::Fail },
                if p.container.is_empty() { p.extension.clone() } else { p.container.clone() },
                "MP4 or MOV",
            )
            .fix(format!("{d} takes MP4 or MOV files: {FIXED_COPY} (H.264 MP4)", d = d.label()), Some(Fix::Reexport)),
        );
    }
    // the picture
    let Some(v) = &p.video else {
        out.push(check("video", "Picture", Status::Fail, "none", "a video stream").fix("this file has no picture; export the sequence as a video", None));
        return finish(d, p, out);
    };
    if !spec.video_codecs.is_empty() {
        let ok = spec.video_codecs.iter().any(|c| v.codec.starts_with(c));
        out.push(
            check("videoCodec", "Video codec", if ok { Status::Pass } else { Status::Fail }, &v.codec, spec.video_codecs.join(" or "))
                .fix(format!("{FIXED_COPY}: H.264"), Some(Fix::Reexport)),
        );
    }
    if spec.chroma_420 {
        let (status, value) = if v.pixel_format.contains("4:2:0") {
            (Status::Pass, v.pixel_format.clone())
        } else if v.pixel_format.contains("4:2:2") || v.pixel_format.contains("4:4:4") || v.pixel_format.contains("RGB") {
            (Status::Fail, v.pixel_format.clone())
        } else {
            (Status::Info, if v.pixel_format.is_empty() { "not known".into() } else { v.pixel_format.clone() })
        };
        out.push(check("chroma", "Chroma subsampling", status, value, "4:2:0").fix(format!("{FIXED_COPY}: 4:2:0 H.264"), Some(Fix::Reexport)));
    }
    if let Some((lo, hi)) = spec.fps {
        let ok = v.fps >= lo - 0.01 && v.fps <= hi + 0.01;
        out.push(
            check("frameRate", "Frame rate", if ok { Status::Pass } else { Status::Fail }, format!("{:.3} fps", v.fps), format!("{lo:.0}–{hi:.0} fps"))
                .fix(format!("{FIXED_COPY} at {} fps", if v.fps > hi { hi as u32 } else { 30 }), Some(Fix::Reexport)),
        );
    } else if v.fps > 60.01 {
        out.push(check("frameRate", "Frame rate", Status::Info, format!("{:.3} fps", v.fps), "up to 60 fps plays as uploaded"));
    }
    if let Some(max) = spec.max_width {
        let ok = v.width <= max;
        out.push(
            check("width", "Width", if ok { Status::Pass } else { Status::Fail }, format!("{} px", v.width), format!("at most {max} px"))
                .fix(format!("{FIXED_COPY}, scaled to {max} px wide or less"), Some(Fix::Reexport)),
        );
    }
    let (w, h) = (v.width.max(1) as f64, v.height.max(1) as f64);
    let ratio = w / h;
    let shape = format!("{}×{} ({})", v.width, v.height, aspect_name(v.width, v.height));
    match spec.shape {
        Shape::Any => {}
        Shape::VerticalRecommended => {
            let status = if !(0.01..=10.0).contains(&ratio) {
                Status::Fail
            } else if (ratio - 9.0 / 16.0).abs() <= 0.02 {
                Status::Pass
            } else {
                Status::Warn
            };
            out.push(check("aspect", "Aspect ratio", status, shape, "9:16 recommended (0.01:1 to 10:1 taken)").fix(
                if status == Status::Fail {
                    "Instagram takes 0.01:1 to 10:1: reframe the sequence".to_string()
                } else {
                    "Instagram crops other shapes to 9:16 in the Reels tab (or shows bars): reframe the sequence to 1080×1920 (a vertical sequence, or Scale to Fill) and export again".to_string()
                },
                None,
            ));
        }
        Shape::VerticalOrSquare => {
            let ok = h >= w;
            out.push(check("aspect", "Aspect ratio", if ok { Status::Pass } else { Status::Fail }, shape, "vertical or square (9:16 best)").fix(
                "YouTube makes Shorts only of vertical or square videos: reframe the sequence to 1080×1920 and export again, or publish it as a regular YouTube video",
                None,
            ));
        }
    }
    if let Some(max) = spec.max_video_kbps {
        match v.kbps {
            Some(k) => out.push(
                check(
                    "bitrate",
                    "Video bitrate",
                    if k <= max { Status::Pass } else { Status::Fail },
                    format!("{:.1} Mbps", k as f64 / 1000.0),
                    format!("at most {} Mbps", max / 1000),
                )
                .fix(format!("{FIXED_COPY} at {} Mbps", spec.video_kbps / 1000), Some(Fix::Reexport)),
            ),
            None => out.push(check("bitrate", "Video bitrate", Status::Info, "not known", format!("at most {} Mbps", max / 1000))),
        }
    }
    finish(d, p, out)
}

/// The checks after the picture: duration, size, sound, structure, loudness.
fn finish(d: Destination, p: &Probe, mut out: Vec<Check>) -> Vec<Check> {
    let spec = d.spec();
    let (lo, hi) = spec.seconds;
    let dur = format!("{:.1} s", p.seconds);
    let need = if hi >= 3600.0 {
        format!("{lo:.0} s to {:.0} h", hi / 3600.0)
    } else if hi >= 60.0 {
        format!("{lo:.0} s to {:.0} min", hi / 60.0)
    } else {
        format!("{lo:.0}–{hi:.0} s")
    };
    let status = if p.seconds + 0.05 < lo || p.seconds > hi + 0.05 { Status::Fail } else { Status::Pass };
    out.push(check("duration", "Duration", status, dur, need).fix(
        if p.seconds > hi {
            format!(
                "trim the sequence to {} or less (set In and Out) and export again",
                if hi >= 60.0 { format!("{:.0} min", hi / 60.0) } else { format!("{hi:.0} s") }
            )
        } else {
            format!("make it at least {lo:.0} s long")
        },
        None,
    ));
    if d == Destination::Youtube && p.seconds > 15.0 * 60.0 {
        out.push(
            check("longVideo", "Longer than 15 minutes", Status::Info, format!("{:.0} min", p.seconds / 60.0), "a verified channel")
                .fix("YouTube takes videos over 15 minutes only from verified channels (https://www.youtube.com/verify)", None),
        );
    }
    let mb = |b: u64| format!("{:.1} MB", b as f64 / (1024.0 * 1024.0));
    let size_ok = p.bytes <= spec.max_bytes;
    // a lower bitrate fits when the length allows a sane one (at least 1.5 Mbps)
    let fits = p.seconds > 0.0 && (spec.max_bytes as f64 * 8.0 * 0.9 / p.seconds / 1000.0) >= 1500.0;
    out.push(check("fileSize", "File size", if size_ok { Status::Pass } else { Status::Fail }, mb(p.bytes), format!("at most {}", mb(spec.max_bytes))).fix(
        if fits { format!("{FIXED_COPY} at a lower bitrate") } else { "trim the video, it cannot fit at a watchable bitrate".to_string() },
        fits.then_some(Fix::Reexport),
    ));
    match &p.audio {
        None => out.push(check("audio", "Sound", Status::Info, "none", "AAC, up to 48 kHz (optional)")),
        Some(a) => {
            if spec.audio_aac {
                let ok = a.codec.contains("AAC");
                out.push(
                    check("audioCodec", "Audio codec", if ok { Status::Pass } else { Status::Fail }, &a.codec, "AAC")
                        .fix(format!("{FIXED_COPY}: AAC"), Some(Fix::Reexport)),
                );
            }
            if let Some(max) = spec.max_audio_rate {
                let ok = a.rate <= max;
                out.push(
                    check(
                        "audioRate",
                        "Audio sample rate",
                        if ok { Status::Pass } else { Status::Fail },
                        format!("{:.1} kHz", a.rate as f64 / 1000.0),
                        format!("at most {} kHz", max / 1000),
                    )
                    .fix(format!("{FIXED_COPY} at 48 kHz"), Some(Fix::Reexport)),
                );
            }
            if let Some(max) = spec.max_audio_channels {
                let ok = a.channels <= max;
                out.push(
                    check("audioChannels", "Audio channels", if ok { Status::Pass } else { Status::Fail }, a.channels.to_string(), "mono or stereo")
                        .fix(format!("{FIXED_COPY}, mixed down to stereo"), Some(Fix::Reexport)),
                );
            }
        }
    }
    if spec.moov_front {
        match &p.boxes {
            Some(b) if b.has_moov => {
                out.push(
                    check(
                        "moovFirst",
                        "Fast start (moov at the front)",
                        if b.moov_first { Status::Pass } else { Status::Warn },
                        if b.moov_first { "at the front" } else { "at the end" },
                        "at the front",
                    )
                    .fix(
                        "Instagram's spec asks for the moov atom at the front of the file: a fast-start copy moves it there without re-encoding",
                        Some(Fix::Faststart),
                    ),
                );
                if b.edit_lists {
                    out.push(check("editLists", "Edit lists", Status::Info, "present", "none, in Instagram's spec").fix(
                        "most MP4 files carry one (an encoder's start delay) and Instagram takes them; if it reports a format error, export a fixed copy or export from FilmCraft with an H.264 preset",
                        None,
                    ));
                }
            }
            _ => {}
        }
    }
    if spec.loudness {
        match (&p.loudness, &p.audio) {
            (Some(l), _) => {
                let off = l.integrated_lufs - SOCIAL_LUFS;
                let status = if off.abs() <= LUFS_TOLERANCE { Status::Pass } else { Status::Warn };
                let how = if off > 0.0 {
                    format!("{} turns louder videos down to about −14 LUFS: {FIXED_COPY}, normalized to −14 LUFS", d.label())
                } else {
                    format!("{} plays it quieter than other videos: {FIXED_COPY}, normalized to −14 LUFS", d.label())
                };
                let measured = if l.seconds + 0.5 < p.seconds {
                    format!("{:.1} LUFS (first {:.0} min)", l.integrated_lufs, l.seconds / 60.0)
                } else {
                    format!("{:.1} LUFS", l.integrated_lufs)
                };
                out.push(check("loudness", "Loudness", status, measured, "−14 LUFS (±2)").fix(how, Some(Fix::Reexport)));
                let tp_ok = l.true_peak_dbtp <= TRUE_PEAK_DBTP + 0.05;
                out.push(
                    check(
                        "truePeak",
                        "True peak",
                        if tp_ok { Status::Pass } else { Status::Warn },
                        format!("{:.1} dBTP", l.true_peak_dbtp),
                        "−1 dBTP or lower",
                    )
                    .fix(format!("peaks above −1 dBTP may distort after the platform's encoding: {FIXED_COPY} (limited to −1 dBTP)"), Some(Fix::Reexport)),
                );
            }
            (None, Some(_)) => {
                out.push(check("loudness", "Loudness", Status::Info, p.loudness_note.clone().unwrap_or_else(|| "not measured".into()), "−14 LUFS"))
            }
            (None, None) => {}
        }
    }
    if d == Destination::YoutubeShort
        && let Some(v) = &p.video
        && (v.width, v.height) != (1080, 1920)
        && v.height >= v.width
    {
        out.push(check("size", "Frame size", Status::Info, format!("{}×{}", v.width, v.height), "1080×1920 is what Shorts show best"));
    }
    out
}

/// "9:16", "16:9", "1:1", or the ratio.
pub fn aspect_name(w: u32, h: u32) -> String {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    if w == 0 || h == 0 {
        return "?".into();
    }
    let g = gcd(w, h).max(1);
    let (a, b) = (w / g, h / g);
    if a <= 64 && b <= 64 { format!("{a}:{b}") } else { format!("{:.2}:1", w as f64 / h as f64) }
}

/// The pre-flight report.
pub fn report(d: Destination, p: &Probe) -> Value {
    let cs = checks(d, p);
    let errors = cs.iter().filter(|c| c.status == Status::Fail).count();
    let warnings = cs.iter().filter(|c| c.status == Status::Warn).count();
    let fix = cs.iter().filter_map(|c| c.fixable).max();
    let unfixable: Vec<&str> = cs.iter().filter(|c| c.status == Status::Fail && c.fixable.is_none()).map(|c| c.id).collect();
    json!({
        "destination": d.id(), "label": d.label(), "path": p.path,
        "ok": errors == 0, "errors": errors, "warnings": warnings,
        "checks": cs, "fixable": fix, "unfixable": unfixable,
        "media": p,
    })
}

// -------------------------------------------------------------------------------------------
// fixed copies
// -------------------------------------------------------------------------------------------

/// Move `moov` in front of `mdat` (`src` → `dst`), shifting every chunk offset by its size; no
/// sample is touched. An `stco` offset that would pass 4 GB is an error (a 64-bit `co64` table
/// grows the box, which this does not do).
pub fn faststart(src: &Path, dst: &Path) -> std::result::Result<(), String> {
    let mut f = std::fs::File::open(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let size = f.metadata().map_err(|e| format!("{}: {e}", src.display()))?.len();
    let read = |off: u64, len: usize| -> std::io::Result<Vec<u8>> {
        let mut g = std::fs::File::open(src)?;
        g.seek(std::io::SeekFrom::Start(off))?;
        let mut v = Vec::new();
        g.take(len as u64).read_to_end(&mut v)?;
        Ok(v)
    };
    let boxes = walk_boxes(&read, size).ok_or_else(|| format!("{} is not an MP4 or MOV file", src.display()))?;
    let (_, moov_at, moov_len) = *boxes.list.iter().find(|(k, _, _)| k == b"moov").ok_or("the file has no moov box")?;
    let (_, mdat_at, _) = *boxes.list.iter().find(|(k, _, _)| k == b"mdat").ok_or("the file has no mdat box")?;
    if moov_len > MAX_MOOV {
        return Err("the moov box is too large to move".into());
    }
    let mut moov = read(moov_at, moov_len as usize).map_err(|e| e.to_string())?;
    if moov_at < mdat_at {
        // already at the front: a plain copy
        std::fs::copy(src, dst).map_err(|e| format!("{}: {e}", dst.display()))?;
        return Ok(());
    }
    shift_chunk_offsets(&mut moov, 8, moov_len as usize, mdat_at, moov_len, 0)?;
    let tmp = dst.with_extension("faststart.tmp");
    let mut write = || -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut copy = |from: u64, to: u64, out: &mut dyn std::io::Write| -> std::io::Result<()> {
            f.seek(std::io::SeekFrom::Start(from))?;
            std::io::copy(&mut (&f).take(to.saturating_sub(from)), out)?;
            Ok(())
        };
        copy(0, mdat_at, &mut out)?;
        out.write_all(&moov)?;
        copy(mdat_at, moov_at, &mut out)?;
        copy(moov_at.saturating_add(moov_len), size, &mut out)?;
        out.flush()?;
        out.get_ref().sync_all()
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: {e}", dst.display()));
    }
    std::fs::rename(&tmp, dst).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", dst.display())
    })
}

/// Add `delta` to the chunk offsets (`stco`, `co64`) at or past `from` in the boxes of
/// `buf[start..end]`, descending into the containers that hold sample tables.
fn shift_chunk_offsets(buf: &mut [u8], start: usize, end: usize, from: u64, delta: u64, depth: u32) -> std::result::Result<(), String> {
    if depth > 8 {
        return Ok(());
    }
    let body = buf.get(start..end).ok_or("damaged moov")?;
    let kids: Vec<([u8; 4], usize, usize)> = children(body).into_iter().map(|(k, s, e)| (k, s + start, e + start)).collect();
    for (kind, s, e) in kids {
        match &kind {
            b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts" | b"dinf" => shift_chunk_offsets(buf, s + 8, e, from, delta, depth + 1)?,
            b"stco" | b"co64" => {
                let wide = &kind == b"co64";
                let n = buf.get(s + 12..s + 16).and_then(|x| x.try_into().ok()).map(u32::from_be_bytes).ok_or("damaged chunk offsets")? as usize;
                let w = if wide { 8 } else { 4 };
                if n.checked_mul(w).and_then(|b| b.checked_add(16)).is_none_or(|need| need > e - s) {
                    return Err("damaged chunk offsets".into());
                }
                for i in 0..n {
                    let at = s + 16 + i * w;
                    let cell = buf.get_mut(at..at + w).ok_or("damaged chunk offsets")?;
                    if wide {
                        let v = u64::from_be_bytes(cell.try_into().map_err(|_| "damaged chunk offsets")?);
                        let v = if v >= from { v.checked_add(delta).ok_or("chunk offset overflow")? } else { v };
                        cell.copy_from_slice(&v.to_be_bytes());
                    } else {
                        let v = u32::from_be_bytes(cell.try_into().map_err(|_| "damaged chunk offsets")?) as u64;
                        let v = if v >= from { v + delta } else { v };
                        let v = u32::try_from(v).map_err(|_| "the file is too large for a fast-start copy (its chunk offsets would pass 4 GB)")?;
                        cell.copy_from_slice(&v.to_be_bytes());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Export settings for a copy of `p` that `d` takes, and what they change (in words).
pub fn fix_settings(d: Destination, p: &Probe, out: &str) -> (filmcraft_export::ExportSettings, Vec<String>) {
    use filmcraft_export::{AudioCodec, AudioSettings, BitrateMode, ExportSettings, Format, LoudnessNormalization, Scaling};
    let spec = d.spec();
    let mut said = vec!["H.264 video, AAC audio, MP4".to_string()];
    let mut s = ExportSettings {
        format: Format::H264,
        path: out.to_string(),
        include_audio: p.audio.is_some(),
        bitrate_mode: BitrateMode::Vbr1Pass,
        bitrate_kbps: spec.video_kbps.max(1000),
        max_bitrate_kbps: Some(spec.max_video_kbps.map_or(spec.video_kbps.saturating_mul(3) / 2, |m| (m as u32).min(spec.video_kbps.saturating_mul(3) / 2))),
        scaling: Scaling::ScaleToFit,
        audio: AudioSettings { codec: AudioCodec::Aac, sample_rate: Some(48_000), channels: 2, bitrate_kbps: spec.audio_kbps, bits: 16, ..Default::default() },
        ..Default::default()
    };
    if let Some(v) = &p.video {
        // the size: within the width limit, the picture's own shape kept (even numbers)
        let (w, h) = (v.width.max(2), v.height.max(2));
        let max_w = spec.max_width.unwrap_or(u32::MAX).min(if h > w { spec.target.0 } else { spec.target.1.max(spec.target.0) });
        if w > max_w {
            let nh = ((h as u64 * max_w as u64 / w as u64) as u32 / 2 * 2).max(2);
            s.frame_size = Some((max_w / 2 * 2, nh));
            said.push(format!("scaled to {}×{nh}", max_w / 2 * 2));
        }
        if let Some((lo, hi)) = spec.fps
            && (v.fps < lo - 0.01 || v.fps > hi + 0.01)
        {
            let r = if v.fps > hi { filmcraft_time::FrameRate::new(hi as i64, 1) } else { filmcraft_time::FrameRate::FPS_30 };
            said.push(format!("{} fps", r.as_f64().round()));
            s.frame_rate = Some(r);
        }
    }
    // the size limit: the bitrate that fits
    if p.seconds > 0.0 {
        let fit = (spec.max_bytes as f64 * 8.0 * 0.85 / p.seconds / 1000.0) as u64;
        let audio = u64::from(spec.audio_kbps);
        let video_fit = fit.saturating_sub(audio).max(1000);
        if video_fit < u64::from(s.bitrate_kbps) {
            s.bitrate_kbps = video_fit as u32;
            s.max_bitrate_kbps = Some((video_fit as u32).saturating_mul(5) / 4);
        }
    }
    said.push(format!("{:.1} Mbps", s.bitrate_kbps as f64 / 1000.0));
    if p.audio.is_some() && spec.loudness {
        s.effects.loudness = LoudnessNormalization { enabled: true, target_lufs: SOCIAL_LUFS, true_peak_dbtp: TRUE_PEAK_DBTP };
        said.push("loudness −14 LUFS, peaks −1 dBTP".into());
    }
    if p.audio.is_some() {
        said.push(format!("48 kHz stereo AAC {} kbps", spec.audio_kbps));
    }
    (s, said)
}

/// Re-encode `src` (opened from `path`) to `out` with `settings` (blocking; `progress` as an export's).
pub fn reexport(
    path: &str,
    src: SharedSource,
    settings: &filmcraft_export::ExportSettings,
    progress: &filmcraft_export::Progress,
) -> std::result::Result<filmcraft_export::Report, String> {
    use filmcraft_project::{ItemKind, MediaClip, MediaRef, Project, SequenceSettings, TrackKind};
    use filmcraft_time::{Tick, TimeRange};
    let info = src.info().clone();
    let rate = info.frame_rate();
    let rate = if rate.num > 0 && rate.den > 0 { rate } else { filmcraft_time::FrameRate::FPS_30 };
    let mut st = SequenceSettings::default();
    if let Some(v) = &info.video {
        st.width = v.width.max(2);
        st.height = v.height.max(2);
        st.par = if v.par.0 > 0 && v.par.1 > 0 { v.par } else { (1, 1) };
    }
    st.frame_rate = rate;
    st.sample_rate = info.audio().map_or(48_000, |a| a.sample_rate.clamp(8_000, 192_000));
    let name = Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into());
    let mut p = Project::new("fixed copy");
    let clip = MediaClip {
        media: MediaRef::File { path: path.to_string() },
        info: info.clone(),
        interpret: Default::default(),
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    let item = p.add_item(&name, filmcraft_project::Label::Iris, ItemKind::Media(clip), None);
    let (has_v, has_a) = (info.video.is_some(), info.has_audio());
    let seq = p.new_sequence("fixed copy", st, usize::from(has_v), usize::from(has_a), None);
    let range = TimeRange { start: Tick::ZERO, duration: info.duration };
    let mut placed = Vec::new();
    if has_v && let Some(ti) = p.make_track_item(item, TrackKind::Video, Tick::ZERO, range, rate) {
        placed.push((TrackKind::Video, ti));
    }
    if has_a && let Some(ti) = p.make_track_item(item, TrackKind::Audio, Tick::ZERO, range, rate) {
        placed.push((TrackKind::Audio, ti));
    }
    let q = p.sequence_mut(seq).ok_or("internal: no sequence for the copy")?;
    for (k, ti) in placed {
        let t = if k == TrackKind::Video { q.video_tracks.first_mut() } else { q.audio_tracks.first_mut() };
        if let Some(t) = t {
            t.items.push(ti);
        }
    }
    let project = Arc::new(p);
    let provider = move |id: filmcraft_project::ItemId| (id == item).then(|| src.clone());
    filmcraft_export::export(&project, seq, settings, &provider, progress).map_err(|e| e.to_string())
}

/// `<dir>/<stem> (<label>).mp4`, numbered when taken.
pub fn fixed_copy_path(path: &str, d: Destination) -> String {
    let p = Path::new(path);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into());
    for n in 1..1000 {
        let name = if n == 1 { format!("{stem} ({}).mp4", d.label()) } else { format!("{stem} ({}) {n}.mp4", d.label()) };
        let c = dir.join(name);
        if !c.exists() {
            return c.to_string_lossy().into_owned();
        }
    }
    dir.join(format!("{stem} ({}) {}.mp4", d.label(), std::process::id())).to_string_lossy().into_owned()
}

// -------------------------------------------------------------------------------------------
// commands
// -------------------------------------------------------------------------------------------

/// A job's answer, filled when it finishes.
pub type Outcome = Arc<Mutex<Option<Value>>>;

/// The fixed-copy jobs' answers (`publish` in `jobs.list`).
#[derive(Default)]
pub struct PublishState {
    pub outcomes: HashMap<u64, Outcome>,
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "publish.destinations",
            label: "Destinations",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: destinations,
            journal: false,
        },
        CommandSpec {
            id: "publish.preflight",
            label: "Check a File for a Destination",
            menu: &[],
            shortcut: None,
            params: r#"{"destination":"mediaFile"|"youtube"|"youtubeShort"|"instagramReel","path":str,"loudness":bool=true}"#,
            enabled: always,
            run: preflight,
            journal: false,
        },
        CommandSpec {
            id: "publish.fixCopy",
            label: "Export a Fixed Copy",
            menu: &[],
            shortcut: None,
            params: r#"{"destination":"youtube"|"youtubeShort"|"instagramReel","path":str,"out":str?,"wait":bool=false}"#,
            enabled: always,
            run: fix_copy,
            journal: false,
        },
        CommandSpec {
            id: "publish.run",
            label: "Publish",
            menu: &[],
            shortcut: None,
            params: r#"{"destination":"youtube"|"youtubeShort"|"instagramReel","path":str,"loudness":bool=false,"wait":bool=false,"title":str?,"description":str?,"tags":[str]?,"privacy":str?,"channel":str?,"playlist":str?,"thumbnail":str?,"caption":str?,"shareToFeed":bool?,"coverSeconds":f64?,"coverUrl":str?,"videoUrl":str?,"uploadOnly":bool?}"#,
            enabled: always,
            run: publish_run,
            journal: false,
        },
    ]
}

fn dest_param(cmd: &str, p: &Value) -> Result<Destination> {
    match p.get("destination") {
        Some(Value::String(s)) => Destination::parse(s).map_err(|e| bad(cmd, e)),
        None | Some(Value::Null) => Err(bad(cmd, "need `destination`: mediaFile, youtube, youtubeShort or instagramReel")),
        Some(_) => Err(bad(cmd, "`destination` must be a string")),
    }
}

fn path_param(cmd: &str, p: &Value, key: &str) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) if s.chars().count() > 4096 || s.contains('\0') => Err(bad(cmd, format!("`{key}` is not a usable path"))),
        Some(Value::String(s)) => Ok(Some(s.trim().to_string())),
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

/// The connect state of a destination, from disk (cheap: asked while drawing).
pub fn connect_state(s: &Session, d: Destination) -> Value {
    match d {
        Destination::MediaFile => json!({"needed": false, "ready": true}),
        Destination::Youtube | Destination::YoutubeShort => match &s.youtube.publisher {
            None => json!({"needed": true, "ready": false, "available": false, "why": "YouTube is not available in this app"}),
            Some(p) => {
                let ready = p.ready();
                json!({"needed": true, "available": true, "ready": ready.is_ok(), "why": ready.err(), "canConnect": p.connect_ready().is_ok(), "connectWhy": p.connect_ready().err(), "connect": "youtube.connect"})
            }
        },
        Destination::InstagramReel => match &s.instagram.publisher {
            None => json!({"needed": true, "ready": false, "available": false, "why": "Instagram is not available in this app"}),
            Some(p) => {
                let ready = p.ready();
                json!({"needed": true, "available": true, "ready": ready.is_ok(), "why": ready.err(), "canConnect": p.connect_ready().is_ok(), "connectWhy": p.connect_ready().err(), "connect": "instagram.connect"})
            }
        },
    }
}

fn destinations(s: &mut Session, _p: &Value) -> Result<Value> {
    let rows: Vec<Value> =
        Destination::ALL.iter().map(|d| json!({"id": d.id(), "label": d.label(), "publish": d.publish_command(), "connect": connect_state(s, *d)})).collect();
    Ok(json!({"destinations": rows}))
}

fn preflight(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "publish.preflight";
    let d = dest_param(cmd, p)?;
    let path = path_param(cmd, p, "path")?.ok_or_else(|| bad(cmd, "need `path`: the file to check"))?;
    let loudness = bool_p(p, "loudness").unwrap_or(true);
    let (probe, _) = probe(s, &path, loudness).map_err(|e| bad(cmd, e))?;
    Ok(report(d, &probe))
}

fn fix_copy(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "publish.fixCopy";
    let d = dest_param(cmd, p)?;
    if d == Destination::MediaFile {
        return Err(bad(cmd, "the Media File takes any export; pick youtube, youtubeShort or instagramReel"));
    }
    let path = path_param(cmd, p, "path")?.ok_or_else(|| bad(cmd, "need `path`: the file to fix"))?;
    let out = match path_param(cmd, p, "out")? {
        Some(o) if Path::new(&o) == Path::new(&path) => return Err(bad(cmd, "`out` must be another file than `path`")),
        Some(o) => o,
        None => fixed_copy_path(&path, d),
    };
    let wait = bool_p(p, "wait").unwrap_or(false);
    let (probe, src) = probe(s, &path, true).map_err(|e| bad(cmd, e))?;
    let before = checks(d, &probe);
    // a failure no copy mends (the length, the shape) makes a copy pointless
    let unfixable: Vec<String> = before
        .iter()
        .filter(|c| c.status == Status::Fail && c.fixable.is_none())
        .map(|c| format!("{}: {}", c.label, c.fix.clone().unwrap_or_default()))
        .collect();
    if !unfixable.is_empty() {
        return Err(bad(cmd, format!("a copy cannot fix this: {}", unfixable.join("; "))));
    }
    let need: Option<Fix> = before.iter().filter(|c| c.status != Status::Pass && c.status != Status::Info).filter_map(|c| c.fixable).max();
    let Some(fix) = need else {
        return Err(bad(cmd, format!("{path} already passes the {} checks; nothing to fix", d.label())));
    };
    let (settings, said) = fix_settings(d, &probe, &out);
    let frames = probe.video.as_ref().map_or(1, |v| (probe.seconds * v.fps).max(1.0) as u64);
    let label = format!("Fixing {} for {}", Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), d.label());
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(if fix == Fix::Reexport { frames } else { 1 }, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = Outcome::default();
    s.publish.outcomes.insert(job.id, outcome.clone());
    let services = s.services.clone();
    let (src_path, out_path) = (path.clone(), out.clone());
    let faststart_too = d.spec().moov_front;
    let id = crate::narration::spawn(s, job, wait, "filmcraft-publish-fix", move || {
        use std::sync::atomic::Ordering;
        let t0 = web_time::Instant::now();
        let r = (|| -> std::result::Result<Value, String> {
            let mut fixes = said;
            match fix {
                Fix::Faststart => {
                    *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Moving the moov box to the front".into();
                    faststart(Path::new(&src_path), Path::new(&out_path))?;
                    fixes = vec!["moov moved to the front (nothing re-encoded)".into()];
                }
                Fix::Reexport => {
                    let mut settings = settings;
                    settings.part_of_batch = true;
                    reexport(&src_path, src, &settings, &prog)?;
                    if faststart_too {
                        *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Moving the moov box to the front".into();
                        let tmp = format!("{out_path}.moov.tmp");
                        faststart(Path::new(&out_path), Path::new(&tmp))?;
                        std::fs::rename(&tmp, &out_path).map_err(|e| format!("{out_path}: {e}"))?;
                        fixes.push("moov at the front".into());
                    }
                }
            }
            if prog.cancel.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(&out_path);
                return Err("cancelled".into());
            }
            // the copy's own pre-flight
            let size = services.file_size(&out_path).map_err(|e| format!("{out_path}: {e}"))?;
            let read = |off: u64, len: usize| services.read_range(&out_path, off, len);
            let boxes = walk_boxes(&read, size);
            let after = match services.reader(&out_path) {
                Some(Ok(r)) => filmcraft_media::reader::open_reader(&out_path, r, &filmcraft_codecs::reader_openers(), &[])
                    .map(|src| report(d, &probe_source(&out_path, size, &*src, boxes, true)))
                    .unwrap_or(Value::Null),
                _ => Value::Null,
            };
            Ok(json!({"path": out_path, "from": src_path, "destination": d.id(), "fixes": fixes, "preflight": after}))
        })();
        let r = r.map(|v| {
            let ok = v["preflight"]["ok"].as_bool().unwrap_or(false);
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
            if ok { "Fixed copy ready".to_string() } else { "Fixed copy written; some checks still fail".to_string() }
        });
        prog.done.store(prog.total.load(Ordering::Relaxed), Ordering::Relaxed);
        crate::narration::finish(&prog, &res, r, t0);
    })?;
    let mut answer = json!({"job": id, "path": out, "fix": fix, "from": path});
    if wait && let Some(j) = s.jobs.iter().find(|j| j.id == id) {
        let v = j.to_json();
        if let Some(e) = v["result"]["error"].as_str() {
            return Err(EngineError::Other(e.to_string()));
        }
        answer["publish"] = outcome_json(s, id);
    }
    Ok(answer)
}

fn publish_run(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "publish.run";
    let d = dest_param(cmd, p)?;
    let Some(target) = d.publish_command() else {
        return Err(bad(cmd, "the Media File destination is the export itself (`file.exportMedia`); pick youtube, youtubeShort or instagramReel"));
    };
    if !p.is_object() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    let path = path_param(cmd, p, "path")?;
    let has_url = p.get("videoUrl").and_then(Value::as_str).is_some_and(|u| !u.trim().is_empty());
    if path.is_none() && !(d == Destination::InstagramReel && has_url) {
        return Err(bad(cmd, "need `path`: the file to publish"));
    }
    // the pre-flight before any byte leaves (loudness only when asked: it reads all the sound)
    let pre = match &path {
        Some(path) => {
            let (probe, _) = probe(s, path, bool_p(p, "loudness").unwrap_or(false)).map_err(|e| bad(cmd, e))?;
            let r = report(d, &probe);
            if r["ok"] != true {
                let failed: Vec<String> = r["checks"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|c| c["status"] == "fail")
                    .map(|c| {
                        format!(
                            "{} is {} ({} takes {}): {}",
                            c["label"].as_str().unwrap_or_default(),
                            c["value"].as_str().unwrap_or_default(),
                            d.label(),
                            c["need"].as_str().unwrap_or_default(),
                            c["fix"].as_str().unwrap_or_default()
                        )
                    })
                    .collect();
                return Err(bad(cmd, format!("{path} does not pass the {} pre-flight: {}", d.label(), failed.join("; "))));
            }
            r
        }
        None => Value::Null,
    };
    let mut params = p.clone();
    if let Some(o) = params.as_object_mut() {
        o.remove("destination");
        o.remove("loudness");
        if d != Destination::InstagramReel {
            // YouTube needs a title: the file's name unless given
            if o.get("title").and_then(Value::as_str).is_none_or(|t| t.trim().is_empty())
                && let Some(path) = &path
            {
                let stem: String = Path::new(path)
                    .file_stem()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
                    .chars()
                    .take(crate::youtube::MAX_TITLE_CHARS)
                    .collect();
                o.insert("title".into(), json!(if stem.trim().is_empty() { "FilmCraft export".to_string() } else { stem }));
            }
        }
    }
    let mut answer = s.execute(target, params)?;
    if let Some(o) = answer.as_object_mut() {
        o.insert("destination".into(), json!(d.id()));
        o.insert("via".into(), json!(target));
        o.insert("preflight".into(), json!({"ok": pre["ok"], "warnings": pre["warnings"], "checks": pre["checks"]}));
    }
    Ok(answer)
}

/// Job `job`'s answer (`null` while it runs, after a failure, or for another job).
pub fn outcome_json(s: &Session, job: u64) -> Value {
    s.publish.outcomes.get(&job).and_then(|o| o.lock().unwrap_or_else(PoisonError::into_inner).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` is a fixed-copy job.
pub fn is_job(s: &Session, job: u64) -> bool {
    s.publish.outcomes.contains_key(&job)
}

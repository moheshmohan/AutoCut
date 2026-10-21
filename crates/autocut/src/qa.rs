//! Automatic QA of a built edit — a showcase ([`crate::showcase::build`]) in particular: the
//! defects a review by hand found in a real build (a shot without the subject in view, sub-frame
//! gaps after slowed shots, the voice quieter than what plays between the lines, clipping),
//! found by machine instead.
//!
//! [`check`] reads the sequence (`sequence.inspect`), renders frames ([`Executor::render`]),
//! exports the mix to a temporary WAV (`file.exportMedia`) and, when a grounding model is at
//! hand, looks for the subject in each shot. Every check gives findings with a time and a
//! severity — error: a viewer sees or hears it; warning: worth a look; info: for the record:
//!
//! | check | finds |
//! |---|---|
//! | `joins` | on the picture track (V2): gaps of a frame or more, gaps that hold a frame boundary (that frame renders black) and overlaps (errors), clips that do not quite touch (warnings); on the shots' sound track (A1): neighbours that do not touch, so their fades meet silence instead of crossfading (warnings; not across a photo, which is silent on purpose); time where no video track has a clip (errors; a picture that ends before the sound: info) |
//! | `black` | rendered frames whose picture is black or nearly (92 % of the frame darker than 5 % luma, so a counter or lower third over black still counts) — sampled every 2 s, at each picture clip's first frame and at the first frame after it — outside dips to black (errors) |
//! | `sides` | rendered frames where a side of the frame — the left or right pillar, the bar above or below — is black while the picture in the middle still shows ([`black_sides`]): sampled around every transition (its first frames, either side of the cut, its last frames; over a dip its first and last quarters) and at the first and last frame of each picture shown whole over a blurred copy (the showcase's `fit: "full"`, a clip over Auto Reframe's Blur Fill track); flagged only where that side shows something in the frame just before or after the transition or in the middle of the picture shown whole, so letterboxing kept on purpose is not one (errors, with the time and the side) |
//! | `mix` | the mix exported as 32-bit float WAV and metered (BS.1770): integrated loudness outside −16…−14 LUFS (±0.5 LU) and true peak above −1 dBTP (warnings); true peak above 0 dBTP or samples at full scale: clipping (error); the median short-term loudness during narration clips (A2) vs between them: the voice less than 4 dB louder (warning), or not louder at all (error); and per stem (`audio.analyzeMix`, [`mix_findings`]), for small speakers: the narration less than 12 LU above the music while it speaks, the Mix's limiter taking more than 2 dB off the peaks, more than a quarter of the mix's energy below 120 Hz, a narration clip with a clipped source (warnings) |
//! | `subject` | shots where the grounding model (`AUTOCUT_ANALYSIS_MODEL`, the analysis' prompt) does not find the subject in a rendered frame (warnings); skipped when no model is reachable |
//! | `privacy` | from what the privacy pass hid ([`Options::privacy`]): a redaction mask covering more than [`Options::max_mask`] (a fifth) of a clip's picture at any keyframe (warnings: a blur that size hides the shot, as one on a close-up of a tablet did); the faces, plates and named places hidden in each entry (`privacy` in the report). And, for a show that hides faces ([`Options::faces`]), the faces the finished picture still shows: frames every half second and at each shot's start rendered as they export, the face detector run on them (`sequence.checkPrivacy`), and every face at least [`Options::face_min_height`] of the frame tall that no redaction mask covers is a finding unless the vision model twice calls it not a person and it is on the robot (inside the shot's subject box) or the detector is not sure of it ([`face_severity`]): an error for a person the detector scores [`FACE_ERROR_SCORE`] or more (on the robot, both looks must say so; away from it, a face nobody could tell counts too), a warning to look at otherwise. Each with its time and box (`unblurredFaces` in the report) and a crop on the privacy sheet; [`QaReport::publish_blocker`] stops an upload when there is one (the bug of 2026-10-21: two visitors next to Ameca, published unblurred, while this check said OK) |
//! | `safe_zones` | captions, titles and lower thirds (`sequence.checkSafeZones`) that reach into the parts of the frame [`Options::platform`]'s own interface covers — YouTube Shorts, Instagram Reels, TikTok; a vertical sequence without one: all of them at once (warnings, with the zone and how much of it is in there); skipped on a horizontal sequence without a platform |
//!
//! The report ([`QaReport`], serde) also holds a contact sheet: one rendered frame per entry
//! under its number, the band red or amber when the entry has an error or a warning
//! ([`write_next_to`] writes `<out>.qa.json` and `<out>.qa.png`), and a privacy sheet: one frame
//! per clip with masks, where most of it is hidden (`<out>.privacy.png`). Nothing here edits the
//! project; a check that cannot run is recorded as skipped, never fatal.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use serde_json::{Value, json};

use crate::Executor;
use crate::llm::ChatModel;
use crate::showcase::{Built, PlacedEntry};

const TPS: i128 = filmcraft_time::TICKS_PER_SECOND as i128;
const TPS_F: f64 = filmcraft_time::TICKS_PER_SECOND as f64;

/// Tracks and clips read from a sequence, at most.
const MAX_TRACKS: usize = 256;
const MAX_CLIPS: usize = 100_000;
/// Shots looked at for the subject, and tiles on the contact sheet, at most.
const MAX_SHOTS: usize = 1000;
const MAX_TILES: usize = 300;
/// Longest mix measured, seconds.
const MAX_MIX_SECONDS: f64 = 4.0 * 3600.0;
/// A gap in the shots' sound longer than this is a deliberate silence, not a join.
const SOUND_JOIN: f64 = 2.0;
/// A frame is black when at least [`BLACK_SHARE`] of it is darker than [`DARK`] (luma, 0..1):
/// what sits over the picture — the counter, a lower third, captions — covers a few percent of a
/// frame, so a black picture under them still counts; a dim shot does not.
const DARK: f64 = 0.05;
const BLACK_SHARE: f64 = 0.92;
/// Loudness below this (LUFS) is silence for the voice comparison.
const SILENT_LUFS: f64 = -70.0;
/// For small speakers ([`mix_findings`]): the narration at least this far above the music while
/// it speaks (LU; 15 is the speech mix's aim, measured 22–28 on a finished show), the Mix's
/// limiter taking at most this much off (dB; the first finished show's took 3.7 dB, the speech
/// mix's 1), and at most this share of the energy below 120 Hz (the first show: 38 %, the
/// speech mix: 15 %).
pub const MIN_SPEECH_OVER_MUSIC: f64 = 12.0;
pub const MAX_LIMITER_DB: f64 = 2.0;
pub const MAX_LOW_SHARE: f64 = 0.25;
/// A limiter set to the ceiling lands a hair over it (−0.99997 dBTP for −1): not a finding.
const PEAK_TOLERANCE: f64 = 0.05;
/// Integrated loudness this close to the target range (LU) is on target: meters and loudness
/// normalisation agree to about half a unit.
const LOUDNESS_TOLERANCE: f64 = 0.5;

/// What to check, and against what.
#[derive(Clone, Debug)]
pub struct Options {
    /// What every shot should show ("robot").
    pub subject: String,
    /// The entries as built ([`Built::layout`]); empty: each clip on the picture track is an
    /// entry of its own.
    pub entries: Vec<PlacedEntry>,
    /// The main picture track, the shots' sound and the narration (showcase layout: V2, A1, A2).
    pub picture_track: String,
    pub sound_track: String,
    pub voice_track: String,
    /// Seconds between frames sampled for the black check.
    pub sample_every: f64,
    /// Longest side of frames rendered for the black check and the contact sheet.
    pub frame_side: u32,
    /// Longest side of frames shown to the grounding model.
    pub ground_side: u32,
    /// Most frames rendered for the black check (sampling thins out beyond it).
    pub max_frames: usize,
    /// Export and measure the mix.
    pub mix: bool,
    /// Where the temporary mixdown goes (default: the system temporary directory).
    pub work_dir: Option<PathBuf>,
    /// Integrated loudness target, LUFS (YouTube plays at −14).
    pub loudness: (f64, f64),
    /// Highest true peak, dBTP.
    pub max_true_peak: f64,
    /// How much louder (dB) the voice must be than what plays between the lines.
    pub voice_margin: f64,
    /// The music's track, for the speech-over-music measure (showcase layout: A3).
    pub music_track: String,
    /// Measure the mix per stem too (`audio.analyzeMix`; about four renders of the mix).
    pub analyze_mix: bool,
    /// The least the narration should stand above the music while it speaks (LU, median).
    pub min_speech_over_music: f64,
    /// The most the Mix's limiter should take off the peaks (dB).
    pub max_limiter_db: f64,
    /// The largest share of the mix's energy below 120 Hz.
    pub max_low_share: f64,
    /// What the privacy pass hid ([`Built::privacy`]); empty: the privacy check is skipped.
    pub privacy: Vec<crate::privacy::Hidden>,
    /// The largest share of a clip's picture one redaction mask may cover.
    pub max_mask: f64,
    /// Look for faces the finished picture shows that no redaction mask hides
    /// (`sequence.checkPrivacy`): on for a show that hides faces.
    pub faces: bool,
    /// Seconds between the frames looked at for faces (each shot's first frames too).
    pub face_every: f64,
    /// Longest side of the frames the face detector sees.
    pub face_side: u32,
    /// The smallest face counted, as a share of the frame's height (the redaction's own rule).
    pub face_min_height: f64,
    /// The platform whose interface captions, titles and lower thirds must keep clear of
    /// (`youtube`, `reels`, `tiktok`, `vertical`); None: all of them on a vertical sequence, the
    /// check skipped on a horizontal one.
    pub platform: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            subject: "robot".into(),
            entries: Vec::new(),
            picture_track: "V2".into(),
            sound_track: "A1".into(),
            voice_track: "A2".into(),
            sample_every: 2.0,
            frame_side: 320,
            ground_side: 640,
            max_frames: 1500,
            mix: true,
            work_dir: None,
            loudness: (-16.0, -14.0),
            max_true_peak: -1.0,
            voice_margin: 4.0,
            music_track: "A3".into(),
            analyze_mix: true,
            min_speech_over_music: MIN_SPEECH_OVER_MUSIC,
            max_limiter_db: MAX_LIMITER_DB,
            max_low_share: MAX_LOW_SHARE,
            privacy: Vec::new(),
            max_mask: 0.2,
            faces: false,
            face_every: 0.5,
            face_side: 1280,
            face_min_height: 0.05,
            platform: None,
        }
    }
}

impl Options {
    /// The checks of a showcase built with `opts`.
    pub fn for_showcase(opts: &crate::showcase::Options, built: &Built) -> Self {
        Self {
            subject: opts.subject.clone(),
            entries: built.layout.clone(),
            privacy: built.privacy.clone(),
            faces: opts.redact.as_ref().is_some_and(|r| r.faces),
            face_min_height: opts.redact.as_ref().and_then(|r| r.min_face_height).filter(|h| h.is_finite() && *h > 0.0).unwrap_or(0.05),
            ..Self::default()
        }
    }
}

/// How much a finding matters (ordered: the worst is the greatest).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// One thing a check found.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    /// `joins`, `black`, `sides`, `mix`, `subject`, `privacy`, `safe_zones` or `sequence`.
    pub check: &'static str,
    pub severity: Severity,
    /// Where, in seconds of the sequence.
    pub seconds: f64,
    /// Where it ends, for a stretch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub until: Option<f64>,
    /// The entry (showcase number) it falls in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip: Option<u64>,
    pub message: String,
}

impl Finding {
    fn new(check: &'static str, severity: Severity, seconds: f64, message: String) -> Self {
        Self { check, severity, seconds, until: None, entry: None, clip: None, message }
    }
    fn until(mut self, t: f64) -> Self {
        self.until = Some(t);
        self
    }
    fn clip(mut self, c: Option<u64>) -> Self {
        self.clip = c;
        self
    }
}

/// How a check went.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub name: &'static str,
    /// `ok`, `findings` or `skipped`.
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Check {
    /// `findings` when the check found a warning or an error (notes for the record are ok).
    fn ran(name: &'static str, findings: &[Finding], note: Option<String>) -> Self {
        let any = findings.iter().any(|f| f.check == name && f.severity >= Severity::Warning);
        Self { name, status: if any { "findings" } else { "ok" }, note }
    }
    fn skipped(name: &'static str, why: impl Into<String>) -> Self {
        Self { name, status: "skipped", note: Some(why.into()) }
    }
}

/// The mix as measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Loudness {
    pub seconds: f64,
    pub integrated_lufs: Option<f64>,
    pub loudness_range_lu: f64,
    pub true_peak_dbtp: Option<f64>,
    pub sample_peak_dbfs: Option<f64>,
    /// Where the first sample at or above full scale is.
    pub first_over: Option<f64>,
    /// Median loudness while the narration speaks, and between its lines.
    pub voice_lufs: Option<f64>,
    pub between_lufs: Option<f64>,
    pub voice_margin_db: Option<f64>,
    /// `shortTerm` (3 s windows) or, with lines or pauses too short for those, `momentary` (400 ms).
    pub window: Option<&'static str>,
    /// From the per-stem analysis ([`QaReport::mix_analysis`]): the narration above the music and
    /// above everything else while it speaks (median, LU), the most the Mix's limiter takes off
    /// (dB), and the mix's share of energy below 120 Hz and between 1 and 4 kHz.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_over_music_lu: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_over_rest_lu: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limiter_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_share: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_share: Option<f64>,
}

/// Everything [`check`] found.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QaReport {
    pub sequence: Option<u64>,
    pub seconds: f64,
    pub fps: f64,
    pub errors: usize,
    pub warnings: usize,
    pub checks: Vec<Check>,
    pub findings: Vec<Finding>,
    pub loudness: Option<Loudness>,
    /// The mix per stem, as `audio.analyzeMix` measured it (bands, speech over the rest,
    /// limiting, clipping in the narration).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mix_analysis: Option<Value>,
    pub frames_rendered: usize,
    /// Where the contact sheet was written ([`write_next_to`]).
    pub contact_sheet: Option<String>,
    /// The contact sheet, PNG.
    #[serde(skip)]
    pub contact_sheet_png: Option<Vec<u8>>,
    /// What the privacy pass hid in each entry.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub privacy: Vec<EntryPrivacy>,
    /// Where the privacy sheet was written ([`write_next_to`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy_sheet: Option<String>,
    /// The privacy sheet, PNG: a frame of each clip with masks, where most of it is hidden.
    #[serde(skip)]
    pub privacy_sheet_png: Option<Vec<u8>>,
    /// Faces the finished picture shows that no redaction mask hides ([`Options::faces`]), with
    /// when and where: what a platform's own blur tool needs if the video is already out.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unblurred_faces: Vec<UnblurredFace>,
}

/// What the privacy pass hid in one entry: faces, number plates, places of named targets (logos,
/// signs), masks, and the largest share of a picture one mask covers.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryPrivacy {
    pub entry: usize,
    pub label: String,
    pub faces: usize,
    pub plates: usize,
    pub named: usize,
    pub masks: usize,
    pub largest_mask: f64,
}

impl QaReport {
    /// Findings of `check` at `severity` or worse.
    pub fn of(&self, check: &str, severity: Severity) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.check == check && f.severity >= severity).collect()
    }

    /// "2 errors, 3 warnings (black 1 error, mix 1 error 1 warning, …); subject skipped (…)".
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for c in &self.checks {
            let errors = self.of(c.name, Severity::Error).len();
            let warnings = self.of(c.name, Severity::Warning).len().saturating_sub(errors);
            match c.status {
                "skipped" => parts.push(format!("{} skipped ({})", c.name, c.note.clone().unwrap_or_default())),
                _ if errors + warnings > 0 => parts.push(format!("{} {errors} error(s) {warnings} warning(s)", c.name)),
                _ => parts.push(format!("{} ok", c.name)),
            }
        }
        format!("{} error(s), {} warning(s): {}", self.errors, self.warnings, parts.join("; "))
    }

    /// Why this video must not be published as it is: the privacy check's errors (a person's
    /// face shown unblurred), or none.
    pub fn publish_blocker(&self) -> Option<String> {
        let errors = self.of("privacy", Severity::Error);
        let first = errors.first()?;
        Some(format!(
            "the privacy check found {} unblurred face(s) of people (the first at {}{}); not published: fix the blur (the faces and times are in the QA report's unblurredFaces and on the privacy sheet)",
            errors.len(),
            stamp(first.seconds),
            first.entry.map(|e| format!(", entry {e}")).unwrap_or_default()
        ))
    }

    pub(crate) fn tally(&mut self) {
        self.findings.sort_by(|a, b| a.seconds.total_cmp(&b.seconds).then(b.severity.cmp(&a.severity)));
        self.errors = self.findings.iter().filter(|f| f.severity == Severity::Error).count();
        self.warnings = self.findings.iter().filter(|f| f.severity == Severity::Warning).count();
    }
}

/// Write `<out>.qa.json` and, when there is one, the contact sheet `<out>.qa.png` next to `out`
/// (the export): the paths written. The privacy sheet, when there is one, goes to
/// `<out>.privacy.png` (its path in [`QaReport::privacy_sheet`]).
pub fn write_next_to(report: &mut QaReport, out: &Path) -> Result<(PathBuf, Option<PathBuf>), String> {
    let json_path = out.with_extension("qa.json");
    let mut png_path = None;
    if let Some(png) = &report.contact_sheet_png {
        let p = out.with_extension("qa.png");
        std::fs::write(&p, png).map_err(|e| format!("{}: {e}", p.display()))?;
        report.contact_sheet = Some(p.display().to_string());
        png_path = Some(p);
    }
    if let Some(png) = &report.privacy_sheet_png {
        let p = out.with_extension("privacy.png");
        std::fs::write(&p, png).map_err(|e| format!("{}: {e}", p.display()))?;
        report.privacy_sheet = Some(p.display().to_string());
    }
    let text = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    std::fs::write(&json_path, format!("{text}\n")).map_err(|e| format!("{}: {e}", json_path.display()))?;
    Ok((json_path, png_path))
}

// ------------------------------------------------------------------ the sequence

#[derive(Clone, Debug, Default, PartialEq)]
struct Clip {
    id: u64,
    name: String,
    start: i64,
    end: i64,
    enabled: bool,
}

/// A transition: its span (ticks), the clips it goes from and to, and its effect.
#[derive(Clone, Debug, Default, PartialEq)]
struct Trans {
    start: i64,
    end: i64,
    from: Option<u64>,
    to: Option<u64>,
    effect: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Track {
    name: String,
    enabled: bool,
    clips: Vec<Clip>,
    /// Dip-to-black transitions, ticks.
    dips: Vec<(i64, i64)>,
    transitions: Vec<Trans>,
}

/// What QA needs of `sequence.inspect`, in ticks; hostile values are dropped or replaced.
#[derive(Clone, Debug, PartialEq)]
struct Seq {
    id: Option<u64>,
    num: i64,
    den: i64,
    aspect: f64,
    /// Taller than wide (a short).
    portrait: bool,
    duration: i64,
    video: Vec<Track>,
    audio: Vec<Track>,
}

fn parse(v: &Value) -> Seq {
    let rate = &v["settings"]["frame_rate"];
    let valid = |x: &Value| x.as_i64().filter(|n| (1..=1_000_000).contains(n));
    let (num, den) = match (valid(&rate["num"]), valid(&rate["den"])) {
        (Some(n), Some(d)) => (n, d),
        _ => (30, 1),
    };
    let (w, h) = (v["settings"]["width"].as_f64().unwrap_or(16.0), v["settings"]["height"].as_f64().unwrap_or(9.0));
    let aspect = if w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0 { (w / h).clamp(0.1, 10.0) } else { 16.0 / 9.0 };
    let track = |t: &Value| {
        let mut clips: Vec<Clip> = t["items"]
            .as_array()
            .map(|a| {
                a.iter()
                    .take(MAX_CLIPS)
                    .filter_map(|c| {
                        Some(Clip {
                            id: c["clip"].as_u64()?,
                            name: c["name"].as_str().unwrap_or_default().chars().take(200).collect(),
                            start: c["start"].as_i64()?,
                            end: c["end"].as_i64()?,
                            enabled: c["enabled"].as_bool().unwrap_or(true),
                        })
                    })
                    .filter(|c| c.end >= c.start)
                    .collect()
            })
            .unwrap_or_default();
        clips.sort_by_key(|c| (c.start, c.end));
        let dips = t["transitions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .take(MAX_CLIPS)
                    .filter(|x| x["effect"].as_str().is_some_and(|e| e.contains("dip_to_black")))
                    .filter_map(|x| {
                        let s = x["start"].as_i64()?;
                        Some((s, s.checked_add(x["duration"].as_i64()?.max(0))?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut transitions: Vec<Trans> = t["transitions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .take(MAX_CLIPS)
                    .filter_map(|x| {
                        let start = x["start"].as_i64()?;
                        Some(Trans {
                            start,
                            end: start.checked_add(x["duration"].as_i64()?.max(0))?,
                            from: x["from"].as_u64(),
                            to: x["to"].as_u64(),
                            effect: x["effect"].as_str().unwrap_or("transition").chars().take(60).collect(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        transitions.sort_by_key(|x| (x.start, x.end));
        Track {
            name: t["name"].as_str().unwrap_or_default().chars().take(200).collect(),
            enabled: t["enabled"].as_bool().unwrap_or(true),
            clips,
            dips,
            transitions,
        }
    };
    let tracks = |k: &str| v[k].as_array().map(|a| a.iter().take(MAX_TRACKS).map(track).collect()).unwrap_or_default();
    Seq {
        id: v["id"].as_u64(),
        num,
        den,
        aspect,
        portrait: aspect < 1.0,
        duration: v["duration"].as_i64().unwrap_or(0).max(0),
        video: tracks("video"),
        audio: tracks("audio"),
    }
}

fn secs(t: i64) -> f64 {
    t as f64 / TPS_F
}

/// "1:05.3".
fn stamp(s: f64) -> String {
    let s = if s.is_finite() { s.max(0.0) } else { 0.0 };
    let m = (s / 60.0).floor();
    format!("{}:{:04.1}", m as u64, s - m * 60.0)
}

impl Seq {
    fn fps(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Ticks per frame as the exact ratio `TPS · den / num`: numerator, denominator.
    fn frame(&self) -> (i128, i128) {
        (TPS * i128::from(self.den), i128::from(self.num))
    }

    /// `ticks` is a frame or longer.
    fn at_least_a_frame(&self, ticks: i64) -> bool {
        let (fd_n, fd_d) = self.frame();
        i128::from(ticks) * fd_d >= fd_n
    }

    /// The first frame at or after `t` (ticks ≥ 0).
    fn frame_ceil(&self, t: i64) -> i64 {
        let (fd_n, fd_d) = self.frame();
        let k = (i128::from(t.max(0)) * fd_d + fd_n - 1) / fd_n;
        i64::try_from(k).unwrap_or(i64::MAX)
    }

    /// The frame showing at `t`.
    fn frame_floor(&self, t: i64) -> i64 {
        let (fd_n, fd_d) = self.frame();
        i64::try_from(i128::from(t.max(0)) * fd_d / fd_n).unwrap_or(i64::MAX)
    }

    /// Where frame `k` starts, ticks.
    fn frame_tick(&self, k: i64) -> i64 {
        let (fd_n, fd_d) = self.frame();
        i64::try_from(i128::from(k) * fd_n / fd_d).unwrap_or(i64::MAX)
    }

    /// The first frame starting in `[a, b)`, if any: it renders whatever is (not) there.
    fn boundary_in(&self, a: i64, b: i64) -> Option<i64> {
        let k = self.frame_ceil(a);
        let (fd_n, fd_d) = self.frame();
        (i128::from(k) * fd_n < i128::from(b) * fd_d).then_some(k)
    }

    /// A track by its label ("V2", "A1").
    fn track(&self, label: &str) -> Option<&Track> {
        let (kind, n) = label.trim().split_at_checked(1)?;
        let i = n.parse::<usize>().ok()?.checked_sub(1)?;
        match kind {
            "V" | "v" => self.video.get(i),
            "A" | "a" => self.audio.get(i),
            _ => None,
        }
    }

    /// Covered stretches of the picture: enabled clips on enabled video tracks, merged.
    fn picture(&self) -> Vec<(i64, i64)> {
        let mut spans: Vec<(i64, i64)> =
            self.video.iter().filter(|t| t.enabled).flat_map(|t| t.clips.iter().filter(|c| c.enabled && c.end > c.start).map(|c| (c.start, c.end))).collect();
        spans.sort_unstable();
        let mut out: Vec<(i64, i64)> = Vec::new();
        for (a, b) in spans {
            match out.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => out.push((a, b)),
            }
        }
        out
    }
}

// ------------------------------------------------------------------ joins

/// Gaps and overlaps between neighbours on `track` (sorted): `(earlier, later, gap ticks)`.
fn neighbours(track: &Track) -> Vec<(&Clip, &Clip, i64)> {
    let clips: Vec<&Clip> = track.clips.iter().filter(|c| c.enabled).collect();
    clips.windows(2).filter_map(|w| Some((*w.first()?, *w.get(1)?))).map(|(a, b)| (a, b, b.start.saturating_sub(a.end))).collect()
}

fn joins(seq: &Seq, opts: &Options, out: &mut Vec<Finding>) -> Check {
    let name = "joins";
    let mut notes = Vec::new();
    match seq.track(&opts.picture_track) {
        None => notes.push(format!("no track {}", opts.picture_track)),
        Some(v) => {
            for (a, b, gap) in neighbours(v) {
                let at = secs(a.end);
                let between = format!("between “{}” and “{}”", a.name, b.name);
                let f = if gap < 0 {
                    Some(Finding::new(
                        name,
                        Severity::Error,
                        secs(b.start),
                        format!("clips overlap on {} by {:.3} s, {between}", opts.picture_track, secs(gap.saturating_neg())),
                    ))
                } else if gap > 0 && seq.at_least_a_frame(gap) {
                    let frames = gap as f64 * seq.fps() / TPS_F;
                    Some(Finding::new(name, Severity::Error, at, format!("a gap of {frames:.1} frames on {} {between}: black there", opts.picture_track)))
                } else if gap > 0 && seq.boundary_in(a.end, b.start).is_some() {
                    Some(Finding::new(
                        name,
                        Severity::Error,
                        at,
                        format!("a sub-frame gap on {} {between} holds a frame start: that frame is black", opts.picture_track),
                    ))
                } else if gap > 0 {
                    Some(Finding::new(
                        name,
                        Severity::Warning,
                        at,
                        format!(
                            "clips on {} do not touch ({:.2} of a frame apart) {between}: no transition can join them",
                            opts.picture_track,
                            gap as f64 * seq.fps() / TPS_F
                        ),
                    ))
                } else {
                    None
                };
                out.extend(f.map(|f| f.until(secs(b.start.max(a.end))).clip(Some(b.id))));
            }
        }
    }
    // where a photo plays: silent on purpose (a still has no sound), not a join that dips
    let photos: Vec<(f64, f64)> = opts.entries.iter().flat_map(|e| &e.shots).filter(|s| s.photo).map(|s| (s.start, s.end)).collect();
    let in_photo = |a: f64, b: f64| photos.iter().any(|(s, e)| *s <= a + 1e-3 && *e >= b - 1e-3);
    match seq.track(&opts.sound_track) {
        None => notes.push(format!("no track {}", opts.sound_track)),
        Some(t) => {
            for (a, b, gap) in neighbours(t) {
                if gap > 0 && in_photo(secs(a.end), secs(b.start)) {
                    continue;
                }
                if gap < 0 {
                    out.push(
                        Finding::new(
                            name,
                            Severity::Error,
                            secs(b.start),
                            format!("sound clips overlap on {} by {:.3} s", opts.sound_track, secs(gap.saturating_neg())),
                        )
                        .clip(Some(b.id)),
                    );
                } else if gap > 0 && secs(gap) < SOUND_JOIN {
                    out.push(
                        Finding::new(
                            name,
                            Severity::Warning,
                            secs(a.end),
                            format!(
                                "sound clips on {} do not touch ({:.0} ms apart, “{}” → “{}”): no crossfade, the sound dips there",
                                opts.sound_track,
                                secs(gap) * 1000.0,
                                a.name,
                                b.name
                            ),
                        )
                        .until(secs(b.start))
                        .clip(Some(b.id)),
                    );
                }
            }
        }
    }
    // time no video track covers
    let covered = seq.picture();
    let mut at = 0i64;
    for (a, b) in &covered {
        if *a > at && (seq.at_least_a_frame(a - at) || seq.boundary_in(at, *a).is_some()) {
            out.push(Finding::new(name, Severity::Error, secs(at), format!("nothing on any video track for {:.2} s: black", secs(a - at))).until(secs(*a)));
        }
        at = at.max(*b);
    }
    if covered.is_empty() && seq.duration > 0 {
        out.push(Finding::new(name, Severity::Error, 0.0, "no picture at all".into()).until(secs(seq.duration)));
    } else if seq.duration > at && seq.at_least_a_frame(seq.duration - at) {
        let tail = secs(seq.duration - at);
        let severity = if tail > 2.0 { Severity::Warning } else { Severity::Info };
        out.push(Finding::new(name, severity, secs(at), format!("the picture ends {tail:.2} s before the sound (black to the end)")).until(secs(seq.duration)));
    }
    Check::ran(name, out, (!notes.is_empty()).then(|| notes.join("; ")))
}

// ------------------------------------------------------------------ frames

/// Mean luma (Rec. 709, 0..1) of a PNG and the share of its pixels darker than [`DARK`]; `None`
/// when it does not decode.
pub fn luma(png: &[u8]) -> Option<(f64, f64)> {
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png).ok()?.to_rgb8();
    let n = (img.width() as usize).saturating_mul(img.height() as usize);
    if n == 0 {
        return None;
    }
    let (mut sum, mut dark) = (0f64, 0usize);
    for p in img.pixels() {
        let [r, g, b] = p.0;
        let y = (0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b)) / 255.0;
        sum += y;
        if y < DARK {
            dark += 1;
        }
    }
    Some((sum / n as f64, dark as f64 / n as f64))
}

/// A frame whose picture is black or nearly (titles or a counter over it included).
pub fn is_black(png: &[u8]) -> Option<bool> {
    luma(png).map(|(_, dark)| dark >= BLACK_SHARE)
}

/// The frames the black check renders: every `sample_every` seconds and at each picture clip's
/// first frame and the first frame after it, up to the end of the picture, none inside a dip to
/// black; at most `max` (the regular samples thin out first).
fn sample_frames(seq: &Seq, opts: &Options) -> Vec<i64> {
    let end = seq.picture().last().map_or(0, |s| s.1);
    let last = seq.frame_ceil(end); // frames 0..last show picture
    if last <= 0 {
        return Vec::new();
    }
    let mut cuts = std::collections::BTreeSet::new();
    if let Some(v) = seq.track(&opts.picture_track) {
        for c in v.clips.iter().filter(|c| c.enabled) {
            cuts.insert(seq.frame_ceil(c.start));
            cuts.insert(seq.frame_ceil(c.end));
        }
    }
    cuts.retain(|k| *k < last);
    let cuts: Vec<i64> = cuts.into_iter().take(opts.max_frames).collect();
    let room = opts.max_frames.saturating_sub(cuts.len()).max(1) as i64;
    let every = if opts.sample_every.is_finite() && opts.sample_every > 0.0 { opts.sample_every } else { 2.0 };
    let step = ((every * seq.fps()).round() as i64).max(1).max((last + room - 1) / room);
    let mut frames: std::collections::BTreeSet<i64> = (0..last).step_by(usize::try_from(step).unwrap_or(1).max(1)).collect();
    frames.extend(cuts);
    let dips: Vec<(i64, i64)> = seq.video.iter().filter(|t| t.enabled).flat_map(|t| t.dips.iter().copied()).collect();
    frames.into_iter().filter(|k| !dips.iter().any(|(a, b)| (*a..=*b).contains(&seq.frame_tick(*k)))).collect()
}

/// The picture clip showing at `t` on the picture track.
fn clip_at<'a>(seq: &'a Seq, opts: &Options, t: i64) -> Option<&'a Clip> {
    seq.track(&opts.picture_track)?.clips.iter().find(|c| c.enabled && c.start <= t && t < c.end)
}

async fn black(ex: &dyn Executor, seq: &Seq, opts: &Options, r: &mut QaReport, progress: &mut dyn FnMut(&str)) -> Check {
    let name = "black";
    let frames = sample_frames(seq, opts);
    if frames.is_empty() {
        return Check::skipped(name, "no picture");
    }
    progress(&format!("rendering {} frames", frames.len()));
    let fd = 1.0 / seq.fps();
    let mut rendered = 0usize;
    // (first, last) black frames of the current run
    let mut run: Option<(i64, i64)> = None;
    let mut runs = Vec::new();
    for k in frames {
        let t = seq.frame_tick(k);
        let black = match ex.render(secs(t), opts.frame_side).await {
            Some(png) => {
                rendered += 1;
                is_black(&png)
            }
            None => None,
        };
        match (black, run) {
            (Some(true), None) => run = Some((k, k)),
            (Some(true), Some((a, _))) => run = Some((a, k)),
            (_, Some(x)) => {
                runs.push(x);
                run = None;
            }
            _ => {}
        }
    }
    runs.extend(run);
    r.frames_rendered += rendered;
    if rendered == 0 {
        return Check::skipped(name, "this backend renders no frames");
    }
    for (a, b) in runs {
        let (ta, tb) = (seq.frame_tick(a), seq.frame_tick(b));
        let what = match clip_at(seq, opts, ta) {
            Some(c) => format!("on “{}”", c.name),
            None => format!("no clip on {} there", opts.picture_track),
        };
        let f = if a == b {
            Finding::new(name, Severity::Error, secs(ta), format!("a black frame at {} (frame {a}), {what}", stamp(secs(ta))))
        } else {
            Finding::new(
                name,
                Severity::Error,
                secs(ta),
                format!("black from {} to {} (sampled frames {a}–{b}), {what}", stamp(secs(ta)), stamp(secs(tb) + fd)),
            )
            .until(secs(tb) + fd)
        };
        r.findings.push(f.clip(clip_at(seq, opts, ta).map(|c| c.id)));
    }
    Check::ran(name, &r.findings, Some(format!("{rendered} frames rendered")))
}

// ------------------------------------------------------------------ black sides

/// A side of the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

impl Side {
    const ALL: [Side; 4] = [Side::Left, Side::Right, Side::Top, Side::Bottom];

    pub fn name(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
            Side::Top => "top",
            Side::Bottom => "bottom",
        }
    }
}

/// How dark the band along one side of a frame is: the share of the darker of its two halves
/// that is black ([`SIDE_DARK`]), the share of the whole band, and its mean luma (0..1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Band {
    pub half: f64,
    pub dark: f64,
    pub luma: f64,
}

/// What the sides check sees of a frame ([`bands`]): the band along each side (left, right, top,
/// bottom), and the mean luma of the frame's centre and the share of it that is black.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bands {
    pub sides: [Band; 4],
    pub centre: f64,
    pub centre_dark: f64,
}

impl Bands {
    pub fn side(&self, s: Side) -> Band {
        self.sides.get(s as usize).copied().unwrap_or_default()
    }
}

/// The bands the sides check looks at, as shares of the frame `[x0, x1, y0, y1]`, in the order of
/// [`Side::ALL`]: the outer 8 % of the frame on each side — wholly inside the pillars beside a
/// portrait or 4:3 picture in a 16:9 frame, or the bars above and below a 16:9 picture in a
/// vertical one — the left and right over rows 15–70 % (clear of a counter, a name lower third,
/// captions), the top and bottom over columns 10–90 %; each in two halves along its length (a
/// whip or a push can black out half of one). And the centre: the middle fifth.
const SIDE_BANDS: [[[f64; 4]; 2]; 4] = [
    [[0.0, 0.08, 0.15, 0.425], [0.0, 0.08, 0.425, 0.70]],
    [[0.92, 1.0, 0.15, 0.425], [0.92, 1.0, 0.425, 0.70]],
    [[0.10, 0.50, 0.0, 0.08], [0.50, 0.90, 0.0, 0.08]],
    [[0.10, 0.50, 0.92, 1.0], [0.50, 0.90, 0.92, 1.0]],
];
const CENTRE: [f64; 4] = [0.4, 0.6, 0.4, 0.6];
/// A pixel at the side of the frame is black below this luma: nothing drawn there (the frame's
/// black, or a dip's colour) is black to within a few levels, while a dimmed, blurred copy of a
/// dark picture is not.
const SIDE_DARK: f64 = 0.015;
/// A side is black when this share of one half of its band is black.
const SIDE_BLACK: f64 = 0.85;
/// In a frame it is compared with, a side shows something when less than this share of its band
/// is black and its mean luma is at least [`SIDE_LIT_LUMA`] (a fill too dim to tell from black
/// is no reference).
const SIDE_LIT: f64 = 0.5;
const SIDE_LIT_LUMA: f64 = 3.0 * SIDE_DARK;
/// The picture still shows when the frame's centre is at least this share as bright as in the
/// frame it is compared with (and not dark: over a dip the sides may go dark with the picture),
/// and less than [`CENTRE_BLACK`] of it is black (a title or a year card over black is not a
/// picture).
const CENTRE_SHOWN: f64 = 0.5;
const CENTRE_BLACK: f64 = 0.25;

/// The bands of a rendered frame (PNG) for the sides check; None when it does not decode or is
/// too small to have them.
pub fn bands(png: &[u8]) -> Option<Bands> {
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    // (dark pixels, pixels, luma sum) of a region given in shares of the frame
    let region = |[x0, x1, y0, y1]: [f64; 4]| {
        let at = |a: f64, n: u32| ((a * f64::from(n)).round() as u32).min(n);
        let (xa, ya) = (at(x0, w).min(w - 1), at(y0, h).min(h - 1));
        let (xb, yb) = (at(x1, w).max(xa + 1), at(y1, h).max(ya + 1));
        let (mut dark, mut n, mut sum) = (0usize, 0usize, 0f64);
        for y in ya..yb {
            for x in xa..xb {
                let Some(p) = img.get_pixel_checked(x, y) else { continue };
                let [r, g, b] = p.0;
                let l = (0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b)) / 255.0;
                n += 1;
                sum += l;
                dark += usize::from(l < SIDE_DARK);
            }
        }
        (dark, n, sum)
    };
    let share = |(d, n, _): (usize, usize, f64)| d as f64 / n.max(1) as f64;
    let mut sides = [Band::default(); 4];
    for (band, [a, b]) in sides.iter_mut().zip(SIDE_BANDS) {
        let (a, b) = (region(a), region(b));
        let n = a.1.saturating_add(b.1).max(1) as f64;
        *band = Band { half: share(a).max(share(b)), dark: a.0.saturating_add(b.0) as f64 / n, luma: (a.2 + b.2) / n };
    }
    let c = region(CENTRE);
    Some(Bands { sides, centre: c.2 / c.1.max(1) as f64, centre_dark: share(c) })
}

/// The sides black in `frame` while its centre still shows a picture, of those that show
/// something in one of the frames it is compared with (`refs`: the frame just before or after
/// the transition, the middle of the picture shown whole) while the centre there is not much
/// brighter: black only around a transition, or where the pillars or the blur fill should be. A
/// side black in every frame it is compared with (letterboxing kept on purpose) is not one, nor
/// a side that darkens with the picture (a dip).
pub fn black_sides(frame: &Bands, refs: &[Bands]) -> Vec<Side> {
    Side::ALL
        .into_iter()
        .filter(|s| {
            frame.side(*s).half >= SIDE_BLACK
                && frame.centre_dark < CENTRE_BLACK
                && refs.iter().any(|r| {
                    let b = r.side(*s);
                    b.dark < SIDE_LIT && b.luma >= SIDE_LIT_LUMA && frame.centre >= SIDE_LIT_LUMA.max(CENTRE_SHOWN * r.centre)
                })
        })
        .collect()
}

/// A picture shown whole over a blurred copy — a showcase shot with `fit: "full"`, or a clip over
/// Auto Reframe's Blur Fill track: the picture's and its copy's clips, its name and span (ticks),
/// and the frame where its copy surely shows (the middle of where it is under the picture).
#[derive(Clone, Debug, PartialEq)]
struct Whole {
    clips: Vec<u64>,
    name: String,
    start: i64,
    end: i64,
    mid: i64,
}

/// The pictures shown whole: the entries' shots with a blurred copy, and the clips over a
/// [`filmcraft_engine::reframe::BLUR_FILL_TRACK`] (the track directly above it), where a copy is
/// under them.
fn shown_whole(seq: &Seq, opts: &Options) -> Vec<Whole> {
    let tick = |s: f64| (s.is_finite() && s.abs() < 1e7).then(|| (s * TPS_F).round() as i64);
    let middle = |a: i64, b: i64| i64::try_from((i128::from(a) + i128::from(b)) / 2).unwrap_or(a);
    let mut out = Vec::new();
    for e in &opts.entries {
        for s in e.shots.iter().filter(|s| s.background.is_some()) {
            let (Some(a), Some(b)) = (tick(s.start), tick(s.end)) else { continue };
            if b > a {
                out.push(Whole {
                    clips: [Some(s.clip), s.background].into_iter().flatten().collect(),
                    name: e.label.clone(),
                    start: a,
                    end: b,
                    mid: seq.frame_floor(middle(a, b)),
                });
            }
        }
    }
    for (i, fills) in seq.video.iter().enumerate().filter(|(_, t)| t.enabled && t.name == filmcraft_engine::reframe::BLUR_FILL_TRACK) {
        let Some(above) = seq.video.get(i.saturating_add(1)) else { continue };
        for f in fills.clips.iter().filter(|c| c.enabled && c.end > c.start) {
            for p in above.clips.iter().filter(|p| p.enabled && p.start < f.end && f.start < p.end) {
                let (a, b) = (p.start.max(f.start), p.end.min(f.end));
                out.push(Whole { clips: vec![p.id, f.id], name: p.name.clone(), start: p.start, end: p.end, mid: seq.frame_floor(middle(a, b)) });
            }
        }
    }
    out.truncate(MAX_SHOTS);
    out
}

/// A frame the sides check looks at: the frames it is compared with ([`black_sides`]), where it
/// is (for the finding), the clip showing there, and its group — the transition or picture it
/// is at: one finding per group.
#[derive(Clone, Debug, PartialEq)]
struct Probe {
    frame: i64,
    refs: Vec<i64>,
    group: usize,
    what: String,
    clip: Option<u64>,
}

/// The frames the sides check looks at: around every transition on an enabled video track —
/// its first two frames, the middle of each half, either side of the cut, its last two — each
/// compared with the frame just before the transition (before the cut) or just after it (after
/// the cut), and with the middle of the picture shown whole on that side of the cut if it is one
/// (over a dip, only its first and last quarters, where the picture is at least half shown);
/// and the first and last frame of every picture shown whole, compared with its middle. Within
/// [`Options::max_frames`] frames rendered.
fn side_probes(seq: &Seq, opts: &Options, whole: &[Whole]) -> Vec<Probe> {
    let last = seq.frame_ceil(seq.picture().last().map_or(0, |s| s.1)); // frames 0..last show picture
    let whole_of = |c: Option<u64>| c.and_then(|c| whole.iter().find(|w| w.clips.contains(&c)));
    let mut probes = Vec::new();
    let mut group = 0usize;
    for t in seq.video.iter().filter(|t| t.enabled) {
        let clip = |c: Option<u64>| c.and_then(|c| t.clips.iter().find(|x| x.id == c));
        let name = |c: Option<u64>| clip(c).map(|x| format!("“{}”", x.name));
        for x in &t.transitions {
            let (ks, ke) = (seq.frame_ceil(x.start), seq.frame_ceil(x.end).min(last));
            if ke <= ks {
                continue;
            }
            // the cut: where the clip it goes from ends, else where the one it goes to starts
            let cut = clip(x.from).map(|c| c.end).or_else(|| clip(x.to).map(|c| c.start)).unwrap_or(x.start / 2 + x.end / 2);
            let kc = seq.frame_ceil(cut).max(ks).min(ke);
            let what = match (name(x.from), name(x.to)) {
                (Some(a), Some(b)) => format!("in the {} from {a} into {b}", x.effect),
                (Some(a), None) => format!("in the {} out of {a}", x.effect),
                (None, Some(b)) => format!("in the {} into {b}", x.effect),
                (None, None) => format!("in a {}", x.effect),
            };
            let mut frames = vec![
                ks,
                ks.saturating_add(1),
                ks + (kc - ks) / 2,
                kc.saturating_sub(1),
                kc,
                kc.saturating_add(1),
                kc + (ke - kc) / 2,
                ke.saturating_sub(2),
                ke.saturating_sub(1),
            ];
            frames.retain(|k| (ks..ke).contains(k));
            // over a dip, only where the picture is at least half shown (its first and last
            // quarters): nearer its middle a dark part of the picture is black too
            if x.effect.starts_with("dip_to") {
                let d = i128::from(x.end) - i128::from(x.start);
                frames.retain(|k| {
                    let t = i128::from(seq.frame_tick(*k));
                    4 * (t - i128::from(x.start)) <= d || 4 * (i128::from(x.end) - t) <= d
                });
            }
            frames.sort_unstable();
            frames.dedup();
            for k in frames {
                let (near, shown, c) = if k < kc {
                    (ks.checked_sub(1).filter(|k| *k >= 0), whole_of(x.from), x.from)
                } else {
                    (Some(ke).filter(|k| *k < last), whole_of(x.to), x.to)
                };
                let refs = near.into_iter().chain(shown.map(|w| w.mid)).collect();
                probes.push(Probe { frame: k, refs, group, what: what.clone(), clip: c });
            }
            group = group.saturating_add(1);
        }
    }
    for w in whole {
        let (first, end) = (seq.frame_ceil(w.start), seq.frame_ceil(w.end).min(last));
        for (k, which) in [(first, "first"), (end.saturating_sub(1), "last")] {
            if (first..end).contains(&k) && k != w.mid {
                let what = format!("at the {which} frame of “{}”, shown whole", w.name);
                probes.push(Probe { frame: k, refs: vec![w.mid], group, what, clip: w.clips.first().copied() });
                group = group.saturating_add(1);
            }
        }
    }
    // within the budget, references included
    let mut frames = std::collections::BTreeSet::new();
    let budget = opts.max_frames.max(1);
    probes
        .into_iter()
        .take_while(|p| {
            frames.insert(p.frame);
            frames.extend(p.refs.iter().copied());
            frames.len() <= budget
        })
        .collect()
}

async fn sides(ex: &dyn Executor, seq: &Seq, opts: &Options, r: &mut QaReport, progress: &mut dyn FnMut(&str)) -> Check {
    let name = "sides";
    let whole = shown_whole(seq, opts);
    let probes = side_probes(seq, opts, &whole);
    if probes.is_empty() {
        return Check::skipped(name, "no transitions and no picture shown whole");
    }
    let frames: std::collections::BTreeSet<i64> = probes.iter().flat_map(|p| std::iter::once(p.frame).chain(p.refs.iter().copied())).collect();
    progress(&format!("rendering {} frames around transitions", frames.len()));
    let mut seen: std::collections::BTreeMap<i64, Bands> = std::collections::BTreeMap::new();
    let mut rendered = 0usize;
    for k in frames {
        if let Some(png) = ex.render(secs(seq.frame_tick(k)), opts.frame_side).await {
            rendered += 1;
            if let Some(b) = bands(&png) {
                seen.insert(k, b);
            }
        }
    }
    r.frames_rendered += rendered;
    if rendered == 0 {
        return Check::skipped(name, "this backend renders no frames");
    }
    // per group, the probes where each side is black
    let mut flagged: std::collections::BTreeMap<usize, std::collections::BTreeMap<Side, Vec<&Probe>>> = std::collections::BTreeMap::new();
    for p in &probes {
        let Some(b) = seen.get(&p.frame) else { continue };
        let refs: Vec<Bands> = p.refs.iter().filter_map(|k| seen.get(k).copied()).collect();
        for s in black_sides(b, &refs) {
            flagged.entry(p.group).or_default().entry(s).or_default().push(p);
        }
    }
    let fd = 1.0 / seq.fps();
    let groups = flagged.len();
    for by_side in flagged.into_values() {
        // sides black on the same frames are one finding ("the left and right sides")
        let mut runs: Vec<(Vec<Side>, Vec<&Probe>)> = Vec::new();
        for (side, ps) in by_side {
            match runs.iter_mut().find(|(_, q)| q.iter().map(|p| p.frame).eq(ps.iter().map(|p| p.frame))) {
                Some((sides, _)) => sides.push(side),
                None => runs.push((vec![side], ps)),
            }
        }
        for (sides, ps) in runs {
            let (Some(a), Some(b)) = (ps.first(), ps.last()) else { continue };
            let names: Vec<&str> = sides.iter().map(|s| s.name()).collect();
            let which = match names.as_slice() {
                [one] => format!("the {one} side is"),
                [rest @ .., last] => format!("the {} and {last} sides are", rest.join(", ")),
                [] => continue,
            };
            let (ta, tb) = (secs(seq.frame_tick(a.frame)), secs(seq.frame_tick(b.frame)));
            let f = if a.frame == b.frame {
                Finding::new(name, Severity::Error, ta, format!("{which} black at {} (frame {}) while the picture shows, {}", stamp(ta), a.frame, a.what))
            } else {
                Finding::new(
                    name,
                    Severity::Error,
                    ta,
                    format!(
                        "{which} black from {} to {} ({} sampled frames, {}–{}) while the picture shows, {}",
                        stamp(ta),
                        stamp(tb + fd),
                        ps.len(),
                        a.frame,
                        b.frame,
                        a.what
                    ),
                )
                .until(tb + fd)
            };
            r.findings.push(f.clip(a.clip));
        }
    }
    let note = format!("{rendered} frames rendered around transitions and {} picture(s) shown whole; {groups} place(s) with black sides", whole.len());
    Check::ran(name, &r.findings, Some(note))
}

// ------------------------------------------------------------------ the subject, the sheet

/// A shot to look at: its entry, picture clip and span (seconds).
#[derive(Clone, Debug, PartialEq)]
struct ShotRef {
    entry: usize,
    first: bool,
    /// A scene's shot: no subject to look for.
    scene: bool,
    clip: Option<u64>,
    start: f64,
    end: f64,
}

/// The entries' shots, or (without entries) each picture clip as an entry of its own.
fn shots(seq: &Seq, opts: &Options) -> Vec<ShotRef> {
    let mut out = Vec::new();
    if opts.entries.is_empty() {
        if let Some(v) = seq.track(&opts.picture_track) {
            for (i, c) in v.clips.iter().filter(|c| c.enabled).enumerate() {
                out.push(ShotRef { entry: i + 1, first: true, scene: false, clip: Some(c.id), start: secs(c.start), end: secs(c.end) });
            }
        }
    } else {
        for e in &opts.entries {
            if e.shots.is_empty() {
                out.push(ShotRef { entry: e.number, first: true, scene: e.scene, clip: None, start: e.start, end: e.end });
            }
            for (i, s) in e.shots.iter().enumerate() {
                out.push(ShotRef { entry: e.number, first: i == 0, scene: e.scene, clip: Some(s.clip), start: s.start, end: s.end });
            }
        }
    }
    out.retain(|s| s.start.is_finite() && s.end.is_finite() && s.end >= s.start);
    out.truncate(MAX_SHOTS);
    out
}

/// The entry `t` (seconds) falls in.
fn entry_at(seq: &Seq, opts: &Options, t: f64) -> Option<usize> {
    if opts.entries.is_empty() {
        let v = seq.track(&opts.picture_track)?;
        let tick = (t * TPS_F).round() as i64;
        return v.clips.iter().filter(|c| c.enabled).position(|c| c.start <= tick && tick < c.end).map(|i| i + 1);
    }
    opts.entries.iter().find(|e| e.start <= t && t < e.end).map(|e| e.number)
}

/// Look for the subject in a frame of every shot (with a model), and render one frame per entry
/// for the contact sheet: `(entry, PNG)`.
async fn subject(
    ex: &dyn Executor,
    seq: &Seq,
    opts: &Options,
    model: Option<&dyn ChatModel>,
    r: &mut QaReport,
    progress: &mut dyn FnMut(&str),
) -> (Check, Vec<(usize, Option<Vec<u8>>)>) {
    let name = "subject";
    let list = shots(seq, opts);
    let mut tiles: Vec<(usize, Option<Vec<u8>>)> = Vec::new();
    let (mut asked, mut failed, mut first_error) = (0usize, 0usize, None);
    if model.is_some() {
        progress(&format!("looking for the {} in {} shots", opts.subject, list.len()));
    }
    for s in &list {
        // the middle frame of the shot: clear of the transitions at its ends
        let mid = seq.frame_floor((((s.start + s.end) / 2.0) * TPS_F).round() as i64);
        let t = secs(seq.frame_tick(mid));
        let side = if model.is_some() { opts.ground_side } else { opts.frame_side };
        // (a scene shows no subject: only its tile for the contact sheet)
        if (model.is_none() || s.scene) && !s.first {
            continue;
        }
        let png = ex.render(t, side).await;
        if png.is_some() {
            r.frames_rendered += 1;
        }
        if let (Some(m), Some(png), false) = (model, &png, s.scene) {
            let dims = image::load_from_memory_with_format(png, image::ImageFormat::Png).ok().map(|i| (i.width(), i.height()));
            if let Some((w, h)) = dims {
                asked += 1;
                let (fw, fh) = crate::analyze::model_frame(w, h);
                match m.complete_json(crate::analyze::GROUND_SYSTEM, &crate::analyze::ground_prompt(&opts.subject), std::slice::from_ref(png)).await {
                    Ok(v) if !crate::analyze::frame_fact(t, &v, fw, fh).present => {
                        let mut f = Finding::new(
                            name,
                            Severity::Warning,
                            t,
                            format!("no {} found at {} (entry {}, the shot's middle frame; {})", opts.subject, stamp(t), s.entry, m.id()),
                        )
                        .clip(s.clip);
                        f.entry = Some(s.entry);
                        r.findings.push(f);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        failed += 1;
                        first_error.get_or_insert_with(|| e.to_string());
                    }
                }
            }
        }
        if s.first && tiles.len() < MAX_TILES {
            tiles.push((s.entry, png));
        }
    }
    let check = match model {
        None => Check::skipped(name, "model unavailable"),
        Some(_) if asked == 0 => Check::skipped(name, "no frames rendered"),
        Some(m) if failed == asked => Check::skipped(name, format!("{} did not answer: {}", m.id(), first_error.unwrap_or_default())),
        Some(m) => Check::ran(
            name,
            &r.findings,
            Some(format!("{} shots checked by {}{}", asked - failed, m.id(), if failed > 0 { format!(", {failed} unanswered") } else { String::new() })),
        ),
    };
    (check, tiles)
}

/// Segments a–g of the digits 0–9, a seven-segment display drawn in code (bit 0 = a, the top;
/// then b top right, c bottom right, d bottom, e bottom left, f top left, g the middle).
const SEGMENTS: [u8; 10] = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f];

fn fill(img: &mut image::RgbaImage, x: u32, y: u32, w: u32, h: u32, c: [u8; 4]) {
    let (x1, y1) = (x.saturating_add(w).min(img.width()), y.saturating_add(h).min(img.height()));
    for yy in y.min(y1)..y1 {
        for xx in x.min(x1)..x1 {
            img.put_pixel(xx, yy, image::Rgba(c));
        }
    }
}

/// `n` in seven-segment digits `h` pixels tall with its top left at `(x, y)`.
fn draw_number(img: &mut image::RgbaImage, n: usize, x: u32, y: u32, h: u32, c: [u8; 4]) {
    let (w, th) = ((h / 2).max(3), (h / 6).max(1));
    let half = h / 2;
    let mut x = x;
    for d in n.to_string().bytes() {
        let segs = SEGMENTS.get(usize::from(d.saturating_sub(b'0'))).copied().unwrap_or(0);
        let parts = [
            (x, y, w, th),
            (x + w - th, y, th, half),
            (x + w - th, y + half, th, h - half),
            (x, y + h - th, w, th),
            (x, y + half, th, h - half),
            (x, y, th, half),
            (x, (y + half).saturating_sub(th / 2), w, th),
        ];
        for (i, (px, py, pw, ph)) in parts.into_iter().enumerate() {
            if segs & (1 << i) != 0 {
                fill(img, px, py, pw, ph, c);
            }
        }
        x = x.saturating_add(w + th * 2);
    }
}

/// The contact sheet: the tiles' frames (`frame_side` wide) under bands with the entry numbers,
/// red for an entry with an error, amber with a warning; PNG.
fn contact_sheet(tiles: &[(usize, Option<Vec<u8>>)], findings: &[Finding], frame_side: u32, aspect: f64) -> Option<Vec<u8>> {
    if tiles.is_empty() {
        return None;
    }
    let aspect = if aspect.is_finite() { aspect.clamp(0.25, 4.0) } else { 16.0 / 9.0 };
    let (band, gap) = (30u32, 4u32);
    let n = tiles.len().min(MAX_TILES) as u32;
    let cols = n.min(6);
    let rows = n.div_ceil(cols.max(1));
    // tiles `frame_side` wide, smaller when the sheet would pass 40 megapixels
    let mut tw = frame_side.clamp(64, 1024);
    let height = |tw: u32| ((f64::from(tw) / aspect).round() as u32).clamp(16, tw.saturating_mul(2));
    while tw > 64 && u64::from(cols * (tw + gap) + gap) * u64::from(rows * (band + height(tw) + gap) + gap) > 40_000_000 {
        tw = tw * 3 / 4;
    }
    let th = height(tw);
    let mut sheet = image::RgbaImage::from_pixel(cols * (tw + gap) + gap, rows * (band + th + gap) + gap, image::Rgba([17, 17, 17, 255]));
    for (i, (entry, png)) in tiles.iter().take(MAX_TILES).enumerate() {
        let (col, row) = (i as u32 % cols, i as u32 / cols);
        let (x, y) = (gap + col * (tw + gap), gap + row * (band + th + gap));
        let worst = findings.iter().filter(|f| f.entry == Some(*entry)).map(|f| f.severity).max();
        let colour = match worst {
            Some(Severity::Error) => [160, 24, 24, 255],
            Some(Severity::Warning) => [138, 101, 0, 255],
            _ => [42, 42, 42, 255],
        };
        fill(&mut sheet, x, y, tw, band, colour);
        draw_number(&mut sheet, *entry, x + 8, y + 5, band - 10, [255, 255, 255, 255]);
        let frame = png.as_deref().and_then(|p| image::load_from_memory_with_format(p, image::ImageFormat::Png).ok());
        match frame {
            Some(f) => {
                let f = image::imageops::resize(&f.to_rgba8(), tw, th, image::imageops::FilterType::Triangle);
                image::imageops::overlay(&mut sheet, &f, i64::from(x), i64::from(y + band));
            }
            None => fill(&mut sheet, x, y + band, tw, th, [60, 0, 60, 255]),
        }
    }
    let mut out = std::io::Cursor::new(Vec::new());
    sheet.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

// ------------------------------------------------------------------ the mix

/// A mixdown as metered: the programme summary and the loudness every 100 ms.
#[derive(Clone, Debug, Default)]
struct Metered {
    seconds: f64,
    integrated: f64,
    range: f64,
    true_peak: f64,
    sample_peak: f64,
    first_over: Option<f64>,
    /// `(end of the 100 ms block, momentary, short-term)`, seconds and LUFS.
    points: Vec<(f64, f64, f64)>,
}

/// Meter a WAV file (read in place) with BS.1770 / EBU R128.
fn meter_wav(path: &Path) -> Result<Metered, String> {
    use filmcraft_media::MediaSource as _;
    let reader = filmcraft_media::reader::FileReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let src = filmcraft_media::wav::WavSource::open("mix.wav", Arc::new(reader)).map_err(|e| e.to_string())?;
    let a = src.info().audio().cloned().ok_or("the mixdown has no sound")?;
    let (rate, channels) = (a.sample_rate, a.channels as usize);
    if !(8_000..=384_000).contains(&rate) || !(1..=8).contains(&channels) {
        return Err(format!("unexpected mixdown format: {rate} Hz, {channels} channels"));
    }
    let frames = src.info().duration.to_units_floor(i64::from(rate)).clamp(0, (MAX_MIX_SECONDS * f64::from(rate)) as i64) as usize;
    let mut m = filmcraft_audio_dsp::loudness::LoudnessMeter::new(f64::from(rate), channels);
    let sub = ((f64::from(rate) * 0.1).round() as usize).max(1);
    let block = sub * 50;
    let mut out = Metered { seconds: frames as f64 / f64::from(rate), ..Metered::default() };
    let mut pos = 0usize;
    while pos < frames {
        let n = block.min(frames - pos);
        let buf = src.audio(pos as i64, n, rate).map_err(|e| e.to_string())?;
        let mut i = 0usize;
        while i < n {
            let j = (i + sub).min(n);
            let slices: Vec<&[f32]> = buf.channels.iter().map(|c| c.get(i..j).unwrap_or(&[])).collect();
            if out.first_over.is_none()
                && let Some(k) = (i..j).find(|k| buf.channels.iter().any(|c| c.get(*k).is_some_and(|x| x.abs() >= 1.0)))
            {
                out.first_over = Some((pos + k) as f64 / f64::from(rate));
            }
            m.process(&slices);
            if j - i == sub {
                out.points.push(((pos + j) as f64 / f64::from(rate), m.momentary(), m.short_term()));
            }
            i = j;
        }
        pos += n;
    }
    out.integrated = m.integrated();
    out.range = m.loudness_range();
    out.true_peak = m.true_peak_dbtp();
    out.sample_peak = 20.0 * m.sample_peak().log10();
    Ok(out)
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v.get(n / 2).copied() } else { Some((v.get(n / 2 - 1)? + v.get(n / 2)?) / 2.0) }
}

/// The loudness while the narration speaks and between its lines: the medians over 100 ms points
/// whose whole window (`w` seconds) lies inside a line, or between the first line's start and the
/// last one's end touching none. `value` picks the window's loudness from a point.
fn voice_and_between(points: &[(f64, f64, f64)], lines: &[(f64, f64)], w: f64, value: fn(&(f64, f64, f64)) -> f64) -> (Vec<f64>, Vec<f64>) {
    let (Some(first), Some(last)) = (lines.iter().map(|l| l.0).reduce(f64::min), lines.iter().map(|l| l.1).reduce(f64::max)) else {
        return (Vec::new(), Vec::new());
    };
    let (mut during, mut between) = (Vec::new(), Vec::new());
    for p in points {
        let (a, b) = (p.0 - w, p.0);
        let v = value(p);
        if !v.is_finite() || v < SILENT_LUFS {
            continue;
        }
        if lines.iter().any(|l| l.0 <= a && b <= l.1) {
            during.push(v);
        } else if a >= first && b <= last && !lines.iter().any(|l| a < l.1 && l.0 < b) {
            between.push(v);
        }
    }
    (during, between)
}

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

async fn mix(ex: &dyn Executor, seq: &Seq, opts: &Options, r: &mut QaReport) -> Check {
    let name = "mix";
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = opts.work_dir.clone().unwrap_or_else(std::env::temp_dir);
    let path = dir.join(format!("filmcraft-qa-{}-{nanos}.wav", std::process::id()));
    let p = json!({"path": path.to_string_lossy(), "format": "wav", "wait": true, "settings": {"audio": {"bits": 32, "float": true}}});
    let exported = ex.execute("file.exportMedia", p).await;
    // the export names the file it wrote (an extension may have been added)
    let written = exported.as_ref().ok().and_then(|v| v["path"].as_str()).map_or_else(|| path.clone(), PathBuf::from);
    let metered = match exported {
        Ok(_) => {
            let w = written.clone();
            tokio::task::spawn_blocking(move || meter_wav(&w)).await.unwrap_or_else(|e| Err(format!("internal error: {e}")))
        }
        Err(e) => Err(format!("the mix could not be exported: {e}")),
    };
    let _ = std::fs::remove_file(&written);
    let _ = std::fs::remove_file(&path);
    let m = match metered {
        Ok(m) => m,
        Err(e) => return Check::skipped(name, e),
    };
    let mut l = Loudness {
        seconds: m.seconds,
        integrated_lufs: finite(m.integrated),
        loudness_range_lu: m.range,
        true_peak_dbtp: finite(m.true_peak),
        sample_peak_dbfs: finite(m.sample_peak),
        first_over: m.first_over,
        ..Loudness::default()
    };
    let (lo, hi) = opts.loudness;
    match l.integrated_lufs {
        None => r.findings.push(Finding::new(name, Severity::Error, 0.0, "the mix is silent".into())),
        Some(i) if i < lo - LOUDNESS_TOLERANCE || i > hi + LOUDNESS_TOLERANCE => r.findings.push(Finding::new(
            name,
            Severity::Warning,
            0.0,
            format!("integrated loudness {i:.1} LUFS, outside the {lo}…{hi} LUFS target (YouTube plays at −14)"),
        )),
        Some(_) => {}
    }
    let clips = l.true_peak_dbtp.is_some_and(|tp| tp > 0.0) || m.first_over.is_some();
    if let Some(tp) = l.true_peak_dbtp {
        if clips {
            let at = m.first_over.unwrap_or(0.0);
            r.findings.push(Finding::new(
                name,
                Severity::Error,
                at,
                format!(
                    "the mix clips: true peak {tp:+.1} dBTP, sample peak {:+.1} dBFS{}",
                    l.sample_peak_dbfs.unwrap_or(0.0),
                    m.first_over.map(|t| format!(", first at full scale at {}", stamp(t))).unwrap_or_default()
                ),
            ));
        } else if tp > opts.max_true_peak + PEAK_TOLERANCE {
            r.findings.push(Finding::new(
                name,
                Severity::Warning,
                0.0,
                format!("true peak {tp:+.1} dBTP, above {} dBTP (may clip once encoded)", opts.max_true_peak),
            ));
        }
    }
    // the voice against what plays between its lines
    let lines: Vec<(f64, f64)> = seq
        .track(&opts.voice_track)
        .map(|t| t.clips.iter().filter(|c| c.enabled && c.end > c.start).map(|c| (secs(c.start), secs(c.end))).collect())
        .unwrap_or_default();
    let mut note = None;
    if lines.is_empty() {
        note = Some(format!("no narration on {}: the voice was not compared", opts.voice_track));
    } else {
        let (d, b) = voice_and_between(&m.points, &lines, 3.0, |p| p.2);
        let (d, b, window) = if d.len() >= 3 && b.len() >= 3 {
            (d, b, "shortTerm")
        } else {
            let (d, b) = voice_and_between(&m.points, &lines, 0.4, |p| p.1);
            (d, b, "momentary")
        };
        match (median(d), median(b)) {
            (Some(v), Some(g)) => {
                let margin = v - g;
                (l.voice_lufs, l.between_lufs, l.voice_margin_db, l.window) = (Some(v), Some(g), Some(margin), Some(window));
                if margin < opts.voice_margin {
                    let severity = if margin <= 0.0 { Severity::Error } else { Severity::Warning };
                    let at = lines.first().map_or(0.0, |x| x.0);
                    r.findings.push(Finding::new(
                        name,
                        severity,
                        at,
                        format!(
                            "the voice ({v:.1} LUFS) is {} what plays between its lines ({g:.1} LUFS); it should be at least {} dB louder",
                            if margin <= 0.0 { format!("{:.1} dB quieter than", -margin) } else { format!("only {margin:.1} dB louder than") },
                            opts.voice_margin
                        ),
                    ));
                }
            }
            _ => note = Some("too little narration, or too little between its lines, to compare".into()),
        }
    }
    // per stem, for small speakers
    if opts.analyze_mix {
        let exists = |label: &str| label.trim_start_matches('A').parse::<usize>().ok().and_then(|k| k.checked_sub(1)).is_some_and(|k| k < seq.audio.len());
        let mut p = json!({});
        if let Some(id) = seq.id {
            p["sequence"] = json!(id);
        }
        if exists(&opts.voice_track) {
            p["speech"] = json!([opts.voice_track]);
        }
        if exists(&opts.music_track) && opts.music_track != opts.voice_track {
            p["music"] = json!([opts.music_track]);
        }
        match ex.execute("audio.analyzeMix", p).await {
            Ok(a) => {
                let num = |v: &Value| v.as_f64().filter(|x| x.is_finite());
                l.speech_over_music_lu = num(&a["speech"]["toMusic"]["medianLu"]);
                l.speech_over_rest_lu = num(&a["speech"]["toRest"]["medianLu"]);
                l.limiter_db = a["limiter"]["present"].as_bool().filter(|p| *p).and_then(|_| num(&a["limiter"]["maxGainReductionDb"]));
                l.low_share = num(&a["mix"]["bands"]["sub"]);
                l.presence_share = num(&a["mix"]["bands"]["presence"]);
                r.findings.extend(mix_findings(&a, opts));
                r.mix_analysis = Some(a);
            }
            Err(e) => {
                let why = format!("the mix was not measured per stem: {e}");
                note = Some(note.map_or(why.clone(), |n| format!("{n}; {why}")));
            }
        }
    }
    r.loudness = Some(l);
    Check::ran(name, &r.findings, note)
}

/// What the per-stem analysis (`audio.analyzeMix`) says about the mix on small speakers, as
/// warnings: speech not far enough above the music while it plays, a limiter working hard on the
/// Mix, too much energy below 120 Hz, a narration clip whose source is clipped.
pub fn mix_findings(a: &Value, opts: &Options) -> Vec<Finding> {
    let name = "mix";
    let mut out = Vec::new();
    let num = |v: &Value| v.as_f64().filter(|x| x.is_finite());
    let from = num(&a["from"]).unwrap_or(0.0);
    if let Some(m) = num(&a["speech"]["toMusic"]["medianLu"])
        && m < opts.min_speech_over_music
    {
        let p10 = num(&a["speech"]["toMusic"]["p10Lu"]).map(|p| format!(", {p:.1} LU in its quietest tenth")).unwrap_or_default();
        out.push(Finding::new(
            name,
            Severity::Warning,
            from,
            format!(
                "while the narration speaks it is only {m:.1} LU above the music (median{p10}); at least {} keeps it understood at low volume: duck the music deeper or lower it",
                opts.min_speech_over_music
            ),
        ));
    }
    if a["limiter"]["present"].as_bool() == Some(true)
        && let Some(gr) = num(&a["limiter"]["maxGainReductionDb"])
        && gr > opts.max_limiter_db
    {
        let over = num(&a["limiter"]["secondsOver2Db"]).unwrap_or(0.0);
        out.push(Finding::new(
            name,
            Severity::Warning,
            from,
            format!(
                "the Mix's limiter takes up to {gr:.1} dB off the peaks ({over:.1} s over 2 dB; at most {} is clean): hold the loudest stem's peaks before the Mix (its level, a compressor or limiter on its track)",
                opts.max_limiter_db
            ),
        ));
    }
    if let Some(low) = num(&a["mix"]["bands"]["sub"])
        && low > opts.max_low_share
    {
        // the stem that brings the most energy below 120 Hz (its share times its power)
        let worst = a["stems"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| Some((s["track"].as_str()?, num(&s["bands"]["sub"])?, num(&s["integratedLufs"])?)))
            .filter(|(_, sub, _)| *sub > opts.max_low_share)
            .max_by(|x, y| (x.1 * 10f64.powf(x.2 / 10.0)).total_cmp(&(y.1 * 10f64.powf(y.2 / 10.0))))
            .map(|(t, sub, _)| format!(" (most from {t}: {:.0} % of its own)", sub * 100.0))
            .unwrap_or_default();
        out.push(Finding::new(
            name,
            Severity::Warning,
            from,
            format!(
                "{:.0} % of the mix's energy is below 120 Hz{worst}, more than {:.0} %: small speakers cannot play it, and it pushes everything else down; high-pass the stems (the speech mix does)",
                low * 100.0,
                opts.max_low_share * 100.0
            ),
        ));
    }
    for c in a["narrationClips"].as_array().into_iter().flatten().filter(|c| c["clipped"] == true).take(MAX_SHOTS) {
        let at = num(&c["seconds"]).unwrap_or(from);
        out.push(
            Finding::new(
                name,
                Severity::Warning,
                at,
                format!(
                    "narration \"{}\" is clipped: {} flat tops (its source was cut off at its peaks); voice it again or lower it before it reaches the mix",
                    c["name"].as_str().unwrap_or("?").chars().take(80).collect::<String>(),
                    c["flatRuns"].as_u64().unwrap_or(0)
                ),
            )
            .clip(c["clip"].as_u64()),
        );
    }
    out
}

// ------------------------------------------------------------------ privacy

/// The entry showing any of `clips` (as its picture or its background copy).
fn entry_of_clips<'a>(opts: &'a Options, clips: &[u64]) -> Option<&'a PlacedEntry> {
    opts.entries.iter().find(|e| e.shots.iter().any(|s| clips.contains(&s.clip) || s.background.is_some_and(|b| clips.contains(&b))))
}

/// What the privacy pass hid (see the module docs): masks over more than [`Options::max_mask`] of
/// a picture as warnings, the counts per entry into the report, and a frame of each clip with
/// masks where most of it is hidden, for the privacy sheet: `(entry, PNG)`.
async fn privacy(ex: &dyn Executor, seq: &Seq, opts: &Options, r: &mut QaReport, progress: &mut dyn FnMut(&str)) -> (Check, Vec<(usize, Option<Vec<u8>>)>) {
    let name = "privacy";
    if opts.privacy.is_empty() && !opts.faces {
        return (Check::skipped(name, "nothing was redacted"), Vec::new());
    }
    let max = if opts.max_mask.is_finite() { opts.max_mask.clamp(0.0, 1.0) } else { 0.2 };
    let masked = opts.privacy.iter().filter(|h| h.masks > 0).count().min(MAX_TILES);
    if masked > 0 {
        progress(&format!("rendering {masked} privacy frames"));
    }
    let (mut per, mut tiles): (Vec<EntryPrivacy>, Vec<(usize, Option<Vec<u8>>)>) = (Vec::new(), Vec::new());
    for h in opts.privacy.iter().take(MAX_CLIPS) {
        let ids: Vec<u64> = std::iter::once(h.clip).chain(h.also.iter().copied()).collect();
        let placed = entry_of_clips(opts, &ids);
        let number = placed.map(|e| e.number).or_else(|| h.largest_at.or(h.most_hidden_at).and_then(|t| entry_at(seq, opts, t)));
        if let Some(n) = number {
            let i = match per.iter().position(|p| p.entry == n) {
                Some(i) => i,
                None => {
                    per.push(EntryPrivacy { entry: n, label: placed.map(|e| e.label.clone()).unwrap_or_default(), ..EntryPrivacy::default() });
                    per.len() - 1
                }
            };
            if let Some(p) = per.get_mut(i) {
                p.faces = p.faces.saturating_add(h.faces);
                p.plates = p.plates.saturating_add(h.plates);
                p.named = p.named.saturating_add(h.named);
                p.masks = p.masks.saturating_add(h.masks);
                p.largest_mask = p.largest_mask.max(h.largest_mask);
            }
        }
        if h.largest_mask > max {
            let at = h.largest_at.unwrap_or(0.0);
            let mask = h.largest_mask_name.as_deref().map(|m| format!(" ({m})")).unwrap_or_default();
            let mut f = Finding::new(
                name,
                Severity::Warning,
                at,
                format!(
                    "a redaction mask{mask} covers {:.0} % of “{}” at {}: a blur that size hides the shot, not a face or a logo",
                    h.largest_mask * 100.0,
                    h.name,
                    stamp(at)
                ),
            )
            .clip(Some(h.clip));
            f.entry = number;
            r.findings.push(f);
        }
        if h.masks > 0 && tiles.len() < MAX_TILES {
            let png = match h.most_hidden_at {
                Some(t) => ex.render(secs(seq.frame_tick(seq.frame_floor((t * TPS_F).round() as i64))), opts.frame_side).await,
                None => None,
            };
            if png.is_some() {
                r.frames_rendered += 1;
            }
            tiles.push((number.unwrap_or(0), png));
        }
    }
    per.sort_by_key(|p| p.entry);
    tiles.sort_by_key(|t| t.0);
    let total = |f: fn(&EntryPrivacy) -> usize| per.iter().map(f).sum::<usize>();
    let hiding = per.iter().filter(|p| p.faces + p.plates + p.named > 0).count();
    let mut note = format!(
        "{} face(s), {} plate(s), {} named place(s) hidden in {hiding} of the {} entries checked",
        total(|p| p.faces),
        total(|p| p.plates),
        total(|p| p.named),
        per.len()
    );
    r.privacy = per;
    if opts.faces {
        let (faces_note, face_tiles) = unblurred(ex, seq, opts, r, progress).await;
        note.push_str("; ");
        note.push_str(&faces_note);
        // the faces shown first: they are what the sheet is for
        tiles.splice(0..0, face_tiles);
    }
    (Check::ran(name, &r.findings, Some(note)), tiles)
}

// ------------------------------------------------------------------ unblurred faces

/// Most frames looked at for faces, and in one `sequence.checkPrivacy` call.
const MAX_FACE_FRAMES: usize = 20_000;
const FACE_CHUNK: usize = 30;
/// Most unblurred faces drawn on the privacy sheet.
const MAX_FACE_TILES: usize = 48;
/// An unblurred face the detector scores at least this is an error (below: a warning).
pub const FACE_ERROR_SCORE: f64 = 0.75;

/// A face the finished picture shows that no redaction mask hides (`sequence.checkPrivacy`):
/// when, where, and what the check made of it. The boxes drive a platform's own blur tool
/// (YouTube Studio's custom blur) when a published video needs one.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnblurredFace {
    /// Seen from `start` to `end` (sequence seconds; looked at every [`Options::face_every`]).
    pub start: f64,
    pub end: f64,
    /// Where it is largest: when, and its box `[x, y, w, h]` in shares of the frame.
    pub seconds: f64,
    #[serde(rename = "box")]
    pub bbox: [f64; 4],
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The media time of that clip there, seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<f64>,
    /// Inside the robot's box (the shot's subject).
    pub on_subject: bool,
    /// What the vision model said: `person`, `notPerson`, `unsure` or `unchecked`.
    pub verdict: String,
    /// Left sharp as the rules allow: a robot's face (twice called not a person, on the subject
    /// or not one the detector is sure of).
    pub allowed: bool,
    /// The finding it makes (none: allowed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<Severity>,
    /// Its box where it was first and last seen (for joining it with the same face in the next
    /// stretch looked at).
    #[serde(skip)]
    pub first_box: [f64; 4],
    #[serde(skip)]
    pub last_box: [f64; 4],
}

/// The faces of one `sequence.checkPrivacy` answer; hostile values are dropped.
pub fn faces_of(v: &Value) -> Vec<UnblurredFace> {
    let fin = |x: &Value| x.as_f64().filter(|v| v.is_finite());
    v["unmasked"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .take(MAX_CLIPS)
        .filter_map(|f| {
            let b = &f["box"];
            let bbox = [fin(&b["x"])?, fin(&b["y"])?, fin(&b["w"])?, fin(&b["h"])?];
            let (start, end, seconds) = (fin(&f["start"])?, fin(&f["end"])?, fin(&f["seconds"])?);
            if !(bbox[2] > 0.0 && bbox[3] > 0.0 && start >= 0.0 && end >= start) {
                return None;
            }
            let verdict = f["verdict"].as_str().filter(|v| matches!(*v, "person" | "notPerson" | "unsure")).unwrap_or("unchecked");
            let at = |k: &str| {
                let b = &f[k];
                let read = || Some([fin(&b["x"])?, fin(&b["y"])?, fin(&b["w"])?, fin(&b["h"])?]);
                read().filter(|b| b[2] > 0.0 && b[3] > 0.0).unwrap_or(bbox)
            };
            Some(UnblurredFace {
                start,
                end,
                seconds,
                bbox,
                score: fin(&f["score"]).unwrap_or(0.0).clamp(0.0, 1.0),
                entry: None,
                clip: f["clip"].as_u64(),
                name: f["name"].as_str().map(|s| s.chars().take(200).collect()),
                media: fin(&f["media"]),
                on_subject: f["onSubject"].as_bool().unwrap_or(false),
                verdict: verdict.to_string(),
                allowed: f["allowed"].as_bool().unwrap_or(false),
                severity: None,
                first_box: at("first"),
                last_box: at("last"),
            })
        })
        .collect()
}

/// Two boxes `[x, y, w, h]` overlap or lie within a face of each other.
fn same_face(a: &[f64; 4], b: &[f64; 4]) -> bool {
    let (ax, ay, bx, by) = (a[0] + a[2] / 2.0, a[1] + a[3] / 2.0, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
    let reach = a[2].max(a[3]).max(b[2]).max(b[3]) * 1.5;
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt() <= reach
}

/// One face seen in neighbouring calls (split at their boundary) as one: same clip, the second
/// starting after the first ends and no more than `gap` seconds later, where the first was last
/// seen (two faces seen at once are two faces). The merged face is what the worse of the two
/// says.
pub fn merge_faces(mut faces: Vec<UnblurredFace>, gap: f64) -> Vec<UnblurredFace> {
    faces.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out: Vec<UnblurredFace> = Vec::new();
    let centre = |b: &[f64; 4]| (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
    for f in faces {
        let (fx, fy) = centre(&f.first_box);
        let near = |o: &UnblurredFace| {
            let (ox, oy) = centre(&o.last_box);
            (ox - fx).powi(2) + (oy - fy).powi(2)
        };
        let into = out
            .iter_mut()
            .filter(|o| o.clip == f.clip && f.start > o.end && f.start - o.end <= gap && same_face(&o.last_box, &f.first_box))
            .min_by(|a, b| near(a).total_cmp(&near(b)));
        if let Some(last) = into {
            let rank = |v: &str| match v {
                "person" => 3,
                "unsure" | "unchecked" => 2,
                "notPerson" => 1,
                _ => 0,
            };
            if rank(&f.verdict) > rank(&last.verdict) {
                last.verdict = f.verdict.clone();
            }
            if f.bbox[3] > last.bbox[3] {
                (last.seconds, last.bbox, last.media) = (f.seconds, f.bbox, f.media);
            }
            last.end = last.end.max(f.end);
            last.last_box = f.last_box;
            last.score = last.score.max(f.score);
            last.on_subject |= f.on_subject;
            last.allowed = face_allowed(last);
            continue;
        }
        out.push(f);
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}

/// What an unblurred face makes: none when it may stay sharp ([`UnblurredFace::allowed`]); an
/// error for a person (or, away from the robot, a face nobody could tell); a warning — to look
/// at, not a stop — for what the detector is not sure of, a face on the robot the two looks
/// disagree about, and one the model calls not a person away from the robot (a robot on a phone's
/// screen, a painting: the redaction blurs those, the face detector being sure of them).
pub fn face_severity(f: &UnblurredFace) -> Option<Severity> {
    if f.allowed {
        return None;
    }
    // on the v5 show the faces left sharp scored 0.75–0.93, the false finds (a shoe, a robot's
    // base, the floor) 0.60–0.69
    if f.score < FACE_ERROR_SCORE {
        return Some(Severity::Warning);
    }
    match f.verdict.as_str() {
        "person" => Some(Severity::Error),
        "notPerson" => Some(Severity::Warning),
        _ if f.on_subject => Some(Severity::Warning),
        _ => Some(Severity::Error),
    }
}

/// May `f` stay sharp: twice called not a person, and on the robot or not one the detector is
/// sure of (the rule `clip.autoRedact` hides by).
fn face_allowed(f: &UnblurredFace) -> bool {
    f.verdict == "notPerson" && (f.on_subject || f.score < f64::from(filmcraft_engine::redact::SURE_FACE))
}

/// The finding of an unblurred face (with its severity).
fn face_finding(f: &UnblurredFace, severity: Severity) -> Finding {
    let what = f.name.as_deref().map(|n| format!(" in “{n}”")).unwrap_or_default();
    let said = match f.verdict.as_str() {
        "person" => "the vision model sees a person",
        "notPerson" => "the detector is sure of it and it is not on the robot",
        "unsure" => "the vision model could not tell what it is",
        _ => "no vision model looked at it",
    };
    let [x, y, w, h] = f.bbox;
    let message = match severity {
        Severity::Error => format!(
            "an unblurred face from {} to {}{what}: {:.0} % of the frame tall at x {:.0} %, y {:.0} % (largest at {}; {said}); it needs a blur",
            stamp(f.start),
            stamp(f.end),
            h * 100.0,
            (x + w / 2.0) * 100.0,
            (y + h / 2.0) * 100.0,
            stamp(f.seconds)
        ),
        _ if f.score < FACE_ERROR_SCORE => format!(
            "something the face detector is not sure of ({:.2}) left unblurred from {} to {}{what} at x {:.0} %, y {:.0} % ({said}): check whether it is a face",
            f.score,
            stamp(f.start),
            stamp(f.end),
            (x + w / 2.0) * 100.0,
            (y + h / 2.0) * 100.0
        ),
        _ if f.verdict == "notPerson" => format!(
            "a face the vision model calls not a person left unblurred from {} to {}{what} at x {:.0} %, y {:.0} %, away from the robot (the detector is sure of it: {:.2}): check what it is",
            stamp(f.start),
            stamp(f.end),
            (x + w / 2.0) * 100.0,
            (y + h / 2.0) * 100.0,
            f.score
        ),
        _ => format!("a face on the robot left sharp from {} to {}{what} ({said}): check that it is the robot's", stamp(f.start), stamp(f.end)),
    };
    let mut out = Finding::new("privacy", severity, f.start, message).until(f.end).clip(f.clip);
    out.entry = f.entry;
    out
}

/// The times looked at for faces: every `face_every` seconds, and each video clip's first frame
/// and a quarter second in (a shot's start is where a face in it shows first).
fn face_times(seq: &Seq, opts: &Options) -> Vec<f64> {
    let end = secs(seq.duration);
    let every = if opts.face_every.is_finite() && opts.face_every >= 0.05 { opts.face_every } else { 0.5 };
    let mut t: Vec<f64> = (0..MAX_FACE_FRAMES).map(|k| k as f64 * every).take_while(|t| *t < end).collect();
    for c in seq.video.iter().filter(|tr| tr.enabled).flat_map(|tr| tr.clips.iter().filter(|c| c.enabled && c.end > c.start)) {
        let first = secs(seq.frame_tick(seq.frame_ceil(c.start)));
        t.push(first);
        if first + 0.25 < secs(c.end) {
            t.push(first + 0.25);
        }
    }
    let half = 0.5 / seq.fps().max(1.0);
    t.retain(|x| x.is_finite() && *x >= 0.0 && *x < end);
    t.sort_by(f64::total_cmp);
    t.dedup_by(|a, b| (*a - *b).abs() < half);
    t.truncate(MAX_FACE_FRAMES);
    t
}

/// The robot's boxes of every redacted clip and its copies, as `sequence.checkPrivacy`'s
/// `exclude`.
fn robot_boxes(opts: &Options) -> Vec<Value> {
    let mut out = Vec::new();
    for h in &opts.privacy {
        for c in std::iter::once(h.clip).chain(h.also.iter().copied()) {
            for b in &h.robots {
                if out.len() < 10_000 {
                    out.push(json!({"clip": c, "x": b[0], "y": b[1], "w": b[2], "h": b[3]}));
                }
            }
        }
    }
    out
}

/// A crop of the frame at `f.seconds` around the face, the face outlined in red, in the frame's
/// shape (for the privacy sheet).
async fn face_tile(ex: &dyn Executor, seq: &Seq, f: &UnblurredFace, side: u32) -> Option<Vec<u8>> {
    let t = secs(seq.frame_tick(seq.frame_floor((f.seconds * TPS_F).round() as i64)));
    let png = ex.render(t, side.max(640)).await?;
    let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png).ok()?.to_rgba8();
    let (w, h) = (f64::from(img.width()), f64::from(img.height()));
    let [bx, by, bw, bh] = f.bbox;
    // four faces tall, the frame's shape, inside the frame
    let ch = (bh * h * 4.0).clamp(h * 0.25, h);
    let cw = (ch * w / h).min(w);
    let (cx, cy) = ((bx + bw / 2.0) * w, (by + bh / 2.0) * h);
    let (x0, y0) = ((cx - cw / 2.0).clamp(0.0, (w - cw).max(0.0)), (cy - ch / 2.0).clamp(0.0, (h - ch).max(0.0)));
    let mut crop = image::imageops::crop_imm(&img, x0 as u32, y0 as u32, (cw as u32).max(1), (ch as u32).max(1)).to_image();
    let (rx0, ry0, rx1, ry1) = (bx * w - x0, by * h - y0, (bx + bw) * w - x0, (by + bh) * h - y0);
    let thick = (ch / 150.0).ceil().max(2.0) as i64;
    let (cw_i, ch_i) = (i64::from(crop.width()), i64::from(crop.height()));
    for k in 0..thick {
        let (l, t, r, b) = (rx0 as i64 - k, ry0 as i64 - k, rx1 as i64 + k, ry1 as i64 + k);
        for x in l..=r {
            for y in [t, b] {
                if (0..cw_i).contains(&x) && (0..ch_i).contains(&y) {
                    crop.put_pixel(x as u32, y as u32, image::Rgba([255, 0, 0, 255]));
                }
            }
        }
        for y in t..=b {
            for x in [l, r] {
                if (0..cw_i).contains(&x) && (0..ch_i).contains(&y) {
                    crop.put_pixel(x as u32, y as u32, image::Rgba([255, 0, 0, 255]));
                }
            }
        }
    }
    let mut out = std::io::Cursor::new(Vec::new());
    crop.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

/// Faces the finished picture shows that no redaction mask hides ([`Options::faces`]): frames
/// every [`Options::face_every`] seconds and at each shot's start, through
/// `sequence.checkPrivacy` (the face detector on the rendered frame, the masks, the robot's
/// boxes and the vision model); each face shown that should not be is a finding — an error for a
/// person — with a crop of it on the privacy sheet. Returns the note and the sheet's tiles.
async fn unblurred(ex: &dyn Executor, seq: &Seq, opts: &Options, r: &mut QaReport, progress: &mut dyn FnMut(&str)) -> (String, Vec<(usize, Option<Vec<u8>>)>) {
    let times = face_times(seq, opts);
    let exclude = robot_boxes(opts);
    let mut faces: Vec<UnblurredFace> = Vec::new();
    let (mut frames, mut masked, mut asked) = (0u64, 0u64, None::<String>);
    let n = times.len();
    for (k, chunk) in times.chunks(FACE_CHUNK).enumerate() {
        if k % 10 == 0 {
            progress(&format!("QA: looking for unblurred faces ({} of {n} frames)", k * FACE_CHUNK));
        }
        let mut p = json!({"seconds": chunk, "maxSide": opts.face_side.clamp(160, 3840), "minFaceHeight": opts.face_min_height.clamp(0.0, 1.0)});
        if !exclude.is_empty() {
            p["exclude"] = Value::Array(exclude.clone());
        }
        match ex.execute("sequence.checkPrivacy", p).await {
            Ok(v) => {
                frames += v["frames"].as_u64().unwrap_or(0);
                masked += v["masked"].as_u64().unwrap_or(0);
                faces.extend(faces_of(&v));
                if let Some(state) = v["verify"]["state"].as_str().filter(|s| *s != "done") {
                    asked = Some(format!("{state}{}", v["verify"]["reason"].as_str().map(|r| format!(": {r}")).unwrap_or_default()));
                }
            }
            Err(e) => {
                let f = Finding::new("privacy", Severity::Warning, 0.0, format!("could not look for unblurred faces: {e}"));
                r.findings.push(f);
                return (format!("unblurred faces not looked for ({e})"), Vec::new());
            }
        }
    }
    let gap = if opts.face_every.is_finite() { (opts.face_every * 2.5).clamp(0.5, 5.0) } else { 1.25 };
    let mut faces = merge_faces(faces, gap);
    for f in &mut faces {
        f.entry = entry_at(seq, opts, f.start);
        f.allowed = face_allowed(f);
        f.severity = face_severity(f);
        if let Some(sev) = f.severity {
            r.findings.push(face_finding(f, sev));
        }
    }
    let shown: Vec<&UnblurredFace> = faces.iter().filter(|f| f.severity == Some(Severity::Error)).collect();
    let mut tiles = Vec::new();
    for f in faces.iter().filter(|f| f.severity.is_some()).take(MAX_FACE_TILES) {
        let png = face_tile(ex, seq, f, opts.face_side).await;
        if png.is_some() {
            r.frames_rendered += 1;
        }
        tiles.push((f.entry.unwrap_or(0), png));
    }
    let allowed = faces.iter().filter(|f| f.allowed).count();
    let note = format!(
        "{frames} frame(s) looked at for faces: {masked} sighting(s) under a mask, {} unblurred face(s) shown{}, {} on the robot to check, {allowed} left sharp as robots{}",
        shown.len(),
        shown.first().map(|f| format!(" (first at {})", stamp(f.start))).unwrap_or_default(),
        faces.iter().filter(|f| f.severity == Some(Severity::Warning)).count(),
        asked.map(|a| format!("; vision check {a}")).unwrap_or_default()
    );
    r.unblurred_faces = faces;
    (note, tiles)
}

// ------------------------------------------------------------------ safe zones

/// The findings of a `sequence.checkSafeZones` answer: a warning for each caption, title or lower
/// third that reaches into the platform's interface. Hostile answers give what can be read.
pub fn safe_zone_findings(v: &Value) -> Vec<Finding> {
    let label = v["label"].as_str().unwrap_or("the platform");
    let finite = |x: &f64| x.is_finite();
    let overlaps = v["overlaps"].as_array().map(Vec::as_slice).unwrap_or_default();
    overlaps
        .iter()
        .take(MAX_CLIPS)
        .filter_map(|o| {
            let seconds = o["seconds"].as_f64().filter(finite)?;
            let kind = match o["kind"].as_str() {
                Some("caption") => "caption",
                Some("graphic") => "title or lower third",
                _ => "overlay",
            };
            let zones = o["zones"].as_array().map(|z| z.iter().filter_map(Value::as_str).take(5).collect::<Vec<_>>().join(" and ")).unwrap_or_default();
            let share = o["share"].as_f64().filter(finite).unwrap_or(0.0).clamp(0.0, 1.0);
            let text: String = o["text"].as_str().unwrap_or_default().chars().take(60).collect();
            let message =
                format!("{kind} \"{text}\" reaches into {label}'s {zones} zone ({:.0} % of it): the platform's own interface covers it there", share * 100.0);
            let f = Finding::new("safe_zones", Severity::Warning, seconds, message).clip(o["clip"].as_u64());
            Some(match o["until"].as_f64().filter(finite) {
                Some(u) => f.until(u),
                None => f,
            })
        })
        .collect()
}

/// Captions, titles and lower thirds against the platform's safe zones ([`Options::platform`]; on
/// a vertical sequence without one, every platform at once).
async fn safe_zones(ex: &dyn Executor, seq: &Seq, opts: &Options, r: &mut QaReport) -> Check {
    let name = "safe_zones";
    let platform = match (&opts.platform, seq.portrait) {
        (Some(p), _) => p.clone(),
        (None, true) => "vertical".to_string(),
        (None, false) => return Check::skipped(name, "not a vertical sequence, and no platform asked for"),
    };
    let mut p = json!({"platform": platform});
    if let Some(id) = seq.id {
        p["sequence"] = json!(id);
    }
    match ex.execute("sequence.checkSafeZones", p).await {
        Ok(v) => {
            let note = format!("{} caption(s) and graphic(s) checked against {}", v["checked"].as_u64().unwrap_or(0), v["label"].as_str().unwrap_or(&platform));
            r.findings.extend(safe_zone_findings(&v));
            Check::ran(name, &r.findings, Some(note))
        }
        Err(e) => Check::skipped(name, e),
    }
}

// ------------------------------------------------------------------ the whole check

/// Check the active sequence (see the module docs). `model`: a grounding vision model for the
/// subject check (None: skipped). Never fails: what cannot be checked is recorded as skipped.
pub async fn check(ex: &dyn Executor, opts: &Options, model: Option<&dyn ChatModel>, progress: &mut dyn FnMut(&str)) -> QaReport {
    let mut r = QaReport::default();
    let seq = match ex.execute("sequence.inspect", json!({})).await {
        Ok(v) => parse(&v),
        Err(e) => {
            r.findings.push(Finding::new("sequence", Severity::Error, 0.0, format!("the sequence could not be read: {e}")));
            r.checks.push(Check::skipped("sequence", e));
            r.tally();
            return r;
        }
    };
    (r.sequence, r.seconds, r.fps) = (seq.id, secs(seq.duration), seq.fps());
    progress("QA: joins");
    let c = joins(&seq, opts, &mut r.findings);
    r.checks.push(c);
    progress("QA: black frames");
    let c = black(ex, &seq, opts, &mut r, progress).await;
    r.checks.push(c);
    progress("QA: black sides");
    let c = sides(ex, &seq, opts, &mut r, progress).await;
    r.checks.push(c);
    progress("QA: subject and contact sheet");
    let (c, tiles) = subject(ex, &seq, opts, model, &mut r, progress).await;
    r.checks.push(c);
    progress("QA: privacy");
    let (c, privacy_tiles) = privacy(ex, &seq, opts, &mut r, progress).await;
    r.checks.push(c);
    progress("QA: safe zones");
    let c = safe_zones(ex, &seq, opts, &mut r).await;
    r.checks.push(c);
    if opts.mix {
        progress("QA: the mix");
        let c = mix(ex, &seq, opts, &mut r).await;
        r.checks.push(c);
    } else {
        r.checks.push(Check::skipped("mix", "not asked"));
    }
    for f in &mut r.findings {
        if f.entry.is_none() {
            f.entry = entry_at(&seq, opts, f.seconds);
        }
    }
    r.contact_sheet_png = contact_sheet(&tiles, &r.findings, opts.frame_side, seq.aspect);
    let privacy_findings: Vec<Finding> = r.findings.iter().filter(|f| f.check == "privacy").cloned().collect();
    r.privacy_sheet_png = contact_sheet(&privacy_tiles, &privacy_findings, opts.frame_side, seq.aspect);
    r.tally();
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(v: Value) -> Seq {
        parse(
            &json!({"id": 1, "settings": {"frame_rate": {"num": 30, "den": 1}, "width": 1920, "height": 1080}, "duration": tick(10.0), "video": v, "audio": []}),
        )
    }

    fn tick(s: f64) -> i64 {
        (s * TPS_F).round() as i64
    }

    fn clip(id: u64, a: f64, b: f64) -> Value {
        json!({"clip": id, "name": format!("c{id}"), "start": tick(a), "end": tick(b)})
    }

    #[test]
    fn frame_math_is_exact() {
        let s = seq(json!([]));
        let f = TPS as i64 / 30;
        assert!(s.at_least_a_frame(f) && !s.at_least_a_frame(f - 1));
        assert_eq!(s.frame_ceil(0), 0);
        assert_eq!(s.frame_ceil(1), 1);
        assert_eq!(s.frame_ceil(f), 1);
        assert_eq!(s.frame_floor(f - 1), 0);
        assert_eq!(s.frame_tick(3), 3 * f);
        // a sub-frame gap holds a frame start only when one falls inside it
        assert_eq!(s.boundary_in(f - 10, f + 10), Some(1));
        assert_eq!(s.boundary_in(f + 10, 2 * f), None, "the next frame starts where the gap ends");
        assert_eq!(s.boundary_in(f, f), None);
        // NTSC
        let mut n = s.clone();
        (n.num, n.den) = (30_000, 1001);
        assert_eq!(n.frame_tick(30_000), 1001 * TPS as i64);
        assert_eq!(n.frame_ceil(n.frame_tick(7)), 7);
    }

    #[test]
    fn hostile_sequences_parse_to_something_safe() {
        for v in
            [json!(null), json!({"settings": {"frame_rate": {"num": 0, "den": -5}}}), json!({"video": [{"items": [{"clip": 1, "start": 9, "end": 2}, "x"]}]})]
        {
            let s = parse(&v);
            assert!(s.num > 0 && s.den > 0);
            assert!(s.video.iter().all(|t| t.clips.is_empty()));
        }
        assert_eq!(seq(json!([])).track("V9"), None);
        assert_eq!(seq(json!([])).track(""), None);
        assert_eq!(seq(json!([])).track("Vx"), None);
    }

    #[test]
    fn joins_find_gaps_overlaps_and_holes() {
        let f = 1.0 / 30.0;
        let s = parse(&json!({
            "settings": {"frame_rate": {"num": 30, "den": 1}},
            "duration": tick(12.0),
            "video": [
                {"items": [clip(1, 0.0, 1.0)]},
                // a slowed shot ending between frames, the next on the following frame: no frame
                // starts in between; then a gap that holds the start of frame 180
                {"items": [clip(2, 0.0, 2.0), clip(3, 2.5, 4.0 + 0.2 * f), clip(4, 4.0 + f, 6.0), clip(5, 6.0 + 0.9 * f, 8.0)]}
            ],
            "audio": [{"items": [clip(10, 0.0, 2.0), clip(11, 2.1, 4.0), clip(12, 4.0, 6.0), clip(13, 9.0, 10.0)]}]
        }));
        let mut out = Vec::new();
        joins(&s, &Options::default(), &mut out);
        let at = |sev: Severity, t: f64| out.iter().any(|x| x.severity == sev && (x.seconds - t).abs() < 1e-6);
        assert!(at(Severity::Error, 2.0), "a half-second gap: {out:#?}");
        assert!(at(Severity::Warning, 4.0 + 0.2 * f), "a sub-frame gap without a frame start: a warning");
        assert!(!at(Severity::Error, 4.0 + 0.2 * f));
        assert!(at(Severity::Error, 6.0), "a sub-frame gap holding a frame start: black");
        assert!(out.iter().any(|x| x.message.contains("nothing on any video track")), "the hole at 2 s on every track");
        assert!(out.iter().any(|x| x.check == "joins" && x.message.contains("sound clips") && (x.seconds - 2.0).abs() < 1e-3));
        assert!(!out.iter().any(|x| x.message.contains("sound clips") && (x.seconds - 6.0).abs() < 1e-3), "a long silence is not a join");
        assert!(out.iter().any(|x| x.severity == Severity::Warning && x.message.contains("ends 4.00 s before the sound")));
    }

    #[test]
    fn samples_skip_dips_and_include_cuts() {
        let mut s = seq(json!([{"items": []}, {"items": [clip(1, 0.0, 4.0), clip(2, 4.0, 8.0)]}]));
        let frames = sample_frames(&s, &Options::default());
        assert!(frames.contains(&0) && frames.contains(&60) && frames.contains(&120), "{frames:?}");
        assert!(!frames.contains(&240), "the picture ends at frame 240");
        s.video[1].dips.push((tick(1.5), tick(2.5)));
        let frames = sample_frames(&s, &Options::default());
        assert!(!frames.contains(&60), "inside the dip");
        let few = sample_frames(&s, &Options { max_frames: 5, ..Options::default() });
        assert!(few.len() <= 6, "{few:?}");
    }

    fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn black_frames_and_not_black_ones() {
        assert_eq!(is_black(&png(64, 36, |_, _| [0, 0, 0])), Some(true));
        assert_eq!(is_black(&png(64, 36, |_, _| [4, 5, 6])), Some(true), "near-black");
        assert_eq!(is_black(&png(64, 36, |x, _| if x < 4 { [255, 255, 255] } else { [0, 0, 0] })), Some(true), "a counter over black is still black");
        assert_eq!(is_black(&png(64, 36, |x, _| if x < 8 { [255, 255, 255] } else { [0, 0, 0] })), Some(false), "an eighth of the frame lit");
        assert_eq!(is_black(&png(64, 36, |_, _| [40, 40, 40])), Some(false), "a dim shot");
        assert_eq!(is_black(&png(64, 36, |x, y| if (x + y) % 3 == 0 { [90, 90, 90] } else { [5, 5, 5] })), Some(false), "a dark, textured shot");
        assert_eq!(is_black(b"not a png"), None);
    }

    #[test]
    fn black_sides_are_told_from_letterboxing() {
        // a portrait picture in a 16:9 frame: its middle third lit, the pillars lit or black
        let wide = |pillars: u8, picture: u8| bands(&png(160, 90, |x, _| if (53..107).contains(&x) { [picture; 3] } else { [pillars; 3] })).unwrap();
        let (filled, black) = (wide(90, 200), wide(0, 200));
        assert!(filled.side(Side::Left).dark < 0.01 && black.side(Side::Left).half > 0.99 && black.side(Side::Top).half < 0.85);
        // black pillars while the picture shows, where the frame compared with has them lit
        assert_eq!(black_sides(&black, &[filled]), [Side::Left, Side::Right]);
        // pillarboxed throughout (black there too), or nothing to compare with: nothing
        assert!(black_sides(&black, &[black]).is_empty());
        assert!(black_sides(&black, &[]).is_empty());
        assert_eq!(black_sides(&black, &[black, filled]), [Side::Left, Side::Right], "one frame that shows them is enough");
        // the picture dark too (the middle of a dip), or the pillars dimmed with it: nothing
        assert!(black_sides(&wide(0, 15), &[filled]).is_empty());
        assert!(black_sides(&wide(0, 60), &[filled]).is_empty(), "the picture at under half its brightness");
        assert!(black_sides(&wide(40, 110), &[filled]).is_empty());
        // a title over black (a year card in a dip) is no picture
        let title = bands(&png(160, 90, |x, y| if (60..100).contains(&x) && (38..52).contains(&y) && x % 3 != 0 { [255; 3] } else { [0; 3] })).unwrap();
        assert!(title.centre > 0.3 && black_sides(&title, &[filled]).is_empty(), "{title:?}");
        // a fill too dim to tell from black is no reference; a dim one is
        assert!(black_sides(&black, &[wide(3, 200)]).is_empty());
        assert_eq!(black_sides(&black, &[wide(18, 200)]), [Side::Left, Side::Right]);
        // one side black (a whip half over), or half of one (a push across it)
        let left = bands(&png(160, 90, |x, _| {
            if x < 53 {
                [0; 3]
            } else if x < 107 {
                [200; 3]
            } else {
                [90; 3]
            }
        }))
        .unwrap();
        assert_eq!(black_sides(&left, &[filled]), [Side::Left]);
        let half = bands(&png(160, 90, |x, y| {
            if x >= 107 && y < 40 {
                [0; 3]
            } else if (53..107).contains(&x) {
                [200; 3]
            } else {
                [90; 3]
            }
        }))
        .unwrap();
        assert_eq!(black_sides(&half, &[filled]), [Side::Right]);
        // a wide picture in a vertical frame: the bars above and below
        let tall = |bars: u8| bands(&png(90, 160, |_, y| if (53..107).contains(&y) { [200; 3] } else { [bars; 3] })).unwrap();
        assert_eq!(black_sides(&tall(0), &[tall(90)]), [Side::Top, Side::Bottom]);
        assert!(black_sides(&tall(0), &[tall(0)]).is_empty());
        // not a PNG, too small: no bands
        assert!(bands(b"nope").is_none() && bands(&png(8, 8, |_, _| [0; 3])).is_none());
        assert!(bands(&png(16, 16, |_, _| [0; 3])).is_some());
    }

    #[test]
    fn the_sides_are_looked_at_around_transitions_and_pictures_shown_whole() {
        let x = |effect: &str, a: f64, b: f64, from: u64, to: u64| json!({"effect": effect, "start": tick(a), "duration": tick(b - a), "from": from, "to": to});
        let v = json!([
            {"name": "Video 1", "items": [clip(11, 2.0, 4.0)]},
            {"name": "Video 2", "items": [clip(1, 0.0, 2.0), clip(2, 2.0, 4.0), clip(3, 4.0, 6.0)],
             "transitions": [x("push", 1.9, 2.1, 1, 2), x("dip_to_black", 3.8, 4.2, 2, 3)]}
        ]);
        let s = seq(v.clone());
        let shot = crate::showcase::PlacedShot { clip: 2, item: 9, start: 2.0, end: 4.0, background: Some(11), ..Default::default() };
        let entry = PlacedEntry { number: 2, label: "Tall robot".into(), start: 2.0, end: 4.0, shots: vec![shot], ..Default::default() };
        let opts = Options { entries: vec![entry], ..Options::default() };
        let whole = shown_whole(&s, &opts);
        assert_eq!(whole, [Whole { clips: vec![2, 11], name: "Tall robot".into(), start: tick(2.0), end: tick(4.0), mid: 90 }]);
        let probes = side_probes(&s, &opts, &whole);
        let at = |k: i64| probes.iter().filter(|p| p.frame == k).map(|p| (p.group, p.refs.clone())).collect::<Vec<_>>();
        // the push into the picture shown whole, frames 57–62 (the cut at 60): before the cut
        // compared with the frame before it, after with the frame after it and the picture's middle
        let push: Vec<i64> = probes.iter().filter(|p| p.group == 0).map(|p| p.frame).collect();
        assert_eq!(push, [57, 58, 59, 60, 61, 62]);
        assert_eq!(at(58), [(0, vec![56])]);
        assert_eq!(at(61), [(0, vec![63, 90])]);
        // the dip out of it, frames 114–125 (the cut at 120): its first and last quarters
        let dip: Vec<i64> = probes.iter().filter(|p| p.group == 1).map(|p| p.frame).collect();
        assert_eq!(dip, [114, 115, 117, 123, 124, 125]);
        assert_eq!(at(115), [(1, vec![113, 90])]);
        assert_eq!(at(124), [(1, vec![126])]);
        assert!(probes[0].what.contains("push from “c1” into “c2”"), "{}", probes[0].what);
        // its first and last frames, compared with its middle
        assert_eq!(at(60), [(0, vec![63, 90]), (2, vec![90])]);
        assert_eq!(at(119), [(3, vec![90])]);
        // Auto Reframe's Blur Fill: the clips over it are shown whole where it is under them
        let mut filled = v.clone();
        filled[0]["name"] = json!("Blur Fill");
        let s = seq(filled);
        let whole = shown_whole(&s, &Options::default());
        assert_eq!(whole.iter().map(|w| (w.clips.clone(), w.mid)).collect::<Vec<_>>(), [(vec![2, 11], 90)]);
        assert_eq!(side_probes(&s, &Options::default(), &whole).len(), probes.len());
        // within the budget; hostile spans and a sequence without picture give nothing to look at
        assert!(side_probes(&seq(v), &Options { max_frames: 5, ..opts.clone() }, &[]).len() <= 5);
        let hostile = seq(json!([{"items": [clip(1, 0.0, 2.0)], "transitions": [
            {"effect": "push", "start": i64::MIN, "duration": i64::MAX, "from": 1},
            {"effect": "push", "start": i64::MAX, "duration": 5},
            {"effect": "x", "start": tick(1.0), "duration": -9}
        ]}]));
        assert!(side_probes(&hostile, &Options::default(), &[]).iter().all(|p| (0..60).contains(&p.frame)));
        assert!(side_probes(&seq(json!([])), &opts, &shown_whole(&seq(json!([])), &opts)).is_empty());
    }

    #[test]
    fn the_voice_is_compared_inside_lines_and_between_them() {
        // a line from 1 s to 5 s at -16, between at -30, then a line from 8 s to 12 s
        let points: Vec<(f64, f64, f64)> =
            (1..=140).map(|i| i as f64 / 10.0).map(|t| (t, if (1.0..5.0).contains(&t) || (8.0..12.0).contains(&t) { -16.0 } else { -30.0 }, -20.0)).collect();
        let (d, b) = voice_and_between(&points, &[(1.0, 5.0), (8.0, 12.0)], 0.4, |p| p.1);
        assert!(!d.is_empty() && !b.is_empty());
        assert_eq!(median(d), Some(-16.0));
        assert_eq!(median(b), Some(-30.0));
        assert_eq!(median(vec![1.0, f64::NAN, 3.0]), Some(2.0));
        assert_eq!(median(vec![]), None);
        assert!(voice_and_between(&points, &[], 3.0, |p| p.2).0.is_empty());
    }

    #[test]
    fn the_mix_on_small_speakers_is_judged_from_its_stems() {
        let opts = Options::default();
        // the first finished show, as measured: fine on speech, but bass-heavy and limited hard
        let before = json!({
            "from": 0.0,
            "mix": {"bands": {"sub": 0.378, "presence": 0.015}},
            "limiter": {"present": true, "maxGainReductionDb": 3.7, "secondsOver2Db": 0.85},
            "speech": {"toMusic": {"medianLu": 25.9, "p10Lu": 20.2}},
            "stems": [
                {"track": "A2", "bands": {"sub": 0.368}, "integratedLufs": -13.2},
                {"track": "A5", "bands": {"sub": 0.974}, "integratedLufs": -22.8},
            ],
            "narrationClips": [{"clip": 7, "name": "Narration 1.wav", "seconds": 2.2, "clipped": false, "flatRuns": 0, "overs": 296}],
        });
        let f = mix_findings(&before, &opts);
        assert_eq!(f.len(), 2, "{f:?}");
        assert!(f.iter().all(|x| x.severity == Severity::Warning && x.check == "mix"));
        assert!(f[0].message.contains("3.7 dB"), "{}", f[0].message);
        assert!(f[1].message.contains("38 %") && f[1].message.contains("A2"), "{}", f[1].message);
        // the speech mix: nothing to say
        let after = json!({
            "mix": {"bands": {"sub": 0.151}},
            "limiter": {"present": true, "maxGainReductionDb": 0.95},
            "speech": {"toMusic": {"medianLu": 27.4}},
            "narrationClips": [],
        });
        assert!(mix_findings(&after, &opts).is_empty());
        // speech buried, a clipped narration clip
        let bad = json!({
            "speech": {"toMusic": {"medianLu": 7.0, "p10Lu": 3.0}},
            "limiter": {"present": false, "maxGainReductionDb": 9.0},
            "narrationClips": [{"clip": 9, "name": "Narration 3.wav", "seconds": 12.5, "clipped": true, "flatRuns": 40}],
        });
        let f = mix_findings(&bad, &opts);
        assert_eq!(f.len(), 2, "no limiter on the Mix: its reduction is not judged: {f:?}");
        assert!(f[0].message.contains("7.0 LU") && f[0].message.contains("3.0 LU"));
        assert_eq!((f[1].clip, f[1].seconds), (Some(9), 12.5));
        // hostile or empty analyses: nothing, no panic
        for v in [json!(null), json!({}), json!({"mix": {"bands": {"sub": "lots"}}, "narrationClips": 3, "speech": [1]})] {
            assert!(mix_findings(&v, &opts).is_empty(), "{v}");
        }
    }

    #[test]
    fn the_contact_sheet_numbers_and_marks_entries() {
        let frame = png(160, 90, |_, _| [0, 120, 0]);
        let tiles: Vec<(usize, Option<Vec<u8>>)> = (1..=8).map(|i| (i, if i == 3 { None } else { Some(frame.clone()) })).collect();
        let mut bad = Finding::new("black", Severity::Error, 1.0, "x".into());
        bad.entry = Some(2);
        let sheet = contact_sheet(&tiles, &[bad], 160, 16.0 / 9.0).unwrap();
        let img = image::load_from_memory(&sheet).unwrap().to_rgba8();
        assert_eq!(img.width(), 6 * 164 + 4);
        assert_eq!(img.height(), 2 * (30 + 90 + 4) + 4);
        // entry 2's band is red, entry 1's grey, and digits are drawn in white
        assert_eq!(img.get_pixel(164 + 4 + 100, 4 + 2).0, [160, 24, 24, 255]);
        assert_eq!(img.get_pixel(4 + 100, 4 + 2).0, [42, 42, 42, 255]);
        assert!((4..34).any(|y| (12..30).any(|x| img.get_pixel(x, y).0 == [255, 255, 255, 255])));
        assert!(contact_sheet(&[], &[], 160, 1.0).is_none());
        // hostile sizes stay bounded
        assert!(contact_sheet(&tiles, &[], u32::MAX, f64::MIN_POSITIVE).is_some());
    }

    /// A 1080×1920 short with one caption (1 s to 3 s) whose bottom is `margin` up the frame.
    async fn short_with_caption(margin: f64) -> (crate::SessionExecutor, Seq) {
        let mut s = filmcraft_engine::Session::default();
        s.execute("file.newSequence", json!({"name": "Short", "width": 1080, "height": 1920, "fps": 30})).unwrap();
        s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
        s.execute("captions.add", json!({"track": "C1", "text": "watch this part", "seconds": 1.0, "durationSeconds": 2.0})).unwrap();
        s.execute("captions.setStyle", json!({"track": "C1", "size": 64, "background": false, "outline": 7, "anchor": "bottom", "margin": margin})).unwrap();
        let ex = crate::SessionExecutor::from(s);
        let seq = parse(&ex.execute("sequence.inspect", json!({})).await.unwrap());
        (ex, seq)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_caption_in_the_bottom_unsafe_zone_is_flagged_and_a_safe_one_passes() {
        // low in the frame, where the channel name and description sit
        let (ex, seq) = short_with_caption(0.1).await;
        assert!(seq.portrait);
        let mut r = QaReport::default();
        let c = safe_zones(&ex, &seq, &Options::default(), &mut r).await;
        assert_eq!(c.status, "findings", "{c:?}");
        let f = r.of("safe_zones", Severity::Warning);
        assert_eq!(f.len(), 1, "{:?}", r.findings);
        assert_eq!((f[0].seconds, f[0].until), (1.0, Some(3.0)));
        assert!(
            f[0].message.contains("bottom") && f[0].message.contains("All vertical platforms") && f[0].message.contains("watch this part"),
            "{}",
            f[0].message
        );
        assert!(f[0].clip.is_some());
        let mut r = QaReport::default();
        let yt = Options { platform: Some("youtube".into()), ..Options::default() };
        safe_zones(&ex, &seq, &yt, &mut r).await;
        assert!(r.findings[0].message.contains("YouTube Shorts"), "{:?}", r.findings);
        // above the bottom zone and narrow enough: nothing to say
        let (ex, seq) = short_with_caption(0.38).await;
        for p in ["youtube", "reels", "tiktok", "vertical"] {
            let mut r = QaReport::default();
            let c = safe_zones(&ex, &seq, &Options { platform: Some(p.into()), ..Options::default() }, &mut r).await;
            assert_eq!((c.status, r.findings.len()), ("ok", 0), "{p}: {:?} {c:?}", r.findings);
        }
        // a horizontal sequence without a platform is not checked; a platform that is not one is
        // a check that could not run, not a failure
        let flat = Seq { portrait: false, ..seq.clone() };
        assert_eq!(safe_zones(&ex, &flat, &Options::default(), &mut QaReport::default()).await.status, "skipped");
        let bad = Options { platform: Some("snapchat".into()), ..Options::default() };
        assert_eq!(safe_zones(&ex, &seq, &bad, &mut QaReport::default()).await.status, "skipped");
    }

    #[test]
    fn unblurred_faces_are_read_merged_and_judged() {
        let face = |start: f64, end: f64, x: f64, verdict: &str, allowed: bool| {
            json!({"start": start, "end": end, "seconds": start, "box": {"x": x, "y": 0.1, "w": 0.07, "h": 0.16}, "score": 0.92,
                   "clip": 10347, "name": "VID_20221014_144910.mp4", "onSubject": true, "verdict": verdict, "allowed": allowed})
        };
        let v = json!({"unmasked": [
            face(427.8, 428.9, 0.3, "notPerson", true),
            // the same face in the next call: merged, the worse answer kept
            face(429.2, 434.0, 0.31, "person", false),
            face(430.0, 431.0, 0.9, "unsure", false),
            // a face beside it at the same time: another face (the two women next to Ameca)
            face(428.0, 433.0, 0.37, "person", false),
            {"start": "x", "box": {}},
            {"start": 1.0, "end": 0.5, "seconds": 1.0, "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}},
            {"start": 1.0, "end": 2.0, "seconds": 1.0, "box": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.1}}
        ]});
        let faces = faces_of(&v);
        assert_eq!(faces.len(), 4, "hostile entries dropped: {faces:?}");
        let merged = merge_faces(faces, 1.25);
        assert_eq!(merged.len(), 3, "{merged:?}");
        let a = &merged[0];
        assert_eq!((a.start, a.end, a.verdict.as_str(), a.allowed), (427.8, 434.0, "person", false));
        assert_eq!(face_severity(a), Some(Severity::Error));
        assert_eq!((merged[1].start, merged[1].bbox[0]), (428.0, 0.37));
        // on the robot, nobody could tell: a warning; a robot's face: nothing
        assert_eq!(face_severity(&merged[2]), Some(Severity::Warning));
        assert_eq!(face_severity(&UnblurredFace { allowed: true, verdict: "notPerson".into(), ..UnblurredFace::default() }), None);
        // away from the robot, unchecked: an error; what the detector is not sure of: a warning
        let unchecked = UnblurredFace { verdict: "unchecked".into(), score: 0.9, ..UnblurredFace::default() };
        assert_eq!(face_severity(&unchecked), Some(Severity::Error));
        let unsure = UnblurredFace { verdict: "person".into(), score: 0.67, ..unchecked.clone() };
        assert_eq!(face_severity(&unsure), Some(Severity::Warning));
        assert!(face_finding(&unsure, Severity::Warning).message.contains("not sure of (0.67)"));
        // called not a person twice: on the robot it may stay; away from it, sure, a warning
        // (a robot on a phone's screen, a painting)
        let robot = UnblurredFace { verdict: "notPerson".into(), on_subject: true, ..unchecked.clone() };
        assert!(face_allowed(&robot));
        let screen = UnblurredFace { verdict: "notPerson".into(), ..unchecked };
        assert!(!face_allowed(&screen));
        assert_eq!(face_severity(&screen), Some(Severity::Warning));
        assert!(face_finding(&screen, Severity::Warning).message.contains("away from the robot"));
        // far apart in time: two faces
        let far = merge_faces(faces_of(&json!({"unmasked": [face(1.0, 2.0, 0.3, "person", false), face(9.0, 10.0, 0.3, "person", false)]})), 1.25);
        assert_eq!(far.len(), 2);
        assert!(faces_of(&json!("nothing")).is_empty());
        // a report with one blocks publishing; warnings do not
        let mut r = QaReport::default();
        r.findings.push(face_finding(a, Severity::Error));
        r.findings.push(Finding::new("privacy", Severity::Warning, 1.0, "a big mask".into()));
        assert!(r.publish_blocker().is_some_and(|b| b.contains("1 unblurred face") && b.contains("7:07.8")), "{:?}", r.publish_blocker());
        r.findings.remove(0);
        assert!(r.publish_blocker().is_none());
    }

    #[test]
    fn hostile_safe_zone_answers_give_what_can_be_read() {
        let v = json!({"label": "TikTok", "overlaps": [
            {"kind": "graphic", "seconds": 2.0, "until": f64::MAX, "zones": ["top", 3], "share": 7.5, "text": "x".repeat(500), "clip": 4},
            {"kind": "caption", "seconds": "soon"},
            {"seconds": 1.0, "share": -1.0},
            "nothing",
        ]});
        let f = safe_zone_findings(&v);
        assert_eq!(f.len(), 2, "{f:?}");
        assert!(f[0].message.contains("title or lower third") && f[0].message.contains("100 %") && f[0].message.chars().count() < 300, "{}", f[0].message);
        assert_eq!(f[0].clip, Some(4));
        assert!(f[1].message.contains("0 %"));
        for v in [json!(null), json!({"overlaps": 3}), json!([1, 2])] {
            assert!(safe_zone_findings(&v).is_empty(), "{v}");
        }
    }

    #[test]
    fn the_report_summarises_and_writes_next_to_the_export() {
        let mut r = QaReport::default();
        r.findings.push(Finding::new("mix", Severity::Error, 3.0, "clips".into()));
        r.findings.push(Finding::new("joins", Severity::Warning, 1.0, "apart".into()));
        r.checks = vec![Check::ran("joins", &r.findings, None), Check::ran("mix", &r.findings, None), Check::skipped("subject", "model unavailable")];
        r.contact_sheet_png = Some(png(4, 4, |_, _| [1, 2, 3]));
        r.tally();
        assert_eq!((r.errors, r.warnings), (1, 1));
        assert_eq!(r.findings[0].seconds, 1.0, "in time order");
        let s = r.summary();
        assert!(s.starts_with("1 error(s), 1 warning(s)") && s.contains("subject skipped (model unavailable)"), "{s}");
        let dir = std::env::temp_dir().join(format!("filmcraft-qa-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (j, p) = write_next_to(&mut r, &dir.join("robots.mp4")).unwrap();
        assert!(j.ends_with("robots.qa.json") && p.as_ref().is_some_and(|p| p.ends_with("robots.qa.png")));
        let v: Value = serde_json::from_slice(&std::fs::read(&j).unwrap()).unwrap();
        assert_eq!(v["errors"], 1);
        assert_eq!(v["findings"][0]["severity"], "warning");
        assert!(v["contactSheet"].as_str().unwrap().ends_with("robots.qa.png"));
        assert!(v.get("contactSheetPng").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

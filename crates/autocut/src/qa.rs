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
//! | `joins` | on the picture track (V2): gaps of a frame or more, gaps that hold a frame boundary (that frame renders black) and overlaps (errors), clips that do not quite touch (warnings); on the shots' sound track (A1): neighbours that do not touch, so their fades meet silence instead of crossfading (warnings); time where no video track has a clip (errors; a picture that ends before the sound: info) |
//! | `black` | rendered frames whose picture is black or nearly (92 % of the frame darker than 5 % luma, so a counter or lower third over black still counts) — sampled every 2 s, at each picture clip's first frame and at the first frame after it — outside dips to black (errors) |
//! | `mix` | the mix exported as 32-bit float WAV and metered (BS.1770): integrated loudness outside −16…−14 LUFS (±0.5 LU) and true peak above −1 dBTP (warnings); true peak above 0 dBTP or samples at full scale: clipping (error); the median short-term loudness during narration clips (A2) vs between them: the voice less than 4 dB louder (warning), or not louder at all (error) |
//! | `subject` | shots where the grounding model (`AUTOCUT_ANALYSIS_MODEL`, the analysis' prompt) does not find the subject in a rendered frame (warnings); skipped when no model is reachable |
//! | `privacy` | from what the privacy pass hid ([`Options::privacy`]): a redaction mask covering more than [`Options::max_mask`] (a fifth) of a clip's picture at any keyframe (warnings: a blur that size hides the shot, as one on a close-up of a tablet did); the faces, plates and named places hidden in each entry (`privacy` in the report) |
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
    /// What the privacy pass hid ([`Built::privacy`]); empty: the privacy check is skipped.
    pub privacy: Vec<crate::privacy::Hidden>,
    /// The largest share of a clip's picture one redaction mask may cover.
    pub max_mask: f64,
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
            privacy: Vec::new(),
            max_mask: 0.2,
        }
    }
}

impl Options {
    /// The checks of a showcase built with `opts`.
    pub fn for_showcase(opts: &crate::showcase::Options, built: &Built) -> Self {
        Self { subject: opts.subject.clone(), entries: built.layout.clone(), privacy: built.privacy.clone(), ..Self::default() }
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
    /// `joins`, `black`, `mix`, `subject` or `sequence`.
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

    fn tally(&mut self) {
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

#[derive(Clone, Debug, Default, PartialEq)]
struct Track {
    enabled: bool,
    clips: Vec<Clip>,
    /// Dip-to-black transitions, ticks.
    dips: Vec<(i64, i64)>,
}

/// What QA needs of `sequence.inspect`, in ticks; hostile values are dropped or replaced.
#[derive(Clone, Debug, PartialEq)]
struct Seq {
    id: Option<u64>,
    num: i64,
    den: i64,
    aspect: f64,
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
        Track { enabled: t["enabled"].as_bool().unwrap_or(true), clips, dips }
    };
    let tracks = |k: &str| v[k].as_array().map(|a| a.iter().take(MAX_TRACKS).map(track).collect()).unwrap_or_default();
    Seq { id: v["id"].as_u64(), num, den, aspect, duration: v["duration"].as_i64().unwrap_or(0).max(0), video: tracks("video"), audio: tracks("audio") }
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
    match seq.track(&opts.sound_track) {
        None => notes.push(format!("no track {}", opts.sound_track)),
        Some(t) => {
            for (a, b, gap) in neighbours(t) {
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
    r.loudness = Some(l);
    Check::ran(name, &r.findings, note)
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
    if opts.privacy.is_empty() {
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
    let note =
        format!("{} face(s), {} plate(s), {} named place(s) hidden in {} entr(ies)", total(|p| p.faces), total(|p| p.plates), total(|p| p.named), per.len());
    r.privacy = per;
    (Check::ran(name, &r.findings, Some(note)), tiles)
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
    progress("QA: subject and contact sheet");
    let (c, tiles) = subject(ex, &seq, opts, model, &mut r, progress).await;
    r.checks.push(c);
    progress("QA: privacy");
    let (c, privacy_tiles) = privacy(ex, &seq, opts, &mut r, progress).await;
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

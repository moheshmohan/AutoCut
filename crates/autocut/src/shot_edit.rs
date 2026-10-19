//! Per-shot editing controls of a showcase plan ([`crate::showcase::Shot`]): what the user says
//! about one shot of an entry, all optional, all checked before anything is built.
//!
//! | shot field | what it does |
//! |---|---|
//! | `from`, `to` | the source window, seconds into the clip, instead of the automatic one ("the part where it sprays the mist"); one of them alone anchors the window there and the entry's timing picks its length |
//! | `speed` | play the shot this many times as fast (0.25–16; `4` fast-forwards a wide shot): it takes `(to - from) / speed` of the timeline, and its own sound is dropped (never sped-up audio under the narration) |
//! | `fit` | `"fill"`: the frame is filled (a portrait clip cropped around its subject, as today); `"full"`: the whole picture is shown — a portrait clip at full height in the middle, over a blurred, darkened, enlarged copy of itself filling the frame |
//! | `focus` | `[x, y]` (0..1 of the clip, x right, y down): the point a `fill` crop keeps in view, instead of the analysis' box (the crop then holds it with a slow tilt, no tracker) |
//! | `mute` | `true`: the shot carries no sound of its own (a robot talking after the narration ends sounds like a garbled voice-over) |
//! | `volume` | its own sound this many dB above (or below: `-20`) the level the mix gives the shots' sound, −60 to +12 |
//!
//! An entry's shots keep their order. A shot with both ends has a fixed length on the timeline;
//! the entry's other shots share what its words still need ([`crate::showcase`]'s windows), and
//! the narration always fits: when fixed shots alone are too short for the words, the footage
//! after the last one runs on, and the build says so.

use serde::{Deserialize, Serialize};

use crate::AutoCutError;
use crate::showcase::Shot;

/// Slowest and fastest a shot plays.
pub const MIN_SPEED: f64 = 0.25;
pub const MAX_SPEED: f64 = 16.0;
/// Shortest source window, seconds.
pub const MIN_WINDOW: f64 = 0.25;
/// Longest a shot plays on the timeline, seconds.
pub const MAX_PLACED: f64 = 60.0;
/// How far `to` may pass the clip's measured length (the analysis rounds it).
const SLACK: f64 = 0.05;
/// Quietest and loudest a shot's own sound may be set, dB from the mix's level.
pub const MIN_VOLUME: f64 = -60.0;
pub const MAX_VOLUME: f64 = 12.0;

/// How a shot fills the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    /// The frame filled: a clip of another shape cropped around its subject.
    Fill,
    /// The whole picture: a clip of another shape over a blurred, darkened copy of itself.
    Full,
}

impl Fit {
    /// `fill` or `full` (any case).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fill" => Some(Self::Fill),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Fill => "fill",
            Self::Full => "full",
        }
    }
}

/// A shot's controls, checked ([`ShotEdit::of`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShotEdit {
    pub from: Option<f64>,
    pub to: Option<f64>,
    /// 1 when not given.
    pub speed: f64,
    pub fit: Option<Fit>,
    pub focus: Option<[f64; 2]>,
    /// No sound of its own.
    pub mute: bool,
    /// Its own sound, dB from the mix's level for the shots.
    pub volume: Option<f64>,
}

impl Default for ShotEdit {
    fn default() -> Self {
        Self { from: None, to: None, speed: 1.0, fit: None, focus: None, mute: false, volume: None }
    }
}

fn invalid(msg: String) -> AutoCutError {
    AutoCutError::Invalid(msg)
}

/// `x` as a message shows it.
fn num(x: f64) -> String {
    if x.is_finite() { format!("{x}") } else { "not a number".into() }
}

impl ShotEdit {
    /// The controls of `shot` (shot `k` of entry `entry`, both from 1 as messages say them),
    /// checked: times finite, from 0, `from` before `to`, inside the clip (when its length is
    /// known), at least [`MIN_WINDOW`] long and at most [`MAX_PLACED`] on the timeline; speed
    /// from [`MIN_SPEED`] to [`MAX_SPEED`]; `fit` `fill` or `full`; `focus` two numbers 0..1.
    pub fn of(shot: &Shot, entry: usize, k: usize) -> Result<Self, AutoCutError> {
        let at = format!("entry {entry}, shot {k}");
        let len = shot.clip_seconds;
        let known = len.is_finite() && len > 0.0;
        let time = |name: &str, v: Option<f64>| -> Result<Option<f64>, AutoCutError> {
            let Some(x) = v else { return Ok(None) };
            if !x.is_finite() || x < 0.0 {
                return Err(invalid(format!("{at}: `{name}` is seconds into the clip, from 0, not {}", num(x))));
            }
            if known && x > len + SLACK {
                return Err(invalid(format!("{at}: `{name}` {x:.2} s is past the end of the clip ({len:.2} s)")));
            }
            Ok(Some(x))
        };
        let from = time("from", shot.from)?;
        let to = time("to", shot.to)?;
        if let (Some(a), Some(b)) = (from, to)
            && b - a < MIN_WINDOW
        {
            return Err(invalid(if b <= a {
                format!("{at}: `to` ({b:.2} s) is not after `from` ({a:.2} s)")
            } else {
                format!("{at}: the window {a:.2}–{b:.2} s is shorter than {MIN_WINDOW} s")
            }));
        }
        if known && from.is_some_and(|a| a > len - MIN_WINDOW) {
            return Err(invalid(format!("{at}: `from` leaves less than {MIN_WINDOW} s of the clip ({len:.2} s)")));
        }
        if to.is_some_and(|b| b < MIN_WINDOW) {
            return Err(invalid(format!("{at}: `to` leaves less than {MIN_WINDOW} s before it")));
        }
        let speed = match shot.speed {
            None => 1.0,
            Some(s) if s.is_finite() && (MIN_SPEED..=MAX_SPEED).contains(&s) => s,
            Some(s) => return Err(invalid(format!("{at}: `speed` is from {MIN_SPEED} to {MAX_SPEED} (4: four times as fast), not {}", num(s)))),
        };
        if let (Some(a), Some(b)) = (from, to)
            && (b - a) / speed > MAX_PLACED
        {
            return Err(invalid(format!("{at}: the window plays {:.0} s on the timeline; at most {MAX_PLACED:.0} (cut it, or speed it up)", (b - a) / speed)));
        }
        let fit = match shot.fit.as_deref() {
            None => None,
            Some(f) => {
                Some(Fit::parse(f).ok_or_else(|| invalid(format!("{at}: `fit` is \"fill\" or \"full\", not \"{}\"", f.chars().take(40).collect::<String>())))?)
            }
        };
        let focus = match shot.focus {
            None => None,
            Some(p) if p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) => Some(p),
            Some(p) => return Err(invalid(format!("{at}: `focus` is [x, y], each 0..1 of the clip, not [{}, {}]", num(p[0]), num(p[1])))),
        };
        let volume = match shot.volume {
            None => None,
            Some(v) if v.is_finite() && (MIN_VOLUME..=MAX_VOLUME).contains(&v) => Some(v),
            Some(v) => return Err(invalid(format!("{at}: `volume` is dB from {MIN_VOLUME} to +{MAX_VOLUME} (-20: much quieter), not {}", num(v)))),
        };
        let mute = shot.mute.unwrap_or(false);
        Ok(Self { from, to, speed, fit, focus, mute, volume })
    }

    /// Both ends given: the shot's length on the timeline is fixed.
    pub fn fixed(&self) -> Option<(f64, f64)> {
        self.from.zip(self.to)
    }

    /// Anything that changes the entry's timing (a window or a speed).
    pub fn timed(&self) -> bool {
        self.from.is_some() || self.to.is_some() || self.speed != 1.0
    }

    /// The shot's own sound plays: not muted, not at another speed.
    pub fn sound(&self) -> bool {
        self.speed == 1.0 && !self.mute
    }

    /// The window of a shot anchored by one end (or none) whose share of the timeline is
    /// `share` seconds: `share × speed` of the clip from `from` (or ending at `to`), inside a clip
    /// `clip_seconds` long with a small margin; `(start, len)`.
    pub fn anchored(&self, share: f64, clip_seconds: f64) -> Option<(f64, f64)> {
        let want = (share.max(MIN_WINDOW) * self.speed).max(MIN_WINDOW);
        let end = if clip_seconds.is_finite() && clip_seconds > 0.0 { clip_seconds } else { f64::INFINITY };
        let (a, len) = match (self.from, self.to) {
            (Some(a), Some(b)) => (a, b - a),
            (Some(a), None) => (a, want.min(end - 0.2 - a)),
            (None, Some(b)) => {
                let a = (b - want).max(0.0);
                (a, b - a)
            }
            (None, None) => return None,
        };
        (a.is_finite() && len.is_finite() && len >= MIN_WINDOW * 0.5).then_some((a, len))
    }
}

/// Every shot's controls in `entries`, checked up front (`edits[i][k]` for shot `k` of entry
/// `i`): the first that makes no sense is an error saying which and why.
pub fn check(entries: &[crate::showcase::Entry]) -> Result<Vec<Vec<ShotEdit>>, AutoCutError> {
    entries.iter().map(|e| e.shots.iter().enumerate().map(|(k, s)| ShotEdit::of(s, e.number, k + 1)).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot(from: Option<f64>, to: Option<f64>, speed: Option<f64>) -> Shot {
        Shot { item: 1, name: "c.mp4".into(), clip_seconds: 30.0, best: 10.0, from, to, speed, ..Shot::default() }
    }

    #[test]
    fn windows_speeds_fits_and_focus_are_checked() {
        let ok = ShotEdit::of(&shot(Some(12.0), Some(18.5), Some(4.0)), 1, 1).unwrap();
        assert_eq!((ok.fixed(), ok.speed, ok.sound(), ok.timed()), (Some((12.0, 18.5)), 4.0, false, true));
        assert_eq!(ShotEdit::of(&shot(None, None, None), 1, 1).unwrap(), ShotEdit::default());
        assert!(!ShotEdit::default().timed() && ShotEdit::default().sound());
        // a window a hair past the measured length is the clip's end
        assert!(ShotEdit::of(&shot(Some(29.0), Some(30.04), None), 1, 1).is_ok());
        let full = Shot { fit: Some("FULL".into()), focus: Some([0.5, 0.2]), ..shot(None, None, None) };
        let e = ShotEdit::of(&full, 1, 1).unwrap();
        assert_eq!((e.fit, e.focus), (Some(Fit::Full), Some([0.5, 0.2])));
        let quiet = ShotEdit::of(&Shot { mute: Some(true), volume: Some(-20.0), ..shot(None, None, None) }, 1, 1).unwrap();
        assert!(quiet.mute && !quiet.sound() && quiet.volume == Some(-20.0) && !quiet.timed());
        for (bad, says) in [
            (shot(Some(-1.0), Some(5.0), None), "from"),
            (shot(Some(f64::NAN), Some(5.0), None), "not a number"),
            (shot(Some(2.0), Some(f64::INFINITY), None), "to"),
            (shot(Some(8.0), Some(5.0), None), "not after"),
            (shot(Some(5.0), Some(5.1), None), "shorter"),
            (shot(Some(5.0), Some(45.0), None), "past the end"),
            (shot(Some(31.0), None, None), "past the end"),
            (shot(Some(29.9), None, None), "leaves less"),
            (shot(None, None, Some(0.0)), "speed"),
            (shot(None, None, Some(1000.0)), "speed"),
            (shot(None, None, Some(f64::NAN)), "speed"),
            (shot(None, None, Some(-2.0)), "speed"),
            (Shot { fit: Some("stretch".into()), ..shot(None, None, None) }, "fit"),
            (Shot { focus: Some([0.5, 1.5]), ..shot(None, None, None) }, "focus"),
            (Shot { focus: Some([f64::NAN, 0.5]), ..shot(None, None, None) }, "focus"),
            (Shot { clip_seconds: 200.0, ..shot(Some(0.0), Some(190.0), Some(2.0)) }, "at most 60"),
            (Shot { volume: Some(40.0), ..shot(None, None, None) }, "volume"),
            (Shot { volume: Some(f64::NEG_INFINITY), ..shot(None, None, None) }, "volume"),
        ] {
            let err = std::panic::catch_unwind(|| ShotEdit::of(&bad, 3, 2)).expect("no panic").expect_err(says).to_string();
            assert!(err.contains("entry 3, shot 2") && err.contains(says), "{says}: {err}");
        }
    }

    #[test]
    fn an_anchored_window_takes_its_share_at_its_speed() {
        let e = ShotEdit { from: Some(10.0), speed: 2.0, ..ShotEdit::default() };
        assert_eq!(e.anchored(3.0, 30.0), Some((10.0, 6.0)));
        // …inside the clip
        assert_eq!(e.anchored(30.0, 30.0).map(|w| (w.0, (w.1 * 100.0).round() / 100.0)), Some((10.0, 19.8)));
        let e = ShotEdit { to: Some(4.0), ..ShotEdit::default() };
        assert_eq!(e.anchored(6.0, 30.0), Some((0.0, 4.0)), "from the clip's start at most");
        assert_eq!(ShotEdit::default().anchored(3.0, 30.0), None);
        // hostile shares stay finite
        let e = ShotEdit { from: Some(1.0), ..ShotEdit::default() };
        assert!(e.anchored(f64::NAN, 30.0).is_some_and(|(a, l)| a.is_finite() && l.is_finite()));
    }
}

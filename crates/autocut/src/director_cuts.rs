//! How each cut is made, from what the two shots show and say, in the edit's [`Style`] — picked
//! once the plan is set, whoever directed it. Every cut stays on its beat; only the way across it
//! changes, and each is an ordinary edit to change afterwards (a transition, a camera keyframe, a
//! trimmed sound).
//!
//! - **Dissolve**: into the ending, and across a jump in time (the shots filmed at least
//!   [`TIME_JUMP`] apart). Cinematic: more of them, and a dip to black across a day; energetic:
//!   a punch in (or a light leak where light effects are allowed).
//! - **Whip**: where the camera is already moving fast. Between two 360° shots their virtual
//!   cameras whip out and in (one of them turning by itself, or its footage moving fast); else a
//!   whip-pan transition (`cam_whip_pan`) the way the faster shot pans. Never two in a row; how
//!   many by the style (subtle 3, energetic 6, cinematic 1), each with the style's whip sound.
//! - **Match cut**: a plain cut lined up with the shot before — the subject where the last one
//!   was (a 360° camera turns so its subject sits there; a normal shot's subject already does),
//!   a pan carrying on the same way, or two pictures framed alike (image embeddings at least
//!   [`SIMILAR`] alike, short of a duplicate).
//! - **J and L cuts** under speech (`clip.splitEdit`): someone who starts talking as their shot
//!   begins is heard half a beat before they are seen; a sentence that runs past its shot's end
//!   finishes over the next picture. Never over the other shot's own speech.
//! - **Cut** otherwise.
//!
//! What is heard on the cuts, and whether light effects are used, is the style's. A video's
//! options or the taste profile can name one (`style`, `lightEffects`, `sounds`, `soundLevel`);
//! otherwise the director chooses it from the shots it planned ([`style_from_shots`]): lively
//! footage is cut energetically, calm wide footage spread over days cinematically, the rest
//! (and footage full of people talking) subtly.

use filmcraft_engine::project::scenes::CameraMove;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Camera, Candidate, Enter, Kind, Plan, rank::DUPLICATE, rank::cosine, round2};
use crate::quick::ItemSpeech;

/// Shots filmed at least this far apart (seconds) are a jump in time.
pub const TIME_JUMP: i64 = 3 * 3600;
/// A day: a jump this long dips to black in the cinematic style (seconds).
const DAY: i64 = 24 * 3600;
/// The camera is moving fast from this (frame widths a second).
const FAST: f64 = 0.25;
/// A pan worth carrying across a match cut (frame widths a second).
const PANNING: f64 = 0.08;
/// Pictures this alike (image embeddings) are framed alike.
pub const SIMILAR: f64 = 0.8;
/// Speech starting this soon after its shot begins leads into it (seconds).
const SPEECH_START: f64 = 0.8;
/// The longest a sentence runs on over the next shot (seconds).
const MAX_TAIL: f64 = 0.8;
/// How far a match cut moves a subject off the middle (shares of the frame).
const MAX_ALIGN: f64 = 0.3;

/// How lively the cuts are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Pace {
    /// Plain and match cuts, a few whips, dissolves across time: the footage first.
    #[default]
    Subtle,
    /// More whips, punch-ins across time, sounds on the cuts.
    Energetic,
    /// Long dissolves, dips to black, a whip at most.
    Cinematic,
}

/// The sounds a cut may make.
pub const SOUNDS: [&str; 5] = ["swish", "chime", "whoosh", "impact", "riser"];

/// The edit's style of cutting: how lively, whether light effects are used, which sounds are
/// heard on the cuts and how loud (0..1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Style {
    pub pace: Pace,
    pub light_effects: bool,
    pub sounds: Vec<String>,
    pub sound_level: f64,
}

impl Default for Style {
    /// The subtle style: a quiet swish under whips, a chime under the title, no light effects.
    fn default() -> Self {
        Style { pace: Pace::Subtle, light_effects: false, sounds: vec!["swish".into(), "chime".into()], sound_level: 0.4 }
    }
}

impl Pace {
    pub fn name(self) -> &'static str {
        match self {
            Pace::Subtle => "subtle",
            Pace::Energetic => "energetic",
            Pace::Cinematic => "cinematic",
        }
    }
}

impl Style {
    /// A pace with its own sounds (light effects as `light` says).
    pub fn of_pace(pace: Pace, light: bool) -> Style {
        let sounds: &[&str] = match pace {
            Pace::Subtle => &["swish", "chime"],
            Pace::Energetic => &["whoosh", "impact", "swish"],
            Pace::Cinematic => &["riser", "chime"],
        };
        Style { pace, light_effects: light, sounds: sounds.iter().map(|k| (*k).to_string()).collect(), sound_level: Style::default().sound_level }
    }

    /// The style of `p` (`style`, `lightEffects`, `sounds`, `soundLevel`), over `base`.
    pub fn from_params(p: &Value, base: &Style) -> Result<Style, String> {
        let mut s = base.clone();
        match p["style"].as_str().map(|v| v.trim().to_ascii_lowercase()) {
            None => {}
            Some(v) => {
                s.pace = match v.as_str() {
                    "subtle" | "calm" | "clean" => Pace::Subtle,
                    "energetic" | "lively" | "punchy" => Pace::Energetic,
                    "cinematic" | "film" => Pace::Cinematic,
                    o => return Err(format!("style: subtle, energetic or cinematic, not `{o}`")),
                };
                // a livelier style brings its own sounds unless they are given
                if p["sounds"].is_null() {
                    s.sounds = Style::of_pace(s.pace, false).sounds;
                }
            }
        }
        if let Some(b) = p["lightEffects"].as_bool() {
            s.light_effects = b;
        }
        if let Some(list) = p["sounds"].as_array() {
            let mut out = Vec::new();
            for v in list.iter().take(SOUNDS.len()) {
                let k = v.as_str().map(|x| x.trim().to_ascii_lowercase()).unwrap_or_default();
                if !SOUNDS.contains(&k.as_str()) {
                    return Err(format!("sounds: any of {}, not `{k}`", SOUNDS.join(", ")));
                }
                if !out.contains(&k) {
                    out.push(k);
                }
            }
            s.sounds = out;
        }
        if let Some(v) = p["soundLevel"].as_f64() {
            if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
                return Err("soundLevel: 0 to 1".into());
            }
            s.sound_level = v;
        }
        Ok(s)
    }

    fn allows(&self, kind: &str) -> bool {
        self.sounds.iter().any(|k| k == kind)
    }

    /// The sound under a whip: a whoosh in an energetic style that allows it, else a swish.
    pub fn whip_sound(&self) -> Option<&'static str> {
        match self.pace {
            Pace::Energetic if self.allows("whoosh") => Some("whoosh"),
            _ if self.allows("swish") => Some("swish"),
            _ if self.allows("whoosh") => Some("whoosh"),
            _ => None,
        }
    }

    /// The sound under the title.
    pub fn title_sound(&self) -> Option<&'static str> {
        match self.pace {
            Pace::Energetic if self.allows("impact") => Some("impact"),
            _ if self.allows("chime") => Some("chime"),
            _ if self.allows("impact") => Some("impact"),
            _ => None,
        }
    }

    /// The sound rising into the ending, when the style has one.
    pub fn ending_sound(&self) -> Option<&'static str> {
        (self.pace == Pace::Cinematic && self.allows("riser")).then_some("riser")
    }

    fn max_whips(&self) -> usize {
        match self.pace {
            Pace::Subtle => 3,
            Pace::Energetic => 6,
            Pace::Cinematic => 1,
        }
    }

    fn max_dissolves(&self) -> usize {
        match self.pace {
            Pace::Subtle | Pace::Energetic => 2,
            Pace::Cinematic => 4,
        }
    }

    /// How long a dissolve is, in beats.
    pub fn dissolve_beats(&self) -> f64 {
        if self.pace == Pace::Cinematic { 1.5 } else { 1.0 }
    }

    /// The camera moves fast enough for a whip from this (frame widths a second).
    fn fast(&self) -> f64 {
        if self.pace == Pace::Energetic { 0.15 } else { FAST }
    }
}

/// Lively: this share of the shots moving fast (or high energy on average, 0..1).
const LIVELY_SHARE: f64 = 0.3;
const LIVELY_ENERGY: f64 = 0.65;
/// Calm: energy at most this on average, and almost nothing moving fast.
const CALM_ENERGY: f64 = 0.4;
const CALM_SHARE: f64 = 0.15;
/// Shots of people talking from this share are cut plainly, so they are heard.
const TALKING_SHARE: f64 = 0.4;
/// A moment this dark (mean brightness, 0..1) is night or a dim room.
const NIGHT: f64 = 0.25;

/// The words of a brief that name a style.
fn brief_pace(brief: &str) -> Option<Pace> {
    let b = brief.to_lowercase();
    let words: Vec<&str> = b.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |list: &[&str]| words.iter().any(|w| list.contains(w));
    if has(&["cinematic", "filmic"]) {
        Some(Pace::Cinematic)
    } else if has(&["energetic", "hype", "punchy", "lively"]) {
        Some(Pace::Energetic)
    } else if has(&["calm", "gentle", "quiet", "subtle", "relaxed"]) {
        Some(Pace::Subtle)
    } else {
        None
    }
}

/// The style the planned shots ask for, when the edit leaves it to the director: a style the
/// brief names; else, by the shots, plain cuts where many of them have people talking, lively
/// cuts where many move fast or their energy is high (light effects too when they are mostly
/// not at night), long dissolves where they are calm and either wide or filmed over days, and
/// the subtle style otherwise. Returns the style and why.
pub fn style_from_shots(plan: &Plan, cands: &[Candidate], brief: &str) -> (Style, String) {
    let used: Vec<&Candidate> = plan.shots.iter().filter_map(|s| cands.iter().find(|c| c.id == s.clip)).collect();
    if let Some(pace) = brief_pace(brief) {
        return (Style::of_pace(pace, false), format!("{}: the brief asks for it", pace.name()));
    }
    if used.is_empty() {
        return (Style::default(), "subtle: no planned shots to judge by".into());
    }
    let n = used.len() as f64;
    let share = |f: &dyn Fn(&Candidate) -> bool| used.iter().filter(|c| f(c)).count() as f64 / n;
    let energy = used.iter().map(|c| if c.energy.is_finite() { c.energy.clamp(0.0, 1.0) } else { 0.0 }).sum::<f64>() / n;
    let fast = share(&|c| c.scores.motion >= FAST);
    let talking = share(&|c| c.talking);
    let night = share(&|c| c.scores.measured && c.scores.brightness < NIGHT);
    let wide = share(&|c| c.wide);
    let times: Vec<i64> = used.iter().filter_map(|c| c.captured).collect();
    let span = match (times.iter().min(), times.iter().max()) {
        (Some(a), Some(b)) => b.saturating_sub(*a),
        _ => 0,
    };
    let of = |x: f64| (x * n).round() as usize;
    let all = used.len();
    if talking >= TALKING_SHARE {
        return (Style::of_pace(Pace::Subtle, false), format!("subtle: people talk in {} of {all} shots, and plain cuts keep them heard", of(talking)));
    }
    if fast >= LIVELY_SHARE || energy >= LIVELY_ENERGY {
        let light = night < 0.5;
        let why = format!(
            "energetic: {} of {all} shots move fast and their energy is {energy:.2}{}",
            of(fast),
            if light { "; light effects, as most are not at night" } else { "; no light effects, as most are at night" }
        );
        return (Style::of_pace(Pace::Energetic, light), why);
    }
    if energy <= CALM_ENERGY && fast < CALM_SHARE && (span >= DAY || wide >= 0.5) {
        let why = if span >= DAY {
            format!("cinematic: calm shots (energy {energy:.2}) filmed over {:.0} days", span as f64 / DAY as f64)
        } else {
            format!("cinematic: calm, wide shots ({} of {all} wide, energy {energy:.2})", of(wide))
        };
        return (Style::of_pace(Pace::Cinematic, false), why);
    }
    (Style::of_pace(Pace::Subtle, false), format!("subtle: {} of {all} shots move fast and their energy is {energy:.2}, neither lively nor calm", of(fast)))
}

/// Where the subject of a shot sits across the frame, 0..1 (a normal shot's subject box; a
/// 360° shot framed on a named subject has it in the middle, or where a match cut put it).
fn subject_x(c: &Candidate, align: Option<f64>) -> Option<f64> {
    match c.kind {
        Kind::Vr => c.looks.first().filter(|l| l.named).map(|_| align.unwrap_or(0.5)),
        _ => c.subject.map(|b| (b[0] + b[2] / 2.0).clamp(0.0, 1.0)),
    }
}

/// Which way a moment's camera pans, if it pans: +1 right, −1 left.
fn pan(c: &Candidate) -> Option<f64> {
    match c.pan {
        CameraMove::PanRight => Some(1.0),
        CameraMove::PanLeft => Some(-1.0),
        _ => None,
    }
}

/// Whether a 360° shot's virtual camera turns by itself.
fn turns(cam: Camera) -> bool {
    matches!(cam, Camera::Orbit | Camera::Drift | Camera::WhipPan | Camera::BarrelRoll)
}

/// The speech of `item` (media seconds), kept segments in order.
fn segments(speech: &[ItemSpeech], item: u64) -> Vec<(f64, f64)> {
    speech.iter().filter(|s| s.item == item).flat_map(|s| s.kept.iter().map(|k| (k.start, k.end))).filter(|(a, b)| a.is_finite() && b > a).collect()
}

/// Pick every cut of `plan` (shots of `cands`, `speech` their words, `beat` seconds) in `style`:
/// each shot's way in (`enter`, a match cut's `align`), its sound's `lead` and `tail`, and `cut`
/// saying why.
pub fn joins(plan: &mut Plan, cands: &[Candidate], speech: &[ItemSpeech], beat: f64, style: &Style) {
    let n = plan.shots.len();
    let find = |id: &str| cands.iter().find(|c| c.id == id);
    let beat = if beat.is_finite() && beat > 0.05 { beat } else { 0.5 };
    let lead = (beat / 2.0).clamp(0.25, 0.6);
    let (mut dissolves, mut whips) = (0usize, 0usize);
    for s in plan.shots.iter_mut() {
        s.enter = Enter::Cut;
        s.align = None;
        s.lead = 0.0;
        s.tail = 0.0;
        s.cut = String::new();
    }
    for k in 1..n {
        let (Some(prev), Some(cur)) = (plan.shots.get(k - 1).cloned(), plan.shots.get(k).cloned()) else { continue };
        let (Some(pc), Some(cc)) = (find(&prev.clip), find(&cur.clip)) else { continue };
        let last = k + 1 == n;
        let apart = match (pc.captured, cc.captured) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a).saturating_abs()),
            _ => None,
        };
        let fast = pc.scores.motion.max(cc.scores.motion) >= style.fast();
        let prev_whip = matches!(prev.enter, Enter::Whip | Enter::Swipe);
        let (mut enter, mut align, mut why) = (Enter::Cut, None, String::from("a cut on the beat"));
        let room = whips < style.max_whips() && !prev_whip;
        if last && n > 2 {
            (enter, why) = (Enter::Dissolve, "a dissolve into the ending".into());
        } else if let Some(gap) = apart.filter(|g| *g >= TIME_JUMP && dissolves < style.max_dissolves()) {
            dissolves += 1;
            let hours = gap as f64 / 3600.0;
            (enter, why) = match style.pace {
                Pace::Cinematic if gap >= DAY => (Enter::DipToBlack, format!("a dip to black: a jump in time ({:.0} days later)", hours / 24.0)),
                Pace::Energetic if style.light_effects => (Enter::LightLeak, format!("a light leak: a jump in time ({hours:.0} h later)")),
                Pace::Energetic => (Enter::Punch, format!("a punch in: a jump in time ({hours:.0} h later)")),
                _ => (Enter::Dissolve, format!("a dissolve: a jump in time ({hours:.0} h later)")),
            };
        } else if room && pc.kind == Kind::Vr && cc.kind == Kind::Vr && (turns(prev.camera) || turns(cur.camera) || fast) {
            whips += 1;
            (enter, why) = (Enter::Whip, "a whip: the cameras turn out of one 360° shot and into the next".into());
        } else if room && fast && (pan(pc).is_some() || pan(cc).is_some()) {
            whips += 1;
            (enter, why) = (Enter::Swipe, "a whip pan: the camera is already moving fast".into());
        } else if let (Some(x), Some(y)) = (subject_x(pc, prev.align), subject_x(cc, None)) {
            if cc.kind == Kind::Vr {
                let a = round2(x.clamp(0.5 - MAX_ALIGN, 0.5 + MAX_ALIGN));
                (enter, align, why) = (Enter::Match, Some(a), "a match cut: the subject where the last one was".into());
            } else if (x - y).abs() < 0.15 {
                (enter, why) = (Enter::Match, "a match cut: the subjects in the same place".into());
            }
        }
        if enter == Enter::Cut {
            if let (Some(a), Some(b)) = (pan(pc), pan(cc))
                && a == b
                && pc.scores.motion.min(cc.scores.motion) >= PANNING
            {
                (enter, why) = (Enter::Match, "a match cut: the pan carries on".into());
            } else if let Some(sim) = cosine(&pc.embedding, &cc.embedding).filter(|s| (SIMILAR..DUPLICATE).contains(s)) {
                (enter, why) = (Enter::Match, format!("a match cut: pictures framed alike ({sim:.2})"));
            }
        }
        // the sound: a J cut into someone talking, an L cut out of a sentence still going
        let len = |s: &super::PlannedShot| f64::from(s.beats) * beat * s.camera.speed();
        let (p_end, c_start) = (prev.from + len(&prev), cur.from);
        let starts_talking = segments(speech, cc.item).iter().any(|(a, _)| *a >= c_start - 0.05 && *a <= c_start + SPEECH_START);
        let prev_talking_at_end = segments(speech, pc.item).iter().any(|(a, b)| *b > p_end - lead && *a < p_end);
        let runs_on = segments(speech, pc.item).iter().filter(|(a, b)| *a < p_end && *b > p_end).map(|(_, b)| b - p_end).fold(0.0, f64::max);
        let cur_talking_at_start = segments(speech, cc.item).iter().any(|(a, b)| *a < c_start + MAX_TAIL && *b > c_start);
        let photo = pc.kind == Kind::Photo || cc.kind == Kind::Photo;
        let mut sound = String::new();
        let whip = matches!(enter, Enter::Whip | Enter::Swipe);
        if !photo && !whip && starts_talking && !prev_talking_at_end && c_start >= lead {
            if let Some(s) = plan.shots.get_mut(k) {
                s.lead = lead;
            }
            sound = format!("; a J cut: their voice comes in {lead:.2} s before the picture");
        } else if !photo && !whip && runs_on > 0.1 && !cur_talking_at_start && p_end + runs_on.min(MAX_TAIL) <= pc.media_seconds {
            let tail = (runs_on + 0.15).min(MAX_TAIL);
            if let Some(s) = plan.shots.get_mut(k - 1) {
                s.tail = tail;
            }
            sound = format!("; an L cut: the sentence finishes over the next shot ({tail:.2} s)");
        }
        if let Some(s) = plan.shots.get_mut(k) {
            s.enter = enter;
            s.align = align;
            s.cut = format!("{why}{sound}");
        }
    }
    if let Some(first) = plan.shots.first_mut() {
        first.cut = "the opening".into();
    }
}

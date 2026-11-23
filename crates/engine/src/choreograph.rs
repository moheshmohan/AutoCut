//! `keyframes.choreograph` (PX5.2): several property changes on one or more targets, placed on a
//! timeline the way agents already place tweens with GSAP, resolved into ordinary keyframes on the
//! clips' effect parameters. Nothing new runs at render time: the keyframes stay editable in
//! Effect Controls like any others, and the whole call is one undo step.
//!
//! The placement grammar follows the public documentation of GSAP's timeline position parameter
//! (gsap.com, "Position Parameter"). No GSAP code was read or used, and nothing of GSAP ships with
//! FilmCraft.
//!
//! | `position` | the tween starts |
//! |---|---|
//! | none | at the end of the timeline so far (GSAP's default, `"+=0"`: the previous tween's end when tweens follow one another) |
//! | a number, `"1.5"` | that many seconds into the clip |
//! | `"<"` / `">"` | with the previous tween's start / at its end |
//! | `"<0.2"`, `"<-0.2"`, `">0.3"`, `">-0.1"` (also `"<+=0.2"`, `">-=0.1"`) | that far from the previous tween's start / end |
//! | `"+=0.2"` / `"-=0.2"` | that far past / before the end of the timeline so far |
//! | `"intro"`, `"intro+=0.5"`, `"intro-=0.5"` | at a label of `labels` (clip seconds), or that far from it |
//! | `"<25%"`, `">-25%"` | a share of the previous tween's duration from its start / end |
//! | `"+=50%"`, `"-=25%"`, `"<+=25%"`, `"intro+=30%"` | a share of this tween's own duration |
//!
//! Unlike GSAP, a label that `labels` does not define is an error (GSAP would add it at the end):
//! a typo should not move a tween silently.
//!
//! Eases: `"none"`/`"linear"`, `"hold"`, the Penner families by GSAP's names (`power1`…`power4`
//! = quad…quint, `sine`, `expo`, `circ`, `back`, `elastic`, `bounce`, each `.in`, `.out` (the
//! default) or `.inOut`; sampled into a custom curve of 32 points, so elastic and bounce are close
//! approximations and parameters such as `back.out(1.7)` are ignored, with a note), `"spring"` or
//! `"spring(dampingRatio, stiffness, mass)"`, `"steps(n)"`, `"wiggle(amplitude, frequency, seed)"`,
//! the keyframe modes `bezier`, `autoBezier`, `continuousBezier`, `easeIn`, `easeOut`, and objects
//! `{spring: {…}}`, `{steps: n}`, `{wiggle: {…}}`, `{curve: [[time, progress], …]}` (see
//! [`crate::keyframe_ease`]). The default is `power1.out`, as in GSAP.
//!
//! A tween writes a keyframe at its start (the `from` value, or what the parameter shows there) and
//! one at its end (`to`), with the ease on the first; keyframes the parameter had strictly inside
//! the tween are replaced. Scale parameters are interpolated in log space (`perceptual`) unless the
//! tween says `perceptual: false`. `stagger` delays each target after the first by `each` seconds,
//! in the order `from` says (`start`, `end`, `center`, or a target index).

use std::collections::BTreeMap;

use filmcraft_project::easing::{Bezier, ease_overshoot};
use filmcraft_project::keyframe::{Interpolation, SegmentEase};
use filmcraft_project::keyframe_ease::CURVE_RANGE;
use filmcraft_project::{ClipId, ParamKind, Sequence};
use filmcraft_time::Tick;
use serde_json::{Map, Value, json};

use crate::commands::{CommandSpec, bad, has_seq, json_to_param, param_value_to_json};
use crate::keyframe_ease::{curve_p, segment_json, spring_p, steps_p, wiggle_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "keyframes.choreograph";
/// Most tweens in one call.
pub const MAX_TWEENS: usize = 200;
/// Most targets (clips × effects) in one call.
pub const MAX_TARGETS: usize = 64;
/// Most labels.
pub const MAX_LABELS: usize = 64;
/// Longest label name, characters.
const MAX_LABEL_CHARS: usize = 64;
/// Latest time and longest duration a tween may name, seconds (a day).
const MAX_SECONDS: f64 = 86_400.0;
/// Points a named ease is sampled at.
const EASE_SAMPLES: usize = 32;
/// GSAP's defaults: half a second, `power1.out`.
const DEFAULT_DURATION: f64 = 0.5;
const DEFAULT_EASE: &str = "power1.out";
/// A tween may end this far past its clip's end (rounding of seconds to ticks).
const END_SLACK: f64 = 1e-6;

/// What a position is measured from.
#[derive(Clone, Debug, PartialEq)]
enum Anchor {
    /// The end of the timeline so far.
    End,
    PrevStart,
    PrevEnd,
    Label(String),
    /// The clip's start (an absolute time).
    Zero,
}

/// How far from the anchor.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Offset {
    Seconds(f64),
    /// A share of the tween's own duration.
    Own(f64),
    /// A share of the previous tween's duration.
    Previous(f64),
}

#[derive(Clone, Debug, PartialEq)]
struct Position {
    anchor: Anchor,
    offset: Offset,
}

/// A tween's ease as keyframe data.
#[derive(Clone, Debug, PartialEq)]
struct Ease {
    mode: Interpolation,
    segment: SegmentEase,
    /// As the caller wrote it (for the report).
    name: String,
}

struct Tween {
    prop: String,
    effect: Option<Value>,
    to: Value,
    from: Option<Value>,
    duration: f64,
    ease: Ease,
    position: Position,
    perceptual: Option<bool>,
    /// Resolved start and end, clip seconds (before any stagger).
    start: f64,
    end: f64,
}

/// One clip effect (or one of its masks) the tweens animate.
#[derive(Clone, Debug, PartialEq)]
struct Target {
    clip: ClipId,
    effect: usize,
    effect_id: String,
    mask: Option<usize>,
    /// Stagger delay, seconds.
    delay: f64,
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Choreograph Keyframes",
        menu: &[],
        shortcut: None,
        params: r#"{"target":{"clip":id|"clips":[id],"effect":str|index="motion"|"effects":[str|index]|"layer":n|name|"layers":[n|name],"mask":n?},"tweens":[{"prop":str,"to":value,"from":value?,"duration":seconds=0.5,"ease":"power1.out"|"none"|"hold"|"power1..4|sine|expo|circ|back|elastic|bounce(.in|.out|.inOut)"|"spring"|"spring(dampingRatio,stiffness,mass)"|"steps(n)"|"wiggle(amplitude,frequency,seed)"|"bezier|easeIn|easeOut|…"|{"spring":{…}}|{"steps":n}|{"wiggle":{…}}|{"curve":[[time,progress]]}?,"position":seconds|"<"|">"|"<0.2"|">-0.1"|"+=0.2"|"-=0.2"|"label"|"label+=0.5"|"<25%"|"+=50%"?,"effect":str|index?,"perceptual":bool?}] (1..200),"labels":{"name":seconds}?,"stagger":seconds|{"each":seconds,"from":"start"|"end"|"center"|index="start"}?,"defaults":{"duration":seconds?,"ease":ease?}?,"dryRun":bool?}"#,
        enabled: has_seq,
        run: choreograph,
        journal: true,
    }]
}

fn choreograph(s: &mut Session, p: &Value) -> Result<Value> {
    let o = p.as_object().ok_or_else(|| bad(CMD, "params are an object {target, tweens, labels?, stagger?, defaults?, dryRun?}"))?;
    known(o, "params", &["target", "tweens", "labels", "stagger", "defaults", "dryRun"])?;
    let dry = match o.get("dryRun") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`dryRun` must be true or false"))?,
    };
    let labels = labels_p(o.get("labels"))?;
    let (default_duration, default_ease) = defaults_p(o.get("defaults"))?;
    let mut notes: Vec<String> = Vec::new();
    let mut tweens = tweens_p(o.get("tweens"), &labels, default_duration, &default_ease, &mut notes)?;
    place(&mut tweens, &labels)?;
    let mut targets = targets_p(s, o.get("target").ok_or_else(|| bad(CMD, "need `target` ({clip, effect?})"))?)?;
    stagger_p(o.get("stagger"), &mut targets)?;
    let plan = Plan { tweens, targets, labels };
    let report = if dry {
        let mut q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
        apply(&mut q, &plan)?
    } else {
        s.edit_sequence_as("Choreograph Keyframes", None, |q, _, _| apply(q, &plan))?
    };
    let tweens: Vec<Value> = plan
        .tweens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut v = json!({"index": i, "prop": t.prop, "start": round(t.start), "end": round(t.end), "duration": round(t.duration), "ease": t.ease.name, "interpolation": t.ease.mode.name()});
            if let Some(e) = &t.effect {
                v["effect"] = e.clone();
            }
            v
        })
        .collect();
    let targets: Vec<Value> = plan
        .targets
        .iter()
        .map(|t| {
            let mut v = json!({"clip": t.clip.0, "effect": t.effect, "effectId": t.effect_id, "delay": round(t.delay)});
            if let Some(m) = t.mask {
                v["mask"] = json!(m);
            }
            v
        })
        .collect();
    let end = plan.tweens.iter().map(|t| t.end).fold(0.0, f64::max) + plan.targets.iter().map(|t| t.delay).fold(0.0, f64::max);
    let written = report.as_array().map_or(0, Vec::len);
    Ok(json!({
        "keyframes": report,
        "written": written,
        "tweens": tweens,
        "targets": targets,
        "labels": plan.labels,
        "duration": round(end),
        "notes": notes,
        "dryRun": dry,
    }))
}

struct Plan {
    tweens: Vec<Tween>,
    targets: Vec<Target>,
    labels: BTreeMap<String, f64>,
}

/// Seconds rounded to the microsecond, for reports.
fn round(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// At most 40 characters of a caller's text, for an error message.
fn clip_text(s: &str) -> String {
    s.chars().take(40).collect()
}

/// Refuse keys an object does not have.
fn known(o: &Map<String, Value>, what: &str, keys: &[&str]) -> Result<()> {
    match o.keys().find(|k| !keys.contains(&k.as_str())) {
        Some(k) => Err(bad(CMD, format!("{what} has no `{}`; it takes {}", clip_text(k), keys.join(", ")))),
        None => Ok(()),
    }
}

/// A finite number of seconds within `0..=MAX_SECONDS`.
fn seconds(v: &Value, what: &str) -> Result<f64> {
    v.as_f64()
        .filter(|x| x.is_finite() && (0.0..=MAX_SECONDS).contains(x))
        .ok_or_else(|| bad(CMD, format!("{what} is a number of seconds from 0 to {MAX_SECONDS}")))
}

fn labels_p(v: Option<&Value>) -> Result<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(out) };
    let o = v.as_object().ok_or_else(|| bad(CMD, "`labels` is {name: seconds}"))?;
    if o.len() > MAX_LABELS {
        return Err(bad(CMD, format!("{} labels; at most {MAX_LABELS}", o.len())));
    }
    for (name, t) in o {
        let ok = !name.is_empty()
            && name.chars().count() <= MAX_LABEL_CHARS
            && !name.starts_with(['<', '>', '+', '-', '.'])
            && !name.starts_with(|c: char| c.is_ascii_digit() || c.is_whitespace())
            && !name.ends_with(char::is_whitespace)
            && !name.contains("+=")
            && !name.contains("-=")
            && !name.contains('%');
        if !ok {
            return Err(bad(
                CMD,
                format!(
                    "label `{}`: a name of up to {MAX_LABEL_CHARS} characters, not starting with a digit, <, >, + or - and without +=, -= or %",
                    clip_text(name)
                ),
            ));
        }
        out.insert(name.clone(), seconds(t, &format!("label `{}`", clip_text(name)))?);
    }
    Ok(out)
}

fn defaults_p(v: Option<&Value>) -> Result<(f64, Value)> {
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok((DEFAULT_DURATION, json!(DEFAULT_EASE))) };
    let o = v.as_object().ok_or_else(|| bad(CMD, "`defaults` is {duration?, ease?}"))?;
    known(o, "`defaults`", &["duration", "ease"])?;
    let duration = match o.get("duration") {
        None | Some(Value::Null) => DEFAULT_DURATION,
        Some(d) => duration_p(d, "`defaults.duration`")?,
    };
    Ok((duration, o.get("ease").filter(|e| !e.is_null()).cloned().unwrap_or(json!(DEFAULT_EASE))))
}

/// A tween duration: more than zero (a jump is a `hold` ease on a short tween), at most a day.
fn duration_p(v: &Value, what: &str) -> Result<f64> {
    let d = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(CMD, format!("{what} is a number of seconds")))?;
    if d <= 0.0 {
        return Err(bad(CMD, format!("{what} is {d}: a duration must be more than 0 (for a jump, ease \"hold\" over a short tween)")));
    }
    if d > MAX_SECONDS {
        return Err(bad(CMD, format!("{what} is over {MAX_SECONDS} s")));
    }
    Ok(d)
}

fn tweens_p(v: Option<&Value>, labels: &BTreeMap<String, f64>, default_duration: f64, default_ease: &Value, notes: &mut Vec<String>) -> Result<Vec<Tween>> {
    let a = v.and_then(Value::as_array).ok_or_else(|| bad(CMD, "need `tweens`: [{prop, to, duration?, ease?, position?}]"))?;
    if a.is_empty() {
        return Err(bad(CMD, "`tweens` is empty"));
    }
    if a.len() > MAX_TWEENS {
        return Err(bad(CMD, format!("{} tweens; at most {MAX_TWEENS}", a.len())));
    }
    let mut out = Vec::with_capacity(a.len());
    for (i, t) in a.iter().enumerate() {
        let what = format!("tweens[{i}]");
        let o = t.as_object().ok_or_else(|| bad(CMD, format!("{what} is {{prop, to, …}}")))?;
        known(o, &what, &["prop", "to", "from", "duration", "ease", "position", "effect", "perceptual"])?;
        let prop = o.get("prop").and_then(Value::as_str).map(str::trim).filter(|p| !p.is_empty()).ok_or_else(|| bad(CMD, format!("{what} needs `prop`")))?;
        let to = o.get("to").filter(|v| !v.is_null()).cloned().ok_or_else(|| bad(CMD, format!("{what} needs `to`")))?;
        let duration = match o.get("duration") {
            None | Some(Value::Null) => default_duration,
            Some(d) => duration_p(d, &format!("{what}.duration"))?,
        };
        let ease = ease_p(o.get("ease").filter(|e| !e.is_null()).unwrap_or(default_ease), &what, notes)?;
        let position = position_p(o.get("position"), labels, &what)?;
        let perceptual = match o.get("perceptual") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_bool().ok_or_else(|| bad(CMD, format!("{what}.perceptual must be true or false")))?),
        };
        let effect = match o.get("effect") {
            None | Some(Value::Null) => None,
            Some(e @ (Value::String(_) | Value::Number(_))) => Some(e.clone()),
            Some(_) => return Err(bad(CMD, format!("{what}.effect is an effect id or index"))),
        };
        out.push(Tween {
            prop: prop.to_string(),
            effect,
            to,
            from: o.get("from").filter(|v| !v.is_null()).cloned(),
            duration,
            ease,
            position,
            perceptual,
            start: 0.0,
            end: 0.0,
        });
    }
    Ok(out)
}

/// A tween's `position` (see the module docs).
fn position_p(v: Option<&Value>, labels: &BTreeMap<String, f64>, what: &str) -> Result<Position> {
    let at_end = Position { anchor: Anchor::End, offset: Offset::Seconds(0.0) };
    let wrong = || bad(CMD, format!("{what}.position: seconds, \"<\", \">\", \"<0.2\", \"+=0.2\", \"-=0.2\", a label or \"label+=0.5\""));
    let s = match v {
        None | Some(Value::Null) => return Ok(at_end),
        Some(Value::Number(n)) => {
            let x = n.as_f64().filter(|x| x.is_finite()).ok_or_else(wrong)?;
            return Ok(Position { anchor: Anchor::Zero, offset: Offset::Seconds(x) });
        }
        Some(Value::String(s)) => s.trim(),
        Some(_) => return Err(wrong()),
    };
    if s.is_empty() {
        return Ok(at_end);
    }
    if let Ok(x) = s.parse::<f64>() {
        return if x.is_finite() { Ok(Position { anchor: Anchor::Zero, offset: Offset::Seconds(x) }) } else { Err(wrong()) };
    }
    let (anchor, rest, after_previous) = if let Some(r) = s.strip_prefix('<') {
        (Anchor::PrevStart, r, true)
    } else if let Some(r) = s.strip_prefix('>') {
        (Anchor::PrevEnd, r, true)
    } else if s.starts_with("+=") || s.starts_with("-=") {
        (Anchor::End, s, false)
    } else {
        let cut = [s.find("+="), s.find("-=")].into_iter().flatten().min().unwrap_or(s.len());
        let (name, rest) = s.split_at(cut);
        let name = name.trim();
        if !labels.contains_key(name) {
            let known: Vec<&str> = labels.keys().map(String::as_str).collect();
            return Err(bad(
                CMD,
                format!("{what}.position: no label `{}` (labels: {})", clip_text(name), if known.is_empty() { "none".to_string() } else { known.join(", ") }),
            ));
        }
        (Anchor::Label(name.to_string()), rest, false)
    };
    let r = rest.trim();
    if r.is_empty() {
        return Ok(Position { anchor, offset: Offset::Seconds(0.0) });
    }
    let (explicit, sign, num) = if let Some(n) = r.strip_prefix("+=") {
        (true, 1.0, n)
    } else if let Some(n) = r.strip_prefix("-=") {
        (true, -1.0, n)
    } else if after_previous {
        // after < and > the += is implied: "<0.2", "<-0.2"
        (false, 1.0, r)
    } else {
        return Err(wrong());
    };
    let num = num.trim();
    let (share, num) = match num.strip_suffix('%') {
        Some(n) => (true, n.trim()),
        None => (false, num),
    };
    let x = num.parse::<f64>().ok().filter(|x| x.is_finite()).ok_or_else(wrong)? * sign;
    let offset = match (share, explicit || !after_previous) {
        (false, _) => Offset::Seconds(x),
        // after += / -= a share is of this tween's duration, after a bare < or > of the previous one's
        (true, true) => Offset::Own(x / 100.0),
        (true, false) => Offset::Previous(x / 100.0),
    };
    Ok(Position { anchor, offset })
}

/// Resolve every tween's start and end in order (clip seconds, before any stagger).
fn place(tweens: &mut [Tween], labels: &BTreeMap<String, f64>) -> Result<()> {
    let (mut end, mut prev) = (0.0f64, None::<(f64, f64)>);
    for (i, t) in tweens.iter_mut().enumerate() {
        let (ps, pe) = prev.unwrap_or((0.0, 0.0));
        let base = match &t.position.anchor {
            Anchor::End => end,
            Anchor::PrevStart => ps,
            Anchor::PrevEnd => pe,
            Anchor::Label(n) => labels.get(n).copied().unwrap_or(0.0),
            Anchor::Zero => 0.0,
        };
        let off = match t.position.offset {
            Offset::Seconds(x) => x,
            Offset::Own(f) => f * t.duration,
            Offset::Previous(f) => f * (pe - ps),
        };
        let start = base + off;
        if !start.is_finite() || start < -END_SLACK {
            return Err(bad(CMD, format!("tweens[{i}] would start at {} s, before the clip starts", round(start))));
        }
        let start = start.max(0.0);
        let stop = start + t.duration;
        if stop > MAX_SECONDS {
            return Err(bad(CMD, format!("tweens[{i}] would end past {MAX_SECONDS} s")));
        }
        (t.start, t.end) = (start, stop);
        prev = Some((start, stop));
        end = end.max(stop);
    }
    Ok(())
}

/// A tween's ease (see the module docs); a note for a part that is approximated or ignored.
fn ease_p(v: &Value, what: &str, notes: &mut Vec<String>) -> Result<Ease> {
    let plain = |mode: Interpolation, name: &str| Ease { mode, segment: SegmentEase::default(), name: name.to_string() };
    let cmd = CMD;
    if let Some(o) = v.as_object() {
        if o.len() != 1 {
            return Err(bad(cmd, format!("{what}.ease is one of {{spring}}, {{steps}}, {{wiggle}}, {{curve}}")));
        }
        let mut segment = SegmentEase::default();
        let (mode, name) = match o.iter().next() {
            Some((k, x)) if k == "spring" => {
                segment.spring = Some(spring_p(cmd, x)?);
                (Interpolation::Spring, "spring")
            }
            Some((k, x)) if k == "steps" => {
                segment.steps = Some(steps_p(cmd, x)?);
                (Interpolation::Steps, "steps")
            }
            Some((k, x)) if k == "wiggle" => {
                segment.wiggle = Some(wiggle_p(cmd, x)?);
                (Interpolation::Wiggle, "wiggle")
            }
            Some((k, x)) if k == "curve" => {
                segment.curve = curve_p(cmd, x)?;
                (Interpolation::Custom, "custom")
            }
            _ => return Err(bad(cmd, format!("{what}.ease is one of {{spring}}, {{steps}}, {{wiggle}}, {{curve}}"))),
        };
        return Ok(Ease { mode, segment, name: name.to_string() });
    }
    let raw =
        v.as_str().map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| bad(cmd, format!("{what}.ease is a name such as \"power2.out\" or an object")))?;
    // name(args)
    let (name, args) = match raw.split_once('(') {
        Some((n, rest)) => {
            let inner = rest.strip_suffix(')').ok_or_else(|| bad(cmd, format!("{what}.ease `{}`: a closing parenthesis is missing", clip_text(raw))))?;
            let nums: Vec<f64> = inner
                .split(',')
                .map(str::trim)
                .filter(|a| !a.is_empty())
                .map(|a| a.parse::<f64>().ok().filter(|x| x.is_finite()))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad(cmd, format!("{what}.ease `{}`: the arguments are finite numbers", clip_text(raw))))?;
            (n.trim(), nums)
        }
        None => (raw, Vec::new()),
    };
    if args.len() > 3 {
        return Err(bad(cmd, format!("{what}.ease `{}`: at most three arguments", clip_text(raw))));
    }
    let lower = name.to_ascii_lowercase();
    let arg = |i: usize| args.get(i).copied();
    match lower.as_str() {
        "none" | "linear" => return Ok(plain(Interpolation::Linear, raw)),
        "hold" => return Ok(plain(Interpolation::Hold, raw)),
        "spring" => {
            let mut o = Map::new();
            for (i, k) in ["dampingRatio", "stiffness", "mass"].into_iter().enumerate() {
                if let Some(x) = arg(i) {
                    o.insert(k.into(), json!(x));
                }
            }
            let segment = SegmentEase { spring: Some(spring_p(cmd, &Value::Object(o))?), ..Default::default() };
            return Ok(Ease { mode: Interpolation::Spring, segment, name: raw.to_string() });
        }
        "steps" => {
            let n = arg(0).ok_or_else(|| bad(cmd, format!("{what}.ease: steps(n) needs n")))?;
            let segment = SegmentEase { steps: Some(steps_p(cmd, &json!(n))?), ..Default::default() };
            return Ok(Ease { mode: Interpolation::Steps, segment, name: raw.to_string() });
        }
        "wiggle" => {
            let mut o = Map::new();
            for (i, k) in ["amplitude", "frequency", "seed"].into_iter().enumerate() {
                if let Some(x) = arg(i) {
                    o.insert(k.into(), json!(x));
                }
            }
            let segment = SegmentEase { wiggle: Some(wiggle_p(cmd, &Value::Object(o))?), ..Default::default() };
            return Ok(Ease { mode: Interpolation::Wiggle, segment, name: raw.to_string() });
        }
        "custom" | "curve" => return Err(bad(cmd, format!("{what}.ease: a custom curve is {{\"curve\": [[time, progress], …]}}"))),
        _ => {}
    }
    if let Some(mode) =
        Interpolation::from_name(name).filter(|m| !matches!(m, Interpolation::Spring | Interpolation::Steps | Interpolation::Wiggle | Interpolation::Custom))
    {
        if !args.is_empty() {
            return Err(bad(cmd, format!("{what}.ease `{}` takes no arguments", clip_text(raw))));
        }
        return Ok(plain(mode, raw));
    }
    // a Penner family by GSAP's names
    let (family, kind) = lower.split_once('.').unwrap_or((lower.as_str(), "out"));
    let family = match family {
        "power0" => return Ok(plain(Interpolation::Linear, raw)),
        "sine" => 0,
        "quad" | "power1" => 1,
        "cubic" | "power2" => 2,
        "quart" | "power3" => 3,
        "quint" | "power4" | "strong" => 4,
        "expo" => 5,
        "circ" => 6,
        "back" => 7,
        "elastic" => 8,
        "bounce" => 9,
        _ => {
            return Err(bad(
                cmd,
                format!(
                    "{what}.ease `{}`: one of none, hold, power1..4, sine, expo, circ, back, elastic, bounce (.in, .out, .inOut), spring, steps(n), wiggle(…), bezier, easeIn, easeOut, or an object",
                    clip_text(raw)
                ),
            ));
        }
    };
    let kind = match kind {
        "in" => 0,
        "out" => 1,
        "inout" => 2,
        _ => return Err(bad(cmd, format!("{what}.ease `{}`: the type is .in, .out or .inOut", clip_text(raw)))),
    };
    if !args.is_empty() {
        notes.push(format!("{what}: the arguments of `{}` are not used (the standard curve is)", clip_text(raw)));
    }
    let index = 1 + family * 3 + kind;
    let n = EASE_SAMPLES as f64 + 1.0;
    let curve = (1..=EASE_SAMPLES)
        .map(|i| {
            let x = i as f64 / n;
            [x, ease_overshoot(index, Bezier::EASE, x).clamp(CURVE_RANGE.0, CURVE_RANGE.1)]
        })
        .collect();
    Ok(Ease { mode: Interpolation::Custom, segment: SegmentEase { curve, ..Default::default() }, name: raw.to_string() })
}

/// The clips and effects the tweens animate.
fn targets_p(s: &Session, v: &Value) -> Result<Vec<Target>> {
    let o = v.as_object().ok_or_else(|| bad(CMD, "`target` is {clip | clips, effect? | effects? | layer? | layers?, mask?}"))?;
    known(o, "`target`", &["clip", "clips", "effect", "effects", "layer", "layers", "mask"])?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let id_of = |v: &Value| v.as_u64().ok_or_else(|| bad(CMD, "a clip is a clip id"));
    let mut clips: Vec<ClipId> = Vec::new();
    if let Some(c) = o.get("clip").filter(|c| !c.is_null()) {
        clips.push(ClipId(id_of(c)?));
    }
    if let Some(a) = o.get("clips").filter(|c| !c.is_null()) {
        let a = a.as_array().ok_or_else(|| bad(CMD, "`target.clips` is a list of clip ids"))?;
        if a.len() > MAX_TARGETS {
            return Err(bad(CMD, format!("{} clips; at most {MAX_TARGETS} targets", a.len())));
        }
        for c in a {
            clips.push(ClipId(id_of(c)?));
        }
    }
    let mut seen = std::collections::HashSet::new();
    clips.retain(|c| seen.insert(*c));
    if clips.is_empty() {
        return Err(bad(CMD, "`target` needs `clip` or `clips`"));
    }
    let mask = match o.get("mask") {
        None | Some(Value::Null) => None,
        Some(m) => Some(m.as_u64().and_then(|m| usize::try_from(m).ok()).ok_or_else(|| bad(CMD, "`target.mask` is a mask index"))?),
    };
    let list = |k: &str, one: &str| -> Result<Vec<Value>> {
        let mut v: Vec<Value> = o.get(one).filter(|x| !x.is_null()).cloned().into_iter().collect();
        if let Some(a) = o.get(k).filter(|x| !x.is_null()) {
            let a = a.as_array().ok_or_else(|| bad(CMD, format!("`target.{k}` is a list")))?;
            if a.len() > MAX_TARGETS {
                return Err(bad(CMD, format!("{} {k}; at most {MAX_TARGETS} targets", a.len())));
            }
            v.extend(a.iter().cloned());
        }
        Ok(v)
    };
    let layers = list("layers", "layer")?;
    let effects = list("effects", "effect")?;
    if !layers.is_empty() && !effects.is_empty() {
        return Err(bad(CMD, "`target` names layers or effects, not both"));
    }
    let mut out = Vec::new();
    for c in clips {
        let (_, it) = seq.find_item(c).ok_or_else(|| bad(CMD, format!("no clip {} in the sequence", c.0)))?;
        if it.reverse {
            return Err(bad(CMD, format!("clip {} plays reversed: its keyframes run backwards; choreograph it before reversing", c.0)));
        }
        if it.frame_hold.is_some() && it.hold_filters {
            return Err(bad(CMD, format!("clip {} holds a frame with Hold Filters on: its effects do not animate", c.0)));
        }
        let mut indices = Vec::new();
        if !layers.is_empty() {
            for l in &layers {
                indices.push(crate::graphics::layer_effect_index(s, c, &json!({"layer": l})).map_err(|e| bad(CMD, format!("clip {}: {e}", c.0)))?.1);
            }
        } else if effects.is_empty() {
            indices.push(effect_index(it, &json!("motion")).ok_or_else(|| bad(CMD, format!("clip {} has no Motion effect", c.0)))?);
        } else {
            for e in &effects {
                indices.push(effect_index(it, e).ok_or_else(|| bad(CMD, format!("clip {} has no effect {}", c.0, clip_text(&e.to_string()))))?);
            }
        }
        for effect in indices {
            let Some(e) = it.effects.get(effect) else { continue };
            if let Some(m) = mask
                && e.masks.get(m).is_none()
            {
                return Err(bad(CMD, format!("clip {}: `{}` has no mask {m}", c.0, e.effect)));
            }
            if out.len() >= MAX_TARGETS {
                return Err(bad(CMD, format!("more than {MAX_TARGETS} targets (clips × effects)")));
            }
            out.push(Target { clip: c, effect, effect_id: e.effect.clone(), mask, delay: 0.0 });
        }
    }
    Ok(out)
}

/// An effect of `it` by id or index.
fn effect_index(it: &filmcraft_project::TrackItem, e: &Value) -> Option<usize> {
    match e {
        Value::Number(n) => n.as_u64().and_then(|i| usize::try_from(i).ok()).filter(|i| *i < it.effects.len()),
        Value::String(id) => it.effects.iter().position(|x| &x.effect == id),
        _ => None,
    }
}

/// `stagger`: each target `each` seconds after the one before it, in the order `from` says.
fn stagger_p(v: Option<&Value>, targets: &mut [Target]) -> Result<()> {
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(()) };
    let (each, from) = match v {
        Value::Number(_) => (seconds(v, "`stagger`")?, Value::Null),
        Value::Object(o) => {
            known(o, "`stagger`", &["each", "from"])?;
            let each = seconds(o.get("each").ok_or_else(|| bad(CMD, "`stagger` needs `each` (seconds)"))?, "`stagger.each`")?;
            (each, o.get("from").cloned().unwrap_or(Value::Null))
        }
        _ => return Err(bad(CMD, "`stagger` is seconds or {each, from?}")),
    };
    let n = targets.len();
    let last = n.saturating_sub(1) as f64;
    let order = |i: usize| -> Result<f64> {
        let i = i as f64;
        Ok(match &from {
            Value::Null => i,
            Value::String(f) if f == "start" => i,
            Value::String(f) if f == "end" => last - i,
            Value::String(f) if f == "center" => (i - last / 2.0).abs(),
            Value::Number(k) => {
                let k =
                    k.as_u64().filter(|k| (*k as usize) < n).ok_or_else(|| bad(CMD, format!("`stagger.from` index is from 0 to {}", n.saturating_sub(1))))?;
                (i - k as f64).abs()
            }
            _ => return Err(bad(CMD, "`stagger.from` is start, end, center or a target index")),
        })
    };
    for (i, t) in targets.iter_mut().enumerate() {
        t.delay = each * order(i)?;
    }
    Ok(())
}

/// Write every tween on every target into `q`; the keyframes written, as they ended up.
fn apply(q: &mut Sequence, plan: &Plan) -> Result<Value> {
    let mut written: Vec<(ClipId, usize, Option<usize>, String, Tick, f64)> = Vec::new();
    for target in &plan.targets {
        let (_, it) = q.find_item_mut(target.clip).ok_or_else(|| bad(CMD, format!("no clip {}", target.clip.0)))?;
        let dur = it.duration.seconds();
        for (i, t) in plan.tweens.iter().enumerate() {
            let (a, b) = (t.start + target.delay, t.end + target.delay);
            if b > dur + END_SLACK {
                return Err(bad(CMD, format!("tweens[{i}] ends at {} s on clip {}, after the clip ends ({} s)", round(b), target.clip.0, round(dur))));
            }
            let at = |s: f64| it.effect_time_at(it.start + Tick::from_seconds_f64(s.clamp(0.0, dur)));
            let (t0, t1) = (at(a), at(b));
            if t1 <= t0 {
                return Err(bad(CMD, format!("tweens[{i}] covers no media time on clip {} (its speed or time remapping)", target.clip.0)));
            }
            let (effect, mask) = match &t.effect {
                Some(e) => (
                    effect_index(it, e).ok_or_else(|| bad(CMD, format!("tweens[{i}]: clip {} has no effect {}", target.clip.0, clip_text(&e.to_string()))))?,
                    None,
                ),
                None => (target.effect, target.mask),
            };
            let e = it.effects.get_mut(effect).ok_or_else(|| bad(CMD, format!("tweens[{i}]: no effect {effect}")))?;
            // as an edit of Motion or Opacity in Effect Controls does, the effect is switched on
            if matches!(e.effect.as_str(), "motion" | "opacity") {
                e.enabled = true;
            }
            let def = if mask.is_none() { e.def().and_then(|d| d.param(&t.prop)) } else { None };
            if def.is_some_and(|d| !d.animatable) {
                return Err(bad(CMD, format!("tweens[{i}]: `{}` cannot be animated", t.prop)));
            }
            if mask.is_none()
                && !e.params.contains_key(&t.prop)
                && let Some(d) = def
            {
                e.params.insert(t.prop.clone(), filmcraft_project::Param::new(d.default.clone()));
            }
            let effect_id = e.effect.clone();
            let names: Vec<String> = match mask {
                Some(_) => filmcraft_project::mask::MASK_PARAMS.iter().map(|p| (*p).to_string()).collect(),
                None => e.params.keys().cloned().collect(),
            };
            let prm = match mask {
                Some(m) => e.masks.get_mut(m).and_then(|m| m.param_mut(&t.prop)),
                None => e.params.get_mut(&t.prop),
            }
            .ok_or_else(|| {
                let mut names = names.clone();
                names.sort();
                names.truncate(30);
                bad(CMD, format!("tweens[{i}]: `{}` has no `{}` (it has {})", effect_id, clip_text(&t.prop), names.join(", ")))
            })?;
            let value = |v: &Value, which: &str| -> Result<filmcraft_project::ParamValue> {
                let x = json_to_param(&prm.value, v).ok_or_else(|| bad(CMD, format!("tweens[{i}].{which} has the wrong type for `{}`", t.prop)))?;
                if let (Some(ParamKind::Float { min, max, .. }), Some(f)) = (def.map(|d| &d.kind), x.as_f64())
                    && !(*min..=*max).contains(&f)
                {
                    return Err(bad(CMD, format!("tweens[{i}].{which}: `{}` is from {min} to {max}, not {f}", t.prop)));
                }
                Ok(x)
            };
            let to = value(&t.to, "to")?;
            let from = match &t.from {
                Some(v) => value(v, "from")?,
                None => prm.value_at(t0),
            };
            if prm.keyframes.is_empty() {
                prm.value = from.clone();
            }
            prm.keyframes.retain(|k| k.time <= t0 || k.time >= t1);
            prm.put_keyframe(t0, from);
            prm.put_keyframe(t1, to);
            let scale = t.prop.contains("scale");
            if let Some(k) = prm.keyframes.iter_mut().find(|k| k.time == t0) {
                k.interp = t.ease.mode;
                let mut segment = t.ease.segment.clone();
                segment.perceptual = t.perceptual.unwrap_or(scale);
                k.ease = Some(Box::new(segment));
                k.tidy_ease();
            }
            for time in [t0, t1] {
                written.push((target.clip, effect, mask, t.prop.clone(), time, if time == t0 { a } else { b }));
            }
        }
    }
    // the keyframes as they ended up (a later tween may have replaced an earlier one's)
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (clip, effect, mask, prop, time, secs) in written {
        if !seen.insert((clip, effect, mask, prop.clone(), time)) {
            continue;
        }
        let Some((_, it)) = q.find_item(clip) else { continue };
        let Some(e) = it.effects.get(effect) else { continue };
        let p = match mask {
            Some(m) => e.masks.get(m).and_then(|m| m.param(&prop)),
            None => e.params.get(&prop),
        };
        let Some(k) = p.and_then(|p| p.keyframes.iter().find(|k| k.time == time)) else { continue };
        let mut v = json!({"clip": clip.0, "effect": effect, "effectId": e.effect, "param": prop, "time": time.0, "seconds": round(secs), "value": param_value_to_json(&k.value)});
        if let Some(m) = mask {
            v["mask"] = json!(m);
        }
        if let (Some(o), Value::Object(seg)) = (v.as_object_mut(), segment_json(k)) {
            o.extend(seg);
        }
        out.push(v);
    }
    Ok(Value::Array(out))
}

#[cfg(test)]
#[path = "choreograph_tests.rs"]
mod tests;

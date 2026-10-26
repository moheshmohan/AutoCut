//! 360° camera moves: the shots a 360° clip makes for a reel or a recap, written as keyframes of
//! its virtual camera (the `vr_reframe` effect, [`crate::vr360`]) that stay editable in Effect
//! Controls like any others.
//!
//! | command | does |
//! |---|---|
//! | `clip.vrMove` | one `move` on 360° clips (`clips`, default the selection, else every 360° clip), over `seconds` from `at` (clip seconds; each move has its own default place and length): `planetIn` (a tiny planet spinning open into the view), `planetOut` (the view folding into a tiny planet), `orbit` (a turn all the way round, `degrees`), `barrelRoll` (a roll about the view's axis), `whipPan` (a fast turn to `yaw`/`pitch`, or `degrees` along, with a little motion blur), `reveal` (from the sky — `from: "ground"`: the ground — down to the view), `pushIn` (the view narrowing by `amount`), `crashZoom` (the same, fast), `drift` (a slow turn of `degrees`); `direction` `right` or `left`; the view a move ends on is the camera's own there unless `yaw` / `pitch` / `fov` say; one undo step |
//! | `clip.hyperlapse360` | a 360° hyperlapse: the clip plays at `speed` % with frame blending, and its camera is held steady by `clip.stabilize360` (`mode`, `strong` by default; a job) |
//!
//! A move replaces the keyframes its parameters had inside its window and keeps those outside:
//! moves, a subject follow (`clip.follow360`) and stabilisation add up on one clip. Keyframes
//! are Bézier, eased into and out of each other as the move wants (a whip is quick in the
//! middle, an orbit nearly even).

use serde_json::{Value, json};

use filmcraft_project::keyframe::Interpolation;
use filmcraft_project::vr::REFRAME;
use filmcraft_project::{ClipId, EffectInstance, Param, ParamValue, TrackItem};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, has_seq, str_p};
use crate::vr360::{STAB_CMD, clips_360, ensure_reframe, media_time, view_json};
use crate::{EngineError, Result, Session};

const MOVE_CMD: &str = "clip.vrMove";
const HYPER_CMD: &str = "clip.hyperlapse360";
/// The motion blur effect a whip pan keyframes.
const BLUR: &str = "directional_blur";
/// Its Blur Length at the whip's fastest, at `strength` 1.
const WHIP_BLUR: f64 = 30.0;
/// Most clips one move is written on.
const MAX_CLIPS: usize = 1000;
/// A tiny planet's view: looking straight down from the far side of the sphere, wide.
const PLANET_PITCH: f64 = -90.0;
const PLANET_FOV: f64 = 150.0;

/// The camera moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    PlanetIn,
    PlanetOut,
    Orbit,
    BarrelRoll,
    WhipPan,
    Reveal,
    PushIn,
    CrashZoom,
    Drift,
}

impl Move {
    pub const ALL: [Move; 9] =
        [Move::PlanetIn, Move::PlanetOut, Move::Orbit, Move::BarrelRoll, Move::WhipPan, Move::Reveal, Move::PushIn, Move::CrashZoom, Move::Drift];

    pub fn id(self) -> &'static str {
        match self {
            Move::PlanetIn => "planetIn",
            Move::PlanetOut => "planetOut",
            Move::Orbit => "orbit",
            Move::BarrelRoll => "barrelRoll",
            Move::WhipPan => "whipPan",
            Move::Reveal => "reveal",
            Move::PushIn => "pushIn",
            Move::CrashZoom => "crashZoom",
            Move::Drift => "drift",
        }
    }

    /// The move an id (or a common name: `tinyPlanet`, `whip`, `roll`) names, any case.
    pub fn from_id(s: &str) -> Option<Move> {
        let k: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        Some(match k.as_str() {
            "planetin" | "tinyplanet" | "tinyplanetintro" | "planet" | "littleplanet" => Move::PlanetIn,
            "planetout" | "tinyplanetoutro" => Move::PlanetOut,
            "orbit" | "spin" | "pan360" => Move::Orbit,
            "barrelroll" | "roll" => Move::BarrelRoll,
            "whippan" | "whip" | "swishpan" => Move::WhipPan,
            "reveal" | "skyreveal" | "tiltdown" => Move::Reveal,
            "pushin" | "zoomin" => Move::PushIn,
            "crashzoom" => Move::CrashZoom,
            "drift" | "slowpan" => Move::Drift,
            _ => return None,
        })
    }

    /// How long it lasts by default, seconds; `None`: the rest of the clip.
    pub fn default_seconds(self) -> Option<f64> {
        match self {
            Move::PlanetIn => Some(2.0),
            Move::PlanetOut => Some(1.6),
            Move::BarrelRoll => Some(0.9),
            Move::WhipPan => Some(0.4),
            Move::Reveal => Some(1.6),
            Move::CrashZoom => Some(0.35),
            Move::Orbit | Move::PushIn | Move::Drift => None,
        }
    }

    /// Whether it closes the clip (its default window is the clip's last seconds).
    pub fn at_end(self) -> bool {
        self == Move::PlanetOut
    }

    /// How hard its keyframes ease (the Bézier handles' influence: 0 nearly even … 1 slow ends,
    /// quick middle).
    fn influence(self) -> f64 {
        match self {
            Move::Orbit => 0.12,
            Move::Drift => 0.3,
            Move::PushIn => 0.4,
            Move::BarrelRoll => 0.5,
            Move::Reveal | Move::PlanetIn | Move::PlanetOut => 0.6,
            Move::WhipPan | Move::CrashZoom => 0.78,
        }
    }
}

/// The virtual camera's view: degrees (`distance` 0 flat … 1 tiny planet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub fov: f64,
    pub distance: f64,
}

impl Cam {
    fn of(e: &EffectInstance, t: Tick) -> Cam {
        let fin = |v: f64, or: f64| if v.is_finite() { v } else { or };
        Cam {
            yaw: fin(e.f64_at("yaw", t), 0.0),
            pitch: fin(e.f64_at("pitch", t), 0.0),
            roll: fin(e.f64_at("roll", t), 0.0),
            fov: fin(e.f64_at("fov", t), 90.0),
            distance: fin(e.f64_at("distance", t), 0.0),
        }
    }
}

/// What shapes a move besides its window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveOptions {
    /// The view it ends on (whip, reveal, planet in): yaw and pitch; else the camera's own there.
    pub yaw: Option<f64>,
    pub pitch: Option<f64>,
    pub fov: Option<f64>,
    /// How far it turns (orbit, drift, whip, roll), degrees; the sign is the direction.
    pub degrees: Option<f64>,
    /// +1 right / clockwise, −1 left.
    pub direction: f64,
    /// A reveal from the ground rather than the sky.
    pub from_ground: bool,
    /// How much a push in narrows the view (0..1: the picture grows by that share).
    pub amount: Option<f64>,
    /// The whip's motion blur, 0..1.
    pub strength: f64,
}

/// One keyframe of a move: clip seconds and the values set there.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub seconds: f64,
    pub values: Vec<(&'static str, f64)>,
}

/// `to` turned the short way from `from` (degrees): the same direction, within 180° of `from`.
fn nearest_turn(from: f64, to: f64) -> f64 {
    let d = (to - from).rem_euclid(360.0);
    from + if d > 180.0 { d - 360.0 } else { d }
}

/// The field of view that makes a picture `1 + amount` times larger than `fov` does.
fn narrowed(fov: f64, amount: f64) -> f64 {
    let a = if amount.is_finite() { amount.clamp(0.0, 4.0) } else { 0.0 };
    (2.0 * ((fov.clamp(1.0, 179.0).to_radians() / 2.0).tan() / (1.0 + a)).atan().to_degrees()).clamp(5.0, 175.0)
}

/// The keyframes of move `m` from `a` to `b` clip seconds, the camera at `start` there before and
/// at `end` at `b`.
pub fn keys(m: Move, a: f64, b: f64, start: Cam, end: Cam, o: &MoveOptions) -> Vec<Key> {
    let dir = if o.direction < 0.0 { -1.0 } else { 1.0 };
    let target_yaw = o.yaw.unwrap_or(end.yaw);
    let target_pitch = o.pitch.unwrap_or(end.pitch).clamp(-90.0, 90.0);
    let target_fov = o.fov.unwrap_or(if end.fov > 140.0 { 90.0 } else { end.fov }).clamp(5.0, 175.0);
    let at = |u: f64| a + (b - a) * u;
    match m {
        Move::PlanetIn => {
            let yaw = target_yaw;
            vec![
                Key { seconds: a, values: vec![("yaw", yaw - 120.0 * dir), ("pitch", PLANET_PITCH), ("roll", 0.0), ("fov", PLANET_FOV), ("distance", 1.0)] },
                // the planet turns a little before it opens
                Key {
                    seconds: at(0.3),
                    values: vec![("yaw", yaw - 90.0 * dir), ("pitch", PLANET_PITCH), ("roll", 0.0), ("fov", PLANET_FOV), ("distance", 1.0)],
                },
                Key {
                    seconds: b,
                    values: vec![("yaw", yaw), ("pitch", target_pitch), ("roll", end.roll), ("fov", target_fov), ("distance", end.distance.min(0.2))],
                },
            ]
        }
        Move::PlanetOut => vec![
            Key {
                seconds: a,
                values: vec![("yaw", start.yaw), ("pitch", start.pitch), ("roll", start.roll), ("fov", start.fov), ("distance", start.distance)],
            },
            Key { seconds: b, values: vec![("yaw", start.yaw + 90.0 * dir), ("pitch", PLANET_PITCH), ("roll", 0.0), ("fov", PLANET_FOV), ("distance", 1.0)] },
        ],
        Move::Orbit => {
            let turn = o.degrees.unwrap_or(360.0 * dir);
            vec![Key { seconds: a, values: vec![("yaw", start.yaw)] }, Key { seconds: b, values: vec![("yaw", start.yaw + turn)] }]
        }
        Move::Drift => {
            let turn = o.degrees.unwrap_or(20.0 * dir);
            vec![Key { seconds: a, values: vec![("yaw", start.yaw)] }, Key { seconds: b, values: vec![("yaw", start.yaw + turn)] }]
        }
        Move::BarrelRoll => {
            let turn = o.degrees.unwrap_or(360.0 * dir);
            vec![
                Key { seconds: a, values: vec![("roll", start.roll), ("fov", start.fov)] },
                // a little wider at its middle: the roll reads as the camera's, not the world's
                Key { seconds: at(0.5), values: vec![("fov", (start.fov + 15.0).min(160.0))] },
                Key { seconds: b, values: vec![("roll", start.roll + turn), ("fov", start.fov)] },
            ]
        }
        Move::WhipPan => {
            let to = match (o.yaw, o.degrees) {
                (Some(y), _) => nearest_turn(start.yaw, y),
                (None, Some(d)) => start.yaw + d,
                (None, None) => start.yaw + 90.0 * dir,
            };
            vec![
                Key { seconds: a, values: vec![("yaw", start.yaw), ("pitch", start.pitch)] },
                Key { seconds: b, values: vec![("yaw", to), ("pitch", o.pitch.unwrap_or(start.pitch).clamp(-90.0, 90.0))] },
            ]
        }
        Move::Reveal => {
            let from = if o.from_ground { -80.0 } else { 80.0 };
            vec![
                Key { seconds: a, values: vec![("yaw", target_yaw), ("pitch", from), ("fov", target_fov)] },
                Key { seconds: b, values: vec![("yaw", target_yaw), ("pitch", target_pitch), ("fov", target_fov)] },
            ]
        }
        Move::PushIn | Move::CrashZoom => {
            let amount = o.amount.unwrap_or(if m == Move::CrashZoom { 0.6 } else { 0.2 });
            let to = o.fov.unwrap_or_else(|| narrowed(start.fov, amount));
            vec![Key { seconds: a, values: vec![("fov", start.fov)] }, Key { seconds: b, values: vec![("fov", to.clamp(5.0, 175.0))] }]
        }
    }
}

/// Set `id` to `v` at `t` as a Bézier keyframe of `influence`.
fn put(e: &mut EffectInstance, id: &str, t: Tick, v: f64, influence: f64) {
    if e.param(id).is_none() {
        e.params.insert(id.to_string(), Param::new(ParamValue::Float(v)));
    }
    let Some(p) = e.param_mut(id) else { return };
    p.put_keyframe(t, ParamValue::Float(v));
    if let Some(k) = p.keyframes.iter_mut().find(|k| k.time == t) {
        k.interp = Interpolation::Bezier;
        k.out_influence = influence;
        k.in_influence = influence;
    }
}

/// Drop `id`'s keyframes from `a` to `b` (media time), both ends in.
fn clear(e: &mut EffectInstance, id: &str, a: Tick, b: Tick) {
    if let Some(p) = e.param_mut(id) {
        let keep: Vec<_> = p.keyframes.iter().filter(|k| k.time < a || k.time > b).cloned().collect();
        if keep.is_empty() && !p.keyframes.is_empty() {
            // the static value the parameter falls back to: what it showed before the window
            p.value = p.value_at(a);
        }
        p.keyframes = keep;
    }
}

/// The window of a move on a clip `dur` seconds long: `[from, to]` clip seconds.
pub fn window(m: Move, dur: f64, at: Option<f64>, seconds: Option<f64>) -> Option<(f64, f64)> {
    if !(dur.is_finite() && dur > 0.0) {
        return None;
    }
    let len = seconds.or(m.default_seconds()).filter(|s| s.is_finite() && *s > 0.0);
    let from = match (at, len) {
        (Some(a), _) => a,
        (None, Some(l)) if m.at_end() => dur - l,
        (None, _) => 0.0,
    };
    let from = if from.is_finite() { from.clamp(0.0, dur) } else { 0.0 };
    let to = len.map_or(dur, |l| (from + l).min(dur));
    (to - from > 1e-3).then_some((from, to))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: MOVE_CMD,
            label: "360 Camera Move",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"move":"planetIn"|"planetOut"|"orbit"|"barrelRoll"|"whipPan"|"reveal"|"pushIn"|"crashZoom"|"drift","at":seconds?,"seconds":f64?,"yaw":deg?,"pitch":-90..90?,"fov":5..175?,"degrees":deg?,"direction":"right"|"left"="right","from":"sky"|"ground"="sky","amount":0..4?,"strength":0..1=0.5}"#,
            enabled: has_seq,
            run: camera_move,
            journal: true,
        },
        CommandSpec {
            id: HYPER_CMD,
            label: "360 Hyperlapse",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"speed":150..5000=600,"mode":"smooth"|"light"|"strong"|"lock"="strong","wait":bool=false}"#,
            enabled: has_seq,
            run: hyperlapse,
            journal: true,
        },
    ]
}

/// A number parameter within `lo..=hi`, if given.
fn num(cmd: &str, p: &Value, key: &str, lo: f64, hi: f64) -> Result<Option<f64>> {
    match p.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            v.as_f64().filter(|x| x.is_finite() && (lo..=hi).contains(x)).map(Some).ok_or_else(|| bad(cmd, format!("`{key}` is a number from {lo} to {hi}")))
        }
    }
}

fn camera_move(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = MOVE_CMD;
    let name = str_p(p, "move").ok_or_else(|| bad(cmd, "need `move`"))?;
    let m = Move::from_id(name)
        .ok_or_else(|| bad(cmd, format!("`move` is one of {}, not `{}`", Move::ALL.map(Move::id).join(", "), name.chars().take(40).collect::<String>())))?;
    let at = num(cmd, p, "at", 0.0, 86_400.0)?;
    let seconds = num(cmd, p, "seconds", 0.04, 86_400.0)?;
    let direction = match str_p(p, "direction") {
        None | Some("right") => 1.0,
        Some("left") => -1.0,
        Some(o) => return Err(bad(cmd, format!("`direction` is right or left, not `{}`", o.chars().take(20).collect::<String>()))),
    };
    let from_ground = match str_p(p, "from") {
        None | Some("sky") => false,
        Some("ground") => true,
        Some(o) => return Err(bad(cmd, format!("`from` is sky or ground, not `{}`", o.chars().take(20).collect::<String>()))),
    };
    let o = MoveOptions {
        yaw: num(cmd, p, "yaw", -100_000.0, 100_000.0)?,
        pitch: num(cmd, p, "pitch", -90.0, 90.0)?,
        fov: num(cmd, p, "fov", 5.0, 175.0)?,
        degrees: num(cmd, p, "degrees", -100_000.0, 100_000.0)?,
        direction,
        from_ground,
        amount: num(cmd, p, "amount", 0.0, 4.0)?,
        strength: num(cmd, p, "strength", 0.0, 1.0)?.unwrap_or(0.5),
    };
    let clips: Vec<ClipId> = clips_360(s, p, cmd)?.iter().map(|it| it.id).take(MAX_CLIPS).collect();
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let frame = (seq.settings.width, seq.settings.height);
    let label = format!("360 Camera Move: {}", m.id());
    let report = s.edit_sequence(&label, |q, _, _| {
        let mut out = Vec::new();
        for c in &clips {
            let (_, it) = q.find_item_mut(*c).ok_or(filmcraft_edit::EditError::NoItem(*c))?;
            let Some((from, to)) = window(m, it.duration.seconds(), at, seconds) else {
                out.push(json!({"clip": c.0, "skipped": "the move's window is outside the clip"}));
                continue;
            };
            apply(it, frame, m, from, to, &o).ok_or_else(|| EngineError::Other("no 360 Reframe effect definition".into()))?;
            let mut v = view_json(*c, it);
            v["move"] = json!(m.id());
            v["from"] = json!((from * 1000.0).round() / 1000.0);
            v["to"] = json!((to * 1000.0).round() / 1000.0);
            out.push(v);
        }
        Ok(out)
    })?;
    Ok(json!({"clips": report}))
}

/// Write move `m` over `[from, to]` clip seconds on `it` (its camera added when missing).
pub(crate) fn apply(it: &mut TrackItem, frame: (u32, u32), m: Move, from: f64, to: f64, o: &MoveOptions) -> Option<()> {
    let (ta, tb) = (media_time(it, from), media_time(it, to));
    let mid = media_time(it, (from + to) / 2.0);
    let e = ensure_reframe(it, frame)?;
    let (start, end) = (Cam::of(e, ta), Cam::of(e, tb));
    let list = keys(m, from, to, start, end, o);
    let written: std::collections::BTreeSet<&str> = list.iter().flat_map(|k| k.values.iter().map(|(id, _)| *id)).collect();
    for id in &written {
        clear(e, id, ta, tb);
    }
    let times: Vec<Tick> = list.iter().map(|k| media_time(it, k.seconds)).collect();
    let e = it.effects.iter_mut().find(|e| e.effect == REFRAME)?;
    for (k, t) in list.iter().zip(&times) {
        for (id, v) in &k.values {
            put(e, id, *t, *v, m.influence());
        }
    }
    if m == Move::WhipPan && o.strength > 0.0 {
        whip_blur(it, ta, mid, tb, o.strength);
    }
    Some(())
}

/// A whip's motion blur: Directional Blur along the turn, none at its ends and most at its middle
/// (added after the camera when the clip has none).
fn whip_blur(it: &mut TrackItem, a: Tick, mid: Tick, b: Tick, strength: f64) {
    if !it.effects.iter().any(|e| e.effect == BLUR) {
        let Some(def) = filmcraft_project::find_effect(BLUR) else { return };
        let mut e = def.instance();
        e.params.insert("direction".into(), Param::new(ParamValue::Float(90.0)));
        e.params.insert("length".into(), Param::new(ParamValue::Float(0.0)));
        let at = it.effects.iter().position(|e| e.effect == REFRAME).map_or(0, |i| i.saturating_add(1)).min(it.effects.len());
        it.effects.insert(at, e);
    }
    let Some(e) = it.effects.iter_mut().find(|e| e.effect == BLUR) else { return };
    clear(e, "length", a, b);
    for (t, v) in [(a, 0.0), (mid, WHIP_BLUR * strength.clamp(0.0, 1.0)), (b, 0.0)] {
        put(e, "length", t, v, 0.5);
    }
}

fn hyperlapse(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = HYPER_CMD;
    let speed = num(cmd, p, "speed", 150.0, 5000.0)?.unwrap_or(600.0);
    let mode = str_p(p, "mode").unwrap_or("strong");
    if !matches!(mode, "smooth" | "light" | "strong" | "lock") {
        return Err(bad(cmd, "`mode` is smooth, light, strong or lock"));
    }
    let clips: Vec<u64> = clips_360(s, p, cmd)?.iter().map(|it| it.id.0).take(MAX_CLIPS).collect();
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
    // the speed first (its length changes: later clips stay where they are), then the steadying
    s.execute("clip.speedDuration", json!({"clips": clips, "speed": speed, "ripple": false, "interpolation": "frameBlending"}))?;
    let steady = s.execute(STAB_CMD, json!({"clips": clips, "mode": mode, "wait": wait}))?;
    Ok(json!({"clips": clips, "speed": speed, "stabilize": steady}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vr360::tests_support::{demo_360, first_clip};

    fn cam() -> Cam {
        Cam { yaw: 10.0, pitch: 5.0, roll: 0.0, fov: 90.0, distance: 0.0 }
    }

    fn value(k: &Key, id: &str) -> Option<f64> {
        k.values.iter().find(|(i, _)| *i == id).map(|(_, v)| *v)
    }

    #[test]
    fn every_move_has_an_id_and_its_keys_start_and_end_its_window() {
        for m in Move::ALL {
            assert_eq!(Move::from_id(m.id()), Some(m));
            let ks = keys(m, 1.0, 3.0, cam(), cam(), &MoveOptions { direction: 1.0, ..Default::default() });
            assert_eq!(ks.first().map(|k| k.seconds), Some(1.0), "{m:?}");
            assert_eq!(ks.last().map(|k| k.seconds), Some(3.0), "{m:?}");
            assert!(ks.iter().all(|k| k.values.iter().all(|(_, v)| v.is_finite())), "{m:?}");
        }
        assert_eq!(Move::from_id("Tiny Planet"), Some(Move::PlanetIn));
        assert_eq!(Move::from_id("whip"), Some(Move::WhipPan));
        assert_eq!(Move::from_id("loop-de-loop"), None);
    }

    #[test]
    fn moves_turn_as_they_say() {
        let o = MoveOptions { direction: 1.0, ..Default::default() };
        // a tiny planet opening onto the view it ends on
        let p = keys(Move::PlanetIn, 0.0, 2.0, cam(), Cam { yaw: 40.0, ..cam() }, &o);
        assert_eq!((value(&p[0], "pitch"), value(&p[0], "distance")), (Some(PLANET_PITCH), Some(1.0)));
        assert_eq!((value(&p[2], "yaw"), value(&p[2], "pitch"), value(&p[2], "distance")), (Some(40.0), Some(5.0), Some(0.0)));
        // an orbit all the way round, left
        let left = MoveOptions { direction: -1.0, ..Default::default() };
        let orbit = keys(Move::Orbit, 0.0, 4.0, cam(), cam(), &left);
        assert_eq!(value(&orbit[1], "yaw"), Some(10.0 - 360.0));
        // a whip to a direction takes the short way round
        let whip = keys(Move::WhipPan, 0.0, 0.4, cam(), cam(), &MoveOptions { yaw: Some(350.0), ..o });
        assert_eq!(value(&whip[1], "yaw"), Some(-10.0));
        // a reveal from the sky down to the view
        let r = keys(Move::Reveal, 0.0, 1.5, cam(), cam(), &o);
        assert_eq!((value(&r[0], "pitch"), value(&r[1], "pitch")), (Some(80.0), Some(5.0)));
        let g = keys(Move::Reveal, 0.0, 1.5, cam(), cam(), &MoveOptions { from_ground: true, ..o });
        assert_eq!(value(&g[0], "pitch"), Some(-80.0));
        // a push in makes the picture 20 % larger
        let push = keys(Move::PushIn, 0.0, 3.0, cam(), cam(), &o);
        let f = value(&push[1], "fov").unwrap();
        assert!(((45.0_f64.to_radians().tan() / (f / 2.0).to_radians().tan()) - 1.2).abs() < 1e-9, "{f}");
        // a roll all the way round, a little wider half way
        let roll = keys(Move::BarrelRoll, 0.0, 1.0, cam(), cam(), &o);
        assert_eq!((value(&roll[2], "roll"), value(&roll[1], "fov")), (Some(360.0), Some(105.0)));
    }

    #[test]
    fn windows_sit_where_the_move_belongs() {
        assert_eq!(window(Move::PlanetIn, 10.0, None, None), Some((0.0, 2.0)));
        assert_eq!(window(Move::PlanetOut, 10.0, None, None), Some((8.4, 10.0)));
        assert_eq!(window(Move::Orbit, 10.0, None, None), Some((0.0, 10.0)));
        assert_eq!(window(Move::WhipPan, 10.0, Some(9.9), None), Some((9.9, 10.0)));
        assert_eq!(window(Move::WhipPan, 10.0, Some(10.0), None), None);
        assert_eq!(window(Move::Drift, f64::NAN, None, None), None);
        assert_eq!(window(Move::Drift, 5.0, Some(f64::INFINITY), Some(2.0)), Some((0.0, 2.0)));
    }

    #[test]
    fn a_move_is_written_on_the_clip_and_keeps_what_is_outside_it() {
        let mut s = demo_360();
        let (c, _) = first_clip(&s);
        let r = s.execute(MOVE_CMD, json!({"clips": [c.0], "move": "planetIn"})).unwrap();
        assert_eq!(r["clips"][0]["move"], json!("planetIn"), "{r}");
        let undo = s.history.undo.len();
        // a whip later in the clip keeps the planet's keys and adds its blur
        s.execute(MOVE_CMD, json!({"clips": [c.0], "move": "whipPan", "at": 3.0, "yaw": 120.0})).unwrap();
        assert_eq!(s.history.undo.len(), undo + 1, "one undo step");
        let it = s.active_sequence().unwrap().find_item(c).unwrap().1.clone();
        let e = it.effect(REFRAME).unwrap();
        let t0 = media_time(&it, 0.0);
        assert_eq!(e.f64_at("distance", t0), 1.0, "still a planet at the start");
        let t3 = media_time(&it, 3.4);
        assert!((e.f64_at("yaw", t3) - 120.0).abs() < 1e-6);
        let yaw = e.param("yaw").unwrap();
        assert!(yaw.keyframes.len() >= 5, "the planet's three and the whip's two: {}", yaw.keyframes.len());
        assert!(yaw.keyframes.iter().all(|k| k.interp == Interpolation::Bezier));
        let blur = it.effect(BLUR).expect("the whip's motion blur");
        assert!(blur.f64_at("length", media_time(&it, 3.2)) > 10.0);
        assert_eq!(blur.f64_at("length", media_time(&it, 3.4)), 0.0);
        // each move by name
        for m in Move::ALL {
            let r = s.execute(MOVE_CMD, json!({"clips": [c.0], "move": m.id()})).unwrap();
            assert!(r["clips"][0]["reframed"] == true, "{m:?}: {r}");
        }
    }

    #[test]
    fn hostile_moves_are_refused() {
        let mut s = demo_360();
        let (c, _) = first_clip(&s);
        for bad in [
            json!({"clips": [c.0]}),
            json!({"clips": [c.0], "move": "loop"}),
            json!({"clips": [c.0], "move": "orbit", "seconds": -1}),
            json!({"clips": [c.0], "move": "orbit", "at": f64::MAX}),
            json!({"clips": [c.0], "move": "reveal", "from": "space"}),
            json!({"clips": [c.0], "move": "whipPan", "direction": "up"}),
            json!({"clips": [c.0], "move": "pushIn", "amount": 99}),
            json!({"clips": [999_999], "move": "orbit"}),
            json!({"clips": [c.0], "speed": 10}),
        ] {
            let id = if bad.get("speed").is_some() { HYPER_CMD } else { MOVE_CMD };
            assert!(s.execute(id, bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_hyperlapse_speeds_the_clip_and_steadies_it() {
        let mut s = demo_360();
        let (c, _) = first_clip(&s);
        let before = s.active_sequence().unwrap().find_item(c).unwrap().1.duration;
        let r = s.execute(HYPER_CMD, json!({"clips": [c.0], "speed": 400, "wait": true})).unwrap();
        assert_eq!(r["speed"], json!(400.0), "{r}");
        let it = s.active_sequence().unwrap().find_item(c).unwrap().1.clone();
        assert!(it.duration.0 * 3 < before.0, "a quarter as long: {:?} from {:?}", it.duration, before);
        assert!(r["stabilize"].is_object(), "{r}");
    }
}

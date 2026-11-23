//! Command parameters of the keyframe eases (PX5.1), shared by `effects.setInterpolation`,
//! `effects.setKeyframe` and `keyframes.choreograph`.
//!
//! | param | value |
//! |---|---|
//! | `spring` | `{dampingRatio?: 0.05..10 = 0.9, stiffness?: 0.1..10000 = 150, mass?: 0.01..100 = 1, maxOvershoot?: 0..1 = 0.25}` (`true`: the default spring) |
//! | `steps` | a whole number of steps, 1..1000 |
//! | `wiggle` | `{amplitude?: 0..1e6 = 10 (the parameter's units), frequency?: 0..120 Hz = 2, seed?: integer = 1}` |
//! | `curve` | up to 64 `[time, progress]` points, time strictly increasing inside (0, 1), progress clamped to −1..2 |
//! | `perceptual` | `true`: interpolate in log space (an even zoom) |
//!
//! `null` removes a number set before. Numbers outside a range are clamped to it; a non-finite
//! number, a key the object does not have, a curve out of order or too long is an error. The
//! damping *ratio* is asked for by that name: other tools' `damping` is a coefficient.

use filmcraft_project::keyframe::{Interpolation, Keyframe, SegmentEase, Spring, Wiggle};
use filmcraft_project::keyframe_ease::{
    CURVE_RANGE, MAX_CURVE_POINTS, MAX_STEPS, MAX_WIGGLE_AMPLITUDE, MAX_WIGGLE_FREQUENCY, SPRING_DAMPING_RATIO, SPRING_MASS, SPRING_MAX_OVERSHOOT,
    SPRING_STIFFNESS,
};
use serde_json::{Map, Value, json};

use crate::Result;
use crate::commands::bad;

/// What a command changes in a keyframe's segment numbers (`None`: left as it is; `Some(None)`:
/// removed).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct EaseEdit {
    pub spring: Option<Option<Spring>>,
    pub steps: Option<Option<u32>>,
    pub wiggle: Option<Option<Wiggle>>,
    pub curve: Option<Vec<[f64; 2]>>,
    pub perceptual: Option<bool>,
}

impl EaseEdit {
    pub fn is_empty(&self) -> bool {
        *self == EaseEdit::default()
    }

    /// Write the changes into `k`'s segment.
    pub fn apply(&self, k: &mut Keyframe) {
        let e: &mut SegmentEase = k.segment_mut();
        if let Some(s) = self.spring {
            e.spring = s;
        }
        if let Some(n) = self.steps {
            e.steps = n;
        }
        if let Some(w) = self.wiggle {
            e.wiggle = w;
        }
        if let Some(c) = &self.curve {
            e.curve = c.clone();
        }
        if let Some(on) = self.perceptual {
            e.perceptual = on;
        }
        k.tidy_ease();
    }
}

/// The ease params of `p` (see the module docs).
pub(crate) fn edit_p(cmd: &str, p: &Value) -> Result<EaseEdit> {
    Ok(EaseEdit {
        spring: removable(p, "spring", |v| spring_p(cmd, v))?,
        steps: removable(p, "steps", |v| steps_p(cmd, v))?,
        wiggle: removable(p, "wiggle", |v| wiggle_p(cmd, v))?,
        curve: removable(p, "curve", |v| curve_p(cmd, v))?.map(Option::unwrap_or_default),
        perceptual: match p.get("perceptual") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_bool().ok_or_else(|| bad(cmd, "`perceptual` must be true or false"))?),
        },
    })
}

/// Absent: `None`; `null`: `Some(None)`; a value: parsed.
fn removable<T>(p: &Value, key: &str, f: impl FnOnce(&Value) -> Result<T>) -> Result<Option<Option<T>>> {
    match p.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(v) => f(v).map(|x| Some(Some(x))),
    }
}

/// The mode an `interpolation` name says, with the count of a `steps(n)` shorthand. An unknown
/// name is Linear, as it always was for this command.
pub(crate) fn mode_p(cmd: &str, name: &str) -> Result<(Interpolation, Option<u32>)> {
    let n = name.trim();
    if let Some(inner) = n.strip_prefix("steps(").and_then(|r| r.strip_suffix(')')) {
        let count = inner.trim().parse::<f64>().map_err(|_| bad(cmd, format!("`{}`: steps(n) needs a whole number", clip(n))))?;
        return Ok((Interpolation::Steps, Some(steps_p(cmd, &json!(count))?)));
    }
    Ok((Interpolation::from_name(n).unwrap_or_default(), None))
}

/// A finite number of object `o` at `key` (also read under its snake_case spelling), clamped to
/// `range`; an error when it is there but is not a finite number.
fn num(cmd: &str, o: &Map<String, Value>, what: &str, key: &str, range: (f64, f64)) -> Result<Option<f64>> {
    let snake: String = key.chars().flat_map(|c| if c.is_ascii_uppercase() { vec!['_', c.to_ascii_lowercase()] } else { vec![c] }).collect();
    match o.get(key).or_else(|| o.get(&snake)) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite())
            .map(|x| Some(x.clamp(range.0, range.1)))
            .ok_or_else(|| bad(cmd, format!("`{what}.{key}` must be a finite number"))),
    }
}

/// Refuse keys an object does not have (a typo would otherwise be ignored).
fn known_keys(cmd: &str, what: &str, o: &Map<String, Value>, keys: &[&str]) -> Result<()> {
    for k in o.keys() {
        let camel: String = {
            let mut up = false;
            k.chars()
                .filter_map(|c| {
                    if c == '_' {
                        up = true;
                        None
                    } else if up {
                        up = false;
                        Some(c.to_ascii_uppercase())
                    } else {
                        Some(c)
                    }
                })
                .collect()
        };
        if !keys.contains(&camel.as_str()) {
            let hint = if what == "spring" && k == "damping" { " (the damping ratio is `dampingRatio`)" } else { "" };
            return Err(bad(cmd, format!("`{what}` has no `{}`{hint}; it takes {}", clip(k), keys.join(", "))));
        }
    }
    Ok(())
}

/// A spring: an object (see the module docs) or `true` for the default.
pub(crate) fn spring_p(cmd: &str, v: &Value) -> Result<Spring> {
    if v.as_bool() == Some(true) {
        return Ok(Spring::DEFAULT);
    }
    let o = v.as_object().ok_or_else(|| bad(cmd, "`spring` is {dampingRatio, stiffness, mass, maxOvershoot} or true"))?;
    known_keys(cmd, "spring", o, &["dampingRatio", "stiffness", "mass", "maxOvershoot"])?;
    let d = Spring::DEFAULT;
    Ok(Spring {
        damping_ratio: num(cmd, o, "spring", "dampingRatio", SPRING_DAMPING_RATIO)?.unwrap_or(d.damping_ratio),
        stiffness: num(cmd, o, "spring", "stiffness", SPRING_STIFFNESS)?.unwrap_or(d.stiffness),
        mass: num(cmd, o, "spring", "mass", SPRING_MASS)?.unwrap_or(d.mass),
        max_overshoot: num(cmd, o, "spring", "maxOvershoot", SPRING_MAX_OVERSHOOT)?.unwrap_or(d.max_overshoot),
    })
}

/// A step count: a whole number from 1 (more than [`MAX_STEPS`] is that many).
pub(crate) fn steps_p(cmd: &str, v: &Value) -> Result<u32> {
    let n = v
        .as_f64()
        .filter(|x| x.is_finite() && x.fract() == 0.0 && *x >= 1.0)
        .ok_or_else(|| bad(cmd, format!("`steps` is a whole number from 1 to {MAX_STEPS}")))?;
    Ok(n.min(f64::from(MAX_STEPS)) as u32)
}

/// A wiggle object (see the module docs).
pub(crate) fn wiggle_p(cmd: &str, v: &Value) -> Result<Wiggle> {
    let o = v.as_object().ok_or_else(|| bad(cmd, "`wiggle` is {amplitude, frequency, seed}"))?;
    known_keys(cmd, "wiggle", o, &["amplitude", "frequency", "seed"])?;
    let d = Wiggle::DEFAULT;
    let seed = match o.get("seed") {
        None | Some(Value::Null) => d.seed,
        Some(s) => s
            .as_u64()
            .or_else(|| s.as_f64().filter(|x| x.is_finite() && x.fract() == 0.0 && (0.0..9.007_199_254_740_992e15).contains(x)).map(|x| x as u64))
            .ok_or_else(|| bad(cmd, "`wiggle.seed` is a whole number from 0"))?,
    };
    Ok(Wiggle {
        amplitude: num(cmd, o, "wiggle", "amplitude", (0.0, MAX_WIGGLE_AMPLITUDE))?.unwrap_or(d.amplitude),
        frequency: num(cmd, o, "wiggle", "frequency", (0.0, MAX_WIGGLE_FREQUENCY))?.unwrap_or(d.frequency),
        seed,
    })
}

/// Custom curve points: `[[time, progress], …]`, at most [`MAX_CURVE_POINTS`], time strictly
/// increasing inside (0, 1).
pub(crate) fn curve_p(cmd: &str, v: &Value) -> Result<Vec<[f64; 2]>> {
    let a = v.as_array().ok_or_else(|| bad(cmd, "`curve` is a list of [time, progress] points"))?;
    if a.len() > MAX_CURVE_POINTS {
        return Err(bad(cmd, format!("`curve` has {} points; at most {MAX_CURVE_POINTS}", a.len())));
    }
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(a.len());
    for (i, pt) in a.iter().enumerate() {
        let xy = pt.as_array().filter(|p| p.len() == 2).and_then(|p| Some([p.first()?.as_f64()?, p.get(1)?.as_f64()?]));
        let [x, y] =
            xy.filter(|[x, y]| x.is_finite() && y.is_finite()).ok_or_else(|| bad(cmd, format!("`curve[{i}]` is [time, progress], two finite numbers")))?;
        if !(x > 0.0 && x < 1.0) {
            return Err(bad(cmd, format!("`curve[{i}]`: time {x} is not inside (0, 1); the ends (0, 0) and (1, 1) are implied")));
        }
        if out.last().is_some_and(|l| x <= l[0]) {
            return Err(bad(cmd, format!("`curve[{i}]`: times must increase (monotonic in time)")));
        }
        out.push([x, y.clamp(CURVE_RANGE.0, CURVE_RANGE.1)]);
    }
    Ok(out)
}

pub(crate) fn spring_json(s: Spring) -> Value {
    json!({"dampingRatio": s.damping_ratio, "stiffness": s.stiffness, "mass": s.mass, "maxOvershoot": s.max_overshoot})
}

/// A keyframe's leaving segment as commands answer it: the mode and the numbers it has.
pub(crate) fn segment_json(k: &Keyframe) -> Value {
    let e = k.segment();
    let mut o = json!({"interpolation": k.interp.name(), "perceptual": e.perceptual});
    if let Some(s) = e.spring {
        o["spring"] = spring_json(s);
    }
    if let Some(n) = e.steps {
        o["steps"] = json!(n);
    }
    if let Some(w) = e.wiggle {
        o["wiggle"] = json!({"amplitude": w.amplitude, "frequency": w.frequency, "seed": w.seed});
    }
    if !e.curve.is_empty() {
        o["curve"] = json!(e.curve);
    }
    o
}

/// At most 40 characters of a caller's text, for an error message.
fn clip(s: &str) -> String {
    s.chars().take(40).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ease_params_are_validated_and_clamped() {
        let c = "test";
        let e = edit_p(
            c,
            &json!({"spring": {"dampingRatio": 99, "stiffness": -5, "mass": 1e300, "maxOvershoot": 0.1}, "steps": 5000, "wiggle": {"amplitude": -3, "seed": 7}, "curve": [[0.5, 9]], "perceptual": true}),
        )
        .unwrap();
        assert_eq!(e.spring, Some(Some(Spring { damping_ratio: 10.0, stiffness: 0.1, mass: 100.0, max_overshoot: 0.1 })));
        assert_eq!(e.steps, Some(Some(MAX_STEPS)));
        assert_eq!(e.wiggle, Some(Some(Wiggle { amplitude: 0.0, seed: 7, ..Wiggle::DEFAULT })));
        assert_eq!(e.curve, Some(vec![[0.5, 2.0]]));
        assert_eq!(e.perceptual, Some(true));
        // snake_case and `true` work; null removes
        assert_eq!(spring_p(c, &json!({"damping_ratio": 1.0})).unwrap().damping_ratio, 1.0);
        assert_eq!(spring_p(c, &json!(true)).unwrap(), Spring::DEFAULT);
        let r = edit_p(c, &json!({"spring": null, "curve": null})).unwrap();
        assert_eq!((r.spring, r.curve), (Some(None), Some(vec![])));
        assert!(edit_p(c, &json!({})).unwrap().is_empty());
        for bad_params in [
            json!({"spring": {"damping": 0.5}}),
            json!({"spring": {"dampingRatio": "soft"}}),
            json!({"spring": {"stiffness": null, "mass": [1]}}),
            json!({"spring": 3}),
            json!({"steps": 0}),
            json!({"steps": -2}),
            json!({"steps": 2.5}),
            json!({"steps": "many"}),
            json!({"wiggle": {"amplitude": 1, "speed": 3}}),
            json!({"wiggle": {"seed": -1}}),
            json!({"wiggle": {"seed": 1.5}}),
            json!({"wiggle": 4}),
            json!({"curve": [[0.5, 0.5], [0.4, 0.6]]}),
            json!({"curve": [[0.5, 0.5], [0.5, 0.6]]}),
            json!({"curve": [[0.0, 0.5]]}),
            json!({"curve": [[1.0, 0.5]]}),
            json!({"curve": [[0.5]]}),
            json!({"curve": [[0.5, 0.5, 0.5]]}),
            json!({"curve": [["a", 0.5]]}),
            json!({"curve": {"x": 1}}),
            json!({"curve": (1..=65).map(|i| json!([f64::from(i) / 100.0, 0.5])).collect::<Vec<_>>()}),
            json!({"perceptual": "yes"}),
        ] {
            assert!(edit_p(c, &bad_params).is_err(), "{bad_params}");
        }
        let msg = edit_p(c, &json!({"spring": {"damping": 0.5}})).unwrap_err().to_string();
        assert!(msg.contains("dampingRatio"), "{msg}");
    }

    #[test]
    fn mode_names_and_the_steps_shorthand() {
        assert_eq!(mode_p("t", "spring").unwrap(), (Interpolation::Spring, None));
        assert_eq!(mode_p("t", "Custom").unwrap(), (Interpolation::Custom, None));
        assert_eq!(mode_p("t", "steps(6)").unwrap(), (Interpolation::Steps, Some(6)));
        assert_eq!(mode_p("t", "steps(1e9)").unwrap(), (Interpolation::Steps, Some(MAX_STEPS)));
        assert!(mode_p("t", "steps(0)").is_err() && mode_p("t", "steps(x)").is_err() && mode_p("t", "steps(NaN)").is_err());
        // unknown names stay Linear, as before
        assert_eq!(mode_p("t", "wobbly").unwrap(), (Interpolation::Linear, None));
    }

    use crate::Session;
    use filmcraft_project::{ClipId, Param};
    use filmcraft_time::Tick;

    /// The demo project with Scale keyframed 100 → 300 % over the middle second of the second V1
    /// clip (clear of its transitions), the playhead at the first keyframe.
    fn keyed() -> (Session, ClipId, Tick, Tick) {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let it = s.active_sequence().unwrap().video_tracks[0].items[1].clone();
        let one = Tick::from_seconds_f64(1.0);
        let from = it.start + Tick(it.duration.0 / 2) - Tick(one.0 / 2);
        assert!(from > it.start && from + one < it.end());
        s.set_playhead(from);
        for (dt, v) in [(Tick::ZERO, 100.0), (one, 300.0)] {
            s.execute("effects.setParam", json!({"clip": it.id.0, "effect": "motion", "param": "scale", "value": v, "time": (from + dt).0, "keyframe": true}))
                .unwrap();
        }
        (s, it.id, it.effect_time_at(from), it.effect_time_at(from + one))
    }

    fn scale(s: &Session, clip: ClipId) -> Param {
        s.active_sequence().unwrap().find_item(clip).unwrap().1.effect("motion").unwrap().params["scale"].clone()
    }

    #[test]
    fn set_interpolation_writes_the_new_modes_with_their_numbers_in_one_undo_step() {
        let (mut s, clip, t0, t1) = keyed();
        let before = scale(&s, clip);
        let undo = s.history.undo.len();
        let r = s
            .execute(
                "effects.setInterpolation",
                json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "interpolation": "spring", "spring": {"dampingRatio": 1.0, "stiffness": 200}, "perceptual": true}),
            )
            .unwrap();
        assert_eq!(r["interpolation"], "spring");
        assert_eq!(r["spring"]["dampingRatio"], 1.0);
        assert_eq!(r["perceptual"], true);
        assert_eq!(s.history.undo.len(), undo + 1);
        let p = scale(&s, clip);
        let k = &p.keyframes[0];
        assert_eq!(k.interp, Interpolation::Spring);
        assert_eq!(k.segment().spring, Some(Spring { damping_ratio: 1.0, stiffness: 200.0, ..Spring::DEFAULT }));
        // critically damped: rises without passing 300 %, lands on it
        let mid = Tick(t0.0 + (t1.0 - t0.0) / 4);
        assert!(p.f64_at(mid) > 200.0 && p.f64_at(mid) < 300.0, "{}", p.f64_at(mid));
        assert_eq!(p.f64_at(t1), 300.0);
        s.undo();
        assert_eq!(scale(&s, clip), before);

        // the steps(n) shorthand, a custom curve, a wiggle; numbers alone keep the mode
        s.execute("effects.setInterpolation", json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "interpolation": "steps(4)"})).unwrap();
        assert_eq!((scale(&s, clip).keyframes[0].interp, scale(&s, clip).keyframes[0].segment().steps), (Interpolation::Steps, Some(4)));
        s.execute("effects.setInterpolation", json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "steps": 8})).unwrap();
        assert_eq!((scale(&s, clip).keyframes[0].interp, scale(&s, clip).keyframes[0].segment().steps), (Interpolation::Steps, Some(8)));
        s.execute("effects.setInterpolation", json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "interpolation": "custom", "curve": [[0.5, 0.9]]}))
            .unwrap();
        assert!((scale(&s, clip).f64_at(Tick(t0.0 + (t1.0 - t0.0) / 2)) - 280.0).abs() < 1e-6);
        let r = s
            .execute(
                "effects.setInterpolation",
                json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "interpolation": "wiggle", "wiggle": {"amplitude": 5, "frequency": 3, "seed": 11}}),
            )
            .unwrap();
        assert_eq!(r["wiggle"]["seed"], 11);
        // a name alone, as before, sets the mode; no name and no numbers is Linear, as before
        s.execute("effects.setInterpolation", json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0})).unwrap();
        assert_eq!(scale(&s, clip).keyframes[0].interp, Interpolation::Linear);
        // effects.setKeyframe edits the numbers without changing the mode
        let r = s.execute("effects.setKeyframe", json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "perceptual": true})).unwrap();
        assert_eq!((r["interpolation"].as_str(), r["perceptual"].as_bool()), (Some("linear"), Some(true)));
        let p = scale(&s, clip);
        assert!((p.f64_at(Tick(t0.0 + (t1.0 - t0.0) / 2)) - 100.0 * 3f64.sqrt()).abs() < 1e-3, "log-space midpoint");
        s.execute(
            "effects.setKeyframe",
            json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "perceptual": false, "curve": null, "wiggle": null, "steps": null}),
        )
        .unwrap();
        assert!(scale(&s, clip).keyframes[0].ease.is_none(), "an emptied segment is stored as before");
    }

    #[test]
    fn hostile_ease_params_change_nothing() {
        let (mut s, clip, t0, _) = keyed();
        let before = scale(&s, clip);
        let undo = s.history.undo.len();
        for p in [
            json!({"spring": {"dampingRatio": "x"}}),
            json!({"spring": {"damping": 0.4}}),
            json!({"steps": 0}),
            json!({"interpolation": "steps(-3)"}),
            json!({"wiggle": {"seed": -4}}),
            json!({"curve": [[0.9, 0.1], [0.2, 0.5]]}),
            json!({"curve": [[0.5, "far"]]}),
            json!({"perceptual": 1}),
        ] {
            for cmd in ["effects.setInterpolation", "effects.setKeyframe"] {
                let mut q = json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0});
                for (k, v) in p.as_object().unwrap() {
                    q[k] = v.clone();
                }
                assert!(s.execute(cmd, q.clone()).is_err(), "{cmd} {q}");
            }
        }
        assert_eq!(scale(&s, clip), before);
        assert_eq!(s.history.undo.len(), undo);
    }

    /// The renderer evaluates the new modes through the same keyframes: a frame mid-spring looks
    /// like the clip with its scale set, unanimated, to the value the model gives there.
    #[test]
    fn the_render_draws_what_the_spring_says() {
        let (mut s, clip, t0, t1) = keyed();
        s.execute(
            "effects.setInterpolation",
            json!({"clip": clip.0, "param": "scale", "mediaTime": t0.0, "interpolation": "spring", "spring": {"dampingRatio": 0.6}, "perceptual": true}),
        )
        .unwrap();
        let it = s.active_sequence().unwrap().find_item(clip).unwrap().1.clone();
        // on a frame: the renderer evaluates a frame at its start
        let fd = s.sequence_rate().frame_duration();
        let at = s.playhead() + Tick::from_seconds_f64(0.2);
        let at = Tick(at.0 / fd.0 * fd.0);
        let v = scale(&s, clip).f64_at(it.effect_time_at(at));
        let linear = 100.0 + 200.0 * (at - s.playhead()).seconds();
        assert!((v - linear).abs() > 20.0, "the spring differs from linear there: {v}");
        s.set_playhead(at);
        let animated = s.render_program(0.25).unwrap();
        // back to before the keyframes, and the same scale set statically
        while scale(&s, clip).is_animated() {
            s.undo();
        }
        s.execute("effects.setParam", json!({"clip": clip.0, "param": "scale", "value": v})).unwrap();
        s.set_playhead(at);
        let still = s.render_program(0.25).unwrap();
        let diff = |a: &filmcraft_render::Image, b: &filmcraft_render::Image| a.px.iter().zip(&b.px).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(diff(&animated, &still) < 1e-4, "{}", diff(&animated, &still));
        // and the picture does show the scale: a linear ramp's value there looks different
        s.execute("effects.setParam", json!({"clip": clip.0, "param": "scale", "value": linear})).unwrap();
        let other = s.render_program(0.25).unwrap();
        assert!(diff(&animated, &other) > 0.05, "{}", diff(&animated, &other));
        let _ = t1;
    }
}

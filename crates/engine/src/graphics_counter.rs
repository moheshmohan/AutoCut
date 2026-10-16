//! An on-screen counter: one graphic clip whose number steps up at given times ("ROBOT #1" …
//! "ROBOT #12"), with a quick scale "pop" each time it changes — the tally of a countdown,
//! a showcase or a list video.
//!
//! | command | does |
//! |---|---|
//! | `graphics.counter` | a text clip from the first time to `end`, its text hold-keyframed to each count, in a corner of the frame, above the clips there — one undo step |
//!
//! It is an ordinary graphic clip: the text, font, colours and position stay editable in the
//! Essential Graphics panel, and the keyframes on its Text and Scale are visible and editable.

use filmcraft_geom::Vec2;
use filmcraft_project::graphic::new_text_layer;
use filmcraft_project::{Interpolation, ParamValue};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, f64_p, has_seq, str_p, u64_p};
use crate::{EngineError, Result, Session};

const ID: &str = "graphics.counter";
/// Most steps one counter holds.
const MAX_STEPS: usize = 999;
/// How long the pop takes to settle, seconds.
const POP_SECONDS: f64 = 0.25;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Counter",
        // needs `times`: a command for agents and scripts, not a bare menu item
        menu: &[],
        shortcut: None,
        params: r##"{"times":[seconds] (when each count appears, ascending),"end":seconds?,"start":n=1,"prefix":str="","suffix":str="","pad":0..6=0,"corner":"topLeft|topRight|bottomLeft|bottomRight"="topRight","size":px=64,"color":"#rrggbb"="#ffffff","pop":bool=true,"track":index?}"##,
        enabled: has_seq,
        run: counter,
        journal: true,
    }]
}

/// The label of count `n`.
pub fn label(prefix: &str, n: u64, pad: usize, suffix: &str) -> String {
    format!("{prefix}{n:0>pad$}{suffix}")
}

fn color_p(p: &Value, k: &str) -> Result<[f32; 4]> {
    let Some(s) = str_p(p, k) else { return Ok([1.0, 1.0, 1.0, 1.0]) };
    let hex = s.trim().trim_start_matches('#');
    let byte = |i: usize| hex.get(i..i + 2).and_then(|h| u8::from_str_radix(h, 16).ok());
    match (hex.len(), byte(0), byte(2), byte(4)) {
        (6, Some(r), Some(g), Some(b)) => Ok([f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0, 1.0]),
        _ => Err(bad(ID, format!("`{k}` must be #rrggbb, not `{s}`"))),
    }
}

fn counter(s: &mut Session, p: &Value) -> Result<Value> {
    let times: Vec<f64> = p
        .get("times")
        .and_then(Value::as_array)
        .ok_or_else(|| bad(ID, "`times` (seconds when each count appears) is required"))?
        .iter()
        .map(|v| v.as_f64().filter(|x| x.is_finite() && *x >= 0.0))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| bad(ID, "`times` must be seconds, zero or more"))?;
    let (Some(&t0), Some(&last)) = (times.first(), times.last()) else { return Err(bad(ID, "`times` is empty")) };
    if times.len() > MAX_STEPS {
        return Err(bad(ID, format!("at most {MAX_STEPS} counts")));
    }
    if times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(bad(ID, "`times` must go up"));
    }
    let end = f64_p(p, "end").unwrap_or(last + 3.0);
    if !end.is_finite() || end <= last {
        return Err(bad(ID, "`end` must come after the last count"));
    }
    let start = u64_p(p, "start").unwrap_or(1).min(1_000_000);
    let prefix: String = str_p(p, "prefix").unwrap_or("").chars().take(60).collect();
    let suffix: String = str_p(p, "suffix").unwrap_or("").chars().take(60).collect();
    let pad = u64_p(p, "pad").unwrap_or(0).min(6) as usize;
    let size = f64_p(p, "size").unwrap_or(64.0);
    if !size.is_finite() || size <= 0.0 {
        return Err(bad(ID, "`size` must be more than zero"));
    }
    let size = size.clamp(8.0, 1000.0);
    let color = color_p(p, "color")?;
    let pop = bool_p(p, "pop").unwrap_or(true);
    let corner = str_p(p, "corner").unwrap_or("topRight");
    let (w, h, rate) =
        s.active_sequence().map(|q| (f64::from(q.settings.width), f64::from(q.settings.height), q.settings.frame_rate)).ok_or(EngineError::NoSequence)?;
    // counts change on frames: snap every time, so the keyframes line up with the clip
    let snap = |t: f64| rate.snap(Tick::from_seconds_f64(t));
    let margin = (w.min(h) * 0.05).round();
    // point text: the position is the alignment point on the first baseline
    let (x, align) = match corner {
        "topLeft" | "bottomLeft" => (margin, 0u32),
        "topRight" | "bottomRight" => (w - margin, 2u32),
        other => return Err(bad(ID, format!("unknown corner `{other}` (topLeft, topRight, bottomLeft, bottomRight)"))),
    };
    let y = if corner.starts_with("top") { margin + size } else { h - margin - size * 0.25 };
    let labels: Vec<String> = (0..times.len() as u64).map(|i| label(&prefix, start + i, pad, &suffix)).collect();
    let first = labels.first().cloned().unwrap_or_default();
    let mut layer = new_text_layer(&first, Vec2::new(x, y), size);
    let set = |l: &mut filmcraft_project::EffectInstance, id: &str, v: ParamValue| {
        if let Some(prm) = l.params.get_mut(id) {
            prm.value = v;
        } else if let Some(d) = filmcraft_project::find_effect(&l.effect).and_then(|e| e.param(id)) {
            let mut prm = filmcraft_project::Param::new(d.default.clone());
            prm.value = v;
            l.params.insert(id.to_string(), prm);
        }
    };
    set(&mut layer, "font_style", ParamValue::Text("Bold".into()));
    set(&mut layer, "align", ParamValue::Choice(align));
    set(&mut layer, "fill_color", ParamValue::Color(color));
    set(&mut layer, "stroke", ParamValue::Bool(true));
    set(&mut layer, "stroke_width", ParamValue::Float((size * 0.08).max(2.0)));
    set(&mut layer, "shadow", ParamValue::Bool(true));
    set(&mut layer, "shadow_distance", ParamValue::Float(size * 0.06));
    set(&mut layer, "name", ParamValue::Text("Counter".into()));
    // the count: hold keyframes at each time (clip time, from the first count)
    let rel = |t: f64| snap(t) - snap(t0);
    if let Some(prm) = layer.params.get_mut("text") {
        for (t, l) in times.iter().zip(&labels) {
            prm.put_keyframe(rel(*t), ParamValue::Text(l.clone()));
        }
        for k in &mut prm.keyframes {
            k.interp = Interpolation::Hold;
        }
    }
    // the pop: 135 % at each change, settling to 100 % with an ease-out
    if pop && let Some(prm) = layer.params.get_mut("scale") {
        for (i, t) in times.iter().enumerate() {
            let settle = (t + POP_SECONDS).min(times.get(i + 1).copied().unwrap_or(end) - 0.01);
            prm.put_keyframe(rel(*t), ParamValue::Float(135.0));
            if settle > *t {
                prm.put_keyframe(rel(settle), ParamValue::Float(100.0));
            }
        }
        // the pop eases out to 100 %, which then holds until the next change
        for k in &mut prm.keyframes {
            k.interp = if matches!(k.value, ParamValue::Float(v) if v > 100.0) { Interpolation::EaseOut } else { Interpolation::Hold };
        }
    }
    let mut place = json!({"seconds": (snap(end) - snap(t0)).seconds(), "time": snap(t0).0});
    if let Some(t) = u64_p(p, "track") {
        place["track"] = json!(t);
    }
    let clip = crate::graphics::new_graphic_clip(s, ID, layer, "Counter", &place)?;
    Ok(json!({"clip": clip.0, "counts": labels.len(), "labels": labels}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    #[test]
    fn labels_pad_and_wrap() {
        assert_eq!(label("ROBOT #", 7, 0, ""), "ROBOT #7");
        assert_eq!(label("", 7, 2, " / 12"), "07 / 12");
    }

    #[test]
    fn counter_steps_through_the_counts() {
        let mut s = demo();
        let undo0 = s.history.undo.len();
        let r = s.execute(ID, json!({"times": [1.0, 3.0, 6.5], "end": 9.0, "prefix": "ROBOT #", "corner": "topRight"})).unwrap();
        assert_eq!(r["labels"], json!(["ROBOT #1", "ROBOT #2", "ROBOT #3"]));
        assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step");
        let clip = filmcraft_project::ClipId(r["clip"].as_u64().unwrap());
        let q = s.active_sequence().unwrap();
        let (_, it) = q.find_item(clip).unwrap();
        let fd = q.settings.frame_rate.frame_duration();
        assert!((it.start - Tick::from_seconds_f64(1.0)).0.abs() <= fd.0, "{:?}", it.start);
        assert!((it.duration - Tick::from_seconds_f64(8.0)).0.abs() <= fd.0, "{:?}", it.duration);
        let layer = it.effects.iter().find(|e| e.effect == "graphic_text").unwrap();
        let text = layer.params.get("text").unwrap();
        // the number holds between changes
        assert_eq!(text.value_at(Tick::from_seconds_f64(1.5)), ParamValue::Text("ROBOT #1".into()));
        assert_eq!(text.value_at(Tick::from_seconds_f64(2.4)), ParamValue::Text("ROBOT #2".into()));
        assert_eq!(text.value_at(Tick::from_seconds_f64(7.0)), ParamValue::Text("ROBOT #3".into()));
        // and pops at each change
        let scale = layer.params.get("scale").unwrap();
        // (2.05: the counts are snapped to frames, so #2 starts at frame 72 ≈ 2.002 s)
        assert!(matches!(scale.value_at(Tick::from_seconds_f64(2.05)), ParamValue::Float(v) if v > 110.0));
        let settled = scale.value_at(Tick::from_seconds_f64(2.5));
        assert!(matches!(settled, ParamValue::Float(v) if (v - 100.0).abs() < 1e-6), "{settled:?} {:?}", scale.keyframes);
    }

    #[test]
    fn bad_requests_change_nothing() {
        let mut s = demo();
        let undo0 = s.history.undo.len();
        for p in [
            json!({}),
            json!({"times": []}),
            json!({"times": [2.0, 1.0]}),
            json!({"times": [1.0, 1.0]}),
            json!({"times": [-1.0]}),
            json!({"times": [1.0], "end": 0.5}),
            json!({"times": [1.0], "corner": "middle"}),
            json!({"times": [1.0], "color": "red"}),
            json!({"times": [1.0], "size": 0}),
        ] {
            assert!(s.execute(ID, p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.history.undo.len(), undo0);
    }
}

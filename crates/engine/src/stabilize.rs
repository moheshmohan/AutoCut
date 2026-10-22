//! Stabiliser levels: `clip.stabilize` (Clip ▸ Stabilize…) puts Warp Stabilizer on video clips
//! at one of [`filmcraft_project::effect::STABILIZER_LEVELS`], and choosing a level in Effect
//! Controls (`effects.setParam` on `level`) writes its settings, as Ultra Key's Setting does.
//!
//! | level | smoothing | crop |
//! |---|---|---|
//! | Recommended | the defaults: Smoothness 50 %, Position, Scale, Rotation | auto-scale up to 150 % |
//! | Minimum Crop | Smoothness 20 %, Position only | auto-scale up to 112 % |
//! | Most Stable | Smoothness 120 %, Detailed Analysis | auto-scale up to 200 % |
//!
//! Changing one of those settings by hand afterwards turns the level to Custom.

use serde_json::{Value, json};

use filmcraft_project::effect::{STABILIZER_LEVELS, stabilizer_level};
use filmcraft_project::{EffectInstance, Param, ParamValue, TrackKind};

use crate::commands::{CommandSpec, bad, clips_p, has_selection, str_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.stabilize";

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Stabilize…",
        menu: &["Clip"],
        shortcut: None,
        params: r#"{"clips":[id]?,"level":"recommended|minimumCrop|mostStable"=recommended}"#,
        enabled: has_selection,
        run: stabilize,
        journal: true,
    }]
}

/// The level index named `name` (`recommended`, `Minimum Crop`, `most_stable`, or an index 1–3).
pub fn level_index(name: &str) -> Option<u32> {
    let squash = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_ascii_lowercase();
    let n = squash(name);
    if let Ok(i) = n.parse::<u32>() {
        return (1..STABILIZER_LEVELS.len() as u32).contains(&i).then_some(i);
    }
    STABILIZER_LEVELS.iter().position(|l| squash(l) == n).map(|i| i as u32).filter(|i| *i > 0)
}

/// Write level `level`'s settings into Warp Stabilizer instance `e` (Custom: nothing).
pub fn apply_level(e: &mut EffectInstance, level: u32) {
    let Some(rows) = stabilizer_level(level) else { return };
    for (id, v) in rows {
        match e.param_mut(id) {
            Some(p) => {
                p.value = v.clone();
                p.keyframes.clear();
            }
            None => {
                e.params.insert((*id).to_string(), Param::new(v.clone()));
            }
        }
    }
    e.params.insert("level".into(), Param::new(ParamValue::Choice(level)));
}

/// After `effects.setParam` set `pid` on Warp Stabilizer `e`: a level writes its settings; one of
/// the settings a level controls, changed by hand, makes the level Custom.
pub fn after_set_param(e: &mut EffectInstance, pid: &str, v: &ParamValue) {
    if e.effect != "warp_stabilizer" {
        return;
    }
    if pid == "level" {
        if let ParamValue::Choice(l) = v {
            apply_level(e, *l);
        }
        return;
    }
    let controlled = stabilizer_level(1).is_some_and(|rows| rows.iter().any(|(id, _)| *id == pid));
    if controlled && let Some(p) = e.param_mut("level") {
        p.value = ParamValue::Choice(0);
    }
}

fn stabilize(s: &mut Session, p: &Value) -> Result<Value> {
    let level = match str_p(p, "level") {
        Some(n) => level_index(n).ok_or_else(|| bad(CMD, format!("unknown level `{n}` (recommended, minimumCrop, mostStable)")))?,
        None if p.get("level").is_some_and(|v| !v.is_null()) => return Err(bad(CMD, "`level` is recommended, minimumCrop or mostStable")),
        None => 1,
    };
    let def = filmcraft_project::find_effect("warp_stabilizer").ok_or_else(|| bad(CMD, "no Warp Stabilizer definition"))?;
    let clips = clips_p(s, p);
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let video: Vec<_> =
        clips.iter().copied().filter(|c| seq.find_item(*c).and_then(|(t, _)| seq.track(t)).is_some_and(|t| t.kind == TrackKind::Video && !t.locked)).collect();
    if video.is_empty() {
        return Err(bad(CMD, "no unlocked video clip given or selected"));
    }
    let label = format!("Stabilize: {}", STABILIZER_LEVELS.get(level as usize).copied().unwrap_or("Recommended"));
    let report = s.edit_sequence(&label, |q, _, _| {
        let mut out = Vec::new();
        for c in &video {
            let (_, it) = q.find_item_mut(*c).ok_or(filmcraft_edit::EditError::NoItem(*c))?;
            let added = !it.effects.iter().any(|e| e.effect == "warp_stabilizer");
            if added {
                it.effects.push(def.instance());
            }
            if let Some(e) = it.effects.iter_mut().find(|e| e.effect == "warp_stabilizer") {
                e.enabled = true;
                apply_level(e, level);
            }
            out.push(json!({"clip": c.0, "added": added}));
        }
        Ok(out)
    })?;
    Ok(json!({"level": STABILIZER_LEVELS.get(level as usize), "clips": report}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::ClipId;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.state.selection.clear();
        s
    }

    fn warp(s: &Session, c: ClipId) -> Option<EffectInstance> {
        s.active_sequence().unwrap().find_item(c).and_then(|(_, i)| i.effects.iter().find(|e| e.effect == "warp_stabilizer").cloned())
    }

    fn f(e: &EffectInstance, k: &str) -> f64 {
        e.param(k).and_then(|p| p.value.as_f64()).unwrap()
    }

    #[test]
    fn levels_write_their_settings_and_hand_edits_make_them_custom() {
        let mut s = demo();
        let c = s.active_sequence().unwrap().video_tracks[0].items[0].id;
        s.execute(CMD, json!({"clips": [c.0], "level": "mostStable"})).unwrap();
        let e = warp(&s, c).expect("added");
        assert_eq!(e.param("level").map(|p| p.value.clone()), Some(ParamValue::Choice(3)));
        assert_eq!(f(&e, "smoothness"), 120.0);
        assert_eq!(f(&e, "max_scale"), 200.0);
        // Minimum Crop crops less than Most Stable: a lower ceiling and less smoothing
        s.execute(CMD, json!({"clips": [c.0], "level": "Minimum Crop"})).unwrap();
        let e = warp(&s, c).unwrap();
        assert!(f(&e, "max_scale") < 150.0 && f(&e, "smoothness") < 50.0);
        assert_eq!(s.active_sequence().unwrap().find_item(c).unwrap().1.effects.iter().filter(|e| e.effect == "warp_stabilizer").count(), 1, "one stabilizer");
        // Effect Controls: choosing a level writes it; a hand edit makes it Custom
        s.execute("effects.setParam", json!({"clip": c.0, "effect": "warp_stabilizer", "param": "level", "value": "Recommended"})).unwrap();
        let e = warp(&s, c).unwrap();
        assert_eq!(f(&e, "smoothness"), 50.0);
        s.execute("effects.setParam", json!({"clip": c.0, "effect": "warp_stabilizer", "param": "smoothness", "value": 80.0})).unwrap();
        assert_eq!(warp(&s, c).unwrap().param("level").map(|p| p.value.clone()), Some(ParamValue::Choice(0)));
        // undo: one step each
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        assert!(warp(&s, c).is_none(), "the first call added it; undone");
    }

    #[test]
    fn bad_levels_and_targets_are_refused() {
        let mut s = demo();
        let c = s.active_sequence().unwrap().video_tracks[0].items[0].id;
        for p in [
            json!({"clips": [c.0], "level": "wobbly"}),
            json!({"clips": [c.0], "level": 7}),
            json!({"clips": [c.0], "level": "0"}),
            json!({"clips": [987_654_321u64]}),
        ] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        let a = s.active_sequence().unwrap().audio_tracks[0].items[0].id;
        assert!(s.execute(CMD, json!({"clips": [a.0]})).is_err(), "sound is not stabilised");
        assert_eq!(level_index("most_stable"), Some(3));
        assert_eq!(level_index("custom"), None);
    }
}

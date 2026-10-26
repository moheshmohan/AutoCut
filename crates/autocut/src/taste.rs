//! The taste profile in AutoCut's pipelines (`autocut.taste`, [`Taste`]): the user's default and
//! the project's laid together, read once per job and laid under the job's own options — a
//! video's options win, the profile fills what they leave out, and what neither says is the
//! pipeline's own default.

use filmcraft_engine::project::taste::Taste;
use serde_json::{Value, json};

use crate::Executor;

/// The profile the job starts from (none when it cannot be read).
pub async fn resolve(ex: &dyn Executor) -> Taste {
    match ex.execute("autocut.taste", json!({})).await {
        Ok(v) => serde_json::from_value::<Taste>(v["resolved"].clone()).map(|t| t.sanitized()).unwrap_or_default(),
        Err(_) => Taste::default(),
    }
}

/// The director's options a profile sets (`style`, `lightEffects`, `sounds`, `soundLevel`,
/// `color`, `colorStrength`, `look`, `lookStrength`, `redact`).
pub fn director_options(t: &Taste) -> Value {
    let mut o = json!({});
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            o[k] = v;
        }
    };
    put("style", t.style.clone().map(Value::String));
    put("lightEffects", t.light_effects.map(Value::Bool));
    put("sounds", t.sounds.clone().map(|s| json!(s)));
    put("soundLevel", t.sound_level.map(|v| json!(v)));
    put("color", t.color.map(Value::Bool));
    put("colorStrength", t.color_strength.map(|v| json!(v)));
    put("look", t.look.clone().map(Value::String));
    put("lookStrength", t.look_strength.map(|v| json!(v)));
    put("redact", t.redact.clone().map(Value::String));
    o
}

/// `p` with `defaults` under it: each key `p` leaves out (or null) taken from `defaults`.
pub fn under(p: &Value, defaults: &Value) -> Value {
    let mut out = p.clone();
    if let (Some(m), Some(d)) = (out.as_object_mut(), defaults.as_object()) {
        for (k, v) in d {
            if m.get(k).is_none_or(Value::is_null) {
                m.insert(k.clone(), v.clone());
            }
        }
    }
    out
}

/// Whether a transition is a light effect (a leak, a sweep, a flare, a flash, a burn).
pub fn is_light_effect(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    id.starts_with("lf_") || ["light", "leak", "flare", "flash", "burn", "glow", "ray", "prism", "bokeh", "sweep"].iter().any(|w| id.contains(w))
}

/// `ids` without the light effects when the profile turns them off (an emptied list: `instead`).
pub fn light_free(t: &Taste, ids: &[String], instead: &str) -> Vec<String> {
    if t.light_effects != Some(false) {
        return ids.to_vec();
    }
    let kept: Vec<String> = ids.iter().filter(|i| !is_light_effect(i)).cloned().collect();
    if kept.is_empty() && !ids.is_empty() { vec![instead.to_string()] } else { kept }
}

/// Whether the profile lets a sound be heard (no list: every sound).
pub fn allows_sound(t: &Taste, kind: &str) -> bool {
    t.sounds.as_ref().is_none_or(|s| s.iter().any(|k| k.eq_ignore_ascii_case(kind)))
}

/// A writer's system prompt with the profile's wording and narration rules after it.
pub fn with_rules(system: &str, t: &Taste) -> String {
    let rules = t.writing_rules();
    if rules.is_empty() { system.to_string() } else { format!("{system} The person's own rules, which come first: {}", rules.join(" ")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_engine::project::taste::Wording;

    #[test]
    fn a_profile_fills_what_a_video_leaves_out() {
        let t = Taste { style: Some("cinematic".into()), sounds: Some(vec!["chime".into()]), redact: Some("off".into()), ..Taste::default() };
        let p = under(&json!({"style": "energetic", "seconds": 20, "redact": null}), &director_options(&t));
        assert_eq!((p["style"].as_str(), p["seconds"].as_f64(), p["redact"].as_str()), (Some("energetic"), Some(20.0), Some("off")));
        assert_eq!(p["sounds"], json!(["chime"]));
        assert!(allows_sound(&t, "chime") && !allows_sound(&t, "swish") && allows_sound(&Taste::default(), "whoosh"));
    }

    #[test]
    fn light_effects_are_left_out_when_the_profile_says_so() {
        let ids: Vec<String> = ["cam_punch_in", "lf_light_leak", "gl_filmburn", "flash", "cross_dissolve"].iter().map(|s| s.to_string()).collect();
        let off = Taste { light_effects: Some(false), ..Taste::default() };
        assert_eq!(light_free(&off, &ids, "dip_to_black"), ["cam_punch_in", "cross_dissolve"]);
        assert_eq!(light_free(&off, &["lf_light_leak".to_string()], "dip_to_black"), ["dip_to_black"]);
        assert_eq!(light_free(&Taste::default(), &ids, "x"), ids);
        let rules = Taste {
            words: Some(vec![Wording { say: "our son".into(), not: "my son".into() }]),
            narration: Some(vec!["Tell it with feeling.".into()]),
            ..Taste::default()
        };
        assert!(with_rules("Write.", &rules).ends_with("Say \"our son\", never \"my son\". Tell it with feeling."));
        assert_eq!(with_rules("Write.", &Taste::default()), "Write.");
    }
}

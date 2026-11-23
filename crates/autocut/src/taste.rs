//! The taste profile in AutoCut's pipelines (`autocut.taste`, [`Taste`]): the user's default and
//! the project's laid together, read once per job and laid under the job's own options — a
//! video's options win, the profile fills what they leave out, and what neither says is the
//! pipeline's own default.
//!
//! **Taste memory (PX10).** What the person's feedback retired (`autocut.feedback`) the
//! pipelines' own defaults skip ([`first_kept`]; a retired sound is never heard:
//! [`allows_sound`]). With `guidance` on (the default), a choice the video and the profile both
//! leave open is dealt against the taste history ([`Deals::deal`], `autocut.deal`) so
//! consecutive videos differ, and the finished video's fingerprint is recorded ([`record`],
//! `autocut.fingerprint`); `guidance: false` keeps each pipeline's own defaults and records
//! nothing.

use std::collections::BTreeMap;

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

/// Whether the profile lets a sound be heard (no list: every sound) and the person's feedback
/// did not retire it.
pub fn allows_sound(t: &Taste, kind: &str) -> bool {
    t.sounds.as_ref().is_none_or(|s| s.iter().any(|k| k.eq_ignore_ascii_case(kind))) && !t.is_retired("sound", kind)
}

/// The first of `ids` (a pipeline's own default first) the person's feedback did not retire;
/// the first when every one is.
pub fn first_kept<'a>(t: &Taste, kind: &str, ids: &[&'a str]) -> &'a str {
    ids.iter().copied().find(|id| !t.is_retired(kind, id)).or_else(|| ids.first().copied()).unwrap_or_default()
}

/// Whether a job deals open choices and records its fingerprint (`guidance`, on unless false).
pub fn guided(p: &Value) -> bool {
    p["guidance"].as_bool() != Some(false)
}

/// The choices one video dealt: kind → id, and each deal's answer for the report.
#[derive(Clone, Debug, Default)]
pub struct Deals {
    pub chosen: BTreeMap<String, String>,
    pub answers: serde_json::Map<String, Value>,
}

impl Deals {
    /// Deal `kind` among `options` (the pipeline's own default first) through `autocut.deal`
    /// against the taste history; None when nothing was dealt (every option retired or ruled out
    /// by the profile, or an engine without the command): the pipeline keeps its default.
    pub async fn deal(&mut self, ex: &dyn crate::Executor, kind: &str, options: &[&str], pipeline: &str) -> Option<String> {
        let r = ex.execute("autocut.deal", json!({"for": kind, "options": options, "pipeline": pipeline, "dryRun": true})).await;
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                self.answers.insert(kind.to_string(), json!({"choice": null, "why": format!("not dealt: {e}")}));
                return None;
            }
        };
        self.answers.insert(kind.to_string(), json!({"choice": r["choice"], "why": r["why"], "from": r["from"], "skipped": r["skipped"]}));
        let choice = r["choice"].as_str()?.to_string();
        self.chosen.insert(kind.to_string(), choice.clone());
        Some(choice)
    }

    /// The deals for a report (none: nothing was dealt).
    pub fn json(&self) -> Option<Value> {
        (!self.answers.is_empty()).then(|| Value::Object(self.answers.clone()))
    }
}

/// Record the finished `sequence`'s fingerprint in the taste history with the choices `deals`
/// made, so the next video can differ: the fingerprint (axes, evidence, how it matches the last
/// videos, warnings) for the report, or why it was not recorded.
pub async fn record(ex: &dyn crate::Executor, sequence: Option<u64>, pipeline: &str, deals: &Deals) -> Value {
    let mut p = json!({"record": true, "pipeline": pipeline, "deals": deals.chosen});
    if let Some(s) = sequence {
        p["sequence"] = json!(s);
    }
    match ex.execute("autocut.fingerprint", p).await {
        Ok(v) => json!({"axes": v["axes"], "evidence": v["evidence"], "matches": v["matches"], "warnings": v["warnings"], "recorded": v["recorded"]}),
        Err(e) => json!({"recorded": false, "error": e}),
    }
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
    fn feedback_retires_the_pipelines_defaults() {
        let t = Taste::default().with_feedback("sound", "whoosh", false, None).with_feedback("look", "none", false, None);
        assert!(!allows_sound(&t, "whoosh") && allows_sound(&t, "swish"));
        assert_eq!(first_kept(&t, "look", &["none", "punchy"]), "punchy");
        assert_eq!(first_kept(&t, "titleStyle", &["fade", "pop"]), "fade");
        assert_eq!(first_kept(&Taste::default().with_feedback("x", "a", false, None), "x", &["a"]), "a", "every one retired: the default");
        assert!(guided(&json!({})) && guided(&json!({"guidance": null})) && !guided(&json!({"guidance": false})));
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

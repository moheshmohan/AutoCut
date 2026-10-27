//! The taste profile (`autocut.taste`): how this person likes their videos made
//! ([`filmcraft_project::taste::Taste`]) — the user's default (`taste.json` in the data
//! directory) and the project's own (kept in the project file), the project's over the default.
//! The director, Quick Short, the showcase and the Malayalam script writer read the two laid
//! together; a video's own options win over both.
//!
//! With nothing to change it says what the profiles are: `default`, `project` and `resolved`.
//! `set` changes fields of the `scope`'s profile (`project` by default — one undo step — or
//! `default`), a `null` field removed; `clear` removes the scope's profile.

use filmcraft_project::taste::{Taste, Wording};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{Result, Session};

pub const TASTE_CMD: &str = "autocut.taste";
/// The default profile's file in the data directory.
pub const FILE: &str = "taste.json";
/// Largest default profile read, bytes.
const MAX_BYTES: u64 = 256 * 1024;

/// What the fields may be.
const STYLES: &[&str] = &["subtle", "energetic", "cinematic"];
const SOUNDS: &[&str] = &["swish", "chime", "whoosh", "impact", "riser", "shutter", "pop", "thump"];
const LOOKS: &[&str] = &["none", "punchy", "soft", "warm", "cool", "golden", "teal"];
const REDACT: &[&str] = &["off", "faces", "plates", "faces,plates"];

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: TASTE_CMD,
        label: "Taste Profile",
        menu: &[],
        shortcut: None,
        params: r#"{"scope":"project"|"default"="project","set":{"style":"subtle"|"energetic"|"cinematic"?,"lightEffects":bool?,"sounds":["swish"|"chime"|"whoosh"|"impact"|"riser"|"shutter"|"pop"|"thump"]?,"soundLevel":0..1?,"color":bool?,"colorStrength":0..1?,"look":"none"|"punchy"|"soft"|"warm"|"cool"|"golden"|"teal"?,"lookStrength":0..100?,"textLanguage":str?,"words":[{"say":str,"not":str}]?,"narration":[str]?,"redact":"off"|"faces"|"plates"|"faces,plates"?}?,"clear":bool=false}"#,
        enabled: always,
        run: taste,
        journal: true,
    }]
}

/// The default profile's path: in the data directory beside the preferences.
fn default_path(s: &Session) -> Option<String> {
    let dir = s.prefs_path.as_ref().and_then(|p| p.parent().map(std::path::Path::to_path_buf)).or_else(crate::autosave::default_data_dir)?;
    Some(dir.join(FILE).display().to_string())
}

/// The user's default profile (empty when there is none or it cannot be read).
pub fn default_taste(s: &Session) -> Taste {
    let Some(path) = default_path(s) else { return Taste::default() };
    if s.services.file_size(&path).map_or(true, |n| n > MAX_BYTES) {
        return Taste::default();
    }
    s.services.read_file(&path).ok().and_then(|b| serde_json::from_slice::<Taste>(&b).ok()).map(|t| t.sanitized()).unwrap_or_default()
}

/// The profile a video starts from: the project's over the user's default.
pub fn resolved(s: &Session) -> Taste {
    let project = s.project.taste.clone().unwrap_or_default().sanitized();
    project.over(&default_taste(s))
}

fn one_of(cmd: &str, k: &str, v: &Value, allowed: &[&str]) -> Result<String> {
    let t = v.as_str().map(|t| t.trim().to_ascii_lowercase()).ok_or_else(|| bad(cmd, format!("`set.{k}` is one of {}", allowed.join(", "))))?;
    if allowed.contains(&t.as_str()) { Ok(t) } else { Err(bad(cmd, format!("`set.{k}` is one of {}, not `{t}`", allowed.join(", ")))) }
}

fn number(cmd: &str, k: &str, v: &Value, max: f64) -> Result<f64> {
    v.as_f64().filter(|x| x.is_finite() && (0.0..=max).contains(x)).ok_or_else(|| bad(cmd, format!("`set.{k}` is 0 to {max}")))
}

fn texts(cmd: &str, k: &str, v: &Value) -> Result<Vec<String>> {
    let a = v
        .as_array()
        .filter(|a| a.len() <= filmcraft_project::taste::MAX_RULES)
        .ok_or_else(|| bad(cmd, format!("`set.{k}` is a list of at most {} texts", filmcraft_project::taste::MAX_RULES)))?;
    a.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| bad(cmd, format!("`set.{k}` holds texts")))).collect()
}

/// `t` with the fields of `set` changed (`null` removes one).
fn changed(cmd: &str, mut t: Taste, set: &serde_json::Map<String, Value>) -> Result<Taste> {
    for (k, v) in set {
        let off = v.is_null();
        match k.as_str() {
            "style" => t.style = if off { None } else { Some(one_of(cmd, k, v, STYLES)?) },
            "lightEffects" => t.light_effects = if off { None } else { Some(v.as_bool().ok_or_else(|| bad(cmd, "`set.lightEffects` is true or false"))?) },
            "sounds" => {
                t.sounds = if off {
                    None
                } else {
                    let list = texts(cmd, k, v)?;
                    for s in &list {
                        one_of(cmd, k, &json!(s), SOUNDS)?;
                    }
                    Some(list.iter().map(|s| s.trim().to_ascii_lowercase()).collect())
                }
            }
            "soundLevel" => t.sound_level = if off { None } else { Some(number(cmd, k, v, 1.0)?) },
            "color" => t.color = if off { None } else { Some(v.as_bool().ok_or_else(|| bad(cmd, "`set.color` is true or false"))?) },
            "colorStrength" => t.color_strength = if off { None } else { Some(number(cmd, k, v, 1.0)?) },
            "look" => t.look = if off { None } else { Some(one_of(cmd, k, v, LOOKS)?) },
            "lookStrength" => t.look_strength = if off { None } else { Some(number(cmd, k, v, 100.0)?) },
            "textLanguage" => {
                t.text_language = if off {
                    None
                } else {
                    let l = v
                        .as_str()
                        .map(str::trim)
                        .filter(|l| !l.is_empty() && l.len() <= 16 && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
                        .ok_or_else(|| bad(cmd, "`set.textLanguage` is a language code (en, ml)"))?;
                    Some(l.to_ascii_lowercase())
                }
            }
            "words" => {
                t.words = if off {
                    None
                } else {
                    let a = v
                        .as_array()
                        .filter(|a| a.len() <= filmcraft_project::taste::MAX_RULES)
                        .ok_or_else(|| bad(cmd, "`set.words` is a list of {say, not}"))?;
                    Some(
                        a.iter()
                            .map(|w| {
                                let get = |f: &str| w[f].as_str().map(str::trim).filter(|x| !x.is_empty()).map(str::to_string);
                                Ok(Wording {
                                    say: get("say").ok_or_else(|| bad(cmd, "`set.words`: each has `say`"))?,
                                    not: get("not").ok_or_else(|| bad(cmd, "`set.words`: each has `not`"))?,
                                })
                            })
                            .collect::<Result<_>>()?,
                    )
                }
            }
            "narration" => t.narration = if off { None } else { Some(texts(cmd, k, v)?) },
            "redact" => t.redact = if off { None } else { Some(one_of(cmd, k, v, REDACT)?) },
            other => return Err(bad(cmd, format!("`set` has no field `{other}`"))),
        }
    }
    Ok(t.sanitized())
}

fn answer(s: &Session) -> Value {
    json!({"default": default_taste(s), "project": s.project.taste.clone().unwrap_or_default(), "resolved": resolved(s), "defaultFile": default_path(s)})
}

fn taste(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = TASTE_CMD;
    let scope = str_p(p, "scope").map(|x| x.trim().to_ascii_lowercase()).unwrap_or_else(|| "project".into());
    if !matches!(scope.as_str(), "project" | "default") {
        return Err(bad(cmd, "`scope` is project or default"));
    }
    let clear = bool_p(p, "clear").unwrap_or(false);
    let set = match p.get("set").filter(|v| !v.is_null()) {
        None => None,
        Some(Value::Object(m)) => Some(m.clone()),
        Some(_) => return Err(bad(cmd, "`set` is an object of the profile's fields")),
    };
    if set.is_none() && !clear {
        return Ok(answer(s));
    }
    let base = if clear {
        Taste::default()
    } else if scope == "project" {
        s.project.taste.clone().unwrap_or_default()
    } else {
        default_taste(s)
    };
    let next = match &set {
        Some(m) => changed(cmd, base, m)?,
        None => base,
    };
    if scope == "project" {
        s.edit("Taste Profile", |pr, _| {
            pr.taste = (!next.is_empty()).then(|| next.clone());
            Ok(())
        })?;
    } else {
        let path = default_path(s).ok_or_else(|| bad(cmd, "no data directory for the default profile"))?;
        let bytes = format!("{}\n", serde_json::to_string_pretty(&next).unwrap_or_default());
        s.services.write_file(&path, bytes.as_bytes()).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    }
    Ok(answer(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(tag: &str) -> (Session, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("filmcraft-taste-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() };
        (s, dir)
    }

    #[test]
    fn the_projects_taste_lies_over_the_users_default() {
        let (mut s, dir) = session("layers");
        s.execute(TASTE_CMD, json!({"scope": "default", "set": {"style": "subtle", "sounds": ["swish", "chime"], "redact": "off", "textLanguage": "en"}}))
            .unwrap();
        assert!(dir.join(FILE).exists());
        let r = s
            .execute(TASTE_CMD, json!({"set": {"style": "Cinematic", "lightEffects": true, "words": [{"say": "our son", "not": "my son"}], "narration": ["tell the story with feeling"]}}))
            .unwrap();
        let res = &r["resolved"];
        assert_eq!((res["style"].as_str(), res["lightEffects"].as_bool(), res["redact"].as_str()), (Some("cinematic"), Some(true), Some("off")));
        assert_eq!(res["sounds"], json!(["swish", "chime"]));
        assert_eq!(resolved(&s).apply_words("my son"), "our son");
        // a field removed; undo puts the project's profile back as it was
        s.execute(TASTE_CMD, json!({"set": {"lightEffects": null}})).unwrap();
        assert_eq!(resolved(&s).light_effects, None);
        s.undo().unwrap();
        assert_eq!(resolved(&s).light_effects, Some(true));
        s.execute(TASTE_CMD, json!({"clear": true})).unwrap();
        assert!(s.project.taste.is_none());
        assert_eq!(resolved(&s).style.as_deref(), Some("subtle"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_profiles_are_refused() {
        let (mut s, dir) = session("bad");
        for bad in [
            json!({"scope": "everyone"}),
            json!({"set": "loud"}),
            json!({"set": {"style": "loud"}}),
            json!({"set": {"sounds": ["laser"]}}),
            json!({"set": {"soundLevel": 3}}),
            json!({"set": {"look": "sepia"}}),
            json!({"set": {"words": [{"say": "x"}]}}),
            json!({"set": {"textLanguage": "en gb!"}}),
            json!({"set": {"mood": "happy"}}),
        ] {
            assert!(s.execute(TASTE_CMD, bad.clone()).is_err(), "{bad}");
        }
        // a damaged default file reads as no default
        std::fs::write(dir.join(FILE), b"{not json").unwrap();
        assert!(default_taste(&s).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

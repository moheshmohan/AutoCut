//! `autocut.deal` and `autocut.feedback` (PX10): choices dealt against the taste history so
//! consecutive videos differ, inside what the taste profile allows, and feedback that retires a
//! choice for good (as the whoosh lesson did, through the same profile).
//!
//! **Dealing.** `autocut.deal {for, options?, seed?, dryRun?, pipeline?}` picks one id for `for`
//! (`look`, `titleStyle`, `transitionFamily`, `music`, `register` or `endCard`) from the real
//! catalog ([`catalog`]; `options` narrows it, in the caller's order). A candidate is skipped when
//! the person's feedback `retired` it, when the taste profile rules it out ([`ruled_out`]: a look
//! other than the profile's, colour looks with the colour pass off, a bouncing title entrance
//! unless motion is playful, light transitions with light effects off, a transition family whose
//! sounds the profile allows none of — `sounds` without whoosh or riser leaves out the families
//! that carry them), and when one of the three newest videos in the history used it. Of the rest
//! (liked ones first), the first in order is dealt, or the one `seed` picks; with none left it
//! falls back to the least recently used allowed candidate and says so; with nothing allowed it
//! deals nothing (`choice: null`: the pipeline keeps its own default). The same history and seed
//! deal the same choice. Shorts stay silent: `pipeline: "shorts"` deals no music. Unless
//! `dryRun`, the choice is kept in the session until the next `autocut.fingerprint {record:
//! true}`, which records it with the video.
//!
//! **Feedback.** `autocut.feedback {about, id, verdict, note?, scope?}` writes the verdict into
//! the taste profile (`default`: the user's, by default; `project`: the project's, one undo
//! step): `dislike` adds the id to `retired` (and takes it out of `liked`), `like` the other way
//! round; `note` is kept in `reasons`. `about` is a deal kind, an axis of the fingerprint
//! (`colorLook`, `titleEntrance`, `close`, `captionStyle`, `opening`) or `sound`. A retired id
//! is never dealt and the pipelines' own defaults skip it; a retired sound is never heard.

use filmcraft_project::caption::CaptionPreset;
use filmcraft_project::creator_transitions::CREATOR_TRANSITIONS;
use filmcraft_project::narration::Delivery;
use filmcraft_project::taste::Taste;
use filmcraft_project::text_anim::AnimKind;
use filmcraft_project::transition_sound::SoundKind;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, str_p};
use crate::fingerprint::{FAMILIES, family_of};
use crate::taste_history::{self, History};
use crate::{Result, Session};

pub const DEAL_CMD: &str = "autocut.deal";
pub const FEEDBACK_CMD: &str = "autocut.feedback";
pub use crate::taste_history::DEAL_KINDS as KINDS;
/// The creator looks (adjustment clips over the frame).
pub const CREATOR_LOOKS: [&str; 3] = ["tech", "vhs", "clean"];
/// How a film can end (the fingerprint's `close`).
pub const END_CARDS: [&str; 5] = ["endCard", "logo", "fadeOut", "title", "cut"];
/// How a film can open (the fingerprint's `opening`).
pub const OPENINGS: [&str; 5] = ["coldOpen", "titleCard", "montage", "talking", "still"];
/// What feedback can be about: the deal kinds and what the pipelines choose without dealing.
pub const FEEDBACK_KINDS: [&str; 9] = ["look", "titleStyle", "transitionFamily", "music", "register", "endCard", "sound", "captionStyle", "opening"];
/// Videos back a choice counts as used.
pub const RECENT: usize = 3;
/// Most `options` one call names.
const MAX_OPTIONS: usize = 64;
/// Longest note.
const MAX_NOTE: usize = 300;

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: DEAL_CMD,
            label: "Deal a Style Choice",
            menu: &[],
            shortcut: None,
            params: r#"{"for":"look"|"titleStyle"|"transitionFamily"|"music"|"register"|"endCard","options":[id]? (narrows the catalog, in this order: the first is the default),"seed":u64? (picks among the fresh choices; none: the first),"dryRun":bool=false (do not keep the choice for the next recorded fingerprint),"pipeline":str? (shorts: no music is dealt)}"#,
            enabled: always,
            run: deal_run,
            journal: false,
        },
        CommandSpec {
            id: FEEDBACK_CMD,
            label: "Style Feedback",
            menu: &[],
            shortcut: None,
            params: r#"{"about":"look"|"titleStyle"|"transitionFamily"|"music"|"register"|"endCard"|"sound"|"captionStyle"|"opening" (or an axis: colorLook, titleEntrance, close),"id":str (from that catalog),"verdict":"dislike"|"like","note":str?,"scope":"default"|"project"="default"}"#,
            enabled: always,
            run: feedback_run,
            journal: true,
        },
    ]
}

/// The ids kind `kind` is chosen from, in catalog order (empty for an unknown kind).
pub fn catalog(kind: &str) -> Vec<&'static str> {
    match kind {
        "look" => crate::auto_color::look_ids().chain(CREATOR_LOOKS).collect(),
        "titleStyle" => AnimKind::IN.iter().map(|k| k.name()).chain(["none"]).collect(),
        "transitionFamily" => FAMILIES.iter().copied().chain(["cut"]).collect(),
        "music" => filmcraft_templates::manifest::MOODS.to_vec(),
        "register" => Delivery::ALL.iter().map(|d| d.id()).collect(),
        "endCard" => END_CARDS.to_vec(),
        "sound" => SoundKind::ALL.iter().map(|k| k.id()).collect(),
        "captionStyle" => CaptionPreset::ALL.iter().map(|p| p.name()).chain(["plain"]).collect(),
        "opening" => OPENINGS.to_vec(),
        _ => Vec::new(),
    }
}

/// The catalog's spelling of `id` for `kind` (any case), if it is in it.
pub fn canonical(kind: &str, id: &str) -> Option<&'static str> {
    let id = id.trim();
    catalog(kind).into_iter().find(|c| c.eq_ignore_ascii_case(id))
}

/// The kind a caller names: a feedback kind, or an axis of the fingerprint for its kind.
pub fn kind_of(about: &str) -> Option<&'static str> {
    let a = about.trim().to_ascii_lowercase();
    Some(match a.as_str() {
        "look" | "colorlook" | "colourlook" => "look",
        "titlestyle" | "titleentrance" => "titleStyle",
        "transitionfamily" | "transitions" => "transitionFamily",
        "music" | "mood" => "music",
        "register" | "delivery" => "register",
        "endcard" | "close" => "endCard",
        "sound" | "sounds" => "sound",
        "captionstyle" | "captions" => "captionStyle",
        "opening" => "opening",
        _ => return None,
    })
}

/// Whether the profile lets sound `kind` be heard: on its list (no list: every one) and not
/// retired.
pub fn allows_sound(t: &Taste, kind: &str) -> bool {
    t.sounds.as_ref().is_none_or(|s| s.iter().any(|k| k.eq_ignore_ascii_case(kind))) && !t.is_retired("sound", kind)
}

/// The sounds a family's transitions carry (from the Creator presets made in it; a whip may take
/// the soft swish instead, as the director's whips do), and whether one of them needs none.
pub fn family_sounds(family: &str) -> (bool, Vec<&'static str>) {
    let mut members = 0usize;
    let mut silent = false;
    let mut sounds: Vec<&'static str> = Vec::new();
    for c in CREATOR_TRANSITIONS.iter().filter(|c| family_of(c.id) == family) {
        members = members.saturating_add(1);
        match c.sound {
            None => silent = true,
            Some(k) if !sounds.contains(&k.id()) => sounds.push(k.id()),
            Some(_) => {}
        }
    }
    if family == "whip" && !sounds.contains(&"swish") {
        sounds.push("swish");
    }
    (silent || members == 0, sounds)
}

/// The colour looks (our looks other than `none`, which the colour pass applies).
fn colour_look(id: &str) -> bool {
    id != "none" && crate::auto_color::look_ids().any(|l| l == id)
}

/// Why the taste profile rules `id` out for `kind`, if it does.
pub fn ruled_out(t: &Taste, kind: &str, id: &str) -> Option<String> {
    match kind {
        "look" => {
            if let Some(l) = t.look.as_deref().filter(|l| !l.eq_ignore_ascii_case(id)) {
                return Some(format!("the taste profile's look is {l}"));
            }
            (t.color == Some(false) && colour_look(id)).then(|| "the taste profile turns the colour pass off".to_string())
        }
        "titleStyle" => {
            (id == "drop" && !t.playful_motion()).then(|| "it bounces, and the taste profile's motion is calm (motion: playful allows it)".to_string())
        }
        "transitionFamily" => {
            if id == "light" && t.light_effects == Some(false) {
                return Some("the taste profile turns light effects off".into());
            }
            let (silent, sounds) = family_sounds(id);
            (!silent && !sounds.iter().any(|k| allows_sound(t, k)))
                .then(|| format!("its transitions carry {}, which the taste profile does not allow", sounds.join(" or ")))
        }
        "sound" => (!allows_sound(t, id)).then(|| "the taste profile's sounds leave it out".to_string()),
        _ => None,
    }
}

/// What a deal chose, from what, what it skipped and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Dealt {
    pub choice: Option<String>,
    pub from: Vec<String>,
    pub skipped: Vec<Value>,
    pub why: String,
}

/// Deal `kind` from `candidates` against `history`, inside `taste` (see the module docs).
pub fn deal(taste: &Taste, history: &History, kind: &str, candidates: &[&str], seed: Option<u64>) -> Dealt {
    let recent = history.recent(RECENT);
    let mut skipped = Vec::new();
    let mut allowed: Vec<&str> = Vec::new();
    let mut fresh: Vec<&str> = Vec::new();
    for id in candidates.iter().copied() {
        if taste.is_retired(kind, id) {
            skipped.push(json!({"id": id, "why": "retired"}));
            continue;
        }
        if let Some(r) = ruled_out(taste, kind, id) {
            skipped.push(json!({"id": id, "why": "not allowed by taste", "detail": r}));
            continue;
        }
        allowed.push(id);
        if let Some(n) = recent.iter().position(|r| r.used(kind).is_some_and(|u| u.eq_ignore_ascii_case(id))) {
            skipped.push(json!({"id": id, "why": format!("used in the last {RECENT}"), "videosAgo": n.saturating_add(1)}));
            continue;
        }
        fresh.push(id);
    }
    let from = candidates.iter().map(|c| c.to_string()).collect();
    if allowed.is_empty() {
        let why = "every candidate is retired or not allowed by the taste profile: nothing is dealt (the pipeline keeps its own default)".to_string();
        return Dealt { choice: None, from, skipped, why };
    }
    if !fresh.is_empty() {
        let liked: Vec<&str> = fresh.iter().copied().filter(|id| taste.is_liked(kind, id)).collect();
        let pool = if liked.is_empty() { &fresh } else { &liked };
        let n = pool.len().max(1);
        let i = seed.map_or(0, |s| usize::try_from(s % n as u64).unwrap_or(0));
        let choice = pool.get(i).copied().unwrap_or_default();
        let liked_why = if liked.is_empty() { "" } else { "liked, and " };
        let pick = match seed {
            Some(s) if n > 1 => format!(" (seed {s} picks it of {n})"),
            _ if n > 1 => format!(" (the first of {n} in order)"),
            _ => String::new(),
        };
        let why = format!("{liked_why}not used in the last {RECENT} videos{pick}");
        return Dealt { choice: Some(choice.to_string()), from, skipped, why };
    }
    // every allowed one was used lately: the one used longest ago
    let last_use = |id: &str| history.records.iter().rev().position(|r| r.used(kind).is_some_and(|u| u.eq_ignore_ascii_case(id))).unwrap_or(usize::MAX);
    let mut best: Option<(&str, usize)> = None;
    for id in &allowed {
        let at = last_use(id);
        if best.is_none_or(|(_, b)| at > b) {
            best = Some((id, at));
        }
    }
    let (choice, at) = best.unwrap_or_default();
    let why = format!("every allowed choice was used in the last {RECENT} videos: {choice} is the least recently used ({} videos ago)", at.saturating_add(1));
    Dealt { choice: Some(choice.to_string()), from, skipped, why }
}

fn kind_p(cmd: &str, p: &Value, key: &str, kinds: &[&str]) -> Result<&'static str> {
    let v = str_p(p, key).ok_or_else(|| bad(cmd, format!("`{key}` is one of {}", kinds.join(", "))))?;
    kind_of(v)
        .filter(|k| kinds.contains(k))
        .ok_or_else(|| bad(cmd, format!("`{key}` is one of {}, not `{}`", kinds.join(", "), v.chars().take(32).collect::<String>())))
}

fn only(cmd: &str, p: &Value, keys: &[&str]) -> Result<()> {
    if !p.is_object() {
        return Err(bad(cmd, "parameters must be an object"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !keys.contains(&k.as_str()))) {
        return Err(bad(cmd, format!("unknown parameter `{}`: {}", k.chars().take(64).collect::<String>(), keys.join(", "))));
    }
    Ok(())
}

fn deal_run(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = DEAL_CMD;
    only(cmd, p, &["for", "options", "seed", "dryRun", "pipeline"])?;
    let kind = kind_p(cmd, p, "for", &KINDS)?;
    let all = catalog(kind);
    let candidates: Vec<&'static str> = match p.get("options").filter(|v| !v.is_null()) {
        None => all.clone(),
        Some(v) => {
            let a = v
                .as_array()
                .filter(|a| !a.is_empty() && a.len() <= MAX_OPTIONS)
                .ok_or_else(|| bad(cmd, format!("`options` is a list of 1 to {MAX_OPTIONS} ids")))?;
            let mut out: Vec<&'static str> = Vec::new();
            for x in a {
                let id = x.as_str().and_then(|i| canonical(kind, i)).ok_or_else(|| bad(cmd, format!("`options`: each is one of {}", all.join(", "))))?;
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            out
        }
    };
    let seed = match p.get("seed").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad(cmd, "`seed` is a whole number"))?),
    };
    let dry = match p.get("dryRun").filter(|v| !v.is_null()) {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(cmd, "`dryRun` is true or false"))?,
    };
    let pipeline = match p.get("pipeline").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_str()
                .map(str::trim)
                .filter(|x| !x.is_empty() && x.len() <= 32 && x.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
                .ok_or_else(|| bad(cmd, "`pipeline` is a short name (direct, quick, showcase, shorts, …)"))?
                .to_ascii_lowercase(),
        ),
    };
    let history = taste_history::read(s);
    let mut out = json!({"for": kind, "history": {"records": history.records.len(), "notes": history.notes}});
    if kind == "music" && pipeline.as_deref() == Some("shorts") {
        out["choice"] = Value::Null;
        out["from"] = json!([]);
        out["skipped"] = json!([]);
        out["why"] = json!("Shorts stay silent: no music is dealt");
        out["kept"] = json!(false);
        return Ok(out);
    }
    let taste = crate::taste::resolved(s);
    let d = deal(&taste, &history, kind, &candidates, seed);
    let kept = !dry && d.choice.is_some();
    if kept && let Some(c) = &d.choice {
        s.taste_deals.insert(kind.to_string(), c.clone());
    }
    out["choice"] = json!(d.choice);
    out["from"] = json!(d.from);
    out["skipped"] = json!(d.skipped);
    out["why"] = json!(d.why);
    out["kept"] = json!(kept);
    if kept {
        out["note"] = json!("kept for the next autocut.fingerprint {record: true}, which records it with the video");
    }
    Ok(out)
}

fn feedback_run(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = FEEDBACK_CMD;
    only(cmd, p, &["about", "id", "verdict", "note", "scope"])?;
    let kind = kind_p(cmd, p, "about", &FEEDBACK_KINDS)?;
    let all = catalog(kind);
    let id = str_p(p, "id").and_then(|i| canonical(kind, i)).ok_or_else(|| bad(cmd, format!("`id` is one of {} for {kind}", all.join(", "))))?;
    let like = match str_p(p, "verdict").map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("dislike") => false,
        Some("like") => true,
        _ => return Err(bad(cmd, "`verdict` is \"dislike\" or \"like\"")),
    };
    let note = match p.get("note").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => {
            Some(v.as_str().filter(|n| n.chars().count() <= MAX_NOTE).ok_or_else(|| bad(cmd, format!("`note` is a text of at most {MAX_NOTE} characters")))?)
        }
    };
    let scope = str_p(p, "scope").map(|x| x.trim().to_ascii_lowercase()).unwrap_or_else(|| "default".into());
    if !matches!(scope.as_str(), "project" | "default") {
        return Err(bad(cmd, "`scope` is default or project"));
    }
    let base = if scope == "project" { s.project.taste.clone().unwrap_or_default() } else { crate::taste::default_taste(s) };
    let next = base.with_feedback(kind, id, like, note);
    crate::taste::save(s, cmd, &scope, next, "Style Feedback")?;
    let resolved = crate::taste::resolved(s);
    let effect = if like {
        format!("{kind} {id} is liked: dealt first among the choices not used in the last {RECENT} videos")
    } else if kind == "sound" {
        format!("the {id} sound is retired: never heard on cuts or titles, whatever the style")
    } else {
        format!("{kind} {id} is retired: never dealt again, and the pipelines' own defaults skip it")
    };
    Ok(json!({
        "about": kind,
        "id": id,
        "verdict": if like { "like" } else { "dislike" },
        "scope": scope,
        "effect": effect,
        "retired": resolved.retired,
        "liked": resolved.liked,
        "taste": crate::taste::answer(s),
    }))
}

#[cfg(test)]
#[path = "deal_tests.rs"]
mod tests;

//! A taste profile (`autocut.taste`): how one person likes their videos made — the cuts and the
//! sounds on them, light effects, the colour pass, the language of on-screen text, wording rules
//! ("our son", never "my son"), how narration is written, the person's own phrases (which the
//! copy and style checks never flag), how things move (`motion`: `playful` lets eases overshoot),
//! privacy, and what the person's feedback retired or liked (`retired`, `liked`, `reasons`:
//! `autocut.feedback`; a retired choice is never dealt and the pipelines' defaults skip it).
//! Every field is optional:
//! a profile is laid over the one below it — the user's default under a project's, a video's own
//! options over both ([`Taste::over`]) — and what none of them says is each pipeline's own
//! default.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Most wording rules, narration rules and sounds a profile keeps.
pub const MAX_RULES: usize = 32;
/// Longest text a profile keeps (a rule, a word), characters.
pub const MAX_TEXT: usize = 300;
/// Most of the person's own phrases a profile keeps.
pub const MAX_PHRASES: usize = 200;
/// Most kinds of retired or liked choices, the longest kind name and id (characters), and the
/// most reasons a profile keeps.
pub const MAX_KINDS: usize = 16;
pub const MAX_KIND: usize = 32;
pub const MAX_ID: usize = 64;
pub const MAX_REASONS: usize = 64;

/// Choices by kind (`look` → `["warm", "vhs"]`).
pub type Choices = BTreeMap<String, Vec<String>>;

/// One wording rule: say `say`, never `not`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wording {
    pub say: String,
    pub not: String,
}

/// A taste profile.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Taste {
    /// How lively the cuts are: `subtle`, `energetic` or `cinematic`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// Light effects (light leaks, sweeps, flares, flashes) allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub light_effects: Option<bool>,
    /// The sounds heard on cuts and titles (`swish`, `chime`, `whoosh`, `impact`, `riser`,
    /// `shutter`, `pop`, `thump`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sounds: Option<Vec<String>>,
    /// How loud they are, 0..1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sound_level: Option<f64>,
    /// The careful colour pass on, how strong (0..1), and a look (`none`, `punchy`, …) at a
    /// strength (%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_strength: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub look: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub look_strength: Option<f64>,
    /// The language on-screen text is written in (`en`, `ml`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_language: Option<String>,
    /// Words used instead of others, in on-screen text and narration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Wording>>,
    /// How narration is written, in the person's own words ("tell the story with feeling, don't
    /// describe what is on screen").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub narration: Option<Vec<String>>,
    /// Phrases the person uses on purpose ("let's dive in", "hello everyone"): `copy.lint` and the
    /// style checks never flag them, whatever their word lists say (decision D6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phrases: Option<Vec<String>>,
    /// What is hidden: `off` (faces sharp), `faces`, `plates`, `faces,plates`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact: Option<String>,
    /// How titles and graphics move: `calm` (the default: nothing passes its mark) or `playful`
    /// (overshooting and bouncing eases are meant, so `sequence.checkMotion` does not flag them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub motion: Option<String>,
    /// Choices the person's feedback retired (`autocut.feedback`), by kind: `look`,
    /// `titleStyle`, `transitionFamily`, `music`, `register`, `endCard`, `sound`, `captionStyle`,
    /// `opening`. A retired id is never dealt (`autocut.deal`) and the pipelines' own defaults
    /// skip it; a retired `sound` is never heard, as if `sounds` left it out. A profile's
    /// retired choices add to the one below it, except what it likes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retired: Option<Choices>,
    /// Choices the person liked: dealt first among those the last videos did not use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub liked: Option<Choices>,
    /// Why, in the person's words: `kind:id` → their note.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasons: Option<BTreeMap<String, String>>,
}

fn text(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(MAX_TEXT).collect::<String>().trim().to_string()
}

fn unit(v: Option<f64>, max: f64) -> Option<f64> {
    v.filter(|x| x.is_finite()).map(|x| x.clamp(0.0, max))
}

/// A kind's name: letters and digits, at most [`MAX_KIND`].
fn kind_name(s: &str) -> String {
    s.chars().filter(char::is_ascii_alphanumeric).take(MAX_KIND).collect()
}

/// An id: no control characters, trimmed, at most [`MAX_ID`].
fn choice_id(s: &str) -> String {
    text(s).chars().take(MAX_ID).collect::<String>().trim().to_string()
}

fn has(m: Option<&Choices>, kind: &str, id: &str) -> bool {
    m.and_then(|m| m.get(kind)).is_some_and(|v| v.iter().any(|x| x.eq_ignore_ascii_case(id.trim())))
}

/// Choices usable as kept: kinds and ids cleaned and capped, duplicates and empties gone, none
/// when nothing is left.
fn choices_sanitized(m: Option<&Choices>) -> Option<Choices> {
    let m = m?;
    let mut out = Choices::new();
    for (k, ids) in m.iter().take(MAX_KINDS) {
        let k = kind_name(k);
        if k.is_empty() {
            continue;
        }
        let mut kept: Vec<String> = Vec::new();
        for id in ids.iter().map(|i| choice_id(i)).filter(|i| !i.is_empty()) {
            if !kept.iter().any(|x| x.eq_ignore_ascii_case(&id)) {
                kept.push(id);
            }
        }
        // the newest are kept (feedback adds at the end)
        let extra = kept.len().saturating_sub(MAX_RULES);
        kept.drain(..extra);
        if !kept.is_empty() {
            out.entry(k).or_default().extend(kept);
        }
    }
    (!out.is_empty()).then_some(out)
}

/// `a` and `b` together, without what `except` names.
fn choices_union(a: Option<&Choices>, b: Option<&Choices>, except: Option<&Choices>) -> Option<Choices> {
    let mut out = Choices::new();
    for m in [b, a].into_iter().flatten() {
        for (k, ids) in m {
            let list = out.entry(k.clone()).or_default();
            for id in ids {
                if !list.iter().any(|x| x.eq_ignore_ascii_case(id)) && !has(except, k, id) {
                    list.push(id.clone());
                }
            }
        }
    }
    out.retain(|_, v| !v.is_empty());
    (!out.is_empty()).then_some(out)
}

impl Taste {
    /// `self` laid over `base`: each field `self` sets wins.
    pub fn over(&self, base: &Taste) -> Taste {
        Taste {
            style: self.style.clone().or_else(|| base.style.clone()),
            light_effects: self.light_effects.or(base.light_effects),
            sounds: self.sounds.clone().or_else(|| base.sounds.clone()),
            sound_level: self.sound_level.or(base.sound_level),
            color: self.color.or(base.color),
            color_strength: self.color_strength.or(base.color_strength),
            look: self.look.clone().or_else(|| base.look.clone()),
            look_strength: self.look_strength.or(base.look_strength),
            text_language: self.text_language.clone().or_else(|| base.text_language.clone()),
            words: self.words.clone().or_else(|| base.words.clone()),
            narration: self.narration.clone().or_else(|| base.narration.clone()),
            phrases: self.phrases.clone().or_else(|| base.phrases.clone()),
            redact: self.redact.clone().or_else(|| base.redact.clone()),
            motion: self.motion.clone().or_else(|| base.motion.clone()),
            retired: choices_union(self.retired.as_ref(), base.retired.as_ref(), self.liked.as_ref()),
            liked: choices_union(self.liked.as_ref(), base.liked.as_ref(), self.retired.as_ref()),
            reasons: match (&self.reasons, &base.reasons) {
                (None, None) => None,
                (a, b) => {
                    let mut r = b.clone().unwrap_or_default();
                    r.extend(a.clone().unwrap_or_default());
                    Some(r)
                }
            },
        }
    }

    /// Usable values: texts trimmed and capped, lists capped, numbers finite and in range (a
    /// project file is not trusted).
    pub fn sanitized(&self) -> Taste {
        let short = |s: &Option<String>| s.as_deref().map(text).filter(|t| !t.is_empty());
        Taste {
            style: short(&self.style),
            light_effects: self.light_effects,
            sounds: self.sounds.as_ref().map(|v| v.iter().map(|s| text(s)).filter(|s| !s.is_empty()).take(MAX_RULES).collect()),
            sound_level: unit(self.sound_level, 1.0),
            color: self.color,
            color_strength: unit(self.color_strength, 1.0),
            look: short(&self.look),
            look_strength: unit(self.look_strength, 100.0),
            text_language: short(&self.text_language),
            words: self.words.as_ref().map(|v| {
                v.iter().map(|w| Wording { say: text(&w.say), not: text(&w.not) }).filter(|w| !w.not.is_empty() && w.say != w.not).take(MAX_RULES).collect()
            }),
            narration: self.narration.as_ref().map(|v| v.iter().map(|s| text(s)).filter(|s| !s.is_empty()).take(MAX_RULES).collect()),
            phrases: self.phrases.as_ref().map(|v| v.iter().map(|s| text(s)).filter(|s| !s.is_empty()).take(MAX_PHRASES).collect()),
            redact: short(&self.redact),
            motion: short(&self.motion),
            retired: choices_sanitized(self.retired.as_ref()),
            liked: choices_sanitized(self.liked.as_ref()),
            reasons: self
                .reasons
                .as_ref()
                .map(|r| {
                    r.iter()
                        .map(|(k, v)| (text(k).chars().take(MAX_KIND + MAX_ID + 1).collect::<String>(), text(v)))
                        .filter(|(k, v)| !k.is_empty() && !v.is_empty())
                        .take(MAX_REASONS)
                        .collect::<BTreeMap<String, String>>()
                })
                .filter(|r| !r.is_empty()),
        }
    }

    /// Whether the person's feedback retired `id` of `kind` (any case).
    pub fn is_retired(&self, kind: &str, id: &str) -> bool {
        has(self.retired.as_ref(), kind, id)
    }

    /// Whether the person liked `id` of `kind` (any case).
    pub fn is_liked(&self, kind: &str, id: &str) -> bool {
        has(self.liked.as_ref(), kind, id)
    }

    /// The profile after a verdict on `id` of `kind`: liked (out of the retired) or retired (out
    /// of the liked), the newest kept when a kind is full, with the person's note.
    pub fn with_feedback(&self, kind: &str, id: &str, like: bool, note: Option<&str>) -> Taste {
        let mut t = self.clone();
        let (kind, id) = (kind_name(kind), choice_id(id));
        if kind.is_empty() || id.is_empty() {
            return t.sanitized();
        }
        let (add, other) = if like { (&mut t.liked, &mut t.retired) } else { (&mut t.retired, &mut t.liked) };
        if let Some(list) = other.as_mut().and_then(|m| m.get_mut(&kind)) {
            list.retain(|x| !x.eq_ignore_ascii_case(&id));
        }
        let list = add.get_or_insert_with(Choices::new).entry(kind.clone()).or_default();
        list.retain(|x| !x.eq_ignore_ascii_case(&id));
        list.push(id.clone());
        if let Some(n) = note.map(text).filter(|n| !n.is_empty()) {
            t.reasons.get_or_insert_with(BTreeMap::new).insert(format!("{kind}:{id}"), n);
        }
        t.sanitized()
    }

    pub fn is_empty(&self) -> bool {
        *self == Taste::default()
    }

    /// Whether the profile declares a playful style of motion (`motion: "playful"`, any case).
    pub fn playful_motion(&self) -> bool {
        self.motion.as_deref().is_some_and(|m| m.trim().eq_ignore_ascii_case("playful"))
    }

    /// `s` with the wording rules applied: each `not` (whole words, any case) made `say`, a
    /// capital kept where the sentence had one.
    pub fn apply_words(&self, s: &str) -> String {
        let mut out = s.to_string();
        for w in self.words.as_deref().unwrap_or_default() {
            out = replace_words(&out, &w.not, &w.say);
        }
        out
    }

    /// The wording and narration rules as instructions for a writer (empty without any).
    pub fn writing_rules(&self) -> Vec<String> {
        let mut v: Vec<String> = self.words.as_deref().unwrap_or_default().iter().map(|w| format!("Say \"{}\", never \"{}\".", w.say, w.not)).collect();
        v.extend(self.narration.iter().flatten().cloned());
        v
    }
}

/// `s` with every whole-word, case-insensitive `from` made `to` (a leading capital kept).
fn replace_words(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }
    let lower = s.to_lowercase();
    let needle = from.to_lowercase();
    // (lower-casing can change byte lengths: only plain text whose length it keeps is changed)
    if lower.len() != s.len() || needle.len() != from.len() {
        return s.to_string();
    }
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric());
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(off) = lower.get(i..).and_then(|rest| rest.find(&needle)) {
        let at = i + off;
        let end = at + needle.len();
        let before = s.get(..at).and_then(|p| p.chars().next_back());
        let after = s.get(end..).and_then(|p| p.chars().next());
        out.push_str(s.get(i..at).unwrap_or_default());
        if word(before) || word(after) {
            out.push_str(s.get(at..end).unwrap_or_default());
        } else {
            let capital = s.get(at..).and_then(|p| p.chars().next()).is_some_and(char::is_uppercase);
            if capital {
                let mut c = to.chars();
                out.extend(c.next().map(|f| f.to_uppercase().collect::<String>()));
                out.push_str(c.as_str());
            } else {
                out.push_str(to);
            }
        }
        i = end;
    }
    out.push_str(s.get(i..).unwrap_or_default());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ours() -> Taste {
        Taste { words: Some(vec![Wording { say: "our son".into(), not: "my son".into() }]), ..Taste::default() }
    }

    #[test]
    fn wording_rules_change_whole_words_and_keep_capitals() {
        let t = ours();
        assert_eq!(t.apply_words("My son built it. Then my son tested it."), "Our son built it. Then our son tested it.");
        assert_eq!(t.apply_words("my sonar works"), "my sonar works", "whole words only");
        assert_eq!(t.apply_words("nothing to change"), "nothing to change");
        assert_eq!(t.writing_rules(), ["Say \"our son\", never \"my son\"."]);
        // text whose case folding changes its length is left as it is
        assert_eq!(Taste { words: Some(vec![Wording { say: "x".into(), not: "İ".into() }]), ..Taste::default() }.apply_words("İstanbul"), "İstanbul");
    }

    #[test]
    fn profiles_layer_and_survive_hostile_files() {
        let default = Taste { style: Some("subtle".into()), sound_level: Some(0.4), redact: Some("off".into()), ..Taste::default() };
        let project = Taste { style: Some("cinematic".into()), light_effects: Some(true), ..ours() };
        let t = project.over(&default);
        assert_eq!((t.style.as_deref(), t.sound_level, t.light_effects, t.redact.as_deref()), (Some("cinematic"), Some(0.4), Some(true), Some("off")));
        assert!(t.words.is_some());
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Taste>(&json).unwrap(), t);
        assert_eq!(serde_json::from_str::<Taste>("{}").unwrap(), Taste::default());
        let bad = Taste {
            sound_level: Some(f64::NAN),
            look_strength: Some(1e9),
            words: Some(vec![Wording { say: "a".into(), not: String::new() }; 100]),
            narration: Some(vec!["x".repeat(10_000); 100]),
            style: Some("\u{0}  ".into()),
            ..Taste::default()
        }
        .sanitized();
        assert_eq!((bad.sound_level, bad.look_strength, bad.style), (None, Some(100.0), None));
        assert!(bad.words.unwrap().is_empty());
        let n = bad.narration.unwrap();
        assert!(n.len() == MAX_RULES && n[0].chars().count() == MAX_TEXT);
    }

    #[test]
    fn feedback_retires_and_likes_and_layers() {
        let t = Taste::default().with_feedback("sound", "whoosh", false, Some("sounds like a rocket")).with_feedback("look", "vhs", true, None);
        assert!(t.is_retired("sound", "Whoosh") && !t.is_retired("sound", "swish") && t.is_liked("look", "vhs"));
        assert_eq!(t.reasons.as_ref().unwrap()["sound:whoosh"], "sounds like a rocket");
        // a later like takes it out of the retired; the empty list goes
        let back = t.with_feedback("sound", "whoosh", true, None);
        assert!(!back.is_retired("sound", "whoosh") && back.is_liked("sound", "whoosh"));
        assert!(back.retired.is_none());
        // the user's retired choices stay under a project's, except what the project likes
        let user = Taste::default().with_feedback("look", "warm", false, None).with_feedback("music", "sad", false, None);
        let project = Taste::default().with_feedback("music", "sad", true, None).with_feedback("look", "teal", false, None);
        let both = project.over(&user);
        assert!(both.is_retired("look", "warm") && both.is_retired("look", "teal") && !both.is_retired("music", "sad") && both.is_liked("music", "sad"));
        // an old profile without them still reads; a hostile one is capped
        assert_eq!(serde_json::from_str::<Taste>(r#"{"style":"subtle"}"#).unwrap().retired, None);
        let mut many = Choices::new();
        for k in 0..40 {
            many.insert(format!("kind{k}\u{7}"), vec!["x".repeat(500); 100]);
        }
        many.insert("look".into(), (0..100).map(|i| format!("id{i}")).collect());
        let bad =
            Taste { retired: Some(many), reasons: Some((0..500).map(|i| (format!("k{i}"), "n".repeat(10_000))).collect()), ..Taste::default() }.sanitized();
        let r = bad.retired.unwrap();
        assert!(r.len() <= MAX_KINDS && r.values().all(|v| v.len() <= MAX_RULES && v.iter().all(|i| i.chars().count() <= MAX_ID)));
        let reasons = bad.reasons.unwrap();
        assert!(reasons.len() == MAX_REASONS && reasons.values().all(|v| v.chars().count() == MAX_TEXT));
        let full = (0..MAX_RULES + 5).fold(Taste::default(), |t, i| t.with_feedback("look", &format!("l{i}"), false, None));
        assert!(full.is_retired("look", &format!("l{}", MAX_RULES + 4)) && !full.is_retired("look", "l0"), "the newest are kept");
    }

    #[test]
    fn own_phrases_are_kept_capped_and_layered() {
        let many = Taste { phrases: Some(vec!["  hello everyone \u{7}".into(); MAX_PHRASES + 50]), ..Taste::default() }.sanitized();
        let kept = many.phrases.unwrap();
        assert_eq!((kept.len(), kept[0].as_str()), (MAX_PHRASES, "hello everyone"));
        let blank = Taste { phrases: Some(vec!["   ".into(), "x".repeat(10_000)]), ..Taste::default() }.sanitized();
        assert_eq!(blank.phrases.unwrap().iter().map(|p| p.chars().count()).collect::<Vec<_>>(), [MAX_TEXT]);
        let base = Taste { phrases: Some(vec!["let's dive in".into()]), ..Taste::default() };
        assert_eq!(Taste::default().over(&base).phrases, base.phrases);
        // how things move: layered like the rest, read in any case
        let playful = Taste { motion: Some(" Playful ".into()), ..Taste::default() }.sanitized();
        assert!(playful.playful_motion() && Taste::default().over(&playful).playful_motion());
        assert!(!Taste { motion: Some("calm".into()), ..Taste::default() }.over(&playful).playful_motion());
        assert!(!Taste::default().playful_motion());
        let json = serde_json::to_value(&base).unwrap();
        assert_eq!(json["phrases"][0], "let's dive in");
    }
}

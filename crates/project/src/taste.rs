//! A taste profile (`autocut.taste`): how one person likes their videos made — the cuts and the
//! sounds on them, light effects, the colour pass, the language of on-screen text, wording rules
//! ("our son", never "my son"), how narration is written, the person's own phrases (which the
//! copy and style checks never flag), and privacy. Every field is optional:
//! a profile is laid over the one below it — the user's default under a project's, a video's own
//! options over both ([`Taste::over`]) — and what none of them says is each pipeline's own
//! default.

use serde::{Deserialize, Serialize};

/// Most wording rules, narration rules and sounds a profile keeps.
pub const MAX_RULES: usize = 32;
/// Longest text a profile keeps (a rule, a word), characters.
pub const MAX_TEXT: usize = 300;
/// Most of the person's own phrases a profile keeps.
pub const MAX_PHRASES: usize = 200;

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
}

fn text(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(MAX_TEXT).collect::<String>().trim().to_string()
}

fn unit(v: Option<f64>, max: f64) -> Option<f64> {
    v.filter(|x| x.is_finite()).map(|x| x.clamp(0.0, max))
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
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == Taste::default()
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
    fn own_phrases_are_kept_capped_and_layered() {
        let many = Taste { phrases: Some(vec!["  hello everyone \u{7}".into(); MAX_PHRASES + 50]), ..Taste::default() }.sanitized();
        let kept = many.phrases.unwrap();
        assert_eq!((kept.len(), kept[0].as_str()), (MAX_PHRASES, "hello everyone"));
        let blank = Taste { phrases: Some(vec!["   ".into(), "x".repeat(10_000)]), ..Taste::default() }.sanitized();
        assert_eq!(blank.phrases.unwrap().iter().map(|p| p.chars().count()).collect::<Vec<_>>(), [MAX_TEXT]);
        let base = Taste { phrases: Some(vec!["let's dive in".into()]), ..Taste::default() };
        assert_eq!(Taste::default().over(&base).phrases, base.phrases);
        let json = serde_json::to_value(&base).unwrap();
        assert_eq!(json["phrases"][0], "let's dive in");
    }
}

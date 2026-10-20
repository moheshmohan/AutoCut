//! `script.lintMalayalam`: a Malayalam narration script checked for what makes it sound written
//! rather than spoken (`docs/malayalam-style.md`), and against the English plan it translates.
//!
//! The script is a Malayalam lines file (`{"intro", "outro", "lines": {"<entry label>": "…"}}`, as
//! `showcase-plan --lines` takes it), a Malayalam showcase plan (`script.lines` beside
//! `entries`), or plain `text`. Each part (intro, outro, an entry's line) gets the findings of
//! [`filmcraft_tts::ml_script::lint`] (long sentences, formal words with a spoken form, numbers
//! in Malayalam words, Latin letters; a sentence without a verb as a hint) and, given the
//! English plan, the fact check of [`filmcraft_tts::ml_facts`] against its English line (every
//! number, name, unit and month kept, no number added). `fix: true` also returns the script with
//! the safe replacements made.
//!
//! The functions are public for the CLI (`filmcraft-cli ml-lint`, `showcase`'s check of a
//! Malayalam script before it is voiced) and AutoCut's assisted translation.

use std::collections::BTreeMap;

use filmcraft_tts::ml_facts::{FactCheck, FactLock};
use filmcraft_tts::ml_script::{self, Finding, Kind, Level, Options};
use serde_json::{Map, Value, json};

/// The lint and the fact lock themselves, for callers above the engine.
pub use filmcraft_tts::{ml_facts as facts, ml_script as lint};

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{Result, Session};

const CMD: &str = "script.lintMalayalam";
/// Largest file read.
const MAX_FILE_BYTES: u64 = 16 << 20;
/// Most parts one script has.
pub const MAX_PARTS: usize = 5000;

/// One part of a script: `intro`, `outro` or an entry's label, and what it says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub key: String,
    pub text: String,
}

/// The English a Malayalam script translates: a plan's spoken parts, its respellings and its
/// entries' labels (whose capitalised words are names).
#[derive(Clone, Debug, Default)]
pub struct English {
    pub parts: Vec<Part>,
    pub pronounce: BTreeMap<String, String>,
    pub labels: Vec<String>,
}

impl English {
    /// The fact lock of this plan.
    pub fn lock(&self) -> FactLock {
        let texts: Vec<&str> = self.parts.iter().map(|p| p.text.as_str()).collect();
        let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
        FactLock::new(&self.pronounce, &texts, &labels)
    }

    /// The English of part `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.parts.iter().find(|p| p.key == key).map(|p| p.text.as_str())
    }

    /// The keys in the order the show speaks them.
    pub fn order(&self) -> Vec<&str> {
        self.parts.iter().map(|p| p.key.as_str()).collect()
    }
}

fn text_of(v: Option<&Value>) -> String {
    v.and_then(Value::as_str).map(str::trim).unwrap_or_default().to_string()
}

/// The spoken parts of a showcase plan (`{"entries": [{"label"}], "script": {"intro", "lines",
/// "outro", "pronounce"}}`), in the order they are said: the intro, each entry with words, the
/// outro. Entries without words (a scene card) are left out.
pub fn english_of_plan(plan: &Value) -> std::result::Result<English, String> {
    let script = plan.get("script").filter(|s| s.is_object()).ok_or("the plan has no `script`")?;
    let entries = plan.get("entries").and_then(Value::as_array).ok_or("the plan has no `entries`")?;
    let lines = script.get("lines").and_then(Value::as_array).ok_or("the plan's script has no `lines`")?;
    if entries.len() > MAX_PARTS {
        return Err(format!("the plan has {} entries; at most {MAX_PARTS}", entries.len()));
    }
    let mut parts = vec![Part { key: "intro".into(), text: text_of(script.get("intro")) }];
    let mut labels = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let label = text_of(e.get("label"));
        let line = text_of(lines.get(i));
        if !label.is_empty() {
            labels.push(label.clone());
        }
        if !line.is_empty() && !label.is_empty() && !parts.iter().any(|p| p.key == label) {
            parts.push(Part { key: label, text: line });
        }
    }
    parts.push(Part { key: "outro".into(), text: text_of(script.get("outro")) });
    parts.retain(|p| !p.text.is_empty());
    let pronounce = script
        .get("pronounce")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string()))).collect())
        .unwrap_or_default();
    Ok(English { parts, pronounce, labels })
}

/// The parts of a Malayalam script: a lines file (`{"intro", "outro", "lines": {label: …}}`), a
/// Malayalam showcase plan, or `{"text": …}`.
pub fn malayalam_parts(v: &Value) -> std::result::Result<Vec<Part>, String> {
    if let Some(t) = v.get("text").and_then(Value::as_str) {
        return Ok(vec![Part { key: "text".into(), text: t.to_string() }]);
    }
    if let Some(by_label) = v.get("lines").and_then(Value::as_object) {
        if by_label.len() > MAX_PARTS {
            return Err(format!("{} lines; at most {MAX_PARTS}", by_label.len()));
        }
        let mut parts = vec![Part { key: "intro".into(), text: text_of(v.get("intro")) }];
        parts.extend(by_label.iter().map(|(k, l)| Part { key: k.clone(), text: text_of(Some(l)) }));
        parts.push(Part { key: "outro".into(), text: text_of(v.get("outro")) });
        parts.retain(|p| !p.text.is_empty());
        return Ok(parts);
    }
    if v.get("script").is_some() {
        return english_of_plan(v).map(|e| e.parts);
    }
    Err("not a Malayalam script: give a lines file ({\"intro\", \"outro\", \"lines\": {\"<label>\": \"…\"}}), a plan with `script`, or `text`".into())
}

/// A lines file of `parts` (`{"intro", "outro", "lines": {label: …}}`).
pub fn lines_file(parts: &[Part]) -> Value {
    let mut lines = Map::new();
    let mut out = Map::new();
    for p in parts {
        match p.key.as_str() {
            "intro" | "outro" => {
                out.insert(p.key.clone(), json!(p.text));
            }
            k => {
                lines.insert(k.to_string(), json!(p.text));
            }
        }
    }
    out.insert("lines".into(), Value::Object(lines));
    Value::Object(out)
}

/// What the lint found in one part.
#[derive(Clone, Debug)]
pub struct PartReport {
    pub key: String,
    pub findings: Vec<Finding>,
    /// Against its English, when the plan was given.
    pub facts: Option<FactCheck>,
    pub fixed: Option<ml_script::Fixed>,
}

impl PartReport {
    pub fn warnings(&self) -> usize {
        self.findings.iter().filter(|f| f.level == Level::Warning).count()
    }

    pub fn facts_missing(&self) -> usize {
        self.facts.as_ref().map_or(0, |f| f.missing.len() + f.added.len())
    }

    /// Lines for people: "<part>: <kind>: «text» → suggestion" (warnings first, then hints when
    /// `hints`), and the facts lost or added.
    pub fn lines(&self, hints: bool) -> Vec<String> {
        let mut out = Vec::new();
        for level in [Level::Warning, Level::Hint] {
            if level == Level::Hint && !hints {
                continue;
            }
            for f in self.findings.iter().filter(|f| f.level == level) {
                let what = match f.kind {
                    Kind::LongSentence | Kind::NoVerb => format!("{}: {}", f.message, clip(&f.text, 60)),
                    _ => format!("«{}» → {}", f.text, f.suggestion),
                };
                let tag = if level == Level::Hint { "hint" } else { f.kind.id() };
                let extra = if matches!(f.kind, Kind::LongSentence | Kind::NoVerb) { format!(" ({})", f.suggestion) } else { String::new() };
                out.push(format!("{}: {tag}: {what}{extra}", self.key));
            }
        }
        if let Some(c) = self.facts.as_ref().filter(|c| !c.is_clean()) {
            out.push(format!("{}: facts: {}", self.key, c.summary()));
        }
        out
    }

    fn to_json(&self, text: &str) -> Value {
        let mut v = json!({"part": self.key, "text": text, "findings": self.findings});
        if let Some(f) = &self.facts {
            v["facts"] = json!(f);
        }
        if let Some(f) = &self.fixed {
            v["fixed"] = json!(f.text);
            v["applied"] = json!(f.applied);
        }
        v
    }
}

fn clip(s: &str, n: usize) -> String {
    let mut c = s.chars();
    let head: String = c.by_ref().take(n).collect();
    if c.next().is_some() { format!("{head}…") } else { head }
}

/// The lint of each part (in `english`'s order when given), with the fact check of each part
/// that has an English line and, with `fix`, its safe replacements made.
pub fn lint_parts(parts: &[Part], english: Option<&English>, opts: &Options, fix: bool) -> Vec<PartReport> {
    let lock = english.map(English::lock);
    let mut ordered: Vec<&Part> = parts.iter().collect();
    if let Some(e) = english {
        let order = e.order();
        let rank = |k: &str| order.iter().position(|o| *o == k).unwrap_or(usize::MAX);
        ordered.sort_by_key(|p| rank(&p.key));
    }
    ordered
        .into_iter()
        .take(MAX_PARTS)
        .map(|p| {
            let findings = ml_script::lint_with(&p.text, opts);
            let facts = match (english.and_then(|e| e.get(&p.key)), &lock) {
                (Some(en), Some(l)) => Some(l.check(en, &p.text)),
                _ => None,
            };
            let fixed = fix.then(|| ml_script::fix_with(&p.text, opts));
            PartReport { key: p.key.clone(), findings, facts, fixed }
        })
        .collect()
}

/// The report of [`lint_parts`] as the command answers it.
pub fn report_json(parts: &[Part], reports: &[PartReport], english: Option<&English>) -> Value {
    let text = |k: &str| parts.iter().find(|p| p.key == k).map(|p| p.text.as_str()).unwrap_or("");
    let mut by_kind = Map::new();
    for k in Kind::ALL {
        let n = reports.iter().flat_map(|r| &r.findings).filter(|f| f.kind == k).count();
        by_kind.insert(k.id().into(), json!(n));
    }
    let warnings: usize = reports.iter().map(PartReport::warnings).sum();
    let hints: usize = reports.iter().flat_map(|r| &r.findings).filter(|f| f.level == Level::Hint).count();
    let missing: usize = reports.iter().filter_map(|r| r.facts.as_ref()).map(|f| f.missing.len()).sum();
    let added: usize = reports.iter().filter_map(|r| r.facts.as_ref()).map(|f| f.added.len()).sum();
    let mut summary = json!({
        "parts": reports.len(),
        "warnings": warnings,
        "hints": hints,
        "byKind": by_kind,
        "clean": warnings == 0 && missing == 0 && added == 0,
    });
    if let Some(e) = english {
        summary["factsMissing"] = json!(missing);
        summary["numbersAdded"] = json!(added);
        // spoken parts of the English the script has no line for
        let have: Vec<&str> = parts.iter().map(|p| p.key.as_str()).collect();
        let untranslated: Vec<&str> = e.parts.iter().map(|p| p.key.as_str()).filter(|k| !have.contains(k)).collect();
        summary["untranslated"] = json!(untranslated);
    }
    let mut out = json!({"parts": reports.iter().map(|r| r.to_json(text(&r.key))).collect::<Vec<_>>(), "summary": summary});
    if reports.iter().any(|r| r.fixed.is_some()) {
        let fixed: Vec<Part> =
            reports.iter().map(|r| Part { key: r.key.clone(), text: r.fixed.as_ref().map_or_else(|| text(&r.key).to_string(), |f| f.text.clone()) }).collect();
        out["fixedScript"] = lines_file(&fixed);
    }
    out
}

fn read_json(s: &Session, path: &str) -> Result<Value> {
    let path = crate::export_tools::expand_home(path);
    let size = s.services.file_size(&path).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
    if size > MAX_FILE_BYTES {
        return Err(bad(CMD, format!("{path} is {size} bytes; at most {MAX_FILE_BYTES}")));
    }
    let bytes = s.services.read_file(&path).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(v) => Ok(v),
        // a plain text file is a script too
        Err(_) => Ok(json!({"text": String::from_utf8_lossy(&bytes)})),
    }
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    let given = ["text", "lines", "plan", "path"].iter().filter(|k| p.get(**k).is_some_and(|v| !v.is_null())).count();
    if given != 1 {
        return Err(bad(CMD, "give the Malayalam script once: `text`, `lines` (a lines file), `plan` (a Malayalam plan) or `path`"));
    }
    let source = if let Some(path) = str_p(p, "path") {
        read_json(s, path)?
    } else if let Some(t) = p.get("text") {
        let t = t.as_str().ok_or_else(|| bad(CMD, "`text` must be a string"))?;
        json!({"text": t})
    } else if let Some(l) = p.get("lines").filter(|v| !v.is_null()) {
        if !l.is_object() {
            return Err(bad(CMD, "`lines` is a lines file: {\"intro\", \"outro\", \"lines\": {\"<label>\": \"…\"}}"));
        }
        if l.get("lines").is_some() { l.clone() } else { json!({"lines": l}) }
    } else {
        p.get("plan").cloned().unwrap_or(Value::Null)
    };
    if p.get("path").is_some_and(|v| !v.is_string()) {
        return Err(bad(CMD, "`path` must be a file path"));
    }
    let parts = malayalam_parts(&source).map_err(|e| bad(CMD, e))?;
    let english = match (p.get("english").filter(|v| !v.is_null()), str_p(p, "englishPath")) {
        (Some(_), Some(_)) => return Err(bad(CMD, "give `english` or `englishPath`, not both")),
        (Some(v), None) => Some(english_of_plan(v).map_err(|e| bad(CMD, format!("english: {e}")))?),
        (None, Some(path)) => Some(english_of_plan(&read_json(s, path)?).map_err(|e| bad(CMD, format!("{path}: {e}")))?),
        (None, None) => None,
    };
    let max_words = match p.get("maxWords").filter(|v| !v.is_null()) {
        None => ml_script::DEFAULT_MAX_WORDS,
        Some(v) => v
            .as_u64()
            .filter(|n| (ml_script::MIN_MAX_WORDS as u64..=ml_script::MAX_MAX_WORDS as u64).contains(n))
            .ok_or_else(|| bad(CMD, format!("`maxWords` is a whole number from {} to {}", ml_script::MIN_MAX_WORDS, ml_script::MAX_MAX_WORDS)))?
            as usize,
    };
    let fix = match p.get("fix").filter(|v| !v.is_null()) {
        None => false,
        Some(_) => bool_p(p, "fix").ok_or_else(|| bad(CMD, "`fix` must be true or false"))?,
    };
    let opts = Options { max_words };
    let reports = lint_parts(&parts, english.as_ref(), &opts, fix);
    Ok(report_json(&parts, &reports, english.as_ref()))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Malayalam Script",
        menu: &[],
        shortcut: None,
        params: r#"{"text":str?,"lines":{"intro":str,"outro":str,"lines":{"<label>":str}}?,"plan":Plan?,"path":str?,"english":Plan?,"englishPath":str?,"maxWords":4..60=14,"fix":bool=false}"#,
        enabled: always,
        run,
        journal: false,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Value {
        json!({
            "entries": [
                {"label": "Promobot Robo-C"},
                {"label": "Scene · 2022"},
                {"label": "Thermite firefighting robot"}
            ],
            "script": {
                "intro": "This is every robot I've met.",
                "lines": [
                    "Promobot's Robo-C makes over six hundred facial expressions.",
                    "",
                    "It pumps two and a half thousand gallons of water a minute and runs for twenty hours on diesel."
                ],
                "outro": "Thanks for watching, and tell me which one you'd most like to meet.",
                "pronounce": {"Robo-C": "Robo-See"}
            }
        })
    }

    #[test]
    fn a_script_is_linted_and_checked_against_its_english() {
        let mut s = Session::default();
        let lines = json!({
            "intro": "ഞാൻ കണ്ട എല്ലാ റോബോട്ടുകളും ഇതിലുണ്ട്.",
            "outro": "കണ്ടതിന് നന്ദി, ഏതാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടം എന്ന് കമന്റിൽ പറയൂ.",
            "lines": {
                "Promobot Robo-C": "പ്രൊമോബോട്ടിന്റെ റോബോ സി-ക്ക് അറുന്നൂറിലധികം മുഖഭാവങ്ങൾ കാണിക്കാൻ പറ്റും.",
                "Thermite firefighting robot": "ഒരു മിനിറ്റിൽ ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ പത്ത് മണിക്കൂർ ഓടും."
            }
        });
        let r = s.execute(CMD, json!({"lines": lines, "english": plan(), "fix": true})).unwrap();
        let parts = r["parts"].as_array().unwrap();
        // in the order the show speaks them
        let keys: Vec<&str> = parts.iter().map(|p| p["part"].as_str().unwrap()).collect();
        assert_eq!(keys, ["intro", "Promobot Robo-C", "Thermite firefighting robot", "outro"]);
        let promo = &parts[1];
        let kinds: Vec<&str> = promo["findings"].as_array().unwrap().iter().map(|f| f["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["numberWords", "formal"]);
        assert_eq!(promo["fixed"], "പ്രൊമോബോട്ടിന്റെ റോബോ സി-ക്ക് 600-ലധികം എക്സ്പ്രഷൻസ് കാണിക്കാൻ പറ്റും.");
        assert!(promo["facts"]["missing"].as_array().unwrap().is_empty(), "{}", promo["facts"]);
        // twenty hours became ten
        let thermite = &parts[2]["facts"];
        assert_eq!(thermite["missing"][0]["text"], "twenty");
        assert_eq!(thermite["added"], json!(["10"]));
        let sum = &r["summary"];
        assert_eq!((sum["factsMissing"].as_u64(), sum["numbersAdded"].as_u64(), sum["clean"].as_bool()), (Some(1), Some(1), Some(false)));
        assert_eq!(sum["byKind"]["formal"], 3, "{sum}");
        assert_eq!(sum["untranslated"], json!([]));
        // the fixed script is a lines file again
        assert_eq!(r["fixedScript"]["outro"], "കണ്ടതിന് താങ്ക്സ്, ഏതാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടം എന്ന് കമന്റിൽ പറയണേ.");
        assert!(r["fixedScript"]["lines"]["Promobot Robo-C"].is_string());
        // a part the English speaks but the script lacks
        let r = s.execute(CMD, json!({"lines": {"intro": "ഹായ്.", "outro": "ബൈ.", "lines": {}}, "english": plan()})).unwrap();
        assert_eq!(r["summary"]["untranslated"], json!(["Promobot Robo-C", "Thermite firefighting robot"]));
    }

    #[test]
    fn text_plans_and_files() {
        let mut s = Session::default();
        let r = s.execute(CMD, json!({"text": "സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു."})).unwrap();
        assert_eq!(r["summary"]["warnings"], 2);
        assert!(r["summary"].get("factsMissing").is_none());
        // a Malayalam plan reads its lines beside its entries
        let mut ml = plan();
        ml["script"]["lines"] = json!(["പ്രൊമോബോട്ട്.", "", "തെർമൈറ്റ്."]);
        ml["script"]["intro"] = json!("ഹായ്.");
        let r = s.execute(CMD, json!({"plan": ml})).unwrap();
        assert_eq!(r["summary"]["parts"], 4);
        // a file: JSON, or plain text
        let dir = std::env::temp_dir().join(format!("fc-ml-lint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("script.txt");
        std::fs::write(&txt, "UAE-യിൽ നിർമ്മിച്ചതാണ്.").unwrap();
        let r = s.execute(CMD, json!({"path": txt.to_string_lossy(), "maxWords": 10})).unwrap();
        assert_eq!(r["parts"][0]["findings"].as_array().unwrap().len(), 2);
        let plan_file = dir.join("plan.json");
        std::fs::write(&plan_file, serde_json::to_vec(&plan()).unwrap()).unwrap();
        let r = s.execute(CMD, json!({"lines": {"Promobot Robo-C": "പ്രൊമോബോട്ട്."}, "englishPath": plan_file.to_string_lossy()})).unwrap();
        let missing: Vec<&str> = r["parts"][0]["facts"]["missing"].as_array().unwrap().iter().map(|m| m["text"].as_str().unwrap()).collect();
        assert_eq!(missing, ["Robo-C", "six hundred"]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn hostile_parameters_are_refused() {
        let mut s = Session::default();
        for p in [
            json!(null),
            json!([]),
            json!({}),
            json!({"text": 5}),
            json!({"text": "a", "lines": {}}),
            json!({"lines": "x"}),
            json!({"plan": {"entries": 3}}),
            json!({"text": "a", "maxWords": 0}),
            json!({"text": "a", "maxWords": 1000}),
            json!({"text": "a", "maxWords": "ten"}),
            json!({"text": "a", "fix": "yes"}),
            json!({"text": "a", "english": {"script": {}}}),
            json!({"text": "a", "english": {}, "englishPath": "/x"}),
            json!({"path": "/no/such/file.json"}),
            json!({"path": 7}),
        ] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        // odd but well-formed scripts are fine
        for p in [json!({"text": ""}), json!({"lines": {"lines": {"x": 5, "y": null}}}), json!({"text": "😀".repeat(5000)})] {
            assert!(s.execute(CMD, p.clone()).is_ok(), "{p}");
        }
    }

    /// The real scripts of a showcase round, when given: the first draft gets many findings and
    /// the rewrite few. `ML_LINT_WRITTEN=old.json ML_LINT_SPOKEN=new.json [ML_LINT_PLAN=plan.json]
    /// cargo test -p filmcraft-engine ml_script -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_scripts() {
        let (Ok(old), Ok(new)) = (std::env::var("ML_LINT_WRITTEN"), std::env::var("ML_LINT_SPOKEN")) else { return };
        let mut s = Session::default();
        let english = std::env::var("ML_LINT_PLAN").ok();
        let lint = |s: &mut Session, path: &str| {
            let mut p = json!({"path": path});
            if let Some(e) = &english {
                p["englishPath"] = json!(e);
            }
            s.execute(CMD, p).unwrap()["summary"].clone()
        };
        let (a, b) = (lint(&mut s, &old), lint(&mut s, &new));
        println!("written: {a}\nspoken: {b}");
        assert!(a["warnings"].as_u64().unwrap() > 2 * b["warnings"].as_u64().unwrap());
    }
}

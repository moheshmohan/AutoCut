//! Assisted translation of a showcase plan's English voice-over into natural spoken Malayalam,
//! with the local script model (Ollama, `AUTOCUT_SCRIPT_MODEL`, `qwen3:8b` by default).
//!
//! The model is given the style of `docs/malayalam-style.md` (short sentences, the English tech
//! words a Malayali creator uses written in Malayalam script, spoken endings, numbers as digits,
//! acronyms letter by letter, names by their sound) and examples: a few built in
//! ([`BUILTIN_EXAMPLES`]), and the user's own approved pairs from a file they keep in the data
//! directory ([`examples_path`], never in the repository; [`remember`] adds a reviewed script's
//! pairs to it). Each English line goes with the facts it must keep ([`FactLock::facts`]: its
//! numbers, names, units, months).
//!
//! Every line the model writes then gets the safe fixes of [`ml_script::fix`], the lint and the
//! fact check; a line with a warning or a lost or added fact goes back to the model once with
//! what to fix, and the better of the two is kept. The result is a lines file
//! (`{"intro", "outro", "lines": {"<label>": …}}`, as `showcase-plan --lines` takes it) and a
//! report of what is left for a person to look at.
//!
//! A small local model's Malayalam is weak (measured with qwen3:8b on a 52-part plan: lint-clean,
//! but 19 parts lost or changed a fact and more changed meaning unchecked; see
//! `docs/malayalam-style.md`): the result is a rough draft, the lint and the fact lock are the
//! dependable part, and the report says where to look. Any writer (a person, or the agent
//! driving the editor) can follow the same guide and check its lines with `script.lintMalayalam`.

use std::path::{Path, PathBuf};

use filmcraft_engine::ml_script::facts::{FactCheck, FactLock};
use filmcraft_engine::ml_script::lint::{self as ml_script, Applied, Finding, Level, Options};
use filmcraft_engine::ml_script::{English, Part, english_of_plan, lines_file};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::AutoCutError;
use crate::llm::ChatModel;

/// Lines per model request.
pub const DEFAULT_BATCH: usize = 3;
/// Most approved pairs the examples file keeps.
pub const MAX_EXAMPLES: usize = 500;
/// The user's pairs shown to the model (the most recent), and the built-in ones with them.
const USER_EXAMPLES_SHOWN: usize = 6;
const BUILTIN_EXAMPLES_SHOWN: usize = 6;
/// Largest examples file read.
const MAX_EXAMPLES_BYTES: u64 = 8 << 20;
/// The context window the translation asks for: Malayalam takes many tokens.
pub const CONTEXT_TOKENS: u32 = 16_384;

/// An English line and how it is said in Malayalam.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pair {
    pub en: String,
    pub ml: String,
}

/// Built-in examples: generic lines in the style, the facts kept.
pub const BUILTIN_EXAMPLES: [(&str, &str); 10] = [
    (
        "This humanoid was designed in Japan and launched in 2022 for about 3,000 dollars.",
        "ഈ ഹ്യൂമനോയിഡ് ജപ്പാനിലാണ് ഡിസൈൻ ചെയ്തത്. 2022-ൽ ലോഞ്ച് ചെയ്തു, വില ഏകദേശം 3,000 ഡോളർ.",
    ),
    (
        "It greets visitors at the entrance and answers their questions in five languages.",
        "എൻട്രൻസിൽ വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്നത് ഇവനാണ്. അഞ്ച് ഭാഷകളിൽ ചോദ്യങ്ങൾക്ക് മറുപടിയും പറയും.",
    ),
    ("Its AI uses a 4K camera and lidar to map the whole room.", "ഇതിന്റെ എ ഐ, ഫോർ കെ ക്യാമറയും ലിഡാറും വെച്ചാണ് റൂം മൊത്തം മാപ്പ് ചെയ്യുന്നത്."),
    (
        "A start-up in Germany built it, and it sorts more than 10,000 parcels an hour.",
        "ജർമനിയിലെ ഒരു സ്റ്റാർട്ടപ്പ് ഉണ്ടാക്കിയതാണ് ഇത്. മണിക്കൂറിൽ 10,000-ലധികം പാഴ്സലുകൾ സോർട്ട് ചെയ്യും.",
    ),
    ("Honestly, it looked so real that I had to look twice.", "സത്യം പറഞ്ഞാൽ, കണ്ടാൽ ശരിക്കും ഒറിജിനലാണെന്നേ തോന്നൂ. ഞാൻ ഒന്നൂടെ നോക്കിപ്പോയി."),
    ("At CES in Las Vegas, Samsung showed its new robot vacuum.", "ലാസ് വേഗാസിൽ നടന്ന സി ഇ എസ് ഷോയിൽ സാംസങ് അവരുടെ പുതിയ റോബോട്ട് വാക്വം കാണിച്ചു."),
    ("The arm can lift up to 20 kilograms and runs for 8 hours on a single charge.", "ഈ ആമിന് 20 കിലോ വരെ ഈസിയായി ഉയർത്താൻ പറ്റും. ഒറ്റ ചാർജിൽ 8 മണിക്കൂർ ഓടും."),
    ("Next up is a delivery robot that finds its own way through the crowd.", "പിന്നെ ഒരു ഡെലിവറി റോബോട്ട്. ആൾക്കൂട്ടത്തിനിടയിലൂടെ ഇവൻ തനിയെ വഴി കണ്ടുപിടിച്ച് പോകും."),
    (
        "Two years later, in March 2024, I saw it again at the same show, and it could even dance.",
        "രണ്ട് വർഷം കഴിഞ്ഞ്, 2024 മാർച്ചിൽ, അതേ ഷോയിൽ വീണ്ടും കണ്ടു. ഇപ്പോൾ ഡാൻസ് ഒക്കെ ചെയ്യും!",
    ),
    ("Thanks for watching, and tell me in the comments which one you liked most.", "കണ്ടതിന് താങ്ക്സ്. ഇതിൽ ഏതാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടപ്പെട്ടത്? കമന്റിൽ പറയണേ."),
];

pub fn builtin_examples() -> Vec<Pair> {
    BUILTIN_EXAMPLES.iter().map(|(en, ml)| Pair { en: en.to_string(), ml: ml.to_string() }).collect()
}

/// The file of approved pairs the user keeps: `<data>/autocut/malayalam-examples.json`.
pub fn examples_path() -> Option<PathBuf> {
    filmcraft_engine::autosave::default_data_dir().map(|d| d.join("autocut").join("malayalam-examples.json"))
}

/// The pairs in an examples file (`[{"en", "ml"}]` or `{"pairs": [...]}`); a missing file has none.
pub fn load_examples(path: &Path) -> Result<Vec<Pair>, AutoCutError> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(AutoCutError::Other(format!("{}: {e}", path.display()))),
    };
    if meta.len() > MAX_EXAMPLES_BYTES {
        return Err(AutoCutError::Other(format!("{} is {} bytes; at most {MAX_EXAMPLES_BYTES}", path.display(), meta.len())));
    }
    let bytes = std::fs::read(path).map_err(|e| AutoCutError::Other(format!("{}: {e}", path.display())))?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| AutoCutError::Other(format!("{}: not JSON ({e})", path.display())))?;
    let list = v.get("pairs").cloned().unwrap_or(v);
    let pairs: Vec<Pair> =
        serde_json::from_value(list).map_err(|e| AutoCutError::Other(format!("{}: not a list of {{\"en\", \"ml\"}} pairs ({e})", path.display())))?;
    Ok(pairs.into_iter().filter(|p| !p.en.trim().is_empty() && !p.ml.trim().is_empty()).take(MAX_EXAMPLES).collect())
}

/// Add `pairs` to the examples file at `path` (made when missing): a pair for an English line
/// already there replaces it, the newest last, at most [`MAX_EXAMPLES`]. The number kept.
pub fn remember(path: &Path, pairs: &[Pair]) -> Result<usize, AutoCutError> {
    let mut all = load_examples(path)?;
    for p in pairs.iter().filter(|p| !p.en.trim().is_empty() && !p.ml.trim().is_empty()) {
        all.retain(|q| q.en.trim() != p.en.trim());
        all.push(p.clone());
    }
    let skip = all.len().saturating_sub(MAX_EXAMPLES);
    let all: Vec<Pair> = all.into_iter().skip(skip).collect();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| AutoCutError::Other(format!("{}: {e}", dir.display())))?;
    }
    let text = serde_json::to_string_pretty(&json!({"pairs": all})).unwrap_or_default();
    std::fs::write(path, format!("{text}\n")).map_err(|e| AutoCutError::Other(format!("{}: {e}", path.display())))?;
    Ok(all.len())
}

/// The pairs of a reviewed Malayalam script and the English plan it translates, by part.
pub fn pairs_of(english: &English, malayalam: &[Part]) -> Vec<Pair> {
    english
        .parts
        .iter()
        .filter_map(|e| malayalam.iter().find(|m| m.key == e.key).map(|m| Pair { en: e.text.clone(), ml: m.text.trim().to_string() }))
        .filter(|p| !p.ml.is_empty())
        .collect()
}

/// How to translate.
#[derive(Clone, Debug)]
pub struct Settings {
    pub max_words: usize,
    /// Send lines with findings back to the model once.
    pub revise: bool,
    /// Lines per request.
    pub batch: usize,
    /// The user's approved pairs ([`load_examples`]).
    pub examples: Vec<Pair>,
}

impl Default for Settings {
    fn default() -> Self {
        Self { max_words: ml_script::DEFAULT_MAX_WORDS, revise: true, batch: DEFAULT_BATCH, examples: Vec::new() }
    }
}

/// One translated part and what is left to look at.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Line {
    pub part: String,
    pub english: String,
    pub malayalam: String,
    pub findings: Vec<Finding>,
    pub facts: FactCheck,
    /// The model rewrote it after the first check, and the rewrite was better.
    pub revised: bool,
    /// The safe fixes made to the model's words.
    pub fixes: Vec<Applied>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Line {
    fn warnings(&self) -> usize {
        self.findings.iter().filter(|f| f.level == Level::Warning).count()
    }

    /// Lower is better: a lost or added fact weighs three warnings.
    fn score(&self) -> usize {
        self.warnings() + 3 * (self.facts.missing.len() + self.facts.added.len())
    }
}

/// A translated plan.
#[derive(Clone, Debug, Serialize)]
pub struct Translated {
    pub model: String,
    pub lines: Vec<Line>,
}

impl Translated {
    /// The lines file (parts the model did not translate are left out).
    pub fn script(&self) -> Value {
        let parts: Vec<Part> = self
            .lines
            .iter()
            .filter(|l| l.error.is_none() && !l.malayalam.is_empty())
            .map(|l| Part { key: l.part.clone(), text: l.malayalam.clone() })
            .collect();
        lines_file(&parts)
    }

    pub fn summary(&self) -> Value {
        let failed: Vec<&str> = self.lines.iter().filter(|l| l.error.is_some()).map(|l| l.part.as_str()).collect();
        let warnings: usize = self.lines.iter().map(Line::warnings).sum();
        let hints: usize = self.lines.iter().flat_map(|l| &l.findings).filter(|f| f.level == Level::Hint).count();
        let missing: usize = self.lines.iter().map(|l| l.facts.missing.len()).sum();
        let added: usize = self.lines.iter().map(|l| l.facts.added.len()).sum();
        let clean = self.lines.iter().filter(|l| l.error.is_none() && l.warnings() == 0 && l.facts.is_clean()).count();
        json!({
            "model": self.model,
            "parts": self.lines.len(),
            "translated": self.lines.len() - failed.len(),
            "failed": failed,
            "clean": clean,
            "warnings": warnings,
            "hints": hints,
            "factsMissing": missing,
            "numbersAdded": added,
            "revised": self.lines.iter().filter(|l| l.revised).count(),
            "fixes": self.lines.iter().map(|l| l.fixes.len()).sum::<usize>(),
        })
    }

    /// Lines for people: each part with something left to look at.
    pub fn notes(&self) -> Vec<String> {
        let mut out = Vec::new();
        for l in &self.lines {
            if let Some(e) = &l.error {
                out.push(format!("{}: not translated ({e})", l.part));
                continue;
            }
            for f in l.findings.iter().filter(|f| f.level == Level::Warning) {
                out.push(format!("{}: {}: «{}» → {}", l.part, f.kind.id(), clip(&f.text, 60), f.suggestion));
            }
            if !l.facts.is_clean() {
                out.push(format!("{}: facts: {}", l.part, l.facts.summary()));
            }
        }
        out
    }
}

fn clip(s: &str, n: usize) -> String {
    let mut c = s.chars();
    let head: String = c.by_ref().take(n).collect();
    if c.next().is_some() { format!("{head}…") } else { head }
}

/// The style the model writes in.
pub const STYLE: &str = "You translate the English voice-over of a YouTube tech video into Malayalam, the way a Malayali tech YouTuber says it to friends: natural spoken Malayalam, never written, news or textbook Malayalam.
- Malayalam script only: no Latin letters at all.
- Short sentences, at most 12 words each: split a long English sentence into two or three.
- Everyday English tech words stay English, written in Malayalam script: വെൽക്കം ചെയ്യുന്നു (not സ്വാഗതം ചെയ്യുന്നു), വിസിറ്റേഴ്സ് (not സന്ദർശകർ), ഡിസൈൻ ചെയ്തു, ലോഞ്ച് ചെയ്തു, ഡെവലപ്പ് ചെയ്യാൻ, കൺട്രോൾ ചെയ്യാം, സ്പീഡ്, ഫുഡ്, എക്സ്പ്രഷൻസ്, ഈസിയായി.
- Spoken forms, not written ones: ഉണ്ടാക്കി (not നിർമ്മിച്ചു), കിട്ടും (not ലഭിക്കും), പറ്റും (not സാധിക്കും), ശരിക്കുള്ള (not യഥാർത്ഥ), കണ്ടുപിടിക്കും (not കണ്ടെത്തും), -ക്ക് വേണ്ടി (not -ക്കായി), അതുകൊണ്ട് (not അതിനാൽ), താങ്ക്സ് (not നന്ദി), പറയണേ (not പറയൂ).
- Spoken endings and particles where they fit: ഉണ്ടാക്കിയതാ, തോന്നൂ, ഉള്ളൂ, ഇവനാണ്, പിന്നെ, ഒക്കെ.
- Numbers as digits (2021, 1,600, 85%): never Malayalam number words for 100 or more. Small counts may be Malayalam words (രണ്ട്, പത്ത്).
- Acronyms and codes letter by letter in Malayalam script: AI → എ ഐ, UAE → യു എ ഇ, 4K → ഫോർ കെ, M02 → എം സീറോ ടു.
- Names of people, companies, products and places in Malayalam script, by how they sound: GITEX → ജൈടെക്സ്, SoftBank → സോഫ്റ്റ്ബാങ്ക്.
- Keep every fact of the English line: every number, name, unit, month and place. Add nothing and drop nothing.
Answer with JSON only.";

/// The examples shown: the user's most recent, then built-in ones.
fn examples_block(user: &[Pair]) -> String {
    let mut shown: Vec<Pair> = user.iter().rev().take(USER_EXAMPLES_SHOWN).cloned().collect();
    shown.reverse();
    shown.extend(builtin_examples().into_iter().take(BUILTIN_EXAMPLES_SHOWN));
    shown.iter().map(|p| format!("EN: {}\nML: {}", p.en, p.ml)).collect::<Vec<_>>().join("\n\n")
}

/// The respellings that matter for `texts`: "GITEX = Jai-tex".
fn sounds_block(english: &English, texts: &[&str]) -> String {
    let said: Vec<String> =
        english.pronounce.iter().filter(|(k, _)| texts.iter().any(|t| t.contains(k.as_str()))).take(40).map(|(k, v)| format!("{k} = \"{v}\"")).collect();
    if said.is_empty() { String::new() } else { format!("How these are said: {}.\n\n", said.join("; ")) }
}

fn keep_list(lock: &FactLock, english: &str) -> String {
    let facts: Vec<String> = lock.facts(english).into_iter().map(|f| f.text).collect();
    let mut seen: Vec<String> = Vec::new();
    for f in facts {
        if !seen.contains(&f) {
            seen.push(f);
        }
    }
    if seen.is_empty() { "-".into() } else { seen.join("; ") }
}

/// The model's lines out of its answer: `{"lines": [...]}` (or a bare list, or `{"1": …}`).
fn answer_lines(v: &Value, n: usize) -> Vec<Option<String>> {
    let list: Vec<Value> = match v.get("lines").or(v.get("malayalam")).or(v.get("translations")) {
        Some(Value::Array(a)) => a.clone(),
        Some(Value::Object(o)) => (1..=n).map(|i| o.get(&i.to_string()).cloned().unwrap_or(Value::Null)).collect(),
        _ => match v {
            Value::Array(a) => a.clone(),
            Value::Object(o) => (1..=n).map(|i| o.get(&i.to_string()).cloned().unwrap_or(Value::Null)).collect(),
            _ => Vec::new(),
        },
    };
    (0..n)
        .map(|i| {
            let item = list.get(i)?;
            let s = item.as_str().map(str::to_string).or_else(|| item.get("ml").or(item.get("malayalam")).and_then(Value::as_str).map(str::to_string))?;
            let s = tidy(&s);
            (s.chars().any(|c| ('\u{0D00}'..='\u{0D7F}').contains(&c))).then_some(s)
        })
        .collect()
}

/// One clean line: no quotes, numbering, labels or line breaks.
fn tidy(s: &str) -> String {
    let s = s.replace(['\n', '\r', '"', '“', '”'], " ");
    let mut t = s.trim();
    // a list number the model copied ("1. ", "2) "), never the digits of a year or a count the
    // line starts with ("2023-ലെ", "12 വർഷം")
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if (1..=2).contains(&digits)
        && let Some(rest) = t.get(digits..).and_then(|r| r.strip_prefix(['.', ')']))
        && rest.starts_with(char::is_whitespace)
    {
        t = rest.trim();
    }
    for prefix in ["ML:", "Malayalam:", "ML :"] {
        if let Some(r) = t.strip_prefix(prefix) {
            t = r.trim();
        }
    }
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A line as the model wrote it, fixed and checked.
fn checked(part: &Part, english: &str, raw: &str, lock: &FactLock, opts: &Options) -> Line {
    let fixed = ml_script::fix_with(raw, opts);
    let findings = ml_script::lint_with(&fixed.text, opts);
    let facts = lock.check(english, &fixed.text);
    Line { part: part.key.clone(), english: english.to_string(), malayalam: fixed.text, findings, facts, revised: false, fixes: fixed.applied, error: None }
}

/// What to tell the model to fix in `line`.
fn fix_points(line: &Line) -> String {
    let mut points: Vec<String> = Vec::new();
    for f in line.findings.iter().filter(|f| f.level == Level::Warning) {
        points.push(match f.kind {
            ml_script::Kind::LongSentence => format!("{} — {}", f.message, f.suggestion),
            _ => format!("say «{}» instead of «{}»", f.suggestion, f.text),
        });
    }
    if !line.facts.missing.is_empty() {
        points.push(format!("keep: {}", line.facts.missing.iter().map(|f| f.text.as_str()).collect::<Vec<_>>().join(", ")));
    }
    if !line.facts.added.is_empty() {
        points.push(format!("the English does not say: {}", line.facts.added.join(", ")));
    }
    points.join(" | ")
}

/// Translate `plan`'s English voice-over (see the module docs).
pub async fn translate(model: &dyn ChatModel, plan: &Value, settings: &Settings, progress: &mut dyn FnMut(&str)) -> Result<Translated, AutoCutError> {
    let english = english_of_plan(plan).map_err(AutoCutError::Invalid)?;
    if english.parts.is_empty() {
        return Err(AutoCutError::Invalid("the plan has no English words to translate".into()));
    }
    let lock = english.lock();
    let opts = Options { max_words: settings.max_words.clamp(ml_script::MIN_MAX_WORDS, ml_script::MAX_MAX_WORDS) };
    let examples = examples_block(&settings.examples);
    let batch = settings.batch.clamp(1, 12);
    let total = english.parts.len();
    let mut lines: Vec<Line> = Vec::with_capacity(total);
    for (b, chunk) in english.parts.chunks(batch).enumerate() {
        progress(&format!(
            "[{}/{total}] translating {}",
            (b * batch + chunk.len()).min(total),
            chunk.iter().map(|p| p.key.as_str()).collect::<Vec<_>>().join(", ")
        ));
        let texts: Vec<&str> = chunk.iter().map(|p| p.text.as_str()).collect();
        let items: String =
            chunk.iter().enumerate().map(|(i, p)| format!("{}. \"{}\"\n   keep: {}", i + 1, p.text, keep_list(&lock, &p.text))).collect::<Vec<_>>().join("\n");
        let user = format!(
            "Examples (English, then how it is said):\n\n{examples}\n\n{}Translate each line below into spoken Malayalam. Keep the facts listed with it.\n\n{items}\n\nJSON: {{\"lines\": [\"<Malayalam of 1>\", …]}} with exactly {} lines, in order.",
            sounds_block(&english, &texts),
            chunk.len()
        );
        let answer = model.complete_json(STYLE, &user, &[]).await;
        let got = match &answer {
            Ok(v) => answer_lines(v, chunk.len()),
            Err(_) => vec![None; chunk.len()],
        };
        for (p, raw) in chunk.iter().zip(got) {
            let line = match raw {
                Some(raw) => checked(p, &p.text, &raw, &lock, &opts),
                // one more try, alone
                None => match single(model, &examples, &english, &lock, p).await {
                    Ok(raw) => checked(p, &p.text, &raw, &lock, &opts),
                    Err(e) => Line {
                        part: p.key.clone(),
                        english: p.text.clone(),
                        malayalam: String::new(),
                        findings: Vec::new(),
                        facts: FactCheck::default(),
                        revised: false,
                        fixes: Vec::new(),
                        error: Some(match &answer {
                            Err(first) => format!("{first}; then {e}"),
                            Ok(_) => e.to_string(),
                        }),
                    },
                },
            };
            lines.push(line);
        }
    }
    if settings.revise {
        let todo: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.error.is_none() && l.score() > 0).map(|(i, _)| i).collect();
        for (b, chunk) in todo.chunks(batch).enumerate() {
            progress(&format!("[{}/{}] revising", (b * batch + chunk.len()).min(todo.len()), todo.len()));
            let items: String = chunk
                .iter()
                .enumerate()
                .filter_map(|(i, at)| lines.get(*at).map(|l| (i, l)))
                .map(|(i, l)| format!("{}. English: \"{}\"\n   Malayalam: \"{}\"\n   fix: {}", i + 1, l.english, l.malayalam, fix_points(l)))
                .collect::<Vec<_>>()
                .join("\n");
            let user = format!(
                "Examples (English, then how it is said):\n\n{examples}\n\nThese Malayalam lines need fixing. Rewrite each one: fix every point listed, keep it natural spoken Malayalam, keep every fact of its English.\n\n{items}\n\nJSON: {{\"lines\": [\"<new Malayalam of 1>\", …]}} with exactly {} lines, in order.",
                chunk.len()
            );
            let Ok(v) = model.complete_json(STYLE, &user, &[]).await else { continue };
            for (at, raw) in chunk.iter().zip(answer_lines(&v, chunk.len())) {
                let (Some(raw), Some(old)) = (raw, lines.get(*at)) else { continue };
                let part = Part { key: old.part.clone(), text: old.english.clone() };
                let mut new = checked(&part, &old.english, &raw, &lock, &opts);
                if new.score() < old.score() {
                    new.revised = true;
                    if let Some(slot) = lines.get_mut(*at) {
                        *slot = new;
                    }
                }
            }
        }
    }
    Ok(Translated { model: model.id(), lines })
}

/// One part translated alone.
async fn single(model: &dyn ChatModel, examples: &str, english: &English, lock: &FactLock, p: &Part) -> Result<String, AutoCutError> {
    let user = format!(
        "Examples (English, then how it is said):\n\n{examples}\n\n{}Translate this line into spoken Malayalam. Keep: {}.\n\n\"{}\"\n\nJSON: {{\"lines\": [\"<Malayalam>\"]}}",
        sounds_block(english, &[p.text.as_str()]),
        keep_list(lock, &p.text),
        p.text
    );
    let v = model.complete_json(STYLE, &user, &[]).await?;
    answer_lines(&v, 1).into_iter().next().flatten().ok_or_else(|| AutoCutError::Llm(format!("{} wrote no Malayalam for {}", model.id(), p.key)))
}

/// The parts of a showcase script by entry label: the intro, each entry with words, the outro.
pub fn script_parts(script: &crate::showcase::Script, entries: &[crate::showcase::Entry]) -> Vec<Part> {
    // what the voice reads: a part's `say` when the plan has one (English on screen, Malayalam
    // spoken), else its text
    let said = |key: &str, text: &str| script.say.get(key).filter(|s| !s.trim().is_empty()).cloned().unwrap_or_else(|| text.to_string());
    let mut parts = vec![Part { key: "intro".into(), text: said("intro", &script.intro) }];
    for (e, l) in entries.iter().zip(&script.lines) {
        if !l.trim().is_empty() && !parts.iter().any(|p| p.key == e.label) {
            parts.push(Part { key: e.label.clone(), text: said(&e.number.to_string(), l) });
        }
    }
    parts.push(Part { key: "outro".into(), text: said("outro", &script.outro) });
    parts.retain(|p| !p.text.trim().is_empty());
    parts
}

/// The lint's summary of a Malayalam showcase script (`script.lintMalayalam`'s).
pub fn check_script(script: &crate::showcase::Script, entries: &[crate::showcase::Entry]) -> Value {
    let parts = script_parts(script, entries);
    let reports = filmcraft_engine::ml_script::lint_parts(&parts, None, &Options::default(), false);
    filmcraft_engine::ml_script::report_json(&parts, &reports, None)["summary"].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FnModel;

    /// English on screen, Malayalam spoken (`say`): the lint checks what the voice reads, so
    /// the English subtitles raised a "latin" warning for every word and --strict-script refused.
    #[test]
    fn a_script_spoken_through_say_is_checked_as_spoken() {
        let entries: Vec<crate::showcase::Entry> =
            vec![serde_json::from_value(serde_json::json!({"number": 1, "label": "Toy", "kind": "scene", "description": "", "unix": 1, "shots": []})).unwrap()];
        let script: crate::showcase::Script = serde_json::from_value(serde_json::json!({
            "intro": "", "lines": ["This is Akku's Tesla."], "outro": "Like and subscribe!",
            "say": {"1": "ഇത് അക്കുവിന്റെ ടെസ്‌ലയാണ്.", "outro": "ലൈക്ക് ചെയ്യണേ!"}
        }))
        .unwrap();
        let parts = script_parts(&script, &entries);
        let texts: Vec<&str> = parts.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(texts, ["ഇത് അക്കുവിന്റെ ടെസ്‌ലയാണ്.", "ലൈക്ക് ചെയ്യണേ!"]);
        assert_eq!(check_script(&script, &entries)["warnings"], serde_json::json!(0), "no Latin warnings for the English subtitles");
        // without `say`, the lines themselves are what is read
        let plain: crate::showcase::Script = serde_json::from_value(serde_json::json!({"intro": "", "lines": ["ഇത്."], "outro": ""})).unwrap();
        assert_eq!(script_parts(&plain, &entries).iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["ഇത്."]);
    }

    #[test]
    fn a_showcase_script_is_checked_by_entry() {
        let entries: Vec<crate::showcase::Entry> = serde_json::from_value(json!([
            {"number": 1, "label": "Pepper", "kind": "robot", "description": "", "unix": 1, "shots": []},
            {"number": 2, "label": "Scene", "kind": "scene", "description": "", "unix": 2, "shots": []}
        ]))
        .unwrap();
        let script = crate::showcase::Script {
            intro: "ഹായ്.".into(),
            lines: vec!["പെപ്പർ സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു.".into(), String::new()],
            outro: "ബൈ.".into(),
            ..Default::default()
        };
        let keys: Vec<String> = script_parts(&script, &entries).into_iter().map(|p| p.key).collect();
        assert_eq!(keys, ["intro", "Pepper", "outro"]);
        let c = check_script(&script, &entries);
        assert_eq!((c["parts"].as_u64(), c["warnings"].as_u64(), c["byKind"]["formal"].as_u64()), (Some(3), Some(2), Some(2)), "{c}");
    }

    fn plan() -> Value {
        json!({
            "entries": [
                {"label": "Pepper"},
                {"label": "A scene"},
                {"label": "Thermite firefighting robot"}
            ],
            "script": {
                "intro": "This is every robot I've met, from 2013 to 2025.",
                "lines": [
                    "Pepper was everywhere, greeting people. SoftBank built it to read emotions, and made about twenty-seven thousand.",
                    "",
                    "It pumps two and a half thousand gallons of water a minute and runs for twenty hours on diesel."
                ],
                "outro": "Thanks for watching, and tell me which one you'd most like to meet.",
                "pronounce": {"SoftBank": "Soft Bank"}
            }
        })
    }

    #[test]
    fn the_examples_follow_the_style_and_keep_their_facts() {
        for p in builtin_examples() {
            let warnings: Vec<Finding> = ml_script::lint(&p.ml).into_iter().filter(|f| f.level == Level::Warning).collect();
            assert!(warnings.is_empty(), "{}: {warnings:?}", p.ml);
            let c = filmcraft_engine::ml_script::facts::check(&p.en, &p.ml);
            assert!(c.is_clean(), "{} / {}: {c:?}", p.en, p.ml);
        }
    }

    #[tokio::test]
    async fn a_plan_is_translated_checked_and_revised() {
        // the model: a formal first draft with a changed number, then a natural rewrite
        let model = FnModel::new("fake", |system, user, _| {
            assert_eq!(system, STYLE);
            assert!(user.contains("EN: ") && user.contains("keep:") || user.contains("fix:"), "{user}");
            Ok(if user.contains("need fixing") {
                // the safe fixes were made before the model sees the line again
                assert!(user.contains("വെൽക്കം ചെയ്യുന്ന") && user.contains("keep: twenty-seven thousand") && user.contains("does not say: 10"), "{user}");
                let mut lines = Vec::new();
                if user.contains("പെപ്പർ") {
                    lines.push("എവിടെ നോക്കിയാലും ആളുകളെ വെൽക്കം ചെയ്യുന്ന പെപ്പർ. ഇമോഷൻസ് മനസ്സിലാക്കാനാണ് സോഫ്റ്റ് ബാങ്ക് ഇതിനെ ഉണ്ടാക്കിയത്. ഏകദേശം 27,000 എണ്ണം ഇറക്കി.");
                }
                if user.contains("ഗാലൻ") {
                    lines.push("ഒരു മിനിറ്റിൽ 2,500 ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ 20 മണിക്കൂർ ഓടും.");
                }
                json!({ "lines": lines })
            } else if user.contains("Pepper") {
                // the intro and Pepper in one batch (batch 2); Pepper loses its number
                json!({"lines": [
                    "\"1. ട്വന്റി തേർട്ടീൻ മുതൽ ട്വന്റി ട്വന്റി ഫൈവ് വരെ ഞാൻ കണ്ട എല്ലാ റോബോട്ടുകളും ഇതിലുണ്ട്.\"",
                    "എവിടെ നോക്കിയാലും ആളുകളെ സ്വാഗതം ചെയ്യുന്ന പെപ്പർ. വികാരങ്ങൾ മനസ്സിലാക്കാൻ സോഫ്റ്റ് ബാങ്ക് ഉണ്ടാക്കി."
                ]})
            } else if user.contains("gallons") {
                json!({"lines": ["ഒരു മിനിറ്റിൽ 2,500 ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ 10 മണിക്കൂർ ഓടും.", "കണ്ടതിന് താങ്ക്സ്. കമന്റിൽ പറയണേ."]})
            } else {
                json!({"lines": []})
            })
        });
        let settings = Settings { batch: 2, ..Settings::default() };
        let mut notes = Vec::new();
        let t = translate(&model, &plan(), &settings, &mut |l| notes.push(l.to_string())).await.unwrap();
        let parts: Vec<&str> = t.lines.iter().map(|l| l.part.as_str()).collect();
        assert_eq!(parts, ["intro", "Pepper", "Thermite firefighting robot", "outro"]);
        // the safe fixes run first: സ്വാഗതം → വെൽക്കം; the rewrite then fixes the rest
        let pepper = &t.lines[1];
        assert!(pepper.revised && pepper.findings.iter().all(|f| f.level != Level::Warning), "{pepper:?}");
        assert!(pepper.facts.is_clean(), "{:?}", pepper.facts);
        let thermite = &t.lines[2];
        assert!(thermite.revised && thermite.facts.is_clean(), "{thermite:?}");
        assert_eq!(t.lines[0].malayalam, "ട്വന്റി തേർട്ടീൻ മുതൽ ട്വന്റി ട്വന്റി ഫൈവ് വരെ ഞാൻ കണ്ട എല്ലാ റോബോട്ടുകളും ഇതിലുണ്ട്.");
        // the lines file showcase-plan --lines takes
        let s = t.script();
        assert_eq!(s["lines"]["Thermite firefighting robot"], "ഒരു മിനിറ്റിൽ 2,500 ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ 20 മണിക്കൂർ ഓടും.");
        assert!(s["intro"].is_string() && s["outro"].is_string() && s["lines"].as_object().unwrap().len() == 2);
        let (p, _) = crate::plan_patch::with_lines(&plan(), &s, "ml").unwrap();
        assert_eq!(p["script"]["lines"][1], "");
        let sum = t.summary();
        assert_eq!((sum["translated"].as_u64(), sum["revised"].as_u64(), sum["factsMissing"].as_u64()), (Some(4), Some(2), Some(0)), "{sum}");
        assert!(notes.iter().any(|n| n.contains("revising")), "{notes:?}");
    }

    #[tokio::test]
    async fn a_model_that_fails_leaves_the_part_out() {
        let model = FnModel::new("broken", |_, _, _| Err(AutoCutError::Llm("Ollama is unreachable".into())));
        let t = translate(&model, &plan(), &Settings::default(), &mut |_| {}).await.unwrap();
        assert!(t.lines.iter().all(|l| l.error.as_deref().is_some_and(|e| e.contains("unreachable"))));
        assert_eq!(t.script()["lines"], json!({}));
        assert_eq!(t.summary()["failed"].as_array().unwrap().len(), 4);
        assert!(t.notes().iter().all(|n| n.contains("not translated")));
        // a model answering English or nothing useful: the same
        let model = FnModel::new("english", |_, _, _| Ok(json!({"lines": ["Hello there.", 5, null]})));
        let t = translate(&model, &plan(), &Settings::default(), &mut |_| {}).await.unwrap();
        assert!(t.lines.iter().all(|l| l.error.is_some()));
        // a plan without words is refused
        assert!(translate(&model, &json!({"entries": [], "script": {"lines": []}}), &Settings::default(), &mut |_| {}).await.is_err());
        assert!(translate(&model, &json!({"nope": 1}), &Settings::default(), &mut |_| {}).await.is_err());
    }

    #[test]
    fn examples_are_remembered() {
        let dir = std::env::temp_dir().join(format!("fc-ml-examples-{}", std::process::id()));
        let path = dir.join("autocut").join("malayalam-examples.json");
        assert!(load_examples(&path).unwrap().is_empty());
        let e = english_of_plan(&plan()).unwrap();
        let ml = vec![
            Part { key: "Pepper".into(), text: "പെപ്പർ.".into() },
            Part { key: "outro".into(), text: "താങ്ക്സ്.".into() },
            Part { key: "nobody".into(), text: "x".into() },
        ];
        let pairs = pairs_of(&e, &ml);
        assert_eq!(pairs.len(), 2);
        assert_eq!(remember(&path, &pairs).unwrap(), 2);
        // the same English again replaces its pair, newest last
        let again = vec![Pair { en: pairs[0].en.clone(), ml: "പെപ്പർ!".into() }];
        assert_eq!(remember(&path, &again).unwrap(), 2);
        let kept = load_examples(&path).unwrap();
        assert_eq!(kept.last().unwrap().ml, "പെപ്പർ!");
        // the model sees them before the built-in ones
        assert!(examples_block(&kept).starts_with(&format!("EN: {}", kept[0].en)));
        // a bad file says so
        std::fs::write(&path, "{\"pairs\": 5}").unwrap();
        assert!(load_examples(&path).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn answers_are_read_leniently() {
        assert_eq!(answer_lines(&json!({"lines": ["എ", "ബി"]}), 2), [Some("എ".to_string()), Some("ബി".to_string())]);
        assert_eq!(answer_lines(&json!({"1": "എ", "2": "ബി"}), 2), [Some("എ".to_string()), Some("ബി".to_string())]);
        assert_eq!(answer_lines(&json!([{"ml": "എ"}]), 2), [Some("എ".to_string()), None]);
        assert_eq!(answer_lines(&json!({"lines": ["English only"]}), 1), [None]);
        assert_eq!(answer_lines(&json!("x"), 1), [None]);
        assert_eq!(tidy("  \"1. ML: ഇത്\n ആണ്.\" "), "ഇത് ആണ്.");
        // regression (a live run): a line starting with a year or a count kept it
        assert_eq!(tidy("2023-ലെ ഒക്ടോബറിൽ."), "2023-ലെ ഒക്ടോബറിൽ.");
        assert_eq!(tidy("12 വർഷം, 48 റോബോട്ടുകൾ."), "12 വർഷം, 48 റോബോട്ടുകൾ.");
        assert_eq!(tidy("2) 2024 മാർച്ചിൽ."), "2024 മാർച്ചിൽ.");
        assert_eq!(tidy("1.8 മീറ്റർ."), "1.8 മീറ്റർ.");
    }
}

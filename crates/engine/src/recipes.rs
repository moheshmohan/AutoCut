//! Recipes (A4): "save this edit as a recipe" — a parameterised chain of engine commands and
//! AutoCut intents (remove silence → captions in my style → my intro template → export →
//! publish), kept as JSON in `<data dir>/Recipes/` and run in one click, as a background job.
//!
//! | command | does |
//! |---|---|
//! | `recipe.list` | the built-in recipes and the user's (`<data dir>/Recipes/*.json`); files that fail to read are listed under `invalid` with why |
//! | `recipe.inspect` | one recipe (`name`: id or name) in full, and its steps as they would run with `params` (a dry run: each step checked, nothing runs) |
//! | `recipe.save` | a recipe from `steps` (and `params`, `description`, `id`), checked, written into the recipes folder (`overwrite` to replace) |
//! | `recipe.run` | job: run a recipe (`name`, or an inline `recipe`) with `params`; each step is checked before it runs (an intent through the AutoCut planner, whose commands are checked against the registry and the recipe allowlist), a command that starts a job is waited for, a failing step stops the run and says which and why; progress and cancel as any job (`jobs.list`, `jobs.cancel`); `dryRun` plans only |
//! | `recipe.delete` | a user recipe |
//! | `recipe.record` | "Save this edit as a recipe": `start` recording the edits made from now on, `stop` (with `name`: saved as a recipe), `cancel`, `status` |
//!
//! **Format** (`filmcraft.recipe`, version 1):
//!
//! ```json
//! {"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:my-clean-up", "name": "My clean-up",
//!  "description": "…",
//!  "params": [{"id": "style", "name": "Caption style", "kind": "choice", "choices": ["bold", "tiktok"], "default": "bold"},
//!             {"id": "output", "name": "Export to", "kind": "path", "default": ""}],
//!  "steps": [{"label": "Remove the silences", "intent": "remove_silence", "params": {"minSeconds": 0.5}},
//!            {"label": "Captions", "intent": "add_captions", "params": {"style": "${style}"}},
//!            {"label": "Export", "command": "file.exportMedia", "params": {"path": "${output}", "preset": "YouTube 1080p Full HD"}, "when": "output"}]}
//! ```
//!
//! A step is a `command` (an engine command id) or an `intent` (an AutoCut intent: `remove_silence`,
//! `remove_fillers`, `add_captions`, `enhance_voice`, `highlight_reel`, `export_video`, …, planned
//! against the open sequence when the step runs). In step parameters a string `"${id}"` is the
//! parameter's value (any type), and `${id}` inside other text its text; `when` names a parameter
//! the step needs to be on (true, a non-empty text, a non-zero number). Parameter kinds: `bool`,
//! `number` (`min`, `max`), `text`, `choice` (`choices`), `path`, `template` (a template id),
//! `json` (any value, e.g. a template's slots).
//!
//! Recipes are untrusted files (they can be shared like templates): sizes and counts are capped,
//! every step must be a command of the recipe allowlist (editing, captions, transcripts, sound,
//! graphics, templates' apply, export, publish — never preferences, the app, other recipes or
//! jobs), parameters are type-checked, and nothing runs until every step checked out.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};

use crate::autocut::CommandPort;
use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{EngineError, Result, Session};

/// `format` of a recipe file.
pub const FORMAT: &str = "filmcraft.recipe";
/// The format version this build reads and writes.
pub const FORMAT_VERSION: u64 = 1;
/// The folder of user recipes under the data directory.
pub const USER_FOLDER: &str = "Recipes";
/// Largest recipe file.
pub const MAX_RECIPE_BYTES: usize = 256 * 1024;
/// Most steps in a recipe.
pub const MAX_STEPS: usize = 64;
/// Most parameters of a recipe.
pub const MAX_PARAMS: usize = 32;
/// Largest step parameters (serialized).
const MAX_STEP_PARAMS_BYTES: usize = 64 * 1024;
/// Deepest nesting of step parameters.
const MAX_DEPTH: usize = 16;
/// Most recipe files read from the folder.
const MAX_FILES: usize = 512;
/// Most commands an intent may plan into.
const MAX_PLANNED: usize = 2000;
/// Longest a step's job is waited for.
const MAX_JOB_WAIT: Duration = Duration::from_secs(6 * 3600);

/// Command families a recipe may run (checked for every command step and every command an
/// intent plans): editing, transcripts, captions, effects, sound, graphics, narration, markers,
/// applying templates, exporting and publishing. Never preferences, the app, the UI, jobs or
/// other recipes.
pub const ALLOWED: &[&str] = &[
    "delivery.preflight",
    "delivery.prepare",
    "audio.configureStems",
    "sequence.",
    "timeline.",
    "source.",
    "markers.",
    "playhead.",
    "edit.",
    "transcript.",
    "captions.",
    "clip.",
    "effects.",
    "essentialSound.",
    "tts.",
    "graphics.",
    "media.",
    "lumetri.",
    "presets.apply",
    "trim.applyDefaultTransition",
    "audio.generate",
    "music.suggest",
    "mixer.addInsert",
    "mixer.setInsert",
    "template.apply",
    "file.exportMedia",
    "file.exportOtio",
    "file.import",
    "publish.run",
    "publish.preflight",
    "publish.fixCopy",
    "autocut.makeShowcase",
];

/// Whether a recipe may run command `id`.
pub fn allowed(id: &str) -> bool {
    !id.starts_with("recipe.") && ALLOWED.iter().any(|f| if f.ends_with('.') { id.starts_with(f) } else { id == *f })
}

// ------------------------------------------------------------------------------------------------
// The format
// ------------------------------------------------------------------------------------------------

/// A parameter's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Bool,
    Number,
    Text,
    Choice,
    Path,
    Template,
    Json,
}

impl ParamKind {
    pub fn id(self) -> &'static str {
        match self {
            ParamKind::Bool => "bool",
            ParamKind::Number => "number",
            ParamKind::Text => "text",
            ParamKind::Choice => "choice",
            ParamKind::Path => "path",
            ParamKind::Template => "template",
            ParamKind::Json => "json",
        }
    }
    pub fn from_id(s: &str) -> Option<ParamKind> {
        [ParamKind::Bool, ParamKind::Number, ParamKind::Text, ParamKind::Choice, ParamKind::Path, ParamKind::Template, ParamKind::Json]
            .into_iter()
            .find(|k| k.id() == s)
    }
}

/// One parameter of a recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: String,
    pub name: String,
    pub kind: ParamKind,
    pub default: Value,
    pub choices: Vec<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub help: String,
}

/// What a step does.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Command(String),
    Intent(String),
}

/// One step of a recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub label: String,
    pub action: Action,
    pub params: Value,
    /// The parameter that switches the step on.
    pub when: Option<String>,
}

/// A recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    pub description: String,
    pub params: Vec<Param>,
    pub steps: Vec<Step>,
}

fn is_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-:.".contains(&b))
}

fn is_param_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.bytes().enumerate().all(|(i, b)| b.is_ascii_alphabetic() || b == b'_' || (i > 0 && b.is_ascii_digit()))
}

fn depth(v: &Value) -> usize {
    // iterative, so a hostile document can't exhaust the stack here
    let mut max = 0;
    let mut stack = vec![(v, 1usize)];
    while let Some((x, d)) = stack.pop() {
        max = max.max(d);
        if d > MAX_DEPTH + 1 {
            return d;
        }
        match x {
            Value::Array(a) => stack.extend(a.iter().map(|y| (y, d + 1))),
            Value::Object(o) => stack.extend(o.values().map(|y| (y, d + 1))),
            _ => {}
        }
    }
    max
}

/// `${id}` references in a value's strings.
fn references(v: &Value, out: &mut Vec<String>) {
    let mut stack = vec![v];
    while let Some(x) = stack.pop() {
        match x {
            Value::String(s) => {
                let mut rest = s.as_str();
                while let Some(i) = rest.find("${") {
                    let after = rest.get(i + 2..).unwrap_or("");
                    match after.find('}') {
                        Some(j) => {
                            out.push(after.get(..j).unwrap_or("").to_string());
                            rest = after.get(j + 1..).unwrap_or("");
                        }
                        None => break,
                    }
                }
            }
            Value::Array(a) => stack.extend(a.iter()),
            Value::Object(o) => stack.extend(o.values()),
            _ => {}
        }
    }
}

fn text(o: &serde_json::Map<String, Value>, k: &str, max: usize, at: &str, errs: &mut Vec<String>) -> Option<String> {
    match o.get(k) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.chars().count() <= max => Some(s.clone()),
        Some(Value::String(_)) => {
            errs.push(format!("{at}.{k}: longer than {max} characters"));
            None
        }
        Some(_) => {
            errs.push(format!("{at}.{k}: expected text"));
            None
        }
    }
}

fn unknown(o: &serde_json::Map<String, Value>, known: &[&str], at: &str, errs: &mut Vec<String>) {
    for k in o.keys().take(64) {
        if !known.contains(&k.as_str()) {
            errs.push(format!("{at}.{k}: unknown field"));
        }
    }
}

fn read_param(v: &Value, at: &str, errs: &mut Vec<String>) -> Option<Param> {
    let Some(o) = v.as_object() else {
        errs.push(format!("{at}: expected an object"));
        return None;
    };
    unknown(o, &["id", "name", "kind", "default", "choices", "min", "max", "help"], at, errs);
    let id = text(o, "id", 64, at, errs).unwrap_or_default();
    if !is_param_id(&id) {
        errs.push(format!("{at}.id: `{id}`: letters, digits and `_`, starting with a letter"));
    }
    let kind = match o.get("kind").and_then(Value::as_str).and_then(ParamKind::from_id) {
        Some(k) => k,
        None => {
            errs.push(format!("{at}.kind: one of bool, number, text, choice, path, template, json"));
            return None;
        }
    };
    let choices: Vec<String> = match o.get("choices") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if a.len() <= 64 => a.iter().filter_map(Value::as_str).filter(|s| s.len() <= 200).map(str::to_string).collect(),
        Some(_) => {
            errs.push(format!("{at}.choices: a list of at most 64 texts"));
            Vec::new()
        }
    };
    if kind == ParamKind::Choice && choices.is_empty() {
        errs.push(format!("{at}.choices: a choice needs choices"));
    }
    let num = |k: &str, errs: &mut Vec<String>| match o.get(k) {
        None | Some(Value::Null) => None,
        Some(x) => match x.as_f64().filter(|n| n.is_finite()) {
            Some(n) => Some(n),
            None => {
                errs.push(format!("{at}.{k}: expected a number"));
                None
            }
        },
    };
    let (min, max) = (num("min", errs), num("max", errs));
    if let (Some(a), Some(b)) = (min, max)
        && a > b
    {
        errs.push(format!("{at}: min is more than max"));
    }
    let p = Param {
        name: text(o, "name", 200, at, errs).unwrap_or_else(|| id.clone()),
        id,
        kind,
        default: o.get("default").cloned().unwrap_or(Value::Null),
        choices,
        min,
        max,
        help: text(o, "help", 1000, at, errs).unwrap_or_default(),
    };
    if !p.default.is_null()
        && let Err(e) = p.check(&p.default)
    {
        errs.push(format!("{at}.default: {e}"));
    }
    Some(p)
}

impl Param {
    /// Check a value for this parameter.
    pub fn check(&self, v: &Value) -> std::result::Result<(), String> {
        let ok = match self.kind {
            ParamKind::Bool => v.is_boolean(),
            ParamKind::Number => match v.as_f64() {
                Some(n) if n.is_finite() => {
                    if self.min.is_some_and(|m| n < m) || self.max.is_some_and(|m| n > m) {
                        return Err(format!(
                            "{n} is out of range ({} to {})",
                            self.min.map_or("…".into(), |m| m.to_string()),
                            self.max.map_or("…".into(), |m| m.to_string())
                        ));
                    }
                    true
                }
                _ => false,
            },
            ParamKind::Text | ParamKind::Path | ParamKind::Template => v.as_str().is_some_and(|s| s.len() <= 4096 && !s.contains('\0')),
            ParamKind::Choice => match v.as_str() {
                Some(s) if self.choices.iter().any(|c| c == s) => true,
                Some(s) => return Err(format!("`{s}` is not one of {}", self.choices.join(", "))),
                None => false,
            },
            ParamKind::Json => depth(v) <= MAX_DEPTH && serde_json::to_vec(v).is_ok_and(|b| b.len() <= MAX_STEP_PARAMS_BYTES),
        };
        if ok { Ok(()) } else { Err(format!("expected a {} value", self.kind.id())) }
    }

    fn to_json(&self) -> Value {
        let mut v = json!({"id": self.id, "name": self.name, "kind": self.kind.id(), "default": self.default});
        if !self.choices.is_empty() {
            v["choices"] = json!(self.choices);
        }
        if let Some(m) = self.min {
            v["min"] = json!(m);
        }
        if let Some(m) = self.max {
            v["max"] = json!(m);
        }
        if !self.help.is_empty() {
            v["help"] = json!(self.help);
        }
        v
    }
}

fn read_step(v: &Value, at: &str, errs: &mut Vec<String>) -> Option<Step> {
    let Some(o) = v.as_object() else {
        errs.push(format!("{at}: expected an object"));
        return None;
    };
    unknown(o, &["label", "command", "intent", "params", "when"], at, errs);
    let action = match (text(o, "command", 128, at, errs), text(o, "intent", 64, at, errs)) {
        (Some(c), None) => Action::Command(c),
        (None, Some(i)) => Action::Intent(i),
        _ => {
            errs.push(format!("{at}: give `command` (an engine command id) or `intent` (an AutoCut intent), one of them"));
            return None;
        }
    };
    let params = o.get("params").cloned().unwrap_or_else(|| json!({}));
    if !params.is_object() {
        errs.push(format!("{at}.params: expected an object"));
    } else if depth(&params) > MAX_DEPTH {
        errs.push(format!("{at}.params: nested deeper than {MAX_DEPTH}"));
    } else if serde_json::to_vec(&params).map_or(true, |b| b.len() > MAX_STEP_PARAMS_BYTES) {
        errs.push(format!("{at}.params: larger than {MAX_STEP_PARAMS_BYTES} bytes"));
    }
    let label = text(o, "label", 200, at, errs).unwrap_or_else(|| match &action {
        Action::Command(c) => c.clone(),
        Action::Intent(i) => i.replace('_', " "),
    });
    Some(Step { label, action, params, when: text(o, "when", 64, at, errs) })
}

impl Recipe {
    /// Read a recipe (JSON), checking the format, sizes, ids, parameters and steps; every problem
    /// is reported at once. Commands and intents are checked against this build by
    /// [`Recipe::check`].
    pub fn from_value(v: &Value) -> std::result::Result<Recipe, Vec<String>> {
        let mut errs = Vec::new();
        let Some(o) = v.as_object() else { return Err(vec!["a recipe is a JSON object".into()]) };
        match o.get("format").and_then(Value::as_str) {
            Some(FORMAT) => {}
            _ => return Err(vec![format!("not a FilmCraft recipe (`format` is not \"{FORMAT}\")")]),
        }
        match o.get("formatVersion").and_then(Value::as_u64) {
            Some(n) if n > FORMAT_VERSION => return Err(vec![format!("made by a newer FilmCraft (format version {n}; this build reads {FORMAT_VERSION})")]),
            Some(n) if n >= 1 => {}
            _ => errs.push("formatVersion: expected 1".into()),
        }
        unknown(o, &["format", "formatVersion", "id", "name", "description", "params", "steps"], "recipe", &mut errs);
        let id = text(o, "id", 128, "recipe", &mut errs).unwrap_or_default();
        if !is_id(&id) {
            errs.push(format!("id: `{id}`: lower-case letters, digits and `:._-`"));
        }
        let name = text(o, "name", 200, "recipe", &mut errs).unwrap_or_default();
        if name.trim().is_empty() {
            errs.push("name: is empty".into());
        }
        let description = text(o, "description", 4000, "recipe", &mut errs).unwrap_or_default();
        let mut params = Vec::new();
        match o.get("params") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) if a.len() <= MAX_PARAMS => {
                for (i, p) in a.iter().enumerate() {
                    if let Some(p) = read_param(p, &format!("params[{i}]"), &mut errs) {
                        if params.iter().any(|x: &Param| x.id == p.id) {
                            errs.push(format!("params[{i}].id: `{}` is used twice", p.id));
                        }
                        params.push(p);
                    }
                }
            }
            Some(_) => errs.push(format!("params: a list of at most {MAX_PARAMS}")),
        }
        let mut steps = Vec::new();
        match o.get("steps") {
            Some(Value::Array(a)) if !a.is_empty() && a.len() <= MAX_STEPS => {
                for (i, s) in a.iter().enumerate() {
                    if let Some(s) = read_step(s, &format!("steps[{i}]"), &mut errs) {
                        steps.push(s);
                    }
                }
            }
            _ => errs.push(format!("steps: a list of 1 to {MAX_STEPS} steps")),
        }
        // references: `${id}` and `when` name parameters
        for (i, s) in steps.iter().enumerate() {
            let mut refs = Vec::new();
            references(&s.params, &mut refs);
            for r in refs {
                if !params.iter().any(|p| p.id == r) {
                    errs.push(format!("steps[{i}].params: `${{{r}}}` is not a parameter"));
                }
            }
            if let Some(w) = &s.when
                && !params.iter().any(|p| &p.id == w)
            {
                errs.push(format!("steps[{i}].when: `{w}` is not a parameter"));
            }
        }
        errs.truncate(100);
        if errs.is_empty() { Ok(Recipe { id, name, description, params, steps }) } else { Err(errs) }
    }

    /// Read a recipe file.
    pub fn parse(bytes: &[u8]) -> std::result::Result<Recipe, Vec<String>> {
        if bytes.len() > MAX_RECIPE_BYTES {
            return Err(vec![format!("the recipe is {} bytes (at most {MAX_RECIPE_BYTES})", bytes.len())]);
        }
        let v: Value = serde_json::from_slice(bytes).map_err(|e| vec![format!("not JSON: {e}")])?;
        Recipe::from_value(&v)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "format": FORMAT,
            "formatVersion": FORMAT_VERSION,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "params": self.params.iter().map(Param::to_json).collect::<Vec<_>>(),
            "steps": self.steps.iter().map(|s| {
                let mut v = json!({"label": s.label, "params": s.params});
                match &s.action {
                    Action::Command(c) => v["command"] = json!(c),
                    Action::Intent(i) => v["intent"] = json!(i),
                }
                if let Some(w) = &s.when {
                    v["when"] = json!(w);
                }
                v
            }).collect::<Vec<_>>(),
        })
    }

    /// The parameter values: the defaults, then `given` (each checked; unknown names refused).
    pub fn values(&self, given: Option<&Value>) -> std::result::Result<BTreeMap<String, Value>, String> {
        let mut out: BTreeMap<String, Value> = self.params.iter().map(|p| (p.id.clone(), p.default.clone())).collect();
        if let Some(g) = given.filter(|g| !g.is_null()) {
            let o = g.as_object().ok_or("`params` is an object: {\"<parameter>\": value}")?;
            for (k, v) in o {
                let p = self.params.iter().find(|p| &p.id == k).ok_or_else(|| {
                    let ids: Vec<&str> = self.params.iter().map(|p| p.id.as_str()).collect();
                    format!("the recipe has no parameter `{k}` (its parameters: {})", ids.join(", "))
                })?;
                p.check(v).map_err(|e| format!("`{k}`: {e}"))?;
                out.insert(k.clone(), v.clone());
            }
        }
        Ok(out)
    }

    /// The steps as they run with `values`: parameters filled in, those switched off marked,
    /// every command checked against the registry and the allowlist, every intent against the
    /// planner's. `Err` names the step.
    pub fn resolve(&self, values: &BTreeMap<String, Value>, intents: Option<&[String]>) -> std::result::Result<Vec<Resolved>, String> {
        let mut out = Vec::new();
        for (i, s) in self.steps.iter().enumerate() {
            let at = format!("step {} ({})", i + 1, s.label);
            let on = s.when.as_ref().is_none_or(|w| truthy(values.get(w).unwrap_or(&Value::Null)));
            let params = substitute(&s.params, values);
            if on {
                match &s.action {
                    Action::Command(c) => {
                        check_command(c, &params).map_err(|e| format!("{at}: {e}"))?;
                    }
                    Action::Intent(intent) => match intents {
                        None => return Err(format!("{at}: AutoCut intents need the AutoCut planner, which this app does not have")),
                        Some(list) if !list.iter().any(|x| x == intent) => {
                            return Err(format!("{at}: `{intent}` is not an AutoCut intent (one of {})", list.join(", ")));
                        }
                        Some(_) => {}
                    },
                }
            }
            out.push(Resolved { index: i, label: s.label.clone(), action: s.action.clone(), params, skipped: !on });
        }
        Ok(out)
    }
}

/// Whether a parameter value switches a step on.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.trim().is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn as_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Fill `${id}` references with the values.
fn substitute(v: &Value, values: &BTreeMap<String, Value>) -> Value {
    match v {
        Value::String(s) => {
            if let Some(id) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}'))
                && !id.contains("${")
                && let Some(x) = values.get(id)
            {
                return x.clone();
            }
            if !s.contains("${") {
                return v.clone();
            }
            let mut out = String::new();
            let mut rest = s.as_str();
            while let Some(i) = rest.find("${") {
                out.push_str(rest.get(..i).unwrap_or(""));
                let after = rest.get(i + 2..).unwrap_or("");
                match after.find('}') {
                    Some(j) => {
                        let id = after.get(..j).unwrap_or("");
                        out.push_str(&values.get(id).map(as_text).unwrap_or_default());
                        rest = after.get(j + 1..).unwrap_or("");
                    }
                    None => {
                        out.push_str(rest.get(i..).unwrap_or(""));
                        rest = "";
                    }
                }
                if out.len() > 64 * 1024 {
                    break;
                }
            }
            out.push_str(rest);
            Value::String(out)
        }
        // (depth is bounded when the recipe is read)
        Value::Array(a) => Value::Array(a.iter().map(|x| substitute(x, values)).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), substitute(x, values))).collect()),
        other => other.clone(),
    }
}

/// A command a recipe runs: in the registry, in the allowlist, parameters an object, any `path`
/// filled in.
pub fn check_command(id: &str, params: &Value) -> std::result::Result<(), String> {
    if crate::commands::find(id).is_none() {
        return Err(format!("unknown command `{id}`"));
    }
    if !allowed(id) {
        return Err(format!("`{id}` is not a command a recipe may run"));
    }
    if !params.is_object() && !params.is_null() {
        return Err(format!("`{id}`: parameters must be an object"));
    }
    if params.get("path").and_then(Value::as_str).is_some_and(|p| p.trim().is_empty()) {
        return Err(format!("`{id}`: `path` is empty (set the parameter it comes from)"));
    }
    Ok(())
}

/// A step as it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub index: usize,
    pub label: String,
    pub action: Action,
    pub params: Value,
    pub skipped: bool,
}

impl Resolved {
    pub fn to_json(&self) -> Value {
        let mut v = json!({"step": self.index + 1, "label": self.label, "params": self.params, "skipped": self.skipped});
        match &self.action {
            Action::Command(c) => v["command"] = json!(c),
            Action::Intent(i) => v["intent"] = json!(i),
        }
        v
    }
}

// ------------------------------------------------------------------------------------------------
// Planning intents
// ------------------------------------------------------------------------------------------------

/// One command an intent was planned into.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedCommand {
    pub id: String,
    pub params: Value,
    pub why: String,
}

/// What the planner made of an intent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Planned {
    pub commands: Vec<PlannedCommand>,
    pub notes: Vec<String>,
    /// Run these, then plan the intent again (a transcript is made first).
    pub replan: bool,
}

/// Plans AutoCut intents for recipe steps, installed by the host (filmcraft-autocut's planner,
/// whose own validator checks every planned command). None: intent steps are refused.
pub trait RecipePlanner: Send + Sync {
    /// The intents it plans.
    fn intents(&self) -> Vec<String>;
    /// Plan `intent` with `params` against the session behind `port` (read with its commands).
    /// Blocking, on the recipe job's thread.
    fn plan(&self, intent: &str, params: &Value, port: &CommandPort) -> std::result::Result<Planned, String>;
}

// ------------------------------------------------------------------------------------------------
// The session's recipes
// ------------------------------------------------------------------------------------------------

/// Edits being recorded ("Save this edit as a recipe").
#[derive(Clone, Debug, Default)]
pub struct Recording {
    pub steps: Vec<(String, Value)>,
}

/// Recipes in a session: the host's planner, recipes kept without a data directory, the jobs'
/// outcomes, the recording.
#[derive(Default)]
pub struct Recipes {
    pub planner: Option<Arc<dyn RecipePlanner>>,
    /// User recipes when there is no data directory (tests, the web).
    pub memory: BTreeMap<String, Value>,
    pub outcomes: HashMap<u64, Arc<Mutex<Option<Value>>>>,
    pub recording: Option<Recording>,
    /// The recipe job running.
    pub running: Option<u64>,
}

/// Most edits one recording keeps.
const MAX_RECORDED: usize = 200;

/// Record a finished top-level edit (from [`Session::execute`]) while recording.
pub fn observe(s: &mut Session, id: &str, params: &Value) {
    let Some(r) = s.recipes.recording.as_mut() else { return };
    if !allowed(id) || r.steps.len() >= MAX_RECORDED || serde_json::to_vec(params).map_or(true, |b| b.len() > MAX_STEP_PARAMS_BYTES) {
        return;
    }
    r.steps.push((id.to_string(), params.clone()));
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Bumped when a recipe is saved or deleted (the panel reads the folder again).
static REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The recipes folder's revision ([`bump_library`]).
pub fn library_revision() -> u64 {
    REVISION.load(Ordering::Relaxed)
}

/// Note that the recipes folder changed.
pub fn bump_library() {
    REVISION.fetch_add(1, Ordering::Relaxed);
}

/// Where user recipes live: `<data dir>/Recipes` (None without a data directory).
pub fn user_dir(s: &Session) -> Option<std::path::PathBuf> {
    s.prefs_path.as_ref().and_then(|p| p.parent()).map(|d| d.join(USER_FOLDER))
}

/// A recipe from the library.
#[derive(Clone, Debug)]
pub struct Entry {
    pub recipe: Recipe,
    /// `builtin` or `user`.
    pub source: &'static str,
    pub path: Option<String>,
}

/// The built-in recipes.
pub fn builtin() -> Vec<Recipe> {
    builtin_json().iter().filter_map(|v| Recipe::from_value(v).ok()).collect()
}

/// The built-in recipes as written (the tests check that each reads).
pub fn builtin_json() -> Vec<Value> {
    vec![
        json!({
            "format": FORMAT, "formatVersion": 1, "id": "builtin:talking-head-clean-up", "name": "Talking-head clean-up",
            "description": "Cut the pauses and the filler words, clean up the voice, and caption what is said: the open sequence of someone talking to the camera, ready to post.",
            "params": [
                {"id": "minSilence", "name": "Shortest pause cut (seconds)", "kind": "number", "default": 0.5, "min": 0.1, "max": 10},
                {"id": "fillers", "name": "Cut filler words", "kind": "bool", "default": true},
                {"id": "enhance", "name": "Clean up the voice", "kind": "bool", "default": true},
                {"id": "captionStyle", "name": "Caption style", "kind": "choice", "choices": ["default", "bold", "clean", "tiktok"], "default": "bold"},
            ],
            "steps": [
                {"label": "Remove the silences", "intent": "remove_silence", "params": {"minSeconds": "${minSilence}"}},
                {"label": "Remove the filler words", "intent": "remove_fillers", "params": {}, "when": "fillers"},
                {"label": "Clean up the voice", "intent": "enhance_voice", "params": {"preset": "clean"}, "when": "enhance"},
                {"label": "Captions in my style", "intent": "add_captions", "params": {"style": "${captionStyle}"}},
            ],
        }),
        json!({
            "format": FORMAT, "formatVersion": 1, "id": "builtin:vertical-short", "name": "Vertical short from a long clip",
            "description": "Keep the best spoken moments of a long clip, turn the sequence vertical (9:16, the picture filling the frame), caption it and export it for Shorts and Reels. Undo brings the long cut back.",
            "params": [
                {"id": "seconds", "name": "Length (seconds)", "kind": "number", "default": 45, "min": 5, "max": 180},
                {"id": "captions", "name": "Captions", "kind": "bool", "default": true},
                {"id": "output", "name": "Export to", "kind": "path", "default": "", "help": "Leave empty to keep the short in the timeline without exporting it."},
            ],
            "steps": [
                {"label": "Keep the best moments", "intent": "highlight_reel", "params": {"targetSeconds": "${seconds}"}},
                {"label": "Make it vertical", "command": "sequence.settings", "params": {"width": 1080, "height": 1920}},
                {"label": "Select every clip", "command": "edit.selectAll", "params": {}},
                {"label": "Fill the vertical frame", "command": "clip.fillFrame", "params": {}},
                {"label": "Clear the selection", "command": "edit.deselectAll", "params": {}},
                {"label": "Captions", "intent": "add_captions", "params": {"style": "tiktok"}, "when": "captions"},
                {"label": "Export", "command": "file.exportMedia", "params": {"path": "${output}", "preset": "Social Vertical 1080×1920"}, "when": "output"},
            ],
        }),
        json!({
            "format": FORMAT, "formatVersion": 1, "id": "builtin:template-publish-youtube", "name": "Apply a template and publish privately to YouTube",
            "description": "Make a new sequence from a video template, export it, and — only when you switch Publish on — upload it to your YouTube channel as a private video.",
            "params": [
                {"id": "template", "name": "Template", "kind": "template", "default": "builtin:talking-head-captions"},
                {"id": "variant", "name": "Layout", "kind": "choice", "choices": ["16:9", "9:16", "1:1", "4:5"], "default": "16:9"},
                {"id": "slots", "name": "What fills the slots", "kind": "json", "default": {}, "help": "As template.apply's `slots`: {\"speaker\": \"/path/clip.mov\"}"},
                {"id": "output", "name": "Export to", "kind": "path", "default": ""},
                {"id": "preset", "name": "Export preset", "kind": "text", "default": "YouTube 1080p Full HD"},
                {"id": "title", "name": "Video title", "kind": "text", "default": ""},
                {"id": "publish", "name": "Publish to YouTube (private)", "kind": "bool", "default": false},
            ],
            "steps": [
                {"label": "Apply the template", "command": "template.apply", "params": {"template": "${template}", "variant": "${variant}", "slots": "${slots}"}},
                {"label": "Export", "command": "file.exportMedia", "params": {"path": "${output}", "preset": "${preset}"}, "when": "output"},
                {"label": "Publish privately to YouTube", "command": "publish.run", "params": {"destination": "youtube", "path": "${output}", "privacy": "private", "title": "${title}"}, "when": "publish"},
            ],
        }),
        json!({"format":FORMAT,"formatVersion":1,"id":"builtin:delivery-quality-gate","name":"Checked 4K delivery",
            "description":"Check privacy at cuts, changed narration, subtitle alignment, screens, motion and audio continuity before a cached 4K delivery. No upload.",
            "params":[{"id":"output","name":"New output file","kind":"path","default":""},{"id":"short","name":"Short with burned captions","kind":"bool","default":false},{"id":"qa","name":"Explicit QA policy","kind":"json","default":{}}],
            "steps":[{"label":"Quality gate and 4K export","command":"delivery.prepare","params":{"path":"${output}","short":"${short}","qa":"${qa}"}}]}),
    ]
}

/// Built-in and user recipes, and the user files that could not be read (path, why).
pub fn library(s: &Session) -> (Vec<Entry>, Vec<(String, String)>) {
    let mut v: Vec<Entry> = builtin().into_iter().map(|recipe| Entry { recipe, source: "builtin", path: None }).collect();
    let mut invalid = Vec::new();
    match user_dir(s) {
        Some(dir) => {
            let dir = dir.to_string_lossy().to_string();
            if let Some(Ok(mut entries)) = s.services.list_entries(&dir) {
                entries.sort_by(|a, b| a.name.cmp(&b.name));
                for e in entries.into_iter().filter(|e| !e.is_dir && !e.name.starts_with('.') && e.name.to_ascii_lowercase().ends_with(".json")).take(MAX_FILES)
                {
                    let path = format!("{dir}/{}", e.name);
                    let read = s.services.file_size(&path).map_err(|e| e.to_string()).and_then(|n| {
                        if n as usize > MAX_RECIPE_BYTES {
                            Err(format!("{n} bytes (at most {MAX_RECIPE_BYTES})"))
                        } else {
                            s.services.read_file(&path).map_err(|e| e.to_string())
                        }
                    });
                    match read.and_then(|b| Recipe::parse(&b).map_err(|e| e.join("; "))) {
                        Ok(r) if r.id.starts_with("builtin:") => invalid.push((path, "a user recipe cannot use a built-in id".into())),
                        Ok(recipe) => v.push(Entry { recipe, source: "user", path: Some(path) }),
                        Err(why) => invalid.push((path, why)),
                    }
                }
            }
        }
        None => {
            for x in s.recipes.memory.values() {
                if let Ok(recipe) = Recipe::from_value(x) {
                    v.push(Entry { recipe, source: "user", path: None });
                }
            }
        }
    }
    (v, invalid)
}

/// A recipe by id or name (case-insensitive).
pub fn find(s: &Session, key: &str, cmd: &str) -> Result<Entry> {
    let key = key.trim();
    if key.is_empty() {
        return Err(bad(cmd, "need `name` (a recipe id or name; see recipe.list)"));
    }
    let (lib, _) = library(s);
    let by_id = lib.iter().find(|e| e.recipe.id == key).cloned();
    by_id.or_else(|| lib.into_iter().find(|e| e.recipe.name.eq_ignore_ascii_case(key))).ok_or_else(|| bad(cmd, format!("no recipe `{key}` (see recipe.list)")))
}

fn summary(e: &Entry) -> Value {
    let r = &e.recipe;
    json!({
        "id": r.id, "name": r.name, "description": r.description, "source": e.source, "path": e.path,
        "params": r.params.iter().map(Param::to_json).collect::<Vec<_>>(),
        "steps": r.steps.len(),
        "intents": r.steps.iter().filter(|s| matches!(s.action, Action::Intent(_))).count(),
    })
}

fn intents(s: &Session) -> Option<Vec<String>> {
    s.recipes.planner.as_ref().map(|p| p.intents())
}

// ------------------------------------------------------------------------------------------------
// Commands
// ------------------------------------------------------------------------------------------------

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec("recipe.list", "List Recipes", "{}", list),
        spec("recipe.inspect", "Inspect Recipe", r#"{"name":id|name,"params":{…}?}"#, inspect),
        spec(
            "recipe.save",
            "Save Recipe",
            r#"{"name":str,"steps":[{"label":str?,"command":id|"intent":str,"params":{…}?,"when":param?}],"params":[{"id","name","kind":"bool"|"number"|"text"|"choice"|"path"|"template"|"json","default","choices"?,"min"?,"max"?}]?,"description":str?,"id":str?,"overwrite":bool=false}"#,
            save,
        ),
        spec("recipe.run", "Run Recipe", r#"{"name":id|name?,"recipe":{…}?,"params":{…}?,"dryRun":bool=false,"wait":bool=false}"#, run),
        spec("recipe.delete", "Delete Recipe", r#"{"name":id|name}"#, delete),
        spec("recipe.record", "Record Recipe", r#"{"action":"start"|"stop"|"cancel"|"status","name":str?,"description":str?}"#, record),
    ]
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let (lib, invalid) = library(s);
    Ok(json!({
        "recipes": lib.iter().map(summary).collect::<Vec<_>>(),
        "invalid": invalid.iter().map(|(p, e)| json!({"path": p, "error": e})).collect::<Vec<_>>(),
        "userFolder": user_dir(s).map(|d| d.to_string_lossy().to_string()),
        "intents": intents(s).unwrap_or_default(),
        "recording": s.recipes.recording.as_ref().map(|r| r.steps.len()),
        "running": s.recipes.running,
    }))
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "recipe.inspect";
    let e = find(s, str_p(p, "name").or_else(|| str_p(p, "id")).unwrap_or(""), CMD)?;
    let mut v = summary(&e);
    v["recipe"] = e.recipe.to_json();
    let plan = e.recipe.values(p.get("params")).and_then(|vals| e.recipe.resolve(&vals, intents(s).as_deref()));
    match plan {
        Ok(steps) => {
            v["plan"] = Value::Array(steps.iter().map(Resolved::to_json).collect());
            v["ready"] = json!(true);
        }
        Err(why) => {
            v["ready"] = json!(false);
            v["problem"] = json!(why);
        }
    }
    Ok(v)
}

/// A recipe from `recipe.save`'s parameters.
fn recipe_of(p: &Value, cmd: &str) -> Result<Recipe> {
    let name = str_p(p, "name").map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad(cmd, "need `name`"))?;
    let id = match str_p(p, "id") {
        Some(id) if id.starts_with("builtin:") => return Err(bad(cmd, "`builtin:` ids are the built-in recipes'")),
        Some(id) => id.to_string(),
        None => format!("user:{}", filmcraft_templates::slug(name)),
    };
    let v = json!({
        "format": FORMAT, "formatVersion": FORMAT_VERSION, "id": id, "name": name,
        "description": p.get("description").cloned().unwrap_or(json!("")),
        "params": p.get("params").cloned().unwrap_or(json!([])),
        "steps": p.get("steps").cloned().unwrap_or(Value::Null),
    });
    Recipe::from_value(&v).map_err(|e| bad(cmd, e.join("; ")))
}

/// Write `r` into the recipes folder (or the session's memory).
fn store(s: &mut Session, r: &Recipe, overwrite: bool, cmd: &str) -> Result<Option<String>> {
    // its commands must exist and be allowed now; intents are checked when it runs
    for (i, st) in r.steps.iter().enumerate() {
        if let Action::Command(c) = &st.action {
            if crate::commands::find(c).is_none() {
                return Err(bad(cmd, format!("step {} ({}): unknown command `{c}`", i + 1, st.label)));
            }
            if !allowed(c) {
                return Err(bad(cmd, format!("step {} ({}): `{c}` is not a command a recipe may run", i + 1, st.label)));
            }
        }
    }
    let (lib, _) = library(s);
    let existing = lib.iter().find(|e| e.recipe.id == r.id && e.source == "user").cloned();
    if existing.is_some() && !overwrite {
        return Err(bad(cmd, format!("a recipe `{}` exists (pass `overwrite: true` to replace it)", r.id)));
    }
    let bytes = serde_json::to_vec_pretty(&r.to_json()).map_err(|e| bad(cmd, e.to_string()))?;
    let Some(dir) = user_dir(s) else {
        s.recipes.memory.insert(r.id.clone(), r.to_json());
        bump_library();
        return Ok(None);
    };
    let path = match existing.and_then(|e| e.path) {
        Some(p) => p,
        None => {
            let stem = filmcraft_templates::slug(r.id.split_once(':').map_or(r.id.as_str(), |(_, x)| x));
            let mut path = dir.join(format!("{stem}.json"));
            let mut n = 2;
            while s.services.file_size(&path.to_string_lossy()).is_ok() && n < 1000 {
                path = dir.join(format!("{stem}-{n}.json"));
                n += 1;
            }
            path.to_string_lossy().to_string()
        }
    };
    if !cfg!(target_arch = "wasm32") {
        std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{}: {e}", dir.display())))?;
    }
    s.services.write_file(&path, &bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    bump_library();
    Ok(Some(path))
}

fn save(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "recipe.save";
    let r = recipe_of(p, CMD)?;
    let path = store(s, &r, bool_p(p, "overwrite").unwrap_or(false), CMD)?;
    let e = Entry { recipe: r, source: "user", path: path.clone() };
    Ok(json!({"recipe": summary(&e), "path": path}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "recipe.delete";
    let e = find(s, str_p(p, "name").or_else(|| str_p(p, "id")).unwrap_or(""), CMD)?;
    if e.source != "user" {
        return Err(bad(CMD, format!("“{}” is built in; only your own recipes can be deleted", e.recipe.name)));
    }
    match &e.path {
        Some(path) => {
            if cfg!(target_arch = "wasm32") {
                return Err(bad(CMD, "deleting files is not available here"));
            }
            std::fs::remove_file(path).map_err(|err| EngineError::Other(format!("{path}: {err}")))?;
        }
        None => {
            s.recipes.memory.remove(&e.recipe.id);
        }
    }
    bump_library();
    Ok(json!({"deleted": e.recipe.id, "path": e.path}))
}

/// Params of a recorded edit that name things of this project (clips, items, tracks).
fn names_this_project(params: &Value) -> bool {
    params
        .as_object()
        .is_some_and(|o| o.keys().any(|k| matches!(k.as_str(), "clip" | "clips" | "item" | "items" | "sequence" | "track" | "tracks" | "marker" | "markers")))
}

fn record(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "recipe.record";
    match str_p(p, "action").unwrap_or("status") {
        "start" => {
            s.recipes.recording = Some(Recording::default());
            Ok(json!({"recording": true, "steps": 0}))
        }
        "cancel" => {
            let n = s.recipes.recording.take().map_or(0, |r| r.steps.len());
            Ok(json!({"recording": false, "discarded": n}))
        }
        "status" => Ok(json!({
            "recording": s.recipes.recording.is_some(),
            "steps": s.recipes.recording.as_ref().map_or(Vec::new(), |r| r.steps.iter().map(|(id, params)| json!({"command": id, "params": params})).collect()),
        })),
        "stop" => {
            let r = s.recipes.recording.take().ok_or_else(|| bad(CMD, "nothing is being recorded (recipe.record {\"action\": \"start\"})"))?;
            let steps: Vec<Value> = r
                .steps
                .iter()
                .map(|(id, params)| {
                    let label = crate::commands::find(id).map_or(id.as_str(), |c| c.label).trim_end_matches('…').to_string();
                    json!({"label": label, "command": id, "params": params})
                })
                .collect();
            let specific: Vec<usize> = r.steps.iter().enumerate().filter(|(_, (_, params))| names_this_project(params)).map(|(i, _)| i + 1).collect();
            let mut out = json!({"recording": false, "steps": steps});
            if !specific.is_empty() {
                out["warnings"] = json!([format!(
                    "steps {} name clips or items of this project; they act on the same ids when the recipe runs elsewhere — edit them (recipe.save) or select first",
                    specific.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
                )]);
            }
            if steps.is_empty() {
                return Ok(out);
            }
            if let Some(name) = str_p(p, "name").filter(|n| !n.trim().is_empty()) {
                let q = json!({"name": name, "description": p.get("description").cloned().unwrap_or(json!("")), "steps": steps});
                let saved = save(s, &q)?;
                out["recipe"] = saved["recipe"].clone();
                out["path"] = saved["path"].clone();
            }
            Ok(out)
        }
        other => Err(bad(CMD, format!("`action` is start, stop, cancel or status, not `{other}`"))),
    }
}

// ------------------------------------------------------------------------------------------------
// Running
// ------------------------------------------------------------------------------------------------

/// Whether a recipe job is running.
fn running(s: &Session) -> Option<u64> {
    let id = s.recipes.running?;
    s.jobs.iter().find(|j| j.id == id).filter(|j| !j.progress.finished.load(Ordering::Relaxed)).map(|j| j.id)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "recipe.run";
    let recipe = match (p.get("recipe").filter(|v| !v.is_null()), str_p(p, "name").or_else(|| str_p(p, "id"))) {
        (Some(r), None) => Recipe::from_value(r).map_err(|e| bad(CMD, e.join("; ")))?,
        (None, Some(key)) => find(s, key, CMD)?.recipe,
        (Some(_), Some(_)) => return Err(bad(CMD, "give `name` or an inline `recipe`, not both")),
        (None, None) => return Err(bad(CMD, "need `name` (a recipe id or name; see recipe.list)")),
    };
    let values = recipe.values(p.get("params")).map_err(|e| bad(CMD, e))?;
    let steps = recipe.resolve(&values, intents(s).as_deref()).map_err(|e| bad(CMD, e))?;
    if bool_p(p, "dryRun").unwrap_or(false) {
        return Ok(json!({"recipe": recipe.id, "plan": steps.iter().map(Resolved::to_json).collect::<Vec<_>>(), "params": values}));
    }
    if cfg!(target_arch = "wasm32") {
        return Err(bad(CMD, "recipes are not available on the web"));
    }
    if let Some(id) = running(s) {
        return Err(bad(CMD, format!("a recipe is already running (job {id}); wait for it or cancel it (jobs.cancel)")));
    }
    if s.recipes.recording.is_some() {
        return Err(bad(CMD, "a recording is in progress; stop it first (recipe.record {\"action\": \"stop\"})"));
    }
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Recipe: {}", recipe.name));
    let total = (steps.len() as u64).saturating_mul(100).max(1);
    job.progress.total.store(total, Ordering::Relaxed);
    *lock(&job.progress.status) = "Starting…".into();
    let id = job.id;
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome: Arc<Mutex<Option<Value>>> = Arc::default();
    s.recipes.outcomes.insert(id, outcome.clone());
    let port = s.autocut.port.handle(Some(prog.clone()));
    let planner = s.recipes.planner.clone();
    let name = recipe.name.clone();
    let rid = recipe.id.clone();
    let body = crate::export_tools::guard_job(prog.clone(), res.clone(), move || {
        let t0 = web_time::Instant::now();
        let mut log: Vec<Value> = Vec::new();
        let r = run_steps(&steps, &port, planner.as_deref(), &prog, &mut log);
        let cancelled = prog.cancel.load(Ordering::Relaxed);
        let (status, result) = match r {
            Ok(path) => {
                let st = format!("“{name}”: {} steps done", steps.iter().filter(|x| !x.skipped).count());
                (
                    st.clone(),
                    Ok(filmcraft_export::Report { path, frames: 0, seconds: t0.elapsed().as_secs_f64(), bytes: 0, render_fps: 0.0, extra_files: Vec::new() }),
                )
            }
            Err(e) => {
                let e = if cancelled { "cancelled".to_string() } else { e };
                (e.clone(), Err(e))
            }
        };
        *lock(&outcome) = Some(json!({"recipe": rid, "name": name, "steps": log, "status": status, "ok": result.is_ok()}));
        *lock(&prog.status) = status;
        if let Err(e) = &result {
            *lock(&prog.error) = Some(e.clone());
        }
        prog.finished.store(true, Ordering::Relaxed);
        *lock(&res) = Some(result);
    });
    s.jobs.push(job);
    s.recipes.running = Some(id);
    if let Err(e) = std::thread::Builder::new().name("filmcraft-recipe".into()).spawn(body) {
        if let Some(j) = s.jobs.iter().find(|j| j.id == id) {
            *lock(&j.progress.error) = Some(e.to_string());
            j.progress.finished.store(true, Ordering::Relaxed);
            *lock(&j.result) = Some(Err(e.to_string()));
        }
        return Err(EngineError::Other(format!("{CMD}: could not start the job: {e}")));
    }
    let mut out = json!({"job": id, "recipe": recipe.id});
    if wait {
        wait_for(s, id);
        if let Some(j) = s.jobs.iter().find(|j| j.id == id) {
            let v = j.to_json();
            out["result"] = v["result"].clone();
            out["recipeRun"] = outcome_json(s, id);
            if let Some(e) = v["result"]["error"].as_str() {
                return Err(EngineError::Other(format!("{CMD}: {e}")));
            }
        }
    }
    Ok(out)
}

/// Serve the job's commands until it finishes (a `wait: true` call: nobody else pumps the port).
fn wait_for(s: &mut Session, job: u64) {
    loop {
        let done = s.jobs.iter().find(|j| j.id == job).is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        if done {
            break;
        }
        if s.pump_port(Duration::from_millis(20)) == 0 {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

/// Job `job`'s outcome (`null` while it runs, or for another job).
pub fn outcome_json(s: &Session, job: u64) -> Value {
    s.recipes.outcomes.get(&job).and_then(|o| lock(o).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` is a recipe run.
pub fn is_job(s: &Session, job: u64) -> bool {
    s.recipes.outcomes.contains_key(&job)
}

fn set_status(prog: &filmcraft_export::Progress, text: String) {
    *lock(&prog.status) = text;
}

/// Run the steps on the job's thread; `Ok` is the last export's path (or empty).
fn run_steps(
    steps: &[Resolved],
    port: &CommandPort,
    planner: Option<&dyn RecipePlanner>,
    prog: &Arc<filmcraft_export::Progress>,
    log: &mut Vec<Value>,
) -> std::result::Result<String, String> {
    let n = steps.len();
    let mut last_path = String::new();
    for (i, st) in steps.iter().enumerate() {
        if port.cancelled() {
            return Err("cancelled".into());
        }
        let base = (i as u64).saturating_mul(100);
        prog.done.store(base, Ordering::Relaxed);
        let at = format!("step {} of {n} ({})", i + 1, st.label);
        if st.skipped {
            log.push(json!({"step": i + 1, "label": st.label, "skipped": true}));
            continue;
        }
        set_status(prog, format!("{}/{n} {}", i + 1, st.label));
        let mut ran: Vec<Value> = Vec::new();
        let r = run_step(st, port, planner, prog, base, &mut ran, &mut last_path);
        let notes = match &r {
            Ok(notes) => notes.clone(),
            Err(_) => Vec::new(),
        };
        let mut entry = json!({"step": i + 1, "label": st.label, "commands": ran, "ok": r.is_ok()});
        if !notes.is_empty() {
            entry["notes"] = json!(notes);
        }
        if let Err(e) = &r {
            entry["error"] = json!(e);
            log.push(entry);
            return Err(format!("{at} failed: {e}"));
        }
        log.push(entry);
    }
    prog.done.store((n as u64).saturating_mul(100), Ordering::Relaxed);
    Ok(last_path)
}

/// Run one step: plan an intent (checking what it plans), run each command, wait for the jobs
/// they start. Returns the planner's notes.
fn run_step(
    st: &Resolved,
    port: &CommandPort,
    planner: Option<&dyn RecipePlanner>,
    prog: &Arc<filmcraft_export::Progress>,
    base: u64,
    ran: &mut Vec<Value>,
    last_path: &mut String,
) -> std::result::Result<Vec<String>, String> {
    match &st.action {
        Action::Command(id) => {
            check_command(id, &st.params)?;
            run_command(id, st.params.clone(), port, prog, base, ran, last_path)?;
            Ok(Vec::new())
        }
        Action::Intent(intent) => {
            let planner = planner.ok_or("AutoCut intents need the AutoCut planner, which this app does not have")?;
            let mut notes = Vec::new();
            // a transcript first (highlight reels): run what the planner says, then plan again
            for _round in 0..3 {
                let plan = planner.plan(intent, &st.params, port)?;
                if plan.commands.len() > MAX_PLANNED {
                    return Err(format!("`{intent}` planned {} commands (at most {MAX_PLANNED})", plan.commands.len()));
                }
                // every planned command is checked before any of them runs
                for c in &plan.commands {
                    check_command(&c.id, &c.params).map_err(|e| format!("the planner's `{}`: {e}", c.id))?;
                }
                notes.extend(plan.notes.iter().cloned());
                for c in &plan.commands {
                    run_command(&c.id, c.params.clone(), port, prog, base, ran, last_path)?;
                }
                if !plan.replan {
                    return Ok(notes);
                }
            }
            Err(format!("`{intent}` could not be planned (it kept asking for another round)"))
        }
    }
}

/// Run one command through the port; a command that starts a job is waited for (its progress
/// shown in the step's share of the run, a cancel passed on).
fn run_command(
    id: &str,
    mut params: Value,
    port: &CommandPort,
    prog: &Arc<filmcraft_export::Progress>,
    base: u64,
    ran: &mut Vec<Value>,
    last_path: &mut String,
) -> std::result::Result<(), String> {
    if port.cancelled() {
        return Err("cancelled".into());
    }
    // jobs run in the background and are waited for here, so the session stays responsive
    let takes_wait = crate::commands::find(id).is_some_and(|c| c.params.contains("\"wait\""));
    if takes_wait && let Some(o) = params.as_object_mut() {
        o.insert("wait".into(), json!(false));
    }
    if params.is_null() {
        params = json!({});
    }
    let r = port.call(id, params)?;
    let mut entry = json!({"command": id, "ok": true});
    if let Some(job) = r.get("job").and_then(Value::as_u64) {
        entry["job"] = json!(job);
        let result = wait_job(job, port, prog, base)?;
        if let Some(p) = result.get("path").and_then(Value::as_str).filter(|p| !p.is_empty()) {
            *last_path = p.to_string();
        }
        entry["result"] = result;
    } else if let Some(p) = r.get("path").and_then(Value::as_str).filter(|_| id == "file.exportMedia") {
        *last_path = p.to_string();
    }
    if let Some(seq) = r.get("sequence").and_then(Value::as_u64) {
        entry["sequence"] = json!(seq);
    }
    ran.push(entry);
    Ok(())
}

/// Wait for job `job` (started by a step), polling `jobs.list` through the port.
fn wait_job(job: u64, port: &CommandPort, prog: &Arc<filmcraft_export::Progress>, base: u64) -> std::result::Result<Value, String> {
    let t0 = web_time::Instant::now();
    loop {
        if port.cancelled() {
            port.cancel_job(job);
            return Err("cancelled".into());
        }
        let jobs = port.call("jobs.list", json!({}))?;
        let Some(j) = jobs.as_array().and_then(|a| a.iter().find(|x| x["id"].as_u64() == Some(job))).cloned() else {
            return Err(format!("job {job} is gone"));
        };
        let frac = j["progress"].as_f64().filter(|x| x.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0);
        prog.done.store(base.saturating_add((frac * 99.0) as u64), Ordering::Relaxed);
        if j["finished"].as_bool() == Some(true) {
            if let Some(e) = j["result"]["error"].as_str() {
                return Err(e.to_string());
            }
            return Ok(j["result"].clone());
        }
        if t0.elapsed() > MAX_JOB_WAIT {
            port.cancel_job(job);
            return Err(format!("job {job} ran longer than {} hours", MAX_JOB_WAIT.as_secs() / 3600));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

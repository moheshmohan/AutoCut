//! Showcase plan patches: a reviewed plan ([`crate::showcase`]'s `--script-in` JSON) changed by
//! review notes, so a new round of a showcase is one file of notes instead of a throwaway script
//! (`filmcraft-cli showcase-plan`).
//!
//! A patch is JSON keyed by entry **label** (labels are stable; numbers change as entries come and
//! go):
//!
//! ```json
//! {
//!   "timezone": "+04:00",
//!   "entries": {
//!     "Pepper": {"line": "…", "chips": ["…"], "delivery": "calm", "place": "Abu Dhabi"},
//!     "Micropolis M02": {
//!       "shots": [{"item": 2407}],
//!       "split": [{"label": "Micropolis M01", "unix": "2022-01-23 13:40:35",
//!                  "shots": [{"item": 2408}, {"item": 2409, "from": 19, "to": 23.5}], "line": "…"}]
//!     },
//!     "Disinfection robot": {"shotEdits": {"14": {"from": 0, "fit": "full"}}},
//!     "Old robot": {"remove": true}
//!   },
//!   "add": [{"label": "Unitree Go2", "kind": "quadruped", "unix": 1731674836,
//!            "shots": [{"item": 4841, "plain": [4, 40], "mute": true}], "line": "…"}],
//!   "script": {"intro": "…", "outro": "{Count} robots in twelve years…", "pronounce": {"H1": "H one"}},
//!   "outro": {"channel": "…"}
//! }
//! ```
//!
//! - **`entries`**: each named entry changed in place. `line`, `chips`, `delivery`, `say` and
//!   `voice` (`{"temperature", "emphasis", "speed"}`, how a Malayalam voice reads it) are its
//!   script parts; `shots` replaces its shots; `shotEdits` changes fields of its shots of an item
//!   (`null` removes one); `unix` is seconds or a date (`"2022-01-23 13:40:35"`, in the patch's
//!   `timezone`, else UTC, unless it carries its own `Z`/`+hh:mm`); `split` makes more entries
//!   from it (a copy of it as patched, with the split's fields over it; its own script parts,
//!   none unless given); `remove: true` drops it; any other field (`label`, `place`, `event`,
//!   `kind`, `description`, `caption`, `facts`, `redact`) is set on the entry, `null` removes it.
//!   An ambiguous label is picked as `"<label> #2"`.
//! - **`add`**: new entries (a `label` and at least one shot).
//! - **Shots** name an `item`. Without `plain`, a shot is the plan's analysed shot of that item
//!   (the same entry's first, else any entry's) with the given fields over it. With `plain: true`,
//!   or when the plan has no shot of the item, it is a plain shot: the clip's name and length from
//!   the project ([`clips_from_inspect`]; or the shot's own `name` / `clip_seconds`), the robot
//!   counted in view across `from`..`to` (`plain: [a, b]`: across that stretch, the window left
//!   automatic). A **photo** (`"photo": true`, or an item the project holds as a still image) is
//!   a shot held `hold` seconds with its slow pan and zoom (`move`: `in`, `out`, `pan`, `none`;
//!   `fit`, `focus`): `{"item": 4977, "photo": true, "hold": 4.0, "move": "in"}`; without an
//!   analysed shot of it, it aims at the picture's middle (or its `focus`). An entry may mix
//!   photos and videos.
//! - **`script`**: `intro`, `outro`, `pronounce` (merged, `null` removes a respelling) and
//!   `delivery` (`intro`/`outro`). `{count}` / `{Count}` in the intro or outro become the robot
//!   count in words, and the old count in words or digits ("Forty-three") is updated.
//! - **Anything else** at the top (`outro`, `style`, `timeline`, `events`, `transitions`, `redact`,
//!   `language`, …) is merged into the plan (JSON merge patch: objects merge, `null` removes).
//!   A leading `~/` of `outro.logo` is expanded.
//!
//! Then the entries are sorted by `unix`, numbered from 1, and `script.lines`, `chips`,
//! `delivery`, `say` and `voice` are rebuilt aligned with them (`voiceDefault` stays) (a scene's chips are dropped, chips are cut
//! to three, `delivery` defaults to warm). [`apply`] reports the entries left without a line.
//!
//! [`fill_places`] gives entries without a `place` the one Google Photos shows for their first
//! shot (a `photos locations` file, [`crate::places`]; `showcase-plan --places-from-locations`).
//!
//! [`with_lines`] writes the plan for another language: `{"intro", "outro", "lines": {"<label>":
//! "…"}}` read over it — every spoken entry needs a line — with `language` set and the English
//! respellings and `say` cleared (chips and on-screen text stay as written).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::AutoCutError;
use crate::showcase::{Entry, Script};

/// Most entries a plan may have, shots an entry, and splits of one entry.
const MAX_ENTRIES: usize = 10_000;
const MAX_SHOTS: usize = 200;
const MAX_SPLITS: usize = 64;
/// Chips an entry shows at most (as [`crate::look::MAX_CHIPS`]).
const CHIPS: usize = crate::look::MAX_CHIPS;
/// The delivery an entry without one gets.
const DEFAULT_DELIVERY: &str = "warm";
/// Fields of an entry patch that are not entry fields.
const RECORD_KEYS: [&str; 9] = ["line", "chips", "delivery", "say", "voice", "shotEdits", "split", "remove", "shots"];
/// Top-level patch keys that are not plan fields.
const PATCH_KEYS: [&str; 4] = ["entries", "add", "script", "timezone"];

const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;

fn bad(m: impl std::fmt::Display) -> AutoCutError {
    AutoCutError::Other(format!("plan patch: {m}"))
}

/// A project clip's name and length (seconds), by item id, and whether it is a photo (a still).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipInfo {
    pub name: String,
    pub seconds: f64,
    pub still: bool,
}

/// The clips of a project, by item id.
pub type Clips = BTreeMap<u64, ClipInfo>;

/// Most project items read.
const MAX_ITEMS: usize = 1_000_000;

/// The media items of a `project.inspect` result (every object with an `item` id and a
/// `duration` in ticks): their names and lengths.
pub fn clips_from_inspect(v: &Value) -> Clips {
    let mut out = Clips::new();
    // an explicit stack: the tree is the project's, nested bins included
    let mut stack: Vec<(&Value, usize)> = vec![(v, 0)];
    while let Some((v, depth)) = stack.pop() {
        if depth > 256 || out.len() >= MAX_ITEMS {
            continue;
        }
        match v {
            Value::Object(o) => {
                if let (Some(item), Some(d)) = (o.get("item").and_then(Value::as_u64), o.get("duration").and_then(Value::as_f64))
                    && d.is_finite()
                    && d >= 0.0
                {
                    let name = o.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                    let still = o.get("type").and_then(Value::as_str) == Some(crate::analyze::STILL_TYPE);
                    out.insert(item, ClipInfo { name, seconds: d / TPS, still });
                }
                stack.extend(o.values().map(|c| (c, depth + 1)));
            }
            Value::Array(a) => stack.extend(a.iter().map(|c| (c, depth + 1))),
            _ => {}
        }
    }
    out
}

/// What [`apply`] made.
#[derive(Clone, Debug, PartialEq)]
pub struct Patched {
    pub plan: Value,
    pub entries: usize,
    /// Entries that are not scenes.
    pub robots: usize,
    /// Entries with a line.
    pub lines: usize,
    /// The labels of robot entries without a line (scenes may be silent).
    pub unlined: Vec<String>,
    /// Scenes without a line.
    pub silent_scenes: usize,
    /// What else happened ("outro: Forty-three → Forty-eight").
    pub notes: Vec<String>,
}

impl Patched {
    /// A short JSON summary.
    pub fn summary(&self) -> Value {
        json!({
            "entries": self.entries,
            "robots": self.robots,
            "lines": self.lines,
            "unlined": self.unlined,
            "silentScenes": self.silent_scenes,
            "outro": self.plan["script"]["outro"],
            "notes": self.notes,
        })
    }
}

/// An entry with its script parts.
#[derive(Clone, Debug)]
struct Rec {
    entry: Map<String, Value>,
    line: String,
    chips: Option<Value>,
    delivery: Option<String>,
    say: Option<String>,
    /// How the voice reads its line (`script.voice`: temperature, emphasis, speed).
    voice: Option<Value>,
}

impl Rec {
    fn label(&self) -> &str {
        self.entry.get("label").and_then(Value::as_str).unwrap_or_default()
    }
    fn unix(&self) -> Option<i64> {
        self.entry.get("unix").and_then(Value::as_i64)
    }
    fn scene(&self) -> bool {
        self.entry.get("kind").and_then(Value::as_str).is_some_and(|k| k.trim().eq_ignore_ascii_case(crate::showcase::SCENE))
    }
    fn shots(&self) -> &[Value] {
        self.entry.get("shots").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default()
    }
}

/// The robots (entries that are not scenes) among `recs`.
fn robots(recs: &[Rec]) -> usize {
    recs.iter().filter(|r| !r.scene()).count()
}

/// The plan's entries with their script parts.
fn records(plan: &Value) -> Result<Vec<Rec>, AutoCutError> {
    let entries = plan.get("entries").and_then(Value::as_array).ok_or_else(|| bad("the base plan has no entries"))?;
    if entries.len() > MAX_ENTRIES {
        return Err(bad(format!("more than {MAX_ENTRIES} entries")));
    }
    let script = plan.get("script").cloned().unwrap_or(Value::Null);
    let lines = script.get("lines").and_then(Value::as_array).cloned().unwrap_or_default();
    if !lines.is_empty() && lines.len() != entries.len() {
        return Err(bad(format!("the base plan has {} script lines for {} entries", lines.len(), entries.len())));
    }
    let by = |k: &str, n: usize| script.get(k).and_then(|m| m.get(n.to_string())).cloned();
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let entry = e.as_object().cloned().ok_or_else(|| bad(format!("entry {} is not an object", i + 1)))?;
            if entry.get("label").and_then(Value::as_str).is_none() {
                return Err(bad(format!("entry {} has no label", i + 1)));
            }
            // (the builder numbers entries by their place, as do the script's maps)
            let n = i + 1;
            Ok(Rec {
                entry,
                line: lines.get(i).and_then(Value::as_str).unwrap_or_default().to_string(),
                chips: by("chips", n).filter(Value::is_array),
                delivery: by("delivery", n).and_then(|d| d.as_str().map(str::to_string)),
                say: by("say", n).and_then(|d| d.as_str().map(str::to_string)),
                voice: by("voice", n).filter(Value::is_object),
            })
        })
        .collect()
}

/// Apply `patches` (in order) to the plan `base`; `clips` gives plain shots their clip's name and
/// length (see the module docs).
pub fn apply(base: &Value, patches: &[Value], clips: &Clips) -> Result<Patched, AutoCutError> {
    let mut recs = records(base)?;
    let old_robots = robots(&recs);
    let base_script = base.get("script").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut intro = base_script.get("intro").and_then(Value::as_str).unwrap_or_default().to_string();
    let mut outro = base_script.get("outro").and_then(Value::as_str).unwrap_or_default().to_string();
    let mut pronounce = base_script.get("pronounce").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut part_delivery: Map<String, Value> = Map::new();
    for k in ["intro", "outro"] {
        if let Some(d) = base_script.get("delivery").and_then(|d| d.get(k)).filter(|d| d.is_string()) {
            part_delivery.insert(k.into(), d.clone());
        }
    }
    let mut part_say: Map<String, Value> = Map::new();
    for k in ["intro", "outro"] {
        if let Some(d) = base_script.get("say").and_then(|d| d.get(k)).filter(|d| d.is_string()) {
            part_say.insert(k.into(), d.clone());
        }
    }
    let mut plan: Map<String, Value> = base.as_object().cloned().unwrap_or_default();
    plan.remove("entries");
    plan.remove("script");
    for (pi, patch) in patches.iter().enumerate() {
        let at = |m: String| bad(if patches.len() > 1 { format!("patch {}: {m}", pi + 1) } else { m });
        let p = patch.as_object().ok_or_else(|| at("a patch is a JSON object".into()))?;
        let tz = match p.get("timezone") {
            None | Some(Value::Null) => 0,
            Some(Value::String(s)) => offset(s).ok_or_else(|| at(format!("timezone: `{}` is not like +04:00", clip(s))))?,
            Some(_) => return Err(at("timezone: a string like +04:00".into())),
        };
        // every shot of the plan before this patch: what a shot by item is copied from
        let pool: Vec<Value> = recs.iter().flat_map(|r| r.shots().iter().cloned()).collect();
        if let Some(entries) = p.get("entries") {
            let entries = entries.as_object().ok_or_else(|| at("entries: an object keyed by entry label".into()))?;
            for (label, spec) in entries {
                let i = find(&recs, label).map_err(|m| at(format!("entries: {m}")))?;
                let spec = spec.as_object().ok_or_else(|| at(format!("entries.{}: an object", clip(label))))?;
                if spec.get("remove").and_then(Value::as_bool) == Some(true) {
                    recs.remove(i);
                    continue;
                }
                let own: Vec<Value> = recs.get(i).map(|r| r.shots().to_vec()).unwrap_or_default();
                let Some(rec) = recs.get_mut(i) else { continue };
                change(rec, spec, &own, &pool, clips, tz).map_err(|m| at(format!("entries.{}: {m}", clip(label))))?;
                let mut more = Vec::new();
                match spec.get("split") {
                    None | Some(Value::Null) => {}
                    Some(Value::Array(splits)) if splits.len() <= MAX_SPLITS => {
                        for (k, s) in splits.iter().enumerate() {
                            let s = s.as_object().ok_or_else(|| at(format!("entries.{}.split[{k}]: an object", clip(label))))?;
                            let mut new = Rec { entry: rec.entry.clone(), line: String::new(), chips: None, delivery: None, say: None, voice: None };
                            change(&mut new, s, &own, &pool, clips, tz).map_err(|m| at(format!("entries.{}.split[{k}]: {m}", clip(label))))?;
                            more.push(new);
                        }
                    }
                    Some(_) => return Err(at(format!("entries.{}.split: a list of up to {MAX_SPLITS} entries", clip(label)))),
                }
                recs.extend(more);
            }
        }
        match p.get("add") {
            None | Some(Value::Null) => {}
            Some(Value::Array(adds)) => {
                for (k, s) in adds.iter().enumerate() {
                    let s = s.as_object().ok_or_else(|| at(format!("add[{k}]: an object")))?;
                    let label = s.get("label").and_then(Value::as_str).map(str::trim).filter(|l| !l.is_empty());
                    let Some(label) = label else { return Err(at(format!("add[{k}]: a new entry needs a label"))) };
                    let mut entry = Map::new();
                    for (key, d) in [("label", json!(label)), ("kind", json!("")), ("description", json!("")), ("unix", Value::Null)] {
                        entry.insert(key.into(), d);
                    }
                    let mut new = Rec { entry, line: String::new(), chips: None, delivery: None, say: None, voice: None };
                    change(&mut new, s, &[], &pool, clips, tz).map_err(|m| at(format!("add.{}: {m}", clip(label))))?;
                    if new.shots().is_empty() {
                        return Err(at(format!("add.{}: a new entry needs shots", clip(label))));
                    }
                    recs.push(new);
                }
            }
            Some(_) => return Err(at("add: a list of entries".into())),
        }
        if recs.len() > MAX_ENTRIES {
            return Err(at(format!("more than {MAX_ENTRIES} entries")));
        }
        if let Some(s) = p.get("script") {
            let s = s.as_object().ok_or_else(|| at("script: an object".into()))?;
            for (k, target) in [("intro", &mut intro), ("outro", &mut outro)] {
                match s.get(k) {
                    None | Some(Value::Null) => {}
                    Some(Value::String(t)) => *target = t.clone(),
                    Some(_) => return Err(at(format!("script.{k}: text"))),
                }
            }
            match s.get("pronounce") {
                None | Some(Value::Null) => {}
                Some(Value::Object(m)) => {
                    for (k, v) in m {
                        match v {
                            Value::Null => {
                                pronounce.remove(k);
                            }
                            Value::String(_) => {
                                pronounce.insert(k.clone(), v.clone());
                            }
                            _ => return Err(at(format!("script.pronounce.{}: text or null", clip(k)))),
                        }
                    }
                }
                Some(_) => return Err(at("script.pronounce: an object of respellings".into())),
            }
            for (field, store) in [("delivery", &mut part_delivery), ("say", &mut part_say)] {
                match s.get(field) {
                    None | Some(Value::Null) => {}
                    Some(Value::Object(m)) => {
                        for (k, v) in m {
                            if k != "intro" && k != "outro" {
                                return Err(at(format!("script.{field}: only intro and outro here (an entry's goes on the entry)")));
                            }
                            match v {
                                Value::Null => {
                                    store.remove(k);
                                }
                                Value::String(_) => {
                                    store.insert(k.clone(), v.clone());
                                }
                                _ => return Err(at(format!("script.{field}.{k}: text or null"))),
                            }
                        }
                    }
                    Some(_) => return Err(at(format!("script.{field}: an object"))),
                }
            }
        }
        for (k, v) in p {
            if PATCH_KEYS.contains(&k.as_str()) {
                continue;
            }
            if k == "number" {
                return Err(at("numbers are given by the order of the entries".into()));
            }
            if v.is_null() {
                plan.remove(k);
            } else {
                merge(plan.entry(k.clone()).or_insert(Value::Null), v, 0);
            }
        }
    }
    finish(plan, recs, old_robots, Parts { intro, outro, pronounce, delivery: part_delivery, say: part_say }, base_script)
}

/// The script's own parts, as patched.
struct Parts {
    intro: String,
    outro: String,
    pronounce: Map<String, Value>,
    delivery: Map<String, Value>,
    say: Map<String, Value>,
}

/// Sort, number and put the plan back together.
fn finish(mut plan: Map<String, Value>, mut recs: Vec<Rec>, old_robots: usize, parts: Parts, base_script: Map<String, Value>) -> Result<Patched, AutoCutError> {
    // undated entries last, the rest in time order (stable: same-time entries keep their order)
    recs.sort_by_key(|r| (r.unix().is_none(), r.unix().unwrap_or(0)));
    let mut entries = Vec::with_capacity(recs.len());
    let (mut lines, mut chips, mut delivery, mut say) = (Vec::with_capacity(recs.len()), Map::new(), parts.delivery, parts.say);
    // the voice settings: the intro's and outro's as they were, the entries' by their new numbers
    let mut voice: Map<String, Value> = base_script
        .get("voice")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter(|(k, v)| (k.as_str() == "intro" || k.as_str() == "outro") && v.is_object()).map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    let (mut unlined, mut silent_scenes) = (Vec::new(), 0usize);
    for (i, r) in recs.iter_mut().enumerate() {
        let n = i + 1;
        r.entry.insert("number".into(), json!(n));
        if let Some(c) = r.chips.as_ref().and_then(Value::as_array).filter(|c| !c.is_empty())
            && !r.scene()
        {
            chips.insert(n.to_string(), Value::Array(c.iter().take(CHIPS).cloned().collect()));
        }
        delivery.insert(n.to_string(), json!(r.delivery.clone().unwrap_or_else(|| DEFAULT_DELIVERY.into())));
        if let Some(s) = &r.say {
            say.insert(n.to_string(), json!(s));
        }
        if let Some(v) = &r.voice {
            voice.insert(n.to_string(), v.clone());
        }
        if r.line.trim().is_empty() {
            if r.scene() {
                silent_scenes += 1;
            } else {
                unlined.push(r.label().to_string());
            }
        }
        lines.push(json!(r.line));
        entries.push(Value::Object(r.entry.clone()));
    }
    let new_robots = robots(&recs);
    let mut notes = Vec::new();
    let intro = count_words(&parts.intro, old_robots, new_robots);
    let outro = count_words(&parts.outro, old_robots, new_robots);
    if outro != parts.outro {
        notes.push(format!("outro: the count is now {} ({new_robots} robots)", words(new_robots, true)));
    }
    // the script: the base's other fields kept, its parts rebuilt
    let mut script = base_script;
    for k in ["intro", "outro", "lines", "templated", "chips", "pronounce", "delivery", "say", "voice"] {
        script.remove(k);
    }
    if !voice.is_empty() {
        script.insert("voice".into(), Value::Object(voice));
    }
    script.insert("intro".into(), json!(intro));
    script.insert("outro".into(), json!(outro));
    script.insert("lines".into(), json!(lines));
    script.insert("templated".into(), json!(0));
    script.insert("chips".into(), Value::Object(chips));
    script.insert("pronounce".into(), Value::Object(parts.pronounce));
    script.insert("delivery".into(), Value::Object(delivery));
    if !say.is_empty() {
        script.insert("say".into(), Value::Object(say));
    }
    expand_logo(&mut plan);
    plan.insert("entries".into(), Value::Array(entries));
    plan.insert("script".into(), Value::Object(script));
    let plan = Value::Object(plan);
    check(&plan)?;
    let lines = recs.iter().filter(|r| !r.line.trim().is_empty()).count();
    Ok(Patched { entries: recs.len(), robots: new_robots, lines, unlined, silent_scenes, notes, plan })
}

/// The plan reads as a showcase does: entries, script, chips.
fn check(plan: &Value) -> Result<(), AutoCutError> {
    let entries: Vec<Entry> = serde_json::from_value(plan["entries"].clone()).map_err(|e| bad(format!("the patched entries do not read as entries: {e}")))?;
    let script: Script = serde_json::from_value(plan["script"].clone()).map_err(|e| bad(format!("the patched script does not read as a script: {e}")))?;
    let numbers: Vec<usize> = entries.iter().map(|e| e.number).collect();
    crate::look::chips_by_entry(&script.chips, &numbers).map_err(|e| bad(e.to_string()))?;
    crate::shot_edit::check(&entries).map_err(|e| bad(e.to_string()))?;
    Ok(())
}

/// The record labelled `label` (exactly, else ignoring case; `"<label> #2"` for the second of
/// several).
fn find(recs: &[Rec], label: &str) -> Result<usize, String> {
    let exact: Vec<usize> = recs.iter().enumerate().filter(|(_, r)| r.label() == label).map(|(i, _)| i).collect();
    let any_case = || -> Vec<usize> { recs.iter().enumerate().filter(|(_, r)| r.label().eq_ignore_ascii_case(label.trim())).map(|(i, _)| i).collect() };
    let found = if exact.is_empty() { any_case() } else { exact };
    match found.as_slice() {
        [i] => return Ok(*i),
        [] => {}
        many => return Err(format!("{} entries are labelled \"{}\": name one as \"{} #2\"", many.len(), clip(label), clip(label))),
    }
    if let Some((name, k)) = label.rsplit_once(" #")
        && let Ok(k) = k.trim().parse::<usize>()
    {
        let same: Vec<usize> = recs.iter().enumerate().filter(|(_, r)| r.label() == name).map(|(i, _)| i).collect();
        if let Some(i) = k.checked_sub(1).and_then(|k| same.get(k)) {
            return Ok(*i);
        }
    }
    Err(format!("no entry is labelled \"{}\"", clip(label)))
}

/// Change `rec` by an entry patch `spec`; `own` are its shots before the patch, `pool` every shot
/// of the plan before it.
fn change(rec: &mut Rec, spec: &Map<String, Value>, own: &[Value], pool: &[Value], clips: &Clips, tz: i64) -> Result<(), String> {
    for (k, v) in spec {
        if RECORD_KEYS.contains(&k.as_str()) {
            continue;
        }
        match k.as_str() {
            "number" => return Err("numbers are given by the order of the entries".into()),
            "label" => {
                let l = v.as_str().map(str::trim).filter(|l| !l.is_empty()).ok_or("label: text")?;
                rec.entry.insert("label".into(), json!(l));
            }
            "unix" => {
                let u = match v {
                    Value::Null => Value::Null,
                    Value::Number(_) => json!(v.as_i64().ok_or("unix: whole seconds")?),
                    Value::String(s) => json!(date(s, tz).ok_or_else(|| format!("unix: `{}` is not a date like 2022-01-23 13:40:35", clip(s)))?),
                    _ => return Err("unix: seconds or a date".into()),
                };
                rec.entry.insert("unix".into(), u);
            }
            _ if v.is_null() => {
                rec.entry.remove(k);
            }
            _ => {
                rec.entry.insert(k.clone(), v.clone());
            }
        }
    }
    let text = |k: &str| -> Result<Option<Option<String>>, String> {
        match spec.get(k) {
            None => Ok(None),
            Some(Value::Null) => Ok(Some(None)),
            Some(Value::String(s)) => Ok(Some(Some(s.clone()))),
            Some(_) => Err(format!("{k}: text")),
        }
    };
    if let Some(l) = text("line")? {
        rec.line = l.unwrap_or_default();
    }
    if let Some(d) = text("delivery")? {
        rec.delivery = d;
    }
    if let Some(s) = text("say")? {
        rec.say = s;
    }
    match spec.get("voice") {
        None => {}
        Some(Value::Null) => rec.voice = None,
        Some(v @ Value::Object(_)) => {
            // read as the builder reads it: a misspelt or out-of-range setting is refused here
            let s: crate::showcase::VoiceSettings = serde_json::from_value(v.clone()).map_err(|e| format!("voice: {e}"))?;
            s.check("of the entry")?;
            rec.voice = Some(v.clone());
        }
        Some(_) => return Err("voice: an object of voice settings (temperature, emphasis, speed)".into()),
    }
    match spec.get("chips") {
        None => {}
        Some(Value::Null) => rec.chips = None,
        Some(Value::Array(c)) if c.iter().all(Value::is_string) => rec.chips = Some(Value::Array(c.clone())),
        Some(_) => return Err("chips: a list of texts".into()),
    }
    match spec.get("shots") {
        None => {}
        Some(Value::Array(list)) if !list.is_empty() && list.len() <= MAX_SHOTS => {
            let shots =
                list.iter().enumerate().map(|(k, s)| shot(s, own, pool, clips).map_err(|m| format!("shots[{k}]: {m}"))).collect::<Result<Vec<_>, _>>()?;
            rec.entry.insert("shots".into(), Value::Array(shots));
        }
        Some(_) => return Err(format!("shots: a list of 1 to {MAX_SHOTS} shots")),
    }
    match spec.get("shotEdits") {
        None | Some(Value::Null) => {}
        Some(Value::Object(edits)) => {
            for (item, fields) in edits {
                let item: u64 = item.trim().parse().map_err(|_| format!("shotEdits: `{}` is not an item id", clip(item)))?;
                let fields = fields.as_object().ok_or_else(|| format!("shotEdits.{item}: an object of shot fields"))?;
                let shots = rec.entry.get_mut("shots").and_then(Value::as_array_mut).ok_or("shotEdits: the entry has no shots")?;
                let mut hit = false;
                for s in shots.iter_mut().filter(|s| s.get("item").and_then(Value::as_u64) == Some(item)) {
                    hit = true;
                    if let Some(o) = s.as_object_mut() {
                        set_fields(o, fields)?;
                    }
                }
                if !hit {
                    return Err(format!("shotEdits: the entry has no shot of item {item}"));
                }
            }
        }
        Some(_) => return Err("shotEdits: an object keyed by item id".into()),
    }
    Ok(())
}

/// Shot fields set over a shot (`null` removes one); `item` and `plain` are not fields.
fn set_fields(o: &mut Map<String, Value>, fields: &Map<String, Value>) -> Result<(), String> {
    for (k, v) in fields {
        match k.as_str() {
            "item" | "plain" => continue,
            _ if v.is_null() => {
                o.remove(k);
            }
            _ => {
                o.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(())
}

/// Two decimals, as the plan writes moments (the nearest double to the rounded decimal).
fn round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap_or(x)
}

/// A shot from a patch's shot spec (see the module docs).
fn shot(spec: &Value, own: &[Value], pool: &[Value], clips: &Clips) -> Result<Value, String> {
    let spec = match spec {
        Value::Number(n) => json!({"item": n}),
        Value::Object(_) => spec.clone(),
        _ => return Err("a shot is {\"item\": id, …}".into()),
    };
    let fields = spec.as_object().ok_or("a shot is an object")?;
    let item = fields.get("item").and_then(Value::as_u64).ok_or("a shot names its `item`")?;
    let of = |list: &[Value]| list.iter().find(|s| s.get("item").and_then(Value::as_u64) == Some(item)).cloned();
    let analysed = of(own).or_else(|| of(pool));
    let num = |k: &str| fields.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
    let known = clips.get(&item);
    // a photo: said so, held as a still by the project, or the plan's shot of it is one
    let said = fields.get("photo").and_then(Value::as_bool);
    let photo = said.unwrap_or(false) || known.is_some_and(|c| c.still) || analysed.as_ref().is_some_and(|a| a.get("photo") == Some(&Value::Bool(true)));
    if said == Some(true) && known.is_some_and(|c| !c.still) {
        return Err(format!("item {item} is a video in the project, not a photo"));
    }
    if photo && said != Some(false) {
        return photo_shot(item, fields, analysed, known);
    }
    let plain = match fields.get("plain") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => analysed.is_none().then_some(None),
        Some(Value::Bool(true)) => Some(None),
        Some(Value::Array(a)) => match a.as_slice() {
            [x, y] => Some(Some((x.as_f64().ok_or("plain: [from, to] in seconds")?, y.as_f64().ok_or("plain: [from, to] in seconds")?))),
            _ => return Err("plain: true, or [from, to] in seconds".into()),
        },
        Some(_) => return Err("plain: true, or [from, to] in seconds".into()),
    };
    let mut out = match (plain, analysed) {
        (None, Some(Value::Object(o))) => o,
        (None, _) => return Err(format!("item {item}: the plan has no shot of it")),
        (Some(window), analysed) => {
            let seconds = num("clip_seconds")
                .or(known.map(|c| c.seconds))
                .or_else(|| analysed.as_ref().and_then(|a| a.get("clip_seconds")).and_then(Value::as_f64))
                .filter(|d| d.is_finite() && *d > 0.0)
                .ok_or_else(|| format!("item {item}: its length is unknown (give --project, or the shot's clip_seconds)"))?;
            let name = fields
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or(known.map(|c| c.name.clone()))
                .or_else(|| analysed.as_ref().and_then(|a| a.get("name")).and_then(Value::as_str).map(str::to_string))
                .unwrap_or_default();
            let (a, b) = window.unwrap_or((num("from").unwrap_or(0.0), num("to").unwrap_or(seconds)));
            if !(a.is_finite() && b.is_finite() && a >= 0.0 && b > a) {
                return Err(format!("item {item}: the stretch in view runs from {a} to {b}"));
            }
            let seen: Vec<f64> = (0..5).map(|k| round2(a + (b - a) * f64::from(k) / 4.0)).collect();
            let best = seen.get(2).copied().unwrap_or(a);
            let mut o = Map::new();
            o.insert("item".into(), json!(item));
            o.insert("name".into(), json!(name));
            o.insert("clip_seconds".into(), json!(seconds));
            o.insert("best".into(), json!(best));
            o.insert("seen".into(), json!(seen));
            o.insert("unseen".into(), json!([]));
            o.insert("largest".into(), Value::Null);
            o
        }
    };
    set_fields(&mut out, fields)?;
    Ok(Value::Object(out))
}

/// A photo shot from a patch's spec: the plan's analysed shot of it (its subject's box) with the
/// spec's fields over it, else a plain one — its name from the project, held up to
/// [`crate::showcase::PHOTO_REACH`], aimed at its middle.
fn photo_shot(item: u64, fields: &Map<String, Value>, analysed: Option<Value>, known: Option<&ClipInfo>) -> Result<Value, String> {
    for k in ["from", "to", "speed", "plain"] {
        if fields.get(k).is_some_and(|v| !v.is_null() && *v != Value::Bool(false)) {
            return Err(format!("item {item} is a photo: it has no `{k}`; give it a `hold` in seconds"));
        }
    }
    let mut out = match analysed {
        Some(Value::Object(o)) if o.get("photo") == Some(&Value::Bool(true)) => o,
        other => {
            let name = fields
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or(known.map(|c| c.name.clone()))
                .or_else(|| other.as_ref().and_then(|a| a.get("name")).and_then(Value::as_str).map(str::to_string))
                .unwrap_or_default();
            let mut o = Map::new();
            o.insert("item".into(), json!(item));
            o.insert("name".into(), json!(name));
            o.insert("clip_seconds".into(), json!(crate::showcase::PHOTO_REACH));
            o.insert("best".into(), json!(0.0));
            o.insert("seen".into(), json!([]));
            o.insert("unseen".into(), json!([]));
            o.insert("largest".into(), Value::Null);
            o
        }
    };
    set_fields(&mut out, fields)?;
    out.insert("photo".into(), Value::Bool(true));
    Ok(Value::Object(out))
}

/// JSON merge patch (RFC 7386) of `patch` into `target`, at most 64 levels deep.
fn merge(target: &mut Value, patch: &Value, depth: usize) {
    let Value::Object(p) = patch else {
        *target = patch.clone();
        return;
    };
    if depth > 64 {
        return;
    }
    if !target.is_object() {
        *target = Value::Object(Map::new());
    }
    if let Value::Object(t) = target {
        for (k, v) in p {
            if v.is_null() {
                t.remove(k);
            } else {
                merge(t.entry(k.clone()).or_insert(Value::Null), v, depth + 1);
            }
        }
    }
}

/// `~/…` in the outro's logo, made absolute (the plan is read where the app runs).
fn expand_logo(plan: &mut Map<String, Value>) {
    let Some(logo) = plan.get_mut("outro").and_then(|o| o.get_mut("logo")) else { return };
    if let Some(rest) = logo.as_str().and_then(|l| l.strip_prefix("~/"))
        && let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty())
    {
        *logo = json!(std::path::Path::new(&home).join(rest).to_string_lossy());
    }
}

/// At most 60 characters of `s`, for messages.
fn clip(s: &str) -> String {
    s.chars().take(60).collect()
}

// ---------------------------------------------------------------------------------------------
// dates

/// Days from 1970-01-01 to the civil date `y-m-d` (proleptic Gregorian, by chrono). A day past
/// the month's last runs on into the next month (`02-31` is March 3rd or 2nd), as it always has.
fn days_from_civil(y: i64, m: u32, d: u32) -> Option<i64> {
    let first = chrono::NaiveDate::from_ymd_opt(i32::try_from(y).ok()?, m, 1)?;
    let days = first.signed_duration_since(chrono::DateTime::UNIX_EPOCH.date_naive()).num_days();
    Some(days + i64::from(d) - 1)
}

/// A UTC offset in seconds: `Z`, `UTC`, `+04:00`, `+0400`, `-5`.
fn offset(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("z") || s.eq_ignore_ascii_case("utc") || s.eq_ignore_ascii_case("gmt") {
        return Some(0);
    }
    let (sign, rest) = match s.as_bytes().first()? {
        b'+' => (1, s.get(1..)?),
        b'-' => (-1, s.get(1..)?),
        _ => return None,
    };
    let (h, m) = match rest.split_once(':') {
        Some((h, m)) => (h, m),
        None if rest.len() == 4 => (rest.get(..2)?, rest.get(2..)?),
        None => (rest, "0"),
    };
    let (h, m): (i64, i64) = (h.parse().ok()?, m.parse().ok()?);
    ((0..=14).contains(&h) && (0..60).contains(&m)).then_some(sign * (h * 3600 + m * 60))
}

/// Seconds since 1970 of `"YYYY-MM-DD[ T]HH:MM[:SS][Z|±HH:MM]"` (or a date alone), in the
/// timezone `tz` (seconds east of UTC) unless it names its own.
fn date(s: &str, tz: i64) -> Option<i64> {
    let s = s.trim();
    let (day, rest) = match s.find([' ', 'T']) {
        Some(i) => (s.get(..i)?, s.get(i + 1..)?.trim()),
        None => (s, ""),
    };
    let mut d = day.split('-');
    let (y, m, dd): (i64, u32, u32) = (d.next()?.parse().ok()?, d.next()?.parse().ok()?, d.next()?.parse().ok()?);
    if d.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&dd) || !(1..=9999).contains(&y) {
        return None;
    }
    // the time, and the zone after it
    let zone_at = rest.find(['Z', 'z', '+', '-']);
    let (time, zone) = match zone_at {
        Some(i) => (rest.get(..i)?, Some(rest.get(i..)?)),
        None => (rest, None),
    };
    let mut t = time.trim().split(':').filter(|p| !p.is_empty());
    let mut part = |max: i64| -> Option<i64> {
        match t.next() {
            None => Some(0),
            Some(p) => p.parse::<i64>().ok().filter(|v| (0..max).contains(v)),
        }
    };
    let (hh, mm, ss) = (part(24)?, part(60)?, part(61)?);
    if t.next().is_some() {
        return None;
    }
    let tz = match zone {
        Some(z) => offset(z)?,
        None => tz,
    };
    Some(days_from_civil(y, m, dd)? * 86_400 + hh * 3600 + mm * 60 + ss - tz)
}

// ---------------------------------------------------------------------------------------------
// the count in words

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/// `n` in English words ("forty-eight", "one hundred and two"), capitalised with `cap`; digits
/// from a thousand on.
pub fn words(n: usize, cap: bool) -> String {
    fn below_100(n: usize) -> String {
        if let Some(w) = ONES.get(n) {
            return (*w).to_string();
        }
        let t = TENS.get(n / 10).copied().unwrap_or_default();
        match n % 10 {
            0 => t.to_string(),
            o => format!("{t}-{}", ONES.get(o).copied().unwrap_or_default()),
        }
    }
    let w = match n {
        0..100 => below_100(n),
        100..1000 => {
            let h = ONES.get(n / 100).copied().unwrap_or_default();
            match n % 100 {
                0 => format!("{h} hundred"),
                r => format!("{h} hundred and {}", below_100(r)),
            }
        }
        _ => n.to_string(),
    };
    if cap {
        let mut c = w.chars();
        c.next().map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
    } else {
        w
    }
}

/// `text` with `{Count}` / `{count}` as the count `new` in words, and the old count `old` in
/// words (any case, a hyphen or a space between the tens and the ones) or digits replaced by the
/// new one in the same style.
pub fn count_words(text: &str, old: usize, new: usize) -> String {
    let mut out = text.replace("{Count}", &words(new, true)).replace("{count}", &words(new, false));
    if old == new || old == 0 {
        return out;
    }
    let lower = out.to_ascii_lowercase();
    let spelled = words(old, false);
    let candidates = [spelled.clone(), spelled.replace('-', " "), old.to_string()];
    for cand in candidates.iter().filter(|c| !c.is_empty()) {
        let mut from = 0;
        let mut result = String::with_capacity(out.len());
        let mut changed = false;
        while let Some(pos) = lower.get(from..).and_then(|s| s.find(cand.as_str())).map(|p| p + from) {
            let end = pos + cand.len();
            let before = out.get(..pos).and_then(|s| s.chars().next_back());
            let after = out.get(end..).and_then(|s| s.chars().next());
            let whole = before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric() && c != '-');
            result.push_str(out.get(from..pos).unwrap_or_default());
            let found = out.get(pos..end).unwrap_or_default();
            if whole {
                let replacement = if cand.chars().all(|c| c.is_ascii_digit()) {
                    new.to_string()
                } else {
                    let w = words(new, found.chars().next().is_some_and(char::is_uppercase));
                    if found.contains(' ') { w.replace('-', " ") } else { w }
                };
                result.push_str(&replacement);
                changed = true;
            } else {
                result.push_str(found);
            }
            from = end;
        }
        if changed {
            result.push_str(out.get(from..).unwrap_or_default());
            out = result;
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// places

/// Give every entry of `plan` without a `place` the place Google Photos shows for its first shot
/// that has one (a `photos locations` file read with [`crate::places::Places::from_json`], shots
/// matched by file name): [`crate::places::fill`] on the plan as written, so a place a note
/// gives is kept. The labels of the entries that got one.
pub fn fill_places(plan: &mut Value, places: &crate::places::Places) -> Vec<String> {
    let mut filled = Vec::new();
    for e in plan.get_mut("entries").and_then(Value::as_array_mut).into_iter().flatten() {
        if e.get("place").and_then(Value::as_str).is_some_and(|p| !p.trim().is_empty()) {
            continue;
        }
        let shots = e.get("shots").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
        let Some(place) = shots.iter().find_map(|s| places.of(s.get("name").and_then(Value::as_str)?)).map(str::to_string) else { continue };
        if let Some(o) = e.as_object_mut() {
            filled.push(o.get("label").and_then(Value::as_str).unwrap_or_default().to_string());
            o.insert("place".into(), json!(place));
        }
    }
    filled
}

// ---------------------------------------------------------------------------------------------
// another language

/// The plan `plan` read in another language: `lines` is `{"intro", "outro", "lines":
/// {"<label>": "…"}}`. Every entry with a line needs one (the error lists those without);
/// entries without a line stay silent. `language` is set ("ml"), the respellings and `say` (made
/// for the English voice) cleared; chips and on-screen text stay as written. Also returns notes
/// (labels in `lines` that match no entry).
pub fn with_lines(plan: &Value, lines: &Value, language: &str) -> Result<(Value, Vec<String>), AutoCutError> {
    let language = language.trim();
    if language.is_empty() || language.len() > 16 || !language.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(bad("the language is a tag like ml"));
    }
    let by_label = lines.get("lines").and_then(Value::as_object).ok_or_else(|| bad("the lines file has no `lines` object keyed by entry label"))?;
    let entries = plan.get("entries").and_then(Value::as_array).ok_or_else(|| bad("the plan has no entries"))?;
    let english = plan["script"]["lines"].as_array().cloned().unwrap_or_default();
    let mut out_lines = Vec::with_capacity(entries.len());
    let mut missing = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let label = e.get("label").and_then(Value::as_str).unwrap_or_default();
        let spoken = english.get(i).and_then(Value::as_str).is_some_and(|l| !l.trim().is_empty());
        if !spoken {
            out_lines.push(json!(""));
            continue;
        }
        match by_label.get(label).and_then(Value::as_str).filter(|l| !l.trim().is_empty()) {
            Some(l) => out_lines.push(json!(l)),
            None => {
                missing.push(label.to_string());
                out_lines.push(json!(""));
            }
        }
    }
    if !missing.is_empty() {
        return Err(bad(format!(
            "no {language} line for {} spoken entr{}: {}",
            missing.len(),
            if missing.len() == 1 { "y" } else { "ies" },
            missing.join("; ")
        )));
    }
    let part = |k: &str| lines.get(k).and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    let (Some(intro), Some(outro)) = (part("intro"), part("outro")) else {
        return Err(bad(format!("the {language} lines need an intro and an outro")));
    };
    let labels: Vec<&str> = entries.iter().filter_map(|e| e.get("label").and_then(Value::as_str)).collect();
    let notes: Vec<String> =
        by_label.keys().filter(|k| !labels.contains(&k.as_str())).map(|k| format!("{language} lines: no entry is labelled \"{}\"", clip(k))).collect();
    let mut out = plan.clone();
    let script = out.get_mut("script").and_then(Value::as_object_mut).ok_or_else(|| bad("the plan has no script"))?;
    script.insert("lines".into(), Value::Array(out_lines));
    script.insert("intro".into(), json!(intro));
    script.insert("outro".into(), json!(outro));
    script.insert("pronounce".into(), json!({}));
    script.remove("say");
    if let Some(o) = out.as_object_mut() {
        o.insert("language".into(), json!(language));
    }
    Ok((out, notes))
}

#[cfg(test)]
#[path = "plan_patch_tests.rs"]
mod tests;

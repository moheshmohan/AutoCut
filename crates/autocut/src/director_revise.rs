//! Revising a directed edit from notes ("1:10 is too long", "warmer at the end", "fewer barrel
//! rolls"): the notes become changes to the saved plan — the common ones read by rules and taken
//! literally, the local model reading only what the rules could not — each checked and applied,
//! and the edit is rebuilt from the changed plan, not planned again. The report's `changes` lists what changed and what was not
//! understood.
//!
//! A change ([`Op`]) names its shot by number (`shot`, from 1) or by a time in the last cut
//! (`at`, `"1:10"` or seconds):
//!
//! | op | does |
//! |---|---|
//! | `shorter` / `longer` | the shot `beats` shorter or longer (2 by default; the edit's length changes with it) |
//! | `remove` | the shot left out |
//! | `camera` | the shot's camera (one its clip can use, fitting its beats) |
//! | `fewer` | at most `max` shots with `camera` (half of them by default; `no barrel rolls`: 0), the rest held or pushed in |
//! | `grade` | the shots between `from` and `to` (all without them) warmer or cooler (`warmth`, ±25), brighter or darker (`brightness`, stops ±0.8), more or less vivid (`saturation`, ±25), over the colour pass |
//! | `text` / `title` | a shot's text, the title (English) |
//! | `style` / `look` | the cutting style (subtle, energetic, cinematic), the colour pass's look |
//! | `swap` | another moment (`clip`) in the shot's place |

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{Camera, Candidate, Kind, Options, Plan, beats_for, chars, english, finite, title_case, window_from};
use crate::AutoCutError;
use crate::llm::ChatModel;

/// Beats a shot is made shorter or longer by default.
const STEP_BEATS: u32 = 2;
/// A grade's reach: warmth (Lumetri temperature), brightness (stops), saturation.
pub const MAX_WARMTH: f64 = 25.0;
pub const MAX_BRIGHTNESS: f64 = 0.8;
pub const MAX_SATURATION: f64 = 25.0;
/// Most changes taken from one set of notes.
const MAX_OPS: usize = 32;
/// Longest notes read, characters.
pub const MAX_NOTES_CHARS: usize = 2000;

/// A shot's colour asked for in a revision, over the colour pass's balance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Grade {
    /// Lumetri temperature added (+ warmer).
    pub warmth: f64,
    /// Exposure added, stops.
    pub brightness: f64,
    /// Saturation added (100 is none).
    pub saturation: f64,
}

impl Grade {
    pub fn is_none(&self) -> bool {
        *self == Grade::default()
    }

    fn add(&mut self, o: &Grade) {
        self.warmth = (self.warmth + o.warmth).clamp(-MAX_WARMTH, MAX_WARMTH);
        self.brightness = (self.brightness + o.brightness).clamp(-MAX_BRIGHTNESS, MAX_BRIGHTNESS);
        self.saturation = (self.saturation + o.saturation).clamp(-MAX_SATURATION, MAX_SATURATION);
    }
}

/// Which shot a change is about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Where {
    /// Its number, from 1.
    Shot(usize),
    /// A time in the last cut, seconds.
    At(f64),
}

/// One change to the plan.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Shorter(Where, u32),
    Longer(Where, u32),
    Remove(Where),
    Camera(Where, Camera),
    Fewer(Camera, Option<u32>),
    Grade(Option<f64>, Option<f64>, Grade),
    Text(Where, String),
    Title(String),
    Style(String),
    Look(String),
    Swap(Where, String),
}

/// `"1:10"`, `"0:05.5"`, `"70"` or a number: seconds.
pub fn time_of(v: &Value) -> Option<f64> {
    if let Some(x) = v.as_f64() {
        return (x.is_finite() && x >= 0.0).then_some(x);
    }
    let s = v.as_str()?.trim();
    let mut total = 0.0;
    for part in s.split(':') {
        let x: f64 = part.trim().parse().ok().filter(|x: &f64| x.is_finite() && *x >= 0.0)?;
        total = total * 60.0 + x;
    }
    (s.split(':').count() <= 3).then_some(total)
}

fn where_of(v: &Value) -> Option<Where> {
    if let Some(n) = v["shot"].as_u64().filter(|n| *n >= 1) {
        return Some(Where::Shot(n as usize));
    }
    time_of(&v["at"]).map(Where::At)
}

/// The changes a model's answer asks for (`{"ops": [...]}`), and what in it was not usable.
pub fn ops_from(v: &Value) -> (Vec<Op>, Vec<String>) {
    let mut ops = Vec::new();
    let mut bad = Vec::new();
    let list = v["ops"].as_array().or_else(|| v["changes"].as_array()).map(Vec::as_slice).unwrap_or_default();
    for o in list.iter().take(MAX_OPS) {
        let kind = o["op"].as_str().unwrap_or_default();
        let beats = o["beats"].as_u64().map_or(STEP_BEATS, |b| b.clamp(1, 16) as u32);
        let cam = || o["camera"].as_str().and_then(Camera::from_id);
        let text = |k: &str| o[k].as_str().map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
        let op = match (kind, where_of(o)) {
            ("shorter", Some(w)) => Some(Op::Shorter(w, beats)),
            ("longer", Some(w)) => Some(Op::Longer(w, beats)),
            ("remove", Some(w)) => Some(Op::Remove(w)),
            ("camera", Some(w)) => cam().map(|c| Op::Camera(w, c)),
            ("fewer", _) => cam().map(|c| Op::Fewer(c, o["max"].as_u64().map(|m| m.min(64) as u32))),
            ("grade", _) => {
                let g = Grade {
                    warmth: finite(&o["warmth"]).unwrap_or(0.0).clamp(-MAX_WARMTH, MAX_WARMTH),
                    brightness: finite(&o["brightness"]).unwrap_or(0.0).clamp(-MAX_BRIGHTNESS, MAX_BRIGHTNESS),
                    saturation: finite(&o["saturation"]).unwrap_or(0.0).clamp(-MAX_SATURATION, MAX_SATURATION),
                };
                (!g.is_none()).then(|| Op::Grade(time_of(&o["from"]), time_of(&o["to"]), g))
            }
            ("text", Some(w)) => text("text").map(|t| Op::Text(w, t)),
            ("title", _) => text("text").or_else(|| text("title")).map(Op::Title),
            ("style", _) => text("style").map(Op::Style),
            ("look", _) => text("look").map(Op::Look),
            ("swap", Some(w)) => text("clip").map(|c| Op::Swap(w, c)),
            _ => None,
        };
        match op {
            Some(op) => ops.push(op),
            None => bad.push(format!("not usable: {}", chars(&o.to_string(), 120))),
        }
    }
    (ops, bad)
}

/// The cameras notes name, by the words people use.
const CAMERA_WORDS: &[(&str, &[Camera])] = &[
    ("barrel roll", &[Camera::BarrelRoll]),
    ("whip", &[Camera::WhipPan]),
    ("orbit", &[Camera::Orbit]),
    ("spin", &[Camera::Orbit, Camera::BarrelRoll]),
    ("tiny planet", &[Camera::PlanetIn, Camera::PlanetOut]),
    ("planet", &[Camera::PlanetIn, Camera::PlanetOut]),
    ("push in", &[Camera::PushIn]),
    ("zoom", &[Camera::PushIn]),
    ("drift", &[Camera::Drift]),
    ("speed ramp", &[Camera::SpeedRamp]),
    ("slow motion", &[Camera::SpeedRamp]),
    ("hyperlapse", &[Camera::Hyperlapse]),
    ("reveal", &[Camera::Reveal]),
    ("follow", &[Camera::Follow]),
];

/// The times written in a sentence (`1:10`, `0:05`).
fn times_in(s: &str) -> Vec<f64> {
    s.split(|c: char| !(c.is_ascii_digit() || c == ':' || c == '.')).filter(|w| w.contains(':')).filter_map(|w| time_of(&json!(w.trim_matches('.')))).collect()
}

/// The changes the common notes ask for, read by rules (no model), and the sentences not read.
/// `length` is the last cut's length, seconds (the start and the end are its first and last
/// quarter).
pub fn rules_ops(notes: &str, length: f64) -> (Vec<Op>, Vec<String>) {
    let mut ops = Vec::new();
    let mut unread = Vec::new();
    let notes = chars(notes, MAX_NOTES_CHARS);
    for raw in notes.split(['\n', ';', '.']).flat_map(|s| s.split(" and ")).map(str::trim).filter(|s| !s.is_empty()) {
        let s = raw.to_lowercase();
        let has = |w: &str| s.contains(w);
        let times = times_in(&s);
        let at = times.first().map(|t| Where::At(*t));
        let quarter = length.max(1.0) / 4.0;
        let range = if times.len() >= 2 {
            (times.first().copied(), times.get(1).copied())
        } else if let Some(t) = times.first() {
            (Some((t - 2.0).max(0.0)), Some(t + 2.0))
        } else if has("end") || has("ending") || has("last") {
            (Some(length - quarter), None)
        } else if has("start") || has("beginning") || has("intro") || has("opening") || has("first") {
            (None, Some(quarter))
        } else {
            (None, None)
        };
        let before = ops.len();
        // the cut's length
        if (has("too long") || has("drags") || has("shorter") || has("trim")) && at.is_some() {
            ops.extend(at.map(|w| Op::Shorter(w, STEP_BEATS)));
        } else if (has("too short") || has("longer") || has("hold it")) && at.is_some() {
            ops.extend(at.map(|w| Op::Longer(w, STEP_BEATS)));
        } else if (has("remove") || has("cut out") || has("drop") || has("delete") || has("get rid")) && at.is_some() {
            ops.extend(at.map(Op::Remove));
        }
        // the cameras
        let none = has("no ") || has("without") || has("get rid");
        if has("fewer") || has("less ") || has("too many") || none {
            for (word, cams) in CAMERA_WORDS {
                if has(word) {
                    for c in *cams {
                        ops.push(Op::Fewer(*c, none.then_some(0)));
                    }
                    break;
                }
            }
        }
        // the colour
        let mut g = Grade::default();
        if has("warmer") {
            g.warmth = 8.0;
        } else if has("cooler") || has("colder") {
            g.warmth = -8.0;
        }
        if has("brighter") || has("too dark") {
            g.brightness = 0.25;
        } else if has("darker") || has("too bright") {
            g.brightness = -0.25;
        }
        if has("more colour") || has("more color") || has("more vivid") || has("more saturated") || has("punchier colour") {
            g.saturation = 8.0;
        } else if has("less saturated") || has("muted") || has("too colourful") || has("too colorful") {
            g.saturation = -8.0;
        }
        if !g.is_none() {
            ops.push(Op::Grade(range.0, range.1, g));
        }
        // the style
        if has("more energetic") || has("punchier") || has("livelier") || has("more exciting") {
            ops.push(Op::Style("energetic".into()));
        } else if has("calmer") || has("more subtle") || has("less busy") || has("simpler") {
            ops.push(Op::Style("subtle".into()));
        } else if has("cinematic") {
            ops.push(Op::Style("cinematic".into()));
        }
        // the title: `title "…"` or `title: …`
        if let Some(i) = s.find("title") {
            let rest = raw.get(i + 5..).unwrap_or_default().trim_start_matches([' ', ':', '=']).trim();
            let t = rest.trim_matches(['"', '“', '”', '\'']).trim();
            if !t.is_empty() && (rest.starts_with('"') || rest.starts_with('“') || raw.contains(':')) {
                ops.push(Op::Title(t.to_string()));
            }
        }
        if ops.len() == before {
            unread.push(chars(raw, 120));
        }
    }
    ops.truncate(MAX_OPS);
    (ops, unread)
}

const REVISE_SYSTEM: &str = "You revise a short video's edit from the viewer's notes. Turn each note into changes to the shot list you are given, using only the operations described, and nothing the notes do not ask for. Times are in the edit as it is. Answer with JSON only.";

/// The prompt that asks the model for the notes' changes.
pub fn revise_prompt(notes: &str, plan: &Plan, wins: &[(f64, f64)], cands: &[Candidate]) -> String {
    let one = |t: &str| t.replace(['\n', '\r', '|'], " ");
    let mmss = |t: f64| format!("{}:{:04.1}", (t / 60.0).floor() as u64, t % 60.0);
    let mut out = format!("Notes: \"{}\"\n", one(&chars(notes, MAX_NOTES_CHARS)));
    out.push_str("The edit (shot | from–to | clip | kind | camera | what it shows | text):\n");
    for (k, (s, w)) in plan.shots.iter().zip(wins).enumerate() {
        let c = cands.iter().find(|c| c.id == s.clip);
        out.push_str(&format!(
            "{} | {}–{} | {} | {} | {} | {} | {}\n",
            k + 1,
            mmss(w.0),
            mmss(w.1),
            s.clip,
            c.map_or("?", |c| super::kind_word(c.kind)),
            s.camera.id(),
            one(&c.map_or(String::new(), |c| chars(&c.caption, 100))),
            s.text.as_deref().unwrap_or("-")
        ));
    }
    let unused: Vec<String> =
        cands.iter().filter(|c| !plan.shots.iter().any(|s| s.clip == c.id)).take(24).map(|c| format!("{} ({})", c.id, one(&chars(&c.caption, 60)))).collect();
    if !unused.is_empty() {
        out.push_str(&format!("Moments not in the edit: {}\n", unused.join("; ")));
    }
    out.push_str("Cameras: ");
    out.push_str(&Camera::ALL.iter().map(|c| c.id()).collect::<Vec<_>>().join(", "));
    out.push_str("\nOperations (a shot by \"shot\": its number, or \"at\": a time like \"1:10\"):\n");
    out.push_str("- {\"op\":\"shorter\"|\"longer\",\"shot\"|\"at\",\"beats\":2}\n- {\"op\":\"remove\",\"shot\"|\"at\"}\n- {\"op\":\"camera\",\"shot\"|\"at\",\"camera\":name}\n- {\"op\":\"fewer\",\"camera\":name,\"max\":n}\n- {\"op\":\"grade\",\"from\":time?,\"to\":time?,\"warmth\":-25..25,\"brightness\":-0.8..0.8,\"saturation\":-25..25}\n- {\"op\":\"text\",\"shot\"|\"at\",\"text\":english}\n- {\"op\":\"title\",\"text\":english}\n- {\"op\":\"style\",\"style\":\"subtle\"|\"energetic\"|\"cinematic\"}\n- {\"op\":\"look\",\"look\":\"none\"|\"punchy\"|\"soft\"|\"warm\"|\"cool\"|\"golden\"|\"teal\"}\n- {\"op\":\"swap\",\"shot\"|\"at\",\"clip\":id}\n");
    out.push_str("Reply with only: {\"ops\": [ ... ]}\n");
    out
}

/// The notes' changes from the model (asked once more when its answer is not JSON).
pub async fn model_ops(
    model: &dyn ChatModel,
    notes: &str,
    plan: &Plan,
    wins: &[(f64, f64)],
    cands: &[Candidate],
) -> Result<(Vec<Op>, Vec<String>), AutoCutError> {
    let prompt = revise_prompt(notes, plan, wins, cands);
    let v = match model.complete_json(REVISE_SYSTEM, &prompt, &[]).await {
        Ok(v) => v,
        Err(_) => model.complete_json(REVISE_SYSTEM, &prompt, &[]).await?,
    };
    Ok(ops_from(&v))
}

/// The shot a change names: its index.
fn index_of(w: Where, wins: &[(f64, f64)], n: usize) -> Option<usize> {
    match w {
        Where::Shot(k) => k.checked_sub(1).filter(|k| *k < n),
        // (a moment past the end, as a player shows it, is the last shot)
        Where::At(t) => wins
            .iter()
            .position(|(a, b)| t >= *a && t < *b)
            .or_else(|| (t >= wins.last()?.1 && t < wins.last()?.1 + 2.0).then(|| n.saturating_sub(1)))
            .filter(|k| *k < n),
    }
}

/// Apply `ops` to `plan` (`wins` the last cut's shot windows, seconds; `beat` seconds; the style,
/// look and title changes go to `o`): what changed and what could not be done, in words.
pub fn apply(plan: &mut Plan, ops: &[Op], cands: &[Candidate], wins: &[(f64, f64)], beat: f64, o: &mut Options) -> Vec<String> {
    let mut said = Vec::new();
    let find = |id: &str| cands.iter().find(|c| c.id == id);
    // shots are named by the last cut: resolve every reference before any shot moves or goes
    let n0 = plan.shots.len();
    let mut gone: Vec<usize> = Vec::new();
    let mut targets: Vec<Option<usize>> = Vec::with_capacity(ops.len());
    for op in ops {
        let w = match op {
            Op::Shorter(w, _) | Op::Longer(w, _) | Op::Remove(w) | Op::Camera(w, _) | Op::Text(w, _) | Op::Swap(w, _) => Some(*w),
            _ => None,
        };
        targets.push(w.and_then(|w| index_of(w, wins, n0)));
    }
    for (op, k) in ops.iter().zip(&targets) {
        let label = |k: usize| format!("shot {}", k + 1);
        match (op, *k) {
            (Op::Shorter(_, b), Some(k)) => {
                let Some(s) = plan.shots.get_mut(k) else { continue };
                let min = beats_for(s.camera.min_seconds(), beat).max(2);
                let to = s.beats.saturating_sub(*b).max(min);
                if to < s.beats {
                    said.push(format!("{}: {} beats shorter ({} → {})", label(k), s.beats - to, s.beats, to));
                    s.beats = to;
                } else {
                    said.push(format!("{}: already as short as its camera allows", label(k)));
                }
            }
            (Op::Longer(_, b), Some(k)) => {
                let Some(s) = plan.shots.get_mut(k) else { continue };
                let to = s.beats + b;
                match find(&s.clip).and_then(|c| window_from(c, f64::from(to) * beat * s.camera.speed())) {
                    Some(from) => {
                        said.push(format!("{}: {b} beats longer ({} → {to})", label(k), s.beats));
                        s.beats = to;
                        s.from = super::round2(from);
                    }
                    None => said.push(format!("{}: its clip has no more footage to hold for", label(k))),
                }
            }
            (Op::Remove(_), Some(k)) => {
                if n0 - gone.len() <= 2 {
                    said.push(format!("{}: kept (an edit needs two shots)", label(k)));
                } else if !gone.contains(&k) {
                    gone.push(k);
                    said.push(format!("{}: removed", label(k)));
                }
            }
            (Op::Camera(_, cam), Some(k)) => {
                let Some(s) = plan.shots.get_mut(k) else { continue };
                let fits = find(&s.clip).is_some_and(|c| cam.for_kind(c.kind)) && f64::from(s.beats) * beat + 1e-9 >= cam.min_seconds();
                if fits {
                    said.push(format!("{}: camera {} → {}", label(k), s.camera.id(), cam.id()));
                    s.camera = *cam;
                } else {
                    said.push(format!("{}: {} does not fit it (its clip, or its {} beats)", label(k), cam.id(), s.beats));
                }
            }
            (Op::Fewer(cam, max), _) => {
                let with: Vec<usize> = plan.shots.iter().enumerate().filter(|(_, s)| s.camera == *cam).map(|(i, _)| i).collect();
                let keep = max.map_or(with.len() / 2, |m| m as usize);
                for &i in with.iter().skip(keep) {
                    let Some(s) = plan.shots.get_mut(i) else { continue };
                    let Some(c) = find(&s.clip) else { continue };
                    // a quieter camera its clip can use, fitting its beats
                    let instead = [Camera::Drift, Camera::PushIn, Camera::Hold]
                        .into_iter()
                        .find(|m| *m != *cam && m.for_kind(c.kind) && f64::from(s.beats) * beat + 1e-9 >= m.min_seconds())
                        .unwrap_or(Camera::Hold);
                    said.push(format!("{}: camera {} → {}", label(i), s.camera.id(), instead.id()));
                    s.camera = instead;
                }
                if with.len() <= keep {
                    said.push(format!("{}: already {} or fewer", cam.id(), keep));
                }
            }
            (Op::Grade(from, to, g), _) => {
                let (a, b) = (from.unwrap_or(0.0), to.unwrap_or(f64::MAX));
                let mut touched = Vec::new();
                for (i, (s, w)) in plan.shots.iter_mut().zip(wins).enumerate() {
                    if w.1 > a && w.0 < b {
                        let mut cur = s.grade.unwrap_or_default();
                        cur.add(g);
                        s.grade = (!cur.is_none()).then_some(cur);
                        touched.push(i + 1);
                    }
                }
                let what = [
                    (g.warmth > 0.0, "warmer"),
                    (g.warmth < 0.0, "cooler"),
                    (g.brightness > 0.0, "brighter"),
                    (g.brightness < 0.0, "darker"),
                    (g.saturation > 0.0, "more vivid"),
                    (g.saturation < 0.0, "less vivid"),
                ]
                .iter()
                .filter(|(on, _)| *on)
                .map(|(_, w)| *w)
                .collect::<Vec<_>>()
                .join(", ");
                if touched.is_empty() {
                    said.push(format!("{what}: no shot there"));
                } else {
                    said.push(format!("shots {}: {what}", touched.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")));
                }
            }
            (Op::Text(_, t), Some(k)) => {
                let Some(s) = plan.shots.get_mut(k) else { continue };
                if english(t) {
                    said.push(format!("{}: text \"{}\"", label(k), chars(t, super::MAX_TITLE_CHARS)));
                    s.text = Some(chars(t, super::MAX_TITLE_CHARS));
                } else {
                    said.push(format!("{}: the text is not English: left as it was", label(k)));
                }
            }
            (Op::Title(t), _) => {
                if english(t) {
                    let t = title_case(&chars(t, super::MAX_TITLE_CHARS));
                    said.push(format!("title: \"{}\" → \"{t}\"", plan.title));
                    plan.title = t.clone();
                    o.title = Some(t.clone());
                    if let Some(first) = plan.shots.first_mut() {
                        first.text = Some(t);
                    }
                } else {
                    said.push("the title is not English: left as it was".into());
                }
            }
            (Op::Style(st), _) => match super::cuts::Style::from_params(&json!({"style": st}), &o.style) {
                Ok(s) => {
                    said.push(format!("style: {st}"));
                    o.style = s;
                }
                Err(e) => said.push(format!("style: {e}")),
            },
            (Op::Look(l), _) => {
                let l = l.to_ascii_lowercase();
                if super::LOOKS.contains(&l.as_str()) {
                    said.push(format!("look: {} → {l}", o.look));
                    o.look = l;
                } else {
                    said.push(format!("look: no look `{l}`"));
                }
            }
            (Op::Swap(_, id), Some(k)) => {
                let used = plan.shots.iter().any(|s| s.clip == *id);
                let Some(s) = plan.shots.get_mut(k) else { continue };
                match find(id) {
                    Some(c) if !used => {
                        let cam = if s.camera.for_kind(c.kind) {
                            s.camera
                        } else if c.kind == Kind::Vr {
                            Camera::Drift
                        } else {
                            Camera::PushIn
                        };
                        match window_from(c, f64::from(s.beats) * beat * cam.speed()) {
                            Some(from) => {
                                said.push(format!("{}: {} → {} ({})", label(k), s.clip, c.id, chars(&c.caption, 60)));
                                s.clip = c.id.clone();
                                s.from = super::round2(from);
                                s.camera = cam;
                                s.look = c.looks.first().map(|l| l.id.clone());
                                s.to = None;
                            }
                            None => said.push(format!("{}: {} is too short for it", label(k), c.id)),
                        }
                    }
                    Some(_) => said.push(format!("{}: {id} is already in the edit", label(k))),
                    None => said.push(format!("{}: no moment {id}", label(k))),
                }
            }
            (_, None) => said.push(format!("no shot there: {}", describe(op))),
        }
    }
    // the removed, last (the numbers above are the last cut's)
    gone.sort_unstable();
    for k in gone.into_iter().rev() {
        if k < plan.shots.len() {
            plan.shots.remove(k);
        }
    }
    if let (Some(first), Some(t)) = (plan.shots.first_mut(), Some(plan.title.clone())) {
        first.text.get_or_insert(t);
    }
    said
}

fn describe(op: &Op) -> String {
    match op {
        Op::Shorter(w, _) => format!("shorter at {w:?}"),
        Op::Longer(w, _) => format!("longer at {w:?}"),
        Op::Remove(w) => format!("remove at {w:?}"),
        Op::Camera(w, c) => format!("camera {} at {w:?}", c.id()),
        Op::Text(w, _) => format!("text at {w:?}"),
        Op::Swap(w, c) => format!("swap in {c} at {w:?}"),
        other => format!("{other:?}"),
    }
}

/// The edit a revision starts from: its plan, the options it was made with, and its report's
/// file.
#[derive(Clone, Debug)]
pub struct Previous {
    pub plan: Plan,
    pub options: Value,
    pub path: Option<String>,
}

/// Largest report read, bytes.
const MAX_REPORT_BYTES: u64 = 32 << 20;

/// The saved edit: a report (`<out>.report.json`, or its JSON) with its `plan` and `options`.
pub fn load(v: &Value) -> Result<Previous, String> {
    let (report, path) = match v {
        Value::String(p) => {
            let path = filmcraft_engine::export_tools::expand_home(p.trim());
            let size = std::fs::metadata(&path).map_err(|e| format!("previous: {path}: {e}"))?.len();
            if size > MAX_REPORT_BYTES {
                return Err(format!("previous: {path} is too large to be a director's report"));
            }
            let text = std::fs::read_to_string(&path).map_err(|e| format!("previous: {path}: {e}"))?;
            (serde_json::from_str::<Value>(&text).map_err(|e| format!("previous: {path}: not JSON ({e})"))?, Some(path))
        }
        Value::Object(_) => (v.clone(), None),
        _ => return Err("revise: `previous` is the report of the edit to revise (<out>.report.json)".into()),
    };
    let plan: Plan = serde_json::from_value(report["plan"].clone()).map_err(|e| format!("previous: the report has no plan to revise ({e})"))?;
    if plan.shots.is_empty() {
        return Err("previous: the report's plan has no shots".into());
    }
    let options = if report["options"].is_object() { report["options"].clone() } else { json!({}) };
    Ok(Previous { plan, options, path })
}

/// The options a report keeps for a revision: what was asked, without the revision's own keys.
pub fn saved_options(p: &Value) -> Value {
    let mut o = p.clone();
    if let Some(m) = o.as_object_mut() {
        for k in ["revise", "previous", "wait", "dryRun"] {
            m.remove(k);
        }
    }
    o
}

/// `old` options with the `new` ones over them; the tempo kept (`bpm`, so the saved plan's beats
/// stay where they were) and, unless a new `export` is given, the old one's name with
/// `-revised`.
pub fn merge(old: &Value, new: &Value, bpm: f64) -> Value {
    let mut out = if old.is_object() { old.clone() } else { json!({}) };
    if let (Some(m), Some(n)) = (out.as_object_mut(), new.as_object()) {
        for (k, v) in n {
            if !v.is_null() {
                m.insert(k.clone(), v.clone());
            }
        }
        if !n.contains_key("export")
            && let Some(e) = m.get("export").and_then(Value::as_str).map(str::to_string)
        {
            let p = std::path::Path::new(&e);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("edit");
            let stem = stem.strip_suffix("-revised").unwrap_or(stem);
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("mp4");
            m.insert("export".into(), json!(p.with_file_name(format!("{stem}-revised.{ext}")).display().to_string()));
        }
        if !m.contains_key("bpm") && bpm.is_finite() && (70.0..=150.0).contains(&bpm) {
            m.insert("bpm".into(), json!(bpm));
        }
    }
    out
}

/// The saved plan on this run's moments: each shot found again by its file and where it starts
/// (a moment's ids can change with the footage); the shots whose moment is gone, in words.
pub fn remap(plan: &Plan, cands: &[Candidate]) -> (Plan, Vec<String>) {
    let mut out = plan.clone();
    let mut lost = Vec::new();
    out.shots.clear();
    for (k, s) in plan.shots.iter().enumerate() {
        let same_file: Vec<&Candidate> = cands.iter().filter(|c| !s.file.is_empty() && c.name == s.file).collect();
        // the same moment (the footage as it was), else the one holding its start, else the nearest
        let found = same_file
            .iter()
            .find(|c| c.id == s.clip)
            .or_else(|| same_file.iter().find(|c| s.from >= c.start - 0.05 && s.from <= c.end))
            .or_else(|| same_file.iter().min_by(|a, b| (a.start - s.from).abs().total_cmp(&(b.start - s.from).abs())))
            .copied()
            .or_else(|| cands.iter().find(|c| s.file.is_empty() && c.id == s.clip));
        match found {
            Some(c) => {
                let mut s = s.clone();
                s.clip = c.id.clone();
                out.shots.push(s);
            }
            None => lost.push(format!(
                "shot {}: its moment ({} at {:.1} s) is not in the footage any more: left out",
                k + 1,
                if s.file.is_empty() { s.clip.as_str() } else { s.file.as_str() },
                s.from
            )),
        }
    }
    (out, lost)
}

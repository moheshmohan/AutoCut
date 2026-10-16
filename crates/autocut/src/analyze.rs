//! Clip analysis: which media items show a subject ("robot"), what it is, when it was recorded,
//! and the moment that shows it best — the facts a showcase edit is planned from.
//!
//! For each video item, frames are sampled through the engine (`media.frame`) and a grounding
//! vision model (Qwen2.5-VL class, [`crate::config::Config::analysis_model`]) is asked, frame
//! by frame, to *locate* the subject: a box, or none. Locating is what these models do reliably;
//! a box also says how big and where the subject is, so the best moment is the frame where it is
//! largest. On the best frames, larger, the model then reads the text it can see — on the
//! subject itself, and on signs and screens around it ([`FrameText`]) — since a robot's name is
//! often printed on it. One more request per clip, over its best frames and with that text, names
//! and describes it; a proper name survives only when the text read on the subject (or a sign
//! that introduces it) spells it, or it is a well-known robot or maker (a 7B model otherwise
//! invents brand names).
//!
//! Capture dates come from `media.captureDate` (sidecars from the Google Photos picker, or the
//! file's own metadata), so results sort chronologically.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, Executor};

/// Frames sampled per clip, at most.
pub const MAX_FRAMES: usize = 8;

/// The order a clip's sample times are looked at when screening with `screen` frames: `screen`
/// of them spread across the clip first, then the rest. With `screen` 0, or at least as many as
/// there are times, the plain order.
fn screening_order(times: &[f64], screen: usize) -> (Vec<f64>, usize) {
    let n = times.len();
    if screen == 0 || screen >= n {
        return (times.to_vec(), n);
    }
    let mut first: Vec<usize> = (0..screen).map(|i| (((i as f64 + 0.5) * n as f64 / screen as f64) - 0.5).round() as usize).map(|k| k.min(n - 1)).collect();
    first.dedup();
    let rest = (0..n).filter(|k| !first.contains(k));
    let order: Vec<f64> = first.iter().copied().chain(rest).filter_map(|k| times.get(k).copied()).collect();
    (order, first.len())
}
/// Seconds between samples on short clips.
const SAMPLE_EVERY: f64 = 3.0;
/// A subject box smaller than this share of the frame is not "showing" it.
const MIN_AREA: f64 = 0.01;
/// Frames read for text per clip (where the subject is largest), and their longest side: small
/// print needs more pixels than locating does.
const TEXT_FRAMES: usize = 2;
const TEXT_SIDE: u32 = 1280;
/// Lines of text kept per place, and characters per line: a model's reply is hostile input, and
/// the lines go into the next prompt.
const MAX_TEXT_LINES: usize = 12;
const MAX_TEXT_CHARS: usize = 60;
/// A label longer than this is cut before it is checked (the model is asked for 2-5 words).
const MAX_LABEL_CHARS: usize = 120;

/// One sampled frame.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameFact {
    pub seconds: f64,
    pub present: bool,
    /// Share of the frame the subject's box covers (0..1).
    pub area: f64,
    /// Box centre, 0..1 of the frame (x right, y down).
    pub center: [f64; 2],
}

/// Everything known about one clip.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ClipFacts {
    pub item: u64,
    pub name: String,
    pub seconds: f64,
    /// RFC 3339 UTC.
    pub created: Option<String>,
    pub unix: Option<i64>,
    /// The subject is in the clip.
    pub present: bool,
    /// Share of sampled frames that show it.
    pub coverage: f64,
    /// What it is ("Pepper humanoid robot", "floor-cleaning robot").
    pub label: String,
    pub kind: String,
    pub description: String,
    /// Text read in its best frames (empty in reports from before text was read).
    #[serde(default)]
    pub text: FrameText,
    /// Where it shows best (seconds into the clip).
    pub best: f64,
    pub frames: Vec<FrameFact>,
    /// Which robot it is: clips showing the same one share it ([`crate::identity`]); `None` in
    /// reports from before identities, and for clips that do not show the subject.
    #[serde(default)]
    pub identity: Option<u32>,
}

/// Text a vision model read in a clip's best frames, by where it is: what is printed on the
/// subject can name it; signs, booths and screens around it mostly name something else (a shop,
/// a university, a sponsor).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct FrameText {
    /// Printed, painted or stuck on the subject itself: a name, a logo, a model number.
    #[serde(default)]
    pub on_subject: Vec<String>,
    /// Signs, booths, banners, shop windows and walls around it, and anything on a screen (its
    /// own included).
    #[serde(default)]
    pub nearby: Vec<String>,
}

/// Words that make a sign *introduce* a robot ("Ameca AI Interaction Guide", "Meet NAME", "NAME
/// Social Robot") rather than name a shop, company or event ("QUOKKA", "ETH zürich RSL").
const SIGN_WORDS: &[&str] =
    &["robot", "robots", "robotic", "robotics", "humanoid", "android", "cobot", "ai", "guide", "assistant", "interaction", "meet", "introducing", "demo"];

/// What a model writes when it read nothing.
const NO_TEXT: &[&str] = &["none", "na", "null", "nil", "notext", "nothing", "illegible", "unreadable", "unknown", "blurry", "empty", "text"];

impl FrameText {
    /// A reply to [`read_prompt`]: `{"on_subject": [..], "nearby": [..]}`. Anything else reads as
    /// no text; lines are cleaned (one line, no quotes or control characters), placeholders
    /// ("none") dropped, repeats merged, and lengths and counts capped.
    pub fn from_reply(v: &Value) -> Self {
        Self { on_subject: text_lines(v.get("on_subject")), nearby: text_lines(v.get("nearby")) }
    }

    pub fn is_empty(&self) -> bool {
        self.on_subject.is_empty() && self.nearby.is_empty()
    }

    /// Add what another frame of the clip shows.
    pub fn merge(&mut self, other: FrameText) {
        for l in other.on_subject {
            push_line(&mut self.on_subject, l);
        }
        for l in other.nearby {
            push_line(&mut self.nearby, l);
        }
    }

    /// The lines that can name the subject, as words: everything on it, and signs around it that
    /// introduce a robot — never a bare shop, company or event name on a window, booth or screen.
    fn naming_lines(&self) -> Vec<Vec<String>> {
        let on = self.on_subject.iter().take(MAX_TEXT_LINES).map(|l| words(l));
        let signs = self.nearby.iter().take(MAX_TEXT_LINES).map(|l| words(l)).filter(|w| w.iter().any(|x| SIGN_WORDS.contains(&x.as_str())));
        on.chain(signs).collect()
    }

    /// Whether a capitalised word of `label` is spelled out by text that can name the subject
    /// (on it, or on a sign introducing it): a name read, not one guessed.
    pub fn names(&self, label: &str) -> bool {
        let (raw, tokens) = label_tokens(label);
        let naming = self.naming_lines();
        raw.iter().enumerate().any(|(i, w)| capitalised(w) && naming.iter().any(|l| spelled(&tokens, i, l)))
    }
}

/// Clean lines out of a reply's list (or lone string); a line may also come as `{"text": ".."}`.
fn text_lines(v: Option<&Value>) -> Vec<String> {
    let items: Vec<&Value> = match v {
        Some(Value::Array(a)) => a.iter().take(MAX_TEXT_LINES * 8).collect(),
        Some(s @ Value::String(_)) => vec![s],
        _ => Vec::new(),
    };
    let mut out = Vec::new();
    for x in items {
        let s = match x {
            Value::String(s) => s.as_str(),
            Value::Object(o) => o.get("text").and_then(Value::as_str).unwrap_or_default(),
            _ => "",
        };
        if let Some(line) = clean_line(s) {
            push_line(&mut out, line);
        }
    }
    out
}

/// One line of read text: no control characters or quotes, single spaces, at most
/// [`MAX_TEXT_CHARS`]; `None` when it holds no letters or digits or says "none".
fn clean_line(s: &str) -> Option<String> {
    let s: String =
        s.chars().take(MAX_TEXT_CHARS * 4).map(|c| if c.is_control() { ' ' } else { c }).filter(|c| !matches!(c, '"' | '“' | '”' | '`' | '\\')).collect();
    let s: String = s.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_TEXT_CHARS).collect();
    let squashed = words(&s).concat();
    (!squashed.is_empty() && !NO_TEXT.contains(&squashed.as_str())).then(|| s.trim().to_string())
}

/// Add `line` unless the list is full or holds it already (in any case or spacing).
fn push_line(lines: &mut Vec<String>, line: String) {
    let key = words(&line).concat();
    if lines.len() < MAX_TEXT_LINES && !lines.iter().any(|l| words(l).concat() == key) {
        lines.push(line);
    }
}

/// The lowercase letter-and-digit runs of `s` ("Swiss-Mile" → ["swiss", "mile"]).
fn words(s: &str) -> Vec<String> {
    s.chars()
        .take(MAX_TEXT_CHARS * 4)
        .collect::<String>()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .take(MAX_TEXT_CHARS)
        .collect()
}

/// A label's words as written and as compared (letters and digits, lowercase), split at spaces
/// and hyphens; at most [`MAX_LABEL_CHARS`] of it.
fn label_tokens(label: &str) -> (Vec<String>, Vec<String>) {
    let label: String = label.trim().chars().take(MAX_LABEL_CHARS).collect();
    let raw: Vec<String> = label.split(|c: char| c.is_whitespace() || c == '-').filter(|w| !w.is_empty()).map(str::to_string).collect();
    let tokens = raw.iter().map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()).collect();
    (raw, tokens)
}

fn capitalised(w: &str) -> bool {
    w.chars().next().is_some_and(char::is_uppercase)
}

/// Whether label word `i` is spelled out by a line's words: alone or joined with its neighbours,
/// it equals one or more consecutive words of the line joined, so "Swiss Mile", "Swiss-Mile" and
/// "SwissMile" all match "SWISS-MILE", and "GO-1" matches "Go1".
fn spelled(tokens: &[String], i: usize, line: &[String]) -> bool {
    const SPAN: usize = 3;
    if tokens.get(i).is_none_or(String::is_empty) {
        return false;
    }
    for a in i.saturating_sub(SPAN - 1)..=i {
        for b in i..(a + SPAN).min(tokens.len()) {
            let joined = tokens.get(a..=b).map(<[String]>::concat).unwrap_or_default();
            for c in 0..line.len() {
                let mut acc = String::new();
                for w in line.iter().skip(c).take(SPAN + 1) {
                    acc.push_str(w);
                    if acc == joined {
                        return true;
                    }
                    if acc.len() >= joined.len() {
                        break;
                    }
                }
            }
        }
    }
    false
}

/// Sample times for a clip of `duration` seconds: every [`SAMPLE_EVERY`] s, at most `max`,
/// skipping the shaky first and last half second.
pub fn sample_times(duration: f64, max: usize) -> Vec<f64> {
    if !duration.is_finite() || duration <= 0.0 || max == 0 {
        return Vec::new();
    }
    let margin = (duration * 0.1).min(0.5);
    let span = (duration - 2.0 * margin).max(0.0);
    let n = ((span / SAMPLE_EVERY).ceil() as usize).clamp(1, max);
    (0..n).map(|i| margin + span * (i as f64 + 0.5) / n as f64).collect()
}

/// The frame Qwen2.5-VL sees for a `w`×`h` image (28-px patches, 1024..1600 patches) — boxes it
/// returns are in these pixels. Mirrors `filmcraft_screenagent::ground::model_frame`.
pub fn model_frame(w: u32, h: u32) -> (f64, f64) {
    let p = 28.0f64;
    let (w, h) = (f64::from(w.max(1)), f64::from(h.max(1)));
    let snap = |v: f64| (v / p).round().max(1.0) * p;
    let (mut fw, mut fh) = (snap(w), snap(h));
    let (min_px, max_px) = (1024.0 * p * p, 1600.0 * p * p);
    if fw * fh > max_px {
        let b = (w * h / max_px).sqrt();
        fw = (w / b / p).floor().max(1.0) * p;
        fh = (h / b / p).floor().max(1.0) * p;
    } else if fw * fh < min_px {
        let b = (min_px / (w * h)).sqrt();
        fw = (w * b / p).ceil().max(1.0) * p;
        fh = (h * b / p).ceil().max(1.0) * p;
    }
    (fw, fh)
}

/// Read a grounding reply: `{"present": bool, "bbox_2d": [x1,y1,x2,y2] | null}` in the model's
/// frame `(fw, fh)`. Hostile numbers mean "not present".
pub fn frame_fact(seconds: f64, reply: &Value, fw: f64, fh: f64) -> FrameFact {
    let none = FrameFact { seconds, present: false, area: 0.0, center: [0.5, 0.5] };
    if reply.get("present").and_then(Value::as_bool) == Some(false) {
        return none;
    }
    let Some(b) = reply.get("bbox_2d").or_else(|| reply.get("box_2d")).and_then(Value::as_array) else { return none };
    let c: Vec<f64> = b.iter().filter_map(Value::as_f64).collect();
    let [x1, y1, x2, y2] = c.as_slice() else { return none };
    if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) || fw <= 0.0 || fh <= 0.0 {
        return none;
    }
    let (x1, x2) = ((x1 / fw).clamp(0.0, 1.0), (x2 / fw).clamp(0.0, 1.0));
    let (y1, y2) = ((y1 / fh).clamp(0.0, 1.0), (y2 / fh).clamp(0.0, 1.0));
    if x2 <= x1 || y2 <= y1 {
        return none;
    }
    let area = (x2 - x1) * (y2 - y1);
    FrameFact { seconds, present: area >= MIN_AREA, area, center: [(x1 + x2) / 2.0, (y1 + y2) / 2.0] }
}

/// The system prompt grounding requests are sent with.
pub const GROUND_SYSTEM: &str = "You locate objects in photos for a video editor.";

/// The user prompt that asks a grounding model where the `subject` is (a box, or not present);
/// read the reply with [`frame_fact`]. Also used by [`crate::qa`] on rendered frames.
pub fn ground_prompt(subject: &str) -> String {
    format!(
        "Is there a {subject} in this image? Answer with JSON only: {{\"present\": true|false, \"bbox_2d\": [x1, y1, x2, y2] of the most prominent {subject} in absolute pixels, or null}}. People, statues, costumes and armour of fictional characters (e.g. Iron Man suits), toys shaped like people and machines that are not a {subject} do not count."
    )
}

/// How to tell commonly filmed robots apart — what a 7B model otherwise confuses (it calls any
/// robot dog "Spot"). Written for this project from public descriptions.
const ROBOT_GUIDE: &str = "Telling robots apart:\n\
- Boston Dynamics Spot: YELLOW and black four-legged robot the size of a large dog, no head, flat sensor block at the front.\n\
- Unitree Go1/Go2 robot dog: smaller GREY or WHITE four-legged robot with a rounded sensor 'face' at the front.\n\
- Pepper: white humanoid about 1.2 m tall, big round eyes, a TABLET ON ITS CHEST, a wheeled base instead of legs.\n\
- NAO: small (58 cm) white humanoid with grey or coloured trim that WALKS on legs.\n\
- Ameca: grey humanoid with a lifelike rubber face and a mechanical neck.\n\
- Industrial robot arms (yellow, orange or white): name a brand ONLY if its logo is readable in the frame.\n\
- Service robots: describe the job ('floor-cleaning robot', 'delivery robot with trays', 'reception robot with a screen').\n\
- Kits, toys and builds: describe them ('tracked robot kit', 'red toy robot', 'robot model kit').\n";

/// What was read in the frames, for the describe prompt: where each line is, and which of it can
/// name the subject.
fn text_note(subject: &str, read: &FrameText) -> String {
    if read.is_empty() {
        return String::new();
    }
    let list = |v: &[String]| v.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", ");
    let mut s = String::from("Text read in the frames:\n");
    if !read.on_subject.is_empty() {
        s.push_str(&format!("- on the {subject} itself: {}\n", list(&read.on_subject)));
    }
    if !read.nearby.is_empty() {
        s.push_str(&format!("- around it (signs, booths, screens): {}\n", list(&read.nearby)));
    }
    s.push_str(&format!(
        "A name or model printed on the {subject} itself is its name: use it in the label rather than a guess from its looks. A sign names it only when it plainly introduces this {subject} (like 'NAME AI Interaction Guide' beside it). A shop, company, university, event or sponsor name on a sign, a booth or a screen is not its name: mention it in the description, never in the label.\n"
    ));
    s
}

fn describe_prompt(subject: &str, read: &FrameText) -> String {
    let guide = if subject.to_lowercase().contains("robot") { ROBOT_GUIDE } else { "" };
    let text = text_note(subject, read);
    format!(
        "These frames come from one video showing a {subject}.\n{guide}{text}Answer with JSON only: {{\"label\": \"what it is in 2-5 words: its name or model when that is printed on it, or a brand or model name ONLY if it clearly matches the guide or a logo is readable; otherwise a plain description such as 'floor-cleaning robot' or 'white robot dog'\", \"kind\": \"humanoid|quadruped|arm|wheeled|toy|drone|other\", \"description\": \"one plain sentence: what it looks like and what it is doing (and the text around it, when that says where it is)\"}}"
    )
}

/// Describe a clip from `imgs` (PNG, in time order) and the text read in it ([`ClipFacts::text`]).
async fn describe(model: &dyn ChatModel, subject: &str, imgs: &[Vec<u8>], f: &mut ClipFacts) -> Result<(), AutoCutError> {
    let v = model.complete_json("You describe videos for a video editor.", &describe_prompt(subject, &f.text), imgs).await?;
    let text = |k: &str, n: usize| v.get(k).and_then(Value::as_str).map(|s| s.trim().chars().take(n).collect::<String>()).unwrap_or_default();
    f.kind = text("kind", 20).to_lowercase();
    f.description = text("description", 300);
    f.label = sanitize_label_with(&text("label", 60), &f.kind, subject, &f.text);
    Ok(())
}

fn read_prompt(subject: &str) -> String {
    format!(
        "Read the text in this photo of a {subject}. Copy only text that is sharp enough to read, exactly as written: never guess, complete or translate a word. Keep each line of text whole, as one entry. Sort the lines by where they are:\n- \"on_subject\": printed, painted or stuck on the {subject}'s own body or casing (a name, a logo, a model number, a sticker);\n- \"nearby\": everything else: signs, booths, banners, posters, shop windows and walls around it, and anything shown on a screen, the {subject}'s own screen included.\nAnswer with JSON only: {{\"on_subject\": [\"...\"], \"nearby\": [\"...\"]}}, empty lists where nothing is legible."
    )
}

/// Read the text in `imgs` (PNG, one request each) and merge it; `None` when no frame could be
/// read (the model did not answer), so a relabel keeps what an earlier pass read.
async fn read_text(model: &dyn ChatModel, subject: &str, name: &str, imgs: &[Vec<u8>], progress: &mut dyn FnMut(&str)) -> Option<FrameText> {
    let mut read: Option<FrameText> = None;
    for png in imgs {
        match model.complete_json("You read text in photos for a video editor.", &read_prompt(subject), std::slice::from_ref(png)).await {
            Ok(v) => read.get_or_insert_with(FrameText::default).merge(FrameText::from_reply(&v)),
            Err(e) => progress(&format!("  {name}: text not read ({e})")),
        }
    }
    read
}

/// One frame of `item` as PNG (longest side `max_side`), or `None` when it cannot be had.
async fn frame_png(ex: &dyn Executor, item: u64, seconds: f64, max_side: u32) -> Option<Vec<u8>> {
    let r = ex.execute("media.frame", json!({"item": item, "seconds": seconds, "maxSide": max_side})).await.ok()?;
    r["pngBase64"].as_str().and_then(crate::b64_decode)
}

/// The (up to) `n` sampled moments where the subject is largest, in time order.
fn best_times(frames: &[FrameFact], n: usize) -> Vec<f64> {
    let mut best: Vec<&FrameFact> = frames.iter().filter(|x| x.present && x.seconds.is_finite()).collect();
    best.sort_by(|a, b| b.area.total_cmp(&a.area));
    let mut times: Vec<f64> = best.iter().take(n).map(|x| x.seconds).collect();
    times.sort_by(f64::total_cmp);
    times
}

/// Read the text in a clip's best frames, fetched larger than for locating.
async fn read_clip_text(model: &dyn ChatModel, ex: &dyn Executor, subject: &str, f: &ClipFacts, progress: &mut dyn FnMut(&str)) -> Option<FrameText> {
    let mut imgs = Vec::new();
    for t in best_times(&f.frames, TEXT_FRAMES) {
        if let Some(png) = frame_png(ex, f.item, t, TEXT_SIDE).await {
            imgs.push(png);
        }
    }
    if imgs.is_empty() {
        return None;
    }
    read_text(model, subject, &f.name, &imgs, progress).await
}

/// `; on it: "ARDET", "Swiss-Mile"` for a progress line, or nothing.
fn read_note(read: &FrameText) -> String {
    if read.on_subject.is_empty() {
        return String::new();
    }
    format!("; on it: {}", read.on_subject.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", "))
}

/// Analyse afresh, or (with `earlier`) only read and describe again from an earlier analysis;
/// then tell which clips show the same robot ([`crate::identity::identify`], at most `compare`
/// model comparisons; 0 joins by name only).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    items: Option<&[u64]>,
    frames_max: usize,
    screen: usize,
    earlier: Option<Vec<ClipFacts>>,
    compare: usize,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ClipFacts>, AutoCutError> {
    let mut facts = match earlier {
        Some(mut facts) => {
            relabel(model, ex, subject, &mut facts, progress).await?;
            facts
        }
        None => analyze(model, ex, subject, items, frames_max, screen, progress).await?,
    };
    crate::identity::identify(model, ex, subject.trim(), &mut facts, compare, progress).await;
    Ok(facts)
}

/// Re-describe every clip that shows the subject, from its (up to three) best frames as the
/// earlier analysis found them, reading the text in them again first: names and descriptions
/// improve without locating again.
pub async fn relabel(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    facts: &mut [ClipFacts],
    progress: &mut dyn FnMut(&str),
) -> Result<(), AutoCutError> {
    let n = facts.len();
    for (k, f) in facts.iter_mut().enumerate() {
        if !f.present {
            continue;
        }
        let mut imgs = Vec::new();
        for t in best_times(&f.frames, 3) {
            if let Some(png) = frame_png(ex, f.item, t, 640).await {
                imgs.push(png);
            }
        }
        if imgs.is_empty() {
            continue;
        }
        if let Some(read) = read_clip_text(model, ex, subject, f, progress).await {
            f.text = read;
        }
        let before = f.label.clone();
        match describe(model, subject, &imgs, f).await {
            Ok(()) => progress(&format!("[{}/{n}] {}: {before} → {}{}", k + 1, f.name, f.label, read_note(&f.text))),
            Err(e) => progress(&format!("[{}/{n}] {}: kept `{before}` ({e})", k + 1, f.name)),
        }
    }
    Ok(())
}

/// Proper names a vision model may use in a robot label: well-known robots and makers. Any other
/// capitalised name in a label is more likely invented than recognised (a 7B model labels a
/// floor cleaner "Kangaroo" with confidence), so the label falls back to a plain description.
const KNOWN_NAMES: &[&str] = &[
    "pepper",
    "nao",
    "spot",
    "atlas",
    "stretch",
    "boston",
    "dynamics",
    "softbank",
    "aldebaran",
    "ameca",
    "sophia",
    "hanson",
    "engineered",
    "arts",
    "unitree",
    "go1",
    "go2",
    "b2",
    "h1",
    "g1",
    "optimus",
    "tesla",
    "digit",
    "agility",
    "figure",
    "asimo",
    "honda",
    "roomba",
    "irobot",
    "aibo",
    "sony",
    "vector",
    "cozmo",
    "anki",
    "loona",
    "moxie",
    "temi",
    "promobot",
    "kuka",
    "abb",
    "fanuc",
    "yaskawa",
    "universal",
    "ur5",
    "ur10",
    "baxter",
    "sawyer",
    "anymal",
    "jibo",
    "lego",
    "mindstorms",
    "dji",
    "robomaster",
    "xiaomi",
    "cyberdog",
    "keenon",
    "pudu",
    "bellabot",
    "gausium",
    "lynx",
    "ubtech",
    "walker",
];

/// Whether a label names a well-known robot or maker: a capitalised word of it is one of
/// [`KNOWN_NAMES`] ("Boston Dynamics Spot", "Pepper humanoid robot"; not "red toy robot").
pub fn famous(label: &str) -> bool {
    let (raw, tokens) = label_tokens(label);
    raw.iter().zip(&tokens).any(|(w, lw)| capitalised(w) && KNOWN_NAMES.contains(&lw.as_str()))
}

/// Keep a model's label only if every capitalised word in it is a known name; otherwise say
/// what it is plainly: "{kind} {subject}" ("quadruped robot"). With no text read: see
/// [`sanitize_label_with`].
pub fn sanitize_label(label: &str, kind: &str, subject: &str) -> String {
    sanitize_label_with(label, kind, subject, &FrameText::default())
}

/// Keep a model's label only if every capitalised word in it is read or known: spelled out by
/// the text read on the subject (in any case, spacing or punctuation: "Swiss Mile" for
/// "SWISS-MILE") or on a sign that introduces it ("Ameca AI Interaction Guide"), or else a
/// well-known robot or maker ([`KNOWN_NAMES`]). A name only on a shop window ("QUOKKA"), a booth
/// or a screen (a sponsor's logo) is not read as the subject's. Otherwise say what it is
/// plainly: "{kind} {subject}" ("quadruped robot").
pub fn sanitize_label_with(label: &str, kind: &str, subject: &str, read: &FrameText) -> String {
    let label: String = label.trim().chars().take(MAX_LABEL_CHARS).collect();
    let label = label.trim();
    let (raw, tokens) = label_tokens(label);
    let naming = read.naming_lines();
    let invented = raw.iter().zip(&tokens).enumerate().any(|(i, (w, lw))| {
        capitalised(w)
            && !lw.is_empty()
            && !KNOWN_NAMES.contains(&lw.as_str())
            && lw != subject
            && !["robot", "robotic", "the", "a", "humanoid", "industrial", "toy", "dog", "arm"].contains(&lw.as_str())
            && !naming.iter().any(|l| spelled(&tokens, i, l))
    });
    let plain_kind = match kind.trim() {
        "" | "other" | "none" => String::new(),
        k => format!("{k} "),
    };
    if label.is_empty() || invented {
        return format!("{plain_kind}{subject}");
    }
    // a bare kind ("wheeled") reads as a label once the subject is added
    if label.eq_ignore_ascii_case(kind) {
        return format!("{plain_kind}{subject}");
    }
    label.to_string()
}

/// One clip: frames → grounding → text → description.
#[allow(clippy::too_many_arguments)]
async fn analyze_clip(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    item: u64,
    name: &str,
    seconds: f64,
    frames_max: usize,
    screen: usize,
    progress: &mut dyn FnMut(&str),
) -> Result<ClipFacts, AutoCutError> {
    let mut frames: Vec<FrameFact> = Vec::new();
    let mut pngs: Vec<(f64, f64, Vec<u8>)> = Vec::new(); // (area, seconds, png)
    let (order, screened) = screening_order(&sample_times(seconds, frames_max), screen);
    for (k, t) in order.into_iter().enumerate() {
        // screening: the spread-out first frames show nothing, so the rest are not looked at
        if k == screened && !frames.iter().any(|f| f.present) {
            break;
        }
        let r = match ex.execute("media.frame", json!({"item": item, "seconds": t, "maxSide": 640})).await {
            Ok(r) => r,
            Err(e) => {
                progress(&format!("  frame {t:.1}s of {name}: {e}"));
                continue;
            }
        };
        let Some(png) = r["pngBase64"].as_str().and_then(crate::b64_decode) else { continue };
        let (w, h) = (r["width"].as_u64().unwrap_or(1) as u32, r["height"].as_u64().unwrap_or(1) as u32);
        let (fw, fh) = model_frame(w, h);
        let reply = model.complete_json(GROUND_SYSTEM, &ground_prompt(subject), std::slice::from_ref(&png)).await;
        let f = match reply {
            Ok(v) => frame_fact(t, &v, fw, fh),
            Err(e) => {
                progress(&format!("  {name} @ {t:.1}s: the model did not answer ({e})"));
                FrameFact { seconds: t, present: false, area: 0.0, center: [0.5, 0.5] }
            }
        };
        if f.present {
            pngs.push((f.area, t, png));
        }
        frames.push(f);
    }
    frames.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
    let shown = frames.iter().filter(|f| f.present).count();
    let coverage = if frames.is_empty() { 0.0 } else { shown as f64 / frames.len() as f64 };
    let best = frames.iter().filter(|f| f.present).max_by(|a, b| a.area.total_cmp(&b.area)).map(|f| f.seconds).unwrap_or(seconds / 2.0);
    let mut facts = ClipFacts {
        item,
        name: name.to_string(),
        seconds,
        created: None,
        unix: None,
        present: shown > 0,
        coverage,
        label: String::new(),
        kind: String::new(),
        description: String::new(),
        text: FrameText::default(),
        best,
        frames,
        identity: None,
    };
    if facts.present {
        // the text it carries and that is around it, then a description that can use it
        facts.text = read_clip_text(model, ex, subject, &facts, progress).await.unwrap_or_default();
        // describe from the (up to) three frames where it is largest, in time order
        pngs.sort_by(|a, b| b.0.total_cmp(&a.0));
        pngs.truncate(3);
        pngs.sort_by(|a, b| a.1.total_cmp(&b.1));
        let imgs: Vec<Vec<u8>> = pngs.into_iter().map(|(_, _, p)| p).collect();
        if let Err(e) = describe(model, subject, &imgs, &mut facts).await {
            progress(&format!("  {name}: no description ({e})"));
            facts.label = sanitize_label("", &facts.kind, subject);
        }
    }
    Ok(facts)
}

/// The media items of a project document (`project.inspect`): `(item, name, seconds)` for
/// everything with video.
pub fn video_items(project: &Value) -> Vec<(u64, String, f64)> {
    let mut out = Vec::new();
    fn walk(v: &Value, out: &mut Vec<(u64, String, f64)>) {
        match v {
            Value::Object(o) => {
                if o.get("type").and_then(Value::as_str) == Some("Movie")
                    && let (Some(item), Some(name), Some(d)) =
                        (o.get("item").and_then(Value::as_u64), o.get("name").and_then(Value::as_str), o.get("duration").and_then(Value::as_f64))
                {
                    out.push((item, name.to_string(), d / filmcraft_time::TICKS_PER_SECOND as f64));
                }
                for x in o.values() {
                    walk(x, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    walk(project, &mut out);
    out.sort_by_key(|(i, _, _)| *i);
    out.dedup_by_key(|(i, _, _)| *i);
    out
}

/// Analyse `items` (or every video item) for `subject`, oldest first. `progress` gets a line per
/// clip. A clip that cannot be read is skipped with a note, never fatal.
pub async fn analyze(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    items: Option<&[u64]>,
    frames_max: usize,
    screen: usize,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ClipFacts>, AutoCutError> {
    let subject = subject.trim();
    if subject.is_empty() {
        return Err(AutoCutError::Invalid("say what to look for (e.g. \"robot\")".into()));
    }
    let project = ex.execute("project.inspect", json!({})).await.map_err(AutoCutError::Executor)?;
    let mut clips = video_items(&project);
    if let Some(want) = items {
        clips.retain(|(i, _, _)| want.contains(i));
    }
    let dates = ex.execute("media.captureDate", json!({"items": clips.iter().map(|c| c.0).collect::<Vec<_>>()})).await.unwrap_or(Value::Null);
    let date_of = |item: u64| -> (Option<String>, Option<i64>) {
        dates["items"]
            .as_array()
            .and_then(|a| a.iter().find(|d| d["item"].as_u64() == Some(item)))
            .map(|d| (d["created"].as_str().map(str::to_string), d["unix"].as_i64()))
            .unwrap_or((None, None))
    };
    let n = clips.len();
    let mut out = Vec::with_capacity(n);
    for (k, (item, name, secs)) in clips.into_iter().enumerate() {
        let mut f = analyze_clip(model, ex, subject, item, &name, secs, frames_max.clamp(1, MAX_FRAMES), screen, progress).await?;
        (f.created, f.unix) = date_of(item);
        progress(&format!(
            "[{}/{n}] {name}: {}",
            k + 1,
            if f.present { format!("{} ({:.0}% of frames){}", f.label, f.coverage * 100.0, read_note(&f.text)) } else { format!("no {subject}") }
        ));
        out.push(f);
    }
    out.sort_by_key(|f| (f.unix.unwrap_or(i64::MAX), f.item));
    Ok(out)
}

/// A stand-in for the engine in tests: a project of clips whose frames are small PNGs.
#[cfg(test)]
mod screening_tests {
    use super::*;

    #[test]
    fn screening_looks_at_spread_frames_first() {
        let times: Vec<f64> = (0..6).map(|k| k as f64).collect();
        let (order, first) = screening_order(&times, 3);
        assert_eq!(first, 3);
        assert_eq!(&order[..3], &[1.0, 3.0, 5.0], "spread across the clip");
        let mut all = order.clone();
        all.sort_by(f64::total_cmp);
        assert_eq!(all, times, "every time, once");
        // off, or more than there are: the plain order
        assert_eq!(screening_order(&times, 0), (times.clone(), 6));
        assert_eq!(screening_order(&times, 9), (times.clone(), 6));
        assert_eq!(screening_order(&[], 3), (vec![], 0));
        assert_eq!(screening_order(&[2.0], 3), (vec![2.0], 1));
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Mutex;

    use serde_json::{Value, json};

    use crate::{ExecFuture, Executor};

    pub(crate) struct FakeEx {
        /// `(item, seconds, unix)` per clip.
        pub clips: Vec<(u64, f64, i64)>,
        /// Every `media.frame` asked for: `(item, seconds, maxSide)`.
        pub frames: Mutex<Vec<(u64, f64, u64)>>,
    }

    impl FakeEx {
        pub fn new(clips: Vec<(u64, f64, i64)>) -> Self {
            Self { clips, frames: Mutex::new(Vec::new()) }
        }
    }

    impl Executor for FakeEx {
        fn execute(&self, id: &str, p: Value) -> ExecFuture {
            let tps = filmcraft_time::TICKS_PER_SECOND as f64;
            let r = match id {
                "project.inspect" => Ok(
                    json!({"root": {"children": self.clips.iter().map(|(i, s, _)| json!({"item": i, "name": format!("c{i}.mp4"), "type": "Movie", "duration": s * tps})).collect::<Vec<_>>()}}),
                ),
                "media.captureDate" => Ok(json!({"items": self.clips.iter().map(|(i, _, u)| json!({"item": i, "unix": u})).collect::<Vec<_>>()})),
                "media.frame" => {
                    let item = p["item"].as_u64().unwrap_or(0);
                    self.frames.lock().unwrap().push((item, p["seconds"].as_f64().unwrap_or(-1.0), p["maxSide"].as_u64().unwrap_or(0)));
                    // a frame a shade per item, 2:1
                    let shade = (item * 40 % 256) as u8;
                    let png = crate::png(8, 4, vec![shade; 8 * 4 * 4], 0).unwrap();
                    Ok(json!({"pngBase64": crate::b64(&png), "width": 8, "height": 4}))
                }
                other => Err(format!("no such command: {other}")),
            };
            Box::pin(async move { r })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FixedModel;

    #[test]
    fn samples_skip_the_edges_and_are_capped() {
        let t = sample_times(30.0, 8);
        assert_eq!(t.len(), 8 + 2 - 2); // 29 s span / 3 s → 10, capped at 8
        assert!(t.iter().all(|x| *x > 0.4 && *x < 29.6), "{t:?}");
        assert_eq!(sample_times(2.0, 8).len(), 1);
        assert!(sample_times(0.0, 8).is_empty());
        assert!(sample_times(f64::NAN, 8).is_empty());
        assert!(sample_times(10.0, 0).is_empty());
    }

    #[test]
    fn frame_facts_from_boxes() {
        let (fw, fh) = (1000.0, 500.0);
        let f = frame_fact(2.0, &json!({"present": true, "bbox_2d": [100, 100, 600, 400]}), fw, fh);
        assert!(f.present);
        assert!((f.area - 0.5 * 0.6).abs() < 1e-9);
        assert_eq!(f.center, [0.35, 0.5]);
        // absent, tiny, hostile, reversed
        assert!(!frame_fact(0.0, &json!({"present": false, "bbox_2d": [0, 0, 900, 400]}), fw, fh).present);
        assert!(!frame_fact(0.0, &json!({"present": true, "bbox_2d": [0, 0, 10, 10]}), fw, fh).present);
        assert!(!frame_fact(0.0, &json!({"present": true, "bbox_2d": "x"}), fw, fh).present);
        assert!(!frame_fact(0.0, &json!({"present": true, "bbox_2d": [500, 0, 100, 10]}), fw, fh).present);
        assert!(!frame_fact(0.0, &json!({"present": true}), fw, fh).present);
        assert!(!frame_fact(0.0, &json!({"present": true, "bbox_2d": [0, 0, 100, 100]}), 0.0, fh).present);
    }

    #[test]
    fn model_frames_match_the_screen_agent() {
        assert_eq!(model_frame(1344, 784), (1344.0, 784.0));
        let (w, h) = model_frame(640, 360);
        assert!(w % 28.0 == 0.0 && h % 28.0 == 0.0 && (w / 28.0) * (h / 28.0) >= 1024.0);
    }

    #[test]
    fn invented_names_become_plain_labels() {
        assert_eq!(sanitize_label("Pepper humanoid robot", "humanoid", "robot"), "Pepper humanoid robot");
        assert_eq!(sanitize_label("Boston Dynamics Spot", "quadruped", "robot"), "Boston Dynamics Spot");
        assert_eq!(sanitize_label("Kangaroo Floor Cleaning Robot", "wheeled", "robot"), "wheeled robot");
        assert_eq!(sanitize_label("Ardet", "quadruped", "robot"), "quadruped robot");
        assert_eq!(sanitize_label("floor-cleaning robot", "wheeled", "robot"), "floor-cleaning robot");
        assert_eq!(sanitize_label("wheeled", "wheeled", "robot"), "wheeled robot");
        assert_eq!(sanitize_label("", "other", "robot"), "robot");
        assert_eq!(sanitize_label("red toy robot", "toy", "robot"), "red toy robot");
    }

    fn read(on_subject: &[&str], nearby: &[&str]) -> FrameText {
        FrameText { on_subject: on_subject.iter().map(|s| s.to_string()).collect(), nearby: nearby.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn names_printed_on_the_robot_survive() {
        let s = |label: &str, kind: &str, t: &FrameText| sanitize_label_with(label, kind, "robot", t);
        // a red wheeled-legged robot: "ARDET" and "Swiss-Mile" on it, "ETH zürich RSL" on the booth
        let ardet = read(&["ARDET", "Swiss-Mile"], &["ETH zürich RSL"]);
        assert_eq!(s("Swiss-Mile ARDET robot", "wheeled", &ardet), "Swiss-Mile ARDET robot");
        assert_eq!(s("Ardet", "wheeled", &ardet), "Ardet", "any case");
        assert_eq!(s("Swiss Mile robot", "wheeled", &ardet), "Swiss Mile robot", "any spacing");
        assert_eq!(s("SwissMile ARDET", "wheeled", &ardet), "SwissMile ARDET", "or none");
        assert_eq!(s("ETH Zurich robot", "wheeled", &ardet), "wheeled robot", "the booth is not its name");
        assert_eq!(s("Ardet", "wheeled", &FrameText::default()), "wheeled robot", "not read: invented");
        // "temi" on a personal robot; "Go1" on a robot dog's side, however it is spelled
        assert_eq!(s("Temi personal robot", "wheeled", &read(&["temi"], &[])), "Temi personal robot");
        assert_eq!(s("GO-1 robot dog", "quadruped", &read(&["Go1"], &[])), "GO-1 robot dog");
        assert_eq!(s("Go 1 robot dog", "quadruped", &read(&["GO 1"], &[])), "Go 1 robot dog");
        // "EMRILL" on cleaning robots: read, so kept (and not without the text)
        assert_eq!(s("Emrill cleaning robot", "wheeled", &read(&["EMRILL"], &[])), "Emrill cleaning robot");
        assert_eq!(s("Emrill cleaning robot", "wheeled", &FrameText::default()), "wheeled robot");
        // "OnRobot" on grippers: the name read; a misreading of it is not
        let grip = read(&["OnRobot"], &[]);
        assert_eq!(s("OnRobot gripper", "arm", &grip), "OnRobot gripper");
        assert_eq!(s("Qrobot gripper", "arm", &grip), "arm robot");
        // a model number printed on an arm, and the same invented without it
        assert_eq!(s("FANUC LR Mate 200iD", "arm", &read(&["FANUC", "LR Mate 200iD"], &[])), "FANUC LR Mate 200iD");
        assert_eq!(s("LR Mate 200iD", "arm", &read(&[], &["DP WORLD"])), "arm robot");
        assert_eq!(s("Terminus", "humanoid", &ardet), "humanoid robot");
        // known names stay allowed without text
        assert_eq!(s("Unitree Go1", "quadruped", &FrameText::default()), "Unitree Go1");
    }

    #[test]
    fn signs_name_a_robot_only_when_they_introduce_one() {
        let s = |label: &str, kind: &str, t: &FrameText| sanitize_label_with(label, kind, "robot", t);
        // "Ameca AI Interaction Guide" beside Ameca (known anyway), and a name the list lacks
        assert_eq!(s("Ameca humanoid robot", "humanoid", &read(&[], &["Ameca AI Interaction Guide"])), "Ameca humanoid robot");
        assert_eq!(s("Nadine humanoid", "humanoid", &read(&[], &["Meet NADINE"])), "Nadine humanoid");
        assert_eq!(s("Nadine humanoid", "humanoid", &read(&[], &["NADINE"])), "humanoid robot", "a bare name on a sign");
        // "QUOKKA" on a kiosk's glass is a shop, not the robot
        assert_eq!(s("Quokka robot", "wheeled", &read(&[], &["QUOKKA"])), "wheeled robot");
        // "DP WORLD" on a Promobot's tablet is a sponsor: the description may say so, the label not
        let promo = read(&[], &["DP WORLD"]);
        assert_eq!(s("DP World service robot", "wheeled", &promo), "wheeled robot");
        assert_eq!(s("Promobot service robot", "wheeled", &promo), "Promobot service robot");
        // whether a label holds a name that was read
        assert!(read(&["ARDET"], &[]).names("Swiss ARDET robot"));
        assert!(read(&[], &["Ameca AI Interaction Guide"]).names("Ameca"));
        assert!(!read(&[], &["QUOKKA"]).names("Quokka robot"));
        assert!(!read(&["ARDET"], &[]).names("wheeled robot"));
    }

    #[test]
    fn text_replies_are_cleaned_and_capped() {
        let t = FrameText::from_reply(&json!({
            "on_subject": ["ARDET", " Swiss-Mile\n", "ardet", "none", "", "...", 42, null, {"text": "Go1"}, "\"temi\"\u{7}"],
            "nearby": "ETH zürich RSL"
        }));
        assert_eq!(t.on_subject, vec!["ARDET", "Swiss-Mile", "Go1", "temi"]);
        assert_eq!(t.nearby, vec!["ETH zürich RSL"], "a lone string is one line");
        // not the shape asked for: no text, never an error
        for v in [json!("ARDET"), json!(null), json!([1, 2]), json!({"on_subject": {"a": 1}, "nearby": 7}), json!({})] {
            assert!(FrameText::from_reply(&v).is_empty(), "{v}");
        }
        // huge lists and absurd lines are capped
        let huge = FrameText::from_reply(&json!({
            "on_subject": (0..10_000).map(|i| format!("NAME{i}")).collect::<Vec<_>>(),
            "nearby": ["x".repeat(1_000_000), format!("{}A", " ".repeat(500_000))]
        }));
        assert_eq!(huge.on_subject.len(), MAX_TEXT_LINES);
        assert_eq!(huge.nearby[0].chars().count(), MAX_TEXT_CHARS);
        assert_eq!(huge.nearby.len(), 1, "a line of blanks is no text");
        // merging frames keeps the first spelling and the cap
        let mut m = read(&["ARDET"], &["QUOKKA"]);
        m.merge(read(&["Ardet", "Swiss-Mile"], &["quokka", "ETH zürich RSL"]));
        assert_eq!(m, read(&["ARDET", "Swiss-Mile"], &["QUOKKA", "ETH zürich RSL"]));
        m.merge(huge);
        assert_eq!(m.on_subject.len(), MAX_TEXT_LINES);
        // an absurd label is cut before it is checked
        assert_eq!(sanitize_label_with(&"ARDET ".repeat(100_000), "wheeled", "robot", &read(&["ARDET"], &[])).chars().count(), MAX_LABEL_CHARS - 1);
    }

    #[test]
    fn old_reports_load_without_text() {
        let mut v = serde_json::to_value(ClipFacts { item: 3, label: "Pepper humanoid robot".into(), ..ClipFacts::default() }).unwrap();
        v.as_object_mut().unwrap().remove("text");
        let f: ClipFacts = serde_json::from_value(v).unwrap();
        assert!(f.text.is_empty());
        assert_eq!(f.label, "Pepper humanoid robot");
    }

    /// Grounding replies: the robot in both frames, largest in the first.
    fn grounded() -> Vec<Value> {
        vec![json!({"present": true, "bbox_2d": [0, 0, 700, 500]}), json!({"present": true, "bbox_2d": [0, 0, 300, 300]})]
    }

    #[tokio::test]
    async fn analysis_reads_the_text_before_describing() {
        let ex = testing::FakeEx::new(vec![(1, 10.0, 1_700_000_000)]);
        let mut answers = grounded();
        answers.extend([
            json!({"on_subject": ["ARDET"], "nearby": ["ETH zürich RSL", "QUOKKA"]}),
            json!({"on_subject": ["Swiss-Mile", "ardet"], "nearby": []}),
            json!({"label": "Swiss-Mile ARDET", "kind": "wheeled", "description": "A red wheeled-legged robot at the ETH Zürich booth."}),
        ]);
        let model = FixedModel::new(answers);
        let facts = analyze(&model, &ex, "robot", None, 2, 0, &mut |_| {}).await.unwrap();
        let f = &facts[0];
        assert!(f.present);
        assert_eq!(f.text, read(&["ARDET", "Swiss-Mile"], &["ETH zürich RSL", "QUOKKA"]));
        assert_eq!(f.label, "Swiss-Mile ARDET", "the name printed on it is kept");
        let asked = model.asked.lock().unwrap();
        assert_eq!(asked.len(), 5, "two frames located, two read, one description");
        assert!(asked[2].1.contains("on_subject") && asked[3].1.contains("on_subject"));
        let describe = &asked[4].1;
        assert!(describe.contains("\"ARDET\", \"Swiss-Mile\"") && describe.contains("\"ETH zürich RSL\""), "{describe}");
        assert!(describe.contains("not its name"), "{describe}");
        // the text is read in larger frames than the robot is located in
        let frames = ex.frames.lock().unwrap();
        assert_eq!(frames.iter().filter(|f| f.2 == u64::from(TEXT_SIDE)).count(), TEXT_FRAMES, "{frames:?}");
    }

    #[tokio::test]
    async fn screening_skips_a_clip_whose_spread_frames_show_nothing() {
        // 30 s: six sample times; screening with three
        let ex = testing::FakeEx::new(vec![(1, 30.0, 1_700_000_000), (2, 30.0, 1_700_000_100)]);
        let nothing = json!({"present": false});
        let mut answers = vec![nothing.clone(), nothing.clone(), nothing.clone()];
        // the second clip: the robot in the second screened frame, then the other three looked at
        answers.extend([nothing.clone(), json!({"present": true, "bbox_2d": [0, 0, 400, 400]}), nothing.clone(), nothing.clone(), nothing.clone(), nothing]);
        answers.extend([json!({"on_subject": [], "nearby": []}), json!({"label": "white robot", "kind": "humanoid", "description": "A white robot."})]);
        let model = FixedModel::new(answers);
        let facts = analyze(&model, &ex, "robot", None, 6, 3, &mut |_| {}).await.unwrap();
        assert!(!facts[0].present && facts[0].frames.len() == 3, "three looks, then skipped: {:?}", facts[0].frames);
        assert!(facts[1].present);
        assert_eq!(facts[1].frames.len(), 6, "a clip that shows it is looked at fully");
        assert!(facts[1].frames.windows(2).all(|w| w[0].seconds < w[1].seconds), "frames in time order");
        assert!((facts[1].coverage - 1.0 / 6.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn without_text_a_guessed_name_is_still_dropped() {
        let ex = testing::FakeEx::new(vec![(1, 10.0, 1_700_000_000)]);
        let mut answers = grounded();
        answers.extend([
            json!("I cannot read anything."),
            json!({"on_subject": [], "nearby": ["QUOKKA"]}),
            json!({"label": "Quokka robot", "kind": "wheeled", "description": "A white robot beside a kiosk."}),
        ]);
        let model = FixedModel::new(answers);
        let facts = analyze(&model, &ex, "robot", None, 2, 0, &mut |_| {}).await.unwrap();
        assert_eq!(facts[0].label, "wheeled robot");
        assert_eq!(facts[0].text, read(&[], &["QUOKKA"]));
    }

    #[tokio::test]
    async fn relabel_reads_text_into_old_reports() {
        let ex = testing::FakeEx::new(vec![(4, 10.0, 1_700_000_000)]);
        // an old report: no text, a sponsor's name on the screen taken for the robot's
        let mut v = serde_json::to_value(ClipFacts {
            item: 4,
            name: "c4.mp4".into(),
            seconds: 10.0,
            present: true,
            coverage: 1.0,
            label: "red toy robot".into(),
            kind: "toy".into(),
            frames: vec![
                FrameFact { seconds: 2.0, present: true, area: 0.2, center: [0.5, 0.5] },
                FrameFact { seconds: 5.0, present: true, area: 0.4, center: [0.5, 0.5] },
                FrameFact { seconds: 8.0, present: false, area: 0.0, center: [0.5, 0.5] },
            ],
            ..ClipFacts::default()
        })
        .unwrap();
        v.as_object_mut().unwrap().remove("text");
        let mut facts: Vec<ClipFacts> = vec![serde_json::from_value(v).unwrap()];
        let model = FixedModel::new(vec![
            json!({"on_subject": ["Promobot"], "nearby": ["DP WORLD"]}),
            json!({"on_subject": [], "nearby": ["DP WORLD"]}),
            json!({"label": "Promobot reception robot", "kind": "wheeled", "description": "A Promobot reception robot whose tablet shows the DP World logo."}),
        ]);
        let mut lines = Vec::new();
        relabel(&model, &ex, "robot", &mut facts, &mut |l| lines.push(l.to_string())).await.unwrap();
        assert_eq!(facts[0].text, read(&["Promobot"], &["DP WORLD"]));
        assert_eq!(facts[0].label, "Promobot reception robot");
        assert!(facts[0].description.contains("DP World"), "a sponsor may be described");
        assert!(lines.iter().any(|l| l.contains("red toy robot → Promobot reception robot") && l.contains("\"Promobot\"")), "{lines:?}");
        // a model that answers nothing keeps the earlier label and text
        let none = FixedModel::new(vec![]);
        relabel(&none, &ex, "robot", &mut facts, &mut |_| {}).await.unwrap();
        assert_eq!((facts[0].label.as_str(), facts[0].text.on_subject.len()), ("Promobot reception robot", 1));
    }

    #[tokio::test]
    async fn run_tells_which_clips_show_the_same_robot() {
        let ex = testing::FakeEx::new(vec![]);
        // an earlier report with nothing to describe again from: then the identity pass
        let day = 1_665_446_400;
        let earlier: Vec<ClipFacts> = (1..=2)
            .map(|i| ClipFacts {
                item: i,
                name: format!("c{i}.mp4"),
                seconds: 10.0,
                unix: Some(day + i as i64 * 86_400),
                present: true,
                coverage: 1.0,
                label: "wheeled robot".into(),
                kind: "wheeled".into(),
                best: 5.0,
                ..ClipFacts::default()
            })
            .collect();
        let model = FixedModel::new(vec![json!({"same": true, "confidence": 0.95})]);
        let facts = run(&model, &ex, "robot", None, 2, 0, Some(earlier.clone()), 10, &mut |_| {}).await.unwrap();
        assert_eq!(facts.iter().map(|f| f.identity).collect::<Vec<_>>(), vec![Some(1), Some(1)]);
        // no comparisons: each its own robot
        let facts = run(&FixedModel::new(vec![]), &ex, "robot", None, 2, 0, Some(earlier), 0, &mut |_| {}).await.unwrap();
        assert_eq!(facts.iter().map(|f| f.identity).collect::<Vec<_>>(), vec![Some(1), Some(2)]);
    }

    #[test]
    fn video_items_walk_the_project_tree() {
        let t = filmcraft_time::TICKS_PER_SECOND as f64;
        let p = json!({"root": {"children": [
            {"item": 3, "name": "b.mp4", "type": "Movie", "duration": 2.0 * t},
            {"children": [{"item": 1, "name": "a.mov", "type": "Movie", "duration": 5.0 * t}, {"item": 9, "name": "music.wav", "type": "Audio", "duration": t}]}
        ]}});
        assert_eq!(video_items(&p), vec![(1, "a.mov".into(), 5.0), (3, "b.mp4".into(), 2.0)]);
    }
}

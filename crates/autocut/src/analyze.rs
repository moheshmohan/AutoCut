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
//! A photo (a still image item) is one frame: located, read and described from that picture
//! ([`ClipFacts::photo`]), its subject's box kept ([`ClipFacts::subject`]) for the slow pan and
//! zoom a showcase gives it, and cached like a clip.
//!
//! Capture dates come from `media.captureDate` (sidecars from the Google Photos picker, or the
//! file's own metadata: QuickTime keys, a photo's EXIF, with its position when it has one), so
//! results sort chronologically.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::analysis_cache::{self, AnalysisCache};
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
    /// A photo (a still image), not a video: looked at once, shown with a slow pan and zoom.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub photo: bool,
    /// The subject's box where it is largest, `[x, y, w, h]` (0..1 of the picture): what a photo's
    /// pan and zoom aims at. Absent in reports from before it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<[f64; 4]>,
    /// Where a photo was taken, from its EXIF: `[latitude, longitude]` (Google Photos' downloads
    /// leave it out; a photo copied off the phone keeps it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gps: Option<[f64; 2]>,
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
    let Some([x, y, w, h]) = reply_box(reply, fw, fh) else { return none };
    let area = w * h;
    FrameFact { seconds, present: area >= MIN_AREA, area, center: [x + w / 2.0, y + h / 2.0] }
}

/// The box of a grounding reply ([`frame_fact`]) as `[x, y, w, h]`, 0..1 of the frame; `None`
/// when it says not present or its numbers make no sense.
pub fn reply_box(reply: &Value, fw: f64, fh: f64) -> Option<[f64; 4]> {
    if reply.get("present").and_then(Value::as_bool) == Some(false) {
        return None;
    }
    let b = reply.get("bbox_2d").or_else(|| reply.get("box_2d")).and_then(Value::as_array)?;
    let c: Vec<f64> = b.iter().filter_map(Value::as_f64).collect();
    let [x1, y1, x2, y2] = c.as_slice() else { return None };
    if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) || fw <= 0.0 || fh <= 0.0 {
        return None;
    }
    let (x1, x2) = ((x1 / fw).clamp(0.0, 1.0), (x2 / fw).clamp(0.0, 1.0));
    let (y1, y2) = ((y1 / fh).clamp(0.0, 1.0), (y2 / fh).clamp(0.0, 1.0));
    (x2 > x1 && y2 > y1).then_some([x1, y1, x2 - x1, y2 - y1])
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

fn describe_prompt(subject: &str, read: &FrameText, photo: bool) -> String {
    let guide = if subject.to_lowercase().contains("robot") { ROBOT_GUIDE } else { "" };
    let text = text_note(subject, read);
    let what = if photo { format!("This photo shows a {subject}.") } else { format!("These frames come from one video showing a {subject}.") };
    format!(
        "{what}\n{guide}{text}Answer with JSON only: {{\"label\": \"what it is in 2-5 words: its name or model when that is printed on it, or a brand or model name ONLY if it clearly matches the guide or a logo is readable; otherwise a plain description such as 'floor-cleaning robot' or 'white robot dog'\", \"kind\": \"humanoid|quadruped|arm|wheeled|toy|drone|other\", \"description\": \"one plain sentence: what it looks like and what it is doing (and the text around it, when that says where it is)\"}}"
    )
}

/// Describe a clip from `imgs` (PNG, in time order) and the text read in it ([`ClipFacts::text`]).
async fn describe(model: &dyn ChatModel, subject: &str, imgs: &[Vec<u8>], f: &mut ClipFacts) -> Result<(), AutoCutError> {
    let system = if f.photo { "You describe photos for a video editor." } else { "You describe videos for a video editor." };
    let v = model.complete_json(system, &describe_prompt(subject, &f.text, f.photo), imgs).await?;
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

/// Analyse afresh — taking every clip seen before from `cache` ([`AnalysisCache`]) — or (with
/// `earlier`) only read and describe again from an earlier analysis; then tell which clips show
/// the same robot ([`crate::identity::identify`], at most `compare` model comparisons; 0 joins by
/// name only).
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
    cache: &AnalysisCache,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ClipFacts>, AutoCutError> {
    let mut facts = match earlier {
        Some(mut facts) => {
            relabel(model, ex, subject, &mut facts, progress).await?;
            facts
        }
        None => analyze(model, ex, subject, items, frames_max, screen, cache, progress).await?,
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

/// What a clip's scene index (`media.scenes`, made by `media.analyzeScenes`) says about the
/// subject: a moment at each shot's keyframes and edges, seen where the shot's description names
/// the subject (in its subjects, caption or actions); the best moment — the keyframe of the
/// best-scoring shot that shows it (sharp, exposed, steady, the aesthetic score), larger subject
/// first; the subject's box there; the caption and setting; the text read in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneFacts {
    pub frames: Vec<FrameFact>,
    pub best: f64,
    pub subject: Option<[f64; 4]>,
    pub label: String,
    pub description: String,
    pub text: Vec<String>,
}

/// Whether `words` name `subject` (a word of it, or its plural, starting a word: "robot" in
/// "robotic arm", "robots", "Robot").
fn names_subject(words: &str, subject: &str) -> bool {
    let subject: Vec<String> = subject.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 3).map(str::to_lowercase).collect();
    let words: Vec<String> = words.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
    !subject.is_empty() && subject.iter().any(|s| words.iter().any(|w| w.starts_with(s.as_str()) || s.starts_with(w.as_str()) && w.len() >= 4))
}

/// Read a `media.scenes` answer for `subject`; `None` when the item has no described index or no
/// shot of it names the subject (the vision model is asked then, as without an index).
pub fn scene_facts(index: &Value, subject: &str) -> Option<SceneFacts> {
    if index.get("analyzed") != Some(&Value::Bool(true)) || index["describer"].as_str().is_none_or(str::is_empty) {
        return None;
    }
    let shots = index["shots"].as_array()?;
    let num = |v: &Value| v.as_f64().filter(|x| x.is_finite());
    let mut frames = Vec::new();
    // (score, area, keyframe, box, shot)
    let mut best: Option<(f64, f64, f64, Option<[f64; 4]>, &Value)> = None;
    for sh in shots.iter().take(10_000) {
        let (Some(a), Some(b)) = (num(&sh["start"]), num(&sh["end"])) else { continue };
        if b <= a {
            continue;
        }
        let d = &sh["description"];
        let subjects: Vec<&Value> = d["subjects"].as_array().map(|v| v.iter().take(16).collect()).unwrap_or_default();
        let named: Vec<&Value> = subjects.iter().copied().filter(|x| x["name"].as_str().is_some_and(|n| names_subject(n, subject))).collect();
        let actions = d["actions"].as_array().map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")).unwrap_or_default();
        let seen = !named.is_empty() || names_subject(&format!("{} {actions}", d["caption"].as_str().unwrap_or_default()), subject);
        // the subject's box, `[x, y, w, h]` of the picture, as the describer gave it
        let bbox = named.iter().find_map(|x| {
            let v: Vec<f64> = x["box"].as_array()?.iter().filter_map(&num).collect();
            let [x, y, w, h] = v.as_slice() else { return None };
            ([*x, *y, *w, *h].iter().all(|c| (0.0..=1.0).contains(c)) && *w > 0.0 && *h > 0.0).then_some([*x, *y, *w, *h])
        });
        let area = bbox.map_or(if seen { 0.05 } else { 0.0 }, |b| b[2] * b[3]);
        let center = bbox.map_or([0.5, 0.5], |b| [b[0] + b[2] / 2.0, b[1] + b[3] / 2.0]);
        let key = sh["keyframes"].as_array().and_then(|k| k.first()).and_then(num).unwrap_or((a + b) / 2.0);
        // the shot's edges (inside it) and its keyframes: the showcase cuts its windows there
        let margin = ((b - a) * 0.1).min(0.2);
        let mut times = vec![a + margin, key, b - margin];
        times.extend(sh["keyframes"].as_array().map(|k| k.iter().filter_map(num).collect::<Vec<_>>()).unwrap_or_default());
        times.sort_by(f64::total_cmp);
        times.dedup_by(|x, y| (*x - *y).abs() < 0.05);
        for t in times {
            frames.push(FrameFact { seconds: t, present: seen && area >= MIN_AREA, area: if seen { area } else { 0.0 }, center });
        }
        if seen && area >= MIN_AREA {
            let score = num(&sh["quality"]["score"]).unwrap_or(0.5).clamp(0.0, 1.0);
            // a sharp, steady shot first, then the larger subject
            let rank = score + 0.5 * bbox.map_or(0.0, |b| (b[2] * b[3]).min(0.5));
            if best.is_none_or(|(r, ..)| rank > r) {
                best = Some((rank, area, key, bbox, sh));
            }
        }
    }
    let (_, _, key, bbox, sh) = best?;
    let d = &sh["description"];
    let label = d["subjects"]
        .as_array()
        .and_then(|v| v.iter().filter_map(|x| x["name"].as_str()).find(|n| names_subject(n, subject)))
        .unwrap_or(subject)
        .chars()
        .take(MAX_LABEL_CHARS)
        .collect();
    let caption = d["caption"].as_str().unwrap_or_default();
    let setting = d["setting"].as_str().unwrap_or_default();
    let description =
        if setting.is_empty() || caption.to_lowercase().contains(&setting.to_lowercase()) { caption.to_string() } else { format!("{caption} ({setting})") };
    let text = d["text"].as_array().map(|v| v.iter().filter_map(Value::as_str).filter_map(clean_line).collect()).unwrap_or_default();
    frames.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
    Some(SceneFacts { frames, best: key, subject: bbox, label, description: description.chars().take(300).collect(), text })
}

/// The scene index's facts for `item`, when there is one that names the subject.
async fn scene_facts_of(ex: &dyn Executor, item: u64, subject: &str) -> Option<SceneFacts> {
    let v = ex.execute("media.scenes", json!({"item": item})).await.ok()?;
    scene_facts(&v, subject)
}

/// One clip: frames → grounding → text → description. The flag says every frame was read and
/// every model call answered — only such an analysis is kept in the cache (one made while the
/// model was down would say "not there" forever). A photo is one frame: located, read and
/// described from that one picture.
#[allow(clippy::too_many_arguments)]
async fn analyze_clip(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    item: u64,
    name: &str,
    seconds: f64,
    photo: bool,
    frames_max: usize,
    screen: usize,
    progress: &mut dyn FnMut(&str),
) -> Result<(ClipFacts, bool), AutoCutError> {
    // the scene index, when the clip has one that names the subject: its shots say where the
    // subject is (no frame-by-frame grounding), the best shot where to look; the vision model
    // still reads the text on it and names it
    if !photo && let Some(sf) = scene_facts_of(ex, item, subject).await {
        return Ok(from_scenes(model, ex, subject, item, name, seconds, sf, progress).await);
    }
    let mut clean = true;
    let mut frames: Vec<FrameFact> = Vec::new();
    let mut pngs: Vec<(f64, f64, Vec<u8>)> = Vec::new(); // (area, seconds, png)
    // the subject's box where it is largest
    let mut largest: Option<(f64, [f64; 4])> = None;
    let times = if photo { vec![0.0] } else { sample_times(seconds, frames_max) };
    let (order, screened) = screening_order(&times, if photo { 0 } else { screen });
    for (k, t) in order.into_iter().enumerate() {
        // screening: the spread-out first frames show nothing, so the rest are not looked at
        if k == screened && !frames.iter().any(|f| f.present) {
            break;
        }
        let r = match ex.execute("media.frame", json!({"item": item, "seconds": t, "maxSide": 640})).await {
            Ok(r) => r,
            Err(e) => {
                progress(&format!("  frame {t:.1}s of {name}: {e}"));
                clean = false;
                continue;
            }
        };
        let Some(png) = r["pngBase64"].as_str().and_then(crate::b64_decode) else {
            clean = false;
            continue;
        };
        let (w, h) = (r["width"].as_u64().unwrap_or(1) as u32, r["height"].as_u64().unwrap_or(1) as u32);
        let (fw, fh) = model_frame(w, h);
        let reply = model.complete_json(GROUND_SYSTEM, &ground_prompt(subject), std::slice::from_ref(&png)).await;
        let f = match reply {
            Ok(v) => {
                let f = frame_fact(t, &v, fw, fh);
                if f.present
                    && largest.is_none_or(|(a, _)| f.area > a)
                    && let Some(b) = reply_box(&v, fw, fh)
                {
                    largest = Some((f.area, b));
                }
                f
            }
            Err(e) => {
                progress(&format!("  {name} @ {t:.1}s: the model did not answer ({e})"));
                clean = false;
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
        photo,
        subject: largest.map(|(_, b)| b),
        gps: None,
    };
    if facts.present {
        // the text it carries and that is around it, then a description that can use it
        match read_clip_text(model, ex, subject, &facts, progress).await {
            Some(read) => facts.text = read,
            None => clean = false,
        }
        // describe from the (up to) three frames where it is largest, in time order
        pngs.sort_by(|a, b| b.0.total_cmp(&a.0));
        pngs.truncate(3);
        pngs.sort_by(|a, b| a.1.total_cmp(&b.1));
        let imgs: Vec<Vec<u8>> = pngs.into_iter().map(|(_, _, p)| p).collect();
        if let Err(e) = describe(model, subject, &imgs, &mut facts).await {
            progress(&format!("  {name}: no description ({e})"));
            facts.label = sanitize_label("", &facts.kind, subject);
            clean = false;
        }
    }
    Ok((facts, clean))
}

/// A clip's facts from its scene index ([`SceneFacts`]): presence, moments, best moment and box
/// from the index; then the vision model reads the text in the best frames and names the subject
/// from them (as without an index). When it does not answer, the index's own words stand. The
/// flag (cacheable) is true when the model answered.
#[allow(clippy::too_many_arguments)]
async fn from_scenes(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    item: u64,
    name: &str,
    seconds: f64,
    sf: SceneFacts,
    progress: &mut dyn FnMut(&str),
) -> (ClipFacts, bool) {
    let shown = sf.frames.iter().filter(|f| f.present).count();
    let coverage = if sf.frames.is_empty() { 0.0 } else { shown as f64 / sf.frames.len() as f64 };
    let mut facts = ClipFacts {
        item,
        name: name.to_string(),
        seconds,
        present: true,
        coverage,
        label: sf.label.clone(),
        description: sf.description.clone(),
        text: FrameText { on_subject: Vec::new(), nearby: sf.text.iter().take(MAX_TEXT_LINES).cloned().collect() },
        best: sf.best,
        frames: sf.frames,
        subject: sf.subject,
        ..Default::default()
    };
    progress(&format!("  {name}: the scene index shows the {subject} ({shown} moment(s)); best at {:.1}s", facts.best));
    let mut clean = true;
    match read_clip_text(model, ex, subject, &facts, progress).await {
        Some(read) => facts.text = read,
        None => clean = false,
    }
    let mut imgs = Vec::new();
    for t in best_times(&facts.frames, 3) {
        if let Some(png) = frame_png(ex, item, t, 640).await {
            imgs.push(png);
        }
    }
    let (label, description) = (facts.label.clone(), facts.description.clone());
    if imgs.is_empty() || describe(model, subject, &imgs, &mut facts).await.is_err() {
        // the index's own words: a plain label (no invented names), its caption
        facts.label = sanitize_label_with(&label, &facts.kind, subject, &facts.text);
        facts.description = description;
        clean = false;
    }
    (facts, clean)
}

/// The cache identity of each file-backed item (`media.status`): its Google Photos id or content
/// identity, read from the file ([`analysis_cache::media_identity`]), else the identity the engine
/// recorded when it imported the file. Empty when the target cannot say (an older app).
async fn identities(ex: &dyn Executor) -> std::collections::HashMap<u64, String> {
    let Ok(status) = ex.execute("media.status", json!({})).await else { return Default::default() };
    status
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let item = m["item"].as_u64()?;
            let from_file = m["path"].as_str().and_then(|p| analysis_cache::media_identity(std::path::Path::new(p)));
            Some((item, from_file.or_else(|| analysis_cache::reported_identity(&m["identity"]))?))
        })
        .collect()
}

/// The media items of a project document (`project.inspect`): `(item, name, seconds)` for
/// everything with video.
pub fn video_items(project: &Value) -> Vec<(u64, String, f64)> {
    media_items(project).into_iter().filter(|m| !m.3).map(|(i, n, s, _)| (i, n, s)).collect()
}

/// The project's type of a still image (`project.inspect`'s `type`).
pub const STILL_TYPE: &str = "Still Image";

/// The video and photo items of a project document (`project.inspect`): `(item, name, seconds,
/// photo)`, in item order. Bins nest; the walk is bounded.
pub fn media_items(project: &Value) -> Vec<(u64, String, f64, bool)> {
    let mut out = Vec::new();
    let mut stack: Vec<(&Value, usize)> = vec![(project, 0)];
    while let Some((v, depth)) = stack.pop() {
        if depth > 256 || out.len() >= 1_000_000 {
            continue;
        }
        match v {
            Value::Object(o) => {
                let kind = o.get("type").and_then(Value::as_str);
                if matches!(kind, Some("Movie") | Some(STILL_TYPE))
                    && let (Some(item), Some(name), Some(d)) =
                        (o.get("item").and_then(Value::as_u64), o.get("name").and_then(Value::as_str), o.get("duration").and_then(Value::as_f64))
                {
                    out.push((item, name.to_string(), d / filmcraft_time::TICKS_PER_SECOND as f64, kind == Some(STILL_TYPE)));
                }
                stack.extend(o.values().map(|x| (x, depth + 1)));
            }
            Value::Array(a) => stack.extend(a.iter().map(|x| (x, depth + 1))),
            _ => {}
        }
    }
    out.sort_by_key(|m| m.0);
    out.dedup_by_key(|m| m.0);
    out
}

/// Analyse `items` (or every video item) for `subject`, oldest first. `progress` gets a line per
/// clip. A clip that cannot be read is skipped with a note, never fatal. A clip whose file was
/// analysed before — for this subject, with this model and these settings — is taken from
/// `cache` instead ([`AnalysisCache::hits`] counts them); a clip analysed now is kept in it.
#[allow(clippy::too_many_arguments)]
pub async fn analyze(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    items: Option<&[u64]>,
    frames_max: usize,
    screen: usize,
    cache: &AnalysisCache,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ClipFacts>, AutoCutError> {
    let subject = subject.trim();
    if subject.is_empty() {
        return Err(AutoCutError::Invalid("say what to look for (e.g. \"robot\")".into()));
    }
    let project = ex.execute("project.inspect", json!({})).await.map_err(AutoCutError::Executor)?;
    let mut clips = media_items(&project);
    if let Some(want) = items {
        clips.retain(|c| want.contains(&c.0));
    }
    let dates = ex.execute("media.captureDate", json!({"items": clips.iter().map(|c| c.0).collect::<Vec<_>>()})).await.unwrap_or(Value::Null);
    let date_of = |item: u64| -> (Option<String>, Option<i64>, Option<[f64; 2]>) {
        dates["items"]
            .as_array()
            .and_then(|a| a.iter().find(|d| d["item"].as_u64() == Some(item)))
            .map(|d| {
                let gps = d["gps"].as_array().and_then(|g| Some([g.first()?.as_f64()?, g.get(1)?.as_f64()?])).filter(|g| g.iter().all(|v| v.is_finite()));
                (d["created"].as_str().map(str::to_string), d["unix"].as_i64(), gps)
            })
            .unwrap_or((None, None, None))
    };
    let n = clips.len();
    let frames_max = frames_max.clamp(1, MAX_FRAMES);
    let known = if cache.enabled() { identities(ex).await } else { Default::default() };
    let model_id = model.id();
    let mut out = Vec::with_capacity(n);
    let mut cached = 0usize;
    for (k, (item, name, secs, photo)) in clips.into_iter().enumerate() {
        // (a photo is looked at once, whatever the frames asked for)
        let (fm, sc) = if photo { (1, 0) } else { (frames_max, screen) };
        let key = known.get(&item).map(|id| AnalysisCache::key(id, subject, &model_id, fm, sc));
        let (mut f, from_cache) = match key.as_deref().and_then(|k| cache.get(k)) {
            Some(kept) => (ClipFacts { item, name: name.clone(), identity: None, ..kept }, true),
            None => {
                let (f, clean) = analyze_clip(model, ex, subject, item, &name, secs, photo, frames_max, screen, progress).await?;
                if let Some(key) = key.as_deref().filter(|_| clean) {
                    cache.put(key, &f);
                }
                (f, false)
            }
        };
        (f.created, f.unix, f.gps) = date_of(item);
        f.photo = photo;
        cached += usize::from(from_cache);
        progress(&format!(
            "[{}/{n}] {name}: {}{}",
            k + 1,
            match (f.present, f.photo) {
                (true, true) => format!("{} (a photo){}", f.label, read_note(&f.text)),
                (true, false) => format!("{} ({:.0}% of frames){}", f.label, f.coverage * 100.0, read_note(&f.text)),
                (false, _) => format!("no {subject}"),
            },
            if from_cache { " (analysed before)" } else { "" }
        ));
        out.push(f);
    }
    if cached > 0 {
        progress(&format!("{cached} of {n} clip(s) were analysed before and came from the analysis cache"));
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
        /// The file behind an item, for `media.status` (none: it reports no files).
        pub files: Vec<(u64, std::path::PathBuf)>,
        /// Items that are photos (still images), and the EXIF position `media.captureDate` gives
        /// them.
        pub stills: Vec<(u64, Option<[f64; 2]>)>,
    }

    impl FakeEx {
        pub fn new(clips: Vec<(u64, f64, i64)>) -> Self {
            Self { clips, frames: Mutex::new(Vec::new()), files: Vec::new(), stills: Vec::new() }
        }
        fn still(&self, item: u64) -> Option<Option<[f64; 2]>> {
            self.stills.iter().find(|(i, _)| *i == item).map(|(_, g)| *g)
        }
    }

    impl Executor for FakeEx {
        fn execute(&self, id: &str, p: Value) -> ExecFuture {
            let tps = filmcraft_time::TICKS_PER_SECOND as f64;
            let r = match id {
                "project.inspect" => Ok(json!({"root": {"children": self.clips.iter().map(|(i, s, _)| {
                    let still = self.still(*i).is_some();
                    json!({"item": i, "name": format!("c{i}.{}", if still { "jpg" } else { "mp4" }), "type": if still { super::STILL_TYPE } else { "Movie" }, "duration": s * tps})
                }).collect::<Vec<_>>()}})),
                "media.captureDate" => {
                    Ok(json!({"items": self.clips.iter().map(|(i, _, u)| json!({"item": i, "unix": u, "gps": self.still(*i).flatten()})).collect::<Vec<_>>()}))
                }
                "media.status" => {
                    Ok(json!(self.files.iter().map(|(i, f)| json!({"item": i, "path": f.to_string_lossy(), "status": "online"})).collect::<Vec<_>>()))
                }
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
    fn scene_index_facts() {
        let index = json!({
            "analyzed": true, "describer": "qwen3-vl-2b",
            "shots": [
                {"start": 0.0, "end": 4.0, "keyframes": [2.0], "quality": {"score": 0.9},
                 "description": {"caption": "A crowd in a hall.", "subjects": [{"name": "people", "box": [0, 0, 1, 1]}]}},
                {"start": 4.0, "end": 10.0, "keyframes": [7.0], "quality": {"score": 0.6},
                 "description": {"caption": "A white humanoid robot waves.", "setting": "an exhibition hall",
                   "subjects": [{"name": "white humanoid robot", "box": [0.2, 0.1, 0.4, 0.8]}], "text": ["PEPPER"]}},
                {"start": 10.0, "end": 12.0, "keyframes": [11.0], "quality": {"score": 0.5},
                 "description": {"caption": "Robots line a stage.", "subjects": [{"name": "robots"}]}}
            ]
        });
        let f = scene_facts(&index, "robot").unwrap();
        // the best-scoring shot that shows the subject, its box's size counting in
        assert_eq!(f.best, 7.0);
        assert_eq!(f.subject, Some([0.2, 0.1, 0.4, 0.8]));
        assert_eq!(f.label, "white humanoid robot");
        assert_eq!(f.description, "A white humanoid robot waves. (an exhibition hall)");
        assert_eq!(f.text, vec!["PEPPER"]);
        // the crowd shot is unseen; moments at each shot's edges and keyframe
        assert!(f.frames.iter().filter(|x| x.seconds < 4.0).all(|x| !x.present));
        assert!(f.frames.iter().any(|x| x.present && (x.seconds - 7.0).abs() < 1e-9));
        assert!(f.frames.windows(2).all(|w| w[0].seconds <= w[1].seconds));
        // not described, not analysed, or the subject nowhere: the vision model is asked
        assert!(scene_facts(&json!({"analyzed": true, "describer": "", "shots": []}), "robot").is_none());
        assert!(scene_facts(&json!({"analyzed": false}), "robot").is_none());
        assert!(scene_facts(&index, "giraffe").is_none());
        // hostile shapes and numbers
        let bad = json!({"analyzed": true, "describer": "x", "shots": [{"start": "a"}, {"start": 5, "end": 1}, {"start": 0, "end": 1e308,
            "description": {"caption": "robot", "subjects": [{"name": "robot", "box": [2, -1, 0, 0]}]}}]});
        let f = scene_facts(&bad, "robot").unwrap();
        assert_eq!(f.subject, None);
        assert!(f.frames.iter().all(|x| x.seconds.is_finite()));
        assert!(names_subject("Robotic arm", "robot") && names_subject("two robots", "robots") && !names_subject("rob", "robot"));
    }

    #[tokio::test]
    async fn a_scene_index_replaces_frame_by_frame_grounding() {
        use crate::analyze::testing::FakeEx;
        // an executor whose item 1 has a scene index showing a robot from 4 to 10 s
        struct WithScenes(FakeEx);
        impl Executor for WithScenes {
            fn execute(&self, id: &str, p: Value) -> crate::ExecFuture {
                if id == "media.scenes" {
                    let v = json!({"analyzed": true, "describer": "qwen3-vl-2b", "shots": [
                        {"start": 0.0, "end": 4.0, "keyframes": [2.0], "quality": {"score": 0.8}, "description": {"caption": "An empty hall."}},
                        {"start": 4.0, "end": 10.0, "keyframes": [7.0], "quality": {"score": 0.8},
                         "description": {"caption": "A robot dog walks.", "subjects": [{"name": "robot dog", "box": [0.3, 0.3, 0.4, 0.4]}]}}]});
                    return Box::pin(async move { Ok(v) });
                }
                self.0.execute(id, p)
            }
        }
        let ex = WithScenes(FakeEx::new(vec![(1, 10.0, 100)]));
        // the model only reads text and describes: no grounding prompt reaches it
        let model = crate::llm::FnModel::new("fake", |_, user, _| {
            assert!(!user.starts_with("Is there a"), "no frame-by-frame grounding: {user}");
            if user.starts_with("Read the text") {
                Ok(json!({"on_subject": [], "nearby": []}))
            } else {
                Ok(json!({"label": "white robot dog", "kind": "quadruped", "description": "A white robot dog walks across a hall."}))
            }
        });
        let facts = analyze(&model, &ex, "robot", None, 8, 0, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
        let f = &facts[0];
        assert!(f.present);
        assert_eq!(f.best, 7.0);
        assert_eq!(f.subject, Some([0.3, 0.3, 0.4, 0.4]));
        assert_eq!(f.label, "white robot dog");
        assert!(f.frames.iter().any(|x| x.present) && f.frames.iter().any(|x| !x.present));
        // the frames the model saw were the best ones, not a sweep of the clip
        let asked: Vec<f64> = ex.0.frames.lock().unwrap().iter().map(|x| x.1).collect();
        assert!(asked.iter().all(|t| *t > 4.0 && *t < 10.0), "{asked:?}");
    }

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
        let facts = analyze(&model, &ex, "robot", None, 2, 0, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
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
        let facts = analyze(&model, &ex, "robot", None, 6, 3, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
        assert!(!facts[0].present && facts[0].frames.len() == 3, "three looks, then skipped: {:?}", facts[0].frames);
        assert!(facts[1].present);
        assert_eq!(facts[1].frames.len(), 6, "a clip that shows it is looked at fully");
        assert!(facts[1].frames.windows(2).all(|w| w[0].seconds < w[1].seconds), "frames in time order");
        assert!((facts[1].coverage - 1.0 / 6.0).abs() < 1e-9);
    }

    /// The answers for two 10 s clips analysed with two frames each: the first shows a robot (two
    /// frames located, two read, one description), the second does not.
    fn robot_then_nothing() -> Vec<Value> {
        let mut answers = grounded();
        answers.extend([
            json!({"on_subject": ["ARDET"], "nearby": []}),
            json!({"on_subject": [], "nearby": []}),
            json!({"label": "ARDET robot", "kind": "wheeled", "description": "A red wheeled robot."}),
            json!({"present": false}),
            json!({"present": false}),
        ]);
        answers
    }

    fn media_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("autocut-analyze-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[tokio::test]
    async fn a_clip_seen_before_is_not_analysed_again() {
        let dir = media_dir("seen");
        let (a, b) = (dir.join("a.mp4"), dir.join("b.mp4"));
        std::fs::write(&a, b"robot clip bytes").unwrap();
        std::fs::write(&b, b"other clip bytes").unwrap();
        let mut ex = testing::FakeEx::new(vec![(1, 10.0, 1_700_000_000), (2, 10.0, 1_700_000_100)]);
        ex.files = vec![(1, a.clone()), (2, b.clone())];
        let data = dir.join("data");
        let model = FixedModel::new(robot_then_nothing());
        let cache = AnalysisCache::in_data_dir(&data);
        let first = analyze(&model, &ex, "robot", None, 2, 0, &cache, &mut |_| {}).await.unwrap();
        assert_eq!((cache.stored(), cache.hits()), (2, 0));
        assert!(first[0].present && !first[1].present);
        // again, over the same library: nothing is asked of the model
        let mute = FixedModel::new(vec![]);
        let again = AnalysisCache::in_data_dir(&data);
        let mut lines: Vec<String> = vec![];
        let second = analyze(&mute, &ex, "robot", None, 2, 0, &again, &mut |l| lines.push(l.to_string())).await.unwrap();
        assert!(mute.asked.lock().unwrap().is_empty(), "the model was not asked");
        assert_eq!(again.hits(), 2);
        // the same facts, from the cache (JSON keeps floats to within an ulp)
        assert_eq!(second.len(), first.len());
        for (s, f) in second.iter().zip(&first) {
            assert_eq!(
                (s.item, &s.name, s.present, &s.label, &s.kind, &s.text, s.best, s.unix),
                (f.item, &f.name, f.present, &f.label, &f.kind, &f.text, f.best, f.unix)
            );
            assert_eq!(s.frames.len(), f.frames.len());
            for (a, b) in s.frames.iter().zip(&f.frames) {
                assert!(a.present == b.present && (a.area - b.area).abs() < 1e-12 && (a.center[1] - b.center[1]).abs() < 1e-12, "{a:?} {b:?}");
            }
        }
        assert!(lines.iter().any(|l| l.contains("2 of 2 clip(s) were analysed before")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("(analysed before)")), "{lines:?}");
        // the same file imported again elsewhere, as another item: still known, with its new place
        let moved = dir.join("moved.mp4");
        std::fs::copy(&a, &moved).unwrap();
        let mut ex2 = testing::FakeEx::new(vec![(9, 10.0, 1_600_000_000)]);
        ex2.files = vec![(9, moved)];
        let got = analyze(&mute, &ex2, "robot", None, 2, 0, &AnalysisCache::in_data_dir(&data), &mut |_| {}).await.unwrap();
        assert_eq!((got[0].item, got[0].name.as_str(), got[0].label.as_str(), got[0].unix), (9, "c9.mp4", "ARDET robot", Some(1_600_000_000)));
        // other settings are another analysis: asked again
        let other = AnalysisCache::in_data_dir(&data);
        let model = FixedModel::new(robot_then_nothing());
        analyze(&model, &ex, "robot", None, 2, 3, &other, &mut |_| {}).await.unwrap();
        assert_eq!(other.hits(), 0, "screening changes the analysis");
        // --fresh: every clip analysed again, and kept
        let fresh = AnalysisCache::in_data_dir(&data).fresh(true);
        let model = FixedModel::new(robot_then_nothing());
        analyze(&model, &ex, "robot", None, 2, 0, &fresh, &mut |_| {}).await.unwrap();
        assert_eq!((fresh.hits(), fresh.stored()), (0, 2));
        assert_eq!(model.asked.lock().unwrap().len(), 7, "both clips analysed again");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn an_analysis_the_model_did_not_answer_is_not_kept() {
        let dir = media_dir("unanswered");
        let a = dir.join("a.mp4");
        std::fs::write(&a, b"robot clip bytes").unwrap();
        let mut ex = testing::FakeEx::new(vec![(1, 10.0, 1_700_000_000)]);
        ex.files = vec![(1, a)];
        let data = dir.join("data");
        // Ollama down: the clip reads as "no robot" — and must not stay that way
        let cache = AnalysisCache::in_data_dir(&data);
        let facts = analyze(&FixedModel::new(vec![]), &ex, "robot", None, 2, 0, &cache, &mut |_| {}).await.unwrap();
        assert!(!facts[0].present);
        assert_eq!(cache.stored(), 0);
        // with the model back, it is analysed (and kept)
        let cache = AnalysisCache::in_data_dir(&data);
        let mut answers = grounded();
        answers.extend([
            json!({"on_subject": [], "nearby": []}),
            json!({"on_subject": [], "nearby": []}),
            json!({"label": "robot", "kind": "other", "description": "A robot."}),
        ]);
        let facts = analyze(&FixedModel::new(answers), &ex, "robot", None, 2, 0, &cache, &mut |_| {}).await.unwrap();
        assert!(facts[0].present);
        assert_eq!((cache.hits(), cache.stored()), (0, 1));
        // a target that cannot say which file an item is: analysed, nothing kept
        let blind = testing::FakeEx::new(vec![(1, 10.0, 1_700_000_000)]);
        let cache = AnalysisCache::in_data_dir(&data);
        analyze(&FixedModel::new(robot_then_nothing()), &blind, "robot", None, 2, 0, &cache, &mut |_| {}).await.unwrap();
        assert_eq!((cache.hits(), cache.stored()), (0, 0));
        std::fs::remove_dir_all(&dir).ok();
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
        let facts = analyze(&model, &ex, "robot", None, 2, 0, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
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
        let facts = run(&model, &ex, "robot", None, 2, 0, Some(earlier.clone()), 10, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
        assert_eq!(facts.iter().map(|f| f.identity).collect::<Vec<_>>(), vec![Some(1), Some(1)]);
        // no comparisons: each its own robot
        let facts = run(&FixedModel::new(vec![]), &ex, "robot", None, 2, 0, Some(earlier), 0, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
        assert_eq!(facts.iter().map(|f| f.identity).collect::<Vec<_>>(), vec![Some(1), Some(2)]);
    }

    #[tokio::test]
    async fn a_photo_is_looked_at_once_and_keeps_its_box_date_and_place() {
        // a 10 s clip and a photo (a still, 5 s by default in the project)
        let mut ex = testing::FakeEx::new(vec![(1, 10.0, 1_700_000_000), (2, 5.0, 1_700_000_100)]);
        ex.stills = vec![(2, Some([25.2, 55.3]))];
        let model = crate::llm::FnModel::new("scripted", |system, user, _| {
            Ok(if system == GROUND_SYSTEM {
                // the model frame of an 8 × 4 picture is 1288 × 644: the box is its right half
                json!({"present": true, "bbox_2d": [644, 0, 1288, 644]})
            } else if system.contains("describe") {
                json!({"label": if user.starts_with("This photo") { "Ameca humanoid" } else { "white robot" }, "kind": "humanoid", "description": "A robot."})
            } else {
                json!({"on_subject": [], "nearby": []})
            })
        });
        let facts = analyze(&model, &ex, "robot", None, 3, 0, &AnalysisCache::off(), &mut |_| {}).await.unwrap();
        let (clip, photo) = (&facts[0], &facts[1]);
        assert!(!clip.photo && clip.frames.len() == 3, "{clip:?}");
        assert!(photo.photo && photo.present, "{photo:?}");
        assert_eq!(photo.frames.len(), 1, "one look at a photo");
        assert_eq!((photo.frames[0].seconds, photo.best), (0.0, 0.0));
        assert_eq!(photo.label, "Ameca humanoid", "described as a photo");
        let b = photo.subject.unwrap();
        assert!((b[0] - 0.5).abs() < 1e-9 && (b[2] - 0.5).abs() < 1e-9 && (b[3] - 1.0).abs() < 1e-9, "{b:?}");
        assert_eq!((photo.gps, photo.unix), (Some([25.2, 55.3]), Some(1_700_000_100)));
        assert_eq!(clip.gps, None);
        // the photo's frames: one to locate, one larger to read
        let frames = ex.frames.lock().unwrap();
        let of_photo: Vec<_> = frames.iter().filter(|f| f.0 == 2).collect();
        assert_eq!(of_photo.len(), 2, "{of_photo:?}");
        assert!(of_photo.iter().all(|f| f.1 == 0.0));
        // a report written and read back keeps it all; an old report without them still loads
        let back: ClipFacts = serde_json::from_value(serde_json::to_value(photo).unwrap()).unwrap();
        assert_eq!(&back, photo);
        let v = serde_json::to_value(clip).unwrap();
        assert!(v.get("photo").is_none() && v.get("subject").is_some() && v.get("gps").is_none(), "{v}");
    }

    #[test]
    fn photos_are_media_items_too() {
        let t = filmcraft_time::TICKS_PER_SECOND as f64;
        let p = json!({"root": {"children": [
            {"item": 3, "name": "b.mp4", "type": "Movie", "duration": 2.0 * t},
            {"item": 4, "name": "IMG_1.HEIC", "type": "Still Image", "duration": 5.0 * t},
            {"children": [{"item": 9, "name": "music.wav", "type": "Audio", "duration": t}]}
        ]}});
        assert_eq!(media_items(&p), vec![(3, "b.mp4".into(), 2.0, false), (4, "IMG_1.HEIC".into(), 5.0, true)]);
        assert_eq!(video_items(&p), vec![(3, "b.mp4".into(), 2.0)]);
        // a hostile, deeply nested tree is walked within bounds
        let mut deep = json!({"item": 1, "name": "x.mp4", "type": "Movie", "duration": t});
        for _ in 0..1000 {
            deep = json!({"children": [deep]});
        }
        assert!(media_items(&deep).is_empty(), "too deep to be a project");
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

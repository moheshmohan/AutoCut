//! Clip analysis: which media items show a subject ("robot"), what it is, when it was recorded,
//! and the moment that shows it best — the facts a showcase edit is planned from.
//!
//! For each video item, frames are sampled through the engine (`media.frame`) and a grounding
//! vision model (Qwen2.5-VL class, [`crate::config::Config::analysis_model`]) is asked, frame
//! by frame, to *locate* the subject: a box, or none. Locating is what these models do reliably;
//! a box also says how big and where the subject is, so the best moment is the frame where it is
//! largest. One more request per clip, over its best frames, names and describes it — a brand
//! or model only when the model is certain (a 7B model otherwise invents brand names).
//!
//! Capture dates come from `media.captureDate` (sidecars from the Google Photos picker, or the
//! file's own metadata), so results sort chronologically.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, Executor};

/// Frames sampled per clip, at most.
pub const MAX_FRAMES: usize = 8;
/// Seconds between samples on short clips.
const SAMPLE_EVERY: f64 = 3.0;
/// A subject box smaller than this share of the frame is not "showing" it.
const MIN_AREA: f64 = 0.01;

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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    /// Where it shows best (seconds into the clip).
    pub best: f64,
    pub frames: Vec<FrameFact>,
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

fn ground_prompt(subject: &str) -> String {
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

fn describe_prompt(subject: &str) -> String {
    let guide = if subject.to_lowercase().contains("robot") { ROBOT_GUIDE } else { "" };
    format!(
        "These frames come from one video showing a {subject}.\n{guide}Answer with JSON only: {{\"label\": \"what it is in 2-5 words: a brand or model name ONLY if it clearly matches the guide or a logo is readable; otherwise a plain description such as 'floor-cleaning robot' or 'white robot dog'\", \"kind\": \"humanoid|quadruped|arm|wheeled|toy|drone|other\", \"description\": \"one plain sentence: what it looks like and what it is doing\"}}"
    )
}

/// Describe a clip again from `frames` (PNG, best first) — after the guide or model changed.
async fn describe(model: &dyn ChatModel, subject: &str, imgs: &[Vec<u8>], f: &mut ClipFacts) -> Result<(), AutoCutError> {
    let v = model.complete_json("You describe videos for a video editor.", &describe_prompt(subject), imgs).await?;
    let text = |k: &str, n: usize| v.get(k).and_then(Value::as_str).map(|s| s.trim().chars().take(n).collect::<String>()).unwrap_or_default();
    f.kind = text("kind", 20).to_lowercase();
    f.description = text("description", 300);
    f.label = sanitize_label(&text("label", 60), &f.kind, subject);
    Ok(())
}

/// Analyse afresh, or (with `earlier`) only describe again from an earlier analysis.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    items: Option<&[u64]>,
    frames_max: usize,
    earlier: Option<Vec<ClipFacts>>,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ClipFacts>, AutoCutError> {
    match earlier {
        Some(mut facts) => {
            relabel(model, ex, subject, &mut facts, progress).await?;
            Ok(facts)
        }
        None => analyze(model, ex, subject, items, frames_max, progress).await,
    }
}

/// Re-describe every clip that shows the subject, from its (up to three) best frames as the
/// earlier analysis found them: names and descriptions improve without locating again.
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
        let mut best: Vec<&FrameFact> = f.frames.iter().filter(|x| x.present).collect();
        best.sort_by(|a, b| b.area.total_cmp(&a.area));
        let mut times: Vec<f64> = best.iter().take(3).map(|x| x.seconds).collect();
        times.sort_by(f64::total_cmp);
        let mut imgs = Vec::new();
        for t in times {
            if let Ok(r) = ex.execute("media.frame", json!({"item": f.item, "seconds": t, "maxSide": 640})).await
                && let Some(png) = r["pngBase64"].as_str().and_then(crate::b64_decode)
            {
                imgs.push(png);
            }
        }
        if imgs.is_empty() {
            continue;
        }
        let before = f.label.clone();
        match describe(model, subject, &imgs, f).await {
            Ok(()) => progress(&format!("[{}/{n}] {}: {before} → {}", k + 1, f.name, f.label)),
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

/// Keep a model's label only if every capitalised word in it is a known name; otherwise say
/// what it is plainly: "{kind} {subject}" ("quadruped robot").
pub fn sanitize_label(label: &str, kind: &str, subject: &str) -> String {
    let label = label.trim();
    let invented = label.split(|c: char| c.is_whitespace() || c == '-').any(|w| {
        let first_upper = w.chars().next().is_some_and(char::is_uppercase);
        let lw: String = w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        first_upper
            && !lw.is_empty()
            && !KNOWN_NAMES.contains(&lw.as_str())
            && lw != subject
            && !["robot", "robotic", "the", "a", "humanoid", "industrial", "toy", "dog", "arm"].contains(&lw.as_str())
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

/// One clip: frames → grounding → description.
#[allow(clippy::too_many_arguments)]
async fn analyze_clip(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    item: u64,
    name: &str,
    seconds: f64,
    frames_max: usize,
    progress: &mut dyn FnMut(&str),
) -> Result<ClipFacts, AutoCutError> {
    let mut frames = Vec::new();
    let mut pngs: Vec<(f64, f64, Vec<u8>)> = Vec::new(); // (area, seconds, png)
    for t in sample_times(seconds, frames_max) {
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
        let reply = model.complete_json("You locate objects in photos for a video editor.", &ground_prompt(subject), std::slice::from_ref(&png)).await;
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
        best,
        frames,
    };
    if facts.present {
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
        let mut f = analyze_clip(model, ex, subject, item, &name, secs, frames_max.clamp(1, MAX_FRAMES), progress).await?;
        (f.created, f.unix) = date_of(item);
        progress(&format!(
            "[{}/{n}] {name}: {}",
            k + 1,
            if f.present { format!("{} ({:.0}% of frames)", f.label, f.coverage * 100.0) } else { format!("no {subject}") }
        ));
        out.push(f);
    }
    out.sort_by_key(|f| (f.unix.unwrap_or(i64::MAX), f.item));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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

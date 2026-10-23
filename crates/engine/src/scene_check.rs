//! Checking words against the pictures they will be shown over (`media.checkText`): before an
//! automatic edit puts a title on screen, how well each candidate text matches the frames of each
//! moment (SigLIP 2's image–text similarity), and whether the vision-language model (Qwen3-VL,
//! in process) sees what a text names in the frames under it.
//!
//! | command | does |
//! |---|---|
//! | `media.checkText {windows, aspect?, texts?, asks?, frames?, wait?}` | a background job (`jobs.list`; `wait: true` blocks): `windows` are moments `[{item, from, to, focus?}]` (media seconds), `texts` the words to match; the answer (`scenes` in the job's entry) has `cosine` and `match` (texts × windows), `similar` (windows × windows), and for each of `asks` (`{window, text, about?}`) the model's `yes`, `no` or `unclear` |
//!
//! **Frames.** `frames` (1–4, default 2) pictures are decoded evenly inside each window (a still:
//! its one picture), at most 640 pixels on their long side; with `aspect` (the edit's frame,
//! width / height) each is cut to what the edit shows of it — that shape around the window's
//! `focus`, as a template's `fill` framing does ([`crop_box`]). **Similarity.** Each window's frames
//! are embedded by SigLIP 2 and averaged; each text is embedded as it is (lower case); `cosine` is
//! their cosine, `match` SigLIP's own calibrated probability (`sigmoid(scale · cos + bias)`), and
//! `similar` the windows' cosine with one another (two moments that look alike). **Asks.** The
//! largest installed describer (Qwen3-VL 4B, else 2B) is shown a window's frames and asked whether
//! the one thing of the text an ask is `about` is visible in them (a describer answers that far
//! more reliably than whether a whole title is true), or, without `about`, whether everything the
//! text names is. The models take the
//! GPU lock while they run and the describer is let go afterwards.
//!
//! Without the models (or in a build without them) the answer says so (`embedder` / `asker`
//! null, a `notes` line) and has no scores: callers fall back to the words. Nothing is changed in
//! the project.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_project::ItemId;
use filmcraft_project::scenes::Embedding;
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p};
use crate::scene_models::{Pic, SceneModels};
use crate::{EngineError, Result, Session};

const CMD: &str = "media.checkText";
/// Windows, texts and asks one call takes, at most.
pub const MAX_WINDOWS: usize = 64;
pub const MAX_TEXTS: usize = 64;
pub const MAX_ASKS: usize = 48;
/// The longest text, characters, and the longest thing an ask is about.
const MAX_TEXT_CHARS: usize = 200;
const MAX_ABOUT_CHARS: usize = 60;
/// Frames a window is looked at by, at most (the describer takes four).
const MAX_FRAMES: usize = 4;
/// Long side of a decoded frame.
const FRAME_SIDE: u32 = 640;
/// Tokens an answer may have.
const ANSWER_TOKENS: usize = 4;

/// How the describer is told to answer.
pub const ASK_SYSTEM: &str = "You check the on-screen text of a video against its pictures. You answer with one word: yes or no.";

/// The question asked about `text` over a window's frames: whether one thing it names (`about`,
/// a word or two of it) is in them — a describer answers that far more reliably than whether a
/// whole title is true — else whether everything it names is.
pub fn ask_question(text: &str, about: Option<&str>) -> String {
    let clean = |s: &str, n: usize| s.chars().take(n).collect::<String>().replace('"', "'");
    let text = clean(text, MAX_TEXT_CHARS);
    match about.map(|a| clean(a.trim(), MAX_ABOUT_CHARS)).filter(|a| !a.is_empty()) {
        Some(a) => format!(
            "These pictures are frames of one moment of a video. The text \"{text}\" will be shown over it. Look carefully: is the \"{a}\" it mentions actually visible in these frames (not only a reflection, a poster or something like it)? Answer yes or no."
        ),
        None => format!(
            "These pictures are frames of one moment of a video. The text \"{text}\" will be shown over it. Is every person, animal, vehicle or object the text names clearly visible in these frames, so the text truthfully describes them? Answer yes or no."
        ),
    }
}

/// How an answer reads: `yes`, `no` or `unclear`.
pub fn read_answer(a: &str) -> &'static str {
    let first: String = a.trim().chars().take_while(|c| c.is_alphabetic()).flat_map(char::to_lowercase).collect();
    match first.as_str() {
        "yes" | "y" | "true" => "yes",
        "no" | "n" | "false" => "no",
        _ => "unclear",
    }
}

type Outcome = Arc<Mutex<Option<std::result::Result<Value, String>>>>;

/// A check job not yet collected.
pub(crate) struct PendingCheck {
    job: u64,
    out: Outcome,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Text Against Frames",
        menu: &[],
        shortcut: None,
        params: r#"{"windows":[{"item":id,"from":seconds,"to":seconds,"focus":[x,y]?}],"aspect":f64?,"texts":[str]?,"asks":[{"window":u32,"text":u32,"about":str?}]?,"frames":u32=2,"wait":bool=false}"#,
        enabled: always,
        run: check,
        journal: false,
    }]
}

/// One moment to look at.
struct Window {
    item: ItemId,
    src: filmcraft_media::SharedSource,
    still: bool,
    from: Tick,
    to: Tick,
    /// The part of the picture the edit shows: the frame shape (width / height) cut around the
    /// focus (fractions of the picture), as a slot's `fill` framing does.
    crop: Option<(f64, [f64; 2])>,
}

/// The `[x, y, w, h]` (pixels) of a `w` × `h` picture that a frame of shape `aspect` (width /
/// height) filled around `focus` shows: the whole picture when the shapes match.
pub fn crop_box(w: usize, h: usize, aspect: f64, focus: [f64; 2]) -> [usize; 4] {
    if w == 0 || h == 0 || !aspect.is_finite() || aspect <= 0.0 {
        return [0, 0, w, h];
    }
    let f = |v: f64| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 };
    let (fx, fy) = (f(focus[0]), f(focus[1]));
    let shape = w as f64 / h as f64;
    if shape > aspect * 1.01 {
        let cw = ((h as f64 * aspect).round() as usize).clamp(1, w);
        let x = ((fx * w as f64 - cw as f64 / 2.0).round().max(0.0) as usize).min(w - cw);
        [x, 0, cw, h]
    } else if shape < aspect / 1.01 {
        let ch = ((w as f64 / aspect).round() as usize).clamp(1, h);
        let y = ((fy * h as f64 - ch as f64 / 2.0).round().max(0.0) as usize).min(h - ch);
        [0, y, w, ch]
    } else {
        [0, 0, w, h]
    }
}

/// The `[x, y, w, h]` part of a straight RGBA8 picture (inside it: [`crop_box`] makes it so).
fn crop_rgba(rgba: &[u8], w: usize, b: [usize; 4]) -> Option<Pic> {
    let [x, y, cw, ch] = b;
    let mut out = Vec::new();
    out.try_reserve_exact(cw.checked_mul(ch)?.checked_mul(4)?).ok()?;
    for row in y..y.checked_add(ch)? {
        let at = row.checked_mul(w)?.checked_add(x)?.checked_mul(4)?;
        out.extend_from_slice(rgba.get(at..at.checked_add(cw.checked_mul(4)?)?)?);
    }
    Some(Pic { rgba: out, w: cw, h: ch })
}

/// The windows of `p` (each item resolved to its media, the span clamped into it).
fn windows(s: &Session, p: &Value) -> Result<Vec<Window>> {
    let list = p.get("windows").and_then(Value::as_array).ok_or_else(|| bad(CMD, "`windows` is a list of {item, from, to}"))?;
    if list.is_empty() || list.len() > MAX_WINDOWS {
        return Err(bad(CMD, format!("1 to {MAX_WINDOWS} windows")));
    }
    let aspect = match p.get("aspect") {
        None | Some(Value::Null) => None,
        Some(v) => {
            Some(v.as_f64().filter(|a| a.is_finite() && (0.1..=10.0).contains(a)).ok_or_else(|| bad(CMD, "`aspect`: the frame's width / height, 0.1 to 10"))?)
        }
    };
    let mut out = Vec::with_capacity(list.len());
    for (k, w) in list.iter().enumerate() {
        let item = w.get("item").and_then(Value::as_u64).map(ItemId).ok_or_else(|| bad(CMD, format!("window {k}: `item` is required")))?;
        let (root, media, _) = s.project.resolve_media(item).ok_or_else(|| bad(CMD, format!("window {k}: item {} is not media", item.0)))?;
        if !media.info.has_video() {
            return Err(bad(CMD, format!("window {k}: item {} has no picture", item.0)));
        }
        let still = matches!(media.info.kind, filmcraft_media::MediaKind::Still);
        let duration = media.duration().max(Tick::ZERO);
        let secs = |key: &str| -> Result<Option<f64>> {
            match w.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => v
                    .as_f64()
                    .filter(|f| f.is_finite() && *f >= 0.0 && *f <= 86_400.0)
                    .map(Some)
                    .ok_or_else(|| bad(CMD, format!("window {k}: `{key}` is seconds"))),
            }
        };
        let (a, b) = (secs("from")?.unwrap_or(0.0), secs("to")?.unwrap_or_else(|| duration.seconds()));
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let clamp = |t: f64| Tick::from_seconds_f64(t).clamp(Tick::ZERO, duration);
        let src = s.source(root).ok_or_else(|| bad(CMD, format!("window {k}: its media is offline or unreadable")))?;
        let focus = match w.get("focus") {
            None | Some(Value::Null) => [0.5, 0.5],
            Some(v) => {
                let f: Option<Vec<f64>> = v.as_array().filter(|a| a.len() == 2).and_then(|a| a.iter().map(|x| x.as_f64().filter(|f| f.is_finite())).collect());
                let f = f.ok_or_else(|| bad(CMD, format!("window {k}: `focus` is [x, y], fractions of the picture")))?;
                [f.first().copied().unwrap_or(0.5).clamp(0.0, 1.0), f.get(1).copied().unwrap_or(0.5).clamp(0.0, 1.0)]
            }
        };
        out.push(Window { item: root, src, still, from: clamp(a), to: clamp(b), crop: aspect.map(|a| (a, focus)) });
    }
    Ok(out)
}

fn texts(p: &Value) -> Result<Vec<String>> {
    match p.get("texts") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= MAX_TEXTS => a
            .iter()
            .map(|t| t.as_str().map(|s| s.trim().chars().take(MAX_TEXT_CHARS).collect::<String>()).filter(|s| !s.is_empty()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| bad(CMD, "`texts` are non-empty strings")),
        Some(_) => Err(bad(CMD, format!("`texts` is a list of at most {MAX_TEXTS} strings"))),
    }
}

/// One question: a window, a text and, optionally, the one thing of it asked about.
#[derive(Clone, Debug, PartialEq)]
struct Ask {
    window: usize,
    text: usize,
    about: Option<String>,
}

fn asks(p: &Value, windows: usize, texts: usize) -> Result<Vec<Ask>> {
    match p.get("asks") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= MAX_ASKS => a
            .iter()
            .map(|x| {
                let index = |k: &str, n: usize| x.get(k).and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok()).filter(|i| *i < n);
                let window = index("window", windows)?;
                let text = index("text", texts)?;
                let about = match x.get("about") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(v.as_str().map(|s| s.trim().chars().take(MAX_ABOUT_CHARS).collect::<String>()).filter(|s| !s.is_empty())?),
                };
                Some(Ask { window, text, about })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| bad(CMD, "each ask is {window, text, about?}: indices into `windows` and `texts`, and a word or two")),
        Some(_) => Err(bad(CMD, format!("`asks` is a list of at most {MAX_ASKS} {{window, text, about?}}"))),
    }
}

/// `n` frames spread evenly inside the window (a still: its one picture).
fn frames_of(w: &Window, n: usize) -> Vec<Pic> {
    let info = w.src.info().video.clone();
    let Some(info) = info else { return Vec::new() };
    let scale = (FRAME_SIDE as f32 / info.width.max(info.height).max(1) as f32).clamp(0.01, 1.0);
    let n = if w.still { 1 } else { n.clamp(1, MAX_FRAMES) };
    let span = (w.to - w.from).max(Tick::ZERO);
    let mut out = Vec::new();
    for k in 0..n {
        let t = if w.still { Tick::ZERO } else { w.from + Tick((span.0 as f64 * (k as f64 + 0.5) / n as f64) as i64) };
        let Ok(vf) = w.src.video_frame(filmcraft_media::FrameRequest { time: t, scale }) else { continue };
        let (rgba, fw, fh) = crate::media_frame::downscale_rgba(&vf.to_rgba8(), vf.width, vf.height, FRAME_SIDE);
        let (fw, fh) = (fw as usize, fh as usize);
        if fw == 0 || fh == 0 {
            continue;
        }
        // what the edit shows of it: the frame's shape around the focus
        let pic = match w.crop {
            Some((aspect, focus)) => crop_rgba(&rgba, fw, crop_box(fw, fh, aspect, focus)),
            None => Some(Pic { rgba, w: fw, h: fh }),
        };
        out.extend(pic);
    }
    // the describer wants one size: frames of one source are, but a damaged one may not be
    if let Some(first) = out.first().map(|p| (p.w, p.h)) {
        out.retain(|p| (p.w, p.h) == first);
    }
    out
}

fn round(v: f32) -> f64 {
    if v.is_finite() { (f64::from(v) * 1e4).round() / 1e4 } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The job: frames, embeddings, the asks.
fn run_check(ws: Vec<Window>, texts: Vec<String>, asks: Vec<Ask>, n: usize, models: Option<Arc<dyn SceneModels>>, prog: &filmcraft_export::Progress) -> Value {
    let mut notes: Vec<String> = Vec::new();
    let m = models.as_deref();
    let embedder = m.and_then(|m| m.embedder_id());
    let asker = if asks.is_empty() { None } else { m.and_then(|m| m.asker_id()) };
    if m.is_none() {
        notes.push("this build has no scene models: nothing was looked at".into());
    } else if embedder.is_none() && !texts.is_empty() {
        notes.push("the embedding model (SigLIP 2) is not downloaded: no similarity".into());
    }
    if !asks.is_empty() && m.is_some() && asker.is_none() {
        notes.push("no describer (Qwen3-VL) is downloaded: nothing was asked".into());
    }
    // the GPU, while the models run
    let _gpu = match m.filter(|_| embedder.is_some() || asker.is_some()).map(|m| m.gpu()) {
        Some(Ok(g)) => g,
        Some(Err(e)) => {
            notes.push(format!("the GPU: {e}"));
            None
        }
        None => None,
    };
    *lock(&prog.status) = "Looking at the frames".into();
    let pics: Vec<Vec<Pic>> = ws
        .iter()
        .map(|w| {
            prog.done.fetch_add(1, Ordering::Relaxed);
            frames_of(w, n)
        })
        .collect();
    for (k, p) in pics.iter().enumerate() {
        if p.is_empty() {
            notes.push(format!("window {k}: no frame decoded"));
        }
    }
    // similarity
    let mut cosine: Option<Vec<Vec<f64>>> = None;
    let mut matched: Option<Vec<Vec<f64>>> = None;
    let mut similar: Option<Vec<Vec<f64>>> = None;
    if let (Some(m), Some(_)) = (m, &embedder) {
        *lock(&prog.status) = "Matching the words to the frames".into();
        let mut wemb: Vec<Option<Embedding>> = Vec::with_capacity(pics.len());
        for p in &pics {
            if p.is_empty() {
                wemb.push(None);
                continue;
            }
            match m.embed_images(p) {
                Ok(v) if !v.is_empty() => {
                    let d = v.first().map_or(0, Vec::len);
                    let mut mean = vec![0f32; d];
                    for e in v.iter().filter(|e| e.len() == d) {
                        mean.iter_mut().zip(e).for_each(|(a, b)| *a += b);
                    }
                    wemb.push(Some(Embedding(mean).normalized()));
                }
                Ok(_) => wemb.push(None),
                Err(e) => {
                    notes.push(format!("embeddings: {e}"));
                    wemb.push(None);
                }
            }
        }
        let mut cs = Vec::with_capacity(texts.len());
        let mut ms = Vec::with_capacity(texts.len());
        for t in &texts {
            match m.embed_text(t) {
                Ok((te, cal)) => {
                    let row: Vec<f32> = wemb.iter().map(|w| w.as_ref().map_or(0.0, |w| w.cosine(&te))).collect();
                    let to_match = |c: f32| match cal {
                        Some(k) if k.scale.is_finite() && k.bias.is_finite() => sigmoid(k.scale * c + k.bias),
                        _ => (c + 1.0) / 2.0,
                    };
                    ms.push(row.iter().map(|c| round(to_match(*c).clamp(0.0, 1.0))).collect());
                    cs.push(row.into_iter().map(round).collect());
                }
                Err(e) => {
                    notes.push(format!("text embedding: {e}"));
                    cs.push(vec![0.0; wemb.len()]);
                    ms.push(vec![0.0; wemb.len()]);
                }
            }
        }
        let sim: Vec<Vec<f64>> = wemb
            .iter()
            .map(|a| {
                wemb.iter()
                    .map(|b| match (a, b) {
                        (Some(a), Some(b)) => round(a.cosine(&b.0)),
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        cosine = Some(cs);
        matched = Some(ms);
        similar = Some(sim);
    }
    // the asks
    let mut answers = Vec::new();
    if let (Some(m), Some(_)) = (m, &asker) {
        for (k, a) in asks.iter().enumerate() {
            if prog.cancel.load(Ordering::Relaxed) {
                break;
            }
            *lock(&prog.status) = format!("Asking about the text: {} of {}", k + 1, asks.len());
            let (Some(frames), Some(text)) = (pics.get(a.window), texts.get(a.text)) else { continue };
            if frames.is_empty() {
                answers.push(json!({"window": a.window, "text": a.text, "about": a.about, "answer": "unclear", "raw": "", "note": "no frame"}));
                continue;
            }
            match m.ask(frames, ASK_SYSTEM, &ask_question(text, a.about.as_deref()), ANSWER_TOKENS) {
                Ok(ans) => {
                    let raw: String = ans.trim().chars().take(40).collect();
                    answers.push(json!({"window": a.window, "text": a.text, "about": a.about, "answer": read_answer(&raw), "raw": raw}));
                }
                Err(e) => {
                    notes.push(format!("ask: {e}"));
                    break;
                }
            }
        }
        m.release();
    }
    notes.sort();
    notes.dedup();
    json!({
        "windows": ws.iter().zip(&pics).map(|(w, p)| json!({"item": w.item.0, "from": w.from.seconds(), "to": w.to.seconds(), "frames": p.len()})).collect::<Vec<_>>(),
        "texts": texts,
        "cosine": cosine,
        "match": matched,
        "similar": similar,
        "answers": answers,
        "embedder": embedder,
        "asker": asker,
        "notes": notes,
    })
}

fn check(s: &mut Session, p: &Value) -> Result<Value> {
    let ws = windows(s, p)?;
    let ts = texts(p)?;
    let asks = asks(p, ws.len(), ts.len())?;
    let n = match p.get("frames") {
        None | Some(Value::Null) => 2,
        Some(v) => v.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| (1..=MAX_FRAMES).contains(n)).ok_or_else(|| bad(CMD, "`frames`: 1 to 4"))?,
    };
    let models = s.scenes.check_models();
    let job = crate::narration::new_job(s, format!("Check Text ({} window{})", ws.len(), if ws.len() == 1 { "" } else { "s" }));
    job.progress.total.store(ws.len() as u64, Ordering::Relaxed);
    let id = job.id;
    let out: Outcome = Arc::default();
    let (prog, res, put) = (job.progress.clone(), job.result.clone(), out.clone());
    let run = move || {
        let t0 = web_time::Instant::now();
        let v = run_check(ws, ts, asks, n, models, &prog);
        let status = format!("Checked {} text(s) against {} window(s)", v["texts"].as_array().map_or(0, Vec::len), v["windows"].as_array().map_or(0, Vec::len));
        *lock(&put) = Some(Ok(v));
        crate::narration::finish(&prog, &res, Ok(status), t0);
    };
    s.scenes.checks.push(PendingCheck { job: id, out });
    let wait = bool_p(p, "wait").unwrap_or(false);
    crate::narration::spawn(s, job, wait, "filmcraft-check-text", run)?;
    let mut answer = json!({"job": id});
    if wait || cfg!(target_arch = "wasm32") {
        crate::scenes::poll(s);
        match crate::scenes::outcome_json(s, id) {
            Some(o) if o.get("error").is_some() => return Err(EngineError::Other(format!("{CMD}: {}", o["error"]))),
            Some(o) => answer["scenes"] = o,
            None => {}
        }
    } else {
        answer["poll"] = json!("jobs.list");
    }
    Ok(answer)
}

/// Collect finished check jobs into the scene outcomes (`scenes` in their `jobs.list` entry).
pub(crate) fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.scenes.checks.len() {
        let job = s.scenes.checks.get(i).map_or(0, |c| c.job);
        let finished = s.jobs.iter().find(|j| j.id == job).is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let c = s.scenes.checks.remove(i);
        let outcome = match lock(&c.out).take() {
            Some(Ok(v)) => v,
            Some(Err(e)) => json!({"error": e}),
            None => json!({"error": "the job ended without a result"}),
        };
        s.scenes.put_outcome(job, outcome);
    }
}

#[cfg(test)]
#[path = "scene_check_tests.rs"]
mod tests;

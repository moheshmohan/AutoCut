//! Long video → shorts (roadmap E3, the Opus-Clip factory, local): `autocut.shorts`.
//!
//! 1. **Sentences** — the transcript's words (Whisper's, with punctuation) are cut into sentences
//!    at `.` `!` `?`, at pauses longer than [`SENTENCE_PAUSE`] and every [`MAX_SENTENCE_WORDS`].
//! 2. **Candidates** — every run of whole sentences `minSeconds`..`maxSeconds` long.
//! 3. **Score** — the highlight reel's interest score ([`crate::highlight::interest`]: content
//!    words, the topic, numbers) per sentence, weighted by how long each is spoken; a length term
//!    (best near the middle of the range); and the opening's hook: a question, a direct address or
//!    a bold claim, said briefly. With the local model the best openings are also rated for how
//!    well they hook a viewer (0..10) and get a short title.
//! 4. **Pick** — the best `count` that do not overlap (a second apart at least).
//! 5. **Build** — each through ordinary engine commands: `sequence.autoReframe` of its range to
//!    the vertical shape (the speaker followed; where nobody is, for a whole clip or a stretch of
//!    one, the whole picture over a blurred copy of itself — `fallback`, `"blur"` by default,
//!    `"letterbox"` for black bars), `transcript.createCaptions` (word-timed) and
//!    `captions.setAnimatedStyle` for Bold animated captions (`captionStyle`: another preset, or
//!    `"plain"` for big outlined whole captions; `captionKeywords`: the local model colours a few
//!    keywords and adds emoji — [`crate::caption_style`]), `graphics.newText` for the title card —
//!    both inside the target `platform`'s safe zone (`"youtube"` by default; `"reels"`,
//!    `"tiktok"`, `"vertical"` for all three): the captions centred on the middle of its safe span
//!    above its bottom zone (plain ones shrunk until `sequence.checkSafeZones` finds them clear of
//!    its side zones), the title in a box across the top of its safe area; what still overlaps
//!    is noted. Then `file.exportMedia` (captions
//!    burned in) and `youtube.upload` (private unless asked).
//!
//! Every step is a command, so the shorts are ordinary sequences in History; a step that fails
//! for one short is noted and the others still get made.

use std::sync::Arc;

use filmcraft_engine::safe_zones::Platform;
use serde::Serialize;
use serde_json::{Value, json};

use crate::caption_style;
use crate::highlight::{Word, interest, parse_words};
use crate::llm::ChatModel;
use crate::{AutoCutError, Executor};

/// A pause this long ends a sentence even without punctuation, seconds.
pub const SENTENCE_PAUSE: f64 = 1.2;
/// The longest sentence, in words (run-on speech is cut).
pub const MAX_SENTENCE_WORDS: usize = 45;
/// Most sentences looked at (a few hours of speech).
const MAX_SENTENCES: usize = 20_000;
/// Most candidates scored.
const MAX_CANDIDATES: usize = 50_000;
/// Openings the local model rates, at most (the best by the other scores).
const MAX_RATED: usize = 36;
/// Openings per model request.
const RATE_BATCH: usize = 12;
/// Shorts are this far apart at least, seconds.
const MIN_GAP: f64 = 1.0;
/// Lead-in before a short's first word and tail after its last, at most (from the silence
/// around them), seconds.
const PRE_ROLL: f64 = 0.2;
const POST_ROLL: f64 = 0.45;

/// A sentence of the transcript.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Sentence {
    /// Positions of its first and last word in the word list.
    pub first: usize,
    pub last: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Cut `words` into sentences (see the module docs).
pub fn sentences(words: &[Word]) -> Vec<Sentence> {
    let mut out: Vec<Sentence> = Vec::new();
    let mut first = 0usize;
    for (i, w) in words.iter().enumerate() {
        if out.len() >= MAX_SENTENCES {
            break;
        }
        let ends = w.text.trim_end_matches(['"', '\'', ')', '”', '’']).ends_with(['.', '!', '?', '…']);
        let pause = words.get(i + 1).is_some_and(|n| n.start - w.end > SENTENCE_PAUSE);
        let long = i + 1 - first >= MAX_SENTENCE_WORDS;
        if ends || pause || long || i + 1 == words.len() {
            if let Some(s) = sentence(words, first, i) {
                out.push(s);
            }
            first = i + 1;
        }
    }
    out
}

fn sentence(words: &[Word], first: usize, last: usize) -> Option<Sentence> {
    let span = words.get(first..=last)?;
    let (a, b) = (span.first()?, span.last()?);
    if !(a.start.is_finite() && b.end.is_finite()) || b.end < a.start {
        return None;
    }
    let text = span.iter().map(|w| w.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
    Some(Sentence { first, last, start: a.start, end: b.end, text })
}

/// A run of whole sentences that could be a short.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    /// Its first and last sentence.
    pub from: usize,
    pub to: usize,
    pub start: f64,
    pub end: f64,
    /// The opening sentence.
    pub hook: String,
    pub text: String,
    pub score: f64,
    /// The local model's hook rating (0..10), when it rated this one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rating: Option<f64>,
    /// The local model's title, when it gave one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Candidate {
    pub fn seconds(&self) -> f64 {
        self.end - self.start
    }
}

/// Words that open a hook: a direct address, a question, a bold claim.
const HOOK_OPENERS: &[&str] = &[
    "you",
    "your",
    "you're",
    "why",
    "how",
    "what",
    "when",
    "here's",
    "here",
    "this",
    "these",
    "imagine",
    "never",
    "stop",
    "nobody",
    "everyone",
    "everybody",
    "most",
    "secret",
    "truth",
    "nothing",
    "don't",
    "if",
];

/// How well a sentence opens a short (0..~1): a question, a hook word first, briefly said.
fn hook_score(s: &str) -> f64 {
    let words: Vec<String> = s.split_whitespace().map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>().to_lowercase()).collect();
    let Some(first) = words.first() else { return 0.0 };
    let mut v = 0.0;
    if s.trim_end().ends_with('?') {
        v += 0.4;
    }
    if HOOK_OPENERS.contains(&first.as_str()) || first.chars().any(|c| c.is_ascii_digit()) {
        v += 0.3;
    }
    if (3..=14).contains(&words.len()) {
        v += 0.2;
    }
    v
}

/// Every run of whole sentences between `min` and `max` seconds long, scored (see the module
/// docs), best first.
pub fn candidates(sentences: &[Sentence], min: f64, max: f64, topic: Option<&str>) -> Vec<Candidate> {
    let (min, max) = (min.max(0.5), max.max(min.max(0.5) + 0.5));
    let target = (min + max) / 2.0;
    let interest_of: Vec<f64> = sentences.iter().map(|s| interest(&s.text, topic)).collect();
    let mut out: Vec<Candidate> = Vec::new();
    'all: for (i, a) in sentences.iter().enumerate() {
        let (mut weighted, mut spoken) = (0.0, 0.0);
        for (j, b) in sentences.iter().enumerate().skip(i) {
            let len = b.end - a.start;
            if len > max {
                break;
            }
            let d = (b.end - b.start).max(0.05);
            weighted += interest_of.get(j).copied().unwrap_or(0.0) * d;
            spoken += d;
            if len < min {
                continue;
            }
            if out.len() >= MAX_CANDIDATES {
                break 'all;
            }
            // best near the middle of the range, a little less towards its ends
            let length = 1.0 - 0.7 * ((len - target).abs() / (max - target).max(1e-6)).min(1.0);
            // speech that fills the time: the share of it spoken
            let density = (spoken / len.max(1e-6)).min(1.0);
            let score = weighted / spoken.max(1e-6) + length + 0.5 * density + hook_score(&a.text);
            let text = sentences.get(i..=j).unwrap_or_default().iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
            out.push(Candidate { from: i, to: j, start: a.start, end: b.end, hook: a.text.clone(), text, score, rating: None, title: None });
        }
    }
    out.sort_by(|x, y| y.score.total_cmp(&x.score).then(x.start.total_cmp(&y.start)));
    out
}

/// The best `count` candidates that do not overlap (at least [`MIN_GAP`] apart), best first.
pub fn pick(cands: &[Candidate], count: usize) -> Vec<Candidate> {
    let mut chosen: Vec<Candidate> = Vec::new();
    let mut order: Vec<&Candidate> = cands.iter().collect();
    order.sort_by(|x, y| total(y).total_cmp(&total(x)).then(x.start.total_cmp(&y.start)));
    for c in order {
        if chosen.len() >= count {
            break;
        }
        if chosen.iter().all(|k| c.end + MIN_GAP <= k.start || c.start >= k.end + MIN_GAP) {
            chosen.push(c.clone());
        }
    }
    chosen
}

/// A candidate's score with the model's hook rating (worth up to 1.5).
fn total(c: &Candidate) -> f64 {
    c.score + c.rating.map_or(0.0, |r| 0.15 * r.clamp(0.0, 10.0))
}

const RATE_SYSTEM: &str = "You judge the opening line of short vertical videos (YouTube Shorts, Reels) cut from a longer talk. Rate how strongly each opening makes a viewer keep watching in the first two seconds: 0 (nothing makes them stay) to 10 (impossible to scroll past). Also give each a short, punchy title of at most six words, using only what the opening and the passage say. Answer with JSON only.";

/// Have `model` rate the openings of the best candidates (their `rating` and `title`); a model
/// that fails or answers nonsense leaves them unrated.
pub async fn rate_hooks(model: &dyn ChatModel, cands: &mut [Candidate], notes: &mut Vec<String>) {
    let n = cands.len().min(MAX_RATED);
    for start in (0..n).step_by(RATE_BATCH) {
        let end = (start + RATE_BATCH).min(n);
        let Some(batch) = cands.get(start..end) else { break };
        let listing: Vec<String> = batch
            .iter()
            .enumerate()
            .map(|(k, c)| format!("{}. opening: \"{}\" — passage: \"{}\"", k + 1, clip_text(&c.hook, 200), clip_text(&c.text, 400)))
            .collect();
        let user = format!(
            "Rate these {} openings.\n{}\nAnswer: {{\"ratings\": [{{\"n\": 1, \"hook\": 0-10, \"title\": \"...\"}}, ...]}}",
            batch.len(),
            listing.join("\n")
        );
        let answer = match model.complete_json(RATE_SYSTEM, &user, &[]).await {
            Ok(v) => v,
            Err(e) => {
                notes.push(format!("hook ratings: {e}"));
                return;
            }
        };
        for r in answer["ratings"].as_array().map(Vec::as_slice).unwrap_or_default() {
            let Some(k) = r["n"].as_u64().and_then(|k| usize::try_from(k).ok()).and_then(|k| k.checked_sub(1)) else { continue };
            let Some(c) = cands.get_mut(start + k).filter(|_| start + k < end) else { continue };
            if let Some(h) = r["hook"].as_f64().filter(|h| h.is_finite()) {
                c.rating = Some(h.clamp(0.0, 10.0));
            }
            if let Some(t) = r["title"].as_str().map(tidy_title).filter(|t| !t.is_empty()) {
                c.title = Some(t);
            }
        }
    }
}

/// At most `n` characters of `s`.
fn clip_text(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => s.get(..i).unwrap_or(s),
        None => s,
    }
}

/// A title on one line, without quotes, at most 80 characters.
fn tidy_title(t: &str) -> String {
    let t: String = t.chars().filter(|c| !c.is_control()).collect();
    let t = t.trim().trim_matches(['"', '\'', '“', '”']).trim();
    clip_text(t, 80).trim().to_string()
}

/// A title from the opening when the model gave none: its first seven words, without the
/// sentence's end.
pub fn title_of(c: &Candidate) -> String {
    if let Some(t) = &c.title {
        return t.clone();
    }
    let words: Vec<&str> = c.hook.split_whitespace().take(7).collect();
    let t = words.join(" ");
    let t = t.trim_end_matches(['.', ',', ';', ':', '!']).to_string();
    let t = tidy_title(&t);
    if t.is_empty() { "Short".into() } else { t }
}

/// A file name from a title: lowercase letters, digits and dashes, at most 40 characters.
pub fn slug(t: &str) -> String {
    let mut s = String::new();
    for c in t.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            s.push(c);
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
        if s.len() >= 40 {
            break;
        }
    }
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { "short".into() } else { s }
}

/// The range a short takes from the source, seconds: its words, plus a little lead-in and tail
/// from the silence around them (never into a neighbouring word).
fn range_of(c: &Candidate, words: &[Word], sents: &[Sentence], total: f64) -> (f64, f64) {
    let first = sents.get(c.from).map_or(0, |s| s.first);
    let last = sents.get(c.to).map_or(0, |s| s.last);
    let before = first.checked_sub(1).and_then(|i| words.get(i)).map_or(0.0, |w| (w.end + c.start) / 2.0);
    let after = words.get(last + 1).map_or(total, |w| (c.end + w.start) / 2.0);
    let a = (c.start - PRE_ROLL).max(before).max(0.0);
    let b = (c.end + POST_ROLL).min(after).min(total.max(c.end));
    (a, b.max(a + 0.1))
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
    let step = crate::PlannedStep::new(id, p.clone(), "shorts");
    crate::validate::validate_step(&step).map_err(AutoCutError::Invalid)?;
    ex.execute(id, p).await.map_err(|e| AutoCutError::Executor(format!("{id}: {e}")))
}

/// What one short became.
#[derive(Clone, Debug, Serialize)]
pub struct Made {
    pub number: usize,
    pub sequence: u64,
    pub name: String,
    pub title: String,
    pub start: f64,
    pub end: f64,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rating: Option<f64>,
    pub hook: String,
    pub text: String,
    /// Whether the speaker was followed (else the picture was fitted whole: over its blurred copy
    /// with the `blur` fallback, the default).
    pub followed: bool,
    /// Clips that got the blurred fill under a stretch (or all) of them.
    #[serde(rename = "blurFill")]
    pub blur_fill: u64,
    /// The captions and title against the platform's safe zones: `{platform, ok, overlaps}`.
    #[serde(rename = "safeZones", skip_serializing_if = "Option::is_none")]
    pub safe_zones: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub youtube: Option<Value>,
    pub notes: Vec<String>,
}

/// The source sequence: a new one from media `item`, the given `sequence`, else the active one.
async fn source(ex: &dyn Executor, p: &Value) -> Result<(u64, Value), AutoCutError> {
    let id = if let Some(item) = p["item"].as_u64() {
        let r = run(ex, "file.newSequenceFromClip", json!({"items": [item]})).await?;
        r["sequence"].as_u64().ok_or_else(|| AutoCutError::Other("no sequence was made from the clip".into()))?
    } else if let Some(q) = p["sequence"].as_u64() {
        run(ex, "sequence.open", json!({"item": q})).await?;
        q
    } else {
        let q = run(ex, "sequence.inspect", json!({})).await.map_err(|_| AutoCutError::NoSequence)?;
        q["id"].as_u64().ok_or(AutoCutError::NoSequence)?
    };
    let info = run(ex, "sequence.inspect", json!({"item": id})).await?;
    Ok((id, info))
}

/// Make the shorts (the `autocut.shorts` job; parameters as the engine checked them).
pub async fn make(ex: &dyn Executor, p: &Value, rater: Option<Arc<dyn ChatModel>>, progress: &mut dyn FnMut(&str)) -> Result<Value, AutoCutError> {
    let count = p["count"].as_u64().unwrap_or(3).clamp(1, 20) as usize;
    let min = p["minSeconds"].as_f64().filter(|v| v.is_finite()).unwrap_or(15.0);
    let max = p["maxSeconds"].as_f64().filter(|v| v.is_finite()).unwrap_or(60.0);
    let aspect = p["aspect"].as_str().unwrap_or("9:16").to_string();
    let topic = p["topic"].as_str();
    let mut notes: Vec<String> = Vec::new();
    progress("finding the source");
    let (src, info) = source(ex, p).await?;
    let src_name = info["name"].as_str().unwrap_or("Video").to_string();
    let length = info["duration"].as_i64().map_or(0.0, |t| t as f64 / filmcraft_time::TICKS_PER_SECOND as f64);
    // the words: the transcript, or Whisper's now
    let mut t = run(ex, "transcript.inspect", json!({})).await?;
    if parse_words(&t).is_none() {
        progress("transcribing");
        run(ex, "transcript.generate", json!({})).await.map_err(|e| {
            AutoCutError::Transcript(format!(
                "the video has no transcript and transcribing failed ({e}); transcribe it first (transcript.generate) or import one (transcript.set)"
            ))
        })?;
        t = run(ex, "transcript.inspect", json!({})).await?;
    }
    let words = parse_words(&t).ok_or_else(|| AutoCutError::Transcript("the transcript has no words".into()))?;
    let sents = sentences(&words);
    let mut cands = candidates(&sents, min, max, topic);
    if cands.is_empty() {
        return Err(AutoCutError::Other(format!(
            "no stretch of whole sentences {min:.0}–{max:.0} s long in {} sentence(s) of speech; try other minSeconds / maxSeconds",
            sents.len()
        )));
    }
    progress(&format!("{} sentence(s), {} candidate short(s)", sents.len(), cands.len()));
    if let Some(m) = &rater {
        // the model rates the openings of the best candidates that do not overlap
        let mut first = pick(&cands, MAX_RATED);
        progress(&format!("rating the openings of the best {} with {}", first.len(), m.id()));
        rate_hooks(m.as_ref(), &mut first, &mut notes).await;
        for r in first {
            if let Some(c) = cands.iter_mut().find(|c| c.from == r.from && c.to == r.to) {
                c.rating = r.rating;
                c.title = r.title;
            }
        }
    }
    let chosen = pick(&cands, count);
    let n = chosen.len();
    let export_dir = p["export"].as_str().map(std::path::PathBuf::from);
    if let Some(d) = &export_dir {
        std::fs::create_dir_all(d).map_err(|e| AutoCutError::Other(format!("export folder {}: {e}", d.display())))?;
    }
    let mut made: Vec<Made> = Vec::new();
    for (k, c) in chosen.iter().enumerate() {
        let number = k + 1;
        let title = title_of(c);
        let name = format!("{src_name} — Short {number}: {}", clip_text(&title, 60));
        progress(&format!("[{number}/{n}] reframing \"{title}\""));
        let (a, b) = range_of(c, &words, &sents, length);
        // whose interface the captions and the title keep clear of
        let platform = p["platform"].as_str().and_then(Platform::from_name).unwrap_or(Platform::Youtube);
        // nobody to follow: the whole picture over a blurred copy of itself (not black bars)
        let fallback = p["fallback"].as_str().unwrap_or("blur");
        let mut rp =
            json!({"sequence": src, "aspect": aspect, "range": {"startSeconds": a, "endSeconds": b}, "name": name, "fallback": fallback, "wait": true});
        if let Some(m) = p["motion"].as_str() {
            rp["motion"] = json!(m);
        }
        if let Some(f) = p["analysisFps"].as_f64() {
            rp["analysisFps"] = json!(f);
        }
        let r = run(ex, "sequence.autoReframe", rp).await?;
        let seq = r["reframe"]["sequence"].as_u64().ok_or_else(|| AutoCutError::Executor(format!("sequence.autoReframe made no sequence: {r}")))?;
        let followed = r["reframe"]["followed"].as_u64().is_some_and(|f| f > 0);
        let blur_fill = r["reframe"]["blurFill"].as_u64().unwrap_or(0);
        let mut short = Made {
            number,
            sequence: seq,
            name: name.clone(),
            title: title.clone(),
            start: a,
            end: b,
            score: (total(c) * 1000.0).round() / 1000.0,
            rating: c.rating,
            hook: c.hook.clone(),
            text: c.text.clone(),
            followed,
            blur_fill,
            safe_zones: None,
            export: None,
            youtube: None,
            notes: Vec::new(),
        };
        // the reframed copy is the active sequence now
        let frame = run(ex, "sequence.inspect", json!({"item": seq})).await.ok();
        let size = |k: &str, or: f64| frame.as_ref().and_then(|f| f["settings"][k].as_f64()).filter(|v| v.is_finite() && *v >= 16.0).unwrap_or(or);
        let (fw, fh) = (size("width", 1080.0), size("height", 1920.0));
        if p["captions"].as_bool() != Some(false) {
            progress(&format!("[{number}/{n}] captions"));
            let look = caption_style::Look::of(p);
            // the local model's keywords and emoji, when asked for (off by default)
            let extra = match (&rater, p["captionKeywords"].as_bool() == Some(true), &look) {
                (Some(m), true, caption_style::Look::Animated(_)) => Some(caption_style::keywords(m.as_ref(), &c.text, &mut short.notes).await),
                _ => None,
            };
            if let Err(e) = captions(ex, platform, &look, extra.as_ref()).await {
                short.notes.push(format!("captions: {e}"));
            }
        }
        if p["title"].as_bool() == Some(true)
            && let Err(e) = title_card(ex, platform, &title, (fw, fh)).await
        {
            short.notes.push(format!("title card: {e}"));
        }
        // what still reaches into the platform's interface is said, not hidden
        match run(ex, "sequence.checkSafeZones", json!({"platform": platform.name(), "sequence": seq})).await {
            Ok(v) => {
                let overlaps = v["overlaps"].as_array().map(Vec::as_slice).unwrap_or_default();
                for o in overlaps.iter().take(20) {
                    short.notes.push(safe_zone_note(o, platform));
                }
                short.safe_zones = Some(json!({"platform": platform.name(), "ok": overlaps.is_empty(), "overlaps": overlaps.len()}));
            }
            Err(e) => short.notes.push(format!("safe zones: {e}")),
        }
        if let Some(dir) = &export_dir {
            let path = dir.join(format!("{:02}-{}.mp4", number, slug(&title)));
            let path_s = path.to_string_lossy().to_string();
            progress(&format!("[{number}/{n}] exporting {}", path.display()));
            let ep = json!({"sequence": seq, "path": path_s, "format": "h264", "burnCaptions": true, "wait": true});
            match run(ex, "file.exportMedia", ep).await {
                Ok(r) => short.export = Some(r["path"].as_str().unwrap_or(&path_s).to_string()),
                Err(e) => short.notes.push(format!("export: {e}")),
            }
        }
        if p["uploadYoutube"].as_bool() == Some(true)
            && let Some(file) = short.export.clone()
        {
            progress(&format!("[{number}/{n}] uploading to YouTube"));
            let mut up = json!({"path": file, "title": clip_text(&title, 100), "privacy": "private", "description": clip_text(&c.text, 4500), "wait": true});
            if let Some(o) = p["youtube"].as_object() {
                for (key, v) in o {
                    up[key] = v.clone();
                }
            }
            // asked for in so many words (`uploadYoutube`): not a command AutoCut's planners may pick
            match ex.execute("youtube.upload", up).await.map_err(|e| AutoCutError::Executor(format!("youtube.upload: {e}"))) {
                Ok(r) => short.youtube = Some(r.get("youtube").cloned().unwrap_or(r)),
                Err(e) => short.notes.push(format!("YouTube: {e}")),
            }
        }
        made.push(short);
    }
    let exported = made.iter().filter(|m| m.export.is_some()).count();
    let status = match (&export_dir, n) {
        (_, 0) => "No shorts made".to_string(),
        (Some(_), _) => format!("Made {n} short(s) from \"{src_name}\", {exported} exported"),
        (None, _) => format!("Made {n} short(s) from \"{src_name}\""),
    };
    Ok(json!({
        "source": src,
        "sentences": sents.len(),
        "candidates": cands.len(),
        "rated": rater.as_ref().map(|m| m.id()),
        "shorts": made,
        "notes": notes,
        "status": status,
    }))
}

/// The captions' size (pixels at 1080 lines) to start from, and the smallest they shrink to so
/// that they stay clear of a platform's side zones.
pub const CAPTION_SIZE: f64 = 76.0;
pub const MIN_CAPTION_SIZE: f64 = 44.0;
/// Room between the captions and the platform's bottom zone, a share of the frame's height.
pub use crate::caption_style::CAPTION_GAP;

/// Captions a few words at a time from the active sequence's transcript (word-timed), in `look`
/// (Bold animated by default, keeping to `platform`'s safe zone and centred on its safe span;
/// `plain`: big outlined captions just above its bottom zone, centred on its safe span and
/// shrunk until they fit it, [`fit_captions`]); `extra`: keywords and emoji from the model.
async fn captions(ex: &dyn Executor, platform: Platform, look: &caption_style::Look, extra: Option<&caption_style::Highlights>) -> Result<(), AutoCutError> {
    let r = run(ex, "transcript.createCaptions", json!({"maxChars": 18, "lines": 1, "minSeconds": 0.5, "maxSeconds": 2.2, "gapFrames": 0, "name": "Captions"}))
        .await?;
    let track = r["track"].as_u64().ok_or_else(|| AutoCutError::Executor("transcript.createCaptions made no track".into()))?;
    caption_style::style_track(ex, track, platform, look, extra).await?;
    match look {
        caption_style::Look::Plain => fit_captions(ex, track, platform, CAPTION_SIZE).await,
        // animated captions keep to the safe span themselves: a word too wide gets a smaller page
        caption_style::Look::Animated(_) => Ok(()),
    }
}

/// How much smaller (a factor, at most 1) the widest caption in `check` (a
/// `sequence.checkSafeZones` answer) must be to fit between `platform`'s side zones at its height,
/// centred `x_offset` (a share of the frame width) from the frame's middle — on the middle of the
/// safe span with [`caption_style::safe_offset`], so it may be as wide as the span; 1 when no
/// caption reaches into them.
pub fn caption_shrink(check: &Value, platform: Platform, x_offset: f64) -> f64 {
    let frame = |k: usize, or: f64| check["frame"][k].as_f64().filter(|v| v.is_finite() && *v >= 1.0).unwrap_or(or);
    let (fw, fh) = (frame(0, 1080.0), frame(1, 1920.0));
    let centre = 0.5 + if x_offset.is_finite() { x_offset.clamp(-0.5, 0.5) } else { 0.0 };
    let mut need: f64 = 1.0;
    for o in check["overlaps"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter(|o| o["kind"] == "caption").take(10_000) {
        let r: Vec<f64> = o["rect"].as_array().map(|a| a.iter().filter_map(Value::as_f64).filter(|v| v.is_finite()).collect()).unwrap_or_default();
        let [x0, y0, x1, y1] = r.as_slice() else { continue };
        let (l, right) = platform.safe_span(y0 / fh, y1 / fh);
        let allowed = 2.0 * (centre - l).min(right - centre) * fw;
        let width = x1 - x0;
        if width > 0.0 && allowed > 0.0 {
            need = need.min(allowed / width);
        }
    }
    need.clamp(0.0, 1.0)
}

/// Shrink the captions of `track` (now at `size`) until none reaches into `platform`'s side zones,
/// down to [`MIN_CAPTION_SIZE`]; a few rounds at most.
async fn fit_captions(ex: &dyn Executor, track: u64, platform: Platform, mut size: f64) -> Result<(), AutoCutError> {
    for _ in 0..4 {
        let check = run(ex, "sequence.checkSafeZones", json!({"platform": platform.name()})).await?;
        let need = caption_shrink(&check, platform, caption_style::safe_offset(platform));
        let next = (size * need * 0.97).floor().max(MIN_CAPTION_SIZE);
        if need >= 1.0 || next >= size {
            break;
        }
        size = next;
        run(ex, "captions.setStyle", json!({"track": track, "size": size})).await?;
    }
    Ok(())
}

/// Where the title card goes: a text box across `platform`'s safe area, just under its top zone
/// (`[x, y]`, `[w, h]` in the pixels of a `w`×`h` frame).
pub fn title_box(platform: Platform, (fw, fh): (f64, f64)) -> ([f64; 2], [f64; 2]) {
    let safe = platform.safe_area();
    let pad = 0.02;
    let x = (safe.x0 + pad) * fw;
    let w = ((safe.width() - 2.0 * pad) * fw).max(1.0);
    ([x.round(), ((safe.y0 + pad) * fh).round()], [w.round(), (0.16 * fh).round()])
}

/// The title card: bold paragraph text centred in [`title_box`].
async fn title_card(ex: &dyn Executor, platform: Platform, title: &str, (fw, fh): (f64, f64)) -> Result<(), AutoCutError> {
    let (position, size) = title_box(platform, (fw, fh));
    let tp = json!({"text": title, "seconds": 2.5, "time": 0, "size": (fw.min(fh) * 0.075).round(), "position": position, "box": size});
    let r = run(ex, "graphics.newText", tp).await?;
    if let Some(clip) = r["clip"].as_u64() {
        // cosmetic: a title that keeps the default alignment and weight is still a title
        run(ex, "graphics.set", json!({"clip": clip, "props": {"align": "center", "font_style": "Bold"}})).await.ok();
    }
    Ok(())
}

/// A short's note for one overlap of a `sequence.checkSafeZones` answer.
fn safe_zone_note(o: &Value, platform: Platform) -> String {
    let kind = if o["kind"] == "caption" { "caption" } else { "title" };
    let text: String = o["text"].as_str().unwrap_or_default().chars().take(60).collect();
    let zones = o["zones"].as_array().map(|z| z.iter().filter_map(Value::as_str).take(5).collect::<Vec<_>>().join(" and ")).unwrap_or_default();
    let at = o["seconds"].as_f64().filter(|s| s.is_finite()).unwrap_or(0.0);
    format!("safe zones: the {kind} \"{text}\" at {at:.1} s reaches into {}'s {zones} zone", platform.label())
}

#[cfg(test)]
#[path = "shorts_tests.rs"]
mod tests;

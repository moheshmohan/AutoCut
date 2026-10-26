//! Animated captions and text effects.
//!
//! | command | does |
//! |---|---|
//! | `captions.setAnimatedStyle` | word-by-word animated captions on a caption track: `preset` `bold` (uppercase heavy sans, black stroke, 2–4 words a page, the spoken word in the active colour popping 1.1×), `clean`, `glow`, `box`, `karaoke`, `twoLine` (classic two-line subtitles), or `off` for whole captions again; `options`: `wordsPerPage`, `activeColor`, `keywordColor`, `keywords`, `emoji` (`{"word": "🔥"}`), `uppercase`, `platform` (keep to `youtube` / `reels` / `tiktok` / `vertical`'s safe zone, centred on its safe span), `pop`, `position` (`top` / `middle` / `bottom`), `margin`, `xOffset`, `size`, `font`, `panel` (a backing panel behind each line: `true` for black at 60 %, a colour `#rrggbbaa`, or `false`), `keepStyle` (keep the track's size, outline and box instead of the preset's) |
//! | `captions.wordTiming` | when each word of the track's captions is spoken, `from` the sequence `transcript` (Whisper's word timestamps) or the `narration` clips (their scripts spread over each clip's speech by the words' lengths, pauses kept) |
//! | `graphics.animateText` | a text layer's animators: `in`, `out`, `loop` — each `{kind, unit, amount, stagger, easing, duration}` (or a kind name; `null` / `"none"` removes it); `clear` removes all |
//! | `graphics.textStyle` | a layer's `glow` (`{color, radius, intensity}`), `innerShadow` (`{color, opacity, angle, distance, blur}`), `curve` (`{bend, radius}` or the bend in percent) and `stroke3` (`{color, width, type}`); `false` turns one off |
//!
//! The data lives in the project ([`filmcraft_project::AnimatedCaptions`] on the track style,
//! [`filmcraft_project::Caption::words`], [`filmcraft_project::text_anim`] on the layer, the
//! layer's appearance parameters) and is drawn by `filmcraft_captions::animated` and
//! `filmcraft_render::text_units` in the preview and the export alike.

use serde_json::{Map, Value, json};

use filmcraft_project::gltransition::EASING_OPTS;
use filmcraft_project::graphic::{self, TEXT_LAYER};
use filmcraft_project::text_anim::{AnimKind, AnimUnit, Animator, TextAnimation};
use filmcraft_project::{AnimatedCaptions, CaptionAnchor, CaptionPreset, CaptionStyle, WordEmoji};
use filmcraft_time::Tick;

use crate::captions::track_param;
use crate::commands::{CommandSpec, bad, bool_p, has_seq, str_p};
use crate::graphics::{layer_effect_index, set_props, target_clip};
use crate::safe_zones::platform_p;
use crate::{EngineError, Result, Session};

type Run = fn(&mut Session, &Value) -> Result<Value>;
type Enabled = fn(&Session) -> std::result::Result<(), String>;

fn spec(id: &'static str, label: &'static str, params: &'static str, enabled: Enabled, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal: true }
}

fn has_caption_track(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    if s.active_sequence().is_some_and(|q| !q.caption_tracks.is_empty()) { Ok(()) } else { Err("the sequence has no caption track".into()) }
}

fn has_graphic(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    target_clip(s, &Value::Null).map(|_| ()).ok_or_else(|| crate::graphics::no_graphic_why(s))
}

/// Most keywords and emoji kept, and their longest text.
const MAX_LIST: usize = 200;
const MAX_WORD_CHARS: usize = 64;

/// A colour: `"#rrggbb[aa]"` or `[r, g, b(, a)]` (0–255).
/// The backing panel `"panel": true` puts behind animated captions: black at 60 %.
const PANEL: [u8; 4] = [0, 0, 0, 153];

pub(crate) fn color_p(v: &Value) -> Option<[u8; 4]> {
    if let Some(a) = v.as_array() {
        let c: Vec<u8> = a.iter().filter_map(|x| x.as_f64().filter(|f| f.is_finite()).map(|n| n.clamp(0.0, 255.0) as u8)).collect();
        return match c.as_slice() {
            [r, g, b] => Some([*r, *g, *b, 255]),
            [r, g, b, a] => Some([*r, *g, *b, *a]),
            _ => None,
        };
    }
    let h = v.as_str()?.trim().trim_start_matches('#');
    let b = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    match h.len() {
        6 => Some([b(0)?, b(2)?, b(4)?, 255]),
        8 => Some([b(0)?, b(2)?, b(4)?, b(6)?]),
        _ => None,
    }
}

fn hex(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3])
}

/// A finite number of `o` at `k`, clamped; an error when present but not a number.
fn num(cmd: &str, o: &Value, k: &str, lo: f64, hi: f64) -> Result<Option<f64>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().filter(|f| f.is_finite()).map(|f| Some(f.clamp(lo, hi))).ok_or_else(|| bad(cmd, format!("`{k}` must be a number"))),
    }
}

fn short(s: &str) -> String {
    s.trim().chars().take(MAX_WORD_CHARS).collect()
}

/// The animated style as JSON (for answers and the UI).
pub fn animated_json(a: &AnimatedCaptions) -> Value {
    json!({
        "preset": a.preset.name(),
        "label": a.preset.label(),
        "wordsPerPage": a.page_words(),
        "activeColor": hex(a.active_color),
        "keywordColor": hex(a.keyword_color),
        "keywords": a.keywords,
        "emoji": a.emoji.iter().map(|e| json!({"word": e.word, "emoji": e.emoji})).collect::<Vec<_>>(),
        "uppercase": a.is_uppercase(),
        "platform": a.platform.map(|p| p.name()),
        "pop": a.pop_scale(),
    })
}

fn set_animated_style(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "captions.setAnimatedStyle";
    let tid = track_param(s, p).ok_or_else(|| bad(CMD, "no such caption track (pass `track`: an id from captions.list, or \"C1\"…)"))?;
    let preset_name = str_p(p, "preset").ok_or_else(|| bad(CMD, "need `preset`: bold, clean, glow, box, karaoke, twoLine or off"))?;
    let off = matches!(preset_name.trim().to_ascii_lowercase().as_str(), "off" | "none" | "plain" | "static");
    let preset = if off {
        None
    } else {
        Some(
            CaptionPreset::from_name(preset_name)
                .ok_or_else(|| bad(CMD, format!("unknown preset `{preset_name}`: bold, clean, glow, box, karaoke, twoLine or off")))?,
        )
    };
    // options in `options`, or beside `preset`
    let o = match p.get("options") {
        Some(v @ Value::Object(_)) => v.clone(),
        Some(Value::Null) | None => p.clone(),
        Some(_) => return Err(bad(CMD, "`options` must be an object")),
    };
    let current = s.active_sequence().and_then(|q| q.caption_track(tid)).map(|t| t.style.clone()).ok_or(EngineError::NoSequence)?;
    let mut style = current.clone();
    match preset {
        None => style.animated = None,
        Some(pr) => {
            // a new preset starts from its look; the same preset keeps the options set before
            let mut a = match &current.animated {
                Some(a) if a.preset == pr => a.clone(),
                Some(a) => AnimatedCaptions { keywords: a.keywords.clone(), emoji: a.emoji.clone(), platform: a.platform, ..AnimatedCaptions::preset(pr) },
                None => AnimatedCaptions::preset(pr),
            };
            let keep = bool_p(&o, "keepStyle").unwrap_or(false);
            if !keep && current.animated.as_ref().is_none_or(|x| x.preset != pr) {
                let base = pr.base_style();
                style = CaptionStyle {
                    font: current.font.clone(),
                    color: current.color,
                    x_offset: current.x_offset,
                    background_color: current.background_color,
                    align: current.align,
                    ..base
                };
            }
            if let Some(n) = num(CMD, &o, "wordsPerPage", 0.0, 8.0)? {
                a.words_per_page = n.round() as u32;
            }
            if let Some(v) = o.get("activeColor") {
                a.active_color = color_p(v).ok_or_else(|| bad(CMD, "`activeColor`: \"#rrggbb\" or [r, g, b]"))?;
            }
            if let Some(v) = o.get("keywordColor") {
                a.keyword_color = color_p(v).ok_or_else(|| bad(CMD, "`keywordColor`: \"#rrggbb\" or [r, g, b]"))?;
            }
            if let Some(v) = o.get("keywords") {
                a.keywords = match v {
                    Value::Null => Vec::new(),
                    Value::String(s) => s.split(',').map(short).filter(|w| !w.is_empty()).take(MAX_LIST).collect(),
                    Value::Array(xs) => xs.iter().filter_map(Value::as_str).map(short).filter(|w| !w.is_empty()).take(MAX_LIST).collect(),
                    _ => return Err(bad(CMD, "`keywords`: a list of words")),
                };
            }
            if let Some(v) = o.get("emoji") {
                a.emoji = emoji_p(v).ok_or_else(|| bad(CMD, "`emoji`: {\"word\": \"🔥\"} or [{\"word\", \"emoji\"}]"))?;
            }
            match o.get("uppercase") {
                Some(Value::Bool(b)) => a.uppercase = Some(*b),
                Some(Value::Null) => a.uppercase = None,
                Some(_) => return Err(bad(CMD, "`uppercase` must be true, false or null")),
                None => {}
            }
            if let Some(pl) = platform_p(CMD, &o, "platform")? {
                a.platform = pl;
            }
            if let Some(v) = num(CMD, &o, "pop", 1.0, 1.5)? {
                a.pop = v as f32;
            }
            style.animated = Some(a);
        }
    }
    match str_p(&o, "position").map(str::to_ascii_lowercase).as_deref() {
        Some("top") => style.anchor = CaptionAnchor::Top,
        Some("middle" | "center" | "centre") => style.anchor = CaptionAnchor::Middle,
        Some("bottom") => style.anchor = CaptionAnchor::Bottom,
        Some(other) => return Err(bad(CMD, format!("`position` is top, middle or bottom, not `{other}`"))),
        None => {}
    }
    if let Some(v) = num(CMD, &o, "margin", 0.0, 0.45)? {
        style.margin = v as f32;
    }
    if let Some(v) = num(CMD, &o, "xOffset", -0.5, 0.5)? {
        style.x_offset = v as f32;
    }
    if let Some(v) = num(CMD, &o, "size", 4.0, 400.0)? {
        style.size = v as f32;
    }
    if let Some(v) = num(CMD, &o, "outline", 0.0, 40.0)? {
        style.outline = v as f32;
    }
    if let Some(f) = str_p(&o, "font") {
        style.font = short(f);
    }
    // a backing panel behind each line, so the words read over bright footage (live: white
    // "box" captions over a white toy car)
    match o.get("panel") {
        None => {}
        Some(Value::Null | Value::Bool(false)) => style.background = false,
        Some(Value::Bool(true)) => {
            style.background = true;
            style.background_color = PANEL;
        }
        Some(v) => {
            let c = color_p(v).ok_or_else(|| bad(CMD, "`panel`: true, false or a colour (\"#rrggbbaa\")"))?;
            style.background = c[3] > 0;
            style.background_color = c;
        }
    }
    let panel = style.background.then(|| hex(style.background_color));
    let answer = json!({
        "track": tid.0,
        "animated": style.animated.as_ref().map(animated_json),
        "panel": panel,
        "size": style.size,
        "outline": style.outline,
        "anchor": format!("{:?}", style.anchor).to_lowercase(),
        "margin": style.margin,
        "xOffset": style.x_offset,
    });
    s.edit("Animated Caption Style", move |pr, st| {
        let seq = st.active_sequence.ok_or(EngineError::NoSequence)?;
        let t = pr.sequence_mut(seq).and_then(|q| q.caption_track_mut(tid)).ok_or(EngineError::NoSequence)?;
        t.style = style;
        Ok(())
    })?;
    Ok(answer)
}

fn emoji_p(v: &Value) -> Option<Vec<WordEmoji>> {
    let pair = |w: &str, e: &str| WordEmoji { word: short(w), emoji: e.trim().chars().take(8).collect() };
    let v = match v {
        Value::Null => Vec::new(),
        Value::Object(m) => m.iter().filter_map(|(w, e)| e.as_str().map(|e| pair(w, e))).collect(),
        Value::Array(xs) => xs.iter().filter_map(|x| Some(pair(x.get("word")?.as_str()?, x.get("emoji")?.as_str()?))).collect(),
        _ => return None,
    };
    Some(v.into_iter().filter(|e| !e.word.is_empty() && !e.emoji.is_empty()).take(MAX_LIST).collect())
}

/// Words of the narration clips on the active sequence's audio tracks, timed: each clip's script
/// (pause markers kept as silences) spread over its speech by the words' lengths (characters,
/// plus a beat after a sentence or a comma), in sequence time; words outside the clip are left
/// out. An estimate: the voices give no word timing (`transcript` from a transcription of the
/// narration is exact).
pub fn narration_words(s: &Session) -> Vec<(String, Tick, Tick)> {
    let Some(q) = s.active_sequence() else { return Vec::new() };
    let mut out = Vec::new();
    for tr in q.audio_tracks.iter().filter(|t| !t.muted) {
        for it in tr.items.iter().filter(|i| i.enabled) {
            let Some(n) = s.project.narrations.get(&it.item) else { continue };
            let speech = n.speech_duration.seconds();
            if !(speech.is_finite() && speech > 0.0) {
                continue;
            }
            // (word, weight) and pauses (seconds)
            enum Bit {
                Word(String, f64),
                Pause(f64),
            }
            let mut bits: Vec<Bit> = Vec::new();
            for seg in filmcraft_tts::script::parse(&n.text) {
                match seg {
                    filmcraft_tts::script::Segment::Pause { millis } => bits.push(Bit::Pause(f64::from(millis) / 1000.0)),
                    filmcraft_tts::script::Segment::Speech(t) => {
                        for w in t.split_whitespace().take(20_000) {
                            let beat = if w.ends_with(['.', '!', '?', '…']) {
                                4.0
                            } else if w.ends_with([',', ';', ':']) {
                                2.0
                            } else {
                                0.0
                            };
                            bits.push(Bit::Word(w.to_string(), w.chars().count() as f64 + 1.0 + beat));
                        }
                    }
                }
            }
            let pauses: f64 = bits.iter().map(|b| if let Bit::Pause(p) = b { *p } else { 0.0 }).sum();
            let weight: f64 = bits.iter().map(|b| if let Bit::Word(_, w) = b { *w } else { 0.0 }).sum();
            let talk = (speech - pauses).max(0.0);
            if weight <= 0.0 || talk <= 0.0 {
                continue;
            }
            let per = talk / weight;
            let speed = if it.speed.is_finite() && it.speed.abs() > 1e-6 { it.speed.abs() } else { 1.0 };
            let mut m = 0.0f64; // media seconds from the start of the speech
            for b in bits {
                match b {
                    Bit::Pause(p) => m += p,
                    Bit::Word(w, wgt) => {
                        let dur = wgt * per;
                        // media time → sequence time through the clip's placement
                        let at = |media: f64| Tick(it.start.0.saturating_add(Tick::from_seconds_f64((media - it.source_in.seconds()) / speed).0));
                        let (a, e) = (at(m), at(m + dur * 0.85));
                        m += dur;
                        if e > it.start && a < it.end() {
                            out.push((w, a.max(it.start), e.min(it.end())));
                        }
                    }
                }
            }
        }
    }
    out.sort_by_key(|w| w.1);
    out
}

fn word_timing(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "captions.wordTiming";
    let tid = track_param(s, p).ok_or_else(|| bad(CMD, "no such caption track"))?;
    let from = str_p(p, "from").unwrap_or("transcript").to_ascii_lowercase();
    let words: Vec<(String, Tick, Tick)> = match from.as_str() {
        "transcript" => crate::transcript::sequence_words(s).into_iter().map(|w| (w.text, w.start, w.end)).collect(),
        "narration" | "tts" => narration_words(s),
        other => return Err(bad(CMD, format!("`from` is transcript or narration, not `{other}`"))),
    };
    if words.is_empty() {
        return Err(bad(
            CMD,
            if from == "transcript" {
                "the sequence has no transcript words: transcribe its clips first (transcript.generate)"
            } else {
                "the sequence has no narration clips with a script (tts.create)"
            },
        ));
    }
    let only: Option<Vec<u64>> = p.get("captions").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect());
    let n_words = words.len();
    let timed = s.edit_sequence("Caption Word Timing", move |q, _, _| {
        let t = q.caption_track_mut(tid).ok_or(EngineError::NoSequence)?;
        if t.locked {
            return Err(bad(CMD, "the caption track is locked"));
        }
        let mut n = 0;
        match &only {
            Some(ids) => {
                for c in t.captions.iter_mut().filter(|c| ids.contains(&c.id.0)) {
                    n += filmcraft_edit::transcript::time_captions_from_words(std::slice::from_mut(c), &words);
                }
            }
            None => n = filmcraft_edit::transcript::time_captions_from_words(&mut t.captions, &words),
        }
        Ok(n)
    })?;
    Ok(json!({"track": tid.0, "from": from, "captions": timed, "words": n_words}))
}

/// An easing by index or name (`"Back Out"`, `"back-out"`, `"backOut"`).
fn easing_p(v: &Value) -> Option<u32> {
    if let Some(n) = v.as_u64() {
        return (n < EASING_OPTS.len() as u64).then_some(n as u32);
    }
    let k: String = v.as_str()?.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect();
    EASING_OPTS.iter().position(|o| o.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>() == k).map(|i| i as u32)
}

/// One animator slot: a kind name, an object, or null / "none" (remove: `Ok(None)`).
fn animator_p(cmd: &str, slot: &str, v: &Value) -> Result<Option<Animator>> {
    let loop_slot = slot == "loop";
    let obj = match v {
        Value::Null | Value::Bool(false) => return Ok(None),
        Value::String(s) if matches!(s.trim().to_ascii_lowercase().as_str(), "none" | "off" | "") => return Ok(None),
        Value::String(s) => json!({"kind": s}),
        Value::Object(_) => v.clone(),
        _ => return Err(bad(cmd, format!("`{slot}`: a kind name or {{kind, unit, amount, stagger, easing, duration}}"))),
    };
    let kname = str_p(&obj, "kind").ok_or_else(|| bad(cmd, format!("`{slot}` needs a `kind`")))?;
    let kind = AnimKind::from_name(kname).ok_or_else(|| bad(cmd, format!("`{slot}`: unknown kind `{kname}`")))?;
    if kind.is_loop() != loop_slot {
        let names: Vec<&str> = if loop_slot { AnimKind::LOOP.iter().map(|k| k.name()).collect() } else { AnimKind::IN.iter().map(|k| k.name()).collect() };
        return Err(bad(cmd, format!("`{slot}` takes {}, not `{kname}`", names.join(", "))));
    }
    let mut a = Animator::preset(kind);
    if let Some(u) = str_p(&obj, "unit") {
        a.unit = AnimUnit::from_name(u).ok_or_else(|| bad(cmd, format!("`{slot}.unit` is letter, word or line, not `{u}`")))?;
    }
    if let Some(v) = num(cmd, &obj, "amount", 0.0, 1000.0)? {
        a.amount = v;
    }
    if let Some(v) = num(cmd, &obj, "stagger", 0.0, 60.0)? {
        a.stagger = v;
    }
    if let Some(v) = num(cmd, &obj, "duration", 0.0, 60.0)? {
        a.duration = v;
    }
    if let Some(v) = obj.get("easing") {
        a.easing = easing_p(v).ok_or_else(|| bad(cmd, format!("`{slot}.easing`: an index or a name of {}", EASING_OPTS.join(", "))))?;
    }
    if let Some(b) = obj.get("bezier").and_then(Value::as_array) {
        let v: Vec<f64> = b.iter().filter_map(Value::as_f64).filter(|f| f.is_finite()).collect();
        if let [x1, y1, x2, y2] = v.as_slice() {
            a.bezier = Some([*x1, *y1, *x2, *y2]);
        }
    }
    Ok(Some(a.sanitized()))
}

pub(crate) fn animation_json(a: &TextAnimation) -> Value {
    let one = |x: &Option<Animator>| {
        x.as_ref().map(|a| {
            json!({
                "kind": a.kind.name(),
                "label": a.kind.label(),
                "unit": a.unit.name(),
                "amount": a.amount,
                "stagger": a.stagger,
                "easing": EASING_OPTS.get(a.easing as usize).copied().unwrap_or("Linear"),
                "duration": a.duration,
            })
        })
    };
    json!({"in": one(&a.enter), "out": one(&a.out), "loop": one(&a.repeat)})
}

fn animate_text(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "graphics.animateText";
    let clip = target_clip(s, p).ok_or_else(|| bad(CMD, "no graphic clip"))?;
    let (l, ei) = layer_effect_index(s, clip, p)?;
    let clear = bool_p(p, "clear").unwrap_or(false);
    let mut slots: Vec<(&str, Option<Animator>)> = Vec::new();
    for slot in ["in", "out", "loop"] {
        if let Some(v) = p.get(slot) {
            slots.push((slot, animator_p(CMD, slot, v)?));
        }
    }
    // a whole animation by name: a text preset's animators (`graphics.animateText {"preset": "pop"}`)
    if let Some(name) = str_p(p, "preset") {
        let kind = AnimKind::from_name(name).ok_or_else(|| bad(CMD, format!("unknown preset `{name}`")))?;
        slots.push((if kind.is_loop() { "loop" } else { "in" }, Some(Animator::preset(kind))));
    }
    if slots.is_empty() && !clear {
        return Err(bad(CMD, "need `in`, `out`, `loop`, `preset` or `clear`"));
    }
    // `merge`: a drag of an animator's number is one undo step; `begin` starts a new one
    let merge = bool_p(p, "merge").unwrap_or(false).then(|| format!("{CMD}:{}:{ei}", clip.0));
    if bool_p(p, "begin").unwrap_or(false) {
        s.history.merge_key = None;
    }
    let anim = s.edit_sequence_as("Animate Text", merge.as_deref(), move |q, _, _| {
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let e = it.effects.get_mut(ei).ok_or_else(|| bad(CMD, "no such layer"))?;
        if e.effect != TEXT_LAYER {
            return Err(bad(CMD, "the layer is not a text layer"));
        }
        let x = e.layer.get_or_insert_with(Default::default);
        let mut a = if clear { TextAnimation::default() } else { x.anim.clone().unwrap_or_default() };
        for (slot, v) in slots {
            match slot {
                "in" => a.enter = v,
                "out" => a.out = v,
                _ => a.repeat = v,
            }
        }
        let answer = animation_json(&a);
        x.anim = (!a.is_empty()).then_some(a);
        if x.is_empty() {
            e.layer = None;
        }
        Ok(answer)
    })?;
    Ok(json!({"clip": clip.0, "layer": l, "animation": anim}))
}

/// `graphics.textStyle` → `graphics.set` properties.
fn text_style_props(p: &Value) -> Result<Map<String, Value>> {
    const CMD: &str = "graphics.textStyle";
    let mut m = Map::new();
    let color = |o: &Value, k: &str| -> Result<Option<Value>> {
        match o.get(k) {
            None => Ok(None),
            Some(v) => color_p(v).map(|c| Some(json!(hex(c)))).ok_or_else(|| bad(CMD, format!("`{k}`: \"#rrggbb\" or [r, g, b]"))),
        }
    };
    if let Some(g) = p.get("glow") {
        m.insert("glow".into(), json!(!matches!(g, Value::Bool(false) | Value::Null)));
        if g.is_object() {
            if let Some(c) = color(g, "color")? {
                m.insert("glow_color".into(), c);
            }
            if let Some(v) = num(CMD, g, "radius", 0.0, 1000.0)? {
                m.insert("glow_radius".into(), json!(v));
            }
            if let Some(v) = num(CMD, g, "intensity", 0.0, 1000.0)? {
                m.insert("glow_intensity".into(), json!(v));
            }
        }
    }
    if let Some(g) = p.get("innerShadow") {
        m.insert("inner_shadow".into(), json!(!matches!(g, Value::Bool(false) | Value::Null)));
        if g.is_object() {
            if let Some(c) = color(g, "color")? {
                m.insert("inner_shadow_color".into(), c);
            }
            for (k, id, hi) in [
                ("opacity", "inner_shadow_opacity", 100.0),
                ("angle", "inner_shadow_angle", 36_000.0),
                ("distance", "inner_shadow_distance", 1000.0),
                ("blur", "inner_shadow_blur", 1000.0),
            ] {
                if let Some(v) = num(CMD, g, k, if k == "angle" { -36_000.0 } else { 0.0 }, hi)? {
                    m.insert(id.into(), json!(v));
                }
            }
        }
    }
    if let Some(c) = p.get("curve") {
        match c {
            Value::Bool(false) | Value::Null => {
                m.insert("curve".into(), json!(0.0));
            }
            Value::Number(n) => {
                let b = n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| bad(CMD, "`curve` must be a number"))?;
                m.insert("curve".into(), json!(b.clamp(-100.0, 100.0)));
            }
            Value::Object(_) => {
                if let Some(v) = num(CMD, c, "bend", -100.0, 100.0)? {
                    m.insert("curve".into(), json!(v));
                }
                if let Some(v) = num(CMD, c, "radius", 0.0, 100_000.0)? {
                    m.insert("curve_radius".into(), json!(v));
                }
            }
            _ => return Err(bad(CMD, "`curve`: the bend in percent (-100..100), {bend, radius} or false")),
        }
    }
    if let Some(st) = p.get("stroke3") {
        m.insert("stroke3".into(), json!(!matches!(st, Value::Bool(false) | Value::Null)));
        if st.is_object() {
            if let Some(c) = color(st, "color")? {
                m.insert("stroke3_color".into(), c);
            }
            if let Some(v) = num(CMD, st, "width", 0.0, 1000.0)? {
                m.insert("stroke3_width".into(), json!(v));
            }
            if let Some(k) = str_p(st, "type") {
                let i =
                    graphic::STROKE_OPTS.iter().position(|o| o.eq_ignore_ascii_case(k)).ok_or_else(|| bad(CMD, "`stroke3.type` is outer, center or inner"))?;
                m.insert("stroke3_type".into(), json!(i));
            }
        }
    }
    if m.is_empty() {
        return Err(bad(CMD, "need `glow`, `innerShadow`, `curve` or `stroke3`"));
    }
    Ok(m)
}

fn text_style(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "graphics.textStyle";
    let clip = target_clip(s, p).ok_or_else(|| bad(CMD, "no graphic clip"))?;
    let (l, ei) = layer_effect_index(s, clip, p)?;
    let props = text_style_props(p)?;
    let curves = props.contains_key("curve") || props.contains_key("curve_radius");
    let is_text = s.active_sequence().and_then(|q| q.find_item(clip)).and_then(|(_, it)| it.effects.get(ei)).is_some_and(|e| e.effect == TEXT_LAYER);
    if curves && !is_text {
        return Err(bad(CMD, "`curve` is for text layers"));
    }
    let tl = s.playhead();
    set_props(s, clip, ei, &props, tl, "Text Style", None)?;
    Ok(json!({"clip": clip.0, "layer": l, "set": Value::Object(props)}))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "captions.setAnimatedStyle",
            "Animated Caption Style",
            r##"{"track":id|"C1"?,"preset":"bold|clean|glow|box|karaoke|twoLine|off","options":{"wordsPerPage":1..8?,"activeColor":"#rrggbb"?,"keywordColor":"#rrggbb"?,"keywords":[str]?,"emoji":{"word":"🔥"}?,"uppercase":bool?,"platform":"youtube|reels|tiktok|vertical|null"?,"pop":1..1.5?,"position":"top|middle|bottom"?,"margin":0..0.45?,"xOffset":-0.5..0.5?,"size":f32?,"outline":f32?,"font":str?,"panel":bool|"#rrggbbaa"?,"keepStyle":bool?}?}"##,
            has_caption_track,
            set_animated_style,
        ),
        spec(
            "captions.wordTiming",
            "Caption Word Timing",
            r#"{"track":id|"C1"?,"from":"transcript|narration"="transcript","captions":[id]?}"#,
            has_caption_track,
            word_timing,
        ),
        spec(
            "graphics.animateText",
            "Animate Text",
            r#"{"clip":id?,"layer":n|name?,"in":kind|{"kind":"fade|typewriter|pop|slideLeft|slideRight|slideDown|slideUp|blurIn|scaleUp|drop|glitch","unit":"letter|word|line"?,"amount":percent?,"stagger":seconds?,"easing":"Back Out"|index?,"duration":seconds?}|null?,"out":(same)?,"loop":{"kind":"wave|bounce|shake|pulse|flicker",…}|null?,"preset":kind?,"clear":bool?,"merge":bool?,"begin":bool?}"#,
            has_graphic,
            animate_text,
        ),
        spec(
            "graphics.textStyle",
            "Text Style",
            r##"{"clip":id?,"layer":n|name?,"glow":false|{"color":"#rrggbb","radius":px,"intensity":percent}?,"innerShadow":false|{"color","opacity","angle","distance","blur"}?,"curve":false|percent|{"bend":-100..100,"radius":px}?,"stroke3":false|{"color","width","type":"outer|center|inner"}?}"##,
            has_graphic,
            text_style,
        ),
    ]
}

#[cfg(test)]
#[path = "text_fx_tests.rs"]
mod tests;

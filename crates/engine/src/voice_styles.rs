//! Voice styles of a cloned voice speaking Malayalam (`filmcraft_tts::styles`): one IndicF5
//! reference per style beside the voice's main one, so an `excited` line reads after an excited
//! stretch of the speaker's own speech and a `calm` one after a calm stretch (F5-TTS's
//! multi-style generation).
//!
//! | command | does |
//! |---|---|
//! | `tts.voiceStyles` | `{"voice"}`: the styles the voice has (where each is from, its words), which reference each delivery reads after, the fallbacks and presets. `set: <style>` makes that style's reference from a stretch of the speaker's speech — `path` (a media file) or `item` (a project item) with `start` / `end` (seconds in it) — and its words: `transcript`, else both Malayalam speech models of the strict check hear them and the result says where they disagree (correct them with `transcript`). The clip ends at the last sentence end its words imply (`trimToSentence`, default true) and is 5–16 s. A style finder pick is taken as it is: `report` (its `styles.json`) and `candidate` (`"excited-1"`), or the pick itself as `entry` — its `style`, `path` (else `item`), `start`, `end` and `transcript`, any of them overridden by the request's own, and kept whole (it already ends at a sentence end; `trimToSentence: true` cuts it). `set: "plain"` makes the stretch the voice's main Malayalam reference (6–15.5 s, as `voice promote` does). `remove: <style>` removes one. Each keeps the voice's references as a version first (`voice versions`, `voice rollback`) |
//!
//! Styles: `conversational`, `warm`, `excited`, `amazed`, `curious`, `calm`, `dramatic`; `plain`
//! is the main reference (also `tts.voiceReference`).

#[cfg_attr(not(feature = "neural-voices"), allow(unused_imports))]
use serde_json::{Value, json};

use filmcraft_project::ItemId;

use crate::commands::{CommandSpec, always, bad, str_p, u64_p};
use crate::{Result, Session};

/// Rate a style's stretch is decoded at (full band, as the main reference's recording).
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
const RATE: u32 = 44_100;
/// Longest stretch decoded, seconds (a style reference is at most 16 s; a little more is allowed
/// for the cut at a sentence end).
pub const MAX_STRETCH_SECONDS: f64 = 46.0;
/// Highest character error rate between the two Malayalam speech models' hearings of a stretch
/// that still counts as their agreeing on its words.
pub const MODELS_AGREE_CER: f64 = 0.15;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "tts.voiceStyles",
        label: "Voice Styles for Malayalam",
        menu: &[],
        shortcut: None,
        params: r#"{"voice":"clone:…","set":"conversational|warm|excited|amazed|curious|calm|dramatic|plain"? (make that style's reference; plain: the main one),"path":file?|"item":id? (the speaker's recording),"start":seconds?,"end":seconds? (the stretch in it, 5–46 s),"transcript":str? (its exact words; left out: both Malayalam speech models hear them),"trimToSentence":bool? (end at the last sentence end of the words; default true, false for a finder pick),"report":"styles.json"?,"candidate":"excited-1"? (a style finder pick),"entry":{style,path,item,start,end,transcript}? (a pick itself),"remove":style?}"#,
        enabled: always,
        run: voice_styles,
        journal: false,
    }]
}

/// What a `tts.voiceStyles` request asks for, checked before anything is decoded or touched.
#[derive(Debug, PartialEq)]
enum Ask {
    List,
    /// `style` `None`: plain, the main reference.
    Set {
        style: Option<&'static str>,
        source: Source,
        transcript: Option<String>,
        to_sentence: bool,
    },
    Remove(&'static str),
}

/// Largest style finder report read, bytes.
const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;

/// `p` with a style finder pick's fields filled in: from `entry` (the pick itself), or from the
/// pick `candidate` (`"excited-1"`) of the finder's report `report` (its `styles.json`). The
/// request's own fields win; a pick that names its file and its project item reads the file; a
/// pick is kept whole unless `trimToSentence` says otherwise (the finder ends it at a sentence
/// end). `p` as it is without either.
fn with_pick(cmd: &str, p: &Value) -> Result<Value> {
    let entry = match (p.get("entry").filter(|v| !v.is_null()), str_p(p, "report"), p.get("candidate").filter(|v| !v.is_null())) {
        (None, None, None) => return Ok(p.clone()),
        (Some(e), None, None) => e.as_object().cloned().ok_or_else(|| bad(cmd, "`entry` is a style finder pick (an object)"))?,
        (None, Some(report), Some(c)) => {
            let id = c.as_str().ok_or_else(|| bad(cmd, "`candidate` is a pick's id (\"excited-1\")"))?;
            pick_in(cmd, report, id)?
        }
        (None, Some(_), None) => return Err(bad(cmd, "give the pick of the report: `candidate` (its id, \"excited-1\")")),
        (None, None, Some(_)) => return Err(bad(cmd, "`candidate` is a pick of a style finder `report` (its styles.json)")),
        _ => return Err(bad(cmd, "give a pick as `entry`, or as `report` and `candidate`, not both")),
    };
    let mut out = p.as_object().cloned().unwrap_or_default();
    for (from, to) in [("style", "set"), ("path", "path"), ("start", "start"), ("end", "end"), ("transcript", "transcript")] {
        if out.get(to).is_none_or(Value::is_null)
            && let Some(v) = entry.get(from).filter(|v| !v.is_null() && v.as_str().is_none_or(|s| !s.trim().is_empty()))
        {
            out.insert(to.into(), v.clone());
        }
    }
    let has_path = out.get("path").is_some_and(|v| !v.is_null());
    if !has_path
        && out.get("item").is_none_or(Value::is_null)
        && let Some(v) = entry.get("item").filter(|v| !v.is_null())
    {
        out.insert("item".into(), v.clone());
    }
    if has_path && p.get("item").is_none_or(Value::is_null) {
        out.remove("item");
    }
    if out.get("trimToSentence").is_none_or(Value::is_null) {
        out.insert("trimToSentence".into(), json!(false));
    }
    for k in ["entry", "report", "candidate"] {
        out.remove(k);
    }
    Ok(Value::Object(out))
}

/// The pick `id` of the style finder report at `path` (`styles: {"excited": [{"id": …}, …]}`).
fn pick_in(cmd: &str, path: &str, id: &str) -> Result<serde_json::Map<String, Value>> {
    let meta = std::fs::metadata(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if meta.len() > MAX_REPORT_BYTES {
        return Err(bad(cmd, format!("{path}: too large for a style finder report")));
    }
    let bytes = std::fs::read(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    let report: Value = serde_json::from_slice(&bytes).map_err(|e| bad(cmd, format!("{path}: not a style finder report ({e})")))?;
    let styles = report.get("styles").and_then(Value::as_object).ok_or_else(|| bad(cmd, format!("{path}: not a style finder report (no `styles`)")))?;
    styles
        .values()
        .filter_map(Value::as_array)
        .flatten()
        .find(|e| e.get("id").and_then(Value::as_str) == Some(id))
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| bad(cmd, format!("{path} has no pick `{}`", id.chars().take(40).collect::<String>())))
}

/// Where a style's stretch is.
#[derive(Debug, PartialEq)]
enum Source {
    Path { path: String, start: f64, end: f64 },
    Item { item: ItemId, start: f64, end: f64 },
}

/// The style id of param `key`, or a refusal that says which there are.
fn style_p(cmd: &str, key: &str, v: &str) -> Result<&'static str> {
    filmcraft_tts::styles::find(v).ok_or_else(|| {
        let plain = matches!(v.trim().to_ascii_lowercase().as_str(), "plain" | "neutral");
        bad(
            cmd,
            if plain {
                "the plain style is the voice's main Malayalam reference: it is replaced (`set: \"plain\"`, tts.voiceReference), not removed".to_string()
            } else {
                format!("`{key}` is a voice style ({}), not `{}`", filmcraft_tts::styles::STYLES.join(", "), v.chars().take(40).collect::<String>())
            },
        )
    })
}

fn ask(cmd: &str, p: &Value) -> Result<Ask> {
    for key in ["voice", "set", "remove", "path", "transcript", "report"] {
        if p.get(key).is_some_and(|v| !v.is_null() && !v.is_string()) {
            return Err(bad(cmd, format!("`{key}` must be a string")));
        }
    }
    let p = &with_pick(cmd, p)?;
    if p.get("set").is_some_and(|v| !v.is_null() && !v.is_string()) {
        return Err(bad(cmd, "`set` must be a string (the pick's `style`)"));
    }
    let set = str_p(p, "set");
    let remove = str_p(p, "remove");
    match (set, remove) {
        (Some(_), Some(_)) => Err(bad(cmd, "give `set` or `remove`, not both")),
        (None, Some(r)) => Ok(Ask::Remove(style_p(cmd, "remove", r)?)),
        (None, None) => Ok(Ask::List),
        (Some(s), None) => {
            // plain is the main reference
            let style = if matches!(s.trim().to_ascii_lowercase().as_str(), "plain" | "neutral") { None } else { Some(style_p(cmd, "set", s)?) };
            let time = |key: &str| -> Result<Option<f64>> {
                match p.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    Some(v) => v
                        .as_f64()
                        .filter(|t| t.is_finite() && (0.0..=86_400.0).contains(t))
                        .map(Some)
                        .ok_or_else(|| bad(cmd, format!("`{key}` is seconds into the recording"))),
                }
            };
            let (start, end) = match (time("start")?, time("end")?) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(bad(cmd, "give the stretch with the style: `start` and `end` (seconds into the recording)")),
            };
            if end <= start || end - start > MAX_STRETCH_SECONDS {
                return Err(bad(cmd, format!("`start`–`end` is a stretch of the recording up to {MAX_STRETCH_SECONDS:.0} s, not {start}–{end}")));
            }
            let source = match (str_p(p, "path").map(str::trim).filter(|x| !x.is_empty()), p.get("item").filter(|v| !v.is_null())) {
                (Some(_), Some(_)) => return Err(bad(cmd, "give `path` or `item`, not both")),
                (Some(path), None) => Source::Path { path: path.to_string(), start, end },
                (None, Some(_)) => {
                    let item = u64_p(p, "item").filter(|_| p["item"].as_u64().is_some()).ok_or_else(|| bad(cmd, "`item` is a project item id"))?;
                    Source::Item { item: ItemId(item), start, end }
                }
                (None, None) => return Err(bad(cmd, "say where the speaker's recording is: `path` (a media file) or `item` (a project item)")),
            };
            let transcript = str_p(p, "transcript").map(|t| t.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|t| !t.is_empty());
            if transcript.as_ref().is_some_and(|t| t.chars().count() > 4000) {
                return Err(bad(cmd, "`transcript` is the stretch's words (at most 4000 characters)"));
            }
            if p.get("transcript").is_some_and(|t| t.as_str().is_some_and(|t| t.trim().is_empty())) {
                return Err(bad(cmd, "`transcript` is empty: give the stretch's words, or leave it out to have them heard"));
            }
            let to_sentence = match p.get("trimToSentence") {
                None | Some(Value::Null) => true,
                Some(Value::Bool(b)) => *b,
                Some(_) => return Err(bad(cmd, "`trimToSentence` must be true or false")),
            };
            Ok(Ask::Set { style, source, transcript, to_sentence })
        }
    }
}

fn voice_styles(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.voiceStyles";
    let voice = str_p(p, "voice").ok_or_else(|| bad(cmd, "`voice` is required (clone:<name>)"))?.to_string();
    if !voice.starts_with(filmcraft_tts::CLONE_PREFIX) {
        return Err(bad(cmd, "`voice` must be a cloned voice (clone:<name>)"));
    }
    let ask = ask(cmd, p)?;
    #[cfg(feature = "neural-voices")]
    {
        let models = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
        let err = |e: filmcraft_tts::TtsError| bad(cmd, e.to_string());
        match ask {
            Ask::List => list(&models, &voice).map_err(err),
            Ask::Remove(style) => {
                let kept = filmcraft_tts::clone::remove_style(&models, &voice, style).map_err(err)?;
                crate::narration::forget_readings(s);
                let mut v = list(&models, &voice).map_err(err)?;
                v["removed"] = json!(style);
                v["kept"] = json!(kept);
                Ok(v)
            }
            Ask::Set { style, source, transcript, to_sentence } => {
                // the voice first: a request for a voice that cannot have styles decodes nothing
                let profile = filmcraft_tts::clone::load_profile(&filmcraft_tts::clone::voices_dir(&models), &voice).map_err(err)?;
                if profile.natural.is_none() {
                    return Err(bad(cmd, format!("{voice} is a timbre-only clone; Malayalam needs a natural clone")));
                }
                let (audio, what, from, to) = stretch(s, cmd, &source)?;
                // the words: given, else heard by both Malayalam speech models
                let (words, given, heard) = match transcript {
                    Some(t) => (t, true, Value::Null),
                    None => hear_both(s, cmd, &models, &audio)?,
                };
                let Some(style) = style else {
                    // plain: the stretch becomes the main reference (the one before is kept)
                    let nr = filmcraft_tts::clone::NewReference {
                        part: filmcraft_tts::clone::Part::Malayalam,
                        audio: &audio,
                        rate: RATE,
                        source: &what,
                        from,
                        to,
                        transcript: Some(&words),
                        note: "set as the plain voice style (tts.voiceStyles)",
                    };
                    let kept = filmcraft_tts::clone::promote(&models, &voice, &nr).map_err(err)?;
                    crate::narration::forget_readings(s);
                    let r = filmcraft_tts::clone::cross_reference(&models, &voice)
                        .map_err(err)?
                        .ok_or_else(|| bad(cmd, "the new main reference did not read back"))?;
                    let check = if given { crate::voice_clone::transcript_check(s, &models, &r) } else { heard };
                    return Ok(json!({
                        "voice": voice, "style": "plain", "main": true, "source": r.source, "from": from, "to": to, "seconds": to - from,
                        "transcript": r.transcript, "given": given, "clip": r.clip, "transcriptCheck": check, "kept": kept,
                    }));
                };
                let n = filmcraft_tts::clone::NewStyle { style, audio: &audio, rate: RATE, source: &what, from, to, transcript: &words, given, to_sentence };
                let set = filmcraft_tts::clone::set_style(&models, &voice, &n).map_err(err)?;
                crate::narration::forget_readings(s);
                let r = &set.reference;
                let check = if given { crate::voice_clone::transcript_check(s, &models, r) } else { heard };
                Ok(json!({
                    "voice": voice,
                    "style": set.style,
                    "source": r.source,
                    "from": r.from,
                    "to": r.to,
                    "seconds": r.to - r.from,
                    "transcript": r.transcript,
                    "given": r.given,
                    "clip": r.clip,
                    "trimmed": set.trimmed,
                    "warning": set.warning,
                    "transcriptCheck": check,
                    "kept": set.version,
                    "styles": list(&models, &voice).map_err(err)?["styles"],
                }))
            }
        }
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, ask);
        Err(bad(cmd, "voice styles are not available in this build (built without `neural-voices`)"))
    }
}

/// The voice's styles, which reference each delivery reads after, and the fallbacks and presets.
#[cfg(feature = "neural-voices")]
fn list(models: &std::path::Path, voice: &str) -> std::result::Result<Value, filmcraft_tts::TtsError> {
    let have = filmcraft_tts::clone::style_references(models, voice)?;
    let main = filmcraft_tts::clone::cross_reference(models, voice)?;
    let has = |s: &str| have.iter().any(|(k, _)| *k == s);
    let reads: serde_json::Map<String, Value> = std::iter::once("plain")
        .chain(filmcraft_tts::styles::STYLES)
        .map(|d| (d.to_string(), json!(filmcraft_tts::styles::resolve(Some(d), &has).unwrap_or("main"))))
        .collect();
    let styles: Vec<Value> = have
        .iter()
        .map(|(s, r)| {
            json!({"style": s, "source": r.source, "from": r.from, "to": r.to, "seconds": r.to - r.from, "transcript": r.transcript, "given": r.given, "clip": r.clip})
        })
        .collect();
    Ok(json!({
        "voice": voice,
        "main": main.map(|r| json!({"source": r.source, "seconds": r.to - r.from, "transcript": r.transcript, "given": r.given})),
        "styles": styles,
        "missing": filmcraft_tts::styles::STYLES.iter().filter(|s| !has(s)).collect::<Vec<_>>(),
        "reads": reads,
        "voiceStyles": crate::narration::voice_styles_json(),
    }))
}

/// The stretch `source` names: mono at [`RATE`], where it is from (`name [a–b s]`), and its times.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
fn stretch(s: &Session, cmd: &str, source: &Source) -> Result<(Vec<f32>, String, f64, f64)> {
    let (src, name, start, end) = match source {
        Source::Path { path, start, end } => {
            let src = s.media.open_file(path, &*s.services).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
            let name = std::path::Path::new(path).file_name().map_or_else(|| path.clone(), |n| n.to_string_lossy().into_owned());
            (src, name, *start, *end)
        }
        Source::Item { item, start, end } => {
            let src = s.source(*item).ok_or_else(|| bad(cmd, format!("item {} is not media that can be read", item.0)))?;
            let name = s.project.item(*item).map(|i| i.name.clone()).unwrap_or_else(|| format!("item {}", item.0));
            (src, name, *start, *end)
        }
    };
    if !src.info().has_audio() {
        return Err(bad(cmd, format!("{name} has no sound")));
    }
    let audio = decode(&src, start, end - start).map_err(|e| bad(cmd, format!("{name}: {e}")))?;
    let got = audio.len() as f64 / f64::from(RATE);
    if got + 0.05 < end - start {
        return Err(bad(cmd, format!("{name} has {got:.1} s of sound from {start} s, not the {:.1} s asked for", end - start)));
    }
    Ok((audio, format!("{name} [{start:.1}–{end:.1} s]"), start, end))
}

/// Mono audio of `src` from `start` for `len` seconds at [`RATE`], read 10 s at a time.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
fn decode(src: &filmcraft_media::SharedSource, start: f64, len: f64) -> std::result::Result<Vec<f32>, String> {
    let total = (len.clamp(0.0, MAX_STRETCH_SECONDS) * f64::from(RATE)) as usize;
    let first = (start.max(0.0) * f64::from(RATE)) as i64;
    let chunk = 10 * RATE as usize;
    let mut out: Vec<f32> = Vec::with_capacity(total);
    while out.len() < total {
        let n = chunk.min(total - out.len());
        let buf = src.audio(first.saturating_add(out.len() as i64), n, RATE).map_err(|e| e.to_string())?;
        let mono = filmcraft_speech::downmix(&buf.channels);
        if mono.is_empty() {
            break;
        }
        let got = mono.len().min(n);
        out.extend(mono.iter().take(got).map(|v| if v.is_finite() { *v } else { 0.0 }));
        if got < n {
            break;
        }
    }
    Ok(out)
}

/// The words of `audio` (mono at [`RATE`]) as both Malayalam speech models of the strict check
/// hear them: the first model's words (the second's when the first cannot hear), `false` (not
/// given), and what each heard and how far apart they are.
#[cfg(feature = "neural-voices")]
fn hear_both(s: &Session, cmd: &str, models: &std::path::Path, audio: &[f32]) -> Result<(String, bool, Value)> {
    let first = crate::narration::hear_malayalam(s, Some(models), audio, RATE);
    let second = crate::narration::hear_malayalam_second(s, Some(models), audio, RATE);
    let heard = |r: &std::result::Result<(String, String), String>| match r {
        Ok((text, model)) => json!({"model": model, "text": text}),
        Err(why) => json!({"skipped": why}),
    };
    let report = |words_from: &str, agree: Option<(f64, bool)>| {
        let mut v = json!({"status": "heard", "wordsFrom": words_from, "heard": [heard(&first), heard(&second)]});
        if let Some((cer, ok)) = agree {
            v["disagreement"] = json!(cer);
            v["agree"] = json!(ok);
            if !ok {
                v["warning"] = json!(format!(
                    "the two Malayalam speech models disagree on the words (character error rate {cer:.2} between them, at most {MODELS_AGREE_CER} expected): listen to the stretch and give its exact words (`transcript`) — IndicF5 aligns its reading to them"
                ));
            }
        } else {
            v["warning"] = json!("only one Malayalam speech model heard the words: check them, and give the exact words (`transcript`) if they are wrong");
        }
        v
    };
    let ok = |t: &str| t.chars().any(char::is_alphanumeric);
    match (&first, &second) {
        (Ok((a, model)), Ok((b, _))) if ok(a) => {
            let cer = filmcraft_tts::heard::score(a, b).cer;
            Ok((a.clone(), false, report(model, Some((cer, cer <= MODELS_AGREE_CER)))))
        }
        (Ok((a, model)), _) if ok(a) => Ok((a.clone(), false, report(model, None))),
        (_, Ok((b, model))) if ok(b) => Ok((b.clone(), false, report(model, None))),
        (Err(why), _) => Err(bad(cmd, format!("the stretch's words are needed: give them (`transcript`), or download a Malayalam speech model ({why})"))),
        _ => Err(bad(cmd, "no words were heard in the stretch: give them (`transcript`)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_checked_before_anything_is_touched() {
        let c = "tts.voiceStyles";
        assert_eq!(ask(c, &json!({"voice": "clone:me"})).unwrap(), Ask::List);
        assert_eq!(ask(c, &json!({"voice": "clone:me", "remove": "Calm"})).unwrap(), Ask::Remove("calm"));
        let set = ask(c, &json!({"set": "excited", "path": "/x/IMG_1.MOV", "start": 12, "end": 22.5, "transcript": "  ഇത്   കൊള്ളാം. "})).unwrap();
        assert_eq!(
            set,
            Ask::Set {
                style: Some("excited"),
                source: Source::Path { path: "/x/IMG_1.MOV".into(), start: 12.0, end: 22.5 },
                transcript: Some("ഇത് കൊള്ളാം.".into()),
                to_sentence: true
            }
        );
        let set = ask(c, &json!({"set": "calm", "item": 7, "start": 0, "end": 9, "trimToSentence": false})).unwrap();
        assert_eq!(set, Ask::Set { style: Some("calm"), source: Source::Item { item: ItemId(7), start: 0.0, end: 9.0 }, transcript: None, to_sentence: false });
        // plain is the main reference
        let set = ask(c, &json!({"set": "Plain", "path": "x", "start": 0, "end": 9})).unwrap();
        assert!(matches!(set, Ask::Set { style: None, .. }), "{set:?}");
        for bad in [
            json!({"set": "excited", "remove": "calm"}),
            json!({"set": "angry", "path": "x", "start": 0, "end": 9}),
            json!({"remove": "plain"}),
            json!({"set": 3}),
            json!({"set": "calm", "path": "x"}),
            json!({"set": "calm", "path": "x", "start": 5}),
            json!({"set": "calm", "path": "x", "start": 9, "end": 5}),
            json!({"set": "calm", "path": "x", "start": -1, "end": 5}),
            json!({"set": "calm", "path": "x", "start": 0, "end": 100}),
            json!({"set": "calm", "path": "x", "start": f64::MAX, "end": 5}),
            json!({"set": "calm", "path": "x", "start": "0", "end": 5}),
            json!({"set": "calm", "start": 0, "end": 9}),
            json!({"set": "calm", "path": "x", "item": 3, "start": 0, "end": 9}),
            json!({"set": "calm", "item": -3, "start": 0, "end": 9}),
            json!({"set": "calm", "item": 2.5, "start": 0, "end": 9}),
            json!({"set": "calm", "item": "3", "start": 0, "end": 9}),
            json!({"set": "calm", "path": "x", "start": 0, "end": 9, "transcript": "   "}),
            json!({"set": "calm", "path": "x", "start": 0, "end": 9, "transcript": "x".repeat(5000)}),
            json!({"set": "calm", "path": "x", "start": 0, "end": 9, "trimToSentence": "yes"}),
            json!({"set": "calm", "path": ["x"], "start": 0, "end": 9}),
        ] {
            assert!(ask(c, &bad).is_err(), "{bad}");
        }
        let plain = ask(c, &json!({"remove": "neutral"})).unwrap_err();
        assert!(plain.to_string().contains("not removed"), "{plain}");
    }

    /// A pick of the style finder (`voice.findStyles`' `styles.json`) is taken as it is: its
    /// style, file, stretch and words, the request's own fields over them.
    #[test]
    fn a_style_finder_pick_is_taken_as_it_is() {
        let c = "tts.voiceStyles";
        let pick = json!({
            "id": "excited-1", "style": "excited", "rank": 1, "path": "/clips/IMG_4184.MOV", "clipName": "IMG_4184.MOV", "item": 1666,
            "start": 41.2, "end": 50.7, "seconds": 9.5, "transcript": "അത് ശരിക്കും ഒരു മാജിക് പോലെ തോന്നി!", "heard": {}, "agreement": 0.92
        });
        let want = Ask::Set {
            style: Some("excited"),
            source: Source::Path { path: "/clips/IMG_4184.MOV".into(), start: 41.2, end: 50.7 },
            transcript: Some("അത് ശരിക്കും ഒരു മാജിക് പോലെ തോന്നി!".into()),
            to_sentence: false,
        };
        // inline: the file (not the item) is read; kept whole
        assert_eq!(ask(c, &json!({"voice": "clone:me", "entry": pick})).unwrap(), want);
        // from the report by id
        let dir = std::env::temp_dir().join(format!("filmcraft-styles-report-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let report = dir.join("styles.json");
        let mut calm = pick.clone();
        calm["id"] = json!("calm-1");
        calm["style"] = json!("calm");
        let plain = json!({"id": "plain-1", "style": "plain", "path": "/clips/a.mov", "item": null, "start": 3.0, "end": 12.0, "transcript": ""});
        std::fs::write(&report, serde_json::to_vec(&json!({"styles": {"excited": [pick], "calm": [calm], "plain": [plain]}})).unwrap()).unwrap();
        let r = report.to_string_lossy().to_string();
        assert_eq!(ask(c, &json!({"voice": "clone:me", "report": r, "candidate": "excited-1"})).unwrap(), want);
        // the request's own fields win: the words corrected, the style moved, a cut asked for
        let fixed = ask(c, &json!({"report": r, "candidate": "calm-1", "transcript": "ശരി.", "set": "warm", "trimToSentence": true})).unwrap();
        assert_eq!(
            fixed,
            Ask::Set {
                style: Some("warm"),
                source: Source::Path { path: "/clips/IMG_4184.MOV".into(), start: 41.2, end: 50.7 },
                transcript: Some("ശരി.".into()),
                to_sentence: true
            }
        );
        // a plain pick without words: the main reference, its words to be heard
        let p = ask(c, &json!({"report": r, "candidate": "plain-1"})).unwrap();
        assert_eq!(
            p,
            Ask::Set { style: None, source: Source::Path { path: "/clips/a.mov".into(), start: 3.0, end: 12.0 }, transcript: None, to_sentence: false }
        );
        // a pick with only its project item reads the item
        let only_item = json!({"style": "calm", "item": 1666, "start": 0.0, "end": 9.0, "transcript": "ശരി."});
        assert!(matches!(ask(c, &json!({"entry": only_item})).unwrap(), Ask::Set { source: Source::Item { item: ItemId(1666), .. }, .. }));
        for bad in [
            json!({"report": r}),
            json!({"candidate": "excited-1"}),
            json!({"report": r, "candidate": "excited-9"}),
            json!({"report": r, "candidate": 3}),
            json!({"report": "/no/such/styles.json", "candidate": "excited-1"}),
            json!({"report": 5, "candidate": "excited-1"}),
            json!({"entry": "excited-1"}),
            json!({"entry": {"style": 3, "path": "x", "start": 0, "end": 9}}),
            json!({"entry": {"style": "excited", "path": "x", "start": 0, "end": 100}}),
            json!({"entry": {"style": "excited"}}),
            json!({"entry": {}, "report": r, "candidate": "excited-1"}),
        ] {
            assert!(ask(c, &bad).is_err(), "{bad}");
        }
        std::fs::write(&report, b"{not json").unwrap();
        assert!(ask(c, &json!({"report": r, "candidate": "excited-1"})).is_err());
        std::fs::write(&report, b"{\"styles\": 3}").unwrap();
        assert!(ask(c, &json!({"report": r, "candidate": "excited-1"})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_commands_are_refused() {
        let mut s = Session::default();
        for p in [
            json!({}),
            json!({"voice": "kokoro-heart"}),
            json!({"voice": 3}),
            json!({"voice": "clone:me", "set": "excited"}),
            json!({"voice": "clone:me", "set": "excited", "path": "/no/such/file.mov", "start": 0, "end": 9}),
            json!({"voice": "clone:me", "set": "excited", "item": 999_999, "start": 0, "end": 9}),
            json!({"voice": "clone:../../etc", "remove": "calm"}),
        ] {
            assert!(s.execute("tts.voiceStyles", p.clone()).is_err(), "{p}");
        }
    }
}

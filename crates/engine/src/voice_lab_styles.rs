//! The style finder and the prosody measure (`voice.findStyles`, `voice.prosody`; feature
//! `neural-voices`): decoding, the models, the files. The measures and the rules are
//! `filmcraft_audio_dsp::prosody` and `filmcraft_tts::voice_lab::styles`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};

use filmcraft_audio_dsp::prosody;
use filmcraft_tts::voice_lab::{self as vl, Target, signal, styles as st};

use super::super::{ProsodyParams, StylesParams};
use super::{Ears, R, RATE, cancelled, decode, file_name, gather, gpu, models, natural_profile, open, out_dir, reference_audio, safe, status, write_json};
use crate::Services;
use crate::media_pool::MediaPool;

/// Candidates kept per clip (never overlapping) for the style finder: all its whole-phrase
/// stretches, near enough.
const PER_CLIP: usize = 40;
/// Longest stretch `voice.prosody` reads from one recording, seconds.
const MAX_SECONDS: f64 = super::super::MAX_PROSODY_SECONDS;

/// The two Malayalam speech models the strict check hears with: the Malayalam Whisper and
/// IndicWhisper (its digits spelled the narration's way before the two are compared). Either
/// may be missing (not downloaded): the other hears alone.
struct Hearers {
    ml: Option<Arc<dyn filmcraft_speech::Transcriber>>,
    indic: Option<Arc<dyn filmcraft_speech::Transcriber>>,
    notes: Vec<String>,
}

impl Hearers {
    fn load(models: &Path) -> Hearers {
        let mut notes = Vec::new();
        let mut one = |id: &str| match filmcraft_speech::load_on(models, id, true) {
            Ok(t) => Some(t),
            Err(e) => {
                notes.push(format!("{id}: {e} (transcript.downloadModel {{\"model\": \"{id}\"}})"));
                None
            }
        };
        let ml = one(filmcraft_speech::models::MALAYALAM_MODEL);
        let indic = one(filmcraft_speech::models::INDICWHISPER_ML);
        Hearers { ml, indic, notes }
    }

    fn hear_with(t: &Arc<dyn filmcraft_speech::Transcriber>, x: &[f32], rate: u32) -> R<String> {
        let mut each = |piece: &[f32]| {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(piece, rate, filmcraft_speech::SAMPLE_RATE);
            t.hear(&x16, "ml").map_err(|e| filmcraft_tts::TtsError::Model(e.to_string()))
        };
        filmcraft_tts::heard::hear_in_pieces(x, rate, filmcraft_tts::heard::HEARD_SECONDS, &mut each).map_err(|e| e.to_string())
    }

    /// What each model heard in `x` (mono at `rate`): `(model, words)`, IndicWhisper's digits
    /// spelled; errors as notes.
    fn hear(&self, x: &[f32], rate: u32, notes: &mut Vec<String>) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        for (id, t, spell) in [(filmcraft_speech::models::MALAYALAM_MODEL, &self.ml, false), (filmcraft_speech::models::INDICWHISPER_ML, &self.indic, true)] {
            let Some(t) = t else { continue };
            match Hearers::hear_with(t, x, rate) {
                Ok(w) if spell => out.push((id, filmcraft_tts::malayalam::spell_numbers(&w, filmcraft_tts::malayalam::Numbers::English))),
                Ok(w) => out.push((id, w)),
                Err(e) => notes.push(format!("{id}: {e}")),
            }
        }
        out
    }
}

/// How alike two texts are as Malayalam is compared (`heard::comparable`): 1 − the mean of the
/// two directions' character error rates, at least 0.
fn agreement(a: &str, b: &str) -> f64 {
    let (ea, eb) = (filmcraft_tts::heard::comparable(a).is_empty(), filmcraft_tts::heard::comparable(b).is_empty());
    if ea || eb {
        // one heard words, the other nothing: no agreement (both nothing: nothing to disagree on)
        return if ea && eb { 1.0 } else { 0.0 };
    }
    let e = 0.5 * (filmcraft_tts::heard::score(a, b).cer + filmcraft_tts::heard::score(b, a).cer);
    (1.0 - e.min(1.0)).max(0.0)
}

/// The words a pick is promoted with: the Malayalam Whisper's when it heard Malayalam, else
/// IndicWhisper's, else whichever heard anything.
fn transcript_of(heard: &[(&'static str, String)]) -> String {
    let ml = |id: &str| heard.iter().find(|(m, w)| *m == id && filmcraft_tts::malayalam::detect(w)).map(|(_, w)| w.clone());
    ml(filmcraft_speech::models::MALAYALAM_MODEL)
        .or_else(|| ml(filmcraft_speech::models::INDICWHISPER_ML))
        .or_else(|| heard.iter().find(|(_, w)| !w.trim().is_empty()).map(|(_, w)| w.clone()))
        .unwrap_or_default()
}

/// The mean direction of embeddings (each scaled to unit length first); the first when they
/// do not agree in length.
fn mean_direction(v: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = v.first() else { return Vec::new() };
    if v.len() == 1 || v.iter().any(|c| c.len() != first.len()) {
        return first.clone();
    }
    let mut sum = vec![0.0f64; first.len()];
    for c in v {
        let n = c.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        if n > 1e-12 && n.is_finite() {
            sum.iter_mut().zip(c).for_each(|(s, x)| *s += f64::from(*x) / n);
        }
    }
    sum.into_iter().map(|x| x as f32).collect()
}

fn features_json(f: &st::Features) -> Value {
    serde_json::to_value(f).unwrap_or(Value::Null)
}

/// `voice.findStyles`.
pub(crate) fn find_styles(sp: &StylesParams, media: &MediaPool, services: &dyn Services, prog: &filmcraft_export::Progress) -> R<Value> {
    let models = models()?;
    let profile = natural_profile(&models, &sp.voice)?;
    let mut ears = Ears::new(&models, true)?;
    status(prog, "measuring the voice's reference");
    let voice_colour = ears.speech_colour(&reference_audio(&models, &profile)?, RATE)?;
    // more recordings of the speaker: their colours averaged with the voice's (each unit length)
    let mut colours = vec![voice_colour.clone()];
    let mut refs_json = Vec::new();
    for r in &sp.speaker_refs {
        status(prog, &format!("measuring the speaker in {}", file_name(&r.path)));
        let src = open(media, services, &r.path)?;
        let secs = src.info().duration.seconds();
        let end = r.end.unwrap_or(secs).min(secs).min(r.start + MAX_SECONDS);
        if !(end.is_finite() && end > r.start + 1.5) {
            return Err(format!("{}: no speech between {:.1} and {end:.1} s for the speaker check", file_name(&r.path), r.start));
        }
        let c = ears.speech_colour(&decode(&src, r.start, end - r.start, RATE)?, RATE)?;
        refs_json.push(json!({"path": r.path, "start": r.start, "end": end, "similarityToVoice": vl::cosine(&c, &voice_colour)}));
        colours.push(c);
    }
    let speaker = mean_direction(&colours);
    let out = out_dir(sp.out.as_deref(), &models, &profile, "styles")?;
    let mut skipped: Vec<Value> = Vec::new();
    let mut asked: Vec<String> = sp.paths.clone();
    asked.extend(sp.items.iter().map(|(_, p)| p.clone()));
    let files = gather(&asked, sp.max_files, &mut skipped);
    prog.total.store(files.len().max(1) as u64, Ordering::Relaxed);
    let settings = vl::FindSettings { min_similarity: sp.min_similarity, per_clip: PER_CLIP };
    let budget = sp.max_minutes * 60.0;
    let (mut scanned, mut clips) = (0.0f64, 0usize);
    let mut measured: Vec<(vl::Candidate, st::Features)> = Vec::new();
    let mut other_language = 0usize;
    let mut no_sentence_end = 0usize;
    let mut per_clip: Vec<Value> = Vec::new();
    let t0 = web_time::Instant::now();
    for (i, f) in files.iter().enumerate() {
        if cancelled(prog) {
            return Err("cancelled".into());
        }
        prog.done.store(i as u64, Ordering::Relaxed);
        let src = match open(media, services, f) {
            Ok(s) => s,
            Err(e) => {
                skipped.push(json!({"path": f, "reason": e}));
                continue;
            }
        };
        let secs = src.info().duration.seconds();
        if !secs.is_finite() || secs < 3.0 {
            skipped.push(json!({"path": f, "reason": "shorter than 3 s"}));
            continue;
        }
        if secs > sp.max_clip_seconds {
            skipped.push(json!({"path": f, "reason": format!("{secs:.0} s, longer than {:.0} s (maxClipSeconds)", sp.max_clip_seconds)}));
            continue;
        }
        if scanned + secs > budget {
            skipped.push(json!({"path": f, "reason": format!("over the {:.0} minutes looked at (maxMinutes)", sp.max_minutes)}));
            continue;
        }
        status(prog, &format!("{}/{}: {} ({secs:.0} s)", i + 1, files.len(), file_name(f)));
        let x = match decode(&src, 0.0, secs, RATE) {
            Ok(x) => x,
            Err(e) => {
                skipped.push(json!({"path": f, "reason": format!("decoding: {e}")}));
                continue;
            }
        };
        scanned += secs;
        clips += 1;
        let sc = match vl::scan(f, &x, RATE, &speaker, &settings, &mut ears, &mut |l| status(prog, l)) {
            Ok(sc) => sc,
            Err(e) => {
                skipped.push(json!({"path": f, "reason": format!("scan: {e}")}));
                continue;
            }
        };
        // what the clip held: how close its speech came to the speaker, and what was kept
        let mut sims: Vec<f64> = sc.chunks.iter().map(|c| c.similarity).collect();
        sims.sort_by(f64::total_cmp);
        let (found_here, kept_before) = (sc.candidates.iter().filter(|c| c.target == Target::Indicf5).count(), measured.len());
        for c in sc.candidates.into_iter().filter(|c| c.target == Target::Indicf5) {
            if sp.malayalam && c.language != "ml" {
                other_language += 1;
                continue;
            }
            if !c.ends_sentence {
                no_sentence_end += 1;
                continue;
            }
            match st::features(signal::cut(&x, RATE, c.span), RATE) {
                Ok(feat) => measured.push((c, feat)),
                Err(e) => skipped.push(json!({"path": f, "reason": format!("prosody of {:.1}–{:.1} s: {e}", c.span.start, c.span.end)})),
            }
        }
        per_clip.push(json!({
            "clip": f, "clipName": file_name(f), "seconds": sc.seconds, "speakerSeconds": sc.speaker_seconds + 0.0, "language": sc.language,
            "chunks": sims.len(), "bestSimilarity": sims.last(), "medianSimilarity": sims.get(sims.len() / 2),
            "candidates": found_here, "measured": measured.len() - kept_before,
        }));
    }
    // the speaker's scale, each candidate's delivery, the picks
    let all_features: Vec<st::Features> = measured.iter().map(|(_, f)| *f).collect();
    let norms = st::norms(&all_features);
    let assessed: Vec<(vl::Candidate, st::Assessment)> = measured.iter().map(|(c, f)| (c.clone(), st::assess(f, &norms))).collect();
    let picks = st::pick(&assessed, sp.top);
    // the picks heard by both Malayalam models (under the GPU lock), and written out
    let dir = out.join("styles");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut notes: Vec<String> = Vec::new();
    let _gpu = if sp.hear && !picks.is_empty() { gpu("voice lab: hearing the style picks")? } else { None };
    let hearers = if sp.hear && !picks.is_empty() { Some(Hearers::load(&models)) } else { None };
    if let Some(h) = &hearers {
        notes.extend(h.notes.iter().cloned());
    }
    let item_of = |path: &str| sp.items.iter().find(|(_, p)| p == path).map(|(id, _)| *id);
    let mut heard_of: Vec<(usize, Vec<(&'static str, String)>)> = Vec::new();
    let mut by_style: serde_json::Map<String, Value> = st::Style::ALL.iter().map(|s| (s.id().to_string(), json!([]))).collect();
    for (k, p) in picks.iter().enumerate() {
        if cancelled(prog) {
            return Err("cancelled".into());
        }
        let Some((c, a)) = assessed.get(p.index) else { continue };
        let Some(feat) = measured.get(p.index).map(|(_, f)| *f) else { continue };
        let id = format!("{}-{}", p.style.id(), p.rank);
        status(prog, &format!("{}/{}: {id} ({} {:.1}–{:.1} s)", k + 1, picks.len(), file_name(&c.clip), c.span.start, c.span.end));
        let audio = open(media, services, &c.clip).and_then(|s| decode(&s, c.span.start, c.span.seconds(), RATE))?;
        let wav = dir.join(format!("{}.wav", safe(&id)));
        filmcraft_tts::wav::write(&wav, &audio, RATE).map_err(|e| e.to_string())?;
        let heard = match heard_of.iter().find(|(i, _)| *i == p.index) {
            Some((_, h)) => h.clone(),
            None => {
                let h = hearers.as_ref().map(|h| h.hear(&audio, RATE, &mut notes)).unwrap_or_default();
                heard_of.push((p.index, h.clone()));
                h
            }
        };
        let agree = match heard.as_slice() {
            [(_, a1), (_, b1)] => Some(agreement(a1, b1)),
            _ => None,
        };
        let shares: serde_json::Map<String, Value> = a.shares.iter().map(|(s, v)| (s.id().to_string(), json!(v))).collect();
        let scores: serde_json::Map<String, Value> = a.scores.iter().map(|(s, v)| (s.id().to_string(), json!(v))).collect();
        let entry = json!({
            "id": id,
            "style": p.style.id(),
            "rank": p.rank,
            "best": p.best,
            "styleOfItsOwn": a.style.id(),
            "confidence": a.share(p.style),
            "fit": p.fit,
            "path": c.clip,
            "clipName": file_name(&c.clip),
            "item": item_of(&c.clip),
            "start": c.span.start,
            "end": c.span.end,
            "seconds": c.span.seconds(),
            "transcript": transcript_of(&heard),
            "heard": heard.iter().map(|(m, w)| (m.to_string(), json!(w))).collect::<serde_json::Map<String, Value>>(),
            "agreement": agree,
            "features": features_json(&feat),
            "z": a.z,
            "shares": shares,
            "scores": scores,
            "similarity": c.similarity,
            "quality": c.quality,
            "referenceScore": c.score,
            "language": c.language,
            "sentence": c.sentence,
            "whisperBaseWords": c.words,
            "wav": wav,
        });
        if let Some(Value::Array(list)) = by_style.get_mut(p.style.id()) {
            list.push(entry);
        }
    }
    drop(_gpu);
    let mut every: Vec<Value> = assessed
        .iter()
        .zip(&measured)
        .map(|((c, a), (_, f))| {
            json!({
                "path": c.clip, "clipName": file_name(&c.clip), "start": c.span.start, "end": c.span.end,
                "style": a.style.id(), "confidence": a.confidence, "features": features_json(f), "z": a.z,
                "similarity": c.similarity, "referenceScore": c.score,
            })
        })
        .collect();
    every.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()).then(a["start"].as_f64().unwrap_or(0.0).total_cmp(&b["start"].as_f64().unwrap_or(0.0))));
    let tally: Vec<String> =
        st::Style::ALL.iter().map(|s| format!("{} {}", s.id(), by_style.get(s.id()).and_then(Value::as_array).map_or(0, Vec::len))).collect();
    let report_path = out.join("styles.json");
    let summary = format!("{} candidates measured in {} clips ({:.0} min); picks: {}", measured.len(), clips, scanned / 60.0, tally.join(", "));
    let report = json!({
        "voice": sp.voice,
        "summary": summary,
        "report": report_path,
        "out": out,
        "settings": {
            "top": sp.top, "language": if sp.malayalam { "ml" } else { "any" }, "hear": sp.hear, "minSimilarity": sp.min_similarity,
            "maxClipSeconds": sp.max_clip_seconds, "maxMinutes": sp.max_minutes, "rate": RATE, "perClip": PER_CLIP,
        },
        "scanned": {
            "files": files.len(), "clips": clips, "minutes": scanned / 60.0, "seconds": t0.elapsed().as_secs_f64(),
            "candidates": measured.len(), "otherLanguage": other_language, "noSentenceEnd": no_sentence_end,
            "skipped": skipped.iter().take(500).collect::<Vec<_>>(), "skippedCount": skipped.len(),
        },
        "speaker": {"voice": sp.voice, "references": refs_json},
        "norms": norms,
        "priors": st::PRIORS,
        "rules": st::Style::ALL.iter().map(|s| (s.id().to_string(), json!(s.rule()))).collect::<serde_json::Map<String, Value>>(),
        "styles": by_style,
        "clips": per_clip,
        "candidates": every,
        "notes": notes,
    });
    write_json(&report_path, &report)?;
    Ok(report)
}

/// A shape as a file name and a report show it.
fn shape_label(s: &prosody::Shape) -> String {
    let mut parts = Vec::new();
    if s.range != 1.0 {
        parts.push(format!("range {}", s.range));
    }
    if s.shift != 0.0 {
        parts.push(format!("shift {:+}", s.shift));
    }
    if s.question_rise != 0.0 {
        parts.push(format!("question {:+}", s.question_rise));
    }
    if s.emphasis_db != 0.0 {
        parts.push(format!("emphasis {} dB", s.emphasis_db));
    }
    if parts.is_empty() { "unchanged".into() } else { parts.join(" ") }
}

fn expressiveness_json(x: &[f32]) -> R<Value> {
    let e = prosody::expressiveness(x, RATE, &prosody::PitchSettings::default()).map_err(|e| e.to_string())?;
    Ok(features_json(&st::Features::from(&e)))
}

/// `voice.prosody`.
pub(crate) fn prosody(pp: &ProsodyParams, media: &MediaPool, services: &dyn Services, prog: &filmcraft_export::Progress) -> R<Value> {
    let out = pp.out.as_ref().map(PathBuf::from);
    if let Some(o) = &out {
        std::fs::create_dir_all(o).map_err(|e| format!("{}: {e}", o.display()))?;
    }
    let mut notes: Vec<String> = Vec::new();
    let (_gpu, hearers) = if pp.hear {
        let models = models()?;
        let g = gpu("voice lab: hearing prosody")?;
        let h = Hearers::load(&models);
        notes.extend(h.notes.iter().cloned());
        (g, Some(h))
    } else {
        (None, None)
    };
    prog.total.store(pp.inputs.len().max(1) as u64, Ordering::Relaxed);
    let mut recordings = Vec::new();
    for (i, seg) in pp.inputs.iter().enumerate() {
        if cancelled(prog) {
            return Err("cancelled".into());
        }
        prog.done.store(i as u64, Ordering::Relaxed);
        status(prog, &format!("{}/{}: {}", i + 1, pp.inputs.len(), file_name(&seg.path)));
        let src = match open(media, services, &seg.path) {
            Ok(s) => s,
            Err(e) => {
                recordings.push(json!({"path": seg.path, "error": e}));
                continue;
            }
        };
        let secs = src.info().duration.seconds();
        let end = seg.end.unwrap_or(secs).min(secs).min(seg.start + MAX_SECONDS);
        if !(end.is_finite() && end > seg.start) {
            recordings.push(json!({"path": seg.path, "error": format!("nothing between {:.2} and {:.2} s", seg.start, end)}));
            continue;
        }
        let x = decode(&src, seg.start, end - seg.start, RATE)?;
        let features = expressiveness_json(&x)?;
        let heard = hearers.as_ref().map(|h| h.hear(&x, RATE, &mut notes)).unwrap_or_default();
        let mut shaped = Vec::new();
        for shape in &pp.shapes {
            let label = shape_label(shape);
            let y = prosody::reshape(&x, RATE, shape, &prosody::PitchSettings::default()).map_err(|e| e.to_string())?;
            let f2 = expressiveness_json(&y)?;
            let mut v = json!({"shape": label, "features": f2});
            if let Some(o) = &out {
                let stem = Path::new(&seg.path).file_stem().map_or_else(|| "audio".to_string(), |s| s.to_string_lossy().into_owned());
                let span = if seg.start > 0.0 || seg.end.is_some() { format!(" {:.1}-{:.1}s", seg.start, end) } else { String::new() };
                let wav = o.join(format!("{}{span} {}.wav", safe(&stem), safe(&label)));
                filmcraft_tts::wav::write(&wav, &y, RATE).map_err(|e| e.to_string())?;
                v["wav"] = json!(wav);
            }
            if let Some(h) = &hearers {
                let h2 = h.hear(&y, RATE, &mut notes);
                let by_model: serde_json::Map<String, Value> = h2
                    .iter()
                    .map(|(m, w)| {
                        let before = heard.iter().find(|(b, _)| b == m).map(|(_, t)| t.as_str()).unwrap_or("");
                        (m.to_string(), json!({"words": w, "cerAgainstOriginal": filmcraft_tts::heard::score(before, w).cer}))
                    })
                    .collect();
                v["heard"] = Value::Object(by_model);
            }
            shaped.push(v);
        }
        recordings.push(json!({
            "path": seg.path,
            "clipName": file_name(&seg.path),
            "start": seg.start,
            "end": end,
            "features": features,
            "heard": heard.iter().map(|(m, w)| (m.to_string(), json!(w))).collect::<serde_json::Map<String, Value>>(),
            "shaped": shaped,
        }));
    }
    let ok = recordings.iter().filter(|r| r.get("error").is_none()).count();
    Ok(json!({
        "summary": format!("{ok} of {} recordings measured{}", pp.inputs.len(), if pp.shapes.is_empty() { String::new() } else { format!(", {} shapes each", pp.shapes.len()) }),
        "recordings": recordings,
        "out": out,
        "notes": notes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_projects_stills_are_not_clips() {
        let dir = std::env::temp_dir().join(format!("fc-styles-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("p.fcproj");
        let doc = json!({"project": {"items": {
            "1": {"kind": {"Media": {"media": {"File": {"path": "/m/a.MOV"}}}}},
            "2": {"kind": {"Media": {"media": {"File": {"path": "/m/b.jpg"}}}}},
            "3": {"kind": {"Media": {"media": {"File": {"path": "/m/c.HEIC"}}}}},
            "4": {"kind": {"Media": {"media": {"File": {"path": "/m/d.wav"}}}}},
        }}});
        std::fs::write(&p, serde_json::to_vec(&doc).unwrap()).unwrap();
        let mut out = Vec::new();
        super::super::project_media(&p, &mut out).unwrap();
        out.sort();
        assert_eq!(out, [PathBuf::from("/m/a.MOV"), PathBuf::from("/m/d.wav")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn helpers() {
        // the mean direction of unit vectors; mismatched or empty input is no panic
        let m = mean_direction(&[vec![2.0, 0.0], vec![0.0, 0.5]]);
        assert!((m[0] - 1.0).abs() < 1e-6 && (m[1] - 1.0).abs() < 1e-6);
        assert_eq!(mean_direction(&[vec![1.0, 2.0], vec![1.0]]), vec![1.0, 2.0]);
        assert!(mean_direction(&[]).is_empty());
        assert_eq!(mean_direction(&[vec![0.0, 0.0], vec![1.0, 0.0]]), vec![1.0, 0.0]);
        // agreement: 1 for the same words, 0 for nothing in common
        assert!((agreement("ഇത് ഒരു റോബോട്ട്", "ഇത് ഒരു റോബോട്ട്") - 1.0).abs() < 1e-9);
        assert_eq!(agreement("ഇത്", ""), 0.0);
        assert_eq!(agreement("", "ഇത്"), 0.0);
        assert_eq!(agreement("", ""), 1.0);
        assert_eq!(shape_label(&prosody::Shape::default()), "unchanged");
        assert_eq!(shape_label(&prosody::Shape { range: 1.3, question_rise: 3.0, ..Default::default() }), "range 1.3 question +3");
        // the words promoted: the Malayalam Whisper's when it heard Malayalam
        let ml = filmcraft_speech::models::MALAYALAM_MODEL;
        let iw = filmcraft_speech::models::INDICWHISPER_ML;
        assert_eq!(transcript_of(&[(iw, "ഒന്ന്".into()), (ml, "രണ്ട്".into())]), "രണ്ട്");
        assert_eq!(transcript_of(&[(ml, "hello".into()), (iw, "ഒന്ന്".into())]), "ഒന്ന്");
        assert_eq!(transcript_of(&[]), "");
    }
}

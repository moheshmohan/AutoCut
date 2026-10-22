//! Text to Speech (narrations): write what the narrator says, pick a voice, and get an audio clip.
//!
//! | command | does |
//! |---|---|
//! | `tts.voices` | languages, voices, vocal pitches, deliveries, pace limits and the pause marker |
//! | `tts.preview` | synthesize without changing the project (`sample: true` speaks the voice's sample sentence); the host plays [`Session::tts_preview`]; `path` also writes a WAV |
//! | `tts.create` | synthesize (or take from the caches: `cached`), write `Narration <n>.wav`, import it and place it at the playhead (one undo step) |
//! | `tts.edit` | change a narration clip's script or voice settings and regenerate it in place (one undo step) |
//! | `tts.inspect` | the narration settings of a clip or item |
//! | `tts.render` | synthesize in a background job into the caches (memory, disk), so the UI's create / edit / preview return at once; already cached: no job |
//! | `tts.downloadVoices` | download the natural voices (Kokoro-82M, 9 US English voices, CMUdict; 336 MB, SHA-256 checked) as a background job |
//!
//! **Where a new narration goes.** At `time` (default: the playhead) on `track` if given (it must
//! be free there), else on the first targeted audio track that is free for the narration's length,
//! else the first free audio track, else on a new audio track. Existing clips are never overwritten.
//!
//! **Editing keeps the clip's length** (user decision, 2026-10-07). The clip keeps its start,
//! duration and speed. When the new speech is longer than the clip, the clip still ends where it
//! did but the file holds the whole speech, so trimming the clip's end out reveals the rest. When it
//! is shorter, the file is padded with silence to the clip's length. The clip's source In is reset
//! to the start of the new speech. Other clips (including other clips of the old narration) are not
//! changed. Each edit writes a new file; the old item and its narration stay, so undo is exact.
//!
//! Files are named `Narration <n>.wav` (mono 32-bit float at the voice's rate, 24 kHz) and saved
//! where voice-over recordings go (Scratch Disks ▸ Captured, else next to the project, else
//! `Narrations` in the data or temporary directory); an edit writes next to the previous file.
//!
//! **Caches.** Synthesis is looked up first in memory ([`Session::tts_cache`], the last few) and
//! then on disk ([`Session::tts_disk`], `<data dir>/narration-cache`, see
//! [`filmcraft_tts::cache`]): keyed by the script, the voice's identity (for a cloned voice its
//! profile too), pitch, pace and delivery, capped at 2 GB, least recently used out first. A rebuild that
//! reads the same lines with the same voice synthesizes nothing; `tts.create` and `tts.render`
//! report `cached`. Native frontends turn the disk cache on; a damaged or unwritable cache only
//! means the voice speaks again.
//!
//! **Delivery** (`plain`, `conversational`, `warm`, `excited`, `amazed`, `curious`, `calm`,
//! `dramatic`) is the feeling a line is spoken with: the natural cloned voices (VoxCPM2) are
//! asked for it in words; the other voices read the script as they always do.
//!
//! **Language** (`language`: `en-US` or `ml-IN`; without it a script written in Malayalam
//! script is Malayalam). Malayalam is spoken by cloned voices through IndicF5
//! (`tts.downloadCloneModel {"model": "indicf5"}`), from the voice sample of a natural clone and
//! that sample's words: the first Malayalam narration of a voice has the speech model hear them
//! (once; `tts.voiceReference` shows them, or sets them by hand). Results carry `notes`: what
//! the Malayalam front end could not read as written. English narrations keep their cache keys.
//!
//! **Numbers** (`numbers`: `english`, the default, or `malayalam`): how a Malayalam narration
//! says numbers written in digits. `english` says years, money, decimals and numbers from 100 up
//! the English way in Malayalam script, as Malayali tech creators speak ("2021-ൽ" → "ട്വന്റി
//! ട്വന്റി വണ്ണിൽ", "$1,600" → "സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളർ"), small counts in Malayalam words;
//! `malayalam` says every number in Malayalam words. Kept with the narration.
//!
//! **The strict check** (`check`: `strict`, the default, or `off`): a Malayalam narration is
//! heard back utterance by utterance by the Malayalam speech model
//! (`filmcraft_speech::models::MALAYALAM_MODEL`, `transcript.downloadModel {"model":
//! "whisper-medium-ml"}`), scored ([`tts::heard`]: the character error rate, and the first six
//! syllables of every sentence on their own — where IndicF5's garble is) and read again with
//! another seed until it is clear, at most four readings, the best kept. Results carry `check`:
//! `status` (`clear`, `unclear`: an utterance never passed, `skipped`: no speech model, with
//! `reason`, `off`), `model` and per utterance its text, what was heard, `cer`, `startCer`,
//! `takes`, the `seed` kept and every try. Checked readings are cached under their own key, the
//! report beside them, so a rebuild reads nothing twice and still reports. English narrations
//! are not checked (`check` is null).

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use serde_json::{Value, json};

use filmcraft_project::{ClipId, Delivery, ItemId, ItemKind, MediaRef, Narration, NumberStyle, TrackId, TrackKind, VocalPitch};
use filmcraft_time::{Tick, TimeRange};
use filmcraft_tts as tts;

use crate::commands::{CommandSpec, always, bad, has_seq, str_p, time_p, u64_p};
use crate::{EngineError, Result, Session};

/// Item name prefix of generated narrations.
const BASE_NAME: &str = "Narration";

/// The narration of a project item.
pub fn narration_of(s: &Session, item: ItemId) -> Option<&Narration> {
    s.project.narrations.get(&item)
}

/// The narration clip a command works on: `clip`, else the first selected clip whose item is a
/// narration.
pub fn target_clip(s: &Session, p: &Value) -> Option<ClipId> {
    let seq = s.active_sequence()?;
    let is_narration = |c: ClipId| seq.find_item(c).is_some_and(|(_, it)| s.project.narrations.contains_key(&it.item));
    match u64_p(p, "clip") {
        Some(c) => Some(ClipId(c)).filter(|c| is_narration(*c)),
        None => s.state.selection.iter().copied().find(|c| is_narration(*c)),
    }
}

fn has_narration_clip(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    target_clip(s, &Value::Null).map(|_| ()).ok_or_else(|| "select a narration clip".into())
}

/// Script and settings from params, on top of `base`.
pub(crate) fn settings(cmd: &str, p: &Value, base: Option<&Narration>) -> Result<Narration> {
    // a setting of the wrong type is an error, never silently replaced by a default
    for key in ["text", "voice", "pitch", "delivery", "language", "numbers"] {
        if p.get(key).is_some_and(|v| !v.is_string()) {
            return Err(bad(cmd, format!("`{key}` must be a string")));
        }
    }
    let text = match str_p(p, "text") {
        Some(t) => t.to_string(),
        None => base.map(|b| b.text.clone()).ok_or_else(|| bad(cmd, "`text` is required"))?,
    };
    let voice = str_p(p, "voice").map(str::to_string).or_else(|| base.map(|b| b.voice.clone())).unwrap_or_else(|| tts::default_voice_id().to_string());
    // the language: given, else the script's (Malayalam script reads as Malayalam), else the
    // narration's own when only its voice settings change
    let malayalam = match str_p(p, "language") {
        Some(l) if tts::malayalam::is_tag(l) => true,
        Some(l) if l.trim().to_ascii_lowercase().starts_with("en") => false,
        Some(l) => return Err(bad(cmd, format!("unknown language `{l}` (en-US, ml-IN)"))),
        None if str_p(p, "text").is_none() && base.is_some() => base.is_some_and(|b| tts::malayalam::is_tag(&b.language)),
        None => tts::malayalam::detect(&text),
    };
    let language = if malayalam {
        if !voice.starts_with(tts::CLONE_PREFIX) {
            return Err(bad(cmd, format!("`{voice}` speaks English; Malayalam is spoken by cloned voices (`clone:<name>`, see tts.clonedVoices)")));
        }
        tts::malayalam::TAG
    } else if voice.starts_with(tts::CLONE_PREFIX) {
        // a cloned voice: its profile is checked when it speaks
        "en-US"
    } else {
        tts::voices_in(None).into_iter().find(|v| v.id == voice).ok_or_else(|| bad(cmd, format!("unknown voice `{voice}` (see tts.voices)")))?.language
    };
    let pitch = match str_p(p, "pitch") {
        Some(v) => VocalPitch::from_id(v).ok_or_else(|| bad(cmd, format!("unknown pitch `{v}` (extraLow, low, default, high, extraHigh)")))?,
        None => base.map(|b| b.pitch).unwrap_or_default(),
    };
    let pace = match p.get("pace") {
        Some(v) => v.as_f64().ok_or_else(|| bad(cmd, "`pace` must be a number"))?,
        None => base.map_or(1.0, |b| b.pace),
    };
    let delivery = match str_p(p, "delivery") {
        Some(v) => Delivery::from_id(v)
            .ok_or_else(|| bad(cmd, format!("unknown delivery `{v}` ({})", Delivery::ALL.iter().map(|d| d.id()).collect::<Vec<_>>().join(", "))))?,
        None => base.map(|b| b.delivery).unwrap_or_default(),
    };
    let take = match p.get("take") {
        Some(v) => v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| bad(cmd, "`take` must be a whole number"))?,
        None => base.map_or(0, |b| b.take),
    };
    // how a Malayalam narration says numbers written in digits (English ignores it)
    let numbers = match str_p(p, "numbers") {
        Some(v) => NumberStyle::from_id(v).ok_or_else(|| bad(cmd, format!("unknown numbers `{v}` (english, malayalam)")))?,
        None => base.map(|b| b.numbers).unwrap_or_default(),
    };
    let n = Narration { text, language: language.to_string(), voice, pitch, pace, delivery, take, numbers, speech_duration: Tick::ZERO };
    n.check().map_err(|e| bad(cmd, e))?;
    Ok(n)
}

/// Synthesized narrations by settings, newest last (`tts.render` fills it in the background so
/// the UI's create / edit / preview don't block).
pub type SynthCache = Arc<Mutex<Vec<(String, Arc<tts::Audio>)>>>;
const CACHE_ENTRIES: usize = 6;

pub(crate) fn cache_key(n: &Narration) -> String {
    let key = format!("{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}", n.voice, n.pitch.id(), n.pace.to_bits(), n.delivery.id(), n.take, n.text);
    // English keeps the key it always had; Malayalam reads numbers in a style
    if tts::malayalam::is_tag(&n.language) { format!("{key}\u{1f}{}\u{1f}{}", n.language, n.numbers.id()) } else { key }
}

/// The front end's style for `n`'s numbers.
fn number_style(n: &Narration) -> tts::malayalam::Numbers {
    match n.numbers {
        NumberStyle::English => tts::malayalam::Numbers::English,
        NumberStyle::Malayalam => tts::malayalam::Numbers::Malayalam,
    }
}

/// The script the voice reads: a Malayalam narration's numbers spelled in its style
/// ([`tts::malayalam::spell_numbers`]); English as written. Synthesis and the disk cache key
/// on it, so a script without digits keeps its cached readings whatever the style.
pub(crate) fn spoken_text(n: &Narration) -> std::borrow::Cow<'_, str> {
    if tts::malayalam::is_tag(&n.language) {
        std::borrow::Cow::Owned(tts::malayalam::spell_numbers(&n.text, number_style(n)))
    } else {
        std::borrow::Cow::Borrowed(&n.text)
    }
}

fn tts_err(cmd: &str, e: tts::TtsError) -> EngineError {
    match e {
        tts::TtsError::NotInstalled => bad(
            cmd,
            "the natural voices (or, for a cloned voice, its cloning model) are not downloaded yet: tts.downloadVoices, tts.downloadCloneModel {\"model\": …}",
        ),
        e => bad(cmd, e.to_string()),
    }
}

/// Synthesize `n` (no cache), under `strict` when given. A cloned voice speaking Malayalam first
/// gets its Malayalam reference when it has none ([`malayalam_reference`]).
fn render(
    n: &Narration,
    models_dir: Option<&std::path::Path>,
    transcriber: Option<&Transcriber>,
    strict: Option<&tts::heard::Strict>,
) -> std::result::Result<(tts::Audio, Vec<tts::heard::Checked>), tts::TtsError> {
    malayalam_reference(n, models_dir, transcriber)?;
    let voice = tts::voice_for(&n.voice, &n.language, models_dir)?;
    match strict {
        Some(st) => voice.synthesize_checked(&spoken_text(n), &params(n), st),
        None => Ok((voice.synthesize(&spoken_text(n), &params(n))?, Vec::new())),
    }
}

/// Whether `p` asks for the strict check: `check` is `"strict"` (the default) or `"off"`
/// (`true` / `false` too).
pub(crate) fn strict_requested(cmd: &str, p: &Value) -> Result<bool> {
    match p.get("check") {
        None | Some(Value::Null) => Ok(true),
        Some(Value::Bool(b)) => Ok(*b),
        Some(Value::String(s)) if s == "strict" => Ok(true),
        Some(Value::String(s)) if s == "off" => Ok(false),
        Some(v) => Err(bad(cmd, format!("`check` must be \"strict\" or \"off\", not {v}"))),
    }
}

/// The strict check applies to `n`: a cloned voice speaking Malayalam (IndicF5) or English
/// (VoxCPM).
fn checkable(n: &Narration) -> bool {
    n.voice.starts_with(tts::CLONE_PREFIX)
}

/// The language a check hears in (ISO 639-1).
fn check_language(n: &Narration) -> &'static str {
    if tts::malayalam::is_tag(&n.language) { "ml" } else { "en" }
}

/// A speech model hearing narration (the strict check): resampled to 16 kHz, long readings heard
/// in pieces cut at pauses.
struct Judge {
    t: Arc<dyn filmcraft_speech::Transcriber>,
    id: String,
    language: &'static str,
}

impl tts::heard::Hearer for Judge {
    fn hear(&self, samples: &[f32], rate: u32) -> std::result::Result<String, String> {
        let mut each = |piece: &[f32]| {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(piece, rate, filmcraft_speech::SAMPLE_RATE);
            self.t.hear(&x16, self.language).map_err(|e| tts::TtsError::Model(e.to_string()))
        };
        tts::heard::hear_in_pieces(samples, rate, tts::heard::HEARD_SECONDS, &mut each).map_err(|e| e.to_string())
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}

/// The Malayalam speech model's check, as results and cache keys name it.
#[cfg(test)]
fn judge_id() -> String {
    judge_id_for(filmcraft_speech::models::MALAYALAM_MODEL)
}

/// Speech model `model`'s check, as results and cache keys name it.
fn judge_id_for(model: &str) -> String {
    format!("{model} check v{}", tts::heard::VERSION)
}

/// Speech model `model` in `dir` hearing `language`, loaded once per process (on the GPU where
/// there is one; the last few kept).
fn judge(dir: &std::path::Path, model: &str, language: &'static str) -> std::result::Result<Arc<Judge>, String> {
    type Judges = Vec<(std::path::PathBuf, String, Arc<Judge>)>;
    static JUDGES: OnceLock<Mutex<Judges>> = OnceLock::new();
    let mut g = JUDGES.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, _, j)) = g.iter().find(|(d, m, j)| d == dir && m == model && j.language == language) {
        return Ok(j.clone());
    }
    let t = filmcraft_speech::load_on(dir, model, true).map_err(|e| e.to_string())?;
    let j = Arc::new(Judge { t, id: judge_id_for(model), language });
    g.push((dir.to_path_buf(), model.to_string(), j.clone()));
    // two Malayalam models and an English one at most
    let extra = g.len().saturating_sub(3);
    g.drain(..extra);
    Ok(j)
}

/// A speech model, loaded when it first hears: a reading found in the cache needs only the
/// check's name.
struct LazyJudge {
    dir: std::path::PathBuf,
    model: &'static str,
    language: &'static str,
}

impl tts::heard::Hearer for LazyJudge {
    fn hear(&self, samples: &[f32], rate: u32) -> std::result::Result<String, String> {
        judge(&self.dir, self.model, self.language)?.hear(samples, rate)
    }

    fn id(&self) -> String {
        judge_id_for(self.model)
    }
}

/// The installed speech model of `candidates` (the first that is), lazily loaded.
fn lazy_model(
    models_dir: Option<&std::path::Path>,
    candidates: &[&'static str],
    language: &'static str,
) -> std::result::Result<Arc<dyn tts::heard::Hearer>, String> {
    let dir = models_dir.ok_or("no data directory for speech models")?;
    if !filmcraft_speech::available() {
        return Err(crate::transcript::NO_SPEECH.to_string());
    }
    for &id in candidates {
        if filmcraft_speech::models::find(id).is_some_and(|m| filmcraft_speech::models::installed(dir, m)) {
            return Ok(Arc::new(LazyJudge { dir: dir.to_path_buf(), model: id, language }));
        }
    }
    let first = candidates.first().copied().unwrap_or_default();
    let size = filmcraft_speech::models::find(first).map_or(0.0, |m| m.size() as f64 / 1e9);
    Err(format!(
        "the {} speech model is not downloaded: transcript.downloadModel {{\"model\": \"{first}\"}} ({size:.1} GB)",
        if language == "ml" { "Malayalam" } else { "English" }
    ))
}

/// What hears narration in `language` (`ml`, `en`): the host's ([`Session::narration_hearer`]),
/// else the Malayalam speech model, or for English Whisper large-v3-turbo (small when turbo is
/// not downloaded). `Err`: why the strict check cannot run (said in the result; the narration is
/// still read).
fn hearer_for(s: &Session, models_dir: Option<&std::path::Path>, language: &'static str) -> std::result::Result<Arc<dyn tts::heard::Hearer>, String> {
    if let Some(h) = &s.narration_hearer {
        return Ok(h.clone());
    }
    match language {
        "ml" => lazy_model(models_dir, &[filmcraft_speech::models::MALAYALAM_MODEL], "ml"),
        _ => lazy_model(models_dir, &[filmcraft_speech::models::TURBO_MODEL, "whisper-small"], "en"),
    }
}

/// The second opinion on Malayalam narration: the host's ([`Session::narration_second`]), else
/// IndicWhisper when downloaded (none for English). IndicWhisper writes spoken numbers as digits;
/// they are spelled the way the narration's script spells them before they are compared.
fn second_for(s: &Session, models_dir: Option<&std::path::Path>, n: &Narration) -> Option<Arc<dyn tts::heard::Hearer>> {
    if let Some(h) = &s.narration_second {
        return Some(h.clone());
    }
    if s.narration_hearer.is_some() || check_language(n) != "ml" {
        return None;
    }
    let inner = lazy_model(models_dir, &[filmcraft_speech::models::INDICWHISPER_ML], "ml").ok()?;
    Some(Arc::new(SpelledNumbers { inner, style: number_style(n) }))
}

/// A recogniser whose digits are spelled in a narration's style (`tts::malayalam::spell_numbers`).
struct SpelledNumbers {
    inner: Arc<dyn tts::heard::Hearer>,
    style: tts::malayalam::Numbers,
}

impl tts::heard::Hearer for SpelledNumbers {
    fn hear(&self, samples: &[f32], rate: u32) -> std::result::Result<String, String> {
        self.inner.hear(samples, rate).map(|t| tts::malayalam::spell_numbers(&t, self.style))
    }

    fn id(&self) -> String {
        format!("{} (numbers spelled)", self.inner.id())
    }
}

/// What the Malayalam speech model hears in `samples` (mono, `rate`), and which model heard it.
/// `Err`: why it cannot hear (no model, this build).
#[cfg(feature = "neural-voices")]
pub(crate) fn hear_malayalam(s: &Session, models_dir: Option<&std::path::Path>, samples: &[f32], rate: u32) -> std::result::Result<(String, String), String> {
    let h = hearer_for(s, models_dir, "ml")?;
    Ok((h.hear(samples, rate)?, h.id()))
}

/// The least similarity of a reading to its voice's sample that passes the voice gate: the
/// owner's cloned readings measured against his sample (crate README of `filmcraft-speech`,
/// "The voice gate") sit well above it; a reading that came out as someone else falls under it.
pub const MIN_VOICE_SIMILARITY: f64 = 0.3;

/// The voice gate of a cloned voice: ECAPA embeddings of readings against the embedding of the
/// voice's own sample (feature `whisper` and the speaker model downloaded).
#[cfg(all(feature = "whisper", feature = "neural-voices"))]
struct VoiceGate {
    model: Arc<filmcraft_speech::ecapa::Ecapa>,
    reference: Vec<f32>,
    voice: String,
}

#[cfg(all(feature = "whisper", feature = "neural-voices"))]
impl tts::heard::VoiceMatch for VoiceGate {
    fn similarity(&self, samples: &[f32], rate: u32) -> std::result::Result<f64, String> {
        let x16 = filmcraft_audio_dsp::resample::resample_sinc(samples, rate, filmcraft_speech::SAMPLE_RATE);
        let e = self.model.embed(&x16).map_err(|e| e.to_string())?;
        Ok(f64::from(filmcraft_speech::ecapa::cosine(&e, &self.reference)))
    }

    fn id(&self) -> String {
        format!("{} {}", filmcraft_speech::models::ECAPA_MODEL, self.voice)
    }
}

/// The voice gate for cloned voice `voice`: the host's ([`Session::narration_voice`]), else the
/// speaker model against the voice's sample, embedded once per voice. None: no speaker model,
/// no sample, or this build.
fn voice_gate_for(s: &Session, models_dir: Option<&std::path::Path>, voice: &str) -> Option<Arc<dyn tts::heard::VoiceMatch>> {
    if let Some(v) = &s.narration_voice {
        return Some(v.clone());
    }
    #[cfg(all(feature = "whisper", feature = "neural-voices"))]
    {
        type Gates = Vec<(String, Arc<VoiceGate>)>;
        static GATES: OnceLock<Mutex<Gates>> = OnceLock::new();
        let dir = models_dir?;
        if s.narration_hearer.is_some() {
            // a host's own recogniser (tests): no models from the data directory
            return None;
        }
        let model = filmcraft_speech::ecapa::load_cached(dir)?;
        let voices = tts::clone::voices_dir(dir);
        let profile = tts::clone::load_profile(&voices, voice).ok()?;
        let key = format!("{voice}|{}|{}", profile.created, profile.revision);
        let mut g = GATES.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, gate)) = g.iter().find(|(k, _)| *k == key) {
            return Some(gate.clone() as Arc<dyn tts::heard::VoiceMatch>);
        }
        let path = tts::clone::sample_path(&voices, &profile).or_else(|| tts::clone::full_sample_path(&voices, &profile).ok())?;
        let (x, rate) = tts::wav::read(&path).ok()?;
        let x16 = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, filmcraft_speech::SAMPLE_RATE);
        let reference = model.embed(&x16).ok()?;
        let gate = Arc::new(VoiceGate { model, reference, voice: voice.to_string() });
        g.push((key, gate.clone()));
        let extra = g.len().saturating_sub(8);
        g.drain(..extra);
        Some(gate as Arc<dyn tts::heard::VoiceMatch>)
    }
    #[cfg(not(all(feature = "whisper", feature = "neural-voices")))]
    {
        let _ = (models_dir, voice);
        None
    }
}

/// The strict check's judges: who hears, who gives a second opinion and how the two combine,
/// and the voice gate.
#[derive(Clone)]
struct Judges {
    hearer: Arc<dyn tts::heard::Hearer>,
    second: Option<Arc<dyn tts::heard::Hearer>>,
    agreement: tts::heard::Agreement,
    voice: Option<Arc<dyn tts::heard::VoiceMatch>>,
    min_similarity: f64,
    policy: tts::heard::Policy,
}

impl Judges {
    fn strict(&self) -> tts::heard::Strict<'_> {
        tts::heard::Strict {
            hearer: self.hearer.as_ref(),
            policy: self.policy,
            second: self.second.as_deref(),
            agreement: self.agreement,
            voice: self.voice.as_deref(),
            min_similarity: self.min_similarity,
        }
    }

    fn id(&self) -> String {
        self.strict().id()
    }

    /// The same check with only its first recogniser: what readings were checked under before
    /// the second opinion and the voice gate (their cache entries are adopted, see [`cached`]).
    fn first_only(&self) -> Judges {
        Judges { second: None, voice: None, ..self.clone() }
    }
}

/// How `n` is checked: by its judges, or not (and why).
#[derive(Clone)]
enum Check {
    /// Not a narration the strict check reads (a voice that is not cloned), or `check: "off"`.
    Off,
    /// Asked for but impossible: the reason.
    Skipped(String),
    Strict(Judges),
}

/// The strict check `n` reads under, when `p` asks for it: `agreement` (`both`, the default:
/// a line is read again only when both Malayalam models disagree with the script; `either`;
/// `first`), `voiceGate` (default on when the speaker model is downloaded) and
/// `minSimilarity` (default [`MIN_VOICE_SIMILARITY`]).
fn check_for(s: &Session, cmd: &str, p: &Value, n: &Narration) -> Result<Check> {
    let on = strict_requested(cmd, p)?;
    let agreement = match p.get("agreement").filter(|v| !v.is_null()) {
        None => tts::heard::Agreement::Both,
        Some(Value::String(a)) => {
            tts::heard::Agreement::from_id(a).ok_or_else(|| bad(cmd, format!("`agreement` is \"both\", \"either\" or \"first\", not \"{a}\"")))?
        }
        Some(v) => return Err(bad(cmd, format!("`agreement` is \"both\", \"either\" or \"first\", not {v}"))),
    };
    let gate = match p.get("voiceGate").filter(|v| !v.is_null()) {
        None => true,
        Some(v) => v.as_bool().ok_or_else(|| bad(cmd, "`voiceGate` must be true or false"))?,
    };
    let min_similarity = match p.get("minSimilarity").filter(|v| !v.is_null()) {
        None => MIN_VOICE_SIMILARITY,
        Some(v) => v.as_f64().filter(|f| f.is_finite() && (-1.0..=1.0).contains(f)).ok_or_else(|| bad(cmd, "`minSimilarity` is a number from -1 to 1"))?,
    };
    if !checkable(n) || !on {
        return Ok(Check::Off);
    }
    let models = crate::transcript::models_dir();
    let language = check_language(n);
    Ok(match hearer_for(s, models.as_deref(), language) {
        Ok(hearer) => Check::Strict(Judges {
            hearer,
            second: if agreement == tts::heard::Agreement::First { None } else { second_for(s, models.as_deref(), n) },
            agreement,
            voice: if gate { voice_gate_for(s, models.as_deref(), &n.voice) } else { None },
            min_similarity,
            policy: if language == "ml" { tts::heard::Policy::default() } else { tts::heard::Policy::english() },
        }),
        Err(why) => Check::Skipped(why),
    })
}

/// The `check` of a result: what the strict check found in `n`'s reading.
fn check_json(n: &Narration, check: &Check, report: Option<&Value>) -> Value {
    match check {
        Check::Off if !checkable(n) => Value::Null,
        Check::Off => json!({"status": "off"}),
        Check::Skipped(why) => json!({"status": "skipped", "reason": why}),
        Check::Strict(j) => {
            let utterances = report.cloned().unwrap_or(Value::Null);
            let clear = utterances.as_array().is_some_and(|u| u.iter().all(|c| c["passed"].as_bool() == Some(true)));
            json!({"status": if clear { "clear" } else { "unclear" }, "model": j.id(), "utterances": utterances})
        }
    }
}

/// Reports of checked readings by memory cache key, newest last (the disk keeps them beside
/// each entry).
static REPORTS: OnceLock<Mutex<Vec<(String, Value)>>> = OnceLock::new();

fn report_put(key: String, report: Value) {
    let mut r = REPORTS.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
    r.retain(|(k, _)| *k != key);
    r.push((key, report));
    let extra = r.len().saturating_sub(64);
    r.drain(..extra);
}

fn report_get(key: &str) -> Option<Value> {
    REPORTS.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

type Transcriber = Arc<dyn filmcraft_speech::Transcriber>;

/// Whether `n` is a cloned voice speaking Malayalam whose Malayalam reference (its voice
/// sample's clip and that clip's transcript) is still to be made.
fn needs_malayalam_reference(n: &Narration, models_dir: Option<&std::path::Path>) -> bool {
    #[cfg(feature = "neural-voices")]
    {
        tts::malayalam::is_tag(&n.language)
            && n.voice.starts_with(tts::CLONE_PREFIX)
            && models_dir.is_some_and(|d| matches!(tts::clone::cross_reference(d, &n.voice), Ok(None)))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (n, models_dir);
        false
    }
}

/// Make the Malayalam reference of `n`'s cloned voice when it has none: the speech model hears
/// the clip's words (in English: the voice sample is the speaker's English). Once per voice.
fn malayalam_reference(n: &Narration, models_dir: Option<&std::path::Path>, transcriber: Option<&Transcriber>) -> std::result::Result<(), tts::TtsError> {
    if !needs_malayalam_reference(n, models_dir) {
        return Ok(());
    }
    #[cfg(feature = "neural-voices")]
    {
        let dir = models_dir.ok_or(tts::TtsError::NotInstalled)?;
        let t = transcriber.ok_or_else(|| {
            tts::TtsError::Invalid(format!(
                "{} speaks Malayalam from its voice sample, whose words must be known: download a speech model (transcript.downloadModel) or give them (tts.voiceReference {{\"voice\": \"{}\", \"transcript\": …}})",
                n.voice, n.voice
            ))
        })?;
        let mut hear = |x16: &[f32]| -> std::result::Result<String, String> {
            let opts = filmcraft_speech::Options { language: Some("en".into()), diarize: false, ..Default::default() };
            t.transcribe(x16, &opts, &mut |_, _| true).map(|tr| tr.text()).map_err(|e| e.to_string())
        };
        tts::clone::prepare_cross_lingual(dir, &n.voice, None, None, &mut hear)?;
    }
    #[cfg(not(feature = "neural-voices"))]
    let _ = transcriber;
    Ok(())
}

/// The speech model, when `n` needs it to make its voice's Malayalam reference (None otherwise,
/// or when there is none: rendering then says how to give the words instead).
fn reference_transcriber(s: &Session, n: &Narration, models_dir: Option<&std::path::Path>) -> Option<Transcriber> {
    if !needs_malayalam_reference(n, models_dir) {
        return None;
    }
    crate::transcript::transcriber(s, &json!({})).ok()
}

/// The median pitch of synthesized speech, Hz (natural voices' builds only): a reading that
/// came out far above its voice's own pitch overacted.
fn median_pitch(samples: &[f32], rate: u32) -> Option<f64> {
    #[cfg(feature = "neural-voices")]
    {
        tts::clone::median_pitch(samples, rate)
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (samples, rate);
        None
    }
}

/// What the voice could not read as written (Malayalam: Latin words, symbols): empty for English.
pub(crate) fn notes(n: &Narration) -> Vec<String> {
    if tts::malayalam::is_tag(&n.language) { tts::malayalam::report(&n.text) } else { Vec::new() }
}

/// How `n` is spoken.
fn params(n: &Narration) -> tts::Params {
    tts::Params { semitones: n.pitch.semitones(), pace: n.pace, style: n.delivery.style(), take: n.take }
}

/// The disk cache's key for `n` read under `check` (None: the voice has no stable identity,
/// e.g. a cloned voice whose profile cannot be read; it is then not cached). A checked reading
/// has its own key: its utterances may have been read again.
fn disk_key(n: &Narration, models_dir: Option<&std::path::Path>, check: &Check) -> Option<String> {
    let identity = tts::cache::voice_identity_for(&n.voice, &n.language, models_dir)?;
    let identity = match check {
        Check::Strict(j) => format!("{identity}|strict {}", j.id()),
        _ => identity,
    };
    Some(tts::cache::DiskCache::key(&identity, &spoken_text(n), &params(n)))
}

/// The memory cache's key for `n` read under `check`.
fn memory_key(n: &Narration, check: &Check) -> String {
    match check {
        Check::Strict(j) => format!("{}\u{1f}strict\u{1f}{}", cache_key(n), j.id()),
        _ => cache_key(n),
    }
}

/// `n` from the memory cache, else the disk cache (then also kept in memory), with the strict
/// check's report of the reading when there is one.
fn cached(s: &Session, n: &Narration, models_dir: Option<&std::path::Path>, check: &Check) -> Option<(Arc<tts::Audio>, Option<Value>)> {
    let key = memory_key(n, check);
    if let Some(a) = s.tts_cache.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, a)| a.clone()) {
        return Some((a, report_get(&key)));
    }
    let disk = s.tts_disk.as_ref()?;
    let dk = disk_key(n, models_dir, check)?;
    let (a, report) = match disk.get(&dk) {
        Some(a) => (Arc::new(a), disk.note(&dk).and_then(|b| serde_json::from_slice::<Value>(&b).ok())),
        None => {
            let (a, report) = adopt(disk, n, models_dir, check)?;
            if let Ok(bytes) = serde_json::to_vec(&report) {
                disk.put(&dk, &a);
                disk.put_note(&dk, &bytes);
            }
            (a, Some(report))
        }
    };
    if let Some(r) = &report {
        report_put(key.clone(), r.clone());
    }
    cache_put(&s.tts_cache, key, a.clone());
    Some((a, report))
}

/// A reading of `n` cached under an earlier check (its first recogniser alone, before the
/// second opinion and the voice gate; or, for English, unchecked), heard now as a whole by
/// `check`'s judges: when it passes it is kept under the new check (no voice reads it again),
/// with a report of that hearing (`adopted`). `None`: nothing to adopt, or it does not pass.
fn adopt(disk: &tts::cache::DiskCache, n: &Narration, models_dir: Option<&std::path::Path>, check: &Check) -> Option<(Arc<tts::Audio>, Value)> {
    let Check::Strict(j) = check else { return None };
    let first = Check::Strict(j.first_only());
    let mut earlier = Vec::new();
    if j.second.is_some() || j.voice.is_some() {
        earlier.push(first);
    }
    if check_language(n) == "en" {
        earlier.push(Check::Off);
    }
    let english = check_language(n) == "en";
    let said = if english { tts::heard::english_words(&spoken_text(n)) } else { spoken_text(n).into_owned() };
    for old in earlier {
        let Some(k) = disk_key(n, models_dir, &old) else { continue };
        let Some(a) = disk.get(&k) else { continue };
        let note = disk.note(&k).and_then(|b| serde_json::from_slice::<Value>(&b).ok());
        // the first recogniser passed every utterance: the second opinion can only agree
        let clear_before = note.as_ref().and_then(Value::as_array).is_some_and(|u| !u.is_empty() && u.iter().all(|c| c["passed"].as_bool() == Some(true)));
        let mut heard = String::new();
        let mut text_ok = clear_before;
        let mut score = tts::heard::Score::default();
        if !clear_before {
            let norm = |t: String| if english { tts::heard::english_words(&t) } else { t };
            heard = norm(j.hearer.hear(&a.samples, a.sample_rate).ok()?);
            score = tts::heard::score(&said, &heard);
            text_ok = j.policy.passes(&score);
            if !text_ok
                && j.agreement == tts::heard::Agreement::Both
                && let Some(h2) = &j.second
                && let Ok(h) = h2.hear(&a.samples, a.sample_rate)
            {
                text_ok = j.policy.passes(&tts::heard::score(&said, &norm(h)));
            }
        }
        let similarity = j.voice.as_ref().and_then(|v| v.similarity(&a.samples, a.sample_rate).ok());
        let voice_ok = similarity.is_none_or(|s| !s.is_finite() || s >= j.min_similarity);
        if text_ok && voice_ok {
            let report = match note.filter(|_| clear_before) {
                Some(Value::Array(mut u)) => {
                    for c in &mut u {
                        c["adopted"] = json!(true);
                    }
                    if let (Some(s), Some(c)) = (similarity, u.first_mut()) {
                        c["similarity"] = json!(s);
                    }
                    Value::Array(u)
                }
                _ => {
                    json!([{"text": said, "heard": heard, "cer": score.cer, "startCer": score.start_cer, "passed": true, "adopted": true, "similarity": similarity}])
                }
            };
            return Some((Arc::new(a), report));
        }
    }
    None
}

/// Keep a fresh reading of `n` (and its strict check's report) in the caches.
fn keep(
    cache: &SynthCache,
    disk: Option<&tts::cache::DiskCache>,
    n: &Narration,
    models_dir: Option<&std::path::Path>,
    check: &Check,
    a: &Arc<tts::Audio>,
    report: &[tts::heard::Checked],
) {
    let report = matches!(check, Check::Strict(_)).then(|| serde_json::to_value(report).unwrap_or(Value::Null));
    if let (Some(disk), Some(k)) = (disk, disk_key(n, models_dir, check)) {
        disk.put(&k, a);
        if let Some(r) = &report
            && let Ok(bytes) = serde_json::to_vec(r)
        {
            disk.put_note(&k, &bytes);
        }
    }
    let key = memory_key(n, check);
    if let Some(r) = report {
        report_put(key.clone(), r);
    }
    cache_put(cache, key, a.clone());
}

/// The audio of `n` under the strict check `p` asks for, whether it came from a cache (nothing
/// was synthesized), and the result's `check`.
fn synthesize(s: &Session, cmd: &str, p: &Value, n: &Narration) -> Result<(Arc<tts::Audio>, bool, Value)> {
    let models = crate::transcript::models_dir();
    let check = check_for(s, cmd, p, n)?;
    if let Some((a, report)) = cached(s, n, models.as_deref(), &check) {
        return Ok((a, true, check_json(n, &check, report.as_ref())));
    }
    let transcriber = reference_transcriber(s, n, models.as_deref());
    let strict = match &check {
        Check::Strict(j) => Some(j.strict()),
        _ => None,
    };
    let (a, report) = render(n, models.as_deref(), transcriber.as_ref(), strict.as_ref()).map_err(|e| tts_err(cmd, e))?;
    let a = Arc::new(a);
    keep(&s.tts_cache, s.tts_disk.as_ref(), n, models.as_deref(), &check, &a, &report);
    let report = serde_json::to_value(&report).unwrap_or(Value::Null);
    Ok((a, false, check_json(n, &check, Some(&report))))
}

fn cache_put(c: &SynthCache, key: String, a: Arc<tts::Audio>) {
    let mut c = c.lock().unwrap_or_else(PoisonError::into_inner);
    c.retain(|(k, _)| *k != key);
    c.push((key, a));
    let extra = c.len().saturating_sub(CACHE_ENTRIES);
    c.drain(..extra);
}

fn package_json() -> Value {
    let dir = crate::transcript::models_dir();
    json!({
        "id": tts::catalog::PACKAGE_ID,
        "name": tts::catalog::PACKAGE_NAME,
        "installed": dir.as_deref().is_some_and(tts::catalog::installed),
        "size": tts::catalog::size(),
        "missingBytes": dir.as_deref().map_or(tts::catalog::size(), tts::catalog::missing_bytes),
        "license": tts::catalog::LICENSE,
        "source": tts::catalog::SOURCE,
        "available": cfg!(feature = "neural-voices"),
    })
}

fn voices(_s: &mut Session, _p: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    Ok(json!({
        "package": package_json(),
        "languages": tts::languages().iter().map(|(id, name)| json!({"id": id, "name": name})).collect::<Vec<_>>(),
        "voices": tts::voices_in(dir.as_deref()),
        "cloned": crate::voice_clone::cloned_json(),
        "defaultVoice": tts::default_voice_id(),
        "pitches": VocalPitch::ALL.iter().map(|p| json!({"id": p.id(), "label": p.label(), "semitones": p.semitones()})).collect::<Vec<_>>(),
        "deliveries": Delivery::ALL.iter().map(|d| json!({"id": d.id(), "label": d.label(), "style": d.style()})).collect::<Vec<_>>(),
        "numbers": NumberStyle::ALL.iter().map(|n| json!({"id": n.id(), "label": n.label()})).collect::<Vec<_>>(),
        "pace": {"min": tts::MIN_PACE, "max": tts::MAX_PACE, "default": 1.0},
        "pauseMarker": tts::script::PAUSE_MARKER,
        "maxTextBytes": tts::MAX_TEXT_BYTES,
        "sampleSentence": tts::SAMPLE_SENTENCE,
    }))
}

fn preview(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.preview";
    let base = target_clip(s, p).and_then(|c| s.active_sequence()?.find_item(c).map(|(_, it)| it.item)).and_then(|i| s.project.narrations.get(&i).cloned());
    let mut q = p.clone();
    if q.get("sample").and_then(Value::as_bool).unwrap_or(false) {
        q["text"] = json!(tts::SAMPLE_SENTENCE);
    }
    let n = settings(cmd, &q, base.as_ref())?;
    let (audio, _, check) = synthesize(s, cmd, &q, &n)?;
    let seconds = audio.seconds();
    let samples = audio.samples.len();
    let rate = audio.sample_rate;
    let mut path = Value::Null;
    if let Some(dest) = str_p(p, "path") {
        let bytes = crate::voiceover::write_wav_f32(&audio.samples, rate);
        s.services.write_file(dest, &bytes).map_err(|e| EngineError::Other(format!("{dest}: {e}")))?;
        path = json!(dest);
    }
    s.tts_preview = Some(audio);
    Ok(
        json!({"seconds": seconds, "samples": samples, "sampleRate": rate, "voice": n.voice, "language": n.language, "path": path, "notes": notes(&n), "check": check}),
    )
}

/// Write `samples` as a new WAV and import it (its own undo step; callers collapse history).
/// `near`: an edit writes next to the narration's previous file unless `dir` is given.
fn write_and_import(s: &mut Session, p: &Value, samples: &[f32], rate: u32, near: Option<ItemId>) -> Result<(ItemId, String)> {
    let beside = near.filter(|_| p.get("dir").is_none()).and_then(|i| match &s.project.items.get(&i)?.kind {
        ItemKind::Media(m) => match &m.media {
            MediaRef::File { path } => std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()).map(|d| d.to_string_lossy().into_owned()),
            MediaRef::Generator(_) => None,
        },
        _ => None,
    });
    let dir = beside.unwrap_or_else(|| crate::voiceover::generated_dir(s, p, "Narrations"));
    let path = crate::voiceover::unique_wav_path(s, &dir, BASE_NAME)?;
    if !cfg!(target_arch = "wasm32") {
        std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
    }
    let bytes = crate::voiceover::write_wav_f32(samples, rate);
    s.services.write_file(&path, &bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    let item = crate::commands::import_bytes(s, &path, bytes.into(), None)?;
    Ok((item, path))
}

fn track_is_free(s: &Session, track: TrackId, range: TimeRange) -> bool {
    s.active_sequence()
        .and_then(|q| q.audio_tracks.iter().find(|t| t.id == track))
        .is_some_and(|t| !t.locked && !t.items.iter().any(|it| it.start < range.end() && range.start < it.end()))
}

/// The track a new narration of `range` goes to (None: add a track).
fn placement(s: &Session, p: &Value, range: TimeRange) -> Result<Option<TrackId>> {
    let cmd = "tts.create";
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    if let Some(v) = p.get("track") {
        let id =
            crate::mixer::strip_ref(seq, v).filter(|id| seq.audio_tracks.iter().any(|t| t.id == *id)).ok_or_else(|| bad(cmd, format!("no audio track {v}")))?;
        if !track_is_free(s, id, range) {
            return Err(bad(cmd, format!("audio track {v} is locked or has clips there")));
        }
        return Ok(Some(id));
    }
    let targeted = s.targeting().targeted;
    let order = seq.audio_tracks.iter().filter(|t| targeted.contains(&t.id)).chain(seq.audio_tracks.iter().filter(|t| !targeted.contains(&t.id)));
    Ok(order.map(|t| t.id).find(|id| track_is_free(s, *id, range)))
}

fn create(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.create";
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let mut n = settings(cmd, p, None)?;
    let at = time_p(s, p, "").unwrap_or(s.playhead());
    if at < Tick::ZERO {
        return Err(bad(cmd, "the start time cannot be negative"));
    }
    let (audio, from_cache, check) = synthesize(s, cmd, p, &n)?;
    let rate = audio.sample_rate.max(1);
    let dur = Tick::from_units(audio.samples.len() as i64, i64::from(rate));
    n.speech_duration = dur;
    let range = TimeRange::new(at, dur);
    let track = placement(s, p, range)?;
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let track = match track {
            Some(t) => t,
            None => {
                let before: Vec<TrackId> = s.active_sequence().map(|q| q.audio_tracks.iter().map(|t| t.id).collect()).unwrap_or_default();
                s.execute("sequence.addTracks", json!({"video": 0, "audio": 1, "audioAfter": "last"}))?;
                s.active_sequence()
                    .and_then(|q| q.audio_tracks.iter().map(|t| t.id).find(|id| !before.contains(id)))
                    .ok_or_else(|| EngineError::Other("could not add an audio track".into()))?
            }
        };
        let (item, path) = write_and_import(s, p, &audio.samples, rate, None)?;
        let seq_rate = s.project.sequence(seq_id).map(|q| q.settings.frame_rate).unwrap_or_default();
        let narration = n.clone();
        let clip = s.edit("New Narration", |pr, st| {
            let mut it = pr
                .make_track_item(item, TrackKind::Audio, at, TimeRange::new(Tick::ZERO, dur), seq_rate)
                .ok_or(EngineError::Other("the narration is not importable".into()))?;
            it.duration = dur;
            let id = it.id;
            let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            let t = seq.audio_tracks.iter_mut().find(|t| t.id == track).ok_or_else(|| bad(cmd, "the narration track was deleted"))?;
            let pos = t.items.partition_point(|x| x.start <= at);
            t.items.insert(pos, it);
            seq.check().map_err(EngineError::Other)?;
            pr.narrations.insert(item, narration);
            st.selection = vec![id];
            Ok(id)
        })?;
        let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
        Ok(json!({
            "item": item.0,
            "clip": clip.0,
            "track": crate::mixer::strip_label(seq, track),
            "start": at.0,
            "duration": dur.0,
            "speechDuration": dur.0,
            "path": path,
            "sampleRate": rate,
            "cached": from_cache,
            "pitchHz": median_pitch(&audio.samples, rate),
            "language": n.language,
            "numbers": n.numbers.id(),
            "notes": notes(&n),
            "check": check,
        }))
    })();
    crate::clip_ops::collapse_history(s, n0, "New Narration");
    if result.is_err() && s.history.undo.len() > n0 {
        // a step failed after the track was added or the file imported: roll back to before
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

/// Samples of media a clip of `duration` at `speed` plays, rounded up (exact for 100 %).
fn needed_samples(duration: Tick, speed: f64, rate: u32) -> usize {
    let rate = i64::from(rate.max(1));
    let media = if speed.is_finite() && speed != 0.0 && (speed.abs() - 1.0).abs() > 1e-12 {
        // speed is a ratio, not edit time: scale the exact tick count
        Tick::from_seconds_f64(duration.seconds() * speed.abs())
    } else {
        duration
    };
    let mut n = media.to_units_floor(rate);
    if Tick::from_units(n, rate) < media {
        n = n.saturating_add(1);
    }
    let cap = (tts::MAX_SECONDS * 2.0) as i64 * rate;
    n.clamp(0, cap) as usize
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.edit";
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let clip = target_clip(s, p).ok_or_else(|| bad(cmd, "select a narration clip (or pass `clip`)"))?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (track, it) = seq.find_item(clip).ok_or_else(|| bad(cmd, "no such clip"))?;
    if seq.audio_tracks.iter().any(|t| t.id == track && t.locked) {
        return Err(bad(cmd, "the clip's track is locked"));
    }
    let (old_item, clip_dur, speed) = (it.item, it.duration, it.speed);
    let base = s.project.narrations.get(&old_item).cloned().ok_or_else(|| bad(cmd, "the clip is not a narration"))?;
    let mut n = settings(cmd, p, Some(&base))?;
    let (audio, _, check) = synthesize(s, cmd, p, &n)?;
    let rate = audio.sample_rate.max(1);
    let speech = audio.samples.len();
    n.speech_duration = Tick::from_units(speech as i64, i64::from(rate));
    // the clip keeps its length: pad the file with silence to what the clip plays
    let needed = needed_samples(clip_dur, speed, rate);
    let mut samples = audio.samples.clone();
    if needed > samples.len() {
        samples.try_reserve(needed - samples.len()).map_err(|e| bad(cmd, format!("unable to allocate the narration: {e}")))?;
        samples.resize(needed, 0.0);
    }
    let file_len = samples.len();
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let (item, path) = write_and_import(s, p, &samples, rate, Some(old_item))?;
        let name = s.project.items.get(&item).map(|i| i.name.clone()).unwrap_or_default();
        let narration = n.clone();
        s.edit("Edit Narration", |pr, _| {
            let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            let (_, it) = seq.find_item_mut(clip).ok_or_else(|| bad(cmd, "the clip was deleted"))?;
            it.item = item;
            it.name = name;
            it.source_in = Tick::ZERO;
            seq.check().map_err(EngineError::Other)?;
            pr.narrations.insert(item, narration);
            Ok(())
        })?;
        Ok(json!({
            "clip": clip.0,
            "item": item.0,
            "previousItem": old_item.0,
            "duration": clip_dur.0,
            "speechDuration": n.speech_duration.0,
            "fileDuration": Tick::from_units(file_len as i64, i64::from(rate)).0,
            "path": path,
            "sampleRate": rate,
            "check": check,
        }))
    })();
    crate::clip_ops::collapse_history(s, n0, "Edit Narration");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.inspect";
    let (item, clip) = match u64_p(p, "item") {
        Some(i) => (ItemId(i), None),
        None => {
            let c = target_clip(s, p).ok_or_else(|| bad(cmd, "pass `item` or `clip`, or select a narration clip"))?;
            let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
            (seq.find_item(c).map(|(_, it)| it.item).ok_or_else(|| bad(cmd, "no such clip"))?, Some(c.0))
        }
    };
    let n = s.project.narrations.get(&item).ok_or_else(|| bad(cmd, "not a narration"))?;
    Ok(json!({"item": item.0, "clip": clip, "narration": n}))
}

pub(crate) fn new_job(s: &mut Session, label: String) -> crate::Job {
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    crate::Job { id, label, progress: Default::default(), result: Default::default() }
}

pub(crate) fn spawn(s: &mut Session, job: crate::Job, wait: bool, name: &str, run: impl FnOnce() + Send + 'static) -> Result<u64> {
    let id = job.id;
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    s.jobs.push(job);
    if wait || cfg!(target_arch = "wasm32") {
        run();
    } else {
        std::thread::Builder::new().name(name.into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    Ok(id)
}

pub(crate) type JobResult = std::sync::Mutex<Option<std::result::Result<filmcraft_export::Report, String>>>;

pub(crate) fn finish(prog: &filmcraft_export::Progress, res: &JobResult, r: std::result::Result<String, String>, t0: web_time::Instant) {
    use std::sync::atomic::Ordering;
    let secs = t0.elapsed().as_secs_f64();
    let r = match r {
        Ok(status) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = status;
            Ok(filmcraft_export::Report { path: String::new(), frames: 0, seconds: secs, bytes: 0, render_fps: 0.0, extra_files: Vec::new() })
        }
        Err(e) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
            *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
            Err(e)
        }
    };
    prog.finished.store(true, Ordering::Relaxed);
    *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
}

/// Synthesize in the background into the caches (memory, and disk when on), so `tts.create` /
/// `tts.edit` / `tts.preview` with the same settings return at once. Already cached (in memory or
/// on disk): no job, `cached: true`.
fn render_job(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.render";
    let base = target_clip(s, p).and_then(|c| s.active_sequence()?.find_item(c).map(|(_, it)| it.item)).and_then(|i| s.project.narrations.get(&i).cloned());
    let mut q = p.clone();
    if q.get("sample").and_then(Value::as_bool).unwrap_or(false) {
        q["text"] = json!(tts::SAMPLE_SENTENCE);
    }
    let n = settings(cmd, &q, base.as_ref())?;
    let models = crate::transcript::models_dir();
    let check = check_for(s, cmd, p, &n)?;
    // memory, then disk: an entry on disk is a quick read, not a job
    if let Some((_, report)) = cached(s, &n, models.as_deref(), &check) {
        return Ok(json!({"job": null, "cached": true, "check": check_json(&n, &check, report.as_ref())}));
    }
    let disk = s.tts_disk.clone();
    let job = new_job(s, "Synthesizing narration".into());
    let (prog, res, cache) = (job.progress.clone(), job.result.clone(), s.tts_cache.clone());
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
    let transcriber = reference_transcriber(s, &n, models.as_deref());
    let id = spawn(s, job, wait, "filmcraft-tts", move || {
        let t0 = web_time::Instant::now();
        *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Synthesizing…".into();
        let strict = match &check {
            Check::Strict(j) => Some(j.strict()),
            _ => None,
        };
        let r = match render(&n, models.as_deref(), transcriber.as_ref(), strict.as_ref()) {
            Ok((a, report)) => {
                let secs = a.seconds();
                let unclear = report.iter().filter(|c| !c.passed).count();
                // keyed after rendering: a Malayalam voice's reference may have been made just now
                keep(&cache, disk.as_ref(), &n, models.as_deref(), &check, &Arc::new(a), &report);
                Ok(match (&check, unclear) {
                    (Check::Strict(_), 0) => format!("{secs:.1} s of speech, every utterance heard clearly"),
                    (Check::Strict(_), k) => format!("{secs:.1} s of speech; {k} utterance(s) never heard clearly (best reading kept)"),
                    (Check::Skipped(why), _) => format!("{secs:.1} s of speech, not checked: {why}"),
                    (Check::Off, _) => format!("{secs:.1} s of speech"),
                })
            }
            Err(tts::TtsError::NotInstalled) => Err("the natural voices are not downloaded yet".to_string()),
            Err(e) => Err(e.to_string()),
        };
        finish(&prog, &res, r, t0);
    })?;
    Ok(json!({"job": id, "cached": false}))
}

/// Download the natural voices (Kokoro-82M, its voices and CMUdict) in the background.
fn download_voices(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.downloadVoices";
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
    if tts::catalog::installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "neural-voices") {
        return Err(bad(cmd, "natural voices are not available in this build (built without the `neural-voices` feature)"));
    }
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
    let job = new_job(s, format!("Downloading {}", tts::catalog::PACKAGE_NAME));
    job.progress.total.store(tts::catalog::missing_bytes(&dir), std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = spawn(s, job, wait, "filmcraft-tts-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_package(&dir, &prog);
        finish(&prog, &res, r.map(|()| "Natural voices installed".to_string()), t0);
    })?;
    Ok(json!({"job": id, "installed": false}))
}

#[cfg(feature = "neural-voices")]
fn download_package(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> =
        tts::catalog::FILES.iter().map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size }).collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&tts::catalog::package_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "neural-voices"))]
fn download_package(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("natural voices are not available in this build".into())
}

fn spec(
    id: &'static str,
    label: &'static str,
    params: &'static str,
    enabled: fn(&Session) -> std::result::Result<(), String>,
    run: fn(&mut Session, &Value) -> Result<Value>,
    journal: bool,
) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec("tts.voices", "List Narration Voices", "{}", always, voices, false),
        spec(
            "tts.preview",
            "Preview Narration",
            r#"{"text":str?,"voice":str?,"language":"en-US|ml-IN"?,"numbers":"english|malayalam"? (Malayalam: how digits are said; english = years, money and numbers from 100 up in English),"pitch":"default"?,"pace":f64?,"delivery":"plain|warm|excited|…"?,"sample":bool?,"clip":id?,"check":"strict|off"? (Malayalam: hear every utterance back and read unclear ones again; strict is the default),"path":str?}"#,
            always,
            preview,
            false,
        ),
        spec(
            "tts.create",
            "New Narration",
            r#"{"text":str,"voice":str?,"language":"en-US|ml-IN"?,"numbers":"english|malayalam"?,"pitch":"extraLow|low|default|high|extraHigh"?,"pace":0.5..2?,"delivery":"plain|conversational|warm|excited|amazed|curious|calm|dramatic"?,"take":0..99?,"time":ticks?,"track":"A1"|id?,"check":"strict|off"?,"dir":str?}"#,
            has_seq,
            create,
            true,
        ),
        spec(
            "tts.edit",
            "Edit Narration",
            r#"{"clip":id?,"text":str?,"voice":str?,"language":"en-US|ml-IN"?,"numbers":"english|malayalam"?,"pitch":str?,"pace":f64?,"delivery":str?,"take":0..99?,"check":"strict|off"?,"dir":str?}"#,
            has_narration_clip,
            edit,
            true,
        ),
        spec("tts.inspect", "Inspect Narration", r#"{"clip":id?,"item":id?}"#, always, inspect, false),
        spec(
            "tts.render",
            "Synthesize Narration",
            r#"{"text":str?,"voice":str?,"language":"en-US|ml-IN"?,"numbers":"english|malayalam"?,"pitch":str?,"pace":f64?,"delivery":str?,"sample":bool?,"clip":id?,"check":"strict|off"?,"wait":bool?}"#,
            always,
            render_job,
            false,
        ),
        spec("tts.downloadVoices", "Download Natural Voices", r#"{"wait":bool?}"#, always, download_voices, false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ear;

    impl tts::heard::Hearer for Ear {
        fn hear(&self, _: &[f32], _: u32) -> std::result::Result<String, String> {
            Ok(String::new())
        }
        fn id(&self) -> String {
            "ear v1".into()
        }
    }

    fn ml() -> Narration {
        settings("t", &json!({"text": "ഇത് ശരിയാണ്.", "voice": "clone:me"}), None).unwrap()
    }

    fn judges(h: Arc<dyn tts::heard::Hearer>) -> Judges {
        Judges {
            hearer: h,
            second: None,
            agreement: tts::heard::Agreement::Both,
            voice: None,
            min_similarity: MIN_VOICE_SIMILARITY,
            policy: tts::heard::Policy::default(),
        }
    }

    struct Gate;

    impl tts::heard::VoiceMatch for Gate {
        fn similarity(&self, _: &[f32], _: u32) -> std::result::Result<f64, String> {
            Ok(0.9)
        }
        fn id(&self) -> String {
            "gate".into()
        }
    }

    #[test]
    fn the_check_is_strict_unless_turned_off() {
        assert!(strict_requested("t", &json!({})).unwrap());
        assert!(strict_requested("t", &json!({"check": null})).unwrap());
        assert!(strict_requested("t", &json!({"check": "strict"})).unwrap());
        assert!(strict_requested("t", &json!({"check": true})).unwrap());
        assert!(!strict_requested("t", &json!({"check": "off"})).unwrap());
        assert!(!strict_requested("t", &json!({"check": false})).unwrap());
        for bad in [json!("loose"), json!(1), json!(["strict"]), json!({})] {
            assert!(strict_requested("t", &json!({"check": bad})).is_err(), "{bad}");
        }
        // cloned voices are checked, in Malayalam and in English; the built-in voices are not
        assert!(checkable(&ml()));
        let en = settings("t", &json!({"text": "Hello.", "voice": "clone:me"}), None).unwrap();
        assert!(checkable(&en));
        assert_eq!((check_language(&ml()), check_language(&en)), ("ml", "en"));
        let basic = settings("t", &json!({"text": "Hello.", "voice": "basic-male"}), None).unwrap();
        assert!(!checkable(&basic));
        let s = Session::default();
        assert!(matches!(check_for(&s, "t", &json!({}), &basic).unwrap(), Check::Off));
        assert!(!matches!(check_for(&s, "t", &json!({}), &en).unwrap(), Check::Off), "English cloned narration is heard back");
        assert!(matches!(check_for(&s, "t", &json!({"check": "off"}), &ml()).unwrap(), Check::Off));
        assert!(check_for(&s, "t", &json!({"check": 3}), &en).is_err(), "a hostile check is refused for English too");
        for bad in
            [json!({"agreement": "most"}), json!({"agreement": 2}), json!({"voiceGate": "yes"}), json!({"minSimilarity": 5}), json!({"minSimilarity": "x"})]
        {
            assert!(check_for(&s, "t", &bad, &ml()).is_err(), "{bad}");
        }
        // the host's recogniser hears, with the host's second opinion and voice gate
        let s = Session {
            narration_hearer: Some(Arc::new(Ear)),
            narration_second: Some(Arc::new(Ear)),
            narration_voice: Some(Arc::new(Gate)),
            ..Session::default()
        };
        let Check::Strict(j) = check_for(&s, "t", &json!({}), &ml()).unwrap() else { panic!("not strict") };
        assert!(j.second.is_some() && j.voice.is_some() && j.agreement == tts::heard::Agreement::Both);
        assert_eq!(j.id(), "ear v1 + ear v1 (both), voice gate >= 0.30");
        assert_eq!(j.first_only().id(), "ear v1", "the earlier check's name: its cache entries are adopted");
        let Check::Strict(j) = check_for(&s, "t", &json!({"agreement": "first", "voiceGate": false}), &ml()).unwrap() else { panic!("not strict") };
        assert!(j.second.is_none() && j.voice.is_none());
        // English reads under the English policy
        let Check::Strict(j) = check_for(&s, "t", &json!({}), &en).unwrap() else { panic!("not strict") };
        assert_eq!(j.policy, tts::heard::Policy::english());
    }

    #[test]
    fn checked_readings_have_their_own_keys_and_reports() {
        let n = ml();
        let strict = Check::Strict(judges(Arc::new(Ear)));
        assert_eq!(memory_key(&n, &Check::Off), cache_key(&n), "unchecked readings keep their keys");
        assert_eq!(memory_key(&n, &Check::Skipped("x".into())), cache_key(&n));
        assert_ne!(memory_key(&n, &strict), cache_key(&n));
        assert!(memory_key(&n, &strict).ends_with("ear v1"));
        // results: English null, off, skipped with its reason, clear or unclear by the utterances
        let en = settings("t", &json!({"text": "Hello.", "voice": "basic-male"}), None).unwrap();
        assert_eq!(check_json(&en, &Check::Off, None), Value::Null);
        assert_eq!(check_json(&n, &Check::Off, None), json!({"status": "off"}));
        assert_eq!(check_json(&n, &Check::Skipped("no model".into()), None), json!({"status": "skipped", "reason": "no model"}));
        let clear = json!([{"passed": true}, {"passed": true}]);
        assert_eq!(check_json(&n, &strict, Some(&clear))["status"], "clear");
        let unclear = json!([{"passed": true}, {"passed": false}]);
        let j = check_json(&n, &strict, Some(&unclear));
        assert_eq!((j["status"].as_str(), j["model"].as_str()), (Some("unclear"), Some("ear v1")));
        assert_eq!(check_json(&n, &strict, None)["status"], "unclear", "a reading without its report is not called clear");
        // the model's check is named without loading it (a cached reading never loads it); a
        // model that cannot load is an error when it would hear, never a crash
        let lazy = LazyJudge { dir: std::env::temp_dir().join("filmcraft-no-such-models"), model: filmcraft_speech::models::MALAYALAM_MODEL, language: "ml" };
        assert_eq!(tts::heard::Hearer::id(&lazy), judge_id());
        assert!(judge_id().starts_with("whisper-medium-ml check v"));
        assert!(tts::heard::Hearer::hear(&lazy, &[0.0; 16_000], 16_000).is_err());
        // reports kept by key, the newest 64
        for i in 0..70 {
            report_put(format!("k{i}"), json!(i));
        }
        assert_eq!(report_get("k69"), Some(json!(69)));
        assert_eq!(report_get("k0"), None);
    }
}

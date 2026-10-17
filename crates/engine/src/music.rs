//! Finding and describing generated music: the soundtrack engine's vocabulary
//! ([`filmcraft_audio_dsp::music::taxonomy`] — mood → genre → tempo → energy → use case) for
//! agents, the planner and the "find music by feel" browser, and the parameters `audio.generate`
//! shares with them.
//!
//! | command | does |
//! |---|---|
//! | `music.suggest` | the styles that fit a brief (`mood`, `genre`, `useCase`, `energy`, `bpm`; synonyms like "suspense" or "chill" are understood), best first, each with how a track of `duration` would be arranged (sections, final hit, ring-out, beats) and the `audio.generate` parameters that make it |
//! | `music.taxonomy` | every mood (with synonyms), genre, use case and style, the stems and the limits |
//!
//! Both are queries: nothing is generated or changed. Hostile parameters (unknown words,
//! non-numbers, out-of-range numbers, curves too long) are refused with an error naming what is
//! accepted.

use serde_json::{Value, json};

use filmcraft_audio_dsp::music::{self, Energy, Genre, Mood, Plan, Request, Stem, Style, taxonomy};
use filmcraft_audio_dsp::synth::MAX_MUSIC_SECONDS;
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, always, bad};
use crate::{Result, Session};

/// Most suggestions returned.
pub const MAX_SUGGESTIONS: usize = 10;

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "music.suggest",
            label: "Suggest Music",
            menu: &[],
            shortcut: None,
            params: r#"{"mood":"epic|tense|uplifting|calm|happy|dark|dreamy|energetic|sad"|synonym?,"genre":"cinematic|electronic|pop|ambient|lofi|piano"?,"useCase":"trailer|vlog|tutorial|…"?,"duration":seconds=60|"durationTicks":ticks?,"energy":0..1|[0..1 ×1..64]?,"bpm":40..200?,"count":1..10=3,"seed":u64=1}"#,
            enabled: always,
            run: suggest,
            journal: false,
        },
        CommandSpec {
            id: "music.taxonomy",
            label: "Music Taxonomy",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: taxonomy_cmd,
            journal: false,
        },
    ]
}

/// A word naming one of `names`, refused with the accepted ones.
fn word<T>(p: &Value, key: &str, cmd: &str, parse: impl Fn(&str) -> Option<T>, names: &[&str]) -> Result<Option<T>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => parse(s).map(Some).ok_or_else(|| bad(cmd, format!("unknown {key} `{s}` ({})", names.join(", ")))),
        Some(v) => Err(bad(cmd, format!("`{key}` must be a word ({}), not {v}", names.join(", ")))),
    }
}

pub(crate) fn mood_p(p: &Value, cmd: &str) -> Result<Option<Mood>> {
    let names: Vec<&str> = Mood::ALL.iter().map(|m| m.id()).collect();
    word(p, "mood", cmd, Mood::parse, &names)
}

pub(crate) fn genre_p(p: &Value, cmd: &str) -> Result<Option<Genre>> {
    let names: Vec<&str> = Genre::ALL.iter().map(|g| g.id()).collect();
    word(p, "genre", cmd, Genre::parse, &names)
}

fn use_case_p(p: &Value, cmd: &str) -> Result<Option<&'static str>> {
    word(p, "useCase", cmd, taxonomy::use_case, &taxonomy::USE_CASES)
}

/// `energy`: a level 0..1, or a curve of 1..64 levels spread over the track.
pub(crate) fn energy_p(p: &Value, cmd: &str) -> Result<Option<Energy>> {
    match p.get("energy") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => match n.as_f64().filter(|e| e.is_finite() && (0.0..=1.0).contains(e)) {
            Some(e) => Ok(Some(Energy::Level(e))),
            None => Err(bad(cmd, format!("`energy` {n} is outside 0..1"))),
        },
        Some(Value::Array(a)) => {
            if a.is_empty() || a.len() > music::arrange::MAX_CURVE_POINTS {
                return Err(bad(cmd, format!("an energy curve has 1 to {} points, not {}", music::arrange::MAX_CURVE_POINTS, a.len())));
            }
            let pts: Vec<f64> = a.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
            Energy::curve(&pts).map(Some).map_err(|e| bad(cmd, e))
        }
        Some(v) => Err(bad(cmd, format!("`energy` must be 0..1 or a list of levels, not {v}"))),
    }
}

/// `bpm`: 40..200 beats a minute.
pub(crate) fn bpm_p(p: &Value, cmd: &str) -> Result<Option<f64>> {
    match p.get("bpm") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|b| b.is_finite() && (music::tempo::MIN_BPM..=music::tempo::MAX_BPM).contains(b))
            .map(Some)
            .ok_or_else(|| bad(cmd, format!("`bpm` must be {}..{}, not {v}", music::tempo::MIN_BPM, music::tempo::MAX_BPM))),
    }
}

/// The length asked for: `durationTicks` (exact), else `duration` seconds; `None` when neither.
/// Zero, negative and non-number lengths are refused; `max` seconds is the most.
pub(crate) fn length_p(p: &Value, cmd: &str, max: f64) -> Result<Option<Tick>> {
    if let Some(v) = p.get("durationTicks").filter(|v| !v.is_null()) {
        let t = v.as_i64().filter(|t| *t > 0).ok_or_else(|| bad(cmd, format!("`durationTicks` must be a positive whole number of ticks, not {v}")))?;
        return Ok(Some(Tick(t).min(Tick::from_seconds_f64(max))));
    }
    match p.get("duration").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            let sec = v.as_f64().filter(|s| s.is_finite() && *s > 0.0).ok_or_else(|| bad(cmd, "`duration` must be more than zero seconds"))?;
            Ok(Some(Tick::from_seconds_f64(sec.min(max))))
        }
    }
}

/// `levels` ({stem: dB | null | "mute"}) and `mute` ([stem]) as engine stem levels.
pub(crate) fn levels_p(p: &Value, cmd: &str) -> Result<[Option<f64>; 4]> {
    let mut levels = [Some(0.0); 4];
    let stem_names: Vec<&str> = Stem::ALL.iter().map(|s| s.id()).collect();
    let stem = |k: &str| Stem::from_id(k).ok_or_else(|| bad(cmd, format!("unknown stem `{k}` ({})", stem_names.join(", "))));
    match p.get("levels") {
        None | Some(Value::Null) => {}
        Some(Value::Object(m)) => {
            for (k, v) in m {
                let s = stem(k)?;
                let slot = levels.get_mut(s.index()).ok_or_else(|| bad(cmd, "no such stem"))?;
                *slot = match v {
                    Value::Null => None,
                    Value::String(w) if w.eq_ignore_ascii_case("mute") => None,
                    _ => Some(
                        v.as_f64()
                            .filter(|db| db.is_finite() && *db <= music::MAX_LEVEL_DB)
                            .ok_or_else(|| bad(cmd, format!("the {k} level must be a number of dB up to {}, or \"mute\"", music::MAX_LEVEL_DB)))?,
                    ),
                };
            }
        }
        Some(v) => return Err(bad(cmd, format!("`levels` must be {{\"drums\": dB, …}}, not {v}"))),
    }
    match p.get("mute") {
        None | Some(Value::Null) => {}
        Some(Value::Array(a)) => {
            for v in a {
                let k = v.as_str().ok_or_else(|| bad(cmd, format!("`mute` lists stems ({}), not {v}", stem_names.join(", "))))?;
                if let Some(slot) = levels.get_mut(stem(k)?.index()) {
                    *slot = None;
                }
            }
        }
        Some(v) => return Err(bad(cmd, format!("`mute` must be a list of stems, not {v}"))),
    }
    if levels.iter().all(Option::is_none) {
        return Err(bad(cmd, "every stem is muted"));
    }
    Ok(levels)
}

fn style_json(s: &Style) -> Value {
    json!({
        "mood": s.mood.id(), "genre": s.genre.id(),
        "bpm": {"min": s.bpm.0, "usual": s.bpm.1, "max": s.bpm.2},
        "energy": s.energy, "useCases": s.use_cases, "feel": s.feel,
    })
}

/// The energy request as JSON (a number or the curve).
fn energy_json(e: &Energy) -> Value {
    match e {
        Energy::Level(l) => json!(l),
        Energy::Curve(c) => json!(c),
    }
}

/// What a planned (or generated) track is: its taxonomy, harmony, sections and exact beat map
/// (ticks from the start of the track).
pub fn track_json(plan: &Plan) -> Value {
    let sections: Vec<Value> = plan
        .sections()
        .iter()
        .map(|s| json!({"kind": s.kind.id(), "start": s.start.0, "bars": s.bars, "energy": (s.energy * 1000.0).round() / 1000.0}))
        .collect();
    let chords: Vec<String> = plan.chords.iter().map(|c| c.name(plan.key)).collect();
    json!({
        "taxonomy": {
            "mood": plan.mood.id(), "genre": plan.genre.id(), "bpm": plan.tempo.bpm(),
            "energy": (plan.energy.nominal() * 1000.0).round() / 1000.0, "useCases": plan.style.use_cases, "feel": plan.style.feel,
        },
        "mood": plan.mood.id(),
        "genre": plan.genre.id(),
        "bpm": plan.tempo.bpm(),
        "key": plan.key_name(),
        "energy": energy_json(&plan.energy),
        "beatsPerBar": music::tempo::BEATS_PER_BAR,
        "bars": plan.arrangement.bars,
        "length": plan.length.0,
        "samples": plan.samples,
        "beats": plan.beats().iter().map(|t| t.0).collect::<Vec<_>>(),
        "downbeats": plan.bars().iter().map(|t| t.0).collect::<Vec<_>>(),
        "sections": sections,
        "chords": chords,
        "ending": plan.ending.0,
        "tail": plan.tail.0,
    })
}

/// The project-item metadata of a generated track: what a "find music by feel" search reads.
pub fn metadata(plan: &Plan) -> Vec<(String, String)> {
    let e = plan.energy.nominal();
    let uses = plan.style.use_cases.join(", ");
    let beat_map = json!({
        "milliBpm": plan.tempo.milli_bpm(), "beatsPerBar": music::tempo::BEATS_PER_BAR, "firstBeat": 0,
        "beats": u64::from(plan.arrangement.bars) * music::tempo::BEATS_PER_BAR + 1, "ending": plan.ending.0, "length": plan.length.0,
    });
    vec![
        (
            "Description".into(),
            format!("{} · {} · {} BPM · {} · energy {e:.2} · {uses}", plan.mood.name(), plan.genre.name(), plan.tempo.bpm(), plan.key_name()),
        ),
        ("Music Mood".into(), plan.mood.name().into()),
        ("Music Genre".into(), plan.genre.name().into()),
        ("Tempo".into(), format!("{}", plan.tempo.bpm())),
        ("Key".into(), plan.key_name()),
        ("Music Energy".into(), format!("{e:.2}")),
        ("Use Case".into(), uses),
        ("Beat Map".into(), beat_map.to_string()),
    ]
}

fn suggest(_s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "music.suggest";
    let mood = mood_p(p, CMD)?;
    let genre = genre_p(p, CMD)?;
    let use_case = use_case_p(p, CMD)?;
    let energy = energy_p(p, CMD)?;
    let bpm = bpm_p(p, CMD)?;
    let length = match length_p(p, CMD, f64::INFINITY)? {
        Some(t) if t.seconds() > MAX_MUSIC_SECONDS + 1e-9 => return Err(bad(CMD, format!("music can be at most {} minutes long", MAX_MUSIC_SECONDS / 60.0))),
        Some(t) if t.seconds() < music::MIN_SECONDS => return Err(bad(CMD, format!("music must be at least {} s long", music::MIN_SECONDS))),
        Some(t) => t,
        None => Tick::from_seconds_f64(60.0),
    };
    let count = match p.get("count").filter(|v| !v.is_null()) {
        None => 3,
        Some(v) => v
            .as_u64()
            .and_then(|c| usize::try_from(c).ok())
            .filter(|c| (1..=MAX_SUGGESTIONS).contains(c))
            .ok_or_else(|| bad(CMD, format!("`count` must be 1..{MAX_SUGGESTIONS}, not {v}")))?,
    };
    let seed = match p.get("seed").filter(|v| !v.is_null()) {
        None => 1,
        Some(v) => v.as_u64().ok_or_else(|| bad(CMD, format!("`seed` must be a whole number from 0, not {v}")))?,
    };
    let q = taxonomy::Query { mood, genre, use_case, energy: energy.as_ref().map(Energy::nominal), bpm };
    let ranked = taxonomy::suggest(&q);
    let mut out = Vec::new();
    for (rank, sug) in ranked.iter().take(count).enumerate() {
        let st = sug.style;
        let req = Request {
            genre: Some(st.genre),
            seed,
            bpm: Some(bpm.unwrap_or(st.bpm.1)),
            energy: Some(energy.clone().unwrap_or(Energy::Level(st.energy))),
            ..Request::new(st.mood, length)
        };
        let plan = music::plan(&req, crate::audio_generate::RATE).map_err(|e| bad(CMD, e.to_string()))?;
        let mut params = json!({
            "kind": "music", "mood": st.mood.id(), "genre": st.genre.id(), "bpm": plan.tempo.bpm(),
            "energy": energy_json(&plan.energy), "durationTicks": length.0, "seed": seed,
        });
        if let Some(obj) = params.as_object_mut() {
            obj.insert("duration".into(), json!(length.seconds()));
        }
        let mut track = track_json(&plan);
        // the beat list is long and follows from the tempo; the sections and ending say enough
        if let Some(obj) = track.as_object_mut() {
            obj.remove("beats");
            obj.remove("downbeats");
            obj.remove("chords");
        }
        out.push(json!({
            "rank": rank + 1, "score": (sug.cost * 1000.0).round() / 1000.0, "why": sug.why,
            "style": style_json(&st), "track": track, "params": params,
        }));
    }
    Ok(json!({
        "suggestions": out,
        "seconds": length.seconds(),
        "vocabulary": {
            "moods": Mood::ALL.iter().map(|m| m.id()).collect::<Vec<_>>(),
            "genres": Genre::ALL.iter().map(|g| g.id()).collect::<Vec<_>>(),
            "useCases": taxonomy::USE_CASES,
        },
    }))
}

fn taxonomy_cmd(_s: &mut Session, _p: &Value) -> Result<Value> {
    Ok(json!({
        "moods": Mood::ALL.iter().map(|m| json!({"id": m.id(), "name": m.name(), "synonyms": m.synonyms(), "minor": m.minor()})).collect::<Vec<_>>(),
        "genres": Genre::ALL.iter().map(|g| json!({"id": g.id(), "name": g.name(), "synonyms": g.synonyms()})).collect::<Vec<_>>(),
        "useCases": taxonomy::USE_CASES,
        "styles": taxonomy::STYLES.iter().map(style_json).collect::<Vec<_>>(),
        "stems": Stem::ALL.iter().map(|s| s.id()).collect::<Vec<_>>(),
        "sections": music::SectionKind::ALL.iter().map(|k| k.id()).collect::<Vec<_>>(),
        "limits": {
            "minSeconds": music::MIN_SECONDS, "maxSeconds": MAX_MUSIC_SECONDS, "maxStemsSeconds": music::MAX_STEMS_SECONDS,
            "maxCurvePoints": music::arrange::MAX_CURVE_POINTS, "bpm": {"min": music::tempo::MIN_BPM, "max": music::tempo::MAX_BPM},
            "levelDb": {"min": music::MIN_LEVEL_DB, "max": music::MAX_LEVEL_DB},
        },
        "instruments": "oscillators",
    }))
}

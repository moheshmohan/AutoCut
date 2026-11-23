//! Clip ▸ Remix: retime a music clip to a target duration by cutting it at musically similar beat
//! boundaries (Premiere's Remix; analysis and planning in [`filmcraft_audio_dsp::remix`], playback
//! in [`filmcraft_render::remix`]).
//!
//! | command | menu | what it does |
//! |---|---|---|
//! | `clip.remix.enable` | Clip ▸ Remix ▸ Enable Remix | analyses the selected audio clip and makes it a remix clip at its current duration |
//! | `clip.remix.properties` | Clip ▸ Remix ▸ Remix Properties… | without parameters returns the settings; with `duration`/`seconds`/`frame`/`timecode`, `segments`, `variations` re-plans |
//! | `clip.remix.revert` | Clip ▸ Remix ▸ Revert Remix | restores the clip as it was before the remix |
//! | `clip.remix` | — | agents: enable (if needed) and remix `clip` (default: the selected audio clip) to a target, optionally fitted to film events, an ending and a stretch |
//! | `media.beats` | — | agents: the tempo, beats, downbeats, sections (and their energy), drops, stops, end and ending of an audio item (the same analysis), with a confidence: to cut picture on the beat and fit music to a film |
//! | `media.pauses` | — | agents: where an audio item's sound pauses (its level `belowDb` under its loudest for at least `minSeconds`), and the stretches of sound between: to cut a narration at a sentence's end |
//!
//! The remix keeps the clip's start and source In, plays the intro from the beginning and ends on
//! the source's end; only the duration changes. The clip's linked partners are not changed. When
//! the remixed clip would run into the next clip on its track the command is refused (nothing is
//! trimmed or moved); remixing shorter always fits. Speed changes and reverse do not apply to a
//! remixed clip (enabling resets them to 100 %).
//!
//! **Fitted to the film** ([`filmcraft_audio_dsp::fit`]): `events [{kind, time|seconds}]`
//! (`burst`, `peak`, `landing`, `stop`, film time from the clip's start) get a real lift of the
//! track (a drop, a rising section start, a section boundary, a downbeat, a stop, the track's own
//! ending) within half a beat, or are reported as missed; `ending` (`own`, `button`, `fade`;
//! default: as the plain remix, which ends with the source) says how the music ends;
//! `maxStretchPct` (0 … 25, default 0) lets the whole remix be time-stretched, pitch kept, to land
//! its end or anchors exactly. Joins sit in stable stretches with the 20 ms equal-power
//! crossfade, and each is rendered and checked: one that changes the level more than 6 dB beyond
//! what the music does at its cut points is refused and the planner tries again without it; a
//! plan that still fails is kept, with a `warnings` entry naming the join. The report adds
//! `eventsLanded`, `joins`, `alternatives` (film lengths that need no join), `ending`, `ended`,
//! `stretchPct`, `fadeOutSeconds` and `warnings`. A fitted remix works with the whole track from
//! the clip's In point (a trimmed clip's track still ends its own way). The fit is stored with
//! the remix, so Remix Properties and the Remix tool keep it when they re-plan.
//!
//! The analysis of a clip (beats, beat self-similarity, structure) is cached per media range and
//! rate, so changing the target or the sliders re-plans instantly. Everything is deterministic.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use filmcraft_audio_dsp::fit;
use filmcraft_audio_dsp::remix::{self as dsp, Analysis, Params};
use filmcraft_project::{ClipId, ItemId, TrackKind};
use filmcraft_render::remix::{Fitted, Remix, TickPiece};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, f64_p, has_seq, u64_p};
use crate::{EngineError, Result, Session};

type Key = (u64, i64, i64, u32);

fn cache() -> &'static Mutex<Vec<(Key, Arc<Analysis>)>> {
    static C: OnceLock<Mutex<Vec<(Key, Arc<Analysis>)>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// The audio clip a menu command works on: the first selected clip on an audio track.
fn selected_audio(s: &Session) -> Option<ClipId> {
    let seq = s.active_sequence()?;
    s.state.selection.iter().copied().find(|c| seq.find_item(*c).and_then(|(t, _)| seq.track(t)).is_some_and(|t| t.kind == TrackKind::Audio))
}

fn check_clip(s: &Session, c: ClipId) -> std::result::Result<(), String> {
    let seq = s.active_sequence().ok_or("no sequence is open")?;
    let (t, it) = seq.find_item(c).ok_or("no such clip")?;
    if seq.track(t).is_none_or(|t| t.kind != TrackKind::Audio) {
        return Err("Remix works on audio clips".into());
    }
    let item = s.project.item(it.item).ok_or("the clip has no media")?;
    if !matches!(item.kind, filmcraft_project::ItemKind::Media(_)) || !item.has_audio() {
        return Err("the clip has no audio media to remix".into());
    }
    Ok(())
}

fn can_enable(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    let c = selected_audio(s).ok_or("select an audio clip")?;
    check_clip(s, c)
}

fn is_remixed(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    let c = selected_audio(s).ok_or("select an audio clip")?;
    let seq = s.active_sequence().ok_or("no sequence is open")?;
    if seq.find_item(c).is_some_and(|(_, i)| filmcraft_render::remix::is_remixed(i)) { Ok(()) } else { Err("the selected clip is not remixed".into()) }
}

fn target_clip(s: &Session, p: &Value, cmd: &str) -> Result<ClipId> {
    let c = match u64_p(p, "clip") {
        Some(c) => ClipId(c),
        None => selected_audio(s).ok_or_else(|| bad(cmd, "select an audio clip or pass `clip`"))?,
    };
    check_clip(s, c).map_err(|e| bad(cmd, e))?;
    Ok(c)
}

/// Target duration from `duration` (ticks), `seconds`, `frame` or `timecode` (a duration).
fn duration_p(s: &Session, p: &Value) -> Option<Tick> {
    if let Some(d) = p.get("duration").and_then(Value::as_i64) {
        return Some(Tick(d));
    }
    let rate = s.sequence_rate();
    if let Some(sec) = f64_p(p, "seconds") {
        return Some(Tick::from_seconds_f64(sec));
    }
    if let Some(f) = p.get("frame").or_else(|| p.get("frames")).and_then(Value::as_i64) {
        return Some(rate.tick_of(f));
    }
    if let Some(tc) = p.get("timecode").and_then(Value::as_str) {
        let df = s.active_sequence().map(|q| q.settings.drop_frame).unwrap_or(false);
        return filmcraft_time::parse_timecode(tc, rate, df, 0).ok().map(|f| rate.tick_of(f));
    }
    None
}

/// Analysis of `[source_in, source_in + dur)` of the clip's media at the sequence rate (cached).
fn analysis(s: &Session, item: ItemId, source_in: Tick, dur: Tick) -> Result<Arc<Analysis>> {
    let sr = s.active_sequence().map(|q| q.settings.sample_rate).unwrap_or(48_000).max(1);
    let m0 = source_in.to_units_floor(sr as i64);
    let len = (source_in + dur).to_units_floor(sr as i64) - m0;
    let max = i64::try_from(u64::from(sr).saturating_mul(dsp::MAX_ANALYSIS_SECONDS)).unwrap_or(i64::MAX);
    if !(0..=max).contains(&len) {
        return Err(EngineError::Other(format!("cannot remix: audio is longer than {} minutes", dsp::MAX_ANALYSIS_SECONDS / 60)));
    }
    let key = (item.0, m0, len, sr);
    if let Some((_, a)) = cache().lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key) {
        return Ok(a.clone());
    }
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let src = filmcraft_render::SourceProvider::source(&provider, item).ok_or_else(|| EngineError::Other("the clip's media is offline".into()))?;
    let buf = src.audio(m0, len as usize, sr).map_err(|e| EngineError::Other(format!("reading the clip's audio: {e}")))?;
    let refs: Vec<&[f32]> = buf.channels.iter().map(Vec::as_slice).collect();
    let a = Arc::new(dsp::analyze(&refs, sr).map_err(|e| EngineError::Other(format!("cannot remix: {e}")))?);
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    c.push((key, a.clone()));
    if c.len() > 8 {
        c.remove(0);
    }
    Ok(a)
}

/// The longest a film event may be from the clip's start, seconds (a day).
const MAX_EVENT_SECONDS: f64 = 86_400.0;

/// The fit a call asks for: `ending`, `events` and `maxStretchPct`, each `None` when not given.
/// Hostile values (an unknown kind or ending, a negative or non-numeric time, more than
/// [`fit::MAX_EVENTS`] events, a stretch outside 0 … [`fit::MAX_STRETCH_PCT`]) are errors.
fn fit_params(p: &Value, cmd: &str) -> Result<(Option<String>, Option<Vec<(String, Tick)>>, Option<f64>)> {
    let given = |k: &str| p.get(k).filter(|v| !v.is_null());
    let ending = match given("ending") {
        None => None,
        Some(v) => match v.as_str().and_then(fit::Ending::parse) {
            Some(fit::Ending::Source) => Some(String::new()),
            Some(e) => Some(e.id().to_string()),
            None => return Err(bad(cmd, "`ending` is \"own\", \"button\", \"fade\" or \"source\"")),
        },
    };
    let events = match given("events") {
        None => None,
        Some(v) => {
            let list = v.as_array().ok_or_else(|| bad(cmd, "`events` is a list of {kind, time|seconds}"))?;
            if list.len() > fit::MAX_EVENTS {
                return Err(bad(cmd, format!("at most {} events", fit::MAX_EVENTS)));
            }
            let mut out = Vec::with_capacity(list.len());
            for (k, e) in list.iter().enumerate() {
                let kind = e
                    .get("kind")
                    .and_then(Value::as_str)
                    .and_then(fit::EventKind::parse)
                    .ok_or_else(|| bad(cmd, format!("event {k}: `kind` is \"burst\", \"peak\", \"landing\" or \"stop\"")))?;
                let at = match (e.get("time").filter(|v| !v.is_null()), e.get("seconds").filter(|v| !v.is_null())) {
                    (Some(t), _) => t.as_i64().filter(|t| (0..=Tick::from_seconds_f64(MAX_EVENT_SECONDS).0).contains(t)).map(Tick),
                    (None, Some(x)) => x.as_f64().filter(|x| x.is_finite() && (0.0..=MAX_EVENT_SECONDS).contains(x)).map(Tick::from_seconds_f64),
                    (None, None) => None,
                };
                let at =
                    at.ok_or_else(|| bad(cmd, format!("event {k}: give `time` (ticks) or `seconds`, from 0 to {MAX_EVENT_SECONDS} s from the clip's start")))?;
                out.push((kind.id().to_string(), at));
            }
            Some(out)
        }
    };
    let stretch = match given("maxStretchPct") {
        None => None,
        Some(v) => Some(
            v.as_f64()
                .filter(|x| x.is_finite() && (0.0..=fit::MAX_STRETCH_PCT).contains(x))
                .ok_or_else(|| bad(cmd, format!("`maxStretchPct` is a number from 0 to {}", fit::MAX_STRETCH_PCT)))?,
        ),
    };
    Ok((ending, events, stretch))
}

/// Reads the analysed range of `item` from `m0` (samples at `sr`, `len` long) for the join check:
/// stereo, silence outside the range.
fn range_reader<'a>(src: &'a dyn filmcraft_media::MediaSource, m0: i64, len: i64, sr: u32) -> impl FnMut(i64, usize) -> Vec<Vec<f32>> + 'a {
    move |pos: i64, n: usize| {
        let mut out = vec![vec![0f32; n]; 2];
        let lo = pos.max(0);
        let hi = pos.saturating_add(n as i64).min(len);
        if hi > lo
            && let Ok(b) = src.audio(m0 + lo, (hi - lo) as usize, sr)
        {
            let off = (lo - pos) as usize;
            for (c, dst) in out.iter_mut().enumerate() {
                let Some(ch) = b.channels.get(c).or_else(|| b.channels.first()) else { continue };
                for (d, v) in dst.iter_mut().skip(off).zip(ch) {
                    *d = if v.is_finite() { *v } else { 0.0 };
                }
            }
        }
        out
    }
}

/// The remix of `c` to `target` with the given sliders and fit: (new duration, state, analysis,
/// the fit).
fn make_plan(s: &Session, c: ClipId, target: Tick, segments: f64, variations: f64, ask: &Fitted) -> Result<(Tick, Remix, Arc<Analysis>, fit::Fit)> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let sr = seq.settings.sample_rate.max(1) as i64;
    let (_, it) = seq.find_item(c).ok_or_else(|| EngineError::Other("no such clip".into()))?;
    let old = filmcraft_render::remix::Remix::of(it);
    let original = old.as_ref().map(|r| r.original).unwrap_or(it.duration);
    // a plain remix works with the clip's music; a fitted one with the whole track from the
    // clip's In point (its own ending is at the track's end, wherever the clip was trimmed)
    let range = if ask.is_plain() {
        original
    } else {
        let media = s.project.item(it.item).map_or(original, |i| i.duration());
        let most = Tick::from_seconds_f64(dsp::MAX_ANALYSIS_SECONDS as f64);
        (media - it.source_in).max(original).min(most.max(original))
    };
    let a = analysis(s, it.item, it.source_in, range)?;
    let params = Params { segments: segments.clamp(0.0, 100.0), variations: variations.clamp(0.0, 100.0) };
    let events = ask.events.iter().filter_map(|(k, t)| Some(fit::Event { kind: fit::EventKind::parse(k)?, at: t.to_units_floor(sr) })).collect();
    let ending = fit::Ending::parse(&ask.ending).unwrap_or_default();
    let req = fit::Request { target: target.to_units_floor(sr), params, events, ending, max_stretch: ask.max_stretch_pct / 100.0 };
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let src = filmcraft_render::SourceProvider::source(&provider, it.item).ok_or_else(|| EngineError::Other("the clip's media is offline".into()))?;
    let mut read = range_reader(src.as_ref(), it.source_in.to_units_floor(sr), a.len, sr as u32);
    let f = fit::fit(&a, &req, &mut read).map_err(|e| EngineError::Other(format!("cannot remix: {e}")))?;
    let plan = &f.plan;
    let pieces = plan.pieces.iter().map(|p| TickPiece { src: it.source_in + Tick::from_units(p.src, sr), len: Tick::from_units(p.len, sr) }).collect();
    let rm = Remix {
        target,
        segments: params.segments,
        variations: params.variations,
        original,
        xfade: Tick::from_units(plan.xfade, sr),
        pieces,
        fade_out: Tick::from_units(plan.fade_out, sr),
        stretch: f.stretch,
        fitted: ask.clone(),
    };
    Ok((Tick::from_units(f.len, sr), rm, a, f))
}

/// Apply `rm` to clip `c` with duration `dur` (one undo step); refuses overlaps.
fn apply(s: &mut Session, label: &str, c: ClipId, dur: Tick, rm: Option<Remix>) -> Result<()> {
    s.edit_sequence(label, move |q, _, _| {
        let (tid, it) = q.find_item(c).ok_or_else(|| EngineError::Other("no such clip".into()))?;
        let start = it.start;
        let tr = q.track(tid).ok_or_else(|| EngineError::Other("no such track".into()))?;
        if let Some(next) = tr.items.iter().filter(|i| i.id != c && i.start >= start).map(|i| i.start).min()
            && start + dur > next
        {
            return Err(EngineError::Other(format!("not enough room: the remixed clip ({:.2} s) would overlap the next clip on {}", dur.seconds(), tr.name)));
        }
        let (_, it) = q.find_item_mut(c).ok_or_else(|| EngineError::Other("no such clip".into()))?;
        it.duration = dur;
        match rm {
            Some(r) => {
                it.speed = 1.0;
                it.reverse = false;
                r.store(it);
            }
            None => {
                filmcraft_render::remix::take(it);
            }
        }
        if let Some(t) = q.track_mut(tid) {
            t.sort();
        }
        Ok(())
    })
}

/// What a fit did, for the report: times of the clip in seconds, of the track in seconds of the
/// media.
fn fit_report(f: &fit::Fit, sr: i64, source_in: Tick) -> Value {
    let sec = |x: i64| ((Tick::from_units(x, sr)).seconds() * 1000.0).round() / 1000.0;
    let media = |x: i64| ((source_in + Tick::from_units(x, sr)).seconds() * 1000.0).round() / 1000.0;
    let s = if f.stretch.is_finite() && f.stretch > 0.0 { f.stretch } else { 1.0 };
    json!({
        "ending": f.ending.id(),
        "ended": f.ended,
        "stretchPct": ((1.0 / s - 1.0) * 10_000.0).round() / 100.0,
        "fadeOutSeconds": sec((f.plan.fade_out as f64 / s).round() as i64),
        "eventsLanded": f.events.iter().map(|e| json!({
            "kind": e.kind.id(),
            "at": sec(e.at),
            "lift": e.lift.map(fit::LiftKind::id),
            "playsAt": e.plays_at.map(sec),
            "offBy": e.off_by.map(sec),
            "landed": e.landed,
        })).collect::<Vec<_>>(),
        "joins": f.joins.iter().map(|j| json!({
            "at": sec((j.at as f64 / s).round() as i64),
            "out": media(j.out),
            "in": media(j.into),
            "score": j.score.map(|x| (f64::from(x) * 100.0).round() / 100.0),
            "levelChangeDb": j.change_db,
            "excessDb": j.excess_db,
            "stable": j.stable,
            "keepsBars": j.keeps_bars,
            "ok": j.ok,
        })).collect::<Vec<_>>(),
        "alternatives": f.alternatives.iter().map(|x| {
            let mut v = json!({"duration": Tick::from_units(x.len, sr).0, "seconds": sec(x.len), "joins": x.joins, "ending": x.ending.id()});
            if x.min != x.len || x.max != x.len {
                v["minSeconds"] = json!(sec(x.min));
                v["maxSeconds"] = json!(sec(x.max));
            }
            v
        }).collect::<Vec<_>>(),
        "warnings": f.warnings,
    })
}

fn report(s: &Session, c: ClipId, a: Option<&Analysis>) -> Value {
    let Some(seq) = s.active_sequence() else { return Value::Null };
    let Some((_, it)) = seq.find_item(c) else { return Value::Null };
    let rm = Remix::of(it);
    let mut v = json!({
        "clip": c.0,
        "remixed": rm.is_some(),
        "duration": it.duration.0,
        "seconds": it.duration.seconds(),
    });
    if let Some(r) = rm {
        let cuts: Vec<Value> = r.pieces.windows(2).map(|w| json!({"out": (w[0].src + w[0].len).0, "in": w[1].src.0})).collect();
        v["target"] = json!(r.target.0);
        v["targetSeconds"] = json!(r.target.seconds());
        v["originalSeconds"] = json!(r.original.seconds());
        v["segments"] = json!(r.segments);
        v["variations"] = json!(r.variations);
        v["pieces"] = Value::Array(r.pieces.iter().map(|p| json!({"src": p.src.0, "len": p.len.0})).collect());
        v["cuts"] = Value::Array(cuts);
        v["errorSeconds"] = json!((it.duration - r.target).seconds());
    }
    if let Some(a) = a {
        v["bpm"] = json!(a.bpm);
        v["beats"] = json!(a.beats.len());
    }
    v
}

/// [`report`] with what the fit did.
fn fit_reported(s: &Session, c: ClipId, a: &Analysis, f: &fit::Fit) -> Value {
    let mut v = report(s, c, Some(a));
    let sr = s.active_sequence().map(|q| q.settings.sample_rate).unwrap_or(48_000).max(1) as i64;
    let source_in = s.active_sequence().and_then(|q| q.find_item(c)).map_or(Tick::ZERO, |(_, it)| it.source_in);
    if let (Some(o), Value::Object(extra)) = (v.as_object_mut(), fit_report(f, sr, source_in)) {
        o.extend(extra);
    }
    v
}

fn enable(s: &mut Session, p: &Value) -> Result<Value> {
    let c = target_clip(s, p, "clip.remix.enable")?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(c).ok_or_else(|| bad("clip.remix.enable", "no such clip"))?;
    if filmcraft_render::remix::is_remixed(it) {
        return Ok(report(s, c, None));
    }
    let dur = it.duration;
    let (d, rm, a, f) = make_plan(s, c, dur, 50.0, 50.0, &Fitted::default())?;
    apply(s, "Enable Remix", c, d, Some(rm))?;
    Ok(fit_reported(s, c, &a, &f))
}

fn remix(s: &mut Session, p: &Value, cmd: &str, label: &str) -> Result<Value> {
    let c = target_clip(s, p, cmd)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(c).ok_or_else(|| bad(cmd, "no such clip"))?;
    let cur = Remix::of(it);
    let target = duration_p(s, p);
    let segments = f64_p(p, "segments");
    let variations = f64_p(p, "variations");
    let (ending, events, stretch) = fit_params(p, cmd)?;
    if target.is_none() && segments.is_none() && variations.is_none() && ending.is_none() && events.is_none() && stretch.is_none() {
        if cmd == "clip.remix" {
            return Err(bad(cmd, "need a target `duration` (ticks), `seconds`, `frame` or `timecode`"));
        }
        return Ok(report(s, c, None));
    }
    if cmd == "clip.remix" && target.is_none() {
        return Err(bad(cmd, "need a target `duration` (ticks), `seconds`, `frame` or `timecode`"));
    }
    let target = target.or(cur.as_ref().map(|r| r.target)).unwrap_or(it.duration);
    if target.0 <= 0 {
        return Err(bad(cmd, "the target duration must be positive"));
    }
    let segments = segments.or(cur.as_ref().map(|r| r.segments)).unwrap_or(50.0);
    let variations = variations.or(cur.as_ref().map(|r| r.variations)).unwrap_or(50.0);
    // what is not given is kept from the clip's remix (a re-plan keeps its fit)
    let kept = cur.as_ref().map(|r| r.fitted.clone()).unwrap_or_default();
    let ask = Fitted { ending: ending.unwrap_or(kept.ending), events: events.unwrap_or(kept.events), max_stretch_pct: stretch.unwrap_or(kept.max_stretch_pct) };
    let (d, rm, a, f) = make_plan(s, c, target, segments, variations, &ask)?;
    apply(s, label, c, d, Some(rm))?;
    Ok(fit_reported(s, c, &a, &f))
}

fn revert(s: &mut Session, p: &Value) -> Result<Value> {
    let c = target_clip(s, p, "clip.remix.revert")?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(c).ok_or_else(|| bad("clip.remix.revert", "no such clip"))?;
    let rm = Remix::of(it).ok_or_else(|| bad("clip.remix.revert", "the clip is not remixed"))?;
    apply(s, "Revert Remix", c, rm.original, None)?;
    Ok(report(s, c, None))
}

/// Longest stretch of an item `media.beats` analyses, seconds (the analysis holds the whole
/// stretch in memory, and its beat similarity grows with the square of the beats).
pub const MAX_BEATS_SECONDS: f64 = 20.0 * 60.0;

/// `media.beats`: the tempo, beats and structure of `item`'s sound over `[sourceIn, sourceIn +
/// duration)` (default: from the start, the whole item up to [`MAX_BEATS_SECONDS`]), times in
/// seconds of the media ([`filmcraft_audio_dsp::structure`]):
///
/// | field | what |
/// |---|---|
/// | `bpm`, `period` | the tempo, and seconds per beat |
/// | `beats` | the attack of each beat |
/// | `meter`, `downbeats` | beats in a bar (4 or 3), and the attack of each bar's first beat |
/// | `sections` | `[{start, end, energy}]`: stretches of like music between boundaries on downbeats; `energy` is its loudness against the loudest section's (1), halving every 10 dB |
/// | `drops` | downbeats where the music lifts into its loud part (the two bars after ≥ 6 dB louder than the two before) |
/// | `stops` | where the sound falls away abruptly (20 dB within 50 ms) |
/// | `musicStart`, `musicEnd` | where the music's sound starts and ends (within 45 dB of its loudest) |
/// | `ending` | `{kind, at}`: `cadence` (a final hit at `at` ringing out), `fade` (fading from `at`) or `cut` (stopping at `at`); `null` without one |
/// | `confidence`, `confidenceParts` | 0 … 1: overall, and of the beats, downbeats and sections |
fn beats(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "media.beats";
    let item = crate::commands::item_p(p, "item").ok_or_else(|| bad(CMD, "need `item`"))?;
    let it = s.project.item(item).ok_or_else(|| bad(CMD, "no such item"))?;
    if !matches!(it.kind, filmcraft_project::ItemKind::Media(_)) || !it.has_audio() {
        return Err(bad(CMD, "the item has no audio media"));
    }
    let length = it.duration();
    let from = Tick(p.get("sourceIn").and_then(Value::as_i64).unwrap_or(0).clamp(0, length.0.max(0)));
    let most = Tick::from_seconds_f64(MAX_BEATS_SECONDS);
    let dur = Tick(p.get("duration").and_then(Value::as_i64).unwrap_or(i64::MAX).clamp(0, (length - from).0.max(0)).min(most.0));
    if dur.0 <= 0 {
        return Err(bad(CMD, "nothing to analyse in that range"));
    }
    let a = analysis(s, item, from, dur)?;
    let sr = i64::from(a.sample_rate.max(1));
    let lead = dsp::beat_lead(a.sample_rate);
    let at = |smp: i64| (from + Tick::from_units(smp.saturating_add(lead), sr)).seconds();
    // structure positions are true sample positions (not beat boundaries), rounded to the ms
    let ms = |x: f64| (x * 1000.0).round() / 1000.0;
    let true_at = |smp: i64| ms((from + Tick::from_units(smp, sr)).seconds());
    let st = &a.structure;
    let beat_at = |i: usize| a.beats.get(i).map(|b| ms(at(*b)));
    let ending = st.ending.map_or(Value::Null, |e| json!({"kind": e.kind.id(), "at": true_at(e.at)}));
    Ok(json!({
        "item": item.0,
        "bpm": a.bpm,
        "period": a.period / sr as f64,
        "beats": a.beats.iter().map(|b| at(*b)).collect::<Vec<_>>(),
        "seconds": dur.seconds(),
        "meter": st.meter,
        "downbeats": st.downbeats.iter().filter_map(|&i| beat_at(i)).collect::<Vec<_>>(),
        "sections": st.sections.iter().map(|x| json!({"start": true_at(x.start), "end": true_at(x.end), "energy": x.energy})).collect::<Vec<_>>(),
        "drops": st.drops.iter().filter_map(|d| beat_at(d.beat)).collect::<Vec<_>>(),
        "stops": st.stops.iter().map(|x| true_at(*x)).collect::<Vec<_>>(),
        "musicStart": true_at(st.music_start),
        "musicEnd": true_at(st.music_end),
        "ending": ending,
        "confidence": st.confidence.overall,
        "confidenceParts": {"beats": st.confidence.beats, "downbeats": st.confidence.downbeats, "sections": st.confidence.sections},
    }))
}

/// Longest stretch of an item `media.pauses` reads, seconds (held in memory at
/// [`PAUSES_RATE`]).
pub const MAX_PAUSES_SECONDS: f64 = 10.0 * 60.0;
/// The rate `media.pauses` reads the sound at: a level envelope needs no more.
const PAUSES_RATE: u32 = 16_000;
/// Its envelope's step and window, seconds.
const PAUSES_HOP: f32 = 0.01;
const PAUSES_WINDOW: f32 = 0.05;

/// `media.pauses`: where `item`'s sound pauses over `[sourceIn, sourceIn + duration)` (default:
/// the whole item, up to [`MAX_PAUSES_SECONDS`]): `pauses` and the stretches of `sound` between
/// them, `{start, end}` in seconds of the media. A pause is at least `minSeconds` (0.25) where
/// the level (50 ms RMS) stays more than `belowDb` (30) under the loudest of the stretch; the
/// silence before the first sound and after the last is not one.
fn pauses(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "media.pauses";
    let item = crate::commands::item_p(p, "item").ok_or_else(|| bad(CMD, "need `item`"))?;
    let it = s.project.item(item).ok_or_else(|| bad(CMD, "no such item"))?;
    if !matches!(it.kind, filmcraft_project::ItemKind::Media(_)) || !it.has_audio() {
        return Err(bad(CMD, "the item has no audio media"));
    }
    let length = it.duration();
    let from = Tick(p.get("sourceIn").and_then(Value::as_i64).unwrap_or(0).clamp(0, length.0.max(0)));
    let most = Tick::from_seconds_f64(MAX_PAUSES_SECONDS);
    let dur = Tick(p.get("duration").and_then(Value::as_i64).unwrap_or(i64::MAX).clamp(0, (length - from).0.max(0)).min(most.0));
    if dur.0 <= 0 {
        return Err(bad(CMD, "nothing to read in that range"));
    }
    let num = |k: &str, lo: f64, hi: f64, default: f64| -> Result<f64> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|f| f.is_finite() && (lo..=hi).contains(f)).ok_or_else(|| bad(CMD, format!("`{k}` is a number from {lo} to {hi}"))),
        }
    };
    let min_pause = num("minSeconds", 0.05, 10.0, 0.25)?;
    let below = num("belowDb", 6.0, 80.0, 30.0)?;
    let sr = PAUSES_RATE;
    let m0 = from.to_units_floor(i64::from(sr));
    let len = usize::try_from(((from + dur).to_units_floor(i64::from(sr)) - m0).max(0)).unwrap_or(0);
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let src = filmcraft_render::SourceProvider::source(&provider, item).ok_or_else(|| EngineError::Other("the item's media is offline".into()))?;
    let buf = src.audio(m0, len, sr).map_err(|e| EngineError::Other(format!("reading the item's sound: {e}")))?;
    let refs: Vec<&[f32]> = buf.channels.iter().map(Vec::as_slice).collect();
    let env = filmcraft_audio_dsp::ducking::envelope_db(&refs, sr as f32, PAUSES_HOP, PAUSES_WINDOW);
    let loudest = env.iter().copied().fold(filmcraft_audio_dsp::ducking::FLOOR_DB, f32::max);
    let at = |x: f64| ((from.seconds() + x) * 1000.0).round() / 1000.0;
    if loudest <= filmcraft_audio_dsp::ducking::FLOOR_DB + 1.0 {
        return Ok(json!({"item": item.0, "pauses": [], "sound": [], "seconds": dur.seconds(), "silent": true}));
    }
    let threshold = (loudest - below as f32).max(-90.0);
    // sound: above the threshold, gaps shorter than a pause bridged, blips under 60 ms dropped
    let sound = filmcraft_audio_dsp::ducking::activity(&env, PAUSES_HOP, threshold, 0.06, min_pause as f32);
    let pauses: Vec<Value> = sound.windows(2).filter_map(|w| Some((w.first()?.1, w.get(1)?.0))).map(|(a, b)| json!({"start": at(a), "end": at(b)})).collect();
    Ok(json!({
        "item": item.0,
        "pauses": pauses,
        "sound": sound.iter().map(|(a, b)| json!({"start": at(*a), "end": at(*b)})).collect::<Vec<_>>(),
        "seconds": dur.seconds(),
        "loudestDb": (f64::from(loudest) * 10.0).round() / 10.0,
    }))
}

/// The tempo and beats of `item`'s sound over `[from, from + dur)`: (bpm, the attack of each
/// beat in media time). The same analysis as `media.beats`, in exact ticks (video templates
/// re-quantize their cuts to them).
pub(crate) fn beat_ticks(s: &Session, item: ItemId, from: Tick, dur: Tick) -> Result<(f64, Vec<Tick>)> {
    let most = Tick::from_seconds_f64(MAX_BEATS_SECONDS);
    let a = analysis(s, item, from, dur.min(most))?;
    let sr = i64::from(a.sample_rate.max(1));
    let lead = dsp::beat_lead(a.sample_rate);
    Ok((a.bpm, a.beats.iter().map(|b| from + Tick::from_units(b.saturating_add(lead), sr)).collect()))
}

/// Remix state of every remixed clip in the active sequence (for agents and the UI).
pub fn remixed_clips(s: &Session) -> HashMap<ClipId, Remix> {
    let mut out = HashMap::new();
    if let Some(q) = s.active_sequence() {
        for t in &q.audio_tracks {
            for i in &t.items {
                if let Some(r) = Remix::of(i) {
                    out.insert(i.id, r);
                }
            }
        }
    }
    out
}

pub fn commands() -> Vec<CommandSpec> {
    const PARAMS: &str = r#"{"clip":id?,"duration":ticks?,"seconds":f64?,"frame":n?,"timecode":str?,"segments":0..100?,"variations":0..100?,"events":[{"kind":"burst"|"peak"|"landing"|"stop","time":ticks?,"seconds":f64?}]?,"ending":"own"|"button"|"fade"|"source"?,"maxStretchPct":0..25=0}"#;
    vec![
        CommandSpec {
            id: "clip.remix.enable",
            label: "Enable Remix",
            menu: &["Clip", "Remix"],
            shortcut: None,
            params: r#"{"clip":id?}"#,
            enabled: can_enable,
            run: enable,
            journal: true,
        },
        CommandSpec {
            id: "clip.remix.properties",
            label: "Remix Properties…",
            menu: &["Clip", "Remix"],
            shortcut: None,
            params: PARAMS,
            enabled: is_remixed,
            run: |s, p| remix(s, p, "clip.remix.properties", "Remix Properties"),
            journal: true,
        },
        CommandSpec {
            id: "clip.remix.revert",
            label: "Revert Remix",
            menu: &["Clip", "Remix"],
            shortcut: None,
            params: r#"{"clip":id?}"#,
            enabled: is_remixed,
            run: revert,
            journal: true,
        },
        CommandSpec {
            id: "clip.remix",
            label: "Remix",
            menu: &[],
            shortcut: None,
            params: PARAMS,
            enabled: has_seq,
            run: |s, p| remix(s, p, "clip.remix", "Remix"),
            journal: true,
        },
        CommandSpec {
            id: "media.beats",
            label: "Beats of Media",
            menu: &[],
            shortcut: None,
            params: r#"{"item":id,"sourceIn":ticks?,"duration":ticks?}"#,
            enabled: crate::commands::always,
            run: beats,
            journal: false,
        },
        CommandSpec {
            id: "media.pauses",
            label: "Pauses in Media",
            menu: &[],
            shortcut: None,
            params: r#"{"item":id,"sourceIn":ticks?,"duration":ticks?,"minSeconds":0.05..10=0.25,"belowDb":6..80=30}"#,
            enabled: crate::commands::always,
            run: pauses,
            journal: false,
        },
    ]
}

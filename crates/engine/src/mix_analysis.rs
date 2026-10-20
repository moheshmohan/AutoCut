//! `audio.analyzeMix`: how a sequence's mix will hold up on small speakers, measured per stem
//! ([`filmcraft_audio_dsp::mixcheck`]).
//!
//! The mix is rendered through the mixer graph as it plays, and again with every limiter on the
//! Mix switched off; each audio track is rendered alone (soloed, the Mix's limiters off), and the
//! sounds coupled to video transitions alone. From those:
//!
//! | field | what |
//! |---|---|
//! | `mix` | integrated loudness, true peak, loudness range and band shares of the mix as it plays |
//! | `preLimiter` | the same before the Mix's limiters, and `limiter`: the gain reduction they apply (largest, 99th percentile, seconds over 1 and 2 dB) |
//! | `stems` | per track (`A1`…, and `transitions`): its role (from its clips' audio types), loudness, true peak, band shares, and the gain reduction of a limiter on the track (`limiter`) |
//! | `speech` | where the speech tracks (Dialogue clips) speak: their loudness, the mix's loudness then, and how far they stand above the music (`toMusic`) and above everything else (`toRest`) — median and quietest tenth of the momentary (400 ms) loudness difference |
//! | `narrationClips` | each speech clip's own signal (clip gain and effects): peak, samples over full scale, flat tops (a clipped source) |
//!
//! Band shares are the energy below 120 Hz (`sub`), 120–500 Hz (`lowMid`), 500 Hz–1 kHz (`mid`),
//! 1–4 kHz (`presence`), 4–6 kHz (`upper`) and above 6 kHz (`air`), fractions of the whole;
//! `octavesDb` the level of each octave band (31.5 Hz…16 kHz) relative to the whole, dB.
//! Nothing is changed; the project's own sequence is never touched (the passes play copies).
//!
//! | command | does |
//! |---|---|
//! | `audio.analyzeMix {sequence?, from?, to?, speech?, music?}` | `from`/`to`: seconds (default: all of it); `speech`, `music`: track labels (`["A2"]`) instead of the clips' audio types |

use serde_json::{Value, json};

use filmcraft_audio_dsp::LoudnessMeter;
use filmcraft_audio_dsp::mixcheck::{self, BAND_NAMES, BandMeter, Bands, ClipMeter, GainReductionMeter, Ratio};
use filmcraft_project::essential::AudioType;
use filmcraft_project::{ItemId, Project, Sequence, Track};
use filmcraft_render::SourceProvider;
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, bad, has_seq, u64_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "audio.analyzeMix";
/// Longest stretch analysed (seconds): four hours, as loudness normalisation.
pub const MAX_SECONDS: f64 = crate::loudness::MAX_SECONDS;
/// Effects on the Mix that limit (switched off for the "before the limiter" pass).
const LIMITERS: &[&str] = &["hard_limiter"];
/// Loudness points every 100 ms.
const HOP_S: f64 = 0.1;

/// What a track carries, from its clips' audio types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Speech,
    Music,
    Effects,
    Ambience,
    Other,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Speech => "speech",
            Role::Music => "music",
            Role::Effects => "effects",
            Role::Ambience => "ambience",
            Role::Other => "other",
        }
    }
}

/// The role of `track`: the audio type most of its enabled clips have (untyped clips count for
/// nothing; a track of untyped clips is `Other`).
pub fn track_role(track: &Track) -> Role {
    let mut n = [0usize; 4];
    for it in track.items.iter().filter(|i| i.enabled) {
        if let Some(e) = &it.essential {
            let k = match e.kind {
                AudioType::Dialogue => 0,
                AudioType::Music => 1,
                AudioType::Sfx => 2,
                AudioType::Ambience => 3,
            };
            if let Some(c) = n.get_mut(k) {
                *c += 1;
            }
        }
    }
    match n.iter().enumerate().filter(|(_, c)| **c > 0).max_by_key(|(_, c)| **c).map(|(k, _)| k) {
        Some(0) => Role::Speech,
        Some(1) => Role::Music,
        Some(2) => Role::Effects,
        Some(3) => Role::Ambience,
        _ => Role::Other,
    }
}

/// One rendering of (a variant of) the sequence, metered.
struct Metered {
    loud: LoudnessMeter,
    bands: BandMeter,
    /// Momentary loudness every 100 ms.
    momentary: Vec<f64>,
}

impl Metered {
    fn new(sr: u32) -> Self {
        Self { loud: LoudnessMeter::new(f64::from(sr), 2), bands: BandMeter::new(f64::from(sr), 2), momentary: Vec::new() }
    }
    /// Feed 100 ms sub-blocks (the last may be shorter).
    fn feed(&mut self, l: &[f32], r: &[f32], sub: usize) {
        for (a, b) in l.chunks(sub).zip(r.chunks(sub)) {
            self.loud.process(&[a, b]);
            self.bands.process(&[a, b]);
            if a.len() == sub {
                self.momentary.push(self.loud.momentary());
            }
        }
    }
    fn json(&self) -> Value {
        json!({
            "integratedLufs": num(self.loud.integrated()),
            "truePeakDbtp": num(self.loud.true_peak_dbtp()),
            "samplePeakDbfs": num(20.0 * self.loud.sample_peak().log10()),
            "loudnessRangeLu": num(self.loud.loudness_range()),
            "bands": bands_json(&self.bands.bands()),
            "octavesDb": octaves_json(&self.bands.bands()),
        })
    }
}

/// A finite number rounded to 0.01, else null.
fn num(x: f64) -> Value {
    if x.is_finite() { json!((x * 100.0).round() / 100.0) } else { Value::Null }
}

/// Band shares rounded to 0.001.
pub fn bands_json(b: &Bands) -> Value {
    Value::Object(BAND_NAMES.iter().zip(b.shares).map(|(n, s)| (n.to_string(), json!((s * 1000.0).round() / 1000.0))).collect())
}

/// Octave-band levels relative to the whole, dB rounded to 0.1, keyed by centre frequency.
pub fn octaves_json(b: &Bands) -> Value {
    Value::Object(
        mixcheck::OCTAVES
            .iter()
            .zip(b.octaves_db)
            .map(|(c, d)| (c.to_string(), if d.is_finite() { json!((d * 10.0).round() / 10.0) } else { Value::Null }))
            .collect(),
    )
}

fn ratio_json(r: Option<Ratio>) -> Value {
    match r {
        Some(r) => json!({"medianLu": num(r.median_lu), "p10Lu": num(r.p10_lu), "activeSeconds": num(r.active_seconds)}),
        None => Value::Null,
    }
}

/// Render `q` over samples `[a0, a1)` and feed `f` each stereo block (1 s at a time).
fn render(
    project: &Project,
    q: &Sequence,
    a0: i64,
    a1: i64,
    sources: &dyn SourceProvider,
    f: &mut dyn FnMut(&[f32], &[f32]),
) -> std::result::Result<(), String> {
    let sr = i64::from(q.settings.sample_rate.max(1));
    let mut pos = a0;
    while pos < a1 {
        let n = usize::try_from((a1 - pos).min(sr)).map_err(|_| "internal: block size")?;
        let b = filmcraft_render::audio::mix_sequence(project, q, pos, n, sources);
        let (Some(l), Some(r)) = (b.channels.first(), b.channels.get(1).or(b.channels.first())) else {
            return Err("the mixer rendered no channels".into());
        };
        f(l, r);
        pos += n as i64;
    }
    Ok(())
}

/// Render `a` over samples `[a0, a1)`, and `b` (a variant of it, e.g. without a limiter) beside
/// it, one second at a time: `f` gets each block of both, time-aligned. Two variants whose
/// mixer graphs differ (an insert more or less) keep their own state between blocks.
#[allow(clippy::type_complexity)]
fn render_pair(
    project: &Project,
    a: &Sequence,
    b: Option<&Sequence>,
    range: (i64, i64),
    sources: &dyn SourceProvider,
    f: &mut dyn FnMut([&[f32]; 2], Option<[&[f32]; 2]>),
) -> std::result::Result<(), String> {
    let (a0, a1) = range;
    let step = i64::from(a.settings.sample_rate.max(1));
    let mut pos = a0;
    while pos < a1 {
        let end = (pos + step).min(a1);
        let mut x = (Vec::new(), Vec::new());
        render(project, a, pos, end, sources, &mut |l, r| x = (l.to_vec(), r.to_vec()))?;
        match b {
            Some(b) => {
                let mut y = (Vec::new(), Vec::new());
                render(project, b, pos, end, sources, &mut |l, r| y = (l.to_vec(), r.to_vec()))?;
                f([&x.0, &x.1], Some([&y.0, &y.1]));
            }
            None => f([&x.0, &x.1], None),
        }
        pos = end;
    }
    Ok(())
}

/// Whether `effects` hold an enabled limiter.
fn limits(effects: &[filmcraft_project::EffectInstance]) -> bool {
    effects.iter().any(|e| e.enabled && LIMITERS.contains(&e.effect.as_str()))
}

/// A copy of `q` with every limiter on audio track `k` switched off.
fn without_track_limiters(q: &Sequence, k: usize) -> Sequence {
    let mut v = q.clone();
    if let Some(t) = v.audio_tracks.get_mut(k) {
        for e in t.effects.iter_mut().filter(|e| LIMITERS.contains(&e.effect.as_str())) {
            e.enabled = false;
        }
    }
    v
}

fn gr_json(g: &filmcraft_audio_dsp::mixcheck::GainReduction) -> Value {
    json!({
        "maxGainReductionDb": num(g.max_db),
        "p99GainReductionDb": num(g.p99_db),
        "secondsOver1Db": num(g.seconds_over_1db),
        "secondsOver2Db": num(g.seconds_over_2db),
    })
}

/// A copy of `q` with every limiter on the Mix switched off.
fn without_limiters(q: &Sequence) -> Sequence {
    let mut v = q.clone();
    for e in v.master_effects.iter_mut().filter(|e| LIMITERS.contains(&e.effect.as_str())) {
        e.enabled = false;
    }
    v
}

/// A copy of `q` (limiters off) where only the audio tracks of `solo` play: soloed, the others
/// not. An empty `solo` mutes every track, so only the transitions' sounds play.
fn only(q: &Sequence, solo: &[usize]) -> Sequence {
    let mut v = without_limiters(q);
    for (k, t) in v.audio_tracks.iter_mut().enumerate() {
        t.solo = solo.contains(&k);
        if solo.is_empty() {
            t.muted = true;
        }
    }
    v
}

/// A copy of `q` (limiters off) with the audio tracks of `muted` muted.
fn muting(q: &Sequence, muted: &[usize]) -> Sequence {
    let mut v = without_limiters(q);
    for (k, t) in v.audio_tracks.iter_mut().enumerate() {
        t.solo = false;
        if muted.contains(&k) {
            t.muted = true;
        }
    }
    v
}

/// Track indices named by `labels` (`"A2"`, or 1-based numbers) in `p[key]`, or `None` when absent.
fn tracks_p(p: &Value, key: &str, n: usize) -> Result<Option<Vec<usize>>> {
    let Some(v) = p.get(key).filter(|v| !v.is_null()) else { return Ok(None) };
    let list: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        other => vec![other],
    };
    let mut out = Vec::new();
    for x in list.iter().take(64) {
        let k = match x {
            Value::String(s) => s.trim().trim_start_matches(['A', 'a']).parse::<usize>().ok(),
            Value::Number(n) => n.as_u64().and_then(|n| usize::try_from(n).ok()),
            _ => None,
        }
        .and_then(|k| k.checked_sub(1))
        .filter(|k| *k < n)
        .ok_or_else(|| bad(CMD, format!("`{key}`: no audio track {x}")))?;
        if !out.contains(&k) {
            out.push(k);
        }
    }
    Ok(Some(out))
}

/// Analyse samples `[a0, a1)` of sequence `seq` (see the module docs).
pub fn analyze(
    project: &Project,
    seq: ItemId,
    range: (i64, i64),
    speech_tracks: Option<Vec<usize>>,
    music_tracks: Option<Vec<usize>>,
    sources: &dyn SourceProvider,
) -> std::result::Result<Value, String> {
    let q = project.sequence(seq).ok_or("no such sequence")?;
    let sr = q.settings.sample_rate.clamp(8_000, 384_000);
    let (a0, a1) = range;
    if a1 <= a0 {
        return Err("nothing to analyse: the range is empty".into());
    }
    let sub = (sr / 10).max(1) as usize;
    let span = (Tick::from_units(a0, i64::from(sr)), Tick::from_units(a1, i64::from(sr)));
    let audible = |t: &Track| !t.muted && t.items.iter().any(|i| i.enabled && i.start < span.1 && i.end() > span.0);
    let roles: Vec<Role> = q.audio_tracks.iter().map(track_role).collect();
    let pick = |want: Role| -> Vec<usize> { (0..q.audio_tracks.len()).filter(|k| roles.get(*k) == Some(&want)).collect() };
    let live = |v: Vec<usize>| -> Vec<usize> { v.into_iter().filter(|k| q.audio_tracks.get(*k).is_some_and(audible)).collect() };
    let speech = live(speech_tracks.unwrap_or_else(|| pick(Role::Speech)));
    let music = live(music_tracks.unwrap_or_else(|| pick(Role::Music)));

    // the mix as it plays, and before the Mix's limiters (time-aligned: the mixer compensates
    // the limiter's look-ahead)
    let pre_q = without_limiters(q);
    let has_limiter = limits(&q.master_effects);
    let mut post = Metered::new(sr);
    let mut pre = Metered::new(sr);
    let mut gr = GainReductionMeter::new(f64::from(sr));
    render_pair(project, q, has_limiter.then_some(&pre_q), range, sources, &mut |[l, r], before| {
        post.feed(l, r, sub);
        if let Some([pl, pr]) = before {
            pre.feed(pl, pr, sub);
            gr.process(&[pl, pr], &[l, r]);
        }
    })?;
    // each audible track alone (and, when it has a limiter, without it: the limiter's work)
    let mut stems = Vec::new();
    let mut alone: Vec<(usize, Vec<f64>, f64)> = Vec::new();
    for (k, t) in q.audio_tracks.iter().enumerate().filter(|(_, t)| audible(t)) {
        let mut m = Metered::new(sr);
        let solo = only(q, &[k]);
        let track_limits = limits(&t.effects);
        let unlimited = track_limits.then(|| without_track_limiters(&solo, k));
        let mut tgr = GainReductionMeter::new(f64::from(sr));
        render_pair(project, &solo, unlimited.as_ref(), range, sources, &mut |[l, r], before| {
            m.feed(l, r, sub);
            if let Some([pl, pr]) = before {
                tgr.process(&[pl, pr], &[l, r]);
            }
        })?;
        let mut v = m.json();
        v["track"] = json!(format!("A{}", k + 1));
        v["name"] = json!(t.name);
        v["role"] = json!(roles.get(k).copied().unwrap_or(Role::Other).name());
        if track_limits {
            v["limiter"] = gr_json(&tgr.summary());
        }
        stems.push(v);
        let integrated = m.loud.integrated();
        alone.push((k, m.momentary, integrated));
    }
    // the transitions' sounds alone
    let has_sounds = q
        .video_tracks
        .iter()
        .filter(|t| t.enabled)
        .flat_map(|t| t.transitions.iter())
        .any(|tr| tr.sound.is_some() && tr.start < span.1 && tr.start + tr.duration > span.0);
    if has_sounds {
        let mut m = Metered::new(sr);
        render(project, &only(q, &[]), a0, a1, sources, &mut |l, r| m.feed(l, r, sub))?;
        let mut v = m.json();
        v["track"] = json!("transitions");
        v["name"] = json!("Transition sounds");
        v["role"] = json!(Role::Effects.name());
        stems.push(v);
    }
    // speech against the music and against everything else
    let series = |tracks: &[usize]| -> std::result::Result<(Vec<f64>, f64), String> {
        // one track: its stem pass measured it already
        if let [k] = tracks
            && let Some((_, s, i)) = alone.iter().find(|(j, _, _)| j == k)
        {
            return Ok((s.clone(), *i));
        }
        let mut m = Metered::new(sr);
        render(project, &only(q, tracks), a0, a1, sources, &mut |l, r| m.feed(l, r, sub))?;
        Ok((m.momentary, m.loud.integrated()))
    };
    let speech_json = if speech.is_empty() {
        Value::Null
    } else {
        let (s, s_int) = series(&speech)?;
        let gate = mixcheck::speech_gate(s_int);
        let to_music = if music.is_empty() {
            None
        } else {
            let (m, _) = series(&music)?;
            mixcheck::speech_ratio(&s, &m, gate, HOP_S)
        };
        let mut rest = Metered::new(sr);
        render(project, &muting(q, &speech), a0, a1, sources, &mut |l, r| rest.feed(l, r, sub))?;
        let to_rest = mixcheck::speech_ratio(&s, &rest.momentary, gate, HOP_S);
        // the mix (as it plays) while the speech is active
        let mut during: Vec<f64> = s.iter().zip(&post.momentary).filter(|(v, m)| v.is_finite() && **v >= gate && m.is_finite()).map(|(_, m)| *m).collect();
        during.sort_by(f64::total_cmp);
        let mix_during = during.get(during.len() / 2).copied().unwrap_or(f64::NEG_INFINITY);
        json!({
            "tracks": speech.iter().map(|k| format!("A{}", k + 1)).collect::<Vec<_>>(),
            "music": music.iter().map(|k| format!("A{}", k + 1)).collect::<Vec<_>>(),
            "loudnessLufs": num(s_int),
            "mixDuringSpeechLufs": num(mix_during),
            "toMusic": ratio_json(to_music),
            "toRest": ratio_json(to_rest),
        })
    };
    // clipping in each speech clip's own signal
    let mut clips = Vec::new();
    let mut clipped = 0usize;
    for k in &speech {
        let Some(t) = q.audio_tracks.get(*k) else { continue };
        for it in t.items.iter().filter(|i| i.enabled && i.start < span.1 && i.end() > span.0).take(10_000) {
            let c0 = it.start.to_units_floor(i64::from(sr)).max(a0);
            let c1 = it.end().to_units_floor(i64::from(sr)).min(a1);
            let mut m = ClipMeter::new();
            let mut pos = c0;
            while pos < c1 {
                let n = (c1 - pos).min(i64::from(sr)) as usize;
                let Some([l, r]) = filmcraft_render::audio::clip_signal(it, pos, n, sr, sources) else { break };
                m.process(&[&l, &r]);
                pos += n as i64;
            }
            let c = m.clipping();
            let bad = c.overs > 0 || c.flat_runs > 0;
            clipped += usize::from(bad);
            clips.push(json!({
                "clip": it.id.0,
                "name": it.name,
                "track": format!("A{}", k + 1),
                "seconds": num(it.start.seconds()),
                "peakDbfs": num(20.0 * f64::from(c.peak).log10()),
                "overs": c.overs,
                "flatRuns": c.flat_runs,
                "clipped": bad,
            }));
        }
    }
    let lim = gr.summary();
    let mut pre_json = if has_limiter { pre.json() } else { post.json() };
    pre_json["note"] = json!(if has_limiter { "the mix with the Mix's limiters off" } else { "the Mix has no limiter: as it plays" });
    Ok(json!({
        "sequence": seq.0,
        "from": num(a0 as f64 / f64::from(sr)),
        "to": num(a1 as f64 / f64::from(sr)),
        "mix": post.json(),
        "preLimiter": pre_json,
        "limiter": {
            "present": has_limiter,
            "maxGainReductionDb": num(lim.max_db),
            "p99GainReductionDb": num(lim.p99_db),
            "secondsOver1Db": num(lim.seconds_over_1db),
            "secondsOver2Db": num(lim.seconds_over_2db),
        },
        "stems": stems,
        "speech": speech_json,
        "narrationClips": clips,
        "clippedNarrationClips": clipped,
    }))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let seq = match u64_p(p, "sequence") {
        Some(id) => {
            let id = ItemId(id);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let q = s.project.sequence(seq).ok_or_else(|| bad(CMD, "no such sequence"))?;
    let sr = f64::from(q.settings.sample_rate.clamp(8_000, 384_000));
    let len = q.duration().seconds();
    let secs = |k: &str| -> Result<Option<f64>> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(None),
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() && x >= 0.0 => Ok(Some(x)),
                _ => Err(bad(CMD, format!("`{k}`: seconds from the sequence's start (0 or more)"))),
            },
        }
    };
    let from = secs("from")?.unwrap_or(0.0).min(len);
    let to = secs("to")?.unwrap_or(len).min(len);
    if to <= from {
        return Err(bad(CMD, "`to` must be after `from`, inside the sequence"));
    }
    if to - from > MAX_SECONDS {
        return Err(bad(CMD, format!("at most {} hours at a time: give `from` and `to`", MAX_SECONDS / 3600.0)));
    }
    let n = q.audio_tracks.len();
    let speech = tracks_p(p, "speech", n)?;
    let music = tracks_p(p, "music", n)?;
    let range = ((from * sr).round() as i64, (to * sr).round() as i64);
    let provider = s.media.full_res_provider(s.project.clone(), s.services.clone());
    analyze(&s.project, seq, range, speech, music, &provider).map_err(|e| bad(CMD, e))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Analyze Mix",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id?,"from":seconds?,"to":seconds?,"speech":["A2"]?,"music":["A3"]?}"#,
        enabled: has_seq,
        run,
        journal: false,
    }]
}

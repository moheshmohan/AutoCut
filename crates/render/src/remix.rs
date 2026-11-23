//! Playback of remixed clips (Clip ▸ Remix): maps the clip's timeline samples through its remix
//! plan to media samples, crossfading at the joints.
//!
//! A remixed clip carries a hidden effect instance [`EFFECT`] (no effect definition, so the
//! Effects panel, Effect Controls, the audio-effect chain and Paste Attributes ignore it). Its
//! parameters hold the Remix settings and the plan:
//!
//! | param | value |
//! |---|---|
//! | `target` | target duration (ticks, as a float) |
//! | `segments`, `variations` | the Remix sliders (0 … 100) |
//! | `original` | the clip duration before the remix (ticks), restored by Revert Remix |
//! | `plan` | text `xfade;src,len;src,len;…` in ticks: crossfade length, then each piece's media start and length |
//! | `fadeOut`, `fadeIn` | (fitted remixes) the fade-out at the end of the plan and the fade-in at its start, ticks of the plan |
//! | `stretch` | (fitted remixes) plan ticks per clip tick: the plan is time-stretched (pitch kept, WSOLA) to the clip |
//! | `ending`, `events`, `maxStretchPct` | (fitted remixes) what it was fitted to, so a re-plan keeps it: the ending (`own`, `button`, `fade`), the film events as text `kind@ticks;…`, the stretch allowed |
//!
//! The fitted parameters are written only when they differ from a plain remix's (no fade, no
//! stretch, no ending, no events), so a plain remix stores exactly what it did before and plans
//! saved before them play unchanged.
//!
//! The plan is computed by the engine (`clip.remix`) with [`filmcraft_audio_dsp::remix`] and
//! [`filmcraft_audio_dsp::fit`]; playback only reads it, so rendering a remixed clip costs no
//! analysis.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_audio_dsp::remix::{self as dsp, Piece};
use filmcraft_audio_dsp::stretch::{self, TimeStretch};
use filmcraft_frame::AudioBuffer;
use filmcraft_project::{EffectInstance, Param, ParamValue, TrackItem};
use filmcraft_time::Tick;

/// Effect id of the hidden remix state.
pub const EFFECT: &str = "remix";

/// One piece of a remix in ticks: `len` ticks of media from media time `src`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickPiece {
    pub src: Tick,
    pub len: Tick,
}

/// What a fitted remix was fitted to (kept so a re-plan keeps it).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fitted {
    /// `""` (as the plain remix), `own`, `button` or `fade`.
    pub ending: String,
    /// Film events: kind (`burst`, `peak`, `landing`, `stop`) and time from the clip's start.
    pub events: Vec<(String, Tick)>,
    pub max_stretch_pct: f64,
}

impl Fitted {
    pub fn is_plain(&self) -> bool {
        self.ending.is_empty() && self.events.is_empty() && self.max_stretch_pct <= 0.0
    }

    fn events_text(&self) -> String {
        self.events.iter().map(|(k, t)| format!("{k}@{}", t.0)).collect::<Vec<_>>().join(";")
    }

    /// Parses [`Fitted::events_text`]; damaged entries are dropped, at most 256 kept.
    fn parse_events(s: &str) -> Vec<(String, Tick)> {
        s.split(';')
            .filter_map(|e| {
                let (k, t) = e.split_once('@')?;
                let k = k.trim();
                (!k.is_empty() && k.len() <= 16 && k.chars().all(|c| c.is_ascii_lowercase())).then_some(())?;
                Some((k.to_string(), Tick(t.trim().parse().ok()?)))
            })
            .take(256)
            .collect()
    }
}

/// The remix state of a clip.
#[derive(Clone, Debug, PartialEq)]
pub struct Remix {
    pub target: Tick,
    pub segments: f64,
    pub variations: f64,
    /// Clip duration before the remix.
    pub original: Tick,
    /// Crossfade at each joint.
    pub xfade: Tick,
    pub pieces: Vec<TickPiece>,
    /// Fade-out at the end of the plan and fade-in at its start (ticks of the plan).
    pub fade_out: Tick,
    pub fade_in: Tick,
    /// Plan ticks per clip tick (1: not stretched).
    pub stretch: f64,
    pub fitted: Fitted,
}

/// Fastest and slowest a remix plays its plan (the stretch [`filmcraft_audio_dsp::fit`] allows).
const STRETCH_RANGE: (f64, f64) = (0.5, 2.0);

impl Remix {
    /// Output length of the plan (before the stretch).
    pub fn len(&self) -> Tick {
        Tick(self.pieces.iter().map(|p| p.len.0).sum())
    }
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The remix of `item`, if it has one.
    pub fn of(item: &TrackItem) -> Option<Remix> {
        let e = item.effects.iter().find(|e| e.effect == EFFECT)?;
        let f = |k: &str| e.param(k).and_then(|p| p.value.as_f64());
        let plan = match e.param("plan").map(|p| &p.value) {
            // stored in ticks: the limit is the longest project, not the sample-domain one
            Some(ParamValue::Text(s)) => dsp::Plan::from_text_within(s, Tick::MAX.0),
            _ => None,
        };
        let (xfade, pieces) = match plan {
            Some(p) => (Tick(p.xfade), p.pieces.iter().map(|p| TickPiece { src: Tick(p.src), len: Tick(p.len) }).collect()),
            None => (Tick::ZERO, Vec::new()),
        };
        let text = |k: &str| match e.param(k).map(|p| &p.value) {
            Some(ParamValue::Text(s)) => s.clone(),
            _ => String::new(),
        };
        let total = pieces.iter().fold(0i64, |a: i64, p: &TickPiece| a.saturating_add(p.len.0));
        let stretch = f("stretch").filter(|v| v.is_finite() && (STRETCH_RANGE.0..=STRETCH_RANGE.1).contains(v)).unwrap_or(1.0);
        let ending = text("ending");
        let ending = if matches!(ending.as_str(), "own" | "button" | "fade") { ending } else { String::new() };
        Some(Remix {
            target: Tick(f("target").unwrap_or(0.0) as i64),
            segments: f("segments").unwrap_or(50.0),
            variations: f("variations").unwrap_or(50.0),
            original: Tick(f("original").unwrap_or(item.duration.0 as f64) as i64),
            xfade,
            pieces,
            fade_out: Tick(f("fadeOut").filter(|v| v.is_finite()).map_or(0, |v| v as i64).clamp(0, total.max(0))),
            fade_in: Tick(f("fadeIn").filter(|v| v.is_finite()).map_or(0, |v| v as i64).clamp(0, total.max(0))),
            stretch,
            fitted: Fitted {
                ending,
                events: Fitted::parse_events(&text("events")),
                max_stretch_pct: f("maxStretchPct").filter(|v| v.is_finite()).unwrap_or(0.0).clamp(0.0, 100.0),
            },
        })
    }

    /// The hidden effect instance that stores this remix.
    pub fn to_effect(&self) -> EffectInstance {
        let plan =
            dsp::Plan { pieces: self.pieces.iter().map(|p| Piece { src: p.src.0, len: p.len.0 }).collect(), xfade: self.xfade.0, fade_out: 0, fade_in: 0 };
        let mut params = BTreeMap::new();
        params.insert("target".to_string(), Param::new(ParamValue::Float(self.target.0 as f64)));
        params.insert("segments".to_string(), Param::new(ParamValue::Float(self.segments)));
        params.insert("variations".to_string(), Param::new(ParamValue::Float(self.variations)));
        params.insert("original".to_string(), Param::new(ParamValue::Float(self.original.0 as f64)));
        params.insert("plan".to_string(), Param::new(ParamValue::Text(if self.pieces.is_empty() { String::new() } else { plan.to_text() })));
        // a fitted remix's extras (a plain remix stores only what it did before)
        if self.fade_out > Tick::ZERO {
            params.insert("fadeOut".to_string(), Param::new(ParamValue::Float(self.fade_out.0 as f64)));
        }
        if self.fade_in > Tick::ZERO {
            params.insert("fadeIn".to_string(), Param::new(ParamValue::Float(self.fade_in.0 as f64)));
        }
        if (self.stretch - 1.0).abs() > 1e-9 {
            params.insert("stretch".to_string(), Param::new(ParamValue::Float(self.stretch)));
        }
        if !self.fitted.ending.is_empty() {
            params.insert("ending".to_string(), Param::new(ParamValue::Text(self.fitted.ending.clone())));
        }
        if !self.fitted.events.is_empty() {
            params.insert("events".to_string(), Param::new(ParamValue::Text(self.fitted.events_text())));
        }
        if self.fitted.max_stretch_pct > 0.0 {
            params.insert("maxStretchPct".to_string(), Param::new(ParamValue::Float(self.fitted.max_stretch_pct)));
        }
        EffectInstance { effect: EFFECT.to_string(), enabled: true, params, masks: Vec::new(), post_fader: false, essential: false, layer: None }
    }

    /// Store this remix on `item` (replacing an earlier one).
    pub fn store(&self, item: &mut TrackItem) {
        let e = self.to_effect();
        match item.effects.iter_mut().find(|x| x.effect == EFFECT) {
            Some(x) => *x = e,
            None => item.effects.push(e),
        }
    }

    /// The plan in samples at `sr` (piece boundaries are floored on the timeline, so consecutive
    /// pieces tile the clip exactly at every rate).
    pub fn plan_at(&self, sr: u32) -> dsp::Plan {
        let r = sr as i64;
        let mut pieces = Vec::with_capacity(self.pieces.len());
        let mut d = Tick::ZERO;
        for p in &self.pieces {
            let a = d.to_units_floor(r);
            d += p.len;
            pieces.push(Piece { src: p.src.to_units_floor(r), len: d.to_units_floor(r) - a });
        }
        dsp::Plan {
            pieces,
            xfade: self.xfade.to_units_floor(r),
            fade_out: self.fade_out.to_units_floor(r).clamp(0, d.to_units_floor(r)),
            fade_in: self.fade_in.to_units_floor(r).clamp(0, d.to_units_floor(r)),
        }
    }

    /// Whether playback time-stretches the plan.
    pub fn stretched(&self) -> bool {
        (self.stretch - 1.0).abs() > 1e-9 && stretch::speed_supported(self.stretch)
    }
}

/// Remove the remix state from `item`; returns it.
pub fn take(item: &mut TrackItem) -> Option<Remix> {
    let r = Remix::of(item)?;
    item.effects.retain(|e| e.effect != EFFECT);
    Some(r)
}

/// Whether `item` is remixed (has a plan that playback follows).
pub fn is_remixed(item: &TrackItem) -> bool {
    item.effects.iter().any(|e| e.effect == EFFECT && e.enabled)
}

/// Audio of a remixed clip for clip-relative timeline samples `[rel0, rel0 + n)` at rate `sr`,
/// reading the media through `read(media_sample, len)`. `None` when the clip is not remixed (the
/// caller then reads the media linearly). Speed and reverse do not apply to remixed clips.
pub fn read(item: &TrackItem, read: &dyn Fn(i64, usize) -> Option<AudioBuffer>, rel0: i64, n: usize, sr: u32) -> Option<AudioBuffer> {
    if !is_remixed(item) {
        return None;
    }
    let rm = Remix::of(item)?;
    if rm.pieces.is_empty() {
        return None;
    }
    let plan = rm.plan_at(sr);
    let channels = read(plan.pieces[0].src, 1).map(|b| b.channel_count()).unwrap_or(2).max(1);
    let mut rd = |s: i64, len: usize| -> Vec<Vec<f32>> {
        let mut b = read(s, len).map(|b| b.channels).unwrap_or_default();
        b.resize(channels, Vec::new());
        for c in b.iter_mut() {
            c.resize(len, 0.0);
        }
        b
    };
    if !rm.stretched() {
        let out = dsp::render(&plan, rel0, n, channels, &mut rd);
        return Some(AudioBuffer { sample_rate: sr, channels: out });
    }
    // stretched: clip sample o plays plan position o · stretch, at its own pitch (WSOLA over
    // the rendered plan); the stretcher's memo of splice points is kept per plan
    let ts = stretcher(item, &rm, sr, channels)?;
    let mut ts = ts.lock().unwrap_or_else(PoisonError::into_inner);
    let mut input = |u: i64, len: usize| dsp::render(&plan, u, len, channels, &mut rd);
    let mut out = vec![vec![0f32; n]; channels];
    ts.render(&mut input, rel0, &mut out).ok()?;
    Some(AudioBuffer { sample_rate: sr, channels: out })
}

/// Stretchers of stretched remixes in use.
const STRETCHERS: usize = 16;

/// The stretcher of a remixed clip's plan at `sr` (one per media, plan, stretch, rate and width).
fn stretcher(item: &TrackItem, rm: &Remix, sr: u32, channels: usize) -> Option<Arc<Mutex<TimeStretch>>> {
    type Cache = Mutex<Vec<(u64, Arc<Mutex<TimeStretch>>)>>;
    static C: OnceLock<Cache> = OnceLock::new();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    item.item.0.hash(&mut h);
    rm.pieces.iter().for_each(|p| (p.src.0, p.len.0).hash(&mut h));
    (rm.xfade.0, rm.fade_out.0, rm.fade_in.0, rm.stretch.to_bits(), sr, channels).hash(&mut h);
    let key = h.finish();
    let cache = C.get_or_init(Default::default);
    let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, ts)) = c.iter().find(|(k, _)| *k == key) {
        return Some(ts.clone());
    }
    let ts = Arc::new(Mutex::new(TimeStretch::new(sr, channels.clamp(1, stretch::MAX_CHANNELS), rm.stretch).ok()?));
    if c.len() >= STRETCHERS {
        c.remove(0);
    }
    c.push((key, ts.clone()));
    Some(ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item() -> TrackItem {
        let d = Tick::from_seconds_f64(1.0).0;
        serde_json::from_value(serde_json::json!({
            "id": 1, "item": 2, "name": "m", "label": "Violet", "start": 0, "duration": d, "source_in": 0,
            "speed": 1.0, "enabled": true, "effects": []
        }))
        .expect("track item")
    }

    fn source(m: i64, len: usize) -> Option<AudioBuffer> {
        // a ramp: sample value = media sample index / 1e5 (silence outside 0 … 48 000)
        let ch: Vec<f32> = (0..len as i64).map(|k| if (0..48_000).contains(&(m + k)) { (m + k) as f32 / 1e5 } else { 0.0 }).collect();
        Some(AudioBuffer { sample_rate: 48_000, channels: vec![ch.clone(), ch] })
    }

    fn remix() -> Remix {
        let s = |x: f64| Tick::from_seconds_f64(x);
        Remix {
            target: s(0.6),
            segments: 50.0,
            variations: 50.0,
            original: s(1.0),
            xfade: s(0.002),
            pieces: vec![TickPiece { src: s(0.0), len: s(0.25) }, TickPiece { src: s(0.65), len: s(0.35) }],
            fade_out: Tick::ZERO,
            fade_in: Tick::ZERO,
            stretch: 1.0,
            fitted: Fitted::default(),
        }
    }

    #[test]
    fn not_remixed_reads_nothing() {
        assert!(read(&item(), &source, 0, 10, 48_000).is_none());
    }

    #[test]
    fn state_roundtrips_through_the_hidden_effect() {
        let mut it = item();
        let r = remix();
        r.store(&mut it);
        r.store(&mut it);
        assert_eq!(it.effects.len(), 1);
        assert_eq!(Remix::of(&it), Some(r.clone()));
        assert!(it.effects[0].def().is_none(), "no definition: hidden from panels and the fx chain");
        assert_eq!(take(&mut it), Some(r));
        assert!(it.effects.is_empty());
    }

    #[test]
    fn a_stored_plan_longer_than_the_sample_limit_round_trips() {
        // 20 s pieces are 5e12 ticks each: past the sample-domain limit, far inside the project's
        let s = |x: f64| Tick::from_seconds_f64(x);
        let r = Remix { pieces: vec![TickPiece { src: Tick::ZERO, len: s(20.0) }, TickPiece { src: s(30.0), len: s(20.0) }], xfade: s(0.1), ..remix() };
        let mut it = item();
        r.store(&mut it);
        assert_eq!(Remix::of(&it), Some(r));
        // a stored plan past the longest project is still refused
        let hostile =
            Remix { pieces: vec![TickPiece { src: Tick::ZERO, len: Tick::MAX }, TickPiece { src: Tick::ZERO, len: Tick(1) }], xfade: Tick::ZERO, ..remix() };
        hostile.store(&mut it);
        assert!(Remix::of(&it).is_some_and(|r| r.pieces.is_empty()));
    }

    #[test]
    fn plays_the_pieces_and_ignores_request_cuts() {
        let mut it = item();
        remix().store(&mut it);
        let total = (0.6 * 48_000.0) as usize;
        let whole = read(&it, &source, 0, total, 48_000).unwrap();
        // away from the joint (12 000) the output is the media ramp of each piece
        assert_eq!(whole.channels[0][100], 100.0 / 1e5);
        assert_eq!(whole.channels[0][11_000], 11_000.0 / 1e5);
        assert_eq!(whole.channels[0][13_000], (31_200 + 1_000) as f32 / 1e5);
        assert_eq!(whole.channels[1][20_000], (31_200 + 8_000) as f32 / 1e5);
        let mut cut: Vec<f32> = Vec::new();
        let mut pos = 0usize;
        let mut step = 3usize;
        while pos < total {
            let n = step.min(total - pos);
            cut.extend(read(&it, &source, pos as i64, n, 48_000).unwrap().channels[0].iter());
            pos += n;
            step = step * 5 % 1999 + 1;
        }
        assert_eq!(cut, whole.channels[0]);
        // other rates tile the same plan
        let p = remix().plan_at(44_100);
        assert_eq!(p.len(), (0.6 * 44_100.0) as i64);
    }

    #[test]
    fn a_plain_remix_stores_what_it_did_before() {
        let mut it = item();
        remix().store(&mut it);
        let keys: Vec<&str> = it.effects[0].params.keys().map(String::as_str).collect();
        assert_eq!(keys, ["original", "plan", "segments", "target", "variations"]);
    }

    #[test]
    fn a_fitted_remix_round_trips_and_fades_out() {
        let s = |x: f64| Tick::from_seconds_f64(x);
        let r = Remix {
            fade_out: s(0.1),
            fade_in: s(0.001),
            fitted: Fitted { ending: "button".into(), events: vec![("peak".into(), s(0.2)), ("landing".into(), s(0.5))], max_stretch_pct: 5.0 },
            ..remix()
        };
        let mut it = item();
        r.store(&mut it);
        assert_eq!(Remix::of(&it), Some(r.clone()));
        let total = (0.6 * 48_000.0) as usize;
        let out = read(&it, &source, 0, total, 48_000).unwrap();
        // the last 0.1 s fades out to nothing; before it the pieces play as they were
        assert_eq!(out.channels[0][13_000], (31_200 + 1_000) as f32 / 1e5);
        let end = out.channels[0][total - 1].abs();
        assert!(end < 1e-3, "{end}");
        let mid = total - 2_400;
        let want = (31_200 + (mid - 12_000)) as f32 / 1e5 * (std::f64::consts::FRAC_PI_4.cos() as f32);
        assert!((out.channels[0][mid] - want).abs() < 2e-3, "{} vs {want}", out.channels[0][mid]);
        // damaged stored values are ignored or bounded
        it.effects[0].params.insert("stretch".into(), Param::new(ParamValue::Float(f64::NAN)));
        it.effects[0].params.insert("fadeOut".into(), Param::new(ParamValue::Float(1e300)));
        it.effects[0].params.insert("ending".into(), Param::new(ParamValue::Text("explode".into())));
        it.effects[0].params.insert("events".into(), Param::new(ParamValue::Text("peak@x;;@5;burst@-7;".repeat(200))));
        let back = Remix::of(&it).unwrap();
        assert_eq!(back.stretch, 1.0);
        assert_eq!(back.fade_out, back.len());
        assert!(back.fitted.ending.is_empty());
        assert!(back.fitted.events.len() <= 256 && back.fitted.events.iter().all(|(k, _)| k == "burst"));
        let _ = read(&it, &source, 0, 1000, 48_000);
    }

    #[test]
    fn a_stretched_remix_plays_longer_at_its_pitch() {
        // a 440 Hz tone; the plan plays its 1 s; stretched by 0.8 the clip plays 1.25 s
        let tone = |m: i64, len: usize| -> Option<AudioBuffer> {
            let ch: Vec<f32> = (0..len as i64)
                .map(|k| if (0..48_000).contains(&(m + k)) { ((m + k) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5 } else { 0.0 })
                .collect();
            Some(AudioBuffer { sample_rate: 48_000, channels: vec![ch.clone(), ch] })
        };
        let s = |x: f64| Tick::from_seconds_f64(x);
        let r = Remix { pieces: vec![TickPiece { src: s(0.0), len: s(1.0) }], xfade: s(0.02), stretch: 0.8, target: s(1.25), ..remix() };
        let mut it = item();
        it.duration = s(1.25);
        r.store(&mut it);
        assert!(Remix::of(&it).unwrap().stretched());
        let n = 60_000;
        let whole = read(&it, &tone, 0, n, 48_000).unwrap();
        // the same in pieces
        let mut cut: Vec<f32> = Vec::new();
        let mut pos = 0usize;
        while pos < n {
            let k = 7_777.min(n - pos);
            cut.extend(read(&it, &tone, pos as i64, k, 48_000).unwrap().channels[0].iter());
            pos += k;
        }
        assert_eq!(cut, whole.channels[0]);
        // still 440 Hz: zero crossings over the middle half second
        let mid = &whole.channels[0][15_000..39_000];
        let crossings = mid.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((crossings as f64 - 220.0).abs() <= 3.0, "{crossings}");
        // and the tone lasts to 1.25 s of the clip
        let late: f32 = whole.channels[0][57_000..59_000].iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(late > 0.3, "{late}");
    }
}

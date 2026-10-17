//! The SoundFont instrument bank.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_audio_dsp::loudness::LoudnessMeter;
use filmcraft_audio_dsp::music::instruments::{Bus, Context, Drum, Kit, mix_hints};
use filmcraft_audio_dsp::music::{self, Genre, Instrument, InstrumentBank, OscillatorBank, Patch, Request, Track};
use filmcraft_audio_dsp::synth::SynthError;
use rustysynth::{Synthesizer, SynthesizerSettings};

use crate::{DRUM_BANK, SoundFont};

/// The General MIDI percussion channel.
const DRUM_CHANNEL: i32 = 9;
/// rustysynth applies note events at its block boundaries: 32 samples is under a millisecond.
const BLOCK: usize = 32;
/// Voices per instrument (stacked layers and long releases).
const POLYPHONY: usize = 128;
/// Samples rendered at a time into the scratch buffers.
const CHUNK: usize = 1024;
/// rustysynth's sample-rate range.
const RATES: std::ops::RangeInclusive<u32> = 16_000..=192_000;
/// How far a SoundFont instrument's level is moved to match its oscillator voice, dB.
const MAX_TRIM_DB: f64 = 30.0;
/// Below this an instrument counts as silent (its preset does not play these notes), LUFS.
const SILENT_LUFS: f64 = -70.0;

/// The General MIDI kit (bank 128 program) for a kit character.
fn kit_program(kit: Kit) -> i32 {
    match kit {
        Kit::Cinematic => 16, // Power
        Kit::Electronic => 24,
        Kit::Pop => 0,   // Standard
        Kit::Lofi => 32, // Jazz
        Kit::Soft => 40, // Brush
    }
}

/// How long a drum rings before it is released, seconds (one-shot samples end sooner on their
/// own; a looped one would otherwise ring for ever).
fn drum_ring(key: u8) -> f64 {
    match Drum::from_key(key) {
        Some(Drum::Crash | Drum::Ride | Drum::HatOpen) => 4.0,
        _ => 1.5,
    }
}

/// Where a patch plays in the font: (channel, bank, program).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Route {
    channel: i32,
    bank: i32,
    program: i32,
}

/// The instrument bank of a [`SoundFont`]: every patch the font has a General MIDI preset for
/// plays on it, level-matched to the oscillator voice it replaces (measured once per patch, genre
/// and sample rate); risers and impacts, patches without a preset and sample rates outside
/// 16–192 kHz play on the oscillators.
pub struct SoundFontBank {
    font: SoundFont,
    label: String,
    /// (patch, genre, rate) → linear gain, or `None` when the font is silent for it.
    gains: Mutex<HashMap<(Patch, Genre, u32), Option<f32>>>,
}

impl std::fmt::Debug for SoundFontBank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoundFontBank").field("font", &self.font).finish()
    }
}

impl SoundFontBank {
    pub fn new(font: SoundFont) -> Self {
        let label = format!("{} (SoundFont)", font.name());
        SoundFontBank { font, label, gains: Mutex::new(HashMap::new()) }
    }

    pub fn font(&self) -> &SoundFont {
        &self.font
    }

    /// A player for one track: it counts the font's failures while that track plays.
    pub fn player(&self) -> FontPlayer<'_> {
        FontPlayer { bank: self, faults: Arc::new(AtomicUsize::new(0)), oscillators: Mutex::new(Vec::new()) }
    }

    fn route(&self, patch: Patch) -> Option<Route> {
        match patch {
            Patch::Fx => None,
            Patch::Kit(kit) => {
                let p = kit_program(kit);
                if self.font.has(DRUM_BANK, p) {
                    Some(Route { channel: DRUM_CHANNEL, bank: DRUM_BANK, program: p })
                } else if self.font.has(DRUM_BANK, 0) {
                    Some(Route { channel: DRUM_CHANNEL, bank: DRUM_BANK, program: 0 })
                } else {
                    None
                }
            }
            p => {
                let program = i32::from(p.gm_program()?);
                self.font.has(0, program).then_some(Route { channel: 0, bank: 0, program })
            }
        }
    }

    /// The level that matches the font's `patch` to the oscillator voice (`None`: silent).
    fn gain(&self, patch: Patch, route: Route, ctx: &Context) -> Option<f32> {
        let key = (patch, ctx.genre, ctx.sample_rate);
        if let Some(g) = self.gains.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
            return *g;
        }
        let phrase = phrase(patch);
        let faults = Arc::new(AtomicUsize::new(0));
        let sf = SfInstrument::new(&self.font, route, patch, ctx, 1.0, faults.clone()).ok().map(|mut i| loudness(&mut i, &phrase, ctx.sample_rate));
        let osc = OscillatorBank.instrument(patch, ctx).ok().map(|mut i| loudness(i.as_mut(), &phrase, ctx.sample_rate));
        let g = match (sf, osc) {
            (Some(sf), _) if !sf.is_finite() || sf < SILENT_LUFS || faults.load(Ordering::Relaxed) > 0 => None,
            (Some(sf), Some(osc)) if osc.is_finite() && osc > SILENT_LUFS => Some(10f64.powf(((osc - sf).clamp(-MAX_TRIM_DB, MAX_TRIM_DB)) / 20.0) as f32),
            (Some(sf), _) => Some(10f64.powf(((-24.0 - sf).clamp(-MAX_TRIM_DB, MAX_TRIM_DB)) / 20.0) as f32),
            (None, _) => None,
        };
        self.gains.lock().unwrap_or_else(PoisonError::into_inner).insert(key, g);
        g
    }
}

/// [`SoundFontBank`] for one track: counts the font's failures and which patches fell back to
/// the oscillators.
pub struct FontPlayer<'a> {
    bank: &'a SoundFontBank,
    faults: Arc<AtomicUsize>,
    oscillators: Mutex<Vec<Patch>>,
}

impl FontPlayer<'_> {
    /// How many times the font failed while this track played (each failure silenced one
    /// instrument).
    pub fn faults(&self) -> usize {
        self.faults.load(Ordering::Relaxed)
    }

    /// The patches played on the oscillators instead (no preset, silent, or effects).
    pub fn on_oscillators(&self) -> Vec<Patch> {
        self.oscillators.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn fallback(&self, patch: Patch, ctx: &Context) -> Result<Box<dyn Instrument>, SynthError> {
        let mut o = self.oscillators.lock().unwrap_or_else(PoisonError::into_inner);
        if !o.contains(&patch) {
            o.push(patch);
        }
        drop(o);
        OscillatorBank.instrument(patch, ctx)
    }
}

impl InstrumentBank for FontPlayer<'_> {
    fn name(&self) -> &str {
        &self.bank.label
    }

    fn instrument(&self, patch: Patch, ctx: &Context) -> Result<Box<dyn Instrument>, SynthError> {
        if !RATES.contains(&ctx.sample_rate) {
            return self.fallback(patch, ctx);
        }
        let Some(route) = self.bank.route(patch) else { return self.fallback(patch, ctx) };
        let Some(gain) = self.bank.gain(patch, route, ctx) else { return self.fallback(patch, ctx) };
        match SfInstrument::new(&self.bank.font, route, patch, ctx, gain, self.faults.clone()) {
            Ok(i) => Ok(Box::new(i)),
            Err(_) => self.fallback(patch, ctx),
        }
    }
}

impl InstrumentBank for SoundFontBank {
    fn name(&self) -> &str {
        &self.label
    }

    fn instrument(&self, patch: Patch, ctx: &Context) -> Result<Box<dyn Instrument>, SynthError> {
        if !RATES.contains(&ctx.sample_rate) {
            return OscillatorBank.instrument(patch, ctx);
        }
        let Some(route) = self.route(patch) else { return OscillatorBank.instrument(patch, ctx) };
        let Some(gain) = self.gain(patch, route, ctx) else { return OscillatorBank.instrument(patch, ctx) };
        match SfInstrument::new(&self.font, route, patch, ctx, gain, Arc::new(AtomicUsize::new(0))) {
            Ok(i) => Ok(Box::new(i)),
            Err(_) => OscillatorBank.instrument(patch, ctx),
        }
    }
}

/// Make the track `req` asks for on `bank` (the oscillators for what the font lacks). If the font
/// fails while playing, the track is made again on the oscillators alone and the note says so.
/// Returns the track (its `instruments` says what played it) and that note.
pub fn generate(req: &Request, sample_rate: u32, bank: &SoundFontBank) -> Result<(Track, Option<String>), SynthError> {
    let player = bank.player();
    let mut track = music::generate_with(req, sample_rate, &player)?;
    if player.faults() > 0 {
        let track = music::generate(req, sample_rate)?;
        return Ok((track, Some(format!("the SoundFont {} failed while playing; this track was played on the oscillators", bank.font.name()))));
    }
    let osc = player.on_oscillators();
    // risers and impacts are always synthesized; say so only when an instrument was
    if osc.iter().any(|p| *p != Patch::Fx) {
        let names: Vec<&str> = osc.iter().filter(|p| **p != Patch::Fx).map(|p| p.id()).collect();
        track.instruments = format!("{} + oscillators ({})", bank.label, names.join(", "));
    }
    Ok((track, None))
}

/// A part played on a SoundFont preset through rustysynth.
struct SfInstrument {
    /// `None` once rustysynth failed (the part is silent from there).
    synth: Option<Synthesizer>,
    channel: i32,
    drums: bool,
    gain: f32,
    send: f32,
    sample_rate: f64,
    /// Samples rendered so far.
    t: u64,
    /// Drum releases due: (sample, key).
    offs: Vec<(u64, u8)>,
    l: Vec<f32>,
    r: Vec<f32>,
    faults: Arc<AtomicUsize>,
}

impl SfInstrument {
    fn new(font: &SoundFont, route: Route, patch: Patch, ctx: &Context, gain: f32, faults: Arc<AtomicUsize>) -> Result<Self, String> {
        let rate = i32::try_from(ctx.sample_rate).map_err(|_| "sample rate out of range".to_string())?;
        let mut settings = SynthesizerSettings::new(rate);
        settings.block_size = BLOCK;
        settings.maximum_polyphony = POLYPHONY;
        settings.enable_reverb_and_chorus = false;
        let made = catch_unwind(AssertUnwindSafe(|| Synthesizer::new(font.inner(), &settings)));
        let mut synth = match made {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(e.to_string()),
            Err(_) => return Err("the SoundFont synthesizer failed to start".into()),
        };
        let (pan, send) = mix_hints(patch, ctx.genre);
        let pan_cc = (64.0 + pan.clamp(-1.0, 1.0) * 63.0).round() as i32;
        let set = catch_unwind(AssertUnwindSafe(|| {
            // bank select (the percussion channel adds 128 itself), program, pan
            synth.process_midi_message(route.channel, 0xB0, 0x00, if route.bank == DRUM_BANK { 0 } else { route.bank });
            synth.process_midi_message(route.channel, 0xC0, route.program, 0);
            synth.process_midi_message(route.channel, 0xB0, 0x0A, pan_cc.clamp(0, 127));
        }));
        if set.is_err() {
            return Err("the SoundFont synthesizer failed to start".into());
        }
        Ok(SfInstrument {
            synth: Some(synth),
            channel: route.channel,
            drums: route.channel == DRUM_CHANNEL,
            gain: if gain.is_finite() { gain } else { 0.0 },
            send: send.clamp(0.0, 1.0) as f32,
            sample_rate: f64::from(ctx.sample_rate),
            t: 0,
            offs: Vec::new(),
            l: vec![0.0; CHUNK],
            r: vec![0.0; CHUNK],
            faults,
        })
    }

    /// Call into rustysynth; a panic silences this part (and is counted) instead of escaping.
    fn call(&mut self, f: impl FnOnce(&mut Synthesizer)) {
        let Some(s) = self.synth.as_mut() else { return };
        if catch_unwind(AssertUnwindSafe(|| f(s))).is_err() {
            self.synth = None;
            self.faults.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn release_due(&mut self) {
        let now = self.t;
        let due: Vec<u8> = self.offs.iter().filter(|(at, _)| *at <= now).map(|(_, k)| *k).collect();
        if due.is_empty() {
            return;
        }
        self.offs.retain(|(at, _)| *at > now);
        let ch = self.channel;
        self.call(|s| {
            for k in due {
                s.note_off(ch, i32::from(k));
            }
        });
    }
}

impl Instrument for SfInstrument {
    fn note_on(&mut self, key: u8, velocity: f32) {
        if !velocity.is_finite() || key > 127 {
            return;
        }
        let v = (velocity.clamp(0.0, 1.0) * 127.0).round() as i32;
        if v < 1 {
            return;
        }
        let ch = self.channel;
        if self.drums {
            // a retriggered drum is released one ring after its last hit
            let ring = (drum_ring(key) * self.sample_rate) as u64;
            self.offs.retain(|(_, k)| *k != key);
            self.offs.push((self.t.saturating_add(ring), key));
            self.call(|s| s.note_on(ch, i32::from(key), v));
        } else {
            // the same key held again is released first, as on the oscillators
            self.call(|s| {
                s.note_off(ch, i32::from(key));
                s.note_on(ch, i32::from(key), v);
            });
        }
    }

    fn note_off(&mut self, key: u8) {
        if self.drums || key > 127 {
            return;
        }
        let ch = self.channel;
        self.call(|s| s.note_off(ch, i32::from(key)));
    }

    fn render(&mut self, bus: &mut Bus<'_>) {
        let n = bus.len();
        let mut done = 0usize;
        while done < n {
            self.release_due();
            let mut m = (n - done).min(CHUNK);
            if let Some(next) = self.offs.iter().map(|(at, _)| *at).min()
                && next > self.t
            {
                m = m.min(usize::try_from(next - self.t).unwrap_or(usize::MAX)).max(1);
            }
            // the scratch buffers leave `self` while rustysynth fills them
            let mut l = std::mem::take(&mut self.l);
            let mut r = std::mem::take(&mut self.r);
            let (Some(lm), Some(rm)) = (l.get_mut(..m), r.get_mut(..m)) else {
                self.l = l;
                self.r = r;
                break;
            };
            lm.fill(0.0);
            rm.fill(0.0);
            self.call(|s| s.render(lm, rm));
            let (g, send) = (self.gain, self.send);
            let span = done..done + m;
            let (Some(bl), Some(br), Some(cl), Some(cr)) =
                (bus.dry_l.get_mut(span.clone()), bus.dry_r.get_mut(span.clone()), bus.send_l.get_mut(span.clone()), bus.send_r.get_mut(span))
            else {
                self.l = l;
                self.r = r;
                break;
            };
            let lanes = bl.iter_mut().zip(br.iter_mut()).zip(cl.iter_mut().zip(cr.iter_mut()));
            for (((dl, dr), (sl, sr)), (x, y)) in lanes.zip(l.iter().zip(r.iter())) {
                let (a, b) = (if x.is_finite() { x * g } else { 0.0 }, if y.is_finite() { y * g } else { 0.0 });
                *dl += a;
                *dr += b;
                *sl += a * send;
                *sr += b * send;
            }
            self.l = l;
            self.r = r;
            done += m;
            self.t = self.t.saturating_add(m as u64);
        }
    }

    fn release_seconds(&self) -> f64 {
        1.0
    }
}

/// A short phrase that shows a patch's level: (start s, length s, key, velocity).
fn phrase(patch: Patch) -> Vec<(f64, f64, u8, f32)> {
    match patch {
        Patch::Kit(_) => {
            let mut v: Vec<(f64, f64, u8, f32)> = (0..8).map(|i| (f64::from(i) * 0.25, 0.1, Drum::HatClosed.gm_key(), 0.7)).collect();
            v.extend([(0.0, 0.1, Drum::Kick.gm_key(), 0.8), (1.0, 0.1, Drum::Kick.gm_key(), 0.8)]);
            v.extend([(0.5, 0.1, Drum::Snare.gm_key(), 0.8), (1.5, 0.1, Drum::Snare.gm_key(), 0.8)]);
            v
        }
        Patch::SynthBass | Patch::SubBass | Patch::RoundBass => (0..4).map(|i| (f64::from(i) * 0.5, 0.45, 33u8, 0.7)).collect(),
        Patch::Pluck | Patch::Brass | Patch::Bell | Patch::Flute | Patch::SawLead => {
            [72u8, 74, 76, 79, 76, 74, 72, 71].iter().enumerate().map(|(i, k)| (i as f64 * 0.25, 0.22, *k, 0.7)).collect()
        }
        _ => (0..2).flat_map(|b| [60u8, 64, 67].map(|k| (f64::from(b), 1.0, k, 0.7))).collect(),
    }
}

/// Integrated loudness of `inst` playing `notes` for two seconds (the dry signal), LUFS.
fn loudness(inst: &mut dyn Instrument, notes: &[(f64, f64, u8, f32)], rate: u32) -> f64 {
    let sr = f64::from(rate);
    let n = (2.0 * sr) as usize;
    let mut ev: Vec<(usize, bool, u8, f32)> = Vec::with_capacity(notes.len() * 2);
    for &(s, l, k, v) in notes {
        ev.push(((s * sr) as usize, true, k, v));
        ev.push((((s + l) * sr) as usize, false, k, v));
    }
    ev.sort_by_key(|e| (e.0, e.1));
    let (mut l, mut r, mut sl, mut sr_) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let (mut at, mut i) = (0usize, 0usize);
    while at < n {
        while let Some(e) = ev.get(i).filter(|e| e.0 <= at) {
            if e.1 {
                inst.note_on(e.2, e.3);
            } else {
                inst.note_off(e.2);
            }
            i += 1;
        }
        let next = ev.get(i).map_or(n, |e| e.0).max(at + 1).min(n);
        if let (Some(a), Some(b), Some(c), Some(d)) = (l.get_mut(at..next), r.get_mut(at..next), sl.get_mut(at..next), sr_.get_mut(at..next)) {
            inst.render(&mut Bus { dry_l: a, dry_r: b, send_l: c, send_r: d });
        }
        at = next;
    }
    let mut m = LoudnessMeter::new(sr, 2);
    m.process(&[&l, &r]);
    m.integrated()
}

#[cfg(test)]
impl SoundFontBank {
    /// Skip the level check for `req`'s genre at `rate`: every patch plays at its own level.
    pub(crate) fn assume_levels(&self, req: &Request, rate: u32) {
        let Ok(plan) = music::plan(req, rate) else { return };
        let mut g = self.gains.lock().unwrap_or_else(PoisonError::into_inner);
        let kits = [Kit::Cinematic, Kit::Electronic, Kit::Pop, Kit::Lofi, Kit::Soft].map(Patch::Kit);
        let tonal = [
            Patch::Strings,
            Patch::WarmPad,
            Patch::SuperSaw,
            Patch::Piano,
            Patch::EPiano,
            Patch::Pluck,
            Patch::Brass,
            Patch::Bell,
            Patch::Flute,
            Patch::SawLead,
            Patch::SynthBass,
            Patch::SubBass,
            Patch::RoundBass,
        ];
        for p in kits.into_iter().chain(tonal) {
            g.insert((p, plan.genre, rate), Some(1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_audio_dsp::music::Tempo;

    fn ctx() -> Context {
        Context { sample_rate: 48_000, tempo: Tempo::from_bpm(120.0), genre: Genre::Pop, seed: 1 }
    }

    fn bufs(n: usize) -> [Vec<f32>; 4] {
        [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]]
    }

    #[test]
    fn a_failure_inside_rustysynth_silences_the_part_and_is_counted() {
        let font = SoundFont::from_bytes(&crate::synthetic::runaway()).unwrap();
        let faults = Arc::new(AtomicUsize::new(0));
        let route = Route { channel: 0, bank: 0, program: 0 };
        let mut inst = SfInstrument::new(&font, route, Patch::Piano, &ctx(), 1.0, faults.clone()).unwrap();
        inst.note_on(60, 0.8);
        let [mut a, mut b, mut c, mut d] = bufs(4096);
        inst.render(&mut Bus { dry_l: &mut a, dry_r: &mut b, send_l: &mut c, send_r: &mut d });
        assert_eq!(faults.load(Ordering::Relaxed), 1);
        assert!(inst.synth.is_none());
        assert!(a.iter().all(|x| x.is_finite()));
        // and it stays quiet, with nothing more to report
        inst.note_on(64, 0.8);
        inst.note_off(64);
        inst.render(&mut Bus { dry_l: &mut a, dry_r: &mut b, send_l: &mut c, send_r: &mut d });
        assert_eq!(faults.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_healthy_font_plays_level_matched() {
        let font = SoundFont::from_bytes(&crate::synthetic::tiny()).unwrap();
        let bank = SoundFontBank::new(font);
        let c = ctx();
        for p in [Patch::Piano, Patch::SynthBass, Patch::Pluck, Patch::Kit(Kit::Pop)] {
            let route = bank.route(p).unwrap();
            let g = bank.gain(p, route, &c).unwrap();
            assert!(g.is_finite() && g > 0.0, "{p:?} {g}");
            // matched: the font's phrase at that gain is as loud as the oscillator's
            let mut sf = SfInstrument::new(&bank.font, route, p, &c, g, Arc::new(AtomicUsize::new(0))).unwrap();
            let mut osc = OscillatorBank.instrument(p, &c).unwrap();
            let (a, b) = (loudness(&mut sf, &phrase(p), 48_000), loudness(osc.as_mut(), &phrase(p), 48_000));
            assert!((a - b).abs() < 0.5, "{p:?}: {a:.1} vs {b:.1} LUFS");
        }
        // risers stay synthesized
        assert!(bank.route(Patch::Fx).is_none());
        // a drum kit the font lacks plays on its standard kit
        assert_eq!(bank.route(Patch::Kit(Kit::Cinematic)), Some(Route { channel: DRUM_CHANNEL, bank: DRUM_BANK, program: 0 }));
    }

    #[test]
    fn drums_ring_and_are_released_after_their_last_hit() {
        let font = SoundFont::from_bytes(&crate::synthetic::tiny()).unwrap();
        let route = Route { channel: DRUM_CHANNEL, bank: DRUM_BANK, program: 0 };
        let mut inst = SfInstrument::new(&font, route, Patch::Kit(Kit::Pop), &ctx(), 1.0, Arc::new(AtomicUsize::new(0))).unwrap();
        inst.note_on(Drum::Snare.gm_key(), 0.9);
        inst.note_off(Drum::Snare.gm_key()); // ignored: drums ring
        assert_eq!(inst.offs.len(), 1);
        inst.note_on(Drum::Snare.gm_key(), 0.9);
        assert_eq!(inst.offs.len(), 1, "a retrigger replaces the pending release");
        let [mut a, mut b, mut c, mut d] = bufs(48_000 * 2);
        inst.render(&mut Bus { dry_l: &mut a, dry_r: &mut b, send_l: &mut c, send_r: &mut d });
        assert!(inst.offs.is_empty(), "released after its ring");
        assert!(a.iter().take(4800).any(|x| x.abs() > 1e-3), "it sounds");
    }
}

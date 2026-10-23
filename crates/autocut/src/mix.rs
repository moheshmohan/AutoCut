//! The showcase's mix ([`mix_down`]): every sound clip typed by its role (Essential Sound), levelled
//! to its role's loudness, the music and the shots' own sound ducked under the narration, and,
//! by default, a mix made for small speakers ([`MixProfile::Speech`]).
//!
//! A TV, a laptop or a phone plays little below about 120 Hz: energy down there makes no sound a
//! viewer hears, but it fills the headroom, so the limiter (and the TV's own dynamics) push
//! everything else down and distort on it; and speech is understood at low volume from what it
//! has between 1 and 4 kHz. Measured on the first finished show (`audio.analyzeMix`), the cloned
//! narration had 37 % of its energy below 120 Hz and 1 % between 1 and 4 kHz, and its peaks
//! made the Mix's limiter take up to 3.7 dB off the whole mix. So, with `speech`:
//!
//! | stem | track effects (editable in the Audio Track Mixer) | level |
//! |---|---|---|
//! | narration | Parametric EQ: high-pass 80 Hz (24 dB/oct), low shelf −6 dB at 200 Hz, −2 dB at 300 Hz, +5 dB at 1.2 kHz, +5 dB at 3 kHz, high shelf +3 dB from 6 kHz (the cuts and lifts as far as the voice needs: measured, [`voice_eq_amount`]); DeEsser (6 kHz, light); compressor 3:1 from −20 dB, 1 ms attack (4–6 dB on speech), +6 dB out; Hard Limiter at −3.5 dB (the onsets the compressor lets through); the track's fader makes up the level the chain took, measured | each line matched to −16 LUFS |
//! | music | Parametric EQ: high-pass 40 Hz, low shelf −6 dB at 120 Hz, and a 4 dB dip around 2.5 kHz while the narration speaks (automation that follows the ducking) | −24 LUFS, ducked 18 dB under the narration |
//! | shots' own sound | Parametric EQ: high-pass 100 Hz, low shelf −3 dB at 150 Hz | −27 LUFS, ducked 15 dB |
//! | whooshes, impacts, risers | Parametric EQ: high-pass 120 Hz (no sub-bass booms) | −26 LUFS; the transitions' own sounds at −10 dB |
//!
//! The Mix keeps its loudness target (`--loudness`) and its true-peak limiter, which then has
//! little to do. `gentle` is the same mix with a voice-preserving narration EQ
//! ([`voice_chain_gentle`]: no low-mid cut, the lift above the vowels), for a cloned voice that
//! must stay closest to its speaker. `cinematic` is the balance before (full-range stems, a
//! shallower duck, louder effects). The effects are ordinary track inserts and keyframes, so the result stays editable.
//!
//! [`remix`] applies a profile to a showcase sequence built before (its track layout: A1 the
//! shots' sound, A2 the narration, A3 the music, A4 and A5 the effects), replacing what an
//! earlier mix put there; `filmcraft-cli showcase-remix` runs it.

use serde::Serialize;
use serde_json::{Value, json};

use crate::showcase::{run, ticks};
use crate::{AutoCutError, Executor};

/// How a show is mixed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MixProfile {
    /// Speech first, for small speakers (see the module docs).
    #[default]
    Speech,
    /// Speech first, with the voice-preserving narration EQ ([`voice_chain_gentle`]).
    Gentle,
    /// The earlier balance: full-range stems, the music ducked 15 dB, louder effects.
    Cinematic,
}

impl MixProfile {
    pub fn name(self) -> &'static str {
        match self {
            MixProfile::Speech => "speech",
            MixProfile::Gentle => "gentle",
            MixProfile::Cinematic => "cinematic",
        }
    }

    /// A plan's or the CLI's `mix`: `speech` (the default), `gentle` or `cinematic`.
    pub fn from_plan(v: &Value) -> Result<Self, AutoCutError> {
        match v {
            Value::Null => Ok(Self::default()),
            Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "speech" | "" => Ok(MixProfile::Speech),
                "gentle" => Ok(MixProfile::Gentle),
                "cinematic" => Ok(MixProfile::Cinematic),
                other => Err(AutoCutError::Invalid(format!("mix: speech, gentle or cinematic, not {other:?}"))),
            },
            other => Err(AutoCutError::Invalid(format!("mix: speech, gentle or cinematic, not {other}"))),
        }
    }

    /// The level of the sounds coupled to transitions (whooshes, risers), dB.
    pub fn transition_gain_db(self) -> f64 {
        match self {
            MixProfile::Speech | MixProfile::Gentle => -10.0,
            MixProfile::Cinematic => -6.0,
        }
    }

    /// The loudness of the shots' own sound, the music and the effects (LUFS) and how far the
    /// music and the shots' sound duck under the narration (dB).
    fn levels(self) -> Levels {
        match self {
            MixProfile::Speech | MixProfile::Gentle => Levels { sound: -27.0, music: -24.0, effects: -26.0, music_duck: -18.0, sound_duck: -15.0 },
            MixProfile::Cinematic => Levels { sound: -27.0, music: -24.0, effects: -23.0, music_duck: -15.0, sound_duck: -10.0 },
        }
    }
}

struct Levels {
    sound: f64,
    music: f64,
    effects: f64,
    music_duck: f64,
    sound_duck: f64,
}

/// The narration's loudness, each line matched to it (LUFS, measured as the mono clip it is).
pub const VOICE_LUFS: f64 = -16.0;
/// Ceiling of the limiter on the mix without a loudness target, dBFS.
pub const CEILING_DB: f64 = -1.0;
/// The shots' own sound when it cannot be measured.
const FALLBACK_SOUND_DB: f64 = -14.0;
/// How far the music's presence band dips while the narration speaks (dB), and where.
pub const CARVE_DB: f64 = -4.0;
const CARVE_HZ: f64 = 2500.0;
/// The ducking's fades (the "Duck Under Dialogue" presets'), seconds.
const DUCK_FADE_S: f64 = 0.8;
/// The most the narration's fader makes up for its chain (dB, either way).
const MAX_VOICE_MAKEUP_DB: f64 = 12.0;
/// The effects a mix puts on the tracks and the Mix ([`remix`] takes them off again first).
pub const MIX_INSERTS: &[&str] = &["parametric_eq", "deesser", "single_band_compressor", "hard_limiter"];

/// Parametric EQ bands the mix leaves off unless it sets them.
fn eq(set: Value) -> Value {
    let mut p = json!({"hp_on": false, "low_on": false, "b1_on": false, "b2_on": false, "mid_on": false, "b4_on": false, "b5_on": false, "high_on": false, "lp_on": false});
    if let (Some(o), Some(s)) = (p.as_object_mut(), set.as_object()) {
        for (k, v) in s {
            o.insert(k.clone(), v.clone());
        }
    }
    p
}

/// The tilt the narration's EQ aims for (dB): its 1–4 kHz octaves this far under its
/// 125–250 Hz ones ([`voice_tilt`]); a typical male voice sits about 10 dB down, and a touch
/// brighter carries better on small speakers.
pub const VOICE_TILT: f64 = -9.0;
/// How far the whole EQ ([`voice_chain`] at amount 1) moves a voice's tilt (dB), measured on the
/// cloned narration (−17.8 → −9.0 dB).
pub const VOICE_EQ_SPAN: f64 = 8.8;
/// The `gentle` profile's target tilt (dB): a typical voice, since every dB brighter moves a
/// cloned voice further from its speaker ([`voice_chain_gentle`]).
pub const VOICE_TILT_GENTLE: f64 = -10.0;
/// How far [`voice_chain_gentle`] at amount 1 moves a voice's tilt (dB), measured on the cloned
/// narration: IndicF5's Malayalam −18.2 → −10.0 dB, VoxCPM2's English −17.7 → −10.2 (the mean).
pub const VOICE_EQ_SPAN_GENTLE: f64 = 7.8;

/// A voice's tilt (dB): the mean level of its 1, 2 and 4 kHz octaves minus that of its 125 and
/// 250 Hz ones, from a stem's `octavesDb` (`audio.analyzeMix`). The cloned narration measured
/// −17.8 dB (dark: 37 % of its energy below 120 Hz, 1 % at 1–4 kHz). `None` when not measured.
pub fn voice_tilt(octaves: &Value) -> Option<f64> {
    let at = |k: &str| octaves[k].as_f64().filter(|x| x.is_finite());
    let hi = (at("1000")? + at("2000")? + at("4000")?) / 3.0;
    let lo = (at("125")? + at("250")?) / 2.0;
    Some(hi - lo)
}

/// How much of the EQ's correction a voice gets (0…1): all of it for a voice as dark as the
/// cloned narration, none for one already at [`VOICE_TILT`] or brighter (a bright voice made
/// brighter only gets harsher and peakier); all of it when the voice could not be measured.
pub fn voice_eq_amount(tilt: Option<f64>) -> f64 {
    eq_amount(tilt, VOICE_TILT, VOICE_EQ_SPAN)
}

/// [`voice_eq_amount`] for [`voice_chain_gentle`] (its target [`VOICE_TILT_GENTLE`]).
pub fn voice_eq_amount_gentle(tilt: Option<f64>) -> f64 {
    eq_amount(tilt, VOICE_TILT_GENTLE, VOICE_EQ_SPAN_GENTLE)
}

fn eq_amount(tilt: Option<f64>, target: f64, span: f64) -> f64 {
    match tilt {
        Some(t) if t.is_finite() && span > 0.0 => ((target - t) / span).clamp(0.0, 1.0),
        _ => 1.0,
    }
}

/// The narration's track effects (speech profile), in order: what each is and its settings, the
/// EQ's cuts and lifts scaled by `amount` ([`voice_eq_amount`]; the high-pass and half the low
/// shelf are always on). The author chose this one over [`voice_chain_gentle`] on their TV
/// (2026-10-21): the clearest words at low volume.
pub fn voice_chain(amount: f64) -> Vec<(&'static str, Value)> {
    let a = if amount.is_finite() { amount.clamp(0.0, 1.0) } else { 1.0 };
    let g = |db: f64| (db * a * 100.0).round() / 100.0;
    let eq_settings = eq(json!({
        // the rumble and the boom below the voice, the low-mid mud
        "hp_on": true, "hp_freq": 80.0, "hp_slope": 1,
        // (half the low shelf whatever the voice: small speakers play nothing down there)
        "low_on": true, "low_freq": 200.0, "low_gain": -3.0 + g(-3.0), "low_q": 0.71,
        "b1_on": true, "b1_freq": 300.0, "b1_gain": g(-2.0), "b1_q": 1.0,
        // presence: what makes words understood at low volume
        "mid_on": true, "mid_freq": 1200.0, "mid_gain": g(5.0), "mid_q": 0.8,
        "b4_on": true, "b4_freq": 3000.0, "b4_gain": g(5.0), "b4_q": 0.8,
        "high_on": true, "high_freq": 6000.0, "high_gain": g(3.0), "high_q": 0.71,
    }));
    with_dynamics(eq_settings)
}

/// The `gentle` profile's narration effects: as [`voice_chain`], the EQ voice-preserving, its
/// lifts scaled by `amount` ([`voice_eq_amount_gentle`]; the high-pass and the low shelf are
/// always on).
///
/// **Voice-preserving** (2026-10-21). Any EQ moves a voice away from its speaker's (the tone
/// colour of the readings against the speaker's genuine speech, `voice.mixCheck`), and how far
/// follows how much its balance changes; but not every band buys the same intelligibility for
/// it. Measured on the show's cloned narration, per band: the low-mid cuts (low shelf −6 dB,
/// −2 dB at 300 Hz) cost about twice the timbre a lift at 2–3.5 kHz costs for the same rise in
/// the 1–4 kHz share, and the lift at 1.2 kHz (where the vowels' formants are) half as much
/// again; consonants — what makes words understood on a TV — are at 1.4–5.7 kHz. So the low
/// shelf stays at the fixed −3 dB, the 300 Hz cut goes, and the lift moves up to 2 and 3.5 kHz:
/// the consonant band (the 2 and 4 kHz octaves' share) ends where the earlier chain left it
/// (−11.0 dB, from −20.4), with about a fifth less change of timbre on the Malayalam voice
/// (similarity to the speaker 0.921 → 0.867 instead of → 0.856) and a quarter less on the English
/// one (0.861 → 0.787 instead of → 0.759); the energy below 120 Hz 19 % instead of 15 %.
pub fn voice_chain_gentle(amount: f64) -> Vec<(&'static str, Value)> {
    let a = if amount.is_finite() { amount.clamp(0.0, 1.0) } else { 1.0 };
    let g = |db: f64| (db * a * 100.0).round() / 100.0;
    let eq_settings = eq(json!({
        // the rumble and the boom below the voice
        "hp_on": true, "hp_freq": 80.0, "hp_slope": 1,
        // (the low shelf whatever the voice: small speakers play nothing down there; no more,
        // and no low-mid cut: the voice's body is its speaker's)
        "low_on": true, "low_freq": 200.0, "low_gain": -3.0, "low_q": 0.71,
        // consonants: what makes words understood at low volume, above the vowels' formants
        "mid_on": true, "mid_freq": 2000.0, "mid_gain": g(5.0), "mid_q": 0.8,
        "b4_on": true, "b4_freq": 3500.0, "b4_gain": g(6.0), "b4_q": 0.8,
        "high_on": true, "high_freq": 6000.0, "high_gain": g(3.0), "high_q": 0.71,
    }));
    with_dynamics(eq_settings)
}

/// The narration's EQ followed by the dynamics both chains share.
fn with_dynamics(eq_settings: Value) -> Vec<(&'static str, Value)> {
    vec![
        ("parametric_eq", eq_settings),
        ("deesser", json!({"frequency": 6000.0, "threshold": -10.0, "reduction": 6.0})),
        // an even level: 3:1, fast enough to catch the syllables' onsets
        ("single_band_compressor", json!({"threshold": -20.0, "ratio": 3.0, "attack": 1.0, "release": 120.0, "output": 6.0})),
        // the few peaks left, so the Mix's limiter has (almost) nothing to do
        ("hard_limiter", json!({"max": -3.5, "boost": 0.0})),
    ]
}

/// The music's EQ (speech profile): no sub-bass, less bass, and a presence band at 0 dB that
/// [`carve_points`] dips while the narration speaks (its `mid_gain`).
pub fn music_eq() -> Value {
    eq(json!({
        "hp_on": true, "hp_freq": 40.0, "hp_slope": 1,
        "low_on": true, "low_freq": 120.0, "low_gain": -6.0, "low_q": 0.71,
        "mid_on": true, "mid_freq": CARVE_HZ, "mid_gain": 0.0, "mid_q": 0.7,
    }))
}

/// The shots' own sound's EQ (speech profile): wind and handling rumble out.
pub fn sound_eq() -> Value {
    eq(json!({"hp_on": true, "hp_freq": 100.0, "hp_slope": 1, "low_on": true, "low_freq": 150.0, "low_gain": -3.0, "low_q": 0.71}))
}

/// The effects' EQ (speech profile): no sub-bass booms.
pub fn effects_eq() -> Value {
    eq(json!({"hp_on": true, "hp_freq": 120.0, "hp_slope": 1}))
}

/// The music's EQ when it carries a video alone (a quick short nobody speaks in): the sub-bass
/// under [`BED_HP_HZ`] out — no small speaker plays it, and it fills the headroom the Mix's
/// limiter then has to make — and the low end `shelf_db` lower below 110 Hz (from
/// [`BED_SHELF_DB`], deeper as far as the bed needs: [`bed_shelf`]); the kick and the bass line
/// stay.
pub fn bed_eq(shelf_db: f64) -> Value {
    let shelf = if shelf_db.is_finite() { shelf_db.clamp(BED_SHELF_MIN_DB, 0.0) } else { BED_SHELF_DB };
    eq(json!({
        "hp_on": true, "hp_freq": BED_HP_HZ, "hp_slope": 1,
        "low_on": true, "low_freq": 110.0, "low_gain": (shelf * 100.0).round() / 100.0, "low_q": 0.71,
    }))
}

/// The bed's high-pass (Hz), its low shelf to start from and the deepest it goes (dB).
pub const BED_HP_HZ: f64 = 40.0;
pub const BED_SHELF_DB: f64 = -4.0;
pub const BED_SHELF_MIN_DB: f64 = -10.0;
/// The share of the bed's energy below 120 Hz its EQ aims for (the QA warns over 25 % of the
/// mix's; the generated electronic beds measured 30–35 % through the starting shelf).
pub const BED_LOW_SHARE: f64 = 0.18;

/// The low shelf that brings a bed measuring `share` of its energy below 120 Hz (through a shelf
/// of `shelf_db`) to [`BED_LOW_SHARE`]: the energy down there scaled so the share is the aim,
/// never above the starting shelf nor below [`BED_SHELF_MIN_DB`].
pub fn bed_shelf(shelf_db: f64, share: f64) -> f64 {
    if !(shelf_db.is_finite() && share.is_finite() && share > 0.0 && share < 1.0) {
        return if shelf_db.is_finite() { shelf_db.clamp(BED_SHELF_MIN_DB, BED_SHELF_DB) } else { BED_SHELF_DB };
    }
    let t = BED_LOW_SHARE;
    let k = t * (1.0 - share) / (share * (1.0 - t));
    (shelf_db + 10.0 * k.log10()).clamp(BED_SHELF_MIN_DB, BED_SHELF_DB)
}
/// The bed's peaks held this far over its loudness (dB, short of the −1 dBTP the Mix's limiter
/// keeps at −14 LUFS: 13 dB), so the limiter has (almost) nothing to do.
pub const BED_PEAK_OVER_LUFS: f64 = 11.0;

/// The music's dynamics when it carries a video alone, for a bed levelled to `lufs`: a gentle
/// compressor on its loudest moments, then a limiter holding its peaks [`BED_PEAK_OVER_LUFS`]
/// over its loudness — the generated beds' drums peak about 18 dB over it.
pub fn bed_dynamics(lufs: f64) -> Vec<(&'static str, Value)> {
    let lufs = if lufs.is_finite() { lufs.clamp(-40.0, -6.0) } else { -16.0 };
    let ceiling = ((lufs + BED_PEAK_OVER_LUFS).min(-1.0) * 10.0).round() / 10.0;
    vec![
        ("single_band_compressor", json!({"threshold": lufs + 4.0, "ratio": 2.0, "attack": 8.0, "release": 150.0, "output": 0.0})),
        ("hard_limiter", json!({"max": ceiling, "boost": 0.0})),
    ]
}

/// The music's presence dip as automation points `(ticks, dB)`: 0 dB, down to [`CARVE_DB`] over
/// `fade` seconds before each stretch the narration speaks (`regions`, seconds; from
/// `essentialSound.generateDucking`), back up after it, as the ducking does.
pub fn carve_points(regions: &[(f64, f64)], fade: f64) -> Vec<(i64, f64)> {
    let end = regions.iter().map(|r| r.1).filter(|b| b.is_finite()).fold(0.0f64, f64::max);
    filmcraft_audio_dsp::ducking::duck_keyframes(regions, 0.0, end + fade.max(0.0) + 1.0, 0.0, CARVE_DB, fade)
        .into_iter()
        .map(|(t, v)| (ticks(t), (v * 100.0).round() / 100.0))
        .collect()
}

/// The sound clips of a show by role.
#[derive(Clone, Debug, Default)]
pub struct Mix {
    /// The shots' own sound.
    pub sounds: Vec<u64>,
    /// Whooshes, impacts, risers.
    pub effects: Vec<u64>,
    /// The narration.
    pub narrations: Vec<u64>,
    pub music: Vec<u64>,
    /// Enhance Voice on the shots' own sound (talking clips): `audio.enhanceVoice` before it is
    /// levelled, the level left to the mix.
    pub enhance_voice: bool,
}

/// Where the stems are (audio track labels).
#[derive(Clone, Debug)]
pub struct MixTracks {
    pub sounds: String,
    pub voice: String,
    pub music: String,
    /// The effects' tracks.
    pub effects: Vec<String>,
}

impl MixTracks {
    /// A showcase's layout: A1 the shots' sound, A2 the narration, A3 the music, A4 and A5 the
    /// effects.
    pub fn showcase() -> Self {
        Self { sounds: "A1".into(), voice: "A2".into(), music: "A3".into(), effects: vec!["A4".into(), "A5".into()] }
    }
}

/// Put `chain` on `strip` (after what is there); the slot of each effect, `None` for one that
/// could not be added or set.
pub(crate) async fn inserts(ex: &dyn Executor, strip: &str, chain: &[(&str, Value)], notes: &mut Vec<String>) -> Vec<Option<u64>> {
    let mut slots = Vec::new();
    for (effect, params) in chain {
        let slot = match run(ex, "mixer.addInsert", json!({"strip": strip, "effect": effect})).await {
            Ok(r) => r["slot"].as_u64(),
            Err(e) => {
                notes.push(format!("mix: {strip}: {effect}: {e}"));
                None
            }
        };
        if let Some(s) = slot
            && let Err(e) = run(ex, "mixer.setInsert", json!({"strip": strip, "slot": s, "params": params})).await
        {
            notes.push(format!("mix: {strip}: {effect}: {e}"));
        }
        slots.push(slot);
    }
    slots
}

/// Mix the show (see the module docs): type every clip, match its loudness to its role's, duck
/// the shots' sound and the music under the narration, put the profile's effects on the stems'
/// tracks, and a limiter on the Mix when there is no loudness target (`limit`). Steps that fail
/// are noted, never fatal: the edit is still usable unmixed. Returns what it did.
pub async fn mix_down(ex: &dyn Executor, mix: &Mix, tracks: &MixTracks, profile: MixProfile, limit: bool, notes: &mut Vec<String>) -> Value {
    let lv = profile.levels();
    let speech = profile != MixProfile::Cinematic;
    let step = async |id: &str, p: Value, notes: &mut Vec<String>| match run(ex, id, p).await {
        Ok(v) => Some(v),
        Err(e) => {
            notes.push(format!("mix: {e}"));
            None
        }
    };
    if !mix.narrations.is_empty() {
        step("essentialSound.setType", json!({"clips": mix.narrations, "type": "dialogue"}), notes).await;
        step("essentialSound.autoMatch", json!({"clips": mix.narrations, "target": VOICE_LUFS}), notes).await;
    }
    if !mix.sounds.is_empty() {
        if mix.enhance_voice {
            step("audio.enhanceVoice", json!({"clips": mix.sounds, "lufs": "none"}), notes).await;
        }
        step("essentialSound.applyPreset", json!({"clips": mix.sounds, "preset": "Duck Under Dialogue", "type": "ambience"}), notes).await;
        if speech {
            step("essentialSound.set", json!({"clips": mix.sounds, "key": "ducking.reduceDb", "value": lv.sound_duck}), notes).await;
        }
        if step("essentialSound.autoMatch", json!({"clips": mix.sounds, "target": lv.sound}), notes).await.is_none() {
            step("clip.audioGain", json!({"clips": mix.sounds, "mode": "set", "db": FALLBACK_SOUND_DB}), notes).await;
        }
    }
    if !mix.effects.is_empty() {
        step("essentialSound.setType", json!({"clips": mix.effects, "type": "sfx"}), notes).await;
        step("essentialSound.autoMatch", json!({"clips": mix.effects, "target": lv.effects}), notes).await;
    }
    if !mix.music.is_empty() {
        step("essentialSound.applyPreset", json!({"clips": mix.music, "preset": "Duck Under Dialogue", "type": "music"}), notes).await;
        if speech {
            step("essentialSound.set", json!({"clips": mix.music, "key": "ducking.reduceDb", "value": lv.music_duck}), notes).await;
        }
        step("essentialSound.autoMatch", json!({"clips": mix.music, "target": lv.music}), notes).await;
    }
    // ducking: the stretches where the narration speaks, for the music's presence dip too
    let mut regions: Vec<(f64, f64)> = Vec::new();
    if !mix.narrations.is_empty() {
        let ducked: Vec<u64> = mix.sounds.iter().chain(&mix.music).copied().collect();
        if !ducked.is_empty()
            && let Some(r) = step("essentialSound.generateDucking", json!({"clips": ducked}), notes).await
        {
            for c in r["clips"].as_array().into_iter().flatten().filter(|c| c["clip"].as_u64().is_some_and(|id| mix.music.contains(&id))) {
                regions.extend(c["regions"].as_array().into_iter().flatten().filter_map(|x| Some((x[0].as_f64()?, x[1].as_f64()?))));
            }
        }
    }
    let mut report = json!({"profile": profile.name()});
    if speech {
        if !mix.narrations.is_empty() {
            // the voice as it is (levelled, nothing on its track yet): how dark and how loud
            let measure = async || run(ex, "audio.analyzeMix", json!({"only": [tracks.voice]})).await;
            let before = match measure().await {
                Ok(a) => Some(a),
                Err(e) => {
                    notes.push(format!("mix: the narration could not be measured ({e}); its EQ is the full correction"));
                    None
                }
            };
            let tilt = before.as_ref().and_then(|a| voice_tilt(&a["stems"][0]["octavesDb"]));
            let (amount, chain) = if profile == MixProfile::Gentle {
                let a = voice_eq_amount_gentle(tilt);
                (a, voice_chain_gentle(a))
            } else {
                let a = voice_eq_amount(tilt);
                (a, voice_chain(a))
            };
            let slots = inserts(ex, &tracks.voice, &chain, notes).await;
            // the chain changes the voice's tone, not its level against the other stems: its
            // fader makes up what the EQ and the compressor took (measured again through them)
            let loudness = |a: &Value| a["stems"][0]["integratedLufs"].as_f64().filter(|x| x.is_finite());
            let after = measure().await.ok();
            let makeup = match (before.as_ref().and_then(loudness), after.as_ref().and_then(loudness)) {
                (Some(l0), Some(l1)) => Some((l0 - l1).clamp(-MAX_VOICE_MAKEUP_DB, MAX_VOICE_MAKEUP_DB)),
                _ => None,
            };
            if let Some(db) = makeup {
                let fader = (db * 100.0).round() / 100.0;
                step("mixer.setStrip", json!({"strip": tracks.voice, "volumeDb": fader}), notes).await;
            }
            report["voice"] = json!({
                "track": tracks.voice,
                "effects": chain.iter().map(|c| c.0).collect::<Vec<_>>(),
                "added": slots.iter().flatten().count(),
                "tiltDb": tilt.map(|t| (t * 10.0).round() / 10.0),
                "eqAmount": (amount * 100.0).round() / 100.0,
                "loudnessLufs": before.as_ref().and_then(loudness),
                "faderDb": makeup.map(|d| (d * 100.0).round() / 100.0),
            });
        }
        if !mix.music.is_empty() {
            let slots = inserts(ex, &tracks.music, &[("parametric_eq", music_eq())], notes).await;
            if let Some(Some(slot)) = slots.first() {
                let points = carve_points(&regions, DUCK_FADE_S);
                if !points.is_empty() {
                    let pts: Vec<Value> = points.iter().map(|(t, v)| json!([t, v])).collect();
                    let lane = format!("fx.{slot}.mid_gain");
                    step("mixer.writeAutomation", json!({"strip": tracks.music, "lane": lane, "points": pts, "tolerance": 0.0}), notes).await;
                }
                report["musicCarve"] = json!({"track": tracks.music, "db": CARVE_DB, "hz": CARVE_HZ, "stretches": regions.len()});
            }
        }
        if !mix.sounds.is_empty() {
            inserts(ex, &tracks.sounds, &[("parametric_eq", sound_eq())], notes).await;
        }
        if !mix.effects.is_empty() {
            for t in &tracks.effects {
                inserts(ex, t, &[("parametric_eq", effects_eq())], notes).await;
            }
        }
    }
    // (a mix normalised to a loudness target gets its limiter from the normalisation)
    if limit && let Some(r) = step("mixer.addInsert", json!({"strip": "Mix", "effect": "hard_limiter", "postFader": true}), notes).await {
        let slot = r["slot"].as_u64().unwrap_or(0);
        step("mixer.setInsert", json!({"strip": "Mix", "slot": slot, "params": {"max": CEILING_DB}}), notes).await;
    }
    report
}

/// Re-mix a showcase sequence built before (the active one, or `sequence`) with `profile`: the
/// track effects a mix puts on its stems' tracks ([`MIX_INSERTS`]) and their automation are taken
/// off, every clip is typed, levelled and ducked again, the profile's effects put on, the
/// transitions' sounds set to the profile's level and, with `loudness`, the mix normalised (else
/// the Mix keeps a limiter). Clips, cuts and the picture are not touched.
pub async fn remix(
    ex: &dyn Executor,
    sequence: Option<u64>,
    profile: MixProfile,
    loudness: Option<&crate::look::Loudness>,
    notes: &mut Vec<String>,
) -> Result<Value, AutoCutError> {
    if let Some(id) = sequence {
        run(ex, "sequence.open", json!({"item": id})).await?;
    }
    let seq = run(ex, "sequence.inspect", json!({})).await?;
    let tracks = MixTracks::showcase();
    let clips_on = |label: &str| -> Vec<u64> {
        let k = label.trim_start_matches('A').parse::<usize>().ok().and_then(|k| k.checked_sub(1));
        k.and_then(|k| seq["audio"].get(k))
            .and_then(|t| t["items"].as_array())
            .map(|a| a.iter().take(100_000).filter_map(|c| c["clip"].as_u64()).collect())
            .unwrap_or_default()
    };
    let mix = Mix {
        sounds: clips_on(&tracks.sounds),
        effects: tracks.effects.iter().flat_map(|t| clips_on(t)).collect(),
        narrations: clips_on(&tracks.voice),
        music: clips_on(&tracks.music),
        enhance_voice: false,
    };
    if mix.narrations.is_empty() && mix.music.is_empty() && mix.sounds.is_empty() {
        return Err(AutoCutError::Invalid("remix: the sequence has no showcase sound (A1 the shots, A2 the narration, A3 the music)".into()));
    }
    // what an earlier mix put on the stems' tracks: their effects (from the last slot back) and
    // the automation of those effects
    let mixer = run(ex, "mixer.inspect", json!({})).await?;
    let mut ours = vec![tracks.sounds.clone(), tracks.voice.clone(), tracks.music.clone()];
    ours.extend(tracks.effects.iter().cloned());
    let mut removed = 0usize;
    for strip in mixer["strips"].as_array().into_iter().flatten().filter(|s| s["ref"].as_str().is_some_and(|r| ours.iter().any(|o| o == r))) {
        let label = strip["ref"].as_str().unwrap_or_default();
        for lane in strip["lanes"].as_object().into_iter().flat_map(|l| l.keys()).filter(|k| k.starts_with("fx.")) {
            run(ex, "mixer.clearLane", json!({"strip": label, "lane": lane})).await.ok();
        }
        let mut slots: Vec<u64> = strip["inserts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|i| i["effect"].as_str().is_some_and(|e| MIX_INSERTS.contains(&e)))
            .filter_map(|i| i["slot"].as_u64())
            .collect();
        slots.sort_unstable_by(|a, b| b.cmp(a));
        for slot in slots {
            if run(ex, "mixer.removeInsert", json!({"strip": label, "slot": slot})).await.is_ok() {
                removed += 1;
            }
        }
    }
    // the narration's fader is the mix's (it makes up for the chain): back to 0 dB first
    run(ex, "mixer.setStrip", json!({"strip": tracks.voice, "volumeDb": 0.0})).await.ok();
    // the Mix keeps one limiter: a normalisation sets it, and without one it stays as it is
    let has_master_limiter = mixer["strips"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| s["kind"] == "mix")
        .flat_map(|s| s["inserts"].as_array().into_iter().flatten())
        .any(|i| i["effect"] == "hard_limiter");
    let mut report = mix_down(ex, &mix, &tracks, profile, loudness.is_none() && !has_master_limiter, notes).await;
    report["removedEffects"] = json!(removed);
    // the transitions' sounds at the profile's level (the same sound: kind, intensity, seed)
    let mut sounds = 0usize;
    for t in seq["video"].as_array().into_iter().flatten().flat_map(|t| t["transitions"].as_array().into_iter().flatten()).take(10_000) {
        let (Some(id), Some(kind)) = (t["id"].as_u64(), t["sound"]["kind"].as_str()) else { continue };
        let mut p =
            json!({"transitions": [id], "kind": kind, "gain": profile.transition_gain_db(), "intensity": t["sound"]["intensity"], "seed": t["sound"]["seed"]});
        if let Some(len) = t["sound"]["length"].as_i64() {
            p["seconds"] = json!(len as f64 / filmcraft_time::TICKS_PER_SECOND as f64);
        }
        match run(ex, "transitions.addSound", p).await {
            Ok(_) => sounds += 1,
            Err(e) => notes.push(format!("mix: transition {id}'s sound: {e}")),
        }
    }
    report["transitionSounds"] = json!(sounds);
    if let Some(l) = loudness {
        match run(ex, "audio.normalizeLoudness", json!({"target": l.target(), "truePeak": crate::look::TRUE_PEAK})).await {
            Ok(r) => report["loudness"] = r,
            Err(e) => notes.push(format!("loudness: {e}")),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_parse_and_default_to_speech() {
        assert_eq!(MixProfile::from_plan(&Value::Null).unwrap(), MixProfile::Speech);
        assert_eq!(MixProfile::from_plan(&json!("Cinematic")).unwrap(), MixProfile::Cinematic);
        assert_eq!(MixProfile::from_plan(&json!(" speech ")).unwrap(), MixProfile::Speech);
        assert_eq!(MixProfile::from_plan(&json!("Gentle")).unwrap(), MixProfile::Gentle);
        for bad in [json!("loud"), json!(3), json!(true), json!(["speech"])] {
            assert!(MixProfile::from_plan(&bad).is_err(), "{bad}");
        }
        assert!(MixProfile::Speech.transition_gain_db() < MixProfile::Cinematic.transition_gain_db());
        let (s, c) = (MixProfile::Speech.levels(), MixProfile::Cinematic.levels());
        assert!(s.music_duck < c.music_duck && s.sound_duck < c.sound_duck && s.effects < c.effects);
        // gentle is speech first too: only the narration's EQ differs
        let g = MixProfile::Gentle.levels();
        assert!(g.music_duck == s.music_duck && g.sound_duck == s.sound_duck && g.effects == s.effects);
        assert_eq!(MixProfile::Gentle.transition_gain_db(), MixProfile::Speech.transition_gain_db());
    }

    #[test]
    fn the_chains_are_the_engines_effects_with_their_own_parameters() {
        let mut all: Vec<(&str, Value)> = voice_chain(1.0);
        all.extend(voice_chain_gentle(1.0));
        all.extend([("parametric_eq", music_eq()), ("parametric_eq", sound_eq()), ("parametric_eq", effects_eq())]);
        for (effect, params) in &all {
            let def = filmcraft_engine::project::find_effect(effect).unwrap_or_else(|| panic!("{effect} is not an effect"));
            for (k, v) in params.as_object().unwrap() {
                let p = def.param(k).unwrap_or_else(|| panic!("{effect} has no parameter {k}"));
                match (&p.kind, v) {
                    (filmcraft_engine::project::ParamKind::Float { min, max, .. }, Value::Number(x)) => {
                        let x = x.as_f64().unwrap();
                        assert!((*min..=*max).contains(&x), "{effect}.{k} = {x} outside {min}…{max}");
                    }
                    (filmcraft_engine::project::ParamKind::Bool, Value::Bool(_)) => {}
                    (filmcraft_engine::project::ParamKind::Choice(c), Value::Number(i)) => assert!((i.as_u64().unwrap() as usize) < c.len(), "{effect}.{k}"),
                    (kind, v) => panic!("{effect}.{k}: {v} for a {kind:?}"),
                }
            }
        }
        // the narration: rumble out, presence up, peaks held
        let eq = &voice_chain(1.0)[0].1;
        assert_eq!((eq["hp_on"].as_bool(), eq["hp_freq"].as_f64()), (Some(true), Some(80.0)));
        assert!(eq["b4_gain"].as_f64().unwrap() > 0.0 && eq["low_gain"].as_f64().unwrap() < -3.0);
        assert_eq!(voice_chain(1.0).iter().map(|c| c.0).collect::<Vec<_>>(), ["parametric_eq", "deesser", "single_band_compressor", "hard_limiter"]);
        // a voice bright already: the high-pass only; half way: half the lift
        let flat = &voice_chain(0.0)[0].1;
        assert_eq!((flat["hp_on"].as_bool(), flat["b4_gain"].as_f64(), flat["low_gain"].as_f64()), (Some(true), Some(0.0), Some(-3.0)));
        assert_eq!(voice_chain(0.5)[0].1["b4_gain"].as_f64(), Some(2.5));
        assert_eq!(voice_chain(f64::NAN)[0].1["b4_gain"].as_f64(), Some(5.0));
        // gentle: no low-mid cut, the lift above the vowels' formants (voice-preserving), the same
        // dynamics after it
        let eq = &voice_chain_gentle(1.0)[0].1;
        assert!(eq["b4_gain"].as_f64().unwrap() > 0.0 && eq["low_gain"].as_f64().unwrap() == -3.0);
        assert_eq!(eq["b1_on"].as_bool(), Some(false));
        assert!(eq["mid_freq"].as_f64().unwrap() >= 2000.0 && eq["b4_freq"].as_f64().unwrap() >= 3000.0);
        assert_eq!(voice_chain_gentle(1.0)[1..], voice_chain(1.0)[1..]);
        assert_eq!(voice_chain_gentle(0.5)[0].1["b4_gain"].as_f64(), Some(3.0));
        assert_eq!(voice_chain_gentle(f64::NAN)[0].1["b4_gain"].as_f64(), Some(6.0));
    }

    #[test]
    fn the_voice_eq_follows_how_dark_the_voice_is() {
        // the cloned narration as measured (octave levels relative to the whole)
        let dark = json!({"63": -6.9, "125": -3.4, "250": -6.7, "500": -9.7, "1000": -22.0, "2000": -23.1, "4000": -23.4, "8000": -34.8});
        let t = voice_tilt(&dark).unwrap();
        assert!((t - -17.8).abs() < 0.05, "{t}");
        assert!(voice_eq_amount(Some(t)) > 0.99);
        // after the chain it sits at the target: nothing more to do
        let treated = json!({"125": -4.7, "250": -6.4, "1000": -14.9, "2000": -13.4, "4000": -15.2});
        assert!((voice_tilt(&treated).unwrap() - VOICE_TILT).abs() < 0.1);
        assert!(voice_eq_amount(voice_tilt(&treated)) < 0.02);
        // a bright voice gets none, one in between some; unmeasured: the full correction
        assert_eq!(voice_eq_amount(Some(-3.0)), 0.0);
        assert!((voice_eq_amount(Some(-13.4)) - 0.5).abs() < 1e-9);
        // gentle aims a dB darker, over its own span
        assert!(voice_eq_amount_gentle(Some(t)) > 0.99);
        assert!((voice_eq_amount_gentle(Some(-13.9)) - 0.5).abs() < 1e-9);
        assert_eq!(voice_eq_amount_gentle(Some(-9.5)), 0.0);
        assert_eq!(voice_eq_amount_gentle(Some(f64::INFINITY)), 1.0);
        assert_eq!(voice_eq_amount(None), 1.0);
        assert_eq!(voice_eq_amount(Some(f64::NAN)), 1.0);
        assert!(voice_tilt(&json!({"125": -3.0})).is_none() && voice_tilt(&json!({"125": null, "250": "x"})).is_none());
    }

    #[test]
    fn the_music_dips_where_the_narration_speaks() {
        let p = carve_points(&[(2.0, 5.0), (9.0, 12.0)], 0.8);
        let at = |s: f64| ticks(s);
        assert_eq!(p.first(), Some(&(at(1.2), 0.0)));
        assert!(p.contains(&(at(2.0), CARVE_DB)) && p.contains(&(at(5.0), CARVE_DB)) && p.contains(&(at(5.8), 0.0)));
        assert!(p.windows(2).all(|w| w[0].0 < w[1].0), "{p:?}");
        assert!(p.iter().all(|(_, v)| (CARVE_DB..=0.0).contains(v)));
        // nothing spoken, or nonsense: no points, no panic
        assert!(carve_points(&[], 0.8).is_empty());
        assert!(carve_points(&[(f64::NAN, 3.0), (4.0, 2.0)], f64::NAN).is_empty());
    }
}

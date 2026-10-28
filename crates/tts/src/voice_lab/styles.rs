//! The style finder: a speaker's IndicF5 reference candidates sorted by delivery.
//!
//! IndicF5 continues its reference clip, so the clip's delivery carries into every line read
//! from it: a lively clip reads lively. The finder measures the prosody of each candidate (whole
//! phrases, 6–12 s, ending at a sentence end) with [`filmcraft_audio_dsp::prosody`] and places
//! it against **the speaker's own spread** over all his candidates, so the styles adapt to the
//! speaker — a quiet speaker's excited is not a loud one's.
//!
//! **Features** ([`Features`]): pitch level (median F0), pitch range (10–90 %, semitones), pitch
//! movement (semitones a second), energy variation over syllable peaks (dB), syllable rate (a
//! second of speech), voiced fraction, and the final contour (semitones: + rising).
//!
//! **Normalising** ([`norms`]): per feature the median and spread (1.4826 × the median absolute
//! deviation) of the speaker's candidates, each pulled towards a prior (the author's measured
//! speech, [`PRIORS`]) with the weight of [`PRIOR_WEIGHT`] candidates — a handful of candidates
//! cannot make a style out of noise — and the spread never under 30 % of the prior's. Each
//! candidate's z-score per feature (clamped to ±3); the pitch level's centre is the speaker's
//! own median always.
//!
//! **Rules** ([`scores`]), with `hi(z) = 1 / (1 + e^(−1.6 z))`, `lo(z) = hi(−z)`,
//! `mid(z) = e^(−z²)` and `rise(F) = 1 / (1 + e^(−(F − 1.5) / 0.6))` of the final contour `F`:
//!
//! | style | score |
//! |---|---|
//! | `excited` | 0.25 hi(range) + 0.2 hi(movement) + 0.2 hi(energy variation) + 0.2 hi(rate) + 0.15 hi(pitch) |
//! | `amazed` | 0.35 hi(range) + 0.3 lo(rate) + 0.2 hi(pitch) + 0.15 hi(energy variation): wide and high, but slower |
//! | `curious` | 0.7 rise(F) + 0.3 max(hi(movement), mid(rate)): a rising end |
//! | `calm` | (0.3 lo(range) + 0.2 lo(movement) + 0.3 lo(rate) + 0.2 lo(energy variation)) × (1 − 0.4 rise) |
//! | `warm` | (0.35 lo(pitch) + 0.2 mid(range) + 0.25 mid(rate + 0.7) + 0.2 lo(energy variation)) × (1 − 0.4 rise): lower, unhurried, still melodic |
//! | `plain` | (0.35 lo(range) + 0.25 lo(movement) + 0.25 mid(rate) + 0.15 mid(energy variation)) × (1 − 0.4 rise): flat at an ordinary pace |
//! | `conversational` | (0.3 mid(range) + 0.2 mid(movement) + 0.3 mid(rate) + 0.2 mid(energy variation)) × (1 − 0.4 rise): the speaker's ordinary |
//!
//! Each candidate's **shares** are a softmax of its scores (temperature [`TEMPERATURE`]); its
//! style is the largest and its **confidence** that share (1/7 when nothing stands out, near 1
//! when one style clearly does). [`pick`] lists the best `top` per style by share × (0.6 + 0.4 ×
//! the reference score), never two overlapping stretches of one clip, and only shares of at least
//! [`MIN_SHARE`]: a style the speaker's clips do not hold stays empty rather than being filled
//! with a poor match. `dramatic` is not told apart by prosody alone and is not offered.

use serde::Serialize;

use super::Candidate;
use crate::TtsError;
use filmcraft_audio_dsp::prosody::{self, Expressiveness, PitchSettings};

/// A delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Style {
    Calm,
    Plain,
    Conversational,
    Warm,
    Excited,
    Amazed,
    Curious,
}

impl Style {
    pub const ALL: [Style; 7] = [Style::Calm, Style::Plain, Style::Conversational, Style::Warm, Style::Excited, Style::Amazed, Style::Curious];

    pub fn id(self) -> &'static str {
        match self {
            Style::Calm => "calm",
            Style::Plain => "plain",
            Style::Conversational => "conversational",
            Style::Warm => "warm",
            Style::Excited => "excited",
            Style::Amazed => "amazed",
            Style::Curious => "curious",
        }
    }

    pub fn from_id(s: &str) -> Option<Style> {
        Style::ALL.into_iter().find(|t| t.id() == s)
    }

    /// The rule, in words (for reports).
    pub fn rule(self) -> &'static str {
        match self {
            Style::Calm => "narrow, slow, steady: low pitch range and movement, slow, even syllables",
            Style::Plain => "flat at an ordinary pace: low pitch range and movement",
            Style::Conversational => "the speaker's ordinary: everything near his middle",
            Style::Warm => "lower pitch, unhurried, still melodic, even syllables",
            Style::Excited => "wide, moving, punchy and fast: high range, movement, energy variation, rate and pitch",
            Style::Amazed => "wide and high but slower: high range and pitch, a slower rate",
            Style::Curious => "a rising end (the last 100 ms of voicing above the 400 ms before)",
        }
    }
}

/// The prosody of a stretch (an [`Expressiveness`], as reports show it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Features {
    pub f0_median_hz: Option<f64>,
    pub f0_range_st: f64,
    pub f0_movement_st_s: f64,
    pub final_contour_st: f64,
    pub energy_variation_db: f64,
    pub syllable_rate: f64,
    pub voiced_fraction: f64,
    pub speech_seconds: f64,
}

impl From<&Expressiveness> for Features {
    fn from(e: &Expressiveness) -> Self {
        Features {
            f0_median_hz: e.f0_median_hz,
            f0_range_st: e.f0_range_st,
            f0_movement_st_s: e.f0_movement_st_s,
            final_contour_st: e.final_contour_st,
            energy_variation_db: e.energy_variation_db,
            syllable_rate: e.syllable_rate,
            voiced_fraction: e.voiced_fraction,
            speech_seconds: e.speech_seconds,
        }
    }
}

/// The prosody of `x` (mono at `rate`), its pitch tracked at 60–500 Hz.
pub fn features(x: &[f32], rate: u32) -> Result<Features, TtsError> {
    let e = prosody::expressiveness(x, rate, &PitchSettings::default()).map_err(|e| TtsError::Invalid(e.to_string()))?;
    Ok(Features::from(&e))
}

/// A centre and a spread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Norm {
    pub centre: f64,
    pub spread: f64,
}

/// What a feature is, on the speaker's scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Norms {
    /// Candidates measured (with a pitch).
    pub count: usize,
    /// Pitch level, semitones above 100 Hz.
    pub pitch_st: Norm,
    pub range_st: Norm,
    pub movement_st_s: Norm,
    pub energy_variation_db: Norm,
    pub syllable_rate: Norm,
}

/// The priors: the author's own conversational Malayalam, measured over his 19 IndicF5
/// candidates in `IMG_3337.MOV` / `IMG_3338.MOV` (crate README, "Delivery styles"): medians and
/// robust spreads, rounded. Pitch has no prior centre (the speaker's own median is it).
pub const PRIORS: Norms = Norms {
    count: 0,
    pitch_st: Norm { centre: 0.0, spread: 1.0 },
    range_st: Norm { centre: 3.8, spread: 1.0 },
    movement_st_s: Norm { centre: 9.5, spread: 2.0 },
    energy_variation_db: Norm { centre: 4.5, spread: 1.0 },
    syllable_rate: Norm { centre: 4.7, spread: 0.6 },
};
/// How many candidates' worth the priors weigh.
pub const PRIOR_WEIGHT: f64 = 4.0;
/// The softmax temperature of the shares.
pub const TEMPERATURE: f64 = 0.08;
/// The least share a pick may have.
pub const MIN_SHARE: f64 = 0.2;

fn median(v: &mut [f64]) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return None;
    }
    let (a, b) = (v.get((n - 1) / 2).copied()?, v.get(n / 2).copied()?);
    Some(0.5 * (a + b))
}

fn norm_of(values: &[f64], prior: Norm, own_centre: bool) -> Norm {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    let n = v.len() as f64;
    let Some(m) = median(&mut v) else { return prior };
    let mut dev: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    let mad = 1.4826 * median(&mut dev).unwrap_or(0.0);
    let centre = if own_centre { m } else { (n * m + PRIOR_WEIGHT * prior.centre) / (n + PRIOR_WEIGHT) };
    let spread = ((n * mad * mad + PRIOR_WEIGHT * prior.spread * prior.spread) / (n + PRIOR_WEIGHT)).sqrt().max(0.3 * prior.spread);
    Norm { centre, spread }
}

fn pitch_st(f: &Features) -> Option<f64> {
    f.f0_median_hz.filter(|h| h.is_finite() && *h > 0.0).map(|h| 12.0 * (h / 100.0).log2())
}

/// The speaker's scale from all his candidates' features (see the module docs).
pub fn norms(all: &[Features]) -> Norms {
    let voiced: Vec<&Features> = all.iter().filter(|f| pitch_st(f).is_some()).collect();
    let col = |g: &dyn Fn(&Features) -> f64| voiced.iter().map(|f| g(f)).collect::<Vec<f64>>();
    let pitch: Vec<f64> = voiced.iter().filter_map(|f| pitch_st(f)).collect();
    Norms {
        count: voiced.len(),
        pitch_st: norm_of(&pitch, PRIORS.pitch_st, true),
        range_st: norm_of(&col(&|f| f.f0_range_st), PRIORS.range_st, false),
        movement_st_s: norm_of(&col(&|f| f.f0_movement_st_s), PRIORS.movement_st_s, false),
        energy_variation_db: norm_of(&col(&|f| f.energy_variation_db), PRIORS.energy_variation_db, false),
        syllable_rate: norm_of(&col(&|f| f.syllable_rate), PRIORS.syllable_rate, false),
    }
}

/// A candidate's features as z-scores on the speaker's scale (±3 at most).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Z {
    pub pitch: f64,
    pub range: f64,
    pub movement: f64,
    pub energy_variation: f64,
    pub rate: f64,
}

fn z1(x: f64, n: Norm) -> f64 {
    let z = (x - n.centre) / n.spread.max(1e-9);
    if z.is_finite() { z.clamp(-3.0, 3.0) } else { 0.0 }
}

pub fn z(f: &Features, n: &Norms) -> Z {
    Z {
        pitch: pitch_st(f).map_or(0.0, |p| z1(p, n.pitch_st)),
        range: z1(f.f0_range_st, n.range_st),
        movement: z1(f.f0_movement_st_s, n.movement_st_s),
        energy_variation: z1(f.energy_variation_db, n.energy_variation_db),
        rate: z1(f.syllable_rate, n.syllable_rate),
    }
}

fn hi(z: f64) -> f64 {
    1.0 / (1.0 + (-1.6 * z).exp())
}

fn lo(z: f64) -> f64 {
    hi(-z)
}

fn mid(z: f64) -> f64 {
    (-z * z).exp()
}

/// How clearly a final contour of `semitones` rises (0…1).
pub fn rise(semitones: f64) -> f64 {
    if !semitones.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + (-(semitones - 1.5) / 0.6).exp())
}

/// Each style's score (0…1) for z-scores `z` and a final contour of `final_st` semitones (the
/// rules in the module docs), in [`Style::ALL`] order.
pub fn scores(z: &Z, final_st: f64) -> [(Style, f64); 7] {
    let q = rise(final_st);
    let level = 1.0 - 0.4 * q;
    let (r, m, e, s, p) = (z.range, z.movement, z.energy_variation, z.rate, z.pitch);
    [
        (Style::Calm, (0.3 * lo(r) + 0.2 * lo(m) + 0.3 * lo(s) + 0.2 * lo(e)) * level),
        (Style::Plain, (0.35 * lo(r) + 0.25 * lo(m) + 0.25 * mid(s) + 0.15 * mid(e)) * level),
        (Style::Conversational, (0.3 * mid(r) + 0.2 * mid(m) + 0.3 * mid(s) + 0.2 * mid(e)) * level),
        (Style::Warm, (0.35 * lo(p) + 0.2 * mid(r) + 0.25 * mid(s + 0.7) + 0.2 * lo(e)) * level),
        (Style::Excited, 0.25 * hi(r) + 0.2 * hi(m) + 0.2 * hi(e) + 0.2 * hi(s) + 0.15 * hi(p)),
        (Style::Amazed, 0.35 * hi(r) + 0.3 * lo(s) + 0.2 * hi(p) + 0.15 * hi(e)),
        (Style::Curious, 0.7 * q + 0.3 * hi(m).max(mid(s))),
    ]
}

/// What a candidate's delivery is.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Assessment {
    pub style: Style,
    /// The style's share (0…1): how clearly it is this style rather than another.
    pub confidence: f64,
    pub z: Z,
    /// Each style's score (0…1).
    pub scores: Vec<(Style, f64)>,
    /// Each style's share (softmax of the scores; they add up to 1).
    pub shares: Vec<(Style, f64)>,
}

impl Assessment {
    pub fn share(&self, s: Style) -> f64 {
        self.shares.iter().find(|(t, _)| *t == s).map_or(0.0, |(_, v)| *v)
    }
}

/// The delivery of a candidate with features `f` on the speaker's scale `n`.
pub fn assess(f: &Features, n: &Norms) -> Assessment {
    let z = z(f, n);
    let sc = scores(&z, f.final_contour_st);
    let top = sc.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<(Style, f64)> = sc.iter().map(|(s, v)| (*s, if v.is_finite() { ((v - top) / TEMPERATURE).exp() } else { 0.0 })).collect();
    let total: f64 = exps.iter().map(|(_, v)| v).sum();
    let shares: Vec<(Style, f64)> = exps.iter().map(|(s, v)| (*s, if total > 0.0 { v / total } else { 1.0 / 7.0 })).collect();
    let (style, confidence) = shares.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((Style::Conversational, 0.0));
    Assessment { style, confidence, z, scores: sc.to_vec(), shares }
}

/// One pick: candidate `index` (into the list given to [`pick`]) as style `style`'s `rank`-th.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Pick {
    pub style: Style,
    /// 1 for the best.
    pub rank: usize,
    pub index: usize,
    /// Share × (0.6 + 0.4 × reference score).
    pub fit: f64,
    /// The style is the candidate's own (its largest share), not a second choice.
    pub best: bool,
}

/// The best `top` candidates per style (see the module docs): `all` pairs each candidate with
/// its assessment.
pub fn pick(all: &[(Candidate, Assessment)], top: usize) -> Vec<Pick> {
    let mut out = Vec::new();
    for style in Style::ALL {
        let mut ranked: Vec<(usize, f64)> = all
            .iter()
            .enumerate()
            .filter(|(_, (_, a))| a.share(style) >= MIN_SHARE)
            .map(|(i, (c, a))| (i, a.share(style) * (0.6 + 0.4 * c.score.clamp(0.0, 1.0))))
            .filter(|(_, f)| f.is_finite())
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut kept: Vec<(usize, f64)> = Vec::new();
        for (i, fit) in ranked {
            if kept.len() >= top {
                break;
            }
            let Some((c, _)) = all.get(i) else { continue };
            let clash = kept.iter().any(|(k, _)| all.get(*k).is_some_and(|(o, _)| o.clip == c.clip && o.span.overlaps(&c.span)));
            if !clash {
                kept.push((i, fit));
            }
        }
        for (rank, (index, fit)) in kept.into_iter().enumerate() {
            let best = all.get(index).is_some_and(|(_, a)| a.style == style);
            out.push(Pick { style, rank: rank + 1, index, fit, best });
        }
    }
    out
}

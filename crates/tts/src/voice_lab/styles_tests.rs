use super::styles::*;
use super::*;

const RATE: u32 = 24_000;

/// Seeded noise in −1…1.
fn noise(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32
}

/// Features `k` spreads (the priors') from the priors' centres, around a 100 Hz voice.
fn feat(pitch: f64, range: f64, movement: f64, energy: f64, rate: f64, final_st: f64) -> Features {
    let p = PRIORS;
    Features {
        f0_median_hz: Some(100.0 * ((pitch * p.pitch_st.spread) / 12.0).exp2()),
        f0_range_st: p.range_st.centre + range * p.range_st.spread,
        f0_movement_st_s: p.movement_st_s.centre + movement * p.movement_st_s.spread,
        final_contour_st: final_st,
        energy_variation_db: p.energy_variation_db.centre + energy * p.energy_variation_db.spread,
        syllable_rate: p.syllable_rate.centre + rate * p.syllable_rate.spread,
        voiced_fraction: 0.6,
        speech_seconds: 8.0,
    }
}

/// The priors' own scale with the pitch centred on 100 Hz.
fn scale() -> Norms {
    Norms { count: 20, pitch_st: Norm { centre: 0.0, spread: PRIORS.pitch_st.spread }, ..PRIORS }
}

#[test]
fn each_archetype_gets_its_style() {
    let n = scale();
    let cases = [
        (Style::Excited, feat(1.5, 2.0, 2.0, 2.0, 2.0, 0.0)),
        (Style::Amazed, feat(1.5, 2.0, 0.5, 0.5, -1.5, 0.0)),
        (Style::Curious, feat(0.0, 0.0, 0.3, 0.0, 0.0, 3.5)),
        (Style::Calm, feat(-0.3, -1.5, -1.5, -1.5, -1.5, -1.0)),
        (Style::Plain, feat(0.0, -1.5, -1.5, 0.0, 0.0, -1.0)),
        (Style::Warm, feat(-1.8, 0.0, 0.0, -1.2, -0.7, -1.0)),
        (Style::Conversational, feat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ];
    for (want, f) in cases {
        let a = assess(&f, &n);
        assert_eq!(a.style, want, "{a:?}");
        let total: f64 = a.shares.iter().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(a.confidence > 1.0 / 7.0 && a.confidence <= 1.0, "{a:?}");
        assert!(a.scores.iter().all(|(_, s)| (0.0..=1.0).contains(s)));
    }
    // a slight rise is no question
    assert_eq!(assess(&feat(0.0, 0.0, 0.0, 0.0, 0.0, 0.8), &n).style, Style::Conversational);
    for s in Style::ALL {
        assert_eq!(Style::from_id(s.id()), Some(s));
        assert!(!s.rule().is_empty());
    }
    assert_eq!(Style::from_id("dramatic"), None);
}

#[test]
fn the_scale_is_the_speakers_own() {
    // a speaker with a narrow range (about 3 semitones) and one with a wide one (about 8)
    let narrow: Vec<Features> =
        (0..30).map(|i| Features { f0_median_hz: Some(95.0), f0_range_st: 2.5 + 0.04 * i as f64, ..feat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }).collect();
    let wide: Vec<Features> =
        (0..30).map(|i| Features { f0_median_hz: Some(180.0), f0_range_st: 7.0 + 0.07 * i as f64, ..feat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }).collect();
    let (a, b) = (norms(&narrow), norms(&wide));
    assert_eq!(a.count, 30);
    let probe = Features { f0_range_st: 5.5, ..narrow[0] };
    assert!(z(&probe, &a).range > 1.0, "{:?}", z(&probe, &a));
    assert!(z(&Features { f0_median_hz: Some(180.0), ..probe }, &b).range < -1.0);
    // the pitch level is measured from the speaker's own median
    assert!((a.pitch_st.centre - 12.0 * (0.95f64).log2()).abs() < 1e-9);
    assert!(z(&Features { f0_median_hz: Some(95.0), ..probe }, &a).pitch.abs() < 1e-9);
    // a handful of candidates leans on the priors; the spread never collapses
    let few = norms(&narrow[..2]);
    assert!(few.range_st.centre > 2.6 && few.range_st.centre < PRIORS.range_st.centre);
    let same: Vec<Features> = vec![narrow[0]; 40];
    assert!(norms(&same).range_st.spread >= 0.3 * PRIORS.range_st.spread);
    // no candidates (or none with a pitch): the priors
    assert_eq!(norms(&[]).range_st, PRIORS.range_st);
    assert_eq!(norms(&[Features::default()]).count, 0);
}

#[test]
fn hostile_features_do_no_harm() {
    let n = scale();
    let bad = Features {
        f0_median_hz: Some(f64::NAN),
        f0_range_st: f64::INFINITY,
        f0_movement_st_s: f64::NAN,
        final_contour_st: f64::NAN,
        energy_variation_db: -1e300,
        syllable_rate: f64::NEG_INFINITY,
        voiced_fraction: f64::NAN,
        speech_seconds: 0.0,
    };
    let a = assess(&bad, &n);
    assert!(a.shares.iter().all(|(_, v)| v.is_finite()) && a.confidence.is_finite());
    let nn = norms(&[bad, bad, Features { f0_median_hz: Some(0.0), ..bad }]);
    assert!(nn.range_st.spread.is_finite() && nn.range_st.centre.is_finite());
    assert!(rise(f64::NAN) == 0.0 && rise(10.0) > 0.99 && rise(-5.0) < 0.01);
    assert!(features(&[], 24_000).is_ok_and(|f| f == Features::default()));
    assert!(features(&[0.1; 100], 0).is_err());
    assert!(features(&[f32::NAN; 24_000], 24_000).is_ok());
}

fn candidate(clip: &str, start: f64, score: f64) -> Candidate {
    Candidate {
        id: String::new(),
        clip: clip.into(),
        span: Span::new(start, start + 8.0),
        target: Target::Indicf5,
        language: "ml".into(),
        similarity: 0.9,
        quality: Quality::default(),
        sentence: 1.0,
        ends_sentence: true,
        length_fit: 1.0,
        language_fit: 1.0,
        score,
        words: String::new(),
    }
}

#[test]
fn picks_are_ranked_spread_and_never_overlap() {
    let n = scale();
    let excited = feat(1.5, 2.0, 2.0, 2.0, 2.0, 0.0);
    let calm = feat(-0.3, -1.5, -1.5, -1.5, -1.5, -1.0);
    let all: Vec<(Candidate, Assessment)> = vec![
        (candidate("a.mov", 0.0, 0.9), assess(&excited, &n)),
        // overlaps the first: never both in one list
        (candidate("a.mov", 4.0, 0.95), assess(&excited, &n)),
        (candidate("b.mov", 0.0, 0.5), assess(&excited, &n)),
        (candidate("b.mov", 20.0, 0.7), assess(&calm, &n)),
        // the same stretch of another clip is not a clash
        (candidate("c.mov", 4.0, 0.6), assess(&excited, &n)),
    ];
    let p = pick(&all, 2);
    let of = |s: Style| p.iter().filter(|x| x.style == s).collect::<Vec<_>>();
    let ex = of(Style::Excited);
    assert_eq!(ex.len(), 2);
    assert_eq!((ex[0].rank, ex[1].rank), (1, 2));
    assert_eq!(ex[0].index, 1, "the better reference first");
    assert!(ex[0].fit >= ex[1].fit && ex.iter().all(|x| x.best));
    assert!(ex.iter().all(|x| x.index != 0), "{ex:?}");
    let ca = of(Style::Calm);
    assert_eq!(ca.len(), 1);
    assert_eq!(ca[0].index, 3);
    // nobody curious: an empty list, not a poor match
    assert!(of(Style::Curious).is_empty());
    assert!(pick(&all, 0).is_empty() && pick(&[], 3).is_empty());
}

/// Speech-like audio: syllables (`rate` a second, 60 % voiced, levels alternating by
/// `energy_db`) of a harmonic voice whose F0 sweeps `range_st` semitones up and down twice a
/// second around `f0`, over a quiet floor.
fn talk(secs: f64, f0: f64, range_st: f64, rate: f64, energy_db: f64) -> Vec<f32> {
    let n = (secs * RATE as f64) as usize;
    let mut phase = 0.0f64;
    let mut seed = 77u64;
    (0..n)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            let f = f0 * ((range_st / 2.0) * (std::f64::consts::TAU * 0.5 * t).sin() / 12.0).exp2();
            phase += std::f64::consts::TAU * f / RATE as f64;
            let k = (t * rate).floor();
            let u = t * rate - k;
            let level = if (k as i64) % 2 == 0 { 1.0 } else { 10f64.powf(-energy_db / 20.0) };
            let env = if u < 0.6 { (std::f64::consts::PI * u / 0.6).sin() * level } else { 0.0 };
            let tone: f64 = (1..30).map(|h| (h as f64 * phase).sin() / h as f64).sum();
            (0.2 * env * tone) as f32 + 0.002 * noise(&mut seed)
        })
        .collect()
}

#[test]
fn lively_and_flat_speech_are_told_apart() {
    let lively = features(&talk(8.0, 120.0, 9.0, 5.5, 8.0), RATE).unwrap();
    let flat = features(&talk(8.0, 100.0, 1.0, 3.5, 1.0), RATE).unwrap();
    assert!(lively.f0_range_st > 3.0 * flat.f0_range_st, "{lively:?} {flat:?}");
    assert!(lively.syllable_rate > flat.syllable_rate + 1.0, "{lively:?} {flat:?}");
    assert!(lively.energy_variation_db > flat.energy_variation_db + 3.0, "{lively:?} {flat:?}");
    // on a speaker's scale made of both and a few in between
    let mid = features(&talk(8.0, 110.0, 4.0, 4.5, 4.0), RATE).unwrap();
    let n = norms(&[lively, flat, mid, mid, mid]);
    let (a, b) = (assess(&lively, &n), assess(&flat, &n));
    assert!(matches!(a.style, Style::Excited | Style::Amazed), "{a:?}");
    assert!(matches!(b.style, Style::Calm | Style::Plain | Style::Warm), "{b:?}");
}

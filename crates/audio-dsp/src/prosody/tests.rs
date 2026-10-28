use super::*;
use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::testutil::Rng;
use std::f64::consts::PI;

/// A voiced, glottal-like signal with F0 `f0(t)`: harmonics falling 6 dB an octave (a glottal
/// source with lip radiation) up to 4 kHz, shaped by two formant resonances (700 Hz and 1.2
/// kHz), amplitude `amp(t)`.
fn voice(rate: u32, secs: f64, f0: impl Fn(f64) -> f64, amp: impl Fn(f64) -> f64) -> Vec<f32> {
    let sr = f64::from(rate);
    let n = (secs * sr) as usize;
    let mut phase = 0.0f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr;
        let f = f0(t);
        phase += 2.0 * PI * f / sr;
        let hmax = ((0.45 * sr).min(4000.0) / f) as usize;
        let v: f64 = (1..=hmax.max(1)).map(|h| (h as f64 * phase).sin() / h as f64).sum();
        out.push((v * amp(t)) as f32);
    }
    let f1 = Coeffs::design(FilterType::Peaking, sr, 700.0, 4.0, 12.0);
    let f2 = Coeffs::design(FilterType::Peaking, sr, 1200.0, 4.0, 8.0);
    let (mut b1, mut b2) = (Biquad::new(), Biquad::new());
    b1.process(&f1, &mut out);
    b2.process(&f2, &mut out);
    let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
    let target = (0..n).map(|i| amp(i as f64 / sr)).fold(0.0, f64::max) as f32;
    out.iter_mut().for_each(|v| *v *= 0.5 * target / peak);
    out
}

fn noise(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| rng.uniform() * amp).collect()
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += *b);
}

fn rms_db(x: &[f32]) -> f64 {
    let e = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * (e + 1e-20).log10()
}

fn max_jump(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
}

/// Relative errors of the voiced frames of `t` against `truth` (Hz or None per frame time),
/// skipping frames within `margin` s of a change in the truth's voicing; and the share of the
/// truly voiced frames tracked as voiced, and of the truly unvoiced tracked as unvoiced.
fn errors(t: &PitchTrack, truth: impl Fn(f64) -> Option<f64>, margin: f64) -> (Vec<f64>, f64, f64) {
    let mut errs = Vec::new();
    let (mut v_tot, mut v_hit, mut u_tot, mut u_hit) = (0, 0, 0, 0);
    for i in 0..t.len() {
        let time = t.time(i);
        let around = [time - margin, time, time + margin].map(|x| truth(x).is_some());
        if around.iter().any(|v| *v != around[1]) {
            continue;
        }
        match truth(time) {
            Some(f) => {
                v_tot += 1;
                if let Some(e) = t.f0[i] {
                    v_hit += 1;
                    errs.push((f64::from(e) - f).abs() / f);
                }
            }
            None => {
                u_tot += 1;
                if t.f0[i].is_none() {
                    u_hit += 1;
                }
            }
        }
    }
    (errs, v_hit as f64 / v_tot.max(1) as f64, u_hit as f64 / u_tot.max(1) as f64)
}

fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5).unwrap_or(f64::NAN)
}

#[test]
fn tracks_voices_and_sines_between_silence_and_noise() {
    for rate in [16_000u32, 24_000, 44_100, 48_000] {
        let sr = f64::from(rate);
        let mut rng = Rng::new(u64::from(rate));
        // 0.3 s silence, 1 s voice at 140 Hz, 0.3 s silence, 1 s sine at 230 Hz, 0.3 s silence
        let mut x = vec![0.0f32; (0.3 * sr) as usize];
        x.extend(voice(rate, 1.0, |_| 140.0, |_| 0.5));
        x.extend(vec![0.0f32; (0.3 * sr) as usize]);
        x.extend((0..sr as usize).map(|i| (0.4 * (2.0 * PI * 230.0 * i as f64 / sr).sin()) as f32));
        x.extend(vec![0.0f32; (0.3 * sr) as usize]);
        // noise 34 dB under the speech everywhere
        let nz = noise(&mut rng, x.len(), 0.01);
        add(&mut x, &nz);
        let t = track(&x, rate, &PitchSettings::default()).unwrap();
        assert_eq!(t.len(), (x.len() as f64 / sr / HOP_S) as usize + 1);
        let truth = |time: f64| {
            if (0.3..1.3).contains(&time) {
                Some(140.0)
            } else if (1.6..2.6).contains(&time) {
                Some(230.0)
            } else {
                None
            }
        };
        let (errs, voiced, unvoiced) = errors(&t, truth, 0.04);
        let med = median(&errs);
        assert!(med < 0.01, "{rate} Hz: median error {med}");
        assert!(errs.iter().all(|e| *e < 0.05), "{rate} Hz: worst error {:?}", errs.iter().copied().fold(0.0, f64::max));
        assert!(voiced > 0.95, "{rate} Hz: {voiced} of the voiced frames tracked");
        assert!(unvoiced > 0.95, "{rate} Hz: {unvoiced} of the silent frames unvoiced");
        // periodicity: high in the voice, low in the noise
        assert!(t.confidence[80] > 0.8 && t.confidence[10] < 0.5, "{} {}", t.confidence[80], t.confidence[10]);
    }
}

#[test]
fn glides_are_followed_without_octave_jumps() {
    let rate = 24_000u32;
    let mut rng = Rng::new(7);
    // up from 90 to 260 Hz in 2 s, down to 120 Hz in 1 s (exponential glides)
    let f0 = |t: f64| if t < 2.0 { 90.0 * (260.0f64 / 90.0).powf(t / 2.0) } else { 260.0 * (120.0f64 / 260.0).powf((t - 2.0).min(1.0)) };
    let mut x = voice(rate, 3.0, f0, |_| 0.5);
    // 20 dB under the voice
    let nz = noise(&mut rng, x.len(), 0.05);
    add(&mut x, &nz);
    let t = track(&x, rate, &PitchSettings::default()).unwrap();
    let (errs, voiced, _) = errors(&t, |time| (0.0..3.0).contains(&time).then(|| f0(time)), 0.03);
    assert!(median(&errs) < 0.01, "median error {}", median(&errs));
    // no octave jumps: every frame within 4 % of the truth
    let worst = errs.iter().copied().fold(0.0, f64::max);
    assert!(worst < 0.04, "worst error {worst}");
    assert!(voiced > 0.97, "{voiced} voiced");
}

#[test]
fn low_and_high_voices() {
    let rate = 16_000u32;
    for (f, s) in [
        (65.0, PitchSettings::default()),
        (110.0, PitchSettings::default()),
        (420.0, PitchSettings::default()),
        (700.0, PitchSettings { min_hz: 150.0, max_hz: 1000.0, voicing: 0.4 }),
    ] {
        let x = voice(rate, 1.0, |_| f, |_| 0.5);
        let t = track(&x, rate, &s).unwrap();
        let (errs, voiced, _) = errors(&t, |time| (0.0..1.0).contains(&time).then_some(f), 0.04);
        assert!(median(&errs) < 0.01, "{f} Hz: median error {}", median(&errs));
        assert!(errs.iter().all(|e| *e < 0.03), "{f} Hz: an octave error");
        assert!(voiced > 0.95, "{f} Hz: {voiced}");
    }
}

#[test]
fn hostile_input_is_refused_or_empty() {
    let s = PitchSettings::default();
    let t = track(&[], 16_000, &s).unwrap();
    assert!(t.is_empty() && t.median_hz().is_none() && t.runs().is_empty());
    for rate in [0u32, 1, 3_999, 768_001, u32::MAX] {
        assert_eq!(track(&[0.0; 100], rate, &s), Err(ProsodyError::Rate(rate)));
    }
    // non-finite samples are silence
    let x = vec![f32::NAN; 8000];
    let t = track(&x, 16_000, &s).unwrap();
    assert_eq!(t.voiced(), 0);
    assert!(t.confidence.iter().all(|c| c.is_finite()) && t.level_db.iter().all(|l| l.is_finite()));
    let mut x = voice(16_000, 1.0, |_| 150.0, |_| 0.5);
    for i in (0..x.len()).step_by(997) {
        x[i] = if i % 2 == 0 { f32::INFINITY } else { f32::NAN };
    }
    let t = track(&x, 16_000, &s).unwrap();
    assert!(t.voiced() > 50, "{}", t.voiced());
    for bad in [
        PitchSettings { min_hz: f64::NAN, ..s },
        PitchSettings { min_hz: 10.0, ..s },
        PitchSettings { max_hz: 70.0, ..s },
        PitchSettings { max_hz: f64::INFINITY, ..s },
        PitchSettings { voicing: 2.0, ..s },
        PitchSettings { voicing: -1.0, ..s },
    ] {
        assert!(matches!(track(&[0.0; 100], 16_000, &bad), Err(ProsodyError::Settings(_))), "{bad:?}");
    }
    // the highest pitch needs four samples a period
    assert!(track(&[0.0; 100], 4_000, &PitchSettings { min_hz: 200.0, max_hz: 1500.0, voicing: 0.4 }).is_err());
    assert!(matches!(check_input(usize::MAX, 4_000), Err(ProsodyError::TooLong(_))));
    assert!(check_input((MAX_SECONDS * 48_000.0) as usize, 48_000).is_ok());
    // one sample, a tiny buffer
    assert_eq!(track(&[0.5], 16_000, &s).unwrap().len(), 1);
}

#[test]
fn a_minute_is_tracked_quickly() {
    let rate = 48_000u32;
    let x = voice(rate, 60.0, |t| 120.0 + 30.0 * (t * 2.0).sin(), |_| 0.5);
    let t0 = std::time::Instant::now();
    let t = track(&x, rate, &PitchSettings::default()).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(t.len(), 6001);
    // about a tenth of a second in an optimised build; generous for debug runs on a busy machine
    assert!(secs < 6.0, "a minute of audio took {secs} s");
}

/// An utterance with a moving contour: 150 Hz ± 5 semitones, 0.7 Hz.
fn contour(t: f64) -> f64 {
    150.0 * ((5.0 / 12.0) * (2.0 * PI * 0.7 * t).sin()).exp2()
}

fn range_st(t: &PitchTrack) -> f64 {
    let lf: Vec<f64> = features::smoothed_log2(t).into_iter().flatten().collect();
    12.0 * (quantile(&lf, 0.9).unwrap() - quantile(&lf, 0.1).unwrap())
}

#[test]
fn a_neutral_shape_returns_the_input_unchanged() {
    let rate = 24_000;
    let mut x = voice(rate, 1.0, contour, |_| 0.5);
    let s = PitchSettings::default();
    assert_eq!(reshape(&x, rate, &Shape::default(), &s).unwrap(), x);
    // bit for bit, whatever it holds
    x[10] = f32::NAN;
    let y = reshape(&x, rate, &Shape::default(), &s).unwrap();
    assert!(y[10].is_nan() && y.len() == x.len());
    // every ratio 1: the input
    let x = voice(rate, 1.0, contour, |_| 0.5);
    let t = track(&x, rate, &s).unwrap();
    assert_eq!(psola(&x, rate, &t, &vec![1.0; t.len()]).unwrap(), x);
    assert_eq!(shape_ratios(&t, &Shape::default()), vec![1.0; t.len()]);
}

#[test]
fn overlap_add_at_ratio_one_rebuilds_the_signal() {
    // a ratio a hair from 1 forces the resynthesis; its grains must add back to the input
    let rate = 24_000;
    let x = voice(rate, 1.0, contour, |_| 0.5);
    let t = track(&x, rate, &PitchSettings::default()).unwrap();
    let y = psola(&x, rate, &t, &vec![1.0 + 1e-7; t.len()]).unwrap();
    let diff = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(diff < 1e-4, "max difference {diff}");
}

#[test]
fn range_scales_the_contour() {
    for (rate, want) in [(24_000, 0.6), (24_000, 1.3), (24_000, 1.6), (16_000, 1.5), (48_000, 2.0)] {
        range_case(rate, want);
    }
}

fn range_case(rate: u32, want: f64) {
    let mut rng = Rng::new(3);
    let mut x = voice(rate, 3.0, contour, |_| 0.5);
    let nz = noise(&mut rng, x.len(), 0.003);
    add(&mut x, &nz);
    let s = PitchSettings::default();
    let before = track(&x, rate, &s).unwrap();
    let r0 = range_st(&before);
    assert!((r0 - 9.0).abs() < 1.0, "the input's range {r0}");
    {
        let y = reshape(&x, rate, &Shape { range: want, ..Default::default() }, &s).unwrap();
        assert_eq!(y.len(), x.len());
        let after = track(&y, rate, &s).unwrap();
        let got = range_st(&after) / r0;
        assert!((got / want - 1.0).abs() < 0.05, "range {want}: got {got}");
        // the median stays
        let (m0, m1) = (before.median_hz().unwrap(), after.median_hz().unwrap());
        assert!((m1 / m0 - 1.0).abs() < 0.03, "range {want}: median {m0} → {m1}");
        // energy kept
        let de = rms_db(&y) - rms_db(&x);
        assert!(de.abs() < 1.0, "range {want}: energy {de} dB");
        // no clicks
        let (jx, jy) = (max_jump(&x), max_jump(&y));
        assert!(jy < 2.0 * jx, "range {want}: largest step {jy} against {jx}");
        assert!(y.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn shift_moves_the_whole_contour() {
    let rate = 16_000;
    let x = voice(rate, 2.0, contour, |_| 0.5);
    let s = PitchSettings::default();
    for st in [-4.0, 3.0] {
        let y = reshape(&x, rate, &Shape { shift: st, ..Default::default() }, &s).unwrap();
        let (m0, m1) = (track(&x, rate, &s).unwrap().median_hz().unwrap(), track(&y, rate, &s).unwrap().median_hz().unwrap());
        let got = semitones(m0, m1);
        assert!((got - st).abs() < 0.25, "shift {st}: {got} semitones");
    }
}

#[test]
fn a_question_rises_at_its_end() {
    let rate = 24_000;
    let sr = f64::from(rate);
    // 1.5 s at 150 Hz, then 0.3 s of silence
    let mut x = voice(rate, 1.5, |_| 150.0, |_| 0.5);
    x.extend(vec![0.0; (0.3 * sr) as usize]);
    let s = PitchSettings::default();
    let y = reshape(&x, rate, &Shape { question_rise: 4.0, question_end: Some(1.6), ..Default::default() }, &s).unwrap();
    let t = track(&y, rate, &s).unwrap();
    let last = t.f0.iter().rposition(Option::is_some).unwrap();
    let end: Vec<f64> = t.f0[last - 4..=last].iter().flatten().map(|f| f64::from(*f)).collect();
    let rise = semitones(150.0, median(&end));
    assert!((rise - 4.0).abs() < 0.6, "the end rose {rise} semitones");
    // the start is left alone
    let start: Vec<f64> = t.f0[10..90].iter().flatten().map(|f| f64::from(*f)).collect();
    assert!((median(&start) / 150.0 - 1.0).abs() < 0.01);
    // and the features see a rising end
    let e = expressiveness_of(&y, rate, &t).unwrap();
    assert!(e.final_contour_st > 1.5, "{}", e.final_contour_st);
    let e0 = expressiveness(&x, rate, &s).unwrap();
    assert!(e0.final_contour_st.abs() < 0.3, "{}", e0.final_contour_st);
}

#[test]
fn unvoiced_audio_is_passed_through() {
    let rate = 24_000;
    let sr = f64::from(rate);
    let mut rng = Rng::new(11);
    let hiss = noise(&mut rng, (0.4 * sr) as usize, 0.1);
    let mut x = hiss.clone();
    x.extend(voice(rate, 1.0, contour, |_| 0.5));
    x.extend(&hiss);
    let y = reshape(&x, rate, &Shape { range: 1.6, ..Default::default() }, &PitchSettings::default()).unwrap();
    let (a, b) = ((0.35 * sr) as usize, (1.45 * sr) as usize);
    assert_eq!(&y[..a], &x[..a], "the hiss before");
    assert_eq!(&y[b..], &x[b..], "the hiss after");
    assert_ne!(&y[a..b], &x[a..b]);
    assert!(max_jump(&y) < 2.0 * max_jump(&x));
}

#[test]
fn voicing_boundaries_are_smooth() {
    // a voice that starts and stops abruptly between silences
    let rate = 24_000;
    let sr = f64::from(rate);
    let mut x = vec![0.0f32; (0.2 * sr) as usize];
    for _ in 0..3 {
        x.extend(voice(rate, 0.25, |t| 130.0 + 200.0 * t, |_| 0.4));
        x.extend(vec![0.0f32; (0.1 * sr) as usize]);
    }
    let y = reshape(&x, rate, &Shape { range: 1.6, shift: 2.0, ..Default::default() }, &PitchSettings::default()).unwrap();
    assert_eq!(y.len(), x.len());
    assert!(max_jump(&y) < 2.0 * max_jump(&x), "{} against {}", max_jump(&y), max_jump(&x));
}

/// Syllables: 120 ms voiced (a raised-cosine envelope) and 80 ms of quiet hiss, five a second,
/// for `secs`; levels alternating between 0 and −6 dB; F0 falling linearly in semitones from
/// `hi` to `lo` around 140 Hz over the stretch.
fn syllable_train(rate: u32, secs: f64, hi: f64, lo: f64, rng: &mut Rng) -> Vec<f32> {
    let sr = f64::from(rate);
    let f0 = |t: f64| 140.0 * ((hi + (lo - hi) * t / secs) / 12.0).exp2();
    let amp = |t: f64| {
        let k = (t / 0.2).floor();
        let u = t - 0.2 * k;
        let level = if (k as i64) % 2 == 0 { 0.5 } else { 0.25 };
        if u < 0.12 { level * (0.5 - 0.5 * (2.0 * PI * u / 0.12).cos()).sqrt() } else { 0.0 }
    };
    let mut x = voice(rate, secs, f0, amp);
    // voice() normalises its peak: put the envelope back
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    x.iter_mut().for_each(|v| *v *= 0.25 / peak);
    let n = noise(rng, x.len(), 0.004);
    add(&mut x, &n);
    let _ = sr;
    x
}

#[test]
fn features_of_a_syllable_train() {
    let rate = 16_000;
    let sr = f64::from(rate);
    let mut rng = Rng::new(5);
    let mut x = syllable_train(rate, 2.0, 3.0, -3.0, &mut rng);
    x.extend(noise(&mut rng, (0.5 * sr) as usize, 0.004));
    x.extend(syllable_train(rate, 2.0, 3.0, -3.0, &mut rng));
    let e = expressiveness(&x, rate, &PitchSettings::default()).unwrap();
    assert!((e.seconds - 4.5).abs() < 0.01);
    assert!((e.syllable_rate - 5.0).abs() < 0.8, "{e:?}");
    assert!((19..=21).contains(&e.syllables), "{e:?}");
    assert!((e.energy_variation_db - 3.0).abs() < 1.2, "{e:?}");
    let m = e.f0_median_hz.unwrap();
    assert!((m / 140.0 - 1.0).abs() < 0.03, "{e:?}");
    // F0 falls 6 semitones in 2 s: 3 semitones a second; the 10th–90th percentile of a uniform
    // fall over 6 semitones is 4.8
    assert!((e.f0_movement_st_s - 3.0).abs() < 1.2, "{e:?}");
    assert!((e.f0_range_st - 4.8).abs() < 0.8, "{e:?}");
    assert!((0.4..=0.8).contains(&e.voiced_fraction), "{e:?}");
    assert!(e.speech_seconds > 3.0 && e.speech_seconds < 4.4, "{e:?}");
    assert!(e.final_contour_st < -0.1, "{e:?}");
}

#[test]
fn a_flat_reading_scores_flatter_than_a_lively_one() {
    let rate = 16_000;
    let mut rng = Rng::new(9);
    let flat = syllable_train(rate, 3.0, 0.5, -0.5, &mut rng);
    let lively = syllable_train(rate, 3.0, 6.0, -6.0, &mut rng);
    let s = PitchSettings::default();
    let (a, b) = (expressiveness(&flat, rate, &s).unwrap(), expressiveness(&lively, rate, &s).unwrap());
    assert!(b.f0_range_st > 4.0 * a.f0_range_st.max(0.2), "{a:?} {b:?}");
    assert!(b.f0_movement_st_s > 3.0 * a.f0_movement_st_s.max(0.1), "{a:?} {b:?}");
}

#[test]
fn emphasis_lifts_the_high_syllables() {
    let rate = 16_000;
    let sr = f64::from(rate);
    // syllables alternating between 120 and 170 Hz (6 semitones apart), equally loud
    let f0 = |t: f64| if ((t / 0.2).floor() as i64) % 2 == 0 { 120.0 } else { 170.0 };
    let amp = |t: f64| {
        let u = t - 0.2 * (t / 0.2).floor();
        if u < 0.14 { 0.5 * (0.5 - 0.5 * (2.0 * PI * u / 0.14).cos()) } else { 0.0 }
    };
    let x = voice(rate, 2.0, f0, amp);
    let s = PitchSettings::default();
    let y = reshape(&x, rate, &Shape { emphasis_db: 4.0, ..Default::default() }, &s).unwrap();
    assert_eq!(y.len(), x.len());
    let seg = |v: &[f32], k: usize| rms_db(&v[(k as f64 * 0.2 * sr) as usize..((k as f64 * 0.2 + 0.14) * sr) as usize]);
    let (mut hi, mut lo) = (Vec::new(), Vec::new());
    for k in 1..9 {
        let d = seg(&y, k) - seg(&x, k);
        if k % 2 == 1 { hi.push(d) } else { lo.push(d) }
    }
    assert!(median(&hi) > 1.5 && median(&hi) < 4.5, "high syllables lifted {hi:?}");
    assert!(median(&lo) < 0.8, "low syllables lifted {lo:?}");
    assert!(y.iter().all(|v| v.abs() <= 0.98));
    // never over the ceiling, even when the input is hot
    let hot: Vec<f32> = x.iter().map(|v| v * 1.9).collect();
    let y = reshape(&hot, rate, &Shape { emphasis_db: 6.0, ..Default::default() }, &s).unwrap();
    let before = hot.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let after = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(after <= before.max(0.98) + 1e-6, "{before} → {after}");
}

#[test]
fn hostile_shapes_and_curves_are_refused() {
    let x = voice(16_000, 0.5, |_| 150.0, |_| 0.5);
    let s = PitchSettings::default();
    for bad in [
        Shape { range: 0.4, ..Default::default() },
        Shape { range: f64::NAN, ..Default::default() },
        Shape { range: 2.5, ..Default::default() },
        Shape { shift: 13.0, ..Default::default() },
        Shape { question_rise: f64::INFINITY, ..Default::default() },
        Shape { question_rise: 2.0, question_end: Some(-1.0), ..Default::default() },
        Shape { question_rise: 2.0, question_end: Some(f64::NAN), ..Default::default() },
        Shape { emphasis_db: 7.0, ..Default::default() },
        Shape { emphasis_db: -1.0, ..Default::default() },
    ] {
        assert!(matches!(reshape(&x, 16_000, &bad, &s), Err(ProsodyError::Settings(_))), "{bad:?}");
    }
    let t = track(&x, 16_000, &s).unwrap();
    assert_eq!(psola(&x, 16_000, &t, &[1.5; 3]), Err(ProsodyError::Mismatch { frames: t.len(), values: 3 }));
    assert!(psola(&x, 16_000, &PitchTrack { hop: 0.0, ..t.clone() }, &vec![1.5; t.len()]).is_err());
    assert!(psola(&x, 1, &t, &vec![1.5; t.len()]).is_err());
    // wild ratios are clamped, NaNs are 1, and the length is kept
    let mut wild = vec![50.0; t.len()];
    wild[3] = f64::NAN;
    wild[4] = -3.0;
    let y = psola(&x, 16_000, &t, &wild).unwrap();
    assert_eq!(y.len(), x.len());
    assert!(y.iter().all(|v| v.is_finite()));
    // a track that does not belong to the audio (longer than it) does no harm
    let long = PitchTrack { hop: HOP_S, f0: vec![Some(200.0); 500], confidence: vec![1.0; 500], level_db: vec![-10.0; 500] };
    let y = psola(&x, 16_000, &long, &vec![1.3; 500]).unwrap();
    assert_eq!(y.len(), x.len());
    // empty audio
    assert!(reshape(&[], 16_000, &Shape { range: 1.5, ..Default::default() }, &s).unwrap().is_empty());
    assert_eq!(expressiveness(&[], 16_000, &s).unwrap(), Expressiveness::default());
    assert!(syllables(&[], 16_000, &PitchTrack::default()).unwrap().is_empty());
    // silence: nothing to say
    let e = expressiveness(&vec![0.0; 16_000], 16_000, &s).unwrap();
    assert_eq!((e.syllables, e.f0_median_hz, e.voiced_fraction), (0, None, 0.0));
}

#[test]
fn helpers() {
    assert!((semitones(100.0, 200.0) - 12.0).abs() < 1e-12);
    assert_eq!(semitones(0.0, 200.0), 0.0);
    assert_eq!(semitones(f64::NAN, 200.0), 0.0);
    assert_eq!(quantile(&[], 0.5), None);
    assert_eq!(quantile(&[f64::NAN], 0.5), None);
    assert_eq!(quantile(&[3.0, 1.0, 2.0], 0.5), Some(2.0));
    assert_eq!(quantile(&[1.0, 2.0], 2.0), Some(2.0));
}

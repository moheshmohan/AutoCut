use super::audition::{self, Pool, Scored};
use super::signal::{self, Span};
use super::*;

const RATE: u32 = 24_000;

/// Seeded noise in −1…1.
fn noise(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32
}

/// Synthetic speech: phrases of syllables (a harmonic tone at `f0`, its level rising and falling
/// four times a second) with `pause` seconds between them, over a noise floor `floor` (linear).
fn speech(phrases: &[f64], pause: f64, f0: f32, floor: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut x = vec![0f32; (0.5 * RATE as f64) as usize];
    for p in phrases {
        let n = (p * RATE as f64) as usize;
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            let env = (std::f32::consts::PI * 4.0 * t).sin().abs();
            let tone: f32 = (1..12).map(|h| (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32).sum();
            x.push(0.25 * env * tone);
        }
        x.extend(std::iter::repeat_n(0.0, (pause * RATE as f64) as usize));
    }
    for v in x.iter_mut() {
        *v += floor * noise(&mut s);
    }
    x
}

#[test]
fn phrases_are_cut_at_pauses() {
    let x = speech(&[2.0, 1.5, 3.0], 0.6, 140.0, 0.001, 1);
    let f = signal::frames(&x, RATE);
    let a = signal::activity(&f);
    let p = signal::phrases(&f, &a);
    assert_eq!(p.len(), 3, "{p:?}");
    assert!((p[0].seconds() - 2.0).abs() < 0.3, "{p:?}");
    // a pause shorter than a quarter second joins two phrases
    let x = speech(&[2.0, 1.5], 0.12, 140.0, 0.001, 1);
    let f = signal::frames(&x, RATE);
    assert_eq!(signal::phrases(&f, &signal::activity(&f)).len(), 1);
}

#[test]
fn quality_tells_clean_from_noisy_clipped_reverberant_and_music() {
    let measure = |x: &[f32]| {
        let f = signal::frames(x, RATE);
        let a = signal::activity(&f);
        let span = Span::new(0.0, x.len() as f64 / RATE as f64);
        signal::quality(x, &f, &a, span, span)
    };
    let clean = speech(&[2.0, 1.5, 2.5, 2.0], 0.5, 140.0, 0.0005, 2);
    let q = measure(&clean);
    assert!(q.snr_db > 35.0, "{q:?}");
    assert!(q.clipped == 0.0 && q.music < 0.2, "{q:?}");
    assert!(q.bandwidth_hz > 1500.0, "{q:?}");
    assert!(q.score > 0.5, "{q:?}");
    let noisy = speech(&[2.0, 1.5, 2.5, 2.0], 0.5, 140.0, 0.05, 2);
    let qn = measure(&noisy);
    assert!(qn.snr_db < q.snr_db - 15.0 && qn.score < q.score, "{qn:?} vs {q:?}");
    // clipped: the same speech driven into full scale
    let hot: Vec<f32> = clean.iter().map(|v| (v * 8.0).clamp(-1.0, 1.0)).collect();
    let qc = measure(&hot);
    assert!(qc.clipped > 0.01 && qc.score < q.score, "{qc:?}");
    // reverberant: an exponential tail (0.8 s to −60 dB) of noise convolved in, coarsely; long
    // pauses, so the tails end in them
    let dry = speech(&[1.5, 1.5, 1.5], 1.2, 140.0, 0.0005, 2);
    let mut s = 9u64;
    let tail: Vec<f32> = (0..(0.8 * RATE as f64) as usize).map(|i| noise(&mut s) * (-6.9 * i as f32 / (0.8 * RATE as f32)).exp() * 0.02).collect();
    let mut wet = dry.clone();
    for (i, v) in dry.iter().enumerate().step_by(4) {
        if v.abs() < 1e-4 {
            continue;
        }
        for (j, t) in tail.iter().enumerate().step_by(4) {
            if let Some(w) = wet.get_mut(i + j) {
                *w += v * t * 4.0;
            }
        }
    }
    let (qd, qr) = (measure(&dry), measure(&wet));
    assert!(qr.reverb_s.unwrap_or(0.0) > 0.3 && qr.reverb_s.unwrap_or(0.0) > qd.reverb_s.unwrap_or(0.0) + 0.1, "{qr:?} vs {qd:?}");
    // music: a steady chord under the speech, loud in the pauses
    let music: Vec<f32> = clean
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let t = i as f32 / RATE as f32;
            v + 0.03 * ((std::f32::consts::TAU * 440.0 * t).sin() + (std::f32::consts::TAU * 554.4 * t).sin() + (std::f32::consts::TAU * 659.3 * t).sin())
        })
        .collect();
    let qm = measure(&music);
    assert!(qm.music > 0.5, "{qm:?}");
    assert!(qm.score < q.score, "{qm:?}");
}

#[test]
fn hostile_audio_never_panics() {
    for x in [vec![], vec![f32::NAN; 5000], vec![f32::INFINITY; 30_000], vec![1.0; 100_000], vec![0.0; 3]] {
        let f = signal::frames(&x, RATE);
        let a = signal::activity(&f);
        let p = signal::phrases(&f, &a);
        let span = Span::new(-5.0, 1e9);
        let q = signal::quality(&x, &f, &a, span, span.widened(3.0));
        assert!(q.score.is_finite() && (0.0..=1.0).contains(&q.score), "{q:?}");
        let _ = signal::chunks(&p, (6.0, 12.0), 1.0, 2.5);
        let _ = signal::clean(&x, RATE, 8.0);
        let _ = signal::balance(&x, RATE);
        let mut y = x.clone();
        signal::normalise(&mut y, RATE, -20.0);
        assert!(y.iter().all(|v| v.is_finite()));
    }
    let _ = signal::clean(&[0.1; 100], 0, 8.0);
    assert_eq!(signal::cut(&[1.0; 10], RATE, Span::new(f64::NAN, f64::INFINITY)).len(), 10);
}

#[test]
fn windows_fit_the_engines_and_never_cut_a_word() {
    // phrases of 2 s with 0.5 s pauses
    let phrases: Vec<Span> = (0..20).map(|i| Span::new(i as f64 * 2.5, i as f64 * 2.5 + 2.0)).collect();
    for t in Target::ALL {
        let b = t.bounds();
        let w = windows(&phrases, &b, 10_000);
        assert!(!w.is_empty());
        for (s, _, _) in &w {
            assert!(s.seconds() >= b.min && s.seconds() <= b.max, "{t:?} {s:?}");
            // edges in pauses
            assert!(!phrases.iter().any(|p| p.start < s.start - 1e-9 && s.start < p.end), "{s:?}");
            assert!(!phrases.iter().any(|p| p.start < s.end && s.end < p.end - 1e-9), "{s:?}");
        }
    }
    // a long pause ends a run
    let gapped = [Span::new(0.0, 5.0), Span::new(8.0, 14.0)];
    assert!(windows(&gapped, &Target::Voxcpm.bounds(), 100).iter().all(|(s, _, _)| s.seconds() < 8.0));
    assert!(length_fit(20.0, &Target::Voxcpm.bounds()) == 1.0 && length_fit(31.0, &Target::Voxcpm.bounds()) == 0.0);
    assert!(length_fit(f64::NAN, &Target::Indicf5.bounds()) == 0.0);
    assert!(length_fit(7.0, &Target::Indicf5.bounds()) < 1.0 && length_fit(7.0, &Target::Indicf5.bounds()) > 0.5);
}

#[test]
fn sentences_are_whole_by_the_words_or_the_pauses() {
    let w = |t: &str, a: f64, b: f64| Word { text: t.into(), start: a, end: b };
    let words =
        vec![w("Hello", 0.0, 0.5), w("there.", 0.6, 1.0), w("This", 1.6, 1.9), w("is", 2.0, 2.2), w("it.", 2.3, 2.6), w("And", 3.0, 3.2), w("more", 3.3, 3.6)];
    // "This is it." exactly
    assert_eq!(sentence_fit(&words, Span::new(1.4, 2.8), 0.6, 0.4), 1.0);
    // ends mid-sentence ("And")
    assert_eq!(sentence_fit(&words, Span::new(1.4, 3.25), 0.6, 0.05), 0.5);
    // cuts a word
    assert_eq!(sentence_fit(&words, Span::new(1.4, 3.45), 0.6, 0.0), 0.5);
    // no words: long pauses still count
    assert_eq!(sentence_fit(&[], Span::new(0.0, 2.0), 0.8, 0.9), 0.7);
    assert_eq!(sentence_fit(&[], Span::new(0.0, 2.0), 0.3, 0.3), 0.0);
}

/// Fake models: the "embedding" is the signal's pitch (two speakers apart), the words one per
/// phrase-second, a sentence end every third word.
struct Fake {
    heard: usize,
}

impl Models for Fake {
    fn colour(&mut self, x: &[f32], rate: u32) -> Result<Vec<f32>, crate::TtsError> {
        let p = crate::clone::median_pitch(x, rate).unwrap_or(0.0) as f32;
        Ok(vec![1.0, p / 100.0, (p / 60.0).sin()])
    }
    fn transcribe(&mut self, x: &[f32], rate: u32) -> Result<Heard, crate::TtsError> {
        self.heard += 1;
        let secs = x.len() as f64 / f64::from(rate);
        let words = (0..secs as usize)
            .map(|i| Word { text: if i % 3 == 2 { "word.".into() } else { "word".into() }, start: i as f64 + 0.1, end: i as f64 + 0.6 })
            .collect();
        Ok(Heard { language: "en".into(), words })
    }
}

#[test]
fn the_scan_finds_the_speaker_only_and_ranks_within_bounds() {
    // the speaker (140 Hz) for 40 s, then someone else (230 Hz) for 40 s
    let mut x = speech(&[3.0; 12], 0.4, 140.0, 0.001, 3);
    x.extend(speech(&[3.0; 12], 0.4, 230.0, 0.001, 4));
    let mut fake = Fake { heard: 0 };
    let me = speech(&[3.0; 4], 0.4, 140.0, 0.001, 5);
    let speaker = fake.colour(&me, RATE).unwrap();
    let mut lines = Vec::new();
    let sc = scan("a.mov", &x, RATE, &speaker, &FindSettings { min_similarity: 0.99, per_clip: 2 }, &mut fake, &mut |l| lines.push(l.to_string())).unwrap();
    assert!(!sc.chunks.is_empty());
    let split = 12.0 * 3.4 + 0.5;
    assert!(sc.chunks.iter().filter(|c| c.span.start > split + 1.0).all(|c| c.similarity < 0.99), "{:?}", sc.chunks);
    assert!(sc.speaker_seconds > 25.0 && sc.speaker_seconds < split + 2.0, "{}", sc.speaker_seconds);
    assert!(fake.heard > 0 && sc.language.as_deref() == Some("en"));
    assert!(!sc.candidates.is_empty());
    for c in &sc.candidates {
        assert!(c.span.end <= split + 1.0, "{c:?}");
        let b = c.target.bounds();
        assert!(c.span.seconds() >= b.min && c.span.seconds() <= b.max, "{c:?}");
        assert!(c.score > 0.0 && c.score <= 1.0);
    }
    for t in Target::ALL {
        let of: Vec<&Candidate> = sc.candidates.iter().filter(|c| c.target == t).collect();
        assert!(of.len() <= 2);
        assert!(of.windows(2).all(|w| !w[0].span.overlaps(&w[1].span)));
    }
    // nobody like the speaker: nothing heard, no candidates
    let mut fake2 = Fake { heard: 0 };
    let none = scan("b.mov", &x, RATE, &[-1.0, 0.0, 0.0], &FindSettings::default(), &mut fake2, &mut |_| {}).unwrap();
    assert!(none.candidates.is_empty() && fake2.heard == 0);
    assert!(scan("c", &x, 0, &speaker, &FindSettings::default(), &mut fake2, &mut |_| {}).is_err());
}

fn cand(clip: &str, t: Target, start: f64, len: f64, score: f64) -> Candidate {
    Candidate {
        id: String::new(),
        clip: clip.into(),
        span: Span::new(start, start + len),
        target: t,
        language: "en".into(),
        similarity: 0.9,
        quality: Quality::default(),
        sentence: 1.0,
        length_fit: 1.0,
        language_fit: 1.0,
        score,
        words: String::new(),
    }
}

#[test]
fn ranking_spreads_over_clips_and_numbers_by_engine() {
    let all = vec![
        cand("a", Target::Voxcpm, 0.0, 20.0, 0.9),
        cand("a", Target::Voxcpm, 30.0, 20.0, 0.85),
        cand("a", Target::Voxcpm, 60.0, 20.0, 0.84),
        cand("b", Target::Voxcpm, 0.0, 20.0, 0.5),
        cand("a", Target::Indicf5, 0.0, 10.0, 0.7),
        cand("a", Target::Indicf5, 5.0, 10.0, 0.8),
    ];
    let r = rank(&all, 3, 2);
    let v: Vec<(&str, &str)> = r.iter().map(|c| (c.id.as_str(), c.clip.as_str())).collect();
    assert_eq!(v, [("voxcpm-1", "a"), ("voxcpm-2", "a"), ("voxcpm-3", "b"), ("indicf5-1", "a")]);
}

#[test]
fn held_out_speech_avoids_the_references_and_takes_turns() {
    let chunk = |a: f64, s: f64| Checked { span: Span::new(a, a + 10.0), similarity: s };
    let scans = vec![
        ClipScan { clip: "a".into(), chunks: vec![chunk(0.0, 0.95), chunk(10.0, 0.96), chunk(20.0, 0.5), chunk(30.0, 0.9)], ..Default::default() },
        ClipScan { clip: "b".into(), chunks: vec![chunk(0.0, 0.91)], ..Default::default() },
    ];
    let h = held_out(&scans, &[("a".into(), Span::new(12.0, 18.0))], 0.85, 25.0);
    // a's 10–20 overlaps the reference, 20–30 is someone else; a and b in turn
    assert_eq!(h, vec![("a".to_string(), Span::new(0.0, 10.0)), ("b".to_string(), Span::new(0.0, 10.0)), ("a".to_string(), Span::new(30.0, 40.0))]);
    assert!(held_out(&[], &[], 0.85, 100.0).is_empty());
}

#[test]
fn auditions_rank_similar_clear_readings_first() {
    let pool = Pool { colour: vec![1.0, 0.0], pitch_hz: Some(120.0), seconds: 60.0, from: vec![] };
    let x = speech(&[1.5], 0.2, 120.0, 0.0005, 7);
    let line = audition::ENGLISH[0];
    let good = audition::take(0, line, line, &x, RATE, &[1.0, 0.1], &pool);
    assert!(good.cer == 0.0 && good.start_cer == 0.0 && good.similarity > 0.99, "{good:?}");
    assert!(good.pitch_hz.is_some_and(|p| (p - 120.0).abs() < 10.0), "{good:?}");
    let garbled = audition::take(0, line, "and so honest Lee I did not", &x, RATE, &[1.0, 0.1], &pool);
    assert!(garbled.start_cer > 0.3 && garbled.cer > 0.3, "{garbled:?}");
    let far = audition::take(0, line, line, &x, RATE, &[0.5, 1.0], &pool);
    let mk = |id: &str, t: Take| audition::summarise(Scored { id: id.into(), target: Some(Target::Voxcpm), ..Default::default() }, vec![t], &pool);
    use audition::Take;
    let mut v = vec![
        mk("garbled", garbled),
        mk("far", far),
        mk("good", good),
        audition::summarise(Scored { id: "none".into(), target: Some(Target::Voxcpm), ..Default::default() }, vec![], &pool),
        mk("ml", audition::take(0, "x", "x", &x, RATE, &[1.0, 0.0], &pool)),
    ];
    v[4].target = Some(Target::Indicf5);
    audition::rank(&mut v);
    let order: Vec<(&str, usize)> = v.iter().map(|s| (s.id.as_str(), s.rank)).collect();
    assert_eq!(order, [("ml", 1), ("good", 1), ("far", 2), ("garbled", 3), ("none", 4)]);
    assert!(audition::score(f64::NAN, 0.0, 0.0, 0, None) == f64::NEG_INFINITY);
    for t in Target::ALL {
        assert!(audition::lines(t).len() >= 3);
        assert_eq!(Target::from_id(t.id()), Some(t));
    }
    assert!(crate::malayalam::detect(audition::MALAYALAM[0]));
}

#[test]
fn clean_up_and_joining_keep_the_speech() {
    let x = speech(&[2.0, 2.0], 0.5, 140.0, 0.01, 8);
    let y = signal::clean(&x, RATE, 8.0);
    assert!(y.len() > RATE as usize * 4 && y.len() <= x.len(), "{}", y.len());
    assert!(y.iter().all(|v| v.is_finite()));
    let j = signal::join(&[&x, &y], RATE, 0.3);
    assert_eq!(j.len(), x.len() + y.len() + (0.3 * RATE as f64) as usize);
    let b = signal::balance(&x, RATE);
    assert!(b.sub >= 0.0 && b.presence >= 0.0 && b.tilt_db.is_finite() && b.consonant_db < 0.0, "{b:?}");
    // a lift at 3 kHz raises the consonant band
    let bright: Vec<f32> = x.iter().enumerate().map(|(i, v)| v + 0.05 * (std::f32::consts::TAU * 3000.0 * i as f32 / RATE as f32).sin()).collect();
    assert!(signal::balance(&bright, RATE).consonant_db > b.consonant_db + 3.0);
    assert_eq!(normalise_language("ta"), "ml");
    assert!(cosine(&[1.0], &[1.0, 2.0]) == 0.0 && cosine(&[0.0, 0.0], &[1.0, 0.0]) == 0.0);
}

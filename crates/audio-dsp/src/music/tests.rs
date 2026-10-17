//! Tests of the soundtrack engine: exact lengths, endings on a bar, stems that sum to the mix,
//! determinism, energy that follows the curve, a beat the analysis finds, hostile requests.

use super::*;
use crate::loudness::LoudnessMeter;

const SR: u32 = 48_000;

fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn req(mood: Mood, seconds: f64) -> Request {
    Request::new(mood, secs(seconds))
}

#[test]
fn plans_hit_the_exact_length_and_end_on_a_bar() {
    let mut checked = 0;
    for mood in Mood::ALL {
        for genre in [None, Some(Genre::Cinematic), Some(Genre::Electronic), Some(Genre::Lofi)] {
            for (i, seconds) in
                [0.05, 0.5, 1.19, 1.3, 2.0, 3.7, 5.0, 7.77, 12.0, 15.5, 29.97, 30.0, 33.333, 47.1, 60.0, 61.25, 90.0, 123.456, 300.0, 899.9, 900.0]
                    .into_iter()
                    .enumerate()
            {
                for (seed, bpm) in [(1u64, None), (7, Some(97.3)), (99, Some(40.0)), (12345, Some(200.0)), (3, Some(128.0))] {
                    let mut r = req(mood, seconds);
                    r.genre = genre;
                    r.seed = seed.wrapping_add(i as u64);
                    r.bpm = bpm;
                    let p = plan(&r, SR).unwrap();
                    // to the sample: round(length · rate)
                    let exact = (i128::from(r.length.0) * i128::from(SR) * 2 + i128::from(TICKS_PER_SECOND)) / (2 * i128::from(TICKS_PER_SECOND));
                    assert_eq!(i128::from(p.samples), exact, "{seconds} s");
                    assert_eq!(p.length, r.length);
                    // the hit is a downbeat, the ring-out between its minimum and a bar longer
                    let bars = u64::from(p.arrangement.bars);
                    assert_eq!(p.ending, p.tempo.bar(bars));
                    assert_eq!(p.ending.0 + p.tail.0, p.length.0);
                    let least = tail_min(p.tempo);
                    if p.length >= least {
                        assert!(p.tail >= least, "{seconds} s at {} BPM: tail {:?}", p.tempo.bpm(), p.tail);
                        assert!(p.tail.0 < least.0 + p.tempo.bar(1).0, "{seconds} s: a whole bar more would have fitted");
                    } else {
                        assert_eq!(bars, 0);
                    }
                    // the blocks fill the bars exactly; the beat map runs from 0 to the hit
                    assert_eq!(p.arrangement.blocks.iter().map(|b| b.bars).sum::<u32>(), p.arrangement.bars);
                    let beats = p.beats();
                    assert_eq!(beats.first(), Some(&Tick::ZERO));
                    assert_eq!(beats.last(), Some(&p.ending));
                    assert_eq!(beats.len() as u64, bars * 4 + 1);
                    assert!(beats.windows(2).all(|w| w[1] > w[0]));
                    assert!(p.bars().iter().all(|b| beats.contains(b)));
                    assert_eq!(p.chords.len() as u64, bars + 1);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 3000);
}

#[test]
fn rendered_tracks_are_exactly_as_long_as_asked() {
    for (seconds, seed, mood) in [(0.05, 1, Mood::Epic), (1.0, 2, Mood::Calm), (4.321, 3, Mood::Energetic), (9.99, 4, Mood::Dreamy), (16.0, 5, Mood::Sad)] {
        let r = Request { seed, ..req(mood, seconds) };
        let t = generate(&r, SR).unwrap();
        assert_eq!(t.mix.left.len() as u64, t.plan.samples, "{seconds}");
        assert_eq!(t.mix.right.len(), t.mix.left.len());
        assert!(t.mix.left.iter().chain(&t.mix.right).all(|x| x.is_finite() && x.abs() <= 1.0));
    }
    // another rate: the same exact rounding
    let t = generate(&req(Mood::Happy, 3.0), 44_100).unwrap();
    assert_eq!(t.mix.left.len(), 132_300);
}

#[test]
fn the_track_ends_on_the_final_hit_and_rings_out() {
    let t = generate(&Request { seed: 4, ..req(Mood::Epic, 20.0) }, SR).unwrap();
    let hit = tempo::sample_of(t.plan.ending, SR) as usize;
    let n = t.mix.left.len();
    let hw = SR as usize / 10;
    let before = rms(&t.mix.left[hit - hw..hit]);
    let at = rms(&t.mix.left[hit..hit + hw]);
    let last = rms(&t.mix.left[n - hw..]);
    assert!(at > before * 0.8, "the hit lands: {at} after, {before} before");
    assert!(last < at * 0.25, "it rings out: {last} at the end, {at} at the hit");
    assert!(t.mix.left[n - 1].abs() < 1e-3 && t.mix.right[n - 1].abs() < 1e-3, "silent at the last sample");
    // the hit is on a downbeat of the grid from 0
    assert_eq!(t.plan.ending, t.plan.tempo.bar(u64::from(t.plan.arrangement.bars)));
    assert!(t.plan.tail.seconds() >= 1.2);
}

#[test]
fn stems_sum_to_the_mix() {
    for (mood, genre) in [(Mood::Epic, None), (Mood::Energetic, Some(Genre::Electronic)), (Mood::Dreamy, Some(Genre::Lofi)), (Mood::Calm, Some(Genre::Piano))] {
        let r = Request { stems: true, genre, seed: 9, ..req(mood, 14.0) };
        let t = generate(&r, SR).unwrap();
        assert_eq!(t.stems.len(), 4);
        assert_eq!(t.stems.iter().map(|(s, _)| *s).collect::<Vec<_>>(), Stem::ALL.to_vec());
        let mut worst = 0.0f32;
        for i in 0..t.mix.left.len() {
            let l: f32 = t.stems.iter().map(|(_, s)| s.left[i]).sum();
            let r: f32 = t.stems.iter().map(|(_, s)| s.right[i]).sum();
            worst = worst.max((l - t.mix.left[i]).abs()).max((r - t.mix.right[i]).abs());
        }
        assert!(worst < 1e-6, "{mood:?}: stems differ from the mix by {worst}");
        for (s, a) in &t.stems {
            assert_eq!(a.left.len(), t.mix.left.len());
            // a calm piano piece has no drums, and the piano plays its own bass
            if !(matches!(s, Stem::Bass | Stem::Drums) && genre == Some(Genre::Piano)) {
                assert!(rms(&a.left) > 1e-4, "{mood:?} {s:?} is not silent");
            }
        }
        // the mix alone is the same music (to rounding)
        let alone = generate(&Request { stems: false, ..r.clone() }, SR).unwrap();
        let diff = alone.mix.left.iter().zip(&t.mix.left).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "{mood:?}: {diff}");
    }
}

#[test]
fn stems_can_be_muted_and_levelled() {
    let r = Request { stems: true, seed: 2, ..req(Mood::Uplifting, 10.0) };
    let mut quiet = r.clone();
    quiet.levels[Stem::Drums.index()] = None;
    let t = generate(&quiet, SR).unwrap();
    let drums = &t.stems[Stem::Drums.index()].1;
    assert!(drums.left.iter().all(|x| *x == 0.0), "muted");
    assert!(rms(&t.mix.left) > 1e-3);
    // every stem muted is refused, as are nonsense levels
    let mut none = r.clone();
    none.levels = [None; 4];
    assert!(generate(&none, SR).is_err());
    for bad in [f64::NAN, f64::INFINITY, 40.0] {
        let mut b = r.clone();
        b.levels[0] = Some(bad);
        assert!(generate(&b, SR).is_err(), "{bad}");
    }
    let mut low = r.clone();
    low.levels[0] = Some(-200.0);
    assert!(generate(&low, SR).is_ok(), "below -60 dB is silence");
}

#[test]
fn the_same_request_gives_the_same_audio() {
    let r = Request { seed: 42, genre: Some(Genre::Pop), ..req(Mood::Happy, 8.0) };
    let a = generate(&r, SR).unwrap();
    let b = generate(&r, SR).unwrap();
    assert_eq!(a.mix, b.mix);
    assert_eq!(a.plan, b.plan);
    let c = generate(&Request { seed: 43, ..r.clone() }, SR).unwrap();
    assert_ne!(a.mix.left, c.mix.left, "another seed, another variation");
}

#[test]
fn energy_follows_the_requested_curve() {
    let loud = |e: Vec<f64>| {
        let t = generate(&Request { energy: Some(Energy::Curve(e)), seed: 3, ..req(Mood::Uplifting, 40.0) }, SR).unwrap();
        let hit = tempo::sample_of(t.plan.ending, SR) as usize;
        let body = &t.mix.left[..hit];
        let q = body.len() / 4;
        (rms(&body[..q]), rms(&body[3 * q..]))
    };
    let (early, late) = loud(vec![0.1, 0.95]);
    assert!(late > early * 1.5, "rising: {early} → {late}");
    let (early, late) = loud(vec![0.95, 0.1]);
    assert!(early > late * 1.5, "falling: {early} → {late}");
}

#[test]
fn beds_sit_at_the_bed_loudness() {
    for mood in [Mood::Epic, Mood::Calm, Mood::Energetic] {
        let t = generate(&Request { seed: 5, ..req(mood, 30.0) }, SR).unwrap();
        let mut m = LoudnessMeter::new(f64::from(SR), 2);
        m.process(&[&t.mix.left, &t.mix.right]);
        assert!((m.integrated() - crate::synth::MUSIC_LUFS).abs() < 1.0, "{mood:?}: {}", m.integrated());
    }
}

#[test]
fn every_mood_in_every_genre_renders() {
    for mood in Mood::ALL {
        for genre in Genre::ALL {
            let t = generate(&Request { genre: Some(genre), seed: 11, ..req(mood, 6.0) }, SR).unwrap();
            assert!(t.mix.left.iter().chain(&t.mix.right).all(|x| x.is_finite() && x.abs() <= 1.0), "{mood:?} {genre:?}");
            assert!(rms(&t.mix.left) > 1e-3, "{mood:?} {genre:?} is not silent");
            assert_eq!((t.plan.mood, t.plan.genre), (mood, genre));
        }
    }
}

#[test]
fn the_beat_analysis_finds_the_beat_it_was_made_on() {
    for (bpm, mood, genre, seconds, seed) in [
        (128.0, Mood::Epic, Genre::Cinematic, 20.0, 1),
        (128.0, Mood::Epic, Genre::Cinematic, 12.0, 11),
        (100.0, Mood::Happy, Genre::Pop, 20.0, 1),
        (120.0, Mood::Energetic, Genre::Electronic, 20.0, 1),
        (110.0, Mood::Uplifting, Genre::Pop, 14.0, 5),
    ] {
        let t = generate(&Request { bpm: Some(bpm), genre: Some(genre), seed, ..req(mood, seconds) }, SR).unwrap();
        let a = crate::remix::analyze(&[&t.mix.left, &t.mix.right], SR).unwrap();
        assert!((a.bpm - bpm).abs() < 2.0, "{genre:?}: made at {bpm}, found {}", a.bpm);
        // on the beat, not between: the beats found sit on the grid the track was written on
        let lead = crate::remix::beat_lead(SR);
        let period = 60.0 / bpm;
        let on = a
            .beats
            .iter()
            .filter(|b| {
                let x = (**b + lead) as f64 / f64::from(SR) / period;
                (x - x.round()).abs() * period < 0.035
            })
            .count();
        assert!(on * 10 >= a.beats.len() * 9, "{genre:?} {seconds} s: {on} of {} beats on the grid", a.beats.len());
    }
}

#[test]
fn hostile_requests_are_refused_or_clamped_never_a_panic() {
    let ok = req(Mood::Tense, 4.0);
    for bad in [
        Request { length: Tick(-1), ..ok.clone() },
        Request { length: Tick::ZERO, ..ok.clone() },
        Request { length: secs(0.01), ..ok.clone() },
        Request { length: secs(901.0), ..ok.clone() },
        Request { length: Tick::MAX, ..ok.clone() },
        Request { length: Tick::MIN, ..ok.clone() },
        Request { stems: true, length: secs(301.0), ..ok.clone() },
        Request { energy: Some(Energy::Curve(vec![])), ..ok.clone() },
        Request { energy: Some(Energy::Curve(vec![0.5, f64::NAN])), ..ok.clone() },
        Request { energy: Some(Energy::Curve(vec![0.5; 65])), ..ok.clone() },
        Request { energy: Some(Energy::Curve(vec![-0.1, 0.5])), ..ok.clone() },
    ] {
        assert!(generate(&bad, SR).is_err(), "{bad:?}");
    }
    assert!(generate(&ok, 0).is_err());
    assert!(generate(&ok, 7_999).is_err());
    assert!(generate(&ok, 1_000_000).is_err());
    // clamped, not refused
    for fine in [
        Request { bpm: Some(f64::NAN), ..ok.clone() },
        Request { bpm: Some(-3.0), ..ok.clone() },
        Request { bpm: Some(1e300), ..ok.clone() },
        Request { key: Some(i32::MIN), ..ok.clone() },
        Request { key: Some(i32::MAX), ..ok.clone() },
        Request { energy: Some(Energy::Level(f64::NAN)), ..ok.clone() },
        Request { energy: Some(Energy::Level(-4.0)), ..ok.clone() },
        Request { seed: u64::MAX, ..ok.clone() },
    ] {
        let t = generate(&fine, SR).unwrap();
        assert!(t.mix.left.iter().all(|x| x.is_finite() && x.abs() <= 1.0), "{fine:?}");
    }
    assert!(generate(&Request { stems: true, length: secs(300.0), ..ok.clone() }, 8_000).is_ok(), "five minutes of stems at a low rate");
}

#[test]
fn sections_and_keys_are_described() {
    let p = plan(&Request { key: Some(3), ..req(Mood::Uplifting, 60.0) }, SR).unwrap();
    assert_eq!(p.key_name(), "C major");
    assert_eq!(plan(&req(Mood::Epic, 60.0), SR).unwrap().key_name(), "A minor");
    let s = p.sections();
    assert_eq!(s.first().map(|x| x.start), Some(Tick::ZERO));
    assert!(s.windows(2).all(|w| w[1].start > w[0].start));
    assert_eq!(s.iter().map(|x| x.bars).sum::<u32>(), p.arrangement.bars);
    assert!(s.iter().all(|x| (0.0..=1.0).contains(&x.energy)));
    // the style's tempo when none is asked for
    assert_eq!(p.tempo.bpm(), Style::of(Mood::Uplifting, None).bpm.1);
}

/// Writes a piano roll and a spectrogram (PPM) of a few tracks into `$MUSIC_DEBUG_DIR`, to look
/// at: `MUSIC_DEBUG_DIR=/tmp/x cargo test -p filmcraft-audio-dsp --release music_pictures -- --ignored`.
#[test]
#[ignore]
fn music_pictures() {
    let Ok(dir) = std::env::var("MUSIC_DEBUG_DIR") else { return };
    for (name, mood, genre, seconds) in
        [("epic", Mood::Epic, Genre::Cinematic, 45.0), ("lofi", Mood::Dreamy, Genre::Lofi, 30.0), ("edm", Mood::Energetic, Genre::Electronic, 30.0)]
    {
        let r = Request { genre: Some(genre), seed: 3, ..req(mood, seconds) };
        let p = plan(&r, SR).unwrap();
        let score = compose::compose(&p);
        // piano roll: 4 px a sixteenth, 3 px a semitone (MIDI 24..108), one colour per stem
        let steps = (p.length.seconds() / (p.tempo.beat_seconds() / 4.0)).ceil() as usize;
        let (w, h) = (steps * 4 + 1, 85 * 3);
        let mut img = vec![[16u8, 16, 20]; w * h];
        for bar in 0..=p.arrangement.bars as usize {
            for y in 0..h {
                if let Some(px) = img.get_mut(y * w + bar * 64) {
                    *px = [60, 60, 70];
                }
            }
        }
        let spp = f64::from(SR) * p.tempo.beat_seconds() / 4.0 / 4.0;
        for part in &score.parts {
            let col = match part.stem {
                Stem::Drums => [200, 200, 200],
                Stem::Bass => [80, 160, 255],
                Stem::Melody => [255, 180, 60],
                Stem::Pad => [120, 220, 120],
            };
            for n in &part.notes {
                let key = if part.stem == Stem::Drums { 24 + (n.key as usize % 12) } else { n.key as usize };
                if !(24..109).contains(&key) {
                    continue;
                }
                let y0 = (108 - key) * 3;
                let x0 = (n.start as f64 / spp) as usize;
                let x1 = ((n.start + n.len) as f64 / spp) as usize;
                for x in x0..=x1.max(x0 + 1).min(w - 1) {
                    for y in y0..y0 + 2 {
                        if let Some(px) = img.get_mut(y * w + x) {
                            *px = col;
                        }
                    }
                }
            }
        }
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.extend(img.iter().flatten());
        std::fs::write(format!("{dir}/{name}-roll.ppm"), ppm).unwrap();
        // spectrogram: 1024-point frames every 20 ms, log frequency 40 Hz .. 16 kHz
        let t = generate(&r, SR).unwrap();
        let hop = SR as usize / 50;
        let frames = t.mix.left.len() / hop;
        let fft = crate::fft::Fft::new(1024);
        let rows = 240;
        let mut spec = vec![[0u8; 3]; frames * rows];
        for f in 0..frames {
            let mut re = vec![0.0f32; 1024];
            let mut im = vec![0.0f32; 1024];
            for i in 0..1024 {
                let x = t.mix.left.get(f * hop + i).copied().unwrap_or(0.0) + t.mix.right.get(f * hop + i).copied().unwrap_or(0.0);
                re[i] = x * (0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / 1024.0).cos());
            }
            fft.forward(&mut re, &mut im);
            for row in 0..rows {
                let hz = 40.0 * (16_000.0f64 / 40.0).powf(row as f64 / rows as f64);
                let bin = ((hz / f64::from(SR) * 1024.0) as usize).clamp(1, 511);
                let mag = ((re[bin] * re[bin] + im[bin] * im[bin]).sqrt() / 256.0).max(1e-9);
                let db = 20.0 * mag.log10();
                let v = ((db + 60.0) / 70.0).clamp(0.0, 1.0);
                spec[(rows - 1 - row) * frames + f] = [(255.0 * v) as u8, (255.0 * v * v) as u8, (255.0 * (1.0 - v) * v * 2.0).min(255.0) as u8];
            }
        }
        let mut ppm = format!("P6\n{frames} {rows}\n255\n").into_bytes();
        ppm.extend(spec.iter().flatten());
        std::fs::write(format!("{dir}/{name}-spec.ppm"), ppm).unwrap();
        let kinds: Vec<String> = p.sections().iter().map(|x| format!("{}{}", x.kind.id(), x.bars)).collect();
        let chords: Vec<String> = p.chords.iter().map(|c| c.name(p.key)).collect();
        println!("{name}: {} BPM, {} bars + {:.2} s ring, [{}]\n  {}", p.tempo.bpm(), p.arrangement.bars, p.tail.seconds(), kinds.join(" "), chords.join(" "));
    }
}

/// Prints how fast tracks render: `cargo test -p filmcraft-audio-dsp --release render_speed -- --ignored --nocapture`.
#[test]
#[ignore]
fn render_speed() {
    for (mood, genre, seconds, stems) in
        [(Mood::Epic, Genre::Cinematic, 120.0, false), (Mood::Energetic, Genre::Electronic, 120.0, true), (Mood::Calm, Genre::Piano, 120.0, false)]
    {
        let t0 = std::time::Instant::now();
        let t = generate(&Request { genre: Some(genre), stems, ..req(mood, seconds) }, SR).unwrap();
        let took = t0.elapsed().as_secs_f64();
        println!("{mood:?} {genre:?} {seconds} s (stems {stems}): {took:.2} s, {:.0}× real time", t.mix.seconds() / took);
        for only in Stem::ALL {
            let mut levels = [None; 4];
            levels[only.index()] = Some(0.0);
            let t0 = std::time::Instant::now();
            let _ = generate(&Request { genre: Some(genre), levels, ..req(mood, seconds) }, SR).unwrap();
            println!("    only {only:?}: {:.2} s", t0.elapsed().as_secs_f64());
        }
    }
}

/// Prints each style's stem loudness (calibration aid): `cargo test -p filmcraft-audio-dsp --release stem_balance -- --ignored --nocapture`.
#[test]
#[ignore]
fn stem_balance() {
    for s in taxonomy::STYLES {
        let t = generate(&Request { stems: true, genre: Some(s.genre), seed: 1, ..req(s.mood, 40.0) }, SR).unwrap();
        let lufs: Vec<String> = t
            .stems
            .iter()
            .map(|(stem, a)| {
                let mut m = LoudnessMeter::new(f64::from(SR), 2);
                m.process(&[&a.left, &a.right]);
                format!("{} {:6.1}", stem.id(), m.integrated())
            })
            .collect();
        let kinds: Vec<String> = t.plan.sections().iter().map(|x| format!("{}{}", x.kind.id(), x.bars)).collect();
        println!("{:10} {:10} {}   [{}]", s.mood.id(), s.genre.id(), lufs.join("  "), kinds.join(" "));
    }
}

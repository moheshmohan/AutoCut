//! The sound of Time Remapping clips (`remapped`): it follows the speed curve, keeps its pitch
//! with Maintain Audio Pitch (varispeed without), and any chunking renders the same samples.

use super::*;
use filmcraft_project::time_remap::{ramp_keyframes, ramp_mean, ramp_preset, remap_effect};
use filmcraft_project::{ClipId, ItemId};

const SR: u32 = 48_000;
const HZ: f64 = 440.0;

fn sine(m0: i64, len: usize) -> Option<AudioBuffer> {
    let ch = (0..len).map(|i| ((m0 + i as i64) as f64 * std::f64::consts::TAU * HZ / SR as f64).sin() as f32).collect();
    Some(AudioBuffer { sample_rate: SR, channels: vec![ch] })
}

/// A clip at 1 s playing 4 s of media with `preset`.
fn clip(preset: &str, keep_pitch: bool) -> TrackItem {
    let span = Tick::from_seconds_f64(4.0);
    let src_in = Tick::from_seconds_f64(2.0);
    let keys = ramp_keyframes(ramp_preset(preset).unwrap(), 1.0, src_in, span, 100.0);
    let mean = ramp_mean(&keys, src_in, span, 100.0).unwrap();
    let duration = Tick((span.0 as f64 / mean).round() as i64);
    let mut effects = filmcraft_project::effect::intrinsic_video();
    if let Some(e) = effects.iter_mut().find(|e| e.effect == "time_remap") {
        *e = remap_effect(keys, 100.0).unwrap();
    }
    TrackItem {
        id: ClipId(1),
        item: ItemId(1),
        name: "c".into(),
        label: filmcraft_project::Label::Iris,
        start: Tick::from_seconds_f64(1.0),
        duration,
        source_in: src_in,
        speed: span.0 as f64 / duration.0 as f64,
        reverse: false,
        enabled: true,
        link: None,
        group: None,
        effects,
        markers: vec![],
        gain_db: 0.0,
        frame_hold: None,
        scale_to_frame: false,
        essential: None,
        multicam: None,
        time_interpolation: Default::default(),
        maintain_pitch: keep_pitch,
        hold_filters: false,
        field_options: None,
        source_channels: Vec::new(),
        audio_stream: 0,
        graphic: None,
    }
}

/// Frequency from rising zero crossings.
fn frequency(x: &[f32]) -> f64 {
    let ups: Vec<usize> = x.windows(2).enumerate().filter(|(_, w)| w[0] < 0.0 && w[1] >= 0.0).map(|(i, _)| i).collect();
    match (ups.first(), ups.last()) {
        (Some(a), Some(b)) if ups.len() > 2 => (ups.len() - 1) as f64 / ((b - a) as f64 / SR as f64),
        _ => 0.0,
    }
}

/// Timeline sample `f` of the way into the clip.
fn at(it: &TrackItem, f: f64) -> i64 {
    ((it.start.seconds() + it.duration.seconds() * f) * SR as f64).round() as i64
}

#[test]
fn maintain_pitch_keeps_the_pitch_where_the_curve_slows_down() {
    // Hero: 0.2× around the middle
    let kept = clip("hero", true);
    let a0 = at(&kept, 0.47);
    let b = remapped(&kept, &sine, a0, 4800, SR).expect("remapped");
    let f = frequency(&b.channels[0]);
    assert!((f - HZ).abs() < HZ * 0.03, "pitch kept: {f} Hz");
    let rms = (b.channels[0].iter().map(|v| v * v).sum::<f32>() / b.channels[0].len() as f32).sqrt();
    assert!((0.3..1.2).contains(&rms), "level: {rms}");
    // varispeed: the pitch drops with the speed
    let vari = clip("hero", false);
    let b = remapped(&vari, &sine, a0, 4800, SR).expect("remapped");
    let f = frequency(&b.channels[0]);
    assert!(f < HZ * 0.45, "varispeed pitch drops: {f} Hz");
    // and rises where the curve is fast (2.5×)
    let b = remapped(&vari, &sine, at(&vari, 0.05), 2400, SR).expect("remapped");
    let f = frequency(&b.channels[0]);
    assert!((f - HZ * 2.5).abs() < HZ * 0.15, "fast varispeed: {f} Hz");
}

#[test]
fn chunks_render_the_same_samples() {
    for keep in [true, false] {
        let it = clip("montage", keep);
        let a0 = at(&it, 0.3);
        let whole = remapped(&it, &sine, a0, 3000, SR).unwrap();
        let first = remapped(&it, &sine, a0, 1234, SR).unwrap();
        let second = remapped(&it, &sine, a0 + 1234, 3000 - 1234, SR).unwrap();
        let joined: Vec<f32> = first.channels[0].iter().chain(&second.channels[0]).copied().collect();
        assert_eq!(whole.channels[0], joined, "keep pitch {keep}");
    }
}

#[test]
fn clips_without_a_curve_are_left_to_the_other_paths() {
    let mut it = clip("hero", true);
    if let Some(e) = it.effects.iter_mut().find(|e| e.effect == "time_remap") {
        e.params.get_mut("speed").unwrap().keyframes.clear();
    }
    assert!(remapped(&it, &sine, 48_000, 100, SR).is_none());
    let mut held = clip("hero", true);
    held.frame_hold = Some(Tick::ZERO);
    assert!(remapped(&held, &sine, 48_000, 100, SR).is_none());
    // hostile sample rate / length: no panic
    assert!(remapped(&clip("hero", true), &sine, 0, 0, SR).is_none());
    assert!(remapped(&clip("hero", true), &sine, 0, 10, 0).is_none());
}

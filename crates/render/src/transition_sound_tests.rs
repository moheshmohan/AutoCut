//! Sounds coupled to video transitions (roadmap C4): mixed into the Mix to the sample, at their
//! gain, gone with the transition, and through a latency-compensated graph.

use filmcraft_project::{ItemId, Project, SequenceSettings, SoundKind, Transition, TransitionId, TransitionSound};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::SourceMap;
use crate::mixer::{mix_graph, transition_sound_samples};

const SR: u32 = 48_000;

/// A sequence (one video track, one audio track, no clips) with a 1 s transition at 2 s carrying
/// `sound`.
fn rig(sound: Option<TransitionSound>) -> (Project, ItemId) {
    let mut p = Project::new("sounds");
    let seq = p.new_sequence("s", SequenceSettings::default(), 1, 1, None);
    let effect = filmcraft_project::find_effect("cross_dissolve").unwrap().instance();
    let tr = Transition {
        id: TransitionId(7),
        effect,
        start: Tick(2 * TICKS_PER_SECOND),
        duration: Tick(TICKS_PER_SECOND),
        from: None,
        to: None,
        align: Default::default(),
        reverse: false,
        sound,
    };
    p.sequence_mut(seq).unwrap().video_tracks[0].transitions.push(tr);
    (p, seq)
}

fn mix(p: &Project, seq: ItemId, start: i64, n: usize) -> Vec<Vec<f32>> {
    mix_graph(p, p.sequence(seq).unwrap(), start, n, &SourceMap::default(), None, &crate::MixStream::new()).channels
}

#[test]
fn a_transition_sound_is_mixed_to_the_sample() {
    for kind in SoundKind::ALL {
        let snd = TransitionSound::new(kind, 3);
        let (first, len) = snd.samples(Tick(2 * TICKS_PER_SECOND), Tick(TICKS_PER_SECOND), SR).unwrap();
        // the midpoint is sample 120 000: whoosh centred, riser ending, impact starting there
        match kind {
            SoundKind::Whoosh | SoundKind::Swish => assert_eq!(first + len as i64 / 2, 120_000),
            SoundKind::Riser => assert_eq!(first + len as i64, 120_000),
            SoundKind::Impact | SoundKind::Chime => assert_eq!(first, 120_000),
        }
        let (p, seq) = rig(Some(snd.clone()));
        let out = mix(&p, seq, 0, 6 * SR as usize);
        let synth = transition_sound_samples(&snd, len, SR).unwrap();
        assert_eq!((synth[0].len(), synth[1].len()), (len, len), "{kind:?}: exactly the span's samples");
        let g = snd.gain();
        for (c, ch) in out.iter().enumerate().take(2) {
            // silent up to the first sample and from the last on, the synth's samples between
            assert!(ch[..first as usize].iter().all(|v| *v == 0.0), "{kind:?} before");
            assert!(ch[first as usize + len..].iter().all(|v| *v == 0.0), "{kind:?} after");
            let body = &ch[first as usize..first as usize + len];
            assert!(body.iter().zip(&synth[c]).all(|(a, b)| (a - b * g).abs() < 1e-6), "{kind:?} channel {c}");
            assert!(body.iter().any(|v| v.abs() > 1e-3), "{kind:?} is heard");
        }
        // the same samples whichever way playback cuts the timeline into requests
        let a = mix(&p, seq, first - 1000, 3000);
        let b: Vec<f32> = [mix(&p, seq, first - 1000, 1001), mix(&p, seq, first + 1, 1999)].iter().flat_map(|m| m[0].clone()).collect();
        assert_eq!(a[0], b, "{kind:?}: block boundaries");
    }
}

#[test]
fn gain_length_and_removal() {
    let mut snd = TransitionSound::new(SoundKind::Impact, 9);
    snd.gain_db = -12.0;
    snd.length = Some(Tick(TICKS_PER_SECOND / 2));
    let (p, seq) = rig(Some(snd.clone()));
    let (first, len) = snd.samples(Tick(2 * TICKS_PER_SECOND), Tick(TICKS_PER_SECOND), SR).unwrap();
    assert_eq!((first, len), (120_000, 24_000));
    let out = mix(&p, seq, 0, 4 * SR as usize);
    let peak = out[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let loud = {
        let mut s = snd.clone();
        s.gain_db = 0.0;
        let (p, seq) = rig(Some(s));
        mix(&p, seq, 0, 4 * SR as usize)[0].iter().fold(0.0f32, |m, v| m.max(v.abs()))
    };
    assert!((peak / loud - 10f32.powf(-12.0 / 20.0)).abs() < 1e-4, "{peak} vs {loud}");
    assert!(out[0][..120_000].iter().chain(&out[0][144_000..]).all(|v| *v == 0.0));
    // no sound, or the transition removed: silence
    let (p, seq) = rig(None);
    assert!(mix(&p, seq, 0, 4 * SR as usize).iter().all(|c| c.iter().all(|v| *v == 0.0)));
    let (mut p, seq) = rig(Some(snd));
    p.sequence_mut(seq).unwrap().video_tracks[0].transitions.clear();
    assert!(mix(&p, seq, 0, 4 * SR as usize).iter().all(|c| c.iter().all(|v| *v == 0.0)));
}

#[test]
fn hidden_tracks_solo_and_hostile_sounds_are_safe() {
    let snd = TransitionSound::new(SoundKind::Whoosh, 1);
    // a video track switched off keeps its transitions quiet
    let (mut p, seq) = rig(Some(snd.clone()));
    p.sequence_mut(seq).unwrap().video_tracks[0].enabled = false;
    assert!(mix(&p, seq, 0, 4 * SR as usize)[0].iter().all(|v| *v == 0.0));
    // soloing an audio track mutes everything else, transition sounds included
    let (mut p, seq) = rig(Some(snd.clone()));
    p.sequence_mut(seq).unwrap().audio_tracks[0].solo = true;
    assert!(mix(&p, seq, 0, 4 * SR as usize)[0].iter().all(|v| *v == 0.0));
    // hostile numbers never panic or leak non-finite samples
    for (gain, intensity, len) in [(f64::NAN, f64::NAN, Some(Tick(i64::MAX))), (1e300, -5.0, Some(Tick(-1))), (-1e300, 1e9, Some(Tick(0)))] {
        let mut s = snd.clone();
        s.gain_db = gain;
        s.intensity = intensity;
        s.length = len;
        let (mut p, seq) = rig(Some(s));
        {
            let tr = &mut p.sequence_mut(seq).unwrap().video_tracks[0].transitions[0];
            tr.start = Tick(i64::MAX / 8);
            tr.duration = Tick(i64::MIN);
        }
        let out = mix(&p, seq, 0, SR as usize);
        assert!(out.iter().all(|c| c.iter().all(|v| v.is_finite())));
        let (p, seq) = rig(Some(snd.clone()));
        assert!(mix(&p, seq, i64::MAX / 4, 64).iter().all(|c| c.iter().all(|v| v.is_finite())));
    }
    // a rate the synth cannot make is silent rather than an error
    assert!(transition_sound_samples(&snd, 100, 1_000).is_none());
}

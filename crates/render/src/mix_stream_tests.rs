//! Stateful audio (mixer inserts, the Mix's limiter, clip effect chains, time-stretch memos)
//! continues only the stream it came from: two renders of the same structure, interleaved chunk by
//! chunk from the same position, each give exactly what they give alone.

use filmcraft_project::{ItemKind, MediaRef, ParamValue};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::MixStream;
use crate::mix_stream::{Continues, StateCache};
use crate::mixer;
use crate::mixer::mix_graph;
use crate::mixer_tests::support::{Rig, noise};

/// Loud noise: the dynamics work hard on it.
fn loud(i: i64, c: usize) -> f32 {
    noise(i, c) * 2.5
}

/// A quieter tone in bursts: another envelope entirely.
fn bursts(i: i64, c: usize) -> f32 {
    let t = i as f32 / 48_000.0;
    let on = if (i / 7_000) % 2 == 0 { 0.9 } else { 0.05 };
    (t * 2.0 * std::f32::consts::PI * (220.0 + 110.0 * c as f32)).sin() * on
}

fn effect(id: &str, params: &[(&str, f64)], post: bool) -> filmcraft_project::EffectInstance {
    let mut e = filmcraft_project::find_effect(id).unwrap().instance();
    for (k, v) in params {
        e.param_mut(k).unwrap().value = ParamValue::Float(*v);
    }
    e.post_fader = post;
    e
}

/// One clip of `signal` on A1 with a compressor insert on the track and a Hard Limiter on the Mix.
fn graph_rig(signal: fn(i64, usize) -> f32) -> Rig {
    let mut r = Rig::new(&[signal]);
    r.track(0).effects.push(effect("dynamics", &[("threshold", -30.0), ("ratio", 8.0), ("release", 2000.0)], false));
    r.p.sequence_mut(r.seq).unwrap().master_effects.push(effect("hard_limiter", &[("max", -12.0), ("boost", 6.0), ("release", 900.0)], true));
    r
}

/// One clip of `signal` on A1 with clip effects (a compressor and a delay), no mixer inserts.
fn clip_rig(signal: fn(i64, usize) -> f32) -> Rig {
    let mut r = Rig::new(&[signal]);
    let item = &mut r.track(0).items[0];
    item.effects.push(effect("dynamics", &[("threshold", -30.0), ("ratio", 8.0), ("release", 2000.0)], false));
    item.effects.push(effect("delay", &[("delay", 0.3), ("feedback", 60.0), ("mix", 50.0)], false));
    r
}

/// One clip of `signal` on A1 slowed to 80 % with Maintain Audio Pitch (a time stretch), its media
/// at `path`.
fn stretch_rig(signal: fn(i64, usize) -> f32, path: &str) -> Rig {
    let mut r = Rig::new(&[signal]);
    let it = &mut r.track(0).items[0];
    it.speed = 0.8;
    it.duration = Tick((10.0 * TICKS_PER_SECOND as f64 / 0.8).round() as i64);
    it.maintain_pitch = true;
    let media = it.item;
    match r.p.item_mut(media).map(|i| &mut i.kind) {
        Some(ItemKind::Media(m)) => m.media = MediaRef::File { path: path.into() },
        _ => panic!("the rig's clip plays a media item"),
    }
    r
}

const CHUNK: usize = 4_800;
const CHUNKS: usize = 12;

/// One chunk, left channel then right.
fn render(r: &Rig, pos: i64, stream: &MixStream) -> Vec<f32> {
    mix_graph(&r.p, r.p.sequence(r.seq).unwrap(), pos, CHUNK, &r.map, None, stream).channels.concat()
}

/// A rig rendered alone, chunk by chunk from 0, in a stream of its own.
fn solo(r: &Rig) -> Vec<f32> {
    let stream = MixStream::new();
    (0..CHUNKS).flat_map(|k| render(r, (k * CHUNK) as i64, &stream)).collect()
}

/// Both rigs rendered chunk by chunk from 0, each in its own stream, interleaved (A B, then B A, …).
fn interleaved(a: &Rig, b: &Rig) -> (Vec<f32>, Vec<f32>) {
    let (sa, sb) = (MixStream::new(), MixStream::new());
    let (mut oa, mut ob) = (Vec::new(), Vec::new());
    for k in 0..CHUNKS {
        let pos = (k * CHUNK) as i64;
        if k % 2 == 0 {
            oa.extend(render(a, pos, &sa));
            ob.extend(render(b, pos, &sb));
        } else {
            ob.extend(render(b, pos, &sb));
            oa.extend(render(a, pos, &sa));
        }
    }
    (oa, ob)
}

fn assert_same(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(x, y)| x.to_bits() != y.to_bits()) {
        panic!("{what}: sample {i} is {} alone but {} interleaved with another stream", want[i], got[i]);
    }
}

#[test]
fn interleaved_mixes_with_the_same_graph_keep_their_own_dynamics() {
    let (a, b) = (graph_rig(loud), graph_rig(bursts));
    let (want_a, want_b) = (solo(&a), solo(&b));
    let (got_a, got_b) = interleaved(&a, &b);
    assert_same("loud", &got_a, &want_a);
    assert_same("bursts", &got_b, &want_b);
}

#[test]
fn interleaved_clips_with_the_same_item_id_keep_their_own_effect_chains() {
    let (a, b) = (clip_rig(loud), clip_rig(bursts));
    assert_eq!(clip_id(&a), clip_id(&b), "the two projects' clips share an id");
    let (want_a, want_b) = (solo(&a), solo(&b));
    let (got_a, got_b) = interleaved(&a, &b);
    assert_same("loud", &got_a, &want_a);
    assert_same("bursts", &got_b, &want_b);
}

/// One clip of `signal` on A1 with the speech effects: Voice Isolation, Enhance Voice, a Voice
/// Changer (stateful STFTs, model memories, a pitch shifter, delay lines).
fn voice_rig(signal: fn(i64, usize) -> f32) -> Rig {
    let mut r = Rig::new(&[signal]);
    let item = &mut r.track(0).items[0];
    item.effects.push(effect("voice_isolation", &[("intensity", 90.0)], false));
    item.effects.push(effect("enhance_voice", &[("amount", 60.0)], false));
    let mut changer = effect("voice_changer", &[("character", 70.0)], false);
    changer.param_mut("mode").unwrap().value = ParamValue::Choice(3);
    item.effects.push(changer);
    r
}

#[test]
fn interleaved_clips_keep_their_own_voice_effects() {
    let (a, b) = (voice_rig(loud), voice_rig(bursts));
    assert_eq!(clip_id(&a), clip_id(&b));
    let (want_a, want_b) = (solo(&a), solo(&b));
    let (got_a, got_b) = interleaved(&a, &b);
    assert_same("loud", &got_a, &want_a);
    assert_same("bursts", &got_b, &want_b);
}

/// The speech effects' latency (40 ms and more) is compensated: with nothing mixed in (Voice
/// Isolation at 0 %, Voice Changer at 0 % mix) the clip sounds exactly where it does without them.
#[test]
fn voice_effect_latency_is_compensated() {
    let plain = solo(&Rig::new(&[bursts]));
    let mut r = Rig::new(&[bursts]);
    let item = &mut r.track(0).items[0];
    item.effects.push(effect("voice_isolation", &[("intensity", 0.0)], false));
    item.effects.push(effect("voice_changer", &[("mix", 0.0)], false));
    let got = solo(&r);
    assert!(crate::audio_fx::latency("voice_isolation", 48_000) >= 1920);
    assert_eq!(got.len(), plain.len());
    let worst = got.iter().zip(&plain).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "out of sync or changed: worst sample differs by {worst}");
}

/// Two projects whose stretched clips have the same media reference but different sound (a file
/// replaced between sessions, two tests importing under one name): each stretches its own sound,
/// exactly as when its media is named apart.
#[test]
fn stretched_clips_of_same_named_media_keep_their_own_splices() {
    let want_a = solo(&stretch_rig(loud, "/stretch-loud.wav"));
    let want_b = solo(&stretch_rig(bursts, "/stretch-bursts.wav"));
    let (a, b) = (stretch_rig(loud, "/stretch-same.wav"), stretch_rig(bursts, "/stretch-same.wav"));
    let (got_a, got_b) = interleaved(&a, &b);
    assert_same("loud", &got_a, &want_a);
    assert_same("bursts", &got_b, &want_b);
}

/// A stream continues itself exactly: chunk by chunk gives what one call gives.
#[test]
fn a_stream_continues_itself_whatever_the_chunks() {
    let r = graph_rig(loud);
    let whole = mix_graph(&r.p, r.p.sequence(r.seq).unwrap(), 0, CHUNK * CHUNKS, &r.map, None, &MixStream::new());
    let left: Vec<f32> = solo(&r).chunks(2 * CHUNK).flat_map(|c| c[..CHUNK].to_vec()).collect();
    assert_same("chunked", &left, &whole.channels[0]);
}

fn clip_id(r: &Rig) -> filmcraft_project::ClipId {
    r.p.sequence(r.seq).unwrap().audio_tracks[0].items[0].id
}

struct At(i64);

impl Continues for At {
    fn next_out(&self) -> i64 {
        self.0
    }
}

#[test]
fn state_cache_is_bounded_least_recently_used_first() {
    let mut c: StateCache<u32, At> = StateCache::new(3, 2);
    for k in 0..3 {
        c.put(k, At(10));
    }
    // a touch keeps key 0; a fourth key drops the least recently used (1)
    let s = c.take(&0, 10).unwrap();
    c.put(0, At(s.0 + 5));
    c.put(3, At(10));
    assert_eq!(c.len(), 3);
    assert!(c.take(&1, 10).is_none());
    assert!(c.take(&0, 15).is_some() && c.take(&2, 10).is_some() && c.take(&3, 10).is_some());
    // at most `per_key` states per key, the oldest dropped
    for at in [1, 2, 3] {
        c.put(7, At(at));
    }
    assert!(c.take(&7, 1).is_none());
    assert!(c.take(&7, 2).is_some() && c.take(&7, 3).is_some());
    // a state is only for the position it continues from
    c.put(8, At(100));
    assert!(c.take(&8, 99).is_none() && c.take(&8, 100).is_some());
    c.put(9, At(1));
    c.put(10, At(1));
    c.forget(|k| *k == 9);
    assert!(c.take(&9, 1).is_none() && c.take(&10, 1).is_some());
}

/// Hostile sizes: a cache of no keys still keeps the newest state.
#[test]
fn state_cache_of_zero_size_still_works() {
    let mut c: StateCache<u32, At> = StateCache::new(0, 0);
    c.put(1, At(5));
    c.put(2, At(5));
    assert!(c.take(&1, 5).is_none() && c.take(&2, 5).is_some());
}

/// Dropping a stream drops the graph states it left behind.
#[test]
fn a_dropped_stream_leaves_nothing_behind() {
    let r = graph_rig(bursts);
    let stream = MixStream::new();
    render(&r, 0, &stream);
    let key = stream.key();
    assert!(mixer::graph_states_of(key) > 0);
    drop(stream);
    assert_eq!(mixer::graph_states_of(key), 0);
}

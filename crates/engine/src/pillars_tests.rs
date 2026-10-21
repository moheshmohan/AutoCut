use filmcraft_project::{ItemId, Label, Project, SequenceSettings, TrackItem};
use filmcraft_time::{FrameRate, Tick};

use super::*;

#[test]
fn the_copies_reach_under_their_pictures_transitions() {
    use PillarEdit::{Copies, Fades, In, Out};
    let span = Span::new;
    let join = |from: Option<u64>, to: Option<u64>, start: i64, end: i64| Join { id: 99, start, end, from, to, dip: None };
    let dip = |j: Join| Join { dip: Some("dip_to_black"), ..j };
    let copies_of = |effect, from, to, start, end| Copies { join: 99, effect, from, to, start, end };
    // pictures 1, 2, 3 cut at 100 and 200; 1 and 3 shown whole (copies 11 and 13)
    let backdrops = [Backdrop::whole(1, 11), Backdrop::whole(3, 13)];
    let copies = [span(11, 0, 100), span(13, 200, 300)];
    // out of a whole picture: its copy stays on to the transition's end; into one: its copy is
    // there from the transition's start
    let joins = [join(Some(1), Some(2), 90, 110), join(Some(2), Some(3), 190, 210)];
    assert_eq!(pillar_edits(&copies, &joins, &backdrops), [Out { clip: 11, by: 10 }, In { clip: 13, by: 10 }]);
    // a dip: the same, the pictures' dip goes to the copies' track (each copy dipping at its own
    // edge) and the pictures fade out and in over it
    let joins = [dip(join(Some(1), Some(2), 90, 110)), dip(join(Some(2), Some(3), 190, 210))];
    assert_eq!(
        pillar_edits(&copies, &joins, &backdrops),
        [
            Out { clip: 11, by: 10 },
            Fades { transition: 99, from: Some(1), to: Some(2), start: 90, end: 110 },
            copies_of("dip_to_black", Some(11), None, 90, 110),
            In { clip: 13, by: 10 },
            Fades { transition: 99, from: Some(2), to: Some(3), start: 190, end: 210 },
            copies_of("dip_to_black", None, Some(13), 190, 210),
        ]
    );
    // two pictures shown whole, cut together: their copies dissolve (or dip) into each other
    let copies2 = [span(11, 0, 100), span(12, 100, 200)];
    let both = [Backdrop::whole(1, 11), Backdrop::whole(2, 12)];
    let j = join(Some(1), Some(2), 90, 110);
    assert_eq!(pillar_edits(&copies2, &[j], &both), [copies_of("cross_dissolve", Some(11), Some(12), 90, 110)]);
    assert_eq!(
        pillar_edits(&copies2, &[dip(j)], &both),
        [Fades { transition: 99, from: Some(1), to: Some(2), start: 90, end: 110 }, copies_of("dip_to_black", Some(11), Some(12), 90, 110)]
    );
    // a copy reaches no further than the next one starts (a short shot between)
    let copies3 = [span(11, 0, 100), span(13, 104, 300)];
    let joins = [join(Some(1), Some(2), 90, 110), join(Some(2), Some(3), 100, 108)];
    assert_eq!(pillar_edits(&copies3, &joins, &backdrops), [Out { clip: 11, by: 4 }]);
    // ... nor further than its media: then a dip goes down to its copy as far as it reaches
    let short = [Span { latest: 105, ..span(11, 0, 100) }, Span { earliest: 197, ..span(13, 200, 300) }];
    let joins = [dip(join(Some(1), Some(2), 90, 110)), dip(join(Some(2), Some(3), 190, 210))];
    assert_eq!(
        pillar_edits(&short, &joins, &backdrops),
        [
            Out { clip: 11, by: 5 },
            Fades { transition: 99, from: Some(1), to: Some(2), start: 90, end: 110 },
            copies_of("dip_to_black", Some(11), None, 90, 105),
            In { clip: 13, by: 3 },
            Fades { transition: 99, from: Some(2), to: Some(3), start: 190, end: 210 },
            copies_of("dip_to_black", None, Some(13), 197, 210),
        ]
    );
    // a transition beside no picture shown whole, a one-sided one inside its picture, a dip
    // between two filled pictures: nothing to do
    let joins = [join(Some(2), Some(4), 400, 420), join(Some(1), None, 80, 100), dip(join(Some(5), Some(6), 500, 520))];
    assert!(pillar_edits(&copies, &joins, &backdrops).is_empty());
    // the outro's dip out of a picture shown whole, at the very end: its copy dips there
    let j = dip(join(Some(3), None, 280, 300));
    assert_eq!(
        pillar_edits(&copies, &[j], &backdrops),
        [Fades { transition: 99, from: Some(3), to: None, start: 280, end: 300 }, copies_of("dip_to_black", Some(13), None, 280, 300)]
    );
    // a copy under only part of its picture (a reframe's blur-filled stretch) reaches only from
    // the edge it is at
    let start_only = [Backdrop { picture: 1, copy: 11, head: true, tail: false }];
    let joins = [join(Some(0), Some(1), -10, 10), join(Some(1), Some(2), 90, 110)];
    assert_eq!(pillar_edits(&[span(11, 0, 50)], &joins, &start_only), [In { clip: 11, by: 10 }]);
    // a dip whose copy has nowhere to dip (another copy touches it at the cut) stays on the
    // pictures, which then do not fade
    let touching = [span(11, 0, 100), span(14, 110, 200)];
    assert_eq!(pillar_edits(&touching, &[dip(join(Some(1), Some(2), 90, 110))], &backdrops), [Out { clip: 11, by: 10 }]);
    // hostile spans: empty or backwards transitions and copies are skipped, extremes saturate
    // (the first copy reaches the second, which then has no room to reach back)
    let joins = [join(Some(1), Some(2), 110, 90), join(Some(1), Some(2), i64::MIN, i64::MAX), join(Some(2), Some(3), i64::MIN, i64::MAX)];
    let copies4 = [span(11, 0, 100), span(13, 300, 200), span(13, 200, 300)];
    let edits = pillar_edits(&copies4, &joins, &backdrops);
    assert_eq!(edits, [Out { clip: 11, by: 100 }], "{edits:?}");
    assert!(pillar_edits(&[span(11, i64::MIN, i64::MAX)], &[join(Some(1), None, i64::MIN, i64::MAX)], &backdrops).is_empty());
    let edits = pillar_edits(&[Span { earliest: i64::MAX, latest: i64::MIN, ..span(11, 0, 100) }], &joins, &backdrops);
    assert!(edits.is_empty(), "{edits:?}");
}

const R: FrameRate = FrameRate::FPS_24;

fn f(n: i64) -> Tick {
    R.tick_of(n)
}

/// A clip of media `item` at frame `start`, `dur` frames long, from media frame `src`.
fn clip(id: u64, item: u64, start: i64, dur: i64, src: i64) -> TrackItem {
    TrackItem {
        id: ClipId(id),
        item: ItemId(item),
        name: format!("c{id}"),
        label: Label::Iris,
        start: f(start),
        duration: f(dur),
        source_in: f(src),
        speed: 1.0,
        reverse: false,
        enabled: true,
        link: None,
        group: None,
        effects: filmcraft_project::effect::intrinsic_video(),
        markers: vec![],
        gain_db: 0.0,
        frame_hold: None,
        scale_to_frame: false,
        essential: None,
        multicam: None,
        time_interpolation: Default::default(),
        maintain_pitch: false,
        hold_filters: false,
        field_options: None,
        source_channels: Vec::new(),
        audio_stream: 0,
        graphic: None,
    }
}

#[test]
fn a_copy_reaches_as_far_as_its_media() {
    // 30 frames from media frame 10 of 50: 10 frames of media before it, 10 after
    let it = clip(1, 7, 100, 30, 10);
    let media = (Tick::ZERO, Some(f(50)));
    assert_eq!(reach(&it, media, R), (f(90).0, f(140).0));
    // reversed, its In edge takes the media after it and its Out edge the media before
    let rev = TrackItem { reverse: true, source_in: f(15), ..it.clone() };
    assert_eq!(reach(&rev, media, R), (f(95).0, f(145).0));
    // at double speed a frame of media is half a frame of the timeline (on frames, inside it)
    let fast = TrackItem { speed: 2.0, duration: f(10), ..it.clone() };
    assert_eq!(reach(&fast, media, R), (f(95).0, f(120).0));
    // a still, a frame hold: as far as it needs; a subclip starting later: no further back
    assert_eq!(reach(&it, (Tick::ZERO, None), R), (f(90).0, i64::MAX));
    assert_eq!(reach(&TrackItem { frame_hold: Some(f(3)), ..it.clone() }, media, R), (i64::MIN, i64::MAX));
    assert_eq!(reach(&it, (f(10), Some(f(40))), R), (f(100).0, f(130).0));
    // hostile: a NaN or zero speed, media that ends before the clip, extreme times
    for speed in [f64::NAN, 0.0, f64::INFINITY, -1.0] {
        let (a, b) = reach(&TrackItem { speed, ..it.clone() }, media, R);
        assert!(a <= it.start.0 && b >= it.end().0 - f(1).0, "{speed}: {a} {b}");
    }
    let (a, b) = reach(&it, (f(60), Some(f(5))), R);
    assert!(a <= it.start.0 && b <= it.end().0 + f(1).0, "{a} {b}");
    let far = TrackItem { start: Tick::MAX, source_in: Tick::MIN, ..it.clone() };
    let _ = reach(&far, (Tick::MAX, Some(Tick::MIN)), R);
}

#[test]
fn a_copy_belongs_to_the_picture_it_shows() {
    // one shot cut in two (a through edit), and another shot of the same media
    let pictures = [clip(1, 7, 0, 30, 0), clip(2, 7, 30, 30, 30), clip(3, 7, 60, 30, 0)];
    // a copy of the second, reaching back under a transition into the first: the second's
    assert_eq!(owner_of(&pictures, &clip(11, 7, 26, 34, 26)), Some(ClipId(2)));
    // a copy of the third reaching back into the second shows other media there: still the third's
    assert_eq!(owner_of(&pictures, &clip(12, 7, 56, 34, -4)), Some(ClipId(3)));
    // other media, another speed, nothing under it: nobody's
    assert_eq!(owner_of(&pictures, &clip(13, 8, 0, 30, 0)), None);
    assert_eq!(owner_of(&pictures, &TrackItem { speed: 2.0, ..clip(14, 7, 0, 30, 0) }), None);
    assert_eq!(owner_of(&pictures, &clip(15, 7, 200, 30, 0)), None);
}

/// A sequence of pictures on V2 over their copies on V1 (24 fps).
fn sequence(pictures: Vec<TrackItem>, copies: Vec<TrackItem>) -> Sequence {
    let mut p = Project::new("t");
    let s = p.new_sequence("s", SequenceSettings { frame_rate: R, ..Default::default() }, 2, 0, None);
    let mut q = p.sequence(s).unwrap().clone();
    q.video_tracks[1].items = pictures;
    q.video_tracks[0].items = copies;
    q
}

fn transition(id: u64, effect: &str, from: Option<u64>, to: Option<u64>, start: i64, dur: i64) -> Transition {
    Transition {
        id: TransitionId(id),
        effect: find_transition(effect, EffectKind::VideoTransition).unwrap().instance(),
        start: f(start),
        duration: f(dur),
        from: from.map(ClipId),
        to: to.map(ClipId),
        align: TransitionAlign::CenterAtCut,
        reverse: false,
        sound: None,
    }
}

fn spans(q: &Sequence, track: usize) -> Vec<(u64, i64, i64)> {
    q.video_tracks[track].items.iter().map(|i| (i.id.0, R.frame_at(i.start), R.frame_at(i.end()))).collect()
}

fn fills(q: &Sequence) -> Vec<(String, Option<u64>, Option<u64>, i64, i64)> {
    q.video_tracks[0]
        .transitions
        .iter()
        .map(|x| (x.effect.effect.clone(), x.from.map(|c| c.0), x.to.map(|c| c.0), R.frame_at(x.start), R.frame_at(x.end())))
        .collect()
}

fn opacity_keys(q: &Sequence, clip: u64) -> Vec<(i64, f64)> {
    let it = q.video_tracks[1].item(ClipId(clip)).unwrap();
    let p = it.effect("opacity").unwrap().param("opacity").unwrap();
    p.keyframes.iter().map(|k| (R.frame_at(k.time), k.value.as_f64().unwrap())).collect()
}

#[test]
fn the_blur_fill_follows_the_transitions_on_the_sequence() {
    // A (whole) dips into B (followed), which pushes into C (whole), which dissolves into D
    // (whole, media for only two frames before it); 30 frames each, all of media 7 from frame 10
    let pictures = vec![clip(1, 7, 0, 30, 10), clip(2, 7, 30, 30, 10), clip(3, 7, 60, 30, 10), clip(4, 7, 90, 30, 2)];
    let copies = vec![clip(11, 7, 0, 30, 10), clip(13, 7, 60, 30, 10), clip(14, 7, 90, 30, 2)];
    let mut q = sequence(pictures, copies);
    q.video_tracks[1].transitions = vec![
        transition(901, "dip_to_black", Some(1), Some(2), 26, 8),
        transition(902, "push", Some(2), Some(3), 56, 8),
        transition(903, "cross_dissolve", Some(3), Some(4), 86, 8),
    ];
    let owners = [(ClipId(11), ClipId(1)), (ClipId(13), ClipId(3)), (ClipId(14), ClipId(4))];
    let reach = |it: &TrackItem| reach(it, (Tick::ZERO, Some(f(200))), R);
    let mut next = 1000;
    let mut alloc = || {
        next += 1;
        next
    };
    through_transitions(&mut q, (1, 0), &owners, &[], &reach, &mut alloc);
    q.check().unwrap();
    // A's copy on under the dip, C's back under the push; C's and D's meet at the cut
    assert_eq!(spans(&q, 0), [(11, 0, 34), (13, 56, 90), (14, 90, 120)]);
    // the dip went down to the copies (A's dipping at its own edge, the very transition, its id
    // kept); C's and D's copies dissolve as their pictures do
    let mut f = fills(&q);
    f.sort();
    assert_eq!(f, [("cross_dissolve".into(), Some(13), Some(14), 86, 94), ("dip_to_black".into(), Some(11), None, 26, 34)]);
    assert!(q.video_tracks[0].transitions.iter().any(|x| x.id.0 == 901));
    let pics: Vec<&str> = q.video_tracks[1].transitions.iter().map(|x| x.effect.effect.as_str()).collect();
    assert_eq!(pics, ["push", "cross_dissolve"], "the pictures keep the others");
    // A fades out over the dip's first half, B in over its second (keyframes in media time: both
    // play their media from frame 10)
    assert_eq!(opacity_keys(&q, 1), [(26 + 10, 100.0), (30 + 10, 0.0)]);
    assert_eq!(opacity_keys(&q, 2), [(10, 0.0), (14, 100.0)]);
    // D's copy cannot reach back past its media (two frames) — and needs not, meeting C's; the
    // reach stops a lone one
    let mut lone = sequence(vec![clip(1, 7, 0, 30, 0), clip(2, 7, 30, 30, 2)], vec![clip(12, 7, 30, 30, 2)]);
    lone.video_tracks[1].transitions = vec![transition(905, "push", Some(1), Some(2), 26, 8)];
    through_transitions(&mut lone, (1, 0), &[(ClipId(12), ClipId(2))], &[], &reach, &mut alloc);
    assert_eq!(spans(&lone, 0), [(12, 28, 60)]);
    // nothing to do without the tracks, owners or transitions
    let before = (spans(&lone, 0), fills(&lone), lone.video_tracks[1].transitions.clone());
    through_transitions(&mut lone, (5, 0), &[(ClipId(12), ClipId(2))], &[], &reach, &mut alloc);
    through_transitions(&mut lone, (1, 0), &[], &[], &reach, &mut alloc);
    assert_eq!((spans(&lone, 0), fills(&lone), lone.video_tracks[1].transitions.clone()), before);
}

#[test]
fn a_second_run_takes_over_the_copies_and_dips_of_the_first() {
    // the first run: A whole, B followed, C whole — A's copy on under the dip into B, C's back
    // under the push from B
    let pictures = vec![clip(1, 7, 0, 30, 10), clip(2, 7, 30, 30, 10), clip(3, 7, 60, 30, 10)];
    let mut q = sequence(pictures.clone(), vec![clip(11, 7, 0, 30, 10), clip(13, 7, 60, 30, 10)]);
    q.video_tracks[1].transitions = vec![transition(901, "dip_to_black", Some(1), Some(2), 26, 8), transition(902, "push", Some(2), Some(3), 56, 8)];
    let reach = |it: &TrackItem| reach(it, (Tick::ZERO, None), R);
    let mut next = 1000;
    let mut alloc = || {
        next += 1;
        next
    };
    through_transitions(&mut q, (1, 0), &[(ClipId(11), ClipId(1)), (ClipId(13), ClipId(3))], &[], &reach, &mut alloc);
    assert_eq!(spans(&q, 0), [(11, 0, 34), (13, 56, 90)]);
    // the second: B filled whole now — the copies reaching under it back off, the dip stays
    let pictures = q.video_tracks[1].items.clone();
    let cleared = clear_under(&mut q.video_tracks[0], &pictures, &[ClipId(2)]);
    assert_eq!(spans(&q, 0), [(11, 0, 30), (13, 60, 90)]);
    assert_eq!(cleared.moved.iter().map(|j| (j.id, j.from, j.to)).collect::<Vec<_>>(), [(901, Some(1), Some(2))]);
    let mut owners = cleared.owners;
    owners.sort();
    assert_eq!(owners, [(ClipId(11), ClipId(1)), (ClipId(13), ClipId(3))]);
    q.video_tracks[0].items.push(clip(12, 7, 30, 30, 10));
    q.video_tracks[0].sort();
    owners.push((ClipId(12), ClipId(2)));
    through_transitions(&mut q, (1, 0), &owners, &cleared.moved, &reach, &mut alloc);
    q.check().unwrap();
    // the copies meet at both cuts: the dip goes over from A's copy into B's, and C's dissolves
    // in from B's under the push; the pictures' fades are left as they were
    assert_eq!(spans(&q, 0), [(11, 0, 30), (12, 30, 60), (13, 60, 90)]);
    let mut f = fills(&q);
    f.sort();
    assert_eq!(f, [("cross_dissolve".into(), Some(12), Some(13), 56, 64), ("dip_to_black".into(), Some(11), Some(12), 26, 34)]);
    assert!(q.video_tracks[0].transitions.iter().any(|x| x.id.0 == 901));
    assert_eq!(opacity_keys(&q, 2), [(10, 0.0), (14, 100.0)]);
    // a third with B followed again (its copy gone): the dip goes back to A's copy alone
    let pictures = q.video_tracks[1].items.clone();
    let cleared = clear_under(&mut q.video_tracks[0], &pictures, &[ClipId(2)]);
    q.video_tracks[0].items.retain(|c| c.id != ClipId(12));
    through_transitions(&mut q, (1, 0), &cleared.owners, &cleared.moved, &reach, &mut alloc);
    q.check().unwrap();
    assert_eq!(spans(&q, 0), [(11, 0, 34), (13, 56, 90)]);
    assert_eq!(fills(&q), [("dip_to_black".into(), Some(11), None, 26, 34)]);
    // copies nobody owns under a refilled picture go; empty tracks are fine
    let mut t = q.video_tracks[0].clone();
    t.items.push(clip(19, 8, 95, 5, 0));
    let c = clear_under(&mut t, &[clip(5, 7, 90, 30, 0)], &[ClipId(5)]);
    assert!(t.items.iter().all(|i| i.id != ClipId(19)) && c.moved.is_empty());
    let mut empty = Track::new(filmcraft_project::TrackId(1), filmcraft_project::TrackKind::Video, "Blur Fill".into());
    let c = clear_under(&mut empty, &[], &[ClipId(5)]);
    assert!(c.moved.is_empty() && c.owners.is_empty());
}

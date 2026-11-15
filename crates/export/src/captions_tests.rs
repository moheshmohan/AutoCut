//! Export ▸ Captions ▸ Embed in Output File: the caption track comes back out of the H.264 file
//! (read with our own MP4 demuxer and SEI parser), the stream still decodes with our H.264
//! decoder, and hostile captions never stop an export.

use super::*;
use filmcraft_captions::embed::{self, Codec, Stream};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{Generator, MediaSource};
use filmcraft_project::{Caption, CaptionFormat, CaptionTrack, ClipId, ItemKind, Label, MediaClip, MediaRef, SequenceSettings, TrackId, TrackKind};
use filmcraft_render::SourceMap;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("fc-export-cc-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().to_string()
}

/// A `frames`-long matte sequence at `rate` with one caption track holding `captions`
/// `(start frame, end frame, text)`.
fn captioned(rate: FrameRate, frames: i64, captions: &[(i64, i64, &str)]) -> (Arc<Project>, ItemId, SourceMap) {
    let mut p = Project::new("cc");
    let g = GeneratorSource::new(Generator::ColorMatte { color: [0.1, 0.2, 0.4, 1.0] }, 64, 36, rate, rate.tick_of(frames));
    let info = g.info().clone();
    let matte = p.add_item(
        "matte",
        Label::Iris,
        ItemKind::Media(MediaClip {
            media: MediaRef::Generator(g.generator.clone()),
            info,
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        }),
        None,
    );
    let seq = p.new_sequence("s", SequenceSettings { width: 64, height: 36, frame_rate: rate, ..Default::default() }, 1, 0, None);
    let v = p.make_track_item(matte, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, rate.tick_of(frames)), rate).unwrap();
    let mut ct = CaptionTrack::new(TrackId(p.alloc_id()), "CC1".into(), CaptionFormat::Cea608);
    for &(s, e, text) in captions {
        let id = ClipId(p.alloc_id());
        ct.captions.push(Caption {
            id,
            start: rate.tick_of(s),
            duration: rate.tick_of(e) - rate.tick_of(s),
            text: text.into(),
            speaker: None,
            cue_id: None,
            settings: String::new(),
            words: Vec::new(),
        });
    }
    let q = p.sequence_mut(seq).unwrap();
    q.video_tracks[0].items.push(v);
    q.caption_tracks.push(ct);
    let mut m = SourceMap::default();
    m.0.insert(matte, Arc::new(g));
    (Arc::new(p), seq, m)
}

/// Every video sample of an MP4 file: (display frame, bytes), in decoding order, and the avcC.
fn samples(bytes: &[u8]) -> (Vec<(i64, Vec<u8>)>, Vec<u8>) {
    let file = filmcraft_isobmff::open(bytes).unwrap();
    let vt = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
    let t = &file.tracks[vt];
    let avcc = match t.codec() {
        Some(filmcraft_isobmff::CodecConfig::Avc(c)) => c.to_bytes(),
        other => panic!("not H.264: {other:?}"),
    };
    let out = (0..t.samples.len())
        .map(|i| {
            let d = i64::from(t.samples[i].duration.max(1));
            (t.presentation_pts(i).unwrap() / d, file.read_sample(bytes, vt, i).unwrap())
        })
        .collect();
    (out, avcc)
}

/// The caption data of each frame, in display order.
fn cc_frames(samples: &[(i64, Vec<u8>)]) -> Vec<(i64, Vec<[u8; 3]>)> {
    let mut v: Vec<(i64, Vec<[u8; 3]>)> = samples.iter().map(|(f, s)| (*f, embed::sample_cc(s, Codec::H264, 4))).collect();
    v.sort_by_key(|x| x.0);
    v
}

fn export_to(p: &Arc<Project>, seq: ItemId, m: &SourceMap, name: &str, s: ExportSettings) -> Vec<u8> {
    let path = tmp(name);
    let s = ExportSettings { path: path.clone(), include_audio: false, ..s };
    export(p, seq, &s, m, &Progress::default()).unwrap();
    std::fs::read(&path).unwrap()
}

/// The embedded captions read back exactly: both the CEA-608 and the CEA-708 stream give the
/// track's text on the same frames (29.97 fps: the line 21 rate itself), through B-frame
/// reordering.
#[test]
fn embedded_captions_round_trip_through_an_h264_file() {
    let rate = FrameRate::FPS_29_97;
    let caps = [(40, 80, "Hello, world!"), (80, 110, "Second caption\nwith two lines"), (125, 140, "Café ♪")];
    let (p, seq, m) = captioned(rate, 150, &caps);
    let bytes = export_to(&p, seq, &m, "cc2997.mp4", ExportSettings { format: Format::H264, embed_captions: true, ..Default::default() });
    let (smp, avcc) = samples(&bytes);
    assert_eq!(smp.len(), 150);
    let frames = cc_frames(&smp);
    assert_eq!(frames.iter().map(|f| f.0).collect::<Vec<_>>(), (0..150).collect::<Vec<_>>(), "one SEI per frame, every frame shown once");
    assert!(frames.iter().all(|(_, t)| t.len() == embed::cc_count(rate)), "cc_count triplets in every frame");
    for stream in [Stream::Cea608, Stream::Cea708] {
        let got: Vec<(i64, i64, String)> =
            embed::decode(&frames, rate, stream).iter().map(|c| (rate.frame_at(c.start), rate.frame_at(c.end), c.text.clone())).collect();
        let want: Vec<(i64, i64, String)> = caps.iter().map(|&(s, e, t)| (s, e, t.to_string())).collect();
        assert_eq!(got, want, "{stream:?}");
    }
    // the stream still decodes: every picture comes out of our H.264 decoder
    let mut dec = filmcraft_h264::Decoder::from_avcc(&avcc).unwrap();
    let mut pictures = 0;
    for (f, s) in &smp {
        pictures += dec.decode(s, *f).unwrap().len();
    }
    pictures += dec.flush().len();
    assert_eq!(pictures, 150);
    // without the option the file carries no caption SEI
    let plain = export_to(&p, seq, &m, "plain2997.mp4", ExportSettings { format: Format::H264, ..Default::default() });
    assert!(cc_frames(&samples(&plain).0).iter().all(|(_, t)| t.is_empty()));
}

/// At 23.976 fps (more than one line 21 slot a frame) and over an export range that starts
/// later, captions keep their text and land within a frame of their times.
#[test]
fn embedded_captions_follow_the_range_and_the_rate() {
    let rate = FrameRate::FPS_23_976;
    let caps = [(30, 60, "Before the range"), (100, 140, "Inside the range"), (150, 190, "Runs past the end")];
    let (p, seq, m) = captioned(rate, 200, &caps);
    let range = TimeRange::from_bounds(rate.tick_of(70), rate.tick_of(170));
    let bytes = export_to(&p, seq, &m, "cc23976.mp4", ExportSettings { format: Format::H264, embed_captions: true, range: Some(range), ..Default::default() });
    let (smp, _) = samples(&bytes);
    assert_eq!(smp.len(), 100);
    let frames = cc_frames(&smp);
    for stream in [Stream::Cea608, Stream::Cea708] {
        let got: Vec<(i64, i64, String)> =
            embed::decode(&frames, rate, stream).iter().map(|c| (rate.frame_at(c.start), rate.frame_at(c.end), c.text.clone())).collect();
        assert_eq!(got.len(), 2, "{stream:?}: {got:?}");
        assert_eq!((got[0].2.as_str(), got[1].2.as_str()), ("Inside the range", "Runs past the end"), "{stream:?}");
        assert!((got[0].0 - 30).abs() <= 1 && (got[0].1 - 70).abs() <= 1, "{stream:?}: {got:?}");
        assert!((got[1].0 - 80).abs() <= 1 && got[1].1 >= 99, "{stream:?}: {got:?}");
    }
}

#[test]
fn embedding_needs_h264_or_hevc() {
    let s = ExportSettings { format: Format::ProRes, embed_captions: true, ..Default::default() };
    let e = s.validate().unwrap_err().to_string();
    assert!(e.contains("H.264 or H.265"), "{e}");
    assert!(ExportSettings { format: Format::H264, embed_captions: true, ..Default::default() }.validate().is_ok());
    assert!(ExportSettings { format: Format::Hevc, embed_captions: true, ..Default::default() }.validate().is_ok());
    // settings JSON: `embedCaptions`, off when missing
    let j: ExportSettings = serde_json::from_value(serde_json::json!({"format": "h264", "embedCaptions": true})).unwrap();
    assert!(j.embed_captions);
    let j: ExportSettings = serde_json::from_value(serde_json::json!({"format": "h264"})).unwrap();
    assert!(!j.embed_captions);
}

/// Control codes, overlapping, zero-length and out-of-range captions, a caption longer than 608
/// can show, and a disabled track: the export completes and only printable text comes back.
#[test]
fn hostile_captions_never_stop_an_export() {
    let rate = FrameRate::FPS_30;
    let long = "x".repeat(5000);
    let ctl: String = (0u8..32).map(char::from).chain("\u{94}\u{2c}漢🎬".chars()).collect();
    let caps = [(0, 1, ctl.as_str()), (5, 5, "zero"), (10, 20, "a"), (12, 18, "overlap"), (20, 90, long.as_str()), (85, 10_000, "past the end")];
    let (p, seq, m) = captioned(rate, 90, &caps);
    let range = Some(TimeRange::new(Tick::ZERO, rate.tick_of(90)));
    let bytes = export_to(&p, seq, &m, "hostile.mp4", ExportSettings { format: Format::H264, embed_captions: true, range, ..Default::default() });
    let frames = cc_frames(&samples(&bytes).0);
    assert_eq!(frames.len(), 90);
    for stream in [Stream::Cea608, Stream::Cea708] {
        for c in embed::decode(&frames, rate, stream) {
            assert!(!c.text.chars().any(|ch| ch.is_control() && ch != '\n'), "{stream:?}: {:?}", c.text);
            assert!(c.text.lines().count() <= 15 && c.text.lines().all(|l| l.chars().count() <= 42), "{stream:?}: {:?}", c.text);
        }
    }
    // a disabled caption track is not embedded
    let mut q = (*p).clone();
    q.sequence_mut(seq).unwrap().caption_tracks[0].enabled = false;
    let bytes = export_to(&Arc::new(q), seq, &m, "disabled.mp4", ExportSettings { format: Format::H264, embed_captions: true, range, ..Default::default() });
    assert!(cc_frames(&samples(&bytes).0).iter().all(|(_, t)| t.is_empty()));
}

/// The caption wrapper answers for the encoder it wraps: a hardware export with embedded captions
/// is reported as the hardware encoder's (not as "software", with a fallback warning), and a
/// software one as software.
#[test]
fn the_caption_wrapper_reports_the_encoder_it_wraps() {
    struct Enc(Option<&'static str>);
    impl VideoEncoder for Enc {
        fn sample_entry(&self) -> SampleEntry {
            SampleEntry::avc(filmcraft_isobmff::AvcConfig::default(), 0, 0)
        }
        fn timescale(&self) -> u32 {
            30
        }
        fn encode(&mut self, _frame: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
            Ok(Vec::new())
        }
        fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
            Ok(Vec::new())
        }
        fn hardware(&self) -> Option<&'static str> {
            self.0
        }
    }
    let rate = FrameRate::FPS_30;
    let (p, seq, _) = captioned(rate, 30, &[(0, 20, "HELLO")]);
    let settings = ExportSettings { format: Format::H264, embed_captions: true, ..Default::default() };
    let schedule = crate::captions_out::schedule(&p, seq, &settings, TimeRange::new(Tick::ZERO, rate.tick_of(30)), rate, 0);
    assert!(schedule.is_some(), "the sequence has captions to embed");
    for hardware in [Some("VideoToolbox"), None] {
        let wrapped = crate::captions_out::CaptionSei::wrap(Box::new(Enc(hardware)), schedule.as_ref(), Format::H264, rate);
        assert_eq!(wrapped.hardware(), hardware);
    }
}

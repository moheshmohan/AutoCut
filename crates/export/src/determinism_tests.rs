//! An export depends only on the project and the settings: the same file on every machine, however
//! many cores render it and however the work is cut into steps.

use super::*;
use crate::tests::project;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("fc-export-det-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().to_string()
}

/// Export on a machine with `threads` cores: `export` renders one frame per core of the pool it
/// runs in.
fn export_with_cores(threads: usize, settings: &ExportSettings) -> Vec<u8> {
    let (p, seq, m) = project();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| export(&p, seq, settings, &m, &Progress::default())).unwrap();
    std::fs::read(&settings.path).unwrap()
}

/// Export a step at a time with `batch` frames per step (the web app uses 1).
fn export_stepped(batch: i64, settings: &ExportSettings) -> Vec<u8> {
    let (p, seq, m) = project();
    let prog = Progress::default();
    let mut ex = Exporter::new(p, seq, settings, &prog).unwrap();
    ex.set_batch(batch);
    while !matches!(ex.step(&m, &prog).unwrap(), Step::Done(_)) {}
    std::fs::read(&settings.path).unwrap()
}

#[test]
fn same_file_whatever_the_core_count() {
    // 24 frames: one full interleave group and a partial one
    for (format, ext) in [(Format::ProRes, "mov"), (Format::H264, "mp4"), (Format::Mjpeg, "mov")] {
        let settings = |name: &str| ExportSettings { format, path: tmp(&format!("{name}.{ext}")), ..Default::default() };
        let reference = export_with_cores(2, &settings(&format!("{}-2", format.id())));
        for threads in [3, 5, 16] {
            let other = export_with_cores(threads, &settings(&format!("{}-{threads}", format.id())));
            assert!(reference == other, "{}: {threads} cores give a different file than 2 cores", format.label());
        }
        for batch in [1, 7] {
            let other = export_stepped(batch, &settings(&format!("{}-b{batch}", format.id())));
            assert!(reference == other, "{}: {batch} frame(s) per step give a different file", format.label());
        }
    }
}

#[test]
fn an_export_adds_up_the_wall_time_of_its_stages() {
    let before: Vec<(Stage, u64)> = stage_times();
    let s = ExportSettings { format: Format::H264, path: tmp("stages.mp4"), ..Default::default() };
    let (p, seq, m) = project();
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let after = stage_times();
    let grew = |stage: Stage| {
        let at = |v: &[(Stage, u64)]| v.iter().find(|(s, _)| *s == stage).map_or(0, |(_, ns)| *ns);
        at(&after) > at(&before)
    };
    // other tests export in parallel, so the counters can only be checked for growth
    for stage in [Stage::Setup, Stage::Render, Stage::Encode, Stage::Convert, Stage::Audio, Stage::Mux, Stage::Finish] {
        assert!(grew(stage), "{} did not grow", stage.name());
    }
    assert_eq!(Stage::ALL.len(), after.len(), "one counter per stage");
}

/// Slice NAL units (types 1 and 5) in a length-prefixed (4-byte) H.264 sample.
fn slices_in(sample: &[u8]) -> usize {
    let (mut n, mut i) = (0, 0);
    while i + 4 <= sample.len() {
        let len = u32::from_be_bytes([sample[i], sample[i + 1], sample[i + 2], sample[i + 3]]) as usize;
        if let Some(&h) = sample.get(i + 4)
            && matches!(h & 0x1f, 1 | 5)
        {
            n += 1;
        }
        i += 4 + len;
    }
    n
}

#[test]
fn h264_slices_follow_the_frame_size_not_the_cores() {
    // 4096 lines = 256 macroblock rows: one slice per four rows on every machine, also on machines
    // with fewer than 64 cores (the encoder's default caps the slices at the core count)
    let path = tmp("tall.mp4");
    let s = ExportSettings { format: Format::H264, path: path.clone(), frame_size: Some((64, 4096)), include_audio: false, ..Default::default() };
    let (p, seq, m) = project();
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = filmcraft_isobmff::open(bytes.as_slice()).unwrap();
    let vt = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
    assert_eq!(file.tracks[vt].samples.len(), 24);
    for i in 0..file.tracks[vt].samples.len() {
        let sample = file.read_sample(bytes.as_slice(), vt, i).unwrap();
        assert_eq!(slices_in(&sample), 64, "picture {i}");
    }
}

// ---------------------------------------------------------------------------------------------
// rendering the next batch while the current one is encoded (job.rs, "Overlapping render and encode")
// ---------------------------------------------------------------------------------------------

use filmcraft_media::SharedSource;
use filmcraft_render::SourceMap;
use std::sync::atomic::AtomicUsize;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
}

/// Step an export on a pool of `threads` cores with `batch` frames per step, rendering ahead or
/// not; returns the file and how many steps rendered ahead.
fn export_overlapped(threads: usize, batch: i64, overlap: bool, settings: &ExportSettings) -> (Vec<u8>, u64) {
    let (p, seq, m) = project();
    pool(threads).install(|| {
        let prog = Progress::default();
        let mut ex = Exporter::new(p, seq, settings, &prog).unwrap();
        ex.set_batch(batch);
        ex.set_overlap(overlap);
        while !matches!(ex.step(&m, &prog).unwrap(), Step::Done(_)) {}
        assert_eq!(prog.done.load(Ordering::Relaxed), prog.total.load(Ordering::Relaxed), "every frame is counted once");
        assert_eq!(ex.frames_ahead(), 0, "nothing is left rendered at the end");
        (std::fs::read(&settings.path).unwrap(), ex.overlapped_steps())
    })
}

#[test]
fn rendering_ahead_does_not_change_the_file() {
    for (format, ext) in [(Format::ProRes, "mov"), (Format::H264, "mp4"), (Format::Mjpeg, "mov")] {
        let settings = |name: &str| ExportSettings { format, path: tmp(&format!("{name}.{ext}")), ..Default::default() };
        let (reference, ahead) = export_overlapped(3, 7, false, &settings(&format!("ov-{}-off", format.id())));
        assert_eq!(ahead, 0, "{}: overlap is off", format.label());
        for (threads, batch) in [(2, 1), (3, 3), (5, 7), (16, 24)] {
            let (other, ahead) = export_overlapped(threads, batch, true, &settings(&format!("ov-{}-{threads}-{batch}", format.id())));
            // 24 frames in one batch (the last case) has nothing to render ahead; the others do
            assert_eq!(ahead > 0, batch < 24, "{}: {ahead} overlapped steps at batch {batch}", format.label());
            assert!(reference == other, "{}: rendering ahead ({threads} cores, {batch} per step) gives a different file", format.label());
        }
    }
}

#[test]
fn rendering_ahead_in_a_two_pass_export() {
    let settings = |name: &str| ExportSettings {
        format: Format::H264,
        path: tmp(name),
        bitrate_mode: BitrateMode::Vbr2Pass,
        bitrate_kbps: 2000,
        include_audio: false,
        ..Default::default()
    };
    let (reference, _) = export_overlapped(3, 5, false, &settings("2pass-off.mp4"));
    let (other, ahead) = export_overlapped(3, 5, true, &settings("2pass-on.mp4"));
    assert!(ahead > 0);
    assert!(reference == other, "two-pass: rendering ahead gives a different file");
    // the whole export, with audio, through `export`: both passes counted once
    let prog = Progress::default();
    let (p, seq, m) = project();
    pool(4).install(|| export(&p, seq, &ExportSettings { include_audio: true, ..settings("2pass-all.mp4") }, &m, &prog)).unwrap();
    assert_eq!(prog.total.load(Ordering::Relaxed), 48);
    assert_eq!(prog.done.load(Ordering::Relaxed), 48);
}

#[test]
fn hdr_with_audio_is_the_same_file_when_the_memory_budget_caps_the_batch() {
    use filmcraft_color::{ColorPipeline, WorkingSpace};
    for (format, ext) in [(Format::ProRes, "mov"), (Format::H264, "mp4")] {
        let run = |name: &str, batch: i64, overlap: bool, budget: Option<u64>| {
            let (p, seq, m) = project();
            let mut p = (*p).clone();
            p.sequence_mut(seq).unwrap().settings.color = ColorPipeline { working: WorkingSpace::Rec2100Pq, ..ColorPipeline::REC709 };
            let p = Arc::new(p);
            let s = ExportSettings { format, path: tmp(&format!("{name}.{ext}")), ..Default::default() };
            pool(3).install(|| {
                let prog = Progress::default();
                let mut ex = Exporter::new(p, seq, &s, &prog).unwrap();
                ex.set_batch(batch);
                ex.set_overlap(overlap);
                if let Some(b) = budget {
                    ex.set_budget(b);
                }
                let mut first_ahead = None;
                loop {
                    let step = ex.step(&m, &prog).unwrap();
                    first_ahead.get_or_insert(ex.frames_ahead());
                    if matches!(step, Step::Done(_)) {
                        break;
                    }
                }
                (std::fs::read(&s.path).unwrap(), first_ahead.unwrap_or(0))
            })
        };
        let (reference, _) = run(&format!("hdrcap-{}-ref", format.id()), 16, false, None);
        // 320 x 180 HDR frames are 691 200 bytes: a budget of six of them lets two batches of three be in
        // flight, so a batch of 16 is cut to 3 and the interleave cut at frame 16 falls in a prefetched batch
        let budget = 6 * frame_bytes_for_test();
        let (capped, ahead) = run(&format!("hdrcap-{}-3", format.id()), 16, true, Some(budget));
        assert_eq!(ahead, 3, "{}: the batch was not capped to 3", format.label());
        assert!(reference == capped, "{}: a capped HDR batch gives a different file", format.label());
        let (one, ahead) = run(&format!("hdrcap-{}-1", format.id()), 16, true, Some(1));
        assert_eq!(ahead, 1);
        assert!(reference == one, "{}: one-frame HDR batches give a different file", format.label());
        let (full, ahead) = run(&format!("hdrcap-{}-16", format.id()), 16, true, None);
        assert_eq!(ahead, 8, "{}: 24 frames in batches of 16 leave 8 ahead", format.label());
        assert!(reference == full, "{}: an uncapped HDR batch gives a different file", format.label());
    }
}

/// Bytes of one HDR frame of the test project (320 x 180, three f32 per pixel).
fn frame_bytes_for_test() -> u64 {
    320 * 180 * 12
}

/// A provider that answers like an asynchronous (web) reader whose bytes have not arrived yet for
/// the calls numbered in `loading` (marks pending, offers no source), and can be told to panic.
struct Flaky<'a> {
    inner: &'a SourceMap,
    calls: AtomicUsize,
    loading: std::ops::Range<usize>,
    panic_from: AtomicUsize,
}

impl<'a> Flaky<'a> {
    fn new(inner: &'a SourceMap, loading: std::ops::Range<usize>) -> Self {
        Self { inner, calls: AtomicUsize::new(0), loading, panic_from: AtomicUsize::new(usize::MAX) }
    }
}

impl SourceProvider for Flaky<'_> {
    fn source(&self, item: ItemId) -> Option<SharedSource> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n >= self.panic_from.load(Ordering::SeqCst) {
            panic!("a source blew up (test)");
        }
        if self.loading.contains(&n) {
            filmcraft_media::pending::mark();
            return None;
        }
        self.inner.source(item)
    }
}

#[test]
fn a_batch_rendered_ahead_with_missing_media_is_never_encoded() {
    let settings = |name: &str| ExportSettings { format: Format::Mjpeg, path: tmp(name), include_audio: false, ..Default::default() };
    let (reference, _) = export_overlapped(3, 3, false, &settings("pending-ref.mov"));
    let (p, seq, m) = project();
    let flaky = Flaky::new(&m, 5..70);
    let s = settings("pending.mov");
    let pendings = pool(3).install(|| {
        let prog = Progress::default();
        let mut ex = Exporter::new(p, seq, &s, &prog).unwrap();
        ex.set_batch(3);
        let mut pendings = 0;
        for _ in 0..1000 {
            match ex.step(&flaky, &prog).unwrap() {
                Step::Done(_) => {
                    assert_eq!(prog.done.load(Ordering::Relaxed), 24);
                    return pendings;
                }
                Step::Pending => {
                    pendings += 1;
                    // the batch that was not complete is not kept for the next step
                    assert_eq!(ex.frames_ahead(), 0);
                }
                Step::Progress => {}
            }
        }
        panic!("the export did not finish");
    });
    assert!(pendings > 0, "the missing media was never noticed");
    // a frame drawn without its media would be black, not the red of the matte
    assert!(std::fs::read(&s.path).unwrap() == reference, "frames rendered without their media were encoded");
}

#[test]
fn cancelling_drops_the_batch_rendered_ahead() {
    let s = ExportSettings { format: Format::Mjpeg, path: tmp("cancel.mov"), ..Default::default() };
    let (p, seq, m) = project();
    pool(3).install(|| {
        let prog = Progress::default();
        let mut ex = Exporter::new(p, seq, &s, &prog).unwrap();
        ex.set_batch(4);
        assert!(matches!(ex.step(&m, &prog).unwrap(), Step::Progress));
        assert_eq!(ex.frames_ahead(), 4, "the second batch was rendered while the first was encoded");
        assert_eq!(prog.done.load(Ordering::Relaxed), 4);
        prog.cancel.store(true, Ordering::Relaxed);
        assert!(matches!(ex.step(&m, &prog), Err(ExportError::Cancelled)));
        assert_eq!(ex.frames_ahead(), 0);
        assert_eq!(prog.done.load(Ordering::Relaxed), 4, "nothing more was encoded");
    });
}

#[test]
fn a_panic_while_rendering_ahead_is_an_error() {
    let s = ExportSettings { format: Format::Mjpeg, path: tmp("panic.mov"), include_audio: false, ..Default::default() };
    let (p, seq, m) = project();
    let flaky = Flaky::new(&m, 0..0);
    pool(3).install(|| {
        let prog = Progress::default();
        let mut ex = Exporter::new(p, seq, &s, &prog).unwrap();
        ex.set_batch(4);
        assert!(matches!(ex.step(&flaky, &prog).unwrap(), Step::Progress));
        // the batch rendered ahead during the next step is the one that panics
        flaky.panic_from.store(flaky.calls.load(Ordering::SeqCst), Ordering::SeqCst);
        let r = ex.step(&flaky, &prog);
        assert!(matches!(r, Err(ExportError::Encode(_))), "{:?}", r.map(|_| ()));
    });
}

/// Remembers the name of every thread that asked for a source.
struct Watched<'a> {
    inner: &'a SourceMap,
    threads: std::sync::Mutex<std::collections::BTreeSet<String>>,
}

impl SourceProvider for Watched<'_> {
    fn source(&self, item: ItemId) -> Option<SharedSource> {
        let name = std::thread::current().name().unwrap_or("").to_string();
        self.threads.lock().unwrap().insert(name);
        self.inner.source(item)
    }
}

#[test]
fn frames_render_on_a_pool_of_the_exports_own() {
    // the encoding side's parallel work (colour conversion, audio mix) goes to the caller's pool;
    // frames rendered there too would keep its workers from it
    let settings = |name: &str| ExportSettings { format: Format::Mjpeg, path: tmp(name), include_audio: false, ..Default::default() };
    let (reference, _) = export_overlapped(3, 4, false, &settings("own-pool-ref.mov"));
    let (p, seq, m) = project();
    let watched = Watched { inner: &m, threads: Default::default() };
    let s = settings("own-pool.mov");
    pool(3).install(|| {
        let prog = Progress::default();
        let mut ex = Exporter::new(p, seq, &s, &prog).unwrap();
        ex.set_batch(4);
        while !matches!(ex.step(&watched, &prog).unwrap(), Step::Done(_)) {}
        assert!(ex.overlapped_steps() > 0);
    });
    let threads = watched.threads.into_inner().unwrap();
    assert!(!threads.is_empty());
    // (the thread decoding the batch after next asks for sources too, and renders nothing)
    assert!(threads.iter().all(|t| t.starts_with("filmcraft-export-render-") || t == "filmcraft-export-decode-ahead"), "frames rendered on {threads:?}");
    assert!(threads.iter().any(|t| t.starts_with("filmcraft-export-render-")), "frames rendered on {threads:?}");
    assert!(std::fs::read(&s.path).unwrap() == reference, "the render pool changed the file");
}

/// Decoding the batch after next while the next one renders (job.rs, "Decoding ahead") asks the
/// sources ahead of the render, and the file stays the same with it, without it, without overlap,
/// at any batch size and core count.
#[test]
fn decoding_ahead_does_not_change_the_file() {
    for (format, ext) in [(Format::ProRes, "mov"), (Format::H264, "mp4")] {
        let settings = |name: &str| ExportSettings { format, path: tmp(&format!("{name}.{ext}")), ..Default::default() };
        let (reference, _) = export_overlapped(3, 7, false, &settings(&format!("warm-{}-off", format.id())));
        for (threads, batch, warm) in [(2, 1, true), (3, 3, true), (3, 3, false), (5, 7, true), (16, 24, true)] {
            let (p, seq, m) = project();
            let watched = Watched { inner: &m, threads: Default::default() };
            let s = settings(&format!("warm-{}-{threads}-{batch}-{warm}", format.id()));
            pool(threads).install(|| {
                let prog = Progress::default();
                let mut ex = Exporter::new(p, seq, &s, &prog).unwrap();
                ex.set_batch(batch);
                ex.set_warm_ahead(warm);
                while !matches!(ex.step(&watched, &prog).unwrap(), Step::Done(_)) {}
                assert_eq!(prog.done.load(Ordering::Relaxed), prog.total.load(Ordering::Relaxed));
            });
            let names = watched.threads.into_inner().unwrap();
            // 24 frames: batches after the next one exist unless a batch holds (nearly) all of them
            let ahead = names.contains("filmcraft-export-decode-ahead");
            assert_eq!(ahead, warm && batch * 2 < 24, "{}: {threads} cores, {batch} per step: decoded ahead on {names:?}", format.label());
            assert!(
                std::fs::read(&s.path).unwrap() == reference,
                "{}: decoding ahead ({threads} cores, {batch} per step) gives a different file",
                format.label()
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// encoders that take Y'CbCr (VideoEncoder::takes_yuv420): frames converted on the render threads
// ---------------------------------------------------------------------------------------------

/// An encoder for exports whose path names it: each packet is a digest of the Y'CbCr planes it was
/// given (converted on the render threads with `takes`, by itself from the RGBA without), so two
/// files are the same exactly when every frame's planes are.
struct PlanesProbe {
    takes: bool,
    rate: FrameRate,
    planes: (Vec<u8>, Vec<u8>, Vec<u8>),
}

fn digest(parts: [&[u8]; 3]) -> Vec<u8> {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut out = Vec::new();
    for p in parts {
        for b in p {
            h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
        }
        out.extend_from_slice(&h.to_be_bytes());
        out.extend_from_slice(&(p.len() as u64).to_be_bytes());
    }
    out
}

impl VideoEncoder for PlanesProbe {
    fn sample_entry(&self) -> filmcraft_isobmff::SampleEntry {
        filmcraft_isobmff::SampleEntry::avc(filmcraft_isobmff::AvcConfig::new(vec![vec![0x67, 0x64, 0, 0x1f]], vec![vec![0x68, 0xee]], 4), 320, 180)
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        assert!(!self.takes, "the RGBA path is not used once the encoder takes Y'CbCr");
        let (y, u, v) = &mut self.planes;
        rgba_to_yuv420_8(f.rgba, f.width as usize, f.height as usize, y, u, v);
        Ok(vec![EncodedPacket { data: digest([y, u, v]), key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        Ok(Vec::new())
    }
    fn takes_yuv420(&self) -> bool {
        self.takes
    }
    fn encode_yuv420(&mut self, f: &EncoderFrame, y: &[u8], u: &[u8], v: &[u8]) -> Result<Vec<EncodedPacket>> {
        assert!(self.takes && f.rgba.is_empty());
        Ok(vec![EncodedPacket { data: digest([y, u, v]), key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
}

fn planes_probe(format: Format, _w: u32, _h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    let takes = match s.path.as_str() {
        p if format == Format::H264 && p.contains("planes-probe-render") => true,
        p if format == Format::H264 && p.contains("planes-probe-encode") => false,
        _ => return None,
    };
    Some(Ok(Box::new(PlanesProbe { takes, rate, planes: Default::default() })))
}

/// An encoder that takes Y'CbCr gets, from the render threads, the planes it would have made from
/// the RGBA itself: the same file with every core count, step size and overlap.
#[test]
fn frames_converted_on_the_render_threads_are_the_ones_the_encoder_converted() {
    register_encoder(planes_probe);
    let settings = |name: &str| ExportSettings { format: Format::H264, path: tmp(&format!("{name}.mp4")), ..Default::default() };
    let (reference, _) = export_overlapped(3, 5, false, &settings("planes-probe-encode-ref"));
    for (threads, batch, overlap) in [(2, 1, true), (3, 5, true), (3, 5, false), (5, 7, true), (16, 24, true)] {
        let name = format!("planes-probe-render-{threads}-{batch}-{overlap}");
        let (file, _) = export_overlapped(threads, batch, overlap, &settings(&name));
        assert!(file == reference, "{threads} cores, {batch} per step, overlap {overlap}: the planes differ");
    }
}

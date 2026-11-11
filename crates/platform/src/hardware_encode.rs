//! The VideoToolbox H.264 and HEVC encoders behind `filmcraft_export::VideoEncoder` (macOS; safe code
//! around [`crate::videotoolbox_encode`]).
//!
//! H.264 hardware encoding is opt-in per export (`ExportSettings::hardware_encoding`, `Auto`): its output
//! depends on the machine, while the built-in encoder's is byte-identical everywhere. The factory
//! declines, so the built-in encoder is used, for everything the hardware path does not take: other
//! formats, MXF (Annex B), two-pass VBR, HDR, odd picture sizes, non-square pixels, and any
//! configuration VideoToolbox cannot create a hardware session for. A hardware encoder that fails in
//! the middle of an export reports an error (the export stops); it cannot hand the stream over to
//! another encoder.
//!
//! HEVC (`Format::Hevc`) has no built-in encoder, so choosing the format is the opt-in, the same
//! configurations are declined, and a declined export fails with "encoder not available" instead of
//! falling back; [`hevc_available`] tells the format list whether this machine can do it at all.
//!
//! On a chip with two media engines, a 4K H.264 export is encoded by two real-time sessions at
//! once, each taking whole keyframe intervals in turn, joined into one stream ([`ParallelEncoder`],
//! [`lanes_for`], `filmcraft_export::EncodeSessions` switches it): 63 → 105 fps at 3840×2160 on M3 Max.

use std::sync::{Once, OnceLock};

use filmcraft_export::{
    BitrateMode, EncodeSessions, EncodedPacket, EncoderFrame, ExportError, ExportSettings, Format, H264Pass, H264Profile, HardwareEncoding, VideoEncoder,
};
use filmcraft_isobmff::{AvcConfig, HevcConfig, SampleEntry};
use filmcraft_time::FrameRate;

use crate::parallel_encode::{Joiner, LaneEncoder, LanePacket};
use crate::videotoolbox_encode::{ParamSets, VtCodec, VtConfig, VtEncoder, VtPacket, VtProfile, VtRate};

/// The hardware encoder's configuration for an export, or why it is not used for it.
pub fn config_for(w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Result<VtConfig, String> {
    let hevc = s.format == Format::Hevc;
    if !hevc && s.hardware_encoding != HardwareEncoding::Auto {
        return Err("hardware encoding is off".into());
    }
    if s.format.is_mxf() {
        return Err("MXF carries an Annex B stream".into());
    }
    if s.bitrate_mode == BitrateMode::Vbr2Pass || !matches!(s.h264_pass, H264Pass::Single) {
        return Err("two-pass encoding".into());
    }
    if s.bitrate_mode == BitrateMode::Crf {
        return Err("CRF (constant quality)".into());
    }
    if !hevc && s.h264_intra_only {
        return Err("all-intra H.264".into());
    }
    if s.signal.is_hdr() {
        return Err("HDR".into());
    }
    if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        return Err("odd picture size".into());
    }
    if s.pixel_aspect.is_some_and(|(n, d)| n != d) {
        return Err("non-square pixels".into());
    }
    let timescale = u32::try_from(rate.num).ok().filter(|n| *n > 0).ok_or("invalid frame rate")?;
    let frame_duration = u32::try_from(rate.den).ok().filter(|d| *d > 0).ok_or("invalid frame rate")?;
    let kbps = s.bitrate_kbps.max(100);
    let max_kbps = s.max_bitrate_kbps.filter(|m| *m >= kbps).unwrap_or(kbps.saturating_mul(3) / 2);
    let fps = f64::from(timescale) / f64::from(frame_duration);
    Ok(VtConfig {
        width: w,
        height: h,
        timescale,
        frame_duration,
        keyframe_interval: s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (fps * 2.0).round().max(1.0) as u32),
        profile: match (hevc, s.h264_profile) {
            (true, _) => VtProfile::HevcMain,
            (false, H264Profile::Baseline) => VtProfile::Baseline,
            (false, H264Profile::Main) => VtProfile::Main,
            (false, H264Profile::High) => VtProfile::High,
            // 10-bit H.264 is the built-in encoder's
            (false, H264Profile::High10) => return Err("High 10".into()),
        },
        rate: match s.bitrate_mode {
            BitrateMode::Cbr => VtRate::Cbr { kbps },
            _ => VtRate::Vbr { target_kbps: kbps, max_kbps },
        },
        realtime: false,
    })
}

static HEVC_PROBE: OnceLock<bool> = OnceLock::new();

/// Whether this machine has a hardware HEVC encoder (asked once, by creating a small session, which
/// takes about 0.1 s): what makes `Format::Hevc` available. [`warm_hevc_probe`] asks it ahead of the
/// first caller.
pub fn hevc_available() -> bool {
    *HEVC_PROBE.get_or_init(|| {
        let probe = VtConfig {
            width: 640,
            height: 360,
            timescale: 30,
            frame_duration: 1,
            keyframe_interval: 30,
            profile: VtProfile::HevcMain,
            rate: VtRate::Vbr { target_kbps: 2000, max_kbps: 3000 },
            realtime: false,
        };
        let result = VtEncoder::new(probe);
        if let Err(why) = &result {
            log::info!("no hardware HEVC encoder: {why}");
        }
        result.is_ok()
    })
}

/// Whether [`hevc_available`] has its answer yet (diagnostics and tests).
pub fn hevc_probed() -> bool {
    HEVC_PROBE.get().is_some()
}

/// Ask [`hevc_available`] now, on a thread of its own, so that the first draw of the format list does
/// not spend its 0.1 s creating a hardware session on the UI thread. Whoever asks while the thread
/// runs waits for the same answer (no longer than asking first); if the thread cannot start, or the
/// probe panics, the first caller asks, as before. Only the first call starts a thread.
pub fn warm_hevc_probe() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("filmcraft-hevc-probe".into()).spawn(|| {
            let _ = std::panic::catch_unwind(hevc_available);
        });
        if let Err(why) = spawned {
            log::info!("HEVC probe not started ahead of time: {why}");
        }
    });
}

/// Bytes of pictures a [`ParallelEncoder`] may hold queued for its sessions (`parallel_encode`,
/// "Memory"): 32 pictures of 3840×2160.
const QUEUE_BYTES: u64 = 384 << 20;

/// The most samples per second one media engine is asked to encode in real time. A real-time
/// session of a stream within it keeps to one engine, so two of them run side by side; above it
/// (3840×2160 at 60 fps), VideoToolbox spreads each session over both engines and two sessions
/// slow each other down (M3 Max: 4096×2160 at 30 fps scales, 3840×2160 at 60 fps does not).
const ENGINE_SAMPLE_RATE: f64 = 4096.0 * 2160.0 * 30.0;

/// Pictures larger than this (2560×1440) are worth two sessions: one encodes 1440p H.264 at about
/// 130 fps, faster than exports render it, while 3840×2160 tops out at 63 fps.
const MIN_PARALLEL_SAMPLES: u64 = 2560 * 1440;

/// Bytes of one 8-bit 4:2:0 picture.
fn picture_bytes(w: u32, h: u32) -> u64 {
    let (w, h) = (u64::from(w), u64::from(h));
    w.saturating_mul(h).saturating_add(w.div_ceil(2).saturating_mul(h.div_ceil(2)).saturating_mul(2))
}

/// Pictures a [`ParallelEncoder`] queues at most: half a segment and two (the overlap two
/// sessions need), within [`QUEUE_BYTES`]; at least two.
pub fn queue_budget(c: &VtConfig) -> usize {
    let half = u64::from(c.keyframe_interval / 2).saturating_add(2);
    let fit = QUEUE_BYTES / picture_bytes(c.width, c.height).max(1);
    usize::try_from(half.min(fit).max(2)).unwrap_or(2)
}

/// How many sessions encode an export (1: one session, the path as it was): two where it pays
/// with `EncodeSessions::Auto`, which is H.264 (with HEVC a second real-time session lowers the
/// quality of both, by 0.8 dB luma PSNR at the same bitrate on M3 Max), a chip with two media
/// engines, a picture larger than 1440p within what one engine encodes in real time, and a
/// keyframe interval the queue covers well enough for two sessions to beat one offline session.
/// `Two` asks for two wherever segments can be cut (a keyframe interval of at least two pictures).
pub fn lanes_for(c: &VtConfig, sessions: EncodeSessions, engines: usize) -> usize {
    let keyint = u64::from(c.keyframe_interval);
    if sessions == EncodeSessions::One || keyint < 2 {
        return 1;
    }
    if sessions == EncodeSessions::Two {
        return 2;
    }
    let samples = u64::from(c.width).saturating_mul(u64::from(c.height));
    let rate = samples as f64 * f64::from(c.timescale) / f64::from(c.frame_duration.max(1));
    // two real-time sessions (52 fps each at 4K) with the queue covering `budget` pictures of a
    // segment against one offline session (63 fps): ahead while the segment is at most 4× the queue
    let covered = keyint <= u64::try_from(queue_budget(c)).unwrap_or(0).saturating_mul(4);
    let pays = c.profile.codec() == VtCodec::H264 && engines >= 2 && samples > MIN_PARALLEL_SAMPLES && rate <= ENGINE_SAMPLE_RATE && covered;
    if pays { 2 } else { 1 }
}

/// The encoder factory `register` puts in front of the built-in ones: a hardware encoder when the
/// export asks for one (H.264 with `hardware_encoding`; HEVC always) and the OS takes the
/// configuration, `None` (the built-in encoder, if there is one) otherwise. Where two sessions pay
/// ([`lanes_for`]) it is a [`ParallelEncoder`].
pub fn videotoolbox_encoder_factory(
    format: Format,
    w: u32,
    h: u32,
    rate: FrameRate,
    s: &ExportSettings,
) -> Option<filmcraft_export::Result<Box<dyn VideoEncoder>>> {
    let name = match format {
        Format::H264 if s.hardware_encoding == HardwareEncoding::Auto => "H.264",
        Format::Hevc => "HEVC",
        _ => return None,
    };
    let config = match config_for(w, h, rate, s) {
        Ok(c) => c,
        Err(why) => {
            log::info!("hardware {name} encoding declined: {why}");
            filmcraft_export::note_hw_encode_declined();
            return None;
        }
    };
    let lanes = lanes_for(&config, filmcraft_export::hw_encode_sessions(), crate::videotoolbox_encode::media_engines());
    if lanes > 1 {
        match ParallelEncoder::new(config.clone(), lanes) {
            Ok(p) => {
                for _ in 0..lanes {
                    filmcraft_export::note_hw_encode_session();
                }
                filmcraft_export::note_hw_encode_parallel();
                log::info!("hardware {name} encoding in {lanes} sessions at once, segments of {} frames", config.keyframe_interval);
                return Some(Ok(Box::new(p)));
            }
            Err(why) => log::info!("hardware {name} encoding in one session: {why}"),
        }
    }
    match VtEncoder::new(config) {
        Ok(vt) => {
            filmcraft_export::note_hw_encode_session();
            Some(Ok(Box::new(HardwareEncoder { vt, y: Vec::new(), u: Vec::new(), v: Vec::new(), decoded: 0 })))
        }
        Err(why) => {
            log::info!("hardware {name} encoding declined: {why}");
            filmcraft_export::note_hw_encode_declined();
            None
        }
    }
}

/// The `hvcC` configuration of an HEVC stream: VideoToolbox's own record when it wrote one.
fn hevc_config(p: &ParamSets) -> Result<HevcConfig, String> {
    let record = p.hvcc.as_deref().ok_or("VideoToolbox wrote no hvcC record for the stream")?;
    let cfg = HevcConfig::parse(record).map_err(|e| format!("unreadable hvcC record: {e}"))?;
    if cfg.vps().is_empty() || cfg.sps().is_empty() || cfg.pps().is_empty() || cfg.length_size != 4 {
        return Err("the hvcC record lacks a parameter set or does not use 4-byte NAL lengths".into());
    }
    Ok(cfg)
}

/// The MP4 sample entry of a stream with these parameter sets (known after the first frame).
fn sample_entry_of(c: &VtConfig, params: Option<&ParamSets>) -> SampleEntry {
    let (w, h) = (c.width.min(u32::from(u16::MAX)) as u16, c.height.min(u32::from(u16::MAX)) as u16);
    match c.profile.codec() {
        VtCodec::H264 => {
            let (sps, pps) = params.map(|p| (p.sps.clone(), p.pps.clone())).unwrap_or_default();
            SampleEntry::avc(AvcConfig::new(sps, pps, 4), w, h)
        }
        // `encode` refuses to go on without a usable record, so a stream that got this far has one
        VtCodec::Hevc => SampleEntry::hevc(params.and_then(|p| hevc_config(p).ok()).unwrap_or_default(), w, h),
    }
}

/// The parameter sets the muxer needs, checked after the first picture.
fn check_params(c: &VtConfig, params: Option<&ParamSets>) -> filmcraft_export::Result<()> {
    let params = params.ok_or_else(|| ExportError::Encode("VideoToolbox returned no parameter sets for the first frame".into()))?;
    if c.profile.codec() == VtCodec::Hevc {
        hevc_config(params).map_err(ExportError::Encode)?;
    }
    Ok(())
}

/// The size of picture `f`, which must be SDR and the size the encoder was made for.
fn picture_size(c: &VtConfig, f: &EncoderFrame) -> filmcraft_export::Result<(usize, usize)> {
    if f.hdr.is_some() {
        return Err(ExportError::Unsupported("hardware encoding does not take HDR pictures".into()));
    }
    if (f.width, f.height) != (c.width, c.height) {
        return Err(ExportError::Encode(format!("picture is {}x{}, the encoder was created for {}x{}", f.width, f.height, c.width, c.height)));
    }
    Ok((f.width as usize, f.height as usize))
}

/// [`VtEncoder`] as a `VideoEncoder`: RGBA pictures in, MP4 samples out.
pub struct HardwareEncoder {
    vt: VtEncoder,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// Compressed frames returned so far.
    decoded: i64,
}

impl HardwareEncoder {
    /// MP4 samples from compressed frames. Without frame reordering every frame is presented at
    /// its decode time, so composition offsets are 0; a frame whose time is not `n × duration`
    /// after the first (a dropped or reordered frame) is an error, never a file with wrong times.
    fn samples(&mut self, packets: Vec<VtPacket>) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let duration = self.vt.config().frame_duration;
        let mut out = Vec::with_capacity(packets.len());
        for p in packets {
            let n = self.decoded;
            self.decoded += 1;
            if n.checked_mul(i64::from(duration)) != Some(p.pts) || p.pts != p.dts {
                return Err(ExportError::Encode(format!(
                    "VideoToolbox returned frame {n} at time {} (decode time {}), expected {}",
                    p.pts,
                    p.dts,
                    n.saturating_mul(i64::from(duration))
                )));
            }
            out.push(EncodedPacket { data: p.data, key: p.key, duration, composition_offset: 0 });
        }
        Ok(out)
    }

    /// The samples of picture `index`'s submission (`packets`: what came out meanwhile).
    fn encoded(&mut self, index: u64, packets: Vec<VtPacket>) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        filmcraft_export::note_hw_encode_frame();
        // the muxer needs the parameter sets after the first group of pictures
        if index == 0 {
            check_params(self.vt.config(), self.vt.parameter_sets().as_ref())?;
        }
        self.samples(packets)
    }
}

impl VideoEncoder for HardwareEncoder {
    fn sample_entry(&self) -> SampleEntry {
        sample_entry_of(self.vt.config(), self.vt.parameter_sets().as_ref())
    }

    fn timescale(&self) -> u32 {
        self.vt.config().timescale
    }

    fn encode(&mut self, f: &EncoderFrame) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let (w, h) = picture_size(self.vt.config(), f)?;
        if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Err(ExportError::Encode("the RGBA picture is smaller than its size".into()));
        }
        filmcraft_export::timed(filmcraft_export::Stage::Convert, || filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut self.y, &mut self.u, &mut self.v));
        let packets = self.vt.encode(f.index, &self.y, &self.u, &self.v).map_err(ExportError::Encode)?;
        self.encoded(f.index, packets)
    }

    /// The export converts the pictures on its render threads (`rgba_to_yuv420_8`, the conversion
    /// [`Self::encode`] runs), so this thread only fills and submits them.
    fn takes_yuv420(&self) -> bool {
        true
    }

    fn encode_yuv420(&mut self, f: &EncoderFrame, y: &[u8], u: &[u8], v: &[u8]) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        picture_size(self.vt.config(), f)?;
        let packets = self.vt.encode(f.index, y, u, v).map_err(ExportError::Encode)?;
        self.encoded(f.index, packets)
    }

    fn flush(&mut self) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let packets = self.vt.flush().map_err(ExportError::Encode)?;
        self.samples(packets)
    }
}

/// One session of a [`ParallelEncoder`], on its own timeline: picture `n` of the session at time
/// `n × duration`.
struct VtLane {
    vt: VtEncoder,
    /// Compressed frames returned so far.
    out: i64,
}

impl VtLane {
    /// The frames in order, each at its picture's time (a dropped or reordered frame is an error).
    fn checked(&mut self, packets: Vec<VtPacket>) -> Result<Vec<LanePacket>, String> {
        let duration = i64::from(self.vt.config().frame_duration);
        let mut out = Vec::with_capacity(packets.len());
        for p in packets {
            let expected = self.out.saturating_mul(duration);
            if p.pts != expected || p.dts != p.pts {
                return Err(format!("VideoToolbox returned frame {} of a session at time {} (decode time {}), expected {expected}", self.out, p.pts, p.dts));
            }
            self.out = self.out.saturating_add(1);
            out.push(LanePacket { data: p.data, key: p.key });
        }
        Ok(out)
    }
}

impl LaneEncoder for VtLane {
    type Params = ParamSets;

    fn encode(&mut self, local: u64, planes: &[u8], keyframe: bool) -> Result<Vec<LanePacket>, String> {
        let c = self.vt.config();
        let (w, h) = (c.width as usize, c.height as usize);
        let chroma = w.div_ceil(2).saturating_mul(h.div_ceil(2));
        let split = planes.split_at_checked(w.saturating_mul(h)).and_then(|(y, rest)| rest.split_at_checked(chroma).map(|(u, v)| (y, u, v)));
        let Some((y, u, v)) = split.filter(|(_, _, v)| v.len() == chroma) else {
            return Err(format!("a picture of {} bytes for a {w}x{h} encoder", planes.len()));
        };
        let packets = self.vt.encode_frame(local, y, u, v, keyframe)?;
        self.checked(packets)
    }

    fn flush(&mut self) -> Result<Vec<LanePacket>, String> {
        let packets = self.vt.flush()?;
        self.checked(packets)
    }

    fn params(&self) -> Option<ParamSets> {
        self.vt.parameter_sets()
    }
}

/// Several real-time [`VtEncoder`] sessions encoding one stream, a segment of one keyframe
/// interval each in turn, joined into one stream (`parallel_encode`): on a chip with two media
/// engines both work at once. The stream is what one session writes, with the same parameter
/// sets, keyframes, bitrate and timestamps; its frames come out up to about a segment later.
pub struct ParallelEncoder {
    joiner: Joiner<VtLane>,
    config: VtConfig,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl ParallelEncoder {
    /// `lanes` real-time sessions (1 to 4) for `config`, or why not.
    pub fn new(config: VtConfig, lanes: usize) -> Result<ParallelEncoder, String> {
        let config = VtConfig { realtime: true, ..config };
        let mut encoders = Vec::new();
        for _ in 0..lanes.clamp(1, 4) {
            encoders.push(VtLane { vt: VtEncoder::new(config.clone())?, out: 0 });
        }
        let joiner = Joiner::new(encoders, u64::from(config.keyframe_interval), queue_budget(&config))?;
        Ok(ParallelEncoder { joiner, config, y: Vec::new(), u: Vec::new(), v: Vec::new() })
    }

    /// Sessions taking pictures now (one is retired if it made other parameter sets than the first).
    pub fn lanes(&self) -> usize {
        self.joiner.active_lanes()
    }

    /// The configuration of its sessions.
    pub fn config(&self) -> &VtConfig {
        &self.config
    }

    fn samples(&self, packets: Vec<LanePacket>) -> Vec<EncodedPacket> {
        let duration = self.config.frame_duration;
        packets.into_iter().map(|p| EncodedPacket { data: p.data, key: p.key, duration, composition_offset: 0 }).collect()
    }

    fn push(&mut self, index: u64, y: &[u8], u: &[u8], v: &[u8]) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let packets = self.joiner.push(index, y, u, v).map_err(ExportError::Encode)?;
        filmcraft_export::note_hw_encode_frame();
        // the muxer needs the parameter sets after the first group of pictures
        if index == 0 {
            check_params(&self.config, self.joiner.params())?;
        }
        Ok(self.samples(packets))
    }
}

impl VideoEncoder for ParallelEncoder {
    fn sample_entry(&self) -> SampleEntry {
        sample_entry_of(&self.config, self.joiner.params())
    }

    fn timescale(&self) -> u32 {
        self.config.timescale
    }

    fn encode(&mut self, f: &EncoderFrame) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let (w, h) = picture_size(&self.config, f)?;
        if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Err(ExportError::Encode("the RGBA picture is smaller than its size".into()));
        }
        let (mut y, mut u, mut v) = (std::mem::take(&mut self.y), std::mem::take(&mut self.u), std::mem::take(&mut self.v));
        filmcraft_export::timed(filmcraft_export::Stage::Convert, || filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut y, &mut u, &mut v));
        let r = self.push(f.index, &y, &u, &v);
        (self.y, self.u, self.v) = (y, u, v);
        r
    }

    /// As [`HardwareEncoder`]: the export converts the pictures on its render threads.
    fn takes_yuv420(&self) -> bool {
        true
    }

    fn encode_yuv420(&mut self, f: &EncoderFrame, y: &[u8], u: &[u8], v: &[u8]) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let (w, h) = picture_size(&self.config, f)?;
        let chroma = w.div_ceil(2).saturating_mul(h.div_ceil(2));
        if y.len() != w.saturating_mul(h) || u.len() != chroma || v.len() != chroma {
            return Err(ExportError::Encode("the Y'CbCr planes do not match the picture size".into()));
        }
        self.push(f.index, y, u, v)
    }

    fn flush(&mut self) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let packets = self.joiner.finish().map_err(ExportError::Encode)?;
        Ok(self.samples(packets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(w: u32, h: u32, fps: u32, keyint: u32, profile: VtProfile) -> VtConfig {
        VtConfig {
            width: w,
            height: h,
            timescale: fps,
            frame_duration: 1,
            keyframe_interval: keyint,
            profile,
            rate: VtRate::Vbr { target_kbps: 45_000, max_kbps: 54_000 },
            realtime: false,
        }
    }

    #[test]
    fn two_sessions_where_they_pay() {
        use EncodeSessions::{Auto, One, Two};
        let uhd = config(3840, 2160, 30, 60, VtProfile::High);
        assert_eq!(lanes_for(&uhd, Auto, 2), 2, "4K30 H.264 on a chip with two engines");
        assert_eq!(lanes_for(&config(4096, 2160, 30, 60, VtProfile::High), Auto, 2), 2);
        assert_eq!(lanes_for(&config(3840, 2160, 24, 48, VtProfile::Main), Auto, 2), 2);
        assert_eq!(lanes_for(&uhd, Auto, 1), 1, "one engine");
        assert_eq!(lanes_for(&uhd, One, 2), 1, "switched off");
        assert_eq!(lanes_for(&config(3840, 2160, 30, 60, VtProfile::HevcMain), Auto, 2), 1, "HEVC loses quality");
        assert_eq!(lanes_for(&config(3840, 2160, 60, 120, VtProfile::High), Auto, 2), 1, "4K60 needs both engines per session");
        assert_eq!(lanes_for(&config(2560, 1440, 30, 60, VtProfile::High), Auto, 2), 1, "1440p: one session is fast enough");
        assert_eq!(lanes_for(&config(1920, 1080, 30, 60, VtProfile::High), Auto, 2), 1);
        assert_eq!(lanes_for(&config(3840, 2160, 30, 300, VtProfile::High), Auto, 2), 1, "a GOP far longer than the queue");
        assert_eq!(lanes_for(&config(3840, 2160, 30, 1, VtProfile::High), Auto, 2), 1, "every picture a keyframe");
        // asked for: wherever segments can be cut
        assert_eq!(lanes_for(&config(640, 360, 30, 30, VtProfile::HevcMain), Two, 1), 2);
        assert_eq!(lanes_for(&config(3840, 2160, 30, 1, VtProfile::High), Two, 2), 1);
        // hostile numbers do not panic
        for c in [
            config(0, 0, 0, 0, VtProfile::High),
            config(u32::MAX, u32::MAX, u32::MAX, u32::MAX, VtProfile::High),
            VtConfig { frame_duration: 0, ..uhd.clone() },
        ] {
            for s in [Auto, One, Two] {
                let n = lanes_for(&c, s, usize::MAX);
                assert!((1..=2).contains(&n));
                assert!(queue_budget(&c) >= 2);
            }
        }
    }

    #[test]
    fn the_queue_fits_its_budget() {
        // 4K: half a 60-frame segment and two, 397 MB
        let uhd = config(3840, 2160, 30, 60, VtProfile::High);
        assert_eq!(queue_budget(&uhd), 32);
        assert!(queue_budget(&uhd) as u64 * picture_bytes(3840, 2160) <= QUEUE_BYTES);
        // a long GOP is capped by the bytes
        assert_eq!(queue_budget(&config(3840, 2160, 30, 600, VtProfile::High)), (QUEUE_BYTES / picture_bytes(3840, 2160)) as usize);
        assert_eq!(queue_budget(&config(3840, 2160, 30, 2, VtProfile::High)), 3);
        assert_eq!(picture_bytes(3, 3), 9 + 2 * 4);
        assert_eq!(picture_bytes(u32::MAX, u32::MAX), u64::MAX);
    }

    #[test]
    fn engines_follow_the_chip() {
        use crate::videotoolbox_encode::engines_of;
        for (brand, n) in [
            ("Apple M3 Max", 2),
            ("Apple M1 Ultra", 2),
            ("Apple M4 Max", 2),
            ("Apple M3 Pro", 1),
            ("Apple M2", 1),
            ("Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz", 1),
            ("", 1),
            ("Maximus Ultra 9000", 1),
        ] {
            assert_eq!(engines_of(brand), n, "{brand}");
        }
    }
}

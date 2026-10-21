//! Export ▸ Captions ▸ Embed in Output File: the sequence's caption track carried in the H.264 /
//! H.265 stream as CEA-608 (CC1) and CEA-708 (service 1) closed captions, ATSC A/53 `cc_data` in
//! a `user_data_registered_itu_t_t35` SEI message in every access unit ([`filmcraft_captions::embed`]).
//!
//! [`CaptionSei`] wraps any video encoder of the format (the built-in H.264 encoder, a hardware
//! H.264 or H.265 encoder): each packet it returns (in decoding order) gets the caption data of
//! the frame it shows, found from its decoding time and composition offset the way the
//! multiplexer lays them out, inserted before its first slice. A packet whose NAL units do not
//! parse is written unchanged.

use std::sync::Arc;

use filmcraft_captions::embed::{Codec, Schedule, Streams};
use filmcraft_isobmff::SampleEntry;
use filmcraft_project::{ItemId, Project};
use filmcraft_time::{FrameRate, TimeRange};

use crate::{EncodedPacket, EncoderFrame, ExportSettings, Format, Result, VideoEncoder};

/// Length of the NAL unit length fields in MP4 / QuickTime samples written by FilmCraft's
/// encoders (and the platforms' hardware encoders).
const NAL_LENGTH_SIZE: usize = 4;

/// The caption schedule of an export, when the settings ask for embedded captions and the
/// sequence has an enabled caption track with captions in the range; `None` otherwise. Times
/// count from the first exported frame (`f0` at `rate`).
pub(crate) fn schedule(project: &Project, seq: ItemId, settings: &ExportSettings, range: TimeRange, rate: FrameRate, f0: i64) -> Option<Arc<Schedule>> {
    if !settings.embed_captions || !settings.format.is_h26x() {
        return None;
    }
    let q = project.sequence(seq)?;
    let track = q.caption_tracks.iter().find(|t| t.enabled)?;
    let start = rate.tick_of(f0);
    let span = range.end() - start;
    let mut doc = filmcraft_captions::document_from_track(track, start);
    doc.cues.retain_mut(|c| {
        c.end = c.end.min(span);
        c.start < c.end
    });
    let s = Schedule::new(&doc, rate, Streams::default());
    (!s.is_empty()).then(|| Arc::new(s))
}

/// A video encoder whose packets carry the caption SEI of the frame they show.
pub(crate) struct CaptionSei {
    inner: Box<dyn VideoEncoder>,
    schedule: Arc<Schedule>,
    codec: Codec,
    /// Decoding time of the next packet (encoder timescale), as the multiplexer counts it.
    dts: i64,
    /// One frame in the encoder timescale (0: unknown, use the decoding order).
    frame: i64,
    /// Packets so far.
    count: i64,
}

impl CaptionSei {
    pub(crate) fn wrap(inner: Box<dyn VideoEncoder>, schedule: Option<&Arc<Schedule>>, format: Format, rate: FrameRate) -> Box<dyn VideoEncoder> {
        let Some(schedule) = schedule else { return inner };
        let codec = match format {
            Format::H264 => Codec::H264,
            Format::Hevc => Codec::Hevc,
            _ => return inner,
        };
        let ts = i128::from(inner.timescale());
        let (num, den) = (i128::from(rate.num), i128::from(rate.den));
        let frame = if num > 0 && (ts * den) % num == 0 { i64::try_from(ts * den / num).unwrap_or(0) } else { 0 };
        Box::new(CaptionSei { inner, schedule: schedule.clone(), codec, dts: 0, frame, count: 0 })
    }

    fn embed(&mut self, packets: Vec<EncodedPacket>) -> Vec<EncodedPacket> {
        let start = self.inner.media_start().unwrap_or(0);
        packets
            .into_iter()
            .map(|mut p| {
                let shown = self.dts.saturating_add(i64::from(p.composition_offset)).saturating_sub(start);
                let index = if self.frame > 0 && shown >= 0 { shown / self.frame } else { self.count };
                self.dts = self.dts.saturating_add(i64::from(p.duration));
                self.count = self.count.saturating_add(1);
                let sei = self.schedule.sei_nal(index, self.codec);
                if let Some(data) = filmcraft_captions::embed::insert_sei(&p.data, &sei, self.codec, NAL_LENGTH_SIZE) {
                    p.data = data;
                }
                p
            })
            .collect()
    }
}

impl VideoEncoder for CaptionSei {
    fn sample_entry(&self) -> SampleEntry {
        self.inner.sample_entry()
    }
    fn timescale(&self) -> u32 {
        self.inner.timescale()
    }
    fn encode(&mut self, frame: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        let p = self.inner.encode(frame)?;
        Ok(self.embed(p))
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        let p = self.inner.flush()?;
        Ok(self.embed(p))
    }
    fn media_start(&self) -> Option<i64> {
        self.inner.media_start()
    }
    fn pass_stats(&self) -> Option<filmcraft_h264enc::PassStats> {
        self.inner.pass_stats()
    }
    fn takes_yuv420(&self) -> bool {
        self.inner.takes_yuv420()
    }
    fn encode_yuv420(&mut self, frame: &EncoderFrame, y: &[u8], u: &[u8], v: &[u8]) -> Result<Vec<EncodedPacket>> {
        let p = self.inner.encode_yuv420(frame, y, u, v)?;
        Ok(self.embed(p))
    }
}

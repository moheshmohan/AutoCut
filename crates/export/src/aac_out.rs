//! AAC Audio: the sequence mix alone, AAC-LC from the same encoder as the MPEG-4 movies, in an
//! MPEG-4 audio file (`.m4a`, the encoder delay signalled by an edit list) or, when the path ends
//! in `.aac`, a raw ADTS stream (ISO/IEC 14496-3 §1.A.2; one header per access unit, no CRC). ADTS
//! cannot signal the encoder delay, so a player starts it with the priming samples.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use filmcraft_isobmff::{Brand, CodecConfig, Mp4Writer, TrackConfig, WriteSample, WriterOptions};
use filmcraft_project::{ItemId, Project};
use filmcraft_render::SourceProvider;
use filmcraft_time::TimeRange;

use crate::audio_out::AudioOut;
use crate::{AudioEncoder, ExportError, ExportSettings, Format, Out, Progress, Result, audio_factories};

/// Whether `path` asks for a raw ADTS stream rather than an MPEG-4 file.
pub(crate) fn is_adts_path(path: &str) -> bool {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("aac"))
}

/// Where the encoded access units go.
enum Sink {
    Mp4 { mux: Mp4Writer<Out>, track: usize, frame: u32 },
    Adts { out: Out, header: AdtsHeader },
}

impl Sink {
    fn write(&mut self, units: Vec<Vec<u8>>) -> Result<()> {
        let io = |e: std::io::Error| ExportError::Io(e.to_string());
        match self {
            Sink::Mp4 { mux, track, frame } => {
                for au in units {
                    mux.write_sample(*track, WriteSample { data: &au, duration: *frame, composition_offset: 0, is_sync: true })
                        .map_err(|e| ExportError::Io(e.to_string()))?;
                }
            }
            Sink::Adts { out, header } => {
                for au in units {
                    out.write_all(&header.bytes(au.len())?).map_err(io)?;
                    out.write_all(&au).map_err(io)?;
                }
            }
        }
        Ok(())
    }

    fn finish(self, settings: &ExportSettings) -> Result<u64> {
        match self {
            Sink::Mp4 { mux, .. } => mux.finish().map_err(|e| ExportError::Io(e.to_string()))?.finish(settings),
            Sink::Adts { out, .. } => out.finish(settings),
        }
    }
}

/// The fixed fields of an ADTS header, from the AudioSpecificConfig.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdtsHeader {
    /// `profile_ObjectType`: the audio object type minus one (1 = AAC-LC).
    profile: u8,
    sf_index: u8,
    channel_config: u8,
}

impl AdtsHeader {
    /// Read the object type, sampling frequency index and channel configuration from an
    /// AudioSpecificConfig; ADTS can carry only object types 1–4 with an indexed rate and a
    /// channel configuration of 1–7.
    pub(crate) fn from_asc(asc: &[u8]) -> Result<AdtsHeader> {
        let (Some(&a), Some(&b)) = (asc.first(), asc.get(1)) else {
            return Err(ExportError::Encode("AAC: the encoder gave no AudioSpecificConfig".into()));
        };
        let object_type = a >> 3;
        let sf_index = ((a & 0x07) << 1) | (b >> 7);
        let channel_config = (b >> 3) & 0x0F;
        if !(1..=4).contains(&object_type) || sf_index > 12 || !(1..=7).contains(&channel_config) {
            return Err(ExportError::Unsupported(format!(
                "an ADTS .aac stream cannot carry this AAC configuration (object type {object_type}, rate index {sf_index}, channels {channel_config}): export .m4a instead"
            )));
        }
        Ok(AdtsHeader { profile: object_type - 1, sf_index, channel_config })
    }

    /// The 7-byte header (`protection_absent`) of an access unit of `payload` bytes.
    pub(crate) fn bytes(&self, payload: usize) -> Result<[u8; 7]> {
        // aac_frame_length is 13 bits and counts the header
        let too_big = || ExportError::Encode(format!("AAC: a {payload}-byte frame does not fit ADTS"));
        let len = u32::try_from(payload).ok().and_then(|n| n.checked_add(7)).filter(|n| *n < 1 << 13).ok_or_else(too_big)?;
        Ok([
            0xFF,
            0xF1, // syncword, MPEG-4, layer 0, protection absent
            (self.profile << 6) | (self.sf_index << 2) | (self.channel_config >> 2),
            ((self.channel_config & 0x03) << 6) | ((len >> 11) & 0x03) as u8,
            ((len >> 3) & 0xFF) as u8,
            (((len & 0x07) << 5) as u8) | 0x1F, // buffer fullness 0x7FF: variable rate
            0xFC,                               // … and one raw data block
        ])
    }
}

/// Encode the mix of `range` to `settings.path`; returns (bytes written, samples per channel).
pub(crate) fn export(
    project: &Arc<Project>,
    seq: ItemId,
    settings: &ExportSettings,
    range: TimeRange,
    sources: &dyn SourceProvider,
    progress: &Progress,
) -> Result<(u64, u64)> {
    let cancelled = || progress.cancel.load(Ordering::Relaxed);
    let mut a = AudioOut::new(project.clone(), seq, settings, range)?;
    let mut enc: Box<dyn AudioEncoder> = audio_factories()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find_map(|f| f(Format::Aac, a.sr, a.channels as u32, settings))
        .ok_or_else(|| ExportError::Unsupported("AAC encoder not available".into()))??;
    let entry = enc.sample_entry();
    let frames = a.remaining();
    // one second per chunk: memory stays flat however long the range is
    let chunk = i64::from(a.sr.max(1));
    if !settings.part_of_batch {
        progress.total.store(frames.div_ceil(chunk as u64).max(1), Ordering::Relaxed);
        progress.set_status(format!("Exporting audio ({})", settings.format.label()));
    }
    a.measure(settings, sources, &cancelled)?;
    *progress.loudness.lock().unwrap_or_else(|e| e.into_inner()) = a.loudness;
    let mut sink = if is_adts_path(&settings.path) {
        let CodecConfig::Aac(cfg) = &entry.codec else {
            return Err(ExportError::Encode("AAC: the encoder's sample entry is not AAC".into()));
        };
        Sink::Adts { header: AdtsHeader::from_asc(&cfg.asc)?, out: Out::create(settings)? }
    } else {
        let mut opts = WriterOptions::new(Brand::Mp4);
        opts.metadata = settings.metadata.udta();
        let mut mux = Mp4Writer::new(Out::create(settings)?, opts).map_err(|e| ExportError::Io(e.to_string()))?;
        let mut track = TrackConfig::new(entry, a.sr);
        track.media_start = Some(i64::from(enc.priming()));
        let track = mux.add_track(track).map_err(|e| ExportError::Io(e.to_string()))?;
        Sink::Mp4 { mux, track, frame: enc.frame_size() }
    };
    while let Some(planar) = a.pull(a.out.saturating_add(chunk), sources) {
        if cancelled() {
            return Err(ExportError::Cancelled);
        }
        sink.write(enc.encode(&planar)?)?;
        if !settings.part_of_batch {
            progress.done.fetch_add(1, Ordering::Relaxed);
        }
    }
    sink.write(enc.flush()?)?;
    let n = sink.finish(settings)?;
    if settings.part_of_batch {
        progress.done.store(1, Ordering::Relaxed);
    }
    Ok((n, frames))
}

/// Most channels [`encode_m4a`] takes.
const MAX_FILE_CHANNELS: usize = 8;

/// Encode planar samples (`channels`, each at `sample_rate`) as AAC-LC at `bitrate_kbps` into an
/// MPEG-4 audio file at `path` (`.m4a`, the encoder delay signalled by an edit list): a sound on
/// its own to share or listen to (the voice lab's auditions). Returns the bytes written.
pub fn encode_m4a(path: &str, channels: &[Vec<f32>], sample_rate: u32, bitrate_kbps: u32) -> Result<u64> {
    if channels.is_empty() || channels.len() > MAX_FILE_CHANNELS {
        return Err(ExportError::Unsupported(format!("AAC audio files take 1–{MAX_FILE_CHANNELS} channels, not {}", channels.len())));
    }
    if sample_rate == 0 {
        return Err(ExportError::Unsupported("audio without a sample rate".into()));
    }
    let mut settings = ExportSettings { format: Format::Aac, path: path.to_string(), ..Default::default() };
    settings.audio.bitrate_kbps = bitrate_kbps;
    let mut enc: Box<dyn AudioEncoder> = audio_factories()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find_map(|f| f(Format::Aac, sample_rate, channels.len() as u32, &settings))
        .ok_or_else(|| ExportError::Unsupported("AAC encoder not available".into()))??;
    let entry = enc.sample_entry();
    let mut mux = Mp4Writer::new(Out::create(&settings)?, WriterOptions::new(Brand::Mp4)).map_err(|e| ExportError::Io(e.to_string()))?;
    let mut track = TrackConfig::new(entry, sample_rate);
    track.media_start = Some(i64::from(enc.priming()));
    let track = mux.add_track(track).map_err(|e| ExportError::Io(e.to_string()))?;
    let mut sink = Sink::Mp4 { mux, track, frame: enc.frame_size() };
    let n = channels.iter().map(Vec::len).min().unwrap_or(0);
    let chunk = (sample_rate as usize).max(1);
    let mut at = 0;
    while at < n {
        let end = (at + chunk).min(n);
        let planar: Vec<Vec<f32>> =
            channels.iter().map(|c| c.get(at..end).unwrap_or(&[]).iter().map(|v| if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }).collect()).collect();
        sink.write(enc.encode(&planar)?)?;
        at = end;
    }
    sink.write(enc.flush()?)?;
    sink.finish(&settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sound_on_its_own_becomes_an_m4a_that_reads_back() {
        let dir = std::env::temp_dir().join(format!("filmcraft-m4a-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.m4a").to_string_lossy().into_owned();
        let rate = 24_000u32;
        let x: Vec<f32> = (0..rate * 2).map(|i| 0.3 * (std::f32::consts::TAU * 440.0 * i as f32 / rate as f32).sin()).collect();
        let bytes = encode_m4a(&path, std::slice::from_ref(&x), rate, 64).unwrap();
        assert!(bytes > 1000 && bytes < 64_000, "{bytes}");
        let src = filmcraft_codecs::open_bytes("tone.m4a", std::fs::read(&path).unwrap().into()).unwrap();
        let a = src.info().audio().cloned().unwrap();
        assert_eq!((a.sample_rate, a.channels), (rate, 1));
        let buf = src.audio(0, rate as usize, rate).unwrap();
        let rms = (buf.channels[0].iter().map(|v| v * v).sum::<f32>() / rate as f32).sqrt();
        assert!((rms - 0.212).abs() < 0.03, "{rms}");
        // hostile input: refused or encoded, never a panic
        assert!(encode_m4a(&path, &[], rate, 64).is_err());
        assert!(encode_m4a(&path, std::slice::from_ref(&x), 0, 64).is_err());
        assert!(encode_m4a(&path, &vec![vec![0.0; 10]; 9], rate, 64).is_err());
        assert!(encode_m4a(&path, &[vec![f32::NAN; 3000]], rate, 64).is_ok());
        assert!(encode_m4a(&path, &[vec![]], rate, 64).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn adts_header_fields_follow_the_asc() {
        // AAC-LC (2), 48 kHz (index 3), stereo: 00010 0011 0010 000
        let h = AdtsHeader::from_asc(&[0x11, 0x90]).unwrap();
        assert_eq!(h, AdtsHeader { profile: 1, sf_index: 3, channel_config: 2 });
        let b = h.bytes(371).unwrap();
        assert_eq!(&b[..2], &[0xFF, 0xF1]);
        assert_eq!(b[2] >> 6, 1);
        assert_eq!((b[2] >> 2) & 0x0F, 3);
        assert_eq!(((b[2] & 1) << 2) | (b[3] >> 6), 2);
        let len = (u32::from(b[3] & 0x03) << 11) | (u32::from(b[4]) << 3) | u32::from(b[5] >> 5);
        assert_eq!(len, 378);
        assert_eq!(((u32::from(b[5] & 0x1F)) << 6) | u32::from(b[6] >> 2), 0x7FF);
        assert_eq!(b[6] & 0x03, 0);
    }

    #[test]
    fn adts_refuses_what_it_cannot_signal() {
        assert!(AdtsHeader::from_asc(&[]).is_err());
        assert!(AdtsHeader::from_asc(&[0x11]).is_err());
        // explicit sampling frequency (index 15)
        assert!(AdtsHeader::from_asc(&[0x17, 0x90, 0, 0, 0]).is_err());
        // object type 5 (SBR) has no ADTS profile
        assert!(AdtsHeader::from_asc(&[0x29, 0x90]).is_err());
        let h = AdtsHeader::from_asc(&[0x11, 0x90]).unwrap();
        assert!(h.bytes(8184).is_ok());
        assert!(h.bytes(8185).is_err());
        assert!(h.bytes(usize::MAX).is_err());
    }

    #[test]
    fn only_a_dot_aac_path_is_adts() {
        assert!(is_adts_path("/x/mix.aac"));
        assert!(is_adts_path("mix.AAC"));
        assert!(!is_adts_path("mix.m4a"));
        assert!(!is_adts_path("aac"));
    }
}

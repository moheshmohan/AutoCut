//! Clean-room FLAC encoder (RFC 9639, Free Lossless Audio Codec).
//!
//! - [`Encoder`]: planar integer samples (8–32 bit, 1–8 channels) → a `.flac` stream: the
//!   [`Encoder::header`] (`fLaC`, STREAMINFO, a VORBIS_COMMENT naming the encoder, with
//!   [`Encoder::header_with_comments`] the speaker layout's [`channel_mask_comment`] when
//!   decoders would assume another), then frames of
//!   a fixed block size. Each channel of each frame is coded as constant, verbatim, a fixed
//!   polynomial predictor or linear prediction, whichever is smallest, with partitioned Rice codes;
//!   stereo frames also try left/side, side/right and mid/side.
//! - The STREAMINFO sizes and MD5 are only known at the end: write the header first, then overwrite
//!   [`STREAMINFO_OFFSET`] with [`Encoder::streaminfo`] once [`Encoder::finish`] has run (or keep
//!   the placeholder, whose zeros mean "unknown" and decode fine).
//!
//! Lossless at every compression level; the level (0–8) only trades encoding time for size.
//!
//! Layer L0: depends only on `filmcraft-bitstream` (+ `thiserror`); no `unsafe`; builds for
//! `wasm32-unknown-unknown`.
//!
//! ```
//! use filmcraft_flac::{Encoder, EncoderConfig, STREAMINFO_OFFSET};
//!
//! let left: Vec<i32> = (0..48_000).map(|i| ((i as f64 * 0.05).sin() * 20_000.0) as i32).collect();
//! let right = left.clone();
//! let mut enc = Encoder::new(EncoderConfig::new(48_000, 2, 16))?;
//! let mut file = enc.header();
//! file.extend(enc.encode(&[&left, &right])?);
//! file.extend(enc.finish());
//! let at = STREAMINFO_OFFSET as usize;
//! file[at..at + 34].copy_from_slice(&enc.streaminfo());
//! assert_eq!(&file[..4], b"fLaC");
//! assert!(file.len() < 48_000 * 2 * 2 / 2); // well under half of 16-bit PCM
//! # Ok::<(), filmcraft_flac::Error>(())
//! ```

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

mod crc;
mod frame;
mod md5;
mod predict;

use frame::{FrameSpec, Stereo};
use predict::{Effort, Subframe};

/// Byte offset of the 34-byte STREAMINFO body in [`Encoder::header`].
pub const STREAMINFO_OFFSET: u64 = 8;

/// The vendor string of the VORBIS_COMMENT block.
const VENDOR: &str = concat!("filmcraft-flac ", env!("CARGO_PKG_VERSION"));

/// The speaker layout, as a WAV channel mask (`dwChannelMask`), that decoders assume for a FLAC
/// stream of `channels` channels without a `WAVEFORMATEXTENSIBLE_CHANNEL_MASK` comment: the
/// channel assignment of RFC 9639 §9.1.3, with the surround pair of 5 and 6 channels read as side
/// surrounds, as libFLAC and ffmpeg do. 0 (no assignment) outside 1–8.
pub fn assumed_channel_mask(channels: u8) -> u32 {
    match channels {
        1 => 0x4,   // front centre
        2 => 0x3,   // front left, right
        3 => 0x7,   // + front centre
        4 => 0x33,  // front left, right, back left, right
        5 => 0x607, // front left, right, centre, side left, right
        6 => 0x60F, // + LFE
        7 => 0x70F, // front left, right, centre, LFE, back centre, side left, right
        8 => 0x63F, // front left, right, centre, LFE, back left, right, side left, right
        _ => 0,
    }
}

/// The `WAVEFORMATEXTENSIBLE_CHANNEL_MASK` comment (RFC 9639 §8.6.2) that records the speaker
/// layout `mask` (a WAV channel mask) of a `channels`-channel stream, when decoders would assume
/// another one ([`assumed_channel_mask`]); None when they read it right without one.
pub fn channel_mask_comment(channels: u8, mask: u32) -> Option<String> {
    (mask != assumed_channel_mask(channels)).then(|| format!("WAVEFORMATEXTENSIBLE_CHANNEL_MASK=0x{mask:04X}"))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("expected {expected} channels, got {got}")]
    Channels { expected: usize, got: usize },
    #[error("channels of unequal length")]
    UnequalChannels,
    #[error("sample {0} does not fit the configured sample size")]
    SampleRange(i32),
}

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// 1 to 1 048 575 Hz.
    pub sample_rate: u32,
    /// 1–8 (FLAC channel order: L R C LFE Ls Rs … for 5.1).
    pub channels: u8,
    /// 8–32. At 32 bits stereo channels are coded independently: the side channel of
    /// left/side, side/right and mid/side coding would need 33 bits.
    pub bits_per_sample: u8,
    /// Samples per channel per frame, 16–65535 (4096 by default).
    pub block_size: u16,
    /// 0 (fastest) to 8 (smallest); 5 by default.
    pub level: u8,
}

impl EncoderConfig {
    pub fn new(sample_rate: u32, channels: u8, bits_per_sample: u8) -> Self {
        EncoderConfig { sample_rate, channels, bits_per_sample, block_size: 4096, level: 5 }
    }
}

pub struct Encoder {
    cfg: EncoderConfig,
    effort: Effort,
    /// Samples waiting for a full block, per channel.
    pending: Vec<Vec<i32>>,
    frame_number: u64,
    total_samples: u64,
    min_frame: u32,
    max_frame: u32,
    md5: md5::Md5,
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Result<Self, Error> {
        if !(1..=8).contains(&cfg.channels) {
            return Err(Error::InvalidConfig("1 to 8 channels"));
        }
        if !(8..=32).contains(&cfg.bits_per_sample) {
            return Err(Error::InvalidConfig("8 to 32 bits per sample"));
        }
        if !(1..=1_048_575).contains(&cfg.sample_rate) {
            return Err(Error::InvalidConfig("sample rate 1 to 1048575 Hz"));
        }
        if cfg.block_size < 16 {
            return Err(Error::InvalidConfig("block size 16 to 65535"));
        }
        let effort = Effort::level(cfg.level);
        Ok(Encoder {
            pending: vec![Vec::new(); usize::from(cfg.channels)],
            cfg,
            effort,
            frame_number: 0,
            total_samples: 0,
            min_frame: 0,
            max_frame: 0,
            // RFC 9639 requires an MD5 signature of the unencoded audio in STREAMINFO; this is an
            // integrity checksum, not a password hash or a security boundary.
            md5: md5::Md5::default(), // NOSONAR
        })
    }

    /// `fLaC`, the STREAMINFO block (sizes and MD5 unknown yet: see [`Encoder::streaminfo`]) and a
    /// VORBIS_COMMENT block with the vendor string.
    pub fn header(&self) -> Vec<u8> {
        // the vendor string alone always fits the block
        self.header_with_comments(&[]).unwrap_or_default()
    }

    /// [`Encoder::header`] with `comments` (`NAME=value`, such as [`channel_mask_comment`]'s) in
    /// the VORBIS_COMMENT block, or [`Error::InvalidConfig`] when they do not fit its 24-bit
    /// length.
    pub fn header_with_comments(&self, comments: &[&str]) -> Result<Vec<u8>, Error> {
        let too_long = Error::InvalidConfig("comments longer than a metadata block holds");
        // vendor length, vendor, comment count, then a length and the text per comment
        let len =
            comments.iter().try_fold(4 + VENDOR.len() + 4, |n, c| n.checked_add(4)?.checked_add(c.len())).filter(|&n| n < 1 << 24).ok_or(too_long.clone())?;
        let le = |n: usize| u32::try_from(n).map(u32::to_le_bytes).map_err(|_| too_long.clone());
        let mut v = b"fLaC".to_vec();
        v.extend_from_slice(&[0, 0, 0, 34]);
        v.extend_from_slice(&self.streaminfo_with(0, 0, 0, [0; 16]));
        // last metadata block, type 4 (VORBIS_COMMENT), little-endian lengths inside
        v.push(0x84);
        let be = u32::try_from(len).map(u32::to_be_bytes).map_err(|_| too_long.clone())?;
        v.extend_from_slice(be.get(1..).unwrap_or_default());
        v.extend_from_slice(&le(VENDOR.len())?);
        v.extend_from_slice(VENDOR.as_bytes());
        v.extend_from_slice(&le(comments.len())?);
        for c in comments {
            v.extend_from_slice(&le(c.len())?);
            v.extend_from_slice(c.as_bytes());
        }
        Ok(v)
    }

    /// The final STREAMINFO body: frame sizes, total samples and MD5 of what has been encoded.
    pub fn streaminfo(&self) -> [u8; 34] {
        self.streaminfo_with(self.min_frame, self.max_frame, self.total_samples, self.md5.digest())
    }

    fn streaminfo_with(&self, min_frame: u32, max_frame: u32, total: u64, md5: [u8; 16]) -> [u8; 34] {
        let mut w = filmcraft_bitstream::BitWriter::new();
        let bs = u32::from(self.cfg.block_size);
        w.write_bits(bs, 16);
        w.write_bits(bs, 16);
        w.write_bits(min_frame.min(0xFF_FFFF), 24);
        w.write_bits(max_frame.min(0xFF_FFFF), 24);
        w.write_bits(self.cfg.sample_rate, 20);
        w.write_bits(u32::from(self.cfg.channels) - 1, 3);
        w.write_bits(u32::from(self.cfg.bits_per_sample) - 1, 5);
        // 36 bits; 0 means unknown, and a count that does not fit is left unknown
        let total = if total >> 36 == 0 { total } else { 0 };
        w.write_bits((total >> 32) as u32, 4);
        w.write_bits(total as u32, 32);
        w.write_bytes(&md5);
        let mut out = [0u8; 34];
        out.copy_from_slice(w.bytes().get(..34).unwrap_or(&[0; 34]));
        out
    }

    /// Add samples (one slice per channel, equal lengths); returns the frames completed.
    pub fn encode(&mut self, planar: &[&[i32]]) -> Result<Vec<u8>, Error> {
        Ok(self.encode_frames(planar)?.concat())
    }

    /// [`Encoder::encode`] with each completed frame on its own (one MP4 / QuickTime sample each).
    pub fn encode_frames(&mut self, planar: &[&[i32]]) -> Result<Vec<Vec<u8>>, Error> {
        if planar.len() != self.pending.len() {
            return Err(Error::Channels { expected: self.pending.len(), got: planar.len() });
        }
        let n = planar.first().map_or(0, |c| c.len());
        if planar.iter().any(|c| c.len() != n) {
            return Err(Error::UnequalChannels);
        }
        // i64: at 32 bits the range is all of i32 (1 << 31 overflows i32)
        let max = (1i64 << (self.cfg.bits_per_sample - 1)) - 1;
        if let Some(&bad) = planar.iter().flat_map(|c| c.iter()).find(|&&s| i64::from(s) > max || i64::from(s) < -max - 1) {
            return Err(Error::SampleRange(bad));
        }
        for (p, c) in self.pending.iter_mut().zip(planar) {
            p.extend_from_slice(c);
        }
        let bs = usize::from(self.cfg.block_size);
        let mut out = Vec::new();
        while self.pending.first().is_some_and(|p| p.len() >= bs) {
            let block: Vec<Vec<i32>> = self.pending.iter_mut().map(|p| p.drain(..bs).collect()).collect();
            out.push(self.frame(&block));
        }
        Ok(out)
    }

    /// Samples per channel waiting for a full block: the length of the frame [`Encoder::finish`]
    /// would write.
    pub fn pending_samples(&self) -> usize {
        self.pending.first().map_or(0, Vec::len)
    }

    /// Encode what is left as a final, shorter frame.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.pending.first().is_none_or(Vec::is_empty) {
            return Vec::new();
        }
        let block: Vec<Vec<i32>> = self.pending.iter_mut().map(std::mem::take).collect();
        self.frame(&block)
    }

    fn frame(&mut self, block: &[Vec<i32>]) -> Vec<u8> {
        let n = block.first().map_or(0, Vec::len);
        let bps = self.cfg.bits_per_sample;
        // the MD5 of the unencoded audio: interleaved, little-endian, whole bytes per sample
        let bytes = usize::from(bps.div_ceil(8));
        let mut raw = Vec::with_capacity(n * block.len() * bytes);
        for i in 0..n {
            for c in block {
                raw.extend_from_slice(c.get(i).copied().unwrap_or(0).to_le_bytes().get(..bytes).unwrap_or_default());
            }
        }
        self.md5.update(&raw);

        let effort = self.effort;
        let code = |x: &[i32], b: u8| predict::choose(x, b, effort);
        let data = match block {
            // the side channel takes one bit more than the samples: at most 32 (i32)
            [l, r] if effort.stereo && bps < 32 => {
                let side: Vec<i32> = l.iter().zip(r).map(|(&a, &b)| a - b).collect();
                let mid: Vec<i32> = l.iter().zip(r).map(|(&a, &b)| (a + b) >> 1).collect();
                let (cl, cr, cs, cm) = (code(l, bps), code(r, bps), code(&side, bps + 1), code(&mid, bps));
                let options =
                    [(Stereo::Independent, cl.1 + cr.1), (Stereo::LeftSide, cl.1 + cs.1), (Stereo::SideRight, cs.1 + cr.1), (Stereo::MidSide, cm.1 + cs.1)];
                let best = options.iter().min_by_key(|o| o.1).map_or(Stereo::Independent, |o| o.0);
                let channels: Vec<(&[i32], Subframe, u8)> = match best {
                    Stereo::Independent => vec![(l, cl.0, bps), (r, cr.0, bps)],
                    Stereo::LeftSide => vec![(l, cl.0, bps), (&side, cs.0, bps + 1)],
                    Stereo::SideRight => vec![(&side, cs.0, bps + 1), (r, cr.0, bps)],
                    Stereo::MidSide => vec![(&mid, cm.0, bps), (&side, cs.0, bps + 1)],
                };
                frame::write(&FrameSpec { number: self.frame_number, sample_rate: self.cfg.sample_rate, bps, stereo: Some(best), channels: &channels })
            }
            _ => {
                let channels: Vec<(&[i32], Subframe, u8)> = block.iter().map(|c| (c.as_slice(), code(c, bps).0, bps)).collect();
                frame::write(&FrameSpec { number: self.frame_number, sample_rate: self.cfg.sample_rate, bps, stereo: None, channels: &channels })
            }
        };
        self.frame_number += 1;
        self.total_samples += n as u64;
        let size = u32::try_from(data.len()).unwrap_or(u32::MAX);
        self.min_frame = if self.min_frame == 0 { size } else { self.min_frame.min(size) };
        self.max_frame = self.max_frame.max(size);
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The VORBIS_COMMENT block of a header: (vendor, comments).
    fn comments(header: &[u8]) -> (String, Vec<String>) {
        let block = &header[4 + 4 + 34..];
        assert_eq!(block[0], 0x84, "the last block, VORBIS_COMMENT");
        let len = u32::from_be_bytes([0, block[1], block[2], block[3]]) as usize;
        let body = &block[4..];
        assert_eq!(body.len(), len);
        let u32_at = |at: usize| u32::from_le_bytes(body[at..at + 4].try_into().unwrap()) as usize;
        let vendor_len = u32_at(0);
        let vendor = String::from_utf8(body[4..4 + vendor_len].to_vec()).unwrap();
        let mut at = 4 + vendor_len;
        let n = u32_at(at);
        at += 4;
        let mut out = Vec::new();
        for _ in 0..n {
            let l = u32_at(at);
            out.push(String::from_utf8(body[at + 4..at + 4 + l].to_vec()).unwrap());
            at += 4 + l;
        }
        assert_eq!(at, body.len());
        (vendor, out)
    }

    #[test]
    fn the_mask_comment_is_written_only_when_decoders_would_assume_otherwise() {
        let comment = |m: &str| Some(format!("WAVEFORMATEXTENSIBLE_CHANNEL_MASK={m}"));
        for channels in 1..=8 {
            assert_eq!(channel_mask_comment(channels, assumed_channel_mask(channels)), None, "{channels}");
        }
        // 5.0 and 5.1 with back surrounds (WAV's usual 5.1 mask) are read as side surrounds without it
        assert_eq!(channel_mask_comment(5, 0x37), comment("0x0037"));
        assert_eq!(channel_mask_comment(6, 0x3F), comment("0x003F"));
        assert_eq!(channel_mask_comment(7, 0x13F), comment("0x013F"));
        // no speaker assignment
        assert_eq!(channel_mask_comment(2, 0), comment("0x0000"));
        assert_eq!(channel_mask_comment(6, 0x3_FFFF), comment("0x3FFFF"));
        assert_eq!((assumed_channel_mask(0), assumed_channel_mask(9), assumed_channel_mask(u8::MAX)), (0, 0, 0));
    }

    #[test]
    fn the_header_carries_its_comments() {
        let enc = Encoder::new(EncoderConfig::new(48_000, 6, 24)).unwrap();
        assert_eq!(comments(&enc.header()), (VENDOR.to_string(), Vec::new()));
        assert_eq!(enc.header_with_comments(&[]).unwrap(), enc.header());
        let mask = channel_mask_comment(6, 0x3F).unwrap();
        let header = enc.header_with_comments(&[&mask, "TITLE=x"]).unwrap();
        assert_eq!(comments(&header), (VENDOR.to_string(), vec![mask, "TITLE=x".to_string()]));
        assert_eq!(header[..STREAMINFO_OFFSET as usize + 34], enc.header()[..STREAMINFO_OFFSET as usize + 34]);
    }

    /// Comments past the block's 24-bit length are refused, not written with a wrapped length.
    #[test]
    fn comments_too_long_for_the_block_are_an_error() {
        let enc = Encoder::new(EncoderConfig::new(48_000, 2, 16)).unwrap();
        let room = (1 << 24) - 1 - (4 + VENDOR.len() + 4 + 4);
        let fits = "x".repeat(room);
        assert!(enc.header_with_comments(&[&fits]).is_ok());
        let over = "x".repeat(room + 1);
        assert!(matches!(enc.header_with_comments(&[&over]), Err(Error::InvalidConfig(_))));
        let many = vec![""; 1 << 22];
        assert!(enc.header_with_comments(&many).is_err());
    }
}

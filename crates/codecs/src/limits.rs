//! How large a picture, and how much picture memory, our own (software) video decoders accept:
//! one process-wide setting for a workflow past 8K, like the hardware decoding switch
//! ([`crate::hw::set_hardware_decoding`]). Each decoder refuses a stream over its limits before
//! it allocates for it (an error the media pool shows as an unreadable clip, never an
//! out-of-memory abort); the defaults let every picture up to 8K through.
//!
//! The limits reach the H.264, VP8, VP9, MPEG-1/2 and ProRes decoders. Decoders already made
//! keep the limits they were made with: a source picks a change up when it is reopened. HEVC, AV1
//! and the other intra codecs keep their own fixed limits, and hardware decoders their OS's.

use std::sync::{PoisonError, RwLock};

/// Limits for the software video decoders made from now on. `None` keeps each decoder's own
/// default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecoderLimits {
    /// The largest picture, in luma samples (width × height). H.264 counts it as the memory of an
    /// 8-bit 4:2:0 picture of that many samples, so a 10-bit or 4:2:2 picture gets the same
    /// memory, not more (default 2^25); VP8 counts it on the macroblock grid (default 2^26); VP9
    /// (default 2^26) and MPEG-1/2 (default 2^25) as the luma area; ProRes as the frame area
    /// (no area limit by default, the plane memory bounds it).
    pub max_luma_samples: Option<u64>,
    /// Bytes the decoded pictures of one decoder may take: what VP9 holds (references, frames in
    /// their post stage and output pictures; 12 GiB by default, above anything its size limits
    /// admit), what MPEG-1/2 holds (anchors, the frame being decoded and one access unit's
    /// output; 4 GiB by default), 1 GiB for either on 32-bit targets; the planes of one ProRes
    /// frame, 1 GiB by default. H.264 and VP8 have no budget of their own: their picture limit
    /// bounds them.
    pub memory_budget: Option<u64>,
}

static LIMITS: RwLock<DecoderLimits> = RwLock::new(DecoderLimits { max_luma_samples: None, memory_budget: None });

/// Set the limits of the software video decoders made from now on (process-wide).
pub fn set_decoder_limits(limits: DecoderLimits) {
    *LIMITS.write().unwrap_or_else(PoisonError::into_inner) = limits;
}

/// The limits new software video decoders get ([`set_decoder_limits`]; all defaults until set).
pub fn decoder_limits() -> DecoderLimits {
    *LIMITS.read().unwrap_or_else(PoisonError::into_inner)
}

impl DecoderLimits {
    /// Apply the limits to an H.264 decoder.
    pub fn apply_h264(&self, dec: &mut filmcraft_h264::Decoder) {
        if let Some(n) = self.max_luma_samples {
            dec.set_max_picture_luma_samples(n);
        }
    }

    /// A VP8 decoder with these limits.
    pub fn vp8_decoder(&self) -> filmcraft_vp8::Decoder {
        match self.max_luma_samples {
            Some(n) => filmcraft_vp8::Decoder::with_max_luma_samples(usize::try_from(n).unwrap_or(usize::MAX)),
            None => filmcraft_vp8::Decoder::new(),
        }
    }

    /// Apply the limits to a VP9 decoder.
    pub fn apply_vp9(&self, dec: &mut filmcraft_vp9::Decoder) {
        if let Some(n) = self.max_luma_samples {
            dec.set_max_picture_luma_samples(n);
        }
        if let Some(b) = self.memory_budget {
            dec.set_memory_budget(b);
        }
    }

    /// Apply the limits to an MPEG-1/2 decoder (before it sees a sequence header).
    pub fn apply_mpeg2(&self, dec: &mut filmcraft_mpeg2v::Decoder) {
        if let Some(n) = self.max_luma_samples {
            dec.set_max_picture_luma_samples(n);
        }
        if let Some(b) = self.memory_budget {
            dec.set_memory_budget(b);
        }
    }

    /// The MPEG-1/2 area limit, for probing a sequence header as the decoder will take it.
    pub fn mpeg2_area(&self) -> u64 {
        self.max_luma_samples.unwrap_or(filmcraft_mpeg2v::MAX_PICTURE_AREA)
    }

    /// The ProRes decoding limits.
    pub fn prores(&self) -> filmcraft_prores::DecodeLimits {
        let d = filmcraft_prores::DecodeLimits::default();
        filmcraft_prores::DecodeLimits {
            max_pixels: self.max_luma_samples.unwrap_or(d.max_pixels),
            max_frame_bytes: self.memory_budget.map_or(d.max_frame_bytes, |b| usize::try_from(b).unwrap_or(usize::MAX)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_keeps_every_decoders_own_limits() {
        let d = DecoderLimits::default();
        assert_eq!(d.prores(), filmcraft_prores::DecodeLimits::default());
        assert_eq!(d.mpeg2_area(), filmcraft_mpeg2v::MAX_PICTURE_AREA);
        assert_eq!(d.vp8_decoder().max_luma_samples(), filmcraft_vp8::MAX_LUMA_SAMPLES);
        let mut h264 = filmcraft_h264::Decoder::with_threads(1);
        d.apply_h264(&mut h264);
        assert_eq!(h264.max_picture_luma_samples(), filmcraft_h264::params::MAX_LUMA_SAMPLES);
        let mut vp9 = filmcraft_vp9::Decoder::with_threads(1);
        d.apply_vp9(&mut vp9);
        assert_eq!((vp9.max_picture_luma_samples(), vp9.memory_budget()), (filmcraft_vp9::MAX_LUMA_SAMPLES, filmcraft_vp9::DECODE_MEMORY_BUDGET));
        let mut mpeg2 = filmcraft_mpeg2v::Decoder::new();
        d.apply_mpeg2(&mut mpeg2);
        assert_eq!((mpeg2.max_picture_luma_samples(), mpeg2.memory_budget()), (filmcraft_mpeg2v::MAX_PICTURE_AREA, filmcraft_mpeg2v::DECODE_MEMORY_BUDGET));
    }

    #[test]
    fn set_limits_reach_every_decoder() {
        let l = DecoderLimits { max_luma_samples: Some(1 << 27), memory_budget: Some(8 << 30) };
        assert_eq!(l.vp8_decoder().max_luma_samples(), 1 << 27);
        let mut h264 = filmcraft_h264::Decoder::with_threads(1);
        l.apply_h264(&mut h264);
        assert_eq!(h264.max_picture_luma_samples(), 1 << 27);
        let mut vp9 = filmcraft_vp9::Decoder::with_threads(1);
        l.apply_vp9(&mut vp9);
        assert_eq!((vp9.max_picture_luma_samples(), vp9.memory_budget()), (1 << 27, 8 << 30));
        let mut mpeg2 = filmcraft_mpeg2v::Decoder::new();
        l.apply_mpeg2(&mut mpeg2);
        assert_eq!((mpeg2.max_picture_luma_samples(), mpeg2.memory_budget()), (1 << 27, 8 << 30));
        assert_eq!(l.mpeg2_area(), 1 << 27);
        let p = l.prores();
        assert_eq!(p.max_pixels, 1 << 27);
        assert_eq!(p.max_frame_bytes, usize::try_from(8u64 << 30).unwrap_or(usize::MAX));
        // hostile values saturate instead of wrapping
        let huge = DecoderLimits { max_luma_samples: Some(u64::MAX), memory_budget: Some(u64::MAX) };
        assert_eq!(huge.prores().max_frame_bytes, usize::MAX);
        assert_eq!(huge.vp8_decoder().max_luma_samples(), usize::MAX);
    }
}

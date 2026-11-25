//! The process-wide decoder limits (`filmcraft_codecs::limits`) reach our own H.264, VP8, VP9,
//! MPEG-2 and ProRes decoders through the media stack. The setting is global, so this binary
//! holds a single test (nothing else may decode while it changes the limits).

mod common;

use common::*;
use filmcraft_codecs::{DecoderLimits, decoder_limits, set_decoder_limits};
use filmcraft_media::FrameRequest;
use std::path::Path;

/// Whether a fresh source of `file` opens and decodes its first frame at 64x48.
fn decodes(name: &str, file: &Path) -> bool {
    let Ok(src) = filmcraft_codecs::open_bytes(name, bytes(file)) else { return false };
    src.video_frame(FrameRequest::full(filmcraft_time::Tick::ZERO)).is_ok_and(|f| (f.width, f.height) == (64, 48))
}

#[test]
fn decoder_limits_reach_every_software_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    assert_eq!(decoder_limits(), DecoderLimits::default());
    let src = ["-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.4"];
    // (file, encoder arguments, whether the decoder has a memory budget)
    let cases: [(&str, &[&str], bool); 5] = [
        ("limits_h264.mp4", &["-c:v", "libx264", "-pix_fmt", "yuv420p"], false),
        ("limits_vp8.webm", &["-c:v", "libvpx", "-b:v", "200k"], false),
        ("limits_vp9.webm", &["-c:v", "libvpx-vp9", "-b:v", "200k"], true),
        ("limits_mpeg2.mpg", &["-c:v", "mpeg2video", "-q:v", "4"], true),
        ("limits_prores.mov", &["-c:v", "prores_ks", "-profile:v", "2"], true),
    ];
    let mut checked = 0;
    for (name, enc, budget) in cases {
        let args: Vec<&str> = src.iter().chain(enc).copied().collect();
        let Some(file) = fixture(&ff, name, &args) else { continue };
        let with = |limits: DecoderLimits| {
            set_decoder_limits(limits);
            assert_eq!(decoder_limits(), limits);
            decodes(name, &file)
        };
        assert!(with(DecoderLimits::default()), "{name}: decodes by default");
        let samples = |n: u64| DecoderLimits { max_luma_samples: Some(n), memory_budget: None };
        assert!(with(samples(64 * 48)), "{name}: decodes at a limit of its own size");
        assert!(!with(samples(64 * 48 - 1)), "{name}: refused one luma sample below its size");
        assert!(with(samples(1 << 27)), "{name}: decodes at a raised limit");
        let tiny_budget = DecoderLimits { max_luma_samples: None, memory_budget: Some(1) };
        assert_eq!(with(tiny_budget), !budget, "{name}: a one-byte budget refuses exactly the decoders that have one");
        assert!(with(DecoderLimits::default()), "{name}: decodes again once the limits are back");
        checked += 1;
    }
    set_decoder_limits(DecoderLimits::default());
    assert!(checked > 0, "no fixture could be generated");
}

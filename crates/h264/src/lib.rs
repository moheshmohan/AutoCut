//! Clean-room pure-Rust H.264 / AVC decoder, implemented from ITU-T Rec. H.264 (ISO/IEC 14496-10).
//!
//! Supported: progressive (frame) coding, 4:2:0 at 8 to 10 bits: Baseline / Main / High, High 10
//! and High 10 Intra profiles — CAVLC and CABAC, I/P/B slices, 8x8 transform, custom scaling
//! matrices, weighted prediction, spatial/temporal direct, multiple slices, long-term references
//! and all MMCOs, deblocking.
//!
//! ```no_run
//! let mut dec = filmcraft_h264::Decoder::new();
//! let stream = std::fs::read("video.h264").unwrap();
//! for pic in dec.decode(&stream, 0).unwrap() {
//!     println!("{}x{} poc {}", pic.width, pic.height, pic.poc);
//! }
//! for pic in dec.flush() {
//!     let _ = pic;
//! }
//! ```

// Index loops over fixed-size blocks read more clearly than iterator chains in codec code.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

mod cabac;
mod cabac_mb;
#[rustfmt::skip]
mod cabac_tables;
mod cavlc;
#[rustfmt::skip]
mod cavlc_tables;
mod deblock;
mod decoder;
pub mod dpb;
mod error;
mod inter;
mod intra;
mod mbtypes;
pub mod params;
mod picture;
pub use picture::set_plane_allocator;
pub mod slice;
mod slicedec;
#[cfg(test)]
mod synth_tests;
mod tables;
mod transform;

pub use decoder::{DecodeStats, Decoder, default_threads};
pub use error::{Error, Result};

/// Colour description from the VUI (ITU-T H.273 code points).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorInfo {
    /// `video_full_range_flag`.
    pub full_range: bool,
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

/// Chroma sampling of a decoded picture (`chroma_format_idc`; 4:4:4 and monochrome are not
/// decoded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 4:2:0: chroma planes are half the luma width and height.
    C420,
    /// 4:2:2: chroma planes are half the luma width and the full luma height.
    C422,
}

impl ChromaFormat {
    /// The decoded format for `ChromaArrayType`, if this decoder handles it.
    pub fn from_idc(idc: u32) -> Option<ChromaFormat> {
        match idc {
            1 => Some(ChromaFormat::C420),
            2 => Some(ChromaFormat::C422),
            _ => None,
        }
    }
    /// log2(SubWidthC), log2(SubHeightC): the shifts from luma to chroma sample positions.
    #[inline(always)]
    pub fn shifts(self) -> (u32, u32) {
        match self {
            ChromaFormat::C420 => (1, 1),
            ChromaFormat::C422 => (1, 0),
        }
    }
}

/// Samples of one plane: pictures whose luma and chroma are both 8-bit produce `U8`, any higher
/// bit depth `U16` (value range `0..(1 << bit_depth)` of the plane).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plane {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl Plane {
    pub fn len(&self) -> usize {
        match self {
            Plane::U8(v) => v.len(),
            Plane::U16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Plane::U8(v) => Some(v),
            Plane::U16(_) => None,
        }
    }
    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Plane::U8(_) => None,
            Plane::U16(v) => Some(v),
        }
    }
    /// Sample `i` widened to u16 (0 past the end).
    pub fn get(&self, i: usize) -> u16 {
        match self {
            Plane::U8(v) => v.get(i).copied().map(u16::from).unwrap_or(0),
            Plane::U16(v) => v.get(i).copied().unwrap_or(0),
        }
    }
    /// Raw bytes as written by ffmpeg `rawvideo` (8-bit, or 16-bit little endian).
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Plane::U8(v) => v.clone(),
            Plane::U16(v) => v.iter().flat_map(|s| s.to_le_bytes()).collect(),
        }
    }
}

/// An 8-bit plane equals its samples; a 16-bit one never equals bytes.
impl PartialEq<[u8]> for Plane {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_u8() == Some(other)
    }
}

impl PartialEq<Vec<u8>> for Plane {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_u8() == Some(&other[..])
    }
}

/// A decoded, cropped picture in planar 4:2:0 ([`Picture::chroma`]), 8 to 10 bits.
#[derive(Clone, Debug)]
pub struct Picture {
    /// Cropped luma width.
    pub width: u32,
    /// Cropped luma height.
    pub height: u32,
    /// Size of the `u` / `v` planes: the luma size divided by SubWidthC / SubHeightC, rounded up.
    pub chroma_width: u32,
    pub chroma_height: u32,
    /// Chroma sampling of the `u` / `v` planes.
    pub chroma: ChromaFormat,
    /// Luma bit depth. The three planes share one storage type: `u8` only when both depths are 8.
    pub bit_depth: u32,
    /// Chroma bit depth; a stream may code it apart from the luma depth.
    pub bit_depth_chroma: u32,
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
    /// Strides in samples.
    pub y_stride: usize,
    pub uv_stride: usize,
    /// Presentation timestamp passed to [`Decoder::decode`] with the access unit of this picture.
    pub pts: i64,
    /// Picture order count.
    pub poc: i32,
    /// IDR picture.
    pub key: bool,
    pub color: ColorInfo,
    /// Sample aspect ratio (0, 0 when unspecified).
    pub sar: (u16, u16),
    /// Decoded in draft mode ([`Decoder::set_draft`]): a non-reference picture whose deblocking
    /// filter was skipped. Its samples are approximate (not the conforming output); no other
    /// picture is affected.
    pub draft: bool,
}

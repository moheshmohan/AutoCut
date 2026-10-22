//! Seeing what to keep private in a picture: faces and vehicle number plates, for FilmCraft's
//! automatic privacy blur (the engine's `clip.autoRedact`).
//!
//! - [`catalog`]: the detector models — OpenCV Zoo's **YuNet** face detector (MIT, Shiqi Yu),
//!   **LPD-YuNet** licence-plate detector (Apache-2.0, OpenCV Zoo / watrix.ai), and for the
//!   profile pass the **D-FINE** person detector (Apache-2.0) and the **RTMPose** pose network
//!   (Apache-2.0, OpenMMLab) — pinned to a revision and SHA-256. They are downloaded after the
//!   user confirms, never bundled.
//! - [`onnx`], [`safetensors`], [`zip`]: strict readers of the models' files (layer lists and
//!   weights; nothing runs).
//! - [`post`]: decoding the detectors' outputs (YuNet's anchor-free grid, LPD-YuNet's prior
//!   boxes) and non-maximum suppression.
//! - [`head`]: from a person's pose to a face to hide (the head box, whether the face shows,
//!   which heads the face detector already found).
//! - [`track`]: linking detections into tracks, filling gaps, smoothing, and the rules for which
//!   tracks are hidden (recognisable faces, legible plates, not excluded).
//! - `net` and [`Detectors`] (feature `model`): the two networks written out by hand, checked
//!   layer by layer against the ONNX file: on the CPU (candle and plain Rust, the reference) and,
//!   on macOS, as fused WGSL compute passes on the GPU (`gpu`, wgpu), the CPU being the fallback.
//! - `dfine`, `rtmpose` and [`ProfileDetector`] (feature `model`): the profile pass's person
//!   detector and pose network, finding the faces YuNet misses: on the CPU (`layers`, the
//!   reference) and, on macOS, wholly on the GPU (`gpunn`: the frame resized, both networks, the
//!   crops and the SimCC maxima as WGSL passes), the CPU being the fallback.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod catalog;
pub mod head;
pub mod onnx;
pub mod post;
pub mod safetensors;
pub mod track;
pub mod zip;

#[cfg(feature = "model")]
mod dfine;
#[cfg(feature = "model")]
mod gpu;
#[cfg(feature = "model")]
mod gpunn;
#[cfg(all(test, feature = "model"))]
mod gpunn_tests;
#[cfg(feature = "model")]
mod layers;
#[cfg(feature = "model")]
mod net;
#[cfg(feature = "model")]
mod people;
#[cfg(all(test, feature = "model"))]
mod profile_tests;
#[cfg(feature = "model")]
mod rtmpose;
#[cfg(feature = "model")]
pub use net::{Detectors, FACE_SIDE, FaceDetector, MAX_GPU_FAILURES, PLATE_SIDE, PlateDetector, gpu_available};
#[cfg(feature = "model")]
pub use people::{ProfileDetector, ProfileParams};

use serde::{Deserialize, Serialize};

/// What a detection is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Face,
    Plate,
    /// Something named by the user (a logo, a sign), found by a vision model.
    Target,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Face => "face",
            Kind::Plate => "plate",
            Kind::Target => "target",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "face" | "faces" => Some(Kind::Face),
            "plate" | "plates" | "licenseplate" | "licenseplates" | "numberplate" | "numberplates" => Some(Kind::Plate),
            _ => None,
        }
    }
}

/// One face or plate found in a picture. Coordinates are in the picture's pixels as the
/// detectors give them; [`Detection::normalized`] makes them shares of the picture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub kind: Kind,
    /// Confidence, 0..1.
    pub score: f32,
    /// `[x, y, w, h]`: top-left corner and size.
    pub bbox: [f32; 4],
    /// A face's five landmarks (right eye, left eye, nose tip, right and left mouth corners);
    /// a plate's four corners, clockwise from the top left; a head found from a pose: its nose,
    /// eyes and ears that were found.
    pub points: Vec<[f32; 2]>,
    /// A face found from a person's pose (the profile pass: a head in profile or half hidden that
    /// the face detector missed), not by the face detector.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pose: bool,
}

impl Detection {
    /// Coordinates as shares of a `w` × `h` picture.
    pub fn normalized(&self, w: f32, h: f32) -> Self {
        let (w, h) = (w.max(1.0), h.max(1.0));
        Self {
            kind: self.kind,
            score: self.score,
            bbox: [self.bbox[0] / w, self.bbox[1] / h, self.bbox[2] / w, self.bbox[3] / h],
            points: self.points.iter().map(|p| [p[0] / w, p[1] / h]).collect(),
            pose: self.pose,
        }
    }
}

/// Why detection failed.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum VisionError {
    /// The models are not (all) downloaded.
    #[error("the detectors are not downloaded yet")]
    NotInstalled,
    /// A model file is damaged or not the network this code implements, or computing failed.
    #[error("detector: {0}")]
    Model(String),
    /// The picture cannot be used (empty, too large, wrong buffer size).
    #[error("{0}")]
    Invalid(String),
}

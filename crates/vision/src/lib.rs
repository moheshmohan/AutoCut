//! Seeing what to keep private in a picture: faces and vehicle number plates, for FilmCraft's
//! automatic privacy blur (the engine's `clip.autoRedact`).
//!
//! - [`catalog`]: the two detector models — OpenCV Zoo's **YuNet** face detector (MIT, Shiqi Yu)
//!   and **LPD-YuNet** licence-plate detector (Apache-2.0, OpenCV Zoo / watrix.ai) — pinned to a
//!   revision and SHA-256. They are downloaded after the user confirms, never bundled.
//! - [`onnx`]: a strict reader of the models' ONNX files (layer list and weights; nothing runs).
//! - [`post`]: decoding the detectors' outputs (YuNet's anchor-free grid, LPD-YuNet's prior
//!   boxes) and non-maximum suppression.
//! - [`track`]: linking detections into tracks, filling gaps, smoothing, and the rules for which
//!   tracks are hidden (recognisable faces, legible plates, not excluded).
//! - `net` and [`Detectors`] (feature `model`): the two networks written out by hand on candle
//!   (pure Rust, CPU), checked layer by layer against the ONNX file.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod catalog;
pub mod onnx;
pub mod post;
pub mod track;

#[cfg(feature = "model")]
mod net;
#[cfg(feature = "model")]
pub use net::{Detectors, FACE_SIDE, FaceDetector, PLATE_SIDE, PlateDetector};

use serde::{Deserialize, Serialize};

/// What a detection is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Face,
    Plate,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Face => "face",
            Kind::Plate => "plate",
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
    /// a plate's four corners, clockwise from the top left.
    pub points: Vec<[f32; 2]>,
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
        }
    }
}

/// Why detection failed.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum VisionError {
    /// The models are not (all) downloaded.
    #[error("the face and plate detectors are not downloaded yet")]
    NotInstalled,
    /// A model file is damaged or not the network this code implements, or computing failed.
    #[error("detector: {0}")]
    Model(String),
    /// The picture cannot be used (empty, too large, wrong buffer size).
    #[error("{0}")]
    Invalid(String),
}

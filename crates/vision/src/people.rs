//! The profile pass: the people in a picture (D-FINE, [`crate::dfine`]), their poses (RTMPose,
//! [`crate::rtmpose`]), and from those the faces the face detector misses — in profile, half
//! hidden — as face detections ([`crate::head::candidates`]).

use std::path::Path;

use crate::dfine::{self, Dfine};
use crate::head::{self, HeadParams, Person};
use crate::rtmpose::RtmPose;
use crate::{Detection, VisionError, catalog, post};

/// The thresholds and sizes of the profile pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileParams {
    /// A person needs at least this score (people under [`crate::track::WEAK_POSE`] only make weak
    /// sightings, which continue a track but never start one).
    pub person_score: f32,
    /// Most people whose pose is looked at in one picture (the surest first).
    pub max_people: usize,
    /// A person's box at least this share of the picture's height (smaller people's faces are
    /// far too small to recognise).
    pub min_person: f32,
    /// Longest side of the picture the person detector sees.
    pub side: usize,
    pub head: HeadParams,
}

impl Default for ProfileParams {
    fn default() -> Self {
        Self { person_score: 0.25, max_people: 24, min_person: 0.04, side: dfine::PERSON_SIDE, head: HeadParams::default() }
    }
}

/// The person detector and the pose network.
pub struct ProfileDetector {
    person: Dfine,
    pose: RtmPose,
    pub params: ProfileParams,
}

impl ProfileDetector {
    /// Load both networks from the models directory ([`catalog::PROFILE_FILES`]).
    pub fn load(models_dir: &Path) -> Result<Self, VisionError> {
        if !catalog::profiles_installed(models_dir) {
            return Err(VisionError::NotInstalled);
        }
        let dir = catalog::package_dir(models_dir);
        let person = Dfine::load(&dir.join(catalog::PERSON_WEIGHTS.name), &dir.join(catalog::PERSON_CONFIG.name))?;
        let pose = RtmPose::load(&dir.join(catalog::POSE.name))?;
        Ok(Self { person, pose, params: ProfileParams::default() })
    }

    /// Load, and put the convolutional parts on this machine's GPU where there is one (the CPU is
    /// the fallback) unless `gpu` is false.
    pub fn load_on(models_dir: &Path, gpu: bool) -> Result<Self, VisionError> {
        let mut d = Self::load(models_dir)?;
        if gpu {
            d.person.use_gpu();
            d.pose.use_gpu();
        }
        Ok(d)
    }

    /// Where the networks' convolutions run now: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if self.person.on_gpu() && self.pose.on_gpu() { "gpu" } else { "cpu" }
    }

    /// The people in an RGBA picture with their poses, in its pixels, best first.
    pub fn people(&self, rgba: &[u8], w: usize, h: usize) -> Result<Vec<Person>, VisionError> {
        let p = &self.params;
        let x = dfine::prepare(rgba, w, h, p.side)?;
        let found = self.person.people(&x, p.person_score.clamp(0.01, 1.0))?;
        let (fw, fh) = (w as f32, h as f32);
        let boxes: Vec<Detection> = found
            .iter()
            .map(|f| {
                let [cx, cy, bw, bh] = f.cxcywh;
                Detection {
                    kind: crate::Kind::Face,
                    score: f.score,
                    bbox: [(cx - bw / 2.0) * fw, (cy - bh / 2.0) * fh, bw * fw, bh * fh],
                    points: Vec::new(),
                    pose: false,
                }
            })
            .filter(|d| d.bbox[3] >= p.min_person * fh)
            .collect();
        // a DETR rarely finds one person twice; when it does, the better box stays
        let mut boxes = post::nms(boxes, 0.7, 300);
        boxes.truncate(p.max_people.min(64));
        let rects: Vec<[f32; 4]> = boxes.iter().map(|d| d.bbox).collect();
        let poses = self.pose.poses(rgba, w, h, &rects)?;
        Ok(boxes.into_iter().zip(poses).map(|(d, keypoints)| Person { bbox: d.bbox, score: d.score, keypoints }).collect())
    }

    /// The faces the face detector missed in an RGBA picture: the heads of the people whose
    /// face shows and is not one of `faces` (the face detector's, normalised), as face
    /// detections normalised to the picture ([`Detection::pose`]).
    pub fn heads(&self, rgba: &[u8], w: usize, h: usize, faces: &[Detection]) -> Result<Vec<Detection>, VisionError> {
        let people = self.people(rgba, w, h)?;
        Ok(head::candidates(&people, faces, &self.params.head, w as f32, h as f32))
    }
}

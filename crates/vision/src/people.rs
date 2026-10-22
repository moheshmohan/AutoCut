//! The profile pass: the people in a picture (D-FINE, [`crate::dfine`]), their poses (RTMPose,
//! [`crate::rtmpose`]), and from those the faces the face detector misses — in profile, half
//! hidden — as face detections ([`crate::head::candidates`]).
//!
//! On the GPU (macOS) the whole pass runs there: the frame goes up once, is resized for the
//! person detector, the people's crops are cut from it, and two small read-backs come down (the
//! people, then their keypoints). The CPU is the fallback, network by network.

use std::path::Path;

use crate::dfine::{self, Dfine, Found};
use crate::gpunn::{GpuUse, Run};
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
    pub(crate) person: Dfine,
    pub(crate) pose: RtmPose,
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

    /// The people in an RGBA picture with their poses, in its pixels, best first: on the GPU
    /// when both networks are there, else (or when that fails) as [`ProfileDetector::people_cpu`].
    pub fn people(&self, rgba: &[u8], w: usize, h: usize) -> Result<Vec<Person>, VisionError> {
        // a picture that cannot be analysed is refused here, not counted as a GPU failure
        let (ow, oh) = dfine::prepared_size(w, h, self.params.side)?;
        if w.checked_mul(h).and_then(|n| n.checked_mul(4)).is_none_or(|n| rgba.len() < n) {
            return Err(VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture", rgba.len())));
        }
        if self.person.on_gpu()
            && self.pose.on_gpu()
            && let Some(people) = GpuUse::run(&self.person.gpu, |r| self.people_gpu(r, rgba, w, h, (ow, oh)))
        {
            return Ok(people);
        }
        self.people_cpu(rgba, w, h)
    }

    /// [`ProfileDetector::people`] with the picture prepared on the CPU (each network on the GPU
    /// where it is, else on the CPU).
    pub fn people_cpu(&self, rgba: &[u8], w: usize, h: usize) -> Result<Vec<Person>, VisionError> {
        let p = &self.params;
        let x = dfine::prepare(rgba, w, h, p.side)?;
        let found = self.person.people(&x, p.person_score.clamp(0.01, 1.0))?;
        let boxes = self.select(&found, w, h);
        let rects: Vec<[f32; 4]> = boxes.iter().map(|d| d.bbox).collect();
        let poses = self.pose.poses(rgba, w, h, &rects)?;
        Ok(boxes.into_iter().zip(poses).map(|(d, keypoints)| Person { bbox: d.bbox, score: d.score, keypoints }).collect())
    }

    /// The whole pass on the GPU (`size`: the person detector's picture size).
    fn people_gpu(&self, r: &Run<'_>, rgba: &[u8], w: usize, h: usize, size: (usize, usize)) -> Result<Vec<Person>, VisionError> {
        let p = &self.params;
        let img = r.upload_rgba(rgba, w, h)?;
        let x = r.resize_area(&img, size.0, size.1)?;
        let found = self.person.people_gpu(r, x, p.person_score.clamp(0.01, 1.0))?;
        let boxes = self.select(&found, w, h);
        let rects: Vec<[f32; 4]> = boxes.iter().map(|d| d.bbox).collect();
        let poses = self.pose.poses_gpu(r, &img, &rects)?;
        if poses.len() != boxes.len() {
            return Err(VisionError::Model("the poses did not come back".into()));
        }
        Ok(boxes.into_iter().zip(poses).map(|(d, keypoints)| Person { bbox: d.bbox, score: d.score, keypoints }).collect())
    }

    /// The people whose poses are looked at: in the picture's pixels, tall enough, a box found
    /// twice suppressed, at most `max_people`.
    fn select(&self, found: &[Found], w: usize, h: usize) -> Vec<Detection> {
        let p = &self.params;
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
        boxes
    }

    /// The faces the face detector missed in an RGBA picture: the heads of the people whose
    /// face shows and is not one of `faces` (the face detector's, normalised), as face
    /// detections normalised to the picture ([`Detection::pose`]).
    pub fn heads(&self, rgba: &[u8], w: usize, h: usize, faces: &[Detection]) -> Result<Vec<Detection>, VisionError> {
        let people = self.people(rgba, w, h)?;
        Ok(head::candidates(&people, faces, &self.params.head, w as f32, h as f32))
    }
}

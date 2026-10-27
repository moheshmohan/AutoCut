//! How a 360° picture maps onto the sphere around the camera: what a file's metadata says (a
//! source's [`crate::MediaSource::spherical`]) and what Interpret Footage ▸ VR Properties keeps in
//! a project.
//!
//! Directions use one frame everywhere: +Z is forward (the middle of an equirectangular picture),
//! +X is to the right and +Y is up. Yaw turns to the right, pitch looks up and roll turns the
//! picture clockwise, all in degrees. The projection maths is `filmcraft_render::vr`.

use serde::{Deserialize, Serialize};

/// Most lenses a dual-fisheye description carries (a project file is not trusted).
pub const MAX_LENSES: usize = 2;
/// Most polynomial terms of a lens model.
pub const MAX_POLY: usize = 8;
/// Most samples of a horizon (nearly seven hours at ten a second).
pub const MAX_HORIZON: usize = 250_000;
/// Largest turn a horizon's steadying makes, degrees (a project file is not trusted).
pub const MAX_STEADY: f32 = 45.0;

/// A picture of the sphere.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Spherical {
    pub projection: Projection,
    /// How the two eyes of stereoscopic video share an equirectangular frame (the left eye's view
    /// is the one shown).
    #[serde(default)]
    pub layout: StereoLayout,
    /// The part of the sphere the picture covers, degrees across and down (360 × 180: all of
    /// it), centred on the forward direction.
    #[serde(default = "full_view")]
    pub captured: [f64; 2],
    /// The camera's orientation when the picture was made — yaw, pitch, roll — which the view
    /// starts from (`sv3d` ▸ `prhd`, V1's initial view).
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub pose: [f64; 3],
    /// The camera or tool that wrote the metadata (`Insta360 X5`), for display.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub camera: String,
    /// Which way was up through the recording, from the camera's own motion sensor: the sphere
    /// is turned at every moment so that up is up, and its heading steadied (a dual-fisheye
    /// picture; a stitched one was levelled when it was stitched).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon: Option<Horizon>,
}

fn full_view() -> [f64; 2] {
    [360.0, 180.0]
}

fn is_zero3(v: &[f64; 3]) -> bool {
    v.iter().all(|x| *x == 0.0)
}

impl Spherical {
    /// A whole-sphere monoscopic equirectangular picture.
    pub fn equirect() -> Self {
        Spherical {
            projection: Projection::Equirectangular,
            layout: StereoLayout::Mono,
            captured: full_view(),
            pose: [0.0; 3],
            camera: String::new(),
            horizon: None,
        }
    }

    /// A dual-fisheye picture of these lenses.
    pub fn dual_fisheye(lenses: Vec<FisheyeLens>, camera: &str) -> Self {
        Spherical {
            projection: Projection::DualFisheye { lenses },
            layout: StereoLayout::Mono,
            captured: full_view(),
            pose: [0.0; 3],
            camera: camera.to_string(),
            horizon: None,
        }
    }

    pub fn is_equirect(&self) -> bool {
        matches!(self.projection, Projection::Equirectangular)
    }

    /// Which way was up `seconds` into the recording, in the camera's own directions, when the
    /// picture is levelled ([`Horizon::up_at`]).
    pub fn up_at(&self, seconds: f64) -> Option<[f64; 3]> {
        self.horizon.as_ref()?.up_at(seconds)
    }

    /// The turn about the vertical that steadies the levelled picture `seconds` into the
    /// recording, degrees to the right ([`Horizon::heading_at`]).
    pub fn heading_at(&self, seconds: f64) -> Option<f64> {
        self.horizon.as_ref()?.heading_at(seconds)
    }

    /// The description with every number made usable: lenses capped and their values finite and
    /// in range, the captured view within the sphere. A project file or a camera can say anything.
    pub fn sanitized(&self) -> Self {
        let mut s = self.clone();
        let fin = |v: f64, d: f64| if v.is_finite() { v } else { d };
        s.captured = [fin(s.captured[0], 360.0).clamp(1.0, 360.0), fin(s.captured[1], 180.0).clamp(1.0, 180.0)];
        s.pose = s.pose.map(|v| fin(v, 0.0).clamp(-360.0, 360.0));
        s.horizon = s.horizon.as_ref().map(Horizon::sanitized);
        if let Projection::DualFisheye { lenses } = &mut s.projection {
            lenses.truncate(MAX_LENSES);
            for l in lenses.iter_mut() {
                *l = l.sanitized();
            }
        }
        s
    }
}

/// Which way was up through a recording, measured by the camera's motion sensor (an Insta360
/// recording's gravity and turning, `filmcraft_codecs`), which levels its horizon — and how the
/// camera swung about the vertical, which steadies its heading: the sway of a walk and the jolts of
/// a selfie stick taken out, the turns of the person holding it kept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Horizon {
    /// Whether the picture is levelled by it (VR Properties ▸ Level Horizon).
    #[serde(default = "yes")]
    pub level: bool,
    /// Whether the levelled picture's heading is steadied by it (VR Properties ▸ Steady).
    #[serde(default = "yes")]
    pub steady: bool,
    /// Seconds between samples, the first at the recording's first picture.
    pub step: f64,
    /// Up in the camera's own directions (+Z through the front lens, +Y up its body), a unit
    /// vector a sample.
    pub up: Vec<[f32; 3]>,
    /// The turn about the vertical that steadies the levelled picture, degrees to the right, a
    /// sample (as many as `up`; none: not steadied).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub heading: Vec<f32>,
}

fn yes() -> bool {
    true
}

impl Horizon {
    /// Up `seconds` into the recording, between samples blended from the nearer two, held before
    /// the first and after the last; `None` when levelling is off or there are no samples.
    pub fn up_at(&self, seconds: f64) -> Option<[f64; 3]> {
        if !self.level || !(self.step.is_finite() && self.step > 0.0) {
            return None;
        }
        let last = self.up.len().checked_sub(1)?;
        let x = if seconds.is_finite() { (seconds / self.step).clamp(0.0, last as f64) } else { 0.0 };
        let i = (x.floor() as usize).min(last);
        let f = x - i as f64;
        let a = self.up.get(i)?;
        let b = self.up.get((i + 1).min(last))?;
        let v: [f64; 3] = std::array::from_fn(|c| f64::from(a[c]) * (1.0 - f) + f64::from(b[c]) * f);
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (n > 1e-6 && n.is_finite()).then(|| v.map(|c| c / n))
    }

    /// The steadying turn `seconds` into the recording, degrees to the right, blended and held
    /// like [`Horizon::up_at`]; `None` when levelling or steadying is off or there are no samples.
    pub fn heading_at(&self, seconds: f64) -> Option<f64> {
        if !(self.level && self.steady) || !(self.step.is_finite() && self.step > 0.0) {
            return None;
        }
        let last = self.heading.len().checked_sub(1)?;
        let x = if seconds.is_finite() { (seconds / self.step).clamp(0.0, last as f64) } else { 0.0 };
        let i = (x.floor() as usize).min(last);
        let f = x - i as f64;
        let a = f64::from(*self.heading.get(i)?);
        let b = f64::from(*self.heading.get((i + 1).min(last))?);
        let v = a * (1.0 - f) + b * f;
        v.is_finite().then_some(v)
    }

    /// The largest steadying turn, degrees.
    pub fn max_steady(&self) -> f64 {
        self.heading.iter().map(|h| f64::from(h.abs())).fold(0.0, f64::max)
    }

    /// How long the samples run, seconds.
    pub fn seconds(&self) -> f64 {
        self.step * self.up.len().saturating_sub(1) as f64
    }

    /// The largest tilt from upright, degrees.
    pub fn max_tilt(&self) -> f64 {
        self.up.iter().map(|u| f64::from(u[1]).clamp(-1.0, 1.0).acos().to_degrees()).fold(0.0, f64::max)
    }

    /// Usable values: a step of a millisecond to a minute, at most [`MAX_HORIZON`] samples, each
    /// a finite unit vector (upright where it was not), and steadying turns finite and within
    /// [`MAX_STEADY`] (none where it was not).
    pub fn sanitized(&self) -> Self {
        let step = if self.step.is_finite() { self.step.clamp(0.001, 60.0) } else { 0.1 };
        let up = self
            .up
            .iter()
            .take(MAX_HORIZON)
            .map(|u| {
                let n = u.iter().map(|c| c * c).sum::<f32>().sqrt();
                if n.is_finite() && n > 1e-6 { u.map(|c| c / n) } else { [0.0, 1.0, 0.0] }
            })
            .collect();
        let heading = self.heading.iter().take(MAX_HORIZON).map(|h| if h.is_finite() { h.clamp(-MAX_STEADY, MAX_STEADY) } else { 0.0 }).collect();
        Horizon { level: self.level, steady: self.steady, step, up, heading }
    }
}

/// How the picture is laid out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Projection {
    /// Longitude across, latitude down (2:1 for the whole sphere).
    Equirectangular,
    /// Two (or one) circular fisheye pictures in one frame, each looking its own way, stitched
    /// where they overlap (an Insta360 `.insv` / `.lrv`, decoded side by side).
    DualFisheye { lenses: Vec<FisheyeLens> },
}

/// Stereoscopic layout of an equirectangular frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StereoLayout {
    #[default]
    Mono,
    /// Left eye on top.
    OverUnder,
    /// Left eye on the left.
    SideBySide,
}

/// One fisheye lens of a dual-fisheye frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FisheyeLens {
    /// The lens's picture in the frame: x, y, width, height as shares of the frame (0..1).
    pub rect: [f64; 4],
    /// The image circle's centre, shares of the lens picture's width and height.
    pub cx: f64,
    pub cy: f64,
    /// The image circle's radius, a share of the lens picture's width.
    pub radius: f64,
    /// The angle between the optical axis and the image circle's edge, degrees (half the lens's
    /// field of view; about 100° for a 360° camera's lenses).
    pub half_fov: f64,
    /// Radial distortion: with t = θ / half_fov, the distance from the centre is
    /// radius · Σ poly[i] · t^(i+1). Empty: equidistant (radius · t).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub poly: Vec<f64>,
    /// Where the lens looks: yaw (0 forward, 180 backward), pitch, and roll of its picture,
    /// degrees.
    pub yaw: f64,
    #[serde(default)]
    pub pitch: f64,
    #[serde(default)]
    pub roll: f64,
}

impl FisheyeLens {
    /// Values made usable (see [`Spherical::sanitized`]): a rectangle inside the frame, a
    /// positive radius, a half field of view between 30° and 135°, finite angles, a capped
    /// polynomial.
    pub fn sanitized(&self) -> Self {
        let fin = |v: f64, d: f64| if v.is_finite() { v } else { d };
        let mut l = self.clone();
        let x = fin(l.rect[0], 0.0).clamp(0.0, 1.0);
        let y = fin(l.rect[1], 0.0).clamp(0.0, 1.0);
        let w = fin(l.rect[2], 1.0).clamp(0.0, 1.0 - x);
        let h = fin(l.rect[3], 1.0).clamp(0.0, 1.0 - y);
        l.rect = [x, y, w, h];
        l.cx = fin(l.cx, 0.5).clamp(-1.0, 2.0);
        l.cy = fin(l.cy, 0.5).clamp(-1.0, 2.0);
        l.radius = fin(l.radius, 0.5).clamp(1e-3, 2.0);
        l.half_fov = fin(l.half_fov, 100.0).clamp(30.0, 135.0);
        l.poly.truncate(MAX_POLY);
        for c in &mut l.poly {
            *c = fin(*c, 0.0).clamp(-100.0, 100.0);
        }
        l.yaw = fin(l.yaw, 0.0).clamp(-360.0, 360.0);
        l.pitch = fin(l.pitch, 0.0).clamp(-360.0, 360.0);
        l.roll = fin(l.roll, 0.0).clamp(-360.0, 360.0);
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip_and_defaults() {
        let lens = FisheyeLens { rect: [0.0, 0.0, 0.5, 1.0], cx: 0.5, cy: 0.5, radius: 0.49, half_fov: 100.0, poly: vec![], yaw: 0.0, pitch: 0.0, roll: 0.0 };
        let s = Spherical::dual_fisheye(vec![lens.clone(), FisheyeLens { yaw: 180.0, rect: [0.5, 0.0, 0.5, 1.0], ..lens }], "Insta360 X5");
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Spherical>(&j).unwrap(), s);
        let e: Spherical = serde_json::from_str(r#"{"projection":{"kind":"equirectangular"}}"#).unwrap();
        assert_eq!(e, Spherical::equirect());
        // a horizon goes with it; levelling is on unless it says otherwise
        let h = Horizon { level: true, steady: true, step: 0.1, up: vec![[0.0, 1.0, 0.0], [0.5, 0.866, 0.0]], heading: vec![0.0, -2.5] };
        let lev = Spherical { horizon: Some(h.clone()), ..s };
        let j = serde_json::to_string(&lev).unwrap();
        assert_eq!(serde_json::from_str::<Spherical>(&j).unwrap(), lev);
        // (a project saved before steadying: levelled, and nothing to steady by)
        let on: Horizon = serde_json::from_str(r#"{"step":0.1,"up":[[0,1,0]]}"#).unwrap();
        assert!(on.level && on.steady && on.heading.is_empty() && on.heading_at(0.0).is_none());
    }

    #[test]
    fn up_is_blended_between_samples_and_held_at_the_ends() {
        let h = Horizon { level: true, steady: true, step: 0.5, up: vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], heading: vec![0.0, 4.0] };
        let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|c| (a[c] - b[c]).abs() < 1e-6);
        assert!(close(h.up_at(-3.0).unwrap(), [0.0, 1.0, 0.0]));
        assert!(close(h.up_at(0.25).unwrap(), [0.5f64.sqrt(), 0.5f64.sqrt(), 0.0]));
        assert!(close(h.up_at(9.0).unwrap(), [1.0, 0.0, 0.0]));
        assert!(close(h.up_at(f64::NAN).unwrap(), [0.0, 1.0, 0.0]));
        assert_eq!((h.seconds(), h.max_tilt().round()), (0.5, 90.0));
        assert!(Horizon { level: false, ..h.clone() }.up_at(0.0).is_none());
        assert!(Horizon { up: vec![], ..h.clone() }.up_at(0.0).is_none());
        // the steadying turn the same way; only with levelling
        assert_eq!((h.heading_at(-1.0), h.heading_at(0.25), h.heading_at(7.0), h.max_steady()), (Some(0.0), Some(2.0), Some(4.0), 4.0));
        assert!(Horizon { steady: false, ..h.clone() }.heading_at(0.25).is_none() && Horizon { level: false, ..h.clone() }.heading_at(0.25).is_none());
        let s = Spherical { horizon: Some(h), ..Spherical::equirect() };
        assert!(s.up_at(0.0).is_some() && Spherical::equirect().up_at(0.0).is_none());
    }

    #[test]
    fn sanitizing_tames_hostile_values() {
        let bad = FisheyeLens {
            rect: [f64::NAN, 2.0, 5.0, -1.0],
            cx: f64::INFINITY,
            cy: -9.0,
            radius: 0.0,
            half_fov: 1e9,
            poly: vec![f64::NAN; 50],
            yaw: f64::NAN,
            pitch: 1e30,
            roll: -1e30,
        };
        let mut s = Spherical::dual_fisheye(vec![bad.clone(), bad.clone(), bad], "");
        s.captured = [f64::NAN, -5.0];
        s.horizon = Some(Horizon {
            level: true,
            steady: true,
            step: f64::NAN,
            up: [[f32::NAN, 0.0, 1.0], [0.0; 3], [0.0, 2.0, 0.0]].repeat(MAX_HORIZON / 3 + 5),
            heading: [f32::NAN, 1e9, -3.0].repeat(MAX_HORIZON / 3 + 5),
        });
        let s = s.sanitized();
        assert_eq!(s.captured, [360.0, 1.0]);
        let h = s.horizon.as_ref().unwrap();
        assert_eq!((h.step, h.up.len()), (0.1, MAX_HORIZON));
        assert_eq!(&h.up[..3], &[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!((h.heading.len(), &h.heading[..3]), (MAX_HORIZON, &[0.0, MAX_STEADY, -3.0][..]));
        let Projection::DualFisheye { lenses } = &s.projection else { panic!() };
        assert_eq!(lenses.len(), MAX_LENSES);
        for l in lenses {
            assert!(l.rect.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!(l.rect[0] + l.rect[2] <= 1.0 && l.rect[1] + l.rect[3] <= 1.0);
            assert!(l.radius > 0.0 && (30.0..=135.0).contains(&l.half_fov));
            assert_eq!(l.poly.len(), MAX_POLY);
            assert!(l.poly.iter().chain([&l.cx, &l.cy, &l.yaw, &l.pitch, &l.roll]).all(|v| v.is_finite()));
        }
    }
}

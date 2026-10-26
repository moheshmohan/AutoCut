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
        Spherical { projection: Projection::Equirectangular, layout: StereoLayout::Mono, captured: full_view(), pose: [0.0; 3], camera: String::new() }
    }

    /// A dual-fisheye picture of these lenses.
    pub fn dual_fisheye(lenses: Vec<FisheyeLens>, camera: &str) -> Self {
        Spherical {
            projection: Projection::DualFisheye { lenses },
            layout: StereoLayout::Mono,
            captured: full_view(),
            pose: [0.0; 3],
            camera: camera.to_string(),
        }
    }

    pub fn is_equirect(&self) -> bool {
        matches!(self.projection, Projection::Equirectangular)
    }

    /// The description with every number made usable: lenses capped and their values finite and
    /// in range, the captured view within the sphere. A project file or a camera can say anything.
    pub fn sanitized(&self) -> Self {
        let mut s = self.clone();
        let fin = |v: f64, d: f64| if v.is_finite() { v } else { d };
        s.captured = [fin(s.captured[0], 360.0).clamp(1.0, 360.0), fin(s.captured[1], 180.0).clamp(1.0, 180.0)];
        s.pose = s.pose.map(|v| fin(v, 0.0).clamp(-360.0, 360.0));
        if let Projection::DualFisheye { lenses } = &mut s.projection {
            lenses.truncate(MAX_LENSES);
            for l in lenses.iter_mut() {
                *l = l.sanitized();
            }
        }
        s
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
        let s = s.sanitized();
        assert_eq!(s.captured, [360.0, 1.0]);
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

//! Spherical (360°) video metadata, from Google's Spherical Video RFCs (public, in the
//! `google/spatial-media` repository's `docs/`):
//!
//! - **V2** — boxes in a visual sample entry: `st3d` (Stereoscopic 3D Video: a full box with one
//!   byte, the stereo mode) and `sv3d` (Spherical Video), a container of `svhd` (a full box with
//!   the metadata source, a null-terminated string) and `proj`, itself holding `prhd` (a full box
//!   with the pose: yaw, pitch and roll as signed 16.16 fixed-point degrees) and the projection:
//!   `equi` (a full box with the bounds cut off the top, bottom, left and right of the sphere as
//!   0.32 fixed-point shares of it), `cbmp` (a full box with the cube map layout and padding), or
//!   another (`mshp`, a mesh), kept by its type.
//! - **V1** — a `uuid` box in the `trak` (UUID `ffcc8263-f855-4a93-8814-587a02521fdd`) holding
//!   RDF/XML in the `GSpherical` namespace: `Spherical`, `Stitched`, `ProjectionType`,
//!   `StereoMode`, the initial view and the cropped area of a partial panorama.
//!
//! Every value comes from a file: sizes are checked, strings are capped, and what cannot be read
//! is left out rather than failing the whole file.

use crate::bytes::{BoxBuf, Cur, FourCc, boxes};
use crate::error::Result;

/// The `uuid` of a Spherical Video V1 box.
pub const SPHERICAL_V1_UUID: [u8; 16] = [0xff, 0xcc, 0x82, 0x63, 0xf8, 0x55, 0x4a, 0x93, 0x88, 0x14, 0x58, 0x7a, 0x02, 0x52, 0x1f, 0xdd];

/// Longest metadata source or V1 XML kept, bytes.
const MAX_TEXT: usize = 64 * 1024;

/// How the two eyes of stereoscopic video share a frame (`st3d`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoMode {
    Mono,
    /// Left eye on top.
    TopBottom,
    /// Left eye on the left.
    LeftRight,
    /// Described elsewhere (a mesh).
    Custom,
    /// Right eye on the left.
    RightLeft,
    Other(u8),
}

impl StereoMode {
    pub fn from_code(c: u8) -> Self {
        match c {
            0 => StereoMode::Mono,
            1 => StereoMode::TopBottom,
            2 => StereoMode::LeftRight,
            3 => StereoMode::Custom,
            4 => StereoMode::RightLeft,
            o => StereoMode::Other(o),
        }
    }
    pub fn code(self) -> u8 {
        match self {
            StereoMode::Mono => 0,
            StereoMode::TopBottom => 1,
            StereoMode::LeftRight => 2,
            StereoMode::Custom => 3,
            StereoMode::RightLeft => 4,
            StereoMode::Other(o) => o,
        }
    }
}

/// The projection of `sv3d` ▸ `proj`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// `equi`: the shares of the sphere cut off each side, 0.32 fixed point (all 0: the whole
    /// sphere).
    Equirect { top: u32, bottom: u32, left: u32, right: u32 },
    /// `cbmp`: cube map layout (0: the RFC's 3×2) and padding pixels.
    Cubemap { layout: u32, padding: u32 },
    /// Another projection box, by its type.
    Other(FourCc),
}

/// The `sv3d` box.
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalBox {
    /// `svhd`: the tool that wrote the metadata.
    pub source: String,
    /// `prhd`: yaw, pitch, roll in degrees.
    pub pose: [f64; 3],
    pub projection: Option<Projection>,
}

impl SphericalBox {
    /// Whole-sphere equirectangular metadata written by `source`.
    pub fn equirect(source: &str) -> Self {
        SphericalBox { source: source.to_string(), pose: [0.0; 3], projection: Some(Projection::Equirect { top: 0, bottom: 0, left: 0, right: 0 }) }
    }

    pub(crate) fn parse(p: &[u8]) -> Result<Self> {
        let mut out = SphericalBox { source: String::new(), pose: [0.0; 3], projection: None };
        for b in boxes(p).map_while(|b| b.ok()) {
            match &b.kind.0 {
                b"svhd" => {
                    let mut c = Cur::new(b.payload, "svhd");
                    c.full_header()?;
                    out.source = c_string(c.rest());
                }
                b"proj" => {
                    for q in boxes(b.payload).map_while(|b| b.ok()) {
                        let mut c = Cur::new(q.payload, "proj");
                        match &q.kind.0 {
                            b"prhd" => {
                                c.full_header()?;
                                let fixed = |v: i32| f64::from(v) / 65536.0;
                                out.pose = [fixed(c.i32()?), fixed(c.i32()?), fixed(c.i32()?)];
                            }
                            b"equi" => {
                                c.full_header()?;
                                out.projection = Some(Projection::Equirect { top: c.u32()?, bottom: c.u32()?, left: c.u32()?, right: c.u32()? });
                            }
                            b"cbmp" => {
                                c.full_header()?;
                                out.projection = Some(Projection::Cubemap { layout: c.u32()?, padding: c.u32()? });
                            }
                            _ if out.projection.is_none() => out.projection = Some(Projection::Other(q.kind)),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }

    pub(crate) fn write(&self, b: &mut BoxBuf) {
        let sv3d = b.start(b"sv3d");
        let h = b.start_full(b"svhd", 0, 0);
        b.bytes(self.source.as_bytes().get(..self.source.len().min(255)).unwrap_or_default());
        b.u8(0);
        b.end(h);
        let proj = b.start(b"proj");
        let ph = b.start_full(b"prhd", 0, 0);
        for v in self.pose {
            b.i32(to_fixed(v));
        }
        b.end(ph);
        match self.projection {
            Some(Projection::Equirect { top, bottom, left, right }) => {
                let e = b.start_full(b"equi", 0, 0);
                for v in [top, bottom, left, right] {
                    b.u32(v);
                }
                b.end(e);
            }
            Some(Projection::Cubemap { layout, padding }) => {
                let e = b.start_full(b"cbmp", 0, 0);
                b.u32(layout);
                b.u32(padding);
                b.end(e);
            }
            // a projection we only know by name has no payload to write
            Some(Projection::Other(_)) | None => {}
        }
        b.end(proj);
        b.end(sv3d);
    }
}

/// Degrees as signed 16.16 fixed point, saturating (a pose is within ±180° anyway).
fn to_fixed(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    (v * 65536.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

/// A null-terminated (or unterminated) UTF-8 string, lossy and capped.
fn c_string(p: &[u8]) -> String {
    let end = p.iter().position(|&c| c == 0).unwrap_or(p.len()).min(MAX_TEXT);
    String::from_utf8_lossy(p.get(..end).unwrap_or_default()).into_owned()
}

/// The `st3d` payload's stereo mode.
pub(crate) fn parse_st3d(p: &[u8]) -> Result<StereoMode> {
    let mut c = Cur::new(p, "st3d");
    c.full_header()?;
    Ok(StereoMode::from_code(c.u8()?))
}

pub(crate) fn write_st3d(b: &mut BoxBuf, mode: StereoMode) {
    let m = b.start_full(b"st3d", 0, 0);
    b.u8(mode.code());
    b.end(m);
}

/// What a Spherical Video V1 `uuid` box says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SphericalV1 {
    pub spherical: bool,
    pub stitched: bool,
    /// `equirectangular` in every file the RFC describes.
    pub projection: String,
    /// `mono`, `top-bottom` or `left-right` (empty: mono).
    pub stereo: String,
    pub stitching_software: String,
    /// Initial view heading, pitch and roll, degrees.
    pub initial_view: [f64; 3],
    /// A partial panorama: the whole panorama's size and where the picture sits in it
    /// (`[full width, full height, cropped width, cropped height, left, top]`, pixels).
    pub cropped: Option<[u64; 6]>,
}

impl SphericalV1 {
    /// Reads the RDF/XML (tags in the `GSpherical` namespace; their prefix may differ).
    pub fn parse(xml: &[u8]) -> Option<Self> {
        let xml = xml.get(..xml.len().min(MAX_TEXT))?;
        let s = String::from_utf8_lossy(xml);
        let tag = |name: &str| xml_value(&s, name);
        let spherical = tag("Spherical")?.eq_ignore_ascii_case("true");
        let num = |name: &str| tag(name).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite());
        let int = |name: &str| tag(name).and_then(|v| v.trim().parse::<u64>().ok());
        let cropped = (|| {
            Some([
                int("FullPanoWidthPixels")?,
                int("FullPanoHeightPixels")?,
                int("CroppedAreaImageWidthPixels")?,
                int("CroppedAreaImageHeightPixels")?,
                int("CroppedAreaLeftPixels")?,
                int("CroppedAreaTopPixels")?,
            ])
        })();
        Some(SphericalV1 {
            spherical,
            stitched: tag("Stitched").is_some_and(|v| v.eq_ignore_ascii_case("true")),
            projection: tag("ProjectionType").unwrap_or_default().to_ascii_lowercase(),
            stereo: tag("StereoMode").unwrap_or_default().to_ascii_lowercase(),
            stitching_software: tag("StitchingSoftware").unwrap_or_default(),
            initial_view: [
                num("InitialViewHeadingDegrees").unwrap_or(0.0),
                num("InitialViewPitchDegrees").unwrap_or(0.0),
                num("InitialViewRollDegrees").unwrap_or(0.0),
            ],
            cropped,
        })
    }

    /// The RDF/XML of a whole-sphere, stitched equirectangular video (`stereo`: `mono`,
    /// `top-bottom` or `left-right`).
    pub fn equirect_xml(software: &str, stereo: &str) -> String {
        let esc = |v: &str| v.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
        let mut x = String::from(
            "<?xml version=\"1.0\"?><rdf:SphericalVideo xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:GSpherical=\"http://ns.google.com/videos/1.0/spherical/\">",
        );
        x.push_str("<GSpherical:Spherical>true</GSpherical:Spherical><GSpherical:Stitched>true</GSpherical:Stitched>");
        x.push_str(&format!("<GSpherical:StitchingSoftware>{}</GSpherical:StitchingSoftware>", esc(software)));
        x.push_str("<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>");
        if stereo != "mono" && !stereo.is_empty() {
            x.push_str(&format!("<GSpherical:StereoMode>{}</GSpherical:StereoMode>", esc(stereo)));
        }
        x.push_str("</rdf:SphericalVideo>");
        x
    }
}

/// The text of the first element whose local name is `name` (`<p:name>text</p:name>` with any
/// prefix, or none), trimmed.
fn xml_value(s: &str, name: &str) -> Option<String> {
    let mut from = 0;
    while let Some(i) = s.get(from..)?.find('<') {
        let open = from + i + 1;
        let close = open + s.get(open..)?.find('>')?;
        let tag = s.get(open..close)?;
        from = close + 1;
        if tag.starts_with('/') || tag.starts_with('?') || tag.ends_with('/') {
            continue;
        }
        let tname = tag.split_whitespace().next().unwrap_or_default();
        let local = tname.rsplit(':').next().unwrap_or(tname);
        if local != name {
            continue;
        }
        let end = from + s.get(from..)?.find('<')?;
        return Some(s.get(from..end)?.trim().to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sv3d_round_trips() {
        let mut b = BoxBuf::new();
        let s = SphericalBox {
            source: "FilmCraft".into(),
            pose: [10.5, -3.25, 0.0],
            projection: Some(Projection::Equirect { top: 1, bottom: 2, left: 3, right: 4 }),
        };
        s.write(&mut b);
        let sv3d = boxes(&b.buf).next().unwrap().unwrap();
        assert_eq!(sv3d.kind, b"sv3d");
        assert_eq!(SphericalBox::parse(sv3d.payload).unwrap(), s);
    }

    #[test]
    fn st3d_round_trips() {
        let mut b = BoxBuf::new();
        write_st3d(&mut b, StereoMode::TopBottom);
        let st = boxes(&b.buf).next().unwrap().unwrap();
        assert_eq!(parse_st3d(st.payload).unwrap(), StereoMode::TopBottom);
    }

    #[test]
    fn v1_xml_reads_back() {
        let x = SphericalV1::equirect_xml("FilmCraft <test>", "top-bottom");
        let v = SphericalV1::parse(x.as_bytes()).unwrap();
        assert!(v.spherical && v.stitched);
        assert_eq!(v.projection, "equirectangular");
        assert_eq!(v.stereo, "top-bottom");
        assert_eq!(v.stitching_software, "FilmCraft &lt;test&gt;");
        assert_eq!(v.cropped, None);
    }

    #[test]
    fn v1_xml_with_other_prefixes_and_crops() {
        let x = "<rdf:SphericalVideo><G:Spherical>true</G:Spherical><G:ProjectionType> Equirectangular </G:ProjectionType>\
                 <G:FullPanoWidthPixels>4096</G:FullPanoWidthPixels><G:FullPanoHeightPixels>2048</G:FullPanoHeightPixels>\
                 <G:CroppedAreaImageWidthPixels>4096</G:CroppedAreaImageWidthPixels><G:CroppedAreaImageHeightPixels>1024</G:CroppedAreaImageHeightPixels>\
                 <G:CroppedAreaLeftPixels>0</G:CroppedAreaLeftPixels><G:CroppedAreaTopPixels>512</G:CroppedAreaTopPixels>\
                 <G:InitialViewHeadingDegrees>90</G:InitialViewHeadingDegrees></rdf:SphericalVideo>";
        let v = SphericalV1::parse(x.as_bytes()).unwrap();
        assert_eq!(v.projection, "equirectangular");
        assert_eq!(v.cropped, Some([4096, 2048, 4096, 1024, 0, 512]));
        assert_eq!(v.initial_view, [90.0, 0.0, 0.0]);
    }

    #[test]
    fn hostile_payloads_never_panic() {
        let mut b = BoxBuf::new();
        SphericalBox::equirect("x").write(&mut b);
        let full = b.buf.clone();
        for n in 0..full.len() {
            let _ = SphericalBox::parse(&full[..n]);
            let mut flipped = full.clone();
            flipped[n] ^= 0xa5;
            let _ = SphericalBox::parse(&flipped);
            let _ = SphericalV1::parse(&full[..n]);
        }
        assert!(SphericalV1::parse(b"<a><Spherical>").is_none());
        assert!(SphericalV1::parse(b"no xml at all").is_none());
        assert!(parse_st3d(&[0, 0]).is_err());
    }
}

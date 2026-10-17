//! Shader transitions: GL Transitions hand-ported to WGSL, each with a Rust reference.
//!
//! [GL Transitions](https://github.com/gl-transitions/gl-transitions) (MIT) is the common
//! currency of shader transitions: a GLSL function `transition(uv)` of `progress`,
//! `getFromColor`, `getToColor` and a few parameters. Each port here has three parts:
//!
//! - `shader/<name>.wgsl`: the WGSL body, keeping the upstream `Author:` / `License:` header (MIT
//!   only; licence text in `shader/LICENSE-gl-transitions.txt`). The host prelude
//!   (`shader/prelude.wgsl`) supplies the bindings, `progress`, `ratio`, the clamped bilinear
//!   `getFromColor` / `getToColor` and the GLSL helpers WGSL lacks (`glsl_mod`, an
//!   order-independent `smooth_step`, `atan2_safe`, an integer `hash` and its per-pixel,
//!   per-cell, per-column and per-row forms, `in_bounds`).
//! - [`cpu`]: the same function in Rust, operation for operation: the reference the GPU is tested
//!   against, and what renders the transition when no GPU (or no working pipeline) is there.
//!   Preview and export both draw with the GPU stage when they can (`filmcraft-gpu`), so they
//!   show the same picture.
//! - A parameter packer turning the catalogue's parameters
//!   ([`filmcraft_project::gltransition`]) into the 12 shader floats, clamping hostile values to
//!   the catalogue's ranges (a shader needing more would extend `Host` with another vec4).
//!
//! Shaders that need inputs the stage does not give yet stay out: `displacement` and `luma` (an
//! extra grayscale texture: a third binding and an asset or a procedural map), and the BSD-licensed
//! `InvertedPageCurl` and `StereoViewer` (MIT only).
//!
//! **The host owns time.** [`eased_progress`] applies the transition's easing (Penner curves or a
//! cubic Bézier, [`super::easing`]) to the progress the timeline computed at the exact frame
//! `Tick`, on the CPU; the CPU reference and the GPU uniforms both use that value, so preview and
//! export agree. At an eased progress of 0 or 1 both paths return the input frames unchanged.
//!
//! **Validation.** [`registry`] parses and validates every module with naga once, collecting
//! errors instead of panicking; the GPU stage (`filmcraft-gpu`) only compiles sources the
//! registry accepted, and renders a shader that fails anyway on the CPU.
//!
//! **Exact pixel positions.** A pixel's uv is its centre times 1 / width and 1 / height, rounded
//! once on the CPU and passed in the uniforms (`Host::inv`): one correctly rounded
//! multiplication, so the reference and every GPU see the same uv (a GPU division may be
//! approximate). GPU compilers may still reassociate arithmetic, so a hard edge that a pixel
//! centre can sit on exactly (uv.x = 0.5 on an odd width, the edge between two cells) gets a
//! small bias in both versions (see BookFlip, windowslice, pixelize).
//!
//! **Porting another transition.**
//! 1. Read the upstream file's own `// License:` header: port MIT only (no header means the
//!    repository's MIT licence; anything else stays out).
//! 2. Copy it to `shader/<UpstreamName>.wgsl`, keep the header, add the upstream link, and
//!    translate: `vec2`→`vec2<f32>`, `float(x)`→`f32(x)`, explicit `.0` literals, `atan(y, x)`→
//!    `atan2_safe`, `mod`→`glsl_mod`, `smoothstep`→`smooth_step`, `mix(a, b, t)` with a vector `t`,
//!    ternaries→`select(f, t, cond)` (both sides are evaluated), `fract(sin(…) * 43758.5453)`→
//!    `rand_pixel` / `rand_cell` (or `rand_column` / `rand_row` for `rand(vec2(uv.x, k))`, the
//!    `_seeded` forms for a seed parameter: never a `sin`-based hash, whose value depends on the
//!    GPU's precision), uniforms→`host.p0..p2` (vec4 slots: no vec3 padding to worry about).
//!    `smoothstep(a, b, x)` with `b < a` (undefined in GLSL, a falling curve on real GPUs)→
//!    `1.0 - smooth_step(b, a, x)`. Rename identifiers WGSL reserves (`from`, `mod`, …).
//!    Sampling is `textureLoad`-based, so it may sit in non-uniform control flow (no
//!    `textureSample`, whose derivatives need uniform flow). Float→int: `i32()` truncates as in
//!    GLSL. Where a `floor` decides a cell, bias it by 0.001 cell so ties fall the same way on
//!    the CPU and every GPU (see pixelize, Mosaic). Upstream's opaque-canvas habits (an alpha
//!    above 1 that an 8-bit target clamps) are clamped to 1; constants such as `sqrt(3.0) / 3.0`
//!    are written out, so both versions round them alike; a `cos` / `sin` far from zero on a
//!    fixed set of angles becomes a table (FilmBurn).
//! 3. Write the Rust reference in [`cpu`] operation for operation (same order, `mix` as
//!    `a·(1−t)+b·t`), add a [`ShaderDef`] here and its parameters (upstream names, units and
//!    defaults) in `filmcraft_project::gltransition`, a `<file>.attribution` sidecar and an
//!    `ATTRIBUTION.md` row, Japanese and Spanish names, then bless `transitions/goldens.txt` and
//!    the golden sheet. The generic transition tests and the GPU parity test pick it up.

use std::sync::OnceLock;

use filmcraft_project::{EffectInstance, ParamKind, ParamValue};
use filmcraft_time::Tick;

use super::easing::{self, Bezier};
use super::{Tx, paint};
use crate::image::Image;

mod cpu;
#[cfg(test)]
mod tests;

use cpu::{Ctx, V2};

/// Number of shader parameter floats (`host.p0`..`host.p2`).
pub const PARAMS: usize = 12;
/// Uniform floats: width, height, eased progress, ratio, the parameters, then 1 / width,
/// 1 / height and two zeros (80 bytes).
pub const UNIFORM_FLOATS: usize = 4 + PARAMS + 4;

/// The host prelude every shader body is appended to.
pub const PRELUDE: &str = include_str!("shader/prelude.wgsl");

/// One ported transition.
pub struct ShaderDef {
    /// Effect id in the catalogue (`gl_…`).
    pub id: &'static str,
    /// The transition's name in the GL Transitions collection.
    pub upstream: &'static str,
    /// Upstream `Author:` header.
    pub author: &'static str,
    /// Upstream `License:` header (SPDX id).
    pub license: &'static str,
    /// The WGSL body (`transition(uv)` and its helpers).
    pub body: &'static str,
    params: fn(&Params) -> [f32; PARAMS],
    cpu: fn(&Ctx, V2) -> [f32; 4],
}

impl ShaderDef {
    /// The complete WGSL module: the host prelude, then the body.
    pub fn wgsl(&self) -> String {
        format!("{PRELUDE}\n// ---- {} ----\n{}", self.upstream, self.body)
    }
}

impl std::fmt::Debug for ShaderDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderDef").field("id", &self.id).field("upstream", &self.upstream).finish_non_exhaustive()
    }
}

const fn def(
    id: &'static str,
    upstream: &'static str,
    author: &'static str,
    body: &'static str,
    params: fn(&Params) -> [f32; PARAMS],
    cpu: fn(&Ctx, V2) -> [f32; 4],
) -> ShaderDef {
    ShaderDef { id, upstream, author, license: "MIT", body, params, cpu }
}

/// Every ported transition, sorted by id.
static SHADERS: &[ShaderDef] = &[
    def("gl_angular", "angular", "Fernando Kuteken", include_str!("shader/angular.wgsl"), |p| pack(&[p.num("starting_angle")]), cpu::angular),
    def("gl_bookflip", "BookFlip", "hong", include_str!("shader/BookFlip.wgsl"), |_| [0.0; PARAMS], cpu::book_flip),
    def(
        "gl_bounce",
        "Bounce",
        "Adrian Purser",
        include_str!("shader/Bounce.wgsl"),
        |p| {
            let c = p.color("shadow_colour");
            pack(&[c[0], c[1], c[2], p.num("shadow_alpha"), p.num("shadow_height").max(0.01), p.num("bounces")])
        },
        cpu::bounce,
    ),
    def(
        "gl_burn",
        "burn",
        "gre",
        include_str!("shader/burn.wgsl"),
        |p| {
            let c = p.color("color");
            pack(&[c[0], c[1], c[2]])
        },
        cpu::burn,
    ),
    def(
        "gl_butterflywavescrawler",
        "ButterflyWaveScrawler",
        "mandubian",
        include_str!("shader/ButterflyWaveScrawler.wgsl"),
        |p| pack(&[p.num("amplitude"), p.num("waves"), p.num("color_separation")]),
        cpu::butterfly_wave_scrawler,
    ),
    def("gl_chessboard", "chessboard", "lql", include_str!("shader/chessboard.wgsl"), |p| pack(&[p.num("grid_num").max(1.0)]), cpu::chessboard),
    def("gl_circleopen", "circleopen", "gre", include_str!("shader/circleopen.wgsl"), |p| pack(&[p.num("smoothness"), p.bit("opening")]), cpu::circle_open),
    def(
        "gl_colorphase",
        "colorphase",
        "gre",
        include_str!("shader/colorphase.wgsl"),
        |p| {
            let n = |id| p.num(id);
            // alpha: upstream's default steps, 0 → 1
            pack(&[n("from_step_r"), n("from_step_g"), n("from_step_b"), 0.0, n("to_step_r"), n("to_step_g"), n("to_step_b"), 1.0])
        },
        cpu::colorphase,
    ),
    def(
        "gl_crosshatch",
        "crosshatch",
        "pthrasher",
        include_str!("shader/crosshatch.wgsl"),
        |p| {
            let (cx, cy) = p.center_uv();
            pack(&[cx, cy, p.num("threshold").max(0.01), p.num("fade_edge")])
        },
        cpu::crosshatch,
    ),
    def("gl_crosswarp", "crosswarp", "Eke Péter", include_str!("shader/crosswarp.wgsl"), |_| [0.0; PARAMS], cpu::crosswarp),
    def("gl_crosszoom", "CrossZoom", "rectalogic", include_str!("shader/CrossZoom.wgsl"), |p| pack(&[p.num("strength")]), cpu::cross_zoom),
    def(
        "gl_cube",
        "cube",
        "gre",
        include_str!("shader/cube.wgsl"),
        |p| pack(&[p.num("persp"), p.num("unzoom"), p.num("reflection"), p.num("floating")]),
        cpu::cube,
    ),
    def(
        "gl_directionalwarp",
        "directionalwarp",
        "pschroen",
        include_str!("shader/directionalwarp.wgsl"),
        |p| {
            let (x, y) = (p.num("direction_x"), p.num("direction_y"));
            // a zero direction has no normal: upstream's default instead
            let (x, y) = if x.hypot(y) < 1e-6 { (-1.0, 1.0) } else { (x, y) };
            pack(&[p.num("smoothness"), x, y])
        },
        cpu::directional_warp,
    ),
    def(
        "gl_doorway",
        "doorway",
        "gre",
        include_str!("shader/doorway.wgsl"),
        |p| pack(&[p.num("reflection"), p.num("perspective"), p.num("depth")]),
        cpu::doorway,
    ),
    def(
        "gl_dreamyzoom",
        "DreamyZoom",
        "Zeh Fernando",
        include_str!("shader/DreamyZoom.wgsl"),
        |p| pack(&[p.num("rotation"), p.num("scale")]),
        cpu::dreamy_zoom,
    ),
    def("gl_filmburn", "FilmBurn", "Anastasia Dunbar", include_str!("shader/FilmBurn.wgsl"), |p| pack(&[p.num("seed")]), cpu::film_burn),
    def("gl_glitchdisplace", "GlitchDisplace", "Matt DesLauriers", include_str!("shader/GlitchDisplace.wgsl"), |_| [0.0; PARAMS], cpu::glitch_displace),
    def("gl_glitchmemories", "GlitchMemories", "Gunnar Roth", include_str!("shader/GlitchMemories.wgsl"), |_| [0.0; PARAMS], cpu::glitch_memories),
    def(
        "gl_hexagonalize",
        "hexagonalize",
        "Fernando Kuteken",
        include_str!("shader/hexagonalize.wgsl"),
        |p| pack(&[p.whole("steps", 0.0), p.num("horizontal_hexagons").max(1.0)]),
        cpu::hexagonalize,
    ),
    def(
        "gl_kaleidoscope",
        "kaleidoscope",
        "nwoeanhinnogaehr",
        include_str!("shader/kaleidoscope.wgsl"),
        |p| pack(&[p.num("speed"), p.num("angle"), p.num("power")]),
        cpu::kaleidoscope,
    ),
    def(
        "gl_luminance_melt",
        "luminance_melt",
        "0gust1",
        include_str!("shader/luminance_melt.wgsl"),
        |p| pack(&[p.bit("melt_down"), p.num("l_threshold"), p.bit("above")]),
        cpu::luminance_melt,
    ),
    def("gl_mosaic", "Mosaic", "Xaychru", include_str!("shader/Mosaic.wgsl"), |p| pack(&[p.num("endx").round(), p.num("endy").round()]), cpu::mosaic),
    def("gl_perlin", "perlin", "Rich Harris", include_str!("shader/perlin.wgsl"), |p| pack(&[p.num("scale"), p.num("smoothness"), p.num("seed")]), cpu::perlin),
    def("gl_pinwheel", "pinwheel", "Mr Speaker", include_str!("shader/pinwheel.wgsl"), |p| pack(&[p.num("speed")]), cpu::pinwheel),
    def(
        "gl_pixelize",
        "pixelize",
        "gre",
        include_str!("shader/pixelize.wgsl"),
        |p| pack(&[p.whole("squares_min_x", 1.0), p.whole("squares_min_y", 1.0), p.whole("steps", 0.0)]),
        cpu::pixelize,
    ),
    def(
        "gl_polar_function",
        "polar_function",
        "Fernando Kuteken",
        include_str!("shader/polar_function.wgsl"),
        |p| pack(&[p.whole("segments", 0.0)]),
        cpu::polar_function,
    ),
    def(
        "gl_polkadotscurtain",
        "PolkaDotsCurtain",
        "bobylito",
        include_str!("shader/PolkaDotsCurtain.wgsl"),
        |p| {
            // upstream's default centre is the bottom left corner
            let (cx, cy) = p.point_uv("center", (0.0, 0.0));
            pack(&[p.num("dots").max(1.0), cx, cy])
        },
        cpu::polka_dots_curtain,
    ),
    def(
        "gl_randomsquares",
        "randomsquares",
        "gre",
        include_str!("shader/randomsquares.wgsl"),
        |p| pack(&[p.whole("size_x", 1.0), p.whole("size_y", 1.0), p.num("smoothness")]),
        cpu::random_squares,
    ),
    def("gl_ripple", "ripple", "gre", include_str!("shader/ripple.wgsl"), |p| pack(&[p.num("amplitude"), p.num("speed")]), cpu::ripple),
    def("gl_simplezoom", "SimpleZoom", "0gust1", include_str!("shader/SimpleZoom.wgsl"), |p| pack(&[p.num("zoom_quickness")]), cpu::simple_zoom),
    def("gl_squeeze", "squeeze", "gre", include_str!("shader/squeeze.wgsl"), |p| pack(&[p.num("color_separation")]), cpu::squeeze),
    def(
        "gl_starwipe",
        "StarWipe",
        "Ben Lucas",
        include_str!("shader/StarWipe.wgsl"),
        |p| {
            let (cx, cy) = p.center_uv();
            let c = p.color("border_color");
            pack(&[p.num("border_thickness"), p.num("star_rotation"), cx, cy, c[0], c[1], c[2], c[3]])
        },
        cpu::star_wipe,
    ),
    def("gl_swap", "swap", "gre", include_str!("shader/swap.wgsl"), |p| pack(&[p.num("reflection"), p.num("perspective"), p.num("depth")]), cpu::swap),
    def("gl_swirl", "Swirl", "Sergey Kosarevsky", include_str!("shader/Swirl.wgsl"), |_| [0.0; PARAMS], cpu::swirl),
    def("gl_tvstatic", "TVStatic", "Brandon Anzaldi", include_str!("shader/TVStatic.wgsl"), |p| pack(&[p.num("offset")]), cpu::tv_static),
    def(
        "gl_waterdrop",
        "WaterDrop",
        "Paweł Płóciennik",
        include_str!("shader/WaterDrop.wgsl"),
        |p| pack(&[p.num("amplitude"), p.num("speed")]),
        cpu::water_drop,
    ),
    def("gl_wind", "wind", "gre", include_str!("shader/wind.wgsl"), |p| pack(&[p.num("size"), p.bit("reversed")]), cpu::wind),
    def(
        "gl_windowslice",
        "windowslice",
        "gre",
        include_str!("shader/windowslice.wgsl"),
        |p| pack(&[p.num("count").max(1.0), p.num("smoothness")]),
        cpu::window_slice,
    ),
];

fn pack(values: &[f32]) -> [f32; PARAMS] {
    let mut out = [0.0; PARAMS];
    for (o, v) in out.iter_mut().zip(values) {
        *o = if v.is_finite() { *v } else { 0.0 };
    }
    out
}

/// Every ported transition, sorted by id.
pub fn shaders() -> &'static [ShaderDef] {
    SHADERS
}

/// The ported transition with effect id `id`.
pub fn find(id: &str) -> Option<&'static ShaderDef> {
    SHADERS.iter().find(|d| d.id == id)
}

/// Whether `id` is a shader transition (drawn by the GPU's shader stage when one is available).
pub fn is_shader(id: &str) -> bool {
    find(id).is_some()
}

// ---------------------------------------------------------------------------------------------
// Validation registry

/// A shader the registry could not accept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderError {
    /// No shader transition has this id.
    Unknown(String),
    /// The WGSL failed to parse or validate (naga's message, with the source location).
    Invalid { id: &'static str, message: String },
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderError::Unknown(id) => write!(f, "no shader transition `{id}`"),
            ShaderError::Invalid { id, message } => write!(f, "shader transition `{id}` is not valid WGSL: {message}"),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Parse and validate a WGSL module with naga. Never panics: a parser or validator panic is
/// caught and reported as an error.
pub fn validate(id: &'static str, source: &str) -> Result<(), ShaderError> {
    let run = || -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
        let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
        validator.validate(&module).map_err(|e| e.emit_to_string(source))?;
        Ok(())
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(ShaderError::Invalid { id, message }),
        Err(_) => Err(ShaderError::Invalid { id, message: "the WGSL validator panicked".into() }),
    }
}

/// Every shader's complete source, validated once.
pub struct Registry {
    entries: Vec<(&'static str, Result<String, ShaderError>)>,
}

impl Registry {
    fn build() -> Registry {
        let entries = SHADERS
            .iter()
            .map(|d| {
                let src = d.wgsl();
                (d.id, validate(d.id, &src).map(|()| src))
            })
            .collect();
        Registry { entries }
    }

    /// The validated WGSL of shader transition `id`.
    pub fn source(&self, id: &str) -> Result<&str, ShaderError> {
        match self.entries.iter().find(|(i, _)| *i == id) {
            Some((_, Ok(src))) => Ok(src),
            Some((_, Err(e))) => Err(e.clone()),
            None => Err(ShaderError::Unknown(id.to_string())),
        }
    }

    /// The shaders that failed validation.
    pub fn errors(&self) -> Vec<&ShaderError> {
        self.entries.iter().filter_map(|(_, r)| r.as_ref().err()).collect()
    }
}

/// The validation registry (built on first use).
pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::build)
}

// ---------------------------------------------------------------------------------------------
// Parameters and time

/// Typed, sanitised access to a transition's parameters (instance value, else the catalogue
/// default; numbers clamped to the catalogue's range, never NaN).
pub(crate) struct Params<'a> {
    e: &'a EffectInstance,
    w: f32,
    h: f32,
    scale: f32,
}

impl Params<'_> {
    fn value(&self, id: &str) -> Option<ParamValue> {
        self.e.param(id).map(|p| p.value.clone()).or_else(|| self.e.def().and_then(|d| d.param(id)).map(|p| p.default.clone()))
    }

    fn num(&self, id: &str) -> f32 {
        let def = self.e.def().and_then(|d| d.param(id));
        let fallback = def.and_then(|p| p.default.as_f64()).filter(|v| v.is_finite()).unwrap_or(0.0);
        let v = self.e.f64_at(id, Tick::ZERO);
        let v = if v.is_finite() { v } else { fallback };
        let v = match def.map(|p| &p.kind) {
            Some(ParamKind::Float { min, max, .. }) if min <= max => v.clamp(*min, *max),
            _ => v.clamp(-1e6, 1e6),
        };
        v as f32
    }

    fn choice(&self, id: &str) -> u32 {
        match self.value(id) {
            Some(ParamValue::Choice(c)) => c,
            _ => 0,
        }
    }

    fn flag(&self, id: &str) -> bool {
        matches!(self.value(id), Some(ParamValue::Bool(true)))
    }

    /// A colour as opaque linear-light premultiplied RGBA (colour params are stored display-encoded).
    fn color(&self, id: &str) -> [f32; 4] {
        let c = match self.value(id) {
            Some(ParamValue::Color(c)) => c,
            _ => [1.0; 4],
        };
        let l = |v: f32| filmcraft_color::srgb_to_linear(if v.is_finite() { v.clamp(0.0, 1.0) } else { 1.0 });
        [l(c[0]), l(c[1]), l(c[2]), 1.0]
    }

    /// The `center` point (sequence pixels, NaN = frame centre) in shader uv (origin bottom
    /// left), limited to a frame around the picture.
    fn center_uv(&self) -> (f32, f32) {
        self.point_uv("center", (0.5, 0.5))
    }

    /// Point `id` (sequence pixels; a NaN coordinate = `default`'s, given in uv) in shader uv
    /// (origin bottom left), limited to a frame around the picture.
    fn point_uv(&self, id: &str, default: (f32, f32)) -> (f32, f32) {
        let v = match self.value(id) {
            Some(ParamValue::Vec2(v)) => v,
            _ => filmcraft_geom::Vec2::new(f64::NAN, f64::NAN),
        };
        let u = if v.x.is_finite() { (v.x as f32 * self.scale / self.w).clamp(-1.0, 2.0) } else { default.0 };
        let t = if v.y.is_finite() { 1.0 - (v.y as f32 * self.scale / self.h).clamp(-1.0, 2.0) } else { default.1 };
        (u, t)
    }

    /// A whole-number parameter, at least `min`.
    fn whole(&self, id: &str, min: f32) -> f32 {
        self.num(id).round().max(min)
    }

    /// A boolean parameter as the shader's 0 / 1.
    fn bit(&self, id: &str) -> f32 {
        if self.flag(id) { 1.0 } else { 0.0 }
    }

    fn bezier(&self) -> Bezier {
        Bezier { x1: self.num("bezier_x1") as f64, y1: self.num("bezier_y1") as f64, x2: self.num("bezier_x2") as f64, y2: self.num("bezier_y2") as f64 }
    }
}

/// `progress` (the timeline's, 0..1) through the transition's easing. NaN counts as 0.
pub fn eased_progress(e: &EffectInstance, progress: f32) -> f32 {
    let p = Params { e, w: 1.0, h: 1.0, scale: 1.0 };
    easing::ease(p.choice("easing"), p.bezier(), progress as f64) as f32
}

/// The uniforms of shader transition `e` on a `w`×`h` frame at timeline progress `progress`:
/// width, height, eased progress, aspect ratio, then the shader's parameters. `scale` converts
/// pixel parameters from sequence pixels to frame pixels. None if `e` is not a shader transition.
pub fn uniforms(e: &EffectInstance, w: u32, h: u32, progress: f32, scale: f32) -> Option<[f32; UNIFORM_FLOATS]> {
    let d = find(&e.effect)?;
    let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
    let p = Params { e, w: wf, h: hf, scale: if scale.is_finite() && scale > 0.0 { scale } else { 1.0 } };
    let mut out = [0.0; UNIFORM_FLOATS];
    out[0] = wf;
    out[1] = hf;
    out[2] = eased_progress(e, progress);
    out[3] = wf / hf;
    out[4..4 + PARAMS].copy_from_slice(&(d.params)(&p));
    out[4 + PARAMS] = 1.0 / wf;
    out[5 + PARAMS] = 1.0 / hf;
    Some(out)
}

/// The CPU reference of shader transition `id` (None for other transitions).
pub(crate) fn apply(id: &str, t: &Tx) -> Option<Image> {
    let d = find(id)?;
    let u = uniforms(t.e, t.w as u32, t.h as u32, t.p, t.scale)?;
    let progress = u[2];
    if progress <= 0.0 {
        return Some(t.a.clone());
    }
    if progress >= 1.0 {
        return Some(t.b.clone());
    }
    let mut p = [0.0; PARAMS];
    p.copy_from_slice(&u[4..4 + PARAMS]);
    let c = Ctx { a: t.a, b: t.b, w: u[0], h: u[1], progress, ratio: u[3], p };
    // the prelude's uv: the pixel centre times 1 / size (see `Host::inv`)
    let (inv_w, inv_h) = (u[4 + PARAMS], u[5 + PARAMS]);
    Some(paint(t.w, t.h, |x, y| (d.cpu)(&c, V2::new(x * inv_w, (c.h - y) * inv_h))))
}

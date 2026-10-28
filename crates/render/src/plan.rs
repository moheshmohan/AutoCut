//! Frame plans for the GPU compositor.
//!
//! [`plan_frame`] resolves what is visible at a time into a list of layers the GPU can draw
//! directly: a decoded source frame (YUV planes or RGBA), the matrix from source pixels to output
//! pixels, an opacity and a blend mode (all 27 of [`Blend`] are composited by the GPU). A media
//! clip whose enabled standard effects all have a GPU implementation ([`crate::gpufx`]) carries
//! them as a [`LayerFx`]: their parameters evaluated at the frame's time, run by the GPU on the
//! clip's working image before Motion places it, in the CPU's order (effects, each limited to its
//! masks → opacity masks → Motion → Opacity / blend). Anything the shaders don't cover yet — other
//! standard effects, adjustment layers, nested sequences, transitions other than dissolves, dips,
//! Push and the shader transitions — is rendered on the CPU for that layer (or the whole frame)
//! and handed over as a pre-composited image, so the GPU path matches the CPU reference; what
//! went to the CPU is counted by reason in [`cpu_work`] (`docs/gpu.md`).
//!
//! A Push or shader transition ([`crate::transitions::shader`], the GL Transitions) is planned as
//! ordered [`PlanStep`] operations on isolated clip inputs: each side is drawn into its own image,
//! the transition combines the two, and the result is composited (Normal) over the canvas of the
//! lower tracks, which is preserved for the blends above it. [`execute_cpu`] runs the same steps
//! on the CPU and is the reference.

use std::sync::Arc;

use filmcraft_frame::VideoFrame;
use filmcraft_geom::Affine;
use filmcraft_media::FrameRequest;
use filmcraft_project::{ItemId, ItemKind, Project, Sequence, TrackItem};
use filmcraft_time::Tick;

use crate::gpufx::FxOp;
use crate::{Blend, RenderOptions, SourceProvider, motion_matrix, output_size};

/// A GPU frame plan retains every resolved source frame. Nested sequences may share the same
/// child sequence many times, so recursion depth alone does not bound this expansion.
const MAX_PLAN_LAYERS: usize = 4096;

/// One layer for the GPU, bottom to top.
#[derive(Clone)]
pub struct PlanLayer {
    pub frame: Arc<VideoFrame>,
    /// Maps frame pixels (0..w, 0..h) to output pixels — or, with [`fx`](Self::fx), working-image
    /// pixels (0..fx.size).
    pub matrix: Affine,
    pub opacity: f32,
    /// How the layer combines with what is under it ([`crate::blend::composite`]).
    pub blend: Blend,
    /// Standard effects to run on the frame before it is placed (None: draw the frame directly).
    pub fx: Option<Arc<LayerFx>>,
    /// Colour management of a Y'CbCr frame that needs more than the default decode (HDR, BT.2020,
    /// P3 media on a Rec. 709 sequence; [`crate::colorman::gpu_input`]): run on the GPU as on the
    /// CPU. None: the default decode.
    pub input: Option<Arc<filmcraft_color::InputTransform>>,
    /// A 360° clip: the frame is a picture of the sphere, drawn through a virtual camera (`matrix`
    /// then places the camera's picture, or the effect stage works on it).
    pub vr: Option<Arc<VrLayer>>,
}

/// A 360° layer's virtual camera ([`crate::vr`]): what it looks at, the frame's model in its own
/// pixels, the size of the picture it makes and the rays a pixel (supersampling taps a side).
#[derive(Clone, Debug, PartialEq)]
pub struct VrLayer {
    pub view: crate::vr::View,
    pub model: crate::vr::SourceModel,
    pub out: (u32, u32),
    pub taps: u32,
}

/// The GPU effect stage of a layer: the frame is decoded into a working image of `size` (the
/// frame box-decimated by `decimation`, as the CPU decodes it), `ops` run on it in order, and the
/// result is placed with the layer's matrix, opacity and blend mode.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerFx {
    pub size: (u32, u32),
    pub decimation: u32,
    pub ops: Vec<FxOp>,
}

impl PlanLayer {
    /// A layer drawing `frame` directly.
    pub fn new(frame: Arc<VideoFrame>, matrix: Affine, opacity: f32, blend: Blend) -> Self {
        Self { frame, matrix, opacity, blend, fx: None, input: None, vr: None }
    }

    /// Size of the picture the matrix places: the working image with effects, else the virtual
    /// camera's picture, else the frame.
    pub fn size(&self) -> (u32, u32) {
        match (&self.fx, &self.vr) {
            (Some(fx), _) => fx.size,
            (None, Some(v)) => v.out,
            (None, None) => (self.frame.width, self.frame.height),
        }
    }
}

#[derive(Clone)]
pub enum FramePlan {
    /// Draw these layers over black.
    Layers { width: usize, height: usize, layers: Vec<PlanLayer> },
    /// Ordered layer runs and two-input operations; each step composites onto the same canvas.
    Composite { width: usize, height: usize, steps: Vec<PlanStep> },
    /// The CPU produced the final image (fallback).
    Image(crate::Image),
}

#[derive(Clone)]
pub enum PlanStep {
    Layers(Vec<PlanLayer>),
    /// Inputs are isolated clip canvases; transition result composites Normal over lower tracks.
    Transition {
        inputs: [Vec<PlanLayer>; 2],
        effect: filmcraft_project::EffectInstance,
        progress: f32,
        scale: f32,
        /// What a content-aware transition (a flow transition, the Content-Aware Wipe) learnt
        /// from the clips it joins ([`crate::transitions::content`]); None for the others. Such
        /// a transition is drawn on the GPU only with one.
        content: Option<Arc<crate::transitions::content::Field>>,
    },
}

#[derive(Default)]
struct PlanStack {
    steps: Vec<PlanStep>,
    exhausted: bool,
    /// Layers held in `steps`, kept as they are pushed so the cap check stays O(1).
    layers: usize,
}

impl PlanStack {
    fn layer_count(&self) -> usize {
        self.layers
    }

    fn push(&mut self, layer: PlanLayer) {
        if self.exhausted {
            return;
        }
        if self.layer_count() >= MAX_PLAN_LAYERS {
            self.exhausted = true;
            return;
        }
        if let Some(PlanStep::Layers(layers)) = self.steps.last_mut() {
            layers.push(layer);
        } else {
            self.steps.push(PlanStep::Layers(vec![layer]));
        }
        self.layers = self.layers.saturating_add(1);
    }

    fn push_transition(
        &mut self,
        inputs: [Vec<PlanLayer>; 2],
        effect: filmcraft_project::EffectInstance,
        progress: f32,
        scale: f32,
        content: Option<Arc<crate::transitions::content::Field>>,
    ) {
        let added: usize = inputs.iter().map(Vec::len).sum();
        if self.exhausted || self.layer_count().saturating_add(added) > MAX_PLAN_LAYERS {
            self.exhausted = true;
            return;
        }
        self.layers = self.layers.saturating_add(added);
        self.steps.push(PlanStep::Transition { inputs, effect, progress, scale, content });
    }

    fn into_layers(self, width: usize, height: usize) -> Option<Vec<PlanLayer>> {
        if self.exhausted {
            return None;
        }
        match self.finish(width, height) {
            FramePlan::Layers { layers, .. } => Some(layers),
            plan => Some(vec![PlanLayer::new(cpu_frame(execute_cpu(&plan)), Affine::IDENTITY, 1.0, Blend::Normal)]),
        }
    }

    fn finish(mut self, width: usize, height: usize) -> FramePlan {
        if self.steps.len() <= 1 && self.steps.first().is_none_or(|s| matches!(s, PlanStep::Layers(_))) {
            let layers = match self.steps.pop() {
                Some(PlanStep::Layers(layers)) => layers,
                _ => Vec::new(),
            };
            FramePlan::Layers { width, height, layers }
        } else {
            FramePlan::Composite { width, height, steps: self.steps }
        }
    }
}

impl FramePlan {
    /// Source storage retained by the plan, before GPU preparation.
    pub fn source_bytes(&self) -> usize {
        let layers = |ls: &[PlanLayer]| ls.iter().map(|l| l.frame.byte_size()).sum::<usize>();
        match self {
            Self::Image(img) => img.px.len() * size_of::<f32>(),
            Self::Layers { layers: ls, .. } => layers(ls),
            Self::Composite { steps, .. } => steps
                .iter()
                .map(|s| match s {
                    PlanStep::Layers(ls) => layers(ls),
                    PlanStep::Transition { inputs, .. } => inputs.iter().map(|ls| layers(ls)).sum(),
                })
                .sum(),
        }
    }

    /// Largest output/source texture side required by the plan.
    pub fn max_side(&self) -> usize {
        let layer_side = |ls: &[PlanLayer]| ls.iter().map(|l| l.frame.width.max(l.frame.height) as usize).max().unwrap_or(0);
        match self {
            Self::Image(img) => img.w.max(img.h),
            Self::Layers { width, height, layers } => (*width).max(*height).max(layer_side(layers)),
            Self::Composite { width, height, steps } => steps
                .iter()
                .map(|s| match s {
                    PlanStep::Layers(ls) => layer_side(ls),
                    PlanStep::Transition { inputs, .. } => inputs.iter().map(|ls| layer_side(ls)).max().unwrap_or(0),
                })
                .fold((*width).max(*height), usize::max),
        }
    }
}

fn cpu_frame(img: crate::Image) -> Arc<VideoFrame> {
    Arc::new(VideoFrame::rgba_f32(img.w as u32, img.h as u32, img.px))
}

/// What GPU plans left to the CPU, by reason: process-wide counts since the start (`perf.stats`
/// `export.gpu.cpuWork`, `docs/gpu.md`). A whole frame counts once per plan (`frame: …`), a layer
/// rendered on the CPU and handed to the GPU as an image once per frame it is in (`layer: …`).
pub mod cpu_work {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, PoisonError};

    static COUNTS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

    pub fn note(reason: String) {
        let mut g = COUNTS.lock().unwrap_or_else(PoisonError::into_inner);
        let n = g.entry(reason).or_insert(0);
        *n = n.saturating_add(1);
    }

    /// The counts by reason.
    pub fn counts() -> BTreeMap<String, u64> {
        COUNTS.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Start counting again from zero.
    pub fn reset() {
        COUNTS.lock().unwrap_or_else(PoisonError::into_inner).clear();
    }
}

/// Why [`plan_frame`] renders the frame of `seq` at `t` on the CPU ([`whole_frame_on_cpu`]).
fn whole_frame_reason(seq: &Sequence, t: Tick) -> String {
    if !seq.settings.color.is_plain() {
        return "frame: HDR or wide-gamut sequence".into();
    }
    for tr in seq.video_tracks.iter().filter(|tr| tr.enabled) {
        if let Some(trn) = tr.transitions.iter().find(|x| x.range().contains(t))
            && !simple_transition(&trn.effect.effect)
            && !two_input_transition(&trn.effect.effect)
        {
            return format!("frame: transition {}", trn.effect.effect);
        }
    }
    "frame: adjustment layer".into()
}

/// Why `item`'s layer is rendered on the CPU (the end of [`push_item`]).
fn cpu_layer_reason(project: &Project, item: &TrackItem, opts: RenderOptions) -> String {
    match project.item(item.item).map(|p| &p.kind) {
        Some(ItemKind::Graphic { .. }) => "layer: graphic with effects or masks".into(),
        Some(ItemKind::Sequence(_)) => "layer: nested sequence".into(),
        _ => match gpu_chain(project, item, opts) {
            Err(why) => why,
            Ok(_) => "layer: colour-managed (log) media, frame blending, or a parameter or mask the GPU does not take".into(),
        },
    }
}

fn simple_transition(id: &str) -> bool {
    matches!(filmcraft_project::creator_transitions::base_id(id), "cross_dissolve" | "dip_to_black" | "dip_to_white")
}

/// A transition planned as a two-input [`PlanStep::Transition`] (Push, the shader transitions).
fn two_input_transition(id: &str) -> bool {
    let id = filmcraft_project::creator_transitions::base_id(id);
    id == "push" || crate::transitions::shader::is_shader(id)
}

/// What the GPU effect stage runs for a media clip: its enabled standard effects (each with its
/// masks, if any) and whether its Opacity effect has masks.
struct GpuChain<'a> {
    effects: Vec<&'a filmcraft_project::EffectInstance>,
    opacity_masks: bool,
}

/// The standard effects of a media clip if the GPU can draw it (any blend mode, every enabled
/// standard effect with a GPU implementation, masked or not; no effects with effects off), and
/// whether its opacity is masked. Err: the clip is rendered on the CPU, and why.
fn gpu_chain<'a>(project: &Project, item: &'a TrackItem, opts: RenderOptions) -> Result<GpuChain<'a>, String> {
    let is_media = project.item(item.item).is_some_and(|p| matches!(p.kind, ItemKind::Media(_) | ItemKind::Subclip { .. }));
    if !is_media {
        return Err("layer: other item".into());
    }
    let opacity_masks = item.has_opacity_masks();
    let mut effects = Vec::new();
    if opts.effects {
        for e in item.effects.iter().filter(|e| e.enabled && crate::in_chain(e)) {
            if !crate::gpufx::GPU_EFFECTS.contains(&e.effect.as_str()) {
                return Err(format!("layer: effect {}", e.effect));
            }
            effects.push(e);
        }
    }
    Ok(GpuChain { effects, opacity_masks })
}

/// The GPU ops of a clip's effects ([`GpuChain`]) for a working image of `w` × `h` (`cx`), in the
/// CPU's order: each effect (limited to its masks), then the opacity masks. None when one of them
/// is not something the GPU stage reproduces.
fn gpu_ops(chain: &GpuChain<'_>, item: &TrackItem, cx: &crate::effects::FxCtx, w: usize, h: usize) -> Option<Vec<FxOp>> {
    let mut ops = Vec::with_capacity(chain.effects.len() + 1);
    for e in &chain.effects {
        let op = FxOp::eval(e, cx, w, h)?;
        let op = match e.masks.iter().any(|m| m.mode != filmcraft_project::MaskMode::None) {
            true => FxOp::Masked { op: Box::new(op), masks: crate::mask::prepare(&e.masks, cx.t, cx.px_scale) },
            false => op,
        };
        ops.push(op.gpu_ok().then_some(op)?);
    }
    if chain.opacity_masks
        && let Some(o) = item.effect("opacity")
    {
        let op = FxOp::OpacityMask { masks: crate::mask::prepare(&o.masks, cx.t, cx.px_scale) };
        ops.push(op.gpu_ok().then_some(op)?);
    }
    Some(ops)
}

/// Plan the frame at timeline `t`.
pub fn plan_frame(project: &Project, seq_id: ItemId, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> FramePlan {
    let Some(seq) = project.sequence(seq_id) else { return FramePlan::Image(crate::Image::new(1, 1)) };
    let (w, h) = output_size(seq, opts.scale);
    if whole_frame_on_cpu(project, seq, t) {
        cpu_work::note(whole_frame_reason(seq, t));
        return FramePlan::Image(crate::render_sequence(project, seq_id, t, opts, sources));
    }
    let mut layers = PlanStack::default();
    push_tracks(project, seq, t, opts, sources, 0, &mut layers);
    if layers.exhausted {
        cpu_work::note(format!("frame: more than {MAX_PLAN_LAYERS} layers"));
        return FramePlan::Image(crate::render_sequence(project, seq_id, t, opts, sources));
    }
    if opts.captions {
        for o in crate::caption_overlays(seq, t, w, h) {
            layers.push(PlanLayer::new(
                Arc::new(VideoFrame::rgba_f32(o.w as u32, o.h as u32, o.px)),
                Affine::translate(o.x as f64, o.y as f64),
                1.0,
                Blend::Normal,
            ));
        }
    }
    if layers.exhausted {
        cpu_work::note(format!("frame: more than {MAX_PLAN_LAYERS} layers"));
        return FramePlan::Image(crate::render_sequence(project, seq_id, t, opts, sources));
    }
    layers.finish(w, h)
}

/// Whether [`plan_frame`] hands the frame at `t` back as one CPU image ([`FramePlan::Image`]). A
/// GPU renderer gains nothing on such a frame, so a caller with only a few of them (the export's
/// pool) can render it on the CPU without taking one.
pub fn is_cpu_frame(project: &Project, seq_id: ItemId, t: Tick) -> bool {
    project.sequence(seq_id).is_none_or(|seq| whole_frame_on_cpu(project, seq, t))
}

/// Count the frame at `t` in [`cpu_work`] with the reason [`is_cpu_frame`] holds for it (a caller
/// that renders such a frame on the CPU without planning it).
pub fn note_cpu_frame(project: &Project, seq_id: ItemId, t: Tick) {
    let why = match project.sequence(seq_id) {
        Some(seq) => whole_frame_reason(seq, t),
        None => "frame: no sequence".into(),
    };
    cpu_work::note(why);
}

/// The frames [`plan_frame`] does not plan: an HDR / wide-gamut sequence composites and converts
/// on the CPU, and so does a frame with an adjustment layer or a complex transition anywhere at
/// `t` (whole-frame fallback).
fn whole_frame_on_cpu(project: &Project, seq: &Sequence, t: Tick) -> bool {
    !seq.settings.color.is_plain() || !layered_at(project, seq, t)
}

/// Whether the frame of `seq` at `t` can be planned as layers: no adjustment layer and no
/// transition the compositor cannot mix itself is showing (a transition into or out of an
/// adjustment layer counts, also on the side of the cut where the other clip is).
pub(crate) fn layered_at(project: &Project, seq: &Sequence, t: Tick) -> bool {
    let adjustment = |it: &TrackItem| project.item(it.item).is_some_and(|p| matches!(p.kind, ItemKind::AdjustmentLayer { .. }));
    for tr in &seq.video_tracks {
        if !tr.enabled {
            continue;
        }
        if let Some(trn) = tr.transitions.iter().find(|x| x.range().contains(t)) {
            if !simple_transition(&trn.effect.effect) && !two_input_transition(&trn.effect.effect) {
                return false;
            }
            if [trn.from, trn.to].into_iter().flatten().filter_map(|c| tr.item(c)).any(adjustment) {
                return false;
            }
        }
        if let Some(it) = tr.item_at(t)
            && adjustment(it)
        {
            return false;
        }
    }
    true
}

/// Push the layers of every video track of `seq` at `t`, bottom track first (`seq` must be
/// [`layered_at`] `t`). `nest` counts the nested sequences already followed to reach `seq`.
fn push_tracks(project: &Project, seq: &Sequence, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider, nest: u32, layers: &mut PlanStack) {
    let (w, h) = output_size(seq, opts.scale);
    for tr in &seq.video_tracks {
        if layers.exhausted {
            return;
        }
        if !tr.enabled {
            continue;
        }
        if let Some(trn) = tr.transitions.iter().find(|x| x.range().contains(t)) {
            // These dissolves are symmetric: Reverse (play B→A backwards) renders the same frames.
            let p = trn.progress(t) as f32;
            let a = trn.from.and_then(|id| tr.item(id)).filter(|i| i.enabled);
            let b = trn.to.and_then(|id| tr.item(id)).filter(|i| i.enabled);
            // (a Creator preset as its base transition)
            let effect = filmcraft_project::creator_transitions::resolve(&trn.effect);
            match effect.effect.as_str() {
                id if two_input_transition(id) => {
                    let mut inputs = [Vec::new(), Vec::new()];
                    for (slot, item) in inputs.iter_mut().zip([a, b]) {
                        if let Some(item) = item {
                            let mut input = PlanStack::default();
                            push_item(project, seq, item, t, opts, sources, 1.0, Some(Blend::Normal), nest, &mut input);
                            let Some(input_layers) = input.into_layers(w, h) else {
                                layers.exhausted = true;
                                return;
                            };
                            *slot = input_layers;
                        }
                    }
                    let progress = if trn.reverse {
                        inputs.swap(0, 1);
                        1.0 - p
                    } else {
                        p
                    };
                    let content = crate::transition_content(project, seq, tr, trn, opts, sources);
                    layers.push_transition(inputs, effect.clone().into_owned(), progress, opts.scale, content);
                }
                "dip_to_black" | "dip_to_white" => {
                    let col = if effect.effect == "dip_to_black" { [0.0, 0.0, 0.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] };
                    layers.push(PlanLayer::new(Arc::new(VideoFrame::rgba_f32(1, 1, col.to_vec())), Affine::scale(w as f64, h as f64), 1.0, Blend::Normal));
                    let (it, k) = if p < 0.5 { (a, 1.0 - p * 2.0) } else { (b, (p - 0.5) * 2.0) };
                    if let Some(it) = it {
                        push_item(project, seq, it, t, opts, sources, k, Some(Blend::Normal), nest, layers);
                    }
                }
                _ if a.is_some() && b.is_some_and(|b| !covers_frame(project, seq, b, t, opts, sources)) => {
                    // cross dissolve between clips that do not both fill the frame (scaled,
                    // moved, transparent): the CPU's linear mix of the two, as a two-input step
                    let mut inputs = [Vec::new(), Vec::new()];
                    for (slot, item) in inputs.iter_mut().zip([a, b]) {
                        if let Some(item) = item {
                            let mut input = PlanStack::default();
                            push_item(project, seq, item, t, opts, sources, 1.0, Some(Blend::Normal), nest, &mut input);
                            let Some(input_layers) = input.into_layers(w, h) else {
                                layers.exhausted = true;
                                return;
                            };
                            *slot = input_layers;
                        }
                    }
                    layers.push_transition(inputs, effect.clone().into_owned(), p, opts.scale, None);
                }
                _ => {
                    // cross dissolve: A at full, B over it at p — premultiplied over is the CPU's
                    // linear mix where B is opaque wherever A shows (it fills the frame)
                    if let Some(it) = a {
                        push_item(project, seq, it, t, opts, sources, 1.0 - if b.is_none() { p } else { 0.0 }, Some(Blend::Normal), nest, layers);
                    }
                    if let Some(it) = b {
                        push_item(project, seq, it, t, opts, sources, p, Some(Blend::Normal), nest, layers);
                    }
                }
            }
            continue;
        }
        let Some(item) = tr.item_at(t) else { continue };
        if !item.enabled {
            continue;
        }
        push_item(project, seq, item, t, opts, sources, 1.0, None, nest, layers);
    }
}

/// Whether a nested sequence can be drawn by pushing its own layers into the plan of the sequence
/// it is in, instead of being rendered to an image on the CPU first. Compositing its layers one
/// by one over what is below gives the same picture as compositing them together first only when
/// they all blend Normal, and the frame must be one that plans as layers in the same colour
/// pipeline. A Push or shader transition inside it is a step of the plan like one in the parent:
/// its result is composited Normal over what is under it (`over` is associative), so the GPU
/// draws it there too.
fn nest_is_plain(project: &Project, seq: &Sequence, nested: &Sequence, ft: Tick) -> bool {
    nested.settings.color == seq.settings.color
        && (nested.settings.width, nested.settings.height) == (seq.settings.width, seq.settings.height)
        && layered_at(project, nested, ft)
        && nested.video_tracks.iter().filter(|tr| tr.enabled).all(|tr| {
            tr.transitions.iter().any(|x| x.range().contains(ft))
                || tr.item_at(ft).filter(|i| i.enabled).is_none_or(|i| crate::opacity_blend(i, i.effect_time_at(ft)).1 == Blend::Normal)
        })
}

/// Push the layer(s) of `item`. `blend` overrides the item's own blend mode: inside a transition
/// the CPU reference mixes the clips and composites the result Normal, ignoring their modes.
/// `nest` counts the multi-camera clips already followed to reach `item` (see
/// [`crate::MAX_NEST_DEPTH`]).
#[allow(clippy::too_many_arguments)]
fn push_item(
    project: &Project,
    seq: &Sequence,
    item: &TrackItem,
    t: Tick,
    opts: RenderOptions,
    sources: &dyn SourceProvider,
    extra_opacity: f32,
    blend: Option<Blend>,
    nest: u32,
    out: &mut PlanStack,
) {
    if out.exhausted {
        return;
    }
    // frame time (`ft`) vs. effect time (`mt`): they differ inside a frame hold without Hold Filters
    let ft = item.source_time_at(t);
    let mt = item.effect_time_at(t);
    let (op, own) = crate::opacity_blend(item, mt);
    let bl = blend.unwrap_or(own);
    // A multi-camera clip that only shows its angle (no effects, untransformed, same frame size)
    // draws the angle's clip directly: no CPU pass over the nested sequence.
    if let Some(ItemKind::Sequence(nested)) = project.item(item.item).map(|p| &p.kind)
        && nest < crate::MAX_NEST_DEPTH
        && let Some(angle) = item.multicam_angle(nested)
        && !(opts.effects && item.has_standard_effects())
        && (nested.settings.width, nested.settings.height) == (seq.settings.width, seq.settings.height)
        && near_identity(&motion_matrix(seq, item, (nested.settings.width, nested.settings.height), Some(nested.settings.par), mt))
        && let Some(tr) = nested.angle_video_track_index(angle).and_then(|i| nested.video_tracks.get(i))
        && !tr.transitions.iter().any(|x| x.range().contains(ft))
        && tr.item_at(ft).is_none_or(|i| crate::opacity_blend(i, i.effect_time_at(ft)).1 != Blend::Dissolve)
    {
        // Inside the nested sequence the angle's clip is composited onto an empty canvas, where
        // every mode but Dissolve is Normal; the multicam clip's own mode then applies to it.
        if let Some(inner) = tr.item_at(ft).filter(|i| i.enabled) {
            push_item(project, nested, inner, ft, opts, sources, extra_opacity * op, Some(bl), nest + 1, out);
        }
        return;
    }
    // A nested sequence that is only shown (no effects, untransformed, fully opaque, blending
    // Normal) contributes its own layers: the compositor draws them like the clips of this
    // sequence, with no CPU pass over the nested sequence. Its captions are part of its picture.
    if let Some(ItemKind::Sequence(nested)) = project.item(item.item).map(|p| &p.kind)
        && nest < crate::MAX_NEST_DEPTH
        && item.multicam_angle(nested).is_none()
        && !(opts.effects && item.has_standard_effects())
        && !item.has_opacity_masks()
        && bl == Blend::Normal
        && extra_opacity * op >= 1.0 - 1e-6
        && near_identity(&motion_matrix(seq, item, (nested.settings.width, nested.settings.height), Some(nested.settings.par), mt))
        && nest_is_plain(project, seq, nested, ft)
    {
        push_tracks(project, nested, ft, opts, sources, nest + 1, out);
        if out.exhausted {
            return;
        }
        let (w, h) = output_size(nested, opts.scale);
        for o in crate::caption_overlays(nested, ft, w, h) {
            out.push(PlanLayer::new(
                Arc::new(VideoFrame::rgba_f32(o.w as u32, o.h as u32, o.px)),
                Affine::translate(o.x as f64, o.y as f64),
                1.0,
                Blend::Normal,
            ));
        }
        return;
    }
    // Graphic clips without standard effects: the layers are rasterised (cached) into one tight
    // image the GPU places as a layer.
    if !(opts.effects && item.has_standard_effects() || item.has_opacity_masks())
        && project.item(item.item).is_some_and(|p| matches!(p.kind, ItemKind::Graphic { .. }))
    {
        let Some(size) = crate::source_size(project, item.item) else { return };
        let (w, h) = output_size(seq, opts.scale);
        let m = Affine::scale(opts.scale as f64, opts.scale as f64).then_apply(&motion_matrix(seq, item, size, None, mt));
        cpu_work::note("layer: graphic (text and shapes rasterised on the CPU)".into());
        if let Some((img, x, y)) = crate::graphic_clip::render_graphic_tight(item, mt, size, &m, w, h) {
            out.push(PlanLayer::new(cpu_frame(img), Affine::translate(x as f64, y as f64), op * extra_opacity, bl));
        }
        return;
    }
    if let Ok(chain) = gpu_chain(project, item, opts) {
        let Some(src) = sources.source(item.item) else { return };
        let Some(size) = crate::clip_size(project, item) else { return };
        let motion = motion_matrix(seq, item, size, crate::clip_par(project, item), mt);
        let lin = ((motion.a * motion.a + motion.b * motion.b).sqrt()).max((motion.c * motion.c + motion.d * motion.d).sqrt());
        // the working scale `item_layer` gives the effects (lowered under a large blur)
        let want = crate::effects_scale(item, mt, crate::frame_scale(lin * opts.scale as f64, Some(&src)), opts);
        // a 360° clip: its virtual camera's picture at `want`, from the source as sharp as it needs
        let reframe = crate::reframing(project, item, t, &src, want);
        let time = crate::video_source_time(item, t, src.info().frame_rate());
        let scale = reframe.as_ref().map_or(want, |r| r.scale);
        let Ok(frame) = src.video_frame(FrameRequest { time, scale }) else { return };
        // HDR and wide-gamut media on a Rec. 709 sequence is converted on the GPU too (#965); log
        // media and other pipelines are converted on the CPU (below), and so are Optical Flow's
        // in-between frames. A Frame Blending in-between frame is the two pictures mixed here
        // (`blended_frame`), the rest of the layer drawn by the GPU.
        let pipe = &seq.settings.color;
        let (frame, input, plain) = match blended_frame(project, seq, item, t, &src, &frame, scale) {
            Some(mixed) => (mixed, None, true),
            None => {
                let cs = crate::colorman::source_space(project, item.item, &frame);
                let managed = crate::colorman::needs_management(pipe, cs, &frame);
                let input = if managed { crate::colorman::gpu_input(project, item.item, pipe, cs, &frame) } else { None };
                let plain = (input.is_some() || !managed) && crate::interpolation_blend(item, t, src.info().frame_rate()).is_none();
                (frame, input, plain)
            }
        };
        if let Some(r) = reframe {
            // the GPU draws the virtual camera's picture straight from the source's planes; its
            // effects then run on that picture, as `item_layer` runs them
            if plain {
                let model = crate::vr::SourceModel::at(&r.spherical, (frame.width as usize, frame.height as usize), time);
                // (near things on a dual fisheye's seam: the lenses' shifts measured on this frame)
                let model = match r.spherical.seam_flow {
                    true => {
                        let frames = crate::vr_seam::seam_frames(&src, time, scale, frame.clone());
                        let seam = crate::vr_seam::seam_of(&model, &frames.iter().map(|f| f.as_ref()).collect::<Vec<_>>());
                        crate::vr_seam::with_seam(model, seam)
                    }
                    false => model,
                };
                let taps = crate::vr::taps(&model, &r.view, r.out.0) as u32;
                let out_size = (r.out.0 as u32, r.out.1 as u32);
                let vr = Some(Arc::new(VrLayer { view: r.view, model, out: out_size, taps }));
                let px_scale = r.out.0 as f32 / size.0.max(1) as f32;
                let m = Affine::scale(opts.scale as f64, opts.scale as f64)
                    .then_apply(&motion)
                    .then_apply(&Affine::scale(1.0 / px_scale as f64, 1.0 / px_scale as f64));
                if chain.effects.is_empty() && !chain.opacity_masks {
                    out.push(PlanLayer { input, vr, ..PlanLayer::new(frame, m, op * extra_opacity, bl) });
                    return;
                }
                let cx = crate::effects::FxCtx {
                    t: mt,
                    px_scale,
                    seconds: (t - item.start).seconds(),
                    timecode: "",
                    clip_name: &item.name,
                    project: Some(project),
                    env: None,
                    frame_rate: seq.settings.frame_rate.as_f64(),
                    working: seq.settings.color.working,
                };
                if let Some(ops) = gpu_ops(&chain, item, &cx, r.out.0, r.out.1) {
                    let fx = Some(Arc::new(LayerFx { size: out_size, decimation: 1, ops }));
                    out.push(PlanLayer { fx, input, vr, ..PlanLayer::new(frame, m, op * extra_opacity, bl) });
                    return;
                }
            }
        } else if plain && !(chain.effects.is_empty() && !chain.opacity_masks) {
            // GPU effect stage: the working image the CPU would decode (`base_layer`), the
            // effects evaluated for it, placed as `item_layer` places it
            let n = crate::decimation(frame.width as f32, size.0 as f32 * want);
            let (lw, lh) = ((frame.width as usize / n).max(1), (frame.height as usize / n).max(1));
            let px_scale = lw as f32 / size.0.max(1) as f32;
            let tc = "";
            let cx = crate::effects::FxCtx {
                t: mt,
                px_scale,
                seconds: (t - item.start).seconds(),
                timecode: tc,
                clip_name: &item.name,
                project: Some(project),
                env: None,
                frame_rate: seq.settings.frame_rate.as_f64(),
                working: seq.settings.color.working,
            };
            if let Some(ops) = gpu_ops(&chain, item, &cx, lw, lh) {
                let m = Affine::scale(opts.scale as f64, opts.scale as f64)
                    .then_apply(&motion)
                    .then_apply(&Affine::scale(1.0 / px_scale as f64, 1.0 / px_scale as f64));
                let fx = LayerFx { size: (lw as u32, lh as u32), decimation: n as u32, ops };
                out.push(PlanLayer { frame, matrix: m, opacity: op * extra_opacity, blend: bl, fx: Some(Arc::new(fx)), input, vr: None });
                return;
            }
        } else if plain {
            // Draft playback at reduced resolution: hand over planes box-filtered to the size
            // drawn instead of the full picture (a quarter at 1/2, a sixteenth at 1/4 of the
            // upload and sampling). The mean is taken over Y'CbCr codes, not linear light as
            // the shader's supersampling does, so this stays limited to the opt-in draft mode.
            let n = if filmcraft_media::cancel::draft() { crate::decimation(frame.width as f32, size.0 as f32 * want) } else { 1 };
            let frame = match frame.box_decimated(n) {
                Some(small) => Arc::new(small),
                None => frame,
            };
            let px_scale = frame.width as f64 / size.0.max(1) as f64;
            let m = Affine::scale(opts.scale as f64, opts.scale as f64).then_apply(&motion).then_apply(&Affine::scale(1.0 / px_scale, 1.0 / px_scale));
            out.push(PlanLayer { input, ..PlanLayer::new(frame, m, op * extra_opacity, bl) });
            return;
        }
    }
    // CPU-rendered layer (standard effects): drawn by the GPU as a pre-rendered canvas image.
    cpu_work::note(cpu_layer_reason(project, item, opts));
    let tc = filmcraft_time::format_time(t, seq.settings.frame_rate, seq.settings.drop_frame, filmcraft_time::TimeDisplay::Timecode, 48_000);
    if let Some((img, op2, _)) = crate::item_layer(project, seq, item, t, opts, sources, &tc) {
        out.push(PlanLayer::new(cpu_frame(img), Affine::IDENTITY, op2 * extra_opacity, bl));
    }
}

/// A Frame Blending clip's in-between frame at timeline `t` for the GPU route: the source frame
/// shown (`frame`, read at `scale`) and the next one mixed in linear light by the clip's weight,
/// both colour managed first — what the CPU reference mixes (`media_image`; the GPU then
/// decimates, places and runs the effects on it, all linear, so the order does not change the
/// picture). `None` when the clip shows one source frame here, uses Frame Sampling or Optical
/// Flow (motion interpolation stays on the CPU), or the next frame cannot be read at that size.
fn blended_frame(
    project: &Project,
    seq: &Sequence,
    item: &TrackItem,
    t: Tick,
    src: &filmcraft_media::SharedSource,
    frame: &Arc<VideoFrame>,
    scale: f32,
) -> Option<Arc<VideoFrame>> {
    if item.time_interpolation != filmcraft_project::TimeInterpolation::FrameBlending {
        return None;
    }
    let (next, wgt) = crate::interpolation_blend(item, t, src.info().frame_rate())?;
    let f2 = src.video_frame(FrameRequest { time: next, scale }).ok().filter(|f| f.width == frame.width && f.height == frame.height)?;
    let pipe = &seq.settings.color;
    let a = crate::colorman::decode(project, item.item, frame, 1, pipe);
    let b = crate::colorman::decode(project, item.item, &f2, 1, pipe);
    let mixed = a.lerp(&b, wgt);
    Some(Arc::new(VideoFrame { pts: frame.pts, ..VideoFrame::rgba_f32(mixed.w as u32, mixed.h as u32, mixed.px) }))
}

/// Whether `item`'s layer is opaque over the whole output at `t`: a media clip without an alpha
/// channel, fully opaque (no opacity masks), with no standard effects (they could make it
/// transparent somewhere), placed unrotated to cover the frame.
fn covers_frame(project: &Project, seq: &Sequence, item: &TrackItem, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> bool {
    if !project.item(item.item).is_some_and(|p| matches!(p.kind, ItemKind::Media(_) | ItemKind::Subclip { .. }))
        || item.has_opacity_masks()
        || (opts.effects && item.has_standard_effects())
    {
        return false;
    }
    let mt = item.effect_time_at(t);
    if crate::opacity_blend(item, mt).0 < 1.0 - 1e-6 {
        return false;
    }
    let Some(src) = sources.source(item.item) else { return false };
    if src.info().video.as_ref().is_none_or(|v| v.has_alpha) {
        return false;
    }
    let Some(size) = crate::clip_size(project, item) else { return false };
    let m = Affine::scale(opts.scale as f64, opts.scale as f64).then_apply(&motion_matrix(seq, item, size, crate::clip_par(project, item), mt));
    if m.b.abs() > 1e-9 || m.c.abs() > 1e-9 {
        return false;
    }
    let (w, h) = output_size(seq, opts.scale);
    let (x0, x1) = (m.e.min(m.e + m.a * size.0 as f64), m.e.max(m.e + m.a * size.0 as f64));
    let (y0, y1) = (m.f.min(m.f + m.d * size.1 as f64), m.f.max(m.f + m.d * size.1 as f64));
    x0 <= 1e-6 && y0 <= 1e-6 && x1 >= w as f64 - 1e-6 && y1 >= h as f64 - 1e-6
}

fn near_identity(m: &Affine) -> bool {
    (m.a - 1.0).abs() < 1e-9 && (m.d - 1.0).abs() < 1e-9 && m.b.abs() < 1e-9 && m.c.abs() < 1e-9 && m.e.abs() < 1e-6 && m.f.abs() < 1e-6
}

/// A layer's working image with its effects applied, on the CPU (the reference for the GPU
/// effect stage).
pub fn effect_image(frame: &VideoFrame, fx: &LayerFx) -> crate::Image {
    effect_image_with(frame, fx, None)
}

/// The picture a layer's matrix places, on the CPU: its frame decoded (and colour managed),
/// seen through its virtual camera when it is a 360° layer, with its effects applied.
pub fn layer_image(l: &PlanLayer) -> crate::Image {
    let Some(v) = &l.vr else {
        return match &l.fx {
            Some(fx) => effect_image_with(&l.frame, fx, l.input.as_deref()),
            None => linear_image(&l.frame, 1, l.input.as_deref()),
        };
    };
    let src = linear_image(&l.frame, 1, l.input.as_deref());
    let mut img = crate::vr::reframe(&src, &v.model, &v.view, v.out.0 as usize, v.out.1 as usize, v.taps as usize);
    for op in l.fx.iter().flat_map(|fx| &fx.ops) {
        op.apply(&mut img);
    }
    img
}

/// [`effect_image`] of a colour-managed layer ([`PlanLayer::input`]).
pub fn effect_image_with(frame: &VideoFrame, fx: &LayerFx, input: Option<&filmcraft_color::InputTransform>) -> crate::Image {
    let mut img = linear_image(frame, fx.decimation.max(1) as usize, input);
    for op in &fx.ops {
        op.apply(&mut img);
    }
    img
}

/// Execute a plan on the CPU (reference for the GPU compositor).
pub fn execute_cpu(plan: &FramePlan) -> crate::Image {
    match plan {
        FramePlan::Image(img) => img.clone(),
        FramePlan::Composite { width, height, steps } => {
            let mut canvas = crate::Image::new(*width, *height);
            for step in steps {
                match step {
                    PlanStep::Layers(layers) => execute_layers(&mut canvas, layers),
                    PlanStep::Transition { inputs, effect, progress, scale, content } => {
                        let [a, b] = inputs.each_ref().map(|layers| {
                            let mut input = crate::Image::new(*width, *height);
                            execute_layers(&mut input, layers);
                            input
                        });
                        let mixed = crate::transitions::apply_with(effect, &a, &b, *progress, *scale, content.as_deref());
                        crate::blend::composite(&mut canvas, &mixed, 1.0, Blend::Normal);
                    }
                }
            }
            canvas
        }
        FramePlan::Layers { width, height, layers } => {
            let mut canvas = crate::Image::new(*width, *height);
            execute_layers(&mut canvas, layers);
            canvas
        }
    }
}

/// A layer's frame (box-decimated by `n`) as a linear working image: the default decode, or
/// `input`'s table and pixel stage as `colorman::decode` runs them.
fn linear_image(frame: &VideoFrame, n: usize, input: Option<&filmcraft_color::InputTransform>) -> crate::Image {
    let (w, h, px) = frame.to_linear_f32_decimated_with(n, input.map(|t| &t.table));
    let mut img = crate::Image { w, h, px };
    if let Some(t) = input.filter(|t| t.has_pixel_stage()) {
        img.map_rgb(|c, _, _| t.apply(c));
    }
    img
}

fn execute_layers(canvas: &mut crate::Image, layers: &[PlanLayer]) {
    let (width, height) = (canvas.w, canvas.h);
    for l in layers {
        let src = layer_image(l);
        let placed = if l.matrix == Affine::scale(width as f64, height as f64) && src.w == 1 && src.h == 1 {
            crate::Image::filled(width, height, src.get(0, 0))
        } else {
            src.transformed(width, height, &l.matrix)
        };
        crate::blend::composite(canvas, &placed, l.opacity, l.blend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattening_an_operation_preserves_its_picture() {
        let frame = Arc::new(VideoFrame::rgba_f32(1, 1, vec![0.25, 0.0, 0.0, 0.5]));
        let steps = vec![PlanStep::Transition {
            inputs: [vec![PlanLayer::new(frame, Affine::IDENTITY, 1.0, Blend::Normal)], vec![]],
            effect: filmcraft_project::find_effect("push").unwrap().instance(),
            progress: 0.0,
            scale: 1.0,
            content: None,
        }];
        let expected = execute_cpu(&FramePlan::Composite { width: 1, height: 1, steps: steps.clone() });
        let layers = PlanStack { steps, exhausted: false, layers: 1 }.into_layers(1, 1).unwrap();
        let got = execute_cpu(&FramePlan::Layers { width: 1, height: 1, layers });
        assert_eq!(got.px, expected.px);
        assert!(got.px.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn layer_plan_has_a_global_budget() {
        let frame = Arc::new(VideoFrame::rgba_f32(1, 1, vec![0.0; 4]));
        let mut stack = PlanStack::default();
        for _ in 0..=MAX_PLAN_LAYERS {
            stack.push(PlanLayer::new(frame.clone(), Affine::IDENTITY, 1.0, Blend::Normal));
        }
        assert!(stack.exhausted);
        assert_eq!(stack.layer_count(), MAX_PLAN_LAYERS);
    }

    /// A 4 × 2 source whose frame `k` (24 fps) is grey `k / 10`.
    struct Ramp {
        info: filmcraft_media::MediaInfo,
    }

    impl filmcraft_media::MediaSource for Ramp {
        fn info(&self) -> &filmcraft_media::MediaInfo {
            &self.info
        }
        fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
            let k = filmcraft_time::FrameRate::FPS_24.frame_at(req.time) as f32;
            let v = (k / 10.0).clamp(0.0, 1.0);
            Ok(Arc::new(VideoFrame { pts: req.time, ..VideoFrame::rgba_f32(4, 2, [v, v, v, 1.0].repeat(8)) }))
        }
        fn audio(&self, _start: i64, frames: usize, rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
            Ok(filmcraft_frame::AudioBuffer::silence(rate, 2, frames))
        }
    }

    #[test]
    fn a_frame_blended_frame_is_planned_for_the_gpu_and_matches_the_cpu() {
        let rate = filmcraft_time::FrameRate::FPS_24;
        let video = filmcraft_media::VideoStreamInfo {
            width: 4,
            height: 2,
            frame_rate: rate,
            par: (1, 1),
            codec: "test".into(),
            pixel_format: "rgba".into(),
            color: filmcraft_color::ColorInfo::REC709,
            has_alpha: false,
            bitrate: None,
            hdr: None,
        };
        let info = filmcraft_media::MediaInfo {
            name: "ramp".into(),
            kind: filmcraft_media::MediaKind::Movie,
            duration: rate.tick_of(48),
            video: Some(video),
            audio_streams: vec![],
            container: "test".into(),
            start_timecode: None,
            file_size: None,
        };
        let mut p = Project::new("t");
        let clip = filmcraft_project::MediaClip {
            media: filmcraft_project::MediaRef::File { path: "/ramp.mov".into() },
            info: info.clone(),
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        };
        let item = p.add_item("ramp", filmcraft_project::Label::Iris, ItemKind::Media(clip), None);
        let seq = p.new_sequence("s", filmcraft_project::SequenceSettings { width: 4, height: 2, frame_rate: rate, ..Default::default() }, 1, 1, None);
        let mut it = p
            .make_track_item(item, filmcraft_project::TrackKind::Video, Tick::ZERO, filmcraft_time::TimeRange::new(Tick::ZERO, rate.tick_of(24)), rate)
            .unwrap();
        it.speed = 0.5;
        it.duration = rate.tick_of(48);
        it.time_interpolation = filmcraft_project::TimeInterpolation::FrameBlending;
        p.sequence_mut(seq).unwrap().video_tracks[0].items.push(it);
        let mut sources = crate::SourceMap::default();
        sources.0.insert(item, Arc::new(Ramp { info }) as filmcraft_media::SharedSource);
        // timeline frame 3 at half speed: media frame 1.5, the two frames half and half
        let t = rate.tick_of(3);
        let opts = RenderOptions::default();
        let plan = plan_frame(&p, seq, t, opts, &sources);
        let FramePlan::Layers { layers, .. } = &plan else { panic!("a layered plan, not one CPU image") };
        assert_eq!(layers.len(), 1);
        let l = &layers[0];
        assert!(l.fx.is_none() && l.input.is_none());
        let expected = crate::render_sequence(&p, seq, t, opts, &sources);
        let got = execute_cpu(&plan);
        assert_eq!((got.w, got.h), (expected.w, expected.h));
        for (a, b) in got.px.iter().zip(&expected.px) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // between frame 1 (0.1) and 2 (0.2), mixed in linear light (float frames are linear)
        let one = |k: f32| k / 10.0;
        let mid = got.px[0];
        assert!(mid > one(1.0) + 1e-4 && mid < one(2.0) - 1e-4, "{mid}");
        // Frame Sampling shows frame 1 itself, with no mix
        let mut q = p.clone();
        q.sequence_mut(seq).unwrap().video_tracks[0].items[0].time_interpolation = filmcraft_project::TimeInterpolation::FrameSampling;
        let sampled = execute_cpu(&plan_frame(&q, seq, t, opts, &sources));
        assert!((sampled.px[0] - one(1.0)).abs() < 1e-4, "{}", sampled.px[0]);
    }
}

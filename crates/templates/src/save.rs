//! Save a sequence as a template (the authoring flow, A3).
//!
//! Clips flagged as placeholders — a clip marker named `slot:<kind>[:<slot id>[:<placement>]]`,
//! which `template.markSlot` adds and every applied template leaves on its slots — become the
//! template's slots, their media replaced by neutral generated placeholders (the procedural demo
//! scenes, colour mattes). Their Motion is rewritten the way a template means it: scale relative
//! to the slot's framing of the media (100 = exactly filling the frame, or the slot's box), so
//! the placeholder — and whatever fills it later — shows the way the clip did. A cropped picture
//! is a picture box (the crop is the box), unless the template the sequence came from names one.
//!
//! The personal media of every other clip is left out (with a warning), so a saved template
//! never carries someone's footage, unless the author bundles it: a bundled file goes in
//! `assets/` with its attribution filled in (its SHA-256, and the author's own file marked as
//! theirs under the template's licence, or what its `.attribution` sidecar says). Generated
//! media, graphics, adjustment layers and Elements sounds (turned back into their stand-ins)
//! stay. Captions are the user's words: with a captions slot their timing stays with neutral
//! text, otherwise they are left out; the caption style is kept.
//!
//! What the format carries beyond the timeline comes from the author's options, else from the
//! template the sequence was made from (`Template` metadata, an earlier apply), else defaults:
//! slot names, prompts, counts, constraints, fit and beat sync; a captions slot; the aspect
//! layouts (16:9, 9:16, 1:1, 4:5) with the base's per-clip hints; the poster frame; the look and
//! the export presets. The result is checked like any package (the strict manifest reader and the
//! cross-checks), so a saved template always reads back.

use std::collections::{BTreeMap, HashMap};

use filmcraft_geom::Vec2;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::graphic::TEXT_LAYER;
use filmcraft_project::gtemplate::{ensure_uids, layer_uid};
use filmcraft_project::{Caption, ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, Param, ParamValue, Project, TrackItem, TrackKind};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::apply::parse_slot_marker;
use crate::manifest::{
    AiHints, AssetEntry, BeatGrid, CaptionSpec, Constraints, Fit, Framing, Look, MAX_VARIANTS, Manifest, Placement, Slot, SlotDefault, SlotKind, Variant,
    default_fit,
};
use crate::{ASSETS_DIR, FORMAT, FORMAT_VERSION, Package, SEQUENCE_FILE, TemplateError};

/// What the author says about one slot; anything left `None` comes from the base template or the
/// defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlotEdit {
    pub name: Option<String>,
    pub prompt: Option<String>,
    pub order: Option<u32>,
    pub min_count: Option<u32>,
    pub beat_sync: Option<bool>,
    pub fit: Option<Fit>,
    pub constraints: Option<Constraints>,
    pub default: Option<SlotDefault>,
    /// Hints for filling the slot automatically; an empty set removes the base's.
    pub ai: Option<AiHints>,
}

/// A captions slot the author asks for.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptionsEdit {
    pub id: String,
    pub name: String,
    /// The video or music slots whose speech it captions; empty: every video slot with sound.
    pub source: Vec<String>,
    pub max_chars: u32,
    pub lines: u32,
    pub prompt: String,
    /// Its place among the slots (default: last).
    pub order: Option<u32>,
}

impl Default for CaptionsEdit {
    fn default() -> Self {
        CaptionsEdit { id: "captions".into(), name: "Captions".into(), source: Vec::new(), max_chars: 32, lines: 1, prompt: String::new(), order: None }
    }
}

/// Whether the template gets a captions slot.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CaptionsChoice {
    /// The base template's captions slot, if it has one.
    #[default]
    Auto,
    Off,
    Slot(CaptionsEdit),
}

/// A file of the user's, bundled in `assets/` (the engine reads it and its sidecar).
#[derive(Clone, Debug, PartialEq)]
pub struct BundleFile {
    /// The media item whose clips use it.
    pub item: ItemId,
    pub bytes: Vec<u8>,
    /// Its file name (made safe and unique in `assets/`).
    pub name: String,
    pub title: String,
    pub author: String,
    /// URL, or "original work".
    pub source: String,
    pub license: String,
}

/// The new template's description and authoring choices.
#[derive(Clone, Debug, Default)]
pub struct SaveOptions {
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub category: String,
    /// The template's own version (default `1.0.0`).
    pub version: String,
    pub tags: Vec<String>,
    /// The template the sequence was made from: its slot details, captions slot, layouts and
    /// hints, poster, look and export presets carry over.
    pub base: Option<Package>,
    /// Per-slot edits by slot id.
    pub slots: BTreeMap<String, SlotEdit>,
    pub captions: CaptionsChoice,
    /// Layouts (`16:9`, `9:16`, `1:1`, `4:5`); empty: the base's, else the sequence's own and its
    /// rotated twin. The sequence's own layout is always one.
    pub variants: Vec<String>,
    /// Where the thumbnail is taken (default: the base's, else a third of the way in).
    pub poster: Option<Tick>,
    pub look: Option<Look>,
    pub export_preset: Option<String>,
    /// The user's own files to bundle.
    pub bundle: Vec<BundleFile>,
}

/// A saved template and what was left out or changed.
#[derive(Clone, Debug)]
pub struct Saved {
    pub package: Package,
    pub warnings: Vec<String>,
}

/// Neutral text of the preview captions of a captions slot.
const PREVIEW_CAPTIONS: [&str; 3] = ["Your words appear here", "as you speak", "one line at a time"];
/// Most preview captions kept.
const MAX_PREVIEW_CAPTIONS: usize = 500;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// `16:9`-style id of a frame size.
pub fn aspect_id(w: u32, h: u32) -> String {
    let g = gcd(w, h);
    format!("{}:{}", w / g.max(1), h / g.max(1))
}

/// The frame size of layout `aspect` (`4:5`) for a sequence of `size`: the same short side.
pub fn variant_size(aspect: [u32; 2], size: (u32, u32)) -> (u32, u32) {
    let s = u64::from(size.0.min(size.1).max(16));
    let [a, b] = [u64::from(aspect[0].max(1)), u64::from(aspect[1].max(1))];
    let even = |x: u64| u32::try_from((x.div_ceil(2) * 2).clamp(16, 8192)).unwrap_or(8192);
    if a >= b { (even(s * a / b), even(s)) } else { (even(s), even(s * b / a)) }
}

fn variant_name(w: u32, h: u32) -> String {
    match (w, h) {
        (w, h) if w == h => "Square".into(),
        (w, h) if w > h => "Landscape".into(),
        (w, h) if u64::from(h) * 10 >= u64::from(w) * 15 => "Vertical".into(),
        _ => "Portrait".into(),
    }
}

fn placeholder(fragment: &mut Project, kind: SlotKind, n: usize, size: (u32, u32), rate: filmcraft_time::FrameRate) -> ItemId {
    let scenes = DemoScene::ALL;
    let (g, size, name) = match kind {
        SlotKind::Logo => (Generator::ColorMatte { color: [0.98, 0.76, 0.18, 1.0] }, (600, 600), format!("Logo Placeholder {}", n + 1)),
        SlotKind::Music => (Generator::Demo(DemoScene::Aurora), size, "Placeholder Music".to_string()),
        _ => (Generator::Demo(scenes.get(n % scenes.len()).copied().unwrap_or(DemoScene::OceanSunset)), size, format!("Placeholder {}", n + 1)),
    };
    let src = GeneratorSource::new(g.clone(), size.0, size.1, rate, Tick(60 * TICKS_PER_SECOND));
    let mut info = src.info().clone();
    info.name = name.clone();
    if kind == SlotKind::Music {
        info.video = None;
        info.kind = MediaKind::AudioOnly;
    }
    let clip = MediaClip {
        media: MediaRef::Generator(g),
        info,
        interpret: Default::default(),
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    fragment.add_item(&name, if kind == SlotKind::Music { Label::Caribbean } else { Label::Iris }, ItemKind::Media(clip), None)
}

fn static_vec2(e: &filmcraft_project::EffectInstance, k: &str) -> Option<Vec2> {
    let p = e.params.get(k)?;
    if p.is_animated() {
        return None;
    }
    p.value.as_vec2()
}

fn static_f64(e: &filmcraft_project::EffectInstance, k: &str, default: f64) -> Option<f64> {
    match e.params.get(k) {
        Some(p) if p.is_animated() => None,
        Some(p) => p.value.as_f64(),
        None => Some(default),
    }
}

fn finite(v: Vec2) -> bool {
    v.x.is_finite() && v.y.is_finite()
}

/// The picture box a cropped picture shows (fractions of the frame): the visible part of the
/// picture after Motion and Crop. `None` when nothing is cropped, when Motion or Crop move, the
/// picture is rotated, or the box is the whole frame.
fn box_from_crop(c: &TrackItem, src: (f64, f64), storage: Option<(u32, u32)>, frame: (f64, f64)) -> Option<[f64; 4]> {
    let ((cw, ch), (w, h)) = (src, frame);
    if !(cw > 0.0 && ch > 0.0 && w > 0.0 && h > 0.0) {
        return None;
    }
    let crop = c.effects.iter().find(|e| e.effect == "crop" && e.enabled)?;
    let side = |k: &str| static_f64(crop, k, 0.0).filter(|x| x.is_finite()).map(|x| (x / 100.0).clamp(0.0, 0.99));
    let (l, t, r, b) = (side("left")?, side("top")?, side("right")?, side("bottom")?);
    if l + t + r + b < 0.005 || l + r >= 0.99 || t + b >= 0.99 {
        return None;
    }
    let mo = c.effects.iter().find(|e| e.effect == "motion");
    let scale = mo.map_or(Some(100.0), |m| static_f64(m, "scale", 100.0))?;
    let rotation = mo.map_or(Some(0.0), |m| static_f64(m, "rotation", 0.0))?;
    if rotation.abs() > 1e-6 || !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let fit = if c.scale_to_frame { (w / cw).min(h / ch) } else { 1.0 };
    let s = scale / 100.0 * fit;
    let (dw, dh) = (cw * s, ch * s);
    let pos = match mo.map(|m| (m.params.get("position").is_some_and(|p| p.is_animated()), static_vec2(m, "position"))) {
        Some((true, _)) => return None,
        Some((false, Some(v))) if finite(v) => v,
        _ => Vec2::new(w / 2.0, h / 2.0),
    };
    // the anchor is in the picture's own pixels (its storage size)
    let (px, py) = storage.map_or((1.0, 1.0), |(a, bb)| (f64::from(a) / cw, f64::from(bb) / ch));
    let anchor = match mo.map(|m| (m.params.get("anchor").is_some_and(|p| p.is_animated()), static_vec2(m, "anchor"))) {
        Some((true, _)) => return None,
        Some((false, Some(v))) if finite(v) && px > 0.0 && py > 0.0 => Vec2::new(v.x / px, v.y / py),
        _ => Vec2::new(cw / 2.0, ch / 2.0),
    };
    let (left, top) = (pos.x - anchor.x * s, pos.y - anchor.y * s);
    let bx = [(left + l * dw) / w, (top + t * dh) / h, dw * (1.0 - l - r) / w, dh * (1.0 - t - b) / h];
    let ok = bx.iter().all(|x| x.is_finite())
        && (-1.0..=2.0).contains(&bx[0])
        && (-1.0..=2.0).contains(&bx[1])
        && bx[2] > 0.0
        && bx[3] > 0.0
        && bx[2] <= 2.0
        && bx[3] <= 2.0;
    let whole = bx[0].abs() < 0.005 && bx[1].abs() < 0.005 && (bx[2] - 1.0).abs() < 0.005 && (bx[3] - 1.0).abs() < 0.005;
    (ok && !whole).then(|| bx.map(|x| (x * 10_000.0).round() / 10_000.0))
}

/// Rewrite a picture clip's Motion the way a template means it: scale relative to the slot's
/// framing of this media in the frame or `area` (100 = exactly filling / fitting it). A boxed
/// clip's anchor and crop are the apply's to set again.
fn normalize_motion(c: &mut TrackItem, src: (f64, f64), frame: (f64, f64), framing: Framing, area: Option<[f64; 4]>) {
    let ((cw, ch), (w, h)) = (src, frame);
    if !(cw > 0.0 && ch > 0.0 && w > 0.0 && h > 0.0) {
        return;
    }
    let (bw, bh) = area.map_or((w, h), |a| (a[2] * w, a[3] * h));
    let k_slot = match framing {
        Framing::Fill => (bw / cw).max(bh / ch),
        Framing::Fit => (bw / cw).min(bh / ch),
        Framing::None => 1.0,
    };
    let k_user = if c.scale_to_frame { (w / cw).min(h / ch) } else { 1.0 };
    let factor = k_user / k_slot;
    if !(factor.is_finite() && (1e-3..=1e3).contains(&factor)) {
        return;
    }
    c.scale_to_frame = false;
    let Some(mo) = c.effects.iter_mut().find(|e| e.effect == "motion") else { return };
    let sp = mo.params.entry("scale".into()).or_insert_with(|| Param::new(ParamValue::Float(100.0)));
    sp.map_values(|v| match v {
        ParamValue::Float(x) if x.is_finite() => ParamValue::Float((x * factor * 1e6).round() / 1e6),
        o => o,
    });
    if area.is_some() {
        if let Some(a) = mo.params.get_mut("anchor")
            && !a.is_animated()
        {
            a.value = ParamValue::Vec2(Vec2::new(f64::NAN, f64::NAN));
        }
        // the apply crops the new picture to the box again
        if let Some(crop) = c.effects.iter_mut().find(|e| e.effect == "crop") {
            for k in ["left", "top", "right", "bottom"] {
                if crop.params.get(k).is_none_or(|p| !p.is_animated()) {
                    crop.params.insert(k.into(), Param::new(ParamValue::Float(0.0)));
                }
            }
        }
    }
}

fn set_layer_text(c: &mut TrackItem, layer: Option<u64>, text: &str) {
    if let Some(e) = c.effects.iter_mut().find(|e| e.effect == TEXT_LAYER && layer.is_none_or(|l| layer_uid(e) == l)) {
        e.params.insert("text".into(), Param::new(ParamValue::Text(text.to_string())));
        if let Some(x) = e.layer.as_mut() {
            x.runs.clear();
        }
    }
}

/// `name` made safe for `assets/` (letters, digits, `.`, `_`, `-`), unique among `taken`.
fn asset_name(name: &str, taken: &[String]) -> String {
    let mut s: String = name
        .chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                Some(c)
            } else if c == ' ' {
                Some('-')
            } else {
                None
            }
        })
        .collect();
    s = s.trim_matches('.').to_string();
    if s.is_empty() {
        s = "asset".into();
    }
    if s.len() > 96 {
        let ext = s.rsplit_once('.').map(|(_, e)| e.chars().take(8).collect::<String>()).unwrap_or_default();
        s = s.chars().take(80).collect();
        if !ext.is_empty() {
            s = format!("{s}.{ext}");
        }
    }
    let (stem, ext) = match s.rsplit_once('.') {
        Some((a, b)) if !a.is_empty() => (a.to_string(), format!(".{b}")),
        _ => (s.clone(), String::new()),
    };
    let mut out = format!("{ASSETS_DIR}{stem}{ext}");
    let mut n = 2;
    while taken.contains(&out) {
        out = format!("{ASSETS_DIR}{stem}-{n}{ext}");
        n += 1;
    }
    out
}

/// One placement as found on the timeline.
struct Found {
    /// The placement index of an earlier apply, else 10 000 + timeline order.
    key: usize,
    clip: u64,
    layer: Option<u64>,
    area: Option<[f64; 4]>,
}

/// `a` as slot `sid` (of `kind`, texts of at most `max_chars`) can hold it: hints for another kind
/// of slot are dropped with a warning; `None` when nothing is left.
fn fit_ai(mut a: AiHints, kind: SlotKind, max_chars: Option<u32>, sid: &str, warnings: &mut Vec<String>) -> Option<AiHints> {
    let before = a.clone();
    if !kind.is_visual() {
        a.purpose = None;
        a.query.clear();
        a.energy = None;
        a.min_quality = None;
        a.chronological = false;
        a.max_hold = None;
    }
    if kind != SlotKind::Video {
        a.speech = None;
    }
    if kind != SlotKind::Text {
        a.prompt.clear();
        a.max_chars = None;
        a.about.clear();
    }
    if let (Some(m), Some(limit)) = (a.max_chars, max_chars) {
        a.max_chars = Some(m.min(limit));
    }
    if a != before {
        warnings.push(format!("slot `{sid}`: hints that do not fit a {} slot were dropped", kind.id()));
    }
    (!a.is_empty()).then_some(a)
}

/// Make a template from sequence `seq` of `p`. `beat_grid` is the music's beats on the
/// sequence's timeline (the engine measures them), if any.
pub fn save(p: &Project, seq: ItemId, opts: &SaveOptions, beat_grid: Option<BeatGrid>) -> Result<Saved, TemplateError> {
    let err = |m: String| TemplateError::Apply(m);
    let src = p.sequence(seq).ok_or_else(|| err("no such sequence".into()))?;
    let mut q = src.clone();
    let rate = q.settings.frame_rate;
    let size = (q.settings.width, q.settings.height);
    let frame = (f64::from(size.0), f64::from(size.1));
    let par = q.settings.par;
    let base = opts.base.as_ref();
    let bm = base.map(|b| &b.manifest);
    let mut warnings = Vec::new();
    let mut fragment = Project::new(&opts.name);
    fragment.next_id = p.next_id;
    let bundled: HashMap<ItemId, &BundleFile> = opts.bundle.iter().map(|b| (b.item, b)).collect();
    // a slot's framing: the author's, the base's, the kind's default
    let framing_of = |sid: &str, kind: SlotKind| -> Framing {
        opts.slots
            .get(sid)
            .and_then(|e| e.fit.as_ref())
            .map(|f| f.frame)
            .or_else(|| bm.and_then(|m| m.slot(sid)).filter(|s| s.kind == kind).map(|s| s.fit.frame))
            .unwrap_or_else(|| default_fit(kind).frame)
    };
    let base_placement = |sid: &str, kind: SlotKind, key: usize| -> Option<&Placement> {
        bm.and_then(|m| m.slot(sid)).filter(|s| s.kind == kind).and_then(|s| s.placements.get(key))
    };
    let base_texts: HashMap<String, Vec<String>> = match base {
        Some(b) => b.manifest.slots.iter().filter(|s| s.kind == SlotKind::Text).map(|s| (s.id.clone(), b.placeholder_texts(&s.id))).collect(),
        None => HashMap::new(),
    };
    // (slot id) → (kind, placements)
    let mut slots: BTreeMap<String, (SlotKind, Vec<Found>)> = BTreeMap::new();
    // base template clip → saved clip (for the base's layout hints)
    let mut base_to_new: HashMap<u64, u64> = HashMap::new();
    let mut copied: HashMap<ItemId, ItemId> = HashMap::new();
    let mut placeholders: HashMap<(String, usize), ItemId> = HashMap::new();
    let mut dropped: Vec<ClipId> = Vec::new();
    let mut assets: Vec<AssetEntry> = Vec::new();
    let mut asset_files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut order = 0usize;
    for kind in [TrackKind::Video, TrackKind::Audio] {
        for t in q.tracks_mut(kind).iter_mut() {
            let mut keep = Vec::new();
            // pieces of one placement (a loop, a hold) become one clip again
            let mut last_slot: Option<(String, usize, usize)> = None;
            for mut c in std::mem::take(&mut t.items) {
                // (a captions slot has no clips: a `slot:captions` flag means nothing)
                let flags: Vec<crate::apply::SlotFlag> =
                    c.markers.iter().filter_map(|m| parse_slot_marker(&m.name)).filter(|f| f.kind != SlotKind::Captions).take(64).collect();
                c.markers.retain(|m| !m.name.starts_with(crate::apply::SLOT_MARKER));
                let item = p.item(c.item);
                match flags.first().cloned() {
                    Some(first) => {
                        let (sk, sid) = (first.kind, first.slot.clone());
                        let pl = first.placement.unwrap_or_else(|| {
                            order += 1;
                            10_000 + order
                        });
                        if let Some((ls, lp, li)) = &last_slot
                            && *ls == sid
                            && *lp == pl
                            && let Some(prev) = keep.get_mut(*li)
                        {
                            let prev: &mut TrackItem = prev;
                            prev.duration = c.end() - prev.start;
                            dropped.push(c.id);
                            continue;
                        }
                        if sk == SlotKind::Text {
                            let Some(it) = item.filter(|i| matches!(i.kind, ItemKind::Graphic { .. })) else {
                                warnings.push(format!("“{}” is flagged as a text slot but is not a graphic; it was left out", c.name));
                                dropped.push(c.id);
                                continue;
                            };
                            let mut layers: Vec<_> = c.effects.iter().filter(|e| filmcraft_project::graphic::is_layer(e)).cloned().collect();
                            ensure_uids(&mut layers);
                            let mut li = layers.into_iter();
                            for e in c.effects.iter_mut().filter(|e| filmcraft_project::graphic::is_layer(e)) {
                                if let Some(n) = li.next() {
                                    *e = n;
                                }
                            }
                            let texts: Vec<u64> = c.effects.iter().filter(|e| e.effect == TEXT_LAYER).map(layer_uid).collect();
                            if texts.is_empty() {
                                warnings.push(format!("“{}” has no text layer; it was left out of the slots", c.name));
                            } else if kind == TrackKind::Video {
                                // one graphic may hold several text slots, a layer each
                                for f in flags.iter().filter(|f| f.kind == SlotKind::Text) {
                                    let layer = f.layer.filter(|l| texts.contains(l)).or(texts.first().copied());
                                    let fpl = f.placement.unwrap_or_else(|| {
                                        order += 1;
                                        10_000 + order
                                    });
                                    // the template's own text again, not what the user typed
                                    if let Some(bp) = base_placement(&f.slot, SlotKind::Text, fpl) {
                                        base_to_new.entry(bp.clip).or_insert(c.id.0);
                                        if let Some(text) = base_texts.get(&f.slot).and_then(|v| v.get(bp.media as usize)).filter(|t| !t.is_empty()) {
                                            set_layer_text(&mut c, layer, text);
                                        }
                                    }
                                    let entry = slots.entry(f.slot.clone()).or_insert((SlotKind::Text, Vec::new()));
                                    if !entry.1.iter().any(|x| x.clip == c.id.0 && x.layer == layer) {
                                        entry.1.push(Found { key: fpl, clip: c.id.0, layer, area: None });
                                    }
                                }
                            }
                            let id = *copied.entry(c.item).or_insert_with(|| fragment.add_item(&it.name, it.label, it.kind.clone(), None));
                            c.item = id;
                        } else {
                            let on_its_track = if sk == SlotKind::Music { kind == TrackKind::Audio } else { kind == TrackKind::Video };
                            let mut area = None;
                            if sk.is_visual() && kind == TrackKind::Video {
                                area = base_placement(&sid, sk, pl).and_then(|b| b.area);
                                if let Some(src) = p.conformed_source_size(c.item, par) {
                                    if area.is_none() {
                                        area = box_from_crop(&c, src, p.source_size(c.item), frame);
                                    }
                                    normalize_motion(&mut c, src, frame, framing_of(&sid, sk), area);
                                }
                            }
                            if on_its_track && let Some(bp) = base_placement(&sid, sk, pl) {
                                base_to_new.entry(bp.clip).or_insert(c.id.0);
                            }
                            let id = *placeholders.entry((sid.clone(), pl)).or_insert_with(|| placeholder(&mut fragment, sk, pl % 10_000, size, rate));
                            c.item = id;
                            c.name = fragment.item(id).map(|i| i.name.clone()).unwrap_or_default();
                            c.source_in = Tick::ZERO;
                            c.speed = 1.0;
                            c.reverse = false;
                            c.frame_hold = None;
                            if on_its_track {
                                slots.entry(sid.clone()).or_insert((sk, Vec::new())).1.push(Found { key: pl, clip: c.id.0, layer: None, area });
                            }
                        }
                        last_slot = Some((sid, pl, keep.len()));
                        keep.push(c);
                    }
                    None => {
                        last_slot = None;
                        // an Elements sound (its synthesized WAV) becomes its stand-in again
                        if let Some((it, e)) = item.and_then(|i| i.metadata.get(crate::ELEMENT_METADATA).map(|e| (i, e)))
                            && crate::is_sound_element_id(e)
                            && it.as_media().is_some()
                        {
                            let len = it.duration();
                            let id = *copied.entry(c.item).or_insert_with(|| crate::builtin::sound_stand_in(&mut fragment, e, &it.name, len));
                            c.item = id;
                            keep.push(c);
                            continue;
                        }
                        // the author's own file, bundled with its attribution
                        if let (Some(b), Some(it)) = (bundled.get(&c.item), item)
                            && let Some(m) = it.as_media()
                        {
                            let id = match copied.get(&c.item) {
                                Some(id) => *id,
                                None => {
                                    let taken: Vec<String> = assets.iter().map(|a| a.path.clone()).collect();
                                    let path = asset_name(&b.name, &taken);
                                    assets.push(AssetEntry {
                                        path: path.clone(),
                                        sha256: crate::sha256_hex(&b.bytes),
                                        title: b.title.clone(),
                                        author: b.author.clone(),
                                        source: b.source.clone(),
                                        license: b.license.clone(),
                                    });
                                    asset_files.insert(path.clone(), b.bytes.clone());
                                    let mut info = m.info.clone();
                                    info.name = path.trim_start_matches(ASSETS_DIR).to_string();
                                    let mc = MediaClip {
                                        media: MediaRef::File { path },
                                        info,
                                        interpret: m.interpret.clone(),
                                        mark_in: None,
                                        mark_out: None,
                                        markers: vec![],
                                        offline: false,
                                        proxy: None,
                                        identity: None,
                                    };
                                    let id = fragment.add_item(&b.title, it.label, ItemKind::Media(mc), None);
                                    copied.insert(c.item, id);
                                    id
                                }
                            };
                            c.item = id;
                            keep.push(c);
                            continue;
                        }
                        let ok = item.is_some_and(|i| match &i.kind {
                            ItemKind::Graphic { .. } | ItemKind::AdjustmentLayer { .. } => true,
                            ItemKind::Media(m) => matches!(m.media, MediaRef::Generator(_)),
                            _ => false,
                        });
                        match (ok, item) {
                            (true, Some(it)) => {
                                let id = *copied.entry(c.item).or_insert_with(|| fragment.add_item(&it.name, it.label, it.kind.clone(), None));
                                c.item = id;
                                keep.push(c);
                            }
                            _ => {
                                warnings.push(format!("“{}” uses your own media and is not a slot; it was left out", c.name));
                                dropped.push(c.id);
                            }
                        }
                    }
                }
            }
            t.items = keep;
        }
    }
    // transitions follow merged pieces; those of left-out clips go
    for t in q.video_tracks.iter_mut().chain(q.audio_tracks.iter_mut()) {
        let spans: Vec<(ClipId, Tick, Tick)> = t.items.iter().map(|c| (c.id, c.start, c.end())).collect();
        let kept = |c: ClipId| spans.iter().any(|s| s.0 == c);
        for tr in t.transitions.iter_mut() {
            let cut = match tr.align {
                filmcraft_project::TransitionAlign::CenterAtCut => tr.start + Tick(tr.duration.0 / 2),
                filmcraft_project::TransitionAlign::StartAtCut => tr.start,
                filmcraft_project::TransitionAlign::EndAtCut => tr.end(),
            };
            if let Some(c) = tr.from
                && !kept(c)
            {
                tr.from = if dropped.contains(&c) { spans.iter().find(|s| s.1 < cut && s.2 >= cut).map(|s| s.0) } else { None };
            }
            if let Some(c) = tr.to
                && !kept(c)
            {
                tr.to = if dropped.contains(&c) { spans.iter().find(|s| s.1 <= cut && s.2 > cut).map(|s| s.0) } else { None };
            }
        }
        t.transitions.retain(|tr| tr.from.is_some_and(kept) || tr.to.is_some_and(kept));
    }
    let duration = q.duration();
    if duration.0 <= 0 {
        return Err(err("the sequence is empty".into()));
    }

    // the slots, with what the base and the author say about them
    let beats_ok = beat_grid.as_ref().is_some_and(|g| g.beats.len() >= 2);
    let mut manifest_slots = Vec::new();
    for (i, (sid, (kind, mut found))) in slots.into_iter().enumerate() {
        found.sort_by_key(|x| x.key);
        let bs = bm.and_then(|m| m.slot(&sid)).filter(|s| s.kind == kind);
        // media indices: the base's (one photo in several places stays one), else one each
        let media: Vec<u32> =
            found.iter().enumerate().map(|(k, f)| bs.and_then(|s| s.placements.get(f.key)).map_or(u32::try_from(k).unwrap_or(0), |p| p.media)).collect();
        let mut count = media.iter().max().map_or(1, |m| m.saturating_add(1)).max(1);
        // every media index needs a placement: renumber when the base's leave holes
        let mut placements: Vec<Placement> =
            found.iter().zip(&media).map(|(f, m)| Placement { clip: f.clip, media: *m, layer: f.layer, area: f.area }).collect();
        if (0..count).any(|m| !placements.iter().any(|p| p.media == m)) {
            for (k, pl) in placements.iter_mut().enumerate() {
                pl.media = u32::try_from(k).unwrap_or(0);
            }
            count = u32::try_from(placements.len()).unwrap_or(1).max(1);
        }
        let mut s = Slot {
            id: sid.clone(),
            kind,
            name: bs.map_or_else(|| sid.replace(['-', '_'], " "), |b| b.name.clone()),
            prompt: bs.map(|b| b.prompt.clone()).unwrap_or_default(),
            order: bs.map_or(u32::try_from(i).unwrap_or(u32::MAX), |b| b.order),
            count,
            min_count: bs.map_or(if matches!(kind, SlotKind::Text | SlotKind::Music) { 0 } else { 1 }, |b| b.min_count),
            placements,
            constraints: bs.map(|b| b.constraints.clone()).unwrap_or_default(),
            fit: bs.map_or_else(|| default_fit(kind), |b| b.fit),
            beat_sync: bs.map_or(beats_ok && kind.is_visual(), |b| b.beat_sync),
            default: bs.and_then(|b| b.default.clone()),
            captions: None,
            ai: bs.and_then(|b| b.ai.clone()),
        };
        if kind == SlotKind::Music {
            s.count = 1;
            s.placements.truncate(1);
            if let Some(pl) = s.placements.first_mut() {
                pl.media = 0;
            }
        }
        if let Some(e) = opts.slots.get(&sid) {
            if let Some(n) = &e.name {
                s.name = n.clone();
            }
            if let Some(n) = &e.prompt {
                s.prompt = n.clone();
            }
            if let Some(n) = e.order {
                s.order = n;
            }
            if let Some(n) = e.min_count {
                s.min_count = n;
            }
            if let Some(n) = e.beat_sync {
                s.beat_sync = n;
            }
            if let Some(f) = &e.fit {
                s.fit = *f;
            }
            if let Some(c) = &e.constraints {
                s.constraints = c.clone();
            }
            if let Some(d) = &e.default {
                s.default = Some(d.clone());
            }
            if let Some(a) = &e.ai {
                s.ai = (!a.is_empty()).then(|| a.clone());
            }
        }
        // hints that no longer fit the slot (its kind, its text's length) are dropped
        if let Some(a) = s.ai.take() {
            s.ai = fit_ai(a, s.kind, s.constraints.max_chars, &sid, &mut warnings);
        }
        s.min_count = s.min_count.min(s.count);
        if s.beat_sync && (!beats_ok || !kind.is_visual()) {
            if beats_ok {
                warnings.push(format!("slot `{sid}` cannot cut on the beat (only picture slots do)"));
            } else {
                warnings.push(format!("slot `{sid}` cut on the beat, but the music's beats could not be measured; it no longer does"));
            }
            s.beat_sync = false;
        }
        // the placements' lengths now: a re-timed edit keeps within the constraints
        let lens: Vec<Tick> = s.placements.iter().filter_map(|pl| q.find_item(ClipId(pl.clip)).map(|(_, c)| c.duration)).collect();
        if let (Some(min), Some(shortest)) = (s.constraints.min_duration, lens.iter().min())
            && *shortest < min
        {
            s.constraints.min_duration = Some(*shortest);
        }
        if let (Some(max), Some(longest)) = (s.constraints.max_duration, lens.iter().max())
            && *longest > max
        {
            s.constraints.max_duration = Some(*longest);
        }
        if s.kind == SlotKind::Text
            && let Some(max) = s.constraints.max_chars
        {
            s.constraints.max_chars = Some(max.max(1));
        }
        manifest_slots.push(s);
    }
    if manifest_slots.is_empty() {
        warnings.push("no clip is flagged as a slot (template.markSlot): the template has no slots".into());
    }

    // a captions slot: the author's, else the base's
    let captions = match &opts.captions {
        CaptionsChoice::Off => None,
        CaptionsChoice::Slot(e) => Some(e.clone()),
        CaptionsChoice::Auto => bm.and_then(|m| m.captions_slot()).map(|s| {
            let spec = s.captions.clone().unwrap_or_default();
            CaptionsEdit {
                id: s.id.clone(),
                name: s.name.clone(),
                source: spec.source,
                max_chars: spec.max_chars,
                lines: spec.lines,
                prompt: s.prompt.clone(),
                order: Some(s.order),
            }
        }),
    };
    if let Some(c) = captions {
        let speaking = |s: &&Slot| matches!(s.kind, SlotKind::Video | SlotKind::Music);
        let mut source: Vec<String> = c.source.iter().filter(|x| manifest_slots.iter().filter(speaking).any(|s| &s.id == *x)).cloned().collect();
        if source.is_empty() {
            // every video slot whose clips have sound
            source = manifest_slots
                .iter()
                .filter(|s| s.kind == SlotKind::Video)
                .filter(|s| s.placements.iter().any(|pl| q.find_item(ClipId(pl.clip)).is_some_and(|(_, c)| c.link.is_some())))
                .map(|s| s.id.clone())
                .collect();
        }
        let id = if crate::manifest::is_slot_id(&c.id) && !manifest_slots.iter().any(|s| s.id == c.id) { c.id.clone() } else { "captions".to_string() };
        if source.is_empty() {
            warnings.push("no video or music slot has sound to caption; the template has no captions slot".into());
        } else if manifest_slots.iter().any(|s| s.id == id) {
            warnings.push(format!("a slot is already called `{id}`; the template has no captions slot"));
        } else {
            source.truncate(16);
            manifest_slots.push(Slot {
                id,
                kind: SlotKind::Captions,
                name: if c.name.trim().is_empty() { "Captions".into() } else { c.name.clone() },
                prompt: c.prompt.clone(),
                order: c.order.unwrap_or_else(|| u32::try_from(manifest_slots.len()).unwrap_or(u32::MAX)),
                count: 1,
                min_count: 0,
                placements: Vec::new(),
                constraints: Constraints::default(),
                fit: default_fit(SlotKind::Captions),
                beat_sync: false,
                default: None,
                captions: Some(CaptionSpec { source, max_chars: c.max_chars.clamp(8, 200), lines: c.lines.clamp(1, 2) }),
                ai: None,
            });
        }
    }
    // a text slot's hints are about a picture slot the template still has
    let pictures: Vec<String> = manifest_slots.iter().filter(|s| s.kind.is_visual()).map(|s| s.id.clone()).collect();
    for s in manifest_slots.iter_mut() {
        if let Some(a) = s.ai.as_mut()
            && !a.about.is_empty()
            && !pictures.contains(&a.about)
        {
            warnings.push(format!("slot `{}`: its texts were about slot `{}`, which the template no longer has", s.id, a.about));
            a.about.clear();
        }
        if s.ai.as_ref().is_some_and(AiHints::is_empty) {
            s.ai = None;
        }
    }
    let has_captions_slot = manifest_slots.iter().any(|s| s.kind == SlotKind::Captions);
    // the captions are the user's words: with a captions slot their timing stays as a neutral
    // preview, without one they go; the style stays
    let had_captions = q.caption_tracks.iter().any(|t| !t.captions.is_empty());
    let caption_style = q.caption_tracks.first().map(|t| t.style.clone()).or_else(|| bm.and_then(|m| m.caption_style.clone()));
    if has_captions_slot {
        q.caption_tracks.truncate(1);
        if let Some(t) = q.caption_tracks.first_mut() {
            t.vtt_blocks.clear();
            t.captions.truncate(MAX_PREVIEW_CAPTIONS);
            for (k, cap) in t.captions.iter_mut().enumerate() {
                *cap = Caption {
                    id: cap.id,
                    start: cap.start,
                    duration: cap.duration,
                    text: PREVIEW_CAPTIONS.get(k % PREVIEW_CAPTIONS.len()).copied().unwrap_or("Captions").to_string(),
                    speaker: None,
                    cue_id: None,
                    settings: String::new(),
                    words: Vec::new(),
                };
            }
        }
    } else {
        q.caption_tracks.clear();
        if had_captions {
            warnings.push("the captions were left out (their text is yours); the caption style is kept".into());
        }
    }
    // the base's other hinted clips (stickers, cards that are no slot): the clip at the same place
    // of the same track, when it shows the same thing
    if let (Some(b), Some(bq)) = (base, base.and_then(|b| b.sequence())) {
        let kept: std::collections::HashSet<u64> = q.all_tracks().flat_map(|t| t.items.iter().map(|c| c.id.0)).collect();
        for hnt in b.manifest.variants.iter().flat_map(|v| v.hints.iter()) {
            if base_to_new.contains_key(&hnt.clip) {
                continue;
            }
            for kind in [TrackKind::Video, TrackKind::Audio] {
                for (ti, bt) in bq.tracks(kind).iter().enumerate() {
                    let Some(ci) = bt.items.iter().position(|c| c.id.0 == hnt.clip) else { continue };
                    let name = bt.items.get(ci).and_then(|c| b.fragment.item(c.item)).map(|i| i.name.clone());
                    if let Some(c) = src.tracks(kind).get(ti).and_then(|t| t.items.get(ci))
                        && kept.contains(&c.id.0)
                        && p.item(c.item).map(|i| i.name.clone()) == name
                    {
                        base_to_new.insert(hnt.clip, c.id.0);
                    }
                }
            }
        }
    }
    let seq_id = fragment.add_item(&opts.name, Label::Forest, ItemKind::Sequence(std::sync::Arc::new(q)), None);

    // the layouts: the sequence's own (as designed: no hints), and the others asked for
    let (w, h) = size;
    let own = aspect_id(w, h);
    let mut wanted: Vec<String> = vec![own.clone()];
    if opts.variants.is_empty() {
        match bm {
            Some(m) => wanted.extend(m.variants.iter().map(|v| v.id.clone())),
            None if w != h => wanted.push(aspect_id(h, w)),
            None => {}
        }
    } else {
        for v in &opts.variants {
            match crate::manifest::parse_aspect(&serde_json::Value::String(v.clone())) {
                Some([a, b]) => wanted.push(aspect_id(a, b)),
                None => warnings.push(format!("`{v}` is not a layout like 16:9; it was left out")),
            }
        }
    }
    let mut seen = Vec::new();
    wanted.retain(|v| {
        let k = v.to_ascii_lowercase();
        let fresh = !seen.contains(&k);
        seen.push(k);
        fresh
    });
    if wanted.len() > MAX_VARIANTS {
        warnings.push(format!("a template has at most {MAX_VARIANTS} layouts; the rest were left out"));
        wanted.truncate(MAX_VARIANTS);
    }
    let seq_preset = src_meta(p, seq, "Template Export Preset");
    let mut variants = Vec::new();
    for id in wanted {
        let bv = bm.and_then(|m| m.variants.iter().find(|v| v.id.eq_ignore_ascii_case(&id)));
        if id == own {
            variants.push(Variant {
                id: id.clone(),
                name: bv.map_or_else(|| variant_name(w, h), |v| v.name.clone()),
                width: w,
                height: h,
                export_preset: seq_preset.clone().or_else(|| bv.and_then(|v| v.export_preset.clone())),
                hints: Vec::new(),
            });
            continue;
        }
        let (vw, vh) = match bv {
            Some(v) => (v.width, v.height),
            None => match crate::manifest::parse_aspect(&serde_json::Value::String(id.clone())) {
                Some(a) => variant_size(a, size),
                None => continue,
            },
        };
        let hints = bv
            .map(|v| {
                v.hints.iter().filter_map(|hnt| base_to_new.get(&hnt.clip).map(|c| crate::manifest::Hint { clip: *c, ..hnt.clone() })).fold(
                    Vec::<crate::manifest::Hint>::new(),
                    |mut acc, hnt| {
                        if !acc.iter().any(|x| x.clip == hnt.clip) {
                            acc.push(hnt);
                        }
                        acc
                    },
                )
            })
            .unwrap_or_default();
        variants.push(Variant {
            id: id.clone(),
            name: bv.map_or_else(|| variant_name(vw, vh), |v| v.name.clone()),
            width: vw,
            height: vh,
            export_preset: bv.and_then(|v| v.export_preset.clone()),
            hints,
        });
    }
    let poster = opts.poster.or_else(|| bm.and_then(|m| m.poster)).map(|t| t.max(Tick::ZERO).min(duration));
    let id = if crate::manifest::is_template_id(&opts.id) && !opts.id.starts_with("builtin:") {
        opts.id.clone()
    } else {
        format!("user:{}", crate::slug(&opts.name))
    };
    let pick = |s: &str, from_base: Option<&String>, fallback: &str| -> String {
        if !s.trim().is_empty() { s.to_string() } else { from_base.filter(|b| !b.trim().is_empty()).cloned().unwrap_or_else(|| fallback.to_string()) }
    };
    let mut beat_grid = beat_grid.filter(|g| g.beats.len() >= 2);
    if let (Some(g), Some(bg)) = (beat_grid.as_mut(), bm.and_then(|m| m.beat_grid.as_ref())) {
        g.mode = bg.mode;
    }
    let m = Manifest {
        format: FORMAT.into(),
        format_version: FORMAT_VERSION,
        id,
        name: opts.name.clone(),
        description: pick(&opts.description, bm.map(|m| &m.description), ""),
        version: pick(&opts.version, None, "1.0.0"),
        author: opts.author.clone(),
        license: opts.license.clone(),
        category: pick(&opts.category, bm.map(|m| &m.category), "My Templates"),
        tags: if opts.tags.is_empty() { bm.map(|m| m.tags.clone()).unwrap_or_default() } else { opts.tags.clone() },
        sequence_file: SEQUENCE_FILE.into(),
        sequence: seq_id.0,
        duration,
        default_variant: own,
        variants,
        slots: manifest_slots,
        poster,
        look: opts.look.clone().or_else(|| bm.and_then(|m| m.look.clone())),
        caption_style,
        export_preset: opts.export_preset.clone().or_else(|| bm.and_then(|m| m.export_preset.clone())),
        beat_grid,
        assets,
    };
    // what the strict reader makes of it is what the template is
    let m = crate::manifest::parse(m.to_json().as_bytes())?;
    let package = Package { manifest: m, fragment, assets: asset_files, docs: BTreeMap::new() };
    package.check().into_result()?;
    Ok(Saved { package, warnings })
}

/// Metadata `key` of item `id`.
fn src_meta(p: &Project, id: ItemId, key: &str) -> Option<String> {
    p.item(id).and_then(|i| i.metadata.get(key).cloned()).filter(|s| !s.trim().is_empty())
}

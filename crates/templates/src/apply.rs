//! Applying a template: its sequence and placeholder media are copied into a project with fresh
//! ids, the cuts are snapped to the beat grid (re-quantized to the music's own beats when the
//! music slot gets a new track), every placement is filled with its slot's media under the
//! slot's auto-fit policy, the clips are laid out for the chosen aspect variant, and the look and
//! caption style are applied. Effects, transitions and titles belong to the placements, not to
//! the media: they stay where the template put them.
//!
//! Everything here is a pure edit of a [`Project`]; the engine wraps it in one undo step and does
//! what needs the media themselves (importing files, measuring beats, Remix).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use filmcraft_geom::Vec2;
use filmcraft_media::MediaKind;
use filmcraft_project::graphic::{self, TEXT_LAYER};
use filmcraft_project::gtemplate::layer_uid;
use filmcraft_project::{
    CaptionFormat, CaptionTrack, ClipId, EffectInstance, ItemId, ItemKind, Label, Marker, MarkerId, MarkerKind, MediaRef, Param, ParamValue, Project, Sequence,
    TrackId, TrackItem, TrackKind, TransitionAlign, TransitionId,
};
use filmcraft_time::{FrameRate, Tick};

use crate::beats::{self, Span, TimeMap};
use crate::fit::{self, Fitted};
use crate::manifest::{Framing, Hint, Manifest, QuantizeMode, ShortFit, Slot, SlotDefault, SlotKind, Variant};
use crate::{Package, TemplateError};

/// Name prefix of the clip markers that flag template slots (`slot:<kind>:<slot id>:<placement>`).
pub const SLOT_MARKER: &str = "slot:";

/// A clip's flag as a template slot placement (a clip marker named
/// `slot:<kind>:<slot id>[:<placement>[:<text layer uid>]]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotFlag {
    pub kind: SlotKind,
    pub slot: String,
    pub placement: Option<usize>,
    /// Text slots: the layer (two text slots may share a graphic clip).
    pub layer: Option<u64>,
}

/// The marker name flagging a clip as `placement` of slot `slot` (and, for text, its `layer`).
pub fn slot_marker_name(kind: SlotKind, slot: &str, placement: Option<usize>, layer: Option<u64>) -> String {
    match (placement, layer) {
        (Some(p), Some(l)) => format!("{SLOT_MARKER}{}:{slot}:{p}:{l}", kind.id()),
        (Some(p), None) => format!("{SLOT_MARKER}{}:{slot}:{p}", kind.id()),
        (None, _) => format!("{SLOT_MARKER}{}:{slot}", kind.id()),
    }
}

/// Read a slot marker name.
pub fn parse_slot_marker(name: &str) -> Option<SlotFlag> {
    let rest = name.strip_prefix(SLOT_MARKER)?;
    let mut parts = rest.splitn(4, ':');
    let kind = SlotKind::from_id(parts.next()?)?;
    let slot = parts.next().filter(|s| crate::manifest::is_slot_id(s)).map(str::to_string).unwrap_or_else(|| kind.id().to_string());
    let placement = parts.next().and_then(|p| p.parse::<usize>().ok()).filter(|p| *p < crate::manifest::MAX_PLACEMENTS);
    let layer = parts.next().and_then(|l| l.parse::<u64>().ok()).filter(|l| *l > 0 && *l < u64::MAX >> 2);
    Some(SlotFlag { kind, slot, placement, layer })
}

/// One medium filling a slot.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaFill {
    /// A media item of the target project.
    pub item: ItemId,
    /// Where in the media to start (media time); else the slot's `portion`.
    pub source_in: Option<Tick>,
    /// The point of the picture to keep in view (fractions of its width and height), all variants.
    pub focus: Option<[f64; 2]>,
    /// The same per variant id (the same photo re-framed for 16:9 and 9:16).
    pub focus_by_variant: BTreeMap<String, [f64; 2]>,
}

impl MediaFill {
    pub fn new(item: ItemId) -> MediaFill {
        MediaFill { item, source_in: None, focus: None, focus_by_variant: BTreeMap::new() }
    }
    fn focus_for(&self, variant: &str) -> [f64; 2] {
        self.focus_by_variant.iter().find(|(k, _)| k.eq_ignore_ascii_case(variant)).map(|(_, v)| *v).or(self.focus).unwrap_or([0.5, 0.5])
    }
}

/// What fills a slot.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotFill {
    Media(Vec<MediaFill>),
    Text(Vec<String>),
}

/// The beats of the music filling the music slot, in media time of its item.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MusicBeats {
    pub beats: Vec<Tick>,
}

/// How to apply a template.
#[derive(Clone, Debug, Default)]
pub struct ApplyOptions {
    /// Variant id or name (default: the template's default variant).
    pub variant: Option<String>,
    /// Name of the new sequence (default: "<template> (<variant>)").
    pub name: Option<String>,
    pub fills: BTreeMap<String, SlotFill>,
    /// Beats of the music fill: the cuts are re-quantized to them. `None` keeps the template's
    /// beat grid (a generated default bed plays at the template's tempo).
    pub music_beats: Option<MusicBeats>,
    /// The look as an effect instance (resolved by the engine from the manifest's reference).
    pub look: Option<EffectInstance>,
    /// Where the package's `assets/` folder is on disk (its parent folder), for templates with
    /// bundled files.
    pub asset_dir: Option<String>,
    /// Refuse to leave a required slot unfilled (otherwise its placeholder stays, with a warning:
    /// a preview of the template).
    pub require_all: bool,
}

/// One filled (or placeholder) placement.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedClip {
    pub slot: String,
    pub placement: usize,
    /// Which of the slot's media (None: the placeholder stayed).
    pub media: Option<usize>,
    pub item: ItemId,
    /// The clips it became (several when it loops or holds).
    pub clips: Vec<ClipId>,
    pub how: String,
    pub start: Tick,
    pub duration: Tick,
    pub source_in: Tick,
    pub speed: f64,
}

/// What [`apply`] made.
#[derive(Clone, Debug, PartialEq)]
pub struct Applied {
    pub sequence: ItemId,
    pub variant: String,
    pub duration: Tick,
    pub placed: Vec<PlacedClip>,
    /// The beat-synced cuts as applied.
    pub cuts: Vec<Tick>,
    /// The cuts moved to new beats.
    pub retimed: bool,
    /// Music clips the engine Remixes to this length.
    pub remix: Vec<(ClipId, Tick)>,
    pub export_preset: Option<String>,
    pub warnings: Vec<String>,
    /// The template's own items added (placeholders, graphics) and their bin.
    pub items: Vec<ItemId>,
    pub bin: filmcraft_project::BinId,
}

fn err(m: impl Into<String>) -> TemplateError {
    TemplateError::Apply(m.into())
}

/// The media length a fit works with: `None` for pictures without one (stills, generated media).
pub fn media_length(p: &Project, item: ItemId) -> Option<Tick> {
    let m = p.item(item)?.as_media()?;
    match m.info.kind {
        MediaKind::Still | MediaKind::Synthetic => None,
        _ => Some(m.info.duration),
    }
}

fn has_audio(p: &Project, item: ItemId) -> bool {
    p.item(item).is_some_and(|i| i.has_audio())
}

/// Check the fills against the slots: known slots, kinds, counts, text lengths, media shapes.
fn check_fills(p: &Project, m: &Manifest, opts: &ApplyOptions, warnings: &mut Vec<String>) -> Result<(), TemplateError> {
    for (id, fill) in &opts.fills {
        let slot = m.slot(id).ok_or_else(|| {
            let ids: Vec<&str> = m.slots.iter().map(|s| s.id.as_str()).collect();
            err(format!("the template has no slot `{id}` (its slots: {})", ids.join(", ")))
        })?;
        let n = match fill {
            SlotFill::Text(t) => {
                if slot.kind != SlotKind::Text {
                    return Err(err(format!("slot `{id}` takes {} media, not text", slot.kind.id())));
                }
                for s in t {
                    if let Some(max) = slot.constraints.max_chars
                        && s.chars().count() > max as usize
                    {
                        return Err(err(format!("slot `{id}`: “{s}” is longer than {max} characters")));
                    }
                }
                t.len()
            }
            SlotFill::Media(list) => {
                if slot.kind == SlotKind::Text {
                    return Err(err(format!("slot `{id}` takes text, not media")));
                }
                for f in list {
                    let it = p.item(f.item).ok_or_else(|| err(format!("slot `{id}`: item {} is not in the project", f.item.0)))?;
                    let mc = it.as_media().ok_or_else(|| err(format!("slot `{id}`: “{}” is not a media item", it.name)))?;
                    if slot.kind == SlotKind::Music {
                        if !mc.info.has_audio() {
                            return Err(err(format!("slot `{id}` takes music, and “{}” has no sound", it.name)));
                        }
                    } else if !mc.info.has_video() {
                        return Err(err(format!("slot `{id}` takes a picture, and “{}” has none", it.name)));
                    }
                    if let Some(v) = &mc.info.video
                        && slot.kind.is_visual()
                    {
                        let (w, h) = filmcraft_project::conformed_size((v.width, v.height), mc.pixel_aspect(), (1, 1));
                        let c = &slot.constraints;
                        let mut wrong = Vec::new();
                        if !c.orientation.accepts(w, h) {
                            wrong.push(format!("is not {}", c.orientation.id()));
                        }
                        if let Some([a, b]) = c.aspect
                            && h > 0.0
                            && ((w / h) / (f64::from(a) / f64::from(b)) - 1.0).abs() > 0.05
                        {
                            wrong.push(format!("is not {a}:{b}"));
                        }
                        if !wrong.is_empty() {
                            let msg = format!("slot `{id}`: “{}” ({w:.0}×{h:.0}) {}", it.name, wrong.join(" and "));
                            if c.strict {
                                return Err(err(msg));
                            }
                            warnings.push(format!("{msg}; it is framed to fit the slot"));
                        }
                    }
                    if let Some(i) = f.source_in
                        && i < Tick::ZERO
                    {
                        return Err(err(format!("slot `{id}`: the in point is negative")));
                    }
                    for (k, v) in f.focus.iter().map(|v| ("", v)).chain(f.focus_by_variant.iter().map(|(k, v)| (k.as_str(), v))) {
                        if !v.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)) {
                            return Err(err(format!("slot `{id}`: focus {k} must be two fractions from 0 to 1")));
                        }
                    }
                }
                list.len()
            }
        };
        if n > slot.count as usize {
            return Err(err(format!("slot `{id}` takes at most {} (got {n})", slot.count)));
        }
        if n > 0 && n < slot.min_count as usize {
            return Err(err(format!("slot `{id}` needs at least {} (got {n})", slot.min_count)));
        }
    }
    for s in &m.slots {
        let given = opts.fills.get(&s.id).map_or(0, |f| match f {
            SlotFill::Media(l) => l.len(),
            SlotFill::Text(t) => t.len(),
        });
        if given == 0 && s.min_count > 0 && s.kind != SlotKind::Text && !matches!(s.default, Some(SlotDefault::Generate { .. })) {
            let msg = format!("slot `{}` ({}) needs {} {}", s.id, s.name, s.min_count, if s.min_count == 1 { "item" } else { "items" });
            if opts.require_all {
                return Err(err(msg));
            }
            warnings.push(format!("{msg}; its placeholder stays"));
        }
    }
    Ok(())
}

/// Old → new ids of the copied sequence.
#[derive(Default)]
struct Ids {
    clips: HashMap<u64, ClipId>,
}

fn remap_sequence(seq: &mut Sequence, p: &mut Project, items: &HashMap<ItemId, ItemId>) -> Result<Ids, TemplateError> {
    let mut ids = Ids::default();
    let mut tracks: HashMap<u64, TrackId> = HashMap::new();
    let mut links: HashMap<u64, u64> = HashMap::new();
    let mut groups: HashMap<u64, u64> = HashMap::new();
    for t in seq.video_tracks.iter_mut().chain(seq.audio_tracks.iter_mut()).chain(seq.submix_tracks.iter_mut()) {
        let id = TrackId(p.alloc_id());
        tracks.insert(t.id.0, id);
        t.id = id;
    }
    for t in seq.caption_tracks.iter_mut() {
        let id = TrackId(p.alloc_id());
        tracks.insert(t.id.0, id);
        t.id = id;
        for c in t.captions.iter_mut() {
            c.id = ClipId(p.alloc_id());
        }
    }
    for t in seq.video_tracks.iter_mut().chain(seq.audio_tracks.iter_mut()) {
        for c in t.items.iter_mut() {
            let id = ClipId(p.alloc_id());
            ids.clips.insert(c.id.0, id);
            c.id = id;
            c.item = *items.get(&c.item).ok_or_else(|| err(format!("clip “{}” uses an item that is not in the template", c.name)))?;
            c.link = c.link.map(|l| *links.entry(l).or_insert_with(|| p.alloc_id()));
            c.group = c.group.map(|g| *groups.entry(g).or_insert_with(|| p.alloc_id()));
            for mk in c.markers.iter_mut() {
                mk.id = MarkerId(p.alloc_id());
            }
            c.multicam = None;
        }
    }
    for t in seq.video_tracks.iter_mut().chain(seq.audio_tracks.iter_mut()) {
        for tr in t.transitions.iter_mut() {
            tr.id = TransitionId(p.alloc_id());
            tr.from = tr.from.and_then(|c| ids.clips.get(&c.0).copied());
            tr.to = tr.to.and_then(|c| ids.clips.get(&c.0).copied());
        }
        t.transitions.retain(|tr| tr.from.is_some() || tr.to.is_some());
    }
    for t in seq.video_tracks.iter_mut().chain(seq.audio_tracks.iter_mut()).chain(seq.submix_tracks.iter_mut()) {
        t.mixer.output = t.mixer.output.and_then(|o| tracks.get(&o.0).copied());
        t.mixer.sends.retain(|s| tracks.contains_key(&s.target.0));
        for s in t.mixer.sends.iter_mut() {
            if let Some(n) = tracks.get(&s.target.0) {
                s.target = *n;
            }
        }
    }
    seq.master_mixer.output = None;
    for mk in seq.markers.iter_mut() {
        mk.id = MarkerId(p.alloc_id());
    }
    seq.multicam = None;
    seq.merged = None;
    Ok(ids)
}

fn locate(seq: &Sequence, clip: ClipId) -> Option<(TrackKind, usize, usize)> {
    for (kind, tracks) in [(TrackKind::Video, &seq.video_tracks), (TrackKind::Audio, &seq.audio_tracks)] {
        for (ti, t) in tracks.iter().enumerate() {
            if let Some(ci) = t.items.iter().position(|c| c.id == clip) {
                return Some((kind, ti, ci));
            }
        }
    }
    None
}

fn clip_ref(seq: &Sequence, clip: ClipId) -> Option<&TrackItem> {
    seq.find_item(clip).map(|(_, c)| c)
}

/// Keyframes of `it` (media time) re-timed for a new duration, relative to its in point.
fn scale_keyframes(it: &mut TrackItem, old: Tick, new: Tick) {
    if old.0 <= 0 || old == new {
        return;
    }
    let speed = it.speed.abs().max(1e-6);
    let r = new.0 as f64 / old.0 as f64;
    let src_in = it.source_in;
    for e in it.effects.iter_mut() {
        for prm in e.params.values_mut() {
            for k in prm.keyframes.iter_mut() {
                let rel = (k.time - src_in).0 as f64 / speed;
                k.time = src_in + Tick((rel * r * speed).round() as i64);
            }
        }
    }
}

/// Re-time the whole sequence through `map` (times land on frames of `rate`).
pub fn apply_time_map(seq: &mut Sequence, map: &TimeMap, rate: FrameRate) {
    if map.is_identity() {
        return;
    }
    let frame = rate.frame_duration().max(Tick(1));
    let f = |t: Tick| rate.snap_nearest(map.map(t));
    for t in seq.video_tracks.iter_mut().chain(seq.audio_tracks.iter_mut()) {
        for it in t.items.iter_mut() {
            let (s, e) = (f(it.start), f(it.end()));
            let d = (e - s).max(frame);
            let responsive = it.graphic.as_ref().is_some_and(|g| g.has_responsive_time());
            if !responsive {
                scale_keyframes(it, it.duration, d);
            }
            it.start = s;
            it.duration = d;
        }
        for tr in t.transitions.iter_mut() {
            let cut = match tr.align {
                TransitionAlign::CenterAtCut => tr.start + Tick(tr.duration.0 / 2),
                TransitionAlign::StartAtCut => tr.start,
                TransitionAlign::EndAtCut => tr.end(),
            };
            let lead = cut - tr.start;
            tr.start = rate.snap_nearest(map.map(cut) - lead);
        }
        t.sort();
    }
    for t in seq.caption_tracks.iter_mut() {
        for c in t.captions.iter_mut() {
            let (s, e) = (f(c.start), f(c.end()));
            c.start = s;
            c.duration = (e - s).max(frame);
        }
    }
    for mk in seq.markers.iter_mut() {
        let end = f(mk.start + mk.duration);
        mk.start = f(mk.start);
        mk.duration = (end - mk.start).max(Tick::ZERO);
    }
    seq.mark_in = seq.mark_in.map(f);
    seq.mark_out = seq.mark_out.map(f);
    if let Some(w) = &mut seq.work_area {
        let end = f(w.end());
        w.start = f(w.start);
        w.duration = (end - w.start).max(frame);
    }
}

/// The music slot's (first) placement clip, if any.
fn music_clip<'a>(m: &'a Manifest, ids: &Ids) -> Option<(&'a Slot, ClipId)> {
    let s = m.slots.iter().find(|s| s.kind == SlotKind::Music)?;
    let c = ids.clips.get(&s.placements.first()?.clip).copied()?;
    Some((s, c))
}

/// Snap the beat-synced cuts to the grid, or re-quantize them to the new music's beats.
fn retime(seq: &mut Sequence, m: &Manifest, ids: &Ids, opts: &ApplyOptions, warnings: &mut Vec<String>) -> (Vec<Tick>, bool) {
    let Some(grid) = &m.beat_grid else { return (Vec::new(), false) };
    let rate = seq.settings.frame_rate;
    let mut edges: Vec<(Tick, Tick, Tick, Tick)> = Vec::new();
    for s in m.slots.iter().filter(|s| s.beat_sync) {
        for pl in &s.placements {
            if let Some(c) = ids.clips.get(&pl.clip).and_then(|c| clip_ref(seq, *c)) {
                edges.push((c.start, c.end(), s.constraints.min_duration.unwrap_or(rate.frame_duration()), s.constraints.max_duration.unwrap_or(Tick::MAX)));
            }
        }
    }
    let music = music_clip(m, ids);
    let anchor = music.and_then(|(_, c)| clip_ref(seq, c)).map_or(Tick::ZERO, |c| c.start);
    let mut cuts: Vec<Tick> = edges.iter().flat_map(|e| [e.0, e.1]).filter(|t| *t >= anchor).collect();
    cuts.sort_unstable();
    cuts.dedup();
    if cuts.is_empty() {
        return (Vec::new(), false);
    }
    let spans: Vec<Span> =
        edges.iter().filter_map(|e| Some(Span { from: cuts.binary_search(&e.0).ok()?, to: cuts.binary_search(&e.1).ok()?, min: e.2, max: e.3 })).collect();
    let new_music = music.and_then(|(s, c)| match (opts.fills.get(&s.id), &opts.music_beats) {
        (Some(SlotFill::Media(list)), Some(b)) => {
            let f = list.first()?;
            let start = clip_ref(seq, c)?.start;
            let from = f.source_in.unwrap_or(Tick::ZERO);
            Some(b.beats.iter().filter(|t| **t >= from).map(|t| start + (*t - from)).collect::<Vec<Tick>>())
        }
        _ => None,
    });
    let (new_grid, mode) = match new_music {
        Some(g) if g.len() >= 2 => (g, grid.mode),
        Some(_) => {
            warnings.push("the music's beats could not be found; the cuts stay on the template's beat grid".into());
            (grid.beats.clone(), QuantizeMode::Nearest)
        }
        None => (grid.beats.clone(), QuantizeMode::Nearest),
    };
    let (new_cuts, w) = beats::quantize(&cuts, anchor, &grid.beats, &new_grid, mode, &spans, rate);
    warnings.extend(w);
    let mut pts = vec![(anchor, anchor)];
    pts.extend(cuts.iter().copied().zip(new_cuts.iter().copied()).filter(|(o, _)| *o != anchor));
    let map = TimeMap::new(pts);
    let changed = !map.is_identity();
    apply_time_map(seq, &map, rate);
    (new_cuts, changed)
}

/// Re-base the keyframes of piece `c` (cut from template clip `t` at timeline `offset`).
fn rebase_keyframes(c: &mut TrackItem, t: &TrackItem, offset: Tick) {
    let ts = t.speed.abs().max(1e-6);
    let cs = c.speed.abs().max(1e-6);
    let (t_in, c_in) = (t.source_in, c.source_in);
    for e in c.effects.iter_mut() {
        for prm in e.params.values_mut() {
            for k in prm.keyframes.iter_mut() {
                let rel = (k.time - t_in).0 as f64 / ts;
                k.time = c_in + Tick(((rel - offset.0 as f64) * cs).round() as i64);
            }
        }
    }
}

fn slot_marker(p: &mut Project, slot: &Slot, placement: usize, layer: Option<u64>, at: Tick) -> Marker {
    Marker {
        id: MarkerId(p.alloc_id()),
        start: at,
        duration: Tick::ZERO,
        name: slot_marker_name(slot.kind, &slot.id, Some(placement), layer),
        comment: slot.name.clone(),
        kind: MarkerKind::Comment,
        color: Label::Mango,
    }
}

/// Flag `it` as `placement` of `slot` (replacing an earlier flag of the same slot and layer; a
/// graphic clip shared by two text slots keeps both).
fn set_slot_marker(p: &mut Project, it: &mut TrackItem, slot: &Slot, placement: usize, layer: Option<u64>) {
    it.markers.retain(|mk| match parse_slot_marker(&mk.name) {
        Some(f) => slot.kind == SlotKind::Text && f.kind == SlotKind::Text && (f.slot != slot.id || f.layer != layer),
        None => !mk.name.starts_with(SLOT_MARKER),
    });
    let mk = slot_marker(p, slot, placement, layer, it.source_in);
    it.markers.push(mk);
}

/// Pieces of `tmpl` showing `item` as `fitted` says.
fn pieces_of(p: &mut Project, tmpl: &TrackItem, item: ItemId, fitted: &Fitted, slot: &Slot, placement: usize, keep_id: bool) -> Vec<TrackItem> {
    let name = p.item(item).map(|i| i.name.clone()).unwrap_or_default();
    let was_graphic = tmpl.graphic.is_some() || tmpl.effects.iter().any(graphic::is_layer);
    let mut out = Vec::new();
    for (k, pc) in fitted.pieces.iter().enumerate() {
        let mut c = tmpl.clone();
        c.id = if k == 0 && keep_id { tmpl.id } else { ClipId(p.alloc_id()) };
        c.item = item;
        c.name = name.clone();
        c.start = tmpl.start + pc.offset;
        c.duration = pc.duration;
        c.source_in = pc.source_in;
        c.speed = pc.speed;
        c.reverse = false;
        c.frame_hold = pc.hold;
        c.multicam = None;
        c.audio_stream = 0;
        c.source_channels = Vec::new();
        if was_graphic {
            c.effects.retain(|e| !graphic::is_layer(e));
            c.graphic = None;
        }
        rebase_keyframes(&mut c, tmpl, pc.offset);
        set_slot_marker(p, &mut c, slot, placement, None);
        out.push(c);
    }
    out
}

/// Replace the clip at `at` with `pieces`; transitions out of it follow the last piece.
fn replace_clip(seq: &mut Sequence, at: (TrackKind, usize, usize), pieces: Vec<TrackItem>) -> Option<Vec<ClipId>> {
    let (kind, ti, ci) = at;
    let track = seq.tracks_mut(kind).get_mut(ti)?;
    let old = track.items.get(ci)?.id;
    let ids: Vec<ClipId> = pieces.iter().map(|c| c.id).collect();
    let (first, last) = (*ids.first()?, *ids.last()?);
    let _replaced: Vec<TrackItem> = track.items.splice(ci..=ci, pieces).collect();
    for tr in track.transitions.iter_mut() {
        if tr.from == Some(old) {
            tr.from = Some(last);
        }
        if tr.to == Some(old) {
            tr.to = Some(first);
        }
    }
    Some(ids)
}

/// Fill one placement with one medium: the clip and its linked partners.
fn place(
    seq: &mut Sequence,
    p: &mut Project,
    clip: ClipId,
    mf: &MediaFill,
    slot: &Slot,
    placement: usize,
    media_index: usize,
) -> Result<(PlacedClip, Option<(ClipId, Tick)>, Option<String>), TemplateError> {
    let rate = seq.settings.frame_rate;
    let at = locate(seq, clip).ok_or_else(|| err(format!("slot `{}`: its clip is missing", slot.id)))?;
    let tmpl = clip_ref(seq, clip).cloned().ok_or_else(|| err(format!("slot `{}`: its clip is missing", slot.id)))?;
    let len = media_length(p, mf.item);
    let fitted = fit::fit(tmpl.duration, len, mf.source_in, &slot.fit, rate).map_err(|e| err(format!("slot `{}`: {e}", slot.id)))?;
    let pieces = pieces_of(p, &tmpl, mf.item, &fitted, slot, placement, true);
    let first = pieces.first().map(|c| (c.source_in, c.speed)).unwrap_or((Tick::ZERO, 1.0));
    let clips = replace_clip(seq, at, pieces).ok_or_else(|| err(format!("slot `{}`: its clip is missing", slot.id)))?;
    // linked partners (the placeholder's own sound, a picture for a sound) follow the media, or
    // go when the media has none of that kind
    if let Some(l) = tmpl.link {
        let other = if at.0 == TrackKind::Video { TrackKind::Audio } else { TrackKind::Video };
        let wants = match other {
            TrackKind::Audio => has_audio(p, mf.item),
            TrackKind::Video => p.item(mf.item).is_some_and(|i| i.has_video()),
        };
        let partners: Vec<ClipId> = seq.tracks(other).iter().flat_map(|t| t.items.iter()).filter(|c| c.link == Some(l)).map(|c| c.id).collect();
        let mut pair_links: Vec<u64> = Vec::new();
        for pid in partners {
            let (Some(pat), Some(ptmpl)) = (locate(seq, pid), clip_ref(seq, pid).cloned()) else { continue };
            if wants && ptmpl.start == tmpl.start && ptmpl.duration == tmpl.duration {
                let mut pp = pieces_of(p, &ptmpl, mf.item, &fitted, slot, placement, true);
                if pair_links.is_empty() {
                    pair_links = (0..pp.len()).map(|_| p.alloc_id()).collect();
                }
                for (c, lk) in pp.iter_mut().zip(pair_links.iter()) {
                    c.link = Some(*lk);
                }
                replace_clip(seq, pat, pp);
            } else if let Some(t) = seq.tracks_mut(other).get_mut(pat.1) {
                t.items.retain(|c| c.id != pid);
                t.transitions.retain(|tr| tr.from != Some(pid) && tr.to != Some(pid));
            }
        }
        let mine = seq.tracks_mut(at.0).get_mut(at.1);
        if let Some(t) = mine {
            for c in t.items.iter_mut().filter(|c| clips.contains(&c.id)) {
                let i = clips.iter().position(|x| *x == c.id).unwrap_or(0);
                c.link = pair_links.get(i).copied();
            }
        }
    }
    let remix = fitted.remix_to.and_then(|to| clips.first().map(|c| (*c, to)));
    Ok((
        PlacedClip {
            slot: slot.id.clone(),
            placement,
            media: Some(media_index),
            item: mf.item,
            clips,
            how: fitted.how.to_string(),
            start: tmpl.start,
            duration: tmpl.duration,
            source_in: first.0,
            speed: first.1,
        },
        remix,
        fitted.note.map(|n| format!("slot `{}`: {n}", slot.id)),
    ))
}

/// The uid of a graphic clip's first text layer.
fn first_text_layer(it: &TrackItem) -> Option<u64> {
    it.effects.iter().find(|e| e.effect == TEXT_LAYER).map(layer_uid).filter(|u| *u > 0)
}

/// Set the text of a text layer (`layer` uid, else the first) of a graphic clip.
fn set_text(seq: &mut Sequence, clip: ClipId, layer: Option<u64>, text: &str) -> bool {
    let Some((_, it)) = seq.find_item_mut(clip) else { return false };
    let Some(e) = it.effects.iter_mut().find(|e| e.effect == TEXT_LAYER && layer.is_none_or(|l| layer_uid(e) == l)) else { return false };
    e.params.insert("text".into(), Param::new(ParamValue::Text(text.to_string())));
    if let Some(x) = e.layer.as_mut() {
        // per-character styles index the old text
        x.runs.clear();
    }
    true
}

/// `x` kept within `[lo, hi]`; the middle when the bounds cross (never `f64::clamp`, which
/// panics on crossed or NaN bounds).
fn between(x: f64, lo: f64, hi: f64) -> f64 {
    if lo <= hi { x.max(lo).min(hi) } else { (lo + hi) / 2.0 }
}

fn map_vec2(prm: &mut Param, f: impl Fn(Vec2) -> Vec2) {
    prm.map_values(|v| match v {
        ParamValue::Vec2(q) if !q.x.is_nan() && !q.y.is_nan() => ParamValue::Vec2(f(q)),
        o => o,
    });
}

fn map_f64(prm: &mut Param, f: impl Fn(f64) -> f64) {
    prm.map_values(|v| match v {
        ParamValue::Float(x) => ParamValue::Float(f(x)),
        o => o,
    });
}

/// Frame a picture clip in a `frame`-sized frame designed at `design`: the template's Motion
/// values are relative to the slot's framing (scale 100 = exactly filling / fitting; position in
/// the design frame), and the focus point of the picture lands where the hint or the template
/// puts it.
fn frame_clip(it: &mut TrackItem, src: (f64, f64), design: (f64, f64), frame: (f64, f64), framing: Framing, focus: [f64; 2], hint: Option<&Hint>) {
    let ((cw, ch), (w0, h0), (w, h)) = (src, design, frame);
    if !(cw > 0.0 && ch > 0.0 && w0 > 0.0 && h0 > 0.0) {
        return;
    }
    let k = match hint.and_then(|h| h.frame).unwrap_or(framing) {
        Framing::Fill => (w / cw).max(h / ch),
        Framing::Fit => (w / cw).min(h / ch),
        Framing::None => 1.0,
    };
    it.scale_to_frame = false;
    let Some(mo) = it.effects.iter_mut().find(|e| e.effect == "motion") else { return };
    // a hint's scale is a percentage of the template's own (animated or not)
    let k_scale = k * hint.and_then(|h| h.scale).map_or(1.0, |s| s / 100.0);
    let sp = mo.params.entry("scale".into()).or_insert_with(|| Param::new(ParamValue::Float(100.0)));
    let scale_static = sp.value.as_f64().unwrap_or(100.0) * k_scale / k.max(1e-9);
    map_f64(sp, |v| v * k_scale);
    let shown = scale_static * k / 100.0;
    let off = Vec2::new((focus[0] - 0.5) * cw * shown, (focus[1] - 0.5) * ch * shown);
    let covering = cw * shown >= w - 0.5 && ch * shown >= h - 0.5;
    let fill = hint.and_then(|h| h.frame).unwrap_or(framing) == Framing::Fill;
    let hinted = hint.and_then(|h| h.position);
    let pos = mo.params.entry("position".into()).or_insert_with(|| Param::new(ParamValue::Vec2(Vec2::new(f64::NAN, f64::NAN))));
    let animated = pos.is_animated();
    let base = |q: Vec2| -> Vec2 { if q.x.is_nan() || q.y.is_nan() { Vec2::new(w / 2.0, h / 2.0) } else { Vec2::new(q.x / w0 * w, q.y / h0 * h) } };
    let clamp = |v: Vec2| -> Vec2 {
        if !fill || !covering || animated {
            return v;
        }
        let (hw, hh) = (cw * shown / 2.0, ch * shown / 2.0);
        Vec2::new(between(v.x, w - hw, hw), between(v.y, h - hh, hh))
    };
    if animated {
        map_vec2(pos, |q| base(q) - off);
    } else {
        let at = match hinted {
            Some([x, y]) => Vec2::new(x * w, y * h),
            None => base(pos.value.as_vec2().unwrap_or(Vec2::new(f64::NAN, f64::NAN))),
        };
        pos.value = ParamValue::Vec2(clamp(at - off));
    }
    // the framing assumes the picture's centre as anchor
    if let Some(a) = mo.params.get_mut("anchor")
        && !a.is_animated()
    {
        a.value = ParamValue::Vec2(Vec2::new(f64::NAN, f64::NAN));
    }
}

/// Lay a graphic clip's layers out for a new frame: positions keep their place as fractions of
/// the frame, each layer scales with the frame's smaller ratio, hints override both.
fn layout_graphic(it: &mut TrackItem, design: (f64, f64), frame: (f64, f64), hint: Option<&Hint>) {
    let ((w0, h0), (w, h)) = (design, frame);
    if !(w0 > 0.0 && h0 > 0.0) {
        return;
    }
    let s = (w / w0).min(h / h0);
    for e in it.effects.iter_mut().filter(|e| graphic::is_layer(e)) {
        let uid = layer_uid(e);
        let lh = hint.and_then(|h| h.layers.iter().find(|l| l.layer == uid));
        if let Some(pp) = e.params.get_mut("position") {
            map_vec2(pp, |q| Vec2::new(q.x / w0 * w, q.y / h0 * h));
            if let Some([x, y]) = lh.and_then(|l| l.position) {
                let now = pp.value.as_vec2().filter(|v| !v.x.is_nan()).unwrap_or(Vec2::new(w / 2.0, h / 2.0));
                let d = Vec2::new(x * w, y * h) - now;
                if pp.is_animated() {
                    map_vec2(pp, |q| q + d);
                } else {
                    pp.value = ParamValue::Vec2(Vec2::new(x * w, y * h));
                }
            }
        }
        match lh.and_then(|l| l.size) {
            Some(size) if e.effect == TEXT_LAYER => {
                e.params.insert("size".into(), Param::new(ParamValue::Float(size)));
            }
            _ => {
                if let Some(sc) = e.params.get_mut("scale") {
                    map_f64(sc, |v| v * s);
                } else {
                    e.params.insert("scale".into(), Param::new(ParamValue::Float(100.0 * s)));
                }
            }
        }
        if let Some(pin) = e.layer.as_mut().and_then(|x| x.pin.as_mut()) {
            pin.offsets.iter_mut().for_each(|o| *o *= s);
        }
    }
    // a graphic clip's own Motion position, if the template set one
    if let Some(pp) = it.effects.iter_mut().find(|e| e.effect == "motion").and_then(|m| m.params.get_mut("position")) {
        map_vec2(pp, |q| Vec2::new(q.x / w0 * w, q.y / h0 * h));
    }
}

/// Apply `pkg` to `p`: a new sequence (returned with what was filled and how).
pub fn apply(p: &mut Project, pkg: &Package, opts: &ApplyOptions) -> Result<Applied, TemplateError> {
    let m = &pkg.manifest;
    let variant: Variant = m.variant(opts.variant.as_deref()).cloned().ok_or_else(|| {
        let ids: Vec<&str> = m.variants.iter().map(|v| v.id.as_str()).collect();
        err(format!("the template has no variant `{}` (its variants: {})", opts.variant.as_deref().unwrap_or(""), ids.join(", ")))
    })?;
    let base = pkg.sequence().ok_or_else(|| err("the template has no sequence"))?;
    let design = (f64::from(base.settings.width), f64::from(base.settings.height));
    let (w1, h1) = (variant.width, variant.height);
    filmcraft_project::validate_frame_size(w1, h1).map_err(err)?;
    let mut warnings = Vec::new();
    check_fills(p, m, opts, &mut warnings)?;

    // the template's own items, with fresh ids, in a bin of their own
    let bin = p.add_bin(&format!("{} – template media", m.name), None);
    let main = ItemId(m.sequence);
    let mut items: HashMap<ItemId, ItemId> = HashMap::new();
    let mut added = Vec::new();
    for (old, it) in &pkg.fragment.items {
        if *old == main {
            continue;
        }
        let mut kind = it.kind.clone();
        match &mut kind {
            ItemKind::Media(mc) => {
                if let MediaRef::File { path } = &mc.media {
                    let dir =
                        opts.asset_dir.as_deref().ok_or_else(|| err(format!("“{}” is a bundled file, and the package's folder is not known", it.name)))?;
                    mc.media = MediaRef::File { path: format!("{}/{path}", dir.trim_end_matches('/')) };
                }
                // generated full-frame placeholders are resolution-independent: draw them at the
                // variant's frame size (smaller ones, such as a square logo stand-in, keep theirs)
                if let (MediaRef::Generator(_), Some(v)) = (&mc.media, mc.info.video.as_mut())
                    && (f64::from(v.width), f64::from(v.height)) == design
                {
                    v.width = w1;
                    v.height = h1;
                }
            }
            ItemKind::Graphic { width, height, .. } | ItemKind::AdjustmentLayer { width, height, .. } => {
                *width = w1;
                *height = h1;
            }
            _ => continue,
        }
        let id = p.add_item(&it.name, it.label, kind, Some(bin));
        if let Some(n) = p.item_mut(id) {
            n.metadata = it.metadata.clone();
        }
        items.insert(*old, id);
        added.push(id);
    }
    let mut seq = base.clone();
    let ids = remap_sequence(&mut seq, p, &items)?;
    seq.settings.width = w1;
    seq.settings.height = h1;
    seq.settings.par = (1, 1);

    // cuts on the beat
    let (cuts, retimed) = retime(&mut seq, m, &ids, opts, &mut warnings);

    // fill the slots, in their order
    let mut slots: Vec<&Slot> = m.slots.iter().collect();
    slots.sort_by_key(|s| s.order);
    let mut placed = Vec::new();
    let mut remix = Vec::new();
    let mut origin: HashMap<ClipId, (u64, Option<String>, usize)> = HashMap::new();
    for slot in slots {
        let fill = opts.fills.get(&slot.id);
        for (pi, pl) in slot.placements.iter().enumerate() {
            let Some(clip) = ids.clips.get(&pl.clip).copied() else { continue };
            match (slot.kind, fill) {
                (SlotKind::Text, _) => {
                    let texts = match fill {
                        Some(SlotFill::Text(t)) => t.as_slice(),
                        _ => &[],
                    };
                    let text = if texts.is_empty() {
                        match &slot.default {
                            Some(SlotDefault::Text(t)) => Some(t.as_str()),
                            _ => None,
                        }
                    } else {
                        texts.get(pl.media as usize).or_else(|| texts.get(pl.media as usize % texts.len())).map(String::as_str)
                    };
                    if let Some(t) = text
                        && !set_text(&mut seq, clip, pl.layer, t)
                    {
                        warnings.push(format!("slot `{}`: its clip has no text layer", slot.id));
                    }
                    if let Some((_, it)) = seq.find_item_mut(clip) {
                        let layer = pl.layer.or_else(|| first_text_layer(it));
                        set_slot_marker(p, it, slot, pi, layer);
                    }
                    if let Some(c) = clip_ref(&seq, clip) {
                        placed.push(PlacedClip {
                            slot: slot.id.clone(),
                            placement: pi,
                            media: (!texts.is_empty()).then_some(pl.media as usize),
                            item: c.item,
                            clips: vec![clip],
                            how: if text.is_some() { "text".into() } else { "placeholder".into() },
                            start: c.start,
                            duration: c.duration,
                            source_in: c.source_in,
                            speed: c.speed,
                        });
                    }
                    origin.insert(clip, (pl.clip, None, pi));
                }
                (_, Some(SlotFill::Media(list))) if !list.is_empty() => {
                    let mi = if (pl.media as usize) < list.len() { pl.media as usize } else { pl.media as usize % list.len() };
                    let Some(mf) = list.get(mi) else { continue };
                    let (pc, rmx, note) = place(&mut seq, p, clip, mf, slot, pi, mi)?;
                    for c in &pc.clips {
                        origin.insert(*c, (pl.clip, Some(slot.id.clone()), mi));
                    }
                    remix.extend(rmx);
                    warnings.extend(note);
                    placed.push(pc);
                }
                _ => {
                    if let Some((_, it)) = seq.find_item_mut(clip) {
                        set_slot_marker(p, it, slot, pi, None);
                    }
                    if let Some(c) = clip_ref(&seq, clip) {
                        placed.push(PlacedClip {
                            slot: slot.id.clone(),
                            placement: pi,
                            media: None,
                            item: c.item,
                            clips: vec![clip],
                            how: "placeholder".into(),
                            start: c.start,
                            duration: c.duration,
                            source_in: c.source_in,
                            speed: c.speed,
                        });
                    }
                    origin.insert(clip, (pl.clip, Some(slot.id.clone()), usize::MAX));
                }
            }
        }
    }

    // the variant's layout
    let hints: HashMap<u64, &Hint> = variant.hints.iter().map(|h| (h.clip, h)).collect();
    let template_clip: HashMap<ClipId, u64> = ids.clips.iter().map(|(o, n)| (*n, *o)).collect();
    let frame = (f64::from(w1), f64::from(h1));
    let par = seq.settings.par;
    for t in seq.video_tracks.iter_mut() {
        for it in t.items.iter_mut() {
            let tc = origin.get(&it.id).map(|o| o.0).or_else(|| template_clip.get(&it.id).copied());
            let hint = tc.and_then(|c| hints.get(&c).copied());
            let kind = p.item(it.item).map(|i| &i.kind);
            match kind {
                Some(ItemKind::Graphic { .. }) => layout_graphic(it, design, frame, hint),
                Some(ItemKind::AdjustmentLayer { .. }) | None => {}
                Some(_) => {
                    let slot = tc.and_then(|c| m.placement_of(c)).map(|(s, _)| s);
                    let framing = slot.map_or(Framing::Fill, |s| s.fit.frame);
                    let focus = match (origin.get(&it.id), slot) {
                        (Some((_, Some(sid), mi)), Some(_)) => match opts.fills.get(sid) {
                            Some(SlotFill::Media(list)) => list.get(*mi).map_or([0.5, 0.5], |f| f.focus_for(&variant.id)),
                            _ => [0.5, 0.5],
                        },
                        _ => [0.5, 0.5],
                    };
                    if let Some(src) = p.conformed_source_size(it.item, par) {
                        frame_clip(it, src, design, frame, framing, focus, hint);
                    }
                }
            }
        }
    }

    // the look on the slots' pictures
    if let Some(look) = &opts.look {
        let picture_slots: Vec<&Slot> = m.slots.iter().filter(|s| matches!(s.kind, SlotKind::Video | SlotKind::Photo)).collect();
        for t in seq.video_tracks.iter_mut() {
            for it in t.items.iter_mut() {
                let tc = origin.get(&it.id).map(|o| o.0);
                if tc.is_some_and(|c| picture_slots.iter().any(|s| s.placements.iter().any(|pl| pl.clip == c))) {
                    let pos = it.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
                    it.effects.insert(pos, look.clone());
                }
            }
        }
    }
    // the caption style on every caption track (a track for captions to come when there is none)
    if let Some(style) = &m.caption_style {
        if seq.caption_tracks.is_empty() {
            seq.caption_tracks.push(CaptionTrack {
                id: TrackId(p.alloc_id()),
                name: "Captions".into(),
                format: CaptionFormat::Subtitle,
                language: String::new(),
                style: style.clone(),
                enabled: true,
                locked: false,
                sync_lock: true,
                captions: Vec::new(),
                vtt_blocks: Vec::new(),
            });
        }
        for t in seq.caption_tracks.iter_mut() {
            t.style = style.clone();
        }
    }
    seq.check().map_err(|e| err(format!("the applied template is not a valid sequence: {e}")))?;
    let duration = seq.duration();
    let name = opts.name.clone().filter(|n| !n.trim().is_empty()).unwrap_or_else(|| format!("{} ({})", m.name, variant.id));
    let export_preset = variant.export_preset.clone().or_else(|| m.export_preset.clone());
    let sid = p.add_item(&name, Label::Forest, ItemKind::Sequence(Arc::new(seq)), None);
    if let Some(it) = p.item_mut(sid) {
        it.metadata.insert("Template".into(), m.id.clone());
        it.metadata.insert("Template Variant".into(), variant.id.clone());
        if let Some(e) = &export_preset {
            it.metadata.insert("Template Export Preset".into(), e.clone());
        }
    }
    Ok(Applied { sequence: sid, variant: variant.id, duration, placed, cuts, retimed, remix, export_preset, warnings, items: added, bin })
}

/// Loop a clip to `target` from its in point (a music clip Remix could not stretch).
pub fn loop_clip(p: &mut Project, seq_id: ItemId, clip: ClipId, target: Tick) -> Result<Vec<ClipId>, TemplateError> {
    let seq = p.sequence(seq_id).ok_or_else(|| err("no such sequence"))?;
    let at = locate(seq, clip).ok_or_else(|| err("no such clip"))?;
    let tmpl = clip_ref(seq, clip).cloned().ok_or_else(|| err("no such clip"))?;
    let rate = seq.settings.frame_rate;
    let len = media_length(p, tmpl.item).ok_or_else(|| err("the clip's media has no length to loop"))?;
    let avail = len - tmpl.source_in;
    let fitted = fit::looped(target, tmpl.source_in, avail, rate.frame_duration()).map_err(err)?;
    let mut out = Vec::new();
    for (k, pc) in fitted.pieces.iter().enumerate() {
        let mut c = tmpl.clone();
        c.id = if k == 0 { tmpl.id } else { ClipId(p.alloc_id()) };
        c.start = tmpl.start + pc.offset;
        c.duration = pc.duration;
        c.source_in = pc.source_in;
        rebase_keyframes(&mut c, &tmpl, pc.offset);
        out.push(c);
    }
    let seq = p.sequence_mut(seq_id).ok_or_else(|| err("no such sequence"))?;
    let track = seq.tracks(at.0).get(at.1).ok_or_else(|| err("no such track"))?;
    if let Some(next) = track.items.iter().filter(|c| c.id != clip && c.start >= tmpl.start).map(|c| c.start).min()
        && tmpl.start + target > next
    {
        return Err(err("not enough room to loop the music"));
    }
    let ids = replace_clip(seq, at, out).ok_or_else(|| err("no such clip"))?;
    seq.check().map_err(err)?;
    Ok(ids)
}

/// Whether a slot loops by default when Remix cannot fit it.
pub fn remix_falls_back(slot: &Slot) -> bool {
    slot.fit.short == ShortFit::Remix
}

//! Save a sequence as a template (the minimal flow; the full authoring and sharing flow is A3).
//!
//! Clips flagged as placeholders — a clip marker named `slot:<kind>[:<slot id>[:<placement>]]`,
//! which `template.markSlot` adds and every applied template leaves on its slots — become the
//! template's slots, their media replaced by generated placeholders. The personal media of
//! every other clip is left out (with a warning), so a saved template never carries someone's
//! footage; generated media, graphics and adjustment layers stay. Captions are left out too
//! (their text is the user's), but the caption style is kept.

use std::collections::{BTreeMap, HashMap};

use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::graphic::TEXT_LAYER;
use filmcraft_project::gtemplate::{ensure_uids, layer_uid};
use filmcraft_project::{ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, Project, TrackKind};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::apply::parse_slot_marker;
use crate::manifest::{BeatGrid, Constraints, Manifest, Placement, Slot, SlotKind, Variant, default_fit};
use crate::{FORMAT, FORMAT_VERSION, Package, SEQUENCE_FILE, TemplateError};

/// The new template's description.
#[derive(Clone, Debug, Default)]
pub struct SaveOptions {
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub category: String,
}

/// A saved template and what was left out.
#[derive(Clone, Debug)]
pub struct Saved {
    pub package: Package,
    pub warnings: Vec<String>,
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// `16:9`-style id of a frame size.
pub fn aspect_id(w: u32, h: u32) -> String {
    let g = gcd(w, h);
    format!("{}:{}", w / g, h / g)
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

/// Make a template from sequence `seq` of `p`. `beat_grid` is the music's beats on the
/// sequence's timeline (the engine measures them), if any.
pub fn save(p: &Project, seq: ItemId, opts: &SaveOptions, beat_grid: Option<BeatGrid>) -> Result<Saved, TemplateError> {
    let err = |m: String| TemplateError::Apply(m);
    let src = p.sequence(seq).ok_or_else(|| err("no such sequence".into()))?;
    let mut q = src.clone();
    let rate = q.settings.frame_rate;
    let size = (q.settings.width, q.settings.height);
    let mut warnings = Vec::new();
    let mut fragment = Project::new(&opts.name);
    fragment.next_id = p.next_id;
    // (slot id) → (kind, placements by (placement index or order, clip, layer))
    let mut slots: BTreeMap<String, (SlotKind, Vec<(usize, u64, Option<u64>)>)> = BTreeMap::new();
    let mut copied: HashMap<ItemId, ItemId> = HashMap::new();
    let mut placeholders: HashMap<(String, usize), ItemId> = HashMap::new();
    let mut dropped: Vec<ClipId> = Vec::new();
    let mut order = 0usize;
    for kind in [TrackKind::Video, TrackKind::Audio] {
        for t in q.tracks_mut(kind).iter_mut() {
            let mut keep = Vec::new();
            // pieces of one placement (a loop, a hold) become one clip again
            let mut last_slot: Option<(String, usize, usize)> = None;
            for mut c in std::mem::take(&mut t.items) {
                let flags: Vec<crate::apply::SlotFlag> = c.markers.iter().filter_map(|m| parse_slot_marker(&m.name)).take(64).collect();
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
                            let prev: &mut filmcraft_project::TrackItem = prev;
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
                                    let entry = slots.entry(f.slot.clone()).or_insert((SlotKind::Text, Vec::new()));
                                    if !entry.1.iter().any(|x| x.1 == c.id.0 && x.2 == layer) {
                                        entry.1.push((fpl, c.id.0, layer));
                                    }
                                }
                            }
                            let id = *copied.entry(c.item).or_insert_with(|| fragment.add_item(&it.name, it.label, it.kind.clone(), None));
                            c.item = id;
                        } else {
                            let id = *placeholders.entry((sid.clone(), pl)).or_insert_with(|| placeholder(&mut fragment, sk, pl % 10_000, size, rate));
                            c.item = id;
                            c.name = fragment.item(id).map(|i| i.name.clone()).unwrap_or_default();
                            c.source_in = Tick::ZERO;
                            c.speed = 1.0;
                            c.reverse = false;
                            c.frame_hold = None;
                            let on_its_track = if sk == SlotKind::Music { kind == TrackKind::Audio } else { kind == TrackKind::Video };
                            if on_its_track {
                                slots.entry(sid.clone()).or_insert((sk, Vec::new())).1.push((pl, c.id.0, None));
                            }
                        }
                        last_slot = Some((sid, pl, keep.len()));
                        keep.push(c);
                    }
                    None => {
                        last_slot = None;
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
    let had_captions = q.caption_tracks.iter().any(|t| !t.captions.is_empty());
    let caption_style = q.caption_tracks.first().map(|t| t.style.clone());
    q.caption_tracks.clear();
    if had_captions {
        warnings.push("the captions were left out (their text is yours); the caption style is kept".into());
    }
    let duration = q.duration();
    if duration.0 <= 0 {
        return Err(err("the sequence is empty".into()));
    }
    let seq_id = fragment.add_item(&opts.name, Label::Forest, ItemKind::Sequence(std::sync::Arc::new(q)), None);
    let mut manifest_slots = Vec::new();
    for (i, (sid, (kind, mut pls))) in slots.into_iter().enumerate() {
        pls.sort_by_key(|x| x.0);
        let mut s = Slot {
            id: sid.clone(),
            kind,
            name: sid.replace(['-', '_'], " "),
            prompt: String::new(),
            order: u32::try_from(i).unwrap_or(u32::MAX),
            count: u32::try_from(pls.len()).unwrap_or(1).max(1),
            min_count: if matches!(kind, SlotKind::Text | SlotKind::Music) { 0 } else { 1 },
            placements: pls
                .iter()
                .enumerate()
                .map(|(k, (_, clip, layer))| Placement { clip: *clip, media: u32::try_from(k).unwrap_or(0), layer: *layer })
                .collect(),
            constraints: Constraints::default(),
            fit: default_fit(kind),
            beat_sync: beat_grid.is_some() && kind.is_visual(),
            default: None,
        };
        if kind == SlotKind::Music {
            s.count = 1;
            s.placements.truncate(1);
            if let Some(pl) = s.placements.first_mut() {
                pl.media = 0;
            }
        }
        manifest_slots.push(s);
    }
    if manifest_slots.is_empty() {
        warnings.push("no clip is flagged as a slot (template.markSlot): the template has no slots".into());
    }
    let (w, h) = size;
    let mut variants = vec![Variant {
        id: aspect_id(w, h),
        name: if w >= h { "Landscape".into() } else { "Vertical".into() },
        width: w,
        height: h,
        export_preset: None,
        hints: Vec::new(),
    }];
    if w != h {
        variants.push(Variant {
            id: aspect_id(h, w),
            name: if w >= h { "Vertical".into() } else { "Landscape".into() },
            width: h,
            height: w,
            export_preset: None,
            hints: Vec::new(),
        });
    }
    let id = if crate::manifest::is_template_id(&opts.id) { opts.id.clone() } else { format!("user:{}", crate::slug(&opts.name)) };
    let m = Manifest {
        format: FORMAT.into(),
        format_version: FORMAT_VERSION,
        id,
        name: opts.name.clone(),
        description: opts.description.clone(),
        version: "1.0.0".into(),
        author: opts.author.clone(),
        license: opts.license.clone(),
        category: opts.category.clone(),
        tags: Vec::new(),
        sequence_file: SEQUENCE_FILE.into(),
        sequence: seq_id.0,
        duration,
        default_variant: variants.first().map(|v| v.id.clone()).unwrap_or_default(),
        variants,
        slots: manifest_slots,
        look: None,
        caption_style,
        export_preset: None,
        beat_grid: beat_grid.filter(|g| g.beats.len() >= 2),
        assets: Vec::new(),
    };
    Ok(Saved { package: Package::new(m, fragment)?, warnings })
}

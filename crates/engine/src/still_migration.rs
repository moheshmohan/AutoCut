//! Load-time migration for projects saved before stills were turned upright by their EXIF
//! Orientation.
//!
//! Before that, a still's size was stored as the file stores it, unturned, and a clip placed from
//! the still baked its Motion anchor point from that size. The picture has decoded upright since,
//! so such a project reopens with a portrait photo scaled up by its aspect ratio and off-centre.
//! What a fresh import gets for free a saved project is given on load: the stored size is swapped
//! to the one the file decodes to now, and the anchor points of clips placed from the still are
//! turned to match.
//!
//! The legacy state is recognised by its shape, not by the file's schema version (a project
//! re-saved by a newer build keeps its stale sizes at the current version): a still is legacy when
//! its decoder turns it onto its side and its stored size is the transpose of the size it decodes
//! to. A fresh import stores the decoded size, so it never matches; a square still is its own
//! transpose and is skipped; a still whose file cannot be read keeps its stored size until it is
//! relinked and the project is opened again.

use std::collections::HashSet;

use filmcraft_media::MediaKind;
use filmcraft_media::still::{Orientation, transposes};
use filmcraft_project::{ItemId, ItemKind, Param, ParamValue, Project, TrackKind};

use crate::{MediaPool, Services};

/// Swap the stored size of every legacy still in `p` and turn the Motion anchor of each clip
/// placed from one. Runs while a project is installed, before it is shown, so no undo step is
/// made. The file behind a still is only probed for the size and orientation its decoder reports
/// ([`MediaPool::still_geometry`]): no picture is decoded and the pool keeps no source.
pub fn migrate_legacy_still_sizes(p: &mut Project, pool: &MediaPool, services: &dyn Services) {
    let legacy: Vec<(ItemId, (u32, u32), Orientation)> = p
        .items
        .iter()
        .filter_map(|(id, item)| {
            let m = item.as_media()?;
            if m.info.kind != MediaKind::Still {
                return None;
            }
            let v = m.info.video.as_ref()?;
            if v.width == v.height {
                return None;
            }
            let (w, h, orientation) = pool.still_geometry(p, *id, services)?;
            (transposes(orientation) && (w, h) == (v.height, v.width)).then_some((*id, (v.width, v.height), orientation))
        })
        .collect();
    for (id, stored, orientation) in legacy {
        if let Some(v) = p.item_mut(id).and_then(|i| i.as_media_mut()).and_then(|m| m.info.video.as_mut()) {
            std::mem::swap(&mut v.width, &mut v.height);
        }
        turn_anchors(p, id, stored, orientation);
    }
}

/// One anchor point from the file's stored pixel coordinates to the upright picture's, for the
/// EXIF Orientation the decoder applied (the mapping it applies to the pixels, in continuous
/// coordinates). A centre point is the same under all four, so the orientation only matters for
/// an anchor off the centre.
fn turn_point(x: f64, y: f64, stored: (u32, u32), orientation: Orientation) -> (f64, f64) {
    let (w, h) = (f64::from(stored.0), f64::from(stored.1));
    match orientation {
        Orientation::Rotate90 => (h - y, x),
        Orientation::Rotate270FlipH => (h - y, w - x),
        Orientation::Rotate270 => (y, w - x),
        // Rotate90FlipH, and anything else: the plain swap
        _ => (y, x),
    }
}

/// The centre of the 1920 × 1080 frame a build from before the subclip size fix resolved a
/// subclip placement's anchor against (see [`turn_anchors`]).
const LEGACY_SUBCLIP_DEFAULT_ANCHOR: (f64, f64) = (960.0, 540.0);

/// The anchor point is in source pixels, so turning the source turns it, the static value and
/// every keyframe. A coordinate left on "auto" (NaN) stays auto and resolves against the new
/// size; Position is in sequence pixels and does not move. Clips placed from a subclip of the
/// still keep the subclip's id: anything that resolves to the still is one of its clips.
///
/// One saved value is not a point of the still at all. A build from before the subclip size fix
/// had no size for a subclip, so placing one saved the centre of a 1920 × 1080 fallback frame as
/// its anchor. On a clip placed from a subclip, an anchor still holding exactly that value with
/// no keyframes is that saved default, and it moves to the upright centre instead of turning.
fn turn_anchors(p: &mut Project, item: ItemId, stored: (u32, u32), orientation: Orientation) {
    let targets: HashSet<ItemId> = p.items.keys().filter(|id| p.resolve_media(**id).is_some_and(|(m, _, _)| m == item)).copied().collect();
    let centre = (f64::from(stored.1) / 2.0, f64::from(stored.0) / 2.0);
    for it in p.items.values_mut() {
        let ItemKind::Sequence(q) = &mut it.kind else { continue };
        if !q.tracks(TrackKind::Video).iter().any(|t| t.items.iter().any(|ti| targets.contains(&ti.item))) {
            continue;
        }
        for t in std::sync::Arc::make_mut(q).tracks_mut(TrackKind::Video) {
            for ti in t.items.iter_mut().filter(|ti| targets.contains(&ti.item)) {
                let from_subclip = ti.item != item;
                for e in ti.effects.iter_mut().filter(|e| e.effect == "motion") {
                    if let Some(param) = e.param_mut("anchor") {
                        turn_param(param, stored, orientation, from_subclip.then_some(centre));
                    }
                }
            }
        }
    }
}

fn turn_param(param: &mut Param, stored: (u32, u32), orientation: Orientation, subclip_centre: Option<(f64, f64)>) {
    if let Some(centre) = subclip_centre
        && param.keyframes.is_empty()
        && let ParamValue::Vec2(v) = &mut param.value
        && (v.x, v.y) == LEGACY_SUBCLIP_DEFAULT_ANCHOR
    {
        (v.x, v.y) = centre;
        return;
    }
    let turn = |value: &mut ParamValue| {
        if let ParamValue::Vec2(v) = value
            && !v.x.is_nan()
            && !v.y.is_nan()
        {
            (v.x, v.y) = turn_point(v.x, v.y, stored, orientation);
        }
    };
    turn(&mut param.value);
    for k in &mut param.keyframes {
        turn(&mut k.value);
    }
}

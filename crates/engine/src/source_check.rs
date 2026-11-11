//! Bounded traversal of the media actually referenced by a delivery sequence.
use crate::Session;
use filmcraft_project::{ClipId, ItemId, ItemKind, MediaRef, TrackKind};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub(crate) fn inspect(s: &Session, root: ItemId) -> (Vec<Value>, Vec<ItemId>) {
    let mut errors = Vec::new();
    let mut media = Vec::new();
    let mut stack = vec![(root, None::<ClipId>, 0usize, false)];
    let (mut active, mut done) = (BTreeSet::new(), BTreeSet::new());
    let mut visited = 0usize;
    while let Some((id, via, depth, exit)) = stack.pop() {
        if exit {
            active.remove(&id);
            done.insert(id);
            continue;
        }
        if done.contains(&id) {
            continue;
        }
        visited = visited.saturating_add(1);
        let mut issue = |reason: String| errors.push(json!({"item":id.0,"clip":via.map(|v|v.0),"reason":reason}));
        if depth > 64 || visited > 10000 || stack.len() > 10000 {
            issue("source graph exceeds limits; split the edit".into());
            break;
        }
        if !active.insert(id) {
            issue("cyclic source reference".into());
            continue;
        }
        stack.push((id, via, depth, true));
        let Some(item) = s.project.item(id) else {
            issue("referenced item is missing".into());
            continue;
        };
        match &item.kind {
            ItemKind::Subclip { parent, .. } => stack.push((*parent, via, depth + 1, false)),
            ItemKind::Sequence(q) => {
                if let Err(error) = q.check_bounds() {
                    issue(error);
                    continue;
                }
                for track in q.video_tracks.iter().chain(&q.audio_tracks).filter(|t| t.enabled && (t.kind == TrackKind::Video || !t.muted)) {
                    for clip in track.items.iter().filter(|c| c.enabled) {
                        if stack.len() >= 10000 {
                            issue("too many source references; split the edit".into());
                            break;
                        }
                        stack.push((clip.item, via.or(Some(clip.id)), depth + 1, false));
                    }
                }
            }
            ItemKind::Media(m) => {
                media.push(id);
                if m.offline {
                    issue("source is marked offline; relink it".into());
                }
                if let MediaRef::File { path } = &m.media
                    && let Err(error) = s.services.file_size(path)
                {
                    issue(format!("{path}: {error}"));
                }
            }
            _ => {}
        }
    }
    (errors, media)
}

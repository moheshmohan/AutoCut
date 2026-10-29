//! Decoding ahead: the media pictures a frame of a sequence will show, asked for before it is
//! rendered.
//!
//! An export renders a batch of frames at once, but each source decodes its pictures one after
//! the other (the GOP cache holds its lock while decoding), so the frames of a batch wait for the
//! decoder in turn and the batch ends with the cores idle while its last frames composite. An
//! exporter that asks for the next batch's pictures on a thread of its own while the current batch
//! composites keeps every decoder busy instead: the pictures are in the decoders' caches by the
//! time those frames are rendered. Software decoding (iPhone HDR video served as VP9 profile 3)
//! and the pre-roll of every new shot (decoding from the keyframe to the clip's In) are what this
//! takes off the critical path.
//!
//! Only the requests change hands: a cached picture is the one the decoder would have produced for
//! the render, so the frames are the same. What is asked mirrors [`crate::render_sequence`]: the
//! clip on every enabled video track, both clips of a transition, and the next source frame of a
//! clip that blends frames; nested sequences, adjustment layers and graphics are left to the
//! render.

use filmcraft_media::FrameRequest;
use filmcraft_project::{ItemKind, Project, Sequence, TrackItem};
use filmcraft_time::Tick;

use crate::{RenderOptions, SourceProvider, effects_scale, interpolation_blend, motion_matrix, video_source_time};

/// Ask the media sources of sequence `seq_id` for the pictures its frame at `t` shows (at the
/// scale the render asks them for), so they are decoded and cached before the frame is rendered.
/// Returns how many pictures were asked for. Errors (offline media, a picture that does not
/// decode) are left for the render to meet.
pub fn warm_frame(project: &Project, seq_id: filmcraft_project::ItemId, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> usize {
    let Some(seq) = project.sequence(seq_id) else { return 0 };
    let mut asked = 0;
    for track in seq.video_tracks.iter().filter(|t| t.enabled) {
        if filmcraft_media::cancel::cancelled() {
            break;
        }
        match track.transitions.iter().find(|tr| tr.range().contains(t)) {
            Some(tr) => {
                for id in [tr.from, tr.to].into_iter().flatten() {
                    if let Some(item) = track.item(id).filter(|i| i.enabled) {
                        asked += warm_item(project, seq, item, t, opts, sources);
                    }
                }
            }
            None => {
                if let Some(item) = track.item_at(t).filter(|i| i.enabled) {
                    asked += warm_item(project, seq, item, t, opts, sources);
                }
            }
        }
    }
    asked
}

/// [`warm_frame`] for one clip: its media picture at `t`, and the next one when it blends frames.
fn warm_item(project: &Project, seq: &Sequence, item: &TrackItem, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> usize {
    let Some(pi) = project.item(item.item) else { return 0 };
    if !matches!(pi.kind, ItemKind::Media(_) | ItemKind::Subclip { .. }) {
        return 0;
    }
    let Some(src_size) = crate::clip_size(project, item) else { return 0 };
    let Some(src) = sources.source(item.item) else { return 0 };
    // the scale `item_layer_ex` asks for (sources that decode reduced pictures cache them by it)
    let mt = item.effect_time_at(t);
    let motion = motion_matrix(seq, item, src_size, crate::clip_par(project, item), mt);
    let lin = (motion.a * motion.a + motion.b * motion.b).sqrt().max((motion.c * motion.c + motion.d * motion.d).sqrt());
    let want = effects_scale(item, mt, crate::frame_scale(lin * opts.scale as f64, Some(&src)), opts);
    // a 360° clip asks for its source as sharp as its virtual camera's view needs
    let want = crate::reframing(project, item, t, &src, want).map_or(want, |r| r.scale);
    let rate = crate::detail_rate(&src);
    let mut asked = 1;
    let _ = src.video_frame(FrameRequest { time: video_source_time(item, t, rate), scale: want });
    if let Some((next, _)) = interpolation_blend(item, t, rate) {
        let _ = src.video_frame(FrameRequest { time: next, scale: want });
        asked += 1;
    }
    asked
}

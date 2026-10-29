//! Object Mask (the idea of Premiere 26's Object Mask): a thing in a clip masked from a click on
//! it, and followed through the clip — the mask takes the thing's shape in every frame it is
//! tracked to.
//!
//! | command | does |
//! |---|---|
//! | `clip.objectMask` | job: the thing at `point` (`[x, y]`, 0..1 of the clip's picture; more points on it in `inside`, points off it in `outside`) on the clip's frame `at` (seconds into the clip; the playhead) cut out by MobileSAM and outlined in `points` points (128) as a mask on `effect` (Opacity) — a new one, `name` (`Object (n)`), `mode` (`add`), `inverted`, or the mask `mask` cut again — then followed `track` (`both`: on to the clip's end and back to its start; `forward`, `backward`, `none`) every `every` seconds (0.1), the mask's tracking method `object` |
//! | `masks.track` | with the mask's method `object` (or `method: "object"`): the thing a mask outlines followed from the playhead, `forward` or `backward` (`frames` at most) |
//!
//! MobileSAM is the cut-out model's (`cutout.status`, `cutout.downloadModel`: downloaded on the
//! user's consent).
//!
//! **How a thing is followed.** On the frame clicked, the points go to MobileSAM: for one point,
//! the surest of its three answers for an ambiguous prompt (a part, the thing, the thing and what
//! it stands on); for more, its one answer. Each frame after, a box tracker
//! ([`filmcraft_render::object_track`], the subject method: median flow with a template that
//! finds the thing again after it was hidden) seeded with the last cut's box says where the thing
//! went; the last outline, moved and sized as its box was, is the shape expected there; MobileSAM
//! is given that box (grown by an eighth) and the expected shape's deepest points, and of its four
//! answers the one overlapping the expected shape most is the thing's, if it overlaps it by at
//! least [`SAME`] — else the frame keeps no outline (masks interpolate between their outlines),
//! and the thing lost for 1.5 s ends the run. The cut's largest piece, and pieces of at least a
//! fifth of it (the thing seen either side of a post), are outlined as one outline. The run back
//! from the clicked frame starts from that frame again.

use std::sync::{Arc, Mutex};

use filmcraft_project::{MaskMode, TrackItem, TrackMethod, TrackingMethod};
use filmcraft_render::object_track::{ANALYSIS_SIDE, ObjectBox, ObjectTracker};
use filmcraft_render::subject::SubjectBox;
use filmcraft_time::Tick;
pub use filmcraft_vision::matte::Matte;
use serde_json::Value;

use crate::commands::{CommandSpec, bad, bool_p, f64_p, has_seq, str_p, u64_p};
use crate::cutout::{FindOutline, Look, Outlining, Start};
use crate::masks::MaskSel;
use crate::{Result, Session};

/// The name of new object masks (`Object (1)`, `Object (2)`, …).
pub const OBJECT: &str = "Object";
pub const CMD: &str = "clip.objectMask";
/// Seconds between outlines by default.
const EVERY: f64 = 0.1;
/// Points of an outline by default.
const POINTS: usize = 128;
/// Long side of the grid shapes are compared on.
const GRID: usize = 256;
/// A cut is the thing's when it overlaps the shape expected at least this much (intersection
/// over union).
pub const SAME: f32 = 0.45;
/// …and on the first frame of a mask followed from its own shape (drawn round the thing).
const SAME_DRAWN: f32 = 0.25;
/// The box MobileSAM is given: the expected shape's grown by this share of its size each way…
const GROW: f32 = 0.25;
/// …and when the cut reaches that box's edge (the thing went further than expected), this.
const GROW_MORE: f32 = 0.75;
/// Two guesses at where the thing went count as one when their middles are this share of its
/// size apart.
const SAME_GUESS: f64 = 0.15;
/// The thing's speed is measured over this long of where it was seen whole…
const SPEED_SECONDS: f64 = 0.5;
/// …from at most this many sightings.
const TRAIL: usize = 16;
/// A guess by the thing's speed goes at most this far ahead (seconds), as a lost thing ends a run.
const LOST_SECONDS: f64 = 1.5;
/// A cut covering at least this share of what the thing covers whole is it seen whole…
const WHOLE: f32 = 0.85;
/// …what it covers whole being the most it covered lately, let go of by this much a second and
/// taking on at most this much a second.
const LET_GO_PER_SECOND: f32 = 0.6;
const GROW_PER_SECOND: f32 = 1.5;
/// A cut is the thing's only when its colours are at least this alike to the thing's…
const ALIKE: f32 = 0.5;
/// …and it is at most this many times the size expected.
const MAX_GROWTH: f32 = 1.6;
/// Pieces outlined besides the largest: at least this share of the cut.
const PIECE: f32 = 0.2;
const MAX_PIECES: usize = 4;
/// Most points a prompt takes.
const MAX_CLICKS: usize = 16;

/// What MobileSAM is asked, in a picture's pixels: points on the thing and off it, and a box
/// round it (`[x0, y0, x1, y1]`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt {
    pub inside: Vec<(f32, f32)>,
    pub outside: Vec<(f32, f32)>,
    pub bbox: Option<[f32; 4]>,
}

/// Cuts things out of pictures: MobileSAM, or a stand-in.
pub trait ObjectCutter: Send + Sync {
    /// An RGBA picture (`w` × `h`) made ready to be asked about (MobileSAM embeds it once).
    fn picture(&self, rgba: &[u8], w: usize, h: usize) -> std::result::Result<Box<dyn Cuts + '_>, String>;
}

/// A picture ready to be asked about.
pub trait Cuts {
    /// The answers to `prompt`, each at the picture's size: the answer for an unambiguous prompt
    /// first, then (when it gives them) the three for an ambiguous one.
    fn cut(&self, prompt: &Prompt) -> std::result::Result<Vec<Matte>, String>;
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Object Mask",
        menu: &["Clip", "Cut-outs"],
        shortcut: None,
        params: r#"{"clip":id?,"effect":index|id?="opacity","point":[x,y]?,"inside":[[x,y]…]?,"outside":[[x,y]…]?,"at":seconds?,"mask":n?,"name":str?,"mode":"add"|"subtract"|"none"|"intersect"|"lighten"|"darken"|"difference"="add","inverted":bool=false,"track":"both"|"forward"|"backward"|"none"="both","every":seconds=0.1,"points":16..256=128,"wait":bool=false}"#,
        enabled: has_seq,
        run: object_mask,
        journal: true,
    }]
}

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

/// MobileSAM, loaded once (a click in the monitor should not wait for it again).
static LOADED: Mutex<Option<Arc<dyn ObjectCutter>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The cutter a job uses: the session's stand-in, else MobileSAM.
fn cutter(s: &Session, cmd: &str) -> Result<Arc<dyn ObjectCutter>> {
    if let Some(c) = &s.cutouts.object {
        return Ok(c.clone());
    }
    if let Some(c) = lock(&LOADED).clone() {
        return Ok(c);
    }
    let c = load(cmd)?;
    *lock(&LOADED) = Some(c.clone());
    Ok(c)
}

#[cfg(feature = "privacy-models")]
struct Sam {
    seg: filmcraft_vision::segment::Segmenter,
}

#[cfg(feature = "privacy-models")]
impl ObjectCutter for Sam {
    fn picture(&self, rgba: &[u8], w: usize, h: usize) -> std::result::Result<Box<dyn Cuts + '_>, String> {
        let e = self.seg.embed(rgba, w, h).map_err(|e| e.to_string())?;
        Ok(Box::new(Embedded { seg: &self.seg, e, w, h }))
    }
}

/// A picture MobileSAM embedded.
#[cfg(feature = "privacy-models")]
struct Embedded<'a> {
    seg: &'a filmcraft_vision::segment::Segmenter,
    e: filmcraft_vision::segment::Embedded,
    w: usize,
    h: usize,
}

#[cfg(feature = "privacy-models")]
impl Cuts for Embedded<'_> {
    fn cut(&self, prompt: &Prompt) -> std::result::Result<Vec<Matte>, String> {
        let answers = self.seg.answers(&self.e, &prompt.inside, &prompt.outside, prompt.bbox).map_err(|e| e.to_string())?;
        Ok(answers.into_iter().map(|m| at_size(m, self.w, self.h)).collect())
    }
}

#[cfg(feature = "privacy-models")]
fn load(cmd: &str) -> Result<Arc<dyn ObjectCutter>> {
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for the models"))?;
    match filmcraft_vision::segment::Segmenter::load(&dir, true) {
        Ok(seg) => Ok(Arc::new(Sam { seg })),
        Err(filmcraft_vision::VisionError::NotInstalled) => Err(bad(cmd, "the segmenter (MobileSAM) is not downloaded yet (cutout.downloadModel)")),
        Err(e) => Err(bad(cmd, e.to_string())),
    }
}

#[cfg(not(feature = "privacy-models"))]
fn load(cmd: &str) -> Result<Arc<dyn ObjectCutter>> {
    Err(bad(cmd, "masking objects is not available in this build (built without the `privacy-models` feature)"))
}

/// A cut-out at `w` × `h` (nearest pixel), as it is when it is that size already.
pub fn at_size(m: Matte, w: usize, h: usize) -> Matte {
    if (m.w, m.h) == (w, h) {
        return m;
    }
    let (mw, mh) = (m.w.max(1), m.h.max(1));
    let alpha = (0..w.saturating_mul(h)).map(|i| m.at(((i % w.max(1)) * mw / w.max(1)).min(mw - 1), ((i / w.max(1)) * mh / h.max(1)).min(mh - 1))).collect();
    Matte { w, h, alpha, score: m.score }
}

// ---------------------------------------------------------------------------------------------
// shapes on a grid
// ---------------------------------------------------------------------------------------------

/// A coarse grid over a picture (its long side [`GRID`] cells) to compare shapes on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Grid {
    gw: usize,
    gh: usize,
    /// Picture pixels a cell, and the picture's size.
    k: f32,
    w: usize,
    h: usize,
}

impl Grid {
    fn new(w: usize, h: usize) -> Self {
        let k = (w.max(h) as f32 / GRID as f32).max(1.0);
        Self { gw: ((w as f32 / k).ceil() as usize).max(1), gh: ((h as f32 / k).ceil() as usize).max(1), k, w, h }
    }

    /// The cells whose middles a closed polygon (picture pixels) covers (even-odd).
    fn fill(&self, poly: &[(f32, f32)]) -> Vec<bool> {
        let mut out = vec![false; self.gw * self.gh];
        let n = poly.len();
        if n < 3 || poly.iter().any(|p| !(p.0.is_finite() && p.1.is_finite())) {
            return out;
        }
        let mut xs: Vec<f32> = Vec::new();
        for gy in 0..self.gh {
            let y = (gy as f32 + 0.5) * self.k;
            xs.clear();
            for i in 0..n {
                let (a, b) = (poly[i], poly[(i + 1) % n]);
                if (a.1 <= y) != (b.1 <= y) {
                    xs.push(a.0 + (b.0 - a.0) * (y - a.1) / (b.1 - a.1));
                }
            }
            xs.sort_by(f32::total_cmp);
            for &[a, b] in xs.as_chunks::<2>().0 {
                let x0 = ((a / self.k - 0.5).ceil().max(0.0)) as usize;
                let x1 = ((b / self.k - 0.5).floor()).min(self.gw as f32 - 1.0);
                if x1 < 0.0 {
                    continue;
                }
                for gx in x0..=(x1 as usize) {
                    if let Some(c) = out.get_mut(gy * self.gw + gx) {
                        *c = true;
                    }
                }
            }
        }
        out
    }

    /// The cells a cut-out (at the picture's size) covers at their middles.
    fn of(&self, m: &Matte) -> Vec<bool> {
        let mut out = vec![false; self.gw * self.gh];
        for gy in 0..self.gh {
            for gx in 0..self.gw {
                let (x, y) = (((gx as f32 + 0.5) * self.k) as usize, ((gy as f32 + 0.5) * self.k) as usize);
                if let Some(c) = out.get_mut(gy * self.gw + gx) {
                    *c = m.at(x, y) >= 128;
                }
            }
        }
        out
    }

    /// Whether a shape reaches the edge of a box (`[x0, y0, x1, y1]`, picture pixels) on a side
    /// where the `w` × `h` picture goes on past it.
    fn reaches(&self, cells: &[bool], b: [f32; 4], w: usize, h: usize) -> bool {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for (i, _) in cells.iter().enumerate().filter(|(_, c)| **c) {
            let (x, y) = (i % self.gw.max(1), i / self.gw.max(1));
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        if x0 > x1 {
            return false;
        }
        // a cell's middle within a cell and a half of the box's side
        let near = |cell: usize, side: f32| ((cell as f32 + 0.5) * self.k - side).abs() <= 1.5 * self.k;
        (b[0] > 1.0 && near(x0, b[0]))
            || (b[1] > 1.0 && near(y0, b[1]))
            || (b[2] < w as f32 - 1.0 && near(x1, b[2]))
            || (b[3] < h as f32 - 1.0 && near(y1, b[3]))
    }

    /// Up to two points deep inside a shape (picture pixels): the deepest, and for a long thing
    /// one far from it nearly as deep.
    fn deepest(&self, cells: &[bool]) -> Vec<(f32, f32)> {
        let (gw, gh) = (self.gw, self.gh);
        if cells.len() != gw * gh {
            return Vec::new();
        }
        // chamfer distance (3-4) to the nearest cell outside, the picture's edge outside too
        let mut d: Vec<u32> = cells.iter().map(|c| if *c { u32::MAX / 2 } else { 0 }).collect();
        let at = |d: &[u32], x: i64, y: i64| if x < 0 || y < 0 || x >= gw as i64 || y >= gh as i64 { 0 } else { d[y as usize * gw + x as usize] };
        for y in 0..gh as i64 {
            for x in 0..gw as i64 {
                let i = y as usize * gw + x as usize;
                if d[i] == 0 {
                    continue;
                }
                let v = [at(&d, x - 1, y) + 3, at(&d, x, y - 1) + 3, at(&d, x - 1, y - 1) + 4, at(&d, x + 1, y - 1) + 4];
                d[i] = d[i].min(v.into_iter().min().unwrap_or(0));
            }
        }
        for y in (0..gh as i64).rev() {
            for x in (0..gw as i64).rev() {
                let i = y as usize * gw + x as usize;
                if d[i] == 0 {
                    continue;
                }
                let v = [at(&d, x + 1, y) + 3, at(&d, x, y + 1) + 3, at(&d, x + 1, y + 1) + 4, at(&d, x - 1, y + 1) + 4];
                d[i] = d[i].min(v.into_iter().min().unwrap_or(0));
            }
        }
        let Some((i0, &d0)) = d.iter().enumerate().max_by_key(|(_, v)| **v) else { return Vec::new() };
        if d0 == 0 {
            return Vec::new();
        }
        let xy = |i: usize| ((i % gw) as f32, (i / gw) as f32);
        let px = |i: usize| ((xy(i).0 + 0.5) * self.k, (xy(i).1 + 0.5) * self.k);
        let mut out = vec![px(i0)];
        // (chamfer units are a third of a cell)
        let reach = 2.0 * d0 as f32 / 3.0;
        let far = d.iter().enumerate().filter(|(i, v)| **v * 5 >= d0 * 3 && (xy(*i).0 - xy(i0).0).hypot(xy(*i).1 - xy(i0).1) >= reach).max_by_key(|(_, v)| **v);
        if let Some((i1, _)) = far {
            out.push(px(i1));
        }
        out
    }
}

/// How much two shapes on one grid overlap: intersection over union (0 when both are empty).
fn iou(a: &[bool], b: &[bool]) -> f32 {
    let (mut both, mut either) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        both += usize::from(*x && *y);
        either += usize::from(*x || *y);
    }
    if either == 0 { 0.0 } else { both as f32 / either as f32 }
}

/// The box `[x0, y0, x1, y1]` round points.
fn bounds(pts: &[(f32, f32)]) -> Option<[f32; 4]> {
    let ok = pts.iter().filter(|p| p.0.is_finite() && p.1.is_finite());
    let b = ok.fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, p| [b[0].min(p.0), b[1].min(p.1), b[2].max(p.0), b[3].max(p.1)]);
    (b[2] > b[0] && b[3] > b[1]).then_some(b)
}

/// A box grown by `by` of its size each way and kept inside a `w` × `h` picture.
fn grown(b: [f32; 4], by: f32, w: usize, h: usize) -> [f32; 4] {
    let (dx, dy) = ((b[2] - b[0]) * by, (b[3] - b[1]) * by);
    [(b[0] - dx).max(0.0), (b[1] - dy).max(0.0), (b[2] + dx).min(w as f32), (b[3] + dy).min(h as f32)]
}

/// A box in pixels as the box tracker takes it: its centre and size as shares of the picture.
fn subject_box(b: [f32; 4], w: usize, h: usize) -> Option<SubjectBox> {
    let (fw, fh) = (w.max(1) as f64, h.max(1) as f64);
    let (x0, y0, x1, y1) = (f64::from(b[0]), f64::from(b[1]), f64::from(b[2]), f64::from(b[3]));
    SubjectBox::new((x0 + x1) / 2.0 / fw, (y0 + y1) / 2.0 / fh, (x1 - x0) / fw, (y1 - y0) / fh)
}

/// `shape` (shares of the picture) moved and sized as its box went from `from` to `to`.
fn moved(shape: &[(f32, f32)], from: &SubjectBox, to: &ObjectBox) -> Vec<(f32, f32)> {
    let (sx, sy) = (to.w / from.w.max(1e-6), to.h / from.h.max(1e-6));
    let ok = |v: f64| v.is_finite() && v > 0.0 && v < 20.0;
    let (sx, sy) = if ok(sx) && ok(sy) { (sx, sy) } else { (1.0, 1.0) };
    shape.iter().map(|(x, y)| (((f64::from(*x) - from.cx) * sx + to.cx) as f32, ((f64::from(*y) - from.cy) * sy + to.cy) as f32)).collect()
}

/// A picture made at most `side` on its long side, for the box tracker.
fn small(rgba: &[u8], w: usize, h: usize, side: usize) -> (Vec<u8>, usize, usize) {
    let k = (side as f32 / w.max(h).max(1) as f32).min(1.0);
    if k >= 1.0 {
        return (rgba.to_vec(), w, h);
    }
    let (tw, th) = (((w as f32 * k).round() as usize).max(1), ((h as f32 * k).round() as usize).max(1));
    crate::cutout::resize_rgba(rgba, w, h, tw, th)
}

// ---------------------------------------------------------------------------------------------
// following the thing
// ---------------------------------------------------------------------------------------------

/// What tells the follower what the thing is on the first frame (shares of the picture): the
/// points clicked on it and off it, or a mask's shape round it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Seed {
    Click { inside: Vec<(f32, f32)>, outside: Vec<(f32, f32)> },
    Shape(Vec<(f32, f32)>),
}

/// The thing as seen whole: the frame (small, for the box tracker), its outline and box (shares
/// of the picture), how much of the picture it covers, its colours, and when (seconds along the
/// run).
#[derive(Clone)]
struct Seen {
    small: (Vec<u8>, usize, usize),
    shape: Vec<(f32, f32)>,
    bbox: SubjectBox,
    area: f32,
    colours: Colours,
    t: f64,
}

/// Follows one thing through the frames an outline job looks at.
pub(crate) struct Follow {
    cutter: Arc<dyn ObjectCutter>,
    seed: Seed,
    /// The first frame's (where the run back starts from), and the last frame it was seen whole
    /// in (what the cuts are compared with: frame to frame, a thing going behind something would
    /// shrink away a little each frame and the mask creep onto what hides it).
    anchor: Option<Seen>,
    whole: Option<Seen>,
    /// The share of the picture the thing covers whole: the largest lately, let go of slowly.
    area: f32,
    /// Where it was seen whole lately: times and box middles (its speed).
    trail: Vec<(f64, f64, f64)>,
    tracker: Option<ObjectTracker>,
}

impl Follow {
    pub fn new(cutter: Arc<dyn ObjectCutter>, seed: Seed) -> Self {
        Self { cutter, seed, anchor: None, whole: None, area: 0.0, trail: Vec::new(), tracker: None }
    }

    /// The thing's outline in this frame (`points` points, the picture's pixels), `None` when it
    /// is not seen.
    pub fn look(&mut self, rgba: &[u8], w: usize, h: usize, points: usize, look: Look) -> std::result::Result<Option<Vec<(f32, f32)>>, String> {
        if w == 0 || h == 0 || rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Ok(None);
        }
        if look.fresh
            && let Some(a) = self.anchor.clone()
        {
            (self.area, self.trail) = (a.area, Vec::new());
            self.seen_whole(a);
        }
        let grid = Grid::new(w, h);
        let px = |p: &(f32, f32)| (p.0 * w as f32, p.1 * h as f32);
        let pic = self.cutter.picture(rgba, w, h)?;
        // the first frame: what the seed names
        if self.whole.is_none() {
            let cut = match &self.seed {
                Seed::Click { inside, outside } => {
                    let prompt = Prompt { inside: inside.iter().map(px).collect(), outside: outside.iter().map(px).collect(), bbox: None };
                    let answers = pic.cut(&prompt)?;
                    // one point is ambiguous (a part, the thing, more): the surest of those answers
                    let ambiguous = inside.len() + outside.len() == 1 && answers.len() > 1;
                    let pool = answers.get(usize::from(ambiguous)..).unwrap_or_default();
                    pool.iter().max_by(|a, b| a.score.total_cmp(&b.score)).cloned()
                }
                Seed::Shape(shape) => {
                    let shape: Vec<(f32, f32)> = shape.iter().map(px).collect();
                    ask(&*pic, rgba, &grid, &shape, GROW / 2.0, None)?.filter(|c| c.0 >= SAME_DRAWN).map(|c| c.1)
                }
            };
            drop(pic);
            let Some(cut) = cut else { return Ok(None) };
            let Some((outline, seen)) = self.seen_as(&cut, rgba, &grid, points, look.t) else { return Ok(None) };
            self.anchor = Some(seen.clone());
            self.area = seen.area;
            self.seen_whole(seen);
            return Ok(Some(outline));
        }
        // after: the guesses at where it went — the box tracker's, the thing's speed's, where it
        // was — the thing as seen whole moved to each; the cut most like what is expected there
        let (Some(whole), Some(tracker)) = (self.whole.as_ref(), self.tracker.as_mut()) else { return Ok(None) };
        let sm = small(rgba, w, h, ANALYSIS_SIDE);
        let obs = tracker.push(&sm.0, sm.1, sm.2, look.t);
        let held = ObjectBox { cx: whole.bbox.cx, cy: whole.bbox.cy, w: whole.bbox.w, h: whole.bbox.h, rotation: 0.0, scale: 1.0 };
        let mut guesses: Vec<ObjectBox> = Vec::with_capacity(3);
        for g in [(!obs.lost).then_some(obs.bbox), ahead(&self.trail, &whole.bbox, look.t), Some(held)].into_iter().flatten() {
            let size = whole.bbox.w.max(whole.bbox.h).max(1e-6);
            if guesses.iter().all(|q| ((q.cx - g.cx) / size).hypot((q.cy - g.cy) / size) > SAME_GUESS) {
                guesses.push(g);
            }
        }
        let mut found: Option<(f32, Matte)> = None;
        for g in &guesses {
            let expect: Vec<(f32, f32)> = moved(&whole.shape, &whole.bbox, g).iter().map(px).collect();
            if let Some(c) = ask(&*pic, rgba, &grid, &expect, GROW, Some(&whole.colours))?
                && found.as_ref().is_none_or(|f| c.0 > f.0)
            {
                found = Some(c);
            }
        }
        drop(pic);
        let Some((_, cut)) = found.filter(|f| f.0 >= SAME) else { return Ok(None) };
        let Some((outline, seen)) = self.seen_as(&cut, rgba, &grid, points, look.t) else { return Ok(None) };
        // what the thing covers whole changes slowly: much more is the thing and what is beside
        // it (taken on a little at a time, it would creep over it)
        let dt = (look.t - whole.t).abs().min(LOST_SECONDS) as f32;
        let (let_go, take_on) = (LET_GO_PER_SECOND.powf(dt), GROW_PER_SECOND.powf(dt));
        if seen.area > MAX_GROWTH * self.area * take_on {
            return Ok(None);
        }
        // seen whole (not shrunk by something in front of it): what the next cuts are compared with
        if seen.area >= WHOLE * self.area * let_go {
            self.area = seen.area.clamp(self.area * let_go, self.area * take_on);
            self.seen_whole(seen);
        }
        Ok(Some(outline))
    }

    /// The outline of a cut and the thing as seen in it.
    fn seen_as(&self, cut: &Matte, rgba: &[u8], grid: &Grid, points: usize, t: f64) -> Option<(Vec<(f32, f32)>, Seen)> {
        let (w, h) = (cut.w, cut.h);
        let outline = filmcraft_vision::matte::outline_pieces(cut, points, PIECE * cut.area(), MAX_PIECES)?;
        let bbox = subject_box(bounds(&outline)?, w, h)?;
        let (fw, fh) = (w.max(1) as f32, h.max(1) as f32);
        let shape = outline.iter().map(|(x, y)| (x / fw, y / fh)).collect();
        let colours = colours(rgba, grid, &grid.of(cut));
        Some((outline, Seen { small: small(rgba, w, h, ANALYSIS_SIDE), shape, bbox, area: cut.area(), colours, t }))
    }

    /// The thing seen whole as `seen`: the box tracker starts from it there, and its speed.
    fn seen_whole(&mut self, seen: Seen) {
        let mut tracker = ObjectTracker::new(TrackingMethod::Subject, seen.bbox);
        let _ = tracker.push(&seen.small.0, seen.small.1, seen.small.2, seen.t);
        self.tracker = Some(tracker);
        self.trail.push((seen.t, seen.bbox.cx, seen.bbox.cy));
        let keep = self.trail.len().saturating_sub(TRAIL);
        self.trail.drain(..keep);
        self.whole = Some(seen);
    }
}

/// Where the thing (`bbox` as seen whole last) is at `t` going on at the speed it went along its
/// `trail` (over its last half second there), for at most [`LOST_SECONDS`].
fn ahead(trail: &[(f64, f64, f64)], bbox: &SubjectBox, t: f64) -> Option<ObjectBox> {
    let &(t1, x1, y1) = trail.last()?;
    let &(t0, x0, y0) = trail.iter().find(|p| (t1 - p.0).abs() <= SPEED_SECONDS)?;
    let dt = t1 - t0;
    let k = (t - t1).min(LOST_SECONDS);
    if !(dt.abs() > 1e-6 && k.is_finite() && k > 0.0) {
        return None;
    }
    let (vx, vy) = ((x1 - x0) / dt, (y1 - y0) / dt);
    Some(ObjectBox { cx: x1 + vx * k, cy: y1 + vy * k, w: bbox.w, h: bbox.h, rotation: 0.0, scale: 1.0 })
}

/// Points off the thing for a prompt with box `b`: the middles of the box's sides and its
/// corners, a little inside it, where the shape expected (`want`) is not (what is round the thing
/// in the box, so MobileSAM does not take it with the thing).
fn beside(grid: &Grid, want: &[bool], b: [f32; 4]) -> Vec<(f32, f32)> {
    let (ix, iy) = ((b[2] - b[0]) * 0.04, (b[3] - b[1]) * 0.04);
    let (x0, y0, x1, y1) = (b[0] + ix, b[1] + iy, b[2] - ix, b[3] - iy);
    let (mx, my) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    [(mx, y0), (mx, y1), (x0, my), (x1, my), (x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        .into_iter()
        .filter(|(x, y)| {
            let (cx, cy) = ((x / grid.k) as usize, (y / grid.k) as usize);
            cx < grid.gw && cy < grid.gh && !want.get(cy * grid.gw + cx).copied().unwrap_or(true)
        })
        .collect()
}

/// A shape's colours: how many of its cells (sampled at their middles) fall in each of 4 × 4 × 4
/// boxes of RGB, as shares.
type Colours = [f32; 64];

fn colours(rgba: &[u8], grid: &Grid, cells: &[bool]) -> Colours {
    let mut c = [0f32; 64];
    let w = grid.w;
    for (i, _) in cells.iter().enumerate().filter(|(_, on)| **on) {
        let (x, y) = ((((i % grid.gw) as f32 + 0.5) * grid.k) as usize, (((i / grid.gw) as f32 + 0.5) * grid.k) as usize);
        if x >= w || y >= grid.h {
            continue;
        }
        if let Some(p) = rgba.get((y * w + x) * 4..(y * w + x) * 4 + 3) {
            let bin = usize::from(p[0] >> 6) * 16 + usize::from(p[1] >> 6) * 4 + usize::from(p[2] >> 6);
            if let Some(b) = c.get_mut(bin) {
                *b += 1.0;
            }
        }
    }
    let n: f32 = c.iter().sum();
    if n > 0.0 {
        c.iter_mut().for_each(|v| *v /= n);
    }
    c
}

/// How alike two shapes' colours are (the Bhattacharyya coefficient: 1 alike, 0 nothing shared).
fn alike(a: &Colours, b: &Colours) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x * y).sqrt()).sum()
}

/// MobileSAM asked for the thing expected as `expect` (picture pixels): given the box round it
/// (grown by `grow`) and its deepest points, the answer most like it and how much (intersection
/// over union on `grid`) — of those coloured like the thing (`colours`, when known) and at most
/// [`MAX_GROWTH`] times its size (not the thing and what it stands on). An answer reaching the
/// box's edge where the picture goes on is asked again with a box grown by [`GROW_MORE`], and
/// that answer taken when it is more like it.
fn ask(
    pic: &dyn Cuts,
    rgba: &[u8],
    grid: &Grid,
    expect: &[(f32, f32)],
    grow: f32,
    colours: Option<&Colours>,
) -> std::result::Result<Option<(f32, Matte)>, String> {
    let (w, h) = (grid.w, grid.h);
    let Some(b) = bounds(expect) else { return Ok(None) };
    let want = grid.fill(expect);
    let inside = grid.deepest(&want);
    if inside.is_empty() {
        return Ok(None);
    }
    let expected = want.iter().filter(|c| **c).count() as f32;
    let most = |answers: Vec<Matte>| {
        answers
            .into_iter()
            .filter_map(|m| {
                let cells = grid.of(&m);
                let size = cells.iter().filter(|c| **c).count() as f32;
                let fits = size <= MAX_GROWTH * expected && colours.is_none_or(|c| alike(&self::colours(rgba, grid, &cells), c) >= ALIKE);
                fits.then(|| (iou(&cells, &want), m))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
    };
    let bbox = grown(b, grow, w, h);
    let first = Prompt { inside: inside.clone(), outside: beside(grid, &want, bbox), bbox: Some(bbox) };
    let Some(first) = most(pic.cut(&first)?) else { return Ok(None) };
    if !grid.reaches(&grid.of(&first.1), bbox, w, h) {
        return Ok(Some(first));
    }
    let more = grown(b, GROW_MORE, w, h);
    let again = most(pic.cut(&Prompt { inside, outside: beside(grid, &want, more), bbox: Some(more) })?);
    Ok(Some(match again {
        Some(a) if a.0 > first.0 => a,
        _ => first,
    }))
}

// ---------------------------------------------------------------------------------------------
// the commands
// ---------------------------------------------------------------------------------------------

/// Points `[[x, y], …]` (or one `[x, y]`) of a parameter, each 0..1 of the picture.
fn points_p(p: &Value, k: &str, cmd: &str) -> Result<Vec<(f32, f32)>> {
    let one = |v: &Value| -> Option<(f32, f32)> {
        let a = v.as_array().filter(|a| a.len() == 2)?;
        let (x, y) = (a.first()?.as_f64()?, a.get(1)?.as_f64()?);
        ((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)).then_some((x as f32, y as f32))
    };
    let wrong = || bad(cmd, format!("`{k}` is [x, y] or [[x, y], …], each 0..1 of the picture"));
    match p.get(k) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(v @ Value::Array(a)) if a.first().is_some_and(Value::is_number) => one(v).map(|q| vec![q]).ok_or_else(wrong),
        Some(Value::Array(a)) if a.len() <= MAX_CLICKS => a.iter().map(|v| one(v).ok_or_else(wrong)).collect(),
        Some(_) => Err(wrong()),
    }
}

/// The name of a new object mask on an effect: `name`, else the next `Object (n)`.
fn new_name(e: &filmcraft_project::EffectInstance, name: Option<&str>) -> String {
    if let Some(n) = name.map(str::trim).filter(|n| !n.is_empty()) {
        return n.chars().take(120).collect();
    }
    let n = e.masks.iter().filter_map(|m| m.name.strip_prefix("Object (")?.strip_suffix(')')?.parse::<usize>().ok()).max().unwrap_or(0) + 1;
    format!("{OBJECT} ({n})")
}

/// A mask's shape at media time `mt` as shares of the clip's picture (`size`).
fn shape_of(m: &filmcraft_project::Mask, mt: Tick, size: (u32, u32)) -> Vec<(f32, f32)> {
    let (w, h) = (f64::from(size.0.max(1)), f64::from(size.1.max(1)));
    m.path_at(mt).flatten(0.5).into_iter().map(|v| ((v.x / w) as f32, (v.y / h) as f32)).collect()
}

fn object_mask(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = CMD;
    let (_, it) = crate::cutout::video_clip(s, p, cmd)?;
    if it.duration <= Tick::ZERO {
        return Err(bad(cmd, "the clip is empty"));
    }
    let ei = crate::masks::effect_index(&it.effects, p.get("effect"), None, it.id).ok_or_else(|| bad(cmd, "no such effect on the clip"))?;
    let e = it.effects.get(ei).ok_or_else(|| bad(cmd, "no such effect on the clip"))?;
    if e.def().is_none_or(|d| d.kind != filmcraft_project::EffectKind::Video) || e.effect == "motion" {
        return Err(bad(cmd, "masks go on the clip's video effects (not Motion)"));
    }
    let mut inside = points_p(p, "point", cmd)?;
    inside.extend(points_p(p, "inside", cmd)?);
    let outside = points_p(p, "outside", cmd)?;
    let into = match u64_p(p, "mask") {
        Some(m) => Some(usize::try_from(m).ok().filter(|m| *m < e.masks.len()).ok_or_else(|| bad(cmd, format!("no mask {m} on the effect")))?),
        None => None,
    };
    if inside.is_empty() && into.is_none() {
        return Err(bad(cmd, "click on the thing: `point` [x, y] (0..1 of the clip's picture)"));
    }
    // the frame: `at` seconds into the clip, else the playhead in it
    let at = match f64_p(p, "at") {
        Some(a) if a.is_finite() && a >= 0.0 && a < it.duration.seconds() => it.start + Tick::from_seconds_f64(a),
        Some(_) => return Err(bad(cmd, "`at` is seconds into the clip, before its end")),
        None => s.playhead(),
    };
    let at = at.clamp(it.start, it.end() - Tick(1));
    let (forward, backward) = match str_p(p, "track").unwrap_or("both") {
        "both" => (true, true),
        "forward" => (true, false),
        "backward" => (false, true),
        "none" => (false, false),
        t => return Err(bad(cmd, format!("`track` `{t}`: both, forward, backward or none"))),
    };
    let mode = match str_p(p, "mode") {
        Some(m) => MaskMode::from_name(m).ok_or_else(|| bad(cmd, format!("unknown mode `{m}`")))?,
        None => MaskMode::Add,
    };
    let inverted = match p.get("inverted") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(cmd, "`inverted` is true or false"))?,
    };
    let (every, points) = crate::cutout::outline_params(p, cmd, Some(EVERY), POINTS)?;
    let size = filmcraft_render::clip_size(&s.project, &it).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    // (a mask given and no points: the thing it outlines there)
    let seed = match into.and_then(|m| e.masks.get(m)).filter(|_| inside.is_empty()) {
        Some(m) => Seed::Shape(shape_of(m, it.source_time_at(at), size)),
        None => Seed::Click { inside, outside },
    };
    let name = match into.and_then(|m| e.masks.get(m)) {
        Some(m) => m.name.clone(),
        None => new_name(e, str_p(p, "name")),
    };
    let cutter = cutter(s, cmd)?;
    let o = Outlining {
        lost_ends: true,
        effect: Some(ei),
        into,
        mode,
        inverted,
        method: Some(TrackMethod::Object),
        start: Some(Start { at, forward, backward, span: None }),
        ..Outlining::new(&name, "Object Mask", "the object", "nothing to mask was found there")
    };
    run(s, cmd, &it, o, every, points, cutter, seed, bool_p(p, "wait").unwrap_or(false))
}

/// `masks.track` for a mask tracked by object: the thing it outlines at `at` (sequence time)
/// followed one way (at most `span`).
pub(crate) fn track_mask(s: &mut Session, it: &TrackItem, sel: MaskSel, at: Tick, backward: bool, span: Option<Tick>, wait: bool) -> Result<Value> {
    let cmd = "masks.track";
    let size = filmcraft_render::clip_size(&s.project, it).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    let m = it.effects.get(sel.effect).and_then(|e| e.masks.get(sel.mask)).ok_or_else(|| bad(cmd, "no such mask"))?;
    if it.duration <= Tick::ZERO {
        return Err(bad(cmd, "the clip is empty"));
    }
    let at = at.clamp(it.start, it.end() - Tick(1));
    let seed = Seed::Shape(shape_of(m, it.source_time_at(at), size));
    let cutter = cutter(s, cmd)?;
    let o = Outlining {
        lost_ends: true,
        effect: Some(sel.effect),
        into: Some(sel.mask),
        method: Some(TrackMethod::Object),
        start: Some(Start { at, forward: !backward, backward, span }),
        ..Outlining::new(&m.name, "Track Mask", "the object", "the segmenter sees nothing the mask outlines")
    };
    run(s, cmd, it, o, EVERY, POINTS, cutter, seed, wait)
}

#[allow(clippy::too_many_arguments)]
fn run(
    s: &mut Session,
    cmd: &'static str,
    it: &TrackItem,
    o: Outlining,
    every: f64,
    points: usize,
    cutter: Arc<dyn ObjectCutter>,
    seed: Seed,
    wait: bool,
) -> Result<Value> {
    let mut follow = Follow::new(cutter, seed);
    let find: FindOutline = Box::new(move |rgba, w, h, n, look| follow.look(rgba, w, h, n, look));
    let r = crate::cutout::outline_job(s, cmd, it, o, every, points, find, wait)?;
    // the mask selected: its handles on the monitor, the track buttons on it
    if let (Some(effect), Some(mask)) = (r["effect"].as_u64(), r["mask"].as_u64()) {
        s.state.selected_mask = Some(MaskSel { clip: it.id, effect: effect as usize, mask: mask as usize });
    }
    Ok(r)
}

#[cfg(test)]
#[path = "object_mask_tests.rs"]
mod tests;

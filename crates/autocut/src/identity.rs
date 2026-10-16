//! The same robot across clips: an identity pass after labelling.
//!
//! Labels alone miscount. The same robot filmed on different days of an expo gets different
//! generic labels ("wheeled robot" one day, "red toy robot" the next) and is counted twice; two
//! different robots filmed one after the other can share a generic label and be counted once.
//! So candidate pairs of clips — the same kind, filmed within [`WITHIN_DAYS`] of each other, not
//! already sharing a specific label — are shown to the vision model side by side (their best
//! frames composed into one image, [`side_by_side`]), and it says whether that is the same
//! individual robot, or at least the same model. Pairs it is sure of, and clips with the same
//! specific label within [`crate::showcase::SAME_EVENT_DAYS`], are joined (union-find) into
//! identity groups: [`ClipFacts::identity`], which [`crate::showcase::group`] counts as one robot.
//!
//! Comparisons are capped: neighbours in time are compared first (the clip just before, then the
//! one before that…), and a pair already joined through others is not asked about.

use std::collections::HashMap;
use std::io::Cursor;

use serde_json::{Value, json};

use crate::Executor;
use crate::analyze::ClipFacts;
use crate::llm::ChatModel;
use crate::showcase::{SAME_EVENT_DAYS, generic};

/// Clips filmed further apart than this are never compared.
pub const WITHIN_DAYS: i64 = 14;
/// Model comparisons per pass, at most (the default; `analyze --compare N`).
pub const MAX_COMPARISONS: usize = 150;
/// The model must be at least this sure to join two clips: a wrong merge drops a robot from the
/// show, which is worse than counting one twice.
pub const SAME_CONFIDENCE: f64 = 0.7;
/// Each half of the composed image: this tall, at most this wide; a bar between them.
const HALF_HEIGHT: u32 = 480;
const HALF_WIDTH: u32 = 960;
const GAP: u32 = 16;
/// Frames fetched for comparing: their longest side, and how many are kept at once.
const FRAME_SIDE: u32 = 640;
const FRAME_CACHE: usize = 64;
/// Largest frame decoded for composing, per side.
const MAX_DECODE_SIDE: u32 = 8192;

const SECONDS_PER_DAY: i64 = 86_400;

/// Disjoint sets of clip indices, joined to the earliest index.
struct Sets(Vec<usize>);

impl Sets {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }

    fn find(&mut self, mut x: usize) -> usize {
        // path halving; never more steps than there are sets
        for _ in 0..self.0.len() {
            let Some(&p) = self.0.get(x) else { return x };
            if p == x {
                return x;
            }
            let gp = self.0.get(p).copied().unwrap_or(p);
            if let Some(slot) = self.0.get_mut(x) {
                *slot = gp;
            }
            x = gp;
        }
        x
    }

    fn join(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if let Some(slot) = self.0.get_mut(ra.max(rb)) {
            *slot = ra.min(rb);
        }
    }
}

/// Read a reply to [`same_prompt`]: `{"same": bool, "confidence": 0..1}` → whether to join.
/// "yes"/"true" strings count, a confidence may come as a percentage; anything else (a missing
/// or absurd confidence included) is "not the same".
pub fn same_reply(v: &Value) -> (bool, f64) {
    let same = match v.get("same") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => matches!(s.trim().to_lowercase().as_str(), "true" | "yes"),
        _ => false,
    };
    let c = v.get("confidence").and_then(|c| c.as_f64().or_else(|| c.as_str().and_then(|s| s.trim().trim_end_matches('%').trim().parse::<f64>().ok())));
    let confidence = match c {
        Some(c) if (0.0..=1.0).contains(&c) => c,
        Some(c) if c > 1.0 && c <= 100.0 => c / 100.0,
        _ => 0.0,
    };
    (same, confidence)
}

/// Asked plainly ("is it the same?"), a 7B model says yes to almost any pair; describing both
/// first grounds the answer in what differs.
fn same_prompt(subject: &str) -> String {
    format!(
        "The left and right halves of this image are frames from two different videos, with a grey bar between them, each showing a {subject}. First describe each {subject} in a few words: its colours, its shape and parts (legs, wheels, arms, head, screen) and any text on it. Then decide: is it the same individual {subject}, or at least the same model of {subject} (the same design, shape and colours) filmed again? Two {subject}s that differ in colour, shape or parts are different, even when both are the same general type.\nAnswer with JSON only: {{\"left\": \"...\", \"right\": \"...\", \"same\": true|false, \"confidence\": 0.0-1.0}}"
    )
}

/// A PNG decoded for composing, within [`MAX_DECODE_SIDE`].
fn decode(png: &[u8]) -> Option<image::RgbaImage> {
    let mut r = image::ImageReader::with_format(Cursor::new(png), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_SIDE);
    limits.max_image_height = Some(MAX_DECODE_SIDE);
    r.limits(limits);
    Some(r.decode().ok()?.to_rgba8())
}

/// `img` scaled to [`HALF_HEIGHT`] tall, or narrower to stay within [`HALF_WIDTH`].
fn fit(img: &image::RgbaImage) -> image::RgbaImage {
    let (w, h) = (f64::from(img.width().max(1)), f64::from(img.height().max(1)));
    let s = (f64::from(HALF_HEIGHT) / h).min(f64::from(HALF_WIDTH) / w);
    let nw = ((w * s).round() as u32).clamp(1, HALF_WIDTH);
    let nh = ((h * s).round() as u32).clamp(1, HALF_HEIGHT);
    image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle)
}

/// Two frames side by side, each [`HALF_HEIGHT`] tall (or narrower, centred), a grey bar
/// between them: one image for the model to compare. `None` when either is not a PNG.
pub fn side_by_side(left: &[u8], right: &[u8]) -> Option<Vec<u8>> {
    let (a, b) = (fit(&decode(left)?), fit(&decode(right)?));
    let w = a.width().saturating_add(GAP).saturating_add(b.width());
    let mut out = image::RgbaImage::from_pixel(w, HALF_HEIGHT, image::Rgba([128, 128, 128, 255]));
    let mid = |img: &image::RgbaImage| i64::from(HALF_HEIGHT.saturating_sub(img.height()) / 2);
    image::imageops::overlay(&mut out, &a, 0, mid(&a));
    image::imageops::overlay(&mut out, &b, i64::from(a.width() + GAP), mid(&b));
    crate::png(out.width(), out.height(), out.into_raw(), 0)
}

/// Whole days between two clips' capture times, when both have one.
fn days_apart(a: &ClipFacts, b: &ClipFacts) -> Option<i64> {
    let (x, y) = (a.unix?, b.unix?);
    Some((x.div_euclid(SECONDS_PER_DAY) - y.div_euclid(SECONDS_PER_DAY)).abs())
}

/// The same specific (not generic) label: already one robot by name.
fn same_name(a: &ClipFacts, b: &ClipFacts, subject: &str) -> bool {
    !generic(&a.label, subject) && a.label.trim().eq_ignore_ascii_case(b.label.trim())
}

/// A clip's best frame for comparing, through a small cache.
async fn best_frame(ex: &dyn Executor, f: &ClipFacts, cache: &mut HashMap<u64, Option<Vec<u8>>>) -> Option<Vec<u8>> {
    if let Some(png) = cache.get(&f.item) {
        return png.clone();
    }
    let at = if f.best.is_finite() { f.best.max(0.0) } else { f.seconds.max(0.0) / 2.0 };
    let r = ex.execute("media.frame", json!({"item": f.item, "seconds": at, "maxSide": FRAME_SIDE})).await.ok();
    let png = r.and_then(|r| r["pngBase64"].as_str().and_then(crate::b64_decode));
    if cache.len() >= FRAME_CACHE {
        cache.clear();
    }
    cache.insert(f.item, png.clone());
    png
}

/// Group `facts` into identities: clips that show the same robot share
/// [`ClipFacts::identity`] (1, 2, … in time order); clips that do not show the subject have
/// none. At most `max_comparisons` model requests; a failed one joins nothing.
pub async fn identify(
    model: &dyn ChatModel,
    ex: &dyn Executor,
    subject: &str,
    facts: &mut [ClipFacts],
    max_comparisons: usize,
    progress: &mut dyn FnMut(&str),
) {
    // the clips that show it, oldest first (undated last)
    let mut order: Vec<usize> = (0..facts.len()).filter(|&i| facts.get(i).is_some_and(|f| f.present)).collect();
    order.sort_by_key(|&i| facts.get(i).map(|f| (f.unix.unwrap_or(i64::MAX), f.item)));
    let clip = |i: usize| facts.get(i);
    let mut sets = Sets::new(facts.len());
    // the same specific name within an event's days: one robot without asking
    for (x, &i) in order.iter().enumerate() {
        for &j in order.iter().skip(x + 1) {
            let (Some(a), Some(b)) = (clip(i), clip(j)) else { continue };
            match days_apart(a, b) {
                Some(d) if d <= SAME_EVENT_DAYS => {
                    if same_name(a, b, subject) {
                        sets.join(i, j);
                    }
                }
                // later clips are further away still
                _ => break,
            }
        }
    }
    // then the model, neighbours in time first
    let mut asked = 0usize;
    let mut joined = 0usize;
    let mut cache: HashMap<u64, Option<Vec<u8>>> = HashMap::new();
    'gaps: for gap in 1..order.len() {
        let mut any_in_reach = false;
        for x in 0..order.len().saturating_sub(gap) {
            let (Some(&i), Some(&j)) = (order.get(x), order.get(x + gap)) else { continue };
            let (Some(a), Some(b)) = (clip(i), clip(j)) else { continue };
            let Some(days) = days_apart(a, b).filter(|d| *d <= WITHIN_DAYS) else { continue };
            any_in_reach = true;
            if !a.kind.trim().eq_ignore_ascii_case(b.kind.trim()) || same_name(a, b, subject) || sets.find(i) == sets.find(j) {
                continue;
            }
            if asked >= max_comparisons {
                progress(&format!("identity: stopped after {asked} comparisons"));
                break 'gaps;
            }
            let (Some(left), Some(right)) = (best_frame(ex, a, &mut cache).await, best_frame(ex, b, &mut cache).await) else { continue };
            let Some(img) = side_by_side(&left, &right) else { continue };
            asked += 1;
            match model.complete_json("You compare robots in photos for a video editor.", &same_prompt(subject), std::slice::from_ref(&img)).await {
                Ok(v) => {
                    let (same, confidence) = same_reply(&v);
                    if same && confidence >= SAME_CONFIDENCE {
                        sets.join(i, j);
                        joined += 1;
                        progress(&format!("  {} = {} ({days} days apart, {confidence:.2})", a.name, b.name));
                    }
                }
                Err(e) => progress(&format!("  {} / {}: not compared ({e})", a.name, b.name)),
            }
        }
        // clips sorted by time: when no pair this far apart is within reach, none further is
        if !any_in_reach {
            break;
        }
    }
    // identities in time order
    let mut ids: HashMap<usize, u32> = HashMap::new();
    let mut next = 1u32;
    let mut assigned: Vec<(usize, u32)> = Vec::with_capacity(order.len());
    for &i in &order {
        let root = sets.find(i);
        let id = *ids.entry(root).or_insert_with(|| {
            let id = next;
            next = next.saturating_add(1);
            id
        });
        assigned.push((i, id));
    }
    for f in facts.iter_mut() {
        f.identity = None;
    }
    for (i, id) in assigned {
        if let Some(f) = facts.get_mut(i) {
            f.identity = Some(id);
        }
    }
    progress(&format!("identity: {} clips, {} robots ({asked} comparisons, {joined} matches)", order.len(), ids.len()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::testing::FakeEx;
    use crate::llm::FixedModel;
    use crate::showcase;

    const DAY: i64 = 1_665_446_400; // 2022-10-11, an expo's first day

    fn clip(item: u64, label: &str, kind: &str, unix: i64) -> ClipFacts {
        ClipFacts {
            item,
            name: format!("c{item}.mp4"),
            seconds: 20.0,
            unix: Some(unix),
            present: true,
            coverage: 1.0,
            label: label.into(),
            kind: kind.into(),
            best: 5.0,
            ..ClipFacts::default()
        }
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        crate::png(w, h, vec![200; (w * h * 4) as usize], 0).unwrap()
    }

    #[test]
    fn replies_are_read_strictly() {
        assert_eq!(same_reply(&json!({"same": true, "confidence": 0.9})), (true, 0.9));
        assert_eq!(same_reply(&json!({"same": "yes", "confidence": "85%"})), (true, 0.85));
        assert_eq!(same_reply(&json!({"same": true, "confidence": 80})), (true, 0.8));
        assert_eq!(same_reply(&json!({"same": true})), (true, 0.0), "no confidence: not sure");
        for v in [json!({"same": true, "confidence": 1e300}), json!({"same": true, "confidence": -1}), json!({"same": true, "confidence": "high"})] {
            assert_eq!(same_reply(&v).1, 0.0, "{v}");
        }
        assert!(!same_reply(&json!("yes")).0);
        assert!(!same_reply(&json!({"same": 1, "confidence": 1.0})).0);
        assert!(!same_reply(&json!({"same": false, "confidence": 1.0})).0);
    }

    #[test]
    fn frames_are_composed_side_by_side() {
        let img = side_by_side(&png(64, 36), &png(36, 64)).unwrap();
        let out = image::load_from_memory(&img).unwrap();
        // landscape: 853 × 480; portrait: 270 × 480; a bar between
        assert_eq!((out.width(), out.height()), (853 + GAP + 270, HALF_HEIGHT));
        // a panorama is narrowed to the half's width, and centred
        let wide = image::load_from_memory(&side_by_side(&png(400, 10), &png(8, 8)).unwrap()).unwrap();
        assert_eq!((wide.width(), wide.height()), (HALF_WIDTH + GAP + HALF_HEIGHT, HALF_HEIGHT));
        // not PNGs
        assert!(side_by_side(b"not a png", &png(8, 8)).is_none());
        assert!(side_by_side(&png(8, 8), &[]).is_none());
        let mut cut = png(64, 36);
        cut.truncate(cut.len() / 2);
        assert!(side_by_side(&cut, &png(8, 8)).is_none());
    }

    #[test]
    fn sets_join_to_the_earliest() {
        let mut s = Sets::new(5);
        s.join(3, 4);
        s.join(4, 1);
        assert_eq!((s.find(3), s.find(4), s.find(1)), (1, 1, 1));
        assert_eq!(s.find(0), 0);
        assert_eq!(s.find(99), 99, "out of range is itself");
    }

    /// One expo: a red wheeled-legged robot filmed on day one (twice, under different generic
    /// labels) and again on day four with its name read; Pepper on days two and four; a wheeled
    /// robot a month later; a clip without one.
    fn expo() -> Vec<ClipFacts> {
        vec![
            clip(1, "wheeled robot", "wheeled", DAY),
            clip(2, "red toy robot", "wheeled", DAY + 600),
            clip(3, "Pepper humanoid robot", "humanoid", DAY + 86_400),
            ClipFacts {
                text: crate::analyze::FrameText { on_subject: vec!["ARDET".into(), "Swiss-Mile".into()], nearby: vec!["ETH zürich RSL".into()] },
                ..clip(4, "Swiss-Mile ARDET robot", "wheeled", DAY + 3 * 86_400)
            },
            clip(5, "Pepper humanoid robot", "humanoid", DAY + 3 * 86_400 + 60),
            clip(6, "wheeled robot", "wheeled", DAY + 30 * 86_400),
            ClipFacts { item: 7, present: false, unix: Some(DAY), ..ClipFacts::default() },
        ]
    }

    fn ids(facts: &[ClipFacts]) -> Vec<Option<u32>> {
        facts.iter().map(|f| f.identity).collect()
    }

    #[tokio::test]
    async fn the_model_joins_clips_of_one_robot() {
        let ex = FakeEx::new(vec![]);
        let mut facts = expo();
        // asked: (1, 2) on day one, then (2, 4); (1, 4) is joined through them by then
        let model = FixedModel::new(vec![json!({"same": true, "confidence": 0.9}), json!({"same": true, "confidence": 0.85})]);
        let mut lines = Vec::new();
        identify(&model, &ex, "robot", &mut facts, MAX_COMPARISONS, &mut |l| lines.push(l.to_string())).await;
        assert_eq!(ids(&facts), vec![Some(1), Some(1), Some(2), Some(1), Some(2), Some(3), None], "{lines:?}");
        assert_eq!(model.asked.lock().unwrap().len(), 2, "Pepper by name, the others too far apart or not alike");
        assert!(model.asked.lock().unwrap()[0].1.contains("same individual robot"));
        assert!(lines.iter().any(|l| l.contains("c1.mp4 = c2.mp4")), "{lines:?}");
        // c2's frame serves both of its comparisons
        assert_eq!(ex.frames.lock().unwrap().iter().filter(|f| f.0 == 2).count(), 1);
        // the showcase counts each identity once, across days
        let entries = showcase::group(&facts, "robot", 0.3);
        let shape: Vec<(&str, Vec<u64>)> = entries.iter().map(|e| (e.label.as_str(), e.shots.iter().map(|s| s.item).collect())).collect();
        assert_eq!(
            shape,
            vec![("Swiss-Mile ARDET robot", vec![1, 2, 4]), ("Pepper humanoid robot", vec![3, 5]), ("wheeled robot", vec![6])],
            "the most specific label names the robot"
        );
    }

    #[tokio::test]
    async fn unsure_or_different_answers_keep_robots_apart() {
        let ex = FakeEx::new(vec![]);
        let mut facts = expo();
        // (1, 2) unsure, (2, 4) not the same, (1, 4) a garbled reply
        let model = FixedModel::new(vec![json!({"same": true, "confidence": 0.5}), json!({"same": false, "confidence": 0.9}), json!("same!")]);
        identify(&model, &ex, "robot", &mut facts, MAX_COMPARISONS, &mut |_| {}).await;
        assert_eq!(ids(&facts), vec![Some(1), Some(2), Some(3), Some(4), Some(3), Some(5), None]);
        assert_eq!(model.asked.lock().unwrap().len(), 3);
        // the showcase keeps them apart: five robots
        assert_eq!(showcase::group(&facts, "robot", 0.3).len(), 5);
        // a model that never answers joins nothing, and never fails the pass
        let mut facts = expo();
        identify(&FixedModel::new(vec![]), &ex, "robot", &mut facts, MAX_COMPARISONS, &mut |_| {}).await;
        assert_eq!(ids(&facts), vec![Some(1), Some(2), Some(3), Some(4), Some(3), Some(5), None]);
    }

    #[tokio::test]
    async fn comparisons_are_capped() {
        let ex = FakeEx::new(vec![]);
        let mut facts = expo();
        let model = FixedModel::new(vec![json!({"same": true, "confidence": 0.9}), json!({"same": true, "confidence": 0.9})]);
        identify(&model, &ex, "robot", &mut facts, 1, &mut |_| {}).await;
        assert_eq!(model.asked.lock().unwrap().len(), 1);
        assert_eq!(ids(&facts), vec![Some(1), Some(1), Some(2), Some(3), Some(2), Some(4), None]);
        let mut facts = expo();
        identify(&model, &ex, "robot", &mut facts, 0, &mut |_| {}).await;
        assert_eq!(model.asked.lock().unwrap().len(), 1, "0: by name only");
        assert_eq!(ids(&facts), vec![Some(1), Some(2), Some(3), Some(4), Some(3), Some(5), None]);
    }

    #[tokio::test]
    async fn a_many_clip_library_stays_bounded() {
        // 300 clips of one kind in one week: neighbours first, at most the cap asked
        let ex = FakeEx::new(vec![]);
        let mut facts: Vec<ClipFacts> = (0..300).map(|i| clip(i, "wheeled robot", "wheeled", DAY + i as i64 * 1_800)).collect();
        let model = FixedModel::new((0..300).map(|_| json!({"same": false, "confidence": 0.9})).collect());
        identify(&model, &ex, "robot", &mut facts, 20, &mut |_| {}).await;
        assert_eq!(model.asked.lock().unwrap().len(), 20);
        assert_eq!(facts.iter().filter_map(|f| f.identity).max(), Some(300));
    }
}

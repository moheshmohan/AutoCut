//! Which moments the director may use. Every moment is ranked on what its picture and sound say
//! about it — sharpness, exposure and steadiness (measured on the moment's own frames by
//! `media.measureMoments` when it can, else the scene index's shot), the aesthetic score, faces
//! (the face detector's, else the describer's people), expressions (the describer's words:
//! smiling, laughing, waving…; there is no expression model), speech worth hearing (English said
//! in it, an exclamation, a word of wonder), and its energy. The unusable are left out (out of
//! focus, too dark, blown out, too shaky for a flat shot), then the near-duplicates of a better
//! moment (image embeddings at least [`DUPLICATE`] alike; without embeddings, the same file's
//! descriptions), as long as enough moments stay to fill the edit; the rest keep filming order.
//! Each moment's [`Verdict`] says why it was kept or dropped.

use serde::Serialize;
use serde_json::{Value, json};

use super::{Candidate, Kind, chars, finite, round2, run};
use crate::Executor;
use crate::quick::ShotInfo;

/// Two moments of different shots whose image embeddings are at least this alike show the same
/// thing (SigLIP 2: different rooms of one trip measured 0.6–0.86).
pub const DUPLICATE: f64 = 0.93;
/// Without embeddings: two moments of one file whose descriptions share this much of their words.
const SAME_WORDS: f64 = 0.8;

/// Words of the describer's that say someone shows a feeling.
const EXPRESSION_WORDS: &[&str] =
    &["smil", "laugh", "grin", "cheer", "wav", "excit", "happy", "joy", "celebrat", "hug", "danc", "jump", "surpris", "thumbs", "clap", "playful"];
/// Words that make a line worth hearing.
const WONDER_WORDS: &[&str] =
    &["wow", "whoa", "look", "amazing", "awesome", "beautiful", "incredible", "yes", "yay", "nice", "cool", "love", "great", "oh", "here", "finally"];

/// What a moment's picture and sound score, 0..1 each.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Scores {
    pub sharp: f64,
    pub exposure: f64,
    pub steady: f64,
    /// The aesthetic score (LAION, 3.5..6.5 → 0..1), when the model ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub looks: Option<f64>,
    pub people: f64,
    pub expression: f64,
    pub speech: f64,
    pub energy: f64,
    /// Shares of crushed blacks and blown whites, the mean brightness and the camera's motion
    /// (frame widths a second): what a flaw is told by.
    pub dark: f64,
    pub blown: f64,
    pub brightness: f64,
    pub motion: f64,
    /// Measured on the moment's own frames (else the scene index's whole shot).
    pub measured: bool,
}

fn unit(v: f64) -> f64 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

impl Scores {
    /// The moment's rank, 0..1: how it looks first, then who is in it, what is said, its energy
    /// (a 360° moment a little more: it can be framed many ways).
    pub fn rank(&self, kind: Kind) -> f64 {
        let looks = self.looks.unwrap_or((self.sharp + self.exposure) / 2.0);
        let r = 0.22 * self.sharp
            + 0.18 * self.exposure
            + 0.12 * self.steady
            + 0.12 * looks
            + 0.10 * self.people
            + 0.08 * self.expression
            + 0.10 * self.speech
            + 0.08 * self.energy
            + if kind == Kind::Vr { 0.04 } else { 0.0 };
        round2(unit(r))
    }

    /// The moment's rank for an edit about places, 0..1: how it looks, how wide and whether
    /// outdoors first; faces, smiles and speech count for little.
    pub fn rank_for_places(&self, kind: Kind, wide: bool, indoor: bool) -> f64 {
        let looks = self.looks.unwrap_or((self.sharp + self.exposure) / 2.0);
        let r = 0.22 * self.sharp
            + 0.18 * self.exposure
            + 0.12 * self.steady
            + 0.20 * looks
            + 0.04 * self.people
            + 0.02 * self.expression
            + 0.02 * self.speech
            + 0.08 * self.energy
            + if wide { 0.06 } else { 0.0 }
            + if indoor { 0.0 } else { 0.04 }
            + if kind == Kind::Vr { 0.02 } else { 0.0 };
        round2(unit(r))
    }

    /// The picture's quality alone, 0..1.
    pub fn quality(&self) -> f64 {
        let base = 0.4 * self.sharp + 0.3 * self.exposure + 0.3 * self.steady;
        unit(match self.looks {
            Some(l) => 0.75 * base + 0.25 * l,
            None => base,
        })
    }

    /// What makes it unusable, if anything.
    pub fn flaw(&self, kind: Kind) -> Option<String> {
        if self.sharp < 0.12 && self.motion < 0.2 {
            return Some(format!("out of focus (sharpness {:.2})", self.sharp));
        }
        if self.dark > 0.5 || self.brightness < 0.1 {
            return Some(format!("too dark ({:.0} % black)", self.dark * 100.0));
        }
        if self.blown > 0.3 {
            return Some(format!("blown out ({:.0} % white)", self.blown * 100.0));
        }
        // (a 360° shot is stabilised when it is used)
        if kind != Kind::Vr && self.steady < 0.25 {
            return Some(format!("too shaky (shake {:.2})", 1.0 - self.steady));
        }
        None
    }

    /// Its strong points and weak ones, in words.
    fn words(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        let mut good = Vec::new();
        let mut weak = Vec::new();
        for (v, g, w) in [
            (self.sharp, "sharp", "soft"),
            (self.exposure, "well exposed", "badly exposed"),
            (self.steady, "steady", "shaky"),
            (self.looks.unwrap_or(0.5), "good-looking", "plain"),
        ] {
            if v >= 0.75 {
                good.push(g);
            } else if v < 0.35 {
                weak.push(w);
            }
        }
        if self.people >= 0.6 {
            good.push("faces in view");
        }
        if self.expression >= 0.5 {
            good.push("smiles or cheering");
        }
        if self.speech >= 0.5 {
            good.push("speech worth hearing");
        }
        if self.energy >= 0.8 {
            good.push("lively");
        }
        (good, weak)
    }
}

/// A moment's scores from its shot (the scene index) and its speech: `said` the English said in
/// it, `share` how much of the shot is kept speech.
pub fn scores_of(s: &ShotInfo, said: Option<&str>, share: f64, energy: f64) -> Scores {
    let q = &s.shot.quality;
    let d = s.description();
    let text = s.shot.text();
    let words: Vec<&str> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let expression = if words.iter().any(|w| EXPRESSION_WORDS.iter().any(|e| w.starts_with(e))) { 1.0 } else { 0.0 };
    let people = match d {
        Some(d) if d.faces_prominent => 0.9,
        Some(d) if d.people > 0 => (0.4 + 0.1 * f64::from(d.people.min(4))).min(0.8),
        _ => 0.0,
    };
    Scores {
        sharp: unit(f64::from(q.sharpness)),
        exposure: unit(f64::from(q.exposure)),
        steady: unit(1.0 - f64::from(q.shake)),
        looks: q.aesthetic.map(f64::from).filter(|a| a.is_finite()).map(|a| unit((a - 3.5) / 3.0)),
        people,
        expression,
        speech: speech_score(said, share),
        energy: unit(energy),
        dark: unit(f64::from(q.dark)),
        blown: unit(f64::from(q.blown)),
        brightness: unit(f64::from(q.brightness)),
        motion: f64::from(q.motion).max(0.0),
        measured: false,
    }
}

/// How much a moment's speech is worth hearing: English said in it (more for an exclamation or
/// a question, more for a word of wonder), else a share of kept speech in another language.
pub fn speech_score(said: Option<&str>, share: f64) -> f64 {
    match said.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => {
            let lower = t.to_lowercase();
            let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
            let mut v: f64 = 0.5;
            if t.contains('!') || t.contains('?') {
                v += 0.2;
            }
            if words.iter().any(|w| WONDER_WORDS.contains(w)) {
                v += 0.3;
            }
            v.min(1.0)
        }
        None if share > 0.3 => 0.35,
        None => 0.0,
    }
}

/// A moment measured on its own frames (`media.measureMoments`'s entry): its picture scores
/// replace the shot's, its faces count.
pub fn apply_measure(c: &mut Candidate, m: &Value) {
    if m["measured"] != true {
        return;
    }
    let sc = &mut c.scores;
    let get = |k: &str| finite(&m[k]).map(unit);
    if let Some(v) = get("sharpness") {
        sc.sharp = v;
    }
    if let Some(v) = get("exposure") {
        sc.exposure = v;
    }
    if let Some(v) = get("shake") {
        sc.steady = 1.0 - v;
    }
    if let Some(v) = get("dark") {
        sc.dark = v;
    }
    if let Some(v) = get("blown") {
        sc.blown = v;
    }
    if let Some(v) = get("brightness") {
        sc.brightness = v;
    }
    if let Some(v) = finite(&m["motion"]) {
        sc.motion = v.max(0.0);
    }
    if let Ok(cam) = serde_json::from_value(m["camera"].clone()) {
        c.pan = cam;
    }
    if let Some(n) = m["faces"].as_u64().filter(|n| *n > 0) {
        let size = get("largestFace").unwrap_or(0.0);
        sc.people = sc.people.max(unit(0.6 + 0.05 * n.min(4) as f64 + 0.3 * (size / 0.2).min(1.0)));
    }
    sc.measured = true;
    c.rank = sc.rank(c.kind);
    c.quality = sc.quality();
}

/// Measure every moving moment on its own frames, a call per item; how many were measured.
pub async fn measure(ex: &dyn Executor, cands: &mut [Candidate], lines: &mut dyn FnMut(&str)) -> usize {
    let mut items: Vec<u64> = cands.iter().filter(|c| c.kind != Kind::Photo).map(|c| c.item).collect();
    items.sort_unstable();
    items.dedup();
    if items.is_empty() {
        return 0;
    }
    lines(&format!("measuring {} moment(s) on their own frames", cands.iter().filter(|c| c.kind != Kind::Photo).count()));
    let mut done = 0;
    for item in items {
        let idx: Vec<usize> = (0..cands.len()).filter(|&i| cands.get(i).is_some_and(|c| c.item == item && c.kind != Kind::Photo)).collect();
        for chunk in idx.chunks(filmcraft_engine::moments::MAX_MOMENTS) {
            let moments: Vec<Value> = chunk.iter().filter_map(|&i| cands.get(i)).map(|c| json!([round2(c.start), round2(c.end)])).collect();
            let Ok(v) = run(ex, "media.measureMoments", json!({"item": item, "moments": moments, "frames": 3})).await else { continue };
            let got = v["moments"].as_array().map(Vec::as_slice).unwrap_or_default();
            for (&i, m) in chunk.iter().zip(got) {
                if let Some(c) = cands.get_mut(i) {
                    apply_measure(c, m);
                    done += usize::from(c.scores.measured);
                }
            }
        }
    }
    done
}

/// Why a moment was kept or dropped.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Verdict {
    pub clip: String,
    pub name: String,
    pub from: f64,
    pub to: f64,
    pub rank: f64,
    pub scores: Scores,
    pub kept: bool,
    pub why: String,
}

/// The cosine similarity of two embeddings (`None`: empty or of other lengths).
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    let c = d / (na.sqrt() * nb.sqrt());
    c.is_finite().then_some(c)
}

fn word_overlap(a: &str, b: &str) -> f64 {
    let set = |s: &str| -> std::collections::BTreeSet<String> {
        s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2).map(str::to_string).collect()
    };
    let (x, y) = (set(a), set(b));
    let union = x.union(&y).count();
    if union == 0 { 0.0 } else { x.intersection(&y).count() as f64 / union as f64 }
}

/// How alike two moments of different shots are (`None`: nothing to tell by).
fn alike(a: &Candidate, b: &Candidate) -> Option<f64> {
    if a.item == b.item && a.shot == b.shot {
        // (moments of one shot are different moments of it, not copies)
        return None;
    }
    match cosine(&a.embedding, &b.embedding) {
        Some(c) => Some(c),
        None if a.item == b.item => Some(word_overlap(&a.caption, &b.caption) * DUPLICATE / SAME_WORDS),
        None => None,
    }
}

/// The moments kept (filming order) and every moment's verdict (in the order they came): the
/// unusable and the near-duplicates dropped while at least `keep` moments stay.
pub fn choose(cands: Vec<Candidate>, keep: usize) -> (Vec<Candidate>, Vec<Verdict>) {
    let n = cands.len();
    let mut dropped: Vec<Option<String>> = vec![None; n];
    let mut left = n;
    // the unusable, the worst first
    let mut by_rank: Vec<usize> = (0..n).collect();
    by_rank.sort_by(|&a, &b| cands.get(a).map_or(0.0, |c| c.rank).total_cmp(&cands.get(b).map_or(0.0, |c| c.rank)));
    let mut kept_though: Vec<Option<String>> = vec![None; n];
    for &i in &by_rank {
        let Some(c) = cands.get(i) else { continue };
        let Some(f) = c.scores.flaw(c.kind) else { continue };
        if left > keep {
            dropped[i] = Some(f);
            left -= 1;
        } else {
            kept_though[i] = Some(f);
        }
    }
    // the near-duplicates of a better moment
    by_rank.reverse();
    let mut kept_so_far: Vec<usize> = Vec::new();
    for &i in &by_rank {
        let Some(c) = cands.get(i) else { continue };
        if dropped[i].is_some() {
            continue;
        }
        let twin = kept_so_far.iter().filter_map(|&k| Some((k, alike(c, cands.get(k)?)?))).filter(|(_, v)| *v >= DUPLICATE).max_by(|a, b| a.1.total_cmp(&b.1));
        match twin {
            Some((k, v)) if left > keep => {
                let other = cands.get(k).map_or("?", |o| o.id.as_str());
                dropped[i] = Some(format!("a near-duplicate of {other} ({:.2} alike), which ranks higher", v.min(1.0)));
                left -= 1;
            }
            _ => kept_so_far.push(i),
        }
    }
    let mut verdicts = Vec::with_capacity(n);
    let mut kept = Vec::with_capacity(left);
    for (i, c) in cands.into_iter().enumerate() {
        let why = match (&dropped[i], &kept_though[i]) {
            (Some(d), _) => format!("dropped: {d}"),
            (None, Some(f)) => format!("kept though {f}: too few moments otherwise"),
            (None, None) => {
                let (good, weak) = c.scores.words();
                let mut w = format!("kept: rank {:.2}", c.rank);
                if !good.is_empty() {
                    w.push_str(&format!(" — {}", good.join(", ")));
                }
                if !weak.is_empty() {
                    w.push_str(&format!("; {}", weak.join(", ")));
                }
                if let Some(s) = c.said.as_deref().filter(|_| c.scores.speech >= 0.5) {
                    w.push_str(&format!("; says \"{}\"", chars(s, 60)));
                }
                w
            }
        };
        verdicts.push(Verdict {
            clip: c.id.clone(),
            name: c.name.clone(),
            from: round2(c.start),
            to: round2(c.end),
            rank: c.rank,
            scores: c.scores.clone(),
            kept: dropped[i].is_none(),
            why,
        });
        if dropped[i].is_none() {
            kept.push(c);
        }
    }
    kept.sort_by(|a, b| a.order.cmp(&b.order).then(a.start.total_cmp(&b.start)));
    (kept, verdicts)
}

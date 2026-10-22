//! Ranking shots for a text query: how much the shot *looks* like it (SigLIP image–text
//! similarity, when the query was embedded), how much its description *says* it (the query's
//! words in the caption, subjects, actions, setting, text…), and whether it is *spoken* in the
//! shot (the query's words in the transcript words within its time).

use filmcraft_project::scenes::Shot;
use serde::Serialize;

/// Words too common to search by.
const STOP: &[&str] = &[
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with", "for", "is", "are", "be", "it", "its", "this", "that", "some", "any", "shot", "shots",
    "clip", "clips", "video", "footage", "scene", "scenes", "where", "who", "which", "show", "shows", "showing", "find", "me", "my",
];

/// Weights of the three signals when all are there.
const W_VISUAL: f32 = 0.6;
const W_TEXT: f32 = 0.3;
const W_SPEECH: f32 = 0.2;

/// One shot to rank.
pub struct Candidate<'a> {
    pub item: u64,
    /// Index of the shot in its item's index.
    pub shot: usize,
    pub data: &'a Shot,
    /// The words spoken during the shot (lower case), when the item has a transcript.
    pub speech: String,
}

/// A ranked shot.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hit {
    pub item: u64,
    pub shot: usize,
    /// 0..1, the combined score the list is sorted by.
    pub score: f32,
    /// Image–text match, 0..1 (absent without an embedding).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visual: Option<f32>,
    /// Share of the query's words in the description.
    pub text: f32,
    /// Share of the query's words spoken in the shot.
    pub speech: f32,
}

/// How cosine similarity becomes a 0..1 match: SigLIP's own sigmoid with its learned scale and
/// bias (`sigmoid(scale · cos + bias)`), or, without them, the cosine rescaled over the
/// candidates (best 1, worst 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub scale: f32,
    pub bias: f32,
}

/// The query's search words: lower case, letters and digits, stop words out, a plural `s`
/// dropped (`robots` finds `robot`).
pub fn terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in query.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).take(64) {
        let w = stem(&w.to_lowercase());
        if !STOP.contains(&w.as_str()) && !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

fn stem(w: &str) -> String {
    let chars = w.chars().count();
    if chars > 3 && w.ends_with("ies") {
        format!("{}y", w.trim_end_matches("ies"))
    } else if chars > 3 && w.ends_with('s') && !w.ends_with("ss") {
        w.trim_end_matches('s').to_string()
    } else {
        w.to_string()
    }
}

/// Share of `terms` found among the words of `text` (stemmed the same way); with several terms,
/// the whole query appearing as a phrase counts a little more.
pub fn word_match(terms: &[String], text: &str, phrase: &str) -> f32 {
    if terms.is_empty() || text.is_empty() {
        return 0.0;
    }
    let words: Vec<String> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| stem(&w.to_lowercase())).collect();
    let found = terms.iter().filter(|t| words.iter().any(|w| w == *t || (t.len() >= 5 && w.starts_with(t.as_str())))).count();
    let share = found as f32 / terms.len() as f32;
    if terms.len() < 2 {
        return share;
    }
    let phrase = phrase.trim().to_lowercase();
    let whole = !phrase.is_empty() && text.to_lowercase().contains(&phrase);
    (0.85 * share + if whole { 0.15 } else { 0.0 }).min(1.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Rank `candidates` for `query` (and its embedding, when the query was embedded), best first,
/// at most `limit`. Shots that match nothing are left out.
pub fn rank(query: &str, embedding: Option<&[f32]>, calibration: Option<Calibration>, candidates: &[Candidate<'_>], limit: usize) -> Vec<Hit> {
    let t = terms(query);
    let cos: Vec<Option<f32>> = candidates.iter().map(|c| embedding.filter(|_| !c.data.embedding.is_empty()).map(|e| c.data.embedding.cosine(e))).collect();
    let (lo, hi) = cos.iter().flatten().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let visual = |c: f32| -> f32 {
        let v = match calibration {
            Some(k) if k.scale.is_finite() && k.bias.is_finite() => sigmoid(k.scale * c + k.bias),
            _ if hi > lo => (c - lo) / (hi - lo),
            _ => 0.5,
        };
        if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
    };
    let mut hits: Vec<Hit> = candidates
        .iter()
        .zip(&cos)
        .map(|(c, cs)| {
            let text = word_match(&t, &c.data.text(), query);
            let speech = word_match(&t, &c.speech, query);
            let v = cs.map(visual);
            let score = match v {
                Some(v) => (W_VISUAL * v + W_TEXT * text + W_SPEECH * speech).min(1.0),
                // no picture to compare: the words alone
                None => ((W_TEXT + W_VISUAL) * text + W_SPEECH * speech).min(1.0),
            };
            Hit { item: c.item, shot: c.shot, score: if score.is_finite() { score } else { 0.0 }, visual: v, text, speech }
        })
        .filter(|h| h.score > 0.0 && (h.text > 0.0 || h.speech > 0.0 || h.visual.is_some_and(|v| v >= 0.05)))
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.item.cmp(&b.item)).then(a.shot.cmp(&b.shot)));
    hits.truncate(limit);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::scenes::{Embedding, ShotDescription, Subject};

    fn shot(caption: &str, subject: &str, emb: &[f32]) -> Shot {
        Shot {
            description: Some(ShotDescription { caption: caption.into(), subjects: vec![Subject { name: subject.into(), bbox: None }], ..Default::default() }),
            embedding: Embedding(emb.to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn query_terms() {
        assert_eq!(terms("Find the shots of robots dancing!"), vec!["robot", "dancing"]);
        assert_eq!(terms("cities and puppies"), vec!["city", "puppy"]);
        assert!(terms("the a of").is_empty());
        assert_eq!(terms("glass"), vec!["glass"]);
    }

    #[test]
    fn embeddings_rank_what_looks_right() {
        // stub embeddings: the query looks like shot 1, a little like shot 2, not like shot 0
        let s = [
            shot("a beach at sunset", "sea", &[0.0, 1.0, 0.0]),
            shot("a robot on a stage", "robot", &[1.0, 0.0, 0.0]),
            shot("a crowd", "people", &[0.6, 0.8, 0.0]),
        ];
        let c: Vec<Candidate> = s.iter().enumerate().map(|(i, d)| Candidate { item: 7, shot: i, data: d, speech: String::new() }).collect();
        let hits = rank("android", Some(&[1.0, 0.0, 0.0]), None, &c, 10);
        assert_eq!(hits.iter().map(|h| h.shot).collect::<Vec<_>>(), vec![1, 2], "no word matches: the picture decides; the unlike shot is left out");
        // the description's words add to the picture
        let hits = rank("crowd", Some(&[1.0, 0.0, 0.0]), None, &c, 10);
        assert_eq!(hits[0].shot, 2, "{hits:?}");
        // calibrated: SigLIP's sigmoid
        let k = Calibration { scale: 10.0, bias: -5.0 };
        let hits = rank("android", Some(&[1.0, 0.0, 0.0]), Some(k), &c, 1);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].visual.unwrap() - 0.993).abs() < 0.01);
    }

    #[test]
    fn words_and_speech_rank_without_embeddings() {
        let s = [shot("a dog runs on grass", "dog", &[]), shot("a cat sleeps", "cat", &[]), Shot::default()];
        let mut c: Vec<Candidate> = s.iter().enumerate().map(|(i, d)| Candidate { item: 1, shot: i, data: d, speech: String::new() }).collect();
        c[2].speech = "and here is my dog again".into();
        let hits = rank("dogs", None, None, &c, 10);
        assert_eq!(hits.iter().map(|h| h.shot).collect::<Vec<_>>(), vec![0, 2]);
        assert!(hits[0].visual.is_none());
        assert!(hits[1].speech > 0.99);
        assert!(rank("zebra", None, None, &c, 10).is_empty());
        assert!(rank("dog", None, None, &c, 0).is_empty());
        // a phrase counts more than scattered words
        let p = [shot("the red car parked by a blue house", "car", &[]), shot("a blue car beside a red house", "car", &[])];
        let c: Vec<Candidate> = p.iter().enumerate().map(|(i, d)| Candidate { item: 1, shot: i, data: d, speech: String::new() }).collect();
        assert_eq!(rank("red car", None, None, &c, 10)[0].shot, 0);
    }

    #[test]
    fn hostile_embeddings_do_not_break_ranking() {
        let s = [shot("x", "y", &[f32::NAN, 1.0]), shot("x", "y", &[1.0]), shot("x", "y", &[0.0, 0.0])];
        let c: Vec<Candidate> = s.iter().enumerate().map(|(i, d)| Candidate { item: 1, shot: i, data: d, speech: String::new() }).collect();
        let hits = rank("q", Some(&[1.0, 0.0]), Some(Calibration { scale: f32::NAN, bias: 0.0 }), &c, 10);
        assert!(hits.iter().all(|h| h.score.is_finite() && (0.0..=1.0).contains(&h.score)));
    }
}

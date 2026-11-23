//! Music fitted to the film (PX2): the pipelines' film events and ending, through `clip.remix`.
//!
//! Direct, Quick Short, Showcase and the trailer pass what their film does — a `burst` at the
//! opening, `peak`s at the hero shots or title hits, a `landing` at the end card, a `stop` at a
//! hard cut to silence — and how the music should end; `clip.remix` lands real lifts of the
//! user's track on them ([`filmcraft_engine`]'s `filmcraft_audio_dsp::fit`) and says how each
//! landed. The shorts stay silent (the user adds a song to them), so they do not use this.

use serde_json::{Value, json};

use crate::showcase::{Beats, run};
use crate::{AutoCutError, Executor};

/// A film event for `clip.remix`: `kind` at `seconds` from the music clip's start.
pub fn event(kind: &str, seconds: f64) -> Option<Value> {
    (seconds.is_finite() && seconds >= 0.0).then(|| json!({"kind": kind, "seconds": (seconds * 1000.0).round() / 1000.0}))
}

/// Up to `most` of the `(time, score)` moments, the best first, each at least `gap` seconds from
/// the others and inside `(lo, hi)`; returned in time order.
pub fn peaks(moments: &[(f64, f64)], most: usize, gap: f64, lo: f64, hi: f64) -> Vec<f64> {
    let mut v: Vec<(f64, f64)> = moments.iter().copied().filter(|(t, s)| t.is_finite() && s.is_finite() && *t > lo && *t < hi).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.total_cmp(&b.0)));
    let mut out: Vec<f64> = Vec::new();
    for (t, _) in v {
        if out.len() >= most {
            break;
        }
        if out.iter().all(|o| (o - t).abs() >= gap) {
            out.push(t);
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// Fit music clip `clip` to `seconds` with `events` and `ending`: the remix's report.
pub async fn fit(ex: &dyn Executor, clip: u64, seconds: f64, events: &[Value], ending: &str) -> Result<Value, AutoCutError> {
    run(ex, "clip.remix", json!({"clip": clip, "seconds": seconds, "events": events, "ending": ending})).await
}

/// What a fit did, for a pipeline's report: how it ended, how each event landed, its joins and
/// its warnings.
pub fn summary(r: &Value) -> Value {
    let joins = r["joins"].as_array().map_or(0, Vec::len);
    let refused = r["joins"].as_array().map_or(0, |j| j.iter().filter(|x| x["ok"] == json!(false)).count());
    json!({
        "ending": r["ending"],
        "ended": r["ended"],
        "seconds": r["seconds"],
        "eventsLanded": r["eventsLanded"],
        "joins": joins,
        "joinsOverLevel": refused,
        "warnings": r["warnings"],
    })
}

/// The fit's warnings that need a look, as pipeline notes (`music: …`): a join kept although it
/// jumps in level, or a fit that fell back to the plain remix. How each event landed stays in
/// the report ([`summary`]).
pub fn notes(r: &Value) -> Vec<String> {
    r["warnings"]
        .as_array()
        .map(|w| {
            w.iter()
                .filter_map(Value::as_str)
                .filter(|w| w.contains("changes the level") || w.contains("could not be fitted"))
                .take(20)
                .map(|w| format!("music: {w}"))
                .collect()
        })
        .unwrap_or_default()
}

/// The beats of a track as `clip.remix` plays it (its report `r`): each piece (`src` and `len`
/// in ticks of the track) plays the track's beats in it, one after the other.
pub fn played_beats(track: &Beats, r: &Value) -> Option<Beats> {
    crate::showcase::remixed_beats(track, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_are_the_best_spaced_moments_in_order() {
        let m = [(5.0, 0.2), (12.0, 0.9), (13.0, 0.95), (30.0, 0.5), (50.0, 0.99), (f64::NAN, 1.0), (70.0, 1.0)];
        assert_eq!(peaks(&m, 2, 8.0, 1.0, 60.0), vec![13.0, 50.0]);
        assert_eq!(peaks(&m, 3, 8.0, 1.0, 60.0), vec![13.0, 30.0, 50.0]);
        assert!(peaks(&m, 0, 8.0, 1.0, 60.0).is_empty());
        assert!(peaks(&[], 3, 8.0, 0.0, 1.0).is_empty());
    }

    #[test]
    fn events_refuse_bad_times() {
        assert!(event("peak", f64::NAN).is_none());
        assert!(event("peak", -1.0).is_none());
        assert_eq!(event("burst", 0.0), Some(json!({"kind": "burst", "seconds": 0.0})));
    }

    #[test]
    fn a_summary_counts_the_joins_over_level() {
        let r = json!({"ending": "own", "ended": "cadence", "seconds": 30.0, "eventsLanded": [{"kind": "peak", "landed": false}], "joins": [{"ok": true}, {"ok": false}], "warnings": []});
        let s = summary(&r);
        assert_eq!(s["joins"], 2);
        assert_eq!(s["joinsOverLevel"], 1);
        assert!(notes(&r).is_empty());
        let r = json!({"warnings": ["the peak at 3.00 s: no lift of the track reaches it", "the join at 4.00 s (from 1.00 s to 9.00 s of the track) changes the level by +9.0 dB, 8.0 dB more than the music does there"]});
        assert_eq!(notes(&r).len(), 1);
        assert!(notes(&r)[0].starts_with("music: the join"));
    }
}

//! Slot auto-fit: how a piece of media conforms to a placement of exact length (the
//! iMovie / CapCut contract — media conforms to the slot, never the reverse).
//!
//! Longer media is trimmed (the caller's `in` point, else the slot's `portion`); shorter media
//! loops, stretches (slows down to at most `minSpeed`), holds its last frame, is handed to Remix
//! (music), or is refused, as the slot's [`Fit`] says. Pictures without a length (stills,
//! generated media) simply take the placement's length. Every piece boundary is on a frame of
//! the sequence; the pieces always add up to exactly the placement's length.

use filmcraft_time::{FrameRate, Tick};

use crate::manifest::{Fit, Portion, ShortFit};

/// Most pieces a looped placement may be cut into.
pub const MAX_LOOP_PIECES: usize = 256;

/// One clip of a fitted placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    /// Timeline offset from the placement's start.
    pub offset: Tick,
    pub duration: Tick,
    /// Media time of the first frame shown.
    pub source_in: Tick,
    pub speed: f64,
    /// A held frame (media time) for the whole piece.
    pub hold: Option<Tick>,
}

/// How a placement was filled.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    pub pieces: Vec<Piece>,
    /// `still`, `exact`, `trimmed`, `looped`, `stretched`, `held` or `remix`.
    pub how: &'static str,
    /// Remix the (single) piece to this length afterwards (music).
    pub remix_to: Option<Tick>,
    /// Something the caller should hear about (a stretch that became a loop…).
    pub note: Option<String>,
}

fn secs(t: Tick) -> String {
    format!("{:.2} s", t.seconds())
}

/// Fit media of length `len` (`None`: no length — a still or generated picture) to a placement of
/// `need`. `chosen_in` is the caller's in point (media time); `rate` is the sequence's frame rate.
pub fn fit(need: Tick, len: Option<Tick>, chosen_in: Option<Tick>, policy: &Fit, rate: FrameRate) -> Result<Fitted, String> {
    if need.0 <= 0 {
        return Err("the placement has no length".into());
    }
    let frame = rate.frame_duration().max(Tick(1));
    let single = |source_in: Tick, how: &'static str| Fitted {
        pieces: vec![Piece { offset: Tick::ZERO, duration: need, source_in, speed: 1.0, hold: None }],
        how,
        remix_to: None,
        note: None,
    };
    let Some(len) = len else {
        return Ok(single(chosen_in.unwrap_or(Tick::ZERO).max(Tick::ZERO), "still"));
    };
    if len.0 <= 0 {
        return Err("the media has no length".into());
    }
    if let Some(i) = chosen_in
        && (i < Tick::ZERO || i >= len)
    {
        return Err(format!("the in point ({}) is outside the media ({})", secs(i), secs(len)));
    }
    if len - chosen_in.unwrap_or(Tick::ZERO) >= need {
        let source_in = match chosen_in {
            Some(i) => i,
            None => {
                let spare = len - need;
                match policy.portion {
                    Portion::Start => Tick::ZERO,
                    Portion::Center => rate.snap(Tick(spare.0 / 2)),
                    Portion::End => spare,
                }
            }
        };
        return Ok(single(source_in, if len == need && chosen_in.is_none() { "exact" } else { "trimmed" }));
    }
    let start = chosen_in.unwrap_or(Tick::ZERO);
    let avail = len - start;
    let short = |why: &str| format!("the media from its in point is {} long, the slot needs {}{why}", secs(avail), secs(need));
    match policy.short {
        ShortFit::Error => Err(short(" (this slot does not loop or stretch)")),
        ShortFit::Stretch => {
            let speed = avail.0 as f64 / need.0 as f64;
            if speed >= policy.min_speed {
                // the clip must not read past the media's end once its length is rounded
                let mut s = speed;
                if (need.0 as f64 * s).round() as i64 > avail.0 {
                    s = (avail.0 - 1).max(1) as f64 / need.0 as f64;
                }
                Ok(Fitted {
                    pieces: vec![Piece { offset: Tick::ZERO, duration: need, source_in: start, speed: s, hold: None }],
                    how: "stretched",
                    remix_to: None,
                    note: None,
                })
            } else {
                let mut f = looped(need, start, avail, frame)?;
                f.note = Some(format!("slowing it to {:.0} % would pass the slot's limit ({:.0} %), so it loops", speed * 100.0, policy.min_speed * 100.0));
                Ok(f)
            }
        }
        ShortFit::Hold => {
            let played = rate.snap(avail);
            if played < frame {
                return Err(short(" (less than a frame to hold)"));
            }
            let hold_at = start + played - frame;
            Ok(Fitted {
                pieces: vec![
                    Piece { offset: Tick::ZERO, duration: played, source_in: start, speed: 1.0, hold: None },
                    Piece { offset: played, duration: need - played, source_in: hold_at, speed: 1.0, hold: Some(hold_at) },
                ],
                how: "held",
                remix_to: None,
                note: None,
            })
        }
        ShortFit::Remix => {
            let played = rate.snap(avail);
            if played < frame {
                return Err(short(" (less than a frame of music)"));
            }
            Ok(Fitted {
                pieces: vec![Piece { offset: Tick::ZERO, duration: played, source_in: start, speed: 1.0, hold: None }],
                how: "remix",
                remix_to: Some(need),
                note: None,
            })
        }
        ShortFit::Loop => looped(need, start, avail, frame),
    }
}

/// `need` filled with repeats of the media's `[start, start + avail)`, each a whole number of
/// frames (the last ends where the placement does).
pub fn looped(need: Tick, start: Tick, avail: Tick, frame: Tick) -> Result<Fitted, String> {
    let step = Tick(avail.0 - avail.0.rem_euclid(frame.0.max(1)));
    if step < frame {
        return Err("the media is shorter than a frame".into());
    }
    let n = need.0.div_euclid(step.0).saturating_add(1);
    if n > MAX_LOOP_PIECES as i64 {
        return Err(format!("the media is too short to loop over the slot ({n} repeats; at most {MAX_LOOP_PIECES})"));
    }
    let mut pieces = Vec::new();
    let mut at = Tick::ZERO;
    while at < need {
        let d = step.min(need - at);
        pieces.push(Piece { offset: at, duration: d, source_in: start, speed: 1.0, hold: None });
        at += d;
    }
    Ok(Fitted { how: if pieces.len() > 1 { "looped" } else { "trimmed" }, pieces, remix_to: None, note: None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_time::TICKS_PER_SECOND;

    const S: i64 = TICKS_PER_SECOND;
    fn fit_of(short: ShortFit) -> Fit {
        Fit { short, ..Fit::default() }
    }
    fn total(f: &Fitted) -> Tick {
        f.pieces.iter().map(|p| p.duration).fold(Tick::ZERO, |a, b| a + b)
    }
    const R: FrameRate = FrameRate::FPS_30;

    #[test]
    fn long_media_is_trimmed_from_the_chosen_portion() {
        let need = Tick(2 * S);
        let f = fit(need, Some(Tick(10 * S)), None, &Fit::default(), R).unwrap();
        assert_eq!((f.how, f.pieces.len(), f.pieces[0].source_in, f.pieces[0].duration), ("trimmed", 1, Tick::ZERO, need));
        let c = fit(need, Some(Tick(10 * S)), None, &Fit { portion: Portion::Center, ..Fit::default() }, R).unwrap();
        assert_eq!(c.pieces[0].source_in, Tick(4 * S));
        let e = fit(need, Some(Tick(10 * S)), None, &Fit { portion: Portion::End, ..Fit::default() }, R).unwrap();
        assert_eq!(e.pieces[0].source_in, Tick(8 * S));
        let chosen = fit(need, Some(Tick(10 * S)), Some(Tick(3 * S)), &Fit::default(), R).unwrap();
        assert_eq!(chosen.pieces[0].source_in, Tick(3 * S));
        assert!(fit(need, Some(Tick(10 * S)), Some(Tick(11 * S)), &Fit::default(), R).is_err());
        assert!(fit(need, Some(Tick(10 * S)), Some(Tick(-1)), &Fit::default(), R).is_err());
    }

    #[test]
    fn stills_take_any_length() {
        let f = fit(Tick(7 * S), None, None, &Fit::default(), R).unwrap();
        assert_eq!((f.how, total(&f)), ("still", Tick(7 * S)));
    }

    #[test]
    fn short_media_loops_in_whole_frames_to_the_exact_length() {
        let need = Tick(5 * S);
        let len = Tick(S + S / 3 + 77); // 1.333… s, not on a frame
        let f = fit(need, Some(len), None, &fit_of(ShortFit::Loop), R).unwrap();
        assert_eq!(f.how, "looped");
        assert_eq!(total(&f), need, "pieces add up to the placement exactly");
        for w in f.pieces.windows(2) {
            assert_eq!(w[0].offset + w[0].duration, w[1].offset, "back to back");
        }
        for p in &f.pieces[..f.pieces.len() - 1] {
            assert_eq!(R.snap(p.duration), p.duration, "whole frames");
            assert!(p.duration <= len);
        }
        // a sliver of media cannot loop over a long placement
        assert!(fit(Tick(3600 * S), Some(Tick(S / 10)), None, &fit_of(ShortFit::Loop), R).is_err());
        assert!(fit(need, Some(Tick(5)), None, &fit_of(ShortFit::Loop), R).is_err());
    }

    #[test]
    fn short_media_stretches_within_the_limit_then_loops() {
        let need = Tick(4 * S);
        let f = fit(need, Some(Tick(3 * S)), None, &fit_of(ShortFit::Stretch), R).unwrap();
        assert_eq!(f.how, "stretched");
        let p = f.pieces[0];
        assert_eq!(p.duration, need);
        assert!((p.speed - 0.75).abs() < 1e-9);
        assert!((need.0 as f64 * p.speed).round() as i64 <= 3 * S, "never reads past the end");
        let l = fit(need, Some(Tick(S)), None, &fit_of(ShortFit::Stretch), R).unwrap();
        assert_eq!(l.how, "looped", "a quarter speed is past the 50 % limit");
        assert!(l.note.is_some());
    }

    #[test]
    fn hold_remix_and_error() {
        let need = Tick(4 * S);
        let h = fit(need, Some(Tick(S + 5)), None, &fit_of(ShortFit::Hold), R).unwrap();
        assert_eq!((h.how, h.pieces.len(), total(&h)), ("held", 2, need));
        assert_eq!(h.pieces[1].hold, Some(Tick(S) - R.frame_duration()));
        let r = fit(need, Some(Tick(3 * S)), None, &fit_of(ShortFit::Remix), R).unwrap();
        assert_eq!((r.how, r.remix_to, r.pieces[0].duration), ("remix", Some(need), Tick(3 * S)));
        assert!(fit(need, Some(Tick(S)), None, &fit_of(ShortFit::Error), R).is_err());
        assert!(fit(Tick::ZERO, Some(Tick(S)), None, &Fit::default(), R).is_err());
        assert!(fit(need, Some(Tick::ZERO), None, &Fit::default(), R).is_err());
    }
}

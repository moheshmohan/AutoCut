//! Shots from cuts: where TransNetV2 (or the fallback detector) says one shot ends and the next
//! begins, turned into contiguous frame ranges, and the frames looked at in each shot.
//!
//! Frame indices here are the analysed frames (the clip decoded at its own rate, at most
//! [`MAX_ANALYSIS_FPS`] a second); the engine maps them to media time.

/// TransNetV2 looks at windows of this many frames…
pub const WINDOW: usize = 100;
/// …and keeps the predictions for the middle this many (the first and last 25 are context).
pub const WINDOW_KEEP: usize = 50;
const WINDOW_CONTEXT: usize = (WINDOW - WINDOW_KEEP) / 2;
/// The frame rate TransNetV2 was trained on is that of ordinary video (24–30); faster footage
/// is analysed at every second (third…) frame so motion looks as it did in training.
pub const MAX_ANALYSIS_FPS: f64 = 30.0;
/// Most frames analysed for one clip (a 2-hour clip at 30 fps).
pub const MAX_FRAMES: usize = 216_000;

/// The frames of each TransNetV2 window for a clip of `n` frames, as indices into the clip's
/// frames: 25 copies of the first frame, the frames, then copies of the last frame up to a whole
/// number of windows; windows start every [`WINDOW_KEEP`] frames. Window `w`'s predictions
/// `25..75` are frames `50·w .. 50·w + 50` of the clip.
pub fn transnet_windows(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let n = n.min(MAX_FRAMES);
    let tail = WINDOW_CONTEXT + WINDOW_KEEP - (if n.is_multiple_of(WINDOW_KEEP) { WINDOW_KEEP } else { n % WINDOW_KEEP });
    let total = WINDOW_CONTEXT + n + tail;
    let index = |p: usize| p.saturating_sub(WINDOW_CONTEXT).min(n - 1);
    let mut out = Vec::new();
    let mut start = 0;
    while start + WINDOW <= total {
        out.push((start..start + WINDOW).map(index).collect());
        start += WINDOW_KEEP;
    }
    out
}

/// The first frames of new shots from per-frame transition probabilities: each run of frames
/// above `threshold` is one transition (a hard cut flags the last frame of the shot before it; a
/// dissolve flags several), and the new shot starts after the run's middle.
pub fn cuts_from_probabilities(p: &[f32], threshold: f32) -> Vec<usize> {
    let thr = if threshold.is_finite() { threshold.clamp(0.01, 0.99) } else { 0.5 };
    let mut cuts = Vec::new();
    let mut run: Option<usize> = None;
    for (i, v) in p.iter().enumerate() {
        let on = v.is_finite() && *v > thr;
        match (on, run) {
            (true, None) => run = Some(i),
            (false, Some(a)) => {
                cuts.push((a + i - 1) / 2 + 1);
                run = None;
            }
            _ => {}
        }
    }
    // a transition still running at the end ends the clip: no new shot
    cuts.retain(|c| *c > 0 && *c < p.len());
    cuts.dedup();
    cuts
}

/// Contiguous shots `(first, end)` (end exclusive) over `n` frames from the first frames of new
/// shots; a shot shorter than `min_frames` joins the one before it (the first joins the next).
pub fn shots_from_cuts(cuts: &[usize], n: usize, min_frames: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let mut bounds: Vec<usize> = cuts.iter().copied().filter(|c| *c > 0 && *c < n).collect();
    bounds.sort_unstable();
    bounds.dedup();
    let min = min_frames.max(1);
    let mut shots: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for c in bounds.into_iter().chain(std::iter::once(n)) {
        if c - start < min {
            // too short: merged with what follows (at the end, with what came before)
            if c == n
                && let Some(last) = shots.last_mut()
            {
                last.1 = n;
                start = n;
            }
            continue;
        }
        shots.push((start, c));
        start = c;
    }
    if start < n {
        match shots.last_mut() {
            Some(last) => last.1 = n,
            None => shots.push((0, n)),
        }
    }
    shots
}

/// `k` frames spread over a shot `[a, b)`: the middles of `k` equal parts (one: the middle).
pub fn keyframes(a: usize, b: usize, k: usize) -> Vec<usize> {
    if b <= a || k == 0 {
        return Vec::new();
    }
    let len = b - a;
    let k = k.min(len);
    let mut out: Vec<usize> = (0..k).map(|i| a + ((2 * i + 1) * len) / (2 * k)).map(|f| f.min(b - 1)).collect();
    out.dedup();
    out
}

/// How many keyframes a shot of `seconds` gets: one for the fast pass; for `full`, one per four
/// seconds, two to three.
pub fn keyframe_count(seconds: f64, full: bool) -> usize {
    if !full || !seconds.is_finite() {
        return 1;
    }
    ((seconds / 4.0).ceil() as usize).clamp(2, 3)
}

/// Every how many decoded frames one is analysed, for footage at `fps`.
pub fn frame_step(fps: f64) -> usize {
    if !fps.is_finite() || fps <= MAX_ANALYSIS_FPS {
        return 1;
    }
    ((fps / MAX_ANALYSIS_FPS).round() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_every_frame_once() {
        for n in [1usize, 2, 49, 50, 51, 99, 100, 101, 250, 1234] {
            let w = transnet_windows(n);
            assert!(w.iter().all(|x| x.len() == WINDOW));
            // the kept middles, in order, are the clip's frames (then padding)
            let kept: Vec<usize> = w.iter().flat_map(|x| x[WINDOW_CONTEXT..WINDOW_CONTEXT + WINDOW_KEEP].to_vec()).collect();
            assert!(kept.len() >= n && kept.len() < n + WINDOW_KEEP, "n={n}: {}", kept.len());
            assert_eq!(&kept[..n], &(0..n).collect::<Vec<_>>()[..], "n={n}");
            assert!(w.iter().flatten().all(|i| *i < n));
            assert_eq!(w[0][0], 0, "padded with the first frame");
        }
        assert!(transnet_windows(0).is_empty());
    }

    #[test]
    fn cuts_from_runs() {
        let mut p = vec![0.0f32; 100];
        p[29] = 0.9; // hard cut after frame 29
        p[60] = 0.7;
        p[61] = 0.8;
        p[62] = 0.6; // a three-frame dissolve
        assert_eq!(cuts_from_probabilities(&p, 0.5), vec![30, 62]);
        p[99] = 0.9; // a transition at the very end: no new shot
        assert_eq!(cuts_from_probabilities(&p, 0.5), vec![30, 62]);
        p[0] = 0.9; // the first frame flagged: the next frame starts a shot
        assert_eq!(cuts_from_probabilities(&p, 0.5), vec![1, 30, 62]);
        assert!(cuts_from_probabilities(&[f32::NAN, 0.9], f32::NAN).is_empty());
        assert!(cuts_from_probabilities(&[], 0.5).is_empty());
    }

    #[test]
    fn shots_are_contiguous_and_merge_short_ones() {
        assert_eq!(shots_from_cuts(&[30, 62], 100, 1), vec![(0, 30), (30, 62), (62, 100)]);
        // a 2-frame flash joins the shot before it
        assert_eq!(shots_from_cuts(&[30, 32, 62], 100, 6), vec![(0, 30), (30, 62), (62, 100)]);
        // a short first shot joins the next, a short last one the one before
        assert_eq!(shots_from_cuts(&[3, 50, 98], 100, 6), vec![(0, 50), (50, 100)]);
        // nothing usable: one shot
        assert_eq!(shots_from_cuts(&[0, 100, 500], 100, 6), vec![(0, 100)]);
        assert_eq!(shots_from_cuts(&[], 3, 6), vec![(0, 3)]);
        assert!(shots_from_cuts(&[1], 0, 6).is_empty());
        // unsorted, repeated
        assert_eq!(shots_from_cuts(&[62, 30, 30], 100, 1), vec![(0, 30), (30, 62), (62, 100)]);
    }

    #[test]
    fn keyframes_spread_inside_the_shot() {
        assert_eq!(keyframes(0, 100, 1), vec![50]);
        assert_eq!(keyframes(10, 20, 2), vec![12, 17]);
        assert_eq!(keyframes(0, 2, 5), vec![0, 1]);
        assert!(keyframes(5, 5, 1).is_empty());
        assert_eq!(keyframe_count(3.0, false), 1);
        assert_eq!(keyframe_count(3.0, true), 2);
        assert_eq!(keyframe_count(30.0, true), 3);
        assert_eq!(frame_step(25.0), 1);
        assert_eq!(frame_step(59.94), 2);
        assert_eq!(frame_step(120.0), 4);
        assert_eq!(frame_step(f64::NAN), 1);
    }
}

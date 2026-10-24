//! Streaming sample-rate conversion for playback, when the output device runs at another rate
//! than the sequence (a 44.1 kHz device playing a 48 kHz sequence).
//!
//! Every output sample's source position comes from its absolute output frame, as an exact
//! rational (`frame × from / to`), and is linearly interpolated, so the output is identical however
//! the stream is cut into blocks: no phase jump at block boundaries. Source frames are pulled in
//! contiguous, non-overlapping spans, so whatever produces them (the mixer, with its effect state)
//! sees one continuous stream. Output lags the source by one source sample (≈21 µs at 48 kHz),
//! which is what lets each block interpolate without reading ahead.

/// Converts a planar stream from `from` Hz to `to` Hz, block by block.
#[derive(Debug, Clone)]
pub struct StreamResampler {
    from: u32,
    to: u32,
    /// Source frames `[held_start, held_start + held[c].len())` kept from the last pull.
    held_start: i64,
    held: Vec<Vec<f32>>,
    /// The output frame the next block is expected to start at (contiguous playback).
    next_out: Option<i64>,
}

/// Maximum source frames fetched in a single resampler pull.
pub const MAX_RESAMPLE_SOURCE_SPAN: usize = 384_000 * 10;

impl StreamResampler {
    /// Rates of zero are treated as 1 Hz rather than dividing by zero.
    pub fn new(from: u32, to: u32) -> Self {
        Self { from: from.max(1), to: to.max(1), held_start: 0, held: Vec::new(), next_out: None }
    }

    pub fn rates(&self) -> (u32, u32) {
        (self.from, self.to)
    }

    /// The source frame whose sample lands at (or just before) output frame `out`.
    fn source_index(&self, out: i64) -> (i64, f32) {
        let p = out as i128 * self.from as i128;
        let to = self.to as i128;
        (p.div_euclid(to) as i64, (p.rem_euclid(to) as f64 / to as f64) as f32)
    }

    /// Output frames `[out, out + n)`, planar. `pull(start, frames)` returns source frames
    /// `[start, start + frames)`, planar (missing samples read as silence). A block that does not
    /// continue the previous one (a seek) starts the stream afresh.
    pub fn process(&mut self, out: i64, n: usize, mut pull: impl FnMut(i64, usize) -> Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if n == 0 {
            return vec![Vec::new(); self.held.len().max(1)];
        }
        let (first, _) = self.source_index(out);
        let (last, _) = self.source_index(out.saturating_add(n as i64 - 1));
        // interpolation reads source frames first - 1 ..= last
        if self.next_out != Some(out) || self.held.is_empty() {
            self.held_start = first - 1;
            self.held.clear();
        }
        let held_end = self.held_start + self.held.first().map_or(0, Vec::len) as i64;
        if last + 1 > held_end {
            let want = ((last + 1 - held_end) as usize).min(MAX_RESAMPLE_SOURCE_SPAN);
            let got = pull(held_end, want);
            if self.held.is_empty() {
                self.held = vec![Vec::new(); got.len().max(1)];
            }
            for (c, h) in self.held.iter_mut().enumerate() {
                let src = got.get(c).map(Vec::as_slice).unwrap_or_default();
                h.extend((0..want).map(|i| src.get(i).copied().unwrap_or(0.0)));
            }
        }
        let mut dst = vec![vec![0.0f32; n]; self.held.len()];
        for (i, frame) in (out..).take(n).enumerate() {
            let (s, frac) = self.source_index(frame);
            // held starts at first - 1, so the earlier frame is always held
            let k = usize::try_from(s - self.held_start).unwrap_or(0);
            for (d, h) in dst.iter_mut().zip(&self.held) {
                let a = k.checked_sub(1).and_then(|j| h.get(j)).copied().unwrap_or(0.0);
                let b = h.get(k).copied().unwrap_or(0.0);
                if let Some(x) = d.get_mut(i) {
                    *x = a + (b - a) * frac;
                }
            }
        }
        // keep what the next block can still read: from its first frame - 1 on
        let (next_first, _) = self.source_index(out.saturating_add(n as i64));
        let drop = (next_first - 1 - self.held_start).clamp(0, self.held.first().map_or(0, Vec::len) as i64) as usize;
        for h in &mut self.held {
            h.drain(..drop);
        }
        self.held_start += drop as i64;
        self.next_out = Some(out.saturating_add(n as i64));
        dst
    }
}

/// Offline rate conversion of a whole mono buffer with a windowed-sinc (Blackman) low-pass at
/// 95 % of the lower Nyquist: no aliasing when going down (24 kHz speech into a 22.05 kHz model),
/// no imaging when going up. Equal rates return a copy; empty input or a zero rate returns empty.
/// The output length is `round(len · to / from)`, capped at 2³¹ samples.
pub fn resample_sinc(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if x.is_empty() || from == 0 || to == 0 {
        return Vec::new();
    }
    if from == to {
        return x.to_vec();
    }
    let ratio = f64::from(to) / f64::from(from);
    let n_out = ((x.len() as f64 * ratio).round() as usize).min(1 << 31);
    // cutoff relative to the source rate; widen the kernel when down-sampling
    let cutoff = 0.95 * 0.5 * ratio.min(1.0);
    const HALF_ZEROS: f64 = 16.0;
    let half = (HALF_ZEROS / (2.0 * cutoff)).ceil() as isize;
    let mut out = Vec::new();
    if out.try_reserve_exact(n_out).is_err() {
        return Vec::new();
    }
    let len = x.len() as isize;
    for k in 0..n_out {
        let pos = k as f64 / ratio;
        let centre = pos.floor() as isize;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for j in (centre - half + 1)..=(centre + half) {
            let d = pos - j as f64;
            let arg = 2.0 * cutoff * d;
            let sinc = if arg.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg) };
            let t = (d / (half as f64 + 1.0) + 1.0) * 0.5; // 0..1 across the window
            let w = if (0.0..=1.0).contains(&t) {
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * t).cos() + 0.08 * (4.0 * std::f64::consts::PI * t).cos()
            } else {
                0.0
            };
            let h = sinc * w;
            wsum += h;
            if (0..len).contains(&j) {
                acc += h * f64::from(x.get(j as usize).copied().unwrap_or(0.0));
            }
        }
        out.push(if wsum.abs() > 1e-12 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn sinc_resampling_keeps_tones_and_rejects_aliases() {
        let tone =
            |hz: f64, rate: u32, n: usize| -> Vec<f32> { (0..n).map(|i| (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin() as f32).collect() };
        let rms = |v: &[f32]| (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / v.len().max(1) as f64).sqrt();
        // 1 kHz survives 24 k → 22.05 k at the right length and level
        let x = tone(1000.0, 24_000, 24_000);
        let y = resample_sinc(&x, 24_000, 22_050);
        assert_eq!(y.len(), 22_050);
        let mid = &y[2000..20_000];
        assert!((rms(mid) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02, "level {}", rms(mid));
        // and is still 1 kHz: count zero crossings over one second
        let zc = y.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        assert!((1990..=2010).contains(&zc), "{zc} crossings");
        // 11.5 kHz is above the new Nyquist (11.025 kHz): it must be removed, not folded down
        let hi = tone(11_500.0, 24_000, 24_000);
        let z = resample_sinc(&hi, 24_000, 22_050);
        assert!(rms(&z[2000..20_000]) < 0.05, "alias {}", rms(&z[2000..20_000]));
        // up-sampling keeps the tone too
        let up = resample_sinc(&tone(440.0, 22_050, 22_050), 22_050, 24_000);
        assert_eq!(up.len(), 24_000);
        assert!((rms(&up[2000..22_000]) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02);
    }

    #[test]
    fn sinc_resampling_edge_cases() {
        assert!(resample_sinc(&[], 24_000, 22_050).is_empty());
        assert!(resample_sinc(&[1.0, 2.0], 0, 22_050).is_empty());
        assert!(resample_sinc(&[1.0, 2.0], 24_000, 0).is_empty());
        assert_eq!(resample_sinc(&[1.0, 2.0], 48_000, 48_000), vec![1.0, 2.0]);
        assert_eq!(resample_sinc(&[0.5; 10], 1, 3).len(), 30);
    }

    use super::*;

    /// A sine at `hz`, sampled at `rate`, from absolute frame `start`; logs every pull.
    fn sine(hz: f64, rate: u32, pulls: &mut Vec<(i64, usize)>) -> impl FnMut(i64, usize) -> Vec<Vec<f32>> + '_ {
        move |start, frames| {
            pulls.push((start, frames));
            let ch: Vec<f32> = (0..frames).map(|i| (2.0 * std::f64::consts::PI * hz * (start + i as i64) as f64 / rate as f64).sin() as f32).collect();
            vec![ch.clone(), ch]
        }
    }

    fn convert(from: u32, to: u32, blocks: &[usize], total: usize) -> (Vec<f32>, Vec<(i64, usize)>) {
        let mut r = StreamResampler::new(from, to);
        let mut pulls = Vec::new();
        let mut out = Vec::new();
        let mut pos = 0i64;
        let mut k = 0;
        while (pos as usize) < total {
            let n = blocks[k % blocks.len()].min(total - pos as usize);
            let b = r.process(pos, n, sine(1000.0, from, &mut pulls));
            assert_eq!(b.len(), 2);
            assert!(b.iter().all(|c| c.len() == n));
            out.extend_from_slice(&b[0]);
            pos += n as i64;
            k += 1;
        }
        (out, pulls)
    }

    #[test]
    fn output_is_the_same_however_the_stream_is_cut() {
        for (from, to) in [(48_000, 44_100), (44_100, 48_000), (48_000, 96_000), (96_000, 48_000)] {
            let (whole, _) = convert(from, to, &[20_000], 20_000);
            for blocks in [&[512][..], &[333], &[1, 7, 512, 64]] {
                let (cut, _) = convert(from, to, blocks, 20_000);
                assert_eq!(cut, whole, "{from}→{to} in blocks {blocks:?}");
            }
        }
    }

    #[test]
    fn a_tone_converts_to_the_same_tone() {
        // 1 kHz at 48 kHz played on a 44.1 kHz device: within linear interpolation's error of the
        // ideal tone, one source sample late
        let (from, to) = (48_000u32, 44_100u32);
        let (out, _) = convert(from, to, &[512], 20_000);
        for (o, x) in out.iter().enumerate().skip(2) {
            let t = o as f64 / to as f64 - 1.0 / from as f64;
            let want = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
            assert!((x - want).abs() < 3e-3, "frame {o}: {x} vs {want}");
        }
    }

    #[test]
    fn the_source_is_pulled_as_one_continuous_stream() {
        for (from, to) in [(48_000, 44_100), (44_100, 48_000)] {
            let (_, pulls) = convert(from, to, &[512, 333], 20_000);
            for w in pulls.windows(2) {
                assert_eq!(w[0].0 + w[0].1 as i64, w[1].0, "{from}→{to}: {:?} then {:?}", w[0], w[1]);
            }
        }
    }

    #[test]
    fn a_seek_starts_afresh_and_hostile_input_stays_finite() {
        let mut r = StreamResampler::new(48_000, 44_100);
        let mut pulls = Vec::new();
        let _ = r.process(0, 512, sine(1000.0, 48_000, &mut pulls));
        let _ = r.process(1_000_000, 512, sine(1000.0, 48_000, &mut pulls));
        assert_eq!(pulls.last().map(|p| p.0), Some(1_000_000 * 48_000 / 44_100 - 1));
        // a mixer returning too few channels and samples, and zero rates
        let mut r = StreamResampler::new(0, 0);
        let b = r.process(5, 64, |_, _| vec![vec![1.0; 3]]);
        assert_eq!(b.len(), 1);
        assert!(b[0].iter().all(|x| x.is_finite()));
        assert!(r.process(0, 0, |_, _| Vec::new()).iter().all(Vec::is_empty));
    }
}

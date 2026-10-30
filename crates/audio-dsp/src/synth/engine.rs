//! Car engines: [`Kind::EngineRev`](super::Kind::EngineRev) blips the throttle twice from idle,
//! [`Kind::EngineLaunch`](super::Kind::EngineLaunch) holds the revs on the limiter, launches and
//! pulls through two gear changes, and [`Kind::EngineFlyby`](super::Kind::EngineFlyby) passes at
//! speed from left to right.
//!
//! A four-stroke engine modelled from its firing pulses: each cylinder fires once every two turns
//! of the crank, its exhaust pulse (a fast attack, a few milliseconds of decay, a little noise)
//! summed with the others', coloured by the exhaust's resonances and a short pipe, driven into a
//! soft clip as the load rises, with the intake's roar over it. Lifting off at high revs crackles
//! (the overrun), and so does a gear change. The revs follow the throttle with the engine's
//! inertia, so one model blips, launches or passes by. The seed picks the engine — 6, 8, 10 or 12
//! cylinders, how unevenly its banks fire (a cross-plane V8's burble), its pipe and redline — and
//! the intensity how hard it is driven. No samples: the same parameters always give the same
//! sound.

use std::f64::consts::TAU;

use super::{Clamped, Pink, Rng, Sweep, fade_tail, pan_gains, reverb, wrap};
use crate::biquad::{Biquad, Coeffs, FilterType};

/// Revs at idle.
const IDLE: f64 = 900.0;
/// Speed of sound, m/s (the fly-by's Doppler shift).
const SOUND: f64 = 343.0;

/// Which engine sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Drive {
    Rev,
    Launch,
    Flyby,
}

/// The engine the seed picks.
struct Engine {
    /// Each cylinder's firing point in the four-stroke cycle (0..1) and its pulse's strength.
    fire: Vec<(f64, f64)>,
    redline: f64,
    /// The exhaust's three resonances, Hz.
    res: [f64; 3],
    /// The pipe's echo, seconds.
    pipe: f64,
}

impl Engine {
    fn pick(seed: u64) -> Engine {
        let mut rng = Rng::new(seed ^ 0x00E6_617E);
        let cylinders = [8usize, 8, 6, 10, 12].get((rng.next_u64() % 5) as usize).copied().unwrap_or(8);
        let fire = (0..cylinders)
            .map(|k| {
                let even = k as f64 / cylinders as f64;
                // the banks fire a little unevenly: the burble
                let jitter = (rng.unit() - 0.5) * 0.14 / cylinders as f64;
                (wrap(even + jitter), 0.7 + 0.6 * rng.unit())
            })
            .collect();
        let redline = match cylinders {
            6 => 7200.0,
            8 => 7600.0,
            10 => 8300.0,
            _ => 8700.0,
        } + 400.0 * rng.unit();
        let scale = 0.8 + 0.45 * rng.unit();
        Engine { fire, redline, res: [105.0 * scale, 310.0 * scale, 880.0 * scale], pipe: 0.0035 + 0.003 * rng.unit() }
    }
}

/// What the driver does at `u` (0..1 through the sound): the throttle, and whether the car is in
/// gear (`false`: free revving, in neutral or on the launch limiter).
fn driver(drive: Drive, u: f64) -> (f64, bool) {
    match drive {
        // two blips, the second harder and longer, then the revs fall back to idle
        Drive::Rev => {
            let on = (0.10..0.19).contains(&u) || (0.45..0.60).contains(&u);
            (if on { 1.0 } else { 0.0 }, false)
        }
        // idle, the revs held on the launch limiter, then three gears and a lift at the very end
        Drive::Launch => {
            if u < 0.08 {
                (0.0, false)
            } else if u < 0.30 {
                (1.0, false)
            } else {
                (if u < 0.95 { 1.0 } else { 0.0 }, true)
            }
        }
        // flat out past the listener, lifting off once it has gone by
        Drive::Flyby => (if u < 0.72 { 1.0 } else { 0.0 }, true),
    }
}

/// One moment of the engine: its revs, how hard it is working (0..1) and whether a gear change
/// started just now.
#[derive(Clone, Copy, Debug)]
struct Moment {
    rpm: f64,
    load: f64,
    shifted: bool,
}

/// The revs over time: they follow the driver with the engine's inertia, bounce on a limiter and
/// drop at each gear change.
struct Revs {
    drive: Drive,
    seconds: f64,
    redline: f64,
    /// The launch limiter.
    hold: f64,
    rpm: f64,
    load: f64,
    /// Seconds the sparks stay cut (a limiter, a gear change).
    cut: f64,
    gear: usize,
    /// Revs the current gear started from.
    gear_from: f64,
}

/// How long each gear of a launch takes to reach the redline, as a share of the sound.
const GEAR_SHARES: [f64; 3] = [0.15, 0.2, 0.26];

impl Revs {
    fn new(drive: Drive, seconds: f64, redline: f64) -> Revs {
        let rpm = if drive == Drive::Flyby { 0.8 * redline } else { IDLE };
        Revs { drive, seconds: seconds.max(0.05), redline, hold: 0.55 * redline, rpm, load: 0.0, cut: 0.0, gear: 0, gear_from: rpm }
    }

    /// The engine `dt` seconds on, at `t` seconds into the sound.
    fn step(&mut self, t: f64, dt: f64) -> Moment {
        let (throttle, in_gear) = driver(self.drive, t / self.seconds);
        let fuel = if self.cut > 0.0 {
            self.cut -= dt;
            0.0
        } else {
            throttle
        };
        let mut shifted = false;
        if !in_gear {
            let top = if self.drive == Drive::Launch { self.hold } else { self.redline };
            let target = IDLE + fuel * (top + 300.0 - IDLE);
            let tau = if target > self.rpm { 0.11 } else { 0.42 };
            self.rpm += (target - self.rpm) * (dt / tau).min(1.0);
            if self.rpm > top && fuel > 0.0 {
                // the limiter cuts the sparks for a moment: the stutter of a held launch
                self.cut = 0.035;
            }
        } else {
            if self.drive == Drive::Launch && self.gear == 0 {
                // the clutch bites: the revs sag from the limiter, then the car pulls
                self.gear = 1;
                self.rpm = self.hold * 0.82;
                self.gear_from = self.rpm;
            }
            if fuel > 0.0 {
                // a launch's gears each take their share of the sound; a fly-by creeps up
                let share = match self.drive {
                    Drive::Launch => GEAR_SHARES.get(self.gear.saturating_sub(1)).copied().unwrap_or(0.3),
                    _ => 2.5,
                };
                let rate = (self.redline - self.gear_from).max(1.0) / (share * self.seconds).max(0.05);
                self.rpm += rate * dt;
            } else if self.cut <= 0.0 {
                self.rpm += (IDLE * 2.5 - self.rpm) * (dt / 1.2).min(1.0);
            }
            if self.rpm >= self.redline && self.drive == Drive::Launch && self.gear < 3 {
                // a gear change: the sparks cut, the revs drop to the next gear's
                self.gear += 1;
                self.cut = 0.07;
                self.rpm = 0.66 * self.redline;
                self.gear_from = self.rpm;
                shifted = true;
            }
            self.rpm = self.rpm.min(self.redline * 1.02);
        }
        self.rpm = self.rpm.clamp(IDLE * 0.6, self.redline * 1.05);
        let lift = if fuel > 0.0 { 1.0 } else { 0.0 };
        self.load += (lift - self.load) * (dt / if lift > self.load { 0.02 } else { 0.06 }).min(1.0);
        Moment { rpm: self.rpm, load: self.load, shifted }
    }
}

/// Fill `left`/`right` with `drive`.
pub(super) fn engine(drive: Drive, q: &Clamped, left: &mut [f32], right: &mut [f32]) {
    let n = left.len();
    if n == 0 {
        return;
    }
    let sr = q.sr;
    let seconds = n as f64 / sr;
    let e = Engine::pick(q.seed);
    let mut rng = Rng::new(q.seed.wrapping_add(17));
    let mut roar_noise = Pink::new(q.seed.wrapping_add(29));
    // each resonance: its filter, its state and how much of it is heard
    let mut res: Vec<(Coeffs, Biquad, f64)> = e
        .res
        .iter()
        .zip([(1.4, 0.9), (2.2, 0.7), (3.0, 0.35)])
        .map(|(f, (qq, w))| (Coeffs::design(FilterType::BandPass, sr, *f, qq, 0.0), Biquad::default(), w))
        .collect();
    let body_c = Coeffs::design(FilterType::LowPass, sr, 1500.0, 0.7, 0.0);
    let mut body = Biquad::default();
    let dc_c = Coeffs::design(FilterType::HighPass, sr, 35.0, 0.7, 0.0);
    let mut dc = Biquad::default();
    let mut roar = Sweep::new(FilterType::BandPass, 1.6);
    let mut air = Sweep::new(FilterType::LowPass, 0.7);
    let mut wind = Sweep::new(FilterType::BandPass, 0.8);
    let pipe_len = ((e.pipe * sr).round() as usize).clamp(1, 4096);
    let mut pipe = vec![0.0f64; pipe_len];
    let mut pipe_at = 0usize;

    let mut revs = Revs::new(drive, seconds, e.redline);
    let mut crank = 0.0f64; // turns of the crank
    let mut pop = 0.0f64;
    let mut pop_decay = 0.01;
    let mut env = 0.0f64; // the pulses' envelope (drives the intake)
    let drive_gain = 0.8 + 1.5 * q.intensity;
    let crackle = 6.0 + 22.0 * q.intensity; // overrun pops a second at the redline
    let dt = 1.0 / sr;

    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        let t = i as f64 * dt;
        let Moment { rpm, load, shifted } = revs.step(t, dt);
        if shifted && rng.unit() < 0.4 + 0.5 * q.intensity {
            // a gear change bangs
            pop = 1.0;
            pop_decay = 0.012 + 0.01 * rng.unit();
        }

        // the fly-by: distance, Doppler shift, the air dulling it and where it is
        let (doppler, near, pan) = if drive == Drive::Flyby {
            let speed = 55.0 + 15.0 * q.intensity; // m/s
            let x = (t - seconds * 0.5) * speed;
            let d0 = 7.0;
            let dist = (x * x + d0 * d0).sqrt();
            let towards = -x / dist * speed; // closing speed
            (SOUND / (SOUND - towards).max(SOUND * 0.3), d0 / dist, (x / (dist + 3.0)).clamp(-1.0, 1.0))
        } else {
            (1.0, 1.0, 0.0)
        };

        let heard_rpm = rpm * doppler;
        crank += heard_rpm / 60.0 * dt;
        let cycle = wrap(crank * 0.5);
        let cycle_s = 120.0 / heard_rpm.max(1.0);
        let attack = 0.000_35;
        let decay = 0.0014 + 0.0026 * (IDLE / rpm).min(1.0);
        let strength = 0.3 + 0.7 * load;
        let mut pulses = 0.0f64;
        for &(at, gain) in &e.fire {
            let since = wrap(cycle - at) * cycle_s;
            pulses += gain * ((-since / decay).exp() - (-since / attack).exp());
        }
        pulses *= strength;
        let grit = pulses * rng.white() * 0.1;
        env += (pulses.abs() - env) * (dt / 0.004).min(1.0);

        // the exhaust: its resonances, a pipe echo, driven harder with the load
        let x = pulses + grit;
        let mut ex = 0.25 * body.tick(&body_c, x);
        for (c, f, w) in res.iter_mut() {
            ex += *w * f.tick(c, x);
        }
        let echo = pipe.get(pipe_at).copied().unwrap_or(0.0);
        let piped = ex + 0.42 * echo;
        if let Some(p) = pipe.get_mut(pipe_at) {
            *p = piped;
        }
        pipe_at = (pipe_at + 1) % pipe_len;
        let g = 1.0 + drive_gain * load;
        let exhaust = (piped * g).tanh() / g.tanh().max(1e-6);

        // the intake roars with the revs and the load
        let rev = (rpm / e.redline).clamp(0.0, 1.1);
        let roar_f = (700.0 + 0.35 * heard_rpm).min(sr * 0.4);
        let intake = roar.tick(roar_noise.next(), roar_f, sr) * 0.16 * load * rev.powf(1.5) * (0.5 + 2.0 * env).min(1.5);

        // lifting off at high revs: the overrun crackles
        if load < 0.15 && rpm > 2800.0 && rng.unit() < crackle * rev * dt {
            pop = 0.6 + 0.4 * rng.unit();
            pop_decay = 0.006 + 0.01 * rng.unit();
        }
        let crack = if pop > 1e-4 {
            let v = pop * (0.8 * rng.white() + 0.5 * (t * 150.0 * TAU).sin());
            pop *= (-dt / pop_decay).exp();
            v
        } else {
            0.0
        };

        let mut m = exhaust + intake + crack * 0.9;
        m = dc.tick(&dc_c, m);
        if drive == Drive::Flyby {
            // the air takes the top off with distance; the wind rushes as it passes
            let dull = 1200.0 + 15_000.0 * near * near;
            m = air.tick(m, dull.min(sr * 0.45), sr) * near;
            m += wind.tick(rng.white(), 600.0 + 1400.0 * near, sr) * 0.12 * near * near;
            let (gl, gr) = pan_gains(pan);
            *l = (m * gl) as f32;
            *r = (m * gr) as f32;
        } else {
            *l = m as f32;
            *r = (m * 0.97) as f32;
        }
    }
    reverb(left, right, sr, 0.7, 30.0, if drive == Drive::Flyby { 8.0 } else { 12.0 });
    fade_tail(left, right);
}

#[cfg(test)]
mod tests {
    use super::super::{Kind, Params, Stereo, generate};
    use super::*;

    const SR: u32 = 48_000;

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    /// The revs of `drive` sampled at 8 kHz over `seconds`.
    fn trace(drive: Drive, seconds: f64, redline: f64) -> Vec<Moment> {
        let dt = 1.0 / 8000.0;
        let mut revs = Revs::new(drive, seconds, redline);
        (0..(seconds / dt) as usize).map(|i| revs.step(i as f64 * dt, dt)).collect()
    }

    fn at(m: &[Moment], u: f64) -> Moment {
        m[((m.len() as f64 * u) as usize).min(m.len() - 1)]
    }

    #[test]
    fn a_rev_blips_twice_and_falls_back_to_idle() {
        let m = trace(Drive::Rev, 3.0, 7600.0);
        assert!(at(&m, 0.06).rpm < 1200.0, "idles first");
        assert!(at(&m, 0.185).rpm > 0.6 * 7600.0, "first blip: {}", at(&m, 0.185).rpm);
        assert!(at(&m, 0.40).rpm < 0.5 * at(&m, 0.185).rpm, "falls between the blips");
        assert!(at(&m, 0.595).rpm > at(&m, 0.185).rpm, "the second blip is harder");
        assert!(at(&m, 0.99).rpm < 2000.0, "back near idle: {}", at(&m, 0.99).rpm);
        assert!(m.iter().all(|x| x.rpm <= 7600.0 * 1.05));
    }

    #[test]
    fn a_launch_holds_the_limiter_then_changes_gear_twice() {
        let redline = 7600.0;
        let m = trace(Drive::Launch, 4.5, redline);
        // on the limiter: the revs bounce round it, never far past
        let held: Vec<f64> = m[(m.len() as f64 * 0.18) as usize..(m.len() as f64 * 0.29) as usize].iter().map(|x| x.rpm).collect();
        assert!(held.iter().all(|r| (0.45 * redline..0.62 * redline).contains(r)), "held near the limiter");
        assert!(held.iter().cloned().fold(f64::MAX, f64::min) < held.iter().cloned().fold(0.0, f64::max) - 100.0, "it stutters");
        let shifts: Vec<usize> = m.iter().enumerate().filter(|(_, x)| x.shifted).map(|(i, _)| i).collect();
        assert_eq!(shifts.len(), 2, "two gear changes");
        for i in shifts {
            assert!(m[i - 1].rpm >= redline * 0.99 && m[i].rpm < 0.7 * redline, "the revs drop at a gear change");
        }
        assert!(at(&m, 0.94).rpm > 0.8 * redline, "pulling hard in third");
    }

    #[test]
    fn a_rev_is_loud_on_the_throttle_and_quiet_at_idle() {
        let s = generate(&Params::new(Kind::EngineRev), SR).unwrap();
        let n = s.left.len() as f64;
        let w = |a: f64, b: f64| rms(&s.left[(n * a) as usize..(n * b) as usize]);
        assert!(w(0.12, 0.19) > 2.0 * w(0.0, 0.08), "blip {} idle {}", w(0.12, 0.19), w(0.0, 0.08));
    }

    #[test]
    fn a_flyby_crosses_from_left_to_right_and_is_loudest_as_it_passes() {
        let s: Stereo = generate(&Params::new(Kind::EngineFlyby), SR).unwrap();
        let third = s.left.len() / 3;
        assert!(rms(&s.left[..third]) > 1.5 * rms(&s.right[..third]), "starts on the left");
        assert!(rms(&s.right[2 * third..]) > 1.5 * rms(&s.left[2 * third..]), "ends on the right");
        let mid = rms(&s.left[third..2 * third]) + rms(&s.right[third..2 * third]);
        let ends = rms(&s.left[..third / 2]) + rms(&s.right[s.right.len() - third / 2..]);
        assert!(mid > 2.0 * ends, "mid {mid} ends {ends}");
    }

    #[test]
    fn engines_differ_by_seed_and_never_leave_the_range() {
        for kind in [Kind::EngineRev, Kind::EngineLaunch, Kind::EngineFlyby] {
            for seed in [1, 2, 3, 4, 5, 6] {
                let mut p = Params::new(kind);
                p.seed = seed;
                p.intensity = (seed as f64) / 6.0;
                let s = generate(&p, 22_050).unwrap();
                assert!(s.left.iter().chain(&s.right).all(|x| x.is_finite() && x.abs() <= 1.0), "{kind:?} {seed}");
                assert!(rms(&s.left) > 1e-3, "{kind:?} {seed} is heard");
            }
        }
        // a hostile seed or length still makes a bounded sound
        let mut p = Params::new(Kind::EngineLaunch);
        p.seed = u64::MAX;
        p.seconds = 1e9;
        assert_eq!(generate(&p, 8_000).unwrap().left.len(), (30.0 * 8_000.0) as usize);
    }
}

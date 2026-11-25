//! Edit like a reference (PX6.2): `reference` on `autocut.direct` and `autocut.quick`.
//!
//! The reference is a video the person liked — a file, a media item, a sequence of the project —
//! measured by `media.measureStyle` (its fingerprint), or a fingerprint kept from an earlier
//! measure (FrameAtlas keeps them for approved videos). Its numbers become the edit's
//! [`Targets`]: the length, the shot-length distribution and the cut rate, the pace over time
//! (the local shot length at sixteen points) and where its bursts of quick cuts are, the energy
//! curve and the loudness curve. The idea of cutting to a reference's measured rhythm is
//! onetake's (non-commercial): idea only, written fresh.
//!
//! **What steers and what is compared.** Direct cuts every shot to the length the reference has
//! at that point of the video (so its median, spread, cut rate, bursts and energy from cutting
//! follow) and, unless asked otherwise, takes its length and a tempo whose beats fit its shots;
//! Quick Short picks the template whose shots are nearest the reference's and takes its length.
//! The energy from movement and the loudness are compared, not imposed: the shots are the
//! person's own, kept in filming order, and the mix stays at the platform's loudness. Options
//! the video names (`seconds`, `bpm`, `template`) always win over the reference. The report's
//! `reference {measured, targets, applied, kept, achieved, distance}` says what was taken and
//! how close the result came.

use serde_json::{Value, json};

use crate::{AutoCutError, Executor};

/// The points of a fingerprint's curves.
pub const POINTS: usize = 16;
/// The shortest and longest shot a target asks for, seconds.
pub const SHOT_RANGE: (f64, f64) = (0.2, 12.0);
/// Most bursts read from a fingerprint.
const MAX_BURSTS: usize = 16;
/// The largest fingerprint (or reference object) read, bytes of JSON.
pub const MAX_BYTES: usize = 16 << 10;
/// The analysis rate of the measures Direct and Quick Short make.
pub const FPS: f64 = 10.0;

fn bad(m: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(format!("reference: {}", m.into()))
}

fn num(v: &Value, lo: f64, hi: f64) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite() && (lo..=hi).contains(x))
}

/// What an edit aims at, from a reference's fingerprint.
#[derive(Clone, Debug, PartialEq)]
pub struct Targets {
    /// The reference's length, seconds.
    pub seconds: f64,
    /// Its shot lengths, seconds.
    pub median: f64,
    pub p25: f64,
    pub p75: f64,
    pub cv: Option<f64>,
    pub cuts_per_minute: f64,
    /// The local shot length at [`POINTS`] points of the video, seconds.
    pub pace: Vec<f64>,
    /// Its bursts: where they start and end (0..1 of the video) and how many shots each has.
    pub bursts: Vec<(f64, f64, u32)>,
    pub burst_share: f64,
    /// The energy curve (0..1) and the loudness curve (dB from its integrated loudness).
    pub energy: Vec<Option<f64>>,
    pub loudness: Option<Vec<Option<f64>>>,
}

impl Targets {
    /// The targets a fingerprint gives, every number checked (a hostile one is refused).
    pub fn from_fingerprint(fp: &Value) -> Result<Targets, String> {
        if !fp.is_object() {
            return Err("a fingerprint is an object (media.measureStyle's `fingerprint`)".into());
        }
        if fp.to_string().len() > MAX_BYTES {
            return Err(format!("the fingerprint is larger than {} KB", MAX_BYTES >> 10));
        }
        if fp["version"].as_u64() != Some(filmcraft_engine::style_measure::FINGERPRINT_VERSION) {
            return Err(format!("only fingerprints of version {} are read", filmcraft_engine::style_measure::FINGERPRINT_VERSION));
        }
        let seconds = num(&fp["seconds"], 0.1, 86_400.0).ok_or("`seconds` is the reference's length")?;
        let shot = &fp["shot"];
        let median = num(&shot["median"], 0.01, 86_400.0).ok_or("the reference has no shots to measure (`shot.median`)")?;
        let p25 = num(&shot["p25"], 0.01, 86_400.0).unwrap_or(median).min(median);
        let p75 = num(&shot["p75"], 0.01, 86_400.0).unwrap_or(median).max(median);
        let cv = num(&shot["cv"], 0.0, 100.0);
        let cuts_per_minute = num(&fp["cutsPerMinute"], 0.0, 100_000.0).ok_or("`cutsPerMinute`")?;
        let curve = |v: &Value, lo: f64, hi: f64| -> Result<Vec<Option<f64>>, String> {
            match v {
                Value::Null => Ok(vec![None; POINTS]),
                Value::Array(a) if a.len() <= 64 => Ok(a.iter().map(|x| num(x, lo, hi)).collect()),
                _ => Err("a curve is a list of at most 64 numbers".into()),
            }
        };
        let raw_pace = curve(&fp["pace"], 0.01, 86_400.0)?;
        let pace: Vec<f64> = if raw_pace.iter().all(Option::is_none) {
            vec![median; POINTS]
        } else {
            let known: Vec<f64> = raw_pace.iter().flatten().copied().collect();
            let fill = known.iter().sum::<f64>() / known.len().max(1) as f64;
            raw_pace.iter().map(|v| v.unwrap_or(fill)).collect()
        };
        let bursts = match &fp["bursts"]["at"] {
            Value::Null => Vec::new(),
            Value::Array(a) => a
                .iter()
                .take(MAX_BURSTS)
                .filter_map(|b| {
                    let x0 = num(&b[0], 0.0, 1.0)?;
                    let x1 = num(&b[1], 0.0, 1.0)?;
                    let n = b[2].as_u64().filter(|n| (1..=10_000).contains(n))?;
                    (x1 > x0).then_some((x0, x1, n as u32))
                })
                .collect(),
            _ => return Err("`bursts.at` is a list of [from, to, shots]".into()),
        };
        let loudness = match &fp["loudness"] {
            Value::Null => None,
            l if l.is_object() => Some(curve(&l["curve"], -100.0, 100.0)?),
            _ => return Err("`loudness` is {integrated, range, curve} or null".into()),
        };
        Ok(Targets {
            seconds,
            median,
            p25,
            p75,
            cv,
            cuts_per_minute,
            pace,
            bursts,
            burst_share: num(&fp["bursts"]["share"], 0.0, 1.0).unwrap_or(0.0),
            energy: curve(&fp["energy"], 0.0, 1.0)?,
            loudness,
        })
    }

    /// The shot length the reference has at `x` (0..1 of the video), seconds: a burst's own shot
    /// length inside a burst, else its pace there (between two points, the nearer weighs more).
    pub fn shot_at(&self, x: f64) -> f64 {
        let x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
        let secs = if let Some(b) = self.bursts.iter().find(|b| x >= b.0 && x < b.1) {
            (b.1 - b.0) * self.seconds / f64::from(b.2.max(1))
        } else {
            let n = self.pace.len().max(1);
            let pos = (x * n as f64 - 0.5).clamp(0.0, (n - 1) as f64);
            let (i, f) = (pos.floor() as usize, pos - pos.floor());
            let a = self.pace.get(i).copied().unwrap_or(self.median);
            let b = self.pace.get(i + 1).copied().unwrap_or(a);
            a + (b - a) * f
        };
        if secs.is_finite() { secs.clamp(SHOT_RANGE.0, SHOT_RANGE.1) } else { self.median.clamp(SHOT_RANGE.0, SHOT_RANGE.1) }
    }

    /// A tempo (70–150 beats a minute) whose beats fit the reference's quicker shots: its 25th
    /// percentile a whole number of beats, as near the default 110 as that allows.
    pub fn bpm(&self) -> f64 {
        let q = self.p25.clamp(SHOT_RANGE.0, SHOT_RANGE.1);
        let n = (q / (60.0 / 110.0)).round().max(1.0);
        let bpm = (60.0 * n / q).clamp(70.0, 150.0);
        (bpm * 10.0).round() / 10.0
    }

    pub fn json(&self) -> Value {
        json!({
            "seconds": self.seconds,
            "shot": {"median": self.median, "p25": self.p25, "p75": self.p75, "cv": self.cv},
            "cutsPerMinute": self.cuts_per_minute,
            "pace": self.pace,
            "bursts": {"share": self.burst_share, "at": self.bursts.iter().map(|b| json!([b.0, b.1, b.2])).collect::<Vec<_>>()},
            "energy": self.energy,
            "loudness": self.loudness,
            "steers": ["shot lengths (pace and bursts)", "cut rate", "length unless asked", "tempo unless asked or a track is given"],
            "compared": ["energy from movement", "loudness (the mix stays at the platform's level)"],
        })
    }
}

/// A reference as an edit uses it: what was given, its measure and its targets.
#[derive(Clone, Debug)]
pub struct Reference {
    pub given: Value,
    pub measured: Value,
    pub targets: Targets,
}

/// The parameters `media.measureStyle` takes for a reference (an item id, a file, or `None`
/// for a fingerprint given as it is).
pub fn measure_params(v: &Value) -> Result<Option<Value>, AutoCutError> {
    match v {
        Value::Number(n) => n.as_u64().map(|i| Some(json!({"item": i, "fps": FPS, "summary": true}))).ok_or_else(|| bad("an item id is a whole number")),
        Value::String(s) if !s.trim().is_empty() && s.len() <= 4096 => Ok(Some(json!({"path": s.trim(), "fps": FPS, "summary": true}))),
        Value::Object(_) => Ok(None),
        _ => Err(bad("a media item or sequence id, a video file, or a fingerprint ({fingerprint} from media.measureStyle)")),
    }
}

/// Measure (or read) the reference `v`.
pub async fn load(ex: &dyn Executor, v: &Value) -> Result<Reference, AutoCutError> {
    let measured = match measure_params(v)? {
        Some(p) => {
            let r = ex.execute("media.measureStyle", p).await.map_err(|e| bad(format!("it could not be measured: {e}")))?;
            r["fingerprint"].clone()
        }
        None => {
            if v.to_string().len() > MAX_BYTES {
                return Err(bad(format!("a fingerprint is at most {} KB", MAX_BYTES >> 10)));
            }
            if v["fingerprint"].is_object() { v["fingerprint"].clone() } else { v.clone() }
        }
    };
    let targets = Targets::from_fingerprint(&measured).map_err(bad)?;
    let given = match v {
        Value::Object(_) => json!({"fingerprint": measured["key"]}),
        other => other.clone(),
    };
    Ok(Reference { given, measured, targets })
}

/// How far `achieved` (a fingerprint, or the rhythm `rhythm_of` gives) is from the reference's
/// `target` fingerprint: each part 0 (the same) … 1 (twice or half, or a curve far off), and the
/// weighted whole.
pub fn distance(target: &Value, achieved: &Value) -> Value {
    let ratio = |a: Option<f64>, t: Option<f64>| -> Option<f64> {
        let (a, t) = (a?, t?);
        (a > 0.0 && t > 0.0).then(|| ((a / t).ln().abs() / std::f64::consts::LN_2).min(1.0))
    };
    let f = |v: &Value| v.as_f64().filter(|x| x.is_finite());
    let curve = |a: &Value, t: &Value, scale: f64, log: bool| -> Option<f64> {
        let (a, t) = (a.as_array()?, t.as_array()?);
        let d: Vec<f64> = a
            .iter()
            .zip(t)
            .filter_map(|(x, y)| {
                let (x, y) = (f(x)?, f(y)?);
                if log { (x > 0.0 && y > 0.0).then(|| ((x / y).ln().abs() / std::f64::consts::LN_2).min(1.0)) } else { Some(((x - y).abs() / scale).min(1.0)) }
            })
            .collect();
        (!d.is_empty()).then(|| d.iter().sum::<f64>() / d.len() as f64)
    };
    let parts = [
        ("shotLength", 0.25, ratio(f(&achieved["shot"]["median"]), f(&target["shot"]["median"]))),
        ("cutRate", 0.2, ratio(f(&achieved["cutsPerMinute"]).map(|x| x + 1.0), f(&target["cutsPerMinute"]).map(|x| x + 1.0))),
        ("pace", 0.2, curve(&achieved["pace"], &target["pace"], 1.0, true)),
        (
            "spread",
            0.1,
            match (f(&achieved["shot"]["cv"]), f(&target["shot"]["cv"])) {
                (Some(a), Some(t)) => Some((a - t).abs().min(1.0)),
                _ => None,
            },
        ),
        (
            "bursts",
            0.1,
            match (f(&achieved["bursts"]["share"]), f(&target["bursts"]["share"])) {
                (Some(a), Some(t)) => Some(((a - t).abs() * 2.0).min(1.0)),
                _ => None,
            },
        ),
        ("energy", 0.1, curve(&achieved["energy"], &target["energy"], 0.3, false)),
        ("loudness", 0.05, curve(&achieved["loudness"]["curve"], &target["loudness"]["curve"], 6.0, false)),
    ];
    let mut out = serde_json::Map::new();
    let (mut sum, mut weight) = (0.0, 0.0);
    for (name, w, d) in parts {
        out.insert(name.into(), d.map_or(Value::Null, |d| json!((d * 1000.0).round() / 1000.0)));
        if let Some(d) = d {
            sum += w * d;
            weight += w;
        }
    }
    let overall = (weight > 0.0).then(|| ((sum / weight) * 1000.0).round() / 1000.0);
    json!({"overall": overall, "parts": out, "scale": "0 the same … 1 twice or half as fast, or a curve far off"})
}

/// The reference part of a report: what was given, measured and aimed at, what of it was taken
/// and kept, and what came out.
pub fn report(r: &Reference, applied: &[String], kept: &[String], achieved: &Value) -> Value {
    json!({
        "given": r.given,
        "measured": r.measured,
        "targets": r.targets.json(),
        "applied": applied,
        "kept": kept,
        "achieved": achieved,
        "distance": distance(&r.measured, if achieved["fingerprint"].is_object() { &achieved["fingerprint"] } else { &achieved["timeline"] }),
    })
}

/// What an edit came out as: its sequence measured as the reference was (`fingerprint`, from the
/// render and the mix), and its own cut points (`timeline`, the shots as placed).
pub async fn achieved(ex: &dyn Executor, seq: Option<u64>, shots: &[(f64, f64)], seconds: f64) -> Value {
    let timeline = filmcraft_engine::style_measure::rhythm_of(shots, seconds);
    let Some(seq) = seq else { return json!({"from": "plan", "timeline": timeline}) };
    match ex.execute("media.measureStyle", json!({"sequence": seq, "fps": FPS, "summary": true})).await {
        Ok(v) => json!({"from": "render", "fingerprint": v["fingerprint"], "timeline": timeline}),
        Err(e) => json!({"from": "plan", "timeline": timeline, "note": format!("the edit could not be measured: {e}")}),
    }
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;

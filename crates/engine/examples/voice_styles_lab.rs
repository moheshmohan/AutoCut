//! The voice-style lab (measurement only): find stretches of a speaker's own Malayalam to use as
//! style references, and measure readings of a script in each voice style — pitch range, speaking
//! rate and the strict check's character error rates. The numbers in
//! `docs/malayalam-narration.md` ("Voice styles") come from here.
//!
//! ```text
//! # phrases of the speaker's clips (whisper-medium-ml), and the liveliest / calmest stretches
//! voice_styles_lab scan <models dir> <media>…
//! # read lines of a showcase script in readings (FILMCRAFT_DATA_DIR = a copy of the data dir)
//! FILMCRAFT_DATA_DIR=<copy> voice_styles_lab read <voice> <script.json> <out dir> [lines] [reading,…]
//! # one WAV's pitch range and rate
//! voice_styles_lab measure <wav> <words>
//! # one engine command (a style reference on the copy: tts.voiceStyles)
//! FILMCRAFT_DATA_DIR=<copy> voice_styles_lab exec tts.voiceStyles '{"voice": …, "set": "excited", …}'
//! ```
//!
//! A reading is `name=delivery[/t<temperature>][/e<emphasis>][/s<speed>]` (`excited`,
//! `plain/t0.7`, `calm/t1/e2/s1` — the calm reference with the model's own sampler); without
//! readings the lab reads the set in the docs. Each reading of a line is written as
//! `<out dir>/<reading>/line<k>.wav`, its numbers to `<out dir>/results.jsonl` and a table to
//! standard output. It reads through the engine (`tts.preview`, the strict check on), so it takes
//! the GPU lock itself.

use std::path::{Path, PathBuf};

use filmcraft_engine::{FsServices, MediaPool, Session};
use filmcraft_tts as tts;
use serde_json::{Value, json};

type Res<T> = Result<T, String>;

/// Pitch range, rate and loudness of speech.
#[derive(Clone, Debug, Default)]
struct Measure {
    /// 10th, 50th and 90th percentile of the voiced frames' pitch, Hz.
    p10: f64,
    p50: f64,
    p90: f64,
    /// 12·log2(p90 / p10): the pitch range, semitones.
    range: f64,
    /// Syllables a second of speech (silences of 150 ms and more left out), and a second of the
    /// whole reading.
    rate: f64,
    rate_overall: f64,
    /// Seconds of speech, and in all.
    speech: f64,
    seconds: f64,
    /// Spread of the 50 ms loudness of speech frames (dB, standard deviation): how much the
    /// voice rises and falls in strength.
    loud_sd: f64,
}

fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return 0.0;
    }
    s.sort_by(f64::total_cmp);
    let i = ((s.len() - 1) as f64 * q).round() as usize;
    s.get(i).copied().unwrap_or(0.0)
}

fn measure(x: &[f32], rate: u32, words: &str) -> Measure {
    let f0 = tts::clone::pitch_frames(x, rate);
    let (p10, p50, p90) = (percentile(&f0, 0.1), percentile(&f0, 0.5), percentile(&f0, 0.9));
    let range = if p10 > 0.0 && p90 > 0.0 { 12.0 * (p90 / p10).log2() } else { 0.0 };
    // speech: 50 ms frames within 35 dB of the loudest; silences of three frames and more are pauses
    let n = (rate as usize / 20).max(1);
    let db: Vec<f64> =
        x.chunks(n).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / c.len().max(1) as f64 + 1e-12).log10()).collect();
    let peak = db.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let loud: Vec<bool> = db.iter().map(|d| *d > peak - 35.0).collect();
    let mut speech_frames = 0usize;
    let mut run = 0usize;
    let (first, last) = (loud.iter().position(|l| *l).unwrap_or(0), loud.iter().rposition(|l| *l).unwrap_or(0));
    for l in loud.iter().take(last + 1).skip(first) {
        if *l {
            speech_frames += 1 + if run < 3 { run } else { 0 };
            run = 0;
        } else {
            run += 1;
        }
    }
    let speech = speech_frames as f64 * 0.05;
    let seconds = x.len() as f64 / f64::from(rate.max(1));
    let mut syl = tts::malayalam::syllables(&tts::malayalam::normalize(&tts::script::strip_marks(words)).text) as f64;
    if syl == 0.0 {
        // no words: syllable nuclei, the loudness peaks of 25 ms frames at least 100 ms apart that
        // stand 3 dB over the troughs either side
        let n = (rate as usize / 40).max(1);
        let e: Vec<f64> =
            x.chunks(n).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / c.len().max(1) as f64 + 1e-12).log10()).collect();
        let top = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut last = 0usize;
        for i in 1..e.len().saturating_sub(1) {
            let (Some(p), Some(c), Some(q)) = (e.get(i - 1), e.get(i), e.get(i + 1)) else { continue };
            let trough = |r: std::ops::Range<usize>| e.get(r).unwrap_or(&[]).iter().copied().fold(f64::INFINITY, f64::min);
            let (l, r) = (trough(i.saturating_sub(6)..i), trough(i + 1..(i + 7).min(e.len())));
            if c >= p && c > q && *c > top - 30.0 && c - l.max(r) >= 3.0 && (last == 0 || i - last >= 4) {
                syl += 1.0;
                last = i;
            }
        }
    }
    let voiced: Vec<f64> = db.iter().zip(&loud).filter(|(_, l)| **l).map(|(d, _)| *d).collect();
    let mean = voiced.iter().sum::<f64>() / voiced.len().max(1) as f64;
    let loud_sd = (voiced.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / voiced.len().max(1) as f64).sqrt();
    Measure { p10, p50, p90, range, rate: syl / speech.max(0.1), rate_overall: syl / seconds.max(0.1), speech, seconds, loud_sd }
}

fn measure_json(m: &Measure) -> Value {
    json!({"pitchP10": m.p10, "pitchP50": m.p50, "pitchP90": m.p90, "rangeSemitones": m.range, "syllablesPerSecond": m.rate,
           "syllablesPerSecondOverall": m.rate_overall, "speechSeconds": m.speech, "seconds": m.seconds, "loudnessSd": m.loud_sd})
}

/// Mono audio of `path` at `rate` (at most `max` seconds, never past its end).
fn decode(path: &str, rate: u32, max: f64) -> Res<Vec<f32>> {
    let src = MediaPool::default().open_file(path, &FsServices).map_err(|e| format!("{path}: {e}"))?;
    let total = (max.min(src.info().duration.seconds()) * f64::from(rate)) as usize;
    let mut out = Vec::new();
    while out.len() < total {
        let n = (10 * rate as usize).min(total - out.len());
        let buf = src.audio(out.len() as i64, n, rate).map_err(|e| e.to_string())?;
        let mono = filmcraft_speech::downmix(&buf.channels);
        if mono.is_empty() {
            break;
        }
        let got = mono.len().min(n);
        out.extend(mono.iter().take(got).map(|v| if v.is_finite() { *v } else { 0.0 }));
        if got < n {
            break;
        }
    }
    Ok(out)
}

/// `scan`: the speech phrases of each clip (an energy gate, pauses of 0.35 s and more between
/// them), and the stretches of 7–14 s of whole phrases (no pause inside over 1.5 s), measured and
/// ranked liveliest and calmest; the best few heard by whisper-medium-ml.
fn scan(args: &[String]) -> Res<()> {
    let (Some(models), clips) = (args.first(), args.get(1..).unwrap_or(&[])) else { return Err("scan <models dir> <media>…".into()) };
    let asr = filmcraft_speech::load_on(Path::new(models), filmcraft_speech::models::MALAYALAM_MODEL, true).map_err(|e| e.to_string())?;
    let mut stretches: Vec<Value> = Vec::new();
    let mut audio: Vec<(String, Vec<f32>)> = Vec::new();
    let vad = filmcraft_speech::silero::SegmentParams { min_silence: 0.35, pad: 0.05, ..Default::default() };
    for clip in clips {
        let x = decode(clip, 16_000, 600.0)?;
        let name = Path::new(clip).file_name().map_or_else(|| clip.clone(), |n| n.to_string_lossy().into_owned());
        let (segs, _) = filmcraft_speech::vad::speech_segments(&x, None, &vad);
        let phrases: Vec<(f64, f64)> = segs.iter().map(|r| (r.start as f64 / 16_000.0, r.end as f64 / 16_000.0)).filter(|(a, b)| b - a >= 0.4).collect();
        println!("\n{name}: {:.1} s, {} phrases", x.len() as f64 / 16_000.0, phrases.len());
        let cut = |a: f64, b: f64| x.get((a * 16_000.0) as usize..((b * 16_000.0) as usize).min(x.len())).unwrap_or(&[]).to_vec();
        // stretches of whole phrases, 7–14 s
        for i in 0..phrases.len() {
            for j in i..phrases.len() {
                let (Some(p), Some(q)) = (phrases.get(i), phrases.get(j)) else { continue };
                let (a, b) = (p.0, q.1);
                if b - a > 14.0 || (j > i && phrases.get(j - 1).is_some_and(|r| q.0 - r.1 > 1.5)) {
                    break;
                }
                if b - a < 7.0 {
                    continue;
                }
                // the rate needs words: estimated at the speaker's ordinary 5 syllables a second of
                // voiced speech; the pitch and loudness do not
                let m = measure(&cut(a, b), 16_000, "");
                stretches.push(json!({"clip": clip, "name": name, "start": a, "end": b, "phrases": j - i + 1, "measure": measure_json(&m)}));
            }
        }
        audio.push((clip.clone(), x));
    }
    // liveliest: wide pitch range, quick, strong rises and falls, pitched up; calmest: the opposite
    let z = |key: &str| -> (f64, f64) {
        let v: Vec<f64> = stretches.iter().filter_map(|s| s["measure"][key].as_f64()).collect();
        let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len().max(1) as f64).sqrt().max(1e-9);
        (mean, sd)
    };
    let (zr, zs, zl, zp) = (z("rangeSemitones"), z("syllablesPerSecond"), z("loudnessSd"), z("pitchP50"));
    let score = |s: &Value| {
        let g = |k: &str, (m, sd): (f64, f64)| (s["measure"][k].as_f64().unwrap_or(m) - m) / sd;
        g("rangeSemitones", zr) + g("syllablesPerSecond", zs) + 0.5 * g("loudnessSd", zl) + g("pitchP50", zp)
    };
    let mut ranked: Vec<(f64, &Value)> = stretches.iter().map(|s| (score(s), s)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    // the best of each, never two overlapping stretches of one clip, heard
    let distinct = |list: Vec<&(f64, &Value)>| {
        let mut kept: Vec<(f64, Value)> = Vec::new();
        for (sc, s) in list {
            let over = kept.iter().any(|(_, k)| {
                k["clip"] == s["clip"]
                    && k["start"].as_f64().unwrap_or(0.0) < s["end"].as_f64().unwrap_or(0.0)
                    && s["start"].as_f64().unwrap_or(0.0) < k["end"].as_f64().unwrap_or(0.0)
            });
            if !over {
                kept.push((*sc, (*s).clone()));
            }
            if kept.len() >= 6 {
                break;
            }
        }
        kept
    };
    for (title, list) in [("liveliest", distinct(ranked.iter().collect())), ("calmest", distinct(ranked.iter().rev().collect()))] {
        println!("\n{title}:");
        for (sc, s) in list {
            let m = &s["measure"];
            let (a, b) = (s["start"].as_f64().unwrap_or(0.0), s["end"].as_f64().unwrap_or(0.0));
            let x = audio.iter().find(|(c, _)| s["clip"].as_str() == Some(c.as_str())).map(|(_, x)| x.as_slice()).unwrap_or(&[]);
            let w = x.get((a * 16_000.0) as usize..((b * 16_000.0) as usize).min(x.len())).unwrap_or(&[]);
            let heard = asr.hear(w, "ml").unwrap_or_default();
            println!(
                "{sc:+5.2}  {} {a:.2}–{b:.2}  {:.1} st  {:.1} peaks/s  p50 {:.0} Hz  sd {:.1} dB  {heard}",
                s["name"].as_str().unwrap_or(""),
                m["rangeSemitones"].as_f64().unwrap_or(0.0),
                m["syllablesPerSecond"].as_f64().unwrap_or(0.0),
                m["pitchP50"].as_f64().unwrap_or(0.0),
                m["loudnessSd"].as_f64().unwrap_or(0.0),
            );
            println!("        {}", s["clip"].as_str().unwrap_or(""));
        }
    }
    Ok(())
}

/// The readings measured when none are named.
const READINGS: [&str; 9] = [
    "main=plain",
    "excited-ref=excited/t1/e2/s1",
    "excited=excited",
    "calm-ref=calm/t1/e2/s1",
    "calm=calm",
    "main-t0.7=plain/t0.7",
    "main-t1.3=plain/t1.3",
    "main-e1.6=plain/e1.6",
    "main-e2.6=plain/e2.6",
];

/// `name=delivery/t…/e…/s…` → its name and the `tts.preview` params it adds.
fn reading(spec: &str) -> Res<(String, Value)> {
    let (name, rest) = spec.split_once('=').ok_or_else(|| format!("`{spec}`: name=delivery[/t…][/e…][/s…]"))?;
    let mut parts = rest.split('/');
    let mut p = json!({"delivery": parts.next().unwrap_or("plain")});
    for part in parts {
        let num = |k: &str| part.strip_prefix(k).and_then(|v| v.parse::<f64>().ok()).ok_or_else(|| format!("`{part}` in `{spec}`"));
        match part.chars().next() {
            Some('t') => p["temperature"] = json!(num("t")?),
            Some('e') => p["emphasis"] = json!(num("e")?),
            Some('s') => p["speed"] = json!(num("s")?),
            _ => return Err(format!("`{part}` in `{spec}`")),
        }
    }
    Ok((name.to_string(), p))
}

/// `read`: lines of a showcase script read in each reading, through the engine.
fn read(args: &[String]) -> Res<()> {
    let [voice, script, out, rest @ ..] = args else { return Err("read <voice> <script.json> <out dir> [lines] [reading,…]".into()) };
    let count: usize = rest.first().and_then(|n| n.parse().ok()).unwrap_or(5);
    let specs: Vec<String> = match rest.get(1) {
        Some(r) => r.split(',').map(str::to_string).collect(),
        None => READINGS.iter().map(|s| s.to_string()).collect(),
    };
    let d: Value = serde_json::from_slice(&std::fs::read(script).map_err(|e| format!("{script}: {e}"))?).map_err(|e| e.to_string())?;
    let mut lines: Vec<(String, String)> = match &d["lines"] {
        Value::Object(m) => m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect(),
        Value::Array(a) => a.iter().enumerate().filter_map(|(k, v)| Some(((k + 1).to_string(), v.as_str()?.to_string()))).collect(),
        _ => Vec::new(),
    };
    lines.retain(|(_, l)| tts::malayalam::detect(l));
    lines.truncate(count);
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut log = String::new();
    let mut s = Session::default();
    println!("| reading | lines | pitch range p10–p90 (st) | syllables/s | first-take CER | kept CER | readings | seconds |");
    println!("|---|---|---|---|---|---|---|---|");
    for spec in &specs {
        let (name, extra) = reading(spec)?;
        let dir = out.join(&name);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let (mut ranges, mut rates, mut first, mut kept, mut takes, mut secs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0u64, 0.0);
        for (k, (key, line)) in lines.iter().enumerate() {
            let path = dir.join(format!("line{}.wav", k + 1));
            let mut p = json!({"text": line, "voice": voice, "language": "ml-IN", "check": "strict", "path": path.to_string_lossy()});
            if let (Some(o), Some(e)) = (p.as_object_mut(), extra.as_object()) {
                o.extend(e.clone());
            }
            let t0 = std::time::Instant::now();
            let r = s.execute("tts.preview", p).map_err(|e| format!("{name} line {}: {e}", k + 1))?;
            let a = s.tts_preview.clone().ok_or("no preview audio")?;
            let m = measure(&a.samples, a.sample_rate, line);
            let utt = r["check"]["utterances"].as_array().cloned().unwrap_or_default();
            let mean = |f: &dyn Fn(&Value) -> Option<f64>| {
                let v: Vec<f64> = utt.iter().filter_map(f).collect();
                v.iter().sum::<f64>() / v.len().max(1) as f64
            };
            let c_first = mean(&|u: &Value| u["tries"][0]["cer"].as_f64());
            let c_kept = mean(&|u: &Value| u["cer"].as_f64());
            let t: u64 = utt.iter().filter_map(|u| u["takes"].as_u64()).sum();
            eprintln!(
                "{name} line {} ({key}): {:.1} st, {:.2} syl/s, CER {c_first:.3} → {c_kept:.3} in {t} readings, {:.0} s",
                k + 1,
                m.range,
                m.rate,
                t0.elapsed().as_secs_f64()
            );
            let row = json!({"reading": name, "spec": spec, "line": k + 1, "key": key, "text": line, "measure": measure_json(&m),
                             "firstCer": c_first, "keptCer": c_kept, "takes": t, "check": r["check"], "wav": path.to_string_lossy()});
            log.push_str(&format!("{row}\n"));
            ranges.push(m.range);
            rates.push(m.rate);
            first.push(c_first);
            kept.push(c_kept);
            takes += t;
            secs += m.seconds;
        }
        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!(
            "| {name} (`{spec}`) | {} | {:.1} | {:.2} | {:.3} | {:.3} | {takes} | {secs:.1} |",
            ranges.len(),
            avg(&ranges),
            avg(&rates),
            avg(&first),
            avg(&kept)
        );
        std::fs::write(out.join("results.jsonl"), &log).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("scan") => scan(rest),
        Some("read") => read(rest),
        // one engine command (`tts.voiceStyles …` on the data directory's voices)
        Some("exec") => {
            let (Some(id), Some(params)) = (rest.first(), rest.get(1)) else { return Err("exec <command> <json params>".into()) };
            let p: Value = serde_json::from_str(params).map_err(|e| e.to_string())?;
            let r = Session::default().execute(id, p).map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?);
            Ok(())
        }
        Some("measure") => {
            let (Some(wav), Some(words)) = (rest.first(), rest.get(1)) else { return Err("measure <wav> <words>".into()) };
            let (x, rate) = tts::wav::read(Path::new(wav)).map_err(|e| e.to_string())?;
            println!("{}", measure_json(&measure(&x, rate, words)));
            Ok(())
        }
        _ => Err("usage: voice_styles_lab scan|read|measure … (see the source)".into()),
    }
}

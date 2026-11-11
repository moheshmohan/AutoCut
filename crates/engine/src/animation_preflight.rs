//! Low-resolution motion, declared typography and audible cue checks for external compositions.
//! A sampled motion check cannot establish semantic choreography or read clipped text: actual
//! text geometry must be supplied by the composition. Missing evidence is reported explicitly.
use crate::commands::{CommandSpec, always, bad};
use crate::{Result, Session};
use filmcraft_media::{FrameRequest, SharedSource};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};
const CMD: &str = "media.preflightAnimation";
/// Decoders may ignore the scale hint. Retain only bounded thumbnails, never hundreds
/// of full-resolution RGBA pictures during a long preflight.
pub(crate) fn motion_pixels(width: u32, height: u32, rgba: Vec<u8>) -> Result<Vec<u8>> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || pixels > 16_777_216 || pixels * 4 != rgba.len() as u64 {
        return Err(bad(CMD, "invalid or oversized decoded picture"));
    }
    if width.max(height) <= 160 {
        return Ok(rgba);
    }
    let side = width.max(height);
    let w = (u64::from(width) * 160 / u64::from(side)).max(1) as u32;
    let h = (u64::from(height) * 160 / u64::from(side)).max(1) as u32;
    let mut reduced = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let sy = u64::from(y) * u64::from(height) / u64::from(h);
        for x in 0..w {
            let sx = u64::from(x) * u64::from(width) / u64::from(w);
            let offset = ((sy * u64::from(width) + sx) * 4) as usize;
            reduced.extend_from_slice(rgba.get(offset..offset + 4).ok_or_else(|| bad(CMD, "invalid decoded picture buffer"))?);
        }
    }
    Ok(reduced)
}
pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Preflight animation",
        menu: &[],
        shortcut: None,
        params: r#"{"path":path,"fps":1..240?,"expected":{"duration":ticks?,"width":u32?,"height":u32?,"fps":f64?,"alpha":bool?},"holds":[{"start":ticks,"end":ticks,"reason":str}]?,"textBounds":[{"name":str,"start":ticks,"end":ticks,"box":[x,y,w,h]}]?,"soundCues":[{"name":str,"time":ticks}]?,"sheet":path?,"expectMotion":bool=true}"#,
        enabled: always,
        run: preflight,
        journal: false,
    }]
}
fn spans(p: &Value, key: &str, end: Tick) -> Result<Vec<(i64, i64)>> {
    let Some(values) = p.get(key) else { return Ok(Vec::new()) };
    let values = values.as_array().filter(|a| a.len() <= 2000).ok_or_else(|| bad(CMD, format!("{key} needs up to 2000 intervals")))?;
    values
        .iter()
        .map(|v| {
            let a = v["start"].as_i64().filter(|a| *a >= 0).ok_or_else(|| bad(CMD, "interval start is ticks"))?;
            let b = v["end"].as_i64().filter(|b| *b > a && *b <= end.0).ok_or_else(|| bad(CMD, "interval end must follow start within the animation"))?;
            if v["reason"].as_str().is_none_or(|s| s.is_empty() || s.len() > 512) {
                return Err(bad(CMD, "intentional hold needs a reason"));
            }
            Ok((a, b))
        })
        .collect()
}
pub(crate) fn frozen(times: &[(Tick, Vec<u8>)], holds: &[(i64, i64)], minimum: Tick) -> Vec<Value> {
    let mut findings = Vec::new();
    let mut run = None;
    for pair in times.windows(2) {
        let Some([(a, first), (b, second)]) = pair.get(..) else { continue };
        let same = first.len() == second.len()
            && !first.is_empty()
            && first
                .as_chunks::<4>()
                .0
                .iter()
                .zip(second.as_chunks::<4>().0)
                .map(|(a, b)| {
                    let (aa, ba) = (f64::from(a[3]) / 255.0, f64::from(b[3]) / 255.0);
                    (f64::from(a[0]) * aa - f64::from(b[0]) * ba).abs()
                        + (f64::from(a[1]) * aa - f64::from(b[1]) * ba).abs()
                        + (f64::from(a[2]) * aa - f64::from(b[2]) * ba).abs()
                        + f64::from(a[3].abs_diff(b[3]))
                })
                .sum::<f64>()
                / (first.len() as f64 * 255.0)
                < 0.0015;
        let held = holds.iter().any(|(start, end)| *start <= a.0 && *end >= b.0);
        if same && !held {
            if run.is_none() {
                run = Some(*a);
            }
        } else if let Some(start) = run.take()
            && a.0.saturating_sub(start.0) >= minimum.0
        {
            findings.push(json!({"code":"frozen","start":start.0,"end":a.0,"seconds":start.seconds()}));
        }
    }
    if let (Some(start), Some((end, _))) = (run, times.last())
        && end.0.saturating_sub(start.0) >= minimum.0
    {
        findings.push(json!({"code":"frozen","start":start.0,"end":end.0,"seconds":start.seconds()}));
    }
    findings
}
fn preflight(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p["path"].as_str().filter(|p| !p.is_empty() && p.len() <= 4096).ok_or_else(|| bad(CMD, "provide render path"))?;
    let source: SharedSource = match p.get("fps") {
        Some(v) => {
            let fps = v.as_f64().filter(|v| v.is_finite() && (1.0..=240.0).contains(v)).ok_or_else(|| bad(CMD, "image-sequence fps must be 1..240"))?;
            crate::media_pool::open_image_sequence(path, s.services.as_ref(), FrameRate::from_f64(fps), "Animation").map_err(|e| bad(CMD, e.to_string()))?
        }
        None => s.media.open_file(path, s.services.as_ref()).map_err(|e| bad(CMD, e.to_string()))?,
    };
    let info = source.info();
    let video = info.video.as_ref().ok_or_else(|| bad(CMD, "render has no picture"))?;
    let end = info.duration;
    if end.0 <= 0 || end.seconds() > 12.0 * 3600.0 {
        return Err(bad(CMD, "animation duration must be positive and at most 12 hours"));
    }
    let holds = spans(p, "holds", end)?;
    let expected = &p["expected"];
    let mut findings = Vec::new();
    let mut skipped = Vec::new();
    let mut issue = |code: &str, detail: Value| findings.push(json!({"code":code,"detail":detail,"time":0}));
    if let Some(duration) = expected["duration"].as_i64()
        && (end.0 as i128 - duration as i128).abs() > i128::from(video.frame_rate.frame_duration().0)
    {
        issue("duration", json!({"expected":duration,"actual":end.0}));
    }
    for (name, actual) in [("width", video.width), ("height", video.height)] {
        if let Some(want) = expected[name].as_u64()
            && want != u64::from(actual)
        {
            issue(name, json!({"expected":want,"actual":actual}));
        }
    }
    if let Some(fps) = expected["fps"].as_f64()
        && (!fps.is_finite() || fps <= 0.0 || (fps - video.frame_rate.num as f64 / video.frame_rate.den.max(1) as f64).abs() > 0.001)
    {
        issue("frameRate", json!({"expected":fps,"actual":video.frame_rate}));
    }
    if expected["alpha"] == true && !video.has_alpha {
        issue("alpha", json!("renderer did not deliver an alpha channel"));
    }
    let count = ((end.seconds() * 10.0).ceil() as usize).clamp(2, 600);
    let scale = (160.0 / video.width.max(video.height).max(1) as f32).min(1.0);
    let mut frames = Vec::new();
    let mut transparency = false;
    for i in 0..count {
        let time = Tick(((end.0 - 1) as i128 * i as i128 / (count - 1) as i128) as i64);
        match source.video_frame(FrameRequest { time, scale }) {
            Ok(f) => {
                if u64::from(f.width) * u64::from(f.height) > 16_777_216 {
                    issue("oversizeDecode", json!(time.0));
                    continue;
                }
                let rgba = f.to_rgba8();
                transparency |= rgba.as_chunks::<4>().0.iter().any(|p| p[3] < 250);
                frames.push((time, motion_pixels(f.width, f.height, rgba)?));
            }
            Err(e) => issue("decode", json!({"time":time.0,"error":e.to_string()})),
        }
    }
    if expected["alpha"] == true && !transparency {
        issue("opaqueAlpha", json!("all sampled pixels are opaque; check the transparent composition"));
    }
    drop(issue);
    if p.get("expectMotion").and_then(Value::as_bool).unwrap_or(true) {
        findings.extend(frozen(&frames, &holds, Tick::from_seconds_f64(1.0)));
    }
    if let Some(bounds) = p.get("textBounds") {
        for text in bounds.as_array().filter(|a| a.len() <= 2000).ok_or_else(|| bad(CMD, "textBounds needs up to 2000 elements"))? {
            let rect: Option<[f64; 4]> = serde_json::from_value(text["box"].clone()).ok();
            let valid = rect.is_some_and(|r| {
                let [x, y, w, h] = r;
                r.iter().all(|v| v.is_finite()) && x >= 0.0 && y >= 0.0 && w > 0.0 && h > 0.0 && x + w <= 1.0 && y + h <= 1.0
            });
            if !valid {
                findings.push(json!({"code":"textBounds","time":text["start"],"name":text["name"],"box":text["box"]}));
            }
        }
    } else {
        skipped.push(json!({"check":"textBounds","reason":"composition supplied no text geometry; review the contact sheet"}));
    }
    if let Some(cues) = p.get("soundCues") {
        for cue in cues.as_array().filter(|a| a.len() <= 2000).ok_or_else(|| bad(CMD, "soundCues needs up to 2000 cues"))? {
            let time = cue["time"].as_i64().filter(|t| *t >= 0 && *t < end.0).ok_or_else(|| bad(CMD, "sound cue outside animation"))?;
            let audio = source.audio(Tick(time).to_units_floor(48000), 4800, 48000);
            let audible = audio.ok().is_some_and(|b| b.channels.iter().flatten().any(|v| v.is_finite() && v.abs() > 0.001));
            if !audible {
                findings.push(json!({"code":"soundCue","time":time,"name":cue["name"],"reason":"no audible signal in the next 100 ms"}));
            }
        }
    } else {
        skipped.push(json!({"check":"soundCues","reason":"no cue timings supplied"}));
    }
    let sheet = if let Some(path_out) = p["sheet"].as_str() {
        let mut isolated = Session::new(s.services.clone());
        let item = if let Some(fps) = p.get("fps") {
            isolated.execute("file.importImageSequence", json!({"path":path,"fps":fps}))?["items"][0].clone()
        } else {
            json!(crate::commands::import_streamed(&mut isolated, path, None)?.0)
        };
        Some(isolated.execute("media.contactSheet", json!({"items":[{"item":item}],"n":12,"side":240,"columns":4,"path":path_out}))?)
    } else {
        None
    };
    Ok(
        json!({"passed":findings.is_empty(),"findings":findings,"skipped":skipped,"duration":end.0,"video":video,"samples":frames.len(),"sampled":true,"sheet":sheet,"holds":p.get("holds"),"textMethod":"declared composition geometry"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ignored_decoder_scale_cannot_retain_large_motion_frames() {
        let reduced = motion_pixels(1920, 1080, vec![255; 1920 * 1080 * 4]).unwrap();
        assert_eq!(reduced.len(), 160 * 90 * 4);
        assert!(motion_pixels(u32::MAX, u32::MAX, Vec::new()).is_err());
        assert!(motion_pixels(10, 10, vec![0; 3]).is_err());
        assert!(motion_pixels(0, 0, Vec::new()).is_err());
    }
    #[test]
    fn freezes_are_timestamped_and_intentional_holds_are_exempt() {
        let times: Vec<_> = (0..40).map(|n| (Tick::from_seconds_f64(n as f64 / 10.0), vec![if n < 10 { n as u8 * 20 } else { 200 }; 64])).collect();
        let f = frozen(&times, &[], Tick::from_seconds_f64(1.0));
        assert_eq!(f.len(), 1);
        assert!(f[0]["seconds"].as_f64().unwrap() < 1.1);
        assert!(frozen(&times, &[(Tick::from_seconds_f64(0.9).0, Tick::from_seconds_f64(4.0).0)], Tick::from_seconds_f64(1.0)).is_empty());
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;
    #[test]
    fn truncated_opaque_text_and_inaudible_cues_are_found() {
        let dir = crate::media_test_util::tmp_dir("animation-preflight");
        let path = dir.join("still.png");
        image::save_buffer(&path, &[100u8; 32 * 32 * 4], 32, 32, image::ColorType::Rgba8).unwrap();
        let mut s = Session::default();
        let result=s.execute(CMD,json!({"path":path,"expected":{"duration":254016000000i64,"width":3840,"alpha":true},"textBounds":[{"name":"title","start":0,"end":254016000000i64,"box":[0.9,0.1,0.4,0.1]}],"soundCues":[{"name":"hit","time":0}]})).unwrap();
        let codes: Vec<_> = result["findings"].as_array().unwrap().iter().filter_map(|v| v["code"].as_str()).collect();
        assert!(!result["passed"].as_bool().unwrap());
        for wanted in ["duration", "width", "frozen", "textBounds", "soundCue"] {
            assert!(codes.contains(&wanted), "{result}");
        }
    }
}

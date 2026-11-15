//! One reusable delivery gate. Failed or unavailable required checks block export; warnings
//! and sampled/declared evidence remain visible. This workflow never uploads or changes labels.
use crate::commands::{CommandSpec, bad, has_seq};
use crate::{EngineError, Result, Session};
use filmcraft_project::ClipId;
use filmcraft_time::Tick;
use serde_json::{Value, json};
use std::collections::BTreeSet;
const CMD: &str = "delivery.preflight";
fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>) -> CommandSpec {
    CommandSpec { id, label, params, run, enabled: has_seq, menu: &[], shortcut: None, journal: false }
}
pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            CMD,
            "Delivery quality gate",
            r#"{"short":bool=false,"privacy":bool=true,"narration":bool=true,"changedNarration":[clip id]?,"mix":bool=true,"holds":[{"start":ticks,"end":ticks,"reason":str}]?,"exemptions":[…audio continuity exemptions],"report":path?}"#,
            preflight,
        ),
        spec(
            "delivery.prepare",
            "Check and prepare delivery",
            r#"{"path":path,"short":bool=false,"settings":ExportSettings?,"qa":{…delivery.preflight},"cache":bool=true,"report":path?}"#,
            prepare,
        ),
    ]
}
fn push(checks: &mut Vec<Value>, name: &str, state: &str, detail: Value) {
    checks.push(json!({"check":name,"state":state,"detail":detail}));
}
fn flag(p: &Value, key: &str, default: bool) -> Result<bool> {
    p.get(key).map_or(Ok(default), |v| v.as_bool().ok_or_else(|| bad(CMD, format!("{key} must be boolean"))))
}
/// The audio at every clip boundary of `q`, measured by `sequence.checkContinuity` 2000 boundaries
/// at a time (its own bound), so a long edit is measured whole instead of failing the gate
/// unmeasured. A click (a step between neighbouring samples) blocks delivery. A loudness jump
/// alone is a warning: speech onsets, music hits and sound effects move more than 6 dB in 100 ms,
/// and finished, approved edits showed hundreds of them (267 in an hour-long travel video).
fn continuity(s: &mut Session, q: &filmcraft_project::Sequence, exemptions: Value) -> (&'static str, Value) {
    let end = q.duration().0;
    let times: BTreeSet<i64> =
        q.video_tracks.iter().chain(&q.audio_tracks).flat_map(|t| &t.items).flat_map(|c| [c.start.0, c.end().0]).filter(|t| *t > 0 && *t < end).collect();
    let times: Vec<i64> = times.into_iter().collect();
    // no boundary still asks once, so malformed exemptions are refused as before
    let chunks: Vec<&[i64]> = if times.is_empty() { vec![&[][..]] } else { times.chunks(2000).collect() };
    let (mut clicks, mut jumps, mut errors, mut measured) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
    for chunk in chunks {
        match s.execute("sequence.checkContinuity", json!({"times":chunk,"exemptions":exemptions})) {
            Ok(v) => {
                measured += v["boundaries"].as_array().map_or(0, Vec::len);
                for finding in v["findings"].as_array().into_iter().flatten() {
                    if finding["click"] == true { clicks.push(finding.clone()) } else { jumps.push(finding.clone()) }
                }
            }
            Err(e) => errors.push(e.to_string()),
        }
    }
    let state = if !clicks.is_empty() || !errors.is_empty() {
        "fail"
    } else if !jumps.is_empty() {
        "warning"
    } else {
        "pass"
    };
    (
        state,
        json!({"clicks":clicks,"loudnessJumps":jumps,"errors":errors,"boundaries":measured,"method":"100 ms RMS each side plus sample discontinuity; a click fails, a loudness jump warns"}),
    )
}

fn preflight(s: &mut Session, p: &Value) -> Result<Value> {
    if let Some(path) = p["report"].as_str() {
        crate::output_safety::protect(s, &s.project, path, CMD)?;
    }
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?.clone();
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    let end = q.duration();
    if end.0 <= 0 {
        return Err(bad(CMD, "sequence is empty"));
    }
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let (source_errors, sources) = crate::source_check::inspect(s, seq);
    // A final-delivery check must see current originals, even when editing used a
    // proxy or a source changed on disk since its last preview.
    for item in sources {
        s.media.remove(item);
    }
    let mut cuts = BTreeSet::new();
    cuts.insert(0);
    cuts.insert(end.0.saturating_sub(1));
    let frame = q.settings.frame_rate.frame_duration().0.max(1);
    for c in q.video_tracks.iter().filter(|t| t.enabled).flat_map(|t| &t.items).filter(|c| c.enabled) {
        for at in [c.start.0, c.end().0] {
            for d in [-frame, 0, frame] {
                let t = at.saturating_add(d);
                if t >= 0 && t < end.0 {
                    cuts.insert(t);
                }
            }
        }
    }
    for n in 0..((end.seconds() / 2.0).ceil() as usize).min(21600) {
        cuts.insert(Tick::from_seconds_f64(n as f64 * 2.0).0.min(end.0 - 1));
    }
    if cuts.len() > 10000 {
        return Err(bad(CMD, "too many cut samples; divide the delivery into chapters"));
    }
    let mut checks = Vec::new();
    let mut rendered = Vec::new();
    let mut frozen_holds = Vec::new();
    if let Some(holds) = p.get("holds") {
        for h in holds.as_array().filter(|a| a.len() <= 2000).ok_or_else(|| bad(CMD, "holds must be intervals"))? {
            let a = h["start"].as_i64().filter(|a| *a >= 0).ok_or_else(|| bad(CMD, "hold start must be nonnegative ticks"))?;
            let b = h["end"].as_i64().filter(|b| *b > a && *b <= end.0).ok_or_else(|| bad(CMD, "hold end outside sequence"))?;
            if h["reason"].as_str().is_none_or(|s| s.is_empty()) {
                return Err(bad(CMD, "hold needs a reason"));
            }
            frozen_holds.push((a, b));
        }
    }
    let project = s.project.clone();
    let provider = s.media.full_res_provider(project.clone(), s.services.clone());
    let scale = (160.0 / q.settings.width.max(q.settings.height).max(1) as f32).min(1.0);
    let samples = ((end.seconds() / 0.5).ceil() as usize).clamp(2, 600);
    let mut color = Vec::new();
    let mut black = Vec::new();
    let mut previous: Option<Tick> = None;
    for n in 0..samples {
        let time = Tick(((end.0 - 1) as i128 * n as i128 / (samples - 1) as i128) as i64);
        if let Some(start) = previous {
            let visible: Vec<_> =
                q.video_tracks.iter().filter(|t| t.enabled).flat_map(|t| &t.items).filter(|c| c.enabled && c.start < time && c.end() > start).collect();
            let transition = q.video_tracks.iter().filter(|t| t.enabled).flat_map(|t| &t.transitions).any(|t| t.start < time && t.end() > start);
            let static_picture = |c: &&filmcraft_project::TrackItem| {
                (c.frame_hold.is_some() || project.resolve_media(c.item).is_some_and(|(_, m, _)| m.info.kind == filmcraft_media::MediaKind::Still))
                    && c.effects.iter().all(|e| !e.is_animated() && (e.def().is_some_and(|d| d.intrinsic) || e.effect == "lumetri"))
            };
            if !transition && !visible.is_empty() && visible.iter().all(static_picture) {
                frozen_holds.push((start.0, time.0));
            }
        }
        previous = Some(time);
        let pic = filmcraft_render::render_sequence(&project, seq, time, filmcraft_render::RenderOptions { scale, ..Default::default() }, &provider);
        let stats = filmcraft_render::color_balance::measure(&pic);
        if stats.blown > 0.05 {
            color.push(json!({"time":time.0,"seconds":time.seconds(),"blownShare":stats.blown}));
        }
        if stats.dark > 0.98 {
            black.push(json!({"time":time.0,"seconds":time.seconds(),"darkShare":stats.dark}));
        }
        rendered.push((time, crate::animation_preflight::motion_pixels(pic.w as u32, pic.h as u32, pic.to_rgba8())?));
    }
    let freezes = crate::animation_preflight::frozen(&rendered, &frozen_holds, Tick::from_seconds_f64(2.0));
    push(
        &mut checks,
        "motion",
        if freezes.is_empty() { "pass" } else { "warning" },
        json!({"findings":freezes,"samples":samples,"holds":frozen_holds,"sampled":true}),
    );
    push(
        &mut checks,
        "colour",
        if color.is_empty() { "pass" } else { "warning" },
        json!({"highlights":color,"working":q.settings.color.working,"autoToneMap":q.settings.color.auto_tone_map,"sampled":true}),
    );
    push(
        &mut checks,
        "blackFrames",
        if black.is_empty() { "pass" } else { "warning" },
        json!({"findings":black,"note":"review intentional dips on the frame sheet"}),
    );
    let missing: Vec<_> = source_errors.iter().filter_map(|v| v["clip"].as_u64()).collect();
    push(&mut checks, "sources", if source_errors.is_empty() { "pass" } else { "fail" }, json!({"offlineClips":missing,"errors":source_errors}));
    let (state, detail) = continuity(s, &q, p.get("exemptions").cloned().unwrap_or(json!([])));
    push(&mut checks, "audioContinuity", state, detail);
    if flag(p, "mix", true)? {
        match s.execute("audio.analyzeMix", json!({})) {
            Ok(v) => {
                let fail = v["mix"]["truePeakDbtp"].as_f64().is_some_and(|v| v > 0.05);
                push(&mut checks, "mix", if fail { "fail" } else { "pass" }, v)
            }
            Err(e) => push(&mut checks, "mix", "fail", json!(e.to_string())),
        }
    } else {
        push(&mut checks, "mix", "skipped", json!("disabled by explicit project policy"));
    }
    if flag(p, "privacy", true)? {
        let mut found = Vec::new();
        let mut failures = Vec::new();
        let times: Vec<_> = cuts.iter().map(|t| Tick(*t).seconds()).collect();
        for chunk in times.chunks(300) {
            match s.execute("sequence.checkPrivacy", json!({"seconds":chunk})) {
                Ok(v) => {
                    if v["shown"].as_u64().unwrap_or(1) > 0 || v["missedFrames"].as_u64().unwrap_or(1) > 0 {
                        found.push(v);
                    }
                }
                Err(e) => failures.push(e.to_string()),
            }
        }
        push(
            &mut checks,
            "privacyAtCuts",
            if found.is_empty() && failures.is_empty() { "pass" } else { "fail" },
            json!({"findings":found,"errors":failures,"samples":times.len(),"sampled":true}),
        );
    } else {
        push(&mut checks, "privacyAtCuts", "skipped", json!("disabled by explicit project policy"));
    }
    if flag(p, "narration", true)? {
        let changed: Option<BTreeSet<u64>> = p
            .get("changedNarration")
            .map(|v| {
                v.as_array()
                    .filter(|a| a.len() <= 2000)
                    .ok_or_else(|| bad(CMD, "changedNarration must be up to 2000 clip ids"))
                    .and_then(|a| a.iter().map(|v| v.as_u64().ok_or_else(|| bad(CMD, "narration id must be unsigned"))).collect())
            })
            .transpose()?;
        let mut voices = Vec::new();
        let mut failed = false;
        let clips: Vec<_> = q
            .audio_tracks
            .iter()
            .flat_map(|t| &t.items)
            .filter(|c| s.project.narrations.contains_key(&c.item) && changed.as_ref().is_none_or(|set| set.contains(&c.id.0)))
            .map(|c| c.id)
            .collect();
        if changed.as_ref().is_some_and(|set| set.iter().any(|id| !clips.contains(&ClipId(*id)))) {
            return Err(bad(CMD, "changedNarration refers to a missing narration clip"));
        }
        for clip in clips {
            let result = s.execute("tts.checkClip", json!({"clip":clip.0}));
            match result {
                Ok(v) => {
                    failed |= v["status"] != "clear";
                    voices.push(json!({"clip":clip.0,"result":v}));
                }
                Err(e) => {
                    failed = true;
                    voices.push(json!({"clip":clip.0,"error":e.to_string()}));
                }
            }
        }
        push(
            &mut checks,
            "changedNarration",
            if failed { "fail" } else { "pass" },
            json!({"clips":voices,"cache":"actual source sample digest + script + strict judge; persistent pronunciation rules retained"}),
        );
    } else {
        push(&mut checks, "changedNarration", "skipped", json!("disabled by explicit project policy"));
    }
    // Caption tracks are compared with the first. A track cut into different cues (a translation
    // rarely breaks where its source does) cannot be told from one that fell behind an edit by
    // timings alone, so a difference is a review warning that says how many cues each track has
    // and where its last one ends. While it failed the gate, a finished two-language edit (249
    // English cues, 338 Malayalam) could not be delivered.
    let span = |t: &filmcraft_project::CaptionTrack| json!({"track":t.id.0,"name":t.name,"cues":t.captions.len(),"lastEnd":t.captions.iter().map(|c| c.start.0.saturating_add(c.duration.0)).max()});
    let mut alignment = Vec::new();
    if let Some(base) = q.caption_tracks.first() {
        for track in q.caption_tracks.iter().skip(1) {
            let same = base.captions.len() == track.captions.len()
                && base.captions.iter().zip(&track.captions).all(|(a, b)| {
                    (a.start.0 as i128 - b.start.0 as i128).abs() <= i128::from(frame)
                        && (a.duration.0 as i128 - b.duration.0 as i128).abs() <= i128::from(frame)
                });
            if !same {
                alignment.push(span(track));
            }
        }
    }
    push(
        &mut checks,
        "subtitleAlignment",
        if alignment.is_empty() { "pass" } else { "warning" },
        json!({"misaligned":alignment,"base":q.caption_tracks.first().map(span),"toleranceTicks":frame,"tracks":q.caption_tracks.len()}),
    );
    if flag(p, "short", false)? {
        match s.execute("sequence.checkSafeZones", json!({"platform":"youtube"})) {
            Ok(v) => push(&mut checks, "safeZones", if v["ok"] == true { "pass" } else { "fail" }, v),
            Err(e) => push(&mut checks, "safeZones", "fail", json!(e.to_string())),
        }
    } else {
        push(&mut checks, "safeZones", "skipped", json!("horizontal delivery uses sidecar subtitles"));
    }
    let mut protected = Vec::new();
    for clip in q.video_tracks.iter().flat_map(|t| &t.items) {
        for e in &clip.effects {
            if crate::auto_color::is_pass(e) && e.params.contains_key("auto_color_screens") && e.masks.is_empty() {
                protected.push(clip.id.0);
            }
        }
    }
    push(
        &mut checks,
        "screenProtection",
        if protected.is_empty() { "pass" } else { "fail" },
        json!({"missingMasks":protected,"method":"saved editable protection masks; review moving screens at source scale"}),
    );
    let passed = checks.iter().all(|c| c["state"] != "fail");
    let mut report = json!({"format":"autocut.deliveryQa","version":1,"engine":crate::render_cache::ENGINE,"sequence":seq.0,"passed":passed,"checks":checks,"subtitlePolicy":if flag(p,"short",false)?{"burn"}else{"sidecar"},"uploadPrivacy":"private"});
    if let Some(path) = p["report"].as_str() {
        for n in 0..1000 {
            let destination = if n == 0 { path.to_string() } else { format!("{}.{n}.json", path.strip_suffix(".json").unwrap_or(path)) };
            crate::output_safety::protect(s, &s.project, &destination, CMD)?;
            report["reportPath"] = json!(destination);
            match s.services.write_file_new(&destination, &serde_json::to_vec_pretty(&report).map_err(|e| bad(CMD, e.to_string()))?) {
                Ok(()) => return Ok(report),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(bad(CMD, error.to_string())),
            }
        }
        return Err(bad(CMD, "too many report revisions; choose a new report path"));
    }
    Ok(report)
}
fn prepare(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "delivery.prepare";
    let requested_path = p["path"].as_str().filter(|s| !s.is_empty() && s.len() <= 4096).ok_or_else(|| bad(cmd, "provide a new output path"))?;
    let short = flag(p, "short", false)?;
    let mut export = json!({"path":requested_path,"preset":"YouTube 2160p 4K Ultra HD","wait":true,"burnCaptions":short,"loudnessLufs":-14.0});
    if short {
        export["width"] = json!(2160);
        export["height"] = json!(3840);
    }
    if let Some(settings) = p.get("settings") {
        export["settings"] = settings.clone();
    }
    let (_, mut resolved) = crate::export_tools::settings_from_params(s, &export, cmd)?;
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    // Validate the exact path the exporter will use, including home expansion and
    // its format extension, before checking collisions or naming sidecars.
    let output_path = crate::export_tools::output_path(s, &export, seq, &resolved, None);
    let path = output_path.as_str();
    export["path"] = json!(path);
    if s.services.file_size(path).is_ok() {
        return Err(bad(cmd, "output exists; choose a new revision path"));
    }
    crate::output_safety::protect(s, &s.project, path, cmd)?;
    let mut qa = p.get("qa").cloned().unwrap_or(json!({}));
    if !qa.is_object() {
        return Err(bad(cmd, "qa must be an object"));
    }
    qa["short"] = json!(short);
    let report_path = p["report"].as_str().map(str::to_string).unwrap_or_else(|| format!("{path}.qa.json"));
    let mut destinations = vec![path.to_string(), format!("{path}.delivery.json")];
    if let Some(q) = s.active_sequence() {
        for n in 0..q.caption_tracks.len() {
            destinations.push(format!("{path}.C{}.srt", n + 1));
        }
    }
    for destination in &destinations {
        if crate::output_safety::aliases(destination, &report_path) {
            return Err(bad(cmd, "report must have its own path, separate from video, subtitles and manifest"));
        }
        crate::output_safety::protect(s, &s.project, destination, cmd)?;
        if s.services.file_size(destination).is_ok() {
            return Err(bad(cmd, format!("delivery output already exists: {destination}; choose a new revision")));
        }
    }
    qa["report"] = json!(report_path);
    let report = preflight(s, &qa)?;
    let report_path = report["reportPath"].as_str().ok_or_else(|| bad(cmd, "quality report was not saved"))?.to_string();
    if report["passed"] != true {
        return Ok(json!({"prepared":false,"blocked":true,"report":report_path,"qa":report}));
    }
    // Delivery writes each track exclusively below; the legacy exporter sidecar
    // can overwrite an unrelated existing file and is not needed here.
    resolved.caption_sidecar = None;
    export["settings"] = serde_json::to_value(&resolved).map_err(|e| bad(cmd, e.to_string()))?;
    resolved.path.clear();
    let settings = serde_json::to_value(resolved).map_err(|e| bad(cmd, e.to_string()))?;
    let caching = flag(p, "cache", true)?;
    // The cache saves a render; it is never a reason to lose one. A plan that cannot be made, a
    // session without a cache folder and a publication that fails all leave the delivery to go
    // ahead uncached, with the reason in `cache.note` (a failed publication after a long export
    // used to fail the delivery, and the retry was refused because the master existed).
    let mut cache_note = None;
    let plan = if caching {
        match s.execute("render.cachePlan", json!({"settings":settings})) {
            Ok(plan) if plan["cacheAvailable"] == true => Some(plan),
            Ok(_) => {
                cache_note = Some("no cache folder is configured: exported without the render cache".to_string());
                None
            }
            Err(e) => {
                cache_note = Some(format!("exported without the render cache: {e}"));
                None
            }
        }
    } else {
        None
    };
    let hit = plan.as_ref().is_some_and(|v| v["hit"] == true);
    let output = if let Some(plan) = plan.as_ref().filter(|_| hit) {
        crate::render_cache::copy_artifact(&plan["artifact"], std::path::Path::new(path)).map_err(|e| bad(cmd, e.to_string()))?;
        json!({"path":path,"cached":true})
    } else {
        let output = s.execute("file.exportMedia", export)?;
        if output["offlineMedia"].as_array().is_some_and(|a| !a.is_empty()) {
            return Err(bad(cmd, "export contains offline placeholders; delivery stopped"));
        }
        if let Some(plan) = &plan
            && let Err(e) = s.execute("render.cacheCommit", json!({"key":plan["key"],"path":path,"manifest":plan}))
        {
            cache_note = Some(format!("the export is complete but was not cached: {e}"));
        }
        output
    };
    let cache = json!({"requested":caching,"hit":hit,"note":cache_note});
    let tracks = s.active_sequence().ok_or(EngineError::NoSequence)?.caption_tracks.clone();
    let mut subtitles = Vec::new();
    for (n, track) in tracks.iter().enumerate() {
        let subtitle = format!("{path}.C{}.srt", n + 1);
        s.execute("captions.export", json!({"track":track.id.0,"format":"srt","path":subtitle,"overwrite":false}))?;
        subtitles.push(json!({"track":track.id.0,"name":track.name,"path":subtitle,"cueIds":track.captions.iter().map(|c|c.id.0).collect::<Vec<_>>() }));
    }
    let final_check = s.execute("publish.preflight", json!({"destination":if short{"youtubeShort"}else{"youtube"},"path":path,"loudness":true}))?;
    let manifest = json!({"format":"autocut.delivery","version":1,"export":output,"cache":cache,"qa":report_path,"publishPreflight":final_check,"subtitles":subtitles,"privacy":"private","captionPolicy":if short{"burn"}else{"sidecar"}});
    s.services
        .write_file_new(&format!("{path}.delivery.json"), &serde_json::to_vec_pretty(&manifest).map_err(|e| bad(cmd, e.to_string()))?)
        .map_err(|e| bad(cmd, e.to_string()))?;
    Ok(json!({"prepared":final_check["ok"]==true,"blocked":final_check["ok"]!=true,"path":path,"cache":cache,"manifest":manifest,"uploaded":false}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_skips_and_subtitle_policy_are_recorded_without_mutation() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let before = s.project.clone();
        let r = s.execute(CMD, json!({"privacy":false,"narration":false,"mix":false})).unwrap();
        assert_eq!(r["subtitlePolicy"], "sidecar");
        assert!(r["checks"].as_array().unwrap().iter().any(|c| c["check"] == "privacyAtCuts" && c["state"] == "skipped"));
        assert_eq!(&*s.project, &*before);
        assert!(s.execute(CMD, json!({"privacy":"no"})).is_err());
    }
    fn check<'a>(report: &'a Value, name: &str) -> &'a Value {
        report["checks"].as_array().unwrap().iter().find(|c| c["check"] == name).unwrap()
    }
    /// More than 2000 clip boundaries used to fail the gate unmeasured ("more than 2000
    /// boundaries"), and any loudness jump failed it: every long or lively edit was blocked.
    #[test]
    fn long_edits_are_measured_whole_and_only_clicks_block() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.execute("sequence.addTracks", json!({"video":1})).unwrap();
        s.edit_sequence("Many cuts", |q, ctx, _| {
            let frame = q.settings.frame_rate.frame_duration();
            let template = q.video_tracks[0].items[0].clone();
            let track = q.video_tracks.last_mut().unwrap();
            for n in 0..2101 {
                let mut clip = template.clone();
                clip.id = filmcraft_project::ClipId(ctx.alloc());
                clip.link = None;
                clip.start = Tick(frame.0 * n);
                clip.duration = frame;
                track.items.push(clip);
            }
            Ok(())
        })
        .unwrap();
        let r = s.execute(CMD, json!({"privacy":false,"narration":false,"mix":false})).unwrap();
        let audio = check(&r, "audioContinuity");
        assert!(audio["detail"]["errors"].as_array().unwrap().is_empty(), "{audio}");
        assert!(audio["detail"]["boundaries"].as_u64().unwrap() > 2000, "{audio}");
        let clicks = !audio["detail"]["clicks"].as_array().unwrap().is_empty();
        assert_eq!(audio["state"] == "fail", clicks, "only a click blocks delivery: {audio}");
    }
    /// Two caption tracks cut into different cues failed the gate: a translation and its source
    /// could never be delivered together.
    #[test]
    fn differently_cut_caption_tracks_are_a_review_warning() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        // a new caption track goes first, so "Translation" is the one the others are compared with
        for (name, lines) in [("Source", &["One long line"][..]), ("Translation", &["Two", "short lines"][..])] {
            let track = s.execute("captions.newTrack", json!({"format":"Subtitle","name":name})).unwrap()["track"].clone();
            let seconds = 2.0 / lines.len() as f64;
            for (n, text) in lines.iter().enumerate() {
                s.execute("captions.add", json!({"track":track,"text":text,"seconds":n as f64 * seconds,"durationSeconds":seconds * 0.5})).unwrap();
            }
        }
        let r = s.execute(CMD, json!({"privacy":false,"narration":false,"mix":false})).unwrap();
        let subtitles = check(&r, "subtitleAlignment");
        assert_eq!(subtitles["state"], "warning", "{subtitles}");
        assert_eq!(subtitles["detail"]["base"]["cues"], 2, "{subtitles}");
        let source = subtitles["detail"]["misaligned"].as_array().unwrap().iter().find(|t| t["name"] == "Source").unwrap();
        assert_eq!(source["cues"], 1, "{subtitles}");
    }
    /// A session without a cache folder planned a miss, exported, then failed on publishing the
    /// cache entry: the delivery was an error although its master was written, and a retry was
    /// refused because that master existed.
    #[test]
    fn delivery_goes_ahead_without_a_cache_folder() {
        let dir = crate::media_test_util::tmp_dir("delivery-no-cache");
        let image = dir.join("source.png");
        image::save_buffer(&image, &[140; 80 * 120 * 3], 80, 120, image::ColorType::Rgb8).unwrap();
        let mut s = Session::default();
        let item = crate::commands::import_streamed(&mut s, image.to_str().unwrap(), None).unwrap();
        s.execute("file.newSequence", json!({"width":80,"height":120,"fps":30,"video":1,"audio":1})).unwrap();
        s.execute("timeline.place", json!({"item":item.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
        let video = dir.join("delivery.mp4");
        let r = s
            .execute("delivery.prepare", json!({"path":video,"settings":{"frameSize":[80,120]},"qa":{"privacy":false,"narration":false,"mix":false}}))
            .unwrap();
        assert!(video.exists(), "{r}");
        assert!(std::path::Path::new(&format!("{}.delivery.json", video.display())).exists(), "{r}");
        assert_eq!(r["cache"]["hit"], false);
        assert!(r["cache"]["note"].as_str().is_some_and(|n| n.contains("cache")), "{r}");
    }
    #[test]
    fn a_required_unavailable_privacy_check_blocks_preparation() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.redacts.detector = Some(std::sync::Arc::new(|_, _, _, _| {
            vec![filmcraft_vision::Detection {
                kind: filmcraft_vision::Kind::Face,
                score: 0.99,
                bbox: [200.0, 200.0, 160.0, 160.0],
                points: Vec::new(),
                pose: false,
            }]
        }));
        let r = s.execute(CMD, json!({"narration":false,"mix":false})).unwrap();
        assert_eq!(r["passed"], false);
        assert!(r["checks"].as_array().unwrap().iter().any(|c| c["check"] == "privacyAtCuts" && c["state"] == "fail"));
    }
}

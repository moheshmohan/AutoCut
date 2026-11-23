//! One reusable delivery gate. Failed or unavailable required checks block export; warnings
//! and sampled/declared evidence remain visible. This workflow never uploads or changes labels.
//!
//! Every check has a severity tier (PX1.4, [`tier_of`]): **Floor** blocks (the checks that
//! blocked before keep blocking exactly as before; the new ones, `textContrast` under 3:1,
//! `figures` not in the fact ledger and `waivers`, warn for this first round, decision D7, until
//! `floorBlocks: true`), **Reflex** is fixed or waived with a reason and evidence in
//! `build/decisions.json` (`{waivers: [{check, code?, time?, reason, evidence}]}`; a waiver
//! without both is refused, and more than five is a finding), **Finish** is reported only. The
//! report's `summary` counts the open findings of each tier and the waived ones. `motionCraft`
//! (PX1.5) is `sequence.checkMotion`'s findings at the Reflex tier: fixed, or waived with evidence.
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
            r#"{"short":bool=false,"privacy":bool=true,"narration":bool=true,"changedNarration":[clip id]?,"mix":bool=true,"motionEnergy":bool=true (qa.motionEnergy, warnings only),"craft":bool=true (sequence.checkCraft: textContrast under 3:1 is Floor, the rest craft),"motionCraft":bool=true (sequence.checkMotion: eases, entrance and exit timing, stagger, loops and pops; Reflex),"copy":bool=true (copy.lint of titles and captions: figures not in the ledger are Floor, the rest copy),"facts":[str|number]? (the fact ledger; default build/facts.json beside the project),"allow":[str]? (phrases copy.lint never flags),"decisions":path? (waivers; default build/decisions.json beside the project),"floorBlocks":bool=false (the new Floor checks block too),"holds":[{"start":ticks,"end":ticks,"reason":str}]?,"exemptions":[…audio continuity exemptions],"report":path?}"#,
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
    // the tiers' settings, read before anything is measured
    let floor_blocks = flag(p, "floorBlocks", false)?;
    let craft_on = flag(p, "craft", true)?;
    let motion_craft_on = flag(p, "motionCraft", true)?;
    let copy_inputs = if flag(p, "copy", true)? { Some((crate::copy_lint::allowed(s, CMD, p)?, crate::copy_lint::ledger(s, CMD, p)?)) } else { None };
    let (decisions, named_decisions) = decisions_path(s, p)?;
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
    // qa.motionEnergy over the whole edit, rendered from the originals: warnings only (decision
    // D7), so it never fails the gate. Still pictures on the timeline count as holds, as above.
    if flag(p, "motionEnergy", true)? {
        let o = crate::motion_energy::Options { holds: frozen_holds.clone(), expect_motion: true, ..Default::default() };
        match crate::motion_energy::measure_sequence(s, seq, &o, true) {
            Ok(v) => push(&mut checks, "motionEnergy", if v["findings"].as_array().is_some_and(|a| !a.is_empty()) { "warning" } else { "pass" }, v),
            Err(e) => push(&mut checks, "motionEnergy", "warning", json!({"error": e.to_string()})),
        }
    } else {
        push(&mut checks, "motionEnergy", "skipped", json!("disabled by explicit project policy"));
    }
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
    // The new Floor checks warn this round (decision D7) and block only with `floorBlocks`; an
    // unavailable one is a Floor failure too.
    let new_floor = |open: bool| {
        if !open {
            "pass"
        } else if floor_blocks {
            "fail"
        } else {
            "warning"
        }
    };
    // sequence.checkCraft: text contrast under 3:1 is Floor, its other findings Reflex or Finish
    if craft_on {
        match s.execute("sequence.checkCraft", json!({"sequence":seq.0,"holds":craft_holds(&frozen_holds, end.0)})) {
            Ok(mut v) => {
                let (floor, rest): (Vec<Value>, Vec<Value>) =
                    take_findings(&mut v).into_iter().partition(|f| f["code"] == "lowContrast" && f["ratio"].as_f64().is_some_and(|r| r < FLOOR_CONTRAST));
                push(
                    &mut checks,
                    "textContrast",
                    new_floor(!floor.is_empty()),
                    json!({"findings":floor.into_iter().map(|f| tiered(f, FLOOR)).collect::<Vec<_>>(),"threshold":FLOOR_CONTRAST,"floorBlocks":floor_blocks,"note":"text under 3:1 against the picture behind it, from sequence.checkCraft; 3:1 up to the WCAG ratio is in `craft`"}),
                );
                let rest: Vec<Value> = rest
                    .into_iter()
                    .map(|f| {
                        let t = craft_tier(f["code"].as_str().unwrap_or(""));
                        tiered(f, t)
                    })
                    .collect();
                let state = if rest.is_empty() { "pass" } else { "warning" };
                if let Some(o) = v.as_object_mut() {
                    o.insert("findings".into(), json!(rest));
                }
                push(&mut checks, "craft", state, v);
            }
            Err(e) => {
                push(&mut checks, "textContrast", new_floor(true), json!({"error":e.to_string(),"floorBlocks":floor_blocks}));
                push(&mut checks, "craft", "warning", json!({"error":e.to_string()}));
            }
        }
    } else {
        push(&mut checks, "textContrast", "skipped", json!("disabled by explicit project policy"));
        push(&mut checks, "craft", "skipped", json!("disabled by explicit project policy"));
    }
    // sequence.checkMotion: how titles, transitions and keyframes move, every finding Reflex
    if motion_craft_on {
        match s.execute("sequence.checkMotion", json!({"sequence":seq.0})) {
            Ok(mut v) => {
                let found: Vec<Value> = take_findings(&mut v).into_iter().map(|f| tiered(f, REFLEX)).collect();
                let state = if found.is_empty() { "pass" } else { "warning" };
                if let Some(o) = v.as_object_mut() {
                    o.insert("findings".into(), json!(found));
                }
                push(&mut checks, "motionCraft", state, v);
            }
            Err(e) => push(&mut checks, "motionCraft", "warning", json!({"error":e.to_string()})),
        }
    } else {
        push(&mut checks, "motionCraft", "skipped", json!("disabled by explicit project policy"));
    }
    // copy.lint of the titles and captions: figures not in the fact ledger are Floor
    if let Some((allow, ledger)) = &copy_inputs {
        let mut v = crate::copy_lint::report(&crate::copy_lint::sequence_segments(&project, &q), allow, ledger);
        let (figures, rest): (Vec<Value>, Vec<Value>) = take_findings(&mut v).into_iter().partition(|f| f["code"] == "unledgeredFigure");
        match ledger {
            crate::copy_lint::LedgerState::Found(_) => push(
                &mut checks,
                "figures",
                new_floor(!figures.is_empty()),
                json!({"findings":figures.into_iter().map(|f| tiered(f, FLOOR)).collect::<Vec<_>>(),"ledger":v["ledger"],"floorBlocks":floor_blocks}),
            ),
            crate::copy_lint::LedgerState::None(why) => push(&mut checks, "figures", "skipped", json!(why)),
            crate::copy_lint::LedgerState::Broken(why) => {
                push(&mut checks, "figures", new_floor(true), json!({"error":format!("the fact ledger could not be read: {why}"),"floorBlocks":floor_blocks}))
            }
        }
        let rest: Vec<Value> = rest
            .into_iter()
            .map(|f| {
                let t = copy_tier(&f);
                tiered(f, t)
            })
            .collect();
        let state = if rest.is_empty() { "pass" } else { "warning" };
        if let Some(o) = v.as_object_mut() {
            o.insert("findings".into(), json!(rest));
        }
        push(&mut checks, "copy", state, v);
    } else {
        push(&mut checks, "figures", "skipped", json!("disabled by explicit project policy"));
        push(&mut checks, "copy", "skipped", json!("disabled by explicit project policy"));
    }
    // waivers of Reflex findings, then every check's tier
    let (state, detail) = waive(s, decisions, named_decisions, &mut checks, frame, floor_blocks);
    push(&mut checks, "waivers", state, detail);
    for c in checks.iter_mut() {
        let tier = tier_of(c["check"].as_str().unwrap_or(""));
        if let Some(o) = c.as_object_mut() {
            o.insert("tier".into(), json!(tier));
        }
    }
    let summary = summary(&checks);
    let blocking: Vec<Value> = checks.iter().filter(|c| c["state"] == "fail").map(|c| c["check"].clone()).collect();
    let passed = checks.iter().all(|c| c["state"] != "fail");
    let mut report = json!({"format":"autocut.deliveryQa","version":1,"engine":crate::render_cache::ENGINE,"sequence":seq.0,"passed":passed,"blocking":blocking,"summary":summary,"floorBlocks":floor_blocks,"checks":checks,"subtitlePolicy":if flag(p,"short",false)?{"burn"}else{"sidecar"},"uploadPrivacy":"private"});
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
// ---- severity tiers (PX1.4) ---------------------------------------------------------------

const FLOOR: &str = "floor";
const REFLEX: &str = "reflex";
const FINISH: &str = "finish";
/// Text contrast under this is a Floor failure (3:1 to the WCAG ratio stays in `craft`).
pub const FLOOR_CONTRAST: f64 = 3.0;
/// More waivers than this is itself a finding.
pub const MAX_WAIVERS: usize = 5;
/// Largest decisions file read, most waivers looked at, and the longest reason or evidence.
const MAX_DECISIONS_BYTES: u64 = 1 << 20;
const MAX_WAIVER_ENTRIES: usize = 1000;
const MAX_WAIVER_TEXT: usize = 2000;
/// Most holds handed to `sequence.checkCraft` (its own bound).
const MAX_CRAFT_HOLDS: usize = 2000;

/// The tier of each check: Floor blocks (the existing blocking checks keep blocking as before;
/// the new ones warn until `floorBlocks`), Reflex is fixed or waived with evidence, Finish is
/// reported only.
pub fn tier_of(check: &str) -> &'static str {
    match check {
        "sources" | "audioContinuity" | "mix" | "privacyAtCuts" | "changedNarration" | "safeZones" | "screenProtection" | "textContrast" | "figures"
        | "waivers" => FLOOR,
        "motion" | "colour" | "blackFrames" | "craft" | "motionCraft" | "copy" => REFLEX,
        _ => FINISH,
    }
}

/// The tier of a `sequence.checkCraft` finding left in `craft` (contrast under 3:1 goes to
/// `textContrast`).
fn craft_tier(code: &str) -> &'static str {
    match code {
        "staticTimeline" | "silence" => FINISH,
        _ => REFLEX,
    }
}

/// The tier of a `copy.lint` finding left in `copy` (figures go to `figures`).
fn copy_tier(f: &Value) -> &'static str {
    match f["code"].as_str().unwrap_or("") {
        "filler" => FINISH,
        "overBudget" if f["level"] != "fail" => FINISH,
        _ => REFLEX,
    }
}

fn tiered(mut f: Value, tier: &str) -> Value {
    if let Some(o) = f.as_object_mut() {
        o.insert("tier".into(), json!(tier));
    }
    f
}

fn take_findings(v: &mut Value) -> Vec<Value> {
    match v.get_mut("findings").map(Value::take) {
        Some(Value::Array(a)) => a,
        _ => Vec::new(),
    }
}

/// `holds` merged and capped for `sequence.checkCraft`, each with a reason.
fn craft_holds(holds: &[(i64, i64)], end: i64) -> Value {
    let mut spans: Vec<(i64, i64)> = holds.iter().map(|(a, b)| ((*a).max(0), (*b).min(end))).filter(|(a, b)| b > a).collect();
    spans.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (a, b) in spans {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    json!(merged.iter().take(MAX_CRAFT_HOLDS).map(|(a, b)| json!({"start": a, "end": b, "reason": "a declared hold or a still picture"})).collect::<Vec<_>>())
}

/// One waiver of `build/decisions.json`, accepted.
struct Waiver {
    index: usize,
    check: String,
    code: Option<String>,
    time: Option<i64>,
    reason: String,
    evidence: String,
    used: usize,
}

/// The decisions file: `p.decisions`, else `build/decisions.json` beside the saved project; and
/// whether it was named.
fn decisions_path(s: &Session, p: &Value) -> Result<(Option<String>, bool)> {
    match p.get("decisions") {
        None | Some(Value::Null) => Ok((crate::copy_lint::beside_project(s, "decisions.json"), false)),
        Some(Value::String(t)) if !t.trim().is_empty() && t.len() <= 4096 && !t.contains('\0') => Ok((Some(crate::export_tools::expand_home(t.trim())), true)),
        Some(_) => Err(bad(CMD, "`decisions` is the path of a decisions file ({waivers: [...]})")),
    }
}

fn refused(index: usize, check: Option<&str>, why: String) -> Value {
    json!({"code": "waiverRefused", "severity": "warn", "tier": FLOOR, "waiver": index, "check": check, "message": format!("waiver {}: {why}", index + 1)})
}

/// The waivers of the decisions file applied to the Reflex checks among `checks`: the `waivers`
/// check's state and detail.
fn waive(s: &Session, path: Option<String>, named: bool, checks: &mut [Value], frame: i64, floor_blocks: bool) -> (&'static str, Value) {
    let mut findings = Vec::new();
    let names: Vec<String> = checks.iter().filter_map(|c| c["check"].as_str().map(str::to_string)).collect();
    let Some(path) = path else {
        return ("pass", json!({"path": null, "waivers": 0, "note": "the project is not saved: no build/decisions.json beside it (or give `decisions`)"}));
    };
    let unreadable = |why: String| json!({"code": "decisionsUnreadable", "severity": "warn", "tier": FLOOR, "path": path, "message": format!("the decisions file could not be read: {why}")});
    let entries: Vec<Value> = if s.services.file_size(&path).is_err() {
        if !named {
            return ("pass", json!({"path": path, "waivers": 0, "note": "no decisions file: nothing is waived"}));
        }
        findings.push(unreadable("it is not there".into()));
        Vec::new()
    } else {
        match s
            .services
            .read_file_bounded(&path, MAX_DECISIONS_BYTES)
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()))
        {
            Ok(Value::Object(mut o)) => match o.remove("waivers") {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::Array(a)) => a,
                Some(_) => {
                    findings.push(unreadable("`waivers` must be a list".into()));
                    Vec::new()
                }
            },
            Ok(_) => {
                findings.push(unreadable("a decisions file is an object {\"waivers\": [...]}".into()));
                Vec::new()
            }
            Err(e) => {
                findings.push(unreadable(e));
                Vec::new()
            }
        }
    };
    if entries.len() > MAX_WAIVERS {
        findings.push(json!({
            "code": "tooManyWaivers", "severity": "warn", "tier": FLOOR, "count": entries.len(), "max": MAX_WAIVERS,
            "message": format!("{} waivers; more than {MAX_WAIVERS} means the Reflex checks are being waived instead of fixed", entries.len()),
        }));
    }
    let text = |w: &Value, k: &str| w.get(k).and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    let mut accepted: Vec<Waiver> = Vec::new();
    for (index, w) in entries.iter().take(MAX_WAIVER_ENTRIES).enumerate() {
        if !w.is_object() {
            findings.push(refused(index, None, "a waiver is an object {check, code?, time?, reason, evidence}".into()));
            continue;
        }
        let Some(check) = text(w, "check") else {
            findings.push(refused(index, None, "a waiver names its `check`".into()));
            continue;
        };
        if !names.contains(&check) {
            findings.push(refused(index, Some(&check), format!("this gate has no check `{}`", check.chars().take(64).collect::<String>())));
            continue;
        }
        match tier_of(&check) {
            REFLEX => {}
            FLOOR => {
                findings.push(refused(index, Some(&check), format!("`{check}` is a Floor check: it is fixed, never waived")));
                continue;
            }
            _ => {
                findings.push(refused(index, Some(&check), format!("`{check}` is a Finish check: it is only reported, there is nothing to waive")));
                continue;
            }
        }
        let (Some(reason), Some(evidence)) = (text(w, "reason"), text(w, "evidence")) else {
            findings.push(refused(index, Some(&check), "a waiver needs both a `reason` and `evidence`".into()));
            continue;
        };
        if reason.chars().count() > MAX_WAIVER_TEXT || evidence.chars().count() > MAX_WAIVER_TEXT {
            findings.push(refused(index, Some(&check), format!("a reason or evidence is at most {MAX_WAIVER_TEXT} characters")));
            continue;
        }
        let code = match w.get("code") {
            None | Some(Value::Null) => None,
            Some(Value::String(c)) => Some(c.clone()),
            Some(_) => {
                findings.push(refused(index, Some(&check), "`code` is a finding's code".into()));
                continue;
            }
        };
        let time = match w.get("time") {
            None | Some(Value::Null) => None,
            Some(t) => match t.as_i64().filter(|t| *t >= 0) {
                Some(t) => Some(t),
                None => {
                    findings.push(refused(index, Some(&check), "`time` is ticks from the start".into()));
                    continue;
                }
            },
        };
        accepted.push(Waiver { index, check, code, time, reason, evidence, used: 0 });
    }
    let mut applied = 0usize;
    for c in checks.iter_mut() {
        let name = c["check"].as_str().unwrap_or("").to_string();
        if tier_of(&name) != REFLEX {
            continue;
        }
        let key = if name == "colour" { "highlights" } else { "findings" };
        let Some(list) = c.get_mut("detail").and_then(|d| d.get_mut(key)).and_then(Value::as_array_mut) else { continue };
        let mut all = !list.is_empty();
        for f in list.iter_mut() {
            let start = f["time"].as_i64().or_else(|| f["start"].as_i64());
            let end = f["end"].as_i64().or(start);
            let at = |t: i64| matches!((start, end), (Some(a), Some(b)) if a.saturating_sub(frame) <= t && t <= b.saturating_add(frame));
            let w =
                accepted.iter_mut().find(|w| w.check == name && w.code.as_deref().is_none_or(|code| f["code"].as_str() == Some(code)) && w.time.is_none_or(at));
            match (w, f.as_object_mut()) {
                (Some(w), Some(o)) => {
                    w.used += 1;
                    applied += 1;
                    o.insert("waived".into(), json!({"waiver": w.index, "reason": w.reason, "evidence": w.evidence}));
                }
                _ => all = false,
            }
        }
        if all
            && c["state"] == "warning"
            && let Some(o) = c.as_object_mut()
        {
            o.insert("state".into(), json!("waived"));
        }
    }
    let unmatched: Vec<Value> =
        accepted.iter().filter(|w| w.used == 0).map(|w| json!({"waiver": w.index, "check": w.check, "code": w.code, "time": w.time})).collect();
    let state = if findings.is_empty() {
        "pass"
    } else if floor_blocks {
        "fail"
    } else {
        "warning"
    };
    (
        state,
        json!({
            "path": path, "waivers": entries.len(), "accepted": accepted.len(), "applied": applied, "unmatched": unmatched,
            "max": MAX_WAIVERS, "findings": findings, "floorBlocks": floor_blocks,
        }),
    )
}

/// What a check holds, by tier: each of its findings (or the check itself when it lists none),
/// and whether it is waived. A check that passed or was skipped holds nothing.
fn items(c: &Value) -> Vec<(&'static str, bool)> {
    if matches!(c["state"].as_str(), Some("pass" | "skipped")) {
        return Vec::new();
    }
    let name = c["check"].as_str().unwrap_or("");
    let tier = tier_of(name);
    let d = &c["detail"];
    let list = |k: &str| d.get(k).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let named = |f: &Value| match f["tier"].as_str() {
        Some("floor") => FLOOR,
        Some("reflex") => REFLEX,
        Some("finish") => FINISH,
        _ => tier,
    };
    let mut v: Vec<(&'static str, bool)> = match name {
        "audioContinuity" => {
            list("clicks").iter().chain(list("errors")).map(|_| (FLOOR, false)).chain(list("loudnessJumps").iter().map(|_| (FINISH, false))).collect()
        }
        "colour" => list("highlights").iter().map(|f| (tier, f.get("waived").is_some())).collect(),
        _ => list("findings").iter().map(|f| (named(f), f.get("waived").is_some())).collect(),
    };
    if v.is_empty() {
        v.push((tier, false));
    }
    v
}

/// The open items of `checks` by tier, and the waived ones.
fn summary(checks: &[Value]) -> Value {
    let (mut floor, mut reflex, mut finish, mut waived) = (0usize, 0usize, 0usize, 0usize);
    for (tier, w) in checks.iter().flat_map(items) {
        match (tier, w) {
            (_, true) => waived += 1,
            (FLOOR, _) => floor += 1,
            (REFLEX, _) => reflex += 1,
            _ => finish += 1,
        }
    }
    json!({"floor": floor, "reflex": reflex, "finish": finish, "waived": waived})
}

/// What a delivery is made from: the project, the sequence, the export settings and the checks
/// asked for. A delivery that stopped after its export is finished by a second run only while
/// this is unchanged, so its subtitles and manifest belong to the master that is already there.
fn edit_digest(s: &Session, seq: filmcraft_project::ItemId, settings: &Value, qa: &Value) -> Result<String> {
    use sha2::{Digest, Sha256};
    let cmd = "delivery.prepare";
    let project = serde_json::to_value(&*s.project).map_err(|e| bad(cmd, e.to_string()))?;
    let edit = json!({"engine":crate::render_cache::ENGINE,"project":project,"sequence":seq.0,"settings":settings,"qa":qa});
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&edit).map_err(|e| bad(cmd, e.to_string()))?)))
}
fn prepare(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "delivery.prepare";
    let requested_path = p["path"].as_str().filter(|s| !s.is_empty() && s.len() <= 4096).ok_or_else(|| bad(cmd, "provide a new output path"))?;
    let short = flag(p, "short", false)?;
    let caching = flag(p, "cache", true)?;
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
    let mut qa = p.get("qa").cloned().unwrap_or(json!({}));
    if !qa.is_object() {
        return Err(bad(cmd, "qa must be an object"));
    }
    qa["short"] = json!(short);
    // Delivery writes each track exclusively below; the legacy exporter sidecar
    // can overwrite an unrelated existing file and is not needed here.
    resolved.caption_sidecar = None;
    export["settings"] = serde_json::to_value(&resolved).map_err(|e| bad(cmd, e.to_string()))?;
    resolved.path.clear();
    let settings = serde_json::to_value(resolved).map_err(|e| bad(cmd, e.to_string()))?;
    // A step that failed after the export (a subtitle, the publish pre-flight, the manifest) used
    // to leave the master without a manifest, and the retry was refused because the master
    // existed: a long export had to be made again under a new name. The export now leaves an
    // unfinished manifest (`complete: false`) that names the edit, and a second run of the same
    // edit to the same path finishes the delivery from the master that is there.
    let manifest_path = format!("{path}.delivery.json");
    let edit = edit_digest(s, seq, &settings, &qa)?;
    let unfinished = match s.services.file_size(path) {
        Err(_) => None,
        Ok(bytes) => {
            let record = s.services.read_file_bounded(&manifest_path, 16 << 20).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            match record.filter(|m| m["format"] == "autocut.delivery" && m["complete"] == false) {
                Some(m) if m["edit"] == edit.as_str() && m["bytes"].as_u64() == Some(bytes) => Some(m),
                Some(_) => return Err(bad(cmd, "an unfinished delivery of a different edit, export or check is at this path; choose a new revision path")),
                None => return Err(bad(cmd, "output exists; choose a new revision path")),
            }
        }
    };
    let resumed = unfinished.is_some();
    crate::output_safety::protect(s, &s.project, path, cmd)?;
    let report_path = p["report"].as_str().map(str::to_string).unwrap_or_else(|| format!("{path}.qa.json"));
    let mut destinations = vec![path.to_string(), manifest_path.clone()];
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
        if !resumed && s.services.file_size(destination).is_ok() {
            return Err(bad(cmd, format!("delivery output already exists: {destination}; choose a new revision")));
        }
    }
    // The checks of the first run stand for a delivery that is being finished: the edit and the
    // checks asked for are the same. They run again only if their report is gone.
    let kept = unfinished.as_ref().and_then(|m| m["qa"].as_str()).filter(|report| s.services.file_size(report).is_ok()).map(str::to_string);
    let report_path = match kept {
        Some(report) => report,
        None => {
            qa["report"] = json!(report_path);
            let report = preflight(s, &qa)?;
            let report_path = report["reportPath"].as_str().ok_or_else(|| bad(cmd, "quality report was not saved"))?.to_string();
            if report["passed"] != true {
                return Ok(json!({"prepared":false,"blocked":true,"report":report_path,"qa":report}));
            }
            report_path
        }
    };
    let (output, cache) = match &unfinished {
        Some(m) => (m["export"].clone(), m["cache"].clone()),
        None => {
            // The cache saves a render; it is never a reason to lose one. A plan that cannot be
            // made, a session without a cache folder and a publication that fails all leave the
            // delivery to go ahead uncached, with the reason in `cache.note` (a failed publication
            // after a long export used to fail the delivery).
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
            (output, json!({"requested":caching,"hit":hit,"note":cache_note}))
        }
    };
    // From here the master exists. A host that cannot leave the unfinished manifest goes on as
    // before: its delivery cannot be finished by a second run.
    let mut marked = resumed;
    if !marked && let Ok(bytes) = s.services.file_size(path) {
        let record = json!({"format":"autocut.delivery","version":1,"complete":false,"edit":edit,"bytes":bytes,"export":output,"cache":cache,"qa":report_path});
        marked = serde_json::to_vec_pretty(&record).is_ok_and(|record| s.services.write_file_new(&manifest_path, &record).is_ok());
    }
    let finished = (|| -> Result<Value> {
        let tracks = s.active_sequence().ok_or(EngineError::NoSequence)?.caption_tracks.clone();
        let mut subtitles = Vec::new();
        for (n, track) in tracks.iter().enumerate() {
            let subtitle = format!("{path}.C{}.srt", n + 1);
            // a second run writes over what the first got to: these paths were free before it
            s.execute("captions.export", json!({"track":track.id.0,"format":"srt","path":subtitle,"overwrite":resumed}))
                .map_err(|e| EngineError::Other(format!("subtitle {subtitle}: {e}")))?;
            subtitles.push(json!({"track":track.id.0,"name":track.name,"path":subtitle,"cueIds":track.captions.iter().map(|c|c.id.0).collect::<Vec<_>>() }));
        }
        let final_check = s.execute("publish.preflight", json!({"destination":if short{"youtubeShort"}else{"youtube"},"path":path,"loudness":true}))?;
        let manifest = json!({"format":"autocut.delivery","version":1,"complete":true,"export":output,"cache":cache,"qa":report_path,"publishPreflight":final_check,"subtitles":subtitles,"privacy":"private","captionPolicy":if short{"burn"}else{"sidecar"}});
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| bad(cmd, e.to_string()))?;
        let written = if marked { s.services.write_file(&manifest_path, &bytes) } else { s.services.write_file_new(&manifest_path, &bytes) };
        written.map_err(|e| bad(cmd, e.to_string()))?;
        Ok(
            json!({"prepared":final_check["ok"]==true,"blocked":final_check["ok"]!=true,"path":path,"cache":cache,"manifest":manifest,"uploaded":false,"resumed":resumed}),
        )
    })();
    finished.map_err(|e| {
        if marked {
            EngineError::Other(format!(
                "{cmd}: the master is exported, but the delivery is not finished: {e}. Run {cmd} again with the same path, settings and checks to finish it without exporting again"
            ))
        } else {
            e
        }
    })
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
    /// Native files, except that subtitles cannot be written while the switch is on: a failure
    /// after the export.
    struct NoSubtitles(std::sync::atomic::AtomicBool);
    impl NoSubtitles {
        fn refuse(&self, path: &str) -> std::io::Result<()> {
            if path.ends_with(".srt") && self.0.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(std::io::Error::other("the disk is full"));
            }
            Ok(())
        }
    }
    impl crate::Services for NoSubtitles {
        fn read_file(&self, path: &str) -> std::io::Result<Vec<u8>> {
            crate::FsServices.read_file(path)
        }
        fn write_file(&self, path: &str, data: &[u8]) -> std::io::Result<()> {
            self.refuse(path)?;
            crate::FsServices.write_file(path, data)
        }
        fn write_file_new(&self, path: &str, data: &[u8]) -> std::io::Result<()> {
            self.refuse(path)?;
            crate::FsServices.write_file_new(path, data)
        }
        fn file_size(&self, path: &str) -> std::io::Result<u64> {
            crate::FsServices.file_size(path)
        }
        fn reader(&self, path: &str) -> Option<std::io::Result<filmcraft_media::SharedReader>> {
            crate::FsServices.reader(path)
        }
        fn read_range(&self, path: &str, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
            crate::FsServices.read_range(path, offset, len)
        }
    }
    /// A one-second captioned edit whose delivery stopped after its export: the session, the
    /// switch that lets subtitles be written again, the master and the delivery's parameters.
    fn unfinished(name: &str) -> (Session, std::sync::Arc<NoSubtitles>, std::path::PathBuf, Value, String) {
        let dir = crate::media_test_util::tmp_dir(name);
        let image = dir.join("source.png");
        image::save_buffer(&image, &[140; 80 * 120 * 3], 80, 120, image::ColorType::Rgb8).unwrap();
        let services = std::sync::Arc::new(NoSubtitles(std::sync::atomic::AtomicBool::new(true)));
        let mut s = Session { services: services.clone(), ..Default::default() };
        let item = crate::commands::import_streamed(&mut s, image.to_str().unwrap(), None).unwrap();
        s.execute("file.newSequence", json!({"width":80,"height":120,"fps":30,"video":1,"audio":1})).unwrap();
        s.execute("timeline.place", json!({"item":item.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
        let track = s.execute("captions.newTrack", json!({"format":"Subtitle","name":"English"})).unwrap()["track"].clone();
        s.execute("captions.add", json!({"track":track,"text":"Hello","seconds":0.1,"durationSeconds":0.5})).unwrap();
        let video = dir.join("delivery.mp4");
        let p = json!({"path":video,"settings":{"frameSize":[80,120]},"qa":{"privacy":false,"narration":false,"mix":false}});
        let failed = s.execute("delivery.prepare", p.clone()).unwrap_err().to_string();
        assert!(video.exists(), "{failed}");
        assert!(failed.contains("disk is full"), "{failed}");
        (s, services, video, p, failed)
    }
    fn manifest(video: &std::path::Path) -> Value {
        serde_json::from_slice(&std::fs::read(format!("{}.delivery.json", video.display())).unwrap()).unwrap()
    }
    /// A step that failed after the export (here a subtitle) left the master without a manifest,
    /// and the retry was refused because the master existed: the export had to be made again
    /// under a new name.
    #[test]
    fn a_delivery_that_stopped_after_its_export_is_finished_by_running_it_again() {
        let (mut s, services, video, p, failed) = unfinished("delivery-resume");
        let exported = std::fs::metadata(&video).unwrap().modified().unwrap();
        services.0.store(false, std::sync::atomic::Ordering::SeqCst);
        let r = s.execute("delivery.prepare", p.clone()).unwrap();
        assert_eq!(r["resumed"], true, "{r}");
        assert!(failed.contains("again") && failed.contains(".C1.srt"), "the failure says what failed and how to finish: {failed}");
        assert_eq!(std::fs::metadata(&video).unwrap().modified().unwrap(), exported, "the master was exported again");
        assert!(std::fs::read_to_string(format!("{}.C1.srt", video.display())).unwrap().contains("Hello"));
        let m = manifest(&video);
        assert_eq!((m["complete"].as_bool(), m["subtitles"].as_array().map(Vec::len)), (Some(true), Some(1)), "{m}");
        assert!(m["qa"].as_str().is_some_and(|report| std::path::Path::new(report).exists()), "{m}");
        assert!(s.execute("delivery.prepare", p).is_err(), "a finished delivery is not made again");
    }
    /// The master at the path is a render of the edit as it was: after another edit, the
    /// unfinished delivery is not picked up.
    #[test]
    fn an_unfinished_delivery_of_another_edit_is_not_picked_up() {
        let (mut s, services, video, p, _) = unfinished("delivery-resume-other-edit");
        services.0.store(false, std::sync::atomic::Ordering::SeqCst);
        let track = s.active_sequence().unwrap().caption_tracks[0].id.0;
        s.execute("captions.add", json!({"track":track,"text":"Later","seconds":0.7,"durationSeconds":0.2})).unwrap();
        let e = s.execute("delivery.prepare", p).unwrap_err().to_string();
        assert!(e.contains("different edit") && !e.contains("again"), "{e}");
        assert_eq!(manifest(&video)["complete"], false);
        assert!(!std::path::Path::new(&format!("{}.C1.srt", video.display())).exists());
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

    /// A saved 640×360 edit: a white photo for 3 s with a white title over it (1:1 contrast)
    /// that states a figure; the session and the project's folder.
    fn titled(name: &str) -> (Session, std::path::PathBuf) {
        let dir = crate::media_test_util::tmp_dir(name);
        let image = dir.join("white.png");
        image::save_buffer(&image, &[255; 640 * 360 * 3], 640, 360, image::ColorType::Rgb8).unwrap();
        let mut s = Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() };
        let item = crate::commands::import_streamed(&mut s, image.to_str().unwrap(), None).unwrap();
        s.execute("file.newSequence", json!({"width":640,"height":360,"fps":30,"video":2,"audio":1})).unwrap();
        s.execute("timeline.place", json!({"item":item.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(3.0).0,"audio":false})).unwrap();
        s.execute("graphics.newText", json!({"text":"Built in 600 hours","size":40,"seconds":2.0,"time":Tick::from_seconds_f64(0.5).0,"track":"V2"})).unwrap();
        s.path = Some(dir.join("edit.fcproj").display().to_string());
        (s, dir)
    }
    const QUICK: &str = r#"{"privacy":false,"narration":false,"mix":false,"motionEnergy":false}"#;
    fn quick(extra: Value) -> Value {
        let mut p: Value = serde_json::from_str(QUICK).unwrap();
        for (k, v) in extra.as_object().cloned().unwrap_or_default() {
            p[k] = v;
        }
        p
    }
    fn write_decisions(dir: &std::path::Path, waivers: Value) {
        std::fs::create_dir_all(dir.join("build")).unwrap();
        std::fs::write(dir.join("build/decisions.json"), serde_json::to_vec(&json!({"waivers":waivers})).unwrap()).unwrap();
    }
    fn findings_of<'a>(report: &'a Value, name: &str) -> Vec<&'a Value> {
        check(report, name)["detail"]["findings"].as_array().map(|a| a.iter().collect()).unwrap_or_default()
    }

    /// Every check carries its tier, and the summary counts each tier's open findings.
    #[test]
    fn every_check_has_a_tier_and_the_summary_counts_them() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let r = s.execute(CMD, json!({"privacy":false,"narration":false,"mix":false})).unwrap();
        let checks = r["checks"].as_array().unwrap();
        for c in checks {
            assert!(matches!(c["tier"].as_str(), Some("floor" | "reflex" | "finish")), "{c}");
        }
        for (name, tier) in [
            ("sources", "floor"),
            ("audioContinuity", "floor"),
            ("mix", "floor"),
            ("privacyAtCuts", "floor"),
            ("changedNarration", "floor"),
            ("safeZones", "floor"),
            ("screenProtection", "floor"),
            ("textContrast", "floor"),
            ("figures", "floor"),
            ("waivers", "floor"),
            ("motion", "reflex"),
            ("colour", "reflex"),
            ("blackFrames", "reflex"),
            ("craft", "reflex"),
            ("motionCraft", "reflex"),
            ("copy", "reflex"),
            ("motionEnergy", "finish"),
            ("subtitleAlignment", "finish"),
        ] {
            assert_eq!(check(&r, name)["tier"], tier, "{name}");
        }
        // an unsaved project has no ledger and no decisions file
        assert_eq!(check(&r, "figures")["state"], "skipped");
        assert_eq!(check(&r, "waivers")["state"], "pass");
        let summary = &r["summary"];
        let total: u64 = ["floor", "reflex", "finish", "waived"].iter().map(|k| summary[k].as_u64().unwrap()).sum();
        let open = checks.iter().filter(|c| !matches!(c["state"].as_str(), Some("pass" | "skipped"))).count() as u64;
        assert!(total >= open, "every open check is counted at least once: {summary} {r:#}");
        assert_eq!(r["floorBlocks"], false);
    }

    /// The new Floor checks (contrast under 3:1, figures not in the ledger) warn by default and
    /// block with `floorBlocks`; every other check comes out the same either way.
    #[test]
    fn new_floor_checks_warn_by_default_and_block_with_floor_blocks() {
        let (mut s, _) = titled("delivery-floor");
        let off = s.execute(CMD, quick(json!({"facts":["300 hours"]}))).unwrap();
        for name in ["textContrast", "figures"] {
            let c = check(&off, name);
            assert_eq!((c["state"].as_str(), c["tier"].as_str()), (Some("warning"), Some("floor")), "{name}: {off:#}");
            assert!(!findings_of(&off, name).is_empty(), "{name}: {off:#}");
            assert!(findings_of(&off, name).iter().all(|f| f["tier"] == "floor"));
        }
        assert_eq!(findings_of(&off, "figures")[0]["figure"], "600");
        assert!(findings_of(&off, "textContrast")[0]["ratio"].as_f64().unwrap() < FLOOR_CONTRAST);
        assert_eq!(off["passed"], true, "nothing new blocks by default: {off:#}");
        assert!(off["blocking"].as_array().unwrap().is_empty());
        assert!(off["summary"]["floor"].as_u64().unwrap() >= 2, "{}", off["summary"]);
        let on = s.execute(CMD, quick(json!({"facts":["300 hours"],"floorBlocks":true}))).unwrap();
        assert_eq!(on["passed"], false);
        assert_eq!(on["blocking"], json!(["textContrast", "figures"]), "{on:#}");
        for c in off["checks"].as_array().unwrap().iter().filter(|c| c["check"] != "textContrast" && c["check"] != "figures") {
            let name = c["check"].as_str().unwrap();
            assert_eq!(check(&on, name)["state"], c["state"], "{name} changed with floorBlocks");
        }
        // the figure in the ledger passes; the leg off is recorded as off
        let r = s.execute(CMD, quick(json!({"facts":["600 hours"],"floorBlocks":true,"craft":false}))).unwrap();
        assert_eq!((check(&r, "figures")["state"].as_str(), check(&r, "textContrast")["state"].as_str()), (Some("pass"), Some("skipped")));
        assert_eq!(r["passed"], true, "{r:#}");
    }

    /// Blocking that came before the tiers is untouched by them: a face at a cut still fails the
    /// gate with `floorBlocks` off, and a waiver cannot clear it.
    #[test]
    fn existing_blocking_checks_still_block_and_are_never_waived() {
        let dir = crate::media_test_util::tmp_dir("delivery-existing");
        let mut s = Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() };
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.path = Some(dir.join("edit.fcproj").display().to_string());
        s.redacts.detector = Some(std::sync::Arc::new(|_, _, _, _| {
            vec![filmcraft_vision::Detection {
                kind: filmcraft_vision::Kind::Face,
                score: 0.99,
                bbox: [200.0, 200.0, 160.0, 160.0],
                points: Vec::new(),
                pose: false,
            }]
        }));
        write_decisions(&dir, json!([{"check":"privacyAtCuts","reason":"a passer-by, blurred later","evidence":"reviewed at 1:1"}]));
        let r = s.execute(CMD, json!({"narration":false,"mix":false,"motionEnergy":false})).unwrap();
        let privacy = check(&r, "privacyAtCuts");
        assert_eq!((privacy["state"].as_str(), privacy["tier"].as_str()), (Some("fail"), Some("floor")), "{r:#}");
        assert_eq!(r["passed"], false);
        assert!(r["blocking"].as_array().unwrap().contains(&json!("privacyAtCuts")));
        let refused = findings_of(&r, "waivers");
        assert_eq!(refused.len(), 1, "{r:#}");
        assert!(refused[0]["message"].as_str().unwrap().contains("Floor check"), "{refused:?}");
        assert!(r["summary"]["floor"].as_u64().unwrap() >= 1);
    }

    /// Reflex findings are waived from `build/decisions.json`; a waiver without both a reason and
    /// evidence, one for a Floor or Finish check, one for no check and more than five are findings.
    #[test]
    fn waivers_are_read_applied_and_refused_when_incomplete() {
        let (mut s, dir) = titled("delivery-waivers");
        let before = s.execute(CMD, quick(json!({}))).unwrap();
        assert_eq!(check(&before, "colour")["state"], "warning", "a white photo blows its highlights: {before:#}");
        write_decisions(
            &dir,
            json!([
                {"check":"colour","reason":"a white backdrop by design","evidence":"frame sheet of 0 to 3 s reviewed"},
                {"check":"colour","reason":"no evidence"},
                {"check":"textContrast","reason":"readable enough","evidence":"looked at it"},
                {"check":"subtitleAlignment","reason":"r","evidence":"e"},
                {"check":"nothing","reason":"r","evidence":"e"},
                {"check":"craft","code":"silence","time":-4,"reason":"r","evidence":"e"},
            ]),
        );
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        let colour = check(&r, "colour");
        assert_eq!(colour["state"], "waived", "{colour}");
        assert!(colour["detail"]["highlights"].as_array().unwrap().iter().all(|h| h["waived"]["evidence"] == "frame sheet of 0 to 3 s reviewed"));
        let codes: Vec<&str> = findings_of(&r, "waivers").iter().map(|f| f["code"].as_str().unwrap()).collect();
        assert_eq!(codes.iter().filter(|c| **c == "waiverRefused").count(), 5, "{r:#}");
        assert!(codes.contains(&"tooManyWaivers"), "six waivers is more than five");
        let waivers = check(&r, "waivers");
        assert_eq!((waivers["state"].as_str(), waivers["detail"]["accepted"].as_u64()), (Some("warning"), Some(1)), "{waivers}");
        assert!(r["summary"]["waived"].as_u64().unwrap() >= 1, "{}", r["summary"]);
        assert_eq!(r["passed"], true, "refused waivers warn this round");
        let r = s.execute(CMD, quick(json!({"floorBlocks":true,"craft":false,"copy":false}))).unwrap();
        assert_eq!(r["blocking"], json!(["waivers"]), "{r:#}");
        // a waiver of one finding by its time leaves the rest open
        let first = before["checks"].as_array().unwrap().iter().find(|c| c["check"] == "colour").unwrap()["detail"]["highlights"][0]["time"].as_i64().unwrap();
        write_decisions(&dir, json!([{"check":"colour","time":first,"reason":"the opening frame","evidence":"frame 0 reviewed"}]));
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        let highlights = check(&r, "colour")["detail"]["highlights"].as_array().unwrap().clone();
        assert!(highlights.len() > 1 && highlights[0]["waived"].is_object() && highlights.iter().skip(1).any(|h| h["waived"].is_null()), "{highlights:?}");
        assert_eq!(check(&r, "colour")["state"], "warning");
        assert_eq!(check(&r, "waivers")["state"], "pass");
        // a damaged file, or a named one that is not there
        std::fs::write(dir.join("build/decisions.json"), b"{not json").unwrap();
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        assert_eq!(findings_of(&r, "waivers")[0]["code"], "decisionsUnreadable", "{r:#}");
        assert_eq!(check(&r, "colour")["state"], "warning");
        let r = s.execute(CMD, quick(json!({"decisions": dir.join("elsewhere.json")}))).unwrap();
        assert_eq!(findings_of(&r, "waivers")[0]["code"], "decisionsUnreadable", "{r:#}");
    }

    /// `sequence.checkMotion` is a Reflex leg: its findings carry the tier, a waiver with a reason
    /// and evidence clears them, and `motionCraft: false` skips it.
    #[test]
    fn motion_craft_is_a_reflex_leg_that_a_waiver_clears() {
        let (mut s, dir) = titled("delivery-motion-craft");
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        assert_eq!(check(&r, "motionCraft")["state"], "pass", "a plain title moves quietly: {:#}", check(&r, "motionCraft"));
        let clip = s.active_sequence().unwrap().video_tracks[1].items[0].id.0;
        s.execute("graphics.animateText", json!({"clip":clip,"in":{"kind":"pop","easing":"Back Out"}})).unwrap();
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        let c = check(&r, "motionCraft");
        assert_eq!((c["state"].as_str(), c["tier"].as_str()), (Some("warning"), Some("reflex")), "{c:#}");
        let found = findings_of(&r, "motionCraft");
        assert!(found.iter().any(|f| f["code"] == "overshoot") && found.iter().all(|f| f["tier"] == "reflex"), "{found:?}");
        assert_eq!(r["passed"], true, "a Reflex warning never blocks");
        assert!(r["summary"]["reflex"].as_u64().unwrap() >= 1);
        // waived with a reason and evidence; one without evidence is refused
        write_decisions(
            &dir,
            json!([{"check":"motionCraft","code":"overshoot","reason":"the brand's logo sting bounces","evidence":"filmstrip of the title reviewed"}]),
        );
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        assert_eq!(check(&r, "motionCraft")["state"], "waived", "{r:#}");
        assert!(findings_of(&r, "motionCraft").iter().all(|f| f["waived"]["evidence"] == "filmstrip of the title reviewed"));
        write_decisions(&dir, json!([{"check":"motionCraft","reason":"meant"}]));
        let r = s.execute(CMD, quick(json!({}))).unwrap();
        assert_eq!(check(&r, "motionCraft")["state"], "warning");
        assert_eq!(findings_of(&r, "waivers")[0]["code"], "waiverRefused");
        let r = s.execute(CMD, quick(json!({"motionCraft":false}))).unwrap();
        assert_eq!(check(&r, "motionCraft")["state"], "skipped");
    }

    #[test]
    fn hostile_tier_parameters_are_refused() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        for (p, want) in [
            (json!({"floorBlocks":"yes"}), "floorBlocks"),
            (json!({"craft":1}), "craft"),
            (json!({"motionCraft":"off"}), "motionCraft"),
            (json!({"copy":"no"}), "copy"),
            (json!({"decisions":5}), "decisions"),
            (json!({"decisions":"a\u{0}b"}), "decisions"),
            (json!({"facts":"600"}), "facts"),
            (json!({"allow":[1]}), "allow"),
        ] {
            let e = s.execute(CMD, p.clone()).unwrap_err().to_string();
            assert!(e.contains(want), "{p}: {e}");
        }
    }
}

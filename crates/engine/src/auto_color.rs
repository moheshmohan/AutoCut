//! A careful colour pass (`clip.autoColor`): each clip's exposure and white balance measured
//! from its own picture and gently corrected, the cameras of an edit brought together, an
//! optional look of our own at low strength — and a check that backs a correction off when it
//! makes skin or greys worse.
//!
//! - **Measured** on a few small renders of the clip's own picture (the GPU's frame renderer
//!   where one is registered, else the CPU's), the other clips of the sequence switched off and
//!   any earlier pass of this one taken away: [`filmcraft_render::color_balance`] finds the
//!   greys, the skin and the median luma, and the Lumetri Basic Correction that balances them,
//!   `strength` of the way (0.7 by default) and capped.
//! - **Matched**: with `match` (on by default) and two clips or more, each clip's saturation and
//!   contrast move half way to the edit's middle — how vivid and how contrasty, never a hue, so
//!   an Insta360 and a phone sit together without one's colour being pushed onto the other.
//! - **Looks** (`look`, none by default): `punchy` (contrast and vibrance), `soft` (faded,
//!   gentler), and the Lumetri looks `warm`, `cool`, `golden`, `teal` — all our own procedural
//!   looks, no LUT or preset from anyone else — at `lookStrength` % (25 by default).
//! - **Checked**: the clip is rendered again with the correction and measured: when its greys
//!   came out more coloured, its skin moved off the skin-tone line, or highlights clipped or
//!   shadows crushed, the correction is halved, then its white balance dropped, then dropped
//!   altogether — and the report says why.
//!
//! An `offset` (`temperature`, `exposure`, `saturation`) is added to what was measured — a
//! revision's "warmer at the end" — and checked like the rest.
//!
//! The correction is a Lumetri Color of its own at the end of the clip's effects, marked with an
//! `auto_color` parameter, so the clip's own grading is never touched and the pass is easy to
//! switch off (`off`: removed; or the effect's switch in Effect Controls) or to change by hand.
//! One undo step. `sheet` writes a picture of each clip before and after.

use std::sync::Arc;

use filmcraft_project::{ClipId, ItemKind, Param, ParamValue, Project, TrackKind};
use filmcraft_render::color_balance::{self, Correction, Stats};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, clips_p, has_seq, str_p, u64_p};
use crate::{EngineError, Result, Session};

pub const AUTO_COLOR_CMD: &str = "clip.autoColor";
/// The parameter that marks the pass's own Lumetri Color.
pub const MARK: &str = "auto_color";
/// Longest side of the pictures measured, pixels.
const MEASURE_SIDE: f64 = 320.0;
/// Most clips in one pass, and rows on a sheet.
const MAX_CLIPS: usize = 200;
const MAX_SHEET_ROWS: usize = 24;
/// How far towards the edit's middle a clip's saturation and contrast move.
const MATCH: f64 = 0.5;

/// Our looks: the Lumetri look (`None`: no look) and the parameters each brings.
const LOOKS: &[(&str, Option<u32>, &[(&str, f64)])] = &[
    ("none", None, &[]),
    ("punchy", None, &[("contrast", 6.0), ("vibrance", 12.0)]),
    ("soft", None, &[("faded_film", 12.0), ("contrast", -4.0)]),
    ("warm", Some(2), &[]),
    ("cool", Some(3), &[]),
    ("golden", Some(7), &[]),
    ("teal", Some(1), &[]),
];

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: AUTO_COLOR_CMD,
        label: "Auto Color",
        menu: &["Clip"],
        shortcut: None,
        params: r#"{"clips":[id]?,"strength":0..1=0.7,"match":bool=true,"look":"none"|"punchy"|"soft"|"warm"|"cool"|"golden"|"teal"="none","lookStrength":0..100=25,"offset":{"temperature":-25..25,"exposure":-0.8..0.8,"saturation":-25..25}?,"offsets":{"<clip id>":{…as offset}}?,"target":"sequence|sdr|pq|hlg"="sequence","protectScreens":{"<clip id>":[[x,y,width,height]]}?,"frames":1..5=3,"sheet":path?,"off":bool=false}"#,
        enabled: has_seq,
        run: auto_color,
        journal: true,
    }]
}

/// Whether an effect is this pass's own Lumetri Color.
pub fn is_pass(e: &filmcraft_project::EffectInstance) -> bool {
    e.effect == "lumetri" && e.params.contains_key(MARK)
}

/// What one clip measured and was given.
struct Clip {
    id: ClipId,
    times: Vec<Tick>,
    before: Stats,
    before_pic: Option<filmcraft_render::Image>,
    correction: Correction,
    said: Vec<String>,
    after: Option<Stats>,
    after_pic: Option<filmcraft_render::Image>,
    backed_off: Option<String>,
    screens: Vec<[f64; 4]>,
    source: Value,
    size: (u32, u32),
    hdr_target: bool,
}

/// Renders a sequence frame: the GPU's frame renderer when this host has one, else the CPU.
struct Renderer(Option<Box<dyn filmcraft_export::FrameRenderer>>);

impl Renderer {
    fn render(
        &mut self,
        project: &Project,
        seq: filmcraft_project::ItemId,
        t: Tick,
        scale: f32,
        sources: &dyn filmcraft_render::SourceProvider,
    ) -> filmcraft_render::Image {
        let opts = filmcraft_render::RenderOptions { scale, ..Default::default() };
        if let Some(r) = self.0.as_mut()
            && let Some(img) = r.render(project, seq, t, opts, sources)
        {
            return img;
        }
        filmcraft_render::render_sequence(project, seq, t, opts, sources)
    }
}

/// `project` with only `clip` showing in sequence `seq` (the other video clips off), its pass
/// taken away and `pass` (a correction's Lumetri) put on instead.
fn probe(project: &Project, seq: filmcraft_project::ItemId, clip: ClipId, pass: Option<&filmcraft_project::EffectInstance>) -> Project {
    let mut p = project.clone();
    if let Some(q) = p.sequence_mut(seq) {
        for t in q.video_tracks.iter_mut() {
            for it in t.items.iter_mut() {
                if it.id == clip {
                    it.effects.retain(|e| !is_pass(e));
                    if let Some(e) = pass {
                        it.effects.push(e.clone());
                    }
                } else {
                    it.enabled = false;
                }
            }
        }
    }
    p
}

/// The frames of `times` stacked into one picture.
fn stack(pics: &[filmcraft_render::Image]) -> Option<filmcraft_render::Image> {
    let first = pics.first()?;
    let (w, h) = (first.w, first.h);
    let same: Vec<&filmcraft_render::Image> = pics.iter().filter(|p| p.w == w && p.h == h).collect();
    let mut out = filmcraft_render::Image::new(w, h.checked_mul(same.len())?);
    for (k, p) in same.iter().enumerate() {
        let at = k.checked_mul(w * h * 4)?;
        out.px.get_mut(at..at + p.px.len())?.copy_from_slice(&p.px);
    }
    Some(out)
}

/// A Lumetri Color holding `c`, the look and its strength, marked as the pass's.
fn pass_effect(c: &Correction, look: &(&str, Option<u32>, &[(&str, f64)]), look_strength: f64) -> Option<filmcraft_project::EffectInstance> {
    let def = filmcraft_project::effect::effect_defs().iter().find(|d| d.id == "lumetri")?;
    let mut e = def.instance();
    let mut set = |k: &str, v: ParamValue| {
        e.params.insert(k.to_string(), Param::new(v));
    };
    set("temperature", ParamValue::Float(c.temperature));
    set("tint", ParamValue::Float(c.tint));
    set("exposure", ParamValue::Float(c.exposure));
    set("saturation", ParamValue::Float(c.saturation));
    set("contrast", ParamValue::Float(c.contrast));
    if let Some(choice) = look.1 {
        set("look", ParamValue::Choice(choice));
        set("look_intensity", ParamValue::Float(look_strength));
    }
    for (k, v) in look.2 {
        let base = if *k == "contrast" { c.contrast } else { 0.0 };
        set(k, ParamValue::Float(base + v * look_strength / 25.0));
    }
    set(MARK, ParamValue::Text("measured".into()));
    Some(e)
}

fn stats_json(s: &Stats) -> Value {
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    json!({
        "medianLuma": r(s.median), "greyCast": r(s.cast), "greyShare": r(s.grey_share),
        "skinHue": s.skin_hue.map(|h| h.round()), "skinShare": r(s.skin_share),
        "saturation": r(s.saturation), "contrast": r(s.contrast), "blown": r(s.blown), "dark": r(s.dark),
    })
}

fn auto_color(s: &mut Session, p: &Value) -> Result<Value> {
    let target = str_p(p, "target").unwrap_or("sequence");
    let working = match target {
        "sequence" => None,
        "sdr" => Some(filmcraft_color::WorkingSpace::Rec709),
        "pq" => Some(filmcraft_color::WorkingSpace::Rec2100Pq),
        "hlg" => Some(filmcraft_color::WorkingSpace::Rec2100Hlg),
        _ => return Err(bad(AUTO_COLOR_CMD, "target is sequence, sdr, pq or hlg")),
    };
    s.edit_group("Source-aware Auto Color", |s| {
        if let Some(working) = working {
            s.edit_sequence("Set colour target", |q, _, _| {
                q.settings.color.working = working;
                q.settings.color.auto_tone_map = true;
                Ok(())
            })?;
        }
        auto_color_inner(s, p)
    })
}

/// Explicit rectangles are normalized original-source coordinates. They become ordinary
/// editable subtract masks before Motion, and survive save/reopen and repeated passes.
fn screens(value: &Value) -> Result<Vec<[f64; 4]>> {
    if value.is_null() {
        return Ok(Vec::new());
    }
    let values = value.as_array().filter(|a| a.len() <= 8).ok_or_else(|| bad(AUTO_COLOR_CMD, "protectScreens needs up to 8 normalized rectangles per clip"))?;
    values
        .iter()
        .map(|v| {
            let r: [f64; 4] = serde_json::from_value(v.clone()).map_err(|_| bad(AUTO_COLOR_CMD, "a screen is [x,y,width,height]"))?;
            let [x, y, w, h] = r;
            if !r.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) || w <= 0.0 || h <= 0.0 || x + w > 1.000001 || y + h > 1.000001 {
                return Err(bad(AUTO_COLOR_CMD, "screen bounds must lie inside the source"));
            }
            Ok(r)
        })
        .collect()
}
fn protected_pass(
    c: &Correction,
    look: &(&str, Option<u32>, &[(&str, f64)]),
    strength: f64,
    screens: &[[f64; 4]],
    size: (u32, u32),
) -> Option<filmcraft_project::EffectInstance> {
    let mut e = pass_effect(c, look, strength)?;
    for (i, &[x, y, w, h]) in screens.iter().enumerate() {
        let (sw, sh) = (f64::from(size.0), f64::from(size.1));
        let mut m =
            filmcraft_project::Mask::new(format!("Protect screen {}", i + 1), filmcraft_project::MaskPath::rect(x * sw, y * sh, (x + w) * sw, (y + h) * sh));
        m.inverted = i == 0;
        if i > 0 {
            m.mode = filmcraft_project::MaskMode::Subtract;
        }
        m.feather = Param::new(ParamValue::Float((sw.min(sh) * 0.003).clamp(0.0, 12.0)));
        e.masks.push(m);
    }
    if !screens.is_empty() {
        e.params.insert("auto_color_screens".into(), Param::new(ParamValue::Text(serde_json::to_string(screens).ok()?)));
    }
    Some(e)
}

fn auto_color_inner(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = AUTO_COLOR_CMD;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let number = |k: &str, default: f64, max: f64, what: &str| -> Result<f64> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|x| x.is_finite() && (0.0..=max).contains(x)).ok_or_else(|| bad(cmd, format!("`{k}` is {what}"))),
        }
    };
    let strength = number("strength", 0.7, 1.0, "0 to 1")?;
    let look_name = str_p(p, "look").map(|l| l.trim().to_ascii_lowercase()).unwrap_or_else(|| "none".into());
    let look = *LOOKS.iter().find(|l| l.0 == look_name).ok_or_else(|| bad(cmd, "`look` is none, punchy, soft, warm, cool, golden or teal"))?;
    let look_strength = number("lookStrength", 25.0, 100.0, "0 to 100 (%)")?;
    let frames = u64_p(p, "frames").unwrap_or(3).clamp(1, 5) as usize;
    // an offset asked for over what is measured: for every clip, or `offsets` per clip
    let offset_of = |o: &Value, what: &str| -> Result<Correction> {
        if !o.is_object() {
            return Err(bad(cmd, format!("`{what}` is {{temperature, exposure, saturation}}")));
        }
        let get = |k: &str, max: f64| -> Result<f64> {
            match o.get(k).filter(|v| !v.is_null()) {
                None => Ok(0.0),
                Some(v) => v.as_f64().filter(|x| x.is_finite() && x.abs() <= max).ok_or_else(|| bad(cmd, format!("`{what}.{k}` is −{max} to {max}"))),
            }
        };
        Ok(Correction {
            temperature: get("temperature", 25.0)?,
            exposure: get("exposure", 0.8)?,
            saturation: 100.0 + get("saturation", 25.0)?,
            ..Correction::default()
        })
    };
    let offset = match p.get("offset").filter(|v| !v.is_null()) {
        None => Correction::default(),
        Some(o) => offset_of(o, "offset")?,
    };
    let mut offsets: std::collections::HashMap<u64, Correction> = std::collections::HashMap::new();
    if let Some(m) = p.get("offsets").filter(|v| !v.is_null()) {
        let m = m.as_object().ok_or_else(|| bad(cmd, "`offsets` maps clip ids to {temperature, exposure, saturation}"))?;
        for (k, v) in m.iter().take(MAX_CLIPS) {
            let id: u64 = k.parse().map_err(|_| bad(cmd, format!("`offsets`: `{k}` is not a clip id")))?;
            offsets.insert(id, offset_of(v, "offsets")?);
        }
    }
    if let Some(v) = p.get("protectScreens") {
        let entries = v.as_object().filter(|m| m.len() <= MAX_CLIPS).ok_or_else(|| bad(cmd, "protectScreens maps clip ids to normalized rectangles"))?;
        for (id, value) in entries {
            id.parse::<u64>().map_err(|_| bad(cmd, "protectScreens key is a clip identity"))?;
            screens(value)?;
        }
    }
    let matching = bool_p(p, "match").unwrap_or(true);
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    // the clips: given, else selected, else every picture clip of the sequence
    let mut ids = clips_p(s, p);
    if ids.is_empty() && p.get("clips").is_none() && p.get("clip").is_none() {
        ids = seq.video_tracks.iter().flat_map(|t| &t.items).map(|i| i.id).collect();
    }
    let picture = |c: &ClipId| {
        seq.find_item(*c).is_some_and(|(t, it)| {
            seq.track(t).is_some_and(|tr| tr.kind == TrackKind::Video) && s.project.item(it.item).is_some_and(|pi| matches!(pi.kind, ItemKind::Media(_)))
        })
    };
    if let Some(c) = ids.iter().find(|c| seq.find_item(**c).is_none()) {
        return Err(bad(cmd, format!("no clip {} in the active sequence", c.0)));
    }
    let ids: Vec<ClipId> = ids.into_iter().filter(picture).take(MAX_CLIPS).collect();
    if ids.is_empty() {
        return Err(bad(cmd, "no picture clip given, selected or in the sequence"));
    }
    if bool_p(p, "off").unwrap_or(false) {
        let n = s.edit_sequence("Auto Color Off", |q, _, _| {
            let mut n = 0;
            for c in &ids {
                if let Some((_, it)) = q.find_item_mut(*c) {
                    let before = it.effects.len();
                    it.effects.retain(|e| !is_pass(e));
                    n += before - it.effects.len();
                }
            }
            Ok(n)
        })?;
        return Ok(json!({"clips": ids.iter().map(|c| c.0).collect::<Vec<_>>(), "removed": n}));
    }
    let (fw, fh) = (f64::from(seq.settings.width.max(1)), f64::from(seq.settings.height.max(1)));
    let scale = (MEASURE_SIDE / fw.max(fh)).clamp(0.02, 1.0) as f32;
    let project: Arc<Project> = s.project.clone();
    let provider = s.media.provider(project.clone(), s.services.clone());
    let mut renderer = Renderer(filmcraft_export::frame_renderer());
    let gpu = renderer.0.is_some();
    // ---- measured as they are
    let mut clips: Vec<Clip> = Vec::new();
    for c in &ids {
        let Some((_, it)) = seq.find_item(*c) else { continue };
        // the middle of the clip, clear of transitions at its ends
        let times: Vec<Tick> = (0..frames)
            .map(|k| {
                let f = 0.2 + 0.6 * (k as f64 + 0.5) / frames as f64;
                it.start + Tick((it.duration.0 as f64 * f) as i64)
            })
            .collect();
        let media = project.item(it.item).and_then(|i| i.as_media());
        let video = media.and_then(|m| m.info.video.as_ref());
        let space = media.and_then(|m| m.interpret.color_space).or_else(|| video.map(|v| filmcraft_color::ColorSpace::from_info(&v.color)));
        let size = video.map_or((1, 1), |v| (v.width, v.height));
        let hdr_target = seq.settings.color.working.is_hdr();
        let given = p.get("protectScreens").and_then(|m| m.get(c.0.to_string()));
        let saved = it.effects.iter().find(|e| is_pass(e)).and_then(|e| e.params.get("auto_color_screens")).and_then(|p| match &p.value {
            ParamValue::Text(v) => serde_json::from_str::<Value>(v).ok(),
            _ => None,
        });
        let screens = screens(given.or(saved.as_ref()).unwrap_or(&Value::Null))?;
        let source = json!({"space":space,"transfer":video.map(|v| v.color.transfer),"primaries":video.map(|v| v.color.primaries),"camera":project.item(it.item).map(|m| json!({"make":m.metadata.get("Make"),"model":m.metadata.get("Model")})),"target":seq.settings.color.working,"toneMap":seq.settings.color.auto_tone_map && !hdr_target,"protection":"source-space masks (track moving screens with masks.track)"});
        let before_project = probe(&project, seq_id, *c, None);
        let pics: Vec<filmcraft_render::Image> = times.iter().map(|t| renderer.render(&before_project, seq_id, *t, scale, &provider)).collect();
        let Some(all) = stack(&pics) else { continue };
        let before = color_balance::measure(&all);
        let (mut correction, mut said) = color_balance::balance(&before, strength);
        if hdr_target {
            correction.exposure = 0.0;
            said.push("HDR target: retain scene exposure; use an explicit offset to change it".into());
        }
        let offset = offsets.get(&c.0).copied().unwrap_or(offset);
        if !offset.is_none() {
            correction.temperature = (correction.temperature + offset.temperature).clamp(-50.0, 50.0);
            correction.exposure = (correction.exposure + offset.exposure).clamp(-1.5, 1.5);
            correction.saturation = (correction.saturation + offset.saturation - 100.0).clamp(60.0, 140.0);
            said.push(format!(
                "asked for: temperature {:+.0}, exposure {:+.2}, saturation {:+.0}",
                offset.temperature,
                offset.exposure,
                offset.saturation - 100.0
            ));
        }
        clips.push(Clip {
            id: *c,
            times,
            before,
            before_pic: pics.get(pics.len() / 2).cloned(),
            correction,
            said,
            after: None,
            after_pic: None,
            backed_off: None,
            screens,
            source,
            size,
            hdr_target,
        });
    }
    // ---- the cameras matched: saturation and contrast half way to the edit's middle
    if matching && clips.len() >= 2 {
        let median = |mut v: Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let target_s = median(clips.iter().map(|c| c.before.saturation).collect());
        let target_c = median(clips.iter().map(|c| c.before.contrast).collect());
        for c in clips.iter_mut() {
            if c.hdr_target {
                continue;
            }
            let (sat, con) = color_balance::match_to(&c.before, target_s, target_c, MATCH);
            let asked = offsets.get(&c.id.0).copied().unwrap_or(offset);
            if (sat - 100.0).abs() >= 1.0 || con.abs() >= 1.0 {
                c.correction.saturation = (sat + asked.saturation - 100.0).clamp(60.0, 140.0);
                c.correction.contrast = con;
                c.said.push(format!("matched to the edit: saturation {sat:.0}, contrast {con:+.0}"));
            }
        }
    }
    // ---- checked: rendered with the correction, backed off where it did harm
    for c in clips.iter_mut() {
        if c.correction.is_none() && look.1.is_none() && look.2.is_empty() {
            continue;
        }
        let tries = [c.correction, c.correction.scaled(0.5), c.correction.scaled(0.5).without_white_balance(), Correction::default()];
        for (k, attempt) in tries.iter().enumerate() {
            let Some(e) = protected_pass(attempt, &look, look_strength, &c.screens, c.size) else { break };
            let after_project = probe(&project, seq_id, c.id, Some(&e));
            let pics: Vec<filmcraft_render::Image> = c.times.iter().map(|t| renderer.render(&after_project, seq_id, *t, scale, &provider)).collect();
            let Some(all) = stack(&pics) else { break };
            let after = color_balance::measure(&all);
            let harm = color_balance::drift(&c.before, &after);
            c.after = Some(after);
            c.after_pic = pics.get(pics.len() / 2).cloned();
            match harm {
                None => {
                    if k > 0 {
                        c.correction = *attempt;
                    }
                    break;
                }
                Some(why) => {
                    c.backed_off = Some(why);
                    c.correction = *attempt;
                }
            }
        }
    }
    // ---- written: one undo step
    let passes: Vec<(ClipId, Option<filmcraft_project::EffectInstance>)> = clips
        .iter()
        .map(|c| {
            let keep = !c.screens.is_empty() || !(c.correction.is_none() && look.1.is_none() && look.2.is_empty());
            (c.id, if keep { protected_pass(&c.correction, &look, look_strength, &c.screens, c.size) } else { None })
        })
        .collect();
    s.edit_sequence("Auto Color", |q, _, _| {
        for (c, e) in &passes {
            if let Some((_, it)) = q.find_item_mut(*c) {
                it.effects.retain(|x| !is_pass(x));
                if let Some(e) = e {
                    it.effects.push(e.clone());
                }
            }
        }
        Ok(())
    })?;
    let report: Vec<Value> = clips
        .iter()
        .map(|c| {
            let k = &c.correction;
            json!({
                "clip": c.id.0,
                "source":c.source,"protectedScreens":c.screens,
                "before": stats_json(&c.before),
                "after": c.after.as_ref().map(stats_json),
                "applied": {"temperature": k.temperature, "tint": k.tint, "exposure": k.exposure, "saturation": k.saturation, "contrast": k.contrast, "look": look.0},
                "did": c.said,
                "backedOff": c.backed_off,
            })
        })
        .collect();
    let mut out = json!({"clips": report, "gpu": gpu, "look": look.0, "strength": strength});
    if let Some(path) = str_p(p, "sheet").map(str::trim).filter(|x| !x.is_empty()) {
        let rows: Vec<Vec<(filmcraft_render::Image, String)>> = clips
            .iter()
            .take(MAX_SHEET_ROWS)
            .filter_map(|c| {
                let b = c.before_pic.clone()?;
                let a = c.after_pic.clone().unwrap_or_else(|| b.clone());
                Some(vec![(b, format!("clip {} before", c.id.0)), (a, format!("after{}", if c.backed_off.is_some() { " (backed off)" } else { "" }))])
            })
            .collect();
        out["sheet"] = crate::contact_sheet::write_images(s, cmd, &rows, path)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    fn picture_clips(s: &Session) -> Vec<ClipId> {
        let q = s.active_sequence().unwrap();
        q.video_tracks
            .iter()
            .flat_map(|t| &t.items)
            .filter(|i| s.project.item(i.item).is_some_and(|p| matches!(p.kind, ItemKind::Media(_))))
            .map(|i| i.id)
            .collect()
    }

    fn passes(s: &Session, c: ClipId) -> Vec<filmcraft_project::EffectInstance> {
        let q = s.active_sequence().unwrap();
        q.find_item(c).unwrap().1.effects.iter().filter(|e| is_pass(e)).cloned().collect()
    }

    fn float(e: &filmcraft_project::EffectInstance, k: &str) -> f64 {
        match e.params.get(k).map(|p| &p.value) {
            Some(ParamValue::Float(v)) => *v,
            _ => f64::NAN,
        }
    }

    #[test]
    fn a_warm_clip_is_cooled_its_own_grading_kept_and_the_pass_switched_off() {
        let mut s = demo();
        let clips = picture_clips(&s);
        let warm = clips[0];
        // the user's own grade: far too warm
        s.execute("effects.apply", json!({"clips": [warm.0], "effect": "lumetri"})).unwrap();
        s.execute("effects.setParam", json!({"clip": warm.0, "effect": "lumetri", "param": "temperature", "value": 70.0})).unwrap();
        let r = s.execute(AUTO_COLOR_CMD, json!({"clips": [warm.0], "match": false})).unwrap();
        let row = &r["clips"][0];
        assert_eq!(row["clip"], warm.0, "{r}");
        let pass = passes(&s, warm);
        assert_eq!(pass.len(), 1, "{r}");
        assert!(float(&pass[0], "temperature") < -3.0 || row["backedOff"].is_string(), "{r}");
        assert!(float(&pass[0], "temperature").abs() <= color_balance::MAX_TEMPERATURE && float(&pass[0], "exposure").abs() <= color_balance::MAX_STOPS);
        // the user's Lumetri is as they left it, first; the pass after it
        let q = s.active_sequence().unwrap();
        let effects = &q.find_item(warm).unwrap().1.effects;
        let mine = effects.iter().position(|e| e.effect == "lumetri" && !is_pass(e)).unwrap();
        assert!(effects.iter().position(is_pass).unwrap() > mine);
        // run again: still one pass; off: none; undo: back
        s.execute(AUTO_COLOR_CMD, json!({"clips": [warm.0], "match": false})).unwrap();
        assert_eq!(passes(&s, warm).len(), 1);
        let r = s.execute(AUTO_COLOR_CMD, json!({"clips": [warm.0], "off": true})).unwrap();
        assert_eq!(r["removed"], 1);
        assert!(passes(&s, warm).is_empty());
        s.undo().unwrap();
        assert_eq!(passes(&s, warm).len(), 1);
    }

    #[test]
    fn the_whole_sequence_is_matched_and_a_look_is_ours_at_low_strength() {
        let mut s = demo();
        let clips = picture_clips(&s);
        let dir = std::env::temp_dir().join(format!("filmcraft-autocolor-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sheet = dir.join("color.png");
        let r = s.execute(AUTO_COLOR_CMD, json!({"look": "warm", "sheet": sheet.display().to_string()})).unwrap();
        assert_eq!(r["clips"].as_array().unwrap().len(), clips.len(), "{r}");
        assert_eq!(r["gpu"], false, "no GPU renderer in unit tests: the CPU measures");
        for c in &clips {
            let pass = passes(&s, *c);
            assert_eq!(pass.len(), 1);
            assert_eq!(pass[0].params.get("look").map(|p| &p.value), Some(&ParamValue::Choice(2)));
            assert_eq!(float(&pass[0], "look_intensity"), 25.0);
            let sat = float(&pass[0], "saturation");
            assert!((88.0..=112.0).contains(&sat), "{sat}");
        }
        assert!(sheet.exists() && r["sheet"]["width"].as_u64().unwrap() > 0, "{r}");
        // warmer, as a revision asks: the offset over what was measured
        let one = clips[0];
        let before = float(&passes(&s, one)[0], "temperature");
        s.execute(AUTO_COLOR_CMD, json!({"clips": [one.0], "match": false, "offset": {"temperature": 10}})).unwrap();
        let after = passes(&s, one);
        assert!(float(&after[0], "temperature") > before.min(0.0), "{before} → {}", float(&after[0], "temperature"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_parameters_are_refused() {
        let mut s = demo();
        for bad in [
            json!({"strength": 2}),
            json!({"offset": {"temperature": 90}}),
            json!({"offset": "warm"}),
            json!({"strength": "x"}),
            json!({"look": "sepia"}),
            json!({"lookStrength": 300}),
            json!({"clips": [999_999]}),
            json!({"clips": []}),
        ] {
            assert!(s.execute(AUTO_COLOR_CMD, bad.clone()).is_err(), "{bad}");
        }
    }
}

#[cfg(test)]
mod source_aware_tests {
    use super::*;
    #[test]
    fn invalid_screen_regions_and_targets_are_atomic() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let before = s.project.clone();
        for p in [json!({"target":"invented"}), json!({"target":"pq","protectScreens":{"1":[[0.8,0.0,0.5,0.5]]}})] {
            assert!(s.execute(AUTO_COLOR_CMD, p).is_err());
            assert_eq!(&*s.project, &*before);
        }
    }
    #[test]
    fn screen_masks_are_editable_source_space_and_survive_rerun() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let id = s
            .active_sequence()
            .unwrap()
            .video_tracks
            .iter()
            .flat_map(|t| &t.items)
            .find(|c| s.project.item(c.item).is_some_and(|i| matches!(i.kind, ItemKind::Media(_))))
            .unwrap()
            .id;
        let p = json!({"clips":[id.0],"look":"punchy","frames":1,"protectScreens":{id.0.to_string():[[0.25,0.25,0.5,0.5]]}});
        s.execute(AUTO_COLOR_CMD, p).unwrap();
        s.execute(AUTO_COLOR_CMD, json!({"clips":[id.0],"look":"punchy","frames":1})).unwrap();
        let q = s.active_sequence().unwrap();
        let (_, c) = q.find_item(id).unwrap();
        let effects: Vec<_> = c.effects.iter().filter(|e| is_pass(e)).collect();
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].masks.len(), 1);
        assert!(effects[0].masks[0].inverted);
        let bytes = serde_json::to_vec(&s.project).unwrap();
        let reopened: Project = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(&reopened, &*s.project);
    }
}

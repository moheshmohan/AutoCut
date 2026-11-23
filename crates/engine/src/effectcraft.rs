//! EffectCraft templates in process (PX4): `.ectemplate` packages (Essential Graphics templates)
//! read and rendered by EffectCraft's own renderer, linked with the cargo feature `effectcraft`
//! ([`filmcraft_effectcraft`]; the packaged AutoCut app has it, workspace builds and the web do
//! not, and then every command here says so).
//!
//! | command | does |
//! |---|---|
//! | `effectcraft.templateInfo` | what a template is: its comp (size, rate, duration, frames), its controls (name, type, default, range or options), the media it carries or needs, its poster, whether it can be rendered here |
//! | `effectcraft.renderTemplate` | a job under the machine's GPU lock: the template filled with `values` and rendered with alpha (ProRes 4444 or a PNG sequence) into a new revision folder, AutoCut's probe of the result, `autocut-import.json`, then the import and, with `place`, the clip above the picture — `motion.render`'s revision folder, probe, import and placement |
//!
//! The template file is read whole, at most [`filmcraft_effectcraft::limits::MAX_TEMPLATE_BYTES`];
//! everything in it is checked before EffectCraft sees it ([`filmcraft_effectcraft`]'s docs), and
//! every value, the size, the rate, the range and the media before the job starts. Footage is
//! decoded by the engine's own decoders (a media pool with the session's openers).
//!
//! **Revision folders.** `out` is a folder that does not exist yet or is empty (else the first
//! free `<template> renders/rNNN` beside the template); a refused or failed render takes away what
//! it made (and the folder when it made it). Once rendered, the frames stay even when the probe
//! refuses them, and `autocut-import.json` is written only after it passed.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports, unused_variables))]

use std::path::Path;
use std::sync::atomic::Ordering;

use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad};
use crate::motion_graphics::{self, path_p, str_of};
use crate::{Result, Session};

const INFO: &str = "effectcraft.templateInfo";
const RENDER: &str = "effectcraft.renderTemplate";

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(INFO, "EffectCraft Template Info", r#"{"template":path (an absolute .ectemplate)}"#, template_info, false),
        spec(
            RENDER,
            "Render EffectCraft Template",
            r##"{"template":path (an absolute .ectemplate),"values":{control name or id: value}? (text: a string; colour: "#rrggbb" or [r,g,b,a?] from 0 to 1; slider, angle: a number; checkbox: true or false; point: [x,y]; dropdown: an option's index; font: a family or {"font","style","size"}; media: an absolute path; effectcraft.templateInfo lists the controls),"size":"WxH"? (default the comp's; the comp's shape),"fps":1..240? (default the comp's),"range":{"start":ticks?,"end":ticks?}? (the template's own time; default all of it),"out":path? (a new or empty revision folder; default <template> renders/rNNN beside the template),"format":"prores4444"|"png"=prores4444 (alpha either way),"gpu":bool=true (EffectCraft's GPU compositor; the CPU renders what it cannot),"import":bool=true,"place":{"track":"V2"?,"time":ticks?|"seconds":f64?}? (on the timeline; default the playhead, above V1),"wait":bool=false}""##,
            render_template,
            true,
        ),
    ]
}

/// The `template` parameter: an absolute path to a file.
fn template_p(cmd: &str, p: &Value) -> Result<std::path::PathBuf> {
    let t = path_p(cmd, p, "template")?.ok_or_else(|| bad(cmd, "`template` is needed: the absolute path of an .ectemplate"))?;
    Ok(t)
}

/// The template's bytes: refused past the cap before anything is read.
fn read_template(cmd: &str, path: &Path) -> Result<Vec<u8>> {
    use std::io::Read as _;
    let max = filmcraft_effectcraft::limits::MAX_TEMPLATE_BYTES;
    let f = std::fs::File::open(path).map_err(|e| bad(cmd, format!("{}: {e}", path.display())))?;
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len > max {
        return Err(bad(cmd, format!("{} is {} MiB; a template is at most {} MiB", path.display(), len >> 20, max >> 20)));
    }
    let mut bytes = Vec::new();
    f.take(max.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| bad(cmd, format!("{}: {e}", path.display())))?;
    if bytes.len() as u64 > max {
        return Err(bad(cmd, format!("{} grew past {} MiB while it was read", path.display(), max >> 20)));
    }
    Ok(bytes)
}

/// Refused up front in a build without EffectCraft.
fn linked(cmd: &str) -> Result<()> {
    if filmcraft_effectcraft::available() { Ok(()) } else { Err(bad(cmd, filmcraft_effectcraft::Error::NotBuilt.to_string())) }
}

fn template_info(_: &mut Session, p: &Value) -> Result<Value> {
    let path = template_p(INFO, p)?;
    linked(INFO)?;
    let bytes = read_template(INFO, &path)?;
    let mut v = filmcraft_effectcraft::template_info(&bytes).map_err(|e| bad(INFO, format!("{}: {e}", path.display())))?;
    v["path"] = json!(str_of(&path));
    v["bytes"] = json!(bytes.len());
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// effectcraft.renderTemplate

/// The staging folder inside a revision (frames, a movie being written, the template's media).
const STAGING: &str = ".effectcraft-render";
/// How long a render waits for the machine's GPU.
const GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(30 * 60);
/// Default revision folders tried next to the template.
const MAX_REVISIONS: u32 = 999;

/// What the render writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// `frames/frame-00000.png`…, straight alpha.
    Png,
    /// `<name>.mov`, ProRes 4444 with alpha.
    ProRes4444,
}

impl Format {
    fn kind(self) -> &'static str {
        match self {
            Format::Png => "pngSequence",
            Format::ProRes4444 => "prores4444",
        }
    }
}

fn format_p(cmd: &str, p: &Value) -> Result<Format> {
    match p.get("format") {
        None | Some(Value::Null) => Ok(Format::ProRes4444),
        Some(Value::String(f)) if f.eq_ignore_ascii_case("prores4444") || f.eq_ignore_ascii_case("prores") => Ok(Format::ProRes4444),
        Some(Value::String(f)) if f.eq_ignore_ascii_case("png") => Ok(Format::Png),
        Some(_) => Err(bad(cmd, "`format` is \"prores4444\" (one movie with alpha) or \"png\" (a sequence with alpha)")),
    }
}

/// `range`: `{"start": ticks?, "end": ticks?}` in the template's own time.
fn range_p(cmd: &str, p: &Value) -> Result<(Option<Tick>, Option<Tick>)> {
    let Some(r) = p.get("range").filter(|v| !v.is_null()) else { return Ok((None, None)) };
    let r = r.as_object().ok_or_else(|| bad(cmd, "`range` is {\"start\": ticks?, \"end\": ticks?} in the template's time"))?;
    let t = |k: &str| -> Result<Option<Tick>> {
        match r.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_i64().map(|t| Some(Tick(t))).ok_or_else(|| bad(cmd, format!("`range.{k}` is ticks (254016000000 a second)"))),
        }
    };
    Ok((t("start")?, t("end")?))
}

/// The revision folder: `out`, else the first free `<template folder>/<template> renders/rNNN`.
fn out_p(cmd: &str, p: &Value, template: &Path) -> Result<std::path::PathBuf> {
    if let Some(out) = path_p(cmd, p, "out")? {
        return Ok(out);
    }
    let stem = template.file_stem().and_then(|s| s.to_str()).filter(|s| !s.is_empty()).unwrap_or("template");
    let base = template.parent().unwrap_or(Path::new("/")).join(format!("{stem} renders"));
    (1..=MAX_REVISIONS)
        .map(|n| base.join(format!("r{n:03}")))
        .find(|d| !d.exists())
        .ok_or_else(|| bad(cmd, format!("{} holds {MAX_REVISIONS} renders: give `out`", base.display())))
}

/// The engine's decoders as EffectCraft's footage: each file opened once through a media pool
/// with the session's openers (and its services' file access).
struct EngineFrames {
    services: std::sync::Arc<dyn crate::Services>,
    pool: crate::media_pool::MediaPool,
    open: std::sync::Mutex<std::collections::BTreeMap<String, Option<filmcraft_media::SharedSource>>>,
}

/// Files kept open by one render.
const MAX_OPEN: usize = 256;

impl EngineFrames {
    fn new(s: &Session) -> EngineFrames {
        let pool = crate::media_pool::MediaPool::default();
        *pool.openers.write().unwrap_or_else(std::sync::PoisonError::into_inner) =
            s.media.openers.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        EngineFrames { services: s.services.clone(), pool, open: Default::default() }
    }

    fn source(&self, path: &str) -> Option<filmcraft_media::SharedSource> {
        if let Some(s) = self.open.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(path) {
            return s.clone();
        }
        // opened outside the lock: another worker reads its own file meanwhile
        let src = self.pool.open_file(path, self.services.as_ref()).ok();
        let mut open = self.open.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if open.len() >= MAX_OPEN {
            open.clear();
        }
        open.insert(path.to_string(), src.clone());
        src
    }
}

impl filmcraft_effectcraft::HostFrames for EngineFrames {
    fn probe(&self, path: &str) -> Option<filmcraft_effectcraft::HostMedia> {
        let src = self.source(path)?;
        let info = src.info();
        let v = info.video.as_ref()?;
        Some(filmcraft_effectcraft::HostMedia {
            width: v.width,
            height: v.height,
            rate: v.frame_rate.sane(),
            duration: info.duration,
            still: info.kind == filmcraft_media::MediaKind::Still,
            has_alpha: v.has_alpha,
            has_audio: !info.audio_streams.is_empty(),
        })
    }

    fn frame(&self, path: &str, t: Tick) -> Option<std::sync::Arc<filmcraft_frame::VideoFrame>> {
        self.source(path)?.video_frame(filmcraft_media::FrameRequest { time: t, scale: 1.0 }).ok()
    }
}

/// ProRes 4444 with alpha into a QuickTime movie, frame by frame.
struct MovOut {
    writer: filmcraft_isobmff::Mp4Writer<std::io::BufWriter<std::fs::File>>,
    track: usize,
    enc: filmcraft_prores::Encoder,
    size: (u32, u32),
    duration: u32,
}

impl MovOut {
    fn new(path: &Path, size: (u32, u32), rate: FrameRate) -> std::result::Result<MovOut, String> {
        use filmcraft_isobmff::{Brand, FourCc, Mp4Writer, SampleEntry, TrackConfig, WriterOptions};
        let (w, h) = size;
        let (Ok(w16), Ok(h16)) = (u16::try_from(w), u16::try_from(h)) else { return Err(format!("a {w}x{h} movie is too large")) };
        let rate = rate.sane();
        let (Ok(timescale), Ok(duration)) = (u32::try_from(rate.num), u32::try_from(rate.den)) else {
            return Err(format!("the rate {}/{} cannot be written", rate.num, rate.den));
        };
        let profile = filmcraft_prores::Profile::P4444;
        let mut cfg = filmcraft_prores::EncoderConfig::new(profile, w, h);
        cfg.encode_alpha = true;
        let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut opts = WriterOptions::new(Brand::Mov);
        opts.metadata.push(("©too".into(), format!("AutoCut with EffectCraft {}", filmcraft_effectcraft::EFFECTCRAFT_VERSION)));
        let mut writer = Mp4Writer::new(std::io::BufWriter::new(file), opts).map_err(|e| e.to_string())?;
        let mut entry = SampleEntry::prores(FourCc(profile.fourcc()), w16, h16);
        filmcraft_export::ColorSignal::default().apply_to(&mut entry, true);
        let track = writer.add_track(TrackConfig::new(entry, timescale)).map_err(|e| e.to_string())?;
        Ok(MovOut { writer, track, enc: filmcraft_prores::Encoder::with_config(cfg), size, duration })
    }

    fn write(&mut self, f: &filmcraft_effectcraft::Frame) -> std::result::Result<(), String> {
        if (f.width, f.height) != self.size {
            return Err(format!("a {}x{} frame in a {}x{} movie", f.width, f.height, self.size.0, self.size.1));
        }
        let (w, h) = (f.width as usize, f.height as usize);
        let mut fr = filmcraft_prores::Frame::new(f.width, f.height, filmcraft_prores::ChromaFormat::Yuv444, 10, true);
        filmcraft_export::to_yuv444_10(&f.rgba, None, w, h, 0.2126, 0.0722, &mut fr.y, &mut fr.cb, &mut fr.cr, fr.alpha.as_deref_mut());
        let data = self.enc.encode(&fr).map_err(|e| format!("ProRes: {e}"))?;
        self.writer
            .write_sample(self.track, filmcraft_isobmff::WriteSample { data: &data, duration: self.duration, composition_offset: 0, is_sync: true })
            .map_err(|e| e.to_string())
    }

    fn finish(self) -> std::result::Result<(), String> {
        use std::io::Write as _;
        let mut w = self.writer.finish().map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())
    }
}

/// Where the frames go.
enum Out {
    Png(std::path::PathBuf),
    Mov(MovOut),
}

impl Out {
    fn write(&mut self, i: u64, f: &filmcraft_effectcraft::Frame) -> std::result::Result<(), String> {
        match self {
            Out::Png(dir) => {
                let png = filmcraft_export::encode_png(f.rgba.clone(), f.width, f.height).map_err(|e| format!("PNG: {e}"))?;
                let path = dir.join(format!("frame-{i:05}.png"));
                std::fs::write(&path, png).map_err(|e| format!("{}: {e}", path.display()))
            }
            Out::Mov(m) => m.write(f),
        }
    }
}

/// One render, everything checked before it starts.
struct Work {
    prepared: filmcraft_effectcraft::Prepared,
    host: std::sync::Arc<EngineFrames>,
    template: std::path::PathBuf,
    output: std::path::PathBuf,
    created: bool,
    format: Format,
    gpu: bool,
    media_values: Vec<String>,
    services: std::sync::Arc<dyn crate::Services>,
    openers: Vec<filmcraft_media::Opener>,
}

impl Work {
    fn asset_name(&self) -> String {
        match self.format {
            Format::Png => "frames".into(),
            Format::ProRes4444 => {
                let stem = self.output.file_name().and_then(|n| n.to_str()).filter(|n| !n.is_empty() && !n.starts_with('.')).unwrap_or("template");
                format!("{stem}.mov")
            }
        }
    }
}

fn render_template(s: &mut Session, p: &Value) -> Result<Value> {
    motion_graphics::poll(s);
    let template = template_p(RENDER, p)?;
    let values = filmcraft_effectcraft::params::values(p.get("values")).map_err(|e| bad(RENDER, e.to_string()))?;
    let size = motion_graphics::size_p(RENDER, p, "size")?;
    let fps = p.get("fps").filter(|v| !v.is_null()).map(|v| motion_graphics::fps_value(RENDER, v, "fps")).transpose()?;
    let range = range_p(RENDER, p)?;
    let format = format_p(RENDER, p)?;
    let gpu = motion_graphics::strict_bool(RENDER, p, "gpu", true)?;
    let import = motion_graphics::strict_bool(RENDER, p, "import", true)?;
    let place = motion_graphics::place_p(s, RENDER, p)?;
    if place.is_some() && !import {
        return Err(bad(RENDER, "`place` needs the render imported (leave `import` on)"));
    }
    let wait = motion_graphics::strict_bool(RENDER, p, "wait", false)?;
    linked(RENDER)?;
    motion_graphics::refuse_while_rendering(s, RENDER)?;
    let bytes = read_template(RENDER, &template)?;
    let output = out_p(RENDER, p, &template)?;
    let created = motion_graphics::revision_dir(RENDER, &output)?;
    let staging = output.join(STAGING);
    if let Err(e) = std::fs::create_dir(&staging) {
        motion_graphics::free_revision(&output, STAGING, created);
        return Err(bad(RENDER, format!("{}: {e}", staging.display())));
    }
    let host = std::sync::Arc::new(EngineFrames::new(s));
    let req = filmcraft_effectcraft::RenderRequest { values: values.clone(), size, fps: fps.map(FrameRate::from_f64), range, gpu };
    let prepared = match filmcraft_effectcraft::prepare(&bytes, &req, &staging.join("media"), host.as_ref()) {
        Ok(p) => p,
        Err(e) => {
            motion_graphics::free_revision(&output, STAGING, created);
            return Err(bad(RENDER, format!("{}: {e}", template.display())));
        }
    };
    let summary = prepared.summary();
    let name = summary["name"].as_str().unwrap_or("template").to_string();
    let media_values: Vec<String> = values
        .values()
        .filter_map(|v| v.as_str().or_else(|| v.get("path").and_then(Value::as_str)))
        .filter(|v| Path::new(v).is_absolute())
        .map(str::to_string)
        .collect();
    let openers = s.media.openers.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    let work = Work { prepared, host, template, output, created, format, gpu, media_values, services: s.services.clone(), openers };
    let job = crate::narration::new_job(s, format!("Rendering an EffectCraft template: {name}"));
    job.progress.total.store(1000, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let out: motion_graphics::Outcome = Default::default();
    let job_id = job.id;
    let planned = work.output.join(work.asset_name());
    // a template renders over a transparent background: it goes above V1
    s.motion_renders.push(motion_graphics::Pending::render(job_id, out.clone(), place, format!("Rendered by EffectCraft from {name}"), import, true));
    let (output, created) = (work.output.clone(), work.created);
    let spawned = crate::narration::spawn(s, job, wait, "filmcraft-effectcraft", move || {
        let t0 = web_time::Instant::now();
        let r = run_render(&work, &prog);
        let status = match &r {
            Ok(_) => Ok("EffectCraft template rendered".to_string()),
            Err(v) => Err(v.get("error").and_then(Value::as_str).unwrap_or("the render failed").to_string()),
        };
        *out.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
        crate::narration::finish(&prog, &res, status, t0);
    });
    if let Err(e) = spawned {
        s.motion_renders.forget(job_id);
        motion_graphics::free_revision(&output, STAGING, created);
        return Err(e);
    }
    let mut v = json!({"job": job_id, "output": str_of(&output), "asset": str_of(&planned), "kind": format.kind(), "template": summary});
    if wait {
        motion_graphics::poll(s);
        let o = motion_graphics::outcome_json(s, job_id).unwrap_or(Value::Null);
        if let Some(e) = o.get("error").and_then(Value::as_str) {
            return Err(bad(RENDER, e.to_string()));
        }
        if let (Some(dst), Some(src)) = (v.as_object_mut(), o.as_object()) {
            for (k, x) in src {
                dst.insert(k.clone(), x.clone());
            }
        }
    }
    Ok(v)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_render(w: &Work, prog: &filmcraft_export::Progress) -> std::result::Result<Value, Value> {
    let status = |t: &str| *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = t.to_string();
    let failed = |e: String| {
        motion_graphics::free_revision(&w.output, STAGING, w.created);
        json!({"error": e, "output": str_of(&w.output), "kept": Value::Null})
    };
    // the GPU is held for the whole render (EffectCraft composites on it)
    let _gpu = if w.gpu {
        status("Waiting for the GPU");
        motion_graphics::gpu("EffectCraft template", GPU_WAIT).map_err(failed)?
    } else {
        None
    };
    if prog.cancel.load(Ordering::Relaxed) {
        return Err(failed("stopped: nothing was rendered".into()));
    }
    let staging = w.output.join(STAGING);
    let name = w.asset_name();
    let made = staging.join(&name);
    let mut out = match w.format {
        Format::Png => {
            std::fs::create_dir(&made).map_err(|e| failed(format!("{}: {e}", made.display())))?;
            Out::Png(made.clone())
        }
        Format::ProRes4444 => Out::Mov(MovOut::new(&made, w.prepared.size(), w.prepared.rate()).map_err(failed)?),
    };
    let total = w.prepared.frames().max(1);
    status(&format!("EffectCraft: rendering {total} frames"));
    let mut sink = |i: u64, f: &filmcraft_effectcraft::Frame| -> std::result::Result<(), String> {
        out.write(i, f)?;
        prog.done.store(((i + 1).saturating_mul(900) / total).min(900), Ordering::Relaxed);
        Ok(())
    };
    let host: std::sync::Arc<dyn filmcraft_effectcraft::HostFrames> = w.host.clone();
    let stats = match w.prepared.render(host, &mut sink, &prog.cancel) {
        Ok(s) => s,
        Err(filmcraft_effectcraft::Error::Stopped) => return Err(failed("stopped: nothing was kept".into())),
        Err(e) => return Err(failed(e.to_string())),
    };
    if let Out::Mov(m) = out {
        m.finish().map_err(failed)?;
    }
    // the render is finished: it moves into the revision and stays from here on
    let asset = w.output.join(&name);
    std::fs::rename(&made, &asset).map_err(|e| json!({"error": format!("{}: {e}; the render is in {}", asset.display(), staging.display()), "kept": str_of(&staging), "output": str_of(&w.output)}))?;
    let _ = std::fs::remove_dir_all(&staging);
    let tool = json!({"name": "EffectCraft", "version": filmcraft_effectcraft::EFFECTCRAFT_VERSION, "pin": filmcraft_effectcraft::PIN});
    let backend = json!({"gpu": stats.gpu, "gpuFrames": stats.gpu_frames, "cpuFrames": stats.cpu_frames, "note": stats.gpu_note, "seconds": stats.seconds});
    let detail = json!({"tool": tool, "backend": backend, "template": w.prepared.summary()});
    if prog.cancel.load(Ordering::Relaxed) {
        return Err(motion_graphics::kept_at(&w.output, "stopped after the render".into(), detail));
    }
    // AutoCut's own probe: the size and rate asked for, every frame there, the alpha
    status("Probing the render");
    prog.done.store(920, Ordering::Relaxed);
    let (fw, fh) = w.prepared.size();
    let fps = w.prepared.rate().as_f64();
    let spec = motion_graphics::ProbeSpec {
        tool: "EffectCraft",
        stated_as: "EffectCraft rendered",
        size_hint: "",
        alpha: true,
        sequence: w.format == Format::Png,
        fps: Some(fps),
        stage: Some((fw, fh)),
        services: w.services.as_ref(),
        openers: &w.openers,
    };
    let stated = motion_graphics::Stated { width: Some(fw), height: Some(fh), fps: Some(fps), length: Some(stats.frames) };
    let probed = motion_graphics::probe_asset(&spec, &asset, stated).map_err(|e| motion_graphics::kept_at(&w.output, e, detail.clone()))?;
    let mut detail = detail;
    detail["probe"] = probed.json.clone();
    // the handoff, as motion.render writes it
    let (command, params) = match (&probed.first, w.format) {
        (Some(first), Format::Png) => ("file.importImageSequence", json!({"path": str_of(first), "fps": probed.fps})),
        _ => ("file.import", json!({"paths": [str_of(&asset)]})),
    };
    let mut dependencies = vec![str_of(&w.template)];
    dependencies.extend(w.media_values.iter().cloned());
    let manifest = json!({
        "command": command, "params": params, "cacheKey": Value::Null, "cached": false, "reviewOnly": false,
        "template": str_of(&w.template), "dependencies": dependencies, "values": w.prepared.summary()["values"].clone(), "tool": detail["tool"].clone(),
    });
    let manifest_file = w.output.join(motion_graphics::IMPORT_MANIFEST);
    let partial = w.output.join(format!("{}.partial", motion_graphics::IMPORT_MANIFEST));
    let text = serde_json::to_string_pretty(&manifest).unwrap_or_default();
    std::fs::write(&partial, format!("{text}\n"))
        .and_then(|()| std::fs::rename(&partial, &manifest_file))
        .map_err(|e| motion_graphics::kept_at(&w.output, format!("{}: {e}", manifest_file.display()), detail.clone()))?;
    prog.done.store(1000, Ordering::Relaxed);
    let mut v = detail;
    v["output"] = json!(str_of(&w.output));
    v["asset"] = json!(str_of(&asset));
    v["kind"] = json!(w.format.kind());
    v["frames"] = json!(stats.frames);
    v["manifest"] = json!(str_of(&manifest_file));
    v["import"] = manifest;
    v["reviewOnly"] = json!(false);
    Ok(v)
}

#[cfg(target_arch = "wasm32")]
fn run_render(_: &Work, _: &filmcraft_export::Progress) -> std::result::Result<Value, Value> {
    Err(json!({"error": "EffectCraft runs on the desktop app"}))
}

#[cfg(test)]
#[path = "effectcraft_tests.rs"]
mod tests;

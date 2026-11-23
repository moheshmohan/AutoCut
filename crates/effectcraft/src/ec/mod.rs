//! The build with EffectCraft: templates read into EffectCraft's project model, filled with
//! values and rendered by its renderer (GPU compositor when asked and present, the CPU reference
//! renderer otherwise and for any frame the GPU declines or fails).

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use effectcraft_project::essential::{self, ControlType, EgControl, EgKind, Override};
use effectcraft_project::keyframe::TextDoc;
use effectcraft_project::{AlphaMode, Footage, FootageKind, ItemId, ItemKind, LayerSource, ParamUi, Project, Property, Value as Ev};
use effectcraft_render::{Backend, LayerCache, RenderOpts, Renderer};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Map, Value, json};

use crate::limits::{MAX_FOOTAGE_PIXELS, MAX_ITEM_PIXELS, MAX_LISTED, MAX_STRING_CHARS, MAX_TEXT_SIZE};
use crate::package::{self, Package};
use crate::{Error, Frame, HostFrames, RenderRequest, RenderStats, Result, color, params};

mod fonts;
mod footage;

/// The deepest precomp nesting rendered (a deeper chain would recurse past the stack).
pub(crate) const MAX_NESTING: usize = 12;
/// The stack the parse and the render run on (EffectCraft recurses through nested comps).
const STACK: usize = 64 << 20;
/// The layer cache of one render (unchanging layers are drawn once).
const LAYER_CACHE: usize = 512 << 20;
/// Numbers a value may set, in magnitude.
const MAX_MAGNITUDE: f64 = 1e7;

fn ec_rate(r: FrameRate) -> effectcraft_time::FrameRate {
    effectcraft_time::FrameRate::new(r.num, r.den)
}

/// `f` under `catch_unwind`, on a scoped thread with a large stack: a panic inside EffectCraft
/// becomes [`Error::Internal`] naming `what`.
fn guarded<T: Send>(what: &str, f: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    let caught = |r: std::thread::Result<Result<T>>| r.unwrap_or_else(|_| Err(Error::Internal(what.to_string())));
    std::thread::scope(|s| {
        let f = AssertUnwindSafe(f);
        match std::thread::Builder::new().name("effectcraft".into()).stack_size(STACK).spawn_scoped(s, move || catch_unwind(f)) {
            Ok(h) => caught(h.join().unwrap_or_else(Err)),
            Err(_) => Err(Error::Internal(format!("{what} (no thread could be started)"))),
        }
    })
}

/// `f` under `catch_unwind` on this thread.
fn caught<T>(what: &str, f: impl FnOnce() -> T) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| Error::Internal(what.to_string()))
}

// ---------------------------------------------------------------------------------------------
// Reading

/// A template read: the package and its project, the comp checked.
struct Loaded {
    pkg: Package,
    project: Project,
    comp: ItemId,
    notes: Vec<String>,
}

fn load(bytes: &[u8]) -> Result<Loaded> {
    let pkg = package::read(bytes)?;
    let (project, notes) = Project::from_json_repaired(&pkg.project).map_err(|e| Error::Template(format!("project.ecproj cannot be read: {e}")))?;
    let comp = ItemId(pkg.comp);
    if project.comp(comp).is_none() {
        return Err(Error::Template(format!("the manifest's comp {} is not a composition of the template's project", pkg.comp)));
    }
    nesting(&project, comp)?;
    Ok(Loaded { pkg, project, comp, notes })
}

/// How deep comps nest below `comp`: refused past [`MAX_NESTING`] (a cycle is EffectCraft's to
/// skip; it is not followed here).
fn nesting(project: &Project, comp: ItemId) -> Result<usize> {
    fn depth(project: &Project, id: ItemId, path: &mut Vec<ItemId>, memo: &mut BTreeMap<ItemId, usize>) -> Result<usize> {
        if let Some(d) = memo.get(&id) {
            return Ok(*d);
        }
        if path.len() > MAX_NESTING {
            return Err(Error::TooLarge(format!("the template nests compositions deeper than {MAX_NESTING}")));
        }
        let Some(c) = project.comp(id) else { return Ok(0) };
        path.push(id);
        let mut d = 0;
        for l in c.layers.iter() {
            if let LayerSource::Comp { item } = l.source
                && !path.contains(&item)
            {
                d = d.max(depth(project, item, path, memo)? + 1);
            }
        }
        path.pop();
        memo.insert(id, d);
        Ok(d)
    }
    depth(project, comp, &mut Vec::new(), &mut BTreeMap::new())
}

/// Every item's size at `scale`: comps and solids become buffers that size, footage is decoded
/// at its own.
fn check_items(project: &Project, scale: f64) -> Result<()> {
    for it in project.items.values() {
        let (w, h, what, scaled) = match &it.kind {
            ItemKind::Comp(c) => (c.width, c.height, "composition", true),
            ItemKind::Solid(s) => (s.width, s.height, "solid", true),
            ItemKind::Footage(f) if f.has_video => (f.width, f.height, "footage", false),
            _ => continue,
        };
        let px = f64::from(w) * f64::from(h) * if scaled { scale * scale } else { 1.0 };
        let cap = if scaled { MAX_ITEM_PIXELS } else { MAX_FOOTAGE_PIXELS };
        if !px.is_finite() || px > cap as f64 {
            return Err(Error::TooLarge(format!("the template's {what} `{}` is {w}x{h}: too large to render here (at most {cap} pixels)", short(&it.name))));
        }
    }
    Ok(())
}

fn short(s: &str) -> String {
    s.chars().take(80).collect()
}

/// The controls of `comp`'s Essential Graphics, groups flattened (each with its `group`).
fn flat_controls(eg: &essential::EssentialGraphics) -> Vec<(&EgControl, Option<u64>)> {
    fn go<'a>(v: &'a [EgControl], group: Option<u64>, depth: usize, out: &mut Vec<(&'a EgControl, Option<u64>)>) {
        for c in v {
            if out.len() >= MAX_LISTED {
                return;
            }
            out.push((c, group));
            if let EgKind::Group { children } = &c.kind
                && depth < 32
            {
                go(children, Some(c.id), depth + 1, out);
            }
        }
    }
    let mut out = Vec::new();
    go(&eg.controls, None, 0, &mut out);
    out
}

fn type_name(t: Option<ControlType>) -> Value {
    t.map_or(Value::Null, |t| json!(t.label().to_ascii_lowercase()))
}

/// A property value as a control shows it.
fn value_json(ty: Option<ControlType>, v: &Ev) -> Value {
    match (ty, v) {
        (Some(ControlType::Font), Ev::Text(t)) => json!({"font": t.font, "style": t.style, "size": t.size}),
        (Some(ControlType::Scale), _) => json!(v.as_f64()),
        (_, Ev::Text(t)) => json!(t.text),
        _ => v.to_json(),
    }
}

fn range_json(ui: &ParamUi) -> Value {
    match ui {
        ParamUi::Slider { min, max, slider_min, slider_max, decimals } => {
            json!({"min": min, "max": max, "sliderMin": slider_min, "sliderMax": slider_max, "decimals": decimals})
        }
        _ => Value::Null,
    }
}

fn control_json(project: &Project, comp: ItemId, eg: &essential::EssentialGraphics, c: &EgControl, group: Option<u64>) -> Value {
    let master = eg.resolve(c.id).unwrap_or(c);
    let mut v = json!({"id": c.id, "name": c.name, "group": group});
    match &master.kind {
        EgKind::Property { .. } => {
            let src = essential::source_prop(project, comp, c);
            let ty = src.and_then(|(_, pr)| essential::effective_type(master, pr));
            v["kind"] = json!(if matches!(c.kind, EgKind::Mirror { .. }) { "mirror" } else { "property" });
            v["type"] = type_name(ty);
            v["default"] = src.map_or(Value::Null, |(_, pr)| value_json(ty, &pr.value_at(effectcraft_time::Tick::ZERO)));
            v["range"] = src.map_or(Value::Null, |(_, pr)| range_json(&pr.ui));
            v["options"] = src.and_then(|(l, pr)| l.popup_options(pr)).map_or(Value::Null, |o| json!(o.iter().take(MAX_LISTED).collect::<Vec<_>>()));
            v["layer"] = src.map_or(Value::Null, |(l, _)| json!(l.name));
            v["property"] = src.map_or(Value::Null, |(_, pr)| json!(pr.name));
            v["animated"] = json!(src.is_some_and(|(_, pr)| pr.is_animated() || pr.has_expression()));
            v["missing"] = json!(src.is_none());
        }
        EgKind::Media { .. } => {
            let item = essential::source_media(project, comp, master);
            let f = item.and_then(|i| project.item(i)).and_then(|it| match &it.kind {
                ItemKind::Footage(f) => Some((it.name.clone(), f)),
                _ => None,
            });
            v["kind"] = json!("media");
            v["type"] = json!("media");
            v["default"] = f.as_ref().map_or(Value::Null, |(name, f)| json!({"item": item.map(|i| i.0), "name": name, "file": file_name(&f.path)}));
            v["missing"] = json!(f.is_none());
        }
        EgKind::Comment { text } => {
            v["kind"] = json!("comment");
            v["text"] = json!(text.chars().take(MAX_STRING_CHARS).collect::<String>());
        }
        EgKind::Group { .. } => v["kind"] = json!("group"),
        EgKind::Mirror { .. } => {
            v["kind"] = json!("mirror");
            v["missing"] = json!(true);
        }
    }
    v
}

fn file_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).chars().take(255).collect()
}

/// The footage items `comp` uses (transitively), with whether the template carries each.
fn footage_json(l: &Loaded) -> Vec<Value> {
    essential::dependencies(&l.project, l.comp)
        .into_iter()
        .filter_map(|id| {
            let it = l.project.item(id)?;
            let ItemKind::Footage(f) = &it.kind else { return None };
            let embedded = l.pkg.media.contains_key(&f.path) || f.sequence.first().is_some_and(|p| l.pkg.media.contains_key(p));
            Some(json!({
                "item": id.0, "name": short(&it.name), "kind": format!("{:?}", f.kind).to_ascii_lowercase(), "file": file_name(&f.path),
                "embedded": embedded, "data": f.data.is_some(), "width": f.width, "height": f.height, "hasVideo": f.has_video, "hasAudio": f.has_audio,
                "frames": f.sequence.len().max(1),
            }))
        })
        .take(MAX_LISTED)
        .collect()
}

fn comp_json(l: &Loaded) -> Value {
    let name = l.project.item(l.comp).map(|i| short(&i.name)).unwrap_or_default();
    let Some(c) = l.project.comp(l.comp) else { return Value::Null };
    let rate = FrameRate::new(c.frame_rate.num, c.frame_rate.den).sane();
    let duration = Tick(c.duration.0);
    let frames = params::frame_count(Tick::ZERO, duration, rate).ok();
    json!({
        "id": l.comp.0, "name": name, "width": c.width, "height": c.height, "pixelAspect": c.pixel_aspect,
        "fps": rate.as_f64(), "rate": {"num": rate.num, "den": rate.den}, "duration": duration.0, "seconds": duration.seconds(),
        "frames": frames, "background": c.background, "layers": c.layers.len(),
    })
}

pub(crate) fn template_info(bytes: &[u8]) -> Result<Value> {
    guarded("reading the template", || {
        let l = load(bytes)?;
        let eg = l.project.comp(l.comp).and_then(|c| c.essential.clone()).unwrap_or_default();
        let controls: Vec<Value> = flat_controls(&eg).into_iter().map(|(c, g)| control_json(&l.project, l.comp, &eg, c, g)).collect();
        let media: Vec<Value> =
            l.pkg.media.iter().take(MAX_LISTED).map(|(k, v)| json!({"entry": k, "item": package::media_entry(k).map(|(i, _)| i), "bytes": v.len()})).collect();
        let poster = l.pkg.poster.as_ref().map(|p| {
            let size = package::png_size(p);
            json!({"entry": "poster.png", "bytes": p.len(), "width": size.map(|s| s.0), "height": size.map(|s| s.1)})
        });
        let manifest = Value::Object(l.pkg.manifest.clone());
        // whether it can be rendered at its own size here (the caps every render checks)
        let renderable = l.project.comp(l.comp).map(|c| params::output_size((c.width, c.height), None).and_then(|(_, scale)| check_items(&l.project, scale)));
        let refused = match renderable {
            Some(Err(e)) => json!(e.to_string()),
            _ => Value::Null,
        };
        Ok(json!({
            "name": l.pkg.name, "version": manifest.get("version"), "generator": manifest.get("generator"),
            "comp": comp_json(&l), "templateName": if eg.name.is_empty() { Value::Null } else { json!(short(&eg.name)) },
            "controls": controls, "media": media, "footage": footage_json(&l), "poster": poster,
            "notes": l.notes.iter().take(MAX_LISTED).collect::<Vec<_>>(), "ignored": l.pkg.ignored.iter().take(MAX_LISTED).collect::<Vec<_>>(),
            "renderable": refused.is_null(), "refused": refused,
            "manifest": manifest, "effectcraft": crate::about(),
        }))
    })
}

// ---------------------------------------------------------------------------------------------
// Values

/// The control `key` names: its id, else its name (exactly, then ignoring case).
fn find_control<'a>(eg: &'a essential::EssentialGraphics, key: &str) -> Option<&'a EgControl> {
    let all = flat_controls(eg);
    if let Ok(id) = key.parse::<u64>()
        && let Some((c, _)) = all.iter().find(|(c, _)| c.id == id)
    {
        return Some(c);
    }
    all.iter().find(|(c, _)| c.name == key).or_else(|| all.iter().find(|(c, _)| c.name.eq_ignore_ascii_case(key))).map(|(c, _)| *c)
}

fn names(eg: &essential::EssentialGraphics) -> String {
    let n: Vec<String> = flat_controls(eg)
        .into_iter()
        .filter(|(c, _)| !matches!(c.kind, EgKind::Group { .. } | EgKind::Comment { .. }))
        .take(40)
        .map(|(c, _)| format!("`{}`", short(&c.name)))
        .collect();
    if n.is_empty() { "it has none".into() } else { n.join(", ") }
}

fn finite_value(v: &Ev) -> bool {
    v.components().iter().all(|x| x.is_finite() && x.abs() <= MAX_MAGNITUDE)
}

fn check_text(name: &str, t: &TextDoc) -> Result<()> {
    let ok = |x: f64, most: f64| x.is_finite() && x.abs() <= most;
    if !(t.size.is_finite() && t.size > 0.0 && t.size <= MAX_TEXT_SIZE) {
        return Err(Error::Values(format!("control `{name}`: a text size from above 0 to {MAX_TEXT_SIZE}")));
    }
    let nums = [t.stroke_width, t.tracking, t.baseline_shift, t.h_scale, t.v_scale, t.tsume, t.box_pos[0], t.box_pos[1]];
    if !nums.iter().all(|x| ok(*x, MAX_MAGNITUDE))
        || !t.leading.is_none_or(|l| ok(l, MAX_MAGNITUDE))
        || !t.box_size.is_none_or(|b| ok(b[0], MAX_MAGNITUDE) && ok(b[1], MAX_MAGNITUDE))
    {
        return Err(Error::Values(format!("control `{name}`: the text's numbers must be finite and reasonable")));
    }
    if t.text.chars().count() > MAX_STRING_CHARS {
        return Err(Error::TooLarge(format!("control `{name}`: the text is longer than {MAX_STRING_CHARS} characters")));
    }
    if !t.fill.iter().chain(&t.stroke).all(|c| c.is_finite()) {
        return Err(Error::Values(format!("control `{name}`: the text's colours must be finite")));
    }
    Ok(())
}

/// What a value should look like, for an error.
fn hint(ty: Option<ControlType>) -> &'static str {
    match ty {
        Some(ControlType::Text) => "a string",
        Some(ControlType::Color) => "\"#rrggbb\", \"#rrggbbaa\" or [r, g, b(, a)] from 0 to 1",
        Some(ControlType::Slider) | Some(ControlType::Angle) => "a number",
        Some(ControlType::Scale) => "a percentage",
        Some(ControlType::Checkbox) => "true or false",
        Some(ControlType::Point) => "[x, y] or [x, y, z]",
        Some(ControlType::Dropdown) => "an option's index (from 0)",
        Some(ControlType::Font) => "a font family, or {\"font\", \"style\", \"size\"}",
        Some(ControlType::Media) => "an absolute file path",
        None => "a value of the property's kind",
    }
}

/// `v` as the value of property `pr` shown as `ty`.
fn coerce(name: &str, ty: Option<ControlType>, pr: &Property, options: Option<usize>, v: &Value) -> Result<Ev> {
    let wrong = || Error::Values(format!("control `{name}` is a {} control: give {}", type_name(ty).as_str().unwrap_or("property"), hint(ty)));
    let out = match (ty, &pr.value) {
        (Some(ControlType::Font), Ev::Text(t)) => {
            let mut t = (**t).clone();
            match v {
                Value::String(f) if !f.trim().is_empty() => t.font = f.clone(),
                Value::Object(o) => {
                    if let Some(f) = o.get("font") {
                        t.font = f.as_str().filter(|f| !f.trim().is_empty()).ok_or_else(wrong)?.to_string();
                    }
                    if let Some(s) = o.get("style") {
                        t.style = s.as_str().ok_or_else(wrong)?.to_string();
                    }
                    if let Some(s) = o.get("size") {
                        t.size = s.as_f64().ok_or_else(wrong)?;
                    }
                }
                _ => return Err(wrong()),
            }
            Ev::Text(Box::new(t))
        }
        // a colour is a hex string or three or four numbers (one would leave the rest as they were)
        (_, Ev::Color(_)) if v.as_array().is_some_and(|a| !(3..=4).contains(&a.len())) => return Err(wrong()),
        _ => pr.value.coerce_json(v).ok_or_else(wrong)?,
    };
    if std::mem::discriminant(&out) != std::mem::discriminant(&pr.value) {
        return Err(wrong());
    }
    if !finite_value(&out) {
        return Err(Error::Values(format!("control `{name}`: numbers must be finite, at most {MAX_MAGNITUDE:e} in size")));
    }
    if let Ev::Text(t) = &out {
        check_text(name, t)?;
    }
    if let (Ev::Enum(i), Some(n)) = (&out, options)
        && *i as usize >= n
    {
        return Err(Error::Values(format!("control `{name}`: option {i} does not exist (0 to {})", n.saturating_sub(1))));
    }
    Ok(out)
}

/// A Media Replacement value: an absolute path (or `{"path": …}`).
fn media_value(name: &str, v: &Value) -> Result<String> {
    let p = match v {
        Value::String(p) => Some(p.as_str()),
        Value::Object(o) => o.get("path").and_then(Value::as_str),
        _ => None,
    };
    match p {
        Some(p) if !p.is_empty() && p.len() <= 4096 && !p.contains('\0') && Path::new(p).is_absolute() => Ok(p.to_string()),
        _ => Err(Error::Values(format!("control `{name}` is a media control: give an absolute file path"))),
    }
}

/// The footage item for a file the host read.
fn footage_item(path: &str, m: &crate::HostMedia) -> Footage {
    let rate = m.rate.sane();
    Footage {
        path: path.to_string(),
        kind: if m.still { FootageKind::Still } else { FootageKind::Video },
        width: m.width.max(1),
        height: m.height.max(1),
        pixel_aspect: 1.0,
        frame_rate: ec_rate(rate),
        duration: effectcraft_time::Tick(m.duration.0.max(rate.frame_duration().0)),
        has_video: true,
        has_audio: m.has_audio,
        alpha: AlphaMode::Straight,
        ..Footage::default()
    }
}

/// The overrides `values` make (and what each applied), media files added to `project`.
fn overrides(project: &mut Project, comp: ItemId, values: &Map<String, Value>, host: &dyn HostFrames) -> Result<(Vec<Override>, Vec<Value>)> {
    if values.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let eg = project
        .comp(comp)
        .and_then(|c| c.essential.clone())
        .filter(|e| !e.controls.is_empty())
        .ok_or_else(|| Error::Values("the template exposes no controls, so it takes no values".into()))?;
    let mut out = Vec::new();
    let mut applied = Vec::new();
    let mut seen = BTreeSet::new();
    for (key, v) in values {
        let c = find_control(&eg, key).ok_or_else(|| Error::Values(format!("the template has no control `{}` (its controls: {})", short(key), names(&eg))))?;
        let master = eg.resolve(c.id).ok_or_else(|| Error::Values(format!("control `{}` mirrors a control the template no longer has", short(&c.name))))?;
        if !seen.insert(master.id) {
            return Err(Error::Values(format!("control `{}` is given twice (by name and id, or through a mirror)", short(&c.name))));
        }
        let name = short(&c.name);
        let value = match &master.kind {
            EgKind::Property { .. } => {
                let (l, pr) = essential::source_prop(project, comp, c)
                    .ok_or_else(|| Error::Values(format!("control `{name}` drives a property the template no longer has")))?;
                let ty = essential::effective_type(master, pr);
                coerce(&name, ty, pr, l.popup_options(pr).map(|o| o.len()), v)?
            }
            EgKind::Media { .. } => {
                let path = media_value(&name, v)?;
                let m = caught("probing a media value", || host.probe(&path))?
                    .ok_or_else(|| Error::Media(format!("control `{name}`: {path} cannot be read as a picture or video")))?;
                let label = project.item(comp).map(|i| i.label).unwrap_or_default();
                let item = project.add_item(&file_name(&path), label, None, ItemKind::Footage(footage_item(&path, &m)));
                Ev::Scalar(item.0 as f64)
            }
            _ => {
                return Err(Error::Values(format!(
                    "control `{name}` is a {} and takes no value",
                    if matches!(c.kind, EgKind::Group { .. }) { "group" } else { "comment" }
                )));
            }
        };
        applied.push(json!({"control": name, "id": c.id, "value": v}));
        out.push(Override { control: c.id, value, keys: Vec::new() });
    }
    Ok((out, applied))
}

// ---------------------------------------------------------------------------------------------
// Media

/// Write the media `project`'s footage takes from the package into `dir` and point the footage
/// at the copies; then check every footage item `comp` uses can be read.
fn media(project: &mut Project, comp: ItemId, pkg: &Package, dir: &Path, host: &dyn HostFrames) -> Result<usize> {
    let mut written: BTreeMap<String, String> = BTreeMap::new();
    let mut write = |entry: &str| -> Result<Option<String>> {
        if let Some(p) = written.get(entry) {
            return Ok(Some(p.clone()));
        }
        let Some(bytes) = pkg.media.get(entry) else { return Ok(None) };
        let out =
            package::media_path(dir, entry).ok_or_else(|| Error::Template(format!("the template's media entry `{}` has an unusable name", short(entry))))?;
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::Media(format!("{}: {e}", parent.display())))?;
        }
        std::fs::write(&out, bytes).map_err(|e| Error::Media(format!("{}: {e}", out.display())))?;
        let s = out.to_string_lossy().into_owned();
        written.insert(entry.to_string(), s.clone());
        Ok(Some(s))
    };
    for it in project.items.values_mut() {
        if let ItemKind::Footage(f) = &mut it.kind {
            if let Some(p) = write(&f.path)? {
                f.path = p;
            }
            for x in &mut f.sequence {
                if let Some(p) = write(x)? {
                    *x = p;
                }
            }
        }
    }
    let mut checked = 0;
    for id in essential::dependencies(project, comp) {
        let Some(it) = project.item(id) else { continue };
        let ItemKind::Footage(f) = &it.kind else { continue };
        if !f.has_video || f.data.is_some() || matches!(f.kind, FootageKind::Audio | FootageKind::Data) {
            continue;
        }
        let name = short(&it.name);
        if f.kind == FootageKind::Model {
            return Err(Error::Media(format!("the template's 3D model `{name}` cannot be rendered here (3D model footage is not supported yet)")));
        }
        let path = if f.kind == FootageKind::Sequence { f.sequence.first().unwrap_or(&f.path) } else { &f.path };
        let readable = !path.is_empty() && Path::new(path).is_absolute() && caught("probing the template's media", || host.probe(path))?.is_some();
        if !readable {
            let why = if pkg.media.keys().any(|k| k == path) || path.starts_with(&*dir.to_string_lossy()) {
                "cannot be read"
            } else {
                "is not in the template and not on this computer"
            };
            return Err(Error::Media(format!("the template's media `{name}` ({}) {why}", file_name(path))));
        }
        checked += 1;
    }
    Ok(checked)
}

// ---------------------------------------------------------------------------------------------
// Preparing and rendering

pub(crate) struct Prepared {
    project: Project,
    comp: ItemId,
    name: String,
    comp_info: Value,
    size: (u32, u32),
    scale: f64,
    rate: FrameRate,
    start: Tick,
    end: Tick,
    frames: u64,
    gpu: bool,
    applied: Vec<Value>,
    notes: Vec<String>,
    media_checked: usize,
}

pub(crate) fn prepare(bytes: &[u8], req: &RenderRequest, media_dir: &Path, host: &dyn HostFrames) -> Result<Prepared> {
    let values = params::values(Some(&Value::Object(req.values.clone())))?;
    guarded("preparing the template", || {
        let Loaded { pkg, mut project, comp, mut notes } = load(bytes)?;
        let comp_info = comp_json(&Loaded { pkg: pkg.clone(), project: project.clone(), comp, notes: Vec::new() });
        let c = project.comp(comp).ok_or_else(|| Error::Template("the template's composition is gone".into()))?;
        let ((w, h), scale) = params::output_size((c.width, c.height), req.size)?;
        let comp_rate = FrameRate::new(c.frame_rate.num, c.frame_rate.den).sane();
        let rate = req.fps.unwrap_or(comp_rate).sane();
        let (start, end) = params::range(req.range.0, req.range.1, Tick(c.duration.0))?;
        let frames = params::frame_count(start, end, rate)?;
        check_items(&project, scale)?;
        let (ov, applied) = overrides(&mut project, comp, &values, host)?;
        if !ov.is_empty() {
            match essential::with_overrides(&project, comp, &ov) {
                Some(p) => project = p,
                None => notes.push("the values changed nothing in the composition".into()),
            }
        }
        let media_checked = media(&mut project, comp, &pkg, media_dir, host)?;
        notes.truncate(MAX_LISTED);
        Ok(Prepared { project, comp, name: pkg.name, comp_info, size: (w, h), scale, rate, start, end, frames, gpu: req.gpu, applied, notes, media_checked })
    })
}

impl Prepared {
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }
    pub(crate) fn rate(&self) -> FrameRate {
        self.rate
    }
    pub(crate) fn frames(&self) -> u64 {
        self.frames
    }
    pub(crate) fn summary(&self) -> Value {
        json!({
            "name": self.name, "comp": self.comp_info, "values": self.applied,
            "width": self.size.0, "height": self.size.1, "scale": self.scale,
            "fps": self.rate.as_f64(), "rate": {"num": self.rate.num, "den": self.rate.den},
            "range": {"start": self.start.0, "end": self.end.0}, "frames": self.frames,
            "media": self.media_checked, "notes": self.notes, "effectcraft": crate::about(),
        })
    }

    /// The comp time of output frame `i`.
    fn time(&self, i: u64) -> Tick {
        let i = i64::try_from(i).unwrap_or(i64::MAX);
        Tick(self.start.0.saturating_add(self.rate.tick_of(i).0))
    }

    pub(crate) fn render(&self, host: Arc<dyn HostFrames>, sink: &mut crate::Sink<'_>, cancel: &AtomicBool) -> Result<RenderStats> {
        guarded("rendering the template", || self.render_frames(host, sink, cancel))
    }

    fn render_frames(&self, host: Arc<dyn HostFrames>, sink: &mut crate::Sink<'_>, cancel: &AtomicBool) -> Result<RenderStats> {
        use effectcraft_render::Accelerator as _;
        fonts::ensure();
        let t0 = std::time::Instant::now();
        let footage = footage::HostFootage::new(host);
        let cache = LayerCache::new(LAYER_CACHE);
        let expr = effectcraft_expr::Expressions;
        let mut stats = RenderStats::default();
        let mut gpu = None;
        if self.gpu {
            match caught("starting the GPU", effectcraft_gpu::Gpu::try_headless) {
                Ok(Ok(g)) => {
                    stats.gpu = Some(g.name());
                    gpu = Some(g);
                }
                Ok(Err(e)) => stats.gpu_note = Some(format!("no GPU here ({e}): rendered on the CPU")),
                Err(e) => stats.gpu_note = Some(format!("{e}: rendered on the CPU")),
            }
        }
        let n = u64::from(self.size.0) * u64::from(self.size.1);
        for i in 0..self.frames {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Stopped);
            }
            let t = effectcraft_time::Tick(self.time(i).0);
            let img = self.frame(&footage, &cache, &expr, &mut gpu, t, i, &mut stats)?;
            if (img.width, img.height) != self.size || img.data.len() as u64 != n {
                return Err(Error::Render(format!(
                    "EffectCraft rendered frame {i} at {}x{}, not the {}x{} asked for",
                    img.width, img.height, self.size.0, self.size.1
                )));
            }
            let frame = Frame { width: img.width, height: img.height, rgba: color::frame_to_rgba8(&img.data) };
            sink(i, &frame).map_err(Error::Render)?;
            stats.frames += 1;
        }
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(stats)
    }

    fn renderer<'a>(
        &'a self,
        footage: &'a footage::HostFootage,
        cache: &'a LayerCache,
        expr: &'a effectcraft_expr::Expressions,
        backend: Backend,
    ) -> Renderer<'a> {
        let opts = RenderOpts { scale: self.scale, motion_blur: true, backend, ..RenderOpts::default() };
        let mut r = Renderer::new(&self.project, footage, opts);
        r.expr = Some(expr);
        r.cache = Some(cache);
        r
    }

    /// One frame: on the GPU while it works (a frame it declines renders on the CPU inside
    /// `comp_frame_scoped`), else on the CPU; a GPU failure or panic ends the GPU's part and the
    /// frame renders again on the CPU, with a cleared layer cache.
    #[allow(clippy::too_many_arguments)]
    fn frame(
        &self,
        footage: &footage::HostFootage,
        cache: &LayerCache,
        expr: &effectcraft_expr::Expressions,
        gpu: &mut Option<effectcraft_gpu::Gpu>,
        t: effectcraft_time::Tick,
        i: u64,
        stats: &mut RenderStats,
    ) -> Result<effectcraft_render::Image> {
        if let Some(g) = gpu.as_ref() {
            let mut r = self.renderer(footage, cache, expr, Backend::Gpu);
            r.accel = Some(g);
            let mut declined = false;
            let attempt = caught(&format!("rendering frame {i} on the GPU"), || {
                r.comp_frame_scoped(self.comp, t, |a| {
                    let out = g.within_memory(|| a.comp_frame(&r, self.comp, t));
                    declined = matches!(out, Ok(None));
                    out
                })
            });
            match attempt {
                Ok(Ok(img)) => {
                    if declined {
                        stats.cpu_frames += 1;
                    } else {
                        stats.gpu_frames += 1;
                    }
                    return Ok(img);
                }
                Ok(Err(e)) => stats.gpu_note = Some(format!("the GPU failed at frame {i} ({e}): it and the rest rendered on the CPU")),
                Err(e) => stats.gpu_note = Some(format!("{e}: it and the rest rendered on the CPU")),
            }
            *gpu = None;
            cache.clear();
        }
        let r = self.renderer(footage, cache, expr, Backend::Cpu);
        let img = caught(&format!("rendering frame {i}"), || r.comp_frame(self.comp, t))?;
        stats.cpu_frames += 1;
        Ok(img)
    }
}

#[cfg(test)]
mod tests;

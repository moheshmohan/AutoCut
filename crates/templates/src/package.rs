//! Template packages: the manifest, the sequence file and the bundled assets, read from a file
//! map (a folder's files or a ZIP's entries) and cross-checked against each other.

use std::collections::{BTreeMap, BTreeSet};

use filmcraft_media::MediaKind;
use filmcraft_project::graphic::{self, TEXT_LAYER};
use filmcraft_project::{ItemId, ItemKind, MediaRef, Project, Sequence, TrackKind, validate_frame_size};
use filmcraft_time::Tick;

use crate::manifest::{self, MAX_TEMPLATE_TICKS, Manifest};
use crate::{ASSETS_DIR, Issues, MANIFEST_FILE, TemplateError};

/// Largest sequence file, bytes.
pub const MAX_SEQUENCE_BYTES: usize = 32 * 1024 * 1024;
/// Most items in a sequence file.
pub const MAX_ITEMS: usize = 4096;
/// Most tracks of one kind in the template sequence.
pub const MAX_TRACKS: usize = 64;
/// Most clips in the template sequence.
pub const MAX_CLIPS: usize = 8192;
/// Most effects on one clip.
pub const MAX_EFFECTS: usize = 128;
/// Most keyframes of one parameter.
pub const MAX_KEYFRAMES: usize = 10_000;

/// A template: manifest, sequence and bundled files.
#[derive(Clone, Debug, PartialEq)]
pub struct Package {
    pub manifest: Manifest,
    /// The sequence file: a project holding the template sequence and its placeholder media.
    pub fragment: Project,
    /// Bundled files (`assets/…` → bytes).
    pub assets: BTreeMap<String, Vec<u8>>,
    /// Documentation kept with the package (`README.md`, `LICENSE`…).
    pub docs: BTreeMap<String, Vec<u8>>,
}

/// Files a package may carry besides the manifest, the sequence file and `assets/`.
fn is_doc(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    !n.contains('/')
        && (n.starts_with("readme")
            || n.starts_with("license")
            || n.starts_with("licence")
            || n.starts_with("attribution")
            || n.ends_with(".md")
            || n.ends_with(".txt"))
}

impl Package {
    /// A package from its parts, checked.
    pub fn new(manifest: Manifest, fragment: Project) -> Result<Package, TemplateError> {
        let p = Package { manifest, fragment, assets: BTreeMap::new(), docs: BTreeMap::new() };
        p.check().into_result()?;
        Ok(p)
    }

    /// Read a package from its files (relative `/`-separated paths → bytes): a folder's files or
    /// a ZIP's entries. A package whose files all sit in one top folder (a zipped
    /// `Name.fcvideo-template/` folder) is read from inside it.
    pub fn from_files(mut files: BTreeMap<String, Vec<u8>>) -> Result<Package, TemplateError> {
        if !files.contains_key(MANIFEST_FILE) {
            let tops: BTreeSet<&str> = files.keys().filter_map(|k| k.split_once('/').map(|(a, _)| a)).collect();
            if let (1, Some(top)) = (tops.len(), tops.first())
                && files.keys().all(|k| k.contains('/'))
            {
                let prefix = format!("{top}/");
                files = files.into_iter().filter_map(|(k, v)| k.strip_prefix(&prefix).map(|r| (r.to_string(), v))).collect();
            }
        }
        for k in files.keys() {
            crate::zip::check_entry_name(k).map_err(|why| TemplateError::Package(format!("`{k}`: {why}")))?;
        }
        let m = files.remove(MANIFEST_FILE).ok_or_else(|| TemplateError::NotTemplate(format!("no {MANIFEST_FILE} in the package")))?;
        let manifest = manifest::parse(&m)?;
        let seq = files
            .remove(&manifest.sequence_file)
            .ok_or_else(|| TemplateError::Invalid(vec![issue("sequenceFile", format!("`{}` is not in the package", manifest.sequence_file))]))?;
        let fragment = decode_fragment(&seq, &manifest.sequence_file)?;
        let mut assets = BTreeMap::new();
        let mut docs = BTreeMap::new();
        let mut is = Issues::default();
        for (k, v) in files {
            if k.starts_with(ASSETS_DIR) {
                assets.insert(k, v);
            } else if is_doc(&k) {
                docs.insert(k, v);
            } else {
                is.push(k, "unexpected file (a package holds template.json, its sequence file, assets/ and README / LICENSE files)");
            }
        }
        is.into_result()?;
        let p = Package { manifest, fragment, assets, docs };
        p.check().into_result()?;
        Ok(p)
    }

    /// Read a ZIP-archived package.
    pub fn from_zip(bytes: &[u8]) -> Result<Package, TemplateError> {
        Package::from_files(crate::zip::read(bytes)?)
    }

    /// The package's files in writing order.
    pub fn files(&self) -> Vec<(String, Vec<u8>)> {
        let mut v = vec![
            (MANIFEST_FILE.to_string(), self.manifest.to_json().into_bytes()),
            (self.manifest.sequence_file.clone(), filmcraft_format::encode(&self.fragment, true)),
        ];
        v.extend(self.assets.iter().map(|(k, d)| (k.clone(), d.clone())));
        v.extend(self.docs.iter().map(|(k, d)| (k.clone(), d.clone())));
        v
    }

    /// The package as a ZIP archive.
    pub fn to_zip(&self) -> Result<Vec<u8>, TemplateError> {
        Ok(crate::zip::write(&self.files())?)
    }

    /// The template sequence.
    pub fn sequence(&self) -> Option<&Sequence> {
        self.fragment.sequence(ItemId(self.manifest.sequence))
    }

    /// Cross-check the manifest, the sequence file and the bundled files.
    pub fn check(&self) -> Issues {
        let mut is = Issues::default();
        check_assets(&self.manifest, &self.assets, &mut is);
        check_fragment(&self.manifest, &self.fragment, &mut is);
        is
    }
}

fn issue(field: impl Into<String>, message: impl Into<String>) -> crate::Issue {
    crate::Issue { field: field.into(), message: message.into() }
}

/// Decode a sequence file (a FilmCraft project file, any schema this build reads).
pub fn decode_fragment(bytes: &[u8], name: &str) -> Result<Project, TemplateError> {
    if bytes.len() > MAX_SEQUENCE_BYTES {
        return Err(TemplateError::Invalid(vec![issue(name, format!("{} bytes (at most {MAX_SEQUENCE_BYTES})", bytes.len()))]));
    }
    filmcraft_format::decode(bytes).map(|l| l.project).map_err(|e| TemplateError::Invalid(vec![issue(name, e.to_string())]))
}

fn check_assets(m: &Manifest, files: &BTreeMap<String, Vec<u8>>, is: &mut Issues) {
    for (i, a) in m.assets.iter().enumerate() {
        match files.get(&a.path) {
            None => is.push(format!("assets[{i}].path"), format!("`{}` is not in the package", a.path)),
            Some(d) if crate::sha256_hex(d) != a.sha256 => is.push(format!("assets[{i}].sha256"), format!("`{}` does not match its SHA-256", a.path)),
            Some(_) => {}
        }
    }
    for k in files.keys() {
        if !m.assets.iter().any(|a| &a.path == k) {
            is.push(k.clone(), "bundled file not listed in the manifest's `assets` (every asset needs its author, source and licence)");
        }
    }
}

fn label(p: &Project, id: ItemId) -> String {
    match p.item(id) {
        Some(it) => format!("sequence item {} (“{}”)", id.0, it.name),
        None => format!("sequence item {}", id.0),
    }
}

fn sane_rate(r: filmcraft_time::FrameRate) -> bool {
    r.num > 0 && r.den > 0 && r.as_f64() <= 1000.0 && r.as_f64() >= 1.0
}

fn check_fragment(m: &Manifest, p: &Project, is: &mut Issues) {
    let f = &m.sequence_file;
    if p.items.len() > MAX_ITEMS {
        is.push(f.clone(), format!("{} items (at most {MAX_ITEMS})", p.items.len()));
        return;
    }
    if !p.luts.is_empty() {
        is.push(format!("{f}: luts"), "LUTs are not supported in templates yet (bundle looks as Lumetri presets)");
    }
    if !p.transcripts.is_empty() || !p.narrations.is_empty() || !p.source_graphics.is_empty() {
        is.push(f.clone(), "transcripts, narrations and source graphics are not supported in templates");
    }
    let main = ItemId(m.sequence);
    for (id, it) in &p.items {
        let at = label(p, *id);
        match &it.kind {
            ItemKind::Sequence(_) if *id == main => {}
            ItemKind::Sequence(_) => is.push(at, "nested and extra sequences are not supported in templates (format version 1)"),
            ItemKind::Subclip { .. } => is.push(at, "subclips are not supported in templates"),
            ItemKind::Graphic { width, height, rate } | ItemKind::AdjustmentLayer { width, height, rate, .. } => {
                if let Err(e) = validate_frame_size(*width, *height) {
                    is.push(at.clone(), e);
                }
                if !sane_rate(*rate) {
                    is.push(at, "frame rate out of range");
                }
            }
            ItemKind::Media(mc) => {
                if let Some(v) = &mc.info.video {
                    if let Err(e) = validate_frame_size(v.width, v.height) {
                        is.push(at.clone(), e);
                    }
                    if !sane_rate(v.frame_rate) {
                        is.push(at.clone(), "frame rate out of range");
                    }
                }
                if !(0..=MAX_TEMPLATE_TICKS).contains(&mc.info.duration.0) {
                    is.push(at.clone(), "media duration out of range");
                }
                if mc.info.audio_streams.len() > filmcraft_media::MAX_AUDIO_STREAMS {
                    is.push(at.clone(), "too many audio streams");
                }
                if mc.proxy.is_some() {
                    is.push(at.clone(), "proxies are not allowed in templates");
                }
                match &mc.media {
                    MediaRef::Generator(_) => {}
                    MediaRef::File { path } => {
                        if !m.assets.iter().any(|a| &a.path == path) {
                            is.push(at, format!("uses `{path}`: a template's media is generated or a bundled asset listed in `assets`"));
                        }
                    }
                }
                if mc.info.kind == MediaKind::ImageSequence {
                    is.push(label(p, *id), "image sequences are not supported in templates");
                }
            }
        }
    }
    let Some(seq) = p.sequence(main) else {
        is.push("sequence", format!("item {} of {f} is not a sequence", m.sequence));
        return;
    };
    if let Err(e) = seq.check() {
        is.push("sequence", e);
        return;
    }
    if seq.multicam.is_some() || seq.merged.is_some() {
        is.push("sequence", "multi-camera and merged-clip sequences are not supported");
    }
    if seq.video_tracks.len() > MAX_TRACKS
        || seq.audio_tracks.len() > MAX_TRACKS
        || seq.submix_tracks.len() > MAX_TRACKS
        || seq.caption_tracks.len() > MAX_TRACKS
    {
        is.push("sequence", format!("at most {MAX_TRACKS} tracks of each kind"));
        return;
    }
    let clips: usize = seq.all_tracks().map(|t| t.items.len()).sum();
    if clips > MAX_CLIPS {
        is.push("sequence", format!("{clips} clips (at most {MAX_CLIPS})"));
        return;
    }
    for t in seq.all_tracks() {
        for c in &t.items {
            let at = format!("sequence clip {} (“{}”)", c.id.0, c.name);
            if p.item(c.item).is_none() {
                is.push(at.clone(), format!("uses item {} which is not in {f}", c.item.0));
            }
            if c.start < Tick::ZERO {
                is.push(at.clone(), "starts before 0");
            }
            if !(c.speed.is_finite() && c.speed.abs() >= 0.01 && c.speed.abs() <= 100.0) {
                is.push(at.clone(), "speed out of range");
            }
            if c.effects.len() > MAX_EFFECTS {
                is.push(at.clone(), format!("more than {MAX_EFFECTS} effects"));
            }
            if c.effects.iter().any(|e| e.params.values().any(|q| q.keyframes.len() > MAX_KEYFRAMES)) {
                is.push(at, format!("a parameter has more than {MAX_KEYFRAMES} keyframes"));
            }
        }
    }
    let dur = seq.duration();
    let frame = seq.settings.frame_rate.frame_duration();
    if (dur - m.duration).abs() > frame {
        is.push("duration", format!("{} ticks, but the sequence is {} ticks long", m.duration.0, dur.0));
    }
    let find = |clip: u64| seq.all_tracks().find_map(|t| t.items.iter().find(|c| c.id.0 == clip).map(|c| (t.kind, c)));
    for (si, s) in m.slots.iter().enumerate() {
        for (pi, pl) in s.placements.iter().enumerate() {
            let at = format!("slots[{si}].placements[{pi}].clip");
            let Some((kind, c)) = find(pl.clip) else {
                is.push(at, format!("clip {} is not in the template sequence", pl.clip));
                continue;
            };
            let item = p.item(c.item);
            match s.kind {
                k if k.is_visual() && kind != TrackKind::Video => is.push(at.clone(), "a picture slot's clip must be on a video track"),
                manifest::SlotKind::Music if kind != TrackKind::Audio => is.push(at.clone(), "a music slot's clip must be on an audio track"),
                manifest::SlotKind::Text => {
                    let is_graphic = item.is_some_and(|i| matches!(i.kind, ItemKind::Graphic { .. }));
                    let texts: Vec<u64> = c.effects.iter().filter(|e| e.effect == TEXT_LAYER).map(filmcraft_project::gtemplate::layer_uid).collect();
                    if !is_graphic || texts.is_empty() {
                        is.push(at.clone(), "a text slot's clip must be a graphic with a text layer");
                    } else if let Some(l) = pl.layer
                        && !texts.contains(&l)
                    {
                        is.push(format!("slots[{si}].placements[{pi}].layer"), format!("the clip has no text layer with uid {l}"));
                    }
                }
                _ => {}
            }
            if let Some(min) = s.constraints.min_duration
                && c.duration < min
            {
                is.push(at.clone(), format!("the clip is shorter than the slot's minDuration ({} < {} ticks)", c.duration.0, min.0));
            }
            if let Some(max) = s.constraints.max_duration
                && c.duration > max
            {
                is.push(at, format!("the clip is longer than the slot's maxDuration ({} > {} ticks)", c.duration.0, max.0));
            }
        }
    }
    for (vi, v) in m.variants.iter().enumerate() {
        if let Err(e) = validate_frame_size(v.width, v.height) {
            is.push(format!("variants[{vi}]"), e);
        }
        for (hi, h) in v.hints.iter().enumerate() {
            let Some((_, c)) = find(h.clip) else {
                is.push(format!("variants[{vi}].hints[{hi}].clip"), format!("clip {} is not in the template sequence", h.clip));
                continue;
            };
            for (li, l) in h.layers.iter().enumerate() {
                if !c.effects.iter().any(|e| graphic::is_layer(e) && filmcraft_project::gtemplate::layer_uid(e) == l.layer) {
                    is.push(format!("variants[{vi}].hints[{hi}].layers[{li}].layer"), format!("clip {} has no graphic layer with uid {}", h.clip, l.layer));
                }
            }
        }
    }
    if let Some(g) = &m.beat_grid
        && g.beats.last().is_some_and(|b| *b > dur + Tick(60 * filmcraft_time::TICKS_PER_SECOND))
    {
        is.push("beatGrid.beats", "beats run more than a minute past the end of the template");
    }
}

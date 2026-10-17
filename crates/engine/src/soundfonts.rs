//! What generated music is played on: the built-in oscillator voices, or a SoundFont
//! (`filmcraft-soundfont`: rustysynth behind a structural check of the file).
//!
//! | command | does |
//! |---|---|
//! | `music.soundfonts` | the downloadable SoundFont (GeneralUser GS: installed, size, source, licence in full, the author's note on sample provenance), the choice in Preferences and what music plays on right now |
//! | `music.downloadSoundfont` | download GeneralUser GS as a background job after the user agreed (`consent: true`), every file SHA-256 checked, into `<data dir>/models/generaluser-gs/` |
//! | `music.useSoundfont` | choose (`soundfont`): `auto` (GeneralUser GS when installed, else the oscillators), `oscillators`, `generaluser-gs`, or the path of the user's own `.sf2`; checked and loaded first, then kept in Preferences (`music.soundfont`) |
//!
//! `audio.generate` and `music.preview` take `soundfont` too (the same words or a path) for one
//! call. Whatever plays it, the same request gives the same arrangement, chords and beat map; a
//! patch the font has no preset for (and every riser or impact) plays on its oscillator voice, and
//! a font that fails while playing is replaced by the oscillators for that track (the result's
//! `warning` says so). GeneralUser GS is never bundled and never downloaded without consent.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use filmcraft_audio_dsp::music;
use filmcraft_soundfont::{SoundFont, SoundFontBank, catalog};

use crate::commands::{CommandSpec, always, bad, bool_p};
use crate::{EngineError, Result, Session};

/// GeneralUser GS when installed, else the oscillators.
pub const AUTO: &str = "auto";
pub const OSCILLATORS: &str = "oscillators";
/// Longest path accepted for a SoundFont.
const MAX_PATH: usize = 4096;

/// Preferences ▸ music: what generated music plays on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MusicPrefs {
    /// `auto`, `oscillators`, `generaluser-gs`, or the path of an `.sf2` file.
    pub soundfont: String,
}

impl Default for MusicPrefs {
    fn default() -> Self {
        MusicPrefs { soundfont: AUTO.into() }
    }
}

/// The SoundFont loaded last, kept so its level checks are reused: (file, size, modified, bank).
#[derive(Default)]
pub struct SoundFonts {
    cache: Option<(PathBuf, u64, Option<std::time::SystemTime>, Arc<SoundFontBank>)>,
}

/// What a track is played on.
#[derive(Clone)]
pub enum Instruments {
    Oscillators,
    Font { bank: Arc<SoundFontBank>, path: PathBuf },
}

impl Instruments {
    /// Identifies the instruments (two renders with equal keys and requests are the same audio).
    pub fn key(&self) -> String {
        match self {
            Instruments::Oscillators => OSCILLATORS.into(),
            Instruments::Font { bank, path } => format!("{}#{}", path.display(), bank.font().bytes()),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Instruments::Oscillators => json!({"kind": OSCILLATORS, "name": "Built-in oscillator voices", "path": null}),
            Instruments::Font { bank, path } => json!({
                "kind": "soundfont", "name": bank.font().name(), "path": path.display().to_string(),
                "presets": bank.font().presets(), "drums": bank.font().has_drums(), "bytes": bank.font().bytes(),
            }),
        }
    }
}

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec("music.soundfonts", "Music SoundFonts", "{}", list),
        spec("music.downloadSoundfont", "Download SoundFont", r#"{"soundfont":"generaluser-gs"?,"consent":bool,"wait":bool?}"#, download),
        spec("music.useSoundfont", "Use SoundFont", r#"{"soundfont":"auto|oscillators|generaluser-gs"|path}"#, use_font),
    ]
}

/// Whether this build downloads (feature `speech-download`, on in the desktop app and CLI).
pub fn download_available() -> bool {
    cfg!(feature = "speech-download")
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

/// The choice a call makes: its `soundfont`, else the preference.
fn choice(s: &Session, p: &Value, cmd: &str) -> Result<String> {
    match p.get("soundfont") {
        None | Some(Value::Null) => Ok(s.prefs.music.soundfont.clone()),
        Some(Value::String(t)) => {
            let t = t.trim();
            if t.is_empty() || t.len() > MAX_PATH || t.contains('\0') {
                return Err(bad(cmd, "`soundfont` must be auto, oscillators, generaluser-gs or the path of an .sf2 file"));
            }
            Ok(t.to_string())
        }
        Some(v) => Err(bad(cmd, format!("`soundfont` must be auto, oscillators, generaluser-gs or a path, not {v}"))),
    }
}

/// The installed GeneralUser GS file, if any.
fn installed_gugs() -> Option<PathBuf> {
    let dir = crate::transcript::models_dir()?;
    catalog::installed(&dir, &catalog::GENERALUSER_GS).then(|| catalog::sf2_path(&dir, &catalog::GENERALUSER_GS))
}

/// Why GeneralUser GS cannot be used yet, and what to run.
fn not_installed(cmd: &str) -> EngineError {
    let p = &catalog::GENERALUSER_GS;
    let missing = crate::transcript::models_dir().map_or(catalog::size(p), |d| catalog::missing_bytes(&d, p));
    bad(
        cmd,
        format!(
            "GeneralUser GS is not downloaded yet ({} from {}, licence: {}); once the user agreed to it, run music.downloadSoundfont {{\"consent\": true}}",
            mb(missing),
            p.source,
            p.license
        ),
    )
}

/// Load `path` (from the cache when it is the same file).
fn load(s: &mut Session, path: &Path) -> std::result::Result<Arc<SoundFontBank>, String> {
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let modified = md.modified().ok();
    if let Some((p, len, m, bank)) = &s.soundfonts.cache
        && p == path
        && *len == md.len()
        && *m == modified
    {
        return Ok(bank.clone());
    }
    let font = SoundFont::open(path).map_err(|e| e.to_string())?;
    let bank = Arc::new(SoundFontBank::new(font));
    s.soundfonts.cache = Some((path.to_path_buf(), md.len(), modified, bank.clone()));
    Ok(bank)
}

/// What `p` (or the preference) chooses to play music on, loaded. A font the call names that is
/// missing or unusable is an error; the preference never fails a generation (a font it names that
/// no longer loads — moved, deleted — gives the oscillators and a warning), nor does `auto`.
pub fn resolve(s: &mut Session, p: &Value, cmd: &str) -> Result<(Instruments, Option<String>)> {
    let explicit = p.get("soundfont").is_some_and(|v| !v.is_null());
    match resolve_choice(s, p, cmd) {
        Err(e) if !explicit => {
            let c = s.prefs.music.soundfont.clone();
            Ok((Instruments::Oscillators, Some(format!("the SoundFont chosen in Preferences ({c}) could not be used, so the oscillators played it: {e}"))))
        }
        r => r,
    }
}

fn resolve_choice(s: &mut Session, p: &Value, cmd: &str) -> Result<(Instruments, Option<String>)> {
    let c = choice(s, p, cmd)?;
    match c.to_ascii_lowercase().as_str() {
        OSCILLATORS | "none" | "builtin" | "built-in" => Ok((Instruments::Oscillators, None)),
        AUTO => match installed_gugs() {
            None => Ok((Instruments::Oscillators, None)),
            Some(path) => match load(s, &path) {
                Ok(bank) => Ok((Instruments::Font { bank, path }, None)),
                Err(e) => Ok((Instruments::Oscillators, Some(format!("GeneralUser GS could not be used ({e}); the oscillators played it")))),
            },
        },
        "generaluser-gs" | "generaluser" | "gugs" => {
            let path = installed_gugs().ok_or_else(|| not_installed(cmd))?;
            let bank = load(s, &path).map_err(|e| bad(cmd, e))?;
            Ok((Instruments::Font { bank, path }, None))
        }
        _ => {
            let path = PathBuf::from(&c);
            let bank = load(s, &path).map_err(|e| bad(cmd, format!("the SoundFont {c}: {e}")))?;
            Ok((Instruments::Font { bank, path }, None))
        }
    }
}

/// Make `req` on `inst`, at the engine's rate. The note says when a font failed and the
/// oscillators played instead.
pub fn render(req: &music::Request, inst: &Instruments) -> std::result::Result<(music::Track, Option<String>), String> {
    match inst {
        Instruments::Oscillators => music::generate(req, crate::audio_generate::RATE).map(|t| (t, None)).map_err(|e| e.to_string()),
        Instruments::Font { bank, .. } => filmcraft_soundfont::generate(req, crate::audio_generate::RATE, bank).map_err(|e| e.to_string()),
    }
}

/// The SoundFont package as JSON.
fn package_json(pkg: &catalog::Package) -> Value {
    let dir = crate::transcript::models_dir();
    let installed = dir.as_deref().is_some_and(|d| catalog::installed(d, pkg));
    json!({
        "id": pkg.id, "name": pkg.name, "version": pkg.version, "author": pkg.author, "description": pkg.description,
        "installed": installed,
        "size": catalog::size(pkg),
        "missingBytes": dir.as_deref().map_or(catalog::size(pkg), |d| catalog::missing_bytes(d, pkg)),
        "license": pkg.license, "licenseText": pkg.license_text, "licenseUrl": pkg.license_url,
        "provenance": pkg.provenance, "source": pkg.source,
        "path": if installed { dir.map(|d| catalog::sf2_path(&d, pkg).display().to_string()) } else { None },
    })
}

fn list(s: &mut Session, _p: &Value) -> Result<Value> {
    let selected = s.prefs.music.soundfont.clone();
    let active = match resolve_choice(s, &json!({}), "music.soundfonts") {
        Ok((inst, warning)) => {
            let mut a = inst.to_json();
            a["warning"] = json!(warning);
            a
        }
        Err(e) => json!({"kind": OSCILLATORS, "name": "Built-in oscillator voices", "path": null, "error": e.to_string()}),
    };
    Ok(json!({
        "soundfonts": catalog::PACKAGES.iter().map(|p| package_json(p)).collect::<Vec<_>>(),
        "selected": selected,
        "active": active,
        "choices": [AUTO, OSCILLATORS, catalog::GENERALUSER_GS.id],
        "downloadAvailable": download_available(),
        "maxFileBytes": filmcraft_soundfont::check::MAX_FILE_BYTES,
    }))
}

fn use_font(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "music.useSoundfont";
    if p.get("soundfont").is_none_or(Value::is_null) {
        return Err(bad(cmd, "`soundfont` is required: auto, oscillators, generaluser-gs or the path of an .sf2 file"));
    }
    let (inst, warning) = resolve(s, p, cmd)?;
    let c = choice(s, p, cmd)?;
    let lower = c.to_ascii_lowercase();
    let keep = match (&inst, lower.as_str()) {
        (_, AUTO) => AUTO.to_string(),
        (Instruments::Oscillators, _) => OSCILLATORS.to_string(),
        (Instruments::Font { .. }, "generaluser-gs" | "generaluser" | "gugs") => catalog::GENERALUSER_GS.id.to_string(),
        (Instruments::Font { path, .. }, _) => std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()).display().to_string(),
    };
    let mut prefs = s.prefs.clone();
    prefs.music.soundfont = keep.clone();
    s.set_prefs(prefs).map_err(|e| EngineError::Other(format!("preferences: {e}")))?;
    let mut active = inst.to_json();
    active["warning"] = json!(warning);
    Ok(json!({"selected": keep, "active": active}))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "music.downloadSoundfont";
    let pkg = match p.get("soundfont") {
        None | Some(Value::Null) => &catalog::GENERALUSER_GS,
        Some(Value::String(id)) => catalog::package(id).ok_or_else(|| bad(cmd, format!("unknown SoundFont `{id}` (generaluser-gs)")))?,
        Some(v) => return Err(bad(cmd, format!("`soundfont` must be a SoundFont id (generaluser-gs), not {v}"))),
    };
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    let missing = catalog::missing_bytes(&dir, pkg);
    if missing == 0 {
        return Ok(json!({"job": null, "installed": true, "soundfont": pkg.id, "path": catalog::sf2_path(&dir, pkg).display().to_string()}));
    }
    if !bool_p(p, "consent").unwrap_or(false) {
        return Err(bad(
            cmd,
            format!(
                "downloading {} {} ({} from {}, licence: {}; {}) needs the user's agreement: show them the licence (music.soundfonts → licenseText) and pass `consent: true` once they agreed",
                pkg.name,
                pkg.version,
                mb(missing),
                pkg.source,
                pkg.license,
                pkg.provenance
            ),
        ));
    }
    if !download_available() {
        return Err(bad(cmd, "downloads are not available in this build (built without `speech-download`)"));
    }
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Downloading {} {}", pkg.name, pkg.version));
    job.progress.total.store(missing, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let target = catalog::package_dir(&dir, pkg);
    let done = format!("{} {} installed", pkg.name, pkg.version);
    let files = pkg.files;
    let id = crate::narration::spawn(s, job, wait, "filmcraft-soundfont-download", move || {
        let t0 = web_time::Instant::now();
        let r = fetch(&target, files, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| done), t0);
    })?;
    // a fresh download is a new file: load it anew
    s.soundfonts.cache = None;
    Ok(json!({"job": id, "installed": false, "soundfont": pkg.id, "bytes": missing}))
}

#[cfg(feature = "speech-download")]
fn fetch(dir: &Path, files: &'static [catalog::FontFile], prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::PoisonError;
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> =
        files.iter().map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size }).collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(dir, &files, &mut progress).map_err(|e| e.to_string())?;
    // what was downloaded must load (the check runs on it)
    let sf2 = files.iter().find(|f| f.name.ends_with(".sf2")).map(|f| dir.join(f.name));
    if let Some(p) = sf2
        && let Err(e) = SoundFont::open(&p)
    {
        let _ = std::fs::remove_file(&p);
        return Err(format!("the downloaded SoundFont does not load: {e}"));
    }
    Ok(())
}

#[cfg(not(feature = "speech-download"))]
fn fetch(_: &Path, _: &'static [catalog::FontFile], _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("downloads are not available in this build".into())
}

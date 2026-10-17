//! Generate Audio panel: make sound effects and music beds in the app — a whoosh, an impact, a
//! riser, a drone, or music — and add them at the playhead. Sound effects are synthesized
//! (oscillators, noise, filters); every cue is written as a WAV file in a `Generated Audio` folder
//! beside the project and placed on a free audio track as one undo step (`audio.generate`, the
//! same command agents and the CLI use).
//!
//! **Find music by feel** (Sound ▸ Music): pick a mood and a genre (or one of the styles that fit
//! them, from `music.suggest`), the energy, length, tempo and key; Preview renders it in the
//! background and plays it (`music.preview`); Regenerate hears another variation; each stem
//! (drums, bass, melody, pad) can be muted or rebalanced, or placed on its own track; Add at
//! playhead places exactly what was heard. Instruments: the built-in oscillator voices, or a
//! SoundFont — GeneralUser GS, downloaded once after the user read its licence and agreed
//! (`music.downloadSoundfont`), or the user's own `.sf2` (`music.useSoundfont`). When the ACE-Step
//! neural model is installed, the Engine menu also offers it (`music.generateNeural`: a description,
//! rendered in the background and placed when ready).
//!
//! The same settings and seed always give the same sound; "New variation" picks another seed.
//!
//! Automation ids: `generateAudio.kind`, `generateAudio.length`, `generateAudio.intensity` (Energy
//! for music), `generateAudio.mood`, `generateAudio.key`, `generateAudio.tempo`,
//! `generateAudio.variation`, `generateAudio.add`; music: `generateAudio.engine`,
//! `generateAudio.genre`, `generateAudio.style.1`…`3`, `generateAudio.preview`,
//! `generateAudio.regenerate`, `generateAudio.stem.<drums|bass|melody|pad>.mute`,
//! `generateAudio.stem.<…>.level`, `generateAudio.stems`, `generateAudio.instruments`,
//! `generateAudio.soundfont.get`, `generateAudio.soundfont.confirm`,
//! `generateAudio.soundfont.cancel`, `generateAudio.soundfont.path`, `generateAudio.soundfont.use`,
//! `generateAudio.prompt` (neural).

use egui::{Rect, pos2};
use filmcraft_audio_dsp::music::{Genre, Stem, Style};
use filmcraft_audio_dsp::synth::{Kind, Mood};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::FilmcraftApp;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 48.0;
/// Keys offered for music and drones.
const KEYS: [&str; 12] = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"];
/// The procedural soundtrack engine (instant; oscillators or a SoundFont).
pub const ENGINE_PROCEDURAL: &str = "procedural";
/// ACE-Step 1.5 text to music (a background job), when its model is installed.
pub const ENGINE_NEURAL: &str = "neural";
/// Stem level range in the panel, dB.
const LEVEL_DB: std::ops::RangeInclusive<f64> = -24.0..=6.0;
/// Styles listed under "Styles that fit".
const STYLES_SHOWN: usize = 3;

/// The cue being made (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateAudioDraft {
    /// `whoosh`, `impact`, `riser`, `drone` or `music`.
    pub kind: String,
    /// Length, seconds.
    pub seconds: f64,
    /// 0..1 (music: its energy).
    pub intensity: f64,
    /// Music: a mood id (`epic`, `tense`, `uplifting`, `calm`, `happy`, `dark`, `dreamy`,
    /// `energetic`, `sad`).
    pub mood: String,
    /// Music and drones.
    pub key: String,
    /// Music, beats per minute.
    pub bpm: f64,
    /// The variation; `None` until "New variation" picks one (or a preview settles one), then
    /// each cue of a kind gets the next seed.
    pub seed: Option<u64>,
    /// Music: a genre id (`cinematic`, `electronic`, `pop`, `ambient`, `lofi`, `piano`), or ""
    /// for the mood's usual one.
    pub genre: String,
    /// Music: place the four stems on their own tracks instead of the mix.
    pub stems: bool,
    /// Music: each stem's level, dB (drums, bass, melody, pad).
    pub levels: [f64; 4],
    /// Music: muted stems (drums, bass, melody, pad).
    pub muted: [bool; 4],
    /// Music: `procedural`, or `neural` (ACE-Step 1.5, when installed).
    pub engine: String,
    /// Neural music: the description (empty: made from the mood, genre and tempo).
    pub prompt: String,
    /// Instruments ▸ Custom: the `.sf2` file to use.
    pub soundfont_path: String,
    /// The GeneralUser GS licence and download confirmation is showing.
    pub confirm_soundfont: bool,
    /// The running GeneralUser GS download.
    pub soundfont_job: Option<u64>,
    /// The preview being rendered.
    pub preview_job: Option<u64>,
    /// The neural generation running.
    pub neural_job: Option<u64>,
    /// A preview is playing.
    pub previewing: bool,
    #[serde(skip)]
    cache: Cache,
}

/// What the panel read from the session (refreshed when stale).
#[derive(Clone, Debug, Default, PartialEq)]
struct Cache {
    /// `music.suggest` for (mood, genre, energy, length).
    suggest_key: String,
    styles: Vec<StyleRow>,
    /// `music.soundfonts`.
    fonts: Option<Value>,
    /// Whether the ACE-Step model is installed (and this build runs it).
    neural: Option<bool>,
    /// When the playing preview ends.
    preview_ends: Option<web_time::Instant>,
}

/// One style that fits the brief.
#[derive(Clone, Debug, PartialEq)]
struct StyleRow {
    mood: String,
    genre: String,
    bpm: f64,
    energy: f64,
    feel: String,
    uses: String,
}

impl Default for GenerateAudioDraft {
    fn default() -> Self {
        Self {
            kind: Kind::Whoosh.id().into(),
            seconds: Kind::Whoosh.default_seconds(),
            intensity: 0.7,
            mood: Mood::Epic.id().into(),
            key: "A".into(),
            bpm: 96.0,
            seed: None,
            genre: String::new(),
            stems: false,
            levels: [0.0; 4],
            muted: [false; 4],
            engine: ENGINE_PROCEDURAL.into(),
            prompt: String::new(),
            soundfont_path: String::new(),
            confirm_soundfont: false,
            soundfont_job: None,
            preview_job: None,
            neural_job: None,
            previewing: false,
            cache: Cache::default(),
        }
    }
}

impl GenerateAudioDraft {
    fn kind(&self) -> Kind {
        Kind::from_id(&self.kind).unwrap_or(Kind::Whoosh)
    }

    fn mood(&self) -> Mood {
        Mood::from_id(&self.mood).unwrap_or(Mood::Epic)
    }

    fn genre(&self) -> Option<Genre> {
        Genre::ALL.into_iter().find(|g| g.id() == self.genre)
    }

    /// The `audio.generate` parameters (placed at the playhead).
    pub fn params(&self) -> Value {
        let k = self.kind();
        if k == Kind::Music {
            return self.music_params();
        }
        let mut p = json!({"kind": k.id(), "duration": self.seconds.clamp(0.1, k.max_seconds()), "intensity": self.intensity.clamp(0.0, 1.0)});
        if k == Kind::Drone {
            p["key"] = json!(self.key);
        }
        if let Some(seed) = self.seed {
            p["seed"] = json!(seed);
        }
        p
    }

    /// The music's `audio.generate` parameters (`music.preview` takes the same, without stems).
    pub fn music_params(&self) -> Value {
        let mut p = json!({
            "kind": "music",
            "duration": finite(self.seconds, 30.0).clamp(0.1, Kind::Music.max_seconds()),
            "energy": finite(self.intensity, 0.7).clamp(0.0, 1.0),
            "mood": self.mood().id(),
            "key": self.key,
            "bpm": finite(self.bpm, 96.0).clamp(40.0, 200.0),
        });
        if let Some(g) = self.genre() {
            p["genre"] = json!(g.id());
        }
        let mut levels = serde_json::Map::new();
        let mut mute = Vec::new();
        for (i, s) in Stem::ALL.iter().enumerate() {
            if self.muted.get(i).copied().unwrap_or(false) {
                mute.push(json!(s.id()));
            } else if let Some(db) = self.levels.get(i).copied().filter(|d| d.is_finite() && d.abs() > 1e-9) {
                levels.insert(s.id().into(), json!(db.clamp(*LEVEL_DB.start(), *LEVEL_DB.end())));
            }
        }
        if !levels.is_empty() {
            p["levels"] = Value::Object(levels);
        }
        if !mute.is_empty() {
            p["mute"] = json!(mute);
        }
        if self.stems {
            p["stems"] = json!(true);
        }
        if let Some(seed) = self.seed {
            p["seed"] = json!(seed);
        }
        p
    }

    /// The `music.preview` parameters.
    pub fn preview_params(&self) -> Value {
        let mut p = self.music_params();
        if let Some(o) = p.as_object_mut() {
            o.remove("stems");
            o.remove("kind");
        }
        p
    }

    /// The `music.generateNeural` description: the prompt, else made from the mood and genre.
    pub fn neural_prompt(&self) -> String {
        let p = self.prompt.trim();
        if !p.is_empty() {
            return p.to_string();
        }
        let style = Style::of(self.mood(), self.genre());
        format!(
            "{} {} instrumental, {}, {:.0} BPM",
            self.mood().name().to_lowercase(),
            style.genre.name().to_lowercase(),
            style.feel,
            self.bpm.clamp(40.0, 200.0)
        )
    }
}

fn finite(x: f64, or: f64) -> f64 {
    if x.is_finite() { x } else { or }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Whoosh => "Whoosh",
        Kind::Impact => "Impact",
        Kind::Riser => "Riser",
        Kind::Drone => "Drone",
        Kind::Music => "Music",
    }
}

fn kind_hint(k: Kind) -> &'static str {
    match k {
        Kind::Whoosh => "A swept noise pass for transitions and fast moves.",
        Kind::Impact => "A cinematic hit: a sub boom, a crack and a long tail.",
        Kind::Riser => "Tension that builds to a cut and stops dead.",
        Kind::Drone => "A low, slowly moving chord under a scene.",
        Kind::Music => {
            "Music arranged to the exact length and ending on a bar: drums, bass, melody and pads in sections, at -20 LUFS so it sits under a voice. Find it by feel, preview it, then add it."
        }
    }
}

/// What the panel asked for this frame (run after the widgets, when the draft is not borrowed).
enum Action {
    Add,
    Preview,
    StopPreview,
    UseSoundfont(String),
    DownloadSoundfont,
    CancelJob(u64),
}

/// `(fraction, status)` of job `id`.
fn job_progress(app: &FilmcraftApp, id: u64) -> Option<(f32, String)> {
    app.session
        .jobs
        .iter()
        .find(|j| j.id == id)
        .map(|j| (j.progress.fraction(), j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()))
}

/// State of job `id`: `None` while running, else its result.
fn job_outcome(app: &FilmcraftApp, id: u64) -> Option<Result<(), String>> {
    use std::sync::atomic::Ordering;
    let Some(j) = app.session.jobs.iter().find(|j| j.id == id) else { return Some(Err("the job disappeared".into())) };
    if !j.progress.finished.load(Ordering::Relaxed) {
        return None;
    }
    Some(match &*j.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner) {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(e.clone()),
        None => Err("the job stopped".into()),
    })
}

/// Follow the panel's jobs: play a preview when it is ready, report downloads and generations.
fn poll(app: &mut FilmcraftApp) {
    if let Some(job) = app.ui.generate_audio.preview_job
        && let Some(done) = job_outcome(app, job)
    {
        app.ui.generate_audio.preview_job = None;
        match app.session.music_preview.outcome(job) {
            Some(Ok(p)) => {
                let what = format!("{} {} · seed {} · {}", p.track.plan.mood.name(), p.track.plan.genre.name(), p.request.seed, p.track.instruments);
                play(app, &what, p.warning.clone());
            }
            Some(Err(e)) => app.ui.status = format!("Music preview: {e}"),
            None => {
                if let Err(e) = done {
                    app.ui.status = format!("Music preview: {e}");
                }
            }
        }
    }
    if let Some(job) = app.ui.generate_audio.soundfont_job
        && let Some(r) = job_outcome(app, job)
    {
        app.ui.generate_audio.soundfont_job = None;
        app.ui.generate_audio.cache.fonts = None;
        app.ui.status = match r {
            Ok(()) => "GeneralUser GS installed: music now plays on it".into(),
            Err(e) => format!("GeneralUser GS download failed: {e}"),
        };
    }
    if let Some(job) = app.ui.generate_audio.neural_job
        && let Some(r) = job_outcome(app, job)
    {
        app.ui.generate_audio.neural_job = None;
        app.ui.status = match r {
            Ok(()) => "Neural music placed".into(),
            Err(e) => format!("Neural music: {e}"),
        };
    }
    let d = &mut app.ui.generate_audio;
    if d.previewing && d.cache.preview_ends.is_some_and(|t| web_time::Instant::now() >= t) {
        d.previewing = false;
        d.cache.preview_ends = None;
    }
}

/// Play the latest preview and say what it is.
fn play(app: &mut FilmcraftApp, what: &str, warning: Option<String>) {
    match app.play_music_preview() {
        Ok(secs) => {
            app.ui.generate_audio.previewing = true;
            let len = std::time::Duration::try_from_secs_f64(if secs.is_finite() { secs.clamp(0.0, 3600.0) } else { 0.0 }).unwrap_or_default();
            app.ui.generate_audio.cache.preview_ends = Some(web_time::Instant::now() + len);
            app.ui.status = format!("Playing {what}");
        }
        Err(e) => {
            app.ui.generate_audio.previewing = false;
            app.ui.status = format!("Preview ready ({what}) — {e}");
        }
    }
    if let Some(w) = warning {
        app.ui.status = format!("{} · {w}", app.ui.status);
    }
}

/// Refresh what the music section shows from the session (cheap: only what is stale).
fn refresh(app: &mut FilmcraftApp) {
    let d = &app.ui.generate_audio;
    if d.kind() != Kind::Music {
        return;
    }
    let key = format!("{}|{}|{:.2}|{:.1}", d.mood, d.genre, d.intensity, d.seconds);
    if d.cache.suggest_key != key {
        let mut q = json!({"mood": d.mood().id(), "duration": finite(d.seconds, 30.0).clamp(1.0, Kind::Music.max_seconds()), "count": STYLES_SHOWN, "energy": finite(d.intensity, 0.7).clamp(0.0, 1.0)});
        if let Some(g) = d.genre() {
            q["genre"] = json!(g.id());
        }
        let styles = app
            .session
            .execute("music.suggest", q)
            .ok()
            .and_then(|r| r["suggestions"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .map(|s| StyleRow {
                mood: s["style"]["mood"].as_str().unwrap_or_default().to_string(),
                genre: s["style"]["genre"].as_str().unwrap_or_default().to_string(),
                bpm: s["style"]["bpm"]["usual"].as_f64().unwrap_or(96.0),
                energy: s["style"]["energy"].as_f64().unwrap_or(0.6),
                feel: s["style"]["feel"].as_str().unwrap_or_default().to_string(),
                uses: s["style"]["useCases"].as_array().map(|u| u.iter().filter_map(Value::as_str).take(3).collect::<Vec<_>>().join(", ")).unwrap_or_default(),
            })
            .collect();
        let c = &mut app.ui.generate_audio.cache;
        c.suggest_key = key;
        c.styles = styles;
    }
    if app.ui.generate_audio.cache.fonts.is_none() {
        app.ui.generate_audio.cache.fonts = app.session.execute("music.soundfonts", json!({})).ok();
    }
    if app.ui.generate_audio.cache.neural.is_none() {
        let installed = filmcraft_engine::music_neural::available()
            && filmcraft_engine::music_neural::models_json().as_array().is_some_and(|m| m.iter().any(|x| x["installed"] == json!(true)));
        app.ui.generate_audio.cache.neural = Some(installed);
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    poll(app);
    refresh(app);
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let has_seq = app.session.active_sequence().is_some();
    let progress = |id: Option<u64>| id.and_then(|j| job_progress(app, j));
    let (font_dl, preview_busy, neural_busy) =
        (progress(app.ui.generate_audio.soundfont_job), app.ui.generate_audio.preview_job, progress(app.ui.generate_audio.neural_job));
    let mut actions: Vec<Action> = Vec::new();
    let auto = &mut app.auto;
    let d = &mut app.ui.generate_audio;
    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - FOOTER_H));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("generate-audio-body"));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut child, |ui| {
        ui.set_max_width(ui.available_width() - 6.0);
        let w = ui.available_width();
        let label = |ui: &mut egui::Ui, text: &str| {
            ui.label(egui::RichText::new(text).color(t.text).size(12.5).strong());
            ui.add_space(2.0);
        };
        label(ui, "Sound");
        let kind = d.kind();
        let cb = egui::ComboBox::from_id_salt("generate-audio-kind").selected_text(kind_name(kind)).width(w).show_ui(ui, |ui| {
            for k in Kind::ALL {
                if ui.selectable_label(k == kind, kind_name(k)).on_hover_text(kind_hint(k)).clicked() && k != kind {
                    d.kind = k.id().into();
                    d.seconds = k.default_seconds();
                    d.seed = None;
                }
            }
        });
        auto.add("generateAudio.kind", visible(ui, cb.response.rect), "Sound");
        let kind = d.kind();
        ui.label(egui::RichText::new(kind_hint(kind)).color(t.text_dim).size(11.0));
        if kind == Kind::Music {
            music_section(ui, d, auto, &t, w, &mut actions, MusicJobs { font: font_dl.clone(), preview: preview_busy, neural: neural_busy.clone() });
            return;
        }
        ui.add_space(10.0);
        label(ui, "Length");
        ui.spacing_mut().slider_width = w - 70.0;
        let s = ui.add(egui::Slider::new(&mut d.seconds, 0.2..=kind.max_seconds()).logarithmic(true).max_decimals(1).suffix(" s"));
        auto.add("generateAudio.length", visible(ui, s.rect), "Length");
        ui.add_space(10.0);
        label(ui, "Intensity");
        let s = ui.add(egui::Slider::new(&mut d.intensity, 0.0..=1.0).fixed_decimals(2));
        auto.add("generateAudio.intensity", visible(ui, s.rect), "Intensity");
        if kind == Kind::Drone {
            ui.add_space(10.0);
            label(ui, "Key");
            key_combo(ui, d, auto, w);
        }
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            let b = ui.button("New variation");
            auto.add("generateAudio.variation", visible(ui, b.rect), "New variation");
            if b.clicked() {
                d.seed = Some(next_seed(d.seed));
            }
            let note = match d.seed {
                Some(s) => format!("variation {s}"),
                None => "each cue a new variation".into(),
            };
            ui.label(egui::RichText::new(note).color(t.text_faint).size(11.0));
        });
    });
    // footer: Add at playhead
    let neural = d.kind() == Kind::Music && d.engine == ENGINE_NEURAL && d.cache.neural == Some(true);
    let fy = rect.max.y - FOOTER_H + 10.0;
    let r = Rect::from_min_size(pos2(rect.min.x + PAD, fy), egui::vec2(rect.width() - PAD * 2.0, 28.0));
    let text = if neural { "Generate and add at playhead" } else { "Add at playhead" };
    let b = ui.put(r, egui::Button::new(egui::RichText::new(text).strong()).fill(if has_seq { t.accent } else { t.field_bg }));
    app.auto.add("generateAudio.add", r, "Add at playhead");
    if b.clicked() && has_seq {
        actions.push(Action::Add);
    }
    if !has_seq {
        b.on_hover_text("Open a sequence first");
    }
    for a in actions {
        run(app, a);
    }
}

/// Jobs the music section shows.
struct MusicJobs {
    font: Option<(f32, String)>,
    preview: Option<u64>,
    neural: Option<(f32, String)>,
}

fn key_combo(ui: &mut egui::Ui, d: &mut GenerateAudioDraft, auto: &mut crate::automation::Registry, w: f32) {
    let cb = egui::ComboBox::from_id_salt("generate-audio-key").selected_text(d.key.clone()).width(w).show_ui(ui, |ui| {
        for k in KEYS {
            if ui.selectable_label(d.key == k, k).clicked() {
                d.key = k.into();
            }
        }
    });
    auto.add("generateAudio.key", visible(ui, cb.response.rect), "Key");
}

/// Find music by feel.
fn music_section(
    ui: &mut egui::Ui,
    d: &mut GenerateAudioDraft,
    auto: &mut crate::automation::Registry,
    t: &crate::theme::Tokens,
    w: f32,
    actions: &mut Vec<Action>,
    jobs: MusicJobs,
) {
    let label = |ui: &mut egui::Ui, text: &str| {
        ui.label(egui::RichText::new(text).color(t.text).size(12.5).strong());
        ui.add_space(2.0);
    };
    let neural_installed = d.cache.neural == Some(true);
    if neural_installed {
        ui.add_space(10.0);
        label(ui, "Engine");
        let name = |e: &str| if e == ENGINE_NEURAL { "ACE-Step 1.5 (neural, background)" } else { "Procedural (instant, previewable)" };
        let cb = egui::ComboBox::from_id_salt("generate-audio-engine").selected_text(name(&d.engine)).width(w).show_ui(ui, |ui| {
            for e in [ENGINE_PROCEDURAL, ENGINE_NEURAL] {
                if ui.selectable_label(d.engine == e, name(e)).clicked() {
                    d.engine = e.into();
                }
            }
        });
        auto.add("generateAudio.engine", visible(ui, cb.response.rect), "Engine");
    }
    let neural = neural_installed && d.engine == ENGINE_NEURAL;

    ui.add_space(10.0);
    label(ui, "Mood");
    let current = d.mood();
    let cb = egui::ComboBox::from_id_salt("generate-audio-mood").selected_text(current.name()).width(w).show_ui(ui, |ui| {
        for m in Mood::ALL {
            if ui.selectable_label(m == current, m.name()).clicked() && m != current {
                d.mood = m.id().into();
                let s = Style::of(m, d.genre());
                d.bpm = s.bpm.1;
                d.intensity = s.energy;
                d.seed = None;
            }
        }
    });
    auto.add("generateAudio.mood", visible(ui, cb.response.rect), "Mood");
    ui.add_space(8.0);
    label(ui, "Genre");
    let g = d.genre();
    let gname = g.map_or_else(|| format!("Any ({})", Style::of(d.mood(), None).genre.name()), |g| g.name().to_string());
    let cb = egui::ComboBox::from_id_salt("generate-audio-genre").selected_text(gname).width(w).show_ui(ui, |ui| {
        if ui.selectable_label(g.is_none(), "Any (the mood's usual)").clicked() {
            d.genre.clear();
        }
        for x in Genre::ALL {
            if ui.selectable_label(g == Some(x), x.name()).clicked() && g != Some(x) {
                d.genre = x.id().into();
                let s = Style::of(d.mood(), Some(x));
                d.bpm = s.bpm.1;
                d.seed = None;
            }
        }
    });
    auto.add("generateAudio.genre", visible(ui, cb.response.rect), "Genre");

    if !d.cache.styles.is_empty() {
        ui.add_space(8.0);
        label(ui, "Styles that fit");
        let rows = d.cache.styles.clone();
        for (i, s) in rows.iter().enumerate() {
            let chosen = s.mood == d.mood && (d.genre == s.genre || (d.genre.is_empty() && i == 0));
            let title = format!("{} · {} · {:.0} BPM", cap(&s.mood), Genre::parse(&s.genre).map_or(s.genre.as_str(), |g| g.name()), s.bpm);
            let r = ui.selectable_label(chosen, egui::RichText::new(title).size(12.0)).on_hover_text(format!("{}\nFor: {}", s.feel, s.uses));
            ui.label(egui::RichText::new(format!("{} — {}", s.feel, s.uses)).color(t.text_faint).size(10.5));
            auto.add(&format!("generateAudio.style.{}", i + 1), visible(ui, r.rect), &format!("Style {}", i + 1));
            if r.clicked() {
                d.mood = s.mood.clone();
                d.genre = s.genre.clone();
                d.bpm = s.bpm;
                d.intensity = s.energy;
                d.seed = None;
            }
        }
    }

    ui.add_space(10.0);
    label(ui, "Energy");
    ui.spacing_mut().slider_width = w - 70.0;
    let s = ui.add(egui::Slider::new(&mut d.intensity, 0.0..=1.0).fixed_decimals(2));
    auto.add("generateAudio.intensity", visible(ui, s.rect), "Energy");
    ui.add_space(8.0);
    label(ui, "Length");
    let max = if neural { 240.0 } else { Kind::Music.max_seconds() };
    let min = if neural { 5.0 } else { 0.2 };
    let s = ui.add(egui::Slider::new(&mut d.seconds, min..=max).logarithmic(true).max_decimals(1).suffix(" s"));
    auto.add("generateAudio.length", visible(ui, s.rect), "Length");
    ui.add_space(8.0);
    label(ui, "Tempo");
    let s = ui.add(egui::Slider::new(&mut d.bpm, 40.0..=200.0).step_by(1.0).suffix(" BPM"));
    auto.add("generateAudio.tempo", visible(ui, s.rect), "Tempo");
    ui.add_space(8.0);
    label(ui, "Key");
    key_combo(ui, d, auto, w);

    if neural {
        ui.add_space(10.0);
        label(ui, "Description");
        let hint = d.neural_prompt();
        let e = ui.add(egui::TextEdit::multiline(&mut d.prompt).hint_text(hint).desired_rows(2).desired_width(w - 4.0));
        auto.add("generateAudio.prompt", visible(ui, e.rect), "Description");
        ui.label(
            egui::RichText::new("ACE-Step renders in the background (seconds to a minute) and places the music at the playhead when it is ready.")
                .color(t.text_dim)
                .size(11.0),
        );
        if let Some((f, status)) = &jobs.neural {
            ui.label(egui::RichText::new(status).color(t.text).size(11.5));
            ui.add(egui::ProgressBar::new(*f).show_percentage());
        }
        return;
    }

    // preview / regenerate
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        let busy = jobs.preview.is_some();
        let text = if busy {
            "Rendering…"
        } else if d.previewing {
            "■ Stop"
        } else {
            "▶ Preview"
        };
        let b = ui.add_enabled(!busy, egui::Button::new(text));
        auto.add("generateAudio.preview", visible(ui, b.rect), "Preview");
        if b.clicked() {
            actions.push(if d.previewing { Action::StopPreview } else { Action::Preview });
        }
        let b = ui.add_enabled(!busy, egui::Button::new("Regenerate"));
        auto.add("generateAudio.regenerate", visible(ui, b.rect), "Regenerate");
        if b.clicked() {
            d.seed = Some(next_seed(d.seed));
            actions.push(Action::Preview);
        }
        let note = match d.seed {
            Some(s) => format!("variation {s}"),
            None => "a new variation".into(),
        };
        ui.label(egui::RichText::new(note).color(t.text_faint).size(11.0));
    });

    // stems
    ui.add_space(10.0);
    label(ui, "Stems");
    // a grid keeps the level sliders aligned whatever the stem names' widths
    egui::Grid::new("generate-audio-stems").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.spacing_mut().slider_width = (w - 170.0).max(60.0);
        for (i, stem) in Stem::ALL.iter().enumerate() {
            let id = stem.id();
            let mut on = !d.muted.get(i).copied().unwrap_or(false);
            let c = ui.checkbox(&mut on, egui::RichText::new(cap(id)).size(12.0));
            auto.add(&format!("generateAudio.stem.{id}.mute"), visible(ui, c.rect), &format!("{} on", cap(id)));
            if let Some(m) = d.muted.get_mut(i) {
                *m = !on;
            }
            if let Some(level) = d.levels.get_mut(i) {
                if !level.is_finite() {
                    *level = 0.0;
                }
                let s = ui.add_enabled(on, egui::Slider::new(level, LEVEL_DB).step_by(0.5).suffix(" dB"));
                auto.add(&format!("generateAudio.stem.{id}.level"), visible(ui, s.rect), &format!("{} level", cap(id)));
            }
            ui.end_row();
        }
    });
    let c = ui.checkbox(&mut d.stems, "Place stems on their own tracks");
    auto.add("generateAudio.stems", visible(ui, c.rect), "Place stems on their own tracks");

    // instruments
    ui.add_space(10.0);
    label(ui, "Instruments");
    instruments(ui, d, auto, t, w, actions, &jobs);
}

/// The instruments the music plays on: the choice, what plays now, the GeneralUser GS download.
fn instruments(
    ui: &mut egui::Ui,
    d: &mut GenerateAudioDraft,
    auto: &mut crate::automation::Registry,
    t: &crate::theme::Tokens,
    w: f32,
    actions: &mut Vec<Action>,
    jobs: &MusicJobs,
) {
    let fonts = d.cache.fonts.clone().unwrap_or(Value::Null);
    let gugs = &fonts["soundfonts"][0];
    let installed = gugs["installed"] == json!(true);
    let selected = fonts["selected"].as_str().unwrap_or("auto").to_string();
    let choice_name = |c: &str| match c {
        "auto" => "Automatic (GeneralUser GS when installed)".to_string(),
        "oscillators" => "Built-in oscillator voices".to_string(),
        "generaluser-gs" => "GeneralUser GS".to_string(),
        path => format!("SoundFont: {}", std::path::Path::new(path).file_name().map_or(path.to_string(), |n| n.to_string_lossy().into_owned())),
    };
    let cb = egui::ComboBox::from_id_salt("generate-audio-instruments").selected_text(choice_name(&selected)).width(w).show_ui(ui, |ui| {
        for c in ["auto", "oscillators"] {
            if ui.selectable_label(selected == c, choice_name(c)).clicked() && selected != c {
                actions.push(Action::UseSoundfont(c.into()));
            }
        }
        let r = ui.add_enabled(installed, egui::Button::selectable(selected == "generaluser-gs", choice_name("generaluser-gs")));
        if r.clicked() && selected != "generaluser-gs" {
            actions.push(Action::UseSoundfont("generaluser-gs".into()));
        }
        if !installed {
            r.on_disabled_hover_text("Download it first (below)");
        }
    });
    auto.add("generateAudio.instruments", visible(ui, cb.response.rect), "Instruments");
    let active = &fonts["active"];
    let now = match active["kind"].as_str() {
        Some("soundfont") => format!("Playing on {}", active["name"].as_str().unwrap_or("a SoundFont")),
        _ => "Playing on the built-in oscillator voices".to_string(),
    };
    ui.label(egui::RichText::new(now).color(t.text_dim).size(11.0));
    if let Some(e) = active["error"].as_str().or(active["warning"].as_str()) {
        ui.label(egui::RichText::new(e).color(t.text_dim).size(11.0));
    }

    if !installed {
        ui.add_space(6.0);
        egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
            ui.set_width(w - 18.0);
            let size = gugs["missingBytes"].as_u64().or(gugs["size"].as_u64()).unwrap_or(0) as f64 / 1e6;
            match (&jobs.font, d.soundfont_job) {
                (Some((f, status)), Some(job)) => {
                    ui.label(egui::RichText::new(status).color(t.text).size(11.5));
                    ui.add(egui::ProgressBar::new(*f).show_percentage());
                    let b = ui.small_button("Cancel");
                    auto.add("generateAudio.soundfont.cancel", visible(ui, b.rect), "Cancel download");
                    if b.clicked() {
                        actions.push(Action::CancelJob(job));
                    }
                }
                _ if d.confirm_soundfont => {
                    ui.label(
                        egui::RichText::new(format!("Download GeneralUser GS {}? {size:.0} MB, once.", gugs["version"].as_str().unwrap_or("")))
                            .color(t.text)
                            .size(12.0)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "By {}, from {}. Licence: {}. {} Every file is checked against a pinned SHA-256.",
                            gugs["author"].as_str().unwrap_or(""),
                            gugs["source"].as_str().unwrap_or(""),
                            gugs["license"].as_str().unwrap_or(""),
                            gugs["provenance"].as_str().unwrap_or("")
                        ))
                        .color(t.text_dim)
                        .size(11.0),
                    );
                    egui::ScrollArea::vertical().id_salt("generate-audio-licence").max_height(140.0).show(ui, |ui| {
                        ui.label(egui::RichText::new(gugs["licenseText"].as_str().unwrap_or("")).monospace().size(10.0).color(t.text));
                    });
                    ui.horizontal(|ui| {
                        let ok = ui.button("I agree — download");
                        auto.add("generateAudio.soundfont.confirm", visible(ui, ok.rect), "Agree and download");
                        let no = ui.button("Cancel");
                        auto.add("generateAudio.soundfont.cancel", visible(ui, no.rect), "Cancel");
                        if ok.clicked() {
                            actions.push(Action::DownloadSoundfont);
                        }
                        if no.clicked() {
                            d.confirm_soundfont = false;
                        }
                    });
                }
                _ => {
                    ui.label(egui::RichText::new("Real instruments: GeneralUser GS, a free General MIDI SoundFont.").color(t.text).size(11.5));
                    let b = ui.button(format!("Get GeneralUser GS ({size:.0} MB)…"));
                    auto.add("generateAudio.soundfont.get", visible(ui, b.rect), "Get GeneralUser GS");
                    if b.clicked() {
                        d.confirm_soundfont = true;
                    }
                }
            }
        });
    }
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let e = ui.add(egui::TextEdit::singleline(&mut d.soundfont_path).hint_text("Your own .sf2 file (path)").desired_width((w - 70.0).max(80.0)));
        auto.add("generateAudio.soundfont.path", visible(ui, e.rect), "SoundFont file");
        let b = ui.add_enabled(!d.soundfont_path.trim().is_empty(), egui::Button::new("Use"));
        auto.add("generateAudio.soundfont.use", visible(ui, b.rect), "Use SoundFont file");
        if b.clicked() {
            actions.push(Action::UseSoundfont(d.soundfont_path.trim().to_string()));
        }
    });
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

fn run(app: &mut FilmcraftApp, a: Action) {
    match a {
        Action::Add => add(app),
        Action::Preview => {
            app.stop_music_preview();
            app.ui.generate_audio.previewing = false;
            let p = app.ui.generate_audio.preview_params();
            match app.session.execute("music.preview", p) {
                Ok(r) => {
                    // what is heard is what Add places
                    if let Some(seed) = r["seed"].as_u64() {
                        app.ui.generate_audio.seed = Some(seed);
                    }
                    match r["job"].as_u64() {
                        Some(job) => {
                            app.ui.generate_audio.preview_job = Some(job);
                            app.ui.status = "Rendering the music preview…".into();
                        }
                        None => {
                            let what = format!(
                                "{} {} · seed {}",
                                r["taxonomy"]["mood"].as_str().unwrap_or(""),
                                r["taxonomy"]["genre"].as_str().unwrap_or(""),
                                r["seed"]
                            );
                            play(app, &what, r["warning"].as_str().map(str::to_string));
                        }
                    }
                }
                Err(e) => app.ui.status = e.to_string(),
            }
        }
        Action::StopPreview => {
            app.stop_music_preview();
            app.ui.generate_audio.previewing = false;
            app.ui.generate_audio.cache.preview_ends = None;
        }
        Action::UseSoundfont(c) => {
            app.ui.status = match app.session.execute("music.useSoundfont", json!({"soundfont": c})) {
                Ok(r) => format!("Music plays on {}", r["active"]["name"].as_str().unwrap_or("the chosen instruments")),
                Err(e) => e.to_string(),
            };
            app.ui.generate_audio.cache.fonts = None;
        }
        Action::DownloadSoundfont => {
            app.ui.generate_audio.confirm_soundfont = false;
            match app.session.execute("music.downloadSoundfont", json!({"consent": true})) {
                Ok(r) => {
                    app.ui.generate_audio.soundfont_job = r["job"].as_u64();
                    if r["job"].is_null() {
                        app.ui.generate_audio.cache.fonts = None;
                    }
                    app.ui.status = "Downloading GeneralUser GS…".into();
                }
                Err(e) => app.ui.status = e.to_string(),
            }
        }
        Action::CancelJob(id) => {
            if let Some(j) = app.session.jobs.iter().find(|j| j.id == id) {
                j.progress.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// Add at playhead: the cue (music: what was previewed, or ACE-Step in the background).
fn add(app: &mut FilmcraftApp) {
    let d = &app.ui.generate_audio;
    if d.kind() == Kind::Music && d.engine == ENGINE_NEURAL && d.cache.neural == Some(true) {
        let mut p = json!({"prompt": d.neural_prompt(), "seconds": finite(d.seconds, 30.0).clamp(5.0, 240.0), "bpm": finite(d.bpm, 96.0).clamp(40.0, 200.0).round(), "atTime": app.session.playhead().0});
        if let Some(seed) = d.seed {
            p["seed"] = json!(seed);
        }
        match app.session.execute("music.generateNeural", p) {
            Ok(r) => {
                app.ui.generate_audio.neural_job = r["job"].as_u64();
                app.ui.status = "Generating music with ACE-Step…".into();
            }
            Err(e) => app.ui.status = e.to_string(),
        }
        return;
    }
    let p = d.params();
    app.ui.status = match app.session.execute("audio.generate", p) {
        Ok(r) => {
            let what = r["path"].as_str().and_then(|p| std::path::Path::new(p).file_stem()).map_or("the cue".into(), |s| s.to_string_lossy().into_owned());
            let stems = r["stems"].as_array().map_or(0, Vec::len);
            if stems > 0 { format!("Added {what} as {stems} stems") } else { format!("Added {what} on {}", r["track"].as_str().unwrap_or("an audio track")) }
        }
        Err(e) => e.to_string(),
    };
}

/// A seed different from `prev`, from the clock (a splitmix step so close times differ widely).
fn next_seed(prev: Option<u64>) -> u64 {
    let now = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut z = now ^ prev.unwrap_or(0).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) % 1_000_000
}

/// The part of `r` the scroll area shows (what a click can reach).
fn visible(ui: &egui::Ui, r: Rect) -> Rect {
    r.intersect(ui.clip_rect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_carry_only_what_the_kind_uses() {
        let d = GenerateAudioDraft::default();
        let p = d.params();
        assert_eq!(p["kind"], "whoosh");
        assert!(p.get("mood").is_none() && p.get("key").is_none() && p.get("seed").is_none());
        let music = GenerateAudioDraft { kind: "music".into(), seconds: 5000.0, bpm: 999.0, seed: Some(7), ..Default::default() };
        let p = music.params();
        assert_eq!((p["mood"].as_str(), p["key"].as_str(), p["seed"].as_u64()), (Some("epic"), Some("A"), Some(7)));
        assert_eq!(p["bpm"], 200.0, "hostile values are clamped");
        assert_eq!(p["duration"], Kind::Music.max_seconds());
        assert!(p.get("genre").is_none() && p.get("levels").is_none() && p.get("mute").is_none() && p.get("stems").is_none());
        // an unknown kind (from a hand-edited state) falls back to a whoosh
        assert_eq!(GenerateAudioDraft { kind: "laser".into(), ..Default::default() }.params()["kind"], "whoosh");
    }

    #[test]
    fn music_params_carry_the_feel_and_the_stem_mix() {
        let d = GenerateAudioDraft {
            kind: "music".into(),
            mood: "calm".into(),
            genre: "piano".into(),
            intensity: f64::NAN,
            levels: [0.0, -6.0, 99.0, 0.0],
            muted: [true, false, false, false],
            stems: true,
            ..Default::default()
        };
        let p = d.params();
        assert_eq!(p["genre"], "piano");
        assert_eq!(p["energy"], 0.7, "a non-number energy falls back");
        assert_eq!(p["mute"], json!(["drums"]));
        assert_eq!(p["levels"], json!({"bass": -6.0, "melody": 6.0}));
        assert_eq!(p["stems"], true);
        let q = d.preview_params();
        assert!(q.get("stems").is_none() && q.get("kind").is_none());
        assert_eq!(q["mute"], json!(["drums"]));
        // an unknown genre or mood (hand-edited) is the mood's usual / epic
        let odd = GenerateAudioDraft { kind: "music".into(), mood: "jazzy".into(), genre: "polka".into(), ..Default::default() };
        assert_eq!(odd.params()["mood"], "epic");
        assert!(odd.params().get("genre").is_none());
        assert!(odd.neural_prompt().contains("epic cinematic"));
        assert_eq!(GenerateAudioDraft { prompt: "  warm synthwave ".into(), ..Default::default() }.neural_prompt(), "warm synthwave");
    }

    #[test]
    fn a_new_variation_differs() {
        let a = next_seed(None);
        assert_ne!(next_seed(Some(a)), a);
    }
}

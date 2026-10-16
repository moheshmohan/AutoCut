//! Text to Speech panel: write what the narrator says, pick a language, voice, vocal pitch and pace,
//! hear it, and save it as a narration clip. Selecting a narration clip on the timeline loads its
//! script and settings here (and brings the panel forward), and Save regenerates it in place.
//! Every action is a `tts.*` command, so the panel, MCP and the control channel share one path.
//!
//! Synthesis runs as a background job (`tts.render`); when it finishes, the panel runs the action
//! (preview, create or edit), which then uses the cached audio, so the window never stalls. The
//! natural voices need a one-time download that the user confirms in the panel (size, source,
//! licence); it runs as a job with progress and Cancel.
//!
//! Automation ids: `tts.language`, `tts.voice`, `tts.hearVoice`, `tts.advanced` (twirl),
//! `tts.pitch`, `tts.pace`, `tts.text`, `tts.addPause`, `tts.preview`, `tts.save`, `tts.new`,
//! `tts.download`, `tts.download.confirm`, `tts.download.cancel`, `tts.cloneOpen` (twirl),
//! `tts.cloneEngine`, `tts.cloneModel`, `tts.cloneName`, `tts.cloneConsent`, `tts.clone`,
//! `tts.cloneDownload.confirm`, `tts.cloneDownload.cancel`.
//!
//! **Cloned voices** (`clone:<name>`) are listed in the voice menu, with how each was made
//! (natural or timbre only, model and quality) under the menu when one is chosen. "Clone a voice"
//! makes one from the speech in the selected clips (`tts.cloneVoice`, a background job) once the
//! user has named it and confirmed it is their own voice or that the speaker agreed. It offers
//! the engine — *Natural* (VoxCPM: the model speaks in the speaker's own voice, accent and rhythm
//! included; VoxCPM1.5 faster, VoxCPM2 best quality) or *Fast* (OpenVoice: a Kokoro voice
//! re-voiced, timbre only) — and the chosen model's one-time download, confirmed in the panel
//! with its size, source and licence, with progress and Cancel.

use egui::{Align2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_project::{ClipId, VocalPitch};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::theme::Tokens;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 48.0;

/// What the panel is writing (UI state, serde so agents can read it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TtsDraft {
    pub text: String,
    pub language: String,
    pub voice: String,
    pub pitch: VocalPitch,
    pub pace: f64,
    pub advanced_open: bool,
    /// The narration clip and item the draft was loaded from (None: a new narration).
    pub loaded_from: Option<(u64, u64)>,
    /// An action waiting for its synthesis job.
    pub pending: Option<Pending>,
    /// The download confirmation is showing.
    pub confirm_download: bool,
    /// The running natural-voice download job.
    pub download_job: Option<u64>,
    /// The "Clone a voice" section is open.
    pub clone_open: bool,
    /// Name for the next cloned voice.
    pub clone_name: String,
    /// The user confirmed it is their voice, or the speaker agreed.
    pub clone_consent: bool,
    /// The running cloning (or cloning-model download) job, and the voice it makes.
    pub clone_job: Option<(u64, Option<String>)>,
    /// How new clones are made: `voxcpm` (natural) or `openvoice` (timbre only).
    pub clone_engine: String,
    /// The VoxCPM release natural clones use.
    pub clone_model: String,
    /// The cloning model's download confirmation is showing.
    pub clone_confirm: bool,
}

/// A command to run when the synthesis job `job` finishes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pending {
    pub cmd: String,
    pub params: serde_json::Value,
    /// Play the preview afterwards.
    pub play: bool,
    pub job: u64,
}

impl Default for TtsDraft {
    fn default() -> Self {
        TtsDraft {
            text: String::new(),
            language: "en-US".into(),
            voice: if natural_installed() { "kokoro-heart".into() } else { filmcraft_tts::default_voice_id().into() },
            pitch: VocalPitch::Default,
            pace: 1.0,
            advanced_open: true,
            loaded_from: None,
            pending: None,
            confirm_download: false,
            download_job: None,
            clone_open: false,
            clone_name: String::new(),
            clone_consent: false,
            clone_job: None,
            clone_engine: "voxcpm".into(),
            clone_model: filmcraft_engine::voice_clone::default_natural_model().into(),
            clone_confirm: false,
        }
    }
}

impl TtsDraft {
    fn params(&self) -> serde_json::Value {
        json!({"text": self.text, "voice": self.voice, "pitch": self.pitch.id(), "pace": self.pace})
    }
}

/// The natural voices can speak: this build has them and their package is downloaded (a build
/// without them must not default to a voice it cannot use, even where the files are on disk).
fn natural_installed() -> bool {
    filmcraft_tts::NATURAL_VOICES && filmcraft_engine::transcript::models_dir().as_deref().is_some_and(filmcraft_tts::catalog::installed)
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

/// Start an action: synthesize in the background (or reuse the cache), then run `cmd`.
fn start(app: &mut FilmcraftApp, cmd: &str, params: serde_json::Value, play: bool) {
    let mut render = params.clone();
    if let Some(o) = render.as_object_mut() {
        o.remove("dir");
        o.remove("time");
    }
    match app.session.execute("tts.render", render) {
        Ok(r) => match r["job"].as_u64() {
            Some(job) => {
                app.ui.tts.pending = Some(Pending { cmd: cmd.to_string(), params, play, job });
                app.ui.status = "Synthesizing narration…".into();
            }
            None => run_now(app, cmd, params, play),
        },
        Err(e) => app.ui.status = e.to_string(),
    }
}

fn run_now(app: &mut FilmcraftApp, cmd: &str, params: serde_json::Value, play: bool) {
    match app.session.execute(cmd, params) {
        Ok(r) => {
            if play {
                if let Err(e) = app.play_tts_preview() {
                    app.ui.status = e;
                }
            } else {
                app.ui.status = match cmd {
                    "tts.create" => format!("Narration added on {}", r["track"].as_str().unwrap_or("an audio track")),
                    _ => "Narration updated".into(),
                };
            }
        }
        Err(e) => app.ui.status = e.to_string(),
    }
}

/// Finish a pending action whose synthesis job is done.
fn poll(app: &mut FilmcraftApp) {
    if let Some(p) = app.ui.tts.pending.clone() {
        match job_outcome(app, p.job) {
            None => {}
            Some(Ok(())) => {
                app.ui.tts.pending = None;
                run_now(app, &p.cmd, p.params, p.play);
            }
            Some(Err(e)) => {
                app.ui.tts.pending = None;
                app.ui.status = format!("Text to Speech: {e}");
            }
        }
    }
    if let Some((j, voice)) = app.ui.tts.clone_job.clone()
        && let Some(r) = job_outcome(app, j)
    {
        app.ui.tts.clone_job = None;
        app.ui.status = match (r, voice) {
            (Ok(()), Some(v)) => {
                app.ui.tts.voice = v.clone();
                format!("Cloned voice ready: {v}")
            }
            (Ok(()), None) => "Cloning model installed".into(),
            (Err(e), _) => format!("Voice cloning failed: {e}"),
        };
    }
    if let Some(j) = app.ui.tts.download_job
        && let Some(r) = job_outcome(app, j)
    {
        app.ui.tts.download_job = None;
        app.ui.status = match r {
            Ok(()) => "Natural voices installed".into(),
            Err(e) => format!("Download of the natural voices failed: {e}"),
        };
    }
}

/// The selected narration clip and its item, if any.
fn selected_narration(app: &FilmcraftApp) -> Option<(ClipId, u64)> {
    let clip = filmcraft_engine::narration::target_clip(&app.session, &serde_json::Value::Null)?;
    let item = app.session.active_sequence()?.find_item(clip)?.1.item;
    Some((clip, item.0))
}

/// Load the selected narration clip into the draft when the selection moves to a different one,
/// and bring the panel forward. Runs every frame, also when the panel is hidden.
pub fn follow_selection(app: &mut FilmcraftApp) {
    poll(app);
    let sel = selected_narration(app);
    match sel {
        Some((clip, item)) if app.ui.tts.loaded_from != Some((clip.0, item)) => {
            let first_time = app.ui.tts.loaded_from.map(|(c, _)| c) != Some(clip.0);
            if let Some(n) = app.session.project.narrations.get(&filmcraft_project::ItemId(item)).cloned() {
                let d = &mut app.ui.tts;
                d.text = n.text;
                d.language = n.language;
                d.voice = n.voice;
                d.pitch = n.pitch;
                d.pace = n.pace;
                d.loaded_from = Some((clip.0, item));
                if first_time {
                    app.show_panel(crate::dock::PanelKind::TextToSpeech);
                }
            }
        }
        None if app.ui.tts.loaded_from.is_some() => {
            // deselected: keep the text so it can be saved as a new narration
            app.ui.tts.loaded_from = None;
        }
        _ => {}
    }
}

enum Action {
    Run(&'static str, serde_json::Value, bool),
    New,
    Download,
    CancelDownload,
    /// Clone a voice named this from the selected clips.
    Clone(String),
    /// Download this cloning model (confirmed).
    DownloadCloneModel(String),
    CancelCloneJob,
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let painter = ui.painter().clone();
    painter.rect_filled(rect, 0.0, t.panel_bg);
    let models = filmcraft_engine::transcript::models_dir();
    let voices = filmcraft_tts::voices_in(models.as_deref());
    let installed = filmcraft_tts::NATURAL_VOICES && models.as_deref().is_some_and(filmcraft_tts::catalog::installed);
    let busy = app.ui.tts.pending.is_some();
    let download = app
        .ui
        .tts
        .download_job
        .and_then(|id| app.session.jobs.iter().find(|j| j.id == id))
        .map(|j| (j.progress.fraction(), j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()));
    let editing = app.ui.tts.loaded_from.map(|(c, _)| c);
    // progress of the jobs the clone section follows (read before the draft is borrowed)
    let clone_jobs: Vec<(u64, f32, String)> = app
        .session
        .jobs
        .iter()
        .map(|j| (j.id, j.progress.fraction(), j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()))
        .collect();
    let mut actions: Vec<Action> = Vec::new();
    let auto = &mut app.auto;
    let d = &mut app.ui.tts;

    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - FOOTER_H));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("tts-body"));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut child, |ui| {
        ui.set_max_width(ui.available_width() - 6.0);
        let w = ui.available_width();
        // mode line: editing a clip, or writing a new narration
        ui.horizontal(|ui| {
            let mode = if editing.is_some() { "Editing the selected narration" } else { "New narration" };
            ui.label(egui::RichText::new(mode).color(t.text_dim).size(11.5));
            if editing.is_some() {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let b = ui.small_button("New");
                    auto.add("tts.new", visible(ui, b.rect), "New narration");
                    if b.clicked() {
                        actions.push(Action::New);
                    }
                });
            }
        });
        ui.add_space(6.0);
        label(ui, &t, "Language");
        let lang_name = filmcraft_tts::LANGUAGES.iter().find(|(id, _)| *id == d.language).map_or(d.language.as_str(), |(_, n)| n);
        let cb = egui::ComboBox::from_id_salt("tts-language").selected_text(lang_name).width(w).show_ui(ui, |ui| {
            for (id, name) in filmcraft_tts::LANGUAGES {
                if ui.selectable_label(d.language == *id, *name).clicked() {
                    d.language = id.to_string();
                }
            }
        });
        auto.add("tts.language", visible(ui, cb.response.rect), "Language");
        ui.add_space(10.0);
        label(ui, &t, "Voice");
        let cloned_list = filmcraft_engine::voice_clone::cloned_json();
        let cloned: Vec<(String, String, String)> = cloned_list
            .as_array()
            .map(|a| a.iter().filter_map(|v| Some((v["id"].as_str()?.to_string(), v["name"].as_str()?.to_string(), clone_summary(v)))).collect())
            .unwrap_or_default();
        let voice_name = voices
            .iter()
            .find(|v| v.id == d.voice)
            .map(|v| v.name.to_string())
            .or_else(|| cloned.iter().find(|(id, _, _)| *id == d.voice).map(|(_, n, _)| format!("{n} (cloned)")))
            .unwrap_or_else(|| d.voice.clone());
        let cb = egui::ComboBox::from_id_salt("tts-voice").selected_text(voice_name).width(w).show_ui(ui, |ui| {
            if !cloned.is_empty() {
                ui.label(egui::RichText::new("Your cloned voices").color(t.text_faint).size(11.0));
                for (id, name, summary) in &cloned {
                    if ui.selectable_label(d.voice == *id, name).on_hover_text(summary).clicked() {
                        d.voice = id.clone();
                    }
                }
            }
            for (engine, heading) in [("neural", "Natural voices"), ("basic", "Basic voices (built in)")] {
                ui.label(egui::RichText::new(heading).color(t.text_faint).size(11.0));
                for v in voices.iter().filter(|v| v.language == d.language && v.engine == engine) {
                    let label = if v.engine == "neural" && !v.installed { format!("{} (download)", v.name) } else { v.name.to_string() };
                    let r = ui.selectable_label(d.voice == v.id, label).on_hover_text(v.description);
                    if r.clicked() {
                        d.voice = v.id.to_string();
                    }
                }
            }
        });
        auto.add("tts.voice", visible(ui, cb.response.rect), "Voice");
        if let Some((_, _, summary)) = cloned.iter().find(|(id, _, _)| *id == d.voice) {
            ui.label(egui::RichText::new(summary).color(t.text_faint).size(11.0));
        }
        // clone a voice from the selected clip
        ui.add_space(4.0);
        let twirl = ui.add(
            egui::Label::new(egui::RichText::new(if d.clone_open { "▾ Clone a voice" } else { "▸ Clone a voice" }).color(t.text_dim).size(11.5))
                .sense(Sense::click()),
        );
        auto.add("tts.cloneOpen", visible(ui, twirl.rect), "Clone a voice");
        if twirl.clicked() {
            d.clone_open = !d.clone_open;
        }
        if d.clone_open {
            let models = filmcraft_engine::voice_clone::models_json();
            let model_id = if d.clone_engine == "openvoice" { "openvoice-v2".to_string() } else { d.clone_model.clone() };
            let model = models.as_array().and_then(|a| a.iter().find(|m| m["id"] == model_id.as_str())).cloned().unwrap_or_default();
            let model_ready = model["installed"].as_bool().unwrap_or(false);
            let clone_progress = d.clone_job.as_ref().and_then(|(id, _)| app_jobs_progress(&clone_jobs, *id));
            egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(w - 18.0);
                let cb = egui::ComboBox::from_id_salt("tts-clone-engine")
                    .selected_text(if d.clone_engine == "openvoice" { "Fast: timbre only" } else { "Natural: your accent and rhythm" })
                    .width(w - 18.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(d.clone_engine != "openvoice", "Natural: your accent and rhythm")
                            .on_hover_text("VoxCPM speaks directly in the cloned voice: timbre, accent, rhythm and delivery.")
                            .clicked()
                        {
                            d.clone_engine = "voxcpm".into();
                            d.clone_confirm = false;
                        }
                        if ui
                            .selectable_label(d.clone_engine == "openvoice", "Fast: timbre only")
                            .on_hover_text("A natural Kokoro voice re-voiced into the cloned timbre (OpenVoice). Faster; the accent and rhythm stay Kokoro's.")
                            .clicked()
                        {
                            d.clone_engine = "openvoice".into();
                            d.clone_confirm = false;
                        }
                    });
                auto.add("tts.cloneEngine", visible(ui, cb.response.rect), "Cloning engine");
                if d.clone_engine != "openvoice" {
                    let label = |m: &serde_json::Value| m["name"].as_str().unwrap_or("").to_string();
                    let current = models
                        .as_array()
                        .and_then(|a| a.iter().find(|m| m["id"] == d.clone_model.as_str()))
                        .map(label)
                        .unwrap_or_else(|| d.clone_model.clone());
                    let cb = egui::ComboBox::from_id_salt("tts-clone-model").selected_text(current).width(w - 18.0).show_ui(ui, |ui| {
                        for m in models.as_array().into_iter().flatten().filter(|m| m["engine"] == "voxcpm") {
                            let id = m["id"].as_str().unwrap_or("").to_string();
                            let installed = m["installed"].as_bool().unwrap_or(false);
                            let text = if installed { label(m) } else { format!("{} (download)", label(m)) };
                            if ui.selectable_label(d.clone_model == id, text).on_hover_text(m["description"].as_str().unwrap_or("")).clicked() {
                                d.clone_model = id;
                                d.clone_confirm = false;
                            }
                        }
                    });
                    auto.add("tts.cloneModel", visible(ui, cb.response.rect), "Cloning model");
                }
                let hint = if d.clone_engine == "openvoice" {
                    "Uses the speech in the selected clip (3 s or more). The timbre is copied; the accent and rhythm are the natural voice's."
                } else {
                    "Uses the best 10–25 s of speech in the selected clip. The voice speaks with your timbre, accent and rhythm; a clear recording works best."
                };
                ui.label(egui::RichText::new(hint).color(t.text_dim).size(11.0));
                let name = ui.add(egui::TextEdit::singleline(&mut d.clone_name).hint_text("Name, e.g. My voice").desired_width(w - 18.0));
                auto.add("tts.cloneName", visible(ui, name.rect), "Cloned voice name");
                let consent = ui.checkbox(&mut d.clone_consent, "This is my voice, or the speaker agreed to it being cloned");
                auto.add("tts.cloneConsent", visible(ui, consent.rect), "Consent to clone");
                let running = d.clone_job.is_some();
                let ready = !d.clone_name.trim().is_empty() && d.clone_consent && !running;
                let mb = model["size"].as_u64().unwrap_or(0) as f64 / 1e6;
                let size = if mb >= 1000.0 { format!("{:.2} GB", mb / 1000.0) } else { format!("{mb:.0} MB") };
                if let Some((frac, status)) = clone_progress.clone().filter(|_| d.clone_job.as_ref().is_some_and(|(_, v)| v.is_none())) {
                    // the model download in progress
                    ui.label(egui::RichText::new(status).color(t.text).size(11.5));
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                    let b = ui.small_button("Cancel");
                    auto.add("tts.cloneDownload.cancel", visible(ui, b.rect), "Cancel download");
                    if b.clicked() {
                        actions.push(Action::CancelCloneJob);
                    }
                } else if !model_ready && d.clone_confirm && !running {
                    ui.label(
                        egui::RichText::new(format!("Download {}? {size}, once.", short_name(model["name"].as_str().unwrap_or("the cloning model"))))
                            .color(t.text)
                            .size(12.0)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "From {}. Licence: {}. Every file is checked against a pinned SHA-256.",
                            model["source"].as_str().unwrap_or(""),
                            model["license"].as_str().unwrap_or("")
                        ))
                        .color(t.text_dim)
                        .size(11.0),
                    );
                    ui.horizontal(|ui| {
                        let ok = ui.button("Download");
                        auto.add("tts.cloneDownload.confirm", visible(ui, ok.rect), "Download cloning model");
                        if ok.clicked() {
                            d.clone_confirm = false;
                            actions.push(Action::DownloadCloneModel(model_id.clone()));
                        }
                        if ui.button("Cancel").clicked() {
                            d.clone_confirm = false;
                        }
                    });
                } else {
                    let text = match (running, model_ready) {
                        (true, _) => "Working…".to_string(),
                        (false, true) => "Clone from selected clip".to_string(),
                        (false, false) => format!("Download cloning model ({size})…"),
                    };
                    let b = ui.add_enabled(ready || (!model_ready && !running), egui::Button::new(text));
                    auto.add("tts.clone", visible(ui, b.rect), "Clone a voice");
                    if b.clicked() {
                        if model_ready {
                            actions.push(Action::Clone(d.clone_name.trim().to_string()));
                        } else {
                            d.clone_confirm = true;
                        }
                    }
                    if let (true, Some((_, status))) = (running, &clone_progress) {
                        ui.label(egui::RichText::new(status).color(t.text_dim).size(11.0));
                    }
                }
            });
        }
        let natural = filmcraft_tts::catalog::find(&d.voice).is_some();
        if natural && !installed {
            ui.add_space(6.0);
            egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(w - 18.0);
                match &download {
                    Some((frac, status)) => {
                        ui.label(egui::RichText::new(status).color(t.text).size(11.5));
                        ui.add(egui::ProgressBar::new(*frac).show_percentage());
                        let b = ui.small_button("Cancel");
                        auto.add("tts.download.cancel", visible(ui, b.rect), "Cancel download");
                        if b.clicked() {
                            actions.push(Action::CancelDownload);
                        }
                    }
                    None if d.confirm_download => {
                        let mb = filmcraft_tts::catalog::size() as f64 / 1e6;
                        ui.label(egui::RichText::new(format!("Download the natural voices? {mb:.0} MB, once.")).color(t.text).size(12.0).strong());
                        ui.label(
                            egui::RichText::new(format!(
                                "From {} and github.com/cmusphinx/cmudict (pronunciations). Licence: {}. Every file is checked against a pinned SHA-256.",
                                filmcraft_tts::catalog::SOURCE,
                                filmcraft_tts::catalog::LICENSE
                            ))
                            .color(t.text_dim)
                            .size(11.0),
                        );
                        ui.horizontal(|ui| {
                            let ok = ui.button("Download");
                            auto.add("tts.download.confirm", visible(ui, ok.rect), "Download");
                            let no = ui.button("Cancel");
                            if ok.clicked() {
                                actions.push(Action::Download);
                            }
                            if no.clicked() {
                                d.confirm_download = false;
                            }
                        });
                    }
                    None => {
                        ui.label(egui::RichText::new("Natural voices need a one-time download.").color(t.text).size(11.5));
                        let b = ui.button(format!("Download natural voices ({:.0} MB)…", filmcraft_tts::catalog::size() as f64 / 1e6));
                        auto.add("tts.download", visible(ui, b.rect), "Download natural voices");
                        if b.clicked() {
                            d.confirm_download = true;
                        }
                    }
                }
            });
        }
        let voice_ready = !natural || installed;
        ui.add_space(4.0);
        let (r, resp) = ui.allocate_exact_size(vec2(130.0, 22.0), Sense::click());
        let col = if resp.hovered() { t.accent_hover } else { t.accent };
        icons::paint(ui.painter(), Rect::from_center_size(pos2(r.min.x + 8.0, r.center().y), vec2(14.0, 14.0)), Icon::Play, col);
        ui.painter().text(pos2(r.min.x + 22.0, r.center().y), Align2::LEFT_CENTER, "Hear this voice", Tokens::semibold(12.0), col);
        auto.add("tts.hearVoice", visible(ui, r), "Hear this voice");
        if resp.clicked() && voice_ready && !busy {
            actions.push(Action::Run("tts.preview", json!({"sample": true, "voice": d.voice, "pitch": d.pitch.id(), "pace": d.pace}), true));
        }
        ui.add_space(8.0);
        let (hr, open) = crate::widgets::section_header(ui, egui::Id::new("tts-advanced"), "Advanced", d.advanced_open, &t, true);
        auto.add("tts.advanced", visible(ui, hr.rect), "Advanced");
        d.advanced_open = open;
        if open {
            ui.add_space(4.0);
            label(ui, &t, "Vocal pitch");
            let cb = egui::ComboBox::from_id_salt("tts-pitch").selected_text(d.pitch.label()).width(w).show_ui(ui, |ui| {
                for p in VocalPitch::ALL {
                    if ui.selectable_label(d.pitch == p, p.label()).clicked() {
                        d.pitch = p;
                    }
                }
            });
            auto.add("tts.pitch", visible(ui, cb.response.rect), "Vocal pitch");
            ui.add_space(10.0);
            label(ui, &t, "Pace");
            ui.spacing_mut().slider_width = w - 56.0;
            let s = ui.add(egui::Slider::new(&mut d.pace, filmcraft_tts::MIN_PACE..=filmcraft_tts::MAX_PACE).step_by(0.05).fixed_decimals(2).suffix("×"));
            auto.add("tts.pace", visible(ui, s.rect), "Pace");
            let (r, _) = ui.allocate_exact_size(vec2(w - 56.0, 14.0), Sense::hover());
            for (x, l) in [(0.0, "0.5×"), (1.0 / 3.0, "1×"), (1.0, "2×")] {
                let px = r.min.x + 6.0 + (r.width() - 12.0) * x;
                ui.painter().text(pos2(px, r.center().y), Align2::CENTER_CENTER, l, Tokens::ui(10.5), t.text_faint);
            }
        }
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Text").color(t.text).size(12.5).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (ir, i) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                icons::paint(ui.painter(), ir, Icon::Info, t.text_faint);
                i.on_hover_text("Add pause inserts [pause 1s]. Change the number for a longer or shorter pause, e.g. [pause 500ms] or [pause 2s].");
                let b = ui.small_button("Add pause");
                auto.add("tts.addPause", visible(ui, b.rect), "Add pause");
                if b.clicked() {
                    if !d.text.is_empty() && !d.text.ends_with(' ') {
                        d.text.push(' ');
                    }
                    d.text.push_str(filmcraft_tts::script::PAUSE_MARKER);
                    d.text.push(' ');
                }
            });
        });
        ui.add_space(4.0);
        let te = egui::TextEdit::multiline(&mut d.text)
            .hint_text("Type what you want the voice to say")
            .desired_width(w)
            .desired_rows(8)
            .char_limit(filmcraft_tts::MAX_TEXT_BYTES);
        let r = ui.add(te);
        auto.add("tts.text", visible(ui, r.rect), "Text");
    });

    // footer: Preview | Save
    let fy = rect.max.y - FOOTER_H + 10.0;
    let half = (rect.width() - PAD * 2.0 - 10.0) / 2.0;
    let natural = filmcraft_tts::catalog::find(&d.voice).is_some();
    let can_say = filmcraft_tts::check_text(&d.text).is_ok() && (!natural || installed) && !busy;
    let pr = Rect::from_min_size(pos2(rect.min.x + PAD, fy), vec2(half, 28.0));
    if button(ui, &t, pr, Icon::Play, if busy { "Synthesizing…" } else { "Preview" }, false, can_say, "tts-preview").clicked() && can_say {
        actions.push(Action::Run("tts.preview", d.params(), true));
    }
    auto.add("tts.preview", pr, "Preview");
    let sr = Rect::from_min_size(pos2(pr.max.x + 10.0, fy), vec2(half, 28.0));
    let can_save = can_say && app.session.active_sequence().is_some();
    if button(ui, &t, sr, Icon::Plus, if editing.is_some() { "Save" } else { "Add to timeline" }, true, can_save, "tts-save").clicked() && can_save {
        let mut p = d.params();
        match editing {
            Some(c) => {
                p["clip"] = json!(c);
                actions.push(Action::Run("tts.edit", p, false));
            }
            None => actions.push(Action::Run("tts.create", p, false)),
        }
    }
    auto.add("tts.save", sr, "Save");

    for a in actions {
        match a {
            Action::New => {
                let keep = (app.ui.tts.language.clone(), app.ui.tts.voice.clone(), app.ui.tts.pitch, app.ui.tts.pace);
                app.session.state.selection.clear();
                let dl = app.ui.tts.download_job;
                app.ui.tts = TtsDraft { language: keep.0, voice: keep.1, pitch: keep.2, pace: keep.3, download_job: dl, ..Default::default() };
            }
            Action::Run(cmd, p, play) => start(app, cmd, p, play),
            Action::Download => {
                app.ui.tts.confirm_download = false;
                match app.session.execute("tts.downloadVoices", json!({})) {
                    Ok(r) => app.ui.tts.download_job = r["job"].as_u64(),
                    Err(e) => app.ui.status = e.to_string(),
                }
            }
            Action::Clone(name) => {
                let clips: Vec<u64> = app.session.state.selection.iter().map(|c| c.0).collect();
                if clips.is_empty() {
                    app.ui.status = "Select a clip with the voice to clone".into();
                } else {
                    let mut p = json!({"name": name, "consent": true, "clips": clips, "engine": app.ui.tts.clone_engine});
                    if app.ui.tts.clone_engine != "openvoice" {
                        p["model"] = json!(app.ui.tts.clone_model);
                    }
                    match app.session.execute("tts.cloneVoice", p) {
                        Ok(r) => {
                            app.ui.tts.clone_job = r["job"].as_u64().map(|j| (j, r["voice"].as_str().map(str::to_string)));
                            app.ui.status = format!("Cloning voice {name}…");
                        }
                        Err(e) => app.ui.status = e.to_string(),
                    }
                }
            }
            Action::DownloadCloneModel(model) => match app.session.execute("tts.downloadCloneModel", json!({"model": model})) {
                Ok(r) => app.ui.tts.clone_job = r["job"].as_u64().map(|j| (j, None)),
                Err(e) => app.ui.status = e.to_string(),
            },
            Action::CancelCloneJob => {
                if let Some((id, _)) = app.ui.tts.clone_job
                    && let Some(j) = app.session.jobs.iter().find(|j| j.id == id)
                {
                    j.progress.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            Action::CancelDownload => {
                if let Some(id) = app.ui.tts.download_job
                    && let Some(j) = app.session.jobs.iter().find(|j| j.id == id)
                {
                    j.progress.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
}

/// `(fraction, status)` of job `id` among `jobs`.
fn app_jobs_progress(jobs: &[(u64, f32, String)], id: u64) -> Option<(f32, String)> {
    jobs.iter().find(|(j, _, _)| *j == id).map(|(_, f, s)| (*f, s.clone()))
}

/// "Natural cloning, faster (VoxCPM1.5)" → "VoxCPM1.5"; a name without parentheses as is.
fn short_name(name: &str) -> &str {
    name.rsplit_once('(').and_then(|(_, r)| r.strip_suffix(')')).filter(|s| !s.is_empty()).unwrap_or(name)
}

/// One line on how a cloned voice (`tts.clonedVoices` entry) was made.
fn clone_summary(v: &serde_json::Value) -> String {
    match v["engine"].as_str() {
        Some("voxcpm") => format!(
            "Cloned · natural ({}) · {} steps, guidance {:.1}",
            v["modelName"].as_str().filter(|s| !s.is_empty()).map(short_name).unwrap_or_else(|| v["model"].as_str().unwrap_or("VoxCPM")),
            v["steps"].as_u64().unwrap_or(10),
            v["cfg"].as_f64().unwrap_or(2.0)
        ),
        _ => format!("Cloned · timbre only (OpenVoice) through {}", v["baseVoice"].as_str().unwrap_or("a natural voice")),
    }
}

/// The part of `r` the scroll area shows (what a click can reach).
fn visible(ui: &egui::Ui, r: Rect) -> Rect {
    r.intersect(ui.clip_rect())
}

fn label(ui: &mut egui::Ui, t: &Tokens, text: &str) {
    ui.label(egui::RichText::new(text).color(t.text).size(12.5).strong());
    ui.add_space(2.0);
}

#[allow(clippy::too_many_arguments)]
fn button(ui: &mut egui::Ui, t: &Tokens, r: Rect, icon: Icon, text: &str, primary: bool, active: bool, id: &str) -> egui::Response {
    let resp = ui.interact(r, egui::Id::new(id), Sense::click());
    let p = ui.painter();
    let (bg, fg) = match (primary, active) {
        (true, true) => (if resp.hovered() { t.accent_hover } else { t.accent }, egui::Color32::WHITE),
        (_, false) => (t.field_bg, t.text_faint),
        (false, true) => (if resp.hovered() { t.hover } else { t.field_bg }, t.text),
    };
    p.rect_filled(r, 4.0, bg);
    if !primary {
        p.rect_stroke(r, 4.0, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    }
    let g = p.layout_no_wrap(text.to_string(), Tokens::semibold(12.5), fg);
    let total = 18.0 + g.size().x;
    let x0 = r.center().x - total / 2.0;
    icons::paint(p, Rect::from_center_size(pos2(x0 + 7.0, r.center().y), vec2(14.0, 14.0)), icon, fg);
    p.galley(pos2(x0 + 18.0, r.center().y - g.size().y / 2.0), g, fg);
    resp
}

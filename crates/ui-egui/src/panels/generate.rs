//! Generate panel: make pictures, clips and sounds with the generation tools installed on this
//! computer (`generate.providers`), fill a gap of the timeline, or change what the frame under
//! the playhead shows. Each button runs the same `generate.*` command agents and the CLI use; the
//! work is a background job, and its result is imported (and placed) when it finishes.
//!
//! The tool in use and what its licence asks to be credited are shown under the tool menu.
//!
//! Automation ids: `generate.mode.<image|video|sound|fill|augment>`, `generate.provider`,
//! `generate.prompt`, `generate.useFrame`, `generate.useSelection`, `generate.transparent`,
//! `generate.place`, `generate.soundKind`, `generate.length`, `generate.track`,
//! `generate.allGaps`, `generate.animate`, `generate.variation`, `generate.run`,
//! `generate.stop`, `generate.history.<n>.place`.

use egui::{Rect, pos2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::FilmcraftApp;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 48.0;
/// The modes, with their names.
const MODES: [(&str, &str); 5] = [("image", "Picture"), ("video", "Clip"), ("sound", "Sound"), ("fill", "Fill gap"), ("augment", "Augment")];
/// Generated items listed under History.
const HISTORY_SHOWN: usize = 8;

/// What is being generated (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateDraft {
    /// `image`, `video`, `sound`, `fill` (a gap of the timeline) or `augment` (the frame under
    /// the playhead, changed).
    pub mode: String,
    /// What to make (for `augment`: what to change).
    pub prompt: String,
    /// The tool, or "" for the first installed one of the kind.
    pub provider: String,
    /// Length, seconds: a clip, a sound, or a placed picture.
    pub seconds: f64,
    /// `sfx` or `music`.
    pub sound_kind: String,
    /// The frame under the playhead is a reference (a clip: its first picture).
    pub use_frame: bool,
    /// The media selected in the Project panel are references (a clip: the first is its start).
    pub use_selection: bool,
    /// A picture with a transparent background (a cut-out).
    pub transparent: bool,
    /// Place the result at the playhead.
    pub place: bool,
    /// Fill gap: the video track.
    pub track: String,
    /// Fill gap: every gap of the track.
    pub all_gaps: bool,
    /// Augment: animate the changed frame.
    pub animate: bool,
    /// The variation; `None`: each generation takes the next one.
    pub seed: Option<u64>,
    /// The running generation.
    pub job: Option<u64>,
    #[serde(skip)]
    cache: Cache,
}

/// What the panel read from the session (refreshed when stale).
#[derive(Clone, Debug, Default, PartialEq)]
struct Cache {
    tools: Option<Value>,
    history: Vec<(u64, String, String)>,
    read: Option<web_time::Instant>,
}

impl Default for GenerateDraft {
    fn default() -> Self {
        Self {
            mode: "image".into(),
            prompt: String::new(),
            provider: String::new(),
            seconds: 4.0,
            sound_kind: "sfx".into(),
            use_frame: false,
            use_selection: false,
            transparent: false,
            place: true,
            track: "V1".into(),
            all_gaps: false,
            animate: false,
            seed: None,
            job: None,
            cache: Cache::default(),
        }
    }
}

fn finite(x: f64, or: f64) -> f64 {
    if x.is_finite() { x } else { or }
}

impl GenerateDraft {
    fn mode(&self) -> &str {
        if MODES.iter().any(|(id, _)| *id == self.mode) { &self.mode } else { "image" }
    }

    /// The kind of tool the mode needs.
    fn kind(&self) -> &'static str {
        match self.mode() {
            "video" | "fill" => "video",
            "sound" if self.sound_kind == "music" => "music",
            "sound" => "sfx",
            _ => "image",
        }
    }

    /// The command and parameters of the Generate button: `selection` is the Project panel's
    /// selected media, `playhead` the time results are placed at.
    pub fn request(&self, selection: &[u64], playhead: i64) -> (&'static str, Value) {
        let prompt = self.prompt.trim();
        let mut p = json!({});
        if !self.provider.is_empty() {
            p["provider"] = json!(self.provider);
        }
        if let Some(seed) = self.seed {
            p["seed"] = json!(seed);
        }
        let secs = |lo: f64, hi: f64, or: f64| finite(self.seconds, or).clamp(lo, hi);
        match self.mode() {
            "video" => {
                p["prompt"] = json!(prompt);
                p["seconds"] = json!(secs(1.0, 60.0, 4.0));
                p["place"] = json!(self.place);
                p["atTime"] = json!(playhead);
                if self.use_frame {
                    p["image"] = json!("playhead");
                } else if let Some(first) = selection.first().filter(|_| self.use_selection) {
                    p["image"] = json!(first);
                }
                ("generate.video", p)
            }
            "sound" => {
                p["prompt"] = json!(prompt);
                p["kind"] = json!(if self.sound_kind == "music" { "music" } else { "sfx" });
                p["seconds"] = json!(secs(0.5, 600.0, 5.0));
                p["place"] = json!(self.place);
                p["atTime"] = json!(playhead);
                ("generate.sound", p)
            }
            "fill" => {
                p["track"] = json!(if self.track.trim().is_empty() { "V1" } else { self.track.trim() });
                p["all"] = json!(self.all_gaps);
                p["atTime"] = json!(playhead);
                if !prompt.is_empty() {
                    p["prompt"] = json!(prompt);
                }
                ("generate.fillGap", p)
            }
            "augment" => {
                p["instruction"] = json!(prompt);
                p["animate"] = json!(self.animate);
                p["atTime"] = json!(playhead);
                p["place"] = json!(self.place);
                if self.use_selection && !selection.is_empty() {
                    p["references"] = json!(selection);
                }
                ("generate.augment", p)
            }
            _ => {
                p["prompt"] = json!(prompt);
                p["transparent"] = json!(self.transparent);
                p["place"] = json!(self.place);
                p["atTime"] = json!(playhead);
                p["seconds"] = json!(secs(0.2, 600.0, 5.0));
                p["fromPlayhead"] = json!(self.use_frame);
                if self.use_selection && !selection.is_empty() {
                    p["references"] = json!(selection);
                }
                ("generate.image", p)
            }
        }
    }
}

/// The part of `r` the scroll area shows (what a click can reach).
fn visible(ui: &egui::Ui, r: Rect) -> Rect {
    r.intersect(ui.clip_rect())
}

fn job_progress(app: &FilmcraftApp, id: u64) -> Option<(f32, String, bool)> {
    use std::sync::atomic::Ordering;
    app.session.jobs.iter().find(|j| j.id == id).map(|j| {
        (
            j.progress.fraction(),
            j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
            j.progress.finished.load(Ordering::Relaxed),
        )
    })
}

/// Follow the running generation and keep the tool list and history fresh.
fn refresh(app: &mut FilmcraftApp) {
    if let Some(job) = app.ui.generate.job {
        let done = job_progress(app, job).is_none_or(|(_, _, finished)| finished);
        if done && filmcraft_engine::generate::running(&app.session) != Some(job) {
            let outcome = filmcraft_engine::generate::outcome_json(&app.session, job);
            app.ui.status = match outcome.as_ref().and_then(|o| o.get("error")).and_then(Value::as_str) {
                Some(e) => format!("Generation: {e}"),
                None => {
                    let n = outcome.as_ref().and_then(|o| o["generated"].as_array()).map_or(0, Vec::len);
                    if n == 1 { "Generated 1 item".into() } else { format!("Generated {n} items") }
                }
            };
            app.ui.generate.job = None;
            app.ui.generate.cache.read = None;
        }
    }
    let stale = app.ui.generate.cache.read.is_none_or(|t| t.elapsed().as_secs_f64() > 3.0);
    if !stale {
        return;
    }
    let tools = filmcraft_engine::generate::providers_json(&app.session);
    let mut history: Vec<(u64, String, String)> = app
        .session
        .project
        .items
        .values()
        .filter(|it| it.metadata.contains_key("Generator"))
        .map(|it| (it.id.0, it.name.clone(), it.metadata.get("Description").cloned().unwrap_or_default()))
        .collect();
    history.sort_by_key(|h| std::cmp::Reverse(h.0));
    history.truncate(HISTORY_SHOWN);
    let c = &mut app.ui.generate.cache;
    c.tools = Some(tools);
    c.history = history;
    c.read = Some(web_time::Instant::now());
}

/// A seed different from `prev`, from the clock.
fn next_seed(prev: Option<u64>) -> u64 {
    let now = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut z = now ^ prev.unwrap_or(0).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) % 1_000_000
}

enum Action {
    Run,
    Stop,
    Place(u64),
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    refresh(app);
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let has_seq = app.session.active_sequence().is_some();
    let busy = app.ui.generate.job.and_then(|j| job_progress(app, j)).filter(|(_, _, finished)| !finished);
    let selected = app.session.state.project_selection.len();
    let mut actions: Vec<Action> = Vec::new();
    let auto = &mut app.auto;
    let d = &mut app.ui.generate;
    let tools = d.cache.tools.clone().unwrap_or(Value::Null);
    let kind = d.kind();
    let of_kind: Vec<&Value> = tools["providers"].as_array().map(|a| a.iter().filter(|p| p["kind"] == kind).collect()).unwrap_or_default();
    let chosen =
        of_kind.iter().find(|p| p["id"] == d.provider.as_str()).or_else(|| of_kind.iter().find(|p| p["installed"] == true)).or(of_kind.first()).copied();
    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - FOOTER_H));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("generate-body"));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut child, |ui| {
        ui.set_max_width(ui.available_width() - 6.0);
        let w = ui.available_width();
        let label = |ui: &mut egui::Ui, text: &str| {
            ui.label(egui::RichText::new(text).color(t.text).size(12.5).strong());
            ui.add_space(2.0);
        };
        let dim = |ui: &mut egui::Ui, text: &str| {
            ui.label(egui::RichText::new(text).color(t.text_dim).size(11.0));
        };
        ui.horizontal_wrapped(|ui| {
            for (id, name) in MODES {
                let b = ui.selectable_label(d.mode() == id, name);
                auto.add(&format!("generate.mode.{id}"), visible(ui, b.rect), name);
                if b.clicked() {
                    d.mode = id.into();
                    d.provider.clear();
                }
            }
        });
        dim(
            ui,
            match d.mode() {
                "video" => "A short clip from a description, optionally starting from a picture.",
                "sound" => "A sound effect or music from a description.",
                "fill" => "Fills the gap under the playhead with a clip that carries on from the frame before it.",
                "augment" => "Changes what the frame under the playhead shows, and lays the result over the clip.",
                _ => "A picture from a description, optionally after reference pictures.",
            },
        );
        ui.add_space(8.0);
        label(ui, "Tool");
        match chosen {
            None => {
                let where_ = tools["manifest"].as_str().unwrap_or("the data folder");
                let why = tools["error"].as_str().map_or_else(|| format!("No {kind} generation tool is set up on this computer."), str::to_string);
                ui.label(egui::RichText::new(why).color(t.text).size(11.5));
                dim(ui, &format!("Describe yours in {where_} (docs/generation.md shows how)."));
            }
            Some(tool) => {
                let name = tool["name"].as_str().unwrap_or("");
                let cb = egui::ComboBox::from_id_salt("generate-provider").selected_text(name).width(w).show_ui(ui, |ui| {
                    for p in &of_kind {
                        let id = p["id"].as_str().unwrap_or("");
                        let mut text = p["name"].as_str().unwrap_or(id).to_string();
                        if p["installed"] != true {
                            text.push_str(" (not installed)");
                        }
                        if ui.selectable_label(tool["id"] == p["id"], text).clicked() {
                            d.provider = id.into();
                        }
                    }
                });
                auto.add("generate.provider", visible(ui, cb.response.rect), "Tool");
                // what the model's licence asks to be shown where it is used
                let credit = tool["attribution"].as_str().filter(|a| !a.is_empty());
                if let Some(credit) = credit {
                    ui.label(egui::RichText::new(format!("Made with {credit}")).color(t.text).size(11.0));
                }
                if let Some(licence) = tool["license"].as_str().filter(|l| !l.is_empty()) {
                    dim(ui, licence);
                }
                if tool["installed"] != true {
                    ui.label(egui::RichText::new("This tool's program is not on this computer.").color(t.text).size(11.0));
                }
            }
        }
        ui.add_space(10.0);
        label(
            ui,
            match d.mode() {
                "augment" => "What to change",
                "fill" => "What happens (optional)",
                "sound" => "Describe the sound",
                "video" => "What happens in the clip",
                _ => "Describe the picture",
            },
        );
        let hint = match d.mode() {
            "augment" => "Add a small red sailboat on the sea",
            "fill" => "Leave empty to carry on from the frame before the gap",
            "sound" => "Waves breaking on rocks, distant gulls",
            "video" => "The camera pushes in slowly while the waves break",
            _ => "A white lighthouse on a rocky coast at golden hour",
        };
        let edit = ui.add(egui::TextEdit::multiline(&mut d.prompt).desired_width(w).desired_rows(4).hint_text(hint));
        auto.add("generate.prompt", visible(ui, edit.rect), "Description");
        ui.add_space(8.0);
        let check = |ui: &mut egui::Ui, auto: &mut crate::automation::Registry, id: &str, value: &mut bool, text: &str| {
            let c = ui.checkbox(value, text);
            auto.add(id, visible(ui, c.rect), text);
        };
        match d.mode() {
            "image" => {
                check(ui, auto, "generate.useFrame", &mut d.use_frame, "Use the frame under the playhead as a reference");
                check(
                    ui,
                    auto,
                    "generate.useSelection",
                    &mut d.use_selection,
                    &format!("Use the media selected in the Project panel as references ({selected})"),
                );
                check(ui, auto, "generate.transparent", &mut d.transparent, "Transparent background (a cut-out)");
                check(ui, auto, "generate.place", &mut d.place, "Place at the playhead");
            }
            "video" => {
                check(ui, auto, "generate.useFrame", &mut d.use_frame, "Start from the frame under the playhead");
                check(ui, auto, "generate.useSelection", &mut d.use_selection, &format!("Start from the picture selected in the Project panel ({selected})"));
                check(ui, auto, "generate.place", &mut d.place, "Place at the playhead");
            }
            "sound" => {
                let music = d.sound_kind == "music";
                let cb = egui::ComboBox::from_id_salt("generate-sound-kind").selected_text(if music { "Music" } else { "Sound effect" }).width(w).show_ui(
                    ui,
                    |ui| {
                        if ui.selectable_label(!music, "Sound effect").clicked() {
                            d.sound_kind = "sfx".into();
                            d.provider.clear();
                        }
                        if ui.selectable_label(music, "Music").clicked() {
                            d.sound_kind = "music".into();
                            d.provider.clear();
                        }
                    },
                );
                auto.add("generate.soundKind", visible(ui, cb.response.rect), "Kind of sound");
                check(ui, auto, "generate.place", &mut d.place, "Place at the playhead");
            }
            "fill" => {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Track").color(t.text_dim).size(11.5));
                    let e = ui.add(egui::TextEdit::singleline(&mut d.track).desired_width(48.0));
                    auto.add("generate.track", visible(ui, e.rect), "Track");
                });
                check(ui, auto, "generate.allGaps", &mut d.all_gaps, "Every gap of the track");
            }
            _ => {
                check(
                    ui,
                    auto,
                    "generate.useSelection",
                    &mut d.use_selection,
                    &format!("Also use the media selected in the Project panel as references ({selected})"),
                );
                check(ui, auto, "generate.animate", &mut d.animate, "Animate the changed frame");
                check(ui, auto, "generate.place", &mut d.place, "Lay it over the clip");
            }
        }
        if matches!(d.mode(), "image" | "video" | "sound") {
            ui.add_space(8.0);
            label(ui, if d.mode() == "image" { "Length on the timeline" } else { "Length" });
            let most = chosen.and_then(|tool| tool["maxSeconds"].as_f64()).filter(|m| m.is_finite() && *m > 1.0).unwrap_or(if d.mode() == "image" {
                30.0
            } else {
                10.0
            });
            ui.spacing_mut().slider_width = w - 70.0;
            let sl = ui.add(egui::Slider::new(&mut d.seconds, 0.5..=most).max_decimals(1).suffix(" s"));
            auto.add("generate.length", visible(ui, sl.rect), "Length");
        }
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            let b = ui.button("New variation");
            auto.add("generate.variation", visible(ui, b.rect), "New variation");
            if b.clicked() {
                d.seed = Some(next_seed(d.seed));
            }
            let note = match d.seed {
                Some(s) => format!("variation {s}"),
                None => "each one a new variation".into(),
            };
            ui.label(egui::RichText::new(note).color(t.text_faint).size(11.0));
        });
        if let Some((fraction, status, _)) = &busy {
            ui.add_space(10.0);
            ui.add(egui::ProgressBar::new(*fraction).text(status.clone()));
            let b = ui.button("Stop");
            auto.add("generate.stop", visible(ui, b.rect), "Stop");
            if b.clicked() {
                actions.push(Action::Stop);
            }
        }
        if !d.cache.history.is_empty() {
            ui.add_space(14.0);
            label(ui, "Generated in this project");
            for (n, (item, name, description)) in d.cache.history.iter().enumerate() {
                ui.horizontal(|ui| {
                    let b = ui.small_button("Place");
                    auto.add(&format!("generate.history.{}.place", n + 1), visible(ui, b.rect), "Place at the playhead");
                    if b.on_hover_text("Place at the playhead").clicked() {
                        actions.push(Action::Place(*item));
                    }
                    let text: String = if description.is_empty() { name.clone() } else { format!("{name}: {description}") };
                    ui.add(egui::Label::new(egui::RichText::new(text).color(t.text_dim).size(11.0)).truncate());
                });
            }
        }
    });
    let ready = chosen.is_some_and(|tool| tool["installed"] == true) && busy.is_none() && (has_seq || matches!(d.mode(), "image" | "video" | "sound"));
    let needs_text = d.mode() != "fill" && d.prompt.trim().is_empty();
    let fy = rect.max.y - FOOTER_H + 10.0;
    let r = Rect::from_min_size(pos2(rect.min.x + PAD, fy), egui::vec2(rect.width() - PAD * 2.0, 28.0));
    let text = match (busy.is_some(), d.mode()) {
        (true, _) => "Generating…",
        (_, "fill") => "Fill the gap",
        (_, "augment") => "Augment the scene",
        _ => "Generate",
    };
    let can = ready && !needs_text;
    let b = ui.put(r, egui::Button::new(egui::RichText::new(text).strong()).fill(if can { t.accent } else { t.field_bg }));
    app.auto.add("generate.run", r, "Generate");
    if b.clicked() && can {
        actions.push(Action::Run);
    }
    if !can {
        b.on_hover_text(if busy.is_some() {
            "One generation at a time"
        } else if needs_text {
            "Describe what to make first"
        } else if chosen.is_none() {
            "Set up a generation tool first"
        } else {
            "Open a sequence first"
        });
    }
    for a in actions {
        run(app, a);
    }
}

fn run(app: &mut FilmcraftApp, a: Action) {
    match a {
        Action::Run => {
            let selection: Vec<u64> = app.session.state.project_selection.iter().map(|i| i.0).collect();
            let (cmd, p) = app.ui.generate.request(&selection, app.session.playhead().0);
            match app.session.execute(cmd, p) {
                Ok(r) => {
                    app.ui.generate.job = r["job"].as_u64();
                    app.ui.status = "Generating in the background…".into();
                }
                Err(e) => app.ui.status = e.to_string(),
            }
        }
        Action::Stop => {
            if let Some(job) = app.ui.generate.job
                && let Err(e) = app.session.execute("jobs.cancel", json!({"job": job}))
            {
                app.ui.status = e.to_string();
            }
        }
        Action::Place(item) => {
            let at = app.session.playhead().0;
            app.ui.status = match app.session.execute("timeline.place", json!({"item": item, "time": at})) {
                Ok(_) => "Placed at the playhead".into(),
                Err(e) => e.to_string(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_mode_asks_its_command_for_what_the_panel_shows() {
        let mut d = GenerateDraft { prompt: "  a lighthouse  ".into(), seed: Some(7), provider: "h3-image".into(), use_selection: true, ..Default::default() };
        let (cmd, p) = d.request(&[4, 9], 1000);
        assert_eq!(cmd, "generate.image");
        assert_eq!(p["prompt"], "a lighthouse");
        assert_eq!((p["seed"].as_u64(), p["provider"].as_str()), (Some(7), Some("h3-image")));
        assert_eq!(p["references"], json!([4, 9]));
        assert_eq!((p["place"].as_bool(), p["atTime"].as_i64()), (Some(true), Some(1000)));
        d.mode = "video".into();
        let (cmd, p) = d.request(&[4, 9], 0);
        assert_eq!((cmd, p["image"].as_u64()), ("generate.video", Some(4)));
        d.use_frame = true;
        assert_eq!(d.request(&[4], 0).1["image"], "playhead");
        d.mode = "sound".into();
        d.sound_kind = "music".into();
        let (cmd, p) = d.request(&[], 0);
        assert_eq!((cmd, p["kind"].as_str()), ("generate.sound", Some("music")));
        assert_eq!(d.kind(), "music");
        d.mode = "fill".into();
        d.track = "  ".into();
        d.prompt.clear();
        let (cmd, p) = d.request(&[], 5);
        assert_eq!((cmd, p["track"].as_str(), p.get("prompt")), ("generate.fillGap", Some("V1"), None));
        d.mode = "augment".into();
        d.prompt = "add snow".into();
        d.animate = true;
        let (cmd, p) = d.request(&[2], 5);
        assert_eq!((cmd, p["instruction"].as_str(), p["animate"].as_bool()), ("generate.augment", Some("add snow"), Some(true)));
        // hostile state read back from an agent: an unknown mode and a broken length still make a sane request
        d.mode = "../../x".into();
        d.seconds = f64::NAN;
        let (cmd, p) = d.request(&[], 0);
        assert_eq!((cmd, p["seconds"].as_f64()), ("generate.image", Some(5.0)));
        d.seconds = f64::INFINITY;
        assert_eq!(d.request(&[], 0).1["seconds"].as_f64(), Some(5.0));
        d.seconds = 1.0e12;
        assert_eq!(d.request(&[], 0).1["seconds"].as_f64(), Some(600.0));
    }

    #[test]
    fn the_draft_round_trips_as_agents_read_and_set_it() {
        let d = GenerateDraft { mode: "augment".into(), prompt: "add a kite".into(), animate: true, ..Default::default() };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["mode"], "augment");
        assert!(v.get("cache").is_none());
        let back: GenerateDraft = serde_json::from_value(v).unwrap();
        assert_eq!(back, d);
        let partial: GenerateDraft = serde_json::from_value(json!({"mode": "sound", "soundKind": "music"})).unwrap();
        assert_eq!((partial.mode.as_str(), partial.sound_kind.as_str(), partial.place), ("sound", "music", true));
    }
}

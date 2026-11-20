//! Generate panel: make pictures, clips and sounds with the generation tools installed on this
//! computer (`generate.providers`), fill a gap of the timeline, change what the frame under
//! the playhead shows, or make a clip again in another look. Each button runs the same `generate.*` command agents and the CLI use; the
//! work is a background job, and its result is imported (and placed) when it finishes.
//!
//! The tool in use and what its licence asks to be credited are shown under the tool menu.
//!
//! Automation ids: `generate.mode.<image|video|sound|fill|augment|restyle>`, `generate.provider`,
//! `generate.prompt`, `generate.useFrame`, `generate.useSelection`, `generate.transparent`,
//! `generate.place`, `generate.soundKind`, `generate.length`, `generate.track`,
//! `generate.allGaps`, `generate.fillFrom`, `generate.animate`, `generate.variations`,
//! `generate.depthMove`, `generate.forClip`, `generate.carryOn`, `generate.restyleHow`,
//! `generate.restyleLook`, `generate.restyleStrength`, `generate.variation`,
//! `generate.run`, `generate.stop`, `generate.history.<n>.place`, `generate.history.<n>.use`.

use egui::{Rect, pos2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::FilmcraftApp;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 48.0;
/// The modes, with their names.
const MODES: [(&str, &str); 6] =
    [("image", "Picture"), ("video", "Clip"), ("sound", "Sound"), ("fill", "Fill gap"), ("augment", "Augment"), ("restyle", "Restyle clip")];
/// Generated items listed under History.
const HISTORY_SHOWN: usize = 8;
/// What a gap is filled from, with the names shown.
const FILL_FROM: [(&str, &str); 5] = [
    ("before", "The frame before the gap"),
    ("after", "The frame after the gap"),
    ("both", "Both: from the frame before to the frame after"),
    ("motion", "The movement before the gap (carries it on)"),
    ("prompt", "The description only"),
];
/// How a placed picture moves in depth (`clip.photo3d`), with the names shown.
const DEPTH_MOVES: [(&str, &str); 9] = [
    ("", "Still"),
    ("auto", "Move in depth: automatic"),
    ("in", "Push in"),
    ("out", "Pull out"),
    ("left", "Drift left"),
    ("right", "Drift right"),
    ("up", "Drift up"),
    ("down", "Drift down"),
    ("circle", "Circle"),
];

/// What is being generated (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateDraft {
    /// `image`, `video`, `sound`, `fill` (a gap of the timeline), `augment` (the frame under
    /// the playhead, changed) or `restyle` (the selected clip, made again).
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
    /// How many to make at once (1 to 4): the first is placed, the others wait in the bin.
    pub variations: u32,
    /// Picture: how it moves in depth once placed ("" = still).
    pub depth_move: String,
    /// Sound: as long as the clip selected in the timeline, starting where it starts.
    pub for_clip: bool,
    /// Fill gap: what the clip is made from (`before`, `after`, `both`, `motion`, `prompt`).
    pub fill_from: String,
    /// Clip: carry on the movement of the picture before the playhead.
    pub carry_on: bool,
    /// Restyle: the way of the tool ("" = the first that is installed).
    pub restyle_how: String,
    /// Restyle: how the picture tool changes the first frame before the clip is made ("" = not).
    pub restyle_look: String,
    /// Restyle: how closely the clip is followed, 0 to 1.
    pub restyle_strength: f64,
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
    /// Item, name, description, and whether it is one of several variations.
    history: Vec<(u64, String, String, bool)>,
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
            variations: 1,
            depth_move: String::new(),
            for_clip: false,
            fill_from: "before".into(),
            carry_on: false,
            restyle_how: String::new(),
            restyle_look: String::new(),
            restyle_strength: 1.0,
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
            "video" | "fill" | "restyle" => "video",
            "sound" if self.sound_kind == "music" => "music",
            "sound" => "sfx",
            _ => "image",
        }
    }

    fn fill_from(&self) -> &str {
        if FILL_FROM.iter().any(|(id, _)| *id == self.fill_from) { &self.fill_from } else { "before" }
    }

    fn depth_move(&self) -> &str {
        if DEPTH_MOVES.iter().any(|(id, _)| *id == self.depth_move) { &self.depth_move } else { "" }
    }

    /// The command and parameters of the Generate button: `selection` is the Project panel's
    /// selected media, `playhead` the time results are placed at, `clip` the clip selected in
    /// the timeline.
    pub fn request(&self, selection: &[u64], playhead: i64, clip: Option<u64>) -> (&'static str, Value) {
        let prompt = self.prompt.trim();
        let several = self.variations.clamp(1, 4);
        let mut p = json!({});
        if !self.provider.is_empty() {
            p["provider"] = json!(self.provider);
        }
        if let Some(seed) = self.seed {
            p["seed"] = json!(seed);
        }
        let secs = |lo: f64, hi: f64, or: f64| finite(self.seconds, or).clamp(lo, hi);
        match self.mode() {
            "video" if self.carry_on => {
                if !prompt.is_empty() {
                    p["prompt"] = json!(prompt);
                }
                p["seconds"] = json!(secs(0.3, 10.0, 2.0));
                p["atTime"] = json!(playhead);
                p["place"] = json!(self.place);
                ("generate.extend", p)
            }
            "video" => {
                p["prompt"] = json!(prompt);
                if several > 1 {
                    p["variations"] = json!(several);
                }
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
                if several > 1 {
                    p["variations"] = json!(several);
                }
                p["place"] = json!(self.place);
                match clip.filter(|_| self.for_clip) {
                    Some(c) => p["clip"] = json!(c),
                    None => {
                        p["seconds"] = json!(secs(0.5, 600.0, 5.0));
                        p["atTime"] = json!(playhead);
                    }
                }
                ("generate.sound", p)
            }
            "fill" => {
                p["track"] = json!(if self.track.trim().is_empty() { "V1" } else { self.track.trim() });
                p["all"] = json!(self.all_gaps);
                p["atTime"] = json!(playhead);
                // "the description only" without one falls back to the frame before
                p["from"] = json!(if self.fill_from() == "prompt" && prompt.is_empty() { "before" } else { self.fill_from() });
                if !prompt.is_empty() {
                    p["prompt"] = json!(prompt);
                }
                ("generate.fillGap", p)
            }
            "restyle" => {
                p["prompt"] = json!(prompt);
                if let Some(c) = clip {
                    p["clip"] = json!(c);
                }
                let how = self.restyle_how.trim();
                if !how.is_empty() {
                    p["how"] = json!(how);
                }
                let look = self.restyle_look.trim();
                if !look.is_empty() {
                    p["look"] = json!(look);
                    if self.use_selection && !selection.is_empty() {
                        p["references"] = json!(selection);
                    }
                }
                p["strength"] = json!(finite(self.restyle_strength, 1.0).clamp(0.0, 1.0));
                p["place"] = json!(self.place);
                ("generate.restyle", p)
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
                if several > 1 {
                    p["variations"] = json!(several);
                }
                if self.place && !self.depth_move().is_empty() {
                    p["move"] = json!(self.depth_move());
                }
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
    let mut history: Vec<(u64, String, String, bool)> = app
        .session
        .project
        .items
        .values()
        .filter(|it| it.metadata.contains_key("Generator"))
        .map(|it| (it.id.0, it.name.clone(), it.metadata.get("Description").cloned().unwrap_or_default(), it.metadata.contains_key("Variation Of")))
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
    Use(u64),
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    refresh(app);
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let has_seq = app.session.active_sequence().is_some();
    let busy = app.ui.generate.job.and_then(|j| job_progress(app, j)).filter(|(_, _, finished)| !finished);
    let selected = app.session.state.project_selection.len();
    let clip = app.session.state.selection.first().map(|c| c.0);
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
                "video" if d.carry_on => "A clip that carries on the movement of the picture just before the playhead.",
                "video" => "A short clip from a description, optionally starting from a picture.",
                "sound" => "A sound effect or music from a description.",
                "fill" => "Fills the gap under the playhead with a clip that carries on from the frame before it.",
                "augment" => "Changes what the frame under the playhead shows, and lays the result over the clip.",
                "restyle" => "Makes the clip selected in the timeline (else the one under the playhead) again: its shapes and movement stay, its look follows the description.",
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
                "restyle" => "What the new clip shows",
                "fill" => "What happens (optional)",
                "sound" => "Describe the sound",
                "video" => "What happens in the clip",
                _ => "Describe the picture",
            },
        );
        let hint = match d.mode() {
            "augment" => "Add a small red sailboat on the sea",
            "restyle" => "A watercolour painting of a lighthouse on a rocky coast",
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
                let shown = DEPTH_MOVES.iter().find(|(id, _)| *id == d.depth_move()).map_or("Still", |(_, name)| *name);
                let cb = egui::ComboBox::from_id_salt("generate-depth-move").selected_text(shown).width(w).show_ui(ui, |ui| {
                    for (id, name) in DEPTH_MOVES {
                        if ui.selectable_label(d.depth_move() == id, name).clicked() {
                            d.depth_move = id.into();
                        }
                    }
                });
                auto.add("generate.depthMove", visible(ui, cb.response.rect), "Movement of the placed picture");
            }
            "video" => {
                check(ui, auto, "generate.useFrame", &mut d.use_frame, "Start from the frame under the playhead");
                check(ui, auto, "generate.useSelection", &mut d.use_selection, &format!("Start from the picture selected in the Project panel ({selected})"));
                let can = chosen.is_some_and(|tool| tool["carriesOn"] == true);
                let text = if can { "Carry on the movement before the playhead" } else { "Carry on the movement before the playhead (this tool cannot)" };
                check(ui, auto, "generate.carryOn", &mut d.carry_on, text);
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
                let text =
                    if clip.is_some() { "As long as the clip selected in the timeline" } else { "As long as the clip selected in the timeline (none is)" };
                check(ui, auto, "generate.forClip", &mut d.for_clip, text);
                check(ui, auto, "generate.place", &mut d.place, "Place at the playhead");
            }
            "fill" => {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Track").color(t.text_dim).size(11.5));
                    let e = ui.add(egui::TextEdit::singleline(&mut d.track).desired_width(48.0));
                    auto.add("generate.track", visible(ui, e.rect), "Track");
                });
                check(ui, auto, "generate.allGaps", &mut d.all_gaps, "Every gap of the track");
                let can = |id: &str| match id {
                    "both" => chosen.is_some_and(|tool| tool["bothEnds"] == true),
                    "motion" => chosen.is_some_and(|tool| tool["carriesOn"] == true),
                    _ => true,
                };
                let shown = FILL_FROM.iter().find(|(id, _)| *id == d.fill_from()).map_or("", |(_, name)| *name);
                let cb = egui::ComboBox::from_id_salt("generate-fill-from").selected_text(shown).width(w).show_ui(ui, |ui| {
                    for (id, name) in FILL_FROM {
                        let text = if can(id) { name.to_string() } else { format!("{name} (this tool cannot)") };
                        if ui.selectable_label(d.fill_from() == id, text).clicked() {
                            d.fill_from = id.into();
                        }
                    }
                });
                auto.add("generate.fillFrom", visible(ui, cb.response.rect), "What the clip is made from");
            }
            "restyle" => {
                let ways: Vec<(String, String, bool)> = chosen
                    .and_then(|tool| tool["restyles"].as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| Some((x["name"].as_str()?.to_string(), x["about"].as_str().unwrap_or("").to_string(), x["installed"] == true)))
                            .collect()
                    })
                    .unwrap_or_default();
                if ways.is_empty() {
                    ui.label(egui::RichText::new("This tool cannot make a clip again (its entry has no `restyles`).").color(t.text).size(11.0));
                } else {
                    let current = ways.iter().find(|x| x.0 == d.restyle_how).or_else(|| ways.iter().find(|x| x.2)).or(ways.first());
                    let (name, about) = current.map_or((String::new(), String::new()), |x| (x.0.clone(), x.1.clone()));
                    let cb = egui::ComboBox::from_id_salt("generate-restyle-how").selected_text(name.as_str()).width(w).show_ui(ui, |ui| {
                        for (id, _, there) in &ways {
                            let text = if *there { id.clone() } else { format!("{id} (its adapter is not on this computer)") };
                            if ui.selectable_label(name == *id, text).clicked() {
                                d.restyle_how = id.clone();
                            }
                        }
                    });
                    auto.add("generate.restyleHow", visible(ui, cb.response.rect), "How the clip is made again");
                    if !about.is_empty() {
                        dim(ui, &about);
                    }
                }
                ui.add_space(6.0);
                label(ui, "Change the first frame first (optional)");
                let e = ui.add(
                    egui::TextEdit::multiline(&mut d.restyle_look)
                        .desired_width(w)
                        .desired_rows(2)
                        .hint_text("Replace the man with the robot from the second picture, in the same pose"),
                );
                auto.add("generate.restyleLook", visible(ui, e.rect), "Change the first frame first");
                check(
                    ui,
                    auto,
                    "generate.useSelection",
                    &mut d.use_selection,
                    &format!("Use the media selected in the Project panel in that change ({selected})"),
                );
                ui.add_space(6.0);
                label(ui, "How closely the clip is followed");
                d.restyle_strength = finite(d.restyle_strength, 1.0).clamp(0.0, 1.0);
                ui.spacing_mut().slider_width = w - 70.0;
                let sl = ui.add(egui::Slider::new(&mut d.restyle_strength, 0.0..=1.0).max_decimals(2));
                auto.add("generate.restyleStrength", visible(ui, sl.rect), "How closely the clip is followed");
                check(ui, auto, "generate.place", &mut d.place, "Lay it over the clip");
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
            if !(d.mode() == "video" && d.carry_on) {
                ui.add_space(6.0);
                label(ui, "How many to choose from");
                d.variations = d.variations.clamp(1, 4);
                let sl = ui.add(egui::Slider::new(&mut d.variations, 1..=4));
                auto.add("generate.variations", visible(ui, sl.rect), "How many to choose from");
            }
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
            for (n, (item, name, description, varied)) in d.cache.history.iter().enumerate() {
                ui.horizontal(|ui| {
                    let b = ui.small_button("Place");
                    auto.add(&format!("generate.history.{}.place", n + 1), visible(ui, b.rect), "Place at the playhead");
                    if b.on_hover_text("Place at the playhead").clicked() {
                        actions.push(Action::Place(*item));
                    }
                    if *varied {
                        let b = ui.small_button("Use");
                        auto.add(&format!("generate.history.{}.use", n + 1), visible(ui, b.rect), "Use this variation");
                        if b.on_hover_text("Put this one in the place of the variation on the timeline").clicked() {
                            actions.push(Action::Use(*item));
                        }
                    }
                    let text: String = if description.is_empty() { name.clone() } else { format!("{name}: {description}") };
                    ui.add(egui::Label::new(egui::RichText::new(text).color(t.text_dim).size(11.0)).truncate());
                });
            }
        }
    });
    let carrying = d.mode() == "video" && d.carry_on;
    let ready =
        chosen.is_some_and(|tool| tool["installed"] == true) && busy.is_none() && (has_seq || (matches!(d.mode(), "image" | "video" | "sound") && !carrying));
    let needs_text = d.mode() != "fill" && !carrying && d.prompt.trim().is_empty();
    let fy = rect.max.y - FOOTER_H + 10.0;
    let r = Rect::from_min_size(pos2(rect.min.x + PAD, fy), egui::vec2(rect.width() - PAD * 2.0, 28.0));
    let text = match (busy.is_some(), d.mode()) {
        (true, _) => "Generating…",
        (_, "fill") => "Fill the gap",
        (_, "video") if carrying => "Carry the shot on",
        (_, "augment") => "Augment the scene",
        (_, "restyle") => "Make the clip again",
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
            let clip = app.session.state.selection.first().map(|c| c.0);
            let (cmd, p) = app.ui.generate.request(&selection, app.session.playhead().0, clip);
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
        Action::Use(item) => {
            app.ui.status = match app.session.execute("generate.pick", json!({"item": item})) {
                Ok(_) => "This variation is on the timeline now".into(),
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
        let (cmd, p) = d.request(&[4, 9], 1000, None);
        assert_eq!(cmd, "generate.image");
        assert_eq!(p["prompt"], "a lighthouse");
        assert_eq!((p["seed"].as_u64(), p["provider"].as_str()), (Some(7), Some("h3-image")));
        assert_eq!(p["references"], json!([4, 9]));
        assert_eq!((p["place"].as_bool(), p["atTime"].as_i64()), (Some(true), Some(1000)));
        d.mode = "video".into();
        let (cmd, p) = d.request(&[4, 9], 0, None);
        assert_eq!((cmd, p["image"].as_u64()), ("generate.video", Some(4)));
        d.use_frame = true;
        assert_eq!(d.request(&[4], 0, None).1["image"], "playhead");
        d.mode = "sound".into();
        d.sound_kind = "music".into();
        let (cmd, p) = d.request(&[], 0, None);
        assert_eq!((cmd, p["kind"].as_str()), ("generate.sound", Some("music")));
        assert_eq!(d.kind(), "music");
        d.mode = "fill".into();
        d.track = "  ".into();
        d.prompt.clear();
        let (cmd, p) = d.request(&[], 5, None);
        assert_eq!((cmd, p["track"].as_str(), p.get("prompt")), ("generate.fillGap", Some("V1"), None));
        d.mode = "augment".into();
        d.prompt = "add snow".into();
        d.animate = true;
        let (cmd, p) = d.request(&[2], 5, None);
        assert_eq!((cmd, p["instruction"].as_str(), p["animate"].as_bool()), ("generate.augment", Some("add snow"), Some(true)));
        // hostile state read back from an agent: an unknown mode and a broken length still make a sane request
        d.mode = "../../x".into();
        d.seconds = f64::NAN;
        let (cmd, p) = d.request(&[], 0, None);
        assert_eq!((cmd, p["seconds"].as_f64()), ("generate.image", Some(5.0)));
        d.seconds = f64::INFINITY;
        assert_eq!(d.request(&[], 0, None).1["seconds"].as_f64(), Some(5.0));
        d.seconds = 1.0e12;
        assert_eq!(d.request(&[], 0, None).1["seconds"].as_f64(), Some(600.0));
    }

    #[test]
    fn variations_movement_and_what_a_gap_is_filled_from_reach_the_command() {
        let mut d = GenerateDraft { prompt: "a cliff".into(), variations: 3, depth_move: "in".into(), ..Default::default() };
        let (cmd, p) = d.request(&[], 0, Some(7));
        assert_eq!((cmd, p["variations"].as_u64(), p["move"].as_str()), ("generate.image", Some(3), Some("in")));
        // a picture that is not placed does not move; one variation is not said
        d.place = false;
        d.variations = 1;
        let p = d.request(&[], 0, None).1;
        assert!(p.get("move").is_none() && p.get("variations").is_none(), "{p}");
        // a sound for the selected clip takes its length from the clip
        d.mode = "sound".into();
        d.for_clip = true;
        let p = d.request(&[], 9, Some(7)).1;
        assert_eq!((p["clip"].as_u64(), p.get("seconds"), p.get("atTime")), (Some(7), None, None));
        let p = d.request(&[], 9, None).1;
        assert_eq!((p.get("clip"), p["atTime"].as_i64()), (None, Some(9)), "no clip selected: at the playhead");
        // a clip that carries the shot on needs no description
        d.mode = "video".into();
        d.carry_on = true;
        d.prompt.clear();
        d.seconds = 60.0;
        let (cmd, p) = d.request(&[], 5, None);
        assert_eq!((cmd, p["seconds"].as_f64(), p.get("prompt")), ("generate.extend", Some(10.0), None));
        d.mode = "fill".into();
        for (from, said) in [("both", "both"), ("motion", "motion"), ("after", "after"), ("prompt", "before"), ("../x", "before")] {
            d.fill_from = from.into();
            assert_eq!(d.request(&[], 0, None).1["from"], said, "{from}");
        }
        d.prompt = "clouds".into();
        d.fill_from = "prompt".into();
        assert_eq!(d.request(&[], 0, None).1["from"], "prompt");
        // a clip made again: the selected clip, the way, and the first frame changed with the selected media
        d.mode = "restyle".into();
        d.prompt = "as a watercolour".into();
        d.use_selection = true;
        let (cmd, p) = d.request(&[3], 0, Some(7));
        assert_eq!(
            (cmd, p["clip"].as_u64(), p["strength"].as_f64(), p.get("how"), p.get("look"), p.get("references")),
            ("generate.restyle", Some(7), Some(1.0), None, None, None)
        );
        d.restyle_how = "edges".into();
        d.restyle_look = " put the robot in ".into();
        d.restyle_strength = f64::NAN;
        let p = d.request(&[3], 0, None).1;
        assert_eq!(
            (p["how"].as_str(), p["look"].as_str(), p["references"][0].as_u64(), p["strength"].as_f64(), p.get("clip")),
            (Some("edges"), Some("put the robot in"), Some(3), Some(1.0), None)
        );
        d.restyle_strength = 7.0;
        assert_eq!(d.request(&[], 0, None).1["strength"].as_f64(), Some(1.0));
        assert_eq!(d.kind(), "video");
        // hostile state read back from an agent
        d.mode = "image".into();
        d.place = true;
        d.variations = u32::MAX;
        d.depth_move = "sideways".into();
        let p = d.request(&[], 0, None).1;
        assert_eq!((p["variations"].as_u64(), p.get("move")), (Some(4), None));
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

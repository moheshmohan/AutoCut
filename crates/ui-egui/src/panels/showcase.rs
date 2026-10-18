//! Showcase panel: review an AutoCut showcase ("every robot I've filmed, in order") before it is
//! built. "Analyze…" finds every clip that shows a subject (filmed since a date, or as a request
//! in words; of the whole project, or only the clips of a Google Photos album) and writes a plan; the entries are listed in date order, each with a thumbnail, its
//! label, description and the line the narrator says, a switch to leave it out, "Merge with
//! previous" (the same robot again) and its shots (each plays in the Source monitor). Title,
//! voice (cloned voices included), mood, intro and outro are the plan's too. "Write script" has
//! the script model write the words again for the entries as reviewed; "Build" and "Build and
//! export" make the sequence (and the export, optionally uploaded to Google Photos).
//!
//! Every action is an `autocut.*` engine command running as a background job (progress and
//! Cancel here, in the Progress panel and in `jobs.list`), so the panel, the CLI, the control
//! channel and MCP agents share one path; a job started elsewhere (an agent, the AutoCut prompt
//! "make a video of every robot since 2020") brings the panel forward and its plan loads here.
//!
//! Automation ids: `showcase.analyzeOpen`, `showcase.subject`, `showcase.since`, `showcase.prompt`,
//! `showcase.album`, `showcase.analyze`, `showcase.analyzeCancel`, `showcase.cancel`, `showcase.title`,
//! `showcase.voice`, `showcase.mood`, `showcase.intro`, `showcase.outro`,
//! `showcase.entry.<i>.include`, `.thumb`, `.label`, `.description`, `.line`, `.merge`, `.shots`,
//! `.shot.<k>`, `showcase.exportPath`, `showcase.upload`, `showcase.writeScript`,
//! `showcase.build`, `showcase.buildExport`; for shorts (`autocut.shorts` on the active
//! sequence: its best passages reframed to a vertical shape, captioned) `showcase.shortsOpen`,
//! `showcase.shortsCount`, `showcase.shortsAspect`, `showcase.shortsExport`, `showcase.shorts`.
//! The draft is UI state: `ui.set {showcase: {...}}`
//! patches it, with `entries` a list (replaced) or `{"<index>": {...}}` (those entries patched).

use std::sync::atomic::Ordering;

use egui::{Rect, Sense, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::FilmcraftApp;
use crate::dock::PanelKind;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 84.0;
/// Entries a draft holds at most (as an engine plan).
const MAX_ENTRIES: usize = filmcraft_engine::autocut::MAX_ENTRIES;
/// Longest label, description and spoken line the fields take.
const MAX_LABEL: usize = 120;
const MAX_TEXT: usize = 600;
/// Thumbnails asked for per frame (the frame cache decodes them in the background).
const THUMBS_PER_FRAME: usize = 64;
/// The shapes "Make shorts" offers.
const SHORT_ASPECTS: [&str; 4] = ["9:16", "1:1", "4:5", "16:9"];
/// Music moods (`audio.generate`).
const MOODS: [(&str, &str); 9] = [
    ("epic", "Epic"),
    ("tense", "Tense"),
    ("uplifting", "Uplifting"),
    ("calm", "Calm"),
    ("happy", "Happy"),
    ("dark", "Dark"),
    ("dreamy", "Dreamy"),
    ("energetic", "Energetic"),
    ("sad", "Sad"),
];

/// One entry under review.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DraftEntry {
    /// Shown in the video.
    pub include: bool,
    /// The same robot as the included entry before it: its shots join that one.
    pub merge_previous: bool,
    pub label: String,
    pub description: String,
    /// What the narrator says over it.
    pub line: String,
    /// Its shots are listed.
    pub open: bool,
    /// The plan's entry as the engine gave it (`number`, `kind`, `unix`, `shots`…); `label` and
    /// `description` above replace its own when it is sent back.
    pub entry: Value,
}

/// The showcase under review (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ShowcaseDraft {
    /// The "Analyze…" form is open.
    pub analyze_open: bool,
    /// What to look for ("robot").
    pub subject: String,
    /// Only clips filmed since: "2020", "2020-06" or "2020-06-15"; empty: every clip.
    pub since: String,
    /// Or the request in words ("make a video of every robot since 2020"), used without `subject`.
    pub prompt: String,
    /// Take the clips from this Google Photos album (its exact title): picked whole, imported,
    /// and only those analysed; empty: the project's clips.
    pub album: String,
    pub title: String,
    /// Narration voice id (`clone:<name>` for a cloned voice); empty: no narration.
    pub voice: String,
    /// Music mood: epic, tense, uplifting, calm, happy, dark, dreamy, energetic or sad.
    pub mood: String,
    pub intro: String,
    pub outro: String,
    pub entries: Vec<DraftEntry>,
    /// Where "Build and export" writes the video; empty: the AutoCut folder, named after the title.
    pub export_path: String,
    /// Upload the export to Google Photos.
    pub upload: bool,
    /// The rest of the plan (subtitle, size…), sent back as it came.
    pub extra: serde_json::Map<String, Value>,
    /// The kept plan's and report's revisions the draft follows (`autocut.showcaseInspect`).
    pub plan_revision: u64,
    pub report_revision: u64,
    /// The latest showcase job the panel has shown, and the one whose end it has reported.
    pub seen_job: Option<u64>,
    pub reported_job: Option<u64>,
    /// The "Shorts" form is open.
    pub shorts_open: bool,
    /// How many shorts "Make shorts" cuts from the active sequence (1..20).
    pub shorts_count: u32,
    /// Their shape: "9:16", "1:1", "4:5" or "16:9".
    pub shorts_aspect: String,
    /// A folder to export them into (captions burned in); empty: not exported.
    pub shorts_export: String,
}

impl Default for ShowcaseDraft {
    fn default() -> Self {
        Self {
            analyze_open: true,
            subject: "robot".into(),
            since: String::new(),
            prompt: String::new(),
            album: String::new(),
            title: String::new(),
            voice: String::new(),
            mood: "epic".into(),
            intro: String::new(),
            outro: String::new(),
            entries: Vec::new(),
            export_path: String::new(),
            upload: false,
            extra: serde_json::Map::new(),
            plan_revision: 0,
            report_revision: 0,
            seen_job: None,
            reported_job: None,
            shorts_open: false,
            shorts_count: 3,
            shorts_aspect: "9:16".into(),
            shorts_export: String::new(),
        }
    }
}

/// Keys of a plan the draft edits itself; the others go to `extra`.
const PLAN_KEYS: [&str; 6] = ["title", "voice", "mood", "entries", "script", "subject"];

impl ShowcaseDraft {
    /// Load a plan (as `autocut.showcasePlan` writes it).
    pub fn load(&mut self, plan: &Value) {
        let text = |v: &Value| v.as_str().unwrap_or_default().to_string();
        self.title = text(&plan["title"]);
        self.voice = text(&plan["voice"]);
        self.mood = plan["mood"].as_str().unwrap_or("epic").to_string();
        if let Some(s) = plan["subject"].as_str().filter(|s| !s.trim().is_empty()) {
            self.subject = s.to_string();
        }
        self.intro = text(&plan["script"]["intro"]);
        self.outro = text(&plan["script"]["outro"]);
        let lines = plan["script"]["lines"].as_array().cloned().unwrap_or_default();
        self.entries = plan["entries"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .take(MAX_ENTRIES)
            .enumerate()
            .map(|(i, e)| DraftEntry {
                include: e["include"].as_bool().unwrap_or(true),
                merge_previous: e["mergePrevious"].as_bool().unwrap_or(false),
                label: text(&e["label"]),
                description: text(&e["description"]),
                line: lines.get(i).map(text).unwrap_or_default(),
                open: false,
                entry: e.clone(),
            })
            .collect();
        self.extra = plan
            .as_object()
            .map(|o| o.iter().filter(|(k, _)| !PLAN_KEYS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
    }

    /// The plan as reviewed, for `autocut.showcaseBuild {plan}`.
    pub fn plan(&self) -> Value {
        let mut p = Value::Object(self.extra.clone());
        if !self.subject.trim().is_empty() {
            p["subject"] = json!(self.subject.trim());
        }
        p["title"] = json!(self.title.trim());
        if !self.voice.trim().is_empty() {
            p["voice"] = json!(self.voice.trim());
        }
        p["mood"] = json!(self.mood);
        p["entries"] = Value::Array(self.reviewed_entries());
        p["script"] = json!({"intro": self.intro, "outro": self.outro, "lines": self.entries.iter().map(|e| e.line.clone()).collect::<Vec<_>>()});
        p
    }

    /// The entries with the review's edits and choices (an entry that is not an object, from a
    /// hand-made state, is sent as it is and refused by the engine).
    fn reviewed_entries(&self) -> Vec<Value> {
        self.entries
            .iter()
            .map(|e| {
                let mut v = e.entry.clone();
                if let Some(o) = v.as_object_mut() {
                    o.insert("label".into(), json!(e.label));
                    o.insert("description".into(), json!(e.description));
                    o.insert("include".into(), json!(e.include));
                    o.insert("mergePrevious".into(), json!(e.merge_previous));
                }
                v
            })
            .collect()
    }

    /// Where "Build and export" writes: the path given, else the AutoCut folder under the title.
    pub fn export_to(&self) -> String {
        let p = self.export_path.trim();
        if p.is_empty() { default_export_path(&self.title) } else { p.to_string() }
    }

    /// `autocut.shorts` parameters from the Shorts form (the active sequence's shorts).
    fn shorts_params(&self) -> Value {
        let aspect = if SHORT_ASPECTS.contains(&self.shorts_aspect.as_str()) { self.shorts_aspect.as_str() } else { "9:16" };
        let mut p = json!({"count": self.shorts_count.clamp(1, 20), "aspect": aspect, "captions": true});
        if !self.shorts_export.trim().is_empty() {
            p["export"] = json!(self.shorts_export.trim());
        }
        p
    }

    /// `autocut.makeShowcase` parameters from the Analyze form.
    fn make_params(&self) -> Value {
        let mut p = json!({"voice": self.voice.trim(), "mood": self.mood});
        if !self.subject.trim().is_empty() {
            p["subject"] = json!(self.subject.trim());
        } else if !self.prompt.trim().is_empty() {
            p["prompt"] = json!(self.prompt.trim());
        }
        if !self.since.trim().is_empty() {
            p["since"] = json!(self.since.trim());
        }
        if !self.album.trim().is_empty() {
            p["album"] = json!(self.album.trim());
        }
        if !self.title.trim().is_empty() {
            p["title"] = json!(self.title.trim());
        }
        p
    }
}

/// Apply `ui.set {showcase: patch}`: fields merged into the draft; `entries` either a list
/// (replacing them) or `{"<index>": {fields}}` (patching those entries).
pub fn patch(app: &mut FilmcraftApp, patch: &Value) -> Result<(), String> {
    let mut patch = patch.clone();
    let entry_patches = match patch.get("entries") {
        Some(Value::Object(m)) => {
            let m = m.clone();
            if let Some(o) = patch.as_object_mut() {
                o.remove("entries");
            }
            Some(m)
        }
        _ => None,
    };
    let mut cur = serde_json::to_value(&app.ui.showcase).map_err(|e| e.to_string())?;
    crate::control::merge(&mut cur, &patch);
    if let Some(m) = entry_patches {
        let entries = cur["entries"].as_array_mut().ok_or("the draft has no entries")?;
        let n = entries.len();
        for (k, v) in m {
            let i: usize = k.parse().map_err(|_| format!("`entries` keys are entry indices, not `{k}`"))?;
            let e = entries.get_mut(i).ok_or_else(|| format!("no entry {i} (there are {n})"))?;
            crate::control::merge(e, &v);
        }
    }
    let mut d: ShowcaseDraft = serde_json::from_value(cur).map_err(|e| format!("bad `showcase`: {e}"))?;
    d.entries.truncate(MAX_ENTRIES);
    app.ui.showcase = d;
    Ok(())
}

/// The latest showcase job's state: `(id, command, running, fraction, status, error)`.
struct JobView {
    id: u64,
    running: bool,
    fraction: f32,
    status: String,
    error: Option<String>,
}

fn job_view(app: &FilmcraftApp) -> Option<JobView> {
    let (id, _) = app.session.autocut.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).job?;
    let j = app.session.jobs.iter().find(|j| j.id == id)?;
    let finished = j.progress.finished.load(Ordering::Relaxed);
    let status = j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    let error = j.progress.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    let total = j.progress.total.load(Ordering::Relaxed);
    let fraction = if total > 0 { (j.progress.done.load(Ordering::Relaxed) as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
    Some(JobView { id, running: !finished, fraction, status, error })
}

/// Every frame, also when the panel is hidden: wake the UI when a job sends the session a
/// command, load a new plan into the draft, bring the panel forward for a new showcase job and
/// report how it ended.
pub fn poll(app: &mut FilmcraftApp, ctx: &egui::Context) {
    if !app.session.autocut.port.has_waker() {
        let c = ctx.clone();
        app.session.set_port_waker(std::sync::Arc::new(move || c.request_repaint()));
    }
    let (plan_rev, report_rev, job, plan) = {
        let st = app.session.autocut.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let plan = (st.plan_revision != app.ui.showcase.plan_revision).then(|| st.plan.clone());
        (st.plan_revision, st.report_revision, st.job, plan)
    };
    let d = &mut app.ui.showcase;
    if report_rev != d.report_revision {
        d.report_revision = report_rev;
        // a new analysis without a new plan: its clips are grouped afresh by Write script
        if plan.is_none() {
            d.entries.clear();
        }
    }
    if let Some(plan) = plan {
        d.plan_revision = plan_rev;
        match plan {
            Some(p) => d.load(&p),
            None => d.entries.clear(),
        }
    }
    let Some((id, _)) = job else { return };
    if d.seen_job != Some(id) {
        d.seen_job = Some(id);
        d.analyze_open = false;
        app.show_panel(PanelKind::Showcase);
    }
    let Some(v) = job_view(app) else { return };
    if v.running {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    } else if app.ui.showcase.reported_job != Some(id) {
        app.ui.showcase.reported_job = Some(id);
        app.ui.status = match v.error {
            Some(e) => format!("Showcase: {e}"),
            None => format!("Showcase: {}", v.status),
        };
    }
}

/// "Make a Showcase…" from the menu: the panel with the Analyze form open.
pub fn open(app: &mut FilmcraftApp) -> Result<Value, String> {
    filmcraft_engine::find_command("autocut.makeShowcase").map_or(Ok(()), |c| (c.enabled)(&app.session))?;
    app.ui.showcase.analyze_open = true;
    app.show_panel(PanelKind::Showcase);
    Ok(json!({"panel": "Showcase"}))
}

/// The export path "Build and export" offers: the AutoCut folder, named after the title.
fn default_export_path(title: &str) -> String {
    let folder = if cfg!(target_os = "macos") { "~/Movies/AutoCut" } else { "~/Videos/AutoCut" };
    let safe: String =
        title.chars().map(|c| if c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c }).take(80).collect();
    let name = safe.trim().trim_matches('.').trim();
    format!("{folder}/{}.mp4", if name.is_empty() { "Showcase" } else { name })
}

/// `YYYY-MM-DD` of Unix seconds (UTC).
fn date(unix: i64) -> String {
    let z = unix.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{m:02}-{d:02}", yoe + era * 400 + i64::from(m <= 2))
}

/// A shot of an entry: its media item, name and best moment (seconds).
fn shots(e: &Value) -> Vec<(u64, String, f64)> {
    e["shots"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .take(64)
        .filter_map(|s| Some((s["item"].as_u64()?, s["name"].as_str().unwrap_or("clip").to_string(), s["best"].as_f64().filter(|b| b.is_finite())?.max(0.0))))
        .collect()
}

enum Action {
    Analyze,
    Shorts,
    WriteScript,
    Build { export: bool },
    Cancel(u64),
    Play { item: u64, seconds: f64 },
}

/// Voices for the narration menu: `(id, name)` — no narration, the cloned voices, then the
/// natural and built-in ones.
fn voices() -> Vec<(String, String)> {
    let mut out = vec![(String::new(), "No narration".to_string())];
    let cloned = filmcraft_engine::voice_clone::cloned_json();
    for v in cloned.as_array().map(Vec::as_slice).unwrap_or_default() {
        if let (Some(id), Some(name)) = (v["id"].as_str(), v["name"].as_str()) {
            out.push((id.to_string(), format!("{name} (cloned)")));
        }
    }
    let models = filmcraft_engine::transcript::models_dir();
    for v in filmcraft_tts::voices_in(models.as_deref()).into_iter().filter(|v| v.installed) {
        out.push((v.id.to_string(), format!("{} ({})", v.name, v.language)));
    }
    out
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let ctx = ui.ctx().clone();
    let job = job_view(app);
    let busy = job.as_ref().is_some_and(|j| j.running);
    let available = filmcraft_engine::autocut::can_run(&app.session);
    let has_report = app.session.autocut.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).report.is_some();
    // thumbnails first (the frame cache needs the app), the first shot's best moment
    let thumbs: Vec<Option<(egui::TextureId, egui::Vec2)>> = (0..app.ui.showcase.entries.len().min(THUMBS_PER_FRAME * 8))
        .map(|i| {
            let first = app.ui.showcase.entries.get(i).and_then(|e| shots(&e.entry).into_iter().next());
            first.and_then(|(item, _, best)| app.thumbnail(&ctx, filmcraft_project::ItemId(item), filmcraft_time::Tick::from_seconds_f64(best), 128))
        })
        .collect();
    let voices = voices();
    let mut actions: Vec<Action> = Vec::new();
    let auto = &mut app.auto;
    let d = &mut app.ui.showcase;

    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - FOOTER_H));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("showcase-body"));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut child, |ui| {
        ui.set_max_width(ui.available_width() - 6.0);
        let w = ui.available_width();
        // what is here, and the Analyze… form
        ui.horizontal(|ui| {
            let shown = d.entries.iter().filter(|e| e.include && !e.merge_previous).count();
            let what = if d.entries.is_empty() { "No showcase planned yet".to_string() } else { format!("{} entries, {shown} in the video", d.entries.len()) };
            ui.label(egui::RichText::new(what).color(t.text_dim).size(11.5));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let b = ui.add_enabled(!busy, egui::Button::new("Analyze…"));
                add(auto, ui, "showcase.analyzeOpen", b.rect, "Analyze…");
                if b.clicked() {
                    d.analyze_open = !d.analyze_open;
                }
            });
        });
        if let Err(why) = &available {
            ui.label(egui::RichText::new(why).color(t.text_faint).size(11.0));
        }
        if d.analyze_open && !busy {
            ui.add_space(6.0);
            egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(w - 18.0);
                let fw = w - 18.0;
                label(ui, &t, "Find every clip that shows");
                let r = ui.add(egui::TextEdit::singleline(&mut d.subject).hint_text("robot").desired_width(fw).char_limit(80));
                add(auto, ui, "showcase.subject", r.rect, "Subject");
                label(ui, &t, "Filmed since (optional)");
                let r = ui.add(egui::TextEdit::singleline(&mut d.since).hint_text("2020 or 2020-06-15").desired_width(fw).char_limit(10));
                add(auto, ui, "showcase.since", r.rect, "Filmed since");
                label(ui, &t, "Or describe the video");
                let r = ui.add(egui::TextEdit::singleline(&mut d.prompt).hint_text("make a video of every robot since 2020").desired_width(fw).char_limit(400));
                add(auto, ui, "showcase.prompt", r.rect, "Describe the video");
                label(ui, &t, "From a Google Photos album (optional)");
                let r = ui.add(egui::TextEdit::singleline(&mut d.album).hint_text("the album's exact title").desired_width(fw).char_limit(500));
                add(auto, ui, "showcase.album", r.rect, "Google Photos album");
                ui.label(
                    egui::RichText::new(
                        "A vision model looks at every video clip, then the clips are grouped into entries and a voice-over is written. Nothing is built until you choose Build.",
                    )
                    .color(t.text_dim)
                    .size(11.0),
                );
                ui.horizontal(|ui| {
                    let ready = available.is_ok() && (!d.subject.trim().is_empty() || !d.prompt.trim().is_empty());
                    let b = ui.add_enabled(ready, egui::Button::new(egui::RichText::new("Analyze").strong()));
                    add(auto, ui, "showcase.analyze", b.rect, "Analyze");
                    if b.clicked() {
                        actions.push(Action::Analyze);
                    }
                    let c = ui.button("Cancel");
                    add(auto, ui, "showcase.analyzeCancel", c.rect, "Cancel");
                    if c.clicked() {
                        d.analyze_open = false;
                    }
                });
            });
        }
        // long video → shorts, from the active sequence
        ui.add_space(6.0);
        let toggle_text = if d.shorts_open { format!("{} ▾", tl!("Shorts")) } else { tl!("Make shorts from this sequence…").to_string() };
        let toggle = ui.add_enabled(!busy, egui::Button::new(toggle_text).small());
        add(auto, ui, "showcase.shortsOpen", toggle.rect, "Make shorts");
        if toggle.clicked() {
            d.shorts_open = !d.shorts_open;
        }
        if d.shorts_open && !busy {
            egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(w - 18.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Shorts")).color(t.text).size(12.0));
                    let r = ui.add(egui::DragValue::new(&mut d.shorts_count).range(1..=20));
                    add(auto, ui, "showcase.shortsCount", r.rect, "How many shorts");
                    let cb = egui::ComboBox::from_id_salt("showcase-shorts-aspect").selected_text(d.shorts_aspect.clone()).show_ui(ui, |ui| {
                        for a in SHORT_ASPECTS {
                            if ui.selectable_label(d.shorts_aspect == a, a).clicked() {
                                d.shorts_aspect = a.to_string();
                            }
                        }
                    });
                    add(auto, ui, "showcase.shortsAspect", cb.response.rect, "Shape");
                });
                let r = ui.add(egui::TextEdit::singleline(&mut d.shorts_export).hint_text(tl!("Export folder (optional)")).desired_width(w - 18.0).char_limit(4096));
                add(auto, ui, "showcase.shortsExport", r.rect, "Export folder");
                let about = tl!("The transcript's best passages, on whole sentences, each a new sequence reframed to follow the speaker, with big captions.");
                ui.label(egui::RichText::new(about).color(t.text_dim).size(11.0));
                let b = ui.add_enabled(available.is_ok(), egui::Button::new(egui::RichText::new(tl!("Make shorts")).strong()));
                add(auto, ui, "showcase.shorts", b.rect, "Make shorts");
                if b.clicked() {
                    actions.push(Action::Shorts);
                }
            });
        }
        // the running (or last failed) job
        if let Some(j) = job.as_ref().filter(|j| j.running || j.error.is_some()) {
            ui.add_space(6.0);
            egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(w - 18.0);
                let text = j.error.clone().unwrap_or_else(|| j.status.clone());
                ui.horizontal_wrapped(|ui| {
                    if j.running {
                        ui.spinner();
                    }
                    ui.label(egui::RichText::new(text).color(if j.error.is_some() { t.hot_text } else { t.text }).size(11.5));
                });
                if j.running {
                    // counts once the job knows them (clips analysed, entries built)
                    if j.fraction > 0.0 {
                        ui.add(egui::ProgressBar::new(j.fraction).desired_width(w - 18.0).fill(t.accent).show_percentage());
                    }
                    let b = ui.small_button("Cancel");
                    add(auto, ui, "showcase.cancel", b.rect, "Cancel");
                    if b.clicked() {
                        actions.push(Action::Cancel(j.id));
                    }
                }
            });
        }
        ui.add_space(10.0);
        label(ui, &t, "Title");
        let r = ui.add(egui::TextEdit::singleline(&mut d.title).hint_text("Every Robot I've Filmed").desired_width(w).char_limit(200));
        add(auto, ui, "showcase.title", r.rect, "Title");
        ui.add_space(6.0);
        ui.columns(2, |cols| {
            if let [left, right] = cols {
                label(left, &t, "Voice");
                let name = voices.iter().find(|(id, _)| *id == d.voice).map_or_else(|| d.voice.clone(), |(_, n)| n.clone());
                let cb = egui::ComboBox::from_id_salt("showcase-voice").selected_text(name).width(left.available_width()).show_ui(left, |ui| {
                    for (id, n) in &voices {
                        if ui.selectable_label(d.voice == *id, n).clicked() {
                            d.voice = id.clone();
                        }
                    }
                });
                add(auto, left, "showcase.voice", cb.response.rect, "Voice");
                label(right, &t, "Mood");
                let mood = MOODS.iter().find(|(id, _)| *id == d.mood).map_or(d.mood.as_str(), |(_, n)| n).to_string();
                let cb = egui::ComboBox::from_id_salt("showcase-mood").selected_text(mood).width(right.available_width()).show_ui(right, |ui| {
                    for (id, n) in MOODS {
                        if ui.selectable_label(d.mood == id, n).clicked() {
                            d.mood = id.to_string();
                        }
                    }
                });
                add(auto, right, "showcase.mood", cb.response.rect, "Mood");
            }
        });
        ui.add_space(6.0);
        label(ui, &t, "Intro");
        let r = ui.add(egui::TextEdit::multiline(&mut d.intro).hint_text("What the narrator says first").desired_width(w).desired_rows(2).char_limit(MAX_TEXT));
        add(auto, ui, "showcase.intro", r.rect, "Intro");
        // the entries, oldest first
        let mut number = 0;
        for (i, e) in d.entries.iter_mut().enumerate() {
            ui.add_space(8.0);
            let shown = e.include && !(e.merge_previous && number > 0);
            if shown {
                number += 1;
            }
            let fill = if e.include { t.field_bg } else { t.panel_bg };
            egui::Frame::new().fill(fill).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(w - 18.0);
                let fw = w - 18.0;
                ui.horizontal(|ui| {
                    let c = ui.checkbox(&mut e.include, "");
                    add(auto, ui, &format!("showcase.entry.{i}.include"), c.rect, "Include in the video");
                    let (tr, resp) = ui.allocate_exact_size(vec2(64.0, 36.0), Sense::click());
                    ui.painter().rect_filled(tr, 2.0, egui::Color32::from_rgb(12, 12, 12));
                    if let Some(Some((tex, sz))) = thumbs.get(i) {
                        ui.painter().image(*tex, crate::panels::monitor::fit(tr, sz.x, sz.y), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), egui::Color32::WHITE);
                    }
                    add(auto, ui, &format!("showcase.entry.{i}.thumb"), tr, "Play in the Source monitor");
                    if let Some((item, _, best)) = shots(&e.entry).into_iter().next()
                        && resp.on_hover_text("Play in the Source monitor").clicked()
                    {
                        actions.push(Action::Play { item, seconds: best });
                    }
                    ui.vertical(|ui| {
                        let head = match (e.include, e.merge_previous && number > 0 && e.include) {
                            (false, _) => "Left out".to_string(),
                            (true, true) => format!("With #{number}"),
                            (true, false) => format!("#{number}"),
                        };
                        let when = e.entry["unix"].as_i64().map(date).unwrap_or_default();
                        ui.label(egui::RichText::new(head).color(t.text).size(12.5).strong());
                        ui.label(egui::RichText::new(when).color(t.text_dim).size(11.0));
                    });
                });
                // its name, what it is like, and the words over it
                let field_w = (fw - CAPTION_W - 6.0).max(40.0);
                egui::Grid::new(("showcase-entry", i)).num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
                    caption(ui, &t, "Name");
                    let r = ui.add(egui::TextEdit::singleline(&mut e.label).hint_text("What it is").desired_width(field_w).char_limit(MAX_LABEL));
                    add(auto, ui, &format!("showcase.entry.{i}.label"), r.rect, "Label");
                    ui.end_row();
                    caption(ui, &t, "About");
                    let r = ui.add(
                        egui::TextEdit::multiline(&mut e.description).hint_text("What it looks like and does").desired_width(field_w).desired_rows(1).char_limit(MAX_TEXT),
                    );
                    add(auto, ui, &format!("showcase.entry.{i}.description"), r.rect, "Description");
                    ui.end_row();
                    if shown || !e.include {
                        caption(ui, &t, "Says");
                        let line = egui::TextEdit::multiline(&mut e.line).hint_text("What the narrator says").desired_width(field_w).desired_rows(2).char_limit(MAX_TEXT);
                        let r = ui.add_enabled(e.include, line);
                        add(auto, ui, &format!("showcase.entry.{i}.line"), r.rect, "Narration line");
                        ui.end_row();
                    }
                });
                ui.horizontal(|ui| {
                    let m = ui.add_enabled(i > 0, egui::Checkbox::new(&mut e.merge_previous, "Merge with previous"));
                    add(auto, ui, &format!("showcase.entry.{i}.merge"), m.rect, "Merge with previous");
                    let list = shots(&e.entry);
                    let twirl = ui.add(
                        egui::Label::new(egui::RichText::new(format!("{} {} shot(s)", if e.open { "▾" } else { "▸" }, list.len())).color(t.text_dim).size(11.5))
                            .sense(Sense::click()),
                    );
                    add(auto, ui, &format!("showcase.entry.{i}.shots"), twirl.rect, "Shots");
                    if twirl.clicked() {
                        e.open = !e.open;
                    }
                });
                if e.open {
                    for (k, (item, name, best)) in shots(&e.entry).into_iter().enumerate() {
                        let b = ui.small_button(format!("▶ {name} at {best:.1} s"));
                        add(auto, ui, &format!("showcase.entry.{i}.shot.{k}"), b.rect, "Play shot");
                        if b.clicked() {
                            actions.push(Action::Play { item, seconds: best });
                        }
                    }
                }
            });
        }
        ui.add_space(8.0);
        label(ui, &t, "Outro");
        let r = ui.add(egui::TextEdit::multiline(&mut d.outro).hint_text("What the narrator says last").desired_width(w).desired_rows(2).char_limit(MAX_TEXT));
        add(auto, ui, "showcase.outro", r.rect, "Outro");
    });

    // footer: where to export, then Write script | Build | Build and export
    let fx = rect.min.x + PAD;
    let fw = rect.width() - PAD * 2.0;
    let row1 = Rect::from_min_size(pos2(fx, rect.max.y - FOOTER_H + 8.0), vec2(fw, 24.0));
    let mut top = ui.new_child(egui::UiBuilder::new().max_rect(row1).id_salt("showcase-export").layout(egui::Layout::left_to_right(egui::Align::Center)));
    let up = top.checkbox(&mut d.upload, "Upload to Google Photos");
    app.auto.add("showcase.upload", up.rect, "Upload to Google Photos");
    let pw = top.available_width();
    let hint = default_export_path(&d.title);
    let r = top.add(egui::TextEdit::singleline(&mut d.export_path).hint_text(hint).desired_width(pw).char_limit(4096));
    app.auto.add("showcase.exportPath", r.rect, "Export to");
    let fy = rect.max.y - FOOTER_H + 44.0;
    let third = (fw - 16.0) / 3.0;
    let included = d.entries.iter().any(|e| e.include);
    let can_write = available.is_ok() && !busy && (!d.entries.is_empty() || has_report);
    let can_build = available.is_ok() && !busy && included;
    let can_export = can_build;
    let buttons = [
        ("showcase.writeScript", "Write script", can_write, Action::WriteScript),
        ("showcase.build", "Build", can_build, Action::Build { export: false }),
        ("showcase.buildExport", "Build and export", can_export, Action::Build { export: true }),
    ];
    for (k, (id, text, enabled, action)) in buttons.into_iter().enumerate() {
        let r = Rect::from_min_size(pos2(fx + k as f32 * (third + 8.0), fy), vec2(third, 28.0));
        let primary = k == 2;
        let b = ui.put(
            r,
            egui::Button::new(egui::RichText::new(text).strong().color(match (enabled, primary) {
                (false, _) => t.text_faint,
                (true, true) => egui::Color32::WHITE,
                (true, false) => t.text,
            }))
            .fill(if primary && enabled { t.accent } else { t.field_bg })
            .sense(if enabled { Sense::click() } else { Sense::hover() }),
        );
        app.auto.add(id, r, text);
        if enabled && b.clicked() {
            actions.push(action);
        }
    }

    for a in actions {
        let r = match a {
            Action::Analyze => {
                let p = app.ui.showcase.make_params();
                app.session.execute("autocut.makeShowcase", p).map(|_| "Finding the clips…")
            }
            Action::Shorts => {
                let p = app.ui.showcase.shorts_params();
                app.session.execute("autocut.shorts", p).map(|_| tl!("Finding the best passages for shorts…"))
            }
            Action::WriteScript => {
                let d = &app.ui.showcase;
                let mut p = json!({"title": d.title.trim(), "voice": d.voice.trim(), "mood": d.mood});
                if !d.subject.trim().is_empty() {
                    p["subject"] = json!(d.subject.trim());
                }
                if !d.entries.is_empty() {
                    p["entries"] = Value::Array(d.reviewed_entries());
                }
                app.session.execute("autocut.showcasePlan", p).map(|_| "Writing the script…")
            }
            Action::Build { export } => {
                let d = &app.ui.showcase;
                let mut p = json!({"plan": d.plan()});
                if export {
                    p["export"] = json!(d.export_to());
                    p["upload"] = json!(d.upload);
                }
                app.session.execute("autocut.showcaseBuild", p).map(|_| "Building the showcase…")
            }
            Action::Cancel(job) => app.session.execute("jobs.cancel", json!({"job": job})).map(|_| "Cancelling…"),
            Action::Play { item, seconds } => {
                let r = app
                    .session
                    .execute("source.open", json!({"item": item}))
                    .and_then(|_| app.session.execute("source.setPlayhead", json!({"seconds": (seconds - 1.0).max(0.0)})));
                match r {
                    Ok(_) => app.play_source().map(|()| "Playing the shot").map_err(filmcraft_engine::EngineError::Other),
                    Err(e) => Err(e),
                }
            }
        };
        app.ui.status = match r {
            Ok(s) => s.to_string(),
            Err(e) => e.to_string(),
        };
    }
}

/// Register widget `id` by the part of `r` the scroll area shows (what a click can reach); one
/// scrolled out of view is not registered, so a click by id can't land on whatever is drawn
/// where it would be.
fn add(auto: &mut crate::automation::Registry, ui: &egui::Ui, id: &str, r: Rect, label: &str) {
    let shown = r.intersect(ui.clip_rect());
    if shown.is_positive() {
        auto.add(id, shown, label);
    }
}

/// Width of an entry's field captions.
const CAPTION_W: f32 = 40.0;

fn caption(ui: &mut egui::Ui, t: &crate::theme::Tokens, text: &str) {
    ui.add_sized([CAPTION_W, 18.0], egui::Label::new(egui::RichText::new(text).color(t.text_dim).size(11.0)));
}

fn label(ui: &mut egui::Ui, t: &crate::theme::Tokens, text: &str) {
    ui.label(egui::RichText::new(text).color(t.text).size(12.5).strong());
    ui.add_space(2.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Value {
        json!({
            "subject": "robot", "title": "Robots", "voice": "clone:me", "mood": "calm", "subtitle": "2020 – 2026", "width": 320,
            "entries": [
                {"number": 1, "label": "red robot", "kind": "toy", "description": "a red robot", "unix": 1_665_741_668, "shots": [{"item": 5, "name": "a.mp4", "clip_seconds": 10.0, "best": 4.0}]},
                {"number": 2, "label": "Spot", "kind": "quadruped", "description": "a robot dog", "unix": null, "shots": [], "include": false, "mergePrevious": true},
            ],
            "script": {"intro": "Hi.", "lines": ["One.", ""], "outro": "Bye."},
        })
    }

    #[test]
    fn a_plan_loads_and_goes_back_with_the_review() {
        let mut d = ShowcaseDraft::default();
        d.load(&plan());
        assert_eq!((d.title.as_str(), d.voice.as_str(), d.mood.as_str(), d.intro.as_str()), ("Robots", "clone:me", "calm", "Hi."));
        assert_eq!(d.entries.len(), 2);
        assert!(d.entries[0].include && !d.entries[1].include && d.entries[1].merge_previous);
        assert_eq!((d.entries[0].line.as_str(), d.entries[1].line.as_str()), ("One.", ""));
        assert_eq!(d.extra.keys().collect::<Vec<_>>(), ["subtitle", "width"]);
        d.entries[0].label = "Ruby".into();
        d.entries[1].include = true;
        d.voice = String::new();
        let p = d.plan();
        assert_eq!(p["entries"][0]["label"], "Ruby");
        assert_eq!(p["entries"][0]["shots"][0]["item"], 5, "the engine's entry goes back whole");
        assert_eq!((p["entries"][1]["include"].clone(), p["entries"][1]["mergePrevious"].clone()), (json!(true), json!(true)));
        assert_eq!((p["subtitle"].as_str(), p["width"].as_u64()), (Some("2020 – 2026"), Some(320)));
        assert!(p.get("voice").is_none(), "no narration");
        assert_eq!(p["script"]["lines"], json!(["One.", ""]));
        // what loads back is what went out
        let mut again = ShowcaseDraft::default();
        again.load(&p);
        let review = |d: &ShowcaseDraft| {
            d.entries.iter().map(|e| (e.include, e.merge_previous, e.label.clone(), e.description.clone(), e.line.clone())).collect::<Vec<_>>()
        };
        assert_eq!(review(&again), review(&d));
    }

    #[test]
    fn hostile_plans_load_without_trusting_them() {
        let mut d = ShowcaseDraft::default();
        for p in [json!(null), json!([1, 2]), json!({"entries": "all", "script": 7}), json!({"entries": [5, {"shots": "x"}], "title": 3})] {
            d.load(&p);
            for e in &d.entries {
                assert!(shots(&e.entry).is_empty(), "{p}");
            }
        }
        let many = json!({"entries": vec![json!({}); MAX_ENTRIES + 10]});
        d.load(&many);
        assert_eq!(d.entries.len(), MAX_ENTRIES);
        // a shot without a best moment, or a non-finite one, is no shot
        assert!(shots(&json!({"shots": [{"item": 1}, {"item": 2, "best": "x"}]})).is_empty());
    }

    #[test]
    fn make_params_prefer_the_subject_over_the_prompt() {
        let d = ShowcaseDraft { subject: " robot ".into(), since: "2020".into(), prompt: "ignored".into(), ..Default::default() };
        assert_eq!(d.make_params(), json!({"subject": "robot", "since": "2020", "voice": "", "mood": "epic"}));
        let d = ShowcaseDraft { subject: String::new(), prompt: "make a video of every robot".into(), title: "Robots".into(), ..Default::default() };
        assert_eq!(d.make_params(), json!({"prompt": "make a video of every robot", "title": "Robots", "voice": "", "mood": "epic"}));
        // a Google Photos album as the seed
        let d = ShowcaseDraft { album: " Robot showcase ".into(), ..Default::default() };
        assert_eq!(d.make_params(), json!({"subject": "robot", "album": "Robot showcase", "voice": "", "mood": "epic"}));
    }

    #[test]
    fn shorts_params_from_the_form() {
        let mut d = ShowcaseDraft::default();
        assert_eq!(d.shorts_params(), json!({"count": 3, "aspect": "9:16", "captions": true}));
        // what an agent or a hand-edited state could put there
        d.shorts_count = 99;
        d.shorts_aspect = "weird".into();
        d.shorts_export = " /tmp/out ".into();
        assert_eq!(d.shorts_params(), json!({"count": 20, "aspect": "9:16", "captions": true, "export": "/tmp/out"}));
        d.shorts_count = 0;
        d.shorts_aspect = "4:5".into();
        assert_eq!(d.shorts_params()["count"], 1);
        assert_eq!(d.shorts_params()["aspect"], "4:5");
    }

    #[test]
    fn dates_and_export_paths() {
        assert_eq!(date(1_665_741_668), "2022-10-14");
        assert_eq!(date(0), "1970-01-01");
        assert!(default_export_path("Every Robot: 2020/2026").ends_with("/Every Robot_ 2020_2026.mp4"));
        assert!(default_export_path("  ").ends_with("/Showcase.mp4"));
        let d = ShowcaseDraft { title: "Robots".into(), ..Default::default() };
        assert!(d.export_to().ends_with("/Robots.mp4"), "the default follows the title");
        let d = ShowcaseDraft { export_path: " /tmp/x.mp4 ".into(), ..d };
        assert_eq!(d.export_to(), "/tmp/x.mp4");
    }
}

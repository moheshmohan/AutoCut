//! Generate Audio panel: make sound effects and music beds in the app — a whoosh, an impact, a
//! riser, a drone, or a music bed in a mood, key and tempo — and add them at the playhead. They
//! are synthesized (oscillators, noise, filters; no samples or models), written as WAV files in
//! a `Generated Audio` folder beside the project and placed on a free audio track as one undo
//! step (`audio.generate`, the same command agents and the CLI use).
//!
//! The same settings and seed always give the same sound; "New variation" picks another seed.
//!
//! Automation ids: `generateAudio.kind`, `generateAudio.length`, `generateAudio.intensity`,
//! `generateAudio.mood`, `generateAudio.key`, `generateAudio.tempo`, `generateAudio.variation`,
//! `generateAudio.add`.

use egui::{Rect, pos2};
use filmcraft_audio_dsp::synth::{Kind, Mood};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::FilmcraftApp;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 48.0;
/// Keys offered for music and drones.
const KEYS: [&str; 12] = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"];

/// The cue being made (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateAudioDraft {
    /// `whoosh`, `impact`, `riser`, `drone` or `music`.
    pub kind: String,
    /// Length, seconds.
    pub seconds: f64,
    /// 0..1.
    pub intensity: f64,
    /// Music: a mood id (`epic`, `tense`, `uplifting`, `calm`, `happy`, `dark`, `dreamy`,
    /// `energetic`, `sad`).
    pub mood: String,
    /// Music and drones.
    pub key: String,
    /// Music, beats per minute.
    pub bpm: f64,
    /// The variation; `None` until "New variation" picks one, then each cue of a kind gets the
    /// next seed.
    pub seed: Option<u64>,
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
        }
    }
}

impl GenerateAudioDraft {
    fn kind(&self) -> Kind {
        Kind::from_id(&self.kind).unwrap_or(Kind::Whoosh)
    }

    /// The `audio.generate` parameters (placed at the playhead).
    pub fn params(&self) -> serde_json::Value {
        let k = self.kind();
        let mut p = json!({"kind": k.id(), "duration": self.seconds.clamp(0.1, k.max_seconds()), "intensity": self.intensity.clamp(0.0, 1.0)});
        if matches!(k, Kind::Music | Kind::Drone) {
            p["key"] = json!(self.key);
        }
        if k == Kind::Music {
            p["mood"] = json!(self.mood);
            p["bpm"] = json!(self.bpm.clamp(40.0, 200.0));
        }
        if let Some(seed) = self.seed {
            p["seed"] = json!(seed);
        }
        p
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Whoosh => "Whoosh",
        Kind::Impact => "Impact",
        Kind::Riser => "Riser",
        Kind::Drone => "Drone",
        Kind::Music => "Music bed",
    }
}

fn kind_hint(k: Kind) -> &'static str {
    match k {
        Kind::Whoosh => "A swept noise pass for transitions and fast moves.",
        Kind::Impact => "A cinematic hit: a sub boom, a crack and a long tail.",
        Kind::Riser => "Tension that builds to a cut and stops dead.",
        Kind::Drone => "A low, slowly moving chord under a scene.",
        Kind::Music => {
            "Music arranged to the exact length and ending on a bar: drums, bass, melody and pads in sections, at -20 LUFS so it sits under a voice."
        }
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let has_seq = app.session.active_sequence().is_some();
    let mut add = false;
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
        ui.add_space(10.0);
        label(ui, "Length");
        ui.spacing_mut().slider_width = w - 70.0;
        let s = ui.add(egui::Slider::new(&mut d.seconds, 0.2..=kind.max_seconds()).logarithmic(true).max_decimals(1).suffix(" s"));
        auto.add("generateAudio.length", visible(ui, s.rect), "Length");
        ui.add_space(10.0);
        label(ui, "Intensity");
        let s = ui.add(egui::Slider::new(&mut d.intensity, 0.0..=1.0).fixed_decimals(2));
        auto.add("generateAudio.intensity", visible(ui, s.rect), "Intensity");
        if kind == Kind::Music {
            ui.add_space(10.0);
            label(ui, "Mood");
            let current = Mood::from_id(&d.mood).unwrap_or(Mood::Epic);
            let cb = egui::ComboBox::from_id_salt("generate-audio-mood").selected_text(current.name()).width(w).show_ui(ui, |ui| {
                for m in Mood::ALL {
                    if ui.selectable_label(m == current, m.name()).clicked() {
                        d.mood = m.id().into();
                    }
                }
            });
            auto.add("generateAudio.mood", visible(ui, cb.response.rect), "Mood");
            ui.add_space(10.0);
            label(ui, "Tempo");
            let s = ui.add(egui::Slider::new(&mut d.bpm, 40.0..=200.0).step_by(1.0).suffix(" BPM"));
            auto.add("generateAudio.tempo", visible(ui, s.rect), "Tempo");
        }
        if matches!(kind, Kind::Music | Kind::Drone) {
            ui.add_space(10.0);
            label(ui, "Key");
            let cb = egui::ComboBox::from_id_salt("generate-audio-key").selected_text(d.key.clone()).width(w).show_ui(ui, |ui| {
                for k in KEYS {
                    if ui.selectable_label(d.key == k, k).clicked() {
                        d.key = k.into();
                    }
                }
            });
            auto.add("generateAudio.key", visible(ui, cb.response.rect), "Key");
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
    let fy = rect.max.y - FOOTER_H + 10.0;
    let r = Rect::from_min_size(pos2(rect.min.x + PAD, fy), egui::vec2(rect.width() - PAD * 2.0, 28.0));
    let b = ui.put(r, egui::Button::new(egui::RichText::new("Add at playhead").strong()).fill(if has_seq { t.accent } else { t.field_bg }));
    app.auto.add("generateAudio.add", r, "Add at playhead");
    if b.clicked() && has_seq {
        add = true;
    }
    if !has_seq {
        b.on_hover_text("Open a sequence first");
    }
    if add {
        let p = app.ui.generate_audio.params();
        app.ui.status = match app.session.execute("audio.generate", p) {
            Ok(r) => format!(
                "Added {} on {}",
                r["path"].as_str().and_then(|p| std::path::Path::new(p).file_stem()).map_or("the cue".into(), |s| s.to_string_lossy().into_owned()),
                r["track"].as_str().unwrap_or("an audio track")
            ),
            Err(e) => e.to_string(),
        };
    }
}

/// A seed different from `prev`, from the clock (a splitmix step so close times differ widely).
fn next_seed(prev: Option<u64>) -> u64 {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
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
        // an unknown kind (from a hand-edited state) falls back to a whoosh
        assert_eq!(GenerateAudioDraft { kind: "laser".into(), ..Default::default() }.params()["kind"], "whoosh");
    }

    #[test]
    fn a_new_variation_differs() {
        let a = next_seed(None);
        assert_ne!(next_seed(Some(a)), a);
    }
}

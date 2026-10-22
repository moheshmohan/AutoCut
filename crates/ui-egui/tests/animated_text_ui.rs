//! Headless UI tests of the animated caption and text effect controls, driven through the
//! control channel: the Captions tab's caption style picker (a live preview tile per style, the
//! platform, word timing) and the Properties panel's Outer Glow, Inner Shadow, Stroke 3, Curve
//! and Text Animation controls.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    shots: Option<std::path::PathBuf>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let shots = std::env::var_os("FILMCRAFT_UI_SHOTS").map(std::path::PathBuf::from);
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if shots.is_some() {
            b = b.wgpu().with_pixels_per_point(1.0);
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, shots };
        d.frames(4);
        d
    }
    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }
    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }
    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }
    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }
    fn rect(&mut self, id: &str) -> [f64; 4] {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        let e = v.as_array().unwrap().iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("no element {id}: {v}")).clone();
        let r: Vec<f64> = e["rect"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        [r[0], r[1], r[2], r[3]]
    }
    /// Scroll the panel holding `id` until it is on screen.
    fn reveal(&mut self, id: &str) {
        for _ in 0..40 {
            let r = self.rect(id);
            if r[1] > 120.0 && r[1] + r[3] < 860.0 {
                return;
            }
            let dy = if r[1] < 120.0 { 120.0 } else { -120.0 };
            self.ok("ui.scroll", json!({"x": r[0] + 4.0, "y": 480.0, "dy": dy}));
            self.frames(2);
        }
    }
    /// Scroll the caption style tiles sideways (`dx` < 0 shows the ones on the right).
    fn slide_tiles(&mut self, dx: f64) {
        let r = self.rect("text.captions.animated.wordTiming");
        let off = self.rect("text.captions.animated.off");
        self.ok("ui.scroll", json!({"x": r[0] + 30.0, "y": off[1] + off[3] / 2.0, "dx": dx}));
        self.frames(3);
    }
    /// With `FILMCRAFT_UI_SHOTS=<dir>`, render the UI (wgpu, no window needed) to `<dir>/<name>.png`.
    fn shot(&mut self, name: &str) {
        let Some(dir) = self.shots.clone() else { return };
        self.frames(8);
        let img = self.harness.render().expect("wgpu render");
        std::fs::create_dir_all(&dir).unwrap();
        img.save(dir.join(format!("{name}.png"))).unwrap();
    }
    fn caption_style(&mut self) -> Value {
        let r = self.exec("captions.list", json!({}));
        r["tracks"][0]["style"].clone()
    }
}

#[test]
fn the_caption_style_picker_sets_animated_styles() {
    let mut d = Driver::demo();
    d.ok("ui.set", json!({"workspace": "Captions and Graphics"}));
    let track = d.exec("captions.newTrack", json!({"format": "Subtitle"}))["track"].as_u64().unwrap();
    d.exec("captions.add", json!({"track": track, "text": "make it count", "seconds": 1.0}));
    d.frames(2);
    d.ok("ui.click", json!({"id": "text.tab.Captions"}));
    d.frames(3);
    // a tile for whole captions and one for each animated style
    let tiles = d.ids("text.captions.animated.");
    for name in ["off", "bold", "clean", "glow", "box", "karaoke"] {
        assert!(tiles.iter().any(|t| *t == format!("text.captions.animated.{name}")), "{name} tile: {tiles:?}");
    }
    assert!(tiles.iter().any(|t| t == "text.captions.animated.wordTiming"), "{tiles:?}");
    assert!(d.caption_style()["animated"].is_null());
    d.ok("ui.click", json!({"id": "text.captions.animated.bold"}));
    d.frames(3);
    assert_eq!(d.caption_style()["animated"]["preset"], "bold");
    d.shot("caption-style-picker");
    // with a style on, the platform picker shows; the Glow tile switches style, Off back to whole captions
    assert!(d.ids("text.captions.animated.platform").contains(&"text.captions.animated.platform".to_string()));
    d.slide_tiles(-1000.0);
    d.ok("ui.click", json!({"id": "text.captions.animated.karaoke"}));
    d.frames(3);
    assert_eq!(d.caption_style()["animated"]["preset"], "karaoke");
    d.exec("captions.setAnimatedStyle", json!({"track": track, "preset": "karaoke", "options": {"platform": "tiktok"}}));
    d.frames(2);
    assert_eq!(d.caption_style()["animated"]["platform"], "tiktok");
    d.slide_tiles(1000.0);
    d.ok("ui.click", json!({"id": "text.captions.animated.off"}));
    d.frames(3);
    assert!(d.caption_style()["animated"].is_null());
    // word timing opens its menu (transcript / narration)
    d.ok("ui.click", json!({"id": "text.captions.animated.wordTiming"}));
    d.frames(3);
}

#[test]
fn properties_offer_glow_inner_shadow_stroke3_curve_and_text_animation() {
    let mut d = Driver::demo();
    d.ok("ui.set", json!({"workspace": "Captions and Graphics"}));
    d.exec("playhead.set", json!({"seconds": 0.0}));
    let clip = d.exec("graphics.newText", json!({"text": "Title", "seconds": 4}))["clip"].as_u64().unwrap();
    d.exec("graphics.selectLayer", json!({"clip": clip, "layers": [0]}));
    d.frames(4);
    for id in [
        "graphics.prop.glow",
        "graphics.prop.inner_shadow",
        "graphics.prop.stroke3",
        "graphics.prop.curve",
        "graphics.prop.curve_radius",
        "graphics.anim.in.kind",
        "graphics.anim.out.kind",
        "graphics.anim.loop.kind",
    ] {
        assert!(d.ids(id).iter().any(|x| x == id), "{id}");
    }
    let layer_glow = |d: &mut Driver| -> bool {
        let s = &d.harness.state().session;
        let (_, it) = s.active_sequence().unwrap().find_item(filmcraft_project::ClipId(clip)).unwrap();
        let e = it.effects.iter().find(|e| e.effect == "graphic_text").unwrap();
        filmcraft_project::graphic::eval_layer(e, filmcraft_time::Tick::ZERO, (1920, 1080)).unwrap().appearance.glow.is_some()
    };
    assert!(!layer_glow(&mut d));
    d.reveal("graphics.prop.glow");
    d.ok("ui.click", json!({"id": "graphics.prop.glow"}));
    d.frames(3);
    assert!(layer_glow(&mut d), "the Outer Glow box turns the glow on");
    // the glow's radius and intensity show once it is on
    assert!(d.ids("graphics.prop.glow_radius").iter().any(|x| x == "graphics.prop.glow_radius"));
    // an animator set: its unit, easing, amount, stagger and duration controls appear
    assert!(d.ids("graphics.anim.in.unit").is_empty());
    d.exec("graphics.animateText", json!({"clip": clip, "layer": 0, "in": "pop", "loop": "wave"}));
    d.frames(3);
    for id in [
        "graphics.anim.in.unit",
        "graphics.anim.in.easing",
        "graphics.anim.in.amount",
        "graphics.anim.in.stagger",
        "graphics.anim.in.duration",
        "graphics.anim.loop.amount",
    ] {
        assert!(d.ids(id).iter().any(|x| x == id), "{id}");
    }
    // dragging the stagger is one undo step
    d.reveal("graphics.anim.in.stagger");
    d.shot("text-animation-controls");
    let undo0 = d.exec("history.list", json!({}))["undo"].as_array().unwrap().len();
    let r = d.rect("graphics.anim.in.stagger");
    let (cx, cy) = (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0);
    d.ok("ui.drag", json!({"from": {"x": cx, "y": cy}, "to": {"x": cx + 40.0, "y": cy}, "steps": 5}));
    d.frames(3);
    let undo1 = d.exec("history.list", json!({}))["undo"].as_array().unwrap().len();
    assert_eq!(undo1, undo0 + 1, "an animator drag is one undo step");
    let stagger = {
        let s = &d.harness.state().session;
        let (_, it) = s.active_sequence().unwrap().find_item(filmcraft_project::ClipId(clip)).unwrap();
        let e = it.effects.iter().find(|e| e.effect == "graphic_text").unwrap();
        e.layer.as_ref().unwrap().anim.as_ref().unwrap().enter.as_ref().unwrap().stagger
    };
    assert!(stagger > filmcraft_project::text_anim::Animator::preset(filmcraft_project::text_anim::AnimKind::Pop).stagger, "{stagger}");
}

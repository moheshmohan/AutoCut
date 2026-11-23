//! Test templates made in code at run time (never committed files): a lower third — a coloured
//! bar with a name over it, optionally a logo picture and a fade — exposed as Essential Graphics
//! controls `Name` (text), `Bar Color` (colour), `Bar Opacity` (slider) and `Logo` (media), and
//! packed as an `.ectemplate`. Only with the features `effectcraft` and `test-support` (or in
//! this crate's tests); a builder that cannot build fails loudly, as a test should.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use effectcraft_project::build::{self, Ids};
use effectcraft_project::essential::{EgControl, EgKind, EssentialGraphics};
use effectcraft_project::keyframe::TextDoc;
use effectcraft_project::{Comp, Expression, Footage, FootageKind, ItemId, ItemKind, Keyframe, LayerSource, Project, Value};
use effectcraft_time::{FrameRate, Tick};
use serde_json::json;

/// A picture the lower third shows at its top right.
#[derive(Clone, Debug)]
pub struct Logo {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Carried in the template (`media/<item>/logo.png`); else it points at `path`.
    pub embedded: bool,
    /// Where an unembedded logo is.
    pub path: String,
}

/// The lower third to build.
#[derive(Clone, Debug)]
pub struct LowerThird {
    pub size: (u32, u32),
    pub rate: (i64, i64),
    pub seconds: f64,
    pub name: String,
    /// The bar's colour, straight RGBA 0..1 (sRGB-encoded, as authored).
    pub bar: [f64; 4],
    /// An expression on the bar's opacity.
    pub opacity_expression: Option<String>,
    /// The bar fades in over the first half second (else it is still).
    pub fade_in: bool,
    pub logo: Option<Logo>,
}

impl Default for LowerThird {
    fn default() -> Self {
        LowerThird {
            size: (320, 180),
            rate: (30, 1),
            seconds: 1.0,
            name: "Ada Lovelace".into(),
            bar: [0.2, 0.4, 0.8, 1.0],
            opacity_expression: None,
            fade_in: false,
            logo: None,
        }
    }
}

/// Where the bar is, in comp pixels: `[x0, y0, x1, y1]`.
pub fn bar_box(size: (u32, u32)) -> [f64; 4] {
    let (w, h) = (f64::from(size.0), f64::from(size.1));
    [10.0, h - 50.0, 10.0 + w * 0.6, h - 10.0]
}

/// Where the logo's centre is, in comp pixels.
pub fn logo_centre(size: (u32, u32), logo: &Logo) -> [f64; 2] {
    [f64::from(size.0) - 10.0 - f64::from(logo.width) / 2.0, 10.0 + f64::from(logo.height) / 2.0]
}

fn uid_of(props: &effectcraft_project::PropGroup, pred: impl Fn(&effectcraft_project::Property) -> bool) -> u64 {
    let mut found = None;
    props.walk("", &mut |_, p| {
        if found.is_none() && pred(p) {
            found = Some(p.uid);
        }
    });
    found.expect("the property is there")
}

impl LowerThird {
    /// The project and its comp's id.
    pub fn project(&self) -> (Project, ItemId) {
        let (w, h) = self.size;
        let mut p = Project::default();
        let comp = Comp::new(w, h, FrameRate::new(self.rate.0, self.rate.1), Tick::from_seconds_f64(self.seconds));
        let cid = p.add_item("Lower Third", Default::default(), None, ItemKind::Comp(comp.clone().into()));
        let mut controls = Vec::new();
        // the bar: a shape layer whose origin is the comp's centre
        let mut bar = build::layer(&mut p, &comp, "Bar", LayerSource::Shape, (w, h), None);
        let [x0, y0, x1, y1] = bar_box(self.size);
        let (cx, cy) = ((x0 + x1) / 2.0 - f64::from(w) / 2.0, (y0 + y1) / 2.0 - f64::from(h) / 2.0);
        let mut next = p.next_id;
        let group = {
            let mut ids = Ids(&mut next);
            let rect = build::shape_rect(&mut ids, [x1 - x0, y1 - y0], [cx, cy], 0.0);
            let fill = build::shape_fill(&mut ids, self.bar);
            build::shape_group(&mut ids, "Bar", vec![rect, fill])
        };
        p.next_id = next;
        bar.props.sub_mut("contents").expect("a shape layer has contents").children.push(group.into());
        let color = uid_of(&bar.props, |pr| pr.match_id == "color" && matches!(pr.value, Value::Color(_)));
        let opacity = bar.props.prop("transform/opacity").expect("a transform").uid;
        if let Some(pr) = bar.props.prop_mut("transform/opacity") {
            if self.fade_in {
                pr.keys = vec![Keyframe::new(Tick::ZERO, Value::Scalar(0.0)), Keyframe::new(Tick::from_seconds_f64(0.5), Value::Scalar(100.0))];
            }
            if let Some(e) = &self.opacity_expression {
                pr.expr = Some(Expression { text: e.clone(), enabled: true });
            }
        }
        // the name: point text on the bar, in the bundled Inter (no system font is looked for)
        let mut text = build::layer(&mut p, &comp, "Name", LayerSource::Text, (w, h), None);
        let doc =
            TextDoc { text: self.name.clone(), font: "Inter".into(), style: "Regular".into(), size: 24.0, fill: [1.0, 1.0, 1.0, 1.0], ..TextDoc::default() };
        text.props.prop_mut("text/sourceText").expect("a text layer has Source Text").value = Value::Text(Box::new(doc));
        if let Some(pos) = text.props.prop_mut("transform/position") {
            pos.value = Value::Vec3([x0 + 10.0, y1 - 12.0, 0.0]);
        }
        let source_text = text.props.prop("text/sourceText").expect("Source Text").uid;
        let mut layers = vec![text, bar];
        let mut eg_id = 1_000_000;
        let mut control = |name: &str, kind: EgKind| {
            eg_id += 1;
            EgControl { id: eg_id, name: name.into(), kind, as_type: None }
        };
        let (text_id, bar_id) = (layers[0].id, layers[1].id);
        controls.push(control("Name", EgKind::Property { layer: text_id, prop: source_text, links: vec![] }));
        controls.push(control("Bar Color", EgKind::Property { layer: bar_id, prop: color, links: vec![] }));
        controls.push(control("Bar Opacity", EgKind::Property { layer: bar_id, prop: opacity, links: vec![] }));
        if let Some(logo) = &self.logo {
            let path = if logo.embedded { "media/0/logo.png".to_string() } else { logo.path.clone() };
            let f = Footage {
                path,
                kind: FootageKind::Still,
                width: logo.width,
                height: logo.height,
                pixel_aspect: 1.0,
                frame_rate: FrameRate::FPS_30,
                duration: Tick::from_seconds_f64(self.seconds),
                has_video: true,
                ..Footage::default()
            };
            let item = p.add_item("logo.png", Default::default(), None, ItemKind::Footage(f));
            let mut l = build::layer(&mut p, &comp, "Logo", LayerSource::Footage { item }, (logo.width, logo.height), None);
            let [lx, ly] = logo_centre(self.size, logo);
            if let Some(pos) = l.props.prop_mut("transform/position") {
                pos.value = Value::Vec3([lx, ly, 0.0]);
            }
            let id = l.id;
            layers.insert(0, l);
            // the logo's item id is what the media entry is named by
            if logo.embedded
                && let Some(ItemKind::Footage(f)) = p.item_mut(item).map(|i| &mut i.kind)
            {
                f.path = format!("media/{}/logo.png", item.0);
            }
            controls.push(control("Logo", EgKind::Media { layer: id }));
        }
        let c = p.comp_mut(cid).expect("the comp");
        for l in layers {
            c.layers.push(l);
        }
        c.essential = Some(EssentialGraphics { name: "Lower Third".into(), controls });
        (p, cid)
    }

    /// The manifest of a template of `project`'s comp `comp`.
    pub fn manifest(&self, project: &Project, comp: ItemId) -> serde_json::Value {
        json!({
            "format": "effectcraft-template", "version": 1, "generator": "filmcraft-effectcraft tests",
            "name": "Lower Third", "comp": comp.0, "compName": "Lower Third",
            "width": self.size.0, "height": self.size.1, "frameRate": self.rate.0 as f64 / self.rate.1 as f64,
            "duration": self.seconds, "controls": [], "media": media_entries(project), "poster": "poster.png",
        })
    }

    /// The `.ectemplate`.
    pub fn build(&self) -> Vec<u8> {
        self.build_with(|_, _| {})
    }

    /// The `.ectemplate` after `edit` changed its project or manifest (hostile templates).
    pub fn build_with(&self, edit: impl FnOnce(&mut Project, &mut serde_json::Value)) -> Vec<u8> {
        let (mut p, cid) = self.project();
        let mut m = self.manifest(&p, cid);
        edit(&mut p, &mut m);
        let mut extra = Vec::new();
        if let Some(logo) = self.logo.as_ref().filter(|l| l.embedded) {
            for e in media_entries(&p) {
                extra.push((e.as_str().unwrap_or_default().to_string(), logo.png.clone()));
            }
        }
        package(&m, &p.to_json(), &extra)
    }
}

/// The `media/…` paths of `project`'s footage.
fn media_entries(project: &Project) -> Vec<serde_json::Value> {
    project
        .items
        .values()
        .filter_map(|it| match &it.kind {
            ItemKind::Footage(f) if f.path.starts_with("media/") => Some(json!(f.path)),
            _ => None,
        })
        .collect()
}

/// A package of `manifest` and `project` (as given) and `extra` entries.
pub fn package(manifest: &serde_json::Value, project: &str, extra: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut files =
        vec![("manifest.json".to_string(), serde_json::to_vec_pretty(manifest).expect("JSON")), ("project.ecproj".to_string(), project.as_bytes().to_vec())];
    files.extend(extra.iter().cloned());
    filmcraft_templates::zip::write(&files).expect("a ZIP")
}

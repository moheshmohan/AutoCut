//! With EffectCraft linked: templates built in code ([`crate::testing`]) read, checked, filled and
//! rendered; hostile templates and values refused without a panic.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use effectcraft_project::{ItemKind, LayerSource};
use filmcraft_frame::VideoFrame;
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use crate::testing::{Logo, LowerThird, package};
use crate::{Error, HostFrames, HostMedia, NoFrames, RenderRequest};

/// A folder of its own for one test (removed when it ends).
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("fc-effectcraft-{}-{name}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A host that reads PNG files with the `image` crate (as the engine reads them with its own
/// decoders), counting the reads.
#[derive(Default)]
struct PngHost {
    reads: Mutex<Vec<String>>,
}

impl HostFrames for PngHost {
    fn probe(&self, path: &str) -> Option<HostMedia> {
        let img = image::open(path).ok()?;
        Some(HostMedia {
            width: img.width(),
            height: img.height(),
            rate: FrameRate::FPS_30,
            duration: Tick::from_seconds_f64(1.0),
            still: true,
            has_alpha: true,
            has_audio: false,
        })
    }
    fn frame(&self, path: &str, _: Tick) -> Option<Arc<VideoFrame>> {
        self.reads.lock().unwrap().push(path.to_string());
        let img = image::open(path).ok()?.to_rgba8();
        Some(Arc::new(VideoFrame::rgba8(img.width(), img.height(), img.into_raw())))
    }
}

/// A host whose every call panics.
struct PanickyHost;

impl HostFrames for PanickyHost {
    fn probe(&self, _: &str) -> Option<HostMedia> {
        panic!("probe")
    }
    fn frame(&self, _: &str, _: Tick) -> Option<Arc<VideoFrame>> {
        panic!("frame")
    }
}

/// A `w`×`h` PNG of one straight RGBA colour.
fn png(w: u32, h: u32, px: [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(w, h, image::Rgba(px));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn controls(info: &Value) -> Vec<&Value> {
    info["controls"].as_array().unwrap().iter().collect()
}

fn control<'a>(info: &'a Value, name: &str) -> &'a Value {
    controls(info).into_iter().find(|c| c["name"] == name).unwrap_or_else(|| panic!("no control {name}: {info:#}"))
}

#[test]
fn a_lower_third_is_described() {
    assert!(crate::available());
    let info = crate::template_info(&LowerThird::default().build()).unwrap();
    assert_eq!(info["name"], "Lower Third");
    let c = &info["comp"];
    assert_eq!((c["width"].as_u64(), c["height"].as_u64(), c["fps"].as_f64(), c["frames"].as_u64()), (Some(320), Some(180), Some(30.0), Some(30)), "{c:#}");
    assert_eq!(c["duration"].as_i64(), Some(Tick::from_seconds_f64(1.0).0));
    let name = control(&info, "Name");
    assert_eq!((name["kind"].as_str(), name["type"].as_str(), name["default"].as_str()), (Some("property"), Some("text"), Some("Ada Lovelace")));
    let bar = control(&info, "Bar Color");
    assert_eq!(bar["type"], "color");
    assert_eq!(bar["default"], json!([0.2, 0.4, 0.8, 1.0]));
    let opacity = control(&info, "Bar Opacity");
    assert_eq!((opacity["type"].as_str(), opacity["range"]["min"].as_f64(), opacity["range"]["max"].as_f64()), (Some("slider"), Some(0.0), Some(100.0)));
    assert_eq!(info["renderable"], true);
    assert_eq!(info["media"], json!([]));
    assert_eq!(info["effectcraft"]["pin"], crate::PIN);
}

#[test]
fn carried_media_and_a_media_control_are_listed() {
    let logo = Logo { png: png(16, 8, [255, 0, 0, 128]), width: 16, height: 8, embedded: true, path: String::new() };
    let info = crate::template_info(&LowerThird { logo: Some(logo), ..Default::default() }.build()).unwrap();
    let media = info["media"].as_array().unwrap();
    assert_eq!(media.len(), 1, "{info:#}");
    assert!(media[0]["entry"].as_str().unwrap().ends_with("/logo.png"));
    let logo = control(&info, "Logo");
    assert_eq!((logo["kind"].as_str(), logo["default"]["file"].as_str()), (Some("media"), Some("logo.png")));
    let footage = info["footage"].as_array().unwrap();
    assert_eq!((footage.len(), footage[0]["embedded"].as_bool()), (1, Some(true)));
}

#[test]
fn hostile_templates_are_refused_without_a_panic() {
    let good = LowerThird::default();
    let (p, cid) = good.project();
    let m = good.manifest(&p, cid);
    let project = p.to_json();
    let says = |bytes: Vec<u8>, what: &str| {
        let e = std::panic::catch_unwind(|| crate::template_info(&bytes)).expect("no panic").unwrap_err().to_string();
        assert!(e.contains(what), "{e:?} should say {what:?}");
    };
    // a manifest naming a comp that is not one
    let mut other = m.clone();
    other["comp"] = json!(999_999);
    says(package(&other, &project, &[]), "not a composition");
    // a project that is not one
    says(package(&m, "{\"schema\": 1, \"items\": 7}", &[]), "project.ecproj cannot be read");
    says(package(&m, "{}", &[]), "project.ecproj cannot be read");
    // numbers JSON cannot hold, and nesting past serde_json's limit
    says(package(&m, &project.replacen("320", "1e400", 1), &[]), "project.ecproj cannot be read");
    let deep = format!("{{\"a\": {}{}}}", "[".repeat(50_000), "]".repeat(50_000));
    says(package(&m, &deep, &[]), "project.ecproj cannot be read");
}

#[test]
fn truncated_and_bit_flipped_templates_never_panic() {
    let bytes = LowerThird::default().build();
    for n in (0..bytes.len()).step_by(97) {
        let r = std::panic::catch_unwind(|| crate::template_info(&bytes[..n]));
        assert!(r.is_ok_and(|r| r.is_err()), "a prefix of {n} bytes");
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut read = 0;
    for _ in 0..300 {
        let mut b = bytes.clone();
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let i = (seed % b.len() as u64) as usize;
        b[i] ^= 1 << (seed >> 61);
        let r = std::panic::catch_unwind(|| crate::template_info(&b));
        assert!(r.is_ok(), "a flipped bit at {i}");
        read += usize::from(r.is_ok_and(|r| r.is_ok()));
    }
    // (flips inside stored data that keep the CRC are rare: most are refused)
    assert!(read < 300);
}

#[test]
fn huge_comps_and_items_are_described_but_refused() {
    let huge = LowerThird { size: (30_000, 30_000), ..Default::default() }.build();
    let info = crate::template_info(&huge).unwrap();
    assert_eq!(info["renderable"], false);
    assert!(info["refused"].as_str().unwrap().contains("too large"), "{info:#}");
    let d = Dir::new("huge");
    let e = crate::prepare(&huge, &RenderRequest::default(), &d.0, &NoFrames).err().unwrap();
    assert!(matches!(e, Error::TooLarge(_)), "{e}");
    // a small render of it is fine…
    let small = crate::prepare(&huge, &RenderRequest { size: Some((300, 300)), ..Default::default() }, &d.0, &NoFrames).unwrap();
    assert_eq!(small.size(), (300, 300));
    // …but a huge solid inside a small comp is refused (it would be one float buffer that size)
    let solid = LowerThird::default().build_with(|p, _| {
        let sid = p.add_item(
            "Huge",
            Default::default(),
            None,
            ItemKind::Solid(effectcraft_project::Solid { color: [1.0; 3], width: 30_000, height: 30_000, pixel_aspect: 1.0 }),
        );
        let cid = p.comps().map(|(i, _)| *i).next().unwrap();
        let comp = p.comp(cid).unwrap().clone();
        let l = effectcraft_project::build::layer(p, &comp, "Huge", LayerSource::Solid { item: sid }, (30_000, 30_000), None);
        p.comp_mut(cid).unwrap().layers.push(l);
    });
    let e = crate::prepare(&solid, &RenderRequest::default(), &d.0, &NoFrames).err().unwrap();
    assert!(matches!(&e, Error::TooLarge(m) if m.contains("solid")), "{e}");
}

#[test]
fn deep_and_cyclic_nesting_is_bounded() {
    let deep = LowerThird::default().build_with(|p, _| {
        let top = p.comps().map(|(i, _)| *i).next().unwrap();
        let mut below = top;
        for i in 0..200 {
            let c = effectcraft_project::Comp::new(64, 64, effectcraft_time::FrameRate::FPS_30, effectcraft_time::Tick::from_seconds_f64(1.0));
            let id = p.add_item(&format!("Nest {i}"), Default::default(), None, ItemKind::Comp(c.clone().into()));
            let l = effectcraft_project::build::layer(p, &c, "Inner", LayerSource::Comp { item: below }, (64, 64), None);
            p.comp_mut(id).unwrap().layers.push(l);
            below = id;
        }
        // the template's comp nests the 200-deep chain
        let c = p.comp(top).unwrap().clone();
        let l = effectcraft_project::build::layer(p, &c, "Chain", LayerSource::Comp { item: below }, (64, 64), None);
        p.comp_mut(top).unwrap().layers.push(l);
    });
    let e = crate::template_info(&deep).unwrap_err();
    assert!(matches!(&e, Error::TooLarge(m) if m.contains("nests compositions")), "{e}");
    // a cycle (the comp inside itself) is not followed forever
    let cyclic = LowerThird::default().build_with(|p, _| {
        let top = p.comps().map(|(i, _)| *i).next().unwrap();
        let c = p.comp(top).unwrap().clone();
        let l = effectcraft_project::build::layer(p, &c, "Self", LayerSource::Comp { item: top }, (64, 64), None);
        p.comp_mut(top).unwrap().layers.push(l);
    });
    assert!(crate::template_info(&cyclic).is_ok());
}

fn prepare_err(values: Value) -> String {
    let d = Dir::new("values");
    let req = RenderRequest { values: values.as_object().unwrap().clone(), ..Default::default() };
    match crate::prepare(&LowerThird::default().build(), &req, &d.0, &NoFrames) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{values} was taken"),
    }
}

#[test]
fn values_are_checked_against_their_controls() {
    let d = Dir::new("ok");
    let req = RenderRequest {
        values: json!({"Name": "Grace Hopper", "bar color": "#ff8800", "Bar Opacity": 50}).as_object().unwrap().clone(),
        size: Some((160, 90)),
        fps: Some(FrameRate::FPS_24),
        range: (Some(Tick::from_seconds_f64(0.5)), None),
        gpu: false,
    };
    let p = crate::prepare(&LowerThird::default().build(), &req, &d.0, &NoFrames).unwrap();
    assert_eq!((p.size(), p.rate(), p.frames()), ((160, 90), FrameRate::FPS_24, 12));
    let s = p.summary();
    assert_eq!(s["values"].as_array().map(Vec::len), Some(3), "{s:#}");
    for (values, says) in [
        (json!({"Title": "x"}), "no control `Title`"),
        (json!({"Name": 7}), "give a string"),
        (json!({"Bar Color": "#zzzzzz"}), "colour"),
        (json!({"Bar Color": [0.1]}), "colour"),
        (json!({"Bar Opacity": "half"}), "a number"),
        (json!({"Bar Opacity": 1e300}), "finite"),
        (json!({"Bar Opacity": [1e9, 2]}), "finite"),
        (json!({"Name": {"text": "x", "size": 1e9}}), "text size"),
        (json!({"Name": "a", "name": "b"}), "given twice"),
    ] {
        let e = prepare_err(values.clone());
        assert!(e.contains(says) || (says == "colour" && e.contains("color")), "{values}: {e:?} should say {says:?}");
    }
}

#[test]
fn media_values_and_carried_media_go_through_the_host() {
    let d = Dir::new("media");
    let logo = Logo { png: png(16, 8, [255, 0, 0, 255]), width: 16, height: 8, embedded: true, path: String::new() };
    let t = LowerThird { logo: Some(logo), ..Default::default() }.build();
    let host = PngHost::default();
    let p = crate::prepare(&t, &RenderRequest::default(), &d.0, &host).unwrap();
    assert_eq!(p.summary()["media"], 1);
    // the carried logo was written under the media folder
    let written: Vec<_> = walk(&d.0);
    assert_eq!(written.len(), 1, "{written:?}");
    // a media value: an absolute path the host reads
    let other = d.0.join("other.png");
    std::fs::write(&other, png(4, 4, [0, 255, 0, 255])).unwrap();
    let req = RenderRequest { values: json!({"Logo": other.to_string_lossy()}).as_object().unwrap().clone(), ..Default::default() };
    assert!(crate::prepare(&t, &req, &d.0, &host).is_ok());
    for bad in [json!({"Logo": "relative.png"}), json!({"Logo": 3}), json!({"Logo": "/nowhere/at/all.png"})] {
        let req = RenderRequest { values: bad.as_object().unwrap().clone(), ..Default::default() };
        let e = crate::prepare(&t, &req, &d.0, &host).err().unwrap().to_string();
        assert!(e.contains("Logo"), "{bad}: {e}");
    }
    // a logo the template does not carry and the computer does not have
    let missing = Logo { png: Vec::new(), width: 16, height: 8, embedded: false, path: "/nowhere/logo.png".into() };
    let t = LowerThird { logo: Some(missing), ..Default::default() }.build();
    let e = crate::prepare(&t, &RenderRequest::default(), &d.0, &host).err().unwrap();
    assert!(matches!(&e, Error::Media(m) if m.contains("not in the template")), "{e}");
    // a host that panics is caught
    let logo = Logo { png: png(2, 2, [1, 2, 3, 4]), width: 2, height: 2, embedded: true, path: String::new() };
    let t = LowerThird { logo: Some(logo), ..Default::default() }.build();
    let e = crate::prepare(&t, &RenderRequest::default(), &d.0, &PanickyHost).err().unwrap();
    assert!(matches!(e, Error::Internal(_)), "{e}");
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.path().is_dir() {
            out.extend(walk(&e.path()));
        } else {
            out.push(e.path());
        }
    }
    out
}

#[test]
fn footage_is_read_once_converted_and_kept() {
    use effectcraft_render::FootageSource;
    let d = Dir::new("footage");
    let path = d.0.join("a.png");
    std::fs::write(&path, png(2, 2, [255, 255, 255, 128])).unwrap();
    let host = Arc::new(PngHost::default());
    let src = super::footage::HostFootage::new(host.clone());
    let f = effectcraft_project::Footage {
        path: path.to_string_lossy().into_owned(),
        kind: effectcraft_project::FootageKind::Still,
        width: 2,
        height: 2,
        has_video: true,
        ..Default::default()
    };
    let a = src.frame(effectcraft_project::ItemId(1), &f, effectcraft_time::Tick(0)).unwrap();
    let b = src.frame(effectcraft_project::ItemId(1), &f, effectcraft_time::Tick(1_000_000_000)).unwrap();
    assert!(Arc::ptr_eq(&a, &b), "a still is read once");
    assert_eq!(host.reads.lock().unwrap().len(), 1);
    // premultiplied, sRGB-encoded: white at half alpha is 0.5 in every channel
    assert!(a.data.iter().all(|p| p.iter().all(|c| (c - 128.0 / 255.0).abs() < 1e-6)), "{:?}", a.data);
    // a host that panics gives no frame, not a panic
    let bad = super::footage::HostFootage::new(Arc::new(PanickyHost));
    assert!(bad.frame(effectcraft_project::ItemId(1), &f, effectcraft_time::Tick(0)).is_none());
    // footage without pictures is never read
    let audio = effectcraft_project::Footage { kind: effectcraft_project::FootageKind::Audio, ..f.clone() };
    assert!(src.frame(effectcraft_project::ItemId(1), &audio, effectcraft_time::Tick(0)).is_none());
}

#[test]
fn craft_fonts_are_added_once() {
    let n = super::fonts::ensure();
    assert_eq!(n, super::fonts::ensure());
    if filmcraft_text::fonts::CRAFT_FONTS.is_empty() {
        assert_eq!(n, 0);
    }
}

#[test]
fn a_frame_renders_on_the_cpu() {
    let d = Dir::new("render");
    let req = RenderRequest { range: (None, Some(Tick::from_seconds_f64(1.0 / 30.0))), ..Default::default() };
    let p = crate::prepare(&LowerThird::default().build(), &req, &d.0, &NoFrames).unwrap();
    assert_eq!(p.frames(), 1);
    let mut got = Vec::new();
    let stats = p
        .render(
            Arc::new(NoFrames),
            &mut |i, f| {
                got.push((i, f.width, f.height, f.rgba.len()));
                Ok(())
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(got, [(0, 320, 180, 320 * 180 * 4)]);
    assert_eq!((stats.frames, stats.cpu_frames, stats.gpu_frames), (1, 1, 0));
    // a sink that refuses stops the render; so does cancelling
    let e = p.render(Arc::new(NoFrames), &mut |_, _| Err("disk full".into()), &AtomicBool::new(false)).unwrap_err();
    assert_eq!(e, Error::Render("disk full".into()));
    let e = p.render(Arc::new(NoFrames), &mut |_, _| Ok(()), &AtomicBool::new(true)).unwrap_err();
    assert_eq!(e, Error::Stopped);
}

// ---------------------------------------------------------------------------------------------
// Rendering (PX4.2)

/// Every frame of `t` rendered with `req` (into a folder of the test's own).
fn render_all(t: &[u8], req: RenderRequest, host: Arc<dyn HostFrames>) -> (Vec<crate::Frame>, crate::RenderStats) {
    let d = Dir::new("frames");
    let p = crate::prepare(t, &req, &d.0, host.as_ref()).unwrap();
    let mut frames = Vec::new();
    let stats = p
        .render(
            host,
            &mut |_, f| {
                frames.push(f.clone());
                Ok(())
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    (frames, stats)
}

/// The first frame of the default lower third with `values`.
fn first_frame(values: Value, gpu: bool) -> (crate::Frame, crate::RenderStats) {
    let req = RenderRequest { values: values.as_object().unwrap().clone(), range: (None, Some(Tick(1))), gpu, ..Default::default() };
    let (mut f, s) = render_all(&LowerThird::default().build(), req, Arc::new(NoFrames));
    (f.remove(0), s)
}

fn px(f: &crate::Frame, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * f.width + x) * 4) as usize;
    [f.rgba[i], f.rgba[i + 1], f.rgba[i + 2], f.rgba[i + 3]]
}

fn near(a: [u8; 4], b: [u8; 4], tol: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tol)
}

/// A point inside the bar, right of any name, and the region the name is drawn in.
const BAR_ONLY: (u32, u32) = (195, 150);
const NAME: ([u32; 2], [u32; 2]) = ([18, 120], [180, 168]);

fn region_differs(a: &crate::Frame, b: &crate::Frame, (lo, hi): ([u32; 2], [u32; 2])) -> usize {
    let mut n = 0;
    for y in lo[1]..hi[1] {
        for x in lo[0]..hi[0] {
            n += usize::from(!near(px(a, x, y), px(b, x, y), 2));
        }
    }
    n
}

#[test]
fn a_lower_third_renders_its_bar_and_its_name_over_transparency() {
    let (f, _) = first_frame(json!({}), false);
    assert_eq!((f.width, f.height), (320, 180));
    // the bar in its authored colour (sRGB levels as authored: 0.2, 0.4, 0.8), opaque
    assert!(near(px(&f, BAR_ONLY.0, BAR_ONLY.1), [51, 102, 204, 255], 1), "{:?}", px(&f, BAR_ONLY.0, BAR_ONLY.1));
    // around it nothing: transparent
    for (x, y) in [(2, 2), (319, 0), (300, 100), (5, 179)] {
        assert_eq!(px(&f, x, y)[3], 0, "({x}, {y})");
    }
    // the name in white on the bar: its strokes white, their antialiased edges between
    let (mut white, mut lighter) = (0, 0);
    for y in NAME.0[1]..NAME.1[1] {
        for x in NAME.0[0]..NAME.1[0] {
            let p = px(&f, x, y);
            white += usize::from(near(p, [255, 255, 255, 255], 8));
            lighter += usize::from(p[0] > 120 && p[3] == 255);
        }
    }
    assert!(white > 20 && lighter > 200, "{white} white and {lighter} lighter pixels of the name");
}

#[test]
fn values_change_the_pixels_where_they_should() {
    let (base, _) = first_frame(json!({}), false);
    let (red, _) = first_frame(json!({"Bar Color": "#ff0000"}), false);
    assert!(near(px(&red, BAR_ONLY.0, BAR_ONLY.1), [255, 0, 0, 255], 1), "{:?}", px(&red, BAR_ONLY.0, BAR_ONLY.1));
    assert_eq!(px(&red, 2, 2), px(&base, 2, 2));
    // another name changes the name's pixels and nothing outside the bar
    let (grace, _) = first_frame(json!({"Name": "Grace Hopper"}), false);
    assert!(region_differs(&base, &grace, NAME) > 50);
    assert_eq!(px(&grace, BAR_ONLY.0, BAR_ONLY.1), px(&base, BAR_ONLY.0, BAR_ONLY.1));
    assert_eq!(region_differs(&base, &grace, ([0, 0], [320, 100])), 0);
}

#[test]
fn alpha_is_kept_through_opacity_and_a_fade() {
    let (half, _) = first_frame(json!({"Bar Opacity": 50}), false);
    let p = px(&half, BAR_ONLY.0, BAR_ONLY.1);
    // straight alpha: the colour stays, the alpha halves
    assert!(near(p, [51, 102, 204, 128], 2), "{p:?}");
    // the fade: nothing at its start, the whole bar from half a second on
    let req = RenderRequest { range: (None, Some(Tick::from_seconds_f64(0.6))), ..Default::default() };
    let (frames, stats) = render_all(&LowerThird { fade_in: true, ..Default::default() }.build(), req, Arc::new(NoFrames));
    assert_eq!((frames.len(), stats.frames), (18, 18));
    let a: Vec<u8> = frames.iter().map(|f| px(f, BAR_ONLY.0, BAR_ONLY.1)[3]).collect();
    assert_eq!(a[0], 0);
    assert!(a.windows(2).take(15).all(|w| w[1] >= w[0]), "{a:?}");
    assert_eq!(a[15], 255, "{a:?}");
}

#[test]
fn a_frame_equals_effectcraft_s_own_cpu_render() {
    // EffectCraft's renderer on the project in memory, no package, no values, no conversion of ours
    let lt = LowerThird::default();
    let (project, comp) = lt.project();
    let reference = effectcraft_render::render_frame(&project, comp, effectcraft_time::Tick::ZERO, 1.0).to_rgba8();
    let (f, _) = first_frame(json!({}), false);
    let worst = f.rgba.iter().zip(&reference).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
    assert!(worst <= 1, "the packaged template renders the same as the project it was made of (worst {worst})");
}

#[test]
fn the_gpu_matches_the_cpu_within_a_level_or_two() {
    if effectcraft_gpu::Gpu::headless().is_none() {
        eprintln!("no GPU here: the GPU parity test is skipped");
        return;
    }
    let (cpu, _) = first_frame(json!({"Name": "Grace Hopper", "Bar Opacity": 75}), false);
    let (gpu, stats) = first_frame(json!({"Name": "Grace Hopper", "Bar Opacity": 75}), true);
    assert!(stats.gpu.is_some(), "{stats:?}");
    let diffs: Vec<u8> = cpu.rgba.iter().zip(&gpu.rgba).map(|(a, b)| a.abs_diff(*b)).collect();
    let worst = diffs.iter().copied().max().unwrap_or(0);
    let off = diffs.iter().filter(|d| **d > 2).count();
    eprintln!("GPU vs CPU: worst {worst}, {off} channels more than 2 apart; {stats:?}");
    assert_eq!(stats.gpu_frames + stats.cpu_frames, 1);
    assert!(off * 1000 <= diffs.len(), "GPU vs CPU: {off} of {} channels more than 2 levels apart (worst {worst}; {stats:?})", diffs.len());
}

#[test]
fn hostile_renders_end_cleanly() {
    // an endless expression is stopped by EffectCraft's own bounds: the frame still renders
    for e in ["while (true) {}", "throw new Error('no')", "(function f() { return f(); })()", "NaN", "1e400"] {
        let t = LowerThird { opacity_expression: Some(e.into()), ..Default::default() }.build();
        let req = RenderRequest { range: (None, Some(Tick(1))), ..Default::default() };
        let r = std::panic::catch_unwind(|| render_all(&t, req, Arc::new(NoFrames)));
        let (frames, _) = r.unwrap_or_else(|_| panic!("{e}: a panic"));
        assert_eq!(frames.len(), 1, "{e}");
    }
    // a logo that is gone by render time: drawn as nothing, no panic
    let d = Dir::new("gone");
    let path = d.0.join("logo.png");
    std::fs::write(&path, png(16, 8, [255, 0, 0, 255])).unwrap();
    let logo = Logo { png: Vec::new(), width: 16, height: 8, embedded: false, path: path.to_string_lossy().into_owned() };
    let t = LowerThird { logo: Some(logo), ..Default::default() }.build();
    let host: Arc<dyn HostFrames> = Arc::new(PngHost::default());
    let p = crate::prepare(&t, &RenderRequest { range: (None, Some(Tick(1))), ..Default::default() }, &d.0, host.as_ref()).unwrap();
    std::fs::remove_file(&path).unwrap();
    let mut n = 0;
    p.render(
        host,
        &mut |_, _| {
            n += 1;
            Ok(())
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(n, 1);
    // a host whose frames panic, after a probe that worked
    struct FramesPanic(PngHost);
    impl HostFrames for FramesPanic {
        fn probe(&self, path: &str) -> Option<HostMedia> {
            self.0.probe(path)
        }
        fn frame(&self, _: &str, _: Tick) -> Option<Arc<VideoFrame>> {
            panic!("decoder")
        }
    }
    let logo = Logo { png: png(16, 8, [255, 0, 0, 255]), width: 16, height: 8, embedded: true, path: String::new() };
    let t = LowerThird { logo: Some(logo), ..Default::default() }.build();
    let (frames, _) = render_all(&t, RenderRequest { range: (None, Some(Tick(1))), ..Default::default() }, Arc::new(FramesPanic(PngHost::default())));
    assert_eq!(frames.len(), 1);
}

#[test]
fn a_carried_logo_is_drawn_from_the_host_s_frames() {
    let logo = Logo { png: png(16, 8, [255, 0, 0, 255]), width: 16, height: 8, embedded: true, path: String::new() };
    let lt = LowerThird { logo: Some(logo.clone()), ..Default::default() };
    let host = Arc::new(PngHost::default());
    let req = RenderRequest { range: (None, Some(Tick::from_seconds_f64(0.1))), ..Default::default() };
    let (frames, _) = render_all(&lt.build(), req, host.clone());
    let [cx, cy] = crate::testing::logo_centre(lt.size, &logo);
    for f in &frames {
        assert!(near(px(f, cx as u32, cy as u32), [255, 0, 0, 255], 1), "{:?}", px(f, cx as u32, cy as u32));
    }
    // a still is read once for all three frames
    assert_eq!((frames.len(), host.reads.lock().unwrap().len()), (3, 1));
}

//! Eyes: the vision model that turns a screenshot into UI elements with coordinates.
//!
//! Qwen2.5-VL grounds in absolute pixels: `{"bbox_2d": [x1, y1, x2, y2], "label": …}` over
//! the image it actually saw. (Asking it for Qwen2-VL's normalized 0–1000 `[y1, x1, y2, x2]`
//! does not work: 2.5 ignores the instruction and answers in its own pixels, which is how a
//! live run once clicked the bottom-left of the window for a search field at the top.) Its
//! runtime snaps every image to 28-px patches and up-scales anything under a token floor, so
//! "the image it saw" is not the PNG we hold: [`model_frame`] computes that frame, we resize
//! to it ourselves before sending, and map the reply back to the caller's PNG pixels.
//!
//! The model runs in the Ollama server AutoCut already requires; nothing here links a model
//! runtime or ships weights. Every number the model emits is hostile until validated (§0.4).

use std::io::Read;

use serde_json::{Value, json};

use crate::{Config, ScreenError, b64, json_objects};

/// A grounded element: what it is and where, in the PNG's pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    pub label: String,
    pub kind: String,
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

impl Element {
    pub fn center(&self) -> (u32, u32) {
        (self.x1.saturating_add(self.x2) / 2, self.y1.saturating_add(self.y2) / 2)
    }
    pub fn area(&self) -> u64 {
        u64::from(self.x2.saturating_sub(self.x1)) * u64::from(self.y2.saturating_sub(self.y1))
    }
}

/// The grounding half of the agent loop; fakes implement this in tests.
pub trait Ground {
    /// List the visible elements `task` asks for (e.g. "Locate every video thumbnail…");
    /// empty when there are none.
    fn inventory(&self, png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError>;
    /// Locate one element by description; `None` when it is not visible.
    fn find(&self, png: &[u8], target: &str) -> Result<Option<Element>, ScreenError>;
}

/// How the model is set up to read text (not to ground).
const READ_SYSTEM: &str = "You read text in screenshots for an automation agent, exactly as it is shown: no guesses, no explanations.";

const SYSTEM: &str =
    "You ground UI screenshots for an automation agent. Coordinates are absolute pixels of the given image, origin top-left, as [x1, y1, x2, y2].";

/// The vision tower's patch: frames are multiples of it.
const PATCH: u32 = 28;
/// Ollama serves qwen2.5vl with `--image-min-tokens 1024`: smaller images are up-scaled.
const MIN_TOKENS: u64 = 1024;
/// Our ceiling: bigger frames cost seconds per call and made dense-grid answers degenerate
/// (one label stamped on every box) in live tests.
const MAX_TOKENS: u64 = 1600;
/// The most elements one reply may contribute (a screen of thumbnails is ~30).
const MAX_ELEMENTS: usize = 64;
/// Coordinates may overshoot the frame by this fraction (models round outward); beyond it
/// the element is nonsense and dropped.
const OVERSHOOT: f64 = 0.03;

/// The frame Qwen2.5-VL sees for a `w`×`h` image: both sides snapped to the 28-px patch,
/// scaled into [`MIN_TOKENS`, `MAX_TOKENS`] patches, aspect kept (Qwen's `smart_resize`).
/// Always at least one patch and at most 8192 px a side, whatever the input.
pub fn model_frame(w: u32, h: u32) -> (u32, u32) {
    let p = f64::from(PATCH);
    let (w, h) = (f64::from(w.max(1)), f64::from(h.max(1)));
    let snap = |v: f64| (v / p).round().max(1.0) * p;
    let (mut fw, mut fh) = (snap(w), snap(h));
    let (min_px, max_px) = (MIN_TOKENS as f64 * p * p, MAX_TOKENS as f64 * p * p);
    if fw * fh > max_px {
        let beta = (w * h / max_px).sqrt();
        fw = (w / beta / p).floor().max(1.0) * p;
        fh = (h / beta / p).floor().max(1.0) * p;
    } else if fw * fh < min_px {
        let beta = (min_px / (w * h)).sqrt();
        fw = (w * beta / p).ceil().max(1.0) * p;
        fh = (h * beta / p).ceil().max(1.0) * p;
    }
    let bound = |v: f64| if v.is_finite() { v.clamp(p, 8192.0) as u32 } else { PATCH };
    (bound(fw), bound(fh))
}

/// A PNG resized to the model's frame, remembering both sizes for mapping back.
pub(crate) struct Framed {
    pub(crate) png: Vec<u8>,
    src: (u32, u32),
    frame: (u32, u32),
}

pub(crate) fn to_model_frame(png: &[u8]) -> Result<Framed, ScreenError> {
    let img = image::load_from_memory(png).map_err(|e| ScreenError::Parse(format!("the screenshot is not a decodable image: {e}")))?;
    let src = (img.width(), img.height());
    if src.0 == 0 || src.1 == 0 {
        return Err(ScreenError::Parse("the screenshot has no area".into()));
    }
    let frame = model_frame(src.0, src.1);
    if frame == src {
        return Ok(Framed { png: png.to_vec(), src, frame });
    }
    let scaled = img.resize_exact(frame.0, frame.1, image::imageops::FilterType::Triangle);
    let mut out = std::io::Cursor::new(Vec::new());
    scaled.write_to(&mut out, image::ImageFormat::Png).map_err(|e| ScreenError::Parse(format!("cannot re-encode the screenshot: {e}")))?;
    Ok(Framed { png: out.into_inner(), src, frame })
}

/// The objects of a reply, with an `{"elements": [...]}` wrapper flattened.
fn reply_elements(reply: &str) -> Vec<Value> {
    let mut out = Vec::new();
    for v in json_objects(reply, MAX_ELEMENTS) {
        match v.get("elements").and_then(Value::as_array) {
            Some(els) => out.extend(els.iter().filter(|e| e.is_object()).cloned()),
            None => out.push(v),
        }
    }
    out.truncate(MAX_ELEMENTS);
    out
}

/// Decode `bbox_2d` (or `box_2d`) rectangles given in `frame` pixels into `src` pixels.
/// Anything hostile — wrong arity, non-numbers, far outside the frame, empty — drops that
/// one element, never the call.
fn decode(objs: &[Value], src: (u32, u32), frame: (u32, u32)) -> Vec<Element> {
    let (fw, fh) = (f64::from(frame.0.max(1)), f64::from(frame.1.max(1)));
    let (sw, sh) = (f64::from(src.0.max(1)), f64::from(src.1.max(1)));
    let mut out = Vec::new();
    for o in objs {
        let Some(b) = o.get("bbox_2d").or_else(|| o.get("box_2d")).and_then(Value::as_array) else { continue };
        let Some(c) = b.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>() else { continue };
        let [x1, y1, x2, y2] = c.as_slice() else { continue };
        let in_x = |v: f64| v.is_finite() && v >= -fw * OVERSHOOT && v <= fw * (1.0 + OVERSHOOT);
        let in_y = |v: f64| v.is_finite() && v >= -fh * OVERSHOOT && v <= fh * (1.0 + OVERSHOOT);
        if !(in_x(*x1) && in_x(*x2) && in_y(*y1) && in_y(*y2)) {
            continue;
        }
        let to_x = |v: f64| v.clamp(0.0, fw) * sw / fw;
        let to_y = |v: f64| v.clamp(0.0, fh) * sh / fh;
        let (x1, y1, x2, y2) = (
            (to_x(*x1).floor() as i64).clamp(0, i64::from(src.0.saturating_sub(1))) as u32,
            (to_y(*y1).floor() as i64).clamp(0, i64::from(src.1.saturating_sub(1))) as u32,
            (to_x(*x2).ceil() as i64).clamp(1, i64::from(src.0)) as u32,
            (to_y(*y2).ceil() as i64).clamp(1, i64::from(src.1)) as u32,
        );
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        let text = |k: &str, n: usize| o.get(k).and_then(Value::as_str).unwrap_or("").chars().take(n).collect::<String>();
        out.push(Element { label: text("label", 80), kind: text("type", 24), x1, y1, x2, y2 });
    }
    out
}

/// The Qwen2.5-VL client over Ollama's `/api/chat`.
pub struct QwenGrounder {
    host: String,
    model: String,
    agent: ureq::Agent,
}

impl QwenGrounder {
    pub fn from_config(cfg: &Config) -> Self {
        Self { host: cfg.ollama_host.clone(), model: cfg.model.clone(), agent: http_agent(cfg.http_timeout) }
    }

    /// `Ok(true)` when the configured model is present in the Ollama server.
    pub fn model_available(&self) -> Result<bool, ScreenError> {
        let resp = self.agent.get(format!("{}/api/tags", self.host)).call().map_err(|e| ScreenError::Model(e.to_string(), self.model.clone()))?;
        let mut body = String::new();
        resp.into_body().into_reader().take(1 << 20).read_to_string(&mut body).map_err(|e| ScreenError::Model(e.to_string(), self.model.clone()))?;
        let v: Value = serde_json::from_str(&body).map_err(|e| ScreenError::Model(e.to_string(), self.model.clone()))?;
        let want = self.model.split(':').next().unwrap_or(&self.model);
        Ok(v["models"].as_array().is_some_and(|ms| {
            ms.iter().any(|m| {
                m["name"].as_str().is_some_and(|n| {
                    let n = n.split(':').next().unwrap_or(n);
                    n == want
                })
            })
        }))
    }

    /// One grounding ask; the reply text, unparsed. No `format: json` — that constrains the
    /// model to one object, while it grounds best answering its trained shape (an array).
    fn chat(&self, user: &str, png: &[u8]) -> Result<String, ScreenError> {
        self.chat_as(SYSTEM, user, png)
    }

    /// What the model reads in `png` when asked `ask` (a transcription, a title): the reply
    /// text, at most 8 000 characters. The image goes in the model's own frame, as for grounding.
    pub fn read_text(&self, png: &[u8], ask: &str) -> Result<String, ScreenError> {
        let framed = to_model_frame(png)?;
        let reply = self.chat_as(READ_SYSTEM, ask, &framed.png)?;
        Ok(reply.chars().take(8000).collect())
    }

    fn chat_as(&self, system: &str, user: &str, png: &[u8]) -> Result<String, ScreenError> {
        let body = json!({
            "model": self.model,
            "stream": false,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [b64(png)]}
            ],
            "options": {"temperature": 0.1, "num_predict": 2048}
        });
        let resp = self.agent.post(format!("{}/api/chat", self.host)).send_json(body).map_err(|e| ScreenError::Model(e.to_string(), self.model.clone()))?;
        let mut text = String::new();
        resp.into_body().into_reader().take(1 << 24).read_to_string(&mut text).map_err(|e| ScreenError::Model(e.to_string(), self.model.clone()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| ScreenError::Parse(format!("Ollama replied with non-JSON: {e}")))?;
        v["message"]["content"].as_str().map(str::to_string).ok_or_else(|| ScreenError::Parse("the model reply had no content".into()))
    }
}

impl Ground for QwenGrounder {
    fn inventory(&self, png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
        let framed = to_model_frame(png)?;
        let user = format!(
            "{task} Output a JSON array with one object per element: [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"...\"}}]. One box per element; never group several elements into one box; never invent elements that are not visible."
        );
        let reply = self.chat(&user, &framed.png)?;
        Ok(decode(&reply_elements(&reply), framed.src, framed.frame))
    }

    fn find(&self, png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
        let framed = to_model_frame(png)?;
        let user = format!(
            "Locate {target}. Output its bounding box as JSON: {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"...\"}}. If it is not visible, output {{\"bbox_2d\": null}}."
        );
        let reply = self.chat(&user, &framed.png)?;
        Ok(decode(&reply_elements(&reply), framed.src, framed.frame).into_iter().next())
    }
}

/// The shared HTTP agent: pure-Rust TLS (rustls + platform verifier) so http://localhost
/// and https:// hosts both work — configured as the speech downloader and the gphotos API
/// client configure theirs.
fn http_agent(global_timeout: std::time::Duration) -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_global(Some(global_timeout))
        .build()
        .into()
}

/// Decode an inventory reply for `png` (boxes in the model frame of `png`'s size) — public
/// for tests and for replaying logged replies. Prose ("There are none.") is an empty list:
/// that is how the model says nothing matches.
pub fn parse_inventory_reply(reply: &str, png: &[u8]) -> Result<Vec<Element>, ScreenError> {
    let src = png_dims(png)?;
    Ok(decode(&reply_elements(reply), src, model_frame(src.0, src.1)))
}

/// Decode a `find` reply into one element; `None` for `{"bbox_2d": null}` or prose.
pub fn parse_find_reply(reply: &str, png: &[u8]) -> Result<Option<Element>, ScreenError> {
    Ok(parse_inventory_reply(reply, png)?.into_iter().next())
}

fn png_dims(png: &[u8]) -> Result<(u32, u32), ScreenError> {
    let img = image::load_from_memory(png).map_err(|e| ScreenError::Parse(format!("the screenshot is not a decodable image: {e}")))?;
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err(ScreenError::Parse("the screenshot has no area".into()));
    }
    Ok((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::GrayImage::from_pixel(w, h, image::Luma([100u8]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn model_frame_snaps_to_patches_within_token_bounds() {
        // the picker capture: 1344x773 → height snapped to 784 (28 x 28), width already 48 patches
        assert_eq!(model_frame(1344, 773), (1344, 784));
        // an exact frame is kept
        assert_eq!(model_frame(1344, 784), (1344, 784));
        // too small: up-scaled to at least MIN_TOKENS patches, aspect ~kept
        let (w, h) = model_frame(800, 600);
        assert!(w % PATCH == 0 && h % PATCH == 0);
        assert!(u64::from(w / PATCH) * u64::from(h / PATCH) >= MIN_TOKENS, "{w}x{h}");
        assert!(((f64::from(w) / f64::from(h)) - 800.0 / 600.0).abs() < 0.06);
        // too big: down-scaled to at most MAX_TOKENS patches
        let (w, h) = model_frame(3456, 1988);
        assert!(u64::from(w / PATCH) * u64::from(h / PATCH) <= MAX_TOKENS, "{w}x{h}");
        // hostile sizes stay bounded and non-empty
        for (a, b) in [(0, 0), (1, 1), (1, u32::MAX), (u32::MAX, u32::MAX), (u32::MAX, 1)] {
            let (w, h) = model_frame(a, b);
            assert!((PATCH..=8192).contains(&w) && (PATCH..=8192).contains(&h), "{a}x{b} → {w}x{h}");
        }
    }

    #[test]
    fn inventory_reads_native_pixel_boxes() {
        // a PNG that is its own model frame: reply pixels are PNG pixels
        let png = test_png(1344, 784);
        let reply = "```json\n[\n {\"bbox_2d\": [60, 405, 171, 444], \"label\": \"0:06\"},\n {\"bbox_2d\": [1275, 83, 1329, 101], \"label\": \"Done\"}\n]\n```";
        let els = parse_inventory_reply(reply, &png).unwrap();
        assert_eq!(els.len(), 2);
        assert_eq!((els[0].x1, els[0].y1, els[0].x2, els[0].y2), (60, 405, 171, 444));
        assert_eq!(els[0].label, "0:06");
        assert_eq!(els[1].center(), (1302, 92));
        // the legacy key and the {"elements": [...]} wrapper read the same
        let wrapped = r#"{"elements":[{"label":"0:06","type":"thumbnail","box_2d":[60,405,171,444]}]}"#;
        let w = parse_inventory_reply(wrapped, &png).unwrap();
        assert_eq!((w[0].x1, w[0].y1, w[0].x2, w[0].y2, w[0].kind.as_str()), (60, 405, 171, 444, "thumbnail"));
    }

    #[test]
    fn boxes_map_from_the_model_frame_back_to_png_pixels() {
        // 1344x773 is sent as 1344x784: y is scaled by 773/784, x untouched
        let png = test_png(1344, 773);
        let els = parse_inventory_reply(r#"[{"bbox_2d": [100, 392, 200, 784]}]"#, &png).unwrap();
        assert_eq!((els[0].x1, els[0].x2), (100, 200));
        assert_eq!(els[0].y1, (392.0f64 * 773.0 / 784.0).floor() as u32);
        assert_eq!(els[0].y2, 773);
    }

    #[test]
    fn hostile_replies_drop_elements_not_the_call() {
        let png = test_png(1344, 784);
        let bad = r#"[
            {"label":"a","bbox_2d":[0,0,10,10]},
            {"label":"b","bbox_2d":[10,10,5,5]},
            {"label":"c","bbox_2d":"nope"},
            {"label":"d","bbox_2d":[1,2,3]},
            {"label":"e","bbox_2d":[null,0,10,10]},
            {"label":"f","bbox_2d":[-400,0,10,10]},
            {"label":"g","bbox_2d":[0,0,99999,10]},
            {"label":"h"},
            {"label":"i","bbox_2d":[1e308,0,10,10]}
        ]"#;
        let els = parse_inventory_reply(bad, &png).unwrap();
        // only 'a' survives: every other entry is hostile, and none may panic
        assert_eq!(els.len(), 1, "{els:?}");
        assert_eq!(els[0].label, "a");
        // a slight overshoot (models round outward) is clamped, not dropped
        let over = parse_inventory_reply(r#"[{"bbox_2d":[-5,0,1350,790]}]"#, &png).unwrap();
        assert_eq!((over[0].x1, over[0].y1, over[0].x2, over[0].y2), (0, 0, 1344, 784));
        // prose means "nothing there", not an error
        assert!(parse_inventory_reply("There are none.", &png).unwrap().is_empty());
        // a flood of elements is capped
        let flood = format!("[{}]", vec![r#"{"bbox_2d":[0,0,10,10]}"#; 500].join(","));
        assert!(parse_inventory_reply(&flood, &png).unwrap().len() <= MAX_ELEMENTS);
    }

    #[test]
    fn find_reply_null_or_prose_means_absent() {
        let png = test_png(1344, 784);
        assert!(parse_find_reply(r#"{"bbox_2d":null}"#, &png).unwrap().is_none());
        assert!(parse_find_reply("There are none.", &png).unwrap().is_none());
        let e = parse_find_reply(r#"```json\n[{"bbox_2d": [382, 307, 962, 345], "label": "search field"}]\n```"#, &png).unwrap().unwrap();
        assert_eq!((e.x1, e.y1, e.x2, e.y2), (382, 307, 962, 345));
        assert!(parse_find_reply(r#"{"bbox_2d":[1,2]}"#, &png).unwrap().is_none());
    }

    #[test]
    fn garbage_png_is_an_error_not_a_panic() {
        assert!(parse_inventory_reply("[]", b"not a png").is_err());
        assert!(to_model_frame(b"not a png").is_err());
    }

    #[test]
    fn framing_resizes_only_when_needed() {
        let exact = test_png(1344, 784);
        let f = to_model_frame(&exact).unwrap();
        assert_eq!((f.src, f.frame), ((1344, 784), (1344, 784)));
        assert_eq!(f.png, exact); // sent as-is
        let f = to_model_frame(&test_png(1344, 773)).unwrap();
        assert_eq!((f.src, f.frame), ((1344, 773), (1344, 784)));
        assert_eq!(png_dims(&f.png).unwrap(), (1344, 784));
    }

    /// A real round-trip through Ollama, gated: AUTOCUT_SCREEN_LIVE=1 cargo test -p
    /// filmcraft-screenagent. Draws a synthetic UI (one blue button on gray) and asks the
    /// configured grounding model to find it; asserts the box overlaps the drawn button.
    #[test]
    fn live_ground_finds_drawn_button() {
        if std::env::var("AUTOCUT_SCREEN_LIVE").ok().as_deref() != Some("1") {
            eprintln!("skipped (AUTOCUT_SCREEN_LIVE != 1)");
            return;
        }
        let mut img = image::RgbImage::from_pixel(800, 600, image::Rgb([200u8, 200, 200]));
        for y in 300..360u32 {
            for x in 320..480u32 {
                img.get_pixel_mut(x, y).0 = [30, 100, 235];
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let png = png.into_inner();
        let cfg = crate::Config::from_env();
        let g = QwenGrounder::from_config(&cfg);
        if !g.model_available().unwrap_or(false) {
            eprintln!("skipped ({} not in Ollama)", cfg.model);
            return;
        }
        let found = g.find(&png, "the blue button").unwrap().expect("the model should find the blue button");
        // overlap with the drawn rectangle, not just anywhere
        let overlap_x = found.x2.min(480).saturating_sub(found.x1.max(320));
        let overlap_y = found.y2.min(360).saturating_sub(found.y1.max(300));
        assert!(overlap_x > 40 && overlap_y > 20, "grounded box {:?}", (found.x1, found.y1, found.x2, found.y2));
    }

    /// Replays a saved window capture through the real model, the agent's way (downscaled
    /// like a live capture, video-tile ask, candidate ranking): AUTOCUT_SCREEN_LIVE_PNG=
    /// /path/to/capture.png cargo test -p filmcraft-screenagent live_inventory -- --nocapture.
    /// Prints what the agent would click; asserts at least one video tile was found.
    #[test]
    fn live_inventory_from_png() {
        let Some(path) = std::env::var_os("AUTOCUT_SCREEN_LIVE_PNG") else {
            eprintln!("skipped (AUTOCUT_SCREEN_LIVE_PNG unset)");
            return;
        };
        let raw = std::fs::read(path).unwrap();
        let (w, h) = png_dims(&raw).unwrap();
        let cap = crate::capture::decode_and_scale(raw, crate::capture::WindowRect { x: 0, y: 0, w: w / 2, h: h / 2 }).unwrap();
        let g = QwenGrounder::from_config(&crate::Config::from_env());
        // AUTOCUT_SCREEN_LIVE_TASK=day_headings: list the date headings it grounds instead
        let task = match std::env::var("AUTOCUT_SCREEN_LIVE_TASK").as_deref() {
            Ok("day_headings") => Some(crate::picker::DAY_HEADINGS),
            _ => None,
        };
        if let Some(task) = task {
            for e in g.inventory(&cap.png, task).unwrap() {
                eprintln!("  {:?} kind {:?} box ({},{})-({},{}) key {:?}", e.label, e.kind, e.x1, e.y1, e.x2, e.y2, crate::picker::date_key(&e.label));
            }
            return;
        }
        let els = g.inventory(&cap.png, crate::picker::VIDEO_TILES).unwrap();
        eprintln!("{} element(s); failed enumeration: {}", els.len(), crate::picker::enumeration_failed(&els, cap.png_w, cap.png_h));
        let tiles = crate::picker::tile_candidates(&els, true, cap.png_w, cap.png_h);
        for t in &tiles {
            eprintln!("  {:>6} box ({},{})-({},{}) click {:?}", t.label, t.x1, t.y1, t.x2, t.y2, crate::picker::click_point(t));
        }
        assert!(!tiles.is_empty(), "no video tiles grounded");
    }
}

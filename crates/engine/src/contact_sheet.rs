//! Contact sheets of source frames: what is in these clips, at a glance, in one picture.
//!
//! | command | does |
//! |---|---|
//! | `media.contactSheet` | frames of media items in one labelled picture (JPEG or PNG at `path`, else `pngBase64`): per item, `n` frames spread evenly over its length (or `from`..`to`), or the exact `at` seconds |
//!
//! Each item starts a row (wrapped every `columns` tiles). A tile is the source frame as decoded
//! ([`crate::media_frame`]: display rotation applied, before any clip effects), its long side
//! `side` pixels — so a portrait clip gets tall tiles and a landscape one wide tiles, neither
//! letterboxed — or, with `scale`, that share of the frame's own size. Each tile is labelled
//! with its item and time (`#2407 12.5s`; `10:01.0` from a minute on, or the time alone with
//! `labels: "time"`) in a 5 × 7 pixel font drawn in code. A frame that cannot be decoded is a
//! dark red tile labelled `?`; only a sheet with no frame at all is an error.
//!
//! Sheets are capped at [`MAX_TILES`] tiles and about [`MAX_PIXELS`] pixels (tiles shrink to
//! fit), so a hostile `n` or `side` costs a bounded amount of memory.

use serde_json::{Value, json};

use filmcraft_project::ItemId;

use crate::commands::{CommandSpec, always, bad, f64_p, str_p, u64_p};
use crate::media_frame::{MAX_SIDE, decode, video_length};
use crate::{Result, Session};

const CMD: &str = "media.contactSheet";
/// Most items, tiles and `at` times a sheet takes.
const MAX_ITEMS: usize = 200;
pub const MAX_TILES: usize = 600;
/// Most pixels the tiles of a sheet take together (they shrink to fit).
pub const MAX_PIXELS: f64 = 48_000_000.0;
/// Space between tiles, pixels.
const GAP: u32 = 4;
const BACKGROUND: [u8; 4] = [17, 17, 17, 255];
const MISSING: [u8; 4] = [70, 12, 12, 255];

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Contact Sheet",
        menu: &[],
        shortcut: None,
        params: r#"{"items":[{"item":id,"from":f64?,"to":f64?,"at":[f64]?}],"n":u32=8,"side":u32=320,"scale":f64?,"columns":u32?,"labels":"item|time"="item","path":str?,"quality":u8=85}"#,
        enabled: always,
        run: sheet,
        journal: false,
    }]
}

/// One tile: its frame (straight RGBA8) and label.
struct Tile {
    rgba: Vec<u8>,
    w: u32,
    h: u32,
    label: String,
    ok: bool,
}

/// `n` times spread evenly over `[a, b]`: the middles of `n` equal parts.
pub fn spread(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 || !(a.is_finite() && b.is_finite()) || b < a {
        return Vec::new();
    }
    (0..n).map(|k| a + (b - a) * (k as f64 + 0.5) / n as f64).collect()
}

/// "12.5s", or "10:01.0" from a minute on.
pub fn stamp(seconds: f64) -> String {
    let s = if seconds.is_finite() { seconds.max(0.0) } else { 0.0 };
    if s < 60.0 {
        return format!("{s:.1}s");
    }
    let m = (s / 60.0).floor();
    format!("{}:{:04.1}", m as u64, s - m * 60.0)
}

fn sheet(s: &mut Session, p: &Value) -> Result<Value> {
    let items = p.get("items").and_then(Value::as_array).filter(|a| !a.is_empty()).ok_or_else(|| bad(CMD, "`items` is a list of {item, from?, to?, at?}"))?;
    if items.len() > MAX_ITEMS {
        return Err(bad(CMD, format!("at most {MAX_ITEMS} items")));
    }
    let n = u64_p(p, "n").unwrap_or(8).clamp(1, 64) as usize;
    let scale = match f64_p(p, "scale") {
        None => None,
        Some(x) if x.is_finite() && (0.01..=1.0).contains(&x) => Some(x),
        Some(_) => return Err(bad(CMD, "`scale` is a share of the frame size, 0.01 to 1")),
    };
    let side = u64_p(p, "side").unwrap_or(320).clamp(32, u64::from(MAX_SIDE)) as u32;
    let only_time = match str_p(p, "labels") {
        None | Some("item") => false,
        Some("time") => true,
        Some(_) => return Err(bad(CMD, "`labels` is item or time")),
    };
    // what to decode: (item, times) per row, checked before anything is decoded
    let mut rows: Vec<(ItemId, Vec<f64>)> = Vec::with_capacity(items.len());
    for (i, it) in items.iter().enumerate() {
        let id = match it {
            Value::Number(_) => it.as_u64(),
            _ => it.get("item").and_then(Value::as_u64),
        }
        .map(ItemId)
        .ok_or_else(|| bad(CMD, format!("items[{i}]: an item id")))?;
        let len = video_length(s, CMD, id)?.seconds();
        let times = match it.get("at").filter(|v| !v.is_null()) {
            Some(Value::Array(a)) if !a.is_empty() && a.len() <= MAX_TILES => a
                .iter()
                .map(|t| t.as_f64().filter(|t| t.is_finite() && *t >= 0.0))
                .collect::<Option<Vec<f64>>>()
                .ok_or_else(|| bad(CMD, format!("items[{i}].at: seconds, zero or more")))?,
            Some(_) => return Err(bad(CMD, format!("items[{i}].at: a list of 1 to {MAX_TILES} times"))),
            None => {
                let num = |k: &str| it.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
                let (a, b) = (num("from").unwrap_or(0.0).max(0.0), num("to").unwrap_or(len).min(len));
                if b <= a {
                    return Err(bad(CMD, format!("items[{i}]: from {a} to {b} is empty (the clip is {len:.1}s)")));
                }
                spread(a, b, n)
            }
        };
        rows.push((id, times));
    }
    let tiles: usize = rows.iter().map(|r| r.1.len()).sum();
    if tiles > MAX_TILES {
        return Err(bad(CMD, format!("{tiles} frames is more than {MAX_TILES}; ask for fewer")));
    }
    let per = u64_p(p, "columns").map_or(n, |c| c.clamp(1, 64) as usize).max(1);
    // each row's tile shape (the item's own) and long side, then all of them shrunk so the
    // sheet as laid out stays within its pixel budget
    let mut plan: Vec<(f64, f64)> = rows
        .iter()
        .map(|(id, _)| {
            let (w, h) = s.project.source_size(*id).filter(|(w, h)| *w > 0 && *h > 0).unwrap_or((1920, 1080));
            let want = match scale {
                Some(k) => (f64::from(w.max(h)) * k).round(),
                None => f64::from(side),
            };
            (want.clamp(16.0, f64::from(MAX_SIDE)), f64::from(w) / f64::from(h))
        })
        .collect();
    let dims = |want: f64, aspect: f64| if aspect >= 1.0 { (want, want / aspect) } else { (want * aspect, want) };
    let (mut sheet_w, mut sheet_h) = (0.0f64, 0.0f64);
    for ((_, times), (want, aspect)) in rows.iter().zip(&plan) {
        let (w, h) = dims(*want, *aspect);
        let full = times.len().div_ceil(per);
        sheet_w = sheet_w.max(times.len().min(per) as f64 * (w + f64::from(GAP)));
        sheet_h += full as f64 * (h + f64::from(GAP));
    }
    let area = sheet_w * sheet_h;
    if area > MAX_PIXELS {
        let f = (MAX_PIXELS / area).sqrt();
        for (want, _) in &mut plan {
            *want = (*want * f).max(16.0);
        }
    }
    let mut made: Vec<Vec<Tile>> = Vec::with_capacity(rows.len());
    let mut report = Vec::with_capacity(tiles);
    for ((id, times), (want, aspect)) in rows.iter().zip(&plan) {
        let want = want.round() as u32;
        let mut row = Vec::with_capacity(times.len());
        for t in times {
            let label = if only_time { stamp(*t) } else { format!("#{} {}", id.0, stamp(*t)) };
            match decode(s, CMD, *id, Some(*t), want) {
                Ok(d) => {
                    report.push(json!({"item": id.0, "seconds": d.seconds, "ok": true}));
                    row.push(Tile { rgba: d.rgba, w: d.width, h: d.height, label, ok: true });
                }
                Err(e) => {
                    report.push(json!({"item": id.0, "seconds": t, "ok": false, "error": e.to_string()}));
                    let (w, h) = dims(f64::from(want), *aspect);
                    row.push(Tile { rgba: Vec::new(), w: (w.round() as u32).max(16), h: (h.round() as u32).max(16), label: format!("{label} ?"), ok: false });
                }
            }
        }
        let mut row = row.into_iter().peekable();
        while row.peek().is_some() {
            made.push(row.by_ref().take(per).collect());
        }
    }
    if !made.iter().flatten().any(|t| t.ok) {
        let why = report.iter().find_map(|r| r["error"].as_str()).unwrap_or("no frame decoded").to_string();
        return Err(bad(CMD, format!("no frame could be decoded: {why}")));
    }
    let (rgba, w, h) = compose(&made).ok_or_else(|| bad(CMD, "the sheet could not be drawn"))?;
    let path = str_p(p, "path").map(str::trim).filter(|p| !p.is_empty());
    let jpeg = path.is_some_and(|p| {
        let l = p.to_ascii_lowercase();
        l.ends_with(".jpg") || l.ends_with(".jpeg")
    });
    let quality = u64_p(p, "quality").unwrap_or(85).clamp(1, 100) as u8;
    let bytes = if jpeg {
        filmcraft_export::encode_still_with_quality(filmcraft_export::Format::JpegSequence, rgba, w, h, false, quality)
    } else {
        filmcraft_export::encode_png(rgba, w, h)
    }
    .map_err(|e| bad(CMD, e.to_string()))?;
    let mut out = json!({"width": w, "height": h, "tiles": report, "missing": made.iter().flatten().filter(|t| !t.ok).count()});
    match path {
        Some(path) => {
            let path = crate::export_tools::expand_home(path);
            s.services.write_file(&path, &bytes).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
            out["path"] = json!(path);
        }
        None => out["pngBase64"] = json!(filmcraft_project::gtemplate::base64_encode(&bytes)),
    }
    Ok(out)
}

/// The rows of tiles in one RGBA picture: rows top to bottom, each as tall as its tallest tile.
fn compose(rows: &[Vec<Tile>]) -> Option<(Vec<u8>, u32, u32)> {
    let width = rows.iter().map(|r| r.iter().map(|t| t.w + GAP).sum::<u32>() + GAP).max()?.max(1);
    let height = rows.iter().map(|r| r.iter().map(|t| t.h).max().unwrap_or(0) + GAP).sum::<u32>() + GAP;
    let (wu, hu) = (width as usize, height as usize);
    if (wu as f64) * (hu as f64) > MAX_PIXELS * 1.5 {
        return None;
    }
    let mut img = Canvas { px: BACKGROUND.repeat(wu.checked_mul(hu)?), w: wu, h: hu };
    let mut y = GAP;
    for row in rows {
        let row_h = row.iter().map(|t| t.h).max().unwrap_or(0);
        let mut x = GAP;
        for t in row {
            let top = y + (row_h - t.h) / 2;
            if t.ok {
                img.blit(&t.rgba, t.w, t.h, x, top);
            } else {
                img.fill(x, top, t.w, t.h, MISSING);
            }
            // the label, white on a black band at the top left
            let k = if t.h < 60 { 1 } else { (t.h / 100).clamp(2, 4) };
            let text_w = (t.label.chars().count() as u32 * 6 * k + 3 * k).min(t.w);
            img.fill(x, top, text_w, 9 * k, [0, 0, 0, 255]);
            img.text(&t.label, x + 2 * k, top + k, k, [255, 255, 255, 255], x + t.w);
            x += t.w + GAP;
        }
        y += row_h + GAP;
    }
    Some((img.px, width, height))
}

struct Canvas {
    px: Vec<u8>,
    w: usize,
    h: usize,
}

impl Canvas {
    fn put(&mut self, x: usize, y: usize, c: [u8; 4]) {
        if x < self.w && y < self.h {
            let i = (y * self.w + x) * 4;
            if let Some(p) = self.px.get_mut(i..i + 4) {
                p.copy_from_slice(&c);
            }
        }
    }

    fn fill(&mut self, x: u32, y: u32, w: u32, h: u32, c: [u8; 4]) {
        for yy in y..y.saturating_add(h) {
            for xx in x..x.saturating_add(w) {
                self.put(xx as usize, yy as usize, c);
            }
        }
    }

    fn blit(&mut self, rgba: &[u8], w: u32, h: u32, x: u32, y: u32) {
        let (wu, hu) = (w as usize, h as usize);
        for row in 0..hu {
            for col in 0..wu {
                if let Some(p) = rgba.get((row * wu + col) * 4..(row * wu + col) * 4 + 4) {
                    self.put(x as usize + col, y as usize + row, [p[0], p[1], p[2], 255]);
                }
            }
        }
    }

    /// `text` in the 5 × 7 font, each pixel `k` × `k`, from `(x, y)`, clipped at `right`.
    fn text(&mut self, text: &str, x: u32, y: u32, k: u32, c: [u8; 4], right: u32) {
        let mut cx = x;
        for ch in text.chars() {
            if cx + 5 * k > right {
                break;
            }
            for (row, bits) in glyph(ch).iter().enumerate() {
                for col in 0..5u32 {
                    if bits & (0x10 >> col) != 0 {
                        self.fill(cx + col * k, y + row as u32 * k, k, k, c);
                    }
                }
            }
            cx += 6 * k;
        }
    }
}

/// A 5 × 7 glyph (one row per byte, bit 4 the left column) for the characters labels use;
/// drawn for FilmCraft (original work). Anything else is a blank.
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '.' => [0, 0, 0, 0, 0, 0x0C, 0x0C],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        's' => [0, 0, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
        '#' => [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A],
        '?' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04],
        _ => [0; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    fn png_size(v: &Value) -> (u32, u32) {
        let png = filmcraft_project::gtemplate::base64_decode(v["pngBase64"].as_str().unwrap()).unwrap();
        let img = image::load_from_memory(&png).unwrap();
        (img.width(), img.height())
    }

    #[test]
    fn rows_of_labelled_frames() {
        let mut s = demo();
        let r = s.execute(CMD, json!({"items": [{"item": 5, "from": 0.5, "to": 2.5}, {"item": 5, "at": [1.0, 2.0]}], "n": 4, "side": 160})).unwrap();
        let tiles = r["tiles"].as_array().unwrap();
        assert_eq!(tiles.len(), 6);
        assert!(tiles.iter().all(|t| t["ok"] == json!(true)), "{r}");
        // evenly spread: the middles of four quarters of 0.5..2.5
        let first: Vec<f64> = tiles.iter().take(4).map(|t| t["seconds"].as_f64().unwrap()).collect();
        assert!((first[0] - 0.75).abs() < 0.05 && (first[3] - 2.25).abs() < 0.05, "{first:?}");
        // two rows, the first four tiles wide
        let (w, h) = png_size(&r);
        assert_eq!((w, h), (r["width"].as_u64().unwrap() as u32, r["height"].as_u64().unwrap() as u32));
        assert!(w >= 4 * 100 && h >= 2 * 50, "{w}x{h}");
        // the label band: black at the top left of the first tile, white text in it
        let png = filmcraft_project::gtemplate::base64_decode(r["pngBase64"].as_str().unwrap()).unwrap();
        let img = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(GAP, GAP).0, [0, 0, 0, 255]);
        assert!((GAP..GAP + 60).any(|x| (GAP..GAP + 9).any(|y| img.get_pixel(x, y).0 == [255, 255, 255, 255])));
    }

    #[test]
    fn scale_columns_and_files() {
        let mut s = demo();
        let dir = std::env::temp_dir().join(format!("fc-sheet-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.jpg");
        let r = s
            .execute(CMD, json!({"items": [{"item": 5, "at": [0.5, 1.0, 1.5]}], "scale": 0.25, "columns": 2, "labels": "time", "path": path.to_string_lossy()}))
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(&[0xFF, 0xD8]), "a JPEG");
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((u64::from(img.width()), u64::from(img.height())), (r["width"].as_u64().unwrap(), r["height"].as_u64().unwrap()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_requests_are_errors_not_crashes() {
        let mut s = demo();
        for p in [
            json!({}),
            json!({"items": []}),
            json!({"items": [{"item": 999}]}),
            json!({"items": [{"item": 11}]}),
            json!({"items": [{"item": 5, "at": [-1]}]}),
            json!({"items": [{"item": 5, "at": "x"}]}),
            json!({"items": [{"item": 5, "from": 9e9}]}),
            json!({"items": [{"item": 5}], "scale": 0.0}),
            json!({"items": [{"item": 5}], "labels": "names"}),
            json!({"items": [{"item": 5, "at": vec![0.5; MAX_TILES + 1]}]}),
            json!({"items": vec![json!({"item": 5}); MAX_ITEMS + 1]}),
        ] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        // huge asks are cut to the caps, not allocated
        let r = s.execute(CMD, json!({"items": [5, 5, 5], "n": 1_000_000, "side": 1_000_000, "columns": 0})).unwrap();
        assert_eq!(r["tiles"].as_array().unwrap().len(), 3 * 64);
        // a long row over many short ones: still within the budget
        let mut items = vec![json!({"item": 5})];
        items.extend(std::iter::repeat_n(json!({"item": 5, "at": [1.0]}), 100));
        let r = s.execute(CMD, json!({"items": items, "n": 64, "side": 4096})).unwrap();
        let (w, h) = png_size(&r);
        assert!(f64::from(w) * f64::from(h) <= MAX_PIXELS * 1.5, "{w}x{h}");
        let (w, h) = png_size(&r);
        assert!(f64::from(w) * f64::from(h) <= MAX_PIXELS * 1.5, "{w}x{h}");
    }

    #[test]
    fn labels_and_spreads() {
        assert_eq!(stamp(12.46), "12.5s");
        assert_eq!(stamp(601.0), "10:01.0");
        assert_eq!(stamp(f64::NAN), "0.0s");
        assert_eq!(spread(0.0, 8.0, 4), vec![1.0, 3.0, 5.0, 7.0]);
        assert!(spread(5.0, 1.0, 3).is_empty() && spread(0.0, 1.0, 0).is_empty() && spread(f64::NAN, 1.0, 2).is_empty());
        assert_eq!(glyph('Z'), [0; 7]);
    }
}

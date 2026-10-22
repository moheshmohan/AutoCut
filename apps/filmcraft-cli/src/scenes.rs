//! `scenes …`: the scene index of media items ([`filmcraft_engine::scenes`]) — analyse clips
//! (shots, quality, camera moves, descriptions, embeddings), show an item's shots, search them in
//! words, pick the best for a purpose, and fetch the models.

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

const USAGE: &str = "\
scenes analyze [media files…] [--items 1,2] [--detail fast|full] [--fresh] [--no-describe] [--no-embed]
               [--no-aesthetic] [--download] [--sheet DIR] [--project p.fcproj --save]
scenes show <item> [--embeddings]
scenes search \"<query>\" [--limit 20] [--items 1,2]
scenes best [--purpose hook|broll|reveal|talking-head|establishing] [--count 5] [--items 1,2]
scenes sheet DIR [--items 1,2]
scenes status
scenes download [fast|full|<model id>]";

fn items(a: &Args) -> Option<Vec<u64>> {
    a.opt("--items").map(|v| {
        v.split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u64>().unwrap_or_else(|_| usage(format!("scenes: --items wants item ids, not {s}"))))
            .collect()
    })
}

fn count(a: &Args, opt: &str, default: u64) -> u64 {
    a.opt(opt).map_or(default, |v| v.parse::<u64>().unwrap_or_else(|_| usage(format!("scenes: {opt} wants a number, not {v}"))))
}

pub async fn run(a: &Args) {
    let what = a.pos(1).unwrap_or_else(|| usage(format!("scenes: what?\n{USAGE}")));
    let allowed = [
        "--items",
        "--detail",
        "--fresh",
        "--no-describe",
        "--no-embed",
        "--no-aesthetic",
        "--download",
        "--sheet",
        "--embeddings",
        "--limit",
        "--purpose",
        "--count",
        "--project",
        "--save",
        "--save-as",
        "--bridge",
        "--data-dir",
        "--compact",
    ];
    if let Some(o) = a.unknown_opt(&allowed) {
        usage(format!("scenes: unknown option {o}\n{USAGE}"));
    }
    let mut b = Backend::open(a);
    match what {
        "analyze" | "analyse" => analyze(a, &mut b).await,
        "show" => {
            let item = a.pos(2).and_then(|v| v.parse::<u64>().ok()).unwrap_or_else(|| usage(format!("scenes show <item>\n{USAGE}")));
            match b.exec("media.scenes", json!({"item": item, "embeddings": a.flag("--embeddings")})).await {
                Ok(v) => print(a, &v),
                Err(e) => fail(format!("scenes: {e}")),
            }
        }
        "search" => {
            let q = a
                .positionals
                .get(2..)
                .map(|p| p.join(" "))
                .filter(|q| !q.trim().is_empty())
                .unwrap_or_else(|| usage(format!("scenes search \"<query>\"\n{USAGE}")));
            let mut p = json!({"query": q, "limit": count(a, "--limit", 20)});
            if let Some(i) = items(a) {
                p["items"] = json!(i);
            }
            match b.exec("media.searchScenes", p).await {
                Ok(v) => {
                    for h in v["hits"].as_array().map(Vec::as_slice).unwrap_or_default() {
                        diag(format_args!(
                            "scenes: {:.2}  #{} {} {:.1}-{:.1}s  {}",
                            h["score"].as_f64().unwrap_or(0.0),
                            h["item"],
                            h["name"].as_str().unwrap_or_default(),
                            h["start"].as_f64().unwrap_or(0.0),
                            h["end"].as_f64().unwrap_or(0.0),
                            h["caption"].as_str().unwrap_or_default()
                        ));
                    }
                    print(a, &v);
                }
                Err(e) => fail(format!("scenes: {e}")),
            }
        }
        "best" => {
            let mut p = json!({"purpose": a.opt("--purpose").unwrap_or("hook"), "count": count(a, "--count", 5)});
            if let Some(i) = items(a) {
                p["items"] = json!(i);
            }
            match b.exec("media.bestShots", p).await {
                Ok(v) => print(a, &v),
                Err(e) => fail(format!("scenes: {e}")),
            }
        }
        "sheet" => {
            let dir = a.pos(2).unwrap_or_else(|| usage(format!("scenes sheet DIR\n{USAGE}")));
            sheet(a, &mut b, dir, items(a)).await;
        }
        "status" => match b.exec("scenes.status", json!({})).await {
            Ok(v) => print(a, &v),
            Err(e) => fail(format!("scenes: {e}")),
        },
        "download" => {
            let model = a.pos(2).unwrap_or("fast");
            download(&mut b, model).await;
        }
        other => usage(format!("scenes: unknown action {other}\n{USAGE}")),
    }
}

/// Download `model` (an id, `fast` or `full`) unless it is installed, saying what and how big.
async fn download(b: &mut Backend, model: &str) {
    let st = b.exec("scenes.status", json!({})).await.unwrap_or_else(|e| fail(format!("scenes: {e}")));
    let ids: Vec<String> = match model {
        "fast" | "full" => st[model]["models"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default(),
        id => vec![id.to_string()],
    };
    for m in st["models"].as_array().map(Vec::as_slice).unwrap_or_default() {
        if ids.iter().any(|i| m["id"] == json!(i)) && m["installed"] != json!(true) {
            diag(format_args!(
                "scenes: downloading {} ({:.2} GB; {}; from {})",
                m["name"].as_str().unwrap_or_default(),
                m["missingBytes"].as_f64().unwrap_or(0.0) / 1e9,
                m["license"].as_str().unwrap_or_default(),
                m["source"].as_str().unwrap_or_default()
            ));
        }
    }
    match b.exec("scenes.downloadModel", json!({"model": model, "wait": true})).await {
        Ok(v) => diag(format_args!("scenes: {}", if v["installed"] == json!(true) { "the models are installed" } else { "downloaded" })),
        Err(e) => fail(format!("scenes: download: {e}")),
    }
}

async fn analyze(a: &Args, b: &mut Backend) {
    let detail = a.opt("--detail").unwrap_or("fast");
    if !matches!(detail, "fast" | "full") {
        usage("scenes: --detail is fast or full");
    }
    // media files given: imported (a file the project holds is reused) and analysed
    let files: Vec<String> = a.positionals.get(2..).unwrap_or_default().iter().map(|p| absolute(p)).collect();
    let mut want = items(a);
    if !files.is_empty() {
        for f in &files {
            if a.opt("--bridge").is_none() && !std::path::Path::new(f).exists() {
                fail(format!("scenes: {f}: no such file"));
            }
        }
        let r = b.exec("file.import", json!({"paths": files, "reuse": true})).await.unwrap_or_else(|e| fail(format!("scenes: import: {e}")));
        let got: Vec<u64> = r["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        want.get_or_insert_with(Vec::new).extend(got);
    }
    if a.flag("--download") {
        download(b, detail).await;
    } else if let Ok(st) = b.exec("scenes.status", json!({})).await {
        let missing = st[detail]["missingBytes"].as_f64().unwrap_or(0.0);
        if missing > 0.0 {
            diag(format_args!("scenes: some {detail} models are not downloaded ({:.1} GB): the rest runs without them; add --download", missing / 1e9));
        }
    }
    let mut p = json!({
        "detail": detail,
        "fresh": a.flag("--fresh"),
        "describe": !a.flag("--no-describe"),
        "embed": !a.flag("--no-embed"),
        "aesthetic": !a.flag("--no-aesthetic"),
        "wait": true,
    });
    if let Some(i) = &want {
        p["items"] = json!(i);
    }
    let t0 = std::time::Instant::now();
    let r = b.exec("media.analyzeScenes", p).await.unwrap_or_else(|e| fail(format!("scenes: {e}")));
    let o = &r["scenes"];
    for it in o["items"].as_array().map(Vec::as_slice).unwrap_or_default() {
        match it["error"].as_str() {
            Some(e) => diag(format_args!("scenes: #{} {}: {e}", it["item"], it["name"].as_str().unwrap_or_default())),
            None => diag(format_args!(
                "scenes: #{} {}: {} shot(s) ({}){}",
                it["item"],
                it["name"].as_str().unwrap_or_default(),
                it["shots"],
                it["boundaries"].as_str().unwrap_or_default(),
                if it["fromCache"] == json!(true) { ", from the cache" } else { "" }
            )),
        }
    }
    for n in o["notes"].as_array().map(Vec::as_slice).unwrap_or_default() {
        diag(format_args!("scenes: note: {}", n.as_str().unwrap_or_default()));
    }
    diag(format_args!(
        "scenes: {} shot(s), {} described, in {:.1}s ({:.2} shots/s)",
        o["shots"],
        o["described"],
        t0.elapsed().as_secs_f64(),
        o["shotsPerSecond"].as_f64().unwrap_or(0.0)
    ));
    if let Some(dir) = a.opt("--sheet") {
        let done: Vec<u64> =
            o["items"].as_array().map(|a| a.iter().filter(|i| i.get("error").is_none()).filter_map(|i| i["item"].as_u64()).collect()).unwrap_or_default();
        sheet(a, b, dir, Some(done)).await;
    }
    print(a, &r);
    b.finish(a).await;
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Write the index of `items` (default: every indexed item) to `dir`: `scenes.json` (each item's
/// `media.scenes`), `sheet.jpg` (every shot's keyframe, labelled with its item and time) and
/// `index.html` (the keyframes with their captions, shot types, camera moves and scores).
async fn sheet(a: &Args, b: &mut Backend, dir: &str, items: Option<Vec<u64>>) {
    let dir = absolute(dir);
    if let Err(e) = std::fs::create_dir_all(std::path::Path::new(&dir).join("frames")) {
        fail(format!("scenes: {dir}: {e}"));
    }
    let ids: Vec<u64> = match items {
        Some(i) => i,
        None => {
            let inspect = b.exec("project.inspect", json!({})).await.unwrap_or_else(|e| fail(format!("scenes: {e}")));
            filmcraft_autocut::analyze::media_items(&inspect).into_iter().map(|m| m.0).collect()
        }
    };
    let mut all = Vec::new();
    let mut sheet_items = Vec::new();
    let mut html = String::from(
        "<!doctype html><meta charset=\"utf-8\"><title>Scene index</title><style>body{font:14px system-ui;background:#111;color:#eee;margin:16px}\
         .shot{display:inline-block;vertical-align:top;width:330px;margin:6px;background:#1c1c1c;padding:6px;border-radius:6px}\
         img{width:100%;border-radius:4px}.meta{color:#9ab;font-size:12px}h2{font-size:16px;margin:18px 6px 4px}</style>\n",
    );
    for id in ids {
        let Ok(v) = b.exec("media.scenes", json!({"item": id})).await else { continue };
        if v["analyzed"] != json!(true) {
            continue;
        }
        let name = v["name"].as_str().unwrap_or_default().to_string();
        html.push_str(&format!(
            "<h2>#{id} {} — {} shot(s), {} · {}</h2>\n",
            html_escape(&name),
            v["shots"].as_array().map_or(0, Vec::len),
            html_escape(v["boundaries"].as_str().unwrap_or_default()),
            html_escape(v["describer"].as_str().unwrap_or("not described"))
        ));
        let mut times = Vec::new();
        for sh in v["shots"].as_array().map(Vec::as_slice).unwrap_or_default() {
            let k = sh["shot"].as_u64().unwrap_or(0);
            let t = sh["keyframes"][0].as_f64().unwrap_or_else(|| sh["start"].as_f64().unwrap_or(0.0));
            times.push(t);
            let file = format!("frames/{id}-{k:03}.jpg");
            let path = std::path::Path::new(&dir).join(&file);
            let _ =
                b.exec("media.contactSheet", json!({"items": [{"item": id, "at": [t]}], "side": 480, "labels": "time", "path": path.to_string_lossy()})).await;
            let d = &sh["description"];
            let q = &sh["quality"];
            html.push_str(&format!(
                "<div class=shot><img src=\"{file}\"><div><b>{}</b></div><div class=meta>{:.1}–{:.1}s · {} · {} · score {:.2} · sharp {:.2} · aesthetic {} · shake {:.2}</div><div class=meta>{}</div><div class=meta>uses: {}</div></div>\n",
                html_escape(d["caption"].as_str().unwrap_or("")),
                sh["start"].as_f64().unwrap_or(0.0),
                sh["end"].as_f64().unwrap_or(0.0),
                html_escape(d["shotType"].as_str().unwrap_or("-")),
                html_escape(sh["camera"].as_str().unwrap_or("-")),
                q["score"].as_f64().unwrap_or(0.0),
                q["sharpness"].as_f64().unwrap_or(0.0),
                q["aesthetic"].as_f64().map_or("-".to_string(), |v| format!("{v:.1}")),
                q["shake"].as_f64().unwrap_or(0.0),
                html_escape(
                    &[d["setting"].as_str().unwrap_or(""), d["mood"].as_str().unwrap_or("")]
                        .iter()
                        .filter(|s| !s.is_empty())
                        .copied()
                        .collect::<Vec<_>>()
                        .join(" · ")
                ),
                html_escape(&d["uses"].as_array().map(|u| u.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).unwrap_or_default())
            ));
        }
        if !times.is_empty() {
            sheet_items.push(json!({"item": id, "at": times}));
        }
        all.push(v);
    }
    let write = |name: &str, bytes: &[u8]| {
        let p = std::path::Path::new(&dir).join(name);
        if let Err(e) = std::fs::write(&p, bytes) {
            fail(format!("scenes: {}: {e}", p.display()));
        }
    };
    write("scenes.json", serde_json::to_string_pretty(&Value::Array(all)).unwrap_or_default().as_bytes());
    write("index.html", html.as_bytes());
    if !sheet_items.is_empty() {
        let path = std::path::Path::new(&dir).join("sheet.jpg");
        if let Err(e) = b.exec("media.contactSheet", json!({"items": sheet_items, "side": 240, "columns": 8, "path": path.to_string_lossy()})).await {
            diag(format_args!("scenes: the contact sheet: {e}"));
        }
    }
    if !a.flag("--compact") {
        diag(format_args!("scenes: wrote {dir}/index.html, scenes.json and sheet.jpg"));
    }
}

//! `audio …`: voice isolation and the one-knob voice boost straight on a file
//! ([`filmcraft_engine::voice_fx`]: `audio.enhanceFile`, `audio.voiceModelStatus`,
//! `audio.downloadVoiceModel`).

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

const USAGE: &str = "\
audio enhance <media file> --out <file.wav|file.m4a> [--amount 0-100] [--lufs -16|mix|none] [--bitrate 160] [--download]
audio isolate <media file> --out <file.wav|file.m4a> [--amount 0-100] [--lufs -16|mix|none] [--bitrate 160] [--download]
audio model [--download]";

fn model_line(st: &Value) -> String {
    format!(
        "{} ({:.1} MB, {}; from {})",
        st["package"].as_str().unwrap_or("the voice model"),
        st["missingBytes"].as_f64().unwrap_or(0.0) / 1e6,
        st["license"].as_str().unwrap_or(""),
        st["source"].as_str().unwrap_or("")
    )
}

pub async fn run(a: &Args) {
    let what = a.pos(1).unwrap_or_else(|| usage(format!("audio: what?\n{USAGE}")));
    let allowed = ["--out", "--amount", "--lufs", "--bitrate", "--download", "--json", "--compact", "--data-dir", "--bridge"];
    if let Some(o) = a.unknown_opt(&allowed) {
        usage(format!("audio: unknown option {o}\n{USAGE}"));
    }
    let mut b = Backend::open(a);
    let mut st = b.exec("audio.voiceModelStatus", json!({})).await.unwrap_or_else(|e| fail(format!("audio: {e}")));
    if st["installed"] != json!(true) && a.flag("--download") {
        diag(format_args!("audio: downloading {}", model_line(&st)));
        b.exec("audio.downloadVoiceModel", json!({"wait": true})).await.unwrap_or_else(|e| fail(format!("audio: download: {e}")));
        st = b.exec("audio.voiceModelStatus", json!({})).await.unwrap_or_else(|e| fail(format!("audio: {e}")));
    }
    match what {
        "model" => {
            diag(format_args!(
                "audio: the effects run {}{}",
                st["model"].as_str().unwrap_or(""),
                if st["neural"] == json!(true) { "" } else { "; add --download for the neural model" }
            ));
            print(a, &st);
        }
        "enhance" | "isolate" => {
            let input = a.pos(2).unwrap_or_else(|| usage(format!("audio {what}: which file?\n{USAGE}")));
            let out = a.opt("--out").unwrap_or_else(|| usage(format!("audio {what}: --out <file.wav|file.m4a> is required\n{USAGE}")));
            if st["neural"] != json!(true) {
                diag(format_args!("audio: the voice model is not installed, so this runs the DSP fallback; add --download for {}", model_line(&st)));
            }
            let mut p = json!({"path": absolute(input), "out": absolute(out), "mode": what});
            if let Some(v) = a.opt("--amount") {
                p["amount"] =
                    json!(v.parse::<f64>().ok().filter(|x| x.is_finite()).unwrap_or_else(|| usage(format!("audio: --amount wants a number, not {v}"))));
            }
            if let Some(v) = a.opt("--lufs") {
                p["lufs"] = v.parse::<f64>().ok().filter(|x| x.is_finite()).map_or_else(|| json!(v), |x| json!(x));
            }
            if let Some(v) = a.opt("--bitrate") {
                p["bitrate"] = json!(v.parse::<u64>().unwrap_or_else(|_| usage(format!("audio: --bitrate wants kbps, not {v}"))));
            }
            match b.exec("audio.enhanceFile", p).await {
                Ok(v) => {
                    let (x, y) = (&v["before"], &v["after"]);
                    diag(format_args!(
                        "audio: {} ({:.1} s) with {}: {} LUFS → {} LUFS, noise floor {} → {} dBFS, 1–4 kHz share {} → {}, {:.0}× real time",
                        v["out"].as_str().unwrap_or(""),
                        v["seconds"].as_f64().unwrap_or(0.0),
                        v["model"].as_str().unwrap_or(""),
                        x["lufs"],
                        y["lufs"],
                        x["noiseFloorDbfs"],
                        y["noiseFloorDbfs"],
                        x["speechBandShare"],
                        y["speechBandShare"],
                        1.0 / v["realTimeFactor"].as_f64().unwrap_or(1.0).max(1e-6)
                    ));
                    print(a, &v);
                }
                Err(e) => fail(format!("audio {what}: {e}")),
            }
        }
        other => usage(format!("audio: unknown {other:?}\n{USAGE}")),
    }
}

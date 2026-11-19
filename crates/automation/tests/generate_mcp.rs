//! `generate.*` over MCP (`command_run` with `wait: true`): the answer says what was made, an
//! asset list with nothing left to make is not a failure, and a failing tool is an error the
//! agent can read. A shell one-liner stands in for the tool.
#![cfg(unix)]

use filmcraft_engine::Session;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::test(flavor = "multi_thread")]
async fn a_waited_generation_says_what_it_made() {
    let dir = std::env::temp_dir().join(format!("fc-generate-mcp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("scene")).unwrap();
    let fixture = dir.join("fixture.png");
    image::RgbaImage::from_pixel(32, 32, image::Rgba([150, 150, 150, 255])).save(&fixture).unwrap();
    let manifest = filmcraft_engine::generate_run::manifest_path().unwrap();
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(
        &manifest,
        format!(
            r#"{{"providers":[{{"id":"stand-in","kind":"image","name":"Stand-in","command":["/bin/sh","-c","cp \"$1\" \"$2\"","sh","{}","{{out}}","{{prompt}}","{{seed}}"]}},{{"id":"broken","kind":"sfx","name":"Broken","command":["/bin/sh","-c","echo no memory left >&2; exit 3","sh","{{out}}"]}}]}}"#,
            fixture.to_string_lossy()
        ),
    )
    .unwrap();
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();
    let (mut tx, server_in) = tokio::io::duplex(1 << 22);
    let (server_out, rx) = tokio::io::duplex(1 << 22);
    tokio::spawn(filmcraft_automation::FilmcraftMcp::headless(session).serve_io(server_in, server_out));
    let mut rx = BufReader::new(rx).lines();
    let hello = json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}});
    tx.write_all(format!("{hello}\n").as_bytes()).await.unwrap();
    rx.next_line().await.unwrap().unwrap();
    tx.write_all(format!("{}\n", json!({"jsonrpc":"2.0","method":"notifications/initialized"})).as_bytes()).await.unwrap();
    let mut n = 0;
    let mut call = async |id: &str, params: Value| -> Value {
        n += 1;
        let msg = json!({"jsonrpc":"2.0","id":n,"method":"tools/call","params":{"name":"command_run","arguments":{"id":id,"params":params}}});
        tx.write_all(format!("{msg}\n").as_bytes()).await.unwrap();
        loop {
            let line: Value = serde_json::from_str(&rx.next_line().await.unwrap().unwrap()).unwrap();
            if line.get("id").and_then(Value::as_u64) == Some(n) {
                return line;
            }
        }
    };
    let text = |r: &Value| -> Value { serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap_or("null")).unwrap_or(Value::Null) };
    let out = dir.join("made").to_string_lossy().into_owned();

    // two variations, the first placed: the answer lists both items and the clip
    let r = call("generate.image", json!({"prompt": "a kite", "variations": 2, "place": true, "seconds": 2, "dir": out, "wait": true})).await;
    assert_ne!(r["result"]["isError"], json!(true), "{r}");
    let v = text(&r);
    let made = v["generated"].as_array().unwrap_or_else(|| panic!("the answer says what was made: {v}"));
    assert_eq!(made.len(), 2, "{v}");
    assert!(made[0]["item"].as_u64().is_some() && made[0]["clip"].as_u64().is_some() && made[1]["clip"].is_null(), "{v}");
    let r = call("generate.pick", json!({"item": made[1]["item"]})).await;
    assert_ne!(r["result"]["isError"], json!(true), "{r}");

    // an asset list: made once, then nothing left to make, which is an answer and not a failure
    let list = dir.join("scene").join("assets.json");
    std::fs::write(&list, json!({"assets": [{"name": "plate", "prompt": "a sky"}]}).to_string()).unwrap();
    let m = list.to_string_lossy().into_owned();
    let r = call("generate.assets", json!({"manifest": m, "wait": true})).await;
    assert_ne!(r["result"]["isError"], json!(true), "{r}");
    assert!(dir.join("scene").join("plate.png").is_file());
    let r = call("generate.assets", json!({"manifest": m, "wait": true})).await;
    assert_ne!(r["result"]["isError"], json!(true), "{r}");
    let v = text(&r);
    assert_eq!((v["making"].as_u64(), v["job"].clone()), (Some(0), Value::Null), "{v}");

    // a tool that fails is an error with its last words; the server goes on
    let r = call("generate.sound", json!({"prompt": "rain", "dir": out, "wait": true})).await;
    assert_eq!(r["result"]["isError"], json!(true), "{r}");
    assert!(r.to_string().contains("no memory left"), "{r}");
    let r = call("generate.providers", json!({})).await;
    assert_eq!(text(&r)["providers"].as_array().map(Vec::len), Some(2), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

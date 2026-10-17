//! `music.suggest` and `music.taxonomy` reach agents over MCP (`command_run`), and a hostile
//! brief comes back as a tool error, never a dead server.

use filmcraft_engine::Session;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::test(flavor = "multi_thread")]
async fn music_suggest_over_mcp() {
    let (mut tx, server_in) = tokio::io::duplex(1 << 22);
    let (server_out, rx) = tokio::io::duplex(1 << 22);
    tokio::spawn(filmcraft_automation::FilmcraftMcp::headless(Session::default()).serve_io(server_in, server_out));
    let mut rx = BufReader::new(rx).lines();
    let mut ask = async |msg: Value| -> Value {
        tx.write_all(format!("{msg}\n").as_bytes()).await.unwrap();
        if msg.get("id").is_none() {
            return Value::Null;
        }
        serde_json::from_str(&rx.next_line().await.unwrap().unwrap()).unwrap()
    };
    ask(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}})).await;
    ask(json!({"jsonrpc":"2.0","method":"notifications/initialized"})).await;
    let text = |r: &Value| -> Value { serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap_or("null")).unwrap_or(Value::Null) };

    let r = ask(json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"music.suggest","params":{"mood":"suspense","duration":45,"count":2}}}})).await;
    assert_ne!(r["result"]["isError"], json!(true), "{r}");
    let v = text(&r);
    assert_eq!(v["suggestions"].as_array().map(Vec::len), Some(2), "{v}");
    assert_eq!(v["suggestions"][0]["style"]["mood"], "tense");
    assert_eq!(v["suggestions"][0]["params"]["kind"], "music");

    let r = ask(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"music.taxonomy"}}})).await;
    assert_eq!(text(&r)["moods"].as_array().map(Vec::len), Some(9), "{r}");

    // a hostile brief is an error the agent can read; the server goes on
    let r = ask(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"music.suggest","params":{"energy":[0.5, "x"], "count": 1e9}}}})).await;
    assert_eq!(r["result"]["isError"], json!(true), "{r}");
    let r = ask(json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"music.suggest","params":{}}}})).await;
    assert_ne!(r["result"]["isError"], json!(true), "{r}");
}

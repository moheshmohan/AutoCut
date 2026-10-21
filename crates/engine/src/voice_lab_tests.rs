use serde_json::json;

use crate::Session;

/// Every hostile request is refused before anything is read, decoded or changed (with or without
/// the models in this build).
#[test]
fn hostile_voice_lab_parameters_are_refused() {
    let mut s = Session::default();
    let bad = [
        ("voice.findReferences", json!({})),
        ("voice.findReferences", json!({"voice": "kokoro-heart", "paths": ["/x"]})),
        ("voice.findReferences", json!({"voice": "clone:me"})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": []})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": [1, 2]})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": ["/x"], "maxMinutes": -1})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": ["/x"], "maxClipSeconds": f64::MAX})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": ["/x"], "minSimilarity": 1.5})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": ["/x"], "top": 0})),
        ("voice.findReferences", json!({"voice": "clone:me", "paths": ["/x"], "maxFiles": "lots"})),
        ("voice.findReferences", json!({"voice": format!("clone:{}", "a".repeat(500)), "paths": ["/x"]})),
        ("voice.audition", json!({"voice": "clone:me"})),
        ("voice.audition", json!({"report": "/x/refs.json"})),
        ("voice.audition", json!({"voice": "clone:me", "report": "/x", "engines": ["kokoro"]})),
        ("voice.audition", json!({"voice": "clone:me", "report": "/x", "engines": []})),
        ("voice.audition", json!({"voice": "clone:me", "report": "/x", "top": 99})),
        ("voice.audition", json!({"voice": "clone:me", "report": "/x", "takes": 0})),
        ("voice.audition", json!({"voice": "clone:me", "report": "/x", "takes": 50})),
        ("voice.audition", json!({"voice": "clone:me", "report": "/x", "variants": "yes"})),
        ("voice.promote", json!({"voice": "clone:me", "report": "/x"})),
        ("voice.promote", json!({"voice": "clone:me", "candidate": "voxcpm-1"})),
        ("voice.promote", json!({"voice": "clone:me", "report": "/x", "candidate": ""})),
        ("voice.promote", json!({"voice": "clone:me", "report": "/x", "candidate": "a", "minSimilarity": f64::NAN})),
        ("voice.versions", json!({})),
        ("voice.rollback", json!({"voice": "clone:me"})),
        ("voice.rollback", json!({"voice": "clone:me", "version": ""})),
        ("voice.mixCheck", json!({})),
        ("voice.mixCheck", json!({"takes": ["a.wav"]})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"]})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"], "chains": {}})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"], "chains": {"x": [{"effect": "gaussian_blur"}]}})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"], "chains": {"x": [{"params": {}}]}})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"], "chains": {"x": "eq"}})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"], "chains": {"x": [{"effect": "parametric_eq", "params": 3}]}})),
        ("voice.mixCheck", json!({"takes": ["a.wav"], "genuine": ["b.wav"], "chains": {"x": []}, "lufs": 10})),
        ("voice.mixCheck", json!({"takes": [], "genuine": ["b.wav"], "chains": {"x": []}})),
    ];
    for (id, p) in bad {
        let r = s.execute(id, p.clone());
        assert!(r.is_err(), "{id} {p}: {r:?}");
    }
    // nothing was started
    assert!(s.jobs.is_empty());
}

#[test]
fn the_voice_lab_commands_are_registered() {
    let ids: Vec<&str> = crate::commands::command_specs().iter().map(|c| c.id).collect();
    for id in ["voice.findReferences", "voice.audition", "voice.promote", "voice.versions", "voice.rollback", "voice.mixCheck"] {
        assert!(ids.contains(&id), "{id}");
    }
}

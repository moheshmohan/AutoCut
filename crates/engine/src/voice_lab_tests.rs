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
        ("voice.findStyles", json!({})),
        ("voice.findStyles", json!({"voice": "clone:me"})),
        ("voice.findStyles", json!({"voice": "kokoro-heart", "paths": ["/x"]})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": []})),
        ("voice.findStyles", json!({"voice": "clone:me", "items": [999_999]})),
        ("voice.findStyles", json!({"voice": "clone:me", "items": "5"})),
        ("voice.findStyles", json!({"voice": "clone:me", "items": [-1]})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "top": 0})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "top": 21})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "top": "3"})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "language": "en"})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "hear": "yes"})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "minSimilarity": 2})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "maxMinutes": 0})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "speakerRefs": "a.wav"})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "speakerRefs": [3]})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "speakerRefs": [{"path": "a.wav", "start": "1"}]})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "speakerRefs": [{"path": "a.wav", "start": 4, "end": 2}]})),
        ("voice.findStyles", json!({"voice": "clone:me", "paths": ["/x"], "speakerRefs": vec!["a.wav"; 21]})),
        ("voice.prosody", json!({})),
        ("voice.prosody", json!({"paths": []})),
        ("voice.prosody", json!({"paths": [""]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "shapes": [{"range": 1.3}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": []})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": [{"range": 3.0}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": [{"range": "big"}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": [{"emphasisDb": 9}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": [{"wobble": 1}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": [{"questionRise": 2, "questionEnd": -1}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "out": "/tmp/x", "shapes": vec![json!({"range": 1.2}); 9]})),
        ("voice.prosody", json!({"segments": [{"path": "a.wav", "start": 5, "end": 2}]})),
        ("voice.prosody", json!({"segments": [{"path": "a.wav", "start": -1}]})),
        ("voice.prosody", json!({"segments": [{"start": 0, "end": 2}]})),
        ("voice.prosody", json!({"segments": [{"path": "a.wav", "start": 0, "end": 4000}]})),
        ("voice.prosody", json!({"paths": ["a.wav"], "hear": 1})),
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
    for id in
        ["voice.findReferences", "voice.audition", "voice.promote", "voice.versions", "voice.rollback", "voice.mixCheck", "voice.findStyles", "voice.prosody"]
    {
        assert!(ids.contains(&id), "{id}");
    }
}

#[test]
fn style_and_prosody_parameters_are_read() {
    let s = Session::default();
    let p = super::styles_params(&s, &json!({"voice": "clone:me", "paths": ["/a", "/b.mov"], "top": 5, "language": "any", "hear": false})).unwrap();
    assert_eq!((p.paths.len(), p.top, p.malayalam, p.hear), (2, 5, false, false));
    let p = super::styles_params(&s, &json!({"voice": "clone:me", "paths": "/a"})).unwrap();
    assert_eq!((p.top, p.malayalam, p.hear, p.min_similarity), (3, true, true, 0.85));
    assert!(p.speaker_refs.is_empty());
    let p =
        super::styles_params(&s, &json!({"voice": "clone:me", "paths": "/a", "speakerRefs": ["/r.mov", {"path": "/s.mov", "start": 2, "end": 30}]})).unwrap();
    assert_eq!(p.speaker_refs.len(), 2);
    assert_eq!((p.speaker_refs[1].start, p.speaker_refs[1].end), (2.0, Some(30.0)));
    let p = super::prosody_params(&json!({
        "paths": ["/a.wav"],
        "segments": [{"path": "/b.mov", "start": 3.5, "end": 12.0}],
        "shapes": [{"range": 1.3}, {"range": 1.6, "questionRise": 3, "questionEnd": null, "emphasisDb": 2}],
        "out": "/tmp/o",
        "hear": true
    }))
    .unwrap();
    assert_eq!(p.inputs.len(), 2);
    assert_eq!((p.inputs[1].start, p.inputs[1].end), (3.5, Some(12.0)));
    assert_eq!(p.shapes.len(), 2);
    assert_eq!((p.shapes[1].range, p.shapes[1].question_rise, p.shapes[1].emphasis_db), (1.6, 3.0, 2.0));
    assert!(p.hear);
}

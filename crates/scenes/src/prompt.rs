//! What the vision-language model is asked about a shot. The answer is read by
//! [`crate::json::parse_description`]; the vocabulary in the prompt is the one it maps. The
//! answer is started for the model (`{"shot_type": "`, [`PREFILL`]), so it answers in JSON from
//! the first token.

/// The system message.
pub const SYSTEM: &str = "You log video footage for a film editor. You describe only what is visible, plainly and exactly.";

/// The start of every answer, given to the model.
pub const PREFILL: &str = "{\"shot_type\": \"";

/// The user message for a shot shown as `frames` keyframes (in time order). With `detail`, a
/// paragraph is asked for too.
pub fn shot_prompt(frames: usize, detail: bool) -> String {
    let what = if frames > 1 {
        format!("These {frames} frames come from one shot of a video, in order.")
    } else {
        "This frame comes from one shot of a video.".to_string()
    };
    let paragraph = if detail { ", \"detail\": \"...\"" } else { "" };
    let paragraph_rule = if detail { " detail: 3 to 5 sentences on who and what is where, what happens, the light and the colours." } else { "" };
    // the lists that can run on (subjects, actions, text) come last: an answer cut short keeps
    // its caption and uses
    format!(
        "{what} Log the shot. Answer with JSON only, in this form:\n\
{{\"shot_type\": \"...\", \"caption\": \"...\"{paragraph}, \"setting\": \"...\", \"mood\": \"...\", \"energy\": \"...\", \
\"people\": 0, \"faces_prominent\": false, \"uses\": [\"...\"], \"subjects\": [{{\"name\": \"...\", \"box\": [x1, y1, x2, y2]}}], \
\"actions\": [\"...\"], \"text\": [\"...\"]}}\n\
Rules: shot_type is one of wide, medium, close-up, extreme close-up, insert, screen. caption: one sentence.{paragraph_rule} \
setting: where. mood: one or two words. energy is one of low, medium, high. people: how many people are visible. \
faces_prominent: whether a face is large enough to recognise. \
uses: the ones that fit of hook (grabs attention at the start), b-roll (cutaway under narration), \
reveal (shows the subject), talking-head (someone speaks to the camera), establishing (shows the place). \
subjects: at most 4, the main one first, each box in 0-1000 coordinates of the frame. \
actions: at most 3, what happens. text: at most 5 lines clearly legible in the frame, copied exactly, else []."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asks_for_every_field_parsed() {
        for detail in [false, true] {
            let p = shot_prompt(1, detail);
            for k in ["shot_type", "subjects", "box", "actions", "setting", "mood", "energy", "\"text\"", "people", "faces_prominent", "caption", "uses"] {
                assert!(p.contains(k), "{k}");
            }
            assert_eq!(p.contains("\"detail\""), detail);
        }
        assert!(shot_prompt(3, false).starts_with("These 3 frames"));
        assert!(PREFILL.starts_with('{'));
    }
}

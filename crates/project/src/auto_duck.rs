//! Per-track auto-ducking: an audio track (music, ambience) whose clips' Volume is lowered while
//! the tracks it is "under" (dialogue, voice-over) have sound, kept up to date when the clips
//! change (the engine's `track.setAutoDuck`, built on Essential Sound's ducking analysis).
//!
//! Everything here is read from project files, so it is hostile: [`AutoDuck::sanitized`] brings
//! the numbers into range before they are used.

use serde::{Deserialize, Serialize};

use crate::{ClipId, TrackId};

/// Duck amount range (dB) and its default.
pub const AMOUNT_DB: (f64, f64) = (0.0, 60.0);
pub const DEFAULT_AMOUNT_DB: f64 = 12.0;
/// Attack (ramp down before the sound) and release (ramp up after it) ranges, seconds.
pub const ATTACK_S: (f64, f64) = (0.001, 5.0);
pub const RELEASE_S: (f64, f64) = (0.001, 10.0);
pub const DEFAULT_ATTACK_S: f64 = 0.25;
pub const DEFAULT_RELEASE_S: f64 = 0.5;
/// Detection sensitivity (Essential Sound's 0–10; 5 reacts to sound above −40 dBFS).
pub const DEFAULT_SENSITIVITY: f64 = 5.0;
/// Clips remembered per track at most (a hostile file cannot make the list unbounded work).
pub const MAX_CLIPS: usize = 10_000;

/// A ducked clip: the level it had before auto-ducking took its Volume over, and what its
/// keyframes were generated from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuckedClip {
    pub clip: ClipId,
    /// Volume level (dB) before ducking.
    pub base_db: f64,
    /// Hash of the clip, the trigger clips over it and the settings when its keyframes were made.
    pub signature: u64,
}

/// Auto-duck settings of an audio track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AutoDuck {
    /// The tracks whose sound lowers this one.
    pub under: Vec<TrackId>,
    /// How far the level goes down (dB, positive).
    pub amount_db: f64,
    /// Ramp down before each passage of sound (seconds).
    pub attack_s: f64,
    /// Ramp back up after it (seconds).
    pub release_s: f64,
    /// Detection sensitivity, 0–10.
    pub sensitivity: f64,
    /// The clips ducked so far.
    pub clips: Vec<DuckedClip>,
}

impl Default for AutoDuck {
    fn default() -> Self {
        AutoDuck {
            under: Vec::new(),
            amount_db: DEFAULT_AMOUNT_DB,
            attack_s: DEFAULT_ATTACK_S,
            release_s: DEFAULT_RELEASE_S,
            sensitivity: DEFAULT_SENSITIVITY,
            clips: Vec::new(),
        }
    }
}

fn clamp(v: f64, (lo, hi): (f64, f64), default: f64) -> f64 {
    if v.is_finite() { v.clamp(lo, hi) } else { default }
}

impl AutoDuck {
    /// The settings with every number in range (non-finite values become the defaults) and the
    /// clip list capped.
    pub fn sanitized(&self) -> AutoDuck {
        let mut under = self.under.clone();
        under.dedup();
        AutoDuck {
            under,
            amount_db: clamp(self.amount_db, AMOUNT_DB, DEFAULT_AMOUNT_DB),
            attack_s: clamp(self.attack_s, ATTACK_S, DEFAULT_ATTACK_S),
            release_s: clamp(self.release_s, RELEASE_S, DEFAULT_RELEASE_S),
            sensitivity: clamp(self.sensitivity, (0.0, 10.0), DEFAULT_SENSITIVITY),
            clips: self.clips.iter().take(MAX_CLIPS).map(|c| DuckedClip { base_db: clamp(c.base_db, (-96.0, 24.0), 0.0), ..*c }).collect(),
        }
    }

    /// The remembered entry of `clip`.
    pub fn clip(&self, clip: ClipId) -> Option<&DuckedClip> {
        self.clips.iter().find(|c| c.clip == clip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostile_numbers_are_brought_into_range() {
        let a = AutoDuck {
            under: vec![TrackId(1), TrackId(1), TrackId(2)],
            amount_db: f64::NAN,
            attack_s: -5.0,
            release_s: 1e300,
            sensitivity: f64::INFINITY,
            clips: vec![DuckedClip { clip: ClipId(3), base_db: 1e9, signature: 7 }],
        };
        let s = a.sanitized();
        assert_eq!(s.under, vec![TrackId(1), TrackId(2)]);
        assert_eq!((s.amount_db, s.attack_s, s.release_s, s.sensitivity), (DEFAULT_AMOUNT_DB, ATTACK_S.0, RELEASE_S.1, DEFAULT_SENSITIVITY));
        assert_eq!(s.clips[0].base_db, 24.0);
        // missing fields read as the defaults
        let d: AutoDuck = serde_json::from_str("{}").unwrap();
        assert_eq!(d, AutoDuck::default());
        let back: AutoDuck = serde_json::from_str(&serde_json::to_string(&a.sanitized()).unwrap()).unwrap();
        assert_eq!(back, a.sanitized());
    }
}

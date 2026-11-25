//! Bounded reuse of shaping plans per face.
//!
//! Building a harfrust shape plan (the face's GSUB / GPOS lookups resolved for a script, direction,
//! language and feature list) costs more than shaping a short run, and caption and title text
//! shapes many short runs of the same face with the same properties. Each [`crate::fonts::Face`]
//! keeps the plans it built last.

use std::sync::{Arc, Mutex, PoisonError};

use harfrust::{Direction, Feature, Language, Script, ShapePlan, ShapePlanKey, Shaper};

/// The plans of one face (without variations). Never shared across faces: a plan holds lookups
/// resolved against one face's tables.
#[derive(Default)]
pub struct ShapePlanCache {
    plans: Mutex<Vec<Arc<ShapePlan>>>,
}

impl ShapePlanCache {
    /// Plans kept per face; the oldest is evicted past it.
    pub const CAPACITY: usize = 32;

    /// A plan for these properties: a kept one, or one built (outside the lock) and kept.
    /// `direction` must not be [`Direction::Invalid`] (harfrust refuses to plan for it).
    pub fn get_or_create(
        &self,
        shaper: &Shaper<'_>,
        direction: Direction,
        script: Option<Script>,
        language: Option<&Language>,
        features: &[Feature],
    ) -> Arc<ShapePlan> {
        let key = ShapePlanKey::new(script, direction).language(language).features(features);
        let kept = |plans: &[Arc<ShapePlan>]| plans.iter().rev().find(|p| key.matches(p)).cloned();
        if let Some(plan) = kept(&self.plans.lock().unwrap_or_else(PoisonError::into_inner)) {
            return plan;
        }
        let plan = Arc::new(ShapePlan::new(shaper, direction, script, language, features));
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        // another thread may have built the same plan meanwhile
        if let Some(made) = kept(&plans) {
            return made;
        }
        if plans.len() >= Self::CAPACITY {
            plans.remove(0);
        }
        plans.push(plan.clone());
        plan
    }

    /// Plans kept.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.plans.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harfrust::{ShapeOptions, Tag, UnicodeBuffer};

    fn buffer(text: &str) -> UnicodeBuffer {
        let mut buf = UnicodeBuffer::new();
        buf.push_str(text);
        buf.set_direction(Direction::LeftToRight);
        buf.guess_segment_properties();
        buf
    }

    /// A kept plan shapes exactly as the plan `shape` builds itself; the same properties get the
    /// same plan back, other features another one.
    #[test]
    fn a_kept_plan_shapes_like_a_fresh_one() {
        let face = crate::fonts::face(0);
        let (Some(font), Some(data)) = (face.font(), face.shaper_data()) else { panic!("the bundled face") };
        let shaper = data.shaper(&font).build();
        let cache = ShapePlanCache::default();
        let no_kern = [Feature::new(Tag::new(b"kern"), 0, ..)];
        for features in [&[][..], &no_kern[..]] {
            for text in ["AVATAR Toyota fi ffl", "Wave, Yes. 0123"] {
                let fresh = shaper.shape(buffer(text), ShapeOptions::new().features(features));
                let b = buffer(text);
                let script = Some(b.script()).filter(|s| *s != harfrust::script::UNKNOWN);
                let plan = cache.get_or_create(&shaper, b.direction(), script, b.language().as_ref(), features);
                let kept = shaper.shape(b, ShapeOptions::new().features(features).plan(Some(&plan)));
                assert_eq!(
                    fresh.glyph_infos().iter().map(|g| (g.glyph_id, g.cluster)).collect::<Vec<_>>(),
                    kept.glyph_infos().iter().map(|g| (g.glyph_id, g.cluster)).collect::<Vec<_>>()
                );
                let pos =
                    |b: &harfrust::GlyphBuffer| b.glyph_positions().iter().map(|p| (p.x_advance, p.y_advance, p.x_offset, p.y_offset)).collect::<Vec<_>>();
                assert_eq!(pos(&fresh), pos(&kept), "{text}");
            }
        }
        // one plan per feature list: Latin text of both strings shares it
        assert_eq!(cache.len(), 2);
        let again = cache.get_or_create(&shaper, Direction::LeftToRight, Some(harfrust::script::LATIN), None, &[]);
        let same = cache.get_or_create(&shaper, Direction::LeftToRight, Some(harfrust::script::LATIN), None, &[]);
        assert!(Arc::ptr_eq(&again, &same));
        assert_eq!(cache.len(), 2);
    }

    /// The cache keeps at most `CAPACITY` plans, evicting the oldest, also when threads fill it
    /// at once.
    #[test]
    fn the_cache_is_bounded_and_shared_by_threads() {
        let face = crate::fonts::face(0);
        let (Some(font), Some(data)) = (face.font(), face.shaper_data()) else { panic!("the bundled face") };
        let cache = ShapePlanCache::default();
        std::thread::scope(|s| {
            for t in 0..4u32 {
                let (cache, font) = (&cache, font.clone());
                s.spawn(move || {
                    let shaper = data.shaper(&font).build();
                    for i in 0..40u32 {
                        let features = [Feature::new(Tag::new(b"kern"), (i * 4 + t) % 50, ..)];
                        cache.get_or_create(&shaper, Direction::LeftToRight, Some(harfrust::script::LATIN), None, &features);
                    }
                });
            }
        });
        assert_eq!(cache.len(), ShapePlanCache::CAPACITY);
    }
}

//! The networks the scene index runs, behind a trait: the downloaded models (feature
//! `scene-models`: TransNetV2, Qwen3-VL, SigLIP 2, CLIP with the LAION head, on the GPU on macOS),
//! or a host's or a test's own.

use std::sync::Arc;

pub use filmcraft_scenes::search::Calibration;

/// A picture for the models: straight RGBA8.
#[derive(Clone, Debug, PartialEq)]
pub struct Pic {
    pub rgba: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

/// How much detail a description has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// One keyframe a shot, the small describer.
    Fast,
    /// Two or three keyframes, the larger describer, a paragraph.
    Full,
}

impl Detail {
    pub fn name(self) -> &'static str {
        match self {
            Detail::Fast => "fast",
            Detail::Full => "full",
        }
    }
}

/// What a describer run cost (for the job's report).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DescribeStats {
    pub batches: usize,
    pub prompt_tokens: usize,
    pub generated: usize,
    pub vision_seconds: f64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub load_seconds: f64,
}

/// Progress of a describer run: `(shots done, shots)`; false stops it.
pub type DescribeProgress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

/// The scene index's networks. Each `*_id` says which model (with its pins) a part uses, `None`
/// when the part is not available; the analysis skips what is not, and keys its cache by what
/// is.
pub trait SceneModels: Send + Sync {
    fn boundaries_id(&self) -> Option<String> {
        None
    }
    fn describer_id(&self, detail: Detail) -> Option<String> {
        let _ = detail;
        None
    }
    fn embedder_id(&self) -> Option<String> {
        None
    }
    fn aesthetic_id(&self) -> Option<String> {
        None
    }
    /// The probability that each frame (48×27 RGB) is a transition.
    fn cuts(&self, frames: &[Vec<u8>], cancelled: &dyn Fn() -> bool) -> Result<Vec<f32>, String> {
        let _ = (frames, cancelled);
        Err("no shot-boundary model".into())
    }
    /// One answer (JSON text, see `filmcraft_scenes::prompt`) per shot (its keyframes).
    fn describe(&self, shots: &[Vec<Pic>], detail: Detail, progress: DescribeProgress<'_>) -> Result<(Vec<String>, DescribeStats), String> {
        let _ = (shots, detail, progress);
        Err("no describer".into())
    }
    /// Unit-length image embeddings.
    fn embed_images(&self, pics: &[Pic]) -> Result<Vec<Vec<f32>>, String> {
        let _ = pics;
        Err("no embedding model".into())
    }
    /// The unit-length embedding of a text query, and how its cosine becomes a match.
    fn embed_text(&self, query: &str) -> Result<(Vec<f32>, Option<Calibration>), String> {
        let _ = query;
        Err("no embedding model".into())
    }
    /// The aesthetic score (about 1..10) of each picture.
    fn aesthetic(&self, pics: &[Pic]) -> Result<Vec<f32>, String> {
        let _ = pics;
        Err("no aesthetic model".into())
    }
    /// Which model answers questions about pictures ([`SceneModels::ask`]), when one can.
    fn asker_id(&self) -> Option<String> {
        None
    }
    /// A short answer (at most `max_tokens` tokens) to `question` about `frames`: one moment of a
    /// video, 1 to 4 pictures of one size. `system` sets how to answer.
    fn ask(&self, frames: &[Pic], system: &str, question: &str, max_tokens: usize) -> Result<String, String> {
        let _ = (frames, system, question, max_tokens);
        Err("no vision-language model".into())
    }
    /// Let go of the big models after a job (the embedder may stay, for search).
    fn release(&self) {}
    /// Hold the GPU for a job (none where there is no GPU or no lock).
    fn gpu(&self) -> Result<Option<Box<dyn std::any::Any + Send>>, String> {
        Ok(None)
    }
}

/// No models: shots from the fallback detector and the picture measures only.
pub struct NoModels;

impl SceneModels for NoModels {}

/// The downloaded models in `models_dir`, loaded when first needed.
pub fn local(models_dir: std::path::PathBuf) -> Option<Arc<dyn SceneModels>> {
    #[cfg(feature = "scene-models")]
    {
        Some(Arc::new(local::Local::new(models_dir)))
    }
    #[cfg(not(feature = "scene-models"))]
    {
        let _ = models_dir;
        None
    }
}

#[cfg(feature = "scene-models")]
mod local {
    use std::path::PathBuf;
    use std::sync::{Mutex, PoisonError};

    use filmcraft_scenes::catalog;
    use filmcraft_scenes::model::aesthetic::Aesthetic;
    use filmcraft_scenes::model::qwen3vl::{Frame, Options, Qwen3Vl};
    use filmcraft_scenes::model::siglip::Siglip;
    use filmcraft_scenes::model::transnet::TransNet;
    use filmcraft_scenes::prompt;

    use super::{Calibration, DescribeProgress, DescribeStats, Detail, Pic, SceneModels};

    /// Shots a describer batch holds: the 2B model decodes 16 at about the cost of a few; the
    /// larger ones with more keyframes fewer.
    fn batch_size(detail: Detail) -> usize {
        match detail {
            Detail::Fast => 16,
            Detail::Full => 4,
        }
    }

    /// The pixels a keyframe is given, and the longest answer.
    fn options(detail: Detail) -> Options {
        match detail {
            Detail::Fast => Options { max_pixels: 200_704, max_new: 320 },
            Detail::Full => Options { max_pixels: 301_056, max_new: 640 },
        }
    }

    pub struct Local {
        dir: PathBuf,
        dev: filmcraft_scenes::model::Device,
        transnet: Mutex<Option<TransNet>>,
        vlm: Mutex<Option<Qwen3Vl>>,
        siglip: Mutex<Option<Siglip>>,
        aesthetic: Mutex<Option<Aesthetic>>,
    }

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `id@sha` of a model's largest file: what the cache keys by.
    fn pinned(id: &str) -> Option<String> {
        let m = catalog::find(id)?;
        let big = m.files.iter().max_by_key(|f| f.size)?;
        Some(format!("{id}@{}", big.sha256.get(..12).unwrap_or(big.sha256)))
    }

    impl Local {
        pub fn new(dir: PathBuf) -> Self {
            Local {
                dir,
                dev: filmcraft_scenes::model::device(),
                transnet: Mutex::new(None),
                vlm: Mutex::new(None),
                siglip: Mutex::new(None),
                aesthetic: Mutex::new(None),
            }
        }

        fn installed(&self, id: &str) -> bool {
            catalog::installed_id(&self.dir, id)
        }

        fn describer(&self, detail: Detail) -> Option<&'static str> {
            // the larger describer when asked for detail and installed, else the fast one
            let want: &[&str] = match detail {
                Detail::Full => &[catalog::FULL_DESCRIBER, catalog::QWEN_8B, catalog::FAST_DESCRIBER],
                Detail::Fast => &[catalog::FAST_DESCRIBER],
            };
            want.iter().copied().find(|id| self.installed(id))
        }
    }

    impl SceneModels for Local {
        fn boundaries_id(&self) -> Option<String> {
            self.installed(catalog::TRANSNET).then(|| pinned(catalog::TRANSNET)).flatten()
        }
        fn describer_id(&self, detail: Detail) -> Option<String> {
            self.describer(detail).and_then(pinned)
        }
        fn embedder_id(&self) -> Option<String> {
            self.installed(catalog::SIGLIP).then(|| pinned(catalog::SIGLIP)).flatten()
        }
        fn aesthetic_id(&self) -> Option<String> {
            self.installed(catalog::AESTHETIC).then(|| pinned(catalog::AESTHETIC)).flatten()
        }

        fn cuts(&self, frames: &[Vec<u8>], cancelled: &dyn Fn() -> bool) -> Result<Vec<f32>, String> {
            let mut g = lock(&self.transnet);
            if g.is_none() {
                *g = Some(TransNet::load(&self.dir.join(catalog::TRANSNET), &self.dev).map_err(|e| e.to_string())?);
            }
            let net = g.as_ref().ok_or("TransNetV2 did not load")?;
            net.predict(frames, cancelled).map_err(|e| e.to_string())
        }

        fn describe(&self, shots: &[Vec<Pic>], detail: Detail, progress: DescribeProgress<'_>) -> Result<(Vec<String>, DescribeStats), String> {
            let id = self.describer(detail).ok_or("no describer is downloaded")?;
            let mut stats = DescribeStats::default();
            let mut g = lock(&self.vlm);
            if g.as_ref().is_none_or(|m| m.id != id) {
                *g = None;
                let t = web_time::Instant::now();
                *g = Some(Qwen3Vl::load(&self.dir.join(id), id, &self.dev).map_err(|e| e.to_string())?);
                stats.load_seconds = t.elapsed().as_secs_f64();
            }
            let vlm = g.as_mut().ok_or("the describer did not load")?;
            let o = options(detail);
            let full = detail == Detail::Full;
            // shots of one picture size and keyframe count decode together
            let mut groups: std::collections::BTreeMap<(usize, (usize, usize)), Vec<usize>> = Default::default();
            for (i, s) in shots.iter().enumerate() {
                let size = s.first().map_or((0, 0), |p| vlm.frame_size(p.w, p.h, o.max_pixels));
                groups.entry((s.len(), size)).or_default().push(i);
            }
            let mut out = vec![String::new(); shots.len()];
            let mut done = 0;
            let total = shots.len();
            for ((frames, _), idx) in groups {
                if frames == 0 {
                    continue;
                }
                for chunk in idx.chunks(batch_size(detail)) {
                    if !progress(done, total) {
                        return Err("stopped".into());
                    }
                    let batch: Vec<Vec<Frame>> =
                        chunk.iter().filter_map(|i| shots.get(*i)).map(|s| s.iter().map(|p| Frame { rgba: &p.rgba, w: p.w, h: p.h }).collect()).collect();
                    let user = prompt::shot_prompt(frames, full);
                    let (answers, st) = vlm.generate(&batch, prompt::SYSTEM, &user, prompt::PREFILL, &o, &|| false).map_err(|e| e.to_string())?;
                    stats.batches += 1;
                    stats.prompt_tokens += st.prompt_tokens * chunk.len();
                    stats.generated += st.generated;
                    stats.vision_seconds += st.vision_seconds;
                    stats.prefill_seconds += st.prefill_seconds;
                    stats.decode_seconds += st.decode_seconds;
                    for (i, a) in chunk.iter().zip(answers) {
                        if let Some(slot) = out.get_mut(*i) {
                            *slot = a;
                        }
                    }
                    done += chunk.len();
                }
            }
            progress(done, total);
            Ok((out, stats))
        }

        fn embed_images(&self, pics: &[Pic]) -> Result<Vec<Vec<f32>>, String> {
            let mut g = lock(&self.siglip);
            if g.is_none() {
                *g = Some(Siglip::load(&self.dir.join(catalog::SIGLIP), &self.dev).map_err(|e| e.to_string())?);
            }
            let m = g.as_ref().ok_or("SigLIP did not load")?;
            let refs: Vec<(&[u8], usize, usize)> = pics.iter().map(|p| (p.rgba.as_slice(), p.w, p.h)).collect();
            m.embed_images(&refs).map_err(|e| e.to_string())
        }

        fn embed_text(&self, query: &str) -> Result<(Vec<f32>, Option<Calibration>), String> {
            if !self.installed(catalog::SIGLIP) {
                return Err("the embedding model is not downloaded".into());
            }
            let mut g = lock(&self.siglip);
            if g.is_none() {
                *g = Some(Siglip::load(&self.dir.join(catalog::SIGLIP), &self.dev).map_err(|e| e.to_string())?);
            }
            let m = g.as_ref().ok_or("SigLIP did not load")?;
            Ok((m.embed_text(query).map_err(|e| e.to_string())?, Some(m.calibration())))
        }

        fn aesthetic(&self, pics: &[Pic]) -> Result<Vec<f32>, String> {
            let mut g = lock(&self.aesthetic);
            if g.is_none() {
                *g = Some(Aesthetic::load(&self.dir.join(catalog::AESTHETIC), &self.dev).map_err(|e| e.to_string())?);
            }
            let m = g.as_ref().ok_or("the aesthetic model did not load")?;
            let refs: Vec<(&[u8], usize, usize)> = pics.iter().map(|p| (p.rgba.as_slice(), p.w, p.h)).collect();
            m.score(&refs).map_err(|e| e.to_string())
        }

        fn asker_id(&self) -> Option<String> {
            self.describer(Detail::Full).and_then(pinned)
        }

        fn ask(&self, frames: &[Pic], system: &str, question: &str, max_tokens: usize) -> Result<String, String> {
            // the larger describer when it is installed: it reads small things the 2B misses
            let id = self.describer(Detail::Full).ok_or("no describer is downloaded")?;
            let first = frames.first().ok_or("no frame to look at")?;
            if frames.len() > filmcraft_scenes::model::qwen3vl::MAX_FRAMES || frames.iter().any(|p| (p.w, p.h) != (first.w, first.h)) {
                return Err("1 to 4 frames of one size".into());
            }
            let mut g = lock(&self.vlm);
            if g.as_ref().is_none_or(|m| m.id != id) {
                *g = None;
                *g = Some(Qwen3Vl::load(&self.dir.join(id), id, &self.dev).map_err(|e| e.to_string())?);
            }
            let vlm = g.as_mut().ok_or("the describer did not load")?;
            let shot: Vec<Frame> = frames.iter().map(|p| Frame { rgba: &p.rgba, w: p.w, h: p.h }).collect();
            let o = Options { max_pixels: options(Detail::Full).max_pixels, max_new: max_tokens.clamp(1, 64) };
            let (answers, _) = vlm.generate(&[shot], system, question, "", &o, &|| false).map_err(|e| e.to_string())?;
            answers.into_iter().next().ok_or_else(|| "no answer".into())
        }

        fn release(&self) {
            *lock(&self.transnet) = None;
            *lock(&self.vlm) = None;
            *lock(&self.aesthetic) = None;
        }

        fn gpu(&self) -> Result<Option<Box<dyn std::any::Any + Send>>, String> {
            let g = filmcraft_scenes::model::gpu_lock(&self.dev, "scene analysis").map_err(|e| e.to_string())?;
            Ok(g.map(|g| Box::new(g) as Box<dyn std::any::Any + Send>))
        }
    }
}

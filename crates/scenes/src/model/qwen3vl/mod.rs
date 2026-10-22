//! Qwen3-VL (Qwen team, Alibaba Cloud; Apache-2.0) on candle: the vision encoder
//! ([`vision`]), the text decoder with interleaved M-RoPE and DeepStack ([`text`]), and batched
//! greedy decoding of one answer per shot. Written for FilmCraft from the model's published
//! configuration and architecture (Hugging Face `transformers`' `qwen3_vl`, Apache-2.0, and
//! `candle-transformers`' port, MIT/Apache-2.0), with the multimodal rotary positions, a
//! resettable key/value cache and errors instead of panics.
//!
//! **Batching.** Every shot of a batch has the same number of keyframes at the same picture
//! size (the engine groups the shots of a clip), so every prompt has the same tokens in the same
//! places: the batch decodes in lockstep with one position for all, and a finished answer just
//! idles until the others end. Decoding is memory-bound, so eight answers cost little more than
//! one.

pub mod config;
mod text;
mod vision;

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::image::{Rgb, resize};
use super::weights::SafeTensors;
use super::{R, err};
use crate::catalog;
use config::Config;

/// The model's weights, read from its files one tensor at a time as the layers take them, so
/// loading never holds a second copy of the model.
pub(crate) struct Weights {
    files: Vec<SafeTensors>,
    dev: Device,
    dtype: DType,
}

impl Weights {
    /// Tensor `name` on `dev`, checked against `shape` before it is read.
    fn read(&mut self, name: &str, shape: &[usize], dev: &Device) -> R<Tensor> {
        let st = self.files.iter_mut().find(|f| f.contains(name)).ok_or_else(|| err(format!("Qwen3-VL: missing weight `{name}`")))?;
        match st.shape(name) {
            Some(s) if s == shape => st.load(name, dev, self.dtype),
            s => Err(err(format!("Qwen3-VL: weight `{name}` is {s:?}, expected {shape:?}"))),
        }
    }
}

/// Tensor `name` of `w` on the model's device, checking its shape.
pub(crate) fn take(w: &mut Weights, name: &str, shape: &[usize]) -> R<Tensor> {
    let dev = w.dev.clone();
    w.read(name, shape, &dev)
}

/// Weights of one input stacked into one matrix (`parts`: names and shapes): joined on the CPU
/// before they go to the device, which then holds only the joined copy. (Joined there, candle's
/// Metal buffer pool would keep the parts too: the model twice over while it loads.)
pub(crate) fn take_stacked(w: &mut Weights, parts: &[(String, Vec<usize>)]) -> R<Tensor> {
    let ts = parts.iter().map(|(n, s)| w.read(n, s, &Device::Cpu)).collect::<R<Vec<_>>>()?;
    Tensor::cat(&ts, 0).and_then(|t| t.to_device(&w.dev)).map_err(err)
}

/// `l` applied to the rows of `x` as one matrix product. candle's `Linear` broadcasts its weight
/// over a 3-D input's batch, which on the GPU reads the weight once per batch row: decoding eight
/// answers then costs eight times the weight traffic of one.
pub(crate) fn apply(l: &candle_nn::Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    use candle_core::Module;
    let dims = x.dims().to_vec();
    let Some((&last, lead)) = dims.split_last() else { return l.forward(x) };
    if lead.len() <= 1 {
        return l.forward(x);
    }
    let rows: usize = lead.iter().product();
    let y = l.forward(&x.reshape((rows, last))?)?;
    let mut shape = lead.to_vec();
    shape.push(y.dim(1)?);
    y.reshape(shape)
}

/// The fewest pixels Qwen3-VL's processor gives a picture.
const MIN_PIXELS: usize = 65_536;
/// Longest prompt plus answer the cache is made for at first (it grows if needed).
const CACHE_LEN: usize = 1_024;
/// Most shots decoded together, and keyframes per shot.
pub const MAX_BATCH: usize = 16;
pub const MAX_FRAMES: usize = 4;
/// Patches the vision encoder takes in one pass.
const VISION_PATCHES: usize = 4096;

/// One frame: straight RGBA8, `w` × `h`.
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub rgba: &'a [u8],
    pub w: usize,
    pub h: usize,
}

/// How to answer.
#[derive(Clone, Debug)]
pub struct Options {
    /// Most pixels a keyframe is given (it is resized to multiples of 32 px within this).
    pub max_pixels: usize,
    /// Most tokens an answer may have.
    pub max_new: usize,
}

/// What a batch cost.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub vision_seconds: f64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

/// The loaded model.
pub struct Qwen3Vl {
    pub id: String,
    cfg: Config,
    vision: vision::Vision,
    text: text::Text,
    tok: tokenizers::Tokenizer,
    dev: Device,
    special: Special,
    /// The shared start of the prompt and its keys and values (batch of one), kept between
    /// batches.
    prefix: Option<(Vec<u32>, Vec<(Tensor, Tensor)>)>,
}

#[derive(Clone, Copy)]
struct Special {
    im_start: u32,
    im_end: u32,
    end_of_text: u32,
    image_pad: u32,
    vision_start: u32,
    vision_end: u32,
}

/// The size a `w` × `h` picture is resized to: multiples of `factor`, about the same aspect,
/// at most `max_pixels` and at least [`MIN_PIXELS`] (Qwen's `smart_resize`).
pub fn smart_size(w: usize, h: usize, factor: usize, max_pixels: usize) -> (usize, usize) {
    let f = factor.max(1) as f64;
    let (w, h) = (w.max(1) as f64, h.max(1) as f64);
    // a picture more than 200:1 is cut to that
    let (w, h) = if w / h > 200.0 {
        (h * 200.0, h)
    } else if h / w > 200.0 {
        (w, w * 200.0)
    } else {
        (w, h)
    };
    let max = max_pixels.max(MIN_PIXELS) as f64;
    let mut hb = ((h / f).round() * f).max(f);
    let mut wb = ((w / f).round() * f).max(f);
    if hb * wb > max {
        let beta = (h * w / max).sqrt();
        hb = ((h / beta / f).floor() * f).max(f);
        wb = ((w / beta / f).floor() * f).max(f);
    } else if hb * wb < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    (wb as usize, hb as usize)
}

impl Qwen3Vl {
    /// Load model `id` (a Qwen3-VL entry of the catalogue) from its directory.
    pub fn load(dir: &Path, id: &str, dev: &Device) -> R<Qwen3Vl> {
        let m = catalog::find(id).filter(|m| m.role == catalog::Role::Describer).ok_or_else(|| err(format!("{id} is not a describer")))?;
        let json = std::fs::read_to_string(dir.join("config.json")).map_err(|e| err(format!("{id}: config.json: {e}")))?;
        let cfg = Config::parse(&json)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| err(format!("{id}: tokenizer.json: {e}")))?;
        let id_of = |t: &str| tok.token_to_id(t).ok_or_else(|| err(format!("{id}: the tokenizer has no {t}")));
        let special = Special {
            im_start: id_of("<|im_start|>")?,
            im_end: id_of("<|im_end|>")?,
            end_of_text: id_of("<|endoftext|>")?,
            image_pad: id_of("<|image_pad|>")?,
            vision_start: id_of("<|vision_start|>")?,
            vision_end: id_of("<|vision_end|>")?,
        };
        if special.image_pad != cfg.image_token || special.vision_start != cfg.vision_start || special.vision_end != cfg.vision_end {
            return Err(err(format!("{id}: the tokenizer and the config disagree on the picture tokens")));
        }
        let dtype = if dev.is_cpu() { DType::F32 } else { DType::BF16 };
        let files = m.files.iter().filter(|f| f.name.ends_with(".safetensors")).map(|f| SafeTensors::open(&dir.join(f.name))).collect::<R<Vec<_>>>()?;
        let mut w = Weights { files, dev: dev.clone(), dtype };
        let vision = vision::Vision::load(&mut w, &cfg.vision, dtype, dev)?;
        let text = text::Text::load(&mut w, &cfg.text, CACHE_LEN, dtype, dev)?;
        Ok(Qwen3Vl { id: id.to_string(), cfg, vision, text, tok, dev: dev.clone(), special, prefix: None })
    }

    /// Fill the cache with the keys and values of `ids` for a batch of `b`: worked out once (as
    /// a batch of one) and copied to every row; the same prompt start again reuses them.
    fn load_prefix(&mut self, ids: &[u32], b: usize) -> R<()> {
        if self.prefix.as_ref().is_none_or(|(p, _)| p != ids) {
            self.text.reset();
            let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.dev).map_err(err)?;
            let x = self.text.embed(&t)?;
            let pos = text::positions(&[Ok(ids.len())], 0);
            let (cos, sin) = self.text.rotary(&pos)?;
            self.text.forward(x, &cos, &sin, &[])?;
            self.prefix = Some((ids.to_vec(), self.text.snapshot()?));
        }
        let kv = self.prefix.as_ref().map(|(_, kv)| kv.clone()).unwrap_or_default();
        self.text.restore(&kv, b)
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    fn encode(&self, s: &str) -> R<Vec<u32>> {
        self.tok.encode(s, false).map(|e| e.get_ids().to_vec()).map_err(|e| err(format!("{}: tokenizer: {e}", self.id)))
    }

    /// The picture size frames of `w` × `h` are given.
    pub fn frame_size(&self, w: usize, h: usize, max_pixels: usize) -> (usize, usize) {
        smart_size(w, h, self.cfg.vision.patch * self.cfg.vision.merge, max_pixels)
    }

    /// Answer `user` (after `system`) about each shot of `shots` (its keyframes), every answer
    /// starting with `prefill`. Shots must have the same number of frames, of one size.
    /// `cancelled` is asked between tokens.
    pub fn generate(
        &mut self,
        shots: &[Vec<Frame<'_>>],
        system: &str,
        user: &str,
        prefill: &str,
        o: &Options,
        cancelled: &dyn Fn() -> bool,
    ) -> R<(Vec<String>, Stats)> {
        let b = shots.len();
        let nf = shots.first().map_or(0, Vec::len);
        if b == 0 || b > MAX_BATCH || nf == 0 || nf > MAX_FRAMES || shots.iter().any(|s| s.len() != nf) {
            return Err(crate::SceneError::Invalid("a batch is 1 to 16 shots with the same number of keyframes (1 to 4)".into()));
        }
        let first = shots.first().and_then(|s| s.first()).ok_or_else(|| err("empty batch"))?;
        let (fw, fh) = self.frame_size(first.w, first.h, o.max_pixels);
        if shots.iter().flatten().any(|f| self.frame_size(f.w, f.h, o.max_pixels) != (fw, fh)) {
            return Err(crate::SceneError::Invalid("the shots of a batch must share a picture size".into()));
        }
        let vc = self.cfg.vision.clone();
        let (gh, gw) = (fh / vc.patch, fw / vc.patch);
        let (mh, mw) = (gh / vc.merge, gw / vc.merge);
        let per_image = mh * mw;
        let t0 = std::time::Instant::now();
        // the pictures, through the vision encoder
        let mut px = Vec::with_capacity(b * nf * gh * gw * self.vision.patch_len());
        for f in shots.iter().flatten() {
            let img: Rgb = resize(f.rgba, f.w, f.h, fw, fh).ok_or_else(|| crate::SceneError::Invalid("a keyframe is empty or damaged".into()))?;
            vision::patchify(&img, vc.patch, vc.temporal, vc.merge, &mut px);
        }
        let patches = Tensor::from_vec(px, (b * nf * gh * gw, self.vision.patch_len()), &self.dev).map_err(err)?;
        // the encoder's attention is per picture and quadratic in its patches: a few pictures at
        // a time keep its memory bounded (about 4096 patches a pass)
        let pics = b * nf;
        let per_pass = (VISION_PATCHES / (gh * gw).max(1)).clamp(1, pics);
        let mut toks = Vec::new();
        let mut decks: Vec<Vec<Tensor>> = Vec::new();
        let mut at = 0;
        while at < pics {
            let k = per_pass.min(pics - at);
            let part = patches.narrow(0, at * gh * gw, k * gh * gw).map_err(err)?;
            let e = self.vision.forward(&part, k, gh, gw)?;
            toks.push(e.tokens);
            decks.push(e.deepstack);
            at += k;
        }
        let deepstack = (0..decks.first().map_or(0, Vec::len))
            .map(|l| Tensor::cat(&decks.iter().filter_map(|d| d.get(l)).collect::<Vec<_>>(), 0))
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(err)?;
        let enc = vision::Encoded { tokens: Tensor::cat(&toks, 0).map_err(err)?, deepstack };
        let hidden = self.cfg.text.hidden;
        let tokens = enc.tokens.reshape((b, nf, per_image, hidden)).map_err(err)?;
        let deep: Vec<Tensor> = enc.deepstack.iter().map(|d| d.reshape((b, nf, per_image, hidden))).collect::<candle_core::Result<_>>().map_err(err)?;
        // (the GPU runs ahead: wait for it so the times are each stage's own)
        self.dev.synchronize().map_err(err)?;
        let vision_seconds = t0.elapsed().as_secs_f64();
        // the prompt: the chat template, the instructions before the pictures so their keys and
        // values are worked out once and shared by every shot (and every batch)
        let sp = self.special;
        let mut head = vec![sp.im_start];
        head.extend(self.encode(&format!("system\n{system}"))?);
        head.push(sp.im_end);
        head.extend(self.encode("\n")?);
        head.push(sp.im_start);
        head.extend(self.encode(&format!("user\n{user}"))?);
        let t1 = std::time::Instant::now();
        self.load_prefix(&head, b)?;
        let past = head.len();
        let mut tail = vec![sp.im_end];
        tail.extend(self.encode("\n")?);
        tail.push(sp.im_start);
        tail.extend(self.encode("assistant\n")?);
        tail.extend(self.encode(prefill)?);
        let mut runs: Vec<Result<Vec<u32>, usize>> = Vec::new();
        let mut text_run = Vec::new();
        for f in 0..nf {
            text_run.push(sp.vision_start);
            runs.push(Ok(std::mem::take(&mut text_run)));
            runs.push(Err(f));
            text_run.push(sp.vision_end);
        }
        text_run.extend(tail);
        runs.push(Ok(text_run));
        let layout: Vec<Result<usize, (usize, usize)>> = runs.iter().map(|r| r.as_ref().map(Vec::len).map_err(|_| (mh, mw))).collect();
        let pos = text::positions(&layout, past as i64);
        let total = past + pos.len();
        let mut parts = Vec::new();
        let mut visual = Vec::new();
        let mut at = 0;
        for r in &runs {
            match r {
                Ok(ids) => {
                    let t = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.dev).map_err(err)?;
                    let e = self.text.embed(&t).and_then(|e| e.broadcast_as((b, ids.len(), hidden)).map_err(err)).and_then(|e| e.contiguous().map_err(err))?;
                    parts.push(e);
                    at += ids.len();
                }
                Err(f) => {
                    parts.push(tokens.narrow(1, *f, 1).and_then(|t| t.squeeze(1)).map_err(err)?);
                    let features = deep.iter().map(|d| d.narrow(1, *f, 1).and_then(|t| t.squeeze(1))).collect::<candle_core::Result<Vec<_>>>().map_err(err)?;
                    visual.push(text::Visual { start: at, len: per_image, features });
                    at += per_image;
                }
            }
        }
        let x = Tensor::cat(&parts, 1).map_err(err)?;
        let (cos, sin) = self.text.rotary(&pos)?;
        let mut logits = self.text.forward(x, &cos, &sin, &visual)?;
        self.dev.synchronize().map_err(err)?;
        let prefill_seconds = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        // greedy decoding in lockstep
        let first_pos = pos.iter().map(|p| p[0].max(p[1]).max(p[2])).max().unwrap_or(0) + 1;
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut done = vec![false; b];
        let mut json: Vec<JsonState> = (0..b).map(|_| JsonState::new(prefill)).collect();
        let mut generated = 0;
        let profile = std::env::var_os("FILMCRAFT_SCENES_PROFILE").is_some();
        let (mut t_wait, mut t_host) = (0f64, 0f64);
        for step in 0..o.max_new.max(1) {
            if cancelled() {
                return Err(crate::SceneError::Cancelled);
            }
            let tw = std::time::Instant::now();
            // the best token of each row on the GPU; a row's logits come over only when its best
            // token would repeat a phrase
            let top: Vec<u32> = logits.argmax_keepdim(candle_core::D::Minus1).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<u32>()).map_err(err)?;
            t_wait += tw.elapsed().as_secs_f64();
            let th = std::time::Instant::now();
            let mut next = Vec::with_capacity(b);
            for (i, ((so_far, finished), state)) in out.iter_mut().zip(done.iter_mut()).zip(json.iter_mut()).enumerate() {
                let best = top.get(i).copied().unwrap_or(sp.end_of_text);
                let banned = banned_after(so_far, NO_REPEAT);
                let id = if banned.contains(&best) {
                    let row: Vec<f32> = logits.get(i).and_then(|r| r.to_vec1::<f32>()).map_err(err)?;
                    pick(&row, so_far, NO_REPEAT)
                } else {
                    best
                };
                if !*finished {
                    if id == sp.im_end || id == sp.end_of_text {
                        *finished = true;
                    } else {
                        so_far.push(id);
                        generated += 1;
                        if let Some(piece) = self.tok.id_to_token(id) {
                            state.feed(&piece);
                        }
                        if state.closed() || looping(so_far) {
                            *finished = true;
                        }
                    }
                }
                next.push(if *finished { sp.end_of_text } else { id });
            }
            if done.iter().all(|d| *d) {
                break;
            }
            let ids = Tensor::from_vec(next, (b, 1), &self.dev).map_err(err)?;
            let x = self.text.embed(&ids)?;
            let p = first_pos + step as i64;
            let (cos, sin) = self.text.rotary(&[[p; 3]])?;
            logits = self.text.forward(x, &cos, &sin, &[])?;
            t_host += th.elapsed().as_secs_f64();
        }
        if profile {
            eprintln!("decode: waiting for the GPU {t_wait:.2}s, host {t_host:.2}s");
        }
        let decode_seconds = t2.elapsed().as_secs_f64();
        let answers = out
            .iter()
            .map(|ids| self.tok.decode(ids, true).map(|t| format!("{prefill}{t}")).map_err(|e| err(format!("{}: tokenizer: {e}", self.id))))
            .collect::<R<Vec<String>>>()?;
        Ok((answers, Stats { prompt_tokens: total, generated, vision_seconds, prefill_seconds, decode_seconds }))
    }
}

/// No `n`-gram of an answer may repeat: greedy decoding falls into loops
/// (`"VoopeoBot", "VoopeoBot", …`), and a repeat this long is never wanted in a description.
const NO_REPEAT: usize = 10;

/// The best token that does not complete an `n`-gram the answer already has.
fn pick(row: &[f32], so_far: &[u32], n: usize) -> u32 {
    let banned = banned_after(so_far, n);
    if banned.is_empty() {
        return argmax(row);
    }
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, x) in row.iter().enumerate() {
        if *x > best.1 && !banned.contains(&(i as u32)) {
            best = (i, *x);
        }
    }
    best.0 as u32
}

/// The tokens that followed earlier occurrences of the answer's last `n - 1` tokens.
fn banned_after(ids: &[u32], n: usize) -> Vec<u32> {
    let Some(tail) = n.checked_sub(1).filter(|k| *k >= 1 && ids.len() >= n).and_then(|k| ids.len().checked_sub(k)).and_then(|s| ids.get(s..)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for w in ids.windows(n) {
        if let Some((last, head)) = w.split_last()
            && head == tail
            && !out.contains(last)
        {
            out.push(*last);
        }
    }
    out
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, x) in v.iter().enumerate() {
        if *x > best.1 {
            best = (i, *x);
        }
    }
    best.0 as u32
}

/// The answer is going round in circles: its last 24 tokens repeat the 24 before them, or its
/// last three tokens came five times in its last 60 (a list of near-repeats: `"Woop", "Woot",
/// "Wool", …`, which the n-gram ban does not catch). The answer is cut there; the JSON reader
/// keeps what came before (the prompt asks for the lists last).
fn looping(ids: &[u32]) -> bool {
    const N: usize = 24;
    let last = ids.len().checked_sub(N).and_then(|i| ids.get(i..));
    let before = ids.len().checked_sub(2 * N).and_then(|i| ids.get(i..i + N));
    if last.is_some() && last == before {
        return true;
    }
    let Some(tail) = ids.len().checked_sub(3).and_then(|i| ids.get(i..)) else { return false };
    let recent = ids.get(ids.len().saturating_sub(60)..).unwrap_or_default();
    recent.windows(3).filter(|w| *w == tail).count() >= 5
}

/// Whether a JSON object has been opened and closed again, following string quoting.
struct JsonState {
    depth: i32,
    opened: bool,
    in_string: bool,
    escaped: bool,
}

impl JsonState {
    fn new(prefix: &str) -> Self {
        let mut s = JsonState { depth: 0, opened: false, in_string: false, escaped: false };
        s.feed(prefix);
        s
    }

    fn feed(&mut self, piece: &str) {
        for c in piece.chars() {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if c == '\\' {
                    self.escaped = true;
                } else if c == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match c {
                '"' => self.in_string = true,
                '{' => {
                    self.depth += 1;
                    self.opened = true;
                }
                '}' => self.depth -= 1,
                _ => {}
            }
        }
    }

    fn closed(&self) -> bool {
        self.opened && self.depth <= 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_sizes_are_multiples_within_the_budget() {
        let (w, h) = smart_size(1920, 1080, 32, 200_000);
        assert_eq!((w % 32, h % 32), (0, 0));
        assert!(w * h <= 200_000 && w > h, "{w}x{h}");
        let (w, h) = smart_size(1080, 1920, 32, 200_000);
        assert!(h > w && w * h <= 200_000);
        // tiny pictures grow to the minimum
        let (w, h) = smart_size(40, 30, 32, 200_000);
        assert!(w * h >= MIN_PIXELS && w % 32 == 0);
        // absurd shapes and sizes stay bounded
        let (w, h) = smart_size(100_000, 1, 32, 200_000);
        assert!(w * h <= 200_000 && w >= 32 && h >= 32);
        let (w, h) = smart_size(0, 0, 32, 0);
        assert!(w >= 32 && h >= 32);
    }

    #[test]
    fn json_state_follows_quotes() {
        let mut s = JsonState::new("{\"shot_type\": \"");
        assert!(!s.closed());
        s.feed("wide\", \"caption\": \"a } brace");
        assert!(!s.closed(), "a brace inside a string");
        s.feed("\\\" quoted\"}");
        assert!(s.closed());
        assert!(!JsonState::new("no json").closed());
    }

    #[test]
    fn loops_are_noticed() {
        let mut v: Vec<u32> = (0..10).collect();
        assert!(!looping(&v));
        let cyc: Vec<u32> = (0..24).collect();
        v.extend(&cyc);
        v.extend(&cyc);
        assert!(looping(&v));
        // near-repeats: `", "W` + something, five times
        let mut w: Vec<u32> = (100..120).collect();
        for k in 0..4 {
            w.extend([7, 8, 9, 200 + k]);
            assert!(!looping(&w), "{k}");
        }
        w.extend([7, 8, 9]);
        assert!(looping(&w));
        assert_eq!(argmax(&[0.1, 3.0, f32::NAN, 2.0]), 1);
        assert_eq!(argmax(&[]), 0);
        // a repeated 3-gram is not completed again: [1 2 3] 1 2 → 3 is banned
        assert_eq!(banned_after(&[1, 2, 3, 1, 2], 3), vec![3]);
        assert!(banned_after(&[1, 2, 3, 4], 3).is_empty());
        assert!(banned_after(&[1], 3).is_empty());
        assert_eq!(pick(&[0.0, 0.0, 0.0, 9.0, 5.0], &[1, 2, 3, 1, 2], 3), 4);
        assert_eq!(pick(&[0.0, 0.0, 0.0, 9.0, 5.0], &[], 3), 3);
    }
}

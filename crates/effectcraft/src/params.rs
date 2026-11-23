//! The checks of a render request that need no EffectCraft: the values' shape, the output size
//! and the frame count.

use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};
use serde_json::{Map, Value};

use crate::limits::{MAX_FRAMES, MAX_KEY_BYTES, MAX_PIXELS, MAX_SIDE, MAX_STRING_CHARS, MAX_VALUE_DEPTH, MAX_VALUE_NODES, MAX_VALUES};
use crate::{Error, Result};

/// The values of a request: absent or `null` is none; else a JSON object of at most
/// [`MAX_VALUES`] entries whose keys (control names or ids) are 1 to [`MAX_KEY_BYTES`] bytes,
/// nested at most [`MAX_VALUE_DEPTH`] deep, [`MAX_VALUE_NODES`] nodes in all, strings at most
/// [`MAX_STRING_CHARS`] characters. What each value means is the control's to say.
pub fn values(v: Option<&Value>) -> Result<Map<String, Value>> {
    let m = match v {
        None | Some(Value::Null) => return Ok(Map::new()),
        Some(Value::Object(m)) => m,
        Some(_) => return Err(Error::Values("`values` is an object: control name → value".into())),
    };
    if m.len() > MAX_VALUES {
        return Err(Error::TooLarge(format!("{} values; at most {MAX_VALUES}", m.len())));
    }
    let mut nodes = 0usize;
    for (k, v) in m {
        if k.is_empty() || k.len() > MAX_KEY_BYTES || k.contains('\0') {
            return Err(Error::Values(format!("a value's key is a control name or id of 1 to {MAX_KEY_BYTES} bytes")));
        }
        walk(k, v, 0, &mut nodes)?;
    }
    Ok(m.clone())
}

fn walk(key: &str, v: &Value, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    if *nodes > MAX_VALUE_NODES {
        return Err(Error::TooLarge(format!("the values hold more than {MAX_VALUE_NODES} JSON nodes")));
    }
    if depth > MAX_VALUE_DEPTH {
        return Err(Error::Values(format!("the value of `{key}` nests deeper than {MAX_VALUE_DEPTH}")));
    }
    match v {
        Value::String(s) if s.chars().count() > MAX_STRING_CHARS => {
            Err(Error::TooLarge(format!("the value of `{key}` is longer than {MAX_STRING_CHARS} characters")))
        }
        Value::Number(n) if !n.as_f64().is_some_and(f64::is_finite) => Err(Error::Values(format!("the value of `{key}` is not a finite number"))),
        Value::Array(a) => a.iter().try_for_each(|x| walk(key, x, depth + 1, nodes)),
        Value::Object(o) => o.iter().try_for_each(|(k, x)| {
            if k.len() > MAX_KEY_BYTES {
                Err(Error::Values(format!("a key inside `{key}` is longer than {MAX_KEY_BYTES} bytes")))
            } else {
                walk(key, x, depth + 1, nodes)
            }
        }),
        _ => Ok(()),
    }
}

/// The output size and the render scale for a comp of `comp` pixels: `want`, else the comp's
/// own size. `want` keeps the comp's shape (each side as the scale rounds it, as EffectCraft
/// sizes a frame); each side at most [`MAX_SIDE`], at most [`MAX_PIXELS`] in all.
pub fn output_size(comp: (u32, u32), want: Option<(u32, u32)>) -> Result<((u32, u32), f64)> {
    let (cw, ch) = comp;
    if cw == 0 || ch == 0 {
        return Err(Error::Template(format!("the composition is {cw}x{ch}")));
    }
    let (size, scale) = match want {
        None => ((cw, ch), 1.0),
        Some((w, h)) => {
            if w == 0 || h == 0 {
                return Err(Error::Values("`size` sides are at least 1".into()));
            }
            let scale = f64::from(w) / f64::from(cw);
            let side = |s: u32| ((f64::from(s) * scale).round().max(1.0)) as u32;
            if (side(cw), side(ch)) != (w, h) {
                return Err(Error::Values(format!("`size` {w}x{h} does not keep the composition's shape ({cw}x{ch}): {w}x{} does", side(ch))));
            }
            ((w, h), scale)
        }
    };
    let (w, h) = size;
    if w > MAX_SIDE || h > MAX_SIDE || u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(Error::TooLarge(format!("a {w}x{h} frame is too large: each side at most {MAX_SIDE}, at most {MAX_PIXELS} pixels; give a smaller `size`")));
    }
    Ok((size, scale))
}

/// The frames from `start` to `end` (comp time) at `rate`: those whose first tick (as
/// [`FrameRate::tick_of`] places it) lies before `end`; at least one, at most [`MAX_FRAMES`].
pub fn frame_count(start: Tick, end: Tick, rate: FrameRate) -> Result<u64> {
    let rate = rate.sane();
    let d = i128::from(end.0) - i128::from(start.0);
    if d <= 0 {
        return Err(Error::Values("the range to render is empty".into()));
    }
    // tick_of(i) < d  ⟺  ceil(i·TPS·den/num) ≤ d − 1  ⟺  i ≤ (d − 1)·num / (TPS·den)
    let n = (d - 1).saturating_mul(i128::from(rate.num)) / (i128::from(TICKS_PER_SECOND) * i128::from(rate.den)) + 1;
    match u64::try_from(n) {
        Ok(n) if n <= MAX_FRAMES => Ok(n),
        _ => Err(Error::TooLarge(format!("{n} frames; at most {MAX_FRAMES} in one render: render a shorter `range`"))),
    }
}

/// The range rendered: `start` and `end` (comp time, each `None` for the comp's own) held to the
/// comp's `duration`.
pub fn range(start: Option<Tick>, end: Option<Tick>, duration: Tick) -> Result<(Tick, Tick)> {
    if duration <= Tick::ZERO {
        return Err(Error::Template("the composition has no duration".into()));
    }
    let s = start.unwrap_or(Tick::ZERO);
    let e = end.unwrap_or(duration);
    if s < Tick::ZERO || s >= duration {
        return Err(Error::Values(format!("`range.start` must lie inside the composition (0 to {:.3} s)", duration.seconds())));
    }
    if e <= s || e > duration {
        return Err(Error::Values(format!("`range.end` must lie after the start and at most at the composition's end ({:.3} s)", duration.seconds())));
    }
    Ok((s, e))
}

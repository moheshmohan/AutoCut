// FilmCraft content-aware wipe: original work, MIT OR Apache-2.0.
//
// A wipe whose edge bends round what stands out in the two pictures. The host carves seams
// through their gradient energy once per transition (Avidan & Shamir 2007; see
// `filmcraft_render::transitions::content`) and hands over when the wipe reaches each cell of
// the frame (`getField(at).x`, 0..1, growing in the wipe's direction); the wipe follows it as a
// luma wipe follows its map, so it starts at progress 0 and is complete at 1.
// Parameters: host.p0.x = softness (0..1 of the wipe).

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let softness = host.p0.x;
    let m = getField(uv_px(uv)).x;
    let k = smooth_step(m, m + softness, progress * (1.0 + softness));
    return mix(getFromColor(uv), getToColor(uv), vec4(k));
}

// Author: Zeh Fernando
// License: MIT
// Description: Zooms and rotates into a bright flash, then out to the incoming image
// Tags: zoom, rotate, flash
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/DreamyZoom.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = rotation (degrees, default 6), host.p0.y = scale (multiplier, default 1.2).
// FilmCraft: the flash is light added to the premultiplied colour with alpha kept at most 1
// (upstream adds it to all four channels), and the angle at the exact centre is pinned to 0.

// Definitions --------
// upstream's value, kept for the same look (it is π/80, not 1/180*PI)
const DEG2RAD: f32 = 0.03926990816987241548078304229099;

// The code proper --------

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let rotation = host.p0.x;
    let scale = host.p0.y;
    // Massage parameters
    let first = progress < 0.5;
    let phase = select((progress - 0.5) * 2.0, progress * 2.0, first);
    let angleOffset = select(mix(-rotation * DEG2RAD, 0.0, phase), mix(0.0, rotation * DEG2RAD, phase), first);
    let newScale = select(mix(scale, 1.0, phase), mix(1.0, scale, phase), first);

    let center = vec2(0.0, 0.0);

    // Calculate the source point
    var p = (uv.xy - vec2(0.5, 0.5)) / newScale * vec2(ratio, 1.0);

    // This can probably be optimized (with distance())
    let angle = atan2_safe(p.y, p.x) + angleOffset;
    let dist = distance(center, p);
    p.x = cos(angle) * dist / ratio + 0.5;
    p.y = sin(angle) * dist + 0.5;
    var c: vec4<f32>;
    if first {
        c = getFromColor(p);
    } else {
        c = getToColor(p);
    }

    // Finally, apply the color
    let light = select(mix(1.0, 0.0, phase), mix(0.0, 1.0, phase), first);
    return vec4(c.rgb + vec3(light), min(c.a + light, 1.0));
}

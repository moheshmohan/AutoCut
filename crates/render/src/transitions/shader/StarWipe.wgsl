// Author: Ben Lucas
// License: MIT
// Description: A growing star reveals the incoming image
// Tags: wipe, shape, star
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/StarWipe.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = border_thickness (of the star's border, default 0.01), host.p0.y =
// star_rotation (default 0.75), host.p0.zw = star_center (uv, default (0.5, 0.5); the host takes
// it from the Center point), host.p1 = border_color (linear premultiplied, default white).
// FilmCraft: the angle at the exact centre is pinned to 0.

const STAR_ANGLE: f32 = 1.2566370614359172;

fn rotate(v: vec2<f32>, theta: f32) -> vec2<f32> {
    let cosTheta = cos(theta);
    let sinTheta = sin(theta);

    return vec2(
        cosTheta * v.x - sinTheta * v.y,
        sinTheta * v.x + cosTheta * v.y
    );
}

fn inStar(uv: vec2<f32>, center: vec2<f32>, radius: f32) -> bool {
    let star_rotation = host.p0.y;
    var uv_centered = uv - center;
    uv_centered = rotate(uv_centered, star_rotation * STAR_ANGLE);
    let theta = atan2_safe(uv_centered.y, uv_centered.x) + PI;

    let uv_rotated = rotate(uv_centered, -STAR_ANGLE * (floor(theta / STAR_ANGLE) + 0.5));

    let slope = 0.3;
    if uv_rotated.y > 0.0 {
        return (radius + uv_rotated.x * slope > uv_rotated.y);
    }
    return (-radius - uv_rotated.x * slope < uv_rotated.y);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let border_thickness = host.p0.x;
    let star_center = host.p0.zw;
    let border_color = host.p1;
    let progressScaled = (2.0 * border_thickness + 1.0) * progress - border_thickness;
    if inStar(uv, star_center, progressScaled) {
        return getToColor(uv);
    } else if inStar(uv, star_center, progressScaled + border_thickness) {
        return border_color;
    }
    return getFromColor(uv);
}

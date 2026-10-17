// Author: Fernando Kuteken
// License: MIT
// Description: Both images turn into hexagons, crossfade, then sharpen again
// Tags: pixelate, hexagon, blend
// Hexagonal math from: http://www.redblobgames.com/grids/hexagons/
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/hexagonalize.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = steps (number of steps, 0 for a continuous effect; default 50, a whole
// number), host.p0.y = horizontalHexagons (number of hexagons across, default 20, at least 1).
// FilmCraft: sqrt(3) / 3 and sqrt(3) / 2 are written as constants, so the CPU and the GPU round
// them the same way.

const SQRT3: f32 = 1.7320508075688772;
const SQRT3_3: f32 = 0.5773502691896258;
const SQRT3_2: f32 = 0.8660254037844386;

struct Hexagon {
    q: f32,
    r: f32,
    s: f32,
}

fn createHexagon(q: f32, r: f32) -> Hexagon {
    return Hexagon(q, r, -q - r);
}

fn roundHexagon(hex: Hexagon) -> Hexagon {
    var q = floor(hex.q + 0.5);
    var r = floor(hex.r + 0.5);
    var s = floor(hex.s + 0.5);

    let deltaQ = abs(q - hex.q);
    let deltaR = abs(r - hex.r);
    let deltaS = abs(s - hex.s);

    if deltaQ > deltaR && deltaQ > deltaS {
        q = -r - s;
    } else if deltaR > deltaS {
        r = -q - s;
    } else {
        s = -q - r;
    }

    return createHexagon(q, r);
}

fn hexagonFromPoint(point_in: vec2<f32>, size: f32) -> Hexagon {
    var point = point_in;
    point.y /= ratio;
    point = (point - 0.5) / size;

    let q = SQRT3_3 * point.x + (-1.0 / 3.0) * point.y;
    let r = 0.0 * point.x + 2.0 / 3.0 * point.y;

    let hex = createHexagon(q, r);
    return roundHexagon(hex);
}

fn pointFromHexagon(hex: Hexagon, size: f32) -> vec2<f32> {
    let x = (SQRT3 * hex.q + SQRT3_2 * hex.r) * size + 0.5;
    let y = (0.0 * hex.q + (3.0 / 2.0) * hex.r) * size + 0.5;

    return vec2(x, y * ratio);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let steps = i32(host.p0.x);
    let horizontalHexagons = host.p0.y;

    var dist = 2.0 * min(progress, 1.0 - progress);
    if steps > 0 {
        dist = ceil(dist * f32(steps)) / f32(steps);
    }

    let size = SQRT3_3 * dist / horizontalHexagons;

    var point = uv;
    if dist > 0.0 {
        point = pointFromHexagon(hexagonFromPoint(uv, size), size);
    }

    return mix(getFromColor(point), getToColor(point), vec4(progress));
}

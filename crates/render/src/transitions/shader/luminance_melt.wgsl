// Author: 0gust1
// License: MIT
// Description: The outgoing image melts away in noisy vertical streaks, depending on its brightness
// Tags: melt, noise, luma
//My own first transition — based on crosshatch code (from pthrasher), using  simplex noise formula (copied and pasted)
//-> cooler with high contrasted images (isolated dark subject on light background f.e.)
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/luminance_melt.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = direction (1 = down, upstream's true, the default; 0 = up),
// host.p0.y = l_threshold (luminance threshold, default 0.8), host.p0.z = above (1 = melt the
// parts above the threshold instead of below; default 0).
// FilmCraft: the per-column `rand(vec2(p.x, 0.1))` is the host's integer hash of the output
// pixel column (identical on CPU and GPU); the luminance is that of the linear colour the host
// works in; the progress == 0 / 1 branches are the host's.

// Simplex noise :
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : MIT
//               2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_2(x: vec2<f32>) -> vec2<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec3<f32>) -> vec3<f32> {
    return mod289_3(((x * 34.0) + 1.0) * x);
}

fn snoise(v: vec2<f32>) -> f32 {
    let C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                 0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                 -0.577350269189626, // -1.0 + 2.0 * C.x
                 0.024390243902439); // 1.0 / 41.0
    // First corner
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);

    // Other corners
    let i1 = select(vec2(0.0, 1.0), vec2(1.0, 0.0), x0.x > x0.y);
    let x12_ = x0.xyxy + C.xxzz;
    let x12 = vec4(x12_.xy - i1, x12_.zw);

    // Permutations
    i = mod289_2(i); // Avoid truncation effects in permutation
    let p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));

    var m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.0));
    m = m * m;
    m = m * m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    let x = 2.0 * fract(p * C.www) - 1.0;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    // Compute final noise value at P
    let gx = a0.x * x0.x + h.x * x0.y;
    let gyz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, vec3(gx, gyz));
}

// Simplex noise -- end

fn luminance(color: vec4<f32>) -> f32 {
    //(0.299*R + 0.587*G + 0.114*B)
    return color.r * 0.299 + color.g * 0.587 + color.b * 0.114;
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let direction = host.p0.x > 0.5;
    let l_threshold = host.p0.y;
    let above = host.p0.z > 0.5;
    let center = vec2(1.0, select(0.0, 1.0, direction));
    let p = uv;
    let x = progress;
    let dist = distance(center, p) - progress * exp(snoise(vec2(p.x, 0.0)));
    let r = x - rand_column(p, 3u);
    let fromColor = getFromColor(p);
    var m = progress * progress * progress;
    if above {
        if dist <= r && luminance(fromColor) > l_threshold {
            m = 1.0;
        }
    } else {
        if dist <= r && luminance(fromColor) < l_threshold {
            m = 1.0;
        }
    }
    return mix(fromColor, getToColor(p), vec4(m));
}

// FilmCraft luma-mask wipe engine: original work, MIT OR Apache-2.0.
//
// One shader behind every Luma Wipe (`filmcraft_project::transition_engines`): the incoming
// picture reaches each pixel when the progress passes the pixel's value in the generated map
// (`getMap`, made by the host from the preset's shape, already feathered and stretched to 0..1).
// Softness widens the moving edge to a band of map values; Invert reverses the order.
// Parameters: host.p0.x = softness (0..1 of the map's range), host.p0.y = invert (0 or 1).

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let softness = host.p0.x;
    var m = getMap(uv);
    if host.p0.y > 0.5 {
        m = 1.0 - m;
    }
    // the band [m, m + softness] is crossed while progress runs 0..1, so the wipe starts at 0
    // and is complete at 1 whatever the softness
    let k = smooth_step(m, m + softness, progress * (1.0 + softness));
    return mix(getFromColor(uv), getToColor(uv), vec4(k));
}

// Radial ("studio") gradient background.
//
// A three-stop radial ramp - inner -> mid -> outer - around a freely
// positioned center, plus an optional multiplicative vignette towards the
// corners of the frame. Distances are measured in units of the canvas width
// so the gradient stays circular (and `radius` keeps its meaning) whatever
// the window's aspect ratio.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Define positions at the four corners of the viewport, at the largest depth
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    // Get the current ndc position, and the "virtual" ndc pos. Usually the same
    // except when camera.set_view_offset() is used.
    let pos = positions[i32(in.index)];
    let virtual_pos = pos * u_stdinfo.ndc_offset.xy + u_stdinfo.ndc_offset.zw;

    var varyings: Varyings;
    varyings.position = vec4<f32>(pos, 0.9999999, 1.0);
    varyings.texcoord = vec2<f32>(virtual_pos * 0.5 + 0.5);
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    let uv = varyings.texcoord.xy;  // 0-1 across the canvas, origin bottom left
    let aspect = u_stdinfo.logical_size.x / max(u_stdinfo.logical_size.y, 1.0);

    // The center is given in image coordinates (y measured from the top)
    let center = vec2<f32>(u_material.center.x, 1.0 - u_material.center.y);

    // Scaling y by the aspect ratio puts the offset into units of the canvas
    // width, i.e. the iso-lines are circles rather than ellipses
    let d = length(vec2<f32>(uv.x - center.x, (uv.y - center.y) / aspect));
    let t = clamp(d / max(u_material.radius, 1e-6), 0.0, 1.0);

    // `falloff` shapes the ramp: > 1 holds the core bright and squeezes the
    // transition towards the rim, < 1 drops off immediately and trails out.
    // The smoothstep flattens both ends so neither the center nor the point
    // where the gradient meets the outer color shows a seam.
    let w = smoothstep(0.0, 1.0, 1.0 - pow(t, max(u_material.falloff, 1e-3)));

    // Three stops: outer at w=0, mid at w=0.5, inner at w=1
    var color = select(
        mix(u_material.color_outer, u_material.color_mid, w * 2.0),
        mix(u_material.color_mid, u_material.color_inner, (w - 0.5) * 2.0),
        w > 0.5,
    );

    // Vignette: darken towards the corners of the frame (not of the gradient)
    if (u_material.vignette > 0.0) {
        let q = vec2<f32>(uv.x - 0.5, (uv.y - 0.5) / aspect);
        let r = length(q) / length(vec2<f32>(0.5, 0.5 / aspect));  // 1 at the corners
        let v = 1.0 - u_material.vignette * smoothstep(0.2, 1.0, r);
        color = vec4<f32>(color.rgb * v, color.a);
    }

    // Make physical color with combined alpha
    let physical_color = srgb2physical(color.rgb);
    let opacity = color.a * u_material.opacity;

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, opacity);
    $$ if write_pick
        // As for pygfx's background: no extra information in the pick info
        out.pick = pick_pack(u32(u_wobject.global_id), 20);
    $$ endif
    return out;
}

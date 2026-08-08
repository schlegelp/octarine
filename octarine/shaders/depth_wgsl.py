"""WGSL snippets for reading the depth buffer in post-processing passes.

pygfx hands effect passes the rendered color and depth textures, and nothing
else - no G-buffer with view-space positions or normals. Any pass that wants
to reason about the geometry (ambient occlusion, outlines, ...) therefore has
to reconstruct that information from depth, which is what these snippets do.

Each of them is a string that gets prepended to a pass' own WGSL. They assume
a `depthTex` binding and a `projection_transform_inv` uniform; see
`octarine.shaders.ao` and `octarine.shaders.outline` for how they are used.
"""

#: Reading the depth buffer and linearizing it to a view-space distance.
DEPTH_WGSL = """
    fn depth_at(idx: vec2i) -> f32 {
        let dims = vec2i(textureDimensions(depthTex));
        return textureLoad(depthTex, clamp(idx, vec2i(0), dims - 1), 0);
    }

    // Distance to the camera in world units. The depth buffer itself is
    // non-linear, and the x/y of the NDC do not affect the result. Note
    // that this is negative behind the camera - which does happen, since
    // pygfx parks orthographic cameras in the middle of the scene.
    fn to_view_depth(depth: f32) -> f32 {
        let h = u_effect.projection_transform_inv * vec4f(0.0, 0.0, depth, 1.0);
        return -h.z / h.w;
    }
"""

#: Full view-space positions from depth. Requires `DEPTH_WGSL`.
VIEW_POS_WGSL = """
    // Full view-space position of a depth value at a texture coordinate.
    // The camera looks down -z, so +z is towards the camera.
    fn to_view_pos(uv: vec2f, depth: f32) -> vec3f {
        let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        let h = u_effect.projection_transform_inv * ndc;
        return h.xyz / h.w;
    }

    fn view_pos_at(idx: vec2i) -> vec3f {
        let dims = vec2i(textureDimensions(depthTex));
        let i = clamp(idx, vec2i(0), dims - 1);
        return to_view_pos((vec2f(i) + 0.5) / vec2f(dims), textureLoad(depthTex, i, 0));
    }
"""

#: View-space normals from depth. Requires `DEPTH_WGSL` and `VIEW_POS_WGSL`.
VIEW_NORMAL_WGSL = """
    // Reconstruct the view-space normal at a pixel. For each axis we use the
    // neighbour whose depth extrapolates best towards the center pixel,
    // which keeps the frame from smearing across silhouettes. Method from
    // https://atyuwen.github.io/posts/normal-reconstruction/ - the same one
    // pygfx's `NormalPass` uses.
    fn view_normal_at(idx: vec2i, depth: f32, pos: vec3f) -> vec3f {
        let h = vec4f(
            depth_at(idx + vec2i(-1, 0)), depth_at(idx + vec2i(1, 0)),
            depth_at(idx + vec2i(-2, 0)), depth_at(idx + vec2i(2, 0)));
        let v = vec4f(
            depth_at(idx + vec2i(0, -1)), depth_at(idx + vec2i(0, 1)),
            depth_at(idx + vec2i(0, -2)), depth_at(idx + vec2i(0, 2)));

        // Error of extrapolating each pair towards the center pixel
        let he = abs((2.0 * h.xy - h.zw) - depth);
        let ve = abs((2.0 * v.xy - v.zw) - depth);

        // Difference vectors, all oriented through the center pixel: `l`/`r`
        // point right (+x in view space), `u`/`d` up the screen (+y in view
        // space, as the texture coordinate's y axis points down)
        let l = pos - view_pos_at(idx + vec2i(-1, 0));
        let r = view_pos_at(idx + vec2i(1, 0)) - pos;
        let u = view_pos_at(idx + vec2i(0, -1)) - pos;
        let d = pos - view_pos_at(idx + vec2i(0, 1));

        let h_deriv = select(r, l, he.x < he.y);
        let v_deriv = select(d, u, ve.x < ve.y);
        return normalize(cross(h_deriv, v_deriv));
    }
"""

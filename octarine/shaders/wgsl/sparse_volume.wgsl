// Sparse volume rendering via two-level (brick map) raycasting.
//
// The volume is defined by two textures:
//  - t_coarse: dense low-res grid over the bounding box; texel = atlas slot + 1, 0 = empty
//  - t_atlas:  occupied bricks of size {{ brick_size }} packed side by side; each
//    atlas cell carries a 1-voxel apron of neighboring voxels so trilinear
//    sampling is seamless across brick borders
//
// The fragment shader marches the coarse grid with a DDA-style loop, skipping
// empty cells analytically (one texture load per {{ brick_size }} voxels of
// empty space) and only fine-sampling inside occupied bricks.
//
// Ray setup follows pygfx's volume_ray.wgsl so that camera handling
// (perspective/ortho, camera inside the volume) behaves identically.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.image_sample.wgsl' $}

const BRICK_SIZE: f32 = {{ brick_size }}.0;
// Atlas cells are the brick plus a 1-voxel apron on each side
const CELL_SIZE: f32 = BRICK_SIZE + 2.0;
const MAX_COARSE_STEPS: i32 = 4096;
const MAX_FINE_STEPS: i32 = 512;

fn get_vol_shape() -> vec3<f32> {
    return vec3<f32>({{ shape_x }}.0, {{ shape_y }}.0, {{ shape_z }}.0);
}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Implicit cube geometry spanning the volume, mirroring volume_ray.wgsl.
    // Voxel centers sit at integer coordinates, so the cube spans
    // -0.5 .. shape - 0.5.
    var indices = array<i32,36>(
        0, 1, 2,   3, 2, 1,   4, 5, 6,   7, 6, 5,   6, 7, 3,   2, 3, 7,
        1, 0, 4,   5, 4, 0,   5, 0, 7,   2, 7, 0,   1, 4, 3,   6, 3, 4,
    );

    let pos1 = vec3<f32>(-0.5);
    let pos2 = get_vol_shape() + pos1;

    var positions = array<vec3<f32>,8>(
        vec3<f32>(pos2.x, pos1.y, pos2.z),
        vec3<f32>(pos2.x, pos1.y, pos1.z),
        vec3<f32>(pos2.x, pos2.y, pos2.z),
        vec3<f32>(pos2.x, pos2.y, pos1.z),
        vec3<f32>(pos1.x, pos1.y, pos1.z),
        vec3<f32>(pos1.x, pos1.y, pos2.z),
        vec3<f32>(pos1.x, pos2.y, pos1.z),
        vec3<f32>(pos1.x, pos2.y, pos2.z),
    );

    let index = i32(in.vertex_index);
    let i0 = indices[index];

    let data_pos = vec4<f32>(positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);

    // Position on the back face of the cube (front faces are culled), in
    // data coordinates (voxels).
    varyings.data_back_pos = vec4<f32>(data_pos);

    // Take care to take into account of the camera flipping any axii.
    // Note: pygfx's volume_ray.wgsl uses the product of the view matrix
    // diagonal here, but that is only a valid proxy for the handedness with
    // axis-aligned-ish cameras - for corner-on orientations (with roll) the
    // diagonal product of a pure rotation can be <= 0, which swaps the
    // reconstructed near/far positions and makes the volume vanish. Use the
    // actual determinant of the rotation part instead.
    let m = u_stdinfo.cam_transform;
    let cam_sign = sign(dot(m[0].xyz, cross(m[1].xyz, m[2].xyz)));

    // NDC positions on the near and far clipping planes, transformed back to
    // data coordinates; used to construct the view ray in the fragment shader.
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    return varyings;
}


// Exit distance (along the ray, measured from p0) out of the given coarse cell.
fn cell_exit_t(p0: vec3<f32>, inv_ray: vec3<f32>, cell: vec3<i32>) -> f32 {
    let cmin = vec3<f32>(cell) * BRICK_SIZE;
    let cmax = cmin + vec3<f32>(BRICK_SIZE);
    let t1 = (cmin - p0) * inv_ray;
    let t2 = (cmax - p0) * inv_ray;
    let t_far = max(t1, t2);
    return min(t_far.x, min(t_far.y, t_far.z));
}

// Origin (in atlas voxels) of the atlas cell in the given slot (0-based).
fn brick_origin(slot: u32) -> vec3<f32> {
    let ab = vec3<u32>(textureDimensions(t_atlas)) / u32(CELL_SIZE);
    let x = slot % ab.x;
    let y = (slot / ab.x) % ab.y;
    let z = slot / (ab.x * ab.y);
    return vec3<f32>(vec3<u32>(x, y, z)) * CELL_SIZE;
}

// Sample the atlas for a brick at `slot`, at position `p` (corner-origin
// voxel coordinates in the volume) inside coarse cell `cell`. The +1 offset
// skips the apron layer; thanks to the apron, trilinear interpolation is
// valid over the whole brick (the clamp is a no-op safety net).
fn sample_brick(slot: u32, cell: vec3<i32>, p: vec3<f32>) -> f32 {
    let local = clamp(
        p - vec3<f32>(cell) * BRICK_SIZE + 1.0,
        vec3<f32>(0.5),
        vec3<f32>(CELL_SIZE - 0.5),
    );
    let uvw = (brick_origin(slot) + local) / vec3<f32>(textureDimensions(t_atlas));
    return textureSampleLevel(t_atlas, s_atlas, uvw, 0.0).r;
}

// Sample the volume at an arbitrary position (used by the MIP refinement,
// whose sub-steps may cross into a neighboring coarse cell).
fn sample_volume(p: vec3<f32>, coarse_dim: vec3<i32>) -> f32 {
    let cell = clamp(vec3<i32>(floor(p / BRICK_SIZE)), vec3<i32>(0), coarse_dim - 1);
    let slot_plus1 = textureLoad(t_coarse, cell, 0).r;
    if (slot_plus1 == 0u) { return 0.0; }
    return sample_brick(slot_plus1 - 1u, cell, p);
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let sizef = get_vol_shape();

    // Positions in data coordinates
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

    // Unit vector pointing in the view direction through this fragment
    var ray = normalize(far_pos - near_pos);
    // Avoid divisions by zero in the DDA
    ray.x = select(ray.x, 1e-6, abs(ray.x) < 1e-6);
    ray.y = select(ray.y, 1e-6, abs(ray.y) < 1e-6);
    ray.z = select(ray.z, 1e-6, abs(ray.z) < 1e-6);
    let inv_ray = 1.0 / ray;

    // Signed distance from back_pos to the first position that must be
    // sampled (either on a front face or on the near plane), in voxels.
    var dist = dot(near_pos - back_pos, ray);
    dist = max(dist, min((-0.5 - back_pos.x) * inv_ray.x, (sizef.x - 0.5 - back_pos.x) * inv_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) * inv_ray.y, (sizef.y - 0.5 - back_pos.y) * inv_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) * inv_ray.z, (sizef.z - 0.5 - back_pos.z) * inv_ray.z));

    let t_len = -dist;
    if (t_len <= 0.0) { discard; }

    let front_pos = back_pos + ray * dist;

    // Work in corner-origin coordinates (0 .. shape) so that brick indices
    // are simple floor divisions.
    let p0 = front_pos + vec3<f32>(0.5);

    let coarse_dim = vec3<i32>(textureDimensions(t_coarse));
    let fine_step = clamp(u_material.step_size, 0.01, BRICK_SIZE);

    // Per-pixel jitter of the sampling phase: decorrelates neighboring rays
    // so that the fixed-step march produces unobtrusive noise instead of
    // coherent moire banding. Stable per pixel (hash of the frag coord).
    let jitter = fract(sin(dot(varyings.position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);

    // ---- Two-level ray march ----

    $$ if mode == 'mip'
        var the_val = -1.0;
        var the_t = 0.0;
    $$ else
        // Front-to-back emission/absorption accumulators (premultiplied)
        var acc_rgb = vec3<f32>(0.0);
        var acc_a = 0.0;
        var the_t = -1.0;  // depth of first contributing sample
    $$ endif

    var t = 0.0;
    for (var i = 0; i < MAX_COARSE_STEPS; i += 1) {
        if (t >= t_len) { break; }
        let p = p0 + ray * t;
        let cell = clamp(
            vec3<i32>(floor(p / BRICK_SIZE)),
            vec3<i32>(0),
            coarse_dim - 1,
        );
        let t_exit = cell_exit_t(p0, inv_ray, cell);
        let slot_plus1 = textureLoad(t_coarse, cell, 0).r;

        if (slot_plus1 != 0u) {
            let slot = slot_plus1 - 1u;
            let t_end = min(t_exit, t_len);
            t += jitter * fine_step;
            for (var j = 0; j < MAX_FINE_STEPS; j += 1) {
                if (t >= t_end) { break; }
                let val = sample_brick(slot, cell, p0 + ray * t);
                $$ if mode == 'mip'
                    if (val > the_val) {
                        the_val = val;
                        the_t = t;
                    }
                $$ else
                    if (val > 0.0) {
                        let color = sampled_value_to_color(vec4<f32>(val, 0.0, 0.0, 1.0));
                        let a = color.a * u_material.opacity;
                        // Opacity correction for the step size
                        let a_step = 1.0 - pow(1.0 - clamp(a, 0.0, 0.9999), fine_step);
                        $$ if colorspace == 'srgb'
                            let physical_rgb = srgb2physical(color.rgb);
                        $$ else
                            let physical_rgb = color.rgb;
                        $$ endif
                        acc_rgb += (1.0 - acc_a) * a_step * physical_rgb;
                        acc_a += (1.0 - acc_a) * a_step;
                        if (the_t < 0.0) { the_t = t; }
                    }
                $$ endif
                t += fine_step;
            }
        }

        // Advance to just inside the next coarse cell; the max() + epsilon
        // guarantees strict progress even in degenerate corner cases (e.g.
        // clamped cells at the volume border at grazing angles).
        t = max(t_exit, t) + 0.01;

        $$ if mode == 'mip'
            if (the_val >= 1.0) { break; }  // cannot get any higher
        $$ else
            if (acc_a >= 0.95) { break; }  // early ray termination
        $$ endif
    }

    // ---- Resolve color and depth ----

    $$ if mode == 'mip'
        if (the_val <= 0.0) { discard; }
        // Refine value and location of the maximum with a divide-by-two
        // search (mirrors pygfx's volume shader); reduces banding that stems
        // from the maximum being quantized to the step positions.
        var substep = fine_step;
        for (var k = 0; k < 4; k += 1) {
            substep = substep * 0.5;
            let t1 = max(the_t - substep, 0.0);
            let t2 = min(the_t + substep, t_len);
            let v1 = sample_volume(p0 + ray * t1, coarse_dim);
            let v2 = sample_volume(p0 + ray * t2, coarse_dim);
            if (v1 >= the_val) {
                the_val = v1;
                the_t = t1;
            } else if (v2 > the_val) {
                the_val = v2;
                the_t = t2;
            }
        }
        let color = sampled_value_to_color(vec4<f32>(the_val, 0.0, 0.0, 1.0));
        $$ if colorspace == 'srgb'
            let physical_color = srgb2physical(color.rgb);
        $$ else
            let physical_color = color.rgb;
        $$ endif
        let out_color = vec4<f32>(physical_color, color.a * u_material.opacity);
    $$ else
        if (the_t < 0.0 || acc_a <= 0.0) { discard; }
        // Un-premultiply for the standard output convention
        let out_color = vec4<f32>(acc_rgb / max(acc_a, 1e-6), acc_a);
    $$ endif

    // Depth at the MIP maximum / first contributing sample, so that opaque
    // objects composite correctly with the volume.
    let hit_data_pos = front_pos + ray * the_t;
    let hit_world_pos = u_wobject.world_transform * vec4<f32>(hit_data_pos, 1.0);
    let hit_ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * hit_world_pos;

    do_alpha_test(out_color.a);

    var out: FragmentOutput;
    out.color = out_color;
    out.depth = hit_ndc_pos.z / hit_ndc_pos.w;

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    // Encode the hit position in (normalized) volume coordinates, matching
    // the convention of pygfx's volume shader.
    let pick_coord = clamp((hit_data_pos + vec3<f32>(0.5)) / sizef, vec3<f32>(0.0), vec3<f32>(1.0));
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(pick_coord.x * 16383.0), 14) +
        pick_pack(u32(pick_coord.y * 16383.0), 14) +
        pick_pack(u32(pick_coord.z * 16383.0), 14)
    );
    $$ endif
    return out;
}

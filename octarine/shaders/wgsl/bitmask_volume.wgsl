// Bitmask sparse volume rendering via two-level (super-brick / brick) raycasting.
//
// The volume is binary occupancy packed as one bit per voxel (see
// `.rle_packing`), indexed by a sparse two-level structure:
//  - t_super:    dense grid over the bbox at brick_size*8; texel = super + 1
//  - s_bricktab: per occupied super, a dense 8^3 table of brick slots
//  - s_bits:     per occupied brick, brick_size^3 / 32 words
//
// There is no apron: every tap of a trilinear sample resolves its own brick
// through the index, so samples cross brick borders correctly on their own.
//
// Ray setup mirrors sparse_volume.wgsl (and hence pygfx's volume_ray.wgsl) so
// that camera handling behaves identically across the sparse-volume shaders.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if mode == 'iso'
    {$ include 'pygfx.light_phong_simple.wgsl' $}
$$ endif

const BI: i32 = {{ brick_size }};
const BF: f32 = {{ brick_size }}.0;
const SI: i32 = 8;                       // bricks per super-brick edge
const WORDS: u32 = {{ words }}u;
const MAX_COARSE_STEPS: i32 = 8192;
const MAX_FINE_STEPS: i32 = 512;

fn get_vol_shape() -> vec3<f32> {
    return vec3<f32>({{ shape_x }}.0, {{ shape_y }}.0, {{ shape_z }}.0);
}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Implicit cube geometry spanning the volume. Voxel centers sit at integer
    // coordinates, so the cube spans -0.5 .. shape - 0.5.
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

    let i0 = indices[i32(in.vertex_index)];
    let data_pos = vec4<f32>(positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.data_back_pos = vec4<f32>(data_pos);

    // Use the determinant of the rotation part rather than the product of the
    // view-matrix diagonal (see sparse_volume.wgsl for why).
    let m = u_stdinfo.cam_transform;
    let cam_sign = sign(dot(m[0].xyz, cross(m[1].xyz, m[2].xyz)));
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    return varyings;
}


// Exit distance (along the ray, from p0) out of the given brick cell.
fn cell_exit_t(p0: vec3<f32>, inv_ray: vec3<f32>, cell: vec3<i32>) -> f32 {
    let cmin = vec3<f32>(cell) * BF;
    let t_far = max((cmin - p0) * inv_ray, (cmin + vec3<f32>(BF) - p0) * inv_ray);
    return min(t_far.x, min(t_far.y, t_far.z));
}

// Brick slot + 1 for a brick cell (0 = empty), via the two-level index.
fn brick_slot(cell: vec3<i32>) -> u32 {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>({{ gbx }}, {{ gby }}, {{ gbz }}))) {
        return 0u;
    }
    let sup = cell / SI;
    let s = textureLoad(t_super, sup, 0).r;
    if (s == 0u) { return 0u; }
    let l = cell % SI;
    return load_s_bricktab(i32((s - 1u) * 512u) + (l.z * SI + l.y) * SI + l.x);
}

// Occupancy of a single voxel: one index walk, one load, one bit test.
//
// A brick-clipped run list would halve the memory here but costs a binary
// search over the brick's runs instead (measured 1.7x slower at equal memory,
// 2.9x at its own optimum) - see the note in `.rle_packing`.
fn point_at(p: vec3<i32>) -> f32 {
    if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>({{ shape_x }}, {{ shape_y }}, {{ shape_z }}))) {
        return 0.0;
    }
    let slot = brick_slot(p / BI);
    if (slot == 0u) { return 0.0; }
    let l = p % BI;
    let idx = u32((l.z * BI + l.y) * BI + l.x);
    let w = load_s_bits(i32((slot - 1u) * WORDS + (idx >> 5u)));
    return f32((w >> (idx & 31u)) & 1u);
}

// Trilinear filtering from 8 point queries. Hardware filtering is unavailable
// (the payload is a bit array, not a texture), but the taps are adjacent bits
// and usually land in the same brick, so they share cache lines.
fn sample_tri(p: vec3<f32>) -> f32 {
    let pf = p - vec3<f32>(0.5);
    let i0 = vec3<i32>(floor(pf));
    let f = pf - floor(pf);
    let c00 = mix(point_at(i0), point_at(i0 + vec3<i32>(1, 0, 0)), f.x);
    let c10 = mix(point_at(i0 + vec3<i32>(0, 1, 0)), point_at(i0 + vec3<i32>(1, 1, 0)), f.x);
    let c01 = mix(point_at(i0 + vec3<i32>(0, 0, 1)), point_at(i0 + vec3<i32>(1, 0, 1)), f.x);
    let c11 = mix(point_at(i0 + vec3<i32>(0, 1, 1)), point_at(i0 + vec3<i32>(1, 1, 1)), f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

$$ if smooth
// Wider filter for the normal: the mean of 8 trilinear taps on the corners of
// a cube of half-width h. That convolves the trilinear tent with a box, so the
// field varies over ~2*(1 + h) voxels instead of 2 and its gradient stops
// tracking individual voxel faces.
//
// Only the *normal* uses this - the surface itself stays on the unsmoothed
// field. Widening the filter shrinks a binary volume (a feature of diameter D
// only peaks at ~1.0 while D > 2 + 2h; below that it drops away and the
// isosurface breaks up), so filtering the surface too would erase thin
// structures. Shading is what carries the voxel look, and this leaves the
// silhouette bit-exact.
fn sample_smooth(p: vec3<f32>) -> f32 {
    let h = u_material.smoothing;
    var s = 0.0;
    for (var i = 0; i < 8; i += 1) {
        let o = vec3<f32>(
            select(-h, h, (i & 1) != 0),
            select(-h, h, (i & 2) != 0),
            select(-h, h, (i & 4) != 0),
        );
        s += sample_tri(p + o);
    }
    return s * 0.125;
}
$$ endif


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let sizef = get_vol_shape();

    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

    var ray = normalize(far_pos - near_pos);
    ray.x = select(ray.x, 1e-6, abs(ray.x) < 1e-6);
    ray.y = select(ray.y, 1e-6, abs(ray.y) < 1e-6);
    ray.z = select(ray.z, 1e-6, abs(ray.z) < 1e-6);
    let inv_ray = 1.0 / ray;

    var dist = dot(near_pos - back_pos, ray);
    dist = max(dist, min((-0.5 - back_pos.x) * inv_ray.x, (sizef.x - 0.5 - back_pos.x) * inv_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) * inv_ray.y, (sizef.y - 0.5 - back_pos.y) * inv_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) * inv_ray.z, (sizef.z - 0.5 - back_pos.z) * inv_ray.z));

    let t_len = -dist;
    if (t_len <= 0.0) { discard; }

    let front_pos = back_pos + ray * dist;
    // Corner-origin coordinates (0 .. shape) so brick indices are floor divisions
    let p0 = front_pos + vec3<f32>(0.5);

    let fine_step = clamp(u_material.step_size, 0.01, BF);

    // ---- two-level ray march ----

    $$ if mode == 'density'
        var acc_a = 0.0;
    $$ endif
    var the_t = -1.0;   // < 0 = nothing hit yet

    var t = 0.0;
    for (var i = 0; i < MAX_COARSE_STEPS; i += 1) {
        if (t >= t_len) { break; }
        let p = p0 + ray * t;
        let cell = vec3<i32>(floor(p / BF));
        let t_exit = cell_exit_t(p0, inv_ray, cell);

        if (brick_slot(cell) != 0u) {
            let t_end = min(t_exit, t_len);
            for (var j = 0; j < MAX_FINE_STEPS; j += 1) {
                if (t >= t_end) { break; }
                // Detection is a raw bit test; only the refinement filters.
                let hit = point_at(vec3<i32>(floor(p0 + ray * t))) > 0.0;
                $$ if mode == 'density'
                    if (hit) {
                        // Beer-Lambert absorption over one step
                        let a_step = 1.0 - exp(-u_material.density * fine_step);
                        acc_a += (1.0 - acc_a) * a_step;
                        if (the_t < 0.0) { the_t = t; }
                    }
                $$ elif mode == 'iso'
                    if (hit) {
                        // Bracket the crossing and bisect on the filtered field
                        var t_lo = max(t - fine_step, 0.0);
                        var t_hi = t;
                        for (var k = 0; k < 6; k += 1) {
                            let t_mid = 0.5 * (t_lo + t_hi);
                            if (sample_tri(p0 + ray * t_mid) > u_material.threshold) {
                                t_hi = t_mid;
                            } else {
                                t_lo = t_mid;
                            }
                        }
                        the_t = t_hi;
                        break;
                    }
                $$ else
                    // "mip": the data is binary, so the first hit is the maximum
                    if (hit) { the_t = t; break; }
                $$ endif
                t += fine_step;
            }
        }

        t = max(t_exit, t) + 0.01;

        $$ if mode == 'density'
            if (acc_a >= 0.95) { break; }   // early ray termination
        $$ else
            if (the_t >= 0.0) { break; }
        $$ endif
    }

    // ---- resolve color and depth ----

    let base_rgb = srgb2physical(u_material.color.rgb);

    $$ if mode == 'iso'
        if (the_t < 0.0) { discard; }

        let hit_p = p0 + ray * the_t;

        // Surface normal from the gradient of the filtered field
        $$ if smooth
            // A difference much narrower than the filter reads nearly the same
            // value at both taps, which leaves noise rather than a direction -
            // so widen the stencil with the filter.
            let d = max(u_material.gradient_delta, 0.5 + u_material.smoothing);
            var grad = vec3<f32>(
                sample_smooth(hit_p + vec3<f32>(d, 0.0, 0.0)) - sample_smooth(hit_p - vec3<f32>(d, 0.0, 0.0)),
                sample_smooth(hit_p + vec3<f32>(0.0, d, 0.0)) - sample_smooth(hit_p - vec3<f32>(0.0, d, 0.0)),
                sample_smooth(hit_p + vec3<f32>(0.0, 0.0, d)) - sample_smooth(hit_p - vec3<f32>(0.0, 0.0, d)),
            );
        $$ else
            let d = u_material.gradient_delta;
            var grad = vec3<f32>(
                sample_tri(hit_p + vec3<f32>(d, 0.0, 0.0)) - sample_tri(hit_p - vec3<f32>(d, 0.0, 0.0)),
                sample_tri(hit_p + vec3<f32>(0.0, d, 0.0)) - sample_tri(hit_p - vec3<f32>(0.0, d, 0.0)),
                sample_tri(hit_p + vec3<f32>(0.0, 0.0, d)) - sample_tri(hit_p - vec3<f32>(0.0, 0.0, d)),
            );
        $$ endif
        // Structures thinner than `d` can leave a (near) zero gradient
        if (dot(grad, grad) < 1e-12) { grad = -ray; }

        // Lighting is evaluated in world space: `spacing` may scale the axes
        // non-uniformly, which does not preserve angles.
        let normal = normalize((transpose(u_wobject.world_transform_inv) * vec4<f32>(grad, 0.0)).xyz);
        let view_dir = normalize((u_wobject.world_transform * vec4<f32>(ray, 0.0)).xyz);
        let is_front = dot(normal, view_dir) > 0.0;
        let reoriented_normal = select(-normal, normal, is_front);
        let out_color = vec4<f32>(
            lighting_phong(reoriented_normal, view_dir, base_rgb),
            u_material.color.a * u_material.opacity,
        );
    $$ elif mode == 'density'
        if (the_t < 0.0 || acc_a <= 0.0) { discard; }
        let out_color = vec4<f32>(
            base_rgb, saturate(acc_a * u_material.color.a * u_material.opacity)
        );
    $$ else
        if (the_t < 0.0) { discard; }
        let out_color = vec4<f32>(base_rgb, u_material.color.a * u_material.opacity);
    $$ endif

    let hit_data_pos = front_pos + ray * the_t;
    let hit_world_pos = u_wobject.world_transform * vec4<f32>(hit_data_pos, 1.0);
    let hit_ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * hit_world_pos;

    do_alpha_test(out_color.a);

    var out: FragmentOutput;
    out.color = out_color;
    out.depth = hit_ndc_pos.z / hit_ndc_pos.w;

    $$ if write_pick
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

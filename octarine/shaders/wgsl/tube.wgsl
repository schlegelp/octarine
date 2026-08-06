// Parametric tube surfaces, generated entirely in the vertex shader.
//
// There is no vertex buffer and no index buffer: every position is synthesized
// from `vertex_index` alone (the same vertex-pulling trick sparse_volume.wgsl
// uses for its implicit cube). Two storage buffers define the tube:
//
//  - s_coefs: {{ stride }} floats per node - position (3), frame quaternion (4),
//    mean radius a0 (1), then {{ k_buf }} cosine and {{ k_buf }} sine
//    coefficients of the radial profile
//        r(theta) = a0 + sum_k [ a_k cos(k*theta) + b_k sin(k*theta) ]
//  - s_edges: (E, 2) node index pairs. A skeleton is a tree, not a chain, so
//    quads are swept per *edge*; sweeping between consecutive node indices
//    would stitch unrelated branches together.
//
// Each quad is one (edge, angular sector) cell of the surface, so the draw is
// 6 * E * n_theta vertices. `n_theta` and `k_max` live in the material uniform
// rather than in a template constant, which is what makes angular LOD a
// uniform write plus a smaller draw call - no upload, no reallocation, and the
// coefficient buffer never moves.
//
// The normal is truncated harder than the position, at `k_normal` rather than
// `k_max`, and that is deliberate. dr/dtheta weights harmonic k by k, so once
// the harmonic magnitudes flatten out at the rasterization floor - which they
// do, there is no knee in real arbor data - every extra harmonic contributes
// more slope than shape. The silhouette keeps improving with `k_max` while the
// shading gets worse: at k_normal = k_max = 4 the normal's median tilt off
// radial is ~35 degrees and its p95 is ~69, which is enough for dot(view, n)
// to go negative on camera-facing fragments and leave dark patches. That is
// honest geometry rather than a bug - a neurite of ~5-voxel radius really is
// that bumpy - but it is not what you want to shade. k_normal = 1 is the
// default, k_normal = 0 the smooth-tube floor.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.light_phong_simple.wgsl' $}

// Floats per node in s_coefs, i.e. 8 + 2 * K of the uploaded buffer. Fixed at
// upload time, hence a template constant rather than a uniform.
const STRIDE: i32 = {{ stride }};
// Number of harmonics actually present in the buffer. The material's `k_max`
// may be lower (angular truncation), never higher.
const KBUF: i32 = {{ k_buf }};


// Unpack the cross-section frame. The stored quaternion is the rotation whose
// *columns* are (u, v, t): u and v span the cross-section plane, t is the
// skeleton tangent (= cross(u, v), not needed here).
fn frame_uv(q: vec4<f32>) -> mat2x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let u = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w));
    let v = vec3<f32>(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w));
    return mat2x3<f32>(u, v);
}


// Radius and its angular derivative at `theta` for node `i`, returned as
// (r, dr/dtheta). cos(k*theta) and sin(k*theta) are walked upward by angle
// addition, so the whole series costs one cos and one sin whatever K is - and
// because d/dtheta [a cos(k t) + b sin(k t)] = k (b cos(k t) - a sin(k t)),
// the derivative (i.e. the analytic normal) falls out of the same loop free.
fn eval_profile(i: i32, theta: f32, kmax: i32) -> vec2<f32> {
    let base = i * STRIDE;
    var r = load_s_coefs(base + 7);  // a0
    var dr = 0.0;

    let c = cos(theta);
    let s = sin(theta);
    var ck = 1.0;  // cos(0 * theta)
    var sk = 0.0;  // sin(0 * theta)

    for (var k = 1; k <= kmax; k += 1) {
        // Advance to k*theta without any further trig calls
        let t = ck * c - sk * s;
        sk = sk * c + ck * s;
        ck = t;

        let a = load_s_coefs(base + 7 + k);
        let b = load_s_coefs(base + 7 + KBUF + k);
        r += a * ck + b * sk;
        dr += f32(k) * (b * ck - a * sk);
    }
    return vec2<f32>(r, dr);
}


// The skeleton point and the cross-section basis at `theta`, with no profile
// applied. Kept separate from the radius so that the position and the normal
// can be evaluated at different truncations off the same frame.
struct NodeFrame {
    pos: vec3<f32>,
    e_r: vec3<f32>,
    e_t: vec3<f32>,
};

fn node_pos(i: i32) -> vec3<f32> {
    let base = i * STRIDE;
    return vec3<f32>(
        load_s_coefs(base), load_s_coefs(base + 1), load_s_coefs(base + 2),
    );
}

// The skeleton tangent, i.e. the third column of the frame rotation (= u x v).
// Independent of theta, so it is available before the angular decode.
fn node_tangent(i: i32) -> vec3<f32> {
    let base = i * STRIDE;
    let x = load_s_coefs(base + 3);
    let y = load_s_coefs(base + 4);
    let z = load_s_coefs(base + 5);
    let w = load_s_coefs(base + 6);
    return vec3<f32>(
        2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y),
    );
}

fn node_frame(i: i32, theta: f32) -> NodeFrame {
    let base = i * STRIDE;
    let p = vec3<f32>(
        load_s_coefs(base), load_s_coefs(base + 1), load_s_coefs(base + 2),
    );
    let q = vec4<f32>(
        load_s_coefs(base + 3), load_s_coefs(base + 4),
        load_s_coefs(base + 5), load_s_coefs(base + 6),
    );
    let uv = frame_uv(q);
    let c = cos(theta);
    let s = sin(theta);

    var out: NodeFrame;
    out.pos = p;
    out.e_r = c * uv[0] + s * uv[1];   // radial
    out.e_t = -s * uv[0] + c * uv[1];  // tangential, i.e. d(e_r)/d(theta)
    return out;
}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let n_theta = max(u_material.n_theta, 3);
    let kmax = clamp(u_material.k_max, 0, KBUF);
    // The normal can only ever be truncated harder than the surface it
    // describes - harmonics the position does not have are meaningless here
    let kmax_n = clamp(u_material.k_normal, 0, kmax);

    let vid = i32(in.vertex_index);
    let quad = vid / 6;
    let corner = vid % 6;
    let e = quad / n_theta;  // which skeleton edge
    let j = quad % n_theta;  // which angular sector

    // The quad as two triangles, wound counter-clockwise as seen from outside
    // the tube (the same convention as a mesh's faces), so that the builtin
    // `front_facing` in fs_main agrees with the outward normal:
    //   (0,0) (0,1) (1,0)  and  (1,0) (0,1) (1,1)
    // where the first index picks the edge end and the second the angular step.
    var ends = array<i32,6>(0, 0, 1,  1, 0, 1);
    var offs = array<i32,6>(0, 1, 0,  0, 1, 1);

    let edge = load_s_edges(e);
    let at_far = ends[corner] == 1;
    let node = select(edge.x, edge.y, at_far);
    let node_other = select(edge.y, edge.x, at_far);

    // Does this edge run along the stored tangent? The edge list may pair its
    // nodes either way round, and swapping them would otherwise reverse the
    // traversal - hence the winding, hence `front_facing`, hence the lighting,
    // leaving a mixed-orientation edge list rendered half-dark. Stepping theta
    // the other way for a reversed edge covers the same four corners in the
    // opposite order, which restores the winding without moving any vertex.
    let along_t = dot(
        node_pos(edge.y) - node_pos(edge.x), node_tangent(edge.x)
    ) >= 0.0;
    let off = select(1 - offs[corner], offs[corner], along_t);

    let jj = (j + off) % n_theta;
    let theta = 2.0 * PI * f32(jj) / f32(n_theta);

    let f_here = node_frame(node, theta);
    let f_other = node_frame(node_other, theta);

    // The rendered position keeps every harmonic the buffer holds
    let r_here = eval_profile(node, theta, kmax);
    let pos = f_here.pos + r_here.x * f_here.e_r;

    // The normal describes a smoother surface (see the header). Both the
    // angular tangent and both ends of the axial difference are evaluated at
    // `kmax_n`, or the normal would not be perpendicular to any one surface.
    let rn_here = eval_profile(node, theta, kmax_n);
    let rn_other = eval_profile(node_other, theta, kmax_n);

    let dp_dtheta = rn_here.y * f_here.e_r + rn_here.x * f_here.e_t;
    // The axial direction comes from the *other* end of this very edge, at the
    // same theta. That is exact for the quad being drawn and, unlike the
    // stored tangent, it accounts for the radius changing along the tube -
    // boutons and varicosities are exactly that, a local bump in a0 along the
    // axis rather than an angular phenomenon. It is one-sided where the CPU
    // mirror's is centred, so the two quads meeting at a ring get slightly
    // different normals; that faceting is expected.
    //
    // Note it is oriented along +t rather than away from this corner's own
    // end, which is what makes the normal outward *by construction*:
    //   cross(dr e_r + r e_t, L t + dR e_r) . e_r = r L > 0
    // for any dr / dR. Testing `dot(n, e_r) > 0` instead would also work in
    // the smooth case, but it fails per-vertex wherever the profile is noisy
    // or non-star-shaped enough to tilt the normal past 90 degrees off radial,
    // and a single flipped vertex leaves an interpolated dark seam across its
    // triangles - one of the ways "the normals look wrong" shows up.
    let p_here = f_here.pos + rn_here.x * f_here.e_r;
    let p_other = f_other.pos + rn_other.x * f_other.e_r;
    let axial_xy = select(p_other - p_here, p_here - p_other, at_far);
    let axial = select(-axial_xy, axial_xy, along_t);

    var n = cross(dp_dtheta, axial);
    let n_len = length(n);
    // Degenerate quad (zero-length edge, zero radius): fall back to radial
    n = select(f_here.e_r, n / max(n_len, 1e-20), n_len > 1e-12);

    let world_pos = u_wobject.world_transform * vec4<f32>(pos, 1.0);
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.normal = vec3<f32>(normalize((u_wobject.world_transform * vec4<f32>(n, 0.0)).xyz));

    $$ if color_mode == 'vertex'
        varyings.color = vec4<f32>(load_s_colors(node));
    $$ endif

    // Picking: the edge and the angular sector are enough to identify the quad
    varyings.edge_index = u32(e);
    varyings.sector = u32(j);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    $$ if color_mode == 'vertex'
        // Per-node colors are already in physical space
        var diffuse_color = varyings.color;
    $$ else
        var diffuse_color = u_material.color;
        diffuse_color = vec4<f32>(srgb2physical(diffuse_color.rgb), diffuse_color.a);
    $$ endif

    diffuse_color.a = diffuse_color.a * u_material.opacity;

    do_alpha_test(diffuse_color.a);

    // Direction from the surface towards the camera
    let view = select(
        normalize(u_stdinfo.cam_transform_inv[3].xyz - varyings.world_pos),
        (u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz,
        is_orthographic(),
    );

    // The normal is analytic and points outward; `is_front` comes from the
    // winding, so lighting_phong()'s reorientation gives the interior of the
    // tube (visible through an open end) two-sided shading rather than a
    // black hole - and an inverted normal shows up immediately.
    let normal = normalize(varyings.normal);
    let physical_color = lighting_phong(is_front, normal, view, diffuse_color.rgb);

    let out_color = vec4<f32>(physical_color, diffuse_color.a);

    var out: FragmentOutput;
    out.color = out_color;

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(varyings.edge_index, 26) +
        pick_pack(varyings.sector, 18)
    );
    $$ endif
    return out;
}

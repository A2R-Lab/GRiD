"""W3 Component D (data-prep half) — foam spherized-URDF parsing + the FLANGE anchor
mapping + self-collision ranges for `grid_collision`.

These are the pure-Python helpers that turn a foam-spherized URDF into a sphere batch
descriptor compatible with `_multitarget.build_target_batch` (W1b/W2a). They are
deliberately DECOUPLED from the codegen/cli/registry wiring (the `grid_collision`
namespace emitter + `--collision` flag co-land later with the W1b.3/W2a.3 registration),
so the ⚠FLANGE mapping — the #1 silent-wrong-frame hazard — can be unit-tested standalone
against real GRiD robots WITHOUT the (heavy, optional) foam toolchain.

KEY DIFFERENCE from HJCD's `foam_spheres.py`: GRiD binds each sphere to ITS OWN GRiD frame
(the `s_Xworld` joint slot), resolved through GRiD's URDFParser — NOT foam's actuated-joint
count and NOT HJCD's URDF-ordinal table. A sphere is orientation-free, so a sphere on a
WELDED (fixed-joint) frame reduces to its nearest MOVABLE ancestor joint with a constant
PRE-COMPOSED offset (`offset_parent = (T_parent<-welded @ [offset_local,1])[:3]`) — no
fixed-anchor `s_Xworld` slot required. `T_parent<-welded` is GRiD's OWN post-`remove_fixed_
joints` transform (`Fixed_Joint.get_transformation_matrix_hom()`), keeping GRiD's baked FK
the single source of truth.
"""
import xml.etree.ElementTree as ET

import numpy as np


def _c_float_literal(v):
    """Format a float as a valid C++ `float` literal. `"{:.9g}".format(1000.0)` yields "1000"
    (no decimal point), so a bare "f" suffix parses as a user-defined literal and the compile
    fails -- ensure a '.'/'e' is present before appending the suffix (integer-valued radii are
    legal spherizer inputs)."""
    s = "{:.9g}".format(float(v))
    if not any(c in s for c in ".eEnN"):   # integer-valued (also guards inf/nan spelled out)
        s += ".0"
    return s + "f"


# --------------------------------------------------------------------------- foam parse
def parse_spherized_urdf(path):
    """foam output: per link, one `<collision><geometry><sphere radius/></geometry>
    <origin xyz/></collision>` per sphere (center in the LINK frame; rpy irrelevant for a
    sphere). Returns `{link_name: [(x, y, z, radius), ...]}` in URDF document order."""
    root = ET.parse(path).getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        spheres = []
        for col in link.findall("collision"):
            sph = col.find("geometry/sphere")
            if sph is None:
                continue
            r = float(sph.get("radius"))
            origin = col.find("origin")
            xyz = (origin.get("xyz") if origin is not None else "0 0 0").split()
            x, y, z = (float(v) for v in xyz)
            spheres.append((x, y, z, r))
        if spheres:
            out[name] = spheres
    return out


def urdf_joint_tree(path):
    """Parse the URDF's native joint tree (retained in foam's spherized output). Returns
    `{child_link_name: joint_name}` — the bridge from a foam sphere's LINK to the GRiD
    Fixed_Joint (matched by joint name) when that link is welded and has no movable joint."""
    root = ET.parse(path).getroot()
    child_to_joint = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_to_joint[child.get("link")] = joint.get("name")
    return child_to_joint


# --------------------------------------------------------------------------- FLANGE mapping
def sphere_anchor_frames(robot, link_names, urdf_path):
    """⚠FLANGE crux. Map each foam link name -> `(anchor_jid, T_compose)` where `anchor_jid`
    is the GRiD movable-joint slot in `s_Xworld` and `T_compose` (4x4) carries the sphere's
    LOCAL offset into that anchor's frame (identity for a movable link; the fixed transform
    for a welded frame). A sphere at local `r` extracts at world `X_world[anchor] @ T_compose
    @ [r, 1]`.

    Resolution per link:
      * MOVABLE link  -> the joint whose child IS this link (`get_joints_by_child_name`);
        `T_compose = I`.
      * WELDED frame  -> the URDF joint feeding this link (by child-link name) matches a GRiD
        `Fixed_Joint` by NAME; anchor = that fixed joint's re-parented movable joint
        (`get_parent()` -> movable joint id), `T_compose = Fixed_Joint.get_transformation_
        matrix_hom()` (GRiD's post-removal parent-movable -> welded transform).
      * ROOT/WORLD frame (fed by NO joint at all, or a fixed joint welded to world parent -1)
        -> `(-1, I)` SKIP sentinel; caller drops these (pedestal-bolted) spheres.

    A mismatch here silently checks collision on the WRONG frame — validate against the
    UR10e/panda/iiwa base-0-monotone-down-chain assertion (see test_collision_flange_mapping)."""
    child_to_joint = urdf_joint_tree(urdf_path)
    frames = {}
    for link_name in link_names:
        movable = robot.get_joints_by_child_name(link_name)
        if movable:
            frames[link_name] = (int(movable[0].get_id()), np.eye(4))
            continue
        # No movable joint has this child. Bridge foam link -> URDF joint name -> GRiD Fixed_Joint.
        jname = child_to_joint.get(link_name)
        if jname is None:
            # No joint feeds this link -> it is the robot root/base frame. Skip (pedestal).
            frames[link_name] = (-1, np.eye(4))
            continue
        fj = robot.get_fixed_joint_by_name(jname)
        if fj is not None:
            parent_joint = robot.get_joint_by_name(fj.get_parent())
            if parent_joint is None:
                # fixed joint welded straight to the world/base (parent -1) -> skip sentinel
                frames[link_name] = (-1, np.eye(4))
                continue
            T = np.asarray(fj.get_transformation_matrix_hom(), dtype=np.float64).reshape(4, 4)
            frames[link_name] = (int(parent_joint.get_id()), T)
            continue
        raise ValueError(
            "FLANGE: cannot map foam link '%s' to a GRiD frame (no movable joint with this "
            "child, and no fixed joint named '%s'). A silent wrong-frame anchor would result."
            % (link_name, jname))
    return frames


def _parent_jid_map(robot):
    """`{jid: parent_jid}` over movable joints (parent movable joint = the joint whose child
    link is this joint's parent link; -1 at the root). Used for self-collision adjacency."""
    child_link_to_jid = {j.get_child(): j.get_id() for j in robot.get_joints_ordered_by_id()}
    parent = {}
    for j in robot.get_joints_ordered_by_id():
        parent[j.get_id()] = child_link_to_jid.get(j.get_parent(), -1)
    return parent


def _anchors_adjacent(parent_jid, a, b):
    """Two sphere anchors are 'adjacent' (skip the pair: they always touch) when they share a
    frame or are directly parent/child in the movable-joint tree."""
    return a == b or parent_jid.get(a, -1) == b or parent_jid.get(b, -1) == a


def build_self_cc_ranges(robot, anchor):
    """Self-collision pairs as `{sphere_i, start_j, end_j}` rows: sphere i is checked against
    spheres [start_j..end_j] (j > i). Adjacent-frame pairs (same/parent/child link) are
    skipped (always in contact). Same-anchor spheres are contiguous in `anchor` (built link by
    link), so the non-adjacent partners of i compress into maximal contiguous [j0,j1] runs."""
    parent_jid = _parent_jid_map(robot)
    n = len(anchor)
    ranges = []
    for i in range(n):
        j = i + 1
        while j < n:
            if _anchors_adjacent(parent_jid, anchor[i], anchor[j]):
                j += 1
                continue
            j0 = j
            while j < n and not _anchors_adjacent(parent_jid, anchor[i], anchor[j]):
                j += 1
            ranges.append((i, j0, j - 1))
    return ranges


def collision_spec_from_urdf(robot, urdf_path, resolution, mesh_resolution=None, out_path=None):
    """One-call URDF -> single-tier collision_spec (for gen_all_code). Spherizes `urdf_path`
    (custom trimesh/analytic spherizer, foam-compatible output) then binds the spheres to GRiD
    frames via build_sphere_tiers. Returns {anchor, offset, radius, self_cc_ranges} -- the dict
    gen_all_code's collision_spec kwarg expects. `resolution` = sphere spacing (m)."""
    from ._spherize import spherize_urdf
    sph_path = spherize_urdf(urdf_path, resolution, out_path=out_path, mesh_resolution=mesh_resolution)
    tier = build_sphere_tiers(robot, {"all": sph_path})["all"]
    return {"anchor": tier["anchor"], "offset": tier["offset"],
            "radius": tier["radius"], "self_cc_ranges": tier["self_cc_ranges"]}


def multi_tier_collision_spec_from_urdf(robot, urdf_path, resolutions, mesh_resolution=None):
    """One-call URDF -> MULTI-tier collision_spec (`{"tiers": [...]}` for gen_all_code). Spherizes
    `urdf_path` once per resolution and binds each to GRiD frames. `resolutions` = iterable of
    sphere spacings (m); sorted DESCENDING so the returned tiers run COARSEST->FINEST (config_free
    uses coarsest for the broad reject + finest for the confirm). Two tiers are named broad/fine;
    more are named tier0..tierK (tier0 = coarsest). A single resolution returns the flat single-tier
    spec (byte-identical to collision_spec_from_urdf)."""
    res = sorted({float(r) for r in resolutions}, reverse=True)  # coarsest (largest spacing) first
    if len(res) == 1:
        return collision_spec_from_urdf(robot, urdf_path, res[0], mesh_resolution=mesh_resolution)
    names = ["broad", "fine"] if len(res) == 2 else ["tier%d" % i for i in range(len(res))]
    from ._spherize import spherize_urdf
    outputs = {names[i]: spherize_urdf(urdf_path, res[i], mesh_resolution=mesh_resolution)
               for i in range(len(res))}
    built = build_sphere_tiers(robot, outputs)
    tiers = []
    for i in range(len(res)):
        b = built[names[i]]
        tiers.append({"name": names[i], "anchor": b["anchor"], "offset": b["offset"],
                      "radius": b["radius"], "self_cc_ranges": b["self_cc_ranges"]})
    return {"tiers": tiers}


def build_sphere_tiers(robot, foam_outputs):
    """`foam_outputs = {tier_name: spherized_urdf_path}` (e.g. broad + fine, two foam runs).
    Returns `{tier: {"n", "anchor"[N], "offset"[3N], "radius"[N], "self_cc_ranges"[R][3]}}`.
    Offsets are PRE-COMPOSED into their anchor frame (welded frames folded onto the movable
    parent). Index-0 (base/pedestal) spheres are skipped, matching HJCD."""
    tiers = {}
    for tier_name, path in foam_outputs.items():
        parsed = parse_spherized_urdf(path)
        frames = sphere_anchor_frames(robot, list(parsed.keys()), path)
        anchor, offset, radius = [], [], []
        for link_name, spheres in parsed.items():
            anchor_jid, T = frames[link_name]
            if anchor_jid < 0:
                continue  # skip root/world (pedestal-bolted) spheres
            for (x, y, z, r) in spheres:
                p = T @ np.array([x, y, z, 1.0])
                anchor.append(int(anchor_jid))
                offset.extend([float(p[0]), float(p[1]), float(p[2])])
                radius.append(float(r))
        tiers[tier_name] = {
            "n": len(anchor), "anchor": anchor, "offset": offset, "radius": radius,
            "self_cc_ranges": build_self_cc_ranges(robot, anchor),
        }
    return tiers


# --------------------------------------------------------------------------- tier normalization
def normalize_collision_tiers(collision_spec):
    """Normalize gen_all_code's `collision_spec` kwarg into an ORDERED (coarsest->finest) list of
    tier dicts `{"name", "suffix", "anchor", "offset", "radius", "self_cc_ranges", "n"}`.

    Accepts either:
      * a FLAT single-tier dict `{"anchor","offset","radius","self_cc_ranges"}` (the pre-tier
        shape; -> one finest tier, suffix ""), or
      * a multi-tier dict `{"tiers": [ {"name","anchor","offset","radius","self_cc_ranges"}, ... ]}`
        listed COARSEST FIRST.

    Suffix assignment: the FINEST (last) tier is the public batch -> suffix "" (so the
    differentiable API + single-tier config_free keep stable names); every coarser tier is
    suffixed "_<name>". A single tier is therefore byte-identical to the pre-tier emission."""
    if "tiers" in collision_spec:
        raw = list(collision_spec["tiers"])
        assert len(raw) >= 1, "collision_spec['tiers'] must be non-empty"
    else:
        raw = [{"name": "", **collision_spec}]
    out = []
    last = len(raw) - 1
    for i, t in enumerate(raw):
        name = t.get("name", "") if i != last else t.get("name", "")
        suffix = "" if i == last else "_" + t["name"]
        out.append({
            "name": name, "suffix": suffix,
            "anchor": list(t["anchor"]), "offset": list(t["offset"]),
            "radius": list(t["radius"]), "self_cc_ranges": list(t["self_cc_ranges"]),
            "n": len(t["anchor"]),
        })
    return out


# --------------------------------------------------------------------------- namespace emitter
def gen_collision_namespace(self, tiers):
    """Emit the sibling `namespace grid_collision { ... }` block (model = gen_grid_plant).

    Called AFTER the `grid` namespace closes, and ONLY when a collision batch is configured.
    The sphere set(s) ARE the multi_target batch(es), so this requires gen_multi_target_position
    to have been emitted per tier (NUM_COLLISION_SPHERES<CAP> == grid::NUM_MULTI_TARGETS<CAP>).
    Reopens the grid_collision namespace already opened by the static geometry header
    (collision/grid_collision_geometry.cuh, W3 Component E) and adds the per-robot BAKED data
    (fp32 radii + self_cc_ranges) plus `config_free` composed over the W1b batched extractor
    and the header's SDF checks.

    `tiers` = ORDERED list (COARSEST -> FINEST) of sphere-density tiers, each a dict:
        {"name": str, "suffix": str, "n": int, "radius": [float]*n,
         "self_cc_ranges": [(sphere_i, start_j, end_j)]}
    The FINEST (last) tier is the PUBLIC one: its suffix is "" so the differentiable
    collision API (collision_distance/cost/..., NUM_COLLISION_SPHERES) and the single-tier
    config_free keep stable, tier-count-invariant names. Coarser tiers are suffixed by name
    (e.g. "_broad") and used ONLY as the broad-phase reject inside the multi-tier config_free.
    A single tier (len==1) is the finest -> suffix "" -> byte-identical to the pre-tier emission.
    """
    assert len(tiers) >= 1, "collision: at least one sphere tier required"
    nv = self.robot.get_num_vel()
    fine = tiers[-1]  # finest tier = the public / differentiable batch (suffix "")
    for t in tiers:
        assert len(t["radius"]) == t["n"], "collision: radius count (%d) != sphere count (%d) [tier %s]" % (
            len(t["radius"]), t["n"], t["name"])
    assert fine["suffix"] == "", "collision: finest tier must be unsuffixed (the public batch)"

    # The geometry header must precede the reopened namespace (it defines the SDFs + opens
    # grid_collision). Emitted at file scope after the grid namespace closes.
    self.gen_add_code_line("")
    self.gen_add_code_line('#include "grid_collision_geometry.cuh"  // W3 Component E: SDF primitives (grid_collision::)')
    self.gen_add_func_doc("Collision namespace: baked sphere data + config_free composed over "
                          "grid::multi_target_position + the static SDF geometry header")
    self.gen_add_code_line("namespace " + self.file_namespace + "_collision {", True)
    # Bring the tier enum into scope so GRID_DEFAULT_RESOURCE_TIER (a macro expanding to a
    # bare TIER_* name defined in namespace grid) resolves inside this sibling namespace,
    # incl. a -DGRID_DEFAULT_RESOURCE_TIER=TIER_LITE/MINIMAL override.
    self.gen_add_code_line("using " + self.file_namespace + "::TIER_SHARED; using " +
                           self.file_namespace + "::TIER_LITE; using " + self.file_namespace + "::TIER_MINIMAL;")
    # The broad->fine link_CC narrowing (Inc4a) keys a uint64 hit-mask by anchor (frame/joint) id,
    # so every collision-sphere anchor id must fit in 64 bits. A future >64-frame robot needs the
    # documented bool[NUM_JOINTS] fallback in grid_cc_config_free instead. Only the multi-tier driver
    # uses the mask, so this (and the per-tier sphere->link tables) are emitted only for 2+ tiers.
    if len(tiers) > 1:
        self.gen_add_code_line("static_assert(" + self.file_namespace + "::NUM_JOINTS <= 64, "
                               "\"link_CC broad-phase mask is a uint64 keyed by frame id; \"")
        self.gen_add_code_line("              \"robots with >64 frames need the bool[NUM_JOINTS] fallback\");")

    # --- baked per-robot data, PER TIER (finest is unsuffixed = the public batch) ---
    for t in tiers:
        sfx = t["suffix"]            # "" for finest, e.g. "_broad" for a coarse tier
        cap = sfx.upper()
        n = t["n"]
        rr = t["self_cc_ranges"]
        r = len(rr)
        flat_ranges = ", ".join(str(v) for row in rr for v in row) if r else "0"
        if len(tiers) > 1:
            self.gen_add_code_line("// collision tier '" + t["name"] + "' (" + str(n) + " spheres" +
                                   (", FINEST/public" if sfx == "" else ", broad-phase") + ")")
        self.gen_add_code_lines([
            "constexpr int NUM_COLLISION_SPHERES" + cap + " = " + str(n) + ";",
            "constexpr int NUM_COLLISION_SELF_CC_RANGES" + cap + " = " + str(r) + ";",
            "static_assert(NUM_COLLISION_SPHERES" + cap + " == grid::NUM_MULTI_TARGETS" + cap + ", "
            "\"collision sphere batch must be the multi_target batch\");",
            # fp32 radii (default collision precision, USER-CONFIRMED); ranges as {i, start_j, end_j} rows.
            "__device__ const float g_collision_sphere_r" + sfx + "[" + str(max(n, 1)) + "] = {" +
            ", ".join(_c_float_literal(rad) for rad in (t["radius"] or [0.0])) + "};",
            "__device__ const int g_collision_self_cc_ranges" + sfx + "[" + str(max(3 * r, 1)) + "] = {" + flat_ranges + "};",
        ])
        if len(tiers) > 1:
            # sphere -> anchor (GRiD frame/joint) id: the bit index into the broad-phase link_CC
            # hit-mask (W3 Inc4a). The mask lets the fine pass skip spheres whose link the broad pass
            # didn't flag. Multi-tier only (the single-tier config_free doesn't narrow); the
            # NUM_JOINTS<=64 static_assert above (also multi-tier-gated) keeps every id in a uint64.
            self.gen_add_code_line(
                "__device__ const int g_collision_sphere_link" + sfx + "[" + str(max(n, 1)) + "] = {" +
                ", ".join(str(a) for a in (t["anchor"] or [0])) + "};")
        # --- fill a caller T-scratch with this tier's baked fp32 radii (cast to T) ---
        self.gen_add_func_doc("Fill s_r[NUM_COLLISION_SPHERES" + cap + "] with the baked fp32 radii cast to T",
                              [], ["s_r is caller shared memory of size NUM_COLLISION_SPHERES" + cap], None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__ __forceinline__")
        self.gen_add_code_line("void load_collision_radii" + sfx + "(T *s_r) {", True)
        self.gen_add_parallel_loop("i", "NUM_COLLISION_SPHERES" + cap)
        self.gen_add_code_line("s_r[i] = static_cast<T>(g_collision_sphere_r" + sfx + "[i]);")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()

    # --- config_free entry point ---
    if len(tiers) == 1:
        # Single tier: inline self + env check on the (finest, unsuffixed) batch. Byte-identical
        # to the pre-tier emission.
        func_params = [
            "s_q is the vector of joint positions",
            "d_robotModel is the initialized model-specific helpers on the GPU",
            "env is the runtime obstacle set (grid_collision::Environment<T>)",
            "s_sphere_pos is caller scratch of size 3*NUM_COLLISION_SPHERES (smem for small N, global for many)",
            "s_sphere_r is caller scratch of size NUM_COLLISION_SPHERES (filled here from the baked radii)",
            "d_workspace is the multi_target FK scratch at TIER_LITE+ (nullptr at TIER_SHARED)"]
        func_notes = [
            "Returns true iff the current configuration q is COLLISION-FREE (self + environment).",
            "Sphere world positions via the W1b batched extractor; SDF self/env checks via the static header.",
            "Every thread computes the same verdict; the self/env range loops are serial (parallelize = W3 perf TODO)."]
        self.gen_add_func_doc("Collision-free test for configuration q (self + environment)", func_notes, func_params, None)
        self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("bool config_free(const T *s_q, const grid::robotModel<T> *d_robotModel, "
                               "const Environment<T> &env, T *s_sphere_pos, T *s_sphere_r, T *d_workspace = nullptr) {", True)
        self.gen_add_code_line("grid::multi_target_position_device<T, RESOURCE_TIER>(s_sphere_pos, s_q, d_robotModel, d_workspace);")
        self.gen_add_code_line("load_collision_radii<T>(s_sphere_r);")
        self.gen_add_sync()
        self.gen_add_code_line("if (grid_cc_self_collision<T>(s_sphere_pos, s_sphere_r, g_collision_self_cc_ranges, NUM_COLLISION_SELF_CC_RANGES)) return false;")
        self.gen_add_code_line("for (int i = 0; i < NUM_COLLISION_SPHERES; ++i) {", True)
        self.gen_add_code_line("if (grid_cc_sphere_in_environment<T>(env, s_sphere_pos[3*i], s_sphere_pos[3*i+1], s_sphere_pos[3*i+2], s_sphere_r[i])) return false;")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("return true;")
        self.gen_add_end_function()
    else:
        # Multi-tier: broad-phase reject (COARSEST tier) -> fine confirm (FINEST tier) via the
        # header's grid_cc_config_free driver. The covering-sphere property (coarser => larger
        # spheres enclosing the fine geometry) makes "broad clear => definitely free" exact, so
        # the verdict is IDENTICAL to a fine-only check but skips the fine pass on clear configs.
        # Middle tiers (if any) are emitted + callable but not used by config_free (the driver is
        # coarsest+finest; a k-level cascade is a labeled header extension).
        broad = tiers[0]
        bsfx, bcap = broad["suffix"], broad["suffix"].upper()
        func_params = [
            "s_q is the vector of joint positions",
            "d_robotModel is the initialized model-specific helpers on the GPU",
            "env is the runtime obstacle set (grid_collision::Environment<T>)",
            "s_broad_pos is caller scratch of size 3*NUM_COLLISION_SPHERES" + bcap + " (broad-phase sphere positions)",
            "s_broad_r is caller scratch of size NUM_COLLISION_SPHERES" + bcap + " (filled here from broad baked radii)",
            "s_fine_pos is caller scratch of size 3*NUM_COLLISION_SPHERES (fine sphere positions)",
            "s_fine_r is caller scratch of size NUM_COLLISION_SPHERES (filled here from fine baked radii)",
            "d_workspace is the multi_target FK scratch at TIER_LITE+ (nullptr at TIER_SHARED)"]
        func_notes = [
            "Returns true iff the current configuration q is COLLISION-FREE (self + environment).",
            "Broad tier '" + broad["name"] + "' rejects clear configs; only possible collisions run the fine tier '" +
            fine["name"] + "'. Verdict == fine-only (covering spheres make the broad reject conservative).",
            "Every thread computes the same verdict; the self/env range loops are serial (parallelize = W3 perf TODO)."]
        self.gen_add_func_doc("Collision-free test for configuration q (broad->fine, self + environment)", func_notes, func_params, None)
        self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("bool config_free(const T *s_q, const grid::robotModel<T> *d_robotModel, "
                               "const Environment<T> &env, T *s_broad_pos, T *s_broad_r, "
                               "T *s_fine_pos, T *s_fine_r, T *d_workspace = nullptr, "
                               "int *dbg_fine_rechecked = nullptr) {", True)
        self.gen_add_code_line("grid::multi_target_position" + bsfx + "_device<T, RESOURCE_TIER>(s_broad_pos, s_q, d_robotModel, d_workspace);")
        self.gen_add_code_line("load_collision_radii" + bsfx + "<T>(s_broad_r);")
        self.gen_add_sync()
        self.gen_add_code_line("grid::multi_target_position_device<T, RESOURCE_TIER>(s_fine_pos, s_q, d_robotModel, d_workspace);")
        self.gen_add_code_line("load_collision_radii<T>(s_fine_r);")
        self.gen_add_sync()
        self.gen_add_code_line("return grid_cc_config_free<T>(env,")
        self.gen_add_code_line("    s_broad_pos, s_broad_r, g_collision_self_cc_ranges" + bsfx + ", NUM_COLLISION_SELF_CC_RANGES" + bcap + ", NUM_COLLISION_SPHERES" + bcap + ", g_collision_sphere_link" + bsfx + ",")
        self.gen_add_code_line("    s_fine_pos, s_fine_r, g_collision_self_cc_ranges, NUM_COLLISION_SELF_CC_RANGES, NUM_COLLISION_SPHERES, g_collision_sphere_link, dbg_fine_rechecked);")
        self.gen_add_end_function()

    # ---- differentiable collision PRIMITIVES + cost (value / gradient / Gauss-Newton hessian) ----
    # The raw building blocks are exposed SEPARATELY from the cost so consumers can assemble any
    # collision objective (hinge, log-barrier, hard constraint) on the underlying derivatives:
    #   collision_distance          -> per-sphere signed clearance d_i(q) (min over env) + surface normal
    #   collision_distance_gradient -> per-sphere clearance Jacobian  d(d_i)/dq[vi] = n_i^T (dp_i/dq_vi)
    # d(d_i)/dq composes the SDF surface normal n_i (grid_cc_nearest_obstacle) with the W2a batched
    # position gradient (grid::multi_target_position_gradient_device, layout s_pos_grad[3*(NV*i+vi)+row]).
    # The cost fns below are thin reductions over these primitives (a hinge on a safety margin):
    #   viol_i = max(0, margin - d_i);  cost = 1/2 weight sum_i viol_i^2;  d(viol_i)/dq = -d(d_i)/dq.
    # Environment-only (self-collision stays the boolean config_free feasibility test). The hard argmin
    # over obstacles is non-smooth where the nearest obstacle switches; a consumer wanting a smooth MPC
    # Hessian can freeze the per-sphere active obstacle across a step (the normal pre-pass is the seam).
    _cc_state_params = [
        "s_q is the vector of joint positions",
        "d_robotModel is the initialized model-specific helpers on the GPU",
        "env is the runtime obstacle set (grid_collision::Environment<T>)",
        "s_sphere_pos is caller scratch of size 3*NUM_COLLISION_SPHERES (sphere world positions)",
        "s_sphere_r is caller scratch of size NUM_COLLISION_SPHERES (filled here from the baked radii)",
        "d_workspace is the multi_target FK scratch at TIER_LITE+ (nullptr at TIER_SHARED)"]

    # PRIMITIVE: per-sphere signed clearance + normal
    self.gen_add_func_doc("collision_distance: per-sphere nearest signed clearance d_i(q) + surface normal (env only)",
                          ["d_i = min over environment obstacles of the signed distance (>0 clear, <0 penetrating).",
                           "s_dist[i] = +1e30 sentinel when the environment is empty. Raw building block for any "
                           "collision objective; the cost fns below reduce over it."],
                          ["s_dist is the per-sphere clearance output (size NUM_COLLISION_SPHERES)",
                           "s_normal is the per-sphere nearest-obstacle unit normal (size 3*NUM_COLLISION_SPHERES)"] + _cc_state_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_distance(T *s_dist, T *s_normal, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T *s_sphere_pos, T *s_sphere_r, T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("grid::multi_target_position_device<T, RESOURCE_TIER>(s_sphere_pos, s_q, d_robotModel, d_workspace);")
    self.gen_add_code_line("load_collision_radii<T>(s_sphere_r);")
    self.gen_add_sync()
    self.gen_add_parallel_loop("i", "NUM_COLLISION_SPHERES")
    self.gen_add_code_line("T nx, ny, nz;")
    self.gen_add_code_line("s_dist[i] = grid_cc_nearest_obstacle<T>(env, s_sphere_pos[3*i], s_sphere_pos[3*i+1], s_sphere_pos[3*i+2], s_sphere_r[i], &nx, &ny, &nz);")
    self.gen_add_code_line("s_normal[3*i+0] = nx; s_normal[3*i+1] = ny; s_normal[3*i+2] = nz;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    # PRIMITIVE: per-sphere clearance Jacobian d(d_i)/dq
    self.gen_add_func_doc("collision_distance_gradient: per-sphere clearance Jacobian s_ddist[i*NV+vi] = d(d_i)/dq_vi = n_i^T dp_i/dq_vi",
                          ["Also returns s_dist (the clearances) so a consumer has value + Jacobian in one call.",
                           "n_i^T (dp_i/dq) composes the SDF normal with grid::multi_target_position_gradient_device.",
                           "s_ddist layout is per-sphere-major: sphere i's NV-gradient is s_ddist[i*NV .. i*NV+NV-1]."],
                          ["s_dist is the per-sphere clearance output (size NUM_COLLISION_SPHERES)",
                           "s_ddist is the per-sphere clearance Jacobian output (size NUM_COLLISION_SPHERES*NUM_VEL, sphere-major)"] +
                          _cc_state_params +
                          ["s_normal is caller scratch of size 3*NUM_COLLISION_SPHERES (nearest-obstacle normals)",
                           "s_pos_grad is caller scratch of size 3*NUM_VEL*NUM_COLLISION_SPHERES (batched dp/dq)"], None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_distance_gradient(T *s_dist, T *s_ddist, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T *s_sphere_pos, T *s_sphere_r, T *s_normal, T *s_pos_grad, "
                           "T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("collision_distance<T, RESOURCE_TIER>(s_dist, s_normal, s_q, d_robotModel, env, s_sphere_pos, s_sphere_r, d_workspace);")
    self.gen_add_code_line("grid::multi_target_position_gradient_device<T, RESOURCE_TIER>(s_pos_grad, s_q, d_robotModel, d_workspace);")
    self.gen_add_sync()
    self.gen_add_parallel_loop("ind", "NUM_COLLISION_SPHERES * " + str(nv))
    self.gen_add_code_line("int vi = ind % " + str(nv) + "; int i = ind / " + str(nv) + ";")
    self.gen_add_code_line("int jb = 3 * (" + str(nv) + " * i + vi);")
    self.gen_add_code_line("s_ddist[i*" + str(nv) + " + vi] = s_normal[3*i+0]*s_pos_grad[jb+0] + s_normal[3*i+1]*s_pos_grad[jb+1] + s_normal[3*i+2]*s_pos_grad[jb+2];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    # PRIMITIVES: UN-REDUCED per-(sphere, obstacle) rows.
    # collision_distance above min-reduces over the environment, and that argmin is exactly where the
    # clearance stops being differentiable: as a sphere slides past two obstacles the winning obstacle
    # switches and d(d_i)/dq jumps. A solver that wants one smooth CONSTRAINT ROW PER PAIR (GATO's
    # obstacle rows) needs the reduction dropped, so these emit the full NUM_COLLISION_SPHERES x n_obs
    # block. Each row IS smooth in q (a single fixed primitive), so the non-smoothness moves out of the
    # dynamics and into the solver's own active-set/max, where it belongs.
    # n_obs = grid_cc_num_obstacles(env) is a RUNTIME count (the obstacle set is not baked), so the
    # output sizes are runtime too -- see the caller-contract notes on each param.
    _cc_pair_note = ("Obstacle o indexes the FLATTENED env: spheres | capsules | cuboids | planes, "
                     "o in [0, n_obs) with n_obs = grid_cc_num_obstacles(env). Pair index is "
                     "pair = i*n_obs + o (sphere-major).")

    self.gen_add_func_doc("collision_distance_pairs: UN-REDUCED signed clearance d_io(q) + normal, for every (sphere, obstacle) pair",
                          ["Same SDFs as collision_distance but WITHOUT the min-over-obstacles reduction, which is "
                           "non-smooth precisely where the nearest obstacle switches. Each pair row is smooth in q.",
                           _cc_pair_note,
                           "n_obs == 0 (empty environment) is well-defined: the loop bound is 0 and nothing is written."],
                          ["s_dist is the per-PAIR clearance output (size NUM_COLLISION_SPHERES*n_obs, RUNTIME-sized)",
                           "s_normal is the per-PAIR unit surface normal (size 3*NUM_COLLISION_SPHERES*n_obs, RUNTIME-sized)"] +
                          _cc_state_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_distance_pairs(T *s_dist, T *s_normal, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T *s_sphere_pos, T *s_sphere_r, T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("grid::multi_target_position_device<T, RESOURCE_TIER>(s_sphere_pos, s_q, d_robotModel, d_workspace);")
    self.gen_add_code_line("load_collision_radii<T>(s_sphere_r);")
    self.gen_add_sync()
    self.gen_add_code_line("const int n_obs = grid_cc_num_obstacles<T>(env);")
    self.gen_add_parallel_loop("ind", "NUM_COLLISION_SPHERES * n_obs")
    self.gen_add_code_line("int o = ind % n_obs; int i = ind / n_obs;")
    self.gen_add_code_line("T nx, ny, nz;")
    self.gen_add_code_line("s_dist[ind] = grid_cc_obstacle_signed<T>(env, o, s_sphere_pos[3*i], s_sphere_pos[3*i+1], s_sphere_pos[3*i+2], s_sphere_r[i], &nx, &ny, &nz);")
    self.gen_add_code_line("s_normal[3*ind+0] = nx; s_normal[3*ind+1] = ny; s_normal[3*ind+2] = nz;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    self.gen_add_func_doc("collision_distance_pairs_gradient: per-PAIR clearance Jacobian s_ddist[pair*NV + vi] = d(d_io)/dq_vi = n_io^T dp_i/dq_vi",
                          ["The un-reduced twin of collision_distance_gradient: one NV-row per (sphere, obstacle) pair, "
                           "each smooth in q. Also returns s_dist so a consumer has value + Jacobian in one call.",
                           _cc_pair_note,
                           "s_ddist layout is pair-major: pair (i,o)'s NV-gradient is s_ddist[pair*NV .. pair*NV+NV-1]."],
                          ["s_dist is the per-PAIR clearance output (size NUM_COLLISION_SPHERES*n_obs, RUNTIME-sized)",
                           "s_ddist is the per-PAIR clearance Jacobian output (size NUM_COLLISION_SPHERES*n_obs*NUM_VEL, pair-major, RUNTIME-sized)"] +
                          _cc_state_params +
                          ["s_normal is caller scratch of size 3*NUM_COLLISION_SPHERES*n_obs (per-pair normals, RUNTIME-sized)",
                           "s_pos_grad is caller scratch of size 3*NUM_VEL*NUM_COLLISION_SPHERES (batched dp/dq)"], None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_distance_pairs_gradient(T *s_dist, T *s_ddist, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T *s_sphere_pos, T *s_sphere_r, T *s_normal, T *s_pos_grad, "
                           "T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("collision_distance_pairs<T, RESOURCE_TIER>(s_dist, s_normal, s_q, d_robotModel, env, s_sphere_pos, s_sphere_r, d_workspace);")
    self.gen_add_code_line("grid::multi_target_position_gradient_device<T, RESOURCE_TIER>(s_pos_grad, s_q, d_robotModel, d_workspace);")
    self.gen_add_sync()
    self.gen_add_code_line("const int n_obs = grid_cc_num_obstacles<T>(env);")
    self.gen_add_parallel_loop("ind", "NUM_COLLISION_SPHERES * n_obs * " + str(nv))
    self.gen_add_code_line("int vi = ind % " + str(nv) + "; int pair = ind / " + str(nv) + "; int i = pair / n_obs;")
    self.gen_add_code_line("int jb = 3 * (" + str(nv) + " * i + vi);")
    self.gen_add_code_line("s_ddist[ind] = s_normal[3*pair+0]*s_pos_grad[jb+0] + s_normal[3*pair+1]*s_pos_grad[jb+1] + s_normal[3*pair+2]*s_pos_grad[jb+2];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    _cc_cost_scalar_params = [
        "margin is the safety distance (cost is a hinge on clearance < margin)",
        "weight is the scalar quadratic penalty weight"]

    # COST value
    self.gen_add_func_doc("collision_cost: value = 1/2 * weight * sum_i max(0, margin - d_i)^2 (environment hinge)",
                          ["Self-contained (no gradient scratch); every thread returns after the serial reduction.",
                           "ACCUMULATE=false overwrites s_out[0]; true adds (fuse with other costs)."],
                          ["s_out is the scalar cost output (s_out[0])"] + _cc_state_params[0:3] + _cc_cost_scalar_params + _cc_state_params[3:], None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_cost(T *s_out, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T margin, T weight, "
                           "T *s_sphere_pos, T *s_sphere_r, T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("grid::multi_target_position_device<T, RESOURCE_TIER>(s_sphere_pos, s_q, d_robotModel, d_workspace);")
    self.gen_add_code_line("load_collision_radii<T>(s_sphere_r);")
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < NUM_COLLISION_SPHERES; ++i) {", True)
    self.gen_add_code_line("T nx, ny, nz;")
    self.gen_add_code_line("T d = grid_cc_nearest_obstacle<T>(env, s_sphere_pos[3*i], s_sphere_pos[3*i+1], s_sphere_pos[3*i+2], s_sphere_r[i], &nx, &ny, &nz);")
    self.gen_add_code_line("T viol = margin - d;")
    self.gen_add_code_line("if (viol > static_cast<T>(0)) acc += static_cast<T>(0.5) * weight * viol * viol;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    # COST gradient (over q; cost is q-only) -- reduction over the clearance Jacobian primitive
    self.gen_add_func_doc("collision_cost_gradient: grad_q[vi] = -sum_i (weight*viol_i) d(d_i)/dq_vi  (viol_i = max(0,margin-d_i))",
                          ["Gradient over q only (size NUM_VEL = " + str(nv) + "); built on collision_distance_gradient.",
                           "ACCUMULATE=false overwrites s_grad_q; true adds."],
                          ["s_grad_q is the q-gradient output (size NUM_VEL)"] + _cc_state_params[0:3] + _cc_cost_scalar_params + _cc_state_params[3:] +
                          ["s_normal is caller scratch of size 3*NUM_COLLISION_SPHERES",
                           "s_dist is caller scratch of size NUM_COLLISION_SPHERES",
                           "s_ddist is caller scratch of size NUM_COLLISION_SPHERES*NUM_VEL (sphere-major clearance Jacobian)",
                           "s_pos_grad is caller scratch of size 3*NUM_VEL*NUM_COLLISION_SPHERES (batched dp/dq)"], None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_cost_gradient(T *s_grad_q, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T margin, T weight, "
                           "T *s_sphere_pos, T *s_sphere_r, T *s_normal, T *s_dist, T *s_ddist, T *s_pos_grad, "
                           "T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("collision_distance_gradient<T, RESOURCE_TIER>(s_dist, s_ddist, s_q, d_robotModel, env, s_sphere_pos, s_sphere_r, s_normal, s_pos_grad, d_workspace);")
    self.gen_add_code_line("// grad_q[vi] = sum_i (weight*viol_i) * d(viol_i)/dq_vi, with d(viol)/dq = -d(clearance)/dq = -s_ddist")
    self.gen_add_parallel_loop("vi", str(nv))
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < NUM_COLLISION_SPHERES; ++i) {", True)
    self.gen_add_code_line("T viol = margin - s_dist[i];")
    self.gen_add_code_line("if (viol > static_cast<T>(0)) g += (weight * viol) * s_ddist[i*" + str(nv) + " + vi];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ACCUMULATE) { s_grad_q[vi] += -g; } else { s_grad_q[vi] = -g; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    # COST Gauss-Newton hessian (over q; PSD) -- outer product of the clearance Jacobian over active spheres
    self.gen_add_func_doc("collision_cost_hessian: GN hessian H[vi,vj] = sum_{active i} weight d(d_i)/dq_vi d(d_i)/dq_vj",
                          ["NUM_VEL x NUM_VEL (= " + str(nv) + "x" + str(nv) + ") column-major; PSD by construction; built on "
                           "collision_distance_gradient. GN term only (residual-weighted SDF curvature dropped -- the "
                           "ratified PSD choice; full-Newton collision hessian = labeled TODO).",
                           "ACCUMULATE=false overwrites; true adds."],
                          ["s_hess is the NUM_VEL x NUM_VEL column-major hessian output"] + _cc_state_params[0:3] + _cc_cost_scalar_params + _cc_state_params[3:] +
                          ["s_normal is caller scratch of size 3*NUM_COLLISION_SPHERES",
                           "s_dist is caller scratch of size NUM_COLLISION_SPHERES",
                           "s_ddist is caller scratch of size NUM_COLLISION_SPHERES*NUM_VEL (sphere-major clearance Jacobian)",
                           "s_pos_grad is caller scratch of size 3*NUM_VEL*NUM_COLLISION_SPHERES (batched dp/dq)"], None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void collision_cost_hessian(T *s_hess, const T *s_q, const grid::robotModel<T> *d_robotModel, "
                           "const Environment<T> &env, T margin, T weight, "
                           "T *s_sphere_pos, T *s_sphere_r, T *s_normal, T *s_dist, T *s_ddist, T *s_pos_grad, "
                           "T *d_workspace = nullptr) {", True)
    self.gen_add_code_line("collision_distance_gradient<T, RESOURCE_TIER>(s_dist, s_ddist, s_q, d_robotModel, env, s_sphere_pos, s_sphere_r, s_normal, s_pos_grad, d_workspace);")
    self.gen_add_parallel_loop("ind", str(nv * nv))
    self.gen_add_code_line("int row = ind % " + str(nv) + "; int col = ind / " + str(nv) + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < NUM_COLLISION_SPHERES; ++i) {", True)
    self.gen_add_code_line("if ((margin - s_dist[i]) > static_cast<T>(0)) h += weight * s_ddist[i*" + str(nv) + " + row] * s_ddist[i*" + str(nv) + " + col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    self.gen_add_end_control_flow()  # close namespace grid_collision

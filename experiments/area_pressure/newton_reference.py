"""Pinned Newton collision-only area-pressure reference, never a dynamics step.

Newton/Warp are imported only by newton_two_sphere_case. The pure export layer
uses saved native force-bearing areas, not visualization-triangle areas.
Native normals point shape0 -> shape1; pressure traction on shape0 is negative.
See Newton 1.7 collide._write_contact_at_index and sdf_hydroelastic.decode_contacts.
"""

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np

MAX_FACES = 16384
_STAGE_NAMES = {
    "broad_pairs", "sdf_pairs", "hydro_broad", "iso_l0", "iso_l1", "iso_l2",
    "iso_voxels", "hydro_faces", "rigid_contacts",
}
_DROP_NAMES = {"hydro_buffer", "hydro_hashtable", "global_buffer", "global_hashtable"}
_RAW_NAMES = {
    "count", "position_depth", "normal_oct", "shape_pairs", "contact_area",
    "contact_pressure", "contact_fingerprints", "surface_vertices", "surface_depth",
    "surface_pairs",
}


def _need(condition, message):
    if not condition:
        raise ValueError("Newton pressure export: " + message)


def _integer(value):
    return type(value) is int and value >= 0


def _array(value, shape, name, integer=False):
    _need(isinstance(value, (list, tuple, np.ndarray)), name + " must be an explicit array")
    array = np.asarray(value)
    if array.size == 0:
        _need(math.prod(shape) == 0, name + " shape mismatch")
        array = np.empty(shape, dtype=np.int64 if integer else np.float64)
    _need(array.shape == shape, name + " shape mismatch")
    _need(array.dtype.kind in ("iu" if integer else "fiu"), name + " dtype")
    _need(bool(np.isfinite(array).all()), name + " must be finite")
    result = np.array(array, dtype=np.int64 if integer else np.float64, copy=True)
    _need(bool(np.isfinite(result).all()), name + " exceeds export dtype")
    return result


def decode_octahedral_normals(encoded):
    """Pure FP64 decoding of the saved two-coordinate native normal encoding."""
    _need(isinstance(encoded, (list, tuple, np.ndarray)), "normal encoding type")
    _need(len(encoded) <= MAX_FACES, "normal capacity")
    values = _array(encoded, (len(encoded), 2), "normal_oct")
    _need(bool((np.abs(values) <= 1.0).all()), "normal encoding outside [-1,1]")
    x, y = values[:, 0], values[:, 1]
    z = 1.0 - np.abs(x) - np.abs(y)
    folded = z < 0.0
    nx = np.where(folded, (1.0 - np.abs(y)) * np.where(x < 0, -1, 1), x)
    ny = np.where(folded, (1.0 - np.abs(x)) * np.where(y < 0, -1, 1), y)
    normals = np.column_stack((nx, ny, z))
    lengths = np.linalg.norm(normals, axis=1)
    _need(bool((lengths > 0.0).all()), "zero normal")
    return normals / lengths[:, None]


def _check_overflow(value, count):
    _need(type(value) is dict and set(value) == {"stages", "dropped"}, "overflow schema")
    stages, dropped = value["stages"], value["dropped"]
    _need(type(stages) is list and len(stages) <= 32, "stage counter budget")
    names = set()
    for row in stages:
        _need(type(row) is dict and set(row) == {"name", "count", "capacity"}, "stage schema")
        name = row["name"]
        _need(type(name) is str and name not in names, "duplicate stage counter")
        names.add(name)
        _need(_integer(row["count"]) and _integer(row["capacity"]), "stage counter type")
        _need(row["count"] <= row["capacity"], "stage overflow: " + name)
        if name in ("hydro_faces", "rigid_contacts"):
            _need(row["count"] == count, "face/output counter mismatch")
    _need(_STAGE_NAMES <= names, "missing stage counter")
    _need(type(dropped) is dict and set(dropped) == _DROP_NAMES, "dropped counter schema")
    _need(all(_integer(v) and v == 0 for v in dropped.values()), "dropped contacts or overflow")
    return json.loads(json.dumps(value, allow_nan=False))


def export_unreduced_surface(
    raw, *, points_world, normals_world, shape_a, shape_b, overflow,
    axis_world=(0.0, 0.0, 1.0), origin_world=(0.0, 0.0, 0.0),
):
    """Validate/copy one complete raw capture and integrate its active faces.

    points_world/normals_world are outputs of Warp's original shape-to-world
    transform API. Normal orientation remains raw pair[0] -> pair[1]. Native
    raw row i corresponds to buffer ID i+1 and visualization face i. Speculative
    faces stay in raw; only negative-separation faces enter active quadrature.
    """
    _need(type(raw) is dict and set(raw) == _RAW_NAMES, "raw schema")
    count = raw["count"]
    _need(_integer(count) and count <= MAX_FACES, "face count/capacity")
    _need(_integer(shape_a) and _integer(shape_b) and shape_a != shape_b, "declared shape IDs")
    counters = _check_overflow(overflow, count)
    pd = _array(raw["position_depth"], (count, 4), "position_depth")
    oct_normal = _array(raw["normal_oct"], (count, 2), "normal_oct")
    decode_octahedral_normals(oct_normal)
    pairs = _array(raw["shape_pairs"], (count, 2), "shape_pairs", True)
    surface_pairs = _array(raw["surface_pairs"], (count, 2), "surface_pairs", True)
    _need(np.array_equal(pairs, surface_pairs), "surface pair mismatch")
    _need(bool((((pairs[:, 0] == shape_a) & (pairs[:, 1] == shape_b)) |
                ((pairs[:, 0] == shape_b) & (pairs[:, 1] == shape_a))).all()), "unexpected shape pair")
    depths = _array(raw["surface_depth"], (count,), "surface_depth")
    _need(np.array_equal(depths, pd[:, 3]), "surface depth mismatch")
    area = _array(raw["contact_area"], (count,), "contact_area")
    pressure = _array(raw["contact_pressure"], (count,), "contact_pressure")
    _need(bool((area >= 0).all() and (pressure >= 0).all()), "negative area/pressure")
    active = pd[:, 3] < 0.0
    _need(bool((pressure[~active] == 0.0).all()), "nonpenetrating pressure")
    fingerprints = _array(raw["contact_fingerprints"], (count,), "contact_fingerprints", True)
    _need(bool((fingerprints >= 0).all()) and len(set(fingerprints.tolist())) == count,
          "invalid or duplicate face fingerprint")
    vertices = _array(raw["surface_vertices"], (3 * count, 3), "surface_vertices")
    points = _array(points_world, (count, 3), "points_world")
    normals = _array(normals_world, (count, 3), "normals_world")
    _need(bool((np.abs(np.linalg.norm(normals, axis=1) - 1.0) <= 2e-6).all()), "world normal not unit")
    axis = _array(axis_world, (3,), "axis_world")
    _need(abs(float(np.linalg.norm(axis)) - 1.0) <= 1e-12, "projection axis not unit")
    origin = _array(origin_world, (3,), "origin_world")
    # Shape0 receives repulsive -p*n; raw order may be swapped by SDF resolution.
    first_normals = normals * np.where(pairs[:, 0] == shape_a, -1.0, 1.0)[:, None]
    selected = np.flatnonzero(active)
    qpoints, qnormals, weights, p = points[active], first_normals[active], area[active], pressure[active]
    scalar_load = weights * p
    force = scalar_load[:, None] * qnormals
    moment = np.cross(qpoints - origin, force)
    total_load = float(scalar_load.sum())
    centroid = (scalar_load @ qpoints / total_load).tolist() if total_load > 0 else None
    raw_copy = {
        "count": count, "raw_ids": list(range(1, count + 1)),
        "position_depth": pd.tolist(), "normal_oct": oct_normal.tolist(),
        "shape_pairs": pairs.tolist(), "contact_area": area.tolist(),
        "contact_pressure": pressure.tolist(), "contact_fingerprints": fingerprints.tolist(),
        "surface_vertices": vertices.tolist(), "surface_depth": depths.tolist(),
        "surface_pairs": surface_pairs.tolist(), "points_world": points.tolist(),
        "normal_shape0_to_shape1_world": normals.tolist(),
    }
    result = {
        "phase": "PRESCRIBED_POSE_COLLISION_ONLY", "status": "CAPTURED_UNREDUCED",
        "unreduced_face_count": count, "active_face_count": len(selected),
        "raw": raw_copy, "overflow": counters,
        "quadrature": {
            "vertices_world": vertices.tolist(),
            "triangles": np.arange(3 * count).reshape(count, 3).tolist(),
            "points_world": qpoints.tolist(), "normals_world": qnormals.tolist(),
            "area_weights_m2": weights.tolist(), "triangle_indices": selected.tolist(),
        },
        "pressure_pa": p.tolist(), "origin_world": origin.tolist(),
        "first_force_per_point_n": force.tolist(), "second_force_per_point_n": (-force).tolist(),
        "first_moment_per_point_nm": moment.tolist(), "second_moment_per_point_nm": (-moment).tolist(),
        "area_m2": float(weights.sum()),
        "projected_area_m2": float(weights @ np.abs(qnormals @ axis)),
        "projection_axis_world": axis.tolist(), "pressure_load_n": total_load,
        "first_force_n": force.sum(axis=0).tolist(), "second_force_n": (-force).sum(axis=0).tolist(),
        "first_moment_nm": moment.sum(axis=0).tolist(), "second_moment_nm": (-moment).sum(axis=0).tolist(),
        "pressure_centroid_world": centroid, "cop_world": None,
    }
    json.dumps(result, allow_nan=False)
    return result


def _count(array):
    if array is None:
        return 0
    data = np.asarray(array.numpy())
    _need(data.size >= 1 and data.dtype.kind in "iu", "native counter layout")
    value = int(data.reshape(-1)[0])
    _need(value >= 0, "negative native counter")
    return value


def _capture_overflow(pipeline, contacts):
    hydro, narrow = pipeline.hydroelastic_sdf, pipeline.narrow_phase
    reducer = hydro.contact_reduction.reducer
    stages = []

    def stage(name, count, capacity):
        stages.append({"name": name, "count": int(count), "capacity": int(capacity)})

    stage("broad_pairs", _count(pipeline.broad_phase_pair_count), pipeline.broad_phase_shape_pairs.shape[0])
    for name, count_name, buffer_name in (
        ("sdf_pairs", "shape_pairs_sdf_sdf_count", "shape_pairs_sdf_sdf"),
        ("gjk_pairs", "gjk_candidate_pairs_count", "gjk_candidate_pairs"),
        ("mesh_pairs", "shape_pairs_mesh_count", "shape_pairs_mesh"),
        ("triangles", "triangle_pairs_count", "triangle_pairs"),
        ("mesh_plane", "shape_pairs_mesh_plane_count", "shape_pairs_mesh_plane"),
        ("mesh_mesh", "shape_pairs_mesh_mesh_count", "shape_pairs_mesh_mesh"),
        ("split_gjk", "split_gjk_work_count", "split_gjk_work_items"),
        ("split_manifold", "split_manifold_work_count", "split_manifold_work_items"),
    ):
        buffer = getattr(narrow, buffer_name)
        stage(name, _count(getattr(narrow, count_name)), 0 if buffer is None else buffer.shape[0])
    if narrow.split_gjk_mpr:
        source = pipeline.broad_phase_pair_count if narrow.sparse_gjk_pairs else narrow.gjk_candidate_pairs_count
        stage("split_query", _count(source), narrow.split_query_results.shape[0])
    stage("hydro_broad", _count(hydro.block_broad_collide_count), hydro.max_num_blocks_broad)
    iso_counts = hydro.iso_buffer_counts.numpy()
    for i in range(3):
        stage("iso_l" + str(i), int(iso_counts[i + 1]), hydro.iso_max_dims[i])
    stage("iso_voxels", _count(hydro.iso_voxel_count), hydro.max_num_iso_voxels)
    stage("hydro_faces", _count(reducer.contact_count), hydro.max_num_face_contacts)
    stage("rigid_contacts", _count(contacts.rigid_contact_count), contacts.rigid_contact_max)
    other = narrow.global_contact_reducer
    return {"stages": stages, "dropped": {
        "hydro_buffer": _count(reducer.buffer_overflows),
        "hydro_hashtable": _count(reducer.ht_insert_failures),
        "global_buffer": 0 if other is None else _count(other.buffer_overflows),
        "global_hashtable": 0 if other is None else _count(other.ht_insert_failures),
    }}


def _positive(value, name):
    _need(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)), name + " type")
    value = float(value)
    _need(math.isfinite(value) and value > 0.0, name + " must be positive finite")
    return value


def newton_two_sphere_case(
    radius_m=0.02, modulus_a_pa=1e5, modulus_b_pa=3e5,
    overlap_m=0.0004, voxel_size_m=0.0005, *, device="cuda:0",
):
    """Run one real Newton collision pass, including at exactly zero overlap.

    No solver or time integration is constructed. The two independent body
    poses are prescribed: A=(0,0,0), B=(0,0,2R-overlap). This is an external
    reference with pinned internal readback, not a Phymeo runtime interface.
    """
    radius = _positive(radius_m, "radius")
    ea, eb = _positive(modulus_a_pa, "modulus_a"), _positive(modulus_b_pa, "modulus_b")
    voxel = _positive(voxel_size_m, "voxel_size")
    _need(isinstance(overlap_m, (int, float, np.integer, np.floating)) and not isinstance(overlap_m, (bool, np.bool_)), "overlap type")
    overlap = float(overlap_m)
    _need(math.isfinite(overlap) and 0 <= overlap <= radius, "overlap outside [0,R]")
    _need(radius <= 1.0 and radius / voxel <= 128 and max(ea, eb) <= 1e12, "reference input resource bound")
    _need(device == "cuda:0", "this frozen reference requires cuda:0")
    _need(version("newton") == "1.7.0" and version("warp-lang") == "1.17.0", "reference dependency versions")
    import newton
    import warp as wp
    from newton.geometry import HydroelasticSDF

    config = HydroelasticSDF.Config(
        reduce_contacts=False, pre_prune_contacts=False, output_contact_surface=True,
        mc_edge_clamp_min=0.0,
    )
    distance = 2 * radius - overlap
    builder = newton.ModelBuilder()
    shape_ids = []
    for label, z, modulus in (("A", 0.0, ea), ("B", distance, eb)):
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, z), wp.quat_identity()), label=label)
        cfg = builder.ShapeConfig(
            is_hydroelastic=True, kh=modulus / radius, margin=0.0, gap=0.0,
            sdf_target_voxel_size=voxel, sdf_texture_format="float32",
        )
        shape_ids.append(builder.add_shape_sphere(body, radius=radius, cfg=cfg, label=label))
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model, reduce_contacts=False, rigid_contact_max=MAX_FACES,
        sdf_hydroelastic_config=config, verify_buffers=True,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    wp.synchronize_device(device)
    hydro = pipeline.hydroelastic_sdf
    _need(hydro is not None, "hydroelastic route unavailable")
    reducer = hydro.contact_reduction.reducer
    count = _count(reducer.contact_count)
    _need(count <= MAX_FACES, "face capture capacity")
    overflow = _capture_overflow(pipeline, contacts)
    _check_overflow(overflow, count)
    surface = hydro.get_contact_surface()
    _need(surface is not None and _count(surface.face_contact_count) == count, "surface counter mismatch")
    raw = {"count": count}
    for field in ("position_depth", "normal", "shape_pairs", "contact_area", "contact_pressure", "contact_fingerprints"):
        raw["normal_oct" if field == "normal" else field] = getattr(reducer, field).numpy()[1:count + 1].copy()
    raw["surface_vertices"] = surface.contact_surface_point.numpy()[:3 * count].copy()
    raw["surface_depth"] = surface.contact_surface_depth.numpy()[:count].copy()
    raw["surface_pairs"] = surface.contact_surface_shape_pair.numpy()[:count].copy()
    body_q = state.body_q.numpy()
    local_shapes = model.shape_transform.numpy()
    shape_bodies = model.shape_body.numpy()

    def transform(values):
        return wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:]))

    transforms = [wp.transform_multiply(transform(body_q[int(body)]), transform(local_shapes[i]))
                  for i, body in enumerate(shape_bodies)]
    local_normals = decode_octahedral_normals(raw["normal_oct"])
    points, normals = [], []
    for i, pair in enumerate(raw["shape_pairs"]):
        x_ws = transforms[int(pair[1])]
        points.append(list(wp.transform_point(x_ws, wp.vec3(*raw["position_depth"][i, :3]))))
        normals.append(list(wp.transform_vector(x_ws, wp.vec3(*local_normals[i]))))
    result = export_unreduced_surface(raw, points_world=points, normals_world=normals,
                                     shape_a=int(shape_ids[0]), shape_b=int(shape_ids[1]), overflow=overflow)
    package_root = Path(newton.__file__).resolve().parent
    sources = {}
    for name in ("_src/sim/builder.py", "_src/sim/collide.py", "_src/geometry/sdf_texture.py",
                 "_src/geometry/sdf_hydroelastic.py", "_src/geometry/contact_reduction_hydroelastic.py",
                 "_src/geometry/contact_reduction_global.py", "_src/geometry/narrow_phase.py"):
        path = package_root / name
        sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    sdf_rows = model._texture_sdf_data.numpy()
    sdf_indices = model._shape_sdf_index.numpy()
    grids = []
    for shape in shape_ids:
        sdf_index = int(sdf_indices[shape])
        _need(sdf_index >= 0, "missing primitive SDF")
        row = sdf_rows[sdf_index]
        metadata = {name: np.asarray(row[name]).tolist() for name in (
            "sdf_box_lower", "sdf_box_upper", "inv_sdf_dx", "subgrid_size", "num_subgrids",
            "voxel_size", "voxel_radius", "subgrids_min_sdf_value", "subgrids_sdf_value_range",
            "paired_samples", "scale_baked",
        )}
        texture = model._texture_sdf_coarse_textures[sdf_index]
        subtexture = model._texture_sdf_subgrid_textures[sdf_index]
        metadata.update(shape_id=int(shape), sdf_index=sdf_index,
                        coarse_texture_dimensions=[texture.width, texture.height, texture.depth],
                        subgrid_texture_dimensions=[subtexture.width, subtexture.height, subtexture.depth])
        grids.append(metadata)
    result.update(
        schema="newton-area-pressure-two-sphere-v1",
        inputs={"radius_m": radius, "modulus_a_pa": ea, "modulus_b_pa": eb,
                "overlap_m": overlap, "voxel_size_m": voxel, "center_a_m": [0.0, 0.0, 0.0],
                "center_b_m": [0.0, 0.0, distance]},
        engine={"newton_version": version("newton"), "warp_version": version("warp-lang"),
                "newton_import_path": str(Path(newton.__file__).resolve()),
                "warp_import_path": str(Path(wp.__file__).resolve()), "source_files": sources,
                "device": device, "solver_steps": 0, "collision_passes": 1,
                "upstream_release_commit": "7e75abe28b8d69522bc0b34837aaed2c303dff5a",
                "release_commit_is_local_vcs_proof": False},
        representation={"pressure_law": "p=-kh*signed_depth; sphere kh=E/R",
                        "kh_pa_per_m": [ea / radius, eb / radius], "margin_m": 0.0, "gap_m": 0.0,
                        "requested_voxel_size_m": voxel, "sdf_texture_format": "float32",
                        "sdf_narrow_band_range_m": list(cfg.sdf_narrow_band_range),
                        "sdf_padding_m": cfg.sdf_padding,
                        "hydro_config": dict(vars(config)), "rigid_contact_max": MAX_FACES,
                        "pipeline_reduce_contacts": False, "snapshot_phase": "same collide return"},
        shape_bindings={"declared_a": int(shape_ids[0]), "declared_b": int(shape_ids[1]),
                        "shape_body": shape_bodies.tolist(), "body_q_xyzw": body_q.tolist(),
                        "shape_local_transform_xyzw": local_shapes.tolist(),
                        "shape_world_transform_xyzw": [list(value) for value in transforms]},
        sdf_grids=grids,
    )
    json.dumps(result, allow_nan=False)
    return result

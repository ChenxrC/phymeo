"""Newton 1.7 native VBD self-contact detector, color accumulation, and discrete state step."""

import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton._src.geometry.kernels import triangle_closest_point
from newton._src.solvers.vbd.particle_vbd_kernels import accumulate_self_contact_force_and_hessian

from .oracles import check_forces, feature_sets, weights_check


@wp.kernel
def _vt_read(
    pos: wp.array(dtype=wp.vec3),
    tri: wp.array2d(dtype=wp.int32),
    pairs: wp.array2d(dtype=wp.int32),
    out: wp.array2d(dtype=wp.vec3),
):
    i = wp.tid()
    v, t = pairs[i, 0], pairs[i, 1]
    closest, bary, _feature = triangle_closest_point(
        pos[tri[t, 0]], pos[tri[t, 1]], pos[tri[t, 2]], pos[v]
    )
    out[i, 0] = pos[v]
    out[i, 1] = closest
    out[i, 2] = bary


@wp.kernel
def _ee_read(
    pos: wp.array(dtype=wp.vec3),
    edge: wp.array2d(dtype=wp.int32),
    pairs: wp.array2d(dtype=wp.int32),
    out: wp.array2d(dtype=wp.vec3),
):
    i = wp.tid()
    a, b = pairs[i, 0], pairs[i, 1]
    a0, a1 = pos[edge[a, 2]], pos[edge[a, 3]]
    b0, b1 = pos[edge[b, 2]], pos[edge[b, 3]]
    st = wp.closest_point_edge_edge(a0, a1, b0, b1, 1.0e-5)
    out[i, 0] = a0 + st[0] * (a1 - a0)
    out[i, 1] = b0 + st[1] * (b1 - b0)
    out[i, 2] = st


def _pairs(det, stem):
    counts = getattr(det, stem + "_count").numpy()
    offsets = getattr(det, stem + "_offsets").numpy()
    values = getattr(det, stem).numpy().reshape(-1, 2)
    assert np.all(counts <= np.diff(offsets)), "native self-contact buffer overflow"
    pairs = []
    for i, n in enumerate(counts):
        records = values[offsets[i] : offsets[i] + n]
        assert np.all(records[:, 0] == i)
        pairs.extend(records.tolist())
    return pairs


def _measure(kind, pairs, model, state, ids):
    if not pairs:
        return []
    arr = wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.int32, device=model.device)
    out = wp.empty((len(pairs), 3), dtype=wp.vec3, device=model.device)
    topo = model.tri_indices if kind == "VT" else model.edge_indices
    wp.launch(
        _vt_read if kind == "VT" else _ee_read,
        dim=len(pairs),
        inputs=[state.particle_q, topo, arr],
        outputs=[out],
        device=model.device,
    )
    geom = out.numpy()
    index = topo.numpy()
    rows = []
    for (a, b), (p, q, params) in zip(pairs, geom, strict=True):
        if kind == "VT":
            ma, mb = [ids[a]], [ids[j] for j in index[b]]
            wa, wb = [1.0], params.tolist()
        else:
            ma, mb = [ids[j] for j in index[a, 2:4]], [ids[j] for j in index[b, 2:4]]
            s, t, _ = map(float, params)
            wa, wb = [1 - s, s], [1 - t, t]
        rows.append(
            dict(
                raw_indices=[a, b],
                material_a=ma,
                material_b=mb,
                weights_a=list(zip(ma, wa, strict=True)),
                weights_b=list(zip(mb, wb, strict=True)),
                point_a_m=p.tolist(),
                point_b_m=q.tolist(),
                distance_m=float(np.linalg.norm(p.astype(float) - q)),
            )
        )
    return rows


def observe(case, device):
    assert version("newton") == "1.7.0"
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    selected = wp.get_device(device)
    assert selected.is_cuda == (device == "cuda:0")
    inp = case["inputs"]
    with wp.ScopedDevice(selected):
        builder = newton.ModelBuilder()
        newton.solvers.SolverVBD.register_custom_attributes(builder)
        ids = []
        for sheet in inp["sheets"]:
            local = {v[0]: i for i, v in enumerate(sheet["vertices"])}
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[v[1:] for v in sheet["vertices"]],
                indices=[local[i] for f in sheet["faces"] for i in f],
                density=1.0,
                tri_ke=0.0,
                tri_ka=0.0,
                tri_kd=0.0,
                tri_drag=0.0,
                tri_lift=0.0,
                edge_ke=0.0,
                edge_kd=0.0,
                particle_radius=0.001,
                label="sheet-" + str(sheet["body"]),
            )
            ids.extend(local)
        builder.color()
        model = builder.finalize(device=selected)
        model.set_gravity((0.0, 0.0, 0.0))
        model.soft_contact_ke = inp["stiffness_n_per_m"]
        model.soft_contact_kd = 0.0
        model.soft_contact_mu = 0.0
        solver = newton.solvers.SolverVBD(
            model,
            iterations=3,
            particle_enable_self_contact=True,
            particle_self_contact_margin=inp["self_margin_m"],
            particle_self_contact_gap=inp["detection_gap_m"],
            particle_topological_contact_filter_threshold=inp["topological_ring"],
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_enable_tile_solve=False,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            rigid_compliant_alm=False,
        )
        state = model.state()
        p = state.particle_q.numpy().copy()
        det = solver.trimesh_collision_detector
        radius = solver.particle_self_contact_margin + solver.particle_self_contact_gap
        det.refit(state.particle_q)
        det.vertex_triangle_collision_detection(radius)
        det.edge_edge_collision_detection(radius)
        wp.synchronize_device(selected)
        vt = _measure("VT", _pairs(det, "vertex_colliding_triangles"), model, state, ids)
        ee = _measure("EE", _pairs(det, "edge_colliding_edges"), model, state, ids)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=selected)
        hess = wp.zeros(model.particle_count, dtype=wp.mat33, device=selected)
        for color in range(len(model.particle_color_groups)):
            wp.launch(
                accumulate_self_contact_force_and_hessian,
                dim=solver.particle_self_contact_evaluation_kernel_launch_size,
                inputs=[
                    inp["dt_s"],
                    color,
                    state.particle_q,
                    state.particle_q,
                    model.particle_colors,
                    model.tri_indices,
                    model.edge_indices,
                    solver.trimesh_collision_info,
                    solver.particle_self_contact_margin,
                    model.soft_contact_ke,
                    model.soft_contact_kd,
                    model.soft_contact_mu,
                    solver.friction_epsilon,
                    solver._self_contact_edge_edge_parallel_epsilon,
                ],
                outputs=[forces, hess],
                device=selected,
                max_blocks=selected.sm_count,
            )
        force = forces.numpy()
        observed = dict(
            device=str(selected),
            newton=version("newton"),
            warp=version("warp-lang"),
            surface_count=len(builder.surface_label),
            surface_labels=list(builder.surface_label),
            raw_particle_material_ids=ids,
            raw_tri_indices=model.tri_indices.numpy().tolist(),
            raw_edge_indices=model.edge_indices.numpy().tolist(),
            search_radius_m=radius,
            interaction_radius_m=solver.particle_self_contact_margin,
            vt=vt,
            ee_directed=ee,
            resize_flags=det.resize_flags.numpy().tolist(),
            min_distance_to_tri_m=det.vertex_colliding_triangles_min_dist.numpy().tolist(),
            min_distance_to_edge_m=det.edge_colliding_edges_min_dist.numpy().tolist(),
            forces_n={str(i): v.tolist() for i, v in zip(ids, force, strict=True)},
            vbd_colors=len(model.particle_color_groups),
        )
        if inp["step"]:
            out = model.state()
            state.clear_forces()
            solver.step(state, out, model.control(), None, inp["dt_s"])
            wp.synchronize_device(selected)
            observed["step"] = dict(
                dt_s=inp["dt_s"],
                initial_m=p.tolist(),
                final_m=out.particle_q.numpy().tolist(),
                final_velocity_m_s=out.particle_qd.numpy().tolist(),
                particle_mass_kg=model.particle_mass.numpy().tolist(),
            )
        return observed


def validate(data, case):
    inp, exp, tol = case["inputs"], case["expected"], case["tolerances"]
    assert data["surface_count"] == len(inp["sheets"]) and not any(data["resize_flags"])
    oracle_vt, oracle_ee = feature_sets(inp)
    mapped_vt = {(r["material_a"][0], tuple(sorted(r["material_b"]))): r for r in data["vt"]}
    pairs = [
        tuple(sorted((tuple(sorted(r["material_a"])), tuple(sorted(r["material_b"])))))
        for r in data["ee_directed"]
    ]
    assert len(mapped_vt) == len(data["vt"]) and mapped_vt.keys() == oracle_vt.keys(), (
        mapped_vt.keys() ^ oracle_vt.keys()
    )
    assert Counter(pairs) == Counter({k: 2 for k in oracle_ee}), set(pairs) ^ oracle_ee.keys()
    for key, row in mapped_vt.items():
        assert abs(row["distance_m"] - oracle_vt[key]) <= tol["distance_m"], (
            key,
            row["distance_m"],
            oracle_vt[key],
        )
    for key, row in zip(pairs, data["ee_directed"], strict=True):
        assert abs(row["distance_m"] - oracle_ee[key]) <= tol["distance_m"]
    if case["family"] in ("vt", "fold"):
        v, t = exp["material_target"]
        record = mapped_vt.get((v, tuple(t)))
    elif case["family"] == "ee":
        a, b = exp["material_target"]
        record = next(
            (
                r
                for r in data["ee_directed"]
                if sorted(r["material_a"]) == a and sorted(r["material_b"]) == b
            ),
            None,
        )
    else:
        record = None
    assert (record is not None) == exp["target_in_detection"]
    if record is not None:
        assert abs(record["distance_m"] - exp["focus_distance_m"]) <= tol["distance_m"]
        if case["family"] == "vt":
            weights_check(record["weights_b"], exp["focus_weights_a"], tol["weight"])
        if case["family"] == "ee":
            weights_check(record["weights_a"], exp["focus_weights_a"], tol["weight"])
            weights_check(record["weights_b"], exp["focus_weights_b"], tol["weight"])
    check_forces(data, case, "newton")
    if "step" in data:
        frame = data["step"]
        dx = np.asarray(frame["final_m"]) - frame["initial_m"]
        assert np.isfinite(dx).all()
        if exp["newton_unique_force_n"] > 0:
            index = {v: i for i, v in enumerate(data["raw_particle_material_ids"])}
            for material, sign in ((exp["focus_weights_a"], -1), (exp["focus_weights_b"], 1)):
                assert sign * sum(dx[index[v], 2] * w for v, w in material) > 1e-7
        else:
            assert np.linalg.norm(dx, axis=1).max() < 2e-6

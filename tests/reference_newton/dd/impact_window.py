"""112 true Newton VBD steps followed by frozen-state native soft self-contact force sampling."""

import argparse
import json
import sys
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from math import sqrt
from pathlib import Path
from time import perf_counter_ns

import newton
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dd.newton_runner import _measure, _pairs, accumulate_self_contact_force_and_hessian  # noqa: E402
from dd.oracles import ee, vt  # noqa: E402

H = 1 / 256
RADIUS = 1 / 16
SEARCH = 3 / 32
K = 128.0
STEPS = 112
IDS = (9, 13, 27, 52, 73, 28)
TRI_EDGES = ((0, 1), (1, 2), (2, 0))


def fixture(family):
    if family == "VT":
        x = [
            [-2, -1, 0],
            [2, -1, 0],
            [0, 2, 0],
            [0, 0, SEARCH],
            [0.125, 0, 1 + SEARCH],
            [0, 0.125, 1 + SEARCH],
        ]
        wa, wb = np.full(3, 1 / 3), np.array([1.0, 0, 0])
        rot = np.eye(3)
    else:
        x = [[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, -1, SEARCH], [0, 1, SEARCH], [0, 0, 1 + SEARCH]]
        wa = wb = np.array([0.5, 0.5, 0])
        rot = np.array([[0.0, 0, 1], [1.0, 0, 0], [0, 1.0, 0]])
    n = rot[:, 2]
    v = np.tile([0.125, -0.0625, 0.0], (6, 1))
    v[3:] -= 0.5 * n
    return (
        (np.asarray(x) @ rot.T + [0.25, -0.25, 0.125]).astype(np.float32),
        v.astype(np.float32),
        wa,
        wb,
        n,
    )


def make_solver(x, v, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    for first in (0, 3):
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(*v[first]),
            vertices=x[first : first + 3].tolist(),
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.0,
            label=f"sheet-{first // 3}",
        )
    builder.particle_mass[:] = [1.0] * 6
    builder.color()
    m = builder.finalize(device=device)
    m.set_gravity((0.0, 0.0, 0.0))
    m.soft_contact_ke = K
    m.soft_contact_kd = 0.0
    m.soft_contact_mu = 0.0
    np.testing.assert_array_equal(m.particle_mass.numpy(), np.ones(6))
    np.testing.assert_array_equal(m.particle_inv_mass.numpy(), np.ones(6))
    np.testing.assert_array_equal(m.tri_indices.numpy(), [[0, 1, 2], [3, 4, 5]])
    assert (
        not np.any(m.gravity.numpy())
        and not np.any(m.tri_materials.numpy())
        and not np.any(m.edge_bending_properties.numpy())
    )
    solver = newton.solvers.SolverVBD(
        m,
        iterations=3,
        particle_enable_self_contact=True,
        particle_self_contact_margin=RADIUS,
        particle_self_contact_gap=SEARCH - RADIUS,
        particle_topological_contact_filter_threshold=2,
        particle_rest_shape_contact_exclusion_radius=0.0,
        particle_enable_tile_solve=False,
        particle_vertex_contact_buffer_size=64,
        particle_edge_contact_buffer_size=64,
        rigid_compliant_alm=False,
    )
    assert (
        solver.particle_self_contact_margin == RADIUS
        and solver.particle_self_contact_margin + solver.particle_self_contact_gap == SEARCH
    )
    return m, solver


def actual_states(m, solver, device):
    current, next_state = m.state(), m.state()
    controls = m.control(clone_variables=False)
    frames = []
    activity = {}
    step_times = []
    state_read_ns = 0
    for tick in range(STEPS + 1):
        started = perf_counter_ns()
        frames.append((current.particle_q.numpy().copy(), current.particle_qd.numpy().copy()))
        state_read_ns += perf_counter_ns() - started
        if tick == STEPS:
            break
        current.clear_forces()
        probe = device.is_cuda and tick == 40
        if probe:
            wp.timing_begin()
        try:
            started = perf_counter_ns()
            solver.step(current, next_state, controls, None, H)
            wp.synchronize_device(device)
            step_times.append(perf_counter_ns() - started)
        finally:
            if probe:
                events = wp.timing_end()
        if probe:
            activity = dict(Counter(e.name for e in events if e.filter == wp.TIMING_KERNEL))
            assert any("accumulate_self_contact_force_and_hessian" in name for name in activity)
        current, next_state = next_state, current
    return (
        frames,
        activity,
        dict(
            solver_step_and_explicit_sync_ns=step_times,
            observed_position_velocity_reads_ns=state_read_ns,
        ),
    )


def face_distance(x):
    a, b = x[:3].tolist(), x[3:].tolist()
    distances = [vt(p, b) for p in a] + [vt(p, a) for p in b]
    distances.extend(ee((a[i], a[j]), (b[k], b[end])) for i, j in TRI_EDGES for k, end in TRI_EDGES)
    return sqrt(max(0.0, min(distances)))


def predicted_forces(vt_rows, ee_rows):
    """Newton's published elastic potential derivative, scattered on the detector's material sites."""
    result = np.zeros((6, 3))
    for kind, features in (("VT", vt_rows), ("EE", ee_rows)):
        for row in features:
            delta = np.array(row["point_a_m"]) - row["point_b_m"]
            d = float(np.linalg.norm(delta))
            if not 0 < d < RADIUS:
                continue
            tau = RADIUS * 0.5
            floor = min(1.0e-5, tau * 0.5)
            if d >= tau:
                magnitude = K * (RADIUS - d)
            elif d > floor:
                magnitude = K * tau**2 / d
            else:
                magnitude = K * tau**2 * (2 * floor - d) / floor**2
            f = magnitude * delta / d
            for node, w in row["weights_a"]:
                result[IDS.index(node)] += w * f
            if kind == "VT":
                for node, w in row["weights_b"]:
                    result[IDS.index(node)] -= w * f
            # Newton's directed EE record deposits on edge a; the reversed record owns edge b.
    return result


def observe(family, m, solver, frames, wa, wb, n, device):
    # All physical steps are finished. Refits can no longer affect the actual VBD trajectory.
    det = solver.trimesh_collision_detector
    probe = m.state()
    prior = wp.empty(6, dtype=wp.vec3, device=device)
    forces = wp.zeros(6, dtype=wp.vec3, device=device)
    hess = wp.zeros(6, dtype=wp.mat33, device=device)
    p0 = frames[0][1].astype(float)
    rows = []
    for tick, (x32, v32) in enumerate(frames):
        x, v = x32.astype(float), v32.astype(float)
        assert x.shape == v.shape == (6, 3) and np.isfinite(x).all() and np.isfinite(v).all()
        started = perf_counter_ns()
        probe.particle_q.assign(x32)
        prior.assign(frames[max(0, tick - 1)][0])
        det.refit(probe.particle_q)
        det.vertex_triangle_collision_detection(SEARCH)
        det.edge_edge_collision_detection(SEARCH)
        wp.synchronize_device(device)
        assert not np.any(det.resize_flags.numpy())
        vts = _measure("VT", _pairs(det, "vertex_colliding_triangles"), m, probe, IDS)
        ees = _measure("EE", _pairs(det, "edge_colliding_edges"), m, probe, IDS)
        forces.zero_()
        hess.zero_()
        for color in range(len(m.particle_color_groups)):
            wp.launch(
                accumulate_self_contact_force_and_hessian,
                dim=solver.particle_self_contact_evaluation_kernel_launch_size,
                inputs=[
                    H,
                    color,
                    prior,
                    probe.particle_q,
                    m.particle_colors,
                    m.tri_indices,
                    m.edge_indices,
                    solver.trimesh_collision_info,
                    RADIUS,
                    m.soft_contact_ke,
                    m.soft_contact_kd,
                    m.soft_contact_mu,
                    solver.friction_epsilon,
                    solver._self_contact_edge_edge_parallel_epsilon,
                ],
                outputs=[forces, hess],
                device=device,
                max_blocks=device.sm_count,
            )
        wp.synchronize_device(device)
        f = forces.numpy().astype(float)
        diagnostic_ns = perf_counter_ns() - started
        expected = predicted_forces(vts, ees)
        assert np.isfinite(f).all()
        np.testing.assert_allclose(f, expected, rtol=2e-4, atol=0.002)
        np.testing.assert_allclose(f.sum(axis=0), 0, rtol=0, atol=0.002)
        g = float((wb @ x[3:] - wa @ x[:3]) @ n)
        u = float((wb @ v[3:] - wa @ v[:3]) @ n)
        active_vt = sum(s["distance_m"] < RADIUS for s in vts)
        active_ee = sum(s["distance_m"] < RADIUS for s in ees)
        prev_v = frames[max(0, tick - 1)][1].astype(float)
        rows.append(
            dict(
                tick=tick,
                time_s=tick * H,
                native_diagnostic_and_readback_ns=diagnostic_ns,
                position_m=x.tolist(),
                velocity_m_s=v.tolist(),
                material_projected_gap_m=g,
                closest_triangle_gap_m=face_distance(x),
                material_relative_velocity_m_s=u,
                normal_force_frozen_kernel_n=float(f[3:].sum(axis=0) @ n),
                frozen_nodal_force_n=f.tolist(),
                force_oracle_max_error_n=float(np.max(np.abs(f - expected))),
                upper_total_momentum_change_ns=(v[3:] - p0[3:]).sum(axis=0).tolist(),
                upper_step_momentum_impulse_n_s=float((v[3:] - prev_v[3:]).sum(axis=0) @ n),
                total_momentum_error_ns=float(np.max(np.abs((v - p0).sum(axis=0)))),
                kinetic_energy_j=float(np.sum(v * v) / 2),
                active_vt=active_vt,
                active_ee_directed=active_ee,
                detector_vt=vts,
                detector_ee_directed=ees,
            )
        )
    return rows


def summarize(rows, n):
    key = "normal_force_frozen_kernel_n"
    ids = [row["tick"] for row in rows if row[key] > 0.001]
    first, last = (ids[0], ids[-1]) if ids else (None, None)
    peak = max(rows, key=lambda row: row[key])
    final = rows[-1]
    leave = rows[last + 1] if last is not None and last < STEPS else None
    return dict(
        first_loaded_tick=first,
        peak_tick=peak["tick"],
        last_loaded_tick=last,
        first_unloaded_tick=leave["tick"] if leave else None,
        full_loading_and_unloading=first is not None
        and 0 < first < peak["tick"] < last < STEPS
        and ids == list(range(first, last + 1)),
        peak_frozen_kernel_n=peak[key],
        min_closest_gap_m=min(r["closest_triangle_gap_m"] for r in rows),
        min_material_projected_gap_m=min(r["material_projected_gap_m"] for r in rows),
        initial_speed_m_s=rows[0]["material_relative_velocity_m_s"],
        entry_speed_m_s=rows[first]["material_relative_velocity_m_s"] if ids else None,
        first_unloaded_speed_m_s=leave["material_relative_velocity_m_s"] if leave else None,
        outgoing_speed_m_s=final["material_relative_velocity_m_s"],
        final_gap_m=final["closest_triangle_gap_m"],
        upper_momentum_impulse_n_s=float(np.asarray(final["upper_total_momentum_change_ns"]) @ n),
        left_sampled_force_integral_n_s=sum(r[key] * H for r in rows[:-1]),
        max_total_momentum_error_n_s=max(r["total_momentum_error_ns"] for r in rows),
        max_force_oracle_error_n=max(r["force_oracle_max_error_n"] for r in rows),
        initial_kinetic_j=rows[0]["kinetic_energy_j"],
        final_kinetic_j=final["kinetic_energy_j"],
        active_feature_counts_at_peak=[peak["active_vt"], peak["active_ee_directed"]],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--family", choices=("VT", "EE"), required=True)
    p.add_argument("--phymeo-receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("provide a fresh receipt")
    assert (
        version("newton") == "1.7.0"
        and version("warp-lang") == "1.17.0"
        and "phymeo" not in sys.modules
    )
    old_bytes = a.phymeo_receipt.read_bytes()
    old = json.loads(old_bytes)
    assert old["family"] == a.family and old["steps"] == STEPS and old["h"] == H
    x0, v0, wa, wb, n = fixture(a.family)
    initial = old["steps_observed"][0]
    np.testing.assert_array_equal(x0, initial["position_m"])
    np.testing.assert_array_equal(v0, initial["velocity_m_s"])
    assert (wb @ x0[3:] - wa @ x0[:3]) @ n == SEARCH and (wb @ v0[3:] - wa @ v0[:3]) @ n == -0.5
    a.cache_root.mkdir(exist_ok=True, parents=True)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    wp.config.kernel_cache_dir = str(a.cache_root.resolve())
    wp.init()
    device = wp.get_device(a.device)
    assert device.is_cuda == (a.device == "cuda:0")
    with wp.ScopedDevice(device):
        m, solver = make_solver(x0, v0, device)
        frames, activity, costs = actual_states(m, solver, device)
        np.testing.assert_array_equal(frames[0][0], x0)
        np.testing.assert_array_equal(frames[0][1], v0)
        rows = observe(a.family, m, solver, frames, wa, wb, n, device)
    summary = summarize(rows, n)
    ref_keys = (
        "first_loaded_step",
        "peak_step",
        "last_loaded_step",
        "peak_n",
        "min_gap_m",
        "outgoing_m_s",
        "total_impulse_ns",
        "max_momentum_error_ns",
    )
    ref_upper_change = (
        np.asarray(old["steps_observed"][-1]["velocity_m_s"])[3:]
        - np.asarray(initial["velocity_m_s"])[3:]
    )
    comparison = {k: old[k] for k in ref_keys}
    comparison["upper_momentum_impulse_n_s"] = float(ref_upper_change.sum(axis=0) @ n)
    report = dict(
        engine=version("newton"),
        warp=version("warp-lang"),
        device=str(device),
        device_name=device.name,
        family=a.family,
        steps=STEPS,
        dt_s=H,
        costs=costs,
        particle_mass_kg=m.particle_mass.numpy().tolist(),
        material_ids=IDS,
        weights=[wa.tolist(), wb.tolist()],
        normal=n.tolist(),
        effective_material_mass_kg=float(1 / (wa @ wa + wb @ wb)),
        interaction_radius_m=solver.particle_self_contact_margin,
        search_radius_m=SEARCH,
        detector_gap_m=solver.particle_self_contact_gap,
        gravity_m_s2=m.gravity.numpy().tolist(),
        triangles=m.tri_indices.numpy().tolist(),
        edges=m.edge_indices.numpy().tolist(),
        stiffness_n_m=K,
        damping=0,
        friction=0,
        structural_stiffness=0,
        solver="newton.solvers.SolverVBD",
        iterations=3,
        color_groups=[c.numpy().tolist() for c in m.particle_color_groups],
        cuda_kernels_on_actual_step_41=activity,
        force_semantics="Native color force/Hessian kernel independently evaluated after motion at each recorded end state, with refreshed pairs. VBD CONTACT_F does not export soft self-contact forces. Actual upper momentum changes are separate observables.",
        reference=dict(
            path=str(a.phymeo_receipt),
            sha256=sha256(old_bytes).hexdigest(),
            mode=old["mode"],
            values=comparison,
        ),
        summary=summary,
        states=rows,
    )
    a.out.write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf8")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

"""Newton 1.7 VBD reference for the preloaded sliding vertex on a finite fixed triangle."""

import argparse
import json
import sys
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dd.newton_runner import _measure, _pairs, accumulate_self_contact_force_and_hessian  # noqa: E402


def solve(reference, device):
    slots = reference["slot_for_material_id"]
    x = np.asarray(reference["initial_position_m"], np.float32)[slots]
    v = np.asarray(reference["initial_velocity_m_s"], np.float32)[slots]
    external = np.asarray(reference["external_force_n"], np.float32)[slots]
    n = np.asarray(reference["normal_world"])
    t = np.asarray(reference["tangent_world"])
    h = reference["dt_s"]
    assert h == 1 / 64 and len(reference["samples"]) == 32
    np.testing.assert_array_equal(v[:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(v[3:], np.tile(t, (3, 1)))
    np.testing.assert_array_equal(external[3], -2 * n)
    assert not np.any(np.delete(external, 3, axis=0))
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
        )
    builder.particle_mass[:] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    builder.color()
    m = builder.finalize(device=device)
    m.set_gravity((0.0, 0.0, 0.0))
    m.soft_contact_ke = 64.0
    m.soft_contact_mu = 0.25
    m.soft_contact_kd = 0.0
    np.testing.assert_array_equal(m.particle_inv_mass.numpy(), [0, 0, 0, 1, 1, 1])
    assert not np.any(m.tri_materials.numpy()) and not np.any(m.edge_bending_properties.numpy())
    s = newton.solvers.SolverVBD(
        m,
        iterations=3,
        friction_epsilon=1 / 32,
        particle_enable_self_contact=True,
        particle_self_contact_margin=1 / 16,
        particle_self_contact_gap=1 / 32,
        particle_topological_contact_filter_threshold=2,
        particle_rest_shape_contact_exclusion_radius=0.0,
        particle_enable_tile_solve=False,
        rigid_compliant_alm=False,
    )
    cur, nxt = m.state(), m.state()
    ctrl = m.control(clone_variables=False)
    states = []
    native = {}
    for i in range(33):
        states.append((cur.particle_q.numpy().copy(), cur.particle_qd.numpy().copy()))
        if i < 32:
            cur.clear_forces()
            cur.particle_f.assign(external)
            probe = device.is_cuda and i == 15
            if probe:
                wp.timing_begin()
            try:
                s.step(cur, nxt, ctrl, None, h)
                wp.synchronize_device(device)
            finally:
                if probe:
                    events = wp.timing_end()
            if probe:
                native = dict(Counter(e.name for e in events if e.filter == wp.TIMING_KERNEL))
                assert any("accumulate_self_contact_force_and_hessian" in name for name in native)
            cur, nxt = nxt, cur
    np.testing.assert_array_equal(states[0][0], x)
    np.testing.assert_array_equal(states[0][1], v)
    probe = m.state()
    previous = wp.empty(6, dtype=wp.vec3, device=device)
    forces = wp.zeros(6, dtype=wp.vec3, device=device)
    hess = wp.zeros(6, dtype=wp.mat33, device=device)
    det = s.trimesh_collision_detector
    rows = []
    work = 0.0
    slip = 0.0
    integral = 0.0
    for i in range(1, 33):
        now, speed = states[i]
        np.testing.assert_array_equal(now[:3], x[:3])
        probe.particle_q.assign(now)
        previous.assign(states[i - 1][0])
        det.refit(probe.particle_q)
        det.vertex_triangle_collision_detection(3 / 32)
        det.edge_edge_collision_detection(3 / 32)
        wp.synchronize_device(device)
        assert not np.any(det.resize_flags.numpy())
        vt = _measure("VT", _pairs(det, "vertex_colliding_triangles"), m, probe, range(6))
        ee = _measure("EE", _pairs(det, "edge_colliding_edges"), m, probe, range(6))
        active = [p for p in vt if p["distance_m"] < 1 / 16]
        assert (
            len(active) == 1
            and active[0]["material_a"] == [3]
            and not any(p["distance_m"] < 1 / 16 for p in ee)
        )
        forces.zero_()
        hess.zero_()
        for color in range(len(m.particle_color_groups)):
            wp.launch(
                accumulate_self_contact_force_and_hessian,
                dim=s.particle_self_contact_evaluation_kernel_launch_size,
                inputs=[
                    h,
                    color,
                    previous,
                    probe.particle_q,
                    m.particle_colors,
                    m.tri_indices,
                    m.edge_indices,
                    s.trimesh_collision_info,
                    s.particle_self_contact_margin,
                    m.soft_contact_ke,
                    m.soft_contact_kd,
                    m.soft_contact_mu,
                    s.friction_epsilon,
                    s._self_contact_edge_edge_parallel_epsilon,
                ],
                outputs=[forces, hess],
                device=device,
                max_blocks=device.sm_count,
            )
        wp.synchronize_device(device)
        f = forces.numpy().astype(float)
        dx = now[3].astype(float) - states[i - 1][0][3]
        ds = dx - n * (n @ dx)
        ft = f[3] - n * (f[3] @ n)
        assert np.isfinite(f).all() and f[3] @ n > 0 and ft @ ds <= 1e-7
        slip += float(np.linalg.norm(ds))
        work += float(ft @ ds)
        integral += float(ft @ t) * h
        rows.append(
            dict(
                tick=i,
                position_m=now.tolist(),
                velocity_m_s=speed.tolist(),
                material_weights=active[0]["weights_b"],
                gap_m=active[0]["distance_m"],
                frozen_nodal_force_n=f.tolist(),
                normal_n=float(f[3] @ n),
                frozen_traction_along_t_n=float(ft @ t),
                accepted_vertex_slip_m=slip,
                sampled_friction_work_j=work,
                integrated_sampled_tangent_impulse_ns=integral,
                actual_vertex_tangent_impulse_ns=float((speed[3] - v[3]) @ t),
            )
        )
    return (
        rows,
        native,
        dict(
            inverse_mass_kg_inv=m.particle_inv_mass.numpy().tolist(),
            normal_stiffness_n_m=m.soft_contact_ke,
            friction=m.soft_contact_mu,
            regularization_m_s=s.friction_epsilon,
            layer_m=s.particle_self_contact_margin,
            search_width_m=s.particle_self_contact_margin + s.particle_self_contact_gap,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--phymeo-receipt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-root", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("provide a fresh receipt")
    assert (
        version("newton") == "1.7.0"
        and version("warp-lang") == "1.17.0"
        and "phymeo" not in sys.modules
    )
    data = a.phymeo_receipt.read_bytes()
    source = json.loads(data)
    a.cache_root.mkdir(exist_ok=True, parents=True)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    wp.config.kernel_cache_dir = str(a.cache_root.resolve())
    wp.init()
    device = wp.get_device(a.device)
    assert device.is_cuda == (a.device == "cuda:0")
    with wp.ScopedDevice(device):
        rows, native, binding = solve(source, device)
    result = dict(
        binding=binding,
        cuda_kernels_on_actual_step_16=native,
        newton=version("newton"),
        warp=version("warp-lang"),
        device=str(device),
        device_name=device.name,
        dt_s=source["dt_s"],
        actual_solver_steps=32,
        source_sha256=sha256(data).hexdigest(),
        reference=source["samples"][-1],
        force_sample="Original native color force/Hessian kernel on consecutive accepted poses after all physical steps",
        samples=rows,
    )
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(
        f"{device}: 32 real steps, vertex slip {rows[-1]['accepted_vertex_slip_m']:.8f} m, sampled work {rows[-1]['sampled_friction_work_j']:.8f} J"
    )


if __name__ == "__main__":
    main()

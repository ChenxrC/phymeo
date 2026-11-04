"""Independent material geometry, virtual-work gradients and open-loop free-node folded contact."""

import json
from contextlib import ExitStack

import numpy as np
import pytest

import phymeo as ph

H = 1 / 16
LEFT = ph.CollisionKey(5, 9, 20)
RIGHT = ph.CollisionKey(5, 9, 30)


def fixture(sign=-1, *, rotated=False, dtype=np.float64, operator=None, disconnected=False):
    from phymeo.solvers.folded_material_seam import FoldedMaterialSeamStepper

    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]] if rotated else np.eye(3), dtype=dtype)
    x = np.array(
        [
            [-0.125, 0, (1 / 32) * np.sqrt(1 + (1 / 16) ** 2) + sign * 0.125 / 16],
            [0, -1, 0],
            [0, 1, 0],
            [-1, 0, sign / 16],
            [1, 0, sign / 16],
        ],
        dtype=dtype,
    )
    v = np.zeros_like(x)
    v[:, 0] = 0.125
    v[0, 0] += 4
    v[3:, 2] = sign * 0.5
    masses = np.array([1.0, 2.0, 2.0, 1.5, 1.5], dtype=dtype)
    order = np.array([3, 1, 4, 0, 2] if rotated else list(range(5)))
    rows = np.argsort(order)
    left = ph.CollisionShape(LEFT, "triangle", (7, 17, 31))
    right = ph.CollisionShape(RIGHT, "triangle", (70, 49, 90) if disconnected else (17, 49, 31))
    reg = ph.CollisionModel([right, left] if rotated else [left, right])
    model = ph.ParticleModel(1 / masses[order], np.zeros(3, dtype=dtype))
    op = FoldedMaterialSeamStepper(
        model,
        reg,
        point_row=int(rows[0]),
        point_material=(5, 61, 2),
        material_rows={7: int(rows[3]), 17: int(rows[1]), 31: int(rows[2]), 49: int(rows[4])},
        layer=ph.NormalLayer(1 / 16, 64.0, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=16.0,
        layer_operator=operator,
    )
    state = op.initial_state(ph.ParticleState((x @ R.T)[order], (v @ R.T)[order]))
    return op, state, rows, masses, R


def reference_frame(x, side):
    # Rows [point, edge A, edge B, left wing, right wing]. Independent Gram-Schmidt frames.
    u = x[2] - x[1]
    u = u / np.linalg.norm(u)
    center = (x[1] + x[2]) / 2
    tangent = x[3 + side] - center
    tangent -= u * (u @ tangent)
    tangent *= (-1 if side == 0 else 1) / np.linalg.norm(tangent)
    return np.stack([tangent, np.cross(tangent, u)]), u, center


def footprint(x, side):
    frame, u, center = reference_frame(x, side)
    other = reference_frame(x, 1 - side)[0]
    direction = frame[1] + other[1]
    direction /= np.linalg.norm(direction)
    p = x[0]
    n = frame[1]
    q = p - n @ (p - center) / (n @ direction) * direction
    tri = x[[1, 2, 3 + side]]
    edges = (tri[1:] - tri[0]).T
    bary = np.linalg.solve(edges.T @ edges, edges.T @ (q - tri[0]))
    return np.array([1 - bary.sum(), *bary]), q


def finite_jacobian(x, side, w):
    def coordinates(y):
        frame, _, _ = reference_frame(y, side)
        return frame @ (y[0] - w @ y[[1, 2, 3 + side]])

    J = np.empty((2, 5, 3))
    eps = 2.0**-17
    for row in range(5):
        for axis in range(3):
            plus = x.copy()
            minus = x.copy()
            plus[row, axis] += eps
            minus[row, axis] -= eps
            J[:, row, axis] = (coordinates(plus) - coordinates(minus)) / (2 * eps)
    return J


def ref_event(x0, x1):
    def scalar(t):
        x = x0 + t * (x1 - x0)
        f0, _, c = reference_frame(x, 0)
        f1, _, _ = reference_frame(x, 1)
        return float((x[0] - c) @ (f0[0] + f1[0]))

    a, b = 0.0, 1.0
    assert scalar(a) < 0 < scalar(b)
    for _ in range(65):
        m = (a + b) / 2
        if scalar(m) < 0:
            a = m
        else:
            b = m
    return (a + b) / 2


def reference_average(x0, x1, pieces):
    # Different integration rule and independently differenced 3D material work, no solver calls.
    nodes, weights = np.polynomial.legendre.leggauss(12)
    result = []
    for a, b, side in pieces:
        S = np.zeros((2, 5, 3))
        S1 = np.zeros_like(S)
        for u, wt in zip(nodes, weights, strict=True):
            t = (a + b + (b - a) * u) / 2
            x = x0 + t * (x1 - x0)
            bary, _ = footprint(x, side)
            val = finite_jacobian(x, side, bary)
            S += (b - a) / 2 * wt * val
            S1 += (b - a) / 2 * wt * t * val
        result.append((S, S1))
    return result


@pytest.mark.parametrize("sign,rotated", [(-1, False), (-1, True), (1, True)])
def test_free_loaded_vertex_crosses_changing_fold_with_independent_reactions(
    sign, rotated, tmp_path
):
    op, start, rows, masses, R = fixture(sign, rotated=rotated)
    load = ph.StepInputs.zeros(5)
    load.force[rows[0]] = R @ [0, 0, -2.0]
    out = op.step(start, load, H)
    assert out.report.accepted and len(out.legs) == 2 and out.normal_force_n == pytest.approx(2)
    a, b = out.legs
    x0 = start.particles.position[rows]
    x1 = out.state.particles.position[rows]
    v0 = start.particles.velocity[rows]
    v1 = out.state.particles.velocity[rows]
    tau = ref_event(x0, x1)
    assert out.event_bracket[0] <= tau <= out.event_bracket[1]
    assert a.t1 == pytest.approx(tau, abs=3e-9) and a.t1 == b.t0
    assert (
        abs(a.start.fold_angle_rad) > 0.06
        and abs(b.end.fold_angle_rad - a.start.fold_angle_rad) > 0.01
    )
    for side, section in enumerate((a, b)):
        assert section.face == (LEFT if side == 0 else RIGHT)
        assert (
            section.activation_gap_bounds_m[0] > 0 and section.activation_gap_bounds_m[1] < 1 / 16
        )
        for t, sample in ((section.t0, section.start), (section.t1, section.end)):
            x = x0 + t * (x1 - x0)
            frame, _, _ = reference_frame(x, side)
            w, proj = footprint(x, side)
            np.testing.assert_allclose(sample.normal, frame[1], atol=3e-10)
            np.testing.assert_allclose(sample.material_point_m, proj, atol=3e-10)
            assert sample.normal_gap_m == pytest.approx(frame[1] @ (x[0] - proj), abs=3e-10)
        assert np.linalg.norm(np.asarray(section.start.normal) - section.end.normal) > 1e-3
    np.testing.assert_allclose(a.end.material_point_m, b.start.material_point_m, atol=2e-10)
    np.testing.assert_allclose(a.end.normal_gap_m, b.start.normal_gap_m, atol=2e-10)
    assert a.end.director_gap_m > a.end.normal_gap_m
    features = [v.feature for v in a.end.spatial]
    assert features == (["edge", "edge"] if sign < 0 else ["face", "face"])
    if sign < 0:
        assert min(v.distance_m for v in a.end.spatial) > a.end.normal_gap_m
    else:
        assert (
            np.linalg.norm(np.asarray(a.end.spatial[0].point_m) - a.end.spatial[1].point_m) > 1e-4
        )
    ref = reference_average(x0, x1, [(0, tau, 0), (tau, 1, 1)])
    traction = np.array([out.tangent_force_scalar_n, out.normal_force_n])
    expected = np.zeros((5, 3))
    for section, (J, J1) in zip(out.legs, ref, strict=True):
        impulse = H * np.tensordot(traction, J, axes=1)
        expected += impulse
        np.testing.assert_allclose(section.nodal_impulses_ns[rows], impulse, atol=3e-8, rtol=0)
        np.testing.assert_allclose(
            section.nodal_work_j[rows], np.sum(impulse * (x1 - x0), axis=1) / H, atol=3e-8, rtol=0
        )
        np.testing.assert_allclose(section.integrated_angular_impulse_nms, 0, atol=2e-10)
    np.testing.assert_allclose(out.nodal_contact_impulse_ns[rows], expected, atol=3e-8)
    dv = v1 - v0
    np.testing.assert_allclose(masses[:, None] * dv, expected + H * load.force[rows], atol=4e-8)
    np.testing.assert_allclose(
        (masses[:, None] * dv).sum(axis=0), H * load.force.sum(axis=0), atol=3e-10
    )
    angular = np.cross(x1, masses[:, None] * v1).sum(axis=0) - np.cross(
        x0, masses[:, None] * v0
    ).sum(axis=0)
    np.testing.assert_allclose(
        angular - H * np.cross(x0, load.force[rows]).sum(axis=0),
        out.discrete_angular_impulse_nms,
        atol=3e-10,
    )
    dx = x1 - x0
    spring = out.state.history
    assert spring.plastic_slip_m > 0 and spring.stored_energy_j > 0
    assert (
        out.tangent_work_j + spring.stored_energy_j + out.plastic_loss_j + out.implicit_loss_j
        == pytest.approx(0, abs=3e-9)
    )
    assert out.normal_work_j == pytest.approx(
        2 * (b.end.normal_gap_m - a.start.normal_gap_m), abs=3e-9
    )
    deltaK = 0.5 * np.sum(masses[:, None] * dv * (v0 + v1))
    assert deltaK + 0.5 * np.sum(masses[:, None] * dv**2) == pytest.approx(
        float(np.sum(load.force[rows] * dx)) + out.tangent_work_j + out.normal_work_j, abs=4e-8
    )
    save(tmp_path / "open-loop-fold.json", start, out, load, rows, masses)


def prescribed(old, target, mass, pieces):
    x0 = old.particles.position
    mat = reference_average(x0, target, pieces)
    j = sum(k for k, _ in mat)
    side = pieces[0][2]
    frame, _, c = reference_frame(x0, side)
    pressure = 64 * (1 / 16 - frame[1] @ (x0[0] - c))
    e0 = old.history.elastic_m if old.history else 0.0
    ds = float(np.sum(j[0] * (target - x0)))
    f = -16 * (e0 + ds)
    assert abs(f) < 0.5 * pressure
    fc = np.tensordot([f, pressure], j, axes=1)
    external = mass[:, None] * ((target - x0) / H - old.particles.velocity) / H - fc
    return ph.StepInputs(external), ds, pressure


def test_preloaded_crossing_transports_around_the_edge_then_rigid_motion_has_zero_work(tmp_path):
    op, base, rows, mass, R = fixture()
    p = base.particles
    p.position[0] = [-1 / 64, 0, (1 / 32) * np.sqrt(1 + (1 / 16) ** 2) - (1 / 16) * (1 / 64)]
    p.velocity.fill(0)
    start = op.initial_state(p)
    first_target = p.position.copy()
    tangent = reference_frame(p.position, 0)[0][0]
    first_target[0] += (1 / 256) * tangent
    drive, _, _ = prescribed(start, first_target, mass, [(0, 1, 0)])
    pre = op.step(start, drive, H)
    assert pre.state.history.regime == "stick" and pre.state.history.elastic_m == pytest.approx(
        1 / 256, abs=5e-10
    )
    expected = pre.state.particles.position.copy()
    expected += np.array([1 / 128, 0, 1 / 64])
    expected[3:, 2] -= 1 / 32
    expected[0, 0] = 1 / 128 + 1 / 128
    tau = ref_event(pre.state.particles.position, expected)
    command, ds, _ = prescribed(pre.state, expected, mass, [(0, tau, 0), (tau, 1, 1)])
    cross = op.step(pre.state, command, H)
    assert cross.legs[0].t1 == pytest.approx(tau, abs=3e-9)
    np.testing.assert_allclose(cross.state.particles.position, expected, atol=7e-10, rtol=0)
    assert (
        cross.state.history.key == pre.state.history.key and cross.state.history.plastic_slip_m == 0
    )
    assert cross.state.history.elastic_m == pytest.approx(
        pre.state.history.elastic_m + ds, abs=8e-10
    )
    lft, rgt = cross.legs[0].end, cross.legs[1].start
    t = np.array(lft.tangent)
    axis = cross.state.particles.position[2] - cross.state.particles.position[1]
    axis /= np.linalg.norm(axis)
    angle = lft.fold_angle_rad
    np.testing.assert_allclose(
        np.cos(angle) * t + np.sin(angle) * np.cross(axis, t), rgt.tangent, atol=1e-10
    )
    assert abs(lft.normal_gap_m - rgt.normal_gap_m) < 2e-10
    # A physical third step applies a 6-degree rigid turn about the shared-edge direction.
    theta = np.deg2rad(6.0)
    Q = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    old = cross.state.particles.position
    new = old @ Q.T + [1 / 32, 1 / 64, -1 / 128]
    third, dturned, _ = prescribed(cross.state, new, mass, [(0, 1, 1)])
    assert abs(dturned) < 5e-10
    rigid = op.step(cross.state, third, H)
    assert len(rigid.legs) == 1 and rigid.state.history.active_steps == 3
    np.testing.assert_allclose(rigid.state.particles.position, new, atol=7e-10, rtol=0)
    assert rigid.state.history.plastic_slip_m == 0
    assert rigid.state.history.stored_energy_j == pytest.approx(
        cross.state.history.stored_energy_j, abs=5e-10
    )
    assert abs(rigid.tangent_work_j) < 5e-10 and abs(rigid.normal_work_j) < 5e-10
    assert (
        abs(rigid.legs[0].nodal_work_j.sum()) < 8e-10 and abs(rigid.legs[0].nodal_work_j[0]) > 1e-4
    )
    np.testing.assert_allclose(
        rigid.state.history.world_elastic_m, Q @ cross.state.history.world_elastic_m, atol=1e-9
    )
    w, q = footprint(old, 1)
    naive = (new[0] - old[0]) - w @ (new[[1, 2, 4]] - old[[1, 2, 4]])
    assert abs(naive @ cross.legs[-1].end.tangent) > 1e-3
    save(tmp_path / "preloaded-fold-crossing.json", pre.state, cross, command, rows, mass)
    save(tmp_path / "objective-rigid-turn.json", cross.state, rigid, third, rows, mass)


def test_spatial_counterexample_and_fold_rejections_are_transactional():
    from phymeo.core.result import StepReport, StepResult

    with pytest.raises(ValueError, match="disconnected"):
        fixture(disconnected=True)
    op, start, rows, m, R = fixture()
    ext = ph.StepInputs.zeros(5)
    ext.force[0] = [0, 0, -2]
    flip = ext.copy()
    flip.force[3:, 2] = 60.0
    with pytest.raises(ph.SolverFailure, match="sign|flattens|angle"):
        op.step(start, flip, H)
    tilt = ext.copy()
    tilt.force[1, 0] += 2.0
    with pytest.raises(ph.SolverFailure, match="axis|transverse"):
        op.step(start, tilt, H)
    p = start.particles
    p.position[4] = [0, 0, 0]
    with pytest.raises(ph.SolverFailure, match="degenerate"):
        op.initial_state(p)
    with pytest.raises(ph.SolverFailure, match="converge"):
        op.step(start, ext, H, max_iterations=1)

    class Reject:
        def step(self, model, state, inputs, dt):
            return StepResult(
                state.copy(), StepReport(False, reason="deliberately rejected particle candidate")
            )

    failed = op.step(start, ext, H, particle_solver=Reject())
    assert failed.state is start and failed.nodal_contact_impulse_ns is None and failed.legs == ()
    a, b = op.step(start, ext, H), op.step(failed.state, ext, H)
    np.testing.assert_array_equal(a.state.particles.position, b.state.particles.position)
    assert a.state.history == b.state.history and a.event_bracket == b.event_bracket
    assert start.history is None and start.tick == 0


def save(path, before, step, external, rows, m):
    from dataclasses import asdict

    path.write_text(
        json.dumps(
            dict(
                dt_s=H,
                backend=step.report.backend,
                normal_backend=step.normal_backend,
                slot_order=rows.tolist(),
                masses_kg=m.tolist(),
                input_force_n=external.force.tolist(),
                start_position=before.particles.position.tolist(),
                start_velocity=before.particles.velocity.tolist(),
                end_position=step.state.particles.position.tolist(),
                end_velocity=step.state.particles.velocity.tolist(),
                history=asdict(step.state.history),
                pressure_n=step.normal_force_n,
                shear_n=step.tangent_force_scalar_n,
                work_t_j=step.tangent_work_j,
                work_n_j=step.normal_work_j,
                normal_energy_j=step.normal_energy_j,
                plastic_loss_j=step.plastic_loss_j,
                implicit_loss_j=step.implicit_loss_j,
                iterations=step.site_iterations,
                closure=step.jacobian_residual,
                quadrature_difference=step.quadrature_difference,
                bracket=step.event_bracket,
                actual_impulse_ns=step.nodal_contact_impulse_ns.tolist(),
                discrete_angular_nms=step.discrete_angular_impulse_nms,
                legs=[
                    dict(
                        a=leg.t0,
                        b=leg.t1,
                        start=asdict(leg.start),
                        end=asdict(leg.end),
                        active_gap_m=leg.activation_gap_bounds_m,
                        nodal_work_j=leg.nodal_work_j.tolist(),
                        node_impulse_ns=leg.nodal_impulses_ns.tolist(),
                        integrated_torque_nms=leg.integrated_angular_impulse_nms,
                    )
                    for leg in step.legs
                ],
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


@pytest.mark.tyx
def test_folded_normal_and_all_five_particle_rows_run_on_the_selected_engine(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solver = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        op, first, rows, m, R = fixture(rotated=True, dtype=np.float32, operator=layer)
        load = ph.StepInputs.zeros(5, dtype=np.float32)
        load.force[rows[0]] = R @ [0, 0, -2.0]
        out = op.step(first, load, H, particle_solver=solver)
        assert out.report.backend == "tyx:" + tyx_mode and out.normal_backend == tyx_mode
        assert layer.stats()["completed_launches"] == 1
        if tyx_mode == "cuda":
            assert layer.stats()["downloads"] == 1
        x0 = first.particles.position[rows].astype(float)
        x1 = out.state.particles.position[rows].astype(float)
        tau = ref_event(x0, x1)
        assert out.event_bracket[0] <= tau <= out.event_bracket[1]
        ref = reference_average(x0, x1, [(0, tau, 0), (tau, 1, 1)])
        load_local = np.array([out.tangent_force_scalar_n, out.normal_force_n])
        J = sum(v[0] for v in ref)
        np.testing.assert_allclose(
            out.nodal_contact_impulse_ns[rows],
            H * np.tensordot(load_local, J, axes=1),
            atol=2e-6,
            rtol=0,
        )
        assert abs(out.legs[0].start.fold_angle_rad - out.legs[-1].end.fold_angle_rad) > 0.01
        save(tmp_path / "folded-free-particles-device.json", first, out, load, rows, m)

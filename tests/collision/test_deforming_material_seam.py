"""Independent free-node trajectories, moving edge events, distributed impulses and energy."""

import json
from contextlib import ExitStack
from decimal import Decimal, localcontext

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial

H = 1 / 8
LEFT = ph.CollisionKey(8, 4, 13)
RIGHT = ph.CollisionKey(8, 4, 27)
NODES = (7, 17, 31, 49)


def fixture(
    *, disconnected=False, structural=False, rotated=False, dtype=np.float64, operator=None
):
    from phymeo.solvers.deforming_material_seam import DeformingMaterialSeamStepper

    ids = list(NODES)
    xy = [[-1, 0, 0], [0, -1, 0], [0, 1, 0], [1, 0, 0]]
    right = (17, 49, 31)
    masses = [1, 3, 3, 1]
    if disconnected:
        ids.extend((71, 81))
        xy.extend((xy[1], xy[2]))
        masses.extend([3, 3])
        right = (71, 49, 81)
    tri = [
        ph.CollisionShape(LEFT, "triangle", (7, 17, 31)),
        ph.CollisionShape(RIGHT, "triangle", right),
    ]
    x = np.array([*xy, [-0.5, 0, 1 / 32]], dtype=dtype)
    m = np.array([*masses, 1.0], dtype=dtype)
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]] if rotated else np.eye(3), dtype=dtype)
    order = list(reversed(range(len(x)))) if rotated else list(range(len(x)))
    rows = np.argsort(order)
    bindings = {node: int(rows[j]) for j, node in enumerate(ids)}
    springs = None
    if structural:
        springs = ClothMaterial(
            ph.DistanceConstraints(
                np.array([[rows[1], rows[2]]]), np.array([1.75]), np.array([0.25])
            )
        )
    model = ph.ParticleModel(1 / m[order], np.zeros(3, dtype=dtype))
    op = DeformingMaterialSeamStepper(
        model,
        ph.CollisionModel(tri[::-1] if rotated else tri),
        material_rows=bindings,
        point_row=int(rows[-1]),
        point_material=(8, 61, 2),
        normal=R[:, 2],
        layer=ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=16.0,
        material=springs,
        layer_operator=operator,
    )
    xyz = (x @ R.T)[order]
    velocity = np.full_like(x, 0.0)
    velocity[:, 0] = 0.25
    velocity[1, 1] = -0.25
    velocity[2, 1] = 0.25
    velocity[-1, 0] += 8
    state = op.initial_state(ph.ParticleState(xyz, (velocity @ R.T)[order]))
    return op, state, rows, m, R


def open_loop(case, *, solver=None, atol=4e-10):
    op, state, rows, m, R = case
    # All four initially moving face nodes have zero imposed force.
    # The only drive is a single predeclared constant on the free point.
    load = ph.StepInputs.zeros(len(m), dtype=state.particles.position.dtype)
    load.force[rows[-1]] = R @ [9 / 16, 0, -9 / 4]
    answer = op.step(state, load, H, particle_solver=solver)
    from phymeo.solvers.deforming_material_seam import DeformingSeamState

    assert isinstance(answer.state, DeformingSeamState)
    assert answer.report.accepted and answer.normal_force_n == pytest.approx(2, abs=atol)
    assert answer.state.history.regime == "slide"
    a, b = answer.segments
    assert (a.face, b.face) == (LEFT, RIGHT)
    assert a.t1 == pytest.approx(0.5, abs=atol) and b.t0 == a.t1
    assert a.exit_point_on_patch_m == pytest.approx(b.entry_point_on_patch_m, abs=atol)
    expected = [(a, {7: 1 / 8, 17: 3 / 16, 31: 3 / 16}), (b, {17: 3 / 16, 31: 3 / 16, 49: 1 / 8})]
    traction = R @ [-0.5, 0, 2.0]
    for seg, integral in expected:
        assert dict(seg.integrated_weights) == pytest.approx(integral, abs=atol)
        for node, j in seg.node_impulses_ns:
            np.testing.assert_allclose(j, -H * integral[node] * traction, atol=atol, rtol=0)
    assert answer.site_iterations > 1 and answer.weight_residual_l1 <= answer.weight_tolerance
    assert answer.event_bracket[0] - atol <= 0.5 <= answer.event_bracket[1] + atol
    assert answer.quadrature_error_l1 <= answer.weight_tolerance
    # Each face mass receives the same x/z acceleration through its own unequal impulse.
    # The shared edge, 0.25 m longer than its rest length, pulls its endpoints toward one another.
    ip = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    contact = np.vstack([-ip[:, None] * traction, traction])
    spring = np.zeros_like(contact)
    spring[1] = R @ [0, 1.0, 0]
    spring[2] = -spring[1]
    old, end = state.particles, answer.state.particles
    dx = end.position[rows].astype(float) - old.position[rows]
    dv = end.velocity[rows].astype(float) - old.velocity[rows]
    np.testing.assert_allclose(
        answer.nodal_contact_impulse_ns[rows], H * contact, atol=atol, rtol=0
    )
    np.testing.assert_allclose(answer.structural_force_n[rows], spring, atol=atol, rtol=0)
    np.testing.assert_allclose(
        m[:, None] * dv, H * (contact + spring + load.force[rows]), atol=atol, rtol=0
    )
    np.testing.assert_allclose(
        (m[:, None] * dv).sum(axis=0), H * load.force.sum(axis=0), atol=atol, rtol=0
    )
    np.testing.assert_allclose(dx, H * end.velocity[rows], atol=atol, rtol=0)
    end_material = end.position[rows[:4]] @ R
    assert np.linalg.norm(end_material[2] - end_material[1]) == pytest.approx(2 + 5 / 96, abs=atol)
    assert answer.structural_energy_j == pytest.approx((1 / 8, 2 * (29 / 96) ** 2), abs=atol)
    areas = []
    for i, j, k in ((0, 1, 2), (1, 3, 2)):
        e = end_material[j] - end_material[i]
        f = end_material[k] - end_material[i]
        areas.append(np.linalg.norm(np.cross(e, f)) / 2)
    np.testing.assert_allclose(areas, [1 + 5 / 192] * 2, atol=atol, rtol=0)
    # Same application points and constant traction are used for both sides' work.
    node_work = sum(sum(w for _, w in s.node_work_j) for s in answer.segments)
    tip_work = sum(s.vertex_work_j for s in answer.segments)
    assert node_work == pytest.approx(np.sum(contact[:4] * dx[:4]), abs=atol)
    assert tip_work == pytest.approx(contact[-1] @ dx[-1], abs=atol)
    assert answer.state.history.plastic_slip_m == pytest.approx(31 / 32, abs=atol)
    assert answer.state.history.tangent_work_j == pytest.approx(-0.5, abs=atol)
    assert answer.state.history.stored_energy_j == pytest.approx(1 / 128, abs=atol)
    assert answer.plastic_loss_j == pytest.approx(31 / 64, abs=atol)
    delta_k = 0.5 * np.sum(m[:, None] * (old.velocity[rows] + end.velocity[rows]) * dv)
    integrator_loss = 0.5 * np.sum(m[:, None] * dv**2)
    assert delta_k + integrator_loss == pytest.approx(
        np.sum((spring + load.force[rows]) * dx) + tip_work + node_work, abs=atol
    )
    return answer, load


@pytest.mark.parametrize("rotated", [False, True])
def test_open_loop_pressure_crosses_two_straining_faces_with_free_nodal_recoil(rotated, tmp_path):
    case = fixture(structural=True, rotated=rotated)
    answer, external = open_loop(case)
    save(tmp_path / "independent-nodal-crossing.json", answer, case[1], external, case[2], case[3])


def save(path, answer, old, external, rows, mass):
    path.write_text(
        json.dumps(
            dict(
                backend=answer.report.backend,
                normal_backend=answer.normal_backend,
                iterations=answer.site_iterations,
                iteration_residuals_l1=answer.iteration_residuals_l1,
                residual_l1=answer.weight_residual_l1,
                tolerance=answer.weight_tolerance,
                quadrature_bound=answer.quadrature_error_l1,
                tau_bracket=answer.event_bracket,
                dt_s=H,
                material_ids=list(NODES),
                point_id=(8, 61, 2),
                particle_rows_for_nodes_then_point=rows.tolist(),
                masses_kg_for_nodes_then_point=mass.tolist(),
                external_force_n=external.force.tolist(),
                from_velocity=old.particles.velocity.tolist(),
                force_on_vertex_n=answer.force_on_vertex_n,
                normal_force_n=answer.normal_force_n,
                structural_energy_j=answer.structural_energy_j,
                plastic_loss_j=answer.plastic_loss_j,
                implicit_loss_j=answer.implicit_loss_j,
                from_position=old.particles.position.tolist(),
                position=answer.state.particles.position.tolist(),
                velocity=answer.state.particles.velocity.tolist(),
                impulses=answer.nodal_contact_impulse_ns.tolist(),
                structural_force=answer.structural_force_n.tolist(),
                plastic_m=answer.state.history.plastic_slip_m,
                work_j=answer.state.history.tangent_work_j,
                stored_j=answer.state.history.stored_energy_j,
                segments=[
                    dict(
                        face=s.face.shape,
                        t0=s.t0,
                        t1=s.t1,
                        weights=s.integrated_weights,
                        impulses=s.node_impulses_ns,
                        vertex_work_j=s.vertex_work_j,
                        node_work_j=s.node_work_j,
                        patch_end=s.exit_point_on_patch_m,
                    )
                    for s in answer.segments
                ],
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def inverse_input(state, target, mass, weights, traction):
    x, v = state.particles.position, state.particles.velocity
    contact = np.vstack((-np.asarray(weights)[:, None] * traction, traction))
    return ph.StepInputs(mass[:, None] * ((target - x) / H - v) / H - contact)


def analytic_integral(c, d, k, a, b):
    # 60-digit primitive of (c+d*s)/(1+k*s). Independent of the solver's Simpson rule.
    with localcontext() as context:
        context.prec = 60

        def rat(value):
            num, den = value
            return Decimal(num) / Decimal(den)

        c, d, k, a, b = map(rat, (c, d, k, a, b))
        return float((d / k) * (b - a) + (c - d / k) / k * ((1 + k * b) / (1 + k * a)).ln())


def test_common_edge_moves_and_changes_length_while_a_preloaded_spring_stays_elastic(tmp_path):
    op, template, rows, m, R = fixture()
    x = template.particles.position
    x[-1, 0] = -3 / 256
    start = op.initial_state(ph.ParticleState(x, np.zeros_like(x)))
    xp = x.copy()
    xp[-1, 0] = -1 / 128
    wl = 5 / 512
    preweight = np.array([wl, (1 - wl) / 2, (1 - wl) / 2, 0.0])
    prepare = inverse_input(start, xp, m, preweight, np.array([-1 / 16, 0, 2.0]))
    first = op.step(start, prepare, H)
    assert first.state.history.regime == "stick" and first.state.history.active_steps == 1
    assert first.state.history.stored_energy_j == pytest.approx(8 * (1 / 256) ** 2, abs=2e-13)
    np.testing.assert_allclose(first.state.particles.position, xp, atol=3e-12, rtol=0)
    target = xp.copy()
    target[:, 0] += 1 / 16
    target[0, 0] -= 1 / 8
    target[3, 0] -= 1 / 16
    target[1, 1] -= 1 / 8
    target[2, 1] += 1 / 8
    target[-1, 0] += 3 / 128
    L = analytic_integral((1, 128), (-3, 128), (1, 8), (0, 1), (1, 3))
    Rw = analytic_integral((-1, 128), (3, 128), (-1, 16), (1, 3), (1, 1))
    wa = np.array([L, (1 - L - Rw) / 2, (1 - L - Rw) / 2, Rw])
    du = 3 / 128 + L / 8 + Rw / 16
    shear = -16 * (1 / 256 + du)
    traction = np.array([shear, 0, 2.0])
    command = inverse_input(first.state, target, m, wa, traction)
    second = op.step(first.state, command, H)
    assert len(second.segments) == 2 and second.state.face == RIGHT
    assert second.event_bracket[0] <= 1 / 3 <= second.event_bracket[1]
    assert second.segments[0].t1 == pytest.approx(1 / 3, abs=2e-11)
    assert (
        abs((1 / 128) / (1 / 16 + 3 / 128) - second.segments[0].t1) > 0.2
    )  # entrance edge alone gives the wrong event
    np.testing.assert_allclose(second.state.particles.position, target, atol=2e-11, rtol=0)
    a, b = second.segments
    assert dict(a.integrated_weights) == pytest.approx(
        {7: L, 17: (1 / 3 - L) / 2, 31: (1 / 3 - L) / 2}, abs=2e-11
    )
    assert dict(b.integrated_weights) == pytest.approx(
        {49: Rw, 17: (2 / 3 - Rw) / 2, 31: (2 / 3 - Rw) / 2}, abs=2e-11
    )
    np.testing.assert_allclose(a.exit_point_on_patch_m, [1 / 48, 0, 0], atol=2e-11, rtol=0)
    assert a.exit_point_on_patch_m == pytest.approx(b.entry_point_on_patch_m, abs=2e-11)
    assert first.state.history.key == second.state.history.key
    assert second.state.history.regime == "stick" and second.state.history.plastic_slip_m == 0
    np.testing.assert_allclose(
        second.state.history.elastic_displacement_m, [1 / 256 + du, 0, 0], atol=2e-11, rtol=0
    )
    assert second.state.history.stored_energy_j == pytest.approx(8 * (1 / 256 + du) ** 2, abs=2e-12)
    work = second.state.history.tangent_work_j - first.state.history.tangent_work_j
    assert work == pytest.approx(shear * du, abs=2e-12)
    assert (
        work
        + second.state.history.stored_energy_j
        - first.state.history.stored_energy_j
        + second.implicit_loss_j
        == pytest.approx(0, abs=2e-12)
    )
    assert sum(x.tangent_work_j for x in second.segments) == pytest.approx(work, abs=2e-12)
    dx = target - xp
    # Two distinct nonzero area strains, both with a common shared edge.
    area0 = np.linalg.norm(np.cross(target[1] - target[0], target[2] - target[0])) / 2
    area1 = np.linalg.norm(np.cross(target[3] - target[1], target[2] - target[1])) / 2
    assert area0 == pytest.approx((1 + 1 / 8) ** 2)
    assert area1 == pytest.approx((1 - 1 / 16) * (1 + 1 / 8))
    total_contact_work = traction @ dx[-1] - np.sum((wa[:, None] * traction) * dx[:4])
    assert sum(
        x.vertex_work_j + sum(w for _, w in x.node_work_j) for x in second.segments
    ) == pytest.approx(total_contact_work, abs=2e-12)
    save(tmp_path / "preloaded-changing-address.json", second, first.state, command, rows, m)


def test_capacity_tilt_disconnected_and_particle_rejection_publish_no_partial_state():
    from phymeo.core.result import StepReport, StepResult

    op, start, rows, m, R = fixture(structural=True)
    external = ph.StepInputs.zeros(5)
    external.force[-1] = [9 / 16, 0, -9 / 4]
    with pytest.raises(ph.SolverFailure, match="converge"):
        op.step(start, external, H, max_iterations=1)
    tilted = external.copy()
    tilted.force[0, 2] = 1.0
    with pytest.raises(ph.SolverFailure, match="tilt"):
        op.step(start, tilted, H)

    class Reject:
        def step(self, model, state, inputs, dt):
            return StepResult(
                state.copy(), StepReport(False, reason="deliberate physical candidate rejection")
            )

    refused = op.step(start, external, H, particle_solver=Reject())
    assert refused.state is start and not refused.report.accepted and refused.segments == ()
    assert refused.nodal_contact_impulse_ns is None
    first = op.step(start, external, H)
    replay = op.step(refused.state, external, H)
    assert first.state.history == replay.state.history and first.segments == replay.segments
    np.testing.assert_array_equal(first.state.particles.position, replay.state.particles.position)
    assert not start.history and start.tick == 0
    dis, state, _, mass, _ = fixture(disconnected=True)
    ext = ph.StepInputs.zeros(len(mass))
    ext.force[-1] = [9 / 16, 0, -9 / 4]
    with pytest.raises(ph.SolverFailure, match="disconnected"):
        dis.step(state, ext, H)
    assert not state.history and state.tick == 0


@pytest.mark.tyx
def test_real_normal_and_particle_advance_move_all_independent_nodes(tyx_mode, tmp_path):
    with ExitStack() as stack:
        law = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solver = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        case = fixture(structural=True, rotated=True, dtype=np.float32, operator=law)
        answer, external = open_loop(case, solver=solver, atol=3e-6)
        assert answer.report.backend == "tyx:" + tyx_mode and answer.normal_backend == tyx_mode
        assert law.stats()["completed_launches"] == 1
        if tyx_mode == "cuda":
            assert law.stats()["downloads"] == 1
        save(tmp_path / "deforming-seam-device.json", answer, case[1], external, case[2], case[3])

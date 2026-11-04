"""Finite windows of material sliding, branchable history, release and real particle motion."""

import numpy as np
import pytest

import phymeo as ph


def fixture(
    *,
    edge=False,
    rotated=False,
    dtype=np.float64,
    movable_lower=False,
    layer_operator=None,
    friction=0.25,
):
    from phymeo.solvers.deformable_sliding import MaterialSlidingStepper

    first = [[-2.0, -1.0, 0.0], [0.0 if edge else 2.0, -1.0, 0.0], [0.0, 2.0, 0.0]]
    x0 = -0.25
    upper = [[x0, 0.0, 1 / 32], [x0 + 0.125, 0.0625, 1.0], [x0 - 0.0625, 0.1875, 0.75]]
    rotation = (
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=dtype)
        if rotated
        else np.eye(3, dtype=dtype)
    )
    xyz = np.asarray(first + upper, dtype=dtype) @ rotation.T
    velocity = np.zeros((6, 3), dtype=dtype)
    velocity[3:] = rotation[:, 0]
    order = np.array([5, 2, 4, 1, 3, 0] if rotated else list(range(6)))
    owner = np.argsort(order)
    shapes = [
        ph.CollisionShape(ph.CollisionKey(7, b, 0), "triangle", ids)
        for b, ids in ((13, (17, 31, 49)), (3, (2, 18, 42)))
    ]
    model = ph.CollisionModel(shapes[::-1] if rotated else shapes)
    bindings = {
        shape.key: ph.ParticleAttachment(tuple(int(owner[i]) for i in range(3 * j, 3 * j + 3)))
        for j, shape in enumerate(shapes)
    }
    inverse_mass = np.array([float(movable_lower)] * 3 + [1.0] * 3, dtype=dtype)
    mass = ph.ParticleModel(inverse_mass[order], np.zeros(3, dtype=dtype))
    law = ph.NormalLayer(1 / 16, 64.0, friction=friction, slip_speed_m_s=1 / 32)
    op = MaterialSlidingStepper(
        mass, model, bindings, law, search_distance_m=3 / 32, layer_operator=layer_operator
    )
    state = op.initial_state(ph.ParticleState(xyz[order], velocity[order]))
    return model, op, state, owner, rotation


def force(state, row, normal, amount):
    out = ph.StepInputs.zeros(state.particles.particle_count, dtype=state.particles.position.dtype)
    out.force[row] = amount * normal
    return out


@pytest.mark.parametrize("h,rotated", [(1 / 32, False), (1 / 64, True)])
def test_constant_preload_slides_with_material_speed_work_and_impulse(h, rotated):
    model, op, state, rows, R = fixture(rotated=rotated)
    top = rows[3]
    tangent, normal = R[:, 0], R[:, 2]
    initial = state
    count = round(0.5 / h)
    with ph.CollisionWorkspace(model) as work:
        for i in range(1, count + 1):
            result = op.step(state, force(state, top, normal, -2.0), h, work)
            assert result.report.accepted and not result.ended
            (row,) = result.contacts
            assert row.phase == ("entered" if i == 1 else "continued")
            assert row.feature.kind == "VT" and row.normal_force_n == pytest.approx(2.0)
            np.testing.assert_allclose(
                result.nodal_tangent_force_n[top], -0.5 * tangent, atol=2e-12
            )
            np.testing.assert_allclose(result.nodal_tangent_force_n.sum(axis=0), 0, atol=2e-13)
            state = result.state
            v = 1 - 0.5 * i * h
            distance = i * h - 0.25 * h * h * i * (i + 1)
            assert state.particles.velocity[top] @ tangent == pytest.approx(v, abs=2e-12)
            assert state.particles.position[top] @ tangent == pytest.approx(
                -0.25 + distance, abs=2e-12
            )
            assert row.history.active_steps == i and row.history.slip_distance_m == pytest.approx(
                distance, abs=2e-12
            )
            assert row.history.tangent_work_j == pytest.approx(-0.5 * distance, abs=2e-12)
            assert np.linalg.norm(row.history.tangent_impulse_on_second_ns) == pytest.approx(
                0.5 * i * h
            )
            assert row.history.normal_impulse_ns == pytest.approx(2 * i * h)
            assert np.dot(
                row.tangent_force_on_second_n, row.tangent_displacement_m
            ) == pytest.approx(row.tangent_work_j)
        assert initial.tick == 0 and initial.active == ()
        assert state.tick == count


def test_unloading_and_return_start_a_distinct_episode_and_old_state_replays():
    model, op, state, rows, R = fixture()
    top = rows[3]
    h = 1 / 32
    n = R[:, 2]

    def advance(s, f):
        return op.step(s, force(s, top, n, f), h, workspace)

    with ph.CollisionWorkspace(model) as workspace:
        first = advance(state, -2)
        assert first.contacts[0].history.start_tick == 0
        lifted = advance(first.state, 30)
        assert lifted.state.particles.position[top, 2] == pytest.approx(1 / 16)
        clear = advance(lifted.state, 0)
        assert not clear.contacts and len(clear.ended) == 1 and clear.state.active == ()
        assert clear.ended[0] == lifted.contacts[0].history
        assert not advance(clear.state, -64).contacts
        returning = advance(advance(clear.state, -64).state, 0)
        renewed = advance(returning.state, 30)
        assert renewed.contacts[0].phase == "entered"
        assert renewed.contacts[0].history.start_tick == returning.state.tick
        assert renewed.contacts[0].history.active_steps == 1
        replayed = advance(first.state, 30)
        assert replayed.contacts == lifted.contacts
        np.testing.assert_array_equal(
            replayed.state.particles.position, lifted.state.particles.position
        )


def test_sliding_across_a_finite_edge_releases_its_material_contacts(tmp_path):
    import json
    from dataclasses import asdict

    model, op, state, rows, R = fixture(edge=True)
    top = rows[3]
    h = 1 / 128
    ended = []
    samples = []
    with ph.CollisionWorkspace(model) as work:
        for tick in range(80):
            result = op.step(state, force(state, top, R[:, 2], -2), h, work)
            assert result.report.accepted
            if tick == 0:
                (site,) = result.contacts
                key = site.history.key
            samples.append(
                dict(
                    tick=tick + 1,
                    actual_position_m=result.state.particles.position[top].tolist(),
                    actual_velocity_m_s=result.state.particles.velocity[top].tolist(),
                    contacts=[
                        dict(
                            kind=c.feature.kind,
                            phase=c.phase,
                            material=asdict(c.material),
                            normal_n=c.normal_force_n,
                            tangent_on_second_n=c.tangent_force_on_second_n,
                            du_m=c.tangent_displacement_m,
                            work_j=c.tangent_work_j,
                            history=asdict(c.history),
                        )
                        for c in result.contacts
                    ],
                    ended=[asdict(e) for e in result.ended],
                )
            )
            ended.extend(result.ended)
            state = result.state
        assert state.particles.position[top, 0] > 0.125
        assert state.active == ()
        old = [e for e in ended if e.key == key]
        assert old and old[0].slip_distance_m > 0.125 and old[0].tangent_work_j < 0
        (tmp_path / "material-sliding-edge-release.json").write_text(
            json.dumps(dict(h_s=h, frames=samples), allow_nan=False), encoding="utf8"
        )


def test_rejected_candidate_retains_the_original_history():
    model, op, start, rows, R = fixture()

    class Reject:
        def step(self, model, state, inputs, dt):
            from phymeo.core.result import StepReport, StepResult

            return StepResult(state.copy(), StepReport(False, reason="test candidate rejection"))

    with ph.CollisionWorkspace(model) as work:
        applied = force(start, rows[3], R[:, 2], -2)
        first = op.step(start, applied, 1 / 64, work)
        rejected = op.step(first.state, applied, 1 / 64, work, particle_solver=Reject())
        assert not rejected.report.accepted and rejected.state is first.state
        assert not rejected.contacts and not rejected.ended
        second = op.step(rejected.state, applied, 1 / 64, work)
        assert second.contacts[0].history.active_steps == 2


def test_two_dynamic_sides_receive_actual_opposite_impulses_and_material_work():
    model, op, state, rows, R = fixture(movable_lower=True)
    old = state.particles
    h = 1 / 64
    with ph.CollisionWorkspace(model) as workspace:
        result = op.step(state, ph.StepInputs.zeros(6), h, workspace)
    assert result.report.accepted
    f = np.array([-0.5, 0.0, 2.0])
    weights = np.array([19 / 48, 13 / 48, 1 / 3])
    expected = np.zeros((6, 3))
    expected[3] = f
    expected[:3] = -weights[:, None] * f
    applied = result.normal_loads.nodal_force_n + result.nodal_tangent_force_n
    np.testing.assert_allclose(applied, expected, atol=2e-12)
    new = result.state.particles
    np.testing.assert_allclose(new.velocity - old.velocity, h * expected, atol=2e-12)
    np.testing.assert_allclose(np.sum(new.velocity - old.velocity, axis=0), 0, atol=2e-12)
    tangential_work = float(np.sum(result.nodal_tangent_force_n * (new.position - old.position)))
    assert result.contacts[0].tangent_work_j == pytest.approx(tangential_work, abs=1e-15)
    # The shifted material particles carry a finite arm for the shear traction.
    angular = np.cross(new.position, new.velocity).sum(axis=0) - np.cross(
        old.position, old.velocity
    ).sum(axis=0)
    np.testing.assert_allclose(
        angular, h * np.cross(old.position, expected).sum(axis=0), atol=2e-12
    )


@pytest.mark.tyx
def test_device_normal_and_particle_step_keep_the_entry_material_shear_work(tyx_mode, tmp_path):
    import json
    from contextlib import ExitStack

    with ExitStack() as stack:
        law = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solve = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        model, op, state, rows, R = fixture(dtype=np.float32, rotated=True, layer_operator=law)
        workspace = stack.enter_context(
            ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache")
        )
        initial = state.particles
        samples = []
        h = 1 / 64
        top = rows[3]
        for i in range(1, 33):
            step = op.step(
                state, force(state, top, R[:, 2], -2), h, workspace, particle_solver=solve
            )
            assert step.report.accepted and step.report.backend == "tyx:" + tyx_mode
            assert step.normal_loads.force_backend == tyx_mode
            (site,) = step.contacts
            state = step.state
            v = 1 - 0.5 * i * h
            d = i * h - 0.25 * h * h * i * (i + 1)
            assert state.particles.velocity[top] @ R[:, 0] == pytest.approx(v, abs=3e-6)
            assert site.history.slip_distance_m == pytest.approx(d, abs=3e-6)
            assert site.history.tangent_work_j == pytest.approx(-0.5 * d, abs=3e-6)
            assert site.history.normal_impulse_ns == pytest.approx(2 * i * h, abs=3e-6)
            samples.append(
                dict(
                    tick=i,
                    slip_m=site.history.slip_distance_m,
                    work_j=site.history.tangent_work_j,
                    nodal_tangent_force_n=step.nodal_tangent_force_n.tolist(),
                    position_m=state.particles.position.tolist(),
                    velocity_m_s=state.particles.velocity.tolist(),
                )
            )
        assert law.stats()["completed_launches"] == 32
        if tyx_mode == "cuda":
            assert law.stats()["downloads"] > 0 and workspace.stats()["completed_launches"] > 0
        (tmp_path / "material-sliding-window.json").write_text(
            json.dumps(
                dict(
                    mode=tyx_mode,
                    dt_s=h,
                    shear_backend="HOST",
                    initial_position_m=initial.position.tolist(),
                    initial_velocity_m_s=initial.velocity.tolist(),
                    slot_for_material_id=rows.tolist(),
                    normal_world=R[:, 2].tolist(),
                    tangent_world=R[:, 0].tolist(),
                    external_force_n=force(state, top, R[:, 2], -2).force.tolist(),
                    normal_backend=tyx_mode,
                    particle_backend=step.report.backend,
                    layer=law.stats(),
                    samples=samples,
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_zero_friction_retains_original_particle_path_and_reports_zero_tangential_impulse():
    model, op, start, rows, R = fixture(friction=0.0)
    reference = start.particles
    bindings = {
        shape.key: ph.ParticleAttachment(
            tuple(range(0, 3) if shape.key.body == 13 else range(3, 6))
        )
        for shape in model.shapes
    }
    normal = ph.DeformableFeatureLoadSystem(
        model, bindings, ph.NormalLayer(1 / 16, 64.0), search_distance_m=3 / 32
    )
    mass = ph.ParticleModel(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), np.zeros(3))
    old_solver = ph.SemiImplicitSolver()
    state = start
    with ph.CollisionWorkspace(model) as new_work, ph.CollisionWorkspace(model) as old_work:
        h = 1 / 64
        for i in range(8):
            ext = force(state, rows[3], R[:, 2], -2.0)
            old = normal.sample_particles(reference, old_work)
            reference = old_solver.step(mass, reference, old.step_inputs(ext), h).state
            out = op.step(state, ext, h, new_work)
            assert out.report.accepted
            state = out.state
            np.testing.assert_array_equal(state.particles.position, reference.position)
            np.testing.assert_array_equal(state.particles.velocity, reference.velocity)
            assert state.particles.position.tobytes() == reference.position.tobytes()
            assert state.particles.velocity.tobytes() == reference.velocity.tobytes()
            np.testing.assert_array_equal(out.nodal_tangent_force_n, np.zeros((6, 3)))
            assert out.contacts[0].history.tangent_work_j == 0.0
            assert out.contacts[0].history.slip_distance_m == pytest.approx((i + 1) * h)


def test_reversing_too_far_in_one_step_rejects_without_a_positive_friction_work():
    model, op, start, rows, R = fixture()
    with ph.CollisionWorkspace(model) as work:
        h = 1 / 64
        regular = force(start, rows[3], R[:, 2], -2.0)
        first = op.step(start, regular, h, work)
        detached = first.state.particles
        detached.position[:] = 123
        back = first.state.particles.position.copy()
        extreme = regular.copy()
        extreme.force[rows[3], 0] = -200.0
        with pytest.raises(ph.SolverFailure, match="add work"):
            op.step(first.state, extreme, h, work)
        assert first.state.tick == 1 and first.state.active[0].active_steps == 1
        np.testing.assert_array_equal(first.state.particles.position, back)
        assert op.step(first.state, regular, h, work).contacts[0].history.active_steps == 2

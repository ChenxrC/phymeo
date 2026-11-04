"""Physical stick, breakaway and release of a material vertex on a finite cloth face."""

import json

import numpy as np
import pytest

import phymeo as ph


def scene(*, angled=False, dtype=np.float64, normal_operator=None, movable_face=False):
    from phymeo.solvers.deformable_stiction import MaterialStickSlipStepper

    xyz = np.array(
        [
            [-2, -1, 0],
            [2, -1, 0],
            [0, 1, 0],
            [0, 0, 1 / 32],
            [0.1875, 0.25, 0.75],
            [-0.375, 0.5, 1.0],
        ],
        dtype=dtype,
    )
    rotation = np.array(
        [[0.8, 0, -0.6], [0, 1, 0], [0.6, 0, 0.8]] if angled else np.eye(3),
        dtype=dtype,
    )
    order = [4, 2, 5, 1, 0, 3] if angled else list(range(6))
    rows = np.argsort(order)
    faces = [
        ph.CollisionShape(ph.CollisionKey(9, 3 if angled else 12, 0), "triangle", (17, 31, 49)),
        ph.CollisionShape(ph.CollisionKey(9, 12 if angled else 3, 0), "triangle", (2, 18, 42)),
    ]
    geometry = ph.CollisionModel(faces[::-1] if angled else faces)
    bindings = {
        face.key: ph.ParticleAttachment(tuple(rows[3 * i : 3 * i + 3]))
        for i, face in enumerate(faces)
    }
    inv = np.array([float(movable_face)] * 3 + [0.5 if angled else 1, 1, 1], dtype=dtype)
    mass = ph.ParticleModel(inv[order], np.zeros(3, dtype=dtype))
    layer = ph.NormalLayer(1 / 16, 64, friction=0.25)
    stepper = MaterialStickSlipStepper(
        mass,
        geometry,
        bindings,
        layer,
        tangent_stiffness_n_per_m=128 if angled else 256,
        static_friction=0.5,
        search_distance_m=3 / 32,
        layer_operator=normal_operator,
    )
    initial = stepper.initial_state(
        ph.ParticleState((xyz @ rotation.T)[order], np.zeros((6, 3), dtype=dtype))
    )
    tangent = rotation @ np.array([0.6, 0.8, 0] if angled else [1, 0, 0], dtype=dtype)
    return geometry, stepper, initial, rows, tangent, rotation[:, 2]


def applied(state, tip, tangent, normal, shear, press=-2.0):
    out = ph.StepInputs.zeros(6, dtype=state.particles.position.dtype)
    out.force[tip] = shear * tangent + press * normal
    return out


def test_hold_distinguishes_elastic_displacement_from_permanent_material_slip(tmp_path):
    geo, op, initial, rows, tangent, normal = scene()
    tip, h, k = rows[3], 1 / 16, 256
    external = applied(initial, tip, tangent, normal, 0.5)
    state, x, v, work, samples = initial, 0.0, 0.0, 0.0, []
    with ph.CollisionWorkspace(geo) as workspace:
        for tick in range(1, 33):
            # Independent scalar mass-spring balance, with a constant 0.5 N horizontal load.
            new_v = (v + h * (0.5 - k * x)) / (1 + h * h * k)
            new_x = x + h * new_v
            row = op.step(state, external, h, workspace)
            assert row.report.accepted and not row.ended
            (contact,) = row.contacts
            shear = float(row.nodal_tangent_force_n[tip] @ tangent)
            assert contact.feature.kind == "VT" and contact.normal_force_n == pytest.approx(2)
            assert contact.history.regime == "stick" and contact.history.start_tick == 0
            assert contact.history.plastic_slip_m == 0
            np.testing.assert_allclose(contact.plastic_increment_m, 0, atol=1e-14)
            assert contact.history.stored_energy_j == pytest.approx(0.5 * k * new_x**2)
            assert shear == pytest.approx(-k * new_x, abs=1e-11)
            assert new_v - v == pytest.approx(h * (0.5 + shear), abs=1e-12)
            assert row.state.particles.position[tip] @ tangent == pytest.approx(new_x, abs=1e-12)
            assert row.state.particles.velocity[tip] @ tangent == pytest.approx(new_v, abs=1e-12)
            work += shear * (new_x - x)
            assert contact.history.tangent_work_j == pytest.approx(work, abs=1e-12)
            np.testing.assert_allclose(row.nodal_tangent_force_n.sum(axis=0), 0, atol=1e-13)
            samples.append(
                dict(
                    tick=tick,
                    actual_x_m=float(row.state.particles.position[tip] @ tangent),
                    actual_v_m_s=float(row.state.particles.velocity[tip] @ tangent),
                    force_n=shear,
                    stored_j=contact.history.stored_energy_j,
                    total_work_j=contact.history.tangent_work_j,
                    plastic_m=contact.history.plastic_slip_m,
                    oracle_x_m=new_x,
                    oracle_v_m_s=new_v,
                )
            )
            state, x, v = row.state, new_x, new_v
        assert x == pytest.approx(0.5 / k, abs=4e-8) and abs(v) < 2e-6
        recovered = op.step(state, applied(state, tip, tangent, normal, 0), h, workspace)
        assert recovered.contacts[0].history.regime == "stick"
        assert recovered.contacts[0].tangent_work_j > 0
        assert recovered.contacts[0].history.stored_energy_j < state.active[0].stored_energy_j
        for _ in range(47):
            recovered = op.step(
                recovered.state, applied(state, tip, tangent, normal, 0), h, workspace
            )
        assert recovered.contacts[0].history.plastic_slip_m == 0
        assert abs(recovered.nodal_tangent_force_n[tip] @ tangent) < 4e-8
        assert recovered.contacts[0].history.stored_energy_j < 3e-18
    (tmp_path / "material-static-hold.json").write_text(json.dumps(samples), encoding="utf8")


@pytest.mark.parametrize("angled", [False, True])
def test_overload_slides_at_the_kinetic_limit_with_material_reactions_and_energy(angled, tmp_path):
    geo, op, start, rows, tangent, normal = scene(angled=angled)
    h, k, mass = 1 / 16, (128 if angled else 256), (2 if angled else 1)
    top = rows[3]
    applied_shear = 12 if angled else 4
    with ph.CollisionWorkspace(geo) as workspace:
        step = op.step(start, applied(start, top, tangent, normal, applied_shear), h, workspace)
        c = step.contacts[0]
        assert c.history.regime == "slide" and c.normal_force_n == pytest.approx(2)
        # Once yielded, the only tangential contact load is mu_k*N=0.5 N.
        v = h * (applied_shear - 0.5) / mass
        dx = h * v
        elastic, plastic = 0.5 / k, dx - 0.5 / k
        assert plastic > 0
        np.testing.assert_allclose(step.state.particles.velocity[top], v * tangent, atol=3e-13)
        np.testing.assert_allclose(c.history.elastic_displacement_m, elastic * tangent, atol=3e-13)
        np.testing.assert_allclose(c.plastic_increment_m, plastic * tangent, atol=3e-13)
        assert c.history.plastic_slip_m == pytest.approx(plastic)
        assert c.tangent_work_j == pytest.approx(-0.5 * dx)
        assert c.plastic_dissipation_j == pytest.approx(0.5 * plastic)
        assert c.history.stored_energy_j == pytest.approx(0.5 * k * elastic**2)
        assert c.tangent_work_j + c.history.stored_energy_j + c.plastic_dissipation_j + (
            c.implicit_dissipation_j
        ) == pytest.approx(0, abs=2e-13)
        np.testing.assert_allclose(step.nodal_tangent_force_n[top], -0.5 * tangent, atol=3e-13)
        for i, w in enumerate((0.25, 0.25, 0.5)):
            np.testing.assert_allclose(
                step.nodal_tangent_force_n[rows[i]], 0.5 * w * tangent, atol=3e-13
            )
        again = op.step(step.state, applied(step.state, top, tangent, normal, 0.75), h, workspace)
        assert again.contacts[0].history.regime == "slide"
        assert again.contacts[0].history.plastic_slip_m > c.history.plastic_slip_m
        # Remove the free tangential velocity. The spring unloads inside the kinetic cone.
        caught = op.step(
            step.state, applied(step.state, top, tangent, normal, -mass * v / h), h, workspace
        )
        restored = caught.contacts[0]
        assert restored.history.regime == "stick" and restored.tangent_work_j > 0
        assert restored.history.plastic_slip_m == pytest.approx(plastic)
        assert restored.history.stored_energy_j < c.history.stored_energy_j
        (tmp_path / "material-breakaway.json").write_text(
            json.dumps(
                dict(
                    mass_kg=mass,
                    k_t_n_per_m=k,
                    dt_s=h,
                    normal_world=normal.tolist(),
                    tangent_world=tangent.tolist(),
                    applied_shear_n=applied_shear,
                    vertex_velocity_m_s=step.state.particles.velocity[top].tolist(),
                    vertex_position_m=step.state.particles.position[top].tolist(),
                    normal_n=c.normal_force_n,
                    nodal_shear_n=step.nodal_tangent_force_n.tolist(),
                    face_weights=c.material.second_weights,
                    elastic_m=c.history.elastic_displacement_m,
                    plastic_m=c.history.plastic_slip_m,
                    tangent_work_j=c.tangent_work_j,
                    stored_j=c.history.stored_energy_j,
                    plastic_loss_j=c.plastic_dissipation_j,
                    implicit_loss_j=c.implicit_dissipation_j,
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_release_drops_elastic_storage_and_reentry_does_not_restore_old_shear():
    geo, op, start, rows, tangent, normal = scene()
    h, top = 1 / 16, rows[3]
    with ph.CollisionWorkspace(geo) as workspace:

        def advance(state, shear, vertical):
            return op.step(
                state, applied(state, top, tangent, normal, shear, vertical), h, workspace
            )

        held = advance(start, 0.5, -2)
        lifted = advance(held.state, 0.5, 14)
        assert lifted.state.particles.position[top] @ normal == pytest.approx(3 / 32)
        vx = lifted.state.particles.velocity[top] @ tangent
        separated = advance(lifted.state, -vx / h, -32)
        assert not separated.contacts and separated.state.active == ()
        assert separated.ended == lifted.state.active
        assert separated.released_spring_energy_j == pytest.approx(
            lifted.state.active[0].stored_energy_j
        )
        assert separated.released_spring_energy_j > 0
        np.testing.assert_array_equal(separated.nodal_tangent_force_n, np.zeros((6, 3)))
        assert separated.state.particles.position[top] @ normal == pytest.approx(1 / 32)
        reentered = advance(separated.state, 0, 14)
        assert reentered.contacts[0].history.start_tick == separated.state.tick
        assert reentered.contacts[0].history.stored_energy_j == pytest.approx(0, abs=1e-14)
        np.testing.assert_allclose(reentered.nodal_tangent_force_n, 0, atol=1e-13)
        replay = advance(lifted.state, -vx / h, -32)
        np.testing.assert_array_equal(
            replay.state.particles.position, separated.state.particles.position
        )
        assert replay.ended == separated.ended


def test_static_limit_is_inclusive_and_invalid_candidate_preserves_the_checkpoint():
    geo, op, start, rows, tangent, normal = scene()
    h, top = 1 / 16, rows[3]
    with ph.CollisionWorkspace(geo) as workspace:
        held = op.step(start, applied(start, top, tangent, normal, 2), h, workspace)
        (c,) = held.contacts
        assert c.history.regime == "stick"
        np.testing.assert_allclose(held.nodal_tangent_force_n[top], -tangent, atol=1e-13)
        with pytest.raises(ph.SolverFailure, match="cross"):
            op.step(held.state, applied(held.state, top, tangent, normal, 0, -64), h, workspace)
        assert held.state.tick == 1 and held.state.active == (c.history,)
        ok = op.step(held.state, applied(held.state, top, tangent, normal, 0), h, workspace)
        assert ok.report.accepted and ok.state.tick == 2

    geo, op, start, *_ = scene(movable_face=True)
    with ph.CollisionWorkspace(geo) as workspace, pytest.raises(ph.SolverFailure, match="fixed"):
        op.step(start, ph.StepInputs.zeros(6), h, workspace)
    assert start.active == () and start.tick == 0


@pytest.mark.tyx
def test_native_normal_and_particle_updates_hold_then_break_away(tyx_mode, tmp_path):
    from contextlib import ExitStack

    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solver = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        geo, op, start, rows, tangent, normal = scene(dtype=np.float32, normal_operator=layer)
        work = stack.enter_context(
            ph.CollisionWorkspace(geo, tyx_mode, cache_root=tmp_path / "cache")
        )
        tip, h, k = rows[3], 1 / 16, 256
        state, x, v, journal = start, 0.0, 0.0, []
        for external_shear in [0.5] * 12 + [4.0] * 3:
            candidate = op.step(
                state,
                applied(state, tip, tangent, normal, external_shear),
                h,
                work,
                particle_solver=solver,
            )
            (contact,) = candidate.contacts
            if external_shear == 0.5:
                v = (v + h * (external_shear - k * x)) / (1 + h * h * k)
                x += h * v
                assert contact.history.regime == "stick"
                assert contact.history.plastic_slip_m == 0
            else:
                v += h * (external_shear - 0.5)
                x += h * v
                assert contact.history.regime == "slide"
            assert candidate.report.backend == "tyx:" + tyx_mode
            assert candidate.normal_loads.force_backend == tyx_mode
            assert candidate.state.particles.position[tip, 0] == pytest.approx(x, abs=8e-7)
            assert candidate.state.particles.velocity[tip, 0] == pytest.approx(v, abs=2e-6)
            state = candidate.state
            journal.append(
                dict(
                    tick=state.tick,
                    x_m=float(state.particles.position[tip, 0]),
                    v_m_s=float(state.particles.velocity[tip, 0]),
                    force_n=candidate.nodal_tangent_force_n[tip].tolist(),
                    normal_n=contact.normal_force_n,
                    regime=contact.history.regime,
                    elastic_m=contact.history.elastic_displacement_m,
                    plastic_m=contact.history.plastic_slip_m,
                    stored_j=contact.history.stored_energy_j,
                    work_j=contact.history.tangent_work_j,
                )
            )
        assert layer.stats()["completed_launches"] == 15
        if tyx_mode == "cuda":
            assert work.stats()["completed_launches"] > 0 and layer.stats()["downloads"] > 0
        (tmp_path / "material-stiction-device.json").write_text(
            json.dumps(
                dict(
                    mode=tyx_mode,
                    tangential_solve="host",
                    normal_operator=layer.stats(),
                    particle_backend=candidate.report.backend,
                    geometry=work.stats(),
                    steps=journal,
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_independent_loaded_vertices_share_a_fixed_face_and_rejection_keeps_both_histories():
    from phymeo.core.result import StepReport, StepResult
    from phymeo.solvers.deformable_stiction import MaterialStickSlipStepper

    base = np.array([[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    upper = np.array([[0.0, 0.0, 1 / 32], [0.1875, 0.25, 0.75], [-0.375, 0.5, 1.0]])
    x = np.vstack((base, upper - [0.5, 0, 0], upper + [0.5, 0, 0]))
    faces = [
        ph.CollisionShape(ph.CollisionKey(2, body, 0), "triangle", (0, 1, 2))
        for body in (90, 51, 24)
    ]
    geometry = ph.CollisionModel(faces)
    attachments = {
        face.key: ph.ParticleAttachment((3 * i, 3 * i + 1, 3 * i + 2))
        for i, face in enumerate(faces)
    }
    model = ph.ParticleModel(np.array([0.0] * 3 + [1.0] * 6), np.zeros(3))
    op = MaterialStickSlipStepper(
        model,
        geometry,
        attachments,
        ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=256,
    )
    start = op.initial_state(ph.ParticleState(x, np.zeros_like(x)))
    load = ph.StepInputs.zeros(9)
    load.force[3], load.force[6] = [0.5, 0, -2], [0, 4, -2]

    class Reject:
        def step(self, model, state, inputs, dt):
            return StepResult(
                state.copy(), StepReport(False, reason="deliberate rejected candidate")
            )

    with ph.CollisionWorkspace(geometry) as work:
        first = op.step(start, load, 1 / 16, work)
        assert sorted(c.history.regime for c in first.contacts) == ["slide", "stick"]
        np.testing.assert_allclose(first.nodal_tangent_force_n[3], [-0.25, 0, 0], atol=1e-13)
        np.testing.assert_allclose(first.nodal_tangent_force_n[6], [0, -0.5, 0], atol=1e-13)
        np.testing.assert_allclose(first.nodal_tangent_force_n.sum(axis=0), 0, atol=1e-13)
        failed = op.step(first.state, load, 1 / 16, work, particle_solver=Reject())
        assert failed.state is first.state and not failed.report.accepted
        assert not failed.contacts and not failed.ended and failed.nodal_tangent_force_n is None

        class WrongMomentum:
            def step(self, model, state, inputs, dt):
                real = ph.SemiImplicitSolver().step(model, state, inputs, dt)
                real.state.velocity[6, 1] += 0.125
                return real

        with pytest.raises(ph.SolverFailure, match="mass response"):
            op.step(first.state, load, 1 / 16, work, particle_solver=WrongMomentum())
        resumed, replay = (
            op.step(failed.state, load, 1 / 16, work),
            op.step(first.state, load, 1 / 16, work),
        )
        np.testing.assert_array_equal(
            resumed.state.particles.position, replay.state.particles.position
        )
        assert resumed.contacts == replay.contacts


def test_one_free_vertex_loaded_by_two_fixed_faces_requires_a_coupled_solve():
    from phymeo.solvers.deformable_stiction import MaterialStickSlipStepper

    xyz = np.array(
        [
            [-2, -1, 0],
            [2, -1, 0],
            [0, 1, 0],
            [0, 1, 1 / 32],
            [0, 0, 3 / 64],
            [0.1875, 0.25, 0.75],
            [-0.375, 0.5, 1],
        ]
    )
    faces = [
        ph.CollisionShape(ph.CollisionKey(2, body, shape), "triangle", vertices)
        for body, shape, vertices in ((3, 0, (0, 1, 2)), (3, 1, (1, 0, 3)), (8, 0, (4, 5, 6)))
    ]
    geometry = ph.CollisionModel(faces)
    attachments = {shape.key: ph.ParticleAttachment(shape.vertex_ids) for shape in faces}
    model = ph.ParticleModel(np.array([0.0] * 4 + [1.0] * 3), np.zeros(3))
    op = MaterialStickSlipStepper(
        model,
        geometry,
        attachments,
        ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=256,
    )
    start = op.initial_state(ph.ParticleState(xyz, np.zeros_like(xyz)))
    with ph.CollisionWorkspace(geometry) as work, pytest.raises(ph.SolverFailure, match="share"):
        op.step(start, ph.StepInputs.zeros(7), 1 / 16, work)
    assert start.tick == 0 and not start.active

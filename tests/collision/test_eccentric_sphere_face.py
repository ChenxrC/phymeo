"""An attached ball follows the rigid COM's physical angular drift against free material."""

import json
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np
import pytest

import phymeo as ph


def scene(
    positions, velocities, arm, *, radius=1 / 16, dtype=np.float64, reverse=False, operator=None
):
    ball = ph.CollisionShape(ph.CollisionKey(7, 4 if reverse else 1, 11), "sphere", (801,), radius)
    face = ph.CollisionShape(ph.CollisionKey(7, 2, 5), "triangle", (17, 31, 9))
    model = ph.CollisionModel([face, ball])
    binds = {
        ball.key: ph.RigidAttachment(0, (arm,)),
        face.key: ph.ParticleAttachment((0, 1, 2)),
    }
    layers = {(ball.key, face.key): ph.NormalLayer(1 / 16, 128)}
    geometry = ph.CollisionLoadSystem(model, binds, layers, layer_operator=operator)
    particles = ph.ParticleState(
        np.array(positions, dtype=dtype), np.array(velocities, dtype=dtype)
    )
    zero = np.zeros((1, 3), dtype=dtype)
    body = ph.RigidBodyState(zero, np.array([[0, 0, 0, 1]], dtype=dtype), zero, zero)
    models = (
        ph.ParticleModel(np.array([1, 2 / 3, 0.5], dtype=dtype), zero[0]),
        ph.RigidBodyModel(np.array([0.5], dtype=dtype), np.ones((1, 3), dtype=dtype), zero[0]),
    )
    return geometry, ph.CoupledState(particles, body), models, ball.key, face.key


def arc_face(*, side=0.0, reverse=False, full=False):
    # At x=0 the endpoint chord is more than one radius outside the finite triangle.
    z = -0.25
    xyz = np.array([[0.25, -0.5, z], [0.75, -0.5, z], [0.5, 0.5, z]])
    xyz[:, 0] += side
    drift = np.array([1 / 8, -1 / 16, 3 / 32])
    data = scene(xyz, np.tile(drift, (3, 1)), (0.0, 0.0, 0.5), reverse=reverse)
    system, initial, _, ball, tri = data
    initial.rigid.velocity[0] = drift
    angle = np.pi * (2 if full else 1)
    initial.rigid.angular_velocity[0, 1] = angle
    turn = ph.RigidTurn((0, 0, 0), drift, (0, 1, 0), angle)
    path = {
        ball: ph.CollisionMotion([[0, 0, 0.5]], turn=turn),
        tri: ph.CollisionMotion(xyz, xyz + drift),
    }
    return data, path


@pytest.mark.parametrize("reverse,full", [(False, False), (True, False), (True, True)])
def test_arc_on_translating_face_finds_interior_contact_that_the_chord_misses(
    reverse, full, tmp_path
):
    (system, state, _, ball, face), path = arc_face(reverse=reverse, full=full)
    drift = state.rigid.velocity[0]
    angle = path[ball].turn.angle
    oracle = np.arccos(-3 / 8) / angle
    assert oracle > 0.5 if not full else oracle < 0.5
    with ph.CollisionWorkspace(system.model) as workspace:
        (pair,) = ph.detect(system.model, path, ph.CollisionPolicy(), workspace).pairs
        assert pair.status == "hit" and pair.sweep is not None
        obs = pair.sweep
        assert obs.start_gap_m > 0 and obs.end_gap_m > 0
        assert obs.first_t[0] <= oracle <= obs.first_t[1]
        assert obs.first_t[1] - obs.first_t[0] <= 1 / 1024
        t = obs.witness_time
        w = obs.witness
        center = t * drift + 0.5 * np.array([np.sin(angle * t), 0, np.cos(angle * t)])
        wp, wf = (w.second_point, w.first_point) if reverse else (w.first_point, w.second_point)
        wm = w.first_weights if reverse else w.second_weights
        assert all(value > 0 for _, value in wm)
        xyz = state.particles.position + drift * t
        np.testing.assert_allclose(
            wf, np.array([dict(wm)[i] for i in (17, 31, 9)]) @ xyz, atol=3e-7
        )
        assert np.linalg.norm(np.asarray(wp) - center) == pytest.approx(1 / 16, abs=3e-7)
        assert wf[2] == pytest.approx(-0.25 + t * drift[2], abs=3e-7)
        # Same start and end vertices, only the center chord replaces the physical turn.
        final = [0.0, 0.0, 0.5 if full else -0.5]
        chord = dict(path)
        chord[ball] = ph.CollisionMotion([[0, 0, 0.5]], [np.asarray(final) + drift])
        assert (
            ph.detect(system.model, chord, ph.CollisionPolicy(), workspace).pairs[0].status
            == "separated"
        )
    (tmp_path / "arc-hit.json").write_text(
        json.dumps(
            dict(oracle=oracle, angle=angle, drift=drift.tolist(), first=asdict(obs)),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def test_whole_rigid_cloth_step_rejects_an_intermediate_hard_hit_and_keeps_published_state():
    (system, hit, models, _, _), _ = arc_face()
    _, clear, _, _, _ = arc_face(side=0.5)[0]
    inputs = ph.CoupledInputs.zeros(3, 1)
    p, r = ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver()
    with ph.CollisionWorkspace(system.model) as workspace:
        safe = system.step(clear, inputs, models, (p, r), 1.0, workspace)
        assert safe.path.pairs[0].status == "separated" and not safe.loads.contacts
        prior = workspace.last_result
        with pytest.raises(ph.SolverFailure, match="core path rejected"):
            system.step(hit, inputs, models, (p, r), 1.0, workspace)
        assert system.last_frame is safe and workspace.last_result is prior
        np.testing.assert_array_equal(hit.rigid.orientation[0], [0, 0, 0, 1])
        again = system.step(clear, inputs, models, (p, r), 1.0, workspace)
        np.testing.assert_array_equal(again.state.rigid.orientation, safe.state.rigid.orientation)


def loaded(mode, root):
    dtype = np.float64 if mode == "host" else np.float32
    tol = 3e-11 if mode == "host" else 3e-6
    points = [[-0.5, -0.5, 0], [0.75, -0.5, 0], [0, 0.5, 0]]
    velocities = [[0, 1 / 32, -0.02], [0, 1 / 32, 0.015], [0, 1 / 32, 0.0]]
    h = 1 / 128
    with ExitStack() as stack:
        if mode == "host":
            p, r, layer = ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver(), None
        else:
            layer = stack.enter_context(ph.TyxNormalLayer(mode, cache_root=root / "cache"))
            p = stack.enter_context(ph.TyxSemiImplicitSolver(mode, cache_root=root / "cache"))
            r = stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode, cache_root=root / "cache"))
        system, s, models, ball, _ = scene(
            points, velocities, (7 / 200, -24 / 200, 0), radius=1 / 8, dtype=dtype, operator=layer
        )
        s.rigid.position[0, 2] = 5 / 32
        s.rigid.velocity[0, 1] = 1 / 32
        s.rigid.orientation[0] = (0, 0, 3 / 5, 4 / 5)
        s.rigid.angular_velocity[0, 1] = 2
        workspace = stack.enter_context(
            ph.CollisionWorkspace(system.model, mode, cache_root=root / "cache")
        )
        if mode == "cuda":
            import tyx

            start_kernels = tyx.runtime_stats().cuda_kernel_launches
        out = system.step(
            s, ph.CoupledInputs.zeros(3, 1, dtype=dtype), models, (p, r), h, workspace
        )
        launches = (
            tyx.runtime_stats().cuda_kernel_launches - start_kernels if mode == "cuda" else None
        )
        (contact,) = out.loads.contacts
        assert contact.gap_m == pytest.approx(1 / 32, abs=tol)
        assert contact.force_n == pytest.approx(4.0, abs=2 * tol)
        expected = np.array([0.2, 0.3, 0.5])[:, None] * [0.0, 0.0, -4.0]
        np.testing.assert_allclose(
            (out.state.particles.velocity - s.particles.velocity) / models[0].inverse_mass[:, None],
            h * expected,
            atol=tol,
        )
        np.testing.assert_allclose(
            2 * (out.state.rigid.velocity - s.rigid.velocity), [[0, 0, 4 * h]], atol=tol
        )
        np.testing.assert_allclose(
            out.state.rigid.angular_velocity - s.rigid.angular_velocity,
            [[0, -0.5 * h, 0]],
            atol=tol,
        )
        dt_omega = h * (2 - 0.5 * h)
        endpoint = (
            out.state.rigid.position[0] + np.array([np.cos(dt_omega), 0, -np.sin(dt_omega)]) / 8
        )
        assert out.path.pairs[0].status == "separated"
        witness = out.path.pairs[0].witness
        radius = np.linalg.norm(np.asarray(witness.first_point) - endpoint)
        assert radius == pytest.approx(1 / 8, abs=3e-6)
        x1 = out.state.particles.position
        final_n = np.cross(x1[1] - x1[0], x1[2] - x1[0])
        assert abs(final_n[0]) > 1e-5
        if mode == "cuda":
            assert launches >= 4 and layer.stats()["completed_launches"] == 1
        (root / "eccentric-coupled-step.json").write_text(
            json.dumps(
                dict(
                    mode=mode,
                    launches=launches,
                    backend=out.path.backend,
                    entry_normal_n=contact.force_n,
                    entry_gap_m=contact.gap_m,
                    particle_impulse=(h * out.loads.inputs.particles.force).tolist(),
                    rigid_impulse=(h * out.loads.inputs.rigid.force).tolist(),
                    rigid_angular_impulse=(h * out.loads.inputs.rigid.torque).tolist(),
                    rigid_omega=out.state.rigid.angular_velocity.tolist(),
                    center=endpoint.tolist(),
                    triangle=x1.tolist(),
                    end_witness=asdict(witness),
                    particle_m_s=out.state.particles.velocity.tolist(),
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_deforming_cloth_and_offset_ball_exchange_force_and_rigid_angular_impulse(tmp_path):
    loaded("host", tmp_path)


@pytest.mark.tyx
def test_same_coupled_step_advances_real_native_participants(tyx_mode, tmp_path):
    loaded(tyx_mode, tmp_path)


def test_offset_motion_rejects_an_unknown_angular_solver_and_a_forged_end_pose():
    from phymeo.core.result import StepResult

    (system, clear, models, _, _), _ = arc_face(side=0.5)
    inputs = ph.CoupledInputs.zeros(3, 1)

    class Unknown:
        def step(self, *args):
            return ph.RigidSemiImplicitSolver().step(*args)

    class Forged(Unknown):
        rigid_motion_contract = ph.RigidSemiImplicitSolver.rigid_motion_contract

        def step(self, *args):
            accepted = super().step(*args)
            result = accepted.state.copy()
            result.orientation[0] = [0, 0, 0, 1]
            return StepResult(result, accepted.report)

    with ph.CollisionWorkspace(system.model) as workspace:
        for solver in (Unknown(), Forged()):
            with pytest.raises(ph.SolverFailure, match="angular drift|continuous rigid motion"):
                system.step(
                    clear, inputs, models, (ph.SemiImplicitSolver(), solver), 1.0, workspace
                )
            assert system.last_frame is None and workspace.last_result is None


def test_rigid_capsule_rotation_is_still_an_explicit_coupled_domain_limit():
    (base, state, models, ball, _), _ = arc_face(side=0.5)
    registrations = tuple(
        ph.CollisionShape(ball, "capsule", (801, 802), 1 / 16) if s.key == ball else s
        for s in base.model.shapes
    )
    bindings = dict(base.bindings)
    bindings[ball] = ph.RigidAttachment(0, ((0, 0, 0.5), (0.125, 0, 0.5)))
    system = ph.CollisionLoadSystem(ph.CollisionModel(registrations), bindings, base.layers)
    with ph.CollisionWorkspace(system.model) as work:
        with pytest.raises(ph.SolverFailure, match="attached capsule requires"):
            system.step(
                state,
                ph.CoupledInputs.zeros(3, 1),
                models,
                (ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver()),
                1.0,
                work,
            )
        assert system.last_frame is None and work.last_result is None


def test_finite_moving_edge_near_miss_preserves_a_four_millimeter_clearance():
    # All material nodes have y >= r+1/256 in the co-translating frame. At t=2/3
    # the projected ball center lies on the lower edge's x interval and z plane.
    (system, state, models, _, _), _ = arc_face()
    minimum_y = 1 / 16 + 1 / 256
    state.particles.position[:, 1] = [minimum_y, minimum_y, minimum_y + 0.75]
    x = 0.5 * np.sin(2 * np.pi / 3)
    assert 0.25 < x < 0.75 and minimum_y - 1 / 16 == 1 / 256
    with ph.CollisionWorkspace(system.model) as work:
        out = system.step(
            state,
            ph.CoupledInputs.zeros(3, 1),
            models,
            (ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver()),
            1.0,
            work,
        )
        (pair,) = out.path.pairs
        assert pair.candidate and pair.status == "separated"
        assert pair.sweep is not None and pair.sweep.first_t is None
        assert not out.loads.contacts

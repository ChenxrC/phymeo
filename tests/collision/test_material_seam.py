"""Pressure stays active while a material vertex crosses exactly one registered shared edge."""

import json
from contextlib import ExitStack

import numpy as np
import pytest

import phymeo as ph

LEFT = ph.CollisionKey(9, 4, 20)
RIGHT = ph.CollisionKey(9, 4, 30)
NORMAL = np.array([0.0, 0.0, 1.0])
TANGENT = np.array([1.0, 0.0, 0.0])
H = 1 / 8


def model(*, disconnected=False, rotated=False, dtype=np.float64, operator=None):
    from phymeo.solvers.material_seam import MaterialSeamStepper, TranslatingMaterialPatch

    rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]]) if rotated else np.eye(3)
    points = {7: (-1, 0, 0), 17: (0, -1, 0), 31: (0, 1, 0), 49: (1, 0, 0)}
    right = (17, 49, 31)
    if disconnected:
        points.update({70: points[17], 90: points[31]})
        right = (70, 49, 90)
    faces = [
        ph.CollisionShape(LEFT, "triangle", (7, 17, 31)),
        ph.CollisionShape(RIGHT, "triangle", right),
    ]
    patch = TranslatingMaterialPatch(
        {i: rot @ p for i, p in points.items()},
        ph.CollisionModel(faces[::-1] if rotated else faces),
    )
    op = MaterialSeamStepper(
        patch,
        point_material=(9, 13, 2),
        point_mass_kg=1.0,
        patch_mass_kg=2.0,
        layer=ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=16.0,
        dtype=dtype,
        layer_operator=operator,
    )
    return op, rot


def initial(op, rot, x, speed):
    base_v = np.array([0.25, 0.125, 0])
    return op.initial_state(
        rot @ [x, 0, 1 / 32],
        rot @ (base_v + [speed, 0, 0]),
        patch_translation=rot @ [0.0, 0.0, 0.0],
        patch_velocity=rot @ base_v,
    )


def load(rot, top=0.0, base=0.0, top_z=-2.0, base_z=2.0, dtype=np.float64):
    return ph.StepInputs(np.array([rot @ [top, 0, top_z], rot @ [base, 0, base_z]], dtype=dtype))


def slip_cross(op, rot, *, solver=None, dtype=np.float64):
    state = initial(op, rot, -0.75, 4.0)
    previous = op.step(state, load(rot, 0.5, -0.5, dtype=dtype), H, particle_solver=solver)
    assert previous.report.accepted and len(previous.segments) == 1
    assert previous.state.history.regime == "slide"
    assert previous.state.history.plastic_slip_m == pytest.approx(15 / 32)
    answer = op.step(previous.state, load(rot, dtype=dtype), H, particle_solver=solver)
    return previous, answer


@pytest.mark.parametrize("rotated", [False, True])
def test_loaded_sliding_crossing_splits_both_material_impulses_and_work(rotated, tmp_path):
    op, R = model(rotated=rotated)
    before, out = slip_cross(op, R)
    assert out.report.accepted
    a, b = out.segments
    tau = 64 / 125
    assert (a.face, b.face) == (LEFT, RIGHT)
    assert (a.t0, a.t1, b.t0, b.t1) == pytest.approx((0, tau, tau, 1))
    assert dict(a.exit_weights) == {7: 0.0, 17: 0.5, 31: 0.5}
    assert dict(b.entry_weights) == {17: 0.5, 31: 0.5, 49: 0.0}
    np.testing.assert_allclose(a.exit_point_on_patch_m, b.entry_point_on_patch_m)
    np.testing.assert_allclose(a.exit_point_on_vertex_m, b.entry_point_on_vertex_m)
    assert out.state.history.start_tick == 0 and out.state.history.active_steps == 2
    assert out.state.history.plastic_slip_m == pytest.approx(15 / 32 + 125 / 256)
    assert out.state.history.stored_energy_j == pytest.approx(1 / 128)
    np.testing.assert_allclose(out.state.history.elastic_displacement_m, R @ [1 / 32, 0, 0])
    end_x = 61 / 256
    expect = [
        (a, tau, {7: 1 / 8, 17: 7 / 16, 31: 7 / 16}),
        (b, 1 - tau, {17: 0.5 - end_x / 4, 31: 0.5 - end_x / 4, 49: end_x / 2}),
    ]
    force = R @ [-0.5, 0, 2.0]
    dx = out.state.particles.position - before.state.particles.position
    total = np.zeros(3)
    for seg, portion, weights in expect:
        np.testing.assert_allclose(seg.impulse_on_vertex_ns, portion * H * force, atol=1e-13)
        for node, impulse in seg.node_impulses_ns:
            np.testing.assert_allclose(impulse, -portion * H * weights[node] * force, atol=1e-13)
            total += impulse
        assert seg.vertex_work_j == pytest.approx(portion * force @ dx[0])
        assert sum(w for _, w in seg.node_work_j) == pytest.approx(-portion * force @ dx[1])
        assert seg.tangent_work_j == pytest.approx(-0.5 * (125 / 256) * portion)
    np.testing.assert_allclose(total, -H * force)
    dv = out.state.particles.velocity - before.state.particles.velocity
    np.testing.assert_allclose(dv[0], R @ [-1 / 16, 0, 0])
    np.testing.assert_allclose(2 * dv[1], R @ [1 / 16, 0, 0])
    np.testing.assert_allclose(out.patch_constraint_angular_impulse_nms, R @ [0, 3 / 2048, 0])
    assert out.state.history.tangent_work_j - before.state.history.tangent_work_j == pytest.approx(
        -125 / 512
    )
    replay = op.step(before.state, load(R), H)
    assert replay.segments == out.segments and replay.state.history == out.state.history
    (tmp_path / "material-seam-loaded.json").write_text(
        json.dumps(
            dict(
                tau=a.t1,
                force_n=out.force_on_vertex_n,
                shared_points=[a.exit_point_on_patch_m, b.entry_point_on_patch_m],
                impulses=[s.node_impulses_ns for s in out.segments],
                tangent_work_j=[s.tangent_work_j for s in out.segments],
                vertex_work_j=[s.vertex_work_j for s in out.segments],
                support_work_j=[s.node_work_j for s in out.segments],
                angular_constraint_nms=out.patch_constraint_angular_impulse_nms,
                plastic_m=out.state.history.plastic_slip_m,
                position=out.state.particles.position.tolist(),
                velocity=out.state.particles.velocity.tolist(),
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def test_elastic_hold_and_plastic_history_both_survive_a_real_seam():
    op, R = model()
    at = initial(op, R, -3 / 128, 1 / 8)
    loaded = op.step(at, load(R, 0.25, -0.25), H)
    crossed = op.step(loaded.state, load(R, 0.5, -0.5), H)
    assert crossed.state.face == RIGHT and crossed.state.history.key == loaded.state.history.key
    assert loaded.state.history.regime == crossed.state.history.regime == "stick"
    assert loaded.state.history.stored_energy_j == pytest.approx(1 / 512)
    assert crossed.state.history.stored_energy_j == pytest.approx(1 / 128)
    assert crossed.state.history.plastic_slip_m == 0
    assert crossed.segments[0].t1 == 0.5 and crossed.segments[1].t0 == 0.5
    assert [part.tangent_work_j for part in crossed.segments] == pytest.approx([-1 / 256, -1 / 256])
    np.testing.assert_allclose(crossed.force_on_vertex_n, [-0.5, 0, 2])

    before, forward = slip_cross(op, R)
    back = op.step(forward.state, load(R, -64), H)
    assert [leg.face for leg in back.segments] == [RIGHT, LEFT]
    assert back.segments[0].t1 == pytest.approx(61 / 128)
    assert (
        back.state.history.key == before.state.history.key and back.state.history.active_steps == 3
    )
    np.testing.assert_allclose(back.state.history.elastic_displacement_m, [-1 / 32, 0, 0])
    assert (
        back.state.history.plastic_slip_m - forward.state.history.plastic_slip_m
        == pytest.approx(7 / 16)
    )


def command_position(state, target, force_on_top):
    x, v = state.particles.position, state.particles.velocity
    mass = np.array([[1.0], [2.0]])
    return ph.StepInputs(
        mass * ((target - x) / H - v) / H - np.array([force_on_top, -force_on_top])
    )


def test_disconnected_material_cannot_steal_history_but_can_engage_after_release():
    op, R = model(disconnected=True)
    start = initial(op, R, -0.75, 4.0)
    pre = op.step(start, load(R, 0.5, -0.5), H)
    with pytest.raises(ph.SolverFailure, match="disconnected"):
        op.step(pre.state, load(R), H)
    assert pre.state.history.active_steps == 1 and pre.state.face == LEFT
    target = pre.state.particles.position.copy()
    target[0, 2] += 0.125
    exiting = op.step(pre.state, command_position(pre.state, target, np.array([-0.5, 0, 2.0])), H)
    assert len(exiting.segments) == 1
    target[0, 0] += 0.5
    free = op.step(exiting.state, command_position(exiting.state, target, np.zeros(3)), H)
    assert not free.segments and free.state.history is None and free.ended == exiting.state.history
    target[0, 2] -= 0.125
    land = op.step(free.state, command_position(free.state, target, np.zeros(3)), H)
    renewed = op.step(land.state, command_position(land.state, target, 2 * NORMAL), H)
    assert renewed.state.face == RIGHT and renewed.state.history.start_tick == land.state.tick
    assert renewed.state.history.key != pre.state.history.key
    assert renewed.state.history.stored_energy_j == 0 and renewed.state.history.plastic_slip_m == 0
    replay = op.step(
        pre.state,
        command_position(pre.state, exiting.state.particles.position, np.array([-0.5, 0, 2.0])),
        H,
    )
    assert replay.segments == exiting.segments


def patch_for(vertices, triangles):
    from phymeo.geometry.material_seam import TranslatingMaterialPatch

    return TranslatingMaterialPatch(
        dict(enumerate(vertices)),
        ph.CollisionModel(
            [
                ph.CollisionShape(ph.CollisionKey(6, 2, j), "triangle", t)
                for j, t in enumerate(triangles)
            ]
        ),
    )


def test_topology_rejects_coincident_faces_vertex_touches_edge_ties_and_multiple_crossings():
    points = [[-1, 0, 0], [0, -1, 0], [0, 1, 0], [1, 0, 0]]
    simple = patch_for(points, [(0, 1, 2), (1, 3, 2)])
    for a, b in (((-0.5, 0, 0.1), (0, 0, 0.1)), ((0, -0.5, 0.1), (0, 0.5, 0.1))):
        with pytest.raises(ph.SolverFailure, match="endpoint|edge"):
            simple.trace(a, b)
    dup = patch_for(points[:3] + points[:3], [(0, 1, 2), (3, 4, 5)])
    with pytest.raises(ph.SolverFailure, match="overlap"):
        dup.trace([-0.5, 0, 0.1], [-0.4, 0, 0.1])
    with pytest.raises(ValueError, match="flipped"):
        patch_for(points, [(0, 1, 2), (1, 2, 3)])
    fan = patch_for(
        [[0, 0, 0], [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)],
    )
    with pytest.raises(ph.SolverFailure, match="vertex"):
        fan.trace([-0.5, -0.1, 0.25], [0.5, 0.1, 0.25])
    strip = patch_for(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4)],
    )
    with pytest.raises(ph.SolverFailure, match="multiple"):
        strip.trace([0.125, 0.25, 0.1], [1.875, 0.25, 0.1])


def test_rejected_physical_candidate_preserves_the_prior_face_and_episode():
    from phymeo.core.result import StepReport, StepResult

    op, R = model()
    before, good = slip_cross(op, R)

    class Reject:
        def step(self, model, state, inputs, dt):
            return StepResult(
                state.copy(), StepReport(False, reason="deliberate candidate rejection")
            )

    bad = op.step(before.state, load(R), H, particle_solver=Reject())
    assert (
        bad.state is before.state
        and not bad.report.accepted
        and bad.segments == ()
        and bad.ended is None
    )
    again = op.step(bad.state, load(R), H)
    assert again.segments == good.segments and again.state.history == good.state.history


@pytest.mark.tyx
def test_real_particle_steps_and_normal_operator_retain_the_same_material_crossing(
    tyx_mode, tmp_path
):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solver = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        op, R = model(rotated=True, dtype=np.float32, operator=layer)
        before, out = slip_cross(op, R, solver=solver, dtype=np.float32)
        assert before.report.backend == out.report.backend == "tyx:" + tyx_mode
        assert out.normal_backend == tyx_mode and layer.stats()["completed_launches"] == 2
        assert out.state.history.plastic_slip_m == pytest.approx(245 / 256)
        assert [s.t1 for s in out.segments] == pytest.approx([64 / 125, 1])
        assert sum(s.tangent_work_j for s in out.segments) == pytest.approx(-125 / 512)
        np.testing.assert_allclose(out.patch_constraint_angular_impulse_nms, R @ [0, 3 / 2048, 0])
        if tyx_mode == "cuda":
            assert layer.stats()["downloads"] == 2
        (tmp_path / "material-seam-device.json").write_text(
            json.dumps(
                dict(
                    backend=out.report.backend,
                    normal=layer.stats(),
                    chart="host-exact",
                    events=[s.t1 for s in out.segments],
                    per_face_impulses=[s.node_impulses_ns for s in out.segments],
                    plastic_m=out.state.history.plastic_slip_m,
                    cumulative_work_j=out.state.history.tangent_work_j,
                    position=out.state.particles.position.tolist(),
                    velocity=out.state.particles.velocity.tolist(),
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )

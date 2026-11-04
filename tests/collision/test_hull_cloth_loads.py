"""A finite solid tool transmits load over the complete closest feature of a moving cloth."""

import json
from contextlib import ExitStack
from dataclasses import asdict
from itertools import product

import numpy as np
import pytest

import phymeo as ph

H = 1 / 128
HALF = np.array([0.25, 0.25, 0.125])
IDS = (82, 13, 21, 35, 61, 56, 72, 3)
LOCAL = np.array(list(product((-1, 1), repeat=3))) * HALF
CLOTH = (17, 31, 7, 49)
AREAS = {"face": 1 / 8, "edge": 0.0, "corner": 0.0, "outside": 0.0}


def setup(kind="face", *, reverse=False, two=False, dtype=np.float64, friction=0.0, layer=None):
    tool = ph.CollisionShape(ph.CollisionKey(2, 5 if reverse else 1, 40), "hull", IDS)
    shifts = {
        "face": 0.0,
        "edge": 0.25,
        "corner": 0.25 + 3 / 128,
        "outside": 0.25 + 3 / 128,
        "far": 0.25 + 1 / 8,
    }
    s = shifts[kind]
    xy = [[s, -1.0], [s + 1.0, 0.0], [s, 1.0]]
    if kind == "corner":
        xy = [[s, 0.25], [s + 0.625, 0.25], [s, 1.0]]
    if two:
        assert kind == "face"
        xy.append([-1.0, 0.0])
    positions = np.array([[*point, 0.0] for point in xy], dtype=dtype)
    nodes = ph.ParticleState(positions, np.zeros_like(positions))
    body = ph.RigidBodyState(
        np.array([[0.0, 0.0, 5 / 32]], dtype=dtype),
        np.array([[0, 0, 0, 1]], dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
    )
    masses = np.array([1, 1.5, 2, 1.25][: len(nodes.position)], dtype=dtype)
    models = (
        ph.ParticleModel(1 / masses, np.zeros(3, dtype=dtype)),
        ph.RigidBodyModel(
            np.array([0.5], dtype=dtype), np.ones((1, 3), dtype=dtype), np.zeros(3, dtype=dtype)
        ),
    )
    shapes = [tool]
    bindings = {tool.key: ph.RigidAttachment(0, LOCAL)}
    laws = {}
    for i, rows in enumerate(((0, 1, 2), (0, 2, 3)) if two else ((0, 1, 2),)):
        face = ph.CollisionShape(
            ph.CollisionKey(2, 3, 25 - i * 12), "triangle", tuple(CLOTH[n] for n in rows)
        )
        shapes.append(face)
        bindings[face.key] = ph.ParticleAttachment(rows)
        laws[tool.key, face.key] = ph.NormalLayer(
            1 / 16, (96 if i == 0 else 32) if two else 128, friction=friction
        )
    model = ph.CollisionModel(shapes[::-1])
    system = ph.CollisionLoadSystem(model, bindings, laws, layer_operator=layer)
    return system, ph.CoupledState(nodes, body), models, tool.key


def ordered_site(point, hull_first):
    if hull_first:
        return point.first_point, point.second_point, point.first_weights, point.second_weights
    return point.second_point, point.first_point, point.second_weights, point.first_weights


@pytest.mark.parametrize(
    "kind,dimension,count,gap,force",
    [
        ("face", 2, 4, 1 / 32, 4.0),
        ("edge", 1, 2, 1 / 32, 4.0),
        ("outside", 1, 2, 5 / 128, 3.0),
        ("corner", 0, 1, 5 / 128, 3.0),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_finite_patch_separates_area_line_corner_and_side_gap(
    kind, dimension, count, gap, force, reverse, tmp_path
):
    system, state, _, tool = setup(kind, reverse=reverse)
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
        pair = work.last_result.pairs[0]
    patch = pair.witness.proximity_patch
    assert pair.status == "separated" and pair.witness.gap_m == pytest.approx(gap, abs=2e-13)
    assert patch.dimension == dimension and len(patch.points) == count
    assert patch.area_m2 == pytest.approx(AREAS[kind], abs=1e-13)
    (record,) = loads.contacts
    assert len(record.sites) == count and record.force_n == pytest.approx(force, abs=1e-13)
    assert sum(s.fraction for s in record.sites) == pytest.approx(1)
    direction = (
        np.array([0.0, 0, 1.0]) if kind in ("face", "edge") else np.array([-3 / 5, 0, 4 / 5])
    )
    np.testing.assert_allclose(loads.inputs.rigid.force[0], force * direction, atol=2e-13)
    coordinates = dict(zip(IDS, LOCAL + state.rigid.position[0], strict=True))
    positions = dict(zip(CLOTH[:3], state.particles.position, strict=True))
    for point in patch.points:
        xp, xq, wp, wq = ordered_site(point, pair.first == tool)
        np.testing.assert_allclose(xp, sum(w * coordinates[k] for k, w in wp), atol=1e-13)
        np.testing.assert_allclose(xq, sum(w * positions[k] for k, w in wq), atol=1e-13)
        np.testing.assert_allclose(np.asarray(xp) - xq, gap * direction, atol=1e-13)
    if kind == "face":
        assert {
            tuple(p[:2])
            for p in (ordered_site(site, pair.first == tool)[0] for site in patch.points)
        } == {(0, -0.25), (0, 0.25), (0.25, -0.25), (0.25, 0.25)}
        np.testing.assert_allclose(
            loads.inputs.particles.force[:, 2], [-1.75, -0.5, -1.75], atol=1e-13
        )
        np.testing.assert_allclose(loads.inputs.rigid.torque[0], [0, -0.5, 0], atol=1e-13)
    if kind == "corner":
        np.testing.assert_allclose(loads.inputs.particles.force[1:], 0, atol=1e-13)

    (tmp_path / "closest-patch.json").write_text(
        json.dumps(
            dict(
                case=kind,
                hull_first=pair.first == tool,
                gap_m=pair.witness.gap_m,
                patch=asdict(patch),
                fractions=[s.fraction for s in record.sites],
                rigid_force_n=loads.inputs.rigid.force.tolist(),
                rigid_torque_nm=loads.inputs.rigid.torque.tolist(),
                cloth_force_n=loads.inputs.particles.force.tolist(),
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def test_tangential_load_acts_at_cloth_material_points_and_respects_the_finite_boundary():
    system, state, _, _ = setup(friction=0.25)
    state.rigid.velocity[0, 0] = 0.3
    with ph.CollisionWorkspace(system.model) as work:
        load = system.sample(state, work)
    (contact,) = load.contacts
    np.testing.assert_allclose(load.inputs.rigid.force[0], [-1, 0, 4], atol=1e-13)
    np.testing.assert_allclose(load.inputs.rigid.torque[0], [0, -0.34375, 0], atol=1e-13)
    assert contact.power_w == pytest.approx(-0.3, abs=1e-13)
    for s in contact.sites:
        np.testing.assert_array_equal(s.application_point_first_m, s.application_point_second_m)
    far, state, _, _ = setup("far")
    with ph.CollisionWorkspace(far.model) as work:
        load = far.sample(state, work)
        assert not load.contacts and not np.any(load.inputs.particles.force)
        state.rigid.position[0, 0] += 0.625
        inside = far.sample(state, work)
        assert inside.contacts[0].force_n == pytest.approx(4, abs=1e-12)


def full_step(mode, root):
    dtype = np.float64 if mode == "host" else np.float32
    tol = 3e-11 if mode == "host" else 4e-6
    with ExitStack() as resources:
        layer = None
        if mode == "host":
            p, r = ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver()
        else:
            layer = resources.enter_context(ph.TyxNormalLayer(mode, cache_root=root / "cache"))
            p = resources.enter_context(ph.TyxSemiImplicitSolver(mode, cache_root=root / "cache"))
            r = resources.enter_context(
                ph.TyxRigidSemiImplicitSolver(mode, cache_root=root / "cache")
            )
        sys, state, models, tool = setup(dtype=dtype, layer=layer)
        x0, v0 = state.particles.position.copy(), state.particles.velocity.copy()
        # Unequal physical velocities tilt the actual finite cloth triangle in this step.
        v0[:, 2] = [-1 / 64, 1 / 32, 1 / 64]
        state.particles.velocity[:] = v0
        work = resources.enter_context(
            ph.CollisionWorkspace(sys.model, mode, cache_root=root / "cache")
        )
        if mode == "cuda":
            import tyx

            prior = tyx.runtime_stats().cuda_kernel_launches
        result = sys.step(state, ph.CoupledInputs.zeros(3, 1, dtype=dtype), models, (p, r), H, work)
        launches = tyx.runtime_stats().cuda_kernel_launches - prior if mode == "cuda" else None
        if mode == "cuda":
            assert launches >= 4 and layer.stats()["sites"] == 4
        (c,) = result.loads.contacts
        assert len(c.sites) == 4 and c.force_n == pytest.approx(4, abs=tol)
        dp = (result.state.particles.velocity - v0) / models[0].inverse_mass[:, None]
        np.testing.assert_allclose(
            dp, H * np.array([[0, 0, -1.75], [0, 0, -0.5], [0, 0, -1.75]]), atol=tol
        )
        dp_body = 2 * (result.state.rigid.velocity - state.rigid.velocity)
        np.testing.assert_allclose(dp_body, [[0, 0, 4 * H]], atol=tol)
        np.testing.assert_allclose(dp.sum(axis=0) + dp_body[0], 0, atol=tol)
        np.testing.assert_allclose(
            result.state.rigid.angular_velocity, [[0, -0.5 * H, 0]], atol=tol
        )
        x1 = result.state.particles.position
        assert abs(np.cross(x1[1] - x1[0], x1[2] - x1[0])[0]) > 1e-5
        assert abs(result.state.rigid.orientation[0, 1]) > 1e-6
        assert result.path.pairs[0].status == "separated"
        (root / "convex-cloth-step.json").write_text(
            json.dumps(
                dict(
                    backend=mode,
                    launches=launches,
                    layer=None if layer is None else layer.stats(),
                    width_m=1 / 16,
                    dt_s=H,
                    mapped_force_n=result.loads.inputs.particles.force.tolist(),
                    mapped_torque_nm=result.loads.inputs.rigid.torque.tolist(),
                    weights=[dict(site.material.second_weights) for site in c.sites],
                    x0_m=x0.tolist(),
                    x1_m=x1.tolist(),
                    v0_m_s=v0.tolist(),
                    v1_m_s=result.state.particles.velocity.tolist(),
                    rigid_linear_impulse_ns=dp_body.tolist(),
                    particle_impulse_ns=dp.tolist(),
                    omega=result.state.rigid.angular_velocity.tolist(),
                    quaternion=result.state.rigid.orientation.tolist(),
                    path_reason=result.path.pairs[0].reason,
                    end_gap_m=result.path.pairs[0].witness.gap_m,
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_box_face_reacts_at_the_finite_centroid_and_tilts_free_cloth(tmp_path):
    full_step("host", tmp_path)


@pytest.mark.tyx
def test_same_cloth_and_rigid_step_with_true_native_participants(tyx_mode, tmp_path):
    full_step(tyx_mode, tmp_path)


def test_two_connected_cloth_faces_share_rows_and_advance_with_internal_distance_constraints(
    tmp_path,
):
    system, state, models, _ = setup(two=True)
    state.particles.velocity[:, 2] = [1 / 64, -1 / 16, -1 / 32, 1 / 32]
    state.particles.velocity[1, 0] = 1 / 16
    state.rigid.velocity[0, 2] = -1 / 16
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]])
    rest = np.linalg.norm(
        state.particles.position[edges[:, 1]] - state.particles.position[edges[:, 0]], axis=1
    )
    structural = ph.DistanceConstraints(edges, rest, np.full(5, 0.02))
    p, r = ph.XPBDSolver(constraints=structural, tolerance=1e-5), ph.RigidSemiImplicitSolver()
    mass = 1 / models[0].inverse_mass[:, None]
    frames = []
    with ph.CollisionWorkspace(system.model) as work:
        for k in range(12):
            before = state
            out = system.step(before, ph.CoupledInputs.zeros(4, 1), models, (p, r), H, work)
            state = out.state
            if k == 0:
                assert sorted(c.force_n for c in out.loads.contacts) == pytest.approx([1, 3])
                assert all(len(c.sites) == 4 for c in out.loads.contacts)
                np.testing.assert_allclose(
                    out.loads.inputs.rigid.torque[0], [0, -0.25, 0], atol=1e-12
                )
            dp = ((state.particles.velocity - before.particles.velocity) * mass).sum(axis=0)
            dpb = 2 * (state.rigid.velocity[0] - before.rigid.velocity[0])
            np.testing.assert_allclose(dp + dpb, 0, atol=1e-10)
            np.testing.assert_allclose(dpb, H * out.loads.inputs.rigid.force[0], atol=1e-10)
            np.testing.assert_allclose(
                state.rigid.angular_velocity - before.rigid.angular_velocity,
                H * out.loads.inputs.rigid.torque,
                atol=1e-10,
            )
            frames.append(
                dict(
                    tick=k + 1,
                    x=state.particles.position.tolist(),
                    v=state.particles.velocity.tolist(),
                    body_x=state.rigid.position.tolist(),
                    body_q=state.rigid.orientation.tolist(),
                    force_by_face=[
                        (c.second.shape, c.force_n, len(c.sites)) for c in out.loads.contacts
                    ],
                    torque=out.loads.inputs.rigid.torque.tolist(),
                    momentum_residual=(dp + dpb).tolist(),
                )
            )
    assert np.max(np.abs(state.particles.position[:, 2])) > 0.001
    assert np.ptp(state.particles.position[:, 2]) > 0.002
    assert any(any(n != 4 for _, _, n in f["force_by_face"]) for f in frames[1:])
    (tmp_path / "two-cloth-faces.json").write_text(
        json.dumps(frames, allow_nan=False), encoding="utf8"
    )


def test_core_crossing_and_collapsed_cloth_fail_without_replacing_the_previous_acceptance():
    system, state, models, _ = setup()
    solvers = ph.SemiImplicitSolver(), ph.RigidSemiImplicitSolver()
    with ph.CollisionWorkspace(system.model) as work:
        good = system.step(state, ph.CoupledInputs.zeros(3, 1), models, solvers, H, work)
        published = work.last_result
        bad = ph.CoupledState(good.state.particles, good.state.rigid)
        bad.rigid.velocity[0, 2] = -24
        with pytest.raises(ph.SolverFailure, match="core path rejected"):
            system.step(bad, ph.CoupledInputs.zeros(3, 1), models, solvers, H, work)
        assert system.last_frame is good and work.last_result is published
        flat = ph.CoupledState(state.particles, state.rigid)
        flat.particles.position[1] = flat.particles.position[0]
        with pytest.raises(ph.SolverFailure, match="degenerate"):
            system.step(flat, ph.CoupledInputs.zeros(3, 1), models, solvers, H, work)
        assert system.last_frame is good and work.last_result is published


def test_offcenter_tetrahedral_tool_loads_only_its_three_real_support_vertices(tmp_path):
    base, state, _, key = setup(reverse=True)
    points = np.array(
        [
            [1 / 8, -1 / 4, -1 / 8],
            [3 / 8, -1 / 4, -1 / 8],
            [1 / 8, 1 / 4, -1 / 8],
            [7 / 32, -1 / 16, 1 / 4],
        ]
    )
    tool = ph.CollisionShape(key, "hull", (65, 9, 41, 12))
    cloth = next(s for s in base.model.shapes if s.key != key)
    binds = dict(base.bindings)
    binds[key] = ph.RigidAttachment(0, points)
    sys = ph.CollisionLoadSystem(ph.CollisionModel([tool, cloth]), binds, base.layers)
    rotation = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])
    state.rigid.position[:] = state.rigid.position @ rotation.T
    state.rigid.orientation[0] = [0.5, 0.5, 0.5, 0.5]
    state.particles.position[:] = state.particles.position @ rotation.T
    with ph.CollisionWorkspace(sys.model) as work:
        loads = sys.sample(state, work)
        pair = work.last_result.pairs[0]
    patch = pair.witness.proximity_patch
    assert patch.dimension == 2 and patch.area_m2 == pytest.approx(1 / 16)
    assert set(patch.second_support) == {65, 9, 41}
    assert {n for p in patch.points for n, w in p.second_weights if w} == {65, 9, 41}
    assert len(loads.contacts[0].sites) == 3 and loads.contacts[0].force_n == pytest.approx(4)
    np.testing.assert_allclose(loads.inputs.rigid.force[0], rotation @ [0, 0, 4], atol=2e-13)
    np.testing.assert_allclose(
        loads.inputs.rigid.torque[0], rotation @ [-1 / 3, -5 / 6, 0], atol=2e-13
    )
    (tmp_path / "tetrahedral-tool.json").write_text(
        json.dumps(
            dict(
                patch=asdict(patch),
                world_force_n=loads.inputs.rigid.force.tolist(),
                world_torque_nm=loads.inputs.rigid.torque.tolist(),
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )

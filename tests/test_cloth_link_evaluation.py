"""One-evaluation reuse must preserve actual geometry, material addresses and forces."""

import importlib.util
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_cloth_link_contact import make_scene

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.constraints.robot_contact import LinkCapsule
from phymeo.core.scene import fingerprint
from phymeo.dynamics.robot_contact import RobotContactSystem
from phymeo.solvers.cloth_link_contact import ClothLinkContacts


def assert_same(actual, expected):
    """Compare every published field without rounding or dropping material metadata."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, Mapping):
        assert tuple(actual) == tuple(expected)
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_same(a, b)
    else:
        assert actual == expected


def contacts(program, *, local_reuse):
    return ClothLinkContacts(
        program.cloth_domain, program.surfaces, program.cloth_links.areas,
        program.robot_contact, program.cloth_links.law, program.cloth_links.operator,
        local_reuse=local_reuse,
    )


def test_local_reuse_parameter_preserves_real_two_newton_split_face_load():
    # Existing public geometry and force sampling must work before the new API RED.
    program, state, _ = make_scene(split=True)
    geometry = program._link_geometry(state)
    expected = program.cloth_links.evaluate(state, geometry)
    np.testing.assert_allclose(expected[0].sum(0), [0, 0, -2], rtol=0, atol=1e-13)
    np.testing.assert_allclose(expected[1]["cloth_link_efforts"]["arm"], [2], rtol=0, atol=1e-13)
    assert len(expected[1]["cloth_links"]) == 2
    assert all(row["normal_force_n"] > 0 for row in expected[1]["cloth_links"])
    # The frozen first RED is unexpected keyword local_reuse on this existing class.
    reused = contacts(program, local_reuse=True)
    assert_same(reused.evaluate(state, geometry), expected)


_DEFAULT = object()
H = 1 / 256
SETTINGS = ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True)
CYCLE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
SHIFT = np.array([3.0, -2.0, 0.5])


def local_scene(*, faces=2, multiple=False, sparse=False, reverse=False, moving=False,
                transformed=False, mass_scale=1.0, local_reuse=_DEFAULT, operator=None,
                enabled=True):
    """A finite 1 m2 strip with physical shared nodes, independent of product output."""
    columns = faces // 2
    grid = np.array([(x, y, 0.0) for x in np.linspace(-1.0, 1.0, columns + 1)
                     for y in (-0.25, 0.25)])
    node_ids = (3 * np.arange(len(grid))[::-1] + 2) if sparse else np.arange(len(grid))
    x = np.tile([4.0, 4.0, -2.0], (int(node_ids.max()) + 1, 1))
    x[node_ids] = grid
    v = np.zeros_like(x)
    if moving:
        v[node_ids] = np.column_stack((1 / 8 + grid[:, 1] / 16,
                                      -grid[:, 0] / 32,
                                      grid[:, 0] / 64 + grid[:, 1] / 32))
    triangles = []
    for column in range(columns):
        a = 2 * column
        triangles.extend((tuple(node_ids[[a, a + 2, a + 3]]),
                          tuple(node_ids[[a, a + 3, a + 1]])))
    named = [(f"face-{i}", ids) for i, ids in enumerate(triangles)]
    if reverse:
        named = [(name, (ids[1], ids[2], ids[0])) for name, ids in reversed(named)]
    parents = np.array([-1, 0] if multiple else [-1])
    count = len(parents)
    offsets = np.array([[0.0, 0.0, 3 / 16], [1 / 8, 0.0, 0.0]][:count])
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
    if transformed:
        x = x @ CYCLE.T + SHIFT
        v = v @ CYCLE.T
        offsets[0] = offsets[0] @ CYCLE.T + SHIFT
        orientations[0] = [0.5, 0.5, 0.5, 0.5]
    cloth = ph.ParticleModel(np.full(len(x), 1 / mass_scale), np.zeros(3))
    robot = ph.RobotModel(
        parents, ("prismatic", "revolute")[:count], offsets, orientations,
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]][:count]),
        np.array([0.5, 0.75][:count]) * mass_scale,
        np.array([[0.0, 0.0, 0.0], [1 / 16, 0.0, 0.0]][:count]),
        np.ones((count, 3)) * mass_scale, np.zeros(3),
    )
    links = [LinkCapsule("beam", "arm", 0,
                         np.array([-0.75, -1 / 16 if multiple else 0.0, 0.0]),
                         np.array([0.75, -1 / 16 if multiple else 0.0, 0.0]), 1 / 16)]
    if multiple:
        links.append(LinkCapsule("tip", "arm", 1,
                                 np.array([-0.75, 1 / 16, 0.0]),
                                 np.array([0.75, 1 / 16, 0.0]), 1 / 16))
    contact = RobotContactSystem({"arm": robot}, capsules=tuple(links))
    scene = ph.SceneModel((ph.DomainBinding("cloth", "particles", cloth),
                           ph.DomainBinding("arm", "robot", robot)))
    surfaces = tuple(ph.TrianglePrismSurface(name, "cloth", ids, half_thickness_m=1 / 32)
                     for name, ids in named)
    edges = np.array(sorted({tuple(sorted((a, b))) for _, (i, j, k) in named
                            for a, b in ((i, j), (j, k), (k, i))}))
    rest = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    material = ClothMaterial(distance=ph.DistanceConstraints(edges, rest, np.full(len(edges), 1 / 64)))
    law = ph.ClothLinkMaterial(1 / 8, 64.0, 0.25 if moving else 0.0,
                               0.375 if moving else 0.0, 1 / 8)
    options = {} if local_reuse is _DEFAULT else {"cloth_link_local_reuse": local_reuse}
    program = ph.CompliantClothContactProgram(
        scene, "cloth", material, surfaces, x, (), robot_contact=contact,
        cloth_link_material=law if enabled else None, cloth_link_operator=operator,
        max_penetration_m=1 / 32, **options,
    )
    speed = np.array([1 / 32, -1 / 8][:count]) if moving else np.zeros(count)
    state = ph.SceneState({"cloth": ph.ParticleState(x, v),
                           "arm": ph.RobotState(np.zeros(count), speed)})
    inputs = ph.SceneInputs({"cloth": ph.StepInputs.zeros(len(x)),
                            "arm": ph.RobotInputs.zeros(count)})
    return program, state, inputs


def clone_state(state):
    cloth, arm = state.domains["cloth"], state.domains["arm"]
    return ph.SceneState({"cloth": ph.ParticleState(cloth.position.copy(), cloth.velocity.copy()),
                           "arm": ph.RobotState(arm.q.copy(), arm.v.copy())},
                          history={key: value.copy() for key, value in state.history.items()})


def canonical_frame(frame):
    subs = tuple(replace(sub, timings={key: 0.0 for key in sub.timings})
                 for sub in frame.report.substeps)
    return replace(frame, report=replace(frame.report, substeps=subs))


@pytest.mark.parametrize("faces", [2, 8])
def test_public_program_keeps_four_step_full_output_and_real_deformation(faces):
    old, first, controls = local_scene(faces=faces, multiple=True, sparse=True, reverse=True, moving=True)
    new, second, other = local_scene(faces=faces, multiple=True, sparse=True, reverse=True,
                                     moving=True, local_reuse=True)
    initial = fingerprint(first)
    with old.stepper(SETTINGS) as a, new.stepper(SETTINGS) as b:
        for _ in range(4):
            left, right = a.step(first, controls, H), b.step(second, other, H)
            assert left.report.accepted and right.report.accepted
            assert fingerprint(canonical_frame(left)) == fingerprint(canonical_frame(right))
            first, second = left.state, right.state
    assert fingerprint(second) != initial
    assert np.linalg.norm(second.domains["arm"].q) > 0
    assert np.linalg.norm(second.domains["cloth"].velocity) > 0
    tri = second.domains["cloth"].position[new.faces]
    ref_normal = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    assert np.max(np.linalg.norm(normals - ref_normal, axis=1)) > 1e-9


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_nodes_ancestors_and_material_order_keep_force_torque_and_work(reverse):
    program, state, _ = local_scene(faces=8, multiple=True, sparse=True, reverse=reverse,
                                    moving=True, local_reuse=True)
    geo = program._link_geometry(state)
    actual = program.cloth_links.evaluate(state, geo)
    assert_same(actual, contacts(program, local_reuse=False).evaluate(state, geo))
    nodal, body = actual
    expected_nodal = np.zeros_like(nodal)
    expected_effort = np.zeros(2)
    link_moment = np.zeros(3)
    law = program.cloth_links.law
    area = dict(zip(program.cloth_links.face_names, program.cloth_links.areas, strict=True))
    v = state.domains["arm"].v
    assert {row["link"] for row in body["cloth_links"]} == {"beam", "tip"}
    for row in body["cloth_links"]:
        assert row["robot"] == "arm"
        for site in row["sites"]:
            ids, weights = zip(*site["face_weights"], strict=True)
            assert tuple(ids) == tuple(sorted(row["face_nodes"]))
            point = sum(w * state.domains["cloth"].position[i] for i, w in zip(ids, weights, strict=True))
            np.testing.assert_allclose(point, site["point_m"], atol=1e-13, rtol=0)
            s = site["axis_parameter"]
            axis_point = (1 - s) * row["axis_ends_m"][0] + s * row["axis_ends_m"][1]
            normal = point - axis_point
            axis_distance = np.linalg.norm(normal)
            normal /= axis_distance
            np.testing.assert_allclose(normal, row["normal"], atol=1e-13, rtol=0)
            assert axis_distance - 1 / 16 - 1 / 32 == pytest.approx(row["gap_m"], abs=1e-13)
            robot_velocity = np.array([0.0, 0.0, v[0]])
            pivot = np.array([1 / 8, 0.0, 3 / 16])
            if row["link"] == "tip":
                robot_velocity += np.cross([0.0, v[1], 0.0], point - pivot)
            relative = np.asarray(weights) @ state.domains["cloth"].velocity[list(ids)] - robot_velocity
            np.testing.assert_allclose(relative, site["relative_velocity_m_s"], atol=1e-14, rtol=0)
            rate = normal @ relative
            fn = 64 * area[row["face"]] * max(0, 1 / 8 - row["gap_m"]) * max(0, 1 - .25 * rate) / len(row["sites"])
            tangent = relative - rate * normal
            force = fn * normal - law.friction * fn * tangent / max(law.slip_speed_m_s, np.linalg.norm(tangent))
            np.testing.assert_allclose(force, site["force_on_cloth_n"], atol=2e-13, rtol=1e-12)
            np.testing.assert_array_equal(site["force_on_link_n"], -site["force_on_cloth_n"])
            for node, weight in zip(ids, weights, strict=True):
                expected_nodal[node] += weight * site["force_on_cloth_n"]
            reaction = site["force_on_link_n"]
            expected_effort[0] += reaction[2]
            if row["link"] == "tip":
                expected_effort[1] += np.cross(point - pivot, reaction)[1]
            link_moment += np.cross(point, reaction)
    np.testing.assert_allclose(nodal, expected_nodal, atol=1e-13, rtol=0)
    np.testing.assert_allclose(body["cloth_link_efforts"]["arm"], expected_effort, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.cross(state.domains["cloth"].position, nodal).sum(0) + link_moment,
                               np.zeros(3), atol=2e-12, rtol=0)
    power = np.sum(nodal * state.domains["cloth"].velocity) + expected_effort @ v
    assert power == pytest.approx(sum(row["power_w"] for row in body["cloth_links"]), abs=2e-13)
    inactive = sorted(set(range(len(nodal))) - set(program.faces.ravel()))
    np.testing.assert_array_equal(nodal[inactive], np.zeros((len(inactive), 3)))


def test_changed_positions_velocities_and_singleton_sites_are_not_cached_across_calls():
    program, state, _ = local_scene(faces=8, multiple=True, sparse=True, moving=True, local_reuse=True)
    first = program.cloth_links.evaluate(state, program._link_geometry(state), phase="entry")
    saved = deepcopy(first)
    assert any(len(row["sites"]) == 2 for row in first[1]["cloth_links"])
    nodes = np.unique(program.faces)
    x = state.domains["cloth"].position
    x[nodes, 2] += x[nodes, 0] / 64 + x[nodes, 0] * x[nodes, 1] / 128
    state.domains["cloth"].velocity[nodes] += [1 / 32, -1 / 64, 1 / 128]
    state.domains["arm"].q[:] = [1 / 128, 1 / 16]
    state.domains["arm"].v[:] = [-1 / 16, 1 / 4]
    geometry = program._link_geometry(state)
    second = program.cloth_links.evaluate(state, geometry, phase="midpoint")
    assert_same(second, contacts(program, local_reuse=False).evaluate(state, geometry, phase="midpoint"))
    assert_same(second, contacts(program, local_reuse=True).evaluate(state, geometry, phase="midpoint"))
    assert_same(first, saved)
    assert any(len(row["sites"]) == 1 for row in second[1]["cloth_links"])
    assert fingerprint(second) != fingerprint(first)
    old_normals = {row["face"]: row["normal"] for row in first[1]["cloth_links"]}
    assert any(np.linalg.norm(row["normal"] - old_normals[row["face"]]) > 1e-7
               for row in second[1]["cloth_links"])


def test_same_local_ids_in_independent_instances_do_not_share_current_inputs():
    a, sa, _ = local_scene(faces=2, local_reuse=True)
    b, sb, _ = local_scene(faces=2, local_reuse=True)
    sb.domains["arm"].q[0] = 1 / 64
    first = a.cloth_links.evaluate(sa, a._link_geometry(sa))
    second = b.cloth_links.evaluate(sb, b._link_geometry(sb))
    assert sum(row["normal_force_n"] for row in first[1]["cloth_links"]) == pytest.approx(2.0)
    assert sum(row["normal_force_n"] for row in second[1]["cloth_links"]) == pytest.approx(1.0)
    assert_same(a.cloth_links.evaluate(sa, a._link_geometry(sa)), first)
    assert_same(second, contacts(b, local_reuse=False).evaluate(sb, b._link_geometry(sb)))


def test_global_rigid_coordinate_change_preserves_material_ids_and_joint_work():
    a, sa, _ = local_scene(faces=8, multiple=True, sparse=True, moving=True, local_reuse=True)
    b, sb, _ = local_scene(faces=8, multiple=True, sparse=True, moving=True,
                           transformed=True, local_reuse=True)
    fa, da = a.cloth_links.evaluate(sa, a._link_geometry(sa))
    fb, db = b.cloth_links.evaluate(sb, b._link_geometry(sb))
    np.testing.assert_allclose(fb, fa @ CYCLE.T, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(db["cloth_link_efforts"]["arm"], da["cloth_link_efforts"]["arm"],
                               atol=2e-12, rtol=1e-12)
    for left, right in zip(da["cloth_links"], db["cloth_links"], strict=True):
        assert (left["robot"], left["link"], left["face"], left["face_nodes"]) == (
            right["robot"], right["link"], right["face"], right["face_nodes"])
        assert right["gap_m"] == pytest.approx(left["gap_m"], abs=1e-13)
        assert right["power_w"] == pytest.approx(left["power_w"], abs=1e-12)
        np.testing.assert_allclose(right["normal"], left["normal"] @ CYCLE.T, atol=1e-13, rtol=0)
        for u, v in zip(left["sites"], right["sites"], strict=True):
            assert u["face_weights"] == v["face_weights"]
            assert u["axis_parameter"] == v["axis_parameter"]
            np.testing.assert_allclose(v["point_m"], u["point_m"] @ CYCLE.T + SHIFT, atol=1e-13, rtol=0)
            np.testing.assert_allclose(v["force_on_cloth_n"], u["force_on_cloth_n"] @ CYCLE.T,
                                       atol=2e-12, rtol=1e-12)


@pytest.mark.parametrize("mass_scale", [1.0, 2.0])
def test_mass_changes_actual_acceleration_without_reusing_an_old_mechanical_state(mass_scale):
    p, state, inputs = local_scene(faces=2, mass_scale=mass_scale, local_reuse=True)
    acceleration, _, _, body = p.accelerations(state, inputs)
    assert sum(row["normal_force_n"] for row in body["cloth_links"]) == pytest.approx(2.0)
    np.testing.assert_allclose(acceleration["cloth"].sum(0), [0, 0, -2 / mass_scale], atol=1e-13, rtol=0)
    np.testing.assert_allclose(acceleration["arm"], [4 / mass_scale], atol=1e-13, rtol=0)
    with p.stepper(SETTINGS) as stepper:
        result = stepper.step(state, inputs, H)
    assert result.report.accepted
    total = mass_scale * result.state.domains["cloth"].velocity.sum(0)
    total += np.array([0, 0, .5 * mass_scale * result.state.domains["arm"].v[0]])
    np.testing.assert_allclose(total, 0, atol=2e-12, rtol=0)


def test_default_and_explicit_false_do_not_enter_the_new_helper(monkeypatch):
    from phymeo.solvers._cloth_link_evaluation import ClothLinkEvaluation

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default path entered local-reuse helper")

    monkeypatch.setattr(ClothLinkEvaluation, "__init__", forbidden)
    a, state, inputs = local_scene()
    b, other, more = local_scene(local_reuse=False)
    assert_same(a.cloth_links.evaluate(state, a._link_geometry(state)),
                b.cloth_links.evaluate(other, b._link_geometry(other)))
    with a.stepper(SETTINGS) as first, b.stepper(SETTINGS) as second:
        left, right = first.step(state, inputs, H), second.step(other, more, H)
    assert left.report.accepted and right.report.accepted
    assert fingerprint(canonical_frame(left)) == fingerprint(canonical_frame(right))


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_opt_in_requires_a_boolean(value):
    p, _, _ = local_scene()
    with pytest.raises((TypeError, ValueError), match="bool|reuse"):
        contacts(p, local_reuse=value)
    with pytest.raises((TypeError, ValueError), match="bool|reuse"):
        local_scene(local_reuse=value)


def test_reuse_without_a_configured_cloth_link_law_is_rejected():
    with pytest.raises(ValueError, match="cloth|material|reuse"):
        local_scene(local_reuse=True, enabled=False)


def test_one_call_prepares_each_motion_once_then_restarts_on_the_next_call(monkeypatch):
    from phymeo.solvers import _cloth_link_evaluation as module

    p, state, _ = local_scene(faces=8, multiple=True, sparse=True, local_reuse=True)
    original = module.CollisionMotion
    counts = []

    def observed(points):
        counts.append(len(points))
        return original(points)

    monkeypatch.setattr(module, "CollisionMotion", observed)
    first = p.cloth_links.evaluate(state, p._link_geometry(state))
    assert counts.count(2) == len(p.cloth_links.links)
    assert counts.count(3) == len(p.surfaces)
    before = counts.copy()
    assert_same(p.cloth_links.evaluate(state, p._link_geometry(state)), first)
    assert counts == before + before


def test_local_snapshot_owns_positions_velocities_and_com_inputs():
    from phymeo.solvers._cloth_link_evaluation import ClothLinkEvaluation
    from phymeo.geometry._capsule_triangle_sweep import capsule_material_sites

    p, state, _ = local_scene(faces=2, sparse=True, moving=True)
    geo = {key: tuple(np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                      for value in row) for key, row in p._link_geometry(state).items()}
    work = ClothLinkEvaluation(p.cloth_links, state, geo)
    witness = work.sample(0, 0)
    site = capsule_material_sites(p.cloth_links.capsules[0], witness, capsule_first=True)[0]
    ids, weights = zip(*site.second_weights, strict=True)
    original = work.relative_velocity(0, 0, ids, np.array(weights), np.array(site.second_point))
    triangles, ends = work.triangles.copy(), work.endpoints(0).copy()
    state.domains["cloth"].position[:] += [2.0, 3.0, 4.0]
    state.domains["cloth"].velocity[:] *= 7
    state.domains["arm"].v[:] = 9
    geo["arm"][2][:] += 10
    geo["arm"][3][:] *= 3
    assert_same(work.relative_velocity(0, 0, ids, np.array(weights), np.array(site.second_point)), original)
    np.testing.assert_array_equal(work.triangles, triangles)
    np.testing.assert_array_equal(work.endpoints(0), ends)
    assert work.sample(0, 0) == witness
    with pytest.raises(ValueError):
        work.triangles[0, 0, 0] = 99
    with pytest.raises(ValueError):
        work.endpoints(0)[0, 0] = 99
    with pytest.raises(ValueError, match="material IDs"):
        work.relative_velocity(0, 0, tuple(reversed(ids)), np.array(weights), np.array(site.second_point))


@pytest.mark.parametrize("kind", ["axis", "degenerate"])
def test_geometry_failure_does_not_poison_next_evaluation(kind):
    p, state, _ = local_scene(faces=2, local_reuse=True)
    valid = p.cloth_links.evaluate(state, p._link_geometry(state))
    bad = clone_state(state)
    if kind == "axis":
        bad.domains["arm"].q[0] = -3 / 16
    else:
        bad.domains["cloth"].position[np.unique(p.faces)] = [0.0, 0.0, 0.0]
    before = fingerprint(bad)
    with pytest.raises(ph.SolverFailure, match="axis|degenerate"):
        p.cloth_links.evaluate(bad, p._link_geometry(bad))
    assert fingerprint(bad) == before
    assert_same(p.cloth_links.evaluate(state, p._link_geometry(state)), valid)


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "negative_normal"])
def test_late_bad_law_rolls_back_and_retry_matches_fresh_program(fault, monkeypatch):
    from phymeo.solvers import cloth_link_contact as module

    p, state, inputs = local_scene(faces=8, sparse=True, moving=True, local_reuse=True)
    original = module._host_law
    calls = 0

    def wrong_midpoint(values):
        nonlocal calls
        calls += 1
        answer = original(values)
        if calls != 2:
            return answer
        if fault == "shape":
            return answer[:, :4]
        answer = answer.copy()
        answer[0, 1] = np.nan if fault == "nonfinite" else -1.0
        return answer

    before = fingerprint(state)
    with p.stepper(SETTINGS) as stepper:
        with monkeypatch.context() as patcher:
            patcher.setattr(module, "_host_law", wrong_midpoint)
            failed = stepper.step(state, inputs, H)
        assert not failed.report.accepted
        assert "cloth-link" in failed.report.reason
        assert fingerprint(failed.state) == before == fingerprint(state)
        retry = stepper.step(state, inputs, H)
    fresh, initial, controls = local_scene(faces=8, sparse=True, moving=True, local_reuse=True)
    with fresh.stepper(SETTINGS) as stepper:
        expected = stepper.step(initial, controls, H)
    assert retry.report.accepted and expected.report.accepted
    assert fingerprint(canonical_frame(retry)) == fingerprint(canonical_frame(expected))


@pytest.mark.tyx
@pytest.mark.parametrize("faces", [2, 8])
def test_actual_tyx_law_keeps_complete_step_output_and_native_counts(faces, tyx_mode, tmp_path):
    with ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "native") as device:
        old, first, inputs = local_scene(faces=faces, multiple=True, sparse=True,
                                         moving=True, operator=device)
        new, second, controls = local_scene(faces=faces, multiple=True, sparse=True,
                                             moving=True, operator=device, local_reuse=True)
        before = device.stats()
        with old.stepper(SETTINGS) as stepper:
            a = stepper.step(first, inputs, H)
        middle = device.stats()
        with new.stepper(SETTINGS) as stepper:
            b = stepper.step(second, controls, H)
        after = device.stats()
        assert a.report.accepted and b.report.accepted
        assert fingerprint(canonical_frame(a)) == fingerprint(canonical_frame(b))
        fields = ("completed_launches", "sites", "uploads", "downloads", "uploaded_bytes", "downloaded_bytes")
        left = {key: middle[key] - before[key] for key in fields}
        right = {key: after[key] - middle[key] for key in fields}
        assert left == right and left["completed_launches"] == 3 and left["sites"] > 0
        assert device.stats()["backend"] == tyx_mode
        if tyx_mode == "cuda":
            assert left["uploads"] == left["downloads"] == 3
            assert left["uploaded_bytes"] == 3 * 6656
            assert left["downloaded_bytes"] == 3 * 2560
        else:
            assert left["uploads"] == left["downloads"] == 0


def benchmark_module():
    name = "_cloth_link_evaluation_benchmark_tests"
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cloth_link_evaluation.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protocol_receipts():
    # Synthetic validator inputs only: these are never saved as native measurements.
    records = []
    for worlds in (1, 32):
        for pair, order in ((1, ("scalar", "reuse")), (2, ("reuse", "scalar")),
                            (3, ("scalar", "reuse"))):
            for lane in order:
                clock = 10000 * (len(records) + 1)
                samples = [600, 300, 450] if lane == "scalar" else [400, 200, 300]
                windows = []
                for index, (phase, elapsed) in enumerate(zip(
                        ("cold", "warm", "formal", "formal", "formal"), (900, 500, *samples), strict=True)):
                    windows.append(dict(phase=phase, index=max(0, index - 2), whole_ns=elapsed,
                                         complete_output_sha256="b" * 64, counter_audit="passed",
                                         counter_delta={"sites": 104 * worlds},
                                         phase_sites=[16 * worlds] + [8 * worlds] * 11,
                                         accepted_world_steps=4 * worlds, instrumented=False))
                records.append(dict(status="complete", source={"head": "a" * 40, "files": {"fixed": "f" * 64}},
                                     source_verified_after=True, pair=pair, implementation=lane,
                                     config=dict(worlds=worlds, faces=8, steps=4, warm_windows=1,
                                                 formal_windows=3, mode="cuda"),
                                     input_manifest_sha256="c" * 64, complete_output_sha256="b" * 64,
                                     actual_device={"real_cuda": True, "device_name": "synthetic protocol device"},
                                     run_start_unix_ns=clock, run_end_unix_ns=clock + 9000,
                                     windows=windows))
    return records


def test_paired_receipt_protocol_requires_complete_same_output_and_keeps_defaults():
    result = benchmark_module().compare_profiles(protocol_receipts(), expected_head="a" * 40)
    assert result["status"] == "complete" and result["default_changed"] is False
    assert len(result["pairs"]) == 6
    assert [row["order"] for row in result["pairs"][:3]] == [
        ["scalar", "reuse"], ["reuse", "scalar"], ["scalar", "reuse"]]
    assert all(row["p50_ratio"] == pytest.approx(1.5) for row in result["pairs"])


@pytest.mark.parametrize("fault", ["output", "source", "counter", "overlap", "missing", "instrumented"])
def test_paired_receipt_protocol_rejects_incomparable_or_incomplete_lanes(fault):
    data = protocol_receipts()
    if fault == "output":
        data[1]["complete_output_sha256"] = "d" * 64
    elif fault == "source":
        data[1]["source"]["files"]["fixed"] = "e" * 64
    elif fault == "counter":
        data[1]["windows"][2]["counter_delta"]["sites"] += 1
    elif fault == "overlap":
        data[1]["run_start_unix_ns"] = data[0]["run_start_unix_ns"]
    elif fault == "missing":
        data.pop()
    else:
        data[1]["windows"][2]["instrumented"] = True
    with pytest.raises(ValueError):
        benchmark_module().compare_profiles(data, expected_head="a" * 40)

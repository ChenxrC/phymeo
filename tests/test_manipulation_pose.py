"""Independent native/analytic pose-audit oracles, not manipulation success."""

from itertools import product

import numpy as np
import pytest

from phymeo.control.manipulation_pose import audit_native_contact_poses


@pytest.fixture
def mj():
    return pytest.importorskip("mujoco")


def model_fixture(mj, kind="box", *, left_pad_pos=".01 .002 -.005", support_pos=".02 -.03 .1"):
    sizes = {
        "box": ".025 .018 .035",
        "sphere": ".025",
        "capsule": ".015 .04",
        "ellipsoid": ".02 .03 .04",
    }
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/>'
        '<worldbody><geom name="support" type="plane" size="1 1 .1" '
        f'pos="{support_pos}" euler=".2 -.3 .15"/>'
        '<body name="object" pos=".12 -.04 .8" quat=".9238795325 0 0 .3826834324">'
        '<freejoint name="object-free"/>'
        f'<geom name="object-shape" type="{kind}" size="{sizes[kind]}" '
        'pos=".02 -.01 .03" euler=".3 -.2 .4" mass=".2"/></body>'
        '<body name="arm" pos=".1 0 .7"><joint name="hinge" type="hinge" axis="0 1 0" '
        'limited="true" range="-1.5 1.5"/>'
        '<geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".2"/>'
        '<body name="slider" pos=".2 0 0"><joint name="slide" type="slide" axis="1 0 0" '
        'limited="true" range="-.1 .1"/><geom size=".025" mass=".1"/>'
        '<body name="tool" pos=".15 0 .04"><geom name="palm" size=".025" mass=".1"/>'
        '<body name="left" pos="0 .06 -.05"><joint name="left-finger" type="slide" '
        'axis="0 -1 0"/><geom name="left-pad" type="box" size=".045 .025 .04" '
        f'pos="{left_pad_pos}" euler=".1 .25 .2" mass=".04"/></body>'
        '<body name="right" pos="0 -.06 -.05"><joint name="right-finger" type="slide" '
        'axis="0 1 0"/><geom name="right-pad" type="box" size=".045 .025 .04" '
        'pos="-.01 .003 .005" euler="-.15 .2 -.1" mass=".04"/></body>'
        "</body></body></body>"
        '<body name="other-object" pos="-.2 .3 .6" quat=".9659258263 .2588190451 0 0">'
        '<freejoint name="other-free"/><geom type="sphere" size=".03" mass=".1"/>'
        "</body></worldbody></mujoco>"
    )


def binary(mj, model):
    data = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, None, data)
    return data.tobytes()


def native(mj, model, q):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_forward(model, data)
    return data


def declaration(mj, model):
    q = model.qpos0.copy()
    for name, value in (
        ("hinge", 0.15),
        ("slide", 0.02),
        ("left-finger", 0.008),
        ("right-finger", 0.011),
    ):
        q[model.jnt_qposadr[model.joint(name).id]] = value
    local = np.array((0.01, 0.02, -0.03))
    data = native(mj, model, q)
    body = model.body("tool").id
    target = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
    options = dict(
        body_name="tool",
        local_point=local,
        pad_geom_names=("left-pad", "right-pad"),
        support_geom_name="support",
        object_geom_names=("object-shape",),
    )
    return q, target, options


def forbidden(*args, **kwargs):
    raise AssertionError("pose audit must not integrate, forward-solve or inverse-solve physics")


def corner_oracle(data, model, geom):
    matrix, center, size = (
        data.geom_xmat[geom].reshape(3, 3),
        data.geom_xpos[geom],
        model.geom_size[geom],
    )
    return np.array(
        [
            center
            + sum((sign[axis] * size[axis] * matrix[:, axis] for axis in range(3)), np.zeros(3))
            for sign in product((-1, 1), repeat=3)
        ]
    )


def interval_oracle(data, model, pad, obj, kind):
    # BOX oracle enumerates transformed vertices, not the production abs(R) @ halfsize formula.
    axes, origin = data.geom_xmat[pad].reshape(3, 3), data.geom_xpos[pad]
    center, rotation, size = (
        data.geom_xpos[obj],
        data.geom_xmat[obj].reshape(3, 3),
        model.geom_size[obj],
    )
    if kind == "box":
        vertices, radius = corner_oracle(data, model, obj), 0.0
    elif kind == "capsule":
        vertices = np.array((center - size[1] * rotation[:, 2], center + size[1] * rotation[:, 2]))
        radius = size[0]
    else:
        vertices, radius = np.array((center,)), size[0]
    result = []
    for axis in range(3):
        values = [float(np.dot(vertex - origin, axes[:, axis])) for vertex in vertices]
        result.append((min(values) - radius, max(values) + radius))
    return np.asarray(result)


@pytest.mark.parametrize("kind", ("box", "sphere", "capsule"))
def test_rotated_pad_tilted_support_and_local_object_transform(mj, kind, monkeypatch):
    model = model_fixture(mj, kind)
    q, target, options = declaration(mj, model)
    oracle = native(mj, model, q)
    model_before, seed_before = binary(mj, model), q.tobytes()
    with monkeypatch.context() as guard:
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, forbidden)
        (candidate,) = audit_native_contact_poses(model, target, [q], ("slide", "hinge"), **options)
    assert candidate.ik.converged and candidate.ik.iterations == 0
    assert binary(mj, model) == model_before and q.tobytes() == seed_before
    plane = model.geom("support").id
    normal = oracle.geom_xmat[plane].reshape(3, 3)[:, 2]
    assert not np.allclose(normal, (0, 0, 1))
    np.testing.assert_allclose(
        candidate.support_origin, oracle.geom_xpos[plane], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(candidate.support_normal, normal, rtol=0, atol=1e-12)
    obj = model.geom("object-shape").id
    assert not np.array_equal(oracle.geom_xpos[obj], oracle.xpos[model.body("object").id])
    for pad in candidate.pads:
        geom = model.geom(pad.geom_name).id
        corners = corner_oracle(oracle, model, geom)
        np.testing.assert_allclose(pad.world_corners, corners, rtol=0, atol=1e-12)
        distances = [float(np.dot(corner - oracle.geom_xpos[plane], normal)) for corner in corners]
        np.testing.assert_allclose(pad.plane_clearances_m, distances, rtol=0, atol=1e-12)
        axes = oracle.geom_xmat[geom].reshape(3, 3)
        analytic_minimum = np.dot(oracle.geom_xpos[geom] - oracle.geom_xpos[plane], normal) - sum(
            model.geom_size[geom, axis] * abs(np.dot(normal, axes[:, axis])) for axis in range(3)
        )
        assert pad.minimum_plane_clearance_m == pytest.approx(analytic_minimum, abs=1e-12)
        (projection,) = pad.object_projections
        interval = interval_oracle(oracle, model, geom, obj, kind)
        np.testing.assert_allclose(projection.pad_local_interval_m, interval, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            projection.world_position, oracle.geom_xpos[obj], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            projection.world_rotation, oracle.geom_xmat[obj].reshape(3, 3), rtol=0, atol=1e-12
        )
        expected_overlap = [
            min(model.geom_size[geom, axis], interval[axis, 1])
            - max(-model.geom_size[geom, axis], interval[axis, 0])
            for axis in range(3)
        ]
        np.testing.assert_allclose(
            projection.signed_overlap_m, expected_overlap, rtol=0, atol=1e-12
        )
        assert np.any(projection.signed_overlap_m < 0), (
            "this separated fixture exercises negative gaps"
        )
        for value in (
            pad.world_corners,
            pad.plane_clearances_m,
            projection.pad_local_interval_m,
            projection.signed_overlap_m,
            projection.world_position,
            projection.world_rotation,
        ):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flags.writeable = True


def test_three_ordered_mirrored_seeds_keep_all_free_and_unselected_coordinates(mj, monkeypatch):
    import phymeo.control.manipulation_pose as module

    model = model_fixture(mj)
    q, target, options = declaration(mj, model)
    qa = [int(model.jnt_qposadr[model.joint(name).id]) for name in ("slide", "hinge")]
    assert min(qa) >= 7
    seeds = [q.copy() for _ in range(3)]
    seeds[1][qa] = (-0.02, -0.2)
    seeds[2][qa] = (0.02, 0.2)
    before, model_before = [seed.tobytes() for seed in seeds], binary(mj, model)
    original, calls = module.solve_native_point_ik, []

    def bounded_call(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(module, "solve_native_point_ik", bounded_call)
        for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
            guard.setattr(mj, name, forbidden)
        results = audit_native_contact_poses(model, target, seeds, ("slide", "hinge"), **options)
    assert len(results) == len(calls) == 3
    untouched = [address for address in range(model.nq) if address not in qa]
    for index, candidate in enumerate(results):
        assert candidate.seed_index == index and candidate.ik.iterations <= 100
        assert calls[index]["max_iterations"] == 100 and calls[index]["tolerance_m"] == 5e-4
        assert calls[index]["damping"] == 1e-3 and calls[index]["max_step"] == 0.15
        assert candidate.q_seed.tobytes() == before[index] == seeds[index].tobytes()
        assert candidate.ik.configuration[untouched].tobytes() == seeds[index][untouched].tobytes()
        data = native(mj, model, candidate.ik.configuration)
        body = model.body("tool").id
        actual = data.xpos[body] + data.xmat[body].reshape(3, 3) @ options["local_point"]
        np.testing.assert_allclose(candidate.ik.position, actual, rtol=0, atol=1e-12)
        assert candidate.ik.error_m == pytest.approx(np.linalg.norm(target - actual), abs=1e-12)
    assert results[0].ik.converged
    assert len({candidate.input_binding for candidate in results}) == 3
    assert binary(mj, model) == model_before
    # Failed local IK is retained with its actual bounded residual, not reclassified globally.
    (far,) = audit_native_contact_poses(model, (10, 10, 10), [q], ("slide", "hinge"), **options)
    assert not far.ik.converged and far.ik.iterations <= 100 and far.ik.error_m > 1
    assert len(far.pads) == 2


def test_projection_overlap_is_not_a_contact_or_grasp_claim(mj):
    model = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="support" type="plane" size="1 1 .1"/>'
        '<body pos=".14 .14 .5"><freejoint/><geom name="object" size=".05" mass=".1"/></body>'
        '<body name="tool" pos="0 0 .5"><joint name="hinge" axis="0 0 1"/>'
        '<geom name="left" type="box" size=".1 .1 .1" mass=".1"/>'
        '<geom name="right" type="box" pos="0 .4 0" size=".1 .1 .1" mass=".1"/>'
        "</body></worldbody></mujoco>"
    )
    (candidate,) = audit_native_contact_poses(
        model,
        (0, 0, 0.5),
        [model.qpos0.copy()],
        ("hinge",),
        body_name="tool",
        local_point=(0, 0, 0),
        pad_geom_names=("left", "right"),
        support_geom_name="support",
        object_geom_names=("object",),
    )
    (projection,) = candidate.pads[0].object_projections
    assert np.all(projection.signed_overlap_m > 0)
    # Exact sphere-to-AABB distance remains positive despite overlap on all three axes.
    distance = np.linalg.norm(np.maximum(np.abs((0.14, 0.14, 0)) - 0.1, 0)) - 0.05
    assert distance > 0
    assert not hasattr(candidate, "grasp") and not hasattr(candidate, "collision_free")


def test_admission_rejects_bad_names_states_shapes_and_budgets_before_ik(mj, monkeypatch):
    import phymeo.control.manipulation_pose as module

    model = model_fixture(mj)
    q, target, options = declaration(mj, model)
    invalid = [
        dict(q_seeds=[]),
        dict(q_seeds=[q] * 4),
        dict(q_seeds=(seed for seed in (q,))),
        dict(joint_names=("hinge", "hinge")),
        dict(joint_names=("object-free",)),
        dict(joint_names=("left-finger",)),
        dict(body_name="missing"),
        dict(pad_geom_names=("left-pad", "palm")),
        dict(pad_geom_names=("left-pad", "left-pad")),
        dict(support_geom_name="palm"),
        dict(object_geom_names=("palm",)),
        dict(object_geom_names=tuple(f"unknown-{i}" for i in range(33))),
        dict(target=(0, np.inf, 0)),
        dict(local_point=(0, 0)),
    ]
    nan, bad_quaternion, moved_free = q.copy(), q.copy(), q.copy()
    nan[-1], bad_quaternion[3:7], moved_free[0] = np.nan, 0, q[0] + 0.01
    invalid.extend(
        (dict(q_seeds=[q, nan]), dict(q_seeds=[bad_quaternion]), dict(q_seeds=[q, moved_free]))
    )
    request = dict(
        model=model, target=target, q_seeds=[q], joint_names=("slide", "hinge"), **options
    )
    before = binary(mj, model)
    monkeypatch.setattr(module, "solve_native_point_ik", forbidden)
    for override in invalid:
        with pytest.raises(ValueError):
            audit_native_contact_poses(**(request | override))
    unsupported = model_fixture(mj, "ellipsoid")
    with pytest.raises(ValueError, match="sphere/capsule/box"):
        audit_native_contact_poses(**(request | {"model": unsupported}))
    monkeypatch.setattr(module, "MAX_MODEL_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        audit_native_contact_poses(**request)
    assert binary(mj, model) == before


def test_bindings_include_target_tcp_seed_order_and_original_model_geometry(mj):
    model = model_fixture(mj)
    q, target, options = declaration(mj, model)
    (first,) = audit_native_contact_poses(model, target, [q], ("slide", "hinge"), **options)
    (again,) = audit_native_contact_poses(model, target, [q], ("slide", "hinge"), **options)
    assert first.input_binding == again.input_binding
    (no_objects,) = audit_native_contact_poses(
        model, target, [q], ("slide", "hinge"), **(options | {"object_geom_names": ()})
    )
    assert all(not pad.object_projections for pad in no_objects.pads)
    assert no_objects.input_binding != first.input_binding
    (shifted,) = audit_native_contact_poses(
        model, target + (0.001, 0, 0), [q], ("slide", "hinge"), **options
    )
    (tcp,) = audit_native_contact_poses(
        model,
        target,
        [q],
        ("slide", "hinge"),
        **(options | {"local_point": options["local_point"] + (0, 0.001, 0)}),
    )
    (reordered,) = audit_native_contact_poses(model, target, [q], ("hinge", "slide"), **options)
    paired = audit_native_contact_poses(model, target, [q, q.copy()], ("slide", "hinge"), **options)
    assert (
        len(
            {
                first.input_binding,
                shifted.input_binding,
                tcp.input_binding,
                reordered.input_binding,
                paired[0].input_binding,
                paired[1].input_binding,
            }
        )
        == 6
    )
    # Recompile geometry: native same-frame flags/inertia are compiler-derived.
    model = model_fixture(mj, left_pad_pos=".011 .002 -.005")
    (changed,) = audit_native_contact_poses(model, target, [q], ("slide", "hinge"), **options)
    assert (
        changed.model_sha256 != first.model_sha256 and changed.input_binding != first.input_binding
    )
    assert not np.array_equal(changed.pads[0].world_corners, first.pads[0].world_corners)
    # Move only this fixture's static plane along its original normal: report
    # negative signed clearance honestly instead of clamping to "safe".
    support_pos = " ".join(
        format(float(v), ".17g") for v in changed.support_origin + changed.support_normal
    )
    model = model_fixture(mj, left_pad_pos=".011 .002 -.005", support_pos=support_pos)
    (penetrating,) = audit_native_contact_poses(model, target, [q], ("slide", "hinge"), **options)
    assert penetrating.pads[0].minimum_plane_clearance_m < 0
    assert penetrating.pads[0].minimum_plane_clearance_m == pytest.approx(
        changed.pads[0].minimum_plane_clearance_m - 1, abs=1e-12
    )
    assert penetrating.input_binding != changed.input_binding
    for value in (
        first.q_seed,
        first.target,
        first.local_point,
        first.support_origin,
        first.support_normal,
    ):
        with pytest.raises(ValueError):
            value.flags.writeable = True

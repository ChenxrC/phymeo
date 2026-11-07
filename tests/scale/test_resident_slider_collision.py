"""Real accepted single-slider poses feed the existing resident device broad phase."""

import json
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.tyx_world_batch import FreeSphere, PrismaticSlider, TyxWorldBatch
from phymeo.dynamics.robot_kinematics import forward_kinematics


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _save_trace(tmp_path, name, data):
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(_jsonable(data), allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _result_trace(result):
    def key(k):
        return (k.world, k.body, k.shape)

    return {
        "state_epoch": result.state_epoch,
        "topology_generation": result.topology_generation,
        "world_keys": result.world_keys,
        "ticks": result.ticks,
        "shape_keys": [key(k) for k in result.shape_keys],
        "shape_revisions": result.shape_revisions,
        "bounds": result.bounds,
        "registered_pairs": [(key(a), key(b)) for a, b in result.registered_pairs],
        "candidates": [(key(a), key(b)) for a, b in result.candidates],
    }


def _api():
    from phymeo.backends._resident_slider_binding import (
        ResidentSliderBinding,
        SliderGeometryFrame,
        compile_slider_binding,
    )

    return ResidentSliderBinding, SliderGeometryFrame, compile_slider_binding


def _static(key, motion):
    from phymeo.backends._resident_collision_plan import ResidentStaticBinding

    return ResidentStaticBinding(key, motion)


def _sphere(key):
    from phymeo.backends._resident_collision_plan import ResidentSphereBinding

    return ResidentSphereBinding(key, key.world)


def _box(low, high):
    return np.asarray(list(product(*zip(low, high, strict=True))), dtype=float)


def _program(
    *,
    origin=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    axis=(1, 0, 0),
    mass=2.0,
    delay=0,
    q=0.0,
    v=0.0,
):
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([origin], dtype=float),
        np.array([orientation], dtype=float),
        np.array([axis], dtype=float),
        np.array([mass], dtype=float),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    drive = ph.DrivenRobotProgram(model, kp=[0.0], kd=[0.0], effort_limit=[4.0], delay_steps=delay)
    initial = drive.initial_state(ph.RobotState(np.array([q]), np.array([v])))
    return drive, initial


def _frame_args(program):
    model = program.model
    return {
        "origin_m": model.joint_position[0],
        "orientation_xyzw": model.joint_orientation[0],
        "axis_local": model.joint_axis[0],
    }


def _binding(key, program, zero_motion):
    binding, _, _ = _api()
    return binding(key, key.world, zero_motion, **_frame_args(program))


def _zero_motion(program, local_vertices):
    geometry = forward_kinematics(program.model, np.zeros(1))
    points = geometry.position[0] + np.asarray(local_vertices) @ geometry.rotation[0].T
    return ph.CollisionMotion(points)


def _contains(bounds, low, high):
    for i in range(3):
        assert Fraction(float(bounds[i])) <= Fraction(low[i])
        assert Fraction(float(bounds[i + 3])) >= Fraction(high[i])


def _row(result, key):
    return result.bounds[result.shape_keys.index(key)]


def _s01(world=7):
    s = np.sqrt(0.5)
    drive, initial = _program(
        origin=(1, -0.5, 0.25),
        orientation=(0, 0, s, s),
        axis=(1, 0, 0),
        q=0.125,
    )
    moving = ph.CollisionKey(world, 3, 11)
    wall = ph.CollisionKey(world, 9, 4)
    local = _box((1 / 8, -1 / 16, -1 / 32), (3 / 8, 1 / 16, 1 / 32))
    zero = _zero_motion(drive, local)
    fixed = ph.CollisionMotion(_box((15 / 16, 1 / 16, 7 / 32), (17 / 16, 3 / 16, 9 / 32)))
    model = ph.CollisionModel(
        (
            ph.CollisionShape(wall, "hull", tuple(range(8))),
            ph.CollisionShape(moving, "hull", tuple(range(101, 109))),
        )
    )
    return drive, initial, model, moving, wall, zero, fixed


@pytest.mark.tyx
def test_s01_actual_slider_step_changes_public_device_bounds(tyx_mode, tmp_path):
    drive, initial, model, moving, wall, zero, fixed = _s01()
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        command = drive.inputs(np.zeros(1), feedforward=np.array([2.0]))
        batch.set_controls({7: command})
        assert batch.step(1 / 4) == {7: True}
        actual = batch.snapshot()[7]
        assert actual.position == (3 / 16,)
        assert actual.velocity == (1 / 4,)
        # The pre-existing physical program must advance first. The valid RED uses
        # the candidate binding module with the old plan's unsupported-slider gate,
        # never a missing-module/import failure as proof of the public gap.
        bindings = (_binding(moving, drive, zero), _static(wall, fixed))
        with batch.prepare_collision(model, bindings) as plan:
            result = batch.collision(plan)
            assert result.world_keys == (7,)
            assert result.ticks.tolist() == [1]
            assert result.registered_pairs == ((moving, wall),)
            assert result.candidates == ((moving, wall),)
            _contains(
                _row(result, moving),
                (Fraction(15, 16), Fraction(-3, 16), Fraction(7, 32)),
                (Fraction(17, 16), Fraction(1, 16), Fraction(9, 32)),
            )
            assert plan.last_result is result
            assert plan.require_current(result) is result
            np.testing.assert_array_equal(bindings[0].zero_motion.start, zero.start)


def _round_scene(
    program, *, world=7, center=(0, 0, 0), radius=1 / 32, wall_low=None, wall_high=None
):
    moving = ph.CollisionKey(world, 3, 11)
    shape = ph.CollisionShape(moving, "sphere", (37,), radius, revision=5)
    zero = _zero_motion(program, [center])
    shapes = [shape]
    bindings = [_binding(moving, program, zero)]
    wall = None
    if wall_low is not None:
        wall = ph.CollisionKey(world, 9, 4)
        shapes.append(ph.CollisionShape(wall, "hull", tuple(range(8))))
        bindings.append(_static(wall, ph.CollisionMotion(_box(wall_low, wall_high))))
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), moving, wall


def _world_axis(program):
    from phymeo.dynamics.rotation import quaternion_matrix

    axis = program.model.joint_axis[0]
    return quaternion_matrix(program.model.joint_orientation)[0] @ (axis / np.linalg.norm(axis))


def _translated_sphere_bounds(zero, radius, axis, q):
    displacement = [Fraction(float(a)) * Fraction(float(q)) for a in axis]
    low = [Fraction(float(zero.start[0, i])) - Fraction(radius) + displacement[i] for i in range(3)]
    high = [
        Fraction(float(zero.start[0, i])) + Fraction(radius) + displacement[i] for i in range(3)
    ]
    return low, high


@pytest.mark.tyx
def test_s02_oblique_negative_q_touch_positive_q_and_resolved_miss(tyx_mode, tmp_path):
    drive, initial = _program(origin=(-0.5, 0.25, 0), axis=(3 / 5, 4 / 5, 0), q=-0.25)
    low, high = (-79 / 160, -1 / 16, 0), (-69 / 160, 3 / 16, 1 / 8)
    model, bindings, moving, wall = _round_scene(
        drive, center=(1 / 8, 0, 1 / 16), wall_low=low, wall_high=high
    )
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        saved = batch.snapshot()[7]
        with batch.prepare_collision(model, bindings) as plan:
            negative = plan.query()
            _contains(
                _row(negative, moving),
                (Fraction(-89, 160), Fraction(3, 160), Fraction(1, 32)),
                (Fraction(-79, 160), Fraction(13, 160), Fraction(3, 32)),
            )
            assert negative.candidates == ((moving, wall),)
            batch.restore({7: replace(saved, position=(0.25,))})
            positive = plan.query()
            _contains(
                _row(positive, moving),
                (Fraction(-41, 160), Fraction(67, 160), Fraction(1, 32)),
                (Fraction(-31, 160), Fraction(77, 160), Fraction(3, 32)),
            )
            assert positive.candidates == ()
        batch.restore({7: saved})
        far = ph.CollisionMotion(
            _box(np.array(low) + [1 / 64, 0, 0], np.array(high) + [1 / 64, 0, 0])
        )
        with batch.prepare_collision(model, (bindings[0], _static(wall, far))) as miss:
            assert miss.query().candidates == ()


@pytest.mark.tyx
def test_s03_real_world_force_projects_once_before_geometry(tyx_mode, tmp_path):
    drive, initial = _program(axis=(3 / 5, 4 / 5, 0))
    model, bindings, moving, _ = _round_scene(drive)
    command = drive.inputs(np.zeros(1))
    command.domains[drive.domain].force[0] = (0, 5, 0)
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: command})
        assert batch.step(1 / 8) == {7: True}
        actual = batch.snapshot()[7]
        assert actual.position == (1 / 32,) and actual.velocity == (1 / 4,)
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            low = (
                Fraction(3, 160) - Fraction(1, 32),
                Fraction(1, 40) - Fraction(1, 32),
                Fraction(-1, 32),
            )
            high = (
                Fraction(3, 160) + Fraction(1, 32),
                Fraction(1, 40) + Fraction(1, 32),
                Fraction(1, 32),
            )
            _contains(_row(result, moving), low, high)
            assert result.candidates == result.registered_pairs == ()


def _multi_shapes(program, *, world=7):
    specs = (
        ("sphere", [(1 / 8, 0, 0)], 1 / 32, (41,)),
        ("capsule", [(0, -1 / 4, 1 / 8), (0, 1 / 4, 1 / 8)], 1 / 64, (9, 73)),
        (
            "hull",
            _box((-1 / 16, -1 / 32, -1 / 8), (1 / 16, 1 / 32, 0)),
            0.0,
            (102, 18, 39, 44, 60, 3, 81, 97),
        ),
    )
    shapes, bindings = [], []
    for i, (kind, vertices, radius, ids) in enumerate(specs):
        key = ph.CollisionKey(world, 3, 21 + i)
        shape = ph.CollisionShape(key, kind, ids, radius, revision=7)
        shapes.append(shape)
        bindings.append(_binding(key, program, _zero_motion(program, vertices)))
    fixed = ph.CollisionKey(world, 9, 4)
    shapes.append(ph.CollisionShape(fixed, "hull", tuple(range(8))))
    bindings.append(_static(fixed, ph.CollisionMotion(_box((-1, -1, -1), (1, 1, 1)))))
    return ph.CollisionModel(tuple(reversed(shapes))), tuple(reversed(bindings))


@pytest.mark.tyx
def test_s04_same_physical_body_keeps_three_material_sets_and_ticks(tyx_mode, tmp_path):
    drive, initial = _program(origin=(1 / 4, -1 / 8, 1 / 16), v=1 / 4)
    model, bindings = _multi_shapes(drive)
    by_key = {b.key: b for b in bindings}
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        assert batch.step(1 / 4) == {7: True}
        assert batch.snapshot()[7].position == (1 / 16,)
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            assert result.ticks.tolist() == [1]
            assert len(result.registered_pairs) == len(result.candidates) == 3
            for shape in model.shapes:
                binding = by_key[shape.key]
                if hasattr(binding, "zero_motion"):
                    expected = binding.zero_motion.start + [1 / 16, 0, 0]
                    _contains(
                        _row(result, shape.key),
                        expected.min(0) - shape.radius,
                        expected.max(0) + shape.radius,
                    )
                    assert result.shape_revisions[result.shape_keys.index(shape.key)] == 7
                    expected_ids = {21: (41,), 22: (9, 73), 23: (102, 18, 39, 44, 60, 3, 81, 97)}
                    assert shape.vertex_ids == expected_ids[shape.key.shape]


def _folded_asset(tmp_path):
    xml = """<robot name="slider_with_fixed_tool">
<link name="base"><collision name="root"><origin xyz="-2 0 0"/>
<geometry><sphere radius="0.03125"/></geometry></collision></link>
<link name="slider"><inertial><origin xyz="0.0625 0 0"/><mass value="2"/>
<inertia ixx="0.125" iyy="0.125" izz="0.125" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="ball"><origin xyz="0.125 0 0"/><geometry><sphere radius="0.03125"/></geometry></collision>
<collision name="rod"><origin rpy="0 1.5707963267948966 0"/>
<geometry><cylinder radius="0.015625" length="0.5"/></geometry></collision></link>
<link name="tool"><inertial><mass value="1"/>
<inertia ixx="0.03125" iyy="0.03125" izz="0.03125" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="tool_box"><geometry><box size="0.125 0.0625 0.03125"/></geometry></collision></link>
<joint name="rail" type="prismatic"><parent link="base"/><child link="slider"/>
<origin xyz="0 0 0.25" rpy="1.5707963267948966 0 0"/><axis xyz="0 1 0"/>
<limit lower="-10" upper="10" effort="4" velocity="2"/></joint>
<joint name="tool_fixed" type="fixed"><parent link="slider"/><child link="tool"/>
<origin xyz="0.125 0.0625 0" rpy="0 0 1.5707963267948966"/></joint></robot>"""
    path = tmp_path / "slider_tool.urdf"
    path.write_text(xml, encoding="utf-8")
    return ph.load_urdf(path, gravity=(0, 0, 0))


def _motion_bounds(shape, motion):
    if shape.curved is None:
        return motion.start.min(0) - shape.radius, motion.start.max(0) + shape.radius
    from phymeo.dynamics.rotation import quaternion_matrix

    body = quaternion_matrix(np.array([motion.body_orientation_xyzw]))[0]
    local = quaternion_matrix(np.array([shape.curved.orientation_xyzw]))[0]
    assert shape.kind == "cylinder"
    radius, half_height = shape.curved.dimensions_m
    axis = (body @ local)[:, 2]
    extent = radius * np.sqrt(np.maximum(0, 1 - axis * axis)) + half_height * np.abs(axis)
    return motion.start[0] - extent, motion.start[0] + extent


@pytest.mark.tyx
def test_s04_folded_tool_uses_real_group_owner_zero_pose_and_public_fk(tyx_mode, tmp_path):
    resident, _, _ = _api()
    asset = _folded_asset(tmp_path)
    assert asset.model.joint_types == ("prismatic",)
    assert asset.sites["tool"].link_index == 0
    drive = ph.DrivenRobotProgram(asset.model, kp=[0.0], kd=[0.0], effort_limit=[4.0])
    initial = drive.initial_state(ph.RobotState(np.zeros(1), np.array([0.25])))
    bindings, shapes, originals = [], [], {}
    for col in reversed(asset.collisions):
        static = asset.sites[col.link_name].link_index is None
        key = ph.CollisionKey(7, 9 if static else 3, col.shape.key.shape)
        shapes.append(replace(col.shape, key=key))
        originals[key] = col
        if static:
            bindings.append(_static(key, asset.collision_motion(np.zeros(1))[col.shape.key]))
            with pytest.raises(ValueError, match="root|active|group"):
                resident.from_asset(asset, col.shape.key, key=key, world=7)
        else:
            bindings.append(resident.from_asset(asset, col.shape.key, key=key, world=7))
    model = ph.CollisionModel(tuple(shapes))
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path / "rt"
    ) as batch:
        assert batch.step(1 / 4) == {7: True}
        assert batch.snapshot()[7].position == (1 / 16,)
        with batch.prepare_collision(model, tuple(bindings)) as plan:
            actual = plan.query()
            expected = asset.collision_motion(np.array([1 / 16]))
            for shape in model.shapes:
                col = originals[shape.key]
                assert shape.vertex_ids == col.shape.vertex_ids
                assert shape.revision == col.shape.revision
                _contains(
                    _row(actual, shape.key), *_motion_bounds(col.shape, expected[col.shape.key])
                )


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0, 1))
def test_s05_forty_real_steps_keep_delay_and_exact_candidate_window(delay, tyx_mode, tmp_path):
    drive, initial = _program(delay=delay)
    moving, wall = ph.CollisionKey(7, 3, 11), ph.CollisionKey(7, 9, 4)
    zero = _zero_motion(drive, _box((-1 / 16,) * 3, (1 / 16,) * 3))
    model = ph.CollisionModel(
        (
            ph.CollisionShape(moving, "hull", tuple(range(8))),
            ph.CollisionShape(wall, "hull", tuple(range(20, 28))),
        )
    )
    fixed = ph.CollisionMotion(_box((1, -1 / 8, -1 / 8), (17 / 16, 1 / 8, 1 / 8)))
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({7: drive.inputs(np.zeros(1), feedforward=np.array([2.0]))})
        with batch.prepare_collision(
            model, (_binding(moving, drive, zero), _static(wall, fixed))
        ) as plan:
            hit_steps, trace = [], []
            for n in range(1, 41):
                accepted = batch.step(1 / 16)
                actual = batch.snapshot()[7]
                row = {"step": n, "acceptance": accepted, "actual_checkpoint": asdict(actual)}
                trace.append(row)
                _save_trace(tmp_path, f"s05-delay{delay}-{tyx_mode}", trace)
                assert accepted == {7: True}
                m = n - delay
                q, v = Fraction(m * (m + 1), 512), Fraction(m, 16)
                assert actual.tick == n
                assert actual.position == (float(q),) and actual.velocity == (float(v),)
                result = plan.query()
                row.update(
                    collision=_result_trace(result),
                    plan_stats=plan.stats(),
                    batch_stats=batch.stats(),
                )
                _save_trace(tmp_path, f"s05-delay{delay}-{tyx_mode}", trace)
                assert result.ticks.tolist() == [n]
                _contains(
                    _row(result, moving),
                    (q - Fraction(1, 16), -1 / 16, -1 / 16),
                    (q + Fraction(1, 16), 1 / 16, 1 / 16),
                )
                if result.candidates:
                    assert result.candidates == ((moving, wall),)
                    hit_steps.append(n)
            assert hit_steps == ([22, 23] if delay == 0 else [23, 24])
            assert actual.position == ((205 / 64,) if delay == 0 else (195 / 64,))


def _mixed_worlds(count):
    worlds, shapes, bindings, controls, references = [], [], [], {}, {}
    for i in reversed(range(count)):
        world = 11 + 17 * i
        kind = 1 if count == 1 else i % 3
        if kind == 0:
            moving, wall = ph.CollisionKey(world, 3, 11), ph.CollisionKey(world, 9, 4)
            sphere = FreeSphere(2, 1 / 8, (-1 / 4, 0, 0), (1 / 4, 0, 0), (0, 0, 0))
            worlds.append((world, sphere))
            shapes.extend(
                (
                    ph.CollisionShape(moving, "sphere", (37,), 1 / 8),
                    ph.CollisionShape(wall, "hull", tuple(range(8))),
                )
            )
            fixed = ph.CollisionMotion(_box((0, -1 / 4, -1 / 4), (1 / 2, 1 / 4, 1 / 4)))
            bindings.extend((_sphere(moving), _static(wall, fixed)))
            references[world] = (moving, wall, fixed, None, None, None)
        else:
            if kind == 1:
                drive, initial, model, moving, wall, zero, fixed = _s01(world)
                rows = (_binding(moving, drive, zero), _static(wall, fixed))
            else:
                drive, initial = _program(origin=(-0.5, 0.25, 0), axis=(3 / 5, 4 / 5, 0), q=-0.25)
                model, rows, moving, wall = _round_scene(
                    drive,
                    world=world,
                    center=(1 / 8, 0, 1 / 16),
                    wall_low=(-79 / 160, -1 / 16, 0),
                    wall_high=(-69 / 160, 3 / 16, 1 / 8),
                )
                zero, fixed = rows[0].zero_motion, rows[1].motion
            worlds.append((world, PrismaticSlider(drive, initial)))
            shapes.extend(model.shapes)
            bindings.extend(rows)
            controls[world] = drive.inputs(np.zeros(1), feedforward=np.array([2.0]))
            references[world] = (
                moving,
                wall,
                fixed,
                zero,
                drive,
                float(initial.domains[drive.domain].q[0]),
            )
    return (
        worlds,
        ph.CollisionModel(tuple(reversed(shapes))),
        tuple(reversed(bindings)),
        controls,
        references,
    )


@pytest.mark.tyx
@pytest.mark.parametrize("count", (1, 4, 32))
def test_s06_mixed_worlds_full_results_two_buffers_without_warm_snapshots(
    count, tyx_mode, tmp_path, monkeypatch
):
    worlds, model, bindings, controls, refs = _mixed_worlds(count)
    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        with batch.prepare_collision(model, bindings) as plan:
            initial_result = plan.query()
            initial_plan_stats, initial_batch_stats = plan.stats(), batch.stats()
            initial_acceptance = batch.step(1 / 8)
            assert all(initial_acceptance.values())
            stepped_result = plan.query()
            warm_entry = batch.snapshot()
            before, runtime_before = plan.stats(), batch.stats()
            trace = {
                "bootstrap": [
                    {
                        "step": 0,
                        "collision": _result_trace(initial_result),
                        "plan_stats": initial_plan_stats,
                        "batch_stats": initial_batch_stats,
                    },
                    {
                        "step": 1,
                        "acceptance": initial_acceptance,
                        "collision": _result_trace(stepped_result),
                    },
                ],
                "actual_warm_entry_checkpoints": {w: asdict(s) for w, s in warm_entry.items()},
                "initial_plan_stats": before,
                "initial_batch_stats": runtime_before,
                "steps": [],
            }
            _save_trace(tmp_path, f"s06-w{count}-{tyx_mode}", trace)

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "warm slider query must not snapshot, upload envelopes or allocate"
                )

            monkeypatch.setattr(batch, "snapshot", forbidden)
            monkeypatch.setattr(batch, "kinematics", forbidden)
            monkeypatch.setattr(plan._engine, "query", forbidden)
            if tyx_mode == "cuda":
                import tyx as tx

                monkeypatch.setattr(tx, "cuda_array", forbidden)
                monkeypatch.setattr(tx, "cuda_empty", forbidden)
            frozen, saved_bounds = None, None
            for n in range(2, 6):
                accepted = batch.step(1 / 8)
                result = plan.query()
                row = {
                    "step": n,
                    "acceptance": accepted,
                    "collision": _result_trace(result),
                    "plan_stats": plan.stats(),
                    "batch_stats": batch.stats(),
                    "oracle_q_v": {},
                }
                trace["steps"].append(row)
                _save_trace(tmp_path, f"s06-w{count}-{tyx_mode}", trace)
                assert all(accepted.values())
                assert result.world_keys == tuple(sorted(refs))
                assert result.ticks.tolist() == [n] * count
                expected_pairs = set()
                for world, (moving, wall, fixed, zero, drive, q0) in refs.items():
                    shape = next(s for s in model.shapes if s.key == moving)
                    if drive is None:
                        center = np.array([-1 / 4 + n / 32, 0, 0])
                        low, high = center - 1 / 8, center + 1 / 8
                        row["oracle_q_v"][world] = {
                            "position_m": center,
                            "velocity_m_s": [1 / 4, 0, 0],
                        }
                    else:
                        q = q0 + n * (n + 1) / 128
                        motion = zero.start + _world_axis(drive) * q
                        low, high = motion.min(0) - shape.radius, motion.max(0) + shape.radius
                        row["oracle_q_v"][world] = {"q_m": q, "v_m_s": n / 8}
                    _contains(_row(result, moving), low, high)
                    if np.all(np.asarray(low) <= fixed.start.max(0)) and np.all(
                        np.asarray(high) >= fixed.start.min(0)
                    ):
                        expected_pairs.add((moving, wall))
                _save_trace(tmp_path, f"s06-w{count}-{tyx_mode}", trace)
                assert set(result.candidates) == expected_pairs
                assert len(result.registered_pairs) == count
                assert all(a.world == b.world for a, b in result.candidates)
                if frozen is None:
                    frozen, saved_bounds = result, result.bounds.copy()
                np.testing.assert_array_equal(frozen.bounds, saved_bounds)
            after = plan.stats()
            trace.update(final_plan_stats=after, final_batch_stats=batch.stats())
            _save_trace(tmp_path, f"s06-w{count}-{tyx_mode}", trace)
            assert after["device_allocations"] == before["device_allocations"]
            assert after["dynamic_envelope_uploaded_bytes"] == 0
            assert after["bounds_kernel_launches"] - before["bounds_kernel_launches"] == 4
            assert (
                after["slider_bounds_kernel_launches"] - before["slider_bounds_kernel_launches"]
                == 4
            )
            assert after["copied_to_host_bytes"] > before["copied_to_host_bytes"]
            assert after["state_buffer_directions"] == (0, 1)
            assert (
                batch.stats()["snapshot_downloaded_bytes"]
                == runtime_before["snapshot_downloaded_bytes"]
            )
            assert (
                batch.stats()["kinematics_downloaded_bytes"]
                == runtime_before["kinematics_downloaded_bytes"]
            )
            for array in (result.bounds, result.ticks):
                with pytest.raises(ValueError):
                    array.setflags(write=True)


@pytest.mark.tyx
def test_s07_restore_same_tick_reset_reject_and_registration(tyx_mode, tmp_path):
    worlds, model, bindings, controls, _ = _mixed_worlds(4)
    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        saved = batch.snapshot()
        slider_world = 28
        with (
            batch.prepare_collision(model, bindings) as first,
            batch.prepare_collision(model, bindings) as second,
        ):
            old = first.query()
            foreign = second.query()
            with pytest.raises(ValueError, match="registration|different"):
                first.require_current(foreign)
            batch.restore({slider_world: replace(saved[slider_world], position=(-1.0,))})
            moved = first.query()
            assert moved.ticks.tolist() == old.ticks.tolist()
            assert moved.state_epoch > old.state_epoch
            key = ph.CollisionKey(slider_world, 3, 11)
            assert not np.array_equal(_row(moved, key), _row(old, key))
            with pytest.raises(ValueError, match="stale|epoch"):
                first.require_current(old)
            batch.reset([slider_world])
            reset = first.query()
            np.testing.assert_array_equal(_row(reset, key), _row(old, key))
            batch.restore({slider_world: replace(saved[slider_world], tick=2**31 - 1)})
            before = first.query()
            accepted = batch.step(1 / 8)
            assert accepted[slider_world] is False and sum(accepted.values()) == 3
            current = first.query()
            np.testing.assert_array_equal(_row(current, key), _row(before, key))
            assert current.ticks[current.world_keys.index(slider_world)] == 2**31 - 1
            assert all(
                t == 1
                for w, t in zip(current.world_keys, current.ticks, strict=True)
                if w != slider_world
            )


@pytest.mark.tyx
def test_s07_query_failure_and_inconsistent_same_world_ticks_do_not_publish(
    tyx_mode, tmp_path, monkeypatch
):
    drive, initial = _program()
    model, bindings = _multi_shapes(drive)
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            first, physical = plan.query(), batch.snapshot()
            original = plan._engine.query_device_bounds

            def fail(**kwargs):
                raise RuntimeError("injected slider BVH failure")

            monkeypatch.setattr(plan._engine, "query_device_bounds", fail)
            with pytest.raises(RuntimeError, match="slider BVH"):
                plan.query()
            assert plan.last_result is first and batch.snapshot() == physical
            assert plan.require_current(first) is first
            monkeypatch.setattr(plan._engine, "query_device_bounds", original)
            good = plan.query()
            native_read = plan._engine._read

            def inconsistent(array):
                value = native_read(array)
                if array is plan._stamp_device:
                    value = value.copy()
                    value[good.shape_keys.index(ph.CollisionKey(7, 3, 22)), 1] += 1
                return value

            monkeypatch.setattr(plan._engine, "_read", inconsistent)
            with pytest.raises(ValueError, match="tick|world|stamp"):
                plan.query()
            assert plan.last_result is good and batch.snapshot() == physical


@pytest.mark.tyx
def test_s08_large_cancellation_and_overflow_are_not_small_final_value_margins(tyx_mode, tmp_path):
    drive, initial = _program(axis=(3 / 5, 4 / 5, 0), q=-float(2**20))
    center = (3 * 2**20 / 5, 4 * 2**20 / 5, 0)
    model, bindings, moving, _ = _round_scene(drive, center=center)
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path / "cancel"
    ) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            result = plan.query()
            _save_trace(
                tmp_path,
                f"s08-cancellation-{tyx_mode}",
                {
                    "collision": _result_trace(result),
                    "plan_stats": plan.stats(),
                    "batch_stats": batch.stats(),
                },
            )
            low, high = _translated_sphere_bounds(
                bindings[0].zero_motion, 1 / 32, _world_axis(drive), -(2**20)
            )
            _contains(_row(result, moving), low, high)
            _contains(_row(result, moving), (-1 / 32,) * 3, (1 / 32,) * 3)
    drive, initial = _program()
    model, bindings, _, _ = _round_scene(drive, center=(float(2**127), 0, 0))
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path / "overflow"
    ) as batch:
        saved = batch.snapshot()[7]
        with batch.prepare_collision(model, bindings) as plan:
            good = plan.query()
            batch.restore({7: replace(saved, position=(float(2**127),))})
            with pytest.raises(ValueError, match="bound|finite|overflow"):
                plan.query()
            actual = batch.snapshot()[7]
            _save_trace(
                tmp_path,
                f"s08-overflow-{tyx_mode}",
                {
                    "last_valid": _result_trace(good),
                    "actual_checkpoint": asdict(actual),
                    "plan_stats": plan.stats(),
                    "batch_stats": batch.stats(),
                },
            )
            assert plan.last_result is good
            assert actual.position == (float(2**127),)


@pytest.mark.tyx
@pytest.mark.parametrize("tiny_axis", (2**-140, -(2**-140)))
def test_s08_subnormal_axis_interval_contains_accepted_physical_displacement(
    tiny_axis, tyx_mode, tmp_path
):
    drive, initial = _program(axis=(1, tiny_axis, 0), q=float(2**100))
    model, bindings, moving, _ = _round_scene(drive)
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        with batch.prepare_collision(model, bindings) as plan:
            low, high = _translated_sphere_bounds(
                bindings[0].zero_motion, 1 / 32, _world_axis(drive), 2**100
            )
            actual = plan.query()
            sign = "positive" if tiny_axis > 0 else "negative"
            _save_trace(
                tmp_path,
                f"s08-small-axis-{sign}-{tyx_mode}",
                {
                    "collision": _result_trace(actual),
                    "plan_stats": plan.stats(),
                    "batch_stats": batch.stats(),
                },
            )
            _contains(_row(actual, moving), low, high)


@pytest.mark.tyx
def test_s08_zero_capacity_empty_plan_and_runtime_ownership(tyx_mode, tmp_path):
    drive, initial, model, moving, wall, zero, fixed = _s01()
    batch = TyxWorldBatch([(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path)
    try:
        bindings = (_binding(moving, drive, zero), _static(wall, fixed))
        small = batch.prepare_collision(model, bindings, candidate_capacity=0)
        old = small.query()
        assert old.candidates == ()
        batch.set_controls({7: drive.inputs(np.zeros(1), feedforward=np.array([2.0]))})
        assert batch.step(1 / 4) == {7: True}
        with pytest.raises(ValueError, match="capacity"):
            small.query()
        assert small.last_result is old
        full = batch.prepare_collision(model, bindings, candidate_capacity=1)
        touched = full.query()
        _save_trace(
            tmp_path,
            f"s08-capacity-{tyx_mode}",
            {
                "before": _result_trace(old),
                "full_result": _result_trace(touched),
                "small_stats": small.stats(),
                "full_stats": full.stats(),
                "batch_stats": batch.stats(),
            },
        )
        assert touched.candidates == ((moving, wall),)
        small.close()
        assert batch.step(1 / 16) == {7: True}
        full.query()
        batch.close()
        _save_trace(
            tmp_path,
            f"s08-normal-close-{tyx_mode}",
            {
                "small_stats": small.stats(),
                "full_stats": full.stats(),
                "batch_stats": batch.stats(),
            },
        )
        assert full.stats()["closed"]
        with pytest.raises(RuntimeError, match="closed"):
            full.query()
    finally:
        batch.close()
    with TyxWorldBatch([], tyx_mode, cache_root=tmp_path / "empty") as empty:
        with empty.prepare_collision(ph.CollisionModel(()), ()) as plan:
            result = plan.query()
            _save_trace(
                tmp_path,
                f"s08-empty-{tyx_mode}",
                {
                    "collision": _result_trace(result),
                    "plan_stats": plan.stats(),
                    "batch_stats": empty.stats(),
                },
            )
            assert result.bounds.shape == (0, 6)
            assert plan.stats()["device_allocations"] == 0
            assert plan.stats()["slider_bounds_kernel_launches"] == 0


def test_s08_frame_and_compiled_axis_are_owned_and_rotation_sign_equivalent():
    resident, frame_type, compile_binding = _api()
    origin, orientation, axis = (
        np.array([1, 2, 3.0]),
        np.array([0, 0, 0, 1.0]),
        np.array([3, 4, 0.0]),
    )
    frame = frame_type(origin, orientation, axis)
    key = ph.CollisionKey(7, 3, 11)
    shape = ph.CollisionShape(key, "sphere", (91,), 1 / 32)
    motion = ph.CollisionMotion([[1, 2, 3]])
    binding = resident(
        key, 7, motion, origin_m=origin, orientation_xyzw=-orientation, axis_local=axis
    )
    owned_axis = np.array([3 / 5, 4 / 5, 0.0])
    actual, intervals = compile_binding(shape, binding, frame, owned_axis)
    origin[:], orientation[:], axis[:] = 0, 7, 9
    assert frame.origin_m == (1, 2, 3) and frame.axis_local == (3 / 5, 4 / 5, 0)
    assert binding.zero_motion is not motion
    assert intervals.dtype == np.float32 and intervals.shape == (6,)
    for i in range(3):
        assert Fraction(float(intervals[i])) <= Fraction(float(owned_axis[i]))
        assert Fraction(float(intervals[i + 3])) >= Fraction(float(owned_axis[i]))
    saved = intervals.copy()
    owned_axis[:] = -7
    np.testing.assert_array_equal(intervals, saved)
    for array in (actual.start, binding.zero_motion.start, intervals):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    "problem",
    (
        "origin",
        "quaternion",
        "axis",
        "world",
        "moving",
        "turn",
        "vertices",
        "plane",
        "owned_axis",
        "missing_frame",
    ),
)
def test_s08_invalid_binding_compile_fails_closed(problem):
    resident, frame_type, compile_binding = _api()
    key = ph.CollisionKey(7, 3, 11)
    frame = frame_type((0, 0, 0), (0, 0, 0, 1), (1, 0, 0))
    args = dict(origin_m=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1), axis_local=(1, 0, 0))
    shape, motion, world, owned = (
        ph.CollisionShape(key, "sphere", (37,), 1 / 32),
        ph.CollisionMotion([[0, 0, 0]]),
        7,
        (1, 0, 0),
    )
    if problem == "origin":
        args["origin_m"] = (1, 0, 0)
    elif problem == "quaternion":
        args["orientation_xyzw"] = (0, 0, 1, 0)
    elif problem == "axis":
        args["axis_local"] = (0, 1, 0)
    elif problem == "world":
        world = 8
    elif problem == "moving":
        motion = ph.CollisionMotion([[0, 0, 0]], [[1, 0, 0]])
    elif problem == "turn":
        motion = ph.CollisionMotion(
            [[0, 0, 0]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 1)
        )
    elif problem == "vertices":
        motion = ph.CollisionMotion([[0, 0, 0], [1, 0, 0]])
    elif problem == "plane":
        shape = ph.CollisionShape(key, "plane", (37,), normal=(0, 0, 1))
    elif problem == "owned_axis":
        owned = (0, 1, 0)
    else:
        frame = None
    with pytest.raises((TypeError, ValueError)):
        binding = resident(key, world, motion, **args)
        compile_binding(shape, binding, frame, owned)


@pytest.mark.parametrize(
    "field,value",
    (
        ("origin_m", (0, np.inf, 0)),
        ("orientation_xyzw", (0, 0, 0, 0)),
        ("orientation_xyzw", (0, 0, 0, 2)),
        ("axis_local", (0, 0, 0)),
        ("axis_local", (0, np.nan, 1)),
    ),
)
def test_s08_invalid_frame_rejects_before_runtime(field, value):
    _, frame_type, _ = _api()
    args = dict(origin_m=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1), axis_local=(1, 0, 0))
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        frame_type(**args)


def test_s04_from_asset_rejects_changed_shape_and_nonprismatic_source(tmp_path):
    resident, frame_type, compile_binding = _api()
    asset = _folded_asset(tmp_path)
    col = next(c for c in asset.collisions if c.name == "ball")
    key = ph.CollisionKey(7, 3, 11)
    binding = resident.from_asset(asset, col.shape.key, key=key, world=7)
    frame = frame_type(
        asset.model.joint_position[0], asset.model.joint_orientation[0], asset.model.joint_axis[0]
    )
    from phymeo.dynamics.rotation import quaternion_matrix

    axis = quaternion_matrix(asset.model.joint_orientation)[0] @ asset.model.joint_axis[0]
    compile_binding(replace(col.shape, key=key), binding, frame, axis)
    for bad in (
        replace(col.shape, key=key, radius=2 * col.shape.radius),
        replace(col.shape, key=key, vertex_ids=(991,)),
        replace(col.shape, key=key, revision=col.shape.revision + 1),
    ):
        with pytest.raises(ValueError, match="source|material|shape"):
            compile_binding(bad, binding, frame, axis)
    changed = replace(asset, model=replace(asset.model, joint_types=("revolute",)))
    with pytest.raises(ValueError, match="prismatic|slider"):
        resident.from_asset(changed, col.shape.key, key=key, world=7)
    with pytest.raises((TypeError, ValueError)):
        resident.from_asset(object(), col.shape.key, key=key, world=7)


@pytest.mark.tyx
@pytest.mark.parametrize("problem", ("different_body", "wrong_frame", "wrong_world"))
def test_s04_invalid_scene_binding_has_no_collision_allocation(problem, tyx_mode, tmp_path):
    resident, _, _ = _api()
    drive, initial = _program()
    model, bindings = _multi_shapes(drive)
    bindings, shapes = list(bindings), list(model.shapes)
    target = next(b for b in bindings if hasattr(b, "zero_motion"))
    i = bindings.index(target)
    if problem == "different_body":
        changed = ph.CollisionKey(7, 4, target.key.shape)
        shapes = [replace(s, key=changed) if s.key == target.key else s for s in shapes]
        bindings[i] = resident(changed, 7, target.zero_motion, **_frame_args(drive))
    elif problem == "wrong_frame":
        bindings[i] = resident(
            target.key,
            7,
            target.zero_motion,
            origin_m=(1, 0, 0),
            orientation_xyzw=(0, 0, 0, 1),
            axis_local=(1, 0, 0),
        )
    else:
        changed = ph.CollisionKey(8, 3, target.key.shape)
        shapes = [replace(s, key=changed) if s.key == target.key else s for s in shapes]
        bindings[i] = resident(changed, 8, target.zero_motion, **_frame_args(drive))
    with TyxWorldBatch(
        [(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path
    ) as batch:
        before = batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings))
        assert batch.stats()["device_allocations"] == before
        assert not batch._collision_plans


@pytest.mark.tyx
def test_s08_slider_close_callback_failure_cleans_all_plans_and_runtime(
    tyx_mode, tmp_path, monkeypatch
):
    drive, initial = _program()
    model, bindings = _multi_shapes(drive)
    batch = TyxWorldBatch([(7, PrismaticSlider(drive, initial))], tyx_mode, cache_root=tmp_path)
    plans = []
    try:
        plans = [batch.prepare_collision(model, bindings) for _ in range(2)]
        results = [p.query() for p in plans]
        original = plans[0].close

        def failed_callback():
            original()
            raise RuntimeError("injected slider close callback")

        monkeypatch.setattr(plans[0], "close", failed_callback)
        with pytest.raises(RuntimeError, match="slider close callback"):
            batch.close()
        stats = [p.stats() for p in plans]
        _save_trace(
            tmp_path,
            f"s08-close-callback-{tyx_mode}",
            {
                "results_before_close": [_result_trace(r) for r in results],
                "plan_stats_after": stats,
                "batch_stats_after": batch.stats(),
            },
        )
        assert all(s["closed"] and s["device_owned_bytes"] == 0 for s in stats)
        assert batch.stats()["device_owned_bytes"] == 0
    finally:
        batch.close()


def test_s04_plain_asset_frame_and_zero_geometry_are_owned(tmp_path):
    resident, frame_type, compile_binding = _api()
    path = tmp_path / "plain_slider.urdf"
    path.write_text(
        """<robot name="ordinary_slider"><link name="base"/>
<link name="moving"><inertial><mass value="2"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="offset_ball"><origin xyz="0.125 0 0"/>
<geometry><sphere radius="0.03125"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="moving"/>
<origin xyz="1 0.25 -0.5" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/>
<limit lower="-10" upper="10" effort="4" velocity="2"/></joint></robot>""",
        encoding="utf-8",
    )
    asset = ph.load_urdf(path, gravity=(0, 0, 0), collapse_fixed_joints=False)
    col = asset.collisions[0]
    assert col.link_index == 0
    key = ph.CollisionKey(7, 3, 11)
    binding = resident.from_asset(asset, col.shape.key, key=key, world=7)
    original = asset.collision_motion(np.zeros(1))[col.shape.key]
    np.testing.assert_array_equal(binding.zero_motion.start, original.start)
    frame = frame_type(
        asset.model.joint_position[0], asset.model.joint_orientation[0], asset.model.joint_axis[0]
    )
    from phymeo.dynamics.rotation import quaternion_matrix

    axis = quaternion_matrix(asset.model.joint_orientation)[0] @ asset.model.joint_axis[0]
    compile_binding(replace(col.shape, key=key), binding, frame, axis)
    before = binding.zero_motion.start.copy()
    asset.model.joint_position.setflags(write=True)
    asset.model.joint_position[0] += 77
    assert binding.origin_m == (1, 0.25, -0.5)
    np.testing.assert_array_equal(binding.zero_motion.start, before)

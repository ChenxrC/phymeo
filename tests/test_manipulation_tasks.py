"""Synthetic predicate contracts and native static admission, not task success.

Authored without execution. Only the parent may compile or run this module.
"""

from dataclasses import replace
import math
import xml.etree.ElementTree as ET

import pytest

from phymeo.training.manipulation_tasks import (
    H, SuccessFrame, evaluate_goal, footprint_iou, manipulation_scenes,
)


SCENES = manipulation_scenes()


def frame(scene, tick=0, *, position=None, fingers=(0, 0), support=True, **changes):
    values = dict(world=11, episode=3, tick=tick, time_s=tick*H,
                  object_position=scene.initial_object_position if position is None else position,
                  object_wxyz=(1, 0, 0, 0), linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0),
                  finger_normal_forces_n=fingers,
                  support_up_force_n=scene.object_mass_kg*9.81 if support else 0)
    values.update(changes)
    return SuccessFrame(**values)


def test_eight_sources_keep_numeric_geometry_language_and_robot_only_control_separate():
    assert len(SCENES) == 8
    assert [s.robot_kinds for s in SCENES] == ["RRPR", "RRRR", "RRRRR", "RRRRRR", "RRRRRRR", "RR", "RRR", "PPPR"]
    assert len({s.source_sha256 for s in SCENES}) == 8
    for scene in SCENES:
        root = ET.fromstring(scene.mjcf)
        assert root.find("equality") is None and root.find("asset") is None
        assert len(root.findall(".//freejoint")) == 1
        assert root.find(".//freejoint").get("name") == "object-free"
        assert {motor.get("joint") for motor in root.findall("actuator/motor")} == set(scene.robot_joint_names)
        assert set(t.key for t in scene.targets) == {"left", "right", "middle"}
        assert set(g.target_key for g in scene.goals) == {"left", "right", "middle"}
        assert all(len(g.instructions_zh) >= 2 and len(g.instructions_en) >= 2 for g in scene.goals)
        assert all(g.object_key == scene.object_key for g in scene.goals)
        assert scene.object_label_zh and scene.object_label_en
        assert all(shape.kind in ("box", "sphere", "capsule") for shape in scene.object_shapes)
        assert len(scene.robot_joint_names) + 6 <= 35
        assert len(scene.robot_joint_names) + 7 <= 45
        if scene.finger_geom_names:
            assert {g.kind for g in scene.goals} == {"grasp", "place"}
        else:
            assert {g.kind for g in scene.goals} == {"pusht"}
            assert len(scene.object_shapes) == 2
            assert len(root.findall("worldbody/site")) == 6


@pytest.mark.parametrize("scene", SCENES, ids=lambda s: s.key)
def test_native_static_scene_and_interpret_admission(scene):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("tyx")
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    model = mj.MjModel.from_xml_string(scene.mjcf)
    assert model.nv == len(scene.robot_joint_names) + 6
    assert model.nq == model.nv + 1 and model.nu == len(scene.actuator_names)
    assert model.nbody <= 128 and model.neq == 0
    joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "object-free")
    assert model.jnt_type[joint] == mj.mjtJoint.mjJNT_FREE
    body = int(model.jnt_bodyid[joint])
    assert model.body_parentid[body] == 0
    assert model.body_mass[body] == pytest.approx(scene.object_mass_kg, abs=1e-12)
    assert all(model.body_inertia[body] > 0)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert data.ncon == 0, "authoring initial geometry must not start interpenetrating"
    with MuJoCoRuntime(compile_mujoco_xml(scene.mjcf, require_asset=False),
                       mode="interpret", implementation="tyx") as runtime:
        assert runtime.integration_snapshot()[0].tick == 0


def test_success_frame_owns_finite_numeric_state_and_rejects_wrong_phase():
    scene = SCENES[0]
    position = list(scene.initial_object_position)
    sample = frame(scene, position=position)
    position[0] = 100
    assert sample.object_position == scene.initial_object_position
    for updates in (dict(object_wxyz=(2, 0, 0, 0)), dict(linear_velocity=(math.nan, 0, 0)),
                    dict(support_up_force_n=math.inf), dict(finger_normal_forces_n=(-1, 0)),
                    dict(phase="endpoint")):
        with pytest.raises(ValueError):
            frame(scene, **updates)


def test_grasp_needs_actual_lift_load_support_removal_and_uninterrupted_dwell():
    scene = SCENES[0]
    goal = scene.goals[scene.default_goal]
    start = evaluate_goal(scene, goal, frame(scene))
    closed = evaluate_goal(scene, goal, frame(scene, 1, fingers=(1, 1)), start.history)
    assert not closed.success and not closed.history.lifted
    raised = (*scene.initial_object_position[:2], scene.initial_object_position[2] + .04)
    with pytest.raises(ValueError, match="begin"):
        evaluate_goal(scene, goal, frame(scene, position=raised, fingers=(1, 1), support=False))
    false_support = evaluate_goal(scene, goal, frame(scene, 2, position=raised, fingers=(1, 1)), closed.history)
    assert not false_support.history.lifted
    history = false_support.history
    for tick in range(3, 204):
        result = evaluate_goal(scene, goal, frame(scene, tick, position=raised, fingers=(1, 1), support=False), history)
        assert result.success is (tick == 203)
        history = result.history
    lost = evaluate_goal(scene, goal, frame(scene, 204, position=raised, support=False), history)
    assert not lost.success and lost.history.held_s == 0


def test_placement_requires_lift_transport_release_and_real_surface_support():
    scene = SCENES[2]
    goal = scene.goals[scene.default_goal]
    target = next(t for t in scene.targets if t.key == goal.target_key)
    history = evaluate_goal(scene, goal, frame(scene)).history
    # Starting at the destination with support cannot invent prior lifting.
    there = (*target.position[:2], scene.initial_object_position[2])
    without_lift = evaluate_goal(scene, goal, frame(scene, 1, position=there), history)
    assert not without_lift.success and not without_lift.history.lifted
    up = (*scene.initial_object_position[:2], scene.initial_object_position[2] + .04)
    lifted = evaluate_goal(scene, goal, frame(scene, 1, position=up, fingers=(1, 1), support=False), history)
    assert lifted.history.lifted and not lifted.history.transported
    carried = (*target.position[:2], up[2])
    moved = evaluate_goal(scene, goal, frame(scene, 2, position=carried, fingers=(1, 1), support=False), lifted.history)
    assert moved.history.transported
    unsupported = evaluate_goal(scene, goal, frame(scene, 3, position=there, support=False), moved.history)
    assert not unsupported.success and not unsupported.history.holding
    hovering = evaluate_goal(scene, goal, frame(scene, 4, position=carried), unsupported.history)
    assert not hovering.history.holding, "claimed support cannot accept a hovering object"
    history = hovering.history
    for tick in range(5, 206):
        result = evaluate_goal(scene, goal, frame(scene, tick, position=there), history)
        assert result.success is (tick == 205)
        history = result.history


def test_pusht_overlap_uses_actual_union_and_orientation_not_center_or_pusher():
    scene = SCENES[5]
    goal = scene.goals[2]
    target = scene.targets[2]
    assert footprint_iou(scene.object_shapes, target.position, (1, 0, 0, 0), target) == pytest.approx(1, abs=1e-12)
    shifted = (target.position[0] + .01, *target.position[1:])
    # Disjoint-interior stem and crossbar, translated in X by 1 cm.
    area = .036*.09 + .13*.036
    overlap = (.036-.01)*.09 + (.13-.01)*.036
    assert footprint_iou(scene.object_shapes, shifted, (1, 0, 0, 0), target) == pytest.approx(overlap/(2*area-overlap), abs=1e-10)
    assert footprint_iou(scene.object_shapes, target.position, (0, 0, 0, 1), target) < goal.footprint_iou_min
    history = evaluate_goal(scene, goal, frame(scene)).history
    wrong = evaluate_goal(scene, goal, frame(scene, 1, position=shifted), history)
    assert not wrong.history.holding
    for tick in range(2, 203):
        result = evaluate_goal(scene, goal, frame(scene, tick, position=target.position), wrong.history)
        assert result.success is (tick == 202)
        wrong = result


@pytest.mark.parametrize("change", ("world", "episode", "goal", "target_geometry", "tick", "time", "object"))
def test_history_rejects_mixed_identity_or_missing_physical_samples(change):
    scene = SCENES[0]
    goal = scene.goals[0]
    history = evaluate_goal(scene, goal, frame(scene)).history
    sample = frame(scene, 1)
    if change == "goal":
        goal = scene.goals[2]
    elif change == "target_geometry":
        targets = tuple(replace(t, position=(t.position[0] + .01, *t.position[1:]))
                        if t.key == goal.target_key else t for t in scene.targets)
        scene = replace(scene, targets=targets)
    elif change == "time":
        sample = replace(sample, time_s=3*H)
    elif change == "object":
        sample = replace(sample, object_key="other-object")
    else:
        sample = replace(sample, **{change: getattr(sample, change) + 1})
    with pytest.raises(ValueError):
        evaluate_goal(scene, goal, sample, history)


def test_scene_and_target_own_constructor_sequences_and_numeric_scalars():
    class MutableFloat:
        def __init__(self, value):
            self.value = value

        def __float__(self):
            return float(self.value)

    scene = SCENES[0]
    fields = ("object_shapes", "targets", "goals", "robot_joint_names",
              "actuator_names", "finger_geom_names", "initial_object_position", "initial_object_wxyz")
    inputs = {name: list(getattr(scene, name)) for name in fields}
    yaw = MutableFloat(scene.targets[0].yaw)
    target = replace(scene.targets[0], yaw=yaw)
    inputs["targets"][0] = target
    scalars = {name: MutableFloat(getattr(scene, name))
               for name in ("object_mass_kg", "support_top_m", "timestep_s")}
    owned = replace(scene, **inputs, **scalars)
    binding = owned.physical_binding
    for values in inputs.values():
        values.clear()
    for value in scalars.values():
        value.value = math.nan
    yaw.value = math.nan
    assert owned.physical_binding == binding
    for name in fields:
        assert type(getattr(owned, name)) is tuple
        assert getattr(owned, name) == getattr(scene, name)
    for name in scalars:
        assert type(getattr(owned, name)) is float
        assert getattr(owned, name) == getattr(scene, name)
    assert type(target.yaw) is float and target.yaw == scene.targets[0].yaw


def test_scene_rejects_invalid_physical_declarations_and_identity_bindings():
    scene = SCENES[0]
    invalid = (
        dict(key=""), dict(object_key=7), dict(robot_kinds="RRBR"),
        dict(object_mass_kg=0), dict(object_mass_kg=math.inf),
        dict(timestep_s=0), dict(timestep_s=math.nan), dict(support_top_m=math.nan),
        dict(initial_object_position=(math.inf, 0, 0)), dict(initial_object_wxyz=(2, 0, 0, 0)),
        dict(object_shapes=()), dict(object_shapes=(scene.targets[0],)),
        dict(targets=(scene.goals[0],)), dict(goals=(scene.targets[0],)),
        dict(targets=(scene.targets[0], scene.targets[0])),
        dict(robot_joint_names=(*scene.robot_joint_names[:-1], 3)),
        dict(robot_joint_names=(scene.robot_joint_names[0],)*len(scene.robot_joint_names)),
        dict(actuator_names=scene.actuator_names[:-1]), dict(finger_geom_names=("one-finger",)),
        dict(goals=(replace(scene.goals[0], target_key="missing"),)),
        dict(goals=(replace(scene.goals[0], object_key="other-object"),)),
        dict(default_goal=True), dict(default_goal=len(scene.goals)),
    )
    for updates in invalid:
        with pytest.raises(ValueError):
            replace(scene, **updates)
    for updates in (dict(yaw=math.nan), dict(key=""), dict(label_en=3)):
        with pytest.raises(ValueError):
            replace(scene.targets[0], **updates)


def test_history_binds_all_scene_physics_even_when_xml_is_identical():
    scene = SCENES[0]
    goal = scene.goals[0]
    history = evaluate_goal(scene, goal, frame(scene)).history
    replacements = (
        dict(object_shapes=(replace(scene.object_shapes[0], size=(.026,)),)),
        dict(object_mass_kg=scene.object_mass_kg + .01), dict(support_top_m=.001),
        dict(timestep_s=2*H),
        dict(initial_object_position=(scene.initial_object_position[0] + .01, *scene.initial_object_position[1:])),
        dict(initial_object_wxyz=(0, 0, 0, 1)),
        dict(support_geom_name="other-support"), dict(finger_geom_names=("other-left", "other-right")),
    )
    for updates in replacements:
        changed = replace(scene, **updates)
        assert changed.source_sha256 == scene.source_sha256
        assert changed.physical_binding != scene.physical_binding
        with pytest.raises(ValueError, match="history crossed"):
            evaluate_goal(changed, goal, frame(changed, 1), history)


def test_missing_history_cannot_restart_at_a_later_supported_tick():
    scene = SCENES[2]
    goal = scene.goals[scene.default_goal]
    assert not evaluate_goal(scene, goal, frame(scene)).success
    for tick in (1, 20):
        with pytest.raises(ValueError, match="tick 0"):
            evaluate_goal(scene, goal, frame(scene, tick))

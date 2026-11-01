"""The task uses the same cloth/patch program and gates with different robot hardware."""

import numpy as np
import pytest

from phymeo.core.robot import RobotState
from phymeo.core.scene import SceneInputs, StepSettings
from phymeo.experiments.compliant_cloth import trace_start
from phymeo.experiments.compliant_handoff import build_handoff
from phymeo.solvers.compliant_cloth_contact import CompliantClothContactProgram


@pytest.mark.parametrize("layout", ["cartesian", "articulated"])
def test_gripper_layouts_drive_real_robot_state(layout):
    case = build_handoff(robot=layout)
    assert type(case.program) is CompliantClothContactProgram
    targets = {"A": [-0.06, 0.039], "B": [0.115, 0.039]}
    controls = {}
    for name, gripper in case.grippers.items():
        state = case.initial.domains[name]
        before = state.q.copy()
        np.testing.assert_allclose(gripper.position(state), targets[name], atol=1e-7)
        if layout == "articulated":
            assert gripper.model.joint_types == ("revolute", "revolute", "prismatic", "prismatic")
        inp, goal = gripper.command(state, np.array(targets[name]) + [0.0, 0.012], 0.002, case.dt)
        np.testing.assert_array_equal(state.q, before)
        assert np.all(np.abs(inp.effort) <= gripper.effort_limits)
        assert np.shares_memory(state.q, goal) is False
        controls[name] = inp
    data = SceneInputs(dict(cloth=case.inputs.domains["cloth"], **controls))
    with case.program.stepper(
        StepSettings(substeps=4, position_iterations=1, velocity_iterations=1)
    ) as runtime:
        result = runtime.step(case.initial, data, 0.001)
    assert result.report.accepted, result.report.reason
    for name in controls:
        before = case.initial.domains[name]
        after = result.state.domains[name]
        assert np.max(abs(after.q - before.q)) > 0.0 and np.max(abs(after.q - before.q)) < 0.002
        assert np.linalg.norm(after.v) > 1e-4


def test_finger_motor_law_is_independent_of_the_arm_it_is_attached_to():
    # Same spring, damping and limit about the measured jaw positions, even under load.
    expected = [2.53, 5.0]
    for layout in ("cartesian", "articulated"):
        g = build_handoff(robot=layout).grippers["A"]
        q = g.initial.q.copy()
        q[list(g.fingers)] = [0.0102, 0.0]
        state = RobotState(q, np.zeros_like(q))
        drive, _ = g.command(state, g.position(state), 0.0124, 0.01)
        np.testing.assert_allclose(drive.effort[list(g.fingers)], expected, atol=1e-12)


def test_receiving_arm_reaches_the_cloth_workspace_before_closing():
    case = build_handoff(robot="articulated")
    gripper = case.grippers["B"]
    # A suspended 12-cm-wide cloth can swing to either side of the donor.
    for x in (-0.17, 0.25):
        for z in (0.025, 0.28):
            _, goal = gripper.command(gripper.initial, np.array([x, z]), 0.0, case.dt)
            np.testing.assert_allclose(
                gripper.position(RobotState(goal, np.zeros_like(goal))), [x, z], atol=1e-7
            )


def test_trace_carries_the_robot_and_finite_patch_bindings():
    case = build_handoff(robot="articulated")
    trace = trace_start(case, "independent_name")
    for key, g in case.grippers.items():
        stored = trace["robot_models"][key]
        assert list(stored["joint_types"]) == list(g.model.joint_types)
        np.testing.assert_equal(stored["joint_position"], g.model.joint_position)
    for row, patch in zip(trace["patches"], case.program.patches, strict=True):
        assert row["link_index"] == patch.geometry.link_index
        np.testing.assert_equal(row["local_vertices_m"], patch.geometry.local_vertices)


def test_invalid_layout_fails_before_simulation():
    with pytest.raises(ValueError, match="robot"):
        build_handoff(robot="undeclared")


def test_handoff_owns_limits_structural_shapes_and_records_their_loads():
    case = build_handoff(robot="articulated")
    system = case.program.robot_contact
    assert system is not None, "handoff is missing body collisions and stops"
    assert set(system.limits) == {"A", "B"}
    assert len(system.capsules) == 6
    assert len(system.planes) == 1
    report = system.evaluate(case.initial.domains)
    assert not report["contacts"], "initial robot bodies intersect"
    trace = trace_start(case, "shaped_arm")
    assert len(trace["robot_contact"]["capsules"]) == 6
    assert set(trace["frames"][0]["body_capsules_m"]) == {c.key for c in system.capsules}
    for name, g in case.grippers.items():
        assert g.limits is system.limits[name]


def test_recorded_collision_bodies_are_bound_to_physical_robot_models():
    from copy import deepcopy
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location(
        "body_exporter", Path(__file__).resolve().parents[1] / "examples/export_compliant_replay.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = trace_start(build_handoff(robot="articulated"), "unrelated_task_name")
    broken = deepcopy(trace)
    broken["frames"][0]["body_capsules_m"]["A_upper"][0][2] += 0.1
    with pytest.raises(ValueError, match="capsule"):
        module.add_link_geometry(broken)
    valid = deepcopy(trace)
    module.add_link_geometry(valid)
    assert valid["frames"] == trace["frames"]

"""Accepted-step feedback and actual motion for a bounded, asset-independent gripper."""

from dataclasses import replace
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.runtime.robot_scene import RobotCommand

SOURCE = Path(__file__).parent / "reference_newton" / "robot" / "mimic_jaws.urdf"
H = 0.001


def _scene(*, q=0.0, center_x=0.25):
    builder = ContactSceneBuilder()
    pads = tuple(
        PadFace.named(side, side + "_rubber", 0, sign, tuple(product((-0.005, 0.005), repeat=2)))
        for side, sign in (("left", 1), ("right", -1))
    )
    builder.add_urdf(
        10,
        SOURCE,
        loader=load_mimic_urdf,
        robot_state=RobotState(np.array([0.0, q]), np.zeros(2)),
        kp=[2700, 6000],
        kd=[95, 35],
        effort_limit=[40, 8],
        pads=pads,
    )
    builder.add_box(20, (0.02, 0.02, 0.02), mass=0.1, position=(center_x, -0.125, 0.25))
    builder.add_sphere(30, 0.015, mass=0.03, position=(0.33, -0.125, 0.245))
    builder.add_box(40, (0.12, 0.12, 0.01), mass=None, position=(0.25, -0.125, 0.22))
    for a, b in combinations((10, 20, 30, 40), 2):
        law = (
            RobotContactMaterial(1200, 1.2, 0.65, 0.04)
            if (a, b) == (10, 20)
            else (
                RobotContactMaterial(2200, 3, 0.45, 0.04)
                if b == 40 and a != 10
                else RobotContactMaterial(6000, 16, 0.35, 0.04)
            )
        )
        builder.pair(a, b, law)
    return builder.build()


def _controller(scene, **overrides):
    from phymeo.control.gripper_acquisition import GripperAcquisition

    options = dict(
        robot=(0, 10),
        payload_key=scene.body_keys[0, 20][0],
        pad_groups=tuple(
            (scene.shape_key(10, side, side + "_rubber"),) for side in ("left", "right")
        ),
        joint_indices=(1,),
        open_q=(0,),
        closed_q=(0.018,),
        speed_limit=(0.02,),
        effort_limit=(40, 8),
        force_target_N=2.0,
        force_band_N=0.25,
        contact_min_N=1.0,
        dwell_s=0.020,
        loss_dwell_s=0.010,
        stop_speed=(0.002,),
        timeout_s=1.5,
    )
    options.update(overrides)
    return GripperAcquisition(scene.assets[0, 10], **options)


def test_close_proposal_tracks_accepted_steps_without_teleport_or_repeat_accumulation():
    with _scene() as scene:
        controller = _controller(scene)
        before = scene.observe()
        first = controller.propose(before, dt_s=H)
        repeat = controller.propose(before, dt_s=H)
        assert first.phase == "closing" and not first.contact_ready
        np.testing.assert_array_equal(first.bilateral_N, [0, 0])
        np.testing.assert_allclose(first.command.target_q, [0, 0.000020], atol=1e-18, rtol=0)
        np.testing.assert_allclose(first.command.target_velocity, [0, 0.02], atol=1e-18, rtol=0)
        np.testing.assert_array_equal(first.command.effort_limit, [40, 8])
        np.testing.assert_array_equal(repeat.command.target_q, first.command.target_q)
        np.testing.assert_array_equal(repeat.command.target_velocity, first.command.target_velocity)
        assert scene.observe().step_number == 0 and scene.last_frame is None
        np.testing.assert_array_equal(scene.observe().robots[0, 10].q, before.robots[0, 10].q)
        with pytest.raises(ValueError):
            scene.step(-H, actions={10: first.command})
        retry = controller.propose(scene.observe(), dt_s=H)
        np.testing.assert_array_equal(retry.command.target_q, first.command.target_q)
        after = scene.step(H, actions={10: retry.command})
        joint = after.robots[0, 10]
        assert joint.requested_command is not None
        assert joint.requested_command.feedforward is None
        np.testing.assert_array_equal(joint.requested_command.target_q, retry.command.target_q)
        np.testing.assert_allclose(joint.command.feedforward, [1.5 * 9.81, 0], atol=1e-12)
        with pytest.raises(ValueError):
            joint.requested_command.target_q.setflags(write=True)
        # The command is a target, not a state write. The real inertia advances by less.
        assert 0 < after.robots[0, 10].q[1] < first.command.target_q[1]
        assert after.robots[0, 10].v[1] > 0
        second = controller.propose(after, dt_s=H)
        np.testing.assert_allclose(second.command.target_q, [0, 0.000040], atol=1e-18, rtol=0)
        again = controller.propose(after, dt_s=H)
        np.testing.assert_array_equal(second.command.target_q, again.command.target_q)
        assert scene.observe().step_number == 1 and scene.observe().time_s == H
        altered_request = RobotCommand(
            retry.command.target_q,
            target_velocity=retry.command.target_velocity,
            effort_limit=retry.command.effort_limit,
            feedforward=[0, 0],
        )
        altered = replace(
            after,
            robots={
                (0, 10): replace(joint, requested_command=altered_request),
            },
        )
        with pytest.raises(ValueError, match="same accepted step"):
            controller.propose(altered, dt_s=H)
        np.testing.assert_array_equal(
            controller.propose(after, dt_s=H).command.target_q, second.command.target_q
        )


def test_future_rollback_and_changed_same_step_are_rejected_without_consuming_history():
    with _scene() as scene:
        controller = _controller(scene)
        first = controller.propose(scene.observe(), dt_s=H)
        after = scene.step(H, actions={10: first.command})
        next_step = controller.propose(after, dt_s=H)
        for bad in (
            replace(after, load_time_s=after.time_s + H),
            replace(after, time_s=after.time_s + H),
            replace(after, step_number=0, time_s=0, load_time_s=None),
            replace(after, step_number=2, time_s=4 * H, load_time_s=H),
            replace(after, step_number=3, time_s=3 * H, load_time_s=2 * H),
        ):
            with pytest.raises(ValueError):
                controller.propose(bad, dt_s=H)
        retry = controller.propose(after, dt_s=H)
        np.testing.assert_array_equal(retry.command.target_q, next_step.command.target_q)
        assert retry.contact_duration_s == next_step.contact_duration_s == 0
        assert scene.observe().step_number == 1


def test_zero_motor_cap_is_physical_and_the_reference_does_not_wind_up_to_closed():
    with _scene() as scene:
        controller = _controller(scene, effort_limit=(40, 0))
        references = []
        for _ in range(12):
            result = controller.propose(scene.observe(), dt_s=H)
            references.append(result.command.target_q[1])
            assert result.command.effort_limit[1] == 0
            after = scene.step(H, actions={10: result.command})
            assert after.robots[0, 10].drive_effort[1] == 0
            assert after.robots[0, 10].q[1] == after.robots[0, 10].v[1] == 0
            assert not result.contact_ready
        assert references[-1] == references[-2] == references[-3]
        assert max(references) < 0.0002


def test_negative_closing_coordinate_produces_real_reverse_motion():
    with _scene() as scene:
        controller = _controller(scene, closed_q=(-0.004,))
        result = controller.propose(scene.observe(), dt_s=H)
        np.testing.assert_allclose(result.command.target_q, [0, -0.000020], atol=1e-18)
        np.testing.assert_allclose(result.command.target_velocity, [0, -0.02], atol=1e-18)
        after = scene.step(H, actions={10: result.command})
        assert result.command.target_q[1] < after.robots[0, 10].q[1] < 0
        assert after.robots[0, 10].v[1] < 0


def test_real_one_sided_load_and_wrong_payload_do_not_certify_bilateral_contact():
    with _scene(q=0.0125, center_x=0.2506) as scene:
        frame = scene.step(H, actions={10: RobotCommand([0, 0.0125])})
        controller = _controller(scene)
        result = controller.propose(frame, dt_s=H)
        np.testing.assert_allclose(result.bilateral_N, [0, 5.28], atol=2e-10, rtol=0)
        assert not result.contact_ready and result.phase == "closing"
        # The loaded side is already beyond the target band; do not squeeze harder.
        assert result.command.target_q[1] < 0.0125
        swapped = _controller(scene, pad_groups=controller.pad_groups[::-1])
        np.testing.assert_allclose(
            swapped.propose(frame, dt_s=H).bilateral_N, [5.28, 0], atol=2e-10
        )
        wrong = _controller(scene, payload_key=scene.body_keys[0, 30][0])
        other = wrong.propose(frame, dt_s=H)
        np.testing.assert_array_equal(other.bilateral_N, [0, 0])
        assert not other.contact_ready
        for value in (-1.0, float("nan")):
            invalid = tuple(dict(row, normal_load_N=value) for row in frame.contacts)
            with pytest.raises(ValueError):
                controller.propose(replace(frame, contacts=invalid), dt_s=H)
        np.testing.assert_array_equal(
            controller.propose(frame, dt_s=H).command.target_q, result.command.target_q
        )


def test_actual_balanced_contact_dwell_then_extraction_reports_loss():
    # Each side has 4*k*delta=2 N. The 0.5 kg reduced screw inertia and original
    # implicit-free PD give tau=kp*(qref-q)/(1+h*(h*kp+kd)/M), hence this qref.
    q = 0.012 + 2 / (4 * 1200)
    reference = q + 4 * (1 + H * (H * 6000 + 35) / 0.5) / 6000
    with _scene(q=q) as scene:
        frame = scene.step(H, actions={10: RobotCommand([0, reference])})
        controller = _controller(scene)
        first = controller.propose(frame, dt_s=H)
        assert not first.contact_ready and first.contact_duration_s == 0
        # This analytic balance is only the initial scalar closing-axis state.
        # The freely falling body and finite support are not in static equilibrium.
        assert first.load_time_s == 0
        np.testing.assert_allclose(first.bilateral_N, [2, 2], atol=2e-8)
        for _ in range(25):
            decision = controller.propose(scene.observe(), dt_s=H)
            scene.step(H, actions={10: decision.command})
        ready = controller.propose(scene.observe(), dt_s=H)
        assert ready.contact_ready and ready.phase == "force_hold"
        assert ready.contact_duration_s >= 0.020
        # After actual coupled steps use the predeclared force band, not the
        # initial analytic force as an unproved invariant of the moving scene.
        assert np.all(abs(ready.bilateral_N - 2) <= 0.25)
        assert abs(scene.observe().robots[0, 10].v[1]) <= 0.002
        external = scene.inputs()
        payload_row = scene.rigid_rows[0, 20]
        external.rigid.force[payload_row, 2] = 10
        height = scene.observe().rigid.position[payload_row, 2]
        lost = None
        for _ in range(70):
            decision = controller.propose(scene.observe(), dt_s=H)
            if decision.phase == "lost":
                lost = decision
                break
            scene.step(H, actions={10: decision.command}, external=external)
        assert lost is not None and lost.reason == "bilateral_load_lost"
        assert lost.loss_duration_s >= 0.010 and not lost.contact_ready
        assert scene.observe().rigid.position[payload_row, 2] > height + 0.02


def test_release_requires_actual_open_zero_load_dwell_and_results_are_immutable():
    with _scene() as scene:
        controller = _controller(scene)
        before = scene.observe()
        first = controller.propose(before, dt_s=H, intent="release")
        assert first.phase == "releasing"
        for _ in range(25):
            result = controller.propose(scene.observe(), dt_s=H, intent="release")
            scene.step(H, actions={10: result.command})
        released = controller.propose(scene.observe(), dt_s=H, intent="release")
        assert released.phase == "released" and not released.contact_ready
        np.testing.assert_array_equal(released.bilateral_N, [0, 0])
        np.testing.assert_allclose(scene.observe().robots[0, 10].q, [0, 0], atol=1e-12, rtol=0)
        with pytest.raises(ValueError):
            released.bilateral_N.setflags(write=True)
        with pytest.raises(ValueError):
            released.command.target_q.setflags(write=True)


@pytest.mark.parametrize(
    "change",
    [
        {"effort_limit": (40, 10)},
        {"speed_limit": (0.3,)},
        {"closed_q": (0,)},
        {"closed_q": (0.05,)},
        {"joint_indices": (True,)},
        {"force_band_N": (2,)},
        {"force_target_N": 0.5},
        {"dwell_s": 0},
    ],
)
def test_invalid_protocol_cannot_bypass_asset_or_force_constraints(change):
    with _scene() as scene:
        with pytest.raises((TypeError, ValueError)):
            _controller(scene, **change)


def test_changed_feedforward_is_not_accepted_as_the_controllers_own_request():
    with _scene() as scene:
        controller = _controller(scene)
        proposal = controller.propose(scene.observe(), dt_s=H)
        issued = proposal.command
        assert issued.feedforward is None
        changed = RobotCommand(
            issued.target_q,
            target_velocity=issued.target_velocity,
            effort_limit=issued.effort_limit,
            feedforward=[0, 1],
        )
        after = scene.step(H, actions={10: changed})
        assert after.step_number == 1
        np.testing.assert_array_equal(after.robots[0, 10].command.feedforward, [0, 1])
        np.testing.assert_array_equal(after.robots[0, 10].requested_command.feedforward, [0, 1])
        with pytest.raises(ValueError, match="command|request"):
            controller.propose(after, dt_s=H)


def test_small_empty_stroke_terminates_after_real_unloaded_closure():
    with _scene() as scene:
        controller = _controller(scene, closed_q=(0.0004,))
        end = None
        for _ in range(200):
            proposal = controller.propose(scene.observe(), dt_s=H)
            if proposal.phase == "empty":
                end = proposal
                break
            scene.step(H, actions={10: proposal.command})
        assert end is not None and end.reason == "closure_limit_without_bilateral_load"
        actual = scene.observe().robots[0, 10]
        assert abs(actual.q[1] - 0.0004) <= 0.0001 and abs(actual.v[1]) <= 0.002
        np.testing.assert_array_equal(end.bilateral_N, [0, 0])
        assert not end.contact_ready and scene.observe().step_number > 0


def test_timeout_uses_accepted_physical_time_with_a_real_stalled_zero_motor():
    with _scene() as scene:
        controller = _controller(scene, effort_limit=(40, 0), timeout_s=0.010)
        end = None
        for _ in range(20):
            proposal = controller.propose(scene.observe(), dt_s=H)
            if proposal.phase == "timeout":
                end = proposal
                break
            # Calling again cannot consume the timeout budget.
            repeated = controller.propose(scene.observe(), dt_s=H)
            assert repeated.phase == proposal.phase
            scene.step(H, actions={10: repeated.command})
        assert end is not None and end.reason == "acquisition_timeout"
        assert end.time_s == pytest.approx(0.010, abs=1e-14)
        actual = scene.observe().robots[0, 10]
        assert actual.q[1] == actual.v[1] == actual.drive_effort[1] == 0
        assert not end.contact_ready

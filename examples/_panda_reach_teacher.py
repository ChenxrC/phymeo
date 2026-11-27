"""Training reference for the free-pin reaching task's existing motor interface.

Solve from each supplied physical reset. The direct teacher uses the fixed
final solution. The waypoint teacher also checks the measured hand and pin
poses before descending, without a phase clock.
"""

import math

import numpy as np

from _panda_grasp_insert_control import _bounded, _rotation_vector
from _panda_grasp_insert_scene import OPEN_JAW_M


_APPROACH_OFFSETS_M = (.075, .050, .025, 0.)


def _motor_limits(mj, model):
    if (model.nq, model.nv, model.nu) != (23, 22, 16):
        raise ValueError("expected the dual-Panda free-pin model with 16 motors")
    names = [f"{side}_panda_joint{i}" for side in ("left", "right") for i in range(1, 8)]
    names.extend(("right_panda_finger_joint1", "right_panda_finger_joint2"))
    limits = np.empty((16, 2))
    for i, name in enumerate(names):
        j = model.joint(name).id
        kind = mj.mjtJoint.mjJNT_HINGE if i < 14 else mj.mjtJoint.mjJNT_SLIDE
        if (model.jnt_type[j] != kind or model.jnt_qposadr[j] != i or model.jnt_dofadr[j] != i
                or not model.jnt_limited[j] or model.actuator_trnid[i, 0] != j
                or model.actuator_trntype[i] != mj.mjtTrn.mjTRN_JOINT
                or not np.array_equal(model.actuator_gear[i], (1., 0., 0., 0., 0., 0.))):
            raise ValueError(f"unexpected joint address, type, limits or motor transmission for {name}")
        limits[i] = model.jnt_range[j]
    if (not np.isfinite(limits).all() or np.any(limits[:14, 1] - limits[:14, 0] <= .002)
            or np.any(limits[14:, 0] > OPEN_JAW_M) or np.any(limits[14:, 1] < OPEN_JAW_M)):
        raise ValueError("invalid Panda joint ranges or unreachable 35 mm finger opening")
    return limits


def build_targets(model, reset_qpos):
    """Return [world,16] FP32 absolute motor targets, verified after rounding."""
    return _build_targets(model, reset_qpos, (0.,))[:, 0]


def build_waypoints(model, reset_qpos):
    """Return [world,4,16] targets at 75, 50, 25 and 0 mm along pin-local -Z."""
    return _build_targets(model, reset_qpos, _APPROACH_OFFSETS_M)


def _build_targets(model, reset_qpos, offsets):
    import mujoco as mj

    resets = np.asarray(reset_qpos, dtype=np.float64)
    if resets.ndim != 2 or resets.shape[0] == 0 or resets.shape[1] != 23 or not np.isfinite(resets).all():
        raise ValueError("reset_qpos must be a finite, nonempty [world,23] array")
    limits = _motor_limits(mj, model)
    if np.any(resets[:, :16] < limits[:, 0]) or np.any(resets[:, :16] > limits[:, 1]):
        raise ValueError("reset outside the physical motor joint limits")
    if np.any(np.abs(np.linalg.norm(resets[:, 19:23], axis=1) - 1.) > 1e-5):
        raise ValueError("reset free-pin quaternion must be unit wxyz")
    # Match the actual 1 mrad buffer used by DeviceTask when it sends arm targets.
    low, high = limits[:14, 0] + .001, limits[:14, 1] - .001
    if np.any(resets[:, :7] < low[:7]) or np.any(resets[:, :7] > high[:7]):
        raise ValueError("left arm reset cannot be held within the controller target limits")
    hand = model.site("right_grasp_center").id
    pin = model.site("part_grasp_center").id
    part = model.body("assembly_part").id
    free = model.joint("assembly_part_free").id
    if (model.site_bodyid[hand] != model.body("right_panda_hand").id or model.site_bodyid[pin] != part
            or model.body_parentid[part] != 0 or model.jnt_bodyid[free] != part
            or model.jnt_type[free] != mj.mjtJoint.mjJNT_FREE
            or model.jnt_qposadr[free] != 16 or model.jnt_dofadr[free] != 16):
        raise ValueError("unexpected grasp sites or free-pin coordinates")

    solve, verify = mj.MjData(model), mj.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    targets = np.empty((len(resets), len(offsets), 16), np.float32)
    length_scale = .12
    damping = np.eye(6) * 2e-5
    for w, reset in enumerate(resets):
        solve.qpos[:] = reset
        mj.mj_kinematics(model, solve)
        pin_p = solve.site_xpos[pin].copy()
        goal_r = solve.site_xmat[pin].reshape(3, 3).copy()
        angles = solve.qpos[7:14]
        np.clip(angles, low[7:], high[7:], out=angles)
        for j, offset in enumerate(offsets):
            # angles is a view of solve.qpos, retaining the previous IK solution.
            goal_p = pin_p - offset * goal_r[:, 2]
            for _ in range(128):
                mj.mj_kinematics(model, solve)
                current_r = solve.site_xmat[hand].reshape(3, 3)
                translation = goal_p - solve.site_xpos[hand]
                rotation = _rotation_vector(mj, goal_r @ current_r.T)
                error = np.r_[translation, length_scale * rotation]
                if not np.isfinite(error).all():
                    raise ValueError(f"nonfinite pickup IK residual in world {w}, waypoint {j}")
                if np.linalg.norm(translation) < 1e-5 and np.linalg.norm(rotation) < 1e-4:
                    break
                mj.mj_comPos(model, solve)
                mj.mj_jacSite(model, solve, jp, jr, hand)
                jac = np.vstack((jp[:, 7:14], length_scale * jr[:, 7:14]))
                increment = jac.T @ np.linalg.solve(jac @ jac.T + damping, error)
                if not np.isfinite(increment).all():
                    raise ValueError(f"nonfinite pickup IK increment in world {w}, waypoint {j}")
                angles[:] = np.clip(angles + _bounded(increment, .12), low[7:], high[7:])

            target = targets[w, j]
            target[:7], target[7:14], target[14:] = reset[:7], angles, OPEN_JAW_M
            # Independent native FK checks the rounded joint target against
            # the requested displacement expressed in the measured pin frame.
            verify.qpos[:] = reset
            verify.qpos[:16] = target
            mj.mj_kinematics(model, verify)
            verified_pin_r = verify.site_xmat[pin].reshape(3, 3)
            local_delta = verified_pin_r.T @ (verify.site_xpos[hand] - verify.site_xpos[pin])
            distance = np.linalg.norm(local_delta - [0., 0., -offset])
            relative = verify.site_xmat[hand].reshape(3, 3).T @ verified_pin_r
            angle = math.acos(float(np.clip((np.trace(relative) - 1.) / 2., -1., 1.)))
            if not (np.isfinite(target).all() and distance < .001 and angle < math.radians(1.)):
                raise ValueError(f"world {w} waypoint {j} IK verification failed: {distance*1000:.4f} mm, "
                                 f"{math.degrees(angle):.4f} degrees")
    return targets


def teacher_action(torch, task, targets):
    """Generate one bounded action for DeviceTask.step, on the targets' device.

    Upload build_targets' result once before the rollout. Arms may move their
    commanded target by up to 0.010 rad per policy decision. The two finger
    actions encode absolute opening and can therefore reach +1 immediately.
    """
    current = task.target
    if (not isinstance(targets, torch.Tensor) or current.ndim != 2 or current.shape[1] != 16
            or targets.shape != current.shape or targets.device != current.device
            or targets.dtype != torch.float32 or current.dtype != torch.float32):
        raise ValueError("expected matching [world,16] FP32 teacher and motor targets on the same device")
    with torch.no_grad():
        valid = torch.isfinite(targets) & torch.isfinite(current)
        arms = ((targets[:, :14] - current[:, :14]) / .02).clamp(-.5, .5)
        jaws = (targets[:, 14:] / .0175 - 1.).clamp(-1., 1.)
        action = torch.cat((arms, jaws), dim=1)
        # Preserve invalid input through clipping for DeviceTask's finite gate
        # and the solver's atomic control rejection, without a host scalar read.
        return torch.where(valid, action, torch.full_like(action, math.nan))


def waypoint_action(torch, task, targets, stage):
    """Update each world's int64 stage in place and return the sixteen motor actions.

    Start stage at zero on reset. Stages advance at most once per call. For
    each waypoint the measured TCP must be within 5 mm laterally, 7.5 mm
    axially and 5 degrees of the pin orientation. Also require a right-arm
    joint error below .06 rad and at least .030 m of actual opening per jaw.
    The last stage holds the grip center with the fingers open.
    """
    return _waypoint_action(torch, task, targets, stage, recoverable=False)


def recoverable_waypoint_action(torch, task, targets, stage):
    """Rejoin the axial descent from an observed, aligned, open-hand pose.

    Use the pin's measured positive (downward) axis. A TCP above its center
    has positive height in the hand-to-pin displacement. The recovery tube
    has radius 5 mm, height [-7.5, 82.5] mm, and a 5 degree orientation limit;
    each measured finger opening must be at least 30 mm. At heights up to
    82.5, 57.5 and 32.5 mm, respectively, the next target is 50, 25 or 0 mm.
    This includes the original 7.5 mm axial arrival tolerance. Stage never
    decreases. Within 7.5 mm of zero height, hold the measured redundant arm
    configuration. All commands use the existing bounded motor interface.
    """
    return _waypoint_action(torch, task, targets, stage, recoverable=True)


def _waypoint_action(torch, task, targets, stage, *, recoverable):
    current = task.target
    if (not isinstance(targets, torch.Tensor) or not isinstance(stage, torch.Tensor)
            or current.ndim != 2 or current.shape[1] != 16 or current.dtype != torch.float32
            or targets.shape != (current.shape[0], 4, 16) or targets.dtype != current.dtype
            or stage.shape != (current.shape[0],) or stage.dtype != torch.int64
            or targets.device != current.device or stage.device != current.device):
        raise ValueError("expected [world,4,16] FP32 waypoints and [world] int64 stage on the motor device")
    with torch.no_grad():
        valid_stage = (stage >= 0) & (stage <= 3)
        index = stage.clamp(0, 3)
        rows = torch.arange(current.shape[0], device=current.device)
        _, (hand_p, _), (pin_p, pin_r) = task.reach.poses(task.q[:, :23])
        pin_axis = pin_r[:, :, 2]
        hand_to_pin = pin_p - hand_p
        axial = (hand_to_pin * pin_axis).sum(-1)
        lateral = hand_to_pin - axial[:, None] * pin_axis
        angular = task.reach_measurement["orientation_rad"]
        aligned = ((lateral.square().sum(-1) <= .005**2) & (angular <= math.radians(5.))
                   & (task.q[:, 14:16] >= .030).all(-1))
        if recoverable:
            in_tube = valid_stage & aligned & (axial >= -.0075) & (axial <= .0825)
            next_stage = 1 + (axial <= .0575).to(torch.int64) + (axial <= .0325).to(torch.int64)
            stage.copy_(torch.where(in_tube, torch.maximum(stage, next_stage), stage))
        else:
            height = (3 - index).to(current.dtype) * .025
            selected = targets[rows, index]
            reached = (aligned & ((axial - height).abs() <= .0075)
                       & ((task.q[:, 7:14] - selected[:, 7:14]).abs().amax(-1) <= .06))
            advance = valid_stage & (index < 3) & reached
            stage.add_(advance.to(torch.int64))
        selected = targets[rows, stage.clamp(0, 3)]

        error = selected - current
        left = (error[:, :7] / .02).clamp(-.9, .9)
        right = error[:, 7:14]
        if recoverable:
            at_endpoint = in_tube & (axial.abs() <= .0075)
            right = torch.where(at_endpoint[:, None], task.q[:, 7:14] - current[:, 7:14], right)
        # Move along the joint-space segment, avoiding seven independent
        # saturations that would distort the path through the pickup fixture.
        right = .9 * right / right.abs().amax(-1, keepdim=True).clamp_min(.018)
        jaws = (selected[:, 14:] / .0175 - 1.).clamp(-1., 1.)
        action = torch.cat((left, right, jaws), dim=1)
        valid = valid_stage[:, None] & torch.isfinite(selected) & torch.isfinite(current)
        return torch.where(valid, action, torch.full_like(action, math.nan))

"""Pickup feedback on accepted dual-Panda q23/v22 using the reach policy's action16."""

import math

import mujoco as mj
import numpy as np

from _panda_grasp_insert_control import ARM_SPEED_RAD_S, OPENING_PER_FINGER_M, _bounded, _rotation_vector
from _panda_grasp_insert_scene import PAD_NAMES
from _panda_reach_teacher import _motor_limits


class PickupTeacher:
    """One teacher per world. Call once per decision with the runner's current absolute motor target.

    The left arm holds its handoff coordinates. Right-arm targets change by at most 0.02 rad per
    decision. Each finger action selects an absolute opening between zero and 35 mm. The parent
    runner owns physics, including initial qvel, targets, motor efforts and negative-control replay.
    """

    def __init__(self, model, initial_qpos, *, decision_dt=.016, lift_height_m=.06):
        if not all(math.isfinite(x) and x > 0 for x in (decision_dt, lift_height_m)):
            raise ValueError("positive finite decision interval and lift height required")
        initial = np.asarray(initial_qpos, dtype=float)
        if initial.shape != (23,) or not np.isfinite(initial).all():
            raise ValueError("expected finite physical handoff q23")
        self.model, self.data = model, mj.MjData(model)
        self.dt, self.lift_height = decision_dt, lift_height_m
        self.limits = _motor_limits(mj, model)
        self.hand, self.pin = (model.site(n).id for n in ("right_grasp_center", "part_grasp_center"))
        self.pads = [model.geom(n).id for n in PAD_NAMES]
        self.part = model.body("assembly_part").id
        self.left_hold = initial[:7].copy()
        self.phase = "align"
        self.close_pose = None
        self.lift_origin = None
        self.lift_command = 0.
        self.opening = OPENING_PER_FINGER_M
        self.contact_samples = 0

    def _pad_contacts(self):
        mj.mj_collision(self.model, self.data)
        touched = set()
        for c in self.data.contact[:self.data.ncon]:
            if c.dist > 0.:
                continue
            a, b = map(int, c.geom)
            if a in self.pads and self.model.geom_bodyid[b] == self.part:
                touched.add(a)
            if b in self.pads and self.model.geom_bodyid[a] == self.part:
                touched.add(b)
        return len(touched) == 2

    def action(self, qpos, qvel, motor_target):
        """Return one normalized FP32 action. Read q/v without modifying the caller's state."""
        q, v, motor_target = [np.asarray(x, dtype=float) for x in (qpos, qvel, motor_target)]
        if (q.shape != (23,) or v.shape != (22,) or motor_target.shape != (16,)
                or not np.isfinite(np.r_[q, v, motor_target]).all()):
            raise ValueError("expected finite q23/v22 and the current 16 absolute motor targets")
        data = self.data
        data.qpos[:], data.qvel[:] = q, v
        mj.mj_kinematics(self.model, data)
        mj.mj_comPos(self.model, data)
        hand, pin = data.site_xpos[[self.hand, self.pin]]
        hand_r, pin_r = data.site_xmat[[self.hand, self.pin]].reshape(2, 3, 3)
        axis = pin_r[:, 2]
        error = pin - hand
        axial = float(axis @ error)
        lateral = np.linalg.norm(error - axial * axis)
        angle = np.linalg.norm(_rotation_vector(mj, pin_r @ hand_r.T))
        jp, jr = np.empty((3, 22)), np.empty((3, 22))
        mj.mj_jacSite(self.model, data, jp, jr, self.hand)
        pin_jp = np.empty_like(jp)
        mj.mj_jacSite(self.model, data, pin_jp, None, self.pin)
        relative_speed = np.linalg.norm((pin_jp - jp) @ v)
        opened = np.min(q[14:16]) >= .029
        aligned = lateral <= .003 and angle <= math.radians(4.)
        if self.phase == "align" and opened and aligned and -.004 <= axial <= .085:
            self.phase = "descend"
        if self.phase == "descend" and (lateral > .006 or angle > math.radians(8.)):
            self.phase = "align"

        target, rotation = pin.copy(), pin_r.copy()
        if self.phase == "align":
            if not opened:
                target, rotation = hand.copy(), hand_r.copy()
            elif axial < .030 and not aligned:
                # Withdraw along the pin axis before sweeping a misaligned pad over the fork.
                target, rotation = hand - .035 * axis, hand_r.copy()
            else:
                target = pin - .045 * axis
        if self.phase == "descend" and opened and aligned and abs(axial) <= .003 and relative_speed <= .035:
            self.phase = "close"
            self.close_pose = (pin.copy(), pin_r.copy())
            self.opening = min(OPENING_PER_FINGER_M, float(np.mean(q[14:16])))
        if self.phase == "close":
            target, rotation = self.close_pose
            self.opening = max(0., self.opening - .070 * self.dt)
            bilateral = self._pad_contacts()
            self.contact_samples = self.contact_samples + 1 if bilateral else 0
            # Allow the commanded jaws to develop squeeze after first geometric contact.
            if (self.contact_samples - 1) * self.dt >= .032 and self.opening <= .006:
                self.phase = "lift"
                self.lift_origin = hand.copy()
        if self.phase in ("lift", "hold"):
            self.opening = 0.
            # Bound both the requested lift rate and lead above the achieved wrist height.
            achieved = float(hand[2] - self.lift_origin[2])
            self.lift_command = min(self.lift_height, self.lift_command + .080 * self.dt, max(0., achieved + .008))
            target = self.lift_origin + np.array([0., 0., self.lift_command])
            rotation = self.close_pose[1]
            if achieved >= self.lift_height - .003:
                self.phase = "hold"

        # Resolved-rate damped IK about the observed state, using the same 0.12 m
        # rotation scale and damping as the original motor-only grasp controller.
        translation = _bounded(target - hand, .012)
        rotvec = _bounded(_rotation_vector(mj, rotation @ hand_r.T), .09)
        jac = np.vstack((jp[:, 7:14], .12 * jr[:, 7:14]))
        increment = jac.T @ np.linalg.solve(jac @ jac.T + 2e-5 * np.eye(6), np.r_[translation, .12 * rotvec])
        right = np.clip(q[7:14] + increment, self.limits[7:14, 0] + .001, self.limits[7:14, 1] - .001)
        delta = right - motor_target[7:14]
        bound = np.minimum(.02, ARM_SPEED_RAD_S * self.dt)
        delta /= max(1., float(np.max(np.abs(delta) / bound)))
        action = np.empty(16, dtype=np.float32)
        action[:7] = np.clip((self.left_hold - motor_target[:7]) / .02, -1., 1.)
        action[7:14] = np.clip(delta / .02, -1., 1.)
        action[14:] = np.clip(self.opening / .0175 - 1., -1., 1.)
        return action

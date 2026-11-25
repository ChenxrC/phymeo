"""An ankle strategy for quiet, level-ground G1 standing.

The linear inverted pendulum has capture point xi = c + v/omega and
xi_dot = omega * (xi - CoP). Choosing CoP = c_initial + 1.5*(xi-c_initial)
gives local error decay at 0.5*omega before support or motor saturation.
All positions and velocities below are in world metres and metres/second.
"""

import mujoco as mj
import numpy as np

CAPTURE_GAIN = 1.5
# The G1 sole's long capsules run from about -0.054 to +0.132 m in local X.
# These shorter levers stay inside their central support strip on a level sole.
HEEL_M = 0.04
TOE_M = 0.09
SIDE_M = 0.015
LOAD_THRESHOLD_N = 1.0
HIP_KP_NM_PER_RAD = 160.0
HIP_KD_NM_S_PER_RAD = 30.0
HIP_MOTOR_FRACTION = .35


def capture_point_target(com_m, velocity_m_s, support_height_m, initial_xy_m, gravity_m_s2):
    com = np.asarray(com_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    initial = np.asarray(initial_xy_m, dtype=float)
    if com.shape != (3,) or velocity.shape != (3,) or initial.shape != (2,):
        raise ValueError("expected a 3D COM, 3D velocity and planar initial target")
    height = com[2] - support_height_m
    if not (
        np.isfinite(com).all()
        and np.isfinite(velocity).all()
        and np.isfinite(initial).all()
        and np.isfinite(height)
        and height > 0
        and np.isfinite(gravity_m_s2)
        and gravity_m_s2 > 0
    ):
        raise ValueError("capture point requires finite state, gravity and positive COM height")
    omega = np.sqrt(gravity_m_s2 / height)
    capture = com[:2] + velocity[:2] / omega
    return float(omega), capture, initial + CAPTURE_GAIN * (capture - initial)


def ankle_balance_moment(offset_xy_m, forward_xy, vertical_force_n):
    """Bound one sole's requested CoP shift, then return its actuator moment on the foot."""
    offset = np.asarray(offset_xy_m, dtype=float)
    forward = np.asarray(forward_xy, dtype=float)
    if offset.shape != (2,) or forward.shape != (2,):
        raise ValueError("expected horizontal offset and foot-forward vectors")
    norm = np.linalg.norm(forward)
    if not (
        np.isfinite(offset).all()
        and np.isfinite(norm)
        and norm > 1e-8
        and np.isfinite(vertical_force_n)
        and vertical_force_n >= 0
    ):
        raise ValueError("invalid sole frame or normal force")
    forward = forward / norm
    left = np.array([-forward[1], forward[0]])
    bounded = (
        np.clip(offset @ forward, -HEEL_M, TOE_M) * forward
        + np.clip(offset @ left, -SIDE_M, SIDE_M) * left
    )
    # A vertical ground force at displacement r produces (r_y, -r_x, 0)*F.
    # The motor must apply its opposite to the foot; the leg receives the reaction.
    return bounded, vertical_force_n * np.array([-bounded[1], bounded[0], 0.0])


class AnkleBalance:
    """Per-trajectory kinematic observer; the caller supplies its last published foot loads."""

    def __init__(self, protocol, initial_q):
        self.model = protocol.model
        self.data = mj.MjData(self.model)
        self.root = protocol.root
        self.weight = protocol.weight
        self.foot_geoms = tuple(protocol.foot_geoms.items())
        self.jacobian = np.zeros((3, self.model.nv))
        self.motors = np.array(
            [
                [protocol.names.index(f"{side}_ankle_{axis}_joint") for axis in ("pitch", "roll")]
                for side in ("left", "right")
            ]
        )
        self.joints = self.model.actuator_trnid[self.motors, 0]
        self.gear = self.model.actuator_gear[self.motors, 0]
        if (
            np.any(self.model.actuator_trntype[self.motors] != mj.mjtTrn.mjTRN_JOINT)
            or np.any(self.model.jnt_type[self.joints] != mj.mjtJoint.mjJNT_HINGE)
            or np.any(self.gear == 0)
            or not np.all(self.model.actuator_ctrllimited[self.motors])
        ):
            raise ValueError("balance requires four directly driven, torque-limited ankle hinges")
        if {side for _, side in self.foot_geoms} != {0, 1}:
            raise ValueError("both feet need identified support geometry")
        self._kinematics(initial_q)
        self.initial_xy = self.data.subtree_com[self.root, :2].copy()

    def _kinematics(self, q):
        self.data.qpos[:] = q
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)

    def _sole_heights(self):
        heights = np.full(2, np.inf)
        for geom, side in self.foot_geoms:
            kind = self.model.geom_type[geom]
            size = self.model.geom_size[geom]
            bottom = self.data.geom_xpos[geom, 2] - size[0]
            if kind == mj.mjtGeom.mjGEOM_CAPSULE:
                bottom -= abs(self.data.geom_xmat[geom, 8]) * size[1]
            elif kind != mj.mjtGeom.mjGEOM_SPHERE:
                raise ValueError("standing support height requires capsule or sphere foot geoms")
            heights[side] = min(heights[side], bottom)
        return heights

    def correct(self, proposal, q, v, previous_foot_vertical_n):
        self._kinematics(q)
        mj.mj_jacSubtreeCom(self.model, self.data, self.jacobian, self.root)
        com = self.data.subtree_com[self.root].copy()
        velocity = self.jacobian @ v
        heights = self._sole_heights()
        previous = (
            np.zeros(2)
            if previous_foot_vertical_n is None
            else np.asarray(previous_foot_vertical_n, dtype=float)
        )
        if previous.shape != (2,) or not np.isfinite(previous).all():
            raise ValueError("previous published vertical foot forces must be two finite values")
        loads = np.where(previous > LOAD_THRESHOLD_N, previous, 0.0)
        total = float(loads.sum())
        weights = loads / total if total else np.zeros(2)
        budget = weights * min(total, self.weight)
        support_height = float(weights @ heights if total else heights.min())
        omega, capture, desired = capture_point_target(
            com, velocity, support_height, self.initial_xy, -self.model.opt.gravity[2]
        )
        roll_joints = self.joints[:, 1]
        ankles = self.data.xanchor[roll_joints, :2]
        center = weights @ ankles if total else ankles.mean(axis=0)
        levers = np.zeros((2, 2))
        moments = np.zeros((2, 3))
        extra_torque = np.zeros((2, 2))
        for side in range(2):
            if budget[side] > 0:
                body = self.model.jnt_bodyid[roll_joints[side]]
                forward = self.data.xmat[body].reshape(3, 3)[:2, 0]
                levers[side], moments[side] = ankle_balance_moment(
                    desired - center, forward, budget[side]
                )
                extra_torque[side] = self.data.xaxis[self.joints[side]] @ moments[side]

        raw = np.array(proposal["unclamped_ctrl"], dtype=float, copy=True)
        ctrl = np.array(proposal["ctrl"], dtype=float, copy=True)
        raw[self.motors] += extra_torque / self.gear
        limits = self.model.actuator_ctrlrange[self.motors]
        ctrl[self.motors] = np.clip(raw[self.motors], limits[:, :, 0], limits[:, :, 1])
        applied = (ctrl[self.motors] - proposal["ctrl"][self.motors]) * self.gear
        corrected = dict(proposal, ctrl=ctrl, unclamped_ctrl=raw, saturated=ctrl != raw)
        obs = {
            "initial_com_xy_m": self.initial_xy.copy(),
            "com_m": com,
            "com_velocity_m_s": velocity,
            "sole_heights_m": heights,
            "support_height_m": support_height,
            "omega_rad_s": omega,
            "capture_xy_m": capture,
            "desired_cop_xy_m": desired,
            "ankle_xy_m": ankles.copy(),
            "previous_foot_vertical_n": None
            if previous_foot_vertical_n is None
            else previous.copy(),
            "allocated_vertical_n": budget,
            "bounded_cop_offset_xy_m": levers,
            "requested_foot_moment_world_nm": moments,
            "requested_joint_delta_nm": extra_torque,
            "applied_joint_delta_nm": applied,
            "ankle_ctrl_limits": limits.copy(),
        }
        return corrected, obs


def pelvic_tilt_feedback(initial_up, current_up, angular_velocity_world):
    """Return pelvis tilt rotation, tilt rate and requested reaction on the legs."""
    sine_axis = np.cross(initial_up, current_up)
    sine = np.linalg.norm(sine_axis)
    cosine = np.clip(np.dot(initial_up, current_up), -1., 1.)
    if sine < 1e-12 and cosine < 0:
        raise ValueError("an inverted pelvis is outside the quiet-standing controller")
    error = sine_axis * (np.arctan2(sine, cosine) / sine) if sine > 0 else sine_axis
    # Rotation about the body's own up vector cannot change its tilt. Discard
    # torque about the initial upright axis as well: yaw is left to the joint PD.
    velocity = np.asarray(angular_velocity_world)
    tilt_rate = velocity - np.dot(velocity, current_up) * current_up
    tilt_rate -= np.dot(tilt_rate, initial_up) * initial_up
    return error, tilt_rate, HIP_KP_NM_PER_RAD * error + HIP_KD_NM_S_PER_RAD * tilt_rate


class HipAnkleBalance(AnkleBalance):
    """Use the original ankle strategy and react against the loaded thighs to right the pelvis."""

    def __init__(self, protocol, initial_q):
        super().__init__(protocol, initial_q)
        self.hip_motors = np.array([
            [protocol.names.index(f"{side}_hip_{axis}_joint") for axis in ("pitch", "roll")]
            for side in ("left", "right")
        ])
        self.hip_joints = self.model.actuator_trnid[self.hip_motors, 0]
        self.hip_gear = self.model.actuator_gear[self.hip_motors, 0]
        limits = self.model.actuator_ctrlrange[self.hip_motors]
        if (
            np.any(self.model.actuator_trntype[self.hip_motors] != mj.mjtTrn.mjTRN_JOINT)
            or np.any(self.model.jnt_type[self.hip_joints] != mj.mjtJoint.mjJNT_HINGE)
            or np.any(self.hip_gear == 0)
            or not np.all(self.model.actuator_ctrllimited[self.hip_motors])
            or np.any(limits[:, :, 0] >= 0)
            or np.any(limits[:, :, 1] <= 0)
        ):
            raise ValueError("hip balance requires four direct bidirectional torque-limited hinges")
        self.initial_up = self.data.xmat[self.root].reshape(3, 3)[:, 2].copy()
        self.rotation_jacobian = np.zeros((3, self.model.nv))

    def correct(self, proposal, q, v, previous_foot_vertical_n):
        corrected, obs = super().correct(proposal, q, v, previous_foot_vertical_n)
        up = self.data.xmat[self.root].reshape(3, 3)[:, 2]
        mj.mj_jacBody(self.model, self.data, None, self.rotation_jacobian, self.root)
        omega = self.rotation_jacobian @ v
        error, rate, desired = pelvic_tilt_feedback(self.initial_up, up, omega)
        # A motor acting +Y on the thigh reacts -Y on a forward-leaning pelvis.
        # Scale continuously by the last accepted support force / robot weight.
        fractions = obs["allocated_vertical_n"] / self.weight
        torques = (self.data.xaxis[self.hip_joints] @ desired) * fractions[:, None]
        motors = self.hip_motors
        limits = self.model.actuator_ctrlrange[motors]
        admitted = np.clip(torques / self.hip_gear, HIP_MOTOR_FRACTION * limits[:, :, 0],
                           HIP_MOTOR_FRACTION * limits[:, :, 1])
        ctrl, raw = corrected["ctrl"], corrected["unclamped_ctrl"]
        baseline = ctrl[motors].copy()
        raw[motors] += admitted
        ctrl[motors] = np.clip(raw[motors], limits[:, :, 0], limits[:, :, 1])
        corrected["saturated"] = raw != ctrl
        obs.update(
            pelvis_up_world=up.copy(), pelvis_initial_up_world=self.initial_up.copy(),
            pelvis_omega_world_rad_s=omega, pelvis_tilt_vector_rad=error,
            pelvis_tilt_rate_world_rad_s=rate, desired_leg_moment_world_nm=desired,
            hip_support_weight_fractions=fractions, hip_requested_joint_delta_nm=torques,
            hip_budgeted_joint_delta_nm=admitted * self.hip_gear,
            hip_applied_joint_delta_nm=(ctrl[motors] - baseline) * self.hip_gear,
        )
        return corrected, obs

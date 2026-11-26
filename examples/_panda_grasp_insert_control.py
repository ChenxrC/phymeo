"""Motor-only, staged feedback baseline for the physical dual-Panda assembly.

No state writes, external wrenches, welds, grasp constraints or thread guides.
The controller owns detached MuJoCo data only for FK/Jacobians and bias forces.
Phases follow fixed wall-clock times, never an assumed grasp/success event.
Use independently scored actual part motion/contact to decide task success.
"""
from dataclasses import dataclass
import math

import numpy as np

# Identical schedule for closed and negative controls, seconds from episode reset.
PHASES = (
    ("settle", 0., .5), ("approach", .5, 2.), ("descend", 2., 3.),
    ("close", 3., 4.), ("lift", 4., 5.2), ("transfer", 5.2, 7.2),
    ("insert", 7.2, 8.7), ("seat", 8.7, 9.7),
    ("release", 9.7, 10.5), ("clear", 10.5, 12.),
)
DURATION_S = 12.
CONTROL_PERIOD_S = .02
OPENING_PER_FINGER_M = .035
MAX_FINGER_EFFORT_N = 20.
ARM_KP = np.array([270., 300., 240., 220., 95., 85., 65.]*2)
ARM_KD = np.array([26., 29., 22., 18., 6., 5., 3.]*2)
ARM_SPEED_RAD_S = np.array([2.175]*4+[2.61]*3)*.8


@dataclass(frozen=True)
class ControlStep:
    ctrl: np.ndarray
    phase: str
    diagnostics: dict


def initial_qpos(scene, model):
    """Fresh episode declaration only; prefer and bind the model's home key."""
    declared = np.asarray(scene.initial_qpos, dtype=float)
    try:
        index = model.key("home").id
    except (KeyError, ValueError):
        index = -1
    q = np.array(model.key_qpos[index] if index >= 0 else declared, copy=True)
    if q.shape != (23,) or not np.isfinite(q).all():
        raise ValueError("full grasp scene requires q23 home")
    if not np.allclose(q, declared, rtol=0., atol=2e-14):
        raise ValueError("home key differs from scene's declared initial state")
    return q


def phase_at(time_s):
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("episode-relative finite nonnegative time required")
    for name, start, end in PHASES:
        if time_s < end:
            u = (time_s-start)/(end-start)
            return name, max(0., u)
    return "complete", 1.


def _smooth(u):
    u = float(np.clip(u, 0., 1.))
    return u*u*u*(10.+u*(-15.+6.*u))


def _bounded(vector, limit):
    a = np.asarray(vector, dtype=float)
    return a * min(1., limit/max(float(np.linalg.norm(a)), 1e-30))


def _rotation_vector(mj, rotation):
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, np.ascontiguousarray(rotation).ravel())
    if quat[0] < 0:
        quat = -quat
    sine = float(np.linalg.norm(quat[1:]))
    if sine < 1e-12:
        return 2.*quat[1:]
    return quat[1:] * (2.*math.atan2(sine, float(quat[0]))/sine)


def _rotation_exp(vector):
    v = np.asarray(vector, dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = v/theta
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+math.sin(theta)*k+(1.-math.cos(theta))*(k@k)


class GraspInsertController:
    """Read q23/v22, return sixteen real motor efforts and phase diagnostics.

    ``open_jaw=True`` alone uses the same feedback law and schedule, so changed
    physical observations may change arm efforts. For the strict matched-motion
    negative, pass the corresponding closed trial's full ``ControlStep.ctrl``
    as ``reference_ctrl`` each step. Its first fourteen motor efforts are copied
    exactly; only the two jaws are replaced. Equal efforts do not assert equal
    arm trajectories under different physical loads.
    """
    def __init__(self, scene, model, *, open_jaw=False):
        import mujoco as mj
        if type(open_jaw) is not bool:
            raise TypeError("open_jaw must be bool")
        if (model.nq, model.nv, model.nu) != (23, 22, 16):
            raise ValueError("expected two seven-axis arms, two driven fingers, one free part")
        self.mj, self.scene, self.model = mj, scene, model
        self.open_jaw = open_jaw
        self.q0 = initial_qpos(scene, model)
        self.data, self.ik_data = mj.MjData(model), mj.MjData(model)
        self.sites = {n: model.site(n).id for n in (
            "right_grasp_center", "part_grasp_center", "assembly_pin_tip", "assembly_hole", "assembly_entry")}
        # Reject unexpected transmissions instead of applying Nm as tendon or
        # position-servo commands. Coordinate order is part of the scene contract.
        for i in range(16):
            joint = int(model.actuator_trnid[i, 0])
            if (int(model.jnt_qposadr[joint]) != i or int(model.jnt_dofadr[joint]) != i
                    or int(model.actuator_trntype[i]) != int(mj.mjtTrn.mjTRN_JOINT)
                    or not np.array_equal(model.actuator_gear[i], [1., 0., 0., 0., 0., 0.])
                    or not model.actuator_ctrllimited[i]
                    or int(model.actuator_gaintype[i]) != int(mj.mjtGain.mjGAIN_FIXED)
                    or model.actuator_gainprm[i, 0] != 1.
                    or np.any(model.actuator_biasprm[i] != 0.)):
                raise ValueError("requires ordered unit-gear effort motors")
        self.limits = np.asarray(model.actuator_ctrlrange).copy()
        self.limits[14:, 0] = np.maximum(self.limits[14:, 0], -MAX_FINGER_EFFORT_N)
        self.limits[14:, 1] = np.minimum(self.limits[14:, 1], MAX_FINGER_EFFORT_N)
        self.right_limits = np.array([model.jnt_range[int(model.actuator_trnid[i, 0])] for i in range(7, 14)])
        self.finger_limits = np.array([model.jnt_range[int(model.actuator_trnid[i, 0])] for i in (14, 15)])
        if np.any(self.finger_limits[:, 1] < OPENING_PER_FINGER_M):
            raise ValueError("gripper cannot reach the declared .035 m per-finger opening")
        self._update(self.q0, np.zeros(22))
        self.home_tcp = self._pose("right_grasp_center")
        self.start_part = self._pose("part_grasp_center")
        self.start_tip = self._pose("assembly_pin_tip")
        self.board_home = self._pose("assembly_hole")
        self.entry_z = float((self.board_home[:3, :3].T @
                              (self._pose("assembly_entry")[:3, 3]-self.board_home[:3, 3]))[2])
        self.tip_from_grasp = np.linalg.inv(self.start_part) @ self.start_tip
        self.tip_rotation_in_board = self.board_home[:3, :3].T @ self.start_tip[:3, :3]
        self.grasp_rotation = self.start_part[:3, :3].copy()
        self.grasp_xyz = self.start_part[:3, 3].copy()
        self.hover_xyz = self.grasp_xyz + np.array([0., 0., .075])
        self.lift_xyz = self.grasp_xyz + np.array([0., 0., .09])
        self.preinsert_depth = -.055
        self.seated_depth = .010
        self.preinsert_tcp = self._nominal_insert(self.board_home, self.preinsert_depth)
        self.seated_tcp = self._nominal_insert(self.board_home, self.seated_depth)
        self.clear_xyz = self.seated_tcp[:3, 3] + np.array([0., 0., .10])
        self._target_q = self.q0[:14].copy()
        self._last_ik_time = -math.inf
        self._last_time = -math.inf
        self._reference_mode = None
        self._release_tcp = None

    def _update(self, q, v):
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        self.mj.mj_fwdPosition(self.model, self.data)
        self.mj.mj_fwdVelocity(self.model, self.data)

    def _pose(self, name):
        index = self.sites[name]
        t = np.eye(4)
        t[:3, 3] = self.data.site_xpos[index]
        t[:3, :3] = self.data.site_xmat[index].reshape(3, 3)
        return t

    def _nominal_insert(self, board, depth):
        tip = np.eye(4)
        tip[:3, :3] = board[:3, :3] @ self.tip_rotation_in_board
        tip[:3, 3] = board[:3, 3]+board[:3, 2]*(self.entry_z+depth)
        return tip @ np.linalg.inv(self.tip_from_grasp)

    def _scheduled_target(self, phase, u):
        a = _smooth(u)
        rotation = self.grasp_rotation.copy()
        depth = None
        if phase == "settle":
            xyz, rotation = self.home_tcp[:3, 3].copy(), self.home_tcp[:3, :3].copy()
        elif phase == "approach":
            xyz = (1-a)*self.home_tcp[:3, 3]+a*self.hover_xyz
            delta = _rotation_vector(self.mj, self.grasp_rotation @ self.home_tcp[:3, :3].T)
            rotation = _rotation_exp(a*delta)@self.home_tcp[:3, :3]
        elif phase == "descend":
            xyz = (1-a)*self.hover_xyz+a*self.grasp_xyz
        elif phase == "close":
            xyz = self.grasp_xyz.copy()
        elif phase == "lift":
            xyz = (1-a)*self.grasp_xyz+a*self.lift_xyz
        elif phase == "transfer":
            xyz = (1-a)*self.lift_xyz+a*self.preinsert_tcp[:3, 3]
        elif phase in ("insert", "seat"):
            depth = ((1-a)*self.preinsert_depth+a*self.seated_depth) if phase == "insert" else self.seated_depth
            t = self._nominal_insert(self.board_home, depth)
            xyz, rotation = t[:3, 3], t[:3, :3]
        elif phase == "release":
            xyz, rotation = self.seated_tcp[:3, 3].copy(), self.seated_tcp[:3, :3].copy()
        else:
            a = a if phase == "clear" else 1.
            xyz = (1-a)*self.seated_tcp[:3, 3]+a*self.clear_xyz
            rotation = self.seated_tcp[:3, :3].copy()
        t = np.eye(4)
        t[:3, 3], t[:3, :3] = xyz, rotation
        return t, depth

    def _ik(self, q, target):
        current = self._target_q[7:14].copy()
        jp, jr = np.empty((3, 22)), np.empty((3, 22))
        self.ik_data.qpos[:] = q
        length_scale = .12
        residual = math.inf
        for _ in range(8):
            self.ik_data.qpos[7:14] = current
            self.mj.mj_kinematics(self.model, self.ik_data)
            self.mj.mj_comPos(self.model, self.ik_data)
            idx = self.sites["right_grasp_center"]
            pos = self.ik_data.site_xpos[idx]
            rot = self.ik_data.site_xmat[idx].reshape(3, 3)
            error = np.r_[target[:3, 3]-pos,
                          length_scale*_rotation_vector(self.mj, target[:3, :3]@rot.T)]
            residual = float(np.linalg.norm(error))
            if residual < 1e-6:
                break
            self.mj.mj_jacSite(self.model, self.ik_data, jp, jr, idx)
            jac = np.vstack((jp[:, 7:14], length_scale*jr[:, 7:14]))
            increment = jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*2e-5, error)
            current += _bounded(increment, .12)
            current = np.clip(current, self.right_limits[:, 0]+.001, self.right_limits[:, 1]-.001)
        return current, residual

    def advance(self, q, qd, time_s, *, reference_ctrl=None, reference_arm_target=None):
        q, v = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
        if q.shape != (23,) or v.shape != (22,) or not np.isfinite(np.r_[q, v]).all():
            raise ValueError("finite observed q23/v22 required")
        phase, u = phase_at(float(time_s))
        if time_s < self._last_time:
            raise ValueError("controller clock cannot go backwards; create a new owner after reset")
        if reference_ctrl is not None and reference_arm_target is not None:
            raise ValueError("select one arm replay mode")
        mode = "effort" if reference_ctrl is not None else "target" if reference_arm_target is not None else "feedback"
        reference = mode != "feedback"
        if reference and not self.open_jaw:
            raise ValueError("arm effort replay is only for the explicit open-jaw negative")
        if self._reference_mode is not None and mode != self._reference_mode:
            raise ValueError("cannot switch reference mode partway through a trial")
        self._reference_mode = mode
        self._update(q, v)
        tcp, tip, board = self._pose("right_grasp_center"), self._pose("assembly_pin_tip"), self._pose("assembly_hole")
        target, depth_target = self._scheduled_target(phase, u)
        if phase in ("release", "clear", "complete"):
            # Open in place at the achieved pose; do not snap back to a nominal
            # pose while the pads still touch the seated workpiece. The strict
            # negative replays the closed trial's resulting arm motor efforts.
            if self._release_tcp is None:
                self._release_tcp = tcp.copy()
            target = self._release_tcp.copy()
            if phase != "release":
                target[:3, 3] += np.array([0., 0., .10])*(_smooth(u) if phase == "clear" else 1.)
        nominal = target.copy()
        correction_m = 0.
        correction_clipped = False
        if depth_target is not None and not reference:
            # Solve for the TCP pose which would put the ACTUAL held tip at the
            # desired pose. Bounded correction cannot chase a dropped part across
            # the cell. Neither this transform nor the part is written to physics.
            desired_tip = self._nominal_insert(board, depth_target) @ self.tip_from_grasp
            corrected = desired_tip @ np.linalg.inv(tip) @ tcp
            delta = corrected[:3, 3]-nominal[:3, 3]
            angle = _rotation_vector(self.mj, corrected[:3, :3]@nominal[:3, :3].T)
            correction_m = float(np.linalg.norm(delta))
            correction_clipped = correction_m > .018 or np.linalg.norm(angle) > .15
            target[:3, 3] += _bounded(delta, .018)
            target[:3, :3] = _rotation_exp(_bounded(angle, .15))@nominal[:3, :3]
        ik_residual = None
        if not reference and time_s-self._last_ik_time >= CONTROL_PERIOD_S-1e-9:
            candidate, ik_residual = self._ik(q, target)
            elapsed = CONTROL_PERIOD_S if not np.isfinite(self._last_ik_time) else min(time_s-self._last_ik_time, .05)
            bound = ARM_SPEED_RAD_S*elapsed
            self._target_q[7:14] += np.clip(candidate-self._target_q[7:14], -bound, bound)
            self._last_ik_time = float(time_s)
        if mode == "target":
            target_q = np.asarray(reference_arm_target, dtype=float)
            if target_q.shape != (14,) or not np.isfinite(target_q).all():
                raise ValueError("expected fourteen finite arm targets")
            self._target_q[:] = target_q
        requested = np.zeros(16)
        requested[:14] = ARM_KP*(self._target_q-q[:14])-ARM_KD*v[:14]+self.data.qfrc_bias[:14]
        if self.open_jaw or phase in ("settle", "approach", "descend", "clear", "complete"):
            opening = OPENING_PER_FINGER_M
        elif phase == "close":
            opening = OPENING_PER_FINGER_M*(1-_smooth(u))
        elif phase == "release":
            opening = OPENING_PER_FINGER_M*_smooth(u)
        else:
            opening = 0.
        opening = np.clip(opening, self.finger_limits[:, 0], self.finger_limits[:, 1])
        requested[14:] = 1400.*(opening-q[14:16])-12.*v[14:16]+self.data.qfrc_bias[14:16]
        ctrl = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        if mode == "effort":
            ref = np.asarray(reference_ctrl, dtype=float)
            if ref.shape != (16,) or not np.isfinite(ref).all():
                raise ValueError("reference_ctrl must be the closed trial's finite sixteen motor efforts")
            if np.any(ref[:14] < self.limits[:14, 0]) or np.any(ref[:14] > self.limits[:14, 1]):
                raise ValueError("reference effort exceeds the same scene's physical limits")
            ctrl[:14] = ref[:14]
        relative = np.linalg.inv(board)@tip
        measured_tip_local = relative[:3, 3]
        grasp_offset = np.linalg.inv(tcp)@self._pose("part_grasp_center")
        diag = dict(
            schema="panda-grasp-insert-control-v1",time_s=float(time_s),phase=phase,
            phase_fraction=float(u),duration_s=DURATION_S,
            control_mode=("replayed-closed-arm-effort/open-jaw" if mode == "effort" else
                          "replayed-closed-arm-target/open-jaw" if mode == "target" else
                          "same-feedback-law/open-jaw" if self.open_jaw else "physical-grasp-feedback"),
            arm_target_q=None if mode == "effort" else self._target_q.tolist(),
            planned_tcp_world=nominal.tolist(),feedback_tcp_world=target.tolist(),
            actual_tcp_world=tcp.tolist(),actual_part_tip_world=tip.tolist(),
            actual_tip_in_board=measured_tip_local.tolist(),
            depth_m=float(measured_tip_local[2]-self.entry_z),
            depth_target_m=depth_target,part_grasp_offset_m=grasp_offset[:3, 3].tolist(),
            tip_axis_tilt_rad=float(math.acos(float(np.clip(relative[2, 2], -1., 1.)))),
            part_feedback_correction_m=correction_m,part_feedback_clipped=bool(correction_clipped),
            ik_weighted_residual_m=ik_residual,jaw_target_m=np.asarray(opening).tolist(),
            applied_ctrl=ctrl.tolist(),effort_units="arm Nm; fingers N",
            arm_saturated_count=None if reference else int(np.count_nonzero(requested[:14] != ctrl[:14])),
            finger_saturated_count=int(np.count_nonzero(requested[14:] != ctrl[14:])),
            physical_success=None,external_wrench_applied=False,
        )
        self._last_time = float(time_s)
        ctrl.setflags(write=False)
        return ControlStep(ctrl, phase, diag)

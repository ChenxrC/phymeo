"""Bounded native-joint diagnostic controls; no physical integration or state writes.

MuJoCo scratch is used only for kinematics, point Jacobians and the bias force.
Returned references and efforts are commands for the actual physical owner,
never replacement robot/object states. This is a HOST diagnostic, not a policy.
"""

from dataclasses import dataclass
from threading import RLock

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError("native diagnostic: " + message)


def _readonly(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _vector(value, size, name):
    array = np.asarray(value)
    _require(
        array.shape == (size,) and array.dtype.kind in "fiu" and np.isfinite(array).all(),
        name + " requires a finite native vector",
    )
    return np.array(array, dtype=np.float64, copy=True)


def _gain(value, size, name):
    array = np.asarray(value)
    if array.ndim == 0 and array.dtype.kind in "fiu":
        array = np.full(size, float(array), dtype=np.float64)
    result = _vector(array, size, name)
    _require(np.all(result >= 0), name + " must be nonnegative")
    return _readonly(result)


def _joints(mj, model, names):
    _require(isinstance(model, mj.MjModel), "an original MuJoCo model is required")
    names = tuple(names)
    _require(
        names
        and all(type(name) is str and name for name in names)
        and len(set(names)) == len(names),
        "joint names must be nonempty and distinct",
    )
    ids = tuple(int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)) for name in names)
    _require(all(joint >= 0 for joint in ids), "unknown joint name")
    kinds = (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE))
    _require(
        all(int(model.jnt_type[joint]) in kinds for joint in ids),
        "only scalar HINGE/SLIDE joints may be commanded",
    )
    qa = tuple(int(model.jnt_qposadr[joint]) for joint in ids)
    va = tuple(int(model.jnt_dofadr[joint]) for joint in ids)
    _require(
        len(set(qa)) == len(qa)
        and len(set(va)) == len(va)
        and all(0 <= q < model.nq for q in qa)
        and all(0 <= v < model.nv for v in va),
        "native joint addresses are invalid",
    )
    lower, upper = np.full(len(ids), -np.inf), np.full(len(ids), np.inf)
    for index, joint in enumerate(ids):
        if model.jnt_limited[joint]:
            interval = model.jnt_range[joint]
            _require(np.isfinite(interval).all() and interval[0] < interval[1], "joint limit range")
            lower[index], upper[index] = interval
    return names, ids, qa, va, lower, upper


def _configuration(mj, model, q):
    result = _vector(q, model.nq, "q")
    for joint in range(model.njnt):
        kind, adr = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
        if kind == int(mj.mjtJoint.mjJNT_FREE):
            quaternion = result[adr + 3 : adr + 7]
        elif kind == int(mj.mjtJoint.mjJNT_BALL):
            quaternion = result[adr : adr + 4]
        else:
            continue
        _require(
            abs(float(quaternion @ quaternion) - 1.0) <= 4e-5,
            "native FREE/BALL quaternion must be unit WXYZ",
        )
    return result


@dataclass(frozen=True)
class NativePDDiagnostic:
    joint_names: tuple
    qpos_addresses: tuple
    dof_addresses: tuple
    actuator_ids: tuple
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    target_velocity: np.ndarray
    bias: np.ndarray
    unclipped_effort: np.ndarray
    effort: np.ndarray
    saturated: np.ndarray


class NativeJointPD:
    """Bias-compensated PD through unique unit-gear stateless native motors.

    The model is borrowed as immutable. q/v are full original native vectors;
    desired_q and optional desired_v follow joint_names, never native padding.
    Unselected controls are fresh zeros. Commands respect original actuator
    control/force limits and an additional absolute 19.9 effort bound.
    """

    def __init__(self, model, joint_names, kp, kd):
        import mujoco as mj

        names, joints, qa, va, lower, upper = _joints(mj, model, joint_names)
        self._mj, self.model = mj, model
        self.joint_names, self.qpos_addresses, self.dof_addresses = names, qa, va
        self.kp, self.kd = _gain(kp, len(names), "kp"), _gain(kd, len(names), "kd")
        self._lower, self._upper = _readonly(lower), _readonly(upper)
        actuators = []
        effort_lower, effort_upper = np.full(len(names), -19.9), np.full(len(names), 19.9)
        transmissions = (int(mj.mjtTrn.mjTRN_JOINT), int(mj.mjtTrn.mjTRN_JOINTINPARENT))
        for index, joint in enumerate(joints):
            candidates = [
                actor
                for actor in range(model.nu)
                if int(model.actuator_trntype[actor]) in transmissions
                and int(model.actuator_trnid[actor, 0]) == joint
            ]
            _require(len(candidates) == 1, "each selected joint requires exactly one native motor")
            actor = candidates[0]
            _require(
                int(model.actuator_trntype[actor]) == int(mj.mjtTrn.mjTRN_JOINT)
                and int(model.actuator_dyntype[actor]) == int(mj.mjtDyn.mjDYN_NONE)
                and int(model.actuator_gaintype[actor]) == int(mj.mjtGain.mjGAIN_FIXED)
                and float(model.actuator_gainprm[actor, 0]) == 1.0
                and int(model.actuator_biastype[actor]) == int(mj.mjtBias.mjBIAS_NONE),
                "selected actuator must be a stateless unit-gain motor without bias",
            )
            gear = model.actuator_gear[actor]
            _require(
                np.isfinite(gear).all() and gear[0] == 1.0 and not np.any(gear[1:]),
                "selected motor requires unit scalar gear",
            )
            for limited, interval in (
                (model.actuator_ctrllimited[actor], model.actuator_ctrlrange[actor]),
                (model.actuator_forcelimited[actor], model.actuator_forcerange[actor]),
                (model.jnt_actfrclimited[joint], model.jnt_actfrcrange[joint]),
            ):
                if limited:
                    _require(
                        np.isfinite(interval).all() and interval[0] < interval[1],
                        "actuator limit range",
                    )
                    effort_lower[index] = max(effort_lower[index], float(interval[0]))
                    effort_upper[index] = min(effort_upper[index], float(interval[1]))
            _require(
                effort_lower[index] <= 0 <= effort_upper[index]
                and effort_lower[index] < effort_upper[index],
                "motor limits must admit zero",
            )
            actuators.append(actor)
        self.actuator_ids = tuple(actuators)
        self._effort_lower, self._effort_upper = _readonly(effort_lower), _readonly(effort_upper)
        self._data, self._lock = mj.MjData(model), RLock()

    def command(self, q, v, desired_q, desired_v=None):
        """Return (full native ctrl, diagnostic), without integrating any state."""
        with self._lock:
            mj, model, data = self._mj, self.model, self._data
            position = _configuration(mj, model, q)
            velocity = _vector(v, model.nv, "v")
            target = _vector(desired_q, len(self.joint_names), "desired_q")
            target_velocity = (
                np.zeros(len(self.joint_names))
                if desired_v is None
                else _vector(desired_v, len(self.joint_names), "desired_v")
            )
            _require(
                np.all(target >= self._lower) and np.all(target <= self._upper),
                "desired_q exceeds original joint limits",
            )
            data.qpos[:] = position
            data.qvel[:] = velocity
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mj_comVel(model, data)
            bias = np.zeros(model.nv, dtype=np.float64)
            mj.mj_rne(model, data, 0, bias)
            qa, va = list(self.qpos_addresses), list(self.dof_addresses)
            with np.errstate(over="raise", invalid="raise"):
                raw = (
                    bias[va]
                    + self.kp * (target - position[qa])
                    + self.kd * (target_velocity - velocity[va])
                )
            _require(np.isfinite(bias).all() and np.isfinite(raw).all(), "nonfinite PD effort")
            effort = np.clip(raw, self._effort_lower, self._effort_upper)
            ctrl = np.zeros(model.nu, dtype=np.float64)
            ctrl[list(self.actuator_ids)] = effort
            diagnostic = NativePDDiagnostic(
                self.joint_names,
                self.qpos_addresses,
                self.dof_addresses,
                self.actuator_ids,
                _readonly(position[qa]),
                _readonly(velocity[va]),
                _readonly(target),
                _readonly(target_velocity),
                _readonly(bias[va]),
                _readonly(raw),
                _readonly(effort),
                _readonly(raw != effort, np.bool_),
            )
            return _readonly(ctrl), diagnostic


@dataclass(frozen=True)
class NativePointIKResult:
    configuration: np.ndarray
    robot_configuration: np.ndarray
    position: np.ndarray
    error_m: float
    iterations: int
    status: str

    @property
    def converged(self):
        return self.status == "converged"


def solve_native_point_ik(
    model,
    target,
    q_seed,
    joint_names,
    *,
    body_name="tool",
    local_point=(0, 0, -0.14),
    damping=1e-3,
    max_step=0.15,
    tolerance_m=5e-4,
    max_iterations=100,
):
    """Bounded local position IK, with no claim of global infeasibility on stall.

    Only selected scalar ancestor joints may change in the returned reference.
    The selected seed/reference is projected to original joint ranges; the
    supplied q_seed, model and every other native coordinate remain untouched.
    At most eight line-search trials are made per bounded DLS iteration.
    """
    import mujoco as mj

    _, joints, qa, va, lower, upper = _joints(mj, model, joint_names)
    _require(type(body_name) is str and body_name, "body_name must be nonempty")
    body = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
    _require(body > 0, "unknown or world IK body")
    ancestors, current = set(), body
    while current:
        ancestors.add(current)
        current = int(model.body_parentid[current])
    _require(
        all(int(model.jnt_bodyid[joint]) in ancestors for joint in joints),
        "selected IK joints must be target-body ancestors",
    )
    target = _vector(target, 3, "target")
    point = _vector(local_point, 3, "local_point")
    q = _configuration(mj, model, q_seed)
    for name, value in (("damping", damping), ("max_step", max_step), ("tolerance_m", tolerance_m)):
        _require(
            type(value) in (float, int) and np.isfinite(value) and value > 0,
            name + " must be finite and positive",
        )
    _require(
        type(max_iterations) is int and 1 <= max_iterations <= 10000,
        "max_iterations must be a bounded positive integer",
    )
    qadr, vadr = list(qa), list(va)
    q[qadr] = np.clip(q[qadr], lower, upper)
    data = mj.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def evaluate(configuration):
        data.qpos[:] = configuration
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        position = data.xpos[body] + data.xmat[body].reshape(3, 3) @ point
        error = target - position
        _require(np.isfinite(position).all() and np.isfinite(error).all(), "nonfinite IK geometry")
        return position.copy(), error, float(np.linalg.norm(error))

    position, error, norm = evaluate(q)
    status, iterations = "iteration-limit", 0
    for iteration in range(max_iterations + 1):
        iterations = iteration
        if norm <= tolerance_m:
            status = "converged"
            break
        if iteration == max_iterations:
            break
        mj.mj_jac(model, data, jacp, jacr, position, body)
        jacobian = jacp[:, vadr]
        _require(np.isfinite(jacobian).all(), "nonfinite IK Jacobian")
        with np.errstate(over="raise", invalid="raise"):
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + float(damping) ** 2 * np.eye(3), error
            )
        _require(np.isfinite(delta).all(), "nonfinite IK increment")
        size = float(np.linalg.norm(delta))
        if size > max_step:
            delta *= max_step / size
        accepted = False
        for trial in range(8):
            candidate = q.copy()
            candidate[qadr] = np.clip(q[qadr] + delta * (0.5**trial), lower, upper)
            next_position, next_error, next_norm = evaluate(candidate)
            if next_norm < norm:
                q, position, error, norm = candidate, next_position, next_error, next_norm
                accepted = True
                break
        if not accepted:
            status = "stalled"
            break
    return NativePointIKResult(
        _readonly(q),
        _readonly(q[qadr]),
        _readonly(position),
        norm,
        iterations,
        status,
    )

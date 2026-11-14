"""Bounded local position/orientation IK using original native joint addresses.

This HOST diagnostic returns references, never physical state updates. A stalled
or exhausted local solve is not a proof of global infeasibility or safe contact.
"""

from dataclasses import dataclass

import numpy as np

from .manipulation_diagnostic import _configuration, _joints, _readonly, _require, _vector
from .manipulation_pose import _ancestors


@dataclass(frozen=True)
class NativePoseIKResult:
    configuration: np.ndarray
    robot_configuration: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    status: str

    @property
    def converged(self):
        return self.status == "converged"


def _rotation(value):
    array = np.asarray(value)
    _require(array.shape == (3, 3) and array.dtype.kind in "fiu" and np.isfinite(array).all(),
             "target_rotation requires a finite numeric 3x3 matrix")
    result = np.array(array, dtype=np.float64, copy=True)
    _require(np.allclose(result.T @ result, np.eye(3), rtol=0, atol=1e-6)
             and abs(float(np.linalg.det(result)) - 1.0) <= 1e-6,
             "target_rotation must be proper SO(3), without projection or reflection")
    return result


def solve_native_pose_ik(
    model, target, target_rotation, q_seed, joint_names, *, body_name="tool",
    local_point=(0, 0, -0.14), orientation_length_m=0.1, damping=1e-3,
    max_step=0.15, tolerance_m=5e-4, tolerance_rad=5e-3, max_iterations=100,
):
    """Solve one fixed world TCP position and body-to-world orientation target.

    Only selected scalar ancestor joints change, within original native limits.
    The copied seed is initially projected to those limits. All FREE and other
    unselected coordinates are preserved byte for byte. Targets and caller data
    are copied, and all returned arrays have immutable owned byte storage.

    e_R = Log(R_target R_current.T) is a world-space rotation vector. MuJoCo's
    matrix-to-quaternion and quaternion-to-velocity routines select the short
    arc with a stable largest-diagonal branch at pi; no cross-axis sine proxy
    or local-frame quaternion subtraction is used. Position and rotation rows
    have weights 1 and orientation_length_m, respectively.

    The analytic geometric angular Jacobian supplies a local DLS direction,
    not the full differential of the logarithm residual. Eight backtracking
    trials check the actual weighted residual. There is no global convergence
    claim, random restart, native forward/inverse solve or physical stepping.
    Step length, tolerances and iteration cap may only tighten these defaults.
    SO(3) admission allows 1e-6 numerical orthogonality/determinant error.
    """
    import mujoco as mj

    _require(isinstance(model, mj.MjModel), "an original MuJoCo model is required")
    _require(0 < model.nq <= 45 and 0 < model.nv <= 35 and model.nbody <= 128,
             "model exceeds finite pose IK dimensions")
    _require(type(joint_names) in (tuple, list) and 1 <= len(joint_names) <= 35,
             "joint_names requires a bounded explicit tuple/list")
    _, joints, qa, va, lower, upper = _joints(mj, model, joint_names)
    _require(type(body_name) is str and body_name, "body_name must be nonempty")
    body = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
    _require(body > 0, "unknown or world IK body")
    _require(all(int(model.jnt_bodyid[joint]) in _ancestors(model, body) for joint in joints),
             "selected IK joints must be target-body ancestors")
    target, point = _vector(target, 3, "target"), _vector(local_point, 3, "local_point")
    desired_rotation = _rotation(target_rotation)
    q = _configuration(mj, model, q_seed)
    for name, value in (("orientation_length_m", orientation_length_m), ("damping", damping),
                        ("max_step", max_step), ("tolerance_m", tolerance_m), ("tolerance_rad", tolerance_rad)):
        _require(type(value) in (float, int) and np.isfinite(value) and value > 0,
                 name + " must be finite and positive")
    _require(max_step <= 0.15 and tolerance_m <= 5e-4 and tolerance_rad <= 5e-3,
             "pose IK step/tolerances may only tighten the fixed upper bounds")
    _require(type(max_iterations) is int and 1 <= max_iterations <= 100,
             "max_iterations must be an integer in 1..100")
    qadr, vadr = list(qa), list(va)
    untouched = np.ones(model.nq, dtype=np.bool_)
    untouched[qadr] = False
    unchanged = q[untouched].tobytes()
    q[qadr] = np.clip(q[qadr], lower, upper)
    data = mj.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def evaluate(configuration):
        data.qpos[:] = configuration
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        rotation = data.xmat[body].reshape(3, 3).copy()
        position = data.xpos[body] + rotation @ point
        quaternion, angular_error = np.zeros(4), np.zeros(3)
        # Left/spatial relative rotation: its axis uses the same world frame as jacr.
        mj.mju_mat2Quat(quaternion, (desired_rotation @ rotation.T).reshape(9))
        mj.mju_quat2Vel(angular_error, quaternion, 1.0)
        position_error = target - position
        error = np.r_[position_error, orientation_length_m * angular_error]
        norms = (float(np.linalg.norm(position_error)), float(np.linalg.norm(angular_error)),
                 float(np.linalg.norm(error)))
        _require(np.isfinite(position).all() and np.isfinite(rotation).all()
                 and np.isfinite(error).all() and np.isfinite(norms).all(), "nonfinite pose IK geometry")
        return position.copy(), rotation, error, *norms

    position, rotation, error, position_norm, angular_norm, norm = evaluate(q)
    status, iterations = "iteration-limit", 0
    for iteration in range(max_iterations + 1):
        iterations = iteration
        if position_norm <= tolerance_m and angular_norm <= tolerance_rad:
            status = "converged"
            break
        if iteration == max_iterations:
            break
        mj.mj_jac(model, data, jacp, jacr, position, body)
        jacobian = np.vstack((jacp[:, vadr], orientation_length_m * jacr[:, vadr]))
        _require(np.isfinite(jacobian).all(), "nonfinite pose IK Jacobian")
        with np.errstate(over="raise", invalid="raise"):
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + float(damping) ** 2 * np.eye(6), error,
            )
        _require(np.isfinite(delta).all(), "nonfinite pose IK increment")
        size = float(np.linalg.norm(delta))
        if size > max_step:
            delta *= max_step / size
        accepted = False
        for trial in range(8):
            candidate = q.copy()
            candidate[qadr] = np.clip(q[qadr] + delta * (0.5 ** trial), lower, upper)
            evaluated = evaluate(candidate)
            if evaluated[-1] < norm:
                q = candidate
                position, rotation, error, position_norm, angular_norm, norm = evaluated
                accepted = True
                break
        if not accepted:
            status = "stalled"
            break
    _require(q[untouched].tobytes() == unchanged, "pose IK changed an unselected coordinate")
    return NativePoseIKResult(_readonly(q), _readonly(q[qadr]), _readonly(position), _readonly(rotation),
                              position_norm, angular_norm, iterations, status)

"""Robot layouts and force-limited drives for the interchangeable handoff fixture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..constraints.robot_contact import JointLimits, LinkCapsule
from ..control.bounded_pd import solve_bounded_pd
from ..control.point_ik import solve_point_ik
from ..core.capabilities import SolverFailure
from ..core.robot import RobotInputs, RobotModel, RobotState
from ..dynamics.robot_dynamics import robot_dynamics_terms
from ..dynamics.robot_kinematics import forward_kinematics, point_jacobians

BASE_Z = 0.039
HALF_OPEN = 0.013
CLOSED = HALF_OPEN - 0.0006
PAD_HALF_LENGTH = 0.022
PAD_HALF_HEIGHT = 0.019
FINGER_REACH = 0.045
SHANK_RADIUS = 0.002


def _cartesian_model(base_x):
    # Two carriage axes and independently actuated opposed fingers.
    return RobotModel(
        np.array([-1, 0, 1, 1]),
        ("prismatic",) * 4,
        np.array(
            [[base_x, 0.0, BASE_Z], [0.0, 0.0, 0.0], [0.0, -HALF_OPEN, 0.0], [0.0, HALF_OPEN, 0.0]]
        ),
        np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        np.array([0.12, 0.10, 0.06, 0.06]),
        np.zeros((4, 3)),
        np.full((4, 3), 0.001),
        np.array([0.0, 0.0, -9.81]),
    )


@dataclass
class Gripper(ABC):
    """A handoff hardware binding. Commands return actuator input and the requested joint target."""

    model: RobotModel
    initial: RobotState
    tool_link: int
    tool_point: np.ndarray
    fingers: tuple[int, int]
    effort_limits: np.ndarray
    limits: JointLimits | None = None
    capsules: tuple[LinkCapsule, ...] = ()
    excluded_pairs: tuple[tuple[str, str], ...] = ()

    def position(self, state):
        pose = forward_kinematics(self.model, state.q)
        tip = pose.position[self.tool_link] + pose.rotation[self.tool_link] @ self.tool_point
        return tip[[0, 2]]

    def velocity(self, state):
        jacobian, _ = point_jacobians(
            self.model, state.q, [self.tool_link], self.tool_point[None, :]
        )
        return (jacobian[0] @ state.v)[[0, 2]]

    def finger_effort(self, state, aperture):
        ids = list(self.fingers)
        effort = 1150.0 * (aperture - state.q[ids]) - 8.0 * state.v[ids]
        return np.clip(effort, -self.effort_limits[ids], self.effort_limits[ids])

    @staticmethod
    def _tool_velocity(value):
        if value is None:
            return np.zeros(2)
        velocity = np.asarray(value, dtype=float)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError("tool velocity must be a finite planar vector")
        return velocity

    @abstractmethod
    def command(self, state, target, aperture, dt, *, tool_velocity_m_s=None):
        pass


class _CartesianGripper(Gripper):
    def command(self, state, target, aperture, dt, *, tool_velocity_m_s=None):
        origin = self.model.joint_position[0, [0, 2]]
        goal = np.r_[target - origin, np.full(2, aperture)]
        carriage = np.array([165.0, 250.0]) * (goal[:2] - state.q[:2]) + 15.0 * (
            self._tool_velocity(tool_velocity_m_s) - state.v[:2]
        )
        carriage[1] += 0.22 * 9.81
        data = RobotInputs.zeros(self.model.dof_count)
        data.effort[:2] = np.clip(carriage, -self.effort_limits[:2], self.effort_limits[:2])
        data.effort[list(self.fingers)] = self.finger_effort(state, aperture)
        return data, goal


class _ArticulatedGripper(Gripper):
    def command(self, state, target, aperture, dt, *, tool_velocity_m_s=None):
        result = solve_point_ik(
            self.model,
            state.q,
            self.tool_link,
            self.tool_point,
            np.array([target[0], 0.0, target[1]]),
            active_joints=(0, 1),
            limits=self.limits,
        )
        if not result.converged:
            raise SolverFailure(f"gripper point IK {result.status}, error {result.error_m:.6g} m")
        goal = result.configuration.copy()
        goal[list(self.fingers)] = aperture
        _, _, _, mass, bias = robot_dynamics_terms(self.model, state)
        held = bias.copy()
        held[list(self.fingers)] = self.finger_effort(state, aperture)
        goal_velocity = np.zeros_like(goal)
        if tool_velocity_m_s is not None:
            desired = self._tool_velocity(tool_velocity_m_s)
            jac, _ = point_jacobians(
                self.model, state.q, [self.tool_link], self.tool_point[None, :]
            )
            planar = jac[0][[0, 2], :2]
            # Damped differential IK uses the same 1 mm/rad damping as point-position IK.
            goal_velocity[:2] = planar.T @ np.linalg.solve(
                planar @ planar.T + 1e-6 * np.eye(2), desired
            )
        drive = solve_bounded_pd(
            mass,
            bias,
            state.q,
            state.v,
            goal,
            goal_velocity,
            np.array([40.0, 35.0, 0.0, 0.0]),
            np.array([2.0, 1.4, 0.0, 0.0]),
            self.effort_limits,
            dt,
            feedforward=held,
        )
        data = RobotInputs.zeros(self.model.dof_count)
        data.effort[:] = drive.effort
        return data, goal


def _articulated_gripper(name, base_xz, lengths, direction, home_x, pad_radius=None):
    upper, fore = direction * np.asarray(lengths)
    base_y = -direction * 0.045
    reach = 0.0 if pad_radius is None else FINGER_REACH
    tip_x = fore + direction * reach
    outer = 0.0 if pad_radius is None else 2 * pad_radius + SHANK_RADIUS
    limits = JointLimits(
        np.array([-2.8, 0.05, -0.004, -0.004]), np.array([2.8, 3.05, 0.014, 0.014]), 2000.0, 8.0
    )
    model = RobotModel(
        np.array([-1, 0, 1, 1]),
        ("revolute", "revolute", "prismatic", "prismatic"),
        np.array(
            [
                [base_xz[0], base_y, base_xz[1]],
                [upper, 0.0, 0.0],
                [tip_x, -base_y - HALF_OPEN, 0.0],
                [tip_x, -base_y + HALF_OPEN, 0.0],
            ]
        ),
        np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
        np.array([[0.0, direction, 0.0], [0.0, direction, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        np.array([0.16, 0.12, 0.06, 0.06]),
        np.array(
            [
                [upper / 2, 0.0, 0.0],
                [fore / 2, 0.0, 0.0],
                [-direction * reach / 2, -outer, 0.0],
                [-direction * reach / 2, outer, 0.0],
            ]
        ),
        np.full((4, 3), 0.001),
        np.array([0.0, 0.0, -9.81]),
    )
    point = np.array([tip_x, -base_y, 0.0])
    seed = np.array([-0.5, 1.7, 0.0, 0.0])
    home = solve_point_ik(
        model,
        seed,
        1,
        point,
        np.array([home_x, 0.0, BASE_Z]),
        active_joints=(0, 1),
        limits=limits,
        tolerance_m=1e-11,
    )
    if not home.converged:
        raise ValueError("articulated gripper cannot reach its initial pose")
    wrist_y = (
        (0.0, -base_y)
        if pad_radius is None
        else (min(0.0, -base_y - HALF_OPEN - outer), max(0.0, -base_y + HALF_OPEN + outer))
    )
    shapes = [
        LinkCapsule(name + "_upper", name, 0, np.zeros(3), np.array([upper, 0.0, 0.0]), 0.008),
        LinkCapsule(name + "_fore", name, 1, np.zeros(3), np.array([fore, 0.0, 0.0]), 0.006),
        LinkCapsule(
            name + "_wrist",
            name,
            1,
            np.array([fore, wrist_y[0], 0.0]),
            np.array([fore, wrist_y[1], 0.0]),
            0.004,
        ),
    ]
    exclusions = [(name + "_upper", name + "_fore")]
    if pad_radius is not None:
        for side, link, lateral in (("left", 2, -outer), ("right", 3, outer)):
            shank = name + "_" + side + "_shank"
            shapes.append(
                LinkCapsule(
                    shank,
                    name,
                    link,
                    np.array([-direction * reach, lateral, 0.0]),
                    np.array([-direction * PAD_HALF_LENGTH, lateral, 0.0]),
                    SHANK_RADIUS,
                )
            )
            exclusions.append((name + "_wrist", shank))
    return _ArticulatedGripper(
        model,
        RobotState(home.configuration, np.zeros(4)),
        1,
        point,
        (2, 3),
        np.array([5.0, 4.0, 5.0, 5.0]),
        limits,
        tuple(shapes),
        tuple(exclusions),
    )


def grippers_for(layout, *, roles=("A", "B"), finger_pad_radius_m=None):
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= {"A", "B"}:
        raise ValueError("unknown or duplicate gripper roles")
    if layout in ("articulated", "articulated-palm"):
        radius = None
        if layout == "articulated-palm":
            radius = 0.002 if finger_pad_radius_m is None else float(finger_pad_radius_m)
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("a mounted gripper requires a positive finite pad radius")
        definitions = (
            ("A", (-0.24, 0.16), (0.16, 0.18), 1.0, -0.06),
            ("B", (0.20, 0.35), (0.26, 0.28), -1.0, 0.115),
        )
        return {
            name: _articulated_gripper(name, base, lengths, direction, home_x, radius)
            for name, base, lengths, direction, home_x in definitions
            if name in roles
        }
    if layout != "cartesian":
        raise ValueError(f"unknown handoff robot layout: {layout}")
    return {
        key: _CartesianGripper(
            _cartesian_model(base),
            RobotState(np.array([offset, 0.0, 0.0, 0.0]), np.zeros(4)),
            1,
            np.zeros(3),
            (2, 3),
            np.array([16.0, 20.0, 5.0, 5.0]),
        )
        for key, base, offset in (("A", -0.06, 0.0), ("B", 0.06, 0.055))
        if key in roles
    }

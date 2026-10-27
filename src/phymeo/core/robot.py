"""Owned FP64 fixed-base trees with one scalar joint per massive link."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._arrays import float_array


def robot_array(value, name, *, ndim, columns=None):
    result = float_array(value, name, ndim=ndim, columns=columns)
    if result.dtype != np.dtype(np.float64):
        raise TypeError(f"{name}: robot reference requires float64")
    return result


@dataclass(frozen=True, eq=False)
class RobotModel:
    """Parent-frame joint offsets, zero child-to-parent xyzw frames, local axes.

    COM and principal inertia axes are expressed in the moving link frame.
    Parent -1 attaches to the fixed identity world frame; all other parents
    precede their children. No floating base, limits, or implicit damping.
    """

    parents: np.ndarray
    joint_types: tuple[str, ...]
    joint_position: np.ndarray
    joint_orientation: np.ndarray
    joint_axis: np.ndarray
    mass: np.ndarray
    com: np.ndarray
    inertia_diagonal: np.ndarray
    gravity: np.ndarray

    def __post_init__(self):
        parents = np.asarray(self.parents)
        if parents.ndim != 1 or parents.dtype.kind not in "iu":
            raise TypeError("parents must be a rank-one integer array")
        count = len(parents)
        if count < 1 or np.any(parents < -1) or np.any(parents >= np.arange(count)):
            raise ValueError("robot needs at least one joint, with parent -1 or earlier index")
        if isinstance(self.joint_types, str):
            raise TypeError("joint_types must be a collection")
        kinds = tuple(self.joint_types)
        if len(kinds) != count or any(
            not isinstance(kind, str) or kind not in ("revolute", "prismatic") for kind in kinds
        ):
            raise ValueError("one revolute/prismatic joint is required per link")
        arrays = {"parents": np.array(parents, dtype=np.int64, copy=True)}
        for name, columns in (
            ("joint_position", 3),
            ("joint_orientation", 4),
            ("joint_axis", 3),
            ("com", 3),
            ("inertia_diagonal", 3),
        ):
            value = robot_array(getattr(self, name), name, ndim=2, columns=columns)
            if len(value) != count:
                raise ValueError(f"{name} must have one row per joint")
            arrays[name] = value
        arrays["mass"] = robot_array(self.mass, "mass", ndim=1)
        arrays["gravity"] = robot_array(self.gravity, "gravity", ndim=1)
        if arrays["mass"].shape != (count,) or arrays["gravity"].shape != (3,):
            raise ValueError("mass or gravity shape does not match the robot")
        if np.any(arrays["mass"] <= 0):
            raise ValueError("robot link mass must be positive")
        for name in ("joint_axis", "joint_orientation"):
            norms = np.hypot.reduce(arrays[name], axis=1)
            if np.any(np.abs(norms - 1) > 1e-10):
                raise ValueError(f"{name} must have unit length")
        inertia = arrays["inertia_diagonal"]
        if np.any(inertia <= 0):
            raise ValueError("COM principal inertia must be positive")
        scaled = inertia / np.max(inertia, axis=1, keepdims=True)
        if np.any(2 * np.max(scaled, axis=1) > np.sum(scaled, axis=1) + 8 * np.finfo(float).eps):
            raise ValueError("COM principal inertia violates triangle inequalities")
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "joint_types", kinds)

    @property
    def dof_count(self):
        return len(self.parents)


@dataclass(eq=False)
class RobotState:
    """Continuous scalar joint coordinates q and velocities v; no angle wrapping."""

    q: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        q, v = robot_array(self.q, "q", ndim=1), robot_array(self.v, "v", ndim=1)
        if q.shape != v.shape:
            raise ValueError("q and v must have identical shape")
        self.q, self.v = q, v


@dataclass(eq=False)
class RobotInputs:
    """Generalized efforts plus world COM forces and torques, excluding gravity."""

    effort: np.ndarray
    force: np.ndarray
    torque: np.ndarray

    def __post_init__(self):
        effort = robot_array(self.effort, "effort", ndim=1)
        force = robot_array(self.force, "force", ndim=2, columns=3)
        torque = robot_array(self.torque, "torque", ndim=2, columns=3)
        if force.shape != torque.shape or force.shape != (len(effort), 3):
            raise ValueError("effort/force/torque must have the same link count")
        self.effort, self.force, self.torque = effort, force, torque

    @classmethod
    def zeros(cls, count):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
            raise ValueError("count must be a non-negative integer")
        return cls(np.zeros(count), np.zeros((count, 3)), np.zeros((count, 3)))


def validate_robot_step(model, state, inputs):
    if not isinstance(model, RobotModel):
        raise TypeError("expected RobotModel")
    if not isinstance(state, RobotState) or not isinstance(inputs, RobotInputs):
        raise TypeError("expected RobotState and RobotInputs")
    count = model.dof_count
    for name, value, shape in (
        ("q", state.q, (count,)),
        ("v", state.v, (count,)),
        ("effort", inputs.effort, (count,)),
        ("force", inputs.force, (count, 3)),
        ("torque", inputs.torque, (count, 3)),
    ):
        value = np.asarray(value)
        if value.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} must use float64")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} has invalid shape or non-finite values")

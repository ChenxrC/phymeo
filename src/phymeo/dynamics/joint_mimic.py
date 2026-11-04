"""One passive scalar URDF linkage, expressed in independent generalized coordinates."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotInputs, RobotState, validate_robot_step
from ..core.validation import finite_real, validate_dt
from .robot_dynamics import _solve_mass_matrix, robot_dynamics_terms

if TYPE_CHECKING:
    from ..builders._urdf_frames import FoldedURDFAsset


def _frozen(values):
    source = np.asarray(values, dtype=np.float64)
    return np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)


def _vector(value, count, name):
    a = np.asarray(value)
    if a.shape != (count,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must have {count} finite real entries")
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class MimicLoads:
    """Full physical joint loads at the input time. The motor acts only on independent joints.

    constraint_effort_full is the mechanism's force on the tree, positive along each
    authored joint coordinate. multiplier_force is its follower component (N or N m).
    contact_effort_full includes all non-actuator generalized loads, excluding gravity.
    """

    actuator_effort_full: np.ndarray
    contact_effort_full: np.ndarray
    constraint_effort_full: np.ndarray
    acceleration_full: np.ndarray
    multiplier_force: float
    constraint_power_W: float


@dataclass(frozen=True, eq=False)
class MimicStep:
    state: RobotState
    full_state: RobotState
    loads: MimicLoads


@dataclass(frozen=True, eq=False)
class MimicRobotAsset:
    """One constrained tree q_full=P q+offset. tree is the explicitly unconstrained full model.

    joint_names, limits, and public motion commands refer to the independent coordinates.
    Raw body ownership and material points remain in tree. The follower has no motor.
    """

    tree: "FoldedURDFAsset"
    driver: str
    follower: str
    multiplier: float
    offset: float
    driver_index: int = field(init=False)
    follower_index: int = field(init=False)
    independent_indices: tuple = field(init=False)
    matrix: np.ndarray = field(init=False, repr=False)
    offset_vector: np.ndarray = field(init=False, repr=False)
    constraint_row: np.ndarray = field(init=False, repr=False)
    joint_names: tuple = field(init=False)
    lower: tuple = field(init=False)
    upper: tuple = field(init=False)
    velocity_limit: tuple = field(init=False)
    effort_limit: tuple = field(init=False)

    def __post_init__(self):
        names = self.tree.joint_names
        if self.driver not in names or self.follower not in names or self.driver == self.follower:
            raise ValueError(
                "mimic requires distinct existing independent driver and follower joints"
            )
        d, f = names.index(self.driver), names.index(self.follower)
        if self.tree.model.joint_types[d] != self.tree.model.joint_types[f]:
            raise ValueError("mimic driver and follower must use the same scalar joint type")
        a = finite_real(self.multiplier, "mimic multiplier")
        b = finite_real(self.offset, "mimic offset")
        if a == 0:
            raise ValueError("a zero mimic multiplier is outside this mechanical linkage")
        indices = tuple(i for i in range(len(names)) if i != f)
        column = indices.index(d)
        P = np.eye(len(names))[:, indices]
        P[f, column] = a
        off = np.zeros(len(names))
        off[f] = b
        row = np.zeros(len(names))
        row[d] = -a
        row[f] = 1.0
        selected = list(indices)
        lower = np.asarray(self.tree.lower)[selected].copy()
        upper = np.asarray(self.tree.upper)[selected].copy()
        lo, hi = sorted(((self.tree.lower[f] - b) / a, (self.tree.upper[f] - b) / a))
        if not np.isfinite((lo, hi)).all():
            raise ValueError("mimic position interval is outside FP64 range")
        lower[column] = max(lower[column], lo)
        upper[column] = min(upper[column], hi)
        if np.any(lower >= upper):
            raise ValueError("empty movable intersection of driver and follower position limits")
        speed = np.array(self.tree.velocity_limit, dtype=float)[selected].copy()
        speed[column] = min(speed[column], self.tree.velocity_limit[f] / abs(a))
        for key, value in (
            ("driver_index", d),
            ("follower_index", f),
            ("independent_indices", indices),
            ("matrix", _frozen(P)),
            ("offset_vector", _frozen(off)),
            ("constraint_row", _frozen(row)),
            ("joint_names", tuple(names[i] for i in indices)),
            ("lower", tuple(lower)),
            ("upper", tuple(upper)),
            ("velocity_limit", tuple(speed)),
            ("effort_limit", tuple(self.tree.effort_limit[i] for i in indices)),
        ):
            object.__setattr__(self, key, value)

    def configuration(self, q):
        u = _vector(q, len(self.joint_names), "independent q")
        if np.any(u < self.lower) or np.any(u > self.upper):
            raise ValueError("joint position outside driver/follower intersection")
        full = self.matrix @ u + self.offset_vector
        if not np.isfinite(full).all():
            raise ValueError("expanded mimic position is outside FP64 range")
        return full

    def full_state(self, state):
        if not isinstance(state, RobotState):
            raise TypeError("expected an independent RobotState")
        q = self.configuration(state.q)
        v = _vector(state.v, len(self.joint_names), "independent velocity")
        if np.any(abs(v) > self.velocity_limit):
            raise ValueError("joint velocity outside driver/follower limit")
        return RobotState(q, self.matrix @ v)

    def reduce_state(self, full):
        """Accept existing full coordinates only if their positions and velocities obey the mechanism."""
        if not isinstance(full, RobotState) or len(full.q) != len(self.tree.joint_names):
            raise ValueError("expected a full RobotState")
        i = list(self.independent_indices)
        reduced = RobotState(full.q[i], full.v[i])
        check = self.full_state(reduced)
        for measured, expected in ((full.q, check.q), (full.v, check.v)):
            scale = np.maximum(1.0, np.maximum(abs(measured), abs(expected)))
            if np.any(abs(measured - expected) > 16 * np.finfo(float).eps * scale):
                raise ValueError("inconsistent full state for mimic constraint")
        return reduced

    def collision_motion(self, q, q_end=None):
        end = None if q_end is None else self.configuration(q_end)
        return self.tree.collision_motion(self.configuration(q), end)

    def authored_link_pose(self, name, q):
        return self.tree.authored_link_pose(name, self.configuration(q))

    def project_jacobian(self, jacobian):
        full = np.asarray(jacobian)
        if (
            full.ndim < 1
            or full.shape[-1] != self.matrix.shape[0]
            or full.dtype.kind not in "fiu"
            or not np.isfinite(full).all()
        ):
            raise ValueError("Jacobian must use the full physical joint columns")
        return full @ self.matrix

    def mass_bias(self, state):
        """Full tree evaluated at the expanded state, then projected into the independent coordinates."""
        *_, mass, bias = robot_dynamics_terms(self.tree.model, self.full_state(state))
        return self.matrix.T @ mass @ self.matrix, self.matrix.T @ bias

    def _receipt(self, state, mass, bias, effort, external, acceleration):
        motor = np.zeros(len(bias))
        motor[list(self.independent_indices)] = effort
        full_acc = self.matrix @ acceleration
        inertial = mass @ full_acc
        balance = inertial + bias - motor - external
        multiplier = float(balance[self.follower_index])
        reaction = multiplier * self.constraint_row
        scale = max(
            1.0, *(np.linalg.norm(x, np.inf) for x in (inertial, bias, motor, external, reaction))
        )
        if (
            not np.isfinite(balance).all()
            or not np.isfinite(scale)
            or np.linalg.norm(balance - reaction, np.inf) > 2e-10 * scale
        ):
            raise SolverFailure("mimic full-coordinate force balance failed")
        power = float(reaction @ (self.matrix @ state.v))
        if not np.isfinite(power):
            raise SolverFailure("mimic constraint power is not finite")
        return MimicLoads(*map(_frozen, (motor, external, reaction, full_acc)), multiplier, power)

    def step(self, state, *, effort, dt_s, external=None):
        """Advance the constrained robot one HOST step. external contains full-tree world COM wrenches.

        Additional generalized effort in external describes a non-motor physical load.
        A rejected candidate leaves state intact. This input does not impose limit-stop impulses.
        """
        h = float(validate_dt(dt_s, np.float64))
        full = self.full_state(state)
        effort = _vector(effort, len(self.joint_names), "actuator effort")
        if np.any(abs(effort) > self.effort_limit):
            raise ValueError("actuator effort exceeds an independent motor rating")
        if external is None:
            external = RobotInputs.zeros(len(full.q))
        validate_robot_step(self.tree.model, full, external)
        _, J, W, mass, bias = robot_dynamics_terms(self.tree.model, full)
        load = (
            external.effort
            + np.einsum("nki,nk->i", J, external.force)
            + np.einsum("nki,nk->i", W, external.torque)
        )
        P = self.matrix
        accel = _solve_mass_matrix(P.T @ mass @ P, effort + P.T @ (load - bias))
        new_v = state.v + h * accel
        new = RobotState(state.q + h * new_v, new_v)
        return MimicStep(
            new, self.full_state(new), self._receipt(state, mass, bias, effort, load, accel)
        )

"""HOST affine scalar mechanisms on one real free root and its full physical tree.

The tree retains every body, inertia and material owner. Only state and motor
coordinates are reduced. No scene, PD, contact law, CUDA step or limit impulse
is introduced here.
"""

from dataclasses import dataclass, field

import numpy as np

from ..builders.floating_urdf import FloatingRobotAsset
from ..core.capabilities import SolverFailure
from ..core.validation import validate_dt
from ..solvers.floating_tree import (
    FloatingRobotInputs,
    _drift_floating_state,
    floating_dynamics_terms,
)
from ._mimic_coordinates import MimicCoordinates, _frozen, compile_mimic_coordinates
from .floating_robot import FloatingRobotState
from .joint_mimic import MimicLoads, MimicRelation, MimicRowForce
from .robot_dynamics import _finite, _numerical, _solve_mass_matrix


def _vector(values, count, name):
    source = np.asarray(values)
    if source.shape != (count,) or source.dtype.kind not in "fiu" or not np.isfinite(source).all():
        raise ValueError(f"{name} must have {count} finite real entries")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(source, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is outside FP64 range")
    return result


def _scale(*vectors):
    result = max(1.0, *(float(np.linalg.norm(value, np.inf)) for value in vectors))
    if not np.isfinite(result):
        raise SolverFailure("floating mimic balance scale is not finite")
    return result


@dataclass(frozen=True, eq=False)
class FloatingMimicStep:
    """One unpublished candidate and input-time complete force receipt.

    state uses 7+k / 6+k independent coordinates; full_state uses 7+n / 6+n.
    Every loads.*_full vector includes root COM force, root torque, then all n
    scalar joints. contact_effort_full includes external non-actuator loads;
    gravity remains in bias. All row powers refer to the input full velocity.
    """

    state: FloatingRobotState
    full_state: FloatingRobotState
    loads: MimicLoads


@dataclass(frozen=True, eq=False)
class FloatingMimicRobotAsset:
    """Wrap a complete FloatingRobotAsset with constant qj=P*u+c.

    tree owns full link/shape/material metadata. joint_names and all limits on
    this wrapper describe independent scalar coordinates only. Public root q
    remains the authored origin/orientation; v uses the root body's world COM
    velocity and world omega, not the aggregate system COM.
    """

    tree: FloatingRobotAsset
    relations_tuple: tuple
    relations: tuple = field(init=False)
    matrix: np.ndarray = field(init=False, repr=False)
    offset_vector: np.ndarray = field(init=False, repr=False)
    constraint_rows: np.ndarray = field(init=False, repr=False)
    independent_indices: tuple = field(init=False)
    velocity_projection: np.ndarray = field(init=False, repr=False)
    joint_names: tuple = field(init=False)
    lower: tuple = field(init=False)
    upper: tuple = field(init=False)
    velocity_limit: tuple = field(init=False)
    effort_limit: tuple = field(init=False)
    _coordinates: MimicCoordinates = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.tree, FloatingRobotAsset):
            raise TypeError("tree must be a full FloatingRobotAsset")
        children = self.tree.model.children
        count = len(self.tree.joint_names)
        if children is None or self.tree.model.body_count != count + 1 or len(self.tree.link_names) != count:
            raise ValueError("floating mimic requires a complete physical scalar-child tree")
        coordinates = compile_mimic_coordinates(
            self.tree.joint_names,
            children.joint_types,
            self.tree.lower,
            self.tree.upper,
            self.tree.velocity_limit,
            self.tree.effort_limit,
            self.relations_tuple,
        )
        raw = tuple(
            (self.tree.joint_names[d], self.tree.joint_names[f], a, b)
            for d, f, a, b in coordinates.indices
        )
        projection = np.zeros((6 + count, 6 + len(coordinates.independent_indices)))
        projection[:6, :6] = np.eye(6)
        projection[6:, 6:] = coordinates.matrix
        for name in (
            "matrix", "offset_vector", "constraint_rows", "independent_indices",
            "joint_names", "lower", "upper", "velocity_limit", "effort_limit",
        ):
            object.__setattr__(self, name, getattr(coordinates, name))
        object.__setattr__(self, "relations_tuple", raw)
        object.__setattr__(self, "relations", tuple(MimicRelation(*row) for row in raw))
        object.__setattr__(self, "velocity_projection", _frozen(projection))
        object.__setattr__(self, "_coordinates", coordinates)

    def configuration(self, independent):
        """Expand only scalar joint positions; root coordinates are not implicit."""
        q = _vector(independent, len(self.joint_names), "independent joint position")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("joint position outside driver/follower intersection")
        with np.errstate(over="ignore", invalid="ignore"):
            full = self.matrix @ q + self.offset_vector
        if not np.isfinite(full).all():
            raise ValueError("expanded mimic position is outside FP64 range")
        return _frozen(full)

    def full_state(self, state):
        """Own and expand a valid independent free-root state into physical order."""
        if not isinstance(state, FloatingRobotState):
            raise TypeError("expected an independent FloatingRobotState")
        count = len(self.joint_names)
        q = _vector(state.q, 7 + count, "independent floating position")
        v = _vector(state.v, 6 + count, "independent floating velocity")
        owned = FloatingRobotState(q, v)
        full_q = self.configuration(owned.q[7:])
        if np.any(abs(owned.v[6:]) > self.velocity_limit):
            raise ValueError("joint velocity outside driver/follower limit")
        with np.errstate(over="ignore", invalid="ignore"):
            full_v = self.matrix @ owned.v[6:]
        if not np.isfinite(full_v).all():
            raise ValueError("expanded mimic velocity is outside FP64 range")
        return FloatingRobotState(np.r_[owned.q[:7], full_q], np.r_[owned.v[:6], full_v])

    def reduce_state(self, full):
        """Reject inconsistent full observations; never silently project a state."""
        if not isinstance(full, FloatingRobotState):
            raise TypeError("expected a full FloatingRobotState")
        count = len(self.tree.joint_names)
        q = _vector(full.q, 7 + count, "full floating position")
        v = _vector(full.v, 6 + count, "full floating velocity")
        owned = FloatingRobotState(q, v)
        indices = np.asarray(self.independent_indices, dtype=np.int64)
        reduced = FloatingRobotState(
            np.r_[owned.q[:7], owned.q[7 + indices]],
            np.r_[owned.v[:6], owned.v[6 + indices]],
        )
        expected = self.full_state(reduced)
        for actual, canonical in ((owned.q, expected.q), (owned.v, expected.v)):
            scale = np.maximum(1.0, np.maximum(abs(actual), abs(canonical)))
            if np.any(abs(actual - canonical) > 16 * np.finfo(float).eps * scale):
                raise ValueError("inconsistent full state for floating mimic constraint")
        return reduced

    def collision_motion(self, q, q_end=None):
        """Read actual full-tree static poses; do not invent a floating CCD chord."""
        if q_end is not None:
            raise ValueError("floating mimic interval collision motion is not implemented")
        state = FloatingRobotState(q, np.zeros(6 + len(self.joint_names)))
        return self.tree.collision_motion(self.full_state(state).q)

    def _receipt(self, full, terms, motor, acceleration):
        """Recover each authored row after descendant forces, including root audit."""
        inertial = terms.mass_matrix @ acceleration
        balance = _finite(inertial + terms.bias - motor - terms.applied)
        root_scale = _scale(inertial[:6], terms.bias[:6], motor[:6], terms.applied[:6])
        if np.linalg.norm(balance[:6], np.inf) > 2e-10 * root_scale:
            raise SolverFailure("floating mimic root force balance failed")
        remaining = balance[6:].copy()
        multipliers = np.zeros(len(self.relations))
        for row in reversed(self._coordinates.forward):
            d, f, ratio, _ = self._coordinates.indices[row]
            multipliers[row] = remaining[f]
            remaining[d] += ratio * multipliers[row]
        reaction = _finite(np.r_[np.zeros(6), self.constraint_rows.T @ multipliers])
        scale = _scale(inertial, terms.bias, motor, terms.applied, reaction)
        if np.linalg.norm(balance - reaction, np.inf) > 2e-10 * scale:
            raise SolverFailure("floating mimic full-coordinate force balance failed")
        projected = _finite(self.velocity_projection.T @ reaction)
        projected_scale = _scale(reaction) * _scale(self.velocity_projection)
        if not np.isfinite(projected_scale) or np.linalg.norm(projected, np.inf) > 2e-10 * projected_scale:
            raise SolverFailure("floating mimic constraint has independent generalized work")
        power = float(reaction @ full.v)
        power_scale = max(1.0, float(np.sum(abs(reaction) * abs(full.v))))
        if not np.isfinite((power, power_scale)).all() or abs(power) > 2e-10 * power_scale:
            raise SolverFailure("floating mimic total constraint power failed")
        rows = []
        for relation, (d, f, ratio, _), value in zip(
            self.relations, self._coordinates.indices, multipliers, strict=True
        ):
            value = float(value)
            driver = -ratio * value
            row_power = float(value * (full.v[6 + f] - ratio * full.v[6 + d]))
            row_scale = max(1.0, abs(value * full.v[6 + f]), abs(driver * full.v[6 + d]))
            if not np.isfinite((driver, row_power, row_scale)).all() or abs(row_power) > 2e-10 * row_scale:
                raise SolverFailure(f"floating mimic row to {relation.follower!r} has invalid power")
            rows.append(MimicRowForce(relation.driver, relation.follower, value, driver, value, row_power))
        if abs(sum(row.constraint_power_W for row in rows) - power) > 2e-10 * power_scale:
            raise SolverFailure("floating mimic row and total powers disagree")
        return MimicLoads(
            _frozen(motor),
            _frozen(terms.applied),
            _frozen(reaction),
            _frozen(acceleration),
            float(multipliers[0]) if len(rows) == 1 else None,
            power,
            tuple(rows),
        )

    @_numerical
    def step(self, state, *, effort, dt_s, external=None):
        """Form one constrained HOST candidate without changing any input.

        Motor effort has k entries. External COM force/torque has all n+1
        physical rows and external.effort must be n zeros. All returned load
        vectors have 6+n entries and are sampled at this input state.
        """
        h = validate_dt(dt_s, np.float64)
        full = self.full_state(state)
        entry = self.reduce_state(full)
        count = len(self.tree.joint_names)
        drive = _vector(effort, len(self.joint_names), "independent actuator effort")
        if np.any(abs(drive) > self.effort_limit):
            raise ValueError("actuator effort exceeds an independent motor rating")
        if external is None:
            external = FloatingRobotInputs.zeros(count)
        if not isinstance(external, FloatingRobotInputs):
            raise TypeError("external must be full FloatingRobotInputs")
        external = FloatingRobotInputs(external.effort, external.force, external.torque)
        if external.effort.shape != (count,) or external.force.shape != (count + 1, 3):
            raise ValueError("external COM loads must cover the full physical tree")
        if np.any(external.effort):
            raise ValueError("external joint effort must be zero; use independent motor effort")
        # Reading full dynamics is not an unconstrained physical step. Only the
        # constrained velocity below reaches the shared drift implementation.
        terms = floating_dynamics_terms(self.tree.model, full, external)
        motor = np.zeros(6 + count)
        motor[6 + np.asarray(self.independent_indices, dtype=np.int64)] = drive
        projection = self.velocity_projection
        mass = _finite(projection.T @ terms.mass_matrix @ projection)
        applied = _finite(projection.T @ (terms.applied + motor - terms.bias))
        acceleration = _solve_mass_matrix(mass, applied)
        full_acceleration = _finite(projection @ acceleration)
        velocity = _finite(entry.v + h * acceleration)
        full_velocity = _finite(projection @ velocity)
        drifted = _drift_floating_state(self.tree.model, full, full_velocity, h)
        candidate = self.reduce_state(drifted)
        # Keep the public affine coordinates canonical after validating the
        # constrained drift; this is only the same constant map's roundoff.
        canonical = self.full_state(candidate)
        receipt = self._receipt(full, terms, motor, full_acceleration)
        return FloatingMimicStep(candidate, canonical, receipt)

"""Pure shared-scene participant for a physical free root and scalar-joint tree.

Contact geometry uses body rows, generalized loads use 6+n velocity rows, and
configuration uses 7+n coordinates. No publication or second integration occurs
here; MultiRobotScene commits every candidate and its event snapshot together.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..builders.floating_urdf import FloatingRobotAsset, load_floating_urdf
from ..constraints.collision_shapes import CollisionKey
from ..control.bounded_pd import solve_bounded_pd
from ..core.validation import validate_dt
from ..dynamics.floating_robot import FloatingRobotState, floating_kinematics, floating_point_jacobians
from ..dynamics.robot_dynamics import _solve_mass_matrix
from ..solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms, step_floating_robot
from .robot_payload_scene import _readonly, _vector
from .robot_scene import RobotCommand, _asset_snapshot


@dataclass(frozen=True, eq=False)
class FloatingRobotCommand:
    """Bounded child drive plus explicit world force/COM torque on root and links.

    joints is None only for a root-only body. external.effort must be zero:
    internal joint actuation belongs exclusively to the bounded RobotCommand.
    None external means zero applied COM forces/torques, not zero gravity.
    """

    joints: RobotCommand | None
    external: FloatingRobotInputs | None = None

    def __post_init__(self):
        if self.joints is not None and not isinstance(self.joints, RobotCommand):
            raise TypeError("joints must be a RobotCommand or None")
        if self.external is not None:
            if not isinstance(self.external, FloatingRobotInputs):
                raise TypeError("external must be FloatingRobotInputs or None")
            if np.any(self.external.effort):
                raise ValueError("external.effort must be zero; use the bounded joint drive")
        object.__setattr__(self, "joints", _asset_snapshot(self.joints))
        object.__setattr__(self, "external", _asset_snapshot(self.external))


@dataclass(frozen=True, eq=False)
class FloatingRobotRegistration:
    asset: FloatingRobotAsset
    q: np.ndarray
    v: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    pads: tuple = ()


def make_floating_registration(path, *, robot_state, kp, kd, effort_limit=None, pads=(), loader=None, gravity=(0.0, 0.0, -9.81)):
    """Load and isolate one real asset; floating pads have no implicit adaptation."""
    if tuple(pads):
        raise ValueError("floating robot pads are not implemented")
    asset = (load_floating_urdf if loader is None else loader)(path, gravity=tuple(gravity))
    if not isinstance(asset, FloatingRobotAsset):
        raise TypeError("floating loader must return FloatingRobotAsset")
    n = len(asset.joint_names)
    if asset.model.body_count != n + 1 or not isinstance(robot_state, FloatingRobotState) or robot_state.q.shape != (7 + n,):
        raise ValueError("floating robot state must match the root and scalar-joint order")
    if any(len(values) != n for values in (asset.link_names, asset.lower, asset.upper, asset.effort_limit, asset.velocity_limit)):
        raise ValueError("floating asset metadata must match its scalar children")
    kp, kd = _vector(kp, n, "kp"), _vector(kd, n, "kd")
    if np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("controller gains must be nonnegative")
    hard = np.array([np.inf if value is None else value for value in asset.effort_limit], dtype=float)
    cap = _vector(hard if effort_limit is None else effort_limit, n, "effort_limit")
    if np.any(cap < 0) or np.any(cap > hard):
        raise ValueError("effort limit is negative or exceeds the URDF rating")
    if not np.array_equal(asset.model.root.gravity, np.asarray(gravity, dtype=float)) or (asset.model.children is not None and not np.array_equal(asset.model.children.gravity, asset.model.root.gravity)):
        raise ValueError("floating loader must preserve the scene gravity for every body")
    return FloatingRobotRegistration(_asset_snapshot(asset), _readonly(robot_state.q), _readonly(robot_state.v), kp, kd, cap)


@dataclass(frozen=True)
class _FloatingContactView:
    """Only actual body-local collider data; no fabricated scalar-DOF model."""

    name: str
    root_link: str
    link_names: tuple
    collisions: tuple
    collision_model: object
    root_xyz: tuple = (0.0, 0.0, 0.0)
    root_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    is_floating: bool = True


@dataclass(frozen=True, eq=False)
class FloatingRobotObservation:
    """Full accepted q/v and load-time COM wrenches; every numeric array is owned.

    applied_effort_full includes contact, external COM loads and bounded internal
    drive, while gravity and analytic inertial terms remain in the dynamics bias.
    fixed_root_contact_wrench is zero because body zero is physically moving.
    """

    q: np.ndarray
    v: np.ndarray
    full_q: np.ndarray
    full_v: np.ndarray
    entry_q: np.ndarray
    entry_v: np.ndarray
    drive_effort: np.ndarray
    contact_effort: np.ndarray
    contact_effort_full: np.ndarray
    link_wrenches_world: np.ndarray
    fixed_root_contact_wrench: np.ndarray
    command: FloatingRobotCommand | None
    requested_command: FloatingRobotCommand | None
    external: FloatingRobotInputs
    applied_effort_full: np.ndarray
    acceleration_full: np.ndarray
    mimic_loads: object | None = None


@dataclass(frozen=True, eq=False)
class FloatingRobotPreparation:
    entry: FloatingRobotState
    physical: FloatingRobotState
    geometry: object
    linear: np.ndarray
    angular: np.ndarray
    full_mass: np.ndarray
    full_bias: np.ndarray
    effort: np.ndarray
    external: FloatingRobotInputs
    resolved: FloatingRobotCommand
    requested: FloatingRobotCommand


def _project(linear, angular, force, torque):
    return np.einsum("nki,nk->i", linear, force) + np.einsum("nki,nk->i", angular, torque)


class FloatingRobotParticipant:
    """Independent full-M candidate step inside the parent's one-scene transaction."""

    is_floating = True

    def __init__(self, address, registration, occupied):
        if not isinstance(registration, FloatingRobotRegistration):
            raise TypeError("expected FloatingRobotRegistration")
        self.address, self.spec = address, _asset_snapshot(registration)
        self.asset = self.spec.asset
        self.body_count = self.asset.model.body_count
        self.velocity_count = 6 + len(self.asset.joint_names)
        self.lower, self.upper = np.asarray(self.asset.lower), np.asarray(self.asset.upper)
        self.speed_limit = np.array([np.inf if value is None else value for value in self.asset.velocity_limit])
        self.initial = FloatingRobotState(self.spec.q, self.spec.v)
        self.validate(self.initial)
        collisions = tuple(replace(row, link_index=row.link_index + 1) for row in self.asset.collisions)
        if any(row.link_index < 0 or row.link_index >= self.body_count for row in collisions):
            raise ValueError("floating collider owner is outside the physical body list")
        self.contact_view = _FloatingContactView(self.asset.name, self.asset.root_link, (self.asset.root_link, *self.asset.link_names), collisions, self.asset.collision_model)
        body_remap, next_body = {}, 0
        for body in sorted({shape.key.body for shape in self.asset.collision_model.shapes}):
            while next_body in occupied:
                next_body += 1
            body_remap[body] = next_body
            occupied.add(next_body)
        keys, names, semantic, shapes = {}, {}, {}, []
        for collision in collisions:
            source = collision.shape
            if source.key in keys:
                raise ValueError("floating asset contains duplicate source collision keys")
            key = CollisionKey(address[0], body_remap[source.key.body], source.key.shape)
            keys[source.key] = key
            shapes.append(replace(source, key=key))
            names[key] = f"{address[1]}:{collision.link_name}/{collision.name}"
            if collision.name_is_authored:
                semantic.setdefault((collision.link_name, collision.name), []).append(key)
        if set(keys) != {shape.key for shape in self.asset.collision_model.shapes}:
            raise ValueError("floating collision geometry does not match the model keys")
        self.key_map, self.names = MappingProxyType(keys), MappingProxyType(names)
        self.named_keys = MappingProxyType({key: tuple(value) for key, value in semantic.items()})
        self.shapes = tuple(shapes)

    @staticmethod
    def copy_state(state):
        return FloatingRobotState(state.q, state.v)

    def validate(self, state):
        if not isinstance(state, FloatingRobotState) or state.q.shape != (self.velocity_count + 1,) or state.v.shape != (self.velocity_count,):
            raise ValueError("floating state must match the physical root and children")
        if not np.isfinite(state.q).all() or not np.isfinite(state.v).all() or abs(np.hypot.reduce(state.q[3:7]) - 1.0) > 1e-10:
            raise ValueError("floating state must be finite with unit XYZW orientation")
        if np.any(state.q[7:] < self.lower) or np.any(state.q[7:] > self.upper):
            raise ValueError("joint position outside the asset interval")
        if np.any(abs(state.v[6:]) > self.speed_limit):
            raise ValueError("joint velocity outside the asset interval")

    def full(self, state):
        self.validate(state)
        return state

    def terms(self, state):
        self.validate(state)
        model = self.asset.model
        geometry = floating_kinematics(model, state)
        local_com = model.root_com_m[None, :]
        if model.children is not None:
            local_com = np.vstack((local_com, model.children.com))
        linear, angular = floating_point_jacobians(model, state, np.arange(self.body_count), local_com)
        dynamics = floating_dynamics_terms(model, state, FloatingRobotInputs.zeros(self.body_count - 1))
        return state, geometry, linear, angular, dynamics.mass_matrix, dynamics.bias

    def poses(self, state):
        self.validate(state)
        source = self.asset.collision_motion(state.q)
        if set(source) != set(self.key_map):
            raise ValueError("floating collision poses do not match the compiled model")
        return {key: source[old] for old, key in self.key_map.items()}

    def _observation(self, state, entry, effort, contact, wrenches, external, applied, acceleration, command=None, requested=None):
        return FloatingRobotObservation(
            q=_readonly(state.q), v=_readonly(state.v), full_q=_readonly(state.q), full_v=_readonly(state.v),
            entry_q=_readonly(entry.q), entry_v=_readonly(entry.v), drive_effort=_readonly(effort),
            contact_effort=_readonly(contact), contact_effort_full=_readonly(contact),
            link_wrenches_world=_readonly(wrenches), fixed_root_contact_wrench=_readonly(np.zeros(6)),
            command=command, requested_command=requested, external=_asset_snapshot(external),
            applied_effort_full=_readonly(applied), acceleration_full=_readonly(acceleration),
        )

    def initial_observation(self):
        zero = np.zeros(self.velocity_count)
        return self._observation(
            self.initial, self.initial, np.zeros(self.body_count - 1), zero,
            np.zeros((self.body_count, 6)), FloatingRobotInputs.zeros(self.body_count - 1), zero, zero,
        )

    def prepare(self, state, command, dt):
        """Choose one bounded motor effort with the free root's Schur response.

        The predictor includes declared external COM force/torque and the full
        analytic gravity/inertial bias. Contact is sampled afterwards and is not
        hidden in a PD feedforward. None feedforward resolves to this predictor's
        effective bias, matching the ordinary command's compensation convention.
        """
        if not isinstance(command, FloatingRobotCommand):
            raise TypeError("floating action must be a FloatingRobotCommand")
        h = validate_dt(dt, np.float64)
        self.validate(state)
        entry = self.copy_state(state)
        physical, geometry, linear, angular, mass, bias = self.terms(entry)
        n = self.body_count - 1
        external = FloatingRobotInputs.zeros(n) if command.external is None else command.external
        if external.force.shape != (self.body_count, 3) or external.effort.shape != (n,) or np.any(external.effort):
            raise ValueError("external COM loads must cover root and children with zero effort")
        external = _asset_snapshot(external)
        if not n:
            if command.joints is not None:
                raise ValueError("a root-only body requires joints=None")
            effort, resolved = np.zeros(0), FloatingRobotCommand(None, external)
        else:
            joint = command.joints
            if not isinstance(joint, RobotCommand):
                raise TypeError("floating scalar children require a RobotCommand")
            target = _vector(joint.target_q, n, "target_q")
            rate = np.zeros(n) if joint.target_velocity is None else _vector(joint.target_velocity, n, "target_velocity")
            if np.any(target < self.lower) or np.any(target > self.upper):
                raise ValueError("joint position target outside the asset interval")
            if np.any(abs(rate) > self.speed_limit):
                raise ValueError("joint velocity target outside the asset interval")
            cap = self.spec.effort_limit if joint.effort_limit is None else _vector(joint.effort_limit, n, "effort_limit")
            if np.any(cap < 0) or np.any(cap > self.spec.effort_limit):
                raise ValueError("command effort limit exceeds configured rating")
            # Eliminate the six physical root accelerations, retaining every
            # root-child cross block. This is not the fixed-root child submatrix.
            root_mass = mass[:6, :6]
            roots_per_joint = np.column_stack([_solve_mass_matrix(root_mass, column) for column in mass[:6, 6:].T])
            effective_mass = mass[6:, 6:] - mass[6:, :6] @ roots_per_joint
            free_bias = bias - _project(linear, angular, external.force, external.torque)
            effective_bias = free_bias[6:] - mass[6:, :6] @ _solve_mass_matrix(root_mass, free_bias[:6])
            ff = effective_bias if joint.feedforward is None else _vector(joint.feedforward, n, "feedforward")
            effort = solve_bounded_pd(
                effective_mass, effective_bias, entry.q[7:], entry.v[6:], target, rate,
                self.spec.kp, self.spec.kd, cap, h, feedforward=ff,
            ).effort
            resolved = FloatingRobotCommand(RobotCommand(target, rate, cap, ff), external)
        return FloatingRobotPreparation(
            entry, physical, geometry, linear, angular, mass, bias, _readonly(effort),
            external, resolved, _asset_snapshot(command),
        )

    def integrate(self, prepared, measured, dt):
        """Apply every COM wrench and limited drive once; return an unpublished candidate."""
        h = validate_dt(dt, np.float64)
        contact = _vector(measured.joint_effort_full, self.velocity_count, "floating contact effort")
        wrenches = np.asarray(measured.link_wrenches_world, dtype=float)
        fixed_root = np.asarray(measured.fixed_root_contact_wrench, dtype=float)
        if wrenches.shape != (self.body_count, 6) or not np.isfinite(wrenches).all() or fixed_root.shape != (6,) or not np.isfinite(fixed_root).all() or np.any(fixed_root):
            raise ValueError("floating contact requires one COM wrench per body and no fixed-root load")
        inputs = FloatingRobotInputs(
            prepared.effort,
            prepared.external.force + wrenches[:, :3],
            prepared.external.torque + wrenches[:, 3:],
        )
        result = step_floating_robot(self.asset.model, prepared.entry, inputs, h)
        if not result.report.accepted:
            raise ValueError("floating dynamics rejected candidate: " + str(result.report.reason))
        candidate = result.state
        self.validate(candidate)
        applied = _project(prepared.linear, prepared.angular, inputs.force, inputs.torque)
        applied[6:] += prepared.effort
        acceleration = (candidate.v - prepared.entry.v) / h
        if not np.isfinite(applied).all() or not np.isfinite(acceleration).all():
            raise ValueError("floating load or acceleration exceeds finite range")
        observation = self._observation(
            candidate, prepared.entry, prepared.effort, contact, wrenches, prepared.external,
            applied, acceleration, prepared.resolved, prepared.requested,
        )
        return candidate, observation

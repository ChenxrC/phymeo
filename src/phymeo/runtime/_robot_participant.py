"""Pure per-robot preparation and candidate integration for a shared HOST scene.

An instance owns immutable asset/drive data and key mappings, not live state.
The scene publishes all RobotState candidates together with the rigid bodies.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..control.bounded_pd import solve_bounded_pd
from ..core.robot import RobotState
from ..dynamics.joint_mimic import MimicRobotAsset
from ..dynamics.robot_dynamics import _solve_mass_matrix, robot_dynamics_terms
from .robot_payload_scene import _readonly, _vector
from .robot_scene import RobotCommand, RobotJointObservation, _asset_snapshot


@dataclass(frozen=True, eq=False)
class RobotPreparation:
    entry: RobotState
    physical: RobotState
    geometry: object
    linear: np.ndarray
    angular: np.ndarray
    full_mass: np.ndarray
    full_bias: np.ndarray
    mass: np.ndarray
    bias: np.ndarray
    effort: np.ndarray
    resolved: RobotCommand
    requested: RobotCommand


class RobotParticipant:
    """One fixed-root ordinary or mimic tree with its own physical owner."""

    def __init__(self, address, registration, occupied):
        self.address = address
        self.spec = _asset_snapshot(registration)
        self.asset = self.spec.asset
        self.mechanism = self.asset if isinstance(self.asset, MimicRobotAsset) else None
        self.tree = self.asset.tree if self.mechanism is not None else self.asset
        self.lower, self.upper = np.asarray(self.asset.lower), np.asarray(self.asset.upper)
        self.speed_limit = np.array(
            [np.inf if value is None else value for value in self.asset.velocity_limit]
        )
        self.initial = RobotState(self.spec.q, self.spec.v)
        self.validate(self.initial)
        body_remap, next_body = {}, 0
        for body in sorted({shape.key.body for shape in self.tree.collision_model.shapes}):
            while next_body in occupied:
                next_body += 1
            body_remap[body] = next_body
            occupied.add(next_body)
        keys, names, semantic, shapes = {}, {}, {}, []
        for collision in self.tree.collisions:
            source = collision.shape
            key = CollisionKey(address[0], body_remap[source.key.body], source.key.shape)
            if source.key in keys:
                raise ValueError("robot asset contains duplicate source collision keys")
            keys[source.key] = key
            shapes.append(replace(source, key=key))
            names[key] = f"{address[1]}:{collision.link_name}/{collision.name}"
            if collision.name_is_authored:
                semantic.setdefault((collision.link_name, collision.name), []).append(key)
        self.key_map = MappingProxyType(keys)
        self.names = MappingProxyType(names)
        self.named_keys = MappingProxyType({key: tuple(value) for key, value in semantic.items()})
        self.shapes = tuple(shapes)

    def validate(self, state):
        if not isinstance(state, RobotState) or state.q.shape != (len(self.asset.joint_names),):
            raise ValueError("robot state must match the independent joint coordinates")
        if not np.isfinite(state.q).all() or not np.isfinite(state.v).all():
            raise ValueError("robot state must be finite")
        if np.any(state.q < self.lower) or np.any(state.q > self.upper):
            raise ValueError("joint position outside the asset interval")
        if np.any(abs(state.v) > self.speed_limit):
            raise ValueError("joint velocity outside the asset interval")

    def full(self, state):
        return state if self.mechanism is None else self.mechanism.full_state(state)

    def terms(self, state):
        physical = self.full(state)
        return (physical, *robot_dynamics_terms(self.tree.model, physical))

    def poses(self, state):
        source = self.asset.collision_motion(state.q)
        if set(source) != set(self.key_map):
            raise ValueError("robot collision poses do not match the compiled model")
        return {key: source[old] for old, key in self.key_map.items()}

    def initial_observation(self):
        full = self.full(self.initial)
        n, full_n = len(self.initial.q), self.tree.model.dof_count
        return RobotJointObservation(
            _readonly(self.initial.q),
            _readonly(self.initial.v),
            _readonly(full.q),
            _readonly(full.v),
            _readonly(self.initial.q),
            _readonly(self.initial.v),
            _readonly(np.zeros(n)),
            _readonly(np.zeros(n)),
            _readonly(np.zeros(full_n)),
            _readonly(np.zeros((full_n, 6))),
            _readonly(np.zeros(6)),
            None,
            None,
        )

    def prepare(self, state, command, dt):
        if not isinstance(command, RobotCommand):
            raise TypeError("robot action must be a RobotCommand")
        n = len(state.q)
        target = _vector(command.target_q, n, "target_q")
        rate = (
            np.zeros(n)
            if command.target_velocity is None
            else _vector(command.target_velocity, n, "target_velocity")
        )
        self.validate(RobotState(target, rate))
        cap = (
            self.spec.effort_limit
            if command.effort_limit is None
            else _vector(command.effort_limit, n, "effort_limit")
        )
        if np.any(cap < 0) or np.any(cap > self.spec.effort_limit):
            raise ValueError("command effort limit exceeds configured rating")
        physical, geo, linear, angular, full_mass, full_bias = self.terms(state)
        if self.mechanism is None:
            mass, bias = full_mass, full_bias
        else:
            projection = self.mechanism.matrix
            mass, bias = projection.T @ full_mass @ projection, projection.T @ full_bias
        ff = bias if command.feedforward is None else _vector(command.feedforward, n, "feedforward")
        effort = solve_bounded_pd(
            mass,
            bias,
            state.q,
            state.v,
            target,
            rate,
            self.spec.kp,
            self.spec.kd,
            cap,
            dt,
            feedforward=ff,
        ).effort
        return RobotPreparation(
            state,
            physical,
            geo,
            linear,
            angular,
            full_mass,
            full_bias,
            mass,
            bias,
            effort,
            RobotCommand(target, rate, cap, ff),
            command,
        )

    def integrate(self, prepared, measured, dt):
        """Return an unpublished state and immutable row; apply each full load once."""
        full_load = measured.joint_effort_full
        load = full_load if self.mechanism is None else self.mechanism.matrix.T @ full_load
        acceleration = _solve_mass_matrix(prepared.mass, prepared.effort + load - prepared.bias)
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = prepared.entry.v + dt * acceleration
            candidate = RobotState(prepared.entry.q + dt * velocity, velocity)
        self.validate(candidate)
        full = self.full(candidate)
        mimic_loads = (
            None
            if self.mechanism is None
            else self.mechanism._receipt(
                prepared.entry,
                prepared.full_mass,
                prepared.full_bias,
                prepared.effort,
                full_load,
                acceleration,
            )
        )
        observation = RobotJointObservation(
            _readonly(candidate.q),
            _readonly(candidate.v),
            _readonly(full.q),
            _readonly(full.v),
            _readonly(prepared.entry.q),
            _readonly(prepared.entry.v),
            _readonly(prepared.effort),
            _readonly(load),
            _readonly(full_load),
            _readonly(measured.link_wrenches_world),
            _readonly(measured.fixed_root_contact_wrench),
            prepared.resolved,
            mimic_loads,
            requested_command=prepared.requested,
        )
        return candidate, observation

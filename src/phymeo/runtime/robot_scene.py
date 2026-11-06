"""A single HOST transaction for one fixed-root URDF and multiple finite rigid bodies.

All contacts use the same entry state. Each robot coordinate and each free body
is integrated once. This first slice is instantaneous compliant contact, without
CCD, robot self-contact, deformable participants, or a device robot solver.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import combinations
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..builders._robot_asset import RobotAsset
from ..builders.urdf import load_urdf
from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion
from ..constraints.robot_contact import RobotContactMaterial
from ..control.bounded_pd import solve_bounded_pd
from ..core.coupled import CoupledInputs
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.robot import RobotState
from ..core.validation import validate_dt
from ..dynamics.joint_mimic import MimicRobotAsset
from ..dynamics.robot_dynamics import _solve_mass_matrix, robot_dynamics_terms
from ..dynamics.rotation import quaternion_matrix
from ..solvers.collision_loads import SurfaceLoads
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ._robot_scene_contacts import RobotSceneContacts, rigid_solid
from .robot_payload_scene import _frozen, _readonly, _vector


@dataclass(frozen=True, eq=False)
class RobotCommand:
    """Independent joint targets; optional feedforward is included inside the motor cap."""

    target_q: np.ndarray
    target_velocity: np.ndarray | None = None
    effort_limit: np.ndarray | None = None
    feedforward: np.ndarray | None = None

    def __post_init__(self):
        for name in ("target_q", "target_velocity", "effort_limit", "feedforward"):
            value = getattr(self, name)
            if value is None and name != "target_q":
                continue
            a = np.asarray(value)
            if a.ndim != 1 or not len(a) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
                raise ValueError(f"{name} must be a nonempty finite real vector")
            object.__setattr__(self, name, _readonly(a))


@dataclass(frozen=True, eq=False)
class RobotRegistration:
    asset: object
    q: np.ndarray
    v: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    pads: tuple


def _asset_snapshot(value, memo=None):
    """Clone validated asset data without recomputing coordinates or inertia.

    Dataclass fields and derived attributes are retained. Every NumPy buffer is
    backed by immutable bytes, including full trees and the mimic projection.
    A second build cannot share writable arrays with its loader or another scene.
    """
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError("asset snapshots require numeric array data")
        result = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    elif is_dataclass(value):
        result = object.__new__(type(value))
        memo[id(value)] = result
        attributes = {f.name: getattr(value, f.name) for f in fields(value)}
        attributes.update(getattr(value, "__dict__", {}))
        for name, item in attributes.items():
            object.__setattr__(result, name, _asset_snapshot(item, memo))
    elif isinstance(value, Mapping):
        result = MappingProxyType(
            {_asset_snapshot(key, memo): _asset_snapshot(item, memo) for key, item in value.items()}
        )
    elif isinstance(value, (tuple, list)):
        result = tuple(_asset_snapshot(item, memo) for item in value)
    elif isinstance(value, (set, frozenset)):
        result = frozenset(_asset_snapshot(item, memo) for item in value)
    else:
        return value
    memo[id(value)] = result
    return result


def make_robot_registration(
    path,
    *,
    robot_state,
    kp,
    kd,
    effort_limit=None,
    pads=(),
    loader=None,
    gravity=(0.0, 0.0, -9.81),
):
    """Load the actual local asset once; take independent copies of initial coordinates."""
    asset = (load_urdf if loader is None else loader)(path, gravity=tuple(gravity))
    if not isinstance(asset, (RobotAsset, MimicRobotAsset)):
        raise TypeError("loader must return RobotAsset or MimicRobotAsset with owned model data")
    n = len(asset.joint_names)
    if not isinstance(robot_state, RobotState) or robot_state.q.shape != (n,):
        raise ValueError("robot_state must match the loaded independent joint order")
    kp, kd = _vector(kp, n, "kp"), _vector(kd, n, "kd")
    if np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("controller gains must be nonnegative")
    hard = np.array([np.inf if v is None else v for v in asset.effort_limit])
    cap = _vector(hard if effort_limit is None else effort_limit, n, "effort_limit")
    if np.any(cap < 0) or np.any(cap > hard):
        raise ValueError("effort limit is negative or exceeds the URDF rating")
    tree = asset.tree if isinstance(asset, MimicRobotAsset) else asset
    if not np.array_equal(tree.model.gravity, np.asarray(gravity, dtype=float)):
        raise ValueError("robot loader must preserve the scene gravity")
    return RobotRegistration(
        _asset_snapshot(asset),
        _readonly(robot_state.q),
        _readonly(robot_state.v),
        kp,
        kd,
        cap,
        tuple(pads),
    )


@dataclass(frozen=True, eq=False)
class RigidSceneObservation:
    """COM positions, body-to-world XYZW poses, and world twists, all immutable."""

    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray

    @classmethod
    def from_state(cls, state):
        return cls(
            *(
                _readonly(getattr(state, field))
                for field in ("position", "orientation", "velocity", "angular_velocity")
            )
        )


@dataclass(frozen=True, eq=False)
class RobotJointObservation:
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
    command: RobotCommand | None
    mimic_loads: object | None
    # Preserve optional request fields separately from the resolved drive command.
    requested_command: RobotCommand | None = None


@dataclass(frozen=True, eq=False)
class RobotSceneObservation:
    """Accepted end pose; force, command and entry channels belong to load_time_s.

    loads.inputs contains only the separately registered rigid body rows;
    the robot receives robots[address].contact_effort. PairLoad retains both
    physical sides and can be read by measure_contact_wrench(). Its mutable
    input arrays and external are owned by each returned observation.
    """

    step_number: int
    time_s: float
    load_time_s: float | None
    robots: Mapping
    rigid: RigidSceneObservation
    entry_rigid: RigidSceneObservation
    external: CoupledInputs
    loads: SurfaceLoads
    contacts: tuple


@dataclass(frozen=True, eq=False)
class RobotSceneGeometry:
    time_s: float
    model: CollisionModel
    poses: Mapping
    names: Mapping
    body_keys: Mapping


@dataclass(frozen=True, eq=False)
class RobotSceneState:
    """Owned writable copies; changing this snapshot cannot alter the live scene."""

    time_s: float
    robots: Mapping
    rigid: RigidBodyState


def _owned(reading):
    return replace(
        reading,
        external=CoupledInputs(reading.external.particles, reading.external.rigid),
        loads=replace(
            reading.loads,
            inputs=CoupledInputs(reading.loads.inputs.particles, reading.loads.inputs.rigid),
        ),
    )


def build_robot_scene(bodies, pairs, *, gravity, backend="host", cache_root=None):
    """Compile the registration graph without changing the existing nonrobot compiler."""
    if backend != "host":
        raise ValueError("robot scenes currently require HOST; no robot CUDA/interpret path")
    robots = [(addr, item) for addr, item in bodies.items() if isinstance(item, RobotRegistration)]
    if len(robots) > 1:
        from .multi_robot_scene import build_multi_robot_scene

        return build_multi_robot_scene(
            bodies, pairs, gravity=gravity, backend=backend, cache_root=cache_root
        )
    if len(robots) != 1:
        raise ValueError("robot scenes require at least one fixed-root robot")
    if len({addr[0] for addr in bodies}) != 1:
        raise ValueError("robot scenes currently require one world")
    # Local import keeps the public builder and this optional route acyclic.
    from .contact_scene import _Body

    robot_addr, registration = robots[0]
    rigid = {addr: item for addr, item in sorted(bodies.items()) if addr != robot_addr}
    if any(not isinstance(item, _Body) for item in rigid.values()):
        raise ValueError("robot scenes do not support deformable/sheet participants")
    for addr, item in rigid.items():
        rigid_solid(item, str(addr))
        if item.inverse_mass == 0 and (np.any(item.velocity) or np.any(item.omega)):
            raise ValueError(f"{addr}: fixed supports must have zero prescribed twist")
    expected = set(combinations(sorted(bodies), 2))
    if set(pairs) != expected:
        raise ValueError(
            "every robot/rigid pair needs a material or explicit exclusion; no self pairs"
        )
    if any(law is not None and not isinstance(law, RobotContactMaterial) for law in pairs.values()):
        raise TypeError("robot scenes require RobotContactMaterial; NormalLayer is not converted")
    return RobotScene(robot_addr, registration, rigid, dict(pairs), gravity)


class RobotScene:
    """One command advances the complete supported scene or leaves all published data unchanged."""

    def __init__(self, robot_addr, registration, rigid_items, pairs, gravity):
        self._addr, self._spec = robot_addr, registration
        self._asset = _asset_snapshot(registration.asset)
        self._mechanism = self._asset if isinstance(self._asset, MimicRobotAsset) else None
        self._tree = self._asset.tree if self._mechanism is not None else self._asset
        self._arm = RobotState(registration.q, registration.v)
        self._lower, self._upper = np.asarray(self._asset.lower), np.asarray(self._asset.upper)
        self._velocity_limit = np.array(
            [np.inf if v is None else v for v in self._asset.velocity_limit]
        )
        self._validate_robot(self._arm)
        self._items = dict(rigid_items)
        self.rigid_rows = MappingProxyType({addr: i for i, addr in enumerate(rigid_items)})
        self.assets = MappingProxyType({robot_addr: self._asset})
        self._robot_keys, self._named_keys = {}, {}
        shapes = [item.shape for item in rigid_items.values()]
        names = {s.key: f"body:{s.key.body}" for s in shapes}
        body_keys = {addr: (item.shape.key,) for addr, item in rigid_items.items()}
        owners = {item.shape.key: addr for addr, item in rigid_items.items()}
        occupied = {addr[1] for addr in (*rigid_items, robot_addr)}
        # IDs of the loaded links are model-local, not scene body IDs.
        local_bodies = sorted({s.key.body for s in self._tree.collision_model.shapes})
        remap, next_body = {}, 0
        for local in local_bodies:
            while next_body in occupied:
                next_body += 1
            remap[local] = next_body
            occupied.add(next_body)
        for col in self._tree.collisions:
            source = col.shape
            key = CollisionKey(robot_addr[0], remap[source.key.body], source.key.shape)
            self._robot_keys[source.key] = key
            shapes.append(replace(source, key=key))
            names[key] = f"{robot_addr[1]}:{col.link_name}/{col.name}"
            owners[key] = robot_addr
            if col.name_is_authored:
                self._named_keys.setdefault((col.link_name, col.name), []).append(key)
        body_keys[robot_addr] = tuple(self._robot_keys.values())
        if len({s.key for s in shapes}) != len(shapes):
            raise ValueError("compiled collision keys are not unique")
        exclusions = []
        for a, b in combinations(shapes, 2):
            oa, ob = owners[a.key], owners[b.key]
            if oa == ob or pairs[tuple(sorted((oa, ob)))] is None:
                exclusions.append((a.key, b.key))
        self._model = CollisionModel(tuple(shapes), tuple(exclusions))
        self._names, self.body_keys = MappingProxyType(names), MappingProxyType(body_keys)
        self._source_keys = MappingProxyType(dict(self._robot_keys))

        def matrix(field, width):
            return np.asarray(
                [getattr(i, field) for i in rigid_items.values()], dtype=float
            ).reshape(-1, width)

        self._body = RigidBodyState(
            matrix("position", 3),
            matrix("orientation", 4),
            matrix("velocity", 3),
            matrix("omega", 3),
        )
        self._rigid_model = RigidBodyModel(
            np.asarray([i.inverse_mass for i in rigid_items.values()], dtype=float),
            matrix("inertia", 3),
            np.asarray(gravity, dtype=float),
        )
        self._integrator = RigidSemiImplicitSolver()
        self._contact = RobotSceneContacts(
            self._tree,
            robot_addr,
            self._items,
            self.rigid_rows,
            pairs,
            self._robot_keys,
            registration.pads,
            self._model,
        )
        from .contact_events import RobotContactEventAdapter, SceneContactEvents

        self._event_adapter = RobotContactEventAdapter(
            robot_addr, self._tree, self._items, self.rigid_rows, pairs, self._robot_keys
        )
        self._contact_events = SceneContactEvents()
        self._lock, self._closed = RLock(), False
        self._time_s, self._step_number = 0.0, 0
        self._geometry = self._geometry_for(self._arm, self._body, 0.0)
        n, full_n = len(self._arm.q), self._tree.model.dof_count
        full = self._full(self._arm)
        joint = RobotJointObservation(
            _readonly(self._arm.q),
            _readonly(self._arm.v),
            _readonly(full.q),
            _readonly(full.v),
            _readonly(self._arm.q),
            _readonly(self._arm.v),
            _readonly(np.zeros(n)),
            _readonly(np.zeros(n)),
            _readonly(np.zeros(full_n)),
            _readonly(np.zeros((full_n, 6))),
            _readonly(np.zeros(6)),
            None,
            None,
        )
        empty = self.inputs()
        snap = RigidSceneObservation.from_state(self._body)
        self._observation = RobotSceneObservation(
            0,
            0.0,
            None,
            MappingProxyType({robot_addr: joint}),
            snap,
            snap,
            empty,
            SurfaceLoads(self.inputs(), ()),
            (),
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("robot scene is closed")

    def _validate_robot(self, state):
        if np.any(state.q < self._lower) or np.any(state.q > self._upper):
            raise ValueError("joint position outside the asset interval")
        if np.any(abs(state.v) > self._velocity_limit):
            raise ValueError("joint velocity outside the asset interval")

    def _full(self, state):
        return state if self._mechanism is None else self._mechanism.full_state(state)

    def _geometry_for(self, arm, body, time_s):
        source = self._asset.collision_motion(arm.q)
        if set(source) != set(self._robot_keys):
            raise ValueError("robot collision poses do not match the compiled model")
        poses = {key: source[old] for old, key in self._robot_keys.items()}
        rotations = quaternion_matrix(body.orientation)
        for addr, item in self._items.items():
            row = self.rigid_rows[addr]
            world = body.position[row] + np.asarray(item.vertices) @ rotations[row].T
            poses[item.shape.key] = CollisionMotion(world)
        if set(poses) != {s.key for s in self._model.shapes}:
            raise ValueError("scene collision poses do not match the compiled model")
        return RobotSceneGeometry(
            time_s, self._model, MappingProxyType(poses), self._names, self.body_keys
        )

    def shape_key(self, robot, link_name, collision_name):
        """Resolve an authored semantic name after this scene's collision-key remapping."""
        if robot != self._addr[1]:
            raise ValueError("unknown robot registration")
        found = self._named_keys.get((link_name, collision_name), ())
        if len(found) != 1:
            raise ValueError("authored collision name is missing or ambiguous")
        return found[0]

    @property
    def state(self):
        with self._lock:
            return RobotSceneState(
                self._time_s,
                MappingProxyType({self._addr: RobotState(self._arm.q, self._arm.v)}),
                self._body.copy(),
            )

    @property
    def last_frame(self):
        with self._lock:
            return None if self._step_number == 0 else _owned(self._observation)

    @property
    def contact_events(self):
        """Immutable events sampled at entry and committed with the accepted step."""
        with self._lock:
            return self._contact_events

    def inputs(self):
        """Allocate independently writable COM forces/torques in rigid_rows order."""
        with self._lock:
            self._require_open()
            return CoupledInputs.zeros(0, self._body.body_count)

    def observe(self):
        with self._lock:
            self._require_open()
            return _owned(self._observation)

    def geometry(self):
        with self._lock:
            self._require_open()
            return self._geometry

    def observe_scene(self):
        with self._lock:
            self._require_open()
            return _owned(self._observation), self._geometry

    def sample(self):
        """Current explicit contact sample; does not advance time or replace the accepted frame."""
        with self._lock:
            self._require_open()
            physical = self._full(self._arm)
            geo, linear, angular, _, _ = robot_dynamics_terms(self._tree.model, physical)
            return self._contact.evaluate(
                physical, geo, linear, angular, self._body, self._geometry.poses
            ).loads

    def enable_coupled_tangent(self, registry, **options):
        """Select explicit material-station spring/Coulomb friction before the first step."""
        with self._lock:
            self._require_open()
            from ._scene_coupled_tangent import configure

            configure(self, registry, **options)

    @property
    def tangent_history(self):
        with self._lock:
            return getattr(self, "_tangent_state", None)

    @property
    def tangent_receipt(self):
        with self._lock:
            value = getattr(self, "_tangent_receipt", None)
            return None if value is None else replace(value)

    def step(self, dt, *, actions, external=None):
        with self._lock:
            self._require_open()
            h = float(validate_dt(dt, np.float64))
            end_time = self._time_s + h
            if not np.isfinite(end_time) or end_time <= self._time_s:
                raise ValueError("time must advance by a representable interval")
            if not isinstance(actions, Mapping) or set(actions) != {self._addr[1]}:
                raise ValueError("actions must select exactly the registered robot")
            command = actions[self._addr[1]]
            if not isinstance(command, RobotCommand):
                raise TypeError("robot action must be a RobotCommand")
            n = len(self._arm.q)
            target = _vector(command.target_q, n, "target_q")
            rate = (
                np.zeros(n)
                if command.target_velocity is None
                else _vector(command.target_velocity, n, "target_velocity")
            )
            self._validate_robot(RobotState(target, rate))
            cap = (
                self._spec.effort_limit
                if command.effort_limit is None
                else _vector(command.effort_limit, n, "effort_limit")
            )
            if np.any(cap < 0) or np.any(cap > self._spec.effort_limit):
                raise ValueError("command effort limit exceeds configured rating")
            if external is None:
                external = self.inputs()
            if not isinstance(external, CoupledInputs):
                raise TypeError("external must be CoupledInputs from scene.inputs()")
            external = CoupledInputs(external.particles, external.rigid)
            if (
                external.particles.force.shape != (0, 3)
                or external.rigid.force.shape != self._body.position.shape
                or external.rigid.force.dtype != np.float64
            ):
                raise ValueError("external forces must match the FP64 registered rigid rows")
            arm, body = self._arm, self._body
            physical = self._full(arm)
            geo, linear, angular, mass, bias = robot_dynamics_terms(self._tree.model, physical)
            P = None if self._mechanism is None else self._mechanism.matrix
            M, b = (mass, bias) if P is None else (P.T @ mass @ P, P.T @ bias)
            ff = (
                b if command.feedforward is None else _vector(command.feedforward, n, "feedforward")
            )
            effort = solve_bounded_pd(
                M,
                b,
                arm.q,
                arm.v,
                target,
                rate,
                self._spec.kp,
                self._spec.kd,
                cap,
                h,
                feedforward=ff,
            ).effort
            measured = self._contact.evaluate(
                physical, geo, linear, angular, body, self._geometry.poses
            )
            tangent = None
            if getattr(self, "_tangent", None) is not None:
                from ._scene_coupled_tangent import robot_prepare

                measured, tangent = robot_prepare(
                    self, measured, geo, linear, M, b, effort, external, h
                )
            full_load = measured.joint_effort_full
            load = full_load if P is None else P.T @ full_load
            acceleration = _solve_mass_matrix(M, effort + load - b)
            v = arm.v + h * acceleration
            candidate_arm = RobotState(arm.q + h * v, v)
            self._validate_robot(candidate_arm)
            forces = RigidBodyInputs(
                external.rigid.force + measured.loads.inputs.rigid.force,
                external.rigid.torque + measured.loads.inputs.rigid.torque,
            )
            result = self._integrator.step(self._rigid_model, body, forces, h)
            if not result.report.accepted:
                raise ValueError(result.report.reason)
            candidate_body = result.state
            geometry = self._geometry_for(candidate_arm, candidate_body, end_time)
            full = self._full(candidate_arm)
            staged_tangent = None
            if tangent is not None:
                from ..dynamics.robot_kinematics import forward_kinematics

                end_geo = forward_kinematics(self._tree.model, candidate_arm.q)
                staged_tangent = tangent.finish(candidate_body, end_geo)
            mimic_loads = (
                None
                if P is None
                else self._mechanism._receipt(arm, mass, bias, effort, full_load, acceleration)
            )
            joint = RobotJointObservation(
                _readonly(candidate_arm.q),
                _readonly(candidate_arm.v),
                _readonly(full.q),
                _readonly(full.v),
                _readonly(arm.q),
                _readonly(arm.v),
                _readonly(effort),
                _readonly(load),
                _readonly(full_load),
                _readonly(measured.link_wrenches_world),
                _readonly(measured.fixed_root_contact_wrench),
                RobotCommand(target, rate, cap, ff),
                mimic_loads,
                requested_command=command,
            )
            reading = RobotSceneObservation(
                self._step_number + 1,
                end_time,
                self._time_s,
                MappingProxyType({self._addr: joint}),
                RigidSceneObservation.from_state(candidate_body),
                RigidSceneObservation.from_state(body),
                external,
                measured.loads,
                tuple(_frozen(row) for row in measured.details),
            )
            # Finish every allocation and public copy before the one state publication.
            public = _owned(reading)
            staged_events = self._event_adapter.stage(
                self._contact_events,
                measured.loads,
                geo=geo,
                rigid=body,
                sample_time_s=self._time_s,
                commit_time_s=end_time,
            )
            tangent_state, tangent_receipt = (
                staged_tangent if staged_tangent is not None else (None, None)
            )
            publication = (
                candidate_arm,
                candidate_body,
                reading,
                geometry,
                staged_events,
                reading.step_number,
                end_time,
                tangent_state,
                tangent_receipt,
            )
            (
                self._arm,
                self._body,
                self._observation,
                self._geometry,
                self._contact_events,
                self._step_number,
                self._time_s,
                self._tangent_state,
                self._tangent_receipt,
            ) = publication
            return public

    def close(self):
        with self._lock:
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

"""Robot trees with fixed or physical floating roots in one HOST transaction.

Contacts all sample the same entry state. Robot/rigid pairs are evaluated once
per robot; rigid/rigid pairs once for the entire scene. No live state is changed
until all candidates, owned observations and contact events are ready.
"""

from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionModel, CollisionMotion
from ..constraints.robot_contact import RobotContactMaterial
from ..core.coupled import CoupledInputs
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.validation import validate_dt
from ..dynamics.rotation import quaternion_matrix
from ..solvers.collision_loads import SurfaceLoads
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ._floating_robot_participant import FloatingRobotParticipant, FloatingRobotRegistration
from ._robot_participant import RobotParticipant
from ._robot_robot_contacts import RobotRobotContacts
from ._robot_scene_contacts import RobotSceneContacts, rigid_solid
from .contact_events import RobotContactEventAdapter, SceneContactEvents
from .robot_payload_scene import _frozen
from .robot_scene import (
    RigidSceneObservation,
    RobotRegistration,
    RobotSceneGeometry,
    RobotSceneObservation,
    RobotSceneState,
    _owned,
)


def build_multi_robot_scene(bodies, pairs, *, gravity, backend="host", cache_root=None):
    """Compile only the declared multi-robot scope; unsupported pairs never disappear."""
    if backend != "host":
        raise ValueError("multi-robot scenes require HOST; no robot CUDA/interpret path")
    robots = {
        addr: item
        for addr, item in sorted(bodies.items())
        if isinstance(item, (RobotRegistration, FloatingRobotRegistration))
    }
    floating = any(isinstance(item, FloatingRobotRegistration) for item in robots.values())
    if not robots or (len(robots) < 2 and not floating):
        raise ValueError("shared robot scenes require multiple fixed trees or a floating tree")
    if len({addr[0] for addr in bodies}) != 1:
        raise ValueError("multi-robot scenes currently require one world")
    from .contact_scene import _Body

    rigid = {addr: item for addr, item in sorted(bodies.items()) if addr not in robots}
    if any(not isinstance(item, _Body) for item in rigid.values()):
        raise ValueError("multi-robot scenes do not support deformable/sheet participants")
    for addr, item in rigid.items():
        rigid_solid(item, str(addr))
        if item.inverse_mass == 0 and (np.any(item.velocity) or np.any(item.omega)):
            raise ValueError(f"{addr}: fixed supports must have zero prescribed twist")
    if set(pairs) != set(combinations(sorted(bodies), 2)):
        raise ValueError(
            "every robot/rigid pair needs a material or explicit exclusion; no self pairs"
        )
    if any(law is not None and not isinstance(law, RobotContactMaterial) for law in pairs.values()):
        raise TypeError(
            "multi-robot scenes require RobotContactMaterial; NormalLayer is not converted"
        )
    return MultiRobotScene(robots, rigid, dict(pairs), gravity)


class MultiRobotScene:
    """HOST fixed and floating trees sharing contact, rigid bodies and one accepted clock.

    Commands use independent joint coordinates, including existing mimic graphs.
    Direct contacts support sphere, finite capsule and convex-hull patches on
    another tree. Self-contact, CCD, floating mimic trees, deformable bodies and the
    opt-in coupled tangential-history mode remain unsupported.
    """

    def __init__(self, registrations, rigid_items, pairs, gravity):
        self._lock, self._closed = RLock(), False
        self._items = dict(rigid_items)
        self.rigid_rows = MappingProxyType({addr: row for row, addr in enumerate(rigid_items)})
        occupied = {addr[1] for addr in (*registrations, *rigid_items)}
        self._participants = {
            addr: (
                FloatingRobotParticipant
                if isinstance(registration, FloatingRobotRegistration)
                else RobotParticipant
            )(addr, registration, occupied)
            for addr, registration in registrations.items()
        }
        self.assets = MappingProxyType({addr: p.asset for addr, p in self._participants.items()})
        self._arms = {addr: p.copy_state(p.initial) for addr, p in self._participants.items()}
        shapes = [item.shape for item in rigid_items.values()]
        names = {shape.key: f"body:{shape.key.body}" for shape in shapes}
        owners = {item.shape.key: addr for addr, item in rigid_items.items()}
        body_keys = {addr: (item.shape.key,) for addr, item in rigid_items.items()}
        for addr, participant in self._participants.items():
            shapes.extend(participant.shapes)
            names.update(participant.names)
            owners.update({shape.key: addr for shape in participant.shapes})
            body_keys[addr] = tuple(shape.key for shape in participant.shapes)
        if len({shape.key for shape in shapes}) != len(shapes):
            raise ValueError("compiled collision keys are not unique")
        exclusions = []
        for first, second in combinations(shapes, 2):
            a, b = owners[first.key], owners[second.key]
            if a == b or pairs[tuple(sorted((a, b)))] is None:
                exclusions.append((first.key, second.key))
        self._model = CollisionModel(tuple(shapes), tuple(exclusions))
        self._names, self.body_keys = MappingProxyType(names), MappingProxyType(body_keys)
        self._robot_contacts = RobotRobotContacts(self._participants, pairs, self._model)

        def matrix(field, width):
            return np.asarray(
                [getattr(item, field) for item in rigid_items.values()], dtype=float
            ).reshape(-1, width)

        self._body = RigidBodyState(
            matrix("position", 3),
            matrix("orientation", 4),
            matrix("velocity", 3),
            matrix("omega", 3),
        )
        self._rigid_model = RigidBodyModel(
            np.asarray([item.inverse_mass for item in rigid_items.values()], dtype=float),
            matrix("inertia", 3),
            np.asarray(gravity, dtype=float),
        )
        self._integrator = RigidSemiImplicitSolver()
        self._contacts = {
            addr: RobotSceneContacts(
                p.contact_view,
                addr,
                self._items,
                self.rigid_rows,
                pairs,
                p.key_map,
                p.spec.pads,
                self._model,
            )
            for addr, p in self._participants.items()
        }
        # This sampler is used once, only for its rigid-pair subroutine. All
        # per-robot calls below explicitly disable that common contribution.
        self._rigid_contacts = next(iter(self._contacts.values()))
        self._event_adapter = RobotContactEventAdapter.for_robots(
            {addr: (p.contact_view, p.key_map) for addr, p in self._participants.items()},
            self._items,
            self.rigid_rows,
            pairs,
        )
        self._contact_events = SceneContactEvents()
        self._time_s, self._step_number = 0.0, 0
        self._geometry = self._geometry_for(self._arms, self._body, 0.0)
        snap = RigidSceneObservation.from_state(self._body)
        self._observation = RobotSceneObservation(
            0,
            0.0,
            None,
            MappingProxyType(
                {addr: p.initial_observation() for addr, p in self._participants.items()}
            ),
            snap,
            snap,
            self.inputs(),
            SurfaceLoads(self.inputs(), ()),
            (),
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("multi-robot scene is closed")

    def _geometry_for(self, arms, body, time_s):
        poses = {}
        for addr, participant in self._participants.items():
            poses.update(participant.poses(arms[addr]))
        rotations = quaternion_matrix(body.orientation)
        for addr, item in self._items.items():
            row = self.rigid_rows[addr]
            points = body.position[row] + np.asarray(item.vertices) @ rotations[row].T
            poses[item.shape.key] = CollisionMotion(points)
        if set(poses) != {shape.key for shape in self._model.shapes}:
            raise ValueError("scene collision poses do not match the compiled model")
        return RobotSceneGeometry(
            time_s, self._model, MappingProxyType(poses), self._names, self.body_keys
        )

    def _sample(self, terms):
        common, details = self._rigid_contacts.evaluate_rigid(self._body, self._geometry.poses)
        force, torque = common.inputs.rigid.force.copy(), common.inputs.rigid.torque.copy()
        contacts, rows, measured = list(common.contacts), list(details), {}
        for addr, (physical, geo, linear, angular) in terms.items():
            item = self._contacts[addr].evaluate(
                physical,
                geo,
                linear,
                angular,
                self._body,
                self._geometry.poses,
                include_rigid=False,
            )
            measured[addr] = item
            force += item.loads.inputs.rigid.force
            torque += item.loads.inputs.rigid.torque
            contacts.extend(item.loads.contacts)
            rows.extend(item.details)
        direct = self._robot_contacts.evaluate(terms, self._geometry.poses)
        for addr, contribution in direct.by_robot.items():
            prior = measured[addr]
            measured[addr] = replace(
                prior,
                joint_effort_full=prior.joint_effort_full + contribution.joint_effort_full,
                link_wrenches_world=prior.link_wrenches_world + contribution.link_wrenches_world,
                fixed_root_contact_wrench=(
                    prior.fixed_root_contact_wrench + contribution.fixed_root_contact_wrench
                ),
            )
        contacts.extend(direct.contacts)
        rows.extend(direct.details)
        inputs = self.inputs()
        inputs.rigid.force[:] = force
        inputs.rigid.torque[:] = torque
        return measured, SurfaceLoads(inputs, tuple(contacts)), tuple(rows)

    def shape_key(self, robot, link_name, collision_name):
        with self._lock:
            self._require_open()
            found = [p for addr, p in self._participants.items() if addr[1] == robot]
            if len(found) != 1:
                raise ValueError("unknown robot registration")
            keys = found[0].named_keys.get((link_name, collision_name), ())
            if len(keys) != 1:
                raise ValueError("authored collision name is missing or ambiguous")
            return keys[0]

    @property
    def state(self):
        with self._lock:
            self._require_open()
            return RobotSceneState(
                self._time_s,
                MappingProxyType(
                    {
                        addr: self._participants[addr].copy_state(state)
                        for addr, state in self._arms.items()
                    }
                ),
                self._body.copy(),
            )

    @property
    def last_frame(self):
        with self._lock:
            self._require_open()
            return None if self._step_number == 0 else _owned(self._observation)

    @property
    def contact_events(self):
        with self._lock:
            self._require_open()
            return self._contact_events

    def inputs(self):
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
        with self._lock:
            self._require_open()
            terms = {addr: p.terms(self._arms[addr])[:4] for addr, p in self._participants.items()}
            return self._sample(terms)[1]

    def enable_coupled_tangent(self, registry, **options):
        with self._lock:
            self._require_open()
            raise ValueError(
                "multi-robot tangent history requires a separate coupled-motion adapter"
            )

    @property
    def tangent_history(self):
        return None

    @property
    def tangent_receipt(self):
        return None

    def step(self, dt, *, actions, external=None):
        with self._lock:
            self._require_open()
            h = float(validate_dt(dt, np.float64))
            end_time = self._time_s + h
            if not np.isfinite(end_time) or end_time <= self._time_s:
                raise ValueError("time must advance by a representable interval")
            if not isinstance(actions, Mapping) or set(actions) != {
                addr[1] for addr in self._participants
            }:
                raise ValueError("actions must select exactly all registered robots")
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
            prepared = {
                addr: p.prepare(self._arms[addr], actions[addr[1]], h)
                for addr, p in self._participants.items()
            }
            terms = {
                addr: (p.physical, p.geometry, p.linear, p.angular) for addr, p in prepared.items()
            }
            measured, loads, details = self._sample(terms)
            candidates, joints = {}, {}
            for addr, participant in self._participants.items():
                candidates[addr], joints[addr] = participant.integrate(
                    prepared[addr], measured[addr], h
                )
            forces = RigidBodyInputs(
                external.rigid.force + loads.inputs.rigid.force,
                external.rigid.torque + loads.inputs.rigid.torque,
            )
            rigid_step = self._integrator.step(self._rigid_model, self._body, forces, h)
            if not rigid_step.report.accepted:
                raise ValueError(rigid_step.report.reason)
            candidate_body = rigid_step.state
            geometry = self._geometry_for(candidates, candidate_body, end_time)
            reading = RobotSceneObservation(
                self._step_number + 1,
                end_time,
                self._time_s,
                MappingProxyType(joints),
                RigidSceneObservation.from_state(candidate_body),
                RigidSceneObservation.from_state(self._body),
                external,
                loads,
                tuple(_frozen(row) for row in details),
            )
            public = _owned(reading)
            events = self._event_adapter.stage(
                self._contact_events,
                loads,
                geo={addr: p.geometry for addr, p in prepared.items()},
                rigid=self._body,
                sample_time_s=self._time_s,
                commit_time_s=end_time,
            )
            publication = (
                candidates,
                candidate_body,
                reading,
                geometry,
                events,
                reading.step_number,
                end_time,
            )
            (
                self._arms,
                self._body,
                self._observation,
                self._geometry,
                self._contact_events,
                self._step_number,
                self._time_s,
            ) = publication
            return public

    def close(self):
        with self._lock:
            self._closed = True

    def __enter__(self):
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, *_):
        self.close()

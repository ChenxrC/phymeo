"""Interactive HOST dynamics of a fixed-base robot, one uniform free body and a finite table."""

from dataclasses import dataclass
from numbers import Integral
from threading import RLock
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..builders.urdf import load_urdf
from ..constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionShape,
)
from ..constraints.robot_contact import RobotContactMaterial
from ..control.bounded_pd import solve_bounded_pd
from ..core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from ..core.robot import RobotState
from ..core.validation import finite_real, validate_dt
from ..dynamics.robot_dynamics import _solve_mass_matrix, robot_dynamics_terms
from ..dynamics.robot_kinematics import _point_jacobians_at
from ..dynamics.rotation import quaternion_matrix
from ..solvers.fixed_table_contact import FixedTableContacts
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ..solvers.sampled_payload import SampledPayload, box_corners
from ..solvers.sphere_link_contact import LinkSolid, SphereLinkContacts
from ..solvers.table_box_patch import TableLinkSolids


def _readonly(values):
    a = np.asarray(values, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _vector(values, n, name):
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {n} finite real values")
    return _readonly(a)


def _frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class PadFace:
    """Material samples on one moving cuboid face, in its authored collision axes.

    axis is 0/1/2 and side is -1/+1. Each (u,v) offset is in metres from the
    face centre, along the two remaining axes in ascending order. Each sample
    has the configured per-sample stiffness and damping.
    """

    shape_key: CollisionKey
    axis: int
    side: int
    offsets_m: tuple

    def __post_init__(self):
        if not isinstance(self.shape_key, CollisionKey):
            raise TypeError("pad shape_key must be a CollisionKey")
        for name, choices in (("axis", (0, 1, 2)), ("side", (-1, 1))):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v not in choices:
                raise ValueError(f"invalid pad {name}")
        uv = np.asarray(self.offsets_m)
        if (
            uv.ndim != 2
            or uv.shape[1] != 2
            or not len(uv)
            or uv.dtype.kind not in "fiu"
            or not np.isfinite(uv).all()
        ):
            raise ValueError("pad offsets must be nonempty finite (u,v) samples")
        pairs = tuple(map(tuple, uv.astype(float)))
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate pad material sample")
        object.__setattr__(self, "offsets_m", pairs)


def _pad_points(asset, pads):
    cols = {c.shape.key: c for c in asset.collisions}
    links, points, sample_keys, excluded = [], [], [], set()
    for pad in pads:
        if not isinstance(pad, PadFace) or pad.shape_key not in cols:
            raise ValueError("pad must select a registered collision shape")
        if pad.shape_key in excluded:
            raise ValueError("duplicate pad collision shape")
        excluded.add(pad.shape_key)
        col = cols[pad.shape_key]
        if col.link_index < 0 or col.shape.kind != "hull":
            raise ValueError("pad face requires a moving cuboid collision shape")
        box = LinkSolid(col.name, col.link_name, col.link_index, "box", col.local_vertices)
        axes = [i for i in range(3) if i != pad.axis]
        offsets = np.asarray(pad.offsets_m)
        if np.any(abs(offsets) > box._half[axes] + 8 * np.finfo(float).eps * box._half[axes]):
            raise ValueError(f"{col.name}: pad sample outside actual cuboid face")
        face = box._center + pad.side * box._half[pad.axis] * box._axes[pad.axis]
        local = face + offsets @ box._axes[axes]
        convert = getattr(asset, "authored_to_model_points", None)
        if convert is not None:
            local = convert(col.link_name, local)
        links.extend([col.link_index] * len(local))
        sample_keys.extend([col.shape.key] * len(local))
        points.extend(local)
    return (
        np.asarray(links, dtype=np.intp),
        np.asarray(points).reshape(-1, 3),
        tuple(sample_keys),
        excluded,
    )


@dataclass(frozen=True, eq=False)
class SceneGeometrySnapshot:
    """One accepted world pose for each registered collision key, directly consumable by sensors."""

    time_s: float
    model: CollisionModel
    poses: Mapping[CollisionKey, CollisionMotion]
    names: Mapping[CollisionKey, str]
    payload_key: CollisionKey
    table_key: CollisionKey


@dataclass(frozen=True, eq=False)
class RobotPayloadObservation:
    """Accepted end state; applied sensor and drive channels belong to load_time_s.

    Rigid velocities and wrenches are in world axes, orientation is body-to-world XYZW.
    Wrenches use N and N m. payload_table_wrench is the table's
    load on the payload about its COM. fixed_table_wrench combines the opposite
    payload reaction and all robot link reactions about the table centre.
    contact_effort includes every robot contact once; drive_effort is separate.
    """

    step_number: int
    time_s: float
    load_time_s: float | None
    q: np.ndarray
    v: np.ndarray
    payload_position_m: np.ndarray
    payload_xyzw: np.ndarray
    payload_velocity_m_s: np.ndarray
    payload_omega_rad_s: np.ndarray
    target_q: np.ndarray
    target_velocity: np.ndarray
    effort_limit: np.ndarray
    drive_effort: np.ndarray
    contact_effort: np.ndarray
    pad_shape_keys: tuple
    pad_points_m: np.ndarray
    pad_forces_on_link_N: np.ndarray
    pad_normal_loads_N: np.ndarray
    payload_table_wrench: np.ndarray
    fixed_table_wrench: np.ndarray
    structural_contacts: tuple
    link_table_contacts: tuple


class RobotPayloadScene:
    """One command advances one physical step. An exception preserves the previous observation.

    The arm can contain sphere, capsule and cuboid colliders. A declared pad replaces
    its solid only for robot/payload contact. Other link material and the independent
    robot/table channel remain active. Box payloads require their corners to stay
    above the support underside and within its finite top footprint. Contacts are
    sampled at the accepted input pose. This HOST workflow has no self-contact or CCD.
    """

    def __init__(
        self,
        asset,
        *,
        robot_state,
        payload_state,
        payload_geometry,
        mass_kg,
        table_center_m,
        table_half_extents_m,
        dt_s,
        kp,
        kd,
        effort_limit=None,
        pads=(),
        table_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        pad_material=RobotContactMaterial(1200.0, 1.2, 0.65, 0.04),
        payload_table_material=RobotContactMaterial(2200.0, 3.0, 0.45, 0.04),
        structure_material=RobotContactMaterial(6000.0, 16.0, 0.35, 0.04),
    ):
        self._asset = asset
        self._dt = float(validate_dt(dt_s, np.float64))
        mass_kg = finite_real(mass_kg, "mass_kg")
        if mass_kg <= 0:
            raise ValueError("payload mass_kg must be positive")
        if not all(
            isinstance(law, RobotContactMaterial)
            for law in (pad_material, payload_table_material, structure_material)
        ):
            raise TypeError("scene contacts require RobotContactMaterial")
        self._pad_law, self._table_law, self._structure_law = (
            pad_material,
            payload_table_material,
            structure_material,
        )
        n = asset.model.dof_count
        self._kp, self._kd = _vector(kp, n, "kp"), _vector(kd, n, "kd")
        if np.any(self._kp < 0) or np.any(self._kd < 0):
            raise ValueError("controller gains must be nonnegative")
        hard = np.array([np.inf if v is None else v for v in asset.effort_limit])
        self._limit = _vector(hard if effort_limit is None else effort_limit, n, "effort_limit")
        if np.any(self._limit < 0) or np.any(self._limit > hard):
            raise ValueError("effort limit exceeds the asset rating or is negative")
        self._lower, self._upper = np.asarray(asset.lower), np.asarray(asset.upper)
        if not isinstance(robot_state, RobotState) or robot_state.q.shape != (n,):
            raise ValueError("robot_state must match the asset joint order")
        if (
            not isinstance(payload_state, RigidBodyState)
            or payload_state.body_count != 1
            or payload_state.position.dtype != np.float64
        ):
            raise ValueError("payload_state requires one FP64 free body")
        self._check_q(robot_state.q)
        self._payload = SampledPayload(payload_geometry=payload_geometry)
        self._rigid = RigidBodyModel(
            np.array([1 / mass_kg]), self._payload.inertia(mass_kg)[None], asset.model.gravity
        )
        self._integrator = RigidSemiImplicitSolver()
        self._indices, self._local_points, self._pad_keys, excluded = _pad_points(asset, pads)
        registry = TableLinkSolids if self._payload.radius is None else SphereLinkContacts
        self._structure = registry.from_asset(asset, exclude_keys=excluded)
        self._table = FixedTableContacts.from_asset(
            asset,
            center_m=table_center_m,
            half_extents_m=table_half_extents_m,
            orientation_xyzw=table_orientation_xyzw,
            material=structure_material,
        )
        self._check_box_domain(payload_state)
        shapes = asset.collision_model.shapes
        worlds = {s.key.world for s in shapes}
        if len(worlds) > 1:
            raise ValueError("robot asset must belong to one world")
        world = next(iter(worlds), 0)
        index = 1 + max([n, *(s.key.body for s in shapes)])
        self._object_key, self._table_key = (
            CollisionKey(world, index, 0),
            CollisionKey(world, index + 1, 0),
        )
        obj = (
            CollisionShape(self._object_key, "sphere", (0,), self._payload.radius)
            if self._payload.radius is not None
            else CollisionShape(self._object_key, "hull", tuple(range(8)))
        )
        table = CollisionShape(self._table_key, "hull", tuple(range(8)))
        self._model = CollisionModel((*shapes, obj, table), asset.collision_model.excluded)
        self._names = MappingProxyType(
            {
                **{c.shape.key: c.name for c in asset.collisions},
                self._object_key: "payload",
                self._table_key: "table",
            }
        )
        self._table_pose = CollisionMotion(
            self._table.center + box_corners(self._table.half) @ self._table.rotation.T
        )
        self._arm = RobotState(robot_state.q, robot_state.v)
        self._body = payload_state.copy()
        self._lock = RLock()
        self._observation = self._read_state(
            0,
            self._arm,
            self._body,
            (np.empty(0),) * 3,
            np.zeros(n),
            np.zeros(n),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty(0),
            np.zeros(6),
            np.zeros(6),
            (),
            (),
        )

    @classmethod
    def from_urdf(cls, path, *, loader=load_urdf, **scene):
        """Load the local asset then use its actual joint order, geometry, masses and bounds."""
        return cls(loader(path), **scene)

    @property
    def asset(self):
        """The imported fixed-base asset with the joint order accepted by step()."""
        return self._asset

    @property
    def dt_s(self):
        return self._dt

    def _check_q(self, q):
        if np.any(q < self._lower) or np.any(q > self._upper):
            raise ValueError("joint position outside the asset interval")

    def _check_box_domain(self, state):
        if self._payload.radius is not None:
            return
        R = quaternion_matrix(state.orientation)[0]
        points = (
            state.position[0] - self._table.center + self._payload.vertices @ R.T
        ) @ self._table.rotation
        if np.any(abs(points[:, :2]) > self._table.half[:2]) or np.any(
            points[:, 2] < -self._table.half[2]
        ):
            raise ValueError(
                "box corner outside supported finite tabletop footprint or below underside"
            )

    def _read_state(
        self,
        step,
        arm,
        body,
        command,
        effort,
        joint_load,
        pad_points,
        pad_forces,
        normals,
        table_load,
        table_reaction,
        structural,
        links,
    ):
        return RobotPayloadObservation(
            step,
            step * self._dt,
            (step - 1) * self._dt if step else None,
            *map(
                _readonly,
                (
                    arm.q,
                    arm.v,
                    body.position[0],
                    body.orientation[0],
                    body.velocity[0],
                    body.angular_velocity[0],
                    *command,
                    effort,
                    joint_load,
                ),
            ),
            self._pad_keys if step else (),
            *map(_readonly, (pad_points, pad_forces, normals, table_load, table_reaction)),
            _frozen(structural),
            _frozen(links),
        )

    def observe(self):
        """Return an immutable accepted snapshot. Before step 1 all measured load arrays are empty or zero."""
        with self._lock:
            return self._observation

    def geometry(self):
        """Current static collision material, including table and free body, at observe().time_s."""
        with self._lock:
            reading = self._observation
            poses = self._asset.collision_motion(reading.q)
            c = reading.payload_position_m
            if self._payload.radius is None:
                c = c + self._payload.vertices @ quaternion_matrix(reading.payload_xyzw[None])[0].T
            else:
                c = c[None]
            poses[self._object_key], poses[self._table_key] = CollisionMotion(c), self._table_pose
            return SceneGeometrySnapshot(
                reading.time_s,
                self._model,
                MappingProxyType(poses),
                self._names,
                self._object_key,
                self._table_key,
            )

    def step(self, target_q, *, target_velocity=None, effort_limit=None, feedforward=None):
        """Choose bounded PD effort and advance both dynamic owners from the same input state.

        target_velocity defaults to zero. feedforward defaults to model bias compensation
        and is included inside the effort cap; supply zeros for uncompensated drives.
        An optional per-command effort_limit may only lower the configured per-joint cap.
        """
        with self._lock:
            return self._step(target_q, target_velocity, effort_limit, feedforward)

    def _step(self, target, target_velocity, effort_limit, feedforward):
        arm, body = self._arm, self._body
        n = len(arm.q)
        target = _vector(target, n, "target_q")
        self._check_q(target)
        rate = (
            np.zeros(n)
            if target_velocity is None
            else _vector(target_velocity, n, "target_velocity")
        )
        cap = self._limit if effort_limit is None else _vector(effort_limit, n, "effort_limit")
        if np.any(cap < 0) or np.any(cap > self._limit):
            raise ValueError(
                "command effort limit must be nonnegative and within configured rating"
            )
        geo, linear, angular, mass, bias = robot_dynamics_terms(self._asset.model, arm)
        ff = bias if feedforward is None else _vector(feedforward, n, "feedforward")
        effort = solve_bounded_pd(
            mass,
            bias,
            arm.q,
            arm.v,
            target,
            rate,
            self._kp,
            self._kd,
            cap,
            self._dt,
            feedforward=ff,
        ).effort
        frames = (
            geo.position,
            geo.rotation,
            geo.com_position,
            np.column_stack((linear @ arm.v, angular @ arm.v)),
        )
        c, v, w = body.position[0], body.velocity[0], body.angular_velocity[0]
        R = quaternion_matrix(body.orientation)[0]
        table = self._table
        payload = self._payload
        if payload.radius is None:
            structure = self._structure.evaluate(
                *frames,
                c,
                payload.half,
                v,
                w,
                self._structure_law,
                payload_rotation=R,
                fixed_reference_m=self._asset.root_xyz,
            )
        else:
            structure = self._structure.evaluate(
                *frames,
                c,
                payload.radius,
                v,
                w,
                self._structure_law,
                fixed_reference_m=self._asset.root_xyz,
            )
        for row in structure.contacts:
            if "table_material_weights" in row:
                row["payload_material_weights"] = row.pop("table_material_weights")
        link_table = table.evaluate(*frames)
        joint_load = structure.joint_load(linear, angular) + link_table.joint_load(linear, angular)
        wrench = structure.payload_wrench.copy()
        ids, local = self._indices, self._local_points
        pts = geo.position[ids] + np.einsum("nij,nj->ni", geo.rotation[ids], local)
        pad_forces, normal_loads = np.empty((0, 3)), np.empty(0)
        if len(ids):
            jac, _ = _point_jacobians_at(self._asset.model, geo, ids, local)
            pads = payload.pad_loads(pts, jac @ arm.v, c, R, v, w, self._pad_law)
            joint_load += np.einsum("nki,nk->i", jac, pads.forces)
            wrench += np.r_[pads.body_force, pads.body_torque]
            pad_forces, normal_loads = pads.forces, pads.normal_loads
        T = table.rotation
        forces, _, arms = payload.table_loads(
            T.T @ (c - table.center),
            T.T @ R,
            T.T @ v,
            T.T @ w,
            np.zeros(3),
            table.half,
            self._table_law,
        )
        forces, arms = forces @ T.T, arms @ T.T
        table_load = np.r_[forces.sum(axis=0), np.cross(arms, forces).sum(axis=0)]
        table_reaction = (
            link_table.table_wrench
            - np.r_[table_load[:3], table_load[3:] + np.cross(c - table.center, table_load[:3])]
        )
        wrench += table_load
        result = self._integrator.step(
            self._rigid, body, RigidBodyInputs(wrench[None, :3], wrench[None, 3:]), self._dt
        )
        if not result.report.accepted:
            raise ValueError(result.report.reason)
        new_v = arm.v + self._dt * _solve_mass_matrix(mass, effort + joint_load - bias)
        new_arm = RobotState(arm.q + self._dt * new_v, new_v)
        self._check_q(new_arm.q)
        self._check_box_domain(result.state)
        observation = self._read_state(
            self._observation.step_number + 1,
            new_arm,
            result.state,
            (target, rate, cap),
            effort,
            joint_load,
            pts,
            pad_forces,
            normal_loads,
            table_load,
            table_reaction,
            structure.contacts,
            link_table.contacts,
        )
        self._arm, self._body, self._observation = new_arm, result.state, observation
        return observation

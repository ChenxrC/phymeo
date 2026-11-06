"""Instantaneous reciprocal contacts for a URDF and ordinary finite rigid bodies.

Geometry and the spring/dashpot/slip law come from the existing robot contact
operators. This module assembles loads; it never advances a participant.
"""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from ..core.coupled import CoupledInputs
from ..core.rigid import RigidBodyInputs
from ..core.state import StepInputs
from ..dynamics.robot_kinematics import _point_jacobians_at
from ..dynamics.rotation import quaternion_matrix
from ..geometry.collision_witness import ConvexContactPoint
from ..solvers.collision_loads import LoadSite, PairLoad, SurfaceLoads
from ..solvers.sampled_payload import SampledPayload
from ..solvers.sphere_link_contact import LinkSolid, SphereLinkContacts
from ..solvers.table_box_patch import TableLinkSolids
from .robot_payload_scene import _pad_points

_BITS = np.array(list(product((0, 1), repeat=3)))


def rigid_solid(item, label, owner=0):
    """Validate the first slice: a COM-centred sphere or a local-axis cuboid."""
    shape = item.shape
    kind = "box" if shape.kind == "hull" else shape.kind
    if kind not in {"box", "sphere"}:
        raise ValueError(f"{label}: robot scenes support box/sphere rigid bodies, got {shape.kind}")
    solid = LinkSolid(label, label, owner, kind, item.vertices, shape.radius)
    if kind == "box":
        if not np.allclose(solid._axes, np.eye(3), atol=1e-12, rtol=0) or not np.allclose(
            solid._center, 0.0, atol=1e-12, rtol=0
        ):
            raise ValueError(f"{label}: rigid box must use its centred principal local axes")
    elif np.any(np.asarray(item.vertices)[0]):
        raise ValueError(f"{label}: free sphere must be centred at its COM")
    return solid


def _weights(shape, pose, point):
    """Original vertex addresses; sphere/capsule skin is carried by the surface point."""
    xyz = np.asarray(pose.start)
    ids = shape.vertex_ids
    if shape.kind == "sphere":
        return ((ids[0], 1.0),)
    if shape.kind == "capsule":
        line = xyz[1] - xyz[0]
        t = float(np.clip((np.asarray(point) - xyz[0]) @ line / (line @ line), 0, 1))
        return tuple((node, weight) for node, weight in zip(ids, (1 - t, t)) if weight)
    if shape.kind != "hull" or len(xyz) != 8:
        raise ValueError(f"{shape.key}: no supported material address")
    edges = xyz[[4, 2, 1]] - xyz[0]
    u = np.linalg.solve(edges.T, np.asarray(point) - xyz[0])
    if np.any(u < -1e-8) or np.any(u > 1 + 1e-8):
        raise ValueError(f"{shape.key}: contact point outside its cuboid material chart")
    u = np.clip(u, 0, 1)
    coeff = np.prod(np.where(_BITS, u, 1 - u), axis=1)
    return tuple((node, float(weight)) for node, weight in zip(ids, coeff) if weight)


class _Records:
    def __init__(self, shapes, poses):
        self.shapes, self.poses = shapes, poses
        self.contacts, self.details = [], []

    def add(self, first, second, row, law, channel):
        # Per-site PairLoad retains varying radial normals of sampled pads.
        # Repeated pair keys are distinct actual samples, never duplicate solves.
        normal = np.asarray(row["normal_world"])
        force = np.asarray(row["force_on_payload_N"])
        pa, pb = row["link_surface_m"], row["payload_surface_m"]
        ca, cb = row["link_com_world_m"], row["payload_com_world_m"]
        point = tuple(row["point_world_m"])
        if second < first:
            first, second = second, first
            normal, force = -normal, -force
            pa, pb, ca, cb = pb, pa, cb, ca
        wa = _weights(self.shapes[first], self.poses[first], pa)
        wb = _weights(self.shapes[second], self.poses[second], pb)
        material = ConvexContactPoint(tuple(pa), tuple(pb), wa, wb)
        weight = float(row["weight"])
        magnitude = float(row["normal_load_N"])
        site = LoadSite(
            material,
            weight,
            float(row["normal_rate_m_s"]),
            magnitude,
            point,
            point,
            tuple(force),
            float(row["pair_power_W"]),
        )
        gap = float(row["gap_m"])
        self.contacts.append(
            PairLoad(
                first,
                second,
                tuple(normal),
                gap,
                magnitude,
                0.5 * law.stiffness * min(gap, 0.0) ** 2 * weight,
                site.power_w,
                (site,),
            )
        )
        self.details.append(
            dict(
                first=first,
                second=second,
                channel=channel,
                feature=row["feature"],
                gap_m=gap,
                normal_world=tuple(normal),
                application_point_first_m=point,
                application_point_second_m=point,
                first_com_world_m=tuple(ca),
                second_com_world_m=tuple(cb),
                first_surface_m=tuple(pa),
                second_surface_m=tuple(pb),
                first_weights=wa,
                second_weights=wb,
                force_on_first_N=tuple(-force),
                force_on_second_N=tuple(force),
                torque_on_first_Nm=tuple(np.cross(np.asarray(point) - ca, -force)),
                torque_on_second_Nm=tuple(np.cross(np.asarray(point) - cb, force)),
                normal_load_N=magnitude,
                fraction=weight,
                power_W=site.power_w,
            )
        )


@dataclass(frozen=True)
class AssembledContacts:
    loads: SurfaceLoads
    joint_effort_full: np.ndarray
    link_wrenches_world: np.ndarray
    fixed_root_contact_wrench: np.ndarray
    details: tuple


class RobotSceneContacts:
    """One evaluation uses the unchanged robot and rigid input states for every pair."""

    def __init__(self, tree, robot_addr, rigid_items, rigid_rows, pairs, key_map, pads, model):
        self.tree, self.robot_addr = tree, robot_addr
        self.items, self.rows, self.pairs = rigid_items, rigid_rows, pairs
        self.key_map = key_map
        self.shapes = {s.key: s for s in model.shapes}
        indices, points, pad_keys, excluded = _pad_points(tree, pads)
        self.pad_indices, self.pad_points, self.pad_keys = indices, points, pad_keys
        full = TableLinkSolids.from_asset(tree).solids
        self.original = {(c.link_index, c.link_name, c.name): c.shape.key for c in tree.collisions}
        self.robot_registry, self.payloads = {}, {}
        for addr, item in rigid_items.items():
            solid = rigid_solid(item, str(addr))
            geometry = (
                {"kind": "sphere", "radius_m": solid.radius}
                if solid.kind == "sphere"
                else {"kind": "box", "full_size_m": (2 * solid._half).tolist()}
            )
            self.payloads[addr] = SampledPayload(payload_geometry=geometry)
            # Pads replace their solid only against dynamic objects, never the support.
            selected = [
                s
                for s in full
                if item.inverse_mass == 0
                or self.original[s.owner_index, s.link, s.shape] not in excluded
            ]
            cls = SphereLinkContacts if solid.kind == "sphere" else TableLinkSolids
            self.robot_registry[addr] = cls(selected)
        self.free_registry = {}
        for a, b in combinations(rigid_items, 2):
            solid = rigid_solid(rigid_items[a], str(a))
            cls = SphereLinkContacts if rigid_items[b].shape.kind == "sphere" else TableLinkSolids
            self.free_registry[a, b] = cls((solid,))

    @staticmethod
    def _evaluate(registry, frames, payload, body, row, law, fixed_reference):
        c, R = body.position[row], quaternion_matrix(body.orientation[row : row + 1])[0]
        args = (
            *frames,
            c,
            payload.radius if payload.radius is not None else payload.half,
            body.velocity[row],
            body.angular_velocity[row],
            law,
        )
        kwargs = dict(fixed_reference_m=fixed_reference)
        if payload.radius is None:
            kwargs["payload_rotation"] = R
        return registry.evaluate(*args, **kwargs)

    def evaluate(self, physical, geo, linear, angular, body, poses, *, include_rigid=True):
        n, count = len(physical.q), body.body_count
        force, torque = np.zeros((count, 3)), np.zeros((count, 3))
        link_load = np.zeros((n, 6))
        root_load = np.zeros(6)
        records = _Records(self.shapes, poses)
        frames = (
            geo.position,
            geo.rotation,
            geo.com_position,
            np.column_stack((linear @ physical.v, angular @ physical.v)),
        )
        ids = self.pad_indices
        points = geo.position[ids] + np.einsum("nij,nj->ni", geo.rotation[ids], self.pad_points)
        jac = (
            _point_jacobians_at(self.tree.model, geo, ids, self.pad_points)[0] if len(ids) else None
        )
        for addr, item in self.items.items():
            law = self.pairs[tuple(sorted((self.robot_addr, addr)))]
            if law is None:
                continue
            row = self.rows[addr]
            raw = self._evaluate(
                self.robot_registry[addr],
                frames,
                self.payloads[addr],
                body,
                row,
                law,
                self.tree.root_xyz,
            )
            link_load += raw.link_wrenches
            force[row] += raw.payload_wrench[:3]
            torque[row] += raw.payload_wrench[3:]
            for hit in raw.contacts:
                original = self.original[hit["owner_index"], hit["link"], hit["shape"]]
                hit = dict(hit, payload_com_world_m=body.position[row].tolist())
                hit["normal_rate_m_s"] = float(
                    np.asarray(hit["relative_velocity_m_s"]) @ hit["normal_world"]
                )
                if hit["owner_index"] < 0:
                    root_load += np.r_[hit["force_on_link_N"], hit["torque_on_link_Nm"]]
                records.add(self.key_map[original], item.shape.key, hit, law, "robot-structure")
            if not len(ids) or item.inverse_mass == 0:
                continue
            payload = self.payloads[addr]
            c, R = body.position[row], quaternion_matrix(body.orientation[row : row + 1])[0]
            samples = payload.pad_loads(
                points, jac @ physical.v, c, R, body.velocity[row], body.angular_velocity[row], law
            )
            force[row] += samples.body_force
            torque[row] += samples.body_torque
            for k in np.flatnonzero(samples.normal_loads):
                p, f = points[k], -samples.forces[k]
                if payload.radius is not None:
                    normal = (c - p) / np.linalg.norm(c - p)
                else:
                    local = R.T @ (p - c)
                    axis = int(np.argmin(payload.half - np.abs(local)))
                    normal = -np.sign(local[axis]) * R[:, axis]
                rel = (
                    body.velocity[row]
                    + np.cross(body.angular_velocity[row], p - c)
                    - jac[k] @ physical.v
                )
                i = int(ids[k])
                # Pad generalized effort is assembled via point J; the COM wrench is also observable.
                link_load[i, :3] += -f
                link_load[i, 3:] += np.cross(p - geo.com_position[i], -f)
                hit = dict(
                    normal_world=normal,
                    force_on_payload_N=f,
                    link_surface_m=p,
                    payload_surface_m=p - samples.depths[k] * normal,
                    point_world_m=p,
                    link_com_world_m=geo.com_position[i],
                    payload_com_world_m=c,
                    weight=1.0,
                    normal_load_N=samples.normal_loads[k],
                    normal_rate_m_s=float(rel @ normal),
                    gap_m=-samples.depths[k],
                    pair_power_W=float(f @ rel),
                    feature="sampled-pad",
                )
                records.add(self.key_map[self.pad_keys[k]], item.shape.key, hit, law, "robot-pad")
        # link_load already contains pad COM wrenches; transpose it exactly once.
        joint = np.einsum("nki,nk->i", linear, link_load[:, :3]) + np.einsum(
            "nki,nk->i", angular, link_load[:, 3:]
        )
        if include_rigid:
            rigid_loads, rigid_details = self.evaluate_rigid(body, poses)
            force += rigid_loads.inputs.rigid.force
            torque += rigid_loads.inputs.rigid.torque
            records.contacts.extend(rigid_loads.contacts)
            records.details.extend(rigid_details)
        loads = SurfaceLoads(
            CoupledInputs(StepInputs.zeros(0), RigidBodyInputs(force, torque)),
            tuple(records.contacts),
        )
        return AssembledContacts(loads, joint, link_load, root_load, tuple(records.details))

    def evaluate_rigid(self, body, poses):
        """Sample all rigid-rigid pairs once, without evaluating or advancing a robot."""
        force = np.zeros((body.body_count, 3))
        torque = np.zeros_like(force)
        records = _Records(self.shapes, poses)
        for (a, b), registry in self.free_registry.items():
            law = self.pairs[a, b]
            if law is None:
                continue
            ia, ib = self.rows[a], self.rows[b]
            Ra = quaternion_matrix(body.orientation[ia : ia + 1])
            local_frames = (
                body.position[ia : ia + 1],
                Ra,
                body.position[ia : ia + 1],
                np.column_stack((body.velocity[ia : ia + 1], body.angular_velocity[ia : ia + 1])),
            )
            raw = self._evaluate(
                registry, local_frames, self.payloads[b], body, ib, law, body.position[ia]
            )
            force[ia] += raw.link_wrenches[0, :3]
            torque[ia] += raw.link_wrenches[0, 3:]
            force[ib] += raw.payload_wrench[:3]
            torque[ib] += raw.payload_wrench[3:]
            for hit in raw.contacts:
                hit = dict(hit, payload_com_world_m=body.position[ib].tolist())
                hit["normal_rate_m_s"] = float(
                    np.asarray(hit["relative_velocity_m_s"]) @ hit["normal_world"]
                )
                records.add(
                    self.items[a].shape.key, self.items[b].shape.key, hit, law, "rigid-pair"
                )
        loads = SurfaceLoads(
            CoupledInputs(StepInputs.zeros(0), RigidBodyInputs(force, torque)),
            tuple(records.contacts),
        )
        return loads, tuple(records.details)

"""Convex extensions of the existing robot/free-body contact assembly.

This adapter samples existing entry poses and returns loads only. It owns no
physical state, integration, cache, or publication transaction.
"""

from dataclasses import replace
from itertools import combinations

import numpy as np

from ..core.coupled import CoupledInputs
from ..core.rigid import RigidBodyInputs
from ..core.state import StepInputs
from ..geometry.contact_patch_adapter import contact_patch_sample
from ..solvers.collision_loads import SurfaceLoads
from ..solvers.contact_patch_loads import sample_patch_loads
from ..solvers.sphere_link_contact import LinkSolid


def requires_convex_owner(tree, items):
    """Keep established OBB/pad routes; add actual general convex geometry."""
    from ._robot_scene_contacts import rigid_solid

    if not items:
        return False
    if any(not isinstance(rigid_solid(item, str(addr)), LinkSolid) for addr, item in items.items()):
        return True
    for col in tree.collisions:
        if col.shape.kind != "hull":
            continue
        try:
            LinkSolid(col.name, col.link_name, col.link_index, "box", col.local_vertices, 0)
        except ValueError:
            return True
    return False


def convex_owner_patch(a, b, poses):
    ma, mb = poses[a.key], poses[b.key]
    if any(m.end is not None or m.turn is not None for m in (ma, mb)):
        raise ValueError("convex owner loads require instantaneous entry poses")
    x, y = np.asarray(ma.start), np.asarray(mb.start)
    if np.any(x.max(axis=0) + a.radius <= y.min(axis=0) - b.radius) or np.any(
        y.max(axis=0) + b.radius <= x.min(axis=0) - a.radius
    ):
        return None
    legacy, patch = _legacy_sphere_patch(a, b, poses)
    return patch if legacy else contact_patch_sample(a, b, x, y)


def _row(site, second_force, ca, cb, va, vb):
    w = site.witness
    return dict(
        normal_world=w.normal,
        force_on_payload_N=second_force,
        link_surface_m=w.first_point,
        payload_surface_m=w.second_point,
        link_com_world_m=ca,
        payload_com_world_m=cb,
        point_world_m=site.load_point,
        weight=site.weight,
        normal_load_N=float(second_force @ w.normal),
        normal_rate_m_s=float((vb - va) @ w.normal),
        gap_m=w.gap_m,
        pair_power_W=float(second_force @ (vb - va)),
        first_weights=w.first_weights,
        second_weights=w.second_weights,
        feature="convex-owner",
    )


class ConvexRobotRigidContacts:
    def __init__(self, tree, robot_addr, items, rows, pairs, key_map, pads, model):
        from ._robot_scene_contacts import RobotSceneContacts, rigid_solid
        from .robot_payload_scene import _pad_points

        _pad_points(tree, pads)  # Resolve ambiguity on the complete authored asset.

        legacy_columns = tuple(c for c in tree.collisions if legacy_robot_collider(c))
        legacy_items = {
            a: item for a, item in items.items() if isinstance(rigid_solid(item, str(a)), LinkSolid)
        }
        self._legacy_keys = {key_map[c.shape.key] for c in legacy_columns}
        self._legacy_items = set(legacy_items)
        # Existing shape pairs retain their quadrature, pads and public features.
        # New geometry never switches unrelated old pairs to a different force law.
        if tuple(pads) and any(
            a not in legacy_items and pairs[tuple(sorted((robot_addr, a)))] is not None
            for a in items
        ):
            raise ValueError(
                "sampled pads against general convex rigid geometry are not implemented"
            )
        self._legacy = RobotSceneContacts(
            replace(tree, collisions=legacy_columns),
            robot_addr,
            legacy_items,
            rows,
            pairs,
            key_map,
            pads,
            model,
        )
        self.tree, self.robot_addr, self.items, self.rows, self.pairs = (
            tree,
            robot_addr,
            items,
            rows,
            pairs,
        )
        self.shapes = {s.key: s for s in model.shapes}
        self.colliders = tuple(
            (c.link_index, self.shapes[key_map[c.shape.key]]) for c in tree.collisions
        )
        for c in tree.collisions:
            validate_convex_shape(c.shape, c.local_vertices)
        for item in items.values():
            validate_convex_shape(item.shape, item.vertices)
        for shape in (*[s for _, s in self.colliders], *[item.shape for item in items.values()]):
            if shape.kind not in {"sphere", "capsule", "hull"}:
                raise ValueError("convex owner adapter supports sphere, capsule and hull only")

    def evaluate(self, physical, geo, linear, angular, body, poses, *, include_rigid=True):
        from ._robot_scene_contacts import AssembledContacts, _Records

        prior = self._legacy.evaluate(
            physical, geo, linear, angular, body, poses, include_rigid=False
        )
        force, torque = (
            prior.loads.inputs.rigid.force.copy(),
            prior.loads.inputs.rigid.torque.copy(),
        )
        link, root = prior.link_wrenches_world.copy(), prior.fixed_root_contact_wrench.copy()
        twists = np.column_stack((linear @ physical.v, angular @ physical.v))
        records = _Records(self.shapes, poses)
        records.contacts.extend(prior.loads.contacts)
        records.details.extend(prior.details)
        for addr, item in self.items.items():
            law = self.pairs[tuple(sorted((self.robot_addr, addr)))]
            if law is None:
                continue
            row = self.rows[addr]
            cb = body.position[row]
            for index, shape in self.colliders:
                if addr in self._legacy_items and shape.key in self._legacy_keys:
                    continue
                patch = convex_owner_patch(shape, item.shape, poses)
                if patch is None:
                    continue
                points = np.array([s.load_point for s in patch.sites])
                ca = np.asarray(self.tree.root_xyz) if index < 0 else geo.com_position[index]
                va = (
                    np.zeros_like(points)
                    if index < 0
                    else twists[index, :3] + np.cross(twists[index, 3:], points - ca)
                )
                vb = body.velocity[row] + np.cross(body.angular_velocity[row], points - cb)
                loads = sample_patch_loads(patch, law, va, vb)
                for k, site in enumerate(patch.sites):
                    fa, fb = loads["first_force"][k], loads["second_force"][k]
                    if not np.any(fa):
                        continue
                    wrench = np.r_[fa, np.cross(points[k] - ca, fa)]
                    if index < 0:
                        root += wrench
                    else:
                        link[index] += wrench
                    force[row] += fb
                    torque[row] += np.cross(points[k] - cb, fb)
                    records.add(
                        shape.key,
                        item.shape.key,
                        _row(site, fb, ca, cb, va[k], vb[k]),
                        law,
                        "robot-structure",
                    )
        joint = np.einsum("nki,nk->i", linear, link[:, :3]) + np.einsum(
            "nki,nk->i", angular, link[:, 3:]
        )
        if include_rigid:
            extra, details = self.evaluate_rigid(body, poses)
            force += extra.inputs.rigid.force
            torque += extra.inputs.rigid.torque
            records.contacts.extend(extra.contacts)
            records.details.extend(details)
        loads = SurfaceLoads(
            CoupledInputs(StepInputs.zeros(0), RigidBodyInputs(force, torque)),
            tuple(records.contacts),
        )
        return AssembledContacts(loads, joint, link, root, tuple(records.details))

    def evaluate_rigid(self, body, poses):
        from ._robot_scene_contacts import _Records

        prior, details = self._legacy.evaluate_rigid(body, poses)
        force, torque = prior.inputs.rigid.force.copy(), prior.inputs.rigid.torque.copy()
        records = _Records(self.shapes, poses)
        records.contacts.extend(prior.contacts)
        records.details.extend(details)
        for a, b in combinations(self.items, 2):
            if a in self._legacy_items and b in self._legacy_items:
                continue
            law = self.pairs[a, b]
            if law is None:
                continue
            ia, ib = self.rows[a], self.rows[b]
            sa, sb = self.items[a].shape, self.items[b].shape
            patch = convex_owner_patch(sa, sb, poses)
            if patch is None:
                continue
            points = np.array([s.load_point for s in patch.sites])
            ca, cb = body.position[ia], body.position[ib]
            va = body.velocity[ia] + np.cross(body.angular_velocity[ia], points - ca)
            vb = body.velocity[ib] + np.cross(body.angular_velocity[ib], points - cb)
            loads = sample_patch_loads(patch, law, va, vb)
            for k, site in enumerate(patch.sites):
                fa, fb = loads["first_force"][k], loads["second_force"][k]
                if not np.any(fa):
                    continue
                force[ia] += fa
                force[ib] += fb
                torque[ia] += np.cross(points[k] - ca, fa)
                torque[ib] += np.cross(points[k] - cb, fb)
                records.add(sa.key, sb.key, _row(site, fb, ca, cb, va[k], vb[k]), law, "rigid-pair")
        return SurfaceLoads(
            CoupledInputs(StepInputs.zeros(0), RigidBodyInputs(force, torque)),
            tuple(records.contacts),
        ), tuple(records.details)


def validate_convex_shape(shape, points):
    """Validate the complete local body before broad rejection can hide it."""
    from ..constraints.collision_shapes import CollisionMotion
    from ..geometry._convex_hull import HullShape

    motion = CollisionMotion(points)
    if len(motion.start) != len(shape.vertex_ids):
        raise ValueError("convex material vertices do not match registration")
    if shape.kind == "hull":
        if shape.radius != 0:
            raise ValueError("rounded hull owner contacts are not implemented")
        HullShape(shape, motion)
    elif shape.kind == "capsule":
        if np.array_equal(motion.start[0], motion.start[1]):
            raise ValueError("capsule needs a nonzero finite axis")
    elif shape.kind != "sphere":
        raise ValueError("unsupported convex owner shape")


def _legacy_sphere_patch(a, b, poses):
    """Positive primitive validation selects the established single-point fast path."""
    from ..geometry.collision_witness import CollisionWitness
    from ..geometry.contact_patch_adapter import ContactPatchSample, ContactSite
    from ._robot_scene_contacts import _weights

    if "sphere" not in (a.kind, b.kind):
        return False, None
    swapped = b.kind != "sphere"
    solid_shape, ball = (b, a) if swapped else (a, b)
    kind = "box" if solid_shape.kind == "hull" else solid_shape.kind
    if kind == "box" and len(solid_shape.vertex_ids) != 8:
        return False, None
    if kind not in {"sphere", "capsule", "box"}:
        return False, None
    try:
        solid = LinkSolid(
            "primitive", "primitive", -1, kind, poses[solid_shape.key].start, solid_shape.radius
        )
    except ValueError:
        if kind == "box":
            return False, None
        raise
    center = np.asarray(poses[ball.key].start[0])
    witness = solid.witness(center, ball.radius)
    if witness is None:
        return True, None
    gap, normal, pa, feature = witness
    pb = center - ball.radius * normal
    wa = _weights(solid_shape, poses[solid_shape.key], pa)
    wb = _weights(ball, poses[ball.key], pb)
    offset = -ball.radius * normal
    other_offset = (
        solid_shape.radius * normal if solid_shape.kind in {"sphere", "capsule"} else None
    )
    if swapped:
        pa, pb, wa, wb = pb, pa, wb, wa
        normal = -normal
        oa, ob = offset, other_offset
    else:
        oa, ob = other_offset, offset
    value = CollisionWitness(
        gap,
        tuple(normal),
        tuple(pa),
        tuple(pb),
        wa,
        wb,
        "legacy-analytic",
        first_offset=None if oa is None else tuple(oa),
        second_offset=None if ob is None else tuple(ob),
    )
    site = ContactSite(value, tuple((pa + pb) / 2), 1.0)
    return True, ContactPatchSample((site,), 0, "legacy-analytic:" + feature)


def patch_feature(patch):
    if patch.source.startswith("legacy-analytic:"):
        return patch.source.split(":", 1)[1]
    return ("convex-point", "convex-line", "convex-face")[patch.dimension]


def legacy_robot_collider(collider):
    if collider.shape.kind in {"sphere", "capsule"}:
        return True
    if collider.shape.kind != "hull" or collider.shape.radius != 0:
        return False
    try:
        LinkSolid(
            collider.name,
            collider.link_name,
            collider.link_index,
            "box",
            collider.local_vertices,
            0.0,
        )
    except ValueError:
        return False
    return True

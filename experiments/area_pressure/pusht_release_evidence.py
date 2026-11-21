"""Observe native shape and material release on both ends of a real PushT step."""

from dataclasses import dataclass

import mujoco
import numpy as np

from phymeo.backends.mujoco_tyx._constraint_model import constraint_pairs
from phymeo.control.manipulation_release_monitor import (
    ReleasePairEvidence,
    ReleasePairKey,
    ReleaseStep,
)


@dataclass(frozen=True)
class NativePose:
    depth: float
    contact: bool
    separated: bool
    speed: float | None


def native_release_poses(model, q, v, pairs):
    """MuJoCo position-stage geometry only. Velocities use both point Jacobians."""
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_fwdPosition(model, data)
    manifolds = {pair: [] for pair in pairs}
    for contact in data.contact:
        key = tuple(sorted(map(int, contact.geom)))
        if key in manifolds:
            manifolds[key].append(contact)
    poses = {}
    for (a, b), manifold in manifolds.items():
        witness = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, a, b, 1.0, witness))
        depth = max(0.0, -min([distance, *(c.dist for c in manifold)]))
        touching = [c for c in manifold if c.dist <= 0]
        speeds = []
        for c in touching:
            ja, jb = (np.zeros((3, model.nv)) for _ in range(2))
            bodies = model.geom_bodyid[c.geom]
            mujoco.mj_jac(model, data, ja, None, c.pos, int(bodies[0]))
            mujoco.mj_jac(model, data, jb, None, c.pos, int(bodies[1]))
            speeds.append(float(c.frame[:3] @ (jb - ja) @ data.qvel))
        if not speeds and distance <= 0:
            raise ValueError("signed native overlap has no contact normal for " + str((a, b)))
        poses[a, b] = NativePose(
            depth, bool(touching), distance > 0 and not touching, min(speeds) if speeds else None
        )
    return poses


def native_loads(pairs, contacts):
    """Accumulate the two true applied forces in stable geometry order."""
    loads = {pair: [0.0, np.zeros(3), np.zeros(3)] for pair in pairs}
    for contact in contacts:
        left, right = map(int, contact.geometry_ids)
        key = tuple(sorted((left, right)))
        if key not in loads:
            raise ValueError("unknown native force receiver")
        a, b = contact.force_on_geometry1_world, contact.force_on_geometry2_world
        if left != key[0]:
            a, b = b, a
        row = loads[key]
        row[0] += float(contact.normal_force)
        row[1] += a
        row[2] += b
    return loads


def _prior_loads(model, pairs, saved):
    # The old writer recorded every table receipt, with one force on the
    # supported body. Its opposite is reconstructed for this seed only.
    if len(saved["supports"]) != saved["detected_contacts"]:
        raise ValueError("archived source omitted native force receivers")
    loads = native_loads(pairs, ())
    for support in saved["supports"]:
        names = support["geometry_names"]
        if names.count("support") != 1:
            raise ValueError("archived support identity")
        ids = tuple(int(model.geom(name).id) for name in names)
        key = tuple(sorted(ids))
        force = np.asarray(support["force_on_supported_world_n"])
        on_a = force if model.geom(key[0]).name != "support" else -force
        row = loads[key]
        row[0] += support["normal_n"]
        row[1] += on_a
        row[2] -= on_a
    return loads


def _build(
    model,
    asset,
    before,
    after,
    tick,
    *,
    previous=None,
    receipt=None,
    geometry=None,
    diagnostics=None,
):
    from .pusht_release_material import material_evidence

    pairs = tuple(tuple(sorted(map(int, pair))) for pair in constraint_pairs(model))
    first = native_release_poses(model, before["q"], before["v"], pairs)
    last = native_release_poses(model, after["q"], after["v"], pairs)
    loads = (
        _prior_loads(model, pairs, previous["native"])
        if previous is not None
        else native_loads(pairs, receipt.contacts)
    )
    rows = []
    for pair in pairs:
        a, b = first[pair], last[pair]
        load, fa, fb = loads[pair]
        key = ReleasePairKey(
            "native-geometry",
            tuple(model.geom(i).name for i in pair),
            tuple(int(model.geom_bodyid[i]) for i in pair),
        )
        rows.append(
            ReleasePairEvidence(
                key,
                a.depth,
                b.depth,
                load,
                a.speed,
                b.speed,
                a.contact,
                b.contact,
                b.separated,
                tuple(fa),
                tuple(fb),
            )
        )
    rows.extend(
        material_evidence(
            model,
            asset,
            before,
            after,
            previous=previous,
            receipt=receipt,
            geometry=geometry,
            diagnostics=diagnostics,
        )
    )
    return ReleaseStep(
        6,
        tick,
        tick + 1,
        before["time"],
        after["time"],
        before["q"],
        before["v"],
        after["q"],
        after["v"],
        tuple(rows),
    )


def build_release_seed(entry, asset):
    previous = entry.previous
    if (
        previous["output_state"] != entry.state
        or previous["original_output_tick"] != entry.tick
        or previous["original_input_tick"] + 1 != entry.tick
    ):
        raise ValueError("release seed differs from archived state")
    return _build(
        entry.model, asset, previous["input_state"], entry.state, entry.tick - 1, previous=previous
    )


def build_release_step(entry, asset, before, after, geometry, *, diagnostics=None):
    receipt = after.last_step
    if (
        receipt is None
        or receipt.tick != after.tick
        or after.tick != before.tick + 1
        or geometry.surface.stamp.step_index != before.tick
        or not np.array_equal(receipt.input_q, before.q)
        or not np.array_equal(receipt.input_v, before.v)
        or receipt.input_time != before.time
        or receipt.output_time != after.time
    ):
        raise ValueError("release geometry/receipt/state phase mismatch")

    def state(snap):
        return dict(q=snap.q, v=snap.v, time=snap.time)

    return _build(
        entry.model,
        asset,
        state(before),
        state(after),
        entry.tick + before.tick,
        receipt=receipt,
        geometry=geometry,
        diagnostics=diagnostics,
    )


def release_support_keys(seed):
    return tuple(
        row.key
        for row in seed.pairs
        if row.key.kind == "native-geometry"
        and "support" in row.key.identity
        and (row.input_contact or row.output_contact or (row.normal_load_n or 0) > 0)
    )

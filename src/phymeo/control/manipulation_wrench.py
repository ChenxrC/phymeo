"""Measure actual object contact forces and moments from public entry arrays.

This HOST diagnostic is stateless. It neither estimates missing contacts nor
claims force closure, support success, grasp stability or dynamic equilibrium.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .manipulation_contact import _array, _index, _reject_area_entry


def _require(condition, message):
    if not condition:
        raise ValueError("object contact wrench: " + message)


def _tuple3(value):
    return tuple(float(component) for component in value)


@dataclass(frozen=True)
class ObjectContactWrenchRecord:
    index: int
    active: bool
    other_geom: int
    other_body: int
    point_world: tuple[float, float, float]
    force_world: tuple[float, float, float]
    couple_world: tuple[float, float, float]
    moment_com_world: tuple[float, float, float]
    normal_load: float
    group_member: bool


@dataclass(frozen=True)
class ObjectContactWrenchMeasurement:
    time_s: float
    input_tick: int
    output_tick: int
    reference_com_world: tuple[float, float, float]
    contacts: tuple[ObjectContactWrenchRecord, ...]
    total_force_world: tuple[float, float, float]
    total_moment_com_world: tuple[float, float, float]
    group_force_world: tuple[float, float, float]
    group_moment_com_world: tuple[float, float, float]
    other_force_world: tuple[float, float, float]
    other_moment_com_world: tuple[float, float, float]
    group_normal_loads: tuple[float, ...]


def measure_object_contact_wrench(
    arrays,
    row,
    *,
    object_body,
    object_com_world,
    group_geom_ids=(),
):
    """Sum every object contact about the caller-supplied entry COM, in world axes.

    The caller must bind COM, body/geom identities and arrays to the same model
    and accepted-step input. COM is NOT inferred from entry_body_position, whose
    values are body origins. There is no native model or hidden state access.
    Units are seconds, newtons and newton-metres; clock time is high minus its
    compensation. Frame rows are world contact axes and the local wrench acts
    on geom2. Reversing to body1 negates force AND intrinsic contact couple.

    group_geom_ids is an explicit tuple of zero to eight unique opposing geom
    IDs. Other object contacts, including table contacts, remain in the total.
    Records include detected inactive contacts with zero load and active=False.
    No frame is normalized or repaired. All returned vectors are owned tuples.
    An invalid entry returns None without reading any stale clock/contact/body
    payload; its metadata and explicit caller arguments must still be valid.
    """
    _require(isinstance(arrays, Mapping), "arrays must be a public NumPy mapping")
    selected, body = _index(row, "row"), _index(object_body, "object_body")
    _require(
        isinstance(group_geom_ids, tuple) and len(group_geom_ids) <= 8,
        "group_geom_ids must be a tuple of zero to eight IDs",
    )
    group = tuple(_index(value, "group geom") for value in group_geom_ids)
    _require(
        len(set(group)) == len(group) and all(value <= 2147483647 for value in group),
        "group geom IDs must be unique native int32 identities",
    )
    raw_com = np.asarray(object_com_world)
    _require(
        raw_com.shape == (3,) and raw_com.dtype.kind in "fiu" and np.isfinite(raw_com).all(),
        "object_com_world must be a finite numeric world 3-vector",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        com = np.array(raw_com, dtype=np.float64, copy=True)
    _require(np.isfinite(com).all(), "object_com_world is not finite in measurement precision")
    metadata = _array(arrays, "entry_meta", (None, 6), "iu")
    worlds = metadata.shape[0]
    _require(
        1 <= worlds <= 64 and selected < worlds and body < 128,
        "world/row/object body exceeds bounded dimensions",
    )
    _reject_area_entry(arrays, selected, worlds)
    valid, before, after, count, active_count, overflow = map(int, metadata[selected])
    _require(
        valid in (0, 1) and 0 <= after <= 2147483647 and overflow == 0,
        "invalid entry validity, output tick or overflow",
    )
    if valid == 0:
        _require(before == count == active_count == 0, "invalid empty-entry metadata")
        return None
    _require(0 <= before < 2147483647 and after == before + 1, "entry ticks are not adjacent")

    clocks = _array(arrays, "entry_clock", (worlds, 4), "f")
    ids = _array(arrays, "entry_contact_ids", (worlds, None, 8), "iu")
    capacity = ids.shape[1]
    _require(1 <= capacity <= 4096, "contact storage exceeds bounded capacity")
    data = _array(arrays, "entry_contact_data", (worlds, capacity, 20), "f")
    positions = _array(arrays, "entry_body_position", (worlds, None, 3), "f")
    bodies = positions.shape[1]
    _require(1 <= bodies <= 128, "body storage exceeds bounded capacity")
    mask = _array(arrays, "entry_body_mask", (worlds, bodies), "iu")
    _require(0 <= active_count <= count <= capacity, "contact counts exceed storage")
    _require(np.all((mask[selected] == 0) | (mask[selected] == 1)), "invalid body mask")
    _require(body < bodies and mask[selected, body] == 1, "object body is invalid or padded")
    _require(
        np.isfinite(clocks[selected]).all() and np.isfinite(positions[selected]).all(),
        "nonfinite entry clock or body position",
    )
    _require(not np.any(positions[selected][mask[selected] == 0]), "nonzero body padding")
    stamp = clocks[selected]
    time_s, output_time = float(stamp[0]) - float(stamp[1]), float(stamp[2]) - float(stamp[3])
    _require(
        isfinite(time_s) and isfinite(output_time) and 0 <= time_s < output_time,
        "entry clock does not advance from input to output",
    )
    contacts, payload = ids[selected], data[selected]
    _require(np.isfinite(payload).all(), "nonfinite contact data")
    _require(
        not np.any(contacts[count:]) and not np.any(payload[count:]), "nonzero contact padding"
    )

    records, geom_bodies = [], {}
    grouped = {geom: index for index, geom in enumerate(group)}
    loads = np.zeros(len(group), dtype=np.float64)
    group_force, group_moment = np.zeros(3), np.zeros(3)
    other_force, other_moment = np.zeros(3), np.zeros(3)
    seen_active = 0
    for index in range(count):
        present, active, geom1, geom2, body1, body2, feature, dim = map(int, contacts[index])
        _require(present == 1 and active in (0, 1), "invalid contact prefix flags")
        _require(
            0 <= geom1 <= 2147483647
            and 0 <= geom2 <= 2147483647
            and geom1 != geom2
            and 0 <= feature <= 2147483647
            and dim in (1, 3, 4, 6),
            "invalid contact geometry, feature or dimension",
        )
        _require(
            0 <= body1 < bodies
            and 0 <= body2 < bodies
            and body1 != body2
            and mask[selected, body1] == mask[selected, body2] == 1,
            "contact body is invalid or padded",
        )
        for geom, owner in ((geom1, body1), (geom2, body2)):
            _require(geom not in geom_bodies or geom_bodies[geom] == owner, "geom changes body")
            geom_bodies[geom] = owner
            _require(geom not in grouped or owner != body, "group geom belongs to the object")
        values = payload[index].astype(np.float64, copy=True)
        frame, wrench = values[4:13].reshape(3, 3), values[13:19]
        _require(
            np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-4)
            and abs(float(np.linalg.det(frame)) - 1.0) <= 1e-4,
            "contact frame must be proper SO(3)",
        )
        _require(
            wrench[0] >= 0 and not np.any(wrench[dim:]), "invalid contact wrench dimension or load"
        )
        _require(active == 1 or not np.any(wrench), "inactive contact carries a wrench")
        seen_active += active
        if body not in (body1, body2):
            continue
        sign = 1.0 if body == body2 else -1.0
        other_geom, other_body = (geom1, body1) if body == body2 else (geom2, body2)
        point = values[1:4]
        with np.errstate(over="ignore", invalid="ignore"):
            force = sign * (frame.T @ wrench[:3])
            couple = sign * (frame.T @ wrench[3:])
            moment = couple + np.cross(point - com, force)
        _require(
            np.isfinite(force).all() and np.isfinite(couple).all() and np.isfinite(moment).all(),
            "nonfinite derived contact wrench",
        )
        member = other_geom in grouped
        with np.errstate(over="ignore", invalid="ignore"):
            if member:
                group_force += force
                group_moment += moment
                loads[grouped[other_geom]] += wrench[0]
            else:
                other_force += force
                other_moment += moment
        records.append(
            ObjectContactWrenchRecord(
                index,
                bool(active),
                other_geom,
                other_body,
                _tuple3(point),
                _tuple3(force),
                _tuple3(couple),
                _tuple3(moment),
                float(wrench[0]),
                member,
            )
        )
    _require(seen_active == active_count, "active contact count differs")
    with np.errstate(over="ignore", invalid="ignore"):
        total_force, total_moment = group_force + other_force, group_moment + other_moment
    _require(
        all(
            np.isfinite(value).all()
            for value in (
                total_force,
                total_moment,
                group_force,
                group_moment,
                other_force,
                other_moment,
                loads,
            )
        ),
        "nonfinite aggregate contact wrench",
    )
    return ObjectContactWrenchMeasurement(
        time_s,
        before,
        after,
        _tuple3(com),
        tuple(records),
        _tuple3(total_force),
        _tuple3(total_moment),
        _tuple3(group_force),
        _tuple3(group_moment),
        _tuple3(other_force),
        _tuple3(other_moment),
        tuple(float(load) for load in loads),
    )

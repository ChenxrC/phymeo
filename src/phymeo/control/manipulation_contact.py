"""Measure finger slip from public accepted-entry observations only.

The caller supplies one coherent workspace readback and its model-bound body and
geom identities. No physical state, native model or runtime internals are read.
This stateless measurement does not infer contact where the entry has none.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError("finger slip: " + message)


def _index(value, name):
    _require(
        isinstance(value, Integral) and not isinstance(value, bool) and value >= 0,
        name + " must be a nonnegative integer",
    )
    return int(value)


def _array(arrays, name, shape, kind):
    _require(name in arrays, "missing " + name)
    value = arrays[name]
    _require(
        isinstance(value, np.ndarray)
        and value.ndim == len(shape)
        and all(want is None or size == want for size, want in zip(value.shape, shape, strict=True))
        and value.dtype.kind in kind,
        "invalid shape or dtype for " + name,
    )
    return value


def _reject_area_entry(arrays, row, worlds):
    """Keep legacy native-only diagnostics from silently omitting area loads."""
    fields = {"area_meta", "area_ids", "area_data", "area_measure_words"}
    if "area_declared_mask" not in arrays:
        _require(not fields.intersection(arrays), "area fields require a declaration mask")
        return
    declared = _array(arrays, "area_declared_mask", (worlds, 1), "iu")
    _require(np.isin(declared, (0, 1)).all(), "invalid area declaration mask")
    _require(
        declared[row, 0] == 0,
        "native-only diagnostic cannot consume declared area contact; use BodyContactWrenchReader",
    )


@dataclass(frozen=True)
class FingerSlipMeasurement:
    time_s: float
    input_tick: int
    output_tick: int
    normal_loads: tuple[float, float]
    slip_speeds: tuple[float, float]


def measure_finger_slip(arrays, row, *, object_body, pad_geom_ids):
    """Sum normal loads and take each pad's largest contact-point tangent speed.

    All three body kinematic fields must be the entry fields, measured in world
    axes at body origins. The normal is the first row of the contact frame.
    Contact order does not affect speed; normal load is contact_data[..., 13].
    Compensated input time is clock[0] - clock[1], not their sum.

    The entry_body_mask is required to reject heterogeneous padding bodies.
    An invalid entry returns None without reading any old contact/body/clock
    payload. A valid entry with no matching active contacts returns zeros.
    Frame freshness and model identity remain the caller's workspace contract.
    No arrays or external state are changed, including on validation failure.
    """
    _require(isinstance(arrays, Mapping), "arrays must be a public NumPy mapping")
    selected = _index(row, "row")
    body = _index(object_body, "object_body")
    _require(
        isinstance(pad_geom_ids, tuple) and len(pad_geom_ids) == 2, "two pad geom IDs required"
    )
    pads = tuple(_index(value, "pad geom") for value in pad_geom_ids)
    _require(pads[0] != pads[1], "pad geom IDs must differ")

    metadata = _array(arrays, "entry_meta", (None, 6), "iu")
    worlds = metadata.shape[0]
    _require(selected < worlds, "row is outside the world array")
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
    data = _array(arrays, "entry_contact_data", (worlds, capacity, 20), "f")
    positions = _array(arrays, "entry_body_position", (worlds, None, 3), "f")
    bodies = positions.shape[1]
    linear = _array(arrays, "entry_body_linear_velocity", (worlds, bodies, 3), "f")
    angular = _array(arrays, "entry_body_angular_velocity", (worlds, bodies, 3), "f")
    mask = _array(arrays, "entry_body_mask", (worlds, bodies), "iu")
    _require(capacity >= 1 and bodies >= 1, "entry storage cannot have zero capacity")
    _require(0 <= active_count <= count <= capacity, "contact counts exceed storage")
    _require(body < bodies and mask[selected, body] == 1, "object body is invalid or padded")
    _require(np.all((mask[selected] == 0) | (mask[selected] == 1)), "invalid body mask")
    for value in (clocks[selected], positions[selected], linear[selected], angular[selected]):
        _require(np.isfinite(value).all(), "nonfinite entry clock or body kinematics")
    stamp = clocks[selected]
    time_s = float(stamp[0]) - float(stamp[1])
    output_time = float(stamp[2]) - float(stamp[3])
    _require(
        isfinite(time_s) and isfinite(output_time) and 0 <= time_s < output_time,
        "entry clock does not advance from input to output",
    )

    contacts, payload = ids[selected], data[selected]
    _require(np.all(contacts[count:, :2] == 0), "contact flags extend beyond detected count")
    _require(np.isfinite(payload).all(), "nonfinite contact data")
    loads, speeds = [0.0, 0.0], [0.0, 0.0]
    seen_active, geom_bodies = 0, {}
    for contact in range(count):
        present, active, geom1, geom2, body1, body2, feature, dim = map(int, contacts[contact])
        _require(present == 1 and active in (0, 1), "invalid contact prefix flags")
        _require(
            geom1 >= 0 and geom2 >= 0 and geom1 != geom2 and feature >= 0 and dim in (1, 3, 4, 6),
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
            _require(geom not in pads or owner != body, "pad geom belongs to the object")
        values = payload[contact].astype(np.float64, copy=True)
        wrench = values[13:19]
        _require(wrench[0] >= 0, "negative contact normal load")
        _require(not np.any(wrench[dim:]), "wrench exceeds contact dimension")
        _require(active == 1 or not np.any(wrench), "inactive contact carries a wrench")
        normal = values[4:7]
        normal_length = float(np.linalg.norm(normal))
        _require(
            isfinite(normal_length) and abs(normal_length - 1.0) <= 1e-4,
            "contact normal is not unit length",
        )
        normal = normal / normal_length
        seen_active += active
        if active == 0:
            continue
        for finger, pad in enumerate(pads):
            pad_body = None
            if body1 == body and geom2 == pad:
                pad_body = body2
            elif body2 == body and geom1 == pad:
                pad_body = body1
            if pad_body is None:
                continue
            point = values[1:4]
            with np.errstate(over="ignore", invalid="ignore"):
                relative = (
                    linear[selected, body].astype(np.float64)
                    + np.cross(
                        angular[selected, body].astype(np.float64),
                        point - positions[selected, body].astype(np.float64),
                    )
                    - linear[selected, pad_body].astype(np.float64)
                    - np.cross(
                        angular[selected, pad_body].astype(np.float64),
                        point - positions[selected, pad_body].astype(np.float64),
                    )
                )
                tangent = relative - normal * np.dot(normal, relative)
                speed = float(np.linalg.norm(tangent))
            loads[finger] += float(wrench[0])
            _require(isfinite(speed) and isfinite(loads[finger]), "nonfinite derived measurement")
            speeds[finger] = max(speeds[finger], speed)
    _require(seen_active == active_count, "active contact count differs")
    return FingerSlipMeasurement(time_s, before, after, tuple(loads), tuple(speeds))

"""Owned, stateless decoding of public accepted area-pressure entries.

This module reads observations; it does not generate contact or infer missing
loads. Callers obtain one coherent mapping with workspace.readback(frame).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from ..geometry.area_pressure import AreaPressureIdentity, _identity, _stamp, _triangle_samples

FIELDS = ("area_meta", "area_ids", "area_data", "area_measure_words", "area_declared_mask")
IDENTITY_FIELDS = ("compliant_body", "rigid_body", "compliant_material", "rigid_material")
ABI = (
    "mujoco-tyx-area-device-entry-v1",
    "on-compliant-A; B opposite",
    "common-world-origin; area_ids[...,5] validity",
    "little-endian binary64 low32,high32 in i32 words",
)


def _require(ok, message):
    if not ok:
        raise ValueError("area pressure readback: " + message)


def _identity_key(value):
    return tuple(getattr(value, name) for name in IDENTITY_FIELDS)


def _validate_moments(flags, values):
    # Same published FP32 representation contract as the device export.
    with np.errstate(over="ignore", invalid="ignore"):
        p, f = values[:, :3], values[:, 9:12]
        left = p[:, [1, 2, 0]] * f[:, [2, 0, 1]]
        right = p[:, [2, 0, 1]] * f[:, [1, 2, 0]]
    available = np.all(
        np.isfinite(left)
        & np.isfinite(right)
        & (np.abs(left) <= 2.0**125)
        & (np.abs(right) <= 2.0**125),
        axis=1,
    )
    _require(np.array_equal(flags, available.astype(np.int32)), "area moment validity")
    l64, r64 = left[available].astype(np.float64), right[available].astype(np.float64)
    bound = 8 * np.finfo(np.float32).eps * (np.abs(l64) + np.abs(r64)) + np.finfo(np.float32).tiny
    _require(
        np.all(np.abs(values[available, 12:15] - (l64 - r64)) <= bound),
        "area FP32 moment representation",
    )


@dataclass(frozen=True)
class AreaPressureSample:
    identity: AreaPressureIdentity
    stamp: object
    pair_index: int
    sample_index: int
    sample_id: tuple
    source_id: tuple
    body_a: int
    body_b: int
    point_world: tuple
    normal_world: tuple
    area_m2: float
    stiffness_n_per_m: float
    surrogate_gap_m: float
    normal_force_n: float
    force_on_a_world: tuple
    moment_on_a_world: tuple | None


@dataclass(frozen=True)
class AreaPressureEntryRow:
    samples: tuple
    digest: str


class AreaPressureEntryReader:
    """Bind material and body identity independently of application task roles."""

    def __init__(self, workspace, identities_by_model):
        binding = workspace.binding
        _require(binding.get("area_pressure") is True, "area observation workspace is required")
        _require(
            tuple(
                binding.get(k) for k in ("area_schema", "area_force", "area_moment", "area_measure")
            )
            == ABI,
            "area observation ABI differs",
        )
        pairs = binding["area_pair_names"]
        expected = {key for key, value in pairs.items() if value}
        _require(
            isinstance(identities_by_model, Mapping) and set(identities_by_model) == expected,
            "identities must exactly cover declared area models",
        )
        owned = {}
        for key in expected:
            values = identities_by_model[key]
            _require(
                type(values) is tuple and len(values) == len(pairs[key]),
                "one ordered identity per declared pair",
            )
            owned[key] = tuple(_identity(v) for v in values)
        self.identities = MappingProxyType(owned)
        self.keys = tuple(binding["world_keys"])
        self.frame_binding = binding
        self.capacity = binding["area_capacity"]
        # Array dimensions are public metadata, not runtime scratch.
        self.native_capacity = binding["entry_contact_capacity"]
        self.body_capacity = binding["dimensions"]["nbody"]
        worlds = {}
        for world, key, _, _ in binding["source"]["worlds"]:
            meta = workspace.model_metadata[key]
            names = tuple(meta["body_names"])
            geom_bodies = tuple(int(v) for v in meta["geom_bodyid"])
            rows = []
            for pair, identity in zip(pairs[key], owned.get(key, ()), strict=True):
                _require(
                    tuple(pair) == (identity.compliant_body, identity.rigid_body),
                    "material identity pair differs from model declaration",
                )
                _require(
                    identity.compliant_body in names and identity.rigid_body in names,
                    "area body name absent from model",
                )
                a, b = names.index(identity.compliant_body), names.index(identity.rigid_body)
                _require(a != b, "area pair has one physical body")
                rows.append((identity, a, b))
            worlds[world] = (tuple(rows), geom_bodies, len(names))
        _require(set(worlds) == set(self.keys), "world/model binding differs")
        self.worlds = MappingProxyType(worlds)

    def validate(self, arrays, frame):
        """Return loads and digest additions; do not mutate cursor or physics."""
        n, cap = len(self.keys), self.capacity
        layouts = {
            "area_meta": ((n, 4), np.int32),
            "area_ids": ((n, cap, 6), np.int32),
            "area_data": ((n, cap, 15), np.float32),
            "area_measure_words": ((n, cap, 2), np.int32),
            "area_declared_mask": ((n, 1), np.int32),
        }
        layouts.update(
            {
                "entry_meta": ((n, 6), np.int32),
                "entry_clock": ((n, 4), np.float32),
                "entry_contact_ids": ((n, self.native_capacity, 8), np.int32),
                "entry_body_mask": ((n, self.body_capacity), np.int32),
            }
        )
        _require(isinstance(arrays, Mapping), "public NumPy mapping required")
        _require(
            frame.binding == self.frame_binding and len(frame.ticks) == n,
            "area frame binding differs",
        )
        for name, (shape, dtype) in layouts.items():
            value = arrays.get(name)
            _require(
                type(value) is np.ndarray and value.shape == shape and value.dtype == dtype,
                "area array layout differs: " + name,
            )
        _require(
            isinstance(frame.area_catalogs, Mapping) and set(frame.area_catalogs) == set(self.keys),
            "area catalog world set differs",
        )
        result = {}
        for row, world in enumerate(self.keys):
            configured, geom_bodies, body_count = self.worlds[world]
            catalog = frame.area_catalogs[world]
            _require(type(catalog) is tuple, "area catalog must be ordered")
            meta = arrays["area_meta"][row]
            ids, data, words = (
                arrays[k][row] for k in ("area_ids", "area_data", "area_measure_words")
            )
            valid, count, active, overflow = map(int, meta)
            entry_valid, tick, after, native_count, native_active, native_overflow = map(
                int, arrays["entry_meta"][row]
            )
            _require(entry_valid in (0, 1) and native_overflow == 0, "native validity/overflow")
            _require(
                0 <= after <= 2147483647 and after == frame.ticks[row], "entry/output tick differs"
            )
            _require(
                0 <= native_active <= native_count <= arrays["entry_contact_ids"].shape[1],
                "native counts/capacity",
            )
            expected_mask = np.zeros(arrays["entry_body_mask"].shape[1], dtype=np.int32)
            if entry_valid:
                expected_mask[:body_count] = 1
            _require(
                np.array_equal(arrays["entry_body_mask"][row], expected_mask),
                "entry body mask differs from model owners",
            )
            if entry_valid:
                _require(
                    0 <= tick < 2147483647 and after == tick + 1,
                    "native entry ticks are not adjacent",
                )
                clock = arrays["entry_clock"][row]
                _require(np.isfinite(clock).all(), "nonfinite entry clock")
                start, end = float(clock[0]) - float(clock[1]), float(clock[2]) - float(clock[3])
                _require(0 <= start < end, "entry clock does not advance")
            else:
                _require(tick == native_count == native_active == 0, "invalid empty entry metadata")
            _require(
                int(arrays["area_declared_mask"][row, 0]) == bool(configured),
                "area declaration mask",
            )
            _require(np.isfinite(data).all(), "nonfinite area payload")
            samples, canonical = [], []
            if not configured or entry_valid == 0:
                _require(
                    not catalog and not any(np.any(v) for v in (meta, ids, data, words)),
                    "unavailable area record contains history",
                )
            else:
                _require(
                    entry_valid == valid == 1 and overflow == 0 and after == tick + 1,
                    "area/native accepted phases differ",
                )
                _require(0 <= active <= count <= cap, "area counts/capacity")
                _require(len(catalog) == len(configured), "area catalog pair count")
                _require(not any(np.any(v[count:]) for v in (ids, data, words)), "area padding")
                _require(
                    active == int(np.count_nonzero(data[:count, 8] > 0)),
                    "positive-force sample count",
                )
                stamp_parts = arrays["entry_clock"][row]
                time_s = float(stamp_parts[0]) - float(stamp_parts[1])
                offset = 0
                for ordinal, (item, (expected_identity, body_a, body_b)) in enumerate(
                    zip(catalog, configured, strict=True)
                ):
                    _require(
                        isinstance(item, Mapping)
                        and set(item) == {"identity", "stamp", "sample_ids", "source_ids"},
                        "area catalog fields",
                    )
                    identity, stamp = _identity(item["identity"]), _stamp(item["stamp"])
                    _require(
                        identity == expected_identity, "area material identity differs from role"
                    )
                    _require(
                        stamp.world_id == str(world)
                        and stamp.step_index == tick
                        and stamp.sample_time_s == time_s,
                        "area catalog historical input stamp differs",
                    )
                    sample_ids, source_ids = _triangle_samples(
                        item["sample_ids"], item["source_ids"]
                    )
                    stop = offset + len(sample_ids)
                    _require(stop <= count, "area catalog exceeds sample prefix")
                    block, values, measures = (
                        ids[offset:stop],
                        data[offset:stop],
                        words[offset:stop],
                    )
                    expected_ids = np.zeros((stop - offset, 5), dtype=np.int32)
                    expected_ids[:, 0] = 1
                    expected_ids[:, 1] = ordinal
                    expected_ids[:, 2] = np.arange(offset, stop)
                    expected_ids[:, 3:5] = (body_a, body_b)
                    _require(
                        np.array_equal(block[:, :5], expected_ids), "area sample/body/pair order"
                    )
                    _require(np.isin(block[:, 5], (0, 1)).all(), "area moment validity")
                    _require(
                        not np.any(values[block[:, 5] == 0, 12:15]), "unavailable moment sentinel"
                    )
                    d_area = np.asarray(measures, dtype="<i4").tobytes()
                    d_area = np.frombuffer(d_area, dtype="<f8")
                    _require(
                        np.isfinite(d_area).all() and np.all(d_area > 0),
                        "positive finite true area",
                    )
                    _require(
                        np.all(values[:, 6] > 0) and np.all(values[:, 8] >= 0),
                        "area stiffness/load",
                    )
                    _require(
                        np.allclose(
                            np.sum(values[:, 3:6].astype(np.float64) ** 2, axis=1),
                            1,
                            rtol=3e-5,
                            atol=2e-6,
                        ),
                        "area unit normal",
                    )
                    _require(
                        np.allclose(
                            values[:, 9:12],
                            values[:, 3:6].astype(np.float64) * values[:, 8:9],
                            rtol=3e-5,
                            atol=2e-6,
                        ),
                        "area force direction",
                    )
                    _validate_moments(block[:, 5], values)
                    for local, (sample_id, source_id) in enumerate(
                        zip(sample_ids, source_ids, strict=True)
                    ):
                        value = values[local]
                        samples.append(
                            AreaPressureSample(
                                identity,
                                stamp,
                                ordinal,
                                offset + local,
                                sample_id,
                                source_id,
                                body_a,
                                body_b,
                                tuple(map(float, value[:3])),
                                tuple(map(float, value[3:6])),
                                float(d_area[local]),
                                float(value[6]),
                                float(value[7]),
                                float(value[8]),
                                tuple(map(float, value[9:12])),
                                tuple(map(float, value[12:15])) if block[local, 5] else None,
                            )
                        )
                    canonical.append(
                        (
                            _identity_key(identity),
                            (
                                stamp.world_id,
                                stamp.step_index,
                                stamp.sample_time_s,
                                stamp.configuration_id,
                            ),
                            sample_ids,
                            source_ids,
                        )
                    )
                    offset = stop
                _require(offset == count, "area catalog does not cover sample prefix")
            _require(
                0 <= native_count <= arrays["entry_contact_ids"].shape[1], "native contact capacity"
            )
            if entry_valid:
                prefix = arrays["entry_contact_ids"][row, :native_count]
                _require(
                    np.isin(prefix[:, 1], (0, 1)).all()
                    and int(np.sum(prefix[:, 1], dtype=np.int64)) == native_active,
                    "native active count/flags",
                )
                _require(
                    not np.any(arrays["entry_contact_ids"][row, native_count:]),
                    "native identity padding",
                )
            declared_bodies = {frozenset((a, b)) for _, a, b in configured}
            for item in arrays["entry_contact_ids"][row, :native_count]:
                _require(item[0] == 1, "native contact prefix")
                first, second = map(int, item[2:4])
                _require(
                    0 <= first < len(geom_bodies) and 0 <= second < len(geom_bodies),
                    "native contact geometry identity",
                )
                _require(
                    tuple(map(int, item[4:6])) == (geom_bodies[first], geom_bodies[second]),
                    "native geometry/body owner mismatch",
                )
                _require(
                    frozenset((geom_bodies[first], geom_bodies[second])) not in declared_bodies,
                    "native contact duplicates a declared area body pair",
                )
            digest = sha256()
            for name in FIELDS:
                digest.update(name.encode("ascii"))
                digest.update(arrays[name][row].tobytes())
            digest.update(
                json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            )
            result[world] = AreaPressureEntryRow(tuple(samples), digest.hexdigest())
        return result

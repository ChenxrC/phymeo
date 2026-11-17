"""Explicit role and history validation for accepted area-pressure observations.

This HOST evaluator consumes solved forces in newtons, not a new contact model.
Native contact and area sample identities remain separate.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

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
        raise ValueError("manipulation area entry: " + message)


def _identity_key(value):
    return tuple(getattr(value, name) for name in IDENTITY_FIELDS)


@dataclass(frozen=True)
class AreaPressureRoleBinding:
    identity: AreaPressureIdentity
    object_side: str | None
    role: str

    def __post_init__(self):
        object.__setattr__(self, "identity", _identity(self.identity))
        _require(
            self.object_side is None
            or (type(self.object_side) is str and self.object_side in ("A", "B")),
            "object_side must be A/B/None",
        )
        _require(
            type(self.role) is str and self.role in ("finger0", "finger1", "support", "other"),
            "unknown role",
        )
        _require(
            self.object_side is not None or self.role == "other", "absent object has no load role"
        )


class AreaSuccessAdapter:
    """Immutable declarations plus stateless complete-frame validation."""

    def __init__(self, workspace, roles, addresses):
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
            isinstance(roles, Mapping) and set(roles) == expected,
            "roles must exactly cover area models",
        )
        roles = {key: roles[key] for key in expected}
        self.keys = tuple(binding["world_keys"])
        self.capacity = binding["area_capacity"]
        self.worlds = {}
        signature = []
        for world, key, _, _ in binding["source"]["worlds"]:
            meta = workspace.model_metadata[key]
            geom_bodies = tuple(int(v) for v in meta["geom_bodyid"])
            object_body, _, _, fingers, support = addresses[world]
            declared = tuple(pairs[key])
            configured = roles.get(key, ())
            _require(
                type(configured) is tuple and len(configured) == len(declared),
                "one ordered role per declared pair is required",
            )
            body_names = tuple(meta["body_names"])
            role_bodies = {f"finger{i}": geom_bodies[g] for i, g in enumerate(fingers)}
            role_bodies["support"] = geom_bodies[support]
            checked = []
            for pair, role in zip(declared, configured, strict=True):
                _require(type(role) is AreaPressureRoleBinding, "role binding type")
                role = AreaPressureRoleBinding(role.identity, role.object_side, role.role)
                identity = role.identity
                _require(
                    tuple(pair) == (identity.compliant_body, identity.rigid_body),
                    "role pair differs from model declaration",
                )
                body_a = body_names.index(identity.compliant_body)
                body_b = body_names.index(identity.rigid_body)
                actual_side = (
                    "A" if body_a == object_body else "B" if body_b == object_body else None
                )
                _require(
                    role.object_side == actual_side, "object side differs from actual Scene body"
                )
                other = body_b if actual_side == "A" else body_a
                named = (
                    [name for name, body in role_bodies.items() if body == other]
                    if actual_side
                    else []
                )
                _require(
                    not named or (len(named) == 1 and role.role == named[0]),
                    "named counterpart must use its unambiguous Scene role",
                )
                if role.role != "other":
                    other = body_b if actual_side == "A" else body_a
                    _require(
                        role.role in role_bodies and other == role_bodies[role.role],
                        "role counterpart differs from named Scene geometry owner",
                    )
                    _require(
                        list(role_bodies.values()).count(other) == 1,
                        "shared body makes the declared role ambiguous",
                    )
                checked.append((role, body_a, body_b))
            self.worlds[world] = (tuple(checked), geom_bodies)
            signature.append(
                (
                    world,
                    key,
                    tuple(
                        (_identity_key(r.identity), r.object_side, r.role, a, b)
                        for r, a, b in checked
                    ),
                )
            )
        self.binding = ("manipulation-area-success-v1", ABI, tuple(signature))

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
            configured, geom_bodies = self.worlds[world]
            catalog = frame.area_catalogs[world]
            _require(type(catalog) is tuple, "area catalog must be ordered")
            meta = arrays["area_meta"][row]
            ids, data, words = (
                arrays[k][row] for k in ("area_ids", "area_data", "area_measure_words")
            )
            valid, count, active, overflow = map(int, meta)
            entry_valid, tick, after, native_count, *_ = map(int, arrays["entry_meta"][row])
            _require(
                int(arrays["area_declared_mask"][row, 0]) == bool(configured),
                "area declaration mask",
            )
            _require(np.isfinite(data).all(), "nonfinite area payload")
            loads, support_z, canonical = [0.0, 0.0], 0.0, []
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
                for ordinal, (item, (role, body_a, body_b)) in enumerate(
                    zip(catalog, configured, strict=True)
                ):
                    _require(
                        isinstance(item, Mapping)
                        and set(item) == {"identity", "stamp", "sample_ids", "source_ids"},
                        "area catalog fields",
                    )
                    identity, stamp = _identity(item["identity"]), _stamp(item["stamp"])
                    _require(identity == role.identity, "area material identity differs from role")
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
                    if role.role.startswith("finger"):
                        loads[int(role.role[-1])] += float(np.sum(values[:, 8], dtype=np.float64))
                    elif role.role == "support":
                        sign = 1 if role.object_side == "A" else -1
                        support_z += sign * float(np.sum(values[:, 11], dtype=np.float64))
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
            declared_bodies = {frozenset((a, b)) for _, a, b in configured}
            for item in arrays["entry_contact_ids"][row, :native_count]:
                _require(item[0] == 1, "native contact prefix")
                first, second = map(int, item[2:4])
                _require(
                    0 <= first < len(geom_bodies) and 0 <= second < len(geom_bodies),
                    "native contact geometry identity",
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
            result[world] = (tuple(loads), support_z, digest.hexdigest())
        return result

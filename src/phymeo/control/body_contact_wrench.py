"""Contact-only body wrenches from one public accepted-input observation.

No dynamics, task success or missing force is inferred. The caller binds the
explicit reference point to this accepted input; it is not automatically a COM.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..backends.area_friction_readback import AreaFrictionEntryReader
from ..backends.area_pressure_readback import AreaPressureEntryReader
from .manipulation_wrench import measure_object_contact_wrench


def _require(ok, message):
    if not ok:
        raise ValueError("body contact wrench: " + message)


def _vector(value):
    return tuple(map(float, value))


MAX_BODY_WRENCH_REQUESTS = 1024


@dataclass(frozen=True)
class BodyContactWrenchRequest:
    """Owned named query at an explicit accepted-input world reference point."""

    world_key: int
    body_name: str
    reference_point_world: tuple
    group_body_names: tuple = ()

    def __post_init__(self):
        _require(type(self.world_key) is int, "unknown world key")
        _require(type(self.body_name) is str and bool(self.body_name), "unknown body name")
        group = self.group_body_names
        _require(
            type(group) is tuple
            and len(group) <= 128
            and all(type(v) is str and bool(v) for v in group)
            and len(set(group)) == len(group)
            and self.body_name not in group,
            "invalid counterpart body group",
        )
        raw = np.asarray(self.reference_point_world)
        _require(
            raw.shape == (3,) and raw.dtype.kind in "fiu" and np.isfinite(raw).all(),
            "reference point must be a finite numeric 3-vector",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            reference = np.array(raw, dtype=np.float64, copy=True)
        _require(np.isfinite(reference).all(), "reference point overflow")
        object.__setattr__(self, "reference_point_world", _vector(reference))
        object.__setattr__(self, "group_body_names", tuple(group))


@dataclass(frozen=True)
class BodyContactWrenchRecord:
    source: str
    index: int
    other_body: int
    other_geom: int | None
    identity: object
    sample_id: tuple | None
    source_id: tuple | None
    active: bool
    point_world: tuple
    force_world: tuple
    couple_world: tuple
    moment_reference_world: tuple
    normal_load: float
    area_m2: float | None
    group_member: bool


@dataclass(frozen=True)
class BodyContactWrenchMeasurement:
    time_s: float
    input_tick: int
    output_tick: int
    reference_point_world: tuple
    contacts: tuple
    native_force_world: tuple
    native_moment_world: tuple
    area_force_world: tuple
    area_moment_world: tuple
    total_force_world: tuple
    total_moment_world: tuple
    group_force_world: tuple
    group_moment_world: tuple
    other_force_world: tuple
    other_moment_world: tuple
    group_normal_loads: tuple


class BodyContactWrenchReader:
    """Stateless HOST diagnostic borrowing an observation workspace.

    Area identities must cover every declared model/pair in order. The native
    and area paths retain distinct records; no area sample gets a fake geom ID.
    Force and moment exclude gravity, actuator, damping and external loads.
    """

    def __init__(self, workspace, *, area_identities=None):
        binding = workspace.binding
        _require(
            binding.get("contact_tokens") is True
            and binding.get("entry_body_kinematics") is True
            and binding.get("entry_phase") == "accepted-step-input",
            "accepted-entry contact workspace required",
        )
        _require(
            binding.get("contact_frame") == "rows-of-contact-axes-in-world"
            and binding.get("contact_wrench_world") == "on-geometry2-at-contact-point:force,couple",
            "native contact ABI differs",
        )
        self._workspace = workspace
        self._keys = tuple(binding["world_keys"])
        has_area = any(row.get("area_pressure_pairs") for row in binding["source"]["buckets"])
        _require(
            not has_area or binding.get("area_pressure") is True,
            "area-declared model requires an area observation workspace",
        )
        self._area = None
        if binding.get("area_pressure"):
            schema = binding.get("area_schema")
            if schema == AreaFrictionEntryReader.ABI[0]:
                reader = AreaFrictionEntryReader
            else:
                _require(
                    schema == AreaPressureEntryReader.ABI[0]
                    and not any(row.get("area_friction") for row in binding["source"]["buckets"]),
                    "area observation ABI differs from declared model laws",
                )
                reader = AreaPressureEntryReader
            self._area = reader(
                workspace, {} if not has_area and area_identities is None else area_identities
            )
        else:
            _require(area_identities is None, "native workspace cannot receive area identities")
        worlds = {}
        for world, model, _, _ in binding["source"]["worlds"]:
            meta = workspace.model_metadata[model]
            worlds[world] = (
                tuple(meta["body_names"]),
                tuple(int(v) for v in meta["geom_bodyid"]),
            )
        _require(set(worlds) == set(self._keys), "world metadata differs")
        self._worlds = MappingProxyType(worlds)

    def _prepare(self, request):
        _require(type(request) is BodyContactWrenchRequest, "expected BodyContactWrenchRequest")
        # Revalidate even a frozen instance: no public attribute is an oracle.
        request = BodyContactWrenchRequest(
            request.world_key,
            request.body_name,
            request.reference_point_world,
            request.group_body_names,
        )
        world_key = request.world_key
        _require(world_key in self._worlds, "unknown world key")
        names, geom_bodies = self._worlds[world_key]
        _require(names.count(request.body_name) == 1, "unknown body name")
        _require(
            all(names.count(v) == 1 for v in request.group_body_names),
            "invalid counterpart body group",
        )
        return (
            world_key,
            names,
            geom_bodies,
            names.index(request.body_name),
            tuple(names.index(v) for v in request.group_body_names),
            np.asarray(request.reference_point_world, dtype=np.float64),
        )

    def measure(self, frame, *, world_key, body_name, reference_point_world, group_body_names=()):
        request = BodyContactWrenchRequest(
            world_key, body_name, reference_point_world, group_body_names
        )
        return self.measure_many(frame, (request,))[0]

    def measure_many(self, frame, requests):
        """Return ordered complete results from one fresh public readback.

        All queries are checked before D2H. Repeated queries are allowed. No
        intermediate result or cross-frame cache is published on failure.
        """
        _require(
            type(requests) is tuple and 1 <= len(requests) <= MAX_BODY_WRENCH_REQUESTS,
            "requests must contain 1..1024 immutable body queries",
        )
        prepared = tuple(self._prepare(request) for request in requests)
        arrays = self._workspace.readback(frame)
        _require(isinstance(arrays, Mapping), "owned public arrays required")
        area_rows = self._area.validate(arrays, frame) if self._area is not None else None
        # This native partition is consumed only after complete area validation.
        native_arrays = {k: v for k, v in arrays.items() if not k.startswith("area_")}
        return tuple(
            self._measure(frame, query, arrays, native_arrays, area_rows) for query in prepared
        )

    def _measure(self, frame, query, arrays, native_arrays, area_rows):
        world_key, names, geom_bodies, body, group, reference = query
        row = self._keys.index(world_key)
        native = measure_object_contact_wrench(
            native_arrays,
            row,
            object_body=body,
            object_com_world=reference,
        )
        _require(int(arrays["entry_meta"][row, 2]) == frame.ticks[row], "entry/output tick differs")
        if native is None:
            return None
        mask = arrays["entry_body_mask"][row]
        expected = np.zeros(len(mask), dtype=mask.dtype)
        expected[: len(names)] = 1
        _require(np.array_equal(mask, expected), "body mask differs from model")
        count = int(arrays["entry_meta"][row, 3])
        for item in arrays["entry_contact_ids"][row, :count]:
            a, b = map(int, item[2:4])
            _require(
                0 <= a < len(geom_bodies)
                and 0 <= b < len(geom_bodies)
                and tuple(map(int, item[4:6])) == (geom_bodies[a], geom_bodies[b]),
                "native geometry/body owner differs",
            )
        records = []
        for contact in native.contacts:
            records.append(
                BodyContactWrenchRecord(
                    "native",
                    contact.index,
                    contact.other_body,
                    contact.other_geom,
                    None,
                    None,
                    None,
                    contact.active,
                    contact.point_world,
                    contact.force_world,
                    contact.couple_world,
                    contact.moment_com_world,
                    contact.normal_load,
                    None,
                    contact.other_body in group,
                )
            )
        if area_rows is not None:
            for sample in area_rows[world_key].samples:
                if body not in (sample.body_a, sample.body_b):
                    continue
                _require(
                    sample.moment_on_a_world is not None,
                    "published area moment unavailable for selected body",
                )
                sign = 1 if body == sample.body_a else -1
                other = sample.body_b if sign == 1 else sample.body_a
                force = sign * np.asarray(sample.force_on_a_world, dtype=np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    moment = sign * np.asarray(sample.moment_on_a_world) - np.cross(
                        reference, force
                    )
                _require(np.isfinite(moment).all(), "translated area moment overflow")
                records.append(
                    BodyContactWrenchRecord(
                        "area",
                        sample.sample_index,
                        other,
                        None,
                        sample.identity,
                        sample.sample_id,
                        sample.source_id,
                        sample.normal_force_n > 0,
                        sample.point_world,
                        _vector(force),
                        (0.0, 0.0, 0.0),
                        _vector(moment),
                        sample.normal_force_n,
                        sample.area_m2,
                        other in group,
                    )
                )
        native_f, native_m, area_f, area_m = (np.zeros(3) for _ in range(4))
        group_f, group_m, other_f, other_m = (np.zeros(3) for _ in range(4))
        loads = np.zeros(len(group))
        with np.errstate(over="ignore", invalid="ignore"):
            for record in records:
                f, m = record.force_world, record.moment_reference_world
                if record.source == "native":
                    native_f += f
                    native_m += m
                else:
                    area_f += f
                    area_m += m
                if record.group_member:
                    group_f += f
                    group_m += m
                    loads[group.index(record.other_body)] += record.normal_load
                else:
                    other_f += f
                    other_m += m
            total_f, total_m = native_f + area_f, native_m + area_m
        _require(
            all(
                np.isfinite(v).all()
                for v in (
                    native_f,
                    native_m,
                    area_f,
                    area_m,
                    total_f,
                    total_m,
                    group_f,
                    group_m,
                    other_f,
                    other_m,
                    loads,
                )
            ),
            "aggregate wrench overflow",
        )
        return BodyContactWrenchMeasurement(
            native.time_s,
            native.input_tick,
            native.output_tick,
            _vector(reference),
            tuple(records),
            _vector(native_f),
            _vector(native_m),
            _vector(area_f),
            _vector(area_m),
            _vector(total_f),
            _vector(total_m),
            _vector(group_f),
            _vector(group_m),
            _vector(other_f),
            _vector(other_m),
            tuple(map(float, loads)),
        )

"""Independent HOST wrench algebra plus one actual CUDA area/native lifecycle.

Hand-authored public array packets below are protocol/algebra fixtures, never
CUDA evidence. The separately marked test obtains all forces from actual steps.
"""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from phymeo.backends.area_pressure_readback import AreaPressureEntryReader
from phymeo.control.body_contact_wrench import BodyContactWrenchReader
from phymeo.control.manipulation_contact import measure_finger_slip
from phymeo.control.manipulation_wrench import measure_object_contact_wrench
from phymeo.geometry.area_pressure import AreaPressureIdentity, AreaPressureStamp

KEYS = (13, 41)
BODY_NAMES = ("world", "object", "finger", "support", "bystander")
GEOM_NAMES = ("object-geom", "finger-geom", "support-geom", "bystander-geom")
IDENTITY = AreaPressureIdentity("object", "support", "hand-pressure", "hand-rigid")
ORIGIN = (0.0, 0.0, 0.0)
REFERENCE = (0.2, -0.3, 0.4)
AREA_CAPACITY = 13


def _owned(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


class HandWorkspace:
    """Only a public readback protocol; no simulated runtime or native model."""

    def __init__(self, arrays, catalogs, *, area=True):
        self.binding = MappingProxyType(
            dict(
                world_keys=KEYS,
                contact_tokens=True,
                entry_body_kinematics=True,
                entry_phase="accepted-step-input",
                contact_frame="rows-of-contact-axes-in-world",
                contact_wrench_world="on-geometry2-at-contact-point:force,couple",
                area_pressure=area,
                area_capacity=AREA_CAPACITY if area else 0,
                entry_contact_capacity=3,
                dimensions=MappingProxyType({"nbody": 5}),
                area_pair_names=MappingProxyType(
                    {"model": (("object", "support"),) if area else ()}
                ),
                area_schema="mujoco-tyx-area-device-entry-v1" if area else None,
                area_force="on-compliant-A; B opposite" if area else None,
                area_moment="common-world-origin; area_ids[...,5] validity" if area else None,
                area_measure="little-endian binary64 low32,high32 in i32 words" if area else None,
                source=MappingProxyType(
                    dict(
                        worlds=((13, "model", 0, 0), (41, "model", 0, 1)),
                        buckets=(
                            MappingProxyType(
                                {"area_pressure_pairs": (("object", "support"),) if area else ()}
                            ),
                        ),
                    )
                ),
            )
        )
        self.model_metadata = MappingProxyType(
            {
                "model": MappingProxyType(
                    dict(
                        body_names=BODY_NAMES,
                        geom_names=GEOM_NAMES,
                        geom_bodyid=_owned(np.array((1, 2, 3, 4), np.int32)),
                    )
                )
            }
        )
        self.arrays = MappingProxyType({name: _owned(value) for name, value in arrays.items()})
        self.frame = SimpleNamespace(
            binding=self.binding,
            ticks=tuple(int(v) for v in arrays["entry_meta"][:, 2]),
            area_catalogs=MappingProxyType(catalogs),
        )
        self.reads = 0

    def readback(self, frame):
        assert frame is self.frame
        self.reads += 1
        return self.arrays


def _points(split):
    # A true .4-by-.1 triangle in y=0, with area .02 and centroid (-1,0,2).
    vertices = np.array(((-1.2, 0, 2 + 1 / 30), (-0.8, 0, 2 + 1 / 30), (-1, 0, 2 - 1 / 15)))
    triangles = (vertices,)
    if split:
        a, b, c = vertices
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        triangles = (
            np.array((a, ab, ca)),
            np.array((ab, b, bc)),
            np.array((ca, bc, c)),
            np.array((ab, bc, ca)),
        )
    barycentric = np.array(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))
    return np.vstack(tuple(barycentric @ triangle for triangle in triangles))


def _packet(*, area=True, split=False, empty=False, invalid=False):
    arrays = dict(
        entry_meta=np.tile(np.array((1, 4, 5, 2, 1, 0), np.int32), (2, 1)),
        entry_clock=np.tile(np.array((0.5, 0.125, 0.625, 0.125), np.float32), (2, 1)),
        entry_contact_ids=np.zeros((2, 3, 8), np.int32),
        entry_contact_data=np.zeros((2, 3, 20), np.float32),
        entry_contact_wrench_world=np.zeros((2, 3, 6), np.float32),
        entry_contact_body_wrench=np.zeros((2, 3, 2, 6), np.float32),
        entry_body_mask=np.ones((2, 5), np.int32),
        entry_body_position=np.zeros((2, 5, 3), np.float32),
        entry_body_linear_velocity=np.zeros((2, 5, 3), np.float32),
        entry_body_angular_velocity=np.zeros((2, 5, 3), np.float32),
    )
    arrays["entry_contact_ids"][:, 0] = (1, 1, 1, 0, 2, 1, 0, 6)
    arrays["entry_contact_ids"][:, 1] = (1, 0, 0, 3, 1, 4, 0, 6)
    data = arrays["entry_contact_data"]
    data[:, 0, 1:4] = (1, 2, 3)
    data[:, 1, 1:4] = (-3, 1, 0)
    data[:, :2, 4:13] = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0))).ravel()
    data[:, 0, 13:19] = (4, 1, 2, 0.75, 0.5, -0.25)
    arrays["entry_contact_wrench_world"][:, 0] = (1, 2, 4, 0.5, -0.25, 0.75)
    arrays["entry_contact_body_wrench"][:, 0, 0] = (-1, -2, -4, -2.5, 1.25, -0.75)
    arrays["entry_contact_body_wrench"][:, 0, 1] = (1, 2, 4, 2.5, -1.25, 0.75)
    catalogs = dict.fromkeys(KEYS, ())
    if area:
        points = _points(split).astype(np.float32)
        count = len(points)
        arrays.update(
            area_meta=np.tile(np.array((1, count, count, 0), np.int32), (2, 1)),
            area_ids=np.zeros((2, AREA_CAPACITY, 6), np.int32),
            area_data=np.zeros((2, AREA_CAPACITY, 15), np.float32),
            area_measure_words=np.zeros((2, AREA_CAPACITY, 2), np.int32),
            area_declared_mask=np.ones((2, 1), np.int32),
        )
        ids, values = arrays["area_ids"], arrays["area_data"]
        ids[:, :count, 0] = 1
        ids[:, :count, 2] = np.arange(count)
        ids[:, :count, 3:6] = (1, 3, 1)
        values[:, :count, :3] = points
        values[:, :count, 3:6] = (0, 1, 0)
        values[:, :count, 6:9] = (1000 * 0.02 / count, -0.15, 3 / count)
        values[:, :count, 9:12] = (0, 3 / count, 0)
        values[:, :count, 12:15] = np.cross(points, values[0, :count, 9:12])
        words = np.array([0.02 / count], dtype="<f8").view("<i4")
        arrays["area_measure_words"][:, :count] = words
        for key in KEYS:
            catalogs[key] = (
                MappingProxyType(
                    dict(
                        identity=IDENTITY,
                        stamp=AreaPressureStamp(str(key), 4, 0.375, "hand-configuration"),
                        sample_ids=tuple((f"triangle-{i // 3}", i % 3) for i in range(count)),
                        source_ids=(("cell", "face"),) * count,
                    )
                ),
            )
    if empty or invalid:
        arrays["entry_meta"][:, 3:5] = 0
        for name, value in arrays.items():
            if name.startswith("entry_contact_"):
                value.fill(0)
        if area:
            arrays["area_meta"][:, 1:] = 0
            for name in ("area_ids", "area_data", "area_measure_words"):
                arrays[name].fill(0)
            catalogs = {
                key: (MappingProxyType({**dict(items[0]), "sample_ids": (), "source_ids": ()}),)
                for key, items in catalogs.items()
            }
    if invalid:
        arrays["entry_meta"].fill(0)
        arrays["entry_clock"].fill(0)
        arrays["entry_body_mask"].fill(0)
        if area:
            arrays["area_meta"].fill(0)
        catalogs = dict.fromkeys(KEYS, ())
    return HandWorkspace(arrays, catalogs, area=area)


def _reader(ws):
    return BodyContactWrenchReader(
        ws, area_identities={"model": (IDENTITY,)} if ws.binding["area_pressure"] else None
    )


def _measure(reader, ws, *, world=13, body="object", reference=ORIGIN, groups=("finger",)):
    before = ws.reads
    result = reader.measure(
        ws.frame,
        world_key=world,
        body_name=body,
        reference_point_world=reference,
        group_body_names=groups,
    )
    assert ws.reads == before + 1
    return result


def _assert_vector(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


def _changed(ws, *, arrays=None, catalogs=None):
    arrays = {k: v.copy() for k, v in ws.arrays.items()} if arrays is None else arrays
    catalogs = dict(ws.frame.area_catalogs) if catalogs is None else catalogs
    return HandWorkspace(arrays, catalogs, area=ws.binding["area_pressure"])


def test_hand_native_couple_area_moment_reference_and_explicit_groups():
    ws = _packet()
    reader = _reader(ws)
    saved = {name: value.tobytes() for name, value in ws.arrays.items()}
    result = _measure(reader, ws)
    assert (result.time_s, result.input_tick, result.output_tick) == (0.375, 4, 5)
    assert result.reference_point_world == ORIGIN
    _assert_vector(result.native_force_world, (1, 2, 4))
    _assert_vector(result.native_moment_world, (2.5, -1.25, 0.75))
    _assert_vector(result.area_force_world, (0, 3, 0))
    _assert_vector(result.area_moment_world, (-6, 0, -3))
    _assert_vector(result.total_force_world, (1, 5, 4))
    _assert_vector(result.total_moment_world, (-3.5, -1.25, -2.25))
    _assert_vector(result.group_force_world, (1, 2, 4))
    _assert_vector(result.group_moment_world, (2.5, -1.25, 0.75))
    _assert_vector(result.other_force_world, (0, 3, 0))
    _assert_vector(result.other_moment_world, (-6, 0, -3))
    assert result.group_normal_loads == (4.0,)
    assert len(result.contacts) == 5
    native = next(c for c in result.contacts if c.source == "native" and c.active)
    assert native.other_body == 2 and native.other_geom == 1
    assert native.identity is native.sample_id is native.source_id is native.area_m2 is None
    _assert_vector(native.couple_world, (0.5, -0.25, 0.75))
    inactive = next(c for c in result.contacts if not c.active)
    assert inactive.other_body == 4 and inactive.normal_load == 0
    _assert_vector(inactive.force_world, ORIGIN)
    for index, contact in enumerate(c for c in result.contacts if c.source == "area"):
        assert contact.identity == IDENTITY and contact.other_geom is None
        assert contact.other_body == 3 and contact.area_m2 == 0.02 / 3
        assert contact.index == index and contact.sample_id == ("triangle-0", index)
        assert contact.source_id == ("cell", "face")
        assert not contact.group_member
        _assert_vector(contact.couple_world, ORIGIN)
    moved = _measure(reader, ws, reference=REFERENCE)
    _assert_vector(moved.native_moment_world, (4.5, -0.85, 0.05))
    _assert_vector(moved.area_moment_world, (-4.8, 0, -3.6))
    _assert_vector(moved.total_moment_world, (-0.3, -0.85, -3.55))
    all_groups = _measure(reader, ws, groups=("support", "finger", "bystander"))
    assert all_groups.group_normal_loads == (3.0, 4.0, 0.0)
    _assert_vector(all_groups.group_force_world, result.total_force_world)
    _assert_vector(all_groups.group_moment_world, result.total_moment_world)
    _assert_vector(all_groups.other_force_world, ORIGIN)
    assert {name: value.tobytes() for name, value in ws.arrays.items()} == saved
    with pytest.raises(FrozenInstanceError):
        result.time_s = 17
    with pytest.raises(FrozenInstanceError):
        native.normal_load = 17


def test_both_sides_cancel_at_common_origin_and_area_split_keeps_total():
    ws = _packet()
    reader = _reader(ws)
    bodies = [
        _measure(reader, ws, body=name, groups=())
        for name in ("object", "finger", "support", "bystander")
    ]
    _assert_vector(np.sum([row.total_force_world for row in bodies], axis=0), ORIGIN)
    _assert_vector(np.sum([row.total_moment_world for row in bodies], axis=0), ORIGIN)
    _assert_vector(bodies[1].total_force_world, (-1, -2, -4))
    _assert_vector(bodies[1].total_moment_world, (-2.5, 1.25, -0.75))
    _assert_vector(bodies[2].total_force_world, (0, -3, 0))
    _assert_vector(bodies[2].total_moment_world, (6, 0, 3))
    refined = _packet(split=True)
    fine = _measure(_reader(refined), refined)
    _assert_vector(fine.total_force_world, bodies[0].total_force_world)
    _assert_vector(fine.total_moment_world, bodies[0].total_moment_world)
    samples = [row for row in fine.contacts if row.source == "area"]
    assert len(samples) == 12 and all(row.normal_load == 0.25 for row in samples)
    assert all(row.area_m2 == 0.02 / 12 for row in samples)
    assert sum(row.area_m2 for row in samples) == pytest.approx(0.02, rel=0, abs=1e-15)
    assert sum(row.normal_load * row.area_m2 for row in samples) < 0.01
    _assert_vector(fine.area_force_world, (0, 3, 0))


def test_generic_area_rows_are_owned_complete_and_identity_bound():
    ws = _packet()
    identity = replace(IDENTITY)
    supplied = {"model": (identity,)}
    reader = AreaPressureEntryReader(ws, supplied)
    supplied.clear()
    object.__setattr__(identity, "compliant_material", "changed-caller-material")
    rows = reader.validate(ws.arrays, ws.frame)
    assert ws.reads == 0 and set(rows) == set(KEYS)
    row = rows[13]
    assert type(row.samples) is tuple and len(row.samples) == 3 and len(row.digest) == 64
    for index, sample in enumerate(row.samples):
        assert sample.identity == IDENTITY and sample.stamp.world_id == "13"
        assert sample.pair_index == 0 and sample.sample_index == index
        assert sample.sample_id == ("triangle-0", index)
        assert sample.source_id == ("cell", "face")
        assert (sample.body_a, sample.body_b) == (1, 3)
        assert sample.area_m2 == 0.02 / 3 and sample.normal_force_n == 1
        assert sample.stiffness_n_per_m == float(np.float32(1000 * 0.02 / 3))
        assert sample.surrogate_gap_m == float(np.float32(-0.15))
        assert sample.normal_world == (0.0, 1.0, 0.0)
        assert sample.moment_on_a_world is not None
    with pytest.raises(FrozenInstanceError):
        row.samples = ()
    with pytest.raises(FrozenInstanceError):
        row.samples[0].area_m2 = 1
    with pytest.raises(ValueError):
        AreaPressureEntryReader(ws, {})
    with pytest.raises(ValueError):
        AreaPressureEntryReader(ws, {"model": (replace(IDENTITY, compliant_body="finger"),)})
    changed = _changed(ws)
    values = {k: v.copy() for k, v in changed.arrays.items()}
    values["area_data"][0, 0, 8:15] *= 2
    changed = _changed(ws, arrays=values)
    updated = reader.validate(changed.arrays, changed.frame)[13]
    assert updated.digest != row.digest and row.samples[0].normal_force_n == 1
    assert updated.samples[0].normal_force_n == 2


def test_complete_frame_area_metadata_and_payload_rejections_do_not_mutate():
    ws = _packet()
    # Every corruption is in the second world; measuring first world must
    # still validate the entire coherent area packet before returning a total.
    for name, index, value in (
        ("area_ids", (1, 0, 1), 1),
        ("area_ids", (1, 0, 3), 2),
        ("area_meta", (1, 1), 2),
        ("area_meta", (1, 3), 1),
        ("area_declared_mask", (1, 0), 0),
        ("area_measure_words", (1, 0, slice(None)), 0),
        ("area_data", (1, 0, 8), np.nan),
        ("area_data", (1, -1, 0), 1),
        ("area_data", (1, 0, 12), 123),
        ("entry_meta", (1, 1), 3),
        ("entry_meta", (1, 4), 0),
    ):
        arrays = {k: v.copy() for k, v in ws.arrays.items()}
        arrays[name][index] = value
        bad = _changed(ws, arrays=arrays)
        saved = {k: v.tobytes() for k, v in bad.arrays.items()}
        with pytest.raises(ValueError):
            _measure(_reader(bad), bad)
        assert bad.reads == 1 and {k: v.tobytes() for k, v in bad.arrays.items()} == saved
    item = ws.frame.area_catalogs[41][0]
    for changes in (
        {"identity": replace(IDENTITY, compliant_material="wrong-material")},
        {"stamp": replace(item["stamp"], sample_time_s=0.375001)},
        {"sample_ids": item["sample_ids"][:-1]},
        {"source_ids": item["source_ids"][:-1]},
    ):
        catalogs = dict(ws.frame.area_catalogs)
        catalogs[41] = (MappingProxyType({**dict(item), **changes}),)
        bad = _changed(ws, catalogs=catalogs)
        with pytest.raises(ValueError):
            _measure(_reader(bad), bad)
        assert bad.reads == 1


def test_native_owner_duplicate_pair_and_bad_wrench_rejections():
    ws = _packet()
    for field, index, value in (
        ("entry_contact_ids", (0, 0, 4), 4),
        ("entry_contact_data", (0, 0, 13), -1),
        ("entry_contact_data", (0, 0, 14), np.inf),
        ("entry_contact_data", (0, -1, 1), 1),
    ):
        arrays = {k: v.copy() for k, v in ws.arrays.items()}
        arrays[field][index] = value
        bad = _changed(ws, arrays=arrays)
        with pytest.raises(ValueError):
            _measure(_reader(bad), bad)
    arrays = {k: v.copy() for k, v in ws.arrays.items()}
    arrays["entry_contact_ids"][0, 0, 2] = 2
    arrays["entry_contact_ids"][0, 0, 4] = 3
    duplicate = _changed(ws, arrays=arrays)
    with pytest.raises(ValueError, match="duplicate"):
        _measure(_reader(duplicate), duplicate)


def test_unavailable_moment_is_real_sentinel_and_only_blocks_relevant_body():
    ws = _packet()
    arrays = {k: v.copy() for k, v in ws.arrays.items()}
    arrays["area_ids"][0, :3, 5] = 0
    arrays["area_data"][0, :3, 12:15] = 0
    forged = _changed(ws, arrays=arrays)
    with pytest.raises(ValueError, match="moment"):
        _measure(_reader(forged), forged)
    arrays = {k: v.copy() for k, v in ws.arrays.items()}
    # FP32 operands are finite but their cross-product terms exceed 2**125.
    arrays["area_ids"][1, :3, 5] = 0
    arrays["area_data"][1, :3, :3] = (2.0**64, 0, 0)
    arrays["area_data"][1, :3, 8] = 2.0**64
    arrays["area_data"][1, :3, 9:12] = (0, 2.0**64, 0)
    arrays["area_data"][1, :3, 12:15] = 0
    unavailable = _changed(ws, arrays=arrays)
    generic = AreaPressureEntryReader(unavailable, {"model": (IDENTITY,)})
    assert all(
        s.moment_on_a_world is None
        for s in generic.validate(unavailable.arrays, unavailable.frame)[41].samples
    )
    reader = _reader(unavailable)
    _assert_vector(_measure(reader, unavailable).total_force_world, (1, 5, 4))
    _assert_vector(
        _measure(reader, unavailable, world=41, body="finger", groups=()).total_force_world,
        (-1, -2, -4),
    )
    with pytest.raises(ValueError, match="moment"):
        _measure(reader, unavailable, world=41)


def test_invalid_empty_and_explicit_query_arguments():
    absent = _packet(invalid=True)
    assert _measure(_reader(absent), absent) is None
    empty = _packet(empty=True)
    result = _measure(_reader(empty), empty)
    assert result is not None and result.contacts == ()
    _assert_vector(result.total_force_world, ORIGIN)
    _assert_vector(result.total_moment_world, ORIGIN)
    ws = _packet()
    reader = _reader(ws)
    for arguments in (
        {"world_key": 99},
        {"body_name": "missing"},
        {"reference_point_world": (np.nan, 0, 0)},
        {"group_body_names": ("finger", "finger")},
        {"group_body_names": ("missing",)},
        {"group_body_names": ("object",)},
    ):
        kwargs = dict(
            world_key=13, body_name="object", reference_point_world=ORIGIN, group_body_names=()
        )
        kwargs.update(arguments)
        with pytest.raises(ValueError):
            reader.measure(ws.frame, **kwargs)
    assert ws.reads == 0
    for label in ("", "duplicate"):
        ambiguous = _packet()
        meta = dict(ambiguous.model_metadata["model"])
        meta["body_names"] = ("world", "object", label, "support", label)
        ambiguous.model_metadata = MappingProxyType({"model": MappingProxyType(meta)})
        bound = _reader(ambiguous)
        for body, groups in ((label, ()), ("object", (label,))):
            with pytest.raises(ValueError):
                bound.measure(
                    ambiguous.frame,
                    world_key=13,
                    body_name=body,
                    reference_point_world=ORIGIN,
                    group_body_names=groups,
                )
        assert ambiguous.reads == 0


def test_native_only_old_consumers_remain_compatible_and_refuse_selected_area():
    native = _packet(area=False)
    old = measure_object_contact_wrench(
        native.arrays, 0, object_body=1, object_com_world=REFERENCE, group_geom_ids=(1,)
    )
    current = _measure(_reader(native), native, reference=REFERENCE)
    assert current.total_force_world == old.total_force_world
    assert current.total_moment_world == old.total_moment_com_world
    assert current.group_force_world == old.group_force_world
    assert current.group_moment_world == old.group_moment_com_world
    assert current.group_normal_loads == old.group_normal_loads
    slip = measure_finger_slip(native.arrays, 0, object_body=1, pad_geom_ids=(1, 2))
    assert slip.normal_loads == (4.0, 0.0) and slip.slip_speeds == (0.0, 0.0)
    # Export capability alone does not declare an area model. A native batch
    # may request the extended workspace and receive legitimate zero banks.
    extended = {name: value.copy() for name, value in native.arrays.items()}
    extended.update(
        area_meta=np.zeros((2, 4), np.int32),
        area_ids=np.zeros((2, AREA_CAPACITY, 6), np.int32),
        area_data=np.zeros((2, AREA_CAPACITY, 15), np.float32),
        area_measure_words=np.zeros((2, AREA_CAPACITY, 2), np.int32),
        area_declared_mask=np.zeros((2, 1), np.int32),
    )
    exported = HandWorkspace(extended, dict.fromkeys(KEYS, ()), area=True)
    exported.binding = MappingProxyType(
        {
            **dict(exported.binding),
            "area_pair_names": MappingProxyType({"model": ()}),
            "source": MappingProxyType(
                {
                    "worlds": exported.binding["source"]["worlds"],
                    "buckets": (MappingProxyType({"area_pressure_pairs": ()}),),
                }
            ),
        }
    )
    exported.frame.binding = exported.binding
    assert _measure(BodyContactWrenchReader(exported), exported, reference=REFERENCE) == current
    for invalid in (False, True):
        area = _packet(invalid=invalid)
        with pytest.raises(ValueError, match="area"):
            measure_object_contact_wrench(area.arrays, 0, object_body=1, object_com_world=ORIGIN)
        with pytest.raises(ValueError, match="area"):
            measure_finger_slip(area.arrays, 0, object_body=1, pad_geom_ids=(1, 2))


def _actual_oracle(batch, arrays, key, row, body, reference):
    native_f, native_m, area_f, area_m = (np.zeros(3) for _ in range(4))
    for index in range(int(arrays["entry_meta"][row, 3])):
        ids = arrays["entry_contact_ids"][row, index]
        if not ids[1] or body not in ids[4:6]:
            continue
        sign = 1 if int(ids[5]) == body else -1
        wrench = arrays["entry_contact_wrench_world"][row, index].astype(np.float64)
        force = sign * wrench[:3]
        point = arrays["entry_contact_data"][row, index, 1:4].astype(np.float64)
        native_f += force
        native_m += sign * wrench[3:] + np.cross(point - reference, force)
    for pair in batch.last_step.receipts[key].raw_contact_fields.get("area_pressure", ()):
        if body not in pair["body_ids"]:
            continue
        sign = 1 if body == pair["body_ids"][0] else -1
        forces = sign * pair["force_on_a"]
        area_f += np.sum(forces, axis=0)
        area_m += np.sum(
            sign * pair["torque_on_a_world_origin"] - np.cross(reference, forces), axis=0
        )
    return native_f, native_m, area_f, area_m


@pytest.mark.cuda
def test_actual_cuda_entry_wrenches_reset_and_fresh_paired_restore(tmp_path):
    from test_manipulation_area_env import KEYS as real_keys
    from test_manipulation_area_env import _declarations, _make
    from test_manipulation_env import evidence, record, step
    from test_mujoco_tyx_area_pressure_lifecycle import _same

    from phymeo.training.manipulation_checkpoint import ManipulationEnvCheckpoint

    declarations = _declarations()
    identities = {
        model: tuple(asset.identity for asset in values)
        for model, values in declarations["assets"].items()
    }
    bodies = ("object", "support-body", "left-finger-body", "right-finger-body")
    with ExitStack() as stack, evidence(tmp_path / "body-wrench-current-entry.json.gz") as records:
        env, batch, _ = _make(stack, declarations)
        revision, initial = batch.checkpoint_revision, batch.integration_snapshot()
        incomplete = stack.enter_context(
            batch.prepare_device_workspace(include_contacts=True, include_area_pressure=False)
        )
        with pytest.raises(ValueError, match="area"):
            BodyContactWrenchReader(incomplete)
        assert batch.checkpoint_revision == revision
        _same(batch.integration_snapshot(), initial)
        reader = BodyContactWrenchReader(env.workspace, area_identities=identities)
        assert (
            reader.measure(
                env.current_frame, world_key=41, body_name="object", reference_point_world=ORIGIN
            )
            is None
        )
        step(env)
        checkpoint = ManipulationEnvCheckpoint.from_bytes(env.checkpoint().to_bytes())
        transition = step(env)
        arrays = env.workspace.readback(env.current_frame)
        expected = {}
        for row, key in enumerate(real_keys):
            model = batch.layout[key].model_key
            names = tuple(env.workspace.model_metadata[model]["body_names"])
            for body_name in bodies:
                body = names.index(body_name)
                value = reader.measure(
                    env.current_frame,
                    world_key=key,
                    body_name=body_name,
                    reference_point_world=ORIGIN,
                )
                expected[(key, body_name)] = value
                oracle = _actual_oracle(batch, arrays, key, row, body, np.zeros(3))
                for actual, truth in zip(
                    (
                        value.native_force_world,
                        value.native_moment_world,
                        value.area_force_world,
                        value.area_moment_world,
                    ),
                    oracle,
                    strict=True,
                ):
                    np.testing.assert_allclose(actual, truth, rtol=3e-5, atol=3e-6)
                np.testing.assert_allclose(
                    value.total_force_world, oracle[0] + oracle[2], rtol=3e-5, atol=3e-6
                )
                np.testing.assert_allclose(
                    value.total_moment_world, oracle[1] + oracle[3], rtol=3e-5, atol=3e-6
                )
                assert value.input_tick == 1 and value.output_tick == 2
            np.testing.assert_allclose(
                np.sum([expected[(key, name)].total_force_world for name in bodies], axis=0),
                0,
                rtol=0,
                atol=3e-6,
            )
            np.testing.assert_allclose(
                np.sum([expected[(key, name)].total_moment_world for name in bodies], axis=0),
                0,
                rtol=0,
                atol=3e-6,
            )
        record(
            records,
            env,
            batch,
            transition,
            measured=tuple((key, name, value) for (key, name), value in expected.items()),
        )
        env.reset((41,), expected_frame=env.current_frame)
        assert (
            reader.measure(
                env.current_frame, world_key=41, body_name="object", reference_point_world=ORIGIN
            )
            is None
        )
        _same(
            reader.measure(
                env.current_frame, world_key=305, body_name="object", reference_point_world=ORIGIN
            ),
            expected[(305, "object")],
        )
        fresh, fresh_batch, _ = _make(stack, declarations)
        fresh.restore(checkpoint, expected_revision=fresh_batch.checkpoint_revision)
        fresh_reader = BodyContactWrenchReader(fresh.workspace, area_identities=identities)
        replay = step(fresh)
        for (key, name), value in expected.items():
            _same(
                fresh_reader.measure(
                    fresh.current_frame, world_key=key, body_name=name, reference_point_world=ORIGIN
                ),
                value,
            )
        record(records, fresh, fresh_batch, replay)

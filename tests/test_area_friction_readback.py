"""Manufactured HOST v2 protocol packets; no runtime or physical qualification."""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_body_contact_wrench import IDENTITY, _owned, _packet

from phymeo.backends.area_friction_readback import (
    NORMAL_LAW,
    AreaFrictionEntryReader,
    AreaFrictionSample,
)
from phymeo.backends.area_pressure_readback import AreaPressureEntryReader

LAW = "final_normal_smooth_v1"
LAW_BINDING = (
    LAW,
    "mujoco-tyx-area-friction-law-v1",
    "actual-packed-point-fp32",
    (("object", "support", "hand-pressure", "hand-rigid", 0.5, 0.003),),
)
IDENTITIES = {"friction": (IDENTITY,), "normal": (IDENTITY,)}


def _fixture(*, empty=False, invalid=False, native_second=False):
    base = _packet(empty=empty, invalid=invalid)
    arrays = {name: value.copy() for name, value in base.arrays.items()}
    old = arrays["area_data"]
    values = np.zeros((*old.shape[:2], 23), np.float32)
    values[:, :, :15] = old
    arrays["area_data"] = values
    count = int(arrays["area_meta"][0, 1])
    if not invalid:
        arrays["area_meta"][0, 0] = 2
    # n=(0,1,0), Fn=1, vt=(.004,0,0), vs=.003: Ft=(-.4,0,0).
    values[0, :count, 15] = -0.4 * values[0, :count, 8]
    values[0, :count, 18:21] = (0.004, 0, 0)
    values[0, :count, 21:23] = (0.5, 0.003)
    values[0, :count, 9:12] += values[0, :count, 15:18]
    values[0, :count, 12:15] = np.cross(values[0, :count, :3], values[0, :count, 9:12])
    pairs = (("object", "support"),)
    binding = {
        **base.binding,
        "area_schema": "mujoco-tyx-area-device-entry-v2",
        "area_pair_names": MappingProxyType(
            {"friction": pairs, "normal": () if native_second else pairs}
        ),
        "area_laws": MappingProxyType({"friction": LAW_BINDING, "normal": None}),
        "source": MappingProxyType(
            dict(
                worlds=((13, "friction", 0, 0), (41, "normal", 1, 0)),
                buckets=(
                    MappingProxyType(dict(area_pressure_pairs=pairs, area_friction=LAW_BINDING)),
                    MappingProxyType(dict(area_pressure_pairs=() if native_second else pairs)),
                ),
            )
        ),
    }
    catalogs = dict(base.frame.area_catalogs)
    if native_second:
        catalogs[41] = ()
        for name in (
            "area_meta",
            "area_ids",
            "area_data",
            "area_measure_words",
            "area_declared_mask",
        ):
            arrays[name][1].fill(0)
    binding = MappingProxyType(binding)
    return SimpleNamespace(
        binding=binding,
        model_metadata=MappingProxyType(
            dict.fromkeys(("friction", "normal"), base.model_metadata["model"])
        ),
        arrays=MappingProxyType({name: _owned(value) for name, value in arrays.items()}),
        frame=SimpleNamespace(
            binding=binding, ticks=base.frame.ticks, area_catalogs=MappingProxyType(catalogs)
        ),
    )


def _reader(ws, *, native_second=False):
    return AreaFrictionEntryReader(ws, {"friction": (IDENTITY,)} if native_second else IDENTITIES)


def _arrays(ws):
    return {name: value.copy() for name, value in ws.arrays.items()}


def _rebind(ws, **changes):
    binding = MappingProxyType({**ws.binding, **changes})
    return SimpleNamespace(
        binding=binding,
        model_metadata=ws.model_metadata,
        arrays=ws.arrays,
        frame=SimpleNamespace(
            binding=binding, ticks=ws.frame.ticks, area_catalogs=ws.frame.area_catalogs
        ),
    )


def test_mixed_v2_has_full_tangent_wrench_and_explicit_normal_only_unknown_slip():
    ws = _fixture()
    result = _reader(ws).validate(ws.arrays, ws.frame)
    friction, normal = result[13].samples, result[41].samples
    assert len(friction) == len(normal) == 3
    assert all(type(sample) is AreaFrictionSample for sample in friction + normal)
    np.testing.assert_allclose(
        np.sum([s.force_on_a_world for s in friction], axis=0), (-1.2, 3, 0), rtol=0, atol=2e-6
    )
    np.testing.assert_allclose(
        np.sum([s.moment_on_a_world for s in friction], axis=0), (-6, -2.4, -3), rtol=0, atol=2e-6
    )
    assert sum(s.area_m2 for s in friction) == pytest.approx(0.02, rel=0, abs=1e-17)
    for sample in friction:
        assert sample.law == LAW and sample.mu == 0.5
        assert sample.slip_speed_m_s == float(np.float32(0.003))
        assert sample.relative_velocity_world == (float(np.float32(0.004)), 0, 0)
        assert sample.tangential_force_on_a_world == (float(np.float32(-0.4)), 0, 0)
        assert sample.identity == IDENTITY and sample.body_a == 1 and sample.body_b == 3
    for sample in normal:
        assert sample.law == NORMAL_LAW
        assert sample.tangential_force_on_a_world == (0, 0, 0)
        assert sample.relative_velocity_world is sample.mu is sample.slip_speed_m_s is None
    native = _fixture(native_second=True)
    assert (
        _reader(native, native_second=True).validate(native.arrays, native.frame)[41].samples == ()
    )


def test_v1_and_v2_abis_cannot_be_silently_reinterpreted():
    old, new = _packet(), _fixture()
    with pytest.raises(ValueError):
        AreaPressureEntryReader(new, IDENTITIES)
    with pytest.raises(ValueError):
        AreaFrictionEntryReader(old, {"model": (IDENTITY,)})
    for name, value in (
        ("area_schema", "unknown"),
        ("entry_phase", "endpoint"),
        ("entry_body_kinematics", False),
    ):
        with pytest.raises(ValueError):
            _reader(_rebind(new, **{name: value}))
    arrays = _arrays(new)
    arrays["area_data"] = arrays["area_data"][:, :, :15].copy()
    with pytest.raises(ValueError):
        _reader(new).validate(arrays, new.frame)


def test_complete_law_material_source_and_actual_parameter_bindings():
    ws = _fixture()
    wrong = (
        LAW,
        LAW_BINDING[1],
        LAW_BINDING[2],
        (("object", "support", "different-material", "hand-rigid", 0.5, 0.003),),
    )
    for declared in (
        {},
        {"friction": LAW_BINDING},
        {"friction": wrong, "normal": None},
        {"friction": None, "normal": None},
        {"friction": (LAW, "v0", LAW_BINDING[2], LAW_BINDING[3]), "normal": None},
    ):
        with pytest.raises(ValueError):
            _reader(_rebind(ws, area_laws=declared))
    declared = dict(ws.binding["area_laws"])
    mutable = _rebind(ws, area_laws=declared)
    captured = _reader(mutable)
    declared["friction"] = None
    with pytest.raises(ValueError):
        captured.validate(mutable.arrays, mutable.frame)
    reader = _reader(ws)
    for column, value in ((21, 0.25), (22, 0), (22, 0.004)):
        changed = _arrays(ws)
        changed["area_data"][0, 0, column] = value
        with pytest.raises(ValueError):
            reader.validate(changed, ws.frame)
    for row, kind in ((0, 1), (0, 3), (1, 2)):
        changed = _arrays(ws)
        changed["area_meta"][row, 0] = kind
        with pytest.raises(ValueError):
            reader.validate(changed, ws.frame)


def test_force_law_tangent_capacity_direction_and_moment_rejections():
    ws = _fixture()
    reader = _reader(ws)
    mutations = (
        (8, -1),
        (9, 1),
        (15, 0.4),
        (16, 0.1),
        (15, -1),
        (18, -0.004),
        (12, 12),
        (7, 0.01),
        (15, np.nan),
        (22, np.inf),
    )
    for column, value in mutations:
        changed = _arrays(ws)
        changed["area_data"][0, 0, column] = value
        if column in (15, 16):
            p = changed["area_data"][0, 0]
            p[9:12] = p[8] * p[3:6] + p[15:18]
            p[12:15] = np.cross(p[:3], p[9:12])
        with pytest.raises(ValueError):
            reader.validate(changed, ws.frame)
    changed = _arrays(ws)
    changed["area_data"][1, 0, 18] = 0.001
    with pytest.raises(ValueError):
        reader.validate(changed, ws.frame)
    changed = _arrays(ws)
    changed["area_data"][0, -1, 22] = 0.003
    with pytest.raises(ValueError):
        reader.validate(changed, ws.frame)


def test_invalid_empty_and_zero_final_load_are_distinct_public_states():
    for invalid in (False, True):
        ws = _fixture(empty=True, invalid=invalid)
        result = _reader(ws).validate(ws.arrays, ws.frame)
        assert all(not row.samples for row in result.values())
        assert tuple(ws.arrays["area_meta"][:, 0]) == ((0, 0) if invalid else (2, 1))
    ws = _fixture()
    arrays = _arrays(ws)
    arrays["area_meta"][0, 2] = 0
    arrays["area_data"][0, :3, 8:18] = 0
    # Geometry remains and actual slip is still exported; zero Fn implies zero Ft.
    result = _reader(ws).validate(arrays, ws.frame)
    assert len(result[13].samples) == 3
    assert all(
        s.normal_force_n == 0 and s.force_on_a_world == (0, 0, 0) for s in result[13].samples
    )
    invalid = _fixture(invalid=True)
    arrays = _arrays(invalid)
    arrays["area_data"][0, 0, 15] = 0.1
    with pytest.raises(ValueError):
        _reader(invalid).validate(arrays, invalid.frame)


def test_inherited_phase_catalog_owner_measure_and_native_duplicate_guards():
    ws = _fixture()
    reader = _reader(ws)
    baseline = reader.validate(ws.arrays, ws.frame)
    for name, index, value in (
        ("entry_meta", (1, 2), 6),
        ("entry_clock", (1, 0), 0.6),
        ("entry_body_mask", (1, 3), 0),
        ("area_ids", (1, 0, 4), 2),
        ("area_meta", (1, 3), 1),
        ("area_measure_words", (1, 0, 1), 0),
    ):
        changed = _arrays(ws)
        changed[name][index] = value
        with pytest.raises(ValueError):
            reader.validate(changed, ws.frame)
    changed = _arrays(ws)
    changed["area_measure_words"][1, 0] = 0
    with pytest.raises(ValueError):
        reader.validate(changed, ws.frame)
    changed = _arrays(ws)
    changed["entry_contact_ids"][1, 0, 2:6] = (0, 2, 1, 3)
    with pytest.raises(ValueError):
        reader.validate(changed, ws.frame)
    catalogs = dict(ws.frame.area_catalogs)
    item = dict(catalogs[41][0])
    item["stamp"] = replace(item["stamp"], configuration_id="other", step_index=3)
    catalogs[41] = (item,)
    bad_frame = SimpleNamespace(binding=ws.binding, ticks=ws.frame.ticks, area_catalogs=catalogs)
    with pytest.raises(ValueError):
        reader.validate(ws.arrays, bad_frame)
    assert reader.validate(ws.arrays, ws.frame) == baseline


def test_owned_samples_and_digest_include_actual_tangent_extension():
    ws = _fixture()
    reader = _reader(ws)
    arrays = _arrays(ws)
    first = reader.validate(arrays, ws.frame)
    sample = first[13].samples[0]
    with pytest.raises(FrozenInstanceError):
        sample.mu = 0.25
    with pytest.raises(FrozenInstanceError):
        sample.identity.compliant_material = "changed"
    arrays["area_data"][0, :3, 15:21] = 0
    arrays["area_data"][0, :3, 9:12] = (0, 1, 0)
    arrays["area_data"][0, :3, 12:15] = np.cross(arrays["area_data"][0, :3, :3], (0, 1, 0))
    second = reader.validate(arrays, ws.frame)
    assert second[13].digest != first[13].digest
    assert second[41].digest == first[41].digest
    assert sample.tangential_force_on_a_world[0] == float(np.float32(-0.4))
    assert second[13].samples[0].tangential_force_on_a_world == (0, 0, 0)


def test_unrepresentable_world_moment_is_none_without_inventing_zero_moment():
    ws = _fixture()
    arrays = _arrays(ws)
    values = arrays["area_data"][0, :3]
    values[:, :3] = (2.0**64, 0, 0)
    values[:, 8] = 2.0**64
    values[:, 9:12] = (0, 2.0**64, 0)
    values[:, 12:21] = 0
    arrays["area_ids"][0, :3, 5] = 0
    result = _reader(ws).validate(arrays, ws.frame)
    assert all(sample.moment_on_a_world is None for sample in result[13].samples)
    assert all(sample.force_on_a_world == (0, 2.0**64, 0) for sample in result[13].samples)

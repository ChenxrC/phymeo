"""Hand-vector contracts for entry contact forces and moments about an explicit COM."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.control.manipulation_wrench import (
    ObjectContactWrenchMeasurement,
    ObjectContactWrenchRecord,
    measure_object_contact_wrench,
)


def frame(capacity=5, bodies=5):
    return {
        "entry_meta": np.array([[1, 4, 5, 0, 0, 0]], dtype=np.int32),
        "entry_clock": np.array([[0.5, 0.125, 0.625, 0.125]], dtype=np.float32),
        "entry_contact_ids": np.zeros((1, capacity, 8), dtype=np.int32),
        "entry_contact_data": np.zeros((1, capacity, 20), dtype=np.float32),
        "entry_body_position": np.zeros((1, bodies, 3), dtype=np.float32),
        "entry_body_mask": np.ones((1, bodies), dtype=np.int32),
    }


def contact(arrays, slot, *, geoms=(10, 20), bodies=(2, 1), point=(2, 2, 3),
            axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)), wrench=(2, 3, 4, 5, 6, 7), dim=6, active=1):
    arrays["entry_contact_ids"][0, slot] = (1, active, *geoms, *bodies, slot, dim)
    values = arrays["entry_contact_data"][0, slot]
    values[0], values[1:4], values[4:13], values[13:19] = -.001, point, np.asarray(axes).reshape(9), wrench
    arrays["entry_meta"][0, 3] = slot + 1
    arrays["entry_meta"][0, 4] = arrays["entry_contact_ids"][0, :slot + 1, 1].sum()


def measure(arrays, **kwargs):
    options = dict(row=0, object_body=1, object_com_world=(1, 2, 3), group_geom_ids=(11, 10))
    return measure_object_contact_wrench(arrays, **(options | kwargs))


def loaded_frame():
    arrays = frame()
    arrays["entry_body_position"][0, 1] = (9, 8, 7)  # Deliberately NOT the supplied COM.
    contact(arrays, 0)
    contact(arrays, 1, geoms=(0, 20), bodies=(0, 1), point=(1, 3, 3),
            axes=((0, 0, 1), (1, 0, 0), (0, 1, 0)), wrench=(6, 0, 0, 0, 0, 0), dim=1)
    contact(arrays, 2, geoms=(11, 20), bodies=(3, 1), point=(1, 2, 3), wrench=(0,) * 6, active=0)
    contact(arrays, 3, geoms=(12, 0), bodies=(4, 0), wrench=(1, 0, 0, 0, 0, 0), dim=3)
    return arrays


def test_eccentric_force_true_couple_and_all_table_contacts_have_distinct_totals():
    arrays = loaded_frame()
    before = {name: value.tobytes() for name, value in arrays.items()}
    result = measure(arrays)
    assert isinstance(result, ObjectContactWrenchMeasurement)
    assert (result.time_s, result.input_tick, result.output_tick) == (.375, 4, 5)
    assert result.reference_com_world == (1, 2, 3)
    assert len(result.contacts) == 3  # The unrelated fourth contact is not an object load.
    first, table, inactive = result.contacts
    assert isinstance(first, ObjectContactWrenchRecord)
    assert (first.index, first.other_geom, first.other_body, first.active, first.group_member) == (0, 10, 2, True, True)
    assert first.point_world == (2, 2, 3)
    assert first.force_world == (2, 3, 4) and first.couple_world == (5, 6, 7)
    # (1,0,0) x (2,3,4) = (0,-4,3), plus the independent contact couple.
    assert first.moment_com_world == (5, 2, 10) and first.normal_load == 2
    assert table.force_world == (0, 0, 6) and table.moment_com_world == (6, 0, 0)
    assert not table.group_member and table.other_body == 0
    assert inactive.index == 2 and not inactive.active and inactive.group_member
    assert inactive.force_world == inactive.couple_world == inactive.moment_com_world == (0, 0, 0)
    assert result.group_normal_loads == (0, 2)  # Explicit (11,10) order, not contact order.
    assert result.group_force_world == (2, 3, 4) and result.group_moment_com_world == (5, 2, 10)
    assert result.other_force_world == (0, 0, 6) and result.other_moment_com_world == (6, 0, 0)
    assert result.total_force_world == (2, 3, 10) and result.total_moment_com_world == (11, 2, 10)
    assert {name: value.tobytes() for name, value in arrays.items()} == before
    ungrouped = measure(arrays, group_geom_ids=())
    assert ungrouped.group_normal_loads == () and ungrouped.group_force_world == (0, 0, 0)
    assert ungrouped.other_force_world == result.total_force_world
    assert ungrouped.other_moment_com_world == result.total_moment_com_world


def test_swapped_contact_sides_preserve_object_wrench_and_reaction_negates_the_couple():
    arrays = frame()
    contact(arrays, 0)
    original = measure(arrays)
    opposite = measure(arrays, object_body=2, group_geom_ids=(), object_com_world=(2, 1, 3))
    reaction = opposite.contacts[0]
    assert (reaction.other_geom, reaction.other_body) == (20, 1)
    assert reaction.force_world == (-2, -3, -4) and reaction.couple_world == (-5, -6, -7)
    # The opposite COM has lever (0,1,0): cross = (-4,0,2).
    assert reaction.moment_com_world == (-9, -6, -5)
    contact(arrays, 0, geoms=(20, 10), bodies=(1, 2),
            axes=((-1, 0, 0), (0, -1, 0), (0, 0, 1)), wrench=(2, 3, -4, 5, 6, -7))
    assert measure(arrays) == original


def test_equal_resultant_forces_do_not_hide_different_moments_or_repeated_group_loads():
    arrays = frame()
    contact(arrays, 0, point=(1, 0, 0), axes=((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            wrench=(4, 0, 0, 0, 0, 0), dim=1)
    right = measure(arrays, object_com_world=(0, 0, 0), group_geom_ids=(10,))
    arrays["entry_contact_data"][0, 0, 1] = -1
    left = measure(arrays, object_com_world=(0, 0, 0), group_geom_ids=(10,))
    assert left.total_force_world == right.total_force_world == (0, 0, 4)
    assert left.total_moment_com_world == (0, 4, 0) and right.total_moment_com_world == (0, -4, 0)
    contact(arrays, 1, point=(1, 0, 0), axes=((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            wrench=(4, 0, 0, 0, 0, 0), dim=1)
    pair = measure(arrays, object_com_world=(0, 0, 0), group_geom_ids=(10,))
    assert pair.group_normal_loads == (8,) and pair.total_force_world == (0, 0, 8)
    assert pair.total_moment_com_world == (0, 0, 0) and len(pair.contacts) == 2


def test_translation_rotation_covariance_and_explicit_com_reference_shift():
    arrays = loaded_frame()
    shifted_com = measure(arrays, object_com_world=(1, 3, 3))
    assert shifted_com.total_force_world == (2, 3, 10)
    assert shifted_com.total_moment_com_world == (1, 2, 12)
    # Rz(90 degrees), then a translation. Expected vectors below are hand rotated.
    rotation, translation = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1))), np.array((10, -7, 2))
    arrays["entry_body_position"][0] = arrays["entry_body_position"][0] @ rotation.T + translation
    for index in range(4):
        values = arrays["entry_contact_data"][0, index]
        values[1:4] = rotation @ values[1:4] + translation
        values[4:13] = (values[4:13].reshape(3, 3) @ rotation.T).reshape(9)
    result = measure(arrays, object_com_world=(8, -6, 5))
    assert result.total_force_world == (-3, 2, 10) and result.total_moment_com_world == (-2, 11, 10)
    assert result.group_force_world == (-3, 2, 4) and result.group_moment_com_world == (-2, 5, 10)
    assert result.contacts[0].couple_world == (-6, 5, 7) and result.contacts[0].point_world == (8, -5, 5)
    assert result.group_normal_loads == (0, 2)


@pytest.mark.parametrize("dim,wrench,force,couple,moment", [
    (1, (2, 0, 0, 0, 0, 0), (2, 0, 0), (0, 0, 0), (0, 0, -2)),
    (3, (2, 3, 4, 0, 0, 0), (2, 3, 4), (0, 0, 0), (4, 0, -2)),
    (4, (2, 3, 4, 5, 0, 0), (2, 3, 4), (5, 0, 0), (9, 0, -2)),
    (6, (2, 3, 4, 5, 6, 7), (2, 3, 4), (5, 6, 7), (9, 6, 5)),
])
def test_each_contact_dimension_preserves_its_real_force_and_couple(dim, wrench, force, couple, moment):
    arrays = frame()
    contact(arrays, 0, point=(0, 1, 0), wrench=wrench, dim=dim)
    result = measure(arrays, object_com_world=(0, 0, 0))
    assert result.total_force_world == force and result.total_moment_com_world == moment
    assert result.contacts[0].couple_world == couple


def test_empty_valid_and_invalid_entries_are_distinct_without_stale_payload_reads():
    empty = measure(frame())
    assert empty.contacts == () and empty.total_force_world == empty.total_moment_com_world == (0, 0, 0)
    assert empty.group_normal_loads == (0, 0)
    invalid = {"entry_meta": np.array([[0, 0, 42, 0, 0, 0]], dtype=np.int32)}
    assert measure(invalid) is None
    stale = frame()
    stale["entry_meta"][:] = invalid["entry_meta"]
    for name in stale:
        if name != "entry_meta":
            stale[name] = object()  # Even shape/dtype access to stale payload is forbidden.
    assert measure(stale) is None
    for index in (1, 3, 4, 5):
        bad = {"entry_meta": invalid["entry_meta"].copy()}
        bad["entry_meta"][0, index] = 1
        with pytest.raises(ValueError):
            measure(bad)


def test_corrupt_metadata_payload_padding_and_ownership_reject_without_mutation():
    cases = [
        ("entry_meta", (0, 0), 2), ("entry_meta", (0, 1), -1), ("entry_meta", (0, 2), 9),
        ("entry_meta", (0, 3), 6), ("entry_meta", (0, 4), 1), ("entry_meta", (0, 5), 1),
        ("entry_clock", (0, 0), np.nan), ("entry_clock", (0, 2), .25),
        ("entry_contact_ids", (0, 0, 0), 0), ("entry_contact_ids", (0, 0, 1), 2),
        ("entry_contact_ids", (0, 0, 2), -1), ("entry_contact_ids", (0, 0, 3), 10),
        ("entry_contact_ids", (0, 0, 4), 5), ("entry_contact_ids", (0, 0, 5), 2),
        ("entry_contact_ids", (0, 0, 6), -1), ("entry_contact_ids", (0, 0, 7), 2),
        ("entry_contact_ids", (0, 3, 2), 10),  # Same geom has a different owner even off-object.
        ("entry_contact_ids", (0, 4, 2), 7),  # Zero flags do not excuse nonzero padding identity.
        ("entry_contact_data", (0, 4, 1), 1), ("entry_contact_data", (0, 4, 1), np.nan),
        ("entry_contact_data", (0, 0, 1), np.inf), ("entry_contact_data", (0, 0, 13), -1),
        ("entry_contact_data", (0, 1, 14), 1), ("entry_contact_data", (0, 2, 13), 1),
        ("entry_contact_data", (0, 3, 16), 1),
        ("entry_body_mask", (0, 1), 0), ("entry_body_mask", (0, 2), 0),
        ("entry_body_mask", (0, 4), 2), ("entry_body_position", (0, 4, 0), np.nan),
    ]
    for name, index, value in cases:
        arrays = loaded_frame()
        arrays[name][index] = value
        before = {key: item.tobytes() for key, item in arrays.items()}
        with pytest.raises(ValueError):
            measure(arrays)
        assert {key: item.tobytes() for key, item in arrays.items()} == before, (name, index, value)
    arrays = frame()
    arrays["entry_body_mask"][0, 4] = 0
    arrays["entry_body_position"][0, 4, 0] = 1
    with pytest.raises(ValueError, match="body padding"):
        measure(arrays)
    contact(arrays, 0, dim=4, wrench=(2, 3, 4, 5, 6, 0))
    arrays["entry_body_position"][0, 4, 0] = 0
    with pytest.raises(ValueError, match="wrench dimension"):
        measure(arrays)


def test_full_frame_proper_rotation_validation_does_not_repair_small_roundoff():
    for axes in (((1, 0, 0), (1, 0, 0), (0, 0, 1)),
                 ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
                 ((1, 0, 0), (0, 2, 0), (0, 0, 1)),
                 ((1, 0, 0), (.001, 1, 0), (0, 0, 1))):
        arrays = frame()
        contact(arrays, 0, axes=axes)
        with pytest.raises(ValueError, match="SO\\(3\\)"):
            measure(arrays)
    arrays = frame()
    contact(arrays, 0, axes=((1.00001, 0, 0), (0, 1, 0), (0, 0, 1)))
    scale = float(arrays["entry_contact_data"][0, 0, 4])
    result = measure(arrays)
    assert result.contacts[0].force_world[0] == 2 * scale != 2
    assert result.contacts[0].couple_world[0] == 5 * scale


def test_bad_shapes_indices_com_and_bounded_storage_reject():
    for options in ({"row": True}, {"row": 1}, {"object_body": False}, {"object_body": 5},
                    {"group_geom_ids": [10]}, {"group_geom_ids": (10, 10)}, {"group_geom_ids": (-1,)},
                    {"group_geom_ids": (True,)}, {"group_geom_ids": tuple(range(9))},
                    {"group_geom_ids": (2147483648,)}, {"group_geom_ids": (20,)},
                    {"object_com_world": (0, 0)}, {"object_com_world": (0, np.nan, 0)},
                    {"object_com_world": (0, 0, np.inf)}, {"object_com_world": (True, False, True)}):
        with pytest.raises(ValueError):
            measure(loaded_frame(), **options)
    replacements = {
        "entry_meta": np.zeros((1, 5), np.int32), "entry_clock": np.zeros((2, 4), np.float32),
        "entry_contact_ids": np.zeros((1, 5, 8), bool), "entry_contact_data": np.zeros((1, 4, 20), np.float32),
        "entry_body_position": np.zeros((1, 5, 2), np.float32), "entry_body_mask": np.ones((1, 4), np.int32),
    }
    for name, value in replacements.items():
        arrays = frame()
        arrays[name] = value
        with pytest.raises(ValueError):
            measure(arrays)
    for name in frame():
        arrays = frame()
        del arrays[name]
        with pytest.raises(ValueError):
            measure(arrays)
    limit = frame(capacity=4096, bodies=128)
    limit = {name: np.broadcast_to(value, (64, *value.shape[1:])) for name, value in limit.items()}
    assert measure(limit, row=63).contacts == ()
    assert measure(limit, row=63, group_geom_ids=tuple(range(8))).group_normal_loads == (0,) * 8
    for name, shape, dtype in (("entry_meta", (65, 6), np.int32), ("entry_meta", (0, 6), np.int32),
                               ("entry_contact_ids", (64, 4097, 8), np.int32),
                               ("entry_contact_ids", (64, 0, 8), np.int32),
                               ("entry_body_position", (64, 129, 3), np.float32),
                               ("entry_body_position", (64, 0, 3), np.float32)):
        arrays = dict(limit)
        arrays[name] = np.zeros(shape, dtype=dtype)
        with pytest.raises(ValueError):
            measure(arrays)


def test_other_world_payload_is_unread_and_outputs_are_frozen_owned_tuples():
    source = loaded_frame()
    arrays = {name: np.repeat(value, 2, axis=0) for name, value in source.items()}
    arrays["entry_meta"][0] = (2, -1, -1, -1, -1, 1)
    for value in arrays.values():
        if value.dtype.kind == "f":
            value[0] = np.nan
        value.setflags(write=False)
    com = np.array((1., 2., 3.))
    result = measure(arrays, row=1, object_com_world=com)
    assert result == measure(source)
    with pytest.raises(FrozenInstanceError):
        result.time_s = 0
    with pytest.raises(FrozenInstanceError):
        result.contacts[0].normal_load = 0
    with pytest.raises(TypeError):
        result.total_force_world[0] = 0
    with pytest.raises(TypeError):
        result.contacts[0].point_world[0] = 0
    com[:] = 0
    for value in source.values():
        value[:] = 0
    assert result.reference_com_world == (1, 2, 3) and result.total_moment_com_world == (11, 2, 10)


def test_finite_payload_with_nonfinite_derived_moment_or_aggregate_rejects():
    arrays = frame()
    arrays["entry_contact_data"] = arrays["entry_contact_data"].astype(np.float64)
    contact(arrays, 0, point=(1e308, 0, 0))
    with pytest.raises(ValueError, match="derived contact wrench"):
        measure(arrays)
    contact(arrays, 0, point=(1, 2, 3), wrench=(1e308, 0, 0, 0, 0, 0), dim=1)
    contact(arrays, 1, point=(1, 2, 3), wrench=(1e308, 0, 0, 0, 0, 0), dim=1)
    with pytest.raises(ValueError, match="aggregate contact wrench"):
        measure(arrays)

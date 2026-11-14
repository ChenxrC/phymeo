"""Hand-vector contracts for public entry contact measurements."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.control.manipulation_contact import FingerSlipMeasurement, measure_finger_slip


def frame(capacity=4):
    arrays = {
        "entry_meta": np.array([[1, 4, 5, 0, 0, 0]], dtype=np.int32),
        "entry_clock": np.array([[0.5, 0.125, 0.625, 0.125]], dtype=np.float32),
        "entry_contact_ids": np.zeros((1, capacity, 8), dtype=np.int32),
        "entry_contact_data": np.zeros((1, capacity, 20), dtype=np.float32),
        "entry_body_position": np.zeros((1, 4, 3), dtype=np.float32),
        "entry_body_linear_velocity": np.zeros((1, 4, 3), dtype=np.float32),
        "entry_body_angular_velocity": np.zeros((1, 4, 3), dtype=np.float32),
        "entry_body_mask": np.ones((1, 4), dtype=np.int32),
    }
    return arrays


def contact(arrays, slot, *, pad=10, pad_body=2, point=(0, 0, 0), load=1.0, reverse=False):
    ids = (1, 1, 20, pad, 1, pad_body, slot, 3)
    normal = (0.0, 1.0, 0.0)
    if reverse:
        ids = (1, 1, pad, 20, pad_body, 1, slot, 3)
        normal = (0.0, -1.0, 0.0)
    arrays["entry_contact_ids"][0, slot] = ids
    data = arrays["entry_contact_data"][0, slot]
    data[0], data[1:4], data[4:7], data[13] = -0.001, point, normal, load
    data[7:10] = (1.0, 0.0, 0.0)
    data[10:13] = (0.0, 0.0, -1.0 if not reverse else 1.0)
    arrays["entry_meta"][0, 3:5] = slot + 1


def measure(arrays, **kwargs):
    options = dict(row=0, object_body=1, pad_geom_ids=(10, 11))
    options.update(kwargs)
    return measure_finger_slip(arrays, **options)


@pytest.mark.parametrize("reverse", [False, True])
def test_rotation_at_each_body_origin_and_contact_order(reverse):
    arrays = frame()
    contact(arrays, 0, point=(1, 0, 0), reverse=reverse)
    arrays["entry_body_position"][0, 2] = (0, 0, 1)
    arrays["entry_body_linear_velocity"][0, 1] = (3, 4, 5)
    arrays["entry_body_linear_velocity"][0, 2] = (1, 2, 3)
    arrays["entry_body_angular_velocity"][0, 1] = (0, 0, 2)
    arrays["entry_body_angular_velocity"][0, 2] = (0, 1, 0)
    # Object point velocity=(3,6,5); pad=(0,2,2). Relative=(3,4,3).
    # Removing the +/-Y normal leaves (3,0,3), hence sqrt(18), not COM speed.
    result = measure(arrays)
    assert isinstance(result, FingerSlipMeasurement)
    assert result.time_s == 0.375
    assert (result.input_tick, result.output_tick) == (4, 5)
    assert result.normal_loads == (1.0, 0.0)
    assert result.slip_speeds == pytest.approx((18**0.5, 0.0))


def test_multiple_contacts_sum_loads_but_take_maximum_speed_per_pad():
    arrays = frame()
    contact(arrays, 0, point=(0, 0, 0), load=0.25)
    contact(arrays, 1, point=(0, 2, 0), load=0.5, reverse=True)
    contact(arrays, 2, pad=11, pad_body=3, point=(0, 3, 0), load=0.75)
    arrays["entry_body_angular_velocity"][0, 1] = (1, 0, 0)
    result = measure(arrays)
    assert result.normal_loads == (0.75, 0.75)
    assert result.slip_speeds == (2.0, 3.0)


def test_pure_normal_velocity_has_no_slip_and_inactive_contacts_do_not_contribute():
    arrays = frame()
    contact(arrays, 0)
    contact(arrays, 1, pad=11, pad_body=3)
    arrays["entry_contact_ids"][0, 1, 1] = 0
    arrays["entry_contact_data"][0, 1, 13:19] = 0
    arrays["entry_meta"][0, 4] = 1
    arrays["entry_body_linear_velocity"][0, 1] = (0, 10, 0)
    result = measure(arrays)
    assert result.normal_loads == (1.0, 0.0)
    assert result.slip_speeds == (0.0, 0.0)


def test_valid_no_contacts_and_no_matching_pad_contacts_are_zero_measurements():
    arrays = frame()
    empty = measure(arrays)
    assert empty.normal_loads == empty.slip_speeds == (0.0, 0.0)
    contact(arrays, 0, pad=12)
    result = measure(arrays)
    assert result.normal_loads == result.slip_speeds == (0.0, 0.0)


def test_invalid_entry_does_not_read_stale_payload_or_invent_a_measurement():
    arrays = frame()
    arrays["entry_meta"][0] = (0, 0, 9, 0, 0, 0)
    for name, value in arrays.items():
        if name != "entry_meta":
            value[:] = 7
    arrays["entry_clock"][:] = np.nan
    arrays["entry_contact_data"][:] = np.nan
    assert measure(arrays) is None
    assert measure({"entry_meta": arrays["entry_meta"]}) is None


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("entry_meta", (0, 0), 2),
        ("entry_meta", (0, 1), -1),
        ("entry_meta", (0, 2), 9),
        ("entry_meta", (0, 3), 5),
        ("entry_meta", (0, 4), 0),
        ("entry_meta", (0, 5), 1),
        ("entry_clock", (0, 0), np.nan),
        ("entry_clock", (0, 2), 0.25),
        ("entry_contact_ids", (0, 0, 0), 0),
        ("entry_contact_ids", (0, 0, 1), 2),
        ("entry_contact_ids", (0, 0, 2), -1),
        ("entry_contact_ids", (0, 0, 4), 9),
        ("entry_contact_ids", (0, 0, 6), -1),
        ("entry_contact_ids", (0, 0, 7), 2),
        ("entry_contact_ids", (0, 2, 0), 1),
        ("entry_contact_data", (0, 0, 1), np.inf),
        ("entry_contact_data", (0, 3, 1), np.nan),
        ("entry_contact_data", (0, 0, 5), 0),
        ("entry_contact_data", (0, 0, 5), 2),
        ("entry_contact_data", (0, 0, 13), -1),
        ("entry_contact_data", (0, 0, 16), 1),
        ("entry_body_linear_velocity", (0, 1, 0), np.nan),
        ("entry_body_angular_velocity", (0, 2, 0), np.inf),
        ("entry_body_mask", (0, 1), 0),
        ("entry_body_mask", (0, 2), 0),
    ],
)
def test_corrupt_entry_is_rejected_without_changing_any_array(field, index, value):
    arrays = frame()
    contact(arrays, 0)
    arrays[field][index] = value
    before = {name: item.tobytes() for name, item in arrays.items()}
    with pytest.raises(ValueError):
        measure(arrays)
    assert {name: item.tobytes() for name, item in arrays.items()} == before


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("entry_meta", np.zeros((1, 5), dtype=np.int32)),
        ("entry_clock", np.zeros((2, 4), dtype=np.float32)),
        ("entry_contact_ids", np.zeros((1, 4, 8), dtype=bool)),
        ("entry_contact_data", np.zeros((1, 3, 20), dtype=np.float32)),
        ("entry_body_linear_velocity", np.zeros((1, 3, 3), dtype=np.float32)),
        ("entry_body_mask", np.ones((1, 3), dtype=np.int32)),
    ],
)
def test_mismatched_shapes_and_noninteger_id_storage_are_rejected(field, replacement):
    arrays = frame()
    arrays[field] = replacement
    with pytest.raises(ValueError):
        measure(arrays)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"row": True},
        {"row": 1},
        {"object_body": 4},
        {"object_body": False},
        {"pad_geom_ids": (10, 10)},
        {"pad_geom_ids": (10, -1)},
        {"pad_geom_ids": (10, True)},
    ],
)
def test_invalid_indices_are_rejected(kwargs):
    with pytest.raises(ValueError):
        measure(frame(), **kwargs)


def test_readonly_inputs_and_frozen_owned_receipt():
    arrays = frame()
    contact(arrays, 0)
    for value in arrays.values():
        value.setflags(write=False)
    before = {name: value.tobytes() for name, value in arrays.items()}
    result = measure(arrays)
    with pytest.raises(FrozenInstanceError):
        result.time_s = 1.0
    with pytest.raises(TypeError):
        result.normal_loads[0] = 2.0
    assert {name: value.tobytes() for name, value in arrays.items()} == before


def test_geom_body_identity_cannot_change_between_contact_features():
    arrays = frame()
    contact(arrays, 0)
    contact(arrays, 1, pad_body=3)
    with pytest.raises(ValueError, match="geom changes body"):
        measure(arrays)


def test_selected_world_is_measured_without_reading_other_world_payloads():
    initial = frame()
    contact(initial, 0, load=0.5)
    arrays = {name: np.repeat(value, 2, axis=0) for name, value in initial.items()}
    arrays["entry_meta"][0] = (1, 9, 4, 0, 0, 1)
    for value in arrays.values():
        if value.dtype.kind == "f":
            value[0] = np.nan
    result = measure(arrays, row=1)
    assert result.normal_loads == (0.5, 0.0)
    assert result.slip_speeds == (0.0, 0.0)

"""HOST-only arithmetic contracts; no physical grasp or force guarantee."""

from dataclasses import FrozenInstanceError

import pytest

from phymeo.control.manipulation_grip import BoundedSlipClosure, SlipClosureReceipt


def controller(**kwargs):
    return BoundedSlipClosure((0.0, 0.0), (0.06, 0.06), **kwargs)


def test_first_call_rate_cap_and_deadband_match_hand_calculation():
    control = controller()
    first = control.update(0.0, (0.03, 0.03), (1.0, 1.0), (0.025, 0.020))
    assert isinstance(first, SlipClosureReceipt)
    assert first.dt == 0.0 and first.correction == (0.0, 0.0)
    second = control.update(0.005, (0.03, 0.03), (1.0, 1.0), (0.025, 0.020))
    # Rates: min(.02, 2*(.025-.015))=.02 and 2*(.020-.015)=.01 m/s.
    assert second.dt == 0.005
    assert second.correction == pytest.approx((0.0001, 0.00005))
    assert second.targets == pytest.approx((0.0301, 0.03005))
    third = control.update(0.010, (0.03, 0.03), (1.0, 1.0), (0.015, 0.014))
    assert third.correction == second.correction
    capped = control.update(0.015, (0.03, 0.03), (1.0, 1.0), (100.0, 100.0))
    assert capped.correction == pytest.approx((0.0002, 0.00015))


@pytest.mark.parametrize("loads", [(0.049, 1.0), (1.0, 0.049), (0.0, 0.0)])
def test_both_contacts_are_required(loads):
    control = controller()
    control.update(0.0, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
    held = control.update(0.01, (0.03, 0.03), loads, (1.0, 1.0))
    assert not held.contact_ready
    assert held.correction == (0.0, 0.0)
    at_threshold = control.update(0.02, (0.03, 0.03), (0.05, 0.05), (1.0, 1.0))
    assert at_threshold.contact_ready
    assert at_threshold.correction == pytest.approx((0.0002, 0.0002))


def test_contact_loss_holds_disabled_clears_and_reenabled_starts_from_zero():
    control = controller()
    control.update(0.0, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
    active = control.update(0.005, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
    held = control.update(0.010, (0.03, 0.03), (0.0, 1.0), (100.0, 100.0))
    assert held.correction == active.correction
    disabled = control.update(0.015, (0.02, 0.025), (1.0, 1.0), (1.0, 1.0), False)
    assert not disabled.enabled
    assert disabled.targets == (0.02, 0.025)
    assert disabled.correction == (0.0, 0.0)
    resumed = control.update(0.020, (0.02, 0.025), (1.0, 1.0), (1.0, 1.0))
    assert resumed.correction == pytest.approx((0.0001, 0.0001))


def test_correction_bound_and_original_joint_headroom_are_independent():
    control = BoundedSlipClosure((0.01, -0.02), (0.06, 0.045), max_correction=0.0003)
    control.update(0.0, (0.03, 0.04495), (1.0, 1.0), (1.0, 1.0))
    first = control.update(0.01, (0.03, 0.04495), (1.0, 1.0), (1.0, 1.0))
    assert first.correction == pytest.approx((0.0002, 0.00005))
    second = control.update(0.02, (0.03, 0.04495), (1.0, 1.0), (1.0, 1.0))
    assert second.correction == pytest.approx((0.0003, 0.00005))
    assert second.targets == pytest.approx((0.0303, 0.045))
    trimmed = control.update(0.025, (0.0599, 0.045), (0.0, 0.0), (1.0, 1.0))
    assert trimmed.correction == pytest.approx((0.0001, 0.0))
    assert trimmed.targets == (0.06, 0.045)
    held = control.update(0.03, (0.03, 0.02), (0.0, 0.0), (1.0, 1.0))
    assert held.correction == trimmed.correction


@pytest.mark.parametrize(
    "change",
    [
        {"time_s": 0.005},
        {"time_s": 0.004},
        {"time_s": 0.02},
        {"time_s": float("inf")},
        {"time_s": True},
        {"base_targets": (0.061, 0.03)},
        {"base_targets": (-0.001, 0.03)},
        {"base_targets": (0.03,)},
        {"base_targets": {0.02, 0.03}},
        {"base_targets": {0.02: 0.03, 0.03: 0.04}},
        {"base_targets": ((0.03,), (0.03,))},
        {"base_targets": (True, 0.03)},
        {"normal_loads": (float("nan"), 1.0)},
        {"normal_loads": (True, 1.0)},
        {"slip_speeds": (0.02, float("inf"))},
        {"slip_speeds": (0.02, False)},
        {"enabled": 1},
    ],
)
def test_invalid_update_preserves_both_time_and_accumulated_correction(change):
    control = controller()
    reference = controller()
    for item in (control, reference):
        item.update(0.0, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
        item.update(0.005, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
    arguments = dict(
        time_s=0.010,
        base_targets=(0.03, 0.03),
        normal_loads=(1.0, 1.0),
        slip_speeds=(1.0, 1.0),
        enabled=True,
    )
    arguments.update(change)
    with pytest.raises(ValueError):
        control.update(**arguments)
    retry = control.update(0.010, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
    expected = reference.update(0.010, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
    assert retry == expected


def test_invalid_first_call_does_not_seed_time_and_receipts_are_owned_immutable_values():
    control = controller()
    with pytest.raises(ValueError):
        control.update(10.0, (0.03, 0.03), (float("nan"), 1.0), (0.02, 0.02))
    base, loads, slips = [0.03, 0.025], [1.0, 1.0], [0.025, 0.02]
    first = control.update(0.0, base, loads, slips)
    base[0], loads[0], slips[0] = 0.05, 0.0, 0.0
    assert first.dt == 0.0 and first.base_targets == (0.03, 0.025)
    assert first.normal_loads == (1.0, 1.0) and first.slip_speeds == (0.025, 0.02)
    with pytest.raises(FrozenInstanceError):
        first.dt = 1.0
    with pytest.raises(TypeError):
        first.targets[0] = 0.06
    with pytest.raises(TypeError):
        first.correction[0] = 0.01
    control.update(0.005, (0.03, 0.025), (1.0, 1.0), (1.0, 1.0))
    assert first.targets == (0.03, 0.025) and first.correction == (0.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower": (0.0, 0.06)},
        {"upper": (0.06, float("inf"))},
        {"lower": (False, 0.0)},
        {"max_correction": -0.001},
        {"slip_threshold": -0.01},
        {"gain": True},
        {"max_rate": float("nan")},
        {"min_normal": -0.01},
        {"max_dt": 0.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    arguments = dict(lower=(0.0, 0.0), upper=(0.06, 0.06))
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        BoundedSlipClosure(**arguments)


def test_zero_gain_and_zero_correction_disable_accumulation():
    for control in (
        controller(gain=0.0),
        controller(max_correction=0.0),
        controller(max_rate=0.0),
    ):
        control.update(0.0, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
        receipt = control.update(0.01, (0.03, 0.03), (1.0, 1.0), (1.0, 1.0))
        assert receipt.targets == (0.03, 0.03)
        assert receipt.correction == (0.0, 0.0)

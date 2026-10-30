"""Actual pinned native finite-gripper trajectories; executed only by serial grant."""

import pytest
from test_gripper_cloth import assert_rebuild_equal, assert_run_physics, observe

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("case", ["hold", "slip", "capture"])
@pytest.mark.parametrize("h", [1 / 8, 1 / 16])
def test_actual_windows_and_complete_rebuild(tyx_mode, case, h):
    first = observe(case, tyx_mode, h, 1 / 2)
    second = observe(case, tyx_mode, h, 1 / 2)
    assert_run_physics(first)
    assert_run_physics(second)
    assert_rebuild_equal(first, second)
    if case == "capture":
        impact_index = 2 if h == 1 / 8 else 5
        at_capture = first.records[impact_index].state.domains["nodes"].position[:, 2]
        final = first.final_state.domains["nodes"].position[:, 2]
        assert (final > at_capture).all()
        assert (first.final_state.domains["nodes"].velocity[:, 2] > 0).all()


@pytest.mark.parametrize("h", [1 / 4, 1 / 8])
def test_actual_active_distance_material(tyx_mode, h):
    first = observe("material", tyx_mode, h, h)
    second = observe("material", tyx_mode, h, h)
    assert_run_physics(first)
    assert_run_physics(second)
    assert_rebuild_equal(first, second)


@pytest.mark.parametrize("case", ["hold", "slip"])
def test_actual_scaled_mass_and_finite_patch_control(tyx_mode, case):
    for reverse in (False, True):
        run = observe(
            case,
            tyx_mode,
            1 / 8,
            1 / 4,
            mass_scale=2.0,
            patch_half_extent_m=(3.0, 5.0),
            vertical_shift_m=3 / 8,
            reverse_declarations=reverse,
        )
        assert_run_physics(run)


@pytest.mark.parametrize("case,substeps", [("hold", 3), ("capture", 4)])
@pytest.mark.parametrize("failure", ["terminal", "close"])
def test_actual_late_fault_rollback_and_fresh_retry(tyx_mode, monkeypatch, case, substeps, failure):
    from test_gripper_cloth_transaction import require_failed_retry_fresh

    require_failed_retry_fresh(monkeypatch, case, tyx_mode, failure, 2, substeps)

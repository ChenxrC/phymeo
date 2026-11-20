"""Independent checks on the original RR input states and the staged CUDA solve."""

import json

import numpy as np
import pytest


def test_archived_control_reader_and_rollout_have_explicit_entry_boundaries(tmp_path):
    from experiments.area_pressure.pusht_entry_rows import iter_controls
    from experiments.area_pressure.pusht_entry_working_set import advance_entries

    start = dict(tick=750, time=0.75, q=np.zeros(9), v=np.zeros(8))
    with pytest.raises(ValueError, match="tick"):
        advance_entries(None, start, [{"tick": 751}], lambda record: None)
    with pytest.raises((FileNotFoundError, ValueError)):
        list(iter_controls(tmp_path, 750, 753))


def test_reference_distinguishes_touch_and_records_missing_legacy_controls():
    from types import SimpleNamespace

    from experiments.area_pressure.pusht_entry_window_reference import _controls_match, _separation

    steps = [
        {"tick": tick, "quadrature": {"clearance_m": gap}}
        for tick, gap in enumerate((-0.01, 0.0, -0.02, 0.0, 0.01))
    ]
    assert _separation(steps)["transition_input_ticks"] == [4]
    model = SimpleNamespace(nu=2, nv=8, nbody=5)
    channels = {
        "ctrl": np.array([0.5, -1.0]),
        "qfrc_applied": np.zeros(8),
        "xfrc_applied": np.zeros((5, 6)),
    }
    command = {**channels, "tick": 750, "trace_line_sha256": "0" * 64}
    assert _controls_match({}, command, model)["values"] == "legacy_values_not_recorded"
    saved = {"entry_control": channels, "trace_line_sha256": command["trace_line_sha256"]}
    assert _controls_match(saved, command, model)["values"] == "matched_recorded_values"
    with pytest.raises(ValueError, match="differs"):
        _controls_match(
            {**saved, "entry_control": {**channels, "ctrl": [0.5, 1.0]}}, command, model
        )
    with pytest.raises(ValueError, match="binding differs"):
        _controls_match({**saved, "trace_line_sha256": "1" * 64}, command, model)


def test_original_row_loader_has_named_entries():
    from experiments.area_pressure.pusht_entry_rows import load_entries

    entries = load_entries()
    assert [
        entries[label]["tick"]
        for label in ("initial", "first_load", "peak", "last_push_entry", "withdrawn")
    ] == [0, 750, 1255, 1699, 1767]


def test_area_response_requires_same_point_on_both_bodies():
    from experiments.area_pressure.pusht_entry_area import area_response

    area = dict(
        points=np.zeros((1, 3)),
        normals=np.array([[1.0, 0.0, 0.0]]),
        weights=np.array([0.25]),
        stiffness=np.array([100.0]),
        initial_gap=np.array([-0.01]),
        jac=np.array([[[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]]),
    )
    generalized, forces = area_response(area, np.array([0.1, 0.0]), 0.01)
    np.testing.assert_allclose(forces, [[0.9, 0, 0]])
    np.testing.assert_allclose(generalized, [0.9, -0.9])


@pytest.mark.cuda
def test_native_device_regularized_working_set():
    from experiments.area_pressure.native_row_device import NativeRowDevice

    m = np.eye(8)
    jac = np.vstack((m[0], -m[0], m[1]))
    with NativeRowDevice(m, jac, np.ones(3), np.array([3.0, -5.0, 2.0])) as device:
        a, forces, diagnostics = device.evaluate(np.zeros(8))
    np.testing.assert_allclose(a[:2], [1.5, 1.0])
    np.testing.assert_allclose(forces, [1.5, 0.0, 1.0])
    assert diagnostics["iterations"] >= 1


@pytest.mark.cuda
def test_continuation_uses_its_own_published_q_and_v():
    import mujoco as mj

    from experiments.area_pressure.pusht_entry_rows import load_entries
    from experiments.area_pressure.pusht_entry_working_set import advance_entries
    from phymeo.training.manipulation_tasks import manipulation_scenes

    model = mj.MjModel.from_xml_string(
        next(s.mjcf for s in manipulation_scenes() if s.key == "pusht-planar")
    )
    first = load_entries()["first_load"]
    controls = [{**first, "tick": first["tick"] + i} for i in range(3)]
    seen = []
    final = advance_entries(model, first, controls, seen.append)
    assert final["tick"] == 753 and len(seen) == 3
    for old, new in zip(seen, seen[1:]):
        np.testing.assert_array_equal(new["entry_q"], old["next_q"])
        np.testing.assert_array_equal(new["entry_v"], old["velocity"])
    for saved in seen:
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(saved["entry_control"][name], first[name])
        assert saved["trace_line_sha256"] == first["trace_line_sha256"]
    assert all(s["area_normal_n"] > 0 and s["support_normal_n"] > 0 for s in seen)
    assert seen[-1]["quadrature"]["clearance_m"] != seen[0]["quadrature"]["clearance_m"]


@pytest.mark.cuda
def test_both_hinge_stops_and_table_rows_keep_original_identities():
    import mujoco as mj

    from experiments.area_pressure.pusht_entry_rows import load_entries
    from experiments.area_pressure.pusht_entry_working_set import _entry_solution
    from phymeo.training.manipulation_tasks import manipulation_scenes

    model = mj.MjModel.from_xml_string(
        next(s.mjcf for s in manipulation_scenes() if s.key == "pusht-planar")
    )
    entry = load_entries()["withdrawn"]
    q = entry["q"].copy()
    q[:2] = [model.jnt_range[0, 1] + 0.002, model.jnt_range[1, 0] - 0.002]
    result = _entry_solution(model, {**entry, "q": q, "ctrl": np.zeros(2), "v": np.zeros(8)})
    assert result["native_limit_rows"] == 2 and result["support_normal_n"] > 0
    stops = {
        row["source_id"]: force
        for row, force in zip(result["owners"], result["native_forces"], strict=True)
        if row["kind"] == "limit"
    }
    assert set(stops) == {0, 1} and all(force > 0 for force in stops.values())


@pytest.mark.cuda
def test_original_five_entry_common_solutions(tmp_path):
    from experiments.area_pressure.pusht_entry_working_set import run_comparison

    report = run_comparison()
    (tmp_path / "rr-entry-comparison.json").write_text(
        json.dumps(report, allow_nan=False), encoding="utf-8"
    )
    assert report["nv"] == 8 and report["retained_candidates"] == 9
    frames = report["frames"]
    for label in ("initial", "first_load", "peak", "last_push_entry", "withdrawn"):
        item = frames[label]
        assert item["max_velocity_error"] < 5e-5
        assert item["max_native_row_force_error"] < 0.002
        assert item["scaled_momentum_residual"] < 2e-5
        assert item["native_kkt"] <= 2e-5
        assert item["outer_iterations"] <= 24
    assert frames["initial"]["area_points"] == frames["withdrawn"]["area_points"] == 0
    for label in ("first_load", "peak", "last_push_entry"):
        assert frames[label]["area_normal_n"] > 0
        assert frames[label]["support_normal_n"] > 0
        assert frames[label]["max_velocity_change_vs_no_area"] > 0
    assert frames["peak"]["native_rows"] > 8
    assert frames["peak"]["max_velocity_change_vs_no_area"] > 1e-8

"""The display separates a physical impulse from a force and preserves its sampling window."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "examples/export_compliant_replay.py"
spec = importlib.util.spec_from_file_location("replay_impulses", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def capture():
    frames = [dict(t=0, contacts={})]
    samples = []
    backbone = []
    impulses = ([0.002, 0.0], [0.0, 0.001], [0.0, 0.0], [0.0, 0.0])
    for i, row in enumerate(impulses):
        t = i / 128
        samples.extend([dict(t=t, patch="table"), dict(t=t, patch="jaw", backing_impulse_ns=row)])
        backbone.append(dict(t=t, total_impulse_ns=sum(row)))
        if i % 2 == 1:
            frames.append(
                dict(
                    t=(i + 1) / 128,
                    force_sample_s=t,
                    contacts={
                        "jaw": {"backing_impulse_ns": row.copy(), "normal_force_n": [2.0, 2.0]}
                    },
                )
            )
    return dict(
        dt_s=1 / 64,
        substeps=2,
        frames=frames,
        samples=samples,
        backing_samples=backbone,
        patches=[dict(name="table"), dict(name="jaw", backing_clearance_m=0.000125)],
    )


def test_window_counts_all_substeps_and_keeps_newtons_and_newton_seconds_distinct():
    data = capture()
    replay.add_backing_windows(data)
    assert "backing_window" not in data["frames"][0]
    event = data["frames"][1]["backing_window"]
    assert event["start_s"] == 0 and event["end_s"] == 1 / 64
    assert event["substep_s"] == 1 / 128
    assert event["impulse_ns_by_patch"] == pytest.approx({"jaw": 0.003})
    assert event["active_substeps"] == 2
    assert data["frames"][1]["contacts"]["jaw"]["backing_impulse_ns"] == [0.0, 0.001]
    assert data["frames"][1]["contacts"]["jaw"]["normal_force_n"] == [2.0, 2.0]
    assert data["frames"][2]["backing_window"]["active_substeps"] == 0


@pytest.mark.parametrize("fault", ["negative", "bad-sum", "missing", "frame", "time", "force"])
def test_inconsistent_backing_input_cannot_be_exported_as_a_valid_impulse(fault):
    data = capture()
    if fault == "negative":
        data["samples"][1]["backing_impulse_ns"][0] = -0.002
    elif fault == "bad-sum":
        data["backing_samples"][0]["total_impulse_ns"] += 0.01
    elif fault == "missing":
        data["samples"].pop(1)
    elif fault == "frame":
        data["frames"][1]["contacts"]["jaw"]["backing_impulse_ns"][0] = 0.001
    elif fault == "time":
        data["samples"][1]["t"] += 1 / 128
    else:
        data["frames"][1]["contacts"]["jaw"]["normal_force_n"][1] = 0.0
    with pytest.raises(ValueError, match="backing"):
        replay.add_backing_windows(data)


def test_legacy_capture_keeps_its_original_force_only_frame():
    data = {"frames": [dict(t=0.0)]}
    replay.add_backing_windows(data)
    assert data == {"frames": [dict(t=0.0)]}

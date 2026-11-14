"""Saved-consumer rejection tests, including nonfinite published observations."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import mujoco as mj
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "measure_saved_wrenches", Path(__file__).parents[1] / "scripts/measure_manipulation_wrenches.py"
)
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)

XML = """<mujoco><worldbody>
<geom name="left-pad" type="sphere" size=".1" pos="-1 0 0"/>
<geom name="right-pad" type="sphere" size=".1" pos="1 0 0"/>
<body name="object"><freejoint/><inertial pos=".25 0 0" mass="1" diaginertia=".1 .1 .1"/>
<geom name="shape" type="sphere" size=".1" pos=".25 0 0"/></body>
<body name="other"><geom type="sphere" size=".1" pos="0 0 1"/></body>
</worldbody></mujoco>"""


def fixture_records():
    model = mj.MjModel.from_xml_string(XML)
    body, left, obj = model.body("object").id, model.geom("left-pad").id, model.geom("shape").id
    ids = [[[1, 1, left, obj, 0, body, 0, 1]]]
    values = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
    source = dict(
        entry_meta=[[1, 0, 1, 1, 1, 0]],
        entry_clock=[[0, 0, 0.001, 0]],
        entry_contact_ids=ids,
        entry_contact_data=[[values]],
        entry_body_mask=[[1] * model.nbody],
        entry_body_position=np.zeros((1, model.nbody, 3)).tolist(),
    )
    frame = dict(
        world=0,
        tick=0,
        time_s=0,
        phase="accepted-entry",
        object_key="object",
        object_position=[0, 0, 0],
        object_wxyz=[2**-0.5, 0, 0, 2**-0.5],
        finger_normal_forces_n=[2, 0],
    )
    first = dict(
        step=1,
        feedback_source=dict(entry_meta=[[0, 0, 0, 0, 0, 0]]),
        success_frames={"0": frame},
        contact_ids=ids,
        contact_data=[[values]],
        contact_world_wrench=[[[2, 0, 0, 0, 0, 0]]],
        contact_body_wrench=[[[[-2, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, -2]]]],
    )
    return [first, dict(step=2, feedback_source=source)]


def write_run(tmp_path, records):
    root = tmp_path / "run"
    root.mkdir()
    model_path = root / "source.xml"
    model_path.write_text(XML, encoding="utf-8")
    trace = root / "trace.jsonl.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as stream:
        for item in records:
            stream.write(json.dumps(item) + "\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                status="COMPLETE",
                steps=2,
                worlds=[dict(world=0, source_file="source.xml", source_sha256=digest(model_path))],
                trace_sha256=digest(trace),
            )
        ),
        encoding="utf-8",
    )
    return root


def forbid(*args, **kwargs):
    raise AssertionError("saved consumer must not run physics")


def test_nonzero_rotated_com_and_published_pair(tmp_path):
    root = write_run(tmp_path, fixture_records())
    output = tmp_path / "output"
    with patch.object(mj, "mj_forward", forbid), patch.object(mj, "mj_step", forbid):
        result = consumer.measure_run(root, output)
    assert result["measured_entries"] == 1
    assert result["unmeasured_entry_steps"] == [2]
    with gzip.open(output / "wrenches.jsonl.gz", "rt", encoding="utf-8") as stream:
        measured = json.loads(next(stream))["measurement"]
    np.testing.assert_allclose(measured["reference_com_world"], (0, 0.25, 0), atol=1e-15)
    np.testing.assert_allclose(measured["total_moment_com_world"], (0, 0, -1.5), atol=1e-15)
    assert measured["total_force_world"] == [2, 0, 0]


@pytest.mark.parametrize("field", ["world", "body", "loads"])
@pytest.mark.parametrize("bad", ["nan", "inf", "shape"])
def test_nonfinite_and_broadcastable_published_arrays_rejected(tmp_path, field, bad):
    records = fixture_records()
    value = float(bad) if bad != "shape" else 0.0
    if field == "world":
        records[0]["contact_world_wrench"] = (
            [[[value]]] if bad == "shape" else [[[value, 0, 0, 0, 0, 0]]]
        )
    elif field == "body":
        records[0]["contact_body_wrench"] = (
            [[[[value], [value]]]]
            if bad == "shape"
            else [[[[value, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, -2]]]]
        )
    else:
        records[0]["success_frames"]["0"]["finger_normal_forces_n"] = (
            [value] if bad == "shape" else [2, value]
        )
    root = write_run(tmp_path, records)
    with pytest.raises(ValueError, match="invalid published"):
        consumer.measure_run(root, tmp_path / "rejected")


def test_same_origin_other_body_cannot_borrow_task_orientation(tmp_path):
    root = write_run(tmp_path, fixture_records())
    with pytest.raises(ValueError, match="only binds the task object"):
        consumer.measure_run(root, tmp_path / "rejected", object_name="other")


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("object_key", "other", "task object identity"),
        ("phase", "endpoint", "phase mismatch"),
        ("tick", 1, "phase mismatch"),
    ],
)
def test_task_binding_and_phase_are_not_only_positions(tmp_path, field, value, reason):
    records = fixture_records()
    records[0]["success_frames"]["0"][field] = value
    root = write_run(tmp_path, records)
    with pytest.raises(ValueError, match=reason):
        consumer.measure_run(root, tmp_path / "rejected")


def current_records():
    import copy

    first, second = fixture_records()
    first["entry"] = copy.deepcopy(second["feedback_source"])
    for key, published in (
        ("entry_contact_wrench_world", "contact_world_wrench"),
        ("entry_contact_body_wrench", "contact_body_wrench"),
    ):
        first["entry"][key] = copy.deepcopy(first[published])
    count = len(first["entry"]["entry_body_mask"][0])
    matrices = np.broadcast_to(np.eye(3), (1, count, 3, 3)).copy()
    matrices[0, 1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    first["entry"]["entry_body_rotation"] = matrices.tolist()
    last = copy.deepcopy(first)
    last["step"] = 2
    last["entry"]["entry_meta"][0][1:3] = [1, 2]
    last["entry"]["entry_clock"][0] = [0.001, 0, 0.002, 0]
    last["success_frames"]["0"].update(tick=1, time_s=0.001)
    return [first, last]


def write_current_run(tmp_path, records):
    root = write_run(tmp_path, records)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(record_entry=True, recorded_steps=2)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_current_entries_cover_last_step_and_arbitrary_group(tmp_path):
    root = write_current_run(tmp_path, current_records())
    with patch.object(mj, "mj_forward", forbid), patch.object(mj, "mj_step", forbid):
        result = consumer.measure_run(root, tmp_path / "output", group_names=("left-pad",))
    assert result["measured_entries"] == 2
    assert result["last_entry_step"] == 2
    assert result["unmeasured_entry_steps"] == []
    assert result["entry_source"] == "current"
    with gzip.open(tmp_path / "output/wrenches.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert [r["measurement"]["input_tick"] for r in rows] == [0, 1]
    assert all(r["measurement"]["group_force_world"] == [2, 0, 0] for r in rows)
    np.testing.assert_allclose(rows[-1]["measurement"]["total_moment_com_world"], (0, 0, -1.5))


@pytest.mark.parametrize("fault", ["missing", "stale", "wrench", "contact"])
def test_current_entry_cannot_borrow_other_step(tmp_path, fault):
    rows = current_records()
    if fault == "missing":
        rows[-1]["entry"] = None
    elif fault == "stale":
        rows[-1]["entry"]["entry_meta"][0][1:3] = [0, 1]
    elif fault == "wrench":
        rows[-1]["entry"]["entry_contact_wrench_world"][0][0][0] = 3
    else:
        rows[-1]["entry"]["entry_contact_data"][0][0][13] = 3
    root = write_current_run(tmp_path, rows)
    with pytest.raises(ValueError):
        consumer.measure_run(root, tmp_path / "rejected")


def test_current_entry_partial_recording_is_not_complete(tmp_path):
    root = write_current_run(tmp_path, current_records())
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["recorded_steps"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete recorded"):
        consumer.measure_run(root, tmp_path / "rejected")

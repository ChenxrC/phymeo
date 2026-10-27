"""Replay must preserve verified experiment data and reject incompatible records."""

import importlib.util
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phymeo.experiments.rigid_cloth import run_scene, save_run


def renderer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_rigid_cloth.py"
    spec = importlib.util.spec_from_file_location("render_rigid_cloth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def traces(tmp_path):
    pair = []
    for name, reaction in (("two_way", True), ("one_way", False)):
        run = run_scene(frames=2, reaction=reaction)
        destination = tmp_path / name
        save_run(run, destination)
        pair.append((run, destination))
    return pair


def test_replay_embeds_exact_traces_metrics_and_offline_controls(traces, tmp_path):
    output = tmp_path / "replay.html"
    renderer().render(traces[0][1], traces[1][1], output)
    html = output.read_text(encoding="utf-8")
    data = json.loads(
        re.search(r'<script id="scene-data" type="application/json">(.*?)</script>', html, re.S)[1]
    )
    assert data["schema"] == "phymeo.rigid_cloth.replay.v1"
    for name, (run, _) in zip(("two_way", "one_way"), traces, strict=True):
        record = data[name]
        for field in (
            "particle_position",
            "rigid_position",
            "rigid_velocity",
            "time",
            "triangles",
            "pinned",
        ):
            np.testing.assert_array_equal(record["arrays"][field], run.arrays[field])
        assert record["metrics"] == run.metrics
        assert record["config"] == run.config
        assert len(record["source"]["source_sha256"]) == 64
    assert 'id="play"' in html and 'id="timeline"' in html
    assert 'id="view-mode"' in html
    assert "<script src=" not in html and "fetch(" not in html


@pytest.mark.parametrize("damage", ["shape", "topology", "metrics", "horizon", "interaction"])
def test_replay_rejects_hash_valid_but_incompatible_or_invalid_trace(traces, tmp_path, damage):
    run, _ = traces[1]
    arrays = {name: value.copy() for name, value in run.arrays.items()}
    config, metrics = dict(run.config), dict(run.metrics)
    if damage == "shape":
        arrays["rigid_position"] = arrays["rigid_position"][:-1]
    elif damage == "topology":
        arrays["triangles"][0, 0] = 99
    elif damage == "metrics":
        metrics["contact_substeps"] += 1
    elif damage == "horizon":
        config["requested_frame_dt"] *= 2
    else:
        config["interaction"] = "two_way_dynamic"
    from phymeo.experiments.rigid_cloth import trajectory_digest

    metrics["trajectory_sha256"] = trajectory_digest(arrays)
    damaged = tmp_path / "damaged"
    save_run(replace(run, arrays=arrays, config=config, metrics=metrics), damaged)
    output = tmp_path / "bad.html"
    with pytest.raises(ValueError):
        renderer().render(traces[0][1], damaged, output)
    assert not output.exists()


def test_replay_escapes_embedded_metadata_and_refuses_overwrite(traces, tmp_path):
    path = traces[0][1] / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["provenance"]["platform"] = "</script><script>alert(1)</script>"
    path.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "replay.html"
    renderer().render(traces[0][1], traces[1][1], output)
    html = output.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    with pytest.raises(FileExistsError):
        renderer().render(traces[0][1], traces[1][1], output)

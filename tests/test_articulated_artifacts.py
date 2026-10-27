"""Offline replay and acceptance consumers must preserve actual trace data."""

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from phymeo.experiments.articulated import run_articulated_scene, save_articulated_run
from phymeo.experiments.prescribed_articulated import run_prescribed_scene, save_prescribed_run


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def traces(tmp_path):
    root, runs = tmp_path / "runs", {}
    for scene in ("S2", "S3", "S4"):
        runs[scene] = run_articulated_scene(scene=scene, frames=2)
        save_articulated_run(runs[scene], root / scene)
    for scene in ("S2", "S3"):
        name = scene + "-driver"
        runs[name] = run_prescribed_scene(scene=scene, frames=2)
        save_prescribed_run(runs[name], root / name)
    return root, runs


def test_replay_contains_exact_three_domain_and_driver_data(traces, tmp_path):
    root, runs = traces
    output = tmp_path / "replay.html"
    script("render_articulated").render(root, output)
    html = output.read_text(encoding="utf-8")
    payload = json.loads(
        re.search(r'<script id="scene-data" type="application/json">(.*?)</script>', html, re.S)[1]
    )
    assert payload["schema"] == "phymeo.articulated.replay.v1"
    assert set(payload["scenes"]) == {"S2", "S3", "S4"}
    assert payload["complete_horizon"] is False
    for scene, groups in payload["scenes"].items():
        for mode, record in groups.items():
            name = scene + "-driver" if mode == "prescribed" else scene
            assert record["config"] == runs[name].config
            assert record["metrics"] == runs[name].metrics
            assert record["arrays"]
            for field, value in record["arrays"].items():
                np.testing.assert_array_equal(value, runs[name].arrays[field])
            if "soft" in runs[name].config["node_domains"]:
                surface = np.array(record["soft_surface"])
                assert surface.shape == (32, 3)
                assert surface.min() >= 0 and surface.max() < 18
    for control in ("play", "step", "reset", "timeline", "scene-select", "clock"):
        assert f'id="{control}"' in html
    assert "fetch(" not in html and "<script src=" not in html


@pytest.mark.parametrize("damage", ["source", "summary", "config", "timeline"])
def test_replay_rejects_incompatible_sources_and_physics_metadata(traces, tmp_path, damage):
    root, _ = traces
    path = root / "S2-driver" / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if damage == "source":
        data["provenance"]["source_sha256"] = "0" * 64
    elif damage == "summary":
        data["metrics"]["contact_substeps"]["cloth"] += 1
    elif damage == "config":
        data["config"]["robot"]["mass"] = 99
    else:
        data["config"]["frames"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "bad.html"
    with pytest.raises(ValueError):
        script("render_articulated").render(root, output)
    assert not output.exists()


def test_replay_escapes_metadata_and_refuses_overwriting(traces, tmp_path):
    root, _ = traces
    # Keep the machine environment identical across records while checking text escaping.
    for directory in root.iterdir():
        path = directory / "run.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["provenance"]["platform"] = "</script><script>alert(1)</script>"
        path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "replay.html"
    script("render_articulated").render(root, output)
    assert "</script><script>alert(1)</script>" not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        script("render_articulated").render(root, output)


def test_acceptance_verifier_refuses_short_smoke_as_complete(traces):
    root, _ = traces
    with pytest.raises(ValueError, match="complete|200"):
        script("verify_articulated_runs").verify(root, root)


@pytest.mark.parametrize("source,target", [("S2", "S4"), ("S3", "S2"), ("S3-driver", "S2-driver")])
def test_replay_rejects_a_valid_scene_in_the_wrong_named_slot(traces, tmp_path, source, target):
    root, _ = traces
    shutil.copytree(root / source, root / target, dirs_exist_ok=True)
    with pytest.raises(ValueError, match="variant"):
        script("render_articulated").render(root, tmp_path / "wrong-scene.html")


def test_repeat_verifier_binds_required_variants_before_certifying_them(
    traces, tmp_path, monkeypatch
):
    root, _ = traces
    shutil.copytree(root / "S2", root / "S4", dirs_exist_ok=True)
    repeat = tmp_path / "repeat"
    shutil.copytree(root, repeat)
    module = script("verify_articulated_runs")
    # Isolate the named-slot contract using real validated small traces. Complete
    # horizon enforcement is tested separately; do not run ten full scenes here.
    two, driver = module.load_articulated_run, module.load_prescribed_run
    monkeypatch.setattr(module, "load_articulated_run", lambda path, **_: two(path))
    monkeypatch.setattr(module, "load_prescribed_run", lambda path, **_: driver(path))
    with pytest.raises(ValueError, match="variant"):
        module.verify(root, repeat)


def test_benchmark_refuses_wrong_variant_before_timing(traces, tmp_path, monkeypatch):
    root, _ = traces
    shutil.copytree(root / "S4", root / "S2", dirs_exist_ok=True)
    module = script("measure_articulated")
    load = module.load_articulated_run
    monkeypatch.setattr(module, "load_articulated_run", lambda path, **_: load(path))

    def must_not_measure(*args, **kwargs):
        raise AssertionError("a wrong-scene record reached the timer")

    monkeypatch.setattr(module, "measure", must_not_measure)
    monkeypatch.setattr(
        sys,
        "argv",
        ["measure_articulated", "--input", str(root), "--output", str(tmp_path / "out.json")],
    )
    with pytest.raises(ValueError, match="variant"):
        module.main()

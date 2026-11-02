"""Replays of a new robot family bind geometry from the trace without a case factory."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phymeo.experiments.compliant_cloth import trace_start
from phymeo.experiments.compliant_handoff import build_handoff

script = Path(__file__).resolve().parents[1] / "examples/export_compliant_replay.py"
spec = importlib.util.spec_from_file_location("replay", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unknown_case_restores_links_from_serialized_rotating_robot():
    data = trace_start(build_handoff(robot="articulated"), "user_scene_without_factory")
    for robot in data["frames"][0]["robots"].values():
        del robot["links_m"]
    del data["robot_parents"]
    normals = {key: row["normal"].copy() for key, row in data["frames"][0]["contacts"].items()}
    module.add_link_geometry(data)
    assert data["robot_parents"]["A"] == [-1, 0, 1, 1]
    for arm in ("A", "B"):
        finger = data["frames"][0]["robots"][arm]["links_m"][2]
        pad = np.array(data["frames"][0]["patch_vertices_m"][arm + "_left"])
        np.testing.assert_allclose(pad.mean(axis=0), finger, atol=2e-15)
    for key, n in normals.items():
        assert data["frames"][0]["contacts"][key]["normal"] == n


@pytest.mark.parametrize("field", ["links", "patch", "parents"])
def test_geometry_mismatch_is_rejected_even_with_full_record(field):
    data = trace_start(build_handoff(robot="articulated"), "user_scene_without_factory")
    if field == "links":
        data["frames"][0]["robots"]["B"]["links_m"][2][0] += 0.02
    elif field == "patch":
        data["frames"][0]["patch_vertices_m"]["B_left"][1][0] += 0.02
    else:
        data["robot_parents"]["B"][3] = 0
    with pytest.raises(ValueError, match="recorded"):
        module.add_link_geometry(data)


@pytest.mark.parametrize("tamper", ["negative_alias", "omitted_patch", "duplicate_patch"])
def test_recorded_attachment_set_cannot_be_aliased_or_dropped(tamper):
    data = trace_start(build_handoff(robot="articulated"), "new_scene")
    patch = next(p for p in data["patches"] if p["name"] == "B_left")
    if tamper == "negative_alias":
        patch["link_index"] = -2  # Python indexing would alias the real link 2.
    elif tamper == "omitted_patch":
        data["patches"].remove(patch)
    else:
        data["patches"].append(patch.copy())
    with pytest.raises((ValueError, TypeError)):
        module.add_link_geometry(data)


def test_export_gzip_binds_input_bytes_and_preserves_per_site_normals(tmp_path):
    import gzip
    import hashlib
    import json

    trace = trace_start(
        build_handoff(robot="articulated", fabric="soft", finger_pad_radius_m=0.002), "finite-pad"
    )
    raw = gzip.compress(json.dumps(trace, allow_nan=False).encode())
    source = tmp_path / "actual.json.gz"
    source.write_bytes(raw)
    output = tmp_path / "motion.html"
    assert module.export(source, output) == hashlib.sha256(raw).hexdigest()
    html = output.read_text(encoding="utf8")
    capture = html.split('<script id="capture" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    cap = json.loads(capture)
    jaw = cap["trajectory"]["frames"][0]["contacts"]["A_left"]
    assert jaw["site_normals"] == trace["frames"][0]["contacts"]["A_left"]["site_normals"]
    assert cap["trajectory"]["patches"][1]["radius_m"] == 0.002
    assert "roundedPadMesh(" in html


@pytest.mark.parametrize("tamper", ["missing", "nonunit", "force"])
def test_finite_pad_replay_requires_the_actual_contact_direction(tamper):
    trace = trace_start(build_handoff(robot="articulated", finger_pad_radius_m=0.002), "finite-pad")
    site = trace["frames"][0]["contacts"]["A_left"]
    if tamper == "missing":
        del site["site_normals"]
    elif tamper == "nonunit":
        site["site_normals"][0] = [0, 0, 0]
    else:
        site["normal_force_n"][0] += 1.0
    with pytest.raises(ValueError, match="contact"):
        module.add_link_geometry(trace)

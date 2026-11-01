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

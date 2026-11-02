"""Tie displayed link forces to the material and link coordinates of the actual force sample."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_cloth_link_contact import make_scene

import phymeo as ph
from phymeo.experiments.compliant_cloth import Case, record_step, trace_start

path = Path(__file__).resolve().parents[1] / "examples/export_compliant_replay.py"
spec = importlib.util.spec_from_file_location("link_replay", path)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def actual_capture():
    prog, state, inputs = make_scene(revolute=True, friction=0.5)
    state.domains["arm"].v[0] = 0.25
    case = Case(prog, state, inputs, 1 / 256, 1)
    trace = trace_start(case, "articulated-link-shaft")
    with prog.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as stepper:
        out = stepper.step(state, inputs, case.dt)
    assert record_step(case, trace, out, 0.0)
    return trace


def test_source_reconstructs_the_same_sample_points_for_face_axis_force_and_power():
    tr = actual_capture()
    frame = tr["frames"][1]
    (r,) = frame["robot_contact"]["cloth_links"]
    assert r["link"] == "beam" and r["face_nodes"] == [0, 1, 2]
    assert len(r["sites"]) >= 1
    for site in r["sites"]:
        weights = dict(site["face_weights"])
        p = sum(
            weights[n] * np.array(x)
            for n, x in zip(r["face_nodes"], r["face_positions_m"], strict=True)
        )
        np.testing.assert_allclose(site["point_m"], p, atol=1e-12)
        assert np.dot(site["force_on_cloth_n"], site["relative_velocity_m_s"]) == pytest.approx(
            site["power_w"]
        )
    exporter.add_link_geometry(tr)


@pytest.mark.parametrize("fault", ["force", "point", "bound", "link", "axis", "power"])
def test_corrupt_structural_force_cannot_be_rendered_as_valid(fault):
    tr = copy.deepcopy(actual_capture())
    row = tr["frames"][1]["robot_contact"]["cloth_links"][0]
    site = row["sites"][0]
    if fault == "force":
        site["force_on_link_n"][0] += 0.1
    elif fault == "point":
        site["point_m"][0] += 0.1
    elif fault == "bound":
        site["face_weights"][0][1] = 1.2
    elif fault == "link":
        row["link"] = "missing"
    elif fault == "axis":
        site["axis_parameter"] = 1.5
    else:
        site["power_w"] += 0.1
    with pytest.raises(ValueError, match="structural"):
        exporter.add_link_geometry(tr)

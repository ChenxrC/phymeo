"""Self-contact forces in the viewer must come from the same sample as the two material sites."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_compliant_self_contact import setup_sheet

import phymeo as ph
from phymeo.experiments.compliant_cloth import Case, record_step, trace_start
from phymeo.experiments.compliant_handoff import build_handoff

path = Path(__file__).resolve().parents[1] / "examples/export_compliant_replay.py"
spec = importlib.util.spec_from_file_location("cloth_self_replay", path)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def capture():
    program, state, inputs = setup_sheet()
    case = Case(program, state, inputs, 1 / 128, 2)
    trace = trace_start(case, "connected-fold")
    with program.stepper(
        ph.StepSettings(substeps=2, position_iterations=1, velocity_iterations=1)
    ) as rt:
        out = rt.step(state, inputs, case.dt)
    assert record_step(case, trace, out, 0.0)
    return trace


def test_actual_two_sided_material_sample_is_saved_and_bound_to_the_displayed_frame():
    trace = capture()
    assert len(trace["self_samples"]) == 2
    assert trace["frames"][1]["self_contact"] == trace["self_samples"][-1]
    assert trace["frames"][1]["force_sample_s"] == trace["self_samples"][-1]["t"]
    (row,) = trace["frames"][1]["self_contact"]["contacts"]
    initial, = trace["frames"][0]["self_contact"]["contacts"]
    assert len(initial["sites"]) == 3
    assert row["faces"] == ["face_0", "face_1"] and row["sites"]
    assert sum(site["fraction"] for site in row["sites"]) == pytest.approx(1.0)
    site = row["sites"][0]
    assert site["first_point_m"] != site["second_point_m"]
    np.testing.assert_allclose(np.add(site["force_on_first_n"], site["force_on_second_n"]), 0)
    exporter.check_self_records(trace)


@pytest.mark.parametrize("corrupt", ["earlier_force", "material_point", "frame_binding", "owner"])
def test_export_rejects_damage_on_either_side_and_every_recorded_substep(corrupt):
    trace = copy.deepcopy(capture())
    sample = trace["self_samples"][0]
    row = sample["contacts"][0]
    if corrupt == "earlier_force":
        row["sites"][0]["force_on_first_n"][2] += 0.5
    elif corrupt == "material_point":
        row["sites"][0]["second_point_m"][1] += 0.05
    elif corrupt == "owner":
        row["first_nodes"][0] = 40
    else:
        trace["frames"][1]["self_contact"] = {**trace["frames"][1]["self_contact"], "t": 99.0}
    with pytest.raises(ValueError, match="self-contact"):
        exporter.check_self_records(trace)


def test_handoff_builder_registers_every_sheet_face_and_pair():
    case = build_handoff(robot="articulated-palm", fabric="soft", grid=(3, 2), self_contact=True)
    program = case.program
    obs = program.observe_self_contact(case.initial)
    assert obs["registered_pairs"] == 6 and obs["contacts"] == ()
    assert program.cloth_self.names == tuple(s.key for s in program.surfaces)

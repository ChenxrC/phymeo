"""HOST contracts only; these tests never load Newton or run a simulation."""

import math

import pytest

from experiments.contact_selection.kamino_reference import (
    case_spec,
    decode_contacts,
    external_wrench,
    summarize_records,
    validate_output,
)


def _raw(force=(2.0, 0.0, 3.0), torque=(0.0, -0.04, 0.0)):
    return {
        "count": 1,
        "shape0": [0],
        "shape1": [1],
        "point0": [[0.02, 0.0, 0.01]],
        "point1": [[0.02, 0.0, 0.11]],
        "normal": [[0.0, 0.0, -1.0]],
        "margin0": [0.0],
        "margin1": [0.0],
        "wrench0": [[*force, *torque]],
        "overflow": 0,
    }


def test_fixed_physical_case_and_owned_configuration():
    a = case_spec("padmm", 0.001)
    b = case_spec("dvi", 0.0005)
    assert a["mass_kg"] == 0.5
    assert a["box_half_extents_m"] == [0.02, 0.02, 0.01]
    assert a["table_half_extents_m"] == [0.04, 0.04, 0.005]
    assert a["initial_pose_xyzw"] == [0, 0, 0.011, 0, 0, 0, 1]
    assert a["steps"] == 600 and b["steps"] == 1200
    a["box_half_extents_m"][0] = 9
    assert case_spec("padmm", 0.001)["box_half_extents_m"][0] == 0.02
    for backend, dt in (("other", 0.001), ("dvi", True), ("padmm", 0.002)):
        with pytest.raises(ValueError):
            case_spec(backend, dt)


def test_real_external_wrench_uses_entry_pose_and_half_open_time():
    pose = [1, 2, 3, 0, 0, 0, 1]
    force, moment, point = external_wrench(0.2, pose)
    assert force == [4.0, 0.0, 0.0]
    assert moment == [0.0, 0.04, 0.0]
    assert point == [1.0, 2.0, 3.01]
    assert external_wrench(0.4, pose)[0] == [0.0, 0.0, 0.0]
    assert external_wrench(0.199, pose)[1] == [0.0, 0.0, 0.0]
    # A 90-degree rotation about x sends the local +z application arm to -y.
    q = math.sqrt(0.5)
    rotated = [1, 2, 3, q, 0, 0, q]
    f, m, p = external_wrench(0.3, rotated)
    assert f == force
    assert m == pytest.approx([0, 0, 0.04], abs=1e-14)
    assert p == pytest.approx([1, 1.99, 3], abs=1e-14)
    assert pose == [1, 2, 3, 0, 0, 0, 1]


@pytest.mark.parametrize("pose", ([0] * 7, [0] * 6, [0, 0, 0, 0, 0, 0, math.nan]))
def test_bad_external_pose_is_rejected(pose):
    with pytest.raises(ValueError):
        external_wrench(0.2, pose)


def test_point_side_wrenches_preserve_public_torque_and_static_origin_difference():
    poses = [[0, 0, 0.1, 0, 0, 0, 1]]
    raw = _raw()
    result = decode_contacts(poses, [[0, 0, 0]], [0, -1], raw)
    c = result[0]
    assert c["point0_world"] == pytest.approx([0.02, 0, 0.11])
    assert c["force0_world"] == [2, 0, 3]
    assert c["force1_world"] == [-2, 0, -3]
    assert c["public_wrench0_world_com"] == [2, 0, 3, 0, -0.04, 0]
    assert c["moment0_com_from_point"] == pytest.approx([0, -0.04, 0])
    assert c["moment1_com_from_point"] == pytest.approx([0, -0.16, 0])
    assert c["moment0_origin"] == pytest.approx([0, 0.16, 0])
    assert c["moment1_origin"] == pytest.approx([0, -0.16, 0])
    # When static is side 0, pinned Newton exports zero torque. Preserve it;
    # the physical moment about the requested common origin is still nonzero.
    swapped = _raw((-2, 0, -3), (0, 0, 0))
    swapped["shape0"], swapped["shape1"] = [1], [0]
    swapped["point0"], swapped["point1"] = raw["point1"], raw["point0"]
    swapped["normal"] = [[0, 0, 1]]
    s = decode_contacts(poses, [[0, 0, 0]], [0, -1], swapped)[0]
    assert s["public_wrench0_world_com"][3:] == [0, 0, 0]
    assert s["moment0_origin"] == pytest.approx([0, -0.16, 0])
    assert s["moment1_com_from_point"] == pytest.approx([0, -0.04, 0])
    result[0]["force0_world"][0] = 99
    assert raw["wrench0"][0][0] == 2


def test_contact_points_use_input_rotation_and_actual_com():
    q = math.sqrt(0.5)
    raw = _raw((0, 1, 0), (0, 0, 0))
    raw["point0"] = [[0.02, 0, 0]]
    raw["point1"] = [[1, 2.02, 3]]
    c = decode_contacts([[1, 2, 3, 0, 0, q, q]], [[0.01, 0, 0]], [0, -1], raw)[0]
    assert c["point0_world"] == pytest.approx([1, 2.02, 3])
    assert c["com0_world"] == pytest.approx([1, 2.01, 3])
    assert c["moment0_com_from_point"] == pytest.approx([0, 0, 0], abs=1e-14)
    assert c["moment0_origin"] == pytest.approx([-3, 0, 1])


@pytest.mark.parametrize("change", ("overflow", "shape", "nan", "short", "normal"))
def test_invalid_contact_publication_rejected(change):
    raw = _raw()
    if change == "overflow":
        raw["overflow"] = 1
    elif change == "shape":
        raw["shape1"] = [8]
    elif change == "nan":
        raw["wrench0"][0][1] = math.inf
    elif change == "short":
        raw["point0"] = []
    else:
        raw["normal"] = [[0, 0, 2]]
    with pytest.raises(ValueError):
        decode_contacts([[0, 0, 0.1, 0, 0, 0, 1]], [[0, 0, 0]], [0, -1], raw)


def _records():
    records = []
    for i in range(600):
        t = i / 1000
        # Manufactured metric trace, not simulated Newton evidence.
        held = i < 400
        z, x = (0.01, 0.0) if held else (-0.01, 0.07)
        rotation = [0, 0, 0, 1] if held else [0, math.sin(0.1), 0, math.cos(0.1)]
        records.append(
            {
                "input_tick": i,
                "output_tick": i + 1,
                "input_time": t,
                "output_time": (i + 1) / 1000,
                "phase": "step-input-contact",
                "entry": {"body_q": [[x, 0, z, *rotation]], "body_qd": [[0] * 6]},
                "exit": {"body_q": [[x, 0, z, *rotation]], "body_qd": [[0] * 6]},
                "contact_force_on_box": [0, 0, 4.905 if held else 0],
                "active_contacts": 4 if held else 0,
                "solver_status": [{"converged": True}],
            }
        )
    records[0]["entry"]["body_q"][0][2] = 0.011
    return records


def test_metrics_distinguish_physical_gates_nonconvergence_and_prefix():
    rows = _records()
    metrics = summarize_records(rows, 0.001, 600)
    assert metrics["settle_force_mean_n"] == pytest.approx(4.905)
    assert metrics["vertical_impulse_n_s"] == pytest.approx(1.962)
    assert metrics["last_loaded_contact_time_s"] == pytest.approx(0.399)
    assert metrics["functional_pass"] and metrics["solver_converged"]
    rows[160]["contact_force_on_box"][2] = 500
    assert not summarize_records(rows, 0.001, 600)["gates"]["settle_support"]
    rows[160]["contact_force_on_box"][2] = 4.905
    rows[100]["solver_status"][0]["converged"] = False
    assert not summarize_records(rows, 0.001, 600)["solver_converged"]
    assert not summarize_records(rows[:100], 0.001, 600)["functional_pass"]
    rows[40]["input_time"] = math.nan
    with pytest.raises(ValueError):
        summarize_records(rows, 0.001, 600)
    rows[40]["input_time"] = 0.04
    rows[50]["input_tick"] = 99
    with pytest.raises(ValueError):
        summarize_records(rows, 0.001, 600)


def test_output_requires_new_absolute_directory(tmp_path):
    with pytest.raises(ValueError):
        validate_output("relative-output")
    with pytest.raises(ValueError):
        validate_output(tmp_path)
    wanted = tmp_path / "fresh"
    assert validate_output(wanted) == wanted
    assert not wanted.exists()


def test_zero_contacts_do_not_consume_finite_stale_padding():
    raw = _raw()
    raw["count"] = 0
    assert decode_contacts([[0, 0, 0.1, 0, 0, 0, 1]], [[0, 0, 0]], [0, -1], raw) == []

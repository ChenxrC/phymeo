"""Predeclared selection rules plus optional archived original-state CUDA controls."""

from copy import deepcopy
from pathlib import Path

import pytest


def _row(tick):
    return dict(
        original_input_tick=tick,
        original_output_tick=tick + 1,
        controller={"stage": "entry"},
        input_state={"q": [tick], "v": [1.0], "time": tick * 0.001},
        control={
            "tick": tick,
            "output_tick": tick + 1,
            "ctrl": [0.2, -0.1],
            "qfrc_applied": [0.0],
            "xfrc_applied": [[0.0] * 6],
        },
    )


def test_selection_uses_first_declared_shallow_input_and_preserves_real_command():
    from experiments.area_pressure.pusht_lateral_controls import select_shallow

    rows = [_row(i) for i in range(3)]
    original = deepcopy(rows)
    depths = {0: 0.0, 1: 0.00011, 2: 0.0003}
    chosen, evidence = select_shallow(
        rows,
        lambda row: dict(
            depth_m=depths[row["original_input_tick"]], area_m2=1e-4, normals_match=True
        ),
    )
    assert chosen == rows[1] and evidence["depth_m"] == 0.00011
    assert chosen["control"] == rows[1]["control"] and rows == original


@pytest.mark.parametrize("fault", ["deep", "wrong-face", "unsupported"])
def test_selection_never_skips_a_bad_first_crossing_to_find_a_later_good_force(fault):
    from experiments.area_pressure.pusht_lateral_controls import select_shallow

    calls = []

    def measure(row):
        calls.append(row["original_input_tick"])
        if fault == "unsupported":
            raise ValueError("unsupported material gradient")
        return dict(
            depth_m=0.002 if fault == "deep" else 0.0002,
            area_m2=1e-4,
            normals_match=fault != "wrong-face",
        )

    with pytest.raises(ValueError):
        select_shallow([_row(1), _row(2)], measure)
    assert calls == [1]


@pytest.mark.cuda
def test_real_saved_faces4_and13_have_independent_public_lateral_controls(tmp_path):
    from experiments.area_pressure.pusht_lateral_controls import run

    folder = Path(__file__).resolve().parents[1] / "artifacts/rr-pose-followthrough-20261121-01"
    paths = [folder / f"face{i}.json" for i in (4, 13)]
    if not all(path.is_file() for path in paths):
        pytest.skip("requires separately archived actual A face4/face13 source records")
    report = run(*paths, tmp_path / "lateral-controls.json")
    assert report["status"] == "verified-short-public-controls"
    cases = report["cases"]
    for face in (4, 13):
        separate = cases[f"face{face}-separated"]
        assert separate["paired_input_exact"] is True
        assert all(v["normal_load_n"] == 0 for v in separate["variants"].values())
        live = cases[f"face{face}-shallow"]
        assert live["paired_input_exact"] is True
        assert 0.0001 <= live["selection"]["depth_m"] <= 0.001
        assert live["variants"]["all-lateral"]["normal_load_n"] > 0
        for variant in live["variants"].values():
            assert variant["receipt_tick"] == 1 and variant["initial_receipt"] is None
            assert variant["fp64_oracle_verified"] is True
            assert variant["opposite_reaction_verified"] is True
    new_face, old_face = cases["face4-shallow"], cases["face13-shallow"]
    assert new_face["variants"]["negative-y"]["normal_load_n"] == 0
    assert new_face["object_dynamic_difference"] > 0
    assert old_face["variants"]["negative-y"]["normal_load_n"] > 0
    assert old_face["old_side_preserved"] is True

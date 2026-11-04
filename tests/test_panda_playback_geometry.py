"""Shape selection for old and new offline Panda trajectories."""

from pathlib import Path
from runpy import run_path

import pytest

display_geometry = run_path(str(Path(__file__).parents[1] / "examples/export_panda_cube.py"))[
    "display_geometry"
]


def test_render_geometry_uses_physical_sizes_and_legacy_default():
    assert display_geometry({}) == {"kind": "box", "size_m": [0.05] * 3}
    box = [0.046, 0.058, 0.04]
    assert display_geometry({"cube_full_size_m": box}) == {"kind": "box", "size_m": box}
    described = {"payload_geometry": {"kind": "box", "full_size_m": box}}
    assert display_geometry(described) == {"kind": "box", "size_m": box}
    with pytest.raises(ValueError, match="disagree"):
        display_geometry({**described, "cube_full_size_m": [0.05] * 3})
    sphere = {"kind": "sphere", "radius_m": 0.025}
    assert display_geometry({"payload_geometry": sphere}) == {
        "kind": "sphere",
        "size_m": [0.05] * 3,
    }
    with pytest.raises(ValueError, match="disagree"):
        display_geometry({"cube_full_size_m": box, "payload_geometry": sphere})


@pytest.mark.parametrize(
    "run",
    [
        {"payload_geometry": {"kind": "sphere", "radius_m": -0.025}},
        {"payload_geometry": {"kind": "ellipsoid"}},
        {"cube_full_size_m": [0.04, float("nan"), 0.06]},
    ],
)
def test_invalid_display_geometry_is_rejected(run):
    with pytest.raises(ValueError):
        display_geometry(run)


def test_force_events_use_entry_pose_even_between_uniform_frames(tmp_path):
    import json
    from types import SimpleNamespace

    read_run = run_path(str(Path(__file__).parents[1] / "examples/export_panda_cube.py"))[
        "read_run"
    ]
    identity = [0, 0, 0, 1]
    rows = [
        {
            "t": (i + 1) * 0.01,
            "q": [],
            "cube": [i + 1, 0, 0],
            "xyzw": identity,
            "pad_normal_N": [0.0, 0.0],
            "table_N": 0.0,
            "structural_contacts": [],
        }
        for i in range(400)
    ]
    for index, force in [(0, 1), (11, 3), (12, 2)]:
        rows[index]["structural_contacts"] = [
            {"shape": "link", "point_world_m": [index, 0, 0], "force_on_payload_N": [0, force, 0]}
        ]
    data = {
        "steps": len(rows),
        "initial_arm_q": [],
        "initial_payload_position_m": [0, 0, 0],
        "initial_payload_xyzw": identity,
        "trace": rows,
    }
    file = tmp_path / "events.json"
    file.write_text(json.dumps(data), encoding="utf8")
    result = read_run(SimpleNamespace(collisions=()), file)
    first = result["frames"][result["jumps"]["first"]]
    peak = result["frames"][result["jumps"]["peak"]]
    assert first[0] == 0 and first[2] == [0, 0, 0] and first[6][0][2] == [0, 1, 0]
    assert peak[0] == rows[10]["t"] and peak[2] == [11, 0, 0] and peak[6][0][2] == [0, 3, 0]
    assert result["frames"][-1][0] == 4 and result["frames"][-1][4:6] == [None, None]
    assert result["load_time"] == "entry"
    del rows[0]["structural_contacts"]
    file.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError, match="incomplete"):
        read_run(SimpleNamespace(collisions=()), file)

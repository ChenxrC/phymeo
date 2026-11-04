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

"""An actuator-driven convex tray supports, transports and releases a freely evolving sheet."""

import gzip
import json

import numpy as np
import pytest


@pytest.mark.parametrize(
    "fabric,tool,friction",
    [
        ("soft", "rectangle", 0.75),
        ("soft", "rectangle", 0.0),
        ("reference", "taper", 0.75),
    ],
    ids=["soft", "zero-friction", "alternate"],
)
def test_actual_support_drag_release_and_ballistic_window(fabric, tool, friction, tmp_path):
    from phymeo.experiments.convex_tool_replay import write_tool_replay
    from phymeo.experiments.convex_tool_transfer import build_tool_transfer, run_tool_transfer

    case = build_tool_transfer(fabric=fabric, footprint=tool, friction=friction)
    trace = run_tool_transfer(case)
    assert len(trace["frames"]) > 1
    with gzip.open(tmp_path / "trajectory.json.gz", "wt", encoding="utf8") as stream:
        json.dump(trace, stream, allow_nan=False)
    write_tool_replay(trace, tmp_path / "replay.html")
    assert trace["status"] == "completed", trace["reason"]
    frames = trace["frames"]
    events = {e["kind"]: e for e in trace["events"]}
    assert (
        events["lifted"]["step"] < events["withdraw"]["step"] < events["independent_flight"]["step"]
    )
    assert (tmp_path / "replay.html").stat().st_size > 5000
    assert trace["qualified"] is (friction > 0)
    h = trace["dt_s"]
    model = trace["model"]
    mass = np.asarray(model["mass_kg"])
    total = mass.sum()
    g = np.asarray(model["gravity_m_s2"])
    assert total == pytest.approx(0.048) and np.linalg.norm(g) == 8
    lift = frames[events["lifted"]["step"]]
    pre = frames[events["withdraw"]["step"]]
    assert lift["com_m"][2] - frames[0]["com_m"][2] >= 0.012
    assert lift["cloth_contact_n"][2] >= 0.5 * total * 8
    horizontal = pre["com_m"][0] - lift["com_m"][0]
    if friction:
        assert horizontal >= 0.015
    else:
        assert abs(horizontal) < 0.0075
    free = events["independent_flight"]
    start = frames[free["start_step"]]
    terminal = frames[free["step"]]
    flight = frames[free["start_step"] + 1 : free["step"] + 1]
    assert len(flight) >= 32 and all(f["material_sites"] == 0 for f in flight)
    assert start["com_m"][2] - terminal["com_m"][2] >= 0.005
    for frame in flight:
        t = (frame["step"] - start["step"]) * h
        np.testing.assert_allclose(
            frame["com_velocity_m_s"],
            np.array(start["com_velocity_m_s"]) + g * t,
            atol=3e-11,
            rtol=0,
        )
        np.testing.assert_allclose(
            frame["com_m"],
            np.array(start["com_m"]) + np.array(start["com_velocity_m_s"]) * t + 0.5 * g * t * t,
            atol=3e-11,
            rtol=0,
        )
    local = np.array(model["tool_local_m"])
    base = np.array(model["base_m"])
    for before, current in zip(frames, frames[1:]):
        v0, v1 = np.array(before["node_velocity_m_s"]), np.array(current["node_velocity_m_s"])
        cloth_force = np.array(current["cloth_contact_n"])
        tool_force = np.array(current["tool_force_n"])
        np.testing.assert_allclose(
            (mass[:, None] * (v1 - v0)).sum(axis=0),
            h * (total * g + cloth_force),
            atol=3e-11,
            rtol=0,
        )
        np.testing.assert_allclose(cloth_force + tool_force, 0, atol=3e-11)
        np.testing.assert_allclose(
            current["tool_vertices_m"],
            local + base + [current["q"][0], 0, current["q"][1]],
            atol=1e-13,
        )
        np.testing.assert_allclose(
            current["com_m"], np.average(current["nodes_m"], axis=0, weights=mass), atol=1e-13
        )
        assert np.all(abs(np.array(current["motor_n"])) <= np.array(model["effort_limits_n"]))


def test_asset_binding_and_original_fabric_presets_are_reusable():
    from phymeo.experiments.convex_tool_transfer import build_tool_transfer
    from phymeo.experiments.sheet_fabric import SheetFabric

    custom = SheetFabric("heldout", 0.002, 400)
    a = build_tool_transfer(fabric=custom, footprint="taper", grid=(3, 2))
    assert a.program.cloth.particle_count == 6 and len(a.program.faces) == 4
    np.testing.assert_allclose(a.program.material.distance.compliance, 0.002)
    np.testing.assert_allclose(a.program.material.bending.compliance, 400)
    with pytest.raises(ValueError):
        build_tool_transfer(footprint=[(0, 0), (1, 1), (0, 1), (1, 0)])

"""Independent static oracles and a saved CUDA push with no new physical integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.area_pressure.capsule_box_reference import (
    axis_box_clearance,
    exterior_tiles,
    integrate,
)

G = 1e6
BOX = np.array([[[-0.04, -0.03, -0.02], [0.04, 0.03, 0]]])
SOURCE = Path("artifacts/mujoco-tyx-push-feedback-20261115-01/world5")


def test_capsule_end_against_plane_and_rotated_world():
    radius, depth = 0.018, 0.002
    axis = [[0.01, 0, radius - depth], [0.01, 0, 0.08]]
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    expected_area = np.pi * depth * (2 * radius - depth)
    expected_load = G * np.pi * depth**2 * (radius - 2 * depth / 3)
    bounds = np.asarray(integrate(axis, BOX, radius, G, np.zeros(3), rotation, depth=9)["bounds"])
    expected = [expected_area, expected_load, -expected_load, 0, 0, 0, 0, 0.01 * expected_load]
    assert np.all(bounds[:, 0] <= np.array(expected) + 1e-13)
    assert np.all(bounds[:, 1] >= np.array(expected) - 1e-13)
    assert np.ptp(bounds[0]) / expected_area < 0.06
    assert np.ptp(bounds[1]) / expected_load < 0.10
    assert axis_box_clearance(np.array(axis), BOX, radius) == pytest.approx(-depth)


def test_exterior_union_removes_hidden_faces_and_duplicate_patches():
    # Two overlapping boxes with the same cross-section form one rectangular solid.
    halves = np.array(
        [[[-0.04, -0.03, -0.02], [0.005, 0.03, 0]], [[-0.01, -0.03, -0.02], [0.04, 0.03, 0]]]
    )

    def area(boxes):
        return sum(4 * half[half > 0].prod() for _, half, _ in exterior_tiles(boxes))

    assert area(halves) == pytest.approx(area(BOX))
    for center, _, normal in exterior_tiles(halves):
        if abs(normal[0]) == 1:
            assert abs(center[0]) == 0.04
    # A sphere entirely inside the union may intersect a buried input face only.
    internal = [[0.005, 0, -0.01]] * 2
    bounds = integrate(internal, halves, 0.003, G, np.zeros(3), np.eye(3))["bounds"]
    assert not np.any(bounds)
    separated = np.array([[-0.1, 0.1, 0.02], [0.06, 0.1, 0.02]])
    assert axis_box_clearance(separated, BOX, 0.018) == pytest.approx(np.hypot(0.07, 0.02) - 0.018)


def test_round_side_and_concave_corner_are_not_union_aabb():
    shapes = np.array(
        [[[-0.01, -0.04, -0.02], [0.01, 0.02, 0.02]], [[-0.04, 0.02, -0.02], [0.04, 0.04, 0.02]]]
    )
    hollow = np.array([[0.025, -0.015, -0.025], [0.025, -0.015, 0.025]])
    assert axis_box_clearance(hollow, shapes, 0.01) == pytest.approx(0.005)
    assert not np.any(integrate(hollow, shapes, 0.01, G, np.zeros(3), np.eye(3))["bounds"])
    side = hollow.copy()
    side[:, 0] = 0.019
    bounds = np.array(integrate(side, shapes, 0.01, G, np.zeros(3), np.eye(3))["bounds"])
    assert bounds[0, 0] > 0 and bounds[1, 0] > 0 and bounds[2, 1] < 0
    assert np.all(bounds[3:5, 0] <= 0) and np.all(bounds[3:5, 1] >= 0)


def test_portable_poses_and_full_model_admission():
    import mujoco as mj

    from experiments.area_pressure.pusht_saved_contact import compare_saved_poses
    from phymeo.backends.area_friction_law import AreaFrictionPairLaw
    from phymeo.backends.mujoco_area_pressure import area_collision_ownership, compile_area_plan
    from phymeo.geometry.area_pressure import AreaPressureIdentity
    from phymeo.training.manipulation_tasks import manipulation_scenes

    report = compare_saved_poses()
    scene = next(s for s in manipulation_scenes() if s.key == report["scene"])
    model = mj.MjModel.from_xml_string(scene.mjcf)
    pair = ("tool", "object")
    ownership = area_collision_ownership(mj, model, (pair,))
    assert len(ownership.replaced_geometry_pairs[0]) == 2
    assert len(ownership.native_geometry_pairs) == 9
    law = AreaFrictionPairLaw(AreaPressureIdentity(*pair, "capsule", "compound-box"), 0.7, 0.01)
    assert model.nv == 8 and sum(model.jnt_limited) == 2
    with pytest.raises(ValueError, match="damped area route cannot mix native constraints"):
        compile_area_plan(mj, model, (pair,), (law,))
    assert not any(s["compatible"] for s in report["peak_fit"]["holdout"].values())
    assert set(report["samples"]) == {
        "initial",
        "first_load",
        "peak",
        "last_push_entry",
        "withdrawn",
    }


@pytest.mark.skipif(
    not (SOURCE / "manifest.json").is_file(), reason="local saved CUDA trace required"
)
def test_saved_native_push_compared_at_same_poses(monkeypatch, tmp_path):
    from experiments.area_pressure import pusht_saved_contact as run

    def forbidden(*args, **kwargs):
        pytest.fail("saved reference must not advance physics")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(run.mj, name, forbidden)
    output = tmp_path / "rr-contact.json"
    monkeypatch.setattr("sys.argv", ["pusht_saved_contact", str(SOURCE), "--output", str(output)])
    run.main()
    report = json.loads(output.read_text("utf-8"))
    samples = report["samples"]
    initial, onset, peak, released = (
        samples[k] for k in ("initial", "first_load", "peak", "withdrawn")
    )
    assert initial["tick"] < onset["tick"] <= peak["tick"] < released["tick"]
    assert report["displacement_while_pushing_m"][1] > 0.001
    assert report["tool_impulse_world_ns"][1] > 0
    assert peak["tool_normal_n"] > 0.01 and peak["support_n"] > 0
    assert released["clearance_m"] > 0.001 and released["support_n"] > 0
    for frame in (initial, released):
        assert frame["detected"] == 0 and not np.any(frame["area_reference"]["9"]["bounds"])
    replaced = {frozenset(pair) for pair in report["replaced_candidates"]}
    native = {frozenset(pair) for pair in report["retained_native_candidates"]}
    assert replaced == {frozenset(("pusher", geom)) for geom in ("object-0", "object-1")}
    assert not replaced & native
    assert all(frozenset(("support", geom)) in native for geom in ("object-0", "object-1"))
    holdout = report["peak_fit"]["holdout"]
    assert holdout["first_load"]["predicted_normal_n"][1] < onset["tool_normal_n"]
    assert (
        holdout["last_push_entry"]["predicted_normal_n"][0]
        > samples["last_push_entry"]["tool_normal_n"]
    )
    bounds = np.array(peak["area_reference"]["9"]["bounds"])
    coarse = np.array(peak["area_reference"]["7"]["bounds"])
    assert bounds[0, 0] > 0 and bounds[1, 0] > 0
    assert bounds[3, 0] > 0
    assert np.all(bounds[:, 0] >= coarse[:, 0] - 1e-12)
    assert np.all(bounds[:, 1] <= coarse[:, 1] + 1e-12)

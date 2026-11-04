"""Real two-hinge center arcs through the ordinary RobotAsset and collision dispatcher."""

import math
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import phymeo as ph

XML = Path(__file__).parent / "reference_newton" / "robot" / "two_hinge_ellipse.xml"
FIRST = math.asin((math.sqrt(322) - 2) / 16) / math.pi
CYCLE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def spheres(asset):
    return {c.name: c.shape for c in asset.collisions}


def query(a, pa, b, pb):
    model = ph.CollisionModel((a, b))
    with ph.CollisionWorkspace(model, "host") as workspace:
        result = ph.detect(model, {a.key: pa, b.key: pb}, ph.CollisionPolicy(), workspace)
    assert result.status == "ok"
    return result.pairs[0]


@pytest.mark.parametrize("sign,reversed_time,transformed", [(1, False, False), (-1, True, True)])
def test_ellipse_has_a_certified_first_time_and_two_surface_material_addresses(
    tmp_path, sign, reversed_time, transformed
):
    source = XML
    origin, basis = np.zeros(3), np.eye(3)
    if transformed:
        source = tmp_path / "reframed.xml"
        source.write_text(
            XML.read_text().replace(
                '<body name="base">', '<body name="base" pos="2 -.5 .75" quat=".5 .5 .5 .5">'
            )
        )
        origin, basis = np.array([2.0, -0.5, 0.75]), CYCLE
    asset = ph.load_mjcf(source)
    shapes = spheres(asset)
    start, end = np.zeros(2), np.array([sign * math.pi, -2 * sign * math.pi])
    if reversed_time:
        start, end = end, start
    paths = asset.collision_motion(start, end)
    moving = shapes["tool"]
    fixed = shapes["north" if sign > 0 else "south"]
    other = shapes["south" if sign > 0 else "north"]
    path = paths[moving.key]

    def exact(t):
        u = 1 - t if reversed_time else t
        return origin + basis @ [1.5 * math.cos(math.pi * u), sign * 0.5 * math.sin(math.pi * u), 0]

    for t in (0, 0.19, 0.5, 0.76, 1):
        np.testing.assert_allclose(path.positions_at(t)[0], exact(t), rtol=0, atol=3e-14)
    pair = query(moving, path, fixed, paths[fixed.key])
    assert pair.candidate and pair.status == "hit", pair
    sweep = pair.sweep
    lo, hi = sweep.first_t
    assert lo <= FIRST <= hi and 0 <= hi - lo <= 2**-27
    assert sweep.start_gap_m > 1 and sweep.end_gap_m > 1
    assert lo <= sweep.witness_time <= hi
    w = sweep.witness
    assert w.gap_m <= 0 and w.normal is not None
    k = 0 if pair.first == moving.key else 1
    for side, shape, center in (
        (k, moving, exact(sweep.witness_time)),
        (1 - k, fixed, paths[fixed.key].start[0]),
    ):
        weights = (w.first_weights, w.second_weights)[side]
        point = (w.first_point, w.second_point)[side]
        offset = (w.first_offset, w.second_offset)[side]
        assert weights == ((shape.vertex_ids[0], 1.0),)
        assert np.linalg.norm(offset) == pytest.approx(shape.radius, abs=1e-12)
        np.testing.assert_allclose(center + offset, point, rtol=0, atol=3e-12)
    clear = query(other, paths[other.key], moving, path)
    assert clear.candidate and clear.status == "separated" and clear.sweep.first_t is None
    print("ellipse", sign, reversed_time, pair.sweep.first_t, w.gap_m, "clear", clear.reason)


def test_nonparallel_hinges_and_an_independently_translating_sphere(tmp_path):
    source = tmp_path / "skew.xml"
    source.write_text(
        XML.read_text().replace(
            'name="elbow" type="hinge" axis="0 0 1"', 'name="elbow" type="hinge" axis="0 1 0"'
        )
    )
    asset = ph.load_mjcf(source)
    shapes = spheres(asset)
    start = np.array([0.2, -0.3])
    end = np.array([1.6, 1.2])
    tool = shapes["tool"]
    path = asset.collision_motion(start, end)[tool.key]

    def center(t):
        a, b = start + (end - start) * t
        reach = 1 + 0.5 * math.cos(b)
        return np.array([reach * math.cos(a), reach * math.sin(a), -0.5 * math.sin(b)])

    for t in (0, 0.21, 0.5, 0.73, 1):
        np.testing.assert_allclose(path.positions_at(t)[0], center(t), rtol=0, atol=3e-14)
    drift = np.array([0.0, 0.0, 0.25])
    fixed = center(0.5) - 0.5 * drift
    target = ph.CollisionShape(ph.CollisionKey(0, 5, 1), "sphere", (71,), 0.0625)
    pair = query(tool, path, target, ph.CollisionMotion([fixed], [fixed + drift]))
    assert pair.status == "hit" and min(pair.sweep.start_gap_m, pair.sweep.end_gap_m) > 0.5
    # Independent center geometry brackets entry on the approach to the known midpoint overlap.
    a, b = 0.0, 0.5
    for _ in range(58):
        m = (a + b) / 2
        if np.linalg.norm(center(m) - (fixed + m * drift)) > 0.125:
            a = m
        else:
            b = m
    lo, hi = pair.sweep.first_t
    assert lo <= b <= hi and hi - lo <= 2**-27
    print("skew-moving-target", b, (lo, hi), pair.sweep.witness.gap_m)


def test_new_path_does_not_turn_unsolved_pairs_and_overbudget_turns_into_clearance():
    asset = ph.load_mjcf(XML)
    shapes = spheres(asset)
    begin = np.zeros(2)
    tool = shapes["tool"]
    target = shapes["north"]
    paths = asset.collision_motion(begin, np.array([64.0, -2.0]))
    cap = query(tool, paths[tool.key], target, paths[target.key])
    assert cap.candidate and cap.status == "unresolved" and cap.sweep is None
    path = asset.collision_motion(begin, np.array([math.pi, -2 * math.pi]))[tool.key]
    floor = ph.CollisionShape(ph.CollisionKey(0, 4, 0), "plane", (82,), normal=(0, 0, 1))
    row = query(tool, path, floor, ph.CollisionMotion([[0.0, 0.0, 0.0]]))
    assert row.candidate and row.status == "unsupported" and row.sweep is None
    tangent = query(tool, path, target, ph.CollisionMotion([[0.0, 0.625, 0.0]]))
    assert tangent.candidate and tangent.status in ("hit", "unresolved")
    print("tangent", tangent.status, tangent.reason, "angle limit", cap.reason)
    if tangent.status == "hit":
        a, b = tangent.sweep.first_t
        assert a <= 0.5 <= b and np.linalg.norm(tangent.sweep.witness.normal) == pytest.approx(1)


def test_third_active_hinge_and_downstream_slide_keep_named_safety_boundaries(tmp_path):
    document = ET.parse(XML)
    forearm = document.find(".//body[@name='forearm']")
    tool = forearm.find("geom[@name='tool']")
    forearm.remove(tool)
    tip = ET.SubElement(forearm, "body", name="tip")
    joint = ET.SubElement(tip, "joint", name="roll", type="hinge", axis="1 0 0", range="-2 2")
    ET.SubElement(tip, "inertial", pos="0 0 0", mass=".1", diaginertia=".01 .01 .01")
    tip.append(tool)
    origin = tmp_path / "third.xml"
    document.write(origin)
    asset = ph.load_mjcf(origin)
    with pytest.raises(ValueError, match="shoulder.*elbow.*roll"):
        asset.collision_motion(np.zeros(3), np.array([0.5, -0.75, 0.2]))
    joint.set("type", "slide")
    document.write(origin)
    asset = ph.load_mjcf(origin)
    with pytest.raises(ValueError, match="roll"):
        asset.collision_motion(np.zeros(3), np.array([0.5, -0.75, 0.2]))


def test_independent_two_hinge_spheres_pass_each_other_between_separated_poses():
    asset = ph.load_mjcf(XML)
    tool = spheres(asset)["tool"]
    other = replace(tool, key=ph.CollisionKey(0, 6, 0), vertex_ids=(91,))
    q0, q1 = np.zeros(2), np.array([math.pi, -2 * math.pi])
    a = asset.collision_motion(q0, q1)[tool.key]
    b = asset.collision_motion(q1, q0)[tool.key]
    pair = query(tool, a, other, b)
    # The centers share y(t); their x separation is 3 |cos(pi*t)|.
    exact = math.acos(1 / 24) / math.pi
    assert pair.status == "hit" and pair.sweep.start_gap_m > 2 and pair.sweep.end_gap_m > 2
    lo, hi = pair.sweep.first_t
    assert lo <= exact <= hi and hi - lo <= 2**-27
    print("two moving arcs", exact, (lo, hi), pair.sweep.witness.gap_m)


def test_slides_before_two_hinges_transport_both_pivots_at_nonzero_q():
    asset = ph.load_mjcf(XML.with_name("joint_interval_arm.xml"))
    by_name = spheres(asset)
    values = {
        "advance": (0.1, 0.8),
        "diagonal": (-0.35, 0.4),
        "spin": (0.15, 0.9),
        "bend": (0.6, -0.4),
    }
    q0, q1 = np.array([values.get(name, (0.0, 0.0)) for name in asset.joint_names]).T
    key = by_name["wrist_tag"].key
    path = asset.collision_motion(q0, q1)[key]
    np.testing.assert_allclose(
        np.subtract(path.outer.pivot_end, path.outer.pivot_start), [0.7, 0.45, 0.6], atol=3e-14
    )
    for t in (0, 0.27, 0.5, 0.91, 1):
        actual = asset.collision_motion(q0 + t * (q1 - q0))[key].start
        np.testing.assert_allclose(path.positions_at(t), actual, atol=3e-14, rtol=0)


@pytest.mark.tyx
def test_device_candidate_covers_the_real_double_hinge_arc(tyx_mode, tmp_path):
    asset = ph.load_mjcf(XML)
    shapes = spheres(asset)
    tool, fixed = shapes["tool"], shapes["north"]
    model = ph.CollisionModel([tool, fixed])
    paths = asset.collision_motion(np.zeros(2), np.array([math.pi, -2 * math.pi]))
    with ph.CollisionWorkspace(
        model, tyx_mode, broad_phase="bvh", cache_root=tmp_path / "device-cache"
    ) as work:
        result = ph.detect(
            model, {x.key: paths[x.key] for x in (tool, fixed)}, ph.CollisionPolicy(), work
        )
        (pair,) = result.pairs
        assert result.backend == tyx_mode and pair.candidate and pair.status == "hit"
        assert pair.sweep.witness.source == "host-two-hinge-spheres"
        lo, hi = pair.sweep.first_t
        assert lo <= FIRST <= hi and hi - lo <= 2**-27
        stats = work.stats()
        assert stats["completed_launches"] > 0
        if tyx_mode == "cuda":
            assert stats["downloads"] > 0
        print(
            "two-hinge device",
            tyx_mode,
            "launches",
            stats["completed_launches"],
            "downloads",
            stats["downloads"],
            "entry",
            (lo, hi),
        )

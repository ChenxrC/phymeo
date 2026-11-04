"""Real two-hinge center arcs through the ordinary RobotAsset and collision dispatcher."""

import math
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

XML = Path(__file__).parent / "reference_newton" / "robot" / "two_hinge_ellipse.xml"
FIRST = math.asin((math.sqrt(322) - 2) / 16) / math.pi
CYCLE = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


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
        source.write_text(XML.read_text().replace('<body name="base">', '<body name="base" pos="2 -.5 .75" quat=".5 .5 .5 .5">'))
        origin, basis = np.array([2., -.5, .75]), CYCLE
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
        return origin + basis @ [1.5 * math.cos(math.pi * u), sign * .5 * math.sin(math.pi * u), 0]

    for t in (0, .19, .5, .76, 1):
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
        (k, moving, exact(sweep.witness_time)), (1-k, fixed, paths[fixed.key].start[0])
    ):
        weights = (w.first_weights, w.second_weights)[side]
        point = (w.first_point, w.second_point)[side]
        offset = (w.first_offset, w.second_offset)[side]
        assert weights == ((shape.vertex_ids[0], 1.0),)
        assert np.linalg.norm(offset) == pytest.approx(shape.radius, abs=1e-12)
        np.testing.assert_allclose(center + offset, point, rtol=0, atol=3e-12)
    clear = query(other, paths[other.key], moving, path)
    assert clear.candidate and clear.status == "separated" and clear.sweep.first_t is None


def test_nonparallel_hinges_and_an_independently_translating_sphere(tmp_path):
    source = tmp_path / "skew.xml"
    source.write_text(XML.read_text().replace('name="elbow" type="hinge" axis="0 0 1"', 'name="elbow" type="hinge" axis="0 1 0"'))
    asset = ph.load_mjcf(source)
    shapes = spheres(asset)
    start = np.array([.2, -.3])
    end = np.array([1.6, 1.2])
    tool = shapes["tool"]
    path = asset.collision_motion(start, end)[tool.key]

    def center(t):
        a, b = start + (end - start) * t
        reach = 1 + .5 * math.cos(b)
        return np.array([reach * math.cos(a), reach * math.sin(a), -.5 * math.sin(b)])
    for t in (0, .21, .5, .73, 1):
        np.testing.assert_allclose(path.positions_at(t)[0], center(t), rtol=0, atol=3e-14)
    drift = np.array([0., 0., .25])
    fixed = center(.5) - .5 * drift
    target = ph.CollisionShape(ph.CollisionKey(0, 5, 1), "sphere", (71,), .0625)
    pair = query(tool, path, target, ph.CollisionMotion([fixed], [fixed + drift]))
    assert pair.status == "hit" and min(pair.sweep.start_gap_m, pair.sweep.end_gap_m) > .5
    # Independent center geometry brackets entry on the approach to the known midpoint overlap.
    a, b = 0., .5
    for _ in range(58):
        m = (a + b) / 2
        if np.linalg.norm(center(m) - (fixed + m * drift)) > .125:
            a = m
        else:
            b = m
    lo, hi = pair.sweep.first_t
    assert lo <= b <= hi and hi - lo <= 2**-27


def test_new_path_does_not_turn_unsolved_pairs_and_overbudget_turns_into_clearance():
    asset = ph.load_mjcf(XML)
    shapes = spheres(asset)
    begin = np.zeros(2)
    tool = shapes["tool"]
    target = shapes["north"]
    paths = asset.collision_motion(begin, np.array([64., -2.]))
    cap = query(tool, paths[tool.key], target, paths[target.key])
    assert cap.candidate and cap.status == "unresolved" and cap.sweep is None
    path = asset.collision_motion(begin, np.array([math.pi, -2 * math.pi]))[tool.key]
    floor = ph.CollisionShape(ph.CollisionKey(0, 4, 0), "plane", (82,), normal=(0, 0, 1))
    row = query(tool, path, floor, ph.CollisionMotion([[0., 0., 0.]]))
    assert row.candidate and row.status == "unsupported" and row.sweep is None
    tangent = query(tool, path, target, ph.CollisionMotion([[0., .625, 0.]]))
    assert tangent.candidate and tangent.status in ("hit", "unresolved")
    if tangent.status == "hit":
        a, b = tangent.sweep.first_t
        assert a <= .5 <= b and np.linalg.norm(tangent.sweep.witness.normal) == pytest.approx(1)


def test_third_active_hinge_and_downstream_slide_keep_named_safety_boundaries(tmp_path):
    src = XML.read_text().replace('<geom name="tool"', '<body name="tip"><joint name="roll" type="hinge" axis="1 0 0" range="-2 2"/><inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/><geom name="tool"').replace('pos="0.5 0 0"/>', 'pos="0.5 0 0"/></body>')
    origin = tmp_path / "third.xml"
    origin.write_text(src)
    asset = ph.load_mjcf(origin)
    with pytest.raises(ValueError, match="shoulder.*elbow.*roll"):
        asset.collision_motion(np.zeros(3), np.array([.5, -.75, .2]))
    origin.write_text(src.replace('name="roll" type="hinge"', 'name="roll" type="slide"'))
    asset = ph.load_mjcf(origin)
    with pytest.raises(ValueError, match="roll"):
        asset.collision_motion(np.zeros(3), np.array([.5, -.75, .2]))

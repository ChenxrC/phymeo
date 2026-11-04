"""Finite planar rigid contact rows carry the clipped polygon and both bodies' materials."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

import phymeo as ph

ZERO = np.zeros(3)
UP = np.array([0.0, 0.0, 1.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
H = 1 / 64


def hulls(*, octagon=False, reverse=False, speed=(0, 0, -1)):
    pad = ph.CollisionKey(0, int(reverse), 1)
    box = ph.CollisionKey(0, int(not reverse), 1)
    pad_local = np.array(list(product((-1.0, 1.0), (-1.0, 1.0), (-0.25, 0.25))))
    if octagon:
        ring = [(1.5, 0), (0, 1.5), (-1.5, 0), (0, -1.5)]
    else:
        ring = list(product((-0.5, 0.5), (-0.5, 0.5)))
    box_local = np.array([(x, y, z) for x, y in ring for z in (-0.5, 0.5)])
    local = {pad: pad_local, box: box_local}
    model = ph.CollisionModel(
        [ph.CollisionShape(key, "hull", tuple(9 + 3 * i for i in range(8))) for key in local]
    )
    center = {pad: -0.25 * UP, box: 0.5 * UP}
    paths = {k: ph.CollisionMotion(x + center[k]) for k, x in local.items()}
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work).pairs
    (polygon,) = row.convex.end.patches
    assert row.status == "touching" and polygon.dimension == 2
    if octagon:
        xy = {(x, y) for x in (-1, 1) for y in (-0.5, 0.5)}
        xy |= {(y, x) for x, y in xy}
    else:
        xy = set(ring)
    expected = {(x, y, 0.0) for x, y in xy}
    assert {v.first_point for v in polygon.points} == expected
    assert {v.second_point for v in polygon.points} == expected
    contacts = np.array([v.first_point for v in polygon.points])
    names = {pad: "pad", box: "box"}
    indices = np.zeros(len(contacts), dtype=np.int64)
    attachments = {names[k]: (indices, contacts - center[k]) for k in local}
    floating = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.ones(1), np.array([[1.0, 0.75, 0.5]]), ZERO), ZERO
    )
    state = ph.FloatingRobotState(np.r_[center[box], IDENTITY], np.r_[speed, ZERO])
    inputs = ph.FloatingRobotInputs(np.empty(0), np.zeros((1, 3)), np.zeros((1, 3)))
    fixed_model = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), ZERO)
    fixed_state = ph.RigidBodyState(center[pad][None, :], IDENTITY[None, :], ZERO[None, :], ZERO[None, :])

    def step(order, *, current=state, owner=local, submitted=row, mu=0.0, sites=attachments):
        rows = []
        for i in order:
            j = 0 if i is None else i
            bindings = {k: (names[k], j) for k in local}
            rows.append((submitted, bindings) if i is None else (submitted, bindings, i))
        return ph.step_floating_contacts(
            {"box": (floating, current, inputs)}, sites, rows, [mu] * len(rows), [mu] * len(rows), H,
            fixed_bodies={"pad": (fixed_model, fixed_state)}, local_vertices=owner,
        )

    return row, pad, box, contacts, local, state, attachments, step


@pytest.mark.parametrize("octagon,reverse", [(False, False), (True, True)])
def test_complete_clipped_polygon_stops_the_loaded_body_without_an_artificial_tip(octagon, reverse):
    pair, _, box, sites, _, state, _, step = hulls(octagon=octagon, reverse=reverse)
    assert len(sites) == (8 if octagon else 4)
    order = tuple(reversed(range(len(sites))))
    out = step(order)
    assert out.report.accepted, out.report.reason
    sign = 1 if pair.second == box else -1
    forces = sign * out.impulses.world_impulses
    arms = sites[list(order)] - state.q[:3]
    np.testing.assert_allclose(forces.sum(0), UP, atol=2e-9, rtol=0)
    np.testing.assert_allclose(np.cross(arms, forces).sum(0), ZERO, atol=2e-9, rtol=0)
    np.testing.assert_allclose(out.states["box"].q, state.q, atol=2e-10, rtol=0)
    np.testing.assert_allclose(out.states["box"].v, np.zeros(6), atol=2e-9, rtol=0)
    assert out.impulses.energy_change == pytest.approx(-0.5, abs=2e-9)


def test_polygon_sliding_friction_and_opening_use_real_site_velocities():
    pair, _, box, sites, _, state, _, step = hulls(speed=(3, 0, -1))
    out = step(range(4), mu=0.25)
    assert out.report.accepted, out.report.reason
    impulses = out.impulses.world_impulses * (1 if pair.second == box else -1)
    np.testing.assert_allclose(impulses.sum(0), [-0.25, 0, 1], atol=2e-8, rtol=0)
    np.testing.assert_allclose(np.cross(sites - state.q[:3], impulses).sum(0), ZERO, atol=2e-8, rtol=0)
    np.testing.assert_allclose(out.states["box"].v, [2.75, 0, 0, 0, 0, 0], atol=2e-8, rtol=0)
    assert all(np.linalg.norm(p[:2]) <= 0.25 * p[2] + 1e-10 for p in impulses)
    upward = ph.FloatingRobotState(state.q, np.r_[UP, ZERO])
    opens = step(range(4), current=upward, mu=0.25)
    assert opens.report.accepted, opens.report.reason
    np.testing.assert_array_equal(opens.impulses.world_impulses, np.zeros((4, 3)))
    assert opens.states["box"].q[2] == pytest.approx(state.q[2] + H)
    with pytest.raises((ValueError, ph.SolverFailure), match="witness|material|polygon"):
        step(range(4), current=opens.states["box"], mu=0.25)


@pytest.mark.parametrize("fault", ["unindexed", "missing", "repeat", "truncated", "material", "body_point"])
def test_bad_face_rows_never_publish_partial_impulses(fault):
    pair, _, box, _, local, state, attachments, step = hulls()
    q, v = state.q.copy(), state.v.copy()
    frozen = {key: val.copy() for key, val in local.items()}
    order = [0, 1, 2, 3]
    owner = {k: x.copy() for k, x in local.items()}
    sites = {k: (ids.copy(), points.copy()) for k, (ids, points) in attachments.items()}
    if fault == "unindexed":
        order = [None]
    elif fault == "missing":
        order.pop()
    elif fault == "repeat":
        order[-1] = 0
    elif fault == "truncated":
        patch = pair.convex.end.patches[0]
        end = replace(pair.convex.end, patches=(replace(patch, points=patch.points[:-1]),))
        pair = replace(pair, convex=replace(pair.convex, end=end))
        order.pop()
    elif fault == "material":
        owner[box][[0, 2]] = owner[box][[2, 0]]
    else:
        sites["box"][1][1, 0] += 0.0625
    try:
        out = step(order, owner=owner, submitted=pair, sites=sites)
    except (ValueError, ph.SolverFailure):
        pass
    else:
        assert not out.report.accepted and out.impulses is None and not out.predicted
        np.testing.assert_array_equal(out.states["box"].v, v)
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_array_equal(state.v, v)
    for key in local:
        np.testing.assert_array_equal(local[key], frozen[key])

"""Public point-cloud hull registrations for the authored rigid contact controls."""

from fractions import Fraction as F
from itertools import product

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "permuted", "clear_gap", "edge_edge", "holdout")


def corners(shape):
    center = shape["center"] if "center" in shape else shape["center_start"]
    axes = shape["half_axes"] if "half_axes" in shape else np.diag(shape["half"])
    return np.array(
        [
            [
                float(F(c) + sum(sign * F(a[k]) for sign, a in zip(signs, axes, strict=True)))
                for k, c in enumerate(center)
            ]
            for signs in product((-1, 1), repeat=3)
        ]
    )


def scene(items, world=0, reverse=False):
    shapes, paths = [], {}
    for i, (points, delta, ids) in enumerate(items):
        key = ph.CollisionKey(world, 1 - i if reverse else i, 10 + i)
        shapes.append(ph.CollisionShape(key, "hull", tuple(ids)))
        turn = None if delta is None else ph.RigidTurn([0, 0, 0], delta, [0, 0, 1], 0.0)
        paths[key] = ph.CollisionMotion(points, turn=turn)
    return ph.CollisionModel(tuple(reversed(shapes))), paths


def authored(name, world=0, reverse=False):
    _, v, g = geometry_for("RR03", name)
    order = v.get("hull_order", list(range(8)))
    items = []
    for shape in (g["A"], g["B"]):
        points = corners(shape)
        items.append((points[order], None, order))
    model, paths = scene(items, world, reverse)
    moving = ph.CollisionKey(world, 0 if reverse else 1, 11)
    b = g["B"]
    paths[moving] = ph.CollisionMotion(
        paths[moving].start, turn=ph.RigidTurn(b["center_start"], b["center_end"], [0, 0, 1], 0.0)
    )
    return model, paths


def sample_paths(model, paths, time):
    result = {}
    for shape in model.shapes:
        p = paths[shape.key]
        if p.turn is not None:
            # Construct the stored pose from the exact binary path rather than
            # interpolating rounded intermediate box centers.
            delta = [F(b) - F(a) for a, b in zip(p.turn.pivot_start, p.turn.pivot_end, strict=True)]
            x = np.array(
                [
                    [float(F(x) + time * d) for x, d in zip(row, delta, strict=True)]
                    for row in p.start
                ]
            )
        else:
            x = p.start
        result[shape.key] = ph.CollisionMotion(x)
    return result


def passing(world=0):
    cube = corners({"center": [0, 0, 0], "half": [0.5, 0.5, 0.5]})
    return scene([(cube, None, range(8)), (cube + [-2, 0, 0], [4, 0, 0], range(8))], world)


def isolated(world=0):
    cube = corners({"center": [0, 0, 0], "half": [0.5, 0.5, 0.5]})
    return scene([(cube, None, range(8)), (cube + [0.5, 1.5, 1], [1.5, -1.5, 0], range(8))], world)


def tetrahedra(world=0):
    a = np.vstack((np.zeros(3), np.eye(3)))
    b = [[-0.375, 0.25, 0.25], [-0.75, 0.125, 0.125], [-0.75, 0.375, 0.125], [-0.75, 0.125, 0.375]]
    return scene([(a, None, (21, 30, 40, 50)), (b, [0.75, 0, 0], (2, 3, 4, 5))], world)


def long_axis_table(world=0):
    x = np.arange(-7, 9) / 8
    a = np.stack([x, x * x, x * x * x], axis=1)
    a -= a.mean(axis=0)
    b = a @ np.array([[1, 2, 0], [-2, 1, 1], [1, 0, 1]]).T
    return scene([(a, None, range(16)), (b, None, range(16))], world)


def sliding_frames(backend, cache_root):
    model, original = authored("base")
    times = (F(0), F(1, 5), F(3, 5), F(4, 5), F(1))
    rows = []
    with ph.CollisionWorkspace(model, backend, cache_root=cache_root) as work:
        for lo, hi in zip(times[:-1], times[1:], strict=True):
            start, end = sample_paths(model, original, lo), sample_paths(model, original, hi)
            motion = {
                key: ph.CollisionMotion(value.start, end[key].start) for key, value in start.items()
            }
            result = ph.detect(model, motion, ph.CollisionPolicy(), work)
            assert work.last_result is result
            rows.append(result.pairs[0])
        stats = work.stats()
    for left, right in zip(rows[:-1], rows[1:], strict=True):
        assert left.convex.end == right.convex.start
    patches = [
        next(p for p in row.convex.start.patches if p.normal == (0.0, 0.0, 1.0)) for row in rows
    ]
    assert [p.area_m2 for p in patches] == [1.0, 0.75, 0.25, 0.0]
    assert [p.dimension for p in patches] == [2, 2, 2, 1]
    assert len({(p.first_support, p.second_support) for p in patches}) == 1
    assert rows[-1].convex.last_time_ratio == (0, 1) and rows[-1].convex.end.relation == "separated"
    return rows, stats


def over_axis_budget(world=0):
    x = np.arange(-19, 21) / 16
    a = np.stack([x, x * x, x * x * x], axis=1)
    a -= a.mean(axis=0)
    b = a @ np.array([[1, 2, 0], [-2, 1, 1], [1, 0, 1]]).T
    return scene([(a, None, range(40)), (b, None, range(40))], world)

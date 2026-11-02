"""Mixed registrations using the original RR04, RD01 and DD02 geometry."""

from dataclasses import asdict
from itertools import combinations
from math import cos, pi, sin

import numpy as np
from case_catalog import catalog_entries

import phymeo as ph


def mixed_scene(reverse=False):
    cases = catalog_entries()
    keys = [ph.CollisionKey(0, i, 0) for i in range(1, 8)]
    keys.extend(
        [
            ph.CollisionKey(1, 1, 0),
            ph.CollisionKey(0, 8, 0),
            ph.CollisionKey(0, 1, 1),
            ph.CollisionKey(0, 5, 1),
        ]
    )
    items, paths, boxes = [], {}, {}

    def add(i, kind, start, end=None, *, radius=0.0, turn=None, group=1, mask=1, vertex_ids=None):
        key = keys[i]
        p = np.array(start, dtype=float)
        q = p if end is None else np.array(end)
        ids = tuple(range(len(p))) if vertex_ids is None else vertex_ids
        items.append(ph.CollisionShape(key, kind, ids, radius, group, mask))
        paths[key] = ph.CollisionMotion(p, q if turn is None else None, turn)
        if turn is None:
            boxes[key] = (
                np.minimum(p.min(0), q.min(0)) - radius,
                np.maximum(p.max(0), q.max(0)) + radius,
            )
        else:
            # All orientations about this pivot fit the documented L1-radius envelope.
            bound = np.max(np.abs(p - turn.pivot_start).sum(1)) + radius
            boxes[key] = (
                np.minimum(turn.pivot_start, turn.pivot_end) - bound,
                np.maximum(turn.pivot_start, turn.pivot_end) + bound,
            )

    rr = cases["RR04"]["geometry"]
    phi = -pi / 3
    rotation = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    p = np.array(rr["A"]["local_axis"]) @ rotation.T
    turn = ph.RigidTurn([0, 0, 0], [0, 0, 0], [0, 0, 1], 2 * pi / 3)
    add(0, "capsule", p, turn=turn, radius=rr["A"]["radius"])
    add(1, "sphere", [rr["B"]["center"]], radius=rr["B"]["radius"])
    rd = cases["RD01"]["geometry"]
    add(2, "triangle", rd["A"]["vertices"])
    add(3, "sphere", [rd["B"]["center_start"]], [rd["B"]["center_end"]], radius=rd["B"]["radius"])
    dd = cases["DD02"]["geometry"]
    for i, name in ((4, "A"), (5, "B")):
        add(i, "triangle", dd[name]["vertices_start"], dd[name]["vertices_end"])
    add(6, "sphere", [[10, 10, 10]], radius=0.125)
    add(7, "sphere", [[0, 0, 0]], radius=0.125)
    add(8, "sphere", [[0, 0, 0]], radius=0.125, group=2, mask=2)
    add(9, "sphere", [[0, 0, 0]], radius=0.125)
    edge_a, edge_b, _ = dd["A"]["vertices_start"]
    add(10, "triangle", [edge_b, edge_a, [0, 0, 1]], vertex_ids=(1, 0, 3))
    excluded = ((keys[2], keys[0]),)
    model = ph.CollisionModel(items[::-1] if reverse else items, excluded)
    expected = {}
    for a, b in combinations(sorted(items, key=lambda s: s.key), 2):
        pair = (a.key, b.key)
        if a.key.world != b.key.world:
            reason = "different_world"
        elif a.key.body == b.key.body and a.kind in {"capsule", "sphere"}:
            reason = "same_rigid_body"
        elif not (a.group & b.mask and b.group & a.mask):
            reason = "mask"
        elif set(pair) == set(excluded[0]):
            reason = "excluded"
        elif np.any(boxes[a.key][0] > boxes[b.key][1]) or np.any(boxes[b.key][0] > boxes[a.key][1]):
            reason = "aabb"
        else:
            reason = "candidate"
        expected[pair] = reason
    return model, paths, keys, expected


def verify_mixed(result, keys, expected):
    assert result.status == "ok"
    assert [(p.first, p.second) for p in result.pairs] == list(expected)
    for p in result.pairs:
        e = expected[p.first, p.second]
        assert p.candidate == (e == "candidate"), (p, e)
        if e in {"different_world", "same_rigid_body", "mask", "excluded"}:
            assert p.status == "filtered" and p.reason == e
        elif e == "aabb":
            assert p.status == "separated" and p.reason == e
    pairs = {(p.first, p.second): p for p in result.pairs}
    for i, j, f in ((0, 1, "RR"), (2, 3, "RD"), (4, 5, "DD")):
        a = pairs[keys[i], keys[j]]
        assert a.family == f and a.candidate and a.status == "unsupported"
    assert pairs[keys[4], keys[10]].candidate
    assert result.required_candidates == sum(v == "candidate" for v in expected.values())


def saved_inputs(model, paths, policy):
    return dict(
        model=asdict(model),
        policy=asdict(policy),
        motion=[
            dict(
                key=asdict(s.key),
                start=paths[s.key].start.tolist(),
                end=None if paths[s.key].end is None else paths[s.key].end.tolist(),
                turn=None if paths[s.key].turn is None else asdict(paths[s.key].turn),
            )
            for s in model.shapes
        ],
    )

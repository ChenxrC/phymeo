"""First contact of a translating mesh solid and a triangle with three independent affine paths."""

import json
from dataclasses import asdict
from fractions import Fraction as F

import numpy as np
import pytest
from test_mesh_triangle import CASES, build, detect, materials

import phymeo as ph

DRIFT = np.array([0.125, -0.0625, 0.25])


def moving(kind, reverse):
    case = dict(next(c for c in CASES if c["id"] == "all-nodes-outside"))
    xy = np.array([[-4.0, -3.0], [4.0, -3.0], [0.0, 5.0]])
    start = np.column_stack((xy, np.full(3, 2.0)))
    end = np.column_stack((xy, np.full(3, -2.0)))
    expected = F(1, 4)
    if kind == "independent":
        end[:, 2] = xy[:, 0] / 4 + xy[:, 1] / 8
        expected = F(8, 19)
    elif kind in ("graze", "miss"):
        a = np.array([0.625, 1.375, 0])
        if kind == "miss":
            a = a + [1 / 128, 1 / 128, 0]
        start = np.stack((a, a + [0.25, 0.25, 0], a + [0.5, 0.25, 0.25]))
        end = start + [1, -1, 0]
        expected = F(3, 8) if kind == "graze" else None
    elif kind == "contained-exit":
        start = np.array(next(c for c in CASES if c["id"] == "contained")["triangle_m"])
        end = start + [0, 0, 1]
        expected = F(0)
    elif kind == "notch":
        case["mesh"] = "u"
        start = np.array([[0.25, 0.125, 0], [-0.5, 0.125, 0], [0, 0.75, 0.125]])
        end = start + [1.25, 0, 0]
        expected = F(3, 5)
    case["triangle_m"] = start
    model, paths, mesh, cloth, matrix, origin = build(case, reverse, reverse)
    paths[mesh.key] = ph.CollisionMotion(
        paths[mesh.key].start, paths[mesh.key].start + matrix @ DRIFT
    )
    xyz = end[::-1] if reverse else end
    paths[cloth.key] = ph.CollisionMotion(paths[cloth.key].start, (xyz + DRIFT) @ matrix.T + origin)
    return model, paths, mesh, cloth, matrix, origin, expected


@pytest.mark.parametrize(
    "kind", ["full-face", "independent", "notch", "contained-exit", "graze", "miss"]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_closed_mesh_and_affine_whole_triangle_first_contact(kind, reverse, tmp_path):
    model, paths, mesh, cloth, matrix, origin, expected = moving(kind, reverse)
    row = detect(model, paths)
    assert row.status == ("separated" if expected is None else "hit"), row.reason
    trace = row.mesh_triangle
    assert (
        row.sweep.start_gap_m == trace.start.witness.gap_m
        and row.sweep.end_gap_m == trace.end.witness.gap_m
    )
    materials(row.witness, model, paths, 1)
    if expected is None:
        assert row.sweep.first_t is row.sweep.witness is trace.first_contact is None
        return
    event = trace.first_contact
    assert row.sweep.first_t == event.interval and row.sweep.witness_time == event.witness_time
    assert row.sweep.witness == event.pose.witness
    lo, hi = event.interval
    assert 0 <= lo <= expected <= hi <= 1 and hi - lo <= 1e-6 and lo <= event.witness_time <= hi
    materials(row.sweep.witness, model, paths, event.witness_time)
    if kind == "contained-exit":
        assert event.pose.relation == "contained" and event.pose.boundary_gap_m == pytest.approx(
            0.25
        )
        assert trace.end.relation == "separated" and not event.pose.boundary
    else:
        assert event.pose.boundary_gap_m <= 1e-6
    if kind == "full-face":
        assert event.pose.boundary_fraction == pytest.approx(1 / 8)
        assert trace.start.relation == trace.end.relation == "separated"
    if kind == "independent":
        w = event.pose.witness
        wc = w.first_weights if row.first == cloth.key else w.second_weights
        want = {53: 0.5, 17: 0.25, 8: 0.25}
        np.testing.assert_allclose(
            [dict(wc).get(i, 0) for i in want], list(want.values()), rtol=0, atol=1e-6
        )
        local = matrix.T @ (np.array(w.first_point) - origin) - event.witness_time * DRIFT
        np.testing.assert_allclose(local, [-1, -1, 1], rtol=0, atol=1e-6)
        assert trace.end.relation == "crossing"
    if kind == "graze":
        assert (
            trace.start.relation == trace.end.relation == "separated"
            and event.pose.relation == "touching"
        )
    (tmp_path / "mesh-triangle-sweep.json").write_text(
        json.dumps(
            dict(
                kind=kind,
                reverse=reverse,
                oracle=str(expected),
                start=trace.start.relation,
                end=trace.end.relation,
                interval=event.interval,
                witness_time=event.witness_time,
                material=asdict(event.pose.witness),
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def test_midstep_triangle_collapse_is_unresolved_even_with_two_valid_ends():
    c = dict(CASES[0], triangle_m=[[-2, 0, 0], [2, 0, 0], [0, 1, 0]])
    model, paths, _, cloth, _, _ = build(c)
    paths[cloth.key] = ph.CollisionMotion(c["triangle_m"], [[-2, 0, 0], [2, 0, 0], [0, -1, 0]])
    row = detect(model, paths)
    assert (
        row.status == "unresolved"
        and row.sweep is None
        and row.reason == "triangle_degenerate_on_path"
    )

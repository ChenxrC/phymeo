"""Containment of a sphere inside a finite, arbitrarily oriented cone."""

import numpy as np
import pytest

import phymeo as ph


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "z, rho, feature", [(0.5, 0.0, "ring"), (0.5, 0.125, "side"), (-1.875, 0.125, "base")]
)
def test_material_witness_and_nonunique_axis(reverse, z, rho, feature):
    # This frame sends local x to y and local z to x.
    shift = np.array([8.0, -4.0, 2.0])
    ball_position = shift + [z, rho, 0.0]
    ka, kb = ph.CollisionKey(3, 1, 5), ph.CollisionKey(3, 2, 8)
    cone, ball = (kb, ka) if reverse else (ka, kb)
    asset = ph.CurvedSolid("cone", (1, 2), (0.5,) * 4)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(cone, "cone", (17,), curved=asset),
            ph.CollisionShape(ball, "sphere", (99,), 0.25),
        ]
    )
    paths = {cone: ph.CollisionMotion([shift]), ball: ph.CollisionMotion([ball_position])}
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    w = row.witness
    signed = -0.25 - min(z + 2, (2 - z - 4 * rho) / np.sqrt(17))
    assert (
        row.status == "overlap"
        and row.support_map.gap_lower_m <= signed <= row.support_map.gap_upper_m
    )
    assert w.gap_m == pytest.approx(signed, abs=1e-13)
    if feature == "ring":
        assert w.normal is None and w.first_point is None and w.second_point is None
    else:
        normal = (
            np.array([-1.0, 0, 0]) if feature == "base" else np.array([1.0, 4, 0]) / np.sqrt(17)
        )
        np.testing.assert_allclose(w.normal, -normal if reverse else normal, atol=1e-13)
        points = dict(zip((row.first, row.second), (w.first_point, w.second_point), strict=True))
        np.testing.assert_allclose(points[ball], ball_position - 0.25 * normal, atol=1e-13)
        np.testing.assert_allclose(
            points[cone], ball_position - (signed + 0.25) * normal, atol=1e-13
        )

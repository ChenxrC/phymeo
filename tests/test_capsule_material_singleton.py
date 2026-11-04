from fractions import Fraction
from math import sqrt

import numpy as np
import pytest
from test_static_capsule_face import FACE, registered

from phymeo.geometry._static_capsule_face import static_capsule_face_sample


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("flip_axis", [False, True])
def test_skew_nearest_pair_preserves_two_original_material_charts(reverse, flip_axis):
    # Minimize (-1 + 2s)^2 + (1/2 + s)^2 against the triangle's y=0 boundary.
    # Both minima are interior to the shaft/edge. Use an exact rigid frame change.
    offset = np.array([16.0, -8.0, 2.0])
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shaft = np.array([[0.5, -1.0, 0.5], [0.5, 1.0, 1.5]])
    if flip_axis:
        shaft = shaft[::-1]
    args = registered(shaft @ rotation.T + offset, FACE @ rotation.T + offset, reverse)
    w, squared_clearance = static_capsule_face_sample(*args)
    assert squared_clearance == Fraction(4, 5) - Fraction(1, 64)
    assert w.gap_m == pytest.approx(2 / sqrt(5) - 1 / 8, rel=1e-15)
    s = 0.7 if flip_axis else 0.3
    endpoint, other = w.capsule_axis_span
    assert endpoint == other and endpoint.axis_parameter == s
    world_axis = np.array([0.5, -0.4, 0.8]) @ rotation.T + offset
    world_face = np.array([0.5, 0, 0]) @ rotation.T + offset
    normal = np.array([0, 1, -2]) @ rotation.T / sqrt(5)
    np.testing.assert_allclose(endpoint.axis_point, world_axis, atol=2e-15, rtol=0)
    np.testing.assert_allclose(endpoint.face_point, world_face, atol=0, rtol=0)
    np.testing.assert_allclose(w.normal, (-normal if reverse else normal), atol=2e-16)
    np.testing.assert_allclose(
        w.second_point if reverse else w.first_point, world_axis + normal / 8, atol=4e-15, rtol=0
    )
    assert dict(endpoint.face_weights) == {23: 0.75, 59: 0.25, 17: 0}
    weights = dict(w.second_weights if reverse else w.first_weights)
    assert weights == pytest.approx({91: 1 - s, 23: s})


def test_a_full_material_interval_still_has_distinct_ends_and_center():
    shaft = [[-1, 0.5, 0.25], [3, 0.5, 0.25]]
    w, signed = static_capsule_face_sample(*registered(shaft, FACE))
    a, b = w.capsule_axis_span
    assert signed == Fraction(3, 64) and a.axis_parameter == 0.25 and b.axis_parameter == 0.625
    assert a.face_point == (0, 0.5, 0) and b.face_point == (1.5, 0.5, 0)
    assert w.second_point == (0.75, 0.5, 0)

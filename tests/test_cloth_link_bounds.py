"""Broad-phase lower bounds for finite shafts and cloth faces."""

import numpy as np
import pytest

from phymeo.solvers.cloth_link_contact import ClothLinkContacts


@pytest.mark.parametrize("offset", [np.zeros(3), np.array([2.0**20, -(2.0**20), 1.0])])
def test_diagonal_shaft_prunes_a_clear_triangle_inside_its_box(offset):
    ends = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]) + offset
    face = np.array([[-0.5, 0.5, 0.0], [-0.75, 0.75, 0.25], [-0.75, 0.75, -0.25]]) + offset
    bound = ClothLinkContacts.bounds_distance(ends, face[None, :, :])[0]
    # x=y on the shaft, and y-x >= 1 on the triangle. The first vertex and
    # the midpoint of the segment attain the perpendicular distance.
    distance = 1.0 / np.sqrt(2.0)
    assert distance - 1e-6 < bound <= distance + np.finfo(float).eps


def test_real_axis_contact_and_zero_length_shaft_stay_candidates():
    ends = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
    face = np.array([[-0.5, -0.25, 0.0], [0.5, -0.25, 0.0], [0.0, 0.5, 0.0]])
    for axis in (ends, np.zeros((2, 3))):
        assert ClothLinkContacts.bounds_distance(axis, face[None, :, :])[0] == 0.0

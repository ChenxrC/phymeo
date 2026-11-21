"""Analytic checks for finite tetrahedron/triangle withdrawal witnesses."""

import numpy as np
import pytest

from experiments.area_pressure.pusht_public_separation import _pair_witness

TET = np.vstack((np.zeros(3), np.eye(3)))


def test_oblique_clearance_with_overlapping_axis_aligned_boxes():
    tri = np.array([[0.4, 0.4, 0.4], [0.6, 0.4, 0.2], [0.2, 0.6, 0.4]])
    assert np.all(tri.min(axis=0) < TET.max(axis=0))
    assert np.all(TET.min(axis=0) < tri.max(axis=0))
    witness = _pair_witness(TET, tri, TET, TET, tri, tri)
    assert witness is not None
    # The orthogonal projection of the first vertex lies inside the face
    # x+y+z=1, so 0.2/sqrt(3) is an independent upper bound on any gap.
    assert 0.11 < witness["gap_lower_m"] <= 0.2 / np.sqrt(3) + 1e-15
    # These enclosures also contain an overlapping translate of that triangle.
    uncertain = _pair_witness(TET, tri, TET, TET, tri - 0.1, tri + 0.1)
    assert uncertain is None or uncertain["gap_lower_m"] <= 0


@pytest.mark.parametrize(
    "tri",
    [
        np.eye(3),
        np.array([[1.0, 0, 0], [1.125, 0, 0], [1.0, 0.125, 0]]),
        np.array([[0.125, 0.125, 0.125], [0.25, 0.125, 0.125], [0.125, 0.25, 0.125]]),
    ],
    ids=["shared-face", "single-vertex-touch", "inside"],
)
def test_touch_and_overlap_do_not_certify_strict_clearance(tri):
    witness = _pair_witness(TET, tri, TET, TET, tri, tri)
    assert witness is None or witness["gap_lower_m"] <= 0

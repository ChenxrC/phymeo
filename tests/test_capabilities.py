"""Physical signatures must own immutable data supplied by plugin authors."""

import numpy as np
import pytest

from phymeo import ProblemSignature


def test_signature_canonicalizes_and_owns_nested_sequences():
    edge = [np.int64(0), np.int64(1), np.float32(1.0), np.float64(0.0)]
    distance = [edge]
    plane = [0.0, "y-up-frictionless-e0-v1"]
    signature = ProblemSignature(distance=distance, plane=plane)
    edge[2] = 99.0
    distance.append([1, 2, 2.0, 0.0])
    plane[0] = 88.0
    assert signature.distance == ((0, 1, 1.0, 0.0),)
    assert signature.plane == (0.0, "y-up-frictionless-e0-v1")
    assert hash(signature) == hash(
        ProblemSignature(distance=((0, 1, 1.0, 0.0),), plane=(0.0, "y-up-frictionless-e0-v1"))
    )


@pytest.mark.parametrize(
    "distance",
    [
        [(0, 1, 1.0)],
        [(0, 1, 1.0, 0.0, 2.0)],
        [(0.0, 1, 1.0, 0.0)],
        [(True, 1, 1.0, 0.0)],
        [(-1, 1, 1.0, 0.0)],
        [(0, 0, 1.0, 0.0)],
        [(0, 1, np.nan, 0.0)],
        [(0, 1, 1.0, np.inf)],
        [(0, 1, -1.0, 0.0)],
        [(0, 1, 1.0, -1.0)],
        [(0, 1, True, 0.0)],
    ],
)
def test_invalid_distance_identity_is_rejected(distance):
    with pytest.raises((TypeError, ValueError)):
        ProblemSignature(distance=distance)


@pytest.mark.parametrize(
    "plane", [[0.0], [0.0, "law", "extra"], [np.inf, "law"], [True, "law"], [0.0, ""], [0.0, 3]]
)
def test_invalid_plane_identity_is_rejected(plane):
    with pytest.raises((TypeError, ValueError)):
        ProblemSignature(plane=plane)


@pytest.mark.parametrize("semantics", [None, 3, "", "  "])
def test_invalid_semantic_version_is_rejected(semantics):
    with pytest.raises((TypeError, ValueError)):
        ProblemSignature(semantics=semantics)

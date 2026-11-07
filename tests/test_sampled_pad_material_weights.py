"""Regression for declared sampled-pad faces polluted by world-coordinate inversion."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.panda import load_panda
from phymeo.runtime.robot_payload_scene import PadFace, _cuboid_face_weights, _pad_points


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 1])
def test_selected_face_keeps_exact_support_and_reconstructs_original_material(axis, side):
    ids = (101, 305, 701, 901, 1001, 1305, 1701, 1901)
    corners = np.array(list(product((0, 1), repeat=3)))
    uv = np.empty(3)
    uv[[i for i in range(3) if i != axis]] = (0.25, 0.75)
    # A positive inverse-fit residue must not invent the opposite surface.
    uv[axis] = 2.0**-55 if side < 0 else 1.0 - 2.0**-52
    weights = _cuboid_face_weights(ids, axis, side, uv)
    selected = {ids[i] for i, bits in enumerate(corners) if bits[axis] == int(side > 0)}
    assert {node for node, _ in weights} == selected
    assert sum(weight for _, weight in weights) == 1.0
    assert all(weight > 0 for _, weight in weights)
    # Independent affine reconstruction, including a nonzero installation.
    world = corners @ np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 4.0], [8.0, 0.0, 0.0]]) + (
        32.0,
        -16.0,
        8.0,
    )
    expected_local = uv.copy()
    expected_local[axis] = int(side > 0)
    expected = expected_local @ np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 4.0], [8.0, 0.0, 0.0]]) + (
        32.0,
        -16.0,
        8.0,
    )
    actual = sum(weight * world[ids.index(node)] for node, weight in weights)
    np.testing.assert_array_equal(actual, expected)
    # This is a selected-feature adapter, never a repair for outside material.
    bad = uv.copy()
    bad[(axis + 1) % 3] = -2e-8
    with pytest.raises(ValueError, match="outside"):
        _cuboid_face_weights(ids, axis, side, bad)


@pytest.mark.parametrize("case", ["mimic", "panda"])
def test_authored_pad_metadata_keeps_original_default_geometry_and_folded_frame(case):
    if case == "mimic":
        asset = load_mimic_urdf(
            Path(__file__).parent / "reference_newton/robot/mimic_jaws.urdf"
        ).tree
        names, axis, signs = (("left", "left_rubber"), ("right", "right_rubber")), 0, (1, -1)
    else:
        asset = load_panda()
        names = (
            ("panda_leftfinger", "panda_leftfinger_pad"),
            ("panda_rightfinger", "panda_rightfinger_pad"),
        )
        axis, signs = 1, (-1, 1)
    pads = tuple(
        PadFace.named(*name, axis, side, tuple(product((-0.005, 0.005), repeat=2)))
        for name, side in zip(names, signs, strict=True)
    )
    before = _pad_points(asset, pads)
    indices, points, keys, excluded, metadata = _pad_points(asset, pads, with_material_weights=True)
    np.testing.assert_array_equal(indices, before[0])
    np.testing.assert_array_equal(points, before[1])
    assert keys == before[2] and excluded == before[3] and len(metadata) == 8
    by_key = {col.shape.key: col for col in asset.collisions}
    bits = tuple(product((0, 1), repeat=3))
    for i, weights in enumerate(metadata):
        col = by_key[keys[i]]
        side = signs[i // 4]
        expected_ids = {
            col.shape.vertex_ids[j] for j, row in enumerate(bits) if row[axis] == int(side > 0)
        }
        assert {node for node, _ in weights} == expected_ids
        authored = sum(
            weight * col.local_vertices[col.shape.vertex_ids.index(node)]
            for node, weight in weights
        )
        model_local = asset.authored_to_model_points(col.link_name, authored)
        np.testing.assert_allclose(model_local, points[i], atol=2e-16, rtol=0)

"""Independent signed overlap and finite paired material witnesses."""

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.solvers.box_link_contact import BoxLinkContacts, box_capsule_witnesses
from phymeo.solvers.sphere_link_contact import LinkSolid


@pytest.mark.parametrize(
    "path,parameters,skins",
    [
        ([[0, 0, 0.75]], [0], [[0, 0, 1]]),
        ([[0, 0, 0.75], [0, 0, 1.5]], [0], [[0, 0, 1]]),
        ([[-0.5, 0, 0.75], [0.5, 0, 0.75]], [0, 1], [[-0.5, 0, 1], [0.5, 0, 1]]),
        ([[-2, 0, 0.75], [2, 0, 0.75]], [0.25, 0.75], [[-1, 0, 1], [1, 0, 1]]),
    ],
)
def test_unique_face_overlap_clips_original_axis_to_finite_box(path, parameters, skins):
    rows = list(
        box_capsule_witnesses(np.zeros(3), np.eye(3), np.ones(3), path, 0.25, capsule_first=True)
    )
    assert len(rows) == len(parameters)
    np.testing.assert_allclose([r["axis_parameter"] for r in rows], parameters)
    np.testing.assert_allclose([r["second_surface"] for r in rows], skins)
    assert sum(r["weight"] for r in rows) == 1
    for row in rows:
        assert row["gap_m"] == -0.5
        np.testing.assert_array_equal(row["normal"], [0, 0, -1])
        np.testing.assert_allclose(
            row["second_surface"] - row["first_surface"], row["gap_m"] * row["normal"]
        )


def test_edge_facet_escape_swaps_and_transforms_without_changing_materials():
    path = np.array([[-2, -0.5, 0], [0.5, 2, 0]])
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    center = np.array([3.0, -2, 4])
    (base,) = box_capsule_witnesses(
        np.zeros(3), np.eye(3), np.ones(3), path, 0.25, capsule_first=True
    )
    assert base["axis_parameter"] == 0.5
    assert base["gap_m"] == pytest.approx(-(0.25 + 1 / (2 * np.sqrt(2))))
    np.testing.assert_allclose(base["normal"], [1 / np.sqrt(2), -1 / np.sqrt(2), 0])
    np.testing.assert_allclose(base["second_surface"], [-1, 1, 0])
    for reverse in (False, True):
        moving = path[::-1] if reverse else path
        for capsule_first in (False, True):
            (row,) = box_capsule_witnesses(
                center,
                rotation,
                np.ones(3),
                moving @ rotation.T + center,
                0.25,
                capsule_first=capsule_first,
            )
            assert row["gap_m"] == base["gap_m"]
            np.testing.assert_allclose(
                row["normal"], rotation @ base["normal"] * (1 if capsule_first else -1)
            )
            first, second = (
                ("first_surface", "second_surface")
                if capsule_first
                else ("second_surface", "first_surface")
            )
            np.testing.assert_allclose(row[first], center + rotation @ base["first_surface"])
            np.testing.assert_allclose(row[second], center + rotation @ base["second_surface"])


@pytest.mark.parametrize("path", [[[0, 0, 0]], [[0, 0, -2], [0, 0, 2]]])
def test_ambiguous_interior_does_not_invent_an_escape_axis(path):
    with pytest.raises(ValueError, match="ambiguous"):
        list(
            box_capsule_witnesses(
                np.zeros(3), np.eye(3), np.ones(3), path, 0.25, capsule_first=True
            )
        )


def test_overlap_loads_use_clipped_sites_and_reciprocal_midpoint_moments():
    solid = LinkSolid("round", "link", 0, "capsule", [[-2, 0, 0.75], [2, 0, 0.75]], 0.25)
    out = BoxLinkContacts([solid]).evaluate(
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.zeros((1, 3)),
        np.zeros((1, 6)),
        np.zeros(3),
        np.ones(3),
        np.zeros(3),
        np.zeros(3),
        RobotContactMaterial(8.0, 0.0, 0.0, 0.1),
        payload_rotation=np.eye(3),
    )
    assert len(out.contacts) == 2
    np.testing.assert_allclose(out.payload_wrench, [0, 0, -4, 0, 0, 0])
    np.testing.assert_allclose(out.link_wrenches[0], -out.payload_wrench)
    assert [r["normal_load_N"] for r in out.contacts] == [2.0, 2.0]


def test_saved_rejected_g1_geometry_has_unique_top_facet_without_pose_special_case():
    path = [
        [1.0465905923930097, -2.3906161910884687, -8.821308578781106e-5],
        [1.2056357335466252, -2.4407932259757725, 0.008624320721593647],
    ]
    (row,) = box_capsule_witnesses(
        [0, 0, -0.1], np.eye(3), [10, 10, 0.1], path, 0.01, capsule_first=True
    )
    assert row["axis_parameter"] == 0
    np.testing.assert_allclose(row["normal"], [0, 0, -1])
    assert row["gap_m"] == pytest.approx(path[0][2] - 0.01)
    np.testing.assert_allclose(row["second_surface"], [path[0][0], path[0][1], 0], atol=1e-15)

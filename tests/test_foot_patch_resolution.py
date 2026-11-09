"""Unresolved OBB rotations must not create a finite corner torque."""

from itertools import product

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.solvers.fixed_table_contact import FixedTableContacts
from phymeo.solvers.sphere_link_contact import LinkSolid


@pytest.mark.parametrize("sine", [-(2.0**-70), 2.0**-70])
def test_subresolution_rotation_preserves_actual_face_load(sine):
    half = np.array([0.12, 0.06, 0.03125])
    vertices = np.array(list(product((-1.0, 1.0), repeat=3))) * half
    foot = LinkSolid("sole", "foot", 0, "box", vertices, 0.0)
    channel = FixedTableContacts(
        [foot],
        center_m=[0, 0, -0.05],
        half_extents_m=[2, 2, 0.05],
        material=RobotContactMaterial(20000, 120, 0),
    )
    rotation = np.array([[1.0, 0, 0], [0, 1.0, -sine], [0, sine, 1.0]])
    position = np.array([[0.0, 0, half[2] - 0.000981]])
    answer = channel.evaluate(position, rotation[None], position, np.zeros((1, 6)))
    assert len(answer.contacts) > 2
    assert all(row["patch_dimension"] == 2 for row in answer.contacts)
    np.testing.assert_allclose(answer.link_wrenches[0], [0, 0, 19.62, 0, 0, 0], atol=1e-10)
    for row in answer.contacts:
        assert row["gap_m"] == pytest.approx(-0.000981, abs=1e-14)
        pa = np.asarray(row["link_surface_m"])
        expected = np.array([weight for _, weight in row["link_material_weights"]])
        assert np.isfinite(pa).all() and np.all(expected >= 0)

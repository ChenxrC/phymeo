"""Tests-first: explicit force-application coordinates do not redefine material slip.

UNEXECUTED authoring; the parent owns all project execution. Frozen public gates
are 1e-6 m / 1e-8 J. No complete grasp window is repeated here.
"""

import numpy as np
import pytest

from phymeo import ContactSceneBuilder, NormalLayer
from phymeo.constraints.collision_shapes import CollisionKey
from phymeo.runtime.material_stations import (
    MaterialStationRegistration,
    MaterialStationRegistry,
)
from phymeo.solvers.tangent_history import StationKey


MOTION_M = 1e-6
ENERGY_J = 1e-8


def _clamp():
    builder = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_box(10, (0.3, 0.3, 0.1), mass=None, position=(0, 0, -0.2))
    builder.add_box(20, (0.1, 0.1, 0.05), mass=1.0)
    builder.add_box(30, (0.3, 0.3, 0.1), mass=None, position=(0, 0, 0.2))
    layer = NormalLayer(0.06, 200.0, 0.2, 0.5)
    builder.pair(10, 20, layer)
    builder.pair(20, 30, layer)
    builder.pair(10, 30, None)
    return builder


def _clamp_registry():
    # The two real finite square patches each have twelve area quadrature sites.
    # Full face IDs are declared: closest-point sparse weights are not identity.
    corners = ((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1))
    anchors = []
    for index, left in enumerate(corners):
        right = corners[(index + 1) % 4]
        for wl, wr in ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)):
            anchors.append((wl * left[0] + wr * right[0], wl * left[1] + wr * right[1]))
    declarations = []
    for side, (first, second, z) in enumerate(((10, 20, -0.05), (20, 30, 0.05))):
        for index, (x, y) in enumerate(anchors):
            declarations.append(
                MaterialStationRegistration(
                    StationKey(0, side * 12 + index),
                    CollisionKey(0, first, 0),
                    CollisionKey(0, second, 0),
                    "second" if side == 0 else "first",
                    (x, y, z),
                    (1, 0, 0),
                    side * 2,
                    side * 2 + 1,
                    first_feature=(1, 3, 5, 7),
                    second_feature=(0, 2, 4, 6),
                )
            )
    return MaterialStationRegistry(declarations)


def test_a01_public_contact_scene_accepts_application_reference_after_real_normal_sample():
    # RED must reach the new keyword after the unchanged real normal law works.
    with _clamp().build() as scene:
        initial = scene.state
        events = scene.contact_events
        loads = scene.sample().loads
        assert len(loads.contacts) == 2
        assert [load.force_n for load in loads.contacts] == pytest.approx([2.0, 2.0], abs=1e-10)
        assert sum(len(load.sites) for load in loads.contacts) == 24
        row = scene.rigid_rows[0, 20]
        np.testing.assert_allclose(loads.inputs.rigid.force[row], 0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(loads.inputs.rigid.torque[row], 0, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(scene.state.rigid.position, initial.rigid.position)
        assert scene.contact_events == events
        assert scene.tangent_history is None
        assert scene.last_frame is None

        scene.enable_coupled_tangent(
            _clamp_registry(),
            stiffness_n_per_m=200.0,
            motion_reference="application",
        )

        external = scene.inputs()
        external.rigid.force[row, 0] = 0.5
        frame = scene.step(1 / 1024, external)
        receipt = scene.tangent_receipt
        assert receipt.accepted
        assert receipt.motion_reference == "application"
        assert receipt.application_motion_error_m <= MOTION_M
        assert abs(receipt.energy_residual_j) <= ENERGY_J
        assert len(scene.tangent_history.active) == 24
        assert {record.update.motion_reference for record in scene.tangent_history.active} == {
            "application"
        }
        np.testing.assert_allclose(
            frame.state.rigid.velocity[row] - initial.rigid.velocity[row],
            (frame.loads.inputs.rigid.force[row] + external.rigid.force[row]) / 1024,
            atol=2e-13,
            rtol=0,
        )


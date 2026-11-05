"""Actual sampled scene forces and physical motion publish tangent state together."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo import ContactSceneBuilder, NormalLayer
from phymeo.constraints.collision_shapes import CollisionKey


def clamp(mass=1.0):
    b = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_box(10, (.3, .3, .1), mass=None, position=(0, 0, -.2))
    b.add_box(20, (.1, .1, .05), mass=mass)
    b.add_box(30, (.3, .3, .1), mass=None, position=(0, 0, .2))
    layer = NormalLayer(.06, 200.0, .2, .5)
    b.pair(10, 20, layer)
    b.pair(20, 30, layer)
    b.pair(10, 30, None)
    return b


def clamp_registry():
    from phymeo.runtime.material_stations import (
        MaterialStationRegistration as Registration,
        MaterialStationRegistry,
    )
    from phymeo.solvers.tangent_history import StationKey

    registrations = []
    for side, (a, b, z) in enumerate(((10, 20, -.05), (20, 30, .05))):
        for index, (x, y) in enumerate(product((-.1, .1), repeat=2)):
            registrations.append(Registration(
                StationKey(0, side * 4 + index),
                CollisionKey(0, a, 0), CollisionKey(0, b, 0),
                "second" if side == 0 else "first",
                (x, y, z), (1, 0, 0), side * 2, side * 2 + 1,
            ))
    return MaterialStationRegistry(registrations)


@pytest.mark.parametrize("mass,force", [(1.0, .5), (2.0, 3.0)])
def test_real_finite_bilateral_load_drives_common_state_and_force(mass, force):
    with clamp(mass).build() as scene:
        scene.enable_coupled_tangent(clamp_registry(), stiffness_n_per_m=200.0)
        row = scene.rigid_rows[0, 20]
        f = scene.inputs()
        f.rigid.force[row, 0] = force
        for step in range(48):
            before = scene.state
            frame = scene.step(1 / 1024, f)
            receipt = scene.tangent_receipt
            assert receipt.accepted
            assert scene.tangent_history.step_index == step + 1
            assert scene.contact_events.commit_time_s == scene.tangent_history.commit_time_s
            assert len(scene.tangent_history.active) == 8
            actual = frame.loads.inputs.rigid.force[row] + f.rigid.force[row]
            np.testing.assert_allclose(
                mass * (frame.state.rigid.velocity[row] - before.rigid.velocity[row]),
                actual / 1024, atol=2e-13, rtol=0,
            )
            assert sum(c.force_n for c in frame.loads.contacts) == pytest.approx(4.0, abs=1e-10)
            assert abs(receipt.energy_residual_j) < 1e-8
        assert frame.state.rigid.position[row, 0] > 0
        assert abs(frame.state.rigid.angular_velocity[row]).max() < 1e-10
        assert abs(frame.loads.inputs.rigid.force[row, 0]) > .01


def test_tangent_failure_leaves_physical_history_geometry_and_events_unchanged(monkeypatch):
    with clamp().build() as scene:
        scene.enable_coupled_tangent(clamp_registry(), stiffness_n_per_m=200.0)
        state, history, events = scene.state, scene.tangent_history, scene.contact_events
        from phymeo.runtime import _scene_coupled_tangent as bridge

        def reject(*args, **kwargs):
            raise ValueError("injected actual motion rejection")
        monkeypatch.setattr(bridge, "finish_tangent_step", reject)
        with pytest.raises(ValueError, match="injected actual motion"):
            scene.step(1 / 1024)
        np.testing.assert_array_equal(scene.state.rigid.position, state.rigid.position)
        assert scene.tangent_history is history
        assert scene.contact_events == events
        assert scene.last_frame is None


def test_default_remains_unconfigured_and_native_opt_in_rejects_before_execution():
    with clamp().build() as scene:
        assert scene.tangent_history is None
        frame = scene.step(1 / 1024)
        assert frame.rigid_report.accepted

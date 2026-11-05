"""A retained loader result and a public asset must not rewrite a live scene's physics."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

ASSETS = Path(__file__).parent / "reference_newton" / "robot"


@pytest.mark.parametrize("kind,source", tuple(product(("plain", "mimic"), ("loader", "public"))))
def test_retained_or_public_asset_cannot_change_live_or_subsequent_build_mass(kind, source):
    path = ASSETS / ("multiobject_pusher.urdf" if kind == "plain" else "mimic_jaws.urdf")
    original = (load_urdf if kind == "plain" else load_mimic_urdf)(path, gravity=(0, 0, 0))
    n = len(original.joint_names)
    b = ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_urdf(
        10,
        path,
        loader=lambda path, *, gravity: original,
        robot_state=RobotState(np.zeros(n), np.zeros(n)),
        kp=np.zeros(n),
        kd=np.zeros(n),
    )
    b.add_sphere(20, 0.02, mass=0.1, position=(4, 4, 4))
    b.pair(10, 20, RobotContactMaterial(2000, 0, 0))
    with b.build() as first:
        exposed = original if source == "loader" else first.assets[0, 10]
        tree = exposed if kind == "plain" else exposed.tree
        try:
            tree.model.mass.setflags(write=True)
        except ValueError:
            # Bytes-backed immutable public snapshots are a valid contract.
            pass
        else:
            tree.model.mass[:] *= 1000
        # Second build is deliberately after the attempted mutation.
        with b.build() as second:
            expected = [0.001] if kind == "plain" else [0, 0.002]
            drive = [2] if kind == "plain" else [0, 1]
            for scene in (first, second):
                reading = scene.step(
                    0.001, actions={10: RobotCommand(np.zeros(n), feedforward=drive)}
                )
                np.testing.assert_allclose(reading.robots[0, 10].v, expected, atol=1e-13, rtol=0)
                assert reading.loads.contacts == ()

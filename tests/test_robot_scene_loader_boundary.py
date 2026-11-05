"""Opaque loader delegates cannot bypass the scene's owned asset snapshot contract."""

from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.urdf import load_urdf
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder


class _OpaqueAsset:
    def __init__(self, asset):
        self.asset = asset

    def __getattr__(self, name):
        return getattr(self.asset, name)


def test_opaque_loader_result_is_rejected_before_robot_registration():
    source = Path(__file__).parent / "reference_newton" / "robot" / "multiobject_pusher.urdf"
    asset = load_urdf(source, gravity=(0, 0, 0))
    opaque = _OpaqueAsset(asset)
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    with pytest.raises(TypeError, match="RobotAsset|MimicRobotAsset"):
        builder.add_urdf(
            10,
            source,
            loader=lambda path, *, gravity: opaque,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
        )

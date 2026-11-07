"""Declared physical inputs for the general scene friction acceptance cases.

UNEXECUTED tests-first. This module builds real scenes and explicit material
registrations. It does not implement an integrator, alter an accepted state, or
infer a persistent station token from the order of sampled contact rows.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.material_stations import (
    MaterialStationRegistration,
    MaterialStationRegistry,
)
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.solvers.tangent_history import StationKey

WORLD = 23
LOWER_TOOL, UPPER_TOOL, BOX = 17, 42, 91
H = 1 / 1024
DELTA = 1 / 1024
RADIUS = 1 / 16
KT = 200.0
BOX_HALF = (0.1, 0.1, 0.05)


@dataclass(frozen=True)
class FrictionCase:
    scene: object
    registry: MaterialStationRegistry
    actions: object
    world: int
    parameters: object

    def close(self):
        self.scene.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _paddle_asset(directory, body, sign):
    path = Path(directory) / f"paddle-{body}.urdf"
    z = sign * (BOX_HALF[2] + RADIUS - DELTA)
    path.write_text(
        f'''<robot name="paddle_{body}"><link name="base"/>
<link name="tool"><inertial><mass value="2"/>
<inertia ixx=".01" iyy=".015" izz=".02" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="ball"><geometry><sphere radius="{RADIUS:.17g}"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<origin xyz="0 0 {z:.17g}"/><axis xyz="1 0 0"/>
<limit lower="-.1" upper=".1" effort="8" velocity="8"/></joint></robot>''',
        encoding="utf-8",
    )
    return path


def make_two_paddles(directory, *, world=WORLD, friction=0.5, reverse=False):
    """F01: two positive-mass P tools, two true normals, and one free box.

    The box and sphere material charts are declared from their authored vertex
    order, before querying contacts. Collision keys come from public semantic
    lookup after compilation; numeric physical body IDs are not shape indices.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    builder = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    declarations = ((LOWER_TOOL, -1), (UPPER_TOOL, 1))
    for body, sign in reversed(declarations) if reverse else declarations:
        builder.add_urdf(
            body,
            _paddle_asset(directory, body, sign),
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0.0],
            kd=[0.0],
            effort_limit=[8.0],
            world=world,
        )
    builder.add_box(BOX, BOX_HALF, mass=1.0, world=world)
    law = RobotContactMaterial(2048.0, 0.0, friction, 0.01)
    builder.pair(LOWER_TOOL, BOX, law, world=world)
    builder.pair(UPPER_TOOL, BOX, law, world=world)
    builder.pair(LOWER_TOOL, UPPER_TOOL, None, world=world)
    scene = builder.build(backend="host")
    try:
        (box_key,) = scene.body_keys[world, BOX]
        registrations = []
        for body, sign, token in ((LOWER_TOOL, -1, 101), (UPPER_TOOL, 1, 202)):
            tool_key = scene.shape_key(body, "tool", "ball")
            first, second = sorted((tool_key, box_key))
            box_feature = (0, 2, 4, 6) if sign < 0 else (1, 3, 5, 7)
            tool_first = tool_key == first
            registrations.append(
                MaterialStationRegistration(
                    StationKey(world, token),
                    first,
                    second,
                    "first" if tool_first else "second",
                    (0.0, 0.0, -sign * RADIUS),
                    (1.0, 0.0, 0.0),
                    1000 + token if tool_first else 2000 + token,
                    2000 + token if tool_first else 1000 + token,
                    first_feature=(0,) if tool_first else box_feature,
                    second_feature=box_feature if tool_first else (0,),
                )
            )
        registry = MaterialStationRegistry(registrations)
        actions = MappingProxyType(
            {body: RobotCommand([0.0], feedforward=[0.0]) for body, _ in declarations}
        )
        parameters = MappingProxyType(
            dict(case="F01", dt_s=H, stiffness_n_per_m=KT, friction=friction, penetration_m=DELTA)
        )
        return FrictionCase(scene, registry, actions, world, parameters)
    except BaseException:
        scene.close()
        raise